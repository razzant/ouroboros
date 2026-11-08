# Module widget host protocol

This document is for anyone writing an interface that runs reviewed `kind: "module"` widgets the way the builtin interface does. It is the host side: how to build the frame, the message grammar between the page and the frame, and when to start and stop a widget. The author side, the `window.OuroborosWidget` API a widget is written against, is in [`CREATING_SKILLS.md`](CREATING_SKILLS.md) ("`kind: "module"` widgets"). The rest of the client contract is in [`CLIENT_CONTRACT.md`](CLIENT_CONTRACT.md).

A widget written for Ouroboros only works in your interface if your host keeps every promise below. Nothing here requires importing the builtin interface. The builtin implementation is `web/modules/widget_module.js` (the page side) and `web/modules/widget_frame.js` (the bootstrap scripts injected into the frame); reading them is optional.

## 1. What the core gives you

- **Cards:** `GET /api/widgets` returns `ui_tabs`, one `WidgetTab` per card: `key`, `skill`, `tab_id`, `title`, `icon`, `ws_prefix`, `render`, `span`, `grid_span`, `revision` and `generation` (live skill tabs).
- **A module card's `render`:** `{kind: "module", entry, entry_type?, assets?, start, appearance?, height?, max_height?}`.
  - `entry` is the reviewed script: a bare filename for classic entries, a skill-relative path for ESM entries. ESM paths retain their `/` separators; segment characters are ASCII letters, digits, `.`, `_` and `-` (CREATING_SKILLS defines the bounds).
  - `start` is `auto`, `manual` or `retain` (§7).
  - `appearance` (`host`, `independent` or `fixed`) records the author's intent only. It changes nothing in the host.
  - `height` is a fixed frame height (320 to 8192 pixels). Without it the frame sizes itself (§5).
  - `max_height` lowers the self-sizing ceiling (320 to 8192, default 8192).
- **Classic sources:** `GET /api/extensions/{skill}/module/{entry}` answers captured JavaScript text, without caching and with any-origin CORS; network authentication still applies. `409` means no live bundle, `404` no captured file. ESM uses the lease below.
- **Change signals:**
  - `revision` is the skill's live content hash. A card whose `revision` changed while it runs is stopped (§6) and started again. `generation` names the live publication: an `entry_type: "module"` card's file lease is bound to it, so in board and chat mounts such a card also restarts when it changes, even with unchanged bytes. Classic cards ignore it.
  - An `extension_lifecycle` WebSocket frame means "read `GET /api/widgets` again". A card that left the list is stopped.
- **The owner's launch choice:** `GET /api/ui/preferences` returns `widget_start_mode`, keyed `"<skill>:<tab_id>"`. It overrides the card's own `start`. `POST /api/ui/preferences` with `{widget_start_mode}` saves it.

For a classic entry, fetch the script with the owner's session, from the page. The builtin host gives up after 25 seconds and shows a retryable error.

## 2. Building the frame

Create one `<iframe>` per running card with exactly these attributes:

| Attribute | Value |
|---|---|
| `sandbox` | `allow-scripts allow-pointer-lock allow-downloads` |
| `allow` | `autoplay; fullscreen; clipboard-write` |
| `allowfullscreen` | present |

Never add `allow-same-origin`: the frame stays an opaque origin with no direct access to the page's cookies, storage or DOM. Cookie attachment to asset requests is browser-dependent; use the host fetch bridge, not a promise that passive requests are always anonymous. Never add top navigation, popups, forms or modals.

Set the frame's `srcdoc` property (not an attribute string) to this document, in this order. `BASE` is empty for classic entries; `ENTRY` is the script element described below:

```html
<!doctype html><html><head><meta http-equiv="Content-Security-Policy" content="POLICY">BASE</head><body><div id="root"></div><script>BRIDGE</script><script>RESIZE</script>ENTRY</body></html>
```

- **`POLICY`** is the document policy below, with `ORIGIN` the page's own origin (for example `https://ouro.example`) and `SKILL` the URL-encoded skill name. An opaque frame's `'self'` matches nothing, so the sources are absolute.

  ```
  default-src 'none'; script-src 'unsafe-inline' 'wasm-unsafe-eval' blob: ORIGIN/api/extensions/SKILL/module/; worker-src blob:; style-src 'unsafe-inline'; img-src data: blob: RESOURCE_BASE; media-src data: blob: RESOURCE_BASE; font-src data: blob: RESOURCE_BASE
  ```

  `RESOURCE_BASE` is this mount's route lease (below, "Route leases"), resolved against the page's origin.

- **`BRIDGE`** is your bootstrap (§3). It must run before the module.
- **`RESIZE`** is your size reporter (§5), or empty when the card has a fixed `height`.
- **`ENTRY`** for classic mode is `<script>MODULE</script>`, with the script text with every `</script` written as `<\/script` and every `<!--` as `<\!--`. It runs as a classic script, even when the file is named `.mjs`.

### Route leases (classic entries and route frames)

Before building a classic frame (and a `kind: "iframe"` frame, §8), the authenticated parent posts
`{skill, tab_id, generation, scope: "routes"}` to `/api/widgets/asset-leases`. Success has the file lease's
shape with `base_url` `/api/widget-resources/TOKEN/`; an ESM tab, another render kind or a stale publication
answers 409. Renew and revoke it exactly like a file lease (below). Give the bootstrap `RESOURCE_BASE` and
expose it as `OuroborosWidget.resourceBase` with `resourceUrl(url)`: a URL under the skill's route prefix or a
relative path maps onto `RESOURCE_BASE`, anything else is returned unchanged.

`GET`/`HEAD` under `RESOURCE_BASE` runs that skill's own route of the same rest path through the ordinary
dispatcher, `Range` and streaming bodies included. No `Cookie` or `Authorization` reaches it, and the core's
own names under the prefix (`manifest`, `module/`, `ui/`, `settings_section`) answer 400. A redirect inside the
skill's prefix is rewritten under `RESOURCE_BASE`; one to anything else on this origin answers 403; another
origin's is passed on. Every response is `no-store` with `Referrer-Policy: no-referrer`; other methods 405,
an unknown or ended lease 404. Revoke, logout, password change and skill unload end its open responses at
once, as a client disconnect would; expiry is noticed when a response next sends or at another lease
operation, so a handler that sends nothing is not cut at the deadline. A file lease never reads routes and a route lease never
reads files.

A route lease is kept while its frame runs, so the frame's document and state survive a failed renewal and a
core restart. Renew it like a file lease, but treat a failed, refused (5xx) or hanging renewal as transient:
retry after 2, 4, 8, 16, 32, then every 40 seconds, and never end the mount on a timer. When renewal answers
409 (the core no longer knows the token: a restart emptied the table, or the lease expired while this page
could not renew), post the mint body again with `token: TOKEN`: the core re-establishes the lease under the
same URL for the tab's current publication (the `generation` may be stale), renews it if this owner still
holds it, answers 409 while the skill is not live yet (keep retrying), and 410 for a token that was revoked
(Stop, even after the core had already dropped it), or ended by logout or a password change, in this core
process; the core remembers the last 4096 such tokens, and a restart forgets them. End the mount only on 400, 401, 403 or 410,
or when the card itself leaves the list.

### Opt-in native ESM files

`entry_type: "module"` is explicit. Omission or `"classic"` preserves classic
semantics regardless of suffix. `entry` plus `assets` form the finite browser
file set defined in CREATING_SKILLS. No recursive discovery or HTML runtime is
part of this mode.

The authenticated parent posts `{skill, tab_id, generation}` to
`/api/widgets/asset-leases`. Success is `{token, base_url, expires_in: 120,
renew_after: 40}`; `base_url` is `/api/widget-assets/TOKEN/`. Resolve it against
the parent's origin as `ASSET_BASE`. Refused/stale publications answer 409;
invalid fields/body answer 400/413; foreign or opaque origins answer 403.

Set `BASE` to `<meta name="referrer" content="no-referrer"><base href="ASSET_BASE">`
and the iframe's `referrerPolicy` to `no-referrer`. Set `ENTRY` to
`<script type="module" crossorigin="anonymous" src="ENTRY_URL"></script>`,
where `ENTRY_URL` resolves `render.entry` from `ASSET_BASE`; escape attributes.
Use this policy (the bootstrap still runs first):

```
default-src 'none'; script-src 'unsafe-inline' 'wasm-unsafe-eval' blob: ASSET_BASE; worker-src blob:; style-src 'unsafe-inline' ASSET_BASE; img-src data: blob: ASSET_BASE; media-src data: blob: ASSET_BASE; font-src data: blob: ASSET_BASE; base-uri ASSET_BASE
```

Only GET/HEAD under that URL reads declared captured bytes. Responses and
refusals carry `no-store`, `nosniff`, `Referrer-Policy: no-referrer`, explicit MIME
and `Access-Control-Allow-Origin: null` without credentials; SVG/document loads
are sandboxed. Invalid paths answer 400, unavailable/unlisted files 404 and
non-read methods 405. No cookie bypasses an invalid lease.

Renew the same URL every 40 seconds with `{token}` at
`POST /api/widgets/asset-leases/renew`; the response has the same shape. Revoke
with `{token}` at `POST /api/widgets/asset-leases/revoke` (idempotent `{ok:true}`).
These controls require the issuing parent authentication context and reject
opaque/cross-site browser requests. They must never enter the child bridge's
allowlist. A lease cannot renew itself. Stop/disposal sends revoke after the
bounded save handshake; pagehide sends a keepalive revoke. For a file lease, a failed renewal ends
the mount; an expired file lease cannot be revived and reopening requires a new mount (route leases above).

Reads and renewal check the same publication plus current enable/review/grant
readiness. Unload/reload, disable or authority withdrawal invalidates it; logout
revokes leases of that authentication context, password changes revoke all,
and restart drops the in-memory table. Logout/password changes also cancel
already admitted mints waiting for a worker; this does not change cookie session
validation. A crash/network loss falls back to expiry,
at most 120 seconds from the last accepted renewal. Already delivered or
authorized in-flight bytes cannot be recalled. The widget can share its URL;
renewal extends a copied URL too. HTTP exposes it in transit; HTTPS protects
transport. It grants no main session or APIs. Redact the token in access/fault
logs, including at a separate reverse proxy.

## 3. The bootstrap inside the frame

Before the module runs, the bootstrap must provide what `CREATING_SKILLS.md` promises the author:

- `window.OuroborosWidget = {fetch, onEvent, onTheme, download, openExternal, resourceBase, resourceUrl, ready}` (`resourceBase` is `''` for ESM; `ready` resolves once the page accepted the channel);
- `window.fetch` replaced by the same bridged `fetch`;
- `window.__ouroWidgetOnDispose(fn)`, which registers a dispose hook (a hook registered while disposing runs at once);
- `window.open(url)` routed to `openExternal`, returning `null`;
- a click on an `<a download>` whose `href` is a frame-created `blob:` URL, a `data:` URL or a URL under the skill's route or this mount's asset prefix becomes `download`;
- a trusted click on any other `<a href>` outside those prefixes becomes `openExternal`;
- a fault channel: listeners on `window` for `error`, `unhandledrejection` and `securitypolicyviolation` post `ouro-widget-error`. Post at most 10 faults per frame, and each distinct kind and message once.

The bootstrap opens a `MessageChannel` and posts one hello to `window.parent` (§4, "The handshake"); every other message it sends or accepts travels on its end of that channel and carries the mount's `nonce`. It never listens to window messages.

`openExternal(url)` accepts only `http:`, `https:` and `mailto:` URLs and needs a user action in the frame. The frame's own `Blob` URLs cannot be fetched by the page, so the bootstrap remembers the `Blob` behind each URL it creates and sends the `Blob` itself.

## 4. The message grammar

Generate a fresh unguessable `nonce` for every mount (the builtin host uses 16 bytes from `crypto.getRandomValues`) and embed it in the bootstrap. Every message in both directions is a plain object with a `type` and that `nonce`.

**The handshake:**

The bootstrap posts `ouro-widget-hello` as `{type: "ouro-widget-hello", nonce}` to `window.parent` with target origin `'*'` and one port of a fresh `MessageChannel` in the transfer list. The page adopts that port only when `event.source` is this frame's `contentWindow`, the message carries a port and the proof matches: the mount's `nonce` for a module frame, the route lease token for a route page (below, and §8). A module frame is adopted once; a later hello is ignored. The page answers `ouro-widget-ready` on the port. From then on every message in the tables below travels on that port only, in both directions: the page never posts to `frame.contentWindow` and ignores window messages other than a hello. A port belongs to the document that created it, so a document that later occupies the frame (a navigation, a redirect, another site) can neither receive the mount's replies, events or theme nor speak for it.

A route page's bootstrap is the same script served at `GET /api/widget-bridge.js` (public, static JavaScript, `Access-Control-Allow-Origin: *`). It runs only in a frame whose URL lies under `/api/widget-resources/TOKEN/`, and its hello is `{type: "ouro-widget-hello", lease: TOKEN}` (its `nonce` is `""`): the lease in the document's own URL path is the proof, since neither a redirect nor a navigation carries the path to another document and the frame's `Referrer-Policy` is `no-referrer`. A route page may hello again from its next own document; adopt the new port and retire the previous document's requests, event and theme subscriptions and port. When a frame `load` follows no hello since the previous `load`, the previous document is gone: retire its channel the same way. In route mode the bootstrap leaves the page's own `fetch`, `window.open`, links, object URLs and fault handling untouched, resolves a relative request URL against the page's own URL, and learns its skill prefix from `ouro-widget-ready`.

**Frame to page:**

| `type` | Fields | Meaning |
|---|---|---|
| `ouro-widget-fetch` | `id`, `url`, `init: {method, headers, body, timeoutMs}` | Start one bridged request (§5). `id` is unique within the frame; `headers` is a list of `[name, value]` pairs. |
| `ouro-widget-fetch-pull` | `id` | The frame wants the next body chunk. |
| `ouro-widget-fetch-abort` | `id` | Abort that request. |
| `ouro-widget-events` | `op: "subscribe" \| "unsubscribe"` | Start or stop forwarding this skill's WebSocket events. |
| `ouro-widget-theme` | `op: "subscribe" \| "unsubscribe"` | Start or stop sending the theme. |
| `ouro-widget-download` | `id`, `name`, `source` | Save a `Blob`, a `data:` URL or a URL under the skill's prefix. |
| `ouro-widget-open-external` | `id`, `url` | Open an external link. |
| `ouro-widget-resize` | `height` | The measured content height in pixels (§5). |
| `ouro-widget-error` | `kind: "error" \| "rejection" \| "csp"`, `message`, `source`, `line` | A fault inside the frame. Show it on the card; builtin logging is described in §5. |
| `ouro-widget-disposed` | | The frame finished its dispose hooks (§6). |

`init.body` is a structured-clone value: `null`, a string, an `ArrayBuffer`, a typed-array view or a `Blob`. Preserve a view's offset and length. There is no base64 wrapper or required transfer list for request buffers. The builtin bridge posts the body without a transfer list; the page passes that cloned value to fetch. Response chunks have the separate transfer behavior below.

**Page to frame:**

| `type` | Fields | Meaning |
|---|---|---|
| `ouro-widget-ready` | `routeBase` | The page adopted this channel; `routeBase` is the skill's absolute route prefix. |
| `ouro-widget-fetch-chunk` | `id`, `phase`, and per phase: `headers`: `status`, `statusText`, `headers` (`[name, value]` pairs); `data`: `chunk` (an `ArrayBuffer`, transferred); `end`; `error`: `error` (text) | One step of a bridged response. |
| `ouro-widget-event` | `event`, `data` | One of the skill's WebSocket events. |
| `ouro-widget-theme` | `theme: "light" \| "dark"` | The page's current theme. |
| `ouro-widget-download-result` | `id`, `result` | The outcome of a download. |
| `ouro-widget-open-external-result` | `id`, `result` | The outcome of opening a link. |
| `ouro-widget-dispose` | | Stop now (§6). |

## 5. What the page does for each message

**Fetch:**
1. Resolve `url` against `ASSET_BASE` for ESM, the page's origin for classic (a route page sends absolute URLs). A URL under `RESOURCE_BASE` stands for the skill route of the same rest path. Refuse, with the `error` text `module widget fetch outside extension route prefix`, unless the result has the page's origin and its path starts with `/api/extensions/<skill>/` for this card's skill, or with this mount's `ASSET_BASE` for GET/HEAD only.
2. Use `credentials: "omit"` for asset reads, the owner's session (`credentials: "same-origin"`) for skill routes. Send the upper-cased method, the headers and the body. Use `redirect: "error"`: the prefix is checked once, so a followed redirect could carry the request and the session anywhere.
3. Post `headers` with the status, status text and every response header.
4. Read the body one chunk per `ouro-widget-fetch-pull`. Post each chunk as a `data` phase with its bytes as a transferred `ArrayBuffer`. Post `end` when the body ends. A `HEAD` request and a 204, 205 or 304 answer have no body.
5. On any failure post `error`.

There is no default timeout. A positive `init.timeoutMs` aborts the request after that many milliseconds, with the error text `widget request timed out`. `ouro-widget-fetch-abort` and stopping the card (§6) abort it too.

**Events.** While the frame is subscribed, forward every WebSocket frame whose `type` starts with this card's `ws_prefix` as `{type: "ouro-widget-event", event: <type without the prefix>, data: <the frame's data, or {}>}`. Other skills' frames never reach the frame.

**Theme.** On `subscribe`, post the current theme at once and again whenever it changes. `unsubscribe` stops that.

**Size.** Only a card without a fixed `height` sizes itself. The frame starts at 320 pixels. On `ouro-widget-resize`, set the frame's height to `min(ceiling, max(320, height + 2))`, where `ceiling` is `max_height` (default 8192) and the 2 pixels are the frame border. Ignore a height that is not a positive number. With a fixed `height`, ignore these messages. The builtin size reporter measures `#root` with a `ResizeObserver`, reports in the next animation frame, posts only a changed integer height, and hides the document's vertical scrollbar below the ceiling.

**Downloads.** A `Blob` or a `data:` URL is saved as it is. A URL must have the page's origin and lie under the skill's prefix or this mount's asset prefix; anything else fails. The builtin host streams a skill-route URL and reads an asset URL with `credentials: "omit"`, saving its finite captured bytes like a `Blob`. Answer with `result: {ok: true, ...}` or `{ok: false, error}`.

**External links.** Open the link only while the page has a user activation. Answer with `result: {ok: true, ...}`, `{ok: false, degraded, ...}` when only a fallback (such as copying the link) was possible, or `{ok: false, error}`. The frame treats `ok` or `degraded` as success.

**Faults.** Show the fault in the card's status (a card in a message shows it there) and leave the card running. The builtin host additionally posts it once per kind and message, at most 10 per mount, to `POST /api/widgets/fault` (`WidgetFaultRequest`): this card's own `skill`, `tab_id` and `revision`, `where: "board"` or `"message"`, and the frame's `kind`, `message` (clipped to 500 characters), `source` (200) and `line`; a value that is not a string is sent as `""`, a line that is not a whole number from 0 as `0`. Never take who the widget is from the message, and count per mount, not per frame: the widget's own code can read the nonce and post directly. Reporting is optional for other clients. A report is best effort: no retry, and a refusal changes nothing on the card. Until the stop finishes (§6) the frame still reports.

Both page and server redact capability-bearing asset URLs. The server normalizes unpaired UTF-16 surrogates in the clipped text to U+FFFD before writing UTF-8; valid emoji remain intact.

## 6. Stopping a widget

Stopping is a handshake:

1. Post `ouro-widget-dispose` on the frame's channel.
2. Keep answering bridged requests: the widget's dispose hooks may save state through its own routes.
3. When `ouro-widget-disposed` arrives, or after 1000 milliseconds, whichever is first: abort every pending request, stop forwarding events and the theme, stop listening for the frame's messages and remove the frame.

A frame that has already gone away, or never said hello, is removed at once. A module frame's first `load` is
its own document; a second `load` means it navigated to another document, which gets no dispose message: stop at
once (step 3), revoke its leases and say so on the card. Replies to a request belong to the port it arrived on:
drop them once that port was retired.

## 7. When widgets run

| `start` | Behaviour |
|---|---|
| `auto` | Starts when the widgets view is shown and stops when the owner leaves it. |
| `manual` | Shows the title, icon and a Start button. Runs only after Start. Leaving the view stops it. |
| `retain` | Starts on the first visit to the widgets view and keeps running on other views until Stop, until the skill leaves the card list, or until the page goes away. |

The owner's choice in `widget_start_mode` overrides `start`. Stop always wins: the builtin host keeps the owner's Stop on that device across reloads until Start or a change to `auto` / `retain`. Installing or enabling a skill never starts a widget: only showing the widgets view does. A frame never outlives the page that hosts it.

On WebKit, Android WebView and Firefox a frame shares the page's thread, so a start that loops or blows up takes the page with it. The builtin host writes each frame's start to a local ledger before the frame gets its document and clears it one second after the frame's first `load`, or when the page hides (its thread outlived the start); a card whose last start with this revision the page did not outlive is not started by itself on any view (its facade, one line, Start). A page opened with `?hold=all` holds every frame for that load.

## 8. Other widget kinds

- **`kind: "iframe"`:** mint a route lease for the card (§2, "Route leases") and give the frame `src` `RESOURCE_BASE` + `render.route`, `referrerPolicy` `no-referrer`, the same `sandbox` and `allow` attributes and no document policy of yours (the page keeps its own HTTP policy and URL base). Keep the lease while it runs. Listen for its hello (§4): a page that loads `/api/widget-bridge.js` gets the module bridge's grammar and relay rules, with no resize. On stop, run the §6 handshake if it joined, otherwise remove the frame at once; then revoke the lease.
- **`kind: "declarative"`:** a component description the host draws itself. The components are listed in `CREATING_SKILLS.md`. An interface that does not draw them should say the card is not supported rather than guess.

## 9. Frames in a message

Ouroboros can show a widget, or a small app of its own, inside its own chat message: a living card ([`CLIENT_CONTRACT.md`](CLIENT_CONTRACT.md) §5 "Living cards"). A frame in a message is not a Widgets card. It never joins the widgets view's cards or launch policy, and leaving the widgets view stops nothing in a message; the message disposes its frames when its row goes, and with the page. The builtin implementation is `web/modules/card_frames.js` and `web/modules/chat_cards.js`.

- **Size.** A frame in a message is at least 120 pixels tall and at most 70 % of the window's height. Taller content is cut until the owner presses Expand (then up to the 8192-pixel ceiling); Collapse cuts it again. There is no 320-pixel floor. A module widget's `height` is honoured up to the cap; a self-sizing one reports through `ouro-widget-resize` as on the widgets view. A declarative widget shown in a message is measured and cut the same way. The builtin host runs at most three frames in one feed, started when they come near the view; a frame far from it rests as the card's words with Open. A running frame has Close, which rests it the same way on that device until Open; a frame whose last start the page did not outlive (§7) rests so too, with its line, and an app opened from that hold starts without its saved state.
- **A skill's widget in a message** is the tab `GET /api/widgets` lists for the card's `{skill, tab_id}`. When it is not listed, show the card's words and say that the skill is off. The module bridge, its events and the theme work as on the widgets view; a widget fault is shown on the card. The card's `input` (an object) gives a declarative widget's form fields of the same names their first values; what the owner typed in this page session wins. A module widget gets no input.
- **An app** is HTML Ouroboros wrote, read as text from `GET /api/cards/{card_id}/app` and never opened as a page. It runs in its own frame, not the module frame: a created `<iframe>` with `sandbox="allow-scripts"` and nothing else (no `allow` attribute), whose `srcdoc` property is set to `<!doctype html><html data-theme="light|dark"><head>`, then this document policy as the head's first element, then a viewport line, the theme and the bridge script, then `</head><body>`, the app's HTML and `</body></html>`:

  ```
  default-src 'none'; script-src 'unsafe-inline' blob: data:; style-src 'unsafe-inline'; img-src data: blob:; media-src data: blob:; font-src data:
  ```

  An app has no fetch, WebSocket or load, no skill route, no request relay, no events, no downloads and no links out. No page policy covers two paths to the network: navigating its own frame (every navigation is a request; when one loads a second document the host removes the frame and says so on the card, while a navigation answered without a document leaves the app running) and WebRTC. The frame keeps the app away from Ouroboros and the page, not from the network; its only author is Ouroboros, which has the network anyway. The theme reaches the app as `data-theme` and the custom properties `--ouro-text`, `--ouro-text-meta`, `--ouro-accent`, `--ouro-surface`, `--ouro-tray`, `--ouro-border`, `--ouro-ok`, `--ouro-warn`, `--ouro-error`, `--ouro-font` and `--ouro-radius`, read where the frame lies: inside a dark voice they are that voice's own inks and surfaces, at mount and at every theme change.
- **The app bridge** is four messages from the frame and one from the page, each carrying the mount's nonce; the page acts only on messages from that frame's own window:
  - `ouro-card-resize {height}`: the content height. The bridge reports it itself as the document grows.
  - `ouro-card-error {kind, message}`: a script error, an unhandled rejection or a policy refusal (`kind` `error`, `rejection` or `csp`), sent by the bridge itself, each once and ten at most; the host shows it on the card.
  - `ouro-card-state {state}`: JSON text of at most 32 KiB. The host keeps it per card and app bytes in the tab's session storage and hands it back at the next mount.
  - `ouro-card-submit {value, summary}`: the host puts one line into the owner's draft, `From «<title>»: <summary>` followed by the value as compact JSON when that is 300 characters or fewer, and focuses the field. It never sends: only the owner's Send does.
  - `ouro-card-theme {theme: {name, css}}`, page to frame: the theme changed.

  Inside the frame an app uses `window.OuroborosCard = {resize(height), state: {value, save(value)}, submit(value, summary), theme, onTheme(callback)}`.

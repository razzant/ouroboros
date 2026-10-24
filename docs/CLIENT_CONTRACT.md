# Ouroboros client contract

A complete client needs this contract, [`WIDGET_HOST_PROTOCOL.md`](WIDGET_HOST_PROTOCOL.md) for widgets, and the core's HTTP/WebSocket endpoints; no source imports are needed.

`GET /api/contract` publishes routes, parameters, headers, statuses, WebSocket types and named JSON Schemas from the running core. The live API is authoritative about shapes; this document is authoritative about meaning: ordering, outcomes, identities and safe retries. Fetch that core's contract; versions differ. Report disagreements as defects. Tests cover structural completeness and selected producer/consumer behavior; §17 lists gaps.

## 1. Compatibility rules

- **Additive evolution.** New fields, routes and message types appear without notice inside an ABI version. Ignore unknown fields and unknown WebSocket message types. Never reject a response because it carries more than you expected.
- **Absence means unknown.** An optional field that is missing is not `false`, `0` or empty. For example, a missing cost is not a free task, and a missing `served_ui` means the core is older than this contract.
- **ABI version.** `GET /api/health` returns `abi_version` (currently `"7.0"`). It changes only when something incompatible changes. A core without `abi_version` and without `/api/contract` predates this contract. You can still talk to it, but none of the published facts below are guaranteed.
- **Feature detection.** Check that a route is listed in the contract before relying on it.
- **Schemas** use a small, draft-agnostic JSON Schema subset:
  - `type`, `properties`, `required`, `enum`, `anyOf` and `items`, with nested objects written inline.
  - `additionalProperties: true` everywhere.
  - `description` taken from the server-side contract type.
- **Opaque values.** A property published only as `{"type": "object"}` or `{"type": "array"}`, or with an empty schema, is opaque. Either the core passes it through from another component (the agent engine's own objects, §13), or its keys are open. Read only the keys this document names in it, and show the rest as data.
- **Methods.** The extension catch-all's `ANY` means `GET`, `HEAD`, `POST`, `PUT`, `DELETE` and `PATCH`, not `OPTIONS`. A selected skill can still refuse a method it does not handle.
- **Requests.** A request schema lists the keys a route reads, and its `required` list the keys it needs. Only some routes check a request against its schema mechanically (unknown keys allowed). The rest check their fields in their own code, so do not rely on any particular refusal for a malformed request. `headers` lists the request headers a route reads; a trailing `!` marks a required one. The core never checks its responses against these schemas, so check what your client depends on.

## 2. Where the client lives

A client and the core must share one browser origin, served from the root path. Cross-origin access and mounting under a sub-path are not supported. The browser session cookie and the WebSocket depend on a shared origin.

| Path prefix | Owner | What it serves |
|---|---|---|
| `/api` | core | HTTP API, including extension routes under `/api/extensions/{skill}/…` |
| `/auth` | core | login/logout operations while the network gate is active; optional public login assets (§3) |
| `/onboarding` | core | the selected setup page or builtin wizard; `/onboarding/static/*` always serves builtin assets |
| `/ws` | core | the WebSocket |
| `/` and `/static` | the interface | the main page and its assets |

The contract publishes these as `core_paths` and `interface_paths`. A reverse proxy that puts a separately hosted interface in front of a core sends the core prefixes to the core and everything else to the interface host. It must:
- pass WebSocket upgrades on `/ws`;
- keep request bodies and server-sent event streams unbuffered;
- never answer a core path with the interface's index page.

### Choosing what the core serves at `/` (`OUROBOROS_UI_DIR`)

| Value | `/` and `/static/*` serve |
|---|---|
| empty (the default) | the builtin interface |
| `none` | nothing. `/` answers a 404 page and `/static/*` answers 404. Use this when the interface is hosted elsewhere. Shells that open the core's own `/` (the desktop window, the Android app, the Telegram Mini App) show that page too |
| an absolute or `~/` directory | that directory, which must contain `index.html`, laid out like the builtin `web/` tree |

The rules for a directory:
- **Files are served as they are.** Every file is published under `/static/`, including dot-files. Point the setting at a dedicated build output, never a home or project folder.
- **No Telegram script injection.** The core never rewrites a directory's HTML. A custom interface that runs as a Telegram Mini App must include Telegram's own script itself.
- **macOS title bar.** The desktop window keeps its ordinary title bar unless the page's `<head>` declares `<meta name="ouroboros-window-chrome" content="integrated">`; a declaring page must then make room for the title bar the window reports in `data-window-chrome` and `--window-chrome-*` ([ARCHITECTURE "Navigation and shared UI contracts"](architecture/03-web-ui-pages-and-buttons.md#navigation-and-shared-ui-contracts)).
- **Fallback to builtin, with an error.** The builtin interface is served instead, and `served_ui.error` says why, when the directory:
  - is relative, missing or not a directory;
  - has no `index.html`;
  - contains the core's data tree, source tree or settings file.
- **File permissions remain the operator's responsibility.** Selection checks the directory and entrypoint's existence, not whether every resource can be opened later; an unreadable file can fail its HTTP request without selecting builtin.
- **The builtin `web/` folder counts as builtin.** A directory that is physically the builtin `web/` tree, under any name, is the builtin interface.

The rules for the setting itself:
- **When it applies.** The choice applies when the core starts. Saving a value different from the one the core started with lists `OUROBOROS_UI_DIR` in `restart_state.restart_keys`. The running core keeps serving its current choice until a restart.
- **Saved value versus environment.** A value saved in settings, including an explicitly saved empty value, beats the `OUROBOROS_UI_DIR` environment variable. The environment applies only while the setting has never been saved. An unrelated settings save never copies the environment's value into the saved settings.
- **Not the same as `--no-ui`.** The `--no-ui` command-line flag only means "open no desktop window". It does not change what the core serves.

`GET /api/state` reports what this core actually serves as `served_ui: {source: "builtin"|"directory"|"none", error?}`. This is the applied choice, not the requested one, and it is not a fingerprint of anyone's asset files.

### Complete setup and sign-in pages

A valid selected directory may also contain these optional files:

```text
index.html          main client
onboarding.html     complete setup page
auth/index.html     complete sign-in page
auth/...            public sign-in assets
```

`GET /onboarding` serves `onboarding.html`; under the active network gate, `GET /auth/login` serves `auth/index.html`. Each is returned unchanged. If its optional file is absent or cannot be read, that whole page uses the builtin implementation. The rules apply only to `source: "directory"`; builtin, `none` and rejected directory selections keep builtin setup/login.

Setup resources use the selected `/static/` tree. Login resources use `/auth/`, which publishes **only the selected `auth/` directory**, including before sign-in, on loopback and without a password. Relative resources in the login document resolve there. `login` and `logout` are reserved names for the gate's operations. Publish only public build assets in `auth/`, never settings or credentials. Missing custom assets answer 404; they never borrow builtin files. The applied directory remains fixed until restart, just like the main page.

`/onboarding?builtin=1` and `/auth/login?builtin=1&next=<path>` explicitly open the builtin pages without changing settings. The builtin wizard loads its own complete `/onboarding/static/` graph; the login form is self-contained. A builtin setup request keeps builtin sign-in when authentication is needed, so it remains recoverable even if both custom pages have broken JavaScript. The server cannot detect a script that fails after the HTML loads: recovery is an explicit action, not an automatic fallback timer.

On macOS launchers containing this change, **Help → Open built-in setup** works independently of page JavaScript in the setup and main windows. A directory or separately hosted copy of the builtin client offers recovery outside its setup iframe. The core-owned builtin main page omits that extra control: it already opens builtin setup. Other clients own their recovery controls. This change adds no native recovery control to Windows/Linux, Android or Telegram shells; their builtin HTTP URLs remain available, but a shell without an address bar needs its own host action or a browser to open them. Old installed launchers do not acquire a new menu until their launcher code is updated.

In a separately hosted deployment with `OUROBOROS_UI_DIR=none`, the core's setup/login pages stay builtin. That interface host can supply its own pages at its own interface addresses and call the same APIs (§3–4); these optional files are specifically for a directory served by the core. See the [minimal client example](examples/client_views/README.md).

### Personal source changes and updates

For a personal version of the builtin application, change its ordinary source through the existing body development and review workflow. A normal merge update brings official changes into those local commits; disjoint edits merge automatically, while textual conflicts and incompatible behavior need the agent or author to resolve and test them. This includes UI code: no separate UI updater or release-note checklist is required. A replacement/reset has different semantics (§15). A separately maintained client owns its releases; updating the core cannot invent that client's new screens. Using the published setup vocabulary (§4) can expose new fields in a data-driven form, but does not automatically implement a new interaction.

### When a page served by the core should reload

The builtin page served by the core carries `<meta name="ouroboros-ui-owner" content="core">`. It is added to the response and is never part of `web/index.html`. Pages without it are owned by someone else. Apply these rules after reconnecting to a restarted core:

1. If `GET /api/state` has no `served_ui`, the core is older than this contract. Keep comparing `sha` exactly as before: a changed `sha` means the page's files changed, so reload.
2. If the page has the marker and `served_ui.source` is `builtin`, compare `sha` as above. Any other source means the core no longer serves this page, so navigate to `/` once.
3. If the page has no marker (a directory interface or a separately hosted one), the interface owns its own refresh. A changed core `sha` means "resynchronize data", not "your files changed".

Nobody tells an unmarked page to reload. A new choice reaches it when someone next loads `/`.

## 3. Authentication

**When the gate applies.** The network auth gate applies only when two things are true: the core has a network password (`OUROBOROS_NETWORK_PASSWORD`), and the core sees the request as coming from a non-loopback address. Loopback requests and cores without a password are not gated. Behind a reverse proxy, the core trusts `X-Forwarded-For` and `X-Forwarded-Proto` only from the addresses named in uvicorn's `FORWARDED_ALLOW_IPS` (default `127.0.0.1`):
- **Same-machine proxy:** it must send `X-Forwarded-For`. Otherwise every remote visitor looks like loopback and passes without a password. Have the proxy overwrite the header with the real peer address, rather than trusting a visitor's claim to be loopback.
- **Proxy on another machine:** it must be named in `FORWARDED_ALLOW_IPS` for TLS to be recognised.

**Proving identity.** Use one of:
- the session cookie set by `/auth/login`;
- a per-request `X-Ouroboros-Password: <password>` header;
- `Authorization: Bearer <password>` or Basic.

**Refusals.** A refused HTTP request answers `401` with `{"error": "Authentication required."}`. A `GET` that asks for `text/html` gets the builtin login form with 401, or redirects to `/auth/login?next=…` when a selected login page exists. Explicit builtin setup recovery keeps the builtin form. Health, login/logout and GET/HEAD of the selected `/auth/` assets are public; the main `/static/` tree, APIs, setup bootstrap and extension resources retain the gate.

**Signing in from a browser interface:**

1. **On 401, go to the login form.** When a core route (for example `GET /api/state` or `GET /api/onboarding`) answers 401, navigate the top-level page to `/auth/login?next=<path>`. Use the page's own path, query and fragment, URL-encoded.
2. **Return to `next`.** After a correct password the form answers `303` to `next`. A JSON client posts `{"password", "next"?}` to `/auth/login` and receives `200 {ok: true, next}` with the cookie. A wrong password answers `401` and sets no session.
3. **What `next` accepts.** It must be a same-origin path: it starts with one `/`, and contains no backslash, no control characters (tab and newline included) and no quotes or angle brackets. Anything else becomes `/`.

**A custom login page** submits JSON to the same `POST /auth/login`, displays its 401 error in its own UI and navigates using the successful response's sanitized `next`. No HTML placeholders or private bootstrap are required. HTML form POST keeps the builtin behavior: a wrong password returns the builtin form with 401, a correct one redirects with 303. Use JSON to retain custom error presentation. `GET /auth/login?builtin=1` always opens the builtin form while the gate is active.

**The session.** It is an HttpOnly, SameSite=Lax cookie named `ouroboros_auth`. It is `Secure` when the core sees TLS. It lasts 30 days, and a core that loses its key file signs everyone out.

**When a session expires:**
- **Requests:** they start answering 401. Run the same login steps.
- **Failed changes:** never resend a change that failed before or during sign-in. Read the current state again and let the person decide.

**The WebSocket.** An unauthenticated `/ws` is refused before the connection is accepted (HTTP 403 during the handshake). Browsers report it as close code 1006, never a custom code. Recognise sign-in problems through an HTTP probe, not through WebSocket close codes.

**Signing out.** `GET` or `POST /auth/logout` deletes the cookie and answers `303` to `next`. Login/logout answer 404 when the gate is inactive; those names never serve selected files. Other `/auth/` paths serve the optional public bundle even without the gate: missing files answer 404, unsupported methods 405. With no bundle those paths answer 404.

## 4. First contact and setup

1. **Check the core.** `GET /api/health` checks the core and returns `abi_version`. `GET /api/contract` lists the routes this core offers.
2. **Check whether setup is needed.** `GET /api/onboarding` answers `204` when a usable model provider is configured (nothing to set up), otherwise `200` with the builtin wizard page.
3. **Load or show setup.** Frame or open `/onboarding` (the selected complete page when present, otherwise builtin), or render your own setup from `GET /api/onboarding/bootstrap`. Force the builtin wizard with `/onboarding?builtin=1`. The readiness probe's 200 HTML body remains builtin for existing clients.

`GET /api/onboarding/bootstrap` returns `{ready, bootstrap}` and writes nothing. `bootstrap` is exactly what the builtin wizard renders. It contains:

- **`contract`**, the setup vocabulary:
  - `providerFields`: credential and endpoint inputs; `settingKey` is the settings key to send.
  - `modelSlots`: model roles; `settingKey`.
  - `reviewModes` and `runtimeModes`: `value` is what the setting stores.
  - `localRoutingModes`.
  - `budgetFields`: with defaults and minimums.
  - `subscriptionFields`: declarations the setup call accepts, not settings.
  - the step texts.
- **`initialState`**, the starting values:
  - one key per row `stateKey`;
  - some fixed keys, such as `providerProfile`.

  A credential that is already configured reads as `secretPlaceholder`, never the secret itself.
- **`modelDefaults`** per provider profile, `modelSuggestions` and `localPresets`.
- **`freshInstall`**, for display only. The server checks it again when setup completes.

Setup can also check what it is about to save: `POST /api/providers/test` probes one provider (a failed probe is `200` with `ok: false`), `POST /api/openai-compatible/models` lists the models of an OpenAI-compatible endpoint, `GET /api/model-catalog` lists selectable models (§14), and the local-model routes start and test a local model (§14). None of them saves anything.

To finish setup, send `POST /api/onboarding/complete`:
- **The body:** the settings to save, keyed by each row's `settingKey`. Add `subscriptionsConnected` and `skipSubscriptionPresets` as needed, and optionally `OUROBOROS_SUBAGENTS` and `allow_empty_review_pool`. A credential field left as `secretPlaceholder` keeps the stored value.
- **What it does:** this is the only setup writer. Settings, the next-start access mode and the completion record are saved together, or not at all.
- **On success:** `200 OnboardingCompleteResponse`. `restart_required` says whether access settings apply only after a restart.
- **If agent accounts can't be checked:** `503` with `can_skip: true` means the connected agent accounts could not be verified and nothing was saved. Resend with `skipSubscriptionPresets: true` to finish without agent defaults.
- **Other refusals:** these follow §9's `saved` rules.

**Mapping setup state to the save body.** For `providerFields`, `modelSlots` and `budgetFields`, copy `initialState[row.stateKey]` to `row.settingKey`. The fixed fields map as follows:

| Initial-state field | Completion-body key |
|---|---|
| `reviewEnforcement` | `OUROBOROS_REVIEW_ENFORCEMENT` |
| `runtimeMode` | `OUROBOROS_RUNTIME_MODE` |
| `skillsRepoPath` | `OUROBOROS_SKILLS_REPO_PATH` |
| `localSource` | `LOCAL_MODEL_SOURCE` |
| `localFilename` | `LOCAL_MODEL_FILENAME` |
| `localContextLength` | `LOCAL_MODEL_CONTEXT_LENGTH` |
| `localGpuLayers` | `LOCAL_MODEL_N_GPU_LAYERS` |
| `localChatFormat` | `LOCAL_MODEL_CHAT_FORMAT` |
| `localRoutingMode` | `LOCAL_ROUTING_MODE` |
| `modelAccounts` | `OUROBOROS_MODEL_ACCOUNTS` |
| `modelContextWindows` | `OUROBOROS_MODEL_CONTEXT_WINDOWS` |
| `processingPreference` | `OUROBOROS_PROCESSING_PREFERENCE` |
| `modelProcessingPreferences` | `OUROBOROS_MODEL_PROCESSING_PREFERENCES` |

`providerProfile` is derived from configured access, not saved. `localPreset` selects values from `localPresets`, not a saved key. Each `providerProfiles` entry contains display strings `label`, `providerCopy` and `modelCopy`. The model-role maps accept JSON objects or JSON text. Role keys are `main`, `light`, `vision`, `consciousness`, `deep_review`, `websearch` and `fallback`; `fallback` holds an ordered array. Processing choices are `standard`, `fast`, `economy`, or empty to inherit.

**Keyless endpoint example.** On a fresh core, this is structurally sufficient when the endpoint and model are the operator's intended access:

```json
{
  "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:11434/v1",
  "OPENAI_COMPATIBLE_API_KEY": "",
  "OUROBOROS_MODEL": "openai-compatible::<model-id>",
  "skipSubscriptionPresets": true,
  "allow_empty_review_pool": true
}
```

Replace `<model-id>` with the endpoint's model id. An empty review pool is an explicit choice, not a production recommendation. Setup validates configuration, not reachability or successful generation; test the provider separately. Optional model slots may be blank to keep inheritance. Completion on an existing install is still a settings write, not an idempotent readiness check. `400` rejects input, `403` is a permission refusal, and `409 onboarding_state_changed` means the settings changed before commit. Apply the `saved` rules to failures.

`POST /api/onboarding/subagents/preview` takes the same body and returns, without saving, the agent roster and the model settings setup would produce.

### Completing setup in a host window

The HTTP completion receipt is the save result. Closing a window or receiving a bridge return value does not prove a save. After a confirmed successful receipt, use the host that actually exists:

- **Separate desktop setup window:** if present, call `window.pywebview.api.onboarding_finished({ok: true, restart_required, runtime_mode})` with the receipt's values. The existing host closes setup and owns a required server restart before opening the main client. The bridge appears asynchronously (`pywebviewready`); detect the actual method rather than infer it from the core version. The main window does not expose this setup-only method. If it is unavailable, keep the saved/restart state visible and offer the host's normal close/restart path.
- **A setup iframe inside the builtin or copied builtin client:** send `parent.postMessage({type: 'ouroboros:onboarding-complete', restart_required, runtime_mode}, location.origin)`. That parent's overlay owns its transition and restart display. This is a client convention; another parent can define its own handoff. Do not call the parent's setup-window bridge or close the main window.
- **A plain top-level page:** navigate to `/` only when no restart is needed. Otherwise show that setup was saved but the new access mode is pending, and use the ordinary restart workflow (§15). A navigation alone does not apply a boot-pinned setting.

On a failed or unknown save, retain the form and follow the existing readback rules; do not send a successful completion message or blindly save again. The native `onboarding_finished` method only reports the HTTP result, and introduces no second settings writer.

## 5. Conversation: live events and history

### State

`GET /api/state` (`StateResponse`) contains:
- **Readiness:** `supervisor_ready` and `supervisor_error`.
- **Activity:** the work projection — `works` (one row per request with its Continue episodes and live members: `activity` working|queued|waiting|paused|unknown — the current episode's own word while it lives (a Continue held for its earlier attempt says Working while that attempt's helpers or runs still work), else what still runs —, `transition`, `call`, typed `because` (the episode's own reason with one English sentence; a decided Pause adds its declared author `by` `{actor, text, at}`), `tails` (helpers, external runs, review work beside an episode that is not executing, runs nobody can confirm: said beside the word, never as it), `live` (something executes), `turn` for what waits for the owner, `release`, member ids, the `controls` its doors accept now and `mark`, its shape: `moving` working, `still` queued or waiting on something that ends by itself, `pause` paused, `ring` unconfirmed, `none`), `rooms` (one status per chat id: word inputs, the same `mark`, `amber` while `turn` holds something for the owner), `work_summary`, `works_complete` and `works_gaps`. A client renders these and derives no state of its own. The frozen root census `active_chat_activities` (`last_line`/`last_line_at`, `tool_calls`, `detached`, `project_id`/`project_name`) and legacy `active_direct_turns` remain for compatibility.
- **Projects:** `projects` (a compact feed that may be capped) and `project_chat_ids` (the complete set).
- **Main's unread:** `main_visible_revision`, Main's room revision (its standalone messages, the count a Project's `visible_revision` is; `null` while unreadable); the shared read cursor is `main_seen_revision` in `/api/ui/preferences` (`null` until a client first sets it; a write is clamped to the revision and never moves back).
- **Current decisions:** `current_questions` (outstanding questions independent of the recent history page; Main's own as `native` identity rows without form fields) and `main_attention` (human-required task actions); `*_complete` flags and `*_gaps` tell a complete empty set from unavailable evidence. On incomplete reads keep earlier content without granting stale controls.
- **Modes and budget:** the access and context modes, and money figures (`null` means unknown).
- **Checkout:** `sha`, the first 8 characters of the commit on disk (the running core's start commit is `running_sha` in `GET /api/update/status`); `branch`, `null` for a detached HEAD or unknown branch.
- **`served_ui`.**

A quiz-bound activity may carry `owner_wait.quiz_id`, the exact question its wait
names, in Main as in a Project; absence supplies no question identity. To open it,
match the exact task and quiz in `owner_quiz` of `GET /api/tasks/{activity_id}`.
Its `question_rooms[quiz_id]`, the shared room resolver's projection (not the
task's contextual `project_id` or immutable publication/reply address), proves
current display membership: `state:known` supplies `chat_id` and `project_id`
(empty for Main) to match Main's chat or the named Project (its Main mirror
included); `unknown` or absent proof permits no form, and hidden chat 0 proves
Main only through that exact quiz's recorded Main publication (§7). Opening is a
read, never a decision submission; stale socket epochs authorize nothing. Wait
and quiz state stay separate (resuming the task does not answer the question);
answer through §7.

### The WebSocket `/ws`

**Format and liveness.** Open one socket per client; every frame is a JSON text message with a `type`. While clients are connected the core sends `heartbeat` about every 15 seconds (proof the connection is alive, not that work is progressing); after about 45 silent seconds, treat the socket as dead and reconnect. Before restarting, the core closes sockets with code 1012.

**Message types.** `ws.messages` in the contract lists every built-in type with each direction's shape:
- **Client to core:** `chat` and `command`.
- **Core to client, stored as history rows:** `chat` (a message or a progress line), `photo`, `video`, `document`, `links`, `quiz` and `card` (a living card, §5 "Living cards").
- **Core to client, signals only:**
  - `panic_accepted` confirms the emergency latch, not process exit (§15).
  - `quiz_state` updates a question card; history shows the merged state.
  - `card_state` says a shown living card changed: read it again (§5 "Living cards").
  - `log` carries one runtime event row in `data` (`data.type` names the event) for live activity displays; the task's durable facts come from §6.
  - `message_annotation`, the routing outcome shown on an owner message, is that row's `chat_annotation` in history.
  - `task_named` gives a card its title; history rows carry `suggested_name`.
  - `projects_changed`, `extension_lifecycle`, `update_status_ready`, `update_progress_changed`, `ui_language_changed` and `ui_i18n_updated` mean "read the named state again".
  - `typing` (transient) and `heartbeat`.

Some `chat` frames are transient notices history never shows, for example `system_type: "initialization_notice"` (a message could not be accepted yet) and a repair/admission receipt with `task_id_pending: true` (no proof a durable task is readable yet).

**Extension messages** are namespaced per skill (`ext_<n>_<token>_<message>`; use `ws_prefix` from `GET /api/widgets`, trailing separator included). Send an object with `type: ws_prefix + <handler-name>` and that handler's fields; the core passes the whole object (no `data` wrapper required). A non-null handler result returns `{type: "<request-type>.reply", data: <result>}` to the requesting socket; no result, no reply. The core adds no request correlation id (the skill's format correlates concurrent requests); refusals/failures are `log` frames with `data.level` and `data.message`, without guaranteed request id or refusal code.

### Sending a message

Send `{"type": "chat", "content": "...", "client_message_id": "<unique id>", "chat_id": <thread>}`:
- **Threads.** `chat_id` 1 is Main; a project's thread is its `chat_id` from `GET /api/projects`, sent with its `project_id`.
- **Attachments.** Upload each file with `POST /api/chat/upload` (multipart field `file`), then list it in `attachments` as `{filename: <stored id from the upload>, display_name, mime}`; the core measures size and checksum itself.
- **Replies.** `reply_to: "<row_address>"` (a work card: `task:<task_id>` of its current episode) names the row meant; the core stores its resolved reference on the owner row as a fact for Ouroboros, routing nothing.
- **What "sent" means.** A socket send only means the socket took the frame; the message is accepted when the core echoes it as a `chat` frame from `role: "user"` with your `client_message_id` and `ingress_accepted: true`. `ingress_pending` and `ingress_dispatched` say what this core process knows of it and are lost on a restart.
- **Retrying.** The same `client_message_id` with the same words and attachments joins the already accepted message; with different content it is refused.
- **What is lost.** A frame sent while the socket is down is lost unless your client keeps it: the core keeps no client outbox.
- **Commands.** Slash commands go as `{"type": "command", "cmd": "/restart"}` and so on, or through `POST /api/command` (§15).

**Refusals and submission correlation.** Schema rejection sends `{type:"log", data:{level:"warning", message:…}}`, without guaranteed correlation. Acceptance failures (reusing an accepted id for different content among them) can send transient `chat` notices with `role:"system"`, `system_type:"initialization_notice"`, the submitting `chat_id` and any supplied `client_message_id`, and command, admission, update and review System replies can name the submission they answer: settle only that submission's Sending indicator, in that room, never a guessed one. A System row alone establishes neither acceptance nor task completion; without an accepted echo, keep the operation unresolved and read history.

**Origin correlation.** Optional `origin_message_ref` (`{chat_id, client_message_id, ts, text_sha256}`) names the accepted owner row; its hash checks that row's text, not reply text. Match its thread and owner-message id, then track `task_id`; an absent reference establishes no causation.

**Thread filtering.** The socket broadcasts several destinations: apply the merge rules below only to stored-row deliveries of the history view being read, never comparing every frame with Main history, inferring Main from a missing `chat_id` or taking transient notices for missing stored rows.

### History rows

In a Project response, `window.latest_message` names the latest eligible non-owner standalone message by stored arrival (card content excluded), not necessarily the last row by timestamp; null means unknown, and its `out_of_order` flag says it sorts above earlier arrivals or lies before the recent read. With a null message, older pages continue the bounded search below the chat offset `window.latest_before`; `window.latest_absent` means the search reached the chat's start without a gap and found no eligible message below the frozen upper bound, not that the chat has no stored rows. These are search observations, not a client's read acknowledgement.

`GET /api/chat/history?chat_id=<thread>` returns `ChatHistoryResponse`. The selected conversation rows are oldest first; replay evidence can follow them regardless of its timestamp and recur on other pages: apply it to the keyed card or question, never count it as another conversation message. Each row of `messages` is a `ChatHistoryMessage`:
- **Always present:** `text` (live frames call it `content`), `role` (`user`, `assistant` or `system`), `ts`, `is_progress`, `system_type` (the row's kind; `""` for an ordinary message) and `task_id`.
- **Identity:** stored rows carry `history_id` (`"<chat|progress>:<offset>"`, the same on every re-read), `history_position` (`{source, offset}`) and a stable `row_address` computed before projection (none without a persisted source). `client_message_id` correlates owner messages and some quiz publications; owner authorship follows `role`.
- **References:** `reply_to` (an owner row's: the row he replied to; an Ouroboros answer's: the owner message its turn answers) is `{address, chat_id, role, author_label, kind, snippet, ts, task_id}` (`snippet`: at most 160 whitespace-collapsed characters) or `{address, unresolved: true}`. A `relay_entry` row's `relay` holds the owner's words passed into the room: `quotes`, every message passed on, each with `text`, `attachments` and `source` (where and when he wrote it), the singular fields repeating the first. A reply to a relay, in its source room, carries `reply_to_relay`.
- **Deliveries:** `msg_type` is `document`, `photo`, `video`, `links`, `quiz` or `card`, with `filename`, `mime`, `download_url`, `caption` and `size_bytes`; `actions` and `title` for links; `quiz` for a question (`quiz.quiz_id`, `quiz.state`, the recorded answer); `card` for a living card (`card.card_id`, `card.kind`, `card.title`, `card.revision` as first published; the row's `text` is its plain fallback).
- **Progress rows** (`is_progress: true`) carry the task-progress facts of live `chat` progress frames (`ChatOutbound`).
- **Rows that are not messages.** Do not show these as bubbles:
  - replay evidence (`system_type` `quiz_answer`, `quiz_closed` or `task_evidence`, or a row with `summary_kind` and `historical_terminal`), facts for cards whose rows lie on other pages. A `task_evidence` carrier may be an in-room parent's only evidence (zero tool calls included; `task_terminal_status` only when the parent is terminal): mint a parent card only from the parent's own rows, carrier or live state, never from a child's `parent_task_id`; a child whose parent has no such evidence here renders as its own card and re-nests when that evidence arrives.
  - `system_type: "task_model_wait"`: a model wait's latest state, in `model_waits`, for the task's card.
- **A rebuilt row:** `lifecycle_virtual: true` describes the skill operation running now, rebuilt with a fresh `ts` on every read and without `history_id`: replace it rather than merge it.
- **Derived and synthesized rows:** in Main a Project question appears as `system_type: "project_question_pointer"` (with `quiz_id`), carrying the question row's `history_id`. A Project's start message rebuilt from its binding (`origin_projected: true`) and a synthesized `origin_omission` System notice (further archived origins were not rendered) have no stored `history_id`.

### Merging live frames and history

The core writes each stored row and its live frame with an identical `ts` string, the same `task_id`, `role`, `system_type`, `is_progress`, `source` (when provided) and, for a text row, words, so a live frame and a history row are the same row when:

| Live frame | History row | Same row when |
|---|---|---|
| an owner `chat` (`role: "user"`) | the owner row | `client_message_id` is equal |
| `quiz` | `msg_type: "quiz"` | `task_id` and `quiz_id` (`quiz.quiz_id`) are equal |
| a Main pointer `chat` with `system_type: "project_question_pointer"` | the pointer row | `task_id` and `quiz_id` are equal |
| `photo`, `video`, `document` or `links` | `msg_type` equal to the frame's `type` | `task_id` and `ts` are equal |
| `card` | `msg_type: "card"` | `card.card_id` is equal |
| any other `chat` | a row with the same `is_progress`, `role` and `system_type` | `task_id`, `ts` and `content` = `text` are all equal |

Only `role:"user"` rows merge by `client_message_id`: on a System frame it only correlates the answered submission and is not persisted, so stored System replies use the other-chat rule. Never match on `ts` or `task_id` alone: one task writes many rows with close timestamps, and the same words can come from different tasks.

**On every connect and reconnect:**
1. Open the socket and keep the frames it delivers.
2. Read `GET /api/state` and the history of the threads on screen.
3. Show the history rows, then apply the kept and later frames, skipping a frame that matches a row you already have.
4. Recognise a stored row you already have by its `history_id` when you read history again or page through it.

The socket has no resume cursor: frames sent while you were disconnected are not resent; history and the task reads (§6) are the record.

### Reading history

- **How much.** `n_human` (default 150, at most 1500) bounds conversation rows and `n_progress` (default 60, at most 600) progress rows.
- **Paging.** While `has_more` is true, pass `next_cursor` back as `cursor` to read older rows. `page_cursor` re-reads the same page. Cursors are opaque.
- **New arrivals.** The newest page carries `after_cursor` (absent when a read failed part-way or the window holds more tasks than a cursor can name); passed back as `cursor`, it returns the rows stored since that read with the room's current work (never paging older, ordered by time) and the next `after_cursor`, so a client that keeps a window open reads its new rows instead of the whole window again. In a project thread `window.latest_message` names the newest message among them when one arrived (absent: none did); `out_of_order` also counts the window it continues.
- **Around a task.** `around=task:<task_id>` (never together with `cursor`) reads the page around that task's earliest row of the thread, wherever it lies: about a quarter of each quota before it and the rest after it. `anchor` says what was found: `{"state": "found", "task_id", "history_id"}` with an ordinary page (`page_cursor` re-reads it, `next_cursor` continues older; it is not the newest page, so it has no `after_cursor`), or `absent` (the thread's saved history holds no row of it a client could show) or `moved` (its rows belong to another thread), both with no rows and no cursors. A source that cannot be read answers `503 history_source_unavailable`, and a task not found while the saved history is still being upgraded answers `503 migrating`: unknown, never absent. A stream holding none of the task's rows carries none of its context: the page names it in `window.truncated_by` (`chat_around_unpositioned` or `progress_around_unpositioned`), and on that page and the older pages it continues to, that stream's `coverage` span is null.
- **Around an exact row.** `around=<row_address>` uses the same bounded page/cursors, locating the canonical identity in both chat and progress. `anchor` carries `row_address` and either `found` with `history_id` and `chat_id`, or `moved` with destination `chat_id`, or `absent`, `ambiguous`, `unavailable`, with no rows/cursors. Only complete readable misses in both streams prove absent; an available exact match survives the other stream's import/failure. Otherwise an import answers `503 migrating` and an unreadable source (a same-time different row beside an unreadable line included) `503 history_source_unavailable`, which may persist: say so rather than promising a retry. Current-room membership (including actual source-reference occurrences) wins; otherwise home follows the existing room rule without source references, among the bound room, written room and Main. Unreadable authority for that choice (the registry's room identities, the home's own row, bindings, the row's task lineage) answers `503 history_source_unavailable`; another room's malformed routing or another row's lineage never does. Inactive/deleted homes and existing rows with no visible home are unavailable; retained context is never the original. Read the returned destination through `/api/projects?chat_id=…` when it is not Main, then request the same address there. Land only on the exact row address or its bound history ID, opening and scrolling containing timelines; a task card/portal is insufficient. Only the target's stream is positioned: the other carries none of its context on that page or the older pages it continues to (`<stream>_around_unpositioned`, null `coverage` span) until the newest page is read again. Showing an old source reads no newer messages. Gestures, newer navigation and leaving invalidate pending reads; a refused or failed read keeps the view and live-edge following; ordinary reopening still shows newest. An unresolved acceptance-time quote retries current state.
- **Errors.** An error answer keeps the same shape, with no rows and `has_more` true, and echoes the cursor you sent as `next_cursor`, so the same request can be repeated:
  - `400 history_cursor_invalid`, or `history_around_invalid` for a malformed task/row `around` or one sent with a cursor;
  - `409 history_view_changed`: which rows belong to the thread changed (its projects or task bindings); start again without a cursor;
  - `409 history_delta_reset`: since an `after_cursor`, more arrived than one read replays, a stored row could not be read, owner-message annotations changed, a task the window shows settled or changed after it settled, or a shown skill operation ended; read the newest page again (in a busy room this is a common answer, not a rare one);
  - `503 history_source_unavailable` or `migrating`.
- **Incomplete windows.** `window.complete` reports the conversation window's known bounds and gaps; `window.truncated_by` names each cause (`quota`, `page`, `lineage_cap`, a stream gap such as `chat_malformed_jsonl`, or `chat_importing`/`progress_importing`: older rows exist below the archive import's frontier and are not readable yet; the older page that would cross it answers 503 `reason_code: migrating`, retry later). Never treat an incomplete window as an empty conversation. `coverage` reports delivered coordinates, not chronology; import gaps use the final spans after quota/lineage deferral. Neither `complete` nor `has_more=false` certifies the entire narration archive: older progress can lie below the conversation floor with no ordinary next page. The builtin client checks older chat spans and recent progress gaps, but does not aggregate older progress-only gaps.

### Living cards

Ouroboros can put an interactive card into its own message: a `card` frame (`CardOutbound`), in history a `msg_type: "card"` row. Both carry only the card's identity `card: {card_id, kind, title, revision}` and its plain `fallback_text` (the row's `text`), which a surface that cannot draw the card shows instead.
- **Body.** `GET /api/cards/{card_id}` (`CardResponse`) returns the current card with exactly one body: `components`, a declarative tree drawn like a declarative widget (`docs/CREATING_SKILLS.md`) with `data` as the values its path fields read and no skill route; `widget`, `{skill, tab_id, input?}`, that skill's tab from `GET /api/widgets` drawn inline (it may no longer be live); or `app`, `{sha256, size, url}`. A card that is gone answers `404 card_not_found`: show the fallback text.
- **Apps.** `GET /api/cards/{card_id}/app` returns the app's HTML as `text/plain` with `nosniff` and `no-store`. Run it only inside an opaque-origin sandboxed frame built from those bytes (`srcdoc`, never `allow-same-origin`) whose policy blocks fetch, WebSockets and loads, never as a page of the core's origin; navigation and WebRTC are outside any page policy (`docs/WIDGET_HOST_PROTOCOL.md` §9). What the app produces goes into the owner's draft; only the owner's Send sends it.
- **Changes.** A `card_state` frame (`CardStateOutbound`) says the card got a new body (`card.revision` grew) or a press was recorded or read (`submissions_summary`); read the card again. History keeps the row as first published; `GET /api/cards/{card_id}` is the current card.
- **Presses** on a component card's `action` or `form` go to `POST /api/decisions` (§7).
- **Keeping a card.** `POST /api/cards/{card_id}/pin` keeps a component card on the Widgets page as a snapshot (`CardPinResponse`); an app or widget card answers `409 card_not_pinnable` and the 51st pin `409 pins_full`. `DELETE /api/cards/{card_id}/pin` removes it. Both broadcast `extension_lifecycle` with `skill: "_cards"`. `GET /api/widgets` lists kept cards after the skills' widgets, with `skill: "_cards"`, `pinned_card: true` and the snapshot in `render`; their presses use the same decisions.
- **Drawing a card (what the builtin client does).** Show the row's `fallback_text` until the body arrives, and whenever it cannot be read (with a retry for a failure, a plain note for `404`); never an empty box. Draw `components` with no skill route (only inline `data:` media), `data` as what path fields read; local components (tabs) never call the core. Each press is a new `request_id` and carries the `revision` it was drawn from; a press that got no answer, a `5xx` or `202 owed` is retried with the same `request_id` and the same values, and after a reload a `once` control's or an action's press recorded `owed` in `submissions_summary` is retried under its `request_id` (a form's next press with values is a new press); a `once` control stays locked after its press was delivered or forwarded (here, from `submissions_summary`, or a `409 card_action_once`); a `409 card_stale` reads the card again (a kept copy says it is out of date). A `widget` card draws the tab `GET /api/widgets` lists for its `{skill, tab_id}`, and says the skill is off when there is none. An `app` runs only as `docs/WIDGET_HOST_PROTOCOL.md` §9 says: an opaque `allow-scripts` frame that cannot fetch or load, whose submit fills the owner's draft and sends nothing; its bridge reports the app's errors, which the card shows. A card's frames in a message are sized and disposed there too.
- **Starts the page did not outlive.** A frame (and a skill's page module, below) shares the page's thread on WebKit, Android WebView and Firefox, so a start that loops takes the page with it. The builtin client writes each start to a local ledger (this browser's or app's storage) before the code runs and clears it once the page outlived the start: a second after it, or when the page hides. An item whose last start the next page finds is not started by itself (its face, its line, the owner's Open or Start; an app opens without its saved state, which it forgets), and `?hold=all` in a browser's address holds every item for one load (the page drops it from the address). Not held: a stop later than that second, or WebKit's later memory kill; held although nothing stopped: an app quit before the start settled (quitting ends the page without `pagehide`), or a start another tab of the same browser has in flight when this one opens. A running frame has Close, kept on that device until Open. The page then posts what it found, once: `POST /api/ui/holds` (`UiHoldsRequest`: `holds: [{key, rev}]` with `app:<card_id>`, `widget:<skill>:<tab_id>` or `module:<contribution key>`, and the raw `surface` facts `pywebview`, `telegram`) names new holds in one System row (`system_type: "ui_hold"`) per owning chat, the room a card was published in or the owner's chat (`UiHoldsResponse`: `named`); a repeat names nothing; `503 chat_unavailable` and `503 in_flight` (another tab is naming the same hold now) wrote nothing, after `503 not_confirmed` a row may be missing, and after any 503 the page reports again. Another client may keep its own ledger and report the same way.

## 6. Projects and tasks

**Projects:**
- **List:** `GET /api/projects` (`ProjectListResponse`): active and deleting projects, at most 200, most recently active first.
- **Create:** `POST /api/projects` (`ProjectCreateRequest` → `ProjectCreateResponse`). One source at most: `path` (attach a folder), `git_url` (clone) or `with_workspace`. Creating an existing `id` with no source changes nothing and answers the existing row with `created: false` (a flag optional on other responses, whose absence proves neither new creation nor failure); with a source it answers 409. A clone that needs credentials answers 401 with `error_code: "auth_required"` (not the network gate).
- **Turn a task into a project:** `POST /api/projects/from-task` (`ProjectFromTaskRequest`) takes no idempotency key: the server derives `handoff_id` from the project, the task and its origin message, so repeating the call for the same task returns the same `handoff_id`, and with the default project id (`task-<task_id>`) a repeat also adopts the project the task's work already has (`adopted: true`). A task bound to another project answers 409. Without `name`, naming may wait for one model call.
- **Update and delete:** `POST /api/projects/{id}/update` accepts `name` (at most 80 characters) and/or strict boolean `pinned`, changing only supplied fields. Old rows read unpinned. Invalid updates answer 400 without writing; missing/inactive rows answer 404, a deletion racing the update 409. `POST /api/projects/{id}/delete` starts deletion (a repeat is harmless). Pins affect sidebar selection/order, not list-API recency, activity, routing or read cursors.
- **Changes from elsewhere:** `projects_changed` announces new projects: add their `chat_id` to the threads you listen to before reading the list again.
- **Choosing a folder to attach:** `GET /api/fs/dirs?path=` lists directories under the home folder only (at most 500, `truncated` beyond that).

**Starting a task.** `POST /api/tasks` (`TaskCreateRequest`). The `task_id` you choose, or the one returned, is the task's identity.
- **Duplicate id:** an existing id answers 409, so a retry with the same id cannot start the task twice.
- **Outcome unknown:** `503` with `status: "unconfirmed"` means the task may or may not be queued; `GET /api/tasks/{task_id}` can answer 404 while it is still queued, so a 404 is no proof it will not run. The same `task_id` sent again answers 409 and never starts a second copy. Never resend under a new id; watch the task and let the person decide.
- **Other 503s:** they come before the task is scheduled, and the attempt is cleaned up: nothing runs, `GET /api/tasks/{task_id}` answers 404, and the same `task_id` can be sent again.

**Reading tasks:** `GET /api/tasks` (filters `status`, `limit`, `queue_only`) and `GET /api/tasks/{task_id}`. Records identify `task_id`, `status`, `ts` and `updated_at`; `result`, artifacts and evidence appear when available. List rows omit bulky evidence; read the detail for the result. `status` accepts a comma-separated filter; `limit` defaults to 50, positive values cap at 500, and `0` requests all rows. `queue_only=1` returns `{tasks:[], queue}` without history.

A Project completion frame/history row may carry `completion_notice`: `{neutral: bool, detail: string, phase: string, salvage?: string, cause?: string, undescribed?: string[]}`, a transport view, not a verdict: neutral means technical caveats stay in task details, never independently approved success. Full outcome/answer fields remain authoritative; `salvage` preserves the labelled intermediate excerpt. Main adds the work's `task_name` and `cause`, a display sentence whose unknown typed reasons read neutrally, with their exact codes in `undescribed` for accessible details. Canonical text and `detail` retain full Project context and codes; authored warning prose stays unchanged. Older rows may omit these fields; replay may derive missing metadata from the canonical result without rewriting retained text. Without recoverable facts, clients retain legacy text rather than parse it into guessed fields.

The terminal lifecycle set is exactly `completed`, `failed`, `cancelled`, `rejected_duplicate`. Other values, including `requested`, `scheduled`, `running`, `interrupted` and a pause or pending cancellation, do not prove completion. Terminal lifecycle alone does not mean objective success.

**Following a task.** Both event methods answer `text/event-stream`:

```text
id: 12
event: task_event
data: {"seq":12,"type":"task_result","task_id":"…","data":{…}}

```

`data.type`, not the SSE event name, identifies the payload. A `task_result` with terminal `data.status` proves completion. EOF alone does not: the wait may have ended or the connection failed.

- **GET, legacy replay:** `cursor` is a nonnegative integer, the last processed SSE `id` / `seq`, not `TaskEvent.event_id`. `wait` defaults to 30 seconds, clamped to 0–120; `0` replays the available view without waiting for future work. The deadline can emit `: heartbeat` and close.
- **POST, v2 replay:** start with `{"v":2,"wait":30}` or a null cursor. Return the complete cursor from the last processed frame unchanged to resume. `wait` must be an integer 0–120. `cursor.view` and `cursor.positions` are opaque; this cursor cannot be exchanged with GET's integer.
- A v2 deadline emits `cursor_checkpoint`; changed lineage/view emits `cursor_replay` and may replay stored events. Diagnostics/checkpoints can reuse `seq`; replayed stored events can get a new `seq`. Process their cursor and diagnostic information even when that sequence was seen. Deduplicate stored v2 events by nonempty `event_id`; replace `task_result` snapshots with the latest snapshot. Do not assume v2 delivery is timestamp-sorted or `seq` identifies the same stored event across replay.

**Schedules** start ordinary tasks later or repeatedly:
- **List:** `GET /api/schedules` (`ScheduledTasksResponse`). Each row's `status` is `active`, `disabled`, `consumed`, `held`, `waiting` or `delete_pending` (a withdrawn row). Rows also carry `display_name`, `kind` (`note` or `task`), `timezone`, `destination`, `waiting_detail` and, once the schedule has run, the last run's `outcome` (`state`, `quiet`, `notice`, `ended_at`, `line`; `state` is `pending_observation` until the settled result is read). `previous_run_open` appears while the host sees the previous run still open; a withdrawn row carries `withdrawal` (`at`, `by`, `reason`, `work_in_flight`: true, false or null when unknown, `restorable`). An unreadable table answers 503, never an empty list.
- **Create or edit:** `POST /api/schedules` (`ScheduleUpsertRequest`). An existing `id` edits that schedule; without an `id` a repeat creates another one. `ok` is true only when the audit record landed (`schedule.audit` is `recorded`). A 409 means nothing was written: the schedule table or its audit could not be used safely.
- **Disable, delete, restore:** `POST /api/schedules/{id}/action` (`ScheduleActionRequest`: `action` and a required `reason`); `DELETE /api/schedules/{id}` also deletes. Delete withdraws (`status: withdrawn`): the row stays, disabled and restorable, until it is removed after the retention window once its work has settled; Restore cancels the withdrawal once no hold remains on it (`withdrawal.restorable`), and posting the same `id` edits the withdrawn row without restoring it. A refusal is also 200, with `ok: false`: `changed` is the durable fact and `status` says what happened. An action governs future runs only; `running_or_queued` reports a run it already admitted (`null` when unknown).

**Downloading a result file:**
- **Where:** `GET /api/tasks/{id}/artifacts/{name}` (optionally one of `relpath`, `archive`, `source` or `wrap_up`).
- **Partial downloads:** `206` is a byte slice, not whole JSON. Wrap up's copy is regenerated per request; never join its ranges across requests.
- **Checking the bytes:** the `x-ouroboros-artifact-sha256` header gives the checksum when the core verified the bytes.
- **Changed after recording:** `409 artifact_identity_changed` means the file changed after it was recorded.

## 7. Questions and decisions

**How a question arrives.** A task asks the owner through a `quiz` frame (`QuizOutbound`), in history a `msg_type: "quiz"` row, updated by `quiz_state` frames. Its state is `open`, `answered`, `closed` (explicitly closed without answering), `expired_terminal` (the task finished; an answer can still be recorded) or `superseded`.

Private Presence contacts carry `presence_contact` source facts separately from authored text: on the live quiz or Main report, and inside `quiz` on a saved question. The voice line renders its `label` and optional `url`; `task_id`, `binding_id` and `source_event_id` retain the source identity. The source grants no owner authority or external reply route.

**Answering.** Send `POST /api/decisions` with `DecisionRequest` and a `decision_id` of `quiz:<task_id>:<quiz_id>`:
- **What to send:** `option_index`, `comment` (nonempty when alone), or both; `option_index`, `answered_index` and `recommended_index` are zero-based positions in `options`. A negative/non-integer index answers `400 option_index_invalid`, one beyond an open card's options `400` with `error:"option_out_of_range"`, and neither index nor comment `400 option_index_required`. An already closed card can instead return its recorded 409 state; an absent quiz is `404 quiz_not_found`.
- **Safe retries:** the same `request_id` returns the already recorded answer with `duplicate: true`, even if the retry's payload differs: display that answer, not the retry's input (a successful replay accepts no changed answer). It also lets the existing delivery-recovery path finish after a partial failure.
- **First answer wins.** A different answer to an already answered question answers `409` with the recorded state.
- **Late answers.** An answer after the task ended is still recorded (`answered_after_terminal`) and goes to the card's `publication_chat_id`, recorded before the card was first shown and never changed, not to its ask-time chat: a question asked in the hidden partition (0) and shown in Main is answered in Main (1), and a converted task keeps that address while history shows its card in the current resolved room. `forwarded:true` confirms the answer entered the ordinary owner-message path there; `forwarded:false` is not delivery: `hidden_chat` and `a2a_chat` have no owner turn, `publication_address_unknown` (an older card without a proven address) stays recorded and owed with no guessed room, and `late_answer_delivery_unknown` means proof of dispatch was lost and nothing is resent. Show this returned outcome; never infer delivery from the room the card appears in.
- **Recorded but not delivered.** `503 quiz_history_write_failed`, `mailbox_write_failed` or `late_answer_not_delivered` mean the answer is recorded but not yet delivered: retry with the same `request_id`.
- **Not recorded.** `503 quiz_answer_unconfirmed` means no answer was recorded; retry the same `request_id` and answer.

**Closing without answering.** For a root task's quiz, send the same endpoint and `decision_id` with a `request_id`, `action: "close"` and an optional `reason` of at most 2000 characters, without `option_index` or `comment`; a blank or omitted owner reason records `Closed by the owner without an answer.`, and a non-string or longer one answers `400 close_reason_invalid`. Closure shares the first-winner rule with answers; show the returned `state`, `closed_reason` and `question_delivery`: a recorded closure is no proof its author received it.

**Removing unknown late delivery from current decisions.** For an answered quiz with unknown late-answer delivery, send `action: "dismiss"` with `decision_id` and `request_id` only (no answer, comment or reason). It removes only the outstanding delivery item, neither resending the answer nor marking it delivered; the recorded answer and unknown outcome stay in history. Another, ineligible state is refused; read the returned facts.

**Pressing a living card's control.** Send the same endpoint with `decision_id` `card:<card_id>:<action_id>` (the control's `submit.id`), a `request_id`, the card `revision` the control was drawn from (`400 revision_required` without it; a card that changed since answers `409 card_stale` with the current `revision`, and nothing is recorded), and for a `form` its field values as `values` (field names to text of at most 4000 characters, numbers, booleans or null; at most 50); other values, or `values` on an `action`, answer `400 values_invalid`. Each press is its own record: the same `request_id` returns the recorded press with `duplicate: true` and retries a delivery still owed, whatever the card became since; a control declared `once` keeps only its first press, and a later one answers `409 card_action_once` with `winner` (its `request_id`, time and `delivery`; a winner still `owed` is retried under its `request_id`). `delivery` says where the press went: `delivered` into the authoring task's mailbox, or its admitted Continue successor's (not proof it was read; a paused, held or queued author keeps it until it continues, `reason_code` `awaiting_recipient_read` on the answer and in `submissions_summary` until it reads it; one that ends first keeps it among its unread mail), `forwarded` as the owner's own message in the card's chat because that task ended (their words only, the pressed label and each value, as a Reply to the card's row; like a late answer it joins the turn an earlier late answer or press of that task started, never another task's turn), or `owed` (`202` with `reason_code`: resend the same `request_id`). `404 card_not_found` or `action_not_found` names a card or control that does not exist now; a replaced card can drop a control.

**Other decisions.** Model-wait choices (`model_wait:<task_id>:<wait_id>`) and persisted routing choices (`routing:<client_message_id>:<token>`; no new routing pickers are produced) use the same endpoint and idempotency rule.

## 8. Stop, pause, continue, resume

**Stop.** `POST /api/tasks/{id}/cancel` takes an optional `TaskCancelRequest`: `cascade`, `stop_policy` and `stop_action_id`.
- **Safe retries:** `stop_action_id` names one Stop action. While that action is pending, an identical request joins it and reusing the id with different arguments answers 409. An immediate Stop records the stop first and answers after it drove the teardown; once the task is no longer active, even an identical retry can answer 404. Read task detail to distinguish the recorded outcome.
- **Stopped:** `200` means the teardown settled (with `cascade: true` when the whole tree was stopped).
- **Recorded, still stopping:** `202` with `cancel_state: "pending"` means the stop is recorded and the supervisor finishes it: a graceful stop under way, or an immediate one whose teardown is not confirmed yet (`reason_code: "teardown_unconfirmed"`, with the stop's `request_id`), including an external run whose cancel its engine has not confirmed, or cannot while the engine is unreachable. No retry is needed: read `GET /api/tasks/{id}` (`cancel_state`) until it settles; an immediate Stop still hardens a graceful one.
- **Nothing to stop:** `404` means the task already finished (completion wins), the id names no task, or a repeated Stop already finished. A task that was created but never started is stopped, not 404, and so is a finished task whose tree a `cascade: true` Stop still finds running (a helper task or an external run).
- **Nothing recorded:** `503 cancel_intent_write_failed` means nothing was recorded or cancelled, so retry; `503 cancel_intent_projection_corrupt` cannot succeed until the projection file is repaired.
- **When to offer Stop:** chat frames carry `cancelable` to say when Stop applies. Do not offer it otherwise.

`stop_policy` is `immediate` (also the omitted/empty default) or `finalize_then_cancel`. A graceful request cannot soften a previously recorded immediate Stop. Task detail has `cancel_state:"pending"` while cancellation is owed, otherwise the field is absent; there is no public `cancel_state:"settled"`.

**Partial work.** No `status:"partial"` exists; read `result`, artifacts and `outcome_axes` separately. Graceful wrap-up can produce best-effort output, immediate cancellation salvage, and natural completion wins a race with Stop. Wrap up's `delegate_terminal_reconciliation.owner_stop_children` keeps child/run material and readers on any exit; detail `wrap_up_document` names its on-read Markdown; terminal execution is not final accounting; engine continuity grants no session transfer. Cancellation keeps recorded evidence; capture can fail or be missing. Artifact states: `pending`, `finalizing`, `ready`, `ready_with_changes`, `ready_no_changes`, `failed`, `missing`, `not_applicable`; read the file list.

**Pause, continue, resume, hurry:**
- **Pause:** `POST /api/tasks/{id}/pause` takes `{request_id}` and may declare who pressed: an interface sends `actor: "owner"` for the owner's own press, and a caller acting for Ouroboros sends `actor: "ouroboros"` with a `reason`, one sentence of at most 400 characters that the owner reads under the work's name. The declaration only names the Pause's author and grants no power; an `actor` the core does not know declares nobody, and a Pause that declared nobody is never shown as the owner's. Any other field answers 400. `202` with `latch_pending` means the pause is recorded and still taking hold; repeat the same `request_id`.
- **Continue:** `POST /api/tasks/{id}/continue` takes exactly `{action_nonce}`. A timeout does not mean it failed: retry with the same nonce. The successor task's id is derived from the nonce, so it cannot start twice.
- **Resume:** `POST /api/tasks/{id}/resume` returns saved work from a budget, owner or technical pause (`TaskResumeResponse`), after fresh control and money checks. Unreadable billing conditions refuse with HTTP 409, `code=billing_authority_unavailable`, the cause in `reason_code`, and explanatory `error`; this is not a missing task or exhausted money. It is not idempotent: a repeat is refused (409 or 404). After an unclear answer, read the task.
- **Hurry:** `POST /api/tasks/{id}/hurry` takes exactly `{request_id}`.

## 9. Settings

**Reading.** `GET /api/settings` (`SettingsReadResponse`) returns every setting with its value, plus `_meta` (`SettingsMeta`):
- **Network facts:** `bind_host` and `recommended_url`.
- **`restart_state`:** which saved settings are waiting for a restart.
- **`setup_contract`:** the same vocabulary as setup.
- **Other information:** the policy and agent roster.

Secrets are masked. A password-type secret reads as the configured placeholder; others read as a short prefix followed by `...`.

**Saving.** `POST /api/settings` takes only the keys you change (a document of setting keys, so its body is published as plain `json`):
- **Masked values:** sending a masked value back means "unchanged". `""` clears a value.
- **Settings with their own writer:**
  - The interface language is changed only through `POST /api/ui/i18n/language`; the settings save reports it in `ignored_keys`.
  - The access, context and safety controls go through `POST /api/owner/*`.
- **On success:** the answer is `SettingsSaveResponse`. `restart_state.restart_keys` lists settings that apply only after a restart; `OUROBOROS_UI_DIR` is one of them.

Every settings writer (Settings, setup, owner controls, language, desktop background) reports failures as `SettingsWriteRefusal`, and `saved` says what happened:

| `saved` | Meaning | What to do |
|---|---|---|
| `false` | nothing was written | correct the input and retry |
| `true` | the settings were written; a later step (`post_commit_failed`) failed | retry that step; never save the same settings again |
| `null` | unknown; the writer outlived the wait | read `GET /api/settings` again before deciding; an unknown outcome is never a reason to retry |

`ignored_keys` is an optional object mapping each skipped key to its own writer, not an array. Unknown ordinary settings keys can be ignored rather than rejected. `immediate_changed` and `next_task_changed` are optional true flags for at least one key in that effect bucket, not lists. Budget-field names, types and minimums come from `setup_contract.budgetFields`. There is no public request intended to induce post-commit failure or an unknown save outcome.

**Owner controls.** Each saves one setting through the same writer:
- `POST /api/owner/runtime-mode` `{mode}`: `light`, `advanced`, `pro` or `cyber_pro`; applies when the core next starts.
- `POST /api/owner/context-mode` `{mode, project_id?}`: `nano`, `low` or `max`; applies to new participants in that room, including while others run. Omitted `project_id` means Main.
- `POST /api/owner/safety-mode` `{mode}`: `full`, `light` or `off`.
- `POST /api/owner/auto-grant` `{enabled}`.
- `POST /api/owner/effort-range` `{min, recommended, max, project_id?}`: effort tiers (`none` to `ultra`), min ≤ max; primary `recommended` is independent. Omitted `project_id` means Main. Both endpoints return `project_id` and the saved `cognition` (`context_mode`, `effort_range`); project summaries carry the same object. Main state fields remain Main's.

`POST /api/owner/capability-ack` (`CapabilityAckRequest`) vouches for one exact route's maximum response or context window. It writes acknowledgement evidence, never settings; a repeat overwrites it and zero revokes that exact route's acknowledgement with an audit.

**Other settings routes:**
- `GET /api/settings?response_limit_preview=1&model=…&local=false&account=…&role=main&window=0` reads stored API evidence and the selected subscription's local catalog without writing. The catalog supplies account binding and maximum response; stored window evidence uses that binding. Optional `window` previews a role draft; omission uses its saved value. `window` returns tokens, kind, reserve, source, observation date, disagreement, retired history and shared text; `window_route` plus `window_ack` supports exact-route Clear.
- `GET`/`POST /api/ui/preferences` hold installation-wide owner preferences shared by windows and devices (including widget arrangement). Unknown keys are refused. These are separate from browser-local appearance; do not initialize shared preferences as if they belonged to one replacement client.
- `POST /api/settings/secret` reveals one stored secret on purpose: exactly one of `{key}`, `{mcp_server_id}` or `{mcp_server_id, header_name}`. It writes nothing; `value` is `""` when the secret is not set.
- `GET`/`POST /api/desktop/autostart` and `/api/desktop/background` (`{enabled}`) read and set the desktop app's start-at-login and keep-running choices. Where this host cannot honour one, its `state` is `unavailable` with a `reason`.

**The interface language:**
- `GET /api/ui/i18n` returns the language and its translation memory.
- `POST /api/ui/i18n/language` chooses a language and broadcasts `ui_language_changed`; `ui_i18n_updated` tells every client to read the memory again.
- `GET /api/ui/i18n/export` downloads a language's memory file; `POST /api/ui/i18n/import` (`UiI18nMemoryDocument`) replaces its imported layer, keeping the owner's own entries; `POST /api/ui/i18n/regenerate` drops generated entries and generates them again in the background. Each bumps the memory's `revision`.

## 10. Files

**Chat attachments.**
- **Uploading:** `POST /api/chat/upload` (multipart `file`) returns `UploadResponse`. `filename` is the stored id; `view.url` is the download link.
- **Lifetime:** an upload stays pending until a message accepts it.
- **Deleting:** `DELETE /api/chat/upload` with `{filename}` deletes only pending uploads (otherwise 409); a repeat answers 404.
- **Downloading:** `GET /api/files/download?upload=<id>`.

**The file browser** works inside one root: the configured file-browser folder. Without one, a loopback request browses the home folder and a remote one is refused (400). Every path is relative to that root:
- **Reading:** `GET /api/files/list` (at most 500 entries, directories first), `read` (text capped at 256 KiB or 120,000 characters), `content` and `download` with `?path=`.
- **Writing:** `POST /api/files/upload` (multipart `file` and `path`; at most 100 MiB), `mkdir`, `write`, `delete` and `transfer`. Each answers `FileChangeResponse`.
- **Repeating a write:** `write` overwrites, so it is safe to repeat. `mkdir`, `upload` and a repeated copy answer 409 (the target exists); a repeated `delete` or `move` answers 404.

Errors:
- `..`, control characters or escaping links answer 400.
- Owner-only files answer 403.
- A target that already exists answers 409.

Files produced by tasks are covered in §6.

## 11. Skills and extensions

**What is installed:**
- `GET /api/extensions` (`ExtensionsIndexResponse`) lists every installed skill (`InstalledSkill`) and what is live. It makes no network call.
- `GET /api/extensions/{skill}/manifest` and `/settings_section` read one skill.
- `GET /api/skills/daemons` lists companion processes.

**One lifecycle lane.** Enabling, disabling, granting, deleting, reviewing, attesting and every install, update or uninstall share one lane. Toggle, grant, delete and attest await their terminal result by default, including for older UI copies and dictation. With `Prefer: respond-async` they answer the result when fast, otherwise `{ok: true, pending: true, job_id, message}` after about ten seconds: accepted work, not completed setup. Catalog install and replacing update answer their job immediately (§12); review may keep the request open for minutes. Closing a connection does not stop its job. Watch `GET /api/skills/lifecycle-queue` (the running job and last 80, in memory) or `system_type: "skill_lifecycle"` progress, then re-read `GET /api/extensions` on settlement. A lost response does not authorize a blind retry.

**The revision guard.** `content_hash` names a skill's revision. An action that sends `expected_content_hash` is refused with 409 when the live revision differs. Re-read `GET /api/extensions` and show the current revision before acting again; that refusal need not include the revision.

**Owner actions on one skill:**
- **Enable or disable:** `POST /api/skills/{skill}/toggle` (`SkillToggleRequest`). Repeating it is harmless.
- **Grant keys or permissions:** `POST /api/skills/{skill}/grants` (`SkillGrantRequest`). Only current manifest requests can be granted; a repeat merges. Optional `enable_after_grant: true` carries an explicit Enable through the same job, even after a pending response. Omission grants only, preserving an existing install-time turn-on request. Terminal `setup_error` means grants succeeded but setup did not; `enabled` reports explicit Enable's result.
- **Review:** `POST /api/skills/{skill}/review` answers 200 with the review's outcome (`SkillReviewResponse`). A review already running for the same revision answers `status: "pending"` with its `job_id`. `GET /api/skills/{skill}/review-history/{job_id}` renders a finished review.
- **Attest a review:** `POST /api/owner/skills/{skill}/attest-review` records the owner's verdict: 200 with `ok: true` (possibly pending above), otherwise 409.
- **Delete:** `POST /api/skills/{skill}/delete` removes an external or self-authored skill and its state; a repeat answers 404.
- **Places:** `GET /api/places` returns records and connector declarations; `POST /api/places` (`PlaceConnectRequest`) connects one. `POST /api/places/{place_id}` (`PlaceUpdateRequest`) atomically edits owner choices; `expected_revision` detects concurrent edits. `DELETE /api/places/{place_id}` disconnects it, keeping room history. `GET /api/places/{place_id}/rooms` (`PlaceRoomsResponse`) returns that Place's captured rooms: canonical `room_chat_id`, its `address` and nullable `unanswered_count`. Unknown counts remain null, unavailable reads answer 503, and an unknown Place answers 404; GET starts no focus. Read messages through the existing `GET /api/chat/history?chat_id=...` cursor contract, not a transport log. Records and connector ingress: CREATING_SKILLS, “Places and focus transport ingress”.
- **Publication check:** `POST /api/skills/{skill}/publish-preflight` is read-only; every status answers the same shape, and `state: "hard_block"` is a refusal. An optional body `{destination: {source_id, mode}}` (`SkillPublishPreflightRequest`; mode `origin`, `entry` or `new_entry`) picks where it would go; the answer's `repository` is that destination (default: where the skill came from, else the hub). Missing publication prerequisites may answer `409` before the skill is looked up; read `reason_code`, not an assumed `404` for an unknown name.
- **Reconcile:** `POST /api/skills/{skill}/reconcile` brings one extension's live state in line with its saved state.

A refused action answers `SkillActionRefusal`: `error`, the `status_code` it was answered with, and optional current facts. For example, a revision-guard refusal carries only `ok`, `error` and `status_code`; absence of revision, review or grant fields proves none of those states. When an action loads or unloads an extension, the core broadcasts `extension_lifecycle` `{skill, action, reason}`: read `GET /api/extensions` and `GET /api/widgets` again.

**Widgets:**
- `GET /api/widgets` (`WidgetsResponse`) lists the widget cards. Each card has a `render.kind`:
  - `iframe`: frame the extension route named by `render.route`.
  - `declarative`: a component description the interface draws; the components are described in `docs/CREATING_SKILLS.md`.
  - `module`: reviewed JavaScript run in a sandboxed frame. [`WIDGET_HOST_PROTOCOL.md`](WIDGET_HOST_PROTOCOL.md) is the complete host protocol: frame, document policy, message grammar, fetch relay, events, theme, size, downloads, links and the dispose handshake.
- `GET /api/extensions/{skill}/module/{entry}` serves reviewed module code (never cached, any origin may load it).
- `POST /api/widgets/fault` (`WidgetFaultRequest`, `WidgetFaultResponse`) keeps one fault a running module widget's card shows as a `widget_fault` row of `GET /api/logs/events`: the card's `skill`, `tab_id` and `revision`, `where` (`board` or `message`) and the frame's `kind`, `message` (at most 500 characters), `source` (200) and `line`. The row has no task and is the page's report, not a verified execution. At most 10 rows per widget and 100 together are written per minute; past that the answer is `logged: false` and the widget's next row carries `suppressed_count`. `400` another shape, `413` a body past 8 KiB, `503` not written. A client may skip reporting.

**Contributions (controls, speech engines and places):** `GET /api/ui/contributions`
(`UiContributionsResponse`, `Cache-Control: no-store`) returns `contributions`,
`offers.stt`, `icons`, `problems` (skill → warning strings), `preferences` and
`voice`. The same fields appear beside `skills`/`live` in `GET /api/extensions`;
`live.contributions` contains the loaded extensions' captured entries.
Instruction skills contribute only while enabled and allowed by their current
review gate; extension entries belong to their loaded publication. Invalid
semantic entries are dropped individually and reported, also by skill preflight.

Each contribution has `key` (`<skill>:<id>`), `skill`, `skill_label` (manifest name),
`id`, `label`, `title`, `icon`, `order`, `revision` and `render: {kind, url}`,
plus either `placement` or `provides`. Placement entries carry `action` and chips
carry `source: {url, refresh_s}`. For `skill.request`, `action.url` and `route_url`
name the host-computed absolute route; providers use `route_url`. Optional
provider fields are `runtime`, `destination`, `languages`, `formats`,
`install_size`, `max_bytes` and `max_seconds`. Place providers require an extension
with the `presence` permission, `id`, `label`, and `destination: vendor:<Name>`;
`scope` defaults to `account` (or `conversation`), `wakes` lists typed wake facts,
`source` uses the chip contract, `settings` names a section, and `read_only: provider`
indicates provider support. Their `tools` list carries full names and declared
`effect`, `target`, `conversation`, `thread` facts. No speech-engine runtime/route
is required. `icon_spec` is `{kind:"host", name}` or `{kind:"svg", svg}` with
sanitized geometry, else `{}` for the fallback; the `icon` declaration remains
a host name or `ui/icon.svg`. SVG assets carry sandbox CSP. Installed and Browse
rows expose `connects` (boolean) and `icon` (the same spec); absent catalog fields
remain compatible. `icons` is the host glyph catalog. `offers.stt` lists non-live built-in providers as
`{skill, skill_label, id, label, destination, install_size, enabled: false, installed}`;
`installed: false` names a built-in this install has not installed (clients send the
owner to the Skills page's Browse, where the built-in source installs it).
There is no marketplace offer until its catalog exposes contribution metadata;
clients can link to the Skills entry instead.

`render.kind: host` runs trusted same-origin code and requires an extension with
`page_module` permission and its ordinary hash-bound grant/review authority.
`GET /api/extensions/{skill}/ui/{revision}/{path}` serves only captured `ui/`
assets of that live revision: `.js` as `text/javascript`, CSS, JSON and images by
type, with no-store and nosniff. Example entry:
`/api/extensions/voice/ui/<revision>/ui/voice.js`; relative imports stay in that
revision. Missing, non-UI, unsupported, unloaded or stale paths answer 404.
Like `module/`, a GET under `ui/<revision>/` is host-owned, never the skill's own
route. These modules are distinct from the opaque-origin widget frame protocol.
The builtin client records each module's start in its start ledger (Living cards above) and does not import a
module again by itself after a start of that revision the page did not outlive; one notice names it with Start it.

`GET`/`POST /api/ui/preferences` persist `voice: {stt:"", tts:"", tts_voice:""}`
(`tts_voice`: the chosen tts provider's voice id, empty for its own pick) and
`contributions: {hidden:[], order:[]}` in the install-wide preference file.
POST merges named nested keys, replaces each supplied list, and rejects unknown
nested keys, non-string keys, strings over 120 characters and lists over 64 items
with 400. A missing provider never clears or substitutes a saved key.
The contribution response includes these preferences and `voice[kind]` as
`{key, available, reason}`: reason is empty when available, `not_selected` for an
empty choice or `provider_unavailable` for a missing chosen engine. Client API
availability is a separate browser fact. Re-read on `extension_lifecycle`, WS
open, the end of a skill lifecycle job (its `chat` row with `system_type:
"skill_lifecycle"` and `lifecycle.phase` completed, failed or cancelled) and
`ouro:skill-lifecycle`; owner toggles invalidate instruction entries too.

Server STT uses POST of raw audio to `route_url`, with its recorded `Content-Type`
and optional `?lang=` (the builtin client sends none). Reply: `{text}`, `{text:"", status:"no_speech"}` or
`{error:{code,message}}`. The existing 512 KiB extension-route cap is unchanged.
Optional `max_bytes` is 1..524288 and `max_seconds` 1..600; `formats` is a subset
of `audio/webm`, `audio/ogg`, `audio/mp4`, `audio/mpeg`, `audio/wav`. The recorder
honors these limits before upload; 393216 bytes at 32 kbps for 60 seconds is a
provider declaration, never a host exception by skill name.

**The skill's own routes:** everything else under `/api/extensions/{skill}/` belongs to the skill.
- A route that does not exist answers 404, and a skill that is not running answers 409.
- A method the skill did not declare answers 405.
- A request body over 512 KiB answers 413.
- A skill that fails answers 502.

**Behind a network password:**
- **Direct frame loads are not portable.** Cookie attachment depends on the browser and request mode. In the separate-host tests, a passive image was refused with 401 in Chromium but sent the session cookie and loaded in WebKit. Do not depend on either outcome for other assets.
- **Loading through the host.** Such files must be fetched through the widget host and loaded from `blob:` URLs, as `docs/CREATING_SKILLS.md` describes.

**Shared controls.** The author kit of shared controls always comes from the core's builtin files, whatever interface is selected.

## 12. Skill sources, the catalog and installs

**Sources** (the owner's list, `OUROBOROS_SKILL_SOURCES`; while it is unset the list is derived from older settings and not written):
- `GET /api/skill-sources` (`SkillSourcesResponse`) lists the sources with their `trust` (`trusted` installs without review, `review` reviews each skill), `confirmed`, and read `state` (`ready`, `reading`, `error` with `error`, or `search` for a registry); `managed: true` marks a row a build's settings supplied (it stays editable). `materialized` says whether the list is saved; `problem` explains a saved list that does not validate (the derived list is in use).
- `POST /api/skill-sources/probe` (`SkillSourceProbeRequest`: `url` or `path`) reads an address once and answers `{kind, suggested_name, owner, found}` (`owner`: the hub's maintainer, else the account the address names; absent for a folder) or `{error}`. Plain HTTP is accepted with `warning: "not_encrypted"`; embedded passwords and HTTP(S) usernames are refused. Nothing is saved.
- `POST /api/skill-sources` (`SkillSourceAddRequest`), `POST /api/skill-sources/{source_id}` (`SkillSourceChangeRequest`: `name`, `trust`, `confirmed`) and `POST .../{source_id}/remove` are owner settings writes answering `SkillSourceChangeResponse` (400 with `code` `invalid_source` or `unreadable_source`). A trust change never rewrites an existing install. `POST .../{source_id}/refresh` starts a background fetch. Local sources load in place as `user_repo`, with ordinary review and global name uniqueness; removing a row stops loading its folder without deleting it. Install never replaces that owner folder. Reconnecting the default public hub restores missing Official (`trusted`) and Community (`review`) rows, both unconfirmed, and preserves any surviving row's choices.

- `POST /api/skill-sources/my-skills` (`MySkillsRequest`: `{}`) uses the GitHub token in Settings to request a private `<login>/ouroboros-skills` repository with an initial commit, or connect an existing repository with its actual visibility; no `gh` CLI is needed. A new git source is named **My skills**, with `trust: "trusted"` and `confirmed: true`; reconnecting preserves the owner's name, trust and confirmation. Success is HTTP 200 `{ok: true, created: bool, repository: "<login>/ouroboros-skills", private: bool, source: SkillSourceRow}` (`MySkillsResponse`), where `private` comes from GitHub's repository response. Failures have `{ok: false, code, error}`: HTTP 400 `no_token`, HTTP 403 `cannot_create` (403/404 on creation; the token needs repository-creation permission), or HTTP 502 `failed`. Publishing uses the git destination path and merges when the account can push; HTTPS GitHub source reads retry authentication refusals with the configured token to share private skills between installations.

**The catalog** shares git clones by repository/ref and snapshots by repository/ref/path. Official/recommended mean exact endorsement through a hub row with `part: "official"`, or the bundled release; other hub repositories keep their entries without these marks. It answers from local data; a stale cache or `?refresh=1` starts a background fetch, and a source never read shows `state: "reading"`:
- `GET /api/skill-catalog` (`SkillCatalogResponse`): `items` (`SkillCatalogItem`, id `<source_id>/<name>`) with `official`, `recommended`, `category`, `owner`, `updated_at` (when the release last changed in its source: a git source's newest commit touching it, a registry's update time; absent when unknown), `signals` (`null` means no data, never zero), `withdrawn` and `installed` (`installed`, or `name_taken` by another source's skill); `categories`; `sources` with their read state.
- `GET /api/skill-catalog/search?q=&source=` (`SkillCatalogSearchResponse`) adds registry (ClawHub) results to a text search; `registry_error` reports an unreachable registry.
- `GET /api/skill-catalog/item/{source_id}/{name}` (`SkillCatalogItemDetailResponse`): the skill's own text, what it can use, versions and links; 404 for an unknown item.

**Install, update, uninstall** (the lifecycle lane, §11):
- `POST /api/skill-catalog/install` (`SkillCatalogInstallRequest`: `id`, optional `replace`, `turn_on_when_ready` default true) answers at once with the started `job_id` and the install's `basis` (`SkillCatalogInstallResponse`). Refusals answer `{ok: false, code, error}`: 404 `not_found`; 409 `withdrawn`, `name_taken` (with `occupant`; `replace: true` replaces it and keeps the replaced copy), `update_instead`, `local_source` or `no_release`. The copy must match the release digest, or nothing lands. A `review` install starts its review after landing and turns on only when the review leaves no blockers; a `trusted` one turns on once it is ready.
- `POST /api/skills/{skill}/update` (`SkillUpdateRequest`: `mode` `replace`, `keep` or `merge`; `digest`, the candidate shown; `local_digest`, the local revision shown; optional boolean `skip_review`). `replace` answers the started `job_id` and keeps the previous copy; `keep` records the skipped release; `merge` asks Ouroboros, through the owner's door like Repair, to merge the release into the local changes (three-way only when the original release can be fetched again: `base_available`). `skip_review: true` is the owner's Update without review, valid only for replace: exact-release `owner_skip` acceptance with preflight. A review-basis update stays off during review; if the new bytes cannot run when review ends, the old payload, state and record return with `update_rejected`. Startup restores interrupted updates; an owner Skip during review keeps the new bytes. A later update clears rejection. A different candidate answers 409 `candidate_changed` with the new `candidate`, a changed local copy `local_changed`; a withdrawn release or a skill that left its source cannot be updated.
- `POST /api/skills/{skill}/uninstall` (`SkillUninstallResponse`) works for every source; grants and the publication receipt survive. A repeat answers 404.
- `GET /api/extensions` rows carry `install` (source, Official, release, `basis`, `modified`, `update_available` as `{version, digest}` or `null`, `withdrawn`, `left_source`, `turn_on_when_ready`, `installed_by`, and the recorded `reason` and `task_id` when there are any) and `category`, from the cached catalogs without network. Optional `install.check` is `{by, notes, at}` for a recorded `self_check`; `pending_update` is `{version, digest}`; `update_rejected` is `{version, digest, reason, at}`, with reason `review_blockers`, `review_unfinished`, `preflight_failed` or `landing_failed` (the update's files or dependencies could not be installed; the previous version was restored). An update is a different release digest, never a different version string. `install.official` requires unchanged release bytes and a matching current endorsement; with no snapshot only the stored flag for unchanged bytes counts. This also restores the mark for unchanged legacy installs. A hub entry may list `withdrawn_releases` by digest; holders receive `install.withdrawn: {reason, version}`, while entry-level withdrawal is unchanged.

**ClawHub** (a third-party registry, also joined into catalog search) keeps its routes:
- `GET /api/marketplace/clawhub/search`: with `q`, a text search; without it, a registry browse paged by `cursor` and `next_cursor`. `official=1` keeps official skills only.
- `GET .../info/{slug}` and `.../preview/{slug}` read registry metadata only; nothing is downloaded.
- `POST .../install` (`MarketplaceInstallRequest`: `slug`, optional `version`, `auto_review` default true, `overwrite` default false). An existing skill is replaced only with `overwrite`. `ok: true` means the skill is installed, even when review found blockers (`review_status: "blockers"`); such a skill does not run until the owner acts. A dependency failure gives `ok: false` and rolls the install back.
- `POST .../update/{name}` (optional `version`) always reviews and overwrites. `POST .../uninstall/{name}`; a repeat answers 400.
- `GET .../installed`. Registry errors answer `{error, code}` with 400, 404, 429 or 502.

The retired hub catalog/preview/install/update/uninstall/publication-clear routes and fields use these successors ([retirement record](architecture/11-frozen-contracts-v1.md#114-recent-abi-retirements)); the ABI version is unchanged. `GET /api/marketplace/ouroboroshub/installed` remains frozen, returning `{count, skills}` (`MarketplaceInstalledResponse`) for hub-source installs and the legacy hub bucket. CLI `marketplace ouroboroshub` aliases `sources`; `install <name>` resolves a public hub offer, while `<source>/<name>` selects a catalog id.

**Publication receipt:** `POST /api/skills/{skill}/publication/clear` forgets the local receipt of a publication (`expected_published` is the receipt as shown; a repeat answers `publication_cleared: false`). The pull request stays on GitHub.

**Starter skills:** `GET /api/onboarding/starter-skills` (`StarterSkillsResponse`) groups built-in, Official and every configured starter id by category, ticked as the starter list says. `OUROBOROS_STARTER_SKILLS` absent or `""` uses defaults; `[]` or `"[]"` selects none. `OUROBOROS_STARTER_SKILLS_TURN_ON` (default true) controls first enablement. `POST /api/onboarding/complete` accepts `starter_skills` (catalog ids) and `skill_source_trust` (`{hub row id: "trusted" | "review"}`, which also confirms those rows) in its one write; the ticked skills install once in the background afterwards.

## 13. Agent accounts

Connected coding subscriptions use Claudexor. Its API v2 objects (`job`, maintenance operation, reset receipt, quota snapshots) pass through; Node/npm preparation is host-owned. Pin: `ouroboros/claudexor_runtime_pin.json`; source: https://github.com/razzant/claudexor. Read published shape keys; treat the rest as data.

**Status:**
- `GET /api/claudexor/status` (`ClaudexorStatusResponse`) is passive: it never starts the engine. `reads.catalog`, `reads.accounts` and `reads.quota` are each `ok` (authoritative, even when empty), `not_read` (never asked) or `failed`. A facet that was not read freshly serves its last value with `facets.<name>.stale: true`. Never read an empty list without `ok` as "no accounts".
- `?view=quota` answers the passive quota view (`ClaudexorQuotaResponse`), described in [`PASSIVE_QUOTA_READ.md`](PASSIVE_QUOTA_READ.md). Check `view: "quota"`: an older core ignores the parameter.
- `POST /api/claudexor/wake` starts the engine when needed and answers a fresh status.

**Signing in an account:**
1. `POST /api/claudexor/login` (`ClaudexorLoginRequest`: `harness`, optional `profile_id`, `transport`, `login_flow`) starts a sign-in job and answers `{job, job_id, ...}`. It can start the engine. It is not idempotent: every call may start a new job. After a lost answer there is no job id to poll; start the sign-in again only when the person asks. A `400` with `code: "credential_profile_required"` asks for a named account (`profile_id`).
2. Poll `GET /api/claudexor/login/{job_id}` every few seconds: it answers a snapshot `{job, cursor, sequence, deviceCode?}`. `deviceCode` (`verificationUrl`, `userCode`) is what the person enters at the vendor. In the engine release the core pins, `job.state` is `queued`, `running` or `waiting_for_input` until it is `succeeded`, `failed`, `cancelled`, `timed_out`, `interrupted_unknown` or `not_supported`.
3. `POST .../login/{job_id}/input` sends one answer the job asked for (`{value}`); `DELETE .../login/{job_id}` cancels. A finished job whose reason is `termination_unconfirmed` is not known to have stopped the vendor's process. `POST .../login/{job_id}/reconcile` checks again; a `409` with code `setup_termination_unconfirmed` asks you to retry it. The builtin client treats such a job as released only once `job.terminationReconciliation.status` is `empty`.
4. Job errors answer `ClaudexorLoginJobProblem` with the engine's `code` and `required_actions`; 404 or 410 means the job is gone.

**Accounts:**
- `PATCH /api/claudexor/credential-profiles/{harness}/{profile_id}` with `{enabled}` turns an account on or off; read status again for the effect.
- `DELETE` on the same path forgets the account's binding. The receipt says what happened to the vendor's own sign-in (`vendorCredentialDisposition`). Every engine refusal on these two answers 503.

**Quota and resets:**
- `POST /api/claudexor/quota/refresh` refreshes every account, or one with `{target: {harness, profile_id}}`. It is passive and safe to repeat.
- `POST /api/claudexor/account-resets` (`ClaudexorAccountResetRequest`) redeems one reset offer. It needs an `Idempotency-Key` header. Keep the key and the body until the reset settles: the same key and body replay the same receipt. Check that the receipt's `request` names your target and offer. Poll `GET /api/claudexor/account-resets/{id}` while `state` is `running`.
- After a lost answer the outcome is unknown: send the same key and body again, or read the receipt by id.

**Agent tool maintenance:**
- `GET /api/claudexor/maintenance/harnesses` passively lists installed, tested and available tools and current operations (rejoin after reload).
- `POST /api/claudexor/maintenance/operations` (`{harness, target}`, `Idempotency-Key` header) installs, updates or rolls back a tool: `202` returns its operation. It can start the engine and prepare Node/npm. `409 maintenance_restart_required` asks for chat-header Restart, then repeat Update; `409/503 maintenance_preparation_failed` reports failure. Optional inspection failure still permits POST.
- `409 maintenance_already_active`: follow `context.operationId`. Poll `GET .../operations/{id}` while `state` is `queued`/`running` or `termination` is `unconfirmed`. `POST .../operations/{id}/cancel` requests cancellation, without proving termination.
- `ClaudexorMaintenanceProblem.error` holds engine or host `{code, message, context}`. Preparation's `context.acceptance=not_submitted, scope=this_attempt` cannot resolve an earlier `503 acceptance=unknown`: keep and resend that same key/body.

**The review pool.** `GET /api/review-pool` lists the agent rows that may review work and what one review costs on each. A broken configuration answers a non-empty `config_error` with an empty pool. A readable catalog with no row marked Reviewer also answers an empty pool: the Main model then reviews alone.

## 14. Models, local model and MCP

**The model catalog.** `GET /api/model-catalog` (`ModelCatalogResponse`) lists selectable models: every configured provider's, or with `source_id` one agent account's. `value` is what a model setting stores. A provider that failed is an `errors` row and the read is still 200: missing models are unknown, not proof of none. `model_sources` and `account_catalogs` are the engine's objects, passed through.

**The local model:**
- `POST /api/local-model/start` (`LocalModelStartRequest`) answers `{status: "starting"}` after the model is downloaded; the request stays open meanwhile. It is not idempotent. Then poll `GET /api/local-model/status` until `status` is `ready` or `error` (`offline`, `downloading` and `loading` come before).
- `412` with `error: "runtime_missing"` means the local runtime is not installed: `POST /api/local-model/install-runtime`, then poll status until `runtime_status` is `install_ok` or `install_error`.
- `POST /api/local-model/test` runs two real completions; a failed probe is still 200. `POST /api/local-model/stop` stops it (and an install in progress) and is safe to repeat.

**MCP servers:**
- `GET /api/mcp/status` lists the configured servers and their last listed tools; it makes no network call.
- `POST /api/mcp/refresh` lists one server's tools (`{server_id}`) or every server's (`{}`); it is safe to repeat.
- `POST /api/mcp/test` connects once to a draft server (`{server}`, with `server_id` to fill masked secrets from that saved server) and saves nothing. A failed test is 200 with `ok: false`.
- `POST /api/mcp/import/preview` requires `{text, servers}` and turns pasted configuration into draft rows without saving. `servers` is the editor's current draft matching metadata; send `[]` for an empty editor. An invalid request shape answers `400`; syntactically invalid pasted configuration answers `200` with `ok:false`, `error` and no entries. HTTP success alone is not a usable preview.

## 15. Updates and core control

**Checking for updates** (`UpdateStatusResponse`):
- `GET /api/update/status` makes no network call:
  - `running_sha`: the commit this core started from; `current_sha`: the checkout's;
  - `available` and `safe_to_apply`;
  - `update_tx`: the update transaction in progress (`{active: false}` when none);
  - `update_progress`: the last admitted update attempt (`{}` only when absent; unreadable state is `outcome:unknown`). `operation_id` and `generation` identify it; `current_generation`, `stage`, `active`, `result`, `outcome` and `concluded_by` separate progress from conclusion. `base_sha`, `target_sha`, `keep_branch` and `stash_note` retain exact source/local-work facts when known. `outcome` is applied/not_applied/rolled_back/failed/unknown, empty while unresolved. It is observation, never recovery authority; a pre-admission refusal remains its HTTP answer;
  - `letter`: the written description of the update, when there is one;
  - `check_ok`: null on status.
- `POST /api/update/check` fetches first, may write the update letter and broadcasts `update_progress_changed`.
- `POST /api/update/preflight` returns a `merge_plan` with `base_sha` and the target.

**Applying an update.** Send `POST /api/update/apply` (`UpdateApplyRequest`) with `strategy`, `expected_base_sha` and `expected_target_sha` from the preflight:
- **Release moved:** 409 `release_moved` means the release changed since the preflight. Run the preflight again.
- **Replacement not applied:** 409 `reason: replace_not_applied`, `abandoned: true` means Git refused the first tree-changing command because its index lock was held. The transaction is disarmed without rollback; `rescue` and `keep_branch` disclose retained preparation artifacts. The lock is untouched. Diagnose it before another deliberate attempt; age alone never authorizes removal. Other partial or unknown failures keep fenced recovery and their error.
- **What `200` means:** inspect `status`, not HTTP success alone. `ok` with `restarting:true` means a restart was requested, not adoption. `assisted_started` means a supervised task owns resolution. `manual` returns a plan without applying or restarting. `restart_required` means the restart that finishes the apply could not be requested (a store-less replace has not moved the checkout yet); show its error and offer the ordinary restart control.
- **Watching it:**
  - `update_progress_changed` frames mean "read the status again"; stages are live; the saved last admitted attempt and conclusion survive restart. `retention_unconfirmed:true` reports a failed or refused save without changing the update result; a never-saved attempt cannot be recovered after process loss. An update check does not erase saved results. The indicator may hide a dismissed terminal attempt locally; the panel retains it until the next attempt.
  - After reconnecting, confirm the executing generation (`running_sha`) against the landed transaction/`merge_commit` identity reported by the update, not blindly against the preflight's official `target_sha`. Preserving local commits can create a distinct two-parent merge. Until the landed identity and completed startup are known, adoption remains unconfirmed.
  - A core whose startup check fails can roll back and restart again. An assisted-resolution rollback completed in the same generation may have no retained conclusion and display an unconfirmed outcome.
  - `update_status_ready` arrives after a fresh start.
- **Retrying:** there is no idempotency key. A non-manual request can answer 409 while an update is active or its expected base/target no longer matches; `manual` may return another plan. When an outcome is unknown, read status and transaction facts before deciding—never infer failure or resend from a dropped connection.

**Rollback.** `POST /api/git/rollback {target}` stops running work, switches to the given commit or tag and restarts. It has no checks and is not safe to repeat. Ask the person first; afterwards confirm `running_sha`. 409 `reason: restore_not_applied` means Git refused the reset at its held index lock, so the reset changed no checkout file, index entry or ref; the lock, and the keep branch and rescue snapshot made before the reset, are retained. No restart is requested; writers are readmitted through the aborted-update path, which keeps admission closed while an update transaction is active or the stopped work's handoff is unconfirmed.

**Promote.** `POST /api/git/promote` moves the local stable branch to the current commit (nothing is pushed).

**Reset.** `POST /api/reset` deletes the data folders and the settings, then restarts. It is irreversible: ask the person first, and never resend after a dropped connection.

**Commands, restart and Panic.** `POST /api/command` (`CommandRequest`) or a WebSocket `command` frame sends a slash command (`/restart`, `/panic`, `/review`, `/evolve`, `/bg`, `/status`); any other text becomes an owner message in Main.
- **What `200` means:** `{status: "ok"}` means queued, except Panic acknowledges its immediate emergency latch, not completed shutdown. The same `visible_task_id` within 5 seconds answers `{ok, deduped, task_id}`. Either `/restart` or `/panic` can end the connection before or after the answer, so a dropped connection is not a failure.
- **Panic:** stops owned work and exits 99; the launcher never restarts it. Never queue, reconnect-flush or offer resend, including chat frames. Desktop uses `pywebview.api.panic_stop()` first, also while the socket reconnects (no veil covers Panic). Panic bypasses confirmations, composer saves/uploads and Telegram command-mode translation. Intercepted typed Panic creates no chat bubble. Only bridge `{status: "stopping"}` or `panic_accepted` clears unchanged sent command text; unknown/refused delivery keeps it. Newer edits, replies and attachments stay. If the room is closed and reopened before acceptance, the replacement field can retain accepted text even after its saved draft is cleared; this remount edge remains unresolved. Browser uses its captured open socket with no buffered output; `panic_accepted` means emergency latch, not exit. Missing acceptance says “Panic was not confirmed”: desktop points to its tray/menu bar, browser to the host computer. No clock agreement or preflight request is required; transmitted bytes can arrive later on that connection. `client_surface` is diagnostic, never owner identity. Unavailable HTTP Panic returns `503:ErrorResponse`; WS sends a declared `log` warning. Unbound external owners never enter ordinary admission. Telegram sends once without an age expiry; its whole-second message date is compared with the host clock at plugin import. Earlier dates are refused, with previous outcome unknown. This assumes aligned clocks: a host ahead rejects fresh Panic until Telegram reaches the import second; a host behind can accept retained commands. Old/fresh messages in the boundary second are indistinguishable and accepted.
- **Restart:** a restarted core comes back on its own; reconnect and resynchronize (§5).

**Spend.** `GET /api/cost-breakdown` reports spend from the physical-attempt ledger. A `503` means the ledger cannot be read: spend is unknown, not zero.

**Versions and diagnostics:**
- `GET /api/git/log` lists recent commits and tags (`GitLogResponse`), the choices for a rollback.
- `GET /api/logs/{name}` tails one log (`chat`, `progress`, `events`, `tools` or `supervisor`; `limit` 1 to 2000, default 100; `task_id` filters). An unknown name answers 404 with the allowed names. `chat` and `progress` are read from the dialogue store: `_line` is the row's durable position among every stored object of its stream, and while the archive import has not settled those counts the endpoint answers 503 `reason_code: migrating` (retry shortly), never a guessed number; the file logs' `_line` is a position within the read window.
- `GET /api/storage` (`StorageResponse`) reads the last finished count of everything Ouroboros owns, a running count's progress, fresh free space and the Temporary space tray, without measuring; `?count=<finished_at>` omits a count the client already holds. `POST /api/storage/measure` starts one complete count in a child process or joins the running one; `POST /api/storage/measure/cancel` kills it and keeps the saved count (202 either way, the same envelope with `started`).
- **Temporary space.** `POST /api/storage/free/preview` lists, without writes, what the app's own retirers would remove now (finished tasks' work folders, working copies of finished work) under a plan id; a second preview joins the running one. `POST /api/storage/free` with `{plan, groups}` (`StorageFreeRequest`) frees exactly the previewed items of those groups, for good: ask the person first, and never resend after a dropped connection (the plan id makes a repeat harmless: a replaced or missing plan answers `started: false` with `reason`). `POST /api/storage/free/cancel` stops the listing (nothing is kept) or the run (what was freed stays freed). All answer 202 with `StorageResponse`; `free.run` separates removed, still deleting, stuck and kept items.
- `GET /api/evolution-data` (`EvolutionDataResponse`) answers at once with the saved version history (`saved_at`, `versions`, this installation's `install` journals) and `refresh` (`idle`, `running` or `failed`; `unmeasured` versions with the first reason): new tags or `force=1` start one background re-measurement, which also retries unmeasured versions, and a failure keeps the saved copy. The earlier `points` (one per tag name; null where unmeasured, memory fields null) and `cached` stay for older clients. Ask again while it runs; a value nobody measured is `null`, never zero.

## 16. Errors and unknown outcomes

- **Error bodies** are `ErrorResponse`: `{error}` and, when present, `code` or `reason_code`. A route may add facts beside them (for example `schema_errors`, `task_id` or a skill's current facts). An unexpected failure can also answer `500` as plain text. Statuses beyond the ones a route lists are those `500`s, the auth gate's `401`, and the router's own `404` and `405` for an unknown path or method. Agent-engine routes may also pass the engine's own status through (§13).
- **Object errors.** Maintenance (`ClaudexorMaintenanceProblem`) returns an engine or host problem in `error`. Accept both strings and objects; show `error.message` for objects.
- **When an outcome is unknown, never resend a change automatically.** The unknown-outcome answers:
  - `POST /api/tasks` → `status: "unconfirmed"` (resending the same `task_id` is safe; a new id is not);
  - settings writers → `saved: null`;
  - routing decisions → `dispatch_unconfirmed`;
  - `POST /api/tasks/{id}/continue` → `continuation_unconfirmed`;
  - maintenance → `503` with `context.acceptance: "unknown"`, and account resets after a lost answer (resend the same key and body);
  - a dropped connection during `/restart`, `/panic`, rollback, reset, a lifecycle-lane job or a local-model start.

  Read the state that answers the question, then let the person decide.
- **The idempotency keys:**

  | Operation | Key |
  |---|---|
  | chat messages | `client_message_id` |
  | tasks | `task_id` |
  | Stop | `stop_action_id` |
  | Pause and Hurry | `request_id` |
  | Continue | `action_nonce` |
  | decisions | `request_id` |
  | account resets and agent tool maintenance | the `Idempotency-Key` header |

  Keep each key with its original request. Repeating the same operation is safe only under that operation's documented replay rule; do not infer a universal changed-payload refusal. In particular, quiz decisions replay the recorded answer for the same `request_id`, even if later input differs (§7). A different request id does not change an already answered quiz. `expected_content_hash` and `expected_state_fingerprint` are revision guards, not idempotency keys.
- **Not idempotent:** sign-in jobs, local-model start, task resume, rollback, reset and update apply.

## 17. Coverage and known gaps

**Published:** every HTTP route the core indexes (all `typed: true`) and every built-in WebSocket message type, each with its shapes; the history row and the live/history merge rule; the module widget host protocol. Structural checks cover the indexed routes and builtin client's calls; consumer tests exercise selected actual responses. A declared shape is not proof that every status branch has been exercised.

**Known gaps:**
- **Engine objects.** The agent engine's objects (§13, §14) are published only to the keys the core itself reads. Their full schemas are the engine's.
- **Open documents.** `POST /api/settings` and `POST /api/ui/preferences` take documents of setting keys (published as plain `json`); `GET /api/settings` and the setup vocabulary name those keys. `log` frame rows, `chat_annotation`, `review_group`, `quiz.options` and the review projections are open objects.
- **Declarative widgets** are drawn by the host from the component vocabulary in `docs/CREATING_SKILLS.md`; there is no separate host protocol for them.
- **Unsupported topologies:** a different origin, a sub-path mount, several cores behind one interface, and more than one person.

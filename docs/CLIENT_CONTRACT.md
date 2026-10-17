# Ouroboros client contract

This document is for anyone writing a complete client of an Ouroboros core: a replacement interface, a script, or a separately hosted copy of the builtin one. It covers setup, sign-in, conversation and history, projects and tasks, questions, Stop, settings, files, skills and widgets, the marketplaces, agent accounts, local models, MCP, updates and Panic. A client needs only this document, [`WIDGET_HOST_PROTOCOL.md`](WIDGET_HOST_PROTOCOL.md) for module widgets, and the core's HTTP and WebSocket endpoints. Nothing here requires reading Ouroboros source or importing its packages.

There are two sources of truth, and each has its own job:

- **`GET /api/contract`** is the machine-readable part. It lists every route, its path parameters, the query parameters and request headers it reads, its request and response bodies by HTTP status, every WebSocket message type, and a JSON Schema for every named shape. Every route publishes its shapes (`typed: true`). The core derives this document from its route and type declarations. Tests check structural coverage across the catalogue and actual responses for selected handlers, producers and consumers, not every response branch. Fetch it from the core you talk to. A copy taken from another version can be wrong for this one.
- **This document** explains how to use those shapes: which call comes first, what a status means for whether something happened, which value identifies an operation, and when a retry is safe.

Where the two seem to disagree, the live `GET /api/contract` is right about shapes and this document is right about meaning. Report the disagreement as a defect. §17 lists what is still described only partly.

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
- **Activity:** global root census `active_chat_activities`: `last_line`/`last_line_at`, `tool_calls`, `detached`, `project_id`/`project_name`; legacy `active_direct_turns` retained.
- **Projects:** `projects` (a compact feed that may be capped) and `project_chat_ids` (the complete set).
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
- **Core to client, stored as history rows:** `chat` (a message or a progress line), `photo`, `video`, `document`, `links` and `quiz`.
- **Core to client, signals only:**
  - `quiz_state` updates a question card; history shows the merged state.
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
- **Deliveries:** `msg_type` is `document`, `photo`, `video`, `links` or `quiz`, with `filename`, `mime`, `download_url`, `caption` and `size_bytes`; `actions` and `title` for links; `quiz` for a question (`quiz.quiz_id`, `quiz.state`, the recorded answer).
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
- **Around one task.** `around=task:<task_id>` (never together with `cursor`) reads the page around that task's earliest row of the thread, wherever it lies: about a quarter of each quota before it and the rest after it. `anchor` says what was found: `{"state": "found", "task_id", "history_id"}` with an ordinary page (`page_cursor` re-reads it, `next_cursor` continues older; it is not the newest page, so it has no `after_cursor`), or `absent` (the thread's saved history holds no row of it a client could show) or `moved` (its rows belong to another thread), both with no rows and no cursors. A source that cannot be read answers `503 history_source_unavailable`: unknown, never absent. A stream holding none of the task's rows carries none of its context: the page names it in `window.truncated_by` (`chat_around_unpositioned` or `progress_around_unpositioned`), and on that page and the older pages it continues to, that stream's `coverage` span is null.
- **Errors.** An error answer keeps the same shape, with no rows and `has_more` true, and echoes the cursor you sent as `next_cursor`, so the same request can be repeated:
  - `400 history_cursor_invalid`, or `history_around_invalid` for an `around` that is not `task:<id>` or comes with a cursor;
  - `409 history_view_changed`: which rows belong to the thread changed (its projects or task bindings); start again without a cursor;
  - `409 history_delta_reset`: since an `after_cursor`, more arrived than one read replays, a stored row could not be read, owner-message annotations changed, a task the window shows settled or changed after it settled, or a shown skill operation ended; read the newest page again (in a busy room this is a common answer, not a rare one);
  - `503 history_source_unavailable` or `migrating`.
- **Incomplete windows.** `window.complete` is true only when nothing bounded the read; `window.truncated_by` names each cause (`quota`, `page`, `lineage_cap`, or a stream gap such as `chat_source_unavailable`). Never treat an incomplete window as an empty conversation. `coverage` reports which stored bytes were delivered, for clients that page; it says nothing about chronology.

### Live calls

A room holds at most one owner call; its audio travels on its own socket, never on `/ws`. `GET /api/contract` describes it under `call`.

- **Opening.** `POST /api/call/open` with `{chat_id, client_message_id?, model?, voice?}` answers `{call_id, ws_path}`. Calls use Google Gemini Live. When the room already has a call, the answer adds `joined: true` and this surface joins that call, which keeps its own model and voice. A refusal is `{ok: false, code, error}`, where `error` is a sentence for the owner; a missing Google AI key is refused here (`missing_credential_google`), before any turn. The defaults come from Settings (Models → Live calls).
- **Typed words during a call.** A message typed in the room still starts its own ordinary turn. The call learns of it as context and speaks about it when it next talks.
- **The socket.** Open `ws_path` with the same authentication as `/ws`. The first text frame is `hello` `{type: "hello", protocol: 1, role: "owner", mic, speaker, rate, label}`, where `rate` is the device's sample rate; the core resamples.
- **Audio frames.** Binary frames carry a 16-byte little-endian header (magic `OC`, version 1, kind 1 = to the core or 2 = from the core, epoch u16, sequence u32, rate u32, stream u16) and then PCM16 mono.
- **Your control frames.** `mute {muted}`, `played {item_id, played_ms}` (milliseconds of that item actually rendered) and `bye {reason}`.
- **The core's control frames.** `state {state, reason, engine, phase?, detail?, action?}`, `activity {activity, label}`, `caption {speaker, text, final, item_id}` and `flush {epoch}`. A call is `preparing` (`phase: "memory"`) while its turn reads memory, `connecting` only while its engine opens, then `live`. A `state` reason is a code (`owner_stop`, `max_minutes`, `missing_credential`, `provider_closed`, `invalid_request`, `transport_lost:<detail>`, …): a client shows its own words for it, with `detail` and the one next step `action` (`settings` or `retry`). For a provider's close or error (`provider_closed`, `invalid_request`, `provider_error`, `rate_limit`) `detail` is the provider's own sentence or empty, never the core's description; for the core's own refusals it is the core's sentence. `activity` is `working` with the tools' own names while a tool batch runs, then `idle`. A `bye` before the engine opens ends the call without opening one.
- **Binding audio to an item.** Output frames carry no item id. An `ouroboros` caption, possibly empty, names the item before its first frame. After `flush`, drop every frame of an older epoch.
- **History.** The owner's speech is stored as owner rows with `source: "call"`. Speech heard while meeting audio was in the call is a System row with `system_type: "call_heard"`. Ouroboros's speech is a narration progress row whose meta carries `spoken {call_id, item_id, status, played_ms}`, where status is `generated`, `played` or `interrupted`.

## 6. Projects and tasks

**Projects:**
- **List:** `GET /api/projects` (`ProjectListResponse`): active and deleting projects, at most 200, most recently active first.
- **Create:** `POST /api/projects` (`ProjectCreateRequest` → `ProjectCreateResponse`). One source at most: `path` (attach a folder), `git_url` (clone) or `with_workspace`. Creating an existing `id` with no source changes nothing and answers the existing row with `created: false` (a flag optional on other responses, whose absence proves neither new creation nor failure); with a source it answers 409. A clone that needs credentials answers 401 with `error_code: "auth_required"` (not the network gate).
- **Turn a task into a project:** `POST /api/projects/from-task` (`ProjectFromTaskRequest`) takes no idempotency key: the server derives `handoff_id` from the project, the task and its origin message, so repeating the call for the same task returns the same `handoff_id`, and with the default project id (`task-<task_id>`) a repeat also adopts the project the task's work already has (`adopted: true`). A task bound to another project answers 409. Without `name`, naming may wait for one model call.
- **Rename and delete:** `POST /api/projects/{id}/update` renames (`{name}`, at most 80 characters). `POST /api/projects/{id}/delete` starts the deletion (a repeat is harmless).
- **Changes from elsewhere:** `projects_changed` announces new projects: add their `chat_id` to the threads you listen to before reading the list again.
- **Choosing a folder to attach:** `GET /api/fs/dirs?path=` lists directories under the home folder only (at most 500, `truncated` beyond that).

**Starting a task.** `POST /api/tasks` (`TaskCreateRequest`). The `task_id` you choose, or the one returned, is the task's identity.
- **Duplicate id:** an existing id answers 409, so a retry with the same id cannot start the task twice.
- **Outcome unknown:** `503` with `status: "unconfirmed"` means the task may or may not be queued; `GET /api/tasks/{task_id}` can answer 404 while it is still queued, so a 404 is no proof it will not run. The same `task_id` sent again answers 409 and never starts a second copy. Never resend under a new id; watch the task and let the person decide.
- **Other 503s:** they come before the task is scheduled, and the attempt is cleaned up: nothing runs, `GET /api/tasks/{task_id}` answers 404, and the same `task_id` can be sent again.

**Reading tasks:** `GET /api/tasks` (filters `status`, `limit`, `queue_only`) and `GET /api/tasks/{task_id}`. Records identify `task_id`, `status`, `ts` and `updated_at`; `result`, artifacts and evidence appear when available. List rows omit bulky evidence; read the detail for the result. `status` accepts a comma-separated filter; `limit` defaults to 50, positive values cap at 500, and `0` requests all rows. `queue_only=1` returns `{tasks:[], queue}` without history.

A Project completion frame/history row may carry `completion_notice`: `{neutral: bool, detail: string, phase: string, salvage?: string}`, a compact transport view, not a verdict: neutral means technical caveats stay in task details, never independently approved success. Full outcome/answer fields remain authoritative; `salvage` preserves the existing intermediate excerpt. Older rows may omit it; replay can derive the missing metadata from the canonical result without rewriting retained text, and with no recoverable source transports retain the legacy text rather than assuming it is status-only.

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
- **List:** `GET /api/schedules` (`ScheduledTasksResponse`). Each row's `status` is `active`, `disabled`, `suppressed`, `consumed`, `held`, `waiting` or `delete_pending`. An unreadable table answers 503, never an empty list.
- **Create or edit:** `POST /api/schedules` (`ScheduleUpsertRequest`). An existing `id` edits that schedule; without an `id` a repeat creates another one. `ok` is true only when the audit record landed (`schedule.audit` is `recorded`). A 409 means nothing was written: the schedule table or its audit could not be used safely.
- **Disable, delete, restore:** `POST /api/schedules/{id}/action` (`ScheduleActionRequest`: `action` and a required `reason`); `DELETE /api/schedules/{id}` also deletes. A refusal is also 200, with `ok: false`: `changed` is the durable fact and `status` says what happened. An action governs future runs only; `running_or_queued` reports a run it already admitted (`null` when unknown).

**Downloading a result file:**
- **Where:** `GET /api/tasks/{id}/artifacts/{name}` (optionally `relpath`, `archive` or `source`).
- **Partial downloads:** byte ranges are supported. A `206` response is a byte slice, possibly an incomplete JSON document even when the original file was JSON, so do not parse it as a complete one.
- **Checking the bytes:** the `x-ouroboros-artifact-sha256` header gives the checksum when the core verified the bytes.
- **Changed after recording:** `409 artifact_identity_changed` means the file changed after it was recorded.

## 7. Questions and decisions

**How a question arrives.** A task asks the owner through a `quiz` frame (`QuizOutbound`), in history a `msg_type: "quiz"` row, updated by `quiz_state` frames. Its state is `open`, `answered`, `closed` (explicitly closed without answering), `expired_terminal` (the task finished; an answer can still be recorded) or `superseded`.

**Answering.** Send `POST /api/decisions` with `DecisionRequest` and a `decision_id` of `quiz:<task_id>:<quiz_id>`:
- **What to send:** `option_index`, `comment` (nonempty when alone), or both; `option_index`, `answered_index` and `recommended_index` are zero-based positions in `options`. A negative/non-integer index answers `400 option_index_invalid`, one beyond an open card's options `400` with `error:"option_out_of_range"`, and neither index nor comment `400 option_index_required`. An already closed card can instead return its recorded 409 state; an absent quiz is `404 quiz_not_found`.
- **Safe retries:** the same `request_id` returns the already recorded answer with `duplicate: true`, even if the retry's payload differs: display that answer, not the retry's input (a successful replay accepts no changed answer). It also lets the existing delivery-recovery path finish after a partial failure.
- **First answer wins.** A different answer to an already answered question answers `409` with the recorded state.
- **Late answers.** An answer after the task ended is still recorded (`answered_after_terminal`) and goes to the card's `publication_chat_id`, recorded before the card was first shown and never changed, not to its ask-time chat: a question asked in the hidden partition (0) and shown in Main is answered in Main (1), and a converted task keeps that address while history shows its card in the current resolved room. `forwarded:true` confirms the answer entered the ordinary owner-message path there; `forwarded:false` is not delivery: `hidden_chat` and `a2a_chat` have no owner turn, `publication_address_unknown` (an older card without a proven address) stays recorded and owed with no guessed room, and `late_answer_delivery_unknown` means proof of dispatch was lost and nothing is resent. Show this returned outcome; never infer delivery from the room the card appears in.
- **Recorded but not delivered.** `503 quiz_history_write_failed`, `mailbox_write_failed` or `late_answer_not_delivered` mean the answer is recorded but not yet delivered: retry with the same `request_id`.

**Closing without answering.** For a root task's quiz, send the same endpoint and `decision_id` with a `request_id`, `action: "close"` and an optional `reason` of at most 2000 characters, without `option_index` or `comment`; a blank or omitted owner reason records `Closed by the owner without an answer.`, and a non-string or longer one answers `400 close_reason_invalid`. Closure shares the first-winner rule with answers; show the returned `state`, `closed_reason` and `question_delivery`: a recorded closure is no proof its author received it.

**Removing unknown late delivery from current decisions.** For an answered quiz with unknown late-answer delivery, send `action: "dismiss"` with `decision_id` and `request_id` only (no answer, comment or reason). It removes only the outstanding delivery item, neither resending the answer nor marking it delivered; the recorded answer and unknown outcome stay in history. Another, ineligible state is refused; read the returned facts.

**Other decisions.** Model-wait choices (`model_wait:<task_id>:<wait_id>`) and persisted routing choices (`routing:<client_message_id>:<token>`; no new routing pickers are produced) use the same endpoint and idempotency rule.

## 8. Stop, pause, continue, resume

**Stop.** `POST /api/tasks/{id}/cancel` takes an optional `TaskCancelRequest`: `cascade`, `stop_policy` and `stop_action_id`.
- **Safe retries:** `stop_action_id` names one Stop action. While that action is pending, an identical request joins it and reusing the id with different arguments answers 409. An immediate Stop records the stop first and answers after it drove the teardown; once the task is no longer active, even an identical retry can answer 404. Read task detail to distinguish the recorded outcome.
- **Stopped:** `200` means the teardown settled (with `cascade: true` when the whole tree was stopped).
- **Recorded, still stopping:** `202` with `cancel_state: "pending"` means the stop is recorded and the supervisor finishes it: a graceful stop under way, or an immediate one whose teardown is not confirmed yet (`reason_code: "teardown_unconfirmed"`, with the stop's `request_id`). No retry is needed: read `GET /api/tasks/{id}` (`cancel_state`) until it settles; an immediate Stop still hardens a graceful one.
- **Nothing to stop:** `404` means the task already finished (completion wins), the id names no task, or a repeated Stop already finished. A task that was created but never started is stopped, not 404.
- **Nothing recorded:** `503 cancel_intent_write_failed` means nothing was recorded or cancelled, so retry; `503 cancel_intent_projection_corrupt` cannot succeed until the projection file is repaired.
- **When to offer Stop:** chat frames carry `cancelable` to say when Stop applies. Do not offer it otherwise.

`stop_policy` is `immediate` (also the omitted/empty default) or `finalize_then_cancel`. A graceful request cannot soften a previously recorded immediate Stop. Task detail has `cancel_state:"pending"` while cancellation is owed, otherwise the field is absent; there is no public `cancel_state:"settled"`.

**Partial work.** There is no single `status:"partial"` lifecycle. Read `result`, artifacts and `outcome_axes` separately: the axes distinguish lifecycle, execution, objective, artifacts and review. Graceful wrap-up can produce best-effort output, immediate cancellation salvage, and natural completion wins a race with Stop. Cancellation does not delete recorded results/artifact evidence, but capture can fail or be missing. Artifact states include `pending`, `finalizing`, `ready`, `ready_with_changes`, `ready_no_changes`, `failed`, `missing`, and an inapplicable axis can say `not_applicable`. Read the actual file list rather than treating cancellation as no output.

**Pause, continue, resume, hurry:**
- **Pause:** `POST /api/tasks/{id}/pause` takes exactly `{request_id}`. `202` with `latch_pending` means the pause is recorded and still taking hold; repeat the same `request_id`.
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

`POST /api/owner/capability-ack` (`CapabilityAckRequest`) vouches for one exact route's maximum response or context window. It writes acknowledgement evidence, never settings, and a repeat overwrites it.

**Other settings routes:**
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

**One lifecycle lane.** Enabling, disabling, granting, deleting, reviewing, attesting and every marketplace install, update or uninstall run one at a time in one lane, inside the HTTP request, until they settle. A request can therefore take minutes (a review runs model calls). Closing the connection does not stop the job. Watch `GET /api/skills/lifecycle-queue` (the running job and the last 80, kept in memory only) or the progress rows with `system_type: "skill_lifecycle"`, and read `GET /api/extensions` again when it settles.

**The revision guard.** `content_hash` names a skill's revision. An action that sends `expected_content_hash` is refused with 409 when the live revision differs. Re-read `GET /api/extensions` and show the current revision before acting again; that refusal need not include the revision.

**Owner actions on one skill:**
- **Enable or disable:** `POST /api/skills/{skill}/toggle` (`SkillToggleRequest`). Repeating it is harmless.
- **Grant keys or permissions:** `POST /api/skills/{skill}/grants` (`SkillGrantRequest`). Only what the current manifest requests can be granted; a repeat merges.
- **Review:** `POST /api/skills/{skill}/review` answers 200 with the review's outcome (`SkillReviewResponse`). A review already running for the same revision answers `status: "pending"` with its `job_id`. `GET /api/skills/{skill}/review-history/{job_id}` renders a finished review.
- **Attest a review:** `POST /api/owner/skills/{skill}/attest-review` records the owner's verdict: 200 with `ok: true`, otherwise 409.
- **Delete:** `POST /api/skills/{skill}/delete` removes an external or self-authored skill and its state; a repeat answers 404.
- **Presence runtime:** `POST /api/owner/skills/{skill}/presence-runtime` changes a presence profile's runtime choice, compare-and-set on `expected_state_fingerprint`.
- **Publication check:** `POST /api/skills/{skill}/publish-preflight` is read-only; every status answers the same shape, and `state: "hard_block"` is a refusal. Missing publication prerequisites may answer `409` before the skill is looked up; read `reason_code`, not an assumed `404` for an unknown name.
- **Reconcile:** `POST /api/skills/{skill}/reconcile` brings one extension's live state in line with its saved state.

A refused action answers `SkillActionRefusal`: `error`, the `status_code` it was answered with, and optional current facts. For example, a revision-guard refusal carries only `ok`, `error` and `status_code`; absence of revision, review or grant fields proves none of those states. When an action loads or unloads an extension, the core broadcasts `extension_lifecycle` `{skill, action, reason}`: read `GET /api/extensions` and `GET /api/widgets` again.

**Widgets:**
- `GET /api/widgets` (`WidgetsResponse`) lists the widget cards. Each card has a `render.kind`:
  - `iframe`: frame the extension route named by `render.route`.
  - `declarative`: a component description the interface draws; the components are described in `docs/CREATING_SKILLS.md`.
  - `module`: reviewed JavaScript run in a sandboxed frame. [`WIDGET_HOST_PROTOCOL.md`](WIDGET_HOST_PROTOCOL.md) is the complete host protocol: frame, document policy, message grammar, fetch relay, events, theme, size, downloads, links and the dispose handshake.
- `GET /api/extensions/{skill}/module/{entry}` serves reviewed module code (never cached, any origin may load it).

**The skill's own routes:** everything else under `/api/extensions/{skill}/` belongs to the skill.
- A route that does not exist answers 404, and a skill that is not running answers 409.
- A method the skill did not declare answers 405.
- A request body over 512 KiB answers 413.
- A skill that fails answers 502.

**Behind a network password:**
- **Direct frame loads are not portable.** Cookie attachment depends on the browser and request mode. In the separate-host tests, a passive image was refused with 401 in Chromium but sent the session cookie and loaded in WebKit. Do not depend on either outcome for other assets.
- **Loading through the host.** Such files must be fetched through the widget host and loaded from `blob:` URLs, as `docs/CREATING_SKILLS.md` describes.

**Shared controls.** The author kit of shared controls always comes from the core's builtin files, whatever interface is selected.

## 12. The marketplaces

Installs, updates and uninstalls run in the lifecycle lane (§11) and answer when they settle.

**ClawHub** (third-party registry):
- `GET /api/marketplace/clawhub/search`: with `q`, a text search; without it, a registry browse paged by `cursor` and `next_cursor`. `official=1` keeps official skills only.
- `GET .../info/{slug}` and `.../preview/{slug}` read registry metadata only; nothing is downloaded.
- `POST .../install` (`MarketplaceInstallRequest`: `slug`, optional `version`, `auto_review` default true, `overwrite` default false). An existing skill is replaced only with `overwrite`.
- `POST .../update/{name}` (optional `version`) always reviews and overwrites. `POST .../uninstall/{name}`; a repeat answers 400.
- `GET .../installed`. Registry errors answer `{error, code}` with 400, 404, 429 or 502.

**OuroborosHub** (the curated catalog):
- `GET /api/marketplace/ouroboroshub/catalog` (cached for two minutes; `q` filters) and `.../preview/{slug}` (any failure is 404).
- `POST .../install` installs, or with `adopt: true` and the `expected_content_hash` shown turns a local copy into the hub skill. A refusal carries a `code`, for example `catalog_identity_conflict`, `adopt_cas_mismatch` or `adopt_not_eligible`.
- `POST .../update/{name}` reads no body. `POST .../uninstall/{name}`. `POST .../publication/{name}/clear` forgets a publication receipt (`expected_published` is the receipt as shown; a repeat answers `publication_cleared: false`).

**Reading an install answer** (`MarketplaceInstallResponse`):
- **Install:** `ok: true` means the skill is installed, even when review found blockers (`review_status: "blockers"`). Such a skill cannot run, and a repair task may be queued (a system message with `task_id_pending`). A dependency failure gives `ok: false` and rolls the install back.
- **OuroborosHub update:** a review that leaves the skill unable to run, a dependency failure or a reload failure gives `ok: false` and restores the previous version (`rolled_back`).

## 13. Agent accounts

Ouroboros runs connected coding subscriptions through its bundled agent engine, Claudexor. These routes relay the engine. Fields named in the engine's terms (`job`, a maintenance operation, a reset receipt, quota snapshots) are the engine's own API v2 objects, passed through: the core pins the engine release in `ouroboros/claudexor_runtime_pin.json`, and the engine's source is https://github.com/razzant/claudexor. Read the keys the published shapes name, and treat the rest as data.

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
- `GET /api/claudexor/maintenance/harnesses` lists what is installed, tested and available per agent tool, and any operation in progress (rejoin it after a reload). It is passive.
- `POST /api/claudexor/maintenance/operations` (`{harness, target}`, an `Idempotency-Key` header) installs, updates or rolls back one tool and answers `202` with the operation. It can start the engine. `409` with code `maintenance_already_active` names the running operation in `context.operationId`: follow that one.
- Poll `GET .../operations/{id}` while `state` is `queued` or `running`, or `termination` is `unconfirmed`. `POST .../operations/{id}/cancel` asks to cancel; the answer is not proof the work stopped.
- Maintenance refusals are `ClaudexorMaintenanceProblem`: `error` is the engine's problem object (`code`, `message`, `context`), not a sentence. A `503` with `context.acceptance: "unknown"` means the request may have been accepted: send the same key and body again.

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
  - `update_progress`: this process's in-memory progress (`{}` when none);
  - `letter`: the written description of the update, when there is one;
  - `check_ok`: null on status.
- `POST /api/update/check` fetches first, may write the update letter and broadcasts `update_progress_changed`.
- `POST /api/update/preflight` returns a `merge_plan` with `base_sha` and the target.

**Applying an update.** Send `POST /api/update/apply` (`UpdateApplyRequest`) with `strategy`, `expected_base_sha` and `expected_target_sha` from the preflight:
- **Release moved:** 409 `release_moved` means the release changed since the preflight. Run the preflight again.
- **Replacement not applied:** 409 `reason: replace_not_applied`, `abandoned: true` means Git refused the first tree-changing command because its index lock was held. The transaction is disarmed without rollback; `rescue` and `keep_branch` disclose retained preparation artifacts. The lock is untouched. Diagnose it before another deliberate attempt; age alone never authorizes removal. Other partial or unknown failures keep fenced recovery and their error.
- **What `200` means:** inspect `status`, not HTTP success alone. `ok` with `restarting:true` means a restart was requested, not adoption. `assisted_started` means a supervised task owns resolution. `manual` returns a plan without applying or restarting. `restart_required` means the restart that finishes the apply could not be requested (a store-less replace has not moved the checkout yet); show its error and offer the ordinary restart control.
- **Watching it:**
  - `update_progress_changed` frames mean "read the status again" (progress is kept in memory only).
  - After reconnecting, confirm the executing generation (`running_sha`) against the landed transaction/`merge_commit` identity reported by the update, not blindly against the preflight's official `target_sha`. Preserving local commits can create a distinct two-parent merge. Until the landed identity and completed startup are known, adoption remains unconfirmed.
  - A core whose startup check fails can roll back and restart again.
  - `update_status_ready` arrives after a fresh start.
- **Retrying:** there is no idempotency key. A non-manual request can answer 409 while an update is active or its expected base/target no longer matches; `manual` may return another plan. When an outcome is unknown, read status and transaction facts before deciding—never infer failure or resend from a dropped connection.

**Rollback.** `POST /api/git/rollback {target}` stops running work, switches to the given commit or tag and restarts. It has no checks and is not safe to repeat. Ask the person first; afterwards confirm `running_sha`. 409 `reason: restore_not_applied` means Git refused the reset at its held index lock, so the reset changed no checkout file, index entry or ref; the lock, and the keep branch and rescue snapshot made before the reset, are retained. No restart is requested; writers are readmitted through the aborted-update path, which keeps admission closed while an update transaction is active or the stopped work's handoff is unconfirmed.

**Promote.** `POST /api/git/promote` moves the local stable branch to the current commit (nothing is pushed).

**Reset.** `POST /api/reset` deletes the data folders and the settings, then restarts. It is irreversible: ask the person first, and never resend after a dropped connection.

**Commands, restart and Panic.** `POST /api/command` (`CommandRequest`) or a WebSocket `command` frame sends a slash command (`/restart`, `/panic`, `/review`, `/evolve`, `/bg`, `/status`); any other text becomes an owner message in Main.
- **What `200` means:** `{status: "ok"}` means the command was queued, not that it ran. The same `visible_task_id` within 5 seconds answers `{ok, deduped, task_id}`. Either `/restart` or `/panic` can end the connection before or after the answer, so a dropped connection is not a failure.
- **Panic:** it stops the core's work and processes and exits (code 99). The desktop launcher does not start it again.
- **Restart:** a restarted core comes back on its own; reconnect and resynchronize (§5).

**Spend.** `GET /api/cost-breakdown` reports spend from the physical-attempt ledger. A `503` means the ledger cannot be read: spend is unknown, not zero.

**Versions and diagnostics:**
- `GET /api/git/log` lists recent commits and tags (`GitLogResponse`), the choices for a rollback.
- `GET /api/logs/{name}` tails one log (`chat`, `progress`, `events`, `tools` or `supervisor`; `limit` 1 to 2000, default 100; `task_id` filters). An unknown name answers 404 with the allowed names.
- `GET /api/storage` reads the last disk-usage measurement without measuring; `POST /api/storage/measure` starts one bounded measurement or joins the running one (202 either way).
- `GET /api/evolution-data` returns the evolution chart's points (`force` recomputes).

## 16. Errors and unknown outcomes

- **Error bodies** are `ErrorResponse`: `{error}` and, when present, `code` or `reason_code`. A route may add facts beside them (for example `schema_errors`, `task_id` or a skill's current facts). An unexpected failure can also answer `500` as plain text. Statuses beyond the ones a route lists are those `500`s, the auth gate's `401`, and the router's own `404` and `405` for an unknown path or method. Agent-engine routes may also pass the engine's own status through (§13).
- **An error that is an object.** Agent tool maintenance refusals (`ClaudexorMaintenanceProblem`) carry the engine's problem object in `error`, not a sentence. A client that shows errors should accept both forms and show `error.message` for an object.
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

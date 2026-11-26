# 4. Server API Endpoints

Every HTTP and WebSocket route my server exposes (`ouroboros/gateway/router.py`) for the SPA, CLI, extensions, and live streaming, with machine-readable shapes published at `GET /api/contract` and `docs/CLIENT_CONTRACT.md` (Host Service routes: §12). Route table tests check this chapter in both directions, so adding or changing an endpoint updates this manual in the same commit.

`NetworkAuthGate` runs `browser_boundary.py` first on every request/handshake, loopback included: foreign/opaque pages get 403 (WS 4403), passive files pass (DEPLOYMENT). `OUROBOROS_NETWORK_PASSWORD` gates non-loopback via an expiring HMAC session cookie (refused WS closes pre-accept with 4401); unset leaves remote access open by operator choice (CLIENT_CONTRACT §3, DEPLOYMENT). A valid `X-Ouroboros-Link` is my linked Ouroboros, on letter routes only (§12). Widget lease reads take only their lease (§3) and authenticate nothing else.

`server_web.ServedUI` (`server_web.py`) applies `OUROBOROS_UI_DIR` (§7) per generation to `/`, `/static/*`, `onboarding.html` and `auth/index.html`; one record feeds the auth gate, builtin index's response-only owner marker and `StateResponse.served_ui` (source, not digest). Builtin `/onboarding/static/*` and the author kit stay independent; builtin files revalidate `private` (ETag); Telegram phones keep only `GET /static/*`. `GzipText` gzips GET 200 text from 1 KiB (checked against declared `Content-Length`, else the first body chunk, so a stream without `Content-Length` whose first chunk is short passes uncompressed; weak ETag), excluding event streams, downloads and outside stored files (selection, `?builtin=1`, `/auth/*`, fallback: CLIENT_CONTRACT §2–4).

Files routes (`/api/files/*`, `gateway/files.py::file_browser_routes()`; parity: §11) resolve inside the configured root: in-root symlinks work, while outside links may list (`is_symlink: true`) but are never read, written, downloaded, deleted or traversed. Chat downloads (`?upload=<id>`) use `chat_uploads.open_upload` in `data/uploads`, independent of the Files root. `gateway/host_service.py::create_host_service_app` builds the separate token-authenticated loopback Host Service, not a public owner API (§12).

| Method | Path | Note |
|---|---|---|
| GET | `/` | |
| GET | `/api/health` | 200 during boot; `boot` doc: §2 |
| GET | `/api/contract` | |
| GET | `/api/state` | |
| GET | `/api/consciousness` | read-only: status, spend by cause, places, recent wakes |
| GET | `/api/extensions` | installed-skill projection; no fetch |
| POST | `/api/skills/{skill}/publish-preflight` | |
| GET | `/api/extensions/{skill}/manifest` | |
| GET | `/api/extensions/{skill}/module/{entry:path}` | reviewed `.js`/`.mjs` from captured texts; `Access-Control-Allow-Origin: *` |
| GET | `/api/extensions/{skill}/ui/{revision}/{path:path}` | captured page-module files (live revision) |
| GET | `/api/widgets` | live tabs/pinned cards, passive; `Cache-Control: no-store` |
| POST | `/api/widgets/fault` | bounded fault |
| GET | `/api/widget-bridge.js` | route-page frame bridge, public |
| POST | `/api/widgets/asset-leases` | parent opens a file or route lease; re-establishes route leases only |
| POST | `/api/widgets/asset-leases/{op}` | renew/revoke lease |
| GET | `/api/extensions/{skill}/settings_section` | |
| ANY | `/api/extensions/{skill}/{rest:path}` | extension dispatch (§13) |
| GET | `/api/skills/daemons` | |
| POST | `/api/skills/{skill}/toggle` | |
| POST | `/api/skills/{skill}/delete` | |
| GET | `/api/skills/lifecycle-queue` | |
| POST | `/api/skills/{skill}/review` | |
| GET | `/api/skills/{skill}/review-history/{job_id}` | ledger record, else legacy `review_history.jsonl` tail |
| POST | `/api/owner/skills/{skill}/attest-review` | owner-only LLM-review skip; deterministic preflight still runs |
| POST | `/api/skills/{skill}/grants` | |
| GET | `/api/skills/{skill}/mcp-connect` | |
| POST | `/api/skills/{skill}/mcp-connect` | |
| POST | `/api/skills/{skill}/reconcile` | |
| GET | `/api/marketplace/clawhub/search` | |
| GET | `/api/marketplace/clawhub/installed` | |
| GET | `/api/marketplace/ouroboroshub/installed` | frozen hub listing |
| GET | `/api/marketplace/clawhub/info/{slug:path}` | |
| GET | `/api/marketplace/clawhub/preview/{slug:path}` | |
| POST | `/api/marketplace/clawhub/install` | |
| POST | `/api/marketplace/clawhub/update/{name}` | |
| POST | `/api/marketplace/clawhub/uninstall/{name}` | |
| POST | `/api/skills/{skill}/publication/clear` | |
| GET | `/api/skill-sources` | |
| POST | `/api/skill-sources` | |
| POST | `/api/skill-sources/my-skills` | |
| POST | `/api/skill-sources/probe` | |
| POST | `/api/skill-sources/{source_id}` | |
| POST | `/api/skill-sources/{source_id}/remove` | |
| POST | `/api/skill-sources/{source_id}/refresh` | |
| GET | `/api/skill-catalog` | |
| GET | `/api/skill-catalog/search` | |
| GET | `/api/skill-catalog/item/{source_id}/{name}` | |
| POST | `/api/skill-catalog/install` | |
| POST | `/api/skills/{skill}/update` | |
| POST | `/api/skills/{skill}/uninstall` | |
| GET | `/api/onboarding/starter-skills` | |
| GET | `/api/files/list` | |
| GET | `/api/files/read` | |
| GET | `/api/files/content` | |
| GET | `/api/files/download` | `upload`: chat attachment, separate authority |
| POST | `/api/files/upload` | |
| POST | `/api/files/mkdir` | |
| POST | `/api/files/write` | |
| POST | `/api/files/delete` | |
| POST | `/api/files/transfer` | |
| GET | `/onboarding` | |
| GET | `/api/onboarding` | |
| GET | `/api/onboarding/bootstrap` | |
| POST | `/api/onboarding/complete` | |
| GET | `/api/onboarding/complete` | |
| POST | `/api/onboarding/subagents/preview` | |
| POST | `/api/onboarding/model-catalog` | draft access, unsaved |
| POST | `/api/onboarding/connection/identity` | |
| GET | `/api/settings` | effects, tokens (§7) |
| POST | `/api/settings` | origin, conditions, receipt (§7) |
| POST | `/api/settings/secret` | |
| GET | `/api/review-pool` | |
| GET | `/api/claudexor/status` | `last_exit`/`memory` (§9), catalog evidence (§3); never wakes |
| POST | `/api/claudexor/quota/refresh` | full/exact; joins an identical attempt in flight (§3) |
| POST | `/api/claudexor/account-resets` | exact request (§3) |
| GET | `/api/claudexor/account-resets/{operation_id}` | receipt read (§3) |
| POST | `/api/claudexor/wake` | |
| GET | `/api/claudexor/restart` | passive preview of what ends (§3) |
| POST | `/api/claudexor/restart` | preview `{token}`: stop, then start |
| GET | `/api/claudexor/maintenance/harnesses` | selection/newest/actions/receipt; `checkLatest` hits vendor HTTPS |
| POST | `/api/claudexor/maintenance/operations` | 202 latest/exact handle; same-key rejoin; per-attempt refusal (§6) |
| GET | `/api/claudexor/maintenance/operations/{operation_id}` | retained engine facts |
| POST | `/api/claudexor/maintenance/operations/{operation_id}/cancel` | ack does not prove termination |
| POST | `/api/claudexor/login` | |
| GET | `/api/claudexor/login/{job_id}` | |
| DELETE | `/api/claudexor/login/{job_id}` | |
| POST | `/api/claudexor/login/{job_id}/input` | |
| POST | `/api/claudexor/login/{job_id}/reconcile` | |
| DELETE | `/api/claudexor/credential-profiles/{harness}/{profile_id}` | |
| PATCH | `/api/claudexor/credential-profiles/{harness}/{profile_id}` | |
| POST | `/api/owner/runtime-mode` | |
| POST | `/api/owner/auto-grant` | |
| POST | `/api/owner/context-mode` | |
| POST | `/api/owner/effort-range` | |
| POST | `/api/owner/subagents-mode` | `{mode, project_id?}` → `{ok, project_id, subagents_mode, cognition}` (§7) |
| POST | `/api/owner/chat-model` | |
| POST | `/api/owner/chat-processing` | |
| POST | `/api/owner/safety-mode` | |
| POST | `/api/owner/safety-model` | |
| POST | `/api/owner/safety-rules` | |
| GET | `/api/places` | places, circles, connectors (`ouroboros/gateway/places_api.py`) |
| POST | `/api/places` | connect place |
| POST | `/api/places/{place_id}` | atomic owner choices |
| GET | `/api/places/{place_id}/rooms` | captured canonical rooms (§12) |
| DELETE | `/api/places/{place_id}` | disconnect, keeping room history |
| POST | `/api/letters` | a linked Ouroboros's letter, answer or message (§12) |
| GET | `/api/letters/{letter_id}` | |
| GET | `/api/letters/{letter_id}/state` | |
| GET | `/api/letters/{letter_id}/result` | |
| POST | `/api/letters/{letter_id}/cancel` | |
| POST | `/api/peer-links/invite` | owner doors (§12) |
| POST | `/api/peer-links/accept` | |
| POST | `/api/peer-links/introduce` | invite token |
| GET | `/api/peer-links/{link_id}` | owner: pairing status |
| POST | `/api/peer-links/{link_id}/confirm` | |
| POST | `/api/peer-links/{link_id}/revoke` | |
| POST | `/api/circles` | |
| POST | `/api/circles/{circle_id}` | |
| DELETE | `/api/circles/{circle_id}` | |
| POST | `/api/owner/capability-ack` | |
| GET | `/api/ui/contributions` | live controls, speech providers, offers |
| GET | `/api/ui/preferences` | |
| POST | `/api/ui/preferences` | |
| POST | `/api/ui/holds` | |
| GET | `/api/ui/i18n` | language, translation memory; never generates |
| POST | `/api/ui/i18n/language` | writes `OUROBOROS_UI_LANGUAGE`; broadcasts `ui_language_changed` |
| POST | `/api/ui/i18n/missing` | |
| POST | `/api/ui/i18n/import` | |
| GET | `/api/ui/i18n/export` | |
| POST | `/api/ui/i18n/regenerate` | drops generated entries; owner/imported stay |
| GET | `/api/desktop/autostart` | |
| POST | `/api/desktop/autostart` | host-OS state, `owner_audit`, no settings mirror |
| GET | `/api/desktop/background` | |
| POST | `/api/desktop/background` | persists `OUROBOROS_DESKTOP_KEEP_RUNNING`; `owner_audit` |
| GET | `/api/model-catalog` | scopes/batch: CLIENT_CONTRACT §14 |
| POST | `/api/openai-compatible/models` | |
| POST | `/api/providers/test` | |
| POST | `/api/tasks` | |
| GET | `/api/tasks` | |
| GET | `/api/tasks/{task_id}` | |
| GET | `/api/tasks/{task_id}/reviews` | paged; gaps |
| GET | `/api/tasks/{task_id}/reviews/skill/{job_id}` | saved detail |
| GET | `/api/tasks/{task_id}/reviews/{record_id}/{seat_id}` | saved source; no paths |
| GET | `/api/tasks/{task_id}/events` | integer rank |
| POST | `/api/tasks/{task_id}/events` | read-only v2 cursor (§3) |
| GET | `/api/tasks/{task_id}/artifacts/{name}` | own stores: POSIX no-follow/Windows NT relative handles; admission before child-store metadata; SVG `sandbox` CSP + `nosniff` (`gateway/task_archive.py`) |
| POST | `/api/tasks/{task_id}/cancel` | (§5) |
| POST | `/api/tasks/{task_id}/hurry` | (§5) |
| POST | `/api/tasks/{task_id}/model` | `ouroboros/gateway/task_model.py`: durable root primary; set/follow/return/wait/keep |
| POST | `/api/tasks/{task_id}/pause` | (§6) |
| POST | `/api/tasks/{task_id}/owner-keep` | `OwnerKeepRequest`; keep/release materials |
| POST | `/api/tasks/{task_id}/continue` | (§6) |
| POST | `/api/tasks/{task_id}/resume` | |
| POST | `/api/call/open` | live call as room turn when Calls (beta) is on (§13) |
| POST | `/api/decisions` | |
| POST | `/api/decisions/secret` | |
| GET | `/api/cards/{card_id}` | living card body/presses |
| GET | `/api/cards/{card_id}/app` | app HTML as `text/plain`, `nosniff`, `no-store` |
| POST | `/api/cards/{card_id}/pin` | pin component card on Widgets |
| DELETE | `/api/cards/{card_id}/pin` | |
| GET | `/api/schedules` | |
| POST | `/api/schedules` | |
| POST | `/api/schedules/{schedule_id}/action` | explicit action/reason (§5) |
| DELETE | `/api/schedules/{schedule_id}` | |
| POST | `/api/command` | |
| GET | `/api/reset` | status (CLIENT_CONTRACT §15) |
| POST | `/api/reset` | legacy; refuses |
| POST | `/api/reset/v1` | confirmed intent (CLIENT_CONTRACT §15) |
| GET | `/api/git/log` | |
| POST | `/api/git/rollback` | |
| POST | `/api/git/promote` | |
| GET | `/api/update/status` | passive, no network/prose |
| POST | `/api/update/check` | |
| POST | `/api/update/preflight` | |
| POST | `/api/update/description` | |
| POST | `/api/update/apply` | |
| GET | `/api/cost-breakdown` | |
| GET | `/api/storage` | passive; `?count=` omits held count |
| POST | `/api/storage/measure` | start/join child count |
| POST | `/api/storage/measure/cancel` | stop child; saved count stays |
| POST | `/api/storage/free/preview` | pure preview; 202 start/join |
| POST | `/api/storage/free` | `StorageFreeRequest`: pool `items`/`groups` or `manual` ids of `view=manual` (`accept_loss: true`); 202 `started`/`reason` |
| GET | `/api/storage/free/items` | `StorageFreeItemsResponse`: `pool`/`excluded`/`manual`/`outcomes` page |
| GET | `/api/services/kept` | `KeptServicesResponse` (§9) |
| POST | `/api/services/kept/{launch_id}/stop` | `reason`; one launch, fresh outcome |
| POST | `/api/storage/free/cancel` | stop preview/run; freed stays freed |
| GET | `/api/evolution-data` | |
| GET | `/api/projects` | list/lookup; unavailable ≠ empty (CLIENT_CONTRACT §6) |
| POST | `/api/projects` | |
| POST | `/api/projects/from-task` | |
| POST | `/api/projects/{project_id}/update` | |
| POST | `/api/projects/{project_id}/delete` | |
| GET | `/api/projects/{project_id}/folder` | |
| POST | `/api/projects/{project_id}/folder` | |
| GET | `/api/fs/dirs` | |
| GET | `/api/chat/history` | |
| GET | `/api/chat/archive` | |
| GET | `/api/chat/search` | |
| GET | `/api/recall` | neighbour door (CLIENT_CONTRACT §5) |
| GET | `/api/logs/{name}` | |
| GET | `/api/logs/{name}/raw` | |
| POST | `/api/chat/upload` | |
| DELETE | `/api/chat/upload` | pending uploads only, else 409 |
| POST | `/api/local-model/start` | |
| POST | `/api/local-model/stop` | |
| GET | `/api/local-model/status` | |
| POST | `/api/local-model/test` | |
| POST | `/api/local-model/install-runtime` | |
| GET | `/api/mcp/status` | read-only listing |
| POST | `/api/mcp/refresh` | list and publish |
| POST | `/api/mcp/sign-in` | sign-in steps |
| POST | `/api/sign-in` | skill sign-in steps (§13) |
| GET | `/api/sign-in/status` | read-only status (§13) |
| POST | `/api/mcp/test` | draft probe, unpublished |
| POST | `/api/mcp/import/preview` | |
| WS | `/ws` | |
| WS | `/ws/call/{call_id}` | admitted call audio/frames (`call_protocol.py`) |
| STATIC | `/static/*` | |
| STATIC | `/onboarding/static/*` | |
| STATIC | `/auth/*` | selected public login assets |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/identity` | |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/call/open` | Presence call admission (§12) |
| WS | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/call/{call_id}/peer` | granted meeting peer (§12) |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/tools/schemas` | |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/allocate-internal` | |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/inject` | |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/operations/{operation_ref:path}` | caller's accepted message, pending→terminal |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/cancel` | cancel owner; typed |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/decision` | `task_decision.answer_decision` (transport skill); same `/api/decisions` answer, winner's Reply file kept; read via `rejoin_decision` |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/ui/language` | shares `ui_i18n.choose_language` |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/place` | connector place state |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/turn` | capture before attention |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/delivery` | |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/history` | |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/work/{work_ref}` | |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/work/{work_ref}` | retains transport queue observations |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/late_result` | return through an admitted call reference (§12) |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/ui/ws-message` | |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/sign-in/accounts` | caller's declared-service accounts (§13) |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/sign-in/token` | caller's own access token (§13) |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/sign-in/handover` | take over caller's earlier sign-ins (§13) |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/notify` | `notify_owner` grant: one signed `skill_notice` System row, no model turn |
| WS | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/events` | |

Update-status, log-tail and schedule-list reads run off-loop. `/api/state` adds `current_questions`/`main_attention` and work projections (`works`, `rooms`) with completeness and typed gaps (§3); `active_chat_activities` stays root-only. GET neither scans chat nor writes: requests share one `gateway/state_publication.py` display publication rebuilt only on revision change (never authority), and actions reuse `/api/decisions` and task endpoints.

### WebSocket protocol

`/ws` streams live browser events (CLIENT_CONTRACT §5); durable truth lives with queue, task, Project, review, skill, settings, cost and update owners for REST/history recovery. `gateway/ws.py` admits JSON objects, owned extension namespaces, `command` text and `chat` text/attachments. Chat acceptance (row → queue → echo) runs off-loop (`gateway._helpers.run_sync_to_completion`), chained in receive order per socket unawaited so Panic/Restart stay responsive behind held chats (§5 Bridge intake); disconnect awaits `settle_to_completion`. Public sockets carry no Host Service or owned-daemon tokens. Feature subscriptions and initial Project chat-id lookup (which can fail) precede first open; each decoded frame emits `message`, then its type event (including reviewed namespaced events for Widgets).

#### WebSocket chat

A `chat` frame carries owner text and/or uploads plus optional `sender_session_id`, `client_message_id`, `chat_id`, `project_id`, `reply_to`/`reply_excerpt` and send-time `client_surface`. The gateway normalizes surface (`client_surface.normalize_client_surface`), stamps `received_at` and shares `LocalChatBridge.enqueue_local_message` with other transports; promotion/steering keep origin surface while chat `transport` chooses the reply route.

Same `client_message_id` retries rejoin only identical text, ordered attachments and Reply. Echo (`ingress_accepted`) and history show dispatch: retry/bus init dispatches `ingress_pending` once, `ingress_dispatched` has entered, unreadable history claims and writes nothing (`supervisor/message_ingress.py`), and `initialization_notice` correlates refusals and explains unusable sources or conflicting ids without reporting boot failure.

Offline frames queue in memory (`ws.js`; overflow drops the oldest, reload clears it); uploads use `queue:false`, and failed writes release new uploads. `createUnconfirmedSends` (`chat_attachments.js`) holds non-command frames by ID with citation and upload refs (no file bytes): echo/history settles the ID, "Send again" reuses it, "Discard" drops it, and a selected reply's "Send as new" returns words and quote to an empty composer. Off-loop acceptance confines attachments to `data/uploads` basenames bound to measured `size`/`sha256` (swapped bytes rejected; model pixels come only from staged files). Web owner identity is fixed (`chat_id` selects a thread, never minting an external owner).

`command` routes through the bridge without rebroadcast (authority/outcomes stay outside `ws.py`). Panic bypasses the queue on the existing socket (§9); header controls keep ordinary routing, and prose never implies a command (BIBLE P5).

#### Frame addressing

Outbound types follow `gateway.contracts.WS_MESSAGE_TYPES` minus `command`. `update_progress_changed` invalidates update observations without proving boot completion; progress carries additive `ChatOutbound` (`GET /api/contract`) where absent optional fields never mean no receipt, cost or result.

Project panels take chat, typing, media and log frames by `chat_id` or `also_rooms` (Main: §3; `projects_changed` extends known ids pre-fetch). The bus stamps `project_thread` by registry membership, not ID range (Telegram unstamped); `chat_activity.mainThreadAccepts` honors web `room_echo` as display-only. `supervisor/log_addressing.py::address_task_event` stamps task log audiences at ingress (`0` is `HIDDEN_CHAT_ID`, not missing); conversation turns carry addresses by value since queued events can outlive a turn row. `message_annotation` updates one canonical owner row; `task_named` updates existing cards only.

Extension WS traffic namespaces via `extension_loader.extension_surface_name()` so extensions cannot shadow built-ins. Each inbound frame resolves its skill and rechecks desired/reviewed/granted/enabled/live state; out-of-process handlers run in their extension child off the event loop, in-process synchronous ones off the ASGI loop, trusted in-process async ones on it; handler failure never ends the socket loop.

Each socket owns one writer and outbox (`ws._Connection`), so slow/half-open browsers delay only themselves; each broadcast is encoded once, for open sockets only, with one loop wake per burst (`ws.py`), and send failures retire the socket and log `broadcast_partial_failure` off-loop while durable producers keep their events.

#### Reconnect and recovery

Reconnect uses bounded exponential backoff and a silence watchdog. One `ws.js decideServedPage` policy (CLIENT_CONTRACT §2, §5) governs post-open reads and socket-down probes so transient drops keep in-page state. Without `served_ui` (an older core) any page, marked or not, and the core's marked page while it serves builtin reload on a changed or unprovable SHA and keep the page and its queued frames on an unchanged SHA (or an unversioned `/api/state` with no remembered SHA, possibly stale assets). That SHA rule stands as-is for an unscoped page without profile comparison; on a scoped page, a confirmed data-profile change drops the previous profile's queued frames (`outbound_dropped`) and reloads even at the same SHA (`ws.js::_adoptServingScope`). The marked page reloads once after a restart applied a directory or `none`; an unmarked page of a current core owns its refresh and only reconnects; failed/non-OK reads never reload, and four healthy probes reload a once-connected page once per disconnect episode (`serverAnswersWhileDown` is an observation, not a cause).

On `open`, each Chat resyncs durable history off-loop (deduped against live frames; reconnect adds no row); on `close`, it withdraws online/accounting presentation. Transient frames are not durable authority: chat rows, task results, queue snapshots, Project revisions, review/cost ledgers and lifecycle state recover.

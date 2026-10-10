# 4. Server API Endpoints

Registry of the browser, CLI and Host Service routes, with non-loopback authentication, file confinement and the WebSocket protocol. `ouroboros/gateway/router.py` owns the executable Route objects; the table below mirrors them one row per route, and tests check that nothing mounted is undocumented and nothing documented is unmounted.

With `OUROBOROS_NETWORK_PASSWORD` configured, non-loopback HTTP and WebSocket access requires authentication; loopback clients bypass the gate, and `/api/health` plus the middleware-owned login/logout paths stay reachable. Browser sessions use a server-keyed, expiring HttpOnly HMAC cookie, `Secure` only under TLS, so a plain-HTTP LAN session does not enter a login loop. An unauthenticated WebSocket is closed with code 4401 before `ws_endpoint` accepts it. Without a password, non-loopback access stays open by explicit operator choice.

File-browser routes are contributed by `gateway/files.py::file_browser_routes()`. `gateway/contracts.py` is the frozen descriptive envelope and endpoint index mirrored by `web/modules/api_types.js` and parity tests; its `TypedDict` classes perform no runtime JSON validation. The loopback Host Service is a separate token-authenticated app assembled by `gateway/host_service.py::create_host_service_app`, not a second public owner API (§12).

Every path-addressed `/api/files/*` operation resolves its requested `path` and refuses when the resolution leaves the configured file root. In-root symlinks stay usable; out-of-root symlinks may be listed with `is_symlink: true` but cannot be read, written, downloaded, deleted or traversed. A chat attachment (`/api/files/download?upload=<id>`) names no path: a separate confined authority opens it by stored upload id through `chat_uploads.open_upload` inside `data/uploads`, independent of the Files root.

| Method | Path | Note |
|---|---|---|
| GET | `/` | |
| GET | `/api/health` | |
| GET | `/api/state` | |
| GET | `/api/extensions` | installed-skill projection; never fetches the hub catalog |
| POST | `/api/skills/{skill}/publish-preflight` | |
| GET | `/api/extensions/{skill}/manifest` | |
| GET | `/api/extensions/{skill}/module/{entry:path}` | reviewed `.js`/`.mjs` from captured texts; `Access-Control-Allow-Origin: *` |
| GET | `/api/widgets` | passive projection of live UI tabs; `Cache-Control: no-store` |
| GET | `/api/extensions/{skill}/settings_section` | |
| ANY | `/api/extensions/{skill}/{rest:path}` | extension dispatch (§13) |
| GET | `/api/skills/daemons` | |
| POST | `/api/skills/{skill}/toggle` | |
| POST | `/api/skills/{skill}/delete` | |
| GET | `/api/skills/lifecycle-queue` | |
| POST | `/api/skills/{skill}/review` | |
| GET | `/api/skills/{skill}/review-history/{job_id}` | bounded tail window of `review_history.jsonl` |
| POST | `/api/owner/skills/{skill}/attest-review` | owner-only skip of the LLM review; deterministic preflight still runs |
| POST | `/api/skills/{skill}/grants` | |
| POST | `/api/skills/{skill}/reconcile` | |
| GET | `/api/marketplace/clawhub/search` | |
| GET | `/api/marketplace/clawhub/installed` | |
| GET | `/api/marketplace/clawhub/info/{slug:path}` | |
| GET | `/api/marketplace/clawhub/preview/{slug:path}` | |
| POST | `/api/marketplace/clawhub/install` | |
| POST | `/api/marketplace/clawhub/update/{name}` | |
| POST | `/api/marketplace/clawhub/uninstall/{name}` | |
| GET | `/api/marketplace/ouroboroshub/catalog` | |
| GET | `/api/marketplace/ouroboroshub/installed` | |
| GET | `/api/marketplace/ouroboroshub/preview/{slug:path}` | |
| POST | `/api/marketplace/ouroboroshub/install` | also the adopt transport (`adopt: true`, `expected_content_hash`) |
| POST | `/api/marketplace/ouroboroshub/update/{name}` | |
| POST | `/api/marketplace/ouroboroshub/uninstall/{name}` | |
| POST | `/api/marketplace/ouroboroshub/publication/{name}/clear` | forgets only the local publication receipt |
| GET | `/api/files/list` | |
| GET | `/api/files/read` | |
| GET | `/api/files/content` | |
| GET | `/api/files/download` | `?upload=<id>`: chat attachment, separate authority (above) |
| POST | `/api/files/upload` | |
| POST | `/api/files/mkdir` | |
| POST | `/api/files/write` | |
| POST | `/api/files/delete` | |
| POST | `/api/files/transfer` | |
| GET | `/onboarding` | |
| GET | `/api/onboarding` | |
| POST | `/api/onboarding/complete` | |
| POST | `/api/onboarding/subagents/preview` | |
| GET | `/api/settings` | |
| POST | `/api/settings` | |
| POST | `/api/settings/secret` | |
| GET | `/api/review-pool` | |
| GET | `/api/claudexor/status` | `last_exit`/`memory` (§9), resource-catalog evidence (§3); `?view=quota`: roster/quota only; never wakes |
| POST | `/api/claudexor/quota/refresh` | full or exact account (§3) |
| POST | `/api/claudexor/account-resets` | exact request and Idempotency-Key (§3) |
| GET | `/api/claudexor/account-resets/{operation_id}` | receipt inspection (§3) |
| POST | `/api/claudexor/wake` | |
| GET | `/api/claudexor/maintenance/harnesses` | passive inspection; optional fresh/latest check |
| POST | `/api/claudexor/maintenance/operations` | Idempotency-Key; 202 operation handle |
| GET | `/api/claudexor/maintenance/operations/{operation_id}` | retained engine facts |
| POST | `/api/claudexor/maintenance/operations/{operation_id}/cancel` | acknowledgement does not prove termination |
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
| POST | `/api/owner/safety-mode` | |
| POST | `/api/owner/skills/{skill}/presence-runtime` | |
| POST | `/api/owner/capability-ack` | |
| GET | `/api/ui/preferences` | |
| POST | `/api/ui/preferences` | |
| GET | `/api/ui/i18n` | interface language and translation memory; a read never starts generation |
| POST | `/api/ui/i18n/language` | writes `OUROBOROS_UI_LANGUAGE`; broadcasts `ui_language_changed` |
| POST | `/api/ui/i18n/missing` | |
| POST | `/api/ui/i18n/import` | |
| GET | `/api/ui/i18n/export` | |
| POST | `/api/ui/i18n/regenerate` | drops generated entries; owner and imported ones stay |
| GET | `/api/desktop/autostart` | |
| POST | `/api/desktop/autostart` | `{enabled}`; host-OS state, `owner_audit`, no settings mirror |
| GET | `/api/desktop/background` | |
| POST | `/api/desktop/background` | `{enabled}`; persists `OUROBOROS_DESKTOP_KEEP_RUNNING`; `owner_audit` |
| GET | `/api/model-catalog` | |
| POST | `/api/openai-compatible/models` | |
| POST | `/api/providers/test` | |
| POST | `/api/tasks` | |
| GET | `/api/tasks` | |
| GET | `/api/tasks/{task_id}` | |
| GET | `/api/tasks/{task_id}/events` | integer rank |
| POST | `/api/tasks/{task_id}/events` | read-only v2 cursor (§3 History reads and the SSE v2 transport) |
| GET | `/api/tasks/{task_id}/artifacts/{name}` | bare name, `?relpath=`, `?archive=<dir>` ZIP, delegated `?source=`; digest drift 409, failed capture 404; 503 without confined-open support (`gateway/task_archive.py`) |
| POST | `/api/tasks/{task_id}/cancel` | body `cascade`, `stop_policy`, `stop_action_id` (§5 Stop policy and hurry) |
| POST | `/api/tasks/{task_id}/hurry` | (§5 Stop policy and hurry) |
| POST | `/api/tasks/{task_id}/pause` | `{request_id}` (§6 Owner Pause of a whole tree) |
| POST | `/api/tasks/{task_id}/continue` | `{action_nonce}` (§6 Owner Continue after a technical interruption) |
| POST | `/api/tasks/{task_id}/resume` | |
| POST | `/api/decisions` | |
| GET | `/api/schedules` | |
| POST | `/api/schedules` | |
| POST | `/api/schedules/{schedule_id}/action` | explicit action and reason (§5 Schedules and follow-ups) |
| DELETE | `/api/schedules/{schedule_id}` | |
| POST | `/api/command` | |
| POST | `/api/reset` | |
| GET | `/api/git/log` | |
| POST | `/api/git/rollback` | |
| POST | `/api/git/promote` | |
| GET | `/api/update/status` | passive: no network, no update letter |
| POST | `/api/update/check` | |
| POST | `/api/update/preflight` | |
| POST | `/api/update/apply` | |
| GET | `/api/cost-breakdown` | |
| GET | `/api/evolution-data` | |
| GET | `/api/projects` | |
| POST | `/api/projects` | |
| POST | `/api/projects/from-task` | |
| POST | `/api/projects/{project_id}/update` | |
| POST | `/api/projects/{project_id}/delete` | |
| GET | `/api/fs/dirs` | |
| GET | `/api/chat/history` | |
| GET | `/api/logs/{name}` | |
| POST | `/api/chat/upload` | |
| DELETE | `/api/chat/upload` | pending uploads only, else 409 |
| POST | `/api/local-model/start` | |
| POST | `/api/local-model/stop` | |
| GET | `/api/local-model/status` | |
| POST | `/api/local-model/test` | |
| POST | `/api/local-model/install-runtime` | |
| GET | `/api/mcp/status` | |
| POST | `/api/mcp/refresh` | |
| POST | `/api/mcp/test` | |
| WS | `/ws` | |
| STATIC | `/static/*` | |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/identity` | |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/tools/schemas` | |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/allocate-internal` | |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/inject` | |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/operations/{operation_ref:path}` | the calling skill's own accepted message, pending through terminal |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/cancel` | the existing cancellation owner; a typed outcome |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/decision` | `task_decision.answer_decision` relayed for a transport skill |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/ui/language` | the same `ui_i18n.choose_language` writer as `/api/ui/i18n/language` |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/turn` | |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/delivery` | |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/work/{work_ref}` | |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/work/{work_ref}` | retains transport queue observations |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/ui/ws-message` | |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/notify` | `notify_owner` grant: one signed `skill_notice` System row, no model turn |
| WS | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/events` | |

`server.py` owns startup, lifespan and static files; `gateway/*` owns HTTP and WebSocket. Update-status, log-tail and schedule-list reads run off the event loop.

### WebSocket protocol

`/ws` delivers live browser events; the queue, task, Project, review, skill, settings, cost and update owners persist the truth a reconnecting client rebuilds through REST and history. `gateway/contracts.py` owns the frozen envelopes and Python/JS parity; `gateway/ws.py` admits JSON objects, owned extension namespaces and `command` text and `chat` text/attachments. Chat acceptance (row → queue → echo) runs off the ASGI loop through `gateway._helpers.run_sync_to_completion`. Each socket chains its chats in receive order without awaiting them, so Panic and Restart stay responsive behind a held chat (§5 Bridge intake and commands); disconnect waits for received chats to settle (`settle_to_completion`). Authentication above governs admission; public sockets never carry Host Service or owned-daemon tokens.

The browser opens one socket for the whole SPA. Feature modules subscribe, and the complete Project chat-id set is fetched, before the first open, so an early Project frame cannot pass for Main traffic. Every decoded frame reaches the generic `message` event and then its type-specific event, so Widgets consume reviewed namespaced events without a second socket.

#### WebSocket chat

A browser `chat` frame carries owner text and/or uploaded attachment references, and optionally `sender_session_id`, `client_message_id`, `force_plan`, `chat_id`, `project_id` and `client_surface`: raw sending-surface facts measured at send time, because the pywebview bridge appears asynchronously after load. The gateway normalizes them (`client_surface.normalize_client_surface`), stamps host `received_at` at receipt, and every transport passes one common enqueue (`LocalChatBridge.enqueue_local_message`). The surface fact is per-message provenance, distinct from the chat-scoped `transport` reply route, assembled at its producer, never inferred at render, and carried from the originating owner turn by promotion or steering (`ouroboros/client_surface.py`).

Acceptance is idempotent on `client_message_id`: a successful browser `send()` means only that the current socket took the frame, not that a task was admitted. A same-id frame rejoins the accepted row when its words and ordered attachment content match and is refused otherwise; the `ingress_accepted` echo and history carry what this process knows of dispatch (`ingress_pending`, `ingress_dispatched`, `ingress_undispatched`), facts a restart loses; nothing then replays, while row and uploads are kept (`supervisor/message_ingress.py`). Unreadable chat history is unknown, so the frame is refused with the `initialization_notice` before anything is claimed or written.

Frames sent while disconnected wait in a bounded process-local queue (oldest dropped), flushed after reconnect and lost on reload: not a second durable outbox. Attachment messages set `queue:false`, because uploads happen just before the send and a retained frame would leave unowned temporaries; on socket loss Chat refuses the message, cleans those and keeps the staged files for an explicit retry. Once the socket took a frame, `chat_attachments.createUnconfirmedSends` keeps it under its id in tab sessionStorage across reloads until that id's echo or history row settles it; an unsettled frame offers "Send again" (the same frame and id, never automatic) and "Discard".

Off-loop acceptance confines attachment specs to upload-root basenames bound to measured `size`/`sha256`, so bytes swapped under the path afterwards are rejected rather than staged; model pixels come only from staged files. The web owner identity is fixed: `chat_id` selects a thread and cannot mint an external owner identity.

A built-in `command` frame carries a slash command into the same bridge with rebroadcast disabled; command routing, owner authorization, queue authority and typed outcomes stay outside the socket module, so the Main header controls reuse the ordinary command contract for Restart, Panic, review, evolution and background consciousness. The socket does not infer intent from command-looking prose (BIBLE P5).

#### Frame addressing

Outbound envelope types are the members of `gateway.contracts.WS_MESSAGE_TYPES` other than the inbound `command`. `update_progress_changed` only invalidates the process-local update observation and is never boot-completion proof. Chat progress carries additive presentation facts (lineage, model lane, delegated route, execution evidence, review projection, cancellation eligibility, outcome axes, artifact references, nullable cost and finality); consumers must not infer a missing receipt, cost or result from an absent optional field.

Project chat, typing, media and log frames carry `chat_id`; each Project panel consumes its own thread, and Main admits only Project question mirrors, the host-stamped Project lifecycle rows (§3 Project handoff receipts) and a Project root's `main_notice`. `projects_changed` carries the new chat id so every tab extends its fan-out set before fetching the registry; when even that loses the race, the server-stamped `project_thread` marker on the frame keeps Main from adopting it (`chat_activity.mainThreadAccepts`). The marker is set once at the message-bus broadcast choke from the registry, a membership lens, not a numeric range, so external transport ids such as Telegram stay unstamped. Task-scoped log events get their audience at supervisor ingress: `supervisor/log_addressing.py::address_task_event` stamps it from host-attested truth (an explicit chat_id of 0 is `HIDDEN_CHAT_ID`, never missing), and direct turns carry their chat by value, because the registry entry dies with the turn while queued events drain later. `message_annotation` updates one canonical owner message without a second bubble; `task_named` updates a card only where that task exists.

Extension WebSocket traffic is namespaced by `extension_loader.extension_surface_name()`, so an extension cannot shadow a built-in type. On each incoming extension frame the gateway resolves the owning skill and reconciles whether its extension is still desired, reviewed, granted, enabled and live; out-of-process handlers run in their extension child off the event loop. A non-`None` result returns as `<request-type>.reply`; a missing or failed handler becomes a typed error log frame rather than ending the socket loop.

Server broadcasts snapshot the client list and send to all clients concurrently, so one slow or half-open browser cannot head-of-line-block every other tab. Failed sends remove only the dead clients and append a durable `broadcast_partial_failure` event; the domain event stays owned by its durable producer. Restart shutdown closes remaining clients with code 1012 so they enter the ordinary reconnect path.

#### Reconnect and recovery

The browser reconnects with bounded exponential delay and shows the reconnect overlay; a watchdog closes a seemingly open connection without inbound frames, so heartbeats prove stream liveness, not task progress. One served-SHA decision (`ws.js decide()`: `keep` / `reload_changed` / `reload_unknown`) governs both the post-open state read and the socket-down recovery probe, so a transient drop cannot destroy in-page state: a changed or unprovable SHA reloads, because a restarted server must not keep old JavaScript or CSS alive in PyWebView; an unchanged SHA keeps the page and its queued outbound frames; an unversioned `/api/state` keeps the page when no SHA was ever remembered, deliberately accepting possibly stale assets. A failed or non-OK state read never reloads, because a server seconds into its life can answer 500; the probe discipline and its one forced reload per disconnect episode live in `web/modules/ws.js`.

Each Chat instance handles `open` by resynchronizing archive-aware durable history and `close` by withdrawing online and accounting presentation; reconnect deduplication covers the overlap between live frames and REST replay, and large history parsing runs off the server event loop. Delivery is live plus replay, not a promise that every transient frame is persisted: durable chat rows, task results, queue snapshots, Project revisions, review ledgers, cost ledgers and lifecycle state remain the recovery authorities.

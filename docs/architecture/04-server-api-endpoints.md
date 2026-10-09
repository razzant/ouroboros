# 4. Server API Endpoints

Endpoint registry for browser, CLI and Host Service routes, their handlers, non-loopback authentication, file confinement and WebSocket protocol. Tests check mounted routes; optional progress sources retain identity and platform limits.

If `OUROBOROS_NETWORK_PASSWORD` is configured, non-loopback HTTP and WebSocket access requires authentication; loopback clients bypass the gate, and `/api/health` plus the middleware-owned login/logout paths stay reachable. Browser sessions use a server-keyed, expiring HttpOnly HMAC cookie; `Secure` is set only under TLS so a plain-HTTP LAN session does not enter a login loop. An unauthenticated WebSocket is closed with code 4401 before `ws_endpoint` accepts it. With no configured password, non-loopback access remains open by explicit operator choice.

Browser/CLI routes live in `ouroboros/gateway/router.py`; file-browser routes are contributed by `gateway/files.py::file_browser_routes()`. `gateway/contracts.py` is the frozen descriptive envelope and endpoint index mirrored by `web/modules/api_types.js` and parity tests; its `TypedDict` classes perform no runtime JSON validation. The loopback Host Service is a separate token-authenticated app assembled by `gateway/host_service.py::create_host_service_app`, not another public owner API.

Every path-addressed `/api/files/*` operation resolves its requested `path` and refuses the operation when that resolution leaves the configured file root. In-root symlinks remain usable; out-of-root symlinks may be listed with `is_symlink: true` but cannot be read, written, downloaded, deleted, or traversed. Resolution is authoritative. A chat attachment (`/api/files/download?upload=<id>`) names no path: it is a separate confined authority, independent of the Files root — only a well-formed stored upload id, opened through the confined open inside `data/uploads` (`chat_uploads.open_upload`: never a link, a directory or another name).

| Method | Path | Handler |
|---|---|---|
| GET | `/` | `server.index_page` |
| GET | `/api/health` | `gateway.state.api_health` |
| GET | `/api/state` | `gateway.state.api_state` |
| GET | `/api/extensions` | `gateway.extensions.api_extensions_index` (unique rows additionally carry `content_hash`, `published` (validated receipt object or null), `published_malformed`; identity-collision rows carry `identity_collision: true` and omit the receipt fields; a hub row's display hints `official_hub_verified`/`owner_attestable` are matched against the fresh display-plane catalog memo and are `null` without one — this GET never fetches the catalog) |
| POST | `/api/skills/{skill}/publish-preflight` | `gateway.skill_publish.api_skill_publish_preflight` |
| GET | `/api/extensions/{skill}/manifest` | `gateway.extensions.api_extension_manifest` |
| GET | `/api/extensions/{skill}/module/{entry:path}` | `gateway.extensions.api_extension_module` (live-registration authorization, reviewed `.js`/`.mjs` siblings from captured texts, `Access-Control-Allow-Origin: *` on every answer) |
| GET | `/api/widgets` | `gateway.widgets.api_widgets` (passive projection of the loader's live UI tabs via `extension_loader.live_widget_projection`; `Cache-Control: no-store`) |
| GET | `/api/extensions/{skill}/settings_section` | `gateway.extensions.api_extension_settings_section` |
| ANY | `/api/extensions/{skill}/{rest:path}` | `gateway.extensions.api_extension_dispatch` |
| GET | `/api/skills/daemons` | `gateway.extensions.api_skill_daemons` |
| POST | `/api/skills/{skill}/toggle` | `gateway.extensions.api_skill_toggle` |
| POST | `/api/skills/{skill}/delete` | `gateway.extensions.api_skill_delete` |
| GET | `/api/skills/lifecycle-queue` | `gateway.extensions.api_skill_lifecycle_queue` |
| POST | `/api/skills/{skill}/review` | `gateway.extensions.api_skill_review` |
| GET | `/api/skills/{skill}/review-history/{job_id}` | `gateway.extensions.api_skill_review_history_detail` (bounded lazy detail from a fixed tail window of `review_history.jsonl`; missing job 404, outside-window honestly unavailable; slot/attempt usage joins from the physical-attempt ledger) |
| POST | `/api/owner/skills/{skill}/attest-review` | `gateway.extensions.api_owner_skill_attest_review` (OWNER-ONLY skip of the LLM review; the deterministic preflight floor still runs, 409 on failure; routes through `run_skill_review_lifecycle` for the post-pass reconcile) |
| POST | `/api/skills/{skill}/grants` | `gateway.extensions.api_skill_grants` |
| POST | `/api/skills/{skill}/reconcile` | `gateway.extensions.api_skill_reconcile` |
| GET | `/api/marketplace/clawhub/search` | `gateway.marketplace.api_marketplace_search` |
| GET | `/api/marketplace/clawhub/installed` | `gateway.marketplace.api_marketplace_installed` |
| GET | `/api/marketplace/clawhub/info/{slug:path}` | `gateway.marketplace.api_marketplace_info` |
| GET | `/api/marketplace/clawhub/preview/{slug:path}` | `gateway.marketplace.api_marketplace_preview` |
| POST | `/api/marketplace/clawhub/install` | `gateway.marketplace.api_marketplace_install` |
| POST | `/api/marketplace/clawhub/update/{name}` | `gateway.marketplace.api_marketplace_update` |
| POST | `/api/marketplace/clawhub/uninstall/{name}` | `gateway.marketplace.api_marketplace_uninstall` |
| GET | `/api/marketplace/ouroboroshub/catalog` | `gateway.marketplace.api_ouroboroshub_catalog` |
| GET | `/api/marketplace/ouroboroshub/installed` | `gateway.marketplace.api_ouroboroshub_installed` |
| GET | `/api/marketplace/ouroboroshub/preview/{slug:path}` | `gateway.marketplace.api_ouroboroshub_preview` |
| POST | `/api/marketplace/ouroboroshub/install` | `gateway.marketplace.api_ouroboroshub_install` (also the adopt transport: `{adopt: true, expected_content_hash}` replaces an external same-name occupant with the sha256-verified catalog payload; adopt forces `auto_review`, conflicts with `overwrite`, typed 400/409/502 codes ride the lifecycle payload) |
| POST | `/api/marketplace/ouroboroshub/update/{name}` | `gateway.marketplace.api_ouroboroshub_update` |
| POST | `/api/marketplace/ouroboroshub/uninstall/{name}` | `gateway.marketplace.api_ouroboroshub_uninstall` |
| POST | `/api/marketplace/ouroboroshub/publication/{name}/clear` | `gateway.marketplace.api_ouroboroshub_clear_publication` (compares the displayed receipt and forgets only that local record) |
| GET | `/api/files/list` | `gateway.files.api_files_list` |
| GET | `/api/files/read` | `gateway.files.api_files_read` |
| GET | `/api/files/content` | `gateway.files.api_files_content` |
| GET | `/api/files/download` | `gateway.files.api_files_download` (`?upload=<id>`: a chat attachment, independent of the Files root; confined open, Range, only byte-proven media inline) |
| POST | `/api/files/upload` | `gateway.files.api_files_upload` |
| POST | `/api/files/mkdir` | `gateway.files.api_files_mkdir` |
| POST | `/api/files/write` | `gateway.files.api_files_write` |
| POST | `/api/files/delete` | `gateway.files.api_files_delete` |
| POST | `/api/files/transfer` | `gateway.files.api_files_transfer` |
| GET | `/onboarding` | `gateway.onboarding_host.onboarding_page` |
| GET | `/api/onboarding` | `gateway.settings.api_onboarding` |
| POST | `/api/onboarding/complete` | `gateway.onboarding.api_onboarding_complete` |
| POST | `/api/onboarding/subagents/preview` | `gateway.onboarding.api_onboarding_subagents_preview` |
| GET | `/api/settings` | `gateway.settings.api_settings_get` |
| POST | `/api/settings` | `gateway.settings.api_settings_post` |
| POST | `/api/settings/secret` | `gateway.settings_secrets.api_settings_secret` |
| GET | `/api/review-pool` | `gateway.settings.api_review_pool` |
| GET | `/api/claudexor/status` | `gateway.claudexor_accounts.api_claudexor_status` (daemon adds nullable `last_exit` and `memory`; saved host exit observation plus a read of the already-running engine, never a wake) |
| POST | `/api/claudexor/quota/refresh` | `gateway.claudexor_quota.api_claudexor_quota_refresh` |
| POST | `/api/claudexor/wake` | `gateway.claudexor_accounts.api_claudexor_wake` |
| POST | `/api/claudexor/login` | `gateway.claudexor_accounts.api_claudexor_login` |
| GET | `/api/claudexor/login/{job_id}` | `gateway.claudexor_accounts.api_claudexor_login_job` |
| DELETE | `/api/claudexor/login/{job_id}` | `gateway.claudexor_accounts.api_claudexor_login_job` |
| POST | `/api/claudexor/login/{job_id}/input` | `gateway.claudexor_accounts.api_claudexor_login_job` |
| POST | `/api/claudexor/login/{job_id}/reconcile` | `gateway.claudexor_accounts.api_claudexor_login_job_reconcile` |
| DELETE | `/api/claudexor/credential-profiles/{harness}/{profile_id}` | `gateway.claudexor_accounts.api_claudexor_credential_profile` |
| PATCH | `/api/claudexor/credential-profiles/{harness}/{profile_id}` | `gateway.claudexor_accounts.api_claudexor_credential_profile` |
| POST | `/api/owner/runtime-mode` | `gateway.settings.api_owner_runtime_mode` |
| POST | `/api/owner/auto-grant` | `gateway.settings.api_owner_auto_grant` |
| POST | `/api/owner/context-mode` | `gateway.settings.api_owner_context_mode` |
| POST | `/api/owner/safety-mode` | `gateway.settings.api_owner_safety_mode` |
| POST | `/api/owner/skills/{skill}/presence-runtime` | `gateway.presence_settings.api_owner_skill_presence_runtime` |
| POST | `/api/owner/capability-ack` | `gateway.settings.api_acknowledge_capability` |
| GET | `/api/ui/preferences` | `gateway.ui_preferences.api_ui_preferences_get` |
| POST | `/api/ui/preferences` | `gateway.ui_preferences.api_ui_preferences_post` |
| GET | `/api/ui/i18n` | `gateway.ui_i18n.api_ui_i18n_get` — the install's interface language and its translation memory; a read never starts generation |
| POST | `/api/ui/i18n/language` | `gateway.ui_i18n.api_ui_i18n_language_post` — choose the language (locked owner-settings write of `OUROBOROS_UI_LANGUAGE`; broadcasts `ui_language_changed`) |
| POST | `/api/ui/i18n/missing` | `gateway.ui_i18n.api_ui_i18n_missing_post` — strings the renderer could not translate, shape-filtered and bounded, queued for the generator |
| POST | `/api/ui/i18n/import` | `gateway.ui_i18n.api_ui_i18n_import_post` — an enterprise or community memory file (schema 1); owner pins stay |
| GET | `/api/ui/i18n/export` | `gateway.ui_i18n.api_ui_i18n_export_get` — the memory file as a download |
| POST | `/api/ui/i18n/regenerate` | `gateway.ui_i18n.api_ui_i18n_regenerate_post` — drop generated entries, keep owner and imported ones |
| GET | `/api/desktop/autostart` | `gateway.desktop_autostart.api_desktop_autostart_get`: host OS `{state}`, plus `reason` when unavailable |
| POST | `/api/desktop/autostart` | `gateway.desktop_autostart.api_desktop_autostart_post`: exactly `{enabled: boolean}`, returns observed `{state}`; 400 invalid body, 409 unavailable, 500 OS failure; public `owner_audit`, no settings mirror |
| GET | `/api/desktop/background` | `gateway.desktop_autostart.api_desktop_background_get`: keep running after the window closes, `{state: unavailable\|off\|on}` plus `reason` when unavailable |
| POST | `/api/desktop/background` | `gateway.desktop_autostart.api_desktop_background_post`: exactly `{enabled: boolean}`; writes `OUROBOROS_DESKTOP_KEEP_RUNNING` under the document lock; 400/409 `saved=false`, 503 contended lock; `owner_audit` |
| GET | `/api/model-catalog` | `gateway.models.api_model_catalog` |
| POST | `/api/openai-compatible/models` | `gateway.models.api_openai_compatible_models` |
| POST | `/api/providers/test` | `gateway.models.api_provider_test` |
| POST | `/api/tasks` | `gateway.tasks.api_tasks_create` |
| GET | `/api/tasks` | `gateway.tasks.api_tasks_list` |
| GET | `/api/tasks/{task_id}` | `gateway.tasks.api_task_get` |
| GET | `/api/tasks/{task_id}/events` | `gateway.tasks.api_task_events` (legacy integer rank) |
| POST | `/api/tasks/{task_id}/events` | `gateway.tasks.api_task_events` (read-only v2 cursor) |
| GET | `/api/tasks/{task_id}/artifacts/{name}` | `gateway.tasks.api_task_artifact` via task_archive/task stores: bare name=top-level file, `?relpath=`=nested file, `?archive=<dir>`=ZIP (artifact_archives). Digests bind files/ZIP members: drift=409 artifact_identity_changed (digest), failed capture=404 artifact_unverified; x-ouroboros-artifact-identity=verified/unmeasured. Delegated JSONL `?source=` requires source_handles/delegated_activity content-addressed name, basename, confined descent and digest. File/chat-media/ZIP/delegated JSONL reads return 503 without confined-open support (issue #1297, Windows); result-published review sources retain read_task_result_source_bytes; a replaced host `acceptance-<sha256>.json` record resolves by host_acceptance_source membership. |
| POST | `/api/tasks/{task_id}/cancel` | `gateway.tasks.api_task_cancel` |
| POST | `/api/tasks/{task_id}/hurry` | `gateway.tasks.api_task_hurry` |
| POST | `/api/tasks/{task_id}/pause` | `gateway.task_pause.api_task_pause`: `{request_id}`, text-free; state requested/paused/released = pausing/settled/resumed replay; 202 `latch_pending` retries ID (§6) |
| POST | `/api/tasks/{task_id}/continue` | `gateway.task_continue.api_task_continue`: `{action_nonce}` → `TaskContinueResponse`; `ContinuationOffer` in detail/history: bound=retry nonce, admitted=successor (§6) |
| POST | `/api/tasks/{task_id}/resume` | `gateway.tasks.api_task_resume` |
| POST | `/api/decisions` | `gateway.tasks.api_decision_answer` |
| GET | `/api/schedules` | `gateway.schedules.api_schedules_list` |
| POST | `/api/schedules` | `gateway.schedules.api_schedules_upsert` |
| POST | `/api/schedules/{schedule_id}/action` | `gateway.schedules.api_schedules_action` |
| DELETE | `/api/schedules/{schedule_id}` | `gateway.schedules.api_schedules_delete` |
| POST | `/api/command` | `gateway.control.api_command` |
| POST | `/api/reset` | `gateway.control.api_reset` |
| GET | `/api/git/log` | `gateway.control.api_git_log` |
| POST | `/api/git/rollback` | `gateway.control.api_git_rollback` |
| POST | `/api/git/promote` | `gateway.control.api_git_promote` |
| GET | `/api/update/status` | `gateway.control.api_update_status` |
| POST | `/api/update/check` | `gateway.control.api_update_check` |
| POST | `/api/update/preflight` | `gateway.control.api_update_preflight` |
| POST | `/api/update/apply` | `gateway.control.api_update_apply` |
| GET | `/api/cost-breakdown` | `gateway.history.make_cost_breakdown_endpoint` (the router's import path; the factory body and `_ACCOUNTING_SUMMARY_FIELDS` live in `gateway.cost_breakdown`) |
| GET | `/api/evolution-data` | `gateway.control.api_evolution_data` |
| GET | `/api/projects` | `gateway.projects.api_projects_list` |
| POST | `/api/projects` | `gateway.projects.api_projects_create` |
| POST | `/api/projects/from-task` | `gateway.projects.api_project_from_task` |
| POST | `/api/projects/{project_id}/update` | `gateway.projects.api_project_update` |
| POST | `/api/projects/{project_id}/delete` | `gateway.projects.api_project_delete` |
| GET | `/api/fs/dirs` | `gateway.projects.api_fs_dirs` |
| GET | `/api/chat/history` | `gateway.history.make_chat_history_endpoint` (saved web ingress: `ingress_accepted: true`; legacy rows omit it. Registry failures leave unclassified non-Main rooms at `latest_message: null`) |
| GET | `/api/logs/{name}` | `gateway.logs.api_logs_tail` |
| POST | `/api/chat/upload` | `gateway.files.api_chat_upload` (+ attachment `view`) |
| DELETE | `/api/chat/upload` | `gateway.files.api_chat_upload_delete` (pending only; else 409) |
| POST | `/api/local-model/start` | `gateway.models.api_local_model_start` |
| POST | `/api/local-model/stop` | `gateway.models.api_local_model_stop` |
| GET | `/api/local-model/status` | `gateway.models.api_local_model_status` |
| POST | `/api/local-model/test` | `gateway.models.api_local_model_test` |
| POST | `/api/local-model/install-runtime` | `gateway.models.api_local_model_install_runtime` |
| GET | `/api/mcp/status` | `gateway.mcp.api_mcp_status` |
| POST | `/api/mcp/refresh` | `gateway.mcp.api_mcp_refresh` |
| POST | `/api/mcp/test` | `gateway.mcp.api_mcp_test` |
| WS | `/ws` | `gateway.ws.ws_endpoint` |
| STATIC | `/static/*` | `server.NoCacheStaticFiles` |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/identity` | `gateway.host_service._api_identity` |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/tools/schemas` | `gateway.host_service._api_tool_schemas` |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/allocate-internal` | `gateway.host_service._api_allocate_internal` |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/inject` | `gateway.host_service._api_chat_inject` |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/operations/{operation_ref:path}` | `gateway.host_service._api_chat_operation` (the calling skill's own accepted message: pending, running with its task or turn, the durable answer, or the terminal task status) |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/cancel` | `gateway.host_service._api_chat_cancel` (the existing cancellation owner on work that message started; a typed outcome, never a cancellation that did not happen) |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/chat/decision` | `gateway.host_service._api_chat_decision` (the `task_decision.answer_decision` ingress relayed for a transport skill) |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/ui/language` | `gateway.host_service._api_ui_language` (the owner's interface-language choice relayed for a transport skill under `inject_chat`; the `ui_i18n.choose_language` writer the browser's `/api/ui/i18n/language` uses, typed refusals passed through) |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/turn` | `gateway.host_service._api_presence_turn` |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/delivery` | `gateway.host_service._api_presence_delivery` |
| GET | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/work/{work_ref}` | `gateway.host_service._api_presence_work` |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/presence/work/{work_ref}` | `gateway.host_service._api_presence_work` (retain attributed transport queue observations) |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/ui/ws-message` | `gateway.host_service._api_ws_message` |
| POST | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/notify` | `gateway.host_notify._api_notify` (`notify_owner` grant: one signed `skill_notice` System row in the owner's chat; no model turn) |
| WS | `127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}/events` | `gateway.host_service._ws_events` |

`server.py` owns startup/lifespan/static files; `gateway/*` owns HTTP/WS. Update status, log-tail and schedule-list reads run off-loop; passive status generates no letter.

Cancel accepts optional `cascade`, `stop_policy` and `stop_action_id` (≤200 characters). Reuse identity for the same action after uncertain response; later Stop uses a new ID. Legacy omission keeps existing ingress semantics without exact replay assurance.

### WebSocket protocol

`/ws` delivers live browser events; queue/task/Project/review/skill/settings/cost/update owners persist truth, reconstructed through REST/history after reconnect. `gateway/contracts.py` owns frozen envelopes and Python/JS parity; `gateway/ws.py` checks JSON objects, owned extension namespaces and non-empty built-in `chat`/`command` payloads. Chat acceptance (ingress lock → canonical row → enqueue → echo) runs off-loop through `gateway._helpers.run_sync_to_completion`. Each socket chains chats in receive order without awaiting them in its receive loop, so Panic/Restart commands on that socket remain inline queue puts even behind a held chat. Disconnect/cancellation waits for received chats to settle (`settle_to_completion`). Authentication above governs admission; public sockets never receive Host Service or owned-daemon tokens.

The browser constructs one socket for the whole SPA. Feature modules subscribe before connection, and the initial complete Project chat-id set is fetched before the first open so an early Project frame cannot be mistaken for Main traffic. `ws.on(type, listener)` stores listeners in insertion-ordered sets and returns a disposer; emission uses a listener snapshot, so a listener added during dispatch does not receive the current frame and disposing one listener cannot skip its neighbor. Every decoded frame first reaches the generic `message` event and then its type-specific event, which lets Widgets consume reviewed namespaced events without duplicating the socket.

A browser `chat` frame contains the owner text and may add `sender_session_id`, `client_message_id`, `force_plan`, uploaded attachment references, `chat_id`, `project_id`, and `client_surface` — raw sending-surface observables measured at SEND time because the pywebview bridge appears asynchronously after load. The gateway normalizes that payload through `client_surface.normalize_client_surface`, stamps host `received_at` at socket receipt, and persists it on the canonical inbound row. Every transport passes the one common enqueue (`LocalChatBridge.enqueue_local_message`), which keeps that earlier host stamp or an accepted row's time, else stamps now; each update carries it, and so does a host channel fact that lacked one, so a row written at dequeue measures intake lag as `ts − client_surface.received_at` (no `logged_at`). The fact is distinct from the `transport` dict (transport is chat-scoped reply routing; the surface fact is per-message provenance). The fact is assembled at its PRODUCER, never inferred at render (the per-producer stamp catalog and closed-key bound: `ouroboros/client_surface.py`); synthetic A2A chats stamp no owner surface (machine traffic never wears one); machine producers stamp nothing (`client_surface` is a reserved schedule-template key rejected at admission); promotion/steering CARRY the originating owner turn's fact. The loop injects a surface note only when sending-surface identity changes within an attempt (viewport excluded — a resize is not a device change). Absence is an honest gap. The client generates a message id when absent and uses it to reconcile its pending bubble, the echoed canonical user row, routing annotations, and mailbox retries; a successful browser `send()` means only that the current socket accepted the frame, not that a task was durably admitted. A frame whose id already names an accepted row rejoins it when its words and ordered attachment content match (no second row; the accepted echo is re-sent) and is refused otherwise. That row comes from the named ingress's process-local index of the chat chain (`message_ingress._AcceptedIds`: one cold fold per host process, then only appended bytes; a rotation folds its remainder, a changed prefix folds again): no send replays history, no historical id is new. A rejoin dispatches only on this process's positive proof that the row's append raised before dispatch (`_UNDISPATCHED`; history marks the row `ingress_undispatched`): the first such retry hands it over once. Only positive process-local call-entry evidence marks `ingress_dispatched` on echo/history; the row's `ingress_process` stamp alone proves acceptance, not dispatch: this live process's row not yet entered or refused (an append still returning, an echo before a deferred dispatch) says `ingress_pending`. After a restart, or once dispatch began, nothing replays; row and uploads are kept. Absent chat history names nothing; history that cannot be read is unknown, so the frame is refused (the initialization notice) before anything is claimed or written. The owner echo is the accepted row as history replays it: its text, views and, when the host wrote the text because no words were sent, `text_placeholder`.

Ordinary frames sent while disconnected enter a process-local queue capped at 100 entries (oldest dropped), flushed in order after reconnect and lost on page reload — not a second durable outbox. Attachment messages deliberately set `queue:false`: uploads occur immediately before send, so retaining only the socket frame would leave unowned temporary files; on socket loss before the send Chat refuses the message, cleans uploaded temporaries best-effort, and retains the staged files for explicit retry. Once the socket took the frame, the composer clears and the bubble shows it, but the frame stays under its `client_message_id` (`chat_attachments.createUnconfirmedSends`, no count bound) until that id's `ingress_accepted` echo or history row settles it: `Input saved` if `ingress_dispatched` (or `ingress_pending`, kept for the next fact), `Saved, not delivered.` (kept) if `ingress_undispatched`, else `Saved; delivery not confirmed.` (dropped, no replay). The tab's sessionStorage keeps its words, id, routing and upload refs (no bytes or `client_surface`) across reloads (`ws.js` `sha-unknown` too) and room teardown, as a released or rebuilt bubble does; `hasPendingWork` keeps a Project room holding one hidden. A socket close, an `initialization_notice`, or a reload's first history read not showing it saved offers "Send again" (same frame and id; never automatic) and "Discard" (forgets the tab's copy, never an upload); a storage refusal or unreadable kept copy is shown. The file-name tail the model reads is appended to every attachment message except the exact `/restart`, which stays the host's command (its files ride that row). Text-only frames keep the queue contract above.

For a chat frame, the gateway validates uploaded filenames as basenames confined under the upload root, exposes the first eligible bounded image (proven from its bytes through the confined open) as native image content, forwards the complete validated attachment set as task-staging metadata — each spec bound to its measured `size`/`sha256`, which staging verifies while copying, so bytes swapped under the path afterwards are rejected rather than staged — and calls the local message bridge with the exact thread, Project, sender-session, client-message, and planning facts. The web owner identity is fixed: `chat_id` selects a thread and cannot mint an external owner identity. If the bridge is not initialized, the socket returns a visible typed System row (`initialization_notice`) rather than accepting the message silently.

A built-in `command` frame carries a slash command and enters the same bridge with rebroadcast disabled; runtime command routing, owner authorization, queue authority, and typed outcomes remain outside the socket module. Main header controls therefore reuse the ordinary command contract for Restart, Panic, review, evolution, and background consciousness; Panic is sent only after the shared dialog returns the strict confirmed boolean. The socket does not infer intent from command-looking prose.

Built-in outbound envelopes: `chat`, `photo`, `video`, `document`, `typing`, `log`, `heartbeat`, `extension_lifecycle`, `message_annotation`, `projects_changed`, `task_named`, `update_status_ready`, and `update_progress_changed`. The latter only invalidates the process-local update observation; it is never boot-completion proof. Chat progress may carry task lineage, role, requested/effective model lane, delegated route, terminal execution evidence, review projection, cancellation eligibility, outcome axes, artifact references, and nullable cost/finality fields — additive presentation facts; consumers must not infer a missing execution receipt, cost, or task result from the absence of one optional field.

Project chat, typing, media and log frames carry `chat_id`; each Project panel consumes its own thread; Main admits Project question mirrors, the host-stamped Project lifecycle rows (`project_handoff`, `project_started`, `project_completion_summary`; §3 Project handoff receipts) and a Project root's `main_notice`. `projects_changed` carries a new chat id so every tab can extend its fan-out set before fetching the registry; when even that ordering loses the race, the server-stamped `project_thread` marker on the frame itself keeps Main from adopting it — set once at the message-bus broadcast choke from the registry (a membership lens, never a numeric range, so external transport ids such as Telegram stay unstamped) and enforced by Main's fan-out gate (`chat_activity.mainThreadAccepts`). Task-scoped LOG events acquire their final chat id at supervisor ingress: worker diagnostics carry only their own `task_id`, and `supervisor/log_addressing.py::address_task_event` stamps the audience from host-attested truth (the precedence chain lives in its docstring; an explicit event chat_id of 0 is the hidden partition, `HIDDEN_CHAT_ID`, never "missing"); direct turns carry their chat BY VALUE, stamped at the producer, because the registry entry dies with the turn while queued events drain later. Addressing is honest — an A2A row keeps its true audience, suppressed only at the broadcast choke (`push_log`) so machine traffic never reaches the browser; the same addressing runs in the server-process append sink and at every supervisor handler owning a suppressed type's explicit push, and a genuinely unaddressable event keeps the legacy chat-0 frame. `message_annotation` updates one canonical owner message without creating another bubble — a refused routing act's frame and its replayed annotation carry the host's `cause` sentence (the machine `reason` stays on the durable row), and the picker's 409 `dispatch_rejected` body carries `cause` beside `reason`; `task_named` updates a card only where that task already exists. Media/document consumers validate MIME, base64, and download-route shapes before building browser URLs.

Extension WebSocket traffic is structurally namespaced by `extension_loader.extension_surface_name()` so an extension cannot shadow a built-in type. On each incoming extension frame the gateway resolves the owning skill and reconciles whether its extension is still desired, reviewed, granted, enabled, and live. A missing or failed handler returns a visible log frame. Out-of-process handlers execute in their extension child off the event loop; in-process handlers first record the required execution/cost disclosure. A non-`None` result returns as `<request-type>.reply`; exceptions become typed error log frames rather than terminating the socket loop.

Server broadcasts snapshot the connected-client list and send to all clients concurrently, so one slow or half-open browser cannot head-of-line-block delivery to every other tab. Failed sends remove only the dead clients and append a durable `broadcast_partial_failure` event; the original domain event stays owned by its durable producer. Restart shutdown closes remaining clients best-effort with code 1012 so they enter the ordinary reconnect path.

The browser reconnects with bounded exponential delay, shows the reconnect overlay, and resets the delay after a successful open; a watchdog closes an apparently open connection after 45 seconds without any inbound frame, so heartbeat traffic proves stream liveness rather than task progress. One served-SHA decision (`ws.js decide()`: keep / reload-changed / reload-unknown) governs both recovery paths so a transient drop cannot destroy in-page state: a changed or no-longer-provable SHA reloads (a restarted server must not keep old JavaScript or CSS alive in PyWebView), an unchanged SHA keeps the page and its queued outbound messages, and an unversioned `/api/state` stays on keep when no non-empty SHA was ever remembered — the owner-selected default under uncertainty, accepting possibly-stale assets as the disclosed tradeoff. A non-OK or failed post-open read is no answer: it never reloads, and the read repeats with a growing pause until the server answers or that socket is gone, because a server seconds into its life can answer 500. While the socket stays down, delayed recovery probes consult `/api/state` without adopting the served SHA; a 200 whose body is not a parseable object counts as a failed probe, probes are single-flight and generation-scoped per disconnect episode, and after several consecutive healthy probes with the socket still down, one forced reload per episode remains as the fuse for a stale browser runtime.

Each Chat instance handles `open` by resynchronizing archive-aware durable history and `close` by withdrawing online/accounting presentation; reconnect deduplication covers overlap between live frames and REST replay, Logs merges the same way, and large history parsing runs off the server event loop. Delivery is live plus replay, not a promise that every transient frame is persisted: durable chat rows, task results, queue snapshots, Project revisions, review ledgers, cost ledgers, and lifecycle state remain the recovery authorities.

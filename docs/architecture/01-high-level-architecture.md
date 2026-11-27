# 1. High-Level Architecture

This part gives me the shape of my body: how the launcher, server, web client, supervisor and agent core fit together, the gateway and CLI/headless boundaries, the devtools boundary, the runtime topology (desktop, systemd, Android, platform substrate) and where my data lives under the data root. Launcher recovery needs an importable shell (§2). The server admits, routes, observes and kills, doing work bounded by logical change; a focus of my mind runs in a custodied worker slot, or on a server thread for a live call or a degraded inline turn (§5 Conversation lane). The curated, hand-kept module directory is §1.1 (`docs/architecture/01-module-directory.md`).

### Code navigation

`query_code` returns source evidence and local syntax facts, never semantic binding. Operations, limits and rationale: [code navigation contract](../code-navigation.md).

### Devtools boundary

`devtools/` stays outside runtime imports and package discovery; review applies and outputs stay external. Throwaway benchmarks stamp `.ouroboros_isolated_benchmark` in their data root to disable JSONL rotation (`supervisor/state.py`). Each devtool's README keeps its module map and methodology: `devtools/benchmarks/cybergym/` pauses dispatch on typed gates (`cybergym_dispatch.py`); `devtools/benchmarks/cowork_bench/` keeps scoring authority in its audit (`METHODOLOGY.md`); `devtools/benchmarks/osworld/` splits runners from their owner leaves; `devtools/macos_wkwebview/` is the optional system-WKWebView composer check. `devtools/e2e_live/` is the live E2E stand (DEVELOPMENT "Live E2E stand").

### Gateway Boundary v1

`ouroboros/gateway/` is the sole inbound browser/CLI boundary and `ouroboros/gateways/` holds thin outbound adapters without business logic; `files.py` and `host_service.py` stay separate trust boundaries. Handlers call runtime owners and never duplicate queue/review/settings/lifecycle policy, keeping UI free of agent-body imports and runtime free of ad-hoc browser contracts. `GET /api/contract` publishes endpoint/shape indexes (`docs/CLIENT_CONTRACT.md`) so replacement UIs need no core source. Owner settings writes go through `gateway/owner_settings.py::_owner_update_settings` (lock acquisition is a precondition; timeout refuses before writing) and `CommitBoundary` marks `saved` on both sides so post-commit failures name their step (§7). Frontend: `web/modules/api_client.js` (§11); extensions: `/api/extensions/<skill>/…` and namespaced WS.

### CLI / Headless Boundary

`ouroboros.cli` (`ouroboros/cli.py` SSOT) is a gateway/queue client, not a second engine: streaming commands write answers, patches, results or JSONL to stdout and progress to stderr. `POST /api/tasks` admits managed roots (§4); CLI refuses non-`root` `delegation_role` (only `schedule_subagent` creates children). Admission reserves task id and slot under one queue lock off-loop, so a cancelled waiter never cancels an admitted task; unconfirmed persistence returns 503 `unconfirmed` and keeps the row. Attachments copy into the task drive before enqueue; stored refs grant no host-path authority.

Workspace tasks default to `memory_mode=forked`; `shared` refuses external workspaces and runs forked in project scope. Stored `memory_mode` records the request and `drive_root` where it runs, so isolation never relies on relabelling. Forked/empty drives live at `data/state/headless_tasks/<task_id>/data`: forks copy `identity.md`, `WORLD.md` and `registry.md`, plus global knowledge topics and index only for non-project forks; never dialogue, scratchpad, mailbox or history; identity/world/scratchpad writes still land on the canonical root, context keeps repo governance and canonical shared memory, and `memory_export.json` is explicit (never auto-merged).

`--detach` returns after admission; `--no-stream` polls. `ouroboros run` exits 0 only on `_is_terminal_success` (completed lifecycle, clean execution, no failed/degraded objective, finished artifacts—reply ≠ "deliverable exists"); `--patch`/`--patch-out` trust `workspace_patch.json` (omitted, no-op and failed patches stay distinct, §6). Partial cost awaits `_await_cost_finality`. Schedules enqueue ordinary tasks (§5): skill `scheduled_tasks` only suggest (`skill_schedule_suggestions.py`), blank timezones take the DST-aware system zone (fixed offset only if unrecoverable), and the schedule digest rides task and consciousness context.

`packaged_cli` bootstraps the managed repo via the embedded interpreter to run `ouroboros.cli` (explicit URLs win); packaged `server` refuses to bypass launcher lifecycle; loopback `run --start` opens the app unless its data-root server is ready. Builds may download Node, ripgrep and Git to bundle them; launch-time selection never downloads: skills use the reviewed Node, shipped or prepared, and wait if absent (§6), generic launches stay PATH-first (`process_interpreters.resolve_process_node`: healthy PATH keeps argv/env; missing/dead PATH permits attested bundled prepend, never in a non-local executor), and `search_code` enumerates policy-approved files before ripgrep so speed widens no authority. Bundled consumers share `bundled_resource_bases()` order (`OUROBOROS_BUNDLE_DIR`, frozen root, interpreter ancestors, source checkout); server/CLI children run the managed repo, never in-bundle modules, and ancestor lookup lets older launchers start newer checkouts. `embedded_python_env()` protects signed bundles: bytecode goes to `data/state/pycache.noindex` and user installs to `data/state/python-userbase` (`pip_install_target_args()` adds `--user` only there), outranking bundle `site-packages` and never pruned (recovery: delete and relaunch).

Workspace binding switches the contextual repo, never the system repo (BIBLE, prompts, review governance); `/api/tasks` and promotion share `workspace_admission.validate_workspace_root()`. Ordinary folders support file/process work; Git needs a worktree. Binding updates defaults, memory, lease, preflight and finalization without dropping top-level tools or Max Architecture. The workspace executor routes processes and is not a sandbox (`executor_ref` is host-owned, `network=none` holds only when implemented, processes enter custody). Preflight writes `workspace_preflight.json` (`tools_on_path`/`tools_missing_from_path` record `shutil.which` presence, not executability, frozen in durable task metadata; failure writes an error summary, never a fictitious artifact).

Directory delegated sessions submit `execution.workspaceKind=directory` on `agent` mode after negotiating `mutability.workspaceKinds` (`directory_strategy=direct|copy`, optional `scope_paths`): the engine owns copy/CAS/apply/discard while the host keeps before/after closure on the existing custody path (no second file engine; `delegate_directory.py`, §6), never judging file/GUI work from an empty text diff. Complete inputs live in the artifact store (`attachment_manifest` preview never substitutes for unreadable full immutable `attachment_manifest_ref`). Router continuations need explicit `predecessor_task_id` (`""` = fresh; §6). Self-modification, external workspace and genesis stay distinct; genesis `deliverable_manifest.json` gaps never fail a task; copy/ZIP captures need stable bytes; global/system installs stay policy-reviewed, and `sudo` runs non-interactively (`sudo -n`).

### Runtime topology

Host sign-in startup is an owner opt-in (This computer, §3) whose only authority is the OS registration (`desktop_autostart.py`: Windows HKCU `Run`; macOS LaunchAgent with `--launch-intent automatic` and no `KeepAlive`; Linux systemd unit for deb/rpm, XDG autostart for AppImage/tar; only packaged launchers exporting a new enough `OUROBOROS_APP_VERSION`; states `unavailable`, `off`, `on`, `other_copy`, `disabled_by_os`; Panic and saved-pause keep authority; `context.py` records `runtime_env.autostart` and `keep_running_after_close`). `OUROBOROS_DESKTOP_KEEP_RUNNING` is offered when `OUROBOROS_DESKTOP_BACKGROUND=1` (Windows/macOS); both on gives a hidden sign-in start with an indicator (§9).

`launcher.py` owns PID lock, bootstrap, server, presentation, restart and cleanup outside the managed repo; `server.py` is the self-editable inner runtime (readiness wait has no clock, §2). Opt-in systemd passes `--launch-intent automatic` so Panic survives, with no restart policy: the launcher owns restarts and the crash fuse ([packaging/systemd/README.md](../../packaging/systemd/README.md)). Spawn custody: POSIX starts a new session/group; Windows creates the server suspended in a kill-on-close Job and refuses without Job custody (only the shared daemon requests breakaway). `data/state/server_process.json` pins birth/command identity: signals re-prove identity, an unmeasurable or unconfirmed predecessor keeps its record and blocks the next generation, and forced cleanup/`kill_worker_tree` spare the daemon subtree (§9).

Every server entry (`server.main`, imported app lifespan) holds `<data>/state/server.lock` for life (`server_entrypoint.admit_data_root`), enforcing one server per data root regardless of repo, PID file or port (records and bindings stay evidence; pre-lock servers go undetected). Under the PID lock, each generation first runs `launcher_server_reaper.py` to reap this install's launcher-started strays on `/proc` hosts only (report-only elsewhere), never on panic or close.

Durable custody (`process_custody.py`; §9): `spawn_supervised()` ledgers `task|session|daemon` in `process_ledger.jsonl`; the reaper kills dead-owner entries by strict `(pid, start_time, cmd_sha256)` or launch identity, never command class, so dev and packaged installs coexist (daemons stay; companions reap on uninstall or foreign generation; log-only default: `process_would_reap`). `start_parent_lifeline()` group-suicides Python entrypoints on spawner death; multiprocessing children await the parent sentinel. At start (`_orphaned_at_start`), `ppid` 1 is live for a multiprocessing child only as a direct `spawn`/`fork` child of PID 1 (ignoring env, so forkserver adoption stays orphaned), and for a plain subprocess only when `OUROBOROS_PARENT_PID` names it.

Lifecycle: each generation resolves its predecessor, checks Panic and starts the server; ready means the child's published endpoint, proven against its handle, answers health. `RESTART_EXIT_CODE` refreshes metadata/deps (§2); `MAX_CRASH_RESTARTS` within `CRASH_WINDOW_SEC` stops auto-restart; Panic cleans up and exits. Linux checks `DISPLAY`/`WAYLAND_DISPLAY` before pywebview (GTK needs a live display; Qt trusts env alone because probing constructs `QGuiApplication`, which can abort); on failure the launcher still supervises the server, prints the authoritative URL and opens the system browser best-effort (outside custody).

#### Android host (experimental)

`android/` shares this repo, Python core, SPA and update channels: the full core runs in an ARM64 GNU/Linux chroot on a Magisk-rooted device with remote LLM APIs (physical qualification limited to Pixel 10a; install/recovery/bridge: `docs/ANDROID_INSTALL.md`).

`android/install.py` provisions `/data/local/ouroboros-phone/rootfs`; `android/provision/` pins upstream inputs and installed dependencies. Inside Linux, `/opt/ouroboros/{repo,data,venv,tools,signing,launcher}` separates mutable source, personal state, venv, tools, personal APK key and immutable seed. `bootstrap/enter-linux` creates private mounts in Android's network namespace and sets `oom_score_adj=0` so root survives and the kernel can recover from OOM. `core-control` is a one-shot start/status entry, never another restart loop: it starts the common launcher with `--host-update` and `--seed-bundle` (§2), keeping any Panic marker on automatic entry.

`bootstrap/update-host` builds the APK (`host/build.py`, persistent `signing/host.keystore`), writes `data/state/android_host.json`, serializes under `android_host.lock` and keeps candidates in `data/android-builds/`. First it calls `provision/runtime.py::ensure_platform`, marking groups `platform_preparing` in `android-sdk/installation.json` before mutating and clearing only after success so an interrupted run or rolled-back Git tree never reads as prepared. `launcher_bootstrap.update_external_host` applies its own `runtime_limits.py` timeout (independent of tool/harness timeouts); the launcher's HTTP readiness window starts only after core spawn, and core health alone claims no native success. Failed or stale native evidence leaves adoption unresolved; ordinary startup keeps the available core and logs `native_update_failed`. Delegated access follows §6: a requested access profile does not prove the kernel can execute its sandbox, so the observed route is qualified.

APK `MainActivity` hosts the SPA while `CoreService` and `BootReceiver` run in `:native` at the same app UID so a WebView renderer replacement never owns the SDK bridge; `BOOT_COMPLETED` uses automatic core entry after unlock (no second core supervisor). `AndroidBridge` exposes typed, bounded operations (packages, Intents, ContentResolver, location, opt-in accessibility, notifications) over a root/host-UID socket; `android-exec` keeps arbitrary owner-root argv capability. Locally built, personally signed APKs sit outside publisher-artifact attestation.

#### Platform substrate

`platform_layer.py` owns OS observations and lock/process primitives; callers own policy. Shared locks accept non-blocking probes on read-only descriptors. `kernel_file_locks_enforced` probes each directory once (never from failed live acquisition): unsupported locks/`ENOLCK` select a recorded name-only tier (`O_EXCL` + identity recheck, no kernel exclusion; callers like money may refuse); unprobeable dirs stay enforced for that call, other errors fail closed. Won locks require readable descriptor/path inode identities (unreadable is no proof); stale recovery skips age grace on confirmed owner death and keeps live writers (unknown keeps grace; permission denial proves no death). Refresh returns ownership (loss abandons work); release unlinks only held identity (POSIX under hold; Windows post-close with bounded retry; `LockFileEx` byte: `platform_layer.py`).

Birth ≠ PID presence: Linux uses boot-qualified ticks, ps time, then bare ticks (cross-boot collision limit; ledger keeps `start_time` + qualified sibling); macOS `process_identity` reads kernel `p_starttime`; Windows uses creation FILETIME and non-signalling handles. Unknown birth permits no kill. Win32 HANDLEs are full-width; false Job termination BOOL is failure. `shell_process.contained_popen` births Windows children suspended in non-breakaway Jobs; POSIX `WNOWAIT` retains leaders until `retire_tracked` measures no live members (Linux group probes use call-local `/proc` census). Root results return; owner/idle/stop retire; Linux `tracked_start` retries pending retirements; unknown retains custody and blocks idle-worker exit. Panic uses held groups/Jobs; group records transfer no Job handle. Bundles: CLI / Headless Boundary; Node: `node_runtime` (lazy export); TCP keepalive: §6 Context fitting.

### Data layout (`~/Ouroboros/`)

Default root (overrides: §7). This tree orients; `docs/PERSISTENCE.md` registers each durable entity's writer, format, schema, retention and reset.

```
~/Ouroboros/
├── .test-evidence/ ← taskless repo tests
├── repo/ ← self-modifying git repo (§2)
│   ├── ouroboros/
│   ├── supervisor/
│   ├── web/ ← SPA (`web/modules/`)
│   ├── docs/ ← books, CHECKLISTS.md, contracts, guides, inventories/
│   └── prompts/ ← SYSTEM.md, SAFETY.md, CONSCIOUSNESS.md
├── data/ ← runtime data root
│   ├── settings.json ← (§7)
│   ├── task_results/ ← `<id>.json` import/export (store-authoritative); artifacts/<task_id>/.artifact_manifest.json, source_handles/delegated_readonly_inputs/ (§6)
│   │   └── artifact_versions/<task_id>/ ← recovery history
│   ├── task_drives/<task_id>/ ← scratch/manifests
│   ├── task_trees/<root>/blackboard.jsonl ← pruned at root end
│   ├── state/
│   │   ├── state.json ← runtime state and cost projection (not monetary authority)
│   │   ├── queue_snapshot.json ← PENDING/RUNNING projection, `worker_pool_disabled_reason` (§5)
│   │   ├── peer_links.json ← links to my other Ouroboros (keys: peer_links/<link_id>/token.json)
│   │   ├── current.sqlite ← current-facts authority (`.exported-<UTC>` retired)
│   │   ├── usage.sqlite ← monetary authority (docs/USAGE_STORE.md)
│   │   ├── usage_attempts.jsonl ← imported journal/downgrade export
│   │   ├── usage_attempts.quarantine.jsonl ← proven-corrupt rows
│   │   ├── usage_import_watermark.json
│   │   ├── storage_measurement.json ← last two counts (not deletion authority)
│   │   ├── test-durations.json
│   │   ├── request_wire_compatibility.json ← exact-route wire evidence
│   │   ├── capability_evidence.json
│   │   ├── openrouter_models.json
│   │   ├── extra-ca-bundle/<digest>.pem ← merged extra CA
│   │   ├── process_ledger.jsonl ← (Runtime topology)
│   │   ├── installation_scope.json
│   │   ├── task_scopes/
│   │   ├── panic_sweep_pending.json
│   │   ├── obligations/ ← set import/export
│   │   ├── migrations.json ← schema generations, dep fingerprint
│   │   ├── body_adoption/ ← handoff, switch.py, installation_stop.py, lock
│   │   ├── owned_processes.json ← ownership set, pending stops (§9)
│   │   ├── server.lock ← server-lifetime DATA lock; never unlinked
│   │   ├── server_port ← HTTP port handoff
│   │   ├── server_port.bindings.json ← endpoint snapshot (not grant/custody)
│   │   ├── server_process.json
│   │   ├── advisory_review.json ← retired rows kept as stored
│   │   ├── code_intel/<repo_key>/inventory.json ← no source
│   │   ├── evolution_metrics_cache.json
│   │   ├── evolution_campaign.json ← objective, progress, budget
│   │   ├── evolution_checkpoints.jsonl
│   │   ├── post_task_evolution_request.json ← idle-tick promotion signal
│   │   ├── post_task_evolution_counter.json ← per-drive `every_n` counter
│   │   ├── scheduled_tasks.json ← cron/one-shot, `kind:"notify"` (§5)
│   │   ├── claudexor_rotation_provisioning.json
│   │   ├── claudexor_route_health.json
│   │   ├── subscription_census.json ← `.request` triggers read
│   │   ├── subagent_last_delegation.json ← not dispatch authority
│   │   ├── update_letter.json
│   │   ├── update_material.json ← checked range for writer
│   │   ├── projects.json ← durable tombstones
│   │   ├── projects.json.committed ← witness/copy; freshness: `projects.json.lock` (§6)
│   │   ├── project_task_bindings.json
│   │   ├── ui_preferences.json ← layout prefs, seen-revision ACKs
│   │   ├── i18n/<tag>.json ← beside `<tag>.pending.json`
│   │   ├── cancel_intents.json
│   │   ├── terminal_deliveries.json ← dedupe, receipts, outbox
│   │   ├── extension_companions.json
│   │   ├── extension_reconcile/
│   │   ├── review_continuations/ ← + corrupt/, archived/
│   │   ├── review_migrations/ ← `<ts>-slots-to-pool.json` rollback snapshots
│   │   ├── workspace_executor_processes/
│   │   ├── headless_tasks/<task_id>/data ← forked/empty child drives
│   │   ├── custody_staging/ ← unserved settlement copies
│   │   ├── custody_trash/ ← settled drives, released files
│   │   ├── betterleaks/ ← explicit installer
│   │   ├── cx/ ← Claudexor: immutable `<version>-<sha12>/`, retained `node/<version>-<platform>.g-*`, install.lock
│   │   ├── skill_grant_receipts.json ← (§13)
│   │   └── skills/<name>/ ← per-skill state (PERSISTENCE.md)
│   ├── claudexor/ ← owned Claudexor home (`CLAUDEXOR_CONFIG_DIR`)
│   ├── memory/
│   │   ├── identity.md
│   │   ├── scratchpad.md ← rendered from `scratchpad_blocks.json`
│   │   ├── dialogue_blocks.json ← with `dialogue_summary.md`
│   │   ├── dialogue_meta.json ← chronicle import cursor, nominations
│   │   ├── chronicle/ ← append-only `records.jsonl` authority, disposable `index.sqlite3`
│   │   ├── WORLD.md
│   │   ├── knowledge/ ← `patterns.md`, `improvement-backlog.md`
│   │   ├── deep_review.md
│   │   ├── registry.md ← optional hand-written source registry (`memory_map`); absent means nothing was recorded
│   │   └── owner_mailbox/
│   ├── projects/<id>/knowledge/
│   ├── observability/ ← private CAS: `blobs/`, `calls/<task_id>/`, `packs/<task>.xz`
│   ├── services/<task_id>/ ← `<service>.log`, `kept_outputs/` (§9)
│   ├── logs/
│   │   ├── chat.jsonl ← frozen at cutover
│   │   ├── progress.jsonl ← frozen at cutover
│   │   ├── chat_annotations.jsonl
│   │   ├── events.jsonl ← lifecycle, `llm_round`/`llm_usage`, errors
│   │   ├── tools.jsonl
│   │   ├── supervisor.jsonl ← workers/supervisor, cancel trail
│   │   └── task_reflections.jsonl ← early reflections, late-settlement rows
│   ├── archive/ ← rotated logs, rescue snapshots, archived repos; never GC'd
│   └── uploads/
├── projects/ ← genesis (`OUROBOROS_SUBAGENT_PROJECTS_ROOT`), never GC'd
├── subagent_worktrees/ ← (`OUROBOROS_SUBAGENT_WORKTREE_ROOT`), GC'd; `copies.noindex/`
├── Deliverables/ ← bare `user_files` (`OUROBOROS_DELIVERABLES_ROOT`), never GC'd
└── ouroboros.pid ← auto-released on crash
```

`scripts/regenerate_inventories.py --show data-layout` probes each entry's last literal segment against repository paths and runtime literals, not full path ownership.

---

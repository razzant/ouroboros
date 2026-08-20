# Persistence owner map

Every durable file and directory the runtime reads or writes, with the module that
authors it, the module that reads it *for behaviour*, and who — if anyone — ever
removes it.

This is an **evidence document**, not a contract. It states what the code at this
commit does. Where it disagrees with the `docs/ARCHITECTURE.md` "Data layout" tree,
§16 records the disagreement instead of silently preferring one; the tree is a
reader's orientation and this map is the derivation.

## Method

Every row was derived from code, not from the layout tree. For each candidate path:

1. find the **path constructor** — the `pathlib` join or the helper that returns it;
2. find the **actual write** — `.write_text` / `.write_bytes` / `open(..., "w"|"a"|"ab")`
   / `json.dump` / `atomic_write_json` / `write_text_atomic` / `replace_atomic`
   (`ouroboros/utils.py`) / `atomic_write_text` (`supervisor/state.py`) /
   `append_jsonl` (`ouroboros/utils.py`) / `update_json_locked` / `os.replace` /
   `shutil` / `mkdir`;
3. find the **deletion** — `unlink`, `rmtree`, `os.remove`, or a named prune/GC/rotate
   helper — and the rule it applies;
4. record the reader that *acts* on the content, separately from readers that merely
   project it to the owner or the UI.

The layout tree at `docs/ARCHITECTURE.md` was used only as a checklist of candidates.
Every entry it names was then re-derived, and the tree was scanned in the other
direction for paths the code produces but the tree omits.

## How to read a row

- **writer(s)** — every module that authors the bytes. Where one writer function has
  several independent *calling* modules, the row says so: a single write seam with
  many callers is a different risk than several writers.
- **authoritative reader** — the module whose behaviour depends on the content.
  `owner/UI only` means nothing in the runtime branches on it; `none` means nothing
  reads it at all, which is a finding, not a shorthand.
- **lifecycle** — who creates it and who removes it. `never pruned` is a derived fact
  (no `unlink`/`rmtree`/retention helper targets the path), not an omission.
- Paths are relative to the data root (`config.DATA_DIR`) unless stated otherwise.
  A task-scoped child drive carries the same tree under its own root.

---

## 1. Settings plane

`settings.json` is the owner's document, and the one durable file whose writer set is
already pinned twice. This section stays consistent with both pins rather than
deriving a different list:
`tests/test_settings_read_seam.py::test_the_three_settings_writers_are_exactly_these_three`
enumerates the AST-visible writers across the five modules that may touch it, and
`tests/test_runtime_mode_authorship.py::test_every_settings_writer_routes_through_the_shared_prologue`
requires every one of them either to route through `prepare_settings_for_persist` or
to be listed as exempt with a reason.

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `settings.json` (`config.SETTINGS_PATH`) | **document writers (3):** `ouroboros/config.py::save_settings` · `ouroboros/gateway/owner_settings.py::_owner_update_settings` · `ouroboros/packaged_cli.py::_save_settings`. **non-document writers (2, tripwire-exempt):** `ouroboros/context_mode_compat.py::normalize_and_persist_context_mode_compat` (one-window raw pair migration under the load lock) · `ouroboros/tools/registry_guard_process.py::_restore_owner_files` (immune-system rollback of snapshotted bytes) | `ouroboros/config.py::load_settings` → `load_settings_lock_held`; owner endpoints read through `ouroboros/gateway/owner_settings.py::_owner_read_settings_raw`. Both apply `ouroboros/config.py::normalize_settings_raw` *before* the defaults merge | created by any document writer; **never pruned**; removed only by `ouroboros/gateway/control.py::api_reset` | All three document writers share `config.serialize_settings`, so the bytes are identical. Staleness fence = `owner_settings.settings_document_digest`. The v7 D03 seam retired both start-time mutators (`launcher_onboarding::prepare_first_run_settings`, `server.py::lifespan`), so a fresh install's first bytes are the owner's own save. |
| `settings.json.lock` (`config._settings_lock_path`) | `ouroboros/config.py::_acquire_settings_lock` (`O_CREAT\|O_EXCL`) | none — mtime only, for staleness | released by `_release_settings_lock`; a lock older than the stale window is unlinked by the next acquirer | Lives beside `settings.json`, **not** under `state/`. A refused lock aborts the write (`TimeoutError` / `SettingsLockUnavailable`); reads proceed unlocked. |
| `settings.tmp`, `settings.json.tmp` | `config.py::save_settings` writes `settings.tmp`; `packaged_cli.py::_save_settings` writes `settings.json.tmp` | none | consumed by `replace_atomic` | The two writers disagree on the temp name, and neither shape matches the `.*.tmp.*` glob that `utils.sweep_stale_temp_files` reaps — a crashed write leaks its temp file permanently. |

Two further modules name `SETTINGS_PATH` without authoring a settings document and are
exempt for stated reasons: `ouroboros/usage_legacy_import.py::_legacy_snapshot` (hashes
it into the usage archive) and `ouroboros/tools/core.py::_data_write` (names it only to
*refuse* agent writes). `ouroboros/colab_bootstrap.py::write_colab_settings` generates a
document for **another** root and is exempt for that reason.

---

## 2. Runtime state plane — `state/`

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `state/state.json` | `supervisor/state.py::_save_state_unlocked` (via `save_state` / `update_state` / `init_state`); `supervisor/state.py::reset_per_task_budget` re-implements the write against a foreign root | `supervisor/state.py::load_state` | created on first miss by `_load_state_unlocked`; **never pruned** | `atomic_write_text` (truncate + fsync + `os.replace`) under `locks/state.lock`; an unlocked write logs loudly. |
| `state/state.last_good.json` | same writers, second write in the same call | `_load_state_unlocked` (recovery fallback only) | written with every `state.json` save; **never pruned** | The recovery path re-saves both on promotion. |
| `state/queue_snapshot.json` | `supervisor/queue_snapshot.py::persist_queue_snapshot` (sole writer) | `supervisor/queue_snapshot.py` restore path; `ouroboros/task_status.py::_load_queue_snapshot` | created at startup; whole-document overwrite; **never pruned** | Eight-plus read-only consumers across tools, gateway and skills. |
| `state/usage_attempts.jsonl` | `ouroboros/usage_ledger.py::_append_rows_locked` (sole writer; callers in `usage_accounting.py`: `reserve_attempt`, `_transition`, `settle_attempt`, `record_unmetered_external_dispatch`, `record_subscription_session`; the legacy-import plane appends through the same sole writer — `usage_legacy_import.py::_ensure_legacy_imported_locked`, see the watermark/archive rows below) | `ouroboros/usage_ledger.py::_read_records_locked` | created on first append; **never rotated or compacted** — the only truncation is `_quarantine_tail`'s `ftruncate` back to the validated prefix | The monetary authority. Append + fsync under `usage_attempts.lock`. `agent_startup_checks.py` warns on size and states the absence of rotation. |
| `state/usage_attempts.quarantine.jsonl` | `ouroboros/usage_ledger.py::_quarantine_tail` | none — forensic | created on first proven-corrupt tail; **never pruned** | Base64 raw row + reason, plus a `usage_ledger_tail_quarantined` row in `logs/events.jsonl`. |
| `state/usage_import_watermark.json` | `ouroboros/usage_legacy_import.py::_ensure_legacy_imported_locked` | `usage_legacy_import.py::_completed_import_watermark` | one-shot; **never pruned** | Idempotence gate; the imported source is archived under `archive/usage_import/<sha16>/`. |
| `state/server_port` | `ouroboros/server_entrypoint.py::write_port_file` | `launcher.py` (`_poll_port_file`); every other reader takes `config.PORT_FILE` | created at bind; **deleted by `launcher.py`** (lifecycle loop and `main`) | Write and delete live in different processes. Plain `write_text` — not atomic. |
| `state/server_process.json` | `launcher.py::_write_server_process_record` (+ `_update_server_process_record_port`) | `launcher.py::_server_process_identity_matches`; cross-checked by `ouroboros/agent_startup_checks.py::check_stray_server_processes` | created at spawn; deleted by `launcher.py::_cleanup_recorded_server_process` | Identity-proven delete only: a foreign record is never unlinked. |
| `state/advisory_review.json` | `ouroboros/review_state.py::_save_state_unlocked` (via `save_state` / `update_state`) | `ouroboros/review_state.py::load_state` | created on first save; **never pruned** (obligations coalesced in `_prepare_state_for_persistence`) | `atomic_write_json` under `locks/advisory_review.lock`; carries `state_version`. |
| `state/advisory_overrides.json` | `ouroboros/tools/review.py::_record_advisory_override` **and** `ouroboros/tools/claude_advisory_review.py::_record_bypass` | `ouroboros/review_evidence.py::build_review_projection` | self-bounded to the last 10 entries; the file itself is **never deleted** | Two independent writer modules through `update_json_locked`. |
| `state/deep_self_review_context.json` | `ouroboros/deep_self_review.py::run_deep_self_review` | owner/UI only | overwritten per review; **never pruned** | |
| `state/code_intel/<repo_key>/inventory.json` | `ouroboros/code_intelligence.py::build_code_inventory` | `ouroboros/code_intelligence.py::load_cached_inventory` | **never pruned** | `repo_key` is a hash of the absolute repo path, so every moved or renamed checkout leaks a directory that nothing reclaims. Schema v2; v1 is rejected on load. |
| `state/evolution_metrics_cache.json` | `ouroboros/utils.py::collect_evolution_metrics` | the same function | created on first `/api/evolution-data`; **never pruned** | Non-`schema:1` documents are ignored rather than migrated. |
| `state/evolution_campaign.json` | `supervisor/evolution_lifecycle.py` (`_write_evolution_campaign`, `link_evolution_rescue`, `record_evolution_commit`, `clear_pending_owner_report`) **and** `ouroboros/agent_startup_checks.py::verify_restart` | `supervisor/evolution_lifecycle.py` | created on campaign start; **never pruned** | CAS-guarded (stale-id and terminal-resurrection refusals) under `locks/state.lock`. |
| `state/evolution_checkpoints.jsonl` | `ouroboros/evolution_checkpoints.py::append_cycle_outcome_checkpoint`, `append_evolution_checkpoint` | `evolution_checkpoints.py::build_solve_capability_digest` (bounded read) | **never pruned** — the reader truncates its window, the file grows without bound | |
| `state/post_task_evolution_request.json` | `ouroboros/post_task_evolution.py::_write_request` (worker) | `ouroboros/post_task_evolution.py::apply_pending_request` (supervisor idle tick) | one-shot: unlinked on consume, and dropped by `drop_pending_request` when the owner-stop sentinel is set | Atomic publish, so a partial write is unobservable. |
| `state/post_task_evolution_counter.json` | `ouroboros/post_task_evolution.py::_counter_due` | the same function | **never pruned** | Monotonic `every_n` cadence counter; best-effort. |
| `state/scheduled_tasks.json` | `supervisor/queue_schedules.py::_write_scheduled_tasks` (via `upsert_scheduled_task`, `remove_scheduled_task`, `sync_skill_schedules`, `check_scheduled_tasks`) | `supervisor/queue_schedules.py::list_scheduled_tasks` | consumed one-shot records pruned by `supervisor/schedule_time.py::prune_consumed_once_records` at the unified GC retention | Whole-document rewrite each tick under the queue lock. |
| `state/claudexor_rotation_provisioning.json` | `ouroboros/claudexor_daemon.py::OwnedClaudexorDaemon._record_rotation_receipt` | **none** | **never pruned** | Write-only forensics: a durable receipt with no in-code consumer. |
| `state/projects.json` | `ouroboros/projects_registry.py::_save` (create / update / begin-fail-complete deletion) | `ouroboros/projects_registry.py::_load` | **never pruned** — tombstones are durable by design | Sidecar `projects.json.lock`; schema-versioned. |
| `state/project_task_bindings.json` | `ouroboros/projects_registry.py::_save_bindings` (via `bind_task_to_project`) | `projects_registry.py::project_binding_for_task` | **never pruned** | One-way enrichment only; lock-free rename-atomic read. |
| `state/ui_preferences.json` | `ouroboros/gateway/ui_preferences.py::api_ui_preferences_post` | `api_ui_preferences_get` | project cursors bounded in the same write | Owner-local layout state only. |
| `state/cancel_intents.json` | `ouroboros/cancel_intents.py` (`request_cancel`, `mark_finalize_control_drained`, `mark_intent_scope`, `claim_intent`, `release_claim`, `settle_intent`) | `ouroboros/cancel_intents.py::active_intents` | rows removed on settle; the file is never deleted | All six writes are `update_json_locked(..., strict_existing_dict=True)`; a corrupt file refuses loudly. Forensics go to `logs/supervisor.jsonl` and are never read back for state. |
| `state/terminal_deliveries.json` | `supervisor/terminal_delivery.py` (`register_delivery`, `register_pending_delivery`, `_bump_replay_attempts`) | `supervisor/terminal_delivery.py::pending_deliveries`, `already_delivered` | bounded by a pending cap and a replay cap | Both drop shapes (replay exhaustion, capacity eviction) are disclosed by `_disclose_exhausted_delivery` — never a silent drop. |
| `state/extension_companions.json` | `ouroboros/extension_companion.py::CompanionSupervisor._write_runtime_snapshot` | `launcher.py` window-close kill sweep | full-snapshot overwrite; **never pruned** | The writer and the acting reader are in different processes. |
| `state/extension_reconcile/*.json` (+ `failed/`) | `ouroboros/extension_reconcile_queue.py::request_extension_reconcile` (worker), `_mark_failed` | `extension_reconcile_queue.py::process_extension_reconcile_requests` (server lifespan) | markers unlinked on success; moved to `failed/` at the attempt cap. `failed/` is **never pruned** | Marker names carry a uuid so an old marker can never unlink a newer one. |
| `state/review_continuations/<task_id>.json` | `ouroboros/task_continuation.py::save_review_continuation` (from `ouroboros/tools/commit_gate.py`) | `task_continuation.py::load_review_continuation` / `list_review_continuations` | cleared by `clear_review_continuation`; retired by `retire_settled_continuations` after the settled window | Collision-safe rename; fail-open on any error. |
| `state/review_continuations/corrupt/` | `ouroboros/task_continuation.py::_quarantine_corrupt_continuation` | `_list_quarantined_corrupt_messages` | **never pruned** | Quarantine, not delete. |
| `state/review_continuations/archived/` | `ouroboros/task_continuation.py::retire_settled_continuations` | none — runtime-unread by design | **never deleted** | Timestamp-suffixed on name collision. |
| `state/workspace_executor_processes/*.json` | `ouroboros/workspace_executor.py::_register_process`, `_register_service_process` | `workspace_executor.py::_iter_process_records` | unlinked by `_forget_process`, `kill_all_foreground`, `_kill_durable_service_records` | Records are validated against a live command hash before any kill, so a reused PID is never targeted. |
| `state/pending_restart_verify.json` (+ `.claimed.<pid>.json`) | `ouroboros/tools/control_runtime.py` (agent restart) **and** `supervisor/evolution_lifecycle.py::request_evolution_restart` | `ouroboros/agent_startup_checks.py::verify_restart` — claims by rename | claim unlinked once the campaign mark is durable, otherwise renamed back; dead-PID claims reclaimed | Two independent writer processes, one arbitrating consumer. |
| `state/crash_report.json` | **no writer in the product tree** (only `tests/test_crash_report.py` creates one) | `ouroboros/agent_startup_checks.py::inject_crash_report`; `ouroboros/context_health.py` | never created by code; deliberately never deleted (the owner removes it) | Read-only contract with no producer: the `RECENT CRASH ROLLBACK` health invariant it feeds cannot currently fire. |
| `state/subagent_worktrees.json` | `ouroboros/subagent_worktrees.py::_save_registry` | `subagent_worktrees.py::_load_registry` | pruned by `prune_orphans` from `ouroboros/server_maintenance.py::_startup_worktree_prune` | A corrupt registry refuses rather than resetting. Its ops lock lives in the worktree root, not `state/`. |
| `state/worker_pids.json` | `supervisor/worker_pool_lifecycle.py::_record_worker_pids` (also called from `supervisor/workers.py`) | `supervisor/worker_pool_lifecycle.py::reap_orphaned_workers` | overwritten each spawn; **never pruned** | Legacy session-leader reap path; the SSOT is `process_ledger.jsonl`. Two overlapping process registries with two independent reapers. |
| `state/process_ledger.jsonl` | `ouroboros/process_custody.py::record_process`; compacted by `_rewrite_ledger` | `ouroboros/process_custody.py::_read_ledger`, `live_kept_service_pids` | survivors-only rewrite from `quiesce_custodied_services` and `reap_orphaned_processes` | The rewrite holds the JSONL append lock. Rows are fingerprinted by command hash. |
| `state/capability_evidence.json` | `ouroboros/capability_evidence.py::_save` (effort ceiling/floor, rejected params, token density, owner ack/revoke) | `ouroboros/capability_evidence.py::_load`; behaviour consumers `llm_capability_policy.py`, `loop_model_call.py`, `context_fit.py`, `reviewer_window.py` | only `revoke_owner_ack` removes rows; density pairs bounded; the file is never deleted | Keyed by route fingerprint. |
| `state/reviewer_slot_last_execution.json` | `ouroboros/reviewer_slot_config.py::record_reviewer_slot_executions` | `reviewer_slot_config.py::reviewer_slot_last_executions` | self-bounded to the newest N | UI projection only. |
| `state/reviewer_slot_api_fallback.json` | `ouroboros/reviewer_slot_config.py::_record_api_fallback_substitution` | owner disclosure surface only | overwritten; **never pruned** | Durable half of the reviewer API-fallback disclosure. |
| `state/subagent_last_delegation.json` | `ouroboros/subagents.py::record_last_delegation` (from `ouroboros/tools/delegate.py`) | `ouroboros/subagents.py::subagent_last_delegation` | overwritten; **never pruned** | Idempotent per `run_id`, so the timestamp is not re-stamped. |
| `state/headless_tasks/<task_id>/data/` | `ouroboros/headless.py::prepare_task_drive` (from `gateway/tasks.py`, `supervisor/worker_promotion.py`, `tools/control_scheduling.py`) | the child agent process (its own drive root) | pruned by `headless.py::prune_headless_task_drives` from `server_maintenance.py::_startup_prune_sweeps` | The physical home of the forked memory plane; see §9. |
| `state/acceptance_fence_acks/<token>.json` | `supervisor/events_worker_reports.py::_handle_acceptance_fence` | `ouroboros/agent.py::_await_acceptance_fence_ack` | the writer compacts the directory before each write; the reader unlinks its own ack | A write failure is loud: the worker fails closed. |
| `state/auth_secret.key` | `ouroboros/server_auth.py::_auth_secret` | the same function | created on first session mint; **never rotated or deleted** | Plain `write_text` + `chmod 0600` — not atomic. Losing it costs one re-login. |
| `state/panic_stop.flag` | `ouroboros/server_control.py::execute_panic_stop` **and** `server.py::_process_bridge_updates` | `supervisor/worker_chat_lane.py::auto_resume_after_restart` | consumed (unlinked) by the reader | Contents disambiguate panic from `owner_restart_no_resume`. |
| `state/owner_restart_no_resume.flag` | `server.py::_process_bridge_updates` | `supervisor/worker_chat_lane.py::auto_resume_after_restart` | consumed (unlinked) by the reader; rolled back on write failure | Paired with `panic_stop.flag` for stable-build compatibility. |
| `state/pycache/` | `launcher.py` at import time; `ouroboros/launcher_bootstrap.py::embedded_python_env`; `ouroboros/packaged_cli.py::_set_global_bytecode_suppression` | CPython (`PYTHONPYCACHEPREFIX`) | mkdir on demand; **never pruned** | Keeps `.pyc` out of the signed bundle. It is also the reason a "hermetic" run that only rebinds module globals still writes into a live data root — the prefix is set for every child process. |
| `state/python-userbase/` | `ouroboros/launcher_bootstrap.py::embedded_python_env` | pip / CPython (`PYTHONUSERBASE`) | **never pruned by design**; recovery is a manual removal | Stale dependencies here can shadow an upgrade; the code states this rather than fixing it. |
| `state/.<name>.tmp.<uuid>` | `supervisor/state.py::atomic_write_text`; `ouroboros/utils.py::write_text_atomic` | none | orphans reaped by `ouroboros/utils.py::sweep_stale_temp_files` from `server_maintenance.py::_startup_prune_sweeps` | The only whole-tree GC that touches `state/`. |

---

## 3. Lock plane

Locks are durable files with no content contract. They are listed because they are
part of the tree an operator sees and because two of them do **not** live where the
layout tree implies.

| path | writer(s) | lifecycle | notes |
|---|---|---|---|
| `locks/state.lock` | `supervisor/state.py` (`STATE_LOCK_PATH`), `supervisor/evolution_lifecycle.py`, `ouroboros/agent_startup_checks.py` | created on demand; **never deleted** | Serialises `state.json` *and* `evolution_campaign.json`. `data/locks/` is absent from the ARCHITECTURE tree (§16). |
| `locks/advisory_review.lock` | `ouroboros/review_state.py` | **never deleted** | |
| `locks/git.lock` | `ouroboros/tools/git_plumbing.py` | **never deleted** | |
| `locks/managed_update.lock` | `supervisor/update_merge.py` | **never deleted** | Fail-closed: an unavailable lock refuses the update. |
| `state/usage_attempts.lock`, `state/usage_import.lock` | `ouroboros/usage_ledger.py::_named_lock`, `ouroboros/usage_legacy_import.py` | **never deleted**; stale-broken by timeout | Deliberately separate so a long import cannot block the hot budget path. |
| `state/skill_lifecycle.lock` | `ouroboros/skill_lifecycle_queue.py` | **never deleted** | |
| `state/.payload_delegation_claim.lock` | `ouroboros/tools/delegate_integration.py` | **never deleted** | Fail-**closed**: an unavailable lock refuses the run. |
| `state/<name>.json.lock` sidecars | `ouroboros/utils.py::update_json_locked`; `projects_registry.py::_file_write_lock`; `gateway/ui_preferences.py::_preferences_lock` | auto-created beside each guarded JSON; **never deleted** | Covers `cancel_intents`, `terminal_deliveries`, `evolution_campaign`, `advisory_overrides`, `projects`, `project_task_bindings`, `ui_preferences`. |
| `<dir>/.append_jsonl_<sha12>.lock` sidecars | `ouroboros/utils.py::append_jsonl` (`jsonl_append_lock_path`); also taken by `supervisor/state.py::rotate_jsonl_log_if_needed` and `ouroboros/project_dialogue.py::append_chat_annotation` | created and unlinked per append; stale-broken after ~10 s | Appear in both `state/` and `logs/`. |
| `memory/.consolidation.lock`, `memory/scratchpad_blocks.json.lock`, `memory/dialogue_blocks.json.lock` | `ouroboros/consolidator.py`, `ouroboros/memory.py` | **never deleted** | |
| `settings.json.lock` | `ouroboros/config.py::_acquire_settings_lock` | see §1 | Beside `settings.json`, not under `state/`. |

---

## 4. Skill state plane — `state/skills/<skill>/`

Root constructor `ouroboros/skill_loader.py::_skills_state_root`; per-skill directory
`skill_state_dir`, which sanitises the name through `canonical_skill_name` and mkdirs
on access. **The directory is never removed**: both uninstall paths deliberately keep
durable state behind.

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `enabled.json` | `ouroboros/skill_loader.py::save_enabled` — 5 independent calling modules: `gateway/extensions.py`, `tools/skill_exec.py`, `launcher_bootstrap.py`, `skill_review_runner.py`, `extension_liveness.py` | `skill_loader.py::load_enabled` | **never pruned** | `enabled=true` is not executability; grants decide that. |
| `review.json` | `ouroboros/skill_loader.py::save_review_state` — callers `skill_review.py`, `skill_owner_attestation.py`, and **`launcher_bootstrap.py`** (`_stamp_native_seed_trust` and the legacy re-pin) | `skill_loader.py::load_review_state` — status is computed live from findings | **never pruned** | The launcher, not the reviewer, is the origin of `review_profile="native_seed"` verdicts. This is the one owner-state file the agent may write through `data_write`. |
| `review_job.json` | `ouroboros/skill_review_runner.py` (`mark_stale_review_job_interrupted`, `_patch_review_job`, `_mark_review_job_timeout`, the start/finish callbacks, the heartbeat) | `skill_review_runner.py::_read_review_job` | **never deleted** — `reconcile_stale_review_jobs` only rewrites `running` to `interrupted` | Undocumented in the layout tree (§16). |
| `grants.json` | `ouroboros/skill_loader.py::save_skill_grants` — callers `gateway/extensions.py` (owner grant UI) and `marketplace/install.py` (`requires.config` bootstrap) | `skill_loader.py::load_skill_grants`, `grant_status_for_skill` | **never pruned** | Partial-approval merge only while `content_hash` and the requested key set still match. |
| `owner_attestation.json` | `ouroboros/skill_owner_attestation.py::run_owner_attestation` (single ingress: the owner attest endpoint) | `skill_loader.py::load_review_state` — presence gates `review_profile="owner_attested"` | no code deletes it; removal is an owner filesystem action. `tools/registry_guard_process.py::_restore_owner_files` unlinks agent-forged copies | Owner state: the agent can never forge it, and a content edit stales it through `content_hash`. |
| `review_history.jsonl` | `ouroboros/skill_review_history.py::append_history` / `append_history_once` — callers `skill_review.py` and `skill_review_runner.py::_append_terminal_history` | `skill_review_history.py::load_history`, `normalize_history` | append-only; **never pruned or rotated** | The gateway serves normalised owner-visible detail without exposing raw reviewer text to chat. |
| `accepted_rebuttals.json` | `ouroboros/skill_review_rebuttals.py::_record_accepted_rebuttal` | `_load_accepted_rebuttals` → injected into later review prompts | **never pruned** | |
| `deps.json` | **two independent writers:** `ouroboros/marketplace/isolated_deps.py::install_isolated_dependencies` and `ouroboros/marketplace/install.py::_write_deps_state` / `restore_payload_state` | `isolated_deps.py::read_deps_state` | three independent deleters: `install.py::uninstall_skill`, `marketplace/ouroboroshub.py::uninstall`, `install.py::restore_payload_state` | A payload-resident mirror lives at `<skill_dir>/.ouroboros_env/`. Only structured `install_specs.auto` is installed — README prose is not. |
| `auto_repair.json` | `ouroboros/gateway/marketplace.py::_maybe_enqueue_marketplace_auto_repair` | the same function (attempted-hash dedup) | **never pruned** | Not in the owner-state filename set, so the agent may write it. |
| `health.json` | `ouroboros/extension_health.py::record_extension_health` (from `extension_loader.py::reload_all`) | `extension_health.py::read_extension_health`, `regressed_extensions` | **never pruned** | Durable `last_known_good` versus `last_observed` regression memory. |
| `auth_token.json` | `ouroboros/extension_plugin_api.py::mint_skill_token` | `ouroboros/gateway/host_service.py::authenticate_token_payload` | re-minted on content-hash change; **never pruned** | |
| `clawhub.json` | `ouroboros/marketplace/provenance.py::write_provenance` | `provenance.py::read_provenance` | deleted by `provenance.py::delete_provenance` (from `uninstall_skill` / `restore_payload_state`) | Undocumented in the layout tree (§16). |
| `self_authored.json` | `ouroboros/tools/core.py::_data_write` (writes both the payload marker and this state mirror) | `skill_loader.py::is_self_authored_skill_dir` | **never pruned**; agent-forged copies unlinked by `registry_guard_process::_restore_owner_files` | Undocumented (§16). |
| `repair_admission.json` | `ouroboros/skill_repair_admission.py::record_repair_admission`, `advance_repair_expected_hash` | `skill_repair_admission.py::load_repair_admission`, CAS-checked in `tools/core.py::_data_write` | **never pruned** | Undocumented (§16). |
| `chat_id_counter.json` | `ouroboros/gateway/host_service.py::HostServiceState.allocate_internal_chat_id` | the same method | **never pruned** | Undocumented (§16). |
| `extension_calls/<uuid>.json`, `.result.json`, `<uuid>.imports/` | `ouroboros/extension_process_runner.py::_run_child` (private dir, mode 0700); the child writes the result | `_run_child` / the child process | removed per dispatch in `_run_child`'s `finally`; a crash leaks them — the names do not match the temp-sweep glob | The layout tree documents the JSON files but not the `.imports/` sibling directories. |
| `__extension_imports/<pid>-<uuid>/skill/` | `ouroboros/extension_import_staging.py::_stage_extension_import_tree` | the Python import machinery | swept by `_sweep_stale_extension_imports` (from `extension_loader.py` load and `reload_all`) and removed on unload | Reaping requires the owner PID to be dead *and* the tree to be past the sweep grace, so a peer worker's live tree is never taken. |

### Skill payloads — `data/skills/<bucket>/<skill>/`

`config.SKILL_SOURCE_SUBDIRS` is exactly `(native, clawhub, external, ouroboroshub)`.
`self_authored` and `user_repo` are *classifications* produced by
`skill_loader.py::_classify_skill_source` / `_classify_skill_location`, not directories:
a self-authored skill is an `external` payload carrying `.self_authored.json`, and a
`user_repo` skill lives outside `data/` under `OUROBOROS_SKILLS_REPO_PATH`.

| path | writer(s) | lifecycle | notes |
|---|---|---|---|
| `data/skills/` and its four buckets | `ouroboros/config.py::ensure_data_skills_dir` (callers `marketplace/install.py`, `launcher_bootstrap.py`) | created on demand; never removed | The whole payload plane is absent from the ARCHITECTURE tree (§16). |
| `native/<skill>/` (+ `.seed-origin`) | `ouroboros/launcher_bootstrap.py::_seed_skills_into`, `_reseed_native_skill_in_place` | seeded once behind a completion marker; an intentionally deleted seed is never resurrected; no code deletes native payloads | `_stamp_native_seed_trust` writes the accompanying `review.json`. |
| `clawhub/<skill>/` (+ `.clawhub.json`) | `ouroboros/marketplace/install.py::install_skill` → `_land_staged_into_data_plane` → `ouroboros/marketplace/fetcher.py::land_staged_tree` | removed by `uninstall_skill`, gated on the provenance sidecar and root containment; rollback through `snapshot_payload_state` / `restore_payload_state` | |
| `ouroboroshub/<skill>/` (+ `.ouroboroshub.json`) | `ouroboros/marketplace/ouroboroshub.py::install` → `land_staged_tree` | removed by `ouroboroshub.py::uninstall`, gated on the marker | |
| `external/<skill>/` (+ `.self_authored.json`) | **agent lane:** `ouroboros/tools/core.py::_data_write`; **owner lane:** `ouroboros/gateway/files.py` (`api_files_write`, `api_files_mkdir`, `api_files_upload`) | deleted only through the owner file browser (`api_files_delete` / `api_files_transfer`); no marketplace uninstall path | Five independent installer/mutator modules touch the payload plane in total. |

---

## 5. Claudexor home, managed runtime, and the update transaction

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `claudexor/` (`CLAUDEXOR_CONFIG_DIR`) | `ouroboros/claudexor_daemon.py::OwnedClaudexorDaemon.ensure_running` (mkdir) | — | never removed | Ouroboros-owned home, never the operator's `~/.claudexor`. |
| `claudexor/ouroboros-owned.json` | `ouroboros/claudexor_daemon.py::_write_ownership_marker` | `claudexor_daemon.py::read_ownership_marker` → `verify_owned_home` | **never deleted** | A marker naming a different data dir marks a foreign home: disclosed, never adopted, never killed. |
| `claudexor/daemon/control-api.json`, credential profiles, runs | **the external daemon process** — Ouroboros never writes them | `ouroboros/gateways/claudexor.py::discover_daemon_at` | engine-owned | Ouroboros mutates profiles and runs only over loopback HTTP; it never deletes vendor credential material itself. |
| `claudexor/daemon.log` | `ouroboros/claudexor_daemon.py::OwnedClaudexorDaemon.ensure_running` (child stdout/stderr sink) | owner only | append-only; **never rotated or pruned** | Unbounded growth. |
| `state/cx/<version>-<sha12>/` | `ouroboros/claudexor_runtime.py::ClaudexorRuntimeManager._promote_archive` | `_managed_command` / `_managed_metadata` | the displaced tree is removed after promotion; staging is cleaned; rollback restores on failure | |
| `state/cx/<install>/managed-runtime.json` | `_promote_archive` | `_read_metadata`, `_other_installed_metadata` | dies with its tree | |
| `state/cx/node/<ver>-<platform>/` + `managed-node.json` | `claudexor_runtime.py::_promote_node` | `_resolve_node` / `_ensure_node` | displaced trees removed; **superseded node pins are not** | `managed-node.json` lives here, not under the app root. |
| `state/cx/cache/<archive>` | `_obtain_archive`, `_fetch_exact_file` | `verify_runtime_archive` / `verify_node_archive` | **never pruned** — verified archives accumulate | |
| `state/cx/install.lock` | `ClaudexorRuntimeManager._install` | `_install_in_progress` (mtime staleness) | released in `finally` | |
| `repo/.git/ouroboros-update-tx.json` | `supervisor/update_merge.py::write_update_tx` — four independent calling modules: `update_merge.py`, `ouroboros/gateway/control.py`, `ouroboros/tools/git.py`, and the clear path in `supervisor/git_ops_reset.py` | `update_merge.py::read_update_tx` / `read_update_tx_strict`; also `ouroboros/server_restart.py`, `supervisor/git_ops_reset.py` | deleted by `clear_update_tx` | **Lives in the repo's git dir, not under `data/`.** The resolver's privilege is bound by an authority fingerprint over immutable fields. |
| `repo/.git/ouroboros-update-intent.json` | `supervisor/git_ops.py::_write_update_intent` (callers `git_ops_updates.py`, `gateway/control.py`) | `git_ops.py::_read_update_intent`, consumed by `git_ops_reset.py` | cleared by `_clear_update_intent` from four independent modules | The reader fails **closed** on a parse error. |
| `repo/.git/ouroboros-managed.json` | `ouroboros/launcher_bootstrap.py::_write_repo_manifest` (sole writer) | `launcher_bootstrap.py::load_repo_manifest`; `supervisor/git_ops.py::_read_managed_repo_meta`, `managed_branch_defaults`, `_is_launcher_managed_repo` | overwritten on re-bootstrap; never deleted | The constant name is declared twice, in `launcher_bootstrap.py` and `supervisor/git_ops.py`. |
| `repo/.git/ouroboros-bootstrap-pending` | `launcher_bootstrap.py::_mark_bootstrap_pin_pending` | `supervisor/git_ops.py::_pin_to_bundle_sha_on_bootstrap` | cleared by `_clear_bootstrap_pin_marker` | |
| `<bundle_dir>/repo_bundle_manifest.json` | **build time only** — `scripts/build_repo_bundle.py` | `launcher_bootstrap.py::_normalize_bundle_manifest`, `_assert_bundle_integrity` | ships inside the app bundle | Neither under the app root nor under `data/`; read-only at runtime. |

---

## 6. Log plane — `logs/`

Two of these rotate. The rest grow without bound; the code says so, and this table
says so rather than leaving the asymmetry to be discovered.

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `logs/chat.jsonl` | `supervisor/message_bus.py::log_chat` (the canonical row writer; `send_with_budget`, the server WS send path and `gateway/control.py` all funnel through it); `ouroboros/post_task_synthesis.py::_run_task_summary`; `ouroboros/skill_review_runner.py::_append_review_chat_summary` | `ouroboros/memory.py` (`read_jsonl_tail`, `recent_dialogue`) → `ouroboros/context.py`; `ouroboros/consolidator.py::consolidate`; `ouroboros/project_dialogue.py` | **rotated** by `supervisor/state.py::rotate_jsonl_log_if_needed` from the supervisor tick, at ~800 KB, into `archive/chat_<ts>.jsonl`; never age-pruned | Rotation is `os.replace` under the same append-lock sidecar as the appends. Suppressed in isolated benchmark roots. |
| `logs/chat_annotations.jsonl` | `ouroboros/project_dialogue.py::append_chat_annotation` (its own `O_APPEND` + fsync, not `append_jsonl`) — callers `supervisor/events_project_routing.py`, `ouroboros/server_owner_routing.py` | owner/UI only — `latest_chat_annotations` | **compacted in place** by `_compact_annotations_locked` at ~800 KB, keeping the latest row per message id that still exists in live chat or the three newest archives; never rotated | Presentation-only by contract; owns no routing state, and a torn final row is ignored. |
| `logs/progress.jsonl` | `supervisor/message_bus.py::send_with_budget`; `ouroboros/consciousness.py::BackgroundConsciousness._emit_progress`; `ouroboros/skill_review_runner.py::_append_interrupted_review_progress` | `ouroboros/memory.py::summarize_progress` → `ouroboros/context.py` | **rotated** at ~800 KB into `archive/progress_<ts>.jsonl` | `agent_startup_checks.py` treats oversize here as broken rotation, unlike events/tools. |
| `logs/events.jsonl` | **≈45 independent writer modules** across the worker, the tool layer, the gateway and the supervisor (see §14) | behavioural readers: `ouroboros/delegate_custody.py::replay` / `open_containment_faults`, `ouroboros/task_status.py` orphan check, `ouroboros/context_health.py`, `ouroboros/task_finalization.py`, `ouroboros/task_pacing.py`, `supervisor/worker_pool_lifecycle.py::_first_worker_event_since` | **never rotated, never pruned**; removed only by `api_reset` | `supervisor/events.py` is a dispatcher facade — it writes only the taxonomy-passthrough row; every real event row is written by an `events_*` leaf. A size warning exists; rotation does not. |
| `logs/tools.jsonl` | `ouroboros/loop_tool_execution.py` (`_append_tool_log`, `_make_timeout_result`); `ouroboros/consciousness.py::_execute_tool` | `ouroboros/memory.py::summarize_tools` → `ouroboros/context.py` | **never rotated, never pruned** | `_append_tool_log` refuses a duplicate write when a task-local root resolves to the same canonical file. |
| `logs/supervisor.jsonl` | **≈25 independent writer modules** (see §14) | `ouroboros/memory.py::summarize_supervisor` → `ouroboros/context.py`; `ouroboros/context_health.py` | **never rotated, never pruned**, and it has no size-warning threshold at all | Explicitly a forensic trail that is never read back for state. |
| `logs/task_reflections.jsonl` | `ouroboros/reflection.py::append_reflection` (full entry) and `append_reflection_routed` (bounded pointer row for project roots) | `ouroboros/memory.py::read_jsonl_tail` → `ouroboros/context.py` | **never rotated, never pruned** | The pointer row carries `write_failed` if the project-side append failed. |
| `logs/containment_faults.jsonl` | `ouroboros/delegate_custody.py::record_containment_fault` / `resolve_containment_fault` | `delegate_custody.py::open_containment_faults` — read whole — feeding a CRITICAL health invariant | **never pruned by design**: an open incident can never age out | The compact projection is written first and the same fact is mirrored into `events.jsonl`, so either landing alone keeps the incident visible. |
| `logs/tasks/task_<task_id>.txt` | `ouroboros/utils.py::sanitize_task_for_event` | **none** — only the pointer string is stamped into the event row | **never pruned** | Full untruncated task text on disk with no consumer. Undocumented (§16). |
| `logs/agent_stdout.log` | `launcher.py::start_agent._stream_output` | owner only | plain append; **never rotated or pruned** | |
| `logs/server.log` (+ `.1`–`.3`) | `server.py` module-scope `RotatingFileHandler` — stdlib logging from every module in the server process | owner/UI only | rotated at 2 MB, three backups | A secret-redacting filter is attached to every root handler. Skipped entirely under pytest against the real default data dir. |
| `logs/launcher.log` (+ `.1`, `.2`) | `launcher.py` module-scope `RotatingFileHandler` | owner only | rotated at 2 MB, two backups | Undocumented (§16). |

---

## 7. Observability and services

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `observability/blobs/<sha256>.<kind>.gz` | `ouroboros/observability.py::write_blob` (`kind="json"`, from `persist_call`) **and** `ouroboros/tools/services.py` (`kind="txt"`, service-log blobs) | **behavioural**: `observability.py::latest_llm_response_text` dereferences a call manifest's `full_payload_ref` into the blob — the salvage chain used by `supervisor/terminal_delivery.py`, `supervisor/task_reaper.py` and `ouroboros/loop_round_limits.py`; `read_blob_ref` additionally serves `scripts/contributor_review_evidence.py` | `prune_observability_blobs` **deletes nothing** — it counts and reports `preserved_indefinitely` | Content-addressed, `0600` under private directories. The retention environment variable is parsed and clamped but has no deleting effect. |
| `observability/calls/<task_id>/<call_id>.json` | `ouroboros/observability.py::write_call_manifest` via `persist_call` — upstream sites in the loop, review substrate, triad review, compaction, vision routing and the physical-attempt capture | **behavioural**: `observability.py::latest_llm_response_text` is the salvage source used by `supervisor/terminal_delivery.py`, `supervisor/task_reaper.py`, `ouroboros/loop_round_limits.py` | counted, never deleted | Ids are regex-sanitised before the join. |
| `observability/salvaged/<task_id>.txt` | `ouroboros/observability.py::preserve_salvaged_output` (from `supervisor/terminal_delivery.py`) | `observability.py::preserved_salvage_path` | **never pruned** | Written on the canonical drive so it outlives the child drive. Undocumented (§16). |
| `services/<task_id>/<service>.log` | `ouroboros/tools/services.py::_start_service` | tool surface only — `_service_logs` returns a redacted bounded tail plus a blob ref | age-based GC (one of the six age-pruned planes — see the GC bullet in §15): `tools/services.py::prune_service_logs` at startup, cutoff `ouroboros/retention.py::age_cutoff(get_gc_retention_days())`; terminal-task sweep through `archive_task_service_logs` | A log larger than the blob cap is neither archived nor deleted — it is retained live with a disclosed path, so an oversize service log is never pruned. |
| `services/<task_id>/<name>.executor.log` | `ouroboros/workspace_executor.py::start_service` (local backend) | the executor record | same prune path | The docker backend writes to a host temp path *outside* the data root, which this runtime never prunes. Undocumented name variant (§16). |

---

## 8. Archive and uploads

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `archive/chat_<ts>.jsonl`, `archive/progress_<ts>.jsonl` | `supervisor/state.py::rotate_jsonl_log_if_needed` | `ouroboros/gateway/_helpers.py::read_rotated_jsonl_entries` (three newest segments), `ouroboros/project_dialogue.py`, `ouroboros/consolidator.py`, `ouroboros/projects_registry.py` | **exempt from every GC by explicit contract** in `supervisor/state.py`: archives are durable history, readers backfill from them, and no retention sweep may be added | A collision suffix keeps name ordering chronological. |
| `archive/rescue/<ts>_<uuid8>/` — `rescue_meta.json`, `changes.diff`, `status.porcelain.txt`, `unmerged.txt`, `merge_msg.txt`, `unpushed_commits.txt`, `untracked/**` | `supervisor/git_ops_rescue.py::_create_rescue_snapshot` — **five independent calling modules**: `git_ops_reset.py`, `git_ops_updates.py`, `update_recovery.py`, `gateway/control.py`, and `rescue_before_destructive_rollback` | `ouroboros/context_health.py` surfaces recent snapshots and tells the agent to read `rescue_meta.json` | **never pruned** | `changes.diff` is written as raw bytes (`_atomic_write_bytes`) because an unmerged-index diff must not round-trip through `str`. A git ref `refs/rescue/<dirname>` is created alongside and never deleted. A disclosure row lands in `logs/supervisor.jsonl` *before* the destructive command. |
| `archive/managed_repo/<epoch>-<uuid8>-<reason>/` | `ouroboros/launcher_bootstrap.py::_archive_existing_repo` | owner only | **never pruned** | Can hold full repository copies. Undocumented (§16). |
| `archive/usage_import/<sha16>/` | `ouroboros/usage_legacy_import.py::_ensure_legacy_imported_locked` | referenced by `usage_import_watermark.json`; re-runs verify byte equality and raise on mismatch | **never pruned** | Immutability-checked, never rewritten. |
| `uploads/<uuid>_<basename>` | `ouroboros/gateway/files.py::api_chat_upload` | `ouroboros/gateway/ws.py` attachment resolution; `ouroboros/tools/vision.py` allow-root | **never pruned**; deleted only by owner action or `api_reset` | |
| `uploads/screenshots/<ts>.png` | `ouroboros/tools/browser.py` native-screenshot injector | attached to the live conversation only | **never pruned** | Undocumented (§16): the agent, not only the owner, writes under `uploads/`. |
| `uploads/views/<ts>_<stem>.<ext>` | `ouroboros/tools/vision.py::view_image` | live conversation only | **never pruned** | Undocumented (§16). |
| `uploads/routed-<uuid><ext>` | `ouroboros/server_owner_routing.py` inline-image staging | `stage_task_attachments` | deleted in the staging `finally` | Transient by construction; a crash between the write and the `finally` leaks it permanently. |

---

## 9. Cognitive memory plane — `memory/`

The agent's own plane. Operators treat it as read-only; the runtime writers below are
the agent's tools, the consolidator, the reflection pass and the child-drive seeder.

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `memory/identity.md` | `ouroboros/tools/control_runtime.py::_update_identity` (agent tool); `ouroboros/memory.py::load_identity` / `ensure_files` (default seed); `ouroboros/headless.py::_copy_stable_memory` (child-drive seed) | `ouroboros/context.py::build_memory_sections` (stable partition) | **never pruned**; wiped only by `gateway/control.py::api_reset` | Blocked from `write_file(root='runtime_data')` by `ouroboros/tool_access.py`. |
| `memory/identity_journal.jsonl` | `ouroboros/tools/control_runtime.py::_update_identity`; `memory.py::ensure_files` creates it empty | `ouroboros/utils.py` memory-growth chart | **never pruned or rotated** | Rollback-grade: stores full old and new content per identity write. |
| `memory/scratchpad.md` | `ouroboros/memory.py::regenerate_scratchpad_md`, `ensure_files` | `ouroboros/context.py::build_memory_sections` (volatile partition) | derived — regenerated on every block change | A legacy flat file with no blocks raises rather than being silently migrated. |
| `memory/scratchpad_blocks.json` | `ouroboros/memory.py::append_scratchpad_block`; `ouroboros/consolidator.py::_consolidate_scratchpad_blocks` — reached from the agent tool (`control_runtime::_update_scratchpad`), the reflection pass (`reflection.py`) and the consolidator | `memory.py::load_scratchpad_blocks` | FIFO-evicted at 10 blocks (evicted blocks journalled first); LLM-consolidated above a size threshold | Guarded by a sidecar lock. |
| `memory/scratchpad_journal.jsonl` | `ouroboros/memory.py::append_scratchpad_block` (append, eviction, failure and legacy-upgrade rows) | `ouroboros/utils.py` memory-growth chart | **never pruned** | The only durable home of FIFO-evicted blocks. |
| `memory/dialogue_blocks.json` | `ouroboros/consolidator.py::_run_block_consolidation`, `_append_gap_block` | `ouroboros/memory.py::load_dialogue_blocks` → `ouroboros/context.py` | blocks compressed into eras; never file-level pruned | Written only from the root post-task worker. |
| `memory/dialogue_meta.json` | `ouroboros/consolidator.py::_run_block_consolidation` | `consolidator.py::should_consolidate`; `memory.py::load_dialogue_meta` | **never pruned** | A cursor over `logs/chat.jsonl`; gap markers are written when the source rotated under the cursor. |
| `memory/dialogue_summary.md` | **no writer** | `ouroboros/context.py::build_memory_sections` (read-only legacy fallback) | never created by this release | Retired flat format; the only memory-plane file with a reader and no writer. |
| `memory/WORLD.md` | `ouroboros/world_profiler.py::generate_world_profile` (from `memory.py::ensure_files` and a first-run `launcher_bootstrap` subprocess); `headless.py::_copy_stable_memory` | `ouroboros/context.py`; `memory.py::load_world_profile` | created once; **never regenerated** unless deleted by hand | Not covered by the cognitive-tool write guard. |
| `memory/registry.md` | `ouroboros/tools/memory_tools.py::_memory_update_registry` (agent tool); `headless.py::_copy_stable_memory` | `ouroboros/context.py`; `_build_registry_digest` | **never pruned** | Also writable through `write_file(root='runtime_data')` — unlike `identity.md` and `scratchpad.md`, it is not in the cognitive-tool guard set. |
| `memory/deep_review.md` | `ouroboros/agent.py` (deep-self-review task path) | `ouroboros/context.py` (truncated for injection) | **overwritten each run, no history, never pruned** | `deep_self_review.py` builds the pack and returns text; the file write is in `agent.py`. |
| `memory/knowledge/<topic>.md` | `ouroboros/tools/knowledge.py::_knowledge_write` (agent tool); `ouroboros/consolidator.py::_write_knowledge_entries`; `headless.py::_copy_stable_memory` | `knowledge.py::_knowledge_read`; the index is injected by `ouroboros/context.py` | **never pruned — no delete tool exists** | Topics validated by `_sanitize_topic`; the consolidator uses the same validator. |
| `memory/knowledge/index-full.md` | `ouroboros/tools/knowledge.py::_rebuild_index` / `_update_index_entry`; `ouroboros/consolidator.py::_rebuild_knowledge_index`; `ouroboros/improvement_backlog.py::_rebuild_index` | `ouroboros/context.py::build_knowledge_sections`; `knowledge.py::_knowledge_list` | fully derived; rebuilt on each write | Three independent rebuilders with slightly different summary extraction. |
| `memory/knowledge/patterns.md` | `ouroboros/reflection.py::_update_patterns` (LLM full-table rewrite) | `ouroboros/context.py::build_knowledge_sections` | bounded to ~20 rows by prompt contract only; **never file-pruned** | Always written to the canonical drive even for project-scoped roots — deliberately cross-project cognition. |
| `memory/knowledge/patterns_history.jsonl` | `ouroboros/reflection.py::_update_patterns` | none at runtime — provenance and recovery | **never pruned** | The only rollback source for the full-replace `patterns.md` write. |
| `memory/knowledge/improvement-backlog.md` | `ouroboros/improvement_backlog.py` (`ensure_backlog_file`, `append_backlog_items`, `merge_backlog_text`, `close_backlog_items`, `groom_backlog`) — reached from the agent tool, `ouroboros/post_task_synthesis.py::_update_improvement_backlog` and `ouroboros/agent_startup_checks.py` | `improvement_backlog.py::format_backlog_digest` → `context.py`, `consciousness.py`, `supervisor/evolution_lifecycle.py`, `post_task_evolution.py` | LLM-groomed and capped by `groom_backlog`; items are closed, never deleted | One **global** store: the write is force-rooted away from project scope and forked drives, and an unparseable write fails closed. Only fingerprinted items are groomable; hand-added items pass through untouched. |
| `memory/knowledge_history.jsonl` | `ouroboros/tools/knowledge.py::_knowledge_write`, `_record_backlog_history` | none at runtime — rollback and audit | **never pruned** | Resolves under `data/projects/<pid>/` for project-scoped tasks; the backlog history is forced to the global copy. |
| `memory/knowledge_journal.jsonl` | `ouroboros/tools/knowledge.py::_knowledge_write` | none at runtime | **never pruned** | Same project scoping. |
| `memory/owner_mailbox/<task_id>.jsonl` | **five independent modules**: `supervisor/steering.py`, `ouroboros/server_owner_routing.py`, `ouroboros/gateway/task_hurry.py`, `supervisor/task_reaper.py`, `ouroboros/tools/core.py::_forward_to_worker` — all through `ouroboros/owner_mailbox.py::write_owner_message` | `owner_mailbox.py::drain_owner_entries` → `ouroboros/loop_round_limits.py`, `ouroboros/review_evidence_sections.py` | **deleted** by `owner_mailbox.py::cleanup_task_mailbox` at task teardown and on same-id requeue | A typed control rail (`owner_text`, `finalize_now`, `hurry`, `control_revoked`), not only user text; the supervisor and the agent write here too. Un-sending is a `control_revoked` row resolved by the reader, never a delete. The only memory-plane file the runtime deletes. |
| `memory/` (whole tree) | — | — | `ouroboros/gateway/control.py::api_reset` | The only whole-plane destructor. |

---

## 10. Project facts plane — `projects/<project_id>/`

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `projects/<pid>/knowledge/<topic>.md`, `index-full.md` | `ouroboros/tools/knowledge.py` (path from `ouroboros/project_facts.py::project_knowledge_dir`, resolved against `config.DATA_DIR`, not the task drive) | `knowledge.py::_knowledge_read`; `context.py::build_knowledge_sections` (project branch) | **never pruned** | Isolated from `memory/knowledge` and from forked child drives by construction. |
| `projects/<pid>/knowledge_history.jsonl`, `knowledge_journal.jsonl` | `ouroboros/tools/knowledge.py::_knowledge_write` | none at runtime | **never pruned** | Fall out of the knowledge directory's parent — implicit, not an explicit constructor. |
| `projects/<pid>/logs/task_reflections.jsonl` | `ouroboros/reflection.py::append_reflection_routed` (FULL text) | `ouroboros/context.py` (bounded tail) | **never pruned** | The canonical log keeps only a bounded pointer row. |
| `projects/<pid>/journal.jsonl` | `ouroboros/tools/project_journal.py::_journal_write` (agent tool) and `append_journal_milestone` | `project_journal.py::_journal_read`, `journal_tail_digest` → `context.py` | **never pruned** | The durable project journal, distinct from the ephemeral task-tree blackboard. Over-limit rows are rejected by the tool; automatic milestones bound themselves with a visible pointer. |
| `projects/<pid>/workpad.md` | `ouroboros/tools/project_journal.py::_workpad_write` (agent tool) | `_workpad_read`; injected into context in full | **never pruned** | Not under `memory/`. Undocumented in the layout tree (§16). |
| `projects/<pid>/` (directory) | created by the writers above | `ouroboros/projects_registry.py::reconcile_projects` registers orphan directories | **never deleted** — `complete_project_deletion` only tombstones the registry row; `api_reset` does not include `projects` | Project deletion is registry-only; the facts store is immortal. |

---

## 11. Task result and artifact plane — `task_results/`

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `task_results/<task_id>.json` | `ouroboros/task_results.py::write_task_result` — one write seam, **20+ independent calling modules** across three process classes (supervisor: `worker_assignment`, `events_task_done`, `cancel_custody`, `queue_snapshot`, …; worker: `agent`, `agent_task_pipeline`, `task_status`, `headless`; gateway: `gateway/tasks`) | `task_results.py::load_task_result`; `ouroboros/task_status.py::load_effective_task_result` | **never pruned** — the only deletion is a scheduling rollback in `tools/control_scheduling.py`, plus `api_reset` | Terminal statuses are sticky and regressions are dropped by a monotonic reducer. This file gates every pruner in §12: a missing result means the drive stays forever. Absent from the layout tree (§16). |
| `task_results/artifacts/<task_id>/` | `ouroboros/headless.py::task_artifacts_dir`; lazily materialised for `root='artifact_store'` by `ouroboros/tool_access.py` | `ouroboros/artifacts.py::collect_task_artifact_records` | **never age-pruned**; removed only on admission rejection | The widest multi-writer surface in the tree — ten-plus independent writer modules (§14). |
| `…/<task_id>/.artifact_manifest.json` | `ouroboros/artifacts.py::copy_file_to_task_artifacts`, `copy_directory_to_task_artifacts` | `artifacts.py::collect_task_artifact_records` | never pruned | Excluded from the artifact records it describes. |
| `…/<task_id>/.scratch_manifest.json` | `ouroboros/artifacts.py::record_task_scratch` (from `ouroboros/tools/shell_effects.py`) | `artifacts.py::read_task_scratch_fingerprints`; `ouroboros/workspace_patch_capture.py` | additive union, capped; never pruned | Written to both the budget drive root and the task drive root. Advisory only — never load-bearing. |
| `…/<task_id>/<artifact files>` | `ouroboros/artifacts.py`, `ouroboros/headless.py` (`memory_export.json`, `deliverable_manifest.json`), `ouroboros/workspace_patch_capture.py` (`workspace.patch`, `workspace_patch.json`), `ouroboros/outcomes.py` (verification artifacts), `ouroboros/task_status.py` (child→parent rebase), `ouroboros/tools/core.py`, `core_artifacts.py`, `shell_outputs.py`, `media.py`, `delegate_integration.py` | `artifacts.py::collect_task_artifact_records`; `gateway/tasks.py` | never pruned; staged attachments removed by `artifacts.py::remove_staged_attachments` | |
| `task_results/artifact_versions/<task_id>/<name>/<stamp>.<sha12>.<name>` | `ouroboros/artifacts.py::_archive_previous_artifact_version` | none at runtime — manual recovery | **rotated to the last 5 versions per artifact name**; the `<task_id>/` directory itself is never removed | Only triggered when a user-file artifact overwrites a differing existing one. Anchored on the drive root, so a child drive builds its own copy. |

---

## 12. Task scratch and swarm plane

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `task_drives/<task_id>/` | the agent tool layer (`ouroboros/tools/core.py::_write_file`, the shell tools) and `ouroboros/delegate_output.py` (`delegated_runs/`); the directory is materialised by `ouroboros/tool_access.py` | root resolution in `ouroboros/tool_access_roots.py` / `tools/tool_context.py` | pruned by `ouroboros/headless.py::prune_task_drives` (root task terminal **and** past `retention.get_gc_retention_days()`) from `server_maintenance.py::_startup_prune_sweeps`; immediate removal on cancel through `remove_subagent_task_drive` | The prune is gated on `task_results/<id>.json`: a missing result file means the drive is never reclaimed. |
| `task_trees/<root_task_id>/blackboard.jsonl` | `ouroboros/task_tree_ledger.py::tree_ledger_append` — callers `ouroboros/tools/task_tree.py::tree_note`, `ouroboros/tools/join_ledger.py`, and `ouroboros/agent_dispatch.py::record_subscription_window_exhausted` | `task_tree_ledger.py::tree_ledger_rows`, `tree_ledger_tail_digest` → `context.py`; `ouroboros/loop_forced_finalization.py`; `ouroboros/task_status.py` | pruned by `headless.py::prune_task_trees` (root terminal or absent, past GC retention) | Anchored on `config.DATA_DIR`, not the task drive, so it survives child-drive forks. Bounded per row and in total. Ephemeral coordination — distinct from the durable project journal in §10. |

---

## 13. Outside the data root

| path | writer(s) | authoritative reader | lifecycle | notes |
|---|---|---|---|---|
| `Deliverables/` (`config.get_deliverables_root`) | the agent tool layer only — `ouroboros/tools/core.py::_write_file(root='user_files')` routed by `ouroboros/tool_access_user_files.py` | `ouroboros/tool_access_roots.py`; readable as the `deliverables` root | **never pruned** — nothing targets this root | A bare filename lands here instead of the home root; an explicit path bypasses the container entirely. |
| `subagent_worktrees/<safe_task_id>/` (`config.get_subagent_worktree_root`) | `ouroboros/subagent_worktrees.py::provision_worktree`; contents written by the child agent | the child agent; patch capture reads it | pruned by `subagent_worktrees.py::prune_orphans` from `server_maintenance.py::_startup_worktree_prune`; explicit `remove_worktree` | `_assert_root_isolated` refuses a root overlapping `repo/` or `data/`, and teardown refuses any path not strictly inside the root. Results live in the task drive, so removal never loses output. |
| `projects/<name>/` (`config.get_subagent_projects_root`) | `ouroboros/subagent_worktrees.py::provision_genesis_project` | `ouroboros/tool_access_roots.py` (read-only root); `ouroboros/project_sources.py`; `ouroboros/coop_checkpoint.py` | **deliberately not registered and never GC-pruned**; removed only by `remove_genesis_project` for a provisioned-but-never-scheduled project | The deliverable itself. Distinct from `data/projects/<project_id>/` in §10 despite the similar name. |
| `ouroboros.pid` (app root) | the launcher's PID lock | the launcher | released on exit; auto-released on crash | Source-mode runs have no PID file; the port lives in `state/server_port`. |

---

## 14. Multi-writer index

Files with more than one independent writer module, ordered by how much the split
matters. A single write seam with many callers is listed separately, because the
failure modes differ: several writers can disagree about format, while many callers of
one seam can only disagree about timing.

**Several independent writer modules**

| path | writer modules |
|---|---|
| `logs/events.jsonl` | ≈45 — the loop family, `agent*`, `consciousness`, `safety`, `delegate_custody`, `task_pacing`, `review_cycles`, `usage_ledger`, `usage_legacy_import`, `server_maintenance`, `subagent_worktrees`, `projects_registry`, `extension_loader`, `python_interpreter`, `owner_hurry`, `skill_review*`, `triad_review`, twelve `tools/*` modules, five `gateway/*` modules, sixteen `supervisor/*` modules, and `server.py` |
| `logs/supervisor.jsonl` | ≈25 — `supervisor/events*`, `workers`, `worker_*`, `queue_snapshot`, `task_lifecycle`, `task_reaper`, `terminal_delivery`, `owner_stop`, `cancel_publication`, `git_ops_*`, `update_merge`, `update_recovery`, plus `ouroboros/cancel_intents`, `process_custody`, `server_liveness`, `tools/control_events`, `tools/control_routing`, `gateway/tasks`, `server.py` |
| `settings.json` | 5 — `config`, `gateway/owner_settings`, `packaged_cli`, `context_mode_compat`, `tools/registry_guard_process` (plus `colab_bootstrap` on a foreign root) |
| `task_results/artifacts/<task_id>/` | 10+ — `artifacts`, `headless`, `workspace_patch_capture`, `outcomes`, `task_status`, `tools/core`, `tools/core_artifacts`, `tools/shell_outputs`, `tools/media`, `tools/delegate_integration` |
| `data/skills/<bucket>/<skill>/` | 5 — `marketplace/install`, `marketplace/ouroboroshub`, `launcher_bootstrap`, `tools/core::_data_write`, `gateway/files` |
| `memory/owner_mailbox/<task_id>.jsonl` | 5 writers, 2 independent deleters (`loop_budget`, `owner_hurry`) |
| `uploads/` | 4 — `gateway/files`, `tools/browser`, `tools/vision`, `server_owner_routing` |
| `archive/` | 4 — `supervisor/state` (rotation), `git_ops_rescue` (`rescue/`), `launcher_bootstrap` (`managed_repo/`), `usage_legacy_import` (`usage_import/`) |
| `logs/chat.jsonl` | 3 — `message_bus::log_chat`, `post_task_synthesis`, `skill_review_runner` |
| `logs/progress.jsonl` | 3 — `message_bus`, `consciousness`, `skill_review_runner` |
| `state/evolution_campaign.json` | 2 — `supervisor/evolution_lifecycle`, `agent_startup_checks::verify_restart` |
| `state/pending_restart_verify.json` | 2 writer processes — `tools/control_runtime` (agent), `supervisor/evolution_lifecycle` (supervisor); one arbitrating consumer |
| `state/advisory_overrides.json` | 2 — `tools/review::_record_advisory_override`, `tools/claude_advisory_review::_record_bypass` |
| `state/panic_stop.flag` | 2 — `server_control::execute_panic_stop`, `server.py::_process_bridge_updates`; consumed by a third module |
| `state/skills/<skill>/deps.json` | 2 writers, 3 deleters |
| `observability/blobs/` | 2 — `observability::persist_call` (`json`), `tools/services` (`txt`) |
| `services/<task_id>/*.log` | 2 — `tools/services`, `workspace_executor` |
| `logs/tools.jsonl` | 2 — `loop_tool_execution`, `consciousness` |

**Split write/delete ownership across processes**

- `state/server_port` — written by the server, deleted by the launcher.
- `state/server_process.json` — written and deleted by the launcher, acted on by `agent_startup_checks`.
- `state/extension_companions.json` — written by the companion supervisor, consumed as a kill list by the launcher.
- `state/worker_pids.json` and `state/process_ledger.jsonl` — two overlapping process registries with two independent reapers.

**One write seam, many callers**

`task_results/<task_id>.json` (20+ modules) · `state/skills/<skill>/enabled.json` (5) ·
`state/skills/<skill>/review.json` (4) · `archive/rescue/<ts>_<uuid8>/` (5) ·
`repo/.git/ouroboros-update-tx.json` (4) · `repo/.git/ouroboros-update-intent.json`
(2 writers, 4 clearers).

---

## 15. Growth index

Durable paths with **no** pruning mechanism of any kind, grouped by what bounds them.

- **Bounded by an internal cap, file never deleted:** `advisory_overrides.json` (last 10),
  `terminal_deliveries.json` (pending cap + replay cap), `capability_evidence.json`
  (density pairs), `reviewer_slot_last_execution.json` (newest N),
  `scratchpad_blocks.json` (FIFO 10), `improvement-backlog.md` (groom cap),
  `patterns.md` (prompt contract), `artifact_versions/` (last 5 per name),
  `blackboard.jsonl` (byte cap).
- **Bounded only by the reader's window — the file itself grows without limit:**
  `logs/events.jsonl`, `logs/tools.jsonl`, `logs/supervisor.jsonl`,
  `logs/task_reflections.jsonl`, `logs/containment_faults.jsonl`,
  `state/usage_attempts.jsonl`, `state/evolution_checkpoints.jsonl`,
  `memory/identity_journal.jsonl`, `memory/scratchpad_journal.jsonl`,
  `memory/knowledge/patterns_history.jsonl`, `memory/knowledge_history.jsonl`.
- **Unbounded accumulation of whole files or trees:** `task_results/<task_id>.json`,
  `task_results/artifacts/`, `archive/**` (GC-exempt by contract),
  `archive/managed_repo/`, `archive/rescue/`, `state/cx/cache/`, `state/cx/node/`
  (superseded pins), `state/code_intel/<repo_key>/` (one per absolute repo path ever
  used), `state/extension_reconcile/failed/`, `state/review_continuations/corrupt/`
  and `archived/`, `uploads/**`, `claudexor/daemon.log`, `logs/agent_stdout.log`,
  `logs/tasks/*.txt`, `projects/<pid>/**`, `Deliverables/**`, `state/python-userbase/`,
  `state/pycache/`, and every lock file listed in §3.
- **Age-based GC exists in exactly six planes this document tracks**, all cut by
  `retention.py::age_cutoff(get_gc_retention_days())` (the exhaustive production
  `age_cutoff` call-site list): service logs (`tools/services.py::prune_service_logs`,
  startup + terminal-task sweep — and it silently exempts logs above the blob cap),
  consumed one-shot schedule receipts (`supervisor/schedule_time.py::
  prune_consumed_once_records`, cut at the `queue_schedules.py` tick), three startup
  sweeps driven by `server_maintenance.py::_startup_prune_sweeps` — headless task
  drives (`headless.py::prune_headless_task_drives`), task drives (`prune_task_drives`)
  and terminal-root task trees (`prune_task_trees`) — and subagent worktrees
  (`subagent_worktrees.py::prune_orphans` from `_startup_worktree_prune`; the worktree
  root lives BESIDE the data root, see its own rows above). `sweep_stale_temp_files`
  reaps `.tmp.<uuid>` orphans only; nothing else ages out.

---

## 16. Observed divergences from the ARCHITECTURE data layout

Recorded during derivation, under code freeze. **Nothing here was changed**; the list
exists so a later editor does not have to re-derive it.

### 16.1 Documented but not produced by code

1. **`logs/skills/`** ("Optional skill/companion runtime logs") has no producer.
   Nothing joins `"logs"` with `"skills"`; `ouroboros/extension_companion.py` has no
   file sink at all and logs through the root logger into `logs/server.log`. Skill
   state lives at `state/skills/<skill>/`, which the tree documents separately.
2. **`state/crash_report.json`** is read by `agent_startup_checks::inject_crash_report`
   and `context_health`, but nothing in the product tree writes it (only a test does).
   The `RECENT CRASH ROLLBACK` health invariant it feeds cannot currently fire.

### 16.2 Produced but not documented

- **Whole planes:** `data/locks/` (which holds `state.lock` — the serialisation point
  for both `state.json` and `evolution_campaign.json`), `data/skills/` (the entire
  skill *payload* plane, while its state sibling is documented in detail), and the
  update-transaction markers in `repo/.git/` (`ouroboros-update-tx.json`,
  `ouroboros-update-intent.json`, `ouroboros-managed.json`, `ouroboros-bootstrap-pending`).
- **`state/`:** `state.last_good.json`, `advisory_overrides.json`,
  `pending_restart_verify.json`, `crash_report.json`, `worker_pids.json`,
  `process_ledger.jsonl`, `subagent_worktrees.json`, `subagent_last_delegation.json`,
  `reviewer_slot_last_execution.json`, `reviewer_slot_api_fallback.json`,
  `capability_evidence.json`, `headless_tasks/<task_id>/`,
  `acceptance_fence_acks/<token>.json`, `auth_secret.key`, `panic_stop.flag`,
  `owner_restart_no_resume.flag`, `pycache/`, `python-userbase/`, and every lock and
  lock sidecar. Several of these are discussed in ARCHITECTURE prose elsewhere; they
  are simply absent from the tree, so the tree is an orientation rather than the SSOT
  it reads as.
- **`state/skills/<skill>/`:** `grants.json`, `review_job.json`, `clawhub.json`,
  `self_authored.json`, `repair_admission.json`, `chat_id_counter.json`, and the
  `extension_calls/<uuid>.imports/` sibling directories.
- **`logs/`:** `agent_stdout.log`, `server.log`, `launcher.log`,
  `logs/tasks/task_<id>.txt`, and the append-lock sidecars.
- **Elsewhere:** `observability/salvaged/`, `services/<task>/<name>.executor.log`,
  `archive/managed_repo/`, `archive/usage_import/`, `uploads/screenshots/`,
  `uploads/views/`, `uploads/routed-*`, `task_results/<task_id>.json`,
  `memory/knowledge/patterns.md`, `memory/knowledge/index-full.md`,
  `projects/<pid>/journal.jsonl`, `projects/<pid>/workpad.md`, and the top-level
  `subagent_worktrees/` and `projects/` roots — the latter named in passing by the
  `Deliverables/` line ("sibling of projects/") but never shown.

### 16.3 Documented owner or shape differs from code

3. `blobs/<sha256>.json.gz` — the real pattern is `<sha256>.<kind>.gz`;
   `tools/services.py` writes `.txt.gz` blobs that the documented pattern excludes.
4. The observability entries imply a retention regime, but `prune_observability_blobs`
   deletes nothing and reports `preserved_indefinitely`, even with the retention
   environment variable set.
5. `services/<task_id>/<service>.log` carries no lifecycle in the tree although it is
   one of the six age-pruned planes (§15) — and its oversize carve-out is invisible there.
6. `archive/` — "Rotated logs, rescue snapshots" understates both the owner
   (`git_ops_rescue::_create_rescue_snapshot`, six callers, plus a `refs/rescue/*` git
   ref) and the contract (`supervisor/state.py` makes `archive/` GC-exempt and forbids
   adding a sweep).
7. `logs/events.jsonl`, `tools.jsonl` and `supervisor.jsonl` are listed beside
   `chat.jsonl` and `progress.jsonl` with no note that the first three never rotate
   while the last two rotate at 800 KB.
8. `memory/owner_mailbox/` — "Per-task user message files" understates a typed control
   rail written by the supervisor and the agent as well as the owner, and omits that
   it is the one memory-plane path the runtime deletes.
9. `memory/deep_review.md` — "written by deep_self_review task" names the trigger; the
   writer is `ouroboros/agent.py`. Each run destroys the previous report with no
   history file.
10. `knowledge/improvement-backlog.md` — described as a durable advisory backlog
    without saying it is a singleton global store force-rooted away from project scope
    and forked drives, and LLM-groomed under a cap.
11. `state/skills/<skill>/review.json` — the launcher (`_stamp_native_seed_trust` and
    the legacy re-pin), not the reviewer, is the origin of `native_seed` verdicts.
12. `__extension_imports/<pid>-<uuid>/skill/` — "created on load, removed on unload"
    is incomplete: a sweeper also reaps orphans on load and on `reload_all`, and a
    tree whose owner PID is alive is deliberately kept when a peer unloads.
13. `data/claudexor/` — the gloss implies Ouroboros owns the credential profiles and
    runs. It writes exactly two things there (`ouroboros-owned.json` and appended
    `daemon.log` bytes); profiles, runs and the descriptor are engine-authored and
    mutated only over loopback HTTP.
14. `state/cx/` — accurate as far as it goes, but omits `node/<ver>-<platform>/managed-node.json`
    and does not say that `cache/` and superseded pins are never reclaimed.
15. `scheduled_tasks.json` — the tree says consumed one-shot receipts age out past the
    unified GC retention, which the code confirms, while `agent_startup_checks.py`
    simultaneously describes that pruning as an unimplemented remediation. One of the
    two statements is stale.
16. `usage_attempts.jsonl` — documented as the monetary authority without disclosing
    that it is never rotated or compacted, which the code states in-line.
17. `state/code_intel/<repo_key>/` — the tree implies one live cache; the key is a hash
    of the absolute repo path, so every moved or renamed checkout leaks a directory.
18. `task_results/artifacts/<task_id>/` carries no retention sentence although every
    sibling in the block does — and none exists in code either.
19. `api_reset` removes `state`, `memory`, `logs`, `archive`, `task_results`, `uploads`
    and `settings.json`, but **not** `projects/`, `task_drives/`, `task_trees/`,
    `observability/` or `services/`. Neither the tree nor the reset prose says so.
20. `data/settings.json.lock` and the two settings temp-file names (`settings.tmp`
    versus `settings.json.tmp`) are undocumented, and neither temp name matches the
    stale-temp sweep glob.

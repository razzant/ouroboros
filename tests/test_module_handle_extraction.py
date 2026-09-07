"""The module-handle extraction (spec §1.9 batch №8, delta D18) and its invariants.

`supervisor/queue.py` and `supervisor/workers.py` could not be split the way every
other v7 module was. Their bodies read module globals that ``init`` /
``init_queue_refs`` REBIND — PENDING, RUNNING, DRIVE_ROOT, WORKERS and the rest —
so a leaf holding `from supervisor.queue import PENDING` would freeze the object it
saw at import time, and a leaf keeping its own copy would be a second answer to the
same question (67 test sites rebind these names on the parent and must keep
working). The owner approved ONE mechanical exception: a declared parent name X is
read as ``_queue().X`` / ``_pool().X`` — a function-local import of the parent — so
the binding is resolved at call time.

The one-time proof that each moved body is otherwise unchanged (AST-equal modulo
exactly that substitution, over the declared set, with zero other differences) is
recorded in the extraction commits. What is pinned HERE is the property that has to
survive every later edit:

* the parent is reached only through a call-time handle, never a top-level import;
* every declared name is really bound by the parent (a typo would silently match
  nothing and make the proof vacuous);
* the declared set is exactly the set the leaf actually reads through the handle —
  neither a stale name nor an undeclared one;
* and, the load-bearing one, NO leaf reads a parent-owned name directly. That is
  the bug class the handle exists to prevent, and it is the one a later "tidy-up"
  would reintroduce by adding an innocent-looking from-import.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

# leaf -> (parent, handle, declared substitution set)
# v7next transplant note: the reference table (ouroboros_v7_wip @ 9f691656)
# carries one row per extracted leaf across every domain, plus the
# queue/worker/loop/git_ops-specific invariant tests below the parametrized
# trio. On this integration branch each row lands with the lane that
# transplants its domain; only the D16 L-C2 usage split exists so far. The
# domain-specific standalone tests travel with their own rows.
LEAVES: dict[str, tuple[str, str, frozenset[str]]] = {
    "ouroboros/usage_legacy_import.py": ("ouroboros/usage_accounting.py", "_usage", frozenset({
        "_legacy_snapshot", "_locked", "_read_records_locked_cached",
    })),
    # D08 lane rows. The queue/pool declared sets grew past the reference table
    # where post-cutoff upstream helpers stayed on the facade (the deferred
    # cancel/custody organ); every set below is the tool-derived exact read set.
    "supervisor/events_project_routing.py": ("supervisor/events.py", "_events", frozenset({
        "_routing_attachments",
    })),
    # F2 addendum: the FUNCTION_DEBT carrier joined its family leaf once the
    # same-qualname relocation rule landed (D11); its one call-time read is
    # the shared delegation-budget helper still owned by the facade.
    "supervisor/events_schedule_task.py": ("supervisor/events.py", "_events", frozenset({
        "_parent_delegation_budget", "get_max_subagent_depth",
    })),
    "supervisor/queue_schedules.py": ("supervisor/queue.py", "_queue", frozenset({
        "DRIVE_ROOT", "PENDING", "RUNNING", "SCHEDULED_TASKS_FILE", "_queue_lock",
        "enqueue_task", "load_state", "persist_queue_snapshot",
    })),
    "supervisor/worker_chat_lane.py": ("supervisor/workers.py", "_pool", frozenset({
        "DRIVE_ROOT", "REPO_DIR", "_chat_agent_lock", "_ephemeral_chat_lock",
        "_get_chat_agent", "_origin_from_mapping", "_repo_writer_turn_allowed",
        "_report_binding_failure", "get_event_q", "load_state",
        "repo_writer_admission_closed", "send_with_budget",
    })),
    "supervisor/worker_pool_lifecycle.py": ("supervisor/workers.py", "_pool", frozenset({
        "DRIVE_ROOT", "REPO_DIR", "WORKERS", "Worker", "_WORKER_PIDS_FILENAME",
        "_get_ctx", "_reconcile_confirmed_dead_review_owner",
        "_verify_worker_sha_after_spawn", "get_event_q", "kill_workers", "load_state",
        "reconstruct_task_cost", "send_with_budget",
    })),
    "supervisor/worker_promotion.py": ("supervisor/workers.py", "_pool", frozenset({
        "DRIVE_ROOT", "PENDING", "REPO_DIR", "RUNNING", "_announce_created_project",
        "_apply_presence_promotion_authority", "_promoted_scheduled_outcome",
        "_reject_promoted_after_attachment_stage", "_relocate_promoted_attachments",
        "_stage_promoted_initial_attachments",
    })),
    # F2.2 lane rows (4fffefb1), pinned here by the ADOPTION truth wave: the four
    # queue/worker leaves the F2.2 cancel-organ train landed proof-green but never
    # entered this table — the f22 ledger entry claimed they were pinned and the
    # tree did not bear it out, so the three invariants below were not running on
    # them. Sets are the tool-derived exact read sets on these bytes.
    "supervisor/queue_snapshot.py": ("supervisor/queue.py", "_queue", frozenset({
        "ACCEPTANCE_FENCES", "BUDGET_ROOT_FENCES", "DRIVE_ROOT", "PENDING",
        "QUEUE_SEQ_COUNTER_REF", "QUEUE_SNAPSHOT_PATH", "RUNNING", "_queue_lock",
        "append_jsonl", "atomic_write_text", "enqueue_task", "parse_iso_to_ts",
        "persist_queue_snapshot", "restore_invalid_depth_admission", "sort_pending",
    })),
    "supervisor/queue_timeouts.py": ("supervisor/queue.py", "_queue", frozenset({
        "DRIVE_ROOT", "FINALIZATION_GRACE_SEC", "HEARTBEAT_STALE_SEC", "PENDING",
        "QUEUE_MAX_RETRIES", "RUNNING", "_enforce_task_timeouts_locked",
        "_ensure_reaper_started", "_has_live_descendant", "_has_pending_descendant",
        "_is_descendant_of", "_queue_lock", "_reap_queue",
        "_request_finalization_grace", "_subtree_progressing", "_task_deadline_ts",
        "_task_drive_for_task", "coerce_chat_identity", "get_per_call_timeout_ceiling_sec",
        "get_task_abs_ceiling_sec", "get_task_idle_timeout_sec", "load_state",
        "persist_queue_snapshot",
    })),
    "supervisor/worker_assignment.py": ("supervisor/workers.py", "_pool", frozenset({
        "DRIVE_ROOT", "PENDING", "RUNNING", "WORKERS",
        "_assignment_depth_reservation_admits", "_cancel_unauthorized_evolution",
        "_drop_assignable_evolution_tasks", "_drop_cancelled_pending",
        "_emit_task_done_terminal", "_evolution_assignment_error",
        "_invalid_depth_deferred", "_normalize_pending_task_depth",
        "_quarantine_invalid_pending_depths",
        "_retry_terminalization_pending_for_assignment", "_running_subagent_count",
        "_terminalize_invalid_pending_depth", "append_jsonl", "load_state",
        "reconstruct_task_cost", "repo_writer_task_allowed", "send_with_budget",
        "utc_now_iso",
    })),
    "supervisor/worker_health.py": ("supervisor/workers.py", "_pool", frozenset({
        "CRASH_TS", "DRIVE_ROOT", "QUEUE_MAX_RETRIES", "RUNNING", "WORKERS",
        "_LAST_SPAWN_TIME", "_SPAWN_GRACE_SEC", "_emit_task_done_terminal",
        "_ensure_workers_healthy_locked", "_reconcile_confirmed_dead_review_owner",
        "_worker_crash_storm_detected", "append_jsonl", "coerce_chat_identity",
        "get_event_q", "kill_workers", "load_state", "reconstruct_task_cost",
        "respawn_worker", "send_with_budget", "terminal_task_metadata",
        "utc_now_iso",
    })),
    # D10 lane rows (oracle ouroboros_v7_wip @ 9f691656). git_ops leaves carry
    # the reference's G1 `_go()` sets re-derived on tip bytes (safe_restart and
    # prepare_managed_update stayed facade defs — the gate refuses f-string
    # reads of rebindable globals — so their reads left the sets). The tools/
    # D10/D35 lane rows (oracle ouroboros_v7_wip @ 9f691656). git_ops leaves
    # carry the reference's G1 `_go()` sets re-derived on tip bytes; D35 teaches
    # the proof to treat f-string reads like ordinary call-time reads and moves
    # the final safe_restart/prepare_managed_update spans. The tools/
    # git leaves declare EVERY parent-scope name their spans read at call time
    # (the reference cut them with plain leaf imports, but the tip monolith's
    # test surface monkeypatches those names on the parent — the module-global
    # patchability contract survives only through `_git()`); the few f-string
    # reads the gate cannot rewrite stay import-bound and are named in each
    # leaf docstring.
    "ouroboros/tools/git_evolution.py": ("ouroboros/tools/git.py", "_git", frozenset({
        "_evolution_commit_authority", "_preserve_evolution_orphan",
        "_record_commit_attempt", "_record_evolution_commit_intent", "run_cmd",
    })),
    "ouroboros/tools/git_plumbing.py": ("ouroboros/tools/git.py", "_git", frozenset({
        "_BINARY_EXTENSIONS", "acquire_exclusive_file_lock", "format_protected_paths",
        "get_runtime_mode", "run_cmd", "system_repo_dir_for", "unlink_lockfile",
        "write_text",
    })),
    "ouroboros/tools/git_repo_edit.py": ("ouroboros/tools/git.py", "_git", frozenset({
        "_CONTENT_OMITTED_PREFIX", "_authorized_managed_update_resolver",
        "_binding_repo_rel", "_binding_targets_system_repo", "_check_shrink_guard",
        "_current_runtime_mode", "_data_skill_path", "_invalidate_advisory",
        "_str_match_replace", "build_resolved_resource_binding", "core_patch_notice",
        "cross_skill_redirect_error", "decide_payload_short_form",
        "is_protected_runtime_path", "is_skill_control_plane_path",
        "mode_allows_protected_write", "normalize_repo_path",
        "normalize_task_constraint", "protected_paths_in",
        "protected_write_block_message", "resolve_payload_path", "safe_relpath",
        "write_text",
    })),
    "ouroboros/tools/git_review_cycle.py": ("ouroboros/tools/git.py", "_git", frozenset({
        "IDENTICAL_DIFF_BLOCK_REASON", "_DOC_ONLY_EXTENSIONS", "_acquire_git_lock",
        "_advisory_and_tests_gate", "_aggregate_review_verdict",
        "_authorized_managed_update_resolver", "_check_overlapping_review_attempt",
        "_current_runtime_mode", "_ensure_gitignore", "_finalize_blocked_review",
        "_finalize_pending_review", "_fingerprint_staged_diff", "_free_cycle_gate",
        "_handle_advisory_pre_review", "_handle_revalidation_failure", "_install_paid_dispatch_stamp",
        "_protected_paths_block_message", "_reconcile_and_clear_review_roster",
        "_record_commit_attempt", "_release_git_lock", "_reconcile_advisory_before_preparation",
        "_reset_commit_review_state",
        "_review_binding_precondition_error", "_review_custody_pending",
        "_review_cycle_infra_failure", "_run_parallel_review",
        "_run_reviewed_stage_cycle", "_stage_candidate_for_review",
        "_subject_binding_mismatch_outcome", "_unstage_binaries",
        "classify_review_block", "handle_revalidation_failure",
        "mode_allows_protected_write", "paths_from_name_status",
        "protected_paths_in", "run_cmd", "safe_relpath",
    })),
    "ouroboros/tools/git_vcs_ops.py": ("ouroboros/tools/git.py", "_git", frozenset({
        "_acquire_git_lock", "_binding_relative_path", "_ff_pull",
        "_limit_git_output", "_release_git_lock",
        "_run_git_network_cmd", "_vcs_binding", "_vcs_result",
        "binding_targets_system_repo", "build_resolved_resource_binding",
        "is_protected_runtime_path", "normalize_repo_path", "protected_paths_in",
        "run_cmd", "safe_relpath",
    })),
    "supervisor/git_ops_remotes.py": ("supervisor/git_ops.py", "_go", frozenset({
        "BRANCH_DEV", "REPO_DIR", "_configure_credential_helper",
        "_git_network_bounded", "_has_remote", "configure_remote",
        "ensure_official_update_remote", "git_capture",
    })),
    "supervisor/git_ops_rescue.py": ("supervisor/git_ops.py", "_go", frozenset({
        "BRANCH_DEV", "DRIVE_ROOT", "REPO_DIR", "_atomic_write_bytes",
        "_collect_repo_sync_state", "_copy_untracked_for_rescue",
        "_create_rescue_snapshot", "_git_dir", "_link_rescue_to_evolution_transaction",
        "_list_remotes", "_managed_remote_branch_for", "_managed_remote_name",
        "_read_managed_repo_meta", "_run_git_process_bounded", "append_jsonl",
        "atomic_write_text", "rescue_before_destructive_rollback",
        "rescue_git_capture", "utc_now_iso",
    })),
    "supervisor/git_ops_reset.py": ("supervisor/git_ops.py", "_go", frozenset({
        "BRANCH_DEV", "BRANCH_STABLE", "DRIVE_ROOT", "REPO_DIR",
        "_admission_gate_for_unsynced_tree",
        "_clear_bootstrap_pin_marker", "_clear_update_intent",
        "_collect_repo_sync_state", "_compute_ref_ahead_count",
        "_create_rescue_snapshot", "_git_dir", "_guard_live_repo_destructive_git",
        "_has_remote", "_maybe_repair_git_index", "_pin_to_bundle_sha_on_bootstrap",
        "_preserve_branch_for_official_reset", "_read_managed_repo_meta",
        "_read_update_intent", "_ref_points_at_ref", "_rescue_untracked_incomplete",
        "_run_git_resilient", "_update_source", "append_jsonl", "git_capture",
        "checkout_and_reset", "current_drive_root", "import_test", "load_state",
        "preserve_local_ref_branch", "rescue_git_capture", "save_state",
        "sync_runtime_dependencies", "utc_now_iso",
    })),
    "supervisor/git_ops_updates.py": ("supervisor/git_ops.py", "_go", frozenset({
        "BRANCH_DEV", "OFFICIAL_UPDATE_REMOTE_URL", "_collect_repo_sync_state",
        "_compute_ref_ahead_count", "_create_rescue_snapshot", "_git_network_bounded",
        "_has_remote", "_list_remotes", "_managed_remote_name",
        "_managed_update_target", "_read_managed_repo_meta",
        "_rescue_untracked_incomplete", "_resolve_managed_update_target",
        "_write_update_intent", "append_jsonl", "current_drive_root",
        "ensure_official_update_remote", "git_capture", "git_fetch_bounded", "load_state",
        "managed_branch_defaults", "managed_update_remote_url",
        "preserve_local_ref_branch", "utc_now_iso",
    })),
    # D07 lane rows (oracle ouroboros_v7_wip @ 9f691656, DEL1 splits re-derived
    # on tip bytes). The custody reconcile set dropped the reference names the
    # tip drift stopped reading (STARTED, START_REQUESTED, _CUSTODY, _iter_rows,
    # event_log_path) and gained retire_settled_registrations (the upstream
    # retirement-decoupling train); control_scheduling declares only
    # load_settings — tests rebind it on the control facade; the projection-only
    # control_subagent_spec/control_task_results leaves carry no handle reads
    # and stay off this table (D08 control-leaf precedent).
    "ouroboros/delegate_custody_reconcile.py": ("ouroboros/delegate_custody.py", "_custody", frozenset({
        "RECONCILED", "RunCustody", "START_FAILED", "TERMINAL_STATES", "_reconcile_one",
        "cancel_and_verify", "close_absent_run", "daemon_says_absent", "emit",
        "is_terminal", "open_runs", "output_disposition", "pending_invocations",
        "record_containment_fault", "record_settled_unread", "record_started",
        "replay", "retire_settled_registrations", "settle_run",
    })),
    "ouroboros/tools/delegate_payload_patch.py": ("ouroboros/tools/delegate_integration.py", "_di", frozenset({
        "_rebind_payload_reference", "_resolved", "payload_content_hash",
    })),
    "ouroboros/tools/subagent_integration_delegated.py": ("ouroboros/tools/subagent_integration.py", "_si", frozenset({
        "_baseline_drifted_paths", "_capture_at_disposition", "_locked_apply",
        "_patch_touched_paths", "_sha256_file", "_stageable_paths",
        "_target_is_system_repo", "_write_verdict", "get_runtime_mode",
    })),
    "ouroboros/tools/control_scheduling.py": ("ouroboros/tools/control.py", "_ctl", frozenset({
        "load_settings",
    })),
    # D07 finisher row (oracle ouroboros_v7_wip @ 9f691656, ledger rows
    # 3468-3476). The ledger's tools/delegate_terminal.py name collided with
    # upstream's own ouroboros/delegate_terminal.py, so the leaf landed as
    # delegate_terminal_evidence.py (owner fork F-2=A). The declared set is
    # maximal on tip bytes: EVERY parent-scope name the moved spans read at
    # call time goes through `_delegate()` (the reference cut this leaf with
    # plain preamble imports and declared only _emit).
    "ouroboros/tools/delegate_terminal_evidence.py": ("ouroboros/tools/delegate.py", "_delegate", frozenset({
        "_Breach", "_PAYLOAD_ENVELOPE_HEADROOM", "_emit", "_home_isolation_breach",
        "_preview_payload", "_resolve_full_primary_output", "_stage_full_output",
        "_widened_access", "add_terminal_source_verification", "custody",
        "home_nested_under_operator_home", "tool_result_limit",
    })),
    # D01 lane rows (L-B loop split + D38 agent dispatch), declared sets
    # re-derived on tip bytes by the transplant tool (reference table:
    # ouroboros_v7_wip @ 9f691656; tip additions over the reference are the
    # same-leaf members tip tests rebind on ouroboros.loop, and
    # _mark_owner_stop_control_drained, which upstream re-homed into
    # supervisor/owner_stop.py while tests still patch it on the loop).
    "ouroboros/agent_dispatch.py": ("ouroboros/agent.py", "_agent", frozenset({
        "envelope_from_task", "write_task_result",
    })),
    "ouroboros/post_task_synthesis.py": ("ouroboros/agent_task_pipeline.py", "_atp", frozenset({
        "_is_root_post_task", "load_task_result",
    })),
    "ouroboros/loop_acceptance.py": ("ouroboros/loop.py", "_loop", frozenset({
        "_append_or_merge_user_message", "_end_task_acceptance_fence",
        "_set_acceptance_decision", "_task_acceptance_eligible", "get_task_review_mode",
    })),
    "ouroboros/loop_acceptance_review.py": ("ouroboros/loop.py", "_loop", frozenset({
        "_append_or_merge_user_message", "_begin_task_acceptance_fence",
        "_collect_acceptance_obligations", "_dispose_obligations_on_clean_pass",
        "_end_task_acceptance_fence", "_execute_task_acceptance_panel",
        "_extract_plain_text_from_content", "_format_obligations_clause",
        "_latch_final_answer_marker", "_mark_root_acceptance_checkpoint",
        "_open_acceptance_obligations", "_set_acceptance_decision",
        "_supersede_task_acceptance_for_evidence_change",
        "_supersede_task_acceptance_for_owner_followup", "_task_acceptance_eligible",
        "_task_acceptance_owner_generation_changed", "_task_acceptance_subtree_snapshot",
        "get_review_enforcement", "get_task_review_mode",
    })),
    # The P6 wrap-up rail (upstream d4fd933c/8bbdac50) prices the forced prompt
    # here, so the budget leaf now reads the forced-finalization prompt tail,
    # its preparer and the server-web predicate through the handle too.
    "ouroboros/loop_budget.py": ("ouroboros/loop.py", "_loop", frozenset({
        "DeliveryCandidate", "_CHILD_ABSORPTION_HOLD_CONTROL",
        "_DELIVERY_HOLD_CONTROLS", "_FORCED_BEST_EFFORT_TAIL",
        "_SKILL_ACTION_HOLD_CONTROL",
        "_TREE_ACCOUNTING_MAX_STALE_SEC", "_append_or_merge_user_message",
        "_arm_delivery_control",
        "_compose_delivery_suffix", "_current_delivery_candidate",
        "_delivery_evidence_state", "_emit_checkpoint_event",
        "_finalize_forced_services", "_finalize_task_services",
        "_force_plan_disclosure", "_forced_fallback_result",
        "_forced_final_answer", "_forced_swarm_router_result",
        "_hold_delivery_for_skill_action", "_live_delivery_candidate",
        "_loop_tree_accounting", "_prepare_forced_prompt",
        "_publish_delivery_candidate",
        "_record_forced_finalization", "_server_web_allowed_by_task",
        "_undispositioned_children",
    })),
    "ouroboros/loop_delivery.py": ("ouroboros/loop.py", "_loop", frozenset({
        "DeliveryCandidate", "_LoopExitContext",
        "_append_or_merge_user_message", "_arm_delivery_control",
        "_child_disposition_state", "_compose_delivery_suffix",
        "_compute_subagent_handoff", "_delivery_evidence_state",
        "_delivery_replace_required", "_direct_child_results",
        "_drain_incoming_messages", "_enforce_swarm_actions",
        "_extract_plain_text_from_content", "_finalize_task_services",
        "_force_plan_disclosure", "_forced_orphan_note",
        "_handle_text_response", "_live_delivery_candidate",
        "_load_direct_child_results", "_maybe_early_finalize",
        "_maybe_enforce_child_absorption_gate",
        "_maybe_inject_finalization_nudges", "_merge_finalization_trace",
        "_project_child_result_dispositions", "_publish_delivery_candidate",
        "_replace_delivery_candidate", "_resolve_delivery_control",
        "_run_task_acceptance_review_once", "_service_finalization_evidence",
        "_supersede_delivery_acceptance_binding",
        "_supersede_task_acceptance_for_evidence_change",
        "_supersede_task_acceptance_for_owner_followup",
    })),
    "ouroboros/loop_forced_finalization.py": ("ouroboros/loop.py", "_loop", frozenset({
        "DeliveryCandidate", "_LoopExitContext",
        "_append_or_merge_user_message", "_call_forced_model_once",
        "_child_disposition_state", "_claimed_child_dispositions",
        "_compose_delivery_suffix", "_current_delivery_candidate",
        "_degrade_retained_delivery_candidate", "_delivery_evidence_state",
        "_delivery_replace_required", "_direct_child_results",
        "_drain_forced_owner_directives", "_drain_incoming_messages",
        "_end_task_acceptance_fence", "_finalize_forced_services",
        "_finalize_task_services", "_force_plan_decision",
        "_force_plan_disclosure", "_force_plan_reminder",
        "_forced_delegation_note", "_forced_fallback_result",
        "_forced_final_answer", "_forced_orphan_note",
        "_forced_swarm_router_result", "_forced_unaccepted_binding",
        "_live_delivery_candidate", "_load_direct_child_results",
        "_merge_finalization_trace", "_resolve_forced_delivery_control_body",
        "_project_child_result_dispositions", "_publish_delivery_candidate",
        "_prepare_forced_prompt",
        "_record_forced_acceptance_bypass", "_record_forced_finalization",
        "_replace_delivery_candidate", "_run_task_acceptance_review_once",
        "_server_web_allowed_by_task",
        "_service_finalization_evidence",
        "_supersede_task_acceptance_for_owner_followup",
        "_swarm_handoff_attempt", "call_llm_with_retry",
        # Upstream e10b3cf3 replaced this leaf's inline dangling-revision write
        # with the acceptance leaf's `terminalize_dangling_revision`, so the raw
        # decision writer is no longer read here.
        "terminalize_dangling_revision",
    })),
    "ouroboros/loop_messages.py": ("ouroboros/loop.py", "_loop", frozenset({
        "_record_owner_directive",
    })),
    "ouroboros/loop_model_call.py": ("ouroboros/loop.py", "_loop", frozenset({
        "_RoundModelCallContext", "_account_compaction_usage", "_call_round_model",
        "_context_overflow_retries", "_context_reclaim_materializations",
        "_context_reclaim_passes", "_dispatch_round_model", "_emit_checkpoint_event",
        "_measure_round_main_fit", "_rebind_context_fit_plan", "_run_main_reclaim",
        "_server_web_allowed_by_task", "_task_deadline_epoch", "call_llm_with_retry",
        "compact_tool_history_llm", "last_physical_attempt_capture", "seal_task_transcript",
    })),
    "ouroboros/loop_nudges.py": ("ouroboros/loop.py", "_loop", frozenset({
        "_TREE_ACCOUNTING_MAX_STALE_SEC", "_append_or_merge_user_message",
        "_emit_checkpoint_event", "_extract_plain_text_from_content",
        "_force_plan_decision", "_loop_tree_accounting", "_skill_finalization_message",
        "_skill_names_touched_by_trace", "get_review_enforcement",
    })),
    # F2.3a D06 lane rows (oracle ouroboros_v7_wip @ 9f691656, re-cut on tip
    # bytes). Declared sets are the tool-derived exact read sets; the
    # projection-only review_records leaf carries no handle reads and stays
    # off this table (D07/D08 precedent). The state custody leaf is a NEW
    # owner (post-cutoff upstream growth, decision 5.3=B one-cut).
    "ouroboros/review_evidence_sections.py": ("ouroboros/review_evidence.py", "_ev", frozenset({
        "truncate_review_artifact", "truncate_within_limit",
    })),
    "ouroboros/review_projection.py": ("ouroboros/review_substrate.py", "_sub", frozenset({
        "DIALOGUE_STATUS_VALUES", "HARDNESS_ADVISORY_VISIBLE", "HARDNESS_HARD_GATE",
        "MAX_PROJECTED_ACTOR_FINDINGS", "OUTCOME_TIER_BEST_EFFORT", "OUTCOME_TIER_BLOCKED",
        "OUTCOME_TIER_SOLVED", "disclosed_list_projection", "panel_reason",
        "projected_finding_row", "provider_for_model", "redact_projection",
        "review_binding_hash", "review_executions_from_actor_usage",
    })),
    "ouroboros/review_state_custody.py": ("ouroboros/review_state.py", "_rs", frozenset({
        "_truncate_review_artifact", "_utc_now", "update_state",
    })),
    "ouroboros/review_state_model.py": ("ouroboros/review_state.py", "_rs", frozenset({
        "CommitReadinessDebtItem", "ObligationItem", "_DEFAULT_TOOL_NAME",
        "_LEGACY_CURRENT_REPO_KEY", "_MAX_ATTEMPT_HISTORY", "_MAX_COMMIT_READINESS_DEBTS",
        "_MAX_RUN_HISTORY", "_OPEN_COMMIT_READINESS_DEBT_STATUSES", "_allocate_prefixed_id",
        "_attempt_has_active_review_custody", "_attempt_history_evictable",
        "_attempt_identity_tuple", "_attempt_review_roster_rows",
        "_commit_readiness_debts_view", "_dedupe_strings", "_filter_lifecycle_records",
        "_filter_repo_scope", "_looks_like_public_obligation_id",
        "_make_obligation_fingerprint", "_max_iso_ts", "_merge_attempt", "_min_iso_ts",
        "_normalize_obligation_item_key", "_parse_iso_ts", "_review_roster_row_is_pending",
        "_strip_attempt_heavy_payload", "_utc_now",
    })),
    "ouroboros/review_state_records.py": ("ouroboros/review_state.py", "_rs", frozenset({
        "_strip_attempt_heavy_payload", "_truncate_review_reason",
    })),
    "ouroboros/review_verdict.py": ("ouroboros/review_substrate.py", "_sub", frozenset({
        "truncate_review_artifact",
    })),
    "ouroboros/tools/review_file_pack.py": ("ouroboros/tools/review_helpers.py", "_rh", frozenset({
        "logger", "redact_prompt_secrets",
    })),
    "ouroboros/tools/review_multi_model.py": ("ouroboros/tools/review.py", "_rev", frozenset({
        "LLMClient", "REVIEW_JSON_ARRAY_CONTRACT", "TYPED_FAILURE_FACT_KEYS", "_REPO_ROOT",
        "_cfg", "_parse_model_response", "_review_operation_fields",
        "_owner_deadline_at", "_review_query_error_payload", "load_governance_doc",
        "review_drive_root", "slot_id_for_row", "truncate_review_artifact",
    })),
    "ouroboros/tools/review_prompt_text.py": ("ouroboros/tools/review_helpers.py", "_rh", frozenset({
        "sanitize_tool_result_for_log",
    })),
    "ouroboros/tools/scope_review_pack.py": ("ouroboros/tools/scope_review.py", "_sr", frozenset({
        "BINARY_EXTENSIONS", "CRITICAL_FINDING_CALIBRATION", "ReviewContextAtlasRequest",
        "StagedDiffUnavailable", "_SCOPE_FAILCLOSED_WINDOW", "_SCOPE_MODEL_CONTEXT_WINDOW",
        "_SENSITIVE_EXTENSIONS", "_SENSITIVE_NAMES", "_TouchedContextStatus",
        "_compute_touched_status", "_effective_scope_input_limit", "_get_scope_model",
        "_load_canonical_context_docs", "_scope_window", "_shared_build_rebuttal_section",
        "_shared_review_history_section", "atlas_assembly_failed",
        "atlas_assembly_failure_reason", "atlas_hard_budget_overflowed",
        "atlas_required_beyond_diff", "atlas_unassembled_required", "build_goal_section",
        "build_scope_review_prompt", "build_scope_section", "build_touched_file_pack",
        "capture_staged_diff", "compile_review_context_atlas", "estimate_tokens",
        "load_checklist_section", "parse_git_name_status", "run_cmd",
        "staged_path_is_binary",
    })),
    # F2.3b D06 lane rows (advisory re-derive on the native-episode form; the
    # scope budget re-derive after PR #383). Same-leaf members that tests
    # monkeypatch on the facades are declared too, so the patch points keep
    # binding through the handle.
    "ouroboros/tools/preflight_review_prompt.py": ("ouroboros/tools/claude_advisory_review.py", "_car", frozenset({
        "CRITICAL_FINDING_CALIBRATION", "_build_blocking_history_section",
        "_get_changed_file_list", "_get_staged_diff", "_mandatory_read_pointer",
        "build_blocking_findings_json_section", "build_goal_section",
        "build_scope_section", "build_skill_host_context", "load_checklist_section",
        "load_governance_doc", "load_state", "make_repo_key",
    })),
    "ouroboros/tools/preflight_review_run.py": ("ouroboros/tools/claude_advisory_review.py", "_car", frozenset({
        "SEVERITY_DRIVEN_ITEMS", "_advisory_native_model", "_advisory_review_diff",
        "_build_advisory_prompt", "_format_advisory_error", "_get_changed_file_list",
        "_get_runtime_diagnostics", "_llm_extract_advisory_items",
        "_mandatory_read_corpus_chars", "_maybe_overflow_skip", "_predispatch_size_skip", "_persist_preflight_record",
        "_run_advisory_delegated",
        "_run_advisory_native", "_syntax_preflight_staged_py_files",
        "advisory_gate_unavailability_reason", "build_advisory_changed_context",
        "emit_review_event", "emit_review_usage", "empty_array_is_verified_clean",
        "extract_json_array", "get_finalization_grace_sec",
        "owner_deadline_exhausted_for_context",
    })),
    "ouroboros/tools/scope_review_budget.py": ("ouroboros/tools/scope_review.py", "_sr", frozenset({
        "_effective_scope_input_limit", "_get_scope_model", "_scope_window",
    })),
    "ouroboros/loop_round_limits.py": ("ouroboros/loop.py", "_loop", frozenset({
        "DeliveryCandidate", "_append_or_merge_user_content",
        "_append_or_merge_user_message", "_current_delivery_candidate",
        "_emit_checkpoint_event", "_finalize_forced_services",
        "_forced_fallback_result", "_forced_final_answer",
        "_handle_forced_finalization", "_owner_marked_content",
        "_provider_unavailable_result", "_record_owner_directive",
        "_task_deadline_epoch", "compact_tool_history_llm", "utc_now",
    })),
}


def _tree(rel: str) -> ast.Module:
    return ast.parse((REPO / rel).read_text(encoding="utf-8"))


def _module_bindings(tree: ast.Module) -> set[str]:
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            # Tuple targets unfold at any depth (the git_ops facade binds its
            # bounded-network aliases as `A, B = x, y` — D10 lane adaptation,
            # same class as the D12 config-extraction Tuple-target fix).
            stack = list(node.targets)
            while stack:
                t = stack.pop()
                if isinstance(t, ast.Name):
                    bound.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    stack.extend(t.elts)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"):
            # Annotation-only bindings: lazy under future annotations, never
            # imported at runtime, so nothing is frozen at import time.
            for sub in node.body:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    bound.update(a.asname or a.name.split(".")[0] for a in sub.names)
    return bound


def _handle_reads(tree: ast.AST, handle: str) -> set[str]:
    reads: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == handle):
            reads.add(node.attr)
    return reads


@pytest.mark.parametrize("leaf", sorted(LEAVES))
def test_each_leaf_reaches_its_parent_only_through_a_call_time_handle(leaf: str) -> None:
    parent, handle, _declared = LEAVES[leaf]
    parent_module = parent[:-3].replace("/", ".")
    tree = _tree(leaf)
    for node in tree.body:  # module scope only: a lazy import inside the handle is the point
        if isinstance(node, ast.ImportFrom):
            assert node.module != parent_module, f"{leaf} imports its parent at module scope"
        if isinstance(node, ast.Import):
            assert all(a.name != parent_module for a in node.names), leaf
    handles = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == handle]
    assert len(handles) == 1, f"{leaf}: expected exactly one {handle}() definition"
    assert [n for n in ast.walk(handles[0]) if isinstance(n, (ast.Import, ast.ImportFrom))], (
        f"{leaf}: {handle}() must import the parent at call time"
    )


@pytest.mark.parametrize("leaf", sorted(LEAVES))
def test_the_declared_set_is_exactly_what_the_leaf_reads_through_the_handle(leaf: str) -> None:
    parent, handle, declared = LEAVES[leaf]
    actual = _handle_reads(_tree(leaf), handle)
    assert actual == set(declared), (
        f"{leaf}: declared {sorted(declared)} but reads {sorted(actual)}"
    )
    bound = _module_bindings(_tree(parent))
    missing = sorted(set(declared) - bound)
    assert missing == [], f"{leaf}: declared names absent from {parent}: {missing}"


@pytest.mark.parametrize("leaf", sorted(LEAVES))
def test_no_leaf_reads_a_parent_owned_name_directly(leaf: str) -> None:
    """The bug the handle exists to prevent: a direct read freezes the binding the
    leaf saw at import time, so `init` rebinding the parent's name — or a test doing
    the same — would leave this module looking at the old object forever."""
    parent, _handle, _declared = LEAVES[leaf]
    leaf_tree = _tree(leaf)
    parent_defs: set[str] = set()
    for node in _tree(parent).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parent_defs.add(node.name)
        elif isinstance(node, ast.Assign):
            # Same recursive unfold as _module_bindings (wave-5 conformance:
            # the shallow variant here let git_ops' tuple-bound aliases
            # escape the direct-read guard).
            stack = list(node.targets)
            while stack:
                t = stack.pop()
                if isinstance(t, ast.Name):
                    parent_defs.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    stack.extend(t.elts)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            parent_defs.add(node.target.id)
    own = _module_bindings(leaf_tree)
    direct = {
        node.id for node in ast.walk(leaf_tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        and node.id in parent_defs and node.id not in own
    }
    assert direct == set(), f"{leaf} reads {sorted(direct)} directly instead of through the handle"


def test_queue_snapshot_path_has_a_single_authority():
    """MIGRATION row 1030 (D18): the queue snapshot path is owned by
    ``supervisor.queue`` alone. ``supervisor.state`` used to carry a second
    global that only agreed with it because both ``init()`` calls received the
    same drive root — an addressing split for durable state."""
    from supervisor import queue as queue_mod
    from supervisor import state as state_mod

    assert hasattr(queue_mod, "QUEUE_SNAPSHOT_PATH")
    assert not hasattr(state_mod, "QUEUE_SNAPSHOT_PATH")

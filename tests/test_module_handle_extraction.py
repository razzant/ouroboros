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
# The ouroboros/loop_*.py rows are the L-B loop split: same owner-approved
# mechanical exception, with `_loop()` as the call-time handle. Their declared
# sets include the loop members and the loop-imported names that tests rebind
# on the parent (`loop.call_llm_with_retry` and friends), so monkeypatching the
# loop keeps intercepting the moved bodies. An `if TYPE_CHECKING:` import of
# annotation-only names does not violate the no-top-level-import rule below:
# it never executes, so nothing is frozen at import time.
# The supervisor/git_ops_*.py rows are the G1 git_ops split (`_go()` as the
# call-time handle): `init` rebinds REPO_DIR/DRIVE_ROOT/BRANCH_* on the parent
# and tests monkeypatch the capture plumbing and sibling members there, so the
# moved bodies read every parent-addressable name through the handle. That
# includes `utc_now_iso`, which the parent re-exports for exactly this reason
# (supervisor/update_recovery.py already reads it as `_g.utc_now_iso`): a leaf
# from-importing it would put the moved bodies on a binding nothing addressing
# the parent can reach. The one name held back is the logger: each leaf binds
# `logging.getLogger("supervisor.git_ops")`, which IS the parent's logger object,
# so moved records keep their `%(name)s` (the ledger's disclosed logger residual).
LEAVES: dict[str, tuple[str, str, frozenset[str]]] = {
    # The DEL1 delegate-family rows (delta D36) follow the same owner-approved
    # mechanical exception: parent names that tests rebind on the historical
    # module surface are read through the call-time handle, everything else
    # moved verbatim or imports from its true owner.
    "ouroboros/delegate_custody_reconcile.py": ("ouroboros/delegate_custody.py", "_custody", frozenset({
        "RECONCILED", "RunCustody", "STARTED", "START_FAILED", "START_REQUESTED", "TERMINAL_STATES",
        "_CUSTODY", "_iter_rows", "_reconcile_one", "cancel_and_verify", "close_absent_run",
        "daemon_says_absent", "emit", "event_log_path", "is_terminal", "open_runs",
        "output_disposition", "pending_invocations", "record_containment_fault",
        "record_settled_unread", "record_started", "replay", "settle_run",
    })),
    "ouroboros/tools/delegate_terminal.py": ("ouroboros/tools/delegate.py", "_delegate", frozenset({
        "_emit",
    })),
    "ouroboros/tools/delegate_payload_patch.py": ("ouroboros/tools/delegate_integration.py", "_di", frozenset({
        "_rebind_payload_reference", "_resolved", "payload_content_hash",
    })),
    "ouroboros/tools/subagent_integration_delegated.py": ("ouroboros/tools/subagent_integration.py", "_si", frozenset({
        "_baseline_drifted_paths", "_capture_at_disposition", "_locked_apply", "_patch_touched_paths",
        "_sha256_file", "_stageable_paths", "_target_is_system_repo", "_write_verdict",
        "get_runtime_mode",
    })),
    "ouroboros/loop_messages.py": ("ouroboros/loop.py", "_loop", frozenset({
        "_record_owner_directive"
    })),
    "ouroboros/loop_acceptance.py": ("ouroboros/loop.py", "_loop", frozenset({
        "_append_or_merge_user_message", "_end_task_acceptance_fence", "_set_acceptance_decision",
        "_task_acceptance_eligible", "get_task_review_mode"
    })),
    "ouroboros/loop_acceptance_review.py": ("ouroboros/loop.py", "_loop", frozenset({
        "_append_or_merge_user_message", "_begin_task_acceptance_fence",
        "_collect_acceptance_obligations", "_dispose_obligations_on_clean_pass",
        "_end_task_acceptance_fence", "_extract_plain_text_from_content", "_format_obligations_clause",
        "_latch_final_answer_marker", "_mark_root_acceptance_checkpoint",
        "_open_acceptance_obligations", "_set_acceptance_decision",
        "_supersede_task_acceptance_for_evidence_change",
        "_supersede_task_acceptance_for_owner_followup", "_task_acceptance_eligible",
        "_task_acceptance_owner_generation_changed", "_task_acceptance_subtree_snapshot",
        "get_review_enforcement", "get_task_review_mode"
    })),
    "ouroboros/loop_round_limits.py": ("ouroboros/loop.py", "_loop", frozenset({
        "DeliveryCandidate", "_append_or_merge_user_message", "_current_delivery_candidate",
        "_emit_checkpoint_event", "_finalize_forced_services", "_forced_fallback_result",
        "_forced_final_answer", "_handle_forced_finalization", "_last_assistant_text",
        "_live_delivery_candidate", "_owner_marked_content", "_record_owner_directive",
        "_task_deadline_epoch", "compact_tool_history_llm", "utc_now"
    })),
    "ouroboros/loop_nudges.py": ("ouroboros/loop.py", "_loop", frozenset({
        "_TREE_ACCOUNTING_MAX_STALE_SEC", "_append_or_merge_user_message", "_emit_checkpoint_event",
        "_extract_plain_text_from_content", "_force_plan_decision", "_loop_tree_accounting",
        "get_review_enforcement"
    })),
    "ouroboros/loop_model_call.py": ("ouroboros/loop.py", "_loop", frozenset({
        "_RoundModelCallContext", "_account_compaction_usage", "_call_round_model",
        "_context_overflow_retries", "_context_reclaim_materializations", "_context_reclaim_passes",
        "_emit_checkpoint_event", "_rebind_context_fit_plan", "_server_web_allowed_by_task",
        "_task_deadline_epoch", "call_llm_with_retry", "compact_tool_history_llm",
        "last_physical_attempt_capture", "seal_task_transcript"
    })),
    "ouroboros/loop_budget.py": ("ouroboros/loop.py", "_loop", frozenset({
        "DeliveryCandidate", "_TREE_ACCOUNTING_MAX_STALE_SEC", "_arm_delivery_control",
        "_compose_delivery_suffix", "_current_delivery_candidate", "_delivery_evidence_state",
        "_emit_checkpoint_event", "_finalize_forced_services", "_finalize_task_services",
        "_force_plan_disclosure", "_forced_fallback_result", "_forced_final_answer",
        "_forced_swarm_router_result", "_live_delivery_candidate", "_loop_tree_accounting",
        "_publish_delivery_candidate", "_record_forced_finalization"
    })),
    "ouroboros/loop_delivery.py": ("ouroboros/loop.py", "_loop", frozenset({
        "DeliveryCandidate", "_LoopExitContext", "_append_or_merge_user_message",
        "_arm_delivery_control", "_child_disposition_state", "_compose_delivery_suffix",
        "_delivery_evidence_state", "_delivery_replace_required", "_direct_child_results",
        "_drain_incoming_messages", "_enforce_swarm_actions", "_extract_plain_text_from_content",
        "_finalize_task_services", "_force_plan_disclosure", "_forced_orphan_note",
        "_handle_forced_finalization", "_handle_text_response", "_live_delivery_candidate",
        "_load_direct_child_results", "_maybe_enforce_child_absorption_gate",
        "_maybe_inject_finalization_nudges", "_merge_finalization_trace",
        "_parse_delivery_control_object", "_project_child_result_dispositions",
        "_publish_delivery_candidate", "_replace_delivery_candidate",
        "_run_task_acceptance_review_once", "_service_finalization_evidence",
        "_supersede_delivery_acceptance_binding", "_supersede_task_acceptance_for_evidence_change",
        "_supersede_task_acceptance_for_owner_followup"
    })),
    "ouroboros/loop_forced_finalization.py": ("ouroboros/loop.py", "_loop", frozenset({
        "DeliveryCandidate", "_LoopExitContext", "_append_or_merge_user_message",
        "_child_disposition_state", "_compose_delivery_suffix", "_current_delivery_candidate",
        "_degrade_retained_delivery_candidate", "_delivery_evidence_state",
        "_delivery_replace_required", "_direct_child_results", "_drain_incoming_messages",
        "_end_task_acceptance_fence", "_finalize_forced_services", "_finalize_task_services",
        "_force_plan_decision", "_force_plan_disclosure", "_force_plan_reminder",
        "_forced_delegation_note", "_forced_fallback_result", "_forced_final_answer",
        "_forced_orphan_note", "_forced_swarm_router_result", "_forced_unaccepted_binding",
        "_live_delivery_candidate", "_load_direct_child_results", "_merge_finalization_trace",
        "_parse_delivery_control_object", "_project_child_result_dispositions",
        "_publish_delivery_candidate", "_record_forced_acceptance_bypass",
        "_record_forced_finalization", "_replace_delivery_candidate",
        "_run_task_acceptance_review_once", "_service_finalization_evidence",
        "_set_acceptance_decision", "_supersede_task_acceptance_for_owner_followup",
        "_swarm_handoff_attempt", "call_llm_with_retry"
    })),
    # The ouroboros/tools/review_*.py and ouroboros/review_*.py rows are the
    # L-C review-stack split: the same owner-approved mechanical exception,
    # with `_rev()` / `_car()` as the call-time handles (delta D37). Their
    # declared sets are the parent bindings tests rebind (monkeypatch or plain
    # attribute assignment) plus cross-leaf member reads, so patching the
    # parent keeps intercepting the moved bodies.
    "ouroboros/tools/review_multi_model.py": ("ouroboros/tools/review.py", "_rev", frozenset({
        "LLMClient", "load_governance_doc", "review_drive_root", "slot_id_for_row",
    })),
    "ouroboros/tools/review_advisory_prompt.py": ("ouroboros/tools/claude_advisory_review.py", "_car", frozenset({
        "_get_changed_file_list", "_get_staged_diff",
    })),
    "ouroboros/tools/review_advisory_run.py": ("ouroboros/tools/claude_advisory_review.py", "_car", frozenset({
        "_ADVISORY_PROMPT_MAX_CHARS", "_build_advisory_prompt", "_get_changed_file_list",
        "_get_staged_diff", "_syntax_preflight_staged_py_files",
        "advisory_gate_unavailability_reason", "build_advisory_changed_context",
        "emit_review_usage",
    })),
    "supervisor/queue_snapshot.py": ("supervisor/queue.py", "_queue", frozenset({
        "ACCEPTANCE_FENCES", "DRIVE_ROOT", "PENDING", "RUNNING", "_queue_lock", "append_jsonl", "atomic_write_text", "enqueue_task",
    })),
    "supervisor/queue_timeouts.py": ("supervisor/queue.py", "_queue", frozenset({
        "DRIVE_ROOT", "FINALIZATION_GRACE_SEC", "HEARTBEAT_STALE_SEC", "PENDING", "QUEUE_MAX_RETRIES", "RUNNING", "_ensure_reaper_started", "_queue_lock", "_reap_queue", "_request_finalization_grace", "get_per_call_timeout_ceiling_sec", "get_task_abs_ceiling_sec", "get_task_idle_timeout_sec", "load_state", "persist_queue_snapshot",
    })),
    "supervisor/queue_schedules.py": ("supervisor/queue.py", "_queue", frozenset({
        "DRIVE_ROOT", "PENDING", "RUNNING", "SCHEDULED_TASKS_FILE", "_queue_lock", "enqueue_task", "load_state", "persist_queue_snapshot",
    })),
    "supervisor/queue_evolution.py": ("supervisor/queue.py", "_queue", frozenset({
        "DRIVE_ROOT", "OBJECTIVE_REPEAT_CAP", "PENDING", "RUNNING", "_read_evolution_campaign", "append_jsonl", "begin_evolution_transaction", "budget_remaining", "enqueue_task", "load_state", "notify_owner_cycle_outcome", "persist_queue_snapshot", "queue_has_task_type", "send_with_budget",
    })),
    "supervisor/worker_promotion.py": ("supervisor/workers.py", "_pool", frozenset({
        "DRIVE_ROOT", "PENDING", "REPO_DIR", "RUNNING",
    })),
    "supervisor/worker_chat_lane.py": ("supervisor/workers.py", "_pool", frozenset({
        "DRIVE_ROOT", "REPO_DIR", "_chat_agent_lock", "_ephemeral_chat_lock", "_get_chat_agent", "_origin_from_mapping", "_repo_writer_turn_allowed", "_report_binding_failure", "get_event_q", "load_state", "send_with_budget",
    })),
    "supervisor/worker_health.py": ("supervisor/workers.py", "_pool", frozenset({
        "CRASH_TS", "DRIVE_ROOT", "QUEUE_MAX_RETRIES", "RUNNING", "WORKERS", "_LAST_SPAWN_TIME", "_SPAWN_GRACE_SEC", "get_event_q", "kill_workers", "load_state", "reconstruct_task_cost", "respawn_worker", "send_with_budget",
    })),
    "supervisor/worker_pool_lifecycle.py": ("supervisor/workers.py", "_pool", frozenset({
        "DRIVE_ROOT", "REPO_DIR", "WORKERS", "Worker", "_WORKER_PIDS_FILENAME", "_get_ctx", "get_event_q", "kill_workers", "load_state", "reconstruct_task_cost", "send_with_budget",
    })),
    "supervisor/worker_assignment.py": ("supervisor/workers.py", "_pool", frozenset({
        "DRIVE_ROOT", "PENDING", "RUNNING", "WORKERS", "_drop_cancelled_pending", "_emit_task_done_terminal", "load_state", "reconstruct_task_cost", "repo_writer_task_allowed", "send_with_budget",
    })),
    "supervisor/update_merge_plan.py": ("supervisor/update_merge.py", "_um", frozenset({
        "_merge_head_sha", "managed_update_constitution_present",
    })),
    "supervisor/git_ops_remotes.py": ("supervisor/git_ops.py", "_go", frozenset({
        "BRANCH_DEV", "REPO_DIR", "_configure_credential_helper", "_has_remote",
        "configure_remote", "ensure_official_update_remote", "git_capture",
    })),
    "supervisor/git_ops_updates.py": ("supervisor/git_ops.py", "_go", frozenset({
        "BRANCH_DEV", "DRIVE_ROOT", "OFFICIAL_UPDATE_REMOTE_URL", "_collect_repo_sync_state",
        "_compute_ref_ahead_count", "_create_rescue_snapshot", "_git_network_bounded",
        "_has_remote", "_list_remotes", "_managed_remote_name", "_managed_update_target",
        "_read_managed_repo_meta", "_rescue_untracked_incomplete", "_resolve_managed_update_target",
        "_write_update_intent", "append_jsonl", "ensure_official_update_remote", "git_capture",
        "git_fetch_bounded", "load_state", "managed_branch_defaults", "preserve_local_ref_branch",
        "utc_now_iso",
    })),
    "supervisor/git_ops_reset.py": ("supervisor/git_ops.py", "_go", frozenset({
        "BRANCH_DEV", "BRANCH_STABLE", "DRIVE_ROOT", "REPO_DIR",
        "_admission_gate_for_unsynced_tree", "_clear_bootstrap_pin_marker", "_clear_update_intent",
        "_collect_repo_sync_state", "_compute_ref_ahead_count", "_create_rescue_snapshot",
        "_git_dir", "_guard_live_repo_destructive_git", "_has_remote", "_maybe_repair_git_index",
        "_pin_to_bundle_sha_on_bootstrap", "_preserve_branch_for_official_reset",
        "_read_managed_repo_meta", "_read_update_intent", "_ref_points_at_ref",
        "_rescue_untracked_incomplete", "_run_git_resilient", "_update_source", "append_jsonl",
        "checkout_and_reset", "git_capture", "import_test", "load_state",
        "preserve_local_ref_branch", "rescue_git_capture", "save_state", "sync_runtime_dependencies",
        "utc_now_iso",
    })),
    "supervisor/git_ops_rescue.py": ("supervisor/git_ops.py", "_go", frozenset({
        "BRANCH_DEV", "DRIVE_ROOT", "REPO_DIR", "_atomic_write_bytes", "_collect_repo_sync_state",
        "_copy_untracked_for_rescue", "_create_rescue_snapshot", "_git_dir",
        "_link_rescue_to_evolution_transaction", "_list_remotes", "_managed_remote_branch_for",
        "_managed_remote_name", "_read_managed_repo_meta", "_run_git_process_bounded",
        "append_jsonl", "atomic_write_text", "rescue_before_destructive_rollback",
        "rescue_git_capture", "utc_now_iso",
    })),
    "ouroboros/agent_dispatch.py": ("ouroboros/agent.py", "_agent", frozenset({
        "write_task_result",
    })),
    "ouroboros/usage_legacy_import.py": ("ouroboros/usage_accounting.py", "_usage", frozenset({
        "_legacy_snapshot", "_locked", "_read_records_locked",
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
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
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
            parent_defs.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            parent_defs.add(node.target.id)
    own = _module_bindings(leaf_tree)
    direct = {
        node.id for node in ast.walk(leaf_tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        and node.id in parent_defs and node.id not in own
    }
    assert direct == set(), f"{leaf} reads {sorted(direct)} directly instead of through the handle"


def test_the_pool_still_owns_its_state_and_worker_main_stays_picklable() -> None:
    """The split moved responsibilities, not state — and not the one function the
    spawn platforms have to re-import by name."""
    import pickle

    from supervisor import workers

    for name in ("REPO_DIR", "DRIVE_ROOT", "MAX_WORKERS", "WORKERS", "PENDING", "RUNNING",
                 "CRASH_TS", "QUEUE_SEQ_COUNTER_REF", "_CTX", "_LAST_SPAWN_TIME"):
        assert hasattr(workers, name), name
    assert workers.worker_main.__module__ == "supervisor.worker_process"
    assert pickle.loads(pickle.dumps(workers.worker_main)) is workers.worker_main
    for leaf in LEAVES:
        if not leaf.startswith("supervisor/worker"):
            continue
        module = __import__(leaf[:-3].replace("/", "."), fromlist=["_"])
        for name in ("WORKERS", "PENDING", "RUNNING", "DRIVE_ROOT"):
            assert not hasattr(module, name), f"{leaf} kept its own {name}"


def test_the_decorator_primitive_is_imported_not_handled() -> None:
    """A decorator runs at IMPORT time, so the one name a call-time handle cannot
    carry is the lifecycle serializer; it lives with its heaviest user and the pool
    imports it back."""
    from supervisor import worker_pool_lifecycle, workers

    assert workers._serialized_worker_lifecycle is worker_pool_lifecycle._serialized_worker_lifecycle
    assert "_serialized_worker_lifecycle" not in LEAVES["supervisor/worker_pool_lifecycle.py"][2]


def test_the_parent_still_owns_the_state_and_the_lock_stays_reentrant() -> None:
    """The split moved responsibilities, not state: one binding, one lock."""
    from supervisor import queue

    for name in ("PENDING", "RUNNING", "QUEUE_SEQ_COUNTER_REF", "ACCEPTANCE_FENCES",
                 "ADMISSION_RESERVATIONS", "DRIVE_ROOT", "FINALIZATION_GRACE_SEC"):
        assert hasattr(queue, name), name
    assert hasattr(queue._queue_lock, "_is_owned")
    for leaf in LEAVES:
        module = __import__(leaf[:-3].replace("/", "."), fromlist=["_"])
        assert not hasattr(module, "PENDING"), f"{leaf} kept its own PENDING"
        assert not hasattr(module, "RUNNING"), f"{leaf} kept its own RUNNING"


def test_the_queue_facade_still_exports_everything_that_moved() -> None:
    """`supervisor.queue` is the single public import surface; the split must not
    make a caller learn which leaf a name landed in."""
    from supervisor import queue

    for name in ("persist_queue_snapshot", "restore_pending_from_snapshot", "parse_iso_to_ts",
                 "enforce_task_timeouts", "check_scheduled_tasks", "list_scheduled_tasks",
                 "upsert_scheduled_task", "remove_scheduled_task", "sync_skill_schedules",
                 "resync_skill_schedules", "queue_deep_self_review_task",
                 "get_evolution_status_snapshot", "enqueue_evolution_task_if_needed"):
        assert hasattr(queue, name), name


def test_every_queue_leaf_is_a_hot_code_path_like_its_parent() -> None:
    """Managed-update conflict labelling names ``supervisor/queue.py``; the split
    must not silently downgrade the label for code that merely moved (the events
    split pins the same parity). ``workers.py`` is unlabeled at base, so its
    leaves inherit that — parity, not blanket labelling."""
    from supervisor.update_merge_policy import HOT_CODE_PATHS

    assert "supervisor/queue.py" in HOT_CODE_PATHS
    for leaf in ("supervisor/queue_evolution.py", "supervisor/queue_schedules.py",
                 "supervisor/queue_snapshot.py", "supervisor/queue_timeouts.py"):
        assert leaf in HOT_CODE_PATHS, leaf

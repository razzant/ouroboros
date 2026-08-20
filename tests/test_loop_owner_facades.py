"""Facade-identity contract for the v7 L-B loop.py leaf owners.

Every member the L-B split moved out of ``ouroboros/loop.py`` got a loop.py
re-export under its historical name, so existing callers and monkeypatching
tests kept working unchanged while the split landed. The private half of that
facade was declared TEMPORARY (spec 4.3-15), and the L3 package spent it: each
name was classified by who actually reads it, and the ones only its own leaf
reads left ``ouroboros.loop`` for good.

Two lists carry that outcome and both are load-bearing.

``LOOP_LEAF_OWNERS`` is what loop.py still re-exports. A name survives here for
one of exactly two reasons, and no other: ``run_llm_loop``'s own body calls it,
or a SIBLING leaf reads it through the D33 call-time handle (``_loop().X``), for
which loop.py is the family's rendezvous binding — retiring those would not
remove a seam, it would replace one shared seam with a mesh of sibling handles.
This test pins the facade identity for them: the loop binding IS the leaf's
object.

``RETIRED_FROM_LOOP`` is what left. Those names are read only by the leaf that
owns them, so the leaf reads them as ordinary module-locals (still late-bound,
so patching the LEAF intercepts) and no ``ouroboros.loop`` binding remains. That
absence is the contract — a well-meaning re-export added back would silently
resurrect a second address for one object — so it is asserted, not assumed.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

# leaf module -> every member the leaf owns (loop.py re-exports each name).
LOOP_LEAF_OWNERS: dict[str, tuple[str, ...]] = {
    "loop_messages": (
        "_emit_checkpoint_event _extract_plain_text_from_content _append_or_merge_user_message "
        "_owner_marked_content _record_owner_directive _initialize_owner_directives _last_assistant_text "
        "_emit_round_progress"
    ),
    "loop_acceptance": (
        "_task_acceptance_eligible _begin_task_acceptance_fence _end_task_acceptance_fence "
        "_supersede_delivery_acceptance_binding _supersede_task_acceptance_for_owner_followup "
        "_task_acceptance_owner_generation_changed _supersede_task_acceptance_for_evidence_change "
        "_task_acceptance_subtree_snapshot _mark_root_acceptance_checkpoint _latch_final_answer_marker "
        "_server_web_allowed_by_task _set_acceptance_decision _collect_acceptance_obligations "
        "_open_acceptance_obligations _dispose_obligations_on_clean_pass _format_obligations_clause "
        "_record_forced_acceptance_bypass"
    ),
    "loop_acceptance_review": (
        "_run_task_acceptance_review_once"
    ),
    "loop_round_limits": (
        "_CompactionRoundContext _task_deadline_epoch _drain_incoming_messages _context_reclaim_passes "
        "_context_reclaim_materializations _context_overflow_retries _run_round_compaction _RoundLimitContext "
        "_account_compaction_usage _handle_round_limit _handle_forced_finalization "
        "_handle_provider_unavailable _maybe_early_finalize _finalize_limit_ctx"
    ),
    "loop_nudges": (
        "_force_plan_decision _force_plan_reminder _force_plan_disclosure _note_nanny_delegate_activity "
        "_inject_round_checkpoints _forced_delegation_note _maybe_inject_finalization_nudges"
    ),
    "loop_model_call": (
        "_run_cross_model_fallback_chain _rebind_context_fit_plan _RoundModelCallContext _call_round_model"
    ),
    "loop_budget": (
        "_check_budget_limits _resolve_task_cost_ceiling _TREE_ACCOUNTING_MAX_STALE_SEC _loop_tree_accounting "
        "_soft_land_exhausted_ceiling _service_finalization_evidence _LoopExitContext _handle_budget_exceeded "
        "_cleanup_loop_resources _finalize_task_services _prepare_post_tool_budget_context"
    ),
    "loop_delivery": (
        "DeliveryCandidate _swarm_handoff_attempt _delivery_evidence_state _publish_delivery_candidate "
        "_replace_delivery_candidate _forced_unaccepted_binding _live_delivery_candidate "
        "_current_delivery_candidate _degrade_retained_delivery_candidate _merge_finalization_trace "
        "_delivery_replace_required _arm_delivery_control _parse_delivery_control_object "
        "_compose_delivery_suffix _no_tool_final_answer"
    ),
    "loop_forced_finalization": (
        "_load_direct_child_results _direct_child_results _child_disposition_state "
        "_project_child_result_dispositions _record_forced_finalization _forced_orphan_note "
        "_maybe_enforce_child_absorption_gate _enforce_swarm_actions _finalize_forced_services "
        "_forced_fallback_result _forced_swarm_router_result _forced_final_answer"
    ),
}


# leaf module -> every member whose TEMPORARY loop.py re-export the L3 package
# retired (spec 4.3-15). ouroboros.loop carries none of these bindings; the one
# member still read outside its owning leaf is _append_or_merge_user_content,
# whose consumers import the owner directly (lazy function-local imports in
# tools/browser.py and tools/vision.py; a frozen module-level import in
# loop_round_limits.py, disclosed at its import site).
RETIRED_FROM_LOOP: dict[str, tuple[str, ...]] = {
    "loop_forced_finalization": (
        "_claimed_child_dispositions _undispositioned_children _run_forced_children_acceptance "
        "_drain_forced_owner_directives _call_forced_model_once _publish_model_forced_candidate "
        "_publish_stale_forced_candidate _resolve_forced_delivery_control"
    ),
    "loop_delivery": (
        "_compute_subagent_handoff _unaccepted_delivery_binding _delivery_acceptance_binding "
        "_ensure_explicit_acceptance_binding _delivery_control_prompt _delivery_keep_allowed "
        "_hold_delivery_for_skill_action _resolve_delivery_control"
    ),
    "loop_budget": (
        "_service_identity_projection"
    ),
    "loop_model_call": (
        "_adopt_fallback_route _snapshot_context_fit_usage _restore_context_fit_usage _context_fit_round_id "
        "_main_context_profile _remember_main_fit _measure_round_main_fit _physical_context_for_fit "
        "_dispatch_round_model _run_main_reclaim _measure_after_reclaim _reproject_actual_overflow_low "
        "_failed_capture_is_comparable _strict_context_shrink_predicate _emit_overflow_retry_skipped"
    ),
    "loop_nudges": (
        "_skill_names_touched_by_trace _skill_finalization_message _build_recent_tool_trace "
        "_maybe_inject_self_check _maybe_inject_time_budget_milestone _maybe_inject_cost_budget_milestone "
        "_DELEGATE_ACTIVITY_TOOLS _nanny_metered_since_delegate_activity _nanny_reminder_due "
        "_nanny_burn_phrase _maybe_inject_nanny_economics_reminder _nanny_finalization_message "
        "_answer_protocol_active _contract_expected_output"
    ),
    "loop_round_limits": (
        "_provider_failure_hint _provider_recovery_hint _mark_owner_stop_control_drained "
        "_owner_stop_window_elapsed _handle_owner_stop_finalization _maybe_deadline_local_finalize"
    ),
    "loop_acceptance_review": (
        "_ACCEPTANCE_REVIEW_CHECKLIST _TaskAcceptanceContext _acceptance_dialogue_quorum "
        "_attach_dialogue_to_host_run _mark_agent_acceptance_runs_advisory _latest_agent_acceptance_evidence "
        "_build_host_acceptance_evidence _execute_task_acceptance_panel _record_host_acceptance_run "
        "_set_applied_host_acceptance_impact _apply_task_acceptance_result _record_acceptance_infra_failure "
        "_prior_acceptance_run _direct_context_fence_state"
    ),
    "loop_acceptance": (
        "ACCEPTANCE_REASON_UNSPECIFIED ACCEPTANCE_DECISION_REASONS _reopen_obligation_row"
    ),
    "loop_messages": (
        "_evict_stale_image_blocks _append_or_merge_user_content _visible_round_text"
    ),
}


def test_loop_owner_facades_preserve_identity():
    import ouroboros.loop as loop

    for leaf, names in LOOP_LEAF_OWNERS.items():
        module = importlib.import_module(f"ouroboros.{leaf}")
        for name in names.split():
            assert getattr(loop, name) is getattr(module, name), f"{leaf}.{name}"


def _loop_body_reads() -> set[str]:
    """Names ``ouroboros/loop.py``'s own code reads, ignoring the re-export block."""
    source = (REPO / "ouroboros" / "loop.py").read_text(encoding="utf-8")
    return {
        node.id for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _sibling_handle_readers() -> dict[str, set[str]]:
    """name -> the loop leaves that read it as ``_loop().name``."""
    readers: dict[str, set[str]] = {}
    for path in sorted((REPO / "ouroboros").glob("loop_*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "_loop"):
                readers.setdefault(node.attr, set()).add(path.stem)
    return readers


def test_every_surviving_private_re_export_still_has_a_reason_to_exist():
    """The retirement's real product is this invariant, not the line count.

    After L3 a private name may sit on ``ouroboros.loop`` for exactly two
    reasons: ``run_llm_loop``'s own body calls it, or a leaf OTHER than its
    owner reads it as ``_loop().name`` — the family's rendezvous binding, whose
    alternative is a mesh of sibling handles. A name that satisfies neither is
    a re-export nobody needs, which is precisely what L3 went looking for; it
    should be retired rather than left here for a future reader to puzzle over.
    """
    body = _loop_body_reads()
    readers = _sibling_handle_readers()
    unjustified = [
        f"{leaf}.{name}"
        for leaf, names in LOOP_LEAF_OWNERS.items()
        for name in names.split()
        if name not in body and not (readers.get(name, set()) - {leaf})
    ]
    assert unjustified == [], (
        "these re-exports have neither a run_llm_loop caller nor a sibling "
        f"handle reader and should be retired: {unjustified}"
    )


def test_the_retired_private_names_own_their_leaf_and_left_the_loop_surface():
    """The L3 retirement, stated as a property rather than a diff: each retired
    name is a real member of its leaf, and ``ouroboros.loop`` no longer binds it
    at all. The second half is the one worth a test — a re-export added back
    "for convenience" would restore a second address for the same object and
    quietly re-open the patch-the-wrong-module trap the retirement closed."""
    import ouroboros.loop as loop

    for leaf, names in RETIRED_FROM_LOOP.items():
        module = importlib.import_module(f"ouroboros.{leaf}")
        for name in names.split():
            assert hasattr(module, name), f"{leaf}.{name} is not owned by its leaf"
            assert not hasattr(loop, name), f"ouroboros.loop still binds {name}"
            assert name not in LOOP_LEAF_OWNERS.get(leaf, "").split(), f"{name} is in both lists"


def test_loop_leaves_keep_the_hot_code_label():
    """Managed-update conflict labelling names ``ouroboros/loop.py``; the split
    must not silently downgrade the label for code that merely moved."""
    from supervisor.update_merge_policy import HOT_CODE_PATHS

    assert "ouroboros/loop.py" in HOT_CODE_PATHS
    for leaf in LOOP_LEAF_OWNERS:
        assert f"ouroboros/{leaf}.py" in HOT_CODE_PATHS, leaf

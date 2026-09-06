"""Facade-identity contract for the v7 L-B loop.py leaf owners (D01 lane).

Every member the L-B split moved out of ``ouroboros/loop.py`` keeps a loop.py
re-export under its historical name, so existing callers and monkeypatching
tests keep working unchanged: the loop binding IS the leaf's object, and the
sibling leaves' D33 call-time handle reads (``_loop().X``) resolve through this
module as the family rendezvous.

v7next transplant note: the reference (ouroboros_v7_wip @ 9f691656) later spent
the private half of this facade (its L3 package, RETIRED_FROM_LOOP) by
re-homing every loop-private test import to its leaf owner. That trimming is a
consumer-rebind wave, not part of the byte-preserving relocation, and does NOT
ride with the D01 lane: on this tree the tip consumer surface still addresses
every moved name at ``ouroboros.loop``, so the FULL re-export surface is the
contract here (see docs/v7next/LEDGER_CORRECTIONS.md, D01 lane).
"""

from __future__ import annotations

import importlib

# leaf module -> every member the leaf owns (loop.py re-exports each name).
LOOP_LEAF_OWNERS: dict[str, str] = {
    "loop_messages": (
        "_emit_checkpoint_event _extract_plain_text_from_content "
        "_append_or_merge_user_message _evict_stale_image_blocks "
        "_append_or_merge_user_content _owner_marked_content _record_owner_directive "
        "_initialize_owner_directives _visible_round_text _emit_round_progress"
    ),
    "loop_acceptance": (
        "_task_acceptance_eligible _begin_task_acceptance_fence "
        "_end_task_acceptance_fence _supersede_delivery_acceptance_binding "
        "_supersede_task_acceptance_for_owner_followup "
        "_task_acceptance_owner_generation_changed "
        "_supersede_task_acceptance_for_evidence_change "
        "_task_acceptance_subtree_snapshot _mark_root_acceptance_checkpoint "
        "_latch_final_answer_marker _server_web_allowed_by_task "
        "ACCEPTANCE_REASON_UNSPECIFIED ACCEPTANCE_DECISION_REASONS "
        "_set_acceptance_decision _collect_acceptance_obligations _reopen_obligation_row "
        "_open_acceptance_obligations _dispose_obligations_on_clean_pass "
        "_format_obligations_clause _record_forced_acceptance_bypass"
    ),
    "loop_acceptance_review": (
        "_ACCEPTANCE_REVIEW_CHECKLIST _TaskAcceptanceContext _acceptance_dialogue_quorum "
        "_attach_dialogue_to_host_run _mark_agent_acceptance_runs_advisory "
        "_latest_agent_acceptance_evidence _build_host_acceptance_evidence "
        "_execute_task_acceptance_panel _record_host_acceptance_run "
        "_set_applied_host_acceptance_impact _apply_task_acceptance_result "
        "_record_acceptance_infra_failure _prior_acceptance_run "
        "_direct_context_fence_state _run_task_acceptance_review_once"
    ),
    "loop_round_limits": (
        "_CompactionRoundContext _drain_incoming_messages _context_reclaim_passes "
        "_context_reclaim_materializations _context_overflow_retries "
        "_run_round_compaction _RoundLimitContext _account_compaction_usage "
        "_handle_round_limit _handle_forced_finalization _handle_owner_stop_finalization "
        "_handle_provider_unavailable _maybe_deadline_local_finalize "
        "_maybe_early_finalize _finalize_limit_ctx"
    ),
    "loop_nudges": (
        "_skill_names_touched_by_trace _skill_finalization_message _force_plan_decision "
        "_force_plan_reminder _force_plan_disclosure _build_recent_tool_trace "
        "_maybe_inject_self_check _maybe_inject_time_budget_milestone "
        "_maybe_inject_cost_budget_milestone _maybe_inject_nanny_economics_reminder "
        "_inject_round_checkpoints _forced_delegation_note _nanny_finalization_message "
        "_maybe_inject_finalization_nudges _answer_protocol_active "
        "_contract_expected_output"
    ),
    "loop_model_call": (
        "_adopt_fallback_route _snapshot_context_fit_usage _restore_context_fit_usage "
        "_run_cross_model_fallback_chain _rebind_context_fit_plan _RoundModelCallContext "
        "_context_fit_round_id _main_context_profile _remember_main_fit "
        "_measure_round_main_fit _physical_context_for_fit _dispatch_round_model "
        "_run_main_reclaim _measure_after_reclaim _reproject_actual_overflow_low "
        "_failed_capture_is_comparable _strict_context_shrink_predicate "
        "_emit_overflow_retry_skipped _call_round_model"
    ),
    "loop_budget": (
        "_check_budget_limits _resolve_task_cost_ceiling _TREE_ACCOUNTING_MAX_STALE_SEC "
        "_loop_tree_accounting _soft_land_exhausted_ceiling "
        "_service_finalization_evidence _LoopExitContext _handle_budget_exceeded "
        "_cleanup_loop_resources _service_identity_projection _finalize_task_services "
        "_prepare_post_tool_budget_context"
    ),
    "loop_delivery": (
        "DeliveryCandidate _swarm_handoff_attempt _compute_subagent_handoff "
        "_delivery_evidence_state _unaccepted_delivery_binding "
        "_delivery_acceptance_binding _publish_delivery_candidate "
        "_replace_delivery_candidate _ensure_explicit_acceptance_binding "
        "_forced_unaccepted_binding _live_delivery_candidate _current_delivery_candidate "
        "_degrade_retained_delivery_candidate _merge_finalization_trace "
        "_delivery_control_prompt _delivery_replace_required _delivery_keep_allowed "
        "_arm_delivery_control _hold_delivery_for_skill_action "
        "_parse_delivery_control_object _resolve_delivery_control "
        "_compose_delivery_suffix _no_tool_final_answer"
    ),
    "loop_forced_finalization": (
        "_load_direct_child_results _direct_child_results _child_disposition_state "
        "_project_child_result_dispositions _record_forced_finalization "
        "_forced_orphan_note _claimed_child_dispositions _undispositioned_children "
        "_maybe_enforce_child_absorption_gate _run_forced_children_acceptance "
        "_enforce_swarm_actions _finalize_forced_services _drain_forced_owner_directives "
        "_call_forced_model_once _publish_model_forced_candidate "
        "_publish_stale_forced_candidate _forced_fallback_result "
        "_forced_swarm_router_result _resolve_forced_delivery_control "
        "_forced_final_answer"
    ),
}


def test_loop_owner_facades_preserve_identity():
    import ouroboros.loop as loop

    for leaf, names in LOOP_LEAF_OWNERS.items():
        module = importlib.import_module(f"ouroboros.{leaf}")
        for name in names.split():
            assert getattr(loop, name) is getattr(module, name), f"{leaf}.{name}"


def test_loop_leaves_keep_the_hot_code_label():
    """Managed-update conflict labelling names ``ouroboros/loop.py``; the split
    must not silently downgrade the label for code that merely moved."""
    from supervisor.update_merge_policy import HOT_CODE_PATHS

    assert "ouroboros/loop.py" in HOT_CODE_PATHS
    for leaf in LOOP_LEAF_OWNERS:
        assert f"ouroboros/{leaf}.py" in HOT_CODE_PATHS, leaf

"""Facade-identity contract for the v7 L-C2 leaf owners.

Every member the L-C2 split moved out of ``ouroboros/agent.py``,
``ouroboros/agent_task_pipeline.py`` and ``ouroboros/usage_accounting.py``
keeps a parent re-export under its historical name, so existing callers and
monkeypatching tests keep working unchanged — the parent binding IS the leaf's
object, the same way the queue, loop and update_merge splits pin their leaves.
The hot-code parity clause pins the update_merge direction of the rule: none of
these parents is a HOT_CODE_PATHS member, so a leaf that merely moved code out
of one must not silently acquire the label either.
"""

from __future__ import annotations

import importlib

# leaf module -> (parent module, every member the leaf owns; the parent
# re-exports each name).
LC2_LEAF_OWNERS: dict[str, tuple[str, str]] = {
    "ouroboros.agent_dispatch": (
        "ouroboros.agent",
        "dispatch_executor_note executor_blocked_outcome _record_executor_resolution "
        "_blocked_executor_terminal _persist_early_origin_stub _budget_exhausted_message "
        "_budget_resume_policy _queued_budget_exhausted_message _physical_calls_after_budget_rail "
        "_initial_effort_for resolve_dispatch_axes _DELEGATE_VERBS preflight_delegate_visibility "
        "reset_nanny_economics_marks emit_dispatch_resolution capability_delta_prompt_block"
    ),
    "ouroboros.post_task_synthesis": (
        "ouroboros.agent_task_pipeline",
        "build_trace_summary _update_improvement_backlog _apply_reflection_memory_actions "
        "_child_task_evidence _pre_synthesis_usage_snapshot _compact_review_projection "
        "_TASK_SUMMARY_PROMPT _summary_row_cost_fields _run_task_summary _run_chat_consolidation "
        "_run_scratchpad_consolidation _run_reflection"
    ),
    "ouroboros.usage_legacy_import": (
        "ouroboros.usage_accounting",
        "IMPORT_REL _legacy_snapshot ensure_legacy_imported _completed_import_watermark "
        "_ensure_legacy_imported_locked"
    ),
}


def test_lc2_owner_facades_preserve_identity():
    for leaf, (parent, names) in LC2_LEAF_OWNERS.items():
        parent_module = importlib.import_module(parent)
        leaf_module = importlib.import_module(leaf)
        for name in names.split():
            assert getattr(parent_module, name) is getattr(leaf_module, name), f"{leaf}.{name}"


def test_lc2_leaves_keep_hot_code_label_parity():
    """Managed-update conflict labelling names none of the L-C2 parents; the
    split must not silently upgrade or downgrade the label for code that merely
    moved — parent and leaves carry the SAME membership."""
    from supervisor.update_merge_policy import HOT_CODE_PATHS

    for leaf, (parent, _names) in LC2_LEAF_OWNERS.items():
        parent_path = parent.replace(".", "/") + ".py"
        leaf_path = leaf.replace(".", "/") + ".py"
        assert (leaf_path in HOT_CODE_PATHS) == (parent_path in HOT_CODE_PATHS), leaf

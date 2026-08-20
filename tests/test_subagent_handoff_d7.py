"""D#7 / P5: the pre-finalization subagent handoff reminder is suppressed ONLY by a
typed exact-hash task-tree decision, explicit cancellation, or absorption — never by
parsing the final PROSE for status words (the removed _final_text_acknowledges_*
keyword gate). A nonterminal, undecided child surfaces the reminder regardless of what
the final text says.
"""

from __future__ import annotations

from types import SimpleNamespace


def _tools(tmp_path):
    ctx = SimpleNamespace(
        task_metadata={"budget_drive_root": str(tmp_path), "root_task_id": "root"},
        budget_drive_root=str(tmp_path),
        drive_root=str(tmp_path),
        task_id="root",
        role="orchestrator",
        _subagent_handoff_signature="",
    )
    return SimpleNamespace(_ctx=ctx)


def _write_child(tmp_path, child_id, status="running", **fields):
    from ouroboros.task_results import write_task_result

    result = fields.pop("result", "partial")
    write_task_result(
        tmp_path, child_id, status,
        parent_task_id="root", root_task_id="root", delegation_role="subagent",
        result=result, **fields,
    )


def test_prose_does_not_suppress_handoff(tmp_path):
    """Even when the final text 'acknowledges' the child in prose, an undecided
    nonterminal child still surfaces the handoff reminder (P5: no keyword gate)."""
    from ouroboros.loop_delivery import _compute_subagent_handoff

    _write_child(tmp_path, "childA", status="running")
    prose = "All set. I am leaving childA running / pending; not complete yet."
    out = _compute_subagent_handoff(_tools(tmp_path), tmp_path, "root", prose)
    assert out, "handoff reminder must fire despite prose acknowledgement"
    assert "childA" in out


def test_structured_discard_suppresses_handoff(tmp_path):
    """A hash-bound discard is excluded while that exact result is unchanged."""
    from ouroboros.loop_delivery import _compute_subagent_handoff
    from ouroboros.tools.join_ledger import _discard_child_result

    _write_child(tmp_path, "childA", status="running")
    tools = _tools(tmp_path)
    assert _discard_child_result(tools._ctx, "childA", "not needed").startswith("Discarded")
    out = _compute_subagent_handoff(tools, tmp_path, "root", "done")
    assert out == "", f"discarded child must not surface a reminder, got: {out!r}"


def test_legacy_task_result_discard_fields_are_not_authority(tmp_path):
    from ouroboros.loop_delivery import _compute_subagent_handoff

    _write_child(tmp_path, "legacy", parent_decision="discarded")
    out = _compute_subagent_handoff(_tools(tmp_path), tmp_path, "root", "done")
    assert "legacy" in out

    _write_child(
        tmp_path,
        "stale-exact",
        parent_decision="discarded",
        child_result_disposition="irrelevant",
        child_result_disposition_sha256="0" * 64,
    )
    out = _compute_subagent_handoff(_tools(tmp_path), tmp_path, "root", "done")
    assert "stale-exact" in out

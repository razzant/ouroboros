"""R9: ``max_children`` and ``may_fan_out`` count a parent's admitted direct children over its
whole lifetime, finished ones included — the schema text says so, and the real count and the
real admission agree. A requested or duplicate-refused child, or one that failed before
admission, is not counted; a refusal past the cap is typed, never soft."""

from __future__ import annotations

import pytest

from tests._store_shared import seed_result


def _child(root, task_id, status, **extra):
    seed_result(root, task_id, {"task_id": task_id, "parent_task_id": "parent", "delegation_role": "subagent",
                                "status": status, **extra})


@pytest.mark.parametrize("status", ["completed", "failed_admitted", "cancelled"])
def test_a_finished_child_still_spends_the_parents_cap(tmp_path, status):
    from ouroboros.tools.control_delegation import check_delegation_admission, durable_direct_child_count

    seed_result(tmp_path, "parent", {"task_id": "parent", "status": "running"})
    if status == "failed_admitted":
        _child(tmp_path, "done", "failed", delegation_admission={"status": "accepted"})
    else:
        _child(tmp_path, "done", status)
    _child(tmp_path, "asked", "requested")
    _child(tmp_path, "dup", "rejected_duplicate")
    _child(tmp_path, "fell", "failed")
    count = durable_direct_child_count(tmp_path, "parent")
    assert count == 1  # only the one admitted child, though it is no longer active
    capped = check_delegation_admission({"max_children": 1}, direct_child_count=count)
    assert (capped.ok, capped.reason_code) == (False, "delegation_rights_max_children")
    single = check_delegation_admission({"may_fan_out": False}, direct_child_count=count)
    assert (single.ok, single.reason_code) == (False, "delegation_rights_may_fan_out")
    assert check_delegation_admission({"max_children": 2}, direct_child_count=count).ok


def test_the_schema_names_the_lifetime_count_not_a_soft_active_cap():
    from ouroboros.tools.control_subagent_spec import schedule_subagent_properties

    properties = schedule_subagent_properties()
    for name in ("max_children", "may_fan_out"):
        text = properties[name]["description"]
        assert "whole lifetime" in text and "finished one" in text, name
        assert "soft" not in text, name

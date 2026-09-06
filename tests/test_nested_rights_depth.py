"""Focused typed admission and depth-provenance tests for nested delegation."""

import json
from types import SimpleNamespace

import pytest

from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.depth_evidence import build_depth_summary
from ouroboros.task_results import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SCHEDULED,
    write_task_result,
)
from ouroboros.tools.control_delegation import (
    admitted_depth_cap,
    check_delegation_admission,
    child_budget_for_schedule,
    durable_direct_child_count,
    schedule_delegation_refusal,
    stamp_depth_provenance,
    stamp_task_assignment_depth,
)


def test_task_depth_parser_rejects_negative_values_before_int_truncation():
    from ouroboros.depth_evidence import TaskDepthError, parse_task_depth

    assert parse_task_depth(None) == 0
    assert parse_task_depth("2") == 2
    # Preserve the historical coercion contract for non-negative legacy values.
    assert parse_task_depth(True) == 1
    assert parse_task_depth(1.5) == 1
    for raw in (-1, -0.5, "-1"):
        with pytest.raises(TaskDepthError) as raised:
            parse_task_depth(raw)
        assert raised.value.code == "negative_task_depth"
    with pytest.raises(TaskDepthError) as raised:
        parse_task_depth("not-a-depth")
    assert raised.value.code == "invalid_task_depth"


def test_persisted_depth_provenance_cannot_exceed_immutable_host_ceiling():
    contract = build_task_contract({
        "delegation_budget": {
            "depth_provenance": {
                "requested_depth": 99,
                "permitted_depth": 99,
                "attempted_depth": 1,
            },
        },
    })

    assert admitted_depth_cap(contract, 7) == 10
    assert admitted_depth_cap(contract, 99) == 10
    child = child_budget_for_schedule(
        contract,
        current_depth=1,
        new_depth=2,
        max_depth=99,
        may_mutate=False,
        may_fan_out=True,
        max_children=0,
        intent_note="",
    )
    assert child["depth_provenance"]["permitted_depth"] == 10


def test_explicit_rights_are_typed_and_legacy_omission_stays_permissive():
    assert check_delegation_admission({"may_delegate": False}).reason_code == "delegation_rights_may_delegate"
    assert check_delegation_admission({"may_delegate": "false"}).reason_code == "delegation_rights_may_delegate"
    assert check_delegation_admission({"may_fan_out": False}, direct_child_count=0).ok
    second = check_delegation_admission({"may_fan_out": "false"}, direct_child_count=1)
    assert second.ok is False and second.reason_code == "delegation_rights_may_fan_out"
    exhausted = check_delegation_admission({"depth_remaining": 0})
    assert exhausted.ok is False and exhausted.reason_code == "delegation_rights_depth_exhausted"
    capped = check_delegation_admission({"max_children": 1}, direct_child_count=1)
    assert capped.ok is False and capped.reason_code == "delegation_rights_max_children"
    assert check_delegation_admission({}, direct_child_count=99).ok
    assert check_delegation_admission({}, direct_child_count=None).ok
    unknown_fanout = check_delegation_admission(
        {"may_fan_out": False}, direct_child_count=None,
    )
    assert unknown_fanout.reason_code == "delegation_rights_child_count_unknown"
    unknown_cap = check_delegation_admission(
        {"max_children": 1}, direct_child_count=None,
    )
    assert unknown_cap.reason_code == "delegation_rights_child_count_unknown"

    narrowed = child_budget_for_schedule(
        {"delegation_budget": {"may_delegate": "false", "may_fan_out": "false"}},
        current_depth=0, new_depth=1, max_depth=3, may_mutate=False,
        may_fan_out=True, max_children=0, intent_note="",
    )
    assert narrowed["may_delegate"] is False
    assert narrowed["may_fan_out"] is False


def test_depth_summary_reports_lower_cap_as_typed_reduction(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "7")
    root_contract = build_task_contract({"delegation_budget": {"depth_remaining": 3}})
    statuses = [
        {
            "task_id": f"child-{depth}",
            "depth_provenance": {
                "requested_depth": 3, "permitted_depth": 2,
                "attempted_depth": depth, "achieved_depth": depth,
            },
        }
        for depth in (1, 2)
    ]
    assert build_depth_summary(root_contract, statuses) == {
        "requested_depth": 3, "permitted_depth": 2,
        "attempted_depth": 2, "achieved_depth": 2,
        "status": "capability_reduced", "host_visible_only": True
    }


def test_depth_summary_is_order_independent_and_allows_chosen_shallower():
    root_contract = build_task_contract({
        "delegation_budget": {
            "depth_remaining": 3,
            "depth_provenance": {
                "requested_depth": 3, "permitted_depth": 3,
                "attempted_depth": 0, "achieved_depth": None,
            },
        },
    })
    mixed = [
        {"depth_provenance": {
            "requested_depth": 3, "permitted_depth": 3,
            "attempted_depth": 1, "achieved_depth": 1,
        }},
        {"depth_provenance": {
            "requested_depth": 3, "permitted_depth": 2,
            "attempted_depth": 2, "achieved_depth": 2,
        }},
    ]
    expected = {
        "requested_depth": 3, "permitted_depth": 2,
        "attempted_depth": 2, "achieved_depth": 2,
        "status": "capability_reduced", "host_visible_only": True
    }
    assert build_depth_summary(root_contract, mixed) == expected
    assert build_depth_summary(root_contract, reversed(mixed)) == expected
    assert build_depth_summary(root_contract, [mixed[0]]) == {
        "requested_depth": 3, "permitted_depth": 3,
        "attempted_depth": 1, "achieved_depth": 1,
        "status": "chosen_shallower", "host_visible_only": True
    }


def test_depth_summary_mixed_requests_uses_strongest_ask_and_branch_status():
    mixed = [
        {"depth_provenance": {
            "requested_depth": 2, "permitted_depth": 2,
            "attempted_depth": 2, "achieved_depth": 2,
        }},
        {"depth_provenance": {
            "requested_depth": 4, "permitted_depth": 4,
            "attempted_depth": 3, "achieved_depth": 3,
        }},
    ]
    expected = {
        "requested_depth": 4, "permitted_depth": 4,
        "attempted_depth": 3, "achieved_depth": 3,
        "status": "chosen_shallower", "host_visible_only": True
    }
    assert build_depth_summary({}, mixed) == expected
    assert build_depth_summary({}, reversed(mixed)) == expected


def test_depth_summary_reduced_chain_decides_over_deeper_achieved_chain():
    mixed = [
        {"depth_provenance": {
            "requested_depth": 5, "permitted_depth": 5,
            "attempted_depth": 5, "achieved_depth": 5,
        }},
        {"depth_provenance": {
            "requested_depth": 3, "permitted_depth": 2,
            "attempted_depth": 2, "achieved_depth": 2,
        }},
    ]
    expected = {
        "requested_depth": 3, "permitted_depth": 2,
        "attempted_depth": 2, "achieved_depth": 2,
        "status": "capability_reduced", "host_visible_only": True,
    }
    assert build_depth_summary({}, mixed) == expected
    assert build_depth_summary({}, reversed(mixed)) == expected


def test_depth_summary_never_recomputes_missing_history_from_live_settings(monkeypatch):
    root_contract = build_task_contract({"delegation_budget": {"depth_remaining": 3}})
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "7")
    assert build_depth_summary(root_contract, []) == {
        "requested_depth": 3, "permitted_depth": None,
        "attempted_depth": 0, "achieved_depth": 0,
        "status": "evidence_unknown", "host_visible_only": True
    }


def test_depth_provenance_follows_explicit_request_through_three_levels():
    root = build_task_contract({"delegation_budget": {"depth_remaining": 3}})
    depth_one = child_budget_for_schedule(
        root, current_depth=0, new_depth=1, max_depth=3, may_mutate=False,
        may_fan_out=True, max_children=0, intent_note="",
    )
    assert depth_one["depth_remaining"] == 2
    assert depth_one["depth_provenance"] == {
        "requested_depth": 3, "permitted_depth": 3,
        "attempted_depth": 1, "achieved_depth": None,
    }
    depth_two = child_budget_for_schedule(
        {"delegation_budget": depth_one}, current_depth=1, new_depth=2,
        max_depth=3, may_mutate=False, may_fan_out=True, max_children=0,
        intent_note="",
    )
    assert depth_two["depth_remaining"] == 1
    assert depth_two["depth_provenance"]["requested_depth"] == 3
    assert depth_two["depth_provenance"]["attempted_depth"] == 2
    depth_three = child_budget_for_schedule(
        {"delegation_budget": depth_two}, current_depth=2, new_depth=3,
        max_depth=3, may_mutate=False, may_fan_out=True, max_children=0,
        intent_note="",
    )
    assert depth_three["depth_remaining"] == 0
    assert depth_three["may_delegate"] is False
    assert depth_three["depth_provenance"] == {
        "requested_depth": 3, "permitted_depth": 3,
        "attempted_depth": 3, "achieved_depth": None,
    }


def test_depth_permission_and_remaining_never_widen_after_settings_raise():
    root = build_task_contract({"delegation_budget": {"depth_remaining": 3}})
    depth_one = child_budget_for_schedule(
        root, current_depth=0, new_depth=1, max_depth=2, may_mutate=False,
        may_fan_out=True, max_children=0, intent_note="",
    )
    assert depth_one["depth_remaining"] == 1
    assert depth_one["depth_provenance"]["permitted_depth"] == 2

    depth_two = child_budget_for_schedule(
        {"delegation_budget": depth_one}, current_depth=1, new_depth=2,
        max_depth=7, may_mutate=False, may_fan_out=True, max_children=0,
        intent_note="",
    )
    assert depth_two["depth_remaining"] == 0
    assert depth_two["may_delegate"] is False
    assert depth_two["depth_provenance"] == {
        "requested_depth": 3, "permitted_depth": 2,
        "attempted_depth": 2, "achieved_depth": None,
    }


def test_admitted_depth_cap_ignores_later_settings_decrease_but_not_fresh_roots():
    contract = build_task_contract({
        "delegation_budget": {
            "depth_remaining": 2,
            "depth_provenance": {
                "requested_depth": 3,
                "permitted_depth": 3,
                "attempted_depth": 1,
                "achieved_depth": None,
            },
        },
    })
    assert admitted_depth_cap(contract, 1) == 3
    assert admitted_depth_cap(contract, 7) == 3
    assert admitted_depth_cap(contract, 0) == 0
    assert admitted_depth_cap({}, 1) == 1
    zero_contract = build_task_contract({
        "delegation_budget": {
            "depth_provenance": {"permitted_depth": 0},
        },
    })
    assert admitted_depth_cap(zero_contract, 7) == 0


def test_persisted_depth_cap_survives_live_decrease_when_narrowing_child_budget():
    parent = build_task_contract({
        "delegation_budget": {
            "depth_remaining": 2,
            "depth_provenance": {
                "requested_depth": 3,
                "permitted_depth": 3,
                "attempted_depth": 1,
                "achieved_depth": None,
            },
        },
    })
    child = child_budget_for_schedule(
        parent,
        current_depth=1,
        new_depth=2,
        max_depth=1,
        may_mutate=False,
        may_fan_out=True,
        max_children=0,
        intent_note="",
    )
    assert child["depth_provenance"]["permitted_depth"] == 3
    assert child["depth_remaining"] == 1


def test_schedule_path_preserves_admitted_cap_after_live_depth_decrease(tmp_path, monkeypatch):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "1")
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    parent_contract = build_task_contract({
        "delegation_budget": {
            "depth_remaining": 2,
            "depth_provenance": {
                "requested_depth": 3,
                "permitted_depth": 3,
                "attempted_depth": 1,
                "achieved_depth": 1,
            },
        },
    })
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=data,
        task_id="parent",
        task_depth=1,
        task_contract=parent_contract,
        task_metadata={
            "task_contract": parent_contract,
            "root_task_id": "root",
            "parent_task_id": "parent",
            "budget_drive_root": str(data),
        },
    )

    result = _schedule_task(
        ctx,
        subagent_id=subagent_id,
        objective="Continue the admitted nested line",
        expected_output="child id",
        memory_mode="empty",
    )

    assert "subtask_depth_limit" not in result
    assert ctx.pending_events
    child_budget = ctx.pending_events[0]["task_contract"]["delegation_budget"]
    assert child_budget["depth_provenance"]["permitted_depth"] == 3


def test_schedule_path_honors_global_zero_depth_switch_for_admitted_lineage(
    tmp_path, monkeypatch,
):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "0")
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    contract = build_task_contract({
        "delegation_budget": {
            "depth_remaining": 2,
            "depth_provenance": {
                "requested_depth": 3,
                "permitted_depth": 3,
                "attempted_depth": 1,
                "achieved_depth": 1,
            },
        },
    })
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=data,
        task_id="parent",
        task_depth=1,
        task_contract=contract,
        task_metadata={"task_contract": contract, "budget_drive_root": str(data)},
    )

    result = _schedule_task(
        ctx,
        subagent_id=subagent_id,
        objective="Attempt continuation while delegation is globally disabled",
        expected_output="child id",
        memory_mode="empty",
    )

    assert "depth limit (0) exceeded" in result
    assert not ctx.pending_events


def test_supervisor_schedule_path_preserves_admitted_cap_after_live_depth_decrease(
    tmp_path, monkeypatch,
):
    from supervisor import events
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(events, "get_max_subagent_depth", lambda: 1)
    contract = build_task_contract({
        "delegation_budget": {
            "depth_remaining": 1,
            "depth_provenance": {
                "requested_depth": 3,
                "permitted_depth": 3,
                "attempted_depth": 2,
                "achieved_depth": None,
            },
        },
    })
    event = _schedule_event("child", "parent", depth=2, drive_root=tmp_path)
    event["task_contract"] = contract
    enqueued = []
    events._handle_schedule_task(event, _fake_ctx(tmp_path, enqueued))

    assert [task["id"] for task in enqueued] == ["child"]
    queued = json.loads((tmp_path / "task_results" / "child.json").read_text())
    assert queued["depth_provenance"]["permitted_depth"] == 3

    monkeypatch.setattr(events, "get_max_subagent_depth", lambda: 0)
    event["task_id"] = "child-zero"
    events._handle_schedule_task(event, _fake_ctx(tmp_path, enqueued))
    rejected = json.loads((tmp_path / "task_results" / "child-zero.json").read_text())
    assert rejected["status"] == "failed"
    assert "depth limit (0)" in rejected["result"]


def test_assignment_preserves_admitted_depth_authority_and_only_adds_achievement():
    contract = build_task_contract({
        "delegation_budget": {
            "depth_remaining": 1,
            "depth_provenance": {
                "requested_depth": 3,
                "permitted_depth": 2,
                "attempted_depth": 1,
                "achieved_depth": None,
            },
        },
    })
    task = {"depth": 1, "task_contract": contract, "metadata": {}}
    fields = stamp_task_assignment_depth(task, max_depth=7)
    assert fields["depth_provenance"] == {
        "requested_depth": 3, "permitted_depth": 2,
        "attempted_depth": 1, "achieved_depth": 1,
    }
    assert task["metadata"]["depth_provenance"] == fields["depth_provenance"]


def test_assignment_does_not_reconstruct_legacy_depth_authority_from_live_settings():
    contract = build_task_contract({
        "delegation_budget": {"depth_remaining": 2},
    })
    task = {"depth": 1, "task_contract": contract, "metadata": {}}

    fields = stamp_task_assignment_depth(task, max_depth=7)

    assert fields["depth_provenance"] == {
        "requested_depth": None,
        "permitted_depth": None,
        "attempted_depth": 1,
        "achieved_depth": 1,
    }


def test_legacy_depth_provenance_branch_respects_host_ceiling():
    contract = build_task_contract({
        "delegation_budget": {"depth_remaining": 20},
    })

    _stamped, provenance = stamp_depth_provenance(
        contract, attempted_depth=1, max_depth=99, achieved_depth=None,
    )

    assert provenance["permitted_depth"] == 10


def test_supervisor_ingress_bounds_legacy_permission_by_admitted_remaining_envelope():
    contract = build_task_contract({
        "delegation_budget": {"depth_remaining": 2},
    })

    stamped, provenance = stamp_depth_provenance(
        contract,
        attempted_depth=1,
        max_depth=7,
        achieved_depth=None,
    )

    assert provenance == {
        "requested_depth": None,
        "permitted_depth": 3,
        "attempted_depth": 1,
        "achieved_depth": None,
    }
    assert stamped["delegation_budget"]["depth_provenance"] == provenance


def test_supervisor_ingress_records_explicit_root_depth_request(tmp_path, monkeypatch):
    from supervisor import events
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    contract = build_task_contract({
        "delegation_budget": {"depth_remaining": 3},
    })
    event = _schedule_event("root", "", depth=0, drive_root=tmp_path)
    event.update({
        "type": "schedule_subagent",
        "chat_id": 1,
        "delegation_role": "root",
        "root_task_id": "root",
        "task_contract": contract,
    })
    enqueued = []

    events._handle_schedule_task(event, _fake_ctx(tmp_path, enqueued))

    assert len(enqueued) == 1
    assert enqueued[0]["task_contract"]["delegation_budget"]["depth_provenance"] == {
        "requested_depth": 3,
        "permitted_depth": 3,
        "attempted_depth": 0,
        "achieved_depth": None,
    }


def test_legacy_budget_reports_unknown_request_but_current_permission():
    budget = child_budget_for_schedule(
        {}, current_depth=0, new_depth=1, max_depth=3, may_mutate=False,
        may_fan_out=True, max_children=0, intent_note="",
    )
    assert budget["depth_provenance"] == {
        "requested_depth": None, "permitted_depth": 3,
        "attempted_depth": 1, "achieved_depth": None,
    }

    # A legacy descendant's `depth_remaining` has already been narrowed by its
    # ancestors. It is not proof of the root's requested envelope.
    descendant = child_budget_for_schedule(
        {"delegation_budget": {"depth_remaining": 2}},
        current_depth=1, new_depth=2, max_depth=4, may_mutate=False,
        may_fan_out=True, max_children=0, intent_note="",
    )
    assert descendant["depth_provenance"] == {
        "requested_depth": None, "permitted_depth": 3,
        "attempted_depth": 2, "achieved_depth": None,
    }
    assert descendant["depth_remaining"] == 1


def test_legacy_contract_does_not_gain_provenance_during_recovery_normalization():
    # Existing delegated rows were fingerprinted without this additive projection.
    # Rebuilding such a row after restart must preserve its canonical budget shape;
    # only an explicitly authored projection is normalized into the frozen contract.
    legacy = build_task_contract({"delegation_budget": {"depth_remaining": 2}})
    assert "depth_provenance" not in legacy["delegation_budget"]
    recovered = build_task_contract({"task_contract": legacy})
    assert recovered["delegation_budget"] == legacy["delegation_budget"]
    explicit = build_task_contract({
        "delegation_budget": {
            "depth_remaining": 2,
            "depth_provenance": {"requested_depth": 3, "attempted_depth": 1},
        },
    })
    assert explicit["delegation_budget"]["depth_provenance"]["requested_depth"] == 3


def test_fresh_depth_default_increases_without_widening_active_cap(monkeypatch):
    from ouroboros.config import get_max_active_subagents_per_root, get_max_subagent_depth, get_max_workers

    monkeypatch.delenv("OUROBOROS_MAX_SUBAGENT_DEPTH", raising=False)
    monkeypatch.delenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", raising=False)
    assert get_max_subagent_depth() == 3
    assert get_max_active_subagents_per_root() == 6
    assert get_max_workers() == 10


def _schedule_event(task_id, parent_id, *, depth=1, drive_root=None):
    root = str(drive_root or "")
    return {
        "type": "schedule_subagent", "task_id": task_id,
        "objective": f"objective-{task_id}", "expected_output": "a result",
        "depth": depth, "parent_task_id": parent_id, "root_task_id": parent_id,
        "delegation_role": "subagent", "memory_mode": "forked", "drive_root": root,
        "child_drive_root": root, "budget_drive_root": root,
    }


def _fake_ctx(tmp_path, enqueued):
    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 0}

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            return None

        def send_with_budget(self, *args, **kwargs):
            return None

    return FakeCtx()


def test_supervisor_admission_enforces_parent_rights_and_allows_one_non_fanout_child(tmp_path, monkeypatch):
    from supervisor import events
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    parent_contract = build_task_contract({"delegation_budget": {"may_fan_out": False}})
    write_task_result(tmp_path, "parent", STATUS_RUNNING, parent_task_id="", root_task_id="parent",
                      delegation_role="root", task_contract=parent_contract)
    enqueued = []
    ctx = _fake_ctx(tmp_path, enqueued)
    events._handle_schedule_task(_schedule_event("child-1", "parent", drive_root=tmp_path), ctx)
    assert [task["id"] for task in enqueued] == ["child-1"]
    queued = json.loads((tmp_path / "task_results" / "child-1.json").read_text(encoding="utf-8"))
    assert queued["depth_provenance"]["achieved_depth"] is None
    events._handle_schedule_task(_schedule_event("child-2", "parent", drive_root=tmp_path), ctx)
    rejected = json.loads((tmp_path / "task_results" / "child-2.json").read_text(encoding="utf-8"))
    assert len(enqueued) == 1
    assert rejected["reason_code"] == "delegation_rights_may_fan_out"


def test_supervisor_rejects_invalid_depth_before_provisioning_or_enqueue(tmp_path, monkeypatch):
    from supervisor import events
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    for index, raw_depth in enumerate((-1, -0.5, "-1", "not-a-depth")):
        task_id = f"invalid-depth-{index}"
        enqueued = []
        events._handle_schedule_task(
            _schedule_event(task_id, "parent", depth=raw_depth, drive_root=tmp_path),
            _fake_ctx(tmp_path, enqueued),
        )
        result = json.loads((tmp_path / "task_results" / f"{task_id}.json").read_text())
        assert enqueued == []
        assert result["status"] == STATUS_FAILED
        assert result["reason_code"] == "invalid_task_depth"


def test_supervisor_rolls_back_subagent_when_scheduled_result_write_fails(
    tmp_path, monkeypatch,
):
    from supervisor import events, task_admission
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    parent_contract = build_task_contract({"delegation_budget": {"may_fan_out": False}})
    write_task_result(
        tmp_path, "parent", STATUS_RUNNING,
        parent_task_id="", root_task_id="parent",
        delegation_role="root", task_contract=parent_contract,
    )
    enqueued = []
    ctx = _fake_ctx(tmp_path, enqueued)
    write_task_result(
        tmp_path, "child-1", "requested",
        parent_task_id="parent", root_task_id="parent",
        delegation_role="subagent", result="Awaiting supervisor acceptance.",
    )

    def enqueue_task(task):
        admitted = dict(task)
        enqueued.append(admitted)
        ctx.PENDING.append(admitted)
        return admitted

    ctx.enqueue_task = enqueue_task
    original_write = task_admission.write_task_result

    def fail_first_scheduled(root, task_id, status, **fields):
        if task_id == "child-1" and status == events.STATUS_SCHEDULED:
            raise OSError("simulated scheduled receipt failure")
        return original_write(root, task_id, status, **fields)

    monkeypatch.setattr(task_admission, "write_task_result", fail_first_scheduled)

    events._handle_schedule_task(
        _schedule_event("child-1", "parent", drive_root=tmp_path), ctx,
    )
    assert [task["id"] for task in enqueued] == ["child-1"]
    assert ctx.PENDING == []
    rejected_first = json.loads(
        (tmp_path / "task_results" / "child-1.json").read_text(encoding="utf-8")
    )
    assert rejected_first["status"] == "failed"
    assert rejected_first["reason_code"] == "scheduled_result_persist_failed"
    assert rejected_first["delegation_admission"]["status"] == "rejected"

    events._handle_schedule_task(
        _schedule_event("child-2", "parent", drive_root=tmp_path), ctx,
    )
    scheduled = json.loads(
        (tmp_path / "task_results" / "child-2.json").read_text(encoding="utf-8")
    )
    assert [task["id"] for task in enqueued] == ["child-1", "child-2"]
    assert [task["id"] for task in ctx.PENDING] == ["child-2"]
    assert scheduled["status"] == "scheduled"
    assert scheduled["delegation_admission"]["status"] == "accepted"
    assert scheduled["delegation_admission"]["direct_child_count"] == 0
    assert len(scheduled["delegation_admission"]["transition_id"]) == 32


def test_supervisor_receipt_rollback_removes_only_its_enqueue_identity(
    tmp_path, monkeypatch,
):
    from supervisor import events, task_admission
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    parent_contract = build_task_contract({"delegation_budget": {"may_fan_out": True}})
    write_task_result(
        tmp_path, "parent", STATUS_RUNNING,
        root_task_id="parent", delegation_role="root", task_contract=parent_contract,
    )
    enqueued = []
    ctx = _fake_ctx(tmp_path, enqueued)
    preexisting = {
        "id": "same-id",
        "root_task_id": "parent",
        "parent_task_id": "parent",
        "delegation_role": "subagent",
    }
    ctx.PENDING.append(preexisting)
    write_task_result(
        tmp_path,
        "same-id",
        "scheduled",
        parent_task_id="parent",
        root_task_id="parent",
        delegation_role="subagent",
        delegation_admission={
            "status": "accepted",
            "direct_child_count": 0,
            "transition_id": "old-transition",
        },
    )

    def enqueue_task(task):
        admitted = dict(task)
        enqueued.append(admitted)
        ctx.PENDING.append(admitted)
        return admitted

    ctx.enqueue_task = enqueue_task
    original_write = task_admission.write_task_result

    def fail_scheduled(root, task_id, status, **fields):
        if task_id == "same-id" and status == events.STATUS_SCHEDULED:
            raise OSError("simulated pre-commit failure")
        return original_write(root, task_id, status, **fields)

    monkeypatch.setattr(task_admission, "write_task_result", fail_scheduled)

    events._handle_schedule_task(
        _schedule_event("same-id", "parent", drive_root=tmp_path), ctx,
    )

    assert len(ctx.PENDING) == 1
    assert ctx.PENDING[0] is preexisting
    preserved = json.loads(
        (tmp_path / "task_results" / "same-id.json").read_text(encoding="utf-8")
    )
    assert preserved["status"] == "scheduled"
    assert preserved["delegation_admission"] == {
        "status": "accepted",
        "direct_child_count": 0,
        "transition_id": "old-transition",
    }


def test_replayed_schedule_event_keeps_one_physical_task_and_transition(
    tmp_path, monkeypatch,
):
    from supervisor import events, queue, state, workers
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    write_task_result(
        tmp_path, "parent", STATUS_RUNNING,
        root_task_id="parent", delegation_role="root",
        task_contract=build_task_contract({"delegation_budget": {"may_fan_out": True}}),
    )
    write_task_result(
        tmp_path, "same-child", "requested",
        parent_task_id="parent", root_task_id="parent",
        delegation_role="subagent", result="Awaiting supervisor acceptance.",
    )
    ctx = _fake_ctx(tmp_path, [])

    def enqueue_task(task):
        admitted = dict(task)
        ctx.PENDING.append(admitted)
        return admitted

    ctx.enqueue_task = enqueue_task
    event = _schedule_event("same-child", "parent", drive_root=tmp_path)
    events._handle_schedule_task(event, ctx)
    first = json.loads(
        (tmp_path / "task_results" / "same-child.json").read_text(encoding="utf-8")
    )
    transition_id = first["delegation_admission"]["transition_id"]

    def unexpected_constraint_resolution(*_args, **_kwargs):
        raise AssertionError("a replay must stop before workspace provisioning")

    monkeypatch.setattr(schedule_module, "_resolve_subagent_constraint", unexpected_constraint_resolution)
    events._handle_schedule_task(event, ctx)
    replayed = json.loads(
        (tmp_path / "task_results" / "same-child.json").read_text(encoding="utf-8")
    )
    assert [task["id"] for task in ctx.PENDING] == ["same-child"]
    assert replayed["delegation_admission"]["transition_id"] == transition_id

    delivered = []

    class FakeWorkerQueue:
        def put(self, task):
            delivered.append(dict(task))

    worker_map = {
        wid: SimpleNamespace(wid=wid, busy_task_id=None, in_q=FakeWorkerQueue())
        for wid in (1, 2)
    }
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", ctx.PENDING)
    monkeypatch.setattr(workers, "RUNNING", ctx.RUNNING)
    monkeypatch.setattr(workers, "WORKERS", worker_map)
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_args, **_kwargs: 100.0)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)

    workers.assign_tasks()

    assert [task["id"] for task in delivered] == ["same-child"]
    assert sum(worker.busy_task_id == "same-child" for worker in worker_map.values()) == 1
    assert ctx.RUNNING["same-child"]["worker_id"] in worker_map


def test_assignment_quarantines_bypassed_invalid_depth_and_normalizes_legacy_rows(
    tmp_path, monkeypatch,
):
    from supervisor import queue, state, workers
    from ouroboros.task_results import load_task_result

    delivered = []
    terminal_events = []

    class FakeWorkerQueue:
        def put(self, task):
            delivered.append(dict(task))

    worker = SimpleNamespace(wid=1, busy_task_id=None, in_q=FakeWorkerQueue())
    pending = [{
        "id": "invalid-pending-depth",
        "type": "task",
        "chat_id": 1,
        "description": "invalid depth",
        "depth": -1,
        "budget_drive_root": str(tmp_path),
    }]
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {1: worker})
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_args, **_kwargs: 100.0)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(
        workers,
        "_emit_task_done_terminal",
        lambda task, task_id, status="failed", **kwargs: terminal_events.append(
            (task_id, status, kwargs)
        ) or True,
    )
    queue.BUDGET_ROOT_FENCES.clear()

    workers.assign_tasks()

    rejected = load_task_result(tmp_path, "invalid-pending-depth")
    assert pending == []
    assert delivered == []
    assert worker.busy_task_id is None
    assert rejected["status"] == STATUS_FAILED
    assert rejected["reason_code"] == "invalid_task_depth"
    assert rejected["depth"] == 0
    assert rejected["raw_task_depth"] == -1
    assert terminal_events and terminal_events[0][0] == "invalid-pending-depth"

    pending.append({
        "id": "legacy-pending-depth",
        "type": "task",
        "chat_id": 1,
        "description": "legacy missing depth",
        "depth": None,
        "budget_drive_root": str(tmp_path),
    })
    workers.assign_tasks()
    assert delivered and delivered[-1]["id"] == "legacy-pending-depth"
    assert delivered[-1]["depth"] == 0


def test_assignment_quarantines_invalid_depth_before_budget_pause(tmp_path, monkeypatch):
    from supervisor import queue, state, workers
    from ouroboros.task_results import load_task_result

    delivered = []
    worker = SimpleNamespace(
        wid=1,
        busy_task_id=None,
        in_q=SimpleNamespace(put=lambda task: delivered.append(dict(task))),
    )
    pending = [{
        "id": "invalid-before-budget",
        "type": "task",
        "chat_id": 1,
        "description": "invalid depth must not become a pause",
        "depth": -0.5,
        "budget_drive_root": str(tmp_path),
    }]
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {1: worker})
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(state, "budget_remaining", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    queue.BUDGET_ROOT_FENCES.clear()

    workers.assign_tasks()

    rejected = load_task_result(tmp_path, "invalid-before-budget")
    assert pending == []
    assert delivered == []
    assert rejected["status"] == STATUS_FAILED
    assert rejected["reason_code"] == "invalid_task_depth"
    assert rejected["raw_task_depth"] == -0.5
    assert "_budget_pause" not in rejected


def test_restore_failed_depth_terminalization_mutates_pending_under_queue_lock(
    tmp_path, monkeypatch,
):
    from supervisor import queue, task_admission
    from ouroboros.utils import utc_now_iso

    pending, running = [], {}
    queue.init_queue_refs(pending, running, {"value": 0})
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    snapshot_path = tmp_path / "state" / "queue_snapshot.json"
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", snapshot_path)
    queue.ACCEPTANCE_FENCES.clear()
    queue.ADMISSION_RESERVATIONS.clear()
    task = {
        "id": "restore-retry-invalid-depth",
        "type": "task",
        "chat_id": 1,
        "description": "retry terminal custody",
        "depth": -1,
        "priority": "not-an-integer",
        "_queue_seq": "not-a-sequence",
    }
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(
        json.dumps({
            "ts": utc_now_iso(),
            "pending": [{"task": task}],
            "running": [],
            "acceptance_fences": [],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        task_admission,
        "terminalize_invalid_depth_restore",
        lambda *_args, **_kwargs: False,
    )
    lock_observations = []
    original_restore = queue.restore_invalid_depth_admission

    def checked_restore(*args, **kwargs):
        lock_observations.append(queue._queue_lock._is_owned())
        return original_restore(*args, **kwargs)

    monkeypatch.setattr(queue, "restore_invalid_depth_admission", checked_restore)

    assert queue.restore_pending_from_snapshot() == 0
    assert lock_observations == [True]
    assert len(pending) == 1
    assert pending[0]["id"] == task["id"]
    assert pending[0]["depth"] == -1
    assert "priority" not in pending[0]
    assert pending[0]["_queue_seq"] == 1

    admitted = queue.enqueue_task({"id": "healthy-after-bad-order", "type": "task", "depth": 0})
    assert admitted["id"] == "healthy-after-bad-order"
    assert [row["id"] for row in pending] == [
        "restore-retry-invalid-depth", "healthy-after-bad-order",
    ]


@pytest.mark.parametrize("raw_value", [float("inf"), float("-inf")])
def test_restore_failed_depth_terminalization_sanitizes_nonfinite_queue_metadata(
    raw_value, tmp_path, monkeypatch,
):
    from supervisor import queue, task_admission
    from ouroboros.utils import utc_now_iso

    pending, running = [], {}
    queue.init_queue_refs(pending, running, {"value": 0})
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    snapshot_path = tmp_path / "state" / "queue_snapshot.json"
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", snapshot_path)
    queue.ACCEPTANCE_FENCES.clear()
    queue.ADMISSION_RESERVATIONS.clear()
    task = {
        "id": "restore-retry-nonfinite-order",
        "type": "task",
        "chat_id": 1,
        "description": "retain malformed depth",
        "depth": -1,
        "priority": raw_value,
        "_queue_seq": raw_value,
    }
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(
        json.dumps({
            "ts": utc_now_iso(),
            "pending": [{"task": task}],
            "running": [],
            "acceptance_fences": [],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        task_admission,
        "terminalize_invalid_depth_restore",
        lambda *_args, **_kwargs: False,
    )

    assert queue.restore_pending_from_snapshot() == 0
    assert len(pending) == 1
    assert pending[0]["id"] == task["id"]
    assert pending[0]["depth"] == -1
    assert "priority" not in pending[0]
    assert pending[0]["_queue_seq"] == 1
    admitted = queue.enqueue_task({"id": "healthy-after-nonfinite-order", "type": "task", "depth": 0})
    assert admitted["id"] == "healthy-after-nonfinite-order"


def test_budget_pause_leaves_unresolved_invalid_depth_in_retry_custody(
    tmp_path, monkeypatch,
):
    from supervisor import queue, state, workers
    from ouroboros.task_results import load_task_result

    pending = [
        {
            "id": "unresolved-before-budget",
            "type": "task",
            "chat_id": 1,
            "description": "retry terminal custody",
            "depth": -1,
            "budget_drive_root": str(tmp_path),
        },
        {
            "id": "healthy-before-budget",
            "type": "task",
            "chat_id": 1,
            "description": "pause this task",
            "depth": 0,
            "budget_drive_root": str(tmp_path),
        },
    ]
    worker = SimpleNamespace(wid=1, busy_task_id=None, reaping=False, in_q=SimpleNamespace(put=lambda _task: None))
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {1: worker})
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(state, "budget_remaining", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    original_terminalize = workers._terminalize_invalid_pending_depth
    monkeypatch.setattr(
        workers,
        "_terminalize_invalid_pending_depth",
        lambda task, detail: (
            False
            if task.get("id") == "unresolved-before-budget"
            else original_terminalize(task, detail)
        ),
    )
    queue.BUDGET_ROOT_FENCES.clear()

    workers.assign_tasks()

    unresolved = next(task for task in pending if task["id"] == "unresolved-before-budget")
    healthy = next(task for task in pending if task["id"] == "healthy-before-budget")
    assert "_budget_pause" not in unresolved
    assert healthy.get("_budget_pause", {}).get("status") == "paused_before_dispatch"
    assert load_task_result(tmp_path, "healthy-before-budget")["status"] == STATUS_SCHEDULED
    assert worker.busy_task_id is None


def test_supervisor_keeps_admission_when_scheduled_write_raises_after_commit(
    tmp_path, monkeypatch,
):
    from supervisor import events, task_admission
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    parent_contract = build_task_contract({"delegation_budget": {"may_fan_out": False}})
    write_task_result(
        tmp_path, "parent", STATUS_RUNNING,
        root_task_id="parent", delegation_role="root", task_contract=parent_contract,
    )
    enqueued = []
    ctx = _fake_ctx(tmp_path, enqueued)

    def enqueue_task(task):
        admitted = dict(task)
        enqueued.append(admitted)
        ctx.PENDING.append(admitted)
        return admitted

    ctx.enqueue_task = enqueue_task
    original_write = task_admission.write_task_result

    def commit_then_raise(root, task_id, status, **fields):
        stored = original_write(root, task_id, status, **fields)
        if task_id == "child" and status == events.STATUS_SCHEDULED:
            raise OSError("simulated post-commit observer failure")
        return stored

    monkeypatch.setattr(task_admission, "write_task_result", commit_then_raise)

    events._handle_schedule_task(
        _schedule_event("child", "parent", drive_root=tmp_path), ctx,
    )

    assert [task["id"] for task in ctx.PENDING] == ["child"]
    scheduled = json.loads(
        (tmp_path / "task_results" / "child.json").read_text(encoding="utf-8")
    )
    assert scheduled["status"] == "scheduled"
    assert scheduled["delegation_admission"]["status"] == "accepted"
    assert len(scheduled["delegation_admission"]["transition_id"]) == 32
    assert scheduled["delegation_admission"]["transition_id"] != "old-transition"


def test_supervisor_rolls_back_when_monotonic_writer_returns_old_terminal(
    tmp_path, monkeypatch,
):
    from supervisor import events
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    parent_contract = build_task_contract({"delegation_budget": {"may_fan_out": True}})
    write_task_result(
        tmp_path, "parent", STATUS_RUNNING,
        root_task_id="parent", delegation_role="root", task_contract=parent_contract,
    )
    write_task_result(
        tmp_path, "child", "completed",
        parent_task_id="parent", root_task_id="parent",
        delegation_role="subagent", result="old terminal child",
    )
    enqueued = []
    ctx = _fake_ctx(tmp_path, enqueued)

    def enqueue_task(task):
        admitted = dict(task)
        enqueued.append(admitted)
        ctx.PENDING.append(admitted)
        return admitted

    ctx.enqueue_task = enqueue_task
    events._handle_schedule_task(
        _schedule_event("child", "parent", drive_root=tmp_path), ctx,
    )

    assert ctx.PENDING == []
    terminal = json.loads(
        (tmp_path / "task_results" / "child.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "completed"
    assert terminal["result"] == "old terminal child"


def test_supervisor_admission_enforces_may_delegate_false_even_when_stringified(tmp_path):
    from supervisor import events

    parent_contract = build_task_contract({"delegation_budget": {"may_delegate": "false"}})
    write_task_result(tmp_path, "parent", STATUS_RUNNING, root_task_id="parent",
                      delegation_role="root", task_contract=parent_contract)
    enqueued = []
    ctx = _fake_ctx(tmp_path, enqueued)
    events._handle_schedule_task(_schedule_event("child", "parent", drive_root=tmp_path), ctx)
    rejected = json.loads((tmp_path / "task_results" / "child.json").read_text(encoding="utf-8"))
    assert enqueued == []
    assert rejected["reason_code"] == "delegation_rights_may_delegate"


def test_direct_child_count_read_gap_is_typed_unknown(tmp_path):
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "corrupt-child.json").write_text("{not-json", encoding="utf-8")
    contract = build_task_contract({"delegation_budget": {"max_children": 1}})

    assert durable_direct_child_count(tmp_path, "parent") is None
    refusal = schedule_delegation_refusal(contract, tmp_path, "parent")
    assert "delegation_rights_child_count_unknown" in refusal


def test_direct_child_count_rejects_schema_empty_result(tmp_path):
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "empty-child.json").write_text("{}\n", encoding="utf-8")
    contract = build_task_contract({"delegation_budget": {"max_children": 1}})

    assert durable_direct_child_count(tmp_path, "parent") is None
    assert "delegation_rights_child_count_unknown" in schedule_delegation_refusal(
        contract, tmp_path, "parent",
    )


def test_schedule_exact_id_preserves_unreadable_result_and_does_not_enqueue(
    tmp_path, monkeypatch,
):
    from supervisor import events
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    result_path = tmp_path / "task_results" / "child-malformed.json"
    result_path.parent.mkdir()
    malformed = b'{"status":"scheduled"'
    result_path.write_bytes(malformed)
    enqueued = []
    notices = []
    ctx = _fake_ctx(tmp_path, enqueued)
    ctx.load_state = lambda: {"owner_chat_id": 7}
    ctx.send_with_budget = lambda *args, **kwargs: notices.append((args, kwargs))

    events._handle_schedule_task(
        _schedule_event("child-malformed", "parent", drive_root=tmp_path), ctx,
    )

    assert enqueued == []
    assert result_path.read_bytes() == malformed
    assert notices
    assert "unreadable" in notices[0][0][1]
    assert notices[0][1]["progress_meta"]["status"] == STATUS_FAILED


def test_late_schedule_lookup_failure_preserves_exact_result(tmp_path, monkeypatch):
    from supervisor import events, task_admission
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    result_path = tmp_path / "task_results" / "late-corrupt.json"
    result_path.parent.mkdir()
    original = b"{late-corrupt"
    monkeypatch.setattr(
        task_admission, "subagent_schedule_owned", lambda *_args, **_kwargs: False,
    )
    def late_refusal(_task):
        result_path.write_bytes(original)
        return {"_admission_blocked": "task_id_lookup_failed"}

    ctx = _fake_ctx(tmp_path, [])
    ctx.enqueue_task = late_refusal
    events._handle_schedule_task(
        _schedule_event("late-corrupt", "parent", drive_root=tmp_path), ctx,
    )
    assert result_path.read_bytes() == original


def test_generic_late_schedule_lookup_failure_preserves_exact_result(tmp_path, monkeypatch):
    from supervisor import events, queue
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    result_path = tmp_path / "task_results" / "generic-late-corrupt.json"
    result_path.parent.mkdir()
    original = b"{generic-late-corrupt"
    result_path.write_bytes(original)
    pending = []
    running = {}
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", running)
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    ctx = _fake_ctx(tmp_path, [])
    ctx.PENDING = pending
    ctx.RUNNING = running
    ctx.enqueue_task = queue.enqueue_task
    event = _schedule_event("generic-late-corrupt", "parent", drive_root=tmp_path)
    event["delegation_role"] = "root"
    event["chat_id"] = 7
    events._handle_schedule_task(event, ctx)

    assert pending == []
    assert result_path.read_bytes() == original


def test_generic_schedule_replay_preserves_valid_exact_result(tmp_path, monkeypatch):
    from supervisor import events, queue
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    write_task_result(
        tmp_path, "generic-existing", "completed",
        root_task_id="generic-existing", delegation_role="root", result="keep me",
    )
    result_path = tmp_path / "task_results" / "generic-existing.json"
    original = result_path.read_bytes()
    pending = []
    running = {}
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", running)
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    ctx = _fake_ctx(tmp_path, [])
    ctx.PENDING = pending
    ctx.RUNNING = running
    ctx.enqueue_task = queue.enqueue_task
    event = _schedule_event("generic-existing", "parent", drive_root=tmp_path)
    event["delegation_role"] = "root"
    event["chat_id"] = 7

    events._handle_schedule_task(event, ctx)

    assert pending == []
    assert result_path.read_bytes() == original


@pytest.mark.parametrize(
    ("status", "location"),
    [(STATUS_SCHEDULED, "pending"), (STATUS_RUNNING, "running")],
)
def test_generic_malformed_replay_preserves_live_exact_result(
    tmp_path, monkeypatch, status, location,
):
    from supervisor import events, queue
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    tid = f"generic-live-{location}"
    write_task_result(
        tmp_path,
        tid,
        status,
        root_task_id=tid,
        delegation_role="root",
        result="keep live work",
    )
    result_path = tmp_path / "task_results" / f"{tid}.json"
    original = result_path.read_bytes()
    pending = []
    running = {}
    if location == "pending":
        pending.append({"id": tid, "depth": 0, "delegation_role": "root"})
    else:
        running[tid] = {"task": {"id": tid, "delegation_role": "root"}}
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", running)
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    enqueued = []
    ctx = _fake_ctx(tmp_path, enqueued)
    ctx.PENDING = pending
    ctx.RUNNING = running
    ctx.enqueue_task = queue.enqueue_task
    event = _schedule_event(tid, "parent", depth=-1, drive_root=tmp_path)
    event["delegation_role"] = "root"
    event["chat_id"] = 7

    events._handle_schedule_task(event, ctx)

    assert result_path.read_bytes() == original
    assert pending == (
        [{"id": tid, "depth": 0, "delegation_role": "root"}]
        if location == "pending" else []
    )
    assert set(running) == ({tid} if location == "running" else set())
    assert not enqueued


@pytest.mark.parametrize("delegation_role", ["root", "subagent"])
def test_malformed_replay_preserves_preexisting_admission_reservation(
    tmp_path, monkeypatch, delegation_role,
):
    from supervisor import events, queue

    tid = f"reserved-{delegation_role}"
    pending, running = [], {}
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", running)
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {tid: "owner-token"})
    enqueued = []
    ctx = _fake_ctx(tmp_path, enqueued)
    ctx.PENDING, ctx.RUNNING = pending, running
    ctx.enqueue_task = queue.enqueue_task
    event = _schedule_event(tid, "parent", depth=-1, drive_root=tmp_path)
    event["delegation_role"], event["chat_id"] = delegation_role, 7

    events._handle_schedule_task(event, ctx)

    assert not (tmp_path / "task_results" / f"{tid}.json").exists()
    assert queue.ADMISSION_RESERVATIONS == {tid: "owner-token"}
    assert pending == []
    assert running == {}
    assert enqueued == []


@pytest.mark.parametrize("delegation_role", ["root", "subagent"])
@pytest.mark.parametrize("late_custody", ["reservation", "pending"])
def test_malformed_replay_rechecks_identity_after_initial_preflight(
    tmp_path, monkeypatch, delegation_role, late_custody,
):
    from supervisor import events, queue, task_admission

    tid = f"late-{delegation_role}-{late_custody}"
    pending, running = [], {}
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", running)
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    calls = 0

    def raced_owned(_ctx, _task_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            if late_custody == "reservation":
                queue.ADMISSION_RESERVATIONS[tid] = "late-token"
            else:
                pending.append({"id": tid, "delegation_role": delegation_role})
            return False
        return bool(
            queue.ADMISSION_RESERVATIONS.get(tid)
            or any(str(row.get("id") or "") == tid for row in pending)
        )

    monkeypatch.setattr(task_admission, "subagent_schedule_owned", raced_owned)
    enqueued = []
    ctx = _fake_ctx(tmp_path, enqueued)
    ctx.PENDING, ctx.RUNNING = pending, running
    ctx.enqueue_task = queue.enqueue_task
    event = _schedule_event(tid, "parent", depth=-1, drive_root=tmp_path)
    event["delegation_role"], event["chat_id"] = delegation_role, 7

    events._handle_schedule_task(event, ctx)

    assert calls == 2
    assert not (tmp_path / "task_results" / f"{tid}.json").exists()
    assert enqueued == []
    if late_custody == "reservation":
        assert queue.ADMISSION_RESERVATIONS == {tid: "late-token"}
        assert pending == []
    else:
        assert queue.ADMISSION_RESERVATIONS == {}
        assert pending == [{"id": tid, "delegation_role": delegation_role}]


def test_supervisor_rejects_count_bounded_child_when_count_scan_fails(
    tmp_path, monkeypatch,
):
    from supervisor import events
    from supervisor import events_schedule_task as schedule_module

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "ouroboros.task_results.list_task_results",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreadable")),
    )
    parent_contract = build_task_contract({
        "delegation_budget": {"may_fan_out": False, "max_children": 1},
    })
    write_task_result(
        tmp_path, "parent", STATUS_RUNNING, root_task_id="parent",
        delegation_role="root", task_contract=parent_contract,
    )
    enqueued = []
    ctx = _fake_ctx(tmp_path, enqueued)

    events._handle_schedule_task(
        _schedule_event("child", "parent", drive_root=tmp_path), ctx,
    )

    rejected = json.loads(
        (tmp_path / "task_results" / "child.json").read_text(encoding="utf-8")
    )
    assert enqueued == []
    assert rejected["reason_code"] == "delegation_rights_child_count_unknown"
    assert rejected["delegation_admission"]["direct_child_count"] is None

"""Review regressions at continuation, browser and actual schedule-control boundaries."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
from tests.test_schedule_occurrence import q as queue_fixture
supervisor, q = supervisor_fixture, queue_fixture


@pytest.mark.parametrize("level", ["observe", "act", "full"])
def test_continue_inherits_owner_or_consciousness_origin_by_value(q, level):
    from supervisor.continuation_admission import _successor_task
    from ouroboros.consciousness_authority import disabled_tools_for, runtime_mode_cap_for
    for kind in ("owner", "consciousness"):
        origin = {"work_kind": kind, "resource_intent": {"kind": "system_repo"},
                  "focus": {"place_id": "room"}, "origin_wake_task_id": "wake"}
        if kind == "consciousness":
            origin.update(initiator="consciousness", consciousness_autonomy=level, consciousness_cause="started")
        before = deepcopy(origin)
        result = {"task_id": "previous", "status": "failed", "objective": "The work", "metadata": origin}
        task = _successor_task(None, "previous", result, {"successor_task_id": f"next-{kind}", "chat_id": 1},
                               {"cause": "budget"}, {"original": {"content": "The work"}}, {})
        assert task["metadata"]["work_kind"] == kind
        task["metadata"]["focus"]["place_id"] = "changed"
        assert origin == before
        admitted = q.queue.enqueue_task(task, restoring_snapshot=True)
        assert admitted["metadata"]["work_kind"] == kind
        if kind == "consciousness":
            assert admitted["metadata"]["consciousness_cause"] == "started"
            assert admitted["metadata"]["runtime_mode_cap"] == runtime_mode_cap_for(level)
            assert admitted["task_contract"]["disabled_tools"] == disabled_tools_for(level)
        else:
            assert "initiator" not in admitted["metadata"] and "runtime_mode_cap" not in admitted["metadata"]


@pytest.mark.parametrize("metadata", [{}, {"work_kind": "owner"}, {"work_kind": "consciousness"},
                                     {"initiator": "consciousness", "consciousness_autonomy": "act"}])
def test_continue_legacy_owner_source_does_not_override_consciousness(metadata):
    from supervisor.continuation_admission import _successor_task
    from ouroboros.consciousness_authority import work_kind
    for ref in (None, {"chat_id": 1, "client_message_id": "owner-request"}):
        task = _successor_task(None, "previous", {"metadata": metadata, "objective": "The work"},
                               {"successor_task_id": "next", "chat_id": 1}, {"cause": "budget"},
                               {"original": {"content": "The work", "origin_message_ref": ref}}, {})
        expected = "owner" if not metadata and ref else work_kind(metadata)
        assert task["metadata"]["work_kind"] == expected
        if metadata.get("initiator"):
            assert task["metadata"]["initiator"] == "consciousness"
            assert task["metadata"]["consciousness_autonomy"] == "act"


def test_real_schedule_restore_is_checked_at_the_callers_kind(supervisor):
    h = supervisor
    h.modes.update(owner="off", consciousness="full")
    for kind in ("owner", "consciousness"):
        h.ctx.task_metadata["work_kind"] = kind
        for action in ("list", "disable", "delete"):
            assert h.module.check_safety("manage_schedules", {"action": action}, ctx=h.ctx)[0]
        assert not h.calls
        if kind == "consciousness":
            h.replies.append({"status": "DANGEROUS", "reason": "Do not re-arm this work"})
        assert h.module.check_safety("manage_schedules", {"action": "restore"}, ctx=h.ctx)[0] is (kind == "owner")
        assert len(h.calls) == (kind == "consciousness")
    h.replies.append({"status": "SAFE", "reason": "A reviewed restoration"})
    assert h.module.check_safety("manage_schedules", {"action": "restore", "schedule_id": "other"}, ctx=h.ctx)[0]
    assert len(h.calls) == 2


@pytest.mark.parametrize("endpoint", ["safety-mode", "safety-model", "safety-rules"])
def test_browser_owner_controls_use_the_work_cap_on_cyber(monkeypatch, endpoint):
    from ouroboros import browser_policy
    from ouroboros.tools.browser import _runtime_mode_for_browser
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(browser_policy, "runtime_service_kind", lambda *_a: "main")
    request = SimpleNamespace(url=f"http://127.0.0.1/api/owner/{endpoint}", method="POST", post_data='{}')
    for cap, blocked in [("light", True), ("", False)]:
        ctx = SimpleNamespace(task_metadata={"runtime_mode_cap": cap})
        reason = browser_policy.browser_request_block_reason(request, ctx, restricted=False,
                                                             runtime_mode=_runtime_mode_for_browser(ctx))
        assert bool(reason) is blocked
        if blocked:
            assert "BROWSER_OWNER_CONTROL_BLOCKED" in reason

"""Host controls remain controls across the real extension dispatch boundary."""
from datetime import datetime, timedelta, timezone

import pytest

from ouroboros import owner_pause
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests.test_async_invocation_lifetime import extension_case
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import owner, run



@pytest.mark.parametrize(("control", "family"), [("open", "builtin"), ("stop", "extension"), ("deadline", "builtin")])
def test_final_control_keeps_builtin_and_extension_meaning(tmp_path, monkeypatch, family, control):
    controls, effects, owners = [None], [], []
    actual = owner_pause.run_tool_handler

    def at_gate(*args, **kwargs):
        if control == "stop":
            controls[0] = "cancelled"
        elif control == "deadline":
            owners[0].task["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        return actual(*args, **kwargs)

    monkeypatch.setattr(owner_pause, "run_tool_handler", at_gate)

    def invoke(registry, name, args):
        with owner(registry, control=controls) as held:
            owners.append(held)
            if control == "open":
                assert not run(registry, name=name, args=args, timeout=5)["is_error"]
            else:
                with pytest.raises(PhysicalDispatchInterrupted) as caught:
                    run(registry, name=name, args=args, timeout=5)
                assert caught.value.control_reason == ("cancelled" if control == "stop" else "deadline")

    if family == "builtin":
        registry, _, _ = _registry(tmp_path, monkeypatch)
        registry.override_handler("knowledge_read", lambda *_a, **_k: effects.append(True) or "ran")
        invoke(registry, "knowledge_read", {"topic": "fixture"})
    else:
        async def body(_value):
            effects.append(True)
            return "ran"
        with extension_case(tmp_path, monkeypatch, body) as case:
            invoke(case.registry, case.name, {"value": "once"})
    assert effects == ([True] if control == "open" else [])


def test_joined_async_start_refusal_preserves_control_and_closes_coroutine(tmp_path, monkeypatch):
    effects = []
    error = PhysicalDispatchInterrupted("cancelled")
    def refused(*_a, **_k):
        raise error
    monkeypatch.setattr(owner_pause, "submit_async_operation", refused)
    async def body(_value):
        effects.append(True)
        return "must not run"
    with extension_case(tmp_path, monkeypatch, body) as case, owner(case.registry):
        with pytest.raises(PhysicalDispatchInterrupted) as caught:
            run(case.registry, name=case.name, args={"value": "unused"}, timeout=5)
    assert caught.value is error and effects == []

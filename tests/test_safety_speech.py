"""A real verdict permits one deliberate identical speech re-issue, never a hold."""
import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
@pytest.mark.parametrize("status", ["SUSPICIOUS", "DANGEROUS"])
@pytest.mark.parametrize("name", ["place_reply", "connector_send"])
def test_speech_flag_retry_and_changed_subject(supervisor, monkeypatch, mode, status, name):
    h = supervisor
    from ouroboros import tool_effects
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: tool_effects.ToolEffect("message", declared=True))
    args = {"text": "hello", "destination": "room"}
    h.replies.append({"status": status, "reason": "owner has not delegated this"})
    assert h.module.check_safety(name, args, ctx=h.ctx)[0] is False
    assert len(h.calls) == 1
    ok, message = h.module.check_safety(name, dict(reversed(list(args.items()))), ctx=h.ctx)
    assert ok and "despite the check" in message
    assert len(h.calls) == 1
    assert h.events[-1]["type"] == "safety_sent_despite_check"
    assert h.module.check_safety(name, {**args, "text": "changed"}, ctx=h.ctx) == (True, "")
    assert len(h.calls) == 2
    h.ctx.task_id = "different-root"
    h.module.check_safety(name, args, ctx=h.ctx)
    assert len(h.calls) == 3


def test_writes_keep_binding_and_effective_mode(supervisor, monkeypatch):
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    h.ctx.task_metadata.update(work_kind="consciousness", runtime_mode_cap="light")
    h.replies.append({"status": "DANGEROUS", "reason": "unsafe"})
    assert not h.module.check_safety("remote_write", {}, ctx=h.ctx)[0]
    assert "Runtime mode: light" in h.calls[-1]["user_prompt"]
    h.ctx.task_metadata.pop("runtime_mode_cap")
    h.replies.append({"status": "DANGEROUS", "reason": "unsafe"})
    ok, message = h.module.check_safety("remote_write", {}, ctx=h.ctx)
    assert ok and "SAFETY_ADVICE" in message
    assert len(h.calls) == 2


def test_terminal_clears_only_the_root_retry_authority(supervisor, monkeypatch):
    h = supervisor
    from ouroboros.task_results import write_task_result
    h.replies.append({"status": "DANGEROUS", "reason": "flag"})
    assert not h.module.check_safety("place_reply", {"text": "hi"}, ctx=h.ctx)[0]
    write_task_result(h.ctx.drive_root, "child", status="completed", root_task_id="work", parent_task_id="work")
    assert "work" in h.module._PRIOR_VERDICTS
    write_task_result(h.ctx.drive_root, "work", status="completed", root_task_id="work")
    assert "work" not in h.module._PRIOR_VERDICTS
    assert h.module.check_safety("place_reply", {"text": "hi"}, ctx=h.ctx)[0]
    assert len(h.calls) == 2


def test_repeated_write_keeps_both_earlier_verdicts(supervisor):
    h = supervisor
    h.replies.extend([{"status": "DANGEROUS", "reason": "first refusal"}, {"status": "SAFE", "reason": "later approval"}])
    assert not h.module.check_safety("remote_write", {}, ctx=h.ctx)[0]
    assert h.module.check_safety("remote_write", {}, ctx=h.ctx)[0]
    h.module.check_safety("remote_write", {"new": 1}, ctx=h.ctx)
    assert "first refusal" in h.calls[-1]["user_prompt"] and "later approval" in h.calls[-1]["user_prompt"]

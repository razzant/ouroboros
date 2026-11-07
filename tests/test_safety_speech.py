"""A real verdict permits one deliberate identical speech re-issue, never a hold."""
import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture


@pytest.mark.parametrize(("status", "sent"), [("SUSPICIOUS", True), ("DANGEROUS", False)])
def test_speech_follows_the_final_assessment(supervisor, monkeypatch, status, sent):
    """SUSPICIOUS speech sends at once with its warning, as for any other call; DANGEROUS returns it unsent."""
    h = supervisor
    from ouroboros import tool_effects
    monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: tool_effects.ToolEffect("message", declared=True))
    h.replies.append({"status": status, "reason": "the audience is wider than the note allows"})
    ok, message = h.module.check_safety("connector_send", {"text": "hello", "to": "room"}, ctx=h.ctx)
    assert ok is sent and "the audience is wider than the note allows" in message
    assert message.startswith("⚠️ SAFETY_WARNING") is sent and ("Message not sent" in message) is not sent


@pytest.mark.parametrize(("name", "status", "mode"), [
    ("place_reply", "DANGEROUS", "advanced"),
    ("connector_send", "DANGEROUS", "cyber_pro"),
])
def test_speech_flag_retry_and_changed_subject(supervisor, monkeypatch, mode, status, name):
    h = supervisor
    from ouroboros import tool_effects
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: tool_effects.ToolEffect("message", declared=True))
    args = {"text": "hello", "destination": "room", "request_id": "r-1"}
    h.replies.append({"status": status, "reason": "owner has not delegated this"})
    ok, message = h.module.check_safety(name, args, ctx=h.ctx)
    assert ok is False and "repeat the identical call, request_id included, in a later step" in message
    assert len(h.calls) == 1
    # The same model call re-presenting the bytes (the host republishing a held reply) decides nothing new.
    ok, message = h.module.check_safety(name, dict(reversed(list(args.items()))), ctx=h.ctx)
    assert not ok and "flagged in this same step" in message and len(h.calls) == 1
    assert all(event["type"] != "safety_sent_despite_check" for event in h.events)
    h.later()
    ok, message = h.module.check_safety(name, dict(reversed(list(args.items()))), ctx=h.ctx)
    assert ok and "despite the check" in message
    assert len(h.calls) == 1
    assert h.events[-1]["type"] == "safety_sent_despite_check"
    assert (h.events[-1]["flagged_in"], h.events[-1]["sent_by"]) == ("llm-round-1", "llm-round-2")
    assert h.module.check_safety(name, {**args, "text": "changed"}, ctx=h.ctx) == (True, "")
    assert len(h.calls) == 2
    # request_id is part of the message's identity: the same words under a new id are checked again (Joi/Eva, 05.11).
    h.replies.append({"status": status, "reason": "owner has not delegated this"})
    assert h.module.check_safety(name, {**args, "request_id": "r-2"}, ctx=h.ctx)[0] is False
    assert len(h.calls) == 3
    h.ctx.task_id = "different-root"
    h.module.check_safety(name, args, ctx=h.ctx)
    assert len(h.calls) == 4


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

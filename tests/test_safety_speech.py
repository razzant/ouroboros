"""A flag returns words once; who may send them after is read when they are sent, never from a hold."""
import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture


@pytest.mark.parametrize(("status", "exit_words"), [("SUSPICIOUS", "repeat the identical call"),
                                                   ("DANGEROUS", "wait for your human")])
def test_flagged_words_return_unsent_once_while_actions_keep_their_levels(supervisor, monkeypatch, status, exit_words):
    """Words flagged at either level return unsent before anything is sent, in an ordinary mode too: SUSPICIOUS
    leaves the re-issue to the mind, DANGEROUS waits for my human. The same verdicts on an action keep their levels:
    SUSPICIOUS runs it with the warning, DANGEROUS refuses it."""
    h = supervisor
    from ouroboros import tool_effects
    monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: tool_effects.ToolEffect("message", declared=True))
    h.replies.append({"status": status, "reason": "the audience is wider than the note allows"})
    ok, message = h.module.check_safety("connector_send", {"text": "hello", "to": "room"}, ctx=h.ctx)
    assert not ok and "Message not sent" in message and "the audience is wider than the note allows" in message
    assert exit_words in message
    assert message.startswith("⚠️ SAFETY_WARNING" if status == "SUSPICIOUS" else "⚠️ SAFETY_VIOLATION")
    monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: tool_effects.BUILTIN)
    h.replies.append({"status": status, "reason": "odd, with legitimate uses"})
    ok, message = h.module.check_safety("remote_write", {"path": "x"}, ctx=h.ctx)
    assert ok is (status == "SUSPICIOUS") and "Message not sent" not in message


@pytest.mark.parametrize(("name", "status", "mode"), [
    ("place_reply", "SUSPICIOUS", "advanced"),
    ("connector_send", "DANGEROUS", "cyber_pro"),
])
def test_speech_flag_retry_and_changed_subject(supervisor, monkeypatch, mode, status, name):
    """Where the mind decides (SUSPICIOUS anywhere, DANGEROUS in Cyber Pro), an identical re-issue from a later model
    call sends the words; the same call re-presenting them decides nothing, and changed words or a new request_id
    are a new subject checked again."""
    h = supervisor
    from ouroboros import tool_effects
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: tool_effects.ToolEffect("message", declared=True))
    args = {"text": "hello", "destination": "room", "request_id": "r-1"}
    h.replies.append({"status": status, "reason": "owner has not delegated this"})
    ok, message = h.module.check_safety(name, args, ctx=h.ctx)
    assert ok is False and "repeat the identical call (its request_id too, if it has one) in a later step" in message
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


def test_dangerous_words_wait_for_my_human_outside_cyber_pro(supervisor, monkeypatch):
    """Outside Cyber Pro words flagged DANGEROUS wait for my human, as a dangerous action stops there: neither a later
    identical re-issue nor the same words under a new request_id sends them or buys a fresh check, changed words are
    checked again, and my human's answered card holding the exact words and destination sends them. The mode is read
    when they are sent: the mind sends an ordinary mode's flag once the work runs in Cyber Pro, and a Cyber Pro flag
    is not spent by work of the same root capped to an ordinary mode."""
    from ouroboros import place_guard
    h = supervisor
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *a, **kw: {
        "places": [{"place_id": "p", "place_label": "Telegram · FusionBrain"}], "targets": {"chat_id": ["-100"]}})
    text = "Atlas passed 70 and 95."
    reply = {"place_id": "p", "text": text, "request_id": "r-1"}
    h.replies.append({"status": "DANGEROUS", "reason": "the note keeps this circle apart", "quote": text})
    ok, message = h.module.check_safety("place_reply", reply, ctx=h.ctx)
    assert not ok and "wait for your human" in message and "repeat the identical call" not in message
    assert "these exact words on their own lines, unprefixed, and the destination Telegram · FusionBrain" in message
    for attempt in (reply, {**reply, "request_id": "r-2"}):
        h.later()
        ok, message = h.module.check_safety("place_reply", attempt, ctx=h.ctx)
        assert not ok and "wait for your human" in message
    assert len(h.calls) == 1 and not any(event["type"] == "safety_sent_despite_check" for event in h.events)
    assert h.module.check_safety("place_reply", {**reply, "text": "Atlas is on track."}, ctx=h.ctx) == (True, "")
    assert len(h.calls) == 2
    h.ctx._owner_directives = [{"source": "owner_quiz_answer", "content": "1. Send it", "quiz_id": "q9", "reply_to": {
        "text": f"Send this to Telegram · FusionBrain?\n{text}\n1. Send it\n2. Not now"}}]
    ok, message = h.module.check_safety("place_reply", reply, ctx=h.ctx)
    assert ok and "(q9)" in message and len(h.calls) == 2
    assert (h.events[-1]["type"], h.events[-1]["quiz_id"]) == ("safety_sent_despite_check", "q9")
    h.ctx._owner_directives = []
    eva = {**reply, "text": "Eva starts on Monday.", "request_id": "r-4"}
    h.replies.append({"status": "DANGEROUS", "reason": "the note keeps this circle apart"})
    assert not h.module.check_safety("place_reply", eva, ctx=h.ctx)[0]
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    h.later()
    ok, message = h.module.check_safety("place_reply", eva, ctx=h.ctx)
    assert ok and "despite the check" in message
    joi = {**reply, "text": "Joi knows Eva.", "request_id": "r-5"}
    h.replies.append({"status": "DANGEROUS", "reason": "the note keeps this circle apart"})
    ok, message = h.module.check_safety("place_reply", joi, ctx=h.ctx)
    assert not ok and "repeat the identical call" in message
    h.ctx.task_metadata["runtime_mode_cap"] = "advanced"
    h.later()
    ok, message = h.module.check_safety("place_reply", joi, ctx=h.ctx)
    assert not ok and "wait for your human" in message and len(h.calls) == 4


def test_a_verdict_marks_speech_on_any_checked_call(supervisor, monkeypatch):
    """A built-in without declarations (github_write) gets speech semantics only from the verdict: flagged as speech
    in Cyber Pro it returns unsent at first issue and sends on a later decision; the same verdict without the mark
    stays advice there. In an ordinary mode marked DANGEROUS words wait for my human like declared ones: a re-issue
    is neither sent nor judged again, and my human's card names the exact call."""
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "advanced")
    secret = {"method": "comment", "repo": "razzant/ouroboros", "number": 1, "body": "token sk-live-1"}
    h.replies.append({"status": "DANGEROUS", "reason": "a credential to a public thread", "speech": True,
                      "quote": "sk-live-1"})
    exact = h.module._exact_call("github_write", secret)
    for attempt in range(2):
        ok, message = h.module.check_safety("github_write", secret, ctx=h.ctx)
        assert not ok and "wait for your human" in message and "repeat the identical call" not in message
        assert f"exact call on its own line of the card: {exact}\n" in message
        assert ('"sk-live-1"' in message) is (attempt == 0)
        h.later()
    assert len(h.calls) == 1, "the identical re-issue waited without a new check"
    assert h.module.check_safety("github_write", {**secret, "number": 3}, ctx=h.ctx)[0] and len(h.calls) == 2
    h.ctx._owner_directives = [{"source": "owner_quiz_answer", "content": "1. Post it", "quiz_id": "q4",
                                "reply_to": {"text": f"Post exactly this?\n{exact}\n1. Post it"}}]
    ok, message = h.module.check_safety("github_write", secret, ctx=h.ctx)
    assert ok and "(q4)" in message and h.events[-1]["quiz_id"] == "q4"
    h.ctx._owner_directives = []
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    comment = {"method": "comment", "repo": "razzant/ouroboros", "number": 2805, "body": "Taking it up."}
    h.replies.append({"status": "DANGEROUS", "reason": "the place note asks for a quiz first", "speech": True,
                      "quote": "Taking it up."})
    ok, message = h.module.check_safety("github_write", comment, ctx=h.ctx)
    assert not ok and "Message not sent" in message
    h.later()
    ok, message = h.module.check_safety("github_write", comment, ctx=h.ctx)
    assert ok and "despite the check" in message and len(h.calls) == 3
    h.replies.append({"status": "DANGEROUS", "reason": "the place note asks for a quiz first", "speech": False})
    ok, message = h.module.check_safety("github_write", {**comment, "body": "Other words."}, ctx=h.ctx)
    assert ok and "Cyber Pro leaves the decision to Ouroboros" in message


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

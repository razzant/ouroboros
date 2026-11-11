"""A flagged call returns once; who may let it through after is read at its next issue, never from a hold."""
import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
from ouroboros.tool_effects import effect_of as published_effect
from ouroboros.place_guard import call_place_facts as published_destinations
supervisor = supervisor_fixture

MIND = "issuing the identical call in a later step"  # the mind's own exit, named only where it is the exit
GATE_HEAD = "These words to this destination were read DANGEROUS in this work:"  # answered for another call
WORDS_WAIT = ("Changed words or another destination are read again; the same words there keep waiting, whatever the "
              "tool or request_id.")


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
@pytest.mark.parametrize("status", ["SUSPICIOUS", "DANGEROUS"])
@pytest.mark.parametrize("name", ["connector_send", "remote_write"])
def test_every_flagged_call_returns_once_before_any_effect(supervisor, monkeypatch, name, status, mode):
    """Words or an action, flagged at either level, return not run in every mode, with the reason. The
    return names its exit at once: the mind for SUSPICIOUS and for DANGEROUS in Cyber Pro (never my human there),
    else my human's card. A SAFE reading of the same call runs it with no note."""
    h = supervisor
    from ouroboros import tool_effects
    words = tool_effects.ToolEffect("message", declared=True) if name == "connector_send" else tool_effects.BUILTIN
    monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: words)
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    h.replies.append({"status": status, "reason": "the audience is wider than the note allows"})
    ok, message = h.module.check_safety(name, {"text": "hello", "to": "room"}, ctx=h.ctx)
    assert not ok and message.startswith("⚠️ SAFETY_RETURNED: not run; nothing was sent or changed.")
    assert f"read this call as {status}" in message and "the audience is wider than the note allows" in message
    mind = status == "SUSPICIOUS" or mode == "cyber_pro"
    assert (MIND in message) is mind and ("waits for your human" in message) is not mind
    assert ("your human" in message) is not mind, "a Cyber Pro or SUSPICIOUS return names no card"
    assert h.module.check_safety(name, {"text": "other", "to": "room"}, ctx=h.ctx) == (True, "")


@pytest.mark.parametrize(("name", "status", "mode"), [
    ("place_reply", "SUSPICIOUS", "advanced"),
    ("connector_send", "DANGEROUS", "cyber_pro"),
    ("remote_write", "SUSPICIOUS", "advanced"),
    ("remote_write", "DANGEROUS", "cyber_pro"),
])
def test_flag_retry_and_changed_subject(supervisor, monkeypatch, mode, status, name):
    """Where the mind decides (SUSPICIOUS anywhere, DANGEROUS in Cyber Pro), the identical call from a later model
    call is let through without a new reading; the step that received the return re-presenting it decides nothing,
    and a changed call or a new request_id is read again. Actions follow the same exits as words."""
    h = supervisor
    from ouroboros import tool_effects
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    effect = tool_effects.BUILTIN if name == "remote_write" else tool_effects.ToolEffect("message", declared=True)
    monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: effect)
    args = {"text": "hello", "destination": "room", "request_id": "r-1"}
    h.replies.append({"status": status, "reason": "owner has not delegated this"})
    ok, message = h.module.check_safety(name, args, ctx=h.ctx)
    assert ok is False and MIND in message
    assert len(h.calls) == 1
    # The same model call re-presenting the bytes (the host republishing a held reply) decides nothing new.
    ok, message = h.module.check_safety(name, dict(reversed(list(args.items()))), ctx=h.ctx)
    assert not ok
    assert message.startswith("⚠️ SAFETY_RETURNED: not run. This exact call was returned earlier in this same step")
    assert MIND in message and len(h.calls) == 1
    assert all(event["type"] != "safety_sent_despite_check" for event in h.events)
    h.later()
    ok, message = h.module.check_safety(name, dict(reversed(list(args.items()))), ctx=h.ctx)
    assert ok and message.startswith("⚠️ SAFETY_NOTE: let through on your later identical call, without a new reading")
    assert f"had read it as {status}: " in message and "owner has not delegated this" in message
    assert len(h.calls) == 1
    assert h.events[-1]["type"] == "safety_sent_despite_check"
    assert (h.events[-1]["flagged_in"], h.events[-1]["sent_by"]) == ("llm-round-1", "llm-round-2")
    assert h.module.check_safety(name, {**args, "text": "changed"}, ctx=h.ctx) == (True, "")
    assert len(h.calls) == 2
    # request_id is part of the call's identity: the same words under a new id are read again.
    h.replies.append({"status": status, "reason": "owner has not delegated this"})
    assert h.module.check_safety(name, {**args, "request_id": "r-2"}, ctx=h.ctx)[0] is False
    assert len(h.calls) == 3
    h.ctx.task_id = "different-root"
    h.module.check_safety(name, args, ctx=h.ctx)
    assert len(h.calls) == 4


@pytest.mark.parametrize(("status", "mode"), [("SUSPICIOUS", "advanced"), ("DANGEROUS", "cyber_pro"),
                                              ("DANGEROUS", "advanced")])
def test_a_step_the_host_cannot_name_is_a_later_step(supervisor, monkeypatch, status, mode):
    """A context with no step stamp gives up the same-step guard, never the exits: its second issue is released, by
    the mind (SUSPICIOUS; DANGEROUS in Cyber Pro) or, in an ordinary mode, by my human's card."""
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    del h.ctx._current_llm_call_meta
    command = {"cmd": "rm -rf ~/scratch"}
    h.replies.append({"status": status, "reason": "deletes a folder"})
    assert not h.module.check_safety("run_command", command, ctx=h.ctx)[0]
    if mode == "advanced" and status == "DANGEROUS":
        h.ctx._owner_directives = [{"source": "owner_quiz_answer", "content": "1. Run it", "quiz_id": "q1",
                                    "reply_to": {"address": "quiz q1 of task work",
                                                 "text": f"Run this?\n{h.module._exact_call('run_command', command)}"}}]
    ok, message = h.module.check_safety("run_command", command, ctx=h.ctx)
    assert ok and message.startswith("⚠️ SAFETY_NOTE: let through on your") and len(h.calls) == 1


def test_a_live_tool_round_is_the_step_a_return_is_read_against(supervisor, monkeypatch):
    """A live call's tool round stamps its ``round_id`` and no model call: the same round re-presenting a returned
    call decides nothing, and the next round's identical call is the mind's release."""
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    h.ctx._current_llm_call_meta = {"round_id": "work:call-tools:1"}
    h.replies.append({"status": "SUSPICIOUS", "reason": "a style note"})
    assert not h.module.check_safety("remote_write", {"x": 1}, ctx=h.ctx)[0]
    ok, message = h.module.check_safety("remote_write", {"x": 1}, ctx=h.ctx)
    assert not ok and "returned earlier in this same step" in message
    h.ctx._current_llm_call_meta = {"round_id": "work:call-tools:2"}
    ok, message = h.module.check_safety("remote_write", {"x": 1}, ctx=h.ctx)
    assert ok and "let through on your later identical call" in message and len(h.calls) == 1
    assert (h.events[-1]["flagged_in"], h.events[-1]["sent_by"]) == ("work:call-tools:1", "work:call-tools:2")


def test_dangerous_words_wait_for_my_human_outside_cyber_pro(supervisor, monkeypatch):
    """Outside Cyber Pro words read DANGEROUS wait for my human: neither a later identical call nor the same words
    under a new request_id lets them through or buys a fresh reading, changed words are read again, and my human's
    answered card holding the exact words and destination lets them through. The mode is read at the issue: the mind
    releases an ordinary mode's flag once the work runs in Cyber Pro, and a Cyber Pro flag is not spent by work of the
    same root capped to an ordinary mode."""
    from ouroboros import place_guard
    h = supervisor
    # As production resolves them: the finish names its place alone; a send names the place's own conversation.
    monkeypatch.setattr(place_guard, "call_place_facts", lambda ctx, name, args, **kw: {
        "places": [{"place_id": "p", "place_label": "Telegram · FusionBrain", "conversation_id": "-100"}],
        "targets": {} if name == "place_reply" else {"chat_id": [args["chat_id"]]}})
    text = "Atlas passed 70 and 95."
    reply = {"place_id": "p", "text": text, "request_id": "r-1"}
    h.replies.append({"status": "DANGEROUS", "reason": "the note keeps this circle apart", "quote": text})
    ok, message = h.module.check_safety("place_reply", reply, ctx=h.ctx)
    assert not ok and "waits for your human" in message and MIND not in message
    assert f"Flagged words: \"{text}\"" in message
    assert "a card holding these exact words on their own lines and the destination Telegram · FusionBrain." in message
    for attempt, head in ((reply, "The Safety Supervisor read this call as DANGEROUS:"),
                          ({**reply, "request_id": "r-2"}, GATE_HEAD)):
        h.later()
        ok, message = h.module.check_safety("place_reply", attempt, ctx=h.ctx)
        assert not ok and "waits for your human" in message and head in message and WORDS_WAIT in message
        assert ("read this call" in message) is (attempt is reply) and "A changed call is read again" not in message
    assert len(h.calls) == 1 and not any(event["type"] == "safety_sent_despite_check" for event in h.events)
    assert h.module.check_safety("place_reply", {**reply, "text": "Atlas is on track."}, ctx=h.ctx) == (True, "")
    assert len(h.calls) == 2
    # The same words to the same destination through another declared door wait too, unread.
    from ouroboros import tool_effects
    monkeypatch.setattr(tool_effects, "effect_of", lambda name, **kw: tool_effects.ToolEffect("message", declared=True)
                        if name == "connector_send" else tool_effects.BUILTIN)
    ok, message = h.module.check_safety("connector_send", {"text": text, "chat_id": "-100"}, ctx=h.ctx)
    assert not ok and "waits for your human" in message and len(h.calls) == 2
    assert GATE_HEAD in message and "read this call" not in message and WORDS_WAIT in message
    h.ctx._owner_directives = [{"source": "owner_quiz_answer", "content": "1. Send it", "quiz_id": "q9", "reply_to": {
        "address": "quiz q9 of task work", "text": f"Send this to Telegram · FusionBrain?\n{text}\n1. Send it"}}]
    ok, message = h.module.check_safety("place_reply", reply, ctx=h.ctx)
    assert ok and message.startswith("⚠️ SAFETY_NOTE: let through on your human's answer (q9)") and len(h.calls) == 2
    assert "a card holding this exact text and destination, not what they answered" in message
    assert (h.events[-1]["type"], h.events[-1]["quiz_id"]) == ("safety_sent_despite_check", "q9")
    assert h.events[-1]["verdicts"][0]["status"] == "DANGEROUS"  # the release keeps the verdict it overrode
    ok, message = h.module.check_safety("place_reply", {**reply, "request_id": "r-7"}, ctx=h.ctx)  # the gated words too
    assert ok and h.events[-1]["type"] == "safety_sent_despite_check" and h.events[-1]["verdicts"] and len(h.calls) == 2
    h.ctx._owner_directives = []
    eva = {**reply, "text": "Eva starts on Monday.", "request_id": "r-4"}
    h.replies.append({"status": "DANGEROUS", "reason": "the note keeps this circle apart"})
    assert not h.module.check_safety("place_reply", eva, ctx=h.ctx)[0]
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    h.later()
    ok, message = h.module.check_safety("place_reply", eva, ctx=h.ctx)
    assert ok and "let through on your later identical call" in message
    joi = {**reply, "text": "Joi knows Eva.", "request_id": "r-5"}
    h.replies.append({"status": "DANGEROUS", "reason": "the note keeps this circle apart"})
    ok, message = h.module.check_safety("place_reply", joi, ctx=h.ctx)
    assert not ok and MIND in message
    h.ctx.task_metadata["runtime_mode_cap"] = "advanced"
    h.later()
    ok, message = h.module.check_safety("place_reply", joi, ctx=h.ctx)
    assert not ok and "waits for your human" in message and len(h.calls) == 4


@pytest.mark.parametrize("release", ["owner_card", "cyber_pro", "owner_closed", "cyber_closed"])
def test_later_dangerous_words_hold_older_suspicious_retry(supervisor, monkeypatch, release):
    """A DANGEROUS word/recipient gate outranks an older exact-call SUSPICIOUS retry, not its owner's release; where
    the card exit is closed my human's answer earns one new reading instead."""
    from ouroboros import tool_effects, place_guard, extension_registry_state
    from ouroboros.extension_surface_names import extension_surface_name
    h = supervisor
    monkeypatch.setattr(tool_effects, "effect_of", published_effect)
    monkeypatch.setattr(place_guard, "call_place_facts", published_destinations)
    name = extension_surface_name("example", "send")
    descriptor = {"skill": "example", "extension_generation": "fixture",
                  extension_registry_state.REVIEWED_DECLARATION: {"extension_generation": "fixture",
                      "provides_place": False, "tool": {"name": "send", "effect": "message", "target": "recipient"}}}
    text = "Fixture report is ready."
    older = {"text": text, "recipient": "fixture-room", "request_id": "first"}
    def check(args):
        return h.module.check_safety(name, args, ctx=h.ctx, ext_tool=descriptor)
    h.replies.append({"status": "SUSPICIOUS", "reason": "confirm the audience"})
    assert not check(older)[0]
    h.later()
    h.replies.append({"status": "DANGEROUS", "reason": "the audience is outside this report's circle"})
    assert not check({**older, "request_id": "second"})[0]
    h.later()
    allowed, message = check(older)
    assert not allowed, "the older exact-call flag must not bypass the newer word/recipient gate"
    assert "waits for your human" in message and MIND not in message
    assert len(h.calls) == 2 and not any(e["type"] == "safety_sent_despite_check" for e in h.events)
    if release in {"owner_card", "owner_closed"}:
        h.ctx._owner_directives = [{"source": "owner_quiz_answer", "content": "Send it", "quiz_id": "q-later",
            "reply_to": {"address": "quiz q-later of task work",
                         "text": f"Send this to fixture-room?\n{text}\n1. Send it"}}]
    else:
        monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    if release.endswith("_closed"):
        from ouroboros.deployment_policy import RELEASE_AFTER_DANGEROUS, REPEAT_AFTER_DANGEROUS
        shut = RELEASE_AFTER_DANGEROUS if release == "owner_closed" else REPEAT_AFTER_DANGEROUS
        monkeypatch.setattr(h.module, "safety_exit_open", lambda exit_: exit_ != shut)
        h.replies.append({"status": "DANGEROUS", "reason": "the audience is still outside the circle"})
    h.later()
    allowed, message = check(older)
    if release.endswith("_closed"):
        assert not allowed and len(h.calls) == 2 + (release == "owner_closed"), (
            "an older flag cannot reopen a closed deployment exit; my human's answer earns one new reading")
        return
    assert allowed and len(h.calls) == 2
    assert h.events[-1]["type"] == "safety_sent_despite_check"
    assert ("(q-later)" if release == "owner_card" else "let through on your later identical call") in message


@pytest.mark.parametrize("second", ["SAFE", "DANGEROUS"])
@pytest.mark.parametrize("gate", [False, True])
def test_a_later_word_of_my_human_earns_a_waiting_call_one_new_reading(supervisor, monkeypatch, gate, second):
    """A DANGEROUS call that waits for my human is answered by its kept verdict however often it is issued, until my
    human writes in this work: then one new reading. SAFE lets it through and the old verdict answers it no more;
    DANGEROUS waits again, unread, until their next word. The same words under a new request_id meet the same gate."""
    from ouroboros import place_guard
    h = supervisor
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *a, **kw: {
        "places": [{"place_id": "p", "place_label": "Telegram · Team", "conversation_id": "-100"}], "targets": {}})
    reply = {"place_id": "p", "text": "Atlas passed 70 and 95.", "request_id": "r-1"}
    issue = lambda n: {**reply, "request_id": f"r-{n}"} if gate else reply
    word = lambda n: {"source": "owner_mailbox", "content": f"Note {n}.", "msg_id": f"chat 1 / m-{n}"}

    def check(n):
        h.later()
        return h.module.check_safety("place_reply", issue(n), ctx=h.ctx)
    h.replies.append({"status": "DANGEROUS", "reason": "the note keeps this circle apart"})
    assert not h.module.check_safety("place_reply", reply, ctx=h.ctx)[0]
    for n in (2, 3):
        ok, message = check(n)
        assert not ok and "A later word of your human in this work earns it one new reading" in message
    assert len(h.calls) == 1, "no new word of my human: the kept verdict answers"
    h.ctx._owner_directives = [word(1)]
    h.replies.append({"status": second, "reason": "they asked for it" if second == "SAFE" else "still apart"})
    assert check(4)[0] is (second == "SAFE") and len(h.calls) == 2, "exactly one new reading"
    h.replies.append({"status": "DANGEROUS", "reason": "read afresh"})
    assert not check(5)[0] and len(h.calls) == 2 + (second == "SAFE"), "SAFE: the old verdict answers no more"
    if second == "DANGEROUS":
        assert not check(6)[0] and len(h.calls) == 2, "waits again until another word"
        h.ctx._owner_directives.append(word(2))
        assert not check(7)[0] and len(h.calls) == 3


def test_an_unchecked_mode_meets_no_flag_wait_or_card(supervisor, monkeypatch):
    """Safety off for this work kind means unchecked: a DANGEROUS wait recorded before the switch and a matching card
    both give way to the disclosed skip, so nothing waits and no Safety note rides on the words; switched back on,
    the wait binds again."""
    h = supervisor
    reply = {"text": "Atlas passed 70 and 95.", "request_id": "r-1"}
    h.replies.append({"status": "DANGEROUS", "reason": "the note keeps this circle apart"})
    assert not h.module.check_safety("place_reply", reply, ctx=h.ctx)[0]
    h.ctx._owner_directives = [{"source": "owner_quiz_answer", "quiz_id": "q1", "content": "1. Send", "reply_to": {
        "address": "quiz q1 of task work", "text": f"Send exactly this?\n{h.module._exact_call('place_reply', reply)}"}}]
    h.modes["owner"] = "off"
    h.later()
    assert h.module.check_safety("place_reply", reply, ctx=h.ctx) == (True, "") and h.events[-1]["type"] == "skip"
    h.modes["owner"], h.ctx._owner_directives = "full", []
    ok, message = h.module.check_safety("place_reply", reply, ctx=h.ctx)
    assert not ok and "waits for your human" in message and len(h.calls) == 1


def test_dangerous_action_in_an_ordinary_mode_waits_for_the_card(supervisor, monkeypatch):
    """New: outside Cyber Pro a DANGEROUS call that declares no words (a GitHub comment, a command) waits for my
    human's card on the exact call, which it never had before: a later identical call is neither let through nor read
    again, a changed call is read again, and the card that holds the exact call lets it through. A verdict's own words
    about speech mark nothing: the same reading in Cyber Pro returns once and the mind's later call lets it through."""
    h = supervisor
    secret = {"method": "comment", "repo": "razzant/ouroboros", "number": 1, "body": "token sk-live-1"}
    h.replies.append({"status": "DANGEROUS", "reason": "a credential to a public thread", "speech": True,
                      "quote": "sk-live-1"})
    exact = h.module._exact_call("github_write", secret)
    for attempt in range(2):
        ok, message = h.module.check_safety("github_write", secret, ctx=h.ctx)
        assert not ok and "waits for your human" in message and MIND not in message
        assert f"a card holding this exact call on its own line: {exact}." in message
        assert '"sk-live-1"' in message, "the reading's quote rides on every return of the call"
        h.later()
    assert len(h.calls) == 1, "the identical later call waited without a new reading"
    assert h.module.check_safety("github_write", {**secret, "number": 3}, ctx=h.ctx)[0] and len(h.calls) == 2
    h.ctx._owner_directives = [{"source": "owner_quiz_answer", "content": "1. Post it", "quiz_id": "q4", "reply_to": {
        "address": "quiz q4 of task work", "text": f"Post exactly this?\n{exact}\n1. Post it"}}]
    ok, message = h.module.check_safety("github_write", secret, ctx=h.ctx)
    assert ok and "(q4)" in message and "this exact call, not what they answered" in message
    assert h.events[-1]["quiz_id"] == "q4"
    h.ctx._owner_directives = []
    command = {"cmd": ["rm", "-rf", "build"]}
    h.replies.append({"status": "DANGEROUS", "reason": "outside the task"})
    for _ in range(2):
        ok, message = h.module.check_safety("remote_write", command, ctx=h.ctx)
        assert not ok and "waits for your human" in message
        h.later()
    assert len(h.calls) == 3
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    comment = {"method": "comment", "repo": "razzant/ouroboros", "number": 2805, "body": "Taking it up."}
    h.replies.append({"status": "DANGEROUS", "reason": "the place note asks for a quiz first", "speech": False})
    ok, message = h.module.check_safety("github_write", comment, ctx=h.ctx)
    assert not ok and MIND in message and "your human" not in message
    h.later()
    ok, message = h.module.check_safety("github_write", comment, ctx=h.ctx)
    assert ok and "let through on your later identical call" in message and len(h.calls) == 4


def test_writes_keep_binding_and_effective_mode(supervisor, monkeypatch):
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    h.ctx.task_metadata.update(work_kind="consciousness", runtime_mode_cap="light")
    h.replies.append({"status": "DANGEROUS", "reason": "unsafe"})
    ok, message = h.module.check_safety("remote_write", {}, ctx=h.ctx)
    assert not ok and "waits for your human" in message
    assert "Runtime mode: light" in h.calls[-1]["user_prompt"]
    h.ctx.task_metadata.pop("runtime_mode_cap")
    h.replies.append({"status": "DANGEROUS", "reason": "unsafe"})
    ok, message = h.module.check_safety("remote_write", {"path": "other"}, ctx=h.ctx)
    assert not ok and MIND in message and "Runtime mode: cyber_pro" in h.calls[-1]["user_prompt"]
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

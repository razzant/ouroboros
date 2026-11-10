"""A deployment's ``safety.allowed_exits`` at Safety's own exits (``safety.check_safety``), every case beside personal OSS.

Three exits stay distinct: an agent's DANGEROUS repeat in Cyber Pro, a human's answered card after DANGEROUS,
and a human's card without a verdict. Explicit lists do not inherit another exit. No-policy behavior stays
upstream; the organization's own checker releases nothing unjudged by default. A changed argument,
request_id included, is a new call: Cyber Pro checks it again, an ordinary mode keeps declared words to the same
destination waiting; actions that are no words keep their mode's rule.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import tool_effects as _tool_effects
from tests._safety_fixtures import supervisor as supervisor_fixture
from tests.test_tool_effects import published_tool as _published_tool

supervisor, published_tool = supervisor_fixture, _published_tool
REAL_EFFECT_OF = _tool_effects.effect_of  # bound before the fixture stubs it
CHECKER = "claudexor::codex=org-checker"  # a subscription route: asked without a stored credential
REPEAT, HUMAN, RELEASE = "repeat_after_dangerous", "release_after_dangerous", "release_unjudged"
CASES = {  # the organization's checker named, and the file's allowed_exits (None: not named)
    "unmanaged": (False, None), "unnamed": (True, None), "unnamed, own checker": (False, None),
    "none": (True, []), "none, own checker": (False, []), "repeat": (True, [REPEAT]),
    "release": (True, [RELEASE]), "both": (True, [RELEASE, REPEAT]),
    "human": (True, [HUMAN]), "all": (True, [REPEAT, HUMAN, RELEASE]),
}
SAID = {REPEAT: "repeat the identical call", RELEASE: "ask your human privately"}
CLOSED = {REPEAT: "lets no repeat send it", RELEASE: "lets no answer of your human release a call unjudged"}


@pytest.fixture
def exits(tmp_path, monkeypatch):
    """``exits(case)`` names that case's policy file as a launch does (none unmanaged) and answers, from the
    contract above, which exit stays open there."""
    from ouroboros import deployment_policy

    def name(case: str) -> dict:
        checker, named = CASES[case]
        monkeypatch.delenv(deployment_policy.GENERATION_ENV, raising=False)
        if case == "unmanaged":
            monkeypatch.delenv(deployment_policy.POLICY_ENV, raising=False)
        else:
            path = tmp_path / "admin" / "deployment-policy.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"version": 1, "safety": {
                **({"checker": {"model": CHECKER}} if checker else {}),
                **({"allowed_exits": named} if named is not None else {})}}), encoding="utf-8")
            monkeypatch.setenv(deployment_policy.POLICY_ENV, str(path))
        deployment_policy.reset_for_tests()
        return {exit_: case == "unmanaged" or (exit_ in named if named is not None else not (checker and exit_ == RELEASE))
                for exit_ in (REPEAT, HUMAN, RELEASE)}

    yield name
    deployment_policy.reset_for_tests()


def _subject(h, name, args, ctx=None):
    """What my human's card must hold for this call (``safety._exit_subject``), read as the check reads it."""
    ctx = ctx or h.ctx
    effect = _tool_effects.effect_of(name, ctx=ctx, is_mcp=name.startswith("mcp_"), args=args)
    return h.module._exit_subject(ctx, name, args, effect)


def _answered(h, quiz, name, args, ctx=None):
    """My human answered a card holding exactly this call's words and every destination."""
    words, destinations, _exact, _labels = _subject(h, name, args, ctx)
    card = f"Send this?\n{words}\n" + "".join(f"To {destination}\n" for destination in destinations) + "1. Yes"
    (ctx or h.ctx)._owner_directives = [{"source": "owner_quiz_answer", "content": "1. Yes", "quiz_id": quiz,
                                         "reply_to": {"address": f"quiz {quiz} of task work", "text": card}}]


def _registry(tmp_path, monkeypatch, *, declared=True):
    """The real first-party dispatch over a message sink in the root ``work``: ``(registry, its context, what the
    handler sent)``. ``declared``: the sink declares a message effect; else only a verdict can mark its speech."""
    from ouroboros import place_guard
    from ouroboros.tools.registry import ToolEntry, ToolRegistry

    if declared:
        monkeypatch.setattr(_tool_effects, "effect_of", lambda *a, **kw: _tool_effects.ToolEffect("message", declared=True))
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *a, **kw: {"places": [{"place_label": "Team room"}]})
    sent: list = []
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.register(ToolEntry("fixture_send", {
        "name": "fixture_send", "description": "fixture",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }, lambda _ctx, text: sent.append(text) or "sent"))
    context = registry._ctx
    context.task_id, context.task_metadata = "work", {"work_kind": "owner"}
    context._current_llm_call_meta = {"llm_call_id": "llm-round-1"}
    return registry, context, sent


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_file_names_the_exits_that_stay_open_and_settings_shows_exactly_them(exits, case):
    from ouroboros import deployment_policy

    expected = exits(case)
    assert {name: deployment_policy.safety_exit_open(name) for name in expected} == expected
    state = deployment_policy.policy_state()
    assert state is None if case == "unmanaged" else state["allowed_exits"] == [
        name for name in (REPEAT, HUMAN, RELEASE) if expected[name]], "the file's order never matters"


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_declared_words_a_verdict_refused_repeat_only_through_an_open_exit(supervisor, exits, published_tool,
                                                                          monkeypatch, case, mode):
    """A connector's declared send (its real published effect) refused DANGEROUS: an open exit sends its identical
    re-issue from a later model call in Cyber Pro and on my human's matching card in an ordinary mode; a closed one
    refuses it from the saved verdict with no model call, card or not, in every mode. The same model call
    re-presenting it decides nothing; a new request_id is checked again in Cyber Pro and keeps waiting elsewhere."""
    h = supervisor
    opened = exits(case)
    mind, card = opened[REPEAT] and mode == "cyber_pro", opened[HUMAN]
    closed = not (mind or card)
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(_tool_effects, "effect_of", REAL_EFFECT_OF)
    name, _tool = published_tool("write", target=[], sends={"when": {"path": "chat.postMessage"},
                                                             "target": "body.channel", "text": "body.text"})
    args = {"path": "chat.postMessage", "body": {"channel": "C1", "text": "Ship it on Friday."}, "request_id": "r-1"}
    h.replies.append({"status": "DANGEROUS", "reason": "the place note asks for a quiz first"})
    ok, message = h.module.check_safety(name, args, ctx=h.ctx)
    assert not ok and "Message not sent" in message and "the place note asks for a quiz first" in message
    assert (SAID[REPEAT] in message, "ask your human privately" in message, CLOSED[REPEAT] in message) == (
        mind, card, closed)
    assert not h.module.check_safety(name, args, ctx=h.ctx)[0], "the same model call decides nothing"
    h.later()
    ok, message = h.module.check_safety(name, dict(reversed(list(args.items()))), ctx=h.ctx)
    assert ok is mind and len(h.calls) == 1, "no second model roll, open or closed"
    assert [event["type"] for event in h.events].count("safety_sent_despite_check") == int(mind)
    assert not closed or ("refused before" in message and SAID[REPEAT] not in message and CLOSED[REPEAT] in message)
    if not mind:  # my human's matching card: an open exit sends it in any mode, a closed one sends nothing
        _answered(h, "q3", name, args)
        h.later()
        ok, message = h.module.check_safety(name, args, ctx=h.ctx)
        assert ok is card and ("(q3)" in message) is card and len(h.calls) == 1
        h.ctx._owner_directives = []
    rechecked = mode == "cyber_pro" or _subject(h, name, args)[2]  # an exact-call card keys no words to a destination
    ok, message = h.module.check_safety(name, {**args, "request_id": "r-2"}, ctx=h.ctx)
    assert ok is bool(rechecked) and len(h.calls) == 1 + bool(rechecked)


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_words_only_the_verdict_saw_keep_their_mode_rule_and_a_closed_exit_keeps_them_refused(
        supervisor, exits, monkeypatch, case, mode):
    """A built-in without declarations whose verdict marks speech: an open exit repeats it in Cyber Pro and sends it
    on my human's exact-call card in an ordinary mode; a closed exit refuses its identical re-issue in both modes,
    card or not. No issue is judged twice."""
    h = supervisor
    opened = exits(case)
    mind, card = opened[REPEAT] and mode == "cyber_pro", opened[HUMAN]
    closed = not (mind or card)
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    comment = {"method": "comment", "repo": "razzant/ouroboros", "number": 2805, "body": "Taking it up."}
    h.replies.extend([{"status": "DANGEROUS", "reason": "the place note asks for a quiz first", "speech": True,
                       "quote": "Taking it up."}] * 2)
    ok, message = h.module.check_safety("github_write", comment, ctx=h.ctx)
    assert not ok and '"Taking it up."' in message
    assert (SAID[REPEAT] in message, "ask your human privately" in message, CLOSED[REPEAT] in message) == (
        mind, card, closed)
    h.later()
    ok, message = h.module.check_safety("github_write", comment, ctx=h.ctx)
    assert ok is mind and len(h.calls) == 1
    if not mind:
        _answered(h, "q5", "github_write", comment)
        h.later()
        ok, message = h.module.check_safety("github_write", comment, ctx=h.ctx)
        assert ok is card and len(h.calls) == 1


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
@pytest.mark.parametrize("case", ["unmanaged", "none", "both"])
def test_a_refused_action_that_is_no_words_keeps_its_mode_rule(supervisor, exits, monkeypatch, case, mode):
    h = supervisor
    exits(case)
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    for _ in range(2):
        h.replies.append({"status": "DANGEROUS", "reason": "deletes the home directory", "speech": False})
        ok, message = h.module.check_safety("run_command", {"cmd": "rm -rf ~"}, ctx=h.ctx)
        assert ok is (mode == "cyber_pro") and ("Cyber Pro leaves the decision" in message) is (mode == "cyber_pro")
        assert not any(text in message for text in (*SAID.values(), *CLOSED.values()))
        h.later()
    assert len(h.calls) == 2, "each issue is checked again"


@pytest.mark.parametrize("case", sorted(CASES))
def test_my_humans_answer_releases_held_words_only_through_an_open_exit(supervisor, exits, monkeypatch, case):
    """The checker cannot answer: the words wait in every mode and the hold names the card only where the exit is
    open. Closed, my human's matching answer releases nothing: the identical re-issue is checked again, waits while
    the checker is down and sends on its verdict once it answers. Reads stay available throughout."""
    from ouroboros import place_guard

    h = supervisor
    open_ = exits(case)[RELEASE]
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *a, **kw: {
        "places": [{"place_id": "p", "place_label": "Telegram · Team"}], "targets": {"chat_id": ["-100"]}})
    text = "Status of the tracks: Atlas passed 70 and 95."
    reply = {"place_id": "p", "text": text, "destination": {"conversation_id": "-100"}}
    h.replies.append(RuntimeError("provider_failed: the checker is down"))
    ok, message = h.module.check_safety("place_reply", reply, ctx=h.ctx)
    assert not ok and "SAFETY_UNAVAILABLE" in message
    assert (SAID[RELEASE] in message, CLOSED[RELEASE] in message) == (open_, not open_)
    _answered(h, "q9", "place_reply", reply)
    assert h.module.check_safety("read_file", {"path": "notes.txt"}, ctx=h.ctx) == (True, "")
    h.later()
    h.replies.append(RuntimeError("provider_failed: still down"))
    ok, message = h.module.check_safety("place_reply", reply, ctx=h.ctx)
    assert ok is open_ and len(h.calls) == (1 if open_ else 2), "closed: the same call is checked again"
    assert [e["quiz_id"] for e in h.events if e["type"] == "safety_sent_without_assessment"] == (["q9"] if open_ else [])
    if not open_:
        assert "SAFETY_UNAVAILABLE" in message and CLOSED[RELEASE] in message
        h.later()  # the checker answers again: the call is judged and sends on its verdict
        assert h.module.check_safety("place_reply", reply, ctx=h.ctx) == (True, "") and len(h.calls) == 3




@pytest.mark.parametrize("case", ["unmanaged", "unnamed", "none", "release"])
def test_my_humans_answer_runs_a_held_mcp_call_only_through_an_open_exit(supervisor, exits, monkeypatch, case):
    h = supervisor
    open_ = exits(case)[RELEASE]
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(_tool_effects, "effect_of", REAL_EFFECT_OF)
    monkeypatch.setattr(_tool_effects, "_MCP_ANNOTATIONS", {})
    name, call = "mcp_slack__conversations_add_message", {"channel_id": "C1", "payload": "Ship it on Friday."}
    exact = h.module._exact_call(name, call)
    h.replies.append(RuntimeError("provider_failed"))
    ok, message = h.module.check_safety(name, call, ctx=h.ctx)
    assert not ok and (f"on its own line of the card: {exact}\n" in message, CLOSED[RELEASE] in message) == (open_, not open_)
    _answered(h, "q7", name, call)
    h.later()
    h.replies.append(RuntimeError("provider_failed"))
    ok, message = h.module.check_safety(name, call, ctx=h.ctx)
    assert ok is open_ and ("(q7)" in message) is open_ and len(h.calls) == (1 if open_ else 2)


@pytest.mark.parametrize("exit_", [REPEAT, RELEASE])
@pytest.mark.parametrize("case", ["unmanaged", "none"])
def test_the_registry_runs_the_tool_only_once_an_open_exit_lets_it(supervisor, exits, monkeypatch, tmp_path, case, exit_):
    """At the real first-party dispatch the handler never runs before its exit's conditions hold: refused or held,
    then re-issued from a later model call (after my human's matching answer, for a hold; a closed repeat refuses
    even that answer)."""
    h = supervisor
    open_ = exits(case)[exit_]
    registry, context, sent = _registry(tmp_path, monkeypatch)
    h.replies.append({"status": "DANGEROUS", "reason": "not delegated"} if exit_ == REPEAT else RuntimeError("down"))
    first = registry.execute_result("fixture_send", {"text": "hello"})
    assert (first.status, first.code, sent) == ("blocked", "SAFETY_VIOLATION", [])
    if exit_ == RELEASE:
        _answered(h, "q1", "fixture_send", {"text": "hello"}, context)
        h.replies.append(RuntimeError("still down"))
    else:  # an ordinary mode's DANGEROUS words leave on my human's card, and a closed repeat refuses even that
        _answered(h, "q1", "fixture_send", {"text": "hello"}, context)
    context._current_llm_call_meta = {"llm_call_id": "llm-round-2"}
    second = registry.execute_result("fixture_send", {"text": "hello"})
    assert (second.status == "ok", sent) == (open_, ["hello"] if open_ else [])
    assert len(h.calls) == (1 if open_ else 1 + (exit_ == RELEASE)), "a closed repeat asks no model; a hold, the checker"


def test_a_refusal_kept_under_an_open_repeat_is_answered_by_its_verdict_once_the_exit_reads_closed(
        supervisor, exits, monkeypatch, tmp_path):
    """Words only the verdict saw, refused in an ordinary mode while the exit was open (each issue would be checked
    again), meet a closed exit at their next issue: the saved verdict answers, no second model roll runs the sink."""
    h = supervisor
    exits("all")
    registry, context, sent = _registry(tmp_path, monkeypatch, declared=False)
    h.replies.extend([{"status": "DANGEROUS", "reason": "not delegated", "speech": True},
                      {"status": "SAFE", "reason": "a second roll"}])
    first = registry.execute_result("fixture_send", {"text": "hello"})
    assert first.status == "blocked" and not any(phrase in first.text for phrase in (SAID[REPEAT], CLOSED[REPEAT]))
    exits("none")
    context._current_llm_call_meta = {"llm_call_id": "llm-round-2"}
    second = registry.execute_result("fixture_send", {"text": "hello"})
    assert (second.status, "refused before" in second.text, len(h.calls), sent) == ("blocked", True, 1, [])


def test_a_verdict_the_history_cannot_keep_is_said_audited_and_ends_the_hold_it_answered(
        supervisor, exits, monkeypatch, tmp_path):
    """Held while the checker was down, then judged DANGEROUS with a verdict the work's history cannot keep: the call
    is refused with that verdict said and audited, never as unjudged, and my human's later matching answer to the old
    hold releases nothing; the next issue is judged again."""
    h = supervisor
    exits("release")  # the unjudged release open, the repeat closed
    registry, context, sent = _registry(tmp_path, monkeypatch)
    h.replies.append(RuntimeError("provider_failed: down"))
    assert registry.execute_result("fixture_send", {"text": "hello"}).status == "blocked"
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET", 2000)
    context._current_llm_call_meta = {"llm_call_id": "llm-round-2"}
    h.replies.append({"status": "DANGEROUS", "reason": "Not delegated. " * 300})
    judged = registry.execute_result("fixture_send", {"text": "hello"}).text
    assert "TOO_LARGE" in judged and "judged" in judged and "first: DANGEROUS — Not delegated." in judged
    assert not any(phrase in judged for phrase in ("not checked", "no verdict", "wait until a check", SAID[RELEASE]))
    assert [event["verdicts"][0]["status"] for event in h.events if event["type"] == "safety_verdict"] == ["DANGEROUS"]
    _answered(h, "q-late", "fixture_send", {"text": "hello"}, context)
    context._current_llm_call_meta = {"llm_call_id": "llm-round-3"}
    h.replies.append({"status": "DANGEROUS", "reason": "Still not delegated."})
    third = registry.execute_result("fixture_send", {"text": "hello"})
    assert (third.status, sent, len(h.calls)) == ("blocked", [], 3), "judged again, never released unjudged"
    assert not [event for event in h.events if event["type"] == "safety_sent_without_assessment"]


@pytest.mark.parametrize("rootless", [False, True])
def test_a_first_unretained_verdict_does_not_become_unchecked_when_a_card_arrives(supervisor, exits, monkeypatch, rootless):
    h = supervisor
    exits("release")  # only unchecked release, not a known-DANGEROUS override
    ctx = (SimpleNamespace(task_metadata={"work_kind": "owner"}, drive_root=h.ctx.drive_root,
                           _current_llm_call_meta={"llm_call_id": "first"}) if rootless else h.ctx)
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET", 1000)
    args = {"text": "kept words", "destination": "room"}
    h.replies.append({"status": "DANGEROUS", "reason": "Not permitted. " * 300})
    assert not h.module.check_safety("place_reply", args, ctx=ctx)[0]
    h.replies.append({"status": "SAFE", "reason": "An unrelated ordinary message."})
    assert h.module.check_safety("place_reply", {**args, "text": "other words"}, ctx=ctx) == (True, "")
    _answered(h, "card-later", "place_reply", args, ctx)
    ctx._current_llm_call_meta = {"llm_call_id": "second"}
    h.replies.append({"status": "DANGEROUS", "reason": "Still not permitted."})
    assert not h.module.check_safety("place_reply", args, ctx=ctx)[0]
    assert len(h.calls) == 3 and not [e for e in h.events if e["type"] == "safety_sent_without_assessment"]


@pytest.mark.parametrize("declared", [True, False])
def test_a_preanswered_card_after_a_required_check_uses_only_the_dangerous_release(supervisor, exits, monkeypatch, declared):
    h = supervisor
    exits("human")
    name = "place_reply" if declared else "github_write"
    args = {"text": "Ship it", "destination": "room"} if declared else {"method": "comment", "body": "Ship it"}
    _answered(h, "before-send", name, args)
    h.replies.append({"status": "DANGEROUS", "reason": "Ask first", "speech": True})
    ok, _ = h.module.check_safety(name, args, ctx=h.ctx)
    if declared:  # the declared message returns its first verdict; the existing exact card then releases it
        assert not ok
        h.later()
        ok, _ = h.module.check_safety(name, args, ctx=h.ctx)
    assert ok and len(h.calls) == 1, "the card is not an unchecked release; a real verdict was obtained"
    assert not [e for e in h.events if e["type"] == "safety_sent_without_assessment"]


def test_an_unkept_verdict_on_an_action_cyber_pro_runs_is_said_without_claiming_its_outcome(
        supervisor, exits, monkeypatch, tmp_path):
    """A non-speech action judged DANGEROUS in Cyber Pro still runs on advice; when the history cannot keep that
    verdict, the assessment it carries says the verdict and audits it, never that the action did not run."""
    h = supervisor
    exits("none")
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET", 2000)
    registry, _context, sent = _registry(tmp_path, monkeypatch, declared=False)
    h.replies.append({"status": "DANGEROUS", "reason": "An action. " * 300, "speech": False})
    outcome = registry.execute_result("fixture_send", {"text": "run"})
    assert (outcome.status, sent) == ("ok", ["run"]) and "Cyber Pro leaves the decision" in outcome.text
    assert "judged" in outcome.text and not any(claim in outcome.text for claim in ("does not run", "no verdict"))
    assert [event["type"] for event in h.events].count("safety_verdict") == 1


@pytest.mark.parametrize("case", ["unmanaged", "none", "both"])
def test_no_exit_is_offered_where_nothing_was_kept(supervisor, exits, monkeypatch, case):
    """A rootless call keeps no refusal or hold, and decisive history that does not fit keeps neither: their refusal
    and hold name no repeat, no closed repeat and no card, and their next issue is checked again."""
    from ouroboros import place_guard

    h = supervisor
    exits(case)
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *a, **kw: {
        "places": [{"place_id": "p", "place_label": "Telegram · Team"}]})
    rootless = SimpleNamespace(task_metadata={"work_kind": "owner"}, drive_root=h.ctx.drive_root,
                               _current_llm_call_meta={"llm_call_id": "llm-round-1"})
    named = (*SAID.values(), *CLOSED.values())
    h.replies.extend([{"status": "DANGEROUS", "reason": "not delegated"}, RuntimeError("provider_failed")])
    for text in ("refused", "held"):
        ok, message = h.module.check_safety("place_reply", {"place_id": "p", "text": text}, ctx=rootless)
        assert not ok and not any(phrase in message for phrase in named), message
    rootless._current_llm_call_meta = {"llm_call_id": "llm-round-2"}
    assert h.module.check_safety("place_reply", {"place_id": "p", "text": "refused"}, ctx=rootless) == (True, "")
    assert len(h.calls) == 3, "nothing kept: the same call is checked again"
    # A refusal whose verdict cannot be kept holds, but a judged call is no unjudged hold: no card, though a hold
    # row (no verdict in it) would fit this budget.
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET", 1000)
    h.replies.append({"status": "DANGEROUS", "reason": "not delegated " * 100})
    ok, message = h.module.check_safety("place_reply", {"place_id": "p", "text": "too much"}, ctx=h.ctx)
    assert not ok and "TOO_LARGE" in message and not any(phrase in message for phrase in named), message
    assert h.module._PRIOR_VERDICTS["work"][h.module._subject_digest("place_reply", {"place_id": "p", "text": "too much"})] == {"judged": True}


def test_a_kept_refusal_is_read_as_its_exit_stands_now(supervisor, exits, monkeypatch):
    """The exit is read where it acts: a refusal kept while it was open answers its identical re-issue from the saved
    verdict once a file closes it, and one kept while it was closed never repeats words only the verdict saw in an
    ordinary mode once a file opens it: they wait for my human, unjudged again. A root's end forgets the refusal: the
    call is then checked again."""
    h = supervisor
    exits("unmanaged")
    monkeypatch.setattr(_tool_effects, "effect_of", lambda *a, **kw: _tool_effects.ToolEffect("message", declared=True))
    args = {"text": "Ship it.", "to": "room"}
    h.replies.append({"status": "DANGEROUS", "reason": "not delegated"})
    assert not h.module.check_safety("connector_send", args, ctx=h.ctx)[0]
    exits("none")
    h.later()
    ok, message = h.module.check_safety("connector_send", args, ctx=h.ctx)
    assert not ok and "refused before" in message and len(h.calls) == 1
    monkeypatch.setattr(_tool_effects, "effect_of", lambda *a, **kw: _tool_effects.BUILTIN)
    comment = {"method": "comment", "repo": "razzant/ouroboros", "number": 1, "body": "Done."}
    h.replies.append({"status": "DANGEROUS", "reason": "quiz first", "speech": True})
    assert not h.module.check_safety("github_write", comment, ctx=h.ctx)[0]
    exits("all")
    h.later()
    ok, message = h.module.check_safety("github_write", comment, ctx=h.ctx)
    assert not ok and "despite" not in message and "wait for your human" in message and len(h.calls) == 2, (
        "never sent by repetition, never judged twice")
    exits("none")
    h.module.clear_safety_work("work")
    monkeypatch.setattr(_tool_effects, "effect_of", lambda *a, **kw: _tool_effects.ToolEffect("message", declared=True))
    h.later()
    assert h.module.check_safety("connector_send", args, ctx=h.ctx) == (True, "") and len(h.calls) == 3

"""Unavailable checks and mandatory evidence overflow cannot become a speech grant."""
import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import tool_effects as _tool_effects
from tests._safety_fixtures import supervisor as supervisor_fixture
from tests.test_tool_effects import published_tool as _published_tool
supervisor, published_tool = supervisor_fixture, _published_tool
REAL_EFFECT_OF = _tool_effects.effect_of  # bound before the fixture stubs it
from ouroboros import place_guard as _place_guard  # noqa: E402
REAL_CALL_PLACE_FACTS = _place_guard.call_place_facts


@pytest.mark.parametrize(("required", "mode"), [("explicit", "advanced"), ("consciousness", "cyber_pro")])
def test_no_checker_hold_and_no_retry_grant(supervisor, monkeypatch, mode, required):
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(h.module, "_resolve_safety_routing", lambda: (False, False, "no backend"))
    if required == "explicit":
        h.values["OUROBOROS_MODEL_SAFETY"] = "explicit/model"
    else:
        h.ctx.task_metadata["work_kind"] = "consciousness"
    for _ in range(2):
        ok, msg = h.module.check_safety("place_reply", {"text": "hello"}, ctx=h.ctx)
        assert not ok and "SAFETY_UNAVAILABLE" in msg
    rows = h.module._PRIOR_VERDICTS.get("work", {}).values()
    assert rows and not any(row.get("flagged") for row in rows) and all(row.get("held") for row in rows)
    assert h.module.check_safety("write_remote", {}, ctx=h.ctx)[0] == (mode == "cyber_pro")
    # Restored checker can approve the exact same speech; it did not consume an earlier flag.
    monkeypatch.setattr(h.module, "_resolve_safety_routing", lambda: (False, False, None))
    assert h.module.check_safety("place_reply", {"text": "hello"}, ctx=h.ctx) == (True, "")
    assert len(h.calls) == 1


@pytest.mark.parametrize("cause", ["outage", "oversize"])
def test_held_words_wait_in_cyber_pro_whatever_tool_carries_them(supervisor, monkeypatch, published_tool, cause):
    """Unjudged words wait in every mode: a generic tool's declared send and an MCP call that is no declared read
    are held like speech in Cyber Pro (the 04.11 slack_api route); that tool's other calls, a write into a
    conversation (a reaction) and a built-in without declarations keep the advice Cyber Pro gives any unjudged call."""
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(_tool_effects, "effect_of", REAL_EFFECT_OF)
    monkeypatch.setattr(_tool_effects, "_MCP_ANNOTATIONS", {})
    name, _tool = published_tool("write", target=[], sends={"when": {"path": "chat.postMessage"}, "target": "body.channel"})
    if cause == "outage":
        h.replies.extend([RuntimeError("provider_failed: Codex reported a failed response")] * 5)
    else:
        monkeypatch.setattr(h.module, "_SAFETY_BUNDLE_BYTE_BUDGET", 10)
    held = "SAFETY_UNAVAILABLE" if cause == "outage" else "TOO_LARGE"
    for words, held_words in (((name, {"path": "chat.postMessage", "body": {"channel": "C1", "text": "hi"}}), True),
                              (("mcp_slack__conversations_add_message", {"channel_id": "C1", "payload": "hi"}), False)):
        ok, message = h.module.check_safety(*words, ctx=h.ctx)
        assert not ok and held in message, words[0]
        # Held words say so; an MCP call, which declares no words or destination, names its exact call for the card.
        assert ("These words wait" in message) is held_words
        assert ("This call waits" in message and "exact call on its own line" in message) is not held_words
    for other in ((name, {"path": "pins.add", "body": {"channel": "C1"}}), ("github_write", {"method": "comment"})):
        ok, message = h.module.check_safety(*other, ctx=h.ctx)
        assert ok and "SAFETY_ADVICE" in message and held in message, other[0]
    reaction, _tool = published_tool("write", target="channel", conversation="channel")  # "a reaction is not speech"
    ok, message = h.module.check_safety(reaction, {"channel": "C1", "name": "eyes", "timestamp": "1.1"}, ctx=h.ctx)
    assert ok and "SAFETY_ADVICE" in message and held in message


@pytest.mark.parametrize("card,sent", [("exact", True), ("other words", False), ("no destination", False),
                                       ("inside a line", False), ("answered before", True)])
def test_my_humans_answer_on_the_exact_words_sends_them(supervisor, monkeypatch, card, sent):
    """Held words wait in every mode and the hold names its exits. My human's answered card holding the exact text as
    whole lines and the destination sends them without a check, recorded as sent without assessment: a card answered
    after the hold, on a later decision (the same model call re-presenting the words decides nothing), or one answered
    before the first send, as a draft shown first. The host checks bytes and destination only."""
    from ouroboros import place_guard
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *a, **kw: {
        "places": [{"place_id": "p", "place_label": "Telegram · FusionBrain"}], "targets": {"chat_id": ["-100"]}})
    text = "Status of the tracks: Atlas passed 70 and 95."
    reply = {"place_id": "p", "text": text, "destination": {"conversation_id": "-100"}}
    words = "Status: all good." if card == "other words" else text
    place = "" if card == "no destination" else " to Telegram · FusionBrain"
    answer = [{"source": "owner_quiz_answer", "content": "1. Send it", "quiz_id": "q9", "reply_to": {
        "address": "quiz q9 of task work", "author": "Ouroboros, task work",
        "text": (f"Send exactly this{place}: {words} — fine?\n" if card == "inside a line"
                 else f"Send exactly this{place}?\n{words}\n") + "1. Send it\n2. Not now\n"
                "(host record: my human pressed option 1)"}}]
    exits = lambda: [(e["quiz_id"], e.get("hold")) for e in h.events if e["type"] == "safety_sent_without_assessment"]
    if card == "answered before":
        h.ctx._owner_directives = answer
        ok, message = h.module.check_safety("place_reply", reply, ctx=h.ctx)
        assert ok and "quiz q9" in message and not h.calls and exits() == [("q9", None)]
        return
    h.replies.append(RuntimeError("provider_failed"))
    ok, message = h.module.check_safety("place_reply", reply, ctx=h.ctx)
    assert not ok and "SAFETY_UNAVAILABLE" in message and "no verdict" in message
    assert "these exact words on their own lines, unprefixed, and the destination Telegram · FusionBrain" in message
    assert "Never route these words through another tool" in message
    h.ctx._owner_directives = answer
    h.replies.append(RuntimeError("provider_failed"))
    assert not h.module.check_safety("place_reply", reply, ctx=h.ctx)[0]  # the same model call decides nothing new
    h.later()
    h.replies.append(RuntimeError("provider_failed"))
    ok, message = h.module.check_safety("place_reply", reply, ctx=h.ctx)
    assert ok is sent and len(h.calls) == (2 if sent else 3)
    assert exits() == ([("q9", "UNAVAILABLE")] if sent else [])
    assert ("quiz q9" in message) is sent


def test_my_humans_yes_on_an_exact_held_mcp_call_runs_it_on_a_later_decision(supervisor, monkeypatch):
    """An MCP call declares no words or destination, so its hold names the exact call for my human's card: an answer
    whose card holds that exact call lets an identical re-issue from a later model call run it unjudged; a card
    holding the call with any other argument decides nothing."""
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(_tool_effects, "effect_of", REAL_EFFECT_OF)
    monkeypatch.setattr(_tool_effects, "_MCP_ANNOTATIONS", {})
    name, call = "mcp_slack__conversations_add_message", {"channel_id": "C1", "payload": "Ship it on Friday."}
    exact = h.module._exact_call(name, call)
    for card, sent in ((exact.replace("Friday", "Monday"), False), (exact, True)):
        h.replies.append(RuntimeError("provider_failed"))
        ok, message = h.module.check_safety(name, call, ctx=h.ctx)
        assert not ok and f"on its own line of the card: {exact}\n" in message
        h.ctx._owner_directives = [{"source": "owner_quiz_answer", "content": "1. Run it", "quiz_id": "q7",
                                    "reply_to": {"address": "quiz q7 of task work", "text": f"Run exactly this?\n{card}"}}]
        h.later()
        h.replies.append(RuntimeError("provider_failed"))
        ok, message = h.module.check_safety(name, call, ctx=h.ctx)
        assert ok is sent and ("quiz q7" in message) is sent
        h.later()
    assert [event["quiz_id"] for event in h.events if event["type"] == "safety_sent_without_assessment"] == ["q7"]


def test_my_humans_yes_reaches_a_declared_send_whose_room_is_no_place(supervisor, monkeypatch, published_tool):
    """A connector's declared send into a room that is no place of it is held with its declared target named for the
    card: my human's answer naming that target sends it on a later decision, one naming another target, or a longer
    one it is a part of, does not."""
    from ouroboros import place_guard
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(_tool_effects, "effect_of", REAL_EFFECT_OF)
    monkeypatch.setattr(place_guard, "call_place_facts", REAL_CALL_PLACE_FACTS)
    name, _tool = published_tool("write", target=[], sends={"when": {"path": "chat.postMessage"},
                                                             "target": "body.channel", "text": "body.text"})
    call = {"path": "chat.postMessage", "body": {"channel": "C1", "text": "Ship it on Friday."}}
    for room, sent in (("C2", False), ("C12", False), ("C1", True)):
        h.replies.append(RuntimeError("provider_failed"))
        ok, message = h.module.check_safety(name, call, ctx=h.ctx)
        assert not ok and "the destination C1 on the card" in message
        h.ctx._owner_directives = [{"source": "owner_quiz_answer", "content": "1. Send it", "quiz_id": "q8",
                                    "reply_to": {"text": f"Send this to {room}?\nShip it on Friday.\n1. Send it"}}]
        h.later()
        h.replies.append(RuntimeError("provider_failed"))
        assert h.module.check_safety(name, call, ctx=h.ctx)[0] is sent
        h.later()


def test_legacy_owner_backend_absence_still_warns(supervisor, monkeypatch):
    h = supervisor
    monkeypatch.setattr(h.module, "_resolve_safety_routing", lambda: (False, False, "no backend"))
    assert h.module.check_safety("write_remote", {}, ctx=h.ctx)[0]
    h.values["OUROBOROS_MODEL_SAFETY"] = "required"
    assert not h.module.check_safety("write_remote", {}, ctx=h.ctx)[0]


@pytest.mark.parametrize("source", ["subject", "directives", "prior"])
def test_mandatory_bundle_never_truncated(supervisor, monkeypatch, source):
    h = supervisor
    monkeypatch.setattr(h.module, "_SAFETY_BUNDLE_BYTE_BUDGET", 2000)
    args = {"text": "small"}
    if source == "subject":
        args["text"] = "tail" * 1000
    elif source == "rules":
        h.values["OUROBOROS_SAFETY_OWNER_RULES"] = "rules" * 1000
    elif source == "directives":
        h.ctx._owner_directives = [{"role": "user", "content": "exact" * 1000}]
    else:
        h.module._PRIOR_VERDICTS["work"] = {"x": {"digest": "x", "tool": "remote_write",
                                                  "verdicts": [{"stage": "first", "status": "SUSPICIOUS",
                                                                "reason": "earlier" * 1000}]}}
    for _ in range(2):
        ok, msg = h.module.check_safety("place_reply", args, ctx=h.ctx)
        assert not ok and "TOO_LARGE" in msg
    assert not h.calls
    h.module._PRIOR_VERDICTS.clear()
    h.values["OUROBOROS_SAFETY_OWNER_RULES"] = ""
    h.ctx._owner_directives = []
    assert h.module.check_safety("place_reply", {"text": "fits"}, ctx=h.ctx)[0]
    assert len(h.calls) == 1


@pytest.mark.parametrize("outage", ["throttled", "failed", "quota", "error_body"])
def test_unavailable_supervisor_route_hands_the_check_to_the_owners_chain(supervisor, monkeypatch, outage):
    from ouroboros import model_slots
    from ouroboros.llm_claudexor import ClaudexorModelError, ClaudexorModelNotDispatched

    h = supervisor
    monkeypatch.setattr(model_slots, "runtime_setting", lambda key, default=None: h.values.get(key, default))
    h.ctx.task_metadata["work_kind"] = "consciousness"  # a Place turn: its speech needs a verdict in every mode
    own = {"throttled": h.module._SafetyRateLimited("429", attempts=2),
           "failed": RuntimeError("provider_failed: Codex reported a failed response"),
           "quota": ClaudexorModelNotDispatched({"code": "subscription_window_exhausted", "message": "window spent"}),
           "error_body": ({"content": ""}, {"provider_error": {"code": "502", "message": "Bad gateway"}})}[outage]
    calls = []

    def call(**kwargs):
        role = kwargs.get("model_role", "safety")
        calls.append((kwargs["light_model"], role, kwargs.get("latch", True), kwargs.get("wait_for_resources")))
        if role != "safety":
            return {"content": json.dumps({"status": "SAFE", "reason": "fine"})}, {}
        if isinstance(own, Exception):
            raise own
        return own

    monkeypatch.setattr(h.module, "_safety_model_call", call)
    h.values["OUROBOROS_MODEL_FALLBACKS"] = "test/checker, fallback/model"  # the Supervisor's own binding is walked once
    assert h.module.check_safety("place_reply", {"text": "hello"}, ctx=h.ctx) == (True, "")
    assert calls == [("test/checker", "safety", True, False), ("fallback/model", "fallback:1", False, False)]
    route = next(event for event in h.events if event["type"] == "safety_route_fallback")
    assert route["unavailable"][0]["model"] == "test/checker"
    assert route["answered"] == {"model": "fallback/model", "role": "fallback:1"}
    assert (h.module._SAFETY_STORM_UNTIL > time.time()) == (outage == "throttled")  # its own storm: later checks skip it
    # A verdict is never an outage: the Supervisor's own DANGEROUS is final and the chain is not asked.
    calls.clear()
    own = ({"content": json.dumps({"status": "DANGEROUS", "reason": "leaks a secret"})}, {})
    ok, msg = h.module.check_safety("place_reply", {"text": "the key is"}, ctx=h.ctx)
    assert not ok and "SAFETY_VIOLATION" in msg and calls == [("test/checker", "safety", True, False)]
    # Nor is an unknown outcome, typed or of a direct call that may have answered: it never moves to another route.
    calls.clear()
    own = ClaudexorModelError({"code": "provider_failed"}, unknown=True)
    with pytest.raises(ClaudexorModelError):
        h.module.check_safety("place_reply", {"text": "again"}, ctx=h.ctx)
    own = TimeoutError("read timed out")
    own.physical_attempt_capture = SimpleNamespace(state="unresolved")
    assert not h.module.check_safety("place_reply", {"text": "once more"}, ctx=h.ctx)[0]
    assert calls == [("test/checker", "safety", True, False)] * 2


def test_with_no_answer_the_supervisors_own_route_decides_as_before(supervisor, monkeypatch):
    from ouroboros import model_slots
    from ouroboros.llm_claudexor import ClaudexorModelError, ClaudexorModelNotDispatched

    h = supervisor
    monkeypatch.setattr(model_slots, "runtime_setting", lambda key, default=None: h.values.get(key, default))
    now, chain_seconds = [1000.0], [0.0]
    monkeypatch.setattr(h.module, "time", SimpleNamespace(time=lambda: now[0], sleep=lambda delay: None))
    h.values["OUROBOROS_MODEL_FALLBACKS"] = "fallback/model"
    h.ctx.task_metadata["work_kind"] = "consciousness"
    replies, calls = {"safety": [], "fallback:0": []}, []
    storm = lambda: h.module._SafetyRateLimited("429", attempts=2)

    def call(**kwargs):
        role = kwargs.get("model_role", "safety")
        calls.append((role, kwargs.get("wait_for_resources")))
        now[0] += 0.0 if role == "safety" else chain_seconds[0]
        reply = replies[role].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"content": reply if isinstance(reply, str) else json.dumps(reply)}, {}

    monkeypatch.setattr(h.module, "_safety_model_call", call)
    walks = lambda: [(e["answered"], [skip["role"] for skip in e["unavailable"]])
                     for e in h.events if e["type"] == "safety_route_fallback"]
    # The Supervisor fails and the chain storms: the Supervisor's own failure decides; the record names both.
    replies["safety"], replies["fallback:0"] = [RuntimeError("overloaded")], [storm()]
    ok, msg = h.module.check_safety("place_reply", {"text": "one"}, ctx=h.ctx)
    assert not ok and "RuntimeError: overloaded" in msg
    assert walks() == [(None, ["safety", "fallback:0"])]
    # The storming chain route is skipped without a provider call in its window.
    calls.clear()
    replies["safety"] = [RuntimeError("overloaded")]
    assert not h.module.check_safety("place_reply", {"text": "two"}, ctx=h.ctx)[0]
    assert calls == [("safety", False)]
    # A chain route whose repair storms gets its own window too; the Supervisor's latch stays closed to it.
    now[0] += 60
    replies["safety"], replies["fallback:0"] = [RuntimeError("overloaded")], ["not json", storm()]
    assert not h.module.check_safety("place_reply", {"text": "repair"}, ctx=h.ctx)[0]
    assert h.module._CHAIN_STORMS == {("fallback/model", False, ""): now[0] + 30} and not h.module._SAFETY_STORM_UNTIL
    assert [e["action"] for e in h.events if e["type"] == "safety_check_rate_limited"] == ["blocked_unchecked_after_retry"]
    # The Supervisor's own storm keeps the window its evidence opened, however late the chain fails after it.
    now[0] += 60
    stormed_at, chain_seconds[0] = now[0], 40.0
    replies["safety"], replies["fallback:0"] = [storm()], [RuntimeError("down")]
    assert not h.module.check_safety("place_reply", {"text": "storm"}, ctx=h.ctx)[0]
    assert h.module._SAFETY_STORM_UNTIL == stormed_at + 30 and now[0] == stormed_at + 40
    # An unknown outcome on the chain still records the routes skipped before it.
    h.events.clear()
    unknown = ClaudexorModelError({"code": "provider_failed"}, unknown=True)
    replies["safety"], replies["fallback:0"] = [RuntimeError("overloaded")], [unknown]
    with pytest.raises(ClaudexorModelError):
        h.module.check_safety("place_reply", {"text": "unknown"}, ctx=h.ctx)
    assert walks() == [(None, ["safety"])]
    # Its own quota refusal keeps its owner question (the configured wait), asked after the chain.
    calls.clear()
    replies["safety"] = [ClaudexorModelNotDispatched({"code": "subscription_window_exhausted", "message": "spent"}),
                         {"status": "SAFE", "reason": "fine"}]
    replies["fallback:0"] = [RuntimeError("down")]
    assert h.module.check_safety("place_reply", {"text": "three"}, ctx=h.ctx) == (True, "")
    assert calls == [("safety", False), ("fallback:0", False), ("safety", None)]
    # A fallback local Supervisor still warns when nothing answers; a remote chain route after it never takes
    # that warning: its unknown outcome or its throttled repair holds the call.
    h.ctx.task_metadata["work_kind"] = "owner"
    monkeypatch.setattr(h.module, "_resolve_safety_routing", lambda: (True, True, None))
    unknown = TimeoutError("read timed out")
    unknown.physical_attempt_capture = SimpleNamespace(state="unresolved")
    for chain, outcome in ((RuntimeError("down"), "WARNING"), (unknown, "UNAVAILABLE"), ("not json", "UNAVAILABLE")):
        now[0] += 60
        replies["safety"], replies["fallback:0"] = [RuntimeError("local offline")], [chain, storm()]
        ok, msg = h.module.check_safety("write_remote", {"chain": repr(chain)}, ctx=h.ctx)
        assert ok == (outcome == "WARNING") and msg.startswith(f"⚠️ SAFETY_{outcome}")


def test_routes_carry_the_effective_binding_of_a_live_switch(supervisor, monkeypatch):
    from ouroboros import model_slots

    h = supervisor
    monkeypatch.setattr(model_slots, "runtime_setting", lambda key, default=None: h.values.get(key, default))
    h.values["OUROBOROS_MODEL_FALLBACKS"] = "model/a, model/b"
    walked = lambda *args: [(model, role) for model, _local, role, _binding in h.module._safety_routes(*args)]
    assert walked("model/a", False) == [("model/a", "safety"), ("model/b", "fallback:1")]
    # A wait-card switch of the Supervisor to model/b for this task: that is what is sent and charged, and
    # model/a becomes a distinct route again while model/b is walked once.
    assert walked("model/a", False, {"safety": {"model": "model/b"}}) == [("model/b", "safety"), ("model/a", "fallback:0")]
    # A switch to a local runtime is sent and charged as local.
    assert [route[:3] for route in h.module._safety_routes("model/a", False, {"fallback:1": {"use_local": True}})] == [
        ("model/a", False, "safety"), ("model/b", True, "fallback:1")]

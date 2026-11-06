"""Unavailable checks and mandatory evidence overflow cannot become a speech grant."""
import json
import time
from types import SimpleNamespace

import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture


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
    assert not h.module._PRIOR_VERDICTS
    assert h.module.check_safety("write_remote", {}, ctx=h.ctx)[0] == (mode == "cyber_pro")
    # Restored checker can approve the exact same speech; it did not consume an earlier flag.
    monkeypatch.setattr(h.module, "_resolve_safety_routing", lambda: (False, False, None))
    assert h.module.check_safety("place_reply", {"text": "hello"}, ctx=h.ctx) == (True, "")
    assert len(h.calls) == 1


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
        h.module._PRIOR_VERDICTS["work"] = {"x": {"reason": "earlier" * 1000}}
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

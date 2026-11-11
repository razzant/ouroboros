"""One explicitly marked catalog row reviews a fixed packet, without tools."""
import json

import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture


def catalog(**overrides):
    return {"enabled": True, "items": [{"subagent_id": "second", "recommended_use": "Check flags",
            "route": {"kind": "api_model", "target_id": "claudexor::source=model", "credential_profile_id": "account"},
            "effort": "high", "delivery": "packet", "rechecks_flags": True, **overrides}]}


def test_mark_validation_and_default_off():
    from ouroboros.configured_subagents import normalize_configured_subagents
    parsed, text = normalize_configured_subagents(catalog())
    assert parsed.items[0].rechecks_flags and json.loads(text)["items"][0]["rechecks_flags"]
    for override in ({"delivery": "native"}, {"rechecks_flags": "true"},
                     {"route": {"kind": "agent_session", "target_id": "codex"}, "delivery": "packet"}):
        with pytest.raises(ValueError):
            normalize_configured_subagents(catalog(**override))
    duplicate = catalog()
    duplicate["items"].append({**duplicate["items"][0], "subagent_id": "other"})
    with pytest.raises(ValueError, match="at most one"):
        normalize_configured_subagents(duplicate)
    unmarked = catalog()
    unmarked["items"][0].pop("rechecks_flags")
    assert not normalize_configured_subagents(unmarked)[0].items[0].rechecks_flags


@pytest.mark.parametrize("second,allowed", [("SAFE", True), ("SUSPICIOUS", False), ("DANGEROUS", False)])
def test_second_verdict_final_for_write(supervisor, second, allowed):
    h = supervisor
    h.values["OUROBOROS_SUBAGENTS"] = json.dumps(catalog())
    h.replies.extend([{"status": "DANGEROUS", "reason": "first reason"}, {"status": second, "reason": "second reason"}])
    assert h.module.check_safety("write_remote", {"x": 1}, ctx=h.ctx)[0] == allowed
    assert len(h.calls) == 2
    call = h.calls[1]
    assert call["light_model"] == "claudexor::source=model"
    assert call["model_role"] == "subagent:second" and call["account"] == "account"
    assert call["effort"] == "high" and call["attempts_limit"] == 1 and call["latch"] is False  # outside the first's latch
    assert call["system_prompt"] == h.calls[0]["system_prompt"]
    assert h.calls[0]["user_prompt"] in call["user_prompt"]
    assert "overturning it needs a reason" in call["user_prompt"]
    assert "first reason" in call["user_prompt"]
    assert "tools" not in call
    h.module.check_safety("write_remote", {"x": 2}, ctx=h.ctx)
    # A returned call's final reading rides beside its tool; a call let through leaves none.
    assert ('"reason": "second reason"' in h.calls[-1]["user_prompt"]) is not allowed
    assert "first reason" not in h.calls[-1]["user_prompt"]
    assert len(h.calls) == 3  # SAFE first verdict never triggers the second stage.


@pytest.mark.parametrize("second,sent", [("SAFE", True), ("DANGEROUS", False)])
def test_words_follow_the_second_stage_and_keep_both_opinions(supervisor, second, sent):
    """The final assessment decides words as any call: a second stage that overturns the flag lets them through; one
    that keeps it returns them with both reasons. Both stage opinions stay in the durable verdict either way."""
    h = supervisor
    h.values["OUROBOROS_SUBAGENTS"] = catalog()
    h.replies.extend([{"status": "DANGEROUS", "reason": "first reason"}, {"status": second, "reason": "second reason"}])
    ok, msg = h.module.check_safety("place_reply", {"text": "hi"}, ctx=h.ctx)
    assert ok is sent and len(h.calls) == 2
    verdict = next(event for event in h.events if event["type"] == "safety_verdict")
    assert [(v["stage"], v["status"], v["reason"]) for v in verdict["verdicts"]] == [
        ("first", "DANGEROUS", "first reason"), ("second", second, "second reason")]
    assert sent or (msg.startswith("⚠️ SAFETY_RETURNED: not run;") and "first: DANGEROUS — first reason" in msg
                    and "second: DANGEROUS — second reason" in msg)


@pytest.mark.parametrize("second,allowed", [("SAFE", True), ("DANGEROUS", False)])
def test_second_check_reads_an_answer_envelope(supervisor, second, allowed):
    h = supervisor
    h.values["OUROBOROS_SUBAGENTS"] = catalog()
    h.replies.extend([{"status": "DANGEROUS", "reason": "first reason"},
                      json.dumps({"answer": json.dumps({"status": second, "reason": "second reason"})})])
    ok, msg = h.module.check_safety("write_remote", {"x": 1}, ctx=h.ctx)
    assert (ok, len(h.calls)) == (allowed, 2)
    assert msg == "" if allowed else "second: DANGEROUS — second reason" in msg


def _quota():
    from ouroboros.llm_claudexor import ClaudexorModelError
    return ClaudexorModelError({"code": "subscription_window_exhausted", "message": "Quota exhausted"},
                               model_role="subagent:second")


def _unknown():
    from ouroboros.llm_claudexor import ClaudexorModelError
    return ClaudexorModelError({"message": "the answer was lost"}, model_role="subagent:second", unknown=True)


@pytest.mark.parametrize("bad", ["broken JSON", {"status": "SAFE", "reason": ""}, RuntimeError("offline"), "too large",
                                 "quota", "unknown"])
def test_a_failed_second_check_keeps_the_first_verdict(supervisor, monkeypatch, bad):
    """A second check that answers nothing (unparseable, a SAFE without a reason, an error, an unknown outcome, its
    quota or sign-in refused, a bundle too large for it) leaves the first verdict standing: the call returns with the
    first reading, never as unread, in one attempt that never waits for resources."""
    h = supervisor
    h.values["OUROBOROS_SUBAGENTS"] = catalog()
    bad = {"quota": _quota, "unknown": _unknown}[bad]() if bad in ("quota", "unknown") else bad
    h.replies.extend([{"status": "DANGEROUS", "reason": "first flag"}] + ([] if bad == "too large" else [bad]))
    if bad == "too large":  # the first bundle fits; the second, which carries it, does not
        first = h.module._safety_model_call
        def shrink(**kwargs):
            monkeypatch.setattr(h.module, "_SAFETY_BUNDLE_BYTE_BUDGET", len(kwargs["user_prompt"].encode("utf-8")) + 1)
            return first(**kwargs)
        monkeypatch.setattr(h.module, "_safety_model_call", shrink)
    ok, msg = h.module.check_safety("place_reply", {"text": "hi"}, ctx=h.ctx)
    assert not ok and msg.startswith("⚠️ SAFETY_RETURNED: not run;") and "first: DANGEROUS — first flag" in msg
    assert "SAFETY_UNAVAILABLE" not in msg and len(h.calls) == (1 if bad == "too large" else 2)
    failed = [event for event in h.events if event["type"] == "safety_second_check_failed"]
    assert len(failed) == 1 and failed[0]["error"]
    verdict = next(event for event in h.events if event["type"] == "safety_verdict")
    assert [v["stage"] for v in verdict["verdicts"]] == ["first"]
    assert bad == "too large" or h.calls[1]["wait_for_resources"] is False


@pytest.mark.parametrize("tool", ["write_remote", "place_reply"])
def test_a_refused_second_check_in_cyber_pro_keeps_the_first_verdict_and_a_stop_still_stops(supervisor, monkeypatch,
                                                                                         tool):
    """In Cyber Pro, where a built-in's hold is advice, a second check refused for its quota still leaves the first
    DANGEROUS reading standing: the write returns, never let through, and words return with the mind's exit, the
    verdict recorded. Stop or Pause during the second check still ends the check."""
    from ouroboros.model_wait import ModelWaitInterrupted

    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    h.values["OUROBOROS_SUBAGENTS"] = catalog()
    h.replies.extend([{"status": "DANGEROUS", "reason": "an undelegated disclosure"}, _quota()])
    ok, message = h.module.check_safety(tool, {"text": "private words"}, ctx=h.ctx)
    assert not ok and message.startswith("⚠️ SAFETY_RETURNED: not run;") and "an undelegated disclosure" in message
    assert "issuing the identical call in a later step" in message and h.calls[1]["wait_for_resources"] is False
    verdict = next(event for event in h.events if event["type"] == "safety_verdict")
    assert [v["stage"] for v in verdict["verdicts"]] == ["first"]
    assert [event["type"] for event in h.events].count("safety_second_check_failed") == 1
    h.replies.extend([{"status": "DANGEROUS", "reason": "an undelegated disclosure"},
                      ModelWaitInterrupted("owner_stop", role="subagent:second")])
    with pytest.raises(ModelWaitInterrupted):
        h.module.check_safety(tool, {"text": "other words"}, ctx=h.ctx)


def test_unmarked_or_disabled_never_second_checks(supervisor):
    h = supervisor
    for n, value in enumerate([catalog(rechecks_flags=False), catalog(enabled=False), {**catalog(), "enabled": False}]):
        h.values["OUROBOROS_SUBAGENTS"] = value
        h.replies.append({"status": "DANGEROUS", "reason": "flag"})
        assert not h.module.check_safety("write_remote", {"n": n}, ctx=h.ctx)[0]
    assert len(h.calls) == 3


def test_an_invalid_roster_keeps_the_first_verdict(supervisor):
    """An unreadable roster is a failed second check: a SAFE first verdict stands as SAFE, a flag as the first
    reading; its later identical issue waits for my human (DANGEROUS words) and buys no new reading."""
    h = supervisor
    h.values["OUROBOROS_SUBAGENTS"] = {"enabled": True, "items": [{"unknown_key": 1}]}
    h.replies.append({"status": "SAFE", "reason": "First checker cleared it"})
    assert h.module.check_safety("place_reply", {"text": "clear"}, ctx=h.ctx)[0]
    h.replies.append({"status": "DANGEROUS", "reason": "First checker flagged it"})
    ok, message = h.module.check_safety("place_reply", {"text": "flagged"}, ctx=h.ctx)
    assert not ok and message.startswith("⚠️ SAFETY_RETURNED: not run;") and "First checker flagged it" in message
    failed = [event for event in h.events if event["type"] == "safety_second_check_failed"]
    assert failed and "its roster is unreadable" in failed[-1]["error"]
    h.later()
    ok, message = h.module.check_safety("place_reply", {"text": "flagged"}, ctx=h.ctx)
    assert not ok and message.startswith("⚠️ SAFETY_RETURNED: not run;") and "waits for your human" in message
    assert len(h.calls) == 2


def test_each_checker_accounts_its_physical_usage(supervisor, monkeypatch):
    h = supervisor
    h.values["OUROBOROS_SUBAGENTS"] = catalog()
    h.ctx.event_queue = object()
    charged = []
    monkeypatch.setattr(h.module, "emit_llm_usage_event", lambda *a, **kw: charged.append((a, kw)))
    def call(**kwargs):
        h.calls.append(kwargs)
        kwargs["on_usage"]({"resolved_model": kwargs["light_model"], "provider": "test",
                            "prompt_tokens": 11, "completion_tokens": 2, "cost": 0.02})
        return {"content": json.dumps({"status": "DANGEROUS" if len(h.calls) == 1 else "SAFE", "reason": "reviewed"})}, {}
    monkeypatch.setattr(h.module, "_safety_model_call", call)
    assert h.module.check_safety("write_remote", {}, ctx=h.ctx)[0]
    assert [row[0][2] for row in charged] == ["test/checker", "claudexor::source=model"]
    assert [row[1]["category"] for row in charged] == ["safety", "safety"]
    assert sum(row[0][4] for row in charged) == 0.04

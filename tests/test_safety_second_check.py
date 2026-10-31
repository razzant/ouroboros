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


@pytest.mark.parametrize("second,allowed", [("SAFE", True), ("SUSPICIOUS", True), ("DANGEROUS", False)])
def test_second_verdict_final_for_write(supervisor, second, allowed):
    h = supervisor
    h.values["OUROBOROS_SUBAGENTS"] = json.dumps(catalog())
    h.replies.extend([{"status": "DANGEROUS", "reason": "first reason"}, {"status": second, "reason": "second reason"}])
    assert h.module.check_safety("write_remote", {"x": 1}, ctx=h.ctx)[0] == allowed
    assert len(h.calls) == 2
    call = h.calls[1]
    assert call["light_model"] == "claudexor::source=model"
    assert call["model_role"] == "subagent:second" and call["account"] == "account"
    assert call["effort"] == "high" and call["attempts_limit"] == 1
    assert call["system_prompt"] == h.calls[0]["system_prompt"]
    assert h.calls[0]["user_prompt"] in call["user_prompt"]
    assert "overturning it needs a reason" in call["user_prompt"]
    assert "first reason" in call["user_prompt"]
    assert "tools" not in call
    h.module.check_safety("write_remote", {"x": 2}, ctx=h.ctx)
    assert "second reason" in h.calls[-1]["user_prompt"]
    assert len(h.calls) == 3  # SAFE first verdict never triggers the second stage.


def test_speech_returns_both_reasons_even_when_second_overturns(supervisor):
    h = supervisor
    h.values["OUROBOROS_SUBAGENTS"] = catalog()
    h.replies.extend([{"status": "SUSPICIOUS", "reason": "first reason"}, {"status": "SAFE", "reason": "delegated"}])
    ok, msg = h.module.check_safety("place_reply", {"text": "hi"}, ctx=h.ctx)
    assert not ok and "first reason" in msg and "delegated" in msg
    assert h.module.check_safety("place_reply", {"text": "hi"}, ctx=h.ctx)[0]
    assert len(h.calls) == 2


@pytest.mark.parametrize("bad", ["broken JSON", {"status": "SAFE", "reason": ""}, RuntimeError("offline")])
def test_second_unavailable_is_a_hold_without_an_extra_attempt(supervisor, bad):
    h = supervisor
    h.values["OUROBOROS_SUBAGENTS"] = catalog()
    h.replies.extend([{"status": "DANGEROUS", "reason": "flag"}, bad])
    ok, msg = h.module.check_safety("place_reply", {}, ctx=h.ctx)
    assert not ok and "UNAVAILABLE" in msg and len(h.calls) == 2
    assert not h.module._PRIOR_VERDICTS


def test_unmarked_or_disabled_never_second_checks(supervisor):
    h = supervisor
    for value in [catalog(rechecks_flags=False), catalog(enabled=False), {**catalog(), "enabled": False}]:
        h.values["OUROBOROS_SUBAGENTS"] = value
        h.replies.append({"status": "DANGEROUS", "reason": "flag"})
        assert not h.module.check_safety("write_remote", {}, ctx=h.ctx)[0]
    assert len(h.calls) == 3


def test_invalid_roster_holds_only_a_flag_and_grants_no_speech_retry(supervisor):
    h = supervisor
    h.values["OUROBOROS_SUBAGENTS"] = {"enabled": True, "items": [{"unknown_key": 1}]}
    h.replies.append({"status": "SAFE", "reason": "First checker cleared it"})
    assert h.module.check_safety("place_reply", {"text": "clear"}, ctx=h.ctx)[0]
    for _ in range(2):
        h.replies.append({"status": "DANGEROUS", "reason": "First checker flagged it"})
        ok, message = h.module.check_safety("place_reply", {"text": "flagged"}, ctx=h.ctx)
        assert not ok and "UNAVAILABLE" in message and "Second-check roster" in message
    assert len(h.calls) == 3  # holds never earn deliberate-reissue authority


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

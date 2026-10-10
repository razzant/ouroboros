"""The shared Light call (``_call_consolidation_llm``): complete requests, measured fit, typed failures.

Reflection and scratchpad consolidation send through this one
call. The old dialogue writer that split blocks around it is retired, so every test
here sends one request directly. ``_LLM``, ``fit``, ``_paths`` and ``_write_chat`` are
shared helpers other memory tests import.
"""
from __future__ import annotations

import json
from math import ceil
from types import SimpleNamespace

import pytest

from ouroboros import consolidator as c
from ouroboros import context_fit
from ouroboros.capability_evidence import CapabilityEvidence


LIGHT_OUTPUT_RESERVE = 16_384


class _Refusal(RuntimeError):
    def __init__(self, message="context length exceeded", *, code="context_length_exceeded", usage=None):
        super().__init__(message)
        self.code = code
        if usage is not None:
            self.usage = usage


class _LLM:
    def __init__(self, *, limit=None, effect=None, usage=None):
        self.limit, self.effect = limit, effect
        self.calls, self.accepted = [], []
        self.usage = usage if usage is not None else {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01,
        }

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if self.effect:
            result = self.effect(self, prompt)
            if result is not None:
                return result
        if self.limit is not None and len(prompt.encode("utf-8")) > self.limit:
            raise _Refusal()
        self.accepted.append(prompt)
        return {"content": f"summary-{len(self.accepted)}"}, dict(self.usage)


@pytest.fixture
def fit(monkeypatch):
    fact = SimpleNamespace(window=100_000, density=1.0, stale=False, tasks=[])
    monkeypatch.setattr(c, "_consolidation_route", lambda: ("test/model", False))

    def resolve(task, *, allow_fetch):
        assert allow_fetch is bool(task["use_local_model"])
        fact.tasks.append(dict(task))
        evidence = CapabilityEvidence(
            fact.window or 0, "confirmed" if fact.window else "unknown", "test", "route-test",
            model=task["model"], provider="openrouter", stale=fact.stale,
        )
        return {"model": task["model"], "provider": "openrouter"}, evidence

    monkeypatch.setattr(context_fit, "resolve_context_fit_route", resolve)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_: fact.density)
    return fact


def _paths(tmp_path):
    return (tmp_path / "logs" / "chat.jsonl", tmp_path / "memory" / "dialogue_blocks.json",
            tmp_path / "memory" / "dialogue_meta.json")


def _write_chat(path, count=100, text_size=80, *, start=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"ts": f"2026-01-01T{index // 60:02d}:{index % 60:02d}:00Z", "direction": "in",
             "text": f"entry-{index} " + ("Ж🙂x" * text_size), "chat_id": 1}
            for index in range(start, start + count)]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return rows


def _call(llm, prompt="Summarize this source.\n" + "source " * 100):
    """One shared Light request, as reflection and scratchpad upkeep send it."""
    return c._call_consolidation_llm(llm, prompt, "Probe")


def _tokens(prompt):
    return context_fit.estimate_context_prompt_tokens([{"role": "user", "content": prompt}], None)


def test_known_capacity_measures_the_whole_request_with_density_and_output_reserve(fit):
    """The fit check counts the whole prompt at the route's density plus the output reserve:
    one token more than the window refuses before any request, the exact fit is sent."""
    prompt = "identity at full length " * 40 + "source " * 400
    fit.density = 2.5
    fit.window = ceil(_tokens(prompt) * fit.density) + LIGHT_OUTPUT_RESERVE
    llm = _LLM()
    content, usage = _call(llm, prompt)
    assert content == "summary-1" and len(llm.calls) == 1
    call = llm.calls[0]
    assert call["messages"][0]["content"] == prompt  # complete, never clipped
    assert call["model_role"] == "light" and call["max_tokens"] == LIGHT_OUTPUT_RESERVE
    assert usage["cost"] == pytest.approx(0.01)

    fit.window -= 1
    refused = _LLM()
    content, usage = _call(refused, prompt)
    assert content == "" and not refused.calls
    [error] = usage["_consolidation_errors"]
    assert error["kind"] == "context_overflow" and error["preflight_only"]
    assert usage["cost"] == 0  # a proven local refusal sends nothing and spends nothing


@pytest.mark.parametrize("stale", [False, True])
def test_unknown_or_stale_capacity_gets_one_ordinary_call(fit, stale):
    fit.window, fit.stale = (1 if stale else None), stale
    llm = _LLM()
    content, usage = _call(llm, "source " * 20_000)
    assert content and len(llm.calls) == 1 and not usage.get("_consolidation_errors")


@pytest.mark.parametrize("raw", [None, "", " \n"])
def test_empty_output_is_non_success_with_real_usage(fit, raw):
    llm = _LLM(effect=lambda *_: ({"content": raw}, {"cost": 0.02}))
    content, usage = _call(llm)
    assert content == "" and len(llm.calls) == 1
    assert usage["cost"] == 0.02
    assert usage["_consolidation_errors"][-1]["kind"] == "empty_summary"


@pytest.mark.parametrize("code", ["auth_required", "subscription_window_exhausted", "model_operation_interrupted", "model_outcome_unknown"])
def test_control_resource_and_unknown_model_errors_still_propagate(fit, code):
    from ouroboros.llm_claudexor import ClaudexorModelError
    error = ClaudexorModelError({"code": code, "message": "context length exceeded"})

    def fail(*_):
        raise error
    llm = _LLM(effect=fail)
    with pytest.raises(ClaudexorModelError) as caught:
        _call(llm)
    assert caught.value is error and len(llm.calls) == 1


def test_wait_interruption_propagates(fit):
    from ouroboros.model_wait import ModelWaitInterrupted
    interruption = ModelWaitInterrupted("cancelled", role="light")

    def fail(*_):
        raise interruption
    llm = _LLM(effect=fail)
    with pytest.raises(ModelWaitInterrupted) as caught:
        _call(llm)
    assert caught.value is interruption and len(llm.calls) == 1


@pytest.mark.parametrize("unresolved", [False, True])
@pytest.mark.parametrize("code", ["provider_failed", "invalid_request"])
def test_confirmed_model_context_refusal_is_typed_but_unknown_custody_propagates(fit, unresolved, code):
    from ouroboros.llm_claudexor import ClaudexorModelError
    fit.window = None
    error = ClaudexorModelError({"code": code, "message": "Controlled provider refusal",
        "context": {"httpStatus": 400, "vendorCode": "context_length_exceeded", "parameter": "input"}})
    error.physical_attempt_capture = SimpleNamespace(state="unresolved" if unresolved else "settled")

    def refuse(*_):
        raise error
    llm = _LLM(effect=refuse)
    if unresolved:
        with pytest.raises(ClaudexorModelError):
            _call(llm)
    else:
        content, usage = _call(llm)
        assert content == ""
        [refused] = usage["_consolidation_errors"]
        assert refused["kind"] == "context_overflow" and not refused["preflight_only"]
        assert usage["cost"] is None  # the refusal reported no cash
    assert len(llm.calls) == 1


def test_generic_unknown_custody_is_typed_even_through_cause(fit):
    fit.window = None
    inner = _Refusal()
    inner.physical_attempt_capture = SimpleNamespace(state="unresolved")

    def fail(*_):
        raise RuntimeError("context length exceeded") from inner
    llm = _LLM(effect=fail)
    content, usage = _call(llm)
    assert not content and len(llm.calls) == 1
    assert usage["_consolidation_errors"][-1]["kind"] == "provider_outcome_unknown"
    assert usage["cost"] is None


@pytest.mark.parametrize("message,code,kind", [
    ("max_tokens exceeds maximum context length", "", "request_too_large"),
    ("context length exceeded", "invalid_api_key", "auth_error"),
    ("request body too large", "", "request_too_large"),
    ("ordinary failure", "invalid_request", "provider_error"),
])
def test_ordinary_refusals_are_classified(fit, message, code, kind):
    def fail(*_):
        raise _Refusal(message, code=code)
    llm = _LLM(effect=fail)
    content, usage = _call(llm)
    assert not content and len(llm.calls) == 1
    assert usage["_consolidation_errors"][-1]["kind"] == kind


def test_refused_attempt_keeps_its_usage_and_ledger_ids(fit):
    fit.window = None

    def refuse(*_):
        error = _Refusal(usage={"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7, "cost": 0.03})
        error.ledger_attempt_ids = ["refused-attempt"]
        raise error
    llm = _LLM(effect=refuse)
    content, usage = _call(llm)
    assert content == "" and len(llm.calls) == 1
    assert usage["cost"] == pytest.approx(0.03)
    assert usage["prompt_tokens"] == 7 and usage["total_tokens"] == 7
    assert usage["ledger_attempt_ids"] == ["refused-attempt"]


def test_light_account_and_manual_window_share_real_context_resolver(monkeypatch):
    from ouroboros import capability_evidence, config
    model = "claudexor::codex=gpt-test"
    accounts = json.dumps({"main": "main-account", "light": "light-account"})
    windows = json.dumps({"main": 100000, "light": 17000})
    settings = {"OUROBOROS_MODEL": model, "OUROBOROS_MODEL_LIGHT": model,
                "OUROBOROS_MODEL_ACCOUNTS": accounts, "OUROBOROS_MODEL_CONTEXT_WINDOWS": windows}
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", accounts)
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(c, "_consolidation_route", lambda: (model, False))
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_: 1.0)
    probes = []

    def probe(_root, **kwargs):
        probes.append(kwargs)
        return CapabilityEvidence(100000, "confirmed", "test", "light-fingerprint", model=model)
    monkeypatch.setattr(capability_evidence, "probe", probe)
    llm = _LLM()
    text = "full source " * 50
    content, _ = _call(llm, text)

    assert content and len(llm.calls) == 1
    [call] = llm.calls
    assert call["model_account_override"] == "light-account"
    assert call["model"] == model and call["model_role"] == "light"
    assert probes and all(p["options"]["credential_profile_id"] == "light-account" for p in probes)
    assert all(p["provider"] == "claudexor" and p["allow_fetch"] is True for p in probes)
    assert context_fit.estimate_context_prompt_tokens(call["messages"]) + 16384 <= 17000
    # The light window (17000), not Main's (100000), bounds the request: a larger one is refused unsent.
    refused = _LLM()
    content, usage = _call(refused, "full source " * 1000)
    assert not content and not refused.calls
    assert usage["_consolidation_errors"][-1]["kind"] == "context_overflow"


def test_local_and_auto_account_are_passed_explicitly(fit, monkeypatch):
    monkeypatch.setattr(c, "_consolidation_route", lambda: ("local-test-model", True))
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"main": "main-pin", "light": ""}))
    llm = _LLM()
    assert _call(llm)[0] and len(llm.calls) == 1
    assert all(call["use_local"] and call["model_account_override"] == "" for call in llm.calls)
    assert all(task["use_local_model"] and task["credential_profile_id"] == "" for task in fit.tasks)


def test_wait_route_override_and_reprepare_remeasure_whole_request(fit, monkeypatch):
    from contextlib import contextmanager
    from ouroboros import model_wait
    from ouroboros.context_budget import SummarizerContextOverflow

    class Waiter:
        overrides = {"light": {"model": "changed/model", "use_local": False,
                                "model_account_override": "changed-pin"}}

        @contextmanager
        def register_reprepare(self, role, callback):
            assert role == "light"
            self.prepare = callback
            yield
    waiter = Waiter()
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    # Simulate the existing wait owner's route-switch callback before dispatch.
    def switch(llm, _):
        if len(llm.calls) == 1:
            fit.window = 16384
            with pytest.raises(SummarizerContextOverflow):
                waiter.prepare({**llm.calls[-1], "_model_observed_route": {"credentialProfileId": "changed-pin"}})
            fit.window = 100000
    llm = _LLM(effect=switch)
    assert _call(llm)[0]
    assert llm.calls[0]["model"] == "changed/model"
    assert llm.calls[0]["model_account_override"] == "changed-pin"
    # The wait's reprepare re-measured the whole request under the observed account.
    assert fit.tasks[1]["model_route"] == {"credentialProfileId": "changed-pin"}
    assert len(llm.calls) == 1 and len(fit.tasks) == 2


def test_unavailable_capacity_reader_retains_ordinary_call(monkeypatch):
    from ouroboros import capability_evidence, config
    monkeypatch.setattr(c, "_consolidation_route", lambda: ("test/model", False))
    monkeypatch.setattr(config, "load_settings", lambda: {"OUROBOROS_MODEL": "test/model"})
    def unavailable(*args, **kwargs):
        raise OSError("catalog unavailable")
    monkeypatch.setattr(capability_evidence, "probe", unavailable)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_: 1.0)
    llm = _LLM()
    assert _call(llm)[0] and len(llm.calls) == 1


@pytest.mark.parametrize("shape", ["usage_finish_reason", "anthropic_stop_reason"])
def test_output_truncation_is_refused_on_every_lane_shape(fit, shape):
    """A summary cut at the output ceiling is withheld whether the lane reports
    the cut as usage.response_finish_reason (OpenAI family) or as the message's
    stop_reason (native Anthropic)."""
    def cut(llm, _):
        if shape == "usage_finish_reason":
            return {"content": "clipped summary"}, {**llm.usage, "response_finish_reason": "length"}
        return {"content": "clipped summary", "stop_reason": "max_tokens"}, dict(llm.usage)
    llm = _LLM(effect=cut)
    content, usage = _call(llm)
    assert content == ""
    assert [error["kind"] for error in usage["_consolidation_errors"]] == ["output_truncated"]


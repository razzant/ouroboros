"""Cross-owner regressions of the shared Light call: actual fit/cache, local wire and typed interruption.

Reflection, scratchpad and knowledge upkeep send through ``_call_consolidation_llm``;
the retired dialogue writer's split/retry tests left with it.
"""
from copy import deepcopy
import json
from math import ceil
import queue
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce, config, consolidator as c, context_fit
from ouroboros.llm import LLMClient
from ouroboros.llm_claudexor import ClaudexorModelError
from tests.test_consolidator_context_fit import _LLM


MODEL = "claudexor::test-source=exact-model"


def _route(profile="account-a", fingerprint="identity-a"):
    return dict(source="test-source", model="exact-model",
                credentialProfileId=profile, accountFingerprint=fingerprint)


@pytest.fixture
def capacity(tmp_path, monkeypatch):
    settings = {"OUROBOROS_MODEL": MODEL, "OUROBOROS_MODEL_LIGHT": MODEL,
                "OUROBOROS_MODEL_ACCOUNTS": {"main": "main-account", "light": ""},
                "OUROBOROS_MODEL_CONTEXT_WINDOWS": {}}
    state = SimpleNamespace(settings=settings, catalog_calls=[], resolutions=[], window=17000,
                            route=_route(), timestamp=ce.utc_now_iso())
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps(settings["OUROBOROS_MODEL_ACCOUNTS"]))
    monkeypatch.setattr(c, "_consolidation_route", lambda: (MODEL, False))
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(ce, "_DENSITY_MEMO", {})

    def catalog(source, credential_profile_id=None, *, requested_model=None):
        state.catalog_calls.append((source, credential_profile_id, requested_model))
        return {**state.route, "observedAt": state.timestamp, "provenance": "fixture metadata transport",
                "models": [{"id": "exact-model", "contextWindow": state.window,
                            "maxContextWindow": state.window, "effectiveContextWindow": 1}]}

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    monkeypatch.setattr(ce, "_generative_probe_window", lambda *_a, **_k: pytest.fail("generation probe"))
    real_resolve = context_fit.resolve_context_fit_route

    def resolve(task, *, allow_fetch):
        resolved = real_resolve(task, allow_fetch=allow_fetch)
        state.resolutions.append((deepcopy(task), resolved[1]))
        return resolved

    monkeypatch.setattr(context_fit, "resolve_context_fit_route", resolve)
    return state


def _call(llm, source="identity\nsource " * 20):
    return c._call_consolidation_llm(llm, source, "Probe")


def _fits(call, window, density=1.0):
    return ceil(context_fit.estimate_context_prompt_tokens(call["messages"]) * density) + 16384 <= window


def _prime(capacity, *, observed=None):
    return context_fit.resolve_context_fit_route(
        {"model": MODEL, "model_role": "light", "use_local_model": False,
         "model_route": observed}, allow_fetch=True)[1]


def test_local_preflight_matches_actual_wire_normalization(capacity, monkeypatch):
    from ouroboros import local_model

    capacity.settings["OUROBOROS_MODEL"] = "local-fixture"
    monkeypatch.setattr(c, "_consolidation_route", lambda: ("local-fixture", True))
    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        get_context_length=lambda: 16384,
        serving_context_evidence=lambda: {"context_window": 16384, "confirmed": True},
    ))
    client = LLMClient(api_key="unused")
    sent = []

    def create(**kwargs):
        sent.append(deepcopy(kwargs))
        return SimpleNamespace(model_dump=lambda: {
            "choices": [{"message": {"role": "assistant", "content": "local summary"}}],
            "usage": {"prompt_tokens": 250, "completion_tokens": 20, "total_tokens": 270}})

    monkeypatch.setattr(client, "_get_local_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    # Establish what the real wire owner does, before asking the Light call to fit.
    client.chat(messages=[{"role": "user", "content": "source"}], model="local-fixture",
                max_tokens=16384, use_local=True)
    assert sent.pop()["max_tokens"] == 4096
    evidence = context_fit.resolve_context_fit_route(
        {"model": "local-fixture", "model_role": "light", "use_local_model": True}, allow_fetch=True)[1]
    assert evidence.window_tokens == 16384

    content, usage = _call(client)

    assert content == "local summary"
    assert len(sent) == 1 and sent[0]["max_tokens"] == 4096
    assert "identity" in sent[0]["messages"][0]["content"]
    assert not usage.get("_consolidation_errors")


@pytest.mark.parametrize("pin", ["", "account-a"])
def test_auto_and_pin_recover_fresh_exact_account_capacity(capacity, monkeypatch, pin):
    capacity.settings["OUROBOROS_MODEL_ACCOUNTS"]["light"] = pin
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps(capacity.settings["OUROBOROS_MODEL_ACCOUNTS"]))
    expected = _prime(capacity)
    assert expected.credential_profile_id == "account-a" and ce.is_known(expected, require_fresh=True)
    capacity.resolutions.clear()
    llm = _LLM()
    source = "whole source Ж🙂 " * 30

    content, _ = _call(llm, source)

    assert content and len(llm.calls) == 1
    [call] = llm.calls
    assert call["messages"][0]["content"] == source
    assert call["model_account_override"] == pin and call["model_role"] == "light"
    assert _fits(call, 17000)
    assert capacity.resolutions and all(ev.route_fp == expected.route_fp for _, ev in capacity.resolutions)
    # The exact account's known window bounds the request: a larger source is refused unsent.
    refused = _LLM()
    content, usage = _call(refused, "whole source Ж🙂 " * 2000)
    assert not content and not refused.calls
    assert usage["_consolidation_errors"][-1]["kind"] == "context_overflow"


@pytest.mark.parametrize("change", ["stale", "missing_identity", "missing_window"])
def test_catalog_without_fresh_complete_evidence_stays_unknown(capacity, change):
    if change == "stale":
        capacity.timestamp = "2020-01-01T00:00:00Z"
    elif change == "missing_identity":
        capacity.route["accountFingerprint"] = ""
    else:
        capacity.window = None
    llm = _LLM()
    assert _call(llm, "whole source Ж🙂 " * 2000)[0]
    assert len(llm.calls) == 1  # unknown capacity: one unchecked request
    assert not ce.is_known(capacity.resolutions[-1][1], require_fresh=True)


@pytest.mark.parametrize("profile", ["account-a", "account-b"])
def test_refusal_rebinds_its_facts_to_the_account_that_refused(capacity, profile):
    capacity.route, capacity.window = _route(profile, "identity-b"), 16_900
    expected = _prime(capacity)
    capacity.route, capacity.window = _route(), 17000
    initial = _prime(capacity)
    assert initial.route_fp != expected.route_fp
    actual = _route(profile, "identity-b")

    def refuse(llm, _prompt):
        error = ClaudexorModelError({"code": "context_length_exceeded", "message": "too long"}, route=actual)
        error.physical_attempt_capture = SimpleNamespace(state="settled")
        raise error

    llm = _LLM(effect=refuse)
    content, usage = _call(llm)
    assert not content and len(llm.calls) == 1 and llm.calls[0]["model_account_override"] == ""
    [refused] = usage["_consolidation_errors"]
    # The refusal belongs to the account that answered, not to the one discovery picked.
    assert refused["route_fp"] == expected.route_fp and refused["capacity_tokens"] == 16_900
    assert not refused["preflight_only"]


def test_exact_account_density_is_read_from_the_existing_evidence_store(tmp_path, capacity):
    capacity.window = 18000
    evidence = _prime(capacity)
    ce.record_token_density(tmp_path, MODEL, route_fp=evidence.route_fp,
                            prompt_chars=400000, prompt_tokens=200000, basis="bounded_proxy")
    density = context_fit._route_calibration_ratio(None, evidence.route_fp, MODEL)
    assert density == 2.0
    llm = _LLM()
    content, usage = _call(llm, "full dense source Ж🙂 " * 30)
    assert content and len(llm.calls) == 1 and _fits(llm.calls[0], 18000, density)
    refused = _LLM()
    content, usage = _call(refused, "full dense source Ж🙂 " * 2000)
    assert not content and not refused.calls
    [failure] = usage["_consolidation_errors"]
    assert failure["kind"] == "context_overflow" and failure["measurement_density"] == density


def test_quota_wait_reprepares_auto_with_the_new_accounts_capacity(tmp_path, capacity, monkeypatch):
    from ouroboros import model_wait

    capacity.window = 100000
    client = LLMClient(api_key="unused")
    calls, accepted = [], []
    monkeypatch.setattr(client, "claudexor_model_sources", lambda: {
        "sources": [{"id": "test-source", "credentialHarness": "fixture"}]})

    def remote(_target, messages, tools, _effort, _max_tokens, _choice, _temperature, **kwargs):
        calls.append(deepcopy(messages))
        assert kwargs["model_role"] == "light" and kwargs["model_account_override"] == ""
        if len(calls) == 1:
            capacity.route, capacity.window = _route("account-b", "identity-b"), 17000
            error = ClaudexorModelError({"code": "subscription_window_exhausted", "message": "quota"}, route=_route())
            error.physical_attempt_capture = SimpleNamespace(state="settled")
            raise error
        assert context_fit.estimate_context_prompt_tokens(messages, tools) + 16384 <= 17000
        accepted.append(messages[0]["content"])
        return {"content": "summary"}, {"cost": None, "claudexor": {"route": capacity.route}}

    monkeypatch.setattr(client, "_chat_remote", remote)
    source = "all source Ж🙂 " * 30
    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, "consolidation-fixture", "running")
    with model_wait.task_model_wait_scope(
        task={"id": "consolidation-fixture"}, drive_root=tmp_path, event_queue=queue.Queue(),
        worker_slot_held=False, owner_control=lambda: None,
    ) as waiter:
        content, usage = _call(client, source)
    assert content == "summary" and len(calls) == 2 and usage["cost"] is None
    assert accepted == [source]
    assert all(row["resolution"] == "resource_available" for row in waiter.waits.values())
    assert any(ev.credential_profile_id == "account-b" and ev.window_tokens == 17000
               for _, ev in capacity.resolutions)


def test_unavailable_route_metadata_keeps_an_ordinary_call(capacity, monkeypatch):
    def unavailable():
        raise OSError("settings read unavailable")

    monkeypatch.setattr(config, "load_settings", unavailable)
    llm = _LLM()
    assert _call(llm)[0]
    assert len(llm.calls) == 1

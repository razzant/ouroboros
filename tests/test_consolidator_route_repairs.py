"""Cross-owner regressions: actual fit/cache, local wire and typed interruption."""
from copy import deepcopy
import json
from math import ceil
import queue
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce, config, consolidator as c, context_fit
from ouroboros.llm import LLMClient
from ouroboros.llm_claudexor import ClaudexorModelError
from ouroboros.model_wait import ModelWaitInterrupted
from tests.test_consolidator_context_fit import (
    _LLM, _paths, _summary, _write_chat, _consolidate, _store, _read_sources, _reader_summary,
)


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
    # Establish what the real wire owner does, before asking consolidation to fit.
    client.chat(messages=[{"role": "user", "content": "source"}], model="local-fixture",
                max_tokens=16384, use_local=True)
    assert sent.pop()["max_tokens"] == 4096
    evidence = context_fit.resolve_context_fit_route(
        {"model": "local-fixture", "model_role": "light", "use_local_model": True}, allow_fetch=True)[1]
    assert evidence.window_tokens == 16384

    content, usage = _summary(client)

    assert content == "local summary"
    assert len(sent) == 1 and all(call["max_tokens"] == 4096 for call in sent)  # one Light operation
    assert all("identity" in call["messages"][0]["content"] for call in sent)
    assert not usage.get("_consolidation_errors")


@pytest.mark.parametrize("change", ["stale", "missing_identity", "missing_window"])
def test_catalog_without_fresh_complete_evidence_stays_unknown(capacity, change):
    if change == "stale":
        capacity.timestamp = "2020-01-01T00:00:00Z"
    elif change == "missing_identity":
        capacity.route["accountFingerprint"] = ""
    else:
        capacity.window = None
    llm = _LLM()
    assert _summary(llm)[0]
    assert len(llm.calls) == 1  # unknown capacity: one unchecked operation
    assert not ce.is_known(capacity.resolutions[-1][1], require_fresh=True)


def test_unavailable_route_metadata_keeps_an_ordinary_call(capacity, monkeypatch):
    def unavailable():
        raise OSError("settings read unavailable")

    monkeypatch.setattr(config, "load_settings", unavailable)
    llm = _LLM()
    assert _summary(llm)[0]
    assert len(llm.calls) == 1


@pytest.mark.parametrize("pin", ["", "account-a"])
def test_auto_and_pin_recover_fresh_exact_account_capacity(tmp_path, capacity, monkeypatch, pin):
    capacity.window = 30000
    capacity.settings["OUROBOROS_MODEL_ACCOUNTS"]["light"] = pin
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps(capacity.settings["OUROBOROS_MODEL_ACCOUNTS"]))
    expected = _prime(capacity)
    assert expected.credential_profile_id == "account-a" and ce.is_known(expected, require_fresh=True)
    capacity.resolutions.clear()
    source = "whole source Ж🙂 " * 18000
    actor, content, usage, reads = _reader_summary(tmp_path, source, capacity.window)
    assert content and reads.source_complete(), usage
    assert source in actor.received[0]
    assert all(call["model_account_override"] == pin and call["model_role"] == "light" for call in actor.calls)
    assert all(context_fit.estimate_context_prompt_tokens(call["messages"], call["tools"]) + 16384 <= capacity.window for call in actor.calls)
    assert all(ev.route_fp == expected.route_fp for _, ev in capacity.resolutions)
    assert len(capacity.catalog_calls) == 2  # next requests reuse the exact route cache


@pytest.mark.parametrize("receipt", ["success", "refusal"])
@pytest.mark.parametrize("profile", ["account-a", "account-b"])
def test_actual_rotated_account_rebinds_next_request_to_its_cache(capacity, receipt, profile):
    capacity.route, capacity.window = _route(profile, "identity-b"), 30000
    expected = _prime(capacity)
    capacity.route, capacity.window = _route(), 50000
    _prime(capacity)
    capacity.resolutions.clear()
    actual = _route(profile, "identity-b")
    def rotate(llm, prompt):
        if len(llm.calls) == 1:
            if receipt == "refusal":
                error = ClaudexorModelError({"code": "context_length_exceeded", "message": "too long"}, route=actual)
                error.physical_attempt_capture = SimpleNamespace(state="settled")
                raise error
            return {"content": "first summary"}, {"cost": None, "claudexor": {"route": actual}}
    llm, route = _LLM(effect=rotate), {}
    content, usage = _summary(llm, model_route=route)
    assert bool(content) == (receipt == "success")
    if receipt == "refusal":
        assert usage["_consolidation_errors"][-1]["capacity_tokens"] == 30000
    assert route == actual
    assert _summary(llm, model_route=route)[0]
    observed = [(task, ev) for task, ev in capacity.resolutions
                if (task.get("model_route") or {}).get("accountFingerprint") == "identity-b"]
    assert observed and all(ev.route_fp == expected.route_fp for _, ev in observed)
    assert all(call["model_account_override"] == "" for call in llm.calls)
    assert context_fit.estimate_context_prompt_tokens(llm.calls[-1]["messages"]) + 16384 <= 30000


def test_exact_account_density_is_read_from_the_existing_evidence_store(tmp_path, capacity):
    capacity.window = 50000
    evidence = _prime(capacity)
    ce.record_token_density(tmp_path, MODEL, route_fp=evidence.route_fp,
        prompt_chars=400000, prompt_tokens=200000, basis="bounded_proxy")
    density = context_fit._route_calibration_ratio(None, evidence.route_fp, MODEL)
    assert density == 2.0
    source = "full dense source Ж🙂 " * 12000
    actor, content, usage, reads = _reader_summary(tmp_path, source, capacity.window)
    assert content and reads.source_complete() and source in actor.received[0], usage
    assert all(ceil(context_fit.estimate_context_prompt_tokens(call["messages"], call["tools"]) * density) + 16384 <= capacity.window
        for call in actor.calls)


def test_quota_wait_reprepares_auto_with_the_new_accounts_capacity(tmp_path, capacity, monkeypatch):
    from ouroboros import model_wait
    capacity.window = 100000
    client = LLMClient(api_key="unused")
    calls = []
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
        return {"content": "summary"}, {"cost": None, "claudexor": {"route": capacity.route}}
    monkeypatch.setattr(client, "_chat_remote", remote)
    with model_wait.task_model_wait_scope(task={"id": "consolidation-fixture"}, drive_root=tmp_path,
        event_queue=queue.Queue(), worker_slot_held=False, owner_control=lambda: None) as waiter:
        content, usage = _summary(client, "all source Ж🙂 " * 1500)
        assert not content and len(calls) == 1  # reprepare refuses BEFORE another physical attempt
        assert usage["_consolidation_errors"][-1]["kind"] == "context_overflow"
        assert _summary(client, "short source")[0]
    assert all(row["resolution"] == "resource_available" for row in waiter.waits.values())
    assert any(ev.credential_profile_id == "account-b" and ev.window_tokens == 17000 for _, ev in capacity.resolutions)


def test_oversized_digest_keeps_all_original_blocks_and_reads_complete_source(tmp_path, capacity):
    from tests.test_memory_pressure_maintenance import SourceReader
    capacity.window = 30000
    chat, blocks, _meta = _paths(tmp_path)
    _write_chat(chat, count=2, text_size=0)
    originals = [{"range": "2025-01-01", "type": "summary", "message_count": 100,
        "content": f"old-{index} " * 9000} for index in range(3)]
    blocks.parent.mkdir(parents=True)
    original = json.dumps(originals).encode()
    blocks.write_bytes(original)
    actor = SourceReader(tmp_path, capacity.window)
    usage = _consolidate(tmp_path, actor, compact_chronicle=True, pressure_fits=lambda: False)
    assert blocks.read_bytes() == original
    assert _store(tmp_path).scan_state()["last_consolidated_offset"] == 2
    assert _store(tmp_path).records(kinds=["digest"])
    assert not usage.get("_consolidation_errors"), usage
    assert actor.received and all(row["content"] in actor.received[0] for row in originals)


@pytest.mark.parametrize("interrupt", ["quota", "owner", "deadline"])
def test_learned_refusal_survives_typed_interruption_before_next_cycle(tmp_path, capacity, interrupt):
    from tests.test_memory_pressure_maintenance import SourceReader
    capacity.window = None
    chat, _blocks, _meta = _paths(tmp_path)
    rows = _write_chat(chat, count=2, text_size=4000)
    original = chat.read_bytes()
    refusal = ClaudexorModelError({"code": "context_length_exceeded", "message": "too long"}, route=_route())
    refusal.physical_attempt_capture = SimpleNamespace(state="settled")
    def reject(*_):
        raise refusal
    first = _LLM(effect=reject)
    _consolidate(tmp_path, first)
    store = _store(tmp_path)
    bound = store.records(kinds=["input_refusal"])[0]["input_limit"]
    original_size = len(first.calls[0]["messages"][0]["content"].encode())
    assert bound["input_bytes"] == original_size - 1
    error = (ClaudexorModelError({"code": "subscription_window_exhausted", "message": "quota"}, route=_route())
        if interrupt == "quota" else ModelWaitInterrupted("finalize_requested" if interrupt == "owner" else "deadline", role="light"))
    def stop(*_):
        raise error
    interrupted = _LLM(effect=stop)
    with pytest.raises(type(error)) as caught:
        _consolidate(tmp_path, interrupted)
    assert caught.value is error and len(interrupted.calls) == 1
    assert len(interrupted.calls[0]["messages"][0]["content"].encode()) < original_size
    assert store.records(kinds=["input_refusal"])[0]["input_limit"] == bound
    assert not store.scan_state().get("last_consolidated_offset")
    actor = SourceReader(tmp_path, 1000000)
    _consolidate(tmp_path, actor)
    assert _read_sources(tmp_path, store) == [rows]
    assert store.scan_state()["last_consolidated_offset"] == 2
    assert chat.read_bytes() == original


@pytest.mark.parametrize("change", ["source", "route"])
def test_refusal_bound_invalidates_for_changed_source_or_route(tmp_path, capacity, change):
    capacity.window = None
    chat, _blocks, _meta = _paths(tmp_path)
    _write_chat(chat, count=2, text_size=0)
    def reject(*_):
        error = ClaudexorModelError({"code": "context_length_exceeded", "message": "too long"}, route=_route())
        error.physical_attempt_capture = SimpleNamespace(state="settled")
        raise error
    _consolidate(tmp_path, _LLM(effect=reject))
    assert _store(tmp_path).records(kinds=["input_refusal"])
    if change == "route":
        capacity.route = _route("account-b", "identity-b")
    second = _LLM()
    _consolidate(tmp_path, second, "changed identity" if change == "source" else "")
    assert len(second.calls) == 2
    assert not second.calls[0]["messages"][0]["content"].startswith("Complete source and instructions")
    assert _store(tmp_path).scan_state()["last_consolidated_offset"] == 2

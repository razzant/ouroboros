"""Complete Light requests and progress-safe consolidation (ported from 583acd22)."""
from __future__ import annotations

import json
from math import ceil
from types import SimpleNamespace

import pytest

from ouroboros import consolidator as c
from ouroboros import context_fit, room_consolidation as rc
from ouroboros.capability_evidence import CapabilityEvidence


LIGHT_OUTPUT_RESERVE = 16_384
DRAFT_HEADING = rc.DRAFT_SOURCE_HEADING + "\n"
CORRECTION_HEADING = rc.CORRECTION_SOURCE_HEADING + "\n"
_RANGE = "2026-01-01 01:00 - 02:00"


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


def _write_chat(path, count=100, text_size=80, *, start=0, task_id="fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"ts": f"2026-01-01T{index // 60:02d}:{index % 60:02d}:00Z", "direction": "in",
             "text": f"entry-{index} " + ("Ж🙂x" * text_size), "chat_id": 1, "task_id": task_id}
            for index in range(start, start + count)]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return rows


def _drafts(prompts):
    return [prompt for prompt in prompts if DRAFT_HEADING in prompt]


def _corrections(prompts):
    return [prompt for prompt in prompts if CORRECTION_HEADING in prompt]


def _summary(llm, text="source" * 100, **kwargs):
    """One real Light source operation; publication/correction belongs to the store writer."""
    prompt = rc.room_draft_prompt(text, room_label="Main", block_range_text=_RANGE,
        message_count=1, identity_text="identity", helper=True)
    fixed = rc.room_draft_prompt("", room_label="Main", block_range_text=_RANGE,
        message_count=1, identity_text="identity", helper=True)
    return c._call_consolidation_llm(llm, prompt, "Room episode", fixed_prompt=fixed, **kwargs)


@pytest.mark.parametrize("stale", [False, True])
def test_unknown_or_stale_capacity_gets_one_ordinary_call(fit, stale):
    fit.window, fit.stale = (1 if stale else None), stale
    llm = _LLM()
    assert _summary(llm)[0]
    assert len(llm.calls) == 1  # one operation with unknown capacity
    assert _drafts(llm.accepted) == llm.accepted and not _corrections(llm.accepted)


def test_generic_unknown_custody_is_not_a_context_retry_even_through_cause(fit):
    fit.window = None
    inner = _Refusal()
    inner.physical_attempt_capture = SimpleNamespace(state="unresolved")
    def fail(*_):
        raise RuntimeError("context length exceeded") from inner
    llm = _LLM(effect=fail)
    content, usage = _summary(llm)
    assert not content and len(llm.calls) == 1
    assert usage["_consolidation_errors"][-1]["kind"] == "provider_outcome_unknown"
    assert usage["cost"] is None


@pytest.mark.parametrize("message,code,kind", [
    ("max_tokens exceeds maximum context length", "", "request_too_large"),
    ("context length exceeded", "invalid_api_key", "auth_error"),
    ("request body too large", "", "request_too_large"),
    ("ordinary failure", "invalid_request", "provider_error"),
])
def test_non_context_refusals_never_split(fit, message, code, kind):
    def fail(*_):
        raise _Refusal(message, code=code)
    llm = _LLM(effect=fail)
    content, usage = _summary(llm)
    assert not content and len(llm.calls) == 1
    assert usage["_consolidation_errors"][-1]["kind"] == kind


def test_local_and_auto_account_are_passed_explicitly(fit, monkeypatch):
    monkeypatch.setattr(c, "_consolidation_route", lambda: ("local-test-model", True))
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"main": "main-pin", "light": ""}))
    llm = _LLM()
    assert _summary(llm)[0]
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
    assert _summary(llm)[0]
    assert llm.calls[0]["model"] == "changed/model"
    assert llm.calls[0]["model_account_override"] == "changed-pin"
    # The wait reprepare remeasures the same operation under the observed account.
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
    assert _summary(llm)[0] and len(llm.calls) == 1


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
    content, usage = _summary(llm)
    assert content == ""
    assert [error["kind"] for error in usage["_consolidation_errors"]] == ["output_truncated"]


def _consolidate(root, llm, identity="", **kwargs):
    from ouroboros.tools.registry import ToolContext
    chat, blocks, meta = _paths(root)
    return c.consolidate(chat, blocks, meta, llm, identity,
        knowledge_context=ToolContext(repo_dir=root, drive_root=root, task_id="fixture"),
        completed_task={"id": "fixture"}, **kwargs)


def _store(root):
    from ouroboros.chronicle_store import ChronicleStore
    return ChronicleStore(root)


def _read_sources(root, store):
    from ouroboros.artifacts import read_actor_source_bytes
    return [json.loads(read_actor_source_bytes(root, ref["task_id"], ref))
        for episode in store.records(kinds=["episode"]) for ref in episode["source_refs"]]


def _reader_summary(root, source, window):
    from ouroboros.tools.registry import ToolContext
    from tests.test_memory_pressure_maintenance import SourceReader
    actor = SourceReader(root, window)
    knowledge = c.KnowledgeReadContext(ToolContext(repo_dir=root, drive_root=root, task_id="fixture"))
    content, usage = _summary(actor, source, knowledge=knowledge)
    return actor, content, usage, knowledge


@pytest.mark.parametrize("code", ["provider_failed", "invalid_request"])
def test_oversized_source_refusal_retains_bound_then_reads_complete_source(tmp_path, fit, code):
    from ouroboros.llm_claudexor import ClaudexorModelError
    from tests.test_memory_pressure_maintenance import SourceReader
    fit.window = None
    chat, _blocks, _meta = _paths(tmp_path)
    rows = _write_chat(chat, count=3, text_size=25000)
    original = chat.read_bytes()
    def refuse(*_):
        error = ClaudexorModelError({"code": code, "message": "Controlled refusal",
            "context": {"httpStatus": 400, "vendorCode": "context_length_exceeded", "parameter": "input"}})
        error.physical_attempt_capture = SimpleNamespace(state="settled")
        raise error
    first = _LLM(effect=refuse)
    refused = _consolidate(tmp_path, first)
    assert len(first.calls) == 1 and refused["cost"] is None
    store = _store(tmp_path)
    assert not store.records(kinds=["episode"])
    bound = store.records(kinds=["input_refusal"])[0]["input_limit"]
    assert bound["input_bytes"] == len(first.calls[0]["messages"][0]["content"].encode()) - 1
    actor = SourceReader(tmp_path, 1000000)
    result = _consolidate(tmp_path, actor)
    assert result["_blocks_written"] == 1
    assert actor.sources and json.dumps(rows, ensure_ascii=False, sort_keys=True) in actor.received[0]
    assert _read_sources(tmp_path, store) == [rows]
    assert store.scan_state()["last_consolidated_offset"] == len(rows)
    assert "last_consolidation_error" not in store.scan_state()
    assert chat.read_bytes() == original


def test_known_capacity_includes_whole_prompt_density_and_output_reserve(tmp_path, fit):
    from tests.test_memory_pressure_maintenance import SourceReader
    fit.window, fit.density = 50000, 2.5
    chat, _blocks, _meta = _paths(tmp_path)
    rows = _write_chat(chat, count=2, text_size=30000)
    identity = "Identity must remain complete. " * 20
    actor = SourceReader(tmp_path, fit.window)
    usage = _consolidate(tmp_path, actor, identity)
    assert usage["_blocks_written"] == 1, usage
    for call in actor.calls:
        tokens = ceil(context_fit.estimate_context_prompt_tokens(call["messages"], call["tools"]) * fit.density)
        assert tokens + call["max_tokens"] <= fit.window
        assert call["model_role"] == "light" and call["max_tokens"] == LIGHT_OUTPUT_RESERVE
    assert all(identity in source for source in actor.received)
    assert _read_sources(tmp_path, _store(tmp_path)) == [rows]
    assert usage["cost"] == pytest.approx(.01 * len(actor.calls))


def test_known_capacity_overhead_refusal_is_typed_without_model_call(tmp_path, fit):
    fit.window = LIGHT_OUTPUT_RESERVE
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=2)
    before = chat.read_bytes()
    llm = _LLM()
    for _ in range(2):
        usage = _consolidate(tmp_path, llm, "large identity" * 300)
        assert usage["_consolidation_errors"][-1]["kind"] == "context_overflow"
        assert usage["cost"] == 0
    store = _store(tmp_path)
    assert not store.scan_state().get("last_consolidated_offset")
    assert store.scan_state()["last_consolidation_error"]["preflight_only"]
    assert not llm.calls and not blocks.exists() and not meta.exists()
    assert chat.read_bytes() == before


def test_unread_pointer_never_publishes_or_advances_and_source_is_retained(tmp_path, fit):
    from ouroboros.artifacts import read_actor_source_bytes
    fit.window = 24000
    chat, _blocks, _meta = _paths(tmp_path)
    _write_chat(chat, count=1, text_size=30000)
    original = chat.read_bytes()
    usage = _consolidate(tmp_path, _LLM())
    error = usage["_consolidation_errors"][-1]
    assert error["kind"] == "source_incomplete"
    ref = error["source_ref"]
    assert "entry-0" in read_actor_source_bytes(tmp_path, ref["task_id"], ref).decode()
    assert error["response_ref"]
    assert not _store(tmp_path).records(kinds=["episode"])
    assert not _store(tmp_path).scan_state().get("last_consolidated_offset")
    assert chat.read_bytes() == original


@pytest.mark.parametrize("window", [30000, 50000])
def test_route_capacity_changes_complete_source_delivery_without_provider_branch(tmp_path, fit, window):
    fit.window = window
    source = "BEGINNING Ж🙂 " + "no-newline-long-source " * 18000 + " DECISIVE END"
    actor, content, usage, reads = _reader_summary(tmp_path, source, window)
    assert content and not usage.get("_consolidation_errors"), usage
    assert reads.source_complete()
    assert len(actor.calls) > 2 and len(actor.received) == 1
    assert source in actor.received[0]
    assert "DECISIVE END" in actor.received[0]


@pytest.mark.parametrize("raw", [None, "", " \n"])
def test_empty_output_is_non_success_with_real_usage(tmp_path, fit, raw):
    chat, _blocks, _meta = _paths(tmp_path)
    _write_chat(chat, count=2)
    llm = _LLM(effect=lambda *_: ({"content": raw}, {"cost": .02}))
    usage = _consolidate(tmp_path, llm)
    assert usage["cost"] == .02 and len(llm.calls) == 1
    assert usage["_consolidation_errors"][-1]["kind"] == "empty_summary"
    assert not _store(tmp_path).records(kinds=["episode"])
    assert not _store(tmp_path).scan_state().get("last_consolidated_offset")


@pytest.mark.parametrize("code", ["auth_required", "subscription_window_exhausted", "model_operation_interrupted", "model_outcome_unknown"])
def test_control_resource_and_unknown_model_errors_still_propagate(tmp_path, fit, code):
    from ouroboros.llm_claudexor import ClaudexorModelError
    chat, _blocks, _meta = _paths(tmp_path)
    _write_chat(chat, count=2)
    error = ClaudexorModelError({"code": code, "message": "context length exceeded"})
    def fail(*_):
        raise error
    llm = _LLM(effect=fail)
    with pytest.raises(ClaudexorModelError) as caught:
        _consolidate(tmp_path, llm)
    assert caught.value is error and len(llm.calls) == 1
    assert not _store(tmp_path).records(kinds=["episode"])
    assert not _store(tmp_path).scan_state().get("last_consolidated_offset")


def test_wait_interruption_after_original_preserves_original_for_correction(tmp_path, fit):
    from ouroboros.model_wait import ModelWaitInterrupted
    chat, _blocks, _meta = _paths(tmp_path)
    rows = _write_chat(chat, count=2)
    def fail(llm, _prompt):
        if len(llm.calls) == 2:
            raise ModelWaitInterrupted("cancelled", role="light")
    llm = _LLM(effect=fail)
    with pytest.raises(ModelWaitInterrupted):
        _consolidate(tmp_path, llm)
    assert len(llm.calls) == 2
    store = _store(tmp_path)
    assert _read_sources(tmp_path, store) == [rows] and not store.records(kinds=["revision"])
    retry = _LLM()
    _consolidate(tmp_path, retry)
    assert len(retry.calls) == 1 and len(store.records(kinds=["episode"])) == 1
    assert len(store.records(kinds=["revision"])) == 1
    assert store.scan_state()["last_consolidated_offset"] == 2


@pytest.mark.parametrize("code", ["provider_failed", "invalid_request"])
def test_unknown_paid_refusal_never_repeats_on_next_cycle(tmp_path, fit, code):
    from ouroboros.llm_claudexor import ClaudexorModelError
    chat, _blocks, _meta = _paths(tmp_path)
    _write_chat(chat, count=2)
    error = ClaudexorModelError({"code": code, "message": "context length exceeded"})
    error.physical_attempt_capture = SimpleNamespace(state="unresolved")
    error.ledger_attempt_ids = ["paid-unknown"]
    def fail(*_):
        raise error
    llm = _LLM(effect=fail)
    with pytest.raises(ClaudexorModelError):
        _consolidate(tmp_path, llm)
    _consolidate(tmp_path, llm)
    assert len(llm.calls) == 1
    assert not _store(tmp_path).records(kinds=["input_refusal", "episode"])
    assert _store(tmp_path).scan_state()["pending_consolidation_outcomes"]


def test_refused_attempt_usage_is_retained_separately_from_later_success(tmp_path, fit):
    fit.window = None
    chat, _blocks, _meta = _paths(tmp_path)
    _write_chat(chat, count=1, text_size=10000)
    error = _Refusal(usage={"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7, "cost": .03})
    error.ledger_attempt_ids = ["refused-attempt"]
    def fail(*_):
        raise error
    usage = _consolidate(tmp_path, _LLM(effect=fail))
    assert usage["cost"] == .03 and usage["prompt_tokens"] == usage["total_tokens"] == 7
    assert usage["ledger_attempt_ids"] == ["refused-attempt"]
    from tests.test_memory_pressure_maintenance import SourceReader
    actor = SourceReader(tmp_path, 1000000)
    later = _consolidate(tmp_path, actor)
    assert later["_blocks_written"] == 1
    assert later["cost"] == pytest.approx(.01 * len(actor.calls))


@pytest.mark.parametrize("failure_at", [1, 2])
def test_auth_failure_preserves_only_already_published_original(tmp_path, fit, failure_at):
    chat, _blocks, _meta = _paths(tmp_path)
    rows = _write_chat(chat, count=2)
    def fail(llm, _prompt):
        if len(llm.calls) == failure_at:
            raise _Refusal("provider refused", code="invalid_api_key")
    llm = _LLM(effect=fail)
    usage = _consolidate(tmp_path, llm)
    assert len(llm.calls) == failure_at and usage["cost"] is None
    assert usage["_consolidation_errors"][-1]["kind"] == "auth_error"
    store = _store(tmp_path)
    assert _read_sources(tmp_path, store) == ([rows] if failure_at == 2 else [])
    assert store.scan_state().get("last_consolidated_offset", 0) == (2 if failure_at == 2 else 0)


def test_publication_failure_never_advances_and_retains_exact_source(tmp_path, fit, monkeypatch):
    from ouroboros.chronicle_store import ChronicleStore
    chat, _blocks, _meta = _paths(tmp_path)
    _write_chat(chat, count=2)
    original = chat.read_bytes()
    store = _store(tmp_path)
    store.import_legacy()
    def reject(*_args, **_kwargs):
        raise OSError("disk unavailable")
    monkeypatch.setattr(ChronicleStore, "append_episode", reject)
    with pytest.raises(OSError, match="disk unavailable"):
        _consolidate(tmp_path, _LLM())
    assert not store.scan_state().get("last_consolidated_offset")
    assert not store.records(kinds=["episode"])
    assert chat.read_bytes() == original
    assert list((tmp_path / "task_results/artifacts/fixture/source_handles/context_checkpoints").glob("*"))


@pytest.mark.parametrize("failing_call", [3, 4])
@pytest.mark.parametrize("unknown", [False, True])
def test_partial_room_failure_never_starts_digest_work(tmp_path, fit, unknown, failing_call):
    chat, blocks, _meta = _paths(tmp_path)
    rows = _write_chat(chat, count=4, text_size=0)
    for row in rows[2:]:
        row["chat_id"] = 2
    chat.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    blocks.parent.mkdir(parents=True)
    original = json.dumps([{"content": "Earlier history " * 100}]).encode()
    blocks.write_bytes(original)
    def fail(llm, _prompt):
        if len(llm.calls) == failing_call:
            error = _Refusal("unknown" if unknown else "auth failed", code="invalid_api_key")
            if unknown:
                error.physical_attempt_capture = SimpleNamespace(state="unresolved")
                error.ledger_attempt_ids = ["unknown-room"]
            raise error
    llm = _LLM(effect=fail)
    usage = _consolidate(tmp_path, llm, compact_chronicle=True, pressure_fits=lambda: False)
    assert len(llm.calls) == failing_call
    assert not _store(tmp_path).records(kinds=["digest"])
    assert blocks.read_bytes() == original
    assert usage["_blocks_written"] == (1 if failing_call == 3 else 2)
    assert _store(tmp_path).scan_state()["last_consolidated_offset"] == (2 if failing_call == 3 else 4)
    assert _store(tmp_path).scan_state()["last_consolidation_error"]


def test_failed_digest_usage_is_accounted_and_original_sources_survive(tmp_path, fit):
    chat, blocks, _meta = _paths(tmp_path)
    _write_chat(chat, count=2, text_size=0)
    blocks.parent.mkdir(parents=True)
    original = json.dumps([{"content": "Earlier history " * 100}]).encode()
    blocks.write_bytes(original)
    def fail(llm, _prompt):
        if len(llm.calls) == 3:
            return {"content": ""}, {"cost": .04}
    usage = _consolidate(tmp_path, _LLM(effect=fail), compact_chronicle=True, pressure_fits=lambda: False)
    assert usage["cost"] == pytest.approx(.06)
    assert blocks.read_bytes() == original
    assert _store(tmp_path).scan_state()["last_consolidated_offset"] == 2
    assert not _store(tmp_path).records(kinds=["digest"])


def test_unknown_capacity_impossible_pointer_is_not_replayed_after_definitive_refusal(tmp_path, fit):
    fit.window = None
    chat, _blocks, _meta = _paths(tmp_path)
    _write_chat(chat, count=1, text_size=4000)
    llm = _LLM(limit=1)
    _consolidate(tmp_path, llm)  # complete source is definitively too large
    _consolidate(tmp_path, llm)  # the smaller retained-source pointer also fails
    sent = len(llm.calls)
    assert sent == 2
    fit.density = 4.5  # density drift cannot invalidate an exact byte refusal
    usage = _consolidate(tmp_path, llm)
    assert len(llm.calls) == sent
    assert not _store(tmp_path).scan_state().get("last_consolidated_offset")
    assert usage["_consolidation_errors"]
    fit.window, llm.limit = 100000, None  # fresh capacity is new route evidence
    _consolidate(tmp_path, llm)
    assert _store(tmp_path).scan_state()["last_consolidated_offset"] == 1


def test_budget_failure_stops_before_other_rooms_and_digest(tmp_path, fit):
    from ouroboros.usage_accounting import BudgetExceeded
    chat, _blocks, _meta = _paths(tmp_path)
    rows = _write_chat(chat, count=2)
    rows[1]["chat_id"] = 2
    chat.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    def exhausted(*_):
        raise BudgetExceeded("budget exhausted")
    llm = _LLM(effect=exhausted)
    result = _consolidate(tmp_path, llm, compact_chronicle=True, pressure_fits=lambda: False)
    assert len(llm.calls) == 1
    assert result["_consolidation_errors"][-1]["kind"] == "budget_exhausted"
    assert not _store(tmp_path).records(kinds=["episode", "digest"])
    assert not _store(tmp_path).scan_state().get("last_consolidated_offset")


def test_light_manual_window_and_account_share_real_context_resolver(monkeypatch):
    from ouroboros import capability_evidence, config
    model = "claudexor::codex=gpt-test"
    settings = {"OUROBOROS_MODEL": model, "OUROBOROS_MODEL_LIGHT": model,
        "OUROBOROS_MODEL_ACCOUNTS": {"main": "main-account", "light": "light-account"},
        "OUROBOROS_MODEL_CONTEXT_WINDOWS": {"main": 100000, "light": 17000}}
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps(settings["OUROBOROS_MODEL_ACCOUNTS"]))
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(c, "_consolidation_route", lambda: (model, False))
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_: 1.0)
    probes = []
    def probe(_root, **kwargs):
        probes.append(kwargs)
        return CapabilityEvidence(100000, "confirmed", "test", "light-fingerprint", model=model)
    monkeypatch.setattr(capability_evidence, "probe", probe)
    llm = _LLM()
    assert _summary(llm, "short source")[0]
    content, usage = _summary(llm, "full source " * 4000)
    assert not content and len(llm.calls) == 1
    assert usage["_consolidation_errors"][-1]["capacity_tokens"] == 17000
    assert all(call["model_account_override"] == "light-account" for call in llm.calls)
    assert all(p["options"]["credential_profile_id"] == "light-account" for p in probes)
    assert all(p["provider"] == "claudexor" and p["allow_fetch"] for p in probes)


@pytest.mark.parametrize("preceding_rooms", [0, 1])
def test_refusal_bound_survives_preceding_published_room(tmp_path, fit, preceding_rooms):
    from tests.test_memory_pressure_maintenance import SourceReader
    fit.window = None
    chat, _blocks, _meta = _paths(tmp_path)
    rows = _write_chat(chat, count=2 * (preceding_rooms + 1), text_size=4000)
    for row in rows[2 * preceding_rooms:]:
        row["chat_id"] = 2
    chat.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    target_call = 2 * preceding_rooms + 1
    def reject_target(llm, _prompt):
        if len(llm.calls) == target_call:
            raise _Refusal()
    first = _LLM(effect=reject_target)
    _consolidate(tmp_path, first)
    store = _store(tmp_path)
    assert len(first.calls) == target_call
    assert store.scan_state().get("last_consolidated_offset", 0) == 2 * preceding_rooms
    assert len(store.records(kinds=["episode"])) == preceding_rooms
    bound = store.records(kinds=["input_refusal"])[0]["input_limit"]
    rejected_size = len(first.calls[-1]["messages"][0]["content"].encode())
    assert bound["input_bytes"] == rejected_size - 1
    actor = SourceReader(tmp_path, 1000000)
    _consolidate(tmp_path, actor)
    assert len(actor.calls[0]["messages"][0]["content"].encode()) < rejected_size
    assert len(store.records(kinds=["episode"])) == preceding_rooms + 1
    assert [row for part in _read_sources(tmp_path, store) for row in part] == rows
    assert store.scan_state()["last_consolidated_offset"] == len(rows)

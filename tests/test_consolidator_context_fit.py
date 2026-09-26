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


def _write_chat(path, count=100, text_size=80, *, start=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"ts": f"2026-01-01T{index // 60:02d}:{index % 60:02d}:00Z", "direction": "in",
             "text": f"entry-{index} " + ("Ж🙂x" * text_size), "chat_id": 1}
            for index in range(start, start + count)]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return rows


def _drafts(prompts):
    return [prompt for prompt in prompts if DRAFT_HEADING in prompt]


def _corrections(prompts):
    return [prompt for prompt in prompts if CORRECTION_HEADING in prompt]


def _source(prompts):
    """Exact source bytes the DRAFT calls received, in order."""
    return "".join(prompt.split(DRAFT_HEADING, 1)[1][:-1] for prompt in _drafts(prompts))


def _corrected_source(prompts):
    """Exact complete source bytes the CORRECTION calls compared against, in order."""
    return "".join(prompt.split(CORRECTION_HEADING, 1)[1][:-1] for prompt in _corrections(prompts))


def _summary(llm, text="source" * 100, **kwargs):
    """Draft and correct one exact source as one Main room of one message (identity resident)."""
    return rc.summarize_source(
        c._light_call(llm, None, {}), text, [],
        lambda part, note: rc.room_draft_prompt(
            part, room_label="Main", block_range_text=_RANGE, message_count=1,
            identity_text="identity", continuation_note=note),
        lambda draft, part, note: rc.correction_prompt(
            draft, part, room_label="Main", scope="dialogue block " + _RANGE,
            identity_text="identity", continuation_note=note),
        **kwargs)


def _prompt_tokens(source, *, identity_text="", message_count=1,
                   first_ts="2026-01-01T01:00", last_ts="2026-01-01T02:00",
                   continuation_note=""):
    prompt = rc.room_draft_prompt(
        source, room_label="Main", block_range_text=rc.block_range(first_ts, last_ts),
        message_count=message_count, identity_text=identity_text, continuation_note=continuation_note,
    )
    return context_fit.estimate_context_prompt_tokens(
        [{"role": "user", "content": prompt}], None,
    )


def _window_for_split(source, *, identity_text="", message_count=1,
                      first_ts="2026-01-01T01:00", last_ts="2026-01-01T02:00",
                      density=1.0, continuation_note="", fraction=0.66):
    """Build a synthetic route window from the real fixed prompt and source.

    Consolidation's output reserve remains the production 16,384 tokens. The
    fixture capacity is derived from the current prompt prefix and a fraction
    of the variable source, so adding attribution guidance cannot make the
    test accidentally exercise an impossible route.
    """
    fixed = _prompt_tokens(
        "", identity_text=identity_text, message_count=message_count,
        first_ts=first_ts, last_ts=last_ts, continuation_note=continuation_note,
    )
    full = _prompt_tokens(
        source, identity_text=identity_text, message_count=message_count,
        first_ts=first_ts, last_ts=last_ts, continuation_note=continuation_note,
    )
    assert full > fixed
    split_capacity = fixed + (full - fixed) * fraction
    return LIGHT_OUTPUT_RESERVE + ceil(split_capacity * density)


def _source_for_split(*, identity_text="", message_count=1,
                      first_ts="2026-01-01T01:00", last_ts="2026-01-01T02:00",
                      multiplier=12):
    """Create variable source whose size follows the current fixed prefix."""
    unit = "complete entry Ж🙂 "
    fixed = _prompt_tokens(
        "", identity_text=identity_text, message_count=message_count,
        first_ts=first_ts, last_ts=last_ts,
    )
    source = unit
    while _prompt_tokens(
        source, identity_text=identity_text, message_count=message_count,
        first_ts=first_ts, last_ts=last_ts,
    ) < fixed * multiplier:
        source += source
    return source


@pytest.mark.parametrize("code", ["provider_failed", "invalid_request"])
def test_oversized_logical_block_splits_complete_source_and_advances_once(tmp_path, fit, monkeypatch, code):
    from ouroboros.llm_claudexor import ClaudexorModelError
    fit.window = None
    chat, blocks, meta = _paths(tmp_path)
    rows = _write_chat(chat, text_size=120)
    source_bytes = chat.read_bytes()
    def reject(llm, prompt):
        if len(prompt.encode("utf-8")) > llm.limit:
            raise ClaudexorModelError({"code": code, "message": "Controlled provider refusal",
                "context": {"httpStatus": 400, "vendorCode": "context_length_exceeded", "parameter": "input"}})
    llm = _LLM(limit=3500, effect=reject)
    advances = []
    advance = c._advance_cursor
    monkeypatch.setattr(c, "_advance_cursor", lambda *args: (advances.append(args[-1]), advance(*args))[-1])

    usage = c.consolidate(chat, blocks, meta, llm)

    assert usage["cost"] is None  # refusal did not report cash
    assert _source(llm.accepted) == c._format_entries_for_block(rows, include_room_labels=True)
    assert chat.read_bytes() == source_bytes
    assert advances == [100]
    saved = json.loads(blocks.read_text())
    assert len(saved) == 1 and saved[0]["message_count"] == 100
    assert json.loads(meta.read_text())["last_consolidated_offset"] == 100
    assert not c.should_consolidate(meta, chat)
    refused = usage["_consolidation_errors"][0]
    assert refused["kind"] == "context_overflow" and not refused["preflight_only"]
    assert refused["resolution"] == "split"  # history kept; the post-task adapter reads it as answered
    # A refusal that was split and then fully summarized is a recovered attempt, not a
    # failed run: the block was written, so no stale error may outlive the advance.
    assert "last_consolidation_error" not in json.loads(meta.read_text())
    assert refused["capacity_tokens"] is None and refused["input_limit"] is None
    sizes = [len(call["messages"][0]["content"].encode("utf-8")) for call in llm.calls]
    for index, size in enumerate(sizes[:-1]):
        if size > llm.limit:
            assert sizes[index + 1] < size


def test_known_capacity_includes_whole_prompt_density_and_output_reserve(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    rows = _write_chat(chat, text_size=140)
    identity = "identity at full length " * 20
    spans = []
    formatted = c._format_entries_for_block(rows, include_room_labels=True, source_spans=spans)
    # Account for the continuation attribution that can appear after a split,
    # as well as the ordinary fixed prefix. This keeps the synthetic route
    # large enough for both while retaining a variable source budget.
    from ouroboros.dialogue_provenance import source_continuation_note
    continuation = source_continuation_note(
        spans, spans[len(spans) // 2][0] + 1, spans[len(spans) // 2][0] + 2,
    )
    fit.density = 2.5
    fit.window = _window_for_split(
        formatted, identity_text=identity, message_count=len(rows),
        first_ts=rows[0]["ts"], last_ts=rows[-1]["ts"], density=fit.density,
        continuation_note=continuation,
    )
    llm = _LLM()

    result = c.consolidate(chat, blocks, meta, llm, identity)

    assert len(llm.calls) > 1
    for call in llm.calls:
        assert identity in call["messages"][0]["content"]
        size = ceil(context_fit.estimate_context_prompt_tokens(call["messages"], call["tools"]) * fit.density)
        assert size + call["max_tokens"] <= fit.window
        assert call["model_role"] == "light" and call["max_tokens"] == 16384
    assert _source(llm.accepted) == c._format_entries_for_block(rows, include_room_labels=True)
    assert result["cost"] == pytest.approx(0.01 * len(llm.calls))


def test_known_capacity_overhead_refusal_is_typed_without_model_call(tmp_path, fit):
    fit.window = 16384
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat)
    original = chat.read_bytes()
    c.atomic_write_json(meta, {"last_consolidated_offset": 0, "chat_log_signature": c._chat_log_signature(chat)})
    cursor = json.loads(meta.read_text())["chat_log_signature"]
    llm = _LLM()
    for _ in range(2):
        usage = c.consolidate(chat, blocks, meta, llm, "large identity" * 300)
        assert usage["_consolidation_errors"][-1]["kind"] == "context_overflow"
        assert usage["cost"] == 0  # proven local preflight, no request
    saved = json.loads(meta.read_text())
    assert saved["chat_log_signature"] == cursor and saved["last_consolidated_offset"] == 0
    assert saved["last_consolidation_error"]["preflight_only"]
    assert not llm.calls and not blocks.exists() and chat.read_bytes() == original


def test_unknown_capacity_impossible_overhead_does_not_replay_next_cycle(tmp_path, fit):
    fit.window = None
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat)
    llm = _LLM(limit=1)
    usage = c.consolidate(chat, blocks, meta, llm)
    first_calls = len(llm.calls)
    assert first_calls > 0
    assert not usage["_consolidation_errors"][-1].get("resolution")  # the unsplittable refusal stays unresolved
    c.consolidate(chat, blocks, meta, llm)
    assert len(llm.calls) == first_calls
    assert not blocks.exists()
    assert json.loads(meta.read_text()).get("last_consolidated_offset", 0) == 0
    # A new route capacity is new evidence; the old refusal must not trap it.
    fit.window, llm.limit = 100_000, None
    c.consolidate(chat, blocks, meta, llm)
    assert json.loads(meta.read_text())["last_consolidated_offset"] == 100


def test_route_capacity_changes_split_shape_without_provider_branch(tmp_path, fit):
    counts = []
    for window in (100_000, 17_000):
        fit.window = window
        chat, blocks, meta = _paths(tmp_path / str(window))
        _write_chat(chat)
        llm = _LLM()
        c.consolidate(chat, blocks, meta, llm)
        assert json.loads(meta.read_text())["last_consolidated_offset"] == 100
        counts.append(len(llm.calls))
    assert counts[0] == 2 < counts[1]  # one draft and one correction, then split parts


def test_single_large_entry_is_lossless_even_without_line_boundaries(fit):
    fit.window = 17000
    source = "no-newline-🙂Ж-end" * 2000
    llm = _LLM()
    content, _ = _summary(llm, source)
    assert content and len(llm.calls) > 2
    # A correction whose fixed prefix overflowed re-drafts its halves, so the
    # DRAFT sources may cover a part twice; the CORRECTED sources — what the
    # published text was checked against — cover the source exactly once.
    assert _corrected_source(llm.accepted) == source


@pytest.mark.parametrize("stale", [False, True])
def test_unknown_or_stale_capacity_gets_one_ordinary_call(fit, stale):
    fit.window, fit.stale = (1 if stale else None), stale
    llm = _LLM()
    assert _summary(llm)[0]
    assert len(llm.calls) == 2  # one unchecked draft, one unchecked correction; no split
    assert _drafts(llm.accepted) == llm.accepted[:1] and _corrections(llm.accepted) == llm.accepted[1:]


@pytest.mark.parametrize("first_success", [False, True])
def test_partial_split_failure_stops_before_siblings_and_preserves_cursor(tmp_path, fit, first_success):
    fit.window = 17000
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat)
    before = chat.read_bytes()
    failure_at = 2 if first_success else 1

    def fail(llm, prompt):
        if len(llm.calls) == failure_at:
            raise _Refusal("provider refused", code="invalid_api_key")
    llm = _LLM(effect=fail)
    usage = c.consolidate(chat, blocks, meta, llm)
    assert len(llm.calls) == failure_at and len(llm.accepted) == failure_at - 1
    assert usage["cost"] is None
    assert usage["_consolidation_errors"][-1]["kind"] == "auth_error"
    assert not blocks.exists() and chat.read_bytes() == before
    assert json.loads(meta.read_text()).get("last_consolidated_offset", 0) == 0


@pytest.mark.parametrize("raw", [None, "", " \n"])
def test_empty_output_is_non_success_with_real_usage(tmp_path, fit, raw):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat)
    llm = _LLM(effect=lambda *_: ({"content": raw}, {"cost": 0.02}))
    result = c.consolidate(chat, blocks, meta, llm)
    assert result["cost"] == 0.02
    assert result["_consolidation_errors"][-1]["kind"] == "empty_summary"
    assert not blocks.exists() and len(llm.calls) == 1
    assert json.loads(meta.read_text()).get("last_consolidated_offset", 0) == 0


@pytest.mark.parametrize("code", ["auth_required", "subscription_window_exhausted", "model_operation_interrupted", "model_outcome_unknown"])
def test_control_resource_and_unknown_model_errors_still_propagate(tmp_path, fit, code):
    from ouroboros.llm_claudexor import ClaudexorModelError
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat)
    error = ClaudexorModelError({"code": code, "message": "context length exceeded"})
    def fail(*_):
        raise error
    llm = _LLM(effect=fail)
    with pytest.raises(ClaudexorModelError) as caught:
        c.consolidate(chat, blocks, meta, llm)
    assert caught.value is error and len(llm.calls) == 1
    assert not blocks.exists() and not meta.exists()


def test_wait_interruption_propagates_after_a_successful_part(tmp_path, fit):
    from ouroboros.model_wait import ModelWaitInterrupted
    fit.window = 17000
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat)
    def fail(llm, _):
        if len(llm.calls) == 2:
            raise ModelWaitInterrupted("cancelled", role="light")
    llm = _LLM(effect=fail)
    with pytest.raises(ModelWaitInterrupted):
        c.consolidate(chat, blocks, meta, llm)
    assert len(llm.calls) == 2 and len(llm.accepted) == 1
    assert not blocks.exists() and not meta.exists()


@pytest.mark.parametrize("unresolved", [False, True])
@pytest.mark.parametrize("code", ["provider_failed", "invalid_request"])
def test_confirmed_model_context_refusal_splits_but_unknown_custody_propagates(fit, unresolved, code):
    from ouroboros.llm_claudexor import ClaudexorModelError
    fit.window = None
    error = ClaudexorModelError({"code": code, "message": "Controlled provider refusal",
        "context": {"httpStatus": 400, "vendorCode": "context_length_exceeded", "parameter": "input"}})
    error.physical_attempt_capture = SimpleNamespace(state="unresolved" if unresolved else "settled")
    def refuse_once(llm, _):
        if len(llm.calls) == 1:
            raise error
    llm = _LLM(effect=refuse_once)
    if unresolved:
        with pytest.raises(ClaudexorModelError):
            _summary(llm)
        assert len(llm.calls) == 1
    else:
        content, usage = _summary(llm)
        # The refusal, then each half drafted and corrected against its own bytes.
        assert content and len(llm.calls) == 5 and usage["cost"] is None
        assert _source(llm.accepted) == _corrected_source(llm.accepted) == "source" * 100


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


def test_refused_attempt_usage_is_merged_with_successful_parts(fit):
    fit.window = None
    def refuse_once(llm, _):
        if len(llm.calls) == 1:
            error = _Refusal(usage={"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7, "cost": 0.03})
            error.ledger_attempt_ids = ["refused-attempt"]
            raise error
    llm = _LLM(effect=refuse_once)
    content, usage = _summary(llm)
    assert content and len(llm.calls) == 5  # refusal + (draft, correction) per half
    assert usage["cost"] == pytest.approx(0.07)
    assert usage["prompt_tokens"] == 47 and usage["total_tokens"] == 67
    assert usage["ledger_attempt_ids"] == ["refused-attempt"]


def test_rotation_append_and_partial_failure_only_advance_completed_chunks(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    rows = _write_chat(chat, count=200, text_size=0)
    rows[100]["text"] = "large source🙂" * 3000
    chat.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    captured = c._chat_log_signature(chat)
    archive = tmp_path / "archive" / "chat_20260101.jsonl"

    def rotate_and_fail(llm, _):
        if len(llm.calls) == 1:
            archive.parent.mkdir()
            chat.rename(archive)
            _write_chat(chat, 100, 0, start=200)
        if len(llm.calls) == 3:
            with chat.open("a") as output:
                output.write(json.dumps({"ts": "2026-01-02T00:00:00Z", "text": "appended tail"}) + "\n")
            raise _Refusal("failed part", code="invalid_request")
    # Chunk 0 (100 short rows) fits one draft + one correction; chunk 1 carries
    # the oversized row, so its first draft is call 3 and fails.
    fit.window = LIGHT_OUTPUT_RESERVE + 6000
    llm = _LLM(effect=rotate_and_fail)
    usage = c.consolidate(chat, blocks, meta, llm)
    assert len(llm.calls) == 3 and usage["cost"] is None
    saved = json.loads(meta.read_text())
    # Chunk 0 (draft + correction) is a complete unit and stays published; the
    # failed chunk 1 is withheld and recorded as this run's own error.
    assert saved["last_consolidated_offset"] == 100 and saved["chat_log_signature"] == captured
    assert saved["last_consolidation_error"]["cursor_offset"] == 100
    assert sum(block["message_count"] for block in json.loads(blocks.read_text())) == 100
    assert c.should_consolidate(meta, chat)

    succeeding = _LLM()
    c.consolidate(chat, blocks, meta, succeeding)
    # Corrected coverage is the invariant: a correction whose prefix overflowed
    # re-drafts its halves, so draft sources may repeat a part.
    assert _corrected_source(succeeding.accepted) == c._format_entries_for_block(rows[100:], include_room_labels=True) + c._format_entries_for_block(c._read_chat_entries(chat)[:100], include_room_labels=True)
    saved = json.loads(meta.read_text())
    assert saved["last_consolidated_offset"] == 100
    assert saved["chat_log_signature"]["first_line_sha256"] == c._chat_log_signature(chat)["first_line_sha256"]
    assert c._read_chat_entries(chat)[saved["last_consolidated_offset"]:][0]["text"] == "appended tail"
    assert "last_consolidation_error" not in saved  # the successful retry retired the stale error


@pytest.mark.parametrize("failing_call", [3, 4])  # era compression, era correction
def test_failed_era_usage_is_accounted_and_original_blocks_survive(tmp_path, fit, failing_call):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, text_size=0)
    originals = [{"range": "2025-01-01", "type": "summary", "message_count": 100, "content": f"old-{i}"} for i in range(10)]
    c.atomic_write_json(blocks, originals)
    def fail_era(llm, _):
        if len(llm.calls) == failing_call:
            return {"content": ""}, {"cost": 0.04}
    llm = _LLM(effect=fail_era)
    usage = c.consolidate(chat, blocks, meta, llm)
    assert len(llm.calls) == failing_call  # the block's draft and correction, then the failed era stage
    assert usage["cost"] == pytest.approx(0.01 * (failing_call - 1) + 0.04)
    assert json.loads(blocks.read_text())[:10] == originals
    assert json.loads(meta.read_text())["last_consolidated_offset"] == 100


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
    text = "full source " * 1000
    content, _ = _summary(llm, text)

    assert content and len(llm.calls) > 1
    assert all(call["model_account_override"] == "light-account" for call in llm.calls)
    assert all(call["model"] == model and call["model_role"] == "light" for call in llm.calls)
    assert all(p["options"]["credential_profile_id"] == "light-account" for p in probes)
    assert all(p["provider"] == "claudexor" and p["allow_fetch"] is True for p in probes)
    assert all(context_fit.estimate_context_prompt_tokens(call["messages"]) + 16384 <= 17000 for call in llm.calls)
    assert _corrected_source(llm.accepted) == text


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
    # The wait's reprepare re-measured the draft under the observed account;
    # the correction call that follows starts from the ordinary route again.
    assert fit.tasks[1]["model_route"] == {"credentialProfileId": "changed-pin"}
    assert len(llm.calls) == 2 and len(fit.tasks) == 3


def test_retry_limit_survives_density_changes_and_source_changes_release_it(tmp_path, fit):
    fit.window = None
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, text_size=0)
    llm = _LLM(limit=1)
    c.consolidate(chat, blocks, meta, llm)
    calls = len(llm.calls)
    fit.density = 4.5
    c.consolidate(chat, blocks, meta, llm)
    assert len(llm.calls) == calls
    c.consolidate(chat, blocks, meta, llm, identity_text="new identity context")
    assert len(llm.calls) > calls


def test_complete_blocks_are_preserved_if_a_summary_write_fails(tmp_path, fit, monkeypatch):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat)
    def fail_write(*_):
        raise OSError("disk unavailable")
    monkeypatch.setattr(c, "_write_locked_json", fail_write)
    with pytest.raises(OSError):
        c.consolidate(chat, blocks, meta, _LLM())
    assert not meta.exists()  # successful inference is not durable cursor progress


@pytest.mark.parametrize("failing_call", [3, 4])  # second block's draft, second block's correction
@pytest.mark.parametrize("unknown", [False, True])
def test_partial_block_failure_does_not_start_era_work(tmp_path, fit, unknown, failing_call):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, count=200, text_size=0)
    originals = [{"range": "2025-01-01", "type": "summary", "message_count": 100, "content": f"old-{i}"} for i in range(10)]
    c.atomic_write_json(blocks, originals)
    def fail_second(llm, _):
        if len(llm.calls) == failing_call:
            error = _Refusal("unknown" if unknown else "auth failed", code="invalid_api_key")
            if unknown:
                error.physical_attempt_capture = SimpleNamespace(state="unresolved")
            raise error
    llm = _LLM(effect=fail_second)
    usage = c.consolidate(chat, blocks, meta, llm)
    assert len(llm.calls) == failing_call
    assert json.loads(blocks.read_text())[:10] == originals
    # The transaction boundary is the logical chunk: the complete first chunk
    # stays published, the failed second chunk (draft or correction) is withheld.
    assert len(json.loads(blocks.read_text())) == 11
    saved = json.loads(meta.read_text())
    assert saved["last_consolidated_offset"] == 100
    # The successful PREFIX advanced the cursor, but the failed suffix is this run's
    # OWN fresh error: advancing must not clear the very failure just recorded.
    assert saved["last_consolidation_error"]["cursor_offset"] == 100
    assert usage["_blocks_written"] == 1


def test_unavailable_capacity_reader_retains_ordinary_call(monkeypatch):
    from ouroboros import capability_evidence, config
    monkeypatch.setattr(c, "_consolidation_route", lambda: ("test/model", False))
    monkeypatch.setattr(config, "load_settings", lambda: {"OUROBOROS_MODEL": "test/model"})
    def unavailable(*args, **kwargs):
        raise OSError("catalog unavailable")
    monkeypatch.setattr(capability_evidence, "probe", unavailable)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_: 1.0)
    llm = _LLM()
    assert _summary(llm)[0] and len(llm.calls) == 2


@pytest.mark.parametrize("preceding_blocks", [0, 1])
@pytest.mark.parametrize("code", ["provider_failed", "invalid_request"])
def test_refusal_bound_survives_preceding_logical_blocks(tmp_path, fit, preceding_blocks, code):
    """Earlier success cannot consume a later unpublished block's refusal."""
    from ouroboros.llm_claudexor import ClaudexorModelError
    from ouroboros.model_wait import ModelWaitInterrupted

    fit.window = None
    chat, blocks, meta = _paths(tmp_path)
    count = 100 * (preceding_blocks + 1)
    _write_chat(chat, count=count, text_size=5)
    raw = chat.read_bytes()
    interruption = ModelWaitInterrupted("deadline", role="light")

    target_draft = 2 * preceding_blocks + 1  # every published block costs a draft and a correction

    def first_cycle(llm, prompt):
        if len(llm.calls) == target_draft:
            error = ClaudexorModelError({"code": code, "message": "Controlled provider refusal",
                "context": {"httpStatus": 400, "vendorCode": "context_length_exceeded", "parameter": "input"}})
            error.physical_attempt_capture = SimpleNamespace(state="settled")
            raise error
        if len(llm.calls) > target_draft:
            raise interruption

    first = _LLM(effect=first_cycle)
    with pytest.raises(ModelWaitInterrupted) as caught:
        c.consolidate(chat, blocks, meta, first)
    assert caught.value is interruption
    rejected = first.calls[target_draft - 1]["messages"][0]["content"]
    saved = json.loads(meta.read_text())
    assert saved["consolidation_retry"]["input_limit"]["input_bytes"] == len(rejected.encode()) - 1
    assert saved.get("last_consolidated_offset", 0) == 0
    assert not blocks.exists() and chat.read_bytes() == raw

    second = _LLM()
    c.consolidate(chat, blocks, meta, second)
    next_prompt = second.calls[target_draft - 1]["messages"][0]["content"]
    assert len(next_prompt.encode()) < len(rejected.encode())
    assert chat.read_bytes() == raw
    final = json.loads(meta.read_text())
    assert final["last_consolidated_offset"] == count
    assert "consolidation_retry" not in final


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


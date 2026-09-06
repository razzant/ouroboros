from __future__ import annotations

import copy
import hashlib
import json
import queue
from collections import deque
from types import SimpleNamespace

import pytest

from tests.test_delivery_forced_finalization import _forced_test_context


def _start_control_episode(
    tmp_path, text="Controlled complete answer.", *, incoming=None,
):
    loop, registry, ctx, trace = _forced_test_context(tmp_path, incoming=incoming)
    initial = loop._replace_delivery_candidate(
        registry, ctx, trace, "Initial complete answer.", control="candidate",
    )
    assert initial.control_episode_seen is False
    loop._arm_delivery_control(registry, ctx, trace)
    assert initial.control_episode_seen is True
    status, resolved = loop._resolve_delivery_control(
        json.dumps({"delivery_control": "replace", "full_answer": text}),
        registry,
        ctx,
        trace,
    )
    candidate = registry._ctx._delivery_candidate
    assert (status, resolved) == ("resolved", text)
    assert candidate.control_episode_seen is True
    assert trace["delivery_candidate"]["control_episode_seen"] is True
    assert registry._ctx._delivery_control_required is False
    return loop, registry, ctx, trace, candidate


@pytest.mark.parametrize(
    "raw",
    [
        '{"delivery_control":"publish"}',
        '{"delivery_control":"keep","delivery_control":"replace"}',
        '{"delivery_control":"replace","full_answer":""}',
        '{"delivery_control":"replace","full_answer":7}',
        '{"delivery_control":"replace","full_answer":"one","full_answer":"two"}',
        '{"delivery_control":"keep","extra":true}',
        '```json\n{"delivery_control":"publish"}\n```',
    ],
)
def test_post_episode_invalid_whole_body_preserves_without_repair(tmp_path, raw):
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    before_messages = copy.deepcopy(ctx.messages)
    before_binding = copy.deepcopy(candidate.acceptance_binding)
    before_revision = candidate.revision
    before_hash = candidate.content_sha256

    status, text = loop._resolve_delivery_control(raw, registry, ctx, trace)

    assert (status, text) == ("resolved", candidate.full_text)
    assert ctx.messages == before_messages
    assert candidate.repair_attempted is False
    assert candidate.revision == before_revision
    assert candidate.content_sha256 == before_hash
    assert candidate.acceptance_binding == before_binding
    assert registry._ctx._delivery_control_required is False
    assert raw not in text

    forced, reason, retained, replaced = loop._resolve_forced_delivery_control(
        registry._ctx, raw,
    )
    assert (forced, reason, retained, replaced) == (
        candidate.full_text, "", True, False,
    )
    assert ctx.messages == before_messages


def test_repair_prompt_starts_control_episode_lineage(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(
        registry, ctx, trace, "Initial complete answer.", control="candidate",
    )
    candidate.finalization_control = loop._SKILL_ACTION_HOLD_CONTROL

    assert candidate.control_episode_seen is False
    assert loop._resolve_delivery_control(
        '{"delivery_control":"keep"}', registry, ctx, trace,
    ) == ("retry", "")
    assert candidate.control_episode_seen is True
    assert any(
        "[DELIVERY_CONTROL_REPAIR]" in str(message.get("content") or "")
        for message in ctx.messages
    )

    repaired_text = "Repaired complete answer."
    assert loop._resolve_delivery_control(
        json.dumps({
            "delivery_control": "replace",
            "full_answer": repaired_text,
        }),
        registry,
        ctx,
        trace,
    ) == ("resolved", repaired_text)
    repaired = registry._ctx._delivery_candidate
    before_messages = copy.deepcopy(ctx.messages)

    assert repaired.control_episode_seen is True
    assert loop._resolve_delivery_control(
        '{"delivery_control":"publish"}', registry, ctx, trace,
    ) == ("resolved", repaired_text)
    assert ctx.messages == before_messages


def test_forced_resolver_ignores_non_candidate_state(tmp_path):
    loop, registry, _ctx, _trace = _forced_test_context(tmp_path)
    registry._ctx._delivery_candidate = object()

    assert loop._resolve_forced_delivery_control(
        registry._ctx, "Plain complete answer.",
    ) == ("Plain complete answer.", "", False, False)


@pytest.mark.parametrize(
    "raw",
    [
        '{"delivery_control":"keep"}',
        '{"delivery_control":"replace","full_answer":"JSON document"}',
        '{"delivery_control":"publish"}',
        ' \n{"delivery_control":"keep"}\n ',
        '{"other":1,"other":2}',
        '{"payload":{"other":1,"other":2}}',
    ],
)
def test_no_episode_protocol_json_is_byte_exact_passthrough(tmp_path, raw):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(
        registry, ctx, trace, "Existing answer.", control="candidate",
    )
    before = (candidate.revision, candidate.content_sha256, copy.deepcopy(candidate.acceptance_binding))

    assert candidate.control_episode_seen is False
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("fresh", raw)
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        raw, "", False, False,
    )
    assert (candidate.revision, candidate.content_sha256, candidate.acceptance_binding) == before


@pytest.mark.parametrize(
    "raw",
    [
        'Docs quote {"delivery_control":"keep"} before continuing.',
        'Final prose.\n{"delivery_control":"keep"}',
        '{"delivery_control":"replace","full_answer":"cut',
        '{"payload":{"delivery_control":"keep","delivery_control":"replace"}}',
        '{"full_answer":"one","full_answer":"two"}',
        '{"other":1,"other":2}',
        '{"payload":{"other":1,"other":2}}',
    ],
)
def test_post_episode_latch_off_residuals_stay_byte_exact(tmp_path, raw):
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    before_messages = copy.deepcopy(ctx.messages)
    before = (candidate.revision, candidate.content_sha256, copy.deepcopy(candidate.acceptance_binding))

    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("fresh", raw)
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        raw, "", False, False,
    )
    assert ctx.messages == before_messages
    assert (candidate.revision, candidate.content_sha256, candidate.acceptance_binding) == before


def test_duplicate_full_answer_without_verb_preserves_stronger_control_rails(
    tmp_path,
):
    raw = '{"full_answer":"one","full_answer":"two"}'
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    parsed, duplicate, embedded = loop._parse_delivery_control_body(raw)

    assert parsed == {"full_answer": "two"}
    assert getattr(parsed, "duplicate_keys", set()) == {"full_answer"}
    assert (duplicate, embedded) == (False, False)
    assert loop._classify_parsed_delivery_control(
        parsed, duplicate, embedded,
    )[0] == "rail_invalid"

    candidate.finalization_control = "owner_revision_required"
    registry._ctx._delivery_control_required = False
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", "")
    assert candidate.repair_attempted is True
    assert any(
        "[DELIVERY_CONTROL_REPAIR]" in str(message.get("content") or "")
        for message in ctx.messages
    )

    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    candidate.finalization_control = loop._SKILL_ACTION_HOLD_CONTROL
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", "")
    assert candidate.finalization_control.startswith("skill_revision_required")

    loop, registry, _ctx, _trace, candidate = _start_control_episode(tmp_path)
    registry._ctx._delivery_control_required = True
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        candidate.full_text, loop.REASON_DELIVERY_CONTROL_DEGRADED, True, False,
    )


def test_arbitrary_duplicates_remain_prose_on_ordinary_unarmed_rails(tmp_path):
    raw = '{"other":1,"other":2}'
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    candidate.finalization_control = "owner_revision_required"
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("fresh", raw)

    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    candidate.finalization_control = loop._SKILL_ACTION_HOLD_CONTROL
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("fresh", raw)

    loop, registry, _ctx, _trace, candidate = _start_control_episode(tmp_path)
    candidate.finalization_control = "skill_revision_required"
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        candidate.full_text, loop.REASON_DELIVERY_CONTROL_DEGRADED, True, False,
    )


@pytest.mark.parametrize(
    ("raw", "expected", "retained", "replaced"),
    [
        ('{"delivery_control":"keep"}', "Free-form improvement.", True, False),
        (
            '```json\n{"delivery_control":"replace",'
            '"full_answer":"Forced **replacement**\\nline two"}\n```',
            "Forced **replacement**\nline two",
            False,
            True,
        ),
    ],
)
def test_forced_latch_off_after_episode_resolves_stale_control(
    tmp_path, raw, expected, retained, replaced,
):
    loop, registry, ctx, trace, _candidate = _start_control_episode(tmp_path)
    candidate = loop._replace_delivery_candidate(
        registry, ctx, trace, "Free-form improvement.", control="candidate",
    )

    assert candidate.control_episode_seen is True
    assert registry._ctx._delivery_control_required is False
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        expected, "", retained, replaced,
    )
    assert raw not in expected


@pytest.mark.parametrize(
    ("raw", "response_meta", "replacement", "degraded_reason"),
    [
        (
            '{"delivery_control":"replace","full_answer":"decoded answer"}',
            {"finish_reason_present": True, "finish_reason": "length", "tool_call_count": 0},
            "decoded answer",
            "provider_terminal",
        ),
        (
            '{"delivery_control":"keep"}',
            {"finish_reason_present": True, "finish_reason": None, "tool_call_count": 0},
            None,
            "provider_terminal",
        ),
        (
            '{"delivery_control":"publish"}',
            {"finish_reason_present": False, "finish_reason": None, "tool_call_count": 1},
            None,
            "provider_terminal",
        ),
    ],
)
def test_provider_terminal_incomplete_control_resolves_before_fallback(
    tmp_path, monkeypatch, raw, response_meta, replacement, degraded_reason,
):
    loop, registry, ctx, trace, retained = _start_control_episode(tmp_path)
    trace["tool_calls"].append({
        "tool": "write_file",
        "status": "ok",
        "result": "new evidence",
        "is_error": False,
    })
    monkeypatch.setattr(
        loop,
        "_call_forced_model_once",
        lambda _ctx: (raw, response_meta),
    )

    text, usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="fallback",
        reason_code="provider_terminal",
        provider_terminal=True,
    )

    published = registry._ctx._delivery_candidate
    assert raw not in text
    assert usage["execution_status"] == "failed"
    assert usage["reason_code"] == "provider_terminal"
    assert published.degraded_reason == degraded_reason
    if replacement is not None:
        assert text == replacement
        assert published is not retained
        assert "STALE-EVIDENCE NOTICE" not in text
        assert returned_trace["delivery_candidate"]["evidence_current"] is True
        assert returned_trace["forced_finalization"]["source"] == "model"
    else:
        assert text.startswith(retained.full_text)
        assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in text
        assert published.acceptance_binding["stale_evidence"] is True
        assert returned_trace["delivery_candidate"]["evidence_current"] is False


@pytest.mark.parametrize(
    "response_meta",
    [
        {"finish_reason_present": True, "finish_reason": "length", "tool_call_count": 0},
        {"finish_reason_present": True, "finish_reason": None, "tool_call_count": 0},
        {"finish_reason_present": False, "finish_reason": None, "tool_call_count": 1},
    ],
)
def test_provider_terminal_incomplete_replace_prefers_current_candidate(
    tmp_path, monkeypatch, response_meta,
):
    loop, registry, ctx, trace, current = _start_control_episode(tmp_path)
    loop._arm_delivery_control(registry, ctx, trace)
    raw = json.dumps({
        "delivery_control": "replace",
        "full_answer": "decoded replacement",
    })
    monkeypatch.setattr(
        loop,
        "_call_forced_model_once",
        lambda _ctx: (raw, response_meta),
    )

    text, usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="fallback",
        reason_code="provider_terminal",
        provider_terminal=True,
    )

    assert text == current.full_text
    assert raw not in text
    assert registry._ctx._delivery_candidate is current
    assert registry._ctx._delivery_control_required is False
    assert usage["execution_status"] == "failed"
    assert usage["reason_code"] == "provider_terminal"
    assert returned_trace["forced_finalization"]["source"] == "retained_candidate"


@pytest.mark.parametrize(
    ("raw", "response_meta", "host_suffix", "provider_unavailable"),
    [
        pytest.param(
            '{"delivery_control":"publish"}',
            {"finish_reason_present": False, "finish_reason": None, "tool_call_count": 1},
            False, False, id="plain-current",
        ),
        *[
            pytest.param(raw, meta, True, False, id=f"host-{shape}-{finish}")
            for shape, raw in (
                ("unknown", '{"delivery_control":"publish"}'),
                ("duplicate", '{"delivery_control":"keep","delivery_control":"replace"}'),
            )
            for finish, meta in (
                ("length", {"finish_reason_present": True, "finish_reason": "length", "tool_call_count": 0}),
                ("null", {"finish_reason_present": True, "finish_reason": None, "tool_call_count": 0}),
                ("tool", {"finish_reason_present": False, "finish_reason": None, "tool_call_count": 1}),
            )
        ],
        pytest.param(
            '{"delivery_control":"publish"}',
            {"finish_reason_present": True, "finish_reason": "length", "tool_call_count": 0},
            True, True, id="provider-unavailable",
        ),
    ],
)
def test_incomplete_armed_invalid_keeps_typed_reason(
    tmp_path, monkeypatch, raw, response_meta, host_suffix, provider_unavailable,
):
    if host_suffix:
        loop, registry, ctx, trace = _forced_test_context(tmp_path)
        retained = loop._replace_delivery_candidate(
            registry, ctx, trace, "Answer\nHOST FACT",
            control="host_suffix", model_text="Answer",
        )
    else:
        loop, registry, ctx, trace, retained = _start_control_episode(tmp_path)
    loop._arm_delivery_control(registry, ctx, trace)
    monkeypatch.setattr(
        loop,
        "_call_forced_model_once",
        lambda _ctx: (raw, response_meta),
    )

    if provider_unavailable:
        text, usage, returned_trace = loop._handle_provider_unavailable(ctx)
        external_reason, execution_status = "provider_unavailable", "infra_failed"
    else:
        text, usage, returned_trace = loop._forced_final_answer(
            ctx, prompt="finalize", fallback_text="fallback",
            reason_code="provider_terminal", provider_terminal=True,
        )
        external_reason, execution_status = "provider_terminal", "failed"

    published = registry._ctx._delivery_candidate
    assert text == ("Answer" if host_suffix else retained.full_text)
    assert raw not in text
    assert published.degraded_reason == loop.REASON_DELIVERY_CONTROL_DEGRADED
    assert registry._ctx._delivery_control_required is False
    assert usage["execution_status"] == execution_status
    assert usage["reason_code"] == external_reason
    assert usage["terminal_origin"] == loop.TERMINAL_ORIGIN_MODEL_FINAL
    assert returned_trace["delivery_candidate"]["degraded_reason"] == (
        loop.REASON_DELIVERY_CONTROL_DEGRADED
    )
    if host_suffix:
        assert published is not retained
        assert published.full_text == published.model_text == text
        assert published.finalization_control == f"forced_replace:{external_reason}"
        assert published.acceptance_binding["degraded_reason"] == external_reason
        assert returned_trace["forced_finalization"]["source"] == (
            "retained_candidate_with_host_suffix"
        )
    else:
        assert published is retained
        assert returned_trace["forced_finalization"]["source"] == "retained_candidate"


@pytest.mark.parametrize(
    "raw",
    [
        '{"delivery_control":"publish"}',
        '{"delivery_control":"keep","delivery_control":"replace"}',
    ],
)
@pytest.mark.parametrize(
    "response_meta",
    [
        {"finish_reason_present": True, "finish_reason": "length", "tool_call_count": 0},
        {"finish_reason_present": True, "finish_reason": None, "tool_call_count": 0},
        {"finish_reason_present": False, "finish_reason": None, "tool_call_count": 1},
    ],
)
def test_provider_terminal_incomplete_armed_invalid_without_candidate_uses_fallback(
    tmp_path, monkeypatch, raw, response_meta,
):
    loop, registry, ctx, _trace = _forced_test_context(tmp_path)
    registry._ctx._delivery_control_required = True
    fallback = "Provider returned no usable complete answer."
    monkeypatch.setattr(
        loop,
        "_call_forced_model_once",
        lambda _ctx: (raw, response_meta),
    )

    text, usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text=fallback,
        reason_code="provider_terminal",
        provider_terminal=True,
    )

    assert text == fallback
    assert text
    assert raw not in text
    assert registry._ctx._delivery_candidate.full_text == fallback
    assert usage["execution_status"] == "failed"
    assert usage["reason_code"] == "provider_terminal"
    assert usage["terminal_origin"] == loop.TERMINAL_ORIGIN_HOST_SALVAGE
    assert returned_trace["forced_finalization"]["source"] == "forced_model_incomplete"


def test_provider_terminal_incomplete_equal_text_replace_is_fresh(
    tmp_path, monkeypatch,
):
    loop, registry, ctx, trace, retained = _start_control_episode(tmp_path)
    trace["tool_calls"].append({
        "tool": "write_file",
        "status": "ok",
        "result": "new evidence",
        "is_error": False,
    })
    raw = json.dumps({
        "delivery_control": "replace",
        "full_answer": retained.full_text,
    })
    monkeypatch.setattr(
        loop,
        "_call_forced_model_once",
        lambda _ctx: (
            raw,
            {"finish_reason_present": True, "finish_reason": "length", "tool_call_count": 0},
        ),
    )

    text, _usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="fallback",
        reason_code="provider_terminal",
        provider_terminal=True,
    )

    published = registry._ctx._delivery_candidate
    assert text == retained.full_text
    assert "STALE-EVIDENCE NOTICE" not in text
    assert published is not retained
    assert published.evidence_revision > retained.evidence_revision
    assert returned_trace["delivery_candidate"]["evidence_current"] is True


@pytest.mark.parametrize(
    ("episode", "raw"),
    [
        (False, '{"delivery_control":"replace","full_answer":"JSON document"}'),
        (True, 'Final prose.\n{"delivery_control":"keep"}'),
    ],
)
def test_provider_terminal_incomplete_preserves_unbound_json_residuals(
    tmp_path, monkeypatch, episode, raw,
):
    if episode:
        loop, registry, ctx, trace, _candidate = _start_control_episode(tmp_path)
        trace["tool_calls"].append({
            "tool": "write_file",
            "status": "ok",
            "result": "new evidence",
            "is_error": False,
        })
    else:
        loop, registry, ctx, trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(
        loop,
        "_call_forced_model_once",
        lambda _ctx: (
            raw,
            {"finish_reason_present": True, "finish_reason": "length", "tool_call_count": 0},
        ),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="fallback",
        reason_code="provider_terminal",
        provider_terminal=True,
    )

    assert text == raw
    assert registry._ctx._delivery_candidate.full_text == raw


@pytest.mark.parametrize(
    "raw",
    [
        '{"other":1,"other":2}',
        '{"payload":{"other":1,"other":2}}',
    ],
)
def test_forced_armed_arbitrary_duplicate_preserves_candidate(
    tmp_path, monkeypatch, raw,
):
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    loop._arm_delivery_control(registry, ctx, trace)
    parsed, duplicate, embedded = loop._parse_delivery_control_body(raw)
    assert getattr(parsed, "has_duplicate_keys", False) is True
    assert (duplicate, embedded) == (False, False)
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: ({"role": "assistant", "content": raw}, 0.0),
    )

    text, _usage, returned_trace = loop._forced_final_answer(
        ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text == candidate.full_text
    assert raw not in text
    published = registry._ctx._delivery_candidate
    assert published.degraded_reason == loop.REASON_DELIVERY_CONTROL_DEGRADED
    assert returned_trace["delivery_candidate"]["degraded_reason"] == (
        loop.REASON_DELIVERY_CONTROL_DEGRADED
    )


def test_forced_historical_keep_after_late_owner_directive_stays_stale(
    tmp_path, monkeypatch,
):
    incoming = queue.Queue()
    loop, registry, ctx, trace, _candidate = _start_control_episode(
        tmp_path, incoming=incoming,
    )
    retained = loop._replace_delivery_candidate(
        registry, ctx, trace, "Free-form improvement.", control="candidate",
    )
    calls = 0

    def forced_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            incoming.put("Late answer constraint")
            content = "Draft before the late owner constraint."
        else:
            content = '{"delivery_control":"keep"}'
        return {"role": "assistant", "content": content}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", forced_model)
    text, _usage, returned_trace = loop._forced_final_answer(
        ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    published = registry._ctx._delivery_candidate
    assert calls == 2
    assert text.startswith(retained.full_text)
    assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in text
    assert published.evidence_revision == retained.evidence_revision
    assert published.acceptance_binding["stale_evidence"] is True
    assert returned_trace["delivery_candidate"]["evidence_current"] is False
    forced = returned_trace["forced_finalization"]
    assert forced["source"] == (
        "model_control_retained_stale_evidence_resume_required"
    )
    assert forced["current_evidence_revision"] > retained.evidence_revision


@pytest.mark.parametrize(
    ("raw", "expected_degraded_reason"),
    [
        ('{"delivery_control":"keep"}', "round_limit"),
        ('{"delivery_control":"publish"}', "delivery_control_degraded"),
    ],
)
def test_forced_armed_retained_control_after_evidence_change_stays_stale(
    tmp_path, monkeypatch, raw, expected_degraded_reason,
):
    loop, registry, ctx, trace, retained = _start_control_episode(tmp_path)
    loop._arm_delivery_control(registry, ctx, trace)
    trace["tool_calls"].append({
        "tool": "write_file",
        "status": "ok",
        "result": "new evidence",
        "is_error": False,
    })
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": raw}, 0.0,
        ),
    )

    text, _usage, returned_trace = loop._forced_final_answer(
        ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    published = registry._ctx._delivery_candidate
    assert text.startswith(retained.full_text)
    assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in text
    assert published.evidence_revision == retained.evidence_revision
    assert published.acceptance_binding["stale_evidence"] is True
    assert published.degraded_reason == expected_degraded_reason
    assert returned_trace["delivery_candidate"]["evidence_current"] is False


def test_forced_equal_text_replace_after_evidence_change_is_fresh(tmp_path, monkeypatch):
    loop, registry, ctx, trace, old = _start_control_episode(tmp_path)
    loop._arm_delivery_control(registry, ctx, trace)
    trace["tool_calls"].append({
        "tool": "write_file",
        "status": "ok",
        "result": "new evidence",
        "is_error": False,
    })
    replacement = json.dumps({
        "delivery_control": "replace",
        "full_answer": old.full_text,
    })
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": replacement}, 0.0,
        ),
    )

    text, _usage, returned_trace = loop._forced_final_answer(
        ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    published = registry._ctx._delivery_candidate
    assert text == old.full_text
    assert "STALE-EVIDENCE NOTICE" not in text
    assert published is not old
    assert published.evidence_revision > old.evidence_revision
    assert "stale_evidence" not in published.acceptance_binding
    assert returned_trace["delivery_candidate"]["evidence_current"] is True


def test_ordinary_latch_off_after_episode_resolves_stale_keep(tmp_path):
    loop, registry, ctx, trace, _candidate = _start_control_episode(tmp_path)
    candidate = loop._replace_delivery_candidate(
        registry, ctx, trace, "Free-form improvement.", control="candidate",
    )
    before_messages = copy.deepcopy(ctx.messages)
    before = (candidate.revision, candidate.content_sha256, copy.deepcopy(candidate.acceptance_binding))

    status, text = loop._resolve_delivery_control(
        '{"delivery_control":"keep"}', registry, ctx, trace,
    )

    assert (status, text) == ("resolved", candidate.full_text)
    assert ctx.messages == before_messages
    assert candidate.repair_attempted is False
    assert (candidate.revision, candidate.content_sha256, candidate.acceptance_binding) == before


def test_multi_improvement_stale_replace_is_decoded_before_acceptance_and_delivery(
    tmp_path,
    monkeypatch,
):
    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.loop as loop
    import supervisor.events as supervisor_events
    import supervisor.message_bus as message_bus
    import supervisor.state as supervisor_state
    from ouroboros.review_substrate import ReviewRunResult
    from ouroboros.tools.registry import ToolRegistry
    from supervisor.terminal_delivery import already_delivered, pending_deliveries

    decoded = "Final **answer**\n\n- one\n- two"
    stale_envelope = json.dumps({
        "delivery_control": "replace",
        "full_answer": decoded,
    })
    responses = iter([
        "Initial complete draft.",
        json.dumps({
            "delivery_control": "replace",
            "full_answer": "Controlled answer v1.",
        }),
        "Free-form improvement v2.",
        "Free-form improvement v3.",
        stale_envelope,
    ])
    model_calls = []
    acceptance_inputs = []
    acceptance_candidates = []
    acceptance_bindings = []
    lineage_flags = []
    nudge_calls = 0

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call(_llm, request_messages, *_args, **_kwargs):
        model_calls.append([dict(row) for row in request_messages])
        return {"role": "assistant", "content": next(responses)}, 0.0

    def first_nudge_only(*_args, **_kwargs):
        nonlocal nudge_calls
        nudge_calls += 1
        return nudge_calls == 1

    def fake_panel(review_ctx):
        content = review_ctx.content
        acceptance_inputs.append(content)
        acceptance_candidates.append(review_ctx.tools._ctx._delivery_candidate)
        acceptance_bindings.append(copy.deepcopy(review_ctx.review_binding))
        lineage_flags.append(acceptance_candidates[-1].control_episode_seen)
        if len(acceptance_inputs) < 4:
            return ReviewRunResult(
                request={"surface": "task_acceptance"},
                actors=[{
                    "slot_id": "host-issue-449",
                    "signal": "FAIL",
                    "parsed": {
                        "outcome_tier": "best_effort",
                        "completion_coach": "Revise the complete answer.",
                    },
                }],
                parsed_findings=[{
                    "severity": "major",
                    "item": "complete_answer_revision",
                    "recommendation": "Revise the complete answer.",
                }],
                aggregate_signal="FAIL",
            )
        return ReviewRunResult(
            request={"surface": "task_acceptance"},
            actors=[{
                "slot_id": "host-issue-449",
                "signal": "PASS",
                "parsed": {
                    "outcome_tier": "solved",
                    "completion_coach": "Ship the decoded answer.",
                    "criteria_used": [{
                        "criterion": "complete answer",
                        "status": "supported",
                        "evidence_refs": ["final_answer_marker"],
                    }],
                },
            }],
            parsed_findings=[],
            aggregate_signal="PASS",
        )

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", first_nudge_only)
    monkeypatch.setattr(loop, "_execute_task_acceptance_panel", fake_panel)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "4")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "10")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.is_direct_chat = False
    registry._ctx.task_contract = {
        "expected_output": "A complete user-facing answer.",
    }
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "issue-449",
    }

    result, usage, trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "do the work"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, *, incident=None: None,
        incoming_messages=queue.Queue(),
        task_id="issue-449",
        drive_root=tmp_path,
    )

    decoded_sha = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
    raw_sha = hashlib.sha256(stale_envelope.encode("utf-8")).hexdigest()
    prompt_counts = [
        sum(
            "[DELIVERY_FINALIZATION_CONTROL]" in str(row.get("content") or "")
            for row in call
        )
        for call in model_calls
    ]
    assert len(model_calls) == 5
    assert prompt_counts == [0, 1, 1, 1, 1]
    assert all(
        "[DELIVERY_CONTROL_REPAIR]" not in str(row.get("content") or "")
        for call in model_calls
        for row in call
    )
    assert acceptance_inputs == [
        "Controlled answer v1.",
        "Free-form improvement v2.",
        "Free-form improvement v3.",
        decoded,
    ]
    assert lineage_flags == [True, True, True, True]
    assert result == decoded
    assert result != stale_envelope
    assert acceptance_candidates[-1].full_text == decoded
    assert acceptance_candidates[-1].revision == 5
    assert trace["delivery_candidate"]["content_sha256"] == decoded_sha
    binding = trace["delivery_candidate"]["acceptance_binding"]
    assert binding["candidate_sha256"] == decoded_sha
    assert binding["binding_hash"] == trace["review_decision"]["binding_hash"]
    assert acceptance_bindings[-1]["candidate_hash"] == decoded_sha
    assert acceptance_bindings[-1]["binding_hash"] == binding["binding_hash"]
    assert trace["review_runs"][-1]["candidate_hash"] == decoded_sha
    assert trace["review_runs"][-1]["candidate_hash"] != raw_sha
    assert trace["review_runs"][-1]["authority"] == "host_root"

    drive_logs = tmp_path / "pipeline-logs"
    drive_logs.mkdir()
    monkeypatch.setattr(
        supervisor_state,
        "reconstruct_task_cost",
        lambda *_args, **_kwargs: {
            "cost_accounting_status": "available",
            "cost_final": True,
            "cost_usd": 0.0,
            "total_rounds": 5,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "reserved_usd": 0.0,
            "unresolved_upper_bound_usd": 0.0,
            "unknown_unmetered": 0,
        },
    )
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    pending = []
    pipeline.emit_task_results(
        env=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path),
        memory=object(),
        llm=object(),
        pending_events=pending,
        task={"id": "issue-449", "type": "task", "chat_id": 7, "text": "do it"},
        text=result,
        usage=usage,
        llm_trace=trace,
        start_time=0.0,
        drive_logs=drive_logs,
        ctx=SimpleNamespace(pending_restart_reason=""),
    )

    send_events = [event for event in pending if event.get("type") == "send_message"]
    assert [event["text"] for event in send_events] == [decoded]
    stored = pipeline.load_task_result(tmp_path, "issue-449")
    assert stored["result"] == decoded
    assert stored["loop_outcome"]["final_text"] == decoded
    owed = pending_deliveries(tmp_path)
    assert len(owed) == 1
    assert owed[0]["text"] == decoded
    assert owed[0]["delivery_id"] == send_events[0]["delivery_id"]

    transported = []
    monkeypatch.setattr(supervisor_events, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        message_bus,
        "load_state",
        lambda: {"owner_id": 1, "session_id": "issue-449-test"},
    )
    monkeypatch.setattr(
        message_bus,
        "_send_markdown",
        lambda chat_id, text, **_kwargs: (transported.append((chat_id, text)) or True, None),
    )
    event_ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        send_with_budget=message_bus.send_with_budget,
        append_jsonl=lambda *_args, **_kwargs: None,
    )
    supervisor_events._handle_send_message(dict(send_events[0]), event_ctx)
    assert already_delivered(tmp_path, send_events[0]["delivery_id"]) is True
    monkeypatch.setattr(supervisor_events, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    supervisor_events._handle_send_message(dict(send_events[0]), event_ctx)

    assert transported == [(7, decoded)]
    assert pending_deliveries(tmp_path) == []
    chat_rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    delivered_rows = [row for row in chat_rows if row.get("task_id") == "issue-449"]
    assert [row["text"] for row in delivered_rows] == [decoded]
    assert delivered_rows[0]["format"] == "markdown"

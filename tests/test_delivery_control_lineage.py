from __future__ import annotations

from tests import _dialogue as dialogue_test

import copy
import hashlib
import json
import queue
from collections import deque
from types import SimpleNamespace

import pytest

from tests.test_delivery_forced_finalization import _forced_test_context, _select_completion
from tests.test_delivery_candidate import _finish_response


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
    _select_completion(registry, ctx, trace, answer=text)
    candidate = registry._ctx._delivery_candidate
    assert candidate.full_text == text
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
        '{"delivery_control":"keep","pending_review":"later"}',
        '{"delivery_control":"replace","full_answer":"x","pending_review":true}',
        '```json\n{"delivery_control":"publish"}\n```',
    ],
)
def test_post_episode_invalid_whole_body_preserves_without_repair(tmp_path, raw):
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    before_messages = copy.deepcopy(ctx.messages)
    before = (candidate.revision, candidate.content_sha256, copy.deepcopy(candidate.acceptance_binding))
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        candidate.full_text, loop.REASON_DELIVERY_CONTROL_DEGRADED, True, False)
    assert ctx.messages == before_messages
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", candidate.full_text)
    assert ctx.messages[-2] == {"role": "assistant", "content": raw}
    assert ctx.messages[-1]["role"] == "user" and "latest whole held response" not in ctx.messages[-1]["content"]
    assert ctx.messages[:-2] == before_messages
    assert (candidate.revision, candidate.content_sha256, candidate.acceptance_binding) == before
    assert registry._ctx._delivery_control_required is True
    assert "DELIVERY_CONTROL_REPAIR" not in str(ctx.messages)


def test_substantive_hold_starts_sticky_completion_episode_lineage(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "Initial complete answer.", control="candidate")
    assert candidate.control_episode_seen is False
    loop._hold_delivery_for_skill_action(registry, trace)
    assert candidate.control_episode_seen is True
    assert loop._resolve_delivery_control('{"delivery_control":"keep"}', registry, ctx, trace) == ("retry", candidate.full_text)
    corrected_text = "Corrected complete answer."
    _select_completion(registry, ctx, trace, answer=corrected_text)
    corrected = registry._ctx._delivery_candidate
    assert corrected.control_episode_seen is True
    before = copy.deepcopy(ctx.messages)
    assert loop._resolve_delivery_control('{"delivery_control":"publish"}', registry, ctx, trace) == ("retry", corrected_text)
    assert ctx.messages[:-2] == before and ctx.messages[-1]["role"] == "user"
    assert "DELIVERY_CONTROL_REPAIR" not in str(ctx.messages)


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
        '{"delivery_control":"replace","full_answer":"cut',
        '{"payload":{"delivery_control":"keep","delivery_control":"replace"}}',
        '{"full_answer":"one","full_answer":"two"}',
        '{"other":1,"other":2}',
        '{"payload":{"other":1,"other":2}}',
    ],
)
def test_post_episode_residuals_stay_byte_exact_but_need_explicit_ordinary_selection(tmp_path, raw):
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    before_messages = copy.deepcopy(ctx.messages)
    before = (candidate.revision, candidate.content_sha256, copy.deepcopy(candidate.acceptance_binding))
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (raw, "", False, False)
    assert ctx.messages == before_messages
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", candidate.full_text)
    assert ctx.messages[-2] == {"role": "assistant", "content": raw}
    assert registry._ctx._completion_held_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    assert (candidate.revision, candidate.content_sha256, candidate.acceptance_binding) == before


def test_duplicate_full_answer_without_verb_preserves_stronger_control_rails(tmp_path):
    raw = '{"full_answer":"one","full_answer":"two"}'
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    parsed, duplicate, embedded = loop._parse_delivery_control_body(raw)
    assert parsed == {"full_answer": "two"} and getattr(parsed, "duplicate_keys", set()) == {"full_answer"}
    assert (duplicate, embedded) == (False, False)
    assert loop._classify_parsed_delivery_control(parsed, duplicate, embedded)[0] == "rail_invalid"
    for control in ("owner_revision_required", loop._SKILL_ACTION_HOLD_CONTROL):
        candidate.finalization_control = control
        registry._ctx._delivery_control_required = False
        assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", candidate.full_text)
        assert candidate.finalization_control == control
        assert ctx.messages[-2]["content"] == raw and ctx.messages[-1]["role"] == "user"
    assert "DELIVERY_CONTROL_REPAIR" not in str(ctx.messages)
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        candidate.full_text, loop.REASON_DELIVERY_CONTROL_DEGRADED, True, False)


def test_arbitrary_duplicates_remain_selectable_prose_during_ordinary_holds(tmp_path):
    raw = '{"other":1,"other":2}'
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    for control in ("owner_revision_required", loop._SKILL_ACTION_HOLD_CONTROL):
        candidate.finalization_control = control
        assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", candidate.full_text)
        assert registry._ctx._completion_held_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    candidate.finalization_control = "skill_revision_required"
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        candidate.full_text, loop.REASON_DELIVERY_CONTROL_DEGRADED, True, False)


@pytest.mark.parametrize(
    ("raw", "expected", "retained", "replaced"),
    [
        ('{"delivery_control":"keep"}', "Free-form improvement.", True, False),
        (
            '```json\n{"delivery_control":"replace",'
            '"full_answer":"Forced **replacement**\\nline two"}\n```',
            "Free-form improvement.",
            True,
            False,
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
            '{"action":"finish","answer":"decoded answer"}',
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
            "delivery_control_degraded",
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
        assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in usage["terminal_host_notice"]
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
        "action": "finish",
        "answer": "decoded replacement",
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
        "action": "finish",
        "answer": retained.full_text,
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
def test_provider_terminal_incomplete_respects_protocol_lineage(
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

    expected = _candidate.full_text if episode else raw
    assert text == expected
    assert registry._ctx._delivery_candidate.full_text == expected


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
    assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in _usage["terminal_host_notice"]
    assert published.evidence_revision == retained.evidence_revision
    assert published.owner_source_sha256 == retained.owner_source_sha256
    assert published.acceptance_binding["stale_evidence"] is True
    # The owner corpus changed without a Main acknowledgment. Its meaning is
    # still undecided, so the material-evidence revision must not be invented.
    assert returned_trace["delivery_candidate"]["owner_source_current"] is False
    assert returned_trace["delivery_candidate"]["evidence_current"] is True
    forced = returned_trace["forced_finalization"]
    assert forced["source"] == (
        "model_control_retained_stale_evidence_resume_required"
    )
    assert forced["current_evidence_revision"] == retained.evidence_revision


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
    assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in _usage["terminal_host_notice"]
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
        "action": "finish",
        "answer": old.full_text,
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


def test_ordinary_post_episode_legacy_keep_stays_private_without_finishing(tmp_path):
    loop, registry, ctx, trace, _candidate = _start_control_episode(tmp_path)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "Free-form improvement.", control="candidate")
    before_messages = copy.deepcopy(ctx.messages)
    before = (candidate.revision, candidate.content_sha256, copy.deepcopy(candidate.acceptance_binding))
    raw = '{"delivery_control":"keep"}'
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", candidate.full_text)
    assert ctx.messages[:-2] == before_messages and ctx.messages[-2]["content"] == raw
    assert ctx.messages[-1]["role"] == "user" and not registry._ctx._completion_held_sha256
    assert (candidate.revision, candidate.content_sha256, candidate.acceptance_binding) == before


def test_multi_improvement_explicit_selection_binds_exact_answer_before_single_delivery(
    tmp_path,
    monkeypatch,
):
    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.loop as loop
    import supervisor.events as supervisor_events
    import supervisor.message_bus as message_bus
    import supervisor.state as supervisor_state
    from ouroboros.tools.registry import ToolRegistry
    from supervisor.terminal_delivery import already_delivered, pending_deliveries

    decoded = "Final **answer**\n\n- one\n- two"
    stale_envelope = json.dumps({"delivery_control": "replace", "full_answer": "Obsolete unselected text"})
    responses = iter([
        "Initial complete draft.", _finish_response("Controlled answer v1."),
        _finish_response("Free-form improvement v2."), _finish_response("Free-form improvement v3."),
        stale_envelope, _finish_response(decoded),
    ])
    model_calls = []
    reviewed = []
    reviewed_candidates = []
    lineage_flags = []
    nudge_calls = 0

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call(_llm, request_messages, *_args, **_kwargs):
        model_calls.append([dict(row) for row in request_messages])
        response = next(responses)
        return ({"role": "assistant", **response} if isinstance(response, dict)
                else {"role": "assistant", "content": response}), 0.0

    def first_nudge_only(*_args, **_kwargs):
        nonlocal nudge_calls
        nudge_calls += 1
        return nudge_calls == 1

    def review_at_delivery(ctx, content, _llm_trace, hold):
        # Blocking holds a failed result until a review passes it: three reviews ask for a
        # revised complete answer and show their findings; the fourth passes the decoded one.
        reviewed.append(content)
        reviewed_candidates.append(ctx._delivery_candidate)
        lineage_flags.append(reviewed_candidates[-1].control_episode_seen)
        if len(reviewed) < 4:
            hold("[Result review] FAIL — complete_answer_revision: Revise the complete answer.")
            return True
        return False

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", first_nudge_only)
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", review_at_delivery)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "required")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "4")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "10")
    from tests.test_loop_acceptance_gate import _seed_acceptance_root
    _seed_acceptance_root(tmp_path, "issue-449", SimpleNamespace())
    # Required reviews retained produced material, never the expected_output (#1731).
    from ouroboros.artifacts import store_task_artifact_bytes
    store_task_artifact_bytes(tmp_path, "issue-449", "deliverable.md", b"Produced deliverable.")
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
        emit_progress=lambda _text, **_facts: None,
        incoming_messages=queue.Queue(),
        task_id="issue-449",
        drive_root=tmp_path,
    )

    decoded_sha = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
    prompt_counts = [
        sum(
            "[DELIVERY_FINALIZATION_CONTROL]" in str(row.get("content") or "")
            for row in call
        )
        for call in model_calls
    ]
    # Six authored rounds: the review that passes the decoded answer delivers it in its own round.
    assert len(model_calls) == 6
    assert prompt_counts[0] == 0 and all(count >= 1 for count in prompt_counts[1:])
    assert all(
        "[DELIVERY_CONTROL_REPAIR]" not in str(row.get("content") or "")
        for call in model_calls
        for row in call
    )
    assert reviewed == [
        "Controlled answer v1.",
        "Free-form improvement v2.",
        "Free-form improvement v3.",
        decoded,
    ]
    assert lineage_flags == [True, True, True, True]
    assert result == decoded
    assert result != stale_envelope
    assert reviewed_candidates[-1].full_text == decoded
    assert reviewed_candidates[-1].revision == 5
    assert trace["delivery_candidate"]["content_sha256"] == decoded_sha
    binding = trace["delivery_candidate"]["acceptance_binding"]
    assert binding["candidate_sha256"] == decoded_sha

    drive_logs = tmp_path / "pipeline-logs"
    drive_logs.mkdir()
    monkeypatch.setattr(
        supervisor_state,
        "reconstruct_task_cost",
        lambda *_args, **_kwargs: {
            "cost_accounting_status": "available",
            "cost_final": True,
            "cost_usd": 0.0,
            "total_rounds": 6,
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
    chat_rows = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    delivered_rows = [row for row in chat_rows if row.get("task_id") == "issue-449"]
    assert [row["text"] for row in delivered_rows] == [decoded]
    assert delivered_rows[0]["format"] == "markdown"


@pytest.mark.parametrize("verb", ["keep", "replace", "publish"])
def test_post_episode_trailing_control_retains_answer_on_both_rails(tmp_path, verb):
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    payload = {"delivery_control": verb}
    if verb == "replace":
        payload["full_answer"] = "Unusable embedded replacement."
    raw = "A stray control preamble.\n" + json.dumps(payload)
    before = (candidate.revision, candidate.content_sha256, copy.deepcopy(candidate.acceptance_binding))
    before_messages = copy.deepcopy(ctx.messages)
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        candidate.full_text, loop.REASON_DELIVERY_CONTROL_DEGRADED, True, False)
    assert ctx.messages == before_messages
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", candidate.full_text)
    assert (candidate.revision, candidate.content_sha256, candidate.acceptance_binding) == before
    assert ctx.messages[:-2] == before_messages and ctx.messages[-2]["content"] == raw
    assert ctx.messages[-1]["role"] == "user" and not registry._ctx._completion_held_sha256
    assert "DELIVERY_CONTROL_REPAIR" not in str(ctx.messages)


# A whole-body JSON array whose DIRECT element is a control envelope (the ``delivery_control``
# key or ``action`` exactly finish/stop) is a malformed protocol attempt under control provenance.
_WRAPPED_CONTROLS = [
    '[{"delivery_control":"keep"}]',
    '[{"delivery_control":"replace","full_answer":"Wrapped replacement."}]',
    '```json\n[{"delivery_control":"keep"}]\n```',
    '[{"action":"finish","answer":"Wrapped answer."}]',
    '[{"action":"stop","rationale":"Wrapped stop."}]',
    '[1, "note", {"delivery_control":"keep"}]',
    '[{"name":"row"}, {"action":"finish","answer_sha256":"0000"}]',
    '[{"delivery_control":"keep","delivery_control":"replace"}]',
    # A repeated key cannot hide the verb its first pair names: a parse keeping the last value would.
    '[{"action":"finish","answer":"x","action":"buy"}]',
    '[{"action":"stop","rationale":"Wrapped stop.","action":"sell"}]',
]
# Ordinary JSON arrays stay answers under the same provenance: data, a string quoting an
# envelope, nested arrays/objects, another verb, a non-string verb, prose before the array.
_ORDINARY_ARRAYS = [
    '[{"name":"row","value":1}]',
    '["{\\"delivery_control\\":\\"keep\\"}"]',
    '[[{"delivery_control":"keep"}]]',
    '[{"payload":{"delivery_control":"keep"}}]',
    '[{"action":"buy","answer":"shares"}]',
    '[{"action":["finish"]}]',
    'Example: [{"delivery_control":"keep"}]',
    '[{"a":1,"a":2}]',
    '[{"action":"buy","action":"sell"}]',
    '[{"rows":[{"action":"finish","action":"buy"}]}]',
]


def _select_by_hash(registry, ctx, trace, sha256):
    from ouroboros.loop_delivery import completion_observation, consume_completion_request
    from ouroboros.tools.control_runtime import stage_completion_request

    registry._ctx._completion_observation = completion_observation(registry._ctx, trace)
    staged = stage_completion_request(registry._ctx, {"action": "finish", "answer_sha256": sha256})
    assert json.loads(staged)["status"] == "completion_requested"
    return consume_completion_request(registry, ctx, trace)


@pytest.mark.parametrize("raw", _WRAPPED_CONTROLS)
def test_post_episode_array_wrapped_control_stays_private_on_both_rails(tmp_path, raw):
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    before = (candidate.revision, candidate.content_sha256, copy.deepcopy(candidate.acceptance_binding))
    before_messages = copy.deepcopy(ctx.messages)
    parsed, duplicate, embedded = loop._parse_delivery_control_body(raw)
    assert (parsed, duplicate, embedded) == (None, False, True)
    # Forced rail, sticky provenance and an armed latch alike: the saved answer, degraded, never the wrapper.
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        candidate.full_text, loop.REASON_DELIVERY_CONTROL_DEGRADED, True, False)
    registry._ctx._delivery_control_required = True
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (
        candidate.full_text, loop.REASON_DELIVERY_CONTROL_DEGRADED, True, False)
    assert ctx.messages == before_messages
    # Ordinary hold: the wrapper is kept in the transcript but its hash is never offered or selectable.
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", candidate.full_text)
    assert ctx.messages[-2] == {"role": "assistant", "content": raw}
    assert "latest whole held response" not in ctx.messages[-1]["content"]
    assert registry._ctx._completion_held_sha256 == "" and registry._ctx._completion_pair_private is True
    assert _select_by_hash(registry, ctx, trace, hashlib.sha256(raw.encode()).hexdigest()) is False
    assert trace["completion_refusals"][-1]["reason"].startswith("selection_unavailable")
    assert registry._ctx._delivery_candidate is candidate
    assert (candidate.revision, candidate.content_sha256, candidate.acceptance_binding) == before
    assert "DELIVERY_CONTROL_REPAIR" not in str(ctx.messages)


@pytest.mark.parametrize("raw", ['[{"action":"stop","rationale":"Wrapped stop."}]',
                                 '[{"action":"stop","rationale":"Wrapped stop.","action":"sell"}]'])
def test_forced_array_wrapped_stop_neither_stops_nor_publishes_the_wrapper(tmp_path, monkeypatch, raw):
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    loop._arm_delivery_control(registry, ctx, trace)
    monkeypatch.setattr(loop, "call_llm_with_retry",
                        lambda *_args, **_kwargs: ({"role": "assistant", "content": raw}, 0.0))

    text, _usage, returned_trace = loop._forced_final_answer(
        ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text == candidate.full_text and raw not in text
    assert registry._ctx._delivery_candidate.degraded_reason == loop.REASON_DELIVERY_CONTROL_DEGRADED
    assert returned_trace["delivery_candidate"]["degraded_reason"] == loop.REASON_DELIVERY_CONTROL_DEGRADED
    assert "task_completion" not in returned_trace


@pytest.mark.parametrize("raw", _ORDINARY_ARRAYS)
def test_ordinary_json_arrays_under_control_provenance_stay_selectable_answers(tmp_path, raw):
    loop, registry, ctx, trace, candidate = _start_control_episode(tmp_path)
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (raw, "", False, False)
    registry._ctx._delivery_control_required = True
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (raw, "", False, False)
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", candidate.full_text)
    held = hashlib.sha256(raw.encode()).hexdigest()
    assert registry._ctx._completion_held_sha256 == held and f"answer_sha256={held}" in ctx.messages[-1]["content"]
    assert _select_by_hash(registry, ctx, trace, held) is True
    assert registry._ctx._delivery_candidate.full_text == raw


@pytest.mark.parametrize("raw", ['[{"delivery_control":"keep"}]', '[{"action":"finish","answer":"x"}]'])
def test_no_episode_json_arrays_are_byte_exact_answers(tmp_path, raw):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "Existing answer.", control="candidate")
    assert candidate.control_episode_seen is False
    assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("fresh", raw)
    assert loop._resolve_forced_delivery_control(registry._ctx, raw) == (raw, "", False, False)

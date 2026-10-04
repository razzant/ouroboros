"""Transactional and chunk-coverage tests for the reclaim materializer."""

from __future__ import annotations

import hashlib
import json
import copy
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction as cc
from ouroboros.context_budget import ContextReclaimRequest

_SPEC = {
    "model": "summary-model",
    "resolved_model": "summary-model",
    "provider": "test",
    "route_fp": "summary-route",
    "effort": "low",
    "output_budget": 32_768,
    "use_local": False,
}


def _unit(call_id: str, fill: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"reasoning-{call_id}",
            "tool_calls": [{
                "id": call_id,
                "function": {
                    "name": "commit_reviewed" if call_id == "warning" else "read_file",
                    "arguments": fill * 3_000 + f"-argument-tail-{call_id}",
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": fill * 3_000 + f"-result-tail-{call_id}",
        },
    ]


def _request(
    messages: list[dict],
    goal: int,
    *,
    allow_partial: bool = True,
    measurement_basis: str = "cold_estimate",
    measurement_density: float = 1.0,
):
    return ContextReclaimRequest(
        route_fp="main-route",
        round_id="round-1",
        transcript_sha256=cc.context_reclaim_transcript_sha256(messages),
        measurement_basis=measurement_basis,
        measurement_density=measurement_density,
        reclaim_goal_tokens=goal,
        allow_partial_shrink=allow_partial,
    )


def _install_pure_dependencies(monkeypatch):
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: dict(_SPEC))
    monkeypatch.setattr(
        cc,
        "_persist_reclaim_checkpoint",
        lambda *_a, **_k: {"path": "calls/checkpoint.json", "sha256": "c" * 64},
    )


def test_typed_overflow_recursively_splits_one_gap_free_hashed_source(monkeypatch, tmp_path):
    text = "line-1\n" + "abcdefghij" * 61 + "\nlast-line"
    root = "unit:12:13:0123456789abcdef"
    initial = cc._part(root, text)
    attempted_sizes: list[int] = []
    call_totals: list[int] = []

    def bounded(parts, **_kwargs):
        attempted_sizes.extend(len(part.text) for part in parts)
        call_totals.append(sum(len(part.text) for part in parts))
        if any(len(part.text) > 90 for part in parts):
            raise cc.SummarizerContextOverflow("typed context_length_exceeded")
        return {part.source_id: f"summary:{part.sha256}" for part in parts}

    monkeypatch.setattr(cc, "_call_summarizer", bounded)
    leaves, summaries, failed = cc._map_complete_parts(
        [initial],
        drive_root=tmp_path,
        task_id="task",
        spec=_SPEC,
        summary_budgets={root: 700},
        usage_total={},
    )

    assert attempted_sizes.count(len(text)) == 1
    # CMP-6: an overflowed payload is never resent at the same TOTAL size —
    # after the first overflowing call every summarizer call must be strictly
    # smaller (a single-leaf split may not resend both halves in one call).
    assert call_totals[0] == len(text)
    assert all(total < len(text) for total in call_totals[1:])
    assert "".join(leaf.text for leaf in leaves) == text
    assert [leaf.start_char for leaf in leaves] == [0, *[leaf.end_char for leaf in leaves[:-1]]]
    assert leaves[-1].end_char == len(text)
    assert all(leaf.root_id == root for leaf in leaves)
    assert all(leaf.source_id.startswith(root + ":") for leaf in leaves)
    assert all(
        leaf.sha256 == hashlib.sha256(leaf.text.encode("utf-8")).hexdigest()
        for leaf in leaves
    )
    assert set(summaries) == {leaf.source_id for leaf in leaves}
    assert failed == set()


@pytest.mark.parametrize("payload_chars", [80_000, 300_000])
def test_large_actor_input_is_complete_and_gap_free(monkeypatch, tmp_path, payload_chars):
    argument_tail = "-DECISIVE-ARGUMENT-TAIL"
    result_tail = "-DECISIVE-RESULT-TAIL"
    messages = [
        {
            "role": "assistant",
            "content": "complete reasoning",
            "tool_calls": [{
                "id": "large",
                "function": {
                    "name": "read_file",
                    "arguments": "a" * payload_chars + argument_tail,
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "large",
            "content": "r" * payload_chars + result_tail,
        },
    ]
    unit = cc._atomic_units(messages)[0]
    assert argument_tail in unit.source_text
    assert result_tail in unit.source_text

    initial = cc._part(unit.unit_id, unit.source_text)
    attempts: set[tuple[tuple[str, int, int, str], ...]] = set()
    call_totals: list[int] = []

    def bounded(parts, **_kwargs):
        identity = tuple(
            (part.source_id, part.start_char, part.end_char, part.sha256)
            for part in parts
        )
        assert identity not in attempts, "typed overflow resent an identical payload"
        attempts.add(identity)
        call_totals.append(sum(len(part.text) for part in parts))
        if any(len(part.text) > 40_000 for part in parts):
            raise cc.SummarizerContextOverflow("typed context_length_exceeded")
        return {part.source_id: f"summary:{part.sha256}" for part in parts}

    monkeypatch.setattr(cc, "_call_summarizer", bounded)
    leaves, summaries, failed = cc._map_complete_parts(
        [initial],
        drive_root=tmp_path,
        task_id="task",
        spec=_SPEC,
        summary_budgets={unit.unit_id: 2_000},
        usage_total={},
    )

    assert "".join(leaf.text for leaf in leaves) == unit.source_text
    # CMP-6: after the first overflowing call every summarizer call carries a
    # strictly smaller total payload (no same-size two-half resend).
    assert all(total < call_totals[0] for total in call_totals[1:])
    assert [leaf.start_char for leaf in leaves] == [
        0,
        *[leaf.end_char for leaf in leaves[:-1]],
    ]
    assert leaves[-1].end_char == len(unit.source_text)
    assert all(
        leaf.sha256 == hashlib.sha256(leaf.text.encode("utf-8")).hexdigest()
        for leaf in leaves
    )
    assert set(summaries) == {leaf.source_id for leaf in leaves}
    assert failed == set()


def test_untyped_overflow_message_marker_authorizes_split():
    """A bare 400 without structured codes still classifies as overflow via the
    shared message markers (same vocabulary as the Main classification seam);
    unrelated provider errors stay non-overflow and fail the unit raw."""
    untyped = RuntimeError(
        "Error code: 400 - {'error': {'message': 'prompt is too long: "
        "250000 tokens > 200000 maximum', 'type': 'invalid_request_error'}}"
    )
    assert cc._typed_context_overflow(untyped)
    assert not cc._typed_context_overflow(RuntimeError("429 rate limit exceeded"))
    assert not cc._typed_context_overflow(RuntimeError("transport reset by peer"))


def test_output_limit_wording_never_authorizes_summarizer_split():
    """OUTPUT-size rejections are not window overflows: splitting the batch
    cannot fix them, so they must fail the unit raw (_UnitSummaryFailure path)
    instead of authorizing a split. A structured overflow CODE still wins over
    output wording — same precedence as the shared Main/local seams."""
    output_limit = RuntimeError("max_tokens 65536 exceeds maximum context length 32768")
    assert not cc._typed_context_overflow(output_limit)

    class _TypedOverflowWithOutputWording(RuntimeError):
        body = {"error": {"code": "context_length_exceeded"}}

    assert cc._typed_context_overflow(
        _TypedOverflowWithOutputWording("max_tokens exceeds maximum context length"))


def test_non_overflow_failure_never_splits_or_retries(monkeypatch, tmp_path):
    calls = 0

    def infrastructure_failure(_parts, **_kwargs):
        nonlocal calls
        calls += 1
        raise cc._UnitSummaryFailure("transport failed")

    monkeypatch.setattr(cc, "_call_summarizer", infrastructure_failure)
    part = cc._part("unit", "x" * 2_000)
    leaves, summaries, failed = cc._map_complete_parts(
        [part],
        drive_root=tmp_path,
        task_id="task",
        spec=_SPEC,
        summary_budgets={"unit": 700},
        usage_total={},
    )
    assert calls == 1
    assert leaves == ()
    assert summaries == {}
    assert failed == {"unit"}


def test_direct_http_response_structured_overflow_is_typed():
    class Response:
        def json(self):
            return {"error": {"code": "context_window_exceeded"}}

    error = RuntimeError("provider rejected summarizer request")
    error.response = Response()
    assert cc._typed_context_overflow(error) is True


def test_physical_capture_structured_overflow_is_typed():
    error = RuntimeError("provider rejected summarizer request")
    error.physical_attempt_capture = SimpleNamespace(
        provider_code=None,
        provider_error_type="prompt_too_long",
    )
    assert cc._typed_context_overflow(error) is True


def test_one_missing_unit_stays_wholly_raw_while_neighbor_applies(monkeypatch, tmp_path):
    _install_pure_dependencies(monkeypatch)
    messages = [*_unit("first", "a"), *_unit("second", "b")]

    def map_units(parts, **_kwargs):
        second = next(part for part in parts if "result-tail-second" in part.text)
        first = next(part for part in parts if "result-tail-first" in part.text)
        return tuple(parts), {second.source_id: "covered second unit"}, {first.root_id}

    monkeypatch.setattr(cc, "_map_complete_parts", map_units)
    rebuilt, receipt, usage = cc.compact_tool_history_llm(
        messages,
        request=_request(messages, 1_000_000),
        drive_root=tmp_path,
        negative_memo=set(),
    )

    assert receipt.status == "applied"
    assert len(receipt.selected_unit_ids) == 2
    assert receipt.goal_reached is False
    assert rebuilt[:2] == messages[:2]
    assert len(rebuilt) == 3
    assert rebuilt[2]["content"][0]["_context_capsule"]["generation"] == 1
    assert usage is None


def test_completed_non_shrink_memo_ignores_goal_drift_but_tracks_route(monkeypatch, tmp_path):
    _install_pure_dependencies(monkeypatch)
    messages = [*_unit("first", "a"), *_unit("second", "b")]
    memo: set[str] = set()
    seen: list[str] = []

    def map_units(parts, **_kwargs):
        seen.extend(part.text for part in parts)
        first = next((part for part in parts if "result-tail-first" in part.text), None)
        failed = {part.root_id for part in parts if "result-tail-second" in part.text}
        summaries = {first.source_id: first.text * 2} if first else {}
        return tuple(parts), summaries, failed

    monkeypatch.setattr(cc, "_map_complete_parts", map_units)
    request = _request(messages, 1_000_000)
    rebuilt, receipt, _ = cc.compact_tool_history_llm(
        messages, request=request, drive_root=tmp_path, negative_memo=memo,
    )

    assert rebuilt is messages
    assert receipt.status == "summarizer_failed"
    assert len(memo) == 1
    assert len(seen) == 2

    seen.clear()
    _, repeated, _ = cc.compact_tool_history_llm(
        messages, request=request, drive_root=tmp_path, negative_memo=memo,
    )
    assert len(repeated.selected_unit_ids) == 1
    assert len(seen) == 1
    assert "result-tail-second" in seen[0]

    seen.clear()
    changed_goal = _request(messages, 999_999)
    _, unchanged_key, _ = cc.compact_tool_history_llm(
        messages, request=changed_goal, drive_root=tmp_path, negative_memo=memo,
    )
    assert len(unchanged_key.selected_unit_ids) == 1
    assert len(seen) == 1
    assert "result-tail-second" in seen[0]

    monkeypatch.setattr(
        cc,
        "_summarizer_spec",
        lambda: {**_SPEC, "route_fp": "different-summary-route"},
    )
    seen.clear()
    _, changed_route, _ = cc.compact_tool_history_llm(
        messages, request=request, drive_root=tmp_path, negative_memo=memo,
    )
    assert len(changed_route.selected_unit_ids) == 2
    assert len(seen) == 2


def test_materialized_units_never_swallow_intervening_user_turn(monkeypatch, tmp_path):
    _install_pure_dependencies(monkeypatch)
    owner_turn = {
        "role": "user",
        "content": "Owner correction must remain byte-for-byte visible.",
    }
    messages = [*_unit("before", "a"), owner_turn, *_unit("after", "b")]

    monkeypatch.setattr(
        cc,
        "_call_summarizer",
        lambda parts, **_kwargs: {
            part.source_id: f"summary {part.sha256}" for part in parts
        },
    )
    rebuilt, receipt, _ = cc.compact_tool_history_llm(
        messages,
        request=_request(messages, 1_000_000),
        drive_root=tmp_path,
        negative_memo=set(),
    )

    assert receipt.status == "applied"
    assert len(rebuilt) == 3
    assert [message for message in rebuilt if message.get("role") == "user"
            and not cc._capsule_metadata(message)[0]] == [owner_turn]
    assert rebuilt[1] == owner_turn
    assert rebuilt[1] is owner_turn


def test_binding_mismatch_and_insufficient_nonpartial_goal_are_pure(monkeypatch):
    _install_pure_dependencies(monkeypatch)
    messages = _unit("bound", "x")
    checkpoints: list[bool] = []
    monkeypatch.setattr(
        cc, "_persist_reclaim_checkpoint",
        lambda *_a, **_k: checkpoints.append(True),
    )
    bad_request = ContextReclaimRequest(
        route_fp="route",
        round_id="round",
        transcript_sha256="0" * 64,
        measurement_basis="cold_estimate",
        measurement_density=1.0,
        reclaim_goal_tokens=100,
    )

    _, mismatch, mismatch_usage = cc.compact_tool_history_llm(
        messages, request=bad_request, negative_memo=set(),
    )
    _, insufficient, insufficient_usage = cc.compact_tool_history_llm(
        messages,
        request=_request(messages, 1_000_000, allow_partial=False),
        negative_memo=set(),
    )

    assert mismatch.status == "binding_mismatch"
    assert insufficient.status == "no_positive_reclaim"
    assert mismatch_usage is insufficient_usage is None
    assert checkpoints == []


@pytest.mark.parametrize("measurement_density", [0.0, -1.0, float("inf"), float("nan")])
def test_measurement_density_must_be_finite_and_positive(measurement_density):
    messages = _unit("invalid-density", "x")

    with pytest.raises(ValueError, match="finite and positive"):
        cc.compact_tool_history_llm(
            messages,
            request=_request(
                messages,
                100,
                measurement_density=measurement_density,
            ),
            negative_memo=set(),
        )


def test_real_checkpoint_manifest_binds_complete_selected_transcript(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    messages = _unit("checkpoint", "x")
    request = _request(messages, 100)
    selection, status = cc._select_units(
        messages,
        request,
        keep_recent=0,
        trace_refs_by_tool_call_id={},
        negative_memo=set(),
        spec=_SPEC,
    )
    assert status == "applied"
    checkpoint_ref = cc._persist_reclaim_checkpoint(
        messages,
        request,
        selection,
        drive_root=tmp_path,
        task_id="checkpoint-task",
    )

    assert checkpoint_ref is not None
    assert checkpoint_ref["root"] == "artifact_store"
    payload = json.loads(read_actor_source_bytes(
        tmp_path, "checkpoint-task", checkpoint_ref,
    ))
    assert payload["messages"] == messages
    assert payload["request"]["transcript_sha256"] == request.transcript_sha256
    assert payload["selection_fingerprint"] == selection.fingerprint
    assert payload["selected_unit_ids"] == [item.unit.unit_id for item in selection.units]


def test_checkpoint_requests_exact_private_payload_and_binds_capsule(monkeypatch, tmp_path):
    from ouroboros import observability

    messages = _unit("credential", "x")
    token_shaped_value = "sk-" + "s" * 40
    messages[0]["tool_calls"][0]["function"]["arguments"] = {
        "api_key": token_shaped_value,
        "payload": "x" * 3_000,
    }
    exact_ref = {"path": "calls/exact-checkpoint.json", "sha256": "e" * 64}
    observed: dict = {}

    def persist_exact(_drive_root, *, payload, keep_raw, **kwargs):
        observed.update({"payload": payload, "keep_raw": keep_raw, "kwargs": kwargs})
        return {"manifest_ref": exact_ref}

    monkeypatch.setattr(cc, "_summarizer_spec", lambda: dict(_SPEC))
    monkeypatch.setattr(observability, "persist_call", persist_exact)
    monkeypatch.setattr(
        cc,
        "_call_summarizer",
        lambda parts, **_kwargs: {
            part.source_id: f"summary {part.sha256}" for part in parts
        },
    )

    rebuilt, receipt, _ = cc.compact_tool_history_llm(
        messages,
        request=_request(messages, 100),
        drive_root=tmp_path,
        negative_memo=set(),
    )

    assert receipt.status == "applied"
    assert observed["keep_raw"] is True
    assert observed["payload"]["messages"] == messages
    assert token_shaped_value in json.dumps(observed["payload"], ensure_ascii=False)
    assert receipt.checkpoint_ref != exact_ref
    assert receipt.checkpoint_ref["root"] == "artifact_store"
    from ouroboros.artifacts import read_actor_source_bytes
    actor_payload = json.loads(read_actor_source_bytes(
        tmp_path, "context_compaction", receipt.checkpoint_ref,
    ))
    assert actor_payload["messages"] == messages
    assert token_shaped_value in json.dumps(actor_payload, ensure_ascii=False)
    capsule = rebuilt[0]["content"][0]["_context_capsule"]
    assert capsule["checkpoint_ref"] == receipt.checkpoint_ref
    assert receipt.checkpoint_ref in capsule["source_refs"]


def test_selection_is_minimum_oldest_prefix_toward_deficit(monkeypatch):
    messages = [*_unit("one", "a"), *_unit("two", "b"), *_unit("three", "c")]
    units = cc._atomic_units(messages)
    request = _request(messages, units[0].predicted_reclaim_tokens + 1)

    selection, status = cc._select_units(
        messages,
        request,
        keep_recent=0,
        trace_refs_by_tool_call_id={},
        negative_memo=set(),
        spec=_SPEC,
    )

    assert status == "applied"
    assert [item.unit.unit_id for item in selection.units] == [
        units[0].unit_id,
        units[1].unit_id,
    ]


def test_fresh_density_selects_one_sufficient_unit_and_receipt_uses_same_basis(
    monkeypatch, tmp_path,
):
    _install_pure_dependencies(monkeypatch)
    messages = [*_unit("first", "a"), *_unit("second", "b")]
    first_cold_ceiling = cc._atomic_units(messages)[0].predicted_reclaim_tokens
    goal = first_cold_ceiling + 1
    monkeypatch.setattr(
        cc,
        "_call_summarizer",
        lambda parts, **_kwargs: {
            part.source_id: f"summary {part.sha256}" for part in parts
        },
    )

    rebuilt, receipt, _ = cc.compact_tool_history_llm(
        messages,
        request=_request(
            messages,
            goal,
            measurement_basis="fresh_route_usage",
            measurement_density=2.0,
        ),
        drive_root=tmp_path,
        negative_memo=set(),
    )

    expected_reclaim = (
        cc._context_tokens_for_messages(messages, 2.0)
        - cc._context_tokens_for_messages(rebuilt, 2.0)
    )
    assert receipt.status == "applied"
    assert len(receipt.selected_unit_ids) == 1
    assert rebuilt[1:] == messages[2:]
    assert receipt.reclaimed_tokens == expected_reclaim
    assert receipt.reclaimed_tokens >= goal
    assert receipt.goal_reached is True


def test_safe_image_reclaim_uses_bounded_context_basis(monkeypatch, tmp_path):
    _install_pure_dependencies(monkeypatch)
    messages = _unit("image-basis", "x")
    messages[1]["content"] = [{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + "A" * 400_000},
        "alt": "terminal screenshot showing one failing assertion",
    }]
    monkeypatch.setattr(
        cc,
        "_call_summarizer",
        lambda parts, **_kwargs: {
            part.source_id: "terminal screenshot: one assertion failed" for part in parts
        },
    )

    rebuilt, receipt, _ = cc.compact_tool_history_llm(
        messages,
        request=_request(messages, 5_000),
        drive_root=tmp_path,
        negative_memo=set(),
    )

    expected = (
        cc._context_tokens_for_messages(messages, 1.0)
        - cc._context_tokens_for_messages(rebuilt, 1.0)
    )
    assert receipt.status == "applied"
    assert receipt.reclaimed_tokens == expected
    assert 0 < receipt.reclaimed_tokens < 5_000
    assert receipt.goal_reached is False


def test_safe_image_does_not_hide_productive_text_neighbor():
    image_unit = _unit("image-first", "x")
    image_unit[1]["content"] = [{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + "A" * 400_000},
        "alt": "terminal screenshot showing one failing assertion",
    }]
    text_unit = _unit("text-neighbor", "z")
    text_unit[0]["tool_calls"][0]["function"]["arguments"] = "a" * 40_000
    text_unit[1]["content"] = "b" * 40_000
    messages = [*image_unit, *text_unit]

    units = cc._atomic_units(messages)
    assert units[0].predicted_reclaim_tokens < 5_000
    selection, status = cc._select_units(
        messages,
        _request(messages, 5_000),
        keep_recent=0,
        trace_refs_by_tool_call_id={},
        negative_memo=set(),
        spec=_SPEC,
    )
    assert status == "applied"
    assert selection is not None
    assert [item.unit.unit_id for item in selection.units] == [
        units[0].unit_id,
        units[1].unit_id,
    ]


def test_corrupt_capsule_contract_or_lineage_stays_raw(monkeypatch, tmp_path):
    _install_pure_dependencies(monkeypatch)
    messages = _unit("capsule", "x")
    monkeypatch.setattr(
        cc,
        "_call_summarizer",
        lambda parts, **_kwargs: {
            part.source_id: f"summary {part.sha256}" for part in parts
        },
    )
    rebuilt, receipt, _ = cc.compact_tool_history_llm(
        messages,
        request=_request(messages, 100),
        drive_root=tmp_path,
        negative_memo=set(),
    )
    assert receipt.status == "applied"
    assert len(cc._atomic_units(rebuilt)) == 1

    corruptions = []
    wrong_block = copy.deepcopy(rebuilt)
    wrong_block[0]["content"][0]["type"] = "image_url"
    corruptions.append(wrong_block)
    wrong_version = copy.deepcopy(rebuilt)
    wrong_version[0]["content"][0]["_context_capsule"]["summary_contract_version"] = 999
    corruptions.append(wrong_version)
    wrong_digest = copy.deepcopy(rebuilt)
    wrong_digest[0]["content"][0]["_context_capsule"]["summary_contract_digest"] = "d" * 64
    corruptions.append(wrong_digest)
    wrong_part = copy.deepcopy(rebuilt)
    wrong_part[0]["content"][0]["_context_capsule"]["parts"][0]["sha256"] = "e" * 64
    corruptions.append(wrong_part)

    assert all(cc._atomic_units(candidate) == () for candidate in corruptions)


def test_map_fanout_remains_bounded_at_eight(monkeypatch, tmp_path):
    _install_pure_dependencies(monkeypatch)
    messages = [
        message
        for index in range(17)
        for message in _unit(f"call-{index}", chr(97 + index))
    ]
    map_sizes: list[int] = []

    def summarize(parts, *, phase, **_kwargs):
        assert phase == "map"
        map_sizes.append(len(parts))
        return {part.source_id: f"summary {part.sha256}" for part in parts}

    monkeypatch.setattr(cc, "_call_summarizer", summarize)
    rebuilt, receipt, _ = cc.compact_tool_history_llm(
        messages,
        request=_request(messages, 1_000_000),
        drive_root=tmp_path,
        negative_memo=set(),
    )

    assert receipt.status == "applied"
    assert map_sizes == [8, 8, 1]
    assert len(rebuilt) == 17


def test_tool_names_and_warning_prose_do_not_create_protection_heuristics():
    messages = _unit("warning", "w")
    messages[1]["content"] += "\n⚠️ REVIEW_BLOCKED"

    units = cc._atomic_units(messages)

    assert len(units) == 1
    assert "commit_reviewed" in units[0].source_text
    assert "REVIEW_BLOCKED" in units[0].source_text


def test_opaque_image_is_raw_but_safe_descriptor_is_eligible():
    opaque = _unit("image", "x")
    opaque[1]["content"] = [{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }]
    described = _unit("image", "x")
    described[1]["content"] = [{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
        "alt": "terminal screenshot showing a passing test",
    }]

    assert cc._atomic_units(opaque) == ()
    safe_units = cc._atomic_units(described)
    assert len(safe_units) == 1
    assert "terminal screenshot showing a passing test" in safe_units[0].source_text
    assert "data:image" not in safe_units[0].source_text

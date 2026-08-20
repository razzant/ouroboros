"""Resolving the delivery-control latch without a repair round.

Split verbatim out of ``tests/test_delivery_forced_finalization.py`` by theme. This
module owns the armed latch resolving replace/keep purely, the degradation of a
malformed, unknown-verb or broken-JSON control to the retained candidate, the prose
that stays the answer, and the legitimate JSON that must pass through untouched while
the latch is off.
"""

from __future__ import annotations

import json

from tests._delivery_candidate_shared import (
    write_child as _write_child,
)

from tests._delivery_forced_shared import _forced_test_context

# ---------------------------------------------------------------------------
# F1 (slime saga): a forced finalization while the delivery-control latch is
# armed must RESOLVE the protocol object purely (no repair round — a hard stop
# may not re-loop), never ship raw {"delivery_control": ...} JSON to the chat
# or the durable result, and never eat legitimate JSON when the latch is off.


def _arm_latch_with_candidate(loop, registry, limit_ctx, trace, text="Retained complete answer."):
    candidate = loop._replace_delivery_candidate(
        registry, limit_ctx, trace, text, control="awaiting_control",
    )
    registry._ctx._delivery_control_required = True  # replace() resets the latch
    return candidate


def test_forced_round_limit_resolves_armed_replace_control(tmp_path, monkeypatch):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    control = json.dumps({
        "delivery_control": "replace",
        "full_answer": "Complete replacement answer for the owner.",
    })
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": control}, 0.0),
    )

    text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)

    assert text.startswith("Complete replacement answer for the owner.")
    assert "delivery_control" not in text
    assert registry._ctx._delivery_control_required is False
    assert registry._ctx._delivery_candidate.full_text == text
    assert usage["reason_code"] == "round_limit"


def test_forced_finalization_resolves_armed_keep_to_retained_candidate(tmp_path, monkeypatch):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: (
            {"role": "assistant", "content": '{"delivery_control":"keep"}'}, 0.0,
        ),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="finalization_grace",
    )

    assert text.startswith("Retained complete answer.")
    assert "delivery_control" not in text
    assert registry._ctx._delivery_control_required is False


def test_forced_finalization_degrades_malformed_control_to_retained_candidate(
    tmp_path, monkeypatch,
):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    # Duplicate protocol key -> invalid control object with control intent.
    malformed = '{"delivery_control":"keep","delivery_control":"replace"}'
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": malformed}, 0.0),
    )

    text, _usage, returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Retained complete answer.")
    assert "delivery_control" not in text
    candidate = registry._ctx._delivery_candidate
    assert candidate.degraded is True
    assert candidate.degraded_reason == "delivery_control_degraded"
    assert returned_trace["delivery_candidate"]["degraded_reason"] == "delivery_control_degraded"


def test_forced_finalization_passes_json_through_when_latch_not_armed(tmp_path, monkeypatch):
    """Legitimate user-facing JSON is never eaten while no control round is open."""
    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    legitimate = json.dumps({"delivery_control": "keep"})
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": legitimate}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith(legitimate)


def test_forced_finalization_degrades_unknown_verb_control_to_retained_candidate(
    tmp_path, monkeypatch,
):
    """An armed latch treats ANY parsed object carrying the protocol key as
    protocol — an unknown verb is a mangled control, never the owner's answer."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    unknown_verb = json.dumps({
        "delivery_control": "publish",
        "full_answer": "text behind an unknown verb",
    })
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": unknown_verb}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Retained complete answer.")
    assert "delivery_control" not in text
    assert "publish" not in text
    candidate = registry._ctx._delivery_candidate
    assert candidate.degraded is True
    assert candidate.degraded_reason == "delivery_control_degraded"


def test_forced_finalization_degrades_broken_json_looking_text_to_retained_candidate(
    tmp_path, monkeypatch,
):
    """Armed latch + JSON-looking text that FAILS to parse: the model was
    explicitly instructed to answer with the protocol object, so a broken
    brace-blob is a mangled protocol attempt — resolve to the retained
    candidate with the typed degraded reason; never ship the broken JSON raw."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    broken = '{"delivery_control": "replace", "full_answer": "truncated mid-'
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": broken}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Retained complete answer.")
    assert '{"delivery_control"' not in text
    candidate = registry._ctx._delivery_candidate
    assert candidate.degraded is True
    assert candidate.degraded_reason == "delivery_control_degraded"


def test_forced_finalization_keeps_armed_prose_as_the_answer(tmp_path, monkeypatch):
    """Armed latch + plain prose (not starting with '{'): the fresh text stands
    — the disclosed residual is prose, never anything JSON-looking."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    prose = "A reconsidered complete prose answer for the owner."
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": prose}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith(prose)
    assert registry._ctx._delivery_control_required is False


def test_forced_finalization_passes_broken_json_through_when_latch_not_armed(
    tmp_path, monkeypatch,
):
    """Unarmed: broken JSON-looking output is an ordinary (bad) answer, not a
    protocol attempt — it passes through untouched."""
    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    broken = '{"some_json_like": "output that never closes'
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": broken}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith(broken)


def test_nonforced_resolver_treats_unknown_verb_object_as_protocol_not_prose(tmp_path):
    """The non-forced resolver's gap: an owner-revision round answered with an
    unknown-verb protocol object previously returned it as FRESH prose (raw JSON
    to the owner). It is control intent: the resolver keeps its repair semantics
    (one repair round), never adopting the raw object as the answer."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    from ouroboros import loop_delivery

    candidate = loop._replace_delivery_candidate(
        registry, limit_ctx, trace, "Retained complete answer.", control="candidate",
    )
    candidate.finalization_control = "owner_revision_required"
    registry._ctx._delivery_control_required = False
    unknown_verb = json.dumps({"delivery_control": "finalize"})

    status, text = loop_delivery._resolve_delivery_control(
        unknown_verb, registry, limit_ctx, trace,
    )

    assert status == "retry"
    assert text == ""
    assert candidate.repair_attempted is True
    assert "DELIVERY_CONTROL_REPAIR" in str(limit_ctx.messages[-1]["content"])

    # Second failure after the one repair round degrades to the retained answer.
    status2, text2 = loop_delivery._resolve_delivery_control(
        unknown_verb, registry, limit_ctx, trace,
    )
    assert status2 == "degraded"
    assert text2 == candidate.full_text
    assert "delivery_control" not in text2


def test_children_unabsorbed_forced_path_never_leaks_protocol_json(tmp_path, monkeypatch):
    """The saga leak: children_unabsorbed fired while the latch was armed and the
    model's protocol JSON went RAW into the owner's chat and the durable result."""
    _write_child(tmp_path, status="running")
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    registry._ctx._child_absorption_reminded = True
    control = json.dumps({
        "delivery_control": "replace",
        "full_answer": "Integrated summary naming the unabsorbed child explicitly.",
    })
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": control}, 0.0),
    )

    result = loop._maybe_enforce_child_absorption_gate(
        registry, limit_ctx, "", limit_ctx.messages, lambda _t: None, trace,
    )

    assert result is not None and result != "continue"
    text, usage, _returned_trace = result
    assert text.startswith("Integrated summary naming the unabsorbed child explicitly.")
    assert "delivery_control" not in text
    assert usage["reason_code"] == "children_unabsorbed"
    assert registry._ctx._delivery_candidate.full_text == text

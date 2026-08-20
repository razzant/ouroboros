"""Drift guard for the agent-context budget and reclaim SSOT."""

from __future__ import annotations

import dataclasses
import inspect
import pathlib

import pytest

from ouroboros import context_budget as cb


def _src(rel: str) -> str:
    return pathlib.Path(rel).read_text(encoding="utf-8")


def test_agent_context_budget_values_pinned():
    """Values are the SSOT; changing them is a deliberate, visible edit."""
    assert cb.OWNER_LOW_TARGET_TOKENS == 200_000
    assert cb.BG_CONTEXT_WARN_CHARS == 600_000
    assert cb.BG_CONTEXT_MAX_CHARS == 1_200_000
    assert cb.BG_STATE_JSON_WARN_CHARS == 200_000
    assert cb.LARGE_CONTEXT_SECTION_CHARS == 200_000
    assert cb.MAX_RECENT_CHAT_TAIL == 1000
    assert not hasattr(cb, "CONTEXT_SOFT_CAP_TOKENS")


def test_reclaim_request_and_receipt_are_exact_frozen_records():
    assert [field.name for field in dataclasses.fields(cb.ContextReclaimRequest)] == [
        "route_fp",
        "round_id",
        "transcript_sha256",
        "measurement_basis",
        "measurement_density",
        "reclaim_goal_tokens",
        "allow_partial_shrink",
    ]
    assert [field.name for field in dataclasses.fields(cb.ContextReclaimReceipt)] == [
        "status",
        "before_transcript_sha256",
        "after_transcript_sha256",
        "selection_fingerprint",
        "selected_unit_ids",
        "reclaimed_tokens",
        "goal_reached",
        "checkpoint_ref",
        "capsule_refs",
    ]
    request = cb.ContextReclaimRequest("route", "round", "a" * 64, "cold_estimate", 1.0, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.reclaim_goal_tokens = 2


def test_call_sites_consume_the_ssot_at_runtime():
    """Consuming modules reference the SSOT and no recorder-only cap remains."""
    from ouroboros import context as ctxmod
    from ouroboros.context import build_llm_messages

    assert ctxmod._LARGE_CONTEXT_SECTION_CHARS == cb.LARGE_CONTEXT_SECTION_CHARS
    assert "soft_cap_tokens" not in inspect.signature(build_llm_messages).parameters
    assert not hasattr(ctxmod, "apply_message_token_soft_cap")


def test_call_sites_import_the_ssot_names():
    # v7 L-B split: sweep the whole loop family so no leaf revives the old names.
    loop_src = "".join(
        _src(f"ouroboros/{path.name}")
        for path in sorted(pathlib.Path("ouroboros").glob("loop*.py"))
    )
    for name in (
        "EMERGENCY_COMPACTION_CHARS",
        "LOW_EMERGENCY_COMPACTION_CHARS",
        "COMPACTION_HYSTERESIS_REGION_GROWTH",
        "COMPACTION_HYSTERESIS_ROUNDS",
    ):
        assert not hasattr(cb, name)
        assert name not in loop_src
    assert "OWNER_LOW_TARGET_TOKENS" in _src("ouroboros/context_fit.py")

    ctx_recent_src = _src("ouroboros/context.py")
    assert "MAX_RECENT_CHAT_TAIL" in ctx_recent_src
    assert "consolidated_offset > 0" in ctx_recent_src

    consc_src = _src("ouroboros/consciousness.py")
    for name in ("BG_CONTEXT_MAX_CHARS", "BG_CONTEXT_WARN_CHARS", "BG_STATE_JSON_WARN_CHARS"):
        assert name in consc_src, f"consciousness.py must consume {name}"

    ctx_src = _src("ouroboros/context.py")
    assert "LARGE_CONTEXT_SECTION_CHARS" in ctx_src
    assert "CONTEXT_SOFT_CAP_TOKENS" not in ctx_src
    assert "soft_cap_tokens" not in ctx_src
    assert "CONTEXT_SOFT_CAP_TOKENS" not in _src("ouroboros/agent.py")
    assert "_soft_cap" not in _src("ouroboros/agent.py")


def test_old_bare_literals_are_gone_from_call_sites():
    """The decisive anti-drift check: no bare literal can outlive the SSOT."""
    for path in sorted(pathlib.Path("ouroboros").glob("loop*.py")):
        assert "> 1_200_000" not in _src(f"ouroboros/{path.name}"), path.name

    consc = _src("ouroboros/consciousness.py")
    assert "= 1_200_000" not in consc
    assert "= 600_000" not in consc
    assert "> 200_000" not in consc

    ctx = _src("ouroboros/context.py")
    assert "= 200_000" not in ctx
    assert "_soft_cap = 200_000" not in _src("ouroboros/agent.py")


def test_overflow_classification_is_one_shared_seam():
    """Main, the local transport, and the summarizer must classify overflow
    through the SAME context_budget helper and constants — a seam keeping a
    private marker copy or private output-size precedence regresses S1 N-1."""
    import ouroboros.context_compaction as cc_mod
    import ouroboros.llm as llm_mod
    import ouroboros.loop_llm_call as loop_call_mod

    assert llm_mod.context_overflow_message is cb.context_overflow_message
    assert cc_mod._context_overflow_message is cb.context_overflow_message
    assert loop_call_mod._context_overflow_message is cb.context_overflow_message
    assert loop_call_mod._output_or_body_size_message is cb.output_or_body_size_message
    assert llm_mod.CONTEXT_OVERFLOW_CODES is cb.CONTEXT_OVERFLOW_CODES
    assert cc_mod._TYPED_CONTEXT_OVERFLOW_CODES is cb.CONTEXT_OVERFLOW_CODES
    assert loop_call_mod._STRUCTURED_CONTEXT_OVERFLOW_CODES is cb.CONTEXT_OVERFLOW_CODES
    # No module keeps a private marker tuple beside the shared one.
    for mod in (llm_mod, cc_mod, loop_call_mod):
        assert not hasattr(mod, "_OUTPUT_OR_BODY_SIZE_MARKERS")


def test_output_size_precedence_lives_inside_the_shared_helper():
    """'max_tokens 65536 exceeds maximum context length 32768' is an OUTPUT
    limit: shrinking the prompt cannot fix it, so it is not a window overflow."""
    probe = "max_tokens 65536 exceeds maximum context length 32768"
    assert cb.output_or_body_size_message(probe)
    assert not cb.context_overflow_message(probe)
    assert cb.context_overflow_message("prompt is too long: 250000 tokens > 200000 maximum")
    assert not cb.context_overflow_message("request body too large")

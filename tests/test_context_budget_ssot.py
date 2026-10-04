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
    # Owner choice of 2026-09-29 (Low 250K, Nano 85K); the memory view's working room
    # is two low-water levels under a window, one under an owner Low/Nano target.
    assert cb.OWNER_LOW_TARGET_TOKENS == 250_000
    assert cb.OWNER_NANO_TARGET_TOKENS == 85_000
    assert cb.MEMORY_VIEW_WORKING_MARGINS == 2
    assert cb.MODE_TARGET_WORKING_MARGINS == 1
    for retired in ("BG_CONTEXT_WARN_CHARS", "BG_CONTEXT_MAX_CHARS", "BG_STATE_JSON_WARN_CHARS", "BG_OBSERVATIONS_WARN_BYTES"):
        assert not hasattr(cb, retired), retired  # a wake-up is a Main turn under Main's budgets
    assert cb.LARGE_CONTEXT_SECTION_CHARS == 200_000
    assert not hasattr(cb, "MAX_RECENT_CHAT_TAIL")  # the request reads no chat tail: the memory view holds the open conversation
    assert cb.CHAT_ARCHIVE_SCAN_WARN_BYTES == 100_000_000
    assert not hasattr(cb, "CONTEXT_SOFT_CAP_TOKENS")
    # Structural low-water divisor of the automatic reclaim pass (12.5 % of the
    # binding boundary): a disclosed design choice, not a setting.
    assert cb.RECLAIM_LOW_WATER_DIVISOR == 8


def test_reclaim_low_water_divisor_is_one_constant_read_at_call_time(monkeypatch):
    """CHECKLISTS item 20: the fit consumes the SSOT name (no bare literal), reads it
    at call time so changing the one constant changes every pass, and the margin
    is the LAST measurement field (appended; older readers stay positional-safe)."""
    from ouroboros import context_fit

    assert "RECLAIM_LOW_WATER_DIVISOR" in _src("ouroboros/context_fit.py")
    assert "/ 8" not in inspect.getsource(context_fit.measure_main_fit)
    assert dataclasses.fields(context_fit.MainFitMeasurement)[-1].name == "low_water_margin_tokens"
    assert context_fit.reclaim_low_water_margin(250_000, 500_000) == 31_250  # target binds
    assert context_fit.reclaim_low_water_margin(None, 70_000) == 8_750  # capacity alone
    assert context_fit.reclaim_low_water_margin(None, None) == 0  # nothing known
    monkeypatch.setattr(cb, "RECLAIM_LOW_WATER_DIVISOR", 4)
    assert context_fit.reclaim_low_water_margin(250_000, 500_000) == 62_500


def test_reclaim_request_and_receipt_are_exact_frozen_records():
    assert [field.name for field in dataclasses.fields(cb.ContextReclaimRequest)] == [
        "route_fp",
        "round_id",
        "transcript_sha256",
        "measurement_basis",
        "measurement_density",
        "reclaim_goal_tokens",
        "allow_partial_shrink",
        "working_note", "expected_view_revision", "keep_unit_ids", "restore_unit_refs", "schema_names",
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
        "observed_view_revision", "view_revision", "retained_unit_ids", "restored_unit_refs",
        "source_refs", "schema_names", "fit",
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
    loop_src = _src("ouroboros/loop.py")
    for name in (
        "EMERGENCY_COMPACTION_CHARS",
        "LOW_EMERGENCY_COMPACTION_CHARS",
        "COMPACTION_HYSTERESIS_REGION_GROWTH",
        "COMPACTION_HYSTERESIS_ROUNDS",
    ):
        assert not hasattr(cb, name)
        assert name not in loop_src
    assert "OWNER_LOW_TARGET_TOKENS" in _src("ouroboros/context_fit.py")

    # The request reads no chat tail: the open conversation is the memory view's, and the
    # readers of the frozen consolidation cursor's tail are gone with it.
    from ouroboros.memory import Memory

    ctx_recent_src = _src("ouroboros/context.py")
    assert "MAX_RECENT_CHAT_TAIL" not in ctx_recent_src
    assert "read_unconsolidated_chat" not in ctx_recent_src
    assert "capture_memory_view" in ctx_recent_src
    memory_src = _src("ouroboros/memory.py")
    assert "last_consolidated_offset" not in memory_src and "_AUTOMATIC_CHAT_" not in memory_src
    for retired in ("read_unconsolidated_chat", "summarize_chat", "format_blocks_as_markdown", "era_host_note"):
        assert not hasattr(Memory, retired), retired
    for kept in ("chat_history", "load_dialogue_blocks", "_durable_dialogue_gaps", "_format_chat_line"):
        assert hasattr(Memory, kept), kept  # chat_history still reads them

    ctx_src = _src("ouroboros/context.py")
    assert "LARGE_CONTEXT_SECTION_CHARS" in ctx_src
    assert "CONTEXT_SOFT_CAP_TOKENS" not in ctx_src
    assert "soft_cap_tokens" not in ctx_src
    assert "CONTEXT_SOFT_CAP_TOKENS" not in _src("ouroboros/agent.py")
    assert "_soft_cap" not in _src("ouroboros/agent.py")


def test_old_bare_literals_are_gone_from_call_sites():
    """The decisive anti-drift check: no bare literal can outlive the SSOT."""
    assert "> 1_200_000" not in _src("ouroboros/loop.py")

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


@pytest.mark.parametrize("reserve", ["max_tokens", "`max_tokens`"])
def test_combined_anthropic_window_rejection_reaches_main_and_advisory(reserve):
    from ouroboros.loop_llm_call import classify_llm_exception
    from ouroboros.tools.claude_advisory_review import _overflow_failure_text

    # The input alone fits; input plus the requested response exceeds the window.
    text = f"input length and {reserve} exceed context limit: 197202 + 21333 > 200000"
    error = RuntimeError(text)
    error.status_code = 400
    error.body = {"error": {"type": "invalid_request_error", "message": text}}
    assert cb.context_overflow_message(text)
    assert not cb.output_or_body_size_message(text)
    assert classify_llm_exception(error).kind == "context_overflow"
    assert _overflow_failure_text(text)
    for output_only in ("max_tokens 65536 exceeds maximum context length 32768",
                        "output tokens exceed context limit", "request body too large"):
        assert cb.output_or_body_size_message(output_only)
        assert not cb.context_overflow_message(output_only)
        assert not _overflow_failure_text(output_only)

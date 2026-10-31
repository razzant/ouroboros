"""
Tests for budget/cost tracking across all tools and pipeline components.
Verifies that real LLM spend from review usage, plan_task, reflection,
consolidation, scope review, and supervisor dedup all reach accounting.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _FakeCtx:
    """Minimal ToolContext stub."""
    def __init__(self):
        self.task_id = "test-task-001"
        self.event_queue = None
        self.pending_events: List[Dict[str, Any]] = []
        self.repo_dir = "/fake/repo"
        self.emit_progress_fn = lambda msg: None


# ---------------------------------------------------------------------------
# Review usage emission (the shared helper every review surface uses)
# ---------------------------------------------------------------------------

class TestAdvisoryUsageEmit:
    """Review usage emission must route through the shared review helper."""

    def _get_fn(self):
        from ouroboros.tools.review_helpers import emit_review_usage
        def _emit(ctx, model, cost_usd, usage, source="advisory", provider="anthropic", session_id="", prompt_chars=0):
            return emit_review_usage(
                ctx,
                model=model,
                provider=provider,
                usage=usage,
                source=source,
                cost_usd=cost_usd,
                session_id=session_id,
                prompt_chars=prompt_chars,
            )
        return _emit

    def test_emit_routes_to_pending_events(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        fn(
            ctx,
            "anthropic/claude-opus-4.6",
            1.23,
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 10,
            },
            session_id="sess-1",
            prompt_chars=1234,
        )
        assert len(ctx.pending_events) == 1
        ev = ctx.pending_events[0]
        assert ev["type"] == "llm_usage"
        assert ev["model"] == "anthropic/claude-opus-4.6"
        assert ev["usage"]["cost"] == 1.23
        assert ev["usage"]["prompt_tokens"] == 100
        assert ev["usage"]["completion_tokens"] == 50
        assert ev["usage"]["cached_tokens"] == 20
        assert ev["usage"]["cache_write_tokens"] == 10
        assert ev["session_id"] == "sess-1"
        assert ev["prompt_chars"] == 1234

    def test_emit_uses_event_queue_when_available(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        ctx.event_queue = MagicMock()
        ctx.event_queue.put_nowait = MagicMock()
        fn(ctx, "anthropic/claude-sonnet-4.6", 0.50, {})
        ctx.event_queue.put_nowait.assert_called_once()
        # Should NOT fall through to pending_events
        assert len(ctx.pending_events) == 0

    def test_emit_source_field(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        fn(ctx, "model-x", 0.0, {}, source="advisory_fallback")
        assert ctx.pending_events[0]["source"] == "advisory_fallback"

    def test_emit_sdk_source_default(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        fn(ctx, "model-x", 0.0, {})
        assert ctx.pending_events[0]["source"] == "advisory"

    def test_emit_noop_on_exception(self):
        """_emit_advisory_usage must never raise — it's a non-critical helper."""
        fn = self._get_fn()
        ctx = _FakeCtx()
        # Pass a broken usage dict (cause internal error)
        fn(ctx, None, "not-a-float", object())  # type: ignore[arg-type]
        # No exception — pending_events may or may not have an entry


@pytest.mark.parametrize("model,expected_provider", [
    ("anthropic::claude-opus-4.6", "anthropic"),
    ("openai::gpt-5.5", "openai"),
    ("openai-compatible::my-model", "openai-compatible"),
    ("cloudru::GigaChat-2-Max", "cloudru"),
    ("gigachat::GigaChat-3-Ultra", "gigachat"),
    ("minimax::MiniMax-M3", "minimax"),
    ("anthropic/claude-opus-4.6", "openrouter"),  # unprefixed → OpenRouter
    ("google/gemini-3.5-flash", "openrouter"),
    ("", "openrouter"),
])
def test_infer_provider_from_model(model, expected_provider):
    """infer_provider_from_model must return correct provider for all prefixes."""
    from ouroboros.pricing import infer_provider_from_model
    assert infer_provider_from_model(model) == expected_provider


class TestScopeReviewProviderAttribution:
    """_emit_usage in scope_review must use correct provider per model prefix."""

    def _get_fn(self):
        from ouroboros.tools.review_helpers import emit_review_usage
        return lambda ctx, model, usage: emit_review_usage(ctx, model=model, usage=usage, source="scope_review")

    @pytest.mark.parametrize("model,expected_provider", [
        ("anthropic::claude-opus-4.6", "anthropic"),
        ("openai::gpt-5.5", "openai"),
        ("minimax::MiniMax-M3", "minimax"),
        ("anthropic/claude-opus-4.6", "openrouter"),
    ])
    def test_provider_per_model_prefix(self, model, expected_provider):
        fn = self._get_fn()
        ctx = _FakeCtx()
        fn(ctx, model, {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.05})
        assert len(ctx.pending_events) == 1
        ev = ctx.pending_events[0]
        assert ev["provider"] == expected_provider


class TestAdvisoryFallbackProviderAttribution:
    """A review fallback's provider kwarg must reflect the fallback model prefix."""

    def _get_fn(self):
        from ouroboros.tools.review_helpers import emit_review_usage
        def _emit(ctx, model, cost_usd, usage, source="advisory", provider="anthropic", session_id="", prompt_chars=0):
            return emit_review_usage(
                ctx,
                model=model,
                provider=provider,
                usage=usage,
                source=source,
                cost_usd=cost_usd,
                session_id=session_id,
                prompt_chars=prompt_chars,
            )
        return _emit

    @pytest.mark.parametrize("model,expected_provider", [
        ("anthropic::claude-3-5-sonnet", "anthropic"),
        ("openai::gpt-5.5-mini", "openai"),
        ("anthropic/claude-sonnet-4.6", "openrouter"),  # un-prefixed → openrouter
    ])
    def test_provider_kwarg_propagated(self, model, expected_provider):
        fn = self._get_fn()
        ctx = _FakeCtx()
        fn(ctx, model, 0.05, {"prompt_tokens": 100}, "advisory_fallback", provider=expected_provider)
        assert len(ctx.pending_events) == 1
        ev = ctx.pending_events[0]
        assert ev["provider"] == expected_provider


class TestReviewUsageFallback:
    """``emit_review_usage`` (review_helpers) must fall back to pending_events."""

    def _get_fn(self):
        from ouroboros.tools.review_helpers import emit_review_usage
        return lambda ctx, model, usage: emit_review_usage(ctx, model=model, usage=usage, source="review")

    def test_routes_to_pending_events_when_no_queue(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        fn(ctx, "anthropic/claude-opus-4.6", {"prompt_tokens": 80, "completion_tokens": 30, "cost": 0.5})
        assert len(ctx.pending_events) == 1
        assert ctx.pending_events[0]["type"] == "llm_usage"

    def test_uses_event_queue_when_available(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        ctx.event_queue = MagicMock()
        ctx.event_queue.put_nowait = MagicMock()
        fn(ctx, "model-x", {})
        ctx.event_queue.put_nowait.assert_called_once()
        assert len(ctx.pending_events) == 0

    def test_pending_fallback_on_queue_error(self):
        """When event_queue.put_nowait raises, fall through to pending_events."""
        fn = self._get_fn()
        ctx = _FakeCtx()
        ctx.event_queue = MagicMock()
        ctx.event_queue.put_nowait = MagicMock(side_effect=Exception("full"))
        fn(ctx, "model-x", {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01})
        assert len(ctx.pending_events) == 1


# ---------------------------------------------------------------------------
# Post-task memory stage cost tracking
# ---------------------------------------------------------------------------

class TestFallbackDraftCostTracking:
    """The paid post-task memory stage must call update_budget_from_usage when its draft spent
    something. (The scratchpad consolidation stage this class used to pin is retired, #918; the
    fallback draft stage is the remaining memory stage with the same accounting.)"""

    def test_update_budget_called_after_a_paid_fallback_draft(self, tmp_path):
        from ouroboros.agent_task_pipeline import _run_memory_fallback_draft
        from ouroboros import memory_fallback

        usage_dict = {"prompt_tokens": 200, "completion_tokens": 100, "cost": 0.02}
        env = MagicMock()
        env.drive_root = tmp_path
        run = memory_fallback.FallbackRun("nothing", usage=usage_dict)

        with patch.object(memory_fallback, "run_fallback_draft", return_value=run), \
             patch("supervisor.state.update_budget_from_usage") as mock_budget:
            assert _run_memory_fallback_draft(env, {"id": "t"}, MagicMock(), tmp_path / "logs", {}) == ""
            mock_budget.assert_called_once_with(usage_dict)

    def test_no_budget_call_when_the_draft_bought_nothing(self, tmp_path):
        from ouroboros.agent_task_pipeline import _run_memory_fallback_draft
        from ouroboros import memory_fallback

        env = MagicMock()
        env.drive_root = tmp_path

        with patch.object(memory_fallback, "run_fallback_draft", return_value=memory_fallback.FallbackRun("not_activated")), \
             patch("supervisor.state.update_budget_from_usage") as mock_budget:
            assert _run_memory_fallback_draft(env, {"id": "t"}, MagicMock(), tmp_path / "logs", {}) == ""
            mock_budget.assert_not_called()

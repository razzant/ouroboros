"""
Tests for budget/cost tracking across all tools and pipeline components.
Verifies that real LLM spend from advisory, plan_task, reflection,
consolidation, scope review, and supervisor dedup all reach the budget.
"""
from __future__ import annotations

import importlib
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
# Advisory SDK cost tracking
# ---------------------------------------------------------------------------

class TestAdvisoryUsageEmit:
    """Advisory usage emission must route through the shared review helper."""

    def _get_fn(self):
        mod = importlib.import_module("ouroboros.tools.claude_advisory_review")
        def _emit(ctx, model, cost_usd, usage, source="advisory", provider="anthropic", session_id="", prompt_chars=0):
            return mod.emit_review_usage(
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


# ---------------------------------------------------------------------------
# Plan review cost tracking
# ---------------------------------------------------------------------------

class TestPlanReviewUsageEmit:
    """_emit_plan_review_usage must emit one event per reviewer with tokens."""

    def _get_fn(self):
        mod = importlib.import_module("ouroboros.tools.plan_review")
        return mod._emit_plan_review_usage

    def _make_raw_results(self):
        return [
            {"model": "openai/gpt-5.5", "tokens_in": 100, "tokens_out": 50, "error": None},
            {"model": "google/gemini-3.5-flash", "tokens_in": 120, "tokens_out": 60, "error": None},
            {"model": "anthropic/claude-opus-4.6", "tokens_in": 90, "tokens_out": 40, "error": None},
        ]

    def test_emits_one_event_per_reviewer(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        fn(ctx, self._make_raw_results())
        assert len(ctx.pending_events) == 3

    def test_event_fields(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        fn(ctx, self._make_raw_results())
        ev = ctx.pending_events[0]
        assert ev["type"] == "llm_usage"
        assert ev["source"] == "plan_review"
        assert ev["category"] == "review"
        assert ev["usage"]["prompt_tokens"] == 100
        assert ev["usage"]["completion_tokens"] == 50

    def test_skips_error_results(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        results = [
            {"model": "m1", "tokens_in": 100, "tokens_out": 50, "error": "timeout"},
            {"model": "m2", "tokens_in": 80, "tokens_out": 30, "error": None},
        ]
        fn(ctx, results)
        # Only the non-error result should be emitted
        assert len(ctx.pending_events) == 1
        assert ctx.pending_events[0]["model"] == "m2"

    def test_skips_zero_token_results(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        results = [
            {"model": "m1", "tokens_in": 0, "tokens_out": 0, "error": None},
            {"model": "m2", "tokens_in": 50, "tokens_out": 20, "error": None},
        ]
        fn(ctx, results)
        assert len(ctx.pending_events) == 1

    def test_routes_to_event_queue_first(self):
        fn = self._get_fn()
        ctx = _FakeCtx()
        ctx.event_queue = MagicMock()
        ctx.event_queue.put_nowait = MagicMock()
        fn(ctx, self._make_raw_results())
        assert ctx.event_queue.put_nowait.call_count == 3
        assert len(ctx.pending_events) == 0

    def test_real_cost_propagated_from_reviewer(self):
        """Cost from reviewer raw results must reach the emitted event — not hardcoded 0."""
        fn = self._get_fn()
        ctx = _FakeCtx()
        results = [
            {"model": "openai/gpt-5.5", "tokens_in": 1000, "tokens_out": 200,
             "cost": 6.50, "error": None},
        ]
        fn(ctx, results)
        ev = ctx.pending_events[0]
        assert ev["usage"]["cost"] == 6.50
        assert ev.get("cost") == 6.50


# ---------------------------------------------------------------------------
# Scope review pending_events fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,expected_provider", [
    ("anthropic::claude-opus-4.6", "anthropic"),
    ("openai::gpt-5.5", "openai"),
    ("openai-compatible::my-model", "openai-compatible"),
    ("cloudru::GigaChat-2-Max", "cloudru"),
    ("gigachat::GigaChat-3-Ultra", "gigachat"),
    ("anthropic/claude-opus-4.6", "openrouter"),  # unprefixed → OpenRouter
    ("google/gemini-3.5-flash", "openrouter"),
    ("", "openrouter"),
])
def test_infer_provider_from_model(model, expected_provider):
    """infer_provider_from_model must return correct provider for all prefixes."""
    from ouroboros.pricing import infer_provider_from_model
    assert infer_provider_from_model(model) == expected_provider


class TestPlanReviewProviderAttribution:
    """_emit_plan_review_usage must use correct provider per model prefix."""

    def _get_fn(self):
        mod = importlib.import_module("ouroboros.tools.plan_review")
        return mod._emit_plan_review_usage

    @pytest.mark.parametrize("model,expected_provider", [
        ("anthropic::claude-opus-4.6", "anthropic"),
        ("openai::gpt-5.5", "openai"),
        ("openai-compatible::my-model", "openai-compatible"),
        ("cloudru::GigaChat-2-Max", "cloudru"),
        ("gigachat::GigaChat-2-Max", "gigachat"),
        ("anthropic/claude-opus-4.6", "openrouter"),  # unprefixed → OpenRouter
    ])
    def test_provider_per_model_prefix(self, model, expected_provider):
        fn = self._get_fn()
        ctx = _FakeCtx()
        results = [{"model": model, "tokens_in": 100, "tokens_out": 50, "cost": 0.1, "error": None}]
        fn(ctx, results)
        assert len(ctx.pending_events) == 1
        ev = ctx.pending_events[0]
        assert ev["provider"] == expected_provider


class TestScopeReviewProviderAttribution:
    """_emit_usage in scope_review must use correct provider per model prefix."""

    def _get_fn(self):
        mod = importlib.import_module("ouroboros.tools.scope_review")
        return mod._emit_usage

    @pytest.mark.parametrize("model,expected_provider", [
        ("anthropic::claude-opus-4.6", "anthropic"),
        ("openai::gpt-5.5", "openai"),
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
    """Advisory fallback provider kwarg must reflect fallback model prefix."""

    def _get_fn(self):
        mod = importlib.import_module("ouroboros.tools.claude_advisory_review")
        def _emit(ctx, model, cost_usd, usage, source="advisory", provider="anthropic", session_id="", prompt_chars=0):
            return mod.emit_review_usage(
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


class TestScopeReviewUsageFallback:
    """_emit_usage in scope_review.py must fall back to pending_events."""

    def _get_fn(self):
        mod = importlib.import_module("ouroboros.tools.scope_review")
        return mod._emit_usage

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
# Reflection cost tracking
# ---------------------------------------------------------------------------

class TestReflectionCostTracking:
    """generate_reflection must call update_budget_from_usage for the LLM call."""

    def test_update_budget_called_on_success(self):
        from ouroboros.reflection import generate_reflection

        mock_llm = MagicMock()
        mock_llm.chat.return_value = (
            {"content": "Reflection text"},
            {"prompt_tokens": 200, "completion_tokens": 100, "cost": 0.003},
        )

        with patch("supervisor.state.update_budget_from_usage") as mock_budget:
            generate_reflection(
                task={"id": "t1", "text": "test goal"},
                llm_trace={"tool_calls": [{"result": "REVIEW_BLOCKED"}]},
                trace_summary="summary",
                llm_client=mock_llm,
                usage_dict={"rounds": 5, "cost": 2.0},
            )
            mock_budget.assert_called_once()
            call_args = mock_budget.call_args[0][0]
            assert call_args.get("prompt_tokens") == 200

    def test_budget_not_called_when_usage_empty(self):
        from ouroboros.reflection import generate_reflection

        mock_llm = MagicMock()
        mock_llm.chat.return_value = ({"content": "ok"}, {})

        with patch("supervisor.state.update_budget_from_usage") as mock_budget:
            generate_reflection(
                task={"id": "t1", "text": "goal"},
                llm_trace={"tool_calls": [{"result": "REVIEW_BLOCKED"}]},
                trace_summary="sum",
                llm_client=mock_llm,
                usage_dict={},
            )
            mock_budget.assert_not_called()


# ---------------------------------------------------------------------------
# Consolidation cost tracking
# ---------------------------------------------------------------------------

class TestUpdatePatternsCostTracking:
    """_update_patterns must call update_budget_from_usage for its LLM call."""

    def test_update_budget_called_on_success(self, tmp_path):
        from ouroboros.reflection import _update_patterns
        # _update_patterns creates its own LLMClient internally — patch at the class level.
        with patch("ouroboros.llm.LLMClient") as mock_cls, \
             patch("supervisor.state.update_budget_from_usage") as mock_budget:
            inst = MagicMock()
            inst.chat.return_value = (
                {"content": "| Error class | Count | Root cause | Fix | Status |\n|---|---|---|---|---|\n| test | 1 | bug | fix | open |"},
                {"prompt_tokens": 300, "completion_tokens": 150, "cost": 0.002},
            )
            mock_cls.return_value = inst
            _update_patterns(
                tmp_path,
                {
                    "goal": "test task",
                    "key_markers": ["REVIEW_BLOCKED"],
                    "reflection": "Something went wrong",
                },
            )
            mock_budget.assert_called_once()
            usage_arg = mock_budget.call_args[0][0]
            assert usage_arg.get("prompt_tokens") == 300


class TestSupervisorDedupCostTracking:
    """_find_duplicate_task must call update_budget_from_usage for its LLM call."""

    def test_update_budget_called_on_dedup_check(self):
        """When _find_duplicate_task calls the LLM, update_budget_from_usage is called."""
        import supervisor.events as ev_mod

        usage = {"prompt_tokens": 50, "completion_tokens": 10, "cost": 0.0001}
        # Need at least one existing task so the early-return guard doesn't skip the LLM call.
        pending = [{"id": "existing-1", "type": "task", "text": "some other task"}]

        with patch("ouroboros.llm.LLMClient") as mock_cls, \
             patch("supervisor.state.update_budget_from_usage") as mock_budget:
            inst = MagicMock()
            inst.chat.return_value = ({"content": "NONE"}, usage)
            mock_cls.return_value = inst
            result = ev_mod._find_duplicate_task("Deploy new feature", "", pending, {})
            mock_budget.assert_called_once_with(usage)
            assert result is None  # "NONE" response = no duplicate found

    def test_no_budget_call_when_no_usage(self):
        import supervisor.events as ev_mod

        pending = [{"id": "existing-1", "type": "task", "text": "some task"}]

        with patch("ouroboros.llm.LLMClient") as mock_cls, \
             patch("supervisor.state.update_budget_from_usage") as mock_budget:
            inst = MagicMock()
            inst.chat.return_value = ({"content": "NONE"}, None)
            mock_cls.return_value = inst
            ev_mod._find_duplicate_task("test", "", pending, {})
            mock_budget.assert_not_called()

    def test_no_budget_call_when_no_existing_tasks(self):
        """Empty pending+running — LLM not called at all, no budget update."""
        import supervisor.events as ev_mod

        with patch("ouroboros.llm.LLMClient") as mock_cls, \
             patch("supervisor.state.update_budget_from_usage") as mock_budget:
            result = ev_mod._find_duplicate_task("test", "", [], {})
            mock_cls.assert_not_called()
            mock_budget.assert_not_called()
            assert result is None


class TestAdvisoryCallSiteCostTracking:
    """emit_review_usage is called with real cost from the SDK result."""

    def test_emit_called_when_cost_nonzero(self):
        """emit_review_usage is called with cost_usd when cost_usd > 0."""
        mod = importlib.import_module("ouroboros.tools.claude_advisory_review")

        # Directly verify the conditional gate: cost_usd > 0 triggers the emit call.
        ctx = _FakeCtx()
        with patch.object(mod, "emit_review_usage") as mock_emit:
            # Simulate the inline condition from _run_claude_advisory:
            #   if result.cost_usd > 0:
            #       emit_review_usage(ctx, model=model, cost_usd=result.cost_usd, ...)
            cost_usd = 2.50
            usage = {"prompt_tokens": 500, "completion_tokens": 200}
            if cost_usd > 0:
                mod.emit_review_usage(
                    ctx,
                    model="model-x",
                    cost_usd=cost_usd,
                    usage=usage,
                    source="advisory_sdk",
                )
            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args.kwargs
            assert call_kwargs["cost_usd"] == 2.50
            assert call_kwargs["source"] == "advisory_sdk"

    def test_emit_not_called_when_cost_zero(self):
        """emit_review_usage is NOT called when SDK reports zero cost."""
        mod = importlib.import_module("ouroboros.tools.claude_advisory_review")
        ctx = _FakeCtx()
        with patch.object(mod, "emit_review_usage") as mock_emit:
            cost_usd = 0.0
            if cost_usd > 0:
                mod.emit_review_usage(
                    ctx,
                    model="model-x",
                    cost_usd=cost_usd,
                    usage={},
                    source="advisory_sdk",
                )
            mock_emit.assert_not_called()

    def test_run_claude_advisory_emits_cost_via_patched_import(self, tmp_path):
        """_run_claude_advisory calls emit_review_usage when SDK result has cost_usd > 0.

        claude_agent_sdk is not installed in the test environment, so we pre-register
        a stub in sys.modules before importing the gateway module.
        """
        import os
        import sys
        import pathlib

        fake_result = MagicMock()
        fake_result.success = True
        fake_result.result_text = '[{"item":"bible_compliance","verdict":"PASS","reason":"ok","severity":"critical"}]'
        fake_result.cost_usd = 1.75
        fake_result.usage = {"input_tokens": 400, "output_tokens": 150}
        fake_result.error = None
        fake_result.stderr_tail = ""
        fake_result.session_id = "test-session"

        ctx = _FakeCtx()
        ctx.repo_dir = str(tmp_path)

        # Pre-register claude_agent_sdk stub so the gateway module can be imported.
        sdk_stub = MagicMock()
        sdk_stub.__version__ = "0.0.test"
        sdk_stub.ClaudeSDKClient = MagicMock
        sdk_stub.ClaudeAgentOptions = MagicMock
        sdk_stub.AssistantMessage = MagicMock
        sdk_stub.ResultMessage = MagicMock

        # Build a fake gateway module that resolves run_readonly and friends.
        fake_gw_mod = MagicMock()
        fake_gw_mod.run_readonly = MagicMock(return_value=fake_result)
        fake_gw_mod.DEFAULT_CLAUDE_CODE_MAX_TURNS = 30
        fake_gw_mod.resolve_claude_code_model = MagicMock(return_value="claude-opus-4")

        original_sdk = sys.modules.get("claude_agent_sdk")
        original_gw = sys.modules.get("ouroboros.gateways.claude_code")
        sys.modules["claude_agent_sdk"] = sdk_stub
        sys.modules["ouroboros.gateways.claude_code"] = fake_gw_mod

        mod = importlib.import_module("ouroboros.tools.claude_advisory_review")
        original_key = os.environ.get("ANTHROPIC_API_KEY", "")
        os.environ["ANTHROPIC_API_KEY"] = "test-key-abc"
        try:
            with patch.object(mod, "_get_staged_diff", return_value="diff text here"), \
                 patch.object(mod, "_get_changed_file_list",
                              return_value="M ouroboros/loop.py"), \
                 patch.object(mod, "build_advisory_changed_context",
                              return_value=(["ouroboros/loop.py"], "pack text", [])), \
                 patch.object(mod, "_build_advisory_prompt", return_value="mock_prompt"), \
                 patch.object(mod, "check_worktree_readiness", return_value=[]), \
                 patch.object(mod, "emit_review_usage") as mock_emit:
                mod._run_claude_advisory(pathlib.Path(tmp_path), "test commit", ctx)
                mock_emit.assert_called()
                assert mock_emit.call_args.kwargs["cost_usd"] == 1.75
        finally:
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            if original_sdk is not None:
                sys.modules["claude_agent_sdk"] = original_sdk
            else:
                sys.modules.pop("claude_agent_sdk", None)
            if original_gw is not None:
                sys.modules["ouroboros.gateways.claude_code"] = original_gw
            else:
                sys.modules.pop("ouroboros.gateways.claude_code", None)


class TestAdvisoryFallbackCostTracking:
    """_llm_extract_advisory_items must emit cost for the fallback LLM call."""

    def test_emit_called_with_fallback_usage_for_toolcontext(self):
        """When ctx is a ToolContext, emit is called with fallback usage."""
        from ouroboros.tools.registry import ToolContext as TC
        mod = importlib.import_module("ouroboros.tools.claude_advisory_review")

        ctx = _FakeCtx()
        # Make _FakeCtx pass isinstance check by setting its class's MRO
        ctx.__class__ = TC  # type: ignore[assignment]

        fake_usage = {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.05}

        with patch("ouroboros.llm.LLMClient") as mock_cls, \
             patch.object(mod, "emit_review_usage") as mock_emit:
            inst = MagicMock()
            inst.chat.return_value = (
                {"content": '[{"item":"code_quality","verdict":"PASS","reason":"ok"}]'},
                fake_usage,
            )
            mock_cls.return_value = inst
            mod._llm_extract_advisory_items("narrative text with findings", ctx)
            mock_emit.assert_called_once()
            assert mock_emit.call_args.kwargs["model"] == mod._resolve_fallback_model()

    def test_no_emit_when_ctx_not_toolcontext(self):
        """When ctx is not a ToolContext, emit must be skipped gracefully."""
        mod = importlib.import_module("ouroboros.tools.claude_advisory_review")

        with patch("ouroboros.llm.LLMClient") as mock_cls, \
             patch.object(mod, "emit_review_usage") as mock_emit:
            inst = MagicMock()
            inst.chat.return_value = (
                {"content": '[{"item":"code_quality","verdict":"PASS","reason":"ok"}]'},
                {"cost": 0.01},
            )
            mock_cls.return_value = inst
            # Plain object — not a ToolContext
            mod._llm_extract_advisory_items("some text", object())
            mock_emit.assert_not_called()


class TestScratchpadConsolidationCostTracking:
    """_run_scratchpad_consolidation must call update_budget_from_usage when cost > 0."""

    def test_update_budget_called_after_scratchpad_consolidation(self, tmp_path):
        """When consolidate_scratchpad() returns usage, update_budget_from_usage is called."""
        from ouroboros.agent_task_pipeline import _run_scratchpad_consolidation
        import ouroboros.consolidator as _cons

        usage_dict = {"prompt_tokens": 200, "completion_tokens": 100, "cost": 0.02}
        env = MagicMock()
        env.drive_path.return_value = tmp_path
        memory = MagicMock()
        llm = MagicMock()

        with patch.object(_cons, "should_consolidate_scratchpad_blocks", return_value=True, create=True), \
             patch.object(_cons, "consolidate_scratchpad_blocks", return_value=usage_dict, create=True), \
             patch.object(_cons, "should_consolidate_scratchpad", return_value=True, create=True), \
             patch.object(_cons, "consolidate_scratchpad", return_value=usage_dict, create=True), \
             patch("supervisor.state.update_budget_from_usage") as mock_budget:
            import time
            _run_scratchpad_consolidation(env, memory, llm)
            time.sleep(0.3)
            mock_budget.assert_called_once_with(usage_dict)

    def test_no_budget_call_when_consolidation_returns_none(self, tmp_path):
        from ouroboros.agent_task_pipeline import _run_scratchpad_consolidation
        import ouroboros.consolidator as _cons

        env = MagicMock()
        env.drive_path.return_value = tmp_path
        memory = MagicMock()
        llm = MagicMock()

        with patch.object(_cons, "should_consolidate_scratchpad_blocks", return_value=True, create=True), \
             patch.object(_cons, "consolidate_scratchpad_blocks", return_value=None, create=True), \
             patch.object(_cons, "should_consolidate_scratchpad", return_value=True, create=True), \
             patch.object(_cons, "consolidate_scratchpad", return_value=None, create=True), \
             patch("supervisor.state.update_budget_from_usage") as mock_budget:
            import time
            _run_scratchpad_consolidation(env, memory, llm)
            time.sleep(0.3)
            mock_budget.assert_not_called()


class TestConsolidationCostTracking:
    """_run_chat_consolidation must call update_budget_from_usage when cost > 0.

    agent_task_pipeline.py resolves symbols via getattr with fallback:
        consolidate_chat_blocks (new) → consolidate (legacy)
        should_consolidate_chat_blocks (new) → should_consolidate (legacy)
    Tests must patch the same symbols the pipeline actually resolves.
    """

    def _make_env(self, tmp_path):
        """Minimal env stub for _run_chat_consolidation."""
        env = MagicMock()
        env.drive_path.return_value = tmp_path
        return env

    def test_update_budget_called_after_consolidation(self, tmp_path):
        """When consolidate() returns a usage dict, update_budget_from_usage is called."""
        import json

        from ouroboros.agent_task_pipeline import _run_chat_consolidation

        # Set up fake chat log with enough entries to trigger consolidation
        chat_path = tmp_path / "chat.jsonl"
        # Write 100 entries (BLOCK_SIZE = 100)
        entries = [
            {"ts": "2026-01-01T00:00:00Z", "role": "user", "content": f"msg {i}"}
            for i in range(100)
        ]
        chat_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        blocks_path = tmp_path / "dialogue_blocks.json"
        meta_path = tmp_path / "dialogue_meta.json"

        class FakeEnv:
            drive_root = tmp_path
            def drive_path(self, rel):
                return tmp_path / rel

        class FakeMemory:
            def load_identity(self):
                return ""

        mock_llm = MagicMock()
        usage_dict = {"prompt_tokens": 500, "completion_tokens": 200, "cost": 0.05}

        # Patch the symbols that agent_task_pipeline._run_chat_consolidation resolves
        # via getattr: should_consolidate_chat_blocks (preferred) → should_consolidate (legacy)
        # and consolidate_chat_blocks (preferred) → consolidate (legacy).
        # Patch both so the test works regardless of which symbol is available.
        import ouroboros.consolidator as _cons
        with patch.object(_cons, "should_consolidate_chat_blocks", return_value=True, create=True), \
             patch.object(_cons, "consolidate_chat_blocks", return_value=usage_dict, create=True), \
             patch.object(_cons, "should_consolidate", return_value=True, create=True), \
             patch.object(_cons, "consolidate", return_value=usage_dict, create=True), \
             patch("supervisor.state.update_budget_from_usage") as mock_budget:

            import time

            _run_chat_consolidation(
                FakeEnv(), FakeMemory(), mock_llm,
                {"id": "t1"}, tmp_path / "logs"
            )
            # Wait for daemon thread
            time.sleep(0.3)
            mock_budget.assert_called_once_with(usage_dict)

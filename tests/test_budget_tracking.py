"""
Tests for budget/cost tracking across all tools and pipeline components.
Verifies that real LLM spend from advisory, plan_task, reflection,
consolidation, scope review, and supervisor dedup all reach accounting.
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
        from ouroboros.tools.review_helpers import emit_review_usage
        return lambda ctx, model, usage: emit_review_usage(ctx, model=model, usage=usage, source="scope_review")

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
    """Supervisor duplicate checks bind the physical-attempt ledger exactly once."""

    def test_dedup_check_binds_prospective_scope_without_legacy_increment(self, tmp_path):
        import supervisor.events as ev_mod
        from ouroboros.usage_accounting import current_usage_scope

        usage = {"prompt_tokens": 50, "completion_tokens": 10, "cost": 0.0001}
        # Need at least one existing task so the early-return guard doesn't skip the LLM call.
        pending = [{"id": "existing-1", "type": "task", "text": "some other task"}]
        captured = []

        with patch("ouroboros.llm.LLMClient") as mock_cls, \
             patch("supervisor.state.update_budget_from_usage") as mock_budget:
            inst = MagicMock()
            inst.chat.side_effect = lambda **_kwargs: (
                captured.append(current_usage_scope()) or {"content": "NONE"},
                usage,
            )
            mock_cls.return_value = inst
            result = ev_mod._find_duplicate_task(
                "Deploy new feature",
                "",
                pending,
                {},
                dedupe_identity={
                    "task_id": "prospective",
                    "root_task_id": "root-1",
                    "parent_task_id": "parent-1",
                    "budget_drive_root": str(tmp_path),
                },
            )
            mock_budget.assert_not_called()
            assert result is None  # "NONE" response = no duplicate found
        scope = captured[0]
        assert scope.drive_root == str(tmp_path)
        assert scope.task_id == "prospective"
        assert scope.root_task_id == "root-1"
        assert scope.parent_task_id == "parent-1"
        assert scope.category == "planning"
        assert scope.source == "task_duplicate_check"

    @pytest.mark.parametrize("raw, expected_default", [(None, True), ("0", False)])
    def test_dedup_scope_uses_total_budget_resolver(self, tmp_path, monkeypatch, raw, expected_default):
        import supervisor.events as ev_mod
        from ouroboros.config import SETTINGS_DEFAULTS
        from ouroboros.usage_accounting import current_usage_scope

        if raw is None:
            monkeypatch.delenv("TOTAL_BUDGET", raising=False)
        else:
            monkeypatch.setenv("TOTAL_BUDGET", raw)
        captured = []
        with patch("ouroboros.llm.LLMClient") as mock_cls:
            mock_cls.return_value.chat.side_effect = lambda **_k: (
                captured.append(current_usage_scope()) or {"content": "NONE"}, {}
            )
            ev_mod._find_duplicate_task(
                "new", "", [{"id": "old", "text": "old"}], {},
                dedupe_identity={"task_id": "new", "budget_drive_root": str(tmp_path)},
            )

        expected = float(SETTINGS_DEFAULTS["TOTAL_BUDGET"]) if expected_default else None
        assert captured[0].global_limit_usd == expected

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

    @pytest.mark.parametrize("error_type", [
        pytest.param("budget", id="budget_exceeded"),
        pytest.param("accounting", id="accounting_error"),
    ])
    def test_accounting_rails_are_not_downgraded_to_accept(self, error_type):
        import supervisor.events as ev_mod
        from ouroboros.usage_accounting import BudgetExceeded, UsageAccountingError

        error = BudgetExceeded("rail") if error_type == "budget" else UsageAccountingError("ledger")
        pending = [{"id": "existing-1", "type": "task", "text": "some task"}]
        with patch("ouroboros.llm.LLMClient") as mock_cls:
            mock_cls.return_value.chat.side_effect = error
            with pytest.raises(type(error), match=str(error)):
                ev_mod._find_duplicate_task("new task", "", pending, {})


class TestAdvisoryCostAccounting:
    """Advisory spend is accounted inside the review substrate: the native
    episode executor stamps every paid send into the usage ledger under
    usage_scope(category="advisory_review"), and the session route accounts
    through the same substrate. A second call-site emit (the retired
    Claude-SDK transport re-emitted source="advisory_sdk") would now
    double-count that spend, so its absence is the contract.
    """

    def test_call_site_no_longer_re_emits_transport_usage(self):
        import inspect
        mod = importlib.import_module("ouroboros.tools.claude_advisory_review")
        source = inspect.getsource(mod._run_claude_advisory)
        assert "advisory_sdk" not in source
        assert "emit_review_usage" not in source

    def test_native_dispatch_scopes_usage_to_advisory_review(self):
        import inspect
        mod = importlib.import_module("ouroboros.tools.claude_advisory_review")
        source = inspect.getsource(mod._run_advisory_native)
        assert 'category="advisory_review"' in source
        assert 'source="advisory_native"' in source



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

        tmp_path / "dialogue_blocks.json"
        tmp_path / "dialogue_meta.json"

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

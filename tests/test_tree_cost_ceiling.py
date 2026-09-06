"""One deciding money number per task tree, and the wrap-up affordability rail.

Covers the graceful in-task cost stop that borrows the ledger fence's own
per-attempt reservation, the cache-aware shape of that reservation, the root
ceiling no tree member may exceed, and the global-budget default.
"""
from __future__ import annotations

import base64
import queue
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ouroboros import task_pacing, usage_accounting
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.loop import _check_budget_limits, _RoundLimitContext


@pytest.fixture(autouse=True)
def _priced_anthropic_route(monkeypatch):
    """An isolated catalog, so the money numbers below are exact and offline."""
    from ouroboros import pricing

    monkeypatch.setattr(pricing, "_cached_pricing", {})
    monkeypatch.setattr(pricing, "_pricing_fetched_at", {})
    monkeypatch.setattr(pricing, "_pricing_retry_after", {})
    monkeypatch.setattr(pricing, "_pricing_fetch_in_progress", set())
    monkeypatch.setattr(
        "ouroboros.llm.fetch_openrouter_pricing",
        # (input, cached_read, cache_write(5m), output) per 1M tokens.
        lambda **_kwargs: {"anthropic/claude-test": (3.0, 0.3, 3.75, 15.0)},
    )
    usage_accounting._reset_task_cache_splits()
    yield
    usage_accounting._reset_task_cache_splits()


def _scoped(task_id, root_id, root_limit=None):
    from ouroboros.usage_accounting import UsageScope, usage_scope

    return usage_scope(UsageScope(
        drive_root=None, task_id=task_id, root_task_id=root_id, root_limit_usd=root_limit,
    ))


def _ctx(**overrides):
    llm = MagicMock()
    values = dict(
        messages=[],
        llm=llm,
        active_model="anthropic/claude-test",
        active_effort="high",
        max_retries=1,
        drive_logs=None,
        task_id="task1",
        round_idx=3,
        event_queue=queue.Queue(),
        accumulated_usage={"cost": 1.0, "_context_prompt_estimate": 400_000},
        task_type="task",
        active_use_local=False,
        max_rounds=100,
        llm_trace={},
    )
    values.update(overrides)
    return _RoundLimitContext(**values)


def _request(**overrides):
    values = dict(
        model="anthropic/claude-test",
        provider="openrouter",
        prompt_tokens_estimate=100_000,
        max_completion_tokens=1_000,
    )
    values.update(overrides)
    return usage_accounting.AttemptRequest(**values)


class TestCacheAwareReservation:
    """The fence prices what the task's own last send actually read from cache."""

    def test_full_write_without_an_observed_split(self):
        cold = usage_accounting._reservation_cost(_request(task_id="t1"))
        assert cold is not None and cold > 0

    def test_observed_split_lowers_the_reservation(self):
        cold = usage_accounting._reservation_cost(_request(task_id="t1"))
        usage_accounting.stash_task_cache_split(
            "t1", "anthropic/claude-test", 95_000, provider="openrouter", ttl_seconds=300.0,
        )
        warm = usage_accounting._reservation_cost(_request(task_id="t1"))
        assert warm is not None and warm < cold

    def test_applied_compaction_makes_the_next_reservation_cold(self, monkeypatch, tmp_path):
        from ouroboros import loop
        from ouroboros.context_budget import ContextReclaimReceipt

        request = _request(task_id="compacted")
        cold = usage_accounting._reservation_cost(request)
        usage_accounting.stash_task_cache_split(
            "compacted", request.model, 90_000,
            provider=request.provider, ttl_seconds=300.0,
        )
        assert usage_accounting._reservation_cost(request) < cold
        receipt = ContextReclaimReceipt(
            status="applied", before_transcript_sha256="a" * 64,
            after_transcript_sha256="b" * 64, selection_fingerprint="c" * 64,
            selected_unit_ids=("unit",), reclaimed_tokens=10, goal_reached=True,
            checkpoint_ref={"path": "checkpoint"}, capsule_refs=(),
        )
        monkeypatch.setattr(
            loop, "compact_tool_history_llm",
            lambda *_a, **_k: ([{"role": "assistant", "content": "summary"}], receipt, {}),
        )
        inner = SimpleNamespace(_pending_compaction=4)
        loop._run_round_compaction(
            [{"role": "user", "content": "before"}],
            loop._CompactionRoundContext(
                tools=SimpleNamespace(_ctx=inner), drive_root=tmp_path,
                drive_logs=tmp_path, task_id="compacted", round_idx=2,
                event_queue=None, emit_progress=lambda _text, *, incident=None: None,
            ),
        )

        assert usage_accounting._reservation_cost(request) == cold

    def test_enabling_a_tool_makes_the_next_reservation_cold(self, tmp_path):
        from ouroboros import loop
        from ouroboros.tool_policy import initial_tool_schemas
        from ouroboros.tools.registry import ToolRegistry

        registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        registry._ctx.task_id = "enabled"
        schemas = initial_tool_schemas(registry)
        schemas[:] = [s for s in schemas if s["function"]["name"] != "read_file"]
        loop._setup_dynamic_tools(registry, schemas, [])
        request = _request(task_id="enabled")
        cold = usage_accounting._reservation_cost(request)
        usage_accounting.stash_task_cache_split(
            "enabled", request.model, 90_000,
            provider=request.provider, ttl_seconds=300.0,
        )
        assert usage_accounting._reservation_cost(request) < cold

        assert "registered late" in registry.execute("enable_tools", {"tools": "read_file"})
        assert usage_accounting._reservation_cost(request) == cold

    def test_direct_route_and_ledger_identity_share_a_split(self):
        usage_accounting.stash_task_cache_split(
            "t1", "anthropic/claude-test", 95_000, provider="anthropic", ttl_seconds=300.0,
        )
        assert usage_accounting.last_task_cache_split(
            "t1", "anthropic::claude-test", provider="anthropic",
        ) == 95_000

    def test_direct_split_is_cold_after_openrouter_fallback(self):
        direct = _request(
            task_id="t1", provider="anthropic", model="anthropic::claude-test",
        )
        fallback = _request(
            task_id="t1", provider="openrouter", model="anthropic/claude-test",
        )
        cold = usage_accounting._reservation_cost(fallback)
        usage_accounting.stash_task_cache_split(
            "t1", direct.model, 95_000, provider=direct.provider, ttl_seconds=300.0,
        )

        assert usage_accounting.last_task_cache_split(
            "t1", direct.model, provider=direct.provider,
        ) == 95_000
        assert usage_accounting.last_task_cache_split(
            "t1", fallback.model, provider=fallback.provider,
        ) is None
        assert usage_accounting._reservation_cost(fallback) == cold

    def test_a_split_of_another_model_is_never_inherited(self):
        cold = usage_accounting._reservation_cost(_request(task_id="t1"))
        usage_accounting.stash_task_cache_split(
            "t1", "anthropic/other-model", 95_000, provider="openrouter", ttl_seconds=300.0,
        )
        assert usage_accounting._reservation_cost(_request(task_id="t1")) == cold

    def test_a_lapsed_split_is_never_inherited(self):
        cold = usage_accounting._reservation_cost(_request(task_id="t1"))
        usage_accounting.stash_task_cache_split(
            "t1", "anthropic/claude-test", 95_000, provider="openrouter", ttl_seconds=-1.0,
        )
        assert usage_accounting._reservation_cost(_request(task_id="t1")) == cold

    def test_a_request_without_a_task_id_still_resolves_its_split(self, tmp_path):
        """Regression: the main-loop request carries no task id of its own.

        The bound scope's id has to reach the request before the reservation is
        priced, otherwise the split is never found and the cache-aware
        reservation silently degrades to a full write on every live round.
        """
        usage_accounting.stash_task_cache_split(
            "live-task", "anthropic/claude-test", 95_000, provider="openrouter", ttl_seconds=300.0,
        )
        cold = usage_accounting._reservation_cost(_request(task_id="unknown-task"))
        with _scoped("live-task", "live-task"):
            merged, _scope = usage_accounting._merge_scope(_request())
            assert merged.task_id == "live-task"
            assert usage_accounting._reservation_cost(merged) < cold

    def test_the_reservation_still_takes_one_positional_argument(self):
        assert usage_accounting._reservation_cost(_request(task_id="t1", reservation_usd=2.0)) == 2.0



def _patch_execute_candidate(monkeypatch, llm_module, execute):
    """Patch the physical candidate executor where the v7 lanes BIND it.

    Upstream's ``LLMClient`` read ``_execute_candidate`` as a module global of
    ``ouroboros.llm``; the v7 split imports it into each lane mixin
    (``llm_anthropic``, ``llm_fallback``, ``llm_gigachat``) from ``llm_attempt``,
    so the historical single patch target no longer reaches the send. Patch the
    facade AND every lane that binds the name."""
    import importlib

    monkeypatch.setattr(llm_module, "_execute_candidate", execute)
    for name in ("ouroboros.llm_anthropic", "ouroboros.llm_fallback", "ouroboros.llm_gigachat", "ouroboros.llm_local"):
        module = importlib.import_module(name)
        if hasattr(module, "_execute_candidate"):
            monkeypatch.setattr(module, "_execute_candidate", execute)

class TestWrapupAffordability:
    """The graceful stop uses the fence's own reservation, and fails open."""

    def test_no_bound_scope_fails_open(self):
        assert task_pacing.wrapup_reservation_fits(
            model="anthropic/claude-test", prompt_tokens=400_000,
            root_cap_usd=50.0, deciding_usd=49.0,
        ) is None

    def test_no_root_cap_fails_open(self):
        with _scoped("t1", "t1"):
            assert task_pacing.wrapup_reservation_fits(
                model="anthropic/claude-test", prompt_tokens=400_000,
                root_cap_usd=None, deciding_usd=1.0,
            ) is None

    def test_unknown_price_fails_open(self):
        with _scoped("t1", "t1"):
            assert task_pacing.wrapup_reservation_fits(
                model="~no-such-model/never-priced", prompt_tokens=400_000,
                root_cap_usd=50.0, deciding_usd=49.0,
            ) is None

    def test_a_wrap_up_that_no_longer_fits_reports_false(self):
        with _scoped("t1", "t1"):
            assert task_pacing.wrapup_reservation_fits(
                model="anthropic/claude-test", prompt_tokens=400_000,
                root_cap_usd=50.0, deciding_usd=49.99,
            ) is False

    def test_a_wrap_up_that_still_fits_reports_true(self):
        with _scoped("t1", "t1"):
            assert task_pacing.wrapup_reservation_fits(
                model="anthropic/claude-test", prompt_tokens=1_000,
                root_cap_usd=50.0, deciding_usd=0.0,
            ) is True

    def test_the_predicate_equals_the_fence_reservation(self):
        with _scoped("t1", "t1"):
            from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS

            bound = usage_accounting._reservation_cost(_request(
                task_id="t1", prompt_tokens_estimate=400_000,
                max_completion_tokens=MAIN_LOOP_MAX_TOKENS,
            ))
            assert task_pacing.wrapup_reservation_fits(
                model="anthropic/claude-test", prompt_tokens=400_000,
                root_cap_usd=bound + 1.0, deciding_usd=1.0 + 1e-6,
            ) is False

    def test_the_predicate_never_reads_the_usage_projection(self, monkeypatch):
        monkeypatch.setattr(
            usage_accounting, "usage_projection",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("per-round ledger scan")),
        )
        with _scoped("t1", "t1"):
            assert task_pacing.wrapup_reservation_fits(
                model="anthropic/claude-test", prompt_tokens=400_000,
                root_cap_usd=50.0, deciding_usd=49.99,
            ) is False

    def test_multimodal_wrapup_matches_raw_candidate_admission(self, tmp_path):
        from ouroboros.context_fit import estimate_context_prompt_tokens
        from ouroboros.llm import LLMClient

        messages = [{"role": "user", "content": [{
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"x" * 300_000).decode()},
        }]}]
        scope = usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="multimodal", root_task_id="multimodal",
            global_limit_usd=100.0,
        )
        with usage_accounting.usage_scope(scope):
            request = task_pacing.prospective_wrapup_attempt_request(
                llm=LLMClient(api_key="unused"), messages=messages,
                model="anthropic/claude-test", reasoning_effort="high",
            )
            assert request.prompt_tokens_estimate > estimate_context_prompt_tokens(messages) * 10
            bound = usage_accounting._reservation_cost(request)
            cap = float(bound) - 1e-6
            assert task_pacing.wrapup_reservation_fits(
                request=request, root_cap_usd=cap, deciding_usd=0.0,
            ) is False
            with pytest.raises(usage_accounting.BudgetExceeded):
                usage_accounting.reserve_attempt(
                    replace(request, root_limit_usd=cap)
                )

    def test_explicit_openrouter_route_matches_cache_aware_admission(self, tmp_path):
        from ouroboros.llm import LLMClient

        scope = usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="routed", root_task_id="routed",
            global_limit_usd=100.0,
        )
        with usage_accounting.usage_scope(scope):
            usage_accounting.stash_task_cache_split(
                "routed", "anthropic/claude-test", 90_000,
                provider="openrouter", ttl_seconds=300.0,
            )
            request = task_pacing.prospective_wrapup_attempt_request(
                llm=LLMClient(api_key="unused"),
                messages=[{"role": "user", "content": "x" * 400_000}],
                model="openrouter::anthropic/claude-test", reasoning_effort="high",
            )
            assert (request.provider, request.model) == ("openrouter", "anthropic/claude-test")
            bound = usage_accounting._reservation_cost(request)
            cap = float(bound) + 1e-6
            assert task_pacing.wrapup_reservation_fits(
                request=request, root_cap_usd=cap, deciding_usd=0.0,
            ) is True
            usage_accounting.reserve_attempt(
                replace(request, root_limit_usd=cap)
            )

    def test_direct_anthropic_route_matches_physical_candidate(self, monkeypatch, tmp_path):
        from ouroboros import llm as llm_module
        from ouroboros.llm import LLMClient

        monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
        client = LLMClient(api_key="unused")
        model = "anthropic::claude-test"
        target = client._resolve_remote_target(model)
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "x" * 1_600},
        ]
        tools = [{
            "type": "function", "function": {
                "name": "inspect", "description": "inspect",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        scope = usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="direct", root_task_id="direct",
            global_limit_usd=100.0,
        )
        captured = {}

        def execute(request, _send, _before_dispatch):
            captured["request"] = request
            return SimpleNamespace(json=lambda: {
                "content": [{"type": "text", "text": "ok"}], "usage": {},
            })

        _patch_execute_candidate(monkeypatch, llm_module, execute)
        with usage_accounting.usage_scope(scope):
            prospective = task_pacing.prospective_wrapup_attempt_request(
                llm=client, messages=messages, model=model,
                reasoning_effort="high", tools=tools,
            )
            client._chat_anthropic(
                target, messages, tools, "high", prospective.max_completion_tokens, "auto",
            )

        actual = captured["request"]
        assert prospective.prompt_tokens_estimate == actual.prompt_tokens_estimate
        assert prospective.candidate_raw_size_bytes == actual.candidate_raw_size_bytes
        assert prospective.candidate_raw_sha256 == actual.candidate_raw_sha256

    def test_direct_openai_route_matches_physical_candidate(self, monkeypatch, tmp_path):
        from ouroboros import llm as llm_module
        from ouroboros.llm import LLMClient

        monkeypatch.setenv("OPENAI_API_KEY", "unused")
        client = LLMClient(api_key="unused")
        model = "openai::gpt-test"
        target = client._resolve_remote_target(model)
        messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "x" * 1600}]
        tools = [{"type": "function", "function": {
            "name": "inspect", "description": "inspect",
            "parameters": {"type": "object", "properties": {}},
        }}]
        captured = {}

        def execute(request, _send, _before_dispatch):
            captured["request"] = request
            return SimpleNamespace(model_dump=lambda: {"choices": [{"message": {"content": "ok"}}], "usage": {}})

        _patch_execute_candidate(monkeypatch, llm_module, execute)
        with usage_accounting.usage_scope(usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="openai", root_task_id="openai", global_limit_usd=100.0,
        )):
            prospective = task_pacing.prospective_wrapup_attempt_request(
                llm=client, messages=messages, model=model, reasoning_effort="high", tools=tools,
            )
            candidate = client._build_remote_candidate(
                target, messages, "high", prospective.max_completion_tokens, "auto", None, tools,
                skip_capability_fetch=True,
            )
            client._normalize_payload_cache_ttl(target, candidate)
            client._create_chat_completion_with_retries(lambda **_kwargs: None, candidate, target)

        actual = captured["request"]
        assert prospective.candidate_raw_size_bytes == actual.candidate_raw_size_bytes
        assert prospective.candidate_raw_sha256 == actual.candidate_raw_sha256

    def test_wire_recovery_matches_physical_candidate(self, monkeypatch, tmp_path):
        from ouroboros import llm as llm_module
        from ouroboros.llm import LLMClient

        monkeypatch.setenv("OPENAI_API_KEY", "unused")
        client = LLMClient(api_key="unused")
        model = "openai::gpt-test"
        target = client._resolve_remote_target(model)
        messages = [{"role": "user", "content": "recover me"}]
        captured = {}
        real_prepare = llm_module.prepare_wire_payload_for_send

        def prepare(target_, payload, *, api_surface):
            prepared = real_prepare(target_, payload, api_surface=api_surface)
            prepared["recovered_wire_field"] = True
            return prepared

        def execute(request, _send, _before_dispatch):
            captured["request"] = request
            return SimpleNamespace(model_dump=lambda: {"choices": [{"message": {"content": "ok"}}], "usage": {}})

        monkeypatch.setattr(llm_module, "prepare_wire_payload_for_send", prepare)
        # v7 split: the physical send binds the wire-recovery preparer from
        # llm_attempt (its own import from request_wire_recovery), so the facade
        # patch alone would leave the real preparer running.
        import ouroboros.llm_attempt as llm_attempt

        monkeypatch.setattr(llm_attempt, "prepare_wire_payload_for_send", prepare)
        _patch_execute_candidate(monkeypatch, llm_module, execute)
        with usage_accounting.usage_scope(usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="recovery", root_task_id="recovery", global_limit_usd=100.0,
        )):
            prospective = task_pacing.prospective_wrapup_attempt_request(
                llm=client, messages=messages, model=model, reasoning_effort="high",
            )
            candidate = client._build_remote_candidate(
                target, messages, "high", prospective.max_completion_tokens, "auto", None, None,
                skip_capability_fetch=True,
            )
            client._normalize_payload_cache_ttl(target, candidate)
            client._create_chat_completion_with_retries(lambda **_kwargs: None, candidate, target)

        actual = captured["request"]
        assert prospective.candidate_raw_size_bytes == actual.candidate_raw_size_bytes
        assert prospective.candidate_raw_sha256 == actual.candidate_raw_sha256


class TestCacheSplitOwnership:
    """Splits belong to one prompt surface; a rebuilt attempt starts cold."""

    def _scope(self, tmp_path, **attribution):
        return usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="shared", root_task_id="shared",
            global_limit_usd=100.0, **attribution,
        )

    def test_reviewer_sends_never_pose_as_the_transcripts_split(self, tmp_path):
        from ouroboros import _usage_cache_splits as splits

        splits.reset_task_cache_splits()
        with usage_accounting.usage_scope(self._scope(tmp_path)):
            splits.stash_task_cache_split("shared", "anthropic/claude-test", 90_000, ttl_seconds=300.0)
        review = self._scope(tmp_path, review_wave_id="w1", review_slot_id="slot_1")
        with usage_accounting.usage_scope(review):
            assert splits.last_task_cache_split("shared", "anthropic/claude-test") is None
            splits.stash_task_cache_split("shared", "anthropic/claude-test", 5, ttl_seconds=300.0)
        with usage_accounting.usage_scope(self._scope(tmp_path)):
            assert splits.last_task_cache_split("shared", "anthropic/claude-test") == 90_000
        splits.invalidate_task_cache_splits("shared")
        with usage_accounting.usage_scope(review):
            assert splits.last_task_cache_split("shared", "anthropic/claude-test") is None

    def test_review_surfaces_with_the_same_slot_do_not_share_a_split(self, tmp_path):
        from ouroboros import _usage_cache_splits as splits

        splits.reset_task_cache_splits()
        triad = self._scope(tmp_path, category="multi_model_review", review_slot_id="slot_1")
        acceptance = self._scope(tmp_path, category="task_acceptance", review_slot_id="slot_1")
        with usage_accounting.usage_scope(triad):
            splits.stash_task_cache_split("shared", "anthropic/claude-test", 70_000, ttl_seconds=300.0)
        with usage_accounting.usage_scope(acceptance):
            assert splits.last_task_cache_split("shared", "anthropic/claude-test") is None
        with usage_accounting.usage_scope(triad):
            assert splits.last_task_cache_split("shared", "anthropic/claude-test") == 70_000

    def test_plan_review_cycles_own_their_split_through_the_wave_id(self, tmp_path, monkeypatch):
        import asyncio

        from ouroboros import review_substrate
        from ouroboros.tools import plan_review_runtime
        from tests.test_reviewer_slot_identity import _fake_ctx, _substrate_stub

        seen, ran = [], []
        real = _substrate_stub(ran)

        def capture(request, **kwargs):
            seen.append(dict(request.usage_attribution))
            return real(request, **kwargs)

        monkeypatch.setattr(review_substrate, "run_review_request", capture)
        monkeypatch.setattr(plan_review_runtime, "LLMClient", lambda *a, **k: object())
        monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
        monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m/one")
        ctx = _fake_ctx(tmp_path)
        slots = plan_review_runtime.plan_review_slots()
        for key in ("plan_review:fp-a:1", "plan_review:fp-b:2"):
            asyncio.run(plan_review_runtime.run_plan_review_slots(
                ctx, slots, system_prompt="s", user_content="u", retry_key=key,
            ))
        assert [row.get("review_wave_id") for row in seen] == ["plan_review:fp-a:1", "plan_review:fp-b:2"]

    def test_a_rebuilt_attempt_starts_from_a_cold_split(self, tmp_path, monkeypatch):
        import queue as queue_module

        import ouroboros.loop as loop_module
        from ouroboros import _usage_cache_splits as splits
        from ouroboros.tools.registry import ToolRegistry

        splits.reset_task_cache_splits()
        splits.stash_task_cache_split("retry1", "anthropic/claude-test", 40_000, ttl_seconds=300.0)
        seen = {}

        class FakeLLM:
            def default_model(self):
                return "anthropic/claude-test"

        def fake_call(*_args, **_kwargs):
            seen["split_at_first_send"] = splits.last_task_cache_split("retry1", "anthropic/claude-test")
            return {"role": "assistant", "content": "FINAL ANSWER: done"}, 0.0

        monkeypatch.setattr(loop_module, "call_llm_with_retry", fake_call)
        loop_module.run_llm_loop(
            messages=[{"role": "user", "content": "again"}],
            tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
            llm=FakeLLM(), drive_logs=tmp_path, emit_progress=lambda _t, *, incident=None: None,
            incoming_messages=queue_module.Queue(), task_id="retry1", drive_root=tmp_path,
        )
        assert seen["split_at_first_send"] is None


class TestGlobalOnlyTreeAccounting:
    def test_tree_spend_is_read_without_a_root_cap(self, monkeypatch, tmp_path):
        import ouroboros.loop as loop_module

        monkeypatch.setattr(
            usage_accounting, "refresh_root_accounting",
            lambda drive_root, root_task_id, max_age_sec: {"accounted_usd": 7.5, "root": root_task_id},
        )
        with usage_accounting.usage_scope(usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="child", root_task_id="root1", global_limit_usd=100.0,
        )):
            info = loop_module._loop_tree_accounting(refresh=True, max_age_sec=0.0)
        assert info == {"accounted_usd": 7.5, "root": "root1"}


    def test_an_uncapped_rooted_attempt_refreshes_the_real_tree_cache(self, tmp_path):
        scope = usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="child-u", root_task_id="root-u", global_limit_usd=100.0,
        )
        with usage_accounting.usage_scope(scope):
            usage_accounting.reserve_attempt(usage_accounting.AttemptRequest(
                model="test/model", provider="openrouter", reservation_usd=2.5, drive_root=tmp_path,
            ))
            entry = usage_accounting.last_root_accounting("root-u")
            assert entry is not None and entry["accounted_usd"] == 2.5
            assert entry["root_limit_usd"] is None
            usage_accounting.reserve_attempt(usage_accounting.AttemptRequest(
                model="test/model", provider="openrouter", reservation_usd=1.0, drive_root=tmp_path,
            ))
            assert usage_accounting.last_root_accounting("root-u")["accounted_usd"] == 3.5

    def test_missing_tree_telemetry_for_a_rooted_task_is_a_disclosed_lower_bound(self, tmp_path):
        with usage_accounting.usage_scope(usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="c", root_task_id="r", global_limit_usd=100.0,
        )):
            deciding, basis = task_pacing.resolve_deciding_spend(
                tree_cost_usd=None, task_cost_usd=4.0, root_cap_usd=None,
            )
        assert (deciding, basis) == (4.0, task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN)
        deciding, basis = task_pacing.resolve_deciding_spend(
            tree_cost_usd=None, task_cost_usd=4.0, root_cap_usd=None,
        )
        assert basis == task_pacing.SPEND_BASIS_OWN_NO_TREE_CAP


class TestExhaustedCeilingSoftLanding:
    def _exhausted(self):
        return task_pacing.resolve_cost_ceiling(100.0, normalize_budget_profile(None), root_cap_usd=0.5)

    def _arm(self, monkeypatch, fits):
        ctx = _ctx()
        request = object()
        monkeypatch.setattr("ouroboros.loop._prepare_forced_prompt", lambda _c, prompt, _t: prompt)
        monkeypatch.setattr(
            task_pacing, "prepared_wrapup_candidate",
            lambda ctx_, messages, **_k: (request, messages),
        )
        monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", lambda **_k: fits)
        monkeypatch.setattr("ouroboros.loop._loop_tree_accounting", lambda **_k: None)
        return ctx, request

    def test_an_unaffordable_wrapup_ends_as_unaffordable_not_a_fence_pause(self, monkeypatch):
        import ouroboros.loop as loop_module

        ctx, _request = self._arm(monkeypatch, False)
        seen = {}
        monkeypatch.setattr(
            "ouroboros.loop._forced_fallback_result",
            lambda ctx_, _t, text, reason, **kw: (seen.update(kw, text=text, reason=reason) or ("x", ctx_.accumulated_usage, {})),
        )
        assert loop_module._soft_land_exhausted_ceiling(ctx, self._exhausted()) is not None
        assert seen["source"] == "budget_wrapup_unaffordable"
        assert "no working room" in seen["text"]
        assert ctx.accumulated_usage["cost_stop_rail"] == "wrapup_reservation_last_fit"

    def test_an_affordable_wrapup_dispatches_the_admitted_candidate(self, monkeypatch):
        import ouroboros.loop as loop_module

        ctx, request = self._arm(monkeypatch, True)
        monkeypatch.setattr(
            "ouroboros.loop._forced_final_answer",
            lambda ctx_, **kwargs: ("wrapped", ctx_.accumulated_usage, {"kwargs": kwargs}),
        )
        result = loop_module._soft_land_exhausted_ceiling(ctx, self._exhausted())
        assert result[2]["kwargs"]["_admitted_request"] is request
        assert result[2]["kwargs"]["_prompt_prepared"] is True
        assert "[BUDGET LIMIT]" in result[2]["kwargs"]["prompt"]
        assert "cost_stop_rail" not in ctx.accumulated_usage


class TestForcedIncompletenessIsAuthoritative:
    def test_replace_control_beside_a_tool_call_still_degrades(self, tmp_path, monkeypatch):
        import json as json_module

        from tests.test_delivery_forced_finalization import _forced_test_context

        loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
        limit_ctx.tool_schemas = [{"type": "function", "function": {"name": "read_file"}}]
        body = json_module.dumps({"delivery_control": "replace", "full_answer": "replaced preamble"})
        monkeypatch.setattr(
            loop, "call_llm_with_retry",
            lambda *_a, **kw: (
                kw["response_meta_out"].update(tool_call_count=1, finish_reason="tool_calls") or
                {"role": "assistant", "content": body,
                 "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                0.0,
            ),
        )
        _text, _usage, trace = loop._handle_round_limit(limit_ctx)
        assert trace.get("forced_finalization", {}).get("source") == "forced_model_incomplete"


class TestOneUsageRowPerDelegatedRun:
    def _executor(self, tmp_path, rows):
        from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment
        from ouroboros.review_substrate import ReviewRequest, ReviewSlot

        request = ReviewRequest(
            surface="plan_review", goal="review", task_id="task-1",
            session_root=str(tmp_path), session_task="review exact evidence",
            policy={"output_contract": "return findings"},
        )
        slot = ReviewSlot(
            slot_id="slot_a", model="fable", route="agent_session",
            session_target="claude=fable", session_profile="profile-a",
        )
        executor = AgentSessionReviewExecutor(
            ReviewAssignment(request=request, slot=slot, custody_root=tmp_path)
        )
        executor.usage_observer = rows.append
        return executor

    def test_a_fresh_executor_reconciling_the_same_run_emits_no_second_row(self, monkeypatch, tmp_path):
        import ouroboros.review_execution as review_execution
        from tests.test_phase4_plan_review_continuity import _session_run

        __import__("ouroboros.delegate_custody_usage", fromlist=["x"])._EMITTED_SESSION_USAGE.clear()
        rows = []
        monkeypatch.setattr(
            review_execution, "run_delegated_review_session",
            lambda **_k: _session_run(run_id="run-same"),
        )
        first = self._executor(tmp_path, rows)
        first.execute()
        second = self._executor(tmp_path, rows)
        second.execute()
        assert len(rows) == 1
        assert rows[0].get("delegated_run_id") == "run-same"

    def test_a_pending_start_then_a_settled_recovery_is_one_row(self, monkeypatch, tmp_path):
        import ouroboros.review_execution as review_execution
        from tests.test_phase4_plan_review_continuity import _session_run

        __import__("ouroboros.delegate_custody_usage", fromlist=["x"])._EMITTED_SESSION_USAGE.clear()
        rows = []

        def pending(**_k):
            exc = RuntimeError("provider outcome unknown")
            exc.delegated_run_id = "run-pending"
            exc.delegated_run_started = True
            raise exc

        monkeypatch.setattr(review_execution, "run_delegated_review_session", pending)
        first = self._executor(tmp_path, rows)
        with pytest.raises(RuntimeError):
            first.execute()
        assert len(rows) == 1 and rows[0].get("cost") is None
        monkeypatch.setattr(
            review_execution, "run_delegated_review_session",
            lambda **_k: _session_run(run_id="run-pending"),
        )
        second = self._executor(tmp_path, rows)
        second.execute()
        assert len(rows) == 1


class TestForcedCandidatePredicate:
    """The admitted wrap-up candidate is checked at the physical send, Main metadata or not."""

    def test_predicate_binds_without_physical_context(self):
        from ouroboros.loop_llm_call import _send_main_candidate

        predicate = object()
        seen = {}

        class FakeLLM:
            def chat(self, **kwargs):
                seen["bound"] = usage_accounting.current_physical_attempt_predicate()
                seen["context"] = usage_accounting.current_physical_attempt_context()
                return {"role": "assistant", "content": "ok"}, {}

        _send_main_candidate(
            FakeLLM(), {}, model="anthropic/claude-test", use_local=False, deadline_ts=None,
            physical_context=None, candidate_predicate=predicate,
        )
        assert seen["bound"] is predicate
        assert seen["context"] is None

    def test_a_mismatched_candidate_is_rejected_before_the_real_send(self, monkeypatch, tmp_path):
        from ouroboros import llm as llm_module
        from ouroboros.llm import LLMClient

        monkeypatch.setenv("OPENAI_API_KEY", "unused")
        client = LLMClient(api_key="unused")
        model = "openai::gpt-test"
        target = client._resolve_remote_target(model)
        sent = []
        real_execute = llm_module._execute_candidate

        def execute(request, send, before_dispatch):
            before_dispatch(SimpleNamespace(attempt_id="a1", drive_root=tmp_path))
            sent.append(request)
            return real_execute(request, send, before_dispatch)

        _patch_execute_candidate(monkeypatch, llm_module, execute)
        with usage_accounting.usage_scope(usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="pred", root_task_id="pred", global_limit_usd=100.0,
        )), usage_accounting.bind_physical_attempt_context(
            None, candidate_predicate=lambda actual: False,
        ):
            candidate = client._build_remote_candidate(
                target, [{"role": "user", "content": "wrap up"}], "high", 64, "auto", None, None,
                skip_capability_fetch=True,
            )
            client._normalize_payload_cache_ttl(target, candidate)
            with pytest.raises(usage_accounting.PhysicalAttemptPreconditionFailed):
                client._create_chat_completion_with_retries(lambda **_kwargs: None, candidate, target)
        assert sent == []


class TestWrapupAffordabilityRail:
    """The loop soft-lands on the rail, and stays silent when it cannot know."""

    def _ceiling(self, root_cap):
        return task_pacing.resolve_cost_ceiling(
            None, normalize_budget_profile(None), root_cap_usd=root_cap,
        )

    def test_the_rail_soft_lands_with_a_typed_stamp(self, monkeypatch):
        ctx = _ctx()
        monkeypatch.setattr(
            "ouroboros.loop._loop_tree_accounting", lambda **_k: {"accounted_usd": 20.0},
        )
        # proxy last-fit → exact probe last-fit → prepared candidate last-fit
        answers, calls, builds = iter((True, False, True, False, True, False)), [], []
        request = object()
        monkeypatch.setattr(
            task_pacing, "prospective_wrapup_attempt_request",
            lambda **kwargs: (builds.append(kwargs), request)[1],
        )
        monkeypatch.setattr(
            task_pacing, "wrapup_reservation_fits",
            lambda **kwargs: (calls.append(kwargs), next(answers))[1],
        )
        monkeypatch.setattr(
            "ouroboros.loop._forced_final_answer",
            lambda ctx_, **kwargs: ("wrapped up", ctx_.accumulated_usage, {"kwargs": kwargs}),
        )
        monkeypatch.setattr(
            "ouroboros.loop._finalize_forced_services",
            lambda ctx_, _trace: ctx_.messages.append({"role": "user", "content": "service evidence"}),
        )
        monkeypatch.setattr(
            "ouroboros.loop._forced_delegation_note", lambda *_args: "\nforced delegation note",
        )

        result = _check_budget_limits(ctx, None, self._ceiling(50.0))

        assert result is not None
        assert ctx.accumulated_usage["cost_stop_rail"] == "wrapup_reservation_last_fit"
        assert result[2]["kwargs"]["reason_code"] == "budget_exhausted"
        assert result[2]["kwargs"]["_prompt_prepared"] is True
        assert "forced delegation note" in result[2]["kwargs"]["prompt"]
        assert "[BUDGET LIMIT]" in builds[-1]["messages"][-1]["content"]
        assert "forced delegation note" in builds[-1]["messages"][-1]["content"]
        assert "service evidence" in str(builds[-1]["messages"])
        # the probe priced a COPY without the forced services' evidence
        assert "service evidence" not in str(builds[0]["messages"])
        assert len(builds) == 2
        assert [call.get("request") for call in calls] == [None, None, request, request, request, request]

    def test_repriced_nondecision_does_not_stamp_a_cost_stop(self, monkeypatch):
        ctx = _ctx(messages=[{"role": "user", "content": "work"}])
        monkeypatch.setattr(
            "ouroboros.loop._loop_tree_accounting", lambda **_k: {"accounted_usd": 20.0},
        )
        # proxy last-fit → the exact probe disagrees (None): nothing destructive may run
        answers = iter((True, False, None, False))
        monkeypatch.setattr(
            task_pacing, "wrapup_reservation_fits", lambda **_kwargs: next(answers),
        )
        monkeypatch.setattr(
            task_pacing, "prospective_wrapup_attempt_request", lambda **_kwargs: object(),
        )

        def destructive(*_a, **_k):
            raise AssertionError("service finalization before a confirmed stop")

        monkeypatch.setattr("ouroboros.loop._prepare_forced_prompt", destructive)
        monkeypatch.setattr("ouroboros.loop._finalize_forced_services", destructive)
        monkeypatch.setattr(task_pacing, "prepared_wrapup_candidate", destructive)

        assert _check_budget_limits(ctx, None, self._ceiling(50.0)) is None
        assert ctx.messages == [{"role": "user", "content": "work"}]
        assert "cost_stop_spend_basis" not in ctx.accumulated_usage
        assert "cost_stop_rail" not in ctx.accumulated_usage

    def _image_messages(self):
        return [{"role": "user", "content": [
            {"type": "text", "text": "inspect"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aaa"}},
        ]}]

    def test_native_images_reprice_even_when_the_proxy_says_two_fit(self, monkeypatch):
        ctx = _ctx(messages=self._image_messages())
        monkeypatch.setattr(
            "ouroboros.loop._loop_tree_accounting", lambda **_k: {"accounted_usd": 20.0},
        )
        # proxy (1, 2) → non-destructive probe (1, 2) → prepared candidate (1, 2)
        answers, calls = iter((True, True, True, False, True, False)), []
        probe, prepared = object(), object()
        monkeypatch.setattr(task_pacing, "prospective_wrapup_attempt_request", lambda **_k: probe)
        monkeypatch.setattr(
            task_pacing, "prepared_wrapup_candidate", lambda ctx_, messages, **_k: (prepared, messages),
        )
        monkeypatch.setattr(
            task_pacing, "wrapup_reservation_fits",
            lambda **kwargs: (calls.append(kwargs), next(answers))[1],
        )
        finalized = []
        monkeypatch.setattr(
            "ouroboros.loop._prepare_forced_prompt", lambda _c, prompt, _t: (finalized.append(1), prompt)[1],
        )
        monkeypatch.setattr(
            "ouroboros.loop._forced_final_answer",
            lambda ctx_, **kwargs: ("wrapped up", ctx_.accumulated_usage, {"kwargs": kwargs}),
        )

        result = _check_budget_limits(ctx, None, self._ceiling(50.0))

        assert result is not None
        assert ctx.accumulated_usage["cost_stop_rail"] == "wrapup_reservation_last_fit"
        assert [call.get("request") for call in calls] == [None, None, probe, probe, prepared, prepared]
        assert finalized == [1]
        assert result[2]["kwargs"]["_admitted_request"] is prepared

    def test_an_image_probe_with_headroom_never_finalizes_services(self, monkeypatch):
        ctx = _ctx(messages=self._image_messages())
        monkeypatch.setattr(
            "ouroboros.loop._loop_tree_accounting", lambda **_k: {"accounted_usd": 20.0},
        )
        answers = iter((True, True, True, True))
        monkeypatch.setattr(task_pacing, "prospective_wrapup_attempt_request", lambda **_k: object())
        monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", lambda **_k: next(answers))

        def destructive(*_a, **_k):
            raise AssertionError("service finalization before a stop decision")

        monkeypatch.setattr("ouroboros.loop._prepare_forced_prompt", destructive)
        monkeypatch.setattr("ouroboros.loop._finalize_forced_services", destructive)
        monkeypatch.setattr(task_pacing, "prepared_wrapup_candidate", destructive)

        assert _check_budget_limits(ctx, None, self._ceiling(50.0)) is None
        assert ctx.messages == self._image_messages()
        assert "cost_stop_rail" not in ctx.accumulated_usage

    def test_a_proxy_stop_is_confirmed_by_the_priced_candidate(self, monkeypatch):
        ctx = _ctx()
        monkeypatch.setattr(
            "ouroboros.loop._loop_tree_accounting", lambda **_k: {"accounted_usd": 20.0},
        )
        # proxy no-fit → exact probe no-fit → prepared candidate no-fit
        answers, calls = iter((False, False, False)), []
        request = object()
        monkeypatch.setattr(
            task_pacing, "prospective_wrapup_attempt_request", lambda **_kwargs: request,
        )
        monkeypatch.setattr(
            task_pacing, "prepared_wrapup_candidate", lambda ctx_, messages, **_k: (request, messages),
        )
        monkeypatch.setattr("ouroboros.loop._prepare_forced_prompt", lambda _c, prompt, _t: prompt)
        monkeypatch.setattr(
            task_pacing, "wrapup_reservation_fits",
            lambda **kwargs: (calls.append(kwargs), next(answers))[1],
        )
        seen = {}
        monkeypatch.setattr(
            "ouroboros.loop._forced_fallback_result",
            lambda ctx_, _trace, text, reason, **kwargs: (
                seen.update(kwargs, text=text, reason=reason) or ("stopped", ctx_.accumulated_usage, {})
            ),
        )

        result = _check_budget_limits(ctx, None, self._ceiling(50.0))

        assert result is not None
        assert seen["source"] == "budget_wrapup_unaffordable"
        assert seen["reason"] == "budget_exhausted"
        assert "not even one wrap-up call" in seen["text"]
        assert ctx.accumulated_usage["cost_stop_rail"] == "wrapup_reservation_last_fit"
        assert [call.get("request") for call in calls] == [None, request, request]

    def test_an_affordable_wrapup_still_reaches_the_ceiling_stop(self, monkeypatch):
        ceiling = self._ceiling(50.0)
        ctx = _ctx()
        monkeypatch.setattr(
            "ouroboros.loop._loop_tree_accounting",
            lambda **_k: {"accounted_usd": ceiling.ceiling_usd + 1.0},
        )
        answers = iter((True, False, True, True))
        monkeypatch.setattr(
            task_pacing, "wrapup_reservation_fits", lambda **_kwargs: next(answers),
        )
        monkeypatch.setattr(
            task_pacing, "prospective_wrapup_attempt_request", lambda **_kwargs: object(),
        )
        monkeypatch.setattr(
            "ouroboros.loop._forced_final_answer",
            lambda ctx_, **kwargs: ("ceiling stop", ctx_.accumulated_usage, {"kwargs": kwargs}),
        )

        result = _check_budget_limits(ctx, None, ceiling)

        assert result is not None
        assert "over the in-task cost ceiling" in result[2]["kwargs"]["prompt"]
        assert "cost_stop_rail" not in ctx.accumulated_usage
        assert ctx.accumulated_usage["cost_stop_spend_basis"]

    def test_captioned_wrapup_candidate_is_the_initial_forced_dispatch(
        self, monkeypatch, tmp_path,
    ):
        import ouroboros.loop as loop_module
        from ouroboros.llm import LLMClient

        ctx = _ctx(drive_logs=tmp_path / "logs", llm=LLMClient(api_key="unused"))
        ctx.messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aaa"},
                    "_caption": "wire caption",
                },
            ],
        }]
        monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
        monkeypatch.setattr(
            "ouroboros.loop._loop_tree_accounting", lambda **_k: {"accounted_usd": 20.0},
        )
        answers = iter((True, False, True, False, True, False))
        monkeypatch.setattr(
            task_pacing, "wrapup_reservation_fits", lambda **_kwargs: next(answers),
        )
        built = {}
        real_build = task_pacing.prospective_wrapup_attempt_request

        def build(**kwargs):
            built["messages"] = kwargs["messages"]
            built["request"] = real_build(**kwargs)
            return built["request"]

        monkeypatch.setattr(task_pacing, "prospective_wrapup_attempt_request", build)
        dispatched = {}

        def call(*_args, **kwargs):
            dispatched["messages"] = kwargs["initial_messages"]
            from ouroboros.llm import _attempt_request, _finalized_physical_candidate
            from ouroboros.request_wire_recovery import request_wire_call_scope

            target = ctx.llm._resolve_remote_target(ctx.active_model)
            with request_wire_call_scope():
                candidate = ctx.llm._build_remote_candidate(
                    target, kwargs["initial_messages"], ctx.active_effort,
                    built["request"].max_completion_tokens, "auto", None, ctx.tool_schemas,
                    skip_capability_fetch=True,
                )
                ctx.llm._normalize_payload_cache_ttl(target, candidate)
                candidate = _finalized_physical_candidate(
                    target, candidate,
                    "messages" if target.get("provider") == "anthropic" else "chat.completions",
                )
            actual = _attempt_request(target, candidate)
            dispatched["accepted"] = kwargs["candidate_predicate"](actual)
            dispatched["sha256"] = actual.candidate_raw_sha256
            return {"content": "wrapped up"}, 0.0

        monkeypatch.setattr(loop_module, "call_llm_with_retry", call)

        result = _check_budget_limits(ctx, None, self._ceiling(50.0))

        assert result is not None
        assert dispatched["messages"] is built["messages"]
        assert dispatched["accepted"] is True
        assert dispatched["sha256"] == built["request"].candidate_raw_sha256
        assert built["messages"][0]["content"][1] == {
            "type": "text", "text": "[image caption: wire caption]",
        }
        assert ctx.messages[0]["content"][1]["type"] == "image_url"

    def test_a_missing_prompt_estimate_keeps_the_rail_silent(self, monkeypatch):
        ctx = _ctx(accumulated_usage={"cost": 1.0})
        monkeypatch.setattr(
            "ouroboros.loop._loop_tree_accounting", lambda **_k: {"accounted_usd": 20.0},
        )
        monkeypatch.setattr(
            task_pacing, "wrapup_reservation_fits",
            lambda **_k: (_ for _ in ()).throw(AssertionError("armed without a prompt size")),
        )

        assert _check_budget_limits(ctx, None, self._ceiling(50.0)) is None

    def test_a_disabled_ceiling_never_arms_the_rail(self, monkeypatch):
        ctx = _ctx()
        monkeypatch.setattr(
            task_pacing, "wrapup_reservation_fits",
            lambda **_k: (_ for _ in ()).throw(AssertionError("armed on a disabled ceiling")),
        )
        disabled = task_pacing.resolve_cost_ceiling(
            None, normalize_budget_profile({"cost_hard_stop_pct": 0}), root_cap_usd=50.0,
        )

        assert _check_budget_limits(ctx, None, disabled) is None

    def test_the_stop_text_names_the_cap_and_the_reason(self):
        text = task_pacing.wrapup_last_fit_text(49.9, self._ceiling(50.0))

        assert "$49.900" in text and "$50.00" in text
        assert "wrap-up call" in text


class TestOneCeilingPerTree:
    """Enabled members keep the proved original root's early stop number."""

    def _profile(self):
        return normalize_budget_profile(None)

    def test_a_non_root_member_never_exceeds_the_root_deciding_number(self):
        root = task_pacing.resolve_cost_ceiling(10.0, self._profile(), root_cap_usd=50.0)
        member = task_pacing.resolve_cost_ceiling(
            100.0, self._profile(), root_cap_usd=50.0, non_root_member=True,
            root_ceiling_usd=root.ceiling_usd,
        )

        assert "global_pct" in root.basis
        assert "root_resolved_ceiling" in member.basis
        assert "global_pct" not in member.basis
        assert "non_root_member" in member.basis
        assert member.ceiling_usd == root.ceiling_usd == 5.0

    def test_a_child_scope_carries_the_root_resolved_ceiling(self):
        from ouroboros.usage_accounting import UsageScope, usage_scope

        scope = UsageScope(
            task_id="child", root_task_id="root", root_limit_usd=50.0,
            root_cost_ceiling_usd=5.0,
        )
        with usage_scope(scope):
            member = task_pacing.resolve_task_cost_ceiling(SimpleNamespace(), 100.0)

        assert member.ceiling_usd == 5.0

    def test_the_scheduled_child_payload_carries_the_root_ceiling(self):
        from supervisor.task_dispatch import build_scheduled_task_payload

        task = build_scheduled_task_payload({
            "tid": "child", "root_task_id": "root", "parent_id": "root",
            "delegation_role": "subagent", "root_cost_ceiling_usd": 5.0,
        })

        assert task["root_cost_ceiling_usd"] == 5.0
        assert task["metadata"]["root_cost_ceiling_usd"] == 5.0

    def test_legacy_missing_carrier_keeps_a_disclosed_local_resolution(self):
        early = task_pacing.resolve_cost_ceiling(
            40.0, self._profile(), root_cap_usd=50.0, non_root_member=True,
        )
        late = task_pacing.resolve_cost_ceiling(
            4.0, self._profile(), root_cap_usd=50.0, non_root_member=True,
        )

        assert late.ceiling_usd < early.ceiling_usd
        assert "original_root_ceiling_unavailable" in late.basis

    def test_without_a_root_cap_the_global_component_still_binds(self):
        member = task_pacing.resolve_cost_ceiling(
            20.0, self._profile(), root_cap_usd=None, non_root_member=True,
        )

        assert "global_pct" in member.basis
        assert member.ceiling_usd == 10.0

    def test_the_default_keeps_the_historical_root_semantics(self):
        positional = task_pacing.resolve_cost_ceiling(20.0, self._profile(), root_cap_usd=50.0)

        assert positional.basis == "min(global_pct, root_cap_minus_margin)"

    def test_a_tree_member_resolves_the_cap_minus_margin_from_its_scope(self):
        with _scoped("child", "root", 50.0):
            ceiling = task_pacing.resolve_task_cost_ceiling(SimpleNamespace(), 40.0)

        assert "non_root_member" in ceiling.basis
        assert ceiling.ceiling_usd == 20.0

    def test_the_root_of_the_tree_keeps_both_components(self):
        with _scoped("root", "root", 50.0):
            ceiling = task_pacing.resolve_task_cost_ceiling(SimpleNamespace(), 40.0)

        assert "global_pct" in ceiling.basis

    def test_the_disclosed_ceiling_is_the_object_the_loop_decides_on(self):
        from ouroboros.loop import _resolve_task_cost_ceiling

        ctx = SimpleNamespace()
        with _scoped("root", "root", 50.0):
            disclosure = task_pacing.in_task_cost_ceiling_disclosure(ctx, 40.0)
            deciding = _resolve_task_cost_ceiling(ctx, 4.0)

        assert deciding is ctx._cost_ceiling
        assert disclosure["ceiling_usd"] == deciding.ceiling_usd
        assert disclosure["state"] == deciding.state

    def test_a_context_that_cannot_be_stashed_still_discloses(self):
        disclosure = task_pacing.in_task_cost_ceiling_disclosure(object(), 40.0)

        assert "state" in disclosure and "rule" in disclosure

    def test_the_checkpoint_and_the_pacing_note_share_one_formatter(self):
        active = task_pacing.resolve_cost_ceiling(
            None, normalize_budget_profile(None), root_cap_usd=50.0,
        )
        line = task_pacing.tree_spend_line(
            {"accounted_usd": 12.0, "root_limit_usd": 50.0}, active,
        )

        assert line.startswith("Task tree spend: ~$12.00")
        assert "in-task cost ceiling" in line and "$50.00 hard tree cap" in line
        assert task_pacing.tree_spend_line({"accounted_usd": None}, active) == ""

    def test_the_rails_line_names_the_binding_bound(self):
        ceiling_binds = task_pacing._headroom_phrase(40.0, 10.0, 2.0)
        wallet_binds = task_pacing._headroom_phrase(3.0, 40.0, 2.0)

        assert ceiling_binds == "$8.00 budget left (in-task cost ceiling binds)"
        assert wallet_binds == "$3.00 budget left (wallet binds)"
        assert task_pacing._headroom_phrase(None, None, None) == "budget left unknown"

    def test_acceptance_rails_use_global_wallet_and_tree_spend(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOTAL_BUDGET", "50")
        for task_id, root_id, cost in (("child", "root", 42.0), ("other", "other", 7.0)):
            with usage_accounting.usage_scope(usage_accounting.UsageScope(
                drive_root=tmp_path, task_id=task_id, root_task_id=root_id,
                global_limit_usd=50.0, root_limit_usd=100.0,
            )):
                reservation = usage_accounting.reserve_attempt(_request(reservation_usd=cost))
                usage_accounting.mark_dispatched(reservation)
                usage_accounting.settle_attempt(reservation, {}, cost_usd=cost, cost_final=True)
        with usage_accounting.usage_scope(usage_accounting.UsageScope(
            drive_root=tmp_path, task_id="child", root_task_id="root",
            global_limit_usd=50.0, root_limit_usd=100.0,
        )):
            line = task_pacing._acceptance_rails_line_inner(
                SimpleNamespace(has_deadline=False), self._profile(), 0,
                {"task_cost_usd": 2.0, "cost_ceiling_usd": 47.0},
                required_blocking=False,
            )

        assert "$2.00 spent this task" in line
        assert "$1.00 budget left (wallet binds)" in line


class TestGlobalBudgetDefault:
    """One number for the global budget, whatever the reader."""

    def test_an_absent_setting_resolves_the_product_default(self, monkeypatch):
        from ouroboros.config import SETTINGS_DEFAULTS
        from ouroboros.settings_setup_contract import resolve_total_budget_usd

        monkeypatch.delenv("TOTAL_BUDGET", raising=False)

        assert resolve_total_budget_usd() == float(SETTINGS_DEFAULTS["TOTAL_BUDGET"])

    def test_an_explicit_zero_stays_no_finite_global_budget(self, monkeypatch):
        from ouroboros.settings_setup_contract import resolve_total_budget_usd

        monkeypatch.setenv("TOTAL_BUDGET", "0")

        assert resolve_total_budget_usd() is None

    def test_junk_falls_back_to_the_product_default(self, monkeypatch):
        from ouroboros.config import SETTINGS_DEFAULTS
        from ouroboros.settings_setup_contract import resolve_total_budget_usd

        monkeypatch.setenv("TOTAL_BUDGET", "not-a-number")

        assert resolve_total_budget_usd() == float(SETTINGS_DEFAULTS["TOTAL_BUDGET"])

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
    def test_non_finite_values_fall_back_to_the_product_default(self, monkeypatch, raw):
        from ouroboros.config import SETTINGS_DEFAULTS
        from ouroboros.settings_setup_contract import resolve_total_budget_usd

        monkeypatch.setenv("TOTAL_BUDGET", raw)

        assert resolve_total_budget_usd() == float(SETTINGS_DEFAULTS["TOTAL_BUDGET"])

    def test_every_reader_agrees_on_the_absent_setting(self, monkeypatch):
        """Regression: an env-less harness install used to see $1 on the loop's
        money axis, no limit at all on the bound scope, and $200 at the ledger
        fence -- so one round of work could reject every later task."""
        from ouroboros.settings_setup_contract import resolve_total_budget_usd
        from ouroboros.usage_accounting import _global_limit

        monkeypatch.delenv("TOTAL_BUDGET", raising=False)
        expected = resolve_total_budget_usd()

        assert expected is not None and expected > 1.0
        assert _global_limit(_request()) == expected

    def test_an_unset_budget_no_longer_rejects_a_task_at_round_one(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TOTAL_BUDGET", raising=False)
        from ouroboros.settings_setup_contract import resolve_total_budget_usd

        assert resolve_total_budget_usd() is not None
        ctx = _ctx(round_idx=1, accumulated_usage={"cost": 1.5})
        disabled = task_pacing.resolve_cost_ceiling(None, normalize_budget_profile(None))

        assert _check_budget_limits(ctx, None, disabled) is None

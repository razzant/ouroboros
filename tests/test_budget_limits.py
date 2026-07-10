"""Tests for _check_budget_limits (per-task soft reminder + global budget guard)
and the v6.56.0 cost axis (contract-driven in-task ceiling + latched milestones)."""
import os
import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ouroboros import task_pacing
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.loop import _check_budget_limits


def _make_args(**overrides):
    """Build default kwargs for _check_budget_limits.

    ``cost_ceiling_usd`` defaults to the historical 50%-of-remaining ceiling so
    the legacy guard tests keep exercising the same semantics the runtime gets
    from ``task_pacing.resolve_cost_ceiling_usd`` with an absent profile.
    """
    defaults = dict(
        budget_remaining_usd=100.0,
        accumulated_usage={"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        round_idx=0,
        messages=[],
        llm=MagicMock(),
        active_model="test-model",
        active_effort="high",
        max_retries=1,
        drive_logs=None,
        task_id="test-task",
        event_queue=queue.Queue(),
        llm_trace={},
        task_type="task",
        use_local=False,
    )
    defaults.update(overrides)
    if "cost_ceiling_usd" not in defaults:
        defaults["cost_ceiling_usd"] = task_pacing.resolve_cost_ceiling_usd(
            defaults["budget_remaining_usd"], normalize_budget_profile(None),
        )
    return defaults


# --- Per-task soft reminder ---

class TestPerTaskSoftReminder:
    """Per-task cost soft reminder (OUROBOROS_PER_TASK_COST_USD).
    
    No hard stop — agent decides whether to continue.
    """

    def test_under_limit_returns_none(self, tmp_path):
        """Below threshold → no intervention."""
        args = _make_args(accumulated_usage={"cost": 3.0}, drive_logs=tmp_path)
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "5.0"}):
            result = _check_budget_limits(**args)
        assert result is None

    def test_at_limit_no_hard_stop(self, tmp_path):
        """At or above threshold → NO hard stop, just soft reminder."""
        messages = []
        args = _make_args(
            accumulated_usage={"cost": 5.5},
            round_idx=10,
            messages=messages,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "5.0"}):
            result = _check_budget_limits(**args)
        # No hard stop
        assert result is None

    def test_soft_reminder_injected_every_10_rounds(self, tmp_path):
        """At threshold + round divisible by 10 → soft note injected."""
        messages = []
        args = _make_args(
            accumulated_usage={"cost": 6.0},
            round_idx=10,
            messages=messages,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "5.0"}):
            result = _check_budget_limits(**args)
        assert result is None
        assert any("[COST NOTE]" in m.get("content", "") for m in messages)

    def test_no_reminder_on_non_10_round(self, tmp_path):
        """At threshold but round not divisible by 10 → no message."""
        messages = []
        args = _make_args(
            accumulated_usage={"cost": 6.0},
            round_idx=7,
            messages=messages,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "5.0"}):
            result = _check_budget_limits(**args)
        assert result is None
        assert not any("[COST NOTE]" in m.get("content", "") for m in messages)

    def test_custom_env_limit(self, tmp_path):
        """Respect custom per-task threshold from env."""
        args = _make_args(accumulated_usage={"cost": 9.0}, drive_logs=tmp_path)
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "10.0"}):
            result = _check_budget_limits(**args)
        # 9.0 < 10.0 → no message at all
        assert result is None

    def test_default_limit_20_no_hard_stop(self, tmp_path):
        """Without env var, default threshold is 20.0 — but still no hard stop."""
        messages = []
        args = _make_args(
            accumulated_usage={"cost": 20.0},
            round_idx=10,
            messages=messages,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OUROBOROS_PER_TASK_COST_USD", None)
            result = _check_budget_limits(**args)
        assert result is None  # soft reminder only, no hard stop


# --- Global budget guard ---

class TestGlobalBudgetGuard:
    """Existing global budget percentage checks."""

    def test_none_budget_returns_none(self, tmp_path):
        """No budget → no checks."""
        args = _make_args(budget_remaining_usd=None, accumulated_usage={"cost": 100.0}, drive_logs=tmp_path)
        result = _check_budget_limits(**args)
        assert result is None

    def test_budget_exhausted(self, tmp_path):
        """Remaining ≤ 0 → immediate stop."""
        args = _make_args(budget_remaining_usd=0.0, accumulated_usage={"cost": 0.01}, drive_logs=tmp_path)
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "999"}):
            result = _check_budget_limits(**args)
        assert result is not None
        text, _, _ = result
        assert "budget exhausted" in text.lower()

    def test_under_50pct_passes(self, tmp_path):
        """Task cost < 50% of remaining → no stop."""
        args = _make_args(
            budget_remaining_usd=10.0,
            accumulated_usage={"cost": 4.9},  # 49% < 50%
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "10.0"}):
            result = _check_budget_limits(**args)
        assert result is None

    def test_over_50pct_triggers(self, tmp_path):
        """Task cost > 50% of remaining budget → stops."""
        llm = MagicMock()
        llm.chat.return_value = ({"content": "done"}, {"prompt_tokens": 10, "completion_tokens": 5})
        args = _make_args(
            budget_remaining_usd=8.0,
            accumulated_usage={"cost": 4.5},  # 4.5/8 = 56% > 50%
            llm=llm,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "10.0"}):
            result = _check_budget_limits(**args)
        assert result is not None

    def test_legacy_info_nudge_removed(self, tmp_path):
        """The old round-gated '[INFO] ... wrap up' nudge is gone (v6.56.0):
        cost awareness now comes from the latched task_pacing milestones."""
        messages = []
        args = _make_args(
            budget_remaining_usd=10.0,
            accumulated_usage={"cost": 3.5},  # 35% — would have nudged before
            round_idx=20,
            messages=messages,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "10.0"}):
            result = _check_budget_limits(**args)
        assert result is None
        assert not any("[INFO]" in m.get("content", "") for m in messages)


# --- use_local propagation ---

class TestUseLocalPropagation:
    """Ensure use_local is passed to _call_llm_with_retry on global budget stop."""

    @patch("ouroboros.loop._call_llm_with_retry")
    def test_global_stop_passes_use_local(self, mock_retry, tmp_path):
        mock_retry.return_value = ({"content": "done"}, {"prompt_tokens": 10, "completion_tokens": 5})
        args = _make_args(
            budget_remaining_usd=6.0,
            accumulated_usage={"cost": 4.0},  # 67% > 50%
            use_local=True,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "10.0"}):
            _check_budget_limits(**args)
        mock_retry.assert_called_once()
        _, kwargs = mock_retry.call_args
        assert kwargs.get("use_local") is True


# --- v6.56.0 cost axis: contract-driven ceiling resolution ---

class TestCostCeilingResolution:
    """task_pacing.resolve_cost_ceiling_usd semantics (budget_profile.cost_hard_stop_pct)."""

    def test_absent_profile_means_historical_50pct(self):
        profile = normalize_budget_profile(None)
        assert task_pacing.resolve_cost_ceiling_usd(10.0, profile) == 5.0

    def test_zero_pct_means_no_ceiling_never_zero_dollars(self):
        profile = normalize_budget_profile({"cost_hard_stop_pct": 0})
        assert task_pacing.resolve_cost_ceiling_usd(10.0, profile) is None

    def test_custom_pct(self):
        profile = normalize_budget_profile({"cost_hard_stop_pct": 25})
        assert task_pacing.resolve_cost_ceiling_usd(10.0, profile) == 2.5

    def test_no_finite_budget_means_silent_axis(self):
        profile = normalize_budget_profile(None)
        assert task_pacing.resolve_cost_ceiling_usd(None, profile) is None
        assert task_pacing.resolve_cost_ceiling_usd(0.0, profile) is None

    def test_malformed_pct_fails_safe_to_default_not_zero(self):
        """A garbage cost_hard_stop_pct must NOT silently become 0 (= no in-task
        stop, the most permissive setting): negative / non-numeric / a 0<v<1
        fraction map to None (the 50% default), while an explicit 0 is honored."""
        for bad in (-5, -0.1, 0.5, "0.5", "abc", [1]):
            profile = normalize_budget_profile({"cost_hard_stop_pct": bad})
            assert profile["cost_hard_stop_pct"] is None, bad
            assert task_pacing.resolve_cost_ceiling_usd(10.0, profile) == 5.0, bad
        # explicit 0 (and "0") stays a deliberate no-stop; whole percents clamp.
        assert normalize_budget_profile({"cost_hard_stop_pct": 0})["cost_hard_stop_pct"] == 0
        assert normalize_budget_profile({"cost_hard_stop_pct": "0"})["cost_hard_stop_pct"] == 0
        assert normalize_budget_profile({"cost_hard_stop_pct": 250})["cost_hard_stop_pct"] == 100


class TestCostCeilingStop:
    """_check_budget_limits consumes the pre-resolved ceiling as one parameter."""

    def test_no_ceiling_means_no_in_task_stop(self, tmp_path):
        """cost_hard_stop_pct=0 → ceiling None → even a huge task spend does not stop."""
        messages = []
        args = _make_args(
            budget_remaining_usd=100.0,
            accumulated_usage={"cost": 90.0},
            cost_ceiling_usd=None,
            messages=messages,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "999"}):
            result = _check_budget_limits(**args)
        assert result is None
        assert messages == []

    def test_custom_ceiling_stops_when_exceeded(self, tmp_path):
        llm = MagicMock()
        llm.chat.return_value = ({"content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})
        args = _make_args(
            budget_remaining_usd=100.0,
            accumulated_usage={"cost": 26.0},
            cost_ceiling_usd=25.0,
            llm=llm,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "999"}):
            result = _check_budget_limits(**args)
        assert result is not None

    def test_cost_equal_to_ceiling_does_not_stop(self, tmp_path):
        """Strict > preserves the historical edge (budget_pct > 0.5)."""
        args = _make_args(
            budget_remaining_usd=100.0,
            accumulated_usage={"cost": 25.0},
            cost_ceiling_usd=25.0,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "999"}):
            result = _check_budget_limits(**args)
        assert result is None


# --- v6.56.0 cost axis: latched milestones + wrap-up (task_pacing content) ---

class TestCostMilestones:
    def test_milestones_latch_once_and_sequence(self):
        ctx = SimpleNamespace()
        kw = dict(start_remaining_usd=20.0, cost_ceiling_usd=10.0)
        # 50% remaining of the $10 ceiling crossed.
        note = task_pacing.build_cost_budget_note(ctx, task_cost=5.1, **kw)
        assert note is not None and "50% remaining" in note.text
        assert note.checkpoint["checkpoint_kind"] == "cost_budget_milestone"
        assert note.checkpoint["hard_stop"] is True
        # Same spend again → latched, silent.
        assert task_pacing.build_cost_budget_note(ctx, task_cost=5.1, **kw) is None
        # 25% remaining crossed.
        note = task_pacing.build_cost_budget_note(ctx, task_cost=7.6, **kw)
        assert note is not None and "25% remaining" in note.text
        # ~80% spent → one-shot wrap-up.
        note = task_pacing.build_cost_budget_note(ctx, task_cost=8.1, **kw)
        assert note is not None and note.checkpoint["checkpoint_kind"] == "cost_budget_wrapup"
        assert task_pacing.build_cost_budget_note(ctx, task_cost=8.2, **kw) is None
        # 10% remaining crossed (wrap-up already latched, no duplicate).
        note = task_pacing.build_cost_budget_note(ctx, task_cost=9.1, **kw)
        assert note is not None and "10% remaining" in note.text
        assert task_pacing.build_cost_budget_note(ctx, task_cost=9.9, **kw) is None

    def test_jump_past_wrapup_with_milestone_suppresses_duplicate_wrapup(self):
        """A single jump deep past 80% spent fires the tightest milestone and
        latches wrap-up, so the next round does not double-note."""
        ctx = SimpleNamespace()
        kw = dict(start_remaining_usd=20.0, cost_ceiling_usd=10.0)
        note = task_pacing.build_cost_budget_note(ctx, task_cost=9.5, **kw)
        assert note is not None and "10% remaining" in note.text
        assert task_pacing.build_cost_budget_note(ctx, task_cost=9.6, **kw) is None

    def test_no_finite_budget_axis_is_silent(self):
        ctx = SimpleNamespace()
        assert task_pacing.build_cost_budget_note(
            ctx, start_remaining_usd=None, cost_ceiling_usd=None, task_cost=999.0,
        ) is None

    def test_uncapped_run_uses_start_snapshot_informationally(self):
        """cost_hard_stop_pct=0: milestones fire against the start snapshot,
        disclose there is no in-task stop, and clamp remaining at 0%."""
        ctx = SimpleNamespace()
        kw = dict(start_remaining_usd=10.0, cost_ceiling_usd=None)
        note = task_pacing.build_cost_budget_note(ctx, task_cost=5.5, **kw)
        assert note is not None and "no in-task cost stop" in note.text
        assert note.checkpoint["hard_stop"] is False
        # Spend past the whole snapshot: clamped, still just the tightest milestone.
        note = task_pacing.build_cost_budget_note(ctx, task_cost=25.0, **kw)
        assert note is not None and "10% remaining" in note.text
        assert "Remaining: ~$0.00" in note.text

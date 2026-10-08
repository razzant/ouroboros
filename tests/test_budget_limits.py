"""Tests for _check_budget_limits (global budget guard + tree-fed in-task ceiling)
and the cost axis (typed v6.91 ceiling states + latched v6.56.0 milestones):
the retired soft note, the global guard, local propagation, ceiling resolution
and the ceiling stop. The tree-fed deciding value, root-accounting telemetry
and cost milestones continue in ``test_budget_limits_tree``."""
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ouroboros import task_pacing
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.loop_budget import _check_budget_limits
from tests._budget_limits_helpers import EXPLICIT_50, _make_args


# --- The retired per-task soft reminder (v6.91) ---

class TestPerTaskSoftNoteRetired:
    """The pre-v6.91 own-cost "[COST NOTE]" keyed to OUROBOROS_PER_TASK_COST_USD
    is gone: since v6.64.0 the same key hard-fences the whole tree at the
    ledger, so the note could never fire before the fence (proven live)."""

    def test_no_soft_note_at_or_above_key_value(self, tmp_path):
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
        assert not any("[COST NOTE]" in m.get("content", "") for m in messages)

    def test_no_stop_from_the_key_alone(self, tmp_path):
        """The key does not stop the loop here; the ledger fence and the typed
        ceiling own that axis."""
        args = _make_args(accumulated_usage={"cost": 20.0}, round_idx=10, drive_logs=tmp_path)
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "5.0"}):
            result = _check_budget_limits(**args)
        assert result is None


# --- Global budget guard ---

class TestGlobalBudgetGuard:
    """Global exhaustion, and an EXPLICIT ``cost_hard_stop_pct=50`` profile's stop
    (``_make_args``' default ceiling); ordinary tasks have no percentage stop."""

    def test_an_ordinary_task_is_not_stopped_at_half_the_wallet(self, tmp_path):
        """Owner 2026-10-07 (#1128): no default early stop. 4.5 of 8 remaining is no
        reason to stop an ordinary task; only real limits on known spend are."""
        args = _make_args(
            budget_remaining_usd=8.0, accumulated_usage={"cost": 7.9}, drive_logs=tmp_path,
            cost_ceiling=task_pacing.resolve_cost_ceiling(8.0, normalize_budget_profile(None)),
        )
        assert args["cost_ceiling"].state == task_pacing.COST_CEILING_DISABLED
        with (
            patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "10.0"}),
            patch("ouroboros.loop.call_llm_with_retry") as model_call,
        ):
            assert _check_budget_limits(**args) is None
        model_call.assert_not_called()

    def test_no_global_budget_and_no_root_cap_is_silent(self, tmp_path):
        """Neither axis finite (GAIA shape, no per-task cap) → the whole cost
        axis stays silent. Note this is decided by the DISABLED ceiling, not by
        an early return on the global number — see the root-cap test below."""
        args = _make_args(budget_remaining_usd=None, accumulated_usage={"cost": 100.0}, drive_logs=tmp_path)
        assert args["cost_ceiling"].state == task_pacing.COST_CEILING_DISABLED
        result = _check_budget_limits(**args)
        assert result is None

    def test_budget_exhausted(self, tmp_path):
        """Remaining ≤ 0 → immediate stop."""
        args = _make_args(budget_remaining_usd=0.0, accumulated_usage={"cost": 0.01}, drive_logs=tmp_path)
        with (
            patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "999"}),
            patch("ouroboros.loop.call_llm_with_retry") as model_call,
        ):
            result = _check_budget_limits(**args)
        assert result is not None
        text, _, _ = result
        assert "budget exhausted" in text.lower()
        model_call.assert_not_called()

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
    """Ensure use_local is passed to call_llm_with_retry on global budget stop."""

    @patch("ouroboros.loop.call_llm_with_retry")
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


# --- v6.91 typed cost ceiling resolution ---

class TestCostCeilingResolution:
    """task_pacing.resolve_cost_ceiling: typed states, global pct component,
    root-cap-minus-margin component."""

    def test_absent_profile_has_no_default_stop(self):
        """Owner 2026-10-07 (#1128): no half-wallet share and no cap-minus-margin for
        an ordinary task — not at a large wallet, a small one, or under any cap."""
        profile = normalize_budget_profile(None)
        for wallet, cap in ((10.0, None), (480.0, 400.0), (1900.0, 100.0), (None, 50.0), (100.0, 0.5)):
            ceiling = task_pacing.resolve_cost_ceiling(wallet, profile, root_cap_usd=cap)
            assert ceiling.state == task_pacing.COST_CEILING_DISABLED, (wallet, cap)
            assert ceiling.ceiling_usd is None and ceiling.planning_margin_usd is None
            assert ceiling.root_cap_usd == cap
            assert ceiling.basis == task_pacing.COST_BASIS_NO_DEFAULT_STOP

    def test_explicit_50_keeps_the_authored_half_of_the_wallet(self):
        ceiling = task_pacing.resolve_cost_ceiling(10.0, EXPLICIT_50)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        assert ceiling.ceiling_usd == 5.0
        assert ceiling.root_cap_usd is None

    def test_zero_pct_means_disabled_never_zero_dollars(self):
        profile = normalize_budget_profile({"cost_hard_stop_pct": 0})
        ceiling = task_pacing.resolve_cost_ceiling(10.0, profile)
        assert ceiling.state == task_pacing.COST_CEILING_DISABLED
        assert ceiling.ceiling_usd is None

    def test_zero_pct_disabled_even_with_tiny_root_cap(self):
        """The bench contract (SWE-Pro: pct=0 + a small root cap) keeps the
        in-task stop fully off — the ledger fence is the only stop."""
        profile = normalize_budget_profile({"cost_hard_stop_pct": 0})
        ceiling = task_pacing.resolve_cost_ceiling(10.0, profile, root_cap_usd=0.5)
        assert ceiling.state == task_pacing.COST_CEILING_DISABLED
        assert ceiling.ceiling_usd is None

    def test_custom_pct(self):
        profile = normalize_budget_profile({"cost_hard_stop_pct": 25})
        ceiling = task_pacing.resolve_cost_ceiling(10.0, profile)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        assert ceiling.ceiling_usd == 2.5

    def test_no_finite_budget_means_disabled_axis(self):
        profile = normalize_budget_profile(None)
        assert task_pacing.resolve_cost_ceiling(None, profile).state == task_pacing.COST_CEILING_DISABLED
        assert task_pacing.resolve_cost_ceiling(0.0, profile).state == task_pacing.COST_CEILING_DISABLED

    def test_root_cap_component_binds_when_smaller(self):
        """Explicit profile: min(pct-of-global, cap − margin): the live wave1/2 shape —
        a huge global remaining must not hide a $100 tree cap."""
        profile = EXPLICIT_50
        ceiling = task_pacing.resolve_cost_ceiling(1900.0, profile, root_cap_usd=100.0)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        assert ceiling.root_cap_usd == 100.0
        assert ceiling.ceiling_usd == 100.0 - task_pacing.COST_PLANNING_MARGIN_USD
        assert ceiling.planning_margin_usd == task_pacing.COST_PLANNING_MARGIN_USD

    def test_global_pct_component_binds_when_smaller(self):
        profile = EXPLICIT_50
        ceiling = task_pacing.resolve_cost_ceiling(10.0, profile, root_cap_usd=100.0)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        assert ceiling.ceiling_usd == 5.0

    def test_root_cap_only_no_finite_global(self):
        """Explicit profile: a per-task cap with an unbounded global still yields an active stop."""
        profile = EXPLICIT_50
        ceiling = task_pacing.resolve_cost_ceiling(None, profile, root_cap_usd=50.0)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        assert ceiling.ceiling_usd == 50.0 - task_pacing.COST_PLANNING_MARGIN_USD

    def test_cap_at_or_below_margin_soft_lands_never_uncapped(self):
        """A root cap at/below the planning margin must resolve to the typed
        soft-land state — the pre-typed shape returned the same None as
        'unlimited' (a $0.50 bench cap would have run uncapped). An explicit
        profile's math; the absent profile has no margin to land under."""
        profile = EXPLICIT_50
        for cap in (0.5, task_pacing.COST_PLANNING_MARGIN_USD):
            ceiling = task_pacing.resolve_cost_ceiling(100.0, profile, root_cap_usd=cap)
            assert ceiling.state == task_pacing.COST_CEILING_EXHAUSTED_SOFT_LAND, cap
            assert ceiling.ceiling_usd is None
            assert ceiling.root_cap_usd == cap

    def test_ceiling_is_never_computed_zero(self):
        profile = EXPLICIT_50
        just_above = task_pacing.COST_PLANNING_MARGIN_USD + 0.01
        ceiling = task_pacing.resolve_cost_ceiling(1000.0, profile, root_cap_usd=just_above)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        assert ceiling.ceiling_usd is not None and ceiling.ceiling_usd > 0
        # The boundary is MEASURED, not implied: `> 0` alone reads as "there is
        # working room", but the bail is exactly the owner's `room <= 0` rule, so
        # a cap one cent above the margin buys exactly one cent of ceiling — a
        # stop-on-the-first-spend ceiling. Widening this into a minimum-room floor
        # would move caps the owner deliberately allows into immediate soft-land,
        # which is an owner decision; pinning the number keeps it visible instead.
        assert round(ceiling.ceiling_usd, 6) == 0.01
        assert ceiling.root_cap_usd == just_above
        assert ceiling.planning_margin_usd == task_pacing.COST_PLANNING_MARGIN_USD
        at_margin = task_pacing.resolve_cost_ceiling(
            1000.0, profile, root_cap_usd=task_pacing.COST_PLANNING_MARGIN_USD,
        )
        assert at_margin.state == task_pacing.COST_CEILING_EXHAUSTED_SOFT_LAND

    def test_per_task_cap_setting_note_states_the_known_spend_limit(self):
        """The owner reads the per-task cap as the task's limit (#1128): the note says
        the cap itself is where known spend stops new calls, that a resumable task pauses
        with its work saved (any other run ends budget-exhausted), and that in-flight calls
        can overshoot — and promises no wrap-up margin, because an ordinary task has none
        (even at a tiny cap)."""
        from ouroboros.settings_setup_contract import build_setup_contract

        fields = {
            str(field.get("settingKey")): field
            for field in build_setup_contract().get("budgetFields", [])
        }
        note = str(fields["OUROBOROS_PER_TASK_COST_USD"]["note"])
        assert "known spend" in note and "pauses" in note and "budget-exhausted" in note and "in flight" in note
        assert "Raising the cap does not resume a paused task" in note
        assert "wrap-up" not in note and "margin" not in note
        assert task_pacing.resolve_cost_ceiling(
            1000.0, normalize_budget_profile(None),
            root_cap_usd=task_pacing.COST_PLANNING_MARGIN_USD,
        ).state == task_pacing.COST_CEILING_DISABLED

    def test_planning_margin_is_absolute_not_pct(self):
        """An explicit profile's margin must not scale with the cap (a pct reserve
        amputated the tail of long tasks — v6.54.4 r1; the money-axis analogue)."""
        profile = EXPLICIT_50
        small = task_pacing.resolve_cost_ceiling(None, profile, root_cap_usd=10.0)
        large = task_pacing.resolve_cost_ceiling(None, profile, root_cap_usd=1000.0)
        assert small.ceiling_usd == 10.0 - task_pacing.COST_PLANNING_MARGIN_USD
        assert large.ceiling_usd == 1000.0 - task_pacing.COST_PLANNING_MARGIN_USD

    def test_malformed_pct_keeps_its_input_contract_and_reads_as_absent(self):
        """The accepted-input contract is unchanged: negative / non-numeric / a 0<v<1
        fraction map to None, never to an explicit 0. None now means what an absent
        profile means — no default stop — and is disclosed as that, never as the
        explicit-zero bench contract."""
        for bad in (-5, -0.1, 0.5, "0.5", "abc", [1]):
            profile = normalize_budget_profile({"cost_hard_stop_pct": bad})
            assert profile["cost_hard_stop_pct"] is None, bad
            ceiling = task_pacing.resolve_cost_ceiling(10.0, profile)
            assert ceiling.state == task_pacing.COST_CEILING_DISABLED, bad
            assert ceiling.basis == task_pacing.COST_BASIS_NO_DEFAULT_STOP, bad
        # explicit 0 (and "0") stays a deliberate no-stop; whole percents clamp.
        assert normalize_budget_profile({"cost_hard_stop_pct": 0})["cost_hard_stop_pct"] == 0
        assert normalize_budget_profile({"cost_hard_stop_pct": "0"})["cost_hard_stop_pct"] == 0
        assert normalize_budget_profile({"cost_hard_stop_pct": 250})["cost_hard_stop_pct"] == 100


# --- В26=A: a wake's tree carries a GRACEFUL ceiling, never a narrowed fence ---

class TestWakeRootCeiling:
    """A Background Consciousness wake-up is the ROOT of its own tree and carries what is
    left of its allowance as `metadata.root_cost_ceiling_usd`. `resolve_cost_ceiling` honors
    that number for the root itself (members inherit the resolved ceiling as before), while
    the ledger fence keeps the owner's per-task cap: one Main attempt reserves several
    dollars up front, and a fence narrowed below that (the earlier `root_limit_usd`
    narrowing) refused every wake of a nearly spent day before its first call — a
    `budget_exhausted` bubble in Main at every heartbeat.
    """

    def _scope(self, tmp_path, monkeypatch, *, metadata=None, per_task_cap="50"):
        """The scope `handle_task` actually binds for one task."""
        from ouroboros.agent import OuroborosAgent
        from ouroboros.usage_accounting import current_usage_scope

        monkeypatch.setattr("ouroboros.subagent_runtime.apply_task_start_settings_or_disclose",
                            lambda *_a, **_k: None)
        monkeypatch.setattr("ouroboros.model_wait.task_model_wait_scope", lambda **_k: nullcontext())
        monkeypatch.setenv("TOTAL_BUDGET", "1000")
        monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", per_task_cap)
        host = SimpleNamespace(
            env=SimpleNamespace(drive_root=tmp_path), _emit_live_log=lambda *_a, **_k: None,
            _event_queue=None, _handle_task_scoped=lambda _task: current_usage_scope(),
        )
        task = {"id": "wake-1", "type": "task"}
        if metadata is not None:
            task["metadata"] = metadata
        return OuroborosAgent.handle_task(host, task)

    def test_without_metadata_the_owner_setting_is_the_cap(self, tmp_path, monkeypatch):
        assert self._scope(tmp_path, monkeypatch).root_limit_usd == 50.0

    def test_the_ceiling_rides_the_scope_and_the_fence_keeps_the_cap(self, tmp_path, monkeypatch):
        scope = self._scope(tmp_path, monkeypatch, metadata={"root_cost_ceiling_usd": 0.66})
        assert scope.root_limit_usd == 50.0 and scope.root_cost_ceiling_usd == 0.66

    def test_metadata_never_narrows_the_fence(self, tmp_path, monkeypatch):
        """No producer narrows the ledger fence through metadata: a fence below one
        attempt's reservation is a refusal before the first call, not a soft landing."""
        assert self._scope(tmp_path, monkeypatch, metadata={"root_limit_usd": 0.66}).root_limit_usd == 50.0
        assert self._scope(tmp_path, monkeypatch, metadata={"root_limit_usd": 0.66}, per_task_cap="0").root_limit_usd is None

    def test_a_thin_producer_allowance_is_the_working_ceiling_without_a_margin(self, tmp_path, monkeypatch):
        """Owner 2026-10-07: $0.66 left of the allowance is $0.66 of work, not an
        immediate landing under a $3 margin. The allowance stays a real restriction."""
        from ouroboros.usage_accounting import usage_scope

        ctx = SimpleNamespace()
        with usage_scope(self._scope(tmp_path, monkeypatch, metadata={"root_cost_ceiling_usd": 0.66})):
            ceiling = task_pacing.resolve_task_cost_ceiling(ctx, 1000.0)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE and ceiling.ceiling_usd == 0.66
        assert ceiling.root_cap_usd == 50.0 and ceiling.basis == "producer_allowance"
        assert ceiling.planning_margin_usd is None
        assert task_pacing.cost_stop_authority(ctx) == task_pacing.COST_STOP_PRODUCER

    def test_an_explicit_profile_keeps_its_margin_under_a_producer(self):
        """Explicit experiment math is kept verbatim, including its producer margin."""
        thin = task_pacing.resolve_cost_ceiling(1000.0, EXPLICIT_50, root_cap_usd=50.0, root_ceiling_usd=0.66)
        assert thin.state == task_pacing.COST_CEILING_EXHAUSTED_SOFT_LAND
        assert thin.basis == "root_ceiling_at_or_below_planning_margin"
        wide = task_pacing.resolve_cost_ceiling(1000.0, EXPLICIT_50, root_cap_usd=50.0, root_ceiling_usd=9.0)
        assert wide.ceiling_usd == 9.0 - task_pacing.COST_PLANNING_MARGIN_USD
        assert "root_ceiling_minus_margin" in wide.basis

    def test_a_root_ceiling_above_the_margin_is_the_working_ceiling(self, tmp_path, monkeypatch):
        from ouroboros.usage_accounting import usage_scope

        with usage_scope(self._scope(tmp_path, monkeypatch, metadata={"root_cost_ceiling_usd": 9.0})):
            ceiling = task_pacing.resolve_task_cost_ceiling(SimpleNamespace(), 1000.0)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        assert ceiling.root_cap_usd == 50.0
        assert ceiling.ceiling_usd == 9.0 and ceiling.basis == "producer_allowance"

    def test_a_member_still_inherits_the_resolved_ceiling(self):
        """The root's number reaches its members as before; a member's own carrier is the
        inherited resolution, never re-derived from a root-only ceiling."""
        member = task_pacing.resolve_cost_ceiling(
            1000.0, normalize_budget_profile({}), root_cap_usd=50.0, non_root_member=True, root_ceiling_usd=6.0)
        assert member.state == task_pacing.COST_CEILING_ACTIVE and member.ceiling_usd == 6.0

class TestCostCeilingStop:
    """_check_budget_limits consumes the typed pre-resolved ceiling."""

    def test_no_active_ceiling_means_no_in_task_stop(self, tmp_path):
        """disabled state → even a huge task spend does not stop here."""
        messages = []
        disabled = task_pacing.resolve_cost_ceiling(
            100.0, normalize_budget_profile({"cost_hard_stop_pct": 0}),
        )
        args = _make_args(
            budget_remaining_usd=100.0,
            accumulated_usage={"cost": 90.0},
            cost_ceiling=disabled,
            messages=messages,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "999"}):
            result = _check_budget_limits(**args)
        assert result is None
        assert messages == []

    def test_none_ceiling_object_means_no_in_task_stop(self, tmp_path):
        args = _make_args(
            budget_remaining_usd=100.0,
            accumulated_usage={"cost": 90.0},
            cost_ceiling=None,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "999"}):
            result = _check_budget_limits(**args)
        assert result is None

    def test_custom_ceiling_stops_when_exceeded(self, tmp_path):
        llm = MagicMock()
        llm.chat.return_value = ({"content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})
        args = _make_args(
            budget_remaining_usd=100.0,
            accumulated_usage={"cost": 26.0},
            cost_ceiling=task_pacing.resolve_cost_ceiling(50.0, EXPLICIT_50),
            llm=llm,
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "999"}):
            result = _check_budget_limits(**args)
        assert result is not None

    def test_cost_equal_to_ceiling_does_not_stop(self, tmp_path):
        """Strict > preserves the explicit profile's historical edge (budget_pct > 0.5)."""
        args = _make_args(
            budget_remaining_usd=100.0,
            accumulated_usage={"cost": 25.0},
            cost_ceiling=task_pacing.resolve_cost_ceiling(50.0, EXPLICIT_50),
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "999"}):
            result = _check_budget_limits(**args)
        assert result is None

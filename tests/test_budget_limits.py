"""Tests for _check_budget_limits (global budget guard + tree-fed in-task ceiling)
and the cost axis (typed v6.91 ceiling states + latched v6.56.0 milestones)."""
import os
import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ouroboros import task_pacing
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.loop_budget import _check_budget_limits
from ouroboros.loop_round_limits import _RoundLimitContext


def _make_args(**overrides):
    """Build default kwargs for _check_budget_limits.

    ``cost_ceiling`` defaults to the typed resolution with no root cap, so the
    legacy guard tests keep exercising the historical 50%-of-global semantics
    the runtime gets from ``task_pacing.resolve_cost_ceiling`` with an absent
    profile.
    """
    llm = MagicMock()
    llm.chat.return_value = (
        {"role": "assistant", "content": ""},
        {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
    )
    defaults = dict(
        budget_remaining_usd=100.0,
        accumulated_usage={"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        round_idx=0,
        messages=[],
        llm=llm,
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
    if "cost_ceiling" not in defaults:
        defaults["cost_ceiling"] = task_pacing.resolve_cost_ceiling(
            defaults["budget_remaining_usd"], normalize_budget_profile(None),
        )
    budget_remaining_usd = defaults.pop("budget_remaining_usd")
    cost_ceiling = defaults.pop("cost_ceiling")
    ctx = _RoundLimitContext(
        messages=defaults["messages"],
        llm=defaults["llm"],
        active_model=defaults["active_model"],
        active_effort=defaults["active_effort"],
        max_retries=defaults["max_retries"],
        drive_logs=defaults["drive_logs"],
        task_id=defaults["task_id"],
        round_idx=defaults["round_idx"],
        event_queue=defaults["event_queue"],
        accumulated_usage=defaults["accumulated_usage"],
        task_type=defaults["task_type"],
        active_use_local=defaults["use_local"],
        max_rounds=100,
        deadline_ts=defaults.get("deadline_ts"),
        llm_trace=defaults["llm_trace"],
    )
    return {
        "ctx": ctx,
        "budget_remaining_usd": budget_remaining_usd,
        "cost_ceiling": cost_ceiling,
    }


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
    """Existing global budget percentage checks."""

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

    def test_absent_profile_means_historical_50pct(self):
        profile = normalize_budget_profile(None)
        ceiling = task_pacing.resolve_cost_ceiling(10.0, profile)
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
        """min(pct-of-global, cap − margin): the live wave1/2 shape — a huge
        global remaining must not hide a $100 tree cap."""
        profile = normalize_budget_profile(None)
        ceiling = task_pacing.resolve_cost_ceiling(1900.0, profile, root_cap_usd=100.0)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        assert ceiling.root_cap_usd == 100.0
        assert ceiling.ceiling_usd == 100.0 - task_pacing.COST_PLANNING_MARGIN_USD
        assert ceiling.planning_margin_usd == task_pacing.COST_PLANNING_MARGIN_USD

    def test_global_pct_component_binds_when_smaller(self):
        profile = normalize_budget_profile(None)
        ceiling = task_pacing.resolve_cost_ceiling(10.0, profile, root_cap_usd=100.0)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        assert ceiling.ceiling_usd == 5.0

    def test_root_cap_only_no_finite_global(self):
        """A per-task cap with an unbounded global still yields an active stop."""
        profile = normalize_budget_profile(None)
        ceiling = task_pacing.resolve_cost_ceiling(None, profile, root_cap_usd=50.0)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        assert ceiling.ceiling_usd == 50.0 - task_pacing.COST_PLANNING_MARGIN_USD

    def test_cap_at_or_below_margin_soft_lands_never_uncapped(self):
        """A root cap at/below the planning margin must resolve to the typed
        soft-land state — the pre-typed shape returned the same None as
        'unlimited' (a $0.50 bench cap would have run uncapped)."""
        profile = normalize_budget_profile(None)
        for cap in (0.5, task_pacing.COST_PLANNING_MARGIN_USD):
            ceiling = task_pacing.resolve_cost_ceiling(100.0, profile, root_cap_usd=cap)
            assert ceiling.state == task_pacing.COST_CEILING_EXHAUSTED_SOFT_LAND, cap
            assert ceiling.ceiling_usd is None
            assert ceiling.root_cap_usd == cap

    def test_ceiling_is_never_computed_zero(self):
        profile = normalize_budget_profile(None)
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

    def test_per_task_cap_setting_note_states_the_immediate_finalization(self):
        """The owner-facing note must not promise a wrap-up a small cap cannot get.

        The field still accepts a cap below the wrap-up margin (owner power is
        preserved), but such a cap resolves to `exhausted_soft_land`, which the
        loop turns into a forced final answer at the TOP of round 0 — zero work
        rounds. The note said only "a graceful wrap-up fires just before"."""
        from ouroboros.settings_setup_contract import build_setup_contract

        fields = {
            str(field.get("settingKey")): field
            for field in build_setup_contract().get("budgetFields", [])
        }
        note = str(fields["OUROBOROS_PER_TASK_COST_USD"]["note"])
        assert f"${task_pacing.COST_PLANNING_MARGIN_USD:.2f}" in note
        assert "finalizes the task immediately" in note
        # A cap at the documented boundary really does behave that way.
        assert task_pacing.resolve_cost_ceiling(
            1000.0, normalize_budget_profile(None),
            root_cap_usd=task_pacing.COST_PLANNING_MARGIN_USD,
        ).state == task_pacing.COST_CEILING_EXHAUSTED_SOFT_LAND

    def test_planning_margin_is_absolute_not_pct(self):
        """The margin must not scale with the cap (a pct reserve amputated the
        tail of long tasks — v6.54.4 r1; the money-axis analogue is pinned)."""
        profile = normalize_budget_profile(None)
        small = task_pacing.resolve_cost_ceiling(None, profile, root_cap_usd=10.0)
        large = task_pacing.resolve_cost_ceiling(None, profile, root_cap_usd=1000.0)
        assert small.ceiling_usd == 10.0 - task_pacing.COST_PLANNING_MARGIN_USD
        assert large.ceiling_usd == 1000.0 - task_pacing.COST_PLANNING_MARGIN_USD

    def test_malformed_pct_fails_safe_to_default_not_zero(self):
        """A garbage cost_hard_stop_pct must NOT silently become 0 (= no in-task
        stop, the most permissive setting): negative / non-numeric / a 0<v<1
        fraction map to None (the 50% default), while an explicit 0 is honored."""
        for bad in (-5, -0.1, 0.5, "0.5", "abc", [1]):
            profile = normalize_budget_profile({"cost_hard_stop_pct": bad})
            assert profile["cost_hard_stop_pct"] is None, bad
            ceiling = task_pacing.resolve_cost_ceiling(10.0, profile)
            assert ceiling.state == task_pacing.COST_CEILING_ACTIVE, bad
            assert ceiling.ceiling_usd == 5.0, bad
        # explicit 0 (and "0") stays a deliberate no-stop; whole percents clamp.
        assert normalize_budget_profile({"cost_hard_stop_pct": 0})["cost_hard_stop_pct"] == 0
        assert normalize_budget_profile({"cost_hard_stop_pct": "0"})["cost_hard_stop_pct"] == 0
        assert normalize_budget_profile({"cost_hard_stop_pct": 250})["cost_hard_stop_pct"] == 100


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
            cost_ceiling=task_pacing.resolve_cost_ceiling(
                50.0, normalize_budget_profile(None),
            ),
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
            cost_ceiling=task_pacing.resolve_cost_ceiling(
                50.0, normalize_budget_profile(None),
            ),
            drive_logs=tmp_path,
        )
        with patch.dict(os.environ, {"OUROBOROS_PER_TASK_COST_USD": "999"}):
            result = _check_budget_limits(**args)
        assert result is None


# --- v6.91 tree-fed deciding value ---

class TestTreeFedDecidingValue:
    """Under a root cap the deciding spend is the root subtree's ledger-accounted
    number from the reserve-time scope telemetry — own cost stays a diagnostic.
    The waves died at tree $84-94 while own cost showed $41-49 and no warning
    ever fired; these pin the closed class."""

    def _scoped(self, root_id, root_limit):
        from ouroboros.usage_accounting import UsageScope, usage_scope

        return usage_scope(UsageScope(
            drive_root=None, task_id=root_id, root_task_id=root_id,
            root_limit_usd=root_limit,
        ))

    def test_tree_spend_over_ceiling_stops_even_when_own_is_low(self, tmp_path):
        from ouroboros import usage_accounting

        llm = MagicMock()
        llm.chat.return_value = ({"content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})
        ceiling = task_pacing.resolve_cost_ceiling(
            1900.0, normalize_budget_profile(None), root_cap_usd=100.0,
        )
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        args = _make_args(
            budget_remaining_usd=1900.0,
            accumulated_usage={"cost": 41.0},  # own: far below the ceiling
            cost_ceiling=ceiling,
            llm=llm,
            drive_logs=tmp_path,
        )
        with self._scoped("root-tree-1", 100.0):
            usage_accounting._stash_root_accounting("root-tree-1", 98.5, 100.0)
            result = _check_budget_limits(**args)
        assert result is not None
        text, usage, _ = result
        assert usage.get("reason_code") == "budget_exhausted"

    def test_tree_spend_under_ceiling_does_not_stop(self, tmp_path):
        from ouroboros import usage_accounting

        ceiling = task_pacing.resolve_cost_ceiling(
            1900.0, normalize_budget_profile(None), root_cap_usd=100.0,
        )
        args = _make_args(
            budget_remaining_usd=1900.0,
            accumulated_usage={"cost": 41.0},
            cost_ceiling=ceiling,
            drive_logs=tmp_path,
        )
        with self._scoped("root-tree-2", 100.0):
            usage_accounting._stash_root_accounting("root-tree-2", 60.0, 100.0)
            result = _check_budget_limits(**args)
        assert result is None

    def test_unknown_tree_falls_back_to_own_cost_never_zero(self, tmp_path):
        """No telemetry for this tree → the deciding value falls back to own
        cost (a real number), never a coerced $0 that would disable the stop.

        No root cap here, so own cost is the COMPLETE picture (there is no tree
        fence at all) and the basis says exactly that."""
        llm = MagicMock()
        llm.chat.return_value = ({"content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})
        ceiling = task_pacing.resolve_cost_ceiling(
            10.0, normalize_budget_profile(None),
        )
        usage = {"cost": 6.0}  # own over the $5 ceiling
        args = _make_args(
            budget_remaining_usd=10.0,
            accumulated_usage=usage,
            cost_ceiling=ceiling,
            llm=llm,
            drive_logs=tmp_path,
        )
        result = _check_budget_limits(**args)
        assert result is not None
        assert usage["cost_stop_spend_basis"] == task_pacing.SPEND_BASIS_OWN_NO_TREE_CAP

    def test_unknown_tree_under_a_root_cap_is_disclosed_not_silent(self, tmp_path, monkeypatch):
        """A root cap exists but the tree number is unavailable this round (no
        stash and the ledger read fails): the stop still fires on own cost (a
        usable lower bound), and BOTH the text and the usage record say the
        substitution happened (BIBLE P1) instead of presenting an own-cost
        number as if it were the tree."""
        from ouroboros import usage_accounting

        def _unavailable(*args, **kwargs):
            raise OSError("ledger unreadable")

        monkeypatch.setattr(usage_accounting, "usage_projection", _unavailable)
        llm = MagicMock()
        llm.chat.return_value = ({"content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})
        ceiling = task_pacing.resolve_cost_ceiling(
            1900.0, normalize_budget_profile(None), root_cap_usd=10.0,
        )
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        usage = {"cost": 9.0}  # over the $10 − margin ceiling
        args = _make_args(
            budget_remaining_usd=1900.0,
            accumulated_usage=usage,
            cost_ceiling=ceiling,
            llm=llm,
            drive_logs=tmp_path,
        )
        # No stash for this root id → tree spend genuinely unknown.
        with self._scoped("root-tree-unknown", 10.0):
            result = _check_budget_limits(**args)
        assert result is not None
        _text, out_usage, _ = result
        assert out_usage["cost_stop_spend_basis"] == task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN
        # The wrap-up prompt the agent actually receives states the basis.
        prompt_text = "\n".join(
            str(m.get("content") or "") for m in args["ctx"].messages if isinstance(m, dict)
        )
        assert "lower bound" in prompt_text and "OWN calls" in prompt_text

    def test_root_cap_ceiling_fires_without_any_global_budget(self, tmp_path):
        """The closed class (v6.91 audit): an explicit non-positive budget makes
        ``budget_remaining_usd`` None, but a live per-task ROOT CAP must still
        stop the task. The pre-fix guard returned None before ever looking at
        the ceiling, so a GAIA-shaped run could never soft-land."""
        from ouroboros import usage_accounting

        llm = MagicMock()
        llm.chat.return_value = ({"content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})
        ceiling = task_pacing.resolve_cost_ceiling(
            None, normalize_budget_profile(None), root_cap_usd=100.0,
        )
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE
        args = _make_args(
            budget_remaining_usd=None,
            accumulated_usage={"cost": 41.0},
            cost_ceiling=ceiling,
            llm=llm,
            drive_logs=tmp_path,
        )
        with self._scoped("root-tree-noglobal", 100.0):
            usage_accounting._stash_root_accounting("root-tree-noglobal", 98.5, 100.0)
            result = _check_budget_limits(**args)
        assert result is not None
        _text, usage, _ = result
        assert usage.get("reason_code") == "budget_exhausted"
        assert usage["cost_stop_spend_basis"] == task_pacing.SPEND_BASIS_TREE

    def test_fresh_stash_costs_no_ledger_read(self, tmp_path, monkeypatch):
        """The deciding surface must not become a per-round ledger read: while
        rounds are shorter than the staleness bound the free stash (refreshed by
        every dispatch) answers, and `usage_projection` is never called."""
        from ouroboros import loop as loop_mod
        from ouroboros import usage_accounting

        calls = {"n": 0}

        def _boom(*args, **kwargs):
            calls["n"] += 1
            raise AssertionError("per-round ledger read")

        monkeypatch.setattr(usage_accounting, "usage_projection", _boom)
        assert loop_mod._TREE_ACCOUNTING_MAX_STALE_SEC > 0
        args = _make_args(
            budget_remaining_usd=1900.0,
            accumulated_usage={"cost": 1.0},
            cost_ceiling=task_pacing.resolve_cost_ceiling(
                1900.0, normalize_budget_profile(None), root_cap_usd=100.0,
            ),
            drive_logs=tmp_path,
        )
        with self._scoped("root-tree-fresh", 100.0):
            usage_accounting._stash_root_accounting("root-tree-fresh", 5.0, 100.0)
            assert _check_budget_limits(**args) is None
        assert calls["n"] == 0

    def test_stash_older_than_the_bound_is_refreshed_once(self, tmp_path, monkeypatch):
        """A round that blocks longer than the bound (the 900s wait_tasks shape
        that killed both waves, during which children spent) pays for exactly
        one real projection read rather than deciding on a stale number."""
        from ouroboros import usage_accounting

        llm = MagicMock()
        llm.chat.return_value = ({"content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})
        reads = {"n": 0}

        def _fresh_projection(drive_root, **kwargs):
            reads["n"] += 1
            return {"accounted_usd": 98.5, "limit_usd": 100.0}

        monkeypatch.setattr(usage_accounting, "usage_projection", _fresh_projection)
        args = _make_args(
            budget_remaining_usd=1900.0,
            accumulated_usage={"cost": 41.0},
            cost_ceiling=task_pacing.resolve_cost_ceiling(
                1900.0, normalize_budget_profile(None), root_cap_usd=100.0,
            ),
            llm=llm,
            drive_logs=tmp_path,
        )
        with self._scoped("root-tree-stale", 100.0):
            # Stash a pre-block number, then age it past the bound.
            usage_accounting._stash_root_accounting("root-tree-stale", 40.0, 100.0)
            with usage_accounting._ROOT_ACCOUNTING_TELEMETRY_LOCK:
                usage_accounting._ROOT_ACCOUNTING_TELEMETRY["root-tree-stale"][
                    "updated_monotonic"
                ] -= 10_000.0
            result = _check_budget_limits(**args)
        assert reads["n"] == 1
        assert result is not None
        _text, usage, _ = result
        assert usage["cost_stop_spend_basis"] == task_pacing.SPEND_BASIS_TREE

    def test_current_inflight_reservation_participates_in_the_stop(self, tmp_path, monkeypatch):
        """G3-4 regression: the stash the loop trusts for 120s must include the
        reservation of the call currently in flight, not the pre-append sum.
        The pre-fix shape: a tree near its cap reserved+settled one more call,
        the loop still saw the pre-call number, and the hard ledger fence fired
        on the next send before the graceful wrap-up ever ran. Here the ceiling
        check stops on the un-settled hold alone, with zero fresh ledger reads."""
        from ouroboros import usage_accounting
        from ouroboros.usage_accounting import AttemptRequest, UsageScope, usage_scope

        def _boom(*args, **kwargs):
            raise AssertionError("per-round ledger read")

        monkeypatch.setattr(usage_accounting, "usage_projection", _boom)
        llm = MagicMock()
        llm.chat.return_value = ({"content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})
        args = _make_args(
            budget_remaining_usd=1900.0,
            accumulated_usage={"cost": 0.5},
            cost_ceiling=task_pacing.resolve_cost_ceiling(
                1900.0, normalize_budget_profile(None), root_cap_usd=100.0,
            ),
            llm=llm,
            drive_logs=tmp_path,
        )
        scope = UsageScope(
            drive_root=tmp_path, task_id="root-inflight", root_task_id="root-inflight",
            global_limit_usd=1900.0, root_limit_usd=100.0,
        )
        with usage_scope(scope):
            usage_accounting.reserve_attempt(AttemptRequest(
                model="test/model", provider="openrouter",
                reservation_usd=99.5, drive_root=tmp_path,
            ))
            # The attempt is still in flight (never settled) — its hold alone
            # must already be visible to the deciding surface.
            entry = usage_accounting.last_root_accounting("root-inflight")
            assert entry is not None and entry["accounted_usd"] == 99.5
            result = _check_budget_limits(**args)
        assert result is not None
        _text, usage, _ = result
        assert usage.get("reason_code") == "budget_exhausted"
        assert usage["cost_stop_spend_basis"] == task_pacing.SPEND_BASIS_TREE


class TestRootAccountingTelemetry:
    def test_stash_roundtrip_and_age(self):
        from ouroboros import usage_accounting

        usage_accounting._stash_root_accounting("root-t-1", 12.5, 100.0)
        entry = usage_accounting.last_root_accounting("root-t-1")
        assert entry is not None
        assert entry["accounted_usd"] == 12.5
        assert entry["root_limit_usd"] == 100.0
        assert entry["age_sec"] >= 0.0

    def test_unknown_root_is_none(self):
        from ouroboros import usage_accounting

        assert usage_accounting.last_root_accounting("no-such-root") is None
        assert usage_accounting.last_root_accounting("") is None

    def test_refresh_reads_ledger_and_updates_stash(self, tmp_path, monkeypatch):
        from ouroboros import usage_accounting

        monkeypatch.setattr(
            usage_accounting, "usage_projection",
            lambda *a, **k: {"accounted_usd": 7.25, "limit_usd": 25.0},
        )
        entry = usage_accounting.refresh_root_accounting(tmp_path, "root-t-2")
        assert entry is not None and entry["accounted_usd"] == 7.25
        assert usage_accounting.last_root_accounting("root-t-2")["root_limit_usd"] == 25.0

    def test_refresh_failure_returns_stale_stash_not_zero(self, tmp_path, monkeypatch):
        from ouroboros import usage_accounting

        usage_accounting._stash_root_accounting("root-t-3", 3.0, 10.0)

        def _boom(*a, **k):
            raise RuntimeError("ledger unavailable")

        monkeypatch.setattr(usage_accounting, "usage_projection", _boom)
        entry = usage_accounting.refresh_root_accounting(tmp_path, "root-t-3")
        assert entry is not None and entry["accounted_usd"] == 3.0

    def test_reserve_attempt_piggybacks_tree_sum(self, tmp_path):
        """The stash is a byproduct of the existing in-lock computation — no new
        ledger read path (the e4a87344 starvation constraint)."""
        from ouroboros import usage_accounting
        from ouroboros.usage_accounting import AttemptRequest, UsageScope, usage_scope

        scope = UsageScope(
            drive_root=tmp_path, task_id="rroot", root_task_id="rroot",
            global_limit_usd=100.0, root_limit_usd=50.0,
        )
        with usage_scope(scope):
            usage_accounting.reserve_attempt(AttemptRequest(
                model="test/model", provider="local", drive_root=tmp_path,
            ))
        entry = usage_accounting.last_root_accounting("rroot")
        assert entry is not None
        # Post-append sum on a fresh tree: the local-provider hold is $0.00.
        assert entry["accounted_usd"] == 0.0
        assert entry["root_limit_usd"] == 50.0

    def test_stash_tracks_reserve_settle_and_release_transitions(self, tmp_path):
        """G3-4: the stash follows every ledger transition in this process —
        reserve includes the fresh hold, settle replaces the hold with the real
        cost, release drops it — so the loop's 120s-trusted snapshot can never
        lag one call behind the fence."""
        from ouroboros import usage_accounting
        from ouroboros.usage_accounting import AttemptRequest, UsageScope, usage_scope

        def _stashed():
            entry = usage_accounting.last_root_accounting("root-transitions")
            assert entry is not None
            return entry["accounted_usd"]

        scope = UsageScope(
            drive_root=tmp_path, task_id="root-transitions", root_task_id="root-transitions",
            global_limit_usd=100.0, root_limit_usd=50.0,
        )
        with usage_scope(scope):
            first = usage_accounting.reserve_attempt(AttemptRequest(
                model="test/model", provider="openrouter",
                reservation_usd=2.5, drive_root=tmp_path,
            ))
            assert _stashed() == 2.5  # the in-flight hold itself
            usage_accounting.mark_dispatched(first)
            assert _stashed() == 2.5  # dispatch moves buckets, not the sum
            usage_accounting.settle_attempt(first, cost_usd=1.0, cost_final=True)
            assert _stashed() == 1.0  # settled cost replaced the hold
            second = usage_accounting.reserve_attempt(AttemptRequest(
                model="test/model", provider="openrouter",
                reservation_usd=3.0, drive_root=tmp_path,
            ))
            assert _stashed() == 4.0  # settled + the new hold
            usage_accounting.release_attempt(second, "not_dispatched")
            assert _stashed() == 1.0  # released hold no longer counts


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

    def test_tree_cost_is_the_deciding_value_and_is_labeled(self):
        """v6.91: the tree-accounted number decides the crossing and is labeled
        honestly (incl. in-flight holds); own cost rides as the diagnostic."""
        ctx = SimpleNamespace()
        note = task_pacing.build_cost_budget_note(
            ctx, start_remaining_usd=200.0, cost_ceiling_usd=97.0,
            task_cost=41.0, tree_cost_usd=50.0,
        )
        assert note is not None and "50% remaining" in note.text
        assert "in-flight holds" in note.text
        assert "own calls ~$41.00" in note.text
        assert note.checkpoint["spend_basis"] == "tree_accounted"
        # The deciding (tree) number and this task's own cost are BOTH recorded,
        # each under the name that means it — see the meaning-stability pin below.
        assert note.checkpoint["deciding_spend_usd"] == 50.0
        assert note.checkpoint["task_cost_usd"] == 41.0

    def test_checkpoint_key_meanings_are_stable_across_the_version_boundary(self):
        """`task_cost_usd` means THIS task's own cost, on every branch.

        v6.91 published the tree-accounted deciding number under that name, so
        the key silently changed axis: a log reader (and `loop.py`'s
        `_acceptance_loop_rails`, which publishes the same name meaning own cost
        and renders it as "$X spent this task") would have read tree spend as own
        spend with no way to tell. Both numbers are now always present under
        names that mean what they say."""
        cases = (
            # kind, task_cost (own), tree_cost_usd, expected basis
            ("cost_budget_milestone", 41.0, 50.0, task_pacing.SPEND_BASIS_TREE),
            ("cost_budget_milestone", 50.0, None, task_pacing.SPEND_BASIS_OWN_NO_TREE_CAP),
            ("cost_budget_wrapup", 41.0, 90.0, task_pacing.SPEND_BASIS_TREE),
            ("cost_budget_wrapup", 90.0, None, task_pacing.SPEND_BASIS_OWN_NO_TREE_CAP),
        )
        for kind, own, tree, expect_basis in cases:
            ctx = SimpleNamespace()
            if kind == "cost_budget_wrapup":
                ctx._cost_budget_milestones_seen = {"50%", "25%", "10%"}
            note = task_pacing.build_cost_budget_note(
                ctx, start_remaining_usd=200.0, cost_ceiling_usd=97.0,
                task_cost=own, tree_cost_usd=tree,
            )
            assert note is not None and note.checkpoint["checkpoint_kind"] == kind
            cp = note.checkpoint
            assert cp["spend_basis"] == expect_basis
            assert cp["task_cost_usd"] == own, (
                f"{kind}: task_cost_usd must stay this task's OWN cost"
            )
            # Present on EVERY branch, so no reader infers the axis from a
            # missing key.
            assert cp["deciding_spend_usd"] == (tree if tree is not None else own)

    def test_own_cost_alone_would_not_have_crossed(self):
        """The wave1/2 blindness pin: own $41 of a $97 ceiling fires nothing,
        tree $50 fires the 50% milestone."""
        silent_ctx = SimpleNamespace()
        assert task_pacing.build_cost_budget_note(
            silent_ctx, start_remaining_usd=200.0, cost_ceiling_usd=97.0,
            task_cost=41.0,
        ) is None

    def test_unknown_tree_cost_falls_back_to_own(self):
        """No root cap → own cost is complete, not a stand-in; the basis is
        still recorded so a reader never has to infer it from a missing key."""
        ctx = SimpleNamespace()
        note = task_pacing.build_cost_budget_note(
            ctx, start_remaining_usd=20.0, cost_ceiling_usd=10.0,
            task_cost=5.1, tree_cost_usd=None,
        )
        assert note is not None and "Spent this task: ~$5.10" in note.text
        assert "lower bound" not in note.text
        assert note.checkpoint["spend_basis"] == task_pacing.SPEND_BASIS_OWN_NO_TREE_CAP

    def test_unknown_tree_cost_under_a_root_cap_is_disclosed(self):
        """Under a tree cap the own-cost fallback is a LOWER BOUND — say so in
        the note and in the checkpoint instead of substituting silently."""
        ctx = SimpleNamespace()
        note = task_pacing.build_cost_budget_note(
            ctx, start_remaining_usd=20.0, cost_ceiling_usd=10.0,
            task_cost=5.1, tree_cost_usd=None, root_cap_usd=13.0,
        )
        assert note is not None
        assert "OWN calls only" in note.text and "lower bound" in note.text
        assert note.checkpoint["spend_basis"] == task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN

    def test_wrapup_note_discloses_the_own_cost_fallback(self):
        """Same disclosure on the ~80% wrap-up note (its own text path)."""
        ctx = SimpleNamespace()
        note = task_pacing.build_cost_budget_note(
            ctx, start_remaining_usd=20.0, cost_ceiling_usd=10.0,
            task_cost=8.5, tree_cost_usd=None, root_cap_usd=13.0,
        )
        # 15% remaining crosses the 25% milestone first; latch it and re-ask.
        assert note is not None and note.checkpoint["checkpoint_kind"] == "cost_budget_milestone"
        assert note.checkpoint["spend_basis"] == task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN
        fresh = SimpleNamespace()
        fresh._cost_budget_milestones_seen = {"50%", "25%", "10%"}
        wrapup = task_pacing.build_cost_budget_note(
            fresh, start_remaining_usd=20.0, cost_ceiling_usd=10.0,
            task_cost=8.5, tree_cost_usd=None, root_cap_usd=13.0,
        )
        assert wrapup is not None and wrapup.checkpoint["checkpoint_kind"] == "cost_budget_wrapup"
        assert "lower bound" in wrapup.text
        assert wrapup.checkpoint["spend_basis"] == task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN

    def test_resolve_deciding_spend_keeps_unknown_unknown(self):
        """Unknown spend stays None end-to-end — never a confident $0."""
        assert task_pacing.resolve_deciding_spend(
            tree_cost_usd=None, task_cost_usd=None, root_cap_usd=100.0,
        ) == (None, task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN)
        assert task_pacing.resolve_deciding_spend(
            tree_cost_usd=7.0, task_cost_usd=3.0, root_cap_usd=None,
        ) == (7.0, task_pacing.SPEND_BASIS_TREE)

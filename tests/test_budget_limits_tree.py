"""Tests for _check_budget_limits, continued from ``test_budget_limits``: the
v6.91 tree-fed deciding value, root-accounting telemetry and the latched
v6.56.0 cost milestones."""
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock

from ouroboros import task_pacing
from ouroboros.loop_budget import _check_budget_limits
from tests._budget_limits_helpers import EXPLICIT_50, _make_args


# --- v6.91 tree-fed deciding value ---

class TestTreeFedDecidingValue:
    """Under a root cap the deciding spend is the root subtree's KNOWN (settled)
    number from the reserve-time scope telemetry — own cost stays a diagnostic.
    These run an explicit ``cost_hard_stop_pct=50`` profile, the only ordinary-free
    source of an in-task ceiling besides a producer allowance.
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
            1900.0, EXPLICIT_50, root_cap_usd=100.0,
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
            usage_accounting._stash_root_accounting("root-tree-1", {"settled_usd": 98.5, "accounted_usd": 98.5}, 100.0)
            result = _check_budget_limits(**args)
        assert result is not None
        text, usage, _ = result
        assert usage.get("reason_code") == "budget_exhausted"

    def test_tree_spend_under_ceiling_does_not_stop(self, tmp_path):
        from ouroboros import usage_accounting

        ceiling = task_pacing.resolve_cost_ceiling(
            1900.0, EXPLICIT_50, root_cap_usd=100.0,
        )
        args = _make_args(
            budget_remaining_usd=1900.0,
            accumulated_usage={"cost": 41.0},
            cost_ceiling=ceiling,
            drive_logs=tmp_path,
        )
        with self._scoped("root-tree-2", 100.0):
            usage_accounting._stash_root_accounting("root-tree-2", {"settled_usd": 60.0, "accounted_usd": 60.0}, 100.0)
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
            10.0, EXPLICIT_50,
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
            1900.0, EXPLICIT_50, root_cap_usd=10.0,
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
            None, EXPLICIT_50, root_cap_usd=100.0,
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
            usage_accounting._stash_root_accounting("root-tree-noglobal", {"settled_usd": 98.5, "accounted_usd": 98.5}, 100.0)
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
                1900.0, EXPLICIT_50, root_cap_usd=100.0,
            ),
            drive_logs=tmp_path,
        )
        with self._scoped("root-tree-fresh", 100.0):
            usage_accounting._stash_root_accounting("root-tree-fresh", {"settled_usd": 5.0, "accounted_usd": 5.0}, 100.0)
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
            return {"settled_usd": 98.5, "accounted_usd": 98.5, "limit_usd": 100.0}

        monkeypatch.setattr(usage_accounting, "usage_projection", _fresh_projection)
        args = _make_args(
            budget_remaining_usd=1900.0,
            accumulated_usage={"cost": 41.0},
            cost_ceiling=task_pacing.resolve_cost_ceiling(
                1900.0, EXPLICIT_50, root_cap_usd=100.0,
            ),
            llm=llm,
            drive_logs=tmp_path,
        )
        with self._scoped("root-tree-stale", 100.0):
            # Stash a pre-block number, then age it past the bound.
            usage_accounting._stash_root_accounting("root-tree-stale", {"settled_usd": 40.0, "accounted_usd": 40.0}, 100.0)
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
        """G3-4 + #1487: the stash the loop trusts for 120s follows every transition,
        and it carries the in-flight hold as disclosed EXPOSURE, not known spend. The
        hold alone stops nothing (owner Q4-A); the moment the call settles, its price
        is known and the ceiling check stops on it — still with zero fresh ledger
        reads, because the settlement refreshed the stash itself."""
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
                1900.0, EXPLICIT_50, root_cap_usd=100.0,
            ),
            llm=llm,
            drive_logs=tmp_path,
        )
        scope = UsageScope(
            drive_root=tmp_path, task_id="root-inflight", root_task_id="root-inflight",
            global_limit_usd=1900.0, root_limit_usd=100.0,
        )
        with usage_scope(scope):
            reservation = usage_accounting.reserve_attempt(AttemptRequest(
                model="test/model", provider="openrouter",
                reservation_usd=99.5, drive_root=tmp_path,
            ))
            # The attempt is still in flight (never settled): its hold is visible
            # exposure, and known spend is still $0 — no stop on a worst case.
            entry = usage_accounting.last_root_accounting("root-inflight")
            assert entry is not None and entry["accounted_usd"] == 99.5 and entry["settled_usd"] == 0.0
            assert _check_budget_limits(**args) is None
            usage_accounting.mark_dispatched(reservation)
            usage_accounting.settle_attempt(reservation, cost_usd=98.0, cost_final=False)
            entry = usage_accounting.last_root_accounting("root-inflight")
            assert entry["settled_usd"] == 98.0 and entry["accounted_usd"] == 98.0  # over the $97 ceiling
            result = _check_budget_limits(**args)
        assert result is not None
        _text, usage, _ = result
        assert usage.get("reason_code") == "budget_exhausted"
        assert usage["cost_stop_spend_basis"] == task_pacing.SPEND_BASIS_TREE


class TestRootAccountingTelemetry:
    def test_stash_roundtrip_and_age(self):
        from ouroboros import usage_accounting

        usage_accounting._stash_root_accounting("root-t-1", {"settled_usd": 12.5, "accounted_usd": 12.5}, 100.0)
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
        # A bucket that carries no known figure stays unknown: exposure is never
        # relabelled as known spend (#1487).
        assert entry["settled_usd"] is None
        assert usage_accounting.last_root_accounting("root-t-2")["root_limit_usd"] == 25.0
        with pytest.raises(TypeError):
            usage_accounting._stash_root_accounting("root-t-2", 7.25, 25.0)

    def test_refresh_failure_returns_stale_stash_not_zero(self, tmp_path, monkeypatch):
        from ouroboros import usage_accounting

        usage_accounting._stash_root_accounting("root-t-3", {"settled_usd": 3.0, "accounted_usd": 3.0}, 10.0)

        def _boom(*a, **k):
            raise RuntimeError("ledger unavailable")

        monkeypatch.setattr(usage_accounting, "usage_projection", _boom)
        entry = usage_accounting.refresh_root_accounting(tmp_path, "root-t-3")
        assert entry is not None and entry["accounted_usd"] == 3.0

    def test_strict_read_never_answers_from_the_cache(self, tmp_path, monkeypatch):
        """Money readers (#1196): a snapshot cached a moment ago is not an
        observation of the ledger NOW. A failed strict read returns None even
        beside a fresh cache; a successful one returns the fresh numbers and
        refreshes the display cache; the display reader keeps its fallback."""
        from ouroboros import usage_accounting

        usage_accounting._stash_root_accounting("root-strict", {"settled_usd": 3.0, "accounted_usd": 3.0}, 10.0)
        assert usage_accounting.last_root_accounting("root-strict")["age_sec"] < 1.0

        def _boom(*a, **k):
            raise RuntimeError("ledger unavailable")

        monkeypatch.setattr(usage_accounting, "usage_projection", _boom)
        # Negative: the 0-age cache is exactly what a strict reader must NOT get.
        assert usage_accounting.refresh_root_accounting(tmp_path, "root-strict", strict=True) is None
        assert usage_accounting.refresh_root_accounting(tmp_path, "root-strict", max_age_sec=30.0,
                                                        strict=True) is None
        # The display reader still falls back to the last snapshot, never to $0.
        display = usage_accounting.refresh_root_accounting(tmp_path, "root-strict", max_age_sec=0.0)
        assert display is not None and display["accounted_usd"] == 3.0
        # Positive: a fresh successful read is the observation, and it refreshes the cache.
        monkeypatch.setattr(usage_accounting, "usage_projection",
                            lambda *a, **k: {"accounted_usd": 4.5, "limit_usd": 10.0})
        fresh = usage_accounting.refresh_root_accounting(tmp_path, "root-strict", strict=True)
        assert fresh["accounted_usd"] == 4.5 and fresh["age_sec"] < 1.0
        assert usage_accounting.last_root_accounting("root-strict")["accounted_usd"] == 4.5

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
            known.append(entry["settled_usd"])
            return entry["accounted_usd"]

        known = []

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
        # Known spend moves only on the settled price, never on a hold.
        assert known == [0.0, 0.0, 1.0, 1.0, 1.0]


# --- v6.56.0 cost axes: latched one-time crossings (task_pacing content) ---

class TestCostMilestones:
    def test_milestones_latch_once_and_sequence(self):
        ctx = SimpleNamespace()
        kw = dict(cost_ceiling_usd=10.0)
        # 50% remaining of the $10 ceiling crossed.
        note = task_pacing.build_cost_budget_note(ctx, task_cost=5.1, **kw)
        assert note is not None and "50% remaining" in note.text
        assert note.checkpoint["checkpoint_kind"] == "cost_budget_milestone"
        assert note.checkpoint["hard_stop"] is True
        assert note.checkpoint["budget_axis"] == "in_task_cost_ceiling"
        # Same spend again → latched, silent.
        assert task_pacing.build_cost_budget_note(ctx, task_cost=5.1, **kw) is None
        # 25% remaining crossed.
        note = task_pacing.build_cost_budget_note(ctx, task_cost=7.6, **kw)
        assert note is not None and "25% remaining" in note.text
        # ~80% spent → one-shot crossing fact.
        note = task_pacing.build_cost_budget_note(ctx, task_cost=8.1, **kw)
        assert note is not None and note.checkpoint["checkpoint_kind"] == "cost_budget_wrapup"
        assert "80% spent crossed" in note.text
        assert task_pacing.build_cost_budget_note(ctx, task_cost=8.2, **kw) is None
        # 10% remaining crossed (the ~80% fact already latched, no duplicate).
        note = task_pacing.build_cost_budget_note(ctx, task_cost=9.1, **kw)
        assert note is not None and "10% remaining" in note.text
        assert task_pacing.build_cost_budget_note(ctx, task_cost=9.9, **kw) is None

    def test_crossings_state_facts_without_strategy(self):
        """Owner 13 Oct: the one-time crossing fact stays, the advice goes."""
        ctx = SimpleNamespace()
        notes = [task_pacing.build_cost_budget_note(ctx, task_cost=cost, cost_ceiling_usd=10.0)
                 for cost in (5.1, 7.6, 8.1, 9.1)]
        assert all(note is not None for note in notes)
        joined = "\n".join(note.text for note in notes)
        for advice in ("planning context", "shortest path", "prefer", "converging", "Start ",
                       "speculative", "FINAL ANSWER", "revert"):
            assert advice not in joined, advice

    def test_jump_past_wrapup_with_milestone_suppresses_duplicate_wrapup(self):
        """A single jump deep past 80% spent fires the tightest milestone and
        latches the ~80% fact, so the next round does not double-note."""
        ctx = SimpleNamespace()
        kw = dict(cost_ceiling_usd=10.0)
        note = task_pacing.build_cost_budget_note(ctx, task_cost=9.5, **kw)
        assert note is not None and "10% remaining" in note.text
        assert task_pacing.build_cost_budget_note(ctx, task_cost=9.6, **kw) is None

    def test_no_tree_bound_axis_is_silent(self):
        ctx = SimpleNamespace()
        assert task_pacing.build_cost_budget_note(
            ctx, cost_ceiling_usd=None, task_cost=999.0,
        ) is None

    def test_without_a_ceiling_the_hard_tree_cap_is_the_bound(self):
        """No in-task stop (ordinary task): the tree's crossing is measured on its
        hard tree cap — never on the shared wallet's start snapshot (#1129)."""
        ctx = SimpleNamespace()
        note = task_pacing.build_cost_budget_note(
            ctx, cost_ceiling_usd=None, task_cost=1.0, tree_cost_usd=26.0, root_cap_usd=50.0)
        assert note is not None and "50% remaining" in note.text
        assert "hard tree cap" in note.text and "Remaining: ~$24.00 of ~$50.00" in note.text
        assert note.checkpoint["budget_axis"] == "hard_tree_cap"
        assert note.checkpoint["base_usd"] == 50.0 and note.checkpoint["hard_stop"] is True

    def test_tree_cost_is_the_deciding_value_and_is_labeled(self):
        """v6.91 + #1487: the tree's known number decides the crossing and is labeled
        honestly (open holds not counted); own cost rides as the diagnostic."""
        ctx = SimpleNamespace()
        note = task_pacing.build_cost_budget_note(
            ctx, cost_ceiling_usd=97.0, task_cost=41.0, tree_cost_usd=50.0,
        )
        assert note is not None and "50% remaining" in note.text
        assert "known spend" in note.text and "open holds not counted" in note.text
        assert "own calls ~$41.00" in note.text
        assert note.checkpoint["spend_basis"] == "tree_known"
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
                ctx, cost_ceiling_usd=97.0, task_cost=own, tree_cost_usd=tree,
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
            assert cp["budget_axis"] == "in_task_cost_ceiling"

    def test_own_cost_alone_would_not_have_crossed(self):
        """The wave1/2 blindness pin: own $41 of a $97 ceiling fires nothing,
        tree $50 fires the 50% milestone."""
        silent_ctx = SimpleNamespace()
        assert task_pacing.build_cost_budget_note(
            silent_ctx, cost_ceiling_usd=97.0, task_cost=41.0,
        ) is None

    def test_unknown_tree_cost_falls_back_to_own(self):
        """No root cap → own cost is complete, not a stand-in; the basis is
        still recorded so a reader never has to infer it from a missing key."""
        ctx = SimpleNamespace()
        note = task_pacing.build_cost_budget_note(
            ctx, cost_ceiling_usd=10.0, task_cost=5.1, tree_cost_usd=None,
        )
        assert note is not None and "Spent this task: ~$5.10" in note.text
        assert "lower bound" not in note.text
        assert note.checkpoint["spend_basis"] == task_pacing.SPEND_BASIS_OWN_NO_TREE_CAP

    def test_unknown_tree_cost_under_a_root_cap_is_disclosed(self):
        """Under a tree cap the own-cost fallback is a LOWER BOUND — say so in
        the note and in the checkpoint instead of substituting silently."""
        ctx = SimpleNamespace()
        note = task_pacing.build_cost_budget_note(
            ctx, cost_ceiling_usd=10.0, task_cost=5.1, tree_cost_usd=None, root_cap_usd=13.0,
        )
        assert note is not None
        assert "OWN calls only" in note.text and "lower bound" in note.text
        assert "$13.00 hard tree cap" in note.text
        assert note.checkpoint["spend_basis"] == task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN

    def test_wrapup_note_discloses_the_own_cost_fallback(self):
        """Same disclosure on the ~80% spent note (its own text path)."""
        ctx = SimpleNamespace()
        kw = dict(cost_ceiling_usd=10.0, task_cost=8.5, tree_cost_usd=None, root_cap_usd=13.0)
        note = task_pacing.build_cost_budget_note(ctx, **kw)
        # 15% remaining crosses the 25% milestone first; latch it and re-ask.
        assert note is not None and note.checkpoint["checkpoint_kind"] == "cost_budget_milestone"
        assert note.checkpoint["spend_basis"] == task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN
        fresh = SimpleNamespace()
        fresh._cost_budget_milestones_seen = {"50%", "25%", "10%"}
        wrapup = task_pacing.build_cost_budget_note(fresh, **kw)
        assert wrapup is not None and wrapup.checkpoint["checkpoint_kind"] == "cost_budget_wrapup"
        assert "lower bound" in wrapup.text
        assert wrapup.checkpoint["spend_basis"] == task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN


# --- #1129: the shared wallet is its own axis, moved by every task's spending ---

class TestWalletAxis:
    @staticmethod
    def _wallet(remaining, limit=100.0):
        return {"status": "known", "remaining_usd": remaining, "limit_usd": limit,
                "observed_at": "2026-10-13T01:00:00Z"}

    def test_neighbour_spending_crosses_the_wallet_while_this_task_spends_nothing(self):
        """The 2026-09-18 shape: a root got "50% remaining" three minutes before the
        wallet stopped it. The wallet axis reads the wallet's CURRENT headroom, so a
        neighbour draining the pool crosses it with this task's own cost unchanged."""
        ctx = SimpleNamespace()
        kw = dict(baseline_usd=40.0, task_cost=0.5)
        assert task_pacing.build_wallet_budget_note(ctx, wallet=self._wallet(39.5), **kw) is None
        # Other tasks spent $20 of the shared pool; this task's own cost is still $0.50.
        note = task_pacing.build_wallet_budget_note(ctx, wallet=self._wallet(19.5), **kw)
        assert note is not None and "shared wallet 50% remaining crossed" in note.text
        assert "other tasks' spending moves it too" in note.text
        assert "own recorded cost: ~$0.50" in note.text
        cp = note.checkpoint
        assert cp["budget_axis"] == "shared_wallet" and cp["checkpoint_kind"] == "cost_budget_milestone"
        assert cp["wallet_remaining_known_usd"] == 19.5 and cp["wallet_baseline_usd"] == 40.0
        assert cp["task_cost_usd"] == 0.5 and "deciding_spend_usd" not in cp
        # Latched: the same headroom stays silent; the next threshold and ~80% fire once each.
        assert task_pacing.build_wallet_budget_note(ctx, wallet=self._wallet(19.0), **kw) is None
        assert "25% remaining" in task_pacing.build_wallet_budget_note(ctx, wallet=self._wallet(9.0), **kw).text
        late = task_pacing.build_wallet_budget_note(ctx, wallet=self._wallet(7.5), **kw)
        assert late is not None and late.checkpoint["checkpoint_kind"] == "cost_budget_wrapup"
        assert "shared wallet 80% spent crossed" in late.text
        assert task_pacing.build_wallet_budget_note(ctx, wallet=self._wallet(7.4), **kw) is None

    def test_own_spend_never_moves_the_wallet_axis(self):
        """Negative witness for the old aggregate: tree/own spend past the start
        snapshot does not cross the wallet while the wallet itself is untouched."""
        ctx = SimpleNamespace()
        assert task_pacing.build_wallet_budget_note(
            ctx, baseline_usd=10.0, wallet=self._wallet(10.0), task_cost=25.0) is None
        # and the tree axis no longer measures against the wallet snapshot at all
        assert task_pacing.build_cost_budget_note(ctx, cost_ceiling_usd=None, task_cost=25.0) is None

    def test_unknown_unlimited_or_missing_baseline_keeps_the_axis_silent(self):
        ctx = SimpleNamespace()
        for wallet in ({"status": "unknown", "reason": "ledger integrity degraded"},
                       {"status": "no_global_limit"}, None):
            assert task_pacing.build_wallet_budget_note(ctx, baseline_usd=40.0, wallet=wallet) is None
        for baseline in (None, 0.0, -1.0):
            assert task_pacing.build_wallet_budget_note(ctx, baseline_usd=baseline, wallet=self._wallet(1.0)) is None
        assert not getattr(ctx, "_wallet_budget_milestones_seen", set())

    def test_axes_latch_independently(self):
        ctx = SimpleNamespace()
        tree = task_pacing.build_cost_budget_note(ctx, cost_ceiling_usd=10.0, task_cost=5.1)
        wallet = task_pacing.build_wallet_budget_note(ctx, baseline_usd=40.0, wallet=self._wallet(19.0))
        assert tree is not None and wallet is not None
        assert ctx._cost_budget_milestones_seen == {"50%"} and ctx._wallet_budget_milestones_seen == {"50%"}

    def test_shared_wallet_observation_types_known_unlimited_and_unknown(self, monkeypatch):
        from ouroboros import usage_accounting
        from ouroboros.usage_accounting import UsageScope, usage_scope

        scope = UsageScope(drive_root=None, task_id="t-w", root_task_id="t-w", global_limit_usd=100.0)
        assert task_pacing.shared_wallet_observation()["reason"] == "no usage scope"
        projections = iter([
            {"limit_usd": 100.0, "remaining_known_usd": 61.5, "integrity_degraded": False},
            {"integrity_degraded": False},
            {"limit_usd": 100.0, "remaining_known_usd": 61.5, "integrity_degraded": True},
        ])
        monkeypatch.setattr(usage_accounting, "usage_projection", lambda *_a, **_k: next(projections))
        with usage_scope(scope):
            known = task_pacing.shared_wallet_observation()
            assert (known["status"], known["remaining_usd"], known["limit_usd"]) == ("known", 61.5, 100.0)
            assert task_pacing.shared_wallet_observation()["status"] == "no_global_limit"
            degraded = task_pacing.shared_wallet_observation()
            assert degraded["status"] == "unknown" and "integrity" in degraded["reason"]

            def _unreadable(*_a, **_k):
                raise OSError("ledger unreadable")

            monkeypatch.setattr(usage_accounting, "usage_projection", _unreadable)
            unreadable = task_pacing.shared_wallet_observation()
            assert (unreadable["status"], unreadable["reason"]) == ("unknown", "ledger unreadable")
            assert "remaining_usd" not in unreadable  # unknown is never a $0 headroom

    def test_one_money_observation_per_round(self, monkeypatch):
        """The round's notices and its facts line share ONE wallet/tree read (#1129:
        no new per-round ledger read)."""
        reads = {"wallet": 0, "tree": 0}

        def wallet():
            reads["wallet"] += 1
            return {"status": "known", "remaining_usd": 5.0, "limit_usd": 10.0}

        def tree():
            reads["tree"] += 1
            return {"settled_usd": 1.0, "accounted_usd": 1.5, "root_limit_usd": 50.0, "age_sec": 3.0,
                    "reservations": [{"id": "r"}]}

        monkeypatch.setattr(task_pacing, "shared_wallet_observation", wallet)
        usage: dict = {}
        first = task_pacing.round_money_observation(usage, 4, tree_reader=tree)
        again = task_pacing.round_money_observation(usage, 4, tree_reader=tree)
        assert first is again and reads == {"wallet": 1, "tree": 1}
        assert first["tree"] == {"settled_usd": 1.0, "accounted_usd": 1.5, "root_limit_usd": 50.0, "age_sec": 3.0}
        task_pacing.round_money_observation(usage, 5, tree_reader=lambda: (_ for _ in ()).throw(OSError("x")))
        assert reads["wallet"] == 2 and usage["_round_money_observation"]["tree"] is None

    def test_resolve_deciding_spend_keeps_unknown_unknown(self):
        """Unknown spend stays None end-to-end — never a confident $0."""
        assert task_pacing.resolve_deciding_spend(
            tree_cost_usd=None, task_cost_usd=None, root_cap_usd=100.0,
        ) == (None, task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN)
        assert task_pacing.resolve_deciding_spend(
            tree_cost_usd=7.0, task_cost_usd=3.0, root_cap_usd=None,
        ) == (7.0, task_pacing.SPEND_BASIS_TREE)


class TestResumedWakeCeilingEnforcement:
    """Initial and resumed own-clock wakes (and their members) pause with ``scope='root'`` at
    their cumulative ceiling even without a per-task hard cap; resumed wakes refuse own-cost
    fallback with ``cause='tree_accounting_unavailable'`` on unreadable or degraded tree reads."""

    def _seed_wake(self, tmp_path, *, root_id="own-wake", initial=4.0, renewed=None, additional=5.0):
        from ouroboros.consciousness_allowance import RESUMED_ALLOWANCE_KEY
        from ouroboros.consciousness_wake import wake_task_metadata
        from ouroboros.task_results import write_task_result

        meta = wake_task_metadata("act", "test", root_cost_ceiling_usd=initial,
                                  root_cost_ceiling_source="allowance")
        extra = ({RESUMED_ALLOWANCE_KEY: {"additional_usd": additional, "ceiling_usd": renewed,
                                          "source": "allowance"}} if renewed is not None else {})
        # Direct wakes may carry no task_contract; both root and members must still inherit the right policy.
        write_task_result(tmp_path, root_id, "running", root_task_id=root_id, metadata=meta, **extra)
        return meta

    def _check_args(self, tmp_path, tool_ctx, ceiling, *, task_id, own_cost):
        args = _make_args(
            budget_remaining_usd=100.0,
            accumulated_usage={"cost": own_cost},
            cost_ceiling=ceiling,
            drive_logs=tmp_path,
            task_id=task_id,
        )
        args["ctx"].tools = SimpleNamespace(_ctx=tool_ctx)
        args["ctx"].drive_root = tmp_path
        return args

    @pytest.mark.parametrize("resumed", [False, True])
    @pytest.mark.parametrize("role", ["root", "member"])
    def test_own_wake_and_member_pause_with_root_scope_without_hard_cap(self, tmp_path, monkeypatch, resumed, role):
        from ouroboros import budget_pause, usage_accounting
        from ouroboros.usage_accounting import UsageScope, usage_scope

        root_id = f"wake-{'resumed' if resumed else 'init'}-{role}"
        task_id = root_id if role == "root" else f"{root_id}-helper"
        expected_ceiling = 9.0 if resumed else 4.0
        meta = self._seed_wake(tmp_path, root_id=root_id, initial=4.0, renewed=9.0 if resumed else None)
        tree_state = {"settled_usd": expected_ceiling - 1.0, "accounted_usd": expected_ceiling - 1.0,
                      "limit_usd": None, "integrity_degraded": False}
        monkeypatch.setattr(usage_accounting, "usage_projection", lambda *_a, **_k: dict(tree_state))
        pauses = []
        monkeypatch.setattr(budget_pause, "request_pause", lambda _ctx, **kw: pauses.append(kw))

        tool_ctx = SimpleNamespace(task_id=task_id, root_task_id=root_id,
                                   drive_root=tmp_path, budget_drive_root=tmp_path,
                                   task_metadata=meta if role == "root" else {})
        scope = UsageScope(drive_root=tmp_path, task_id=task_id, root_task_id=root_id,
                           root_limit_usd=None, root_cost_ceiling_usd=4.0)
        with usage_scope(scope):
            ceiling = task_pacing.resolve_task_cost_ceiling(tool_ctx, 100.0)
            policy = task_pacing.cost_stop_policy(tool_ctx)
            assert ceiling.state == task_pacing.COST_CEILING_ACTIVE and ceiling.ceiling_usd == expected_ceiling
            assert ceiling.root_cap_usd is None
            assert ceiling.basis == ("producer_allowance" if role == "root" else "root_resolved_ceiling")
            assert policy["own_wake_allowance"] is True and policy["resumed_wake_allowance"] is resumed
            assert _check_budget_limits(**self._check_args(tmp_path, tool_ctx, ceiling,
                                                           task_id=task_id, own_cost=0.5)) is None
            tree_state["settled_usd"] = tree_state["accounted_usd"] = expected_ceiling
            if not resumed:
                usage_accounting._stash_root_accounting(root_id, tree_state, None)
            stopped = _check_budget_limits(**self._check_args(tmp_path, tool_ctx, ceiling,
                                                              task_id=task_id, own_cost=0.5))
        assert stopped is not None and stopped[1]["reason_code"] == "budget_exhausted"
        assert stopped[1]["cost_stop_spend_basis"] == task_pacing.SPEND_BASIS_TREE
        assert pauses and pauses[-1]["rail"] == budget_pause.RAIL_GRACEFUL_CEILING
        assert pauses[-1]["scope"] == "root"

    def test_other_producer_without_hard_cap_keeps_global_scope_and_exact_gte(self, tmp_path, monkeypatch):
        from ouroboros import budget_pause, usage_accounting
        from ouroboros.usage_accounting import UsageScope, usage_scope

        tree_state = {"settled_usd": 4.0 - 1e-10, "accounted_usd": 4.0 - 1e-10,
                      "limit_usd": None, "integrity_degraded": False}
        monkeypatch.setattr(usage_accounting, "usage_projection", lambda *_a, **_k: dict(tree_state))
        pauses = []
        monkeypatch.setattr(budget_pause, "request_pause", lambda _ctx, **kw: pauses.append(kw))
        tool_ctx = SimpleNamespace(task_id="other-prod", root_task_id="other-prod",
                                   drive_root=tmp_path, budget_drive_root=tmp_path, task_metadata={})
        scope = UsageScope(drive_root=tmp_path, task_id="other-prod", root_task_id="other-prod",
                           root_limit_usd=None, root_cost_ceiling_usd=4.0)
        with usage_scope(scope):
            ceiling = task_pacing.resolve_task_cost_ceiling(tool_ctx, 100.0)
            assert _check_budget_limits(**self._check_args(tmp_path, tool_ctx, ceiling,
                                                           task_id="other-prod", own_cost=0.5)) is None
            tree_state["settled_usd"] = tree_state["accounted_usd"] = 4.0
            usage_accounting._stash_root_accounting("other-prod", tree_state, None)
            stopped = _check_budget_limits(**self._check_args(tmp_path, tool_ctx, ceiling,
                                                              task_id="other-prod", own_cost=0.5))
        assert stopped is not None and pauses[-1]["scope"] == "global"

    @pytest.mark.parametrize("role", ["root", "member"])
    def test_resumed_wake_and_member_refuse_own_cost_fallback_on_unreadable_or_degraded_tree(
        self, tmp_path, monkeypatch, role,
    ):
        from ouroboros import budget_pause, loop_budget, usage_accounting
        from ouroboros.usage_accounting import BillingAuthorityUnavailable, UsageScope, usage_scope

        meta = self._seed_wake(tmp_path, root_id="wake-resumed-2", initial=4.0, renewed=9.0)
        task_id = "wake-resumed-2" if role == "root" else "wake-child-2"
        tool_ctx = SimpleNamespace(task_id=task_id, root_task_id="wake-resumed-2",
                                   drive_root=tmp_path, budget_drive_root=tmp_path,
                                   task_metadata=meta if role == "root" else {})
        scope = UsageScope(drive_root=tmp_path, task_id=task_id, root_task_id="wake-resumed-2",
                           root_limit_usd=None, root_cost_ceiling_usd=4.0)

        # Even with a fresh-looking stash and tiny own cost, a failed strict read must NOT admit spend.
        usage_accounting._stash_root_accounting("wake-resumed-2", {"settled_usd": 1.0, "accounted_usd": 1.0}, None)
        monkeypatch.setattr(usage_accounting, "usage_projection",
                            lambda *_a, **_k: (_ for _ in ()).throw(OSError("temporarily unreadable")))
        with usage_scope(scope):
            ceiling = task_pacing.resolve_task_cost_ceiling(tool_ctx, 100.0)
            args = self._check_args(tmp_path, tool_ctx, ceiling, task_id=task_id, own_cost=0.5)
            with pytest.raises(BillingAuthorityUnavailable) as raised:
                _check_budget_limits(**args)
        assert raised.value.kind == "unreadable"
        assert raised.value.cause == "tree_accounting_unavailable"
        assert "unreadable" in str(raised.value)

        # Existing authority handler parks on member RAIL_BILLING_AUTHORITY (scope="task"), not money exhaustion.
        pause_calls = []
        monkeypatch.setattr(budget_pause, "request_pause",
                            lambda ctx, **kw: pause_calls.append(kw) or (_ for _ in ()).throw(
                                budget_pause.BudgetPauseRequested({"rail": kw["rail"], "cause": kw.get("cause")})))
        exit_ctx = loop_budget._LoopExitContext(
            tools=args["ctx"].tools, drive_root=tmp_path, task_id=task_id,
            event_queue=None, drive_logs=tmp_path, accumulated_usage=args["ctx"].accumulated_usage, llm_trace={},
        )
        with pytest.raises(budget_pause.BudgetPauseRequested):
            loop_budget._handle_authority_unavailable(raised.value, exit_ctx, limit_ctx=args["ctx"])
        assert pause_calls == [{"rail": budget_pause.RAIL_BILLING_AUTHORITY, "scope": "task",
                                "reason_text": str(raised.value), "cause": "tree_accounting_unavailable"}]

        # Degraded integrity uses the same typed cause, carrying integrity_degraded in the detail message.
        monkeypatch.setattr(usage_accounting, "usage_projection",
                            lambda *_a, **_k: {"settled_usd": 1.0, "accounted_usd": 1.0,
                                               "limit_usd": None, "integrity_degraded": True})
        with usage_scope(scope):
            with pytest.raises(BillingAuthorityUnavailable) as degraded:
                _check_budget_limits(**self._check_args(tmp_path, tool_ctx, ceiling,
                                                        task_id=task_id, own_cost=0.5))
        assert degraded.value.kind == "unreadable"
        assert degraded.value.cause == "tree_accounting_unavailable"
        assert "integrity_degraded" in str(degraded.value)

"""The plan panel's health snapshot (``plan_review_runtime.plan_panel_health_snapshot``),
read before an ask dispatches: only positive structural evidence (a dated future window
exhaustion, a typed dead pool) keeps a seat out, health is read per PINNED profile, and an
unknown or transient answer fails open (every named participant is sent)."""

from __future__ import annotations

from tests.test_plan_task_tool import _slots


# --------------------------------------------- panel health is PROFILE-scoped


def _profile_slots(*specs):
    """``specs`` = (slot_id, session_target, profile) → agent_session ``ReviewSlot``s."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot

    return [
        ReviewSlot(slot_id=sid, model="delegated", effort="high",
                   role_hint="plan reviewer", route=ReviewRouteKind.AGENT_SESSION,
                   session_target=target, session_profile=profile)
        for sid, target, profile in specs
    ]


def _patch_snapshot_health(monkeypatch, answer):
    """Run the REAL snapshot against a fake daemon; ``answer(route_id, model, pin)``
    returns ``(reason, reset_at)``. Records every ask so memo scoping is provable."""
    import ouroboros.claudexor_daemon as cd
    import ouroboros.subagents as sa

    asked: list[tuple[str, str, str]] = []

    class _Gateway:
        def close(self):
            pass

    def _health(gateway, route_id, shape, *, route_model="", pinned_profile=""):
        asked.append((route_id, route_model, pinned_profile))
        return answer(route_id, route_model, pinned_profile)

    monkeypatch.setattr(cd, "owned_daemon_provisioned", lambda: True)
    monkeypatch.setattr(cd, "ensure_owned_gateway", _Gateway)
    monkeypatch.setattr(sa, "route_health", _health)
    return asked


_SPENT = ("subscription_window_exhausted", "2030-01-01T00:00:00+00:00")
_HEALTHY = ("", "")


def test_snapshot_skips_a_slot_whose_pinned_profile_is_spent(monkeypatch):
    """A slot pinning a spent account is skipped even though the route AGGREGATE
    (the unpinned answer a sibling account still vouches for) reads healthy: the
    pin rides into the health read exactly as it does at dispatch."""
    from ouroboros.tools.plan_review_runtime import plan_panel_health_snapshot

    asked = _patch_snapshot_health(
        monkeypatch, lambda rid, model, pin: _SPENT if pin == "spent-acct" else _HEALTHY)
    slots = _profile_slots(("s1", "codex=gpt-5.6-sol", "spent-acct"))
    assert plan_panel_health_snapshot(slots) == {
        "s1": {"failure_code": "subscription_window_exhausted",
               "reset_at": "2030-01-01T00:00:00+00:00"}}
    assert asked == [("codex", "gpt-5.6-sol", "spent-acct")]


def test_same_route_different_profiles_do_not_share_one_health_verdict(monkeypatch):
    """The memo is keyed by the SUBJECT, pin included: two rows on the same
    harness+model but different accounts are asked separately, so exactly the spent
    one is skipped and the healthy one still dispatches."""
    from ouroboros.tools.plan_review_runtime import (
        plan_health_skip_rows, plan_panel_health_snapshot,
    )

    asked = _patch_snapshot_health(
        monkeypatch, lambda rid, model, pin: _SPENT if pin == "spent-acct" else _HEALTHY)
    slots = _profile_slots(("s1", "codex=gpt-5.6-sol", "spent-acct"),
                           ("s2", "codex=gpt-5.6-sol", "live-acct"))
    evidence = plan_panel_health_snapshot(slots)
    assert set(evidence) == {"s1"}
    assert asked == [("codex", "gpt-5.6-sol", "spent-acct"),
                     ("codex", "gpt-5.6-sol", "live-acct")]
    live, rows = plan_health_skip_rows(slots, evidence)
    assert [s.slot_id for s in live] == ["s2"]
    assert len(rows) == 1 and rows[0]["slot_id"] == "s1"
    assert "'spent-acct'" in rows[0]["error"] and rows[0]["cost"] == 0.0
    # And the memo still WORKS: a repeated identical subject is asked only once.
    asked.clear()
    plan_panel_health_snapshot(slots + _profile_slots(
        ("s3", "codex=gpt-5.6-sol", "spent-acct")))
    assert asked == [("codex", "gpt-5.6-sol", "spent-acct"),
                     ("codex", "gpt-5.6-sol", "live-acct")]


def test_unpinned_slots_keep_the_route_wide_answer(monkeypatch):
    """No regression for rows that pin nothing: the ask carries an empty profile
    (rotation stays Claudexor's business) and the skip row says so plainly."""
    from ouroboros.tools.plan_review_runtime import (
        plan_health_skip_rows, plan_panel_health_snapshot,
    )

    asked = _patch_snapshot_health(monkeypatch, lambda rid, model, pin: _SPENT)
    slots = _profile_slots(("s1", "codex=gpt-5.6-sol", ""))
    evidence = plan_panel_health_snapshot(slots)
    assert set(evidence) == {"s1"} and asked == [("codex", "gpt-5.6-sol", "")]
    _live, rows = plan_health_skip_rows(slots, evidence)
    assert "route-wide" in rows[0]["error"] and "pins no credential profile" in rows[0]["error"]


def test_a_seat_the_fan_out_drops_on_a_spent_route_asks_the_subscription_census_for_a_read(monkeypatch, tmp_path):
    """The seat the fan-out drops asks for a read; the health snapshot itself only reads, and a healthy seat that
    dispatches asks nothing."""
    from ouroboros import config
    from ouroboros import subscription_census as census
    from ouroboros.tools.plan_review_runtime import plan_health_skip_rows, plan_panel_health_snapshot

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    _patch_snapshot_health(monkeypatch, lambda rid, model, pin: _SPENT if pin == "spent-acct" else _HEALTHY)
    healthy = _profile_slots(("s2", "codex=gpt-5.6-sol", "live-acct"))
    assert plan_health_skip_rows(healthy, plan_panel_health_snapshot(healthy)) == (healthy, [])
    spent = _profile_slots(("s1", "codex=gpt-5.6-sol", "spent-acct"))
    evidence = plan_panel_health_snapshot(spent)
    assert census.request_mtime() is None  # the snapshot is a passive reader
    _live, rows = plan_health_skip_rows(spent, evidence)
    assert len(rows) == 1 and census.request_mtime() is not None


def test_transient_and_unknown_health_still_fail_open_for_a_pinned_slot(monkeypatch):
    """Fail-open is unchanged by the narrowing: an undated exhaustion, a transient
    daemon state and an unknown reason on a PINNED row all dispatch (no skip row)."""
    from ouroboros.tools.plan_review_runtime import plan_panel_health_snapshot

    for reason, reset in (("subscription_window_exhausted", ""),
                          ("daemon_recovery_only", ""),
                          ("route_status_disabled", ""),
                          ("", "2001-01-01T00:00:00Z")):
        _patch_snapshot_health(monkeypatch, lambda rid, model, pin, r=reason, t=reset: (r, t))
        assert plan_panel_health_snapshot(
            _profile_slots(("s1", "codex=gpt-5.6-sol", "spent-acct"))) == {}, reason


def test_structural_skip_predicate_requires_positive_evidence():
    """Unknown/undated/stale/transient states DISPATCH; only a dated future window
    exhaustion or a typed dead-pool code is skip evidence (roast pts 4/9)."""
    from ouroboros.tools.plan_review_runtime import _structural_skip_code

    assert _structural_skip_code("", "2030-01-01T00:00:00Z") == "subscription_window_exhausted"
    assert _structural_skip_code("subscription_window_exhausted", "") == ""  # undated
    assert _structural_skip_code("credential_pool_exhausted", "") == "credential_pool_exhausted"
    assert _structural_skip_code("", "2001-01-01T00:00:00Z") == ""  # stale reset
    assert _structural_skip_code("", "not-a-time") == "" and _structural_skip_code("daemon_recovery_only", "") == ""
    assert _structural_skip_code("route_status_disabled", "") == ""  # not window evidence


def test_snapshot_transient_daemon_death_reads_unknown_never_structural(monkeypatch):
    """A ClaudexorUnavailable during the snapshot (daemon_recovery_only, dead socket)
    yields None (unknown, fail-open) — never skip rows."""
    import ouroboros.claudexor_daemon as cd
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools.plan_review_runtime import plan_panel_health_snapshot

    monkeypatch.setattr(cd, "owned_daemon_provisioned", lambda: True)

    def _dying():
        raise ClaudexorUnavailable("daemon_recovery_only", "daemon is serving recovery only")

    monkeypatch.setattr(cd, "ensure_owned_gateway", _dying)
    assert plan_panel_health_snapshot(_slots(("s1", "m/a"), ("s2", "m/b", "session"))) is None
    # An api_chat-only panel has no route health source: the snapshot trivially ran.
    assert plan_panel_health_snapshot(_slots(("s1", "m/a"))) == {}

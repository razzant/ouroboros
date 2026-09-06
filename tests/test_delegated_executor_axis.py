"""The executor axis: the harness setting, the rule table, and the dispatch behind them.

Split verbatim out of ``tests/test_delegated_subagent_transport.py`` by theme. This
module owns the narrow ``OUROBOROS_SUBAGENT_HARNESS`` key and its route parsing, the
auto/harness/native rule table, the nanny verb allowlists, and the resolution rows the
real dispatch entry point produces — including the canonical event and the
parent-facing beacon a spent subscription window leaves behind.
"""

from __future__ import annotations

import json
import pytest
from ouroboros import subagents
from ouroboros.gateways import claudexor as cx
from ouroboros.loop_llm_call import SUBSCRIPTION_WINDOW_EXHAUSTED
from ouroboros.provider_models import MODEL_SETTING_KEYS
from ouroboros.tool_capabilities import (
    ACTING_SUBAGENT_TOOL_NAMES,
    LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
)

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _HealthStub,
    _dispatch,
    _owned_gateway_uses_each_test_transport,
)


NANNY_TOOLS = {"delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer"}


def test_subagent_harness_key_stays_out_of_the_model_key_sweep():
    # A session-only route is not an API model identity: leaking it into
    # MODEL_SETTING_KEYS would poison credential planning, pricing and provenance.
    assert "OUROBOROS_SUBAGENT_HARNESS" not in MODEL_SETTING_KEYS


@pytest.mark.parametrize("raw,expected", [
    ("", None),
    ("codex", subagents.DelegationRoute("codex", "", "")),
    ("codex=gpt-5.4-mini", subagents.DelegationRoute("codex", "gpt-5.4-mini", "")),
    ("codex=gpt-5.4-mini:low", subagents.DelegationRoute("codex", "gpt-5.4-mini", "low")),
    # The documented grammar is harness[=model][:effort] — the effort bracket is
    # not tied to the model one. Splitting on `=` first made the whole string the
    # route id, which then failed at dispatch as an unknown route.
    ("claude:high", subagents.DelegationRoute("claude", "", "high")),
    # A typo with an empty head is "no route", not a route named "=opus".
    ("=opus", None),
    ("=model:high", None),
])
def test_route_parsing_is_opaque(raw, expected):
    assert subagents.parse_subagent_harness(raw) == expected


def test_an_unparseable_configured_route_is_disclosed_not_silent(monkeypatch, caplog):
    """A non-empty OUROBOROS_SUBAGENT_HARNESS that parses to nothing ("=opus") used
    to be silently identical to "never configured" — ALL delegation moved onto
    metered API children with no trace anywhere the operator looks."""
    import logging

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "=opus")
    with caplog.at_level(logging.WARNING, logger="ouroboros.subagents"):
        assert subagents.get_subagent_harness() is None
    assert any("unparseable" in r.message for r in caplog.records)

    # The two legitimate "no route" spellings stay silent.
    for quiet in ("", "off"):
        caplog.clear()
        monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", quiet)
        with caplog.at_level(logging.WARNING, logger="ouroboros.subagents"):
            assert subagents.get_subagent_harness() is None
        assert not caplog.records


def test_an_explicit_off_is_a_decision_an_empty_value_is_not(monkeypatch):
    """Both spellings mean "no delegated route"; they differ in owner intent.

    Settings' Subagents section turns delegation on by itself once a subscription
    is connected, and it may only do that over a value nobody decided. Without a
    distinguishable "off" the owner's own Off saved as empty and came back On on
    the next load — an un-saveable choice. Runtime behaviour is identical.
    """
    assert subagents.parse_subagent_harness("off") is None
    assert subagents.parse_subagent_harness("OFF") is None
    assert subagents.parse_subagent_harness("  off  ") is None
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "off")
    assert subagents.get_subagent_harness() is None
    assert subagents.resolve_subagent_executor("auto", route=None).executor == "native"


def test_get_subagent_harness_reads_the_env_key(monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=some-model:high")
    route = subagents.get_subagent_harness()
    assert route is not None and route.route_id == "some-route"
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "")
    assert subagents.get_subagent_harness() is None


ROUTE = subagents.DelegationRoute("some-route", "m", "low")


def test_rule_auto_without_harness_runs_native():
    res = subagents.resolve_subagent_executor("auto", route=None)
    assert (res.executor, res.reason) == ("native", "harness_not_configured")


def test_rule_auto_with_healthy_harness_delegates():
    res = subagents.resolve_subagent_executor("auto", route=ROUTE)
    assert (res.executor, res.reason) == ("harness", "harness_ready")


def test_rule_auto_with_every_profile_spent_falls_back_to_the_api_loudly():
    """Owner decision D28. It used to dispatch the child as a NANNY anyway, whose very
    first `delegate_start` was then refused with this SAME fact (executed and pinned
    below) — a spent dispatch, and the child left to improvise a fallback in prose.
    `auto` now falls back to the metered API at the one point that still costs nothing,
    typed, with the reset instant riding along so waiting stays a visible option."""
    res = subagents.resolve_subagent_executor("auto", route=ROUTE, reset_at="2030-01-01T00:00:00Z")
    assert res.executor == "native", "auto must not be dispatched onto a spent substrate"
    assert res.reason == SUBSCRIPTION_WINDOW_EXHAUSTED
    assert res.reset_at == "2030-01-01T00:00:00Z"
    assert not res.blocked, "never a permanent block while metered keys exist"


def test_rule_auto_with_unavailable_harness_falls_native_with_a_visible_marker():
    res = subagents.resolve_subagent_executor("auto", route=ROUTE, unavailable_reason="daemon_unreachable")
    assert (res.executor, res.reason) == ("native", "daemon_unreachable")


@pytest.mark.parametrize("kwargs,reason", [
    ({"route": None}, "harness_not_configured"),
    ({"route": ROUTE, "unavailable_reason": "daemon_unreachable"}, "daemon_unreachable"),
    ({"route": ROUTE, "reset_at": "2030-01-01T00:00:00Z"}, SUBSCRIPTION_WINDOW_EXHAUSTED),
])
def test_rule_explicit_harness_blocks_instead_of_spending_api_money(kwargs, reason):
    res = subagents.resolve_subagent_executor("harness", **kwargs)
    assert res.blocked and res.reason == reason


def test_rule_native_is_native_whatever_the_state():
    res = subagents.resolve_subagent_executor("native", route=ROUTE, unavailable_reason="x")
    assert (res.executor, res.reason) == ("native", "requested_native")


def test_unknown_executor_is_rejected():
    with pytest.raises(ValueError):
        subagents.resolve_subagent_executor("magic")


def test_both_child_allowlists_can_see_the_nanny_verbs():
    assert NANNY_TOOLS <= LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert NANNY_TOOLS <= ACTING_SUBAGENT_TOOL_NAMES


def test_there_is_no_hurry_verb():
    from ouroboros.tools import delegate

    names = {entry.name for entry in delegate.get_tools()}
    assert names == NANNY_TOOLS


def test_delegate_start_refuses_typed_when_no_route_is_configured(tmp_path, monkeypatch):
    from ouroboros.tools.delegate import _delegate_start
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    payload = json.loads(_delegate_start(ctx, "do a thing"))
    assert payload["status"] == "refused"
    assert payload["reason"] == "subagent_selection_required"


def test_one_exhausted_credential_profile_does_not_take_the_harness_offline():
    """Defect D (D28): the readiness predicate reported a blocker as soon as ANY window
    of the harness was spent, so one exhausted account took the WHOLE harness offline
    while its siblings were live — an outage invented out of a healthy substrate, and
    the `harness` executor is a PIN, so the caller was refused rather than re-routed.

    Readiness is per SNAPSHOT now — the engine emits one per credential subject, so in
    practice one per account: the harness is usable while ANY of its snapshots is, and
    when they are all spent the instant reported is the EARLIEST, because the first to
    heal makes the harness usable again. The reader groups by `subject.harness` and
    deliberately never interprets `subject.subject_id`: WHICH profile a run lands on
    stays Claudexor's business, so no rotation moves into Ouroboros."""
    from ouroboros.subagents import _exhausted_window

    def _snap(profile, *, spent, reset="2026-08-03T12:00:00Z", harness="some-route",
              freshness="fresh", applies=None):
        # `subject_id` is the REAL QuotaSubject key for a credential profile
        # (packages/schema/src/quota.ts; the object is `.strict()`, so the `profile`
        # this fixture used to invent would be rejected by the engine's own parser).
        constraint = ({"used_ratio": 1.0, "resets_at": reset} if spent
                      else {"used_ratio": 0.4, "resets_at": reset})
        if applies is not None:
            constraint["applies_to_models"] = applies
        return {"subject": {"harness": harness, "subject_id": profile},
                "freshness": freshness, "constraints": [constraint]}

    class _Quota:
        def __init__(self, snaps, absences=None):
            self._snaps, self._absences = snaps, absences
        def quota_snapshots(self): return self._snaps
        def quota_absences(self): return self._absences or []

    # ONE of two profiles spent: the harness is still usable, so no blocker at all.
    mixed = _Quota([_snap("acct-a", spent=True, reset="2026-08-03T10:00:00Z"),
                    _snap("acct-b", spent=False)])
    assert _exhausted_window(mixed, "some-route") == (False, "")

    # ALL profiles spent: a blocker, at the EARLIEST reset (the first one to heal).
    both = _Quota([_snap("acct-a", spent=True, reset="2026-08-03T12:00:00Z"),
                   _snap("acct-b", spent=True, reset="2026-08-03T10:00:00Z")])
    assert _exhausted_window(both, "some-route") == (True, "2026-08-03T10:00:00Z")

    # A single-profile harness (no profile field at all) behaves exactly as before.
    single = _Quota([{"subject": {"harness": "some-route"}, "freshness": "fresh",
                      "constraints": [{"used_ratio": 1.0, "resets_at": "2026-08-03T09:00:00Z"}]}])
    assert _exhausted_window(single, "some-route") == (True, "2026-08-03T09:00:00Z")

    # Another harness's exhaustion is not ours, and a STALE snapshot never blocks.
    other = _Quota([_snap("acct-a", spent=True, harness="other-route")])
    assert _exhausted_window(other, "some-route") == (False, "")
    stale = _Quota([_snap("acct-a", spent=True, freshness="stale")])
    assert _exhausted_window(stale, "some-route") == (False, "")

    # And the live sibling wins even when the spent one is listed second.
    reordered = _Quota([_snap("acct-b", spent=False), _snap("acct-a", spent=True)])
    assert _exhausted_window(reordered, "some-route") == (False, "")


def test_a_model_scoped_window_does_not_block_a_route_pinned_to_another_model():
    """The live incident (2026-08-06): the claude route was pinned to opus, its ONE
    readable profile carried `weekly_scoped:Fable used_ratio=1.0` next to a healthy
    five-hour window, and the whole route read as spent until the Fable weekly reset —
    $82 of metered spend for a subscription that was free for opus the entire time.
    A window scoped to models this route never uses is someone else's exhaustion."""
    from ouroboros.subagents import _exhausted_window

    fable_scoped = {"subject": {"harness": "some-route", "subject_id": "acct"},
                    "freshness": "fresh",
                    "constraints": [
                        {"used_ratio": 0.0, "resets_at": "2026-08-07T00:00:00Z"},
                        {"used_ratio": 1.0, "resets_at": "2026-08-11T00:00:00Z",
                         "applies_to_models": ["fable", "claude-fable-5", "best"]},
                    ]}

    class _Quota:
        def __init__(self, snaps, absences=None):
            self._snaps, self._absences = snaps, absences
        def quota_snapshots(self): return self._snaps
        def quota_absences(self): return self._absences or []

    quota = _Quota([fable_scoped])
    # Pinned to opus: the Fable weekly window does not apply, the route is usable.
    assert _exhausted_window(quota, "some-route", "opus") == (False, "")
    # Pinned to fable (either alias direction): the scoped window DOES apply, and the
    # profile's healthy sibling constraint does not rescue it (a spent window blocks
    # its own profile whatever the other windows say).
    assert _exhausted_window(quota, "some-route", "fable") == (True, "2026-08-11T00:00:00Z")
    assert _exhausted_window(quota, "some-route", "claude-fable-5") == (True, "2026-08-11T00:00:00Z")
    # No model pin: any scoped window may apply to whatever model the run lands on.
    assert _exhausted_window(quota, "some-route", "") == (True, "2026-08-11T00:00:00Z")


def test_a_spent_window_with_no_reset_instant_is_still_spent():
    """The inverse defect (three reviewers independently): a fully-used window whose
    constraint named neither `resets_at` nor `cooldown_until` produced no collectable
    reset, and the old single-string contract could only express exhaustion AS a
    reset — so a positively spent route read back as healthy and D28's loud fallback
    never fired. Exhaustion and its healing instant are separate facts now."""
    from ouroboros.subagents import _exhausted_window, route_health, delegated_run_shape

    undated = {"subject": {"harness": "some-route", "subject_id": "acct"},
               "freshness": "fresh", "constraints": [{"used_ratio": 1.0}]}

    class _Quota:
        def __init__(self, snaps): self._snaps = snaps
        def quota_snapshots(self): return self._snaps
        def quota_absences(self): return []

    assert _exhausted_window(_Quota([undated]), "some-route") == (True, "")

    # And through the ONE health reader: an undated exhaustion still reaches the rule
    # table as `subscription_window_exhausted`, as the REASON with an empty reset.
    class _Gateway(_Quota):
        engine_version = "9.9.9"
        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                   "accessProfilesSupported": ["readonly"]}]}

    unavailable, reset_at = route_health(
        _Gateway([undated]), "some-route", delegated_run_shape(False))
    assert (unavailable, reset_at) == ("subscription_window_exhausted", "")


def test_an_unreadable_profile_keeps_the_route_usable():
    """Exhaustion needs POSITIVE evidence for the WHOLE route. A profile whose quota
    endpoint answered 429 (or whose refresh failed) is an ABSENCE — unknown, not
    spent — so the readable-but-spent minority must not speak for the route: the
    daemon owns rotation and refuses typed at start time if the route is truly empty.
    (The live incident's second layer: the backup account's usage endpoint kept
    429-ing, so the one readable profile's Fable window silenced the whole harness.)"""
    from ouroboros.subagents import _exhausted_window

    spent = {"subject": {"harness": "some-route", "subject_id": "acct-a"},
             "freshness": "fresh",
             "constraints": [{"used_ratio": 1.0, "resets_at": "2026-08-11T00:00:00Z"}]}
    absence = {"subject": {"harness": "some-route", "subject_id": "acct-b"},
               "reason": "refresh_failed", "detail": "oauth/usage responded 429"}
    foreign_absence = {"subject": {"harness": "other-route", "subject_id": "acct-x"},
                       "reason": "refresh_failed", "detail": "oauth/usage responded 429"}

    class _Quota:
        def __init__(self, snaps, absences=None):
            self._snaps, self._absences = snaps, absences
        def quota_snapshots(self): return self._snaps
        def quota_absences(self): return self._absences or []

    # An absence on THIS route fail-opens it; a foreign route's absence changes nothing.
    assert _exhausted_window(_Quota([spent], [absence]), "some-route") == (False, "")
    assert _exhausted_window(
        _Quota([spent], [foreign_absence]), "some-route"
    ) == (True, "2026-08-11T00:00:00Z")

    # A gateway with no absence reader at all (test stubs, older fakes) keeps the
    # plain positive-evidence answer.
    class _NoAbsences:
        def __init__(self, snaps): self._snaps = snaps
        def quota_snapshots(self): return self._snaps

    assert _exhausted_window(
        _NoAbsences([spent]), "some-route") == (True, "2026-08-11T00:00:00Z")


def test_dispatch_row_auto_without_a_route_runs_native(monkeypatch):
    res = _dispatch("auto", route="", monkeypatch=monkeypatch)
    assert (res.executor, res.reason) == ("native", "harness_not_configured")


def test_dispatch_row_auto_with_a_healthy_route_becomes_a_nanny(monkeypatch):
    res = _dispatch("auto", monkeypatch=monkeypatch)
    assert (res.executor, res.reason) == ("harness", "harness_ready")


def test_dispatch_row_auto_with_every_profile_spent_falls_back_to_the_api(monkeypatch):
    """D28 through the REAL dispatch entry point, with the disclosure it owes.

    Three destinations (p2's `capability_delta` chain composed with this at
    synthesis): the durable `subagent_executor_resolved` row the dispatch emits, the
    child's own prompt note, and the parent-facing envelope's
    `effective_executor` / `capability_delta`."""
    from ouroboros.agent import dispatch_executor_note, resolve_dispatch_axes

    res = _dispatch("auto", stub=_HealthStub(reset_at="2030-01-01T00:00:00Z"), monkeypatch=monkeypatch)
    assert res.executor == "native" and not res.blocked
    assert res.reason == SUBSCRIPTION_WINDOW_EXHAUSTED
    assert res.reset_at == "2030-01-01T00:00:00Z"

    # Destination 2: the child is told it fell back, that the money is real, and when
    # the substrate would have healed — it must not discover any of that by spending.
    note = dispatch_executor_note(res)
    assert "CAPABILITY DELTA" in note and "METERED" in note
    assert "2030-01-01T00:00:00Z" in note

    # Destination 3: the parent reads what actually ran, and that it diverged —
    # through the REAL resolution seam, not a hand-built envelope: the dispatch
    # stamps the record and rebuilds the envelope from it (one writer).
    task = {"id": "t-child", "type": "task", "delegation_role": "subagent",
            "requested_executor": "auto"}
    resolve_dispatch_axes(task)
    envelope = task["subagent_envelope"]
    assert envelope["executor"] == "auto"
    assert envelope["effective_executor"] == "native"
    assert envelope["capability_delta"]["reason"] == SUBSCRIPTION_WINDOW_EXHAUSTED
    assert envelope["capability_delta"]["reduced"] is True

    # And the PIN keeps the opposite answer: it exists to refuse metered spend.
    pinned = _dispatch("harness", stub=_HealthStub(reset_at="2030-01-01T00:00:00Z"),
                       monkeypatch=monkeypatch)
    assert pinned.blocked and pinned.reason == SUBSCRIPTION_WINDOW_EXHAUSTED


def test_dispatch_row_auto_with_an_unavailable_route_runs_native_with_a_visible_marker(monkeypatch):
    from ouroboros.agent import dispatch_executor_note

    res = _dispatch("auto", raises=cx.ClaudexorUnavailable("daemon_unreachable", "no daemon"),
                    monkeypatch=monkeypatch)
    assert (res.executor, res.reason) == ("native", "daemon_unreachable")
    # "Visible" is the whole point of this row: the child must not discover the
    # fallback by spending.
    note = dispatch_executor_note(res)
    assert "METERED" in note and "daemon_unreachable" in note


def test_dispatch_row_explicit_harness_blocks_and_never_reaches_the_native_path(monkeypatch):
    for stub, raises in (
        (_HealthStub(status="unavailable"), None),
        (None, cx.ClaudexorUnavailable("daemon_unreachable", "no daemon")),
    ):
        res = _dispatch("harness", stub=stub, raises=raises, monkeypatch=monkeypatch)
        # The regression this exists for: a pin that silently becomes a metered native
        # run bills the owner for precisely what the pin was asked to prevent.
        assert res.executor != "native", res
        assert res.blocked, res
    res = _dispatch("harness", route="", monkeypatch=monkeypatch)
    assert res.blocked and res.reason == "harness_not_configured"


def test_dispatch_row_native_is_native_and_asks_the_daemon_nothing(monkeypatch):
    from ouroboros.gateways import claudexor as gw
    from ouroboros.subagents import dispatch_executor_resolution

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route")

    def _boom(*a, **k):
        raise AssertionError("a native request must not touch the daemon")

    monkeypatch.setattr(gw, "ClaudexorGateway", _boom)
    res = dispatch_executor_resolution({"delegation_role": "subagent", "requested_executor": "native"})
    assert (res.executor, res.reason) == ("native", "requested_native")


def test_a_blocked_pin_ends_the_task_unrun_instead_of_spending(monkeypatch):
    from ouroboros.agent import executor_blocked_outcome

    res = _dispatch("harness", raises=cx.ClaudexorUnavailable("daemon_unreachable", "x"),
                    monkeypatch=monkeypatch)
    text, usage = executor_blocked_outcome(res)
    assert usage == {"execution_status": "infra_failed",
                     "reason_code": "subagent_executor_unavailable"}
    assert "NOT run on metered API tokens" in text
    # No visible marker for a blocked run: there is no child to inform.
    from ouroboros.agent import dispatch_executor_note
    assert dispatch_executor_note(res) == ""


def test_a_plain_task_is_not_subject_to_the_executor_axis(monkeypatch):
    """The guard lives at the PRODUCTION entry point, `agent.resolve_dispatch_axes`:
    a task with no `delegation_role: subagent` resolves no axes at all and never
    reaches the daemon. (There used to be a second, test-only wrapper in `agent.py`
    carrying its own copy of this guard while production went through
    `resolve_subagent_dispatch`; the guard is pinned where it actually runs.)"""
    from ouroboros.agent import resolve_dispatch_axes
    from ouroboros.gateways import claudexor as gw

    def _boom(*a, **k):
        raise AssertionError("a plain task must not touch the daemon")

    monkeypatch.setattr(gw, "ClaudexorGateway", _boom)
    task = {"type": "improvement"}
    assert resolve_dispatch_axes(task) is None
    assert "effective_executor" not in task


def test_an_acting_child_is_health_checked_against_the_profile_it_will_ask_for(monkeypatch):
    # A route that can only read is not a usable substrate for a child that must write.
    res = _dispatch("harness", stub=_HealthStub(profiles=("readonly",)),
                    monkeypatch=monkeypatch, acting=True)
    assert res.blocked and res.reason == "access_profile_unsupported:workspace_write"
    res = _dispatch("harness", stub=_HealthStub(profiles=("readonly",)), monkeypatch=monkeypatch)
    assert res.executor == "harness"


def test_a_route_that_declares_only_the_confined_profile_is_admitted_not_refused(monkeypatch):
    """Ouroboros must not refuse the run Claudexor would admit.

    A delegated run is externally confined, so the engine rewrites `workspace_write` to
    `external_sandbox_full` before it checks the manifest — and a route whose adapter
    stands its own sandbox down in favour of that boundary declares only the confined
    profile. `opencode` is exactly that route (`["full", "external_sandbox_full",
    "inherit_native"]`, given the profile so a delegated mutating run on macOS could
    exist at all). Comparing the literal blocked a pinned `harness` executor outright
    and dropped `auto` to a metered native child for no reason on either side.
    """
    opencode = ("full", "external_sandbox_full", "inherit_native")
    res = _dispatch("harness", stub=_HealthStub(profiles=opencode),
                    monkeypatch=monkeypatch, acting=True)
    assert res.executor == "harness" and not res.blocked
    # The fallback is the DELEGATED run's alone: a read-only child asks for `readonly`,
    # the engine leaves it `readonly`, and opencode really cannot serve it.
    res = _dispatch("harness", stub=_HealthStub(profiles=opencode), monkeypatch=monkeypatch)
    assert res.blocked and res.reason == "access_profile_unsupported:readonly"
    # And a route with neither profile still refuses the acting child.
    res = _dispatch("harness", stub=_HealthStub(profiles=("readonly", "inherit_native")),
                    monkeypatch=monkeypatch, acting=True)
    assert res.blocked and res.reason == "access_profile_unsupported:workspace_write"


def test_a_stale_unknown_executor_value_degrades_to_auto_not_to_a_crash(monkeypatch):
    res = _dispatch("a-value-from-an-older-build", monkeypatch=monkeypatch)
    assert res.executor == "harness" and res.requested == "auto"


def test_executor_resolution_row_also_lands_in_canonical_events(tmp_path):
    """W3 adjacent (c): a delegated child's forked drive is pruned with the task,
    so the subagent_executor_resolved row must ALSO land in the canonical
    events.jsonl (the accounting root the task already carries). The root
    agent's own drive IS canonical — no duplicate row there."""
    import json
    from types import SimpleNamespace

    from ouroboros.agent import _record_executor_resolution

    child_logs = tmp_path / "child_drive" / "logs"
    canonical = tmp_path / "data"
    child_logs.mkdir(parents=True)
    (canonical / "logs").mkdir(parents=True)

    dispatch = SimpleNamespace(executor_resolution=SimpleNamespace(
        requested="auto", executor="native",
        reason=SUBSCRIPTION_WINDOW_EXHAUSTED, reset_at="2030-01-01T00:00:00Z", route=None,
    ))
    task = {"id": "child1", "budget_drive_root": str(canonical)}
    _record_executor_resolution(child_logs, task, dispatch)

    def _rows(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    child_rows = _rows(child_logs / "events.jsonl")
    canon_rows = _rows(canonical / "logs" / "events.jsonl")
    assert len(child_rows) == 1 and len(canon_rows) == 1
    assert canon_rows[0]["type"] == "subagent_executor_resolved"
    assert canon_rows[0]["reason"] == SUBSCRIPTION_WINDOW_EXHAUSTED
    assert canon_rows[0]["reset_at"] == "2030-01-01T00:00:00Z"

    # Same drive (the root agent): exactly one row, no self-duplicate.
    root_task = {"id": "root1", "budget_drive_root": str(canonical)}
    _record_executor_resolution(canonical / "logs", root_task, dispatch)
    canon_rows = _rows(canonical / "logs" / "events.jsonl")
    assert len([r for r in canon_rows if r["task_id"] == "root1"]) == 1


def test_subscription_window_exhausted_beacon_wakes_the_waiting_parent(tmp_path, monkeypatch):
    """W3 adjacent (c): the D28 spent-window resolution appends a typed ADVISORY
    delegation_constraint to the task-tree ledger (reset_at + child id), riding
    the attention channel the wait tools already early-wake on — and the
    enforcement reducer skips it (advisory = disclosure, not a gate)."""
    from types import SimpleNamespace

    from ouroboros import task_tree_ledger as ledger_mod
    from ouroboros.agent import _record_executor_resolution
    from ouroboros.tools.control_delegation import effective_delegation_budget

    monkeypatch.setattr(ledger_mod, "DATA_DIR", tmp_path)
    child_logs = tmp_path / "child_drive" / "logs"
    child_logs.mkdir(parents=True)

    dispatch = SimpleNamespace(executor_resolution=SimpleNamespace(
        requested="auto", executor="native",
        reason=SUBSCRIPTION_WINDOW_EXHAUSTED, reset_at="2030-01-01T00:00:00Z", route=None,
    ))
    task = {"id": "childbeacon1", "parent_task_id": "parentroot1", "root_task_id": "parentroot1"}
    _record_executor_resolution(child_logs, task, dispatch)

    beacons = ledger_mod.tree_ledger_attention_after("parentroot1", "")
    assert len(beacons) == 1
    row = beacons[0]
    assert row["kind"] == "delegation_constraint"
    assert row["needs_parent_attention"] is True
    payload = row["payload"]
    assert payload["advisory"] is True
    assert payload["reset_at"] == "2030-01-01T00:00:00Z"
    assert payload["child_task_id"] == "childbeacon1"
    assert payload["reason"] == SUBSCRIPTION_WINDOW_EXHAUSTED

    # Advisory: the schedule-time enforcement reducer must NOT gate on it.
    decision = effective_delegation_budget(
        {}, missing_capabilities=[],
        unresolved_constraints=ledger_mod.open_delegation_constraints("parentroot1"),
        write_surface="", role="researcher", requested_lane="", intended_lane="light",
        active_child_count=0,
    )
    assert decision.ok

    # A healthy (non-exhausted) resolution appends NO beacon.
    healthy = SimpleNamespace(executor_resolution=SimpleNamespace(
        requested="auto", executor="harness", reason="harness_ready", reset_at="", route=None,
    ))
    _record_executor_resolution(child_logs, {"id": "childbeacon2", "parent_task_id": "parentroot1",
                                             "root_task_id": "parentroot1"}, healthy)
    assert len(ledger_mod.tree_ledger_attention_after("parentroot1", "")) == 1

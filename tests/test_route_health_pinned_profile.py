"""route_health admission contract: the engine, not the default store, decides.

Two owner decisions shaped this surface. 2026-08-18 (reviewer-slot incident):
a pinned credential profile must reach the engine even when the harness row's
aggregate status reads "unavailable" — that status describes the DEFAULT
credential store only. 2026-08-28 (cx-delegation sprint, «статус обманывает,
игнорируй его и всё равно пробуй запустить» + 7=A): the same is true for
UNPINNED routes — a pool-only harness (agy, INV-135) read "unavailable"
FOREVER while its real accounts lived in the engine's credential-profile pool,
so the aggregate doctor STATUS stopped being a refusal entirely. The row's
`enabled` field is different: it is the owner's settings toggle and still
refuses unpinned routes as `route_disabled`.
Admission belongs to the engine: an empty or exhausted pool answers the start
POST with its own typed refusal (INV-135 ``credential_pool_exhausted``), which
under the pre-start charter costs zero model rounds. The engine's belt
capability row (``delegation.available``) is likewise not consulted: Ouroboros
runs never request the belt (no ``extra_mcp_servers``).

What still refuses, pinned or not: catalog absence, access-profile fit for the
shape, the delegated-marker version floor, and positively-known quota
exhaustion for the route's model.

Offline unit fixtures only: a duck-typed gateway answering the manifest
questions route_health asks. No daemon, no network.
"""

from ouroboros.subagents import delegated_run_shape, route_health


class _Gateway:
    """agy-shaped catalog row: default store dead forever, pool-only accounts.

    ``enabled`` defaults True — the field is the OWNER's settings toggle, and
    the INV-135 case is a harness the owner did NOT disable whose doctor
    status still reads unavailable forever."""

    engine_version = "9.9.9"

    def __init__(self, *, status="unavailable", enabled=True, delegation=None):
        self._row = {
            "id": "agy-like", "enabled": enabled, "status": status,
            "accessProfilesSupported": ["readonly", "workspace_write"],
        }
        if delegation is not None:
            self._row["delegation"] = delegation

    def agent_capabilities(self):
        return {"harnesses": [dict(self._row)]}

    def quota_snapshots(self):
        return []


def test_aggregate_row_status_is_not_a_refusal_pinned_or_not():
    shape = delegated_run_shape(False)
    gw = _Gateway()
    # Unpinned: the dead default-store status no longer blocks the dispatch —
    # the engine's own typed answer (e.g. credential_pool_exhausted) is the
    # authority, and under the pre-start charter it costs zero model rounds.
    assert route_health(gw, "agy-like", shape) == ("", "")
    # Pinned: identical admission (the 2026-08-18 pinned precedent, now the
    # general rule).
    assert route_health(gw, "agy-like", shape, pinned_profile="acct-1") == ("", "")
    # Even an enabled+ok row answers the same way: the pair carries no signal
    # this reader consumes.
    assert route_health(
        _Gateway(status="ok", enabled=True), "agy-like", shape,
    ) == ("", "")


def test_owner_disabled_toggle_still_refuses_unpinned():
    # `enabled` is not the doctor's status: the engine schema defines it as the
    # owner's settings switch ("routing excludes it regardless of doctor
    # status"). An explicit owner "no" survives the status-refusal removal; a
    # pinned profile keeps its historical skip (the pin is itself an explicit
    # owner row).
    shape = delegated_run_shape(False)
    disabled = _Gateway(status="ok", enabled=False)
    assert route_health(disabled, "agy-like", shape) == ("route_disabled", "")
    assert route_health(disabled, "agy-like", shape, pinned_profile="acct-1") == ("", "")


def test_catalog_absence_still_refuses_pinned_or_not():
    shape = delegated_run_shape(False)
    # A route the catalog does not carry at all has no engine row to be
    # authoritative about: still refused, pinned or not.
    assert route_health(_Gateway(), "missing-route", shape) == (
        "route_not_in_capability_catalog", "")
    assert route_health(_Gateway(), "missing-route", shape, pinned_profile="acct-1") == (
        "route_not_in_capability_catalog", "")


def test_shape_checks_still_refuse_pinned_or_not():
    # The access-profile check judges the SHAPE, not the credentials: a pin
    # cannot make a read-only route writable.
    class _ReadOnly(_Gateway):
        def __init__(self):
            super().__init__(status="ok", enabled=True)
            self._row["accessProfilesSupported"] = ["readonly"]

    acting = delegated_run_shape(True)
    unavailable, _ = route_health(_ReadOnly(), "agy-like", acting, pinned_profile="acct-1")
    assert unavailable == "access_profile_unsupported:workspace_write"
    unavailable, _ = route_health(_ReadOnly(), "agy-like", acting)
    assert unavailable == "access_profile_unsupported:workspace_write"
    # And the delegated-marker version floor stays: an engine that would 400 the
    # marker is refused before a token is spent, pinned or not.
    old = _Gateway(status="ok", enabled=True)
    old.engine_version = "0.0.1"
    unavailable, _ = route_health(old, "agy-like", acting, pinned_profile="acct-1")
    assert unavailable == "engine_rejects_delegated_marker"


def test_the_belt_capability_row_is_not_consulted():
    """The catalog row's ``delegation`` object describes the engine's OWN
    delegate strategy (MCP belt injection) — a capability Ouroboros runs never
    request. Refusing (or decorating a refusal) on it manufactured a
    "structurally cannot delegate" verdict the engine never gave and blocked
    routes that admit marker-based delegated runs (cursor acting runs under a
    pinned profile). It is ignored for every shape."""
    structural = _Gateway(
        delegation={"available": False, "reason": "manifest_unsupported"})
    assert route_health(structural, "agy-like", delegated_run_shape(True)) == ("", "")
    assert route_health(structural, "agy-like", delegated_run_shape(False)) == ("", "")
    assert route_health(_Gateway(delegation={}), "agy-like", delegated_run_shape(True)) == ("", "")


def test_blocked_pin_wording_is_uniform_without_the_structural_verdict():
    """The ``:delegation_`` wording branch retired with the belt refusal: the
    host no longer manufactures a "waiting will not heal it" verdict the engine
    did not give. Every blocked pin gets the honest generic terminal."""
    from ouroboros.agent import executor_blocked_outcome
    from ouroboros.subagents import SubagentExecutorResolution

    text, usage = executor_blocked_outcome(SubagentExecutorResolution(
        requested="harness", executor="blocked",
        reason="route_status_degraded"))
    assert "Reschedule once the route recovers" not in text
    assert "route_status_degraded" in text and "did not substitute another model, route or substrate" in text
    assert "NOT run and spent nothing" in text  # the pin still spent nothing
    assert usage == {"execution_status": "infra_failed", "reason_code": "subagent_executor_unavailable",
                     "unavailable_reason": "route_status_degraded", "alternatives": [], "host_fallback": False,
                     "terminal_origin": "host_notice"}

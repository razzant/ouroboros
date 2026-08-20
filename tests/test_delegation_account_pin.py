"""Unified-accounts sprint: the Delegation account pin (D-U5/D-U6, §K.7, §L.4).

Split out of tests/test_delegated_subagent_transport.py, whose size is pinned
by the one-way byte ratchet; the shared transport fixtures are imported from
that module.
"""

from __future__ import annotations

import json

from ouroboros import subagents
from tests._delegated_transport_shared import (  # noqa: F401 — autouse fixture
    _owned_gateway_uses_each_test_transport,
)
from tests.test_delegated_run_accounting import _plain_ctx  # noqa: F401 — shared fixture


def test_the_account_pin_is_a_sibling_key_folded_into_the_route(monkeypatch):
    """Unified-accounts sprint (D-U5, frozen contract §L.4): the OPTIONAL account
    pin lives in its OWN key, OUROBOROS_SUBAGENT_PROFILE, read ONLY here — the
    route grammar (`harness[=model][:effort]`) gains no fourth position, so no
    other parser of that grammar moves. Empty pin = the engine's rotation pool
    (D28), and a pin with no route pins nothing."""
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=some-model:high")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROFILE", "koshak")
    route = subagents.get_subagent_harness()
    assert route == subagents.DelegationRoute(
        "some-route", "some-model", "high", profile_id="koshak")

    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROFILE", "")
    assert subagents.get_subagent_harness().profile_id == ""

    # A pin without a route is not a route: delegation stays off.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROFILE", "koshak")
    assert subagents.get_subagent_harness() is None


def test_a_pinned_route_is_judged_by_its_own_subject_exactly():
    """Unified-accounts sprint (§K.7, D-U6 strict pin): with a non-empty account
    pin the run can only ever land on THAT subject, so a healthy sibling must
    not vouch a spent pinned account into a dispatch the engine is certain to
    refuse — the exact inverse of the harness-wide rule, which stays in force
    for automatic rotation (empty pin). Every fail-open rule holds per subject:
    an unreadable pinned quota is UNKNOWN, not spent."""
    from ouroboros.subagents import _exhausted_window

    def _snap(profile, *, spent, reset="2026-08-03T12:00:00Z"):
        constraint = ({"used_ratio": 1.0, "resets_at": reset} if spent
                      else {"used_ratio": 0.4, "resets_at": reset})
        return {"subject": {"harness": "some-route", "subject_id": profile},
                "freshness": "fresh", "constraints": [constraint]}

    class _Quota:
        def __init__(self, snaps, absences=None):
            self._snaps, self._absences = snaps, absences
        def quota_snapshots(self): return self._snaps
        def quota_absences(self): return self._absences or []

    spent_pin_live_sibling = _Quota([_snap("koshak", spent=True),
                                     _snap("acct-b", spent=False)])
    # Harness-wide (unpinned): the live sibling keeps the route usable.
    assert _exhausted_window(spent_pin_live_sibling, "some-route") == (False, "")
    # Pinned to the spent account: the sibling cannot vouch for it.
    assert _exhausted_window(spent_pin_live_sibling, "some-route",
                             "", "koshak") == (True, "2026-08-03T12:00:00Z")
    # Pinned to the live account while the sibling is spent: healthy.
    assert _exhausted_window(_Quota([_snap("koshak", spent=False),
                                     _snap("acct-b", spent=True)]),
                             "some-route", "", "koshak") == (False, "")
    # A pinned account with NO readable snapshot is unknown, not spent —
    # whatever the siblings say (positive-evidence rule, per subject).
    assert _exhausted_window(_Quota([_snap("acct-b", spent=True)]),
                             "some-route", "", "koshak") == (False, "")
    # A typed absence row for the pinned subject fail-opens it exactly as the
    # harness-wide reader does for the whole route…
    absence = {"subject": {"harness": "some-route", "subject_id": "koshak"}}
    assert _exhausted_window(_Quota([_snap("koshak", spent=True)], [absence]),
                             "some-route", "", "koshak") == (False, "")
    # …and a sibling's absence says nothing about the pinned subject.
    sibling_absence = {"subject": {"harness": "some-route", "subject_id": "acct-b"}}
    assert _exhausted_window(_Quota([_snap("koshak", spent=True)], [sibling_absence]),
                             "some-route", "", "koshak") == (True, "2026-08-03T12:00:00Z")


def _waited_run(tmp_path, monkeypatch, summary, requested_model="m",
                requested_profile=""):
    """Drive one terminal `delegate_wait` for `summary`; return the agent payload.

    Profile-aware sibling of the transport module's `_waited_run`: the custody
    row's REQUESTED model and account pin are under test-control — the
    requested-vs-applied disclosures compare them against the engine summary's
    own `model` / `authRoute` receipt."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    class _Stub:
        def handshake(self, **_kw): return {}
        def get_run(self, rid, **_kw): return {"lastSeq": 9, "summary": dict(summary)}
        def remove_project(self, pid): pass
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-a", route_id="r", model=requested_model,
        profile_id=requested_profile,
        project_id="p", project_owned=False)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t-a"
    ctx.task_metadata = {"root_task_id": "t-a"}
    payload = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    delegate._CUSTODY.clear()
    return payload


def test_the_receipt_carries_the_requested_and_applied_account(tmp_path, monkeypatch):
    """Unified-accounts sprint (D-U5/§K.7): the APPLIED account is the engine's
    own authRoute settlement receipt — the same fact the ledger row and the
    SETTLED event carry — and the REQUESTED pin replays off the durable STARTED
    row, so a requested-vs-ran mismatch is disclosable in the «Last delegated
    run» line. Absence stays absence: telemetry that predates the receipt
    writes '', never the request dressed up as the applied account."""
    from ouroboros.subagents import subagent_last_delegation

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "acct-data")
    _waited_run(tmp_path / "acct", monkeypatch,
                {"state": "succeeded", "spendUsd": 0.0, "model": "m",
                 "authRoute": {"profileId": "codex-default"}},
                requested_profile="koshak")
    record = subagent_last_delegation()
    assert record["requested_profile"] == "koshak"
    assert record["applied_profile"] == "codex-default"

    # No receipt, no invention: an engine whose telemetry predates authRoute
    # leaves the applied account empty beside the recorded request.
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "acct-data-2")
    _waited_run(tmp_path / "acct2", monkeypatch,
                {"state": "succeeded", "spendUsd": 0.0, "model": "m"},
                requested_profile="koshak")
    bare = subagent_last_delegation()
    assert bare["requested_profile"] == "koshak"
    assert bare["applied_profile"] == ""


class _PinStubGateway:
    """The smallest start-capable gateway; every request body lands in `seen`."""

    def __init__(self, seen):
        self._seen = seen

    def handshake(self, **_kw): return {}
    def agent_capabilities(self):
        return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                               "accessProfilesSupported": ["readonly"]}]}
    def quota_snapshots(self): return []
    def find_project_id(self, root): return "prj-existing"
    def start_run(self, request, *, idempotency_key=""):
        self._seen["request"] = request
        return {"runId": "run-1"}
    def close(self): pass


def test_the_account_pin_rides_the_start_body_and_the_custody_row(tmp_path, monkeypatch):
    """Unified-accounts sprint (D-U5): OUROBOROS_SUBAGENT_PROFILE rides the run
    request verbatim as `credentialProfileId` — the same wire contract the
    reviewer slots author — and the REQUESTED pin lands on the durable custody
    row beside the requested model, so settlement can disclose requested-vs-ran.
    The field enters the stored canonical body: a retry_of replay (which
    replays that body byte-identically) carries the pin with it."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    seen = {}
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROFILE", "koshak")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _PinStubGateway(seen))
    delegate._CUSTODY.clear()
    delegate._delegate_start(_plain_ctx(tmp_path), "x")
    assert seen["request"]["credentialProfileId"] == "koshak"
    assert delegate._CUSTODY["run-1"].profile_id == "koshak"
    delegate._CUSTODY.clear()


def test_an_unpinned_start_sends_no_credential_profile_field(tmp_path, monkeypatch):
    """No pin, no field: an absent credentialProfileId IS the engine's
    rotation-pool contract (D28) — sending an empty string would pin nothing
    and change the canonical body for every existing install."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    seen = {}
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.delenv("OUROBOROS_SUBAGENT_PROFILE", raising=False)
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _PinStubGateway(seen))
    delegate._CUSTODY.clear()
    delegate._delegate_start(_plain_ctx(tmp_path), "x")
    assert "credentialProfileId" not in seen["request"]
    delegate._CUSTODY.clear()


def test_a_retry_health_check_judges_the_stored_pin_not_the_current_setting(
        tmp_path, monkeypatch):
    """Unified-accounts sprint (D-U5 + D-U6): the retry resolver reads
    `credentialProfileId` off the STORED canonical body into
    `DelegationRoute.profile_id` (`_resolve_retry_invocation`), so a retry's
    pre-flight health check judges the account the replayed body actually names
    — never OUROBOROS_SUBAGENT_PROFILE as it reads today. Proven in BOTH
    directions through the real reader (`route_health` → `_exhausted_window`):
    with the STORED pin spent and the drifted current pin live the retry
    refuses typed with the stored subject's reset, and with the readings
    flipped it dispatches, replaying the stored pin on the wire."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    def _snap(profile, *, spent, reset):
        constraint = ({"used_ratio": 1.0, "resets_at": reset} if spent
                      else {"used_ratio": 0.4, "resets_at": reset})
        return {"subject": {"harness": "some-route", "subject_id": profile},
                "freshness": "fresh", "constraints": [constraint]}

    quota: dict = {"snapshots": []}
    script = ["transport_error", "ok"]
    bodies: list = []

    class _Stub:
        def handshake(self, **_kw): return {}
        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                   "accessProfilesSupported": ["readonly"]}]}
        def quota_snapshots(self): return quota["snapshots"]
        def quota_absences(self): return []
        def find_project_id(self, root): return "prj-existing"
        def start_run(self, request, *, idempotency_key=""):
            bodies.append(request)
            if script.pop(0) == "transport_error":
                raise gw.ClaudexorUnavailable("daemon_unreachable",
                                              "daemon fell over mid-POST")
            return {"runId": "run-1"}
        def close(self): pass

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROFILE", "stored-pin")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()

    # 1. The intended start stores its canonical body — pin included — then the
    #    POST's outcome is lost, leaving the invocation pending.
    lost = json.loads(delegate._delegate_start(_plain_ctx(tmp_path), "the intended work"))
    assert lost["reason"] == "daemon_unreachable"
    token = lost["pending_invocation_id"]
    assert bodies[0]["credentialProfileId"] == "stored-pin"

    # 2. The pin drifts before the retry.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROFILE", "drifted-pin")

    # 3. STORED pin spent, drifted pin live: judging today's setting would
    #    dispatch onto a window the engine is certain to refuse; judging the
    #    stored subject refuses typed, with ITS reset instant.
    quota["snapshots"] = [_snap("stored-pin", spent=True, reset="2026-08-20T00:00:00Z"),
                          _snap("drifted-pin", spent=False, reset="2026-08-21T00:00:00Z")]
    blocked = json.loads(delegate._delegate_start(_plain_ctx(tmp_path), "the intended work",
                                                  retry_of=token))
    assert blocked["reason"] == "subscription_window_exhausted", blocked
    assert blocked["reset_at"] == "2026-08-20T00:00:00Z"
    assert len(bodies) == 1, "a health-refused retry must never reach the wire"

    # 4. Readings flipped: the stored subject is live while today's setting
    #    names a spent account — the retry dispatches, and the wire carries the
    #    STORED body byte-identically, drifted setting notwithstanding.
    quota["snapshots"] = [_snap("stored-pin", spent=False, reset="2026-08-20T00:00:00Z"),
                          _snap("drifted-pin", spent=True, reset="2026-08-21T00:00:00Z")]
    retried = json.loads(delegate._delegate_start(_plain_ctx(tmp_path), "the intended work",
                                                  retry_of=token))
    assert retried["status"] == "started", retried
    assert bodies[-1] == bodies[0], "the retry replays the RECORDED body"
    assert bodies[-1]["credentialProfileId"] == "stored-pin"
    delegate._CUSTODY.clear()


def test_the_account_pin_is_a_persisted_setting_the_gateway_accepts():
    """D-U5 persistence: the gateway merges only keys present in the settings
    defaults, so ``OUROBOROS_SUBAGENT_PROFILE`` must be a member — without it
    the Settings UI sends the owner's pin and the backend silently drops it
    (the exact hunk the v6.105 adoption first lost).
    """
    from ouroboros.gateway.settings import _merge_settings_payload
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS

    assert SETTINGS_DEFAULTS["OUROBOROS_SUBAGENT_PROFILE"] == ""
    merged = _merge_settings_payload({}, {"OUROBOROS_SUBAGENT_PROFILE": "koshak"})
    assert merged["OUROBOROS_SUBAGENT_PROFILE"] == "koshak"
    cleared = _merge_settings_payload(merged, {"OUROBOROS_SUBAGENT_PROFILE": ""})
    assert cleared["OUROBOROS_SUBAGENT_PROFILE"] == ""

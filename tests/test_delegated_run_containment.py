"""The delegated-run marker and the containment it must actually deliver.

Split verbatim out of ``tests/test_delegated_subagent_transport.py`` by theme. This
module owns the engine version floors for the read-only and mutating lanes, the scoped
home a mutating run asks for, and the rule that an isolation no artifact proves is
disclosed rather than relayed as a fact.
"""

from __future__ import annotations

import json
import httpx
import pytest
from ouroboros import subagents
from ouroboros.config import (
    CLAUDEXOR_MIN_VERSION,
    CLAUDEXOR_PROTOCOL_MAJOR,
)
from ouroboros.gateways import claudexor as cx

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _HealthStub,
    _dispatch,
    _gateway,
    _isolation_stub,
    _owned_gateway_uses_each_test_transport,
    _started_request,
    _waiting,
    _write_attempt,
    _write_failed_attempt,
)


def test_a_mutating_run_asks_for_a_scoped_home_and_a_read_only_one_does_not(tmp_path, monkeypatch):
    """The marker is what confines an in-place run; without it the harness inherits the
    operator's `$HOME` and the daemon token in it. It must ride with `isolation: live`
    and must NOT appear on a read-only run, whose envelope is scoped already and whose
    lane has to keep working against a daemon that does not know the field."""
    request, payload = _started_request(tmp_path, acting=True, monkeypatch=monkeypatch)
    assert request["execution"]["delegated"] is True, request["execution"]
    # And what the nanny is told at START is that the home was ASKED for — never that it
    # was applied, which only the run's own artifacts can say. Dropping this leaves the
    # nanny with `isolation: live` alone, the exact shape that reads as "confined".
    assert payload["scoped_home_requested"] is True, payload
    request, payload = _started_request(tmp_path, acting=False, monkeypatch=monkeypatch)
    assert "execution" not in request
    assert payload["scoped_home_requested"] is False, payload


def test_an_engine_without_the_marker_refuses_the_mutating_lane_and_keeps_the_read_only_one(
    tmp_path, monkeypatch,
):
    """The floor is a VERSION, not hope and not a probe, and it is a floor for exactly one
    thing: whether the engine's SCHEMA accepts the marker. `RunExecution` is strict and has
    no `delegated` key below 3.3.0, so the field is a 400 (verified live against the running
    daemon), and the capability catalog lists TOP-LEVEL request keys only, so a nested marker
    is undiscoverable — the version is the only answer available.

    The refusal must be typed and must happen BEFORE the run starts, because the alternative
    is spending a dispatch on a request the engine will reject outright.
    """
    _, refusal = _started_request(tmp_path, acting=True, monkeypatch=monkeypatch,
                                  engine_version=CLAUDEXOR_MIN_VERSION, expect="refused")
    assert refusal["reason"] == "engine_rejects_delegated_marker", refusal
    assert refusal["executor"] == "blocked", refusal
    # Read-only delegation sends no marker, so the same old daemon still serves it.
    request, payload = _started_request(tmp_path, acting=False, monkeypatch=monkeypatch,
                                        engine_version=CLAUDEXOR_MIN_VERSION)
    assert payload["status"] == "started" and "execution" not in request


def test_the_dispatcher_refuses_the_same_engine_the_nanny_would(monkeypatch):
    """The twin surface. `route_health` is the ONE health reader, so the decision made at
    DISPATCH — before a token is spent — must agree with the nanny's own. An `auto` child
    falls back to a NATIVE run with the visible marker (never to an uncontained delegated
    one); an explicit `harness` pin becomes a typed blocker; read-only is untouched."""
    from ouroboros.agent import dispatch_executor_note

    old = _HealthStub(engine_version=CLAUDEXOR_MIN_VERSION)
    res = _dispatch("auto", stub=old, monkeypatch=monkeypatch, acting=True)
    assert (res.executor, res.reason) == ("native", "engine_rejects_delegated_marker")
    assert "engine_rejects_delegated_marker" in dispatch_executor_note(res)
    res = _dispatch("harness", stub=_HealthStub(engine_version=CLAUDEXOR_MIN_VERSION),
                    monkeypatch=monkeypatch, acting=True)
    assert res.blocked and res.reason == "engine_rejects_delegated_marker"
    # A read-only child needs no marker, so the same engine is a healthy substrate.
    res = _dispatch("auto", stub=_HealthStub(engine_version=CLAUDEXOR_MIN_VERSION),
                    monkeypatch=monkeypatch)
    assert (res.executor, res.reason) == ("harness", "harness_ready")


@pytest.mark.parametrize("engine, serves_read_only, admits_mutating", [
    # Below the TRANSPORT floor: no lane at all, refused at handshake.
    ("3.1.9", False, False),
    # The engine the operator is actually RUNNING. A floor above this one is not caution,
    # it is an outage: read-only delegation stops working against the only live daemon.
    ("3.2.0", True, False),
    ("3.2.1", True, False),
    # The MARKER lands in 3.3.0: `RunExecution` gains `delegated` and the request stops
    # being a 400. 3.3.0-3.3.1 apply no OS boundary and 3.3.2 applies one only where the
    # host has a mechanism — a difference this floor deliberately does NOT try to encode,
    # because a version cannot: the run is admitted and what it actually got is read back
    # per attempt and disclosed.
    ("3.3.0", True, True),
    ("3.3.1", True, True),
    ("3.3.2", True, True),
    ("3.4.0", True, True),
])
def test_the_two_floors_sit_at_the_measured_bands(engine, serves_read_only, admits_mutating):
    """The floor VALUES, not just the code that reads them (docs/DELEGATED_ADMISSION.md).

    Every other test here spells the old engine `CLAUDEXOR_MIN_VERSION` and the new one
    `CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION`, so the wiring is pinned and the NUMBERS are not:
    both constants could be moved to any pair with the transport floor below the mutating
    one and the whole suite stayed green. That is how a transport floor came to sit above
    the operator's own running daemon and a mutating floor came to sit at the release that
    ships one host's boundary.

    The bands are measured, not assumed (2026-08-03, live 3.2.0 daemon + the Claudexor
    tree): the read-only body comes back with the fake-root error and `fieldErrors: {}`,
    while the mutating body is rejected on `/execution/delegated` before the root is even
    looked at, and `RunExecution.delegated` first exists in 3.3.0. The mutating floor is
    the MARKER release for that reason and no other — the boundary that ships in 3.3.2 is
    macOS-only (`docs/DELEGATED_CONFINEMENT.md` §8), so pinning here to 3.3.2 would have
    encoded "a boundary exists" into a number that says the same thing on a host where
    none does.
    """
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR, "compatible": True,
            "engine": {"version": engine},
        })

    with _gateway(handler) as gateway:
        if serves_read_only:
            gateway.handshake()
        else:
            with pytest.raises(cx.ClaudexorUnavailable) as excinfo:
                gateway.handshake()
            assert excinfo.value.code == "engine_too_old"

    # The two floors are asked of the SHAPE, at the one health reader. An engine between
    # them serves read-only and refuses mutating — the asymmetry is the whole design, and
    # collapsing the floors would cost the owner a working lane.
    stub = _HealthStub(engine_version=engine)
    acting = subagents.route_health(stub, "some-route", subagents.delegated_run_shape(True))[0]
    assert (acting == "") is admits_mutating, acting
    if not admits_mutating:
        assert acting == "engine_rejects_delegated_marker"
    assert subagents.route_health(
        stub, "some-route", subagents.delegated_run_shape(False))[0] == "", \
        "read-only sends no marker, so no engine that can talk at all may lose the lane"


def test_asking_for_a_scoped_home_is_not_evidence_that_one_was_applied(tmp_path, monkeypatch):
    """The whole point: the request is a request. An engine that accepted the marker and
    then ran the harness in the operator's own home has produced a CONTAINMENT FAULT, and
    the only witness is the attempt's own artifact — Claudexor projects the applied HOME
    fact onto no `/v2` response (only the boundary half reaches `candidates[].confinement`)."""
    run_dir = tmp_path / "run-1"
    home = tmp_path / "operator-home"
    home.mkdir()
    monkeypatch.setattr(cx, "operator_home", lambda: home)

    # (a) the engine recorded the fact as NOT applied
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir)
    _write_attempt(run_dir, isolated=False, home_dir=str(home))
    out = _waiting(tmp_path, monkeypatch)
    assert out["status"] == "refused" and out["reason"] == "home_isolation_not_applied", out
    assert cancelled["reason"] == "home_isolation_not_applied"

    # (b) it claims isolation while naming the operator's own home — the claim is the lie
    # the artifact check exists to catch, so the boolean alone is not the verification.
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir)
    _write_attempt(run_dir, isolated=True, home_dir=str(home))
    out = _waiting(tmp_path, monkeypatch)
    assert out["reason"] == "home_isolation_not_applied", out

    # (c) it recorded no fact at all: UNPROVEN, which is not the same as breached. A
    # fault needs a fact; the honesty of an undisclosed attempt belongs in the report,
    # not in a cancellation. See the failure-record test below for why absence is
    # the ordinary case rather than a suspicious one.
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir)
    _write_attempt(run_dir, isolated=None, home_dir="")
    out = _waiting(tmp_path, monkeypatch)
    assert out["status"] == "progress" and cancelled == {}, out

    # (d) a scoped home really applied: the run is left alone and keeps reporting progress
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir)
    _write_attempt(run_dir, isolated=True, home_dir=str(tmp_path / "scoped-home"))
    out = _waiting(tmp_path, monkeypatch)
    assert out["status"] == "progress", out
    assert cancelled == {}

    # (e) Phase A3 (Poltergeist sprint, grok-simplified rule): a HOME NESTED inside
    # the operator's own home is NOT a breach — with OR without a recorded OS
    # boundary. The engine roots every scoped home under its runtime dir, which
    # lives under $HOME on every host it supports, and on a host with no boundary
    # mechanism (every non-macOS host today) it CANNOT record one — so the old
    # nested-without-mechanism rule cancelled every mutating Linux run post-factum
    # (the colleague's issue-2 class). The boundary-less nested shape flows to the
    # EXISTING disclosed-unconfined path instead; only a recorded FALSE and the
    # equality case above stay faults. mechanism=None models the boundary-less
    # engine record.
    for nested in (home / "tmp" / "harness", home / "sub", home / "a" / "b" / "c"):
        nested.mkdir(parents=True, exist_ok=True)
        cancelled = _isolation_stub(monkeypatch, run_dir=run_dir)
        _write_attempt(run_dir, isolated=True, home_dir=str(nested), mechanism=None)
        out = _waiting(tmp_path, monkeypatch)
        assert out["status"] == "progress" and cancelled == {}, (nested, out)

    # ...and the SAME nested home WITH the proven boundary stays fine too.
    nested = home / ".claudexor-runtime" / "projects" / "x" / "home"
    nested.mkdir(parents=True, exist_ok=True)
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir)
    _write_attempt(run_dir, isolated=True, home_dir=str(nested))  # proven seatbelt
    out = _waiting(tmp_path, monkeypatch)
    assert out["status"] == "progress" and cancelled == {}, out

    # ...and a SIBLING of the operator home is still legitimately scoped: the fix must
    # not turn "shares a parent directory" into a breach.
    sibling = tmp_path / "operator-home-2"
    sibling.mkdir(exist_ok=True)
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir)
    _write_attempt(run_dir, isolated=True, home_dir=str(sibling))
    out = _waiting(tmp_path, monkeypatch)
    assert out["status"] == "progress" and cancelled == {}, out


def test_absence_of_the_artifact_is_no_evidence_and_a_read_only_run_is_never_faulted(
    tmp_path, monkeypatch,
):
    """Two ways this check could be wrong in the OTHER direction, both of which would
    cancel healthy runs: an attempt writes its record when it FINISHES, so a young run
    legitimately has none; and a read-only child never sent the marker, so its artifacts
    say nothing about a confinement it did not ask for."""
    run_dir = tmp_path / "run-1"
    home = tmp_path / "operator-home"
    home.mkdir()
    monkeypatch.setattr(cx, "operator_home", lambda: home)

    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir)
    out = _waiting(tmp_path, monkeypatch)          # no attempts dir at all
    assert out["status"] == "progress" and cancelled == {}

    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir, effective_access="readonly")
    _write_attempt(run_dir, isolated=False, home_dir=str(home))
    out = _waiting(tmp_path, monkeypatch, acting=False)
    assert out["status"] == "progress", out
    assert cancelled == {}


def test_an_attempt_that_recorded_no_home_fact_is_not_a_containment_fault(tmp_path, monkeypatch):
    """An attempt record can legitimately state no HOME fact. `AC.attemptFailureRecord`
    (orchestrator.ts:3512 and :5088) spreads the applied facts into an errored record
    today, but `harness_home_isolated` is the one OPTIONAL member — omitted when the
    attempt died before its home was decided — and an engine older than 3.3.2 wrote
    attempt_id/harness_id/cost/errored/phase/errors and nothing else. "a01 errored, a02
    repaired it" is the ORDINARY path of the converge loop that Ouroboros's own
    `mode: agent` run takes, so a missing fact must be no evidence — exactly the line
    `_widened_access` already draws for an undisclosed access profile.

    Faulting on it cancels a correctly-confined, finished, SUCCESSFUL run and throws its
    terminal payload away, and tells the nanny that an ordinary harness failure was a
    containment fault it must not retry."""
    run_dir = tmp_path / "run-1"
    home = tmp_path / "operator-home"
    home.mkdir()
    monkeypatch.setattr(cx, "operator_home", lambda: home)

    # The engine's own repair loop: a01 errored, a02 ran confined, the run succeeded.
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir, state="succeeded")
    _write_failed_attempt(run_dir, attempt="a01")
    _write_attempt(run_dir, isolated=True, home_dir=str(tmp_path / "scoped"), attempt="a02")
    out = _waiting(tmp_path, monkeypatch)
    assert out["status"] == "terminal" and cancelled == {}, out
    # Honest, though: one attempt proved nothing, so the run's confinement is not proven.
    # `os_boundary` is empty for the same reason — a01 named no mechanism, and one
    # unconfined attempt is an unconfined run.
    assert out["containment"] == {
        "verified": False, "attempts": 2, "disclosed": 1, "os_boundary": "",
        "nested_under_operator_home": False,
        "note": "not every attempt of this run recorded a harness-HOME fact, so its "
                "confinement is UNPROVEN — do not report it as isolated",
    }, out

    # And a lone failed attempt on a live run is a task failure, not a containment fault.
    cancelled = _isolation_stub(monkeypatch, run_dir=(only := tmp_path / "run-2"))
    _write_failed_attempt(only, attempt="a01")
    out = _waiting(tmp_path, monkeypatch)
    assert out["status"] == "progress" and cancelled == {}, out


def test_the_relayed_result_never_claims_an_isolation_no_artifact_proves(tmp_path, monkeypatch):
    """What the nanny hands its parent must distinguish PROVEN from merely asked: a run
    that disclosed no harness-HOME fact is unproven, and reporting it as isolated is the
    same untrue claim in a different place."""
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate import _terminal_payload

    run_dir = tmp_path / "run-1"
    detail = {"summary": {"state": "succeeded", "runDir": str(run_dir)}}

    payload = _terminal_payload("run-1", detail, delegated_run_shape(True))
    assert payload["containment"]["verified"] is False
    assert "UNPROVEN" in payload["containment"]["note"]

    # An artifact that records a BREACH must not read as proof either: this verdict is
    # judged by the same predicate that halts the run, not by having been reached after
    # it, so it cannot be turned into a false "verified" by a change of call site.
    monkeypatch.setattr(cx, "operator_home", lambda: tmp_path / "operator-home")
    _write_attempt(run_dir, isolated=True, home_dir=str(tmp_path / "operator-home"))
    assert _terminal_payload("run-1", detail, delegated_run_shape(True))[
        "containment"]["verified"] is False

    _write_attempt(run_dir, isolated=True, home_dir=str(tmp_path / "scoped-home"))
    payload = _terminal_payload("run-1", detail, delegated_run_shape(True))
    assert payload["containment"] == {
        "verified": True, "attempts": 1, "disclosed": 1, "os_boundary": "seatbelt",
        "nested_under_operator_home": False,
        "note": "every attempt recorded a scoped harness HOME outside the operator's own "
                "AND an applied seatbelt boundary, proven against a path it denies",
    }
    # A mechanism WITHOUT the denied path it was proven against is a promise, not an
    # applied fact — the exact shape 3.3.2's evidence block exists to replace.
    _write_attempt(run_dir, isolated=True, home_dir=str(tmp_path / "scoped-home"),
                   mechanism=None)
    unproven = tmp_path / "run-1" / "attempts" / "a01" / "attempt.yaml"
    unproven.write_text(unproven.read_text(encoding="utf-8")
                        + 'confinement_mechanism: "seatbelt"\n', encoding="utf-8")
    claimed = _terminal_payload("run-1", detail, delegated_run_shape(True))["containment"]
    assert claimed["os_boundary"] == "" and claimed["verified"] is False, claimed

    # A read-only run asked for nothing, so it claims nothing.
    assert "containment" not in _terminal_payload("run-1", detail, delegated_run_shape(False))


def test_a_run_with_no_os_boundary_is_disclosed_in_three_places_and_still_allowed(
    tmp_path, monkeypatch,
):
    """The scoped HOME is not the boundary, so a run that got only the HOME must not read
    like a run that got both — and it must still RUN.

    Before this, the two were BYTE-IDENTICAL here: an attempt with a kernel-enforced
    boundary and an attempt with none both produced
    `{verified: true, ... "every attempt recorded a scoped harness HOME outside the
    operator's own"}`, because the reader asked only about `harness_home_isolated`. The
    only thing standing between that report and a genuinely unconfined run was a VERSION
    floor pinned at the release that ships the boundary — and Claudexor's own
    `docs/DELEGATED_CONFINEMENT.md` §8 says that boundary is macOS-only, so the same
    number means "confined" on one host and nothing on another.

    The fix is not a refusal and not an OS test. Ouroboros asks the engine what it
    APPLIED, and where nothing was applied it says so LOUDLY in the three places
    AGENTS.md names — the durable record, the child's prompt, and the parent's result —
    while the work goes ahead (the child already holds a shell in this worktree).
    """
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate import _terminal_payload

    run_dir = tmp_path / "run-1"
    home = tmp_path / "operator-home"
    home.mkdir()
    monkeypatch.setattr(cx, "operator_home", lambda: home)
    detail = {"summary": {"state": "succeeded", "runDir": str(run_dir)}}
    scoped = str(tmp_path / "scoped-home")

    # (1) THE PARENT'S RESULT distinguishes the two runs. This is the assertion the old
    # reader could not make: same HOME evidence, opposite verdicts.
    _write_attempt(run_dir, isolated=True, home_dir=scoped, mechanism="seatbelt")
    confined = _terminal_payload("run-1", detail, delegated_run_shape(True))["containment"]
    _write_attempt(run_dir, isolated=True, home_dir=scoped, mechanism=None)
    bare = _terminal_payload("run-1", detail, delegated_run_shape(True))["containment"]
    assert confined != bare, "a boundary and no boundary must not report identically"
    assert (confined["os_boundary"], confined["verified"]) == ("seatbelt", True), confined
    assert (bare["os_boundary"], bare["verified"]) == ("", False), bare
    assert "NO OS-ENFORCED BOUNDARY" in bare["note"], bare
    assert "daemon token" in bare["note"], "say what is reachable, not just that it failed"

    # The predicate is the APPLIED MECHANISM, never the host OS. A mechanism Ouroboros
    # has never heard of counts as a boundary: the day a Linux one ships, this reader is
    # already right, and it never had a `sys.platform` branch to go stale.
    _write_attempt(run_dir, isolated=True, home_dir=scoped, mechanism="landlock")
    future = _terminal_payload("run-1", detail, delegated_run_shape(True))["containment"]
    assert (future["os_boundary"], future["verified"]) == ("landlock", True), future

    # (2) THE DURABLE RECORD carries it, and the run is NOT cancelled or refused.
    _write_attempt(run_dir, isolated=True, home_dir=scoped, mechanism=None)
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir, state="succeeded")
    out = _waiting(tmp_path, monkeypatch)
    assert out["status"] == "terminal" and cancelled == {}, out
    assert out["containment"]["os_boundary"] == "", out
    events = [json.loads(line) for line in
              (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    unconfined = [e for e in events if e["type"] == "delegate_run_unconfined"]
    assert len(unconfined) == 1, events
    assert unconfined[0]["run_id"] == "run-1" and unconfined[0]["os_boundary"] == ""
    assert "NO OS-ENFORCED BOUNDARY" in unconfined[0]["note"]

    # A run that DID get a boundary writes no such line — the durable record states the
    # gap, it does not narrate every healthy run.
    _write_attempt(run_dir, isolated=True, home_dir=scoped, mechanism="seatbelt")
    _isolation_stub(monkeypatch, run_dir=run_dir, state="succeeded")
    _waiting(tmp_path, monkeypatch)
    events = [json.loads(line) for line in
              (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([e for e in events if e["type"] == "delegate_run_unconfined"]) == 1, events


def test_linux_shaped_run_is_disclosed_unconfined_with_the_engines_reason_not_cancelled(
    tmp_path, monkeypatch,
):
    """Phase A3, the exact incident shape: a Linux host has no boundary mechanism,
    so the engine records `home_isolated: true`, a scoped home NESTED under $HOME,
    NO mechanism, and its typed `confinement_unavailable_reason`. The run must NOT
    be cancelled post-factum (the old rule cancelled every mutating Linux run);
    the reason AMPLIFIES the unconfined disclosure — parent payload and durable
    record — and is never an admission token."""
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate import _terminal_payload

    run_dir = tmp_path / "run-1"
    home = tmp_path / "operator-home"
    nested = home / ".claudexor-runtime" / "projects" / "x" / "home"
    nested.mkdir(parents=True)
    monkeypatch.setattr(cx, "operator_home", lambda: home)

    _write_attempt(
        run_dir, isolated=True, home_dir=str(nested), mechanism=None,
        unavailable_reason="no_boundary_mechanism_for_host: linux",
    )
    # The run keeps reporting progress — no cancellation.
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir)
    out = _waiting(tmp_path, monkeypatch)
    assert out["status"] == "progress" and cancelled == {}, out

    # Parent payload: unconfined, with the engine's own reason beside the note.
    detail = {"summary": {"state": "succeeded", "runDir": str(run_dir)}}
    containment = _terminal_payload("run-1", detail, delegated_run_shape(True))["containment"]
    assert containment["verified"] is False and containment["os_boundary"] == ""
    assert containment["confinement_unavailable_reason"] == "no_boundary_mechanism_for_host: linux"
    assert "no_boundary_mechanism_for_host: linux" in containment["note"]

    # Durable record: the unconfined row carries the same reason.
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir, state="succeeded")
    out = _waiting(tmp_path, monkeypatch)
    assert out["status"] == "terminal" and cancelled == {}, out
    events = [json.loads(line) for line in
              (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    unconfined = [e for e in events if e["type"] == "delegate_run_unconfined"]
    assert len(unconfined) == 1, events
    assert unconfined[0]["confinement_unavailable_reason"] == "no_boundary_mechanism_for_host: linux"

    # The reason is NOT an admission token: a recorded FALSE stays a fault even
    # when a reason sits beside it.
    _write_attempt(
        run_dir, isolated=False, home_dir=str(home), mechanism=None,
        unavailable_reason="no_boundary_mechanism_for_host: linux",
    )
    cancelled = _isolation_stub(monkeypatch, run_dir=run_dir)
    out = _waiting(tmp_path, monkeypatch)
    assert out["reason"] == "home_isolation_not_applied", out
    assert cancelled["reason"] == "home_isolation_not_applied"


def test_the_child_is_told_its_boundary_is_a_request_and_not_a_fact(tmp_path, monkeypatch):
    """Destination 2. The child is the only party that can act on this at the time it
    matters, and it is also the party that writes the answer the parent reads — so it is
    told, in its own instructions, not to describe itself as sandboxed.

    It cannot be told WHICH way it went: nothing at start knows. The engine decides per
    attempt and records the fact afterwards, so the honest thing to hand the child is the
    uncertainty plus the behaviour it implies. A read-only child asked for no boundary and
    is told nothing about one."""
    request, _ = _started_request(tmp_path, acting=True, monkeypatch=monkeypatch)
    instructions = request["instructions"]
    assert "not guaranteed" in instructions.lower(), instructions
    assert "sandboxed or confined" in instructions, instructions
    assert "Work as if there is no boundary" in instructions, instructions

    request, _ = _started_request(tmp_path, acting=False, monkeypatch=monkeypatch)
    assert "boundary" not in request["instructions"].lower(), request["instructions"]

"""Phase 3: Claudexor transport, the nanny verbs, and their accounting/failure classes."""

from __future__ import annotations

import json
import pathlib
import httpx
import pytest
from ouroboros.config import (
    CLAUDEXOR_MIN_VERSION,
    CLAUDEXOR_PROTOCOL_MAJOR,
)
from ouroboros.gateways import claudexor as cx
from ouroboros.loop_llm_call import (
    SUBSCRIPTION_WINDOW_EXHAUSTED,
    classify_llm_exception,
)

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _LiveRunStub,
    _event_types,
    _gateway,
    _nanny_ctx,
    _owned_gateway_uses_each_test_transport,
)


def test_discovery_missing_descriptor_is_a_typed_refusal(tmp_path):
    with pytest.raises(cx.ClaudexorUnavailable) as excinfo:
        cx.discover_daemon(tmp_path)
    assert excinfo.value.code == "daemon_not_discovered"


def test_discovery_reads_host_port_and_token(tmp_path):
    daemon_dir = tmp_path / ".claudexor" / "v3" / "daemon"
    daemon_dir.mkdir(parents=True)
    (daemon_dir / "token").write_text("tok\n", encoding="utf-8")
    (daemon_dir / "control-api.json").write_text(json.dumps({
        "host": "127.0.0.1", "port": 4242, "tokenPath": str(daemon_dir / "token"),
    }), encoding="utf-8")
    endpoint = cx.discover_daemon(tmp_path)
    assert (endpoint.host, endpoint.port, endpoint.token) == ("127.0.0.1", 4242, "tok")


@pytest.mark.parametrize("host,loopback", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("[::1]", True),
    ("127.1.2.3", True), ("fe80::1%lo0", False),
    # The exfiltration shapes: a plain external name, a name that merely LOOKS like a
    # loopback literal, an address that resolves off-host, and the wildcard bind.
    ("evil.example.com", False), ("127.0.0.1.evil.com", False),
    ("10.0.0.5", False), ("0.0.0.0", False), ("169.254.169.254", False),
])
def test_the_daemon_token_is_only_ever_sent_to_loopback(tmp_path, host, loopback):
    """P34P1.3: `discover_daemon` accepted ANY host from control-api.json and the
    gateway shipped the whole-/v2 bearer there. The loopback boundary was documented
    and unenforced, so anything able to write one file under ~/.claudexor could
    redirect the daemon token to a host it controls — token exfiltration plus
    authenticated SSRF, from a file write. The refusal is typed and happens BEFORE any
    client exists; a name is never resolved (a name that resolves to loopback now can
    resolve elsewhere on the next lookup), so only literal loopback addresses and the
    exact name `localhost` pass."""
    daemon_dir = tmp_path / ".claudexor" / "v3" / "daemon"
    daemon_dir.mkdir(parents=True)
    (daemon_dir / "token").write_text("super-secret-daemon-token\n", encoding="utf-8")
    (daemon_dir / "control-api.json").write_text(json.dumps({
        "host": host, "port": 4242, "tokenPath": str(daemon_dir / "token"),
    }), encoding="utf-8")

    if loopback:
        endpoint = cx.discover_daemon(tmp_path)
        assert endpoint.host == host and endpoint.token == "super-secret-daemon-token"
        return
    with pytest.raises(cx.ClaudexorUnavailable) as excinfo:
        cx.discover_daemon(tmp_path)
    assert excinfo.value.code == "daemon_endpoint_not_loopback"
    assert host in str(excinfo.value)
    # The token must not have travelled: the refusal precedes client construction.
    assert "super-secret-daemon-token" not in str(excinfo.value)


@pytest.mark.parametrize("token_name, token_bytes", [
    # A path out of a JSON descriptor can carry an embedded null: `read_text` raises a
    # bare `ValueError`, which is NOT an `OSError`.
    ("to\x00ken", None),
    # The token FILE can hold bytes that are not UTF-8: `UnicodeDecodeError` is likewise
    # a `ValueError` and not an `OSError`.
    ("token", b"\xff\xfetok"),
])
def test_an_unreadable_token_is_a_typed_refusal_not_a_bare_ValueError(
        tmp_path, token_name, token_bytes):
    """The half of the v6.87.44 widening nothing could falsify.

    The suite referenced `daemon_token_unreadable` nowhere and its only `tokenPath`
    fixture was a valid path, so reverting the catch to `except OSError` left every test
    green while a `ValueError` escaped `discover_daemon` untyped — past the
    `except ClaudexorUnavailable` in `subagents.py` and `delegate.py`, as a traceback.
    The `isinstance` assertion below is the actual claim: the caller's handler catches it.
    """
    daemon_dir = tmp_path / ".claudexor" / "v3" / "daemon"
    daemon_dir.mkdir(parents=True)
    token_path = str(daemon_dir / token_name)
    if token_bytes is not None:
        pathlib.Path(token_path).write_bytes(token_bytes)
    (daemon_dir / "control-api.json").write_text(json.dumps({
        "host": "127.0.0.1", "port": 4242, "tokenPath": token_path,
    }), encoding="utf-8")

    with pytest.raises(cx.ClaudexorUnavailable) as excinfo:
        cx.discover_daemon(tmp_path)
    assert excinfo.value.code == "daemon_token_unreadable"
    assert isinstance(excinfo.value, cx.ClaudexorUnavailable)
    # The descriptor read four lines up refuses the identical shape. Asserting the pair
    # is what keeps them from drifting apart again.
    (daemon_dir / "control-api.json").unlink()
    (daemon_dir / "control-api.json").write_bytes(b"\xff\xfe{}")
    with pytest.raises(cx.ClaudexorUnavailable) as sibling:
        cx.discover_daemon(tmp_path)
    assert sibling.value.code == "daemon_descriptor_unreadable"


def test_handshake_sends_the_protocol_header_and_pins_the_minimum_version():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["header"] = request.headers.get(cx.PROTOCOL_HEADER)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={
            "protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR,
            "compatible": True,
            "engine": {"version": CLAUDEXOR_MIN_VERSION},
        })

    with _gateway(handler) as gateway:
        gateway.handshake()
    assert seen["header"] == str(CLAUDEXOR_PROTOCOL_MAJOR)
    assert seen["auth"] == "Bearer secret-token"


def test_handshake_refuses_an_engine_older_than_the_minimum():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR, "compatible": True,
            "engine": {"version": "0.9.0"},
        })

    with _gateway(handler) as gateway:
        with pytest.raises(cx.ClaudexorUnavailable) as excinfo:
            gateway.handshake()
    assert excinfo.value.code == "engine_too_old"


def test_handshake_refuses_an_incompatible_protocol():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"protocolMajor": 2, "compatible": False})

    with _gateway(handler) as gateway:
        with pytest.raises(cx.ClaudexorUnavailable) as excinfo:
            gateway.handshake()
    assert excinfo.value.code == "protocol_incompatible"


def test_project_registration_is_a_required_first_step():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v2/projects":
            assert request.headers.get("Idempotency-Key")
            return httpx.Response(200, json={"id": "prj-1", "root": "/tmp/x"})
        return httpx.Response(404, json={
            "code": "project_not_registered", "message": "register the root first", "retryable": False,
        })

    with _gateway(handler) as gateway:
        assert gateway.register_project("/tmp/x") == "prj-1"
        with pytest.raises(cx.ClaudexorUnavailable) as excinfo:
            gateway.start_run({"prompt": "hi"})
    assert excinfo.value.code == "project_not_registered"
    assert ("POST", "/v2/projects") in calls


def test_the_window_class_is_chosen_by_the_code_not_by_sniffing_the_context():
    """`quota` was never a Claudexor code. The classifier keys on the real one —
    `subscription_window_exhausted`, the RunFailureCode the engine actually emits — so
    an unrelated refusal carrying a stray reset-shaped key is not announced as a spent
    subscription window and put on a retry timer."""
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={
            "code": "subscription_window_exhausted", "message": "window spent", "retryable": True,
            "context": {"resetsAt": "2030-01-01T00:00:00Z"},
        })

    with _gateway(handler) as gateway:
        with pytest.raises(cx.ClaudexorSubscriptionWindowExhausted) as excinfo:
            gateway.get_run("run-1")
    assert excinfo.value.reset_at == "2030-01-01T00:00:00Z"

    def conflict(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={
            "code": "idempotency_conflict", "message": "same key, different body",
            "retryable": False, "context": {"cooldownUntil": "2030-01-01T00:00:00Z"},
        })

    with _gateway(conflict) as gateway:
        with pytest.raises(cx.ClaudexorUnavailable) as excinfo:
            gateway.get_run("run-1")
    assert excinfo.value.code == "idempotency_conflict"
    assert not isinstance(excinfo.value, cx.ClaudexorSubscriptionWindowExhausted)


def test_an_unreachable_daemon_is_a_typed_refusal_not_a_crash():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _gateway(handler) as gateway:
        with pytest.raises(cx.ClaudexorUnavailable) as excinfo:
            gateway.handshake()
    assert excinfo.value.code == "daemon_unreachable"


def test_the_daemon_token_is_never_returned_to_callers():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"snapshots": []})

    with _gateway(handler) as gateway:
        assert "secret-token" not in json.dumps(gateway.quota_snapshots())


def test_managed_secret_write_uses_the_non_journaled_control_route():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"name": "anthropic", "stored": True})

    with _gateway(handler) as gateway:
        receipt = gateway.set_secret("anthropic", "test-value")

    assert seen == {
        "method": "POST",
        "path": "/v2/secrets",
        "body": {"name": "anthropic", "value": "test-value"},
    }
    assert receipt == {"name": "anthropic", "stored": True}


def test_a_per_request_bound_reaches_httpx_and_absence_is_not_an_unbounded_call():
    """The per-request bound had no pin at all: `_request` could be mutated to ignore
    `timeout_sec` outright and 253 tests stayed green, because everything that exercises
    it goes through `MockTransport`, which sees the request and never the timeout. So the
    one caller that needs it — a `delegate_wait` poll bounded by what its window has left
    — was relying on transport behaviour nothing checked.

    Both directions matter, and the second is the subtle one. Present, the value must
    ARRIVE at the client call (a bound that is computed and dropped is a 60s read wearing
    a five-second name). Absent, the kwarg must not be passed AT ALL: httpx reads an
    explicit `timeout=None` as "no timeout whatsoever", the exact opposite of the client
    default it would otherwise inherit, so the harmless-looking `timeout=timeout_sec`
    turns every ordinary call unbounded."""
    calls = []

    class _Recorder:
        def request(self, method, path, **kwargs):
            calls.append(kwargs)
            return httpx.Response(200, json={"id": path.rsplit("/", 1)[-1], "summary": {}})

    gateway = cx.ClaudexorGateway(cx.DaemonEndpoint("127.0.0.1", 1, "secret-token"))
    gateway.close()
    gateway._client = _Recorder()

    gateway.get_run("run-1", timeout_sec=5.0)
    assert "timeout" in calls[-1], "a computed bound that never reaches httpx is not a bound"
    assert calls[-1]["timeout"].read == 5.0, calls[-1]
    assert calls[-1]["timeout"].connect == cx._CONNECT_TIMEOUT_SEC, calls[-1]

    gateway.get_run("run-2")
    assert "timeout" not in calls[-1], \
        "an absent bound must inherit the client default, and httpx reads timeout=None as NO timeout"


# -- 3.4 the nanny verbs -------------------------------------------------------


# -- 3.6 accounting ------------------------------------------------------------


# -- 3.7 the failure class -----------------------------------------------------


def test_the_transport_error_code_is_the_failure_class_name():
    assert cx.ClaudexorSubscriptionWindowExhausted("x").code == SUBSCRIPTION_WINDOW_EXHAUSTED


def test_the_window_class_is_transient_and_scheduled_by_its_reset():
    exc = cx.ClaudexorSubscriptionWindowExhausted("spent", reset_at="2030-01-01T00:00:00Z")
    classification = classify_llm_exception(exc)
    assert classification.kind == SUBSCRIPTION_WINDOW_EXHAUSTED
    assert classification.kind != "quota_exhausted"
    assert classification.retry_same_request is True
    # Scheduled by the reset instant, never by the 60s-capped exponential backoff.
    assert classification.retry_after_sec is not None
    assert classification.retry_after_sec > 60.0
    assert classification.reset_at == "2030-01-01T00:00:00Z"


def test_a_billing_refusal_stays_permanently_classified():
    classification = classify_llm_exception(RuntimeError("402 payment required"))
    assert classification.kind == "quota_exhausted"
    assert classification.retry_same_request is False
    assert classification.retry_after_sec is None


# -- 4. the executor axis actually reaches dispatch -----------------------------


# One row of the rule table per case, resolved through the REAL dispatch entry point
# rather than through the pure function it wraps.


# -- 4. mutating AND read-only children, one nanny, one transport ---------------


# -- 5. the delegated-run marker and the containment it must actually deliver ----
#
# Without `execution.delegated`, Claudexor gives an in-place (`live`) run the OPERATOR's
# real `$HOME` — which holds `~/.claudexor/v3/daemon/token`, a bearer token for the whole
# `/v2` control API. A mutating delegated child is exactly that shape.


# -- 3.8 custody is durable, not process-local ---------------------------------


def test_an_absent_run_closes_now_and_its_registration_survives_for_the_sweep(tmp_path):
    """P34R.4 decoupled (owner 2026-08-30): CLOSED_ABSENT lands immediately;
    the registration survives replay on project_owned (no wholesale clear) and
    the registration sweep retries until the daemon accepts (404 = discharged)."""
    import ouroboros.delegate_custody as dc
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    class _AbsentRunGateway:
        """get_run 404s (run gone); remove_project is temporarily unreachable."""
        def __init__(self): self.removals, self.remove_fails = [], True
        def handshake(self, **_kw): return {}
        def get_run(self, rid, **_kw):
            raise ClaudexorUnavailable("not_found", "no such run", status_code=404)
        def remove_project(self, pid):
            self.removals.append(pid)
            if self.remove_fails:
                raise ClaudexorUnavailable("daemon_unreachable", "socket died", status_code=0)
        def close(self): pass

    gateway = _AbsentRunGateway()
    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-gone", task_id="t-a", route_id="r", model="m",
        project_id="prj-owned", project_owned=True, ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()

    # 1. Retire unreachable: the run FACT lands now; the debt survives.
    out = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: gateway)
    assert [o["action"] for o in out] == ["absent"]
    kinds = _event_types(tmp_path)
    assert "delegate_run_project_retire_failed" in kinds, "the failure is disclosed"
    assert "delegate_run_closed_absent" in kinds, "fact lands independently"
    assert dc.open_runs(tmp_path) == [], "run custody is over"
    owned = dc.owned_project_registrations(tmp_path)
    assert [c.run_id for c in owned] == ["run-gone"] and owned[0].project_owned is True

    # 2. Daemon recovers: the sweep discharges the obligation.
    gateway.remove_fails = False
    dc._CUSTODY.clear()
    dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: gateway)
    kinds = _event_types(tmp_path)
    assert "delegate_run_project_retired" in kinds
    assert dc.owned_project_registrations(tmp_path) == []
    assert gateway.removals == ["prj-owned"] * 3  # 3 lanes: close, 2 sweeps

    # 3. Absence is discharge: a 404 on the remove itself closes the run.
    class _AllGone(_AbsentRunGateway):
        def remove_project(self, pid):
            raise ClaudexorUnavailable("not_found", "no such project", status_code=404)

    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-gone-2", task_id="t-b", route_id="r", model="m",
        project_id="prj-2", project_owned=True, ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()
    out = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: _AllGone())
    assert [o["action"] for o in out] == ["absent"]
    assert dc.open_runs(tmp_path) == []
    dc._CUSTODY.clear()


# -- 3.9 cancellation reports only what it verified ----------------------------


# -- 3.10 settlement is atomic --------------------------------------------------


def test_settlement_follows_the_ledger_and_the_registration_debt_survives(tmp_path, monkeypatch):
    """Decoupled (owner 2026-08-30): settlement follows the ledger row alone —
    a sibling holding the shared project must not turn a SUCCEEDED run into an
    unsettled one; the registration debt survives on project_owned for the
    sweep, and the idempotent ledger row is never written twice."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    failing = {"now": True}

    class _Stub(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "summary": {"state": "succeeded", "spendUsd": 0.0,
                                              "inputTokens": 3, "outputTokens": 2}}
        def remove_project(self, pid):
            if failing["now"]:
                raise gw.ClaudexorUnavailable("daemon_unreachable", "cannot retire")

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    entry = delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m",
        project_id="prj-ours", project_owned=True, root_task_id="t-a", ledger_root=str(tmp_path))
    assert dc.record_started(tmp_path, entry) is True, "the authoritative row must land"
    ctx = _nanny_ctx(tmp_path)

    first = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    assert first["settlement"]["settled"] is True, "settles on the ledger row"
    assert first["settlement"]["project_retired"] is False, "the debt is disclosed"
    assert entry.settled is True and entry.project_owned is True
    assert "delegate_run_settled" in _event_types(tmp_path)
    assert [c.run_id for c in dc.owned_project_registrations(tmp_path)] == ["run-1"]

    failing["now"] = False
    second = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    delegate._CUSTODY.clear()
    assert second["settlement"]["settled"] is True
    rows = [json.loads(l) for l
            in (tmp_path / "state" / "usage_attempts.jsonl").read_text().splitlines()]
    sessions = [r for r in rows if r.get("kind") == "subscription_session"]
    assert len(sessions) == 1, "the idempotent ledger row must not be written twice"
    assert dc.replay(tmp_path)["run-1"].settled is True

    # An idempotent re-start writes a SECOND started row for the same run. Replaying it
    # must not forget the settlement, or the orphan sweep would be handed a run that has
    # already finished and would try to cancel and re-retire it forever.
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m",
        project_id="prj-ours", project_owned=True, root_task_id="t-a", ledger_root=str(tmp_path)))
    delegate._CUSTODY.clear()
    assert dc.replay(tmp_path)["run-1"].settled is True
    assert "run-1" not in {c.run_id for c in dc.open_runs(tmp_path)}


# -- 3.11 a large result is delivered, not severed -----------------------------


# -- 3.12 reconciliation on restart / parent terminalization -------------------


def test_the_public_wait_is_event_only_and_its_outer_bound_matches_task_lifetime():
    """The model cannot request a fake return window; the host renews quiet windows."""
    import os

    from ouroboros.config import (
        DELEGATE_WAIT_CEILING_SEC,
        DELEGATE_WAIT_WINDOW_MAX_SEC,
        get_delegate_wait_max_sec,
        get_task_abs_ceiling_sec,
    )
    from ouroboros.delegate_progress import EXTERNAL_WAIT_LEASE_CEILING_SEC
    from ouroboros.loop_tool_execution import _DEADLINE_CLAMPED_TOOLS, _PER_CALL_TIMEOUT_TOOLS
    from ouroboros.tools.delegate import get_tools

    entry = next(e for e in get_tools() if e.schema["name"] == "delegate_wait")
    assert "wait_sec" not in entry.schema["parameters"]["properties"]
    assert entry.timeout_sec == get_task_abs_ceiling_sec() + 120
    assert DELEGATE_WAIT_WINDOW_MAX_SEC < DELEGATE_WAIT_CEILING_SEC < EXTERNAL_WAIT_LEASE_CEILING_SEC
    assert (DELEGATE_WAIT_WINDOW_MAX_SEC, DELEGATE_WAIT_CEILING_SEC,
            EXTERNAL_WAIT_LEASE_CEILING_SEC) == (1800, 2100, 2400)
    # ...and neither escape hatch applies to this tool, which is why the ToolEntry
    # value really is the bound. The task deadline is a separate concern and is
    # honoured INSIDE the tool (see the wait-window test below), which is why the
    # outer clamp still must not apply: it would thread-kill the graceful return.
    assert "delegate_wait" not in _PER_CALL_TIMEOUT_TOOLS
    assert "delegate_wait" not in _DEADLINE_CLAMPED_TOOLS

    previous = os.environ.get("OUROBOROS_DELEGATE_WAIT_MAX_SEC")
    os.environ["OUROBOROS_DELEGATE_WAIT_MAX_SEC"] = "7200"
    try:
        # The configurable max clamps to the hard window max — NOT to the
        # ToolEntry timeout: raising the executor timeout must never silently
        # widen the askable window.
        assert get_delegate_wait_max_sec() == DELEGATE_WAIT_WINDOW_MAX_SEC
    finally:
        if previous is None:
            os.environ.pop("OUROBOROS_DELEGATE_WAIT_MAX_SEC", None)
        else:
            os.environ["OUROBOROS_DELEGATE_WAIT_MAX_SEC"] = previous


# -- the window is a WINDOW: the timer waits, the human's stream does not ------


# ---------------------------------------------------------------------------
# BR1-2: the delegate split has no import cycle — one-way seams only
# ---------------------------------------------------------------------------


def test_delegate_split_modules_import_standalone_without_the_facade():
    """The module split's seam pattern is ONE-WAY: an extracted module never
    imports the facade back. `delegate_interactions` used to import `_fail` /
    `_emit` / `_owned_run` from `ouroboros.tools.delegate` — a cycle with the
    facade's own top-level import of the cluster. Each extracted module must
    import standalone in a FRESH interpreter, and none of them may pull the
    facade into sys.modules as a side effect."""
    import subprocess
    import sys

    for module in ("ouroboros.delegate_shared",
                   "ouroboros.delegate_interactions",
                   "ouroboros.delegate_output",
                   "ouroboros.delegate_progress",
                   "ouroboros.delegate_containment",
                   "ouroboros.delegate_custody"):
        probe = (
            f"import sys; import {module}; "
            "assert 'ouroboros.tools.delegate' not in sys.modules, "
            f"'{module} pulled the facade back in'"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"{module} failed to import standalone: {result.stderr}")


def test_facade_reexports_are_the_same_objects_as_their_owners():
    """Monkeypatch targets keep working only when the facade re-export IS the
    owner's object — probe identity, not just importability."""
    from ouroboros import delegate_shared
    from ouroboros.tools import delegate

    assert delegate._fail is delegate_shared._fail
    assert delegate._emit is delegate_shared._emit
    assert delegate._owned_run is delegate_shared._owned_run

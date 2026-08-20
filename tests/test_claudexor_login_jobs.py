"""No-terminal login UX: the disclosure-driven jobs and the paste-code input proxy.

Split verbatim out of ``tests/test_claudexor_owned_daemon.py`` by theme. This module
owns the operations catalog that gates the transport, the single operation-specific
success envelope, the bounded top-level required actions on a control problem, the
exact daemon routes each job uses, the absence and conflict statuses that ride
through typed, and the input endpoint that proxies a pasted code to the engine.

Everything here is offline: no daemon is spawned, no network is touched.
"""

import json

import pytest



# ---------------------------------------------------------------------------
# No-terminal login UX (3.3.7 contract): disclosure-driven claude/cursor jobs
# and the paste-code input proxy.
# ---------------------------------------------------------------------------


def test_login_disclosure_capability_reads_the_operations_catalog():
    """The engine advertises its disclosure-driven login modes by implementing
    the setup-job input route; the predicate reads the /v2/operations catalog
    under the operation's EXACT id, and fails closed on everything else.

    The id is the whole pin. Accepting the PATH template as a second,
    independent yes made the answer true for an operation with ANY id — a
    route shape is not an identity — and a false positive is the expensive
    direction: it sends a pre-3.3.7 engine down the transportless path, whose
    daemon-side default is the Terminal.app handoff D30 forbids."""
    from ouroboros.gateway.claudexor_accounts import _login_disclosure_native

    by_id = [{"id": "post:setup.jobs", "method": "POST", "path": "/v2/setup/jobs"},
             {"id": "post:setup.jobs.id.input", "method": "POST", "path": "/v2/setup/jobs/:id/input"}]
    assert _login_disclosure_native(by_id) is True
    # The id carries the capability even when the path is spelled differently.
    assert _login_disclosure_native([{"id": "post:setup.jobs.id.input", "path": ""}]) is True
    # The route SHAPE alone never does — a foreign id is not this operation.
    foreign_id = [{"id": "whatever", "method": "POST", "path": "/v2/setup/jobs/:id/input"}]
    assert _login_disclosure_native(foreign_id) is False
    without = [{"id": "post:setup.jobs.id.cancel", "method": "POST",
                "path": "/v2/setup/jobs/:id/cancel"}, "not-a-dict"]
    assert _login_disclosure_native(without) is False
    assert _login_disclosure_native([]) is False


def test_login_request_transport_default_is_capability_gated():
    """On a disclosure-native engine a non-codex login OMITS the transport so
    the engine hosts the flow itself (oauth_url in the snapshot overlay, no
    Terminal, no attach command). On an older engine the omitted transport
    would be the forbidden Terminal.app handoff, so client_pty stays forced."""
    from ouroboros.gateway.claudexor_accounts import _build_login_request

    native = _build_login_request("claude", "", "", "", disclosure_native=True)
    assert "transport" not in native
    legacy = _build_login_request("claude", "", "", "", disclosure_native=False)
    assert legacy["transport"] == "client_pty"
    # An EXPLICIT client_pty ask survives on any engine (the card's fallback).
    explicit = _build_login_request("cursor", "", "client_pty", "", disclosure_native=True)
    assert explicit["transport"] == "client_pty"
    # Codex is untouched by the capability: its device flow was already
    # daemon-hosted, transport stays absent either way.
    for flag in (True, False):
        codex = _build_login_request("codex", "", "", "device_auth", disclosure_native=flag)
        assert "transport" not in codex and codex["loginFlow"] == "device_auth"


def _create_login(monkeypatch, tmp_path, body: dict, *, operations, raises=False):
    """Run the REAL create path against a fake daemon, answering the probe with
    ``operations`` (or raising for the catalog-unreadable case). Returns
    ``(answer, request_body_actually_sent)``."""
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway.claudexor_accounts import _login_create
    from ouroboros.gateways import claudexor as gw

    sent: dict = {}

    class FakeDaemon:
        def ensure_running(self):
            return object()

        def reconcile_rotation(self, gateway):
            pass  # B3 reconcile is not this test's subject

    class FakeGateway:
        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def operations(self):
            if raises:
                raise gw.ClaudexorUnavailable("daemon_unreachable", "catalog unreadable")
            return operations

        def setup_job_create(self, request, *, idempotency_key=""):
            sent.clear()
            sent.update(request)
            return {"id": "job-1", "state": "queued"}

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)
    return _login_create(body), sent


_INPUT_OP = {"id": "post:setup.jobs.id.input", "method": "POST",
             "path": "/v2/setup/jobs/:id/input"}


def test_login_create_transport_is_gated_by_the_executed_probe(monkeypatch, tmp_path):
    """End-to-end through the real create path, not just the pure predicate.

    A disclosure-native engine hosts the flow itself: no transport, no attach
    command, and the answer DISCLOSES the capability it decided on
    (`disclosure_native`) so the card can demote the fallback honestly. An
    engine whose catalog merely mounts a same-shaped route under a foreign id
    is NOT that engine: it must still be forced to client_pty, because the
    transportless default on an old engine is the forbidden Terminal.app
    handoff."""
    native, sent = _create_login(monkeypatch, tmp_path, {"harness": "claude"},
                                operations=[_INPUT_OP])
    assert native["job"] == {"id": "job-1", "state": "queued"}
    assert "job" not in native["job"], "create must not emit the old job.job envelope"
    assert native["disclosure_native"] is True
    assert "transport" not in sent
    # No client_pty job ⇒ no attach command to demote into Advanced.
    assert "attach_command" not in native

    wrong_id, sent = _create_login(
        monkeypatch, tmp_path, {"harness": "claude"},
        operations=[{"id": "post:setup.jobs.input", "method": "POST",
                     "path": "/v2/setup/jobs/:id/input"}])
    assert wrong_id["disclosure_native"] is False
    assert sent["transport"] == "client_pty"
    assert wrong_id["attach_command"].endswith("claudexor setup attach job-1")


def test_login_create_fails_closed_when_the_catalog_cannot_be_read(monkeypatch, tmp_path):
    """The probe's `except` path, executed: an unreadable catalog is not a
    capability claim. It degrades to the attach fallback (works on every
    engine) instead of gambling the transportless default."""
    answer, sent = _create_login(monkeypatch, tmp_path, {"harness": "cursor"},
                                operations=[], raises=True)
    assert answer["disclosure_native"] is False
    assert sent["transport"] == "client_pty"
    assert "attach_command" in answer


def test_login_create_keeps_the_codex_invariant_on_both_engines(monkeypatch, tmp_path):
    """Codex is untouched by the capability: its device flow was always
    daemon-hosted, so the transport stays absent (and loginFlow rides only for
    codex) whether or not the engine is disclosure-native — and a job with no
    client_pty transport never carries an attach command."""
    for operations in ([_INPUT_OP], []):
        answer, sent = _create_login(monkeypatch, tmp_path,
                                     {"harness": "codex", "login_flow": "device_auth"},
                                     operations=operations)
        assert "transport" not in sent
        assert sent["loginFlow"] == "device_auth"
        assert "attach_command" not in answer
        assert answer["disclosure_native"] is bool(operations)


def _input_request(job_id: str, body: dict):
    from starlette.requests import Request

    payload = json.dumps(body).encode()

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request({
        "type": "http", "method": "POST",
        "path": f"/api/claudexor/login/{job_id}/input",
        "headers": [(b"content-type", b"application/json")], "query_string": b"",
        "path_params": {"job_id": job_id},
    }, receive)


def _job_request(job_id: str, method: str, suffix: str = ""):
    from starlette.requests import Request

    return Request({
        "type": "http", "method": method,
        "path": f"/api/claudexor/login/{job_id}{suffix}",
        "headers": [], "query_string": b"", "path_params": {"job_id": job_id},
    })


def _invoke_login_job_handler(op: str, job_id: str = "j1"):
    import asyncio

    from ouroboros.gateway.claudexor_accounts import (
        api_claudexor_login_job,
        api_claudexor_login_job_reconcile,
    )

    if op == "reconcile":
        return asyncio.run(api_claudexor_login_job_reconcile(
            _job_request(job_id, "POST", "/reconcile")))
    method = "DELETE" if op == "cancel" else "GET"
    return asyncio.run(api_claudexor_login_job(_job_request(job_id, method)))


def test_login_job_success_envelopes_are_single_and_operation_specific(monkeypatch, tmp_path):
    """Snapshot is already an envelope; bare-job operations wrap exactly once."""
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway.claudexor_accounts import _login_job_call
    from ouroboros.gateways import claudexor as gw

    snapshot = {
        "job": {"id": "j1", "state": "running", "phase": "awaiting_user"},
        "cursor": "cur-1",
        "sequence": 7,
        "deviceCode": {"user_code": "ABCD-EFGH", "verification_uri": "https://example.test"},
    }
    bare = {"id": "j1", "state": "cancelled", "outcome": {"reason": "cancelled_by_user"}}
    seen = []

    class FakeGateway:
        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def setup_job_call(self, job_id, op, *, value=""):
            seen.append((job_id, op, value))
            return snapshot if op == "snapshot" else bare

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    assert _login_job_call("j1", "snapshot") == snapshot
    assert _login_job_call("j1", "cancel") == {"job": bare}
    assert _login_job_call("j1", "input", value=" code ") == {"job": bare, "ok": True}
    assert _login_job_call("j1", "reconcile") == {"job": bare}
    assert seen == [
        ("j1", "snapshot", ""),
        ("j1", "cancel", ""),
        ("j1", "input", " code "),
        ("j1", "reconcile", ""),
    ]


def test_control_problem_required_actions_are_top_level_and_bounded():
    """The typed continuation follows the daemon's exact ControlProblem field."""
    import httpx

    from ouroboros.gateway.claudexor_accounts import _login_job_problem
    from ouroboros.gateways.claudexor import ClaudexorGateway

    gateway = object.__new__(ClaudexorGateway)
    actions = [f"action-{index}-" + ("x" * 600) for index in range(20)]
    problem = gateway._problem(httpx.Response(409, json={
        "code": "setup_termination_unconfirmed",
        "message": "still checking",
        "requiredActions": actions,
        "context": {"requiredActions": ["wrong-place"]},
    }))
    assert problem.code == "setup_termination_unconfirmed"
    assert problem.status_code == 409
    assert len(problem.required_actions) == 16
    assert problem.required_actions == tuple(item[:512] for item in actions[:16])
    browser = _login_job_problem(problem, "reconcile")
    browser_body = json.loads(browser.body)
    assert browser.status_code == 409
    assert browser_body["code"] == "setup_termination_unconfirmed"
    assert browser_body["required_actions"] == list(problem.required_actions)

    nested_only = gateway._problem(httpx.Response(409, json={
        "code": "setup_termination_unconfirmed",
        "message": "still checking",
        "context": {"requiredActions": ["retry_setup_reconciliation"]},
    }))
    assert nested_only.required_actions == ()
    assert "required_actions" not in json.loads(
        _login_job_problem(nested_only, "reconcile").body)


def test_gateway_setup_job_operations_use_the_exact_daemon_routes():
    from ouroboros.gateways.claudexor import ClaudexorGateway

    gateway = object.__new__(ClaudexorGateway)
    calls = []

    def request(method, path, *, json_body=None, **_kwargs):
        calls.append((method, path, json_body))
        return {"id": "j1", "state": "running"}

    gateway._request = request
    assert gateway.setup_job_call("j1", "snapshot")["id"] == "j1"
    assert gateway.setup_job_call("j1", "cancel")["id"] == "j1"
    assert gateway.setup_job_call("j1", "input", value=" code ")["id"] == "j1"
    assert gateway.setup_job_call("j1", "reconcile")["id"] == "j1"
    assert calls == [
        ("GET", "/v2/setup/jobs/j1/snapshot", None),
        ("POST", "/v2/setup/jobs/j1/cancel", None),
        ("POST", "/v2/setup/jobs/j1/input", {"value": " code "}),
        ("POST", "/v2/setup/jobs/j1/reconcile", None),
    ]


@pytest.mark.parametrize("op", ["snapshot", "cancel", "reconcile"])
@pytest.mark.parametrize("status", [404, 410])
def test_login_job_absence_statuses_pass_through(monkeypatch, tmp_path, op, status):
    """Job absence is a client-custody verdict for exactly these operations."""
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateways import claudexor as gw

    seen = []

    class Missing:
        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def setup_job_call(self, job_id, actual_op, *, value=""):
            seen.append(actual_op)
            raise gw.ClaudexorUnavailable(
                f"http_{status}", "job is no longer available", status_code=status)

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", Missing)

    response = _invoke_login_job_handler(op)
    assert response.status_code == status
    assert json.loads(response.body)["code"] == f"http_{status}"
    assert seen == [op]


@pytest.mark.parametrize("op", ["snapshot", "cancel", "reconcile"])
def test_login_job_409_is_reconcile_scoped(monkeypatch, tmp_path, op):
    """Reconcile has a typed 409 continuation; poll/cancel remain unknown 503s."""
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateways import claudexor as gw

    class Unconfirmed:
        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def setup_job_call(self, job_id, actual_op, *, value=""):
            assert actual_op == op
            raise gw.ClaudexorUnavailable(
                "setup_termination_unconfirmed",
                "process-group emptiness is not proven",
                status_code=409,
                required_actions=("retry_setup_reconciliation",),
            )

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", Unconfirmed)

    response = _invoke_login_job_handler(op)
    body = json.loads(response.body)
    if op == "reconcile":
        assert response.status_code == 409
        assert body["code"] == "setup_termination_unconfirmed"
        assert body["required_actions"] == ["retry_setup_reconciliation"]
    else:
        assert response.status_code == 503
        assert "required_actions" not in body


def test_login_reconcile_validates_job_id_before_daemon_work():
    response = _invoke_login_job_handler("reconcile", job_id="")
    assert response.status_code == 400
    assert b"job_id is required" in response.body


def test_login_input_endpoint_validates_before_any_daemon_work():
    import asyncio

    from ouroboros.gateway.claudexor_accounts import api_claudexor_login_job

    missing_job = asyncio.run(api_claudexor_login_job(_input_request("", {"value": "x"})))
    assert missing_job.status_code == 400 and b"job_id is required" in missing_job.body
    missing_value = asyncio.run(api_claudexor_login_job(_input_request("j1", {})))
    assert missing_value.status_code == 400 and b"value is required" in missing_value.body
    # The cap mirrors the engine's ControlSetupJobInputRequest (1..1024), read
    # off the ORIGINAL string — not a trimmed rewrite of it.
    oversized = asyncio.run(api_claudexor_login_job(_input_request("j1", {"value": "x" * 1025})))
    assert oversized.status_code == 400
    assert asyncio.run(api_claudexor_login_job(_input_request("j1", {"value": " " * 1025}))).status_code == 400
    # STRICT body shape: `value` must already BE a string, and the body must
    # already BE an object. A coerced str(123) is not a sign-in code, and a
    # non-object body used to reach `.get` and raise (a 500 for a 400 fault).
    for bad in (123, None, True, ["ABCD"], {"v": "ABCD"}, ""):
        refused = asyncio.run(api_claudexor_login_job(_input_request("j1", {"value": bad})))
        assert refused.status_code == 400, bad
        assert b"value is required" in refused.body, bad
    for body in ("just-a-string", ["ABCD"], 7, None):
        refused = asyncio.run(api_claudexor_login_job(_input_request("j1", body)))
        assert refused.status_code == 400, body


def test_login_input_endpoint_proxies_the_code_to_the_engine(monkeypatch, tmp_path):
    """Thin proxy: the value rides through to the engine's input route
    verbatim and the answer comes back; nothing is stored or interpreted."""
    import asyncio

    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway.claudexor_accounts import api_claudexor_login_job
    from ouroboros.gateways import claudexor as gw

    seen = {}

    class FakeGateway:
        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def setup_job_call(self, job_id, op, *, value=""):
            seen["job_id"], seen["op"], seen["value"] = job_id, op, value
            return {"jobId": job_id, "state": "running", "phase": "verifying"}

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    resp = asyncio.run(api_claudexor_login_job(_input_request("j1", {"value": " ABCD-1234 "})))
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["ok"] is True and body["job"]["state"] == "running"
    # UNCHANGED — a proxy that trims is a proxy that decides. Whichever side
    # normalizes a pasted code (the card does, before it posts) must be the
    # side that owns the meaning; this edge only validates and forwards, so
    # what the engine reads is exactly what the caller sent.
    assert seen == {"job_id": "j1", "op": "input", "value": " ABCD-1234 "}


def test_login_input_engine_404_is_a_typed_capability_gap(monkeypatch, tmp_path):
    """DEGRADED-ENGINE PATH: an engine that predates the input route (or no
    longer knows the job) answers 404; the proxy types it as
    input_not_supported so the card can fall back to the Advanced attach
    affordance. A 404 on the POLL keeps its ordinary job-absence meaning (no
    capability spin) and therefore passes through without that input code."""
    import asyncio

    from starlette.requests import Request

    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway.claudexor_accounts import api_claudexor_login_job
    from ouroboros.gateways import claudexor as gw

    class Refusing:
        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def setup_job_call(self, job_id, op, *, value=""):
            raise gw.ClaudexorUnavailable("http_404", "no such route", status_code=404)

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", Refusing)

    resp = asyncio.run(api_claudexor_login_job(_input_request("j1", {"value": "x"})))
    assert resp.status_code == 404
    body = json.loads(resp.body)
    assert body["code"] == "input_not_supported"

    # The SAME engine 404 on a GET poll is job absence, not an input capability
    # verdict: preserve the status/code but never relabel it input_not_supported.
    poll = Request({
        "type": "http", "method": "GET", "path": "/api/claudexor/login/j1",
        "headers": [], "query_string": b"", "path_params": {"job_id": "j1"},
    })
    polled = asyncio.run(api_claudexor_login_job(poll))
    assert polled.status_code == 404
    assert json.loads(polled.body)["code"] == "http_404"

    class Down:
        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            raise gw.ClaudexorUnavailable("daemon_unreachable", "gone")

        def setup_job_call(self, job_id, op, *, value=""):  # pragma: no cover - unreached
            raise AssertionError

    monkeypatch.setattr(gw, "ClaudexorGateway", Down)
    down = asyncio.run(api_claudexor_login_job(_input_request("j1", {"value": "x"})))
    assert down.status_code == 503 and b"daemon_unreachable" in down.body


def test_login_input_409_conflicts_ride_through_typed(monkeypatch, tmp_path):
    """The engine's TYPED input conflicts (final 3.3.7 contract) pass through
    verbatim as 409 + code — setup_input_not_applicable (the callback already
    completed; no code needed) and setup_input_already_submitted (a repeat
    the authoritative server refused). Answers, not failures: the card maps
    the code to friendly copy, so the proxy must not collapse them into 503."""
    import asyncio

    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway.claudexor_accounts import api_claudexor_login_job
    from ouroboros.gateways import claudexor as gw

    class Conflicted:
        code = "setup_input_not_applicable"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def setup_job_call(self, job_id, op, *, value=""):
            raise gw.ClaudexorUnavailable(
                Conflicted.code, "input refused for this flow/phase", status_code=409)

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", Conflicted)

    for code in ("setup_input_not_applicable", "setup_input_already_submitted"):
        Conflicted.code = code
        resp = asyncio.run(api_claudexor_login_job(_input_request("j1", {"value": "x"})))
        assert resp.status_code == 409, code
        body = json.loads(resp.body)
        assert body["code"] == code


def test_unified_accounts_capability_reads_the_operations_catalog():
    """The unified-account-model marker is the EXACT catalog id of the engine's
    new `GET /v2/account-pools` operation (frozen contract §L.2) — never the
    path spelling, and an unreadable catalog (spelled `[]` by the caller) is
    the old model. Same discipline as `_login_disclosure_native`."""
    from ouroboros.gateway.claudexor_accounts import _unified_accounts_native

    by_id = [{"id": "get:quota"}, {"id": "get:account-pools", "path": "/v2/account-pools"}]
    assert _unified_accounts_native(by_id) is True
    # The id alone is sufficient; the path alone is NOT the marker.
    assert _unified_accounts_native([{"id": "get:account-pools"}]) is True
    assert _unified_accounts_native(
        [{"id": "get:something-else", "path": "/v2/account-pools"}]) is False
    assert _unified_accounts_native([{"id": "get:quota"}]) is False
    assert _unified_accounts_native([]) is False
    assert _unified_accounts_native([None, "get:account-pools"]) is False

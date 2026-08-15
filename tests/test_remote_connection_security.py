"""Owner-only remote-connection SURFACE (RWS v2, D6): the HTTP half.

Trust boundary (D6, documented — NOT enforced by command blacklists): the
connection store lives inside the operator's OS account, so a process running AS
that account can physically read any file the account can. The donor's shell
command blacklists (`OWNER_CONNECTIONS_SELF_CALL_BLOCKED` for `curl`/`ouroboros
connections …` command lines and `OWNER_CONNECTION_STATE_READ_BLOCKED` for
`cat`/`cp`/`rg` over the store path) are security theater over that boundary and
are deliberately NOT transferred — see the ``ouroboros/connection_store.py``
module docstring and docs/ARCHITECTURE.md §4.

What IS pinned is the deterministic guard set on the SURFACE side:
- the owner namespace needs owner authentication even on loopback,
  and its predicate does not fall for percent-encoded or lookalike paths;
- the store file is unreachable through the structured file tools and through
  the gateway file browser, and the API projects metadata ROWS rather than
  handing out the file or its path;
- the eight routes keep durable store metadata and live broker projections
  separate, and a broker projection can never manufacture Home's own
  bootstrap/health evidence.

Store schema/lock/atomic-writer behaviour is pinned in
``tests/test_connection_store.py`` (the store module owns it); this file never
re-tests it.
"""

import json
import os

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.connection_store import add_connection, get_connection
from ouroboros.gateway.connections import (
    api_connection_bootstrap,
    api_connection_dirs,
    api_connection_reconnect,
    api_connection_retire,
    api_connection_retrust,
    api_connection_test,
    api_connections_add,
    api_connections_list,
)
from ouroboros.workspace_diagnostics import RemoteWorkspaceError


class _FakeRemoteService:
    """Stands in for the Lane 1 broker (``remote_workspace``) service object."""

    def __init__(self):
        self.host_id = "host-a"
        self.cancelled = []
        self.reconnects = 0

    def status(self, connection_id=None):
        return {"connections": []}

    def test_connection(self, connection, timeout_sec=10):
        return {
            "ok": True,
            "status": "ready",
            "phase": "connect",
            "host_id": self.host_id,
            "system": "Linux",
            "machine": "x86_64",
        }

    def bootstrap(self, connection, timeout_sec=30):
        return {
            "ok": True,
            "status": "ready",
            "phase": "bootstrap",
            "completion": "installed",
            "host_id": self.host_id,
            "system": "Linux",
            "machine": "x86_64",
            "build": "execd-test-build",
        }

    def reconnect_connection(self, connection, timeout_sec=120):
        self.reconnects += 1
        return {
            "ok": True,
            "status": "ready",
            "phase": "reconcile",
            "completion": "reconciled",
            "host_id": self.host_id,
            "sessions": 1,
        }

    def list_directories(self, connection, remote_root="", timeout_sec=10):
        return {
            "ok": True,
            "path": remote_root or "/srv",
            "parent": "/" if remote_root else "",
            "dirs": [{"name": "repo", "path": "/srv/repo", "is_git": True}],
        }

    def has_active_lease(self, connection_id):
        return False

    def cancel_connection(self, connection_id):
        self.cancelled.append(connection_id)
        return 0


def _connection_routes() -> list[Route]:
    return [
        Route("/api/owner/connections", api_connections_list, methods=["GET"]),
        Route("/api/owner/connections", api_connections_add, methods=["POST"]),
        Route(
            "/api/owner/connections/{connection_id}/test",
            api_connection_test,
            methods=["POST"],
        ),
        Route(
            "/api/owner/connections/{connection_id}/bootstrap",
            api_connection_bootstrap,
            methods=["POST"],
        ),
        Route(
            "/api/owner/connections/{connection_id}/reconnect",
            api_connection_reconnect,
            methods=["POST"],
        ),
        Route(
            "/api/owner/connections/{connection_id}/retrust",
            api_connection_retrust,
            methods=["POST"],
        ),
        Route(
            "/api/owner/connections/{connection_id}/dirs",
            api_connection_dirs,
            methods=["GET"],
        ),
        Route(
            "/api/owner/connections/{connection_id}",
            api_connection_retire,
            methods=["DELETE"],
        ),
    ]


def _connection_app(tmp_path, service):
    app = Starlette(routes=_connection_routes())
    app.state.remote_connections_path = tmp_path / "remote_connections.json"
    if service is not None:
        app.state.remote_workspace_service = service
    return app


def test_eight_connection_routes_keep_metadata_and_live_state_separate(tmp_path):
    service = _FakeRemoteService()
    store = tmp_path / "remote_connections.json"
    client = TestClient(_connection_app(tmp_path, service))
    added = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    )
    assert added.status_code == 201
    connection_id = added.json()["connection"]["id"]

    # A broker projection is not allowed to manufacture the Home-owned
    # bootstrap/health evidence used by the Project picker (durable compatibility
    # from the owner store + process-local freshness).
    service.status = lambda requested_id=None, _cid=connection_id: {
        "connections": [{
            "connection_id": _cid,
            "status": "ready",
            "bootstrap_compatible": True,
            "health_fresh": True,
        }]
    }
    unbootstrapped = client.get("/api/owner/connections").json()["connections"]
    assert unbootstrapped[0]["bootstrap_compatible"] is False
    assert unbootstrapped[0]["health_fresh"] is False

    tested = client.post(f"/api/owner/connections/{connection_id}/test")
    assert tested.status_code == 200
    assert tested.json()["status"] == "ready"
    assert tested.json()["platform"] == "Linux"
    assert tested.json()["architecture"] == "x86_64"
    assert tested.json()["bootstrap_compatible"] is False
    assert tested.json()["health_fresh"] is False
    # A plain reachability probe never pins trust — only bootstrap does.
    assert get_connection(connection_id, store)["expected_host_id"] == ""

    bootstrapped = client.post(f"/api/owner/connections/{connection_id}/bootstrap")
    assert bootstrapped.status_code == 200
    assert bootstrapped.json()["completion"] == "installed"
    assert bootstrapped.json()["bootstrap_compatible"] is True
    assert bootstrapped.json()["health_fresh"] is True
    assert get_connection(connection_id, store)["expected_host_id"] == "host-a"

    reconnected = client.post(f"/api/owner/connections/{connection_id}/reconnect")
    assert reconnected.status_code == 200
    assert reconnected.json()["completion"] == "reconciled"
    assert service.reconnects == 1

    dirs = client.get(f"/api/owner/connections/{connection_id}/dirs?path=%2Fsrv")
    assert dirs.status_code == 200
    assert dirs.json()["dirs"][0]["path"] == "/srv/repo"

    listed = client.get("/api/owner/connections").json()["connections"]
    assert listed[0]["ssh_alias"] == "build"
    assert listed[0]["bootstrap_compatible"] is True
    assert listed[0]["health_fresh"] is True
    assert listed[0]["status"] == "ready"
    assert "password" not in json.dumps(listed)

    service.host_id = "host-b"
    retrusted = client.post(
        f"/api/owner/connections/{connection_id}/retrust",
        json={"confirm": True, "old_host_id": "host-a", "new_host_id": "host-b"},
    )
    assert retrusted.status_code == 200
    assert retrusted.json()["connection"]["expected_host_id"] == "host-b"
    # Retrust invalidates the bootstrap evidence, durable half included: a NEW
    # host identity has never been proven compatible.
    assert retrusted.json()["bootstrap_compatible"] is False
    assert retrusted.json()["health_fresh"] is False

    retired = client.delete(f"/api/owner/connections/{connection_id}")
    assert retired.status_code == 200
    assert retired.json()["connection"]["lifecycle"] == "retired"
    assert service.cancelled == [connection_id]

    retrust_retired = client.post(
        f"/api/owner/connections/{connection_id}/retrust",
        json={"confirm": True, "old_host_id": "host-b", "new_host_id": "host-c"},
    )
    assert retrust_retired.status_code == 409
    assert retrust_retired.json()["error_code"] == "connection_retired"


def test_retrust_requires_explicit_confirmation_and_a_fresh_identity_pair(tmp_path):
    service = _FakeRemoteService()
    client = TestClient(_connection_app(tmp_path, service))
    connection_id = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    ).json()["connection"]["id"]
    client.post(f"/api/owner/connections/{connection_id}/bootstrap")

    unconfirmed = client.post(f"/api/owner/connections/{connection_id}/retrust", json={})
    assert unconfirmed.status_code == 400
    assert unconfirmed.json()["error_code"] == "confirmation_required"

    service.host_id = "host-b"
    stale = client.post(
        f"/api/owner/connections/{connection_id}/retrust",
        json={"confirm": True, "old_host_id": "host-a", "new_host_id": "host-a"},
    )
    assert stale.status_code == 409
    assert stale.json()["error_code"] == "host_identity_confirmation_stale"
    store = tmp_path / "remote_connections.json"
    assert get_connection(connection_id, store)["expected_host_id"] == "host-a"


def test_bootstrap_against_a_changed_host_identity_refuses_and_keeps_the_pin(tmp_path):
    service = _FakeRemoteService()
    client = TestClient(_connection_app(tmp_path, service))
    connection_id = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    ).json()["connection"]["id"]
    client.post(f"/api/owner/connections/{connection_id}/bootstrap")

    service.host_id = "host-impostor"
    changed = client.post(f"/api/owner/connections/{connection_id}/test")

    assert changed.status_code == 409
    assert changed.json()["error_code"] == "host_identity_changed"
    # The vocabulary's spelling, not this seam's own: the gateway said `retrust` here
    # while everything else that means the same action says `retrust_host`
    # (`remote_refusal_actions`), and an owner cannot tell two names for one action
    # apart from two different actions.
    assert changed.json()["action"] == "retrust_host"
    store = tmp_path / "remote_connections.json"
    assert get_connection(connection_id, store)["expected_host_id"] == "host-a"


def test_connection_failure_keeps_typed_stderr_and_completion(tmp_path):
    """The projection runs over the REAL typed error, not a look-alike.

    The gateway reads `code`/`phase`/`completion` and calls `diagnostic()`; this
    drives the actual `RemoteWorkspaceError` through it, so the projection cannot
    pass against a stand-in whose shape has quietly drifted from the transport's.
    """

    class BrokenService(_FakeRemoteService):
        def test_connection(self, connection, timeout_sec=10):
            raise RemoteWorkspaceError(
                "ssh_command_failed",
                "Remote SSH operation failed.",
                phase="connect",
                completion="not_started",
                retryable=True,
                details={"stderr": "Permission denied (publickey)."},
            )

    client = TestClient(_connection_app(tmp_path, BrokenService()))
    connection_id = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    ).json()["connection"]["id"]

    failed = client.post(f"/api/owner/connections/{connection_id}/test")

    assert failed.status_code == 503
    payload = failed.json()
    assert payload["status"] == "degraded"
    assert payload["error_code"] == "ssh_command_failed"
    assert payload["completion"] == "not_started"
    assert (
        payload["diagnostic"]["details"]["stderr"]
        == "Permission denied (publickey)."
    )


def test_reconnect_without_known_project_session_is_typed_failure(tmp_path):
    class DisconnectedService(_FakeRemoteService):
        def reconnect_connection(self, connection, timeout_sec=120):
            return {
                "ok": False,
                "status": "disconnected",
                "phase": "reconcile",
                "completion": "not_started",
                "error_code": "remote_session_disconnected",
                "action": "readmit_project",
            }

    client = TestClient(_connection_app(tmp_path, DisconnectedService()))
    connection_id = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    ).json()["connection"]["id"]

    failed = client.post(f"/api/owner/connections/{connection_id}/reconnect")

    assert failed.status_code == 503
    assert failed.json()["status"] == "disconnected"
    assert failed.json()["action"] == "readmit_project"


def test_transport_dependent_routes_fail_typed_while_no_broker_runs(tmp_path):
    """With no broker in this server, every transport-dependent route refuses TYPED.

    This is the condition that actually occurs now that the transport is in the
    build: the module imports, but no broker was registered (a server whose
    lifespan has not started one, or one whose startup failed). The answer must be
    a typed refusal with a restart hint — never a silent no-op, and never a
    destructive mutation that only LOOKS safe because the lease check found
    nothing to protect. The store-only list/add routes keep working.
    """

    client = TestClient(_connection_app(tmp_path, None))
    connection_id = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    ).json()["connection"]["id"]
    assert client.get("/api/owner/connections").status_code == 200

    for method, path in (
        ("POST", f"/api/owner/connections/{connection_id}/test"),
        ("POST", f"/api/owner/connections/{connection_id}/bootstrap"),
        ("POST", f"/api/owner/connections/{connection_id}/reconnect"),
        ("GET", f"/api/owner/connections/{connection_id}/dirs"),
        ("DELETE", f"/api/owner/connections/{connection_id}"),
    ):
        response = client.request(method, path)
        assert response.status_code == 503, path
        assert response.json()["error_code"] == "remote_service_unavailable", path
        assert response.json()["action"] == "restart_ouroboros", path
    # The refusal is structural, not destructive: the row is still active.
    store = tmp_path / "remote_connections.json"
    assert get_connection(connection_id, store)["lifecycle"] == "active"


def test_live_connection_projection_bounds_and_scrubs_diagnostics():
    from ouroboros.gateway.connections import _public_live_fields

    projected = _public_live_fields({
        "status": "degraded",
        "diagnostic": {
            "message": "api_key=super-secret-value-1234 " + ("x" * 9000),
            "details": {"stderr": "permission denied"},
        },
        "log_refs": [
            {"stream": "stderr", "blob_id": f"log-{index}"} for index in range(100)
        ],
        "warnings": [
            {"code": "ssh_alias_forwarding_neutralized", "directives": ["localforward"]}
            for _index in range(100)
        ],
    })

    assert projected["status"] == "degraded"
    assert "super-secret-value-1234" not in projected["diagnostic"]["message"]
    assert "REDACTED" in projected["diagnostic"]["message"]
    assert len(projected["diagnostic"]["message"]) == 4000
    assert projected["diagnostic"]["details"]["stderr"] == "permission denied"
    assert len(projected["log_refs"]) == 32
    assert len(projected["warnings"]) == 32
    assert projected["warnings"][0]["code"] == "ssh_alias_forwarding_neutralized"


@pytest.mark.parametrize(
    "path",
    [
        "/api/owner/connections",
        "/api/owner/connections/conn-1/test",
        "/api%2Fowner%2Fconnections%2Fconn-1%2Fdirs",
        "%252Fapi%252Fowner%252Fconnections%252Fconn-1",
        "http://127.0.0.1:8765/api/owner/connections",
    ],
)
def test_owner_connections_path_predicate_decodes_bounded_aliases(path):
    from ouroboros.server_auth import is_owner_connections_path

    assert is_owner_connections_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/api/owner/connectionsevil",
        "/api/owner/connections-archive",
        "/api/owner/skills/connections",
    ],
)
def test_owner_connections_path_predicate_rejects_lookalikes(path):
    from ouroboros.server_auth import is_owner_connections_path

    assert not is_owner_connections_path(path)


def test_owner_connection_namespace_requires_auth_even_on_loopback(monkeypatch):
    import ouroboros.server_auth as server_auth

    inner = Starlette(routes=[
        Route("/api/owner/connections", lambda request: JSONResponse({"ok": True})),
        Route(
            "/api/owner/connectionsevil",
            lambda request: JSONResponse({"lookalike": True}),
        ),
    ])
    client = TestClient(server_auth.NetworkAuthGate(inner), client=("127.0.0.1", 12345))

    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "")
    missing = client.get("/api/owner/connections")
    assert missing.status_code == 503
    assert missing.json()["error_code"] == "owner_auth_not_configured"
    assert client.get("/api/owner/connectionsevil").status_code == 200

    monkeypatch.setattr(
        server_auth, "get_configured_network_password", lambda: "owner-secret"
    )
    required = client.get("/api/owner/connections")
    assert required.status_code == 401
    assert required.json()["error_code"] == "owner_auth_required"
    wrong = client.get(
        "/api/owner/connections", headers={"X-Ouroboros-Password": "wrong"}
    )
    assert wrong.status_code == 401
    assert wrong.json() == required.json()
    # A malformed body must be refused by AUTH, never parsed by a handler first.
    malformed = client.post(
        "/api/owner/connections",
        content=b"{not-json",
        headers={"content-type": "application/json"},
    )
    assert malformed.status_code == 401
    assert malformed.json() == required.json()
    accepted = client.get(
        "/api/owner/connections", headers={"X-Ouroboros-Password": "owner-secret"}
    )
    assert accepted.status_code == 200


def test_connection_store_is_unreachable_through_file_tools_gateway_and_api(
    tmp_path,
    monkeypatch,
):
    """The D6 replacement for the donor's shell blacklists.

    The OS account is the documented trust boundary, so nothing here pretends to
    stop a determined same-account process. What is pinned is that no ORDINARY
    surface hands the store over by accident: the structured data tools refuse
    it, the gateway file browser treats it as owner-only state (so it is hidden
    from listings and refused for read/write/delete/transfer), and the owner API
    projects metadata rows rather than the file or its location.
    """

    from ouroboros import config
    from ouroboros.gateway import files as gateway_files
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    store = drive / "state" / "remote_connections.json"
    add_connection(name="Build host", ssh_alias="build", path=store)
    monkeypatch.setattr(config, "REMOTE_CONNECTIONS_PATH", store)

    # 1. Structured file tools (the store module's own denial predicate).
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    assert "DATA_READ_BLOCKED" in _read_file(
        ctx, "state/remote_connections.json", root="runtime_data"
    )

    # 2. Gateway file browser: owner-only, including the lock/temp/hardlink
    #    aliases and any parent directory a recursive operation would sweep.
    alias = store.with_name("connection-alias.json")
    os.link(store, alias)
    assert gateway_files._is_owner_only_file(store) is True
    assert gateway_files._is_owner_only_file(alias) is True
    assert (
        gateway_files._is_owner_only_file(store.with_name(store.name + ".lock"))
        is True
    )
    assert gateway_files._contains_owner_only_file(store.parent) is True

    # 3. Owner API: rows only. The response never carries the store location,
    #    so a UI/CLI consumer has no path to hand to a file tool.
    service = _FakeRemoteService()
    app = Starlette(routes=_connection_routes())
    app.state.remote_connections_path = store
    app.state.remote_workspace_service = service
    client = TestClient(app)
    body = client.get("/api/owner/connections").text
    assert "build" in body
    assert "remote_connections.json" not in body
    assert str(store) not in body


def test_a_raised_reconnect_stays_inside_the_contract_status_vocabulary(tmp_path):
    """The producer, the contract, and every consumer must use ONE vocabulary.

    ``remote_worker_proxy.reconnect_failure`` used to answer ``status="error"``, a
    sixth word no consumer knows: ``_record_runtime_health`` ignores it, and the
    browser reducer reads an unknown status as *no typed status at all* and falls
    back to derivation — which hides the Reconnect button exactly when a reconnect
    is the fix. It also shipped ``diagnostic`` as a STRING while the contract types
    it ``Dict[str, Any]``, so all three layers that carry the field dropped the
    reason that had just been computed and sent.
    """

    from typing import get_args, get_type_hints

    from ouroboros.gateway.contracts import ConnectionStateOutbound
    from ouroboros.remote_worker_proxy import reconnect_failure
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    # NotRequired[Literal[...]] — unwrap one level, so the vocabulary is READ from
    # the contract instead of being restated in the test.
    status_hint = get_type_hints(ConnectionStateOutbound, include_extras=True)["status"]
    contract_states = set(get_args(get_args(status_hint)[0]))
    assert contract_states == {
        "connecting", "ready", "degraded", "disconnected", "unknown",
    }

    raised = reconnect_failure(
        "conn_1",
        RemoteWorkspaceError(
            "ssh_eof",
            "Remote SSH operation failed.",
            phase="reconcile",
            completion="not_started",
            retryable=True,
            details={"stderr": "connection closed"},
        ),
    )
    absent = reconnect_failure("conn_1")
    for row in (raised, absent):
        assert row["status"] in contract_states, row["status"]
        assert isinstance(row["diagnostic"], dict), row["diagnostic"]
    assert raised["error_code"] == "ssh_eof"
    assert raised["diagnostic"]["message"]
    assert absent["error_code"] == "remote_session_absent"

    # And the whole way to the wire: the gateway keeps the mapping (it drops a
    # non-dict diagnostic), so the owner can read WHY the reconnect failed.
    class RaisingService(_FakeRemoteService):
        def reconnect_connection(self, connection, timeout_sec=120):
            return reconnect_failure(
                str(connection.get("id") or ""),
                RemoteWorkspaceError(
                    "ssh_eof",
                    "Remote SSH operation failed.",
                    phase="reconcile",
                    completion="not_started",
                    retryable=True,
                    details={"stderr": "connection closed"},
                ),
            )

    client = TestClient(_connection_app(tmp_path, RaisingService()))
    connection_id = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    ).json()["connection"]["id"]
    failed = client.post(f"/api/owner/connections/{connection_id}/reconnect")
    assert failed.status_code == 503
    payload = failed.json()
    assert payload["status"] == "degraded"
    assert payload["error_code"] == "ssh_eof"
    assert payload["diagnostic"]["details"]["stderr"] == "connection closed"


def test_one_normalization_gives_every_typed_code_a_single_wire_spelling(tmp_path):
    """``REMOTE_TRANSPORT_UNAVAILABLE`` vs ``remote_transport_unavailable``.

    The admission authority spells its code as a Python CONSTANT while the
    transport already spells it lower-case, and both reach the same browser
    reducer and the same CLI exit-code sets — which compare case-SENSITIVELY. The
    normalization therefore lives ONCE, at the gateway boundary
    (``gateway/_helpers.py::wire_error_code``); a ``.lower()`` per response site is
    how one of the two spellings escaped in the first place.
    """

    from ouroboros.gateway._helpers import wire_error_code
    from ouroboros.workspace_admission import REMOTE_TRANSPORT_UNAVAILABLE

    assert REMOTE_TRANSPORT_UNAVAILABLE.isupper()
    assert wire_error_code(REMOTE_TRANSPORT_UNAVAILABLE) == "remote_transport_unavailable"
    assert wire_error_code("  Mixed_Case  ") == "mixed_case"
    assert wire_error_code(None) == ""

    class ShoutingService(_FakeRemoteService):
        def test_connection(self, connection, timeout_sec=10):
            return {
                "ok": False,
                "status": "degraded",
                "phase": "connect",
                "error_code": REMOTE_TRANSPORT_UNAVAILABLE,
            }

    client = TestClient(_connection_app(tmp_path, ShoutingService()))
    connection_id = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    ).json()["connection"]["id"]
    refused = client.post(f"/api/owner/connections/{connection_id}/test")
    # Lower-case on the wire, so `remote_task_state.js::isRemoteTransportUnavailable`
    # and `cli_connections._UNSERVABLE_CODES` both recognise it.
    assert refused.json()["error_code"] == "remote_transport_unavailable"


def test_a_reconnect_with_nothing_admitted_is_not_a_generic_refusal():
    """Exit 4, not 2: no retry and no owner credential will produce a session.

    ``ouroboros connections reconnect`` answers 503 with
    ``remote_session_disconnected``/``remote_session_absent``. Neither code was in
    the enumerated unservable set, so a script saw the same exit 2 it gets for an
    ordinary decline and could not tell "nothing is admitted here" from "refused".
    """

    from ouroboros.cli_connections import _OWNER_ACTION_CODES, _UNSERVABLE_CODES

    for code in ("remote_session_disconnected", "remote_session_absent"):
        assert code in _UNSERVABLE_CODES
        assert code not in _OWNER_ACTION_CODES
    # Every member is already in the wire spelling the gateway now guarantees.
    assert all(code == code.lower() for code in _UNSERVABLE_CODES | _OWNER_ACTION_CODES)


def test_a_bootstrapped_connection_is_still_bootstrapped_after_a_restart(tmp_path):
    """The picker must not go blind because Ouroboros restarted.

    "A compatible remote executor is installed on that host" is a fact about the
    HOST, established by an owner action — it does not stop being true when Home
    restarts. It used to live only in `_RUNTIME_EVIDENCE`, a module-level dict, and
    `_record_runtime_health` early-returned unless a bootstrap had already been seen
    in the SAME process. So after a restart every connection read back
    `bootstrap_compatible: false`, "New Project → SSH" was permanently empty, and the
    dialog's own copy ("run Bootstrap, or Test to refresh health") described a
    recovery that could not work: Test could never restore a flag that only existed
    if a bootstrap had happened since start.

    Health FRESHNESS is the half that is legitimately process-local, so a restart
    does clear it — and one Test brings it back, which is exactly what the copy says.
    """

    from ouroboros.gateway import connections as gateway_connections

    service = _FakeRemoteService()
    store = tmp_path / "remote_connections.json"
    client = TestClient(_connection_app(tmp_path, service))
    connection_id = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    ).json()["connection"]["id"]

    bootstrapped = client.post(f"/api/owner/connections/{connection_id}/bootstrap")
    assert bootstrapped.json()["bootstrap_compatible"] is True
    assert bootstrapped.json()["health_fresh"] is True
    # The durable half is in the OWNER STORE, where owner state belongs.
    row = get_connection(connection_id, store)
    assert row["bootstrapped_at"]
    assert row["bootstrap_build"] == "execd-test-build"

    # A restart: the process-local half is all that is lost.
    gateway_connections._RUNTIME_EVIDENCE.clear()
    after_restart = client.get("/api/owner/connections").json()["connections"][0]
    assert after_restart["bootstrap_compatible"] is True
    assert after_restart["health_fresh"] is False

    # And Test — not another Bootstrap — is enough to make it selectable again,
    # which is what `project_create.js::remoteConnectionChoices` requires and what
    # the dialog's copy promises.
    tested = client.post(f"/api/owner/connections/{connection_id}/test")
    assert tested.status_code == 200
    assert tested.json()["bootstrap_compatible"] is True
    assert tested.json()["health_fresh"] is True
    assert tested.json()["status"] == "ready"

    # Retrust invalidates the durable claim too: a DIFFERENT host has never been
    # proven compatible with this build.
    service.host_id = "host-b"
    client.post(
        f"/api/owner/connections/{connection_id}/retrust",
        json={"confirm": True, "old_host_id": "host-a", "new_host_id": "host-b"},
    )
    assert get_connection(connection_id, store)["bootstrapped_at"] is None
    assert client.get("/api/owner/connections").json()["connections"][0][
        "bootstrap_compatible"
    ] is False


def test_a_broker_projection_still_cannot_manufacture_durable_bootstrap_evidence(tmp_path):
    """The separation the durable move must not weaken.

    `api_connections_list` layers Home's evidence LAST, so a broker `status()` row
    claiming `bootstrap_compatible: true` for a connection that was never
    bootstrapped is overwritten rather than trusted.
    """

    service = _FakeRemoteService()
    client = TestClient(_connection_app(tmp_path, service))
    connection_id = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    ).json()["connection"]["id"]
    service.status = lambda requested_id=None, _cid=connection_id: {
        "connections": [{
            "connection_id": _cid,
            "status": "ready",
            "bootstrap_compatible": True,
            "health_fresh": True,
            "build": "smuggled-build",
        }]
    }

    listed = client.get("/api/owner/connections").json()["connections"][0]
    assert listed["bootstrap_compatible"] is False
    assert listed["health_fresh"] is False
    # A retired connection is never compatible either, whatever the store remembers.
    client.post(f"/api/owner/connections/{connection_id}/bootstrap")
    client.delete(f"/api/owner/connections/{connection_id}")
    retired = client.get("/api/owner/connections").json()["connections"][0]
    assert retired["lifecycle"] == "retired"
    assert retired["bootstrap_compatible"] is False


@pytest.mark.serial  # mutates the module-global supervisor queue (docs/DEVELOPMENT.md)
def test_a_connection_state_frame_names_the_tasks_it_concerns(tmp_path, monkeypatch):
    """M6: the task-scoped frame the whole Chat remote surface was waiting for.

    `ConnectionStateOutbound` declares `task_id`/`project_id`, and both browser
    consumers were written for them — but no producer ever set either. Chat fills
    `remoteTaskStates` ONLY from these frames, so the map stayed empty forever and
    its remote card, per-task Reconnect/Cancel, and typing-indicator suppression
    never rendered once. (The Activity subtab escaped only because it seeds its map
    from durable task rows, giving a connection-wide frame something to fan out to.)

    Home does know the relation: the queue holds each task's SEALED placement, and
    the same walk already answers "is this connection busy?" for the owner
    mutations. So the broadcast emits the connection-wide frame first — Settings →
    Connections needs it, and it is the only frame when nothing is running — then one
    task-scoped frame per live task on that connection, carrying the `project_id`
    Chat needs to scope the frame to its own thread.
    """

    from supervisor import queue as supervisor_queue

    frames: list[dict] = []

    class _Bridge:
        def broadcast(self, payload):
            frames.append(dict(payload))

    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: _Bridge())

    service = _FakeRemoteService()
    client = TestClient(_connection_app(tmp_path, service))
    connection_id = client.post(
        "/api/owner/connections",
        json={"name": "Build host", "ssh_alias": "build"},
    ).json()["connection"]["id"]

    # With nothing queued, the connection-wide frame is the only one — and it names
    # no task, because inventing one would be a lie about scope.
    client.post(f"/api/owner/connections/{connection_id}/test")
    assert len(frames) == 1
    assert frames[0]["type"] == "connection_state"
    assert frames[0]["connection_id"] == connection_id
    assert "task_id" not in frames[0]

    # A real remote task in the queue, with the placement sealed the way admission
    # seals it — so the walk reads the same authority the fences do.
    remote_task = {
        "id": "task-7",
        "project_id": "remote-app",
        "metadata": {
            "_sealed_workspace_ref": {
                "kind": "ssh",
                "connection_id": connection_id,
                "remote_root": "/srv/work/app",
                "workspace_id": "ws-1",
            },
        },
    }
    local_task = {"id": "task-8", "workspace_root": str(tmp_path)}
    other_connection_task = {
        "id": "task-9",
        "project_id": "elsewhere",
        "metadata": {
            "_sealed_workspace_ref": {
                "kind": "ssh",
                "connection_id": "conn_other",
                "remote_root": "/srv/other",
                "workspace_id": "ws-2",
            },
        },
    }
    frames.clear()
    with supervisor_queue._queue_lock:
        supervisor_queue.PENDING.extend([remote_task, local_task, other_connection_task])
    try:
        client.post(f"/api/owner/connections/{connection_id}/reconnect")
    finally:
        with supervisor_queue._queue_lock:
            for task in (remote_task, local_task, other_connection_task):
                supervisor_queue.PENDING.remove(task)

    assert [frame.get("task_id", "") for frame in frames] == ["", "task-7"]
    scoped = frames[1]
    assert scoped["project_id"] == "remote-app"
    assert scoped["connection_id"] == connection_id
    # The task frame carries the SAME bounded live projection as the wide one, so
    # the two surfaces cannot disagree about what happened.
    assert scoped["status"] == frames[0]["status"] == "ready"
    assert scoped["completion"] == "reconciled"


def test_the_queue_walk_answers_busy_and_fan_out_from_one_traversal(monkeypatch):
    """Fail CLOSED for the owner mutation, fail EMPTY for the broadcast.

    The two readings of "which tasks are live on this connection" must come from ONE
    walk, but they disagree about the ambiguous case: a destructive retrust/retire
    must treat a lookup failure as BUSY, while a live broadcast must not invent task
    frames it could not verify. So the walk returns `None` for "could not tell" and
    each caller resolves it in its own direction.
    """

    from ouroboros.gateway import connections as gateway_connections

    assert gateway_connections._live_connection_tasks("conn_absent") == []
    assert gateway_connections._connection_busy(_FakeRemoteService(), "conn_absent") is False

    class _Exploding:
        def has_active_lease(self, connection_id):
            raise AssertionError("must not be consulted when the queue walk failed")

    monkeypatch.setattr(gateway_connections, "_live_connection_tasks", lambda _cid: None)
    assert gateway_connections._connection_busy(_Exploding(), "conn_1") is True


def test_a_bounded_collection_never_shrinks_silently(tmp_path):
    """A cap that leaves no trace reads exactly like data that was never there.

    The live projection bounds `log_refs`/`warnings` to 32 records and bounds every
    collection INSIDE a diagnostic to 32 entries, and it used to do both invisibly:
    the owner could not tell "four warnings" from "four of nine", and a diagnostic
    whose details had been trimmed looked complete. Which is the worst ambiguity to
    leave in a forensic payload — the one thing an operator reads it for is what is
    NOT there.

    So each bounded list ships its pre-cap TOTAL (one number, not a number plus a
    flag that could disagree with it: truncation IS `count > len(list)`), and each
    bounded collection inside a diagnostic keeps a visible omission entry, the way
    the depth guard already announced itself as `[truncated]`.
    """

    from ouroboros.gateway.connections import _LIVE_COLLECTION_LIMIT, _public_live_fields

    over = _LIVE_COLLECTION_LIMIT + 7
    projected = _public_live_fields({
        "status": "degraded",
        "warnings": [{"code": f"w{index}"} for index in range(over)],
        "log_refs": [{"stream": f"s{index}"} for index in range(over)],
        "diagnostic": {
            "code": "ssh_eof",
            "details": {
                "env": {f"KEY_{index}": index for index in range(over)},
                "lines": [f"line {index}" for index in range(over)],
            },
        },
    })
    assert len(projected["warnings"]) == _LIVE_COLLECTION_LIMIT
    assert projected["warnings_count"] == over
    assert len(projected["log_refs"]) == _LIVE_COLLECTION_LIMIT
    assert projected["log_refs_count"] == over

    details = projected["diagnostic"]["details"]
    # The trimmed mapping keeps a marker key; the trimmed list keeps a last row.
    assert "…" in details["env"]
    assert f"of {over} entries omitted" in details["env"]["…"]
    assert len(details["lines"]) == _LIVE_COLLECTION_LIMIT + 1
    assert f"of {over} entries omitted" in details["lines"][-1]

    # Nothing is added when nothing was cut: a complete list stays exactly itself.
    intact = _public_live_fields({
        "status": "ready",
        "warnings": [{"code": "only"}],
        "diagnostic": {"details": {"stderr": "boom"}},
    })
    assert intact["warnings_count"] == 1
    assert intact["diagnostic"]["details"] == {"stderr": "boom"}
    assert "log_refs" not in intact and "log_refs_count" not in intact


def test_the_three_observed_host_id_readers_look_in_the_same_three_places():
    """Retrust is assembled from a field all three surfaces must be able to see.

    `gateway/connections.py::_observed_host_id` had a fourth arm reading
    `admission_evidence`, which NOTHING in the codebase produces — while its browser
    and CLI twins had only three. A dead arm plus a mirror divergence in the one
    lookup a trust decision depends on.
    """

    import pathlib
    import re

    from ouroboros.cli_connections import _observed_cli_host_id
    from ouroboros.gateway.connections import _observed_host_id

    payload = {"handshake": {"host_id": "host-from-handshake"}}
    assert _observed_host_id(payload) == "host-from-handshake"
    assert _observed_cli_host_id(payload) == "host-from-handshake"
    diagnostic_only = {"diagnostic": {"observed_host_id": "host-from-diagnostic"}}
    assert _observed_host_id(diagnostic_only) == "host-from-diagnostic"
    assert _observed_cli_host_id(diagnostic_only) == "host-from-diagnostic"
    # The envelope's own field wins over both, in every reader.
    direct = {"host_id": "host-direct", **payload}
    assert _observed_host_id(direct) == "host-direct"
    assert _observed_cli_host_id(direct) == "host-direct"

    # And the browser twin reads the same three sources — no more, no fewer.
    ui = (
        pathlib.Path(__file__).resolve().parent.parent
        / "web" / "modules" / "connections_ui.js"
    ).read_text(encoding="utf-8")
    body = re.search(r"function observedHostId\(payload\) \{(.*?)\n\}", ui, re.S)
    assert body, "connections_ui.js must keep observedHostId"
    sources = re.findall(r"payload\??\.?(\w+)", body.group(1))
    assert set(sources) == {"handshake", "diagnostic"}, sources
    assert "host_id" in body.group(1) and "observed_host_id" in body.group(1)
    # The arm that read a key nothing produces is gone from BOTH sides.
    assert "admission_evidence" not in ui

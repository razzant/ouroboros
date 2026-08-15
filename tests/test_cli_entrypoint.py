from __future__ import annotations

import ast
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


def test_server_subcommand_sanitizes_argv(monkeypatch):
    from ouroboros import cli

    seen = {}

    class FakeServer:
        @staticmethod
        def main():
            seen["argv"] = list(sys.argv)
            return 0

    monkeypatch.setitem(sys.modules, "server", FakeServer)
    monkeypatch.setattr(sys, "argv", ["ouroboros", "server", "--host", "127.0.0.1", "--port", "9000"])

    result = cli._server_command(SimpleNamespace(host="127.0.0.1", port=9000, no_ui=True))

    assert result == 0
    assert seen["argv"] == ["ouroboros"]
    assert json.loads(__import__("os").environ["OUROBOROS_SERVER_REEXEC_ARGV_JSON"]) == [
        "-m",
        "ouroboros.cli",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        "9000",
    ]
    assert sys.argv == ["ouroboros", "server", "--host", "127.0.0.1", "--port", "9000"]


def test_settings_context_mode_posts_owner_endpoint(monkeypatch):
    from ouroboros import cli

    seen = {}

    class FakeClient:
        def request(self, method, path, body=None):
            seen["request"] = (method, path, body)
            return {"ok": True, "context_mode": body["mode"]}

    monkeypatch.setattr(cli, "_client", lambda _args, **_kwargs: FakeClient())

    result = cli._owner_context_mode_command(SimpleNamespace(mode="low"))

    assert result == 0
    assert seen["request"] == ("POST", "/api/owner/context-mode", {"mode": "low"})


# ── `ouroboros connections …` (RWS v2, D6) ───────────────────────────────────
# The command family lives in ouroboros/cli_connections.py; cli.py keeps only
# the parser wiring, the owner-password header, and GatewayHTTPError.


def test_connections_cli_prompts_before_request_and_never_uses_environment(
    monkeypatch,
    capsys,
):
    from ouroboros import cli, cli_connections

    seen = []
    monkeypatch.setenv("OUROBOROS_NETWORK_PASSWORD", "must-not-be-read")
    monkeypatch.setattr(
        cli_connections,
        "_read_owner_password",
        lambda: seen.append("prompt") or "typed-secret",
    )

    class FakeClient:
        def __init__(self, base_url="", timeout=30.0, *, owner_password=""):
            seen.append(("client", owner_password))

        def request(self, method, path, body=None, **kwargs):
            seen.append((method, path, body))
            return {
                "connections": [
                    {
                        "id": "conn-1",
                        "name": "Build",
                        "ssh_alias": "build",
                        "lifecycle": "active",
                        "status": "ready",
                    }
                ]
            }

    monkeypatch.setattr(cli, "OuroborosHTTPClient", FakeClient)
    assert cli.main(["connections", "list", "--json"]) == 0
    assert seen[0] == "prompt"
    assert seen[1] == ("client", "typed-secret")
    assert "must-not-be-read" not in json.dumps(seen)
    assert '"conn-1"' in capsys.readouterr().out


def test_connections_cli_has_stable_owner_and_conflict_exit_codes(monkeypatch, capsys):
    from ouroboros import cli, cli_connections

    monkeypatch.setattr(cli_connections, "_read_owner_password", lambda: "typed-secret")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            raise cli.GatewayHTTPError(
                503,
                {
                    "error": "Owner authentication is not configured.",
                    "error_code": "owner_auth_not_configured",
                    "action": "configure_network_password",
                },
            )

    monkeypatch.setattr(cli, "OuroborosHTTPClient", FakeClient)
    assert cli.main(["connections", "list", "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "owner_auth_not_configured"

    class ConflictClient(FakeClient):
        def request(self, *args, **kwargs):
            raise cli.GatewayHTTPError(
                409,
                {"error": "active", "error_code": "active_lease"},
            )

    monkeypatch.setattr(cli, "OuroborosHTTPClient", ConflictClient)
    assert cli.main(["connections", "retire", "conn-1", "--json"]) == 5


@pytest.mark.parametrize(
    ("error_code", "action", "message"),
    [
        # the gateway's own three
        ("remote_transport_unavailable", "await_remote_transport", "transport is not available"),
        ("remote_service_unavailable", "restart_ouroboros", "service is unavailable"),
        ("remote_service_invalid_response", "restart_ouroboros", "invalid response"),
        # the broker's
        ("remote_workspace_unavailable", "restart_ouroboros", "broker is not configured"),
        ("bootstrap_unsupported", "", "does not expose bootstrap"),
        ("directory_listing_unsupported", "", "does not expose directory listing"),
        ("reconnect_unsupported", "", "does not expose reconnect"),
        ("broker_closed", "", "broker is closed"),
        # the client / target platform / bundle identity
        ("unsupported_ssh_client", "", "ssh client is unsupported"),
        ("remote_platform_unsupported", "", "platform is unsupported"),
        ("remote_glibc_too_old", "", "glibc is too old"),
        ("execd_bundle_invalid", "", "bundle is invalid"),
        ("execd_artifact_mismatch", "", "artifact does not match"),
        ("capability_mismatch", "", "capability manifests differ"),
    ],
)
def test_connections_cli_maps_every_unservable_code_to_exit_four(
    monkeypatch, capsys, error_code, action, message
):
    """One exit code for the whole "retrying will not help" class.

    A script does not care WHICH layer refused — it cares that no retry and no
    owner action will change the answer, which is exit 4 rather than 2 (refusal) or
    3 (owner action required). Every code the gateway, the broker, the OpenSSH
    client checks and the bundle identity checks can emit for that reason is walked
    here, because a code missing from the set degrades silently to a generic
    refusal and a script would retry forever.
    """

    from ouroboros import cli, cli_connections

    monkeypatch.setattr(cli_connections, "_read_owner_password", lambda: "typed-secret")

    class RefusingClient:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            payload = {"error": message, "error_code": error_code, "phase": "connect"}
            if action:
                payload["action"] = action
            raise cli.GatewayHTTPError(503, payload)

    monkeypatch.setattr(cli, "OuroborosHTTPClient", RefusingClient)
    assert cli.main(["connections", "bootstrap", "conn-1"]) == 4
    stderr = capsys.readouterr().err
    assert message in stderr
    if action:
        assert f"action={action}" in stderr


def test_a_transient_broker_refusal_is_exit_two_so_a_script_may_retry_it(
    monkeypatch, capsys
):
    """`broker_overloaded` used to be exit 4, which promised the opposite of the truth.

    The broker raises it with `retryable=True` for a full request queue or too many
    in-flight requests — Home capacity, which clears on its own — while exit 4 tells a
    script "retrying will not help and no owner action will fix it either". The refusal
    register (`remote_refusal_actions`) names `retry` as the action for this code, and
    the mechanical cross-check in `tests/test_remote_refusal_action_proofs.py` is what
    found the two authorities disagreeing. Exit 2 is the honest answer until the
    taxonomy grows a class of its own for a transient refusal.
    """

    from ouroboros import cli, cli_connections
    from ouroboros.remote_refusal_actions import ACTION_RETRY, REFUSAL_ACTIONS

    assert REFUSAL_ACTIONS["broker_overloaded"] == ACTION_RETRY
    assert "broker_overloaded" not in cli_connections._UNSERVABLE_CODES
    monkeypatch.setattr(cli_connections, "_read_owner_password", lambda: "typed-secret")

    class OverloadedClient:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            raise cli.GatewayHTTPError(503, {
                "error": "broker is overloaded",
                "error_code": "broker_overloaded",
                "phase": "stream",
                "action": ACTION_RETRY,
            })

    monkeypatch.setattr(cli, "OuroborosHTTPClient", OverloadedClient)
    assert cli.main(["connections", "bootstrap", "conn-1"]) == 2
    assert "action=retry" in capsys.readouterr().err


def test_owner_password_refuses_piped_stdin_without_controlling_tty(monkeypatch):
    from ouroboros import cli, cli_connections

    monkeypatch.setattr(cli_connections.os, "name", "posix")

    def no_tty(*args, **kwargs):
        raise OSError("no controlling tty")

    monkeypatch.setattr(cli_connections.os, "open", no_tty)
    try:
        cli_connections._read_owner_password()
    except cli.CLIError as exc:
        assert "controlling terminal" in str(exc)
    else:
        raise AssertionError("piped stdin must not be accepted for owner password")


def test_connections_json_reports_non_tty_auth_without_request(monkeypatch, capsys):
    from ouroboros import cli, cli_connections

    monkeypatch.setattr(
        cli_connections,
        "_read_owner_password",
        lambda: (_ for _ in ()).throw(cli.CLIError("controlling terminal required")),
    )
    monkeypatch.setattr(
        cli,
        "OuroborosHTTPClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("no request is allowed without TTY authentication")
        ),
    )
    assert cli.main(["connections", "list", "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "owner_auth_required"
    assert payload["action"] == "run_from_controlling_terminal"


def test_connections_add_requires_named_flags():
    import pytest

    from ouroboros import cli

    parsed = cli.build_parser().parse_args([
        "connections", "add", "--name", "Build", "--ssh-alias", "build",
    ])
    assert parsed.name == "Build"
    assert parsed.ssh_alias == "build"
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["connections", "add", "Build", "build"])
    assert exc.value.code == 2


def test_connections_parser_exposes_exactly_seven_admin_commands(capsys):
    """The CLI surface is exactly the owner endpoint family — no more, no less.

    ``reconnect`` IS present (unlike the donor CLI): the gateway mounts the
    route, so leaving it out would force an owner to reach for curl.
    """

    import pytest

    from ouroboros import cli

    parser = cli.build_parser()
    root_subparsers = next(
        action for action in parser._actions
        if getattr(action, "dest", None) == "command"
    )
    connections_parser = root_subparsers.choices["connections"]
    connection_subparsers = next(
        action for action in connections_parser._actions
        if getattr(action, "dest", None) == "connections_command"
    )
    assert set(connection_subparsers.choices) == {
        "list", "add", "test", "bootstrap", "reconnect", "retrust", "retire",
    }
    for forbidden in ("status", "remove", "delete", "run", "cancel"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["connections", forbidden])
        assert exc.value.code == 2
    capsys.readouterr()


def test_connections_cli_incompatible_exit_and_human_action(monkeypatch, capsys):
    from ouroboros import cli, cli_connections

    monkeypatch.setattr(cli_connections, "_read_owner_password", lambda: "typed-secret")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            raise cli.GatewayHTTPError(
                503,
                {
                    "error": "remote executor protocol is incompatible",
                    "error_code": "incompatible_protocol",
                    "phase": "handshake",
                    "action": "bootstrap",
                },
            )

    monkeypatch.setattr(cli, "OuroborosHTTPClient", FakeClient)
    assert cli.main(["connections", "test", "conn-1"]) == 4
    stderr = capsys.readouterr().err
    assert "phase=handshake" in stderr
    assert "action=bootstrap" in stderr


def test_connections_cli_human_result_exposes_ssh_alias_warning(capsys):
    from ouroboros import cli_connections

    cli_connections._print_connection_result(
        {
            "connection_id": "conn-1",
            "status": "ready",
            "warnings": [{
                "code": "ssh_alias_forwarding_neutralized",
                "directives": ["localforward"],
            }],
        },
        as_json=False,
    )

    captured = capsys.readouterr()
    assert "status=ready" in captured.out
    assert "ssh_alias_forwarding_neutralized" in captured.err


def test_connections_cli_all_unservable_codes_exit_four(capsys):
    from ouroboros import cli, cli_connections

    for error_code in (
        "remote_transport_unavailable",
        "remote_service_unavailable",
        "remote_service_invalid_response",
        "unsupported_ssh_client",
        "remote_platform_unsupported",
        "remote_libc_unsupported",
        "remote_glibc_too_old",
        "execd_preamble_invalid",
        "execd_release_unselected",
        "execd_bundle_unavailable",
        "execd_bundle_invalid",
        "execd_artifact_mismatch",
        "capability_mismatch",
        "incompatible_protocol",
    ):
        error = cli.GatewayHTTPError(
            503,
            {"error": "remote target is incompatible", "error_code": error_code},
        )
        assert cli_connections._connection_error_exit(error, as_json=True) == 4, error_code
        assert json.loads(capsys.readouterr().out)["error_code"] == error_code


def test_connections_retrust_requires_explicit_tty_confirmation_before_post(monkeypatch):
    from ouroboros import cli, cli_connections

    calls = []
    confirmations = []
    monkeypatch.setattr(cli_connections, "_read_owner_password", lambda: "typed-secret")
    monkeypatch.setattr(
        cli_connections,
        "_confirm_host_retrust",
        lambda old, new: confirmations.append((old, new)) or False,
    )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path, body=None, **kwargs):
            calls.append((method, path, body))
            if path == "/api/owner/connections":
                return {"connections": [{"id": "conn-1", "expected_host_id": "old-host"}]}
            if path.endswith("/test"):
                return {"ok": False, "observed_host_id": "new-host"}
            raise AssertionError("retrust mutation must not happen after decline")

    monkeypatch.setattr(cli, "OuroborosHTTPClient", FakeClient)
    assert cli.main(["connections", "retrust", "conn-1", "--json"]) == 2
    assert confirmations == [("old-host", "new-host")]
    assert all(not path.endswith("/retrust") for _method, path, _body in calls)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_CLI_FORBIDDEN_MODULES = (
    "ouroboros.connection_store",
    "ouroboros.remote_workspace",
    "ouroboros.remote_ssh",
    "ouroboros.gateway.connections",
    "ouroboros.gateway.projects",
    "ouroboros.projects_registry",
    "subprocess",
)
_CLI_ALLOWED_FIRST_PARTY = frozenset({
    "ouroboros.cli",
    "ouroboros.config",
    # A NAME TABLE, not an authority. `remote_refusal_actions` holds the closed
    # vocabulary of owner actions and the code→action register; it decides nothing about
    # the store, the trust rules or the transport, and it imports none of them (stdlib
    # plus two constants, which is what lets it travel into the execd bundle). The CLI
    # reads it for the same reason it must not restate an error code: the alternative is
    # a second spelling of an action the browser already names, and this surface had two
    # such pairs before the register existed.
    "ouroboros.remote_refusal_actions",
    # Not an authority either. platform_layer is a stdlib-only leaf that owns the
    # platform facts — here the POSIX controlling-terminal device NAME, which is the
    # whole platform difference (`os.open` is portable and each call site handles
    # Windows two lines above). It decides nothing about the store, the trust rules
    # or the transport, and `tests/test_platform_guard.py` requires the CLI to reach
    # it: the `/dev/tty` literal cannot be removed any other way.
    "ouroboros.platform_layer",
})


def _cli_module_reaches(source: str) -> set[str]:
    """Every module name this source can reach, in ANY import spelling.

    Covers, and each of these was verified to slip past the previous version, which
    looked only at `ast.Import.names` and `ast.ImportFrom.module`:

    * a RELATIVE import — `from .connection_store import get_connection` gives
      `node.module == "connection_store"`, which passed the forbidden-name check AND
      passed the first-party whitelist because it does not start with "ouroboros".
      Normalised here to its absolute name using `node.level`.
    * `importlib.import_module("ouroboros.connection_store")` and `__import__(...)`.
    * `sys.modules["ouroboros.remote_ssh"]`.
    * a dotted parent: importing `ouroboros.gateway.connections` also records
      `ouroboros.gateway`, so banning a package bans reaching through it.

    BOUNDARY: a module name assembled at runtime (`import_module("ouroboros." + part)`,
    an f-string, a name read from settings) carries no literal and is NOT caught, and a
    module reached through an already-imported third object is not either. What is closed
    is every form that SPELLS the module name literally. The residue is bounded by the
    CLI's own shape — it takes no module-name argument — and by review.
    """

    reached: set[str] = set()

    def _record(name: str) -> None:
        name = name.strip()
        if not name:
            return
        reached.add(name)
        parts = name.split(".")
        for index in range(1, len(parts)):
            reached.add(".".join(parts[:index]))

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _record(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                # Relative to the `ouroboros` package: level 1 is the package itself.
                prefix = "ouroboros" if node.level == 1 else ""
                module = f"{prefix}.{module}".strip(".") if module else prefix
            _record(module)
            for alias in node.names:
                # `from ouroboros import connection_store` imports a MODULE; `from
                # ouroboros.cli import CLIError` imports a NAME. Statically the two are
                # the same shape, so ask the filesystem which one this is — recording a
                # symbol as a module made every `from x import Y` look like a reach.
                candidate = f"{module}.{alias.name}" if module else alias.name
                probe = REPO_ROOT.joinpath(*candidate.split("."))
                if probe.with_suffix(".py").exists() or (probe / "__init__.py").exists():
                    _record(candidate)
        elif isinstance(node, ast.Call):
            callee = node.func
            label = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
            if label in {"import_module", "__import__"}:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        _record(arg.value)
        elif isinstance(node, ast.Subscript):
            # sys.modules["ouroboros.remote_ssh"]
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "modules"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                _record(node.slice.value)
    return reached


def _assert_cli_is_thin(module_name: str, endpoint: str) -> None:
    source = (
        pathlib.Path(__file__).resolve().parent.parent / "ouroboros" / module_name
    ).read_text(encoding="utf-8")
    reached = _cli_module_reaches(source)
    for forbidden in _CLI_FORBIDDEN_MODULES:
        assert forbidden not in reached, (
            f"{module_name} can reach {forbidden} — the gateway is the only authority "
            "for the store and the transport"
        )
    first_party = {name for name in reached if name.startswith("ouroboros")}
    assert first_party <= _CLI_ALLOWED_FIRST_PARTY | {"ouroboros"}, sorted(
        first_party - _CLI_ALLOWED_FIRST_PARTY - {"ouroboros"}
    )
    assert endpoint in source


def test_connections_cli_is_a_thin_client_with_no_store_or_transport_logic():
    """docs/DEVELOPMENT.md: CLI commands parse flags and render output.

    The store module and the ssh transport must not be reachable from the CLI —
    the gateway is the only authority for both.
    """

    _assert_cli_is_thin("cli_connections.py", "/api/owner/connections")


@pytest.mark.parametrize(
    "spelling",
    [
        "from .connection_store import get_connection",
        "from ouroboros.connection_store import get_connection",
        "from ouroboros import connection_store",
        "import ouroboros.connection_store",
        "import ouroboros.connection_store as store",
        'import importlib\nm = importlib.import_module("ouroboros.connection_store")',
        'm = __import__("ouroboros.connection_store")',
        'import sys\nm = sys.modules["ouroboros.connection_store"]',
        "from ouroboros.gateway.connections import api_connections_list",
    ],
)
def test_the_cli_import_scan_sees_every_spelling_of_a_forbidden_reach(spelling):
    """Each of these reaches the store; the relative form used to pass both checks."""

    assert "ouroboros.connection_store" in _cli_module_reaches(
        spelling
    ) or "ouroboros.gateway.connections" in _cli_module_reaches(spelling), spelling


def test_the_cli_import_scan_states_its_own_boundary():
    """The runtime-assembled residue must stay named, or the claim silently widens."""

    doc = _cli_module_reaches.__doc__ or ""
    assert "BOUNDARY" in doc and "assembled at runtime" in doc
    # And it really is blind to that form — if this starts failing, narrow the paragraph.
    assert "ouroboros.connection_store" not in _cli_module_reaches(
        'import importlib\nm = importlib.import_module("ouroboros." + name)'
    )


def test_the_rebind_path_finally_has_a_caller(monkeypatch, capsys):
    """M9: endpoint, compare-and-set, contract and client method — and no caller.

    A remote Project's placement is rebindable by design: the update endpoint takes
    the same two halves a create takes, admits them identically, and advances
    ``routing_generation`` so work already resolved against the previous target is
    refused at insertion rather than run there. Every layer existed. The only call
    site (`project_create.js`) passed a name, so ``project_has_live_tasks``,
    ``project_routing_generation_changed`` and ``project_not_active`` were codes
    nothing could produce — and after retiring a connection there was no way to move
    its Projects, while Settings → Connections promised they could be rebound.
    """

    import json

    from ouroboros import cli

    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path, body=None, **kwargs):
            calls.append((method, path, body))
            return {"project": {
                "id": "remote-app",
                "name": "Remote app",
                "lifecycle": "active",
                "routing_generation": 3,
                "placement": {
                    "kind": "ssh",
                    "connection_id": "conn-2",
                    "remote_root": "/srv/new",
                    "workspace_id": "ws-9",
                },
            }}

    monkeypatch.setattr(cli, "OuroborosHTTPClient", FakeClient)
    assert cli.main([
        "projects", "rebind", "remote app/1", "--connection", "conn-2",
        "--remote-root", "/srv/new", "--json",
    ]) == 0
    method, path, body = calls[0]
    # The id is ONE path segment: a slash inside it is escaped, not silently turned
    # into a different route.
    assert (method, path) == ("POST", "/api/projects/remote%20app%2F1/update")
    # BOTH halves, always: the gateway refuses half a placement rather than guessing,
    # and it never names the workspace identity — the target allocates that.
    assert body == {"connection_id": "conn-2", "remote_root": "/srv/new"}
    assert "workspace_id" not in body and "placement" not in body
    assert json.loads(capsys.readouterr().out)["project"]["routing_generation"] == 3

    # Both halves are REQUIRED at the parser, so half a rebind never reaches the wire.
    import pytest

    for argv in (
        ["projects", "rebind", "remote-app", "--connection", "conn-2"],
        ["projects", "rebind", "remote-app", "--remote-root", "/srv/new"],
    ):
        with pytest.raises(SystemExit) as exc:
            cli.build_parser().parse_args(argv)
        assert exc.value.code == 2
    capsys.readouterr()

    # And the human listing shows WHERE each project lives, which is the fact an
    # owner needs before choosing a rebind target.
    calls.clear()
    FakeClient.request = lambda self, method, path, body=None, **k: {"projects": [
        {"id": "remote-app", "name": "Remote app", "routing_generation": 2,
         "placement": {"kind": "ssh", "connection_id": "conn-1", "remote_root": "/srv/work"}},
        {"id": "local-app", "name": "Local app", "working_dir": "/home/me/code"},
    ]}
    assert cli.main(["projects", "list"]) == 0
    listed = capsys.readouterr().out
    assert "ssh:conn-1:/srv/work" in listed
    assert "/home/me/code" in listed


def test_rebind_refusals_map_onto_the_one_shared_exit_taxonomy(monkeypatch, capsys):
    """The rebind's typed 409s are scriptable, through the SAME table as connections.

    Two exit-code tables for one set of gateway refusal codes would drift, and the
    codes are the contract a script depends on — so `cli_projects` reuses
    `cli_connections`'s renderer instead of restating it.
    """

    import json

    from ouroboros import cli
    from ouroboros.cli_connections import render_typed_error_exit, _connection_error_exit

    assert render_typed_error_exit is _connection_error_exit

    def refuse(status, payload):
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                raise cli.GatewayHTTPError(status, payload)

        monkeypatch.setattr(cli, "OuroborosHTTPClient", FakeClient)
        code = cli.main([
            "projects", "rebind", "remote-app", "--connection", "conn-2",
            "--remote-root", "/srv/new", "--json",
        ])
        return code, json.loads(capsys.readouterr().out)

    # A rebind under live work, and a rebind that lost the compare-and-set: both
    # conflicts (exit 5), both carrying their next step.
    code, payload = refuse(409, {
        "error": "project has queued or running tasks",
        "error_code": "project_has_live_tasks",
        "action": "wait_or_cancel_tasks",
    })
    assert code == 5
    assert payload["action"] == "wait_or_cancel_tasks"
    assert refuse(409, {
        "error": "project_routing_generation_changed",
        "error_code": "project_routing_generation_changed",
        "action": "reload_projects",
    })[0] == 5
    # The target cannot be consulted at all: not serviceable, exit 4.
    assert refuse(503, {
        "error": "unreachable",
        "error_code": "remote_transport_unavailable",
        "action": "bootstrap_connection",
    })[0] == 4
    # A malformed placement is an ordinary refusal, exit 2.
    assert refuse(400, {
        "error": "half a placement",
        "error_code": "invalid_remote_placement",
    })[0] == 2


def test_the_projects_cli_is_as_thin_as_the_connections_cli():
    """Same rule: flags in, one endpoint call, typed answer out. No registry reach.

    Uses the same scanner as its sibling, so hardening one hardens both — the two gates
    had drifted into byte-for-byte copies of a check with the same hole in each.
    """

    source = (
        pathlib.Path(__file__).resolve().parent.parent / "ouroboros" / "cli_projects.py"
    ).read_text(encoding="utf-8")
    reached = _cli_module_reaches(source)
    for forbidden in (
        "ouroboros.projects_registry",
        "ouroboros.workspace_admission",
        "ouroboros.gateway.projects",
        "ouroboros.connection_store",
        "subprocess",
    ):
        assert forbidden not in reached, forbidden
    first_party = {name for name in reached if name.startswith("ouroboros")}
    assert first_party <= {"ouroboros", "ouroboros.cli", "ouroboros.cli_connections"}, sorted(
        first_party
    )
    assert "/api/projects/" in source

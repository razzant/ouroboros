"""Owned Claudexor daemon (D30): isolation root, discovery cutover, thin proxies.

Everything here is offline: no daemon is spawned, no network is touched. The
live login flow is the daemon's own product surface and is exercised by the
phase acceptance run, not by unit tests.
"""
import json
import pathlib
import shlex

import pytest

from ouroboros import claudexor_daemon as owned


def _write_descriptor(config_dir: pathlib.Path, *, port: int = 45678) -> None:
    daemon_dir = config_dir / "daemon"
    daemon_dir.mkdir(parents=True, exist_ok=True)
    (daemon_dir / "token").write_text("tok-owned", encoding="utf-8")
    (daemon_dir / "control-api.json").write_text(json.dumps({
        "host": "127.0.0.1", "port": port, "tokenPath": str(daemon_dir / "token"),
    }), encoding="utf-8")


def test_owned_config_dir_is_data_plane():
    from ouroboros.config import DATA_DIR

    config_dir = owned.owned_config_dir()
    assert str(config_dir).startswith(str(DATA_DIR))
    # The operator's personal state must never be the owned root.
    assert ".claudexor" not in str(config_dir.relative_to(pathlib.Path(DATA_DIR)))


def test_attach_login_command_targets_the_owned_home():
    """The fallback card's copy-paste command (D30): the user's own terminal,
    the OWNED config dir — never a terminal surface inside the UI."""
    # Pin the syntax this exact assertion describes; the host default is
    # PowerShell on Windows and is tested independently below.
    argv = [
        "/managed/node",
        "/managed/runtime/claudexord.bundle.cjs",
        "setup",
        "attach",
    ]
    command = owned.attach_login_command("job-123", argv=argv, shell="posix")
    config_root = str(owned.owned_config_dir())
    assert command == (
        f"CLAUDEXOR_CONFIG_DIR={shlex.quote(config_root)} "
        "CLAUDEXOR_DAEMON_SOCK='' "
        "/managed/node /managed/runtime/claudexord.bundle.cjs setup attach job-123"
    )


def test_attach_login_shell_selects_the_host_default(monkeypatch):
    from ouroboros import platform_layer

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    assert owned.attach_login_shell() == "posix"
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    assert owned.attach_login_shell() == "powershell"


def test_attach_login_command_quotes_posix_and_powershell(monkeypatch):
    """The fallback is inert text for an explicitly labelled shell, including
    the path/argument characters that break ad-hoc interpolation."""
    monkeypatch.setattr(owned, "owned_config_dir",
                        lambda: pathlib.PurePosixPath("/tmp/Ouroboros profile's data"))
    posix_argv = [
        "/tmp/Ouroboros runtime's/node",
        "/tmp/Ouroboros runtime's/claudexord.bundle.cjs",
        "setup",
        "attach",
    ]
    posix = owned.attach_login_command("job '7", argv=posix_argv, shell="posix")
    assert posix == ("CLAUDEXOR_CONFIG_DIR='/tmp/Ouroboros profile'\"'\"'s data' "
                     "CLAUDEXOR_DAEMON_SOCK='' "
                     "'/tmp/Ouroboros runtime'\"'\"'s/node' "
                     "'/tmp/Ouroboros runtime'\"'\"'s/claudexord.bundle.cjs' "
                     "setup attach 'job '\"'\"'7'")

    monkeypatch.setattr(owned, "owned_config_dir",
                        lambda: pathlib.PureWindowsPath(r"C:\Users\O'Brien\Ouroboros data"))
    powershell_argv = [
        r"C:\Program Files\O'Brien\Node\node.exe",
        r"C:\Program Files\O'Brien\Claudexor\claudexord.bundle.cjs",
        "setup",
        "attach",
    ]
    powershell = owned.attach_login_command(
        "job '7", argv=powershell_argv, shell="powershell")
    assert powershell == ("$env:CLAUDEXOR_CONFIG_DIR='C:\\Users\\O''Brien\\Ouroboros data'; "
                          "$env:CLAUDEXOR_DAEMON_SOCK=''; "
                          "& 'C:\\Program Files\\O''Brien\\Node\\node.exe' "
                          "'C:\\Program Files\\O''Brien\\Claudexor\\claudexord.bundle.cjs' "
                          "'setup' 'attach' 'job ''7'")


def test_resolve_attach_login_argv_binds_serving_identity_and_maps_failures(monkeypatch):
    from ouroboros import claudexor_runtime as runtime
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    engine = {
        "version": "3.7.0",
        "sha": "a" * 40,
        "entry": "/managed/runtime/claudexord.bundle.cjs",
    }

    class Manager:
        error = None

        def resolve_serving_role_command(self, **kwargs):
            assert kwargs == {
                "engine_version": engine["version"],
                "engine_build_sha": engine["sha"],
                "engine_entry": engine["entry"],
                "role": "setup_attach",
            }
            if self.error is not None:
                raise self.error
            return ["/managed/node", engine["entry"]]

    manager = Manager()
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    assert owned.resolve_attach_login_argv(engine) == [
        "/managed/node", engine["entry"], "setup", "attach",
    ]

    cases = (
        ("runtime_role_unavailable", "terminal_transport_unsupported", 409, ()),
        ("runtime_serving_tree_unavailable", "terminal_transport_unavailable", 409, ()),
        ("runtime_serving_node_unavailable", "terminal_transport_unavailable", 409, ()),
        ("runtime_probe_identity_mismatch", "terminal_transport_probe_failed", 503,
         ("retry_setup_login",)),
    )
    for runtime_code, browser_code, status, actions in cases:
        manager.error = runtime.ClaudexorRuntimeError(runtime_code, "fixture")
        with pytest.raises(ClaudexorUnavailable) as excinfo:
            owned.resolve_attach_login_argv(engine)
        assert (excinfo.value.code, excinfo.value.status_code) == (browser_code, status)
        assert excinfo.value.required_actions == actions


def test_resolve_claudexord_explicit_setting_must_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", str(tmp_path / "missing"))
    assert owned.resolve_claudexord() == ""
    real = tmp_path / "claudexord"
    real.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", str(real))
    assert owned.resolve_claudexord() == str(real)


def test_discover_daemon_prefers_owned_home_once_provisioned(monkeypatch, tmp_path):
    """The D30 cutover: default discovery flips to the owned daemon exactly
    when it is provisioned, and stays on the operator layout before that."""
    from ouroboros.gateways import claudexor as gateway_mod

    owned_dir = tmp_path / "data" / "claudexor"
    operator_home = tmp_path / "operator"
    monkeypatch.setattr(owned, "owned_config_dir", lambda: owned_dir)
    monkeypatch.setattr(owned, "owned_descriptor_path",
                        lambda: owned_dir / "daemon" / "control-api.json")
    monkeypatch.setattr(owned, "owned_daemon_provisioned",
                        lambda: (owned_dir / "daemon" / "control-api.json").is_file())
    monkeypatch.setattr(gateway_mod, "operator_home", lambda: operator_home)

    # Not provisioned: the operator layout is the discovery target (and its
    # absence is the typed refusal, proving the owned home was NOT consulted).
    with pytest.raises(gateway_mod.ClaudexorUnavailable) as err:
        gateway_mod.discover_daemon()
    assert "operator" in str(err.value)

    # Provisioned: the owned endpoint wins without any explicit home argument.
    _write_descriptor(owned_dir, port=45679)
    endpoint = gateway_mod.discover_daemon()
    assert (endpoint.port, endpoint.token) == (45679, "tok-owned")

    # An explicit home still reads that home verbatim (delegation callers).
    with pytest.raises(gateway_mod.ClaudexorUnavailable):
        gateway_mod.discover_daemon(home=operator_home)


def test_discover_daemon_at_reads_override_layout(tmp_path):
    from ouroboros.gateways.claudexor import discover_daemon_at

    _write_descriptor(tmp_path / "cfg")
    endpoint = discover_daemon_at(tmp_path / "cfg")
    assert (endpoint.host, endpoint.port) == ("127.0.0.1", 45678)


def test_stop_never_kills_a_daemon_it_did_not_start():
    manager = owned.OwnedClaudexorDaemon()
    assert manager.stop() is False  # nothing self-started -> nothing to kill


def test_ensure_running_without_binary_is_a_typed_refusal(monkeypatch, tmp_path):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros import claudexor_runtime as runtime

    import ouroboros.config as config_mod
    # Ownership is verified FIRST (never adopt); this test is about the binary,
    # so the home must be legitimately ours: under the (patched) data plane.
    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(owned, "owned_daemon_provisioned", lambda: False)
    class MissingRuntime:
        def ensure(self):
            raise runtime.ClaudexorRuntimeError(
                "claudexord_not_installed", "fixture runtime is absent"
            )

    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: MissingRuntime())
    manager = owned.OwnedClaudexorDaemon()
    with pytest.raises(ClaudexorUnavailable) as err:
        manager.ensure_running()
    assert err.value.code == "claudexord_not_installed"


def test_status_payload_not_provisioned_never_spawns(monkeypatch, tmp_path):
    from ouroboros.gateway.claudexor_accounts import _status_payload

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(owned, "owned_daemon_provisioned", lambda: False)
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: owned.OwnedClaudexorDaemon())
    payload = _status_payload(include_models=True)
    assert payload["daemon"]["state"] == "not_provisioned"
    assert payload["harnesses"] == [] and payload["quota"] == []
    assert not (tmp_path / "cfg").exists()  # read-only: nothing provisioned


def test_login_capable_harness_ids_reads_the_manifest_auth_block():
    """Finding #3: only harnesses whose manifest declares a native_session auth
    source have a login concept. API-key-only adapters (raw-api, openrouter)
    must not surface as fake-loginable accounts."""
    from ouroboros.gateway.claudexor_accounts import _login_capable_harness_ids

    rows = [
        {"id": "codex", "manifest": {"capability_profile": {"auth": {
            "supported_sources": ["native_session", "provider_auth_file"]}}}},
        {"id": "cursor", "manifest": {"capability_profile": {"auth": {
            "supported_sources": ["native_session", "api_key_env"]}}}},
        {"id": "openrouter", "manifest": {"capability_profile": {"auth": {
            "supported_sources": ["api_key_env"]}}}},
        {"id": "raw-api", "manifest": None},  # unavailable: no manifest at all
        "not-a-dict",
    ]
    assert _login_capable_harness_ids(rows) == {"codex", "cursor"}


def test_zero_readable_manifests_is_a_read_failure_not_an_empty_filter():
    """Review delta 1 edge: a /v2/harnesses answer that SUCCEEDED but carried
    zero readable manifests says nothing about auth — the helper answers None
    so the caller fails open exactly like the ClaudexorUnavailable path,
    instead of filtering every row out of the panel."""
    from ouroboros.gateway.claudexor_accounts import _login_capable_harness_ids

    assert _login_capable_harness_ids([]) is None
    assert _login_capable_harness_ids([
        {"id": "codex", "manifest": None},
        {"id": "raw-api"},
        "not-a-dict",
    ]) is None
    # ONE readable manifest is enough to trust the read (even when it grants
    # nobody a login): the filter then applies, it does not fail open.
    assert _login_capable_harness_ids([
        {"id": "codex", "manifest": None},
        {"id": "openrouter", "manifest": {"capability_profile": {"auth": {
            "supported_sources": ["api_key_env"]}}}},
    ]) == set()


def test_status_payload_filters_api_key_only_adapters(monkeypatch, tmp_path):
    """The catalog projection AND the daemon's native pseudo-rows are filtered
    to login-capable harnesses; a transient manifest-read failure fails OPEN."""
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw

    class FakeDaemon:
        def status_dict(self):
            return {"state": "running"}

    class FakeGateway:
        engine_version = "9.9.9"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            return {"harnesses": [
                {"id": "codex", "displayName": "Codex CLI", "status": "ok", "enabled": True},
                {"id": "openrouter", "displayName": "Raw API (openai)", "status": "degraded", "enabled": True},
            ]}

        def harnesses(self):
            return [
                {"id": "codex", "manifest": {"capability_profile": {"auth": {
                    "supported_sources": ["native_session"]}}}},
                {"id": "openrouter", "manifest": {"capability_profile": {"auth": {
                    "supported_sources": ["api_key_env"]}}}},
            ]

        def credential_profiles(self):
            return {"profiles": [
                # Review delta 2: named wrappers pass the same predicate. An
                # api_key-kind profile registered for a non-loginable harness
                # must not render a fake-loginable account row…
                {"profile": {"harness_id": "openrouter", "profile_id": "or-key",
                             "kind": "api_key"}, "status": {}, "identity": {}},
                # …while a capable-harness profile survives untouched.
                {"profile": {"harness_id": "codex", "profile_id": "koshak",
                             "kind": "native_session"}, "status": {}, "identity": {}},
            ], "harnessAccounts": [
                {"harness_id": "codex", "native_login_detected": True},
                {"harness_id": "openrouter", "native_login_detected": False},
            ]}

        def quota_snapshots(self):
            return []

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    payload = _status_payload(include_models=False)
    assert [h["id"] for h in payload["harnesses"]] == ["codex"]
    assert [r["harness_id"] for r in payload["profiles"]["harnessAccounts"]] == ["codex"]
    assert [w["profile"]["profile_id"] for w in payload["profiles"]["profiles"]] == ["koshak"]

    class FlakyGateway(FakeGateway):
        def harnesses(self):
            raise gw.ClaudexorUnavailable("daemon_unreachable", "transient")

    monkeypatch.setattr(gw, "ClaudexorGateway", FlakyGateway)
    payload = _status_payload(include_models=False)
    # Fail-open: a blip in the manifest read must not blank the panel.
    assert [h["id"] for h in payload["harnesses"]] == ["codex", "openrouter"]
    assert len(payload["profiles"]["harnessAccounts"]) == 2
    assert len(payload["profiles"]["profiles"]) == 2


def test_a_vouched_login_survives_an_unreadable_manifest(monkeypatch, tmp_path):
    """Review delta 1: a harness whose manifest is null/unreadable must not
    lose the account the owner is really logged into — the daemon's own
    ``native_login_detected`` vouches the row. The un-vouched, non-capable
    adapter stays filtered (raw-api must NOT come back)."""
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw

    class FakeDaemon:
        def status_dict(self):
            return {"state": "running"}

    class FakeGateway:
        engine_version = "9.9.9"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            return {"harnesses": [
                # Manifest-less codex: the daemon vouches the login on the row.
                {"id": "codex", "displayName": "Codex CLI", "enabled": True,
                 "native_login_detected": True},
                {"id": "cursor", "displayName": "Cursor", "enabled": True},
                {"id": "raw-api", "displayName": "Raw API", "enabled": True},
            ]}

        def harnesses(self):
            return [
                {"id": "codex", "manifest": None},  # unreadable, NOT un-capable
                {"id": "cursor", "manifest": {"capability_profile": {"auth": {
                    "supported_sources": ["native_session"]}}}},
                {"id": "raw-api", "manifest": None},
            ]

        def credential_profiles(self):
            return {"profiles": [
                # A wrapper for the vouched-but-unreadable harness survives too.
                {"profile": {"harness_id": "codex", "profile_id": "backup",
                             "kind": "native_session"}, "status": {}, "identity": {}},
            ], "harnessAccounts": [
                {"harness_id": "codex", "native_login_detected": True,
                 "identity": {"email": "owner@example.com"}},
                {"harness_id": "cursor", "native_login_detected": False},
                {"harness_id": "raw-api", "native_login_detected": False},
            ]}

        def quota_snapshots(self):
            return []

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    payload = _status_payload(include_models=False)
    assert [h["id"] for h in payload["harnesses"]] == ["codex", "cursor"]
    accounts = payload["profiles"]["harnessAccounts"]
    # The logged-into row keeps its identity; raw-api stays out.
    assert [r["harness_id"] for r in accounts] == ["codex", "cursor"]
    assert accounts[0]["identity"] == {"email": "owner@example.com"}
    assert [w["profile"]["profile_id"] for w in payload["profiles"]["profiles"]] == ["backup"]


def test_account_removal_is_the_engine_contract_and_refuses_out_loud(monkeypatch, tmp_path):
    """The FIFTH thin proxy: removing a named account is the daemon's own
    ``DELETE /v2/credential-profiles/:harness/:profileId``.

    Two invariants, one test. Ouroboros deletes NO vendor credential material
    itself — the whole handler is one forwarded call — and an engine refusal
    comes back AS a refusal (503), never as a cheerful ok that would leave the
    owner believing an account is gone while it still rotates."""
    import asyncio

    from starlette.requests import Request

    from ouroboros.claudexor_daemon import owned_config_dir  # noqa: F401  (patched below)
    from ouroboros.gateway import claudexor_accounts as accounts
    from ouroboros.gateways import claudexor as gw
    import ouroboros.claudexor_daemon as owned

    deleted: list = []
    receipt = {
        "profile": {
            "profile_id": "work",
            "harness_id": "codex",
            "display_name": "Work",
            "credential_kind": "config_dir_login",
            "isolation_locator": "/data/claudexor/profiles/codex-work",
            "secret_ref": None,
            "enabled": True,
            "created_at": None,
        },
        "removed": True,
        "credentialCleanup": "config_dir_removed",
        "cleanupWarning": "owned profile storage cleanup needs manual inspection",
        "vendorCredentialDisposition": {
            "owner": "vendor", "state": "left_unchanged", "scope": "os_user",
        },
    }

    class FakeGateway:
        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def delete_credential_profile(self, harness_id, profile_id):
            if refuse:
                raise gw.ClaudexorUnavailable("profile_in_use", "still running work")
            deleted.append((harness_id, profile_id))
            return receipt

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _cfg: object())

    def _call(harness, profile_id):
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        request = Request({
            "type": "http", "method": "DELETE",
            "path": f"/api/claudexor/credential-profiles/{harness}/{profile_id}",
            "headers": [], "query_string": b"",
            "path_params": {"harness": harness, "profile_id": profile_id},
        }, receive)
        return asyncio.run(accounts.api_claudexor_credential_profile(request))

    refuse = False
    ok = _call("codex", "work")
    assert ok.status_code == 200
    assert deleted == [("codex", "work")], "the handler forwards and does nothing else"
    assert json.loads(ok.body) == receipt, "the complete engine receipt survives verbatim"

    refuse = True
    denied = _call("codex", "work")
    assert denied.status_code == 503
    assert b"profile_in_use" in denied.body
    assert deleted == [("codex", "work")], "a refusal removed nothing"

    # A native CLI login has no profile id, and no route: this process cannot
    # honestly sign a vendor CLI out, so it refuses at the edge instead of
    # inventing a deletion.
    refuse = False
    bare = _call("codex", "")
    assert bare.status_code == 400
    assert b"profile_id" in bare.body


def test_account_enabled_toggle_is_the_engine_contract(monkeypatch, tmp_path):
    """The Enabled toggle shares the credential-profile route (PATCH beside
    DELETE) and is the same thin-proxy rule: one forwarded call carrying the
    engine's own strict ``{enabled}`` body, a refusal answered AS a refusal,
    and a body that is not one JSON boolean refused at this edge before any
    daemon work — nothing is coerced for the engine."""
    import asyncio
    import json

    from starlette.requests import Request

    from ouroboros.gateway import claudexor_accounts as accounts
    from ouroboros.gateways import claudexor as gw
    import ouroboros.claudexor_daemon as owned

    patched: list = []

    class FakeGateway:
        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def update_credential_profile(self, harness_id, profile_id, *, enabled):
            if refuse:
                raise gw.ClaudexorUnavailable("profile_unknown", "no such registry row")
            patched.append((harness_id, profile_id, enabled))
            return {"profile": {}, "status": {}}

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _cfg: object())

    def _call(harness, profile_id, body):
        raw = json.dumps(body).encode("utf-8") if body is not None else b""

        async def receive():
            return {"type": "http.request", "body": raw, "more_body": False}

        request = Request({
            "type": "http", "method": "PATCH",
            "path": f"/api/claudexor/credential-profiles/{harness}/{profile_id}",
            "headers": [(b"content-type", b"application/json")], "query_string": b"",
            "path_params": {"harness": harness, "profile_id": profile_id},
        }, receive)
        return asyncio.run(accounts.api_claudexor_credential_profile(request))

    refuse = False
    ok = _call("codex", "work", {"enabled": False})
    assert ok.status_code == 200
    assert patched == [("codex", "work", False)], "the handler forwards and does nothing else"
    body = json.loads(ok.body)
    assert body == {"ok": True, "harness": "codex", "profile_id": "work", "enabled": False}

    refuse = True
    denied = _call("codex", "work", {"enabled": True})
    assert denied.status_code == 503
    assert b"profile_unknown" in denied.body
    assert patched == [("codex", "work", False)], "a refusal toggled nothing"

    refuse = False
    for bad in ({"enabled": "true"}, {"enabled": 1}, {}, None):
        answer = _call("codex", "work", bad)
        assert answer.status_code == 400, f"non-boolean body {bad!r} must refuse at the edge"
        assert b"enabled" in answer.body
    assert patched == [("codex", "work", False)], "no invalid body reached the daemon"

    bare = _call("codex", "", {"enabled": True})
    assert bare.status_code == 400
    assert b"profile_id" in bare.body


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


def test_pinned_engine_serves_the_account_pools_marker_id():
    """Cross-repo byte-assertion (unified-accounts sprint obligation): from
    Claudexor 3.6.0 the engine's /v2/operations catalog serves the pool-
    authority read under the EXACT id `get:account-pools`. The engine derives
    ids from routes (`method.toLowerCase() + ':' + path minus its '/v2/'
    prefix, [:/<>]+ folded to '.'), and claudexor pins the same literal from
    its side (control-api.test.ts asserts the catalog row for
    /v2/account-pools carries this id verbatim). If either repo respells it,
    the feature detect quietly answers False and every install degrades to
    the legacy accounts rendering — the deliberate cheap direction of
    `_unified_accounts_native`, which is exactly why no behavioral test would
    notice. The assertion is gated on the tracked runtime pin so a deliberate
    pre-3.6 pin rollback leaves it dormant instead of red."""
    from ouroboros.claudexor_runtime import load_runtime_pin
    from ouroboros.gateway.claudexor_accounts import (
        _ACCOUNT_POOLS_OPERATION_ID,
        _unified_accounts_native,
    )

    pin = load_runtime_pin()
    assert pin is not None, "the tracked runtime pin must select a release"
    major, minor, _patch = (int(part) for part in pin.version.split("."))
    if (major, minor) < (3, 6):
        pytest.skip(
            f"pinned engine {pin.version} predates the unified account model"
        )
    assert _ACCOUNT_POOLS_OPERATION_ID == "get:account-pools"
    # A 3.6-shaped catalog slice — the accounts-surface rows exactly as the
    # pinned engine generates them — satisfies the feature detect...
    catalog_3_6 = [
        {"id": "get:quota", "method": "GET", "path": "/v2/quota"},
        {"id": "get:account-pools", "method": "GET", "path": "/v2/account-pools"},
        {"id": "get:credential-profiles", "method": "GET",
         "path": "/v2/credential-profiles"},
        {"id": "post:accounts-migration.rollback", "method": "POST",
         "path": "/v2/accounts-migration/rollback"},
    ]
    assert _unified_accounts_native(catalog_3_6) is True
    # ...and the same catalog without the one marker row is the legacy model:
    # no neighbouring accounts route may stand in for the marker.
    without_marker = [
        op for op in catalog_3_6 if op["id"] != _ACCOUNT_POOLS_OPERATION_ID
    ]
    assert _unified_accounts_native(without_marker) is False


def test_status_payload_stamps_the_unified_accounts_fact(monkeypatch, tmp_path):
    """`unified_accounts` rides every status answer: True only when the
    operations catalog was READ and carries the account-pools marker; an old
    engine reads False, and a catalog read failure fails CLOSED to False (the
    legacy rendering is correct on every engine; a guessed True is not)."""
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw
    import ouroboros.claudexor_daemon as owned

    class FakeDaemon:
        def status_dict(self):
            return {"state": "running"}

    class FakeGateway:
        engine_version = "9.9.9"
        operations_answer: list = [{"id": "get:account-pools"}]

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            return {"harnesses": []}

        def harnesses(self):
            return []

        def credential_profiles(self):
            return {"profiles": [], "harnessAccounts": [], "accountPools": []}

        def quota_snapshots(self):
            return []

        def operations(self):
            return type(self).operations_answer

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    assert _status_payload(include_models=False)["unified_accounts"] is True

    FakeGateway.operations_answer = [{"id": "get:quota"}]
    assert _status_payload(include_models=False)["unified_accounts"] is False

    class BrokenCatalog(FakeGateway):
        def operations(self):
            raise gw.ClaudexorUnavailable("daemon_unreachable", "catalog read died")

    monkeypatch.setattr(gw, "ClaudexorGateway", BrokenCatalog)
    payload = _status_payload(include_models=False)
    assert payload["unified_accounts"] is False, "an unreadable catalog fails closed to the old model"
    # …and the absorbed catalog read never downgrades the real facets.
    assert payload["reads"] == {"catalog": "ok", "accounts": "ok", "quota": "ok"}

    class NoCatalogMethod(FakeGateway):
        operations = None

    monkeypatch.setattr(gw, "ClaudexorGateway", NoCatalogMethod)
    assert _status_payload(include_models=False)["unified_accounts"] is False

    class UnifiedWire(FakeGateway):
        # The unified engine's full accounts body (frozen contract §L.1):
        # every account a named registry row, the legacy key an empty
        # compatibility list, the routing verdict in the ADDITIVE pool key.
        operations_answer = [{"id": "get:account-pools"}]

        def harnesses(self):
            # A populated manifest read turns the visibility filters ON —
            # exactly the path that rewrites the profiles body's other keys.
            return [{"id": "codex", "manifest": {"capability_profile": {
                "auth": {"supported_sources": ["native_session"]}}}}]

        def credential_profiles(self):
            return {
                "profiles": [{"profile": {"harness_id": "codex",
                                          "profile_id": "codex-default"}}],
                "harnessAccounts": [],
                "accountPools": [{"harness_id": "codex",
                                  "next_up": {"kind": "profile",
                                              "profileId": "codex-default"}}],
            }

    monkeypatch.setattr(gw, "ClaudexorGateway", UnifiedWire)
    served = _status_payload(include_models=False)
    # The ADDITIVE pool key rides the accounts facet through the visibility
    # filters untouched: the store's dual-wire nextUpAccount reader and the
    # onboarding dual-read both consume it from this one served payload.
    assert served["unified_accounts"] is True
    assert served["profiles"]["accountPools"] == [
        {"harness_id": "codex",
         "next_up": {"kind": "profile", "profileId": "codex-default"}}]
    assert served["profiles"]["harnessAccounts"] == []
    assert [w["profile"]["profile_id"] for w in served["profiles"]["profiles"]] == ["codex-default"]

    class StoppedDaemon:
        def status_dict(self):
            return {"state": "stale"}

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: StoppedDaemon())
    assert _status_payload(include_models=False)["unified_accounts"] is False


def test_login_endpoint_validates_before_any_daemon_work():
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.claudexor_accounts import api_claudexor_login

    async def _call(body: dict):
        payload = json.dumps(body).encode()

        async def receive():
            return {"type": "http.request", "body": payload, "more_body": False}

        request = Request({
            "type": "http", "method": "POST", "path": "/api/claudexor/login",
            "headers": [(b"content-type", b"application/json")], "query_string": b"",
        }, receive)
        return await api_claudexor_login(request)

    missing = asyncio.run(_call({}))
    assert missing.status_code == 400 and b"harness is required" in missing.body
    bad_transport = asyncio.run(_call({"harness": "codex", "transport": "carrier"}))
    assert bad_transport.status_code == 400 and b"transport" in bad_transport.body


def test_external_recovery_preflights_before_profile_or_job_mutation(monkeypatch):
    """An old or unprobeable serving tree cannot strand a client_pty job."""
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.claudexor_accounts import api_claudexor_login
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    calls = {"profile": 0, "setup": 0}
    failure = {"exc": ClaudexorUnavailable(
        "terminal_transport_unsupported",
        "old serving runtime has no setup_attach role",
        status_code=409,
    )}

    class Gateway:
        def handshake(self):
            return {"engine": {
                "version": "3.6.0",
                "sha": "a" * 40,
                "entry": "/managed/runtime/claudexord.bundle.cjs",
            }}

        def agent_capabilities(self):
            return {"harnesses": [{
                "id": "one", "setupLogin": {"mode": "external_terminal"},
            }]}

        def create_credential_profile(self, *_args):
            calls["profile"] += 1

        def setup_job_create(self, _body):
            calls["setup"] += 1
            return {"id": "stranded"}

    class GatewayCtx:
        def __enter__(self):
            return Gateway()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: GatewayCtx())

    def fail_attach_preflight(_engine):
        raise failure["exc"]

    monkeypatch.setattr(owned, "resolve_attach_login_argv", fail_attach_preflight)

    async def call():
        payload = json.dumps({"harness": "one", "profile_id": "named"}).encode()

        async def receive():
            return {"type": "http.request", "body": payload, "more_body": False}

        request = Request({
            "type": "http", "method": "POST", "path": "/api/claudexor/login",
            "headers": [(b"content-type", b"application/json")], "query_string": b"",
        }, receive)
        return await api_claudexor_login(request)

    unsupported = asyncio.run(call())
    assert unsupported.status_code == 409
    assert json.loads(unsupported.body) == {
        "error": "old serving runtime has no setup_attach role",
        "code": "terminal_transport_unsupported",
    }
    assert calls == {"profile": 0, "setup": 0}

    failure["exc"] = ClaudexorUnavailable(
        "terminal_transport_probe_failed",
        "serving runtime probe failed",
        status_code=503,
        required_actions=("retry_setup_login",),
    )
    probe_failed = asyncio.run(call())
    assert probe_failed.status_code == 503
    assert json.loads(probe_failed.body) == {
        "error": "serving runtime probe failed",
        "code": "terminal_transport_probe_failed",
        "required_actions": ["retry_setup_login"],
    }
    assert calls == {"profile": 0, "setup": 0}


def test_login_create_passes_only_marked_daemon_create_verdicts_through(monkeypatch):
    """Frozen setup-create 400/409/retryable-503 problems keep their typed
    browser envelope; the same status before setup creation stays generic."""
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.claudexor_accounts import api_claudexor_login
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    engine_said = ('harness "zephyr" has no default credential store: sign in '
                   'from a named account (add one first, then start the login from it)')
    refusal = {"exc": ClaudexorUnavailable("http_400", engine_said, status_code=400),
               "stage": "create"}

    class _Gateway:
        def agent_capabilities(self):
            if refusal["stage"] == "capabilities":
                raise refusal["exc"]
            return {"harnesses": [{"id": "zephyr", "setupLogin": {"mode": "in_app"}}]}

        def operations(self):
            return {}

        def create_credential_profile(self, harness, profile_id):
            return {}

        def setup_job_create(self, request_body):
            if refusal["stage"] == "create":
                raise refusal["exc"]
            return {"id": "job-1", "status": "running"}

    class _GatewayCtx:
        def __enter__(self):
            # A handshake-stage refusal raises BEFORE any gateway exists — the
            # narrowing must keep it a 503 even at status 400, because only the
            # job CREATE answers about the requested login shape.
            if refusal["stage"] == "handshake":
                raise refusal["exc"]
            return _Gateway()

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(
        "ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: _GatewayCtx())

    async def _call():
        payload = json.dumps({"harness": "zephyr"}).encode()

        async def receive():
            return {"type": "http.request", "body": payload, "more_body": False}

        request = Request({
            "type": "http", "method": "POST", "path": "/api/claudexor/login",
            "headers": [(b"content-type", b"application/json")], "query_string": b"",
        }, receive)
        return await api_claudexor_login(request)

    answer = asyncio.run(_call())
    assert answer.status_code == 400
    body = json.loads(answer.body)
    assert body["error"] == engine_said, "the engine's sentence rides through verbatim"
    assert body["code"] == "http_400"
    assert "required_actions" not in body, "an absent continuation is absent, not []"

    # A refusal that names a continuation keeps it (bounded by the transport).
    refusal["exc"] = ClaudexorUnavailable(
        "http_400", engine_said, status_code=400,
        required_actions=("add_named_account",))
    with_actions = json.loads(asyncio.run(_call()).body)
    assert with_actions["required_actions"] == ["add_named_account"]

    # All frozen pre-job terminal verdict classes keep the same problem
    # envelope: unsupported/unavailable is a 409 with an external-terminal
    # continuation; a failed bounded helper probe is retryable 503 with both
    # retry and external-terminal continuations.
    for code in ("terminal_transport_unavailable", "terminal_transport_unsupported"):
        refusal["exc"] = ClaudexorUnavailable(
            code, f"{code} detail", status_code=409,
            required_actions=("use_external_terminal",))
        unavailable = asyncio.run(_call())
        assert unavailable.status_code == 409
        assert json.loads(unavailable.body) == {
            "error": f"{code} detail",
            "code": code,
            "required_actions": ["use_external_terminal"],
        }

    refusal["exc"] = ClaudexorUnavailable(
        "terminal_transport_probe_failed", "terminal helper probe timed out",
        status_code=503,
        required_actions=("retry_setup_login", "use_external_terminal"))
    probe_failed = asyncio.run(_call())
    assert probe_failed.status_code == 503
    assert json.loads(probe_failed.body) == {
        "error": "terminal helper probe timed out",
        "code": "terminal_transport_probe_failed",
        "required_actions": ["retry_setup_login", "use_external_terminal"],
    }

    # The same typed-looking 503 before setup_job_create is not a setup-create
    # verdict. Handshake/transport and capability-discovery failures retain the
    # generic gateway 503 and must not leak code/actions into this contract.
    for stage in ("handshake", "capabilities"):
        refusal["stage"] = stage
        refusal["exc"] = ClaudexorUnavailable(
            "terminal_transport_probe_failed", "terminal helper probe timed out",
            status_code=503,
            required_actions=("retry_setup_login", "use_external_terminal"))
        before_create = asyncio.run(_call())
        assert before_create.status_code == 503
        before_body = json.loads(before_create.body)
        assert before_body == {
            "error": "terminal_transport_probe_failed: terminal helper probe timed out",
        }

    # No 400 verdict — an unreachable daemon (status 0) and a daemon 5xx — is
    # never promoted to one: both stay the proxy's honest 503.
    refusal["stage"] = "create"
    for exc in (
        ClaudexorUnavailable("daemon_unreachable", "connect refused"),
        ClaudexorUnavailable("http_500", "boom", status_code=500),
        ClaudexorUnavailable(
            "http_503", "untyped setup refusal", status_code=503,
            required_actions=("retry",)),
    ):
        refusal["exc"] = exc
        answer = asyncio.run(_call())
        assert answer.status_code == 503
        assert exc.code.encode() in answer.body
        assert set(json.loads(answer.body)) == {"error"}

    # A 400 from the HANDSHAKE stage is engine/protocol trouble, not a verdict
    # about the requested login shape: the pass-through is scoped to the job
    # CREATE and everything earlier stays the proxy's honest 503.
    refusal["exc"] = ClaudexorUnavailable("http_400", "protocol mismatch", status_code=400)
    refusal["stage"] = "handshake"
    answer = asyncio.run(_call())
    assert answer.status_code == 503
    assert b"http_400" in answer.body


def test_profile_create_problem_surfaces_typed_and_starts_no_setup(monkeypatch):
    """A non-duplicate profile-registration verdict keeps status, code and
    action at the browser boundary; it never falls through to setup create."""
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.claudexor_accounts import api_claudexor_login
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    setups = []
    refusal = {"exc": ClaudexorUnavailable(
        "credential_profile_required", "name this account", status_code=400,
        required_actions=("add_named_account",))}

    class Gateway:
        def agent_capabilities(self):
            return {"harnesses": [{"id": "one", "setupLogin": {"mode": "in_app"}}]}

        def create_credential_profile(self, *_args):
            raise refusal["exc"]

        def setup_job_create(self, body):
            setups.append(body)
            return {"id": "impossible"}

    class GatewayCtx:
        def __enter__(self):
            return Gateway()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: GatewayCtx())

    async def _call():
        payload = json.dumps({"harness": "one", "profile_id": "named"}).encode()

        async def receive():
            return {"type": "http.request", "body": payload, "more_body": False}

        request = Request({
            "type": "http", "method": "POST", "path": "/api/claudexor/login",
            "headers": [(b"content-type", b"application/json")], "query_string": b"",
        }, receive)
        return await api_claudexor_login(request)

    answer = asyncio.run(_call())
    assert answer.status_code == 400
    assert json.loads(answer.body) == {
        "error": "name this account",
        "code": "credential_profile_required",
        "required_actions": ["add_named_account"],
    }
    refusal["exc"] = ClaudexorUnavailable(
        "profile_name_invalid", "bad profile name", status_code=422)
    invalid = asyncio.run(_call())
    assert invalid.status_code == 422
    assert json.loads(invalid.body) == {
        "error": "bad profile name", "code": "profile_name_invalid",
    }
    refusal["exc"] = ClaudexorUnavailable(
        "profile_storage_failed", "disk write failed", status_code=500)
    failed = asyncio.run(_call())
    assert failed.status_code == 503
    assert json.loads(failed.body) == {
        "error": "disk write failed", "code": "profile_storage_failed",
    }
    assert setups == []


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


def test_setup_login_projection_preserves_all_four_wire_states():
    """Optional and nullable are different protocol facts; malformed current
    evidence must never collapse onto the one legacy state."""
    from ouroboros.gateway.claudexor_accounts import _harness_setup_login

    assert _harness_setup_login(
        {"harnesses": [{"id": "one"}]}, "one") == ("absent", "")
    assert _harness_setup_login(
        {"harnesses": [{"id": "one", "setupLogin": None}]}, "one") == ("null", "")
    assert _harness_setup_login(
        {"harnesses": [{"id": "one", "setupLogin": {
            "mode": "in_app", "futureField": True,
        }}]}, "one") == ("object", "in_app")
    assert _harness_setup_login(
        {"harnesses": [{"id": "one", "setupLogin": {
            "mode": "external_terminal",
        }}]}, "one") == ("object", "external_terminal")
    for malformed in (
        {},
        {"harnesses": []},
        {"harnesses": [{"id": "other"}]},
        {"harnesses": [{"id": "one"}, {"id": "one"}]},
        {"harnesses": [{"id": "one", "setupLogin": {}}]},
        {"harnesses": [{"id": "one", "setupLogin": {"mode": "future"}}]},
        {"harnesses": [{"id": "one", "setupLogin": "in_app"}]},
    ):
        assert _harness_setup_login(malformed, "one") == ("malformed", "")


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
    # Codex client_pty has one legal flow. A caller-provided device flow cannot
    # override the transport invariant, while non-codex never receives the
    # codex-only field.
    codex_external = _build_login_request(
        "codex", "", "client_pty", "device_auth", disclosure_native=True)
    assert codex_external == {
        "harness": "codex", "action": "login", "authRequest": "subscription",
        "transport": "client_pty", "loginFlow": "browser_redirect",
    }
    non_codex = _build_login_request(
        "claude", "", "client_pty", "device_auth", disclosure_native=True)
    assert "loginFlow" not in non_codex
    # Codex is untouched by the capability: its device flow was already
    # daemon-hosted, transport stays absent either way.
    for flag in (True, False):
        codex = _build_login_request("codex", "", "", "device_auth", disclosure_native=flag)
        assert "transport" not in codex and codex["loginFlow"] == "device_auth"


def _claudexor_3_6_capabilities(harness: str) -> dict:
    """Schema-shaped 3.6.0 catalog fixture: the new key is truly absent."""
    row = {
        "id": harness,
        "enabled": True,
        "displayName": harness,
        "status": "ok",
        "providerFamily": "unknown",
        "enabledIntents": ["coding"],
        "disabledIntents": [],
        "reasons": [],
        "configuredModel": None,
        "configuredModelValid": None,
        "models": {"source": "none", "count": 0, "verifiedAgainst": None},
        "webPolicy": "none",
        "attachmentInputs": [],
        "effortLevels": [],
        "accessProfilesSupported": [],
        "readonlyMechanism": "none",
        "delegation": {
            "available": False,
            "reason": "manifest_unsupported",
            "remediation": "Choose a harness with Delegate support.",
            "requiresFullAccess": False,
        },
    }
    return {
        "ok": True,
        "version": "3.6.0",
        "generatedAt": "2026-08-19T00:00:00.000Z",
        "git": {
            "status": "ready", "version": "2.50.1", "detail": "git version 2.50.1",
            "remediation": None,
        },
        "harnesses": [row],
        "availableHarnesses": [harness],
        "modes": ["ask", "plan", "agent"],
        "runControlKeys": ["prompt"],
        "outputSchemaDialects": [{
            "dialect": "draft-07",
            "uri": "http://json-schema.org/draft-07/schema#",
            "defaultWhenOmitted": True,
        }],
        "mutability": {
            "readOnlyModes": ["ask", "plan"],
            "writeModes": ["agent"],
            "isolationKinds": ["envelope", "live"],
            "workspaceModes": ["in_place", "isolated"],
            "accessProfiles": ["readonly", "workspace_write", "full"],
            "applyModes": ["apply", "commit", "branch", "pr"],
        },
        "cliCommands": [{
            "id": "ask", "mutability": "read", "stability": "stable", "recovery": False,
        }],
        "mcpTools": ["claudexor_ask"],
        "runApplyStates": ["not_applied", "applied", "applied_review_blocked", "reverted"],
    }


def _create_login(monkeypatch, tmp_path, body: dict, *, operations,
                  capabilities=None, raises=False):
    """Run the REAL create path against a fake daemon, answering the probe with
    ``operations`` (or raising for the catalog-unreadable case). Returns
    ``(answer, request_body_actually_sent)``."""
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway.claudexor_accounts import _login_create
    from ouroboros.gateways import claudexor as gw

    sent: dict = {}
    engine = {
        "version": "3.7.0",
        "sha": "a" * 40,
        "entry": str(tmp_path / "runtime" / "claudexord.bundle.cjs"),
    }
    attach_argv = [
        str(tmp_path / "node" / "node"),
        engine["entry"],
        "setup",
        "attach",
    ]

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
            return {"engine": engine}

        def agent_capabilities(self):
            if capabilities is not None:
                return capabilities
            return _claudexor_3_6_capabilities(str(body.get("harness") or ""))

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
    monkeypatch.setattr(
        owned,
        "resolve_attach_login_argv",
        lambda actual: attach_argv if actual == engine else pytest.fail(
            "attach recovery must bind to the handshaken serving engine"),
    )
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
    assert native["setup_login_source"] == "legacy_global_operation"
    assert "transport" not in sent
    # No client_pty job ⇒ no attach command to demote into Advanced.
    assert "attach_command" not in native

    wrong_id, sent = _create_login(
        monkeypatch, tmp_path, {"harness": "claude"},
        operations=[{"id": "post:setup.jobs.input", "method": "POST",
                     "path": "/v2/setup/jobs/:id/input"}])
    assert wrong_id["disclosure_native"] is False
    assert wrong_id["setup_login_source"] == "legacy_global_operation"
    assert sent["transport"] == "client_pty"
    assert wrong_id["attach_command"] == owned.attach_login_command(
        wrong_id["job_id"],
        argv=[
            str(tmp_path / "node" / "node"),
            str(tmp_path / "runtime" / "claudexord.bundle.cjs"),
            "setup",
            "attach",
        ],
        shell=wrong_id["attach_shell"],
    )


def test_login_create_prefers_each_harness_setup_mode_over_global_operations(monkeypatch, tmp_path):
    """Two rows can publish different host-effective modes. The exact target
    row selects an omitted request and a global operation cannot override it;
    an explicit client_pty recovery request remains explicit."""
    capabilities = {"harnesses": [
        {"id": "one", "setupLogin": {"mode": "in_app"}},
        {"id": "two", "setupLogin": {"mode": "external_terminal"}},
    ]}
    in_app, sent = _create_login(
        monkeypatch, tmp_path, {"harness": "one"},
        operations=[], capabilities=capabilities, raises=True)
    assert "transport" not in sent
    assert in_app["disclosure_native"] is True
    assert in_app["setup_login_source"] == "per_harness"
    assert "attach_command" not in in_app

    explicit, sent = _create_login(
        monkeypatch, tmp_path, {"harness": "one", "transport": "client_pty"},
        operations=[], capabilities=capabilities, raises=True)
    assert sent["transport"] == "client_pty"
    assert explicit["disclosure_native"] is False
    assert explicit["setup_login_source"] == "per_harness"
    assert "attach_command" in explicit

    external, sent = _create_login(
        monkeypatch, tmp_path, {"harness": "two"},
        operations=[_INPUT_OP], capabilities=capabilities, raises=True)
    assert sent["transport"] == "client_pty"
    assert external["disclosure_native"] is False
    assert external["setup_login_source"] == "per_harness"
    assert external["attach_shell"] in {"posix", "powershell"}
    assert "attach_command" in external


def test_null_setup_login_preserves_omitted_and_explicit_transport(monkeypatch, tmp_path):
    """Null delegates support admission to setup create without consulting
    the legacy operation catalog. Omitted remains omitted; an explicit
    client_pty request remains exact and receives the attach metadata."""
    capabilities = {"harnesses": [{"id": "one", "setupLogin": None}]}
    native, sent = _create_login(
        monkeypatch, tmp_path, {"harness": "one"},
        operations=[], capabilities=capabilities, raises=True)
    assert "transport" not in sent
    assert native["disclosure_native"] is True
    assert native["setup_login_source"] == "setup_job_admission"
    assert "attach_command" not in native

    external, sent = _create_login(
        monkeypatch, tmp_path, {"harness": "one", "transport": "client_pty"},
        operations=[], capabilities=capabilities, raises=True)
    assert sent["transport"] == "client_pty"
    assert external["disclosure_native"] is False
    assert external["setup_login_source"] == "setup_job_admission"
    assert "attach_command" in external


@pytest.mark.parametrize("profile_id,expected_calls", [
    ("", {"operations": 0, "profile": 0, "setup": 1}),
    ("named", {"operations": 0, "profile": 1, "setup": 0}),
])
def test_null_setup_login_unsupported_admission_makes_no_mutation(
        monkeypatch, profile_id, expected_calls):
    """The producer rejects an unsupported unnamed family at setup admission,
    and a named family at its pre-write profile admission. Neither path uses
    the legacy catalog or creates durable profile/job state."""
    from ouroboros.gateway.claudexor_accounts import _login_create
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    calls = {"operations": 0, "profile": 0, "setup": 0}
    durable = {"profiles": [], "jobs": []}
    refusal = ClaudexorUnavailable(
        "invalid_request", "unsupported harness", status_code=400)

    class Gateway:
        def agent_capabilities(self):
            return {"harnesses": [{"id": "one", "setupLogin": None}]}

        def operations(self):
            calls["operations"] += 1
            return [_INPUT_OP]

        def create_credential_profile(self, harness, profile_id):
            calls["profile"] += 1
            if harness == "one":
                raise refusal
            durable["profiles"].append((harness, profile_id))

        def setup_job_create(self, request):
            calls["setup"] += 1
            if request.get("harness") == "one":
                raise refusal
            durable["jobs"].append(dict(request))
            return {"id": "job-created", "state": "queued"}

    class GatewayCtx:
        def __enter__(self):
            return Gateway()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: GatewayCtx())
    with pytest.raises(ClaudexorUnavailable) as unsupported:
        _login_create({"harness": "one", "profile_id": profile_id})
    assert unsupported.value is refusal
    assert calls == expected_calls
    assert durable == {"profiles": [], "jobs": []}


def test_malformed_setup_login_fails_before_any_mutation_or_legacy_fallback(monkeypatch):
    from ouroboros.gateway.claudexor_accounts import _login_create
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    calls = {"operations": 0, "profile": 0, "setup": 0}

    class Gateway:
        def agent_capabilities(self):
            return {"harnesses": [{"id": "one", "setupLogin": {"mode": "future"}}]}
        def operations(self): calls["operations"] += 1
        def create_credential_profile(self, *_args): calls["profile"] += 1
        def setup_job_create(self, *_args): calls["setup"] += 1

    class GatewayCtx:
        def __enter__(self): return Gateway()
        def __exit__(self, *_args): return False

    monkeypatch.setattr(
        "ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: GatewayCtx())
    with pytest.raises(ClaudexorUnavailable) as malformed:
        _login_create({"harness": "one", "profile_id": "named"})
    assert malformed.value.code == "setup_login_capability_malformed"
    assert malformed.value.status_code == 503
    assert calls == {"operations": 0, "profile": 0, "setup": 0}


def test_profile_create_suppresses_only_typed_or_exactly_read_back_duplicate(monkeypatch):
    """Current typed duplicate is sufficient. The old generic 409 needs an
    exact canonical row; every other conflict/failure stops before setup."""
    from ouroboros.gateway.claudexor_accounts import _login_create
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    state = {"error": None, "profiles": {}, "reads": 0, "setups": 0}

    class Gateway:
        def agent_capabilities(self):
            return {"harnesses": [{"id": "one", "setupLogin": {"mode": "in_app"}}]}

        def create_credential_profile(self, *_args):
            if state["error"]:
                raise state["error"]

        def credential_profiles(self):
            state["reads"] += 1
            return state["profiles"]

        def setup_job_create(self, _body):
            state["setups"] += 1
            return {"id": f"job-{state['setups']}", "state": "queued"}

    class GatewayCtx:
        def __enter__(self):
            return Gateway()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: GatewayCtx())

    state["error"] = ClaudexorUnavailable(
        "credential_profile_exists", "already exists", status_code=409)
    assert _login_create({"harness": "one", "profile_id": "named"})["job_id"] == "job-1"
    assert state["reads"] == 0

    state["error"] = ClaudexorUnavailable("http_409", "conflict", status_code=409)
    state["profiles"] = {"profiles": [{"profile": {
        "harness_id": "one", "profile_id": "named",
    }}]}
    assert _login_create({"harness": "one", "profile_id": "named"})["job_id"] == "job-2"
    assert state["reads"] == 1

    # Same id under another harness does not prove this registration.
    state["profiles"] = {"profiles": [{"profile": {
        "harness_id": "other", "profile_id": "named",
    }}]}
    with pytest.raises(ClaudexorUnavailable) as generic:
        _login_create({"harness": "one", "profile_id": "named"})
    assert generic.value is state["error"]
    assert state["setups"] == 2

    # A typed non-duplicate 409 remains a refusal even if a coincidental row
    # exists; exact read-back is reserved for old generic 409 shapes.
    state["error"] = ClaudexorUnavailable(
        "credential_profile_ambiguous", "choose one", status_code=409,
        required_actions=("disable_extra_profiles",))
    state["profiles"] = {"profiles": [{"profile": {
        "harness_id": "one", "profile_id": "named",
    }}]}
    with pytest.raises(ClaudexorUnavailable) as typed:
        _login_create({"harness": "one", "profile_id": "named"})
    assert typed.value.code == "credential_profile_ambiguous"
    assert state["reads"] == 2, "typed conflicts must not trigger another read-back"
    assert state["setups"] == 2


def test_legacy_360_internal_error_duplicate_requires_exact_profile_reread(monkeypatch):
    """The real 3.6.0 409 code is generic ``internal_error``. It is safe only
    after one canonical read finds the exact legacy profile row."""
    from ouroboros.gateway.claudexor_accounts import _login_create
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    legacy_profiles = json.loads((
        pathlib.Path(__file__).resolve().parent.parent
        / "web/tests/fixtures/credential_profiles_response.json"
    ).read_text(encoding="utf-8"))
    state = {"profiles": legacy_profiles, "reads": 0, "setups": 0}
    conflict = ClaudexorUnavailable(
        "internal_error",
        "could not register the profile: duplicate credential profile koshak for harness codex",
        status_code=409,
    )

    class Gateway:
        def agent_capabilities(self):
            return _claudexor_3_6_capabilities("codex")

        def operations(self):
            return [_INPUT_OP]

        def create_credential_profile(self, *_args):
            raise conflict

        def credential_profiles(self):
            state["reads"] += 1
            return state["profiles"]

        def setup_job_create(self, _body):
            state["setups"] += 1
            return {"id": "legacy-job", "state": "queued"}

    class GatewayCtx:
        def __enter__(self):
            return Gateway()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: GatewayCtx())

    answer = _login_create({"harness": "codex", "profile_id": "koshak"})
    assert answer["job_id"] == "legacy-job"
    assert state == {"profiles": legacy_profiles, "reads": 1, "setups": 1}

    state.update({"profiles": {"profiles": []}, "reads": 0, "setups": 0})
    with pytest.raises(ClaudexorUnavailable) as unknown:
        _login_create({"harness": "codex", "profile_id": "koshak"})
    assert unknown.value is conflict
    assert state["reads"] == 1
    assert state["setups"] == 0


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

    capabilities = {"harnesses": [{
        "id": "codex", "setupLogin": {"mode": "in_app"},
    }]}
    external, sent = _create_login(
        monkeypatch, tmp_path,
        {"harness": "codex", "transport": "client_pty", "login_flow": "device_auth"},
        operations=[], capabilities=capabilities, raises=True,
    )
    assert sent["transport"] == "client_pty"
    assert sent["loginFlow"] == "browser_redirect"
    assert external["disclosure_native"] is False
    assert external["attach_command"] == owned.attach_login_command(
        external["job_id"],
        argv=[
            str(tmp_path / "node" / "node"),
            str(tmp_path / "runtime" / "claudexord.bundle.cjs"),
            "setup",
            "attach",
        ],
        shell=external["attach_shell"],
    )


def _missing_cli_job(harness="cursor", job_id="setup-1"):
    return {
        "jobId": job_id,
        "harness": harness,
        "action": "login",
        "state": "not_supported",
        "phase": "completed",
        "outcome": {"reason": "not_supported"},
        "command": None,
    }


def _install_receipt(**updates):
    receipt = {
        "ok": True,
        "dryRun": False,
        "exitCode": 0,
        "target": "local",
        "harness": "cursor",
        "command": "vendor install command",
        "installLocation": "~/.local/bin",
        "installedBinary": "/managed/claudexor/bin/cursor-agent",
        "installedVersion": "1.2.3",
        "pinnedVersion": None,
        "verification": "unattended_unpinned",
    }
    receipt.update(updates)
    return receipt


def test_immediate_missing_cli_trigger_requires_exact_job_and_engine(monkeypatch):
    from types import SimpleNamespace

    from ouroboros import claudexor_runtime as runtime
    from ouroboros.claudexor_daemon import is_immediate_missing_cli_job as _is_immediate_missing_cli_job

    pin = SimpleNamespace(
        version="3.4.0", build_sha="1" * 40, cli_entrypoint="claudexor.bundle.cjs")
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: SimpleNamespace(pin=pin))
    gateway = SimpleNamespace(engine_version=pin.version, engine_build_sha=pin.build_sha)
    job = _missing_cli_job()
    assert _is_immediate_missing_cli_job(job, "cursor", gateway) is True

    mutations = (
        {"state": "failed"},
        {"phase": "verifying"},
        {"outcome": {"reason": "timed_out"}},
        {"command": "claudexor setup attach setup-1"},
        {"authorization": {}},
        {"nativeCommand": {}},
        {"harness": "codex"},
        {"action": "logout"},
        {"jobId": ""},
    )
    for mutation in mutations:
        assert _is_immediate_missing_cli_job({**job, **mutation}, "cursor", gateway) is False
    without_command = dict(job)
    without_command.pop("command")
    assert _is_immediate_missing_cli_job(without_command, "cursor", gateway) is False
    gateway.engine_build_sha = "2" * 40
    assert _is_immediate_missing_cli_job(job, "cursor", gateway) is False
    pin.cli_entrypoint = None
    gateway.engine_build_sha = pin.build_sha
    assert _is_immediate_missing_cli_job(job, "cursor", gateway) is False


@pytest.mark.parametrize("retry_job", [
    {"jobId": "setup-2", "state": "queued"},
    _missing_cli_job(job_id="setup-2"),
])
def test_login_installs_and_retries_exactly_once(monkeypatch, retry_job):
    from types import SimpleNamespace

    from ouroboros import claudexor_runtime as runtime
    from ouroboros.gateway import claudexor_accounts as accounts

    pin = SimpleNamespace(
        version="3.4.0", build_sha="1" * 40, cli_entrypoint="claudexor.bundle.cjs")
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: SimpleNamespace(pin=pin))
    requests, installs = [], []

    class Gateway:
        engine_version = pin.version
        engine_build_sha = pin.build_sha

        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def agent_capabilities(self):
            return {"harnesses": [{"id": "cursor", "setupLogin": None}]}
        def operations(self):
            pytest.fail("current null must not consult the legacy operations catalog")

        def setup_job_create(self, request):
            requests.append(dict(request))
            return _missing_cli_job() if len(requests) == 1 else retry_job

    monkeypatch.setattr(owned, "ensure_owned_gateway", lambda: Gateway())
    monkeypatch.setattr(owned, "install_missing_harness_cli", installs.append)
    answer = accounts._login_create({"harness": "cursor"})

    assert installs == ["cursor"]
    assert len(requests) == 2 and requests[0] == requests[1]
    assert "transport" not in requests[0]
    assert answer["job"] == retry_job
    assert answer["setup_login_source"] == "setup_job_admission"


def test_install_success_receipt_is_strict_and_provenance_is_a_pair():
    from ouroboros.claudexor_daemon import _valid_install_success

    assert _valid_install_success(_install_receipt(), "cursor") is True
    assert _valid_install_success(_install_receipt(
        installerSha256="a" * 64, installerByteLength=123), "cursor") is True
    assert _valid_install_success(_install_receipt(
        harness="codex", verification="release_verified", pinnedVersion="1.2.3",
        installLocation="~/.claudexor/node/bin"), "codex") is True
    assert _valid_install_success(_install_receipt(
        harness="opencode", verification="deterministic_only", pinnedVersion="1.2.3",
        installLocation="~/.claudexor/node/bin"), "opencode") is True

    missing_binary = _install_receipt()
    missing_binary.pop("installedBinary")
    missing_version = _install_receipt()
    missing_version.pop("installedVersion")
    invalid = (
        missing_binary,
        missing_version,
        _install_receipt(pinnedVersion="latest"),
        _install_receipt(verification="release_verified"),
        _install_receipt(installerSha256="a" * 64),
        _install_receipt(installerSha256="A" * 64, installerByteLength=123),
        _install_receipt(installerSha256="a" * 64, installerByteLength=0),
        _install_receipt(installerSha256="a" * 64, installerByteLength=True),
        _install_receipt(
            verification="release_verified", pinnedVersion="1.2.3",
            installerSha256="a" * 64, installerByteLength=123),
        _install_receipt(code="unexpected"),
        _install_receipt(refusal="unexpected"),
        _install_receipt(message="unexpected"),
        _install_receipt(verification="human_observed"),
        _install_receipt(verification={}),
        _install_receipt(exitCode=True),
        _install_receipt(installedBinary="cursor-agent"),
        _install_receipt(installedBinary=""),
        _install_receipt(installedVersion=""),
        _install_receipt(installedVersion="   "),
        _install_receipt(installedVersion="v" * 257),
    )
    assert all(not _valid_install_success(receipt, "cursor") for receipt in invalid)


def test_installer_invocation_is_exact_grouped_and_stdout_bounded(monkeypatch):
    import io
    import subprocess
    from types import SimpleNamespace

    from ouroboros import claudexor_runtime as runtime
    from ouroboros import platform_layer

    command = ["/exact/node", "/exact/closure/claudexor.bundle.cjs"]
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: SimpleNamespace(
        ensure_cli_command=lambda: command))
    monkeypatch.setattr(platform_layer, "subprocess_new_group_kwargs", lambda: {
        "start_new_session": True})
    monkeypatch.setattr(platform_layer, "merge_hidden_kwargs", dict)
    seen = {}

    class Process:
        stdout = io.BytesIO(json.dumps(_install_receipt()).encode())

        def wait(self, timeout):
            seen.setdefault("timeouts", []).append(timeout)
            return 0

    import ouroboros.tools.shell as shell
    from ouroboros.claudexor_daemon import owned_config_dir

    def popen(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        # /panic snapshots the tracked set under this lock: the spawn must be
        # atomic with registration, so the lock is held HERE (round-2 finding).
        seen["lock_held_during_spawn"] = shell._subprocess_lock.locked()
        return Process()

    monkeypatch.setattr(owned.subprocess, "Popen", popen)
    owned.install_missing_harness_cli("cursor")
    assert seen["argv"] == [
        *command, "harness", "install", "cursor",
        "--target", "local", "--yes", "--json",
    ]
    env = seen["kwargs"].pop("env")
    assert seen["kwargs"] == {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
        "start_new_session": True,
    }
    # The installer acts on the OWNED data plane, exactly like the daemon it
    # repairs — never the operator's personal Claudexor home.
    assert env["CLAUDEXOR_CONFIG_DIR"] == str(owned_config_dir())
    assert "CLAUDEXOR_DAEMON_SOCK" not in env
    assert "CLAUDEXOR_CONTROL_PORT" not in env
    assert seen["lock_held_during_spawn"] is True
    output, state = bytearray(), {}
    owned._drain_installer_stdout(
        io.BytesIO(b"x" * (owned._HARNESS_INSTALL_STDOUT_LIMIT + 1)), output, state)
    assert len(output) == owned._HARNESS_INSTALL_STDOUT_LIMIT
    assert state == {"overflow": True}


def test_installer_timeout_kills_the_process_tree(monkeypatch):
    import io
    import subprocess
    from types import SimpleNamespace

    from ouroboros import claudexor_runtime as runtime
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros import platform_layer

    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: SimpleNamespace(
        ensure_cli_command=lambda: ["/exact/node", "/exact/cli.cjs"]))
    monkeypatch.setattr(platform_layer, "subprocess_new_group_kwargs", lambda: {})
    monkeypatch.setattr(platform_layer, "merge_hidden_kwargs", dict)
    waits = []

    class Process:
        pid = 42
        stdout = io.BytesIO()

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("installer", timeout)
            return -9

    proc = Process()
    monkeypatch.setattr(owned.subprocess, "Popen", lambda *_a, **_kw: proc)
    killed = []
    import ouroboros.tools.shell as shell

    monkeypatch.setattr(shell, "_kill_process_group", killed.append)
    with pytest.raises(ClaudexorUnavailable) as excinfo:
        owned.install_missing_harness_cli("cursor")
    assert excinfo.value.code == "harness_install_timeout"
    assert killed == [proc]
    from ouroboros.config import get_claudexor_harness_install_timeout_sec

    assert waits == [get_claudexor_harness_install_timeout_sec(), 10]
    assert proc not in shell._active_subprocesses, "a timed-out installer must not stay tracked"


def test_installer_child_is_panic_tracked_for_its_whole_run(monkeypatch):
    """/panic kills only tools.shell-tracked subprocess trees before os._exit
    (BIBLE: every subprocess tree stops immediately). The vendor installer must
    therefore sit in that set exactly while it runs — the allowlisted raw Popen
    is legitimate ONLY because of this registration."""
    import io
    from types import SimpleNamespace

    from ouroboros import claudexor_runtime as runtime
    from ouroboros import platform_layer
    import ouroboros.tools.shell as shell

    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: SimpleNamespace(
        ensure_cli_command=lambda: ["/exact/node", "/exact/cli.cjs"]))
    monkeypatch.setattr(platform_layer, "subprocess_new_group_kwargs", lambda: {})
    monkeypatch.setattr(platform_layer, "merge_hidden_kwargs", dict)
    observed = {}

    class Process:
        pid = 43
        stdout = io.BytesIO(json.dumps(_install_receipt()).encode())

        def wait(self, timeout):
            observed["tracked_during_wait"] = self in shell._active_subprocesses
            return 0

    proc = Process()
    monkeypatch.setattr(owned.subprocess, "Popen", lambda *_a, **_kw: proc)
    owned.install_missing_harness_cli("cursor")
    assert observed["tracked_during_wait"] is True, "panic sweep must see the live installer"
    assert proc not in shell._active_subprocesses, "custody ends with the request"


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


# ---------------------------------------------------------------------------
# Phase 6, owner directive #1: the executor fact reaches the chat frame.
# «бейдж точно нужен, но не рекламный … что ТУТ бабл \ субагент на codex»
# ---------------------------------------------------------------------------


def _agent_with_metadata(task, task_id="child-1"):
    import types

    from ouroboros.agent import OuroborosAgent

    agent = object.__new__(OuroborosAgent)
    agent._current_task_metadata = {
        "delegation_role": "subagent", "role": "impl", "root_task_id": "r",
        "parent_task_id": "p", "model": "m", "task_group_id": "g",
    }
    agent._current_task_id = task_id
    # Since synthesis the fact is read from the ONE record the dispatch
    # resolution stamped onto the task (`resolve_subagent_dispatch` ->
    # record_fields) — the same principle this file always asserted ("a
    # projection of the decision, never a second derivation"), one level
    # stronger: the projection reads the durable record, not a live object.
    agent._record_executor_facts(task if isinstance(task, dict) else {})
    return agent, types


def test_resolved_harness_route_reaches_the_frame_assembler():
    """The chip's fact comes from the ONE place the executor was decided: the
    dispatch resolution is stamped onto the live metadata that the canonical
    frame assembler already projects — never re-derived per surface."""
    agent, _ = _agent_with_metadata(
        {"effective_executor": "harness", "executor_route": "codex"})
    frame = agent._subagent_progress_meta("running")
    assert frame["executor_route"] == "codex"
    # The frame keeps carrying the execution facts it always did.
    assert frame["subagent_event"] == "running"
    assert frame["delegation_role"] == "subagent"


def test_no_executor_fact_when_the_run_is_native_blocked_or_undecided():
    """Absent fact -> empty/absent, so the renderer draws NO chip: the native
    API path is the ordinary case and must not print 'api' on every bubble."""
    native, _ = _agent_with_metadata(
        {"effective_executor": "native", "executor_route": ""}, "child-2")
    assert native._subagent_progress_meta("running")["executor_route"] == ""
    # A blocked or unresolved dispatch records nothing at all.
    blocked, _ = _agent_with_metadata(
        {"effective_executor": "blocked", "executor_route": "codex"}, "child-3")
    assert "executor_route" not in blocked._current_task_metadata
    undecided, _ = _agent_with_metadata({}, "child-4")
    assert "executor_route" not in undecided._current_task_metadata


def test_the_executor_fact_survives_history_replay_and_the_frozen_contract():
    """End-to-end plumbing: the field is in the progress-meta allowlist (so a
    reloaded bubble keeps its chip) and in BOTH contract mirrors."""
    from ouroboros.gateway.contracts import ChatOutbound
    from ouroboros.gateway.history import _PROGRESS_META_FIELDS

    assert "executor_route" in _PROGRESS_META_FIELDS
    assert "executor_route" in ChatOutbound.__annotations__
    js = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "api_types.js")
    assert "executor_route" in js.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Stale owned-daemon lifecycle (owner directive, pre-synthesis): dead -> restart
# under the same supervision + reconcile; alive-but-foreign -> typed disclosure,
# no kill; foreign home -> never adopt.
# ---------------------------------------------------------------------------



def _stale_home(config_dir: pathlib.Path, *, marker_data_dir: str) -> None:
    """A provisioned home whose daemon is DEAD: descriptor points at a closed
    port, token present, ownership marker written."""
    import json
    import socket

    daemon_dir = config_dir / "daemon"
    daemon_dir.mkdir(parents=True, exist_ok=True)
    (daemon_dir / "token").write_text("tok-dead", encoding="utf-8")
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()  # port free again -> connection refused = dead daemon
    (daemon_dir / "control-api.json").write_text(json.dumps({
        "host": "127.0.0.1", "port": dead_port,
        "tokenPath": str(daemon_dir / "token"),
    }), encoding="utf-8")
    (config_dir / owned.OWNERSHIP_MARKER).write_text(json.dumps({
        "owner": "ouroboros", "data_dir": marker_data_dir,
    }), encoding="utf-8")


def _point_owned_home(monkeypatch, config_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
    monkeypatch.setattr(owned, "owned_config_dir", lambda: config_dir)
    monkeypatch.setattr(owned, "owned_descriptor_path",
                        lambda: config_dir / "daemon" / "control-api.json")
    monkeypatch.setattr(owned, "owned_daemon_provisioned",
                        lambda: (config_dir / "daemon" / "control-api.json").is_file())
    import ouroboros.config as config_mod
    monkeypatch.setattr(config_mod, "DATA_DIR", data_dir)


def test_a_spawn_that_never_publishes_a_descriptor_does_not_leave_the_child_running(
        monkeypatch, tmp_path):
    """The timeout branch raised its typed refusal and walked away from the process it
    had just started. That child is OURS and it is alive — holding the config dir, its
    log, and whatever port it eventually binds — and `self._proc` still pointed at it,
    so the NEXT `ensure_running` spawned a SECOND daemon beside the first. Every retry
    added one. `stop()` could not clean up either: by contract it only ever terminates
    a daemon we successfully started, and this one never became reachable."""
    import sys

    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", sys.executable)
    monkeypatch.setattr(owned, "_SPAWN_WAIT_SEC", 0.3)
    monkeypatch.setattr(owned, "_SPAWN_POLL_SEC", 0.05)

    class _NeverReadyChild:
        """Alive, but it never writes a descriptor — so discovery never succeeds."""

        pid = 424242

        def __init__(self):
            self.terminated = 0

        def poll(self):
            return None                      # still running

        def terminate(self):
            self.terminated += 1

    child = _NeverReadyChild()
    import ouroboros.platform_layer as platform_layer
    monkeypatch.setattr(platform_layer, "process_group_id", lambda _pid: 0)
    monkeypatch.setattr(owned, "spawn_supervised", lambda *a, **k: child, raising=False)
    import ouroboros.process_custody as custody_mod
    monkeypatch.setattr(custody_mod, "spawn_supervised", lambda *a, **k: child)

    manager = owned.OwnedClaudexorDaemon()
    with pytest.raises(ClaudexorUnavailable) as err:
        manager.ensure_running()
    assert err.value.code == "daemon_spawn_failed"

    # The child we started is stopped, and the handle is forgotten so the next
    # attempt starts ONE daemon rather than a second one beside a live orphan.
    assert child.terminated == 1, "the spawned child was left running"
    assert manager._proc is None
    assert manager.stop() is False


def test_first_spawn_loser_attaches_to_the_winners_endpoint(monkeypatch, tmp_path):
    """A concurrent first-use winner is success, not a false spawn failure."""
    from ouroboros import claudexor_runtime as runtime
    from ouroboros.gateways.claudexor import DaemonEndpoint

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    monkeypatch.setattr(owned, "verify_owned_home", lambda: "")

    class ReadyRuntime:
        def ensure(self):
            return ["/fixture/node", "/fixture/claudexord.bundle.cjs"]

        def status(self):
            return {"source": "download"}

    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: ReadyRuntime())

    class ExitedLoser:
        pid = 424243

        def poll(self):
            return 1

        def terminate(self):
            raise AssertionError("an exited loser must not be terminated")

    child = ExitedLoser()
    import ouroboros.process_custody as custody_mod

    monkeypatch.setattr(custody_mod, "spawn_supervised", lambda *_args, **_kwargs: child)
    endpoint = DaemonEndpoint(host="127.0.0.1", port=45681, token="winner-token")
    manager = owned.OwnedClaudexorDaemon()
    monkeypatch.setattr(manager, "_classify_liveness", lambda: (None, "not_provisioned", ""))
    monkeypatch.setattr(manager, "_alive_endpoint", lambda: endpoint)

    assert manager.ensure_running() is endpoint
    assert manager._proc is None
    assert manager.stop() is False


def test_spawn_loser_keeps_polling_for_delayed_winner_after_child_exit(monkeypatch, tmp_path):
    """An exited first-spawn loser must not end the winner's publication window."""
    from ouroboros import claudexor_runtime as runtime
    from ouroboros.gateways.claudexor import DaemonEndpoint

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    monkeypatch.setattr(owned, "verify_owned_home", lambda: "")
    monkeypatch.setattr(owned, "_SPAWN_WAIT_SEC", 0.15)
    monkeypatch.setattr(owned, "_SPAWN_POLL_SEC", 0.01)

    class ReadyRuntime:
        def ensure(self):
            return ["/fixture/node", "/fixture/claudexord.bundle.cjs"]

        def status(self, *args, **kwargs):
            return {"source": "download"}

    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: ReadyRuntime())

    class ExitedLoser:
        pid = 424244

        def poll(self):
            return 1

        def terminate(self):
            raise AssertionError("an exited loser must not be terminated")

    child = ExitedLoser()
    import ouroboros.process_custody as custody_mod

    monkeypatch.setattr(custody_mod, "spawn_supervised", lambda *_args, **_kwargs: child)
    endpoint = DaemonEndpoint(host="127.0.0.1", port=45682, token="winner-token")
    polls = iter([None, None, endpoint])
    seen = []

    def delayed_alive_endpoint():
        seen.append(True)
        return next(polls, endpoint)

    manager = owned.OwnedClaudexorDaemon()
    monkeypatch.setattr(manager, "_classify_liveness", lambda: (None, "not_provisioned", ""))
    monkeypatch.setattr(manager, "_alive_endpoint", delayed_alive_endpoint)

    assert manager.ensure_running() is endpoint
    assert len(seen) >= 3
    assert manager._proc is None


def test_dead_owned_daemon_is_restarted_and_reconciled(monkeypatch, tmp_path):
    """The stale case end-to-end: descriptor exists, daemon dead, ownership
    marker OURS -> ensure_running restarts under the same supervision
    chokepoint and reconciles by fresh discovery + an AUTHENTICATED handshake
    against the NEW descriptor the restarted daemon wrote.

    The scripted daemon serves its /v2/handshake IN-PROCESS (the sandbox kills
    exec'd children that bind sockets); the supervised child is a harmless
    sleeper, so the supervision arguments and the stop() path stay real.
    """
    import http.server
    import json as _json
    import subprocess as sp
    import sys
    import threading

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    _stale_home(config_dir, marker_data_dir=str(data_dir.resolve()))
    old_descriptor = (config_dir / "daemon" / "control-api.json").read_text()

    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", sys.executable)
    spawned: dict = {}
    servers: list = []
    import ouroboros.process_custody as custody_mod

    def fake_spawn(cmd, **kwargs):
        # The SAME chokepoint ensure_running calls; acts like claudexord:
        # mint a fresh token, serve an authenticated handshake, REWRITE the
        # discovery descriptor — then hand back a real supervised child.
        spawned["cmd"] = list(cmd)
        spawned["kwargs"] = {k: kwargs.get(k) for k in ("purpose", "scope")}
        home = pathlib.Path(kwargs["env"]["CLAUDEXOR_CONFIG_DIR"])
        daemon_dir = home / "daemon"
        daemon_dir.mkdir(parents=True, exist_ok=True)
        token = "tok-restarted"
        (daemon_dir / "token").write_text(token, encoding="utf-8")

        class _Daemon(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                # Drain the request body or the NEXT keep-alive request parses
                # mid-stream (the "{json}GET" unsupported-method failure shape).
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                ok = self.headers.get("Authorization") == f"Bearer {token}"
                body = _json.dumps({"compatible": True, "protocolMajor": 3,
                                    "engine": {"version": "9.9.9"}}).encode() if ok else b"{}"
                self.send_response(200 if ok else 401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), _Daemon)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        (daemon_dir / "control-api.json").write_text(_json.dumps({
            "host": "127.0.0.1", "port": server.server_address[1],
            "tokenPath": str(daemon_dir / "token"),
        }), encoding="utf-8")
        # A NEW process group, exactly like the real spawn_supervised child:
        # stop() kills by GROUP id, and a group-sharing fake would take the
        # test process down with it (the SIGKILL-137 this fixture first hit).
        return sp.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                        stdin=sp.DEVNULL, stdout=sp.DEVNULL, stderr=sp.DEVNULL,
                        start_new_session=True)

    monkeypatch.setattr(custody_mod, "spawn_supervised", fake_spawn)

    manager = owned.OwnedClaudexorDaemon()
    assert manager.status_dict()["state"] == "stale"
    try:
        endpoint = manager.ensure_running()
        # Reconciled: the NEW descriptor was re-read and answered our token.
        new_descriptor = (config_dir / "daemon" / "control-api.json").read_text()
        assert new_descriptor != old_descriptor
        assert endpoint.port == _json.loads(new_descriptor)["port"]
        assert spawned["kwargs"] == {"purpose": "claudexor_daemon", "scope": "session"}
        assert manager.status_dict()["state"] == "running"
        # The provision moment (re)wrote OUR ownership marker.
        assert owned.read_ownership_marker()["data_dir"] == str(data_dir.resolve())
        # Restart-only-ours: stop() terminates the SELF-STARTED child.
        assert manager.stop() is True
    finally:
        manager.stop()
        for server in servers:
            server.shutdown()


def test_foreign_responder_on_stale_port_is_disclosed_not_killed(monkeypatch, tmp_path):
    """A live daemon that REFUSES our token on the stale port is foreign:
    typed disclosure, no kill — and it does not block restarting OUR daemon."""
    import http.server
    import json as _json
    import threading

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    _stale_home(config_dir, marker_data_dir=str(data_dir.resolve()))

    class _Refuser(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = b"{}"
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    foreign = http.server.HTTPServer(("127.0.0.1", 0), _Refuser)
    threading.Thread(target=foreign.serve_forever, daemon=True).start()
    try:
        descriptor = config_dir / "daemon" / "control-api.json"
        body = _json.loads(descriptor.read_text())
        body["port"] = foreign.server_address[1]
        descriptor.write_text(_json.dumps(body), encoding="utf-8")

        manager = owned.OwnedClaudexorDaemon()
        status = manager.status_dict()
        assert status["state"] == "foreign_daemon"
        assert "REFUSED our home's token" in (status["last_error"] or "")
        # No kill: stop() only ever touches a self-started process.
        assert manager.stop() is False
    finally:
        foreign.shutdown()


def _exact_runtime_pin(runtime_mod):
    """A valid exact pin (all five Node platforms) for lifecycle fixtures."""
    node_artifacts = {
        key: runtime_mod.NodeRuntimeArtifact(
            archive_url=f"https://node.example.test/node-v24.16.0-{key}.tar.gz",
            sha256="a" * 64,
            size_bytes=1,
            executable=f"node-v24.16.0-{key}/bin/node",
        )
        for key in ("darwin-arm64", "darwin-x64", "linux-arm64", "linux-x64", "win32-x64")
    }
    return runtime_mod.ClaudexorRuntimePin(
        version="3.4.0",
        build_sha="1" * 40,
        protocol_major=3,
        archive_url="https://example.test/releases/runtime.tar.gz",
        sha256="b" * 64,
        size_bytes=1,
        node_version="24.16.0",
        node_artifacts=node_artifacts,
        entrypoint="dist/claudexord.js",
    )


def test_live_daemon_serving_the_exact_pin_is_never_repaired_in_place(monkeypatch, tmp_path):
    """S1 guard: when the live authenticated handshake already matches the pin
    (version + build SHA), ensure_running returns the live endpoint WITHOUT
    touching its serving directory — even when the on-disk copy of that SAME
    target is broken. Disk repair belongs to the next natural start (owner
    decision 2A: side-by-side, current work is never touched)."""
    from ouroboros import claudexor_runtime as runtime
    from ouroboros.gateways.claudexor import DaemonEndpoint

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)

    pin = _exact_runtime_pin(runtime)
    manager = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)

    # The daemon's own pinned target on disk is corrupted: any disk probe of
    # this target fails, which without the guard would drive _install and a
    # promote that replaces the serving directory under the live process.
    target = runtime.managed_runtime_dir(pin)
    target.mkdir(parents=True)
    (target / "managed-runtime.json").write_text("{corrupt", encoding="utf-8")

    def no_install(*_args, **_kwargs):
        raise AssertionError("a live pinned daemon's serving directory must not be repaired")

    monkeypatch.setattr(manager, "_install", no_install)
    import ouroboros.process_custody as custody_mod

    def no_spawn(*_args, **_kwargs):
        raise AssertionError("no spawn may happen while the pinned daemon is live")

    monkeypatch.setattr(custody_mod, "spawn_supervised", no_spawn)

    endpoint = DaemonEndpoint(host="127.0.0.1", port=45695, token="live")
    daemon = owned.OwnedClaudexorDaemon()

    def live_classify():
        daemon._engine_version = pin.version
        daemon._engine_build_sha = pin.build_sha
        return endpoint, "running", ""

    monkeypatch.setattr(daemon, "_classify_liveness", live_classify)
    before = sorted(p.name for p in target.parent.iterdir())

    assert daemon.ensure_running() is endpoint
    # The serving directory was not replaced, repaired, or cleaned up.
    assert (target / "managed-runtime.json").read_text(encoding="utf-8") == "{corrupt"
    assert sorted(p.name for p in target.parent.iterdir()) == before
    assert daemon._proc is None


def test_staged_update_activates_only_at_the_next_natural_start(monkeypatch, tmp_path):
    """Staged-activation orchestration (owner decision 2A), end to end: with a
    live OLD daemon, ensure() stages a DIFFERENT exact target and
    ensure_running still answers the old endpoint with no spawn and no stop;
    once the old daemon dies naturally, the next start spawns exactly the
    staged command."""
    from ouroboros import claudexor_runtime as runtime
    from ouroboros.gateways.claudexor import DaemonEndpoint

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    monkeypatch.setattr(owned, "verify_owned_home", lambda: "")

    new_command = ["/fixture/node", "/fixture/state/cx/3.4.0-111111111111/dist/claudexord.js"]
    ensures: list = []

    class _StagedPin:
        version = "3.4.0"
        build_sha = "1" * 40

    class StagingRuntime:
        pin = _StagedPin()

        def ensure(self):
            ensures.append("ensure")
            return list(new_command)

        def status(self):
            return {"source": "download"}

    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: StagingRuntime())

    old_endpoint = DaemonEndpoint(host="127.0.0.1", port=45696, token="old")
    new_endpoint = DaemonEndpoint(host="127.0.0.1", port=45697, token="new")
    daemon = owned.OwnedClaudexorDaemon()
    old_alive = {"value": True}

    def classify():
        if old_alive["value"]:
            daemon._engine_version = "3.2.1"
            daemon._engine_build_sha = "2" * 40
            return old_endpoint, "running", ""
        daemon._engine_version = ""
        daemon._engine_build_sha = ""
        return None, "stale", "connection refused"

    monkeypatch.setattr(daemon, "_classify_liveness", classify)

    spawns: list = []

    class _LiveChild:
        pid = 424244

        def poll(self):
            return None

        def terminate(self):
            raise AssertionError("staging must never stop a daemon")

    import ouroboros.process_custody as custody_mod

    monkeypatch.setattr(
        custody_mod,
        "spawn_supervised",
        lambda command, **_kwargs: spawns.append(list(command)) or _LiveChild(),
    )
    monkeypatch.setattr(daemon, "_alive_endpoint", lambda: new_endpoint)

    # Phase 1: the OLD endpoint keeps serving; the new target is only staged.
    assert daemon.ensure_running() is old_endpoint
    assert ensures == ["ensure"]
    assert spawns == []
    assert daemon._proc is None  # nothing spawned, nothing stopped

    # Phase 2: the old daemon died naturally; the next start selects the
    # staged exact target the previous ensure() prepared.
    old_alive["value"] = False
    assert daemon.ensure_running() is new_endpoint
    assert ensures == ["ensure", "ensure"]
    assert spawns == [new_command]


def test_a_home_marked_for_another_data_plane_is_never_adopted(monkeypatch, tmp_path):
    """The never-adopt rule: a marker naming a different data plane makes
    ensure_running refuse typed BEFORE any spawn — restart there = adoption."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    _stale_home(config_dir, marker_data_dir=str(tmp_path / "someone-elses-data"))
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", "/bin/true")

    manager = owned.OwnedClaudexorDaemon()
    assert manager.status_dict()["ownership_problem"]
    with pytest.raises(ClaudexorUnavailable) as err:
        manager.ensure_running()
    assert err.value.code == "foreign_daemon_home"


def test_spawn_env_prepends_onto_the_hosts_own_path_key(monkeypatch, tmp_path):
    """WINDOWS PATH-KEY REGRESSION (first live 3-OS gate run on the 3.3.8 pin):
    os.environ materializes with the native "Path" key there, a plain dict
    lookup of "PATH" misses it, and the spawned daemon received a PATH holding
    only the Node bin dir — the engine then refused with git_missing. The env
    builder must prepend onto whichever casing the host actually has and must
    not add a second, differently-cased key beside it."""
    import os as os_mod
    import sys

    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", sys.executable)
    monkeypatch.setattr(owned, "_SPAWN_WAIT_SEC", 0.05)
    monkeypatch.setattr(owned, "_SPAWN_POLL_SEC", 0.01)

    # Windows-style environment: the variable exists only as "Path".
    fake_environ = {k: v for k, v in os_mod.environ.items() if k.upper() != "PATH"}
    fake_environ["Path"] = "C:/hostedtoolcache/git/bin"
    monkeypatch.setattr(os_mod, "environ", fake_environ)

    captured = {}

    class _NeverReadyChild:
        pid = 424244

        def poll(self):
            return None

        def terminate(self):
            pass

    def _capture_spawn(command, **kwargs):
        captured["env"] = kwargs["env"]
        return _NeverReadyChild()

    import ouroboros.platform_layer as platform_layer

    monkeypatch.setattr(platform_layer, "process_group_id", lambda _pid: 0)
    import ouroboros.process_custody as custody_mod

    monkeypatch.setattr(custody_mod, "spawn_supervised", _capture_spawn)

    manager = owned.OwnedClaudexorDaemon()
    with pytest.raises(ClaudexorUnavailable):
        manager.ensure_running()

    env = captured["env"]
    node_bin = str(pathlib.Path(sys.executable).parent)
    assert "PATH" not in env, "a second differently-cased PATH key was added"
    assert env["Path"].startswith(node_bin + os_mod.pathsep)
    assert env["Path"].endswith("C:/hostedtoolcache/git/bin")


def test_spawn_env_never_leaves_an_empty_path_component(monkeypatch, tmp_path):
    """EMPTY PATH COMPONENT == CWD. The env builder composes the child's PATH as
    f"{command_bin}{os.pathsep}{inherited}", so a host whose environment carries
    no PATH at all (a scrubbed service manager, a bare container unit) yields a
    TRAILING EMPTY component -- and an empty component means the current working
    directory on POSIX. That would make CWD an executable search root for a
    long-lived daemon that shells out to tools of its own.

    Measured, not argued: with a trailing-empty PATH a bare-name binary in the
    working directory EXECUTED (rc 0); the same exec with the empty component
    removed raised FileNotFoundError."""
    import os as os_mod
    import sys

    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", sys.executable)
    monkeypatch.setattr(owned, "_SPAWN_WAIT_SEC", 0.05)
    monkeypatch.setattr(owned, "_SPAWN_POLL_SEC", 0.01)

    # A host environment with no PATH key in any casing.
    fake_environ = {k: v for k, v in os_mod.environ.items() if k.upper() != "PATH"}
    monkeypatch.setattr(os_mod, "environ", fake_environ)

    captured = {}

    class _NeverReadyChild:
        pid = 424245

        def poll(self):
            return None

        def terminate(self):
            pass

    def _capture_spawn(command, **kwargs):
        captured["env"] = kwargs["env"]
        return _NeverReadyChild()

    import ouroboros.platform_layer as platform_layer

    monkeypatch.setattr(platform_layer, "process_group_id", lambda _pid: 0)
    import ouroboros.process_custody as custody_mod

    monkeypatch.setattr(custody_mod, "spawn_supervised", _capture_spawn)

    manager = owned.OwnedClaudexorDaemon()
    with pytest.raises(ClaudexorUnavailable):
        manager.ensure_running()

    env = captured["env"]
    path_key = next((k for k in env if k.upper() == "PATH"), "")
    assert path_key, "the child received no PATH at all"
    components = env[path_key].split(os_mod.pathsep)
    assert "" not in components, (
        "an empty PATH component reached the daemon's child environment "
        f"({env[path_key]!r}); on POSIX that is the current working directory"
    )
    assert components[0] == str(pathlib.Path(sys.executable).parent)


def test_status_payload_fans_out_the_independent_daemon_reads(monkeypatch, tmp_path):
    """The four catalog/manifest/profile/quota GETs run CONCURRENTLY.

    Each costs seconds daemon-side (it re-probes the coding-agent CLIs on every
    read), so serialized they made the Providers panel wait for their SUM — ~23s
    on a warm daemon, with nothing on screen (owner report, 2026-08-08).

    The pin is a rendezvous, not a stopwatch: all four reads must meet at one
    barrier before any returns, which only a genuine fan-out can do. Serialized
    code times out at the barrier instead of failing on a wall-clock margin no
    loaded CI machine can honor (review lens, 2026-08-08).
    """
    import threading

    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw

    rendezvous = threading.Barrier(4, timeout=10)

    class FakeDaemon:
        def status_dict(self):
            return {"state": "running"}

    class FakeGateway:
        engine_version = "3.3.13"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            rendezvous.wait()
            return {"harnesses": [{"id": "codex", "displayName": "Codex CLI",
                                   "status": "ok", "enabled": True}]}

        def harnesses(self):
            rendezvous.wait()
            return [{"id": "codex", "manifest": {"capability_profile": {"auth": {
                "supported_sources": ["native_session"]}}}}]

        def credential_profiles(self):
            rendezvous.wait()
            return {"profiles": [], "harnessAccounts": []}

        def quota_snapshots(self):
            rendezvous.wait()
            return [{"subject": {"harness": "codex"}}]

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    # Serialized, the first read blocks forever waiting for siblings that never
    # start: the barrier breaks and the call raises instead of quietly passing.
    payload = _status_payload(include_models=False)

    assert [h["id"] for h in payload["harnesses"]] == ["codex"]
    assert payload["quota"] and payload["profiles"] == {"profiles": [], "harnessAccounts": []}


def test_status_payload_keeps_typed_unreachable_when_a_fanned_out_read_refuses(monkeypatch, tmp_path):
    """Concurrency must not change WHAT a refusal means: a catalog read that
    raises still lands as the typed unreachable daemon state, not a half-filled
    panel that looks healthy."""
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    class FakeDaemon:
        def status_dict(self):
            return {"state": "running"}

    class FakeGateway:
        engine_version = "3.3.13"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            raise ClaudexorUnavailable("daemon_unreachable", "gone mid-read")

        def harnesses(self):
            return []

        def credential_profiles(self):
            return {}

        def quota_snapshots(self):
            return []

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    payload = _status_payload(include_models=False)
    assert payload["daemon"]["state"] == "unreachable"
    assert "daemon_unreachable" in payload["daemon"]["last_error"]
    assert payload["harnesses"] == []


def _reads_probe(monkeypatch, tmp_path, daemon_state, failing_facet="", malformed=None):
    """Drive _status_payload with one facet optionally refusing, or answering
    with a body that does not carry the envelope it promised — either one the
    transport already collapsed to ``{}``, or an object that kept only half of
    the keys it owes."""
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    class FakeDaemon:
        def status_dict(self):
            return {"state": daemon_state}

    def refuse_if(name):
        if failing_facet == name:
            raise ClaudexorUnavailable("daemon_unreachable", f"{name} refused")

    class FakeGateway:
        engine_version = "3.3.13"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            refuse_if("catalog")
            if malformed == "catalog":
                return {}
            return {"harnesses": [{"id": "codex", "displayName": "Codex CLI",
                                   "status": "ok", "enabled": True}]}

        def harnesses(self):
            return [{"id": "codex", "manifest": {"capability_profile": {"auth": {
                "supported_sources": ["native_session"]}}}}]

        def credential_profiles(self):
            refuse_if("accounts")
            if malformed == "accounts":
                return {}
            if malformed == "accounts_half_named":
                return {"profiles": []}          # native rows key missing
            if malformed == "accounts_half_native":
                return {"harnessAccounts": []}   # named profiles key missing
            return {"profiles": [], "harnessAccounts": []}

        def quota_snapshots(self):
            refuse_if("quota")
            return []

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)
    return _status_payload(include_models=False)


@pytest.mark.parametrize("facet", ["catalog", "accounts"])
def test_status_payload_calls_a_normalized_empty_envelope_a_failed_read(
    monkeypatch, tmp_path, facet
):
    """A NON-OBJECT 2xx body — null, a list, a string — is collapsed by the
    transport into an empty ``{}`` (``ClaudexorGateway.agent_capabilities`` /
    ``credential_profiles`` both end in
    ``return body if isinstance(body, dict) else {}``), so it arrives looking
    like a legitimate empty answer. Without the envelope check it is published
    as `ok` — an AUTHORITATIVE nothing — and one daemon-side schema drift
    silently restores the owner-visible lie. The type check alone cannot see
    this: `{}` IS a dict. (A drifted OBJECT is not normalized at all; it reaches
    the same verdict through the same check — see the half-envelope test.)"""
    payload = _reads_probe(monkeypatch, tmp_path, "running", malformed=facet)

    assert payload["reads"][facet] == "failed", "a body that answered nothing read as ok"
    # The SIBLING facets are untouched — the envelope check must not become a
    # second way for one read to speak for another.
    for other in ("catalog", "accounts", "quota"):
        if other != facet:
            assert payload["reads"][other] == "ok"


@pytest.mark.parametrize("daemon_state", ["not_provisioned", "stale", "foreign_daemon"])
def test_status_payload_marks_every_facet_unread_when_the_daemon_is_not_running(
    monkeypatch, tmp_path, daemon_state
):
    """A lazily-started daemon is the ORDINARY idle state, so this is the path the
    owner actually sees. Empty collections here mean "never asked" — the panel
    printed "no account connected" for three harnesses while two claude profiles,
    a cursor profile and two native sessions sat on disk (owner report,
    2026-08-08). The payload must SAY it was not read (BIBLE P1: a gap is a gap)."""
    payload = _reads_probe(monkeypatch, tmp_path, daemon_state)

    assert payload["reads"] == {
        "catalog": "not_read", "accounts": "not_read", "quota": "not_read",
    }
    assert payload["harnesses"] == [] and payload["profiles"] == {}


def test_status_payload_marks_facets_ok_when_the_daemon_answered(monkeypatch, tmp_path):
    """The other half of the contract: after a successful read an EMPTY collection
    is authoritative — it really does mean "no account" — otherwise the honest
    hedge would never step aside and the panel could never say anything."""
    payload = _reads_probe(monkeypatch, tmp_path, "running")

    assert payload["reads"] == {"catalog": "ok", "accounts": "ok", "quota": "ok"}
    assert [h["id"] for h in payload["harnesses"]] == ["codex"]


def test_status_payload_discloses_a_refused_per_harness_model_read(monkeypatch, tmp_path):
    """`include=models` asks the daemon for each harness's model list separately,
    and one of those reads can refuse while the catalog itself landed. The row
    carries `models_error` so the UI can say "not checked" instead of calling a
    saved model undiscovered — a verdict about a search nobody ran. Nothing
    pinned the field's emission, so dropping it left every suite green."""
    from ouroboros.gateway.claudexor_accounts import _status_payload
    from ouroboros.gateways import claudexor as gw
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    class FakeDaemon:
        def status_dict(self):
            return {"state": "running"}

    class FakeGateway:
        engine_version = "3.3.13"

        def __init__(self, endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            return {"harnesses": [{"id": "codex", "displayName": "Codex CLI",
                                   "status": "ok", "enabled": True}]}

        def harnesses(self):
            return [{"id": "codex", "manifest": {"capability_profile": {"auth": {
                "supported_sources": ["native_session"]}}}}]

        def harness_models(self, harness_id):
            raise ClaudexorUnavailable("daemon_unreachable", "model read refused")

        def credential_profiles(self):
            return {"profiles": [], "harnessAccounts": []}

        def quota_snapshots(self):
            return []

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: FakeDaemon())
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(gw, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gw, "ClaudexorGateway", FakeGateway)

    payload = _status_payload(include_models=True)

    row = payload["harnesses"][0]
    assert row["models"] == []
    assert row["models_error"] == "daemon_unreachable", (
        "a refused model read is indistinguishable from a harness with no models"
    )
    # The CATALOG itself answered, so its facet stays authoritative.
    assert payload["reads"]["catalog"] == "ok"


@pytest.mark.parametrize("half", ["accounts_half_named", "accounts_half_native"])
def test_status_payload_calls_half_an_account_envelope_a_failed_read(monkeypatch, tmp_path, half):
    """The accounts envelope carries TWO collections — named credential profiles
    and the daemon's native per-harness rows — and the owner's machine has both
    kinds. Accepting an envelope that brought only one made the missing half an
    authoritative empty: exactly the reported bug ("no account connected" beside
    accounts that exist), reached through a half-answer instead of a lazy daemon.
    The engine schema declares both inside a strict object
    (`ControlCredentialProfilesResponse`): `profiles` is REQUIRED outright and
    `harnessAccounts` carries `.default([])`, so a validating daemon always
    materializes the pair and a body missing either one is a read that did not
    answer."""
    payload = _reads_probe(monkeypatch, tmp_path, "running", malformed=half)

    assert payload["reads"]["accounts"] == "failed", (
        "half an account envelope was published as an authoritative empty"
    )
    assert payload["reads"]["catalog"] == "ok" and payload["reads"]["quota"] == "ok"


@pytest.mark.parametrize("failing", ["catalog", "accounts", "quota"])
def test_status_payload_classifies_each_fanned_out_facet_independently(
    monkeypatch, tmp_path, failing
):
    """ORDER-INDEPENDENCE. The facets are read concurrently, so a sibling's refusal
    must never downgrade a facet whose own read landed — and the verdict must not
    depend on which `.result()` the code happened to touch first. Consuming them
    in sequence used to report `accounts` as unread whenever `catalog` raised."""
    payload = _reads_probe(monkeypatch, tmp_path, "running", failing_facet=failing)

    expected = {"catalog": "ok", "accounts": "ok", "quota": "ok"}
    expected[failing] = "failed"
    assert payload["reads"] == expected
    # The refusal is still disclosed on the daemon, and the surviving facets keep
    # their payload instead of blanking the whole panel.
    assert payload["daemon"]["state"] == "unreachable"
    assert "daemon_unreachable" in payload["daemon"]["last_error"]
    if failing != "catalog":
        assert [h["id"] for h in payload["harnesses"]] == ["codex"]


def test_status_payload_reads_block_matches_the_declared_gateway_contract(monkeypatch, tmp_path):
    """PRODUCER pin: the wire always carries the full `reads` block with the exact
    keys the frozen gateway contract declares, so a consumer may key on it without
    defensive guessing."""
    from typing import get_type_hints

    from ouroboros.gateway.contracts import ClaudexorStatusReads

    payload = _reads_probe(monkeypatch, tmp_path, "running")

    assert set(payload["reads"]) == set(get_type_hints(ClaudexorStatusReads))
    assert set(payload["reads"].values()) <= {"ok", "not_read", "failed"}


def test_wake_endpoint_starts_the_daemon_and_returns_the_fresh_reading(monkeypatch, tmp_path):
    """The owner-initiated start behind the panel's Refresh button.

    The status GET is side-effect-free by contract, which leaves Refresh unable
    to do anything about a sleeping daemon — an owner who just wants to SEE
    their accounts had to start a login job or a delegated run. This endpoint is
    that missing action: it ensures the daemon, then answers with the reading it
    just made possible.
    """
    import asyncio

    from ouroboros.gateway import claudexor_accounts as accounts

    started = {"n": 0}

    class FakeGateway:
        def close(self):
            pass

    def fake_ensure():
        started["n"] += 1
        return FakeGateway()

    order = []

    def fake_ensure_ordered():
        order.append("ensure")
        return fake_ensure()

    def fake_status(include_models):
        order.append("read")
        return {"daemon": {"state": "running"},
                "reads": {"catalog": "ok", "accounts": "ok", "quota": "ok"}}

    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", fake_ensure_ordered)
    monkeypatch.setattr(accounts, "_status_payload", fake_status)

    response = asyncio.run(accounts.api_claudexor_wake(object()))

    # ORDER is the whole promise of this endpoint: a read taken BEFORE the
    # daemon exists answers with the same nothing Refresh already had, which is
    # the state the owner pressed the button to leave. The docstring said
    # "ensures the daemon, then answers with the reading it just made possible"
    # while a mocked constant payload made the sequence unobservable.
    assert order == ["ensure", "read"], f"the wake read the status before starting anything: {order}"


    assert started["n"] == 1, "the wake must actually ensure the daemon"
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["daemon"]["state"] == "running"
    assert body["reads"]["accounts"] == "ok", "the answer is the POST-wake reading"


def test_wake_endpoint_discloses_a_typed_refusal_instead_of_a_generic_error(monkeypatch, tmp_path):
    """A cold machine can refuse for reasons the owner can act on (no binary, a
    foreign daemon home). That reason must reach the panel as a typed 503 — the
    button says why it could not start, rather than returning silently to idle."""
    import asyncio

    from ouroboros.gateway import claudexor_accounts as accounts
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def refuse():
        raise ClaudexorUnavailable("claudexord_not_installed", "no managed binary")

    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", refuse)

    response = asyncio.run(accounts.api_claudexor_wake(object()))

    assert response.status_code == 503
    assert "claudexord_not_installed" in json.loads(response.body)["error"]


# ---------------------------------------------------------------------------
# The proxy count is a claim, and claims drift.
# ---------------------------------------------------------------------------


def test_the_proxy_count_in_the_docs_matches_the_handlers_that_exist(tmp_path):
    """The module said "three THIN proxies" while a handler inside it introduced
    itself as "A FOURTH thin proxy" — a file contradicting itself in the only two
    places a reader looks first. A hand-counted number in prose cannot be trusted
    to be re-counted when the fifth one lands, so it is asserted instead.

    ``docs/ARCHITECTURE.md`` carries the same count in its gateway map, and it is
    checked against the ROUTES rather than the handlers: a proxy the map never
    names is a proxy nobody discovers from the architecture doc.
    """
    import inspect
    import re

    from ouroboros.gateway import claudexor_accounts as accounts

    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    handlers = sorted(
        name for name, obj in vars(accounts).items()
        if name.startswith("api_claudexor_") and inspect.iscoroutinefunction(obj)
    )
    expected = words[len(handlers)]
    docstring = (accounts.__doc__ or "").lower()
    assert f"{expected} thin proxies" in docstring, (
        f"{len(handlers)} handlers ({', '.join(handlers)}) but the module docstring "
        f"does not say \"{expected} THIN proxies\""
    )

    arch = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "ARCHITECTURE.md") \
        .read_text(encoding="utf-8")
    line = next(ln for ln in arch.splitlines() if "claudexor_accounts.py" in ln)
    assert f"{expected} thin proxies" in line.lower(), (
        f"the gateway map still counts a different number of claudexor proxies: {line.strip()[:160]}"
    )

    # Every REGISTERED path is named in that map entry, so a new proxy cannot
    # land undocumented behind an updated count.
    from ouroboros.gateway.router import collect_routes

    paths = {
        route.path for route in collect_routes(data_dir=tmp_path)
        if getattr(route, "path", "").startswith("/api/claudexor/")
    }
    assert paths, "no /api/claudexor/ routes are registered"
    for path in sorted(paths):
        # The map spells path params by name, not by their brace form for the
        # two-segment removal route; compare on the stable prefix.
        prefix = re.split(r"\{", path)[0].rstrip("/")
        assert prefix in line, f"{path} is registered but the gateway map never names it"


# ---------------------------------------------------------------------------
# Rotation reconcile (B3, owner decision 5=A literal): GET -> conditional POST
# on EVERY ensure — no read-path TTL; the non-blocking lock only dedups
# concurrent ensures. Persisted policies are never overwritten; A6+ engines
# that own kind-aware "auto" defaults are skipped; ANY failure (the daemon's
# startup "recovery only" refusal included) simply retries on the next ensure;
# an actual patch leaves a durable receipt under state/.
# ---------------------------------------------------------------------------


class _ReconcileGateway:
    """Offline gateway double for reconcile_rotation: records reads/patches."""

    def __init__(self, *, engine_version="3.5.0", snapshot=None,
                 harnesses=("codex", "claude"), get_error=None):
        self.engine_version = engine_version
        self._snapshot = {"harnesses": dict(snapshot or {})}
        self._harnesses = list(harnesses)
        self._get_error = get_error
        self.get_calls = 0
        self.patches = []

    def get_settings(self):
        self.get_calls += 1
        if self._get_error is not None:
            raise self._get_error
        return self._snapshot

    def agent_capabilities(self):
        return {"harnesses": [{"id": hid} for hid in self._harnesses]}

    def patch_settings(self, request):
        self.patches.append(request)
        return {}


def _rotation_receipt_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / "state" / "claudexor_rotation_provisioning.json"


def test_reconcile_patches_only_harnesses_without_a_persisted_limit_action(
        monkeypatch, tmp_path):
    """Owner decision 3=A: an explicitly persisted fail/ask/rotate is the
    owner's (or engine's) word — reconcile defaults ONLY the absent ones."""
    data_dir = tmp_path / "data"
    _point_owned_home(monkeypatch, data_dir / "claudexor", data_dir)

    gateway = _ReconcileGateway(
        snapshot={
            "codex": {"profileLimitAction": "fail"},     # explicit: untouched
            "claude": {"profileLimitAction": "rotate"},  # already on: untouched
        },
        harnesses=("codex", "claude", "cursor"),         # cursor: no policy row
    )
    manager = owned.OwnedClaudexorDaemon()
    manager.reconcile_rotation(gateway)

    assert gateway.patches == [
        {"harnesses": {"cursor": {"profileLimitAction": "rotate"}}}
    ]
    # The durable receipt names the daemon identity and exactly what changed.
    receipt = json.loads(_rotation_receipt_path(data_dir).read_text(encoding="utf-8"))
    assert receipt["patched_harnesses"] == ["cursor"]
    assert receipt["limit_action"] == "rotate"
    assert receipt["daemon_config_dir"] == str(data_dir / "claudexor")
    assert receipt["engine_version"] == "3.5.0"
    assert receipt["ts"]


def test_reconcile_skips_the_post_when_nothing_is_missing(monkeypatch, tmp_path):
    """Idempotence: a fully policied daemon gets a GET and NOTHING else —
    no settings POST, no receipt claiming a change that never happened."""
    data_dir = tmp_path / "data"
    _point_owned_home(monkeypatch, data_dir / "claudexor", data_dir)

    gateway = _ReconcileGateway(snapshot={
        "codex": {"profileLimitAction": "rotate"},
        "claude": {"profileLimitAction": "fail"},
    })
    manager = owned.OwnedClaudexorDaemon()
    manager.reconcile_rotation(gateway)

    assert gateway.get_calls == 1
    assert gateway.patches == []
    assert not _rotation_receipt_path(data_dir).exists()


def test_reconcile_never_blanket_posts_on_settings_shape_drift(monkeypatch, tmp_path):
    """Review fix 10: a snapshot whose `harnesses` is missing or not a dict is
    UNKNOWN state, not "nothing persisted" — reconcile must return early instead
    of blanket-POSTing rotate over judgments it simply failed to read."""
    data_dir = tmp_path / "data"
    _point_owned_home(monkeypatch, data_dir / "claudexor", data_dir)

    for drifted in ({}, {"harnesses": None}, {"harnesses": []}, {"harnesses": "x"}, "not-a-dict"):
        gateway = _ReconcileGateway(harnesses=("codex", "claude"))
        gateway._snapshot = drifted
        manager = owned.OwnedClaudexorDaemon()
        manager.reconcile_rotation(gateway)
        assert gateway.get_calls == 1, drifted
        assert gateway.patches == [], drifted
    assert not _rotation_receipt_path(data_dir).exists()


def test_every_ensure_reads_and_posts_only_when_something_is_missing(monkeypatch, tmp_path):
    """Owner decision 5=A, literal: EVERY ensure does the GET and computes the
    missing set — the second ensure reads again and still POSTs nothing when
    nothing is missing; a harness discovered later is patched on the very next
    ensure (no TTL window to wait out, no process-lifetime boolean)."""
    data_dir = tmp_path / "data"
    _point_owned_home(monkeypatch, data_dir / "claudexor", data_dir)

    manager = owned.OwnedClaudexorDaemon()
    first = _ReconcileGateway(harnesses=("codex",))
    manager.reconcile_rotation(first)
    assert first.patches == [
        {"harnesses": {"codex": {"profileLimitAction": "rotate"}}}
    ]

    # The very next ensure DOES read — and POSTs nothing when nothing is missing.
    second = _ReconcileGateway(
        snapshot={"codex": {"profileLimitAction": "rotate"}}, harnesses=("codex",))
    manager.reconcile_rotation(second)
    assert second.get_calls == 1
    assert second.patches == []

    # A harness discovered later is patched IMMEDIATELY (and only it — codex
    # now shows a persisted rotate); the receipt reflects the real POST.
    third = _ReconcileGateway(
        snapshot={"codex": {"profileLimitAction": "rotate"}},
        harnesses=("codex", "grok"),
    )
    manager.reconcile_rotation(third)
    assert third.get_calls == 1
    assert third.patches == [
        {"harnesses": {"grok": {"profileLimitAction": "rotate"}}}
    ]


def test_concurrent_ensures_are_single_flighted(monkeypatch, tmp_path):
    """The non-blocking lock's ONLY job: a reconcile already in flight covers a
    concurrent ensure — the overlapping caller neither reads nor patches. The
    lock never gates a later, non-overlapping ensure."""
    data_dir = tmp_path / "data"
    _point_owned_home(monkeypatch, data_dir / "claudexor", data_dir)

    manager = owned.OwnedClaudexorDaemon()
    overlapped = _ReconcileGateway(harnesses=("codex",))
    assert manager._rotation_lock.acquire(blocking=False)  # a reconcile is "in flight"
    try:
        manager.reconcile_rotation(overlapped)
        assert overlapped.get_calls == 0 and overlapped.patches == []
    finally:
        manager._rotation_lock.release()
    after = _ReconcileGateway(harnesses=("codex",))
    manager.reconcile_rotation(after)  # lock free again: the ensure reconciles
    assert after.get_calls == 1


def test_any_reconcile_failure_retries_on_the_very_next_ensure(monkeypatch, tmp_path):
    """The exact race that silenced the old spawn-only patch: the daemon's
    startup window answers 'serving recovery only'. With no read-path TTL the
    special case collapses — that refusal, and every ORDINARY failure too,
    simply retries on the next ensure."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    data_dir = tmp_path / "data"
    _point_owned_home(monkeypatch, data_dir / "claudexor", data_dir)

    manager = owned.OwnedClaudexorDaemon()
    refusing = _ReconcileGateway(get_error=ClaudexorUnavailable(
        "daemon_recovery_only",
        "daemon is serving recovery only; product routes are closed",
        status_code=503,
    ))
    manager.reconcile_rotation(refusing)
    assert refusing.get_calls == 1

    healed = _ReconcileGateway(harnesses=("codex",))
    manager.reconcile_rotation(healed)   # immediately after: retried, no wait
    assert healed.get_calls == 1
    assert healed.patches == [
        {"harnesses": {"codex": {"profileLimitAction": "rotate"}}}
    ]

    # An ORDINARY failure retries on the next ensure exactly the same way.
    fresh = owned.OwnedClaudexorDaemon()
    broken = _ReconcileGateway(get_error=ClaudexorUnavailable(
        "http_500", "settings route exploded", status_code=500))
    fresh.reconcile_rotation(broken)
    after = _ReconcileGateway(harnesses=("codex",))
    fresh.reconcile_rotation(after)
    assert after.get_calls == 1
    assert after.patches == [
        {"harnesses": {"codex": {"profileLimitAction": "rotate"}}}
    ]


def test_reconcile_never_patches_a_home_ownership_rejects(monkeypatch, tmp_path):
    """Never-adopt extends to settings writes: a home whose marker names a
    different data plane is not ours to reconfigure."""
    data_dir = tmp_path / "data"
    _point_owned_home(monkeypatch, data_dir / "claudexor", data_dir)
    monkeypatch.setattr(owned, "verify_owned_home",
                        lambda: "ownership marker names a different data plane")

    gateway = _ReconcileGateway()
    owned.OwnedClaudexorDaemon().reconcile_rotation(gateway)

    assert gateway.get_calls == 0
    assert gateway.patches == []
    assert not _rotation_receipt_path(data_dir).exists()


def test_reconcile_skips_engines_that_own_auto_semantics(monkeypatch, tmp_path):
    """An A6+ engine defaults limit actions itself, kind-aware (subscription ->
    rotate, metered API key -> fail). A blanket 'rotate' from this side would
    overwrite that judgment, so the reconcile stands down entirely."""
    data_dir = tmp_path / "data"
    _point_owned_home(monkeypatch, data_dir / "claudexor", data_dir)

    gateway = _ReconcileGateway(engine_version="3.6.0")
    owned.OwnedClaudexorDaemon().reconcile_rotation(gateway)

    assert gateway.get_calls == 0
    assert gateway.patches == []


def test_ensure_owned_gateway_reconciles_rotation_on_every_ensure(monkeypatch):
    """The ONE funnel: spawn AND attach consumers all pass through
    ensure_owned_gateway, so the reconcile riding it covers both — including
    the attach paths the old spawn-only patch never reached."""
    from ouroboros.gateways import claudexor as gateway_mod

    endpoint = gateway_mod.DaemonEndpoint(host="127.0.0.1", port=45699, token="tok")

    class _Manager:
        def __init__(self):
            self.reconciled = []

        def ensure_running(self):
            return endpoint

        def reconcile_rotation(self, gateway):
            self.reconciled.append(gateway)

    manager = _Manager()
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: manager)

    class _Gateway:
        def __init__(self, ep):
            assert ep is endpoint
            self.handshakes = 0

        def handshake(self, **_kw):
            self.handshakes += 1
            return {"compatible": True}

        def close(self):
            raise AssertionError("a healthy ensure must hand the gateway to the caller open")

    monkeypatch.setattr(gateway_mod, "ClaudexorGateway", _Gateway)

    gateway = owned.ensure_owned_gateway()
    assert gateway.handshakes == 1
    # Reconcile ran, ONCE, against the very gateway the caller receives —
    # and only AFTER the authenticated handshake (identity before writes).
    assert manager.reconciled == [gateway]


def test_reconcile_is_best_effort_and_never_breaks_the_ensure(monkeypatch, tmp_path):
    """A reconcile hiccup must never eat the delegation/login that ensured the
    daemon: patch failures are logged and swallowed, not raised."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    data_dir = tmp_path / "data"
    _point_owned_home(monkeypatch, data_dir / "claudexor", data_dir)

    class _PatchExplodes(_ReconcileGateway):
        def patch_settings(self, request):
            raise ClaudexorUnavailable("http_500", "patch exploded", status_code=500)

    gateway = _PatchExplodes(harnesses=("codex",))
    owned.OwnedClaudexorDaemon().reconcile_rotation(gateway)  # must not raise
    assert not _rotation_receipt_path(data_dir).exists(), \
        "no receipt may claim a patch that never landed"


def test_receipt_write_failure_warns_loudly_and_never_breaks_the_reconcile(
        monkeypatch, tmp_path, caplog):
    """Post-merge follow-up (sol finding 4): a failed receipt write is no longer a
    bare swallowed warning — it names the path and the error. Best-effort stands:
    the reconcile (and thus the ensure) still succeeds, and the patch is NOT
    treated as receipted — the next ensure's GET sees the values present and
    correctly skips, so the only residual is the missing receipt itself."""
    import logging

    import ouroboros.utils as utils

    data_dir = tmp_path / "data"
    _point_owned_home(monkeypatch, data_dir / "claudexor", data_dir)

    def _no_space(path, text, *args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(utils, "write_text_atomic", _no_space)
    gateway = _ReconcileGateway(harnesses=("codex",))
    with caplog.at_level(logging.WARNING, logger="ouroboros.claudexor_daemon"):
        owned.OwnedClaudexorDaemon().reconcile_rotation(gateway)  # must not raise
    assert gateway.patches == [
        {"harnesses": {"codex": {"profileLimitAction": "rotate"}}}
    ], "the policy POST itself landed; only the receipt is missing"
    assert not _rotation_receipt_path(data_dir).exists()
    warned = [r for r in caplog.records if r.levelno == logging.WARNING
              and "receipt write failed" in r.getMessage()]
    assert warned, "the failed receipt write must warn loudly"
    message = warned[0].getMessage()
    assert "claudexor_rotation_provisioning.json" in message, "the warning names the path"
    assert "No space left on device" in message, "the warning names the error"


def test_handshake_records_the_engine_build_sha_beside_its_version():
    """The auto-install trigger compares the LIVE engine's build sha with the
    reviewed pin's, so the handshake must keep that fact, not just the version.
    An engine that reports no sha leaves it empty rather than guessed."""
    import httpx

    from ouroboros.gateways import claudexor as cx

    def _handshake(engine: dict) -> cx.ClaudexorGateway:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "protocolMajor": cx.CLAUDEXOR_PROTOCOL_MAJOR,
                "compatible": True,
                "engine": engine,
            })

        gateway = cx.ClaudexorGateway(cx.DaemonEndpoint("127.0.0.1", 1, "secret-token"))
        gateway._client = httpx.Client(
            base_url="http://127.0.0.1:1",
            transport=httpx.MockTransport(handler),
            headers=dict(gateway._client.headers),
        )
        with gateway:
            gateway.handshake()
            return gateway

    stamped = _handshake({"version": cx.CLAUDEXOR_MIN_VERSION, "sha": "a" * 40})
    assert stamped.engine_version == cx.CLAUDEXOR_MIN_VERSION
    assert stamped.engine_build_sha == "a" * 40

    unstamped = _handshake({"version": cx.CLAUDEXOR_MIN_VERSION})
    assert unstamped.engine_version == cx.CLAUDEXOR_MIN_VERSION
    assert unstamped.engine_build_sha == ""

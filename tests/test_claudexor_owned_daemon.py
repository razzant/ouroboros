"""Owned Claudexor daemon (D30): isolation root, discovery cutover, thin proxies.

Everything here is offline: no daemon is spawned, no network is touched. The
live login flow is the daemon's own product surface and is exercised by the
phase acceptance run, not by unit tests.
"""
import json
import pathlib

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
    command = owned.attach_login_command("job-123")
    assert command.startswith(f"CLAUDEXOR_CONFIG_DIR={owned.owned_config_dir()} ")
    assert command.endswith("claudexor setup attach job-123")


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
            return {"ok": True}

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
    affordance. Every other refusal stays an ordinary 503 — and a 404 on the
    POLL keeps its ordinary meaning (no capability spin)."""
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

    # The SAME engine 404 on a GET poll is an ordinary lane refusal (503):
    # input_not_supported is typed only where the input capability was asked.
    poll = Request({
        "type": "http", "method": "GET", "path": "/api/claudexor/login/j1",
        "headers": [], "query_string": b"", "path_params": {"job_id": "j1"},
    })
    polled = asyncio.run(api_claudexor_login_job(poll))
    assert polled.status_code == 503

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
    rotations = []
    monkeypatch.setattr(manager, "_enable_rotation", rotations.append)

    assert manager.ensure_running() is endpoint
    assert rotations == [endpoint]
    assert manager._proc is None
    assert manager.stop() is False


@pytest.mark.serial
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


@pytest.mark.serial
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
    monkeypatch.setattr(daemon, "_enable_rotation", lambda _endpoint: None)

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

"""Which harnesses can log in, what a manifest vouches for, and how an account is removed.

Split verbatim out of ``tests/test_claudexor_owned_daemon.py`` by theme. This module
owns the manifest auth block that decides whether a harness has a login concept at
all, the read-failure that must never be reported as an empty filter, the status
payload's api-key-only filtering, the vouched login that survives an unreadable
manifest, and the account-removal contract with the engine.

Everything here is offline: no daemon is spawned, no network is touched.
"""

import json


from ouroboros import claudexor_daemon as owned


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


def test_login_create_passes_the_daemon_400_verdict_through(monkeypatch):
    """A create-time daemon 400 is a typed VERDICT about the requested login
    shape (e.g. a harness with no default credential store refusing a default
    login and telling the owner to sign in from a named account), not daemon
    unavailability. It must reach the browser with its original status, the
    stable code and the engine's own sentence VERBATIM, in the frozen
    ``ClaudexorLoginJobProblem`` envelope — the card keys its
    name-the-account face on the structural pair (create-time, 400), which a
    blanket 503 collapse made unreachable. Anything the daemon did not answer
    with a 400 stays the proxy's honest 503."""
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.claudexor_accounts import api_claudexor_login
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    engine_said = ('harness "zephyr" has no default credential store: sign in '
                   'from a named account (add one first, then start the login from it)')
    refusal = {"exc": ClaudexorUnavailable("http_400", engine_said, status_code=400),
               "stage": "create"}

    class _Gateway:
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

    # No 400 verdict — an unreachable daemon (status 0) and a daemon 5xx — is
    # never promoted to one: both stay the proxy's honest 503.
    for exc in (ClaudexorUnavailable("daemon_unreachable", "connect refused"),
                ClaudexorUnavailable("http_500", "boom", status_code=500)):
        refusal["exc"] = exc
        answer = asyncio.run(_call())
        assert answer.status_code == 503
        assert exc.code.encode() in answer.body

    # A 400 from the HANDSHAKE stage is engine/protocol trouble, not a verdict
    # about the requested login shape: the pass-through is scoped to the job
    # CREATE and everything earlier stays the proxy's honest 503.
    refusal["exc"] = ClaudexorUnavailable("http_400", "protocol mismatch", status_code=400)
    refusal["stage"] = "handshake"
    answer = asyncio.run(_call())
    assert answer.status_code == 503
    assert b"http_400" in answer.body


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

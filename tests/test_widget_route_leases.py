"""The route lease: a framed widget's own GET/HEAD routes at /api/widget-resources/, through the real gate.

The skill is loaded in process and its routes run under the real extension dispatcher. Open streams are
driven at the ASGI level so that revocation, logout, a password change, expiry and skill unload can be
observed ending a response that would otherwise never finish.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import extension_loader as loader
from ouroboros import extension_registry_state as registry
from ouroboros import server_auth
from ouroboros.gateway.extensions import api_extension_dispatch
from ouroboros.gateway.widget_assets import RESOURCE_PREFIX, api_widget_asset_lease, api_widget_bridge_script
from ouroboros.gateway.widgets import api_widgets
from ouroboros.live_conversation import feature_configuration
from ouroboros.skill_loader import (SkillReviewState, compute_content_hash, find_skill, requested_skill_permissions,
                                    save_review_state, save_skill_grants)
from tests._shared import clean_extension_runtime_state

NAME = "route_media"
CLIP = bytes(range(256)) * 64

PLUGIN = '''
import asyncio, pathlib
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
HERE = pathlib.Path(__file__).parent
STATE = {"live_chunks": 0, "live_open": 0, "acted": 0}

async def echo(request):
    return {"path": request.url.path, "query": request.url.query, "cookie": request.headers.get("cookie"),
            "authorization": request.headers.get("authorization"), "method": request.method}

async def clip(_request):
    return FileResponse(HERE / "clip.bin", media_type="application/octet-stream")

async def live(_request):
    async def chunks():
        STATE["live_open"] += 1
        try:
            while True:
                STATE["live_chunks"] += 1
                yield b"x" * 64
                await asyncio.sleep(0.01)
        finally:
            STATE["live_open"] -= 1
    return StreamingResponse(chunks(), media_type="application/octet-stream")

async def state(_request):
    return dict(STATE)

async def act(_request):
    STATE["acted"] += 1
    return {"ok": True}

def go(target):
    async def handler(_request):
        return RedirectResponse(target, status_code=307)
    return handler

def register(api):
    api.register_route("echo", echo)
    api.register_route("media/clip.bin", clip)
    api.register_route("live", live)
    api.register_route("state", state)
    api.register_route("act", act, methods=("GET", "POST"))
    api.register_route("go-own", go("/api/extensions/route_media/echo?x=1"))
    api.register_route("go-relative", go("echo?y=2"))
    api.register_route("go-absolute", go("http://testserver/api/extensions/route_media/media/clip.bin"))
    api.register_route("go-other", go("/api/extensions/other_skill/echo"))
    api.register_route("go-core", go("/api/state"))
    api.register_route("go-dots", go("/api/extensions/route_media/%2e%2e/other_skill/echo"))
    api.register_route("go-foreign", go("http://foreign.example/landing"))
    api.register_route("manifest-shadow", echo)
    api.register_ui_tab("classic", "Classic", render={"kind": "module", "entry": "widget.js", "start": "auto"})
    api.register_ui_tab("page", "Page", render={"kind": "iframe", "route": "echo", "start": "auto"})
    api.register_ui_tab("esm", "ESM", render={"kind": "module", "entry_type": "module", "entry": "main.js", "start": "auto"})
'''


def _install(root):
    skill = root / "skills" / "external" / NAME
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"""---
name: {NAME}
description: Route lease fixture.
version: 0.1.0
type: extension
entry: plugin.py
permissions: [widget, route]
---
Route lease fixture.
""", encoding="utf-8", newline="\n")
    (skill / "plugin.py").write_text(PLUGIN, encoding="utf-8", newline="\n")
    (skill / "widget.js").write_text("document.getElementById('root').textContent = 'classic';", encoding="utf-8")
    (skill / "main.js").write_text("document.getElementById('root').textContent = 'esm';", encoding="utf-8")
    (skill / "clip.bin").write_bytes(CLIP)
    save_review_state(root, NAME, SkillReviewState(status="pass", content_hash=compute_content_hash(skill, manifest_entry="plugin.py")))
    from ouroboros.skill_loader import save_enabled
    save_enabled(root, NAME, True)
    return skill


@pytest.fixture
def env(tmp_path, monkeypatch):
    clean_extension_runtime_state()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(tmp_path / "repo-skills"))
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "fixture-password")
    _install(tmp_path)
    assert loader.load_extension(find_skill(tmp_path, NAME), lambda: {}, drive_root=tmp_path) is None
    app = Starlette(routes=[
        Route("/api/widgets", api_widgets),
        Route("/api/widgets/asset-leases", api_widget_asset_lease, methods=["POST"]),
        Route("/api/widgets/asset-leases/{op}", api_widget_asset_lease, methods=["POST"]),
        Route("/api/widget-bridge.js", api_widget_bridge_script),
        Route("/api/extensions/{skill}/{rest:path}", api_extension_dispatch,
              methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"]),
    ])
    app.state.drive_root = tmp_path
    app.state.repo_dir = __import__("pathlib").Path(__file__).resolve().parents[1]
    gate = server_auth.NetworkAuthGate(app)
    owner = TestClient(gate, client=("192.0.2.8", 12345))
    assert owner.post("/auth/login", json={"password": "fixture-password"}).status_code == 200
    # Another machine's browser without the session: it holds only what the frame could see.
    stranger = TestClient(gate, client=("192.0.2.9", 23456))
    # A headerless loopback client (CLI, sidecar) observes the fixture's counters through any session change.
    native = TestClient(gate, client=("127.0.0.1", 34567))
    try:
        yield SimpleNamespace(owner=owner, stranger=stranger, native=native, gate=gate, root=tmp_path)
    finally:
        owner.close()
        stranger.close()
        native.close()
        clean_extension_runtime_state()


def _lease(client, tab, scope="routes", expect=200):
    row = next(t for t in client.get("/api/widgets").json()["ui_tabs"] if t["tab_id"] == tab)
    body = {k: row[k] for k in ("skill", "tab_id", "generation")}
    if scope is not None:
        body["scope"] = scope
    response = client.post("/api/widgets/asset-leases", json=body)
    assert response.status_code == expect, response.text
    return response.json()


@pytest.mark.parametrize("tab", ["classic", "page"])
def test_a_route_lease_runs_only_its_own_get_routes_without_any_session(env, tab):
    lease = _lease(env.owner, tab)
    assert lease["base_url"].startswith(RESOURCE_PREFIX) and lease["expires_in"] == 120
    response = env.stranger.get(lease["base_url"] + "echo?q=1", headers={"cookie": "ouroboros_auth=forged", "authorization": "Bearer x"})
    assert response.status_code == 200, response.text
    assert response.json() == {"path": f"/api/extensions/{NAME}/echo", "query": "q=1", "cookie": None,
                               "authorization": None, "method": "GET"}
    assert response.headers["cache-control"] == "no-store" and response.headers["referrer-policy"] == "no-referrer"
    # The owner's own cookie does not travel either.
    assert env.owner.get(lease["base_url"] + "echo").json()["cookie"] is None
    head = env.stranger.head(lease["base_url"] + "media/clip.bin")
    assert head.status_code == 200 and head.headers["content-length"] == str(len(CLIP)) and head.content == b""
    # Without the lease the same stranger reaches nothing: the network password still applies.
    assert env.stranger.get(f"/api/extensions/{NAME}/echo").status_code == 401


def test_range_requests_seek_through_the_lease(env):
    base = _lease(env.owner, "classic")["base_url"]
    part = env.stranger.get(base + "media/clip.bin", headers={"range": "bytes=1000-1009"})
    assert part.status_code == 206 and part.content == CLIP[1000:1010]
    assert part.headers["content-range"] == f"bytes 1000-1009/{len(CLIP)}"


def test_lease_kinds_and_scopes_never_cross(env):
    routes = _lease(env.owner, "classic")
    files = _lease(env.owner, "esm", scope=None)
    assert files["base_url"].startswith("/api/widget-assets/")
    # A file token never reads routes, a route token never reads files.
    assert env.stranger.get(f"{RESOURCE_PREFIX}{files['token']}/echo").status_code == 404
    assert env.stranger.get(f"/api/widget-assets/{routes['token']}/main.js").status_code == 404
    # An ESM tab mints no route lease; an unknown scope or extra field is refused.
    _lease(env.owner, "esm", expect=409)
    _lease(env.owner, "classic", scope="everything", expect=400)
    assert env.stranger.get(f"{RESOURCE_PREFIX}{'x' * 43}/echo").status_code == 404
    assert env.stranger.get(f"{RESOURCE_PREFIX}short/echo").status_code == 400
    base = routes["base_url"]
    for path in ("manifest", "module/widget.js", "ui/rev/x.png", "settings_section", "a//b", ""):
        assert env.stranger.get(base + path).status_code == 400, path
    # The ASGI server hands over decoded dot segments (clients normalize literal ones first).
    for rest in ("..", "a/../../state", "./echo"):
        messages, run = _asgi_get(env.gate, base + rest)
        asyncio.run(run())
        assert messages[0]["status"] == 400, rest
    # GET/HEAD only, even for a route that also takes POST, and no handler runs.
    assert env.stranger.post(base + "act").status_code == 405
    assert _state(env)["acted"] == 0
    # The route lease is not an API credential: no cookie fallback, no control authority.
    assert env.stranger.post("/api/widgets/asset-leases/renew", json={"token": routes["token"]}).status_code == 401
    assert env.stranger.get("/api/widgets", headers={"authorization": f"Bearer {routes['token']}"}).status_code == 401


@pytest.mark.parametrize("route,location", [
    ("go-own", "{base}echo?x=1"),
    ("go-relative", "{base}echo?y=2"),
    ("go-absolute", "{base}media/clip.bin"),
    ("go-foreign", "http://foreign.example/landing"),
])
def test_own_redirects_stay_under_the_lease_and_foreign_ones_carry_nothing(env, route, location):
    base = _lease(env.owner, "classic")["base_url"]
    response = env.stranger.get(base + route, follow_redirects=False)
    assert response.status_code == 307, response.text
    assert response.headers["location"] == location.format(base=base)
    assert response.headers["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize("route", ["go-other", "go-core", "go-dots"])
def test_a_redirect_to_another_skill_or_the_core_is_refused(env, route):
    base = _lease(env.owner, "classic")["base_url"]
    response = env.stranger.get(base + route, follow_redirects=False)
    assert response.status_code == 403 and "location" not in response.headers
    assert "own skill" in response.json()["error"]


def test_disable_and_unload_end_reads_and_renewal(env):
    from ouroboros.skill_loader import save_enabled

    lease = _lease(env.owner, "page")
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 200
    save_enabled(env.root, NAME, False)
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 404
    assert env.owner.post("/api/widgets/asset-leases/renew", json={"token": lease["token"]}).status_code == 409


# --- open responses -------------------------------------------------------------------------------------------


def _asgi_get(gate, path, *, disconnect: threading.Event | None = None):
    """Run one GET through the gate; returns (messages, task-runner) where messages fill as the gate sends."""
    messages: list[dict] = []
    token, _, rest = path[len(RESOURCE_PREFIX):].partition("/")
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": [(b"host", b"testserver")], "client": ("192.0.2.9", 1), "server": ("testserver", 80)}

    async def run():
        sent_request = False

        async def receive():
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {"type": "http.request", "body": b"", "more_body": False}
            while not (disconnect and disconnect.is_set()):
                await asyncio.sleep(0.01)
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        await asyncio.wait_for(gate(scope, receive, send), timeout=10)

    return messages, run


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _state(env):
    return env.native.get(f"/api/extensions/{NAME}/state").json()


@pytest.mark.parametrize("end", ["revoke", "logout", "password", "expiry", "unload"])
def test_ending_the_lease_ends_an_open_live_stream(env, monkeypatch, end):
    lease = _lease(env.owner, "classic")
    messages, run = _asgi_get(env.gate, lease["base_url"] + "live")
    errors: list[BaseException] = []
    runner = threading.Thread(target=lambda: _capture(errors, lambda: asyncio.run(run())), daemon=True)
    runner.start()
    assert _wait_for(lambda: sum(m["type"] == "http.response.body" for m in messages) >= 3), messages[:2]
    assert messages[0]["type"] == "http.response.start" and messages[0]["status"] == 200
    assert _state(env)["live_open"] == 1
    leases = env.gate.widget_assets
    if end == "revoke":
        leases.revoke(leases._leases[lease["token"]].owner, lease["token"])
    elif end == "logout":
        assert env.owner.get("/auth/logout", follow_redirects=False).status_code == 303
    elif end == "password":
        monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "changed-password")
        env.stranger.get("/api/health")  # any request through the gate observes the change
    elif end == "expiry":
        real = time.monotonic
        monkeypatch.setattr("ouroboros.gateway.widget_assets.time",
                            SimpleNamespace(monotonic=lambda: real() + 10_000))
    else:
        assert loader.unload_extension(NAME)
    runner.join(timeout=10)
    assert not runner.is_alive(), "the open stream did not end"
    assert errors == []
    bodies = [m for m in messages if m["type"] == "http.response.body"]
    # A chunked body ends cleanly with its final empty message; nothing follows it.
    assert bodies[-1] == {"type": "http.response.body", "body": b"", "more_body": False}
    if end != "unload":
        assert _wait_for(lambda: _state(env)["live_open"] == 0)
        assert env.stranger.get(lease["base_url"] + "echo").status_code == 404


def test_a_client_that_leaves_ends_the_stream_and_detaches_it(env):
    lease = _lease(env.owner, "classic")
    gone = threading.Event()
    messages, run = _asgi_get(env.gate, lease["base_url"] + "live", disconnect=gone)
    runner = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
    runner.start()
    assert _wait_for(lambda: sum(m["type"] == "http.response.body" for m in messages) >= 2)
    gone.set()
    runner.join(timeout=10)
    assert not runner.is_alive()
    assert _wait_for(lambda: _state(env)["live_open"] == 0)
    assert env.gate.widget_assets._streams == {}
    # The lease itself is still good for the next request.
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 200


def _capture(errors, fn):
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion
        errors.append(exc)


# --- a skill that also offers a call ---------------------------------------------------------------------------


@pytest.fixture
def call_capable(monkeypatch):
    """The same skill also registers a conversation feature, through the real PluginAPI and grants."""
    module = sys.modules[__name__]
    original_install = module._install
    monkeypatch.setattr(module, "PLUGIN", PLUGIN.replace("def register(api):", """async def never_called(host):
    raise AssertionError("no call starts here")

def register(api):
    from ouroboros.contracts.live_conversation import LiveConfig
    api.register_live_conversation("call", never_called, configure=LiveConfig)"""))

    def install(root):
        skill = original_install(root)
        manifest = skill / "SKILL.md"
        manifest.write_text(manifest.read_text(encoding="utf-8").replace(
            "permissions: [widget, route]", 'plugin_api: "2.3"\npermissions: [widget, route, live_conversation]'),
            encoding="utf-8", newline="\n")
        _approve_call_capable(root, skill)
        return skill

    monkeypatch.setattr(module, "_install", install)


def _approve_call_capable(root, skill):
    digest = compute_content_hash(skill, manifest_entry="plugin.py")
    save_review_state(root, NAME, SkillReviewState(status="pass", content_hash=digest))
    permissions = requested_skill_permissions(["widget", "route", "live_conversation"])
    save_skill_grants(root, NAME, [], content_hash=digest, requested_keys=[],
                      requested_permissions=permissions, granted_permissions=permissions)


@contextlib.contextmanager
def _open_media(env):
    """An endless leased response; only the publication's own cancellation can end it here (no disconnect)."""
    lease = _lease(env.owner, "classic")
    messages, run = _asgi_get(env.gate, lease["base_url"] + "live")
    errors: list[BaseException] = []
    runner = threading.Thread(target=lambda: _capture(errors, lambda: asyncio.run(run())), daemon=True)
    runner.start()
    assert _wait_for(lambda: sum(m["type"] == "http.response.body" for m in messages) >= 3), messages[:2]
    assert _state(env)["live_open"] == 1
    try:
        yield SimpleNamespace(runner=runner, messages=messages, bundle=registry._extensions[NAME])
    finally:
        loader.unload_extension(NAME)
        runner.join(timeout=8)
        assert not runner.is_alive() and not errors


def _new_payload(env):
    """A reviewed edit: the next reconcile owes an automatic replacement of the live publication."""
    skill = find_skill(env.root, NAME).skill_dir
    plugin = skill / "plugin.py"
    plugin.write_text(plugin.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8", newline="\n")
    _approve_call_capable(env.root, skill)


def _reconcile(env):
    return loader.reconcile_extension(NAME, env.root, lambda: {}, repo_path="")


def test_leased_media_is_cancelled_by_replacement_it_cannot_hold(call_capable, env):
    assert registry._extensions[NAME].conversation_features == [f"{NAME}:call"]
    with _open_media(env) as media:
        assert media.bundle.active_call_work == 0 and media.bundle.supervised_futures
        assert registry.realtime_load_refusal(NAME) == ""
        _new_payload(env)
        assert _reconcile(env)["action"] == "extension_loaded"
        media.runner.join(timeout=8)
        assert not media.runner.is_alive(), "replacing the publication must end its open media"
        replacement = registry._extensions[NAME]
        assert replacement is not media.bundle and replacement.generation_digest != media.bundle.generation_digest
        assert media.bundle.supervised_futures == [] and replacement.supervised_futures == []
        assert _state(env)["live_open"] == 0


def test_a_real_call_admission_still_refuses_replacement_beside_media(call_capable, env):
    with _open_media(env) as media:
        with feature_configuration(f"{NAME}:call"):
            assert media.bundle.active_call_work == 1
            assert registry.realtime_load_refusal(NAME) == "live_call_active"
            _new_payload(env)
            refused = _reconcile(env)
            assert (refused["reason"], refused["action"]) == ("live_call_active", "extension_reload_pending")
            assert registry._extensions[NAME] is media.bundle
            chunks = len(media.messages)
            assert _wait_for(lambda: len(media.messages) > chunks), "the refused reload must leave media running"
        assert media.bundle.active_call_work == 0 and registry.realtime_load_refusal(NAME) == ""
        assert media.runner.is_alive()
        assert _reconcile(env)["action"] == "extension_loaded"
        media.runner.join(timeout=8)
        assert not media.runner.is_alive() and registry._extensions[NAME] is not media.bundle


def test_unload_cleans_up_media_and_call_work_together(call_capable, env):
    with _open_media(env) as media:
        with feature_configuration(f"{NAME}:call"):
            loader.unload_extension(NAME)
            media.runner.join(timeout=8)
            assert not media.runner.is_alive() and NAME not in registry._extensions
        assert media.bundle.active_call_work == 0 and media.bundle.supervised_futures == []
    assert registry.realtime_load_refusal(NAME) == ""


def test_an_isolated_dependency_route_streams_through_the_lease_and_ends_with_it(tmp_path, monkeypatch):
    """The out-of-process dispatcher (RouteStreamResponse) under a lease: the child streams an endless body
    to the gate, and revoking the lease ends the response and the child through the existing disconnect path."""
    from tests._extension_loader_shared import _add_fake_native_dep, _mark_isolated_deps_installed, _prepare_extension

    clean_extension_runtime_state()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "")
    plugin = (
        "import asyncio\n"
        "from starlette.responses import StreamingResponse\n"
        "async def _live(request):\n"
        "    async def chunks():\n"
        "        while True:\n"
        "            yield b'x' * 64\n"
        "            await asyncio.sleep(0.02)\n"
        "    return StreamingResponse(chunks(), media_type='application/octet-stream')\n"
        "def register(api):\n"
        "    api.register_route('live', _live)\n"
        "    api.register_ui_tab('page', 'Page', render={'kind': 'iframe', 'route': 'live', 'start': 'auto'})\n"
    )
    loaded, skills_root, drive_root = _prepare_extension(tmp_path, "oop_media", plugin, permissions=["route", "widget"],
                                                         extra_frontmatter="dependencies:\n  - dummy_pkg\n")
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(skills_root))
    try:
        assert loader.load_extension(loaded, lambda: {}, drive_root=drive_root) is None
        assert loader.list_routes()["/api/extensions/oop_media/live"]["out_of_process"] is True
        app = Starlette(routes=[
            Route("/api/widgets", api_widgets),
            Route("/api/widgets/asset-leases", api_widget_asset_lease, methods=["POST"]),
            Route("/api/widgets/asset-leases/{op}", api_widget_asset_lease, methods=["POST"]),
            Route("/api/extensions/{skill}/{rest:path}", api_extension_dispatch, methods=["GET", "HEAD", "POST"]),
        ])
        app.state.drive_root = drive_root
        app.state.repo_dir = __import__("pathlib").Path(__file__).resolve().parents[1]
        gate = server_auth.NetworkAuthGate(app)
        with TestClient(gate, client=("127.0.0.1", 45678)) as native:
            lease = _lease(native, "page")
            messages, run = _asgi_get(gate, lease["base_url"] + "live")
            errors: list[BaseException] = []
            runner = threading.Thread(target=lambda: _capture(errors, lambda: asyncio.run(run())), daemon=True)
            runner.start()
            assert _wait_for(lambda: sum(m["type"] == "http.response.body" for m in messages) >= 3, timeout=30), messages[:2]
            assert messages[0]["status"] == 200
            leases = gate.widget_assets
            leases.revoke(leases._leases[lease["token"]].owner, lease["token"])
            runner.join(timeout=20)
            assert not runner.is_alive(), "the child's open stream did not end"
            assert errors == []
            assert [m for m in messages if m["type"] == "http.response.body"][-1]["more_body"] is False
            assert native.get(lease["base_url"] + "live").status_code == 404
    finally:
        clean_extension_runtime_state()


# --- terminal slash, re-establishing a lease, the frame bridge script ---------------------------------------


def test_a_registered_route_with_a_terminal_slash_is_reachable_through_the_lease(env, monkeypatch):
    """A route registered as ``folder/`` answers through the dispatcher; the lease keeps that shape."""
    from ouroboros.extension_registry_state import _routes

    mount = f"/api/extensions/{NAME}/folder/"
    spec = dict(_routes[f"/api/extensions/{NAME}/echo"])
    monkeypatch.setitem(_routes, mount, spec)
    base = _lease(env.owner, "classic")["base_url"]
    assert env.native.get(f"/api/extensions/{NAME}/folder/").status_code == 200
    response = env.stranger.get(base + "folder/")
    assert response.status_code == 200 and response.json()["path"] == mount
    # A redirect to the same shape stays under the lease; the traversal negatives above still hold.
    from ouroboros.gateway.widget_assets import _granted_location
    own = "http://testserver"
    assert _granted_location("/api/extensions/x/folder/", own, own + "/api/extensions/x/go", "/api/extensions/x/",
                             "/api/widget-resources/T/") == "/api/widget-resources/T/folder/"
    assert _granted_location("/api/extensions/x/", own, own + "/api/extensions/x/go", "/api/extensions/x/",
                             "/api/widget-resources/T/") is None


def _reopen(client, lease, tab="classic", **extra):
    row = next(t for t in client.get("/api/widgets").json()["ui_tabs"] if t["tab_id"] == tab)
    body = {**{k: row[k] for k in ("skill", "tab_id", "generation")}, "scope": "routes", "token": lease["token"], **extra}
    return client.post("/api/widgets/asset-leases", json=body)


def test_a_lost_route_lease_is_re_established_under_the_same_url(env):
    """A core restart empties the table: the old URL 404s and renewal 409s until the parent re-establishes it."""
    lease = _lease(env.owner, "page")
    env.gate.widget_assets._leases.clear()  # what a restart leaves
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 404
    assert env.owner.post("/api/widgets/asset-leases/renew", json={"token": lease["token"]}).status_code == 409
    again = _reopen(env.owner, lease, "page", generation="from-before-the-restart")
    assert again.status_code == 200 and again.json()["base_url"] == lease["base_url"]
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 200
    # Re-establishing a live lease of the same mount is a renewal; another tab or owner cannot take it.
    assert _reopen(env.owner, lease, "page").status_code == 200
    assert _reopen(env.owner, lease, "classic").status_code == 409
    with TestClient(env.gate, client=("127.0.0.1", 1)) as local:
        assert _reopen(local, lease, "page").status_code == 409


@pytest.mark.parametrize("end", ["revoke", "logout", "password"])
def test_an_ended_route_lease_is_never_re_established(env, monkeypatch, end):
    lease = _lease(env.owner, "page")
    if end == "revoke":
        assert env.owner.post("/api/widgets/asset-leases/revoke", json={"token": lease["token"]}).status_code == 200
        assert _reopen(env.owner, lease, "page").status_code == 410
        return
    if end == "logout":
        env.owner.get("/auth/logout", follow_redirects=False)
    else:
        monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "changed-password")
    assert env.owner.post("/api/widgets/asset-leases/renew", json={"token": lease["token"]}).status_code == 401
    password = "fixture-password" if end == "logout" else "changed-password"
    assert env.owner.post("/auth/login", json={"password": password}).status_code == 200
    assert _reopen(env.owner, lease, "page").status_code == 410
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 404


def test_re_establishing_is_only_for_well_formed_route_leases(env):
    lease = _lease(env.owner, "classic")
    assert _reopen(env.owner, {"token": "short"}).status_code == 400
    files = _lease(env.owner, "esm", scope=None)
    row = next(t for t in env.owner.get("/api/widgets").json()["ui_tabs"] if t["tab_id"] == "esm")
    body = {k: row[k] for k in ("skill", "tab_id", "generation")}
    assert env.owner.post("/api/widgets/asset-leases", json={**body, "token": files["token"]}).status_code == 400
    assert _reopen(env.owner, lease, "esm").status_code == 409


def test_the_frame_bridge_script_is_public_static_code(env):
    """Route pages load it from their opaque frame, which has no session: no password, no boundary refusal."""
    for client, headers in ((env.stranger, {"origin": "null", "sec-fetch-site": "cross-site", "sec-fetch-dest": "script"}),
                            (env.native, {})):
        response = client.get("/api/widget-bridge.js", headers=headers)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/javascript")
        assert response.headers["access-control-allow-origin"] == "*"
        assert "function ouroWidgetFrameBridge(config)" in response.text
        assert "ouro-widget-hello" in response.text
    assert env.stranger.post("/api/widget-bridge.js", headers={"origin": "null", "sec-fetch-site": "cross-site"}).status_code == 403


# --- Stop wins over recovery; default ports are the same origin ------------------------------------------------


def test_stop_ends_a_token_the_table_no_longer_holds(env):
    """After expiry (or a restart) the live entry is gone: Stop still ends the token, so it is not re-established."""
    from dataclasses import replace

    lease = _lease(env.owner, "page")
    leases = env.gate.widget_assets
    leases._leases[lease["token"]] = replace(leases._leases[lease["token"]], expires=0)
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 404
    assert env.owner.post("/api/widgets/asset-leases/revoke", json={"token": lease["token"]}).status_code == 200
    assert _reopen(env.owner, lease, "page").status_code == 410
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 404


def test_stop_during_an_admitted_re_establishment_has_the_last_word(env, monkeypatch):
    """A restarted core (empty table) admits a recovery, then Stop arrives before the worker mutates the table:
    the late worker finds the token ended and the stopped URL stays unreadable."""
    import threading

    from ouroboros.gateway.widget_assets import WidgetAssetLeases

    lease = _lease(env.owner, "page")
    env.gate.widget_assets = leases = WidgetAssetLeases()
    entered, proceed, finished = threading.Event(), threading.Event(), {}
    real_open = leases.open

    def admitted_open(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(leases, "open", admitted_open)
    worker = threading.Thread(target=lambda: finished.setdefault("reply", _reopen(env.owner, lease, "page")))
    worker.start()
    try:
        assert entered.wait(5)
        assert env.owner.post("/api/widgets/asset-leases/revoke", json={"token": lease["token"]}).status_code == 200
        assert env.stranger.get(lease["base_url"] + "echo").status_code == 404
    finally:
        proceed.set()
        worker.join(5)
    assert finished["reply"].status_code == 410
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 404


def test_recovery_without_stop_and_another_owners_stop_keep_working(env):
    """The preserved paths: an unstopped mount re-establishes after a restart; another owner cannot stop a live lease."""
    lease = _lease(env.owner, "page")
    env.gate.widget_assets._leases.clear()  # what a restart leaves
    assert _reopen(env.owner, lease, "page").status_code == 200
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 200
    with TestClient(env.gate, client=("127.0.0.1", 2)) as other_owner:
        assert other_owner.post("/api/widgets/asset-leases/revoke", json={"token": lease["token"]}).status_code == 200
    assert env.stranger.get(lease["base_url"] + "echo").status_code == 200
    assert _reopen(env.owner, lease, "page").status_code == 200
    # A malformed token is never recorded.
    assert env.owner.post("/api/widgets/asset-leases/revoke", json={"token": "short"}).status_code == 200
    assert "short" not in env.gate.widget_assets._ended


@pytest.mark.parametrize("target,expected", [
    ("http://testserver:80/api/extensions/route_media/echo", "{base}echo"),
    ("HTTP://TESTSERVER/api/extensions/route_media/echo?x=1", "{base}echo?x=1"),
    ("http://testserver:80/api/extensions/other_skill/echo", None),
    ("http://testserver:80/api/state", None),
    ("http://testserver:8080/api/extensions/route_media/echo", "http://testserver:8080/api/extensions/route_media/echo"),
])
def test_an_own_redirect_spelled_with_the_default_port_stays_under_the_lease(env, monkeypatch, target, expected):
    """Through the real gate and dispatcher: an explicit default port is the same origin; other skills and the core
    stay refused; another port is another origin and passes unchanged (without the lease)."""
    from starlette.responses import RedirectResponse
    from ouroboros.extension_registry_state import _routes

    base = _lease(env.owner, "page")["base_url"]
    mount = f"/api/extensions/{NAME}/go-absolute"

    async def redirect(_request):
        return RedirectResponse(target, status_code=307)

    monkeypatch.setitem(_routes, mount, {**_routes[mount], "handler": redirect})
    frame = {"origin": "null", "sec-fetch-site": "cross-site"}
    response = env.stranger.get(base + "go-absolute", headers=frame, follow_redirects=False)
    if expected is None:
        assert response.status_code == 403 and "location" not in response.headers
        return
    assert response.status_code == 307 and response.headers["location"] == expected.format(base=base)
    if expected.startswith("{base}"):
        followed = env.stranger.get(base + "go-absolute", headers=frame, follow_redirects=True)
        assert followed.status_code == 200 and followed.json()["path"] == f"/api/extensions/{NAME}/echo"


def test_https_default_port_is_the_same_origin():
    from ouroboros.gateway.widget_assets import _granted_location

    own, prefix, grant = "https://core.example", "/api/extensions/s/", "/api/widget-resources/T/"
    assert _granted_location("https://core.example:443/api/extensions/s/a", own, own + "/api/extensions/s/go", prefix, grant) == grant + "a"
    assert _granted_location("https://core.example:443/api/state", own, own + "/api/extensions/s/go", prefix, grant) is None
    assert _granted_location("http://core.example/api/extensions/s/a", own, own + "/api/extensions/s/go", prefix, grant) == "http://core.example/api/extensions/s/a"


def test_short_ui_route_stays_behind_the_boundary(tmp_path, monkeypatch):
    """A registered GET `ui/action` must not pass the browser boundary as a passive file.

    The passive exception covers only the static routes the host actually serves — `module/{entry}`
    and `ui/{revision}/{path}` — so a two-segment `ui/action` reaches the extension dispatcher and
    must be refused for a cross-site browser, while native clients and genuine static files keep
    working. Reproduced through the real `collect_routes` router and dispatcher.
    """
    from ouroboros.gateway.router import collect_routes

    clean_extension_runtime_state()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(tmp_path / "repo-skills"))
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "fixture-password")
    import tests.test_widget_route_leases as _mod
    monkeypatch.setattr(_mod, "PLUGIN",
        PLUGIN.replace('api.register_route("act", act, methods=("GET", "POST"))',
                       'api.register_route("act", act, methods=("GET", "POST"))\n'
                       '    api.register_route("ui/action", act, methods=("GET", "POST"))'))
    _install(tmp_path)
    assert loader.load_extension(find_skill(tmp_path, NAME), lambda: {}, drive_root=tmp_path) is None
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    app.state.repo_dir = __import__("pathlib").Path(__file__).resolve().parents[1]
    app.state.events = []
    app.state.broadcast_ws_sync = app.state.events.append
    gate = server_auth.NetworkAuthGate(app)
    stranger = TestClient(gate, client=("192.0.2.9", 23456))
    native = TestClient(gate, client=("127.0.0.1", 34567))
    base = f"/api/extensions/{NAME}/"
    try:
        counter = lambda: native.get(base + "state").json()["acted"]
        # A cross-site browser cannot drive the registered GET handler, under any header shape.
        for headers in ({"sec-fetch-site": "cross-site"},
                        {"origin": "null", "sec-fetch-site": "cross-site"},
                        {"origin": "http://foreign.example", "sec-fetch-site": "cross-site"}):
            before = counter()
            blocked = stranger.get(base + "ui/action", headers=headers)
            assert blocked.status_code == 403 and blocked.json()["code"] == "cross_origin_request", headers
            assert counter() == before
        # A native client (no browser headers) still reaches the dynamic route and its effect runs.
        before = counter()
        assert native.get(base + "ui/action").status_code == 200
        assert counter() == before + 1
        # Genuine passive files are still not refused by the boundary, even with cross-site headers: the
        # reviewed classic module entry loads (loopback, as in the default install the finding concerns).
        asset = native.get(base + "module/widget.js", headers={"origin": "null", "sec-fetch-site": "cross-site"})
        assert asset.status_code == 200 and asset.text == "document.getElementById('root').textContent = 'classic';"
    finally:
        stranger.close()
        native.close()
        clean_extension_runtime_state()

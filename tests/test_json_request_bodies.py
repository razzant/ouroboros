"""JSON request bodies at the real routes behind the real ``NetworkAuthGate``.

``gateway._helpers.request_json_or`` reads a body whatever its ``Content-Type``. A body with no bytes is
the route's documented absent body; a present body that holds no JSON object (form fields, broken or
wrongly encoded text, bare whitespace, ``null``, an array or a scalar) never becomes that default: the
route refuses it in its own published shape. Each case records the effect its route would cause, so a
refusal is proved by an empty record, and the browser boundary and the network password still answer
before any route reads a body.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

import ouroboros.server_auth as server_auth
from ouroboros.gateway import _helpers
from ouroboros.gateway.router import collect_routes
from tests.test_client_contract_bindings import assert_bound

LABELS = [None, "application/json", "text/plain;charset=UTF-8", "application/x-www-form-urlencoded"]
# An unterminated array nested past the parser's recursion limit. pytest names a parameter by its value unless it
# has an id, and exports the running test's name as ``PYTEST_CURRENT_TEST``: Windows refuses a variable over
# 32,767 characters, so this body's own 100,000-character name failed the setup and teardown of every test it keys.
DEEPLY_NESTED = pytest.param(b"[" * 100_000, id="nested-past-the-recursion-limit")
UNREADABLE = [b"enabled=true", b"{", b"{}x", b'{"a": "\xff"}', b" ", b"\r\n", DEEPLY_NESTED,
              b"null", b"[]", b"[1]", b"0", b"false", b'""', b'"text"']


async def _echo(request):
    return JSONResponse({"value": await _helpers.request_json_or(request, "absent")})


@pytest.fixture
def echo():
    with TestClient(Starlette(routes=[Route("/echo", _echo, methods=["POST"])])) as client:
        yield client


@pytest.mark.parametrize("label", LABELS)
def test_an_object_is_read_whatever_its_label(echo, label):
    response = echo.post("/echo", content='{"a": [1, "é"], "b": null}'.encode(),
                         headers={"content-type": label} if label else {})
    assert response.json() == {"value": {"a": [1, "é"], "b": None}}


def test_only_a_body_without_bytes_is_absent(echo):
    def chunks():
        yield b""

    assert echo.post("/echo").json() == echo.post("/echo", content=chunks()).json() == {"value": "absent"}


@pytest.mark.parametrize("body", UNREADABLE)
def test_a_present_body_without_an_object_is_none(echo, body):
    assert echo.post("/echo", content=body, headers={"content-type": "application/json"}).json() == {"value": None}


def test_a_body_that_names_its_own_test_is_short_enough_for_windows():
    """A parameter without an id is named by its value, and pytest exports the running test's name as
    ``PYTEST_CURRENT_TEST``: on Windows a 100,000-byte body failed the setup and teardown of every test it keys, and
    nowhere else. This reads the bodies themselves, so a retry of one test alone or a ``-k`` selection cannot hide it."""
    assert all(len(case) <= 100 if isinstance(case, bytes) else bool(case.id) for case in UNREADABLE)


# --- real routes -----------------------------------------------------------------------------------


def _skill_action(monkeypatch, seen):
    from ouroboros.gateway import extensions

    async def action(_request, skill, verb, body):
        seen.append((verb, skill, body))
        return {"ok": True}

    monkeypatch.setattr(extensions, "_run_owner_skill_action", action)


def _resume(monkeypatch, seen):
    from supervisor import queue

    monkeypatch.setattr(queue, "resume_budget_paused_task", lambda task_id: seen.append(("resume", task_id)) or {"ok": True})


def _mcp(monkeypatch, seen):
    from ouroboros.gateway import mcp

    class Manager:
        def refresh_all(self):
            seen.append(("refresh_all",))
            return {"refreshed": {}}

        def refresh_server(self, server_id):
            seen.append(("refresh", server_id))
            return {"ok": True}

        def test_server(self, server, settings=None):
            seen.append(("test", server.get("url")))
            return {"ok": True}

    monkeypatch.setattr(mcp, "_ensure_configured", lambda: None)
    monkeypatch.setattr(mcp, "get_manager", Manager)
    monkeypatch.setattr(mcp, "load_settings", lambda: {})


def _regenerate(monkeypatch, seen):
    from ouroboros.gateway import ui_i18n

    def update(_root, tag, _change, create=False):
        seen.append(("regenerate", tag))
        return {"revision": 1}

    monkeypatch.setattr(ui_i18n.memory, "update_memory", update)
    monkeypatch.setattr(ui_i18n.memory, "current_language", lambda: "ru")
    monkeypatch.setattr(ui_i18n.memory, "stats", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(ui_i18n.memory, "pending_count", lambda *_args: 0)
    for name in ("_broadcast", "_notify", "_source_hashes"):
        monkeypatch.setattr(ui_i18n, name, lambda *_args: None)


def _clawhub_update(monkeypatch, seen):
    from ouroboros.gateway import marketplace

    class Updated:
        ok = True

    def update(_root, _repo, *, sanitized_name, version, **_kwargs):
        seen.append(("update", sanitized_name, version))
        return Updated()

    async def job(*, runner, **_kwargs):
        return await runner()

    monkeypatch.setattr(marketplace, "update_skill", update)
    monkeypatch.setattr(marketplace, "run_lifecycle_job", job)
    monkeypatch.setattr(marketplace, "_serialize_install_result", lambda result: {"ok": result.ok})


def _source_change(monkeypatch, seen):
    from ouroboros.gateway import skill_sources

    async def settings_write(request, write):
        return write(request)

    def write_sources(_request, _change, audit):
        seen.append(("edit", audit))
        return JSONResponse({"ok": True})

    monkeypatch.setattr(skill_sources, "_settings_write", settings_write)
    monkeypatch.setattr(skill_sources, "_write_sources", write_sources)


def _publish_preflight(monkeypatch, seen):
    from ouroboros.gateway import skill_publish

    class Ready:
        payload, status_code = {"ok": True}, 200

    def build(_root, skill, *, admission_problem=None, destination=None):
        seen.append(("preflight", skill, destination))
        return Ready()

    monkeypatch.setattr(skill_publish, "build_skill_publish_preflight", build)
    monkeypatch.setattr(skill_publish, "_task_admission_problem", lambda _request: None)


def _auto_grant(monkeypatch, seen):
    from ouroboros.gateway import settings

    monkeypatch.setattr(settings, "_owner_update_settings", lambda change: seen.append(("save", change({}))))
    monkeypatch.setattr(settings, "_owner_audit", lambda *_args: None)
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")


class Case(NamedTuple):
    token: str                         # the published route
    path: str
    install: Callable[[Any, list], None]
    body: bytes                        # an object the route acts on
    acted: tuple                       # what that object records
    absent: tuple | None               # what an empty body records; None: the route needs a body


CASES = {
    "resume": Case("POST /api/tasks/{task_id}/resume", "/api/tasks/t1/resume", _resume,
                   b'{"reason": "go on"}', ("resume", "t1"), ("resume", "t1")),
    "skill_delete": Case("POST /api/skills/{skill}/delete", "/api/skills/demo/delete", _skill_action,
                         b'{"expected_content_hash": "abc"}', ("delete", "demo", {"expected_content_hash": "abc"}),
                         ("delete", "demo", {})),
    "skill_grants": Case("POST /api/skills/{skill}/grants", "/api/skills/demo/grants", _skill_action,
                         b'{"enable_after_grant": true}', ("grant", "demo", {"enable_after_grant": True}),
                         ("grant", "demo", {})),
    "attest_review": Case("POST /api/owner/skills/{skill}/attest-review", "/api/owner/skills/demo/attest-review",
                          _skill_action, b'{"expected_content_hash": "abc"}',
                          ("attest", "demo", {"expected_content_hash": "abc"}), ("attest", "demo", {})),
    "mcp_refresh": Case("POST /api/mcp/refresh", "/api/mcp/refresh", _mcp,
                        b'{"server_id": "alpha"}', ("refresh", "alpha"), ("refresh_all",)),
    "mcp_test": Case("POST /api/mcp/test", "/api/mcp/test", _mcp,
                     b'{"server": {"id": "alpha", "transport": "http", "url": "http://127.0.0.1:9/mcp"}}',
                     ("test", "http://127.0.0.1:9/mcp"), None),
    "i18n_regenerate": Case("POST /api/ui/i18n/regenerate", "/api/ui/i18n/regenerate", _regenerate,
                            b'{"language": "de"}', ("regenerate", "de"), ("regenerate", "ru")),
    "clawhub_update": Case("POST /api/marketplace/clawhub/update/{name}", "/api/marketplace/clawhub/update/demo",
                           _clawhub_update, b'{"version": "1.2.0"}', ("update", "demo", "1.2.0"),
                           ("update", "demo", None)),
    "source_change": Case("POST /api/skill-sources/{source_id}", "/api/skill-sources/src1", _source_change,
                          b'{"trust": "review"}', ("edit", {"change": "edit", "source": "src1", "trust": "review"}),
                          ("edit", {"change": "edit", "source": "src1"})),
    "publish_preflight": Case("POST /api/skills/{skill}/publish-preflight", "/api/skills/demo/publish-preflight",
                              _publish_preflight, b'{"destination": "hub"}', ("preflight", "demo", "hub"),
                              ("preflight", "demo", None)),
    "auto_grant": Case("POST /api/owner/auto-grant", "/api/owner/auto-grant", _auto_grant, b'{"enabled": true}',
                       ("save", {"OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "true"}), None),
}


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "")
    inner = Starlette(routes=collect_routes(data_dir=tmp_path))
    inner.state.drive_root = tmp_path
    with TestClient(server_auth.NetworkAuthGate(inner)) as client:
        yield client


@pytest.fixture(params=sorted(CASES))
def case(request, monkeypatch):
    seen: list = []
    chosen = CASES[request.param]
    chosen.install(monkeypatch, seen)
    return chosen, seen


@pytest.mark.parametrize("label", LABELS)
def test_an_object_reaches_the_effect_whatever_its_label(api, case, label):
    chosen, seen = case
    response = api.post(chosen.path, content=chosen.body, headers={"content-type": label} if label else {})
    assert response.status_code == 200, response.text
    assert seen == [chosen.acted]


def test_an_empty_body_keeps_the_routes_absent_body(api, case):
    chosen, seen = case
    response = api.post(chosen.path)
    if chosen.absent is None:
        assert response.status_code == 400 and seen == []
        assert_bound(chosen.token, response)
    else:
        assert response.status_code == 200 and seen == [chosen.absent]


@pytest.mark.parametrize("body", UNREADABLE)
def test_an_unreadable_body_is_refused_with_no_effect(api, case, body):
    chosen, seen = case
    response = api.post(chosen.path, content=body, headers={"content-type": "application/x-www-form-urlencoded"})
    assert response.status_code == 400 and seen == [], response.text
    assert_bound(chosen.token, response)
    if chosen.token == "POST /api/skills/{skill}/publish-preflight":
        assert response.json()["skill"] == "demo"


def test_provenance_and_the_password_answer_before_any_body(api, case, monkeypatch):
    chosen, seen = case
    refused = api.post(chosen.path, content=chosen.body, headers={
        "origin": "http://evil.example", "sec-fetch-site": "cross-site", "content-type": "text/plain"})
    assert refused.status_code == 403 and refused.json()["code"] == "cross_origin_request"
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "secret")
    assert api.post(chosen.path, content=chosen.body).status_code == 401  # the test client is not loopback
    assert seen == []


def test_cancel_keeps_whitespace_as_its_absent_body(api, monkeypatch):
    """Cancel alone treats a whitespace body as absent (its legacy single-task cancel); other bytes are read."""
    from ouroboros.gateway import tasks

    seen: list = []
    monkeypatch.setattr(tasks, "migrating_error", lambda _root: None)
    monkeypatch.setattr(tasks, "_record_stop", lambda task_id, **kwargs: seen.append((task_id, kwargs)))
    for body in (b"", b" \r\n", b'{"cascade": true}'):
        assert api.post("/api/tasks/t1/cancel", content=body, headers={"content-type": "text/plain"}).status_code == 404
    assert [kwargs["cascade"] for _task, kwargs in seen] == [False, False, True]
    seen.clear()
    for body in (b"cascade=true", b"null", b"[]"):
        response = api.post("/api/tasks/t1/cancel", content=body)
        assert response.status_code == 400 and seen == [], response.text

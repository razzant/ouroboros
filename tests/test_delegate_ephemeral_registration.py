"""Host-minted roots register nothing; other roots register with the engine's ``created``.

A root Ouroboros mints for one task, run or review declares ``scope.ephemeral`` on an
engine that accepts it and never becomes a permanent engine project. Every other root
keeps its registration, read with one keyed POST once the engine's answers carry
``created``; an engine whose answers lack it keeps the previous GET-then-POST flow.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_registration_policy as policy
from ouroboros.config import CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION
from ouroboros.gateways import claudexor as cx
from tests._delegated_transport_shared import (  # noqa: F401 - autouse transport binding
    _delegating_ctx,
    _gateway,
    _owned_gateway_uses_each_test_transport,
)


@pytest.fixture(autouse=True)
def _fresh_field_memory(monkeypatch):
    monkeypatch.setattr(policy, "_CREATED_REPORTED", {}, raising=False)


class _Engine:
    """Project registry with the engine's semantics: idempotent per root."""

    def __init__(self, version="3.25.1", *, reports_created=True, known=None, refuse_post=False):
        self.engine_version, self.engine_build_sha = version, "build-1"
        self.reports_created, self.refuse_post = reports_created, refuse_post
        self.known = dict(known or {})
        self.calls = []

    def find_project_id(self, root):
        self.calls.append(("GET", root))
        return self.known.get(root, "")

    def project_registration(self, root):
        self.calls.append(("POST", root))
        if self.refuse_post:
            raise cx.ClaudexorUnavailable("daemon_busy", "daemon RPC timeout", status_code=503)
        fresh = root not in self.known
        self.known.setdefault(root, f"prj-{len(self.known) + 1}")
        return self.known[root], (fresh if self.reports_created else None)

    def register_project(self, root):
        return self.project_registration(root)[0]


@pytest.mark.parametrize("version,ephemeral", [(CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION, True), ("3.3.6", False)])
def test_a_minted_root_registers_nothing_where_the_engine_accepts_the_field(version, ephemeral):
    engine = _Engine(version, reports_created=False)
    got = policy.resolve_registration(engine, "/w/task", "", "readonly", minted=True)
    if ephemeral:
        assert got == ("", "", False, True) and engine.calls == []
    else:  # the older engine's strict scope refuses the field: the root registers as before
        assert got == ("prj-1", "prj-1", False, False)
        assert engine.calls == [("GET", "/w/task"), ("POST", "/w/task")]
    user = policy.resolve_registration(_Engine(version, reports_created=False), "/user", "/snap", "workspace_write")
    assert user == ("prj-1", "prj-1", True, False), "a user root is never ephemeral (#362 keeps it)"


def test_an_engine_reporting_created_registers_with_one_post_and_never_lists_projects():
    engine = _Engine(known={"/user": "prj-1"})
    # Unobserved engine, root already registered: the previous GET adopts it and one
    # optional POST learns that the answers carry ``created``.
    assert policy.resolve_registration(engine, "/user", "", "readonly") == ("prj-1", "", False, False)
    assert engine.calls == [("GET", "/user"), ("POST", "/user")]
    engine.calls.clear()
    assert policy.resolve_registration(engine, "/fresh", "", "readonly") == ("prj-2", "prj-2", False, False)
    assert policy.resolve_registration(engine, "/user", "", "readonly") == ("prj-1", "", False, False)
    assert engine.calls == [("POST", "/fresh"), ("POST", "/user")], "no GET /v2/projects once learned"


def test_an_engine_without_created_keeps_the_previous_flow():
    engine = _Engine(reports_created=False, known={"/user": "prj-1"})
    assert policy.resolve_registration(engine, "/user", "", "readonly") == ("prj-1", "", False, False)
    assert engine.calls == [("GET", "/user"), ("POST", "/user")]  # the one learning POST
    engine.calls.clear()
    assert policy.resolve_registration(engine, "/user", "", "readonly") == ("prj-1", "", False, False)
    assert policy.resolve_registration(engine, "/fresh", "", "readonly") == ("prj-2", "prj-2", False, False)
    assert engine.calls == [("GET", "/user"), ("GET", "/fresh"), ("POST", "/fresh")]


def test_the_learning_post_beside_a_found_root_is_optional():
    engine = _Engine(known={"/user": "prj-1"}, refuse_post=True)
    assert policy.resolve_registration(engine, "/user", "", "readonly") == ("prj-1", "", False, False)
    assert policy._CREATED_REPORTED == {}, "an unanswered POST teaches nothing"
    with pytest.raises(cx.ClaudexorUnavailable):  # a root that needs registering still needs the POST
        policy.resolve_registration(engine, "/fresh", "", "readonly")


def test_a_gateway_without_the_receipt_keeps_the_previous_flow_exactly():
    calls = []

    class _Legacy:
        engine_version = "3.25.1"

        def find_project_id(self, root):
            calls.append(("GET", root))
            return "prj-existing" if root == "/user" else ""

        def register_project(self, root):
            calls.append(("POST", root))
            return "prj-new"

    assert policy.resolve_registration(_Legacy(), "/user", "", "readonly") == ("prj-existing", "", False, False)
    assert policy.resolve_registration(_Legacy(), "/fresh", "", "readonly") == ("prj-new", "prj-new", False, False)
    assert calls == [("GET", "/user"), ("GET", "/fresh"), ("POST", "/fresh")]


@pytest.mark.parametrize("answer,created", [
    ({"id": "prj-1", "root": "/x", "created": True}, True),
    ({"id": "prj-1", "root": "/x", "created": False}, False),
    ({"id": "prj-1", "root": "/x"}, None),  # an engine that predates the field
    ({"id": "prj-1", "root": "/x", "created": "true"}, None),  # not the typed field
])
def test_the_gateway_reads_created_by_field_presence(answer, created):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("Idempotency-Key")))
        return httpx.Response(200, json=answer)

    with _gateway(handler) as gateway:
        assert gateway.project_registration("/x") == ("prj-1", created)
        assert gateway.register_project("/x") == "prj-1"
    assert [(method, path) for method, path, _key in seen] == [("POST", "/v2/projects")] * 2
    assert all(key for _method, _path, key in seen), "registration stays keyed"


# -- the delegated start, on the wire -------------------------------------------------

def _start(tmp_path, monkeypatch, *, version, register_root=False, folderless=False):
    """One read-only ``_delegate_start`` against a recording engine double."""
    from ouroboros import delegate_readonly_inputs
    from ouroboros.subagent_worktrees import _registry_path
    from ouroboros.tools import delegate

    data = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    ctx = _delegating_ctx(tmp_path, acting=False, task_id="t-ephemeral")
    if register_root:  # the host's own registry names the root: a child's copy
        _registry_path().parent.mkdir(parents=True, exist_ok=True)
        _registry_path().write_text(json.dumps({"worktrees": [{"path": str(ctx.repo_dir)}]}), encoding="utf-8")
    scratch = tmp_path / "scratch" / "inv"
    if folderless:
        scratch.mkdir(parents=True)
        monkeypatch.setattr(delegate_readonly_inputs, "prepare_folderless_inputs",
                            lambda _ctx, _invocation: (str(scratch), "\nREADONLY INPUTS\n"))
    engine, seen = _Engine(version, known={str(ctx.repo_dir): "prj-user"}), {}

    class _Daemon:
        engine_version, engine_build_sha = engine.engine_version, engine.engine_build_sha

        def handshake(self, **_kw): return {}
        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                   "accessProfilesSupported": ["readonly"]}]}
        def quota_snapshots(self): return []
        def find_project_id(self, root): return engine.find_project_id(root)
        def project_registration(self, root): return engine.project_registration(root)
        def register_project(self, root): return engine.register_project(root)
        def start_run(self, request, *, idempotency_key=""):
            seen["request"] = request
            return {"runId": "run-ephemeral", "runDir": str(tmp_path / "run")}
        def close(self): pass

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(cx, "ClaudexorGateway", lambda *a, **k: _Daemon())
    payload = json.loads(delegate._delegate_start(ctx, "survey the tree").text)
    assert payload["status"] == "started", payload
    row = custody.replay(custody.custody_root(ctx))["run-ephemeral"]
    return seen["request"]["scope"], engine.calls, row, (scratch if folderless else ctx.repo_dir)


@pytest.mark.parametrize("minted", ["registry_checkout", "folderless_scratch"])
def test_a_host_minted_root_starts_ephemeral_and_registers_nothing(tmp_path, monkeypatch, minted):
    scope, calls, row, root = _start(tmp_path, monkeypatch, version="3.25.1",
                                     register_root=minted == "registry_checkout",
                                     folderless=minted == "folderless_scratch")
    assert scope == {"kind": "project", "root": str(root), "ephemeral": True}
    assert calls == [], "no GET or POST /v2/projects for a one-shot root"
    assert (row.project_id, row.project_owned, row.project_persistent) == ("", False, False)


def test_a_minted_root_on_an_older_engine_registers_as_before(tmp_path, monkeypatch):
    scope, calls, row, root = _start(tmp_path, monkeypatch, version="3.3.6", register_root=True)
    assert scope == {"kind": "project", "root": str(root)}, "an older strict schema never sees the field"
    assert calls == [("GET", str(root)), ("POST", str(root))]
    assert (row.project_id, row.project_owned) == ("prj-user", False)


def test_a_user_root_keeps_its_registration_and_wire(tmp_path, monkeypatch):
    scope, calls, row, root = _start(tmp_path, monkeypatch, version="3.25.1")
    assert scope == {"kind": "project", "root": str(root)}
    assert calls[0] == ("GET", str(root)) and (row.project_id, row.project_owned) == ("prj-user", False)

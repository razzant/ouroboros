"""Host-minted roots register nothing; other roots register with the engine's ``created``.

A root Ouroboros mints for one task, run or review (a folder inside a task's own drive
included) declares ``scope.ephemeral`` on an engine that accepts it and never becomes a
permanent engine project; a delegated plan thread on it does the same from
``CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION``. Every other root keeps its registration, read
with one keyed POST once the engine's answers carry ``created``; an engine whose answers
lack it keeps the previous GET-then-POST flow.
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
    _select_transport_access,
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

def _start(tmp_path, monkeypatch, *, version, register_root=False, folderless=False, under=(), mode=None, known=True):
    """One read-only ``_delegate_start`` against a recording engine double; ``under`` moves the
    started folder below the data root (``tmp_path``, the custody drive)."""
    from ouroboros import delegate_readonly_inputs
    from ouroboros.subagent_worktrees import _put_record
    from ouroboros.tools import delegate

    data = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    ctx = _delegating_ctx(tmp_path, acting=False, task_id="t-ephemeral")
    if under:
        ctx.repo_dir = tmp_path.joinpath(*under)
        ctx.repo_dir.mkdir(parents=True)
    _select_transport_access(monkeypatch, "readonly")  # technical access decides the shape
    if register_root:  # the host's own copy record names the root: a child's copy
        _put_record({"path": str(ctx.repo_dir)}, data)
    scratch = tmp_path / "scratch" / "inv"
    if folderless:
        scratch.mkdir(parents=True)
        monkeypatch.setattr(delegate_readonly_inputs, "prepare_readonly_inputs",
                            lambda _ctx, _invocation, *_selection, thread_folder=None: delegate_readonly_inputs.ReadonlyInputs(
                                str(scratch), "\nREADONLY INPUTS\n", [], ""))
    engine, seen = _Engine(version, known={str(ctx.repo_dir): "prj-user"} if known else {}), {}

    class _Daemon:
        engine_version, engine_build_sha = engine.engine_version, engine.engine_build_sha

        def handshake(self, **_kw): return {}
        def agent_capabilities(self):
            return {"modes": ["ask", "plan", "agent"], "harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                   "modes": ["ask", "plan", "agent"],
                                   "accessProfilesSupported": ["readonly"]}]}
        def quota_snapshots(self): return []
        def find_project_id(self, root): return engine.find_project_id(root)
        def project_registration(self, root): return engine.project_registration(root)
        def register_project(self, root): return engine.register_project(root)
        def create_thread(self, request, *, idempotency_key):
            seen["thread"] = request
            return {"id": "thread-ephemeral"}
        def start_run(self, request, *, idempotency_key=""):
            seen["request"] = request
            return {"runId": "run-ephemeral", "runDir": str(tmp_path / "run")}
        def close(self): pass

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(cx, "ClaudexorGateway", lambda *a, **k: _Daemon())
    payload = json.loads(delegate._delegate_start(ctx, "survey the tree", mode=mode).text)
    assert payload["status"] == "started", payload
    row = custody.replay(custody.custody_root(ctx))["run-ephemeral"]
    return seen["request"]["scope"], engine.calls, row, (scratch if folderless else ctx.repo_dir)


def test_a_plan_start_keeps_its_owned_registration_below_the_thread_floor(tmp_path, monkeypatch):
    from ouroboros.config import CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION

    for version, ephemeral in [("5.0.1", False), (CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION, True)]:
        case = tmp_path / version
        case.mkdir()
        scope, calls, row, root = _start(case, monkeypatch, version=version, known=False, mode="plan",
                                        under=("task_drives", "task-1", "work"))
        assert row.thread_id == "thread-ephemeral"
        if ephemeral:
            assert scope.get("ephemeral") is True and calls == [] and not row.project_id and not row.project_owned
        else:
            assert "ephemeral" not in scope and calls == [("GET", str(root)), ("POST", str(root))]
            assert row.project_id == "prj-1" and row.project_owned


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


@pytest.mark.parametrize("under,minted", [
    (("task_drives", "t-ephemeral", "work", "repo"), True),
    (("state", "headless_tasks", "t-child", "data", "repo"), True),
    (("projects", "p-1"), False),  # an owner's project under the data root is an identity
    (("task_drives",), False),  # the drives' parent is no task's drive
])
def test_a_folder_inside_a_task_drive_starts_ephemeral(tmp_path, monkeypatch, under, minted):
    scope, calls, row, root = _start(tmp_path, monkeypatch, version="3.25.1", under=under)
    if minted:  # the drive custody pass retires the folder with its task: no engine identity
        assert scope == {"kind": "project", "root": str(root), "ephemeral": True}
        assert calls == [] and (row.project_id, row.project_owned, row.project_persistent) == ("", False, False)
    else:
        assert scope == {"kind": "project", "root": str(root)} and calls[0] == ("GET", str(root))


def test_minted_root_names_the_host_layouts_and_follows_links(tmp_path):
    data = tmp_path / "data"
    outside = tmp_path / "owner-project"
    outside.mkdir()
    (data / "task_drives" / "t-1").mkdir(parents=True)
    (data / "task_drives" / "t-1" / "link").symlink_to(outside)
    assert policy.minted_root(str(data / "task_drives" / "t-1" / "work"), data)
    assert policy.minted_root(str(data / "task_drives" / "t-1"), data)
    assert policy.minted_root(str(data / "state" / "headless_tasks" / "t-2" / "data"), data)
    assert policy.minted_root(str(data / "state" / "review_checkouts" / "tok" / "repo"), data)
    for root in (data, data / "task_drives", data / "state" / "headless_tasks", data / "projects" / "p-1",
                 data / "task_drives" / "t-1" / "link", outside):
        assert not policy.minted_root(str(root), data), root  # a link names where it leads


@pytest.mark.parametrize("scope", [{"ephemeral": True}, {}])
def test_a_delegated_plan_thread_on_a_minted_root_is_ephemeral_from_the_thread_floor(scope):
    from types import SimpleNamespace

    from ouroboros import config
    from ouroboros.delegate_plan import bind_thread

    created = []

    class Engine:
        def create_thread(self, request, *, idempotency_key):
            created.append(request)
            return {"id": "thread-1"}

    request = {"scope": {"kind": "project", "root": "/w/copy", **scope}, "mode": "plan", "access": "readonly"}
    engine = Engine()
    for version, allowed in [("5.0.1", False), ("5.0.2", False), (config.CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION, True)]:
        engine.engine_version = version
        assert bind_thread(engine, request, SimpleNamespace(route_id="r", profile_id=""), "inv-1") == {
            "thread_id": "thread-1"}
        assert created[-1]["scope"] == {
            "kind": "project", "root": "/w/copy", **(scope if allowed else {})}

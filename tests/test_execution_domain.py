"""A managed deployment's execution domain: the pinned executor, the work engine and the binding each native run
carries from its pending invocation onward, and Panic of the dedicated containers (``ouroboros.execution_domain``).

Each behaviour is proved at a real consumer, managed and unmanaged: the policy reader; ``run_command``,
``run_script`` and ``start_service`` through the registry over a faked Docker CLI; the work engine's own gateway
over a mocked loopback transport; the custody writers and readers (pending invocation, STARTED, replay, absence,
the per-engine sweep); the delegate tools; and ``execute_panic_stop``. No container, daemon, VM or network is used.
"""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess
import time
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import execution_domain as domain
from ouroboros.gateways import claudexor as wire
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests._safety_fixtures import disable_safety
from tests._store_shared import seed_result
from tests._workspace_executor_shared import fake_docker_cli

WORK_PORT, STORE = 47111, "store-work"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A policy named the way a deployment's launch names it, an administrator folder and a work root."""
    from ouroboros import deployment_policy

    admin, work = tmp_path / "admin", tmp_path / "work"
    admin.mkdir()
    work.mkdir()
    monkeypatch.delenv(deployment_policy.POLICY_ENV, raising=False)
    monkeypatch.delenv(deployment_policy.GENERATION_ENV, raising=False)
    deployment_policy.reset_for_tests()

    def manage(section=None, **fields):
        path = admin / "deployment-policy.json"
        path.write_text(json.dumps({"version": 1, **fields, **({"execution": section} if section else {})}),
                        encoding="utf-8")
        monkeypatch.setenv(deployment_policy.POLICY_ENV, str(path))
        deployment_policy.reset_for_tests()
        return path

    yield SimpleNamespace(root=tmp_path, admin=admin, work=work.resolve(), manage=manage)
    deployment_policy.reset_for_tests()


def section(env, *, engine=True, containers=("ouro-work", "ouro-engine"), state_root=None, **executor):
    raw = {"type": "docker_exec", "container_name": "ouro-work", "network": "host",
           "path_mappings": [{"host_path": str(env.work), "backend_path": str(env.work)}], **executor}
    result = {"executor": raw, **({"dedicated_containers": list(containers)} if containers is not None else {}),
              **({"skill_state_root": state_root} if state_root else {})}
    if engine:
        result["engine"] = {"descriptor": str(env.admin / "work-engine.json")}
    return result


def work_engine(env, monkeypatch, *, store=STORE, runs=None, starts=None):
    """The administrator's descriptor and token for the work engine, and that engine over a mocked loopback
    transport; any other port answers 599, so a request meant for the work engine cannot reach another."""
    token = env.admin / "work-engine.token"
    token.write_text("work-token", encoding="utf-8")
    (env.admin / "work-engine.json").write_text(json.dumps(
        {"host": "127.0.0.1", "port": WORK_PORT, "tokenPath": str(token)}), encoding="utf-8")
    calls = []

    def answer(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.headers.get("Idempotency-Key")))
        if request.url.port != WORK_PORT:
            return httpx.Response(599, json={"code": "not_the_work_engine"})
        if request.url.path == "/v2/handshake":
            return httpx.Response(200, json={"compatible": True, "protocolMajor": wire.CLAUDEXOR_PROTOCOL_MAJOR,
                "engine": {"version": wire.CLAUDEXOR_MIN_VERSION, "sha": "work"}, "servingMode": "normal",
                **({"storeId": store} if store else {})})
        if request.url.path == "/v2/runs" and request.method == "POST":
            (starts if starts is not None else []).append(json.loads(request.content))
            return httpx.Response(200, json={"runId": "run-w-new"})
        if request.url.path.startswith("/v2/runs/") and request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={"accepted": True})
        if request.url.path.startswith("/v2/runs/"):
            rid = request.url.path.split("/")[3]
            if runs is not None and rid in runs:
                return httpx.Response(200, json=runs[rid])
            return httpx.Response(404, json={"code": "run_not_found", "message": "no such run"})
        return httpx.Response(404, json={"code": "not_modeled"})

    client = httpx.Client
    monkeypatch.setattr(wire.httpx, "Client", lambda **kw: client(**kw, transport=httpx.MockTransport(answer)))
    return calls


def started(root, rid, *, engine_ref=None, store="", task="task-one"):
    entry = custody.RunCustody(run_id=rid, task_id=task, invocation_id=f"inv-{rid}", route_id="test", model="m",
                               access="readonly", mode="ask", ledger_root=str(root), engine_ref=dict(engine_ref or {}),
                               daemon_store_id=store)
    assert custody.record_started(root, entry)
    return current.run(root, rid)


def work_ref(env, store=STORE):
    return {"kind": domain.WORK_ENGINE, "descriptor": str(env.admin / "work-engine.json"), "store": store}


def _delegate_ctx(root, monkeypatch, task_id="task-one"):
    """A real registry's context for the delegate tools, its task recorded as running."""
    import ouroboros.safety as safety
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry._ctx.task_id = task_id
    write_task_result(root, task_id, "running", root_task_id=task_id, task_attempt=1)
    registry._ctx.task_metadata = {"root_task_id": task_id, "parent_task_id": task_id}
    return registry._ctx


# --- the policy section ----------------------------------------------------------------------------------------

def test_without_an_execution_section_nothing_is_managed_here(env):
    from ouroboros.workspace_executor import executor_ref_from_ctx

    assert domain.current() is None and domain.managed_executor_ref() is None  # personal OSS: no policy at all
    env.manage(None, access={"max": "pro"})  # a managed deployment that names no execution domain
    assert domain.current() is None and not domain.panic_scope()
    ctx = SimpleNamespace(executor_ref={"type": "docker_exec", "container_name": "task-box",
                                        "workspace_host_path": str(env.work), "workspace_backend_path": "/w"})
    assert executor_ref_from_ctx(ctx).container_name == "task-box"  # a task's own executor stays its own


def test_an_execution_section_pins_one_domain(env):
    env.manage(section(env))
    found = domain.current()
    assert found.executor.kind == "docker_exec" and found.executor.managed and found.executor.container_name == "ouro-work"
    assert [(str(m.host_path), m.backend_path) for m in found.executor.mappings] == [(str(env.work), str(env.work))]
    assert found.engine_descriptor == str(env.admin / "work-engine.json")
    assert found.dedicated_containers == ("ouro-work", "ouro-engine") and domain.panic_scope()


@pytest.mark.parametrize("change,reason", [
    (lambda s, e: s["executor"].update(type="local"), "must be 'docker_exec'"),
    (lambda s, e: s["executor"].pop("container_name"), "container_name must name one container"),
    (lambda s, e: s["executor"].update(container_name="a b"), "container_name must name one container"),
    (lambda s, e: s["executor"].update(path_mappings=[]), "path_mappings must list at least one mapping"),
    (lambda s, e: s["executor"].update(network="bridge"), "network must be 'host' or 'none'"),
    (lambda s, e: s["executor"].update(extra=1), "unsupported field 'extra'"),
    (lambda s, e: s.update(sandbox={}), "unsupported field 'sandbox'"),
    (lambda s, e: s["executor"]["path_mappings"].append({"host_path": "relative", "backend_path": "/r"}),
     "host_path must be an absolute path"),
    (lambda s, e: s["engine"].update(descriptor="engine.json"), "descriptor must be an absolute path"),
    (lambda s, e: s.update(dedicated_containers=[]), "must name the work engine's container"),
    (lambda s, e: s.update(dedicated_containers=["a", "a"]), "must list distinct container names"),
])
def test_an_unusable_execution_section_never_reads_as_unmanaged(env, change, reason):
    from ouroboros import config, deployment_policy

    document = section(env)
    change(document, env)
    env.manage(document)
    for read in (domain.current, deployment_policy.current, config.load_settings):
        with pytest.raises(deployment_policy.DeploymentPolicyError, match=reason):
            read()


@pytest.mark.parametrize("which", ["data root", "checkout"])
def test_the_domain_may_not_reach_what_enforces_the_policy(env, monkeypatch, which):
    from ouroboros import deployment_policy

    roots = {"data root": env.root / "service-data" / "data", "checkout": env.root / "service-code" / "repo"}
    for name, path in roots.items():
        path.mkdir(parents=True)
        monkeypatch.setenv("OUROBOROS_DATA_DIR" if name == "data root" else "OUROBOROS_REPO_DIR", str(path))
    for host in (roots[which] / "inner", roots[which].parent):  # inside it, or containing it
        document = section(env)
        document["executor"]["path_mappings"] = [{"host_path": str(host), "backend_path": "/srv/work"}]
        env.manage(document)
        with pytest.raises(deployment_policy.DeploymentPolicyError, match=f"overlaps the {which}"):
            domain.current()
    document = section(env)
    document["engine"]["descriptor"] = str(roots[which] / "engine.json")
    env.manage(document)
    with pytest.raises(deployment_policy.DeploymentPolicyError, match=f"overlaps the {which}"):
        domain.current()


def test_a_skill_state_root_is_the_work_domains_own_and_kept_apart(env, monkeypatch):
    """``skill_state_root`` is a normalized folder the work sees at its own address; neither it nor a mapping may
    reach the policy file, the work engine's descriptor or the service's import path."""
    from ouroboros import deployment_policy

    env.manage(section(env, state_root=str(env.work / "skill-state")))
    assert domain.current().skill_state_root == str(env.work / "skill-state")
    outside, shifted = env.root / "elsewhere", env.root / "shifted"
    for root, reason in ((str(env.work / "a" / ".." / "b"), "normalized path"), ("state", "absolute path"),
                         (str(outside), "same address"), (str(shifted / "state"), "same address")):
        document = section(env, state_root=root)
        document["executor"]["path_mappings"].append({"host_path": str(shifted), "backend_path": "/srv/shifted"})
        env.manage(document)
        with pytest.raises(deployment_policy.DeploymentPolicyError, match=reason):
            domain.current()
    for target, which in ((env.admin, "deployment policy"), (env.admin / "work-engine.json", "work engine descriptor")):
        document = section(env)
        document["executor"]["path_mappings"].append({"host_path": str(target.parent if target.is_file() else target),
                                                       "backend_path": "/srv/admin"})
        env.manage(document)
        with pytest.raises(deployment_policy.DeploymentPolicyError, match=f"overlaps the {which}"):
            domain.current()
    library = env.work / "site-packages"
    library.mkdir()
    monkeypatch.syspath_prepend(str(library))  # the service imports from inside the work mapping: refused
    env.manage(section(env))
    with pytest.raises(deployment_policy.DeploymentPolicyError, match="overlaps the service's import path"):
        domain.current()


# --- the pinned executor at its real consumers -------------------------------------------------------------------

def _registry(env, workspace, executor_ref=None, task_id="exec-task"):
    """A real registry whose active workspace is ``workspace``: an allowed cwd, mapped into the domain or not."""
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    data, system = env.root / "drive", env.root / "system"
    data.mkdir(exist_ok=True)
    system.mkdir(exist_ok=True)
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace,
                                     workspace_mode="external", task_id=task_id, executor_ref=executor_ref or {}))
    return registry


def _docker(monkeypatch, calls):
    real_run = subprocess.run

    def fake(cmd, **kwargs):
        if list(cmd[:1]) != ["docker"]:
            return real_run(cmd, **kwargs)
        calls.append([str(part) for part in cmd])
        if cmd[1:3] == ["inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="host\n", stderr="")
        if cmd[1] == "exec" and "_ouroboros_proc_observe" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="exited\n", stderr="")
        if cmd[1] == "exec" and "nohup" in str(cmd[-1]):
            return subprocess.CompletedProcess(cmd, 0, stdout="4242\n", stderr="")
        if cmd[1] == "exec":
            return subprocess.CompletedProcess(cmd, 0, stdout="completed", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    fake_docker_cli(monkeypatch, fake)


def test_unmanaged_task_executor_and_host_fallback_are_unchanged(env, monkeypatch):
    disable_safety(monkeypatch)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    outside = env.root / "outside"
    outside.mkdir()
    task_box = {"type": "docker_exec", "container_name": "task-box",
                "path_mappings": [{"host_path": str(env.work), "backend_path": "/w"}]}
    host, mapped = _registry(env, outside, task_box), _registry(env, env.work, task_box)
    calls = []
    _docker(monkeypatch, calls)
    marker = outside / "ran-on-host"
    text = host.execute("run_command", {"cmd": ["touch", str(marker)]})
    assert marker.exists(), text  # outside the task executor's mappings the command runs on the host, as before
    assert not any(call[1] == "exec" for call in calls)
    mapped.execute("run_command", {"cmd": ["true"]})
    execs = [call for call in calls if call[1] == "exec"]
    assert execs and {call[call.index("--workdir") + 2] for call in execs} == {"task-box"}


def test_managed_work_runs_only_in_the_pinned_container_and_never_falls_back(env, monkeypatch):
    disable_safety(monkeypatch)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    env.manage(section(env))
    task_box = {"type": "docker_exec", "container_name": "task-box",  # a task's own selector names a wider box
                "path_mappings": [{"host_path": str(env.root), "backend_path": "/all"}]}
    outside = env.root / "outside"
    outside.mkdir()
    mapped, registry = _registry(env, env.work, task_box), _registry(env, outside, task_box)
    calls = []
    _docker(monkeypatch, calls)
    mapped.execute("run_command", {"cmd": ["true"]})
    execs = [call for call in calls if call[1] == "exec"]
    assert execs and all(call[call.index("--workdir") + 1] == str(env.work) for call in execs)
    assert {call[call.index("--workdir") + 2] for call in execs} == {"ouro-work"}  # the pin, never the task's box
    calls.clear()
    marker = outside / "ran-on-host"
    for tool, args in (("run_command", {"cmd": ["touch", str(marker)]}),
                       ("run_script", {"interpreter": "sh", "script": f"touch {marker}"}),
                       ("start_service", {"name": "web", "cmd": ["touch", str(marker)]}),
                       ("verify_and_record", {"contract_kind": "explicit_command", "check": f"touch {marker}"})):
        text = registry.execute(tool, args)
        assert "execution domain" in text and "nothing runs on the service" in text, (tool, text)
        assert not marker.exists(), tool
    assert not any(call[1] == "exec" for call in calls)  # refused before any backend start, too


# --- the work engine's gateway -----------------------------------------------------------------------------------

def test_the_work_gateway_proves_its_store_and_withholds_its_run_directory(env, monkeypatch):
    env.manage(section(env))
    detail = {"summary": {"runId": "run-w", "state": "running", "runDir": "/engine/runs/run-w"}, "lastSeq": 2}
    calls = work_engine(env, monkeypatch, runs={"run-w": detail})
    with domain.work_gateway({"kind": domain.WORK_ENGINE, "descriptor": str(env.admin / "work-engine.json")}) as gw:
        assert gw.engine_ref == work_ref(env) and gw.store_identity == STORE
        read = gw.get_run("run-w")
    assert "runDir" not in read["summary"] and read["summary"]["workRunDir"] == "/engine/runs/run-w"
    assert {path for _method, path, _key in calls} == {"/v2/handshake", "/v2/runs/run-w"}


@pytest.mark.parametrize("case,code", [
    ("other_descriptor", "execution_domain_changed"),
    ("no_engine", "execution_domain_unavailable"),
    ("unmanaged", "execution_domain_unavailable"),
    ("other_store", "work_engine_store_changed"),
    ("no_store", "work_engine_identity_unproven"),
])
def test_a_work_reference_never_reaches_another_engine(env, monkeypatch, case, code):
    env.manage(section(env, engine=case != "no_engine", containers=None if case == "no_engine" else ("e",)))
    if case == "unmanaged":
        from ouroboros import deployment_policy

        monkeypatch.delenv(deployment_policy.POLICY_ENV)
        deployment_policy.reset_for_tests()
    work_engine(env, monkeypatch, store="" if case == "no_store" else "store-now")
    ref = {**work_ref(env, store="store-now" if case != "other_store" else "store-then")}
    if case == "other_descriptor":
        ref["descriptor"] = str(env.admin / "old-engine.json")
    with pytest.raises(ClaudexorUnavailable) as raised:
        domain.run_gateway(ref, lambda: pytest.fail("an owned-engine default for a work run"))
    assert raised.value.code == code


def test_a_start_naming_a_path_the_domain_does_not_see_is_never_sent(env, monkeypatch):
    env.manage(section(env))
    starts = []
    work_engine(env, monkeypatch, starts=starts)
    with domain.work_gateway(work_ref(env)) as gw:
        for request in ({"scope": {"root": str(env.root / "service-only")}},
                        {"scope": {"root": str(env.work)}, "execution": {"auxiliaryWritableRoots": [str(env.root)]}}):
            with pytest.raises(ClaudexorUnavailable) as raised:
                gw.start_run(request, idempotency_key="inv-1")
            assert raised.value.code == "work_path_unmapped"
        assert gw.start_run({"scope": {"root": str(env.work / "repo")}}, idempotency_key="inv-2")["runId"]
    assert [start["scope"]["root"] for start in starts] == [str(env.work / "repo")]


# --- custody: bound before the POST, replayed, and the only engine that answers for the run -----------------------

def test_the_engine_rides_the_pending_invocation_started_row_and_replay(env, tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    ref = work_ref(env)
    body = {"prompt": "p", "scope": {"kind": "project", "root": str(env.work)}}
    assert custody.record_start_requested(drive, run_id="", task_id="t", idempotency_key="k", invocation_id="inv-w",
                                          request=body, project_id="", route="r", engine_ref=ref)
    record = custody.invocation_record(drive, "inv-w")
    assert record["state"] == "pending" and record["engine_ref"] == ref and record["request"] == body
    assert [row.get("engine_ref") for row in custody.pending_invocations(drive)] == [ref]
    assert custody.record_start_requested(drive, run_id="", task_id="t", idempotency_key="k2", invocation_id="inv-s",
                                          request=body, project_id="", route="r")
    assert "engine_ref" not in custody.invocation_record(drive, "inv-s")  # an owned-engine row keeps its shape
    entry = started(drive, "run-w", engine_ref=ref, store=STORE)
    assert entry.engine_ref == ref and entry.daemon_store_id == STORE
    custody.record_started(drive, custody.RunCustody(run_id="run-w", task_id="task-one"))  # a later bare row
    assert current.run(drive, "run-w").engine_ref == ref  # first wins
    rows = [row for row in custody.custody_rows(drive) if row["type"] == custody.STARTED]
    owned = started(drive, "run-s")
    assert owned.engine_ref == {} and all("engine_ref" not in row for row in custody.custody_rows(drive)
                                          if row["type"] == custody.STARTED and row["run_id"] == "run-s")
    assert rows[0]["engine_ref"] == ref


def _gateway(store, ref=None):
    return SimpleNamespace(store_identity=store, engine_ref=ref, close=lambda: None)


def test_another_engines_absence_never_settles_a_run(env, tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    ref = work_ref(env)
    work_run, owned_run = started(drive, "run-w", engine_ref=ref, store=STORE), started(drive, "run-s", store="store-s")
    gone = ClaudexorUnavailable("run_not_found", "gone", status_code=404)
    assert not custody.daemon_says_absent(gone, gateway=_gateway(STORE, ref), run=work_run)  # no answering-store proof
    gone.problem = {"context": {"storeId": STORE}}
    assert not custody.daemon_says_absent(gone, gateway=_gateway("store-s"), run=work_run)  # the owned engine's 404
    assert not custody.daemon_says_absent(gone, gateway=_gateway(STORE), run=work_run)  # same store id, no binding
    assert not custody.daemon_says_absent(gone, gateway=_gateway(STORE, {**ref, "store": "other"}), run=work_run)
    assert custody.daemon_says_absent(gone, gateway=_gateway(STORE, ref), run=work_run)
    assert not custody.daemon_says_absent(gone, gateway=_gateway("store-s", ref), run=owned_run)  # and the reverse
    assert not custody.daemon_says_absent(gone, gateway=_gateway("store-s"), run=owned_run)
    owned_gone = ClaudexorUnavailable("run_not_found", "gone", status_code=404)
    owned_gone.problem = {"context": {"storeId": "store-s"}}
    assert custody.daemon_says_absent(owned_gone, gateway=_gateway("store-s"), run=owned_run)

    def absent(*_a, **_kw):
        raise gone

    wrong = SimpleNamespace(store_identity="store-s", engine_ref=None, get_run=absent)
    assert custody._reconcile_one(drive, wrong, work_run)["action"] == "unreadable"
    assert not current.run(drive, "run-w").settled


def test_a_registration_is_retired_only_through_its_own_engine(env, tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    drive.mkdir()
    entry = custody.RunCustody(run_id="run-w", task_id="t", project_id="prj-w", project_owned=True,  # a review
                               source="review_substrate:seat", engine_ref=work_ref(env), daemon_store_id=STORE,  # seat's run
                               ledger_root=str(drive))
    assert custody.record_started(drive, entry)
    assert custody.emit(drive, custody.SETTLED, {"run_id": "run-w", "task_id": "t", "route": "r"})
    assert current.run(drive, "run-w").settled
    removed = []
    owned = SimpleNamespace(store_identity="store-s", engine_ref=None,
                            remove_project=lambda pid: (removed.append(("owned", pid)), (_ for _ in ()).throw(
                                ClaudexorUnavailable("project_not_found", "gone", status_code=404)))[0])
    custody.retire_project(drive, owned, current.run(drive, "run-w"))
    assert removed == [] and current.run(drive, "run-w").project_owned  # the owned engine's 404 retires nothing
    work = SimpleNamespace(store_identity=STORE, engine_ref=work_ref(env), remove_project=lambda pid: removed.append(("work", pid)))
    custody.retire_project(drive, work, current.run(drive, "run-w"))
    assert removed == [("work", "prj-w")] and not current.run(drive, "run-w").project_owned


def test_the_sweep_replays_and_settles_each_engines_work_only_through_that_engine(env, tmp_path, monkeypatch):
    from ouroboros import delegate_custody_reconcile as reconcile

    env.manage(section(env))
    drive = tmp_path / "drive"
    drive.mkdir()
    ref = work_ref(env)
    started(drive, "run-s", store="store-s")
    started(drive, "run-w", engine_ref=ref, store=STORE)
    body = {"prompt": "p", "scope": {"kind": "project", "root": str(env.work)}}
    assert custody.record_start_requested(drive, run_id="", task_id="gone", idempotency_key="k", invocation_id="inv-w",
                                          request=body, project_id="", route="r", engine_ref=ref)
    seen = {"owned": [], "work": []}

    def engine(name, store, engine_ref):
        def get_run(rid, **_kw):
            seen[name].append(("get", rid))
            return {"summary": {"runId": rid, "state": "running"}, "lastSeq": 1}

        def start_run(request, idempotency_key=""):
            seen[name].append(("start", idempotency_key, request))
            return {"runId": "run-w-replayed", "storeId": store}  # admission receipt, not the earlier handshake
        return SimpleNamespace(store_identity=store, engine_ref=engine_ref, handshake=lambda: {}, close=lambda: None,
                               get_run=get_run, start_run=start_run, cancel_run=lambda rid, **_kw: {"accepted": True})

    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: engine("work", STORE, ref))
    outcomes = reconcile._reconcile_each(drive, custody.open_runs(drive), lambda: engine("owned", "store-s", None),
                                         pending=custody.pending_invocations(drive), replay_pending=lambda: True)
    assert ("get", "run-s") in seen["owned"] and ("get", "run-w") not in seen["owned"]
    assert ("get", "run-w") in seen["work"] and ("get", "run-s") not in seen["work"]
    assert not any(step[0] == "start" for step in seen["owned"]), "a work engine's lost create never reaches another"
    assert [(key, request) for step, key, request in (s for s in seen["work"] if s[0] == "start")] == [("inv-w", body)]
    assert {o.get("run_id") for o in outcomes} >= {"run-s", "run-w"}
    # The recovered run keeps the engine and store its invocation was bound to, so the NEXT sweep asks that engine
    # alone: the owned engine's 404 never closes the live work run (critic probe recovered_engine_probe).
    recovered = current.run(drive, "run-w-replayed")
    assert recovered.engine_ref == ref and recovered.daemon_store_id == STORE
    seen["owned"].clear()
    seen["work"].clear()

    def owned_answering_absent():
        gateway = engine("owned", "store-s", None)

        def get_run(rid, **_kw):
            seen["owned"].append(("get", rid))
            raise ClaudexorUnavailable("run_not_found", "no such run", status_code=404)
        gateway.get_run = get_run
        return gateway

    reconcile._reconcile_each(drive, custody.open_runs(drive), owned_answering_absent,
                              pending=custody.pending_invocations(drive), replay_pending=lambda: True)
    assert ("get", "run-w-replayed") in seen["work"] and ("get", "run-w-replayed") not in seen["owned"]
    assert not current.run(drive, "run-w-replayed").settled


# --- the delegate tools: start, retry, continuation and controls ---------------------------------------------------

def test_a_start_binds_its_engine_and_a_retry_or_continuation_keeps_it(env, tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon

    drive = tmp_path / "drive"
    drive.mkdir()
    owned_opened = []
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **_kw: owned_opened.append(1) or _gateway("s"))
    facts, gateway = domain.start_engine(drive)
    assert facts == {} and owned_opened == [1]  # unmanaged: the owned engine, as before
    env.manage(section(env))
    work_engine(env, monkeypatch)
    facts, gateway = domain.start_engine(drive)
    assert facts == {"engine_ref": work_ref(env)} and gateway.engine_ref == work_ref(env)
    gateway.close()
    body = {"prompt": "p"}
    custody.record_start_requested(drive, run_id="", task_id="t", idempotency_key="k", invocation_id="inv-s",
                                   request=body, project_id="", route="r")
    facts, _gateway_s = domain.start_engine(drive, retry_token="inv-s")
    assert facts == {} and owned_opened == [1, 1]  # a pending owned-engine create is replayed where it was bound
    custody.record_start_requested(drive, run_id="", task_id="t", idempotency_key="k", invocation_id="inv-old",
                                   request=body, project_id="", route="r", engine_ref=work_ref(env, "store-then"))
    with pytest.raises(ClaudexorUnavailable, match="another store"):
        domain.start_engine(drive, retry_token="inv-old")  # never re-posted into a store that does not hold it
    started(drive, "run-w", engine_ref=work_ref(env), store=STORE)
    facts, gateway = domain.start_engine(drive, continuation_token="run-w")
    assert facts == {"engine_ref": work_ref(env)}
    gateway.close()


def test_a_work_start_with_an_unseen_path_is_refused_before_its_row(env, tmp_path):
    from ouroboros.delegate_start_claims import claimed_start_request

    env.manage(section(env))
    drive = tmp_path / "drive"
    drive.mkdir()
    landed, refusal = claimed_start_request(drive, task_id="t", invocation_id="inv-new", engine_ref=work_ref(env),
                                            request={"scope": {"root": str(env.root / "service-only")}})
    assert not landed and refusal["reason"] == "work_path_unmapped" and refusal["definitely_unrun"] is True
    assert custody.invocation_record(drive, "inv-new") is None


def test_wait_and_cancel_reach_the_engine_the_run_was_started_on(env, tmp_path, monkeypatch):
    from ouroboros.tools import delegate

    ctx = _delegate_ctx(tmp_path, monkeypatch)
    ref = work_ref(env)
    started(tmp_path, "run-w", engine_ref=ref, store=STORE)
    started(tmp_path, "run-s", store="store-s")
    reached = []

    def fake(name, store, engine_ref):
        def cancel_run(rid, **_kw):
            reached.append((name, "cancel", rid))
            return {"accepted": True}

        def get_run(rid, **_kw):
            reached.append((name, "get", rid))
            return {"summary": {"runId": rid, "state": "cancelled", "effectiveAccess": "readonly"}, "lastSeq": 3}
        return SimpleNamespace(store_identity=store, engine_ref=engine_ref, engine_version="5.1.0", handshake=lambda **_k: {},
                               close=lambda: None, cancel_run=cancel_run, get_run=get_run)

    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: fake("work", STORE, r))
    monkeypatch.setattr(wire, "ClaudexorGateway", lambda *a, **k: fake("owned", "store-s", None))
    for rid in ("run-w", "run-s"):
        delegate._delegate_cancel(ctx, rid, reason="test")
    assert {(name, rid) for name, step, rid in reached if step == "cancel"} == {("work", "run-w"), ("owned", "run-s")}
    assert not any(name == "owned" and rid == "run-w" for name, _step, rid in reached)


# --- Panic -------------------------------------------------------------------------------------------------------

def _panic(tmp_path, monkeypatch):
    """The real Panic coordinator and its execution-domain attempt; every other native stop effect is replaced, as
    ``test_launcher_panic.disarm_emergency_owners`` does, so no process-wide latch outlives this test."""
    from ouroboros import server_control

    monkeypatch.setattr(server_control, "_restart_stop_requested", False)
    for path in ("ouroboros.startup_historical_audit.audit.stop", "ouroboros.maintenance.child.stop",
                 "ouroboros.storage.count.stop", "ouroboros.storage_free.free.stop"):
        monkeypatch.setattr(path, lambda: None)
    for path in ("ouroboros.tools.shell.kill_all_tracked_subprocesses", "ouroboros.workspace_executor.kill_all_foreground",
                 "ouroboros.tools.services.kill_all_services", "ouroboros.extension_companion.panic_kill_all",
                 "ouroboros.process_custody.stop_ledgered_processes"):
        monkeypatch.setattr(path, lambda *a, **kw: [])
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda **kw: None)
    monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon", lambda **kw: None if kw.get("create") is False
                        else SimpleNamespace(stop_outcome=lambda: "nothing_to_stop"))
    monkeypatch.setattr("ouroboros.process_scope.panic_scope_census", lambda root: None)
    monkeypatch.setattr(server_control, "stop_restart_successor", lambda: [])
    monkeypatch.setattr(server_control.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda _root: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    started_at = time.monotonic()
    with pytest.raises(SystemExit) as raised:
        server_control.execute_panic_stop(None, lambda **_: None, data_dir=tmp_path, panic_exit_code=99,
                                          log=logging.getLogger("execution-domain-panic"))
    return raised.value.code, time.monotonic() - started_at


def test_panic_kills_the_measured_containers_and_never_waits_for_them(env, tmp_path, monkeypatch):
    import threading

    env.manage(section(env))
    calls, answered = [], threading.Event()

    def docker(*args):
        calls.append(list(args))
        if args[:3] == ("inspect", "-f", "{{.Id}}"):
            return subprocess.CompletedProcess(args, 0, stdout=f"{args[3]}-id\n", stderr="")
        if args[0] == "kill":
            answered.wait(10)  # a Docker that answers slowly: Panic's exit is not its to wait for
        return subprocess.CompletedProcess(args, 0, stdout="false\n", stderr="")

    monkeypatch.setattr(domain, "_docker", docker)
    monkeypatch.setattr(domain, "_MEASURED", {})
    assert domain.measure_containers() == {"ouro-work": "ouro-work-id", "ouro-engine": "ouro-engine-id"}
    calls.clear()
    code, elapsed = _panic(tmp_path, monkeypatch)
    still_asking = ["kill", "ouro-work-id", "ouro-engine-id"] in calls and not answered.is_set()
    answered.set()
    for thread in [t for t in threading.enumerate() if t.name.startswith("panic-")]:
        thread.join(10)
    assert code == 99 and elapsed < 4.5, elapsed  # the flag writes are bounded; the unanswered kill is not awaited
    assert still_asking  # requested by measured identity, never by name, and still unanswered at the exit


def test_panic_settlement_reports_only_what_docker_answered(env, tmp_path, monkeypatch):
    env.manage(section(env))
    monkeypatch.setattr(domain, "_MEASURED", {"ouro-work": "id-work", "ouro-engine": "id-engine"})
    answers = {("inspect", "-f", "{{.State.Running}}", "id-work"): (0, "false\n", ""),
               ("inspect", "-f", "{{.State.Running}}", "id-engine"): (0, "true\n", ""),
               ("inspect", "-f", "{{.Id}} {{.State.Running}}", "ouro-work"): (1, "", "Error: No such object"),
               ("inspect", "-f", "{{.Id}} {{.State.Running}}", "ouro-engine"): (0, "id-engine true\n", "")}
    monkeypatch.setattr(domain, "_docker", lambda *args: subprocess.CompletedProcess(args, *answers[args]))
    settled = domain.panic_settle(tmp_path)
    assert settled["stopped"] == ["ouro-work"] and set(settled["unconfirmed"]) == {"ouro-engine"}
    answers[("inspect", "-f", "{{.Id}} {{.State.Running}}", "ouro-work")] = (0, "id-new true\n", "")
    settled = domain.panic_settle(tmp_path)
    assert "recreated" in settled["unconfirmed"]["ouro-work"] and "ouro-work" not in settled["stopped"]
    row = json.loads((tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["type"] == "process_stop_unconfirmed" and row["purpose"] == "execution_domain"


def test_unmanaged_panic_names_no_execution_domain(env, tmp_path, monkeypatch):
    monkeypatch.setattr(domain, "_docker", lambda *a: pytest.fail("an unmanaged Panic runs no Docker command"))
    code, _elapsed = _panic(tmp_path, monkeypatch)
    assert code == 99


# --- the owned engine keeps raw inference; observers and passes reach each engine for its own objects ------------

def test_model_operations_keep_the_owned_engine_while_the_work_engine_serves_runs(env, tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon as daemon
    from ouroboros.gateways import claudexor_metadata as metadata

    env.manage(section(env))
    owned_home = tmp_path / "owned"
    (owned_home / "daemon").mkdir(parents=True)
    (owned_home / "daemon" / "token").write_text("owned-token", encoding="utf-8")
    (owned_home / "daemon" / "control-api.json").write_text(json.dumps(
        {"host": "127.0.0.1", "port": 47222, "tokenPath": str(owned_home / "daemon" / "token")}), encoding="utf-8")
    token = env.admin / "work-engine.token"
    token.write_text("work-token", encoding="utf-8")
    (env.admin / "work-engine.json").write_text(json.dumps(
        {"host": "127.0.0.1", "port": WORK_PORT, "tokenPath": str(token)}), encoding="utf-8")
    handshakes = []

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/handshake":
            handshakes.append(request.url.port)
            return httpx.Response(200, json={"compatible": True, "protocolMajor": wire.CLAUDEXOR_PROTOCOL_MAJOR,
                "engine": {"version": wire.CLAUDEXOR_MIN_VERSION}, "servingMode": "normal",
                "storeId": STORE if request.url.port == WORK_PORT else "store-owned"})
        return httpx.Response(200, json={"operations": []})

    client = httpx.Client
    monkeypatch.setattr(wire.httpx, "Client", lambda **kw: client(**kw, transport=httpx.MockTransport(answer)))
    monkeypatch.setattr(daemon, "attach_home", lambda: None)
    monkeypatch.setattr(daemon, "owned_config_dir", lambda: owned_home)
    monkeypatch.setattr(daemon, "get_owned_daemon", lambda: SimpleNamespace(reconcile_rotation=lambda _gw: None))

    def ensure():
        gateway = wire.ClaudexorGateway(wire.discover_daemon_at(owned_home))
        gateway.handshake()
        return gateway

    for _round in range(2):
        with metadata.model_gateway(ensure) as model:
            assert model._endpoint.port == 47222 and model.store_identity == "store-owned"
        with domain.work_gateway(work_ref(env)) as work:
            assert work._endpoint.port == WORK_PORT and work.store_identity == STORE
    # the owned engine's handshake stays proven beside the work engine's: model calls never re-handshake for it
    assert handshakes.count(47222) == 1 and handshakes.count(WORK_PORT) == 2


def test_review_seats_start_on_the_work_engine_and_recover_on_their_own(env, monkeypatch):
    from ouroboros import claudexor_daemon

    env.manage(section(env))
    work_engine(env, monkeypatch)
    with pytest.raises(ClaudexorUnavailable) as raised:
        domain.seat_gateway(None, str(env.root / "service-only"))
    assert raised.value.code == "work_path_unmapped"
    facts, gateway = domain.seat_gateway(None, str(env.work))
    assert facts == {"engine_ref": work_ref(env)}
    gateway.close()
    owned = []
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **_kw: owned.append(1) or _gateway("s"))
    facts, _owned_gateway = domain.seat_gateway({"state": "pending", "request": {}}, str(env.work))
    assert facts == {} and owned == [1]  # a seat recovered without a binding keeps the owned engine it began on


def test_the_body_observes_a_work_run_through_its_own_engine(env, tmp_path, monkeypatch):
    import threading

    from ouroboros import delegate_collection as body

    def runtime(root, factory):  # a stopped body runtime: its observation methods, no loop
        stop = threading.Event()
        stop.set()
        result = body.BodyRuntime(root, stop_event=stop, gateway_factory=factory)
        result.thread.join(10)
        assert not result.thread.is_alive()
        return result

    ref = work_ref(env)
    entry = started(tmp_path, "run-w", engine_ref=ref, store=STORE)
    asked = []
    owned = SimpleNamespace(store_identity="store-s", engine_ref=None, close=lambda: None,
                            get_run=lambda *a, **_k: asked.append(("owned",) + a) or {})

    def gone(*a, **_k):
        asked.append(("work",) + a)
        absent = ClaudexorUnavailable("run_not_found", "gone", status_code=404)
        absent.problem = {"context": {"storeId": STORE}}
        raise absent

    work = SimpleNamespace(store_identity=STORE, engine_ref=ref, close=lambda: asked.append(("closed",)), get_run=gone)
    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: work)
    observer = runtime(tmp_path, lambda: owned)
    observer._observe(entry)
    assert asked[0][0] == "work" and not any(step[0] == "owned" for step in asked)
    assert current.run(tmp_path, "run-w").settled  # its own engine's 404 is the absence; nobody else's could be
    observer._drop_observer_gateway()
    assert ("closed",) in asked


def test_the_router_answers_each_store_with_its_engine_and_waits_when_it_is_unreachable(env, monkeypatch):
    env.manage(section(env))
    owned = _gateway("store-s")
    work = SimpleNamespace(store_identity=STORE, engine_ref=work_ref(env), close=lambda: None)
    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: work)
    clients = domain.EngineClients(lambda: owned)
    assert clients.reach(custody.RunCustody(run_id="r")) is owned and clients.reach({"engine_ref": work_ref(env)}) is work
    assert clients.for_store("store-s") is owned and clients.for_store(STORE) is work and clients.for_store("") is owned
    assert clients.for_store("store-gone") is owned  # no engine serves it: the owner's store check keeps it waiting
    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: (_ for _ in ()).throw(
        ClaudexorUnavailable("daemon_unreachable", "down")))
    stand_in = domain.EngineClients(lambda: owned).reach({"engine_ref": work_ref(env)})
    assert stand_in.store_identity == ""
    with pytest.raises(ClaudexorUnavailable, match="down"):
        stand_in.take_run_result("run-w", idempotency_key="k")


def test_a_taken_decision_reaches_the_engine_that_holds_the_result(env, tmp_path, monkeypatch):
    from ouroboros import delegate_result_taken as taken
    from ouroboros import review_thread_retention as rt
    from ouroboros.task_store import read_document

    env.manage(section(env))
    task, root = "task-taken", tmp_path / "data"
    root.mkdir()
    seed_result(root, task, {"task_id": task, "status": "completed", "ts": "2026-11-08T00:00:00+00:00"})
    run = custody.RunCustody(run_id="run-1", task_id=task, root_task_id=task, settled=True, terminal_state="succeeded",
                             snapshot_id="snap-1", execution_root="/copies/run-1", target_root="/owner/repo",
                             access="workspace_write", mode="agent", patch_captured=True,
                             engine_ref=work_ref(env), daemon_store_id=STORE)
    monkeypatch.setattr(current, "runs", lambda _root, _where="", params=(): {"run-1": run} if params[0] == task else {})
    monkeypatch.setattr(rt, "tree_hold", lambda _root, _task, _doc, *, live, census: ("", {"run-1": run}))
    owned_told, told = [], []
    owned = SimpleNamespace(store_identity="store-s", engine_ref=None, run_result_taken_supported=lambda: True,
                            take_run_result=lambda rid, idempotency_key: owned_told.append(rid))
    work = SimpleNamespace(store_identity=STORE, engine_ref=work_ref(env), close=lambda: None,
                           run_result_taken_supported=lambda: True,
                           take_run_result=lambda rid, idempotency_key: told.append(rid) or {
                               "accepted": True, "status": "taken"})

    def a_pass():
        report = {"holds": {}, "observed": []}
        result = SimpleNamespace(root=root, live=lambda _task: False, census=lambda: set(), report=report,
                                 stopped=False, engine=lambda: owned, stop=lambda: False)
        result.hold = lambda reason: report["holds"].__setitem__(reason, report["holds"].get(reason, 0) + 1)
        result.work = domain.EngineClients(result.engine)
        return result

    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: (_ for _ in ()).throw(
        ClaudexorUnavailable("daemon_unreachable", "down")))
    assert taken.tell_engine(a_pass(), task) == rt.WAIT  # unreachable now: it waits; nobody records it absent
    assert owned_told == [] and not (read_document(root, task) or {}).get(taken.RECORDS)
    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: work)
    assert taken.tell_engine(a_pass(), task) == rt.DONE
    assert told == ["run-1"] and owned_told == [] and read_document(root, task)[taken.RECORDS]["run-1"]["state"] == "taken"


def test_supervision_uses_the_runs_recorded_engine(env, tmp_path, monkeypatch):
    from ouroboros import delegate_supervision as supervision
    from tests._run_wait_shared import publish, scripted_reads

    started(tmp_path, "run-w", engine_ref=work_ref(env), store=STORE)
    started(tmp_path, "run-u", engine_ref=work_ref(env), store=STORE)
    started(tmp_path, "run-s", store="store-s")
    work, owned = object(), object()
    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: work)
    monkeypatch.setattr(wire, "ClaudexorGateway", lambda: owned)
    assert supervision._loop_gateway(current.run(tmp_path, "run-w")) is work
    assert supervision._loop_gateway(current.run(tmp_path, "run-s")) is owned
    # A wake's one read goes through the engine its own run's custody names (``_publish``).
    ctx, reads = _delegate_ctx(tmp_path, monkeypatch), []
    scripted_reads(monkeypatch, lambda _ctx, rid, *_a, gateway=None: reads.append((rid, gateway)) or json.dumps(
        {"status": "no_progress", "run_id": rid}))
    publish(ctx, "run-w")
    publish(ctx, "run-s")
    monkeypatch.setattr(domain, "work_gateway", lambda *_a, **_kw: (_ for _ in ()).throw(
        ClaudexorUnavailable("daemon_unreachable", "down")))
    assert supervision._loop_gateway(current.run(tmp_path, "run-w")) is None
    monkeypatch.setattr(custody, "lookup", lambda *_a: (_ for _ in ()).throw(OSError("custody temporarily unreadable")))
    monkeypatch.setattr(supervision, "_loop_gateway", lambda *_a: pytest.fail("unreadable custody chose an engine"))
    publish(ctx, "run-u")  # the per-call reader recovers custody itself; no guessed engine
    assert reads == [("run-w", work), ("run-s", owned), ("run-u", None)]


# --- attached launches: stdio MCP servers, skill scripts and dependency builds run in the pinned executor ----------

FAKE_DOCKER = r"""import json, os, sys
args = sys.argv[1:]
with open(LOG, "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\n")
if args[:1] == ["inspect"]:  # the network check: the container's mode, as the test set it
    print(open(NETWORK).read().strip())
    sys.exit(0)
if args[:1] != ["exec"]:
    sys.exit(0)
index, workdir = 1, None
while args[index].startswith("-"):
    workdir = args[index + 1] if args[index] == "--workdir" else workdir
    index += 2 if args[index] in ("--env", "--workdir") else 1
# The container's own /tmp is a folder of the test's; the command runs here as the container would run it.
command = [part.replace("/tmp/ouroboros-exec-", CONTAINER_TMP + "/ouroboros-exec-")
           for part in args[index + 1:]]
os.chdir(workdir or "/")
os.environ["IN_FAKE_CONTAINER"] = "1"  # what runs through ``docker exec`` says so; a process the service started does not
if command[:1] == ["sh"] and os.path.exists("/bin/dash"):  # the reference container's /bin/sh: dash's POSIX rules, the
    command[0] = "/bin/dash"  # asynchronous list's stdin /dev/null before explicit redirections included
os.execvp(command[0], command)
"""


# The container's process facilities on a host without them (macOS): ``setsid`` makes the launched command the
# leader of its own session, as util-linux's does for a background job, and the observer reads ``ps`` with the
# same three-valued contract as the /proc observer (``docker_process_observer``): a group is ``exited`` only
# when no live process has the recorded leader's pid or pgid.
SETSID_SHIM = "import os, sys\ntry:\n    os.setsid()\nexcept OSError:\n    pass\nos.execvp(sys.argv[1], sys.argv[1:])\n"
PS_OBSERVER = r'''
_ouroboros_proc_observe() {
  _opid="$1"; _oscope="$2"
  case "$_opid" in ''|0|*[!0-9]*) printf unknown; return 2;; esac
  case "$_oscope" in leader|group) ;; *) printf unknown; return 2;; esac
  _orows=$(ps -A -o pid= -o pgid= -o stat= 2>/dev/null) || { printf unknown; return 2; }
  _osurvivors=""
  while read -r _op _og _os; do
    [ -n "$_op" ] || continue
    case "$_os" in Z*) continue;; esac
    if [ "$_op" = "$_opid" ] || { [ "$_oscope" = group ] && [ "$_og" = "$_opid" ]; }; then _osurvivors="$_osurvivors $_op"; fi
  done <<OUROBOROS_PS
$_orows
OUROBOROS_PS
  if [ -n "$_osurvivors" ]; then printf 'surviving process group members:%s\n' "$_osurvivors" >&2; printf running
  else printf exited; fi
}'''


def fake_docker(env, monkeypatch, network="bridge"):
    """A ``docker`` first on PATH that runs ``docker exec``'s command locally, as the work container would; a
    real CLI cannot answer instead (the PATH bootstrap is held and DOCKER_HOST names no daemon). It reads no
    variable of its own: the service starts its CLI with only the CLI's configuration. The container side keeps
    Linux's session and group semantics (``SETSID_SHIM``, ``PS_OBSERVER``), so settlement observes real groups."""
    import os
    import shutil
    import sys

    from ouroboros import platform_layer, workspace_executor

    bin_dir, container_tmp, data = env.root / "bin", env.root / "container-tmp", env.root / "data"
    for folder in (bin_dir, container_tmp, data):
        folder.mkdir()
    cli, log, mode = bin_dir / "docker", env.root / "docker.jsonl", env.root / "network-mode"
    mode.write_text(network, encoding="utf-8")
    cli.write_text(f"#!{sys.executable}\nLOG, CONTAINER_TMP, NETWORK = {str(log)!r}, {str(container_tmp)!r}, "
                   f"{str(mode)!r}\n" + FAKE_DOCKER, encoding="utf-8")
    cli.chmod(0o755)
    if not shutil.which("setsid"):
        (bin_dir / "setsid").write_text(f"#!{sys.executable}\n" + SETSID_SHIM, encoding="utf-8")
        (bin_dir / "setsid").chmod(0o755)
    if not os.path.isdir("/proc/self"):
        monkeypatch.setattr(workspace_executor, "_docker_proc_observer_function", lambda: PS_OBSERVER.strip())
    monkeypatch.setattr(platform_layer, "_PATH_BOOTSTRAPPED", True)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("DOCKER_HOST", f"unix://{env.root}/no-daemon.sock")
    monkeypatch.setattr(domain, "_data_root", lambda: data)
    assert shutil.which("docker") == str(cli)
    return SimpleNamespace(
        tmp=container_tmp, calls=lambda: [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else [],
        records=lambda: list((data / "state" / "workspace_executor_processes").glob("*.json")),
        network=lambda value: mode.write_text(value, encoding="utf-8"))


def attached(docker, needle):
    """The work-domain launches (``docker exec -i <container> sh -c <wrapper>``) whose command holds ``needle``."""
    return [call for call in docker.calls() if call[:2] == ["exec", "-i"] and needle in call[-1]]


def test_an_attached_launch_keeps_the_services_facts_and_refuses_what_the_domain_does_not_see(env, monkeypatch):
    import os

    from ouroboros.workspace_executor_paths import ManagedPathRefused

    assert domain.attached_launch(["server"], str(env.work), {"TOKEN": "t"}) is None  # unmanaged: nothing changes
    env.manage(section(env))
    own_bin, scope = env.work / "deps" / "bin", "OURO_TASK_SCOPE_" + "a" * 32
    launch = domain.attached_launch(["server", "--flag"], str(env.work), {
        "TOKEN": "secret", "PATH": f"{own_bin}{os.pathsep}{os.environ['PATH']}", "HOME": os.environ["HOME"],
        "TMPDIR": str(env.work / "tmp"), scope: "1", "API_BASE": "https://example.invalid/v1",
        "PIP_CONFIG_FILE": "/dev/null", "SHELL": "/bin/zsh"})
    assert launch.argv[:3] == ["docker", "exec", "-i"] and launch.argv[-4:-2] == ["ouro-work", "sh"]
    assert launch.argv[launch.argv.index("--workdir") + 1] == str(env.work)
    forwarded = {launch.env[launch.argv[i + 1]]: launch.argv[i + 1] for i, part in enumerate(launch.argv) if part == "--env"}
    # what the caller chose crosses; the service's own PATH, home and shell, and every scope marker, stay here
    assert sorted(forwarded) == sorted(["secret", str(own_bin), str(env.work / "tmp"), "https://example.invalid/v1",
                                        "/dev/null"])
    assert f'PATH="${{{forwarded[str(own_bin)]}}}:$PATH"' in launch.argv[-1]  # then the container's own search
    assert os.environ["HOME"] not in launch.argv[-1] and "server --flag" in launch.argv[-1]
    assert launch.env[scope] == "1" and launch.env["PATH"] == os.environ["PATH"]  # the service's CLI and census
    assert launch.pidfile.startswith("/tmp/ouroboros-exec-")
    for cwd, variables in ((str(env.root / "service-only"), {}), (str(env.work), {"CONFIG": str(env.root / "x")}),
                           (str(env.work), {"PYTHONPATH": f"{env.work}{os.pathsep}{env.root / 'lib'}"}),
                           (str(env.work), {"HOME": str(env.root / "home")}),
                           (str(env.work), {"PATH": f"{env.root / 'bin'}{os.pathsep}{os.environ['PATH']}"})):
        with pytest.raises(ManagedPathRefused, match="nothing runs on the service"):
            domain.attached_launch(["server"], cwd, variables)


def test_a_stdio_mcp_server_runs_in_the_pinned_executor_and_its_group_is_settled(env, monkeypatch):
    import sys

    from ouroboros import mcp_client

    pytest.importorskip("mcp")
    env.manage(section(env))
    docker = fake_docker(env, monkeypatch)
    workdir, witness = env.work / "project", env.work / "observed.jsonl"
    workdir.mkdir()
    script = env.work / "mcp_probe.py"
    script.write_text("""import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    with open(sys.argv[1], "a", encoding="utf-8") as witness:
        witness.write(json.dumps({"method": request.get("method"), "cwd": os.getcwd(),
                                  "token": os.environ.get("TOKEN")}) + "\\n")
    method = request.get("method")
    result = ({"protocolVersion": request["params"]["protocolVersion"], "capabilities": {"tools": {}},
               "serverInfo": {"name": "probe", "version": "1"}} if method == "initialize" else
              {"tools": [{"name": "probe", "description": "p", "inputSchema": {"type": "object"}}]}
              if method == "tools/list" else {})
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
""", encoding="utf-8")
    raw = {"id": "probe", "enabled": True, "transport": "stdio", "command": sys.executable,
           "args": [str(script), str(witness)], "cwd": str(workdir), "env_from_settings": {"TOKEN": "TEST_MCP_KEY"}}
    settings = {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 10, "TEST_MCP_KEY": "synthetic-token"}
    manager = mcp_client.MCPManager()
    manager.reconfigure({**settings, "MCP_SERVERS": [raw]})
    refreshed = manager.refresh_server("probe")
    assert refreshed["ok"] and refreshed["tool_count"] == 1, refreshed
    rows = [json.loads(line) for line in witness.read_text(encoding="utf-8").splitlines()]
    assert rows and all(row["cwd"] == str(workdir) and row["token"] == "synthetic-token" for row in rows)
    starts = [call for call in docker.calls() if call[:2] == ["exec", "-i"]]
    assert starts and all(call[call.index("--workdir") + 1] == str(workdir) and "ouro-work" in call for call in starts)
    assert all("synthetic-token" not in json.dumps(call) for call in docker.calls())  # values ride the CLI's env
    assert not docker.records() and not list(docker.tmp.glob("*.pid"))  # every group confirmed ended, then retired
    manager.reconfigure({**settings, "MCP_SERVERS": [{**raw, "id": "outside", "cwd": str(env.root)}]})
    before = len(docker.calls())
    refused = manager.refresh_server("outside")
    assert not refused["ok"] and "outside this deployment's execution domain" in refused["error"], refused
    assert len(docker.calls()) == before and not docker.records()  # refused before any start, here or there
    monkeypatch.setenv("PATH", str(env.root / "no-cli"))  # the service has no Docker CLI: the SDK's spawn fails
    manager.reconfigure({**settings, "MCP_SERVERS": [raw]})
    assert not manager.refresh_server("probe")["ok"]
    assert len(docker.calls()) == before and not docker.records()  # the never-started CLI's record is abandoned


def test_skill_scripts_and_dependency_builds_run_in_the_pinned_executor_or_not_at_all(env, monkeypatch):
    import os

    from ouroboros.marketplace import isolated_deps
    from ouroboros.tools import skill_exec
    from ouroboros.workspace_executor_paths import ManagedPathRefused

    env.manage(section(env))
    docker = fake_docker(env, monkeypatch)
    skill = env.work / "skills" / "probe"
    skill.mkdir(parents=True)
    inherited = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}
    rc, out, err, overflowed = skill_exec._run_skill_subprocess(
        ["sh", "-c", 'printf "%s|%s" "$PWD" "$TOKEN"'], cwd=str(skill), env={"TOKEN": "t-1", **inherited},
        timeout_sec=20, stdout_cap=4096, stderr_cap=4096)
    assert (rc, out.decode(), overflowed) == (0, f"{skill}|t-1", False), err
    build_env = isolated_deps._installer_env(skill / ".deps", ecosystem="python")  # the builds' own environment
    result = isolated_deps._run(["sh", "-c", 'echo "$HOME" > build.txt'], cwd=skill, env=build_env, timeout_sec=20)
    assert result == {"cmd": ["sh", "-c", "..."], "returncode": 0}  # the build's own argv, never the CLI's
    assert (skill / "build.txt").read_text(encoding="utf-8").strip() == str(skill / ".deps" / "home")
    starts = [call[-1] for call in docker.calls() if call[:2] == ["exec", "-i"]]
    # the script and the build, each once; the build's own folders (tmp, home, cache) are made there as the work
    assert sum('printf "%s|%s"' in shell for shell in starts) == 1 and sum("build.txt" in shell for shell in starts) == 1
    assert len(starts) == 5 and sum("mkdir" in shell for shell in starts) == 3, starts
    assert not docker.records() and not list(docker.tmp.glob("*.pid"))  # each group ended and its record retired
    before, marker = len(docker.calls()), env.root / "ran-on-service"
    with pytest.raises(PermissionError, match="nothing runs on the service"):  # the tool's pre-exec cause
        skill_exec._run_skill_subprocess(["touch", str(marker)], cwd=str(env.root), env={}, timeout_sec=5,
                                         stdout_cap=64, stderr_cap=64)
    with pytest.raises(ManagedPathRefused):
        isolated_deps._run(["touch", str(marker)], cwd=env.root, env={}, timeout_sec=5)
    with pytest.raises(ManagedPathRefused, match="PIP_CACHE_DIR"):
        isolated_deps._run(["true"], cwd=skill, env={"PIP_CACHE_DIR": str(env.root / "cache")}, timeout_sec=5)
    assert not marker.exists() and len(docker.calls()) == before and not docker.records()
    from ouroboros.tools import shell_process

    monkeypatch.setattr(shell_process, "_panic_requested", True)  # admission closed: refused before any Popen
    with pytest.raises(shell_process.CommandAdmissionClosed):
        skill_exec._run_skill_subprocess(["true"], cwd=str(skill), env={}, timeout_sec=5, stdout_cap=64, stderr_cap=64)
    with pytest.raises(shell_process.CommandAdmissionClosed):
        isolated_deps._run(["true"], cwd=skill, env={}, timeout_sec=5)
    assert len(docker.calls()) == before and not docker.records()  # each adopted record abandoned, nothing ran


def _reviewed_skill(env, monkeypatch, name, script, *, runtime="python3", entry="run.py", skills=None):
    """A reviewed, enabled script skill in a skills checkout (inside the work mapping unless ``skills`` names another
    folder); the service keeps its review and enablement in its own drive."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state
    from ouroboros.tools.registry import ToolContext

    skills, drive = skills or env.work / "skills", env.root / "drive"
    (skills / name / "scripts").mkdir(parents=True)
    (skills / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Probe.\nversion: 0.1.0\ntype: script\nruntime: {runtime}\ntimeout_sec: 30\n"
        f"scripts:\n  - name: {entry}\n    description: Probe.\n---\n# body\n", encoding="utf-8")
    (skills / name / "scripts" / entry).write_text(script, encoding="utf-8")
    (env.root / "repo").mkdir(exist_ok=True)
    drive.mkdir(exist_ok=True)
    save_enabled(drive, name, True)
    save_review_state(drive, name, SkillReviewState(status="pass", content_hash=compute_content_hash(skills / name)))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    return ToolContext(repo_dir=env.root / "repo", drive_root=drive), drive


def test_a_reviewed_skill_keeps_its_state_in_the_work_domain_across_runs(env, monkeypatch):
    """The real tool, managed: the script runs in the pinned executor, writes its state into the administrator's work
    root and reads it back on its next run; the service keeps its own records apart and spawns only its CLI."""
    from ouroboros.tools import skill_exec

    ctx, drive = _reviewed_skill(env, monkeypatch, "counter", (
        "import os, pathlib\n"
        "state = pathlib.Path(os.environ['OUROBOROS_SKILL_STATE_DIR']) / 'count'\n"
        "count = int(state.read_text(encoding='utf-8')) + 1 if state.exists() else 1\n"
        "state.write_text(str(count))\n"
        "print(count, state.parent, os.environ.get('IN_FAKE_CONTAINER'))\n"))
    root = env.work / "skill-state"
    env.manage(section(env, state_root=str(root)))
    docker = fake_docker(env, monkeypatch)
    runs = [json.loads(skill_exec._handle_skill_exec(ctx, skill="counter", script="scripts/run.py", args=[]))
            for _ in range(2)]
    assert [(run["exit_code"], run["stdout"].split()) for run in runs] == [
        (0, ["1", str(root / "counter"), "1"]), (0, ["2", str(root / "counter"), "1"])]  # ran in the container
    assert (root / "counter" / "count").read_text(encoding="utf-8") == "2"
    assert not (drive / "state" / "skills" / "counter" / "count").exists()  # the service's records: untouched
    made = [call for call in docker.calls() if call[2:4] == ["mkdir", "-p"]]
    assert made == [["exec", "ouro-work", "mkdir", "-p", "--", str(root / "counter")]] * 2  # made there, as its user
    assert len([call for call in docker.calls() if call[:2] == ["exec", "-i"]]) == 2 and not docker.records()


def test_a_managed_skill_without_its_work_folders_is_refused_before_anything_starts(env, monkeypatch, tmp_path):
    """No ``skill_state_root``, or an installed skill under the data root with no ``staging_root`` to copy its
    reviewed payload into (``work_skill_payloads``): refused, and nothing runs, here or in the container."""
    from ouroboros.tools import skill_exec
    from ouroboros.workspace_executor_paths import ManagedPathRefused

    marker = env.root / "ran"
    script = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('ran')\n"
    ctx, _drive = _reviewed_skill(env, monkeypatch, "inside", script)
    env.manage(section(env))
    docker = fake_docker(env, monkeypatch)
    with pytest.raises(ManagedPathRefused, match="names no skill_state_root"):  # the registry shows it as TOOL_ERROR
        skill_exec._handle_skill_exec(ctx, skill="inside", script="scripts/run.py", args=[])
    env.manage(section(env, state_root=str(env.work / "skill-state")))
    ctx, _drive = _reviewed_skill(env, monkeypatch, "outside", script, skills=tmp_path / "installed-skills")
    answer = skill_exec._handle_skill_exec(ctx, skill="outside", script="scripts/run.py", args=[])
    assert answer.startswith("⚠️ SKILL_EXEC_ERROR: OS error running skill: ") and "names no staging_root" in answer \
        and "nothing runs on the service instead" in answer, answer
    assert not marker.exists() and not [call for call in docker.calls() if call[:2] == ["exec", "-i"]]
    assert not docker.records()


def test_env_deno_permission_and_go_build_use_the_one_work_state_path(env, monkeypatch):
    from ouroboros.tools import skill_exec

    root = env.work / "skill-state"
    env.manage(section(env, state_root=str(root)))
    docker = fake_docker(env, monkeypatch)
    seen = []
    monkeypatch.setattr(skill_exec, "_run_skill_subprocess", lambda cmd, **kw: seen.append(
        (list(cmd), dict(kw["env"]))) or (0, b"", b"", False))
    monkeypatch.setattr(skill_exec, "_resolve_runtime_binary", lambda runtime: (f"/opt/{runtime}/bin/{runtime}", ""))
    ctx, _drive = _reviewed_skill(env, monkeypatch, "denoprobe", "console.log(1)\n", runtime="deno", entry="run.ts")
    skill_exec._handle_skill_exec(ctx, skill="denoprobe", script="scripts/run.ts", args=[])
    state = str(root / "denoprobe")
    (deno, deno_env), = seen
    assert f"--allow-write={state}" in deno and deno_env["OUROBOROS_SKILL_STATE_DIR"] == state
    ctx, _drive = _reviewed_skill(env, monkeypatch, "goprobe", "package main\nfunc main() {}\n", runtime="go",
                                  entry="main.go")
    skill_exec._handle_skill_exec(ctx, skill="goprobe", script="scripts/main.go", args=["x"])
    state = str(root / "goprobe")
    (build, build_env), (run, _run_env) = seen[1:]
    out = build[build.index("-o") + 1]
    folder = out.rsplit("/", 1)[0]
    assert build[:2] == ["/opt/go/bin/go", "build"] and folder.startswith(f"{state}/go-exec-") and run == [out, "x"]
    assert (build_env["GOCACHE"], build_env["GOPATH"], build_env["OUROBOROS_SKILL_STATE_DIR"]) == (
        f"{state}/go-cache", f"{state}/go", state)
    calls = docker.calls()
    assert ["exec", "ouro-work", "mkdir", "--", folder] in calls and ["exec", "ouro-work", "rm", "-rf", "--", folder] in calls
    assert not pathlib.Path(folder).exists()  # made and removed in the work domain (here, by the fake CLI)


def test_network_none_is_checked_before_every_attached_start(env, monkeypatch):
    """``network: none`` holds only while the container has none: the executor's own check runs before each attached
    start; a container with a network refuses with nothing started, and an unchecked one never passes."""
    import sys

    from ouroboros import mcp_client
    from ouroboros.tools import skill_exec

    env.manage(section(env, network="none"))
    docker = fake_docker(env, monkeypatch, network="bridge")
    with pytest.raises(RuntimeError, match="requires Docker NetworkMode=none") as refused:
        domain.attached_launch(["server"], str(env.work), {})
    assert refused.value.process_not_started is True
    with pytest.raises(PermissionError, match="NetworkMode=none"):  # the skill tool's pre-exec cause
        skill_exec._run_skill_subprocess(["true"], cwd=str(env.work), env={}, timeout_sec=5, stdout_cap=64,
                                         stderr_cap=64)
    if __import__("importlib").util.find_spec("mcp") is not None:
        manager = mcp_client.MCPManager()
        manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 5, "MCP_SERVERS": [
            {"id": "netprobe", "enabled": True, "transport": "stdio", "command": sys.executable, "args": ["-V"],
             "cwd": str(env.work)}]})
        assert "NetworkMode=none" in manager.refresh_server("netprobe")["error"]
    assert {call[0] for call in docker.calls()} == {"inspect"} and not docker.records()  # nothing was started
    docker.network("none")
    assert domain.attached_launch(["server"], str(env.work), {}) is not None
    monkeypatch.setattr("ouroboros.workspace_executor._assert_docker_network_none", lambda _name: (
        _ for _ in ()).throw(subprocess.TimeoutExpired("docker", 10)))
    with pytest.raises(RuntimeError, match="could not be checked"):
        domain.attached_launch(["server"], str(env.work), {})
    env.manage(section(env))  # ``host``: the executor's word for networking allowed, which it does not check
    assert domain.attached_launch(["server"], str(env.work), {}) is not None


def test_folder_commands_are_the_works_own_and_check_the_network_first(env, monkeypatch):
    """The commands that make a skill's state and build folders resolve inside the work domain: ``network: none`` is
    checked before them as before any start, and one the CLI did not confirm is never reported as unstarted."""
    root, drive = env.work / "skill-state", env.root / "drive"
    env.manage(section(env, network="none", state_root=str(root)))
    docker = fake_docker(env, monkeypatch, network="bridge")
    with pytest.raises(RuntimeError, match="requires Docker NetworkMode=none") as refused:
        domain.skill_state_dir(drive, "counter")
    assert refused.value.process_not_started is True
    with pytest.raises(RuntimeError, match="requires Docker NetworkMode=none"):
        with domain.build_dir(root / "counter", "go-exec-"):
            pytest.fail("no build folder is made in a container that has a network")
    assert [call[0] for call in docker.calls()] == ["inspect", "inspect"] and not root.exists()  # no exec at all
    docker.network("none")
    assert domain.skill_state_dir(drive, "counter") == root / "counter" and (root / "counter").is_dir()
    assert [call[:3] for call in docker.calls()[2:]] == [["inspect", "-f", "{{.HostConfig.NetworkMode}}"],
                                                         ["exec", "ouro-work", "mkdir"]]
    (root / "blocked").write_text("a file where the folder would go", encoding="utf-8")
    env.manage(section(env, state_root=str(root)))  # ``host``: the network is not checked
    with pytest.raises(RuntimeError, match="did not confirm mkdir .*it may have run there") as unconfirmed:
        domain.skill_state_dir(drive, "blocked")
    assert not hasattr(unconfirmed.value, "process_not_started")


def test_a_work_result_is_told_while_the_owned_engine_is_down(env, tmp_path, monkeypatch):
    from ouroboros import delegate_result_taken as taken
    from ouroboros import review_thread_retention as rt
    from ouroboros.task_store import read_document

    env.manage(section(env))
    task, root = "task-down", tmp_path / "data"
    root.mkdir()
    seed_result(root, task, {"task_id": task, "status": "completed", "ts": "2026-11-08T00:00:00+00:00"})
    common = dict(task_id=task, root_task_id=task, settled=True, terminal_state="succeeded", target_root="/owner/repo",
                  access="workspace_write", mode="agent", patch_captured=True)
    runs = {"run-w": custody.RunCustody(run_id="run-w", snapshot_id="snap-w", execution_root="/copies/run-w",
                                        engine_ref=work_ref(env), daemon_store_id=STORE, **common),
            "run-s": custody.RunCustody(run_id="run-s", snapshot_id="snap-s", execution_root="/copies/run-s",
                                        daemon_store_id="store-s", **common)}
    owned_runs = {"value": runs}
    monkeypatch.setattr(current, "runs", lambda _root, _where="", params=(): owned_runs["value"])
    monkeypatch.setattr(rt, "tree_hold", lambda _root, _task, _doc, *, live, census: ("", owned_runs["value"]))
    told = []
    work = SimpleNamespace(store_identity=STORE, engine_ref=work_ref(env), close=lambda: None,
                           run_result_taken_supported=lambda: True,
                           take_run_result=lambda rid, idempotency_key: told.append(rid) or {
                               "accepted": True, "status": "taken"})
    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: work)

    def a_pass():  # the owned engine cannot be opened at all
        report = {"holds": {}, "observed": []}
        result = SimpleNamespace(root=root, live=lambda _task: False, census=lambda: set(), report=report,
                                 stopped=False, engine=lambda: None, stop=lambda: False)
        result.hold = lambda reason: report["holds"].__setitem__(reason, report["holds"].get(reason, 0) + 1)
        result.work = domain.EngineClients(result.engine)
        return result

    down = a_pass()
    assert taken.tell_engine(down, task) == rt.WAIT  # the owned engine's run waits, named; the work engine's is told
    assert told == ["run-w"] and down.report["holds"] == {"result_engine_unavailable": 1}
    assert set(read_document(root, task)[taken.RECORDS]) == {"run-w"}
    owned_runs["value"] = {"run-w": runs["run-w"]}
    assert taken.tell_engine(a_pass(), task) == rt.DONE and told == ["run-w"]  # recorded once, never told again


def test_thread_retention_asks_each_objects_own_engine(env, tmp_path, monkeypatch):
    """Managed, a release or thread waits only for the engine holding it: the owned engine's absence holds nothing
    of the work engine's, and what it does hold is named."""
    from ouroboros import claudexor_daemon
    from ouroboros import review_thread_retention as rt

    env.manage(section(env))
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: (_ for _ in ()).throw(
        ClaudexorUnavailable("daemon_unreachable", "down")))
    work = SimpleNamespace(store_identity=STORE, engine_ref=work_ref(env), close=lambda: None,
                           thread_retirement_supported=lambda: True)
    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: work)
    released = []
    monkeypatch.setattr(rt, "drive_release", lambda _root, _task, record, engine, *, may_send: released.append(
        (record["thread_id"], engine)) or "accepted")
    monkeypatch.setattr(rt, "seat_threads", lambda *_a, **_kw: ({}, True))
    monkeypatch.setattr(rt, "tree_hold", lambda *_a, **_kw: ("", {}))
    monkeypatch.setattr("ouroboros.owner_keep.is_kept", lambda *_a: False)
    document = {rt.RELEASES: {"thread-w": {"state": "bound", "thread_id": "thread-w", "store_id": STORE}}}
    monkeypatch.setattr("ouroboros.task_store.read_document", lambda _root, _task: document)

    def a_pass():
        return rt._Pass(tmp_path, "2026-11-08T00:00:00+00:00", engine=None, live=lambda _task: False,
                        census=lambda: set(), stop=lambda: False)

    first = a_pass()
    assert first.task("task-r") == rt.DONE and released == [("thread-w", work)] and first.report["holds"] == {}
    document[rt.RELEASES]["thread-s"] = {"state": "bound", "thread_id": "thread-s", "store_id": "store-s"}
    released.clear()
    second = a_pass()
    assert second.task("task-r") == rt.WAIT and released == [("thread-w", work)]
    assert second.report["holds"] == {"engine_unavailable": 1}  # the owned engine's own release waits, named
    unreachable = a_pass()
    monkeypatch.setattr(domain, "work_gateway", lambda r, **_kw: (_ for _ in ()).throw(
        ClaudexorUnavailable("daemon_unreachable", "down")))
    host = SimpleNamespace(started_ts="2026-11-09T00:00:00+00:00", engine_ref=work_ref(env), run_id="run-w")
    assert unreachable.thread("task-r", "thread-x", [], [host], {}) == rt.WAIT
    assert unreachable.report["holds"] == {"engine_unavailable": 1}


# --- a work repository's file writes and status/diff run AS the work domain ----------------------------

def _work_git_repo_with_clean_filter(env):
    """The active workspace is a work git repo whose ``*.txt`` clean filter records whether it ran in the
    container (``IN_FAKE_CONTAINER``) and passes content through; a committed file is edited in the worktree,
    so status and diff must run the filter to compare it. Returns ``(repo, witness)``; the witness is cleared
    after setup (``git add`` ran the filter on the service), so only the measured run writes to it."""
    import os
    import shlex
    import subprocess as sp

    repo, witness, script = env.work, env.root / "filter-ran", env.root / "mark-clean.sh"
    script.write_text("#!/bin/sh\nprintf '%s\\n' \"${IN_FAKE_CONTAINER:-service}\" >> "
                      + shlex.quote(str(witness)) + "\ncat\n", encoding="utf-8")
    script.chmod(0o755)
    env_git = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
               "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

    def git(*args):
        sp.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env_git)

    git("init", "-q")
    git("config", "filter.mark.clean", str(script))
    git("config", "filter.mark.smudge", "cat")
    (repo / ".gitattributes").write_text("*.txt filter=mark\n", encoding="utf-8")
    (repo / "note.txt").write_text("one\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "base")
    (repo / "note.txt").write_text("one\ntwo\n", encoding="utf-8")  # a worktree edit status/diff must clean
    witness.unlink(missing_ok=True)
    return repo, witness


def test_a_work_repo_file_write_lands_as_the_work_domain_with_bytes_on_stdin(env, monkeypatch):
    """A write to a work-domain file runs in the pinned container, its bytes on stdin and never the shell, so an
    ancestor swap could reach only the work domain and the service never opens that pathname for writing."""
    env.manage(section(env))
    docker = fake_docker(env, monkeypatch)
    registry = _registry(env, env.work)
    secret = "print('to stdin, never argv')\n"
    text = registry.execute("write_file", {"root": "active_workspace", "path": "pkg/app.py", "content": secret})
    assert (env.work / "pkg" / "app.py").read_text(encoding="utf-8") == secret, text
    writes = attached(docker, "mv -f")
    assert len(writes) == 1 and "mkdir -p" in writes[0][-1]  # one container overwrite, its folder made there
    assert secret.strip() not in " ".join(writes[0])  # the employee's content rode stdin, never the shell argv
    registry.execute("edit_text", {"root": "active_workspace", "path": "pkg/app.py",
                                    "old_str": "to stdin, never argv", "new_str": "edited in the work domain"})
    assert "edited in the work domain" in (env.work / "pkg" / "app.py").read_text(encoding="utf-8")
    registry.execute("write_file", {"root": "active_workspace", "path": "pkg/app.py",
                                    "content": "# tail\n", "mode": "append"})
    assert (env.work / "pkg" / "app.py").read_text(encoding="utf-8").endswith("# tail\n")
    assert attached(docker, "cat >>")  # the append grew it there


def test_a_work_tree_status_and_diff_run_their_filters_in_the_work_container(env, monkeypatch):
    """status and diff over a work repository run its configured clean filter — a supported, tested capability —
    IN the container, so an employee's filter never executes with the service's authority."""
    from ouroboros.tools import git_vcs_ops

    env.manage(section(env))
    repo, witness = _work_git_repo_with_clean_filter(env)
    docker = fake_docker(env, monkeypatch)
    registry = _registry(env, repo)
    binding = git_vcs_ops._git()._vcs_binding(registry._ctx, None, root="active_workspace", path=".")
    diff = git_vcs_ops._git_diff(registry._ctx, _resolved_binding=binding)
    assert "two" in diff  # the worktree change is seen
    assert witness.exists() and set(witness.read_text(encoding="utf-8").split()) == {"1"}  # the filter ran, and only in the container
    ran = attached(docker, "git diff")
    assert ran and all(str(repo) in call[-1] for call in ran)  # entered by the container, in the repository
    # status reports M from the stat cache without cleaning, but it too runs in the container, so any filter it
    # does invoke runs there and never on the service.
    status = git_vcs_ops._git_status(registry._ctx, _resolved_binding=binding)
    assert "note.txt" in status and attached(docker, "git status")
    assert not docker.records()  # every launch's group confirmed ended and its record retired


def test_unmanaged_work_file_write_stays_on_the_service(env, monkeypatch):
    """Without a policy the write is byte-identical to before: it lands directly on the service, no container start."""
    docker = fake_docker(env, monkeypatch)  # a CLI is on PATH, but nothing here should call it
    registry = _registry(env, env.work)
    registry.execute("write_file", {"root": "active_workspace", "path": "a.py", "content": "x = 1\n"})
    assert (env.work / "a.py").read_text(encoding="utf-8") == "x = 1\n"
    assert not [call for call in docker.calls() if call[:1] == ["exec"]]  # the service wrote it, no work container


def test_apply_patch_writes_and_deletes_run_as_the_work_domain(env, monkeypatch):
    """apply_patch's multi-file write and delete over a work tree run in the pinned container, so neither an
    added file's write nor a delete can be redirected by a path race out of the work domain."""
    env.manage(section(env))
    (env.work / "old.py").write_text("gone = True\n", encoding="utf-8")
    docker = fake_docker(env, monkeypatch)
    registry = _registry(env, env.work)
    patch = ("*** Begin Patch\n"
             "*** Add File: pkg/created.py\n"
             "+print('added as the work domain')\n"
             "*** Delete File: old.py\n"
             "*** End Patch\n")
    text = registry.execute("apply_patch", {"patch": patch})
    assert (env.work / "pkg" / "created.py").read_text(encoding="utf-8") == "print('added as the work domain')\n", text
    assert not (env.work / "old.py").exists(), text
    assert attached(docker, "mv -f")  # the add landed in the container
    assert [call for call in attached(docker, "rm -f") if "old.py" in call[-1]]  # and the delete ran there
    assert not docker.records()


# --- custody of an attached launch: the group, not only its leader; an ambiguous start is settled -------------

def test_an_attached_launch_keeps_custody_of_its_group_until_every_member_ended(env, monkeypatch):
    """The leader exits while a member of its group still runs: settlement does not take the leader's end for the
    group's. The executor's bounded group stop ends the member and only then is the record retired, so nothing the
    launch started outlives its custody (critic probe attached-lifetime)."""
    import time

    from ouroboros.tools import skill_exec

    env.manage(section(env))
    docker = fake_docker(env, monkeypatch)
    skill, later = env.work / "skills" / "probe", env.work / "helper-ran-after-settle"
    skill.mkdir(parents=True)
    rc, _out, err, _ = skill_exec._run_skill_subprocess(  # the helper detaches its streams, so the CLI ends first
        ["sh", "-c", f"(sleep 3; touch {later}) </dev/null >/dev/null 2>&1 & exit 0"], cwd=str(skill), env={},
        timeout_sec=20, stdout_cap=4096, stderr_cap=4096)
    assert rc == 0, err
    assert not docker.records() and not list(docker.tmp.glob("*.pid"))  # retired only after the group ended
    time.sleep(3.5)
    assert not later.exists()  # the member was stopped at settlement, never left running outside custody


@pytest.mark.parametrize("uncertain", [False, True])
def test_skill_bookkeeping_failure_stops_the_backend_or_retains_uncertainty(env, monkeypatch, uncertain):
    from ouroboros.tools import skill_exec
    from ouroboros.work_domain import WorkDomainUnsettled

    env.manage(section(env))
    docker = fake_docker(env, monkeypatch)
    began, later = env.work / "began", env.work / "later"
    real_settle, pending = domain.AttachedLaunch.settle, []
    if uncertain:
        def unconfirmed(launch):
            pending.append(launch)
            return False
        monkeypatch.setattr(domain.AttachedLaunch, "settle", unconfirmed)
    def failed_bookkeeping():
        deadline = time.monotonic() + 5
        while not began.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert began.exists(), "the backend must actually start before the injected bookkeeping error"
        raise OSError("injected accounting failure")
    try:
        with pytest.raises(WorkDomainUnsettled if uncertain else OSError) as failure:
            skill_exec._run_skill_subprocess(["sh", "-c", f"touch {began}; sleep 3; touch {later}"],
                cwd=str(env.work), env={}, timeout_sec=10, stdout_cap=4096, stderr_cap=4096,
                on_spawn=failed_bookkeeping)
        assert bool(docker.records()) == uncertain
        if uncertain:
            assert "unconfirmed" in str(failure.value) and "accounting failure" in str(failure.value.__context__)
    finally:
        for launch in pending:
            assert real_settle(launch), "restore the actual bounded cleanup after the simulated unconfirmed read"
    time.sleep(3.2)
    assert not later.exists() and not docker.records()


def test_an_ambiguous_mcp_startup_failure_settles_the_container_side_and_never_abandons_it(env, monkeypatch):
    """The SDK can fail attaching its pipes AFTER the Docker CLI was spawned: that start is settled, so the
    container side is confirmed ended or stays recorded. Only the service's own proof that it has no CLI to start
    abandons, before anything is adopted (critic probe mcp_startup_oserror)."""
    import asyncio
    import contextlib
    import sys

    from ouroboros import mcp_client

    env.manage(section(env))
    docker = fake_docker(env, monkeypatch)
    monkeypatch.setattr(mcp_client, "_redacted_sdk_logs", lambda *_a: contextlib.nullcontext())
    monkeypatch.setattr(mcp_client, "_stderr_diagnostic", lambda *_a: "")
    marker = env.work / "server-ran"

    def launched():
        launch = domain.attached_launch([sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ran')"],
                                        str(env.work), {})
        calls = []
        for name in ("adopt", "abandon", "settle"):
            original = getattr(launch, name)
            setattr(launch, name, lambda *a, _o=original, _n=name, **k: calls.append(_n) or _o(*a, **k))
        return launch, calls

    @contextlib.asynccontextmanager
    async def spawned_then_failed(params, errlog):
        process = await asyncio.create_subprocess_exec(params.command, *params.args, env=params.env,
                                                       stdin=asyncio.subprocess.PIPE,
                                                       stdout=asyncio.subprocess.PIPE, stderr=errlog)
        await process.wait()  # the server ran in the container before the SDK failed
        raise OSError("injected pipe attachment failure after spawn")
        yield  # pragma: no cover

    monkeypatch.setattr(mcp_client, "stdio_client", spawned_then_failed)

    async def start(launch):
        params = SimpleNamespace(command=launch.argv[0], args=launch.argv[1:], env=launch.env)
        async with mcp_client._stdio_with_diagnostics(params, SimpleNamespace(id="probe"), launch):
            pytest.fail("the injected start never yields")

    launch, calls = launched()
    with pytest.raises(OSError, match="injected pipe attachment"):
        asyncio.run(start(launch))
    assert marker.exists() and calls == ["adopt", "settle"]  # it ran, so it was settled, never abandoned
    assert not docker.records()  # settlement confirmed the container side ended and retired its record
    marker.unlink()
    launch, calls = launched()
    before = len(docker.calls())
    launch.env["PATH"] = str(env.root / "no-cli")  # positive pre-spawn proof: the service has no CLI to start
    with pytest.raises(FileNotFoundError) as missing:
        asyncio.run(start(launch))
    assert missing.value.process_not_started is True and missing.value.mcp_unsent_joined is True
    assert calls == [] and not marker.exists() and len(docker.calls()) == before and not docker.records()

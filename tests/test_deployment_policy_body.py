"""A managed deployment's serving program is its administrator's: every supported door that would replace it refuses.

Each door is proved in pairs at its real effect owner: managed (a policy file named by the launch) refuses with the
typed ``managed_by_deployment`` before any lock, fetch, writer fence, write, subprocess or restart, and personal OSS
(no policy) reaches the same owner's next step as before. Work that never touches the serving checkout or
interpreter (a bound body candidate, another workspace, an already-installed runtime) stays available managed.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests.body_candidate_support import git, isolate, make_ctx, make_serving
from tests.test_body_adoption import Scene  # real cold entries over the real package-init hook and switch helper
from tests.test_git_bootstrap_receipt import bootstrap  # noqa: F401 - the shared real-Git restart fixture
from tests.test_macos_source_entry import FrozenContinues, _lifecycle, launch  # noqa: F401 - the real lifecycle loop

MANAGED = "managed_by_deployment"


@pytest.fixture
def policy(tmp_path, monkeypatch):
    """``policy(True)`` names a policy file the way a deployment's launch does; ``policy(False)`` leaves none."""
    from ouroboros import deployment_policy

    def name(managed: bool) -> None:
        monkeypatch.delenv(deployment_policy.GENERATION_ENV, raising=False)
        if managed:
            path = tmp_path / "admin" / "deployment-policy.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"version": 1, "access": {"max": "pro"}}), encoding="utf-8")
            monkeypatch.setenv(deployment_policy.POLICY_ENV, str(path))
        else:
            monkeypatch.delenv(deployment_policy.POLICY_ENV, raising=False)
        deployment_policy.reset_for_tests()

    yield name
    deployment_policy.reset_for_tests()


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
@pytest.mark.parametrize("managed", [True, False])
def test_files_routes_keep_the_policy_and_its_aliases_out_of_employee_documents(policy, tmp_path, monkeypatch, alias, managed):
    """A Files root may contain the administrator's policy. Reading/downloading or changing it through these
    supported doors refuses, including links; an ordinary file and a file no policy names keep their behavior."""
    from ouroboros.gateway import files

    policy(True)
    source = tmp_path / "admin" / "deployment-policy.json"
    target = source
    if alias != "direct":
        target = tmp_path / f"{alias}.json"
        if alias == "symlink":
            target.symlink_to(source)
        else:
            os.link(source, target)
    before = source.read_bytes()
    if not managed:
        policy(False)
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(tmp_path))
    ordinary, destination = tmp_path / "notes.txt", tmp_path / "destination"
    ordinary.write_text("ordinary work", encoding="utf-8")
    destination.mkdir()
    routes = [Route(f"/files/{name}", getattr(files, f"api_files_{name}"), methods=[method])
              for name, method in (("read", "GET"), ("content", "GET"), ("download", "GET"),
                                   ("write", "POST"), ("delete", "POST"), ("transfer", "POST"), ("upload", "POST"))]
    rel = target.relative_to(tmp_path).as_posix()
    with TestClient(Starlette(routes=routes)) as client:
        for route in ("read", "content", "download"):
            response = client.get(f"/files/{route}", params={"path": rel})
            assert response.status_code == (403 if managed else 200), (route, response.text)
            assert client.get(f"/files/{route}", params={"path": "notes.txt"}).status_code == 200
        if managed:
            for route, body in (("write", {"path": rel, "content": "changed"}), ("delete", {"path": rel}),
                                ("transfer", {"source_path": rel, "destination_dir": "destination", "mode": "copy"}),
                                ("transfer", {"source_path": rel, "destination_dir": "destination", "mode": "move"}),
                                ("delete", {"path": "admin"})):
                response = client.post(f"/files/{route}", json=body)
                assert response.status_code == 403, (route, response.text)
            response = client.post("/files/upload", data={"path": str(target.parent.relative_to(tmp_path))},
                                   files={"file": (target.name, b"changed", "application/json")})
            assert response.status_code == 403, response.text
        assert client.post("/files/write", json={"path": "notes.txt", "content": "still editable"}).status_code == 200
    assert source.read_bytes() == before and target.exists() and not list(destination.iterdir())


def _calls(monkeypatch, target: str, result=None):
    """Replace ``target`` with a recorder answering ``result``; returns the list of calls."""
    seen: list = []

    def record(*args, **kwargs):
        seen.append((args, kwargs))
        return result() if callable(result) else result

    monkeypatch.setattr(target, record)
    return seen


# --- control doors ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("managed", [True, False])
def test_update_apply_refuses_before_reading_the_request(policy, monkeypatch, managed):
    from ouroboros.gateway import control

    policy(managed)
    validated = _calls(monkeypatch, "ouroboros.gateway.schema.validate_ingress", lambda: ["strategy is required"])
    app = Starlette(routes=[Route("/api/update/apply", control.api_update_apply, methods=["POST"])])
    with TestClient(app) as client:
        answer = client.post("/api/update/apply", json={})
    if managed:
        assert answer.status_code == 403 and answer.json()["code"] == MANAGED, answer.text
        assert validated == [], "no plan, fetch or fence was reached"
    else:
        assert answer.status_code == 400 and validated, answer.text  # the ordinary path, unchanged


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize("door", ["ui_rollback", "vcs_rollback"])
def test_restore_refuses_before_the_lock_and_the_writer_fence(policy, monkeypatch, managed, door):
    from starlette.responses import JSONResponse

    from ouroboros.gateway import control

    policy(managed)
    lock = _calls(monkeypatch, "ouroboros.gateway.control._index_lock_precondition",
                  lambda: JSONResponse({"error": "stop here"}, status_code=409))
    restarts: list = []
    status, payload = control.rollback_through_door("HEAD~1", door=door, request_restart=lambda: restarts.append(1))
    if managed:
        assert (status, payload["code"]) == (403, MANAGED) and lock == [] and restarts == []
    else:
        assert status == 409 and len(lock) == 1 and restarts == []


@pytest.mark.parametrize("managed", [True, False])
def test_saving_a_recovery_point_refuses_on_both_doors(policy, monkeypatch, managed):
    from starlette.responses import JSONResponse

    from ouroboros.gateway import control
    from supervisor import events_runtime_controls as events

    policy(managed)
    locks = _calls(monkeypatch, "ouroboros.gateway.control._acquire_repo_mutation_lock",
                   lambda: (None, JSONResponse({"error": "stop here"}, status_code=409)))
    app = Starlette(routes=[Route("/api/git/promote", control.api_git_promote, methods=["POST"])])
    with TestClient(app) as client:
        answer = client.post("/api/git/promote")
    promoted = _calls(monkeypatch, "supervisor.git_ops.promote_branch_exact", (False, {"error": "stop here"}))
    for name in ("acquire_update_lock", "release_update_lock", "active_update_tx"):
        monkeypatch.setattr(f"supervisor.update_merge.{name}", lambda *_a, **_k: None)
    notices: list = []
    ctx = SimpleNamespace(load_state=lambda: {"owner_chat_id": 7}, BRANCH_DEV="ouroboros", BRANCH_STABLE="ouroboros-stable",
                          REPO_DIR=".", send_with_budget=lambda chat, text, **_kw: notices.append(text))
    events._handle_promote_to_stable({}, ctx)  # the agent's promote_to_stable
    if managed:
        assert answer.status_code == 403 and answer.json()["code"] == MANAGED and locks == []
        assert promoted == [] and "refused" in notices[0] and "administrator" in notices[0]
    else:
        assert answer.status_code == 409 and len(locks) == 1
        assert len(promoted) == 1 and "refused" not in notices[0]


# --- adoption and the restart seam -----------------------------------------------------------------------

@pytest.mark.parametrize("managed", [True, False])
def test_adoption_is_refused_before_any_handoff(policy, monkeypatch, tmp_path, managed):
    from ouroboros import body_adoption

    policy(managed)
    reached = _calls(monkeypatch, "ouroboros.body_adoption._git",
                     lambda: (_ for _ in ()).throw(RuntimeError("reached the checks")))
    row = {"repo_dir": str(tmp_path / "serving"), "path": str(tmp_path / "candidate"), "base_sha": "0" * 40}
    if managed:
        with pytest.raises(body_adoption.AdoptionRefused) as refused:
            body_adoption.authorize_row(tmp_path, row, "f" * 40, reason="agent_restart")
        assert refused.value.code == MANAGED and reached == []
    else:
        with pytest.raises(RuntimeError, match="reached the checks"):
            body_adoption.authorize_row(tmp_path, row, "f" * 40, reason="agent_restart")


@pytest.mark.parametrize("managed", [True, False])
def test_a_handoff_authorized_before_management_is_never_armed(policy, monkeypatch, tmp_path, managed):
    from ouroboros import body_adoption, body_switch

    policy(managed)
    serving = make_serving(tmp_path)
    handoff = {"id": "a1", "phase": "authorized", "restart_bound": True, "repo_dir": str(serving)}
    monkeypatch.setattr(body_adoption, "read", lambda _data: dict(handoff))
    monkeypatch.setattr(body_adoption, "_managed_update_active", lambda _data: False)
    monkeypatch.setattr(body_adoption, "_log", lambda *_a, **_k: None)
    monkeypatch.setattr(body_switch, "installation_panic", lambda _data: False)
    monkeypatch.setattr(body_switch, "record", lambda *_a, **_k: None)
    phase = body_adoption._arm_locked(tmp_path, worker_exits={"unconfirmed": []}, live_children=[],
                                      owned_stop={"state": "completed"}, owner_restart=False)
    pointer = pathlib.Path(body_switch.pointer_path(str(serving)))
    assert (phase, pointer.exists()) == (("authorized", False) if managed else ("armed", True))


@pytest.mark.parametrize("managed", [True, False])
def test_a_pending_authority_switch_waits_untouched(policy, monkeypatch, tmp_path, managed):
    from ouroboros import store_switch

    policy(managed)
    intent = tmp_path / store_switch.INTENT_REL
    intent.parent.mkdir(parents=True, exist_ok=True)
    intent.write_text(json.dumps({"door": "update_replace", "target_sha": "f" * 40}), encoding="utf-8")
    before = intent.read_bytes()
    seam = _calls(monkeypatch, "ouroboros.store_switch.run_switch_seam", {"status": "switched"})
    outcome = store_switch.resume_pending_switch(tmp_path, repo_dir=tmp_path)
    if managed:
        assert outcome["status"] == MANAGED and seam == [] and intent.read_bytes() == before
    else:
        assert outcome == {"status": "switched"} and len(seam) == 1


@pytest.mark.parametrize("managed", [True, False])
def test_a_recorded_update_is_neither_finalized_nor_rolled_back_at_boot(policy, monkeypatch, managed):
    from supervisor import update_merge

    policy(managed)
    monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda: ("corrupt", {}))
    quarantined = _calls(monkeypatch, "supervisor.update_merge.quarantine_corrupt_update_tx_marker", {"quarantined": True})
    outcome = update_merge._finalize_managed_update_locked(True)
    if managed:
        assert outcome["reason"] == MANAGED and quarantined == []
    else:
        assert outcome == {"quarantined": True} and len(quarantined) == 1


# --- restart, startup and what earlier generations recorded ------------------------------------------------

def _served(repo: pathlib.Path) -> dict:
    """What a restart of the same program never changes: HEAD, its branch and every file's bytes."""
    files = {str(path.relative_to(repo)): path.read_bytes() for path in sorted(repo.rglob("*"))
             if path.is_file() and ".git" not in path.relative_to(repo).parts}
    branch = subprocess.run(["git", "symbolic-ref", "-q", "--short", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    return {"head": git(repo, "rev-parse", "HEAD"), "branch": branch, "files": files}


def _pip_spawns(monkeypatch) -> list:
    """Every ``pip install`` the code under test starts (refused here); any other process (Git) runs as it would."""
    from ouroboros.tools import shell_process

    real, seen = shell_process.contained_popen, []

    def spawn(popen_cls, cmd, *args, **kwargs):
        if list(cmd)[1:4] == ["-m", "pip", "install"]:
            seen.append(list(cmd))
            raise AssertionError("pip was started")
        return real(popen_cls, cmd, *args, **kwargs)

    monkeypatch.setattr(shell_process, "contained_popen", spawn)
    return seen


@pytest.mark.parametrize("managed", [True, False])
def test_a_restart_runs_the_same_program_and_leaves_a_recorded_update_intent(bootstrap, policy, monkeypatch,  # noqa: F811
                                                                             managed):
    """The owner/agent Restart and the launcher-managed boot share ``safe_restart`` behind the update lock:
    managed, no checkout, reset, clean, install or recovery point runs and an earlier update intent stays."""
    from ouroboros.server_restart import _safe_restart_serialized
    from supervisor import git_ops, git_ops_reset

    repo, drive, calls = bootstrap
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_MANAGED_REPO_DIR", str(repo))
    monkeypatch.setattr(git_ops_reset, "_SMOKED", {})
    first = git(repo, "rev-list", "--max-parents=0", "HEAD")
    git(repo, "branch", "ouroboros-stable", first)
    git(repo, "checkout", "-q", "--detach", first)  # the deployed program is not the dev branch's tip
    (repo / "admin-notes.txt").write_text("the administrator's file\n", encoding="utf-8")
    git_ops._write_update_intent({"branch": "ouroboros", "target_sha": "f" * 40})  # from before the policy
    intent = git_ops._update_intent_marker_path()
    recorded, before = intent.read_bytes(), _served(repo)
    policy(managed)
    fallback = _calls(monkeypatch, "supervisor.git_ops_reset._recovery_checkout", (False, "fallback reached"))
    if managed:  # no test boundary hides an install: only the policy stops it
        monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE", raising=False)
    spawned = _pip_spawns(monkeypatch)
    ok, message = _safe_restart_serialized(git_ops.safe_restart, reason="owner_restart",
                                           unsynced_policy="rescue_and_reset")
    assert ok, message
    assert "smoke" in calls and fallback == [] and spawned == []
    if managed:
        assert message == "OK: the deployed program, unchanged" and "reset" not in calls
        assert _served(repo) == before and intent.read_bytes() == recorded
        rows = [json.loads(line) for line in (drive / "logs" / "supervisor.jsonl").read_text().splitlines()]
        assert rows[-1]["type"] == MANAGED and rows[-1]["reason"] == "owner_restart"
    else:
        assert message == "OK: ouroboros" and "reset" in calls and not intent.exists()
        assert (_served(repo)["branch"], _served(repo)["head"]) == ("ouroboros", git(repo, "rev-parse", "ouroboros"))


@pytest.mark.parametrize("managed", [True, False])
def test_an_import_failure_is_told_and_never_falls_back_to_an_older_program(bootstrap, policy, monkeypatch,  # noqa: F811
                                                                           managed):
    from supervisor import git_ops

    repo, _drive, calls = bootstrap
    measured = subprocess.run

    def failing_smoke(cmd, **kwargs):
        if len(cmd) == 3 and cmd[1] == "-c" and cmd[2].startswith("import ouroboros,"):
            calls.append("smoke")
            return subprocess.CompletedProcess(cmd, 1, "", "ImportError: broken")
        return measured(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing_smoke)
    before = _served(repo)
    policy(managed)
    fallback = _calls(monkeypatch, "supervisor.git_ops_reset._recovery_checkout", (False, "fallback reached"))
    ok, message = git_ops.safe_restart(reason="bootstrap")
    assert not ok and "smoke" in calls
    if managed:
        assert fallback == [] and _served(repo) == before
        assert message.startswith("Import test failed (rc=1); no other version or package was tried.")
        assert "only its administrator replaces it" in message
    else:
        assert len(fallback) == 1 and message.endswith("fallback reached")


@pytest.mark.parametrize("managed", [True, False])
def test_the_one_install_chokepoint_installs_nothing_into_a_managed_interpreter(policy, monkeypatch, tmp_path,
                                                                              managed):
    """``sync_runtime_dependencies`` serves bootstrap (the direct server's too), Restart and the fallbacks; the
    body switch's own command reads the same answer."""
    from supervisor import git_ops, git_ops_reset

    policy(managed)
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE", raising=False)  # the test boundary would hide the difference
    monkeypatch.setattr(git_ops_reset.sys, "frozen", False, raising=False)
    command = git_ops_reset.runtime_dependency_command(tmp_path)
    spawned = _pip_spawns(monkeypatch)
    outcome = git_ops_reset.sync_runtime_dependencies("bootstrap_local_dev")
    if managed:
        assert command is None and outcome == (True, MANAGED) and spawned == []
    else:
        assert command[1:4] == ["-m", "pip", "install"] and len(spawned) == 1
        assert outcome[0] is False and "pip was started" in outcome[1]


@pytest.mark.parametrize("managed", [True, False])
def test_the_launcher_restarts_the_same_program_without_syncing_its_packages(launch, monkeypatch, policy,  # noqa: F811
                                                                            managed):
    """Exit 42 under the launcher (desktop apps, the managed-body image): the bundle metadata is still read."""
    module = launch.module
    monkeypatch.setattr(module, "launcher_sources_changed", lambda *a, **kw: False)
    _lifecycle(launch, monkeypatch, returncode=42)
    policy(managed)
    with pytest.raises(FrozenContinues):
        module.agent_lifecycle_loop()
    assert launch.calls == (["server", "sync"] if managed else ["server", "sync", "deps"])


@pytest.mark.parametrize("managed", [True, False])
def test_an_update_left_mid_merge_defers_a_restart_and_says_who_resolves_it(policy, monkeypatch, managed):
    from ouroboros import server_restart
    from supervisor import update_merge

    policy(managed)
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda **_kw: object())
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _fh: None)
    monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda: ("valid", {"phase": "assisted_resolution"}))
    restarts: list = []
    ok, message = server_restart._safe_restart_serialized(lambda **kw: restarts.append(kw) or (True, "ok"),
                                                          reason="owner_restart", unsynced_policy="rescue_and_reset")
    assert not ok and restarts == [] and message.startswith("Managed update merge is still being resolved")
    assert ("its administrator resolves it" in message) is managed


@pytest.fixture
def adoption(tmp_path, monkeypatch):
    return Scene(tmp_path, monkeypatch)


@pytest.mark.parametrize("managed", [True, False])
def test_an_armed_adoption_is_abandoned_untouched_and_the_start_goes_on(adoption, policy, managed):
    """Armed before the policy (unmanaged), then a real cold ``python server.py``: managed, the captured helper
    never runs, the program starts as deployed, the adoption is closed as not applied with its candidate kept
    and the owner is told; unmanaged, the same start switches as before."""
    from ouroboros import body_adoption, deployment_policy

    scene = adoption
    policy(False)
    scene.commit()
    scene.authorize()
    assert scene.arm() == "armed" and scene.pointer().exists()
    before = _served(scene.serving)
    policy(managed)
    started = scene.boot()
    assert started.returncode == 0, started.stderr
    if not managed:
        assert scene.reports()[-1]["a"] == "GEN_CAND" and git(scene.serving, "rev-parse", "HEAD") == scene.cand
        return
    assert [(r["server"], r["a"], r["b"]) for r in scene.reports()] == [("GEN_OLD", "GEN_OLD", "GEN_OLD")]
    assert _served(scene.serving) == before and not scene.pointer().exists()
    handoff = body_adoption.read(scene.data)
    assert handoff["phase"] == "abandoned" and handoff["events"][-1]["detail"] == MANAGED
    again = scene.boot()  # the next start is an ordinary one
    assert again.returncode == 0 and scene.reports()[-1]["a"] == "GEN_OLD" and _served(scene.serving) == before
    settled = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert settled["outcome"] == "not_applied" and deployment_policy.BODY_STATUS in settled["note"]
    assert settled["note"].endswith("The serving body is unchanged; the candidate is retained.")
    closed = [json.loads(line) for line in (scene.data / "logs/supervisor.jsonl").read_text().splitlines()]
    assert closed[-1]["type"] == "body_adoption_closed" and closed[-1]["outcome"] == f"abandoned:{MANAGED}"
    assert scene.candidate.is_dir() and git(scene.serving, "cat-file", "-t", scene.cand) == "commit"


def test_a_tree_switched_before_the_policy_starts_as_it_is(adoption, policy):
    """A switch completed by an unmanaged generation: the managed start imports that tree, writes nothing."""
    from ouroboros import body_adoption

    scene = adoption
    policy(False)
    scene.commit()
    scene.authorize()
    assert scene.arm() == "armed" and scene.boot().returncode == 0 and scene.phase() == "switched"
    before = _served(scene.serving)
    policy(True)
    started = scene.boot()
    assert started.returncode == 0, started.stderr
    assert scene.reports()[-1]["a"] == "GEN_CAND" and _served(scene.serving) == before
    assert body_adoption.read(scene.data)["phase"] == "switched"


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize("record", ["switching", "unreadable", "missing"])
def test_a_possibly_mixed_tree_never_runs_its_helper_or_imports_under_management(policy, tmp_path, managed, record):
    """A pointer whose record says the tree may be mixed, cannot say, or is gone: managed, neither the captured
    helper nor the tree runs; without a policy the helper runs as before (a gone one stops the boot as before)."""
    serving = make_serving(tmp_path)
    helper = tmp_path / ("gone" if record == "missing" else "helper")
    if record != "missing":
        helper.mkdir()
        (helper / "switch.py").write_text(
            "import pathlib\npathlib.Path(SERVING_ROOT, 'helper-ran').write_text('yes')\n", encoding="utf-8")
    if record == "switching":
        (helper / "handoff.json").write_text(json.dumps({"phase": "switching", "repo_dir": str(serving)}),
                                             encoding="utf-8")
    (serving / ".git" / "ouroboros-body-adoption").write_text(f"{helper}\n", encoding="utf-8")
    before = _served(serving)
    env = {key: value for key, value in os.environ.items() if key != "OUROBOROS_DEPLOYMENT_POLICY"}
    if managed:
        env["OUROBOROS_DEPLOYMENT_POLICY"] = str(tmp_path / "admin" / "deployment-policy.json")
    entry = subprocess.run([sys.executable, "-c", "import ouroboros"], cwd=str(serving), env=env,
                           capture_output=True, text=True, timeout=60)
    if managed:
        assert entry.returncode == 3 and "administrator decides" in entry.stderr, entry.stderr
        assert not (serving / "helper-ran").exists() and _served(serving) == before
    elif record == "missing":
        assert entry.returncode == 3 and "unusable" in entry.stderr, entry.stderr
    else:
        assert entry.returncode == 0, entry.stderr
        assert (serving / "helper-ran").read_text() == "yes"


# --- agent tools on the serving checkout ----------------------------------------------------------------

@pytest.fixture
def body(tmp_path, monkeypatch):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    return serving, data


def _registry(serving, data, ctx):
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=serving, drive_root=data)
    registry.set_context(ctx)
    return registry


@pytest.mark.parametrize("managed", [True, False])
def test_serving_checkout_edits_refuse_while_a_candidate_keeps_working(body, policy, managed):
    serving, data = body
    policy(managed)
    # No task identity: the legacy contract edits and restores the serving checkout itself.
    legacy = _registry(serving, data, make_ctx(serving, data, "", task_metadata={}))
    edited = legacy.execute("write_file", {"path": "ouroboros/mod_a.py", "content": "GEN = 'EDITED'\n"})
    (serving / "ouroboros" / "mod_b.py").write_text("GEN = 'DIRTY'\n", encoding="utf-8")
    restored = legacy.execute("vcs_restore", {"root": "system_repo", "confirm": True, "paths": ["ouroboros/mod_b.py"]})
    if managed:
        assert "MANAGED_BY_DEPLOYMENT" in edited and "MANAGED_BY_DEPLOYMENT" in restored, (edited, restored)
        assert (serving / "ouroboros" / "mod_a.py").read_text() == "GEN = 'GEN_OLD'\n"
        assert (serving / "ouroboros" / "mod_b.py").read_text() == "GEN = 'DIRTY'\n"
    else:
        assert (serving / "ouroboros" / "mod_a.py").read_text() == "GEN = 'EDITED'\n", edited
        assert (serving / "ouroboros" / "mod_b.py").read_text() == "GEN = 'GEN_OLD'\n", restored
        git(serving, "checkout", "--", "ouroboros/mod_a.py")
    # A task's own write lands in its body candidate, managed or not; the serving checkout stays as deployed.
    task = _registry(serving, data, make_ctx(serving, data, "root-candidate"))
    written = task.execute("write_file", {"path": "ouroboros/mod_a.py", "content": "GEN = 'CANDIDATE'\n"})
    assert "MANAGED_BY_DEPLOYMENT" not in written, written
    assert (serving / "ouroboros" / "mod_a.py").read_text() == "GEN = 'GEN_OLD'\n"


# --- dependencies of the serving interpreter -------------------------------------------------------------

@pytest.mark.parametrize("managed", [True, False])
def test_the_local_runtime_install_refuses_before_pip(policy, monkeypatch, managed):
    from ouroboros import local_model
    from ouroboros.gateway.models import api_local_model_install_runtime

    policy(managed)
    installs: list = []
    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        _runtime_status="missing", install_runtime=lambda: installs.append(1)))
    app = Starlette(routes=[Route("/install", api_local_model_install_runtime, methods=["POST"])])
    with TestClient(app) as client:
        answer = client.post("/install")
    if managed:
        assert answer.status_code == 403 and answer.json()["code"] == MANAGED and installs == []
    else:
        assert answer.json() == {"status": "installing"} and installs == [1]


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize("installed", [True, False])
def test_browser_tools_install_no_package_into_a_managed_interpreter(policy, monkeypatch, managed, installed):
    """The Playwright package would land in the serving interpreter; browser binaries and an installed package
    stay as they are."""
    import builtins

    from ouroboros.tools import browser

    policy(managed)
    real_import = builtins.__import__

    def importer(name, *args, **kwargs):
        if name == "playwright" and not installed:
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importer)
    monkeypatch.setattr(browser.sys, "frozen", False, raising=False)
    pip = _calls(monkeypatch, "ouroboros.tools.browser._install_beside_calls")
    launched = _calls(monkeypatch, "ouroboros.tools.browser._pinned_chromium", "/fixture/chromium")
    if managed and not installed:
        with pytest.raises(RuntimeError, match="administrator"):
            browser._ensure_playwright_installed(engine="chromium")
        assert pip == [] and launched == []
        return
    assert browser._ensure_playwright_installed(engine="chromium") == "/fixture/chromium"
    assert len(pip) == (0 if installed else 1) and len(launched) == 1


# --- the update channel --------------------------------------------------------------------------------

@pytest.mark.parametrize("managed", [True, False])
def test_the_update_channel_stays_as_deployed(tmp_path, monkeypatch, policy, managed):
    from ouroboros import config, deployment_policy, route_choice
    from ouroboros.gateway.owner_settings import _owner_write_settings

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    monkeypatch.setattr(route_choice, "_VIEW", None)
    monkeypatch.setenv("OUROBOROS_UPDATE_CHANNEL", "qa")  # the deployment's own launch variable
    (data / "settings.json").write_text(json.dumps({"TOTAL_BUDGET": 50}), encoding="utf-8")
    policy(managed)
    _owner_write_settings({**config.load_settings(), "TOTAL_BUDGET": 25})  # carries the launch value back: no change
    assert json.loads((data / "settings.json").read_text())["TOTAL_BUDGET"] == 25
    change = {**config.load_settings(), "OUROBOROS_UPDATE_CHANNEL": "development"}
    if managed:
        with pytest.raises(deployment_policy.DeploymentPolicyRefusal, match="OUROBOROS_UPDATE_CHANNEL"):
            _owner_write_settings(change)
        assert json.loads((data / "settings.json").read_text()).get("OUROBOROS_UPDATE_CHANNEL") != "development"
    else:
        _owner_write_settings(change)
        assert json.loads((data / "settings.json").read_text())["OUROBOROS_UPDATE_CHANNEL"] == "development"


# --- what Settings, the wizard and the Updates page are told ----------------------------------------------

@pytest.mark.parametrize("managed", [True, False])
def test_settings_and_the_wizard_offer_only_what_the_policy_allows(tmp_path, monkeypatch, managed):
    from ouroboros import deployment_policy
    from ouroboros.gateway import control
    from ouroboros.settings_setup_contract import build_setup_bootstrap

    path = tmp_path / "admin" / "deployment-policy.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"version": 1, "access": {"max": "advanced"}, "safety": {
        "min_coverage": "full", "checker": {"model": "anthropic@safety::checker",
                                            "fallbacks": ["organization-checker (local)"]}}}), encoding="utf-8")
    monkeypatch.delenv(deployment_policy.GENERATION_ENV, raising=False)
    if managed:
        monkeypatch.setenv(deployment_policy.POLICY_ENV, str(path))
    else:
        monkeypatch.delenv(deployment_policy.POLICY_ENV, raising=False)
    deployment_policy.reset_for_tests()
    monkeypatch.setattr("supervisor.git_ops.compute_managed_update_status", lambda fetch: {"managed": False})
    monkeypatch.setattr("supervisor.update_merge.active_update_tx", lambda: None)
    try:
        bootstrap = build_setup_bootstrap({}, "web")
        update = control._managed_update_payload(fetch=False, include_tags=False)
    finally:
        deployment_policy.reset_for_tests()
    modes = [mode["value"] for mode in bootstrap["contract"]["runtimeModes"]]
    managed_slots = [slot["slot"] for slot in bootstrap["contract"]["modelSlots"] if slot["managed"]]
    if managed:
        assert modes == ["light", "advanced"] and managed_slots == ["safety"]
        assert bootstrap["deployment"]["checker"]["connections"] == ["anthropic@safety"]
        assert bootstrap["deployment"]["checker"]["local"] is True
        assert "administrator" in update["managed_by_deployment"]
    else:
        assert modes == ["light", "advanced", "pro", "cyber_pro"] and managed_slots == []
        assert bootstrap["deployment"] is None and "managed_by_deployment" not in update

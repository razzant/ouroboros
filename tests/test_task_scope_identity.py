"""Task lifetime follows bound identity, never a service-log label such as ``manual``."""
from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ouroboros import process_scope as ps
from ouroboros.tools import services
from ouroboros.tools.registry import ToolContext
from tests.test_run_command_schema_truth import _bound

pytestmark = [pytest.mark.serial, pytest.mark.skipif(os.name == "nt", reason="task census is POSIX-only")]


def test_invocation_coordinates_are_atomic_and_preserve_explicit_non_task(tmp_path):
    from ouroboros.owner_pause import _member_coordinates
    from ouroboros.tool_custody import invocation_binding

    fallback = {"drive_root": tmp_path / "other", "task_id": "fallback"}
    assert ps.task_launch_target({"drive_root": tmp_path, "task_id": "manual"}, **fallback) == (tmp_path, "manual")
    for binding in ({"drive_root": "", "task_id": ""}, {"task_id": "named"}, {"drive_root": tmp_path}):
        assert ps.task_launch_target(binding, **fallback) is None, "never mix two owners' coordinates"
    assert ps.task_launch_target({}, drive_root=tmp_path, task_id="manual") == (tmp_path, "manual")
    assert ps.task_launch_target({}) is None
    source = SimpleNamespace(task_id="manual", drive_root=tmp_path / "execution",
                             task_metadata={"budget_drive_root": str(tmp_path / "canonical")})
    root, _, task = _member_coordinates(source)
    assert ps.task_launch_target({}, drive_root=root, task_id=task) == (tmp_path / "canonical", "manual")
    source.non_task_operation = True
    with _bound(source):
        assert ps.task_launch_target(invocation_binding(), **fallback) is None


@pytest.mark.parametrize("executor_kind", [None, "local"], ids=["host", "executor"])
@pytest.mark.parametrize("case", ["manual", "ordinary", "unbound", "non-task", "non-task-named", "kept"])
@pytest.mark.parametrize("invoked", [False, True], ids=["context", "invocation"])
def test_service_scope_uses_identity_not_its_display_fallback(tmp_path, monkeypatch, executor_kind, case, invoked):
    from ouroboros import config as cfg

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    workspace, drive = tmp_path / "workspace", tmp_path / "data"
    workspace.mkdir()
    drive.mkdir()
    task = "" if case == "unbound" else "ordinary-task" if case in ("ordinary", "non-task-named") else "manual"
    ctx = ToolContext(repo_dir=tmp_path / "system", drive_root=drive, workspace_root=workspace,
                      workspace_mode="external", task_id=task)
    ctx.non_task_operation = case in ("non-task", "non-task-named")
    ctx.task_lifecycle_bound = False
    if executor_kind:
        ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace),
                            "workspace_backend_path": "/workspace"}
    child = workspace / "child.py"
    child.write_text("import json,os,pathlib,time\n"
                     "pathlib.Path('markers.json').write_text(json.dumps([k for k in os.environ "
                     "if k.startswith('OURO_TASK_SCOPE_')]))\n"
                     "print('READY',flush=True)\ntime.sleep(60)\n")
    try:
        with _bound(ctx) if invoked else nullcontext():
            result = json.loads(services._start_service(ctx, [sys.executable, str(child)], name="identity",
                                keep_alive=case == "kept", readiness={"log_contains": "READY", "timeout_sec": 30}))
        (tmp_path / "start-result.json").write_text(json.dumps(result, indent=2))
        assert result["ready"], json.dumps(result, sort_keys=True)
        markers = json.loads((workspace / "markers.json").read_text())
        record = ps.task_scope_record(drive, task or "manual")
        if case in ("manual", "ordinary"):
            assert record and record["task_id"] == task
            assert markers == [record["marker"]] and record["groups"]
        else:
            assert markers == [] and record is None, "display fallback and non-task facts confer no lifetime"
    finally:
        stopped = services._stop_service(ctx, name="identity")
        (tmp_path / "stop-result.json").write_text(stopped)

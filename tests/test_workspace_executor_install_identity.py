"""Executor ownership is checked before directories, durable records or launch."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import body_candidate, install_identity, process_custody
from ouroboros import workspace_executor as executor


def _tree(root):
    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
            for path in root.rglob("*")}


@pytest.fixture
def roots(tmp_path, monkeypatch):
    a, b, bare = tmp_path / "A", tmp_path / "B", tmp_path / "isolated"
    for root in (a, b):
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
        install_identity.initialize_install()
    monkeypatch.setattr(executor, "_process_command_sha256", lambda pid: "")
    return a, b, bare


@pytest.mark.parametrize("operation", ["state_dir", "register"])
@pytest.mark.parametrize("layout", ["", "task_drives/task", "state/headless_tasks/task/data"])
@pytest.mark.parametrize("bound", [False, True])
def test_custody_preparation_refuses_foreign_but_owns_local_and_bare(
        roots, monkeypatch, operation, layout, bound):
    a, b, bare = roots
    binding = {}
    monkeypatch.setattr(executor, "invocation_binding", lambda: binding)

    def act(root):
        binding.clear()
        if bound:
            binding["drive_root"] = str(root)
        explicit = b if bound else root
        if operation == "state_dir":
            return executor._state_dir(explicit)
        return executor._register_process(explicit, {"record_type": "foreground", "host_pid": 0})

    before = _tree(a)
    with pytest.raises(install_identity.StoreForeign) as refused:
        act(a / layout)
    assert refused.value.reason_code == "foreign_install_root"
    assert _tree(a) == before

    for root in (b / layout, bare / layout):
        result = act(root)
        state = root / "state" / "workspace_executor_processes"
        assert state.is_dir()
        if operation == "register":
            assert result.parent == state and json.loads(result.read_text())["record_type"] == "foreground"
        else:
            assert result == state
    assert not (bare / install_identity.IDENTITY_PATH).exists()


@pytest.mark.parametrize("kind", ["local", "docker_exec"])
@pytest.mark.parametrize("operation", ["execute", "direct", "service"])
@pytest.mark.parametrize("bound", [False, True])
def test_executor_launch_refuses_foreign_before_any_effect_and_uses_selected_owned_root(
        roots, tmp_path, monkeypatch, kind, operation, bound):
    a, b, bare = roots
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    binding, launches, prepared, records = {}, [], [], []
    monkeypatch.setattr(executor, "invocation_binding", lambda: binding)
    monkeypatch.setattr(executor, "bootstrap_process_path", lambda: prepared.append("bootstrap"))
    monkeypatch.setattr(body_candidate, "executor_environment", lambda *a, **k: prepared.append("environment"))
    monkeypatch.setattr(executor, "_executor_service_env", lambda: {})
    monkeypatch.setattr(executor, "_docker_exec_completed", lambda *a: True)
    monkeypatch.setattr(executor, "_retire_docker_completion", executor._forget_process)
    monkeypatch.setattr(executor, "_wait_readiness", lambda *a: None)
    monkeypatch.setattr(executor, "_service_state", lambda *a: "exited")

    class Process:
        pid, returncode = 12345, 0

        def communicate(self, timeout=None):
            return "12345\n", ""

        def poll(self):
            return self.returncode

    def popen(kind, cmd, **kwargs):
        launches.append((kind, kwargs))
        return Process(), lambda: None

    def spawn(cmd, **kwargs):
        launches.append(("local_service", kwargs))
        proc = Process()
        kwargs["on_spawn"](proc)
        return proc

    register = executor._register_process

    def record(root, payload):
        path = register(root, payload)
        records.append(path)
        return path

    monkeypatch.setattr(executor, "_handed_off_popen", popen)
    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)
    monkeypatch.setattr(executor, "_register_process", record)
    ref = {"type": kind, "container_name": "fake", "network": "host",
           "workspace_host_path": str(workspace), "workspace_backend_path": "/workspace"}

    def act(root):
        binding.clear()
        if bound:
            binding["drive_root"] = str(root)
        # Binding is the custody owner even when the context points elsewhere.
        other_root = b if root.is_relative_to(a) else a
        ctx = SimpleNamespace(drive_root=other_root if bound else root, executor_ref=ref, task_id="")
        if operation == "execute":
            return executor.execute(ctx, ["unused"], workspace, 1)
        if operation == "service":
            return executor.start_service(ctx, name="test", cmd=["unused"], host_cwd=workspace,
                                          cwd_root="active_workspace", readiness={}, outputs=[], before_outputs={})
        normalized = executor.normalize_executor_ref(ref)
        if kind == "local":
            return executor._execute_local(normalized, ["unused"], workspace, 1,
                                           drive_root=ctx.drive_root, local_env={})
        return executor._execute_docker(normalized, ["unused"], "/workspace", 1, drive_root=ctx.drive_root)

    before = _tree(a)
    layout = "state/headless_tasks/task/data"
    with pytest.raises(install_identity.StoreForeign) as refused:
        act(a / layout)
    assert refused.value.reason_code == "foreign_install_root"
    assert _tree(a) == before and not launches and not prepared and not records

    for root in (b / layout, bare / layout):
        result = act(root)
        assert records[-1].parent == root / "state" / "workspace_executor_processes"
        if operation == "service":
            assert result["backend_pid"] == "12345"
            if kind == "local":
                assert launches[-1][1]["drive_root"] == root
                assert (root / "services" / "manual" / "test.executor.log").is_file()
            executor._SERVICES.clear()
        else:
            assert result.returncode == 0 and result.operation_outcome == "completed"
    assert len(launches) == 2 and _tree(a) == before
    assert not (bare / install_identity.IDENTITY_PATH).exists()

"""Exercise the real shell shim without Docker, network, or benchmark packages."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

from devtools.benchmarks.cowork_bench.resource_limits import (
    ADMISSION_DIR_NAME,
    LABEL_KEY,
    prepare_resource_env,
)

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(os.name == "nt" or not BASH, reason="Cowork runner requires POSIX Bash")


@pytest.fixture
def envelope(tmp_path: pathlib.Path) -> tuple[dict[str, str], pathlib.Path]:
    binary_dir = tmp_path / "bin with spaces"
    binary_dir.mkdir()
    fake = binary_dir / "docker"
    fake.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = ps ]; then printf "%s" "${FAKE_DOCKER_PS:-}"; exit "${FAKE_DOCKER_PS_EXIT:-0}"; fi\n'
        'printf "%s\\0" "$@" > "$FAKE_DOCKER_ARGS"\n'
        'printf "%s" "${BASH_ENV:-}" > "$FAKE_DOCKER_BASH_ENV"\n'
        'printf "docker-output\\n"\n'
        'exit "${FAKE_DOCKER_EXIT:-0}"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    original = {key: value for key, value in os.environ.items() if key != "BASH_ENV"}
    original.update({
        "PATH": f"{binary_dir}{os.pathsep}{os.defpath}",
        "FAKE_DOCKER_ARGS": str(tmp_path / "argv"),
        "FAKE_DOCKER_BASH_ENV": str(tmp_path / "bash_env"),
    })
    env = prepare_resource_env(
        original, run_root=tmp_path, docker_host="unix:///run/example.sock",
        resource_root=tmp_path, min_free_bytes=0,
    )
    return env, tmp_path


def invoke(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([BASH, env["COWORK_DOCKER_SHIM"], *args], env=env,
                          capture_output=True, text=True, timeout=10)


def recorded(root: pathlib.Path) -> list[str]:
    return (root / "argv").read_bytes().decode("utf-8").rstrip("\0").split("\0")


@pytest.mark.parametrize("prefix", [("run",), ("create",), ("container", "run"), ("container", "create")])
@pytest.mark.serial
def test_every_container_creation_has_limits_and_label(envelope, prefix):
    env, root = envelope
    tail = ["--rm", "image:fixed", "sh", "-c", "printf 'quoted value'"]
    result = invoke(env, *prefix, *tail)
    assert result.returncode == 0
    assert result.stdout == "docker-output\n"
    args = recorded(root)
    assert args[:len(prefix)] == list(prefix)
    assert args[len(prefix):-len(tail)] == [
        "--cpus", "4", "--memory", "16g", "--memory-swap", "16g",
        "--pids-limit", "512", "--pull=never", "--label",
        f"{LABEL_KEY}={env['COWORK_RUN_LABEL']}",
    ]
    assert args[-len(tail):] == tail
    assert (root / "bash_env").read_text(encoding="utf-8") == ""


@pytest.mark.serial
def test_discovery_survives_official_path_reset(envelope):
    env, root = envelope
    # These are the upstream runner's actual PATH and DOCKER assignments.
    command = (
        'export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"; '
        'DOCKER=$(which docker 2>/dev/null || echo "/usr/local/bin/docker"); '
        '"$DOCKER" run --rm image:fixed true'
    )
    result = subprocess.run([BASH, "-c", command], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert recorded(root)[0:3] == ["run", "--cpus", "4"]
    assert (root / "bash_env").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("args", [
    ("exec", "agent-1", "sh", "-c", "printf '$TOKEN with spaces'"),
    ("rm", "-fv", "agent-1"),
    ("network", "rm", "net-1"),
    ("container", "inspect", "agent-1"),
])
@pytest.mark.serial
def test_noncreation_preserves_exact_arguments_exit_and_stdout(envelope, args):
    env, root = envelope
    env["FAKE_DOCKER_EXIT"] = "7"
    (root / "resource_stop").write_text("budget_exhausted\n", encoding="utf-8")
    result = invoke(env, *args)
    assert result.returncode == 7
    assert result.stdout == "docker-output\n"
    assert not result.stderr
    assert recorded(root) == list(args)


@pytest.mark.serial
def test_network_is_labeled_without_container_flags(envelope):
    env, root = envelope
    result = invoke(env, "network", "create", "network with spaces")
    assert result.returncode == 0
    assert recorded(root) == [
        "network", "create", "--label", f"{LABEL_KEY}={env['COWORK_RUN_LABEL']}",
        "network with spaces",
    ]


@pytest.mark.parametrize("prefix", [("run",), ("network", "create")])
@pytest.mark.serial
def test_stop_file_refuses_creations_and_preserves_first_reason(envelope, prefix):
    env, root = envelope
    stop = root / "resource_stop"
    stop.write_text("budget_exhausted\n", encoding="utf-8")
    result = invoke(env, *prefix, "unused")
    assert result.returncode == 75
    assert "admission_stopped" in result.stderr
    assert stop.read_text(encoding="utf-8") == "budget_exhausted\n"
    assert not (root / "argv").exists()


@pytest.mark.serial
def test_low_disk_refuses_before_docker_and_keeps_teardown_available(envelope):
    env, root = envelope
    df = pathlib.Path(env["COWORK_REAL_DOCKER"]).parent / "df"
    df.write_text('#!/bin/bash\nprintf "Filesystem 1024-blocks Used Available Capacity Mounted\\n/dev/fake 10 9 1 90%% /fake\\n"\n',
                  encoding="utf-8")
    df.chmod(0o755)
    env["COWORK_MIN_FREE_BYTES"] = "2048"
    result = invoke(env, "run", "image", "true")
    assert result.returncode == 75
    assert (root / "resource_stop").read_text(encoding="utf-8") == "disk_reserve_reached\n"
    assert not (root / "argv").exists()
    assert invoke(env, "rm", "-fv", "own-container").returncode == 0
    assert recorded(root) == ["rm", "-fv", "own-container"]


def test_prepare_keeps_input_and_binds_paths_label_and_daemon(envelope):
    env, root = envelope
    clean = {key: value for key, value in env.items() if key != "BASH_ENV"}
    before = dict(clean)
    other = prepare_resource_env(clean, run_root=root / "other", docker_host="unix:///other.sock",
                                 resource_root=root)
    assert clean == before
    assert pathlib.Path(other["COWORK_REAL_DOCKER"]).is_absolute()
    assert other["COWORK_RUN_LABEL"] != env["COWORK_RUN_LABEL"]
    assert other["COWORK_MIN_FREE_BYTES"] == str(200 * 1024**3)
    assert other["DOCKER_HOST"] == "unix:///other.sock"
    assert other["TMPDIR"] == str(root)
    assert other["COWORK_STOP_FILE"] == str(root / "other" / "resource_stop")
    assert other["COWORK_ADMISSION_DIR"] == str(root / "other" / ADMISSION_DIR_NAME)


def test_prepare_does_not_silently_replace_shell_startup(envelope):
    env, root = envelope
    with pytest.raises(ValueError, match="unset BASH_ENV"):
        prepare_resource_env({**env, "BASH_ENV": "/existing/startup.sh"}, run_root=root,
                             docker_host="unix:///run/example.sock", resource_root=root)


AGENT_EXEC = ("exec", "agent-4242-0a1b2c3d", "timeout", "-k", "30", "3600", "/opt/venv/bin/python3", "-u",
              "/workspace/main_ouroboros.py", "--task_dir", "task one", "--max_steps", "100", "--phase", "agent")
EVAL_EXEC = ("exec", "eval-4242-0a1b2c3d", "/opt/venv/bin/python3", "-u", "/workspace/main_ouroboros.py",
             "--task_dir", "task one", "--phase", "eval")


@pytest.mark.serial
def test_agent_exec_records_its_exact_task_and_keeps_argv(envelope):
    env, root = envelope
    assert invoke(env, *AGENT_EXEC).returncode == 0
    assert recorded(root) == list(AGENT_EXEC)
    admission = pathlib.Path(env["COWORK_ADMISSION_DIR"])
    assert (admission / "4242-0a1b2c3d.task").read_text(encoding="utf-8") == "task one\n"


@pytest.mark.parametrize(("listing", "exit_code", "state"), [
    ("", "0", "absent"),
    ("agent-4242-0a1b2c3d-sidecar\n", "0", "absent"),
    ("agent-4242-0a1b2c3d\n", "0", "present"),
    ("", "1", "unknown"),
])
@pytest.mark.serial
def test_eval_exec_receives_only_daemon_proven_agent_removal(envelope, listing, exit_code, state):
    env, root = envelope
    env.update(FAKE_DOCKER_PS=listing, FAKE_DOCKER_PS_EXIT=exit_code)
    assert invoke(env, *EVAL_EXEC).returncode == 0
    assert recorded(root) == ["exec", "-e", f"COWORK_AGENT_CONTAINER_STATE={state}", *EVAL_EXEC[1:]]


@pytest.mark.parametrize(("name", "recorded_refusal"), [("eval-4242-0a1b2c3d", True), ("pg-4242-0a1b2c3d", False)])
@pytest.mark.serial
def test_refused_eval_container_is_recorded_as_before_eval_evidence(envelope, name, recorded_refusal):
    env, root = envelope
    (root / "resource_stop").write_text("campaign_budget_reserve\n", encoding="utf-8")
    result = invoke(env, "run", "-d", "--name", name, "image", "sleep", "1800")
    assert result.returncode == 75
    refusal = pathlib.Path(env["COWORK_ADMISSION_DIR"]) / "4242-0a1b2c3d.eval_refused"
    assert refusal.exists() is recorded_refusal
    if recorded_refusal:
        assert refusal.read_text(encoding="utf-8") == "admission_stopped\n"
    assert not (root / "argv").exists()


@pytest.mark.serial
def test_evidence_failure_never_blocks_the_runner_command(envelope):
    env, root = envelope
    blocked = root / "not-a-directory"
    blocked.write_text("", encoding="utf-8")
    env["COWORK_ADMISSION_DIR"] = str(blocked)
    assert invoke(env, *AGENT_EXEC).returncode == 0
    assert recorded(root) == list(AGENT_EXEC)

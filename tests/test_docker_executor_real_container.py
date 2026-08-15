"""The residue the docker golden-trace STUB cannot reach: a real container.

The golden fixtures (`docker_exec_mapped_cwd`, `docker_exec_unmapped_root`) pin
what the dispatch pipeline COMPUTES and hands to the backend — routing decision,
host→backend cwd projection, container name, network mode, recorded executor
trace — behind a stub `docker` on PATH, because a byte-identical fixture must not
depend on a daemon, a registry or a network.

What a stub can never prove is that the projection it recorded is the spelling a
real `docker exec` actually honours: the stub is told `--workdir /workspace/sub`
and reports it back, which is agreement with itself. That is the one claim worth
checking against reality, so it is checked HERE, in the integration lane, and
skipped rather than faked when no image is available.

Deliberately still not covered by anything in this repo, stated so it is not
mistaken for done: the in-container timeout teardown path (setsid + pidfile stop
shell), signal delivery to the backend pid, and `docker inspect` network=none
enforcement against a live container. Those need a purpose-built image and a
process-lifetime harness.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.serial]

_IMAGE = "ubuntu:24.04"


def _docker_usable() -> str:
    if not shutil.which("docker"):
        return "docker CLI not on PATH"
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"docker daemon unreachable: {exc}"
    if info.returncode != 0:
        return f"docker daemon unreachable: {info.stderr.strip()[:200]}"
    images = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if _IMAGE not in (images.stdout or ""):
        # No pull: the registry is not reachable in every environment this runs
        # in, and a test that hangs on a pull is worse than a skipped one.
        return f"{_IMAGE} not present locally (no pull attempted by design)"
    return ""


@pytest.fixture()
def running_container(tmp_path: pathlib.Path):
    reason = _docker_usable()
    if reason:
        pytest.skip(reason)
    workspace = tmp_path / "ws"
    (workspace / "sub").mkdir(parents=True)
    (workspace / "sub" / "marker.txt").write_text("in-sub\n", encoding="utf-8")
    name = f"ouroboros-golden-{uuid.uuid4().hex[:10]}"
    created = subprocess.run(
        [
            "docker", "run", "-d", "--rm", "--name", name,
            "-v", f"{workspace}:/workspace",
            _IMAGE, "sleep", "300",
        ],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"could not start container: {created.stderr.strip()[:300]}")
    try:
        yield name, workspace
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True, text=True, timeout=60, check=False,
        )


def test_the_projected_backend_cwd_is_the_spelling_a_real_container_honours(
    running_container,
):
    """`/workspace/sub` is not just what the stub echoed back — it is where the
    process actually lands, and it is the same host directory."""

    from ouroboros.workspace_executor import map_host_path, normalize_executor_ref

    name, workspace = running_container
    executor = normalize_executor_ref({
        "kind": "docker_exec",
        "container_name": name,
        "network": "host",
        "workspace_host_path": str(workspace),
        "workspace_backend_path": "/workspace",
    })
    assert executor is not None
    backend_cwd = map_host_path(executor, workspace / "sub")
    assert backend_cwd == "/workspace/sub"

    proc = subprocess.run(
        ["docker", "exec", "--workdir", backend_cwd, name, "sh", "-lc", "pwd; cat marker.txt"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "/workspace/sub" in proc.stdout
    # The mount is the SAME directory, so the projection is a real identity and
    # not merely a string the stub agreed with.
    assert "in-sub" in proc.stdout


def test_a_host_path_outside_the_mapping_has_no_backend_spelling(running_container):
    """The unmapped branch's premise: there is nothing to project, hence the
    documented host fallback rather than an invented in-container path."""

    from ouroboros.workspace_executor import map_host_path, normalize_executor_ref

    name, workspace = running_container
    executor = normalize_executor_ref({
        "kind": "docker_exec",
        "container_name": name,
        "network": "host",
        "workspace_host_path": str(workspace),
        "workspace_backend_path": "/workspace",
    })
    with pytest.raises(ValueError, match="outside executor mappings"):
        map_host_path(executor, workspace.parent / "not_mapped")

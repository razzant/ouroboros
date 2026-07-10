"""ProgramBench cleanroom adapter primitives."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from devtools.benchmarks.common.manifests import write_json
from devtools.benchmarks.common.official_commands import programbench_eval_cmd, programbench_info_cmd
from devtools.benchmarks.programbench.schemas import task_body

IMAGE_TAG = "task_cleanroom_v6"
DOCKER_PLATFORM = "linux/amd64"
REFERENCE_EXECUTABLE_BASENAME = "reference_executable"
AGENT_EXECUTABLE_BASENAME = "executable"
_INSTRUCTION_TEMPLATE_PATH = pathlib.Path(__file__).resolve().parent / "instruction_template.md"
# Explicit SETTLED statuses (mirrors ouroboros.task_status.SETTLED_STATUSES and the
# terminal-bench harbor adapter). cancel_requested is deliberately NOT terminal here:
# the supervisor finalizes it to cancelled shortly after, and a wait loop should
# surface the settled record, not the cancel-intent latch.
_SETTLED_TASK_STATUSES = frozenset({"completed", "failed", "cancelled", "rejected_duplicate"})


def docker_executor_ref(
    *,
    container_name: str,
    workspace_host_path: pathlib.Path,
    workspace_backend_path: str = "/workspace",
) -> dict[str, Any]:
    return {
        "type": "docker_exec",
        "id": container_name,
        "container_name": container_name,
        "network": "none",
        "workspace_host_path": str(pathlib.Path(workspace_host_path).resolve(strict=False)),
        "workspace_backend_path": workspace_backend_path,
    }


def default_protected_backend_paths() -> list[str]:
    return [
        f"/workspace/{REFERENCE_EXECUTABLE_BASENAME}",
        REFERENCE_EXECUTABLE_BASENAME,
    ]


def build_ouroboros_task_body(
    *,
    instruction: str,
    workspace_host_path: pathlib.Path,
    container_name: str,
    protected_backend_paths: list[str] | None = None,
    task_id: str = "",
) -> dict[str, Any]:
    ensure_git_workspace(workspace_host_path)
    protected = protected_backend_paths or default_protected_backend_paths()
    return task_body(
        description=instruction,
        workspace_root=str(pathlib.Path(workspace_host_path).resolve(strict=False)),
        executor_ref=docker_executor_ref(container_name=container_name, workspace_host_path=workspace_host_path),
        protected_paths=protected,
        task_id=task_id,
    )


def preflight_cleanroom_container(container_name: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["docker", "inspect", container_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker inspect failed for {container_name}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout or "[]")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"docker inspect returned no container data for {container_name}")
    info = data[0]
    config = info.get("Config") if isinstance(info, dict) else {}
    host_config = info.get("HostConfig") if isinstance(info, dict) else {}
    image = str((config or {}).get("Image") or (info or {}).get("Image") or "")
    network = str((host_config or {}).get("NetworkMode") or "")
    if "task_cleanroom" not in image:
        raise RuntimeError(f"ProgramBench container must use a task_cleanroom image, got {image!r}")
    if network != "none":
        raise RuntimeError(f"ProgramBench inference container must use Docker NetworkMode=none, got {network!r}")
    return {"image": image, "network": network}


def ensure_git_workspace(workspace_root: pathlib.Path) -> None:
    root = pathlib.Path(workspace_root).resolve(strict=False)
    probe = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True, timeout=10)
    if probe.returncode == 0 and pathlib.Path((probe.stdout or "").strip()).resolve(strict=False) == root:
        return
    subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    subprocess.run(["git", "config", "user.email", "ouroboros-bench@example.invalid"], cwd=root, check=True, timeout=10)
    subprocess.run(["git", "config", "user.name", "Ouroboros Bench"], cwd=root, check=True, timeout=10)


def create_submission_tarball(
    workspace_root: pathlib.Path,
    out_path: pathlib.Path,
    *,
    protected_paths: list[str] | None = None,
    workspace_backend_path: str = "/workspace",
) -> pathlib.Path:
    root = pathlib.Path(workspace_root).resolve(strict=False)
    protected = _protected_submission_paths(root, protected_paths or [], workspace_backend_path=workspace_backend_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if _skip_submission_path(rel):
                continue
            resolved = path.resolve(strict=False)
            if any(_path_matches(resolved, protected_path) for protected_path in protected):
                continue
            tar.add(path, arcname=rel.as_posix(), recursive=False)
    return out_path


def _protected_submission_paths(root: pathlib.Path, protected_paths: list[str], *, workspace_backend_path: str) -> list[pathlib.Path]:
    protected: list[pathlib.Path] = []
    backend_prefix = str(workspace_backend_path or "/workspace").rstrip("/")
    for raw in protected_paths:
        text = str(raw or "").strip()
        if not text:
            continue
        if text == backend_prefix:
            protected.append(root)
            continue
        if text.startswith(backend_prefix + "/"):
            rel = text[len(backend_prefix) + 1:]
            protected.append((root / rel).resolve(strict=False))
            continue
        candidate = pathlib.Path(text)
        if candidate.is_absolute():
            continue
        protected.append((root / candidate).resolve(strict=False))
    return list(dict.fromkeys(protected))


def _path_matches(candidate: pathlib.Path, protected: pathlib.Path) -> bool:
    if candidate == protected:
        return True
    try:
        candidate.relative_to(protected)
        return True
    except ValueError:
        return False


def _skip_submission_path(rel: pathlib.PurePath) -> bool:
    # ProgramBench submissions are SOURCE submissions: the official eval rebuilds
    # via compile.sh. Shipping the agent-built root binary would mask compile
    # failures (and the reference must never leave the cleanroom), so both
    # root-level binaries stay out of the tarball by name.
    if rel.as_posix() in {AGENT_EXECUTABLE_BASENAME, REFERENCE_EXECUTABLE_BASENAME}:
        return True
    parts = set(rel.parts)
    return bool(parts & {
        ".git",
        ".ouroboros",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        "build",
        "dist",
        "htmlcov",
    }) or rel.name in {".DS_Store", ".coverage", "coverage.xml"} or rel.suffix in {".pyc", ".pyo", ".log", ".tmp"}


def cleanroom_image_ref(image_name: str, *, tag: str = IMAGE_TAG) -> str:
    base = str(image_name or "").strip()
    if not base:
        raise ValueError("image_name is required")
    if ":" in base:
        return base
    return f"{base}:{tag}"


def container_name_for_instance(instance_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(instance_id or "").strip()).strip("-._")
    slug = slug[:48] or "instance"
    return f"ouroboros-pb-{slug}"


def build_instruction(instance: dict[str, Any], *, template_path: pathlib.Path | None = None) -> str:
    path = template_path or _INSTRUCTION_TEMPLATE_PATH
    template = path.read_text(encoding="utf-8")
    values = {
        "instance_id": str(instance.get("instance_id") or ""),
        "repository": str(instance.get("repository") or ""),
        "language": str(instance.get("language") or ""),
        "difficulty": str(instance.get("difficulty") or ""),
    }
    return re.sub(
        r"\{\{(\w+)\}\}",
        lambda match: values.get(match.group(1), ""),
        template,
    )


def _docker_run(args: list[str], *, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr.strip()}")
    return proc


def pull_cleanroom_image(image_name: str, *, tag: str = IMAGE_TAG, max_attempts: int = 6) -> str:
    image = cleanroom_image_ref(image_name, tag=tag)
    retryable = ("TLS handshake timeout", "proxyconnect", "context deadline exceeded", "connection reset")
    last_error: RuntimeError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            _docker_run(["docker", "pull", "--platform", DOCKER_PLATFORM, image], timeout=3600)
            return image
        except RuntimeError as exc:
            last_error = exc
            msg = str(exc)
            if attempt >= max_attempts or not any(token in msg for token in retryable):
                raise
            delay = min(60, 5 * attempt)
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    return image


def _ensure_workspace_gitignore_entries(workspace_root: pathlib.Path, entries: list[str]) -> None:
    root = pathlib.Path(workspace_root).resolve(strict=False)
    path = root / ".gitignore"
    existing = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    merged = list(existing)
    for entry in entries:
        text = str(entry or "").strip()
        if not text or text in merged:
            continue
        merged.append(text)
    if merged != existing:
        path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")


def prepare_seeded_workspace(
    workspace_root: pathlib.Path,
    *,
    workspace_backend_path: str = "/workspace",
) -> dict[str, Any]:
    """Move the seeded reference binary off the agent build path and make it runnable."""
    root = pathlib.Path(workspace_root).expanduser().resolve(strict=False)
    seeded_reference = root / AGENT_EXECUTABLE_BASENAME
    reference_path = root / REFERENCE_EXECUTABLE_BASENAME
    if reference_path.is_file():
        # Already normalized (idempotent re-entry). A co-existing ./executable
        # here is the AGENT'S build product on a solved workspace — renaming it
        # over the reference would corrupt the protected binary; leave both, the
        # submission tarball excludes them by name.
        pass
    elif seeded_reference.is_dir():
        raise RuntimeError(f"agent build path {AGENT_EXECUTABLE_BASENAME!r} must be a file path, not a directory")
    elif seeded_reference.is_file():
        seeded_reference.rename(reference_path)
    else:
        raise RuntimeError(
            f"seeded workspace is missing a reference binary at "
            f"{AGENT_EXECUTABLE_BASENAME!r} or {REFERENCE_EXECUTABLE_BASENAME!r}"
        )
    mode = reference_path.stat().st_mode
    # task_cleanroom seeds ---x--x--x (execute-only). Native Linux FS can exec that,
    # but Mac→Colima virtiofs bind mounts return EACCES unless the owner has read too.
    # Ouroboros resource_policy still blocks read_bytes/copy on this path via tools.
    reference_path.chmod(mode | 0o511)
    _ensure_workspace_gitignore_entries(
        root,
        [
            f"/{REFERENCE_EXECUTABLE_BASENAME}",
            f"/{AGENT_EXECUTABLE_BASENAME}",
        ],
    )
    backend_prefix = str(workspace_backend_path or "/workspace").rstrip("/")
    return {
        "reference_host_path": str(reference_path),
        "reference_backend_path": f"{backend_prefix}/{REFERENCE_EXECUTABLE_BASENAME}",
        "agent_executable_backend_path": f"{backend_prefix}/{AGENT_EXECUTABLE_BASENAME}",
    }


def verify_reference_executable_runnable(
    container_name: str,
    *,
    workspace_backend_path: str = "/workspace",
) -> dict[str, Any]:
    backend_prefix = str(workspace_backend_path or "/workspace").rstrip("/")
    reference_backend_path = f"{backend_prefix}/{REFERENCE_EXECUTABLE_BASENAME}"
    test_proc = _docker_run(
        [
            "docker",
            "exec",
            container_name,
            "sh",
            "-lc",
            f"./{REFERENCE_EXECUTABLE_BASENAME} --version 2>/dev/null || "
            f"./{REFERENCE_EXECUTABLE_BASENAME} --help 2>/dev/null || "
            f"./{REFERENCE_EXECUTABLE_BASENAME} -h 2>/dev/null || "
            f"test -x ./{REFERENCE_EXECUTABLE_BASENAME}",
        ],
        timeout=30,
        check=False,
    )
    result = {
        "reference_backend_path": reference_backend_path,
        "probe_returncode": int(test_proc.returncode),
    }
    if test_proc.returncode != 0:
        stderr = (test_proc.stderr or "").strip()
        raise RuntimeError(
            f"reference binary is not runnable in {container_name} at {reference_backend_path}"
            + (f": {stderr}" if stderr else "")
        )
    return result


def seed_workspace_from_image(
    image_name: str,
    workspace_root: pathlib.Path,
    *,
    tag: str = IMAGE_TAG,
    workspace_backend_path: str = "/workspace",
) -> dict[str, Any]:
    image = cleanroom_image_ref(image_name, tag=tag)
    root = pathlib.Path(workspace_root).expanduser().resolve(strict=False)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    create = _docker_run(
        ["docker", "create", "--platform", DOCKER_PLATFORM, image],
        timeout=120,
    )
    container_id = (create.stdout or "").strip()
    if not container_id:
        raise RuntimeError(f"docker create returned no container id for {image}")
    try:
        _docker_run(
            ["docker", "cp", f"{container_id}:{workspace_backend_path}/.", str(root)],
            timeout=600,
        )
    finally:
        _docker_run(["docker", "rm", "-f", container_id], timeout=60, check=False)
    reference_layout = prepare_seeded_workspace(root, workspace_backend_path=workspace_backend_path)
    ensure_git_workspace(root)
    return {
        "image": image,
        "workspace_root": str(root),
        "seeded_from": workspace_backend_path,
        **reference_layout,
    }


def stop_cleanroom_container(container_name: str) -> None:
    _docker_run(["docker", "rm", "-f", container_name], timeout=60, check=False)


def start_cleanroom_container(
    container_name: str,
    image_name: str,
    workspace_root: pathlib.Path,
    *,
    tag: str = IMAGE_TAG,
    cpus: str = "4",
    memory: str = "16g",
    workspace_backend_path: str = "/workspace",
) -> dict[str, Any]:
    image = cleanroom_image_ref(image_name, tag=tag)
    root = pathlib.Path(workspace_root).expanduser().resolve(strict=False)
    stop_cleanroom_container(container_name)
    cmd = [
        "docker",
        "run",
        "-d",
        "--platform",
        DOCKER_PLATFORM,
        "--name",
        container_name,
        "--network",
        "none",
        "-v",
        f"{root}:{workspace_backend_path}",
        "--cpus",
        str(cpus),
        "--memory",
        str(memory),
        "--memory-swap",
        str(memory),
        "--cap-drop",
        "SYS_PTRACE",
        image,
        "sleep",
        "infinity",
    ]
    proc = _docker_run(cmd, timeout=120)
    container_id = (proc.stdout or "").strip()
    preflight = preflight_cleanroom_container(container_name)
    reference_probe = verify_reference_executable_runnable(
        container_name,
        workspace_backend_path=workspace_backend_path,
    )
    return {
        "container_name": container_name,
        "container_id": container_id,
        "image": image,
        "preflight": preflight,
        "reference_probe": reference_probe,
        "workspace_root": str(root),
        "workspace_backend_path": workspace_backend_path,
    }


def ouroboros_api_request(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: int = 30,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ouroboros API {method} {path} failed ({exc.code}): {detail}") from exc
    return json.loads(raw) if raw.strip() else {}


def terminal_task_status(payload: dict[str, Any] | None) -> str:
    """The payload's EXPLICIT status when settled, else ''.

    Terminal detection reads only the task result's declared ``status`` field —
    never heuristics over error/reason text (a prior harness misread a
    completed-on-fallback run as provider_unavailable that way).
    """
    status = str((payload or {}).get("status") or "").strip().lower()
    return status if status in _SETTLED_TASK_STATUSES else ""


def classify_infra_failure(payload: dict[str, Any] | None) -> bool:
    """Harbor-adapter classification: infra-failed results are not capability signals."""
    data = payload or {}
    reason_code = str(data.get("reason_code") or "")
    axes = data.get("outcome_axes") if isinstance(data.get("outcome_axes"), dict) else {}
    execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
    return (
        reason_code == "llm_api_error"
        or str(execution.get("status") or "") == "infra_failed"
        or str(execution.get("reason_code") or "") == "llm_api_error"
    )


def _checkpoint_task_id(path: pathlib.Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("task_id") or "") if isinstance(data, dict) else ""


def _write_checkpoint(path: pathlib.Path | None, task_id: str, payload: dict[str, Any]) -> None:
    """Atomically persist the latest known task state (crash/resume evidence)."""
    if path is None:
        return
    record = {
        "schema": "ouroboros.benchmark.programbench.task_checkpoint.v1",
        "task_id": task_id,
        "ts_unix": time.time(),
        "status": str(payload.get("status") or ""),
        "task_result": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def submit_and_wait(
    base_url: str,
    body: dict[str, Any],
    *,
    timeout_sec: float = 21600.0,
    poll_interval_sec: float = 5.0,
    checkpoint_path: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    """Submit a task via the gateway and poll until an explicit settled status.

    With ``checkpoint_path`` set, every poll atomically persists the latest task
    result, and a restarted harness re-attaches to the recorded task_id instead
    of re-submitting — a crash/timeout no longer discards hours of in-flight
    agent work (a prior 0/5 debug run lost a full 6h task exactly this way). A
    checkpoint whose task_id the server no longer knows (e.g. data root reset)
    falls back to a fresh submission.
    """
    path = pathlib.Path(checkpoint_path) if checkpoint_path is not None else None
    task_id = _checkpoint_task_id(path)
    if task_id:
        try:
            latest = ouroboros_api_request(
                base_url,
                "GET",
                f"/api/tasks/{urllib.parse.quote(task_id)}",
                timeout=60,
            )
        except RuntimeError:
            task_id = ""
        else:
            _write_checkpoint(path, task_id, latest)
            if terminal_task_status(latest):
                return latest
    if not task_id:
        created = ouroboros_api_request(base_url, "POST", "/api/tasks", body)
        task_id = str(created.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"task creation did not return task_id: {created!r}")
        _write_checkpoint(path, task_id, {"status": "submitted"})
    deadline = time.time() + max(1.0, float(timeout_sec))
    latest: dict[str, Any] = {}
    while time.time() < deadline:
        latest = ouroboros_api_request(
            base_url,
            "GET",
            f"/api/tasks/{urllib.parse.quote(task_id)}",
            timeout=60,
        )
        _write_checkpoint(path, task_id, latest)
        if terminal_task_status(latest):
            return latest
        time.sleep(max(0.5, float(poll_interval_sec)))
    raise TimeoutError(f"task {task_id} did not finish within {timeout_sec}s (last status={latest.get('status')!r})")


def run_official_eval(run_root: pathlib.Path) -> dict[str, Any]:
    eval_proc = subprocess.run(programbench_eval_cmd(run_root), capture_output=True, text=True)
    info_proc = subprocess.run(programbench_info_cmd(run_root), capture_output=True, text=True)
    result = {
        "eval": {
            "cmd": programbench_eval_cmd(run_root),
            "returncode": eval_proc.returncode,
            "stdout": eval_proc.stdout,
            "stderr": eval_proc.stderr,
        },
        "info": {
            "cmd": programbench_info_cmd(run_root),
            "returncode": info_proc.returncode,
            "stdout": info_proc.stdout,
            "stderr": info_proc.stderr,
        },
    }
    write_json(pathlib.Path(run_root) / "programbench_eval_result.json", result)
    return result

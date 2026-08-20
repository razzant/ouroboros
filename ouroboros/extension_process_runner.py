"""Out-of-process execution for native-risk extension skills.

The host process may safely catalog and dispatch extensions whose isolated
dependencies include native wheels: plugin import and handler execution happen
in a short-lived child process, so Rust/C aborts cannot take down server.py.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

from starlette.requests import Request
from starlette.responses import FileResponse, Response, StreamingResponse

from ouroboros.provider_models import MODEL_PROVIDER_CREDENTIAL_KEYS
from ouroboros.platform_layer import merge_hidden_kwargs, subprocess_new_group_kwargs
from ouroboros.skill_loader import find_skill, grant_status_for_skill, skill_state_dir
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.shell import _active_subprocesses, _kill_process_group, _subprocess_lock
from ouroboros.tools.skill_exec import _scrub_env
from ouroboros.usage_accounting import current_usage_scope, record_unmetered_external_dispatch
from ouroboros.utils import sanitize_tool_result_for_log

_NATIVE_SUFFIXES = {".so", ".pyd", ".dylib", ".dll"}
_STDOUT_CAP = 512 * 1024
_STDERR_CAP = 128 * 1024
_INPUT_CAP = 1024 * 1024
_RESULT_CAP = 512 * 1024
_CATALOG_TIMEOUT_SEC = 30
_RUNTIME_MODE_ENV_KEYS = ("OUROBOROS_BOOT_RUNTIME_MODE", "OUROBOROS_RUNTIME_MODE")
_POSIX_SIGNAL_NAMES = {
    1: "SIGHUP",
    2: "SIGINT",
    3: "SIGQUIT",
    6: "SIGABRT",
    9: "SIGKILL",
    11: "SIGSEGV",
    13: "SIGPIPE",
    14: "SIGALRM",
    15: "SIGTERM",
}
_POSIX_SIGABRT = 6


class ExtensionProcessError(RuntimeError):
    """A child extension process failed without crashing the host."""

    def __init__(self, message: str, *, failure_kind: str = "error") -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


def _format_child_returncode(returncode: int) -> str:
    """Render child deaths in operator-readable form without trusting stderr."""

    try:
        code = int(returncode)
    except (TypeError, ValueError):
        return f"returncode={returncode}"
    if code < 0:
        signum = -code
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = ""
        if not sig_name or re.fullmatch(r"SIG\d+", sig_name):
            sig_name = _POSIX_SIGNAL_NAMES.get(signum, sig_name or f"SIG{signum}")
        return f"signal={sig_name}({signum}), returncode={code}"
    if code >= 128:
        signum = code - 128
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = ""
        if not sig_name or re.fullmatch(r"SIG\d+", sig_name):
            sig_name = _POSIX_SIGNAL_NAMES.get(signum, sig_name or f"SIG{signum}")
        if sig_name:
            return f"signal={sig_name}({signum}), returncode={code}"
    return f"returncode={code}"


def _quiet_python_abort() -> None:
    """Terminate a macOS extension child without asking CrashReporter for a dialog."""

    try:
        sys.stderr.write("Ouroboros extension child intercepted os.abort(); exiting quietly with code 134.\n")
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(134)


def _quiet_sigabrt(signum, _frame) -> None:
    """Exit from child SIGABRT handlers instead of letting macOS show a crash dialog."""

    try:
        raw_signum = int(signum or signal.SIGABRT)
    except Exception:
        raw_signum = _POSIX_SIGABRT
    exit_signum = _POSIX_SIGABRT if raw_signum == int(signal.SIGABRT) else raw_signum
    try:
        sys.stderr.write(f"Ouroboros extension child intercepted SIGABRT({raw_signum}); exiting quietly.\n")
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(128 + exit_signum)


def _bootstrap_quiet_child_crash_reporting() -> Dict[str, Any]:
    """Best-effort macOS child-only crash UX guard before plugin import."""

    status: Dict[str, Any] = {"enabled": False, "platform": sys.platform, "actions": [], "warnings": []}
    if sys.platform != "darwin" or os.environ.get("OUROBOROS_EXTENSION_PROCESS_CHILD") != "1":
        return status
    status["enabled"] = True
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        status["actions"].append("disable_core_dumps")
    except Exception as exc:
        status["warnings"].append(f"core_dump_limit_failed:{type(exc).__name__}")
    try:
        signal.signal(signal.SIGABRT, _quiet_sigabrt)
        status["actions"].append("quiet_sigabrt_handler")
    except Exception as exc:
        status["warnings"].append(f"sigabrt_handler_failed:{type(exc).__name__}")
    try:
        os.abort = _quiet_python_abort  # type: ignore[method-assign]
        status["actions"].append("quiet_python_os_abort")
    except Exception as exc:
        status["warnings"].append(f"os_abort_patch_failed:{type(exc).__name__}")
    if status["warnings"]:
        try:
            sys.stderr.write(
                "Ouroboros extension child quiet-crash bootstrap warning: "
                + ", ".join(status["warnings"])
                + "\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
    return status


def extension_has_native_deps(skill_dir: pathlib.Path) -> bool:
    """Return True if a skill payload or isolated env contains native modules."""

    root = pathlib.Path(skill_dir)
    try:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in _NATIVE_SUFFIXES:
                return True
    except OSError:
        return False
    return False


def extension_requires_process_isolation(skill_dir: pathlib.Path, dependency_site_dirs_enabled: bool) -> bool:
    """Policy hook for native-risk extension isolation."""

    return bool(dependency_site_dirs_enabled) or extension_has_native_deps(skill_dir)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(v) for v in value]
        return str(value)


def _write_child_result(payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    raw_result_path = str(payload.get("result_path") or "")
    if not raw_result_path:
        return
    result_path = pathlib.Path(raw_result_path)
    _write_private_json(result_path, result, cap_bytes=_RESULT_CAP)


def _ensure_private_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _write_private_json(path: pathlib.Path, payload: Dict[str, Any], *, cap_bytes: int | None = None, overflow_error: str | None = None) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if cap_bytes is not None and len(data) > cap_bytes:
        if overflow_error:
            raise ExtensionProcessError(overflow_error)
        data = json.dumps(
            {"ok": False, "error": "extension child protocol result exceeded safety cap"},
            ensure_ascii=False,
        ).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fd = -1
            fh.write(data)
    finally:
        if fd >= 0:
            os.close(fd)


def _tool_context_payload(ctx: ToolContext) -> Dict[str, Any]:
    return {
        "task_id": str(ctx.task_id or ""),
        "current_chat_id": ctx.current_chat_id,
        "current_task_type": str(ctx.current_task_type or ""),
        "drive_root": str(ctx.drive_root or ""),
        "workspace_root": str(ctx.workspace_root or ""),
        "workspace_mode": str(ctx.workspace_mode or ""),
        "memory_mode": str(ctx.memory_mode or ""),
        "budget_drive_root": str(getattr(ctx, "budget_drive_root", "") or ""),
        "project_id": str(getattr(ctx, "project_id", "") or ""),
        "task_depth": int(ctx.task_depth or 0),
        "task_metadata": _json_safe(dict(ctx.task_metadata or {})),
        "task_contract": _json_safe(dict(getattr(ctx, "task_contract", {}) or {})),
    }


def _apply_tool_context_payload(ctx: ToolContext, payload: Dict[str, Any]) -> ToolContext:
    ctx.task_id = str(payload.get("task_id") or "") or None
    raw_chat_id = payload.get("current_chat_id")
    ctx.current_chat_id = raw_chat_id if isinstance(raw_chat_id, int) else None
    ctx.current_task_type = str(payload.get("current_task_type") or "") or None
    raw_drive_root = str(payload.get("drive_root") or "")
    if raw_drive_root:
        ctx.drive_root = pathlib.Path(raw_drive_root)
    workspace_root = str(payload.get("workspace_root") or "")
    ctx.workspace_root = pathlib.Path(workspace_root) if workspace_root else None
    ctx.workspace_mode = str(payload.get("workspace_mode") or "")
    ctx.memory_mode = str(payload.get("memory_mode") or "")
    ctx.budget_drive_root = str(payload.get("budget_drive_root") or "")
    ctx.project_id = str(payload.get("project_id") or "")
    try:
        ctx.task_depth = int(payload.get("task_depth") or 0)
    except (TypeError, ValueError):
        ctx.task_depth = 0
    task_metadata = payload.get("task_metadata")
    ctx.task_metadata = dict(task_metadata) if isinstance(task_metadata, dict) else {}
    task_contract = payload.get("task_contract")
    ctx.task_contract = dict(task_contract) if isinstance(task_contract, dict) else {}
    return ctx


def _child_python() -> str:
    return sys.executable


def _child_env(
    *,
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    skill_name: str,
    skill_dir: pathlib.Path,
    env_allowlist: List[str],
    granted_keys: List[str],
) -> Dict[str, str]:
    env = _scrub_env(env_allowlist, skill_state_dir(drive_root, skill_name), skill_name, granted_keys=granted_keys)
    env["OUROBOROS_DATA_DIR"] = str(drive_root)
    env["OUROBOROS_REPO_DIR"] = str(repo_dir)
    env["OUROBOROS_EXTENSION_PROCESS_CHILD"] = "1"
    for key in _RUNTIME_MODE_ENV_KEYS:
        if os.environ.get(key):
            env[key] = str(os.environ[key])
    # WA6: carry bytecode suppression into the scrubbed child env so an extension
    # subprocess running the embedded python never writes __pycache__/*.pyc into a
    # signed+notarized macOS .app bundle (which would break the codesign seal).
    for key in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"):
        if os.environ.get(key):
            env[key] = str(os.environ[key])
    pythonpath = str(repo_dir)
    if env.get("PYTHONPATH"):
        pythonpath = os.pathsep.join([pythonpath, env["PYTHONPATH"]])
    env["PYTHONPATH"] = pythonpath
    env["PYTHONUNBUFFERED"] = "1"
    # Host Service loopback access so an out-of-process child/companion can relay
    # WS progress (send_ws_message) and subscribe to host events. Reserved and
    # non-overridable by the skill; token is per-skill, content-hash bound.
    try:
        from ouroboros.extension_loader import mint_skill_token
        from ouroboros.gateway.host_service import DEFAULT_HOST_SERVICE_HOST, host_service_port

        token = mint_skill_token(skill_state_dir(drive_root, skill_name), skill_name, skill_dir)
        if token:
            env["HOST_SERVICE_TOKEN"] = token
            env["HOST_SERVICE_URL"] = f"http://{DEFAULT_HOST_SERVICE_HOST}:{host_service_port()}"
    except Exception:
        pass
    return env


def _drain(pipe: Any, cap: int, out: bytearray, overflow: Dict[str, bool], label: str) -> None:
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                return
            remaining = cap - len(out)
            if remaining <= 0:
                overflow[label] = True
                return
            out.extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow[label] = True
                return
    except (OSError, ValueError):
        return


def _run_child(
    payload: Dict[str, Any],
    *,
    skill_dir: pathlib.Path,
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    env: Dict[str, str],
    timeout_sec: int,
    on_spawn: Callable[[], None] | None = None,
) -> Dict[str, Any]:
    calls_dir = skill_state_dir(drive_root, str(payload.get("skill_name") or "")) / "extension_calls"
    _ensure_private_dir(calls_dir)
    input_path = calls_dir / f"{uuid.uuid4().hex}.json"
    result_path = calls_dir / f"{uuid.uuid4().hex}.result.json"
    import_root_base = calls_dir / f"{uuid.uuid4().hex}.imports"
    payload = dict(payload)
    payload["result_path"] = str(result_path)
    _write_private_json(
        input_path,
        payload,
        cap_bytes=_INPUT_CAP,
        overflow_error="extension child protocol input exceeded safety cap",
    )
    env = dict(env)
    env["OUROBOROS_EXTENSION_IMPORT_ROOT_BASE"] = str(import_root_base)
    cmd = [_child_python(), "-m", "ouroboros.extension_process_runner", str(input_path)]
    kwargs: Dict[str, Any] = {
        "cwd": str(repo_dir),
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    kwargs.update(subprocess_new_group_kwargs())
    proc = subprocess.Popen(cmd, **merge_hidden_kwargs(kwargs))  # noqa: S603 - argv is host-constructed
    with _subprocess_lock:
        _active_subprocesses.add(proc)
    try:
        if on_spawn is not None:
            on_spawn()
    except BaseException:
        # Popen has already dispatched the child.  A failed durable disclosure
        # must stop it rather than leave an untracked external execution.
        try:
            _kill_process_group(proc)
            proc.wait(timeout=2)
        except Exception:
            pass
        with _subprocess_lock:
            _active_subprocesses.discard(proc)
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe:
                    pipe.close()
            except OSError:
                pass
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass
        shutil.rmtree(import_root_base, ignore_errors=True)
        raise
    stdout = bytearray()
    stderr = bytearray()
    overflow = {"stdout": False, "stderr": False}
    out_thread = threading.Thread(target=_drain, args=(proc.stdout, _STDOUT_CAP, stdout, overflow, "stdout"), daemon=True)
    err_thread = threading.Thread(target=_drain, args=(proc.stderr, _STDERR_CAP, stderr, overflow, "stderr"), daemon=True)
    out_thread.start()
    err_thread.start()
    deadline = time.monotonic() + max(1, int(timeout_sec))
    try:
        while proc.poll() is None:
            if overflow["stdout"] or overflow["stderr"]:
                _kill_process_group(proc)
                raise ExtensionProcessError("extension child output exceeded safety cap")
            if time.monotonic() >= deadline:
                _kill_process_group(proc)
                raise ExtensionProcessError(
                    f"extension child timed out after {timeout_sec}s",
                    failure_kind="timeout",
                )
            time.sleep(0.05)
        out_thread.join(timeout=2)
        err_thread.join(timeout=2)
        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            safe_stderr = sanitize_tool_result_for_log(stderr_text)[-2000:] if stderr_text else ""
            code_detail = _format_child_returncode(int(proc.returncode or 0))
            detail = f"{code_detail}; {safe_stderr}" if safe_stderr else code_detail
            raise ExtensionProcessError(f"extension child exited abnormally: {detail}")
        if not result_path.exists():
            raise ExtensionProcessError("extension child did not write protocol result")
        if result_path.stat().st_size > _RESULT_CAP:
            raise ExtensionProcessError("extension child protocol result exceeded safety cap")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise ExtensionProcessError(f"extension child returned invalid JSON: {exc}") from exc
        if not result.get("ok", False):
            raise ExtensionProcessError(str(result.get("error") or "extension child failed"))
        return dict(result)
    finally:
        try:
            if proc.poll() is None:
                _kill_process_group(proc)
            proc.wait(timeout=2)
        except Exception:
            pass
        with _subprocess_lock:
            _active_subprocesses.discard(proc)
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass
        shutil.rmtree(import_root_base, ignore_errors=True)
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe:
                    pipe.close()
            except OSError:
                pass


def _record_extension_dispatch(
    *,
    dispatch_id: str,
    drive_root: pathlib.Path,
    skill_name: str,
    surface_kind: str,
    surface: str,
    ctx: ToolContext | None = None,
) -> str:
    """Disclose one spawned extension child outside core model metering."""

    bound = current_usage_scope()
    meta = getattr(ctx, "task_metadata", {}) if ctx is not None else {}
    meta = meta if isinstance(meta, dict) else {}
    task_id = str(
        (getattr(ctx, "task_id", "") if ctx is not None else "")
        or meta.get("task_id")
        or meta.get("subagent_task_id")
        or (bound.task_id if bound is not None else "")
        or ""
    )
    system_root = f"extension:{skill_name}"
    root_task_id = str(
        meta.get("root_task_id")
        or (getattr(ctx, "root_task_id", "") if ctx is not None else "")
        or (bound.root_task_id if bound is not None else "")
        or task_id
        or system_root
    )
    if not task_id:
        task_id = root_task_id
    parent_task_id = str(
        meta.get("parent_task_id")
        or (getattr(ctx, "parent_task_id", "") if ctx is not None else "")
        or (bound.parent_task_id if bound is not None else "")
        or ""
    )
    return record_unmetered_external_dispatch(
        dispatch_id,
        drive_root=pathlib.Path(drive_root).resolve(strict=False),
        provider="external-extension",
        task_id=task_id,
        root_task_id=root_task_id,
        parent_task_id=parent_task_id,
        category="external_skill",
        source=f"extension_{surface_kind}:{skill_name}:{surface}",
    )


def disclose_inprocess_extension_dispatch(
    spec: Dict[str, Any],
    *,
    drive_root: pathlib.Path,
    surface_kind: str,
    surface: str,
    ctx: ToolContext | None = None,
) -> str:
    """Record one admitted in-process handler that can bypass core metering."""
    probe = spec.get("_model_credential_probe")
    if not callable(probe) or not probe():
        return ""
    return _record_extension_dispatch(
        dispatch_id=f"extension:{surface_kind}:{uuid.uuid4().hex}",
        drive_root=pathlib.Path(drive_root),
        skill_name=str(spec.get("skill") or "unknown"),
        surface_kind=surface_kind,
        surface=surface,
        ctx=ctx,
    )


def _base_env_for_skill(skill: Any, drive_root: pathlib.Path, repo_dir: pathlib.Path) -> Dict[str, str]:
    return _child_env(
        drive_root=drive_root,
        repo_dir=repo_dir,
        skill_name=skill.name,
        skill_dir=skill.skill_dir,
        env_allowlist=[],
        granted_keys=[],
    )


def _extension_has_model_credentials(skill: Any, drive_root: pathlib.Path) -> bool:
    """Whether an OOP extension can actually read a funded model credential."""
    grants = grant_status_for_skill(pathlib.Path(drive_root), skill)
    granted = {str(key).strip().upper() for key in (grants.get("granted_keys") or [])}
    manifest = getattr(skill, "manifest", None)
    permissions = {str(item).strip() for item in (getattr(manifest, "permissions", None) or [])}
    allowed = {
        str(item).strip().upper()
        for item in (getattr(manifest, "env_from_settings", None) or [])
        if str(item).strip()
    }
    candidates = granted & allowed & MODEL_PROVIDER_CREDENTIAL_KEYS
    if "read_settings" not in permissions or not candidates:
        return False
    from ouroboros.config import load_settings

    settings = load_settings()
    return any(str(settings.get(key) or "").strip() for key in candidates)


def catalog_extension_surfaces(skill: Any, *, drive_root: pathlib.Path, repo_dir: pathlib.Path, skills_repo_path: pathlib.Path | None = None) -> Dict[str, Any]:
    env = _base_env_for_skill(skill, pathlib.Path(drive_root), pathlib.Path(repo_dir))
    model_capable = _extension_has_model_credentials(skill, pathlib.Path(drive_root))
    dispatch_id = f"extension:catalog:{uuid.uuid4().hex}"
    return _run_child(
        {
            "mode": "catalog",
            "skill_name": skill.name,
            "drive_root": str(drive_root),
            "repo_dir": str(repo_dir),
            "skills_repo_path": str(skills_repo_path or skill.skill_dir.parent),
        },
        skill_dir=skill.skill_dir,
        drive_root=pathlib.Path(drive_root),
        repo_dir=pathlib.Path(repo_dir),
        env=env,
        timeout_sec=_CATALOG_TIMEOUT_SEC,
        on_spawn=(
            lambda: _record_extension_dispatch(
                dispatch_id=dispatch_id,
                drive_root=pathlib.Path(drive_root),
                skill_name=skill.name,
                surface_kind="catalog",
                surface="register",
            )
        ) if model_capable else None,
    )


def dispatch_extension_tool_subprocess(ext_tool: Dict[str, Any], ctx: ToolContext, args: Dict[str, Any]) -> str:
    meta = getattr(ctx, "task_metadata", {})
    bound = current_usage_scope()
    dispatch_drive_root = pathlib.Path(
        (meta.get("budget_drive_root") if isinstance(meta, dict) else "")
        or getattr(ctx, "budget_drive_root", "")
        or (bound.drive_root if bound is not None else "")
        or getattr(ctx, "drive_root", "")
        or "."
    ).resolve(strict=False)
    skill = _skill_for_dispatch(
        str(ext_tool.get("skill") or ""),
        dispatch_drive_root,
        pathlib.Path(str(ext_tool.get("skills_repo_path") or ctx.repo_dir)),
    )
    env = _base_env_for_skill(skill, dispatch_drive_root, pathlib.Path(ctx.repo_dir))
    model_capable = _extension_has_model_credentials(skill, dispatch_drive_root)
    dispatch_id = f"extension:tool:{uuid.uuid4().hex}"
    result = _run_child(
        {
            "mode": "tool",
            "skill_name": skill.name,
            "surface": str(ext_tool.get("name") or ""),
            "args": dict(args or {}),
            "ctx": _tool_context_payload(ctx),
            "drive_root": str(dispatch_drive_root),
            "repo_dir": str(ctx.repo_dir),
            "skills_repo_path": str(ext_tool.get("skills_repo_path") or ctx.repo_dir),
        },
        skill_dir=skill.skill_dir,
        drive_root=dispatch_drive_root,
        repo_dir=pathlib.Path(ctx.repo_dir),
        env=env,
        timeout_sec=max(1, int(ext_tool.get("timeout_sec") or 60)),
        on_spawn=((
            lambda: _record_extension_dispatch(
                dispatch_id=dispatch_id,
                drive_root=dispatch_drive_root,
                skill_name=skill.name,
                surface_kind="tool",
                surface=str(ext_tool.get("name") or ""),
                ctx=ctx,
            )
        ) if model_capable else None),
    )
    return str(result.get("result") or "")


def dispatch_extension_route_subprocess(spec: Dict[str, Any], request_payload: Dict[str, Any], *, drive_root: pathlib.Path, repo_dir: pathlib.Path) -> Dict[str, Any]:
    skills_repo_path = pathlib.Path(str(spec.get("skills_repo_path") or repo_dir))
    skill = _skill_for_dispatch(str(spec.get("skill") or ""), pathlib.Path(drive_root), skills_repo_path)
    env = _base_env_for_skill(skill, pathlib.Path(drive_root), pathlib.Path(repo_dir))
    model_capable = _extension_has_model_credentials(skill, pathlib.Path(drive_root))
    dispatch_id = f"extension:route:{uuid.uuid4().hex}"
    return _run_child(
        {
            "mode": "route",
            "skill_name": skill.name,
            "surface": str(spec.get("path") or ""),
            "request": request_payload,
            "drive_root": str(drive_root),
            "repo_dir": str(repo_dir),
            "skills_repo_path": str(skills_repo_path),
        },
        skill_dir=skill.skill_dir,
        drive_root=pathlib.Path(drive_root),
        repo_dir=pathlib.Path(repo_dir),
        env=env,
        timeout_sec=max(1, int(spec.get("timeout_sec") or 60)),
        on_spawn=((
            lambda: _record_extension_dispatch(
                dispatch_id=dispatch_id,
                drive_root=pathlib.Path(drive_root),
                skill_name=skill.name,
                surface_kind="route",
                surface=str(spec.get("path") or ""),
            )
        ) if model_capable else None),
    )


def dispatch_extension_ws_subprocess(spec: Dict[str, Any], msg: Dict[str, Any], *, drive_root: pathlib.Path, repo_dir: pathlib.Path) -> Any:
    skills_repo_path = pathlib.Path(str(spec.get("skills_repo_path") or repo_dir))
    skill = _skill_for_dispatch(str(spec.get("skill") or ""), pathlib.Path(drive_root), skills_repo_path)
    env = _base_env_for_skill(skill, pathlib.Path(drive_root), pathlib.Path(repo_dir))
    model_capable = _extension_has_model_credentials(skill, pathlib.Path(drive_root))
    dispatch_id = f"extension:ws:{uuid.uuid4().hex}"
    result = _run_child(
        {
            "mode": "ws",
            "skill_name": skill.name,
            "surface": str(spec.get("type") or ""),
            "message": dict(msg or {}),
            "drive_root": str(drive_root),
            "repo_dir": str(repo_dir),
            "skills_repo_path": str(skills_repo_path),
        },
        skill_dir=skill.skill_dir,
        drive_root=pathlib.Path(drive_root),
        repo_dir=pathlib.Path(repo_dir),
        env=env,
        timeout_sec=max(1, int(spec.get("timeout_sec") or 60)),
        on_spawn=((
            lambda: _record_extension_dispatch(
                dispatch_id=dispatch_id,
                drive_root=pathlib.Path(drive_root),
                skill_name=skill.name,
                surface_kind="ws",
                surface=str(spec.get("type") or ""),
            )
        ) if model_capable else None),
    )
    return result.get("result")


def _skill_for_dispatch(skill_name: str, drive_root: pathlib.Path, skills_repo_path: pathlib.Path) -> Any:
    skill = find_skill(drive_root, skill_name, repo_path=str(skills_repo_path))
    if skill is None:
        raise ExtensionProcessError(f"extension skill {skill_name!r} is missing")
    return skill


def _load_child_extension(skill_name: str, drive_root: pathlib.Path, repo_dir: pathlib.Path, skills_repo_path: pathlib.Path) -> None:
    from ouroboros.config import load_settings
    from ouroboros.extension_loader import load_extension
    from ouroboros.skill_loader import discover_skills

    skills = discover_skills(drive_root, repo_path=str(skills_repo_path))
    skill = next((item for item in skills if item.name == skill_name), None)
    if skill is None:
        raise ExtensionProcessError(f"extension skill {skill_name!r} is missing")
    err = load_extension(
        skill,
        load_settings,
        drive_root=drive_root,
        skills=skills,
        repo_path=str(skills_repo_path),
        _force_in_process=True,
    )
    if err:
        raise ExtensionProcessError(err)


def _surface_catalog() -> Dict[str, Any]:
    from ouroboros.extension_loader import get_tool, list_companion_names, list_routes, list_ws_handlers, snapshot

    snap = snapshot()
    return {
        "companions": list_companion_names(),
        "tools": [
            {
                key: _json_safe(value)
                for key, value in (get_tool(name) or {}).items()
                if key != "handler"
            }
            for name in snap.get("tools", [])
        ],
        "routes": [
            {
                key: _json_safe(value)
                for key, value in spec.items()
                if key != "handler"
            }
            for spec in list_routes().values()
        ],
        "ws_handlers": [
            {
                key: _json_safe(value)
                for key, value in spec.items()
                if key != "handler"
            }
            for spec in list_ws_handlers().values()
        ],
        "ui_tabs": snap.get("ui_tabs", []),
        "settings_sections": snap.get("settings_sections", []),
    }


async def _run_maybe_async(value: Any) -> Any:
    if inspect.iscoroutine(value):
        return await value
    return value


async def _call_tool(surface: str, args: Dict[str, Any], drive_root: pathlib.Path, repo_dir: pathlib.Path, ctx_payload: Dict[str, Any] | None = None) -> Any:
    from ouroboros.extension_loader import get_tool

    tool = get_tool(surface)
    if not tool or not callable(tool.get("handler")):
        raise ExtensionProcessError(f"extension tool {surface!r} is not registered")
    handler = tool["handler"]
    ctx = _apply_tool_context_payload(
        ToolContext(repo_dir=repo_dir, drive_root=drive_root),
        dict(ctx_payload or {}),
    )
    # ctx calling-convention from the descriptor (decided on the RAW handler at
    # register time); fall back to inspecting the unwrapped handler for legacy
    # tools registered before the flag existed.
    _wants = tool.get("wants_ctx")
    if _wants is None:
        _wants = _handler_wants_ctx(inspect.unwrap(handler))
    result = (
        handler(ctx, **dict(args or {}))
        if _wants
        else handler(**dict(args or {}))
    )
    return await _run_maybe_async(result)


def _handler_wants_ctx(handler: Any) -> bool:
    """True when the handler's first parameter is a ctx slot (canonical form).

    Extension tool handlers are either ``fn(ctx, **args)`` (canonical) or
    ``fn(**args)`` / ``fn(named=...)`` (ctx-less). Dispatch on the first
    parameter's name so the args dict can still bind named parameters.
    """
    import inspect

    try:
        params = list(inspect.signature(handler).parameters.values())
    except (TypeError, ValueError):
        return True  # builtins/C callables: keep the historical ctx-first call
    if not params:
        return False
    first = params[0]
    if first.kind == first.VAR_POSITIONAL:
        return True
    if first.kind in (first.POSITIONAL_ONLY, first.POSITIONAL_OR_KEYWORD):
        return first.name in {"ctx", "context", "_ctx", "tool_context"}
    return False


async def _request_from_payload(payload: Dict[str, Any], drive_root: pathlib.Path, repo_dir: pathlib.Path) -> Request:
    body = base64.b64decode(str(payload.get("body_b64") or ""))
    sent = False

    async def receive() -> Dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    headers = [
        (str(k).lower().encode("latin-1", errors="ignore"), str(v).encode("latin-1", errors="ignore"))
        for k, v in (payload.get("headers") or [])
    ]
    scope = {
        "type": "http",
        "method": str(payload.get("method") or "GET").upper(),
        "path": str(payload.get("path") or "/"),
        "query_string": str(payload.get("query_string") or "").encode("utf-8"),
        "headers": headers,
        "path_params": dict(payload.get("path_params") or {}),
        "app": SimpleNamespace(state=SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir)),
        "scheme": "http",
        "server": ("127.0.0.1", 0),
        "client": ("127.0.0.1", 0),
    }
    return Request(scope, receive)


async def _response_to_payload(response: Response, scope: Dict[str, Any]) -> Dict[str, Any]:
    started: Dict[str, Any] = {
        "status_code": int(getattr(response, "status_code", 200) or 200),
        "headers": dict(getattr(response, "headers", {}) or {}),
    }
    body = bytearray()
    received = False

    async def receive() -> Dict[str, Any]:
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Dict[str, Any]) -> None:
        msg_type = str(message.get("type") or "")
        if msg_type == "http.response.start":
            started["status_code"] = int(message.get("status") or started["status_code"])
            headers = {}
            for raw_key, raw_value in message.get("headers") or []:
                key = bytes(raw_key).decode("latin-1", errors="ignore")
                value = bytes(raw_value).decode("latin-1", errors="ignore")
                if key.lower() != "content-length":
                    headers[key] = value
            started["headers"] = headers
        elif msg_type == "http.response.body":
            chunk = bytes(message.get("body") or b"")
            if len(body) + len(chunk) > _RESULT_CAP:
                raise ExtensionProcessError("extension child response body exceeded safety cap")
            body.extend(chunk)

    await response(scope, receive, send)
    headers = dict(started.get("headers") or {})
    headers.pop("content-length", None)
    return {
        "kind": "response",
        "status_code": int(started.get("status_code") or 200),
        "headers": headers,
        "media_type": getattr(response, "media_type", None),
        "body_b64": base64.b64encode(bytes(body)).decode("ascii"),
    }


async def _streaming_response_to_payload(response: StreamingResponse) -> Dict[str, Any]:
    body = bytearray()
    async for chunk in response.body_iterator:
        if isinstance(chunk, str):
            chunk = chunk.encode(getattr(response, "charset", "utf-8") or "utf-8")
        chunk = bytes(chunk or b"")
        if len(body) + len(chunk) > _RESULT_CAP:
            raise ExtensionProcessError("extension child response body exceeded safety cap")
        body.extend(chunk)
    headers = dict(response.headers)
    headers.pop("content-length", None)
    return {
        "kind": "response",
        "status_code": int(response.status_code),
        "headers": headers,
        "media_type": getattr(response, "media_type", None),
        "body_b64": base64.b64encode(bytes(body)).decode("ascii"),
    }


def _file_response_to_payload(response: FileResponse) -> Dict[str, Any]:
    path = pathlib.Path(response.path)
    if path.stat().st_size > _RESULT_CAP:
        raise ExtensionProcessError("extension child response body exceeded safety cap")
    headers = dict(response.headers)
    headers.pop("content-length", None)
    return {
        "kind": "response",
        "status_code": int(response.status_code),
        "headers": headers,
        "media_type": getattr(response, "media_type", None),
        "body_b64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


async def _call_route(surface: str, request_payload: Dict[str, Any], drive_root: pathlib.Path, repo_dir: pathlib.Path) -> Dict[str, Any]:
    from ouroboros.extension_loader import list_routes

    spec = list_routes().get(surface)
    if not spec or not callable(spec.get("handler")):
        raise ExtensionProcessError(f"extension route {surface!r} is not registered")
    request = await _request_from_payload(request_payload, drive_root, repo_dir)
    result = spec["handler"](request)
    result = await _run_maybe_async(result)
    if isinstance(result, FileResponse):
        return _file_response_to_payload(result)
    if isinstance(result, StreamingResponse):
        return await _streaming_response_to_payload(result)
    if isinstance(result, Response):
        return await _response_to_payload(result, request.scope)
    if isinstance(result, (dict, list)):
        return {"kind": "json", "status_code": 200, "data": _json_safe(result)}
    return {"kind": "text", "status_code": 200, "text": str(result)}


async def _call_ws(surface: str, msg: Dict[str, Any]) -> Any:
    from ouroboros.extension_loader import list_ws_handlers

    spec = list_ws_handlers().get(surface)
    if not spec or not callable(spec.get("handler")):
        raise ExtensionProcessError(f"extension WS handler {surface!r} is not registered")
    return await _run_maybe_async(spec["handler"](dict(msg or {})))


def _child_main(input_path: str) -> None:
    payload = json.loads(pathlib.Path(input_path).read_text(encoding="utf-8"))
    drive_root = pathlib.Path(payload["drive_root"])
    repo_dir = pathlib.Path(payload["repo_dir"])
    skills_repo_path = pathlib.Path(payload.get("skills_repo_path") or repo_dir)
    skill_name = str(payload["skill_name"])
    _bootstrap_quiet_child_crash_reporting()
    try:
        _load_child_extension(skill_name, drive_root, repo_dir, skills_repo_path)
        mode = str(payload.get("mode") or "")
        if mode == "catalog":
            result = _surface_catalog()
        elif mode == "tool":
            result = {"result": _json_safe(asyncio.run(_call_tool(str(payload.get("surface") or ""), dict(payload.get("args") or {}), drive_root, repo_dir, dict(payload.get("ctx") or {}))))}
        elif mode == "route":
            result = {"route": asyncio.run(_call_route(str(payload.get("surface") or ""), dict(payload.get("request") or {}), drive_root, repo_dir))}
        elif mode == "ws":
            result = {"result": _json_safe(asyncio.run(_call_ws(str(payload.get("surface") or ""), dict(payload.get("message") or {}))))}
        else:
            raise ExtensionProcessError(f"unknown extension child mode {mode!r}")
        _write_child_result(payload, {"ok": True, **result})
    except BaseException as exc:
        _write_child_result(payload, {"ok": False, "error": sanitize_tool_result_for_log(f"{type(exc).__name__}: {exc}")})
        raise SystemExit(0)
    finally:
        try:
            from ouroboros.extension_loader import unload_extension

            unload_extension(skill_name)
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m ouroboros.extension_process_runner <payload.json>")
    try:
        from ouroboros.process_custody import start_parent_lifeline

        start_parent_lifeline(label="extension-runner")
    except Exception:
        pass
    _child_main(sys.argv[1])

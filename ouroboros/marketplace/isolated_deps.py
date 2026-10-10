"""Per-skill isolated dependency installation helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import pathlib
import shutil
import subprocess
import sys
import threading
import uuid
from typing import Any, Dict, List

from ouroboros.marketplace.install_specs import install_specs_hash, relative_install_path
from ouroboros.skill_loader import skill_state_dir, skill_state_dir_path
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso, replace_atomic
from ouroboros.verified_download import fetch_exact_file, verify_exact_file


ENV_DIRNAME = ".ouroboros_env"
FINGERPRINT_FILENAME = "fingerprint.json"
DEPS_STATE_FILENAME = "deps.json"
_DEFAULT_TIMEOUT_SEC = 600
_INSTALLER_STDERR_TAIL_BYTES = 12_000
# PYTHONDONTWRITEBYTECODE/PYTHONPYCACHEPREFIX (WA6): preserve bytecode suppression
# into the curated installer env so embedded `python -m venv` / `pip install` never
# write stdlib *.pyc back into a signed+notarized macOS .app bundle (which would
# break the codesign seal). Caches land in data/state/pycache via the inherited prefix.
_SAFE_ENV_KEYS = {
    "PATH", "SYSTEMROOT", "LANG", "LC_ALL", "LC_CTYPE",
    "PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX",
}


def isolated_env_dir(skill_dir: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(skill_dir) / ENV_DIRNAME


def isolated_bin_dirs(skill_dir: pathlib.Path) -> List[pathlib.Path]:
    env_root = isolated_env_dir(skill_dir)
    candidates = [
        env_root / "bin",
        env_root / "python" / ("Scripts" if os.name == "nt" else "bin"),
        env_root / "node" / "node_modules" / ".bin",
        env_root / "cargo" / "bin",
    ]
    return [path for path in candidates if path.exists()]


def python_runtime_binary(skill_dir: pathlib.Path) -> pathlib.Path | None:
    bin_dir = isolated_env_dir(skill_dir) / "python" / ("Scripts" if os.name == "nt" else "bin")
    candidate = bin_dir / ("python.exe" if os.name == "nt" else "python")
    return candidate if candidate.is_file() else None


def augment_env_for_skill_deps(env: Dict[str, str], skill_dir: pathlib.Path) -> Dict[str, str]:
    out = dict(env)
    env_root = isolated_env_dir(skill_dir)
    bins = [str(path) for path in isolated_bin_dirs(skill_dir)]
    if bins:
        current = out.get("PATH", "")
        out["PATH"] = os.pathsep.join([*bins, current]) if current else os.pathsep.join(bins)
    python_bin = python_runtime_binary(skill_dir)
    if python_bin:
        out["VIRTUAL_ENV"] = str(python_bin.parent.parent)
    node_modules = env_root / "node" / "node_modules"
    if node_modules.is_dir():
        out["NODE_PATH"] = str(node_modules)
    return out


def _installer_env(env_root: pathlib.Path, *, ecosystem: str = "", cache_root: pathlib.Path | None = None) -> Dict[str, str]:
    tmp_dir = env_root / "tmp"
    home_dir = env_root / "home"
    cache_dir = cache_root or env_root / "cache"
    for path in (tmp_dir, home_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    env = {key: os.environ[key] for key in _SAFE_ENV_KEYS if key in os.environ}
    env.update({
        "HOME": str(home_dir), "USERPROFILE": str(home_dir),
        "APPDATA": str(home_dir / "AppData" / "Roaming"),
        "LOCALAPPDATA": str(home_dir / "AppData" / "Local"),
        "TMPDIR": str(tmp_dir), "TMP": str(tmp_dir), "TEMP": str(tmp_dir),
        "PYTHONNOUSERSITE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_CACHE_DIR": str(cache_dir / "pip"), "PIP_CONFIG_FILE": os.devnull,
        "npm_config_cache": str(cache_dir / "npm"), "npm_config_userconfig": str(env_root / "npmrc"),
        "CARGO_HOME": str(env_root / "cargo" / "home"), "CARGO_TARGET_DIR": str(env_root / "cargo" / "target"),
    })
    if ecosystem == "node":
        # npm's `#!/usr/bin/env node` shebang must resolve a working runtime.
        # node_runtime owns the emergency verdict, the byte-identical healthy
        # case, and the disclosed residuals (npm itself is not bundled: an
        # absent npm still fails honestly upstream, and an npm launcher with an
        # ABSOLUTE node shebang ignores PATH).
        from ouroboros.platform_layer import prepend_skill_node_emergency_path

        prepend_skill_node_emergency_path(env)
    return env


def _pipe_tail(pipe: Any, out: Dict[str, bytes], key: str, max_bytes: int) -> None:
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            data = pipe.read(4096)
            if not data:
                break
            chunks.append(bytes(data))
            total += len(data)
            while total > max_bytes and chunks:
                extra = total - max_bytes
                if len(chunks[0]) <= extra:
                    total -= len(chunks.pop(0))
                else:
                    chunks[0] = chunks[0][extra:]
                    total -= extra
                    break
    finally:
        try:
            pipe.close()
        except Exception:
            pass
        out[key] = b"".join(chunks)[-max_bytes:]


def _run(cmd: List[str], *, cwd: pathlib.Path, env: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
    from subprocess import Popen
    from ouroboros.platform_layer import merge_hidden_kwargs, subprocess_new_group_kwargs
    from ouroboros.tools.shell import _active_subprocesses, _kill_process_group, _subprocess_lock

    stderr_tail: Dict[str, bytes] = {}
    kwargs: Dict[str, Any] = {
        "cwd": str(cwd),
        "env": env,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
    }
    kwargs.update(subprocess_new_group_kwargs())
    proc = Popen(cmd, **merge_hidden_kwargs(kwargs))  # noqa: S603 - argv template is controlled.
    stderr_thread = None
    if getattr(proc, "stderr", None) is not None:
        stderr_thread = threading.Thread(
            target=_pipe_tail,
            args=(proc.stderr, stderr_tail, "stderr", _INSTALLER_STDERR_TAIL_BYTES),
            daemon=True,
        )
        stderr_thread.start()
    with _subprocess_lock:
        _active_subprocesses.add(proc)
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        raise
    finally:
        with _subprocess_lock:
            _active_subprocesses.discard(proc)
        if stderr_thread is not None:
            stderr_thread.join(timeout=1)
    result = {"cmd": cmd[:2] + ["..."] if len(cmd) > 2 else list(cmd), "returncode": proc.returncode}
    stderr_text = stderr_tail.get("stderr", b"").decode("utf-8", errors="replace").strip()
    if stderr_text:
        result["stderr_tail"] = stderr_text
    return result


def _ensure_python_env(env_root: pathlib.Path, timeout_sec: int) -> pathlib.Path:
    venv_dir = env_root / "python"
    if not venv_dir.exists():
        result = _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=env_root, env=_installer_env(env_root, ecosystem="python"), timeout_sec=timeout_sec)
        if result["returncode"] != 0:
            detail = result.get("stderr_tail") or ""
            raise RuntimeError("python venv creation failed" + (f": {detail}" if detail else ""))
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _install_python_packages(packages: List[str], env_root: pathlib.Path, timeout_sec: int, *,
                             allow_source_build: bool = False, cache_root: pathlib.Path | None = None, review_check: Any = None) -> List[Dict[str, Any]]:
    if not packages:
        return []
    bin_dir = _ensure_python_env(env_root, timeout_sec)
    python_bin = bin_dir / ("python.exe" if os.name == "nt" else "python")
    if review_check is not None:
        review_check()
    flags = [] if allow_source_build else ["--only-binary=:all:"]
    result = _run([str(python_bin), "-m", "pip", "install", *flags, *packages], cwd=env_root,
                  env=_installer_env(env_root, ecosystem="python", cache_root=cache_root), timeout_sec=timeout_sec)
    if result["returncode"] != 0:
        detail = result.get("stderr_tail") or ""
        raise RuntimeError("pip install failed" + (f": {detail}" if detail else ""))
    return [result]


def _install_node_package(package: str, env_root: pathlib.Path, timeout_sec: int, *,
                          allow_install_scripts: bool = False, cache_root: pathlib.Path | None = None, review_check: Any = None) -> List[Dict[str, Any]]:
    from ouroboros.platform_layer import bootstrap_process_path

    # A GUI-launched macOS process starts with a truncated PATH; enrich it the
    # same way the process tools do BEFORE deciding npm is absent (T13).
    bootstrap_process_path()
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is not available on PATH")
    node_root = env_root / "node"
    node_root.mkdir(parents=True, exist_ok=True)
    env = _installer_env(env_root, ecosystem="node", cache_root=cache_root)
    env["npm_config_prefix"] = str(node_root)
    name = package.rsplit("@", 1)[0] if "@" in package[1:] else package
    package_manifest = node_root / "node_modules" / name / "package.json"
    previous_version = (read_json_dict(package_manifest) or {}).get("version")
    if review_check is not None:
        review_check()
    flags = [] if allow_install_scripts else ["--ignore-scripts"]
    result = _run([npm, "install", *flags, "--prefix", str(node_root), package], cwd=env_root, env=env, timeout_sec=timeout_sec)
    if result["returncode"] != 0:
        detail = result.get("stderr_tail") or ""
        raise RuntimeError(f"npm install {package!r} failed" + (f": {detail}" if detail else ""))
    results = [result]
    if allow_install_scripts and previous_version and previous_version == (read_json_dict(package_manifest) or {}).get("version"):
        # An already installed version can have skipped postinstall earlier.
        # npm install calls that version up-to-date; rebuild only the explicitly
        # authorized package, never every package sharing this environment.
        if review_check is not None:
            review_check()
        rebuilt = _run([npm, "rebuild", "--prefix", str(node_root), package], cwd=env_root, env=env, timeout_sec=timeout_sec)
        results.append(rebuilt)
        if rebuilt["returncode"] != 0:
            raise RuntimeError(f"npm rebuild {package!r} failed: {rebuilt.get('stderr_tail', '')}")
    skill_node_modules = env_root.parent / "node_modules"
    target_node_modules = node_root / "node_modules"
    if target_node_modules.exists() and not skill_node_modules.exists():
        try:
            skill_node_modules.symlink_to(target_node_modules, target_is_directory=True)
        except OSError:
            # Some Windows configurations disallow symlinks. PATH + NODE_PATH
            # still cover CommonJS and CLI binaries; ESM import users will see
            # a normal module-resolution error instead of a privileged fallback.
            pass
    return results


def _install_target(env_root: pathlib.Path, relative: str, *, allow_root: bool = False) -> pathlib.Path:
    target = env_root / relative_install_path(relative, allow_root=allow_root)
    if not target.resolve().is_relative_to(env_root.resolve()):
        raise ValueError(f"install path escapes its environment: {relative!r}")
    return target


def _output_facts(env_root: pathlib.Path, relative: str) -> Dict[str, Any]:
    path = _install_target(env_root, relative)
    if not path.is_file():
        raise RuntimeError(f"declared install output is missing: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"path": relative, "sha256": digest.hexdigest(), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino, "device": stat.st_dev}


def _install_download(spec: Dict[str, Any], env_root: pathlib.Path, cache_root: pathlib.Path,
                      timeout_sec: int) -> Dict[str, Any]:
    def verify(path):
        return verify_exact_file(path, size_bytes=spec["size_bytes"], sha256=spec["sha256"],
                                 code_prefix="skill_resource", label="skill resource")
    cache = cache_root / "resources" / spec["sha256"]
    source = fetch_exact_file(url=spec["url"], destination=cache, verify=verify,
                              size_bytes=spec["size_bytes"], overflow_code="skill_resource_size_mismatch",
                              failure_code="skill_resource_download_failed", label="skill resource",
                              timeout_sec=timeout_sec)
    target = _install_target(env_root, spec["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.installing")
    try:
        shutil.copyfile(source, temporary)
        verify(temporary)
        replace_atomic(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"url": spec["url"], "sha256": spec["sha256"], "size_bytes": spec["size_bytes"],
            "target": spec["target"], "version": str(spec.get("version") or ""), "downloaded": True}


def _install_actions(spec: Dict[str, Any], env_root: pathlib.Path, cache_root: pathlib.Path,
                     timeout_sec: int, logs: List[Dict[str, Any]], review_check: Any = None) -> Dict[str, Any]:
    expected = list(spec.get("outputs") or [])
    if spec["kind"] == "download" and spec["target"] not in expected:
        expected.append(spec["target"])
    steps, check = spec.get("steps") or [], spec.get("check")
    if not steps and check is None and not spec.get("bins"):
        return {"outputs": [_output_facts(env_root, p) for p in expected], "executable_ready": None}
    commands = [step["argv"][0] for step in [*steps, *([check] if check is not None else [])]]
    ecosystem = "node" if spec["kind"] in {"node", "npm"} or any(cmd in {"node", "npm"} for cmd in commands) else ""
    base_env = _installer_env(env_root, ecosystem=ecosystem, cache_root=cache_root)
    for index, step in enumerate(steps):
        env = augment_env_for_skill_deps(base_env, env_root.parent)
        if review_check is not None:
            review_check()
        result = _run_install_step(step, env_root, env, timeout_sec)
        logs.append({"kind": "build", "step": index, **result})
        if result["returncode"] != 0:
            raise RuntimeError(f"install step {index} failed: {result.get('stderr_tail', '')}")
    env = augment_env_for_skill_deps(base_env, env_root.parent)
    missing = [name for name in spec.get("bins") or [] if not shutil.which(name, path=env.get("PATH", ""))]
    if missing:
        raise RuntimeError(f"declared install binaries are unavailable: {missing}")
    if check is not None:
        if review_check is not None:
            review_check()
        result = _run_install_step(check, env_root, env, timeout_sec)
        logs.append({"kind": "readiness_check", **result})
        if result["returncode"] != 0:
            raise RuntimeError(f"executable readiness check failed: {result.get('stderr_tail', '')}")
    outputs = [_output_facts(env_root, relative) for relative in expected]
    return {"outputs": outputs, "executable_ready": True if check is not None else None}


def _run_install_step(step: Dict[str, Any], env_root: pathlib.Path, env: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
    cwd = _install_target(env_root, step.get("cwd", "."), allow_root=True)
    cmd = list(step["argv"])
    # Resolve through the dependency PATH on Windows as well as POSIX. All
    # remaining arguments stay literal; no shell expansion or build language.
    if not pathlib.Path(cmd[0]).is_absolute():
        selected = str((cwd / cmd[0]).resolve()) if "/" in cmd[0] or "\\" in cmd[0] else shutil.which(cmd[0], path=env.get("PATH", ""))
        if not selected and cmd[0] in {"python", "python3"}:
            selected = sys.executable  # Same embedded-Python fallback as skill_exec.
        if selected:
            cmd[0] = selected
    return _run(cmd, cwd=cwd, env=env, timeout_sec=timeout_sec)


def _resolved_packages(env_root: pathlib.Path) -> List[Dict[str, Any]]:
    from ouroboros.extension_isolated_deps import _isolated_python_site_dirs

    resolved = []
    for dist in importlib.metadata.distributions(path=[str(p) for p in _isolated_python_site_dirs(env_root.parent)]):
        row = {"kind": "python", "name": str(dist.metadata.get("Name") or ""), "version": dist.version}
        for filename, key in (("RECORD", "record_sha256"), ("METADATA", "metadata_sha256")):
            content = dist.read_text(filename)
            if content is not None:
                row[key] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        direct = dist.read_text("direct_url.json")
        if direct:
            row["direct_url"] = json.loads(direct)
        resolved.append(row)
    lock = read_json_dict(env_root / "node" / "package-lock.json") or {}
    for path, package in (lock.get("packages") or {}).items():
        if path and isinstance(package, dict):
            resolved.append({"kind": "npm", "path": path, **{key: package[key] for key in ("version", "resolved", "integrity") if key in package}})
    return resolved


def _reviewed_install_binding(drive_root: pathlib.Path, skill_name: str, skill_dir: pathlib.Path,
                              specs: List[Dict[str, Any]], expected_hash: str = "") -> str:
    """Revalidate new executable declarations through the existing review owner."""
    from ouroboros.skill_dependencies import payload_declared_install_specs
    from ouroboros.skill_loader import load_skill

    loaded = load_skill(skill_dir, drive_root)
    if loaded is None or loaded.load_error or loaded.name != skill_name:
        raise RuntimeError("reviewed install payload cannot be resolved")
    if not loaded.review.gate_for(loaded.content_hash)["executable_review"]:
        raise RuntimeError("install declarations require a fresh executable skill review")
    if expected_hash and loaded.content_hash != expected_hash:
        raise RuntimeError("skill changed during dependency installation; re-review before retrying")
    if install_specs_hash(payload_declared_install_specs(loaded)) != install_specs_hash(specs):
        raise RuntimeError("install specs differ from the hash-covered reviewed declaration")
    return loaded.content_hash


def install_isolated_dependencies(
    drive_root: pathlib.Path,
    skill_name: str,
    skill_dir: pathlib.Path,
    specs: List[Dict[str, Any]],
    *,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Install declared specs through existing process and dependency owners.

    `installed` records package/resource delivery; executable_ready is unknown
    unless the author declared a successful concrete check. Old package specs
    retain their wheel-only/ignore-scripts behavior without a new probe gate.
    """
    extended = any(spec.get("kind") == "download" or spec.get("steps") or spec.get("check") is not None
                   or spec.get("allow_source_build") is True or spec.get("allow_install_scripts") is True for spec in specs)
    reviewed_hash = _reviewed_install_binding(drive_root, skill_name, skill_dir, specs) if extended else ""
    review_check = (lambda: _reviewed_install_binding(drive_root, skill_name, skill_dir, specs, reviewed_hash)) if extended else None
    env_root = isolated_env_dir(skill_dir)
    env_root.mkdir(parents=True, exist_ok=True)
    cache_root = skill_state_dir(drive_root, skill_name) / "dependency_cache"
    installed: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    pending_python: List[Dict[str, Any]] = []
    failure: Dict[str, Any] = {}
    resolved: List[Dict[str, Any]] = []

    def finish(spec: Dict[str, Any], **facts: Any) -> None:
        row = {"kind": spec["kind"], "package": spec.get("package", ""), "bins": list(spec.get("bins") or []),
               "installed": True, "executable_ready": False if spec.get("check") is not None else None, **facts}
        installed.append(row)
        row.update(_install_actions(spec, env_root, cache_root, timeout_sec, logs, review_check))

    def flush_python() -> None:
        if not pending_python:
            return
        logs.extend(_install_python_packages([s["package"] for s in pending_python], env_root, timeout_sec,
                    allow_source_build=pending_python[0].get("allow_source_build") is True, cache_root=cache_root, review_check=review_check))
        for spec in pending_python:
            finish(spec)
        pending_python.clear()

    try:
        platform_names = {sys.platform, f"{sys.platform}-{platform.machine().lower()}"}
        for spec in specs:
            platforms = spec.get("platforms") or []
            if platforms and not platform_names.intersection(platforms):
                installed.append({"kind": spec["kind"], "package": spec.get("package", ""),
                                  "installed": False, "status": "not_applicable", "platforms": platforms})
                continue
            kind = str(spec.get("kind") or "").lower()
            if kind in {"pip", "pipx", "uv"}:
                if pending_python and (pending_python[0].get("allow_source_build") is True) != (spec.get("allow_source_build") is True):
                    flush_python()
                pending_python.append(spec)
                # An explicit step/check may depend on this package and precede
                # later resources; plain consecutive pip specs remain batched.
                if spec.get("steps") or spec.get("check"):
                    flush_python()
                continue
            flush_python()
            if kind in {"node", "npm"}:
                logs.extend(_install_node_package(str(spec["package"]), env_root, timeout_sec,
                            allow_install_scripts=spec.get("allow_install_scripts") is True, cache_root=cache_root, review_check=review_check))
                finish(spec)
            elif kind == "download":
                finish(spec, **_install_download(spec, env_root, cache_root, timeout_sec))
            else:
                raise RuntimeError(f"unsupported isolated install kind: {kind}")
        flush_python()
        if review_check is not None:
            review_check()
    except Exception as exc:
        failure = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        resolved = _resolved_packages(env_root)
    except Exception as exc:
        logs.append({"kind": "dependency_metadata_error", "error": f"{type(exc).__name__}: {exc}"})
    checked = [row["executable_ready"] for row in installed if row.get("executable_ready") is not None]
    fingerprint = {
        "schema_version": 1, "installed_at": utc_now_iso(), "skill": skill_name,
        "env_dir": ENV_DIRNAME, "specs_hash": install_specs_hash(specs), "reviewed_content_hash": reviewed_hash,
        "installed": installed, "resolved_packages": resolved, "logs": logs[-10:],
        "executable_ready": bool(checked) and all(checked) and not failure if checked else None,
        "status": "failed" if failure else "installed", "error": failure.get("error", ""),
    }
    atomic_write_json(env_root / FINGERPRINT_FILENAME, fingerprint, trailing_newline=True)
    atomic_write_json(skill_state_dir(drive_root, skill_name) / DEPS_STATE_FILENAME, fingerprint, trailing_newline=True)
    if failure:
        raise RuntimeError(failure["error"])
    return fingerprint


def read_deps_state(
    drive_root: pathlib.Path,
    skill_name: str,
    skill_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """Return persisted deps.json, optionally verified against the live env.

    Non-creating: a missing state dir reads like a missing file ({})."""
    try:
        path = skill_state_dir_path(drive_root, skill_name) / DEPS_STATE_FILENAME
        state = read_json_dict(path) or {}
    except Exception:
        return {}
    if skill_dir is None:
        return state
    fingerprint = read_json_dict(isolated_env_dir(skill_dir) / FINGERPRINT_FILENAME) or {}
    if str(state.get("status") or "") != "installed":
        # The payload-resident fingerprint.json is AGENT-WRITABLE (it lives in
        # the skill dir): it may only ever corroborate a durable deps.json
        # record, never substitute for one. A skill shipping a forged
        # "installed" fingerprint without the runtime-side install record
        # stays non-executable.
        if str(fingerprint.get("status") or "") == "installed":
            state_hash = str(state.get("specs_hash") or "")
            fingerprint_hash = str(fingerprint.get("specs_hash") or "")
            if state_hash and state_hash == fingerprint_hash:
                return fingerprint
        return state
    state_hash = str(state.get("specs_hash") or "")
    fingerprint_hash = str(fingerprint.get("specs_hash") or "")
    if str(fingerprint.get("status") or "") != "installed":
        return {**state, "status": "missing", "error": "isolated environment fingerprint is missing"}
    if fingerprint_hash != state_hash:
        return {**state, "status": "stale", "error": "isolated environment fingerprint is stale"}
    records = state.get("installed") or []
    if not isinstance(records, list):
        return {**state, "status": "stale", "error": "installed output records are malformed"}
    for installed in records:
        if not isinstance(installed, dict) or not isinstance(installed.get("outputs", []), list):
            return {**state, "status": "stale", "error": "installed output records are malformed"}
        for output in installed.get("outputs") or []:
            try:
                path = _install_target(isolated_env_dir(skill_dir), output["path"])
                stat = path.stat()
                stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
                recorded = tuple(output.get(key) for key in ("size_bytes", "mtime_ns", "inode", "device"))
                if stamp != recorded:
                    # Preserve fast unchanged reads for large resources. A changed
                    # stat needs byte verification, never a guessed digest match.
                    verify_exact_file(path, size_bytes=output["size_bytes"], sha256=output["sha256"],
                                      code_prefix="skill_output", label="installed output")
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                return {**state, "status": "stale", "error": f"installed output unavailable: {exc}"}
    return state

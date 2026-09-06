from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess
import threading
import time
import uuid
from subprocess import Popen
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.config import get_skills_repo_path, load_settings
from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
from ouroboros.platform_layer import merge_hidden_kwargs, subprocess_new_group_kwargs
from ouroboros.provider_models import MODEL_PROVIDER_CREDENTIAL_KEYS
from ouroboros.tools.process_facts import publish_process_facts as _publish_process_facts
from ouroboros.skill_loader import (
    SkillPayloadUnreadable,
    compute_content_hash,
    discover_skills,
    find_skill,
    grant_status_for_skill,
    save_enabled,
    skill_conflict_status,
    skill_review_gate,
    skill_state_dir,
    summarize_skills,
)
from ouroboros.skill_review import review_skill as _review_skill_impl
from ouroboros.skill_review_status import normalize_skill_review_status
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    canonical_data_root,
)
from ouroboros.tools.shell import (
    _active_subprocesses,
    _kill_process_group,
    _subprocess_lock,
)
from ouroboros.usage_accounting import current_usage_scope, record_unmetered_external_dispatch
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

_HARD_TIMEOUT_CEILING_SEC = 300
_SKILL_REVIEW_TOOL_TIMEOUT_SEC = int(os.environ.get("OUROBOROS_SKILL_REVIEW_TOOL_TIMEOUT_SEC", "1800"))
_DEFAULT_TIMEOUT_SEC = 60
_MAX_STDOUT_BYTES = 256 * 1024
_MAX_STDERR_BYTES = 128 * 1024

_ALLOWED_RUNTIMES = {
    "python": ("python", "python3"),
    "python3": ("python3", "python"),
    "bash": ("bash",),
    "node": ("node",),
    "deno": ("deno",),
    "ruby": ("ruby",),
    "go": ("go",),
}

_ALWAYS_FORWARDED_ENV = frozenset({
    "PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "TMPDIR", "TMP", "TEMP",
    # WA6: a python/python3 skill can fall back to the embedded sys.executable
    # (_resolve_runtime_binary); forward the bytecode-suppression policy so it
    # never writes __pycache__/*.pyc into a signed macOS bundle (parity with
    # isolated_deps._SAFE_ENV_KEYS and extension_process_runner._child_env).
    "PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX",
})

_FORBIDDEN_ENV_FORWARD_KEYS = FORBIDDEN_SKILL_SETTINGS


def _resolve_runtime_binary(runtime: str) -> Tuple[Optional[str], str]:
    """Resolve a skill runtime binary: ``(path, "")`` or ``(None, reason)``.

    The reason (when known) carries the node health verdicts verbatim so the
    surface error can say WHY nothing was usable, not just that it was not.
    """
    import sys
    if runtime == "node":
        # Skill-family node precedence is owned by
        # platform_layer.select_skill_node_runtime: bundled-first (the signed
        # runtime macOS code-signing enforcement cannot SIGKILL inside the
        # packaged app), with a health ROLLBACK to a working PATH node when the
        # bundled one is absent or execution-probed broken. A provably dead
        # candidate is never selected while a usable neighbour exists.
        try:
            from ouroboros.platform_layer import select_skill_node_runtime
            selected, info = select_skill_node_runtime()
            if selected:
                return selected, ""
            return None, info
        except Exception as exc:
            # An unexpected selector failure must not be invisible: the PATH
            # scan below still runs, but the operator sees WHY the skill-family
            # precedence was skipped (T16).
            log.warning(
                "select_skill_node_runtime failed (%s: %s); falling back to a plain PATH scan",
                type(exc).__name__, exc, exc_info=True,
            )
    candidates = _ALLOWED_RUNTIMES.get(runtime or "", ())
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved, ""
    if runtime in ("python", "python3") and sys.executable:
        resolved = pathlib.Path(sys.executable)
        if resolved.is_file():
            return str(resolved), ""
    return None, ""


def _scrub_env(
    manifest_env_keys: List[str],
    skill_state_dir_path: pathlib.Path,
    skill_name: str,
    granted_keys: List[str] | None = None,
) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for key in _ALWAYS_FORWARDED_ENV:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    if manifest_env_keys:
        settings = load_settings()
        from ouroboros.skill_loader import requested_core_setting_keys
        protected_upper = {k.upper() for k in _FORBIDDEN_ENV_FORWARD_KEYS}
        protected_upper.update(requested_core_setting_keys(list(manifest_env_keys or [])))
        granted_upper = {str(k).strip().upper() for k in (granted_keys or []) if str(k).strip()}
        allow = {str(k).strip() for k in manifest_env_keys if str(k).strip()}
        for key in allow:
            canonical = key.upper()
            if canonical in protected_upper and canonical not in granted_upper:
                log.warning(
                    "Skill %s asked env_from_settings for %s; refusing without explicit grant.",
                    skill_name, key,
                )
                continue
            val = settings.get(canonical) if canonical in protected_upper else settings.get(key)
            if val is None or val == "":
                continue
            env[canonical if canonical in protected_upper else key] = str(val)
    env["OUROBOROS_SKILL_NAME"] = skill_name
    env["OUROBOROS_SKILL_STATE_DIR"] = str(skill_state_dir_path)
    return env


def _drain_pipe_with_cap(pipe, cap: int, buf: bytearray, overflow_flag: Dict[str, bool], label: str) -> None:
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                return
            remaining = cap - len(buf)
            if remaining <= 0:
                overflow_flag[label] = True
                return
            if len(chunk) > remaining:
                buf.extend(chunk[:remaining])
                overflow_flag[label] = True
                return
            buf.extend(chunk)
    except (OSError, ValueError):
        return


def _run_skill_subprocess(
    cmd: List[str],
    *,
    cwd: str,
    env: Dict[str, str],
    timeout_sec: int,
    stdout_cap: int,
    stderr_cap: int,
    on_spawn: Optional[Callable[[], None]] = None,
) -> Tuple[int, bytes, bytes, bool]:
    popen_kwargs: Dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
    }
    popen_kwargs.update(subprocess_new_group_kwargs())
    popen_kwargs = merge_hidden_kwargs(popen_kwargs)
    _child_started_ts = time.monotonic()
    try:
        proc = Popen(cmd, **popen_kwargs)  # noqa: S603 — cmd is a vetted list, not shell
    except OSError as exc:
        # Pre-exec: no child ever existed (a runtime binary that vanished
        # between resolution and spawn). The platform's exception class is the
        # typed cause; there is no exit code to invent for it.
        _publish_process_facts(
            started_ts=_child_started_ts, pre_exec_failure=type(exc).__name__,
        )
        raise
    with _subprocess_lock:
        _active_subprocesses.add(proc)
    try:
        if on_spawn is not None:
            on_spawn()
    except BaseException:
        # The child exists once Popen returns.  If the durable disclosure
        # cannot be written, fail closed without leaving an untracked process.
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
        raise

    stdout_buf = bytearray()
    stderr_buf = bytearray()
    overflow_flag = {"stdout": False, "stderr": False}

    stdout_thread = threading.Thread(
        target=_drain_pipe_with_cap,
        args=(proc.stdout, stdout_cap, stdout_buf, overflow_flag, "stdout"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_pipe_with_cap,
        args=(proc.stderr, stderr_cap, stderr_buf, overflow_flag, "stderr"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    deadline = time.monotonic() + max(1, int(timeout_sec))
    overflowed = False
    timed_out = False
    try:
        while True:
            if overflow_flag["stdout"] or overflow_flag["stderr"]:
                overflowed = True
                _kill_process_group(proc)
                break
            if proc.poll() is not None:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _kill_process_group(proc)
                break
            time.sleep(0.05)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            proc.wait(timeout=2)
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
    finally:
        with _subprocess_lock:
            _active_subprocesses.discard(proc)
        try:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
        except OSError:
            pass

    # Typed process facts for the skill child, published where the truth is
    # known (D02): the deadline kill and the output-cap kill have no exit code
    # of their own to report and say so as host kills, while a child that
    # returned reports its exact code — including a NEGATIVE one, which the
    # ``or 0`` return below flattens for the legacy text renderer.
    _publish_process_facts(
        returncode=proc.returncode if not (timed_out or overflowed) else None,
        started_ts=_child_started_ts,
        timed_out=timed_out,
        killed_by_host=timed_out or overflowed,
    )
    if timed_out:
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout_sec,
            output=bytes(stdout_buf),
            stderr=bytes(stderr_buf),
        )
    return proc.returncode or 0, bytes(stdout_buf), bytes(stderr_buf), overflowed


def _record_skill_exec_dispatch(
    ctx: ToolContext,
    *,
    dispatch_id: str,
    skill_name: str,
    script_rel: str,
) -> str:
    """Disclose one successfully spawned script skill outside core metering."""

    bound = current_usage_scope()
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    task_id = str(
        getattr(ctx, "task_id", "")
        or meta.get("task_id")
        or meta.get("subagent_task_id")
        or (bound.task_id if bound is not None else "")
        or ""
    )
    root_task_id = str(
        meta.get("root_task_id")
        or getattr(ctx, "root_task_id", "")
        or (bound.root_task_id if bound is not None else "")
        or task_id
        or f"skill_exec:{skill_name}"
    )
    if not task_id:
        task_id = root_task_id
    parent_task_id = str(
        meta.get("parent_task_id")
        or getattr(ctx, "parent_task_id", "")
        or (bound.parent_task_id if bound is not None else "")
        or ""
    )
    drive_root = pathlib.Path(
        str(
            meta.get("budget_drive_root")
            or getattr(ctx, "budget_drive_root", "")
            or (bound.drive_root if bound is not None else "")
            or getattr(ctx, "drive_root", "")
            or "."
        )
    ).resolve(strict=False)
    return record_unmetered_external_dispatch(
        dispatch_id,
        drive_root=drive_root,
        provider="external-skill",
        task_id=task_id,
        root_task_id=root_task_id,
        parent_task_id=parent_task_id,
        category="external_skill",
        source=f"skill_exec:{skill_name}:{script_rel}",
    )


def _bound_timeout(requested_sec: Any) -> int:
    try:
        timeout = int(requested_sec)
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT_SEC
    if timeout <= 0:
        timeout = _DEFAULT_TIMEOUT_SEC
    return min(timeout, _HARD_TIMEOUT_CEILING_SEC)


def _cap(data: bytes, limit: int, label: str) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n⚠️ OMISSION NOTE: skill_exec truncated {label} at "
        f"{limit} bytes (total {len(data)})."
    )


def _emit_skill_lifecycle_event(
    ctx: ToolContext,
    *,
    event_type: str,
    skill: str,
    script: str,
    exit_code: int | None = None,
    error: str = "",
) -> None:
    event = {
        "type": event_type,
        "ts": utc_now_iso(),
        "task_id": getattr(ctx, "task_id", "") or "",
        "skill": skill,
        "script": script,
    }
    if exit_code is not None:
        event["exit_code"] = int(exit_code)
    if error:
        event["error"] = str(error)
    event_queue = getattr(ctx, "event_queue", None)
    if event_queue is not None:
        try:
            event_queue.put_nowait(event)
            return
        except Exception:
            log.debug("Could not queue skill lifecycle event", exc_info=True)
    try:
        append_jsonl(pathlib.Path(ctx.drive_root) / "logs" / "events.jsonl", event)
    except Exception:
        log.debug("Could not append skill lifecycle event", exc_info=True)
    try:
        from ouroboros.event_bus import SKILL_LIFECYCLE, publish_event

        publish_event(SKILL_LIFECYCLE, event)
    except Exception:
        log.debug("Could not publish skill lifecycle event", exc_info=True)


def _render_skill_exec_result(
    ctx: ToolContext,
    *,
    payload: Dict[str, Any],
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    overflowed: bool,
) -> str:
    skill_name = str(payload.get("skill") or "")
    script_rel = str(payload.get("script") or "")
    returncode = int(payload.get("exit_code") or 0)
    payload = {
        **payload,
        "output_overflow": overflowed,
        "stdout": _cap(stdout_bytes, _MAX_STDOUT_BYTES, "stdout"),
        "stderr": _cap(stderr_bytes, _MAX_STDERR_BYTES, "stderr"),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if overflowed or returncode != 0:
        error = (
            "stdout/stderr byte cap exceeded"
            if overflowed
            else f"exited with code {returncode}"
        )
        _emit_skill_lifecycle_event(
            ctx,
            event_type="skill_exec_failed",
            skill=skill_name,
            script=script_rel,
            exit_code=int(returncode),
            error=error,
        )
        if overflowed:
            header = (
                f"⚠️ SKILL_EXEC_OVERFLOW: skill {skill_name!r} script {script_rel!r} "
                f"exceeded stdout/stderr byte caps (stdout<={_MAX_STDOUT_BYTES}B, "
                f"stderr<={_MAX_STDERR_BYTES}B) and was killed."
            )
        else:
            header = (
                f"⚠️ SKILL_EXEC_FAILED: skill {skill_name!r} script "
                f"{script_rel!r} exited with code {returncode}."
            )
        return f"{header}\n\n{rendered}"
    _emit_skill_lifecycle_event(
        ctx,
        event_type="skill_exec_finished",
        skill=skill_name,
        script=script_rel,
        exit_code=0,
    )
    return rendered


def _resolve_script_path(
    skill_dir: pathlib.Path,
    script_rel: str,
    *,
    reviewed_paths: Optional[List[pathlib.Path]] = None,
) -> Optional[pathlib.Path]:
    rel = (script_rel or "").strip()
    if not rel or rel.startswith("/") or rel.startswith("~"):
        return None
    if ".." in pathlib.PurePosixPath(rel).parts:
        return None
    candidate = (skill_dir / rel).resolve()
    try:
        candidate.relative_to(skill_dir.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    if reviewed_paths is not None:
        reviewed = {p.resolve() for p in reviewed_paths}
        if candidate not in reviewed:
            return None
    return candidate


def _skill_tool_preflight(
    ctx: ToolContext,
    binding: ResolvedResourceBinding | None = None,
) -> Optional[str]:
    if binding is not None:
        return None
    repo_path = get_skills_repo_path()
    if repo_path:
        return None
    if discover_skills(canonical_data_root(ctx), repo_path=""):
        return None
    return (
        "⚠️ SKILLS_UNAVAILABLE: No skills are discoverable. Point "
        "OUROBOROS_SKILLS_REPO_PATH at a local checkout in Settings → "
        "Behavior → External Skills Repo, or install skills into the data plane."
    )


def _handle_list_skills(ctx: ToolContext, **_kwargs: Any) -> str:
    err = _skill_tool_preflight(ctx)
    if err:
        return err
    drive_root = canonical_data_root(ctx)
    summary = summarize_skills(drive_root)
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _handle_review_skill(
    ctx: ToolContext,
    skill: str = "",
    review_rebuttal: str = "",
    _resolved_binding: ResolvedResourceBinding | None = None,
    **_kwargs: Any,
) -> str:
    skill_name = str(skill or "").strip()
    if not skill_name:
        return "⚠️ SKILL_REVIEW_ERROR: 'skill' argument is required."
    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx, root="skill_payload", operation="review", path=".",
            skill_name=skill_name,
        )
    except Exception as exc:
        return f"⚠️ SKILL_REVIEW_ERROR: {exc}"
    err = _skill_tool_preflight(ctx, binding)
    if err:
        return err
    from ouroboros.skill_review import (
        _count_attempts_for_content,
        _load_accepted_rebuttals,
        render_skill_review_block,
    )
    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking

    def _review_with_optional_rebuttal(review_ctx: ToolContext, review_name: str):
        if str(review_rebuttal or "").strip():
            return _review_skill_impl(
                review_ctx,
                review_name,
                review_rebuttal=review_rebuttal,
                _resolved_binding=binding,
            )
        return _review_skill_impl(
            review_ctx, review_name, _resolved_binding=binding,
        )

    payload = run_skill_review_lifecycle_blocking(
        ctx,
        skill_name,
        source="tool",
        review_impl=_review_with_optional_rebuttal,
        _resolved_binding=binding,
    )
    drive_root = binding.state_drive_root
    content_hash = str(payload.get("content_hash") or "")
    attempt_idx = _count_attempts_for_content(drive_root, skill_name, content_hash) if content_hash else 1
    if attempt_idx <= 0:
        attempt_idx = 1
    accepted_rebuttals = _load_accepted_rebuttals(drive_root, skill_name)
    markdown = render_skill_review_block(
        payload,
        attempt_idx=attempt_idx,
        accepted_rebuttals=accepted_rebuttals,
    )
    # The rendered block above already contains every finding (items x reviewers,
    # full reasons, fix suggestions, convergence, rebuttals) — that is what the
    # agent needs. The previously appended raw JSON payload repeated the same
    # review 3-4 more times (findings + raw_actor_records + raw_result +
    # advisory_result) in the agent's reasoning context, feeding context overflow
    # on multi-round skill work. Forensic raw records remain on disk in
    # state/skills/<name>/review.json (Skills page + on-demand reads).
    return markdown


def _skill_deps_exec_block(drive_root: pathlib.Path, loaded: Any) -> str:
    deps_status, reason = _skill_deps_not_ready(drive_root, loaded)
    if not reason:
        return ""
    return (
        f"⚠️ SKILL_EXEC_BLOCKED: skill {loaded.name!r} isolated "
        f"dependencies are not ready (status={deps_status!r}). "
        "Re-run skill_review so a fresh executable review can reinstall dependencies."
    )


def _skill_deps_not_ready(drive_root: pathlib.Path, loaded: Any) -> tuple[str, str]:
    try:
        from ouroboros.marketplace.install_specs import install_specs_hash as _specs_hash
        from ouroboros.marketplace.isolated_deps import read_deps_state
        from ouroboros.skill_dependencies import auto_install_specs_for_skill

        auto_specs = auto_install_specs_for_skill(drive_root, loaded)
        if not auto_specs:
            return "", ""
        deps_state = read_deps_state(drive_root, loaded.name, loaded.skill_dir)
        deps_status = str(deps_state.get("status") or "pending")
        if deps_status != "installed":
            return deps_status, "status"
        if str(deps_state.get("specs_hash") or "") != _specs_hash(auto_specs):
            return deps_status, "fingerprint"
        return "", ""
    except Exception:
        log.debug("skill deps readiness probe failed", exc_info=True)
        return "", ""


def _non_executable_review_message(prefix: str, skill_name: str, status: str, *, stale: bool = False) -> str:
    gate = skill_review_gate(status, stale=stale)
    normalized_status = normalize_skill_review_status(status)
    if gate["blocking_reason"] == "blocker_findings_under_blocking_enforcement":
        return (
            f"⚠️ {prefix}: skill {skill_name!r} review status is 'blockers' "
            "and review enforcement is blocking, so it is not executable. "
            "Fix the listed blocker findings or switch review enforcement to "
            "advisory and reload the skill state."
        )
    stale_note = f", stale={stale}" if stale else ""
    return (
        f"⚠️ {prefix}: skill {skill_name!r} review status is {normalized_status!r}{stale_note}, "
        f"not executable ({gate['blocking_reason']}). A fresh executable review is required. {gate['summary']}"
    )


def _non_text_script_refusal(script_path: pathlib.Path, script_rel: str) -> str:
    """Execution-seam guard (#447 X4): review admission carries non-UTF-8
    payload files as DESCRIPTORS instead of hard-blocking them, so the exec
    layer independently refuses to hand a non-text blob (a zipapp/PK archive
    renamed to a declared script name) to an interpreter. A declared script is
    a reviewed TEXT artifact by contract. Returns the typed refusal or ""."""
    try:
        # Decode the WHOLE file: a 64 KiB prefix check would pass a script with
        # a binary tail (and falsely refuse a multibyte char straddling the
        # boundary). The file is about to be executed anyway — one full read
        # here is not the expensive part.
        script_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return (
            f"⚠️ SKILL_EXEC_ERROR: script {script_rel!r} is not valid UTF-8 "
            "text; declared skill scripts are reviewed text artifacts and a "
            "binary blob cannot be executed through skill_exec."
        )
    return ""


def _extension_not_executable_message(loaded: Any, drive_root: pathlib.Path) -> str:
    """Explain why an extension does not execute, with its typed live state.

    The prior wording asserted unconditionally that ``register(api)`` had already
    run, which is false for an extension that failed to load.
    """
    try:
        from ouroboros.extension_loader import runtime_state_for_loaded_skill

        live = runtime_state_for_loaded_skill(loaded, drive_root)
    except Exception:
        log.debug("skill_exec extension liveness lookup failed", exc_info=True)
        live = {}
    facts = (
        f"live_loaded={bool(live.get('live_loaded'))}, "
        f"desired_live={bool(live.get('desired_live'))}, "
        f"reason={str(live.get('reason') or 'unknown')!r}, "
        f"process={str(live.get('process') or 'unknown')!r}"
    )
    if live.get("load_error"):
        facts += f", load_error={str(live.get('load_error'))!r}"
    return (
        f"⚠️ SKILL_EXEC_EXTENSION: skill {loaded.name!r} is a "
        "type=extension plugin and does not execute through the "
        f"subprocess substrate. Its current live state is {facts} "
        "(inspect the registered surfaces via the snapshot produced by "
        "``ouroboros.extension_loader.snapshot()``). Use its "
        "provider-safe ``ext_<len>_<token>_*`` tools, "
        "``/api/extensions/<skill>/...`` routes, or provider-safe "
        "extension WebSocket handlers instead."
    )


def _handle_skill_exec(
    ctx: ToolContext,
    skill: str = "",
    script: str = "",
    args: Optional[List[str]] = None,
    **_kwargs: Any,
) -> str:
    skill_name = str(skill or "").strip()
    script_rel = str(script or "").strip()
    if not skill_name or not script_rel:
        return "⚠️ SKILL_EXEC_ERROR: both 'skill' and 'script' are required."
    err = _skill_tool_preflight(ctx)
    if err:
        return err

    drive_root = canonical_data_root(ctx)
    loaded = find_skill(drive_root, skill_name)
    if loaded is None:
        return (
            f"⚠️ SKILL_EXEC_ERROR: skill {skill_name!r} not found in "
            "OUROBOROS_SKILLS_REPO_PATH."
        )
    if loaded.load_error:
        return (
            f"⚠️ SKILL_EXEC_ERROR: skill {skill_name!r} manifest is broken "
            f"({loaded.load_error}). Fix the skill package and re-review."
        )
    if loaded.manifest.is_extension():
        return _extension_not_executable_message(loaded, drive_root)
    if not loaded.manifest.is_script():
        return (
            f"⚠️ SKILL_EXEC_ERROR: skill {skill_name!r} has type "
            f"{loaded.manifest.type!r}. Only 'script' skills can execute "
            "via skill_exec in Phase 3."
        )
    if not loaded.enabled:
        return (
            f"⚠️ SKILL_EXEC_BLOCKED: skill {skill_name!r} is disabled. "
            "Enable it after review in the Skills UI (Phase 5) or via "
            "the dedicated enable tool."
        )
    conflict = skill_conflict_status(loaded, discover_skills(drive_root))
    if conflict:
        names = list(conflict.get("skills") or [])
        return (
            f"⚠️ SKILL_EXEC_BLOCKED: skill {skill_name!r} conflicts with "
            f"enabled skills {names}. Disable the conflicting skill first."
        )
    try:
        current_hash = compute_content_hash(
            loaded.skill_dir,
            manifest_entry=loaded.manifest.entry,
            manifest_scripts=loaded.manifest.scripts,
        )
    except SkillPayloadUnreadable as exc:
        return (
            f"⚠️ SKILL_EXEC_ERROR: skill {skill_name!r} payload became unreadable "
            f"({exc}). Fix the skill package and re-run skill_review before "
            "executing."
        )
    stale = loaded.review.is_stale_for(current_hash)
    gate = skill_review_gate(loaded.review.status, stale=stale)
    if stale:
        return (
            f"⚠️ SKILL_EXEC_BLOCKED: skill {skill_name!r} was edited since "
            f"the last review. Re-run skill_review(skill={skill_name!r}) "
            "before executing."
        )
    if not gate["executable_review"]:
        return _non_executable_review_message("SKILL_EXEC_BLOCKED", skill_name, loaded.review.status, stale=stale)
    deps_block = _skill_deps_exec_block(drive_root, loaded)
    if deps_block:
        return deps_block

    runtime = (loaded.manifest.runtime or "").strip().lower()
    runtime_binary, runtime_unavailable_reason = _resolve_runtime_binary(runtime)
    try:
        from ouroboros.marketplace.isolated_deps import python_runtime_binary
        if runtime in {"python", "python3"}:
            isolated = python_runtime_binary(loaded.skill_dir)
            runtime_binary = str(isolated) if isolated is not None else runtime_binary
    except Exception:
        log.debug("Could not resolve isolated Python runtime", exc_info=True)
    if runtime_binary is None:
        return (
            f"⚠️ SKILL_EXEC_ERROR: skill {skill_name!r} declared runtime "
            f"{runtime!r} is not in the allowlist {sorted(set(_ALLOWED_RUNTIMES))} "
            f"or the matching binary is not on PATH."
            + (f" ({runtime_unavailable_reason})" if runtime_unavailable_reason else "")
        )

    def _canonical_declared_path(declared_name: str) -> Optional[pathlib.Path]:
        name = declared_name.strip()
        if not name:
            return None
        if "/" in name or name.startswith("."):
            return _resolve_script_path(loaded.skill_dir, name)
        return _resolve_script_path(loaded.skill_dir, f"scripts/{name}")

    declared_scripts: List[pathlib.Path] = []
    declared_by_name: Dict[str, pathlib.Path] = {}
    for entry in loaded.manifest.scripts or []:
        if not isinstance(entry, dict):
            continue
        declared_name = str(entry.get("name") or "").strip()
        if not declared_name:
            continue
        canonical = _canonical_declared_path(declared_name)
        if canonical is None:
            continue
        if canonical not in declared_scripts:
            declared_scripts.append(canonical)
        declared_by_name[declared_name] = canonical
        if "/" not in declared_name:
            declared_by_name[f"scripts/{declared_name}"] = canonical

    script_path: Optional[pathlib.Path] = declared_by_name.get(script_rel.strip())
    if script_path is None:
        script_path = _resolve_script_path(
            loaded.skill_dir, script_rel, reviewed_paths=declared_scripts
        )
    if script_path is None:
        return (
            f"⚠️ SKILL_EXEC_ERROR: script {script_rel!r} is not a declared "
            "script for this skill. Only names listed under the manifest's "
            "``scripts:`` array can execute via skill_exec (assets/* and "
            "SKILL.md body are reviewed content but not executable payload). "
            "Add the script to the manifest and re-run skill_review."
        )

    if (non_text := _non_text_script_refusal(script_path, script_rel)):
        return non_text
    cmd = [runtime_binary, str(script_path)]
    if args is None:
        extra_args: List[Any] = []
    elif isinstance(args, str):
        return (
            "⚠️ SKILL_EXEC_ERROR: 'args' must be a list of scalar "
            "strings/numbers, not a single string. Wrap as ['alpha'] "
            "for a one-element argv."
        )
    elif isinstance(args, (list, tuple)):
        extra_args = list(args)
    else:
        return (
            "⚠️ SKILL_EXEC_ERROR: 'args' must be a list of scalar "
            f"strings/numbers. Got {type(args).__name__}={args!r}."
        )
    for arg in extra_args:
        if not isinstance(arg, (str, int, float)) or isinstance(arg, bool):
            return (
                "⚠️ SKILL_EXEC_ERROR: args must be a list of scalar "
                f"strings/numbers. Element {arg!r} ({type(arg).__name__}) "
                "is not allowed."
            )
        cmd.append(str(arg))

    timeout = _bound_timeout(loaded.manifest.timeout_sec)

    state_dir = skill_state_dir(drive_root, loaded.name)
    grants = grant_status_for_skill(drive_root, loaded)
    missing_core = list(grants.get("missing_keys") or [])
    missing_permissions = list(grants.get("missing_permissions") or [])
    if missing_core or missing_permissions:
        requested = []
        if missing_core:
            requested.append(f"core settings keys {missing_core}")
        if missing_permissions:
            requested.append(f"permissions {missing_permissions}")
        return (
            "⚠️ SKILL_EXEC_GRANT_REQUIRED: skill "
            f"{loaded.name!r} requests {' and '.join(requested)}. "
            "Grant them from the Skills UI after a fresh executable review before execution."
        )
    env = _scrub_env(
        manifest_env_keys=list(loaded.manifest.env_from_settings or []),
        skill_state_dir_path=state_dir,
        skill_name=loaded.name,
        granted_keys=list(grants.get("granted_keys") or []),
    )
    try:
        from ouroboros.marketplace.isolated_deps import augment_env_for_skill_deps

        env = augment_env_for_skill_deps(env, loaded.skill_dir)
    except Exception:
        log.debug("Could not augment skill env with isolated dependencies", exc_info=True)

    # E2BIG hygiene (C5): byte-accurate argv+env budget against the REAL exec
    # environment, checked before spawn (type validation above only proves the
    # args are scalars). No automatic file/stdin fallback — a skill accepts
    # bulk input via files only when its own manifest says so — so an
    # over-budget call is a typed refusal pointing at the skill's file inputs.
    from ouroboros.argv_budget import argv_budget_excess

    _argv_excess = argv_budget_excess(cmd, env=env)
    if _argv_excess:
        return (
            f"⚠️ SKILL_EXEC_ARGV_TOO_LARGE: refusing to spawn {loaded.name!r}: "
            f"{_argv_excess} Write bulk payloads to a file the skill reads "
            "instead of passing them as args."
        )

    # TOCTOU narrowing: re-hash the payload immediately before spawn. The
    # gate-time hash above ran before grants/env/deps resolution — a write
    # landing in that window would execute unreviewed code under a PASS verdict.
    try:
        spawn_hash = compute_content_hash(
            loaded.skill_dir,
            manifest_entry=loaded.manifest.entry,
            manifest_scripts=loaded.manifest.scripts,
        )
    except SkillPayloadUnreadable as exc:
        return (
            f"⚠️ SKILL_EXEC_ERROR: skill {skill_name!r} payload became unreadable "
            f"right before execution ({exc})."
        )
    if spawn_hash != current_hash:
        return (
            f"⚠️ SKILL_EXEC_BLOCKED: skill {skill_name!r} payload changed between "
            "the review-freshness check and execution. Re-run skill_review."
        )

    dispatch_id = f"skill_exec:{uuid.uuid4().hex}"
    model_capable = any(str(env.get(key) or "").strip() for key in MODEL_PROVIDER_CREDENTIAL_KEYS)
    try:
        returncode, stdout_bytes, stderr_bytes, overflowed = _run_skill_subprocess(
            cmd,
            cwd=str(loaded.skill_dir),
            env=env,
            timeout_sec=timeout,
            stdout_cap=_MAX_STDOUT_BYTES,
            stderr_cap=_MAX_STDERR_BYTES,
            on_spawn=((
                lambda: _record_skill_exec_dispatch(
                    ctx,
                    dispatch_id=dispatch_id,
                    skill_name=loaded.name,
                    script_rel=script_rel,
                )
            ) if model_capable else None),
        )
    except subprocess.TimeoutExpired as exc:
        _emit_skill_lifecycle_event(
            ctx,
            event_type="skill_exec_failed",
            skill=loaded.name,
            script=script_rel,
            error=f"timeout after {timeout}s",
        )
        return (
            f"⚠️ SKILL_EXEC_TIMEOUT: skill {skill_name!r} script "
            f"{script_rel!r} exceeded {timeout}s limit.\n"
            f"stdout_partial:\n{_cap(exc.stdout or b'', _MAX_STDOUT_BYTES, 'stdout')}\n"
            f"stderr_partial:\n{_cap(exc.stderr or b'', _MAX_STDERR_BYTES, 'stderr')}"
        )
    except FileNotFoundError:
        _emit_skill_lifecycle_event(
            ctx,
            event_type="skill_exec_failed",
            skill=loaded.name,
            script=script_rel,
            error=f"runtime binary {runtime_binary!r} unavailable",
        )
        return (
            f"⚠️ SKILL_EXEC_ERROR: runtime binary {runtime_binary!r} is no "
            "longer available."
        )
    except OSError as exc:
        _emit_skill_lifecycle_event(
            ctx,
            event_type="skill_exec_failed",
            skill=loaded.name,
            script=script_rel,
            error=f"OS error running skill: {exc}",
        )
        return f"⚠️ SKILL_EXEC_ERROR: OS error running skill: {exc}"

    return _render_skill_exec_result(
        ctx,
        payload={
            "skill": loaded.name,
            "script": script_rel,
            "runtime": runtime,
            "exit_code": int(returncode),
            "timeout_sec": timeout,
        },
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        overflowed=overflowed,
    )


_TRUE_LITERALS = {"true", "yes", "on", "1"}
_FALSE_LITERALS = {"false", "no", "off", "0"}


def _coerce_bool_arg(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_LITERALS:
            return True
        if lowered in _FALSE_LITERALS:
            return False
    return None


def _handle_toggle_skill(
    ctx: ToolContext,
    skill: str = "",
    enabled: Any = None,
    **_kwargs: Any,
    ) -> str:
    skill_name = str(skill or "").strip()
    if not skill_name:
        return "⚠️ SKILL_TOGGLE_ERROR: 'skill' argument is required."
    if enabled is None:
        return "⚠️ SKILL_TOGGLE_ERROR: 'enabled' (true|false) is required."
    coerced = _coerce_bool_arg(enabled)
    if coerced is None:
        return (
            "⚠️ SKILL_TOGGLE_ERROR: 'enabled' must be a boolean or one of "
            f"{sorted(_TRUE_LITERALS | _FALSE_LITERALS)}. "
            f"Got {enabled!r} ({type(enabled).__name__})."
        )
    err = _skill_tool_preflight(ctx)
    if err:
        return err

    drive_root = canonical_data_root(ctx)
    from ouroboros.skill_lifecycle_queue import skill_lifecycle_file_lock

    with skill_lifecycle_file_lock(drive_root):
        loaded = find_skill(drive_root, skill_name)
        if loaded is None:
            return (
                f"⚠️ SKILL_TOGGLE_ERROR: skill {skill_name!r} not found in "
                "OUROBOROS_SKILLS_REPO_PATH."
            )
        collision_load_error = loaded.load_error.lower().startswith("skill name collision:")
        if coerced and loaded.load_error:
            return (
                f"⚠️ SKILL_TOGGLE_ERROR: skill {skill_name!r} cannot be enabled "
                f"— loader rejected it ({loaded.load_error})."
            )
        if coerced:
            conflict = skill_conflict_status(loaded, discover_skills(drive_root))
            if conflict:
                names = list(conflict.get("skills") or [])
                return (
                    f"⚠️ SKILL_TOGGLE_ERROR: cannot enable {loaded.name!r} while "
                    f"conflicting skills are enabled: {names}. Disable them first."
                )
            stale = loaded.review.is_stale_for(loaded.content_hash)
            gate = skill_review_gate(loaded.review.status, stale=stale)
            grants = grant_status_for_skill(drive_root, loaded)
            if not gate["executable_review"]:
                return _non_executable_review_message(
                    "SKILL_TOGGLE_ERROR",
                    skill_name,
                    loaded.review.status,
                    stale=stale,
                )
            if not grants.get("all_granted", True):
                missing_bits = []
                if grants.get("missing_keys"):
                    missing_bits.append(f"keys={grants.get('missing_keys')}")
                if grants.get("missing_permissions"):
                    missing_bits.append(f"permissions={grants.get('missing_permissions')}")
                missing_text = f" ({', '.join(missing_bits)})" if missing_bits else ""
                return (
                    "⚠️ SKILL_TOGGLE_ERROR: cannot enable until requested grants "
                    f"are approved{missing_text}."
                )
            deps_status, deps_reason = _skill_deps_not_ready(drive_root, loaded)
            if deps_reason == "status":
                return (
                    f"⚠️ SKILL_TOGGLE_ERROR: skill {loaded.name!r} declares "
                    f"isolated dependencies (status={deps_status!r}). "
                    "Re-run skill_review (PASS triggers a deps re-install) before enabling."
                )
            if deps_reason == "fingerprint":
                return (
                    f"⚠️ SKILL_TOGGLE_ERROR: skill {loaded.name!r} dependency "
                    "fingerprint is stale (provenance changed since last install). "
                    "Re-run skill_review before enabling."
                )
        if not coerced and collision_load_error:
            extension_action = None
            extension_reason = "name_collision"
            from ouroboros import extension_loader
            if loaded.name in extension_loader.snapshot()["extensions"]:
                extension_loader.unload_extension(loaded.name)
                extension_action = "extension_unloaded"
            stale = loaded.review.is_stale_for(loaded.content_hash)
            gate = skill_review_gate(loaded.review.status, stale=stale)
            return json.dumps({"skill": loaded.name, "enabled": False, "review_status": loaded.review.status, "review_gate": gate, "executable_review": gate["executable_review"], "extension_action": extension_action, "extension_reason": extension_reason, "message": f"Skill {loaded.name!r} was not persisted as disabled because its sanitized identity collides with another skill directory. Rename one of the directories first."}, ensure_ascii=False, indent=2)
        save_enabled(drive_root, loaded.name, coerced, actor="agent_tool", reason=str(ctx.task_id or ""))
        loaded.enabled = coerced
        extension_action = None
        extension_reason = "not_extension"
        extension_load_error_msg = ""
        extension_process = ""
        extension_server_reconcile = ""
        from ouroboros import extension_loader
        if loaded.manifest.is_extension() or loaded.name in extension_loader.snapshot()["extensions"]:
            from ouroboros.config import load_settings as _load_settings
            live_state = extension_loader.reconcile_extension(
                loaded.name, drive_root, _load_settings,
                selected_skill=loaded, retry_load_error=True,
                revert_enabled_on_error=coerced,
            )
            extension_action = live_state.get("action")
            extension_reason = str(live_state.get("reason") or "")
            extension_load_error_msg = str(live_state.get("load_error") or "")
            extension_process = str(live_state.get("process") or "")
            extension_server_reconcile = str(live_state.get("server_reconcile") or "")
        # Mirror schedule readiness immediately (parallel to the HTTP toggle path).
        try:
            from supervisor.queue import resync_skill_schedules

            resync_skill_schedules(drive_root)
        except Exception:
            log.debug("toggle_skill schedule sync failed", exc_info=True)
        stale = loaded.review.is_stale_for(loaded.content_hash)
        gate = skill_review_gate(loaded.review.status, stale=stale)
        # Atomic enable: reconcile reverts enabled.json to False when the out-of-process
        # catalog/register dry-run fails, so report the effective (reverted) state — not
        # the requested one — to avoid an enabled=True report over a disabled-on-disk skill.
        reverted = coerced and extension_action == "extension_load_error"
        effective_enabled = coerced and not reverted
        message = (
            f"cannot enable {loaded.name!r}: {extension_load_error_msg or 'extension failed to load'}"
            if reverted else f"Skill {loaded.name!r} enabled={effective_enabled}"
        )
        return json.dumps({"skill": loaded.name, "enabled": effective_enabled, "review_status": loaded.review.status, "review_gate": gate, "executable_review": gate["executable_review"], "extension_action": extension_action, "extension_reason": extension_reason, "process": extension_process, "server_reconcile": extension_server_reconcile, "message": message}, ensure_ascii=False, indent=2)

_LIST_SCHEMA = {
    "name": "list_skills",
    "description": (
        "List external skill packages discovered in OUROBOROS_SKILLS_REPO_PATH. "
        "Returns counts + per-skill metadata (name, type, enabled, review_status, and "
        "available_for_execution, which is the SCRIPT-execution flag only). Extension "
        "rows also carry desired_live, live_loaded, live_reason, load_error and process, "
        "which is where an extension's liveness actually lives. Read-only."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_REVIEW_SCHEMA = {
    "name": "skill_review",
    "description": (
        "Run reviewer-slot skill review on one external skill package using the "
        "shared reviewer-slot configuration and scored against the "
        "Skill Review Checklist section in docs/CHECKLISTS.md. Persists the "
        "verdict to data/state/skills/<name>/review.json with a content "
        "hash so a later edit invalidates the review automatically. "
        "Max Review Cycles semantics: an identical snapshot never buys a new "
        "panel — a recorded substantive verdict REPLAYS free, keyed by (review "
        "group, content hash, panel-contract fingerprint), so only changed "
        "content, a changed panel contract, or a NEW rebuttal dispatches "
        "reviewers again. The shared OUROBOROS_REVIEW_MAX_CYCLES ceiling "
        "bounds PAID panel dispatches per ceiling key (root task for "
        "task-driven reviews; the current content snapshot for the manual "
        "lane); on exhaustion the review is refused free with a typed "
        "review_cycles_exhausted event and the skill stays honestly PENDING — "
        "never executable without a real verdict."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Skill name (directory name in OUROBOROS_SKILLS_REPO_PATH).",
            },
            "review_rebuttal": {
                "type": "string",
                "description": (
                    "Optional rebuttal to prior review findings. Use only when "
                    "you have code-grounded evidence that a previous finding was "
                    "a false positive or already addressed. The rebuttal is "
                    "identified by CONTENT: its sha256 buys exactly ONE paid "
                    "panel wave when new to the current snapshot's streak; "
                    "repeating an already-adjudicated rebuttal replays the "
                    "recorded verdict free, and no rebuttal buys past the "
                    "paid-cycle ceiling."
                ),
            },
        },
        "required": ["skill"],
    },
}

_EXEC_SCHEMA = {
    "name": "skill_exec",
    "description": (
        "Execute a script from an external skill package. The skill must be "
        "enabled and carry a fresh executable review verdict. Only type=script "
        "skills execute via this substrate — type=instruction skills are "
        "catalogued + reviewable but have no executable payload by "
        "design; type=extension skills run IN-PROCESS via the Phase 4 "
        "extension_loader (calling skill_exec on an extension returns "
        "SKILL_EXEC_EXTENSION pointing at that surface). The ``script`` "
        "argument must match a "
        "``name`` entry in the manifest's ``scripts:`` array (SKILL.md "
        "body and assets/* are reviewed content but not executable). "
        "Runtime allowlist: python/python3/bash/node/deno/ruby/go. The subprocess "
        "runs with cwd=skill_dir, a scrubbed env (env_from_settings "
        "keys only), panic-kill tracking, and a timeout from the "
        "manifest (capped at 300s). v5.1.2 Frame A: OUROBOROS_RUNTIME_MODE "
        "no longer gates execution — light, advanced, and pro all let "
        "reviewed + enabled skills run. Light still blocks repo "
        "self-modification and the runtime_mode elevation ratchet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Skill name (directory name in OUROBOROS_SKILLS_REPO_PATH).",
            },
            "script": {
                "type": "string",
                "description": (
                    "Relative path of the script inside the skill directory "
                    "(e.g. 'scripts/fetch.py'). Absolute paths and '..' "
                    "traversal are rejected."
                ),
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional argv for the script.",
            },
        },
        "required": ["skill", "script"],
    },
}

_TOGGLE_SCHEMA = {
    "name": "toggle_skill",
    "description": (
        "Enable or disable a skill. Disabled skills are excluded from "
        "skill_exec regardless of review status. Enabling requires a fresh "
        "executable review and any requested key or host-permission grants."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Skill name.",
            },
            "enabled": {
                "type": "boolean",
                "description": "True to enable, False to disable.",
            },
        },
        "required": ["skill", "enabled"],
    },
}


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="list_skills",
            schema=_LIST_SCHEMA,
            handler=_handle_list_skills,
            is_code_tool=False,
            timeout_sec=30,
        ),
        ToolEntry(
            name="skill_review",
            schema={**_REVIEW_SCHEMA, "name": "skill_review"},
            handler=_handle_review_skill,
            is_code_tool=False,
            timeout_sec=_SKILL_REVIEW_TOOL_TIMEOUT_SEC,
        ),
        ToolEntry(
            name="skill_exec",
            schema=_EXEC_SCHEMA,
            handler=_handle_skill_exec,
            is_code_tool=False,
            timeout_sec=_HARD_TIMEOUT_CEILING_SEC,
        ),
        ToolEntry(
            name="toggle_skill",
            schema=_TOGGLE_SCHEMA,
            handler=_handle_toggle_skill,
            is_code_tool=False,
            timeout_sec=15,
        ),
    ]

__all__ = [
    "get_tools",
    "_ALLOWED_RUNTIMES",
    "_HARD_TIMEOUT_CEILING_SEC",
    "_SKILL_REVIEW_TOOL_TIMEOUT_SEC",
]

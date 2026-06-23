"""Read-only skill payload syntax/contract preflight for heal-mode agents.

Tri-model review remains authoritative. Preflight uses argv-only subprocesses,
cwd=skill_dir, scrubbed env, 30s timeout, in-process Python compile(), and no
review/enablement/grant state mutation.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import shutil
import subprocess
from subprocess import Popen
import json
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.shell import _active_subprocesses, _kill_process_group, _subprocess_lock
from ouroboros.platform_layer import (
    merge_hidden_kwargs,
    subprocess_new_group_kwargs,
)
from ouroboros.contracts.skill_manifest import (
    SkillManifest,
    SkillManifestError,
    parse_skill_manifest_text,
)

log = logging.getLogger(__name__)

_PREFLIGHT_TIMEOUT_SEC = 30
_PREFLIGHT_MAX_OUTPUT_BYTES = 16 * 1024
_PREFLIGHT_HARD_FILE_LIMIT = 60  # independent preflight headroom (skill_review now uses a pack-level token budget)

# Extension -> argv template + runtime; {path} is substituted into argv only.
_VALIDATORS: Dict[str, Tuple[List[str], str]] = {
    ".js": (["node", "--check", "{path}"], "node"),
    ".mjs": (["node", "--check", "{path}"], "node"),
    ".cjs": (["node", "--check", "{path}"], "node"),
    ".sh": (["bash", "-n", "{path}"], "bash"),
    ".bash": (["bash", "-n", "{path}"], "bash"),
}


def _resolve_runtime(runtime: str) -> Optional[str]:
    if runtime == "python3":
        return shutil.which("python3") or shutil.which("python")
    if runtime == "node":
        # Prefer the bundled, signed node over a PATH (Homebrew) node that macOS
        # code-signing enforcement may SIGKILL inside the packaged app.
        try:
            from ouroboros.platform_layer import resolve_bundled_node
            bundled = resolve_bundled_node()
            if bundled:
                return bundled
        except Exception:
            log.debug("resolve_bundled_node failed", exc_info=True)
    return shutil.which(runtime)


def _run_check(cmd: List[str], cwd: pathlib.Path) -> Dict[str, Any]:
    """Run validator argv through panic-tracked subprocess machinery; never raises."""
    popen_kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
        "cwd": str(cwd),
        "env": {
            "PATH": str(__import__("os").environ.get("PATH", "")),
            "HOME": str(__import__("os").environ.get("HOME", "")),
            "LANG": "C.UTF-8",
        },
    }
    popen_kwargs.update(subprocess_new_group_kwargs())
    try:
        proc = Popen(cmd, **merge_hidden_kwargs(popen_kwargs))  # noqa: S603 — argv array
    except FileNotFoundError as exc:
        return {"returncode": -1, "stdout": "", "stderr": f"runtime not found: {exc}", "timeout": False}
    with _subprocess_lock:
        _active_subprocesses.add(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=_PREFLIGHT_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            try:
                _kill_process_group(proc)
            except Exception:  # pragma: no cover
                log.debug("kill_process_tree failed", exc_info=True)
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except Exception:
                stdout, stderr = b"", b""
            return {
                "returncode": -9,
                "stdout": stdout.decode("utf-8", errors="replace")[:_PREFLIGHT_MAX_OUTPUT_BYTES],
                "stderr": stderr.decode("utf-8", errors="replace")[:_PREFLIGHT_MAX_OUTPUT_BYTES],
                "timeout": True,
            }
    finally:
        with _subprocess_lock:
            _active_subprocesses.discard(proc)
    return {
        "returncode": int(proc.returncode or 0),
        "stdout": (stdout or b"").decode("utf-8", errors="replace")[:_PREFLIGHT_MAX_OUTPUT_BYTES],
        "stderr": (stderr or b"").decode("utf-8", errors="replace")[:_PREFLIGHT_MAX_OUTPUT_BYTES],
        "timeout": False,
    }


def _run_python_syntax_check(path: pathlib.Path) -> Dict[str, Any]:
    """Use compile() so Python syntax checks stay read-only."""
    try:
        text = path.read_text(encoding="utf-8")
        compile(text, str(path), "exec")
        return {"returncode": 0, "stdout": "", "stderr": "", "timeout": False}
    except Exception as exc:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "timeout": False,
        }


def _validate_widget_render(render: Any, *, source: str) -> Dict[str, Any]:
    """Validate a declarative/module widget render block without importing plugin code."""
    try:
        from ouroboros.extension_loader import _validate_ui_render  # pylint: disable=W0212
        from ouroboros.contracts.plugin_api import ExtensionRegistrationError

        _validate_ui_render(render if isinstance(render, dict) else {})
        return {"item": "widget_schema", "source": source, "ok": True, "detail": "ok"}
    except ExtensionRegistrationError as exc:
        return {"item": "widget_schema", "source": source, "ok": False, "detail": str(exc)}
    except Exception as exc:
        return {
            "item": "widget_schema",
            "source": source,
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _literal_widget_renders_from_plugin(plugin_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Extract literal top-level widget render dicts without importing plugin.py."""
    try:
        tree = ast.parse(plugin_path.read_text(encoding="utf-8"), filename=str(plugin_path))
    except Exception:
        return []
    renders: List[Dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        if not isinstance(value, dict):
            continue
        kind = str(value.get("kind") or "").strip()
        if kind not in {"declarative", "module", "iframe"}:
            continue
        targets = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        source = targets[0] if targets else f"line {getattr(node, 'lineno', '?')}"
        renders.append({"source": source, "render": value})
    return renders


def _widget_schema_findings(skill_dir: pathlib.Path, manifest: Optional[SkillManifest]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if manifest is not None and isinstance(manifest.ui_tab, dict):
        render = manifest.ui_tab.get("render")
        findings.append(_validate_widget_render(render, source="manifest.ui_tab.render"))
    plugin = skill_dir / "plugin.py"
    if plugin.is_file():
        for item in _literal_widget_renders_from_plugin(plugin):
            findings.append(
                _validate_widget_render(
                    item.get("render"),
                    source=f"plugin.py:{item.get('source')}",
                )
            )
    return findings


def _plugin_permission_findings(skill_dir: pathlib.Path, manifest: Optional[SkillManifest]) -> List[Dict[str, Any]]:
    """Statically catch common PluginAPI calls whose manifest permission is missing."""
    if manifest is None or not manifest.is_extension():
        return []
    plugin = skill_dir / (manifest.entry or "plugin.py")
    if not plugin.is_file():
        return []
    required_by_call = {
        "register_route": "route",
        "register_tool": "tool",
        "register_ui_tab": "widget",
        "register_settings_section": "widget",
        "register_ws_handler": "ws_handler",
        "send_ws_message": "ws_handler",
        "get_settings": "read_settings",
    }
    try:
        tree = ast.parse(plugin.read_text(encoding="utf-8"), filename=str(plugin))
    except Exception:
        return []
    seen: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            perm = required_by_call.get(func.attr)
            if perm and perm not in seen:
                seen[perm] = getattr(node, "lineno", 0)
    declared = set(manifest.permissions or [])
    findings: List[Dict[str, Any]] = []
    for perm, line in sorted(seen.items()):
        findings.append({
            "item": "permission_static",
            "source": f"{plugin.name}:{line}" if line else plugin.name,
            "permission": perm,
            "ok": perm in declared,
            "detail": "ok" if perm in declared else f"plugin calls PluginAPI surface requiring permission {perm!r}",
        })
    return findings


def _handle_skill_preflight(
    ctx: ToolContext,
    skill: str = "",
    paths: Optional[List[str]] = None,
    **_kwargs: Any,
) -> str:
    skill_name = str(skill or "").strip()
    if not skill_name:
        return "⚠️ SKILL_PREFLIGHT_ERROR: 'skill' is required."

    from ouroboros.skill_loader import find_skill

    drive_root = pathlib.Path(ctx.drive_root)
    loaded = find_skill(drive_root, skill_name)
    if loaded is None:
        return f"⚠️ SKILL_PREFLIGHT_ERROR: skill {skill_name!r} not found."

    skill_dir = loaded.skill_dir.resolve()

    # Broken manifests still become findings; keep other validators running.
    manifest_findings: List[Dict[str, Any]] = []
    widget_findings: List[Dict[str, Any]] = []
    permission_findings: List[Dict[str, Any]] = []
    manifest: Optional[SkillManifest] = None
    manifest_path = None
    for candidate in ("SKILL.md", "skill.json"):
        cand = skill_dir / candidate
        if cand.is_file():
            manifest_path = cand
            break
    if manifest_path is None:
        manifest_findings.append({"item": "manifest_present", "ok": False, "detail": "no SKILL.md / skill.json"})
    else:
        try:
            text = manifest_path.read_text(encoding="utf-8")
            manifest = parse_skill_manifest_text(text)
            manifest_findings.append({"item": "manifest_parse", "ok": True, "detail": "ok"})
            widget_findings.extend(_widget_schema_findings(skill_dir, manifest))
            permission_findings.extend(_plugin_permission_findings(skill_dir, manifest))
            if manifest.entry:
                entry = (skill_dir / manifest.entry).resolve()
                ok = entry.is_file()
                try:
                    entry.relative_to(skill_dir)
                except ValueError:
                    ok = False
                manifest_findings.append({
                    "item": "manifest_entry_exists",
                    "ok": ok,
                    "detail": manifest.entry if ok else f"missing or escaping entry: {manifest.entry}",
                })
            for script in manifest.scripts or []:
                name = str(script.get("name") or "").strip()
                if not name:
                    continue
                rel = name if "/" in name or name.startswith(".") else f"scripts/{name}"
                script_path = (skill_dir / rel).resolve()
                ok = script_path.is_file()
                try:
                    script_path.relative_to(skill_dir)
                except ValueError:
                    ok = False
                manifest_findings.append({
                    "item": "manifest_script_exists",
                    "ok": ok,
                    "detail": rel if ok else f"missing or escaping script: {rel}",
                })
        except (OSError, UnicodeDecodeError, SkillManifestError) as exc:
            manifest_findings.append({
                "item": "manifest_parse",
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            })
            widget_findings.extend(_widget_schema_findings(skill_dir, None))

    # paths scopes recent edits; otherwise walk the reviewable payload surface.
    files_to_check: List[pathlib.Path] = []
    path_findings: List[Dict[str, Any]] = []
    if paths:
        for raw in paths:
            rel = str(raw or "").strip()
            if not rel or rel.startswith("/") or rel.startswith("~") or ".." in rel.split("/"):
                path_findings.append({"path": rel, "runtime": "", "ok": False, "stderr": "invalid or escaping path", "stdout": "", "timeout": False})
                continue
            target = (skill_dir / rel).resolve()
            try:
                target.relative_to(skill_dir)
            except ValueError:
                path_findings.append({"path": rel, "runtime": "", "ok": False, "stderr": "path escapes skill directory", "stdout": "", "timeout": False})
                continue
            if target.is_file():
                files_to_check.append(target)
            else:
                path_findings.append({"path": rel, "runtime": "", "ok": False, "stderr": "path not found", "stdout": "", "timeout": False})
    else:
        try:
            from ouroboros.skill_loader import _iter_payload_files  # pylint: disable=W0212
            for path in _iter_payload_files(
                skill_dir,
                manifest_entry=loaded.manifest.entry,
                manifest_scripts=loaded.manifest.scripts,
            ):
                files_to_check.append(path.resolve())
        except Exception as exc:
            log.debug("preflight discovery failed", exc_info=True)
            return f"⚠️ SKILL_PREFLIGHT_ERROR: payload discovery failed: {exc}"

    omitted_count = 0
    omitted_files: List[str] = []
    if len(files_to_check) > _PREFLIGHT_HARD_FILE_LIMIT:
        omitted = files_to_check[_PREFLIGHT_HARD_FILE_LIMIT:]
        omitted_count = len(omitted)
        omitted_files = [str(path.relative_to(skill_dir)) for path in omitted[:20]]
        files_to_check = files_to_check[:_PREFLIGHT_HARD_FILE_LIMIT]

    file_findings: List[Dict[str, Any]] = list(path_findings)
    for path in files_to_check:
        suffix = path.suffix.lower()
        if suffix == ".py":
            result = _run_python_syntax_check(path)
            ok = result["returncode"] == 0 and not result["timeout"]
            file_findings.append({
                "path": str(path.relative_to(skill_dir)),
                "runtime": "python",
                "ok": ok,
                "returncode": result["returncode"],
                "timeout": result["timeout"],
                "stderr": result["stderr"][:2000],
                "stdout": result["stdout"][:2000],
            })
            continue
        validator = _VALIDATORS.get(suffix)
        if validator is None:
            continue
        argv_template, runtime = validator
        runtime_path = _resolve_runtime(runtime)
        rel_path = str(path.relative_to(skill_dir))
        if runtime_path is None:
            # Missing external runtime is an environment gap, not a syntax
            # verdict. Skip it (do not block); tri-model review still reads the
            # file in full.
            file_findings.append({
                "path": rel_path,
                "runtime": runtime,
                "ok": True,
                "skipped": True,
                "skip_reason": "runtime_unavailable",
                "detail": f"{runtime} not on PATH — syntax not verified; relying on tri-model review",
            })
            continue
        cmd = [runtime_path] + [str(path) if part == "{path}" else part for part in argv_template[1:]]
        result = _run_check(cmd, cwd=skill_dir)
        rc = result["returncode"]
        timed_out = bool(result["timeout"])
        if timed_out or (isinstance(rc, int) and rc < 0):
            # The validator process itself failed to run to completion — e.g. a
            # Homebrew `node` killed by macOS code-signing enforcement
            # (SIGKILL), or a timeout. This is infrastructure, not a syntax
            # error: only a clean non-zero exit (rc > 0) means bad syntax.
            # Skip so a working skill is not falsely blocked; tri-model review
            # remains the authoritative gate.
            file_findings.append({
                "path": rel_path,
                "runtime": runtime,
                "ok": True,
                "skipped": True,
                "skip_reason": "validator_timeout" if timed_out else "validator_killed",
                "detail": (
                    f"{runtime} syntax not verified ("
                    + ("timed out" if timed_out else f"process killed, signal {-rc}")
                    + "); relying on tri-model review"
                ),
                "returncode": rc,
                "timeout": timed_out,
                "stderr": result["stderr"][:2000],
            })
            continue
        ok = rc == 0
        file_findings.append({
            "path": rel_path,
            "runtime": runtime,
            "ok": ok,
            "returncode": rc,
            "timeout": timed_out,
            "stderr": result["stderr"][:2000],
            "stdout": result["stdout"][:2000],
        })

    # ok iff every contract check passes and every file that was ACTUALLY
    # validated (not skipped) is clean. Skipped findings (missing/killed/
    # timed-out validators) never block — they are environment limitations, not
    # syntax failures — and a file set with nothing syntax-checkable (e.g. only
    # .txt/.md, or all validators skipped) is tolerated; tri-model review stays
    # authoritative. A real syntax error (rc > 0) on any non-skipped file blocks.
    # A file count beyond the syntax-check headroom is a DEGRADED note, NOT a block: the
    # skill-review pass now packs the whole payload under a pack-level token budget (chunked
    # when oversized) and reads every file, so an arbitrary preflight file-count cap must not
    # re-introduce the hard gate the token budget replaced. omitted files are surfaced below.
    overall_ok = (
        all(f.get("ok") for f in manifest_findings)
        and all(f.get("ok") for f in widget_findings)
        and all(f.get("ok") for f in permission_findings)
        and all(f.get("ok") for f in file_findings if not f.get("skipped"))
    )
    skipped_files = [f for f in file_findings if f.get("skipped")]
    payload = {
        "skill": skill_name,
        "skill_dir": str(skill_dir),
        "manifest": manifest_findings,
        "widgets": widget_findings,
        "permissions": permission_findings,
        "files": file_findings,
        "files_checked": len(file_findings),
        "files_failed": sum(1 for f in file_findings if not f.get("ok") and not f.get("skipped")),
        "files_skipped": len(skipped_files),
        "omitted_count": omitted_count,
        "omitted_files": omitted_files,
        "ok": bool(overall_ok),
    }
    notes: List[str] = []
    if skipped_files:
        notes.append(
            f"{len(skipped_files)} validator(s) could not run "
            f"({', '.join(sorted({str(f.get('skip_reason') or 'skipped') for f in skipped_files}))}); "
            "syntax for those files was not verified and tri-model review remains authoritative."
        )
    if omitted_count:
        notes.append(
            f"{omitted_count} file(s) beyond the {_PREFLIGHT_HARD_FILE_LIMIT}-file syntax-check headroom "
            "were not individually syntax-checked; the skill-review pass reads every file under its "
            "pack-level token budget (chunked when oversized) and remains authoritative."
        )
    if notes:
        # Surface the degradation explicitly (no silent skip): tri-model review
        # is authoritative for these files.
        payload["degraded"] = True
        payload["degraded_note"] = " ".join(notes)
    return json.dumps(payload, ensure_ascii=False, indent=2)


_PREFLIGHT_SCHEMA = {
    "name": "skill_preflight",
    "description": (
        "Read-only payload syntax/contract validator for one skill. Runs Python "
        "compile() (no __pycache__), node --check, and bash -n on every reviewable file "
        "(or just the ones in `paths` if provided), plus a manifest "
        "parse and static widget render-schema validation. Cheap and offline (no LLM, no review.json mutation, "
        "no review status change). Heal-mode agents use this before "
        "calling skill_review so silly syntax errors are caught "
        "without spending tri-model review tokens. Argv-only "
        "subprocess invocation, cwd=skill_dir, scrubbed env, 30s "
        "per-file cap, panic-tracked process group."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Skill name (directory basename in the skills tree).",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional subset of payload files (relative to the "
                    "skill dir) to validate. Empty = walk the same "
                    "surface skill_review reads."
                ),
            },
        },
        "required": ["skill"],
    },
}


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="skill_preflight",
            schema=_PREFLIGHT_SCHEMA,
            handler=_handle_skill_preflight,
            is_code_tool=False,
            timeout_sec=120,
        ),
    ]


__all__ = ["get_tools"]

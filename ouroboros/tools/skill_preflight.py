"""Read-only skill payload syntax/contract preflight for heal-mode agents.

Tri-model review remains authoritative. Preflight uses argv-only subprocesses,
cwd=skill_dir, scrubbed env, 30s timeout, in-process Python compile(), and no
review/enablement/grant state mutation.
"""

from __future__ import annotations

import ast
import logging
import os
import pathlib
import shutil
import subprocess
import time
from subprocess import Popen
import json
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    load_bound_skill,
)
from ouroboros.tools.process_facts import publish_process_facts, signal_name_for_returncode
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
from ouroboros.contracts.plugin_api import ExtensionRegistrationError
from ouroboros.extension_ui_validation import (
    validate_settings_schema,
    validate_ui_render,
)

log = logging.getLogger(__name__)

_PREFLIGHT_TIMEOUT_SEC = 30
_PREFLIGHT_MAX_OUTPUT_BYTES = 16 * 1024
_PREFLIGHT_HARD_FILE_LIMIT = 60  # independent preflight headroom (skill_review now uses a pack-level token budget)
# The scrubbed validator env keeps the Windows process-base variables a child
# needs to START: without SystemRoot a Windows process cannot initialise the
# system services it loads (node dies in libuv's Winsock startup before it
# reads the script), so `node --check` on a VALID file exited non-zero and
# preflight reported a syntax error that did not exist (windows-latest,
# 7.0.0-rc.9). TEMP/TMP back the msys `bash -n` /tmp. Forwarded only when set,
# so a POSIX child env is byte-identical to before; the same base-env class as
# extension_companion._COMPANION_BASE_ENV_KEYS (SYSTEMROOT/WINDIR/TEMP/TMP) and
# CPython's own test.support.script_helper ("Windows requires at least the
# SYSTEMROOT environment variable to start").
_WINDOWS_BASE_ENV_KEYS: Tuple[str, ...] = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP")

# Extension -> argv template + runtime; {path} is substituted into argv only.
_VALIDATORS: Dict[str, Tuple[List[str], str]] = {
    ".js": (["node", "--check", "{path}"], "node"),
    ".mjs": (["node", "--check", "{path}"], "node"),
    ".cjs": (["node", "--check", "{path}"], "node"),
    ".sh": (["bash", "-n", "{path}"], "bash"),
    ".bash": (["bash", "-n", "{path}"], "bash"),
}

# A declared module-widget entry is injected as a CLASSIC inline <script>, so the
# grammar that matters is the Script goal, not the module goal. `node --check`
# accepts top-level import/export in a .js file; `new vm.Script(...)` rejects it
# exactly as the browser will, and it is suffix-independent.
_CLASSIC_SCRIPT_VALIDATOR: Tuple[List[str], str] = (
    [
        "node",
        "-e",
        "const fs=require('fs'),vm=require('vm');"
        "new vm.Script(fs.readFileSync(process.argv[1],'utf8'),{filename:process.argv[1]})",
        "--",
        "{path}",
    ],
    "node",
)


def _resolve_runtime(runtime: str) -> Tuple[Optional[str], str]:
    """Resolve a validator runtime: ``(path, "")`` or ``(None, reason)``."""
    if runtime == "python3":
        return (shutil.which("python3") or shutil.which("python")), ""
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
        except Exception:
            log.debug("select_skill_node_runtime failed", exc_info=True)
    return shutil.which(runtime), ""


def _run_check(cmd: List[str], cwd: pathlib.Path) -> Dict[str, Any]:
    """Run validator argv through panic-tracked subprocess machinery; never raises.

    Returns the validator's own facts, never a synthesized stand-in: a
    validator the host killed on the preflight deadline, and one that never
    reached exec, both report ``returncode=None`` beside the typed reason
    (``timeout`` / ``pre_exec_failure``). Earlier revisions answered ``-9`` and
    ``-1`` there, which read downstream as real POSIX signal deaths — including
    on Windows, where the host kill produces neither. The abnormal outcomes are
    also published on the typed process-facts channel so the preflight CALL's
    result_meta, tools.jsonl row and UI card carry them; a clean validator is
    described by its finding and leaves the channel alone."""
    env: Dict[str, str] = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C.UTF-8",
    }
    env.update({key: os.environ[key] for key in _WINDOWS_BASE_ENV_KEYS if os.environ.get(key)})
    popen_kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
        "cwd": str(cwd),
        "env": env,
    }
    popen_kwargs.update(subprocess_new_group_kwargs())
    _check_started_ts = time.monotonic()
    try:
        proc = Popen(cmd, **merge_hidden_kwargs(popen_kwargs))  # noqa: S603 — argv array
    except FileNotFoundError as exc:
        publish_process_facts(
            started_ts=_check_started_ts, pre_exec_failure="FileNotFoundError",
        )
        return {
            "returncode": None,
            "stdout": "",
            "stderr": f"runtime not found: {exc}",
            "timeout": False,
            "pre_exec_failure": "FileNotFoundError",
        }
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
            publish_process_facts(
                started_ts=_check_started_ts, timed_out=True, killed_by_host=True,
            )
            return {
                "returncode": None,
                "stdout": stdout.decode("utf-8", errors="replace")[:_PREFLIGHT_MAX_OUTPUT_BYTES],
                "stderr": stderr.decode("utf-8", errors="replace")[:_PREFLIGHT_MAX_OUTPUT_BYTES],
                "timeout": True,
                "killed_by_host": True,
            }
    finally:
        with _subprocess_lock:
            _active_subprocesses.discard(proc)
    returncode = int(proc.returncode or 0)
    if returncode < 0:
        # A validator killed by something OTHER than our deadline (the macOS
        # code-signing SIGKILL this skip path exists for) is a real signal
        # death: publish it so the call carries the exit code and signal name.
        publish_process_facts(returncode=returncode, started_ts=_check_started_ts)
    return {
        "returncode": returncode,
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


def _validate_widget_render(
    render: Any,
    *,
    source: str,
    settings: bool = False,
) -> Dict[str, Any]:
    """Validate one statically resolved UI declaration without importing plugin code."""
    try:
        clean = validate_settings_schema(render) if settings else validate_ui_render(render)
        row = {
            "item": "widget_schema",
            "source": source,
            "ok": True,
            "verified": True,
            "detail": "ok",
        }
        if clean.get("kind") == "module":
            # The normalized entry is what the frame will fetch; carry it so the
            # caller can check the declared file actually exists on disk.
            row["entry"] = clean["entry"]
        return row
    except ExtensionRegistrationError as exc:
        return {
            "item": "widget_schema",
            "source": source,
            "ok": False,
            "verified": True,
            "detail": str(exc),
        }
    except Exception as exc:
        return {
            "item": "widget_schema",
            "source": source,
            "ok": False,
            "verified": True,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _dynamic_ui_schema_finding(*, source: str) -> Dict[str, Any]:
    return {
        "item": "widget_schema",
        "source": source,
        "ok": True,
        "verified": False,
        "skipped": True,
        "skip_reason": "dynamic_ui_schema",
        "detail": "schema is dynamic; runtime registration remains the fail-closed validator",
    }


def _simple_helper_return(node: ast.FunctionDef) -> Optional[ast.AST]:
    """Return the sole safe expression from a zero-argument literal helper."""
    args = node.args
    if (
        node.decorator_list
        or args.posonlyargs
        or args.args
        or args.kwonlyargs
        or args.vararg is not None
        or args.kwarg is not None
    ):
        return None
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    return body[0].value


def _resolve_static_ui_value(
    node: ast.AST,
    *,
    assignments: Dict[str, ast.AST],
    helpers: Dict[str, ast.AST],
    shadowed: frozenset[str] = frozenset(),
    resolving: frozenset[str] = frozenset(),
) -> tuple[bool, Any]:
    """Resolve only literals, module literal names, or safe zero-arg helpers."""
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    if isinstance(node, ast.Name):
        marker = f"name:{node.id}"
        if node.id in shadowed or marker in resolving or node.id not in assignments:
            return False, None
        return _resolve_static_ui_value(
            assignments[node.id],
            assignments=assignments,
            helpers=helpers,
            resolving=resolving | {marker},
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.args
        and not node.keywords
        and node.func.id not in shadowed
        and node.func.id in helpers
    ):
        marker = f"helper:{node.func.id}"
        if marker in resolving:
            return False, None
        return _resolve_static_ui_value(
            helpers[node.func.id],
            assignments=assignments,
            helpers=helpers,
            resolving=resolving | {marker},
        )
    return False, None


def _function_local_bindings(node: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Return names bound by one function without entering nested scopes."""
    bindings = {
        arg.arg
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    }
    if node.args.vararg is not None:
        bindings.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        bindings.add(node.args.kwarg.arg)
    global_names: set[str] = set()
    nonlocal_names: set[str] = set()

    class BindingVisitor(ast.NodeVisitor):
        def visit_Name(self, item: ast.Name) -> None:
            if isinstance(item.ctx, ast.Store):
                bindings.add(item.id)

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
            bindings.add(item.name)

        def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
            bindings.add(item.name)

        def visit_ClassDef(self, item: ast.ClassDef) -> None:
            bindings.add(item.name)

        def visit_Import(self, item: ast.Import) -> None:
            for alias in item.names:
                bindings.add(alias.asname or alias.name.split(".", 1)[0])

        def visit_ImportFrom(self, item: ast.ImportFrom) -> None:
            for alias in item.names:
                if alias.name != "*":
                    bindings.add(alias.asname or alias.name)

        def visit_ExceptHandler(self, item: ast.ExceptHandler) -> None:
            if item.name:
                bindings.add(item.name)
            self.generic_visit(item)

        def visit_MatchAs(self, item: ast.MatchAs) -> None:
            if item.name:
                bindings.add(item.name)
            self.generic_visit(item)

        def visit_MatchStar(self, item: ast.MatchStar) -> None:
            if item.name:
                bindings.add(item.name)

        def visit_MatchMapping(self, item: ast.MatchMapping) -> None:
            if item.rest:
                bindings.add(item.rest)
            self.generic_visit(item)

        def visit_Lambda(self, item: ast.Lambda) -> None:
            return

        def visit_Global(self, item: ast.Global) -> None:
            global_names.update(item.names)

        def visit_Nonlocal(self, item: ast.Nonlocal) -> None:
            nonlocal_names.update(item.names)

    visitor = BindingVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return frozenset(bindings - global_names - nonlocal_names)


def _enclosing_function_bindings(
    node: ast.AST,
    *,
    parents: Dict[ast.AST, ast.AST],
    cache: Dict[ast.AST, frozenset[str]],
) -> frozenset[str]:
    bindings: set[str] = set()
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local = cache.get(current)
            if local is None:
                local = _function_local_bindings(current)
                cache[current] = local
            bindings.update(local)
        current = parents.get(current)
    return frozenset(bindings)


def _registered_ui_schema_findings(plugin_path: pathlib.Path, *, source_name: str) -> List[Dict[str, Any]]:
    """Analyze only actual PluginAPI UI registration calls, without executing code."""
    try:
        tree = ast.parse(plugin_path.read_text(encoding="utf-8"), filename=str(plugin_path))
    except Exception:
        return []
    assignments: Dict[str, ast.AST] = {}
    helpers: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignments[node.target.id] = node.value
        elif isinstance(node, ast.FunctionDef):
            returned = _simple_helper_return(node)
            if returned is not None:
                helpers[node.name] = returned

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    entry_receivers: Dict[ast.AST, str] = {}
    module_classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "register":
            continue
        positional = (*node.args.posonlyargs, *node.args.args)
        if positional:
            entry_receivers[node] = positional[0].arg
    function_bindings: Dict[ast.AST, frozenset[str]] = {}

    findings: List[Dict[str, Any]] = []
    schema_keywords = {
        "register_ui_tab": ("render", False),
        "register_settings_section": ("schema", True),
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        registration = node.func.attr
        target = schema_keywords.get(registration)
        if target is None:
            continue
        keyword_name, settings = target
        keyword = next((item for item in node.keywords if item.arg == keyword_name), None)
        if keyword is None:
            continue
        source = f"{source_name}:{getattr(node, 'lineno', '?')} {registration}.{keyword_name}"
        current = parents.get(node)
        nested_scope = False
        entry_receiver = ""
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if current in entry_receivers:
                    entry_receiver = entry_receivers[current]
                    break
                nested_scope = True
            current = parents.get(current)
        if not entry_receiver:
            continue
        receiver = node.func.value
        if (
            isinstance(receiver, ast.Name)
            and receiver.id != entry_receiver
            and receiver.id in module_classes
        ):
            # A local class can expose an identically named method without
            # participating in the PluginAPI entry contract.
            continue
        if nested_scope or not isinstance(receiver, ast.Name) or receiver.id != entry_receiver:
            # An alias/proxy/closure may still reach PluginAPI at runtime, but
            # static preflight cannot prove that without executing plugin code.
            findings.append(_dynamic_ui_schema_finding(source=source))
            continue
        resolved, value = _resolve_static_ui_value(
            keyword.value,
            assignments=assignments,
            helpers=helpers,
            shadowed=_enclosing_function_bindings(
                node,
                parents=parents,
                cache=function_bindings,
            ),
        )
        if not resolved:
            findings.append(_dynamic_ui_schema_finding(source=source))
            continue
        findings.append(_validate_widget_render(value, source=source, settings=settings))
    return findings


def _widget_entry_exists_finding(
    skill_dir: pathlib.Path,
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Check that a module widget's declared render.entry file is really there."""
    entry = str(row.get("entry") or "").strip()
    if not entry:
        return None
    try:
        target = (skill_dir / entry).resolve()
        target.relative_to(skill_dir.resolve())
        ok = target.is_file()
    except ValueError:
        ok = False
    return {
        "item": "widget_entry_exists",
        "source": str(row.get("source") or ""),
        "ok": ok,
        "verified": True,
        "detail": entry if ok else f"missing or escaping module widget entry: {entry}",
    }


def _widget_schema_findings(skill_dir: pathlib.Path, manifest: Optional[SkillManifest]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if manifest is not None and isinstance(manifest.ui_tab, dict):
        render = manifest.ui_tab.get("render")
        schema_row = _validate_widget_render(render, source="manifest.ui_tab.render")
        findings.append(schema_row)
        # Generated before the containment early return below: an escaping
        # plugin path must not silently drop the manifest entry check.
        entry_row = _widget_entry_exists_finding(skill_dir, schema_row)
        if entry_row is not None:
            findings.append(entry_row)
    entry_name = str(manifest.entry or "plugin.py") if manifest is not None else "plugin.py"
    plugin = (skill_dir / entry_name).resolve()
    try:
        relative_plugin = plugin.relative_to(skill_dir.resolve())
    except ValueError:
        return findings
    if plugin.is_file():
        for schema_row in _registered_ui_schema_findings(plugin, source_name=relative_plugin.as_posix()):
            findings.append(schema_row)
            entry_row = _widget_entry_exists_finding(skill_dir, schema_row)
            if entry_row is not None:
                findings.append(entry_row)
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
    # Receiver proof (#447 A6): a method NAME alone does not prove a PluginAPI
    # call — `OtherLibrary().get_settings()` made a skill permanently
    # non-executable (STATUS_PENDING, no advisory override). Only a call whose
    # receiver is provably the `register(api)` parameter (or an alias assigned
    # from it) keeps blocking power; an unproven receiver DEGRADES to an ok=True
    # note — the disposition every other unprovable preflight check already gets
    # — and the tri-model review / runtime permission gate stay authoritative.
    api_names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "register"
            and node.args.args
        ):
            api_names.add(node.args.args[0].arg)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Name)
            and node.value.id in api_names
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    api_names.add(target.id)
    seen: dict[str, tuple[int, bool]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            perm = required_by_call.get(func.attr)
            if not perm:
                continue
            proven = isinstance(func.value, ast.Name) and func.value.id in api_names
            line, seen_proven = seen.get(perm, (0, False))
            if perm not in seen or (proven and not seen_proven):
                seen[perm] = (getattr(node, "lineno", 0), proven)
    declared = set(manifest.permissions or [])
    findings: List[Dict[str, Any]] = []
    for perm, (line, proven) in sorted(seen.items()):
        if perm in declared:
            detail = "ok"
        elif proven:
            detail = f"plugin calls PluginAPI surface requiring permission {perm!r}"
        else:
            detail = (
                f"call names a PluginAPI-like surface requiring permission {perm!r}, but the "
                "receiver is not provably the plugin's `api` object — not statically required "
                "(reviewers and the runtime permission gate stay authoritative)"
            )
        finding: Dict[str, Any] = {
            "item": "permission_static",
            "source": f"{plugin.name}:{line}" if line else plugin.name,
            "permission": perm,
            "ok": perm in declared or not proven,
            "detail": detail,
        }
        if perm not in declared and not proven:
            finding["degraded"] = True
        findings.append(finding)
    return findings


def _presence_profile_findings(
    skill_dir: pathlib.Path,
    manifest: SkillManifest,
) -> List[Dict[str, Any]]:
    """Validate the optional reviewed presence declaration without live state."""
    try:
        from ouroboros.presence_profile import (
            PresenceProfileError,
            parse_presence_profile,
            presence_profile_fingerprint,
        )
    except Exception as exc:
        log.exception("presence profile preflight failed")
        return [{
            "item": "presence_profile",
            "ok": False,
            "code": "presence_internal_error",
            "field": "presence",
            "detail": f"{type(exc).__name__}: presence validation failed",
        }]
    try:
        profile = parse_presence_profile(manifest, skill_dir)
    except PresenceProfileError as exc:
        return [{
            "item": "presence_profile",
            "ok": False,
            "code": exc.code,
            "field": exc.field,
            "detail": str(exc),
        }]
    except Exception as exc:
        log.exception("presence profile preflight failed")
        return [{
            "item": "presence_profile",
            "ok": False,
            "code": "presence_internal_error",
            "field": "presence",
            "detail": f"{type(exc).__name__}: presence validation failed",
        }]
    if profile is None:
        return []
    return [{
        "item": "presence_profile",
        "ok": True,
        "code": "",
        "field": "presence",
        "detail": "ok",
        "profile_fingerprint": presence_profile_fingerprint(profile),
        "capability_requests": len(profile.capability_requests),
    }]


def _handle_skill_preflight(
    ctx: ToolContext,
    skill: str = "",
    paths: Optional[List[str]] = None,
    _resolved_binding: ResolvedResourceBinding | None = None,
    **_kwargs: Any,
) -> str:
    skill_name = str(skill or "").strip()
    if not skill_name:
        return "⚠️ SKILL_PREFLIGHT_ERROR: 'skill' is required."

    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx, root="skill_payload", operation="review", path=".",
            skill_name=skill_name,
        )
    except Exception as exc:
        return f"⚠️ SKILL_PREFLIGHT_ERROR: {exc}"
    loaded = load_bound_skill(binding)
    if loaded is None:
        return f"⚠️ SKILL_PREFLIGHT_ERROR: skill {skill_name!r} not found."

    skill_dir = binding.base_path

    # Broken manifests still become findings; keep other validators running.
    manifest_findings: List[Dict[str, Any]] = []
    widget_findings: List[Dict[str, Any]] = []
    permission_findings: List[Dict[str, Any]] = []
    presence_findings: List[Dict[str, Any]] = []
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
            presence_findings.extend(_presence_profile_findings(skill_dir, manifest))
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

    # Declared module-widget entries are parsed with classic-script grammar below.
    module_entries = {
        (skill_dir / str(f.get("detail") or "")).resolve()
        for f in widget_findings
        if f.get("item") == "widget_entry_exists" and f.get("ok")
    }

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
        classic_script = path in module_entries
        validator = _CLASSIC_SCRIPT_VALIDATOR if classic_script else _VALIDATORS.get(suffix)
        if validator is None:
            continue
        argv_template, runtime = validator
        grammar = {"grammar": "classic_script"} if classic_script else {}
        runtime_path, runtime_reason = _resolve_runtime(runtime)
        rel_path = str(path.relative_to(skill_dir))
        if runtime_path is None:
            # Missing/unusable external runtime is an environment gap, not a
            # syntax verdict. Skip it (do not block) and disclose the health
            # reason; tri-model review still reads the file in full.
            reason_note = f" ({runtime_reason})" if runtime_reason else ""
            file_findings.append({
                "path": rel_path,
                "runtime": runtime,
                "ok": True,
                "skipped": True,
                "skip_reason": "runtime_unavailable",
                "detail": (
                    f"{runtime} not usable{reason_note} — syntax not verified; "
                    "relying on tri-model review"
                ),
                **grammar,
            })
            continue
        cmd = [runtime_path] + [str(path) if part == "{path}" else part for part in argv_template[1:]]
        result = _run_check(cmd, cwd=skill_dir)
        rc = result["returncode"]
        timed_out = bool(result["timeout"])
        pre_exec = str(result.get("pre_exec_failure") or "")
        if timed_out or pre_exec or (isinstance(rc, int) and rc < 0):
            # The validator process itself failed to run to completion — e.g. a
            # Homebrew `node` killed by macOS code-signing enforcement
            # (SIGKILL), a preflight-deadline kill, or a runtime that vanished
            # between resolution and exec. This is infrastructure, not a syntax
            # error: only a clean non-zero exit (rc > 0) means bad syntax.
            # Skip so a working skill is not falsely blocked; tri-model review
            # remains the authoritative gate.
            if timed_out:
                skip_reason, detail = "validator_timeout", "timed out"
            elif pre_exec:
                skip_reason, detail = "validator_not_started", f"never started: {pre_exec}"
            else:
                # The signal NAME comes from the one host signal table, and is
                # empty on a platform with no such signal — the disclosed
                # residual, never a fabricated POSIX name.
                signal_name = signal_name_for_returncode(rc)
                skip_reason = "validator_killed"
                detail = f"process killed, {signal_name or f'exit {rc}'}"
            finding = {
                "path": rel_path,
                "runtime": runtime,
                "ok": True,
                "skipped": True,
                "skip_reason": skip_reason,
                "detail": f"{runtime} syntax not verified ({detail}); relying on tri-model review",
                "returncode": rc,
                "timeout": timed_out,
                "stderr": result["stderr"][:2000],
                **grammar,
            }
            if pre_exec:
                finding["pre_exec_failure"] = pre_exec
            if result.get("killed_by_host"):
                finding["killed_by_host"] = True
            file_findings.append(finding)
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
            **grammar,
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
        and all(f.get("ok") for f in presence_findings)
        and all(f.get("ok") for f in file_findings if not f.get("skipped"))
    )
    skipped_files = [f for f in file_findings if f.get("skipped")]
    dynamic_ui_findings = [
        finding
        for finding in widget_findings
        if finding.get("skip_reason") == "dynamic_ui_schema"
    ]
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
    if presence_findings:
        payload["presence"] = presence_findings
    notes: List[str] = []
    if skipped_files:
        notes.append(
            f"{len(skipped_files)} validator(s) could not run "
            f"({', '.join(sorted({str(f.get('skip_reason') or 'skipped') for f in skipped_files}))}); "
            "syntax for those files was not verified and tri-model review remains authoritative."
        )
    if dynamic_ui_findings:
        notes.append(
            f"{len(dynamic_ui_findings)} dynamic UI schema registration(s) could not be "
            "statically verified; runtime registration remains fail-closed."
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
        "compile() (no __pycache__), bash -n, and node syntax checks (a declared module-widget "
        "entry gets classic-script grammar) on every reviewable file, or just `paths`, plus a manifest "
        "parse, module-widget entry existence, and static render-schema validation. Cheap and offline (no LLM, no review.json mutation, "
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

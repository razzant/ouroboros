# tests/golden_traces/capture.py — trace recorder for the ToolRegistry.execute pipeline.
#
# Wraps (monkeypatch-style, with full restore) the key guard functions of the
# dispatch pipeline and records (call order, function name, small deterministic
# detail payload) per guard, plus the final text execute() returned. All strings
# pass through a Normalizer that replaces scenario tmp roots with placeholders
# ({REPO}, {DRIVE}, {WS}, {TMP}, ...) and strips run-to-run noise (pids,
# timestamps, git shas, durations).
#
# check_safety (ouroboros/safety.py) is REPLACED by a stub returning
# (True, None) — no LLM, no network — but the call itself (tool name + sorted
# argument names) is recorded. TOOL_POLICY stays untouched: the stub sits at
# the check_safety boundary only.
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import sys
from typing import Any, Callable, Dict, List

import ouroboros.safety as safety_mod
import ouroboros.tools.registry as registry_mod
from ouroboros.tools.registry import ToolRegistry


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
_REGEX_PASSES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"pid=\d+"), "pid={PID}"),
    (re.compile(r"\bpid \d+\b"), "pid {PID}"),
    # JSON-rendered process/timing fields (run_command / services / executor trace).
    (re.compile(r"\"(pid|pgid)\": \d+"), '"\\1": {PID}'),
    (re.compile(r"\"(uptime_sec|elapsed_sec)\": \d+(?:\.\d+)?"), '"\\1": {DUR}'),
    # observability.write_blob gzip size: the gzip FNAME header embeds the temp
    # file's basename (contains the pid), so the byte size varies per run/env.
    (re.compile(r"\"compressed_size\": \d+"), '"compressed_size": {SIZE}'),
    (re.compile(r"\"executor_id\": \"[0-9a-f]+\""), '"executor_id": "{ID}"'),
    # run_script temp script files carry a uuid4 hex name under the task drive.
    (re.compile(r"script_[0-9a-f]{16,40}\.py"), "script_{HEX}.py"),
    (re.compile(r"\b[0-9a-f]{40}\b"), "{SHA}"),
    (re.compile(r"\b[0-9a-f]{7,12}(?=\.\.|\s|$)"), "{SHA}"),
    # ISO timestamps (verify receipts, service uptime stamps).
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "{TS}"),
    # Elapsed / uptime seconds: "0.03s", "uptime=1.2s", "in 0.01 s".
    (re.compile(r"\b\d+(?:\.\d+)?s\b"), "{DUR}"),
    (re.compile(r"uptime=\d+(?:\.\d+)?"), "uptime={DUR}"),
]


class Normalizer:
    """Replace scenario roots and machine noise in strings with placeholders."""

    def __init__(self, roots: Dict[str, str]):
        # roots: placeholder name -> raw path text. Longest expansion first so
        # nested roots (repo under tmp base) win over their parents.
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        auto = dict(roots)
        auto.setdefault("PYTHON", sys.executable)
        auto.setdefault("DATA", os.environ.get("OUROBOROS_DATA_DIR", ""))
        auto.setdefault("SRC", str(pathlib.Path(registry_mod.__file__).resolve().parents[2]))
        auto.setdefault("HOME", str(pathlib.Path.home()))
        for placeholder, raw in auto.items():
            if not raw:
                continue
            p = pathlib.Path(raw)
            variants = {str(p), str(p.resolve(strict=False)), os.path.realpath(str(p))}
            for text in variants:
                if text and text != "/" and text not in seen:
                    seen.add(text)
                    pairs.append((text, "{%s}" % placeholder))
        pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
        self._pairs = pairs

    def text(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        out = value
        for raw, placeholder in self._pairs:
            out = out.replace(raw, placeholder)
        for pattern, repl in _REGEX_PASSES:
            out = pattern.sub(repl, out)
        return out

    def tree(self, value: Any) -> Any:
        """Normalize strings recursively inside plain containers."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {str(k): self.tree(value[k]) for k in sorted(value, key=str)}
        if isinstance(value, (list, tuple)):
            return [self.tree(v) for v in value]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, (set, frozenset)):
            return sorted(self.text(str(v)) for v in value)
        if isinstance(value, pathlib.Path):
            return self.text(str(value))
        return f"<{type(value).__name__}>"


def _norm_repr(value: Any, norm: Normalizer) -> str:
    """Stable repr for handler args: strings/paths normalized, dict keys sorted."""
    if isinstance(value, str):
        return repr(norm.text(value))
    if isinstance(value, pathlib.Path):
        return repr(norm.text(str(value)))
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_norm_repr(v, norm) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{str(k)!r}: {_norm_repr(value[k], norm)}" for k in sorted(value, key=str))
        return "{" + items + "}"
    return f"<{type(value).__name__}>"


def _shape(value: Any) -> str:
    """Type-shape marker for values whose CONTENT is nondeterministic (snapshots)."""
    if value is None:
        return "none"
    return type(value).__name__


# --------------------------------------------------------------------------- #
# Recorder + patching
# --------------------------------------------------------------------------- #
class TraceRecorder:
    def __init__(self, norm: Normalizer, remote_seam: bool = False):
        self.norm = norm
        # The remote seam's functions run on a LOCAL dispatch too (`prepare_operation`
        # answers "local" and `bind_execution_args` binds the shell-guarded tools), so
        # recording them unconditionally would rewrite every existing fixture. The switch
        # is per SCENARIO rather than per call: a placement is a property of the context,
        # so a flag that follows the context cannot make a local trace grow an event.
        self.remote_seam = bool(remote_seam)
        self.events: List[Dict[str, Any]] = []
        self._n = 0

    def record(self, fn: str, **detail: Any) -> None:
        self._n += 1
        event: Dict[str, Any] = {"call": self._n, "fn": fn}
        for key in sorted(detail):
            event[key] = self.norm.tree(detail[key])
        self.events.append(event)


@contextlib.contextmanager
def capture_execute_trace(recorder: TraceRecorder):
    """Wrap the pipeline guard functions with recording; restore on exit.

    Module-level functions are patched on ouroboros.tools.registry (the names
    execute() actually resolves, including the tool_access/runtime_mode_policy
    imports bound into that namespace); methods are patched on ToolRegistry;
    check_safety is patched on ouroboros.safety (execute imports it per call).
    """
    saved: list[tuple[Any, str, Any]] = []

    def patch(obj: Any, attr: str, new: Callable) -> None:
        saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, new)

    R = registry_mod
    norm = recorder.norm
    rec = recorder.record

    # --- module-level guards ------------------------------------------------
    orig_normalize = R._normalize_dispatch_path_args

    def normalize_dispatch(ctx, name, args):
        ret = orig_normalize(ctx, name, args)
        rec("_normalize_dispatch_path_args", tool=name, arg_keys=sorted(args), ret=ret)
        return ret

    patch(R, "_normalize_dispatch_path_args", normalize_dispatch)

    orig_disabled = R._disabled_tools

    def disabled_tools(ctx):
        ret = orig_disabled(ctx)
        rec("_disabled_tools", ret=sorted(ret))
        return ret

    patch(R, "_disabled_tools", disabled_tools)

    orig_availability = R._builtin_tool_availability

    def builtin_availability(name, ctx=None):
        ret = orig_availability(name, ctx)
        rec("_builtin_tool_availability", tool=name, ret=list(ret))
        return ret

    patch(R, "_builtin_tool_availability", builtin_availability)

    orig_resource = R._resource_allowed

    def resource_allowed(ctx, key):
        ret = orig_resource(ctx, key)
        rec("_resource_allowed", key=key, ret=ret)
        return ret

    patch(R, "_resource_allowed", resource_allowed)

    orig_ws_block = R.workspace_mode_block_reason

    def ws_block(ctx):
        ret = orig_ws_block(ctx)
        rec("workspace_mode_block_reason", ret=ret)
        return ret

    patch(R, "workspace_mode_block_reason", ws_block)

    orig_light_redirect = R.light_cognitive_or_root_redirect

    def light_redirect(name, args):
        ret = orig_light_redirect(name, args)
        rec("light_cognitive_or_root_redirect", tool=name, ret=ret)
        return ret

    patch(R, "light_cognitive_or_root_redirect", light_redirect)

    orig_protected = R.protected_paths_in

    def protected_in(paths):
        ret = orig_protected(paths)
        rec(
            "protected_paths_in",
            paths=[norm.text(str(p)) for p in paths],
            matches=[norm.text(str(getattr(m, "path", m))) for m in ret],
        )
        return ret

    patch(R, "protected_paths_in", protected_in)

    orig_light_snap = R._light_repo_snapshot

    def light_snap(repo_dir):
        ret = orig_light_snap(repo_dir)
        rec("_light_repo_snapshot", ret_shape=_shape(ret))
        return ret

    patch(R, "_light_repo_snapshot", light_snap)

    orig_ref_snap = R._git_ref_snapshot

    def ref_snap(repo_dir):
        ret = orig_ref_snap(repo_dir)
        rec("_git_ref_snapshot", ret_shape=_shape(ret))
        return ret

    patch(R, "_git_ref_snapshot", ref_snap)

    orig_compose = R._compose_execute_result

    def compose(result, route_note, safety_msg):
        rec("_compose_execute_result", route_note=route_note, safety_msg=safety_msg or "")
        return orig_compose(result, route_note, safety_msg)

    patch(R, "_compose_execute_result", compose)

    # --- ToolRegistry methods -----------------------------------------------
    orig_ephemeral = ToolRegistry._ephemeral_block

    def ephemeral(self, name, ext_tool=None, is_mcp=False):
        ret = orig_ephemeral(self, name, ext_tool, is_mcp)
        rec("_ephemeral_block", tool=name, is_mcp=bool(is_mcp), ret=ret)
        return ret

    patch(ToolRegistry, "_ephemeral_block", ephemeral)

    orig_gate = ToolRegistry._subagent_and_update_gate

    def gate(self, name, entry, ext_tool, is_mcp, local_readonly, acting, grants):
        ret = orig_gate(self, name, entry, ext_tool, is_mcp, local_readonly, acting, grants)
        rec(
            "_subagent_and_update_gate",
            tool=name, local_readonly=bool(local_readonly), acting=bool(acting), ret=ret,
        )
        return ret

    patch(ToolRegistry, "_subagent_and_update_gate", gate)

    orig_heal = ToolRegistry._heal_mode_block

    def heal(self, name, args, task_constraint, ext_tool, is_mcp):
        ret = orig_heal(self, name, args, task_constraint, ext_tool, is_mcp)
        rec("_heal_mode_block", tool=name, ret=ret)
        return ret

    patch(ToolRegistry, "_heal_mode_block", heal)

    orig_python = ToolRegistry._resolve_python_predispatch

    # `*rest, **kwargs`: this wrapper RECORDS a payload, it does not re-declare the
    # pipeline. Mirroring the seam's exact signature is what made the harness break
    # on every internal-argument change (the operation's prepare fact door, then the
    # resolved binding) while the recorded payload stayed identical — so it forwards
    # opaquely instead. The recorded payload is deliberately unchanged.
    def resolve_python(self, name, args, runtime_mode, effective_constraint, *rest, **kwargs):
        out_args, resolution, block = orig_python(
            self, name, args, runtime_mode, effective_constraint, *rest, **kwargs
        )
        rec(
            "_resolve_python_predispatch",
            tool=name, runtime_mode=runtime_mode, arg_keys=sorted(out_args),
            resolution_error=str(getattr(resolution, "error_reason", "") or ""),
            block=block,
        )
        return out_args, resolution, block

    patch(ToolRegistry, "_resolve_python_predispatch", resolve_python)

    orig_shell_check = ToolRegistry._run_shell_safety_check

    # Forwarded opaquely for the same reason as `resolve_python`: the seam carries
    # both the resolved binding and the operation's target, and neither appears in
    # the recorded payload.
    def shell_check(self, args, runtime_mode, *rest, **kwargs):
        ret = orig_shell_check(self, args, runtime_mode, *rest, **kwargs)
        rec("_run_shell_safety_check", runtime_mode=runtime_mode, arg_keys=sorted(args), ret=ret or "")
        return ret

    patch(ToolRegistry, "_run_shell_safety_check", shell_check)

    orig_owner_snap = ToolRegistry._snapshot_owner_files

    def owner_snap(self, *rest, **kwargs):
        ret = orig_owner_snap(self, *rest, **kwargs)
        rec("_snapshot_owner_files", entries=len(ret))
        return ret

    patch(ToolRegistry, "_snapshot_owner_files", owner_snap)

    orig_worktree_snap = ToolRegistry._worktree_status_snapshot

    def worktree_snap(self):
        ret = orig_worktree_snap(self)
        rec("_worktree_status_snapshot", ret_shape=_shape(ret))
        return ret

    patch(ToolRegistry, "_worktree_status_snapshot", worktree_snap)

    orig_invoke = ToolRegistry._invoke_builtin_handler

    def invoke(self, name, entry, args, *rest, **kwargs):
        rec(
            "_invoke_builtin_handler",
            tool=name,
            arg_keys=sorted(args),
            args={str(k): _norm_repr(args[k], norm) for k in sorted(args, key=str)},
        )
        return orig_invoke(self, name, entry, args, *rest, **kwargs)

    patch(ToolRegistry, "_invoke_builtin_handler", invoke)

    orig_post = ToolRegistry._run_shell_post_checks

    def post_checks(self, result, *, light_repo_before=None, workspace_refs_before=None,
                    tool_name="run_command", **kwargs):
        rec(
            "_run_shell_post_checks",
            tool=tool_name,
            light_snapshot=_shape(light_repo_before),
            refs_snapshot=_shape(workspace_refs_before),
        )
        return orig_post(
            self, result,
            light_repo_before=light_repo_before,
            workspace_refs_before=workspace_refs_before,
            tool_name=tool_name,
            **kwargs,
        )

    patch(ToolRegistry, "_run_shell_post_checks", post_checks)

    # --- the REMOTE seam (§3.1): recorded only for an ssh scenario -----------
    # Order matters here as much as it does locally, and it is a DIFFERENT order: the
    # native branch replaces the handler, so `_invoke_builtin_handler` never appears
    # and `execute_native_operation` does. The two placement-blind refusals are the
    # ones that must appear BEFORE `prepare_operation` — a fixture showing a prepare
    # ahead of them would be the class of bug this seam exists to prevent.
    remote = recorder.remote_seam

    orig_secret = R.subagent_secret_path_refusal

    def secret_refusal(ctx, name, args):
        ret = orig_secret(ctx, name, args)
        if remote:
            rec("subagent_secret_path_refusal", tool=name, ret=ret)
        return ret

    patch(R, "subagent_secret_path_refusal", secret_refusal)

    orig_interpreter = R.script_interpreter_refusal

    def interpreter_refusal(name, args):
        ret = orig_interpreter(name, args)
        if remote:
            rec("script_interpreter_refusal", tool=name, ret=ret)
        return ret

    patch(R, "script_interpreter_refusal", interpreter_refusal)

    orig_prepare = R.prepare_operation

    def prepare_operation(ctx, name, args, **kwargs):
        prepared = orig_prepare(ctx, name, args, **kwargs)
        if remote:
            rec(
                "prepare_operation",
                tool=name,
                placement=str(getattr(prepared, "placement", "")),
                operation=str(getattr(prepared, "operation", "")),
                native_routed=bool(getattr(prepared, "native_routed", False)),
                available=bool(getattr(prepared, "available", True)),
            )
        return prepared

    patch(R, "prepare_operation", prepare_operation)

    orig_bind = R.bind_execution_args

    def bind_execution_args(prepared, ctx, args, **kwargs):
        ret = orig_bind(prepared, ctx, args, **kwargs)
        if remote:
            rec("bind_execution_args", bound=bool(getattr(ret, "bound", False)))
        return ret

    patch(R, "bind_execution_args", bind_execution_args)

    orig_native_execute = R.execute_native_operation

    def native_execute(ctx, prepared, **kwargs):
        if remote:
            rec("execute_native_operation", operation=str(getattr(prepared, "operation", "")))
        return orig_native_execute(ctx, prepared, **kwargs)

    patch(R, "execute_native_operation", native_execute)

    orig_filter_listing = R.filter_native_listing

    def filter_listing(ctx, name, text):
        ret = orig_filter_listing(ctx, name, text)
        if remote:
            rec("filter_native_listing", tool=name, changed=bool(ret != text))
        return ret

    patch(R, "filter_native_listing", filter_listing)

    # --- LLM safety supervisor: deterministic stub, call still recorded ------
    def check_safety_stub(tool_name, arguments, messages=None, ctx=None, python_resolution=None):
        rec("check_safety", tool=str(tool_name), arg_keys=sorted(arguments or {}))
        return True, None

    patch(safety_mod, "check_safety", check_safety_stub)

    try:
        yield recorder
    finally:
        for obj, attr, original in reversed(saved):
            setattr(obj, attr, original)


# --------------------------------------------------------------------------- #
# Scenario runner + fixture serialization
# --------------------------------------------------------------------------- #
def run_calls(
    registry: ToolRegistry, calls: list, norm: Normalizer, remote_seam: bool = False,
) -> List[Dict[str, Any]]:
    """Execute each (tool, args) call under capture; one trace block per call."""
    out: List[Dict[str, Any]] = []
    for tool, args in calls:
        recorder = TraceRecorder(norm, remote_seam=remote_seam)
        with capture_execute_trace(recorder):
            result = registry.execute(tool, dict(args))
        out.append({
            "tool": tool,
            "args": norm.tree(args),
            "events": recorder.events,
            "result": norm.text(str(result)),
        })
    return out


def scenario_document(scenario, base: pathlib.Path) -> Dict[str, Any]:
    """Build the full normalized trace document for one scenario."""
    run = scenario.build(base)
    roots = dict(run.roots)
    roots.setdefault("TMP", str(base))
    norm = Normalizer(roots)
    try:
        calls = run_calls(run.registry, run.calls, norm, remote_seam=run.remote_seam)
    finally:
        # An ssh scenario owns a live broker and a patched module attribute. The
        # teardown runs even when a call raised, or a failing scenario would leave a
        # global service registered and poison every test after it in the lane.
        if run.cleanup is not None:
            run.cleanup()
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "calls": calls,
        "guard_calls_total": sum(len(c["events"]) for c in calls),
    }


def dump_fixture(doc: Dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

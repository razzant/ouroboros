"""Scoped process configuration and facts at the handler→loop boundary.

R5 (node-runtime sprint, stream B): the process-tool handler measures its child
directly (returncode, POSIX signal name, wall-clock duration, and — via the
stream-A resolver attestation — the physically resolved runtime) and publishes
the facts through a THREAD-LOCAL slot; ``loop_tool_execution`` consumes them
for the same tool call and merges them into the typed ``result_meta``.

Thread-local, not ctx-scoped, because the tool executor runs each handler in
its own worker thread — the slot is therefore naturally per-in-flight-call,
and an ABANDONED (outer-timeout) handler thread that finishes late writes only
its own thread's slot and can never contaminate a later call's facts. Selected Settings values stay in the scoped launch environment; diagnostics
are redacted before publication. Runtime provenance is also rendered in the
result. A record with no typed
publication carries no process facts at all — under the typed-result organ
(D02) prose is never harvested into typed fields.

This lives outside ``tools/shell.py`` deliberately: it is a loop↔handler seam
(``tools/verify.py`` consumes the attested runtime through it too; run_script reads the
loop's persisted dispatch-source ref through it), and keeping it here keeps
``shell.py`` under the repository's 1600-line hard size gate.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import pathlib
import shutil
import threading
import time
from typing import Dict

from ouroboros.platform_layer import posix_signal_name


def settings_environment_allowed(ctx) -> bool:
    """The existing service authority to select new Settings references."""
    from ouroboros.config import get_runtime_mode
    from ouroboros.presence_authority import presence_ceiling_from_context
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency
    from ouroboros.tool_access import active_tool_profile, _TOP_LEVEL_PRINCIPAL_PROFILES

    profile = active_tool_profile(ctx)
    cyber_actor = profile == "acting_subagent" and mode_has_unrestricted_agency(get_runtime_mode())
    return (cyber_actor or profile in (_TOP_LEVEL_PRINCIPAL_PROFILES | {"operator_control"})) and presence_ceiling_from_context(ctx) is None


def redact_process_data(value):
    """Mask selected values at result/receipt egress, retaining typed host facts."""
    from ouroboros.secret_masking import redact_known_values

    prepared = getattr(_process_facts_tls, "environment", None)
    secrets = prepared[1] if prepared is not None else ()
    if isinstance(value, dict):
        # Host vocabulary/identity is not a child echo. In particular a one-char
        # selected secret must not rewrite PASS/status or receipt identity hashes.
        typed = {"tool", "status", "code", "contract_kind", "expected_match", "ts", "receipt_ts",
                 "criterion_source", "check_rendering", "root", "source", "kind",
                 "sha256", "content_sha256", "input_sha256", "output_sha256"}
        return {key: item if key in typed else redact_process_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_process_data(item) for item in value]
    return redact_known_values(value, secrets)


@contextlib.contextmanager
def process_environment_scope(ctx, name, arguments):
    """One admitted Settings snapshot shared by resolution and nested launchers."""
    from ouroboros.config import load_settings, runtime_settings
    from ouroboros.workspace_executor import resolve_process_env, validate_process_env
    from ouroboros.tools.tool_result import ToolResult

    prior = getattr(_process_facts_tls, "environment", None)
    old_runtime = getattr(_process_facts_tls, "runtime_provenance", None)
    prepared, error = prior, None
    if prior is None and name in {"run_command", "run_script", "verify_and_record"}:
        try:
            refs = validate_process_env(arguments.get("env_from_settings"))
            if refs and name == "verify_and_record" and arguments.get("contract_kind") not in {"visible_verifier", "explicit_command", "explicit_metric"}:
                raise ValueError("env_from_settings applies only to run-kind verification")
            if refs and not settings_environment_allowed(ctx):
                error = ToolResult(status="blocked", code="ACCESS_BLOCKED",
                    text="⚠️ PROCESS_ENV_REFERENCE_BLOCKED: this task cannot select settings-backed process environment.")
            else:
                prepared = resolve_process_env(None, refs,
                    settings=runtime_settings(settings_reader=load_settings) if refs else None)
        except ValueError as exc:
            error = ToolResult(status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR: {exc}")
    _process_facts_tls.environment = prepared
    if prior is None:
        _process_facts_tls.runtime_provenance = None
    try:
        yield error
    finally:
        _process_facts_tls.environment = prior
        _process_facts_tls.runtime_provenance = old_runtime


def process_environment_tool(handler):
    """Also support direct helpers; mask before the registry persists results."""
    @functools.wraps(handler)
    def invoke(ctx, *args, **kwargs):
        from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, _published_tool_result, _replace_tool_result

        name = {"_run_shell": "run_command", "_run_script": "run_script", "_verify_and_record": "verify_and_record"}[handler.__name__]
        arguments = dict(kwargs)
        if name == "verify_and_record" and args:
            arguments.setdefault("contract_kind", args[0])
        with process_environment_scope(ctx, name, arguments) as error:
            if error is not None:
                return _publish_tool_result(ctx, _replace_tool_result(
                    error, meta_updates={"operation_outcome": "completed_no_effect"}))
            try:
                result = handler(ctx, *args, **kwargs)
            except Exception as exc:
                if getattr(_process_facts_tls, "environment", (None, ()))[1]:
                    raise RuntimeError(redact_process_data(str(exc))) from None
                raise
            published = _published_tool_result(ctx, None)
            text = redact_process_data(result)
            provenance = getattr(_process_facts_tls, "runtime_provenance", None)
            if provenance and name != "run_script":
                text += "\n\nRuntime selection: " + json.dumps(provenance, ensure_ascii=False, sort_keys=True)
            if isinstance(published, ToolResult) and published.text == result:
                text = _publish_tool_result(ctx, _replace_tool_result(published, text=text,
                    meta_updates=redact_process_data(dict(published.meta))))
            return text
    return invoke


def _script_source(ctx):
    """``(meta fact, sentence)`` for where this call's script text survives; never raises.

    The loop persisted the call's full redacted arguments, script included, before
    dispatch. That record is cited, never copied again; its absence or a failed
    lookup changes only this note, never the outcome it is attached to.
    """
    from ouroboros.tool_call_log import current_args_source
    from ouroboros.presence_authority import PRESENCE_OWN_WORK_SCOPE, presence_caller_binding

    try:
        source = current_args_source()
        ref = source.get("args_source_ref")
        if source.get("args_source_status") == "ready" and isinstance(ref, dict) and ref.get("call_id") and ref.get("sha256"):
            task_id = str(getattr(ctx, "task_id", "") or "")
            if not task_id:
                status = "unavailable:task_identity"
                return {"status": status}, f"retention is unavailable or unknown ({status})"
            arguments = {
                "task_id": task_id,
                "call_id": str(ref["call_id"]), "manifest_sha256": str(ref["sha256"]),
            }
            if presence_caller_binding(ctx) is not None:
                arguments["presence_scope"] = PRESENCE_OWN_WORK_SCOPE
            source_read = {"tool": "get_task_result", "arguments": arguments}
            fact = {
                "status": "ready", "call_id": str(ref["call_id"]),
                "sha256": str(ref["sha256"]), "source_read": source_read,
            }
            rendered = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
            return fact, f"its redacted source is get_task_result({rendered})"
        status = str(source.get("args_source_status") or "no dispatch record for this call")
    except Exception as exc:  # observational: the outcome stands
        status = f"unavailable:{type(exc).__name__}"
    if status == "no dispatch record for this call":
        # A direct call outside loop dispatch positively has no retained
        # dispatch record.  Preserve that established compatibility wording;
        # failures while looking up or persisting a record remain unknown.
        return {"status": status}, f"its text was not retained ({status})"
    return {"status": status}, f"retention is unavailable or unknown ({status})"


def _publish_script_source(ctx, before, after, fact):
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, _published_tool_result, _replace_tool_result

    published = _published_tool_result(ctx, None)
    if isinstance(published, ToolResult) and published.text == before:
        return _publish_tool_result(ctx, _replace_tool_result(published, text=after, meta_updates={"script_source": fact}))
    return after


def wrap_script_result(ctx, result, audit_note, script_path):
    """Close a run_script result with where its script text survives the removed temp file."""
    from ouroboros.tools.tool_result import _wrap_run_script_process_result

    fact, held = _script_source(ctx)
    try:
        removed = "temporary file cleanup failed" if pathlib.Path(script_path).exists() else "temporary file removed after the run"
    except OSError:
        removed = "temporary file state unknown"
    wrapped = _wrap_run_script_process_result(ctx, result, audit_note, f"{script_path} ({removed}; {held})")
    return _publish_script_source(ctx, wrapped, wrapped, fact)


def script_refusal_with_source(ctx, refusal):
    """A run_script refused before any process started: no script file or effect is
    claimed, only where the requested script text survives."""
    fact, held = _script_source(ctx)
    return _publish_script_source(ctx, refusal, f"{refusal}\n# script_source: nothing ran; {held}", fact)


def selected_process_environment():
    prepared = getattr(_process_facts_tls, "environment", None)
    return prepared[0] if prepared is not None else {}


def process_path_for_cwd(path, cwd):
    """Resolve PATH search entries against the child cwd, preserving lexical paths."""
    return os.pathsep.join(
        str(pathlib.Path(part) if pathlib.Path(part).is_absolute() else pathlib.Path(cwd) / part)
        for part in path.split(os.pathsep)
    )


def record_runtime_selection(ctx, argv, cwd, environment):
    """Non-executing provenance over the actual launch environment; never rewrite argv."""
    from ouroboros.workspace_executor import executor_ref_from_ctx, map_host_path

    trace = getattr(ctx, "_active_interpreter_resolution", None)
    requested = str(getattr(trace, "requested_interpreter", "") or (argv[0] if argv else ""))
    name = pathlib.Path(requested).name.lower()
    python = getattr(trace, "family", "") == "python" or name.startswith("python")
    wrapper = name in {"env", "sh", "bash", "zsh", "dash", "cmd", "cmd.exe", "powershell", "pwsh"}
    if not python and not wrapper:
        return
    executor = executor_ref_from_ctx(ctx)
    remote = False
    if executor is not None and executor.kind != "local":
        try:
            map_host_path(executor, pathlib.Path(cwd))
            remote = True
        except ValueError:
            pass
    selected = ""
    source = str(getattr(trace, "environment", "") or "explicit")
    unknown = ""
    if wrapper:
        source, unknown = "wrapper", "runtime inside wrapper not inspected"
    elif remote:
        source, unknown = "backend_path", "physical backend executable not reported"
    else:
        spelling = str(argv[0])
        if os.path.dirname(spelling):
            candidate = pathlib.Path(spelling)
            candidate = candidate if candidate.is_absolute() else pathlib.Path(cwd) / candidate
            if candidate.is_file() and os.access(candidate, os.X_OK):
                selected = os.path.abspath(candidate)  # lexical venv path, not realpath
        else:
            path = process_path_for_cwd(environment.get("PATH", os.defpath), cwd)
            selected = shutil.which(spelling, path=path) or ""
            source = "PATH" if not trace or not trace.changed else source
        if not selected:
            unknown = "executable not established on the launch PATH"
    _process_facts_tls.runtime_provenance = redact_process_data({
        "requested": requested, "selected_path": selected or None, "source": source,
        "unknown_reason": unknown, "version": getattr(trace, "runtime_version", "") or None,
    })


def signal_name_for_returncode(returncode) -> str:
    """POSIX signal name for a NEGATIVE subprocess returncode, '' otherwise.

    SSOT for the signal-name derivation: the shell result renderer, the typed
    process facts, and the verify_and_record receipt all read this one helper so
    a killed child is named identically everywhere. Windows residual (disclosed,
    not faked): a killed process there reports a large POSITIVE exit code (e.g.
    0xC0000005), so no signal name can be derived and this returns '' —
    signal-death naming is POSIX-only.
    """
    try:
        rc = int(returncode)
    except (TypeError, ValueError):
        return ""
    if rc >= 0:
        return ""
    return posix_signal_name(abs(rc))


_process_facts_tls = threading.local()


@contextlib.contextmanager
def process_facts_handoff(function):
    """Bridge this invocation's channel to its joined handler executor.

    The admitted environment is a snapshot, not a request to resolve Settings
    again. Only the joined invocation returns its measured facts to the loop.
    """
    fields = ("environment", "runtime_provenance", "facts")
    state = {key: getattr(_process_facts_tls, key, None) for key in fields}
    def invoke(*args, **kwargs):
        prior = {key: getattr(_process_facts_tls, key, None) for key in fields}
        try:
            for key, value in state.items():
                setattr(_process_facts_tls, key, value)
            return function(*args, **kwargs)
        finally:
            state.update({key: getattr(_process_facts_tls, key, None) for key in fields})
            for key, value in prior.items():
                setattr(_process_facts_tls, key, value)
    try:
        yield invoke
    finally:
        for key in ("facts", "runtime_provenance"):
            setattr(_process_facts_tls, key, state[key])

# The complete typed fact family this channel owns. When typed facts exist for
# a call, they are authoritative for EVERY member — including the ABSENCE of a
# member (a typed publication without ``signal`` means the child was not
# signal-killed, however much the child's own stdout may spell ``signal=...``).
# ``timed_out`` / ``killed_by_host`` / ``pre_exec_failure`` are the members that
# exist precisely where an exit code does NOT: a child the host killed on its
# deadline, a child the host killed for any other reason (output cap, abandoned
# validator), and a child that never reached exec at all. They are what makes
# the honest "no exit code" cases legible instead of silent — and they are the
# ONLY signal-death-shaped fact a Windows kill can carry, because
# ``TerminateProcess`` leaves a large POSITIVE exit status and no POSIX signal
# number, so the partition below can name nothing there (declared residual).
PROCESS_FACT_KEYS = (
    "exit_code",
    "signal",
    "duration_ms",
    "resolved_runtime",
    "runtime_provenance",
    "timed_out",
    "killed_by_host",
    "pre_exec_failure",
    "ws_relay_failures",
)


def active_resolved_runtime(ctx) -> str:
    """Stream-A seam (node-runtime sprint): the interpreter resolver (python
    pre-dispatch; node post-gates) sets ``ctx._process_resolved_runtime`` to
    the ABSOLUTE physical
    executable it substituted for this call — set ONLY when execution runs
    something other than the literal recorded argv (an argv rewrite or an
    emergency PATH prepend), scoped to the handler invocation exactly like
    ``ctx._active_python_resolution``. Absent/empty means the argv executed as
    written. Read here so the typed result_meta and the verify_and_record
    receipt disclose the same fact from the same slot."""
    try:
        return str(getattr(ctx, "_process_resolved_runtime", "") or "").strip()
    except Exception:
        return ""


def publish_process_facts(
    *, returncode=None, started_ts: float, resolved_runtime: str = "",
    timed_out: bool = False, killed_by_host: bool = False,
    pre_exec_failure: str = "", ws_relay_failures=None,
) -> Dict[str, object]:
    """Publish this thread's typed process facts for the in-flight process tool.

    ``duration_ms`` is always recorded; ``exit_code``/``signal`` only when the
    child actually returned a code (a timeout or a pre-exec failure has none).
    Returns the facts it published. ``timed_out`` says the host stopped the child on its deadline,
    ``killed_by_host`` that the host killed it at all (a deadline kill sets
    both), and ``pre_exec_failure`` carries the typed cause — the exception
    class the platform raised — of a child that never started. On POSIX a
    host kill is normally ALSO visible as ``signal``; on Windows it is not,
    and ``killed_by_host`` beside whatever ``TerminateProcess`` left in
    ``exit_code`` is the whole honest fact the platform gives.
    """
    facts: Dict[str, object] = {
        "duration_ms": max(0, int((time.monotonic() - float(started_ts)) * 1000)),
    }
    if timed_out:
        facts["timed_out"] = True
    if killed_by_host:
        facts["killed_by_host"] = True
    if pre_exec_failure:
        facts["pre_exec_failure"] = str(pre_exec_failure)[:120]
    if returncode is not None:
        try:
            facts["exit_code"] = int(returncode)
        except (TypeError, ValueError):
            pass
        else:
            name = signal_name_for_returncode(returncode)
            if name:
                facts["signal"] = name
    if resolved_runtime:
        facts["resolved_runtime"] = resolved_runtime
    # Child-reported delivery diagnostics cannot author measured exit/kill facts.
    # Only fixed categories and positive integer counts cross this boundary;
    # message text, URLs, tokens and arbitrary child metadata never do.
    if isinstance(ws_relay_failures, dict):
        failures = {
            key: ws_relay_failures[key]
            for key in ("missing_transport", "rate_limited", "http_client_error",
                        "http_server_error", "http_error", "transport_error")
            if type(ws_relay_failures.get(key)) is int and ws_relay_failures[key] > 0
        }
        if failures:
            facts["ws_relay_failures"] = failures
    if provenance := getattr(_process_facts_tls, "runtime_provenance", None):
        facts["runtime_provenance"] = provenance
    facts = redact_process_data(facts)
    _process_facts_tls.facts = facts
    # Returned so a producer that ALSO discloses these facts elsewhere (the
    # verify_and_record receipt) copies the published ones instead of deriving
    # its own second time — one derivation, two disclosures.
    return dict(facts)


def consume_last_process_facts() -> "Dict[str, object] | None":
    """Read-and-CLEAR the typed process facts published on this thread.

    Called by loop_tool_execution: once defensively before dispatching a
    process tool (drops stale facts) and once after, to merge the fresh facts
    into the call's ``result_meta``. Returns ``None`` when nothing was
    published (e.g. an argument-error path where no process ran)."""
    facts = getattr(_process_facts_tls, "facts", None)
    _process_facts_tls.facts = None
    return facts if isinstance(facts, dict) else None


def describe_returncode(returncode: int, *, cwd=None, binding=None,
                        lived_ms: "int | None" = None, resolved_runtime: str = "") -> str:
    """Render a return code with signal details when applicable.

    On a SIGNAL death the caller may disclose the child's lifetime (a
    millisecond count names the kernel-kill incident class) and the physical
    runtime that actually ran (T11); non-signal renderings are unchanged.
    ``binding`` is any resolved resource binding (root/source/skill_name).
    """
    suffix: list = []
    signal_name = signal_name_for_returncode(returncode)
    if signal_name:
        suffix.append(f"signal={signal_name}")
        if lived_ms is not None:
            suffix.append(f"lived={lived_ms}ms")
        if resolved_runtime:
            suffix.append(f"runtime={resolved_runtime}")
    if cwd is not None:
        suffix.append(f"cwd={pathlib.Path(cwd).resolve(strict=False)}")
    rendered_suffix = f" ({', '.join(suffix)})" if suffix else ""
    target_suffix = ""
    if binding is not None:
        target = [f"root={binding.root}", f"source={binding.source}"]
        if binding.skill_name:
            target.append(f"skill={binding.skill_name}")
        target_suffix = "; " + ", ".join(target)
    return f"exit_code={returncode}{rendered_suffix}{target_suffix}"

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
loop's persisted dispatch-source ref through it). It also holds ``check_output_views``, the masked
bounded views and durable whole-output source of a ``verify_and_record`` check. Keeping it here keeps
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

    secrets = selected_process_secrets()
    if isinstance(value, dict):
        # Host vocabulary/identity is not a child echo. In particular a one-char
        # selected secret must not rewrite PASS/status or receipt identity hashes.
        typed = {"tool", "status", "code", "contract_kind", "expected_match", "ts", "receipt_ts",
                 "criterion_source", "check_rendering", "root", "source", "kind",
                 "sha256", "content_sha256", "input_sha256", "output_sha256",
                 # The retained-output omission codes are host-owned controlled vocabulary and
                 # round-trip unchanged in the durable row and the typed result alike: a selected
                 # value equal to one must not rewrite the code (arbitrary text elsewhere masks).
                 "output_source_unavailable", "output_source_call_unavailable"}
        return {key: item if key in typed else redact_process_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_process_data(item) for item in value]
    return redact_known_values(value, secrets)


def masked_process_text(raw):
    """``raw`` masked once for every durable or shown copy of a process's text.

    The established diagnostic redaction (conventional token shapes, URL credentials, credential keys) reads the
    ORIGINAL text while the selected Settings values' occurrences are kept in view, so a range the first policy masks
    cannot leave the remainder of a selected value behind. With no selected occurrence the established single-source
    call runs, byte for byte. The WHOLE composition is one strict seam — walk construction, the conventional walk and
    the final rendering included: any masking failure raises ``MaskingFailed`` and the caller withholds the output
    (``output_source_unavailable: masking_failed``) instead of showing or storing a lossy mask; host-only typed facts
    pass ``redact_process_data`` untouched."""
    from ouroboros.secret_masking import MaskingFailed, selected_value_walk
    from ouroboros.utils import sanitize_tool_result_for_log

    text = str(raw or "")
    try:
        walk = selected_value_walk(text, selected_process_secrets())
        if walk is None:
            return redact_process_data(sanitize_tool_result_for_log(text, strict=True))
        return walk.finish(sanitize_tool_result_for_log(text, walk=walk, strict=True))
    except MaskingFailed:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed: no partial text, no other leak of the failure
        raise MaskingFailed(f"the composed mask could not complete ({type(exc).__name__})") from exc


WITHHELD_PROCESS_OUTPUT = ("The process text is withheld: its mask could not complete (masking_failed). This note "
                           "carries no text; the check's own exit status, verdict and receipt are unaffected.")

WITHHELD_CHECK_OUTPUT = ("The process text is withheld: its mask could not complete (masking_failed); no output "
                         "source was retained and no reader is offered.")


def masked_process_or_withheld(raw):
    """``masked_process_text(raw)`` for a diagnostic note, or the withheld note when the mask cannot complete.

    A note may lose its text that way; the caller's own verdict, exit status and receipt are not affected, and this
    helper is the diagnostic note's alone — the retained check source is a different fact and is not claimed here."""
    from ouroboros.secret_masking import MaskingFailed

    try:
        return masked_process_text(raw)
    except MaskingFailed:
        return WITHHELD_PROCESS_OUTPUT


_RAW_JSON_LINE_BREAKS = str.maketrans(dict.fromkeys("\x85\u2028\u2029", "\n"))


def receipt_line_safe(text):
    """``text`` with the line breaks ``str.splitlines`` honours but JSON writes raw (U+0085, U+2028, U+2029) folded to
    ``\\n``: a bounded preview stored in a JSONL receipt keeps its row one line for ``read_receipts``."""
    return str(text or "").translate(_RAW_JSON_LINE_BREAKS)


def check_output_views(ctx, raw, *, captured=False, view_cap=4000, summary_cap=20000):
    """``(summary, reply_view, disclosure)`` of a check's PROCESS output (stdout, then ``"\\n" + stderr``).

    Durable receipt evidence is bounded (``summary_cap``) and the reply is bounded separately for transport
    (``view_cap``); the cut is always disclosed, never silent (BIBLE P1). ONE masked text (``masked_process_text``)
    feeds both views and the retained copy, before anything is sliced or stored; the caller's matcher keeps the raw
    text. Views are built from the text ``read_file`` pages (CRLF folded): whole and trimmed when it fits the reply,
    else cut at the real beginning with the whitespace it has. When that body is longer than the reply view, the same
    masked text is retained whole as a ``tool_results`` source of this task in the canonical store
    (``persist_exact_text_source``: verified read-back, no rerun). The reply names the ``read_file`` call for the end,
    by line because ``start_char`` counts inside a line window, and states the retained body's own size and lines;
    the disclosure holds a compact ``output_source`` ref. Only that call (the address) is checked against selected
    values: one they would alter is not offered and nothing is stored for it; prose is masked by the registry wrapper
    as any reply. The call is then consulted once against this task's dispatch (``_offered_read``): offered with its
    Presence-bound arguments and the line its window starts on only if that window delivers the end; one it would
    refuse, retarget or bind short of the end keeps the source and its ref, and the reply says why instead of offering
    it (``output_source_call_unavailable``); the offered arguments are checked against selected values as well. Any other gap is said (``output_source_unavailable``), never an address.
    The summary folds the line breaks JSON writes raw (``receipt_line_safe``); the source keeps them. Notes about
    anything but the process (a byte comparison) are the caller's, beside these views.

    ``captured``: the check never returned (Stop now, timeout). What the host holds is "Captured masked output"; an
    observed exit proves nothing about the rest of what the process wrote, and none held is said so. ``raw=None``:
    the wait returned no stream at all, so whether the check wrote anything is unknown, and said so."""
    from ouroboros.tool_result_delivery import reader_text

    if captured and raw is None:
        return (unknown := "The interrupted wait returned none of the check's output; whether it wrote any is unknown."), unknown, {}
    from ouroboros.secret_masking import MaskingFailed

    try:
        masked = masked_process_text(raw)
    except MaskingFailed:
        return WITHHELD_CHECK_OUTPUT, WITHHELD_CHECK_OUTPUT, {"output_source_unavailable": "masking_failed"}
    if captured and not masked:
        return (none := "No captured output is available."), none, {}
    body = reader_text(masked)  # what read_file pages: the views, the stated size, lines and tail all come from this one text
    text = body.strip() if len(body) <= view_cap else body
    summary, reply = (text if len(text) <= cap else text[:cap] + f"\n…[truncated {len(text) - cap} of {len(text)} chars]"
                      for cap in (summary_cap, view_cap))
    label = "Captured" if captured else "Full"
    if captured:
        summary, reply = (f"Captured masked output:\n{part}" for part in (summary, reply))
    summary = receipt_line_safe(summary)
    if len(body) <= view_cap:
        return summary, reply, {}
    from ouroboros.artifacts import actor_source_reference, persist_exact_text_source, task_artifact_dir_path, task_id_for_artifacts
    from ouroboros.tool_access import canonical_data_root

    gone = (f"No verified address for the {label.lower()} output ({len(body)} chars): this view is cut at {view_cap}, "
            f"the receipt summary keeps up to {summary_cap}.")
    try:
        task_id, source_id = task_id_for_artifacts(ctx), "verify_and_record-output"
        planned = actor_source_reference(category="tool_results", source_id=source_id, data=masked.encode("utf-8"), extension="txt")
        # read_file counts start_char inside its max_lines window, so the tail is addressed by its own line; it points
        # at the real end (trailing whitespace excluded), never before the start.
        rows, begin = body.splitlines(keepends=True), 0
        end = len(body.rstrip()) or len(body)  # the real end: trailing whitespace is not this run's content
        tail = max(0, end - view_cap)
        first = next((n for n, row in enumerate(rows) if (begin := begin + len(row)) > tail), len(rows) - 1)
        reader = {"root": "artifact_store", "path": planned["path"], "start_line": first + 1, "max_lines": len(rows) - first,
                  "start_char": tail - (begin - len(rows[first])), "max_chars": view_cap}
        if redact_process_data(_read_call(reader)) != _read_call(reader):  # the address itself must survive the wrapper's masking
            return summary, reply + "\n" + gone, {"output_source_unavailable": "address_masked"}
        # Canonical store, as the other retained tool sources: custody never promotes a fork's own ``source_handles``,
        # and ``read_file(root='artifact_store')`` reaches this one through ``retained_actor_source_read_target``.
        _, ref, issue = persist_exact_text_source(canonical_data_root(ctx), task_id, source_id=source_id, text=masked)
        if issue or ref.get("path") != planned["path"]:
            return summary, reply + "\n" + gone, {"output_source_unavailable": str((issue or {}).get("status") or "mismatch")}
        held = f"{label} masked output of this same run ({len(body)} chars, {len(rows)} lines, sha256 {planned['sha256']}) is retained"
        retained = {"output_source": {k: ref[k] for k in ("kind", "root", "path", "size", "sha256")}}
    except Exception as exc:  # noqa: BLE001 - the bounded views stand; the gap is said, never an address
        return summary, reply + "\n" + gone, {"output_source_unavailable": type(exc).__name__}
    try:  # the named call is this task's own dispatch: its bindings consulted per cut, never a read
        reader, paging, code, why = _offered_read(ctx, reader, task_artifact_dir_path(canonical_data_root(ctx), task_id, create=False) / ref["path"], rows, end)
    except Exception as exc:  # noqa: BLE001 - admission that cannot be established is not permission
        code, why = "admission_unknown", f"its admission could not be established ({type(exc).__name__})."
    if not code and redact_process_data(call := _read_call(reader)) != call:  # the advertised, bound arguments too
        code, why = "address_masked", "a selected value would alter its bound arguments."
    if code:  # the source and its ref stand: another reader may open it; this task's call would not read its tail
        return summary, reply + (f"\n{held} at {ref['path']!r} (artifact_store, output_source), but this task cannot read "
                                 f"its end through the named call: {why}"), {**retained, "output_source_call_unavailable": code}
    return summary, reply + f"\n{held}; read its end without rerunning: {call}; {paging}", retained


def _read_call(args):
    return f"read_file({', '.join(f'{key}={value!r}' for key, value in args.items())})"


def _offered_read(ctx, reader, target, rows, end):
    """``(args, paging, code, reason)`` for the proposed ``reader``. ``args`` is that proposal as this task's dispatch
    would run it: the registry's Presence argument bindings and capability ceiling (``_presence_bound_args``) applied
    once, nothing refitted. On it then run the resource guard, the REAL ``read_file`` entry's public-argument
    preparation (``tool_resolution._prepare_public_builtin_args`` — no second kwargs enumeration, no registry built
    here), the dispatch's path normalization exactly as ``_execute_admitted_text`` runs it (for a read tool its only
    reachable effect is the in-place root/path rewrite these arguments then carry; the redirect it turns into a
    refusal is a write-tool shape), and only then ``read_file``'s own admission on the FULL prepared arguments as the
    handler calls it (profile, binding, Presence resource grants, runtime-data and protected paths, ``bucket`` and
    ``skill_name`` included). With no real ``read_file`` record registered there is no entry to prepare against and
    nothing establishes the dispatch: the consultation refuses (``admission_unknown``), it never skips onward. The
    same-target check compares the exact RESOLVED physical target after those full root checks: a root label the
    handler admits as an alias of the same file is allowed, a different file is not; the admission's normalized root
    is applied to the advertised arguments, because that is the label the handler reads. Then
    ``_render_line_slice``'s arithmetic over the retained ``rows``, never a read, decides whether that window
    delivers the last real character (``end``). An empty or whitespace-only range past it is no delivery:
    ``presence_window``. An argument refusal is the stable typed omission ``arg_refused`` and keeps the genuine
    retained source, never a fabricated call. ``paging`` names the window's effective start_line; earlier text is
    requested only as far as the bindings permit."""
    from ouroboros.tools import core, tool_resolution
    from ouroboros.tools.core_file_tools import (_coerce_line_window, _coerce_max_chars, _coerce_start_char,
                                                 read_admission)
    from ouroboros.tools.registry_core import _presence_bound_args
    from ouroboros.tools.registry_guards import _capability_resource_guard_result

    def said(refusal):
        return str(getattr(refusal, "text", refusal) or getattr(refusal, "code", "")).replace("⚠️ ", "").strip()

    args, refusal = _presence_bound_args(ctx, "read_file", dict(reader))
    if refusal:
        return reader, "", "presence_refused", said(refusal)
    if (refusal := _capability_resource_guard_result(ctx, "read_file", args)) is not None:
        return args, "", "dispatch_refused", said(refusal)
    entry = next((item for item in core.get_tools() if item.name == "read_file"), None)
    if entry is None:
        # No real tool record: nothing here prepared these arguments and nothing can run them — a refusal, never a
        # bypass of the skipped checks into an advertised call.
        return args, "", "admission_unknown", "read_file is not registered in this process's tool set."
    if arg_error := tool_resolution._prepare_public_builtin_args(entry, args):
        return args, "", "arg_refused", said(arg_error)
    tool_resolution._normalize_dispatch_path_args_result(ctx, "read_file", args)  # rewrites args in place, as dispatch reads them
    root, binding, refusal = read_admission(ctx, args.get("root"), args.get("path"),
                                            bucket=str(args.get("bucket") or ""),
                                            skill_name=str(args.get("skill_name") or ""))
    if refusal is not None:
        return args, "", "read_refused", said(refusal)
    if root != args.get("root"):
        args["root"] = root  # advertise the normalized selector this admission resolved and the handler reads with
    if binding is None or pathlib.Path(binding.target_path).resolve() != pathlib.Path(target).resolve():
        return args, "", "target_changed", f"read_file resolves {getattr(binding, 'root', args.get('root'))}:{args.get('path')} to another file."
    # The span read_file delivers for exactly these arguments, in source characters (``_render_line_slice``).
    start, span = _coerce_line_window(args.get("start_line", 1), args.get("max_lines", 2000))
    start = max(1, min(start, len(rows) + 1))
    stop = min(start + span - 1, len(rows))
    base, size = sum(map(len, rows[:start - 1])), sum(map(len, rows[start - 1:stop]))
    offset, budget = min(_coerce_start_char(args.get("start_char", 0)), size), _coerce_max_chars(args.get("max_chars"))
    delivered, reach = base + offset, base + (size if budget is None else min(size, offset + budget))
    changed = sorted(key for key in set(args) | set(reader) if args.get(key) != reader.get(key))
    if not delivered < end <= reach:
        return args, "", "presence_window", (f"Presence argument bindings set {', '.join(changed)} for read_file; that window "
                                             f"(lines {start}-{stop}, characters {delivered}-{reach}) misses this run's end at {end}.")
    # Equality with the proposal proves no freedom (a binding may pin the proposed value), so the guidance is qualified.
    return args, (f"this window starts at start_line={start}; earlier characters or lines take another start_char or "
                  "start_line only where this task's bindings permit."), "", ""


@contextlib.contextmanager
def process_environment_scope(ctx, name, arguments):
    """One current Settings selection at dispatch, shared by nested launchers."""
    from ouroboros.config import load_settings, selection_settings
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
                    settings=selection_settings(settings_reader=load_settings) if refs else None)
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


def wrap_script_result(ctx, result, audit_note, script_path, note_meta=None):
    """Close a run_script result with where its script text survives the removed temp file."""
    from ouroboros.tools.tool_result import _wrap_run_script_process_result

    fact, held = _script_source(ctx)
    try:
        removed = "temporary file cleanup failed" if pathlib.Path(script_path).exists() else "temporary file removed after the run"
    except OSError:
        removed = "temporary file state unknown"
    wrapped = _wrap_run_script_process_result(ctx, result, audit_note, f"{script_path} ({removed}; {held})", note_meta)
    return _publish_script_source(ctx, wrapped, wrapped, fact)


def script_refusal_with_source(ctx, refusal):
    """A run_script refused before any process started: no script file or effect is
    claimed, only where the requested script text survives."""
    fact, held = _script_source(ctx)
    return _publish_script_source(ctx, refusal, f"{refusal}\n# script_source: nothing ran; {held}", fact)


def selected_process_environment():
    prepared = getattr(_process_facts_tls, "environment", None)
    return prepared[0] if prepared is not None else {}


def selected_process_secrets():
    """The selected Settings values of this scope's launch environment: the one reader of the values both maskers use."""
    prepared = getattr(_process_facts_tls, "environment", None)
    return prepared[1] if prepared is not None else ()


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
    "memory_oom",
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
    pre_exec_failure: str = "", ws_relay_failures=None, memory_oom=None,
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
    # The kernel's own OOM counts for this command's memory leaf (``command_memory.outcome``): fixed
    # keys, integer counts and booleans, None where unread. Separate observations, never a verdict.
    if isinstance(memory_oom, dict):
        observed = {key: memory_oom[key] for key in ("victim_in_leaf", "leaf_local_oom", "commands_local_oom", "unknown")
                    if key in memory_oom and (memory_oom[key] is None or type(memory_oom[key]) in (bool, int))}
        if observed:
            facts["memory_oom"] = observed
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


def _memory_reason(fact: dict) -> str:
    """A human reading of the kernel's OOM counts for the command's memory group. The group's kill and the
    command's own exit status are separate observations: the kill may have taken any process of the group
    (a descendant, or all of them), so it is never attributed to the process whose status is reported."""
    seen = []
    if fact.get("victim_in_leaf"):
        seen.append("the kernel recorded an out-of-memory kill in its memory group (any process of the group)")
    elif fact.get("victim_in_leaf") is False:
        seen.append("no kill in its memory group was recorded")
    if fact.get("commands_local_oom"):
        seen.append(f"{fact['commands_local_oom']} out-of-memory event(s) hit the commands group")
    if fact.get("unknown"):
        seen.append("the kernel's memory events could not all be read")
    return "memory: " + "; ".join(seen or ["no out-of-memory event recorded"])


def describe_returncode(returncode: int, *, cwd=None, binding=None,
                        lived_ms: "int | None" = None, resolved_runtime: str = "", memory_oom=None) -> str:
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
    if isinstance(memory_oom, dict) and (signal_name == "SIGKILL" or memory_oom.get("victim_in_leaf")):
        suffix.append(_memory_reason(memory_oom))  # after the leader's own signal; a group kill may end a plain exit too
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

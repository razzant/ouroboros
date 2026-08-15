"""Dependency-light native workspace primitives shared by Home and execd.

This module deliberately contains no task/model/review authority.  It accepts
already-authorized, canonical arguments and performs only target-native facts
and effects below one workspace root.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ouroboros.export_policy_contract import (
    QUESTION_EXPORT,
    QUESTION_NAMED_SOURCE,
    QUESTION_NONE,
    AliasIndex,
    export_disclosure_block,
    export_policy_hash,
    identity_spelling,
    judged_exclusion,
    normalize_export_policy,
    policy_filtered_note,
    policy_from_facts,
)
from ouroboros.platform_layer import (
    kill_process_group_id,
    kill_process_tree,
    process_group_id,
    process_group_status,
    subprocess_new_group_kwargs,
    terminate_process_group_id,
)
from ouroboros.workspace_diagnostics import (
    ProcessExecutionResult,
    ToolExecutionEnvelope,
    diagnostic_from_exception,
)
from ouroboros.workspace_media_native import extract_video_frames
from ouroboros.workspace_native_contract import (
    MANDATORY_REMOTE_NATIVE_OPERATIONS as MANDATORY_REMOTE_NATIVE_OPERATIONS,
    native_error_result as _error_result,
    admitted_native_operation,
    REMOTE_NATIVE_KERNEL_MODULES as REMOTE_NATIVE_KERNEL_MODULES,
    REMOTE_NATIVE_OPERATION_MODULE as REMOTE_NATIVE_OPERATION_MODULE,
    BoundedProcessStream,
    NativeExecutionControl,
    NativeOperationResult,
    NativePreparedOperation,
    ProcessSpool,
    ProcessSpoolSink,
    SERVICE_NAME_PATTERN,
    autocorrect_grep_backslash_pipe as _maybe_autocorrect_grep_backslash_pipe,
    native_relative_spelling,
    open_process_spool_sinks,
    process_capture_trace,
    process_log_artifact,
    process_quota_notes,
    render_process_result_text as _render_process_result_text,
    seal_process_spool_sinks,
    validate_remote_native_operation_map as validate_remote_native_operation_map,
)
# The confinement kernel: ONE door for reads (`native_target`) and ONE for mutations
# (`native_mutation_target`). Its own module rather than four helpers here, because the
# append escape this branch closed existed precisely because the rule was decided at the
# open site instead of once — see that module's docstring.
from ouroboros.workspace_edit_native import (
    _apply_patch as _native_apply_patch,
    _edit_batch as _native_edit_batch,
)
from ouroboros.workspace_native_paths import (
    atomic_write as _atomic_write,
    native_cwd as _cwd,
    native_mutation_target as _mutation_target,
    native_target as _target,
    open_confined_source as _open_source,
    path_is_relative_to as path_is_relative_to,
    workspace_root_dir as _workspace_root,
)
from ouroboros.workspace_payload_native import (
    attach_remote_verification_facts, collect_declared_outputs,
    execute_inline_script,
    execute_reviewed_payload,
    scratch_fingerprints,
    snapshot_declared_outputs,
    validate_declared_output_context,
    validate_reviewed_payload,
)
from ouroboros.workspace_query_native import (
    classify_workspace_path,
    execute_git_workspace_operation,
    execute_workspace_query_operation,
)
from ouroboros.workspace_snapshot_native import (
    guarded_patch_apply as _guarded_patch_apply,
    snapshot_operation,
)

_SERVICE_LOG_TAIL_MAX = 80_000
@dataclass
class _NativeService:
    name: str
    service_id: str
    proc: subprocess.Popen[bytes] | None
    log_path: pathlib.Path
    cwd: pathlib.Path | None
    command: list[str]
    started_at_ms: int
    pgid: int
    control: NativeExecutionControl | None = None
    released: bool = False
    readiness: dict[str, Any] | None = None
    ready: bool = False
    outputs: tuple[str, ...] = ()
    keep_alive: bool = False
    declared_outputs_before: dict[str, Any] | None = None


_SERVICES_BY_TASK_NAME: dict[tuple[str, str], _NativeService] = {}
_SERVICES_BY_ID: dict[str, _NativeService] = {}
_SERVICES_LOCK = threading.RLock()


_relative_text = native_relative_spelling


def prepare_native_operation(
    workspace_root: pathlib.Path | str,
    tool: str,
    args: Mapping[str, Any],
    *,
    task_id: str = "",
    blobs: Mapping[str, bytes] | None = None,
) -> NativePreparedOperation:
    """Resolve target-native facts without producing an effect.

    The returned ``execution_args`` are the exact values Home must authorize.
    In particular, a Python interpreter is selected here, on the target, before
    Home safety sees argv; no launcher may select a different binary later.
    """

    operation = admitted_native_operation(tool)
    root = _workspace_root(workspace_root)
    execution_args = {
        str(key): value
        for key, value in args.items()
        if not str(key).startswith("_")
    }
    facts: dict[str, Any] = {
        "workspace_root": root.as_posix(),
        "task_id": str(task_id or ""),
    }
    if isinstance((protected_rows := args.get("_protected_paths")), list):
        if len(protected_rows) > 1000:
            raise ValueError("protected path policy exceeds the supported limit")
        facts["protected_paths"] = [
            _relative_text(item)
            for item in protected_rows
            if str(item or "").strip()
        ]
    # Home's export policy arrives as a DOCUMENT and is bound into the operation's
    # facts, so every kernel below applies the exact same object and the manifest it
    # returns carries the hash Home can check it against. Normalization happens HERE,
    # once, and a malformed, unknown-channel or unknown-FIELD document refuses the
    # prepare rather than being silently downgraded to a default policy — a source
    # that cannot tell which rules it was given must not choose its own.
    if (raw_policy := args.get("_export_policy")) is not None:
        facts["export_policy"] = normalize_export_policy(raw_policy)
        facts["export_policy_hash"] = export_policy_hash(facts["export_policy"])
    if operation == "execute_reviewed_payload":
        execution_args, payload_facts = validate_reviewed_payload(
            execution_args,
            dict(blobs or {}),
        )
        payload = execution_args["payload"]
        runtime_name = str(payload.get("runtime") or "python3")
        if execution_args["kind"] == "extension_tool":
            runtime_name = "python3"
        allowed = {"python", "python3", "bash", "sh", "node", "deno", "ruby", "go"}
        if pathlib.PurePath(runtime_name).name not in allowed:
            raise PermissionError("reviewed payload runtime is not allowlisted")
        resolved_runtime = shutil.which(runtime_name)
        if not resolved_runtime:
            raise FileNotFoundError(
                f"reviewed payload runtime unavailable on target: {runtime_name}"
            )
        facts.update(payload_facts)
        facts["resolved_runtime"] = str(pathlib.Path(resolved_runtime).resolve())
        return NativePreparedOperation(execution_args=execution_args, native_facts=facts)

    if operation in {
        "read_file",
        "list_files",
        "write_file",
        "edit_text",
        "search_code",
        "query_code",
        "classify_ambiguous_workspace_path",
        "extract_video_frames",
    }:
        key = "path"
        if operation == "classify_ambiguous_workspace_path":
            raw_path = str(execution_args.get(key) or "")
            if not raw_path.startswith("/"):
                raise ValueError("ambiguous workspace classifier requires an absolute path")
            facts["candidate_path"] = raw_path
        else:
            rel = _relative_text(execution_args.get(key), default=".")
            execution_args[key] = rel
            # The media channel turns a workspace file into exported frame blobs, so
            # the export policy applies to its SOURCE, at prepare — and to the file the
            # resolution really lands on, which is what the door judges. Every other
            # operation here is only having a FACT assembled; its own executing door
            # asks the question that belongs to it.
            media = operation == "extract_video_frames"
            facts["resolved_path"] = _target(
                root,
                rel,
                question=QUESTION_EXPORT if media else QUESTION_NONE,
                facts=facts,
                channel="media_frames" if media else "",
            ).as_posix()
    elif operation in {"run_command", "run_script", "start_service", "verify_remote_check"}:
        cwd = _cwd(root, execution_args.get("cwd"))
        execution_args["cwd"] = cwd.as_posix()
        facts["resolved_cwd"] = cwd.as_posix()
        if operation in {"run_command", "run_script", "start_service"}:
            facts["declared_outputs_before"] = snapshot_declared_outputs(
                root, execution_args, policy=policy_from_facts(facts)
            )
            scratch_fingerprints(root, execution_args)
            if execution_args.get("scratch"):
                probe = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                if probe.returncode or probe.stdout.strip() != b"true":
                    raise PermissionError(
                        "scratch requires a Git-worktree command cwd"
                    )
            for raw in execution_args.get("scratch") or []:
                candidate = (
                    pathlib.Path(str(raw))
                    if pathlib.Path(str(raw)).is_absolute()
                    else cwd / str(raw)
                ).resolve(strict=False)
                # The two halves of the rule that were missing here. Home's
                # `_scratch_safety_reason` confines a declared throwaway to the COMMAND
                # CWD and refuses a directory; this side confined it only to the
                # workspace root and said nothing about directories. Both matter for the
                # same reason the git-tracked check does: scratch is excluded from the
                # workspace patch, so a path outside the cwd (or a whole directory) is a
                # way to keep real work out of the deliverable.
                if not path_is_relative_to(candidate, cwd):
                    raise PermissionError(
                        f"scratch path escapes the command cwd: {raw}"
                    )
                if candidate.is_dir():
                    raise PermissionError(
                        f"scratch path is a directory, not a throwaway file: {raw}"
                    )
                rel = candidate.relative_to(root).as_posix()
                tracked = subprocess.run(
                    ["git", "ls-files", "--error-unmatch", "--", rel],
                    cwd=str(root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                if tracked.returncode == 0:
                    raise PermissionError(
                        f"scratch path is git-tracked, not throwaway: {rel}"
                    )

    if operation in {"run_command", "start_service", "verify_remote_check"}:
        argv = [str(item) for item in execution_args.get("cmd") or []]
        if operation == "run_command":
            argv, autocorrect_note = _maybe_autocorrect_grep_backslash_pipe(argv)
            if autocorrect_note:
                execution_args["cmd"] = argv
                facts["autocorrect_note"] = autocorrect_note
        requested = argv[0] if argv else ""
        if pathlib.PurePath(requested).name in {"python", "python3"}:
            candidates = (
                root / ".venv" / "bin" / "python",
                root / "venv" / "bin" / "python",
            )
            resolved = next((path for path in candidates if path.is_file()), None)
            if resolved is None:
                found = shutil.which(requested)
                if not found:
                    raise FileNotFoundError(f"interpreter unavailable on target: {requested}")
                resolved = pathlib.Path(found)
            argv[0] = str(resolved)
            execution_args["cmd"] = argv
            facts["interpreter"] = str(resolved)
            facts["interpreter_requested"] = requested
    elif operation == "run_script":
        requested = str(execution_args.get("interpreter") or "python3")
        if pathlib.PurePath(requested).name in {"python", "python3"}:
            candidates = (
                root / ".venv" / "bin" / "python",
                root / "venv" / "bin" / "python",
            )
            resolved = next((path for path in candidates if path.is_file()), None)
            if resolved is None:
                found = shutil.which(requested)
                if not found:
                    raise FileNotFoundError(f"interpreter unavailable on target: {requested}")
                resolved = pathlib.Path(found)
            execution_args["interpreter"] = str(resolved)
            facts["interpreter"] = str(resolved)
            facts["interpreter_requested"] = requested

    if operation in {"service_status", "service_logs", "stop_service"}:
        service_ref = args.get("_service_ref")
        if isinstance(service_ref, Mapping):
            service_id = str(service_ref.get("service_id") or "")
            if service_id:
                facts["service_id"] = service_id
                output_context = validate_declared_output_context(root, service_ref)
                facts["service_cwd"] = output_context["cwd"]
                facts["service_outputs"] = output_context["outputs"]
                facts["service_declared_outputs_before"] = output_context["before"]
                facts["service_ready"] = bool(service_ref.get("ready", False))

    return NativePreparedOperation(execution_args=execution_args, native_facts=facts)


def _read_file(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any] | None = None,
) -> ToolExecutionEnvelope:
    rel = _relative_text(args.get("path"))
    # The policy was BOUND here and its echo hash verified, and then nothing applied it:
    # this door shipped whatever byte it was pointed at while `search_code` beside it
    # filtered AND disclosed. One source, so the answer is a refusal; the judgement is the
    # DOOR's, over the resolved file AND its aliases.
    # Through the DESCRIPTOR door: `_target` returns a name, and reading a name a moment
    # after judging it is "checked by name, used by name" — a reviewer swapped the file for
    # a symlink to `.env` inside the applier call and got the secret out of a byte channel.
    path, handle = _open_source(
        root, rel, question=QUESTION_NAMED_SOURCE, facts=native_facts
    )
    # `fdopen` takes ownership, so the fd is closed on any exit from the block.
    with os.fdopen(handle, "r", encoding="utf-8", errors="replace") as stream:
        content = stream.read()
    lines = content.splitlines(keepends=True)
    try:
        start = int(args.get("start_line", 1))
    except (TypeError, ValueError):
        start = 1
    try:
        limit = int(args.get("max_lines", 2000))
    except (TypeError, ValueError):
        limit = 2000
    start = max(1, start)
    limit = max(1, limit)
    start = min(start, len(lines) + 1)
    end = min(len(lines), start + limit - 1)
    body = "".join(lines[start - 1 : end])
    return ToolExecutionEnvelope(
        text=f"# active_workspace:{rel} — lines {start}–{end} of {len(lines)}\n{body}",
        trace={
            "completion": "complete",
            "path": rel,
            # Emitted on EVERY read, exclusions or not, so the presence of the block
            # never encodes whether something was filtered (`export_disclosure_block`).
            # The path is DECLARED as exported, which is what gives Home's returned-
            # manifest check something to re-evaluate: it used to get none from here.
            **export_disclosure_block(
                native_facts, [], [identity_spelling(root, path) or rel],
                question=QUESTION_NAMED_SOURCE,
            ),
        },
    )


def _list_files(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any] | None = None,
) -> ToolExecutionEnvelope:
    rel = _relative_text(args.get("path"))
    path = _target(
        root, rel, question=QUESTION_NAMED_SOURCE, facts=native_facts, must_exist=True
    )
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {rel}")
    limit = max(1, min(10_000, int(args.get("max_entries") or 500)))
    document = policy_from_facts(native_facts)
    aliases = None if document is None else AliasIndex(root, document)
    excluded: list[dict[str, str]] = []
    rows: list[str] = []
    # The NAMES handed over, DECLARED: this door and the query walk were the two
    # disclosure sites passing no `exported` list, so Home's backstop — fixed for the read
    # and declared-output channels — stayed arithmetic over an empty set for exactly these.
    listed: list[str] = []
    truncated = False
    for item in sorted(path.iterdir()):
        if len(rows) >= limit:
            rows.append(f"...(truncated at {limit})")
            truncated = True
            break
        # Containment via `identity_spelling`: it answers "" outside the root and, unlike
        # `Path.resolve`, raises no bare `RuntimeError` on a cycle (which killed a listing).
        if not identity_spelling(root, item):
            continue
        display = item.relative_to(root).as_posix()
        # A TREE channel: an excluded entry is DISCLOSED work, not a refusal (D7) — an
        # entry silently absent makes the model reason from a false premise. Judged on all
        # three spellings, because a listing is how a model learns which names exist; the
        # hardlink half was a stated residual ("a whole-tree scan per `ls`") and is gone.
        reason = judged = ""
        if document is not None:
            reason, _sentence, judged = judged_exclusion(
                root, item, display, document,
                question=QUESTION_NAMED_SOURCE, aliases=aliases,
            )
        if reason:
            # `judged` names the spelling the policy excluded — the entry's own name unless
            # an ALIAS was the finding, which Home cannot re-derive from an innocent one.
            excluded.append({"path": display, "reason": reason, "judged": judged})
            continue
        listed.append(display)
        rows.append(display + ("/" if item.is_dir() else ""))
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if excluded:
        text += "\n\n" + policy_filtered_note("LIST_POLICY_FILTERED", excluded)
    return ToolExecutionEnvelope(
        text=text,
        # A policy exclusion belongs in `completion` for the same reason a truncation
        # does: the answer does not cover the directory it was asked about.
        trace={
            "completion": "complete" if not (truncated or excluded) else "partial",
            "path": rel,
            "truncated": truncated,
            **export_disclosure_block(
                native_facts, excluded, listed, question=QUESTION_NAMED_SOURCE
            ),
        },
    )


def _write_file(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any] | None = None,
) -> ToolExecutionEnvelope:
    rows = args.get("files")
    items: list[dict[str, Any]]
    if isinstance(rows, list) and rows:
        items = [dict(row) for row in rows if isinstance(row, Mapping)]
    else:
        items = [{"path": args.get("path"), "content": args.get("content")}]
    mode = str(args.get("mode") or "overwrite")
    results: list[str] = []
    written_paths: list[str] = []
    # EVERY row is RESOLVED and judged BEFORE the first byte is written: a batch that
    # refuses halfway has already applied the rows before it, and a policy refusal must
    # not leave a partially mutated tree. Home judges the whole batch up front too. The
    # pre-pass RESOLVES rather than judging a spelling, because resolution is where the
    # judgement happens — the old shape judged `innocent.bin` and wrote through it into the
    # protected `golden.bin`.
    for item in items:
        _mutation_target(root, _relative_text(item.get("path")), facts=native_facts)
    for item in items:
        rel = _relative_text(item.get("path"))
        path = _mutation_target(root, rel, facts=native_facts)
        body = str(item.get("content") or "")
        try:
            if mode == "append":
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(body)
            else:
                shrink = _tracked_shrink_block(
                    root, rel, path, body, bool(args.get("force", False))
                )
                if shrink:
                    return ToolExecutionEnvelope(
                        text=shrink,
                        trace={"completion": "complete", "paths": [rel]},
                    )
                _atomic_write(path, body.encode("utf-8"))
        except OSError as exc:
            diagnostic = diagnostic_from_exception(
                exc,
                request_id=str(args.get("_request_id") or ""),
                operation_id=str(args.get("_operation_id") or ""),
                phase="execute",
                details={"operation": "write_file"},
            )
            already = ", ".join(results) if results else "(none)"
            return ToolExecutionEnvelope(
                text=(
                    f"⚠️ FILE_WRITE_ERROR on '{rel}': {diagnostic.message}\n"
                    f"Successfully written before error: {already}"
                ),
                diagnostic=diagnostic,
                trace={"completion": "completed", "paths": written_paths},
            )
        results.append(
            f"active_workspace:{rel} ({len(body)} chars)"
        )
        written_paths.append(rel)
    summary = ", ".join(results)
    return ToolExecutionEnvelope(
        text=(
            f"✅ Written {len(results)} file(s): {summary}\n"
            "Files are on disk in the active workspace. Do not commit; "
            "the headless runner will emit a patch artifact."
        ),
        trace={"completion": "complete", "paths": written_paths},
    )


def _edit_text(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any] | None = None,
) -> ToolExecutionEnvelope:
    rel = _relative_text(args.get("path"))
    # TWO doors, TWO questions, because `edit_text` does two things; the refusals are a
    # UNION. Both used to ask MUTATION ("it reads in order to write") — and MUTATION drops
    # the credential classes by design, so the read inherited the narrowing and
    # `edit_text('.env', old_str='zzz')` answered `File preview (first 2000 chars)` with the
    # whole file `read_file('.env')` was refused (also `id_rsa`, `credentials.json`,
    # `.env.prod`). Reading in order to write is still reading; the stated cost is that an
    # excluded file can be WRITTEN and no longer EDITED. MUTATION goes FIRST so the typed
    # code is the write refusal ("may not write at all" outranks "may not read"), which also
    # puts the policy ahead of the read door's existence check.
    write_path = _mutation_target(root, rel, facts=native_facts)
    read_path = _target(
        root, rel, question=QUESTION_NAMED_SOURCE, facts=native_facts, must_exist=True
    )
    old = str(args.get("old_str") or "")
    new = str(args.get("new_str") or "")
    if not old:
        return ToolExecutionEnvelope(
            text="⚠️ STR_REPLACE_ERROR: old_str is required (cannot be empty).",
            trace={"completion": "complete", "matched": 0, "path": rel},
        )
    content = read_path.read_text(encoding="utf-8")
    count = content.count(old)
    if count == 0:
        return ToolExecutionEnvelope(
            text=(
                f"⚠️ STR_REPLACE_ERROR: old_str not found in {rel}.\n"
                f"File preview (first 2000 chars):\n{content[:2000]}"
            ),
            trace={"completion": "complete", "matched": 0, "path": rel},
        )
    if count != 1:
        positions: list[str] = []
        start = 0
        for _ in range(min(count, 5)):
            index = content.index(old, start)
            positions.append(f"line {content[:index].count(chr(10)) + 1}")
            start = index + 1
        return ToolExecutionEnvelope(
            text=(
                f"⚠️ STR_REPLACE_ERROR: old_str found {count} times in {rel} "
                f"(must be unique). Occurrences at: {', '.join(positions)}. "
                "Include more surrounding context in old_str to make it unique."
            ),
            trace={"completion": "complete", "matched": count, "path": rel},
        )
    updated = content.replace(old, new, 1)
    _atomic_write(write_path, updated.encode("utf-8"))
    replacement_line = updated[:updated.index(new)].count("\n") + 1
    context_start = max(0, replacement_line - 3)
    context_lines = updated.splitlines()[
        context_start:replacement_line + len(new.splitlines()) + 2
    ]
    context_preview = "\n".join(
        f"{context_start + index + 1:>4}| {line}"
        for index, line in enumerate(context_lines)
    )
    return ToolExecutionEnvelope(
        text=(
            f"✅ Replaced in active_workspace:{rel} (line {replacement_line}).\n"
            f"Context:\n{context_preview}\n\n"
            "File is on disk but NOT committed.\n"
            "Do not commit; the headless runner will emit a patch artifact."
        ),
        trace={"completion": "complete", "matched": 1, "path": rel},
    )







def _tracked_shrink_block(
    root: pathlib.Path,
    rel: str,
    target: pathlib.Path,
    new_content: str,
    force: bool,
) -> str:
    if force or not target.exists() or target.is_symlink():
        return ""
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if tracked.returncode:
            return ""
        old_len = len(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, subprocess.SubprocessError):
        return ""
    new_len = len(new_content)
    if old_len <= 0 or new_len >= old_len * 0.7:
        return ""
    pct = round(new_len / old_len * 100)
    return (
        f"⚠️ WRITE_BLOCKED: new content for '{rel}' is {pct}% of original "
        f"({old_len} -> {new_len} chars). This looks like accidental truncation. "
        "Use edit_text for surgical edits, or pass force=true to confirm "
        "intentional rewrite."
    )


def _run_process(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    cmd: list[str],
    control: NativeExecutionControl | None,
    env: Mapping[str, str] | None = None,
    native_facts: Mapping[str, Any] | None = None,
    process_registry: tuple[set[Any], Any] | None = None,
    process_spool: ProcessSpool | None = None,
    backend: str = "ssh_exec",
) -> NativeOperationResult:
    cwd = _cwd(root, args.get("cwd"))
    timeout = max(
        1.0,
        float(args.get("timeout_sec") or args.get("timeout") or 120),
    )
    started = time.monotonic()
    effective_cmd, local_autocorrect_note = (
        _maybe_autocorrect_grep_backslash_pipe([str(part) for part in cmd])
    )
    proc = subprocess.Popen(
        effective_cmd,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env) if env is not None else None,
        **subprocess_new_group_kwargs(),
    )
    if (pgid := process_group_id(proc.pid)) <= 0 and control is not None:
        kill_process_tree(proc)
        raise RuntimeError("could not resolve child process group")
    if control is not None:
        try:
            control.register_process(pgid=pgid)
        except Exception:
            _kill_process_group(proc)
            raise
    if process_registry is not None:
        processes, lock = process_registry
        with lock:
            processes.add(proc)
    stdout_capture = BoundedProcessStream()
    stderr_capture = BoundedProcessStream()

    def _signal_quota_terminate(_reason: str) -> None:
        # Signal only: the finally block below owns the full teardown ladder.
        if pgid > 0:
            terminate_process_group_id(pgid)
        else:
            kill_process_tree(proc)

    spool_sinks = open_process_spool_sinks(
        process_spool or getattr(control, "process_spool", None),
        _signal_quota_terminate,
    )

    def _drain(
        stream: Any,
        target: BoundedProcessStream,
        sink: ProcessSpoolSink | None,
    ) -> None:
        if stream is None:
            return
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            target.append(chunk)
            if sink is not None:
                sink.write(chunk)

    stdout_thread = threading.Thread(
        target=_drain,
        args=(proc.stdout, stdout_capture, spool_sinks.get("stdout")),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain,
        args=(proc.stderr, stderr_capture, spool_sinks.get("stderr")),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        deadline = started + timeout
        while proc.poll() is None:
            if control is not None and control.cancelled():
                _kill_process_group(proc)
                raise InterruptedError("remote operation cancelled")
            if time.monotonic() >= deadline:
                _kill_process_group(proc)
                raise subprocess.TimeoutExpired(cmd, timeout)
            time.sleep(0.05)
    finally:
        # Custody is not released while a foreground process group (or one of
        # its inherited pipe writers) can still be alive.  This keeps timeout,
        # cancellation, and normal-return teardown on the same ownership path.
        if proc.poll() is None:
            _kill_process_group(proc)
        else:
            proc.wait()
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            _kill_process_group(proc)
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
        if pgid > 0:
            _stop_residual_process_group(pgid)
        if control is not None:
            release = getattr(control, "release_process", None)
            if callable(release):
                release(pgid=pgid)
        if process_registry is not None:
            processes, lock = process_registry
            with lock:
                processes.discard(proc)
    stdout = stdout_capture.preview("stdout")
    stderr = stderr_capture.preview("stderr")
    spool_rows, spool_facts = seal_process_spool_sinks(spool_sinks)
    capture_meta = process_capture_trace(
        {"stdout": stdout_capture, "stderr": stderr_capture}, spool_facts
    )
    result = ProcessExecutionResult(
        returncode=int(proc.returncode or 0),
        stdout=stdout,
        stderr=stderr,
        args=effective_cmd,
        backend_trace={
            "backend": backend,
            "cwd": cwd.as_posix(),
            "duration_ms": int((time.monotonic() - started) * 1000),
            "output_capture": capture_meta,
        },
    )
    autocorrect_note = str(
        (native_facts or {}).get("autocorrect_note")
        or local_autocorrect_note
        or ""
    )
    text = _render_process_result_text(
        result,
        cwd=cwd,
        notes=autocorrect_note + process_quota_notes(spool_facts),
    )
    blobs: dict[str, bytes] = {}
    artifacts: list[dict[str, Any]] = []
    for stream_name, capture in (
        ("stdout", stdout_capture),
        ("stderr", stderr_capture),
    ):
        row, inline = process_log_artifact(
            stream_name, capture, spool_rows.get(stream_name)
        )
        if row is None:
            continue
        if inline is not None:
            blobs[str(row["blob_id"])] = inline
        artifacts.append(row)
    scratch = scratch_fingerprints(root, args)
    output_notes: list[str] = []
    output_failed = False
    output_excluded: list[dict[str, str]] = []
    output_exported: list[str] = []
    if result.returncode == 0 and args.get("outputs"):
        (
            output_blobs, output_artifacts, output_notes, output_failed,
            output_excluded, output_exported,
        ) = (
            collect_declared_outputs(
                root,
                args,
                (
                    native_facts.get("declared_outputs_before", {})
                    if isinstance(native_facts, Mapping)
                    else {}
                ),
                policy_from_facts(native_facts),
            )
        )
        blobs.update(output_blobs)
        artifacts.extend(output_artifacts)
        marker = (
            "⚠️ ARTIFACT_OUTPUT_ERROR"
            if output_failed
            else (
                "ARTIFACT_OUTPUTS"
                if output_artifacts
                else "ARTIFACT_OUTPUT_NOTE"
            )
        )
        text += f"\n\n{marker}:\n" + "\n".join(
            f"- {note}" for note in output_notes
        )
    if scratch:
        text += (
            "\n\n⚠️ SCRATCH_REMAINS: declared scratch still on disk after "
            "the command: " + ", ".join(list(scratch)[:5])
            + ". It is excluded from the workspace patch, but delete it before "
            "finishing so it does not linger."
        )
    return NativeOperationResult(
        ToolExecutionEnvelope(
            text=text,
            process=result,
            artifacts=tuple(artifacts),
            trace={
                **result.backend_trace,
                "output_blobs": artifacts,
                "output_capture": capture_meta,
                "scratch_fingerprints": scratch,
                "artifact_output_failed": output_failed,
                # D7 disclosure for the declared-output channel. Additive: the
                # operation's own completion is unchanged, because a policy
                # exclusion is a disclosed omission, not a failed operation. The
                # exported paths ride with it so Home's returned-manifest check has
                # something to re-evaluate rather than an empty list.
                **export_disclosure_block(
                    native_facts, output_excluded, output_exported
                ),
            },
        ),
        blobs,
    )


def _kill_process_group(proc: subprocess.Popen[Any]) -> None:
    pgid = process_group_id(proc.pid)
    if pgid <= 0:
        kill_process_tree(proc)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        return
    terminate_process_group_id(pgid)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        kill_process_group_id(pgid)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    if proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def _stop_residual_process_group(pgid: int, *, grace_sec: float = 2.0) -> None:
    """Ensure no descendant remains in a foreground command's custody group."""

    if pgid <= 0 or process_group_status(pgid) == "gone":
        return
    terminate_process_group_id(pgid)
    deadline = time.monotonic() + max(0.0, grace_sec)
    while time.monotonic() < deadline:
        if process_group_status(pgid) == "gone":
            return
        time.sleep(0.05)
    kill_process_group_id(pgid)
    deadline = time.monotonic() + max(0.0, grace_sec)
    while time.monotonic() < deadline:
        if process_group_status(pgid) == "gone":
            return
        time.sleep(0.05)


def _start_service(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    control: NativeExecutionControl | None,
    task_id: str,
    native_facts: Mapping[str, Any],
) -> ToolExecutionEnvelope:
    name = str(args.get("name") or "service").strip() or "service"
    # REFUSE what the local handler refuses (`tools/services._sanitize_service_name`),
    # instead of silently rewriting it into a filename. Sanitizing is not the gentler
    # option here: `a/b` and `a_b` both became `a_b.log`, so two services shared one log
    # and `service_logs` could hand back the other one's output — a rename the caller
    # never asked for and cannot see. Same regex, same bound, one rule on both routes.
    if not SERVICE_NAME_PATTERN.fullmatch(name):
        raise ValueError("name must match [A-Za-z0-9_.-]{1,80}")
    cmd = [str(part) for part in args.get("cmd") or []]
    if not cmd:
        raise ValueError("cmd is required")
    cwd = _cwd(root, args.get("cwd"))
    readiness = (
        dict(args.get("readiness") or {})
        if isinstance(args.get("readiness"), Mapping)
        else {}
    )
    try:
        raw_timeout = float(readiness.get("timeout_sec", 5))
    except (TypeError, ValueError) as exc:
        raise ValueError("readiness.timeout_sec must be numeric") from exc
    # A NEGATIVE timeout is refused, not clamped to zero — the local route refuses it
    # (`services._readiness_timeout`), and clamping answers a nonsensical request with a
    # readiness check that never waits, which looks exactly like a service that came up
    # instantly. The 25 s ceiling is a real bound and stays a clamp.
    if raw_timeout < 0:
        raise ValueError("readiness.timeout_sec must be non-negative")
    readiness_timeout = min(25.0, raw_timeout)
    contains = str(
        readiness.get("stdout_contains")
        or readiness.get("log_contains")
        or ""
    )
    # Through the confinement door, not `root / ...`: `.ouroboros/services` is inside the
    # workspace, so a previous `run_command` can replace any component of it with a
    # symlink, and `log_path.open("ab")` below follows one exactly as `write_file`'s
    # append did. The directory is confined BEFORE it is created, or `mkdir(parents=True)`
    # would materialize `services/` inside whatever `.ouroboros` points at.
    log_dir = _mutation_target(root, ".ouroboros/services", facts=native_facts)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = _mutation_target(
        root, f".ouroboros/services/{name}.log", facts=native_facts
    )
    with _SERVICES_LOCK:
        service_key = (str(task_id or ""), name)
        existing = _SERVICES_BY_TASK_NAME.get(service_key)
        if existing is not None:
            existing_rc = (
                existing.proc.poll()
                if existing.proc is not None
                else (
                    0
                    if process_group_status(existing.pgid) == "gone"
                    else None
                )
            )
            if existing_rc is None:
                raise RuntimeError(f"service already running: {name}")
        log_handle = log_path.open("ab")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                **subprocess_new_group_kwargs(),
            )
        finally:
            log_handle.close()
        service_id = uuid.uuid4().hex
        if (pgid := process_group_id(proc.pid)) <= 0 and control is not None:
            kill_process_tree(proc)
            raise RuntimeError("could not resolve service process group")
        if control is not None:
            try:
                control.register_process(
                    pgid=pgid,
                    keep_alive=bool(args.get("keep_alive", False)),
                    service_id=service_id,
                )
            except Exception:
                _kill_process_group(proc)
                raise
        record = _NativeService(
            name=name,
            service_id=service_id,
            proc=proc,
            log_path=log_path,
            cwd=cwd,
            command=cmd,
            started_at_ms=int(time.time() * 1000),
            pgid=pgid if pgid > 0 else proc.pid,
            control=control,
            readiness=readiness,
            outputs=tuple(str(item) for item in args.get("outputs") or []),
            keep_alive=bool(args.get("keep_alive", False)),
            declared_outputs_before=dict(
                native_facts.get("declared_outputs_before") or {}
            ),
        )
        _SERVICES_BY_TASK_NAME[service_key] = record
        _SERVICES_BY_ID[service_id] = record
    deadline = time.monotonic() + readiness_timeout
    while time.monotonic() <= deadline:
        if not contains:
            record.ready = True
            break
        try:
            record.ready = contains in record.log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )[-20_000:]
        except OSError:
            record.ready = False
        if record.ready or proc.poll() is not None:
            break
        time.sleep(0.2)
    payload = {
        "service_id": service_id,
        "name": name,
        "state": "running" if proc.poll() is None else "exited",
        "ready": record.ready,
        "returncode": proc.poll(),
        "cwd": cwd.as_posix(),
        "outputs": list(record.outputs),
        "keep_alive": record.keep_alive,
        "note": "started",
    }
    return ToolExecutionEnvelope(
        text=json.dumps(payload, sort_keys=True),
        trace={
            "completion": "complete",
            "service_ref": {
                "kind": "ssh_exec",
                "service_id": service_id,
                "name": name,
                "ready": record.ready,
                "outputs": list(record.outputs),
                "keep_alive": record.keep_alive,
                "cwd": cwd.as_posix(),
                "declared_outputs_before": dict(
                    record.declared_outputs_before or {}
                ),
            },
        },
    )


def _service_record(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any],
    task_id: str,
    control: NativeExecutionControl | None,
) -> _NativeService | None:
    name = str(args.get("name") or "service")
    with _SERVICES_LOCK:
        service_id = str(native_facts.get("service_id") or "")
        if service_id:
            record = _SERVICES_BY_ID.get(service_id)
            if record is not None:
                if record.name != name:
                    return None
                task_record = _SERVICES_BY_TASK_NAME.get((str(task_id or ""), name))
                return record if task_record is record else None
            recover = (
                getattr(control, "recover_service", None)
                if control is not None
                else None
            )
            if not callable(recover):
                return None
            recovered = recover(service_id=service_id, name=name)
            if not isinstance(recovered, Mapping):
                return None
            if (
                str(recovered.get("service_id") or "") != service_id
                or str(recovered.get("task_id") or "") != str(task_id or "")
            ):
                return None
            try:
                pgid = int(recovered.get("pgid"))
            except (TypeError, ValueError):
                return None
            service_cwd = _cwd(
                root,
                str(native_facts.get("service_cwd") or root),
            )
            # Same rule as `_start_service`, so a RECOVERED record cannot name a log file
            # the starter would have refused, and the path goes through the confinement
            # door instead of being assembled from `root /` — a recovery is exactly when
            # nobody has re-checked what `.ouroboros/services` currently points at.
            if not SERVICE_NAME_PATTERN.fullmatch(name):
                return None
            record = _NativeService(
                name=name,
                service_id=service_id,
                proc=None,
                log_path=_mutation_target(
                    root, f".ouroboros/services/{name}.log", facts=native_facts
                ),
                cwd=service_cwd,
                command=[],
                started_at_ms=int(
                    recovered.get("started_at_ms")
                    or recovered.get("registered_at_ms")
                    or 0
                ),
                pgid=pgid,
                control=control,
                ready=bool(native_facts.get("service_ready", False)),
                outputs=tuple(
                    str(item)
                    for item in native_facts.get("service_outputs") or []
                ),
                keep_alive=bool(recovered.get("keep_alive", False)),
                declared_outputs_before=dict(
                    native_facts.get("service_declared_outputs_before") or {}
                ),
            )
            _SERVICES_BY_ID[service_id] = record
            _SERVICES_BY_TASK_NAME[(str(task_id or ""), name)] = record
            return record
        return _SERVICES_BY_TASK_NAME.get((str(task_id or ""), name))


def _service_status(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any],
    task_id: str,
    control: NativeExecutionControl | None,
) -> ToolExecutionEnvelope:
    record = _service_record(
        root,
        args,
        native_facts=native_facts,
        task_id=task_id,
        control=control,
    )
    if record is None:
        return ToolExecutionEnvelope(
            text="⚠️ SERVICE_NOT_FOUND",
            trace={"completion": "complete", "running": False},
        )
    rc = (
        record.proc.poll()
        if record.proc is not None
        else (0 if process_group_status(record.pgid) == "gone" else None)
    )
    if rc is not None:
        _release_service_process(record)
    payload = {
        "name": record.name,
        "service_ref": {
            "kind": "ssh_exec",
            "service_id": record.service_id,
            "name": record.name,
            "ready": bool(record.ready),
            "outputs": list(record.outputs),
            "keep_alive": bool(record.keep_alive),
            "cwd": record.cwd.as_posix() if record.cwd is not None else "",
            "declared_outputs_before": dict(
                record.declared_outputs_before or {}
            ),
        },
        "running": rc is None,
        "returncode": rc,
        "ready": bool(record.ready),
        "outputs": list(record.outputs),
        "keep_alive": bool(record.keep_alive),
        "cwd": record.cwd.as_posix() if record.cwd is not None else "",
        "started_at_ms": record.started_at_ms,
    }
    return ToolExecutionEnvelope(
        text=json.dumps(payload, sort_keys=True),
        trace={"completion": "complete", **payload},
    )


def _service_logs(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any],
    task_id: str,
    control: NativeExecutionControl | None,
) -> ToolExecutionEnvelope:
    record = _service_record(
        root,
        args,
        native_facts=native_facts,
        task_id=task_id,
        control=control,
    )
    if record is None:
        return ToolExecutionEnvelope(
            text="⚠️ SERVICE_NOT_FOUND",
            trace={"completion": "complete"},
        )
    tail = max(1, min(_SERVICE_LOG_TAIL_MAX, int(args.get("tail") or 8000)))
    data = record.log_path.read_bytes()[-tail:]
    return ToolExecutionEnvelope(
        text=data.decode("utf-8", errors="replace"),
        trace={
            "completion": "complete",
            "service_ref": {
                "kind": "ssh_exec",
                "service_id": record.service_id,
                "name": record.name,
                "ready": bool(record.ready),
                "outputs": list(record.outputs),
                "keep_alive": bool(record.keep_alive),
                "cwd": record.cwd.as_posix() if record.cwd is not None else "",
                "declared_outputs_before": dict(
                    record.declared_outputs_before or {}
                ),
            },
        },
    )


def _stop_service(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any],
    task_id: str,
    control: NativeExecutionControl | None,
) -> NativeOperationResult:
    record = _service_record(
        root,
        args,
        native_facts=native_facts,
        task_id=task_id,
        control=control,
    )
    if record is None:
        return NativeOperationResult(
            ToolExecutionEnvelope(
                text="⚠️ SERVICE_NOT_FOUND",
                trace={"completion": "complete"},
            )
        )
    if record.proc is not None:
        if record.proc.poll() is None:
            _kill_process_group(record.proc)
    else:
        stop = (
            getattr(record.control, "stop_service", None)
            if record.control is not None
            else None
        )
        if not callable(stop) or not stop(service_id=record.service_id):
            raise RuntimeError(
                "custody could not verify and stop the recovered service"
            )
    _release_service_process(record)
    with _SERVICES_LOCK:
        _SERVICES_BY_TASK_NAME.pop((str(task_id or ""), record.name), None)
        _SERVICES_BY_ID.pop(record.service_id, None)
    blobs: dict[str, bytes] = {}
    artifacts: list[dict[str, Any]] = []
    notes: list[str] = []
    output_failed = False
    output_excluded: list[dict[str, str]] = []
    output_exported: list[str] = []
    if record.outputs and record.cwd is not None:
        (
            blobs, artifacts, notes, output_failed, output_excluded, output_exported,
        ) = collect_declared_outputs(
            root,
            {"cwd": record.cwd.as_posix(), "outputs": list(record.outputs)},
            record.declared_outputs_before or {},
            policy_from_facts(native_facts),
        )
    text = f"OK: service '{record.name}' stopped."
    if notes:
        marker = (
            "⚠️ ARTIFACT_OUTPUT_ERROR"
            if output_failed
            else ("ARTIFACT_OUTPUTS" if artifacts else "ARTIFACT_OUTPUT_NOTE")
        )
        text += f"\n\n{marker}:\n" + "\n".join(f"- {note}" for note in notes)
    return NativeOperationResult(
        ToolExecutionEnvelope(
            text=text,
            artifacts=tuple(artifacts),
            trace={
                "completion": "complete",
                "artifact_output_failed": output_failed,
                **export_disclosure_block(
                    native_facts, output_excluded, output_exported
                ),
                "service_ref": {
                    "kind": "ssh_exec",
                    "service_id": record.service_id,
                    "name": record.name,
                },
            },
        ),
        blobs,
    )


def _release_service_process(record: _NativeService) -> None:
    if record.released:
        return
    record.released = True
    release = (
        getattr(record.control, "release_process", None)
        if record.control is not None
        else None
    )
    if callable(release):
        release(pgid=record.pgid, service_id=record.service_id)


def execute_native_operation(
    workspace_root: pathlib.Path | str,
    tool: str,
    canonical_args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any] | None = None,
    blobs: Mapping[str, bytes] | None = None,
    task_id: str = "",
    control: NativeExecutionControl | None = None,
) -> NativeOperationResult:
    """Execute one authorized native operation and return typed evidence."""

    # The SAME gate prepare uses: two copies of the allowlist check were two answers
    # waiting to disagree. Raising before the `try` is deliberate — the broad handler
    # below turns exceptions into a successful envelope carrying a diagnostic, which
    # for a contract refusal would report drift as a completed operation.
    operation = admitted_native_operation(tool)
    args = {
        str(key): value
        for key, value in canonical_args.items()
        if not str(key).startswith("_")
    }
    native_facts = dict(native_facts or {})
    supplied_blobs = dict(blobs or {})
    root: pathlib.Path
    try:
        root = _workspace_root(workspace_root)
        if native_facts:
            attested_root = str(native_facts.get("workspace_root") or "")
            if attested_root and attested_root != root.as_posix():
                raise PermissionError("prepared native workspace root changed")
        if operation == "read_file":
            return NativeOperationResult(_read_file(root, args, native_facts=native_facts))
        if operation == "list_files":
            return NativeOperationResult(_list_files(root, args, native_facts=native_facts))
        if operation == "write_file":
            return NativeOperationResult(_write_file(root, args, native_facts=native_facts))
        if operation == "edit_text":
            return NativeOperationResult(_edit_text(root, args, native_facts=native_facts))
        if operation == "apply_patch":
            return NativeOperationResult(_native_apply_patch(root, args, native_facts=native_facts))
        if operation == "edit_batch":
            return NativeOperationResult(_native_edit_batch(root, args, native_facts=native_facts))
        if operation in {"search_code", "query_code"}:
            return NativeOperationResult(execute_workspace_query_operation(root, operation, args, native_facts))
        if operation == "run_command":
            cmd = [str(part) for part in args.get("cmd") or []]
            if not cmd:
                raise ValueError("cmd is required")
            return _run_process(
                root,
                args,
                cmd=cmd,
                control=control,
                native_facts=native_facts,
            )
        if operation == "run_script":
            return execute_inline_script(
                root,
                args,
                control=control,
                native_facts=native_facts,
                process_runner=_run_process,
                atomic_write=_atomic_write,
            )
        if operation == "execute_reviewed_payload":
            return execute_reviewed_payload(
                root,
                args,
                native_facts=native_facts,
                blobs=supplied_blobs,
                control=control,
                process_runner=_run_process,
            )
        if operation in {"vcs_status", "vcs_diff"}:
            return execute_git_workspace_operation(
                root,
                operation,
                args,
                native_facts,
            )
        if operation == "snapshot_manifest_and_blob_export":
            return snapshot_operation(
                root,
                protected_paths=tuple(native_facts.get("protected_paths") or ()),
                policy=policy_from_facts(native_facts),
            )
        if operation == "guarded_patch_apply":
            return NativeOperationResult(
                _guarded_patch_apply(
                    root,
                    args,
                    supplied_blobs,
                    protected_paths=tuple(
                        native_facts.get("protected_paths") or ()
                    ),
                    policy=policy_from_facts(native_facts),
                )
            )
        if operation == "classify_ambiguous_workspace_path":
            return NativeOperationResult(classify_workspace_path(root, args))
        if operation == "start_service":
            return NativeOperationResult(
                _start_service(
                    root,
                    args,
                    control=control,
                    task_id=str(task_id or ""),
                    native_facts=native_facts,
                )
            )
        if operation == "service_status":
            return NativeOperationResult(
                _service_status(
                    root,
                    args,
                    native_facts=native_facts,
                    task_id=str(task_id or ""),
                    control=control,
                )
            )
        if operation == "service_logs":
            return NativeOperationResult(
                _service_logs(
                    root,
                    args,
                    native_facts=native_facts,
                    task_id=str(task_id or ""),
                    control=control,
                )
            )
        if operation == "stop_service":
            return _stop_service(
                root,
                args,
                native_facts=native_facts,
                task_id=str(task_id or ""),
                control=control,
            )
        if operation == "verify_remote_check":
            cmd = [str(part) for part in args.get("cmd") or []]
            if not cmd:
                raise ValueError("cmd is required")
            return attach_remote_verification_facts(
                root,
                args,
                _run_process(
                    root, args, cmd=cmd, control=control, native_facts=native_facts
                ),
                native_facts=native_facts,
            )
        if operation == "extract_video_frames":
            return extract_video_frames(
                root,
                # Asked again here, not deferred to prepare: "the caller upstream
                # already asked it" is the reasoning that produced a leak once already.
                _target(
                    root,
                    args.get("path"),
                    question=QUESTION_EXPORT,
                    facts=native_facts,
                    channel="media_frames",
                    must_exist=True,
                ),
                args,
                control=control,
                kill_process_group=_kill_process_group,
                native_facts=native_facts,
            )
        # NOT contract drift — the one gate above already refused anything off the
        # allowlist, so reaching here means an operation this build DECLARES has no
        # branch: a wiring bug in this module, where a bare exception is the honest
        # answer. It stays so the dispatch cannot fall through and return None.
        raise ValueError(f"admitted native operation has no dispatch branch: {operation}")
    except subprocess.TimeoutExpired as exc:
        return _error_result(
            exc,
            operation=operation,
            args=args,
            domain="process",
            completion="unknown",
        )
    except InterruptedError as exc:
        return _error_result(
            exc,
            operation=operation,
            args=args,
            domain="process",
            completion="unknown",
        )
    except BaseException as exc:
        return _error_result(exc, operation=operation, args=args)

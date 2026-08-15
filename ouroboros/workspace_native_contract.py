"""Dependency-light public contract for the execd native workspace kernel."""

from __future__ import annotations

import hashlib
import pathlib
import re
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from ouroboros.remote_contracts import refuse_unknown_members
from ouroboros.workspace_diagnostics import (
    ProcessExecutionResult,
    ToolExecutionEnvelope,
    diagnostic_from_exception,
    render_diagnostic_text,
    sanitize_execution_text,
)

PROCESS_PREVIEW_HEAD_BYTES = 32_000
PROCESS_PREVIEW_TAIL_BYTES = 32_000
PROCESS_FULL_CAPTURE_BYTES = 16_000_000
REVIEWED_PAYLOAD_FILE_CAP = 512
REVIEWED_PAYLOAD_FILE_BYTES = 8 * 1024 * 1024
REVIEWED_PAYLOAD_TOTAL_BYTES = 32 * 1024 * 1024
DECLARED_OUTPUT_FILE_CAP = 1000
DECLARED_OUTPUT_TOTAL_BYTES = 32 * 1024 * 1024
_GREP_TOOLS = frozenset(("grep", "egrep", "fgrep"))
_GREP_REGEX_MODE_FLAGS = frozenset((
    "-E", "--extended-regexp",
    "-P", "--perl-regexp",
    "-F", "--fixed-strings",
    "-G", "--basic-regexp",
))
_GREP_BACKSLASH_PIPE_PATTERN = re.compile(r'\\\|')
_NO_MATCH_EXIT_TOOLS = frozenset(("grep", "egrep", "fgrep", "rg", "ag", "ack"))


def describe_process_returncode(
    returncode: int,
    *,
    cwd: pathlib.Path | str | None = None,
) -> str:
    """Render a return code identically on Home and target."""

    suffix: list[str] = []
    if int(returncode) < 0:
        signal_num = abs(int(returncode))
        try:
            signal_name = signal.Signals(signal_num).name
        except ValueError:
            signal_name = f"SIG{signal_num}"
        suffix.append(f"signal={signal_name}")
    if cwd is not None:
        suffix.append(f"cwd={pathlib.Path(cwd).resolve(strict=False)}")
    rendered_suffix = f" ({', '.join(suffix)})" if suffix else ""
    return f"exit_code={returncode}{rendered_suffix}"


def format_process_output(
    stdout: str,
    stderr: str,
    *,
    limit: int = 50_000,
) -> str:
    """Render bounded stdout/stderr sections identically on Home and target."""

    parts: list[str] = []
    if str(stdout or "").strip():
        parts.append(f"STDOUT:\n{stdout}")
    if str(stderr or "").strip():
        parts.append(f"STDERR:\n{stderr}")
    rendered = "\n\n".join(parts) if parts else "STDOUT:\n(empty)"
    if len(rendered) > limit:
        rendered = (
            rendered[: limit // 2]
            + "\n...(truncated)...\n"
            + rendered[-limit // 2 :]
        )
    return rendered


def process_is_search_no_match(res: ProcessExecutionResult) -> bool:
    tool = pathlib.Path(str(res.args[0] if res.args else "")).name.lower()
    return (
        int(res.returncode) == 1
        and tool in _NO_MATCH_EXIT_TOOLS
        and not str(res.stderr or "").strip()
    )


def autocorrect_grep_backslash_pipe(
    cmd: list[str],
) -> tuple[list[str], str]:
    if not cmd or pathlib.Path(cmd[0]).name.lower() not in _GREP_TOOLS:
        return cmd, ""
    tool = pathlib.Path(cmd[0]).name.lower()
    explicit = tool in ("egrep", "fgrep")
    if not explicit:
        for arg in cmd[1:]:
            if arg in _GREP_REGEX_MODE_FLAGS:
                explicit = True
                break
            if (
                arg.startswith("-")
                and not arg.startswith("--")
                and any(flag in arg[1:] for flag in ("E", "P", "F", "G"))
            ):
                explicit = True
                break
    if explicit:
        return cmd, ""
    corrected = list(cmd)
    changed_args: list[str] = []
    for idx, arg in enumerate(corrected[1:], start=1):
        if _GREP_BACKSLASH_PIPE_PATTERN.search(arg):
            corrected[idx] = _GREP_BACKSLASH_PIPE_PATTERN.sub("|", arg)
            changed_args.append(arg)
    if not changed_args:
        return cmd, ""
    corrected.insert(1, "-E")
    return corrected, (
        "⚠️ SHELL_REGEX_AUTO_CORRECTED: converted grep backslash-escaped "
        "alternation (\\|) to extended regex mode (`grep -E`) and rewrote "
        f"{changed_args!r} to use `|`.\n"
    )

# The service-name rule, in the CONTRACT because both routes have to refuse the same
# spellings and the target is the side that turns a name into a filename. `tools/services`
# imports this instead of keeping its own copy: the copy is how the native route ended up
# SANITIZING (`re.sub`) what the local route REFUSED, which silently merged two services
# into one log file.
SERVICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")

MANDATORY_REMOTE_NATIVE_OPERATIONS: frozenset[str] = frozenset({
    "apply_patch",
    "classify_ambiguous_workspace_path",
    "edit_batch",
    "edit_text",
    "execute_reviewed_payload",
    "extract_video_frames",
    "guarded_patch_apply",
    "list_files",
    "query_code",
    "read_file",
    "run_command",
    "run_script",
    "search_code",
    "service_logs",
    "service_status",
    "snapshot_manifest_and_blob_export",
    "start_service",
    "stop_service",
    "vcs_diff",
    "vcs_status",
    "verify_remote_check",
    "write_file",
})

REMOTE_NATIVE_OPERATION_MODULE: dict[str, str] = {
    name: "ouroboros.workspace_native"
    for name in sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS)
}


def admitted_native_operation(tool: Any) -> str:
    """The ONE gate on the native-operation SET, shared by prepare and execute.

    There were two copies — `prepare_native_operation` and
    `execute_native_operation` each spelled `if operation not in
    MANDATORY_REMOTE_NATIVE_OPERATIONS: raise ValueError(...)` — which is the same
    "one rule × N doors" shape the export document exists to end, on the table right
    next to it. It lives HERE, beside the allowlist it reads, and the refusal is the
    typed contract-drift one (`remote_contracts`): an operation Home routed and this
    build's allowlist does not carry is a build pair that disagrees, not a caller
    mistake, and the owner's action is the same as for any other contract.
    """

    operation = str(tool or "").strip()
    if operation not in MANDATORY_REMOTE_NATIVE_OPERATIONS:
        refuse_unknown_members(
            "native_operations",
            unknown=[operation],
            understood=MANDATORY_REMOTE_NATIVE_OPERATIONS,
            member="operations",
        )
    return operation

REMOTE_NATIVE_KERNEL_MODULES: frozenset[str] = frozenset({
    *REMOTE_NATIVE_OPERATION_MODULE.values(),
    "ouroboros.code_intelligence",
    # The export-policy DOCUMENT and its one evaluator travel with the kernel,
    # because the source applies the policy mechanically. The Home AUTHORITY that
    # DECIDES what a document says (`ouroboros.remote_export_policy`) does not, and
    # is on the forbidden-import list precisely so it never can.
    "ouroboros.export_policy_contract",
    "ouroboros.execd_task_files",
    "ouroboros.shell_parse",
    "ouroboros.utils",
    "ouroboros.workspace_media_native",
    # The confinement kernel (`native_target` / `native_mutation_target`). It travels
    # with the bundle because the target is where a path is actually OPENED, so the
    # refusal has to live there and not only in Home's mirror of the rule.
    "ouroboros.workspace_native_paths",
    "ouroboros.workspace_payload_native",
    "ouroboros.workspace_query_native",
    "ouroboros.workspace_snapshot_native",
})


@runtime_checkable
class NativeExecutionControl(Protocol):
    """Execd-owned cancellation/custody callbacks for spawned process groups.

    An implementation MAY also expose an optional ``process_spool`` attribute
    holding a ``ProcessSpool`` already bound to the current task/operation; the
    kernel reads it with ``getattr`` so a control object without one keeps the
    memory-only capture behavior unchanged.
    """

    cancelled: Callable[[], bool]
    register_process: Callable[..., None]
    release_process: Callable[..., None]
    recover_service: Callable[..., Mapping[str, Any] | None]
    stop_service: Callable[..., bool]


@dataclass
class BoundedProcessStream:
    """Hash a whole process stream while retaining bounded head/tail evidence."""

    total_bytes: int = 0
    newline_count: int = 0
    last_byte: int | None = None
    digest: Any = field(default_factory=hashlib.sha256)
    head: bytearray = field(default_factory=bytearray)
    tail: bytearray = field(default_factory=bytearray)
    full: bytearray | None = field(default_factory=bytearray)

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        self.newline_count += chunk.count(b"\n")
        self.last_byte = chunk[-1]
        self.digest.update(chunk)
        if len(self.head) < PROCESS_PREVIEW_HEAD_BYTES:
            self.head.extend(chunk[: PROCESS_PREVIEW_HEAD_BYTES - len(self.head)])
        self.tail.extend(chunk)
        if len(self.tail) > PROCESS_PREVIEW_TAIL_BYTES:
            del self.tail[:-PROCESS_PREVIEW_TAIL_BYTES]
        if self.full is not None:
            if len(self.full) + len(chunk) <= PROCESS_FULL_CAPTURE_BYTES:
                self.full.extend(chunk)
            else:
                self.full = None

    @property
    def total_lines(self) -> int:
        if self.total_bytes <= 0:
            return 0
        return self.newline_count + (0 if self.last_byte == ord("\n") else 1)

    def metadata(self, stream_name: str) -> dict[str, Any]:
        previewed = min(
            self.total_bytes,
            PROCESS_PREVIEW_HEAD_BYTES + PROCESS_PREVIEW_TAIL_BYTES,
        )
        omitted_newlines = max(
            0,
            self.newline_count
            - bytes(self.head).count(b"\n")
            - bytes(self.tail).count(b"\n"),
        )
        return {
            "stream": stream_name,
            "sha256": self.digest.hexdigest(),
            "total_bytes": self.total_bytes,
            "total_lines": self.total_lines,
            "previewed_bytes": previewed,
            "omitted_bytes": max(0, self.total_bytes - previewed),
            "omitted_newlines": omitted_newlines,
            "full_log_available": self.full is not None,
        }

    def preview(self, stream_name: str) -> str:
        if self.total_bytes <= PROCESS_PREVIEW_HEAD_BYTES + PROCESS_PREVIEW_TAIL_BYTES:
            return bytes(self.full or self.head).decode("utf-8", errors="replace")
        meta = self.metadata(stream_name)
        marker = (
            f"\n… {stream_name}: omitted {meta['omitted_bytes']} bytes "
            f"({meta['omitted_newlines']} newline separators); "
            f"total={meta['total_bytes']} bytes/{meta['total_lines']} lines; "
            f"sha256={meta['sha256']} …\n"
        )
        return (
            bytes(self.head).decode("utf-8", errors="replace")
            + marker
            + bytes(self.tail).decode("utf-8", errors="replace")
        )


@runtime_checkable
class ProcessSpoolSink(Protocol):
    """One process stream's spool: bounded append, seal, and trace facts."""

    write: Callable[[bytes], int]
    seal: Callable[[], dict[str, Any] | None]
    trace: Callable[[], dict[str, Any]]


@runtime_checkable
class ProcessSpool(Protocol):
    """Operation-scoped spool factory (execd owns the implementation)."""

    open_stream: Callable[..., ProcessSpoolSink]


def open_process_spool_sinks(
    spool: ProcessSpool | None,
    terminate: Callable[[str], None],
    *,
    streams: tuple[str, ...] = ("stdout", "stderr"),
) -> dict[str, ProcessSpoolSink]:
    """Open one spool sink per stream; no spool means no sinks and no change."""

    if spool is None:
        return {}
    return {
        name: spool.open_stream(stream=name, terminate=terminate)
        for name in streams
    }


def seal_process_spool_sinks(
    sinks: Mapping[str, ProcessSpoolSink],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Seal every sink after the process ended; return (rows, trace facts)."""

    rows: dict[str, dict[str, Any]] = {}
    facts: dict[str, dict[str, Any]] = {}
    for name, sink in sinks.items():
        row = sink.seal()
        if row:
            rows[name] = dict(row)
        facts[name] = dict(sink.trace())
    return rows, facts


def process_capture_trace(
    captures: Mapping[str, BoundedProcessStream],
    spool_facts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per-stream capture metadata, with the spool's own facts folded in."""

    facts = dict(spool_facts or {})
    trace: dict[str, Any] = {}
    for name, capture in captures.items():
        meta = dict(capture.metadata(name))
        if name in facts:
            meta["spool"] = dict(facts[name])
            meta["full_log_available"] = int(facts[name].get("rejected_bytes") or 0) == 0
        trace[name] = meta
    return trace


def process_quota_notes(spool_facts: Mapping[str, Mapping[str, Any]]) -> str:
    """Disclose every stream whose quota forced a process-group termination."""

    return "".join(
        f"⚠️ PROCESS_LOG_QUOTA: {name} reached the {facts['quota_scope']} spool "
        f"quota after {facts['accepted_bytes']} accepted bytes; the process group "
        f"was terminated and every accepted byte is sealed as "
        f"{facts.get('blob_id') or '(unsealed)'}.\n"
        for name, facts in sorted(spool_facts.items())
        if facts.get("quota_scope")
    )


def process_log_artifact(
    stream_name: str,
    capture: BoundedProcessStream,
    spool_row: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, bytes | None]:
    """Return one stream's full-log artifact row plus inline bytes, if any.

    A sealed spool row wins: its blob lives on the target and is fetched on
    demand, so an oversized stream keeps every accepted byte instead of losing
    the whole artifact the way the memory-only capture must past
    ``PROCESS_FULL_CAPTURE_BYTES``.  Without a spool the honest answer for an
    oversized stream stays "no artifact" — the metadata (exact size/line
    counters plus the whole-stream hash) is the recovery handle, never a false
    full-log claim.
    """

    if capture.total_bytes <= PROCESS_PREVIEW_HEAD_BYTES + PROCESS_PREVIEW_TAIL_BYTES:
        return None, None
    if spool_row is not None:
        return dict(spool_row), None
    if capture.full is None:
        return None, None
    data = bytes(capture.full)
    digest = hashlib.sha256(data).hexdigest()
    return (
        {
            "name": f"{stream_name}.txt",
            "blob_id": digest,
            "sha256": digest,
            "size": len(data),
            "mime": "text/plain",
            "truncated": False,
            "full_log": True,
        },
        data,
    )


def render_process_result_text(
    result: ProcessExecutionResult,
    *,
    cwd: pathlib.Path | str | None = None,
    notes: str = "",
) -> str:
    """Render the one process-result text both placements must produce."""

    if process_is_search_no_match(result):
        return (
            notes
            + f"{describe_process_returncode(result.returncode, cwd=cwd)} (no matches)\n"
            + format_process_output(result.stdout, "")
        )
    if result.returncode:
        return (
            notes
            + "⚠️ SHELL_EXIT_ERROR: command exited with "
            + f"{describe_process_returncode(result.returncode, cwd=cwd)}.\n\n"
            + format_process_output(result.stdout, result.stderr)
        )
    return (
        notes
        + f"{describe_process_returncode(0, cwd=cwd)}\n"
        + format_process_output(result.stdout, result.stderr)
    )


@dataclass(frozen=True)
class NativePreparedOperation:
    execution_args: dict[str, Any]
    native_facts: dict[str, Any] = field(default_factory=dict)


# ── the bundled authorize-phase fact block (RWS v2 §3.1 step 2) ─────────────
#
# The three keys below are the WIRE NAMES of the facts the Home authorize phase
# consumes through `ouroboros.execution_facts.RemoteExecutionFacts`. They live in
# the contract module because both sides must agree on them and neither may
# import the other: the target FILLS them here, Home READS them there, and the
# only way a guard can see a target fact is through this block. There is exactly
# ONE of these per operation — a per-fact RPC is the structural bound this
# removes (codex #17).
NATIVE_FACT_PATH_STATS = "path_stats"
NATIVE_FACT_GIT_TOPLEVELS = "git_toplevels"
NATIVE_FACT_INTERPRETERS = "interpreters"

# A prepare response travels in one control frame, so the block is bounded. The
# cap is a REFUSAL, not a truncation: a silently short fact block would let a
# guard authorize over facts it thinks it has.
NATIVE_FACT_PATH_CAP = 1200


def native_relative_spelling(value: Any, *, default: str = ".") -> str:
    """Canonicalize a workspace-relative path spelling, identically on both sides.

    Pure string work — no filesystem, no ``pathlib`` — so Home and the target reach
    the same answer about the same spelling. That is why it lives in the CONTRACT
    rather than in the kernel: the target normalizes the ``path`` argument of every
    root-labelled operation during prepare, and Home has to reproduce that
    normalization to tell "the target tidied my spelling" from "the target
    substituted a different file". While only the target could compute it, Home had
    nothing to compare against and the two were indistinguishable.

    Raises ``ValueError`` on an absolute path, a traversal, or a NUL/control-character
    payload: those are refusals, not spellings to canonicalize. The third one is here
    for ROUTE PARITY — ``utils.safe_relpath`` has rejected it on the local route since
    long before there was a remote one, and a spelling that one route refuses and the
    other accepts is a spelling Home cannot reproduce, so the target's tidied answer
    would read as a substituted file rather than the refusal it should have been.
    """

    text = str(value if value is not None else default).strip().replace("\\", "/")
    text = text or default
    for char in text:
        if char == "\x00":
            raise ValueError("workspace-native path contains NUL byte")
        # `\t\n\r` are excluded from the refusal for the same reason `safe_relpath`
        # excludes them: matching the local rule EXACTLY is the property, and a native
        # route that refused MORE than the local one would be a new asymmetry pointing
        # the other way, not a fix. (Leading/trailing whitespace is already stripped
        # above, so this only concerns an interior one.)
        if ord(char) < 0x20 and char not in ("\t", "\n", "\r"):
            raise ValueError(
                f"workspace-native path contains control character U+{ord(char):04X}"
            )
    if text.startswith("/"):
        raise ValueError("workspace-native path must be relative")
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError("workspace-native path contains traversal")
    return "/".join(parts) or "."


def target_path_stat(path: pathlib.Path | str) -> dict[str, Any]:
    """One target path, in the shape `execution_facts.PathFact` is built from.

    ``kind`` follows symlinks so it matches ``Path.is_dir()`` exactly, while
    ``symlink`` reports whether the REQUESTED spelling was itself a link — the
    two answers the protected-path guards need to tell an escape from a
    directory. A broken symlink reads as missing, as ``exists()`` does.
    """

    candidate = pathlib.Path(str(path))
    canonical = candidate.resolve(strict=False)
    try:
        symlink = candidate.is_symlink()
    except OSError:
        symlink = False
    try:
        info = candidate.stat()
    except (OSError, ValueError):
        return {"canonical": canonical.as_posix(), "kind": "missing", "symlink": symlink, "size": -1}
    if pathlib.Path(candidate).is_dir():
        kind = "dir"
    elif candidate.is_file():
        kind = "file"
    else:
        kind = "other"
    return {
        "canonical": canonical.as_posix(),
        "kind": kind,
        "symlink": symlink,
        "size": int(info.st_size),
    }


def bundle_target_facts(
    facts: dict[str, Any],
    *,
    root: pathlib.Path,
    paths: Mapping[str, Any],
    git_dirs: Mapping[str, Any],
    interpreters: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill the bundled fact block on `facts`, in place, and return it.

    ``paths`` maps the SPELLING a Home guard will ask by to the target path to
    stat. Both spellings are recorded — the one asked for and the canonical one —
    because a guard may hold either, and a miss is a typed refusal rather than a
    Home probe.
    """

    stats: dict[str, Any] = {}
    for spelling, target in {root.as_posix(): root, **dict(paths)}.items():
        key = str(spelling or "").strip()
        if not key:
            continue
        row = target_path_stat(target)
        stats[key] = row
        stats.setdefault(str(row["canonical"]), row)
        if len(stats) > NATIVE_FACT_PATH_CAP:
            raise ValueError(
                f"prepare fact block exceeds {NATIVE_FACT_PATH_CAP} paths; "
                "the operation names more paths than one prepare can answer for"
            )
    facts[NATIVE_FACT_PATH_STATS] = stats
    facts[NATIVE_FACT_GIT_TOPLEVELS] = {
        str(key): str(value or "") for key, value in dict(git_dirs).items() if str(key or "").strip()
    }
    facts[NATIVE_FACT_INTERPRETERS] = {
        str(key): str(value or "") for key, value in dict(interpreters).items() if str(key or "").strip()
    }
    return facts


def _git_toplevel(directory: pathlib.Path, run_git: Callable[[list[str]], Any]) -> str:
    """The worktree toplevel of a target directory, or ``""`` if it is not one.

    Same semantics as the Home probe in ``execution_facts.LocalExecutionFacts``:
    a non-zero exit, a missing git, or a timeout all read as "not a worktree"
    rather than as an error, so the two placements answer the same question the
    same way.
    """

    try:
        probe = run_git(["git", "-C", str(directory), "rev-parse", "--show-toplevel"])
    except Exception:
        return ""
    if getattr(probe, "returncode", 1) != 0:
        return ""
    text = str(getattr(probe, "stdout", "") or "").strip()
    return pathlib.Path(text).resolve(strict=False).as_posix() if text else ""


def bundle_prepared_facts(
    native_facts: dict[str, Any],
    *,
    root: pathlib.Path,
    run_git: Callable[[list[str]], Any],
) -> dict[str, Any]:
    """Derive the bundled authorize-phase block from an already-prepared operation.

    The operation's own prepare already resolved every path it touches — this
    turns those resolutions into the fact shapes Home's guards read, without
    re-parsing the arguments and without a second round trip. The block is added
    for EVERY operation, including the ones that resolve nothing, so the presence
    of the keys never encodes which tool ran.
    """

    paths: dict[str, Any] = {}
    for key in ("resolved_path", "resolved_cwd", "candidate_path", "service_cwd"):
        spelling = str(native_facts.get(key) or "").strip()
        if spelling:
            paths[spelling] = pathlib.Path(spelling)
    for relative in native_facts.get("protected_paths") or []:
        text = str(relative or "").strip()
        if text:
            paths[text] = root.joinpath(*[part for part in text.split("/") if part])
    git_dirs: dict[str, Any] = {root.as_posix(): _git_toplevel(root, run_git)}
    cwd_text = str(native_facts.get("resolved_cwd") or "").strip()
    if cwd_text and cwd_text != root.as_posix():
        git_dirs[cwd_text] = _git_toplevel(pathlib.Path(cwd_text), run_git)
    interpreters: dict[str, Any] = {}
    resolved_interpreter = str(native_facts.get("interpreter") or "").strip()
    if resolved_interpreter:
        # Three spellings for one fact, and the REQUESTED one is the load-bearing
        # entry: Home's python pre-resolution asks by the token the model wrote
        # (`python3`), and a bundle that only knew the resolved path would make that
        # a typed miss — the operation would then launch the unresolved token while
        # the target had already chosen a different binary, so the authorized argv
        # and the executed argv would differ by construction.
        requested = str(native_facts.get("interpreter_requested") or "").strip()
        if requested:
            interpreters[requested] = resolved_interpreter
        interpreters[resolved_interpreter] = resolved_interpreter
        interpreters[pathlib.PurePath(resolved_interpreter).name] = resolved_interpreter
    runtime = str(native_facts.get("resolved_runtime") or "").strip()
    if runtime:
        interpreters[runtime] = runtime
        interpreters[pathlib.PurePath(runtime).name] = runtime
    return bundle_target_facts(
        native_facts,
        root=root,
        paths=paths,
        git_dirs=git_dirs,
        interpreters=interpreters,
    )


@dataclass(frozen=True)
class NativeOperationResult:
    envelope: ToolExecutionEnvelope
    blobs: dict[str, bytes] = field(default_factory=dict)


def native_error_result(
    exc: BaseException,
    *,
    operation: str,
    args: Mapping[str, Any],
    phase: str = "execute",
    domain: str = "filesystem",
    completion: str = "not_started",
) -> NativeOperationResult:
    """The per-operation refusal TEXT, beside the diagnostic vocabulary it renders.

    Moved out of `workspace_native`, which sits at its module ceiling, and into the module
    that already owns `NativeOperationResult` and imports the diagnostic renderers. It is
    a pure mapper from an exception to one operation's owner-facing sentence: no policy, no
    filesystem, no decision — which is why it belongs beside the vocabulary and not beside
    the doors.
    """

    diagnostic = diagnostic_from_exception(
        exc,
        request_id=str(args.get("_request_id") or ""),
        operation_id=str(args.get("_operation_id") or ""),
        phase=phase,
        domain=domain,  # type: ignore[arg-type]
        completion=completion,  # type: ignore[arg-type]
        details={"operation": operation},
    )
    rel = sanitize_execution_text(str(args.get("path") or "."))
    message = diagnostic.message
    if operation == "read_file":
        text = (
            f"⚠️ NOT_FOUND: active_workspace:{rel}"
            if isinstance(exc, FileNotFoundError)
            else f"⚠️ READ_FILE_ERROR: {type(exc).__name__}: {message}"
        )
    elif operation == "list_files":
        if isinstance(exc, FileNotFoundError):
            text = f"⚠️ LIST_FILES_ERROR: Directory not found: {rel}"
        elif isinstance(exc, NotADirectoryError):
            text = f"⚠️ LIST_FILES_ERROR: {message}"
        else:
            text = f"⚠️ LIST_FILES_ERROR ({type(exc).__name__}): {message}"
    elif operation == "write_file":
        text = (
            f"⚠️ FILE_WRITE_ERROR on '{rel}': {message}\n"
            "Successfully written before error: (none)"
        )
    elif operation == "edit_text":
        text = (
            f"⚠️ STR_REPLACE_ERROR: file not found: {rel}"
            if isinstance(exc, FileNotFoundError)
            else f"⚠️ STR_REPLACE_ERROR: {type(exc).__name__}: {message}"
        )
    elif operation == "search_code":
        text = (
            f"⚠️ SEARCH_ERROR: path not found: active_workspace:{rel}"
            if isinstance(exc, FileNotFoundError)
            else f"⚠️ SEARCH_ERROR: {type(exc).__name__}: {message}"
        )
    elif operation == "run_command" and isinstance(
        exc,
        subprocess.TimeoutExpired,
    ):
        timeout = max(
            1,
            int(float(args.get("timeout_sec") or args.get("timeout") or 120)),
        )
        cwd = str(args.get("cwd") or ".")
        text = (
            "⚠️ TOOL_TIMEOUT (run_command): command exceeded the per-command "
            f"timeout of {timeout}s and its subprocess tree was terminated "
            f"(cwd={pathlib.Path(cwd).resolve(strict=False)}). NOTE: this is "
            "the per-command FOREGROUND timeout, NOT the task deadline. For "
            "genuinely long-running compute (training, sampling, large "
            "builds/downloads), start it with start_service and poll "
            "service_status/service_logs while you do other work, or pass an "
            "explicit timeout_sec=<seconds> (up to the per-call ceiling) — and "
            "preserve a best-effort deliverable before the task deadline."
        )
    else:
        text = render_diagnostic_text(diagnostic)
    return NativeOperationResult(
        ToolExecutionEnvelope(
            text=text,
            diagnostic=diagnostic,
            trace={"operation": operation},
        )
    )

def validate_remote_native_operation_map(
    operation_modules: Mapping[str, str] = REMOTE_NATIVE_OPERATION_MODULE,
) -> None:
    """Fail if the explicit execd operation map is incomplete or over-broad."""

    names = frozenset(str(name) for name in operation_modules)
    missing = sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS - names)
    unexpected = sorted(names - MANDATORY_REMOTE_NATIVE_OPERATIONS)
    if missing or unexpected:
        raise ValueError(
            "remote native operation map mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    invalid = sorted(
        name
        for name, module in operation_modules.items()
        if not str(module or "").startswith("ouroboros.")
    )
    if invalid:
        raise ValueError(
            f"remote native operation modules must be Ouroboros modules: {invalid}"
        )

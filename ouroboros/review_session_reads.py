"""Harness-observed read receipts for a delegated review session.

A native inspection episode folds its OWN host-executed ``read_file`` receipts
over the required-source manifest (``review_native_episode._read_coverage``):
the host ran every read, so it knows the exact delivered character range. A
delegated Claudexor session reads through the vendor harness instead, so the
host sees none of those calls — but the daemon records each one as a
``tool_call`` event in the run's own ``<runDir>/attempts/*/events.jsonl``. This
module turns those events into receipts in the SAME vocabulary and folds them
over the SAME manifest, so a session row carries measured coverage with
provenance ``harness_observed`` instead of ``host_observed``.

These are diagnostic inferences, not host attestations of delivered bytes.
Incomplete journal evidence never invalidates a paid verdict or requires a
repeat review. A call the journal never reports as completed, a
command form this grammar does not model, a read whose delivered extent the
event never names, a pipeline, a redirection, or a file whose current bytes no
longer match the manifest row contributes no measured range. Missing ranges
describe the limits of this journal, not the reviewer's understanding.

Supported grammar — everything else counts as not read:
  - completion: a ``tool_call`` counts only when the journal also carries the
    ``tool_result`` for the same ``tool.use_id`` with ``status: "ok"`` and,
    where the harness names one, ``exit_code: 0``. A call that failed, exited
    non-zero, or never finished delivered nothing.
  - ``kind: "file"`` reads (``Read``/``read``/``view_file``): an integer
    ``offset``/``limit`` pair in the event's tool input is the delivered line
    window; a ``Read`` naming no window delivers from the first line up to the
    reader's own line bound, so only a file within that bound is proven whole;
    anything else is an opened file of unproven extent.
  - ``kind: "command"`` shell reads: ``cat``/``nl`` (whole file),
    ``sed -n 'A,Bp'`` (also ``A,$p`` and ``Ap``), ``head -n N``, ``tail -n N``,
    ``tail -n +N``. ``rg``/``grep``/``wc``/``ls``/``git`` and every other
    command search or inspect; they deliver no inferred file extent. Compound
    commands, pipelines and redirections are outside this small grammar;
    their overall exit code does not establish each command's output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import posixpath
import re
import shlex
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("review_session_reads")

# Character ranges address the universal-newline text ABI the host reader
# delivers (`tools/core_file_tools._read_source_text`).
RANGE_BASIS = "unicode_text_universal_newlines"
READ_PROVENANCE = "harness_observed"
# The one root a delegated session's reads address: the run's scope root.
SESSION_ROOT = "session_root"
_OUTSIDE_ROOT = "outside_session_root"
# Rows addressed at a repository root are the rows those reads reach — the same
# repository-root vocabulary the deep-review coverage reader uses.
REPOSITORY_ROOT_NAMES = frozenset({"", "active_workspace", "system_repo", SESSION_ROOT})
# The engine's run layout: one journal per attempt under the run directory.
_ATTEMPTS_DIR, _EVENTS_FILE = "attempts", "events.jsonl"
# Claude's file reader stops at this many lines when the call names no window.
_DEFAULT_READ_LINES = 2000
_READ_TOOL_NAMES = frozenset({"Read", "read", "view_file"})
_HARNESS_BY_TOOL = {"Read": "claude", "Bash": "claude", "read": "cursor", "view_file": "cursor",
                    "shell": "cursor", "command": "codex"}
_SEPARATORS = frozenset({"&&", "||", ";", "&"})
# Redirections are outside the modeled grammar, whether spaced or glued.
_REDIRECTION = re.compile(r"^(?:\d*(?:>>?|<<?|&>)|>\|)")
_WHOLE_FILE_COMMANDS = frozenset({"cat", "nl"})
_SHELLS = frozenset({"bash", "sh", "zsh", "dash"})
_SED_RANGE = re.compile(r"^(\d+)(?:,(\d+|\$))?p$")
_NUMERIC_OPTION = re.compile(r"^-\d+$")


def session_read_facts(run_dir: str, policy: Any, *, session_root: str,
                       store: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The read facts a SETTLED delegated review session carries.

    The same usage keys a native episode attaches
    (``review_native_episode._episode_source_facts``), with ``read_provenance``
    naming the weaker provenance: the host executed none of these reads. The
    manifest is the surface's ``native_required_sources`` or, when it declared
    none, its ``observed_sources`` — sources the host only OBSERVES (plan
    review's own-room snapshot): the fold and the coverage facts are identical,
    but ``native_incomplete`` is set for required sources only, so an observed
    source never reads as a capability delta. A session whose surface declared
    neither claims nothing at all.

    The session's verdict is PAID evidence, so every failure here is a
    disclosure instead of a refusal: a journal this host cannot find, read or
    parse is an ``unobserved`` coverage, never an exception out of settlement.

    ``store`` locates the durable record of the parsed reads in the operation's
    own actor-readable source store — ``{"root", "task_id", "source_id",
    "run_id"}``; without it, or when the write fails, the coverage stands alone.
    """
    policy = policy if isinstance(policy, dict) else {}
    if "native_required_sources" not in policy and "observed_sources" not in policy:
        return {}
    required = "native_required_sources" in policy
    unobserved = {"native_read_coverage": {"status": "unobserved",
                                           "reason": "session_events_unavailable", "sources": []},
                  "read_provenance": "unobserved"}
    try:
        events = session_event_files(run_dir)
        if not events:
            return unobserved
        receipts = parse_session_read_receipts(events, scope_root=session_root)
        coverage = fold_session_coverage(
            receipts, policy.get("native_required_sources") if required else policy.get("observed_sources"),
            resolve_file=session_source_reader(session_root))
    except Exception:
        log.debug("session read-coverage fold failed", exc_info=True)
        return unobserved
    facts = {"native_read_coverage": coverage,
             "native_history_source": _stored_read_history(store, policy, receipts, coverage),
             "read_provenance": READ_PROVENANCE}
    if required and coverage["status"] == "incomplete":
        facts["native_incomplete"] = "required_source_coverage_incomplete"
    return facts


def _stored_read_history(store: Any, policy: Dict[str, Any], receipts: List[Dict[str, Any]],
                         coverage: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the parsed reads retrievable where a native episode keeps its own
    history; a store failure leaves the coverage standing on its own."""
    if not isinstance(store, dict) or store.get("root") is None:
        return {}
    payload = {"required_sources": policy.get("native_required_sources"),
               "required_sources_ref": policy.get("native_required_sources_ref"),
               "observed_sources": policy.get("observed_sources"),
               "read_receipts": receipts, "coverage": coverage,
               "read_provenance": READ_PROVENANCE,
               "delegated_run_id": str(store.get("run_id") or "")}
    try:
        from ouroboros.artifacts import store_actor_source_bytes

        return store_actor_source_bytes(
            store["root"], str(store.get("task_id") or "review"), category="context_checkpoints",
            source_id=str(store.get("source_id") or "session-reads"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), extension="json")
    except Exception:
        log.debug("session read-history persistence failed", exc_info=True)
        return {}


def session_event_files(run_dir: str) -> List[pathlib.Path]:
    """Every attempt journal under one Claudexor run directory, oldest first."""
    root = pathlib.Path(str(run_dir or "")) / _ATTEMPTS_DIR
    try:
        return sorted(entry / _EVENTS_FILE for entry in root.iterdir()
                      if (entry / _EVENTS_FILE).is_file())
    except (OSError, ValueError, RuntimeError):
        return []


def parse_session_read_receipts(events_paths: Iterable[Any], *, scope_root: str) -> List[Dict[str, Any]]:
    """Infer file extents from completed reads in the harness journal.

    A receipt needs a COMPLETED call: the journal carries a ``tool_result`` for
    the same ``tool.use_id`` with ``status: "ok"`` and, where the harness names
    one, ``exit_code: 0``. A call that failed, exited non-zero or never
    finished cannot establish a range here. Call and result may arrive in
    either order, so the journals are collected first and paired afterwards.
    """
    calls: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    completed: set = set()
    for events_path in events_paths:
        try:
            handle = pathlib.Path(events_path).open(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        with handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("type") if isinstance(event, dict) else None
                if kind not in ("tool_call", "tool_result"):
                    continue
                tool = event.get("tool") if isinstance(event.get("tool"), dict) else {}
                use_id = str(tool.get("use_id") or "")
                if kind == "tool_result":
                    code = tool.get("exit_code")
                    if (use_id and str(tool.get("status") or "") == "ok"
                            and (code is None or (type(code) is int and code == 0))):
                        completed.add(use_id)
                    continue
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                arguments = payload.get("input") if isinstance(payload.get("input"), dict) else {}
                calls.append((use_id, tool, arguments))
    receipts: List[Dict[str, Any]] = []
    seen: set = set()
    for use_id, tool, arguments in calls:
        if use_id not in completed:
            continue
        for receipt in _tool_call_receipts(tool, arguments, scope_root):
            # One extent counted once: the journal keeps an in-progress and a
            # finished row for the same call, and a repeated identical read
            # adds nothing to an interval union.
            key = json.dumps([receipt["opened_root"], receipt["opened_path"],
                              receipt["whole_file"], receipt.get("start_line"),
                              receipt.get("end_line"), receipt.get("from_end_lines")])
            if key not in seen:
                seen.add(key)
                receipts.append(receipt)
    return receipts


def fold_session_coverage(receipts: List[Dict[str, Any]], required_sources: Any, *,
                          resolve_file: Callable[[Dict[str, Any]], Optional[str]]) -> Dict[str, Any]:
    """Fold journal-derived intervals over the caller's required manifest.

    The same four states and the same row shape ``_read_coverage`` produces, so
    one consumer reads a native episode and a delegated session alike. Each
    row's identity is verified against the CURRENT candidate file: a file whose
    bytes no longer hash to the declared source is a ``source_gap``, because
    whatever the harness read there, it was not this revision.
    """
    if not isinstance(required_sources, list):
        return {"status": "unobserved", "reason": "required_source_manifest_missing", "sources": []}
    if not required_sources:
        # The same shape the native episode returns for an empty manifest, so
        # one reducer (`scope_required_sources.coverage_state`) reads both.
        return {"status": "complete", "reason": "declared_empty", "sources": [],
                "required_source_count": 0}
    rows: List[Dict[str, Any]] = []
    for source in required_sources:
        row = dict(source) if isinstance(source, dict) else {"source": source}
        try:
            total = row["complete_chars"]
            if (type(total) is not int or total < 0 or len(row["source_revision"]) != 64
                    or len(row["complete_sha256"]) != 64 or row["range_basis"] != RANGE_BASIS):
                raise ValueError("invalid required source identity")
            if row.get("coverage_basis") == "delivered_inline":
                rows.append({**row, "status": "complete", "covered_chars": total,
                             "missing_ranges": []})
                continue
            text = resolve_file(row)
            if not isinstance(text, str):
                row.update(status="unobserved", reason="required_source_unreadable")
            elif len(text) != total or hashlib.sha256(text.encode("utf-8")).hexdigest() != row["complete_sha256"]:
                row.update(status="incomplete", reason="source_gap",
                           missing_ranges=[[0, total]], covered_chars=0)
            else:
                row.update(_folded_row(receipts, row, text, total, scope_root=str(getattr(resolve_file, "scope_root", ""))))
        except (KeyError, TypeError, ValueError, OSError):
            row.update(status="unobserved", reason="required_source_identity_unavailable")
        rows.append(row)
    return {"status": ("complete" if all(r["status"] == "complete" for r in rows)
                       else "incomplete" if any(r["status"] == "incomplete" for r in rows)
                       else "unobserved"),
            "sources": rows, "required_source_count": len(rows)}


def session_source_reader(session_root: str) -> Callable[[Dict[str, Any]], Optional[str]]:
    """Read a manifest row from the candidate tree the session reviewed.

    The session's scope root IS the candidate repository, so only rows addressed
    at a repository root are reachable from its reads; any other root (runtime
    data, a skill payload) stays unreadable and therefore unobserved — unless the
    row carries a host-declared absolute ``file`` (an observed artifact-store
    source outside the workspace), which is read as written. Text is normalized
    to the universal-newline ABI the manifest is written on.
    """
    base = pathlib.Path(str(session_root or ".")).resolve(strict=False)

    def read(row: Dict[str, Any]) -> Optional[str]:
        declared = str(row.get("file") or "")
        if declared and pathlib.Path(declared).is_absolute():
            target = pathlib.Path(declared)
        else:
            if str(row.get("root", "")) not in REPOSITORY_ROOT_NAMES:
                return None
            relative = posixpath.normpath(str(row.get("path") or "")).removeprefix("./")
            if not relative or relative == "." or relative.startswith(("..", "/")):
                return None
            target = base.joinpath(*relative.split("/"))
        try:
            raw = target.read_bytes()
            return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except (OSError, ValueError, UnicodeDecodeError):
            return None

    read.scope_root = str(session_root or "")  # the receipts' root, for host-declared absolute files
    return read


def _folded_row(receipts: List[Dict[str, Any]], row: Dict[str, Any], text: str, total: int,
                *, scope_root: str = "") -> Dict[str, Any]:
    # Start offset of every line plus the text end, on the SAME line definition
    # the host reader renders by (``str.splitlines``), so a harness's line window
    # and the manifest's character ranges meet on one basis.
    offsets, cursor = [0], 0
    for line in text.splitlines(keepends=True):
        cursor += len(line)
        offsets.append(cursor)
    # A receipt matches the row it opened: a repository row by its path under the session
    # root; a host-declared absolute `file` by the same normalization the receipts use.
    opened = (_normalized_path(str(row["file"]), scope_root) if row.get("file")
              else (str(row.get("path") or ""), SESSION_ROOT))
    delivered = [receipt for receipt in receipts
                 if isinstance(receipt, dict) and receipt.get("tool") == "read_file"
                 and receipt.get("outcome") == "executed" and receipt.get("delivered") is True
                 and not receipt.get("source_gap")
                 and (receipt.get("opened_path"), receipt.get("opened_root")) == opened]
    spans = [span for receipt in delivered
             for span in (_receipt_span(receipt, total, offsets),) if span is not None]
    cursor, missing = 0, []
    for low, high in sorted(spans):
        if low > cursor:
            missing.append([cursor, low])
        cursor = max(cursor, high)
    if cursor < total:
        missing.append([cursor, total])
    return {"status": ("complete" if spans and not missing else
                       "unobserved" if delivered and not spans else "incomplete"), "missing_ranges": missing,
            "covered_chars": total - sum(high - low for low, high in missing)}


def _receipt_span(receipt: Dict[str, Any], total: int, offsets: List[int]) -> Optional[Tuple[int, int]]:
    lines = len(offsets) - 1
    if receipt.get("whole_file"):
        bound = receipt.get("bounded_lines")
        # A reader that stops at its own line bound proves nothing about a
        # longer file: its delivered extent is unknown, not the whole file.
        return None if isinstance(bound, int) and 0 < bound < lines else (0, total)
    start, end, from_end = receipt.get("start_line"), receipt.get("end_line"), receipt.get("from_end_lines")
    if isinstance(from_end, int) and from_end > 0:
        start, end = max(1, lines - from_end + 1), lines
    if not isinstance(start, int) or start < 1 or start > lines:
        return None
    end = lines if end is None else end
    if not isinstance(end, int):
        return None
    end = min(end, lines)
    return (offsets[start - 1], offsets[end]) if end >= start else None


def _tool_call_receipts(tool: Dict[str, Any], arguments: Dict[str, Any],
                        scope_root: str) -> List[Dict[str, Any]]:
    name, kind = str(tool.get("name") or ""), str(tool.get("kind") or "")
    target = str(tool.get("target") or "")
    if kind == "file" and name in _READ_TOOL_NAMES:
        return _file_read_receipts(name, target, arguments, scope_root)
    if kind == "command":
        return _command_receipts(name, target, scope_root)
    # Search, web, MCP, edit and write tools deliver no proven file extent.
    return []


def _file_read_receipts(name: str, target: str, arguments: Dict[str, Any],
                        scope_root: str) -> List[Dict[str, Any]]:
    addressed = target.split(":", 1)[1] if target[:5].lower() == "read:" else target
    raw = str(arguments.get("file_path") or arguments.get("path") or addressed)
    offset, limit = arguments.get("offset"), arguments.get("limit")
    if type(limit) is int and limit > 0:
        start = offset if type(offset) is int and offset >= 1 else 1
        return _receipts(name, target, raw, scope_root, start_line=start, end_line=start + limit - 1)
    if name == "Read" and offset is None and limit is None:
        return _receipts(name, target, raw, scope_root, whole_file=True, bounded_lines=_DEFAULT_READ_LINES)
    # The file was opened; nothing in the event says how much of it arrived.
    return _receipts(name, target, raw, scope_root)


def _command_receipts(name: str, target: str, scope_root: str) -> List[Dict[str, Any]]:
    receipts: List[Dict[str, Any]] = []
    script = _shell_script(target).strip()
    if len(script.splitlines()) != 1:
        return []  # Do not infer execution inside a shell control-flow body.
    for line in script.splitlines():
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
            lexer.whitespace_split, lexer.commenters = True, ""
            tokens = list(lexer)
        except ValueError:
            continue  # unbalanced quoting: this line proves nothing
        if not tokens or any(token in _SEPARATORS or "|" in token for token in tokens):
            # A successful compound command need not have run every read.
            # Unknown shapes stay diagnostic; they never deny review authority.
            continue
        receipts.extend(_segment_receipts(tokens, name, target, scope_root))
    return receipts


def _shell_script(target: str) -> str:
    """The script a ``/bin/bash -lc "…"`` wrapper carries, else the target itself."""
    try:
        tokens = shlex.split(target, comments=False, posix=True)
    except ValueError:
        return target
    if len(tokens) >= 3 and posixpath.basename(tokens[0]) in _SHELLS:
        for index, token in enumerate(tokens[1:-1], start=1):
            if token.startswith("-") and token.endswith("c"):
                return tokens[index + 1]
    return target


def _segment_receipts(tokens: List[str], name: str, target: str, scope_root: str) -> List[Dict[str, Any]]:
    if not tokens or any(_REDIRECTION.match(token) for token in tokens):
        return []
    command, args = posixpath.basename(tokens[0]), tokens[1:]
    if command in _WHOLE_FILE_COMMANDS:
        return [row for operand in _operands(args)
                for row in _receipts(name, target, operand, scope_root, whole_file=True)]
    if command == "sed":
        return _sed_receipts(args, name, target, scope_root)
    if command in ("head", "tail"):
        return _head_tail_receipts(command, args, name, target, scope_root)
    return []


def _operands(args: List[str], valued: Tuple[str, ...] = ()) -> List[str]:
    operands, skip = [], False
    for token in args:
        if skip:
            skip = False
        elif token.startswith("-") and len(token) > 1:
            skip = token in valued
        else:
            operands.append(token)
    return operands


def _sed_receipts(args: List[str], name: str, target: str, scope_root: str) -> List[Dict[str, Any]]:
    if any(token.startswith("-") and token not in ("-n", "-e") for token in args):
        return []
    operands = _operands(args)
    scripts = [token for token in operands if _SED_RANGE.match(token)]
    paths = [token for token in operands if not _SED_RANGE.match(token)]
    if len(scripts) != 1 or not paths:
        return []  # several scripts or none: the delivered extent is not this one range
    start, stop = _SED_RANGE.match(scripts[0]).groups()
    end = None if stop == "$" else (int(stop) if stop else int(start))
    return [row for path in paths
            for row in _receipts(name, target, path, scope_root, start_line=int(start), end_line=end)]


def _head_tail_receipts(command: str, args: List[str], name: str, target: str,
                        scope_root: str) -> List[Dict[str, Any]]:
    count, from_start = 10, False
    for index, token in enumerate(args):
        if not token.startswith("-") or len(token) == 1:
            continue
        value = args[index + 1] if token == "-n" and index + 1 < len(args) else token[1:]
        # A byte window, a quiet flag, `head -n -N` or any other unmodelled
        # option leaves the delivered extent outside this grammar.
        if not (token == "-n" or _NUMERIC_OPTION.match(token)) or value.startswith("-"):
            return []
        from_start = value.startswith("+")
        try:
            count = int(value.lstrip("+"))
        except ValueError:
            return []
    if count <= 0:
        return []
    receipts: List[Dict[str, Any]] = []
    for path in _operands(args, valued=("-n",)):
        if command == "head" and not from_start:
            receipts.extend(_receipts(name, target, path, scope_root, start_line=1, end_line=count))
        elif command == "tail" and from_start:
            receipts.extend(_receipts(name, target, path, scope_root, start_line=count))
        elif command == "tail":
            receipts.extend(_receipts(name, target, path, scope_root, from_end_lines=count))
    return receipts


def _receipts(name: str, target: str, raw: str, scope_root: str, *, start_line: Optional[int] = None,
              end_line: Optional[int] = None, whole_file: bool = False, bounded_lines: int = 0,
              from_end_lines: int = 0) -> List[Dict[str, Any]]:
    opened, root = _normalized_path(raw, scope_root)
    if not opened:
        return []
    receipt: Dict[str, Any] = {
        "tool": "read_file", "outcome": "executed", "delivered": True,
        "opened_path": opened, "opened_root": root, "whole_file": bool(whole_file),
        "provenance": READ_PROVENANCE,
        "evidence": {"harness": _HARNESS_BY_TOOL.get(name, "unknown"), "tool": name,
                     "raw_target": str(target)[:300]},
    }
    for key, value in (("start_line", start_line), ("end_line", end_line),
                       ("bounded_lines", bounded_lines or None), ("from_end_lines", from_end_lines or None)):
        if value is not None:
            receipt[key] = int(value)
    if not whole_file and start_line is None and not from_end_lines:
        # Opened, extent unproven: disclosure in the record, zero coverage.
        receipt["range_unobserved"] = True
    return [receipt]


def _normalized_path(raw: str, scope_root: str) -> Tuple[str, str]:
    """The repo-relative path a read opened, and the root it opened it under."""
    text = str(raw or "").strip()
    if not text or text.startswith("-") or any(char in text for char in "$*?"):
        return "", ""
    candidate = pathlib.Path(text)
    if candidate.is_absolute():
        base = pathlib.Path(str(scope_root or ".")).resolve(strict=False)
        try:
            return candidate.resolve(strict=False).relative_to(base).as_posix(), SESSION_ROOT
        except (ValueError, OSError):
            return candidate.as_posix(), _OUTSIDE_ROOT
    normalized = posixpath.normpath(text.replace("\\", "/")).removeprefix("./")
    if not normalized or normalized == "." or normalized.startswith(".."):
        return normalized, _OUTSIDE_ROOT
    return normalized, SESSION_ROOT

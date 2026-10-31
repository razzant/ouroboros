"""Evidence manifest for ``plan_task`` — pure functions, bounded I/O on
agent-declared locators only (companion of ``ouroboros.tools.plan_spec``).

Bounds (every cut disclosed, never silent): ``EVIDENCE_PER_ITEM_BYTES`` /
``EVIDENCE_TOTAL_BYTES`` — attached text per locator and per packet (~10k /
~30k tokens at chars/4); ``EVIDENCE_HASH_BYTES_LIMIT`` — a source above it is
refused unread (``too_large:<bytes>``), never streamed. Path locators go through
``review_helpers._SENSITIVE_*`` (lexical name AND resolved target) and the
caller's ``allowed_roots``; URLs are listed, never fetched; the resolver never
raises on agent-controlled input (every failure is a typed omission).
"""

from __future__ import annotations

import codecs
import ast
import errno
from hashlib import sha256
import pathlib
import json
import re
import stat
from typing import Any, Callable, Iterable, Optional

from ouroboros.tools.plan_spec import (
    _TASK_LOCATOR_PREFIX,
    _is_url,
    _resolve_locator_path,
    _under,
)
from ouroboros.tools.review_helpers import (
    _BINARY_SNIFF_BYTES,
    _SENSITIVE_EXTENSIONS,
    _SENSITIVE_NAMES,
    _raw_bytes_binary,
)

EVIDENCE_PER_ITEM_BYTES = 40_000
EVIDENCE_TOTAL_BYTES = 120_000
EVIDENCE_HASH_BYTES_LIMIT = 8 * 1024 * 1024  # never stream/hash a source above this (too_large)

_SELECTOR_RE = re.compile(
    r"^(?P<locator>.+)::(?:(?:lines=(?P<line_start>[1-9][0-9]*)-(?P<line_end>[1-9][0-9]*))|"
    r"(?:bytes=(?P<byte_start>[0-9]+)-(?P<byte_end>[0-9]+))|"
    r"(?:tail=(?P<tail>[1-9][0-9]*))|(?:symbol=(?P<symbol>[A-Za-z_][A-Za-z0-9_.]*)))$"
)


def _split_selector(locator: str) -> tuple[str, Optional[dict], Optional[str]]:
    """Parse the additive exact-selector suffix carried by a locator.

    Line and byte endpoints are inclusive. Byte offsets are zero-based. A
    selector-looking suffix with an invalid range is a typed omission rather
    than a path that happens not to exist.
    """
    match = _SELECTOR_RE.fullmatch(locator)
    if match:
        base = match.group("locator")
        if match.group("line_start"):
            start, end = int(match.group("line_start")), int(match.group("line_end"))
            return base, {"kind": "line_range", "start": start, "end": end}, None if end >= start else "invalid_selector"
        if match.group("byte_start"):
            start, end = int(match.group("byte_start")), int(match.group("byte_end"))
            return base, {"kind": "byte_range", "start": start, "end": end}, None if end >= start else "invalid_selector"
        if match.group("tail"):
            return base, {"kind": "tail", "bytes": int(match.group("tail"))}, None
        return base, {"kind": "symbol", "name": str(match.group("symbol"))}, None
    if "::" in locator and locator.rsplit("::", 1)[-1].split("=", 1)[0] in {"lines", "bytes", "tail", "symbol"}:
        return locator, None, "invalid_selector"
    return locator, None, None


def _selected_payload(raw: bytes, selector: dict, path: Optional[pathlib.Path] = None, *, physical_lf: bool = False) -> tuple[Optional[dict], Optional[str]]:
    digest = sha256(raw).hexdigest()
    kind = selector["kind"]
    selected: bytes
    if kind == "byte_range":
        start, end = int(selector["start"]), int(selector["end"])
        if start >= len(raw) or end >= len(raw):
            return None, "selector_out_of_range"
        selected = raw[start:end + 1]
    elif kind == "tail":
        count = int(selector["bytes"])
        selected = raw[-count:]
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, "binary"
        lines = text.splitlines(keepends=True)
        if physical_lf:
            parts = text.split("\n")
            lines = [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
        if kind == "line_range":
            start, end = int(selector["start"]), int(selector["end"])
            if start > len(lines) or end > len(lines):
                return None, "selector_out_of_range"
            selected = "".join(lines[start - 1:end]).encode("utf-8")
        else:
            if path is None or path.suffix != ".py":
                return None, "symbol_selector_unsupported"
            try:
                tree = ast.parse(text)
            except SyntaxError:
                return None, "symbol_source_invalid"
            parts = str(selector["name"]).split(".")
            scope: Any = tree
            node = None
            for part in parts:
                body = getattr(scope, "body", [])
                node = next(
                    (item for item in body
                     if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and item.name == part),
                    None,
                )
                if node is None:
                    break
                scope = node
            if node is None:
                return None, "symbol_not_found"
            start = int(getattr(node, "lineno", 0) or 0)
            end = int(getattr(node, "end_lineno", 0) or 0)
            if not start or not end:
                return None, "symbol_range_unavailable"
            selector.update({"start_line": start, "end_line": end})
            selected = "".join(lines[start - 1:end]).encode("utf-8")
    try:
        selected_text = selected.decode("utf-8")
    except UnicodeDecodeError:
        return None, "binary"
    return {
        "sha256": digest,
        "bytes": len(raw),
        "text": selected_text,
        "selection_sha256": sha256(selected).hexdigest(),
        "selection_bytes": len(selected),
        "selector": selector,
    }, None


def _read_selected_evidence(
    path: pathlib.Path, selector: dict, read_text: Optional[Callable[[pathlib.Path], str]], hash_limit: int,
) -> tuple[Optional[dict], Optional[str]]:
    try:
        raw = (str(read_text(path)).encode("utf-8") if read_text is not None else path.read_bytes())
    except UnicodeError:
        return None, "binary"
    except OSError:
        return None, "unreadable"
    if len(raw) > hash_limit:
        return None, f"too_large:{len(raw)}"
    return _selected_payload(raw, selector, path)


def _decode_head(head: bytes, *, complete: bool) -> str:
    """Strict UTF-8; a truncated head drops ONLY its final partial character (incremental
    decoder, ``final=False``) — invalid bytes anywhere raise ``UnicodeDecodeError``."""
    return codecs.getincrementaldecoder("utf-8")().decode(head, final=complete)


def _payload_from_text(text: str, per_item: int) -> tuple[Optional[dict], Optional[str]]:
    """In-memory evidence payload: full sha256/bytes, head bounded to ``per_item`` bytes.
    Unencodable text (lone surrogates) → ``binary``."""
    try:
        full = text.encode("utf-8")
        head = text if len(full) <= per_item else _decode_head(full[:per_item], complete=False)
    except UnicodeError:
        return None, "binary"
    return {"sha256": sha256(full).hexdigest(), "bytes": len(full), "text": head}, None


def _payload_from_file(path: pathlib.Path, per_item: int, hash_limit: int) -> tuple[Optional[dict], Optional[str]]:
    """Stream the file once: full-content sha256/bytes, head bounded to ``per_item`` bytes
    (never the whole file in memory); sources above ``hash_limit`` are refused unread
    (``too_large:<bytes>``). → (payload, None) or (None, reason)."""
    digest, head, total = sha256(), bytearray(), 0
    try:
        size = path.stat().st_size
        if size > hash_limit:
            return None, f"too_large:{size}"
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(65_536)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                if len(head) < per_item:
                    head.extend(chunk[: per_item - len(head)])
    except OSError:
        return None, "unreadable"
    if _raw_bytes_binary(bytes(head[:_BINARY_SNIFF_BYTES])):
        return None, "binary"
    try:
        text = _decode_head(bytes(head), complete=total <= per_item)
    except UnicodeDecodeError:
        return None, "binary"
    return {"sha256": digest.hexdigest(), "bytes": total, "text": text}, None


def _read_evidence(
    path: pathlib.Path, read_text: Optional[Callable[[pathlib.Path], str]], per_item: int, hash_limit: int,
) -> tuple[Optional[dict], Optional[str]]:
    """→ (payload, omission_reason). Binary/undecodable → ``binary``; I/O failure → ``unreadable``."""
    if read_text is None:
        return _payload_from_file(path, per_item, hash_limit)
    try:
        return _payload_from_text(str(read_text(path)), per_item)
    except UnicodeError:
        return None, "binary"
    except OSError:
        return None, "unreadable"


def _path_kind(path: pathlib.Path) -> str:
    """``missing`` · ``directory`` · ``file`` · ``symlink_loop`` · ``unreadable`` (stat failure or a
    non-regular node: fifo/device/socket would block or never end a read)."""
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        return "symlink_loop" if exc.errno == errno.ELOOP else "unreadable"
    except (ValueError, RuntimeError):
        return "unreadable"
    if stat.S_ISDIR(mode):
        return "directory"
    return "file" if stat.S_ISREG(mode) else "unreadable"


def _sensitive(path: pathlib.PurePath) -> bool:
    return path.suffix.lower() in _SENSITIVE_EXTENSIONS or path.name.lower() in _SENSITIVE_NAMES


def resolve_evidence(
    locators: Iterable[str],
    *,
    active_root: str | pathlib.Path,
    allowed_roots: Iterable[str | pathlib.Path],
    read_text: Optional[Callable[[pathlib.Path], str]] = None,
    per_item_bytes: int = EVIDENCE_PER_ITEM_BYTES,
    total_bytes: int = EVIDENCE_TOTAL_BYTES,
    resolve_task: Optional[Callable[[str], Optional[str]]] = None,
    resolve_chat: Optional[Callable[[str], Optional[dict | str]]] = None,
    hash_bytes_limit: int = EVIDENCE_HASH_BYTES_LIMIT,
    deny_paths: Iterable[str | pathlib.Path] = (),
) -> dict:
    """Resolve agent-declared evidence locators into a bounded manifest.

    ``{declared: [..], attached: [{locator, kind: path|task, sha256, bytes,
    attached_bytes, text}], omissions: [{locator, reason}], bounds: {...}}`` in
    declared order. Reasons: ``outside_allowed_roots`` · ``sensitive``
    (``review_helpers._SENSITIVE_*`` policy) · ``missing`` · ``directory`` (no
    tree walk) · ``binary`` · ``unreadable`` (I/O failure, over-long name,
    non-regular file) · ``symlink_loop`` · ``too_large:<bytes>`` (above
    ``hash_bytes_limit``, never read) · ``truncated_to_N`` (head attached;
    sha256/bytes describe the FULL source) · ``budget_exhausted`` ·
    ``url_not_fetched`` (host never fetches; ``file://`` is a path) ·
    ``task_not_found`` · ``duplicate_locator`` · ``unsupported_locator`` ·
    ``denied_path`` (a caller-declared deny root — the runtime data plane and the
    live settings file, refused INDEPENDENTLY of allowed-root membership so a
    custom subject root can never make credentials attachable). Never
    raises on agent-controlled locators. ``read_text`` overrides file reading
    (tests / HEAD snapshots); ``resolve_task`` serves ``task:<id>``.
    """
    active = pathlib.Path(active_root).resolve(strict=False)
    roots = [pathlib.Path(r).resolve(strict=False) for r in allowed_roots]
    denied = [pathlib.Path(r).resolve(strict=False) for r in deny_paths if str(r or "").strip()]
    declared: list[str] = []
    attached: list[dict] = []
    omissions: list[dict] = []
    remaining = max(0, int(total_bytes))
    per_item = max(1, int(per_item_bytes))

    def attach(locator: str, kind: str, payload: dict) -> None:
        nonlocal remaining
        # Attached text is the source's own bytes (or its named head/selection): the
        # path, root and deny rules decide what may attach, never its content.
        head_bytes = len(payload["text"].encode("utf-8"))
        if head_bytes > remaining:
            omissions.append({"locator": locator, "reason": "budget_exhausted"})
            return
        remaining -= head_bytes
        row = {
            "locator": locator, "kind": kind, "sha256": payload["sha256"],
            "bytes": payload["bytes"], "attached_bytes": head_bytes, "text": payload["text"],
        }
        for key in ("selector", "selection_sha256", "selection_bytes"):
            if key in payload:
                row[key] = payload[key]
        if payload.get("secrets_redacted"):  # a retained source masked before capture says so
            row["secrets_redacted"] = True
        attached.append(row)
        if "selector" not in payload and payload["bytes"] > per_item:
            omissions.append({"locator": locator, "reason": f"truncated_to_{per_item}"})

    for raw in locators or []:
        locator = str(raw or "").strip()
        if not locator:
            continue
        if locator in declared:
            omissions.append({"locator": locator, "reason": "duplicate_locator"})
            continue
        declared.append(locator)
        source_locator, selector, selector_error = _split_selector(locator)
        if selector_error:
            omissions.append({"locator": locator, "reason": selector_error})
            continue
        if _is_url(source_locator):
            omissions.append({"locator": locator, "reason": "url_not_fetched"})
            continue
        if source_locator.startswith((_TASK_LOCATOR_PREFIX, "chat:")):
            kind = "task" if source_locator.startswith(_TASK_LOCATOR_PREFIX) else "chat"
            reader = resolve_task if kind == "task" else resolve_chat
            if reader is None:
                omissions.append({"locator": locator, "reason": "unsupported_locator"})
                continue
            try:
                source = reader(source_locator.split(":", 1)[1].strip())
            except (OSError, ValueError) as exc:
                omissions.append({"locator": locator, "reason": f"{kind}_unreadable:{type(exc).__name__}"})
                continue
            text = source.get("text") if isinstance(source, dict) else source
            if text is None:
                payload, reason = None, f"{kind}_not_found"
            elif selector:
                payload, reason = _selected_payload(str(text).encode("utf-8"), selector, physical_lf=kind == "chat")
            else:
                payload, reason = _payload_from_text(str(text), per_item)
            if payload is None:
                omissions.append({"locator": locator, "reason": reason})
            else:
                if isinstance(source, dict) and source.get("secrets_redacted"):
                    payload["secrets_redacted"] = True
                attach(locator, kind, payload)
                if isinstance(source, dict) and source.get("coverage"):
                    coverage = source["coverage"]
                    omissions.extend({"locator": locator, "reason": f"chat_history_gap:{gap['kind']}"}
                                     for section in coverage.values() if isinstance(section, dict)
                                     for gap in section.get("gaps", []))
                    omissions.extend({"locator": locator, "reason": f"chat_history_gap:{coverage[key]}"}
                                     for key in ("room_gap", "ordering_gap") if coverage.get(key))
                    if coverage.get("mailbox", {}).get("complete") is False:
                        omissions.append({"locator": locator, "reason": "chat_history_gap:mailbox_incomplete"})
            continue
        path, reason = _resolve_locator_path(source_locator, active)
        if path is None:
            omissions.append({"locator": locator, "reason": reason})
            continue
        if any(path == deny or _under(path, deny) for deny in denied):
            omissions.append({"locator": locator, "reason": "denied_path"})
            continue
        if not any(_under(path, root) for root in roots):
            omissions.append({"locator": locator, "reason": "outside_allowed_roots"})
            continue
        if _sensitive(path) or _sensitive(pathlib.PurePosixPath(source_locator.replace("\\", "/"))):
            omissions.append({"locator": locator, "reason": "sensitive"})  # lexical AND resolved name
            continue
        kind = _path_kind(path)
        if kind != "file":
            omissions.append({"locator": locator, "reason": kind})
            continue
        payload, reason = (
            _read_selected_evidence(path, selector, read_text, hash_bytes_limit)
            if selector else _read_evidence(path, read_text, per_item, hash_bytes_limit)
        )
        if payload is None:
            omissions.append({"locator": locator, "reason": reason or "unreadable"})
            continue
        attach(locator, "path", payload)
    return {
        "declared": declared, "attached": attached, "omissions": omissions,
        "bounds": {"per_item_bytes": per_item, "total_bytes": max(0, int(total_bytes))},
    }


def task_evidence_reader(root: pathlib.Path) -> Callable[[str], Optional[str]]:
    """Task-result projection with the WHOLE result; the evidence resolver hashes,
    budgets and attaches it. One bound: ``EVIDENCE_PER_ITEM_BYTES`` cuts the attached
    head with the cut named (``truncated_to_N``) while sha256/bytes describe the full
    projection, and a locator selector reads the original. An inner preview here would
    hash the preview, so two results differing past it would share one identity and a
    tail selector would read the omission marker instead of the result."""
    from ouroboros.task_results import load_task_result
    def _read(task_id: str) -> Optional[str]:
        try:
            record = load_task_result(root, task_id)
        except Exception:
            return None
        if not isinstance(record, dict):
            return None
        projection = {
            "task_id": task_id,
            "status": record.get("status"),
            "reason_code": record.get("reason_code"),
            "ts": record.get("ts"),
        }
        if "terminal_host_notice" in record:
            # Before the result: a bounded head keeps the host's own disclosure.
            projection["terminal_host_notice"] = str(record["terminal_host_notice"] or "")
        projection["result"] = str(record.get("result") or "")
        return json.dumps(projection, ensure_ascii=False, indent=2, default=str)
    return _read

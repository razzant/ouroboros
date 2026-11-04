"""Evidence manifest for ``plan_task`` — pure functions, bounded I/O on
agent-declared locators only (companion of ``ouroboros.tools.plan_spec``).

Bounds (every cut disclosed, never silent): ``EVIDENCE_PER_ITEM_BYTES`` /
``EVIDENCE_TOTAL_BYTES`` — attached text per locator and per packet (~10k /
~30k tokens at chars/4); ``EVIDENCE_HASH_BYTES_LIMIT`` — a source above it is
refused unread (``too_large:<bytes>``), never streamed. Path locators go through
the caller's deny roots, ``review_helpers._SENSITIVE_*`` (lexical name AND resolved
target), the caller's ``allowed_roots`` and then ``read_file``'s own read admission;
beneath a deny root only the caller's own task stores attach (``own_task_stores`` and
``evidence_read_admission`` bind those task facts, as ``task_evidence_reader`` binds
task results), and one seen in part keeps its whole source (``own_source_retention``);
URLs are listed, never fetched; the resolver never raises on
agent-controlled input (every failure is a typed omission).
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
    keep: Optional[bytearray] = None,
) -> tuple[Optional[dict], Optional[str]]:
    try:
        raw = (str(read_text(path)).encode("utf-8") if read_text is not None else path.read_bytes())
    except UnicodeError:
        return None, "binary"
    except OSError:
        return None, "unreadable"
    if len(raw) > hash_limit:
        return None, f"too_large:{len(raw)}"
    if keep is not None:
        keep.extend(raw)
    return _selected_payload(raw, selector, path)


def _decode_head(head: bytes, *, complete: bool) -> str:
    """Strict UTF-8; a truncated head drops ONLY its final partial character (incremental
    decoder, ``final=False``) — invalid bytes anywhere raise ``UnicodeDecodeError``."""
    return codecs.getincrementaldecoder("utf-8")().decode(head, final=complete)


def _payload_from_text(text: str, per_item: int, keep: Optional[bytearray] = None) -> tuple[Optional[dict], Optional[str]]:
    """In-memory evidence payload: full sha256/bytes, head bounded to ``per_item`` bytes.
    Unencodable text (lone surrogates) → ``binary``."""
    try:
        full = text.encode("utf-8")
        head = text if len(full) <= per_item else _decode_head(full[:per_item], complete=False)
    except UnicodeError:
        return None, "binary"
    if keep is not None:
        keep.extend(full)
    return {"sha256": sha256(full).hexdigest(), "bytes": len(full), "text": head}, None


def _payload_from_file(path: pathlib.Path, per_item: int, hash_limit: int,
                       keep: Optional[bytearray] = None) -> tuple[Optional[dict], Optional[str]]:
    """Stream the file once: full-content sha256/bytes, head bounded to ``per_item`` bytes
    (never the whole file in memory unless ``keep`` asks for the hashed bytes, then at most
    ``hash_limit`` of them); sources above ``hash_limit`` are refused unread
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
                if keep is not None:
                    if total > hash_limit:  # grew past the bound while read: refused like a larger source
                        return None, f"too_large:{total}"
                    keep.extend(chunk)
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
    keep: Optional[bytearray] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """→ (payload, omission_reason). Binary/undecodable → ``binary``; I/O failure → ``unreadable``.
    ``keep`` receives the exact bytes the payload's sha256/bytes and head were computed from."""
    if read_text is None:
        return _payload_from_file(path, per_item, hash_limit, keep)
    try:
        return _payload_from_text(str(read_text(path)), per_item, keep)
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


def _own_store_root(path: pathlib.Path, stores: list, denied: list) -> Optional[str]:
    """The typed root of the own store holding ``path`` when every deny entry holding it lies above that store."""
    holding = [deny for deny in denied if path == deny or _under(path, deny)]
    return next((label for label, base in stores
                 if _under(path, base) and not any(_under(deny, base) for deny in holding)), None)


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
    own_stores: Iterable[tuple[str, str | pathlib.Path]] = (),
    admit: Optional[Callable[[Optional[str], pathlib.Path], str]] = None,
    retain: Optional[Callable[[bytes], dict]] = None,
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
    custom subject root can never make credentials attachable). ``admit(root, path)`` decides each file that
    passed them: ``""`` admits, else its reason (``read_blocked:<CODE>``, ``read_target_differs``); ``root`` is
    the typed root of the caller's own task store ``(root, base)`` in ``own_stores`` holding the file, else None.
    An own-store file that no deny entry at or inside its store holds skips the deny and allowed-root checks; without
    ``admit`` nothing beneath a deny root attaches. An attached own-store file seen in part (a head or a selection)
    hands ``retain`` the exact bytes its part and hash were read from; the row carries what it returns (``ref``,
    ``file``) and its failure is the omission ``source_unretained:<Error>``. Never raises on agent-controlled
    locators. ``read_text`` overrides file reading (tests / HEAD snapshots); ``resolve_task`` serves ``task:<id>``.
    """
    active = pathlib.Path(active_root).resolve(strict=False)
    roots = [pathlib.Path(r).resolve(strict=False) for r in allowed_roots]
    denied = [pathlib.Path(r).resolve(strict=False) for r in deny_paths if str(r or "").strip()]
    stores = [(str(label), pathlib.Path(base).resolve(strict=False)) for label, base in own_stores] if admit else []
    declared: list[str] = []
    attached: list[dict] = []
    omissions: list[dict] = []
    remaining = max(0, int(total_bytes))
    per_item = max(1, int(per_item_bytes))

    def attach(locator: str, kind: str, payload: dict) -> Optional[dict]:
        nonlocal remaining
        # Attached text is the source's own bytes (or its named head/selection): the
        # path, root and deny rules decide what may attach, never its content.
        head_bytes = len(payload["text"].encode("utf-8"))
        if head_bytes > remaining:
            omissions.append({"locator": locator, "reason": "budget_exhausted"})
            return None
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
        return row

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
        store = _own_store_root(path, stores, denied)
        if store is None and any(path == deny or _under(path, deny) for deny in denied):
            omissions.append({"locator": locator, "reason": "denied_path"})
            continue
        if store is None and not any(_under(path, root) for root in roots):
            omissions.append({"locator": locator, "reason": "outside_allowed_roots"})
            continue
        if _sensitive(path) or _sensitive(pathlib.PurePosixPath(source_locator.replace("\\", "/"))):
            omissions.append({"locator": locator, "reason": "sensitive"})  # lexical AND resolved name
            continue
        kind = _path_kind(path)
        if kind != "file":
            omissions.append({"locator": locator, "reason": kind})
            continue
        if admit is not None and (refused := admit(store, path)):
            omissions.append({"locator": locator, "reason": refused})
            continue
        # An own-store file goes with its task's drive: the bytes kept from this one read are the ones
        # hashed, cut and selected, so a part the council sees keeps exactly that whole, never a reread.
        keep = bytearray() if store is not None and retain is not None else None
        payload, reason = (
            _read_selected_evidence(path, selector, read_text, hash_bytes_limit, keep)
            if selector else _read_evidence(path, read_text, per_item, hash_bytes_limit, keep)
        )
        if payload is None:
            omissions.append({"locator": locator, "reason": reason or "unreadable"})
            continue
        row = attach(locator, "path", payload)
        if row is not None and keep is not None and (selector or payload["bytes"] > per_item):
            try:
                held = retain(bytes(keep))
                row.update(ref=dict(held["ref"]), file=str(held["file"]))
            except Exception as exc:  # the part stays attached; its whole source is named absent, never implied
                omissions.append({"locator": locator, "reason": f"source_unretained:{type(exc).__name__}"})
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


def own_task_stores(ctx: Any) -> list[tuple[str, pathlib.Path]]:
    """The caller's own ``task_drive`` and ``artifact_store`` at the host's addresses on its canonical and
    execution drives, typed by root for ``evidence_read_admission``. Never another task's, nor a store a
    symlink moved elsewhere: a moved store is not where the host put it."""
    from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
    from ouroboros.headless import TASK_DRIVES_DIR
    from ouroboros.tool_access import canonical_data_root

    own = task_id_for_artifacts(ctx)
    drives = dict.fromkeys((canonical_data_root(ctx), pathlib.Path(ctx.drive_root).resolve(strict=False)))
    stores = [(label, base) for drive in drives for label, base in (
        ("task_drive", drive / TASK_DRIVES_DIR / own), ("artifact_store", task_artifact_dir_path(drive, own)))]
    return [(label, base) for label, base in stores if base.resolve(strict=False) == base]


def own_source_retention(root: pathlib.Path, task_id: str) -> Callable[[bytes], dict]:
    """``resolve_evidence``'s ``retain`` for the asking task: the whole source behind a part the council saw, kept
    write-once in its artifact store under ``root``, where its plan record lives (``store_actor_source_bytes``)
    → ``{"ref": task_source, "file": its placement}``. The ask row carries the ref, which the record's source
    closure follows (``review_source_closure``) through a child's promotion and the drive cleanup that removes
    the original. Stored as text, never ``json``: these are the file's bytes, not a host record to rebind."""
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path

    def retain(raw: bytes) -> dict:
        ref = store_actor_source_bytes(root, task_id, category="context_checkpoints", source_id="plan-evidence",
                                       data=raw, extension="txt")
        return {"ref": ref, "file": str(task_artifact_dir_path(root, task_id, create=False) / ref["path"])}
    return retain


def evidence_read_admission(ctx: Any, governance: Optional[pathlib.Path] = None
                            ) -> Callable[[Optional[str], pathlib.Path], str]:
    """``read_file``'s ``read_admission`` (profile, Presence ceiling, runtime-data rules, protected artifacts) for
    one evidence file, under its own store's typed root or else the root ``read_file`` selects for that absolute
    address: ``""`` when it admits exactly that file, ``read_blocked:<CODE>`` when it refuses, and
    ``read_target_differs`` when the read would bind another file at that address (an alias to a canonical
    copy, which attaches by its own address): one path is never admitted and another read. ``governance`` is
    the host's serving governance root (``plan_ask.planning_roots``): while a bound body candidate is what
    ``system_repo`` names, a file physically under that exact root that ``read_file``'s own selection does not
    admit as itself keeps its own bytes through a ``system_repo`` binding to that root, which the same admission
    checks; the candidate's file is never read in its place."""
    from ouroboros.tool_access import (
        ResolvedResourceBinding,
        active_tool_profile,
        canonical_data_root,
        resource_root_path,
    )
    from ouroboros.tools.core_file_tools import read_admission
    from ouroboros.tools.tool_resolution import _root_containing_absolute_path

    system_repo = resource_root_path(ctx, "system_repo")
    serving = pathlib.Path(governance).resolve(strict=False) if governance is not None else None
    if serving == system_repo:
        serving = None  # no candidate: read_file's own selection already binds this root

    def decide(root: Optional[str], path: pathlib.Path, base: Optional[pathlib.Path] = None) -> str:
        """One admission: under ``root``, else the root ``read_file`` selects; ``base`` binds ``system_repo`` there."""
        try:
            binding = None if base is None else ResolvedResourceBinding(
                profile=active_tool_profile(ctx), root="system_repo", operation="read", base_path=base,
                target_path=path, source="system_repo", skill_name="", state_drive_root=canonical_data_root(ctx))
            root = root or _root_containing_absolute_path(ctx, "read_file", str(path)) or "active_workspace"
            _root, binding, refusal = read_admission(ctx, root, str(path), resolved_binding=binding)
        except Exception as exc:  # a broken carrier refuses this file; the resolver itself never raises
            return f"read_blocked:{type(exc).__name__}"
        if refusal is not None:
            code = getattr(refusal, "code", "")
            if not code:  # a protected path's refusal is its final text, "⚠️ RESOURCE_POLICY_BLOCKED: …"
                match = re.search(r"[A-Z][A-Z0-9_]{2,}", str(refusal))
                code = match[0] if match else "REFUSED"
            return f"read_blocked:{code}"
        return "" if pathlib.Path(binding.target_path) == path else "read_target_differs"

    def admit(root: Optional[str], path: pathlib.Path) -> str:
        verdict = decide(root, path)
        if (verdict and root is None and serving is not None and path != serving and _under(path, serving)
                and not _under(path, system_repo)):
            verdict = decide("system_repo", path, serving)
        return verdict
    return admit

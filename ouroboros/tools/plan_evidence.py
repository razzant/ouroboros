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
from hashlib import sha256
import pathlib
import stat
from typing import Any, Callable, Iterable, Mapping, Optional

from ouroboros.tools.plan_spec import (
    _TASK_LOCATOR_PREFIX,
    _canonical_hash,
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
    """``missing`` · ``directory`` · ``file`` (regular) · ``unreadable`` (stat failure or a
    non-regular node: fifo/device/socket would block or never end a read)."""
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return "missing"
    except (OSError, ValueError, RuntimeError):
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
        # The filename policy cannot see CONTENT: an allowed notes.md may quote an API key,
        # and attached text goes verbatim to external reviewers. Redact with the shared
        # prompt-secret redactor; sha256/bytes keep describing the ORIGINAL source and the
        # redaction is disclosed on the row.
        from ouroboros.tools.review_helpers import redact_prompt_secrets

        payload["text"], redacted = redact_prompt_secrets(payload["text"])
        head_bytes = len(payload["text"].encode("utf-8"))
        if head_bytes > remaining:
            omissions.append({"locator": locator, "reason": "budget_exhausted"})
            return
        remaining -= head_bytes
        row = {
            "locator": locator, "kind": kind, "sha256": payload["sha256"],
            "bytes": payload["bytes"], "attached_bytes": head_bytes, "text": payload["text"],
        }
        if redacted:
            row["secrets_redacted"] = True
        attached.append(row)
        if payload["bytes"] > per_item:
            omissions.append({"locator": locator, "reason": f"truncated_to_{per_item}"})

    for raw in locators or []:
        locator = str(raw or "").strip()
        if not locator:
            continue
        if locator in declared:
            omissions.append({"locator": locator, "reason": "duplicate_locator"})
            continue
        declared.append(locator)
        if _is_url(locator):
            omissions.append({"locator": locator, "reason": "url_not_fetched"})
            continue
        if locator.startswith(_TASK_LOCATOR_PREFIX):
            if resolve_task is None:
                omissions.append({"locator": locator, "reason": "unsupported_locator"})
                continue
            text = resolve_task(locator[len(_TASK_LOCATOR_PREFIX):].strip())
            payload, reason = (None, "task_not_found") if text is None else _payload_from_text(str(text), per_item)
            if payload is None:
                omissions.append({"locator": locator, "reason": reason})
            else:
                attach(locator, "task", payload)
            continue
        path, reason = _resolve_locator_path(locator, active)
        if path is None:
            omissions.append({"locator": locator, "reason": reason})
            continue
        if any(path == deny or _under(path, deny) for deny in denied):
            omissions.append({"locator": locator, "reason": "denied_path"})
            continue
        if not any(_under(path, root) for root in roots):
            omissions.append({"locator": locator, "reason": "outside_allowed_roots"})
            continue
        if _sensitive(path) or _sensitive(pathlib.PurePosixPath(locator.replace("\\", "/"))):
            omissions.append({"locator": locator, "reason": "sensitive"})  # lexical AND resolved name
            continue
        kind = _path_kind(path)
        if kind != "file":
            omissions.append({"locator": locator, "reason": kind})
            continue
        payload, reason = _read_evidence(path, read_text, per_item, hash_bytes_limit)
        if payload is None:
            omissions.append({"locator": locator, "reason": reason or "unreadable"})
            continue
        attach(locator, "path", payload)
    return {
        "declared": declared, "attached": attached, "omissions": omissions,
        "bounds": {"per_item_bytes": per_item, "total_bytes": max(0, int(total_bytes))},
    }


def evidence_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Identity of the evidence set for the plan fingerprint (F4): declared locators,
    attached (locator, sha256, bytes), omission (locator, reason) and the locators
    reviewers requested (W3: a request for something already declared still changes
    what the next wave is about) — NEVER the text."""
    return _canonical_hash({
        "declared": list(manifest.get("declared") or []),
        "reviewer_requested": list(manifest.get("reviewer_requested") or []),
        "attached": [
            {"locator": a.get("locator"), "sha256": a.get("sha256"), "bytes": a.get("bytes")}
            for a in manifest.get("attached") or []
        ],
        "omissions": [
            {"locator": o.get("locator"), "reason": o.get("reason")}
            for o in manifest.get("omissions") or []
        ],
    })

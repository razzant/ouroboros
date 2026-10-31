"""Reviewable skill payload: what a reviewer may see, read from one frozen copy.

Owns the payload's binary policy (loadable native code is refused by its magic bytes, never by
its filename; other binaries become ``{path,size,mime_from_name,sha256}`` descriptors), the text
read that refuses unreadable or non-UTF-8 runtime payloads, the per-file index of the frozen copy a
seat reads (``frozen_payload_files``), the one pasted rendering of it (``payload_pack``: never split
into chunks — a packet seat that cannot hold it receives the index and a plain "not delivered"
statement, ``tools/review_skill``), and the retained read-only copy bound to the loader hash
(``frozen_payload_source``) that every seat on either side of an edit reads.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

# Payload-file binary policy (X4/В21). Loadable native code is unreviewable by
# LLMs: files whose CONTENT starts with a loader magic number are hard-blocked
# from the skill payload surface — judged by magic bytes, never by filename.
# These prefixes are unambiguous (never legitimate text), so they block
# regardless of UTF-8 validity. WebAssembly (``WASM_MAGIC``) is deliberately NOT
# here: the host never loads it natively — it runs only inside the browser's
# sandboxed widget frame — so a module is admitted as a content-hash-bound
# descriptor the reviewer sees, routed there by its magic even when its bytes
# happen to decode as UTF-8 (the canonical 8-byte module does).
WASM_MAGIC = b"\x00asm"
_EXECUTABLE_MAGICS: Tuple[Tuple[bytes, str], ...] = (
    (b"\x7fELF", "ELF executable / shared object"),
    (b"\xfe\xed\xfa\xce", "Mach-O executable (32-bit)"),
    (b"\xfe\xed\xfa\xcf", "Mach-O executable (64-bit)"),
    (b"\xce\xfa\xed\xfe", "Mach-O executable (32-bit, little-endian)"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable (64-bit, little-endian)"),
    (b"\xca\xfe\xba\xbe", "Mach-O universal (fat) binary / Java class"),
    (b"\xca\xfe\xba\xbf", "Mach-O universal (fat, 64-bit) binary"),
    (b"\xbe\xba\xfe\xca", "Mach-O universal (fat, byte-swapped) binary"),
    (b"\xbf\xba\xfe\xca", "Mach-O universal (fat, 64-bit byte-swapped) binary"),
)

# Lexical download filter retained ONLY for the marketplace fetcher's coarse pre-gate
# (ouroboros/marketplace/fetcher.py). Skill REVIEW itself judges file CONTENT — loader
# magic bytes, ``executable_magic_kind`` — never filenames (X4/В21): a renamed ELF is
# still blocked; a text file with a scary extension stays reviewable.
_LOADABLE_BINARY_EXTENSIONS = frozenset(
    {".so", ".dylib", ".dll", ".pyc", ".pyo", ".node", ".exe", ".bin"}
)


def executable_magic_kind(data: bytes, *, is_utf8_text: bool) -> str:
    """Return the loadable-executable kind for ``data``'s magic bytes, or ``""``.

    PE (``MZ``, 2 bytes) and other-version .pyc magics are too short /
    version-varying to be unambiguous on text, so they are judged only on
    content that is NOT valid UTF-8 — a genuine PE or .pyc body never is."""
    for magic, kind in _EXECUTABLE_MAGICS:
        if data.startswith(magic):
            return kind
    import importlib.util

    if data.startswith(importlib.util.MAGIC_NUMBER):
        # The current interpreter's .pyc magic (the .pyc the host could actually
        # import) is 4 exact bytes — unambiguous, so judged on any content.
        return "compiled Python bytecode (.pyc)"
    if not is_utf8_text:
        if data[:2] == b"MZ":
            return "PE/DOS executable"
        # Deliberately NO looser .pyc heuristic here: "bytes 2-3 == CRLF" also
        # matches legacy-encoded text files (false hard-block). A .pyc compiled
        # by a DIFFERENT Python version cannot be imported by this host at all,
        # so it falls to the descriptor path — disclosed residual, not a gap.
    return ""


class SkillBinaryPayload(RuntimeError):
    """Raised for loadable-executable skill payloads (judged by content magic
    bytes); other binary payloads — non-UTF-8 files and WebAssembly admitted by
    magic — become ``{path,size,mime_from_name,sha256}`` descriptors."""

    def __init__(self, relpath: str, size_bytes: int, kind: str = "") -> None:
        super().__init__(
            f"Skill file {relpath!r} is a loadable executable ({kind or 'magic bytes'}, "
            f"{size_bytes} bytes); review hard-blocks native code in the skill surface."
        )
        self.relpath = relpath
        self.size_bytes = size_bytes
        self.kind = kind


_SkillBinaryPayload = SkillBinaryPayload  # the historical spelling of the typed refusal


class _SkillFileUnreadable(RuntimeError):
    """Raised when a runtime-reachable file cannot be read; review fails closed."""

    def __init__(self, relpath: str, err: BaseException) -> None:
        super().__init__(
            f"Skill file {relpath!r} unreadable: {type(err).__name__}: {err}"
        )
        self.relpath = relpath
        self.err = err


def binary_file_descriptor(relpath: str, data: bytes, *, filename: str = "") -> Dict[str, Any]:
    """Typed descriptor for a binary (non-UTF-8 or WebAssembly), non-executable payload file: the review
    pack carries ``{path,size,mime_from_name,sha256}`` instead of raw bytes."""
    import mimetypes

    return {
        "path": relpath,
        "size": len(data),
        # Named honestly: guessed from the FILENAME, not sniffed from bytes — a
        # hostile blob named cert.png would present as image/png; the reviewer
        # must not read this as a content attestation (size/sha256 do not lie).
        "mime_from_name": mimetypes.guess_type(filename or relpath)[0] or "application/octet-stream",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _read_skill_file(
    path: pathlib.Path, *, relpath: str = ""
) -> tuple[Optional[str], bytes, Optional[Dict[str, Any]]]:
    """Read one skill file: ``(text, sha256_digest, descriptor)`` — exactly one set.
    Native executable magic blocks ordinary review; Cyber retains these bytes
    through a descriptor. Binary formats never become text merely because they
    decode as UTF-8. Unreadable source remains an actual read failure."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        # Fail closed; placeholders would let review pass over missing payload.
        raise _SkillFileUnreadable(relpath or path.name, exc) from exc
    rel = relpath or path.name
    try:
        text: Optional[str] = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    kind = executable_magic_kind(data, is_utf8_text=text is not None)
    if kind:
        from ouroboros.config import get_runtime_mode
        from ouroboros.runtime_mode_policy import runtime_mode_at_least

        if not runtime_mode_at_least(get_runtime_mode(), "cyber_pro"):
            raise SkillBinaryPayload(rel, len(data), kind)
    digest = hashlib.sha256(data).digest()
    if text is not None and not kind and not data.startswith(WASM_MAGIC):
        return text, digest, None
    descriptor = binary_file_descriptor(rel, data, filename=path.name)
    if kind:
        descriptor["format_from_magic"] = kind
    return None, digest, descriptor


def frozen_payload_files(
    view: pathlib.Path,
    *,
    manifest_entry: str = "",
    manifest_scripts: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Per-file facts of the frozen payload copy a reviewer reads: the loader's own file set
    (``_iter_payload_files``) read by ``_read_skill_file``; a text file carries the identity a
    ``read_file`` receipt stamps (``source_text_identity``) and its text, a binary file stays its
    descriptor — a descriptor never claims its bytes were inspected."""
    from ouroboros.skill_loader import _iter_payload_files  # pylint: disable=W0212
    from ouroboros.tools.scope_required_sources import source_text_identity

    view = pathlib.Path(view).resolve()
    entries: List[Dict[str, Any]] = []
    for file_path in _iter_payload_files(view, manifest_entry=manifest_entry, manifest_scripts=manifest_scripts):
        rel = file_path.relative_to(view).as_posix()
        text, digest, descriptor = _read_skill_file(file_path, relpath=rel)
        if descriptor is not None:
            entries.append({"path": rel, "binary": descriptor})
            continue
        raw = text.encode("utf-8")
        entries.append({"path": rel, "size": len(raw), "sha256": digest.hex(), "text": text,
                        **source_text_identity(raw)})
    return entries


def payload_pack(entries: List[Dict[str, Any]]) -> str:
    """The one pasted rendering of the frozen payload: every text file in a fenced block, a binary file as
    its descriptor. Never truncated and never split: a seat that cannot hold it is told so and reads the files
    (a retrieving seat) or judges the index (a packet seat)."""
    if not entries:
        return "(empty skill directory — no manifest, no payload)"
    blocks = []
    for entry in entries:
        rel = str(entry["path"])
        if entry.get("binary"):
            body = json.dumps(entry["binary"], indent=2, sort_keys=True)
            blocks.append(f"### {rel} (binary file — descriptor only, content not inlined)\n\n```json\n{body}\n```")
        else:
            blocks.append(f"### {rel}\n\n```\n{entry['text']}\n```")
    return "\n\n".join(blocks)


def _loader_inventory(skill: Any) -> Dict[str, Any]:
    return {"manifest_entry": skill.manifest.entry, "manifest_scripts": skill.manifest.scripts}


def frozen_payload_source(drive_root: Any, skill: Any, content_hash: str) -> Dict[str, Any]:
    """The wave's read-only frozen payload: the loader inventory kept at loader hash
    ``content_hash`` — the files that hash covers, without the dependency env and
    caches both leave out — named by the skill folder and that hash.

    A native episode reads it in place and a read-only session by absolute path, so
    seats sent on either side of an edit of the live skill_dir review the same bytes;
    a writable seat checks its own copy of the whole folder (``skill_checking_source``).
    A payload that moved past the reviewed hash before its view was kept yields
    ``{"unavailable": ...}``.
    """
    from functools import partial

    from ouroboros.review_checking import bound_directory_source
    from ouroboros.skill_loader import compute_content_hash
    from ouroboros.subagent_worktrees import _copy_payload_inventory

    inventory = _loader_inventory(skill)
    return bound_directory_source(skill.skill_dir, data_root=drive_root, expected_hash=content_hash,
                                  content_hash_of=lambda root: compute_content_hash(root, **inventory),
                                  copy=partial(_copy_payload_inventory, preserve_links=False, **inventory))


def skill_checking_source(drive_root: Any, skill: Any, row_plan: Dict[str, Any],
                          content_hash: str) -> Dict[str, Any] | None:
    """A writable seat checks a copy of the whole skill folder — its dependencies, links,
    modes and empty folders — whose loader hash is ``content_hash``; None when every seat
    keeps the read-only inspection boundary."""
    from types import SimpleNamespace

    from ouroboros.review_checking import bound_directory_source, writable_seat
    from ouroboros.skill_loader import compute_content_hash

    routes = list(row_plan.get("routes") or [])
    accesses = row_plan.get("session_accesses") or [""] * len(routes)
    if not any(writable_seat(SimpleNamespace(route=route, session_access=access))
               for route, access in zip(routes, accesses)):
        return None
    return bound_directory_source(skill.skill_dir, data_root=drive_root, expected_hash=content_hash,
                                  content_hash_of=lambda root: compute_content_hash(root, **_loader_inventory(skill)))


__all__ = [
    "WASM_MAGIC", "SkillBinaryPayload", "binary_file_descriptor", "executable_magic_kind", "frozen_payload_files",
    "frozen_payload_source", "payload_pack", "skill_checking_source",
]

"""Reviewable skill payload: what a reviewer may see, and how much of it.

Owns the assembly of the skill file pack the reviewer reads: the pack-level
token budget derived from the review stack's prompt-budget SSOT, the text read
that refuses unreadable or non-UTF-8 runtime payloads, the binary extensions
named early on the refusal path, and the split of an over-budget skill into
budget-sized packs reviewed in separate passes. The three typed refusals live
here with the reads that raise them, so a caller can distinguish a
shrink-the-file case from an opaque-payload case without parsing a message.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.skill_review_passes import (
    WASM_MAGIC,
    SkillBinaryPayload as _SkillBinaryPayload,
    binary_file_descriptor,
    executable_magic_kind,
)
from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET
from ouroboros.utils import estimate_tokens


# The reviewable skill payload is bound by ONE pack-level token budget (the review
# stack's SSOT REVIEW_PROMPT_TOKEN_BUDGET), not per-file / file-count BYTE caps: a 76 KB
# data file or a 41-file skill is fully reviewable when the whole pack fits a 1M-context
# reviewer. Loadable executables / unreadable files are still refused (safety, not size).
# Headroom reserves the rest of the reviewer prompt (governance docs + checklist + framing).

_SKILL_PACK_TOKEN_HEADROOM = 120_000


def _skill_pack_token_budget() -> int:
    """Estimated-token budget for the assembled skill file pack alone (SSOT
    REVIEW_PROMPT_TOKEN_BUDGET minus headroom for the rest of the reviewer prompt)."""
    return max(1, REVIEW_PROMPT_TOKEN_BUDGET - _SKILL_PACK_TOKEN_HEADROOM)


# Lexical download filter retained ONLY for the marketplace fetcher's coarse pre-gate
# (ouroboros/marketplace/fetcher.py). Skill REVIEW itself judges file CONTENT — loader
# magic bytes, see ``skill_review_passes.executable_magic_kind`` — never filenames
# (X4/В21): a renamed ELF is still blocked; a text file with a scary extension stays reviewable.
_LOADABLE_BINARY_EXTENSIONS = frozenset(
    {".so", ".dylib", ".dll", ".pyc", ".pyo", ".node", ".exe", ".bin"}
)


class _SkillFileOverBudget(RuntimeError):
    """Raised when a SINGLE skill file alone exceeds the reviewer token budget, so it
    cannot be placed in any budget-sized review pack without truncating it (which
    review refuses). Honest-pending: the maintainer must shrink/split that one file.

    The whole-skill over-budget case is NOT an error — it is split into multiple
    budget-sized packs and reviewed in separate passes (see ``_build_skill_file_packs``
    and ``_run_chunked_skill_review``)."""

    def __init__(self, relpath: str, tokens: int, budget: int) -> None:
        super().__init__(
            f"Skill file {relpath!r} alone is ~{tokens} tokens > {budget} reviewer budget."
        )
        self.relpath = relpath
        self.tokens = tokens
        self.budget = budget


class _SkillFileUnreadable(RuntimeError):
    """Raised when a runtime-reachable file cannot be read; review fails closed."""

    def __init__(self, relpath: str, err: BaseException) -> None:
        super().__init__(
            f"Skill file {relpath!r} unreadable: {type(err).__name__}: {err}"
        )
        self.relpath = relpath
        self.err = err


def _read_skill_file(
    path: pathlib.Path, *, relpath: str = ""
) -> tuple[Optional[str], bytes, Optional[Dict[str, Any]]]:
    """Read one skill file: ``(text, sha256_digest, descriptor)`` — exactly one set.
    Loadable executables (CONTENT magic bytes, never filename) hard-block review;
    WebAssembly (``WASM_MAGIC``, even when its bytes decode as UTF-8) and other
    non-UTF-8 files yield a typed descriptor instead of raw bytes."""
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
        raise _SkillBinaryPayload(rel, len(data), kind)
    digest = hashlib.sha256(data).digest()
    if text is not None and not data.startswith(WASM_MAGIC):
        return text, digest, None
    return None, digest, binary_file_descriptor(rel, data, filename=path.name)


def _build_skill_file_packs(
    skill_dir: pathlib.Path,
    *,
    manifest_entry: str = "",
    manifest_scripts: Optional[List[Dict[str, Any]]] = None,
    expected_content_hash: str = "",
) -> List[str]:
    """Return the fenced-code review pack(s) mirroring the skill content-hash surface.

    Normally ONE pack. When the whole pack would exceed the reviewer token budget,
    the files are split into multiple budget-sized packs (greedy by file) so each is
    reviewed in a SEPARATE pass and EVERY byte is still reviewed — never silently
    truncated. A single file that alone exceeds the budget cannot be split without
    truncating it, so it raises ``_SkillFileOverBudget`` (honest-pending).

    The bound is ONE pack-level token budget, not per-file BYTE caps. Loadable
    executables (content magic bytes) / unreadable files are still refused by
    ``_read_skill_file``; other non-UTF-8 files enter the pack as descriptors."""
    from ouroboros.skill_loader import _iter_payload_files, reduce_skill_content_hash  # pylint: disable=W0212

    skill_dir = skill_dir.resolve()
    files = _iter_payload_files(
        skill_dir,
        manifest_entry=manifest_entry,
        manifest_scripts=manifest_scripts,
    )
    if not files:
        if expected_content_hash and reduce_skill_content_hash([]) != expected_content_hash:
            raise _SkillFileUnreadable("(payload snapshot)", RuntimeError("skill payload changed after hashing"))
        return ["(empty skill directory — no manifest, no payload)"]

    budget = _skill_pack_token_budget()
    packs: List[str] = []
    current: List[str] = []
    current_tokens = 0
    file_digests: List[tuple[str, bytes]] = []
    for file_path in files:
        rel = file_path.relative_to(skill_dir).as_posix()
        body, file_digest, descriptor = _read_skill_file(file_path, relpath=rel)
        file_digests.append((rel, file_digest))
        if descriptor is not None:  # typed descriptor, never raw non-UTF-8 bytes
            body = json.dumps(descriptor, indent=2, sort_keys=True)
            rel_head = f"{rel} (binary file — descriptor only, content not inlined)"
            block = f"### {rel_head}\n\n```json\n{body}\n```"
        else:
            block = f"### {rel}\n\n```\n{body}\n```"
        block_tokens = estimate_tokens(block)
        if block_tokens > budget:
            # One file too large to review in a single pass without truncating it.
            raise _SkillFileOverBudget(rel, block_tokens, budget)
        if current and current_tokens + block_tokens > budget:
            packs.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        packs.append("\n\n".join(current))
    if expected_content_hash and reduce_skill_content_hash(file_digests) != expected_content_hash:
        raise _SkillFileUnreadable("(payload snapshot)", RuntimeError("skill payload changed after hashing"))
    return packs

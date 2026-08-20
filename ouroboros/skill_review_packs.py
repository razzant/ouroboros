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

import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET
from ouroboros.utils import estimate_tokens


# The reviewable skill payload is bound by ONE pack-level token budget (reusing the
# review stack's SSOT REVIEW_PROMPT_TOKEN_BUDGET) instead of arbitrary per-file /
# file-count BYTE caps: a 76 KB data file or a 41-file skill is fully reviewable when
# the whole pack fits a 1M-context reviewer. Binary / unreadable files are still
# refused (those are safety, not size). Headroom reserves the rest of the reviewer
# prompt (governance docs + checklist + framing) so the SKILL pack alone is bounded.
_SKILL_PACK_TOKEN_HEADROOM = 120_000

def _skill_pack_token_budget() -> int:
    """Estimated-token budget for the assembled skill file pack alone (SSOT
    REVIEW_PROMPT_TOKEN_BUDGET minus headroom for the rest of the reviewer prompt)."""
    return max(1, REVIEW_PROMPT_TOKEN_BUDGET - _SKILL_PACK_TOKEN_HEADROOM)

# Loadable native code is unreviewable by LLMs. All non-UTF-8 runtime-reachable
# files are blocked; this set names common categories early in the error path.
_LOADABLE_BINARY_EXTENSIONS = frozenset(
    {
        ".so", ".dylib", ".dll",          # native shared libs
        ".pyc", ".pyo",                    # precompiled Python
        ".node",                           # Node.js native addons
        ".wasm",                           # WebAssembly (loadable by node/python)
        ".exe", ".bin",                    # generic executables
    }
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


class _SkillBinaryPayload(RuntimeError):
    """Raised for non-UTF-8 runtime payloads that reviewers cannot inspect."""

    def __init__(self, relpath: str, size_bytes: int) -> None:
        super().__init__(
            f"Skill file {relpath!r} is binary ({size_bytes} bytes); "
            "review refuses opaque payloads in the executable surface."
        )
        self.relpath = relpath
        self.size_bytes = size_bytes

def _read_skill_text(path: pathlib.Path, *, relpath: str = "") -> str:
    """Read a text skill file; refuse unreadable or binary payloads. The reviewable
    SIZE is bound ONCE at the pack level (see ``_build_skill_file_packs``), not by an
    arbitrary per-file byte cap, so a large legitimate text/data file is reviewable."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        # Fail closed; placeholders would let review pass over missing payload.
        raise _SkillFileUnreadable(relpath or path.name, exc) from exc
    lowered = path.name.lower()
    if any(lowered.endswith(ext) for ext in _LOADABLE_BINARY_EXTENSIONS):
        raise _SkillBinaryPayload(relpath or path.name, len(data))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Any non-UTF-8 runtime-reachable file blocks review.
        raise _SkillBinaryPayload(relpath or path.name, len(data)) from exc


def _build_skill_file_packs(
    skill_dir: pathlib.Path,
    *,
    manifest_entry: str = "",
    manifest_scripts: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Return the fenced-code review pack(s) mirroring the skill content-hash surface.

    Normally ONE pack. When the whole pack would exceed the reviewer token budget,
    the files are split into multiple budget-sized packs (greedy by file) so each is
    reviewed in a SEPARATE pass and EVERY byte is still reviewed — never silently
    truncated. A single file that alone exceeds the budget cannot be split without
    truncating it, so it raises ``_SkillFileOverBudget`` (honest-pending).

    The bound is ONE pack-level token budget, not arbitrary per-file/file-count BYTE
    caps. Binary / unreadable files are still refused by ``_read_skill_text`` (those
    are safety, not size)."""
    from ouroboros.skill_loader import _iter_payload_files  # pylint: disable=W0212

    skill_dir = skill_dir.resolve()
    files = _iter_payload_files(
        skill_dir,
        manifest_entry=manifest_entry,
        manifest_scripts=manifest_scripts,
    )
    if not files:
        return ["(empty skill directory — no manifest, no payload)"]

    budget = _skill_pack_token_budget()
    packs: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for file_path in files:
        rel = file_path.relative_to(skill_dir).as_posix()
        body = _read_skill_text(file_path, relpath=rel)
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
    return packs

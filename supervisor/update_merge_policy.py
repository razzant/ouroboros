"""Per-path policy for the managed-update merge (P2).

A small DECLARATIVE table on top of the conflict list git reports. It answers
"may the official updater auto-resolve THIS conflict, and how?" — a DIFFERENT
question from ``runtime_mode_policy`` ("may Ouroboros WRITE this path?"), so it
lives here next to ``git_ops``, not there.

Docs are the special class: Ouroboros rewrites its own README/ARCHITECTURE on
self-modification, so those conflict on essentially every update. A doc-only
conflict is Ouroboros-reconciled (a prose merge) and must NOT downgrade an
otherwise-clean code update to "manual". Constitutional / review-policy docs
(BIBLE, CHECKLISTS, SAFETY) are EXCLUDED — they stay visible/manual, never
silently auto-resolved. Code conflicts are always real → assisted/manual; the
updater never auto-picks a side of code.
"""

from __future__ import annotations

import posixpath
from typing import Dict, List

# Release docs whose conflicts may be auto-reconciled (prose). Exact names, the
# CHANGELOG family, and any markdown under docs/. Kept declarative on purpose.
AUTO_RECONCILE_DOC_EXACT = frozenset({"README.md"})
AUTO_RECONCILE_DOC_PREFIXES = ("docs/",)

# Constitutional / review-policy docs: NEVER auto-resolved; surfaced for review.
PROTECTED_DOC_PATHS = frozenset(
    {
        "BIBLE.md",
        "docs/CHECKLISTS.md",
        "prompts/SAFETY.md",
    }
)

# Conflict-prone core files — a UI LABEL only (a sharper "code conflict" hint in
# the preflight table), NOT a policy gate: ANY non-doc conflict is already
# "conflicting" regardless of whether it is in this set.
HOT_CODE_PATHS = frozenset(
    {
        "ouroboros/loop.py",
        "ouroboros/tools/control.py",
        "ouroboros/tools/registry.py",
        "ouroboros/config.py",
        "supervisor/queue.py",
        "supervisor/events.py",
    }
)


def _norm(path: str) -> str:
    """Repo-relative POSIX path, no leading ``./``. Pure string normalization."""
    normalized = posixpath.normpath(str(path or "").replace("\\", "/"))
    return normalized[2:] if normalized.startswith("./") else normalized.lstrip("/")


def is_protected_doc(path: str) -> bool:
    """A constitutional / review-policy doc that must never be auto-resolved."""
    return _norm(path) in PROTECTED_DOC_PATHS


def is_auto_reconcile_doc(path: str) -> bool:
    """A release doc whose conflict may be Ouroboros-reconciled (never a protected doc)."""
    p = _norm(path)
    if p in PROTECTED_DOC_PATHS:
        return False
    if p in AUTO_RECONCILE_DOC_EXACT:
        return True
    if posixpath.basename(p).upper().startswith("CHANGELOG"):
        return True
    return p.endswith(".md") and any(p.startswith(prefix) for prefix in AUTO_RECONCILE_DOC_PREFIXES)


def is_hot_code(path: str) -> bool:
    """Whether a path is a known conflict-prone core file (UI label only)."""
    return _norm(path) in HOT_CODE_PATHS


def classify_conflicts(conflict_paths: List[str]) -> Dict[str, object]:
    """Classify a merge's unmerged paths into the update's resolution policy.

    Returns ``kind``:
      - ``"clean"``:         no unmerged paths.
      - ``"doc_reconcile"``: every unmerged path is an auto-reconcilable release doc.
      - ``"conflicting"``:   at least one code path OR protected doc is unmerged.
    plus the split lists (``doc``/``code``/``protected``) and the hot-code label set.
    """
    paths = [str(p).strip() for p in (conflict_paths or []) if str(p).strip()]
    doc_paths: List[str] = []
    code_paths: List[str] = []
    protected_paths: List[str] = []
    for path in paths:
        if is_protected_doc(path):
            protected_paths.append(path)
        elif is_auto_reconcile_doc(path):
            doc_paths.append(path)
        else:
            code_paths.append(path)
    if not paths:
        kind = "clean"
    elif code_paths or protected_paths:
        kind = "conflicting"
    else:
        kind = "doc_reconcile"
    return {
        "kind": kind,
        "doc_conflict_paths": doc_paths,
        "code_conflict_paths": code_paths,
        "protected_conflict_paths": protected_paths,
        "hot_code_paths": [p for p in code_paths if is_hot_code(p)],
    }

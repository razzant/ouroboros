"""Runtime-mode policy for protected Ouroboros source surfaces.

``advanced`` is allowed to evolve the application layer, but must not casually
rewrite the core contracts, safety files, or release/managed-repo invariants.
``pro`` may touch those paths, but commits still flow through the normal
triad + scope review gate.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable


SAFETY_CRITICAL_PATHS = frozenset({
    "BIBLE.md",
    "ouroboros/safety.py",
    "ouroboros/runtime_mode_policy.py",
    "ouroboros/tools/extension_dispatch.py",
    "ouroboros/tools/registry.py",
    "prompts/SAFETY.md",
})

FROZEN_CONTRACT_PATH_PREFIXES = (
    "ouroboros/contracts/",
)

FROZEN_CONTRACT_PATHS = frozenset({
    "tests/test_contracts.py",
    "docs/CHECKLISTS.md",
    "ouroboros/gateway/contracts.py",
})

RELEASE_INVARIANT_PATHS = frozenset({
    ".github/workflows/ci.yml",
    "Ouroboros.spec",
    "build.sh",
    "build_linux.sh",
    "build_windows.ps1",
    "scripts/build_repo_bundle.py",
    "ouroboros/launcher_bootstrap.py",
    "ouroboros/repo_remotes.py",
    "supervisor/git_ops.py",
    "supervisor/update_merge.py",
    "supervisor/update_merge_policy.py",
})

PROTECTED_RUNTIME_PATH_PREFIXES = FROZEN_CONTRACT_PATH_PREFIXES
PROTECTED_RUNTIME_PATHS = (
    SAFETY_CRITICAL_PATHS
    | FROZEN_CONTRACT_PATHS
    | RELEASE_INVARIANT_PATHS
)

# Case-insensitive lookup tables. On case-insensitive filesystems (macOS HFS+
# default, Windows NTFS), `write_file(path="bible.md", ...)` writes to BIBLE.md
# but the literal string "bible.md" doesn't match SAFETY_CRITICAL_PATHS' uppercase
# entry, bypassing the safety guard. Matching the lowercased form via these
# frozensets closes the bypass.
_SAFETY_CRITICAL_LOWER = frozenset(p.lower() for p in SAFETY_CRITICAL_PATHS)
_FROZEN_CONTRACT_LOWER = frozenset(p.lower() for p in FROZEN_CONTRACT_PATHS)
_FROZEN_CONTRACT_PREFIXES_LOWER = tuple(p.lower() for p in FROZEN_CONTRACT_PATH_PREFIXES)
_RELEASE_INVARIANT_LOWER = frozenset(p.lower() for p in RELEASE_INVARIANT_PATHS)


@dataclass(frozen=True)
class ProtectedPath:
    path: str
    category: str


def normalize_repo_path(path: str) -> str:
    """Normalize a repo-relative path to forward-slash POSIX form."""
    cleaned = str(path or "").strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return pathlib.PurePosixPath(cleaned).as_posix()


def protected_path_category(path: str) -> str:
    """Return the protected-surface category for *path*, or ``""``.

    Lookup is case-insensitive. On case-insensitive filesystems (macOS
    HFS+ default, Windows NTFS), `write_file(path="bible.md", ...)` writes to
    BIBLE.md but the literal lowercase string would bypass the strict
    uppercase membership check. Compare lowercased forms to close the
    bypass.
    """
    norm = normalize_repo_path(path)
    if not norm or norm == ".":
        return ""
    norm_lower = norm.lower()
    if norm in SAFETY_CRITICAL_PATHS or norm_lower in _SAFETY_CRITICAL_LOWER:
        return "safety-critical"
    if (
        norm in FROZEN_CONTRACT_PATHS
        or norm_lower in _FROZEN_CONTRACT_LOWER
        or any(norm.startswith(prefix) for prefix in FROZEN_CONTRACT_PATH_PREFIXES)
        or any(norm_lower.startswith(prefix) for prefix in _FROZEN_CONTRACT_PREFIXES_LOWER)
    ):
        return "frozen-contract"
    if norm in RELEASE_INVARIANT_PATHS or norm_lower in _RELEASE_INVARIANT_LOWER:
        return "release-invariant"
    return ""


def is_protected_runtime_path(path: str) -> bool:
    return bool(protected_path_category(path))


def protected_paths_in(paths: Iterable[str]) -> list[ProtectedPath]:
    found: list[ProtectedPath] = []
    seen: set[str] = set()
    for path in paths:
        norm = normalize_repo_path(path)
        if norm in seen:
            continue
        category = protected_path_category(norm)
        if category:
            found.append(ProtectedPath(path=norm, category=category))
            seen.add(norm)
    return found


def mode_allows_protected_write(runtime_mode: str) -> bool:
    return str(runtime_mode or "").strip().lower() == "pro"


def format_protected_paths(paths: Iterable[ProtectedPath | str]) -> str:
    rendered: list[str] = []
    for item in paths:
        if isinstance(item, ProtectedPath):
            rendered.append(f"{item.path} ({item.category})")
        else:
            category = protected_path_category(str(item))
            rendered.append(
                f"{normalize_repo_path(str(item))} ({category})"
                if category else normalize_repo_path(str(item))
            )
    return ", ".join(rendered)


def protected_write_block_message(
    *,
    path: str,
    runtime_mode: str,
    action: str,
) -> str:
    norm = normalize_repo_path(path)
    category = protected_path_category(norm)
    return (
        f"⚠️ CORE_PROTECTION_BLOCKED: runtime_mode={runtime_mode!r} refuses "
        f"to {action} protected {category or 'core'} path: {norm}. "
        "Switch to runtime_mode='pro' and let the normal triad + scope review "
        "cover the protected core/contract/release change before commit."
    )


def core_patch_notice(paths: Iterable[ProtectedPath | str]) -> str:
    return (
        "⚠️ CORE_PATCH_NOTICE: runtime_mode='pro' is editing protected "
        "Ouroboros core/contract/release surface(s): "
        f"{format_protected_paths(paths)}. These changes can be committed only "
        "through the normal triad + scope review pipeline."
    )

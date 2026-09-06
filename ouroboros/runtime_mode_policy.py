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
    # The v7 D04 split moved guard/resolution bodies out of the protected
    # registry without moving any of the risk, so every inventory that
    # protects the parent must cover the leaves (label parity — same rule as
    # the git_ops family).
    "ouroboros/tools/registry_guard_process.py",
    "ouroboros/tools/registry_guards.py",
    # F3.1 typed-organ leaves: the registry class body and the typed result
    # vocabulary re-homed out of the protected registry — same label parity.
    "ouroboros/tools/registry_core.py",
    "ouroboros/tools/tool_catalog.py",
    "ouroboros/tools/tool_context.py",
    "ouroboros/tools/tool_resolution.py",
    "ouroboros/tools/tool_result.py",
    "prompts/SAFETY.md",
})

FROZEN_CONTRACT_PATH_PREFIXES = (
    "ouroboros/contracts/",
)

FROZEN_CONTRACT_PATHS = frozenset({
    "tests/test_contracts.py",
    "docs/CHECKLISTS.md",
    # The standing-disclosure archive is the same binding reviewer contract as
    # its parent — extracted for pack size, not demoted (#447 stage 3).
    "docs/CHECKLISTS_ARCHIVE.md",
    "ouroboros/gateway/contracts.py",
    "ouroboros/size_ratchet_manifest.py",
})

# The whole git_ops family as ONE derived constant (owner decision, batch 5
# item 15): the facade plus its G1 leaves. Every protection inventory that
# covers the parent consumes this set instead of hand-listing the members.
GIT_OPS_FAMILY_PATHS = frozenset(
    {"supervisor/git_ops.py"}
    | {f"supervisor/git_ops_{leaf}.py" for leaf in ("remotes", "rescue", "reset", "updates")}
)

RELEASE_INVARIANT_PATHS = frozenset({
    ".github/workflows/ci.yml",
    "Ouroboros.spec",
    "build.sh",
    "build_linux.sh",
    "build_windows.ps1",
    "scripts/build_repo_bundle.py",
    "ouroboros/launcher_bootstrap.py",
    "ouroboros/repo_remotes.py",
    # The v7 G1 split moved the remote/managed-update/checkout-reset/rescue
    # bodies out of the protected git_ops facade without moving any of the
    # risk, so every inventory that protects the parent must cover the leaves
    # (label parity — same rule as the D04 registry block above). The family
    # is one derived constant so a future leaf cannot be forgotten here while
    # existing elsewhere; a glob completeness test pins list-vs-tree parity.
    *GIT_OPS_FAMILY_PATHS,
    "supervisor/update_merge.py",
    "supervisor/update_merge_policy.py",
    # The F2.4 update-engine re-split moved the planner/materializer bodies —
    # the carrier engine's three insertion points — out of the protected
    # update_merge facade, and the D34 span resolver rewrites worktree files
    # under the update lock; every inventory that protects the parent must
    # cover them (label parity — same rule as the G1 block above).
    "supervisor/update_merge_plan.py",
    "supervisor/update_carriers.py",
    # Upstream's own redesign split the candidate/carrier primitives (stash
    # restore, failed-update preservation, tests-evidence proof) out of the
    # protected update_merge facade without listing the leaf — an upstream gap
    # the F2.4 lane disclosed. Closed additively here (label parity — same
    # rule as the two blocks above; additive-literal precedent D10/#419).
    "supervisor/update_candidate.py",
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

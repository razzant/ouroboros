"""Runtime-mode policy for protected Ouroboros source surfaces.

``advanced`` is allowed to evolve the application layer, but must not casually
rewrite the core contracts, safety files, or release/managed-repo invariants.
``pro`` may touch those paths, but commits still flow through the normal
review gate.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable

from ouroboros.settings_scales import _RUNTIME_MODE_RANK


def runtime_mode_rank(runtime_mode: str) -> int:
    """Return the ordered runtime-mode rank without duplicating the vocabulary.

    ``settings_scales`` is the owner of the persisted enum and rank. Unknown
    values remain below every known mode.
    """
    return int(_RUNTIME_MODE_RANK.get(str(runtime_mode or "").strip().lower(), -1))


def runtime_mode_at_least(runtime_mode: str, minimum: str) -> bool:
    """Whether ``runtime_mode`` meets the named ordered capability floor."""
    mode_rank = runtime_mode_rank(runtime_mode)
    minimum_rank = runtime_mode_rank(minimum)
    return mode_rank >= 0 and minimum_rank >= 0 and mode_rank >= minimum_rank


def mode_has_unrestricted_agency(runtime_mode: str) -> bool:
    """Whether internal permission and review decisions are advisory only.

    Cyber Pro keeps independent findings and actual operation failures intact;
    those facts do not grant an internal mechanism authority to veto an action.
    Consumers use the existing effective runtime-mode source, not saved future
    settings or a second task-local permission flag.
    """
    return runtime_mode_at_least(runtime_mode, "cyber_pro")


def protected_bible_history_delete_reason(
    raw_cmd: object, *,
    protect_bible: bool = True, identity_path: pathlib.Path | None = None,
    cwd: pathlib.Path | None = None, bible_path: pathlib.Path | None = None,
) -> str:
    """Return a refusal for physical BIBLE deletion or repository history rewrites.

    This is deliberately a small argv/verb predicate at the existing shell
    guard seam.  It does not classify arbitrary content or restrict ordinary
    ``rm`` commands elsewhere.
    """
    try:
        from ouroboros.shell_parse import collect_leading_env, shell_segments

        delete_heads = {"rm", "unlink", "mv"}
        history_verbs = {"filter-branch", "filter-repo", "rebase", "replace"}
        work_dir = cwd or pathlib.Path.cwd()
        bible_target = (bible_path or work_dir / "BIBLE.md").resolve(strict=False)

        def _protected_target(candidate: str) -> bool:
            path = pathlib.Path(candidate.replace("\\", "/"))
            target = (work_dir / path).resolve(strict=False)
            if protect_bible and str(target).casefold() == str(bible_target).casefold():
                return True
            return bool(identity_path is not None and (
                str(target).casefold() == str(identity_path.resolve(strict=False)).casefold()
            ))

        def _bible_path(
            words: list[str], *, path_flag_only: bool = False,
        ) -> bool:
            """Recognize an explicit BIBLE.md path, including --path= forms."""
            candidates: list[str] = []
            expect_value = False
            for word in words:
                token = str(word).strip("'\"")
                if expect_value:
                    candidates.append(token)
                    expect_value = False
                    continue
                if token in {"--path", "--path-file", "--paths"}:
                    expect_value = True
                    continue
                if token.startswith("--path="):
                    candidates.append(token.split("=", 1)[1])
                    continue
                if not path_flag_only:
                    candidates.append(token)
            return any(
                _protected_target(candidate)
                for candidate in candidates
            )


        for segment in shell_segments(raw_cmd):
            _env, argv = collect_leading_env(segment)
            if not argv:
                continue
            head = pathlib.PurePath(str(argv[0])).name.lower().removesuffix(".exe")
            words = [str(item).replace("\\", "/") for item in argv[1:]]
            if head in {"sh", "bash", "zsh"}:
                nested = ""
                for index, word in enumerate(words[:-1]):
                    if word in {"-c", "--command"}:
                        nested = words[index + 1]
                        break
                if nested:
                    nested_reason = protected_bible_history_delete_reason(
                        nested, protect_bible=protect_bible,
                        identity_path=identity_path, cwd=cwd, bible_path=bible_target,
                    )
                    if nested_reason:
                        return nested_reason
                    continue
            operands = [word for word in words if not word.startswith("-")]
            bible = _bible_path(operands[:-1] if head == "mv" else words)
            if head in delete_heads and bible:
                label = "IDENTITY" if any(
                    pathlib.PurePath(word.strip("'\"")).name.casefold() == "identity.md"
                    for word in words
                ) else "BIBLE"
                return f"{label}_DELETE_BLOCKED: protected identity history must remain physically present."
            if head == "git":
                verbs = [word.lower() for word in words if not word.startswith("-")]
                deleting = bool(verbs and (
                    verbs[0] in {"rm", "mv"}
                    or verbs[0] == "update-index" and any(flag in words for flag in ("--remove", "--force-remove"))
                ))
                git_targets = operands[1:-1] if verbs and verbs[0] == "mv" else operands[1:]
                if deleting and _bible_path(git_targets):
                    label = "IDENTITY" if any(
                        pathlib.PurePath(word.strip("'\"")).name.casefold() == "identity.md"
                        for word in words
                    ) else "BIBLE"
                    return f"{label}_DELETE_BLOCKED: git rm/git mv cannot remove or rename protected identity files."
                if verbs and verbs[0] in history_verbs and _bible_path(
                    words, path_flag_only=(verbs[0] in {"filter-branch", "filter-repo"})
                ):
                    return "BIBLE_HISTORY_REWRITE_BLOCKED: BIBLE history must remain physically recoverable."
        return ""
    except Exception:
        return ""


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
    # The provider-canary job `release-preflight` requires keeps its body in a
    # reusable workflow and its branch-push trigger in a wrapper — both outside
    # the protected ci.yml, neither moving any of the risk — so every inventory
    # that protects the parent must cover them (label parity — same rule as
    # the G1 block below).
    ".github/workflows/provider-canary.yml",
    ".github/workflows/provider-canary-push.yml",
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
    return runtime_mode_at_least(runtime_mode, "pro")


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
    target_modes = "runtime_mode='pro' or 'cyber_pro'" if str(runtime_mode).strip().lower() == "cyber_pro" else "runtime_mode='pro'"
    return (
        f"⚠️ CORE_PROTECTION_BLOCKED: runtime_mode={runtime_mode!r} refuses "
        f"to {action} protected {category or 'core'} path: {norm}. "
        f"Switch to {target_modes} and let the normal review panel "
        "cover the protected core/contract/release change before commit."
    )


def core_patch_notice(paths: Iterable[ProtectedPath | str]) -> str:
    from ouroboros.config import get_runtime_mode

    if mode_has_unrestricted_agency(get_runtime_mode()):
        return (
            "⚠️ CORE_PATCH_NOTICE: Cyber Pro is editing Ouroboros core/contract/release "
            f"surface(s): {format_protected_paths(paths)}. Independent review evidence "
            "remains separate from Ouroboros's decision to continue."
        )
    return (
        "⚠️ CORE_PATCH_NOTICE: runtime_mode='pro' or 'cyber_pro' is editing protected "
        "Ouroboros core/contract/release surface(s): "
        f"{format_protected_paths(paths)}. These changes can be committed only "
        "through the normal review gate."
    )

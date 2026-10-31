"""Static eligibility rules for workspace patch/snapshot capture (SSOT).

Extracted from ``ouroboros/headless.py`` when that module crossed its size gate:
these are the PURE rules — no git, no filesystem — deciding which paths may ride
in a captured workspace patch or a delegated-run execution snapshot: env/cache
directories, junk build artifacts, incidental lockfiles, credential-shaped names.
``headless`` re-exports every name here (same objects) and keeps the I/O checks
(`_untracked_blob_exclude_reason`) and the combined predicate
(`untracked_capture_veto_reason`) beside its git helpers, so the one composite
the snapshot and the patch both ask cannot drift from these rules.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, List

# Version 3: generated output directories are no longer excluded by NAME. The
# project's own `.gitignore` (honoured through `--exclude-standard`), the 5 MiB
# per-file cap below, git's own binary verdict and the credential-name check
# decide what a workspace.patch may carry; a project whose deliverable IS its
# build output must not have it silently dropped. The benchmark capture script
# owns its own any-depth rule and does not import this module. This is
# patch-transport hygiene only (artifact path/extension + git's binary
# verdict), never code/content inference (Bible P5).
_PATCH_EXCLUDE_RULES_VERSION = 3
_PATCH_MAX_UNTRACKED_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB per untracked file
_TOP_LEVEL_EXCLUDE_DIRS = {".ouroboros", ".venv", "venv", "env"}
_ANY_SEGMENT_EXCLUDE_DIRS = {
    ".cache",
    ".mypy_cache",
    ".npm",
    ".pnpm-store",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".yarn",
    "__pycache__",
    "node_modules",
}
# Junk file tails and caches the dir-sets above don't already cover. Runtime
# dumps, compiled bytecode and coverage output only; generated-output
# directories are the project's own .gitignore decision, not a name rule here.
_PATCH_JUNK_RE = re.compile(
    r"appendonlydir|\.rdb$|\.aof$|\.tmp$|\.pid$|\.sock$"
    r"|\.pyc$|\.pyo$|\.DS_Store|(^|/)\.coverage$"
    r"|coverage\.xml$|(^|/)htmlcov/"
)
_LOCKFILE_MANIFESTS = {
    "package-lock.json": "package.json",
    "npm-shrinkwrap.json": "package.json",
    "yarn.lock": "package.json",
    "pnpm-lock.yaml": "package.json",
    "go.sum": "go.mod",
    "Cargo.lock": "Cargo.toml",
    "poetry.lock": "pyproject.toml",
    "Pipfile.lock": "Pipfile",
    "composer.lock": "composer.json",
    "Gemfile.lock": "Gemfile",
}
# A delegated copy's NEW files under its named evidence folder: retained beside the
# capture by their owner (``subagent_integration_delegated.run_named_evidence``).
NAMED_EVIDENCE_EXCLUSION = "named evidence, retained separately by its host"
_SENSITIVE_EXAMPLE_SUFFIXES = (".example", ".sample", ".template", ".dist")
_SENSITIVE_KEY_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
_SENSITIVE_FILENAMES = {
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "aws-credentials.json",
    "credentials",
    "credentials.json",
    "gcp-service-account.json",
    "service-account.json",
    "secrets.json",
    "token.json",
}


def _patch_exclude_reason(rel: str) -> str:
    posix = str(rel).replace("\\", "/")
    parts = pathlib.PurePosixPath(posix).parts
    if not parts:
        return ""
    if parts[0] in _TOP_LEVEL_EXCLUDE_DIRS:
        return f"top-level env/cache directory: {parts[0]}"
    for part in parts:
        if part in _ANY_SEGMENT_EXCLUDE_DIRS:
            return f"env/cache directory segment: {part}"
    if _PATCH_JUNK_RE.search(posix):
        return f"junk artifact: {posix}"
    return ""


def _lockfile_manifest_for(rel: str) -> str:
    posix = str(rel).replace("\\", "/")
    path = pathlib.PurePosixPath(posix)
    manifest = _LOCKFILE_MANIFESTS.get(path.name)
    return path.with_name(manifest).as_posix() if manifest else ""


def _incidental_lockfile_excludes(changed_paths: List[str]) -> set[str]:
    changed = {str(path or "").replace("\\", "/") for path in changed_paths if str(path or "").strip()}
    lock_to_manifest = {
        path: manifest
        for path in changed
        for manifest in [_lockfile_manifest_for(path)]
        if manifest
    }
    if not lock_to_manifest:
        return set()
    if not (changed - set(lock_to_manifest)):
        return set()
    return {path for path, manifest in lock_to_manifest.items() if manifest not in changed}


def _sensitive_untracked_reason(rel: str) -> str:
    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    if mode_has_unrestricted_agency(get_runtime_mode()):
        return ""
    name = pathlib.PurePosixPath(str(rel).replace("\\", "/")).name
    lower = name.lower()
    is_dotenv_secret = lower.startswith(".env") or lower.endswith(".env") or ".env." in lower
    if is_dotenv_secret and not lower.endswith(_SENSITIVE_EXAMPLE_SUFFIXES):
        return "dotenv secret"
    return ""


__all__ = [
    "NAMED_EVIDENCE_EXCLUSION",
    "_ANY_SEGMENT_EXCLUDE_DIRS",
    "_LOCKFILE_MANIFESTS",
    "_PATCH_EXCLUDE_RULES_VERSION",
    "_PATCH_JUNK_RE",
    "_PATCH_MAX_UNTRACKED_FILE_BYTES",
    "_SENSITIVE_EXAMPLE_SUFFIXES",
    "_SENSITIVE_FILENAMES",
    "_SENSITIVE_KEY_NAMES",
    "_TOP_LEVEL_EXCLUDE_DIRS",
    "_incidental_lockfile_excludes",
    "_lockfile_manifest_for",
    "_patch_exclude_reason",
    "_sensitive_untracked_reason",
]


def format_patch_exclusions(manifest: Any) -> str:
    """One disclosure line for files the capture dropped per policy, or ''.

    The per-file exclusions (#447 F5) live in the workspace_patch manifest,
    which no parent-facing surface used to render — a dropped deliverable hid
    behind an affirmative "Integrated N file(s)" success line. This renders the
    manifest rows the rules above produce, so it belongs beside them rather
    than in either of the two integration tools that display it.
    """
    entries, evidence = [], 0
    for key in ("sensitive_blocked", "untracked_excluded", "tracked_excluded"):
        for item in (manifest or {}).get(key) or []:
            if isinstance(item, dict):
                path, reason = str(item.get("path") or ""), str(item.get("reason") or "")
            else:
                path, reason = str(item or ""), ""
            if path and reason == NAMED_EVIDENCE_EXCLUSION:
                evidence += 1  # not a lost deliverable: its owner keeps the bytes
            elif path:
                entries.append(f"{path} ({reason})" if reason else path)
    kept = (f"\n{evidence} new file(s) under the copy's named evidence folder are evidence, not part of "
            "this patch: named_evidence reports their retention.") if evidence else ""
    if not entries:
        return kept
    shown = "; ".join(entries[:8])
    more = (
        f" and {len(entries) - 8} more (full list with reasons: the child's "
        "workspace_patch.json artifact)"
        if len(entries) > 8 else ""
    )
    return (
        f"\n⚠️ {len(entries)} file(s) EXCLUDED from this patch by capture policy: "
        f"{shown}{more}. Excluded content is NOT in this patch: if one is a real "
        "deliverable, recover it from the child workspace/snapshot if that "
        "still exists, or have it re-produced."
    ) + kept

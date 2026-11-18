"""Static eligibility rules for workspace patches (SSOT).

These are the static rules for workspace patches: env/cache directories, junk
artifacts, incidental lockfiles and credential-shaped names. The capture supplies
bounded pnpm ownership evidence. ``headless`` re-exports these rules and the I/O
checks from ``workspace_patch_capture``. Result-commit snapshots do not filter
incidental lockfiles.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, Callable, Iterable, List

# Version 3: generated output directories are no longer excluded by NAME. The
# project's own `.gitignore` (honoured through `--exclude-standard`), the 5 MiB
# per-file cap below, git's own binary verdict and the credential-name check
# decide what a workspace.patch may carry; a project whose deliverable IS its
# build output must not have it silently dropped. The benchmark capture script
# owns its own any-depth rule and does not import this module. This is
# patch-transport hygiene only (artifact path/extension + git's binary
# verdict), never code/content inference (Bible P5).
# Version 4: a pnpm lockfile may own a changed manifest below its directory.
_PATCH_EXCLUDE_RULES_VERSION = 4
_PATCH_MAX_UNTRACKED_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB per untracked file
_LOCKFILE_METADATA_BYTES = 5 * 1024 * 1024  # bound YAML parsing and baseline blob reads
_INCIDENTAL_LOCKFILE_REASON = "incidental lockfile: no matching changed manifest found"
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


def _pnpm_importer_manifests(lockfile: str, raw: bytes) -> set[str]:
    # Compose nodes, not objects: pnpm importer labels are paths, not YAML values
    # (such as `no` or `123`); do not expand YAML aliases/merges into data.
    import yaml

    if len(raw) > _LOCKFILE_METADATA_BYTES:
        raise ValueError(f"lockfile exceeds the {_LOCKFILE_METADATA_BYTES}-byte metadata limit")
    try:
        document = yaml.compose(raw, Loader=getattr(yaml, "CBaseLoader", yaml.BaseLoader))
    except (yaml.YAMLError, UnicodeError, RecursionError) as exc:
        raise ValueError("lockfile YAML could not be read") from exc
    if not isinstance(document, yaml.MappingNode):
        raise ValueError("lockfile has no top-level mapping")
    fields = {}
    for key, value in document.value:
        if not isinstance(key, yaml.ScalarNode) or key.value in fields:
            raise ValueError("lockfile fields are ambiguous")
        fields[key.value] = value
    importers = fields.get("importers")
    if not isinstance(fields.get("lockfileVersion"), yaml.ScalarNode) or not isinstance(importers, yaml.MappingNode):
        raise ValueError("lockfile has no identifiable importer map")
    directory = pathlib.PurePosixPath(lockfile).parent
    manifests = set()
    for name, importer in importers.value:
        if not isinstance(name, yaml.ScalarNode) or not isinstance(importer, yaml.MappingNode):
            raise ValueError("lockfile importer map is incomplete")
        path = pathlib.PurePosixPath(name.value)
        if not path.is_absolute() and ".." not in path.parts and "\\" not in name.value:
            manifests.add((directory / path / "package.json").as_posix())
    return manifests


def _incidental_lockfile_excludes(
    changed_paths: List[str], *, pnpm_lock_sources: Callable[[str], Iterable[bytes]] | None = None,
    diagnostics: list[dict] | None = None,
) -> set[str]:
    changed = {str(path or "").replace("\\", "/") for path in changed_paths if str(path or "").strip()}
    lock_to_manifest = {
        path: manifest
        for path in changed
        for manifest in [_lockfile_manifest_for(path)]
        if manifest
    }
    if not (changed - set(lock_to_manifest)):
        return set()
    incidental = {path for path, manifest in lock_to_manifest.items() if manifest not in changed}
    if pnpm_lock_sources is None or not incidental:
        return incidental
    manifests = {path for path in changed if pathlib.PurePosixPath(path).name == "package.json"}
    for lockfile in sorted(incidental):
        path = pathlib.PurePosixPath(lockfile)
        if path.name != "pnpm-lock.yaml" or not any(
            pathlib.PurePosixPath(manifest).is_relative_to(path.parent) for manifest in manifests
        ):
            continue
        try:
            keep = any(manifests & _pnpm_importer_manifests(lockfile, raw) for raw in pnpm_lock_sources(lockfile))
        except (ValueError, OSError) as exc:
            keep = True
            if diagnostics is not None:
                diagnostics.append({"reason": "pnpm_membership_unknown", "path": lockfile,
                                    "detail": str(exc), "advisory": True})
        if keep:
            incidental.remove(lockfile)
    return incidental


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


def declared_capture_paths(manifest: dict) -> list[str]:
    """Paths this Git capture intended to deliver, including independent file results.

    A rejection records the selection without staging it. Applying still uses Git's
    own patch reader and verified file-result rows for its authority and path set.
    """
    text_paths = set(manifest.get("tracked_changed") or ()) | set(manifest.get("untracked_included") or ())
    text_paths.difference_update(row["path"] for row in manifest.get("tracked_excluded") or ()
                                 if isinstance(row, dict) and row.get("path"))
    return sorted(text_paths | {row["path"] for row in manifest.get("file_output_changes") or ()
                                if isinstance(row, dict) and row.get("path")})


def format_patch_exclusions(manifest: Any, carried=None, outcome: str = "retained", unstaged=()) -> str:
    """Disclose exclusions alongside the verified file results and this call's effects.

    New evidence belongs to its own retention owner. A policy exclusion without a
    verified file result still needs recovery advice, even after a successful apply.
    """
    manifest = manifest or {}
    verified = manifest.get("file_results_carried", ()) if carried is None else carried
    held = {str(row["path"] if isinstance(row, dict) else row) for row in verified}
    reasons, excluded, evidence = {}, [], 0
    for key in ("sensitive_blocked", "untracked_excluded", "tracked_excluded"):
        for item in manifest.get(key) or ():
            path, reason = (item.get("path"), item.get("reason")) if isinstance(item, dict) else (item, "")
            if path and reason == NAMED_EVIDENCE_EXCLUSION:
                evidence += 1
            elif path:
                name = f"{path} ({reason})" if reason else path
                if path in held:
                    reasons[path] = name
                else:
                    excluded.append(name)

    def listed(names):
        suffix = f" and {len(names) - 8} more (see workspace_patch.json)" if len(names) > 8 else ""
        return "; ".join(names[:8]) + suffix

    lines = []
    if excluded:
        lines.append(f"⚠️ {len(excluded)} file(s) EXCLUDED from this patch by capture policy: {listed(excluded)}. "
                     "If one is a deliverable, recover it from the child workspace or reproduce it.")
    if held:
        action = {"retained": "retained; this call applied and staged none",
                  "applied": "delivered and staged except where marked",
                  "reverted": "rolled back with the rest of the attempted transfer",
                  "incomplete": "with an incomplete transfer; inspect the working tree and index"}[outcome]
        labels = [reasons.get(path, path) + (" [unstaged]" if outcome == "applied" and path in unstaged else "")
                  for path in sorted(held)]
        lines.append(f"{len(held)} separate file result(s) {action}: {listed(labels)}.")
    if evidence:
        lines.append(f"{evidence} new file(s) under the copy's named evidence folder are evidence, not part of "
                     "this patch: named_evidence reports their retention.")
    if error := manifest.get("file_results_unverified"):
        lines.append(f"Separate file results could not be verified: {error}.")
    for row in manifest.get("other_artifact_faults") or ():
        name = row.get("relpath") or row.get("name") or row.get("path")
        errors = "; ".join(map(str, row.get("errors") or ()))
        lines.append(f"⚠️ Other artifact in the result bundle: {name} is {row['status']}: {errors or 'no error recorded'}.")
    return "".join("\n" + line for line in lines)

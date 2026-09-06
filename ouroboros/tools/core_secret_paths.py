"""Read-denial policy for RESTRICTED subagents (leaf of ``tools/core``).

Which locations a read-only / acting / constraint-less delegated child may NOT
read: owner secrets and control state under the data root, credential-shaped
files in the repository, and the listing-side redaction that hides such
entries. Extracted from ``core_file_tools`` so the child contract (pinned
byte-for-byte by ``tests/test_credential_shapes.py``) has one owner; the
``tools/core`` facade keeps every historical import path.
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING, List

from ouroboros.contracts.skill_payload_policy import (
    SKILL_OWNER_STATE_FILENAMES as _SKILL_OWNER_STATE_FILENAMES,
    is_skill_owner_state_alias,
    is_skill_owner_state_target as _is_skill_owner_state_target,
)
from ouroboros.credential_shapes import (
    CREDENTIAL_FILE_SUFFIXES,
    CREDENTIAL_NAME_RE,
    SUBAGENT_CREDENTIAL_FILE_NAMES as _SUBAGENT_SECRET_FILE_NAMES,
)

if TYPE_CHECKING:  # pragma: no cover
    from ouroboros.tools.registry import ToolContext


def is_restricted_subagent_profile(ctx: ToolContext) -> bool:
    # Fail-closed SSOT for subagent READ restrictions (secret/control denials):
    # read-only subagents, acting subagents, and delegated subagents with a
    # missing/invalid constraint are ALL barred from reading owner secrets/control
    # state. Acting children may WRITE their isolated surface but never read owner
    # secrets; the resource WRITE distinction lives in _local_readonly_resource_block.
    from ouroboros.tool_access import active_tool_profile
    return active_tool_profile(ctx) in ("local_readonly_subagent", "acting_subagent")


def _is_subagent_secret_data_path(norm: str) -> bool:
    text = str(norm or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    if not text:
        return False
    parts = [part.lower() for part in text.split("/") if part and part != "."]
    if not parts:
        return False
    if any(part in {"auth", "credentials", "secrets", "tokens"} for part in parts):
        return True
    name = parts[-1]
    normalized_names = {name, name.lstrip(".")}
    if name.lstrip(".") == "settings.tmp":
        normalized_names.add("settings.json")
    for protected_name in (_SUBAGENT_SECRET_FILE_NAMES | _SKILL_OWNER_STATE_FILENAMES):
        bare = name.lstrip(".")
        if bare.startswith(f"{protected_name}.tmp") or bare.startswith(f"{protected_name}.lock"):
            normalized_names.add(protected_name)
    if normalized_names & (_SUBAGENT_SECRET_FILE_NAMES | _SKILL_OWNER_STATE_FILENAMES):
        return True
    if name.startswith(".env") or name.endswith(".env") or ".env." in name:
        return True
    if name.endswith(CREDENTIAL_FILE_SUFFIXES):
        return True
    return bool(CREDENTIAL_NAME_RE.search(name))


def _is_subagent_secret_repo_path(norm: str) -> bool:
    text = str(norm or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    parts = [part.lower() for part in text.split("/") if part and part != "."]
    if ".git" in parts or any(part in {"auth", "credentials", "secrets", "tokens"} for part in parts):
        return True
    if not parts:
        return False
    name = parts[-1]
    if name in _SUBAGENT_SECRET_FILE_NAMES or name == "settings.tmp":
        return True
    if name.startswith(".env") or name.endswith(".env") or ".env." in name:
        return True
    if name.endswith(CREDENTIAL_FILE_SUFFIXES):
        return True
    if CREDENTIAL_NAME_RE.search(name):
        suffix = pathlib.PurePosixPath(name).suffix.lower()
        return suffix in {"", ".json", ".env", ".key", ".pem", ".p12", ".pfx", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".conf"}
    return False


def _is_subagent_secret_repo_target(target: pathlib.Path, repo_root: pathlib.Path) -> bool:
    root = pathlib.Path(repo_root).resolve(strict=False)
    try:
        rel = str(pathlib.Path(target).resolve(strict=False).relative_to(root)).replace(os.sep, "/")
    except (OSError, ValueError):
        rel = str(target).replace(os.sep, "/")
    if _is_subagent_secret_repo_path(rel):
        return True
    secret_candidates = [
        root / ".git" / "credentials",
        root / ".git" / "config",
    ]
    try:
        secret_candidates.extend(
            candidate
            for candidate in root.iterdir()
            if candidate.is_file() and _is_subagent_secret_repo_path(candidate.name)
        )
    except OSError:
        pass
    return any(
        candidate.is_file()
        and target.exists()
        and target.samefile(candidate)
        for candidate in secret_candidates
    )


def _filter_subagent_secret_repo_listing(items: List[str], repo_root: pathlib.Path) -> List[str]:
    filtered: List[str] = []
    redacted = 0
    root = pathlib.Path(repo_root).resolve(strict=False)
    for item in items:
        marker = item.rstrip("/")
        if marker.startswith("⚠️") or marker.startswith("...("):
            filtered.append(item)
            continue
        if _is_subagent_secret_repo_path(marker) or _is_subagent_secret_repo_target(root / marker, root):
            redacted += 1
            continue
        filtered.append(item)
    if redacted:
        filtered.append(f"⚠️ {redacted} secret/control entr{'y' if redacted == 1 else 'ies'} hidden from this subagent.")
    return filtered


def _filter_subagent_secret_listing(items: List[str], data_root: pathlib.Path) -> List[str]:
    filtered: List[str] = []
    redacted = 0
    root = pathlib.Path(data_root).resolve(strict=False)
    for item in items:
        marker = item.rstrip("/")
        if marker.startswith("⚠️") or marker.startswith("...("):
            filtered.append(item)
            continue
        target = root / marker
        try:
            resolved_rel = str(pathlib.Path(target).resolve(strict=False).relative_to(root)).replace(os.sep, "/")
        except (OSError, ValueError):
            resolved_rel = marker
        if (
            _is_subagent_secret_data_path(marker)
            or _is_subagent_secret_data_path(resolved_rel)
            or _is_skill_owner_state_target(target, root)
            or is_skill_owner_state_alias(target, root)
            or any(
                candidate.is_file()
                and _is_subagent_secret_data_path(candidate.name)
                and target.exists()
                and target.samefile(candidate)
                for candidate in root.iterdir()
            )
        ):
            redacted += 1
            continue
        filtered.append(item)
    if redacted:
        filtered.append(f"⚠️ {redacted} secret/control entr{'y' if redacted == 1 else 'ies'} hidden from this subagent.")
    return filtered


def restricted_data_roots(ctx: ToolContext) -> list[pathlib.Path]:
    """All runtime roots used by file admission, including a forked child's parent.

    The root set comes from the existing vision/file parity owner: child drive,
    canonical budget drive, and configured runtime admission root. A child's
    isolated drive must not hide canonical owner state nested in its repository.
    """
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.config import DATA_DIR

    values = [getattr(ctx, "drive_root", "")]
    try:
        values.append(canonical_data_root(ctx))
    except Exception:
        pass
    values.append(DATA_DIR)
    roots = []
    for value in values:
        if not str(value or "").strip():
            continue
        try:
            root = pathlib.Path(value).expanduser().resolve(strict=False)
        except (OSError, ValueError, RuntimeError):
            continue
        if root not in roots:
            roots.append(root)
    return roots

"""Read-denial policy for RESTRICTED subagents (leaf of ``tools/core``).

Which locations a read-only / acting / constraint-less delegated child may NOT
read: owner secrets and control state under the data root, credential-shaped
files in the repository, and the listing-side redaction that hides such
entries. The child contract has one owner across read/list/search/query;
ordinary source directory names do not identify credential stores. The
``tools/core`` facade keeps the shared import surface.
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
    """Repo source names do not confer owner authority or identify a secret store."""
    text = str(norm or "").replace("\\", "/").strip()
    parts = pathlib.PurePosixPath(text).parts
    if ".git" in {part.lower() for part in parts}:
        return True
    name = pathlib.PurePosixPath(text).name.lower()
    if name in (_SUBAGENT_SECRET_FILE_NAMES - {"settings.json", "settings.json.lock"}):
        return True
    # Match the live dotenv forms, not an ordinary example/template payload.
    if name.startswith(".env") or name.endswith(".env") or ".env." in name:
        return not name.endswith((".example", ".sample", ".template", ".dist"))
    if name.endswith(CREDENTIAL_FILE_SUFFIXES):
        return True
    # Credential records remain private; an arbitrary source/config suffix or
    # a directory named auth/tokens is not a credential location.
    return bool(CREDENTIAL_NAME_RE.search(name)) and pathlib.PurePosixPath(name).suffix in {"", ".json", ".env"}


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

"""Read-denial policy for RESTRICTED subagents (leaf of ``tools/core``).

Which locations a read-only / acting / constraint-less delegated child may NOT
read: owner secrets and control state under the data root, repository credential
locations, and the listing-side redaction that hides such
entries. The child contract has one owner across read/list/search/query;
ordinary source directory names do not identify credential stores. The
``tools/core`` facade keeps the shared import surface.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Callable, List

from ouroboros.contracts.skill_payload_policy import (
    SKILL_OWNER_STATE_FILENAMES as _SKILL_OWNER_STATE_FILENAMES,
    is_skill_owner_state_alias,
    is_skill_owner_state_target as _is_skill_owner_state_target,
)
from ouroboros.credential_shapes import SUBAGENT_CREDENTIAL_FILE_NAMES as _SUBAGENT_SECRET_FILE_NAMES

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
    # These are stores at the runtime root, not directory-name patterns in
    # task payloads, source trees or artifact stores beneath it.
    if parts[0] in {"auth", "credentials", "secrets", "tokens"}:
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
    return False


def _is_subagent_secret_repo_path(norm: str, *, credential_names: bool = True) -> bool:
    """Repo source names do not confer owner authority or identify a secret store."""
    text = str(norm or "").replace("\\", "/").strip()
    parts = pathlib.PurePosixPath(text).parts
    if ".git" in {part.lower() for part in parts}:
        return True
    if not credential_names:
        return False
    # Exact credential names remain protected at every depth. Ordinary source
    # directories such as src/auth and public certificate suffixes do not
    # identify a credential store.
    name = pathlib.PurePosixPath(text).name.lower()
    if name in (_SUBAGENT_SECRET_FILE_NAMES - {"settings.json", "settings.json.lock"}):
        return True
    # Match the live dotenv forms, not an ordinary example/template payload.
    if name.startswith(".env") or name.endswith(".env") or ".env." in name:
        return not name.endswith((".example", ".sample", ".template", ".dist"))
    return False


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


def _is_subagent_secret_repo_target(
    target: pathlib.Path, repo_root: pathlib.Path, *, data_root: pathlib.Path | None = None,
    ctx: ToolContext | None = None,
) -> bool:
    return make_subagent_secret_target_check(repo_root, data_root=data_root, ctx=ctx)(target)


def make_subagent_secret_target_check(
    repo_root: pathlib.Path, *, data_root: pathlib.Path | None = None,
    ctx: ToolContext | None = None,
) -> Callable[[pathlib.Path], bool]:
    """Prepare shared locations for one file-tool traversal, never across calls.

    Only roots and candidate path names are retained. Each target still resolves
    and checks live owner-state aliases and file identity before admission.
    """
    from ouroboros.credential_shapes import owner_credential_locations
    from ouroboros.tool_access import resource_root_path

    root = pathlib.Path(repo_root).resolve(strict=False)
    protected, allowed = owner_credential_locations(pathlib.Path.home())
    protected = [path.resolve(strict=False) for path in protected]
    # Host-selected task/output roots hold ordinary content. Their filenames
    # do not turn them into repository credentials, while resolved owner-state
    # aliases and VCS internals below retain their separate protection.
    content_roots = []
    if ctx is not None:
        for kind in ("task_drive", "artifact_store"):
            try:
                content_roots.append(resource_root_path(ctx, kind))
            except (AttributeError, OSError, TypeError, ValueError):
                break  # Keep earlier roots, matching the original short-circuit lookup.
    data_roots = restricted_data_roots(ctx) if ctx is not None else []
    if data_root is not None:
        data_roots.append(pathlib.Path(data_root).resolve(strict=False))
    data_roots = list(dict.fromkeys(data_roots))
    secret_candidates = [root / ".git" / "credentials", root / ".git" / "config"]
    for data in data_roots:
        try:
            secret_candidates.extend(candidate for candidate in data.iterdir()
                                     if candidate.is_file() and _is_subagent_secret_data_path(candidate.name))
        except OSError:
            pass
    try:
        secret_candidates.extend(candidate for candidate in root.iterdir()
                                 if candidate.is_file() and _is_subagent_secret_repo_path(candidate.name))
    except OSError:
        pass
    def is_secret(target: pathlib.Path) -> bool:
        target = pathlib.Path(target).resolve(strict=False)
        if target not in allowed and any(target.is_relative_to(path) for path in protected):
            return True
        task_content = any(target.is_relative_to(path) for path in content_roots)
        for data in data_roots:
            try:
                data_rel = target.relative_to(data).as_posix()
            except ValueError:
                data_rel = ""
            if data_rel and (
                (not task_content and _is_subagent_secret_data_path(data_rel))
                or _is_skill_owner_state_target(target, data)
                or is_skill_owner_state_alias(target, data)
            ):
                return True
        try:
            rel = target.relative_to(root).as_posix()
        except ValueError:
            rel = target.as_posix()
        if _is_subagent_secret_repo_path(rel, credential_names=not task_content):
            return True
        return any(candidate.is_file() and target.exists() and target.samefile(candidate)
                   for candidate in secret_candidates)

    return is_secret


def _filter_subagent_secret_repo_listing(
    items: List[str], repo_root: pathlib.Path, *, data_root: pathlib.Path | None = None,
    ctx: ToolContext | None = None, base_path: pathlib.Path | None = None,
    secret_check: Callable[[pathlib.Path], bool] | None = None,
) -> List[str]:
    filtered: List[str] = []
    redacted = 0
    root = pathlib.Path(repo_root).resolve(strict=False)
    base = pathlib.Path(base_path).resolve(strict=False) if base_path is not None else root
    secret_check = secret_check or make_subagent_secret_target_check(root, data_root=data_root, ctx=ctx)
    for item in items:
        marker = item.rstrip("/")
        if marker.startswith("⚠️") or marker.startswith("...("):
            filtered.append(item)
            continue
        if secret_check(base / marker):
            redacted += 1
            continue
        filtered.append(item)
    if redacted:
        filtered.append(f"⚠️ {redacted} secret/control entr{'y' if redacted == 1 else 'ies'} hidden from this subagent.")
    return filtered


def _filter_subagent_secret_listing(
    items: List[str], data_root: pathlib.Path, *, ctx: ToolContext | None = None,
    secret_check: Callable[[pathlib.Path], bool] | None = None,
) -> List[str]:
    """List data/task payloads against their real runtime and repository owners."""
    from ouroboros.tools.registry import active_repo_dir_for

    return _filter_subagent_secret_repo_listing(
        items, active_repo_dir_for(ctx) if ctx is not None else data_root,
        data_root=data_root if ctx is None else None, ctx=ctx, base_path=data_root,
        secret_check=secret_check,
    )

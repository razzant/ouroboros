"""Retained runtime-control command policy and restricted action identity.

File readers do not use credential shapes or helper-only exclusions. The
runtime target predicates below serve the existing shell-control policy;
the profile predicate also serves browser actions and origin delegation.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Callable

from ouroboros.contracts.skill_payload_policy import (
    SKILL_OWNER_STATE_FILENAMES as _SKILL_OWNER_STATE_FILENAMES,
    is_skill_owner_state_alias,
    is_skill_owner_state_target as _is_skill_owner_state_target,
)
from ouroboros.credential_shapes import RUNTIME_SECRET_FILE_NAMES
from ouroboros.tool_access_paths import runtime_data_roots as runtime_data_roots

if TYPE_CHECKING:  # pragma: no cover
    from ouroboros.tools.registry import ToolContext


def is_restricted_subagent_profile(ctx: ToolContext) -> bool:
    """Identity for restricted actions, independent from inherited file reads."""
    from ouroboros.tool_access import active_tool_profile
    profile = active_tool_profile(ctx)
    if profile == "acting_subagent":
        try:
            from ouroboros.config import get_runtime_mode
            from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

            if mode_has_unrestricted_agency(get_runtime_mode()):
                return False
        except Exception:
            pass
    return profile in ("local_readonly_subagent", "acting_subagent")


def _is_runtime_secret_data_path(norm: str, *, include_owner_state: bool = True) -> bool:
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
    protected_names = RUNTIME_SECRET_FILE_NAMES | (_SKILL_OWNER_STATE_FILENAMES if include_owner_state else set())
    for protected_name in protected_names:
        bare = name.lstrip(".")
        if bare.startswith(f"{protected_name}.tmp") or bare.startswith(f"{protected_name}.lock"):
            normalized_names.add(protected_name)
    if normalized_names & protected_names:
        return True
    if name.startswith(".env") or name.endswith(".env") or ".env." in name:
        return True
    return False


def _is_runtime_secret_repo_path(norm: str, *, credential_names: bool = True) -> bool:
    """Repo source names do not confer owner authority or identify a secret store."""
    text = str(norm or "").replace("\\", "/").strip()
    parts = pathlib.PurePosixPath(text).parts
    if ".git" in {part.lower() for part in parts}:
        return True
    if not credential_names:
        return False
    # A project filename alone does not identify an owner credential store.
    # Match the live dotenv forms, not an ordinary example/template payload.
    name = pathlib.PurePosixPath(text).name.lower()
    if name.startswith(".env") or name.endswith(".env") or ".env." in name:
        return not name.endswith((".example", ".sample", ".template", ".dist"))
    return False


def is_runtime_secret_target(
    target: pathlib.Path, repo_root: pathlib.Path, *, data_root: pathlib.Path | None = None,
    ctx: ToolContext | None = None,
    include_owner_state: bool = True,
) -> bool:
    return make_runtime_secret_target_check(repo_root, data_root=data_root, ctx=ctx,
                                          include_owner_state=include_owner_state)(target)


def make_runtime_secret_target_check(
    repo_root: pathlib.Path, *, data_root: pathlib.Path | None = None,
    ctx: ToolContext | None = None,
    include_owner_state: bool = True,
) -> Callable[[pathlib.Path], bool]:
    """Prepare locations for one runtime-control command check, never across calls.

    Read-only typed state readers may exclude nonsecret owner mutation state.
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
    data_roots = runtime_data_roots(ctx) if ctx is not None else []
    if data_root is not None:
        data_roots.append(pathlib.Path(data_root).resolve(strict=False))
    data_roots = list(dict.fromkeys(data_roots))
    secret_candidates = [root / ".git" / "credentials", root / ".git" / "config"]
    for data in data_roots:
        try:
            secret_candidates.extend(candidate for candidate in data.iterdir()
                                     if candidate.is_file() and _is_runtime_secret_data_path(
                                         candidate.name, include_owner_state=include_owner_state))
        except OSError:
            pass
    try:
        secret_candidates.extend(candidate for candidate in root.iterdir()
                                 if candidate.is_file() and _is_runtime_secret_repo_path(candidate.name))
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
                (not task_content and _is_runtime_secret_data_path(data_rel, include_owner_state=include_owner_state))
                or (include_owner_state and (_is_skill_owner_state_target(target, data)
                                            or is_skill_owner_state_alias(target, data)))
            ):
                return True
        try:
            rel = target.relative_to(root).as_posix()
        except ValueError:
            rel = target.as_posix()
        if _is_runtime_secret_repo_path(rel, credential_names=not task_content):
            return True
        return any(candidate.is_file() and target.exists() and target.samefile(candidate)
                   for candidate in secret_candidates)

    return is_secret

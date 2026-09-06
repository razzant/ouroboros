"""Physical path primitives for the access matrix.

Every span is extracted VERBATIM from the parent's tip bytes by
scripts/v7next_transplant.py (D18/D33 module-handle split, proof-checked);
the parent re-exports every moved name, so historical imports and
monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import os
import pathlib

from ouroboros.tool_access_types import _ALL_ROOTS

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only imports (inert at runtime)
    from ouroboros.tool_access_types import ResourceRoot
    from typing import Any


def _tool_access():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros import tool_access

    return tool_access


def _user_files_root() -> pathlib.Path:
    """Filesystem base for the ``user_files`` resource root.

    Defaults to the owner's real home. A jailed/benchmark runtime can redirect it
    to a scratch directory via ``OUROBOROS_USER_FILES_ROOT`` so a task physically
    cannot resolve the owner's real home (e.g. ``~/file1.txt`` secret files). Any
    unusable value falls back to the real home — fail-safe, never broadens reach.
    """
    raw = (os.environ.get("OUROBOROS_USER_FILES_ROOT") or "").strip()
    if raw:
        try:
            return pathlib.Path(raw).expanduser().resolve(strict=False)
        except Exception:
            # ANY unusable value (bad path, unknown ``~user`` RuntimeError, odd OS error)
            # fails safe to the real home — the doc's "any unusable value" contract.
            pass
    return pathlib.Path.home().resolve(strict=False)


def _deliverables_root() -> pathlib.Path:
    """Container for UNNAMED user deliverables, JAIL-AWARE: when the user_files home is
    redirected (``OUROBOROS_USER_FILES_ROOT``) and no explicit
    ``OUROBOROS_DELIVERABLES_ROOT`` is set, keep unnamed deliverables INSIDE the jail so a
    bare ``write_file(root='user_files', path='answer.txt')`` stays reachable and in-bounds
    instead of escaping to the real ``~/Ouroboros/Deliverables`` (which the outside-home
    check would then reject). Otherwise the global config default applies.
    """
    from ouroboros.config import get_deliverables_root

    jail = (os.environ.get("OUROBOROS_USER_FILES_ROOT") or "").strip()
    explicit = (os.environ.get("OUROBOROS_DELIVERABLES_ROOT") or "").strip()
    if explicit:
        return pathlib.Path(explicit).expanduser().resolve(strict=False)
    if jail and not explicit:
        return (_user_files_root() / "Deliverables").resolve(strict=False)
    return pathlib.Path(get_deliverables_root()).expanduser().resolve(strict=False)


def normalize_root(root: str | None, *, default: ResourceRoot = "active_workspace") -> ResourceRoot:
    candidate = str(root or default).strip() or default
    if candidate not in _ALL_ROOTS:
        raise ValueError(f"unknown root {candidate!r}; expected one of {sorted(_ALL_ROOTS)}")
    return candidate  # type: ignore[return-value]


def path_is_relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        pathlib.Path(path).resolve(strict=False).relative_to(pathlib.Path(root).resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def normalize_root_relative(root: pathlib.Path, path: str) -> str:
    """Map a model-supplied path to a root-relative string when it redundantly
    encodes the root, so structural/read tools accept the paths an agent
    naturally writes: an absolute path inside the active root (e.g. ``/app/foo``
    under a workspace rooted at ``/app``) and a single redundant root-basename
    prefix (``app/foo``). Returns a RELATIVE string only — it never widens
    access: callers still apply ``safe_relpath`` + a ``relative_to`` confinement
    check, so a genuine escape is still rejected downstream.

    - absolute & inside root  -> stripped to relative
    - absolute & outside root -> returned unchanged (caller's check rejects it)
    - redundant root-basename prefix, existence-guarded -> stripped
    - otherwise -> unchanged
    """

    text = str(path or "").strip().replace("\\", "/")
    if not text or text in (".", "./"):
        return text
    try:
        root_resolved = pathlib.Path(root).resolve(strict=False)
    except (OSError, ValueError):
        return text
    # (A) absolute path that already points inside the root.
    if _tool_access().is_absolute_path_text(text):
        try:
            return pathlib.Path(text).resolve(strict=False).relative_to(root_resolved).as_posix()
        except (OSError, ValueError):
            return text  # outside root -> let the caller's confinement reject it
    # (B) redundant root-basename prefix ('app' or 'app/x' when root basename is
    # 'app'). Strip it UNLESS the root contains a real same-named subdir (then
    # 'app/x' is ambiguously a genuine nested path and is kept). Gating on the
    # absence of that subdir — not on the target existing — lets NEW write/create
    # targets ('app/new.py' -> 'new.py') normalize too, while a real 'app/'
    # subdir is never mis-stripped. Only ever shortens toward root (no escape).
    base = root_resolved.name
    if base and (text == base or text.startswith(base + "/")):
        try:
            if not (root_resolved / base).is_dir():
                return text[len(base):].lstrip("/") or "."
        except (ValueError, OSError):
            # `..`/traversal or stat error: leave unchanged so the caller's
            # confinement produces the canonical (not a generic) error.
            return text
    return text


def _path_is_relative_to_casefold(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path_parts = pathlib.Path(path).resolve(strict=False).parts
        root_parts = pathlib.Path(root).resolve(strict=False).parts
    except (OSError, ValueError):
        return False
    if len(path_parts) < len(root_parts):
        return False
    return tuple(part.casefold() for part in path_parts[: len(root_parts)]) == tuple(
        part.casefold() for part in root_parts
    )


def paths_overlap_casefold(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Return True when two paths overlap under case-insensitive path semantics."""

    return _path_is_relative_to_casefold(left, right) or _path_is_relative_to_casefold(right, left)


def workspace_mode_block_reason(ctx: Any) -> str:
    mode = str(getattr(ctx, "workspace_mode", "") or "").strip()
    workspace_root = getattr(ctx, "workspace_root", None)
    if not mode or workspace_root is None:
        return ""
    try:
        workspace = pathlib.Path(workspace_root).resolve(strict=False)
    except (OSError, TypeError, ValueError):
        return "workspace_root is invalid"
    protected_values = (
        ("Ouroboros system repo", getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir", None)),
        ("Ouroboros repo", getattr(ctx, "repo_dir", None)),
        ("Ouroboros data drive", getattr(ctx, "drive_root", None)),
        (
            "Ouroboros parent data drive",
            (getattr(ctx, "task_metadata", {}) or {}).get("budget_drive_root")
            if isinstance(getattr(ctx, "task_metadata", {}), dict)
            else "",
        ),
    )
    for label, value in protected_values:
        if not value:
            continue
        try:
            protected = pathlib.Path(value).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            continue
        if (
            path_is_relative_to(workspace, protected)
            or path_is_relative_to(protected, workspace)
            or paths_overlap_casefold(workspace, protected)
        ):
            return f"workspace_root overlaps the {label}"
    return ""


def canonical_data_root(ctx: Any) -> pathlib.Path:
    """Return canonical skill data: task budget → context budget → task drive."""
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    for candidate in (metadata.get("budget_drive_root"), getattr(ctx, "budget_drive_root", "")):
        text = str(candidate or "").strip()
        if text:
            return pathlib.Path(text).resolve(strict=False)
    return pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False)


def normalize_runtime_data_path(data_root: pathlib.Path, path: str) -> str:
    """Normalize historical runtime-data prefixes before physical binding."""
    norm = str(path or ".").strip().replace("\\", "/")
    norm = norm[2:] if norm.startswith("./") else norm
    stripped = norm.lstrip("/")
    root_text = str(pathlib.Path(data_root)).rstrip("/").lstrip("/")
    if root_text and stripped.startswith(root_text):
        return stripped[len(root_text):].lstrip("/") or "."
    if stripped.startswith(".tmp-data-"):
        _prefix, separator, after = stripped.partition("/")
        if separator:
            return after[len("data/"):] if after.startswith("data/") else after
    return norm or "."

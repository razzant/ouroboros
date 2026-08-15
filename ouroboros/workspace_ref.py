"""Dependency-light task/project workspace placement contract (RWS v2 §3.1).

``WorkspaceRef`` is the SINGLE persisted placement descriptor: a sealed,
immutable discriminated union of ``local | ssh`` variants stored in task
metadata and queue snapshots. The executor-facing ``ExecutorRef``
(``ouroboros.workspace_executor``) is a DERIVED projection of this ref and is
never persisted independently, so a placement/executor disagreement is
structurally impossible. The kind ``docker_exec`` is deliberately RESERVED
here: docker execution is an executor-derived projection of a local placement
today; if it ever becomes a persisted placement variant it gets its own
discriminated dataclass, and until then a serialized ``docker_exec`` ref is
rejected loudly rather than half-supported.

A local ref materializes to a real ``pathlib.Path``; an SSH ref has NO Home
path and raises :class:`RemoteWorkspacePathError` when one is requested — the
typed refusal happens at this seam, BEFORE any ``.resolve()``/``.relative_to()``
consumer can misread a remote spelling as a Home path (Appendix C-2, P1).

Legacy durable records (pre-ref task dicts / queue snapshots carrying only a
``workspace_root`` string) normalize ADDITIVELY to the local variant on read
via :func:`normalize_legacy_workspace_ref`.

This module stays dependency-light (stdlib only): transport/execd code may
import the placement contract without pulling Home authorities.
"""

from __future__ import annotations

import dataclasses
import pathlib
import posixpath
from collections.abc import Mapping
from typing import Any, Literal, Union

__all__ = [
    "HOME_NATIVE_ROOTS",
    "LocalWorkspaceRef",
    "RemoteWorkspacePathError",
    "SEALED_WORKSPACE_REF_KEY",
    "SSH_NATIVE_ROOTS",
    "SshWorkspaceRef",
    "WorkspaceRef",
    "is_remote_workspace",
    "isolate_sealed_placement",
    "normalize_legacy_workspace_ref",
    "normalize_remote_root",
    "normalize_remote_root_relative",
    "normalize_workspace_ref",
    "root_is_target_native",
    "workspace_display_root",
    "workspace_ref_for",
]

# The task-metadata / queue-snapshot key the sealed placement rides under.
# Sealed at admission; readers use workspace_ref_for(), nothing re-derives it.
SEALED_WORKSPACE_REF_KEY = "_sealed_workspace_ref"

# Reserved (not yet persisted) discriminator: see the module docstring.
_RESERVED_KINDS = frozenset({"docker_exec"})


class RemoteWorkspacePathError(ValueError):
    """Raised when Home code asks an SSH workspace for a native Home path."""


def _clean_identity(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"workspace_ref.{field} is required")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError(f"workspace_ref.{field} contains a control character")
    return text


def normalize_remote_root(value: Any) -> str:
    """Validate + canonicalize a target-native POSIX workspace root."""

    text = _clean_identity(value, "remote_root").replace("\\", "/")
    if not text.startswith("/"):
        raise ValueError("workspace_ref.remote_root must be an absolute POSIX path")
    if any(part in {".", ".."} for part in text.split("/") if part):
        raise ValueError("workspace_ref.remote_root must not contain traversal segments")
    normalized = posixpath.normpath(text)
    if normalized in {"", ".", "/"}:
        raise ValueError("workspace_ref.remote_root must name a git worktree root")
    return normalized


@dataclasses.dataclass(frozen=True)
class LocalWorkspaceRef:
    """Local placement: the workspace is a directory on the Home host."""

    local_root: str
    kind: Literal["local"] = "local"

    def home_path(self) -> pathlib.Path:
        return pathlib.Path(self.local_root)

    def to_payload(self) -> dict[str, str]:
        return {"kind": self.kind, "local_root": self.local_root}


@dataclasses.dataclass(frozen=True)
class SshWorkspaceRef:
    """SSH placement: the workspace lives on a remote host; no Home path exists."""

    connection_id: str
    remote_root: str
    workspace_id: str
    kind: Literal["ssh"] = "ssh"

    def home_path(self) -> pathlib.Path:
        raise RemoteWorkspacePathError(
            "SSH workspace has no native Home path; route the operation through its executor"
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "connection_id": self.connection_id,
            "remote_root": self.remote_root,
            "workspace_id": self.workspace_id,
        }


WorkspaceRef = Union[LocalWorkspaceRef, SshWorkspaceRef]


# ── Root-placement matrix (RATIFIED Q2а, 2026-07-31) ─────────────────────────
# Only `active_workspace` is SSH-NATIVE: it is the one root whose bytes live on
# the execution target. Every other resource root stays HOME-native — the drive
# roots, the owner's files and the installed skill payloads are Home state, and
# the system repo is Home BY DEFINITION (a remote task must never resolve it).
# Docker is absent from this matrix on purpose: docker execution is an
# executor-derived projection of a LOCAL placement, so all of its roots are Home
# paths that the executor maps, exactly as today.
SSH_NATIVE_ROOTS = frozenset({"active_workspace"})
HOME_NATIVE_ROOTS = frozenset({
    "system_repo",
    "runtime_data",
    "task_drive",
    "artifact_store",
    "user_files",
    "subagent_projects",
    "deliverables",
    "skill_payload",
})


def isolate_sealed_placement(task: Any) -> None:
    """Detach a task's sealed-placement payload from the dict its creator still holds.

    The queue stores a SHALLOW copy of a task, so its ``metadata`` (and the sealed
    payload inside it) stays aliased to the creator's dict. That is how a placement
    becomes retroactively editable: whoever built the task can rewrite the target of a
    task that is already queued. Reads are already safe on their own — every read goes
    through :func:`workspace_ref_for` and returns a FROZEN dataclass — but the payload
    the queue persists must be the creator's LAST word, so it is copied once at
    insertion. The payload is a flat string mapping, so one level of copying is deep.
    """
    if not isinstance(task, dict):
        return
    metadata = task.get("metadata")
    if not isinstance(metadata, Mapping):
        return
    sealed = metadata.get(SEALED_WORKSPACE_REF_KEY)
    if not isinstance(sealed, Mapping):
        return
    task["metadata"] = {**metadata, SEALED_WORKSPACE_REF_KEY: dict(sealed)}


def workspace_display_root(ref: Any) -> str:
    """The placement's OWN spelling of its root, or ``""`` for a workspace-less task.

    A Home path for a local ref, the target-native POSIX root for an ssh ref. This is
    what the owner-visible plumbing shows (task record, prompt block, status
    projections), and it is a STRING on purpose: a remote spelling that reached
    ``pathlib`` is the exact postmortem failure the placement contract exists to remove.
    Presence still means "this is a workspace task" for every string-plumbing consumer;
    the sealed ref decides whether the spelling may become a path.
    """
    if ref is None:
        return ""
    return str(ref.home_path()) if ref.kind == "local" else str(ref.remote_root)


def root_is_target_native(ref: Any, root: Any) -> bool:
    """Whether ``root`` resolves on the EXECUTION TARGET under this placement.

    False for every local/docker placement (all roots are Home paths there) and
    for every Home-native root under an ssh placement.
    """
    if not isinstance(ref, SshWorkspaceRef):
        return False
    return str(root) in SSH_NATIVE_ROOTS


def normalize_remote_root_relative(remote_root: Any, path: Any) -> str:
    """Target-spelling analogue of ``tool_access.normalize_root_relative``.

    Pure ``posixpath``: a target path must never be normalized by Home
    ``pathlib`` — a Windows Home would rewrite separators, and a Home
    ``resolve()`` consults the WRONG filesystem for symlinks and existence.

    Only the ABSOLUTE arm exists here, and only because it is decidable from
    spellings alone: an absolute path already inside the target root becomes
    relative to it, exactly as the Home version does. The Home version's other
    arm — stripping a redundant root-basename prefix — needs the fact "does the
    root contain a same-named subdirectory", and this function runs BEFORE
    prepare, so asking for it would mean a probe issued before the operation
    exists: precisely the per-fact RPC the protocol removes. That arm is
    therefore the TARGET's business, where the path is resolved in its own
    spelling space and a genuinely wrong one is refused rather than silently
    rewritten on Home. Leaving `app/x` unstripped is safe: it simply resolves
    under the root, and the confinement check still catches a real escape.
    """
    text = str(path or "").strip().replace("\\", "/")
    if not text or text in (".", "./"):
        return text
    try:
        root = normalize_remote_root(remote_root)
    except ValueError:
        return text
    if not text.startswith("/"):
        return text
    candidate = posixpath.normpath(text)
    if candidate == root:
        return "."
    prefix = root.rstrip("/") + "/"
    return candidate[len(prefix):] if candidate.startswith(prefix) else text


def normalize_workspace_ref(raw: Any) -> WorkspaceRef | None:
    """Return a sealed normalized ref, or ``None`` for an absent ref.

    Accepts an already-typed ref (pass-through) or a serialized payload dict.
    Unknown fields and unknown kinds fail loudly — a durable record with a
    placement this build cannot honor must never be silently coerced.
    """

    if raw is None or raw == "" or raw == {}:
        return None
    if isinstance(raw, (LocalWorkspaceRef, SshWorkspaceRef)):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("workspace_ref must be an object")
    kind = str(raw.get("kind") or "").strip().lower()
    if kind == "local":
        unknown = set(raw) - {"kind", "local_root"}
        if unknown:
            raise ValueError(f"workspace_ref.local has unknown fields: {sorted(unknown)}")
        root_text = _clean_identity(raw.get("local_root"), "local_root")
        root = pathlib.Path(root_text).expanduser()
        if not root.is_absolute():
            raise ValueError("workspace_ref.local_root must be an absolute path")
        return LocalWorkspaceRef(local_root=str(root.resolve(strict=False)))
    if kind == "ssh":
        unknown = set(raw) - {"kind", "connection_id", "remote_root", "workspace_id"}
        if unknown:
            raise ValueError(f"workspace_ref.ssh has unknown fields: {sorted(unknown)}")
        return SshWorkspaceRef(
            connection_id=_clean_identity(raw.get("connection_id"), "connection_id"),
            remote_root=normalize_remote_root(raw.get("remote_root")),
            workspace_id=_clean_identity(raw.get("workspace_id"), "workspace_id"),
        )
    if kind in _RESERVED_KINDS:
        raise ValueError(
            f"workspace_ref.kind '{kind}' is reserved: docker execution is an "
            "executor-derived projection of a local placement, not a persisted variant"
        )
    raise ValueError("workspace_ref.kind must be 'local' or 'ssh'")


def sealed_child_placement(raw_ref: Any, workspace_root: Any = None) -> dict:
    """The placement fields a CHILD task inherits, ready to splat into its payload.

    A child inherits the parent's SEALED placement, not merely its root spelling.
    Without the ref a remote parent's target spelling normalizes as the LOCAL variant
    in the child — a fabricated Home path, which is precisely the silent fallback the
    placement contract forbids. A child runs where its parent runs.

    Returns ``{}`` for a local parent, so a caller splats this unconditionally and no
    task-creation path has to remember the rule. It lives here rather than at any one
    creation site because there are three of them, and the one that forgets is the one
    that breaks.
    """
    ref = normalize_legacy_workspace_ref(raw_ref, workspace_root)
    return {SEALED_WORKSPACE_REF_KEY: ref.to_payload()} if ref is not None else {}


def normalize_legacy_workspace_ref(raw_ref: Any, workspace_root: Any = None) -> WorkspaceRef | None:
    """Normalize a durable record: explicit ref wins; a bare legacy
    ``workspace_root`` string reads additively as the local variant; neither
    present means no workspace."""

    ref = normalize_workspace_ref(raw_ref)
    if ref is not None:
        return ref
    root_text = str(workspace_root or "").strip()
    if not root_text:
        return None
    return normalize_workspace_ref({"kind": "local", "local_root": root_text})


def _field(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def workspace_ref_for(ctx: Any) -> WorkspaceRef | None:
    """Read the sealed placement from a live context or durable task/result record."""

    raw = None
    for metadata in (_field(ctx, "task_metadata"), _field(ctx, "metadata")):
        if isinstance(metadata, Mapping):
            raw = metadata.get(SEALED_WORKSPACE_REF_KEY)
        if raw not in (None, "", {}):
            break
    return normalize_workspace_ref(raw)


def is_remote_workspace(ctx: Any) -> bool:
    ref = workspace_ref_for(ctx)
    return bool(ref and ref.kind == "ssh")

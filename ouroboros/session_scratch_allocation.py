"""Allocation of one new session scratch container (called directly by ``workspace_preflight._session_scratch``).

Owns the order container -> marker -> empty workspace -> record, the exact rollback of a failed new allocation and the
one private-home placement of a full local in-place run (#4093); identity, records, directory checks and release stay
in ``session_scratch``. Imports it as a module so a collaborator's name is read at call time."""
from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile
from typing import Any, Callable, Dict, Optional, Tuple

from ouroboros import session_scratch as custody
from ouroboros import work_domain


class ScratchRollbackFailed(OSError):
    """Removing a failed new container's own known names failed: nothing may fall back past it."""


def discard(path: pathlib.Path, managed: bool) -> None:
    """Remove only this fresh, unpublished container's two known children, then it, else ``ScratchRollbackFailed``
    (``WorkDomainUnsettled`` propagates intact so an unconfirmed managed rollback is never masked)."""
    try:
        if managed:
            if work_domain.kind(path / custody.SCRATCH_WORKSPACE) == "dir":
                work_domain.remove_empty_dir(path / custody.SCRATCH_WORKSPACE)
            work_domain.remove(path / custody.SCRATCH_MARKER)
            work_domain.remove_empty_dir(path)
        else:
            with contextlib.suppress(FileNotFoundError):
                (path / custody.SCRATCH_WORKSPACE).rmdir()
            with contextlib.suppress(FileNotFoundError):
                (path / custody.SCRATCH_MARKER).unlink()
            path.rmdir()
    except work_domain.WorkDomainUnsettled:
        raise  # an unconfirmed managed removal leaves an unsettled writer; never wrap it as ScratchRollbackFailed
    except OSError as exc:
        raise ScratchRollbackFailed(f"cannot remove the unpublished scratch {path}: {exc}") from exc


def stage_scratch(parent: Optional[pathlib.Path], prefix: str, mode: int, nonce_row: Callable[[pathlib.Path], Dict[str, Any]],
                  execution: pathlib.Path, managed: bool) -> Tuple[pathlib.Path, pathlib.Path, Dict[str, Any]]:
    """One new container -> marker -> empty workspace, NOT yet recorded; a failure removes its own names first."""
    from ouroboros.test_environment import write_scratch_marker

    path = custody.work_session_scratch(execution) if managed else pathlib.Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    try:
        row = nonce_row(path)
        if managed:
            # A work user may differ from the service UID. This new, non-secret marker must be readable
            # through the service's confined open too, without a work-domain cat/link fallback.
            work_domain.write_file(path / custody.SCRATCH_MARKER, (row["nonce"] + "\n").encode("ascii"), mode=0o644)
        else:
            custody._owned_directory(parent, (path.name,), execution)
            write_scratch_marker(path, row["nonce"])
        workspace = custody._workspace(path, mode)
        if managed and (problem := custody._marker_matches(path, row["nonce"])):
            raise ValueError(f"new scratch marker is not readable by its owner: {problem}")
    except work_domain.WorkDomainUnsettled:
        raise  # its writer may still be using this new container; custody must settle it, not a rollback
    except Exception:
        discard(path, managed)
        raise
    return path, workspace, row


def _published(record: pathlib.Path, nonce: str) -> Optional[bool]:
    """Whether RECORD now names this allocation's NONCE: False only when it is absent or names another (proven not
    published), None when it cannot be read or parsed (uncertain: nothing may be removed on that)."""
    try:
        row = json.loads(record.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return None
    return row.get("nonce") == nonce if isinstance(row, dict) else None


def prepare_session_scratch(record: pathlib.Path, execution: pathlib.Path, private: bool = False) -> pathlib.Path:
    """Under the caller's preparation lock: reuse its recorded workspace or allocate a schema-2 container.

    Publish only after container -> marker -> empty workspace. Failed preparation rolls back only those
    known names of this fresh, ungranted allocation, never a previous container. A missing legacy directory
    may allocate anew, but a mismatch never upgrades or adopts it. Restoration with the same nonce at the
    same address deliberately inherits the old holder and cleanup, including files added after restoration.
    A matched private home child whose ``w`` is missing gets it back in the same mode, 0700.
    PRIVATE (a full local in-place run) first tries one ``<physical home>/.obs-<id>`` child (``w`` 0700) when
    ``test_environment.private_scratch_problem`` allows; only an expected filesystem or validation failure (``OSError``,
    ``ValueError``) before the record is written, after its own names are removed, tries the shared parent once. A
    failed removal, an unsettled writer or a failed record write never falls back, and a record write whose outcome
    cannot be read removes nothing.
    """
    from ouroboros.test_environment import (
        PRIVATE_SCRATCH_PREFIX, git_ancestor, new_scratch_identity, private_scratch_home, private_scratch_problem,
        session_scratch_parent,
    )
    from ouroboros.utils import atomic_write_json

    if custody.has_record(record.parent):
        row = custody._scratch_record(record)
        try:
            path = custody._recorded_scratch(record, row)
        except FileNotFoundError:
            pass  # OS aging may have removed it; a literal retry never prepares again
        else:
            if git_ancestor(path) is not None or custody._overlaps(path, execution):
                raise ValueError(f"{path} overlaps a Git working tree or the execution root")
            if row.get("schema_version") != 2:
                return path
            return custody._workspace(path, 0o700 if custody.private_home_child(path) else 0o777)
    managed = work_domain.managed()
    home = None
    if private and not managed:
        with contextlib.suppress(OSError):  # an undecidable home is no private placement
            candidate = private_scratch_home()
            home = candidate if candidate is not None and not private_scratch_problem(candidate) else None
    staged = None
    if home is not None:
        try:
            staged = stage_scratch(home, PRIVATE_SCRATCH_PREFIX, 0o700, new_scratch_identity, execution, False)
        except (ScratchRollbackFailed, work_domain.WorkDomainUnsettled):
            raise
        except (OSError, ValueError):  # expected placement failure; its own names are already removed
            staged = None
    if staged is None:
        shared = pathlib.Path(work_domain.staging_path("scratch")) if managed else session_scratch_parent().resolve()
        if not managed and git_ancestor(shared) is not None:
            raise ValueError(f"{shared} is inside a Git working tree")
        staged = stage_scratch(shared, "obs-", 0o777, new_scratch_identity, execution, managed)
    path, workspace, row = staged
    try:
        atomic_write_json(record, row, fsync=True)
    except work_domain.WorkDomainUnsettled:
        raise
    except Exception:
        if _published(record, row["nonce"]) is False:  # proven unpublished; else the record may name it: keep it
            discard(path, managed)
        raise
    return workspace

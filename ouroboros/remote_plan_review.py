"""Plan review and turn-diff evidence for a REMOTE workspace (RWS v2, D12).

Plan review reads the subject tree: it snapshots the files a plan says it will touch,
it builds a context atlas over them, and the reviewer judges the plan against what is
actually there. All of that is filesystem-shaped and all of it runs on Home, so a
remote subject needs a verified mirror — the same one the edit bridge uses, with the
same policy applied at the source.

Two deliberate departures from the donor:

**No ctx mutation.** The donor cached the snapshot as `ctx._remote_plan_review_snapshot`
and had a decorator delete it in a `finally`, while two unrelated modules read the
attribute — one of them as a boolean "is this a verified snapshot" flag. So the
lifetime of a temporary directory was a property of a mutable attribute on a shared
object, and the answer to "is this review reading a mirror" was whatever that attribute
happened to be. Here the mirror travels as an explicit value: `open_plan_subject`
returns it, the caller owns it (it is a context manager), and every consumer takes it
as a parameter.

**A turn diff that does not export the tree twice.** `review_evidence.collect_turn_diff`
wants the working-tree diff. Materializing a second full mirror for it would double the
transfer on every review tick, and — worse — before this module its `active_repo_dir()`
call raised for a remote placement and fell back to `ctx.repo_dir`, so a remote task's
review evidence showed a diff of the OUROBOROS repository. The answer is the native
`vcs_diff` operation, which computes the diff ON the target and returns text, plus a
short-lived memo so the several evidence consumers in one tick ask once.

D7: a POLICY-FILTERED mirror is reviewed, not refused. The reviewer is told, in the
evidence horizon, exactly which paths the mirror does not contain and why — a plan
about a repo containing one `.env` must not be unreviewable, and a reviewer who is not
told about an omission will reason as if the tree were whole.
"""

from __future__ import annotations

import pathlib
import subprocess
import time
from collections.abc import Mapping
from typing import Any

# Plan review reads the same tree the edit bridge writes to, so it uses the same
# channel: one document per purpose, and a mirror is the purpose here.
PLAN_REVIEW_CHANNEL = "workspace_snapshot"
TURN_DIFF_OPERATION = "vcs_diff"
TURN_DIFF_CHANNEL = "workspace_patch"
# How long one review tick's evidence answer stays reusable. A tick is seconds; this
# is bounded so the memo can never become "the diff from some earlier turn".
_TURN_DIFF_TTL_SEC = 30.0
_TURN_DIFF_CACHE_MAX = 8
_turn_diff_cache: dict[str, tuple[float, str]] = {}
# How many withheld paths the evidence horizon lists before it discloses a remainder.
_OMISSION_ROW_LIMIT = 100


def open_plan_subject(ctx: Any, files_to_touch: Any = ()) -> Any:
    """The verified mirror plan review reads, or ``None`` for a local placement.

    Returned rather than stashed: the caller owns the lifetime (it is a context
    manager), so "is this review reading a mirror" is a value the caller holds instead
    of an attribute on a shared object that something else may have deleted.
    """

    from ouroboros.remote_transfer import materialize_remote_snapshot
    from ouroboros.workspace_ref import is_remote_workspace

    if not is_remote_workspace(ctx):
        return None
    snapshot = materialize_remote_snapshot(ctx, channel=PLAN_REVIEW_CHANNEL)
    try:
        _assert_planned_paths_present(snapshot, files_to_touch)
    except BaseException:
        snapshot.close()
        raise
    return snapshot


def _assert_planned_paths_present(snapshot: Any, files_to_touch: Any) -> None:
    """Refuse a plan whose named paths the mirror cannot show — by name.

    D7's honest half. A path the policy excluded is NOT in the mirror, and a reviewer
    asked to judge a plan about it would be judging an absence. Saying which path and
    that the owner's policy withheld it is actionable; a silent "(new file)" snapshot
    for a file that exists on the target is not.
    """

    from ouroboros.remote_patch_bridge import RemotePatchError

    excluded = {row["path"] for row in snapshot.exclusions()}
    if not excluded:
        return
    named = sorted(
        str(raw).replace("\\", "/").strip("/")
        for raw in files_to_touch or []
        if str(raw or "").strip()
    )
    blocked = [
        path for path in named
        if path in excluded or any(path.startswith(f"{item}/") for item in excluded)
    ]
    if blocked:
        raise RemotePatchError(
            f"the plan names {len(blocked)} path(s) the owner's export policy withheld "
            f"from the reviewed snapshot, so nothing can review a plan about them: "
            f"{blocked[:10]}. They exist on the target; the mirror does not contain them."
        )


def plan_subject_root(snapshot: Any, local_subject: pathlib.Path) -> pathlib.Path:
    """The subject root plan review reads: the mirror when remote, else the local one."""

    return pathlib.Path(snapshot.root) if snapshot is not None else local_subject


def snapshot_omission_rows(snapshot: Any) -> list[dict[str, str]]:
    """The disclosed omissions, in the evidence horizon's row shape.

    An explicit PARAMETER, not a ctx read: the reviewer's omission list and the mirror
    it describes must be the same object, and the donor's version could be asked about
    a snapshot that had already been closed and deleted.

    The row list is bounded, and the bound DISCLOSES ITSELF as a final row: an omission
    list that silently drops omissions is the worst possible place to elide, because the
    reviewer reads exactly this list to learn what it could not see (BIBLE P1).
    """

    if snapshot is None:
        return []
    exclusions = list(snapshot.exclusions())
    rows = [
        {"section": "remote_workspace_snapshot", "path": row["path"], "reason": row["reason"]}
        for row in exclusions[:_OMISSION_ROW_LIMIT]
    ]
    if len(exclusions) > len(rows):
        rows.append({
            "section": "remote_workspace_snapshot",
            "path": f"⚠️ {len(exclusions) - len(rows)} further omission(s) not listed",
            "reason": (
                f"omission list bounded at {_OMISSION_ROW_LIMIT} rows; "
                f"{len(exclusions)} paths were withheld from the mirror in total"
            ),
        })
    return rows


def verified_snapshot_result(
    repo_dir: pathlib.Path, relative_path: str
) -> subprocess.CompletedProcess:
    """Read one confined file out of an ALREADY-VERIFIED mirror.

    Shaped like the `git show HEAD:<path>` result it replaces, so the caller's
    size/binary/sensitive handling stays one code path for both placements. The mirror
    has no git history — it is a materialized tree — so a filesystem read IS the HEAD
    snapshot here, and it is legitimate evidence precisely because the materialization
    verified every byte against the manifest.
    """

    root = pathlib.Path(repo_dir).resolve(strict=False)
    argv = ["verified-filesystem-snapshot", str(relative_path)]
    candidate = (root / str(relative_path)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        # A path escaping the mirror is not a missing file; refusing it as unreadable
        # keeps the caller from rendering content from outside the reviewed tree.
        return subprocess.CompletedProcess(argv, 128, b"", b"path escapes verified snapshot")
    if candidate.is_file():
        return subprocess.CompletedProcess(argv, 0, candidate.read_bytes(), b"")
    return subprocess.CompletedProcess(
        argv, 128, b"", b"path does not exist in verified snapshot"
    )


def remote_turn_diff(ctx: Any, *, limit: int = 20000) -> str:
    """The target's working-tree diff, computed ON the target and memoized.

    Native rather than snapshot-derived on purpose: the diff is what the target's own
    git says, so asking it costs one operation instead of a full tree export, and the
    answer is the same one a local task's `git diff HEAD` gives. The memo exists
    because several evidence consumers ask within one review tick and each ask is a
    round trip; it is bounded in both age and size so it can never answer for an
    earlier turn.
    """

    from ouroboros.remote_transfer import RemoteTransferService
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError
    from ouroboros.workspace_executor import SshExecutorUnavailableError, executor_ref_from_ctx
    from ouroboros.workspace_ref import is_remote_workspace, workspace_ref_for

    if not is_remote_workspace(ctx):
        return ""
    task_id = str(getattr(ctx, "task_id", "") or "")
    now = time.monotonic()
    cached = _turn_diff_cache.get(task_id)
    if cached is not None and now - cached[0] <= _TURN_DIFF_TTL_SEC:
        return cached[1]
    from ouroboros.remote_export_policy import build_export_policy

    ref = workspace_ref_for(ctx)
    policy = build_export_policy(
        ctx, channel=TURN_DIFF_CHANNEL, workspace_root=ref.remote_root
    )
    try:
        envelope = RemoteTransferService().export_operation(
            executor_ref_from_ctx(ctx),
            TURN_DIFF_OPERATION,
            policy.arg_payload(),
            task_id=task_id,
            echo_args=False,
        )
    except (RemoteWorkspaceError, SshExecutorUnavailableError, ValueError):
        # Review evidence is best-effort by contract (`collect_turn_diff` returns ""
        # when there is no diff to be had). What must NOT happen is the pre-existing
        # fallback to `ctx.repo_dir`, which showed a remote task the OUROBOROS repo's
        # diff and labelled it the task's own working tree.
        return ""
    text = str(getattr(envelope, "text", "") or "")
    if len(text) > limit:
        text = text[:limit] + f"\n… remote turn diff truncated at {limit} chars"
    if len(_turn_diff_cache) >= _TURN_DIFF_CACHE_MAX:
        _turn_diff_cache.clear()
    _turn_diff_cache[task_id] = (now, text)
    return text


def forget_remote_turn_diff(ctx_or_task_id: Any) -> None:
    """Drop one task's memoized diff, so the next tick asks the target again."""

    task_id = (
        ctx_or_task_id
        if isinstance(ctx_or_task_id, str)
        else str(getattr(ctx_or_task_id, "task_id", "") or "")
    )
    _turn_diff_cache.pop(task_id, None)


def remote_snapshot_evidence(snapshot: Any) -> dict[str, Any]:
    """The additive D7 block the review evidence carries about the mirror."""

    if snapshot is None:
        return {}
    manifest = snapshot.manifest if isinstance(snapshot.manifest, Mapping) else {}
    return {
        "remote_snapshot": {
            "partial": snapshot.partial,
            "excluded_count": int(manifest.get("policy_excluded_count") or 0),
            "policy_hash": str(manifest.get("policy_hash") or ""),
            "fingerprint": str(manifest.get("fingerprint") or ""),
            "note": snapshot.omission_note(),
        }
    }


__all__ = [
    "PLAN_REVIEW_CHANNEL",
    "TURN_DIFF_CHANNEL",
    "TURN_DIFF_OPERATION",
    "forget_remote_turn_diff",
    "open_plan_subject",
    "plan_subject_root",
    "remote_snapshot_evidence",
    "remote_turn_diff",
    "snapshot_omission_rows",
    "verified_snapshot_result",
]

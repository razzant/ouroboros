"""Applying a Home-decided patch to a remote workspace (RWS v2, D12).

Two faculties need this and they need the same thing. A subagent that worked in a
remote workspace returns a `workspace.patch` its parent must integrate; the Claude
Agent SDK runs on Home, edits a mirror, and its work must reach the target. In both
cases the DECISION is Home's and the MUTATION is the target's, and what connects them
is `workspace_snapshot_native.guarded_patch_apply` — which applies a patch only if the
target still looks exactly like the tree Home reviewed, and rolls back the exact
original bytes otherwise.

That guard needs four things, and the donor sent two. It passed
`expected_fingerprint` and `patch_blob_id` and omitted `changes` and
`expected_content_fingerprint`, so `_validated_changes` raised "changes must be a
list" on every call and the whole remote path was dead on arrival (donor review 2.10).
It is not a missing parameter: `changes` is the guard's precondition contract. Each row
declares the path's state BEFORE (compared field-by-field against the target's own
live snapshot row) and AFTER, and the guard proves the patch touches exactly the
declared paths and nothing else. Without it the guard cannot tell "this patch is for
this tree" from "this patch parses".

So Home computes those preconditions the only way they can be computed honestly: from
a VERIFIED mirror. The before rows are the materialized snapshot's own entries, the
after rows come from re-running the SAME snapshot algorithm over the mirror once the
patch is applied there (`snapshot_workspace`, imported rather than reimplemented — a
second implementation of the fingerprint is a second answer to what the tree is), and
the post-fingerprint Home sends is therefore the fingerprint the target will compute
if, and only if, the patch does on the target what it did on the mirror.

D7 note: the mirror may be POLICY-FILTERED, and that is a normal disclosed state. A
patch that touches an excluded path is refused by the guard — correctly, because the
reviewed mirror never contained it — and the refusal says which path and that it was
excluded, rather than reporting a mysterious precondition failure.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

PATCH_APPLY_OPERATION = "guarded_patch_apply"
# The channel each door's export is judged under (`export_policy_contract`).
SUBAGENT_PATCH_CHANNEL = "subagent_patch"
# A patch bigger than this is not a review artifact any more.
MAX_PATCH_BYTES = 64 * 1024 * 1024
_GIT_TIMEOUT_SEC = 60


class RemotePatchError(RuntimeError):
    """A remote apply that must not proceed, with the reason Home actually has."""

    code = "REMOTE_PATCH_REFUSED"


@dataclasses.dataclass(frozen=True)
class PatchPreconditions:
    """Everything `guarded_patch_apply` needs to know this patch is for this tree."""

    patch: bytes
    changes: list[dict[str, Any]]
    expected_fingerprint: str
    expected_content_fingerprint: str
    expected_head: str
    expected_index_sha256: str
    protected_paths: list[str]

    @property
    def patch_blob_id(self) -> str:
        return hashlib.sha256(self.patch).hexdigest()

    def apply_args(self) -> dict[str, Any]:
        return {
            "expected_fingerprint": self.expected_fingerprint,
            "expected_content_fingerprint": self.expected_content_fingerprint,
            "expected_head": self.expected_head,
            "expected_index_sha256": self.expected_index_sha256,
            "patch_blob_id": self.patch_blob_id,
            # The row the donor omitted, and the reason its remote apply never ran.
            "changes": self.changes,
            "_protected_paths": self.protected_paths,
        }


def _git(mirror: pathlib.Path, argv: Sequence[str], *, allow_failure: bool = False) -> bytes:
    result = subprocess.run(
        ["git", *argv],
        cwd=str(mirror),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=_GIT_TIMEOUT_SEC,
    )
    if result.returncode and not allow_failure:
        raise RemotePatchError(
            f"git {' '.join(argv[:2])} failed in the mirror: "
            f"{result.stderr.decode('utf-8', errors='replace')[:400]}"
        )
    return result.stdout


def open_patch_mirror(ctx: Any, *, channel: str) -> Any:
    """Materialize a verified mirror for ONE channel and make it patchable.

    The single door for both patch faculties, and the reason it exists is that the
    channel must be the same for the snapshot and for the apply: a manifest's
    fingerprint includes the hash of the policy that produced it, so a mirror taken
    under one channel and a patch applied under another is refused by the target with
    a fingerprint mismatch — a correct patch failing for a bookkeeping reason. Binding
    the channel once, here, makes that mismatch unrepresentable.

    The caller owns the returned snapshot (it is a context manager).
    """

    from ouroboros.remote_transfer import materialize_remote_snapshot

    snapshot = materialize_remote_snapshot(ctx, channel=channel)
    try:
        init_mirror_repo(snapshot.root)
    except BaseException:
        snapshot.close()
        raise
    return snapshot


def init_mirror_repo(mirror: pathlib.Path) -> None:
    """Make the verified mirror a throwaway one-commit repo.

    Two reasons, both practical: `git apply` and `git diff` want a work tree, and a
    single baseline commit is what lets a Home-side editor's changes be read back out
    as an exact patch. It is throwaway on purpose — the mirror's git history is never
    the target's, and nothing here is ever pushed anywhere.
    """

    _git(mirror, ["init", "-q"])
    _git(mirror, ["config", "user.name", "Ouroboros Snapshot"])
    _git(mirror, ["config", "user.email", "snapshot@ouroboros.invalid"])
    _git(mirror, ["add", "-A"])
    _git(mirror, ["commit", "-qm", "remote snapshot", "--allow-empty"])


def mirror_patch(mirror: pathlib.Path) -> bytes:
    """The exact binary patch of everything changed in the mirror since baseline."""

    untracked = [
        part.decode("utf-8", errors="surrogateescape")
        for part in _git(mirror, ["ls-files", "-z", "--others", "--exclude-standard"]).split(b"\0")
        if part
    ]
    if untracked:
        # Intent-to-add, so a NEW file appears in the diff instead of being invisible.
        _git(mirror, ["add", "-N", "--", *untracked])
    return _git(
        mirror,
        ["diff", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", "--no-color", "HEAD", "--"],
    )


def assert_mirror_baseline_intact(mirror: pathlib.Path) -> None:
    """Refuse a mirror whose history moved: the diff would no longer be the edit."""

    count = _git(mirror, ["rev-list", "--count", "HEAD"]).strip()
    if count != b"1":
        raise RemotePatchError(
            "the Home mirror's git history moved (a commit was made in it), so the "
            "difference from the baseline is no longer the reviewed edit"
        )


def patch_touched_paths(mirror: pathlib.Path, patch: bytes) -> set[str]:
    """The paths a patch touches, derived with the TARGET's own parser.

    `_patch_numstat_paths` is imported rather than reimplemented for the same reason
    the fingerprint is: the target computes this set to check that the patch touches
    exactly the declared paths, so Home computing it a second way would mean the two
    sides disagree about what a rename record means and the guard would refuse a
    correct patch. Both directions are asked because a pure deletion and a pure
    creation each show up in only one of them.
    """

    from ouroboros.workspace_snapshot_native import _patch_numstat_paths

    try:
        paths = _patch_numstat_paths(mirror, patch, reverse=False)
        paths |= _patch_numstat_paths(mirror, patch, reverse=True)
    except ValueError as exc:
        raise RemotePatchError(str(exc)) from exc
    return {path for path in paths if path}


def snapshot_entry_rows(root: pathlib.Path, document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The snapshot algorithm's OWN entry rows for a Home tree, keyed by path.

    `workspace_snapshot_native.snapshot_workspace` is imported rather than
    reimplemented: it is the function that defines what an entry row and a content
    fingerprint ARE, and computing the target's expected post-state with a second
    implementation would mean Home and the target disagree about the tree the moment
    the two drift.
    """

    from ouroboros.workspace_snapshot_native import snapshot_workspace

    manifest, _blobs = snapshot_workspace(root, policy=dict(document))
    if manifest.get("integrity_complete") is not True or manifest.get("unstable") is not False:
        raise RemotePatchError(
            "the Home mirror could not be observed stably after the edit, so the "
            "post-state Home would attest is not knowable"
        )
    return {
        str(row["path"]): dict(row)
        for row in manifest.get("entries") or []
        if isinstance(row, Mapping)
    }


def content_fingerprint(rows: Mapping[str, Mapping[str, Any]]) -> str:
    """The target's own content fingerprint over a set of entry rows."""

    import json

    entries = [dict(rows[path]) for path in sorted(rows)]
    return hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_preconditions(
    snapshot: Any,
    mirror: pathlib.Path,
    patch: bytes,
    *,
    document: Mapping[str, Any],
) -> PatchPreconditions:
    """Derive the guard's full precondition set from a verified mirror.

    The mirror must ALREADY have the patch applied — the after rows are read from the
    filesystem, not predicted, so what Home attests is what a real application of this
    patch produced on a tree that was proven identical to the target's.
    """

    if not patch:
        raise RemotePatchError("the patch is empty; there is nothing to apply")
    if len(patch) > MAX_PATCH_BYTES:
        raise RemotePatchError(
            f"the patch is {len(patch)} bytes, over the {MAX_PATCH_BYTES}-byte remote apply limit"
        )
    manifest = dict(snapshot.manifest)
    before_rows = {
        str(row["path"]): dict(row)
        for row in manifest.get("entries") or []
        if isinstance(row, Mapping)
    }
    after_rows = snapshot_entry_rows(mirror, document)
    touched = patch_touched_paths(mirror, patch)
    if not touched:
        raise RemotePatchError("git could not name the paths this patch touches")
    excluded = {row["path"] for row in snapshot.exclusions()}
    blocked = sorted(
        path for path in touched
        if path in excluded or any(path.startswith(f"{item}/") for item in excluded)
    )
    if blocked:
        # An HONEST refusal (D7): the reviewed mirror never contained these paths, so
        # a change against them was never reviewed. Saying WHICH paths and that the
        # policy excluded them is the difference between an actionable refusal and a
        # mysterious precondition failure.
        raise RemotePatchError(
            f"the patch touches {len(blocked)} path(s) the owner's export policy excluded "
            f"from the reviewed snapshot, so the change was never reviewed: {blocked[:10]}. "
            "Nothing was applied."
        )
    changes = [
        {"path": path, "before": before_rows.get(path), "after": after_rows.get(path)}
        for path in sorted(touched)
    ]
    git_facts = manifest.get("git") if isinstance(manifest.get("git"), Mapping) else {}
    return PatchPreconditions(
        patch=patch,
        changes=changes,
        expected_fingerprint=str(manifest.get("fingerprint") or ""),
        expected_content_fingerprint=content_fingerprint(after_rows),
        expected_head=str(git_facts.get("head") or ""),
        expected_index_sha256=str(git_facts.get("index_sha256") or ""),
        protected_paths=[str(item) for item in manifest.get("protected_paths") or []],
    )


def apply_on_target(
    ctx: Any,
    preconditions: PatchPreconditions,
    *,
    channel: str,
) -> dict[str, Any]:
    """Send the patch and its preconditions to the target; return the apply trace.

    The blob moves through the transfer service's one export door, and the target
    revalidates everything Home attested: the live fingerprint, HEAD, the index, each
    change row's before-state, and that the patch touches exactly the declared paths.
    Home deciding and the target proving is the whole shape — a Home-side "it applied
    cleanly on the mirror" is evidence about the mirror, not about the target.
    """

    from ouroboros.remote_export_policy import build_export_policy
    from ouroboros.remote_transfer import RemoteTransferService
    from ouroboros.workspace_executor import executor_ref_from_ctx
    from ouroboros.workspace_ref import workspace_ref_for

    ref = workspace_ref_for(ctx)
    if getattr(ref, "kind", "") != "ssh":
        raise RemotePatchError("a remote patch apply requires a sealed ssh placement")
    policy = build_export_policy(ctx, channel=channel, workspace_root=ref.remote_root)
    envelope = RemoteTransferService().export_operation(
        executor_ref_from_ctx(ctx),
        PATCH_APPLY_OPERATION,
        {**preconditions.apply_args(), **policy.arg_payload()},
        blobs={preconditions.patch_blob_id: preconditions.patch},
        task_id=str(getattr(ctx, "task_id", "") or ""),
        # The target normalizes and re-derives the change rows it will enforce, so
        # echoing Home's request back would compare a request with its own validation.
        echo_args=False,
    )
    trace = getattr(envelope, "trace", None)
    trace = dict(trace) if isinstance(trace, Mapping) else {}
    if str(trace.get("completion") or "") != "complete":
        raise RemotePatchError(
            f"the target refused the guarded apply "
            f"(completion={trace.get('completion') or 'unknown'}): "
            f"{str(getattr(envelope, 'text', '') or '')[:600]}"
        )
    return trace


__all__ = [
    "MAX_PATCH_BYTES",
    "PATCH_APPLY_OPERATION",
    "SUBAGENT_PATCH_CHANNEL",
    "PatchPreconditions",
    "RemotePatchError",
    "apply_on_target",
    "assert_mirror_baseline_intact",
    "build_preconditions",
    "content_fingerprint",
    "init_mirror_repo",
    "mirror_patch",
    "open_patch_mirror",
    "patch_touched_paths",
    "snapshot_entry_rows",
]

"""Home-decided patches reaching a remote workspace (D12 channel `subagent_patch`),
plus the Home mirror they are decided on.

The donor's remote apply sent `expected_fingerprint` and `patch_blob_id` and omitted
`changes`, so `guarded_patch_apply` raised "changes must be a list" on every call and
the path was dead (donor review 2.10). These tests pin the full precondition set,
the post-state fingerprint Home attests, and the two refusals that matter: a diverged
fingerprint, and a patch that touches a path the export policy excluded from the
reviewed mirror.

The target here is the REAL `workspace_snapshot_native` — the same `snapshot_workspace`
and `guarded_patch_apply` that run on execd — over a real git worktree. Only the wire
is stubbed, so what is being tested is the actual contract between the two sides.
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest

from ouroboros.remote_patch_bridge import (
    RemotePatchError,
    build_preconditions,
    content_fingerprint,
    mirror_patch,
    open_patch_mirror,
    patch_touched_paths,
    snapshot_entry_rows,
)
from ouroboros.remote_transfer import (
    RemoteSnapshotError,
    RemoteWorkspaceSnapshot,
    materialize_remote_snapshot,
)
from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY
from ouroboros.workspace_snapshot_native import guarded_patch_apply, snapshot_workspace


def _git(root: pathlib.Path, *argv: str) -> None:
    subprocess.run(
        ["git", *argv], cwd=str(root), check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


@pytest.fixture
def target_repo(tmp_path):
    """A real git worktree standing in for the remote workspace."""
    root = tmp_path / "srv" / "app"
    root.mkdir(parents=True)
    (root / "app.py").write_text("print('one')\n", encoding="utf-8")
    (root / "keep").mkdir()
    (root / "keep" / "notes.md").write_text("notes\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.invalid")
    _git(root, "config", "user.name", "T")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


class _Ctx:
    def __init__(self, tmp_path, target_repo):
        self.task_id = "task-patch"
        self.drive_root = tmp_path / "drive"
        self.repo_dir = tmp_path / "repo"
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.task_metadata = {
            SEALED_WORKSPACE_REF_KEY: {
                "kind": "ssh", "connection_id": "conn-1",
                "remote_root": str(target_repo), "workspace_id": "ws-1",
            }
        }


class _Target:
    """Prepare/execute against the REAL native snapshot + guarded-apply kernel."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        self.applied: list[dict] = []
        self.last_envelope = None

    def prepare(self, ref, *, tool, args, blobs, task_id, **_kw):
        from ouroboros.remote_workspace import PreparedRemoteCall

        self._args, self._blobs = dict(args), dict(blobs)
        return PreparedRemoteCall(
            request_id="req", operation_id=f"op-{tool}", tool=tool,
            prepared_token="tok", prepared_hash="0" * 64, expires_at_ms=1 << 62,
            execution_args={}, native_facts={
                "export_policy": args.get("_export_policy") or {},
            },
        )

    def execute_prepared(self, ref, prepared, *, canonical_args, task_id, **_kw):
        from ouroboros.remote_worker_proxy import envelope_from_dict
        from ouroboros.workspace_snapshot_native import snapshot_operation

        policy = self._args.get("_export_policy") or None
        if prepared.tool == "snapshot_manifest_and_blob_export":
            result = snapshot_operation(self.root, policy=policy)
            self._snapshot_blobs = dict(result.blobs)
            envelope = result.envelope
        else:
            envelope = guarded_patch_apply(
                self.root, self._args, self._blobs,
                protected_paths=tuple(self._args.get("_protected_paths") or ()),
                policy=policy,
            )
            self.applied.append(dict(self._args))
        self.last_envelope = envelope
        return envelope_from_dict({
            "text": envelope.text, "diagnostic": None, "process": None,
            "artifacts": [dict(row) for row in envelope.artifacts or ()],
            "trace": dict(envelope.trace or {}),
        })

    def fetch_blob(self, ref, blob_id, *, max_bytes, task_id):
        return self._snapshot_blobs[blob_id]

    def abort_prepared(self, *_a, **_kw):
        return True


@pytest.fixture
def remote(tmp_path, target_repo, monkeypatch):
    ctx = _Ctx(tmp_path, target_repo)
    target = _Target(target_repo)
    monkeypatch.setattr(
        "ouroboros.workspace_executor._remote_service", lambda executor, phase: target
    )
    return ctx, target, target_repo


# ── the mirror is evidence, not a copy ───────────────────────────────────────
def test_the_mirror_is_verified_against_the_manifest(remote):
    ctx, _target, target_repo = remote
    with materialize_remote_snapshot(ctx) as snapshot:
        assert isinstance(snapshot, RemoteWorkspaceSnapshot)
        assert (snapshot.root / "app.py").read_text() == "print('one')\n"
        assert (snapshot.root / "keep" / "notes.md").read_text() == "notes\n"
        # `.git` is an excluded directory, so the mirror is a working tree, not a repo.
        assert not (snapshot.root / ".git").exists()
        assert snapshot.partial is False and snapshot.omission_note() == ""
        cleanup = snapshot._cleanup_root
    assert not cleanup.exists()


def test_a_policy_filtered_mirror_is_materialized_and_says_what_is_missing(remote):
    ctx, _target, target_repo = remote
    (target_repo / ".env").write_text("TOKEN=1\n", encoding="utf-8")

    with materialize_remote_snapshot(ctx) as snapshot:
        # D7: a `.env` in a remote repo must NOT take the mirror down with it.
        assert snapshot.partial is True
        assert not (snapshot.root / ".env").exists()
        assert [row["path"] for row in snapshot.exclusions()] == [".env"]
        note = snapshot.omission_note()
    assert "POLICY-FILTERED" in note and ".env" in note


def test_an_integrity_failure_is_still_fail_closed(remote, monkeypatch):
    ctx, target, _root = remote
    real = snapshot_workspace

    def unstable(root, **kwargs):
        manifest, blobs = real(root, **kwargs)
        manifest["unstable"] = True
        return manifest, blobs

    monkeypatch.setattr("ouroboros.workspace_snapshot_native.snapshot_workspace", unstable)
    with pytest.raises(RemoteSnapshotError) as excinfo:
        materialize_remote_snapshot(ctx)
    assert "unstable" in str(excinfo.value)


# ── the preconditions the donor omitted ──────────────────────────────────────
def _edit_mirror(mirror: pathlib.Path) -> bytes:
    (mirror / "app.py").write_text("print('two')\n", encoding="utf-8")
    (mirror / "added.txt").write_text("new\n", encoding="utf-8")
    return mirror_patch(mirror)


def test_the_precondition_set_is_complete_and_the_target_accepts_it(remote):
    from ouroboros.remote_patch_bridge import SUBAGENT_PATCH_CHANNEL, open_patch_mirror

    ctx, target, target_repo = remote
    with open_patch_mirror(ctx, channel=SUBAGENT_PATCH_CHANNEL) as snapshot:
        patch = _edit_mirror(snapshot.root)
        pre = build_preconditions(
            snapshot, snapshot.root, patch, document=snapshot.policy_document
        )
        args = pre.apply_args()
        # All four: the donor sent the first two and omitted the rest.
        assert args["expected_fingerprint"] == snapshot.manifest["fingerprint"]
        assert args["patch_blob_id"] == pre.patch_blob_id
        assert args["expected_content_fingerprint"] and len(args["expected_content_fingerprint"]) == 64
        assert {row["path"] for row in args["changes"]} == {"app.py", "added.txt"}
        # An edited path declares its exact before-state; a new one declares None.
        by_path = {row["path"]: row for row in args["changes"]}
        assert by_path["app.py"]["before"]["sha256"] and by_path["app.py"]["after"]["sha256"]
        assert by_path["app.py"]["before"]["sha256"] != by_path["app.py"]["after"]["sha256"]
        assert by_path["added.txt"]["before"] is None
        assert by_path["added.txt"]["after"]["size"] == len("new\n")

        from ouroboros.remote_patch_bridge import SUBAGENT_PATCH_CHANNEL, apply_on_target

        trace = apply_on_target(ctx, pre, channel=SUBAGENT_PATCH_CHANNEL)

    assert trace["completion"] == "complete" and trace["changed"] == 2
    assert (target_repo / "app.py").read_text() == "print('two')\n"
    assert (target_repo / "added.txt").read_text() == "new\n"
    # The post-state fingerprint Home ATTESTED is the one the target computed.
    assert trace["after"]["content_fingerprint"] == pre.expected_content_fingerprint


def test_the_post_fingerprint_is_the_targets_own_arithmetic(remote):
    """Home computes the after fingerprint with the target's own algorithm, not a
    parallel implementation — a second implementation is a second answer."""
    from ouroboros.remote_patch_bridge import SUBAGENT_PATCH_CHANNEL, open_patch_mirror

    ctx, _target, target_repo = remote
    with open_patch_mirror(ctx, channel=SUBAGENT_PATCH_CHANNEL) as snapshot:
        patch = _edit_mirror(snapshot.root)
        rows = snapshot_entry_rows(snapshot.root, snapshot.policy_document)
        assert content_fingerprint(rows) == build_preconditions(
            snapshot, snapshot.root, patch, document=snapshot.policy_document
        ).expected_content_fingerprint


def test_a_diverged_fingerprint_is_refused_and_nothing_is_applied(remote):
    from ouroboros.remote_patch_bridge import SUBAGENT_PATCH_CHANNEL, open_patch_mirror

    ctx, target, target_repo = remote
    with open_patch_mirror(ctx, channel=SUBAGENT_PATCH_CHANNEL) as snapshot:
        patch = _edit_mirror(snapshot.root)
        pre = build_preconditions(
            snapshot, snapshot.root, patch, document=snapshot.policy_document
        )
        # Somebody else touched the target between review and apply.
        (target_repo / "keep" / "notes.md").write_text("moved on\n", encoding="utf-8")

        from ouroboros.remote_patch_bridge import SUBAGENT_PATCH_CHANNEL, apply_on_target

        with pytest.raises(RemotePatchError) as excinfo:
            apply_on_target(ctx, pre, channel=SUBAGENT_PATCH_CHANNEL)

    assert "refused the guarded apply" in str(excinfo.value)
    assert (target_repo / "app.py").read_text() == "print('one')\n"
    assert not (target_repo / "added.txt").exists()


def test_a_patch_touching_an_excluded_path_is_refused_by_name(remote):
    ctx, _target, target_repo = remote
    (target_repo / ".env").write_text("TOKEN=1\n", encoding="utf-8")
    with open_patch_mirror(ctx, channel="subagent_patch") as snapshot:
        # The mirror never contained `.env`, so a change against it was never reviewed.
        (snapshot.root / ".env").write_text("TOKEN=2\n", encoding="utf-8")
        patch = mirror_patch(snapshot.root)
        with pytest.raises(RemotePatchError) as excinfo:
            build_preconditions(
                snapshot, snapshot.root, patch, document=snapshot.policy_document
            )
    message = str(excinfo.value)
    assert ".env" in message and "never reviewed" in message
    assert target_repo.joinpath(".env").read_text() == "TOKEN=1\n"


def test_an_empty_or_oversized_patch_is_refused_before_anything_moves(remote):
    ctx, _target, _root = remote
    with open_patch_mirror(ctx, channel="subagent_patch") as snapshot:
        with pytest.raises(RemotePatchError) as excinfo:
            build_preconditions(snapshot, snapshot.root, b"", document=snapshot.policy_document)
    assert "nothing to apply" in str(excinfo.value)


def test_a_moved_mirror_baseline_is_refused(remote):
    from ouroboros.remote_patch_bridge import assert_mirror_baseline_intact

    ctx, _target, _root = remote
    with open_patch_mirror(ctx, channel="subagent_patch") as snapshot:
        assert_mirror_baseline_intact(snapshot.root)
        (snapshot.root / "app.py").write_text("x\n", encoding="utf-8")
        _git(snapshot.root, "commit", "-aqm", "second")
        with pytest.raises(RemotePatchError) as excinfo:
            assert_mirror_baseline_intact(snapshot.root)
    assert "history moved" in str(excinfo.value)


def test_touched_paths_cover_creation_and_deletion(remote):
    ctx, _target, _root = remote
    with open_patch_mirror(ctx, channel="subagent_patch") as snapshot:
        (snapshot.root / "keep" / "notes.md").unlink()
        (snapshot.root / "fresh.txt").write_text("f\n", encoding="utf-8")
        patch = mirror_patch(snapshot.root)
        assert patch_touched_paths(snapshot.root, patch) == {"keep/notes.md", "fresh.txt"}


# ── the parent's integration tool on a remote workspace ──────────────────────
def _child_patch_artifacts(drive_root: pathlib.Path, child: str, patch: bytes) -> None:
    """Write the artifacts headless finalization would leave for a child's patch."""
    import hashlib
    import json

    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.headless import ARTIFACT_STATUS_READY_WITH_CHANGES

    art = task_artifact_dir_path(drive_root, child, create=True)
    (art / "workspace.patch").write_bytes(patch)
    (art / "workspace_patch.json").write_text(
        json.dumps({
            "status": ARTIFACT_STATUS_READY_WITH_CHANGES,
            "sha256": hashlib.sha256(patch).hexdigest(),
            "tracked_changed": ["app.py"],
            "untracked_included": ["added.txt"],
            "diffstat": " 2 files changed",
        }),
        encoding="utf-8",
    )


def _child_result(drive_root: pathlib.Path, child: str, parent: str) -> None:
    from ouroboros.task_results import task_result_path
    from ouroboros.utils import atomic_write_json

    atomic_write_json(
        task_result_path(pathlib.Path(drive_root), child, create=True),
        {"task_id": child, "parent_task_id": parent, "status": "completed"},
    )


def test_a_child_patch_is_integrated_into_the_remote_workspace(remote, tmp_path):
    from ouroboros.remote_patch_bridge import SUBAGENT_PATCH_CHANNEL, open_patch_mirror
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    ctx, target, target_repo = remote
    # Produce a realistic child patch by editing a mirror of the same tree.
    with open_patch_mirror(ctx, channel=SUBAGENT_PATCH_CHANNEL) as snapshot:
        patch = _edit_mirror(snapshot.root)
    _child_patch_artifacts(ctx.drive_root, "child01", patch)
    _child_result(ctx.drive_root, "child01", ctx.task_id)

    message = _integrate_subagent_patch(ctx, task_id="child01", decision="apply")

    assert message.startswith("✅ Applied subagent patch from child01")
    assert "REMOTE workspace" in message
    assert (target_repo / "app.py").read_text() == "print('two')\n"
    assert (target_repo / "added.txt").read_text() == "new\n"
    # HEAD and the post-state content fingerprint are BOTH named in the answer, so
    # the owner can reconcile the claim against the target's own trace.
    assert "content fingerprint" in message


def test_a_child_patch_that_no_longer_applies_is_refused_before_the_target(remote, tmp_path):
    """The tool re-materializes at integration time, so a patch written against an
    older tree is caught ON THE MIRROR — nothing is offered to the target at all."""
    from ouroboros.remote_patch_bridge import SUBAGENT_PATCH_CHANNEL, open_patch_mirror
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    ctx, target, target_repo = remote
    with open_patch_mirror(ctx, channel=SUBAGENT_PATCH_CHANNEL) as snapshot:
        patch = _edit_mirror(snapshot.root)
    _child_patch_artifacts(ctx.drive_root, "child02", patch)
    _child_result(ctx.drive_root, "child02", ctx.task_id)
    # The same line the child rewrote has moved on independently.
    (target_repo / "app.py").write_text("print('somebody else')\n", encoding="utf-8")

    message = _integrate_subagent_patch(ctx, task_id="child02", decision="apply")

    assert message.startswith("⚠️ INTEGRATE_PATCH_CONFLICT")
    assert target.applied == []
    assert (target_repo / "app.py").read_text() == "print('somebody else')\n"
    assert not (target_repo / "added.txt").exists()


def test_a_target_that_moves_after_review_is_refused_by_the_guard(remote, tmp_path):
    """The fingerprint divergence that MATTERS: between Home's decision and the apply."""
    from ouroboros.remote_patch_bridge import SUBAGENT_PATCH_CHANNEL, open_patch_mirror
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    ctx, target, target_repo = remote
    with open_patch_mirror(ctx, channel=SUBAGENT_PATCH_CHANNEL) as snapshot:
        patch = _edit_mirror(snapshot.root)
    _child_patch_artifacts(ctx.drive_root, "child04", patch)
    _child_result(ctx.drive_root, "child04", ctx.task_id)

    # Mutate the target in the window between the mirror export and the apply.
    real_prepare = target.prepare

    def racing_prepare(ref, *, tool, args, blobs, task_id, **kw):
        prepared = real_prepare(ref, tool=tool, args=args, blobs=blobs, task_id=task_id, **kw)
        if tool == "guarded_patch_apply":
            (target_repo / "keep" / "notes.md").write_text("moved on\n", encoding="utf-8")
        return prepared

    target.prepare = racing_prepare

    message = _integrate_subagent_patch(ctx, task_id="child04", decision="apply")

    assert message.startswith("⚠️ INTEGRATE_REMOTE_REFUSED")
    assert "SNAPSHOT_FINGERPRINT_MISMATCH" in message
    assert (target_repo / "app.py").read_text() == "print('one')\n"
    assert not (target_repo / "added.txt").exists()


def test_a_home_target_root_is_refused_for_a_remote_task(remote, tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    ctx, _target, _root = remote
    _child_patch_artifacts(ctx.drive_root, "child03", b"diff --git a/x b/x\n")
    _child_result(ctx.drive_root, "child03", ctx.task_id)

    message = _integrate_subagent_patch(
        ctx, task_id="child03", decision="apply", target_root=str(tmp_path),
    )
    assert message.startswith("⚠️ INTEGRATE_TARGET_FORBIDDEN")
    assert "remote host" in message

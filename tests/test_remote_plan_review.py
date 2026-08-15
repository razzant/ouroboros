"""Plan review and turn-diff evidence over a REMOTE workspace (D12).

Three properties, each of which was a real defect before:

* the subject the reviewer reads is the VERIFIED mirror, not a Home path and not the
  Ouroboros checkout;
* the mirror's lifetime is an explicit value the caller owns — the donor stashed it on
  `ctx` where a decorator deleted it and two unrelated modules read it;
* the turn diff comes from the TARGET. `collect_turn_diff` used to fall back to
  `ctx.repo_dir` when `active_repo_dir()` refused, so a remote task's review evidence
  was a diff of the Ouroboros repository presented as the task's own working tree.

Plus D7: a policy-filtered mirror is reviewed with the omission named, and a plan that
names a withheld path is refused by name rather than reviewed against an absence.
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest

from ouroboros.remote_patch_bridge import RemotePatchError
from ouroboros.remote_plan_review import (
    PLAN_REVIEW_CHANNEL,
    forget_remote_turn_diff,
    open_plan_subject,
    plan_subject_root,
    remote_snapshot_evidence,
    remote_turn_diff,
    snapshot_omission_rows,
    verified_snapshot_result,
)
from ouroboros.review_evidence import collect_turn_diff
from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY


def _git(root: pathlib.Path, *argv: str) -> None:
    subprocess.run(
        ["git", *argv], cwd=str(root), check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


@pytest.fixture
def target_repo(tmp_path):
    root = tmp_path / "srv" / "app"
    root.mkdir(parents=True)
    (root / "app.py").write_text("print('one')\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.invalid")
    _git(root, "config", "user.name", "T")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


class _Ctx:
    def __init__(self, tmp_path, target_repo, *, remote=True):
        self.task_id = "task-plan"
        self.drive_root = tmp_path / "drive"
        self.repo_dir = tmp_path / "home-repo"
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.system_repo_dir = self.repo_dir
        self.task_metadata = {}
        if remote:
            self.task_metadata[SEALED_WORKSPACE_REF_KEY] = {
                "kind": "ssh", "connection_id": "conn-1",
                "remote_root": str(target_repo), "workspace_id": "ws-1",
            }


class _Target:
    def __init__(self, root):
        self.root = root
        self.operations: list[str] = []

    def prepare(self, ref, *, tool, args, blobs, task_id, **_kw):
        from ouroboros.remote_workspace import PreparedRemoteCall

        self._args = dict(args)
        self.operations.append(tool)
        return PreparedRemoteCall(
            request_id="r", operation_id=f"op-{tool}", tool=tool,
            prepared_token="tok", prepared_hash="0" * 64, expires_at_ms=1 << 62,
            execution_args={}, native_facts={"export_policy": args.get("_export_policy") or {}},
        )

    def execute_prepared(self, ref, prepared, *, canonical_args, task_id, **_kw):
        from ouroboros.remote_worker_proxy import envelope_from_dict
        from ouroboros.workspace_snapshot_native import snapshot_operation

        if prepared.tool == "snapshot_manifest_and_blob_export":
            result = snapshot_operation(self.root, policy=self._args.get("_export_policy"))
            self._blobs = dict(result.blobs)
            envelope = result.envelope
            return envelope_from_dict({
                "text": envelope.text, "diagnostic": None, "process": None,
                "artifacts": [dict(r) for r in envelope.artifacts or ()],
                "trace": dict(envelope.trace or {}),
            })
        diff = subprocess.run(
            ["git", "diff", "--no-color", "HEAD"], cwd=str(self.root),
            capture_output=True, text=True, timeout=20,
        ).stdout
        return envelope_from_dict({
            "text": diff, "diagnostic": None, "process": None,
            "artifacts": [], "trace": {"completion": "complete"},
        })

    def fetch_blob(self, ref, blob_id, *, max_bytes, task_id):
        return self._blobs[blob_id]

    def abort_prepared(self, *_a, **_kw):
        return True


@pytest.fixture
def remote(tmp_path, target_repo, monkeypatch):
    ctx = _Ctx(tmp_path, target_repo)
    target = _Target(target_repo)
    monkeypatch.setattr(
        "ouroboros.workspace_executor._remote_service", lambda executor, phase: target
    )
    forget_remote_turn_diff(ctx)
    yield ctx, target, target_repo
    forget_remote_turn_diff(ctx)


# ── the subject the reviewer reads ───────────────────────────────────────────
def test_the_plan_subject_is_the_verified_mirror(remote):
    ctx, _target, target_repo = remote
    snapshot = open_plan_subject(ctx, ["app.py"])
    try:
        assert snapshot is not None
        root = plan_subject_root(snapshot, pathlib.Path(ctx.repo_dir))
        assert root == snapshot.root != pathlib.Path(ctx.repo_dir)
        assert (root / "app.py").read_text() == "print('one')\n"
    finally:
        snapshot.close()
    # The caller owns the lifetime, so closing it really removes the mirror.
    assert not snapshot.root.exists()


def test_a_local_placement_opens_no_mirror(tmp_path, target_repo):
    ctx = _Ctx(tmp_path, target_repo, remote=False)
    assert open_plan_subject(ctx, ["app.py"]) is None
    local = pathlib.Path(ctx.repo_dir)
    assert plan_subject_root(None, local) == local
    assert snapshot_omission_rows(None) == []
    assert remote_snapshot_evidence(None) == {}


def test_the_head_snapshot_of_a_mirror_is_a_verified_filesystem_read(remote):
    ctx, _target, _root = remote
    snapshot = open_plan_subject(ctx, ["app.py"])
    try:
        # The mirror has no git history, so `git show HEAD:` cannot answer; the read is
        # legitimate evidence because materialization verified every byte.
        result = verified_snapshot_result(snapshot.root, "app.py")
        assert result.returncode == 0 and result.stdout == b"print('one')\n"
        missing = verified_snapshot_result(snapshot.root, "nope.py")
        assert missing.returncode == 128 and b"does not exist" in missing.stderr
        escaped = verified_snapshot_result(snapshot.root, "../outside.py")
        assert escaped.returncode == 128 and b"escapes" in escaped.stderr
    finally:
        snapshot.close()


def test_the_review_helper_reads_the_mirror_when_told_to(remote):
    from ouroboros.tools.review_helpers import build_head_snapshot_section

    ctx, _target, _root = remote
    snapshot = open_plan_subject(ctx, ["app.py"])
    try:
        section, included = build_head_snapshot_section(
            snapshot.root, ["app.py"], verified_filesystem_snapshot=True
        )
        assert "print('one')" in section
        assert included == frozenset({"app.py"})
    finally:
        snapshot.close()


# ── D7: a filtered mirror is reviewed, with the omission named ───────────────
def test_a_filtered_mirror_is_reviewed_and_the_omission_reaches_the_reviewer(remote):
    ctx, _target, target_repo = remote
    (target_repo / ".env").write_text("TOKEN=1\n", encoding="utf-8")
    snapshot = open_plan_subject(ctx, ["app.py"])
    try:
        # A `.env` must NOT make the plan unreviewable (the donor's fail-closed
        # conflation did exactly that).
        assert snapshot is not None and snapshot.partial is True
        rows = snapshot_omission_rows(snapshot)
        assert rows == [{"section": "remote_workspace_snapshot", "path": ".env",
                         "reason": "sensitive_file"}]
        evidence = remote_snapshot_evidence(snapshot)["remote_snapshot"]
        assert evidence["partial"] is True and evidence["excluded_count"] == 1
        assert "POLICY-FILTERED" in evidence["note"]
    finally:
        snapshot.close()


def test_a_plan_naming_a_withheld_path_is_refused_by_name(remote):
    ctx, _target, target_repo = remote
    (target_repo / ".env").write_text("TOKEN=1\n", encoding="utf-8")
    with pytest.raises(RemotePatchError) as excinfo:
        open_plan_subject(ctx, ["app.py", ".env"])
    message = str(excinfo.value)
    assert ".env" in message and "withheld" in message
    assert "exist on the target" in message


# ── the turn diff comes from the target, once per tick ────────────────────────
def test_the_turn_diff_is_the_targets_own_diff(remote):
    ctx, target, target_repo = remote
    (target_repo / "app.py").write_text("print('changed')\n", encoding="utf-8")

    diff = remote_turn_diff(ctx)

    assert "print('changed')" in diff and "app.py" in diff
    assert target.operations == ["vcs_diff"]


def test_one_review_tick_asks_the_target_once(remote):
    ctx, target, target_repo = remote
    (target_repo / "app.py").write_text("print('changed')\n", encoding="utf-8")

    first = remote_turn_diff(ctx)
    second = remote_turn_diff(ctx)

    assert first == second
    # A full snapshot export per evidence consumer is what the memo prevents; a native
    # diff is one operation, and asking twice in one tick is still one.
    assert target.operations == ["vcs_diff"]
    forget_remote_turn_diff(ctx)
    remote_turn_diff(ctx)
    assert target.operations == ["vcs_diff", "vcs_diff"]


def test_collect_turn_diff_no_longer_shows_the_ouroboros_repo(remote):
    ctx, target, target_repo = remote
    # Make the HOME repo dirty in a way that would show up under the old fallback.
    home = pathlib.Path(ctx.repo_dir)
    _git(home, "init", "-q")
    _git(home, "config", "user.email", "h@h.invalid")
    _git(home, "config", "user.name", "H")
    (home / "home_only.py").write_text("HOME ONLY\n", encoding="utf-8")
    _git(home, "add", "-A")
    _git(home, "commit", "-qm", "home base")
    (home / "home_only.py").write_text("HOME ONLY CHANGED\n", encoding="utf-8")
    (target_repo / "app.py").write_text("print('changed')\n", encoding="utf-8")

    diff = collect_turn_diff(ctx)

    assert "print('changed')" in diff
    assert "home_only" not in diff
    assert target.operations == ["vcs_diff"]


def test_an_unavailable_transport_yields_no_evidence_rather_than_the_wrong_one(
    tmp_path, target_repo, monkeypatch
):
    ctx = _Ctx(tmp_path, target_repo)
    forget_remote_turn_diff(ctx)

    def unavailable(executor, phase):
        from ouroboros.workspace_executor import SshExecutorUnavailableError

        raise SshExecutorUnavailableError("no broker in this process")

    monkeypatch.setattr("ouroboros.workspace_executor._remote_service", unavailable)
    # Best-effort by contract — but the answer is "" and never Home's own diff.
    assert collect_turn_diff(ctx) == ""
    forget_remote_turn_diff(ctx)


def test_the_channel_is_the_snapshot_channel(remote):
    ctx, target, _root = remote
    snapshot = open_plan_subject(ctx, [])
    try:
        assert snapshot.policy_document["channel"] == PLAN_REVIEW_CHANNEL
    finally:
        snapshot.close()

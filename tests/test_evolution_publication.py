"""Publishing an evolution commit: the orphan ref, the git lock, and the authority to promote.

Split out of ``tests/test_evolution_state_integrity_v3.py`` by theme: the orphan ref a later
normal push cannot publish and its safe CAS fallback, the review refused when the claim is
gone, the post-commit CAS and binding failures, the push that alone stays under the git
lock, the revoked publication that anchors nothing, and the exact claim a promote carries.
"""

from __future__ import annotations

import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from tests._evolution_state_shared import _patch_commit_seam


def test_evolution_orphan_ref_cannot_be_published_by_later_normal_push(
    tmp_path, monkeypatch,
):
    from ouroboros.tools import git as git_tools
    from supervisor import git_ops

    repo, remote = tmp_path / "repo", tmp_path / "remote.git"

    def _git(*args, cwd=repo, check=True):
        return subprocess.run(
            ["git", *args], cwd=cwd, check=check, capture_output=True, text=True,
        )

    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True,
    )
    repo.mkdir()
    _git("init", "-b", "ouroboros")
    _git("config", "user.name", "Test")
    _git("config", "user.email", "test@example.com")
    _git("remote", "add", "origin", str(remote))
    (repo / "file.txt").write_text("base\n", encoding="utf-8")
    (repo / "peer.txt").write_text("peer-base\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "base")
    base_sha = _git("rev-parse", "HEAD").stdout.strip()
    _git("tag", "-a", "v-base", "-m", "base")
    _git("push", "-u", "origin", "ouroboros")
    _git("push", "origin", "--tags")
    (repo / "file.txt").write_text("orphan\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "orphan")
    orphan_sha = _git("rev-parse", "HEAD").stdout.strip()
    _git("tag", "-a", "v-orphan", "-m", "orphan")
    (repo / "peer.txt").write_text("peer-concurrent-edit\n", encoding="utf-8")

    note = git_tools._preserve_evolution_orphan(
        SimpleNamespace(repo_dir=repo), orphan_sha, created_tag="v-orphan",
    )

    assert "CONTAINMENT_FAILED" not in note
    assert _git("rev-parse", "HEAD").stdout.strip() == base_sha
    private_ref = f"refs/ouroboros/evolution-orphans/{orphan_sha}"
    assert _git("rev-parse", private_ref).stdout.strip() == orphan_sha
    assert _git("show-ref", "--verify", "refs/tags/v-orphan", check=False).returncode != 0
    assert _git("rev-parse", "refs/tags/v-base^{commit}").stdout.strip() == base_sha
    assert (repo / "peer.txt").read_text(encoding="utf-8") == "peer-concurrent-edit\n"
    assert _git("status", "--porcelain").stdout.strip()

    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    pushed, _message = git_ops.push_to_remote("ouroboros", push_tags=True)

    assert pushed is True
    assert _git("rev-parse", "refs/heads/ouroboros", cwd=remote).stdout.strip() == base_sha
    assert _git("show-ref", "--verify", "refs/tags/v-orphan", cwd=remote, check=False).returncode != 0
    assert _git("show-ref", "--verify", private_ref, cwd=remote, check=False).returncode != 0
    assert _git("cat-file", "-e", orphan_sha, cwd=remote, check=False).returncode != 0

    # A separate Git writer may advance the branch after the atomic containment
    # transaction. Worktree alignment must not move that ref back to the parent.
    (repo / "file.txt").write_text("second orphan\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "second orphan")
    second_orphan = _git("rev-parse", "HEAD").stdout.strip()
    base_tree = _git("rev-parse", f"{base_sha}^{{tree}}").stdout.strip()
    concurrent = subprocess.run(
        ["git", "commit-tree", base_tree, "-p", base_sha],
        cwd=repo,
        input="concurrent branch update\n",
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()
    real_subprocess_run = subprocess.run
    interleaved = {"done": False}

    def _interleave_after_ref_transaction(cmd, *args, **kwargs):
        proc = real_subprocess_run(cmd, *args, **kwargs)
        if cmd[:3] == ["git", "update-ref", "--stdin"] and proc.returncode == 0 and not interleaved["done"]:
            real_subprocess_run(
                ["git", "update-ref", "refs/heads/ouroboros", concurrent, base_sha],
                cwd=repo, check=True, capture_output=True, text=True,
            )
            interleaved["done"] = True
        return proc

    monkeypatch.setattr(git_tools.subprocess, "run", _interleave_after_ref_transaction)
    note = git_tools._preserve_evolution_orphan(
        SimpleNamespace(repo_dir=repo), second_orphan,
    )

    assert "CONTAINMENT_FAILED" not in note
    assert "concurrent branch update" in note
    assert _git("rev-parse", "HEAD").stdout.strip() == concurrent
    assert _git(
        "rev-parse", f"refs/ouroboros/evolution-orphans/{second_orphan}",
    ).stdout.strip() == second_orphan


def test_orphan_ref_transaction_failure_falls_back_to_safe_ref_cas(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools
    from supervisor import git_ops

    repo, remote = tmp_path / "repo", tmp_path / "remote.git"
    real_run = subprocess.run

    def _git(*args, cwd=repo, check=True):
        return real_run(
            ["git", *args], cwd=cwd, check=check, capture_output=True, text=True,
        )

    real_run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)
    repo.mkdir()
    _git("init", "-b", "ouroboros")
    _git("config", "user.name", "Test")
    _git("config", "user.email", "test@example.com")
    _git("remote", "add", "origin", str(remote))
    (repo / "file.txt").write_text("base\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "base")
    base_sha = _git("rev-parse", "HEAD").stdout.strip()
    _git("push", "-u", "origin", "ouroboros")
    (repo / "file.txt").write_text("orphan\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "orphan")
    orphan_sha = _git("rev-parse", "HEAD").stdout.strip()
    _git("tag", "-a", "v-orphan", "-m", "orphan")

    def _fail_transactions(cmd, *args, **kwargs):
        if cmd[:3] == ["git", "update-ref", "--stdin"]:
            # BYTES streams: the transaction call deliberately runs in binary mode
            # (text-mode pipes CRLF-mangle --stdin commands on Windows).
            return subprocess.CompletedProcess(cmd, 1, b"", b"injected transaction failure")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(git_tools.subprocess, "run", _fail_transactions)

    note = git_tools._preserve_evolution_orphan(
        SimpleNamespace(repo_dir=repo), orphan_sha, created_tag="v-orphan",
    )

    assert "CONTAINMENT_FAILED" not in note
    assert _git("rev-parse", "HEAD").stdout.strip() == base_sha
    assert _git(
        "rev-parse", f"refs/ouroboros/evolution-orphans/{orphan_sha}",
    ).stdout.strip() == orphan_sha
    assert _git("show-ref", "--verify", "refs/tags/v-orphan", check=False).returncode != 0

    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    pushed, _message = git_ops.push_to_remote("ouroboros", push_tags=True)

    assert pushed is True
    assert _git("rev-parse", "refs/heads/ouroboros", cwd=remote).stdout.strip() == base_sha
    assert _git("cat-file", "-e", orphan_sha, cwd=remote, check=False).returncode != 0


def test_evolution_commit_refuses_review_when_claim_is_gone(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools

    reviewed = []
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    _patch_commit_seam(monkeypatch, "_evolution_commit_authority",
        lambda *a, **k: ({}, {"ok": False, "reason": "owner_stopped"}),
    )
    _patch_commit_seam(monkeypatch, "_run_reviewed_stage_cycle",
        lambda *a, **k: reviewed.append(True) or {"status": "passed"},
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        branch_dev="ouroboros",
        current_task_type="evolution",
        task_id="evo",
        task_metadata={},
    )

    result = git_tools._repo_commit_push(ctx, "test commit")

    assert "EVOLUTION_AUTHORITY_REVOKED" in result
    assert reviewed == []


def test_postcommit_cas_failure_returns_local_orphan_after_binding(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools
    from supervisor import evolution_lifecycle

    claim = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo"}
    tagged, contained = [], []
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    _patch_commit_seam(monkeypatch, "_evolution_commit_authority", lambda *a, **k: (claim, {"ok": True}))
    _patch_commit_seam(monkeypatch, "_verify_reviewed_commit_binding", lambda *a, **k: (True, ""))
    _patch_commit_seam(monkeypatch, "_run_reviewed_stage_cycle",
        lambda *a, **k: {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post", "binding": {}},
        },
    )
    _patch_commit_seam(monkeypatch, "run_cmd",
        lambda cmd, cwd=None: "d" * 40 if cmd[:3] == ["git", "rev-parse", "HEAD"] else "",
    )
    monkeypatch.setattr(
        evolution_lifecycle,
        "record_evolution_commit",
        lambda **kwargs: {"ok": False, "reason": "owner_stopped", "commit_sha": kwargs["commit_sha"]},
    )
    monkeypatch.setattr(
        git_tools,
        "_auto_tag_on_version_bump",
        lambda *a, **k: tagged.append(True) or "",
    )
    _patch_commit_seam(monkeypatch, "_preserve_evolution_orphan",
        lambda *a, **k: contained.append((a, k)) or "contained",
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        branch_dev="ouroboros",
        current_task_type="evolution",
        task_id="evo",
        task_metadata={"evolution_transaction": claim},
    )

    result = git_tools._repo_commit_push(ctx, "test commit")

    assert "EVOLUTION_COMMIT_ORPHANED" in result
    assert "d" * 40 in result
    assert tagged == [True]
    assert len(contained) == 1


@pytest.mark.parametrize(
    ("task_type", "expected_order"),
    [
        ("evolution", ["authority", "push", "release", "publish"]),
        ("task", ["release", "push", "publish"]),
    ],
)
def test_only_evolution_push_stays_under_git_lock(
    tmp_path, monkeypatch, task_type, expected_order,
):
    from ouroboros.tools import git as git_tools

    claim = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo"}
    order = []
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: order.append("release"))
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    _patch_commit_seam(monkeypatch, "_evolution_commit_authority", lambda *a, **k: (claim, {"ok": True}))
    _patch_commit_seam(monkeypatch, "_verify_reviewed_commit_binding", lambda *a, **k: (True, ""))
    _patch_commit_seam(monkeypatch, "_run_reviewed_stage_cycle",
        lambda *a, **k: {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post", "binding": {}},
        },
    )
    _patch_commit_seam(monkeypatch, "run_cmd",
        lambda cmd, cwd=None: "d" * 40 if cmd[:3] == ["git", "rev-parse", "HEAD"] else "",
    )
    monkeypatch.setattr(git_tools, "_auto_tag_on_version_bump", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_record_evolution_commit_receipt", lambda *a, **k: "")
    monkeypatch.setattr(
        git_tools,
        "_evolution_publication_stopped_result",
        lambda *a, **k: order.append("authority") or "",
    )
    monkeypatch.setattr(git_tools, "_auto_push", lambda *a, **k: order.append("push") or "")
    monkeypatch.setattr(
        git_tools,
        "_publish_reviewed_commit",
        lambda *a, **k: order.append("publish") or "ok",
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        branch_dev="ouroboros",
        current_task_type=task_type,
        task_id="evo",
        task_metadata={"evolution_transaction": claim},
    )

    assert git_tools._repo_commit_push(ctx, "test commit", skip_tests=True) == "ok"
    assert order == expected_order


def test_revoked_publication_does_not_record_or_anchor_success(tmp_path, monkeypatch):
    from ouroboros import mutation_attribution
    from ouroboros.tools import git as git_tools

    claim = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo"}
    sha = "d" * 40
    attempts, baselines, contained, pushed = [], [], [], []
    authority = iter([
        (claim, {"ok": True}),
        (claim, {"ok": True}),
        (claim, {"ok": True}),
        (claim, {"ok": False, "reason": "owner_stopped"}),
    ])
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", ("root", "task")))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_record_commit_attempt", lambda *a, **k: attempts.append((k.get("status") or a[2], k)))
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    _patch_commit_seam(monkeypatch, "_evolution_commit_authority", lambda *a, **k: next(authority))
    _patch_commit_seam(monkeypatch, "_verify_reviewed_commit_binding", lambda *a, **k: (True, ""))
    def reviewed(review_ctx, *args, **kwargs):
        review_ctx._last_triad_raw_results = [{"raw": "triad"}]
        review_ctx._last_scope_raw_result = {"raw": "scope"}
        review_ctx._review_degraded_reasons = ["recorded"]
        return {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post", "binding": {}},
        }
    _patch_commit_seam(monkeypatch, "_run_reviewed_stage_cycle", reviewed)
    _patch_commit_seam(monkeypatch, "run_cmd", lambda cmd, cwd=None: sha if cmd[:3] == ["git", "rev-parse", "HEAD"] else "")
    monkeypatch.setattr(git_tools, "_auto_tag_on_version_bump", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_record_evolution_commit_receipt", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_preserve_evolution_orphan", lambda *a, **k: contained.append(True) or "contained")
    monkeypatch.setattr(git_tools, "_auto_push", lambda *a, **k: pushed.append(True) or "")
    monkeypatch.setattr(mutation_attribution, "advance_mutation_baseline", lambda *a, **k: baselines.append(a))
    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, branch_dev="ouroboros",
        current_task_type="evolution", task_id="evo",
        task_metadata={"evolution_transaction": claim},
        _scope_review_history={"keep": True},
    )

    result = git_tools._repo_commit_push(ctx, "test commit", skip_tests=True)

    assert "EVOLUTION_PUBLICATION_STOPPED" in result
    assert contained == [True]
    assert pushed == []
    assert baselines == []
    statuses = [status for status, _details in attempts]
    assert "succeeded" not in statuses and statuses[-1] == "failed"
    failed = attempts[-1][1]
    assert failed["fingerprint_status"] == "matched"
    assert failed["pre_review_fingerprint"] == "pre"
    assert failed["post_review_fingerprint"] == "post"
    assert failed["triad_raw_results"] == [{"raw": "triad"}]
    assert failed["scope_raw_result"] == {"raw": "scope"}
    assert failed["degraded_reasons"] == ["recorded"]
    assert not getattr(ctx, "last_reviewed_commit_sha", "")
    assert ctx._scope_review_history == {"keep": True}


def test_postcommit_binding_failure_contains_evolution_commit(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools

    claim = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo"}
    contained = []
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    _patch_commit_seam(monkeypatch, "_evolution_commit_authority", lambda *a, **k: (claim, {"ok": True}))
    _patch_commit_seam(monkeypatch, "_run_reviewed_stage_cycle",
        lambda *a, **k: {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post", "binding": {}},
        },
    )
    _patch_commit_seam(monkeypatch, "run_cmd",
        lambda cmd, cwd=None: "2" * 40 if cmd[:3] == ["git", "rev-parse", "HEAD"] else "",
    )
    _patch_commit_seam(monkeypatch, "_verify_reviewed_commit_binding",
        lambda *a, **k: (False, "tree mismatch"),
    )
    _patch_commit_seam(monkeypatch, "_preserve_evolution_orphan",
        lambda *a, **k: contained.append((a, k)) or "contained",
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        current_task_type="evolution",
        task_id="evo",
        task_metadata={"evolution_transaction": claim},
    )

    result = git_tools._repo_commit_push(ctx, "test commit")

    assert "REVIEW_BINDING_FAILED" in result
    assert "contained" in result
    assert len(contained) == 1
    assert contained[0][1] == {}


def test_final_tag_binding_failure_cannot_record_restart_receipt(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools

    claim = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo"}
    recorded = []
    contained = []
    binding_results = iter([(True, ""), (False, "tag target mismatch")])
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    _patch_commit_seam(monkeypatch, "_evolution_commit_authority", lambda *a, **k: (claim, {"ok": True}))
    _patch_commit_seam(monkeypatch, "_verify_reviewed_commit_binding", lambda *a, **k: next(binding_results),
    )
    _patch_commit_seam(monkeypatch, "_run_reviewed_stage_cycle",
        lambda *a, **k: {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {
                "fingerprint": "post",
                "binding": {"expected_tag": "v-test"},
            },
        },
    )
    _patch_commit_seam(monkeypatch, "run_cmd",
        lambda cmd, cwd=None: "1" * 40 if cmd[:3] == ["git", "rev-parse", "HEAD"] else "",
    )
    monkeypatch.setattr(
        git_tools,
        "_auto_tag_on_version_bump",
        lambda *a, **k: " [tagged: v-test]",
    )
    _patch_commit_seam(monkeypatch, "_preserve_evolution_orphan",
        lambda *a, **k: contained.append((a, k)) or "contained",
    )
    _patch_commit_seam(monkeypatch, "_record_evolution_commit_receipt",
        lambda *a, **k: recorded.append(True) or "",
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        branch_dev="ouroboros",
        current_task_type="evolution",
        task_id="evo",
        task_metadata={"evolution_transaction": claim},
    )

    result = git_tools._repo_commit_push(ctx, "test commit")

    assert "REVIEW_BINDING_FAILED" in result
    assert recorded == []
    assert len(contained) == 1
    assert contained[0][1]["created_tag"] == "v-test"


def test_evolution_publication_authority_requires_exact_head(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools

    monkeypatch.setattr(
        "supervisor.evolution_lifecycle.check_evolution_authority",
        lambda **kwargs: {"ok": True, "reason": ""},
    )
    _patch_commit_seam(monkeypatch, "run_cmd",
        lambda cmd, cwd=None: "b" * 40 if cmd[:3] == ["git", "rev-parse", "HEAD"] else "",
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        task_id="evo",
        task_metadata={"evolution_transaction": {
            "campaign_id": "camp",
            "transaction_id": "tx",
            "task_id": "evo",
        }},
    )

    _, authority = git_tools._evolution_commit_authority(ctx, commit_sha="a" * 40)

    assert authority["ok"] is False
    assert authority["reason"] == "head_mismatch"


def test_evolution_promote_event_carries_exact_claim():
    from ouroboros.tools import control

    ctx = SimpleNamespace(
        current_task_type="evolution",
        task_id="evo-task",
        task_metadata={"evolution_transaction": {
            "campaign_id": "campaign",
            "transaction_id": "transaction",
            "task_id": "evo-task",
            "commit_sha": "",
        }},
        last_reviewed_commit_sha="a" * 40,
        pending_events=[],
    )

    control._promote_to_stable(ctx, "reviewed")

    event = ctx.pending_events[0]
    assert event["type"] == "promote_to_stable"
    assert event["reason"] == "reviewed"
    assert event["evolution_claim"] == {
        "campaign_id": "campaign",
        "transaction_id": "transaction",
        "task_id": "evo-task",
        "commit_sha": "a" * 40,
    }


def test_promote_to_stable_rechecks_evolution_claim_without_changing_normal_flow(
    tmp_path, monkeypatch,
):
    from supervisor import events, evolution_lifecycle

    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.strip()

    _git("init", "-b", "ouroboros")
    _git("config", "user.name", "Test")
    _git("config", "user.email", "test@example.com")
    (repo / "file.txt").write_text("base\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "base")
    base_sha = _git("rev-parse", "HEAD")
    _git("branch", "ouroboros-stable", base_sha)
    (repo / "file.txt").write_text("reviewed\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "reviewed")
    reviewed_sha = _git("rev-parse", "HEAD")
    sent = []
    ctx = SimpleNamespace(
        REPO_DIR=repo,
        BRANCH_DEV="ouroboros",
        BRANCH_STABLE="ouroboros-stable",
        load_state=lambda: {"owner_chat_id": 1},
        send_with_budget=lambda chat_id, message: sent.append(message),
    )
    monkeypatch.setattr(
        evolution_lifecycle,
        "check_evolution_authority",
        lambda **claim: {
            "ok": claim.get("campaign_id") == "valid",
            "reason": "owner_stopped" if claim.get("campaign_id") != "valid" else "",
        },
    )

    events._handle_promote_to_stable({
        "type": "promote_to_stable",
        "evolution_claim": {
            "campaign_id": "revoked",
            "transaction_id": "tx",
            "task_id": "evo",
            "commit_sha": reviewed_sha,
        },
    }, ctx)
    assert _git("rev-parse", "ouroboros-stable") == base_sha
    assert "owner_stopped" in sent[-1]

    events._handle_promote_to_stable({
        "type": "promote_to_stable",
        "evolution_claim": {
            "campaign_id": "",
            "transaction_id": "",
            "task_id": "",
            "commit_sha": "",
        },
    }, ctx)
    assert _git("rev-parse", "ouroboros-stable") == base_sha
    assert "commit_receipt_missing" in sent[-1]

    events._handle_promote_to_stable({
        "type": "promote_to_stable",
        "evolution_claim": {
            "campaign_id": "valid",
            "transaction_id": "tx",
            "task_id": "evo",
            "commit_sha": reviewed_sha,
        },
    }, ctx)
    assert _git("rev-parse", "ouroboros-stable") == reviewed_sha

    _git("branch", "-f", "ouroboros-stable", base_sha)
    events._handle_promote_to_stable({"type": "promote_to_stable"}, ctx)
    assert _git("rev-parse", "ouroboros-stable") == reviewed_sha

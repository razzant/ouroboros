"""Repository, context and gateway builders shared by the run-isolation suites.

Split out of ``tests/test_delegated_run_isolation.py`` when that module was divided by
theme; every definition is verbatim, so each sibling suite keeps the exact seeded
target, nanny context, custody entry and stub gateways it was written against.
"""

from __future__ import annotations

import pathlib
import subprocess


from ouroboros import delegate_custody as custody


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check,
    )


def _seed_target(tmp_path: pathlib.Path) -> pathlib.Path:
    """A target tree with every capture class: tracked, staged, unstaged,
    untracked-eligible, and untracked-sensitive."""
    target = tmp_path / "target"
    target.mkdir()
    _git(target, "init")
    (target / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(target, "add", "-A")
    _git(target, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed")
    (target / "tracked.txt").write_text("one\ntwo\n", encoding="utf-8")   # unstaged mod
    (target / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(target, "add", "staged.txt")                                     # staged add
    (target / "untracked.txt").write_text("loose\n", encoding="utf-8")    # eligible
    (target / ".env").write_text("SECRET=1\n", encoding="utf-8")          # sensitive
    return target


def _nanny_ctx(tmp_path, target, monkeypatch):
    """A nanny ToolContext whose active root IS the target external workspace,
    with the module-default snapshot/registry roots pinned inside the test tmp."""
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snaps"))
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    drive = tmp_path / "drive"
    drive.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    ctx.workspace_root = str(target)
    ctx.workspace_mode = "external"
    ctx.task_id = "t-nanny"
    ctx.task_metadata = {}
    return ctx


def _isolated_entry(ctx, target, handle, *, run_id="run-1", settled=True):
    entry = custody.RunCustody(
        run_id=run_id, task_id="t-nanny", route_id="some-route",
        snapshot_id=handle.snapshot_id, execution_root=handle.path,
        baseline_sha=handle.baseline_sha, target_root=str(target),
        authority_source="external_workspace_root", settled=settled,
    )
    custody._CUSTODY[entry.run_id] = entry
    return entry


class _TerminalSweepGateway:
    """A daemon for the orphan sweep: recovery re-POSTs bind a run; every asked
    run is already terminal-succeeded; controls are accepted."""

    def __init__(self, run_id="run-rec", state="succeeded"):
        self.run_id, self.state = run_id, state

    def handshake(self, **_kw):
        return {"compatible": True}

    def start_run(self, request, *, idempotency_key=""):
        return {"runId": self.run_id}

    def get_run(self, rid, **_kw):
        return {"lastSeq": 2, "summary": {"state": self.state, "spendUsd": 0.0,
                                          "model": "m", "effectiveAccess": "workspace_write"}}

    def cancel_run(self, rid, reason=""):
        return {"accepted": True, "status": "ok"}

    def remove_project(self, pid):
        return {}

    def close(self):
        pass


def _binding_request_row(task_id, invocation_id, handle):
    """The exact START_REQUESTED payload delegate.py records for a mutating start."""
    body = {"prompt": "do work", "access": "workspace_write", "mode": "agent",
            "primaryHarness": "some-route", "model": "", "effort": "", "maxSeconds": 600,
            "execution": {"isolation": "live", "delegated": True},
            "scope": {"kind": "project", "root": handle.path}}
    return dict(
        run_id="", task_id=task_id, idempotency_key=f"k-{invocation_id}",
        invocation_id=invocation_id, max_seconds=600, request=body,
        project_id=f"prj-{invocation_id}", project_owned=True, route="some-route",
        root_task_id="", parent_task_id="",
        snapshot_id=handle.snapshot_id, execution_root=handle.path,
        baseline_sha=handle.baseline_sha, target_root=handle.target_root,
        authority_source="acting_constraint")


class _HealthEnv:
    """The minimal env build_health_invariants needs, rooted at one data dir."""

    def __init__(self, data: pathlib.Path):
        self.drive_root = data
        self._data = data

    def drive_path(self, rel=""):
        return self._data / rel

    def repo_path(self, rel=""):
        return self._data / "repo" / rel

"""v6.52.2 — ephemeral `scratch=[...]` (Fix #1) + verify_and_record exit-masking sensor (Fix #2).

Both are GENERAL, leak-free, advisory-only. Additive: with no scratch / no masking, behavior is
unchanged. Includes the false-completion adversarial coverage DEVELOPMENT.md "Loop / State-Machine Changes" mandates for the
loop nudge (one-shot, fires only on a masked unreconciled PASS, suppressed by a later clean pass,
ordered after the red nudge, advisory).
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

from ouroboros.tools.registry import ToolRegistry

# Spawns real OS subprocesses (git, the `sys.executable` Python, and a POSIX `sh` in one guarded
# case) via _tracked_subprocess_run — not parallel-safe under `pytest -n auto` (DEVELOPMENT.md
# "Parallel CI and the serial marker"). Python invocations use `sys.executable` (cross-platform);
# the only POSIX-shell-dependent test is skipped when `sh` is unavailable (e.g. Windows).


# ----------------------------------------------------------------------------- helpers
def _git_ws(parent, name="proj"):
    """A git worktree under user_files (scratch requires a git-worktree cwd)."""
    ws = parent / name
    ws.mkdir()
    _git(ws, "init", "-q")
    return ws


def _reg(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    home = tmp_path / "home"
    # NEUTRAL layout, not home/Ouroboros/data: the registry gets explicit
    # roots. The pytest fail-closed guard (`utils.assert_test_data_path`, the
    # issue #455 class) reads `os.path.expanduser("~")`, deliberately NOT
    # `Path.home()`, so the `Path.home` patch below neither trips nor moves it;
    # keeping these roots off a live-shaped `Ouroboros/data` path just keeps
    # the fixture visibly hermetic rather than relying on that distinction.
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    desktop = home / "Desktop"
    for d in (repo, data, desktop):
        d.mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry._ctx.task_id = "task1"
    return registry, repo, data, desktop


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.email=t@e", "-c", "user.name=T", *args], cwd=repo, check=True, capture_output=True)


# ============================================================ Fix #1: ephemeral scratch


def test_scratch_adopts_preexisting_untracked_file(tmp_path, monkeypatch):
    """v6.56.0: an existing UNTRACKED file in the git worktree is ADOPTED as scratch (not blocked);
    its current sha is recorded so a re-declaration is idempotent and it stays excluded from the patch."""
    from ouroboros.artifacts import read_task_scratch_fingerprints
    from hashlib import sha256

    registry, _repo, data, desktop = _reg(tmp_path, monkeypatch)
    ws = _git_ws(desktop)
    existing = ws / "adopted.py"
    existing.write_bytes(b"throwaway v1\n")  # exact bytes (no Windows \n→\r\n) so the sha is stable
    result = registry.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", "print('noop')"], "cwd": str(ws), "scratch": [str(existing)]},
    )
    assert not result.startswith("⚠️ SCRATCH_BLOCKED"), result
    assert "exit_code=0" in result
    # adoption recorded the current sha at declaration
    fps = read_task_scratch_fingerprints(data, "task1")
    assert fps.get(str(existing.resolve())) == sha256(b"throwaway v1\n").hexdigest()
    # re-declaring the same path in a second command is also fine (idempotent)
    result2 = registry.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", "print('noop2')"], "cwd": str(ws), "scratch": [str(existing)]},
    )
    assert not result2.startswith("⚠️ SCRATCH_BLOCKED"), result2


def test_scratch_tracked_path_still_blocked(tmp_path, monkeypatch):
    """The masking guard survives adoption: a git-TRACKED file can never be relabeled scratch."""
    registry, _repo, _data, desktop = _reg(tmp_path, monkeypatch)
    ws = _git_ws(desktop)
    tracked = ws / "real_source.py"
    tracked.write_text("real\n")
    _git(ws, "add", "real_source.py")
    _git(ws, "commit", "-q", "-m", "add source")
    result = registry.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", "print('noop')"], "cwd": str(ws), "scratch": [str(tracked)]},
    )
    assert result.startswith("⚠️ SCRATCH_BLOCKED"), result
    assert "git-tracked" in result


def test_scratch_directory_declaration_refused(tmp_path, monkeypatch):
    """v6.56.0 review regression: an existing untracked DIRECTORY declared scratch
    must be refused (a dir can't be sha-fingerprinted or excluded file-by-file, so
    silently adopting it would let its contents leak into the deliverable)."""
    registry, _repo, _data, desktop = _reg(tmp_path, monkeypatch)
    ws = _git_ws(desktop)
    scratch_dir = ws / "tmpwork"
    scratch_dir.mkdir()
    (scratch_dir / "inner.txt").write_text("x")
    result = registry.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", "print('noop')"], "cwd": str(ws), "scratch": [str(scratch_dir)]},
    )
    assert result.startswith("⚠️ SCRATCH_BLOCKED"), result
    assert "directory" in result


def test_scratch_traversal_outside_cwd_is_blocked(tmp_path, monkeypatch):
    registry, _repo, _data, desktop = _reg(tmp_path, monkeypatch)
    ws = _git_ws(desktop)
    result = registry.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", "print('noop')"], "cwd": str(ws), "scratch": ["../escape.py"]},
    )
    assert result.startswith("⚠️ SCRATCH_BLOCKED"), result
    assert "escapes the command cwd" in result


def test_headless_excludes_declared_scratch_from_workspace_patch(tmp_path):
    from hashlib import sha256
    from ouroboros.headless import SCRATCH_MANIFEST_NAME, write_workspace_patch_artifacts

    repo = tmp_path / "ws"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "--allow-empty", "-m", "base", "-q")
    (repo / "real_deliverable.txt").write_text("keep\n")          # genuine new file -> included
    scratch_body = "// throwaway\n"
    (repo / "scratch_probe_test.go").write_text(scratch_body)      # declared scratch -> excluded (sha match)
    scratch_sha = sha256((repo / "scratch_probe_test.go").read_bytes()).hexdigest()

    art = tmp_path / "artifacts"
    art.mkdir()
    (art / SCRATCH_MANIFEST_NAME).write_text(json.dumps({"schema_version": 2, "scratch": {
        str((repo / "scratch_probe_test.go").resolve()): scratch_sha
    }}))
    _arts, manifest = write_workspace_patch_artifacts(repo, art, task={"id": "t", "workspace_root": str(repo)})
    included = manifest.get("untracked_included") or []
    excluded = [e.get("path") for e in (manifest.get("untracked_excluded") or [])]
    assert "real_deliverable.txt" in included
    assert "scratch_probe_test.go" not in included
    assert "scratch_probe_test.go" in excluded

    # CRITICAL FIX (round-1 review): a LATER real file at the SAME path (different content) must NOT
    # be dropped — the manifest is fingerprint-gated, not path-authoritative.
    (repo / "scratch_probe_test.go").write_text("// a REAL later edit, different content\n")
    art2 = tmp_path / "artifacts2"
    art2.mkdir()
    (art2 / SCRATCH_MANIFEST_NAME).write_text(json.dumps({"schema_version": 2, "scratch": {
        str((repo / "scratch_probe_test.go").resolve()): scratch_sha  # STALE sha
    }}))
    _a2, manifest2 = write_workspace_patch_artifacts(repo, art2, task={"id": "t", "workspace_root": str(repo)})
    assert "scratch_probe_test.go" in (manifest2.get("untracked_included") or [])  # sha mismatch -> included

    # 0-regression: with NO scratch manifest, the file IS included.
    art3 = tmp_path / "artifacts3"
    art3.mkdir()
    _a3, manifest3 = write_workspace_patch_artifacts(repo, art3, task={"id": "t", "workspace_root": str(repo)})
    assert "scratch_probe_test.go" in (manifest3.get("untracked_included") or [])


# ============================================================ Fix #2: exit-masking sensor
@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("sh"), reason="exercises a POSIX-shell masked check (sh + tail)")
def test_verify_and_record_sets_masking_flag_on_receipt(tmp_path, monkeypatch):
    from ouroboros.outcomes import read_verification_receipts

    registry, repo, data, _desktop = _reg(tmp_path, monkeypatch)
    # masked: `echo hi | tail -1` exits 0 (tail), masking any upstream failure; expected matches.
    out = str(registry.execute(
        "verify_and_record",
        {"contract_kind": "explicit_command", "check": ["sh", "-c", "echo hi | tail -1"], "expected": "hi", "cwd": str(repo)},
    ))
    rec = read_verification_receipts(data, "task1")[-1]
    assert rec["status"] == "pass"  # FLAG-ONLY: masking does not flip the verdict
    assert rec.get("check_exit_masking") is True
    assert "pipeline_tail" in (rec.get("check_exit_masking_reasons") or [])
    # The author reads it at call time, in run_command's words, not only at the pre-final check.
    assert "EXIT_MASKING_NOTE: exit_code=0 belongs to the last stage (pipeline_tail)" in out
    # clean check: no masking flag, no note
    out = str(registry.execute(
        "verify_and_record",
        {"contract_kind": "explicit_command", "check": ["sh", "-c", "echo hi"], "expected": "hi", "cwd": str(repo)},
    ))
    assert not read_verification_receipts(data, "task1")[-1].get("check_exit_masking")
    assert "EXIT_MASKING_NOTE" not in out
    # A non-zero exit already says what happened: the flag stays on the receipt, the note does not repeat it.
    out = str(registry.execute(
        "verify_and_record",
        {"contract_kind": "explicit_command", "check": ["sh", "-c", "echo hi | tail -1; exit 3"], "cwd": str(repo)},
    ))
    assert read_verification_receipts(data, "task1")[-1].get("check_exit_masking") is True
    assert "EXIT_MASKING_NOTE" not in out


@pytest.mark.parametrize("head", ["sh", "bash", "zsh", "dash", "ash"])
@pytest.mark.parametrize("absolute", [False, True])
def test_subshell_masking_uses_typed_shell_grammar(head, absolute):
    from ouroboros.tools.verify import check_exit_masking

    executable = "/bin/" + head if absolute else head
    for command, reason in [
        ("make test|tail", "pipeline_tail"),
        ("make test||true", "|| true"),
        ("make test; true", "; true"),
        ("make test; exit 0", "exit 0"),
        ("make test|'tail'", "pipeline_tail"),
    ]:
        expected = check_exit_masking([executable, "-c", command])
        assert expected == (True, [reason])
        assert check_exit_masking([executable, "-c", "(" + command + ")"]) == expected
    assert check_exit_masking([executable, "-c", "(false)|(tail)"]) == (True, ["pipeline_tail"])
    for literal in ["echo '|' tail", "echo '||' true", "echo ';' true", "echo '(' false '|tail)'", r"echo \(false\|tail\)"]:
        assert check_exit_masking([executable, "-c", literal]) == (False, [])

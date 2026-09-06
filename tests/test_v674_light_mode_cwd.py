"""v6.74.0 (D1): the light-mode shell guard resolves the cwd BEFORE judging repo targets.

`repo_target_mentioned` used to join the RAW cwd string onto repo_dir, so a
resource-root LABEL (``cwd="task_drive"``) resolved to ``<repo>/task_drive`` —
"inside the repo" by construction — and a legitimate task-drive write was
light-blocked with a message advising the very root that was used. The guard
now receives the RESOLVED work dir from the shared ``resolve_shell_cwd``
resolver; a resolution failure fails closed with the standard cwd block.
"""
from __future__ import annotations

import pathlib

import pytest

from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.shell_guards import light_shell_repo_mutation, repo_target_mentioned


def _registry(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")
    reg._ctx.task_id = "t1"
    return reg


# ---- unit level: the guard judges the RESOLVED work dir -------------------


def test_resource_label_cwd_is_not_read_as_repo_relative(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "drive" / "task_drives" / "t1"
    drive.mkdir(parents=True)
    # Old behavior: cwd="task_drive" joined onto repo -> <repo>/task_drive ->
    # a relative write target read as inside the repo -> false block.
    assert light_shell_repo_mutation(
        "touch out.txt", repo_dir=repo, cwd="task_drive", work_dir=drive,
    ) is False
    # Legacy fallback (no resolved work_dir) reproduces the historical bug —
    # pinned so the fix is observable and the resolver hoist stays mandatory.
    assert light_shell_repo_mutation(
        "touch out.txt", repo_dir=repo, cwd="task_drive",
    ) is True


def test_external_path_cwd_write_is_allowed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    assert light_shell_repo_mutation(
        "touch out.txt", repo_dir=repo, cwd=str(external), work_dir=external,
    ) is False
    assert repo_target_mentioned(
        ["touch", "out.txt"], repo_dir=repo, work_dir=external,
    ) is False


def test_genuine_repo_target_still_blocks(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    # Absolute repo path is blocked regardless of the resolved cwd. Argv-list
    # form: an f-string embeds Windows backslashes that shlex mangles (the CI
    # 3-OS matrix caught exactly this in the first tag run).
    assert light_shell_repo_mutation(
        ["touch", str(repo / "x.py")], repo_dir=repo, cwd=str(external), work_dir=external,
    ) is True
    # Relative target with the repo itself as the resolved work dir is blocked.
    assert light_shell_repo_mutation(
        ["touch", "x.py"], repo_dir=repo, work_dir=repo,
    ) is True


def test_versioned_interpreter_basename_still_classified(tmp_path):
    """python3.11 / absolute versioned paths must engage the inline-write fence.

    CI's interpreter basename is unversioned ("python"), so the four
    public-surface light-fence tests cannot catch this class there; this pins
    the classification host-independently. Regression: exact-set matching of
    {"python", "python3", ...} let a versioned agent python (the resolver
    injects OUROBOROS_AGENT_PYTHON or sys.executable into argv[0]) bypass
    detect_interpreter_inline entirely.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    inline_write = "open('probe.txt', 'w').write('x')"
    for exe in ("python3.11", "python3.12", "/opt/homebrew/bin/python3.11", "python3.11.exe"):
        assert light_shell_repo_mutation(
            [exe, "-c", inline_write],
            repo_dir=repo, cwd=str(repo), work_dir=repo,
            detect_interpreter_inline=True,
        ) is True, exe


# ---- registry level: the resolver is hoisted above the light guard ---------


@pytest.mark.serial
def test_light_mode_task_drive_label_cwd_is_not_light_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    # Portable write (no `touch` on Windows shells — the v6.73.1 class):
    # a python inline write in the resolved task-drive cwd.
    result = reg.execute("run_command", {
        "cmd": ["python", "-c", "import pathlib; pathlib.Path('out.txt').write_text('x')"],
        "cwd": "task_drive",
    })
    assert "LIGHT_MODE_BLOCKED" not in result, result[:300]
    task_drive = pathlib.Path(reg._ctx.task_drive_root())
    assert (task_drive / "out.txt").exists()


def test_light_mode_versioned_interpreter_runtime_data_write_is_registry_blocked(tmp_path, monkeypatch):
    """Registry-level twin of ``test_versioned_interpreter_basename_still_classified``.

    The unit test above pins shell_guards' inline-write fence; this pins the
    OTHER half of INFRA-1 — ``ToolRegistry._run_shell_safety_check``'s
    ``runtime_data_scan`` classifier — which that unit test cannot see. The
    four public-surface light-fence tests run ``sys.executable``, whose
    basename on CI is unversioned, so reverting the registry classifier to the
    exact set {"python", "python3", ...} keeps them green there. This test
    spells the versioned basename explicitly and is host-independent: the
    guard must refuse BEFORE execution, so ``python3.11`` need not exist on
    the host — and where it does exist, the write would land and fail the
    no-file assertion instead.
    """
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    # A runtime_data path outside the task's own roots (drive_root/uploads is
    # neither this task's task_drive nor its artifact_store).
    target = tmp_path / "drive" / "uploads" / "probe-report.html"
    result = reg.execute("run_command", {
        "cmd": [
            "python3.11",
            "-c",
            (
                "from pathlib import Path\n"
                f"p = Path({str(target)!r})\n"
                "p.parent.mkdir(parents=True, exist_ok=True)\n"
                "p.touch()\n"
            ),
        ],
        "cwd": str(reg._ctx.task_drive_root()),
    })
    assert "LIGHT_MODE_BLOCKED" in result, result[:300]
    assert "runtime_data" in result
    assert not target.exists()


def test_light_mode_repo_write_still_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": "touch marker.py"})
    assert "LIGHT_MODE_BLOCKED" in result, result[:300]
    assert not (pathlib.Path(reg._ctx.repo_dir) / "marker.py").exists()


def test_light_mode_cwd_resolution_failure_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": "touch out.txt", "cwd": "/etc"})
    assert "SHELL_CWD_BLOCKED" in result, result[:300]


def test_light_mode_versioned_interpreter_triggers_runtime_data_scan(tmp_path, monkeypatch):
    """The registry half of the versioned-basename fix: `python3.11` must engage
    ToolRegistry's light-mode runtime_data scan exactly like `python`.

    The command is deliberately a PURE READ (no coarse write indicator — pathlib
    ``read_text``, no ``open(``), so `writeish` is False and the ONLY thing that
    can start the scan is the interpreter classification itself: with the
    exact-set match ({"python", "python3", ...}) a versioned agent python walked
    straight past the scan and read secret-named runtime_data (settings.json at
    the drive root) that the same command spelled `python -c` is blocked from.
    `runtime_data_guard_targets` always handled startswith("python") internally —
    the invocation trigger was the untested bypass."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    secret = pathlib.Path(reg._ctx.drive_root) / "settings.json"
    read_cmd = f"import pathlib; print(pathlib.Path({str(secret)!r}).read_text())"

    result = reg.execute_result("run_command", {"cmd": ["python3.11", "-c", read_cmd]})
    assert (result.status, result.code) == ("blocked", "LIGHT_MODE_BLOCKED")
    assert str(secret) in result.text

    # Parity pin: the unversioned spelling of the same command is blocked the
    # same way — the versioned basename must not be the weaker path.
    unversioned = reg.execute_result("run_command", {"cmd": ["python", "-c", read_cmd]})
    assert (unversioned.status, unversioned.code) == (result.status, result.code)

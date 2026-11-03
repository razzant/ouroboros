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
import shlex
from subprocess import CompletedProcess

import pytest

from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.shell_guards import light_shell_repo_mutation, repo_target_mentioned


def _registry(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")
    reg._ctx.task_id = "t1"
    reg._ctx.task_metadata = {"work_kind": "owner"}  # an owner task's minted kind
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




@pytest.mark.serial
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




@pytest.mark.parametrize("head", ["sh", "bash", "zsh", "dash", "ash"])
@pytest.mark.parametrize("tool_name", ["run_command", "verify_and_record"])
def test_posix_wrappers_preserve_light_read_and_deliverable_guards(
    tmp_path, monkeypatch, head, tool_name,
):
    """Every accepted shell spelling reaches the same existing physical-target guards.

    Inspection needs no installed shell: no process or LLM is launched. Both
    process tools use their real argument normalizer and pre-execution guard.
    """
    from ouroboros.tools.registry_guard_process import _run_shell_safety_check
    from ouroboros.tools.shell_guards import (
        process_shell_guard_args, shell_writer_targets_protected, writer_target_rows,
    )
    from ouroboros.tools.write_shape import interpreter_write_shape

    reg = _registry(tmp_path)
    repo = pathlib.Path(reg._ctx.repo_dir)
    (repo / "BIBLE.md").write_text("Constitution fixture", encoding="utf-8")
    user_root = tmp_path / "user-home"
    deliverables = user_root / "Deliverables"
    deliverables.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(user_root))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))

    def guard(body, cwd="system_repo"):
        field = "check" if tool_name == "verify_and_record" else "cmd"
        args = process_shell_guard_args(tool_name, {field: [head, "-c", body], "cwd": cwd})
        return _run_shell_safety_check(reg, args, "light")

    assert guard("cat BIBLE.md") is None
    assert shell_writer_targets_protected([head, "-c", "cat BIBLE.md"]) is False
    assert interpreter_write_shape([head, "-c", "python3 -c 'print(2 > 1)'"]) is False
    assert interpreter_write_shape([head, "-c", "printf result > report.txt"]) is True

    # Root actors retain user_files output authority even with an external
    # project attached; the selected physical cwd decides the write target.
    project = tmp_path / "project"
    project.mkdir()
    reg._ctx.workspace_root = project
    reg._ctx.workspace_mode = "external"
    assert guard("printf result > report.txt", "user_files") is None
    assert guard("cat BIBLE.md") is None

    reg._ctx.workspace_root = None
    reg._ctx.workspace_mode = ""
    for body in ("rm ordinary.py", "rm BIBLE.md", "rm ../drive/state/state.json"):
        refusal = guard(body)
        assert refusal is not None, body
        assert (refusal.status, refusal.code) == ("blocked", "CORE_PROTECTION_BLOCKED" if "BIBLE_DELETE_BLOCKED" in refusal.text else "LIGHT_MODE_BLOCKED"), refusal
    assert shell_writer_targets_protected([head, "-c", "rm BIBLE.md"]) is True
    rows = writer_target_rows([head, "-c", "printf result > report.txt"])
    assert [target for _argv, targets, _inline, _unknown in rows for target in targets] == ["report.txt"]
    assert not (deliverables / "report.txt").exists()  # inspection never executes
    assert (repo / "BIBLE.md").read_text(encoding="utf-8") == "Constitution fixture"



@pytest.mark.parametrize("head", ["sh", "bash", "zsh", "dash", "ash"])
@pytest.mark.parametrize("body,returncode,masked", [
    ("printf verified", 0, False),
    ("printf verified | tail -1", 0, True),
    ("printf verified; exit 7", 7, False),
])
def test_posix_wrapper_verification_preserves_exit_and_masking_receipt(
    tmp_path, monkeypatch, head, body, returncode, masked,
):
    """Real verification/receipt consumers; only the OS process is substituted.

    Shell availability must not erase dash/ash coverage on macOS or Windows.
    The supplied process exit remains verdict authority, masking stays advisory.
    """
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.verify import _verify_and_record

    reg = _registry(tmp_path)
    argv = [head, "-c", body]
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return CompletedProcess(command, returncode, "verified", "")

    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", run)
    result = _verify_and_record(
        reg._ctx, contract_kind="explicit_command", check=argv, expected="verified",
    )
    assert calls == [(argv, str(pathlib.Path(reg._ctx.repo_dir).resolve()))]
    receipt = read_verification_receipts(reg._ctx.drive_root, "t1")[-1]
    assert receipt["status"] == ("pass" if returncode == 0 else "fail"), result
    assert receipt["returncode"] == returncode
    assert shlex.split(receipt["check"]) == argv
    assert receipt.get("check_exit_masking", False) is masked
    assert receipt.get("check_exit_masking_reasons", []) == (["pipeline_tail"] if masked else [])

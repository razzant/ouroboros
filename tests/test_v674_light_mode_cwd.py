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
from ouroboros.tools.shell_guards import light_shell_repo_mutation


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


# ---- registry level: the resolver is hoisted above the light guard ---------




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

    def guard(body, cwd="system_repo", tail=()):
        field = "check" if tool_name == "verify_and_record" else "cmd"
        args = process_shell_guard_args(tool_name, {field: [head, "-c", body, *tail], "cwd": cwd})
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
    # An empty argument keeps its position (#2919): `-c ''` runs an empty body and the
    # next word is only $0, while an empty $0 leaves the real body inspected.
    assert guard("", tail=("rm ordinary.py",)) is None
    assert guard("rm ordinary.py", tail=("",)).code == "LIGHT_MODE_BLOCKED"
    rows = writer_target_rows([head, "-c", "printf result > report.txt"])
    assert [target for _argv, targets, _inline, _unknown in rows for target in targets] == ["report.txt"]
    assert not (deliverables / "report.txt").exists()  # inspection never executes
    assert (repo / "BIBLE.md").read_text(encoding="utf-8") == "Constitution fixture"

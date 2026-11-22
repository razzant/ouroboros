"""v6.74.0 (D1): the light-mode shell guard judges targets against the RESOLVED cwd.

A resource-root LABEL (``cwd="task_drive"``) once resolved to ``<repo>/task_drive``,
"inside the repo" by construction, and a legitimate task-drive write was
light-blocked. The guard receives the RESOLVED work dir from the shared
``resolve_shell_cwd`` resolver; every POSIX shell wrapper reaches the same guards.
"""
from __future__ import annotations

import pathlib

import pytest

from ouroboros.tools.registry import ToolRegistry


def _registry(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")
    reg._ctx.task_id = "t1"
    reg._ctx.task_metadata = {"work_kind": "owner"}  # an owner task's minted kind
    return reg


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
    from ouroboros.tools.shell_audit import _writer_targets_for_output_audit
    from ouroboros.tools.shell_guards import process_shell_guard_args

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
    # An empty argument keeps its position (#2919): `-c ''` runs an empty body and the
    # next word is only $0, while an empty $0 leaves the real body inspected.
    assert guard("", tail=("rm ordinary.py",)) is None
    assert guard("rm ordinary.py", tail=("",)).code == "LIGHT_MODE_BLOCKED"
    assert _writer_targets_for_output_audit([head, "-c", "printf result > report.txt"]) == {"report.txt"}
    assert not (deliverables / "report.txt").exists()  # inspection never executes
    assert (repo / "BIBLE.md").read_text(encoding="utf-8") == "Constitution fixture"

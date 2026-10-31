"""Owner-facing tool contracts verified through their actual consumers."""
from __future__ import annotations

import hashlib
import subprocess

import pytest

from tests.test_vcs_target_binding import _registry, _git
from tests.test_process_environment import process_context as _process_context

process_context = _process_context

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("fallback", [False, True])
def test_brace_search_returns_same_selected_languages(process_context, monkeypatch, fallback):
    from ouroboros import code_search_rg

    registry, _ctx, workspace, _data = process_context
    for name in ("one.js", "two.css", "three.py", "space é.js"):
        (workspace / name).write_text("needle\n", encoding="utf-8")
    if fallback:
        monkeypatch.setattr(code_search_rg, "_rg_binary", lambda: "")
    elif not code_search_rg._rg_binary():
        pytest.skip("rg unavailable; forced fallback has separate coverage")
    result = registry.execute_result("search_code", {"query": "needle", "include": "*.{js,css}"})
    assert result.status == "ok", result.text
    assert all(name in result.text for name in ("one.js", "two.css", "space é.js"))
    assert "three.py" not in result.text
    excluded = registry.execute("search_code", {"query": "needle", "include": "*.css", "path": "one.js"})
    assert "No matches" in excluded and "0 files searched" in excluded
    empty = registry.execute("search_code", {"query": "needle", "include": "*.{rs,go}"})
    assert "0 file" in empty


def test_explicit_rg_list_is_filtered_before_read(tmp_path, monkeypatch):
    from ouroboros import code_search_rg
    from tests.test_code_search_rg import _install_fake_rg

    selected, excluded = tmp_path / "allowed.js", tmp_path / "excluded.py"
    selected.write_text("needle\n", encoding="utf-8")
    excluded.write_text("needle\n", encoding="utf-8")
    _install_fake_rg(tmp_path, monkeypatch)
    read = []
    result = code_search_rg.search_with_rg([selected, excluded], "needle", regex=False,
        include="*.{js,css}", path_allowed=lambda path: read.append(path) or True)
    assert read and set(read) == {selected}
    assert [match.path for match in result.matches] == [selected]
    assert result.files_selected == 1


def test_vcs_read_diff_explicit_trees_index_and_worktree(tmp_path, monkeypatch):
    registry, _ctx, _system, project = _registry(tmp_path)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    base = _git(project, "rev-parse", "HEAD")
    target = project / "project.txt"
    target.write_text("committed\n", encoding="utf-8")
    _git(project, "commit", "-am", "work")
    head = _git(project, "rev-parse", "HEAD")
    target.write_text("indexed\n", encoding="utf-8")
    _git(project, "add", "project.txt")
    target.write_text("working\n", encoding="utf-8")
    observed = {}
    for name, args in {
        "trees": {"base": base, "head": head},
        "index": {"base": base, "staged": True},
        "working": {"base": base},
        "default": {},
    }.items():
        result = registry.execute_result("vcs_read", {**args, "action": "diff"})
        assert result.status == "ok", result.text
        observed[name] = result.text
        if args.get("base"):
            assert result.meta["comparison"]["base_tree"] == _git(project, "rev-parse", f"{base}^{{tree}}")
    assert "+committed" in observed["trees"] and "+working" not in observed["trees"]
    assert "+indexed" in observed["index"] and "+working" not in observed["index"]
    assert "+working" in observed["working"]
    assert "-indexed" in observed["default"] and "Comparison:" not in observed["default"]
    for args in ({"head": head}, {"base": base, "head": head, "staged": True}):
        assert registry.execute_result("vcs_read", {**args, "action": "diff"}).code == "TOOL_ARG_ERROR"
    assert "GIT_ERROR" in registry.execute("vcs_read", {"action": "diff", "base": "missing-ref"})


def test_patch_provenance_does_not_change_application_bytes(tmp_path):
    from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts

    registry, _ctx, _system, project = _registry(tmp_path)
    base = _git(project, "rev-parse", "HEAD")
    _git(project, "checkout", "-qb", "task-branch")
    (project / "project.txt").write_text("branch difference\n", encoding="utf-8")
    _git(project, "commit", "-am", "branch")
    (project / "project.txt").write_text("corrected work\n", encoding="utf-8")
    output = tmp_path / "artifacts"
    _artifacts, manifest = write_workspace_patch_artifacts(project, output, task={
        "metadata": {"workspace_preflight": {"git": {"head": base}}},
        "workspace_preflight": {"git": {"head": base}},
    })
    patch = (output / "workspace.patch").read_bytes()
    assert manifest["base_head"] == base and manifest["base_provenance"] == "admission_head"
    assert manifest["current_branch"] == "task-branch"
    assert manifest["tracking_upstream"] == ""
    assert "does not attribute authorship" in manifest["base_explanation"]
    assert "vcs_read" in manifest["comparison_note"]
    assert manifest["sha256"] == hashlib.sha256(patch).hexdigest()
    expected = subprocess.check_output(["git", "diff", "--binary", "--no-ext-diff", "--no-color", base, "--", "project.txt"], cwd=project)
    assert patch == expected
    destination = tmp_path / "apply"
    subprocess.run(["git", "clone", "-q", str(project), str(destination)], check=True)
    _git(destination, "checkout", "--detach", base)
    subprocess.run(["git", "apply", str(output / "workspace.patch")], cwd=destination, check=True)
    assert (destination / "project.txt").read_text(encoding="utf-8") == "corrected work\n"


def test_external_edit_footer_does_not_forbid_authorized_git(tmp_path, monkeypatch):
    registry, ctx, _system, workspace = _registry(tmp_path)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    written = registry.execute_result("write_file", {"path": "note.txt", "content": "old text\n"})
    edited = registry.execute_result("edit_text", {"path": "note.txt", "old_str": "old", "new_str": "new"})
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "new text\n"
    for result in (written, edited):
        assert result.status == "ok", result.text
        assert "Do not commit" not in result.text
        assert "headless runner captures a workspace patch" in result.text
    from ouroboros.tools.edit_ops import workspace_edit_note
    from ouroboros.contracts.task_constraint import TaskConstraint
    ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(workspace))
    assert "Do not commit this self-worktree" in workspace_edit_note(ctx)


def test_cwd_argument_repair_and_lazy_artifact_schema(process_context):
    registry, _ctx, _workspace, _data = process_context
    result = registry.execute_result("run_command", {"cmd": ["unused"], "root": "system_repo"})
    assert result.code == "TOOL_ARG_ERROR" and "Use cwd=system_repo" in result.text
    description = registry.get_schema_by_name("write_file")["function"]["description"]
    assert "created lazily" in description and "artifact_store" in description

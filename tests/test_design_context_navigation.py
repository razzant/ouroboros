"""DESIGN orientation reaches real author/API projections, with source identity."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import context
from ouroboros.context_layout import task_design_navigation, task_design_source
from tests.test_book_context_capture import _plan
from tests.test_body_candidate import _registry, body as _body
from tests.test_doc_context import _make_env_and_memory

body = _body

DESIGN = "# Design\n## Attention\nBODY_RULE_NOT_DELIVERED\n```md\n## Fake\n```\n### Signals\nDETAIL_NOT_DELIVERED\n"


@pytest.mark.parametrize("role", ["main", "api", "declared_nanny"])
def test_design_map_reaches_all_author_modes_and_keeps_common_prefix(tmp_path, role):
    env, memory = _make_env_and_memory(tmp_path)
    path = env.repo_dir / "docs/DESIGN.md"
    path.write_text(DESIGN)
    task = {"id": "design-author", "type": "task", "text": "Investigate signals"}
    if role != "main":
        task.update(delegation_role="subagent", configured_subagent={"route": {
            "kind": "agent_session" if role == "declared_nanny" else "api_model"}})
    if role.startswith("declared"):
        task["task_contract"] = {"input_sources": "declared", "context_requires_self_body_docs": False}
    core = context._capture_context_core(env, memory, task, None, None)
    plan = _plan(env, core, task, "max")
    captured = {}
    for mode in ("max", "low", "nano"):
        messages = plan.messages_for(mode)
        text = json.dumps(messages, ensure_ascii=False)
        captured[mode] = text
        assert "## DESIGN.md (navigation map)" in text
        assert "- Attention — lines 2-8" in text
        assert "- Signals — lines 7-8" in text
        assert json.dumps(str(path), ensure_ascii=False)[1:-1] in text
        assert "BODY_RULE_NOT_DELIVERED" not in text and "DETAIL_NOT_DELIVERED" not in text
        assert "- Fake" not in text
        assert "availability is not evidence of reading or application" in text
        # A'/D/B remain reusable between task roots; source addresses ride C.
        blocks = plan.projection(mode).system_message()["content"]
        assert all("DESIGN.md (navigation map)" not in block["text"]
                   for block in blocks if block.get("cache_control"))
    path.write_text("## Later source revision\n")
    assert {mode: json.dumps(plan.messages_for(mode), ensure_ascii=False) for mode in captured} == captured


def _ctx(env, **kwargs):
    return SimpleNamespace(repo_dir=env.repo_dir, system_repo_dir=env.repo_dir,
                           drive_root=env.drive_root, **kwargs)


def test_selected_copy_bytes_win_and_default_false_cannot_hide_body(tmp_path):
    env, memory = _make_env_and_memory(tmp_path)
    (env.repo_dir / "docs/DESIGN.md").write_text("## Installed heading\n")
    copy = tmp_path / "selected-copy"
    (copy / "docs").mkdir(parents=True)
    (copy / "docs/DESIGN.md").write_text(DESIGN)
    task = {"id": "body-copy", "text": "Plan", "workspace_root": str(copy),
            "workspace_mode": "self_worktree", "task_contract": {"context_requires_self_body_docs": False},
            "metadata": {"workspace_copy": {"execution_root": str(copy), "source_is_system_repo": True}}}
    core = context._capture_context_core(env, memory, task, None, None)
    text = json.dumps(_plan(env, core, task, "max").messages_for("max"), ensure_ascii=False)
    assert json.dumps(str(copy / "docs/DESIGN.md"), ensure_ascii=False)[1:-1] in text and "- Attention" in text
    assert "Installed heading" not in text


@pytest.mark.parametrize("kind", ["ordinary", "foreign_copy", "folderless"])
def test_foreign_and_nobody_sources_never_receive_body_design(tmp_path, kind):
    env, memory = _make_env_and_memory(tmp_path)
    (env.repo_dir / "docs/DESIGN.md").write_text("## Installed policy\n")
    foreign = tmp_path / "another-project"
    (foreign / "docs").mkdir(parents=True)
    (foreign / "docs/DESIGN.md").write_text("## Foreign lookalike\n")
    task = {"id": "foreign", "text": "Plan"}
    if kind == "folderless":
        task["project_id"] = "project-without-folder"
    else:
        task.update(workspace_root=str(foreign), workspace_mode="external")
    if kind == "foreign_copy":
        task.update(workspace_mode="self_worktree", context_requires_self_body_docs=True,
                    metadata={"workspace_copy": {"execution_root": str(foreign), "source_is_system_repo": False}})
    core = context._capture_context_core(env, memory, task, None, None)
    assert "DESIGN.md" not in core.dynamic_head_text
    text = json.dumps(_plan(env, core, task, "max").messages_for("max"))
    assert "Installed policy" not in text and "Foreign lookalike" not in text


def test_missing_design_is_a_visible_source_gap_in_actual_author_input(tmp_path):
    env, memory = _make_env_and_memory(tmp_path)
    task = {"id": "missing-design", "text": "Plan"}
    core = context._capture_context_core(env, memory, task, None, None)
    text = json.dumps(_plan(env, core, task, "max").messages_for("max"))
    assert "DESIGN.md source unavailable" in text
    assert json.dumps(str(env.repo_dir / "docs/DESIGN.md"))[1:-1] in text
    assert "No DESIGN content or navigation delivered" in text


def test_explicit_folderless_self_work_points_to_system_source(tmp_path):
    env, _memory = _make_env_and_memory(tmp_path)
    (env.repo_dir / "docs/DESIGN.md").write_text(DESIGN)
    ctx = _ctx(env, project_id="folderless", task_metadata={})
    task = {"context_requires_self_body_docs": True}
    assert task_design_source(ctx, task) == env.repo_dir
    assert "- Attention" in task_design_navigation(ctx, task)


def test_explicit_flag_does_not_import_body_policy_into_ordinary_room_folder(tmp_path):
    env, _memory = _make_env_and_memory(tmp_path)
    foreign = tmp_path / "room-folder"
    (foreign / "docs").mkdir(parents=True)
    (foreign / "docs/DESIGN.md").write_text("## Foreign rules\n")
    ctx = _ctx(env, is_direct_chat=True, project_id="room",
               task_metadata={"_project_room_dir": str(foreign)})
    assert task_design_navigation(ctx, {"context_requires_self_body_docs": True}) == ""


def test_folderless_prepared_candidate_survives_restore_in_author_and_session_delivery(body):
    from pathlib import Path

    from ouroboros import body_candidate
    from ouroboros.agent import Env
    from ouroboros.memory import Memory
    from ouroboros.subagent_work_order import session_design_navigation
    from ouroboros.tools.tool_resolution import active_repo_dir_for
    from tests.body_candidate_support import make_ctx

    serving, data = body
    ctx = make_ctx(serving, data, "folderless-design", project_id="folderless")
    scratch = active_repo_dir_for(ctx)
    registry = _registry(serving, data, ctx)
    assert "is new" in registry.execute("prepare_self_change", {})
    candidate = Path(body_candidate.descriptor(ctx)["path"])
    source = candidate / "docs/DESIGN.md"
    source.parent.mkdir()
    source.write_text(DESIGN)
    env, memory = Env(repo_dir=serving, drive_root=data), Memory(drive_root=data, repo_dir=serving)
    task = {"id": ctx.task_id, "text": "Plan", "context_requires_self_body_docs": False}

    def delivered(bound):
        assert active_repo_dir_for(bound) == scratch  # preparing a body never reclassifies scratch
        assert task_design_source(bound, task) == candidate
        core = context._capture_context_core(env, memory, task, None, bound)
        for mode in ("max", "low", "nano"):
            text = json.dumps(_plan(env, core, task, mode).messages_for(mode))
            assert json.dumps(str(source))[1:-1] in text and "- Attention" in text
            assert "BODY_RULE_NOT_DELIVERED" not in text
        text = session_design_navigation(bound, target_root=str(scratch), source_root=str(scratch))
        assert str(source) in text and "- Attention" in text
        foreign = data.parent / "foreign"
        foreign.mkdir(exist_ok=True)
        assert session_design_navigation(bound, target_root=str(foreign), source_root=str(foreign)) == ""
        for foreign_kind in ("workspace", "room"):
            other = make_ctx(serving, data, ctx.task_id, project_id="folderless", is_direct_chat=True)
            body_candidate.restore(other)
            if foreign_kind == "workspace":
                other.workspace_root, other.workspace_mode = foreign, "external"
            else:
                other.task_metadata["_project_room_dir"] = str(foreign)
            assert active_repo_dir_for(other) == foreign
            assert task_design_source(other, task) is None

    delivered(ctx)
    restored = make_ctx(serving, data, ctx.task_id, project_id="folderless")
    assert body_candidate.restore(restored)["path"] == str(candidate)
    delivered(restored)
    assert "is resumed" in _registry(serving, data, restored).execute(
        "prepare_self_change", {"resume": ctx.task_id})
    delivered(restored)


@pytest.mark.parametrize("own_body", [True, False], ids=["recognized-fork", "foreign-repository"])
def test_isolated_copy_uses_body_identity_not_only_shared_git_directory(tmp_path, own_body):
    from ouroboros.subagent_work_order import session_design_navigation
    from ouroboros.workspace_copies import source_is_system_repo
    from tests.test_review_body_fact import FORK_URL, FOREIGN_URL, _git, _repo

    system = _repo(tmp_path / "system", remote=FORK_URL)
    source = _repo(tmp_path / "source", remote=FORK_URL if own_body else FOREIGN_URL)
    execution = tmp_path / "execution"
    _git(tmp_path, "clone", "-q", str(source), str(execution))
    (execution / "docs").mkdir()
    (execution / "docs/DESIGN.md").write_text(DESIGN, encoding="utf-8")
    # Admission's flag asks only whether the repositories share a Git directory.
    shared_git = source_is_system_repo(source, system)
    assert shared_git is False
    binding = {"source_root": str(source), "execution_root": str(execution),
               "source_is_system_repo": shared_git}
    ctx = SimpleNamespace(repo_dir=system, system_repo_dir=system,
                          workspace_root=execution, workspace_mode="self_worktree",
                          task_metadata={"workspace_copy": binding}, drive_root=tmp_path / "data")
    expected = execution if own_body else None
    assert task_design_source(ctx, {}) == expected
    nav = session_design_navigation(ctx, target_root=str(execution), source_root=str(execution))
    if own_body:
        assert str(execution / "docs/DESIGN.md") in nav and "- Attention" in nav
    else:
        assert nav == ""

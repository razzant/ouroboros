"""Public body-authoring boundaries: serving aliases, child sources and process env."""
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

from ouroboros import body_candidate, subagent_worktrees
from tests.body_candidate_support import candidate_commit, git, isolate, make_ctx, make_serving

pytestmark = pytest.mark.serial


@pytest.fixture
def body(tmp_path, monkeypatch):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    return serving, data


def registry(ctx):
    from ouroboros.tools.registry import ToolRegistry
    tools = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    tools.set_context(ctx)
    return tools


@pytest.mark.parametrize("tool", ["write_file", "edit_text", "apply_patch", "edit_batch"])
@pytest.mark.parametrize("spelling", ["basename", "absolute", "backslash", "whitespace"])
def test_all_edit_consumers_keep_serving_alias_on_first_and_later_writes(body, tool, spelling):
    serving, data = body
    ctx = make_ctx(serving, data, "aliases")
    tools = registry(ctx)

    def target(rel):
        value = str(serving / rel) if spelling == "absolute" else f"{serving.name}/{rel}"
        return value.replace("/", "\\") if spelling == "backslash" else f" {value} " if spelling == "whitespace" else value

    for rel in ("ouroboros/mod_a.py", "ouroboros/mod_b.py"):
        path = target(rel)
        args = {"write_file": {"path": path, "content": "GEN = 'GEN_CAND'\n"},
                "edit_text": {"path": path, "old_str": "GEN_OLD", "new_str": "GEN_CAND"},
                "edit_batch": {"edits": [{"path": path, "old_str": "GEN_OLD", "new_str": "GEN_CAND"}]},
                "apply_patch": {"patch": f"*** Begin Patch\n*** Update File: {path}\n@@\n-GEN = 'GEN_OLD'\n+GEN = 'GEN_CAND'\n*** End Patch"}}[tool]
        result = tools.execute_result(tool, args)
        assert result.status == "ok", result
        candidate = pathlib.Path(ctx.repo_dir)
        assert (candidate / rel).read_text() == "GEN = 'GEN_CAND'\n"
        assert (serving / rel).read_text() == "GEN = 'GEN_OLD'\n"
        assert not (candidate / serving.name).exists()


@pytest.mark.parametrize("prepared", [False, True])
def test_patch_add_delete_keep_serving_targets_and_real_nested_paths(body, prepared):
    serving, data = body
    ctx = make_ctx(serving, data, "patch-targets")
    tools = registry(ctx)
    if prepared:
        body_candidate.prepare(ctx)
    result = tools.execute_result("apply_patch", {"patch":
        f"*** Begin Patch\n*** Add File: {serving.name}/new.txt\n+hello\n"
        f"*** Delete File: {serving}/ouroboros/gone.py\n*** End Patch"})
    assert result.status == "ok", result
    candidate = pathlib.Path(ctx.repo_dir)
    assert (candidate / "new.txt").read_text() == "hello\n"
    assert not (candidate / "ouroboros/gone.py").exists()
    assert (serving / "ouroboros/gone.py").exists() and not (serving / "new.txt").exists()
    (candidate / serving.name).mkdir()
    nested = tools.execute_result("apply_patch", {"patch":
        f"*** Begin Patch\n*** Add File: {serving.name}/nested.txt\n+nested\n*** End Patch"})
    assert nested.status == "ok", nested
    assert (candidate / serving.name / "nested.txt").read_text() == "nested\n"


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("spelling", ["absolute", "relative", "sibling", "symlink"])
def test_explicit_serving_child_source_is_copied_from_candidate(body, monkeypatch, tmp_path, prepared, external, spelling):
    from ouroboros.tools.control_scheduling import _child_workspace
    from ouroboros.workspace_copies import admitted_copy_metadata
    from supervisor.events_subagent_admission import _resolve_subagent_constraint
    serving, data = body
    ctx = make_ctx(serving, data, "parent")
    if external:
        project = tmp_path / "project"
        project.mkdir()
        ctx.workspace_root, ctx.workspace_mode = project, "external"
    if prepared:
        body_candidate.prepare(ctx)
    selected = str(serving)
    if spelling in {"relative", "sibling"}:
        import os
        base = ctx.workspace_root or ctx.repo_dir
        # ``sibling`` names the checkout through its parent (``../repo`` from an unbound
        # parent), the spelling whose meaning the first binding would otherwise move.
        selected = (os.path.relpath(serving, base) if spelling == "relative"
                    else os.path.join(os.path.relpath(serving.parent, base), serving.name))
    elif spelling == "symlink":
        alias = tmp_path / "serving-alias"
        alias.symlink_to(serving, target_is_directory=True)
        selected = str(alias)
    params = {"workspace_root": selected, "write_surface": "self_worktree"}
    assert body_candidate.authoring_seam(ctx, "schedule_subagent", params) is None
    source, mode, _, error = _child_workspace(ctx, ctx.task_metadata, params)
    assert not error
    candidate = pathlib.Path(body_candidate.descriptor(ctx)["path"])
    assert pathlib.Path(source) == candidate
    (candidate / "parent-wip.txt").write_text("unfinished")
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    constraint, child, _, error = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=serving, DRIVE_ROOT=data), tid="child", parent_task_id="parent",
        requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
        workspace_root=source, workspace_mode=mode, base_sha="")
    assert not error
    copy = admitted_copy_metadata(child)
    assert pathlib.Path(copy["source_root"]) == candidate
    assert (pathlib.Path(child) / "parent-wip.txt").read_text() == "unfinished"
    assert constraint["write_root"] == child and not (serving / "parent-wip.txt").exists()
    # Drive the return consumer as well: the source binding is the apply target.
    from tests.test_isolated_project_copies import capture
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    (pathlib.Path(child) / "ouroboros/mod_b.py").write_text("GEN = 'CHILD'\n")
    task = {"id": "child", "parent_task_id": "parent", "workspace_root": child,
            "workspace_mode": "self_worktree", "task_constraint": constraint,
            "workspace_copy": copy, "metadata": {"workspace_copy": copy}}
    capture((serving, candidate, data, candidate.parent), pathlib.Path(child), task)
    applied = _integrate_subagent_patch(ctx, task_id="child")
    assert "✅ Integrated" in applied, applied
    assert (candidate / "ouroboros/mod_b.py").read_text() == "GEN = 'CHILD'\n"
    assert (candidate / "parent-wip.txt").read_text() == "unfinished"
    assert (serving / "ouroboros/mod_b.py").read_text() == "GEN = 'GEN_OLD'\n"


@pytest.mark.parametrize("selection", ["read_only", "omitted", "relative", "absolute"])
def test_unavailable_project_focus_keeps_own_body_child_on_the_candidate(body, monkeypatch, selection):
    import ouroboros.safety as safety
    from tests._shared import configure_test_subagent
    serving, data = body
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    ctx = make_ctx(serving, data, "parent", is_direct_chat=True, project_id="project-fixture")
    ctx.task_metadata["_project_room_note"] = "Selected project folder is unavailable"
    options = {"read_only": {}, "omitted": {"write_surface": "self_worktree"},
               "relative": {"write_surface": "self_worktree", "workspace_root": "relative"},
               "absolute": {"write_surface": "self_worktree", "workspace_root": str(serving)}}[selection]
    result = registry(ctx).execute_result("schedule_subagent", {
        "subagent_id": configure_test_subagent(monkeypatch), "objective": "Body work", "expected_output": "Patch",
        **options})
    if selection == "relative":
        assert result.code == "TOOL_ARG_ERROR" and "available parent folder" in result.text, result
        assert not ctx.pending_events and not body_candidate.is_bound(ctx)
        return
    assert result.status == "ok", result.text
    if selection == "read_only":
        assert not body_candidate.is_bound(ctx) and not body_candidate.descriptor(ctx)
        return
    candidate = body_candidate.descriptor(ctx)["path"]
    assert body_candidate.is_bound(ctx) and pathlib.Path(candidate) != serving
    assert ctx.pending_events[-1]["workspace_root"] == candidate


@pytest.mark.parametrize("parent", ["folderless_project", "metadata_serving", "metadata_foreign", "explicit_foreign",
                                    "explicit_missing", "explicit_sibling"])
def test_own_body_child_source_is_the_schedulers_own_selection(body, monkeypatch, tmp_path, parent):
    """The seam binds exactly when the scheduler's selection copies the serving checkout."""
    import ouroboros.safety as safety
    from tests._shared import configure_test_subagent
    serving, data = body
    foreign = tmp_path / "home" / "foreign"
    foreign.mkdir(parents=True)
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(foreign.parent))
    ctx = make_ctx(serving, data, "parent", project_id="project-fixture" if parent == "folderless_project" else "")
    if parent.startswith("metadata_"):
        ctx.task_metadata["workspace_root"] = str(serving if parent == "metadata_serving" else foreign)
    options = {"workspace_root": str(foreign / "absent" if parent == "explicit_missing" else foreign)
               } if parent.startswith("explicit_") else {}
    if parent == "explicit_sibling":  # read from the unbound parent's serving folder, before the seam binds
        options = {"workspace_root": f"../{serving.name}"}
    result = registry(ctx).execute_result("schedule_subagent", {
        "subagent_id": configure_test_subagent(monkeypatch), "objective": "Body work", "expected_output": "Patch",
        "write_surface": "self_worktree", **options})
    if parent == "explicit_missing":
        assert result.code == "TOOL_ARG_ERROR" and "not a directory" in result.text, result
        assert not ctx.pending_events and not body_candidate.is_bound(ctx)
        return
    assert result.status == "ok", result.text
    source = ctx.pending_events[-1]["workspace_root"]
    if parent.endswith("_foreign"):
        assert not body_candidate.is_bound(ctx) and source == str(foreign)
        return
    candidate = body_candidate.descriptor(ctx)["path"]
    assert body_candidate.is_bound(ctx) and source == candidate and pathlib.Path(candidate) != serving


def test_read_only_child_of_a_bound_parent_reads_the_candidate_and_writes_nothing(body, monkeypatch):
    import ouroboros.safety as safety
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent
    from tests.test_helper_start_folder import _admit_event
    serving, data = body
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    ctx = make_ctx(serving, data, "parent")
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(body_candidate.descriptor(ctx)["path"])
    (candidate / "ouroboros/mod_a.py").write_text("GEN = 'PARENT_WIP'\n")
    tools = registry(ctx)
    result = tools.execute_result("schedule_subagent", {
        "subagent_id": configure_test_subagent(monkeypatch), "objective": "Read the body", "expected_output": "Notes"})
    assert result.status == "ok", result.text
    row = _admit_event(tools, data, serving)
    constraint = normalize_task_constraint(row["task_constraint"])
    assert constraint.mode == "local_readonly_subagent" and row["workspace_root"] == str(candidate)
    child = ToolContext(repo_dir=serving, drive_root=data, task_id=row["id"], task_metadata=row["metadata"],
                        workspace_root=pathlib.Path(row["workspace_root"]), workspace_mode=row["workspace_mode"],
                        task_constraint=constraint)
    tools.set_context(child)
    assert "PARENT_WIP" in tools.execute("read_file", {"path": "ouroboros/mod_a.py"})
    for name, options in (("write_file", {"path": "ouroboros/mod_a.py", "content": "GEN = 'CHILD'\n"}),
                          ("run_command", {"cmd": [sys.executable, "-c", "open('made.txt', 'w')"]})):
        assert tools.execute_result(name, options).code == "ACCESS_BLOCKED", name  # read-only authority kept
    assert (candidate / "ouroboros/mod_a.py").read_text() == "GEN = 'PARENT_WIP'\n"
    assert not (candidate / "made.txt").exists() and (serving / "ouroboros/mod_a.py").read_text() == "GEN = 'GEN_OLD'\n"


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("executor", [False, True])
@pytest.mark.parametrize("consumer", ["run_command", "run_script", "verify_and_record", "start_service"])
def test_process_consumers_use_full_isolated_body_environment(body, monkeypatch, child, executor, consumer):
    from ouroboros.workspace_copies import admitted_copy_metadata
    from supervisor.events_subagent_admission import _resolve_subagent_constraint
    serving, data = body
    ctx = make_ctx(serving, data, "env-parent")
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    if child:
        monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
        constraint, root, mode, error = _resolve_subagent_constraint(
            SimpleNamespace(REPO_DIR=serving, DRIVE_ROOT=data), tid="env-child", parent_task_id="env-parent",
            requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
            workspace_root=str(candidate), workspace_mode="self_worktree", base_sha="")
        assert not error
        candidate = pathlib.Path(root)
        ctx = make_ctx(serving, data, "env-child", workspace_root=root, workspace_mode=mode,
                       task_constraint=constraint, task_metadata={"delegation_role": "subagent",
                       "root_task_id": "env-parent", "workspace_copy": admitted_copy_metadata(root)})
    if executor:
        # API roots use this backend when an external workspace mapping covers the copy.
        ctx.executor_ref = {"type": "local", "workspace_host_path": str(candidate), "workspace_backend_path": "/workspace"}
        if not child:
            project = serving.parent / "project"
            project.mkdir()
            ctx.workspace_root, ctx.workspace_mode = str(project), "external"
    monkeypatch.setenv("EXAMPLE_API_KEY", "synthetic-key-must-be-absent")
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    script = "import os, json, pathlib; pathlib.Path('environment.json').write_text(json.dumps(dict(os.environ)))"
    args = {"cmd": [sys.executable, "-c", script]} if consumer == "run_command" else {"script": script, "interpreter": sys.executable}
    if consumer == "verify_and_record":
        args = {"contract_kind": "explicit_command", "check": [sys.executable, "-c", script]}
    if consumer == "start_service":
        script += "; print('environment-written', flush=True)"
        args = {"name": "child-env", "cmd": [sys.executable, "-c", script],
                "readiness": {"log_contains": "environment-written", "timeout_sec": 2}}
    args["cwd"] = str(candidate)
    result = registry(ctx).execute_result(consumer, args)
    assert result.status == "ok", result
    if consumer == "start_service":
        registry(ctx).execute("stop_service", {"name": "child-env"})
    env = json.loads((candidate / "environment.json").read_text())
    root = candidate.with_name(candidate.name + ".env")
    assert pathlib.Path(env["HOME"]) == root / "home"
    assert pathlib.Path(env["OUROBOROS_DATA_DIR"]) == root / "data"
    assert pathlib.Path(env["OUROBOROS_REPO_DIR"]) == candidate
    assert pathlib.Path(env["OUROBOROS_SETTINGS_PATH"]) == root / "data/settings.json"
    assert "EXAMPLE_API_KEY" not in env and "OUROBOROS_MANAGED_BY_LAUNCHER" not in env
    assert not (serving / "environment.json").exists()


def test_gc_honours_nested_preflight_retention_marker(body):
    from ouroboros.task_results import write_task_result
    from ouroboros.test_environment import retain_tree
    serving, data = body
    ctx = make_ctx(serving, data, "retained")
    bound = body_candidate.prepare(ctx)
    env = body_candidate.process_environment(ctx, ctx.repo_dir)
    nested = pathlib.Path(env["TMPDIR"]) / "preflight" / "candidate"
    nested.mkdir(parents=True)
    (nested / "evidence.txt").write_text("unconfirmed reader")
    assert retain_tree(nested, "teardown unconfirmed")
    write_task_result(data, "retained", "completed", result="done")
    subagent_worktrees.prune_orphans(retention_days=0)
    assert (nested / "evidence.txt").read_text() == "unconfirmed reader"
    assert pathlib.Path(bound["path"]).exists()  # keep imports and registry discovery too
    assert body_candidate.find("retained")


@pytest.mark.parametrize("removal", ["remove_worktree", "prune_orphans"])
@pytest.mark.parametrize("retained", [False, True])
def test_own_body_child_copy_removal_disposes_its_sibling_environment(body, monkeypatch, removal, retained):
    from ouroboros.test_environment import retain_tree
    from ouroboros.workspace_copies import admitted_copy_metadata
    from supervisor.events_subagent_admission import _resolve_subagent_constraint
    serving, data = body
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    constraint, root, mode, error = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=serving, DRIVE_ROOT=data), tid="env-child", parent_task_id="env-parent",
        requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
        workspace_root="", workspace_mode="", base_sha="")
    assert not error
    child = make_ctx(serving, data, "env-child", workspace_root=root, workspace_mode=mode, task_constraint=constraint,
                     task_metadata={"delegation_role": "subagent", "root_task_id": "env-parent",
                                    "workspace_copy": admitted_copy_metadata(root)})
    env_root = pathlib.Path(body_candidate.process_environment(child, root)["HOME"]).parent
    assert env_root == pathlib.Path(root).with_name(pathlib.Path(root).name + ".env") and env_root.is_dir()
    if retained:
        nested = env_root / "tmp" / "preflight"
        nested.mkdir(parents=True)
        assert retain_tree(nested, "teardown unconfirmed")
    if removal == "remove_worktree":
        assert subagent_worktrees.remove_worktree(task_id="env-child") is True
    else:
        assert subagent_worktrees.prune_orphans(retention_days=0)["removed"] == 1
    assert not pathlib.Path(root).exists()
    assert env_root.is_dir() is retained  # a nested teardown's retention marker keeps the scratch it names


def test_gc_compares_to_serving_linked_head_not_common_head(body, tmp_path):
    from ouroboros.task_results import write_task_result
    common, data = body
    serving = tmp_path / "linked-serving"
    git(common, "worktree", "add", "-b", "serving", str(serving), "HEAD")
    ctx = make_ctx(serving, data, "linked")
    bound = body_candidate.prepare(ctx)
    sha = candidate_commit(pathlib.Path(bound["path"]))
    git(common, "merge", "--ff-only", sha)
    row = body_candidate.find("linked")
    assert body_candidate.unique_work(row)["unadopted_commits"] == 1
    write_task_result(data, "linked", "completed", result="done")
    subagent_worktrees.prune_orphans(retention_days=0)
    assert git(serving, "rev-parse", bound["branch"]) == sha
    assert git(serving, "show", body_candidate.PIN_PREFIX + bound["candidate_id"] + ":ouroboros/mod_a.py") == "GEN = 'GEN_CAND'"


def test_gc_rechecks_late_retention_and_discard_keeps_missing_checkout_scratch(body):
    from ouroboros.task_results import write_task_result
    from ouroboros.test_environment import retain_tree
    serving, data = body
    ctx = make_ctx(serving, data, "late-retention")
    bound = body_candidate.prepare(ctx)
    env = body_candidate.process_environment(ctx, ctx.repo_dir)
    write_task_result(data, ctx.task_id, "completed", result="done")
    row = body_candidate.find(ctx.task_id)
    verdict = body_candidate.retention_verdict(row, expired=True, data_dir=data)
    assert verdict["remove"]
    nested = pathlib.Path(env["TMPDIR"]) / "preflight"
    nested.mkdir()
    assert retain_tree(nested, "unconfirmed after verdict")
    assert not body_candidate.verdict_holds(row, verdict, data_dir=data)
    # Scratch retains independent evidence even after a checkout disappears.
    git(serving, "worktree", "remove", "--force", bound["path"])
    body_candidate.discard_scratch(row)
    assert (nested / "OUROBOROS_RETAINED.txt").exists()


def test_foreign_copy_keeps_source_and_explicit_process_environment(body, tmp_path, monkeypatch):
    from ouroboros.tools.control_scheduling import _child_workspace
    serving, data = body
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    ctx = make_ctx(serving, data, "foreign-parent", workspace_root=str(foreign), workspace_mode="external")
    params = {"workspace_root": str(foreign), "write_surface": "self_worktree"}
    assert body_candidate.authoring_seam(ctx, "schedule_subagent", params) is None
    source, _, _, error = _child_workspace(ctx, ctx.task_metadata, params)
    assert not error and pathlib.Path(source) == foreign and not body_candidate.is_bound(ctx)
    ctx.workspace_mode = "self_worktree"
    ctx.task_constraint = {"mode": "acting_subagent", "surface": "self_worktree", "write_root": str(foreign)}
    ctx.task_metadata = {"delegation_role": "subagent", "workspace_copy": {
        "execution_root": str(foreign), "source_root": str(tmp_path / "foreign-source"), "source_is_system_repo": False}}
    monkeypatch.setenv("EXAMPLE_API_KEY", "foreign-fixture-value")
    result = registry(ctx).execute_result("run_command", {"cmd": [sys.executable, "-c",
        "import os,pathlib; pathlib.Path('credential.txt').write_text(os.environ['EXAMPLE_API_KEY'])"]})
    assert result.status == "ok", result
    assert (foreign / "credential.txt").read_text() == "foreign-fixture-value"
    assert not foreign.with_name(foreign.name + ".env").exists()

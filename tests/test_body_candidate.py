"""Own-body candidates: real Git, real registry dispatch, no serving-tree effect (#1539).

Every scenario builds a miniature serving body under ``tmp_path`` and drives the
product owners themselves — the tool dispatcher, the private-copy records and
their GC, the child-copy admission, the commit and review seams.
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from ouroboros import body_candidate, subagent_worktrees
from tests.body_candidate_support import REPO, candidate_commit, git, isolate, make_ctx, make_serving


@pytest.fixture
def body(tmp_path, monkeypatch):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    return serving, data


def _tree_bytes(repo: pathlib.Path) -> dict:
    return {str(p.relative_to(repo)): p.read_bytes() for p in sorted(repo.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(repo).parts}


def _registry(serving, data, ctx):
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=serving, drive_root=data)
    registry.set_context(ctx)
    return registry


def _terminal(data, task_id, status="completed"):
    from ouroboros.task_results import write_task_result

    write_task_result(data, task_id, status, result="done")


# --------------------------------------------------------------------------- #
# Prepare-first: file, process, delegation
# --------------------------------------------------------------------------- #
def test_ordinary_body_write_lands_in_a_candidate_and_serving_bytes_do_not_change(body):
    serving, data = body
    before, head = _tree_bytes(serving), git(serving, "rev-parse", "HEAD")
    ctx = make_ctx(serving, data, "root-write")
    registry = _registry(serving, data, ctx)

    first = registry.execute("write_file", {"path": "ouroboros/mod_a.py", "content": "GEN = 'GEN_CAND'\n"})
    second = registry.execute("write_file", {"root": "system_repo", "path": "notes/new.txt", "content": "n\n"})

    assert "GEN_CAND" not in (serving / "ouroboros/mod_a.py").read_text()
    assert _tree_bytes(serving) == before and git(serving, "status", "--porcelain") == ""
    assert git(serving, "rev-parse", "HEAD") == head
    bound = body_candidate.descriptor(ctx)
    candidate = pathlib.Path(bound["path"])
    assert (candidate / "ouroboros/mod_a.py").read_text() == "GEN = 'GEN_CAND'\n", (first, second)
    assert (candidate / "notes/new.txt").read_text() == "n\n"
    assert bound["base_sha"] == head and git(candidate, "rev-parse", "--abbrev-ref", "HEAD") == bound["branch"]
    # One identity for every consumer: body root, commit branch, serving accessor.
    assert pathlib.Path(ctx.repo_dir) == pathlib.Path(ctx.system_repo_dir) == candidate
    assert ctx.branch_dev == bound["branch"]
    assert body_candidate.serving_repo_dir_for(ctx) == serving
    # A candidate is the own body, never an external workspace.
    from ouroboros.workspace_copies import source_is_system_repo

    assert source_is_system_repo(candidate, serving) and not ctx.is_workspace_mode()


def test_candidate_starts_from_the_accepted_commit_not_from_serving_dirt(body):
    serving, data = body
    (serving / "ouroboros/mod_b.py").write_text("GEN = 'OWNER_MANUAL_EDIT'\n")
    (serving / "owner-untracked.txt").write_text("mine\n")
    ctx = make_ctx(serving, data, "root-dirt")

    bound = body_candidate.prepare(ctx)

    candidate = pathlib.Path(bound["path"])
    assert (candidate / "ouroboros/mod_b.py").read_text() == "GEN = 'GEN_OLD'\n"
    assert not (candidate / "owner-untracked.txt").exists()
    assert git(candidate, "status", "--porcelain") == ""
    # The owner's manual edits are neither copied nor reset.
    assert (serving / "ouroboros/mod_b.py").read_text() == "GEN = 'OWNER_MANUAL_EDIT'\n"
    assert (serving / "owner-untracked.txt").read_text() == "mine\n"


def test_explicit_prepare_gives_processes_the_candidate_cwd_and_an_isolated_environment(body):
    serving, data = body
    before = _tree_bytes(serving)
    ctx = make_ctx(serving, data, "root-process")
    registry = _registry(serving, data, ctx)

    prepared = registry.execute("prepare_self_change", {})
    assert "body candidate" in prepared and "is new" in prepared
    assert "is bound" in registry.execute("prepare_self_change", {})  # idempotent
    candidate = pathlib.Path(body_candidate.descriptor(ctx)["path"])

    from ouroboros.tools.shell_process import _shell_env_for_cwd

    env = _shell_env_for_cwd(ctx, candidate / "ouroboros")
    env_root = candidate.with_name(candidate.name + ".env")
    assert pathlib.Path(env["OUROBOROS_DATA_DIR"]) == (env_root / "data").resolve()
    assert pathlib.Path(env["HOME"]) == (env_root / "home").resolve()
    assert pathlib.Path(env["OUROBOROS_REPO_DIR"]) == candidate.resolve()
    assert env["OUROBOROS_PYTEST_ACTIVE"] == "1"
    # A real process with that environment and the bound default cwd writes only the candidate.
    script = ("import os, pathlib; from ouroboros import mod_a; "
              "pathlib.Path('made-by-test.txt').write_text(os.environ['OUROBOROS_DATA_DIR']); "
              "pathlib.Path(os.environ['OUROBOROS_DATA_DIR'], 'touched').write_text(mod_a.__file__)")
    subprocess.run([sys.executable, "-c", script], cwd=str(ctx.repo_dir), env=env, check=True)
    assert (candidate / "made-by-test.txt").exists() and (env_root / "data" / "touched").exists()
    assert str(candidate) in (env_root / "data" / "touched").read_text()
    assert _tree_bytes(serving) == before and not (data / "touched").exists()
    # Outside the candidate nothing is isolated: the serving environment is not replaced.
    assert body_candidate.process_environment(ctx, serving) is None
    assert _shell_env_for_cwd(ctx, serving).get("HOME") == os.environ.get("HOME")


def test_prepare_names_the_folder_that_stays_active_beside_the_candidate(body, tmp_path):
    """A root bound to its project folder keeps that folder for default writes, commands
    without cwd and delegate_start; the receipt said every consumer moved to the candidate,
    and a delegated run then snapshotted the project folder the author did not expect."""
    serving, data = body
    room = tmp_path / "project-folder"
    room.mkdir()
    ctx = make_ctx(serving, data, "root-room", workspace_root=str(room), workspace_mode="external")
    assert ctx.is_workspace_mode()
    text = _registry(serving, data, ctx).execute("prepare_self_change", {})
    candidate = body_candidate.descriptor(ctx)["path"]
    assert "review_change with root='system_repo'" in text and f"workspace_root={candidate}" in text
    assert f"delegate_start runs still use {room}" in text and "default process cwd" not in text
    plain = make_ctx(serving, data, "root-plain")
    text = _registry(serving, data, plain).execute("prepare_self_change", {})
    assert "Body writes, default process cwd, child copies, review and commit_reviewed now use it" in text
    assert "still use" not in text


def test_acting_body_child_copies_the_candidate_and_returns_into_it(body, monkeypatch):
    serving, data = body
    ctx = make_ctx(serving, data, "root-delegate")
    assert body_candidate.authoring_seam(ctx, "schedule_subagent", {"write_surface": "self_worktree"}) is None
    candidate = pathlib.Path(body_candidate.descriptor(ctx)["path"])
    (candidate / "ouroboros/mod_a.py").write_text("GEN = 'PARENT_WIP'\n")  # uncommitted parent work

    from ouroboros.tools.control_scheduling import _child_workspace

    workspace_root, workspace_mode, parent_workspace, error = _child_workspace(ctx, ctx.task_metadata, {})
    assert (workspace_root, workspace_mode, error) == (str(candidate), "self_worktree", "")
    assert parent_workspace["root"] == str(candidate.resolve()) and parent_workspace["source"] == "system_repo"

    # The supervisor's real admission provisions the child from that source.
    from types import SimpleNamespace
    from supervisor.events_subagent_admission import _resolve_subagent_constraint

    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    constraint, child_root, child_mode, reject = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=serving, DRIVE_ROOT=data), tid="child-1",
        requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
        workspace_root=workspace_root, workspace_mode=workspace_mode, base_sha="", parent_task_id="root-delegate")
    assert reject == "" and child_mode == "self_worktree"
    assert (pathlib.Path(child_root) / "ouroboros/mod_a.py").read_text() == "GEN = 'PARENT_WIP'\n"

    from ouroboros.workspace_copies import admitted_copy_metadata

    copy = admitted_copy_metadata(child_root)
    # The recorded source is what integrate_subagent_patch applies to: the candidate, own body.
    assert pathlib.Path(copy["source_root"]) == candidate.resolve() and copy["source_is_system_repo"] is True
    assert "PARENT_WIP" not in (serving / "ouroboros/mod_a.py").read_text()
    # The child's own row is an ordinary task copy; the candidate row is not matched by a child id.
    assert subagent_worktrees.remove_worktree(task_id="root-delegate") is False
    assert candidate.is_dir()


@pytest.mark.parametrize("spelling", ["root_basename", "absolute"])
def test_serving_spellings_of_a_body_path_reach_the_same_candidate_file_before_and_after_binding(body, spelling):
    """``repo/server.py`` and the absolute serving path name the body file, on the first
    write (which binds the candidate) and on every later one (bound)."""
    serving, data = body
    before = _tree_bytes(serving)
    ctx = make_ctx(serving, data, f"root-spelling-{spelling}")
    registry = _registry(serving, data, ctx)

    def target(rel: str) -> str:
        return str(serving / rel) if spelling == "absolute" else f"{serving.name}/{rel}"

    entry = (serving / "server.py").read_text() + "# candidate entry\n"
    first = registry.execute_result("write_file", {"path": target("server.py"), "content": entry})
    later = registry.execute_result("write_file", {"path": target("ouroboros/mod_a.py"),
                                                   "content": "GEN = 'GEN_CAND'\n"})
    edited = registry.execute_result("edit_text", {"path": target("ouroboros/mod_b.py"),
                                                   "old_str": "GEN_OLD", "new_str": "GEN_EDIT"})

    candidate = pathlib.Path(body_candidate.descriptor(ctx)["path"])
    assert [r.status for r in (first, later, edited)] == ["ok", "ok", "ok"], (first, later, edited)
    assert (candidate / "server.py").read_text() == entry
    assert (candidate / "ouroboros/mod_a.py").read_text() == "GEN = 'GEN_CAND'\n"
    assert (candidate / "ouroboros/mod_b.py").read_text() == "GEN = 'GEN_EDIT'\n"
    assert not (candidate / serving.name).exists()
    assert _tree_bytes(serving) == before and git(serving, "status", "--porcelain") == ""


def test_candidate_refusals_reach_the_model_as_typed_blocked_results(body):
    """Automatic (first write) and explicit preparation refusals are registered codes,
    never a generic tool error, and leave the serving tree and foreign work untouched."""
    serving, data = body
    before = _tree_bytes(serving)
    ctx = make_ctx(serving, data, "root-occupied")
    occupied = subagent_worktrees._resolve_root() / f"body_{subagent_worktrees._safe_name('root-occupied')}"
    occupied.mkdir(parents=True)
    (occupied / "foreign.txt").write_text("not this task's\n")
    registry = _registry(serving, data, ctx)

    automatic = registry.execute_result("write_file", {"path": f"{serving.name}/server.py", "content": "# x\n"})
    explicit = registry.execute_result("prepare_self_change", {})
    missing = registry.execute_result("prepare_self_change", {"resume": "no-such-candidate"})

    for result, code in ((automatic, "CANDIDATE_PATH_OCCUPIED"), (explicit, "CANDIDATE_PATH_OCCUPIED"),
                         (missing, "CANDIDATE_MISSING")):
        assert (result.status, result.code) == ("blocked", code), result
        assert result.text.startswith(f"⚠️ {code}: "), result
    assert _tree_bytes(serving) == before and not body_candidate.is_bound(ctx)
    assert [path.name for path in occupied.iterdir()] == ["foreign.txt"]


def test_candidate_environment_failure_refuses_the_process_and_never_uses_the_serving_environment(body, monkeypatch):
    serving, data = body
    ctx = make_ctx(serving, data, "root-env-refused")
    registry = _registry(serving, data, ctx)
    assert "body candidate" in registry.execute("prepare_self_change", {})
    candidate = pathlib.Path(ctx.repo_dir)
    script = [sys.executable, "-c", "import pathlib; pathlib.Path('ran.txt').write_text('ran')"]
    service = {"name": "env-refused", "cwd": str(candidate), "cmd": script, "readiness": {"timeout_sec": 2}}

    from ouroboros import test_environment

    def unavailable(*_args, **_kwargs):
        raise OSError("isolated root could not be created")

    monkeypatch.setattr(test_environment, "isolated_environment", unavailable)
    started = registry.execute_result("start_service", dict(service))
    command = registry.execute_result("run_command", {"cmd": script})
    # The executor-backed (local) start refuses the same typed way, before any process or log.
    ctx.executor_ref = {"type": "local", "workspace_host_path": str(candidate), "workspace_backend_path": "/workspace"}
    executor = registry.execute_result("start_service", dict(service))
    ctx.executor_ref = None
    # No branch of the service start falls back to the serving environment, whatever fails.
    monkeypatch.setattr(body_candidate, "process_environment",
                        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("registry unreadable")))
    unknown = registry.execute_result("start_service", dict(service))
    registry.execute("stop_service", {"name": "env-refused"})

    for result in (started, command, executor):
        assert (result.status, result.code) == ("blocked", "CANDIDATE_ENVIRONMENT_UNAVAILABLE"), result
        # A missing process is not whole-call no-effect: a service replacement
        # may already have retired its previous instance before this refusal.
        assert "operation_outcome" not in result.meta, result
    assert unknown.status == "error" and "registry unreadable" in unknown.text, unknown  # fails closed
    assert not (candidate / "ran.txt").exists()
    assert not list((data / "services").rglob("env-refused.executor.log"))


def test_every_candidate_refusal_code_is_a_registered_blocked_tool_code():
    import re

    from ouroboros.tools.tool_result import TOOL_CODE_SPECS

    source = pathlib.Path(body_candidate.__file__).read_text(encoding="utf-8")
    codes = set(re.findall(r'CandidateRefused\(\s*"([A-Z_]+)"', source))
    assert {"CANDIDATE_MISSING", "CANDIDATE_PATH_OCCUPIED", "CANDIDATE_ENVIRONMENT_UNAVAILABLE"} <= codes
    statuses = {code: getattr(TOOL_CODE_SPECS.get(code), "status", None) for code in codes}
    assert statuses == dict.fromkeys(codes, "blocked")


@pytest.mark.parametrize("prepared", [False, True])
def test_pr_integration_runs_in_the_candidate_and_never_moves_the_serving_checkout(body, prepared):
    """fetch → integration branch → cherry-pick → adaptation → staged merge, as the tools chain
    them, from an unbound task and from a prepared one: the serving checkout keeps its branch
    and bytes, the integration branch starts at the candidate, the contributor keeps the credit."""
    serving, data = body
    git(serving, "checkout", "-q", "-b", "pr/7")  # what fetch_pr_ref leaves: the PR head as a local ref
    (serving / "ouroboros/contrib.py").write_text("X = 7\n")
    git(serving, "add", "-A")
    git(serving, "commit", "-q", "-m", "contribution",
        env={"GIT_AUTHOR_NAME": "Contributor", "GIT_AUTHOR_EMAIL": "contrib@example.invalid"})
    pr_sha = git(serving, "rev-parse", "HEAD")
    git(serving, "checkout", "-q", "ouroboros")
    before, head = _tree_bytes(serving), git(serving, "rev-parse", "HEAD")
    ctx = make_ctx(serving, data, f"root-pr-{prepared}")
    registry = _registry(serving, data, ctx)
    if prepared:
        assert "body candidate" in registry.execute("prepare_self_change", {})

    created = registry.execute_result("create_integration_branch", {"pr_number": 7})
    picked = registry.execute_result("cherry_pick_pr_commits", {"shas": [pr_sha]})
    registry.execute("write_file", {"path": "ouroboros/contrib.py", "content": "X = 8\n"})  # the adaptation
    adapted = registry.execute_result("stage_adaptations", {})
    staged = registry.execute_result("stage_pr_merge", {"branch": "integrate/pr-7"})

    assert [r.status for r in (created, picked, adapted, staged)] == ["ok"] * 4, (created, picked, adapted, staged)
    assert _tree_bytes(serving) == before and git(serving, "status", "--porcelain") == ""
    assert (git(serving, "rev-parse", "HEAD"), git(serving, "rev-parse", "--abbrev-ref", "HEAD")) == (head, "ouroboros")
    bound = body_candidate.descriptor(ctx)
    candidate = pathlib.Path(bound["path"])
    assert bound["branch"] in adapted.text and "always `ouroboros`" not in adapted.text
    from ouroboros.tools.git_pr import get_tools
    schema = next(tool.schema for tool in get_tools() if tool.name == "stage_pr_merge")
    assert "always `ouroboros`" not in schema["description"]
    assert "candidate" in schema["description"]
    assert git(candidate, "rev-parse", "--abbrev-ref", "HEAD") == bound["branch"]
    assert git(candidate, "merge-base", "integrate/pr-7", bound["branch"]) == bound["base_sha"]
    assert git(candidate, "rev-parse", "MERGE_HEAD") == git(candidate, "rev-parse", "integrate/pr-7")
    credited = git(candidate, "log", "-1", "--format=%an <%ae>", "integrate/pr-7")
    assert credited == "Contributor <contrib@example.invalid>"
    assert "Co-authored-by: Contributor <contrib@example.invalid>" in staged.text
    assert (candidate / "ouroboros/contrib.py").read_text() == "X = 8\n"
    from ouroboros.review_substrate import review_repo_dirs_for

    assert review_repo_dirs_for(ctx) == (serving.resolve(), candidate.resolve())  # review reads the staged merge here
    assert "ouroboros/contrib.py" in git(candidate, "diff", "--cached", "--name-only")


def test_seam_leaves_reads_processes_and_foreign_roots_alone(body):
    serving, data = body
    ctx = make_ctx(serving, data, "root-reads")
    for name, args in (("read_file", {"path": "server.py"}), ("run_command", {"cmd": "pytest"}),
                       ("write_file", {"root": "task_drive", "path": "x.txt", "content": "x"}),
                       ("schedule_subagent", {"objective": "read only"})):
        assert body_candidate.authoring_seam(ctx, name, args) is None
        assert not body_candidate.is_bound(ctx), name
    assert body_candidate.list_candidates() == []
    # No task identity: the legacy contract, no candidate.
    anonymous = make_ctx(serving, data, "", task_metadata={})
    assert body_candidate.authoring_seam(anonymous, "write_file", {"path": "a.txt", "content": "a"}) is None
    assert not body_candidate.is_bound(anonymous)


# --------------------------------------------------------------------------- #
# Authority is preserved: modes, protected paths, deliberate Cyber exception
# --------------------------------------------------------------------------- #
def test_light_mode_refuses_as_before_and_prepares_nothing(body, monkeypatch):
    serving, data = body
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    ctx = make_ctx(serving, data, "root-light")
    registry = _registry(serving, data, ctx)

    assert "LIGHT_MODE_BLOCKED" in registry.execute(
        "write_file", {"path": "ouroboros/mod_a.py", "content": "x\n"})
    assert "LIGHT_MODE_BLOCKED" in registry.execute("prepare_self_change", {})
    assert body_candidate.list_candidates() == [] and not body_candidate.is_bound(ctx)
    assert (serving / "ouroboros/mod_a.py").read_text() == "GEN = 'GEN_OLD'\n"


def test_protected_path_policy_applies_to_the_candidate_as_to_the_body(body, monkeypatch):
    serving, data = body
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    ctx = make_ctx(serving, data, "root-protected")
    registry = _registry(serving, data, ctx)

    refused = registry.execute("write_file", {"path": "BIBLE.md", "content": "# rewritten\n"})

    assert "CORE_PROTECTION_BLOCKED" in refused
    assert (serving / "BIBLE.md").read_text() == "# Constitution\n"
    candidate = pathlib.Path(body_candidate.descriptor(ctx)["path"])
    assert (candidate / "BIBLE.md").read_text() == "# Constitution\n"


def test_bound_task_cannot_reach_the_serving_checkout_through_another_root(body, monkeypatch, tmp_path):
    serving, data = body
    monkeypatch.setenv("HOME", str(tmp_path))  # the serving checkout sits under the owner's home, as installed
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    before = _tree_bytes(serving)
    ctx = make_ctx(serving, data, "root-escape")
    registry = _registry(serving, data, ctx)
    registry.execute("prepare_self_change", {})
    candidate = pathlib.Path(ctx.repo_dir)
    target = serving / "ouroboros/mod_a.py"

    by_label = registry.execute("write_file", {"root": "user_files", "path": str(target), "content": "X = 1\n"})
    # The shell write guard reads POSIX redirects; Windows has no `sh` to exercise it with.
    by_shell = (registry.execute("run_command", {"cmd": ["sh", "-c", f"echo hacked > {target}"]})
                if os.name != "nt" else "BLOCKED (POSIX shell redirect guard not exercised on Windows)")
    in_candidate = registry.execute("run_command", {"cmd": [
        sys.executable, "-c",
        "import os, pathlib; print(os.getcwd()); pathlib.Path('made-here.txt').write_text('ok\\n')"]})

    assert "USER_FILES_PATH_BLOCKED" in by_label or "blocked" in by_label.lower(), by_label
    assert "BLOCKED" in by_shell, by_shell
    assert _tree_bytes(serving) == before
    # A real process through the dispatcher runs in the candidate by default.
    assert str(candidate) in in_candidate and (candidate / "made-here.txt").read_text() == "ok\n", in_candidate
    # The same guards still protect the candidate as the body it is.
    from ouroboros.tools.registry_guards import _git_protected_roots

    protected = {pathlib.Path(root).resolve() for root in _git_protected_roots(registry)}
    assert {serving.resolve(), candidate.resolve()} <= protected
    from ouroboros.tool_access_paths import workspace_mode_block_reason

    ctx.workspace_root, ctx.workspace_mode = serving / "ouroboros", "external"
    assert "serving repo" in workspace_mode_block_reason(ctx)


def test_in_place_is_cyber_pros_explicit_decision_only(body, monkeypatch):
    serving, data = body
    ctx = make_ctx(serving, data, "root-inplace")
    registry = _registry(serving, data, ctx)
    refusal = registry.execute_result("prepare_self_change", {"in_place": True})
    assert (refusal.status, refusal.code) == ("blocked", "ACCESS_BLOCKED")
    assert "IN_PLACE_REQUIRES_CYBER_PRO" in refusal.text

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    assert "writes the serving checkout directly" in registry.execute("prepare_self_change", {"in_place": True})
    registry.execute("write_file", {"path": "notes/in-place.txt", "content": "direct\n"})
    assert (serving / "notes/in-place.txt").read_text() == "direct\n"
    assert body_candidate.list_candidates() == []


def test_a_bound_candidate_cannot_be_switched_to_in_place_by_a_second_prepare(body, monkeypatch):
    serving, data = body
    ctx = make_ctx(serving, data, "root-already-bound")
    registry = _registry(serving, data, ctx)
    assert "body candidate" in registry.execute("prepare_self_change", {})
    bound = body_candidate.descriptor(ctx)

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    refusal = registry.execute_result("prepare_self_change", {"in_place": True})
    assert (refusal.status, refusal.code) == ("blocked", "ACCESS_BLOCKED")
    assert "CANDIDATE_ALREADY_BOUND" in refusal.text
    assert body_candidate.descriptor(ctx) == bound


def test_subagents_and_the_update_resolver_never_get_a_candidate_of_their_own(body):
    serving, data = body
    child = make_ctx(serving, data, "child-9", task_metadata={"root_task_id": "root-9", "delegation_role": "subagent"},
                     task_constraint={"mode": "acting_subagent", "surface": "self_worktree",
                                      "write_root": str(serving), "base_sha": ""})
    with pytest.raises(body_candidate.CandidateRefused) as refused:
        body_candidate.prepare(child)
    assert refused.value.code == "CANDIDATE_NOT_APPLICABLE"
    assert body_candidate.authoring_seam(child, "write_file", {"path": "a.txt", "content": "a"}) is None


@pytest.mark.parametrize("mode", ["acting_subagent", "local_readonly_subagent"])
def test_a_child_in_cyber_pro_is_told_where_it_works_and_resume_stays_refused(body, monkeypatch, mode):
    """The tool stays in a Cyber Pro child's catalog; an ordinary call is an OK fact, not a failed call."""
    serving, data = body
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    constraint = {"mode": mode}
    if mode == "acting_subagent":
        constraint.update(surface="self_worktree", write_root=str(serving), base_sha="")
    child = make_ctx(serving, data, "child-cyber", task_metadata={"root_task_id": "root-c", "delegation_role": "subagent"},
                     task_constraint=constraint)
    if mode == "local_readonly_subagent":  # not offered to a read-only child; its handler answers alike if reached
        from ouroboros.tools.control_runtime import _prepare_self_change

        assert _prepare_self_change(child).startswith("OK: no separate body candidate is prepared for a subagent")
        assert "⚠️ CANDIDATE_NOT_APPLICABLE" in _prepare_self_change(child, resume="cand-of-another")
        assert body_candidate.list_candidates() == []
        return
    registry = _registry(serving, data, child)  # Cyber Pro's acting child keeps the whole catalog (P0)
    assert "prepare_self_change" in {tool["function"]["name"] for tool in registry.schemas()}
    ordinary = registry.execute_result("prepare_self_change", {})
    assert (ordinary.status, ordinary.code) == ("ok", "OK"), ordinary.text
    assert "the parent owns candidate preparation and integration" in ordinary.text
    assert body_candidate.list_candidates() == [] and not body_candidate.is_bound(child)

    resumed = registry.execute_result("prepare_self_change", {"resume": "cand-of-another"})
    assert (resumed.status, resumed.code) == ("blocked", "CANDIDATE_NOT_APPLICABLE"), resumed.text


# --------------------------------------------------------------------------- #
# Durable binding: retry / new worker / continuation / explicit resume
# --------------------------------------------------------------------------- #
def test_retry_or_new_worker_rebinds_the_same_candidate_and_children_do_not(body):
    serving, data = body
    first = make_ctx(serving, data, "root-retry")
    bound = body_candidate.prepare(first)
    (pathlib.Path(bound["path"]) / "wip.txt").write_text("unfinished\n")

    again = make_ctx(serving, data, "root-retry")
    assert body_candidate.restore(again)["path"] == bound["path"]
    assert pathlib.Path(again.repo_dir) == pathlib.Path(bound["path"]) and again.branch_dev == bound["branch"]
    assert (pathlib.Path(again.repo_dir) / "wip.txt").read_text() == "unfinished\n"

    child = make_ctx(serving, data, "child-of-retry",
                     task_metadata={"root_task_id": "root-retry", "delegation_role": "subagent"})
    assert body_candidate.restore(child) == {} and pathlib.Path(child.repo_dir) == serving
    stranger = make_ctx(serving, data, "another-root")
    assert body_candidate.restore(stranger) == {}


def test_lost_checkout_starts_a_new_candidate_and_keeps_the_old_branch(body):
    serving, data = body
    first = body_candidate.prepare(make_ctx(serving, data, "root-lost"))
    tip = candidate_commit(pathlib.Path(first["path"]))
    shutil.rmtree(first["path"])

    again_ctx = make_ctx(serving, data, "root-lost")
    assert body_candidate.restore(again_ctx) == {}  # nothing to bind; the serving tree is not a fallback target
    again = body_candidate.prepare(again_ctx)

    assert again["state"] == "new" and again["previous_branch"] == first["branch"]
    assert again["branch"] != first["branch"] and git(serving, "rev-parse", first["branch"]) == tip
    assert pathlib.Path(again["path"]).is_dir() and len(body_candidate.list_candidates()) == 1


def test_candidate_interrupted_before_its_checkout_is_populated_before_anyone_authors_it(body, monkeypatch):
    serving, data = body
    real_populate = body_candidate._populate
    monkeypatch.setattr(body_candidate, "_populate", lambda row, data_dir=None: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        body_candidate.prepare(make_ctx(serving, data, "root-crash"))  # died between admin dir and checkout
    row = body_candidate.find("root-crash")
    assert row and not row.get("ready") and not (pathlib.Path(row["path"]) / "server.py").exists()
    monkeypatch.setattr(body_candidate, "_populate", real_populate)

    ctx = make_ctx(serving, data, "root-crash")
    bound = body_candidate.restore(ctx)  # the next worker: the EMPTY index must never be handed out

    assert (pathlib.Path(bound["path"]) / "server.py").exists()
    assert git(bound["path"], "status", "--porcelain") == "" and body_candidate.find("root-crash")["ready"] is True


def test_parallel_first_writes_prepare_exactly_one_candidate(body):
    import threading

    serving, data = body
    ctx = make_ctx(serving, data, "root-parallel")
    results, errors = [], []

    def write(index):
        try:
            results.append(body_candidate.authoring_seam(ctx, "write_file", {"path": f"f{index}.txt", "content": "x"}))
        except Exception as exc:  # pragma: no cover - a failure is the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert errors == [] and results == [None] * 6
    assert len(body_candidate.list_candidates()) == 1
    assert git(ctx.repo_dir, "status", "--porcelain") == ""  # bound only after the checkout was populated


@pytest.mark.parametrize("dirty", [False, True])
def test_continuation_inherits_the_exact_predecessor_candidate(body, dirty):
    serving, data = body
    predecessor = make_ctx(serving, data, "pred-1")
    bound = body_candidate.prepare(predecessor)
    candidate = pathlib.Path(bound["path"])
    tip = candidate_commit(candidate)
    if dirty:
        (candidate / "ouroboros/mod_b.py").write_text("GEN = 'UNCOMMITTED'\n")
        (candidate / "fresh.bin").write_bytes(bytes(range(256)) * 8)
    successor = make_ctx(serving, data, "succ-1",
                         task_contract={"predecessor_authority": {"source": {"task_id": "pred-1"}}})

    with pytest.raises(body_candidate.CandidateRefused) as live:
        body_candidate.prepare(successor)
    assert live.value.code == "CANDIDATE_OWNER_LIVE"
    assert body_candidate.find("pred-1")["task_id"] == "pred-1"

    _terminal(data, "pred-1")
    inherited = body_candidate.prepare(successor)

    assert inherited["path"] == bound["path"] and inherited["state"] == "resumed"
    row = body_candidate.find("succ-1")
    assert row["owners"] == ["pred-1", "succ-1"] and body_candidate.find("pred-1") is None
    assert git(candidate, "rev-parse", "HEAD") == tip
    if dirty:
        assert (candidate / "ouroboros/mod_b.py").read_text() == "GEN = 'UNCOMMITTED'\n"
        assert (candidate / "fresh.bin").read_bytes() == bytes(range(256)) * 8


def test_unrelated_task_never_inherits_and_explicit_resume_names_one_exact_candidate(body):
    serving, data = body
    owner = make_ctx(serving, data, "direct-turn-1")
    bound = body_candidate.prepare(owner)
    (pathlib.Path(bound["path"]) / "wip.txt").write_text("turn one\n")
    _terminal(data, "direct-turn-1")

    # A later turn in the same room is NOT a continuation: no implicit adoption by room or recency.
    later = make_ctx(serving, data, "direct-turn-2")
    fresh = body_candidate.prepare(later)
    assert fresh["state"] == "new" and fresh["path"] != bound["path"]

    third = make_ctx(serving, data, "direct-turn-3")
    with pytest.raises(body_candidate.CandidateRefused) as missing:
        body_candidate.prepare(third, resume="no-such-candidate")
    assert missing.value.code == "CANDIDATE_MISSING"
    resumed = body_candidate.prepare(third, resume=bound["candidate_id"])
    assert resumed["path"] == bound["path"] and resumed["state"] == "resumed"
    assert (pathlib.Path(third.repo_dir) / "wip.txt").read_text() == "turn one\n"
    # One task authors one candidate.
    with pytest.raises(body_candidate.CandidateRefused) as owned:
        body_candidate.prepare(later, resume=bound["candidate_id"])
    assert owned.value.code == "CANDIDATE_ALREADY_OWNED"


def test_a_fresh_bind_names_open_neighbours_with_the_paths_they_changed_now(body):
    """Two foci opened candidates 43 seconds apart and edited the same files (audit 05.11, item 18): the second
    focus's first body write and its explicit receipt now name the open neighbour with the paths it changed at
    that moment; an ended owner's candidate is only counted, and a later write of the same task adds nothing."""
    serving, data = body
    ended = make_ctx(serving, data, "ended-task")
    body_candidate.prepare(ended)
    _terminal(data, "ended-task")
    wake = make_ctx(serving, data, "focus-wake")
    quiet = _registry(serving, data, wake).execute("write_file", {"path": "ouroboros/mod_b.py", "content": "GEN = 'W'\n"})
    assert "Other body candidates" not in quiet  # an ended neighbour alone is the Runtime fact's, not a note
    candidate = pathlib.Path(body_candidate.descriptor(wake)["path"])
    candidate_commit(candidate, files={"ouroboros/mod_b.py": "GEN = 'W'\n"})
    (candidate / "ouroboros/mod_a.py").write_text("GEN = 'W2'\n")
    (candidate / "scratch.txt").write_text("plan\n")
    records = sorted(map(str, body_candidate.list_candidates()))

    owner = make_ctx(serving, data, "owner-project")
    registry = _registry(serving, data, owner)
    first = registry.execute("write_file", {"path": "ouroboros/mod_a.py", "content": "GEN = 'O'\n"})
    line = (f"- body_focus-wake: task focus-wake (no task record), branch candidate/focus-wake, base "
            f"{git(serving, 'rev-parse', 'HEAD')[:12]}, path {candidate}: changed ouroboros/mod_a.py, "
            "ouroboros/mod_b.py, scratch.txt")
    assert f"Other body candidates, read now (never stored):\n{line}" in first and "ended-task" not in first
    assert "Other body candidates" not in registry.execute("write_file", {"path": "x.txt", "content": "x\n"})
    receipt = registry.execute("prepare_self_change", {})
    assert receipt.endswith(f"\n{line}\n- 1 more belong to ended tasks: retained work, listed in the Runtime "
                            "body_candidates fact; prepare_self_change(resume=<id>) continues one.")
    # Read at the moment, never stored: the neighbour's record holds no paths (only the new candidate is added).
    assert [row for row in map(str, body_candidate.list_candidates()) if "owner-project" not in row] == records


def test_a_neighbour_read_is_bounded_and_says_what_it_could_not_read(body, monkeypatch):
    serving, data = body
    wake = make_ctx(serving, data, "focus-wake")
    candidate = pathlib.Path(body_candidate.prepare(wake)["path"])
    for index in range(10):
        (candidate / f"n{index}.txt").write_text("x\n")
    _terminal(data, "someone-else")  # the task store exists: the neighbour's own record is what is missing
    owner = make_ctx(serving, data, "owner-project")
    note = body_candidate.neighbour_note(owner)
    assert "task focus-wake (no task record)" in note and note.endswith(
        "changed n0.txt, n1.txt, n2.txt, n3.txt, n4.txt, n5.txt, n6.txt, n7.txt (+2 more in that path)")
    with monkeypatch.context() as patch:
        patch.setattr(body_candidate, "_NEIGHBOUR_READ_SECONDS", 0.0)
        assert body_candidate.neighbour_note(owner).endswith(": changed paths not read (time bound)")
    with monkeypatch.context() as patch:
        import ouroboros.task_store as task_store

        patch.setattr(task_store, "task_drive_status", lambda *_a, **_k: 1 / 0)
        assert "task focus-wake (status unreadable)" in body_candidate.neighbour_note(owner)  # unknown is not ended
    shutil.rmtree(candidate)
    assert body_candidate.neighbour_note(owner).endswith(": its checkout is missing")


# --------------------------------------------------------------------------- #
# Retention: unique work outlives age; an incomplete capture keeps the original
# --------------------------------------------------------------------------- #
def _expire_all():
    return subagent_worktrees.prune_orphans(retention_days=0)


def test_gc_removes_only_candidates_without_unique_work(body):
    serving, data = body
    clean = body_candidate.prepare(make_ctx(serving, data, "gc-clean"))
    junk_ctx = make_ctx(serving, data, "gc-junk")
    junk = body_candidate.prepare(junk_ctx)
    (pathlib.Path(junk["path"]) / "ouroboros/__pycache__").mkdir()
    (pathlib.Path(junk["path"]) / "ouroboros/__pycache__/mod_a.cpython-311.pyc").write_bytes(b"\0cache")
    env_root = pathlib.Path(body_candidate.process_environment(junk_ctx, junk_ctx.repo_dir)["HOME"]).parent

    kept_fresh = subagent_worktrees.prune_orphans(retention_days=30)
    assert kept_fresh["removed"] == 0 and pathlib.Path(clean["path"]).is_dir()

    # A clean checkout still belongs to its live owner; age cannot remove its working address.
    assert _expire_all() == {"removed": 0, "kept": 2}
    _terminal(data, "gc-clean")
    _terminal(data, "gc-junk")
    report = _expire_all()

    # A name the patch rules leave out is no evidence the file is temporary: that candidate stays.
    assert report["removed"] == 1
    assert not pathlib.Path(clean["path"]).exists() and pathlib.Path(junk["path"]).is_dir() and env_root.is_dir()
    (pathlib.Path(junk["path"]) / "ouroboros/__pycache__/mod_a.cpython-311.pyc").unlink()
    assert _expire_all()["removed"] == 1

    assert not pathlib.Path(clean["path"]).exists() and not pathlib.Path(junk["path"]).exists()
    assert not env_root.exists()
    assert git(serving, "branch", "--list", "candidate/*") == ""
    assert body_candidate.list_candidates() == []


def test_gc_keeps_clean_unadopted_commits_despite_temporary_refs(body):
    serving, data = body
    bound = body_candidate.prepare(make_ctx(serving, data, "gc-committed"))
    candidate = pathlib.Path(bound["path"])
    tip = candidate_commit(candidate)
    git(serving, "branch", "temporary-copy", tip)
    _terminal(data, "gc-committed")
    work = body_candidate.unique_work(body_candidate.find("gc-committed"))
    assert work["unadopted_commits"] == 1 and work["tracked"] == work["loose"] == []
    assert _expire_all()["removed"] == 1
    assert git(serving, "rev-parse", bound["branch"]) == tip
    assert git(serving, "show", body_candidate.PIN_PREFIX + bound["candidate_id"] +
               ":ouroboros/mod_a.py") == "GEN = 'GEN_CAND'"


def test_clean_commit_reachable_from_serving_head_can_expire(body):
    serving, data = body
    bound = body_candidate.prepare(make_ctx(serving, data, "gc-adopted"))
    tip = candidate_commit(pathlib.Path(bound["path"]))
    git(serving, "merge", "--ff-only", tip)
    _terminal(data, "gc-adopted")
    assert body_candidate.unique_work(body_candidate.find("gc-adopted"))["unadopted_commits"] == 0
    assert _expire_all()["removed"] == 1
    assert git(serving, "rev-parse", "HEAD") == tip


def test_gc_preserves_dirty_untracked_ignored_binary_and_committed_work_before_deleting(body):
    serving, data = body
    bound = body_candidate.prepare(make_ctx(serving, data, "gc-unique"))
    candidate = pathlib.Path(bound["path"])
    tip = candidate_commit(candidate)
    binary = os.urandom(3 * 1024 * 1024)  # oversize for a patch artifact, ordinary for Git
    (candidate / "ouroboros/mod_b.py").write_text("GEN = 'DIRTY'\n")
    (candidate / "untracked.txt").write_text("loose\n")
    (candidate / "big.bin").write_bytes(binary)
    (candidate / "local-notes").mkdir()
    (candidate / "local-notes/ignored.md").write_text("ignored but unique\n")
    (candidate / "ouroboros/gone.py").unlink()

    # While its owner has no terminal record the work is simply kept: unknown is not ended.
    assert _expire_all() == {"removed": 0, "kept": 1} and candidate.is_dir()
    assert git(serving, "for-each-ref", body_candidate.PIN_PREFIX) == ""
    _terminal(data, "gc-unique")

    report = _expire_all()

    pin = body_candidate.PIN_PREFIX + bound["candidate_id"]
    assert report["removed"] == 1 and not candidate.exists()
    assert git(serving, "rev-parse", f"{pin}^") == tip
    assert git(serving, "show", f"{pin}:ouroboros/mod_b.py") == "GEN = 'DIRTY'"
    assert git(serving, "show", f"{pin}:untracked.txt") == "loose"
    assert git(serving, "show", f"{pin}:local-notes/ignored.md") == "ignored but unique"
    shown = subprocess.run(["git", "show", f"{pin}:big.bin"], cwd=serving, capture_output=True, check=True).stdout
    assert shown == binary
    assert git(serving, "ls-tree", "-r", "--name-only", pin, "--", "ouroboros/gone.py") == ""
    # The reviewed commits stay on their branch; nothing reached the serving tree.
    assert git(serving, "rev-parse", bound["branch"]) == tip
    assert git(serving, "status", "--porcelain") == ""


def test_incomplete_capture_retains_the_original_directory(body):
    serving, data = body
    bound = body_candidate.prepare(make_ctx(serving, data, "gc-incomplete"))
    candidate = pathlib.Path(bound["path"])
    (candidate / "untracked.txt").write_text("loose\n")
    nested = candidate / "nested-repo"
    nested.mkdir()
    git(nested, "init", "-q")
    (nested / "inner.txt").write_text("git cannot hold this as a file\n")
    git(nested, "add", "-A")
    git(nested, "commit", "-q", "-m", "inner")
    _terminal(data, "gc-incomplete")

    report = _expire_all()

    assert report["removed"] == 0 and report["kept"] == 1
    assert (nested / "inner.txt").read_text() == "git cannot hold this as a file\n"
    assert (candidate / "untracked.txt").read_text() == "loose\n"
    assert body_candidate.find("gc-incomplete")["path"] == bound["path"]
    assert git(serving, "for-each-ref", body_candidate.PIN_PREFIX) == ""
    events = (data / "logs" / "events.jsonl").read_text()
    assert "body_candidate_retained" in events and "capture_incomplete" in events


def test_an_unreadable_candidate_record_is_unknown_never_absent(body):
    """Each candidate is one record file: a soft listing skips an unreadable one, a strict
    read refuses, and the commit it holds is unknown (``None``), never proven lost."""
    serving, data = body
    held = body_candidate.prepare(make_ctx(serving, data, "rec-broken"))
    other = body_candidate.prepare(make_ctx(serving, data, "rec-healthy"))
    tip = candidate_commit(pathlib.Path(held["path"]))
    assert body_candidate.holds_commit("rec-broken", tip, data) is True
    record = subagent_worktrees._record_file(held["path"], data)
    record.write_text("{", encoding="utf-8")

    assert body_candidate.find("rec-broken", data) is None
    with pytest.raises(subagent_worktrees.SubagentWorktreeRegistryCorrupt):
        body_candidate.find("rec-broken", data, strict=True)
    assert body_candidate.holds_commit("rec-broken", tip, data) is None
    assert body_candidate.holds_commit("no-such-task", tip, data) is False, "no record is evidence"
    assert body_candidate.find("rec-healthy", data)["path"] == other["path"]
    assert pathlib.Path(held["path"]).is_dir() and record.read_text(encoding="utf-8") == "{"


def test_gc_counts_from_last_use_and_a_missing_checkout_keeps_its_commits(body):
    serving, data = body
    used = body_candidate.prepare(make_ctx(serving, data, "gc-used"))
    lost = body_candidate.prepare(make_ctx(serving, data, "gc-lost"))
    tip = candidate_commit(pathlib.Path(lost["path"]))
    shutil.rmtree(lost["path"])
    now = __import__("time").time()
    for row in subagent_worktrees._records(data):  # one record file per candidate
        subagent_worktrees._put_record(
            {**row, "created_at": 1.0, "used_at": now if row["task_id"] == "gc-used" else 1.0}, data)

    report = subagent_worktrees.prune_orphans(retention_days=7)

    assert report == {"removed": 1, "kept": 1}
    assert pathlib.Path(used["path"]).is_dir()
    assert git(serving, "rev-parse", lost["branch"]) == tip  # the branch outlives its checkout


# --------------------------------------------------------------------------- #
# Commit and review bind to the candidate; governance stays the serving body
# --------------------------------------------------------------------------- #
def test_review_subject_is_the_candidate_and_governance_is_the_serving_body(body):
    serving, data = body
    ctx = make_ctx(serving, data, "root-review")
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    (candidate / "docs").mkdir()
    (candidate / "docs/CHECKLISTS.md").write_text("# candidate rewrites the rubric\n")

    from ouroboros.review_substrate import review_repo_dirs_for

    governance, subject = review_repo_dirs_for(ctx)
    assert governance == serving.resolve() and subject == candidate.resolve()
    assert not (governance / "docs/CHECKLISTS.md").exists()  # the changed rule is subject, not authority
    # The review checklists come from the body that is RUNNING, never from a context's candidate.
    from ouroboros.tools import review_helpers

    assert pathlib.Path(review_helpers.REPO_ROOT).resolve() == pathlib.Path(__file__).resolve().parents[1]


def test_commit_seams_use_the_candidate_branch_and_never_the_serving_push(body, monkeypatch):
    serving, data = body
    ctx = make_ctx(serving, data, "root-commit")
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    from ouroboros.tools import git as git_tools

    pushes = []
    monkeypatch.setattr(git_tools, "_auto_push", lambda repo_dir: pushes.append(repo_dir) or " [pushed: x]")
    # Whole-tree staging of the single-writer candidate: no shared-checkout baseline is consulted.
    assert git_tools._task_attributed_commit_paths(ctx, None) == (None, None, "", None)
    came_from_detached, error = git_tools._prepare_review_commit_worktree(ctx, None)
    assert (came_from_detached, error) == (False, "")
    assert git(candidate, "rev-parse", "--abbrev-ref", "HEAD") == ctx.branch_dev
    assert git(serving, "rev-parse", "--abbrev-ref", "HEAD") == "ouroboros"

    sha = candidate_commit(candidate)
    note = body_candidate.publication_note(ctx) or git_tools._auto_push(ctx.repo_dir)
    assert "not pushed" in note and ctx.branch_dev in note and pushes == []
    body_candidate.record_reviewed_commit(ctx, sha)
    row = body_candidate.find("root-commit")
    assert row["reviewed_commits"] == [sha] and body_candidate.unreviewed_commits(row, sha) == []
    raw = candidate_commit(candidate, "raw shell commit", files={"x.txt": "x\n"})
    assert body_candidate.unreviewed_commits(body_candidate.find("root-commit"), raw) == [raw]
    # An unbound context keeps today's serving push.
    plain = make_ctx(serving, data, "root-plain")
    assert body_candidate.publication_note(plain) == ""


@pytest.mark.parametrize("conversation", [True, False])
def test_only_a_baseline_of_the_body_attributes_its_candidate(body, tmp_path, conversation):
    """Opus, 8 Nov 2026: every root now baselines its admitted Git folder, a direct turn's too. That folder's
    baseline says nothing about the body: a candidate the turn authors keeps the explicit/whole-tree staging
    contract. A queued root's baseline observes the body (its serving checkout), so its candidate is recorded and
    its commits are attributed, as before."""
    from ouroboros.mutation_attribution import capture_root_baseline
    from ouroboros.tools import git as git_tools

    serving, data = body
    folder = tmp_path / "project"
    folder.mkdir()
    git(folder, "init", "-q")
    (folder / "README.md").write_text("an admitted project\n", encoding="utf-8")
    git(folder, "add", "-A")
    git(folder, "commit", "-qm", "base")
    task_id = "root-direct" if conversation else "root-queued"
    ctx = make_ctx(serving, data, task_id)
    evidence = capture_root_baseline(data, {"id": task_id}, system_repo=serving, workspace=folder,
                                     conversation=conversation)
    assert sorted(row["surface_type"] for row in evidence["baseline"]["surfaces"]) == (
        ["external_workspace"] if conversation else ["external_workspace", "system_repo"])
    receipt = body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    (candidate / "ouroboros" / "mod_a.py").write_text("GEN = 'GEN_CAND'\n", encoding="utf-8")
    staged = git_tools._task_attributed_commit_paths(ctx, None)
    if conversation:
        assert receipt["attribution"] == {"state": "no_baseline"}
        assert staged == (None, None, "", None), "the folder's baseline never attributes another root"
    else:
        assert receipt["attribution"]["state"] == "original"
        selected, _attribution, error, binding = staged
        assert (selected, error, binding) == (["ouroboros/mod_a.py"], "", (data, task_id))


def test_candidate_start_service_uses_isolated_environment(body):
    """An ordinary (non-executor) service launched inside a candidate must not inherit live data."""
    import json
    serving, data = body
    ctx = make_ctx(serving, data, "root-service")
    registry = _registry(serving, data, ctx)
    assert "body candidate" in registry.execute("prepare_self_change", {})
    candidate = pathlib.Path(body_candidate.descriptor(ctx)["path"])
    result_text = registry.execute("start_service", {
        "name": "candidate-env",
        "cwd": str(candidate),
        "cmd": [sys.executable, "-c", "import os; print(os.environ['OUROBOROS_DATA_DIR'], flush=True)"],
        "readiness": {"log_contains": str(candidate.with_name(candidate.name + ".env") / "data"), "timeout_sec": 2},
    })
    result, _ = json.JSONDecoder().raw_decode(result_text.lstrip())
    assert result["state"] in {"running", "exited"}
    log = data / "services" / "root-service" / "candidate-env.log"
    assert str(candidate.with_name(candidate.name + ".env") / "data") in log.read_text()
    registry.execute("stop_service", {"name": "candidate-env"})


# --------------------------------------------------------------------------- #
# Serving Git identity: a failed probe is never "no repository" (#4277)
# --------------------------------------------------------------------------- #
def test_serving_worktree_reads_gits_own_answers(tmp_path):
    """Git's own answers classify the checkout: a broken, empty or missing checkout is a
    refusal, never a plain install (#4277). An empty `.git` draws Git's plain miss, yet
    it is metadata where Git searched. The plain and bare/`.git`-admin answers are
    witnessed through both doors below; every candidate prepared above witnesses a root."""
    (tmp_path / "broken" / ".git").parent.mkdir()
    (tmp_path / "broken" / ".git").write_text("gitdir: definitely-not-a-gitdir\n")
    (tmp_path / "empty" / ".git").mkdir(parents=True)
    for checkout in ("broken", "empty"):
        with pytest.raises(body_candidate.CandidateRefused) as refused:
            body_candidate._serving_worktree(tmp_path / checkout)
        assert refused.value.code == "CANDIDATE_UNAVAILABLE" and "could not be confirmed" in refused.value.text

    with pytest.raises(body_candidate.CandidateRefused) as refused:
        body_candidate._serving_worktree(tmp_path / "does-not-exist")
    assert refused.value.code == "CANDIDATE_UNAVAILABLE" and "could not be probed" in refused.value.text


_NO_REPO = "fatal: not a git repository (or any of the parent directories): .git\n"


def _boundary_miss(mount) -> str:
    return (f"fatal: not a git repository (or any parent up to mount point {mount})\n"
            "Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).\n")


def _probe_answers(monkeypatch, stderr="", stdout="", returncode=128):
    outcome = subprocess.CompletedProcess(("git", "rev-parse", "--is-inside-work-tree"), returncode, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(subagent_worktrees, "_git", lambda *_a, **_k: outcome)


@pytest.mark.parametrize("stdout, outcome", [
    ("true\n\n", True), ("true\n", "could not be confirmed"), ("true", "could not be confirmed"),
    ("true\nsub/\n", "inside another repository's Git working tree"), ("false\n\n", "outside the Git working tree")])
def test_only_gits_whole_two_line_answer_confirms_the_root(tmp_path, monkeypatch, stdout, outcome):
    """A root needs Git's whole answer: the work-tree word and its prefix line, empty at the
    root. A reply cut before the prefix line is unconfirmed, never a root; a nested folder is
    explained without Git's raw prefix."""
    _probe_answers(monkeypatch, stdout=stdout, returncode=0)
    if outcome is True:
        assert body_candidate._serving_worktree(tmp_path) is True
        return
    with pytest.raises(body_candidate.CandidateRefused) as refused:
        body_candidate._serving_worktree(tmp_path)
    assert refused.value.code == "CANDIDATE_UNAVAILABLE" and outcome in refused.value.text
    assert "sub/" not in refused.value.text


@pytest.mark.parametrize("stderr, plain", [
    (_NO_REPO, True),
    (_boundary_miss("/Volumes"), True),
    ("fatal: not a git repository (or any of the parent directories)", False),
    (_NO_REPO + "fatal: the toolchain licence was not accepted\n", False),
    (_boundary_miss("/Volumes").split("\n", 1)[0], False),
])
def test_only_gits_whole_canonical_discovery_miss_is_a_plain_install(tmp_path, monkeypatch, stderr, plain):
    """Only Git's whole canonical miss (normal or filesystem boundary) means no repository;
    a truncated or extended diagnostic that merely starts like it is a refusal."""
    _probe_answers(monkeypatch, stderr)
    if plain:
        assert body_candidate._serving_worktree(tmp_path) is False
        return
    with pytest.raises(body_candidate.CandidateRefused) as refused:
        body_candidate._serving_worktree(tmp_path)
    assert refused.value.code == "CANDIDATE_UNAVAILABLE" and "could not be confirmed" in refused.value.text


@pytest.mark.parametrize("metadata, plain", [
    ("above", True), ("outer", True), ("outer, respelled", True), ("volume", False), ("between", False),
    ("install", False)])
def test_a_reported_filesystem_boundary_bounds_the_metadata_search(tmp_path, monkeypatch, metadata, plain):
    """A plain volume mounted below a checkout: Git searches the install up to the volume root,
    then names the first parent it did not search (the outer checkout). `.git` there or above
    belongs to the outer checkout, however Git spells that folder; `.git` at the volume root,
    between it and the install, or in the install counts."""
    above = tmp_path.resolve() / "top"
    outer = above / "checkout"
    volume = outer / "volume"
    install = volume / "sub" / "app"
    install.mkdir(parents=True)
    places = {"above": above, "outer": outer, "outer, respelled": outer, "volume": volume,
              "between": volume / "sub", "install": install}
    (places[metadata] / ".git").mkdir()
    reported = outer.with_name("CHECKOUT") if metadata == "outer, respelled" else outer
    if not reported.exists():
        pytest.skip("a case-sensitive volume has one spelling per folder")
    _probe_answers(monkeypatch, _boundary_miss(reported))
    if plain:
        assert body_candidate._serving_worktree(install) is False
        return
    with pytest.raises(body_candidate.CandidateRefused, match="could not be confirmed"):
        body_candidate._serving_worktree(install)


def test_git_tracing_neither_hides_a_plain_install_nor_stops_for_other_commands(tmp_path, monkeypatch):
    """Requested Git tracing (environment and config trace2) writes beside the canonical miss;
    the identity read alone switches it off, so a genuine plain install stays plain while
    every other Git command keeps the owner's trace."""
    plain = tmp_path / "wheel-install"
    plain.mkdir()
    trace = tmp_path / "git-trace.log"
    (tmp_path / "trace2.gitconfig").write_text("[trace2]\n\tnormalTarget = 2\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "trace2.gitconfig"))
    monkeypatch.setenv("GIT_TRACE", str(trace))
    monkeypatch.setenv("GIT_TRACE_SETUP", "2")
    monkeypatch.setenv("GIT_TRACE2_PERF", "2")
    # The fixture really traces: a plain read in the same locale is not the bare canonical miss.
    traced = subagent_worktrees._git(plain, "rev-parse", "--is-inside-work-tree", check=False,
                                     env=subagent_worktrees._host_env(LC_ALL="C"))
    assert traced.returncode == 128 and not body_candidate._PROBE_NO_REPO.fullmatch(traced.stderr)
    trace.unlink()

    assert body_candidate._serving_worktree(plain) is False
    assert not trace.exists()  # off for the identity read only
    other = subagent_worktrees._git(plain, "version")
    assert other.returncode == 0 and other.stderr.strip() and trace.read_text().strip()


def _spy_write_file(registry) -> list:
    """A no-effect stand-in for write_file: records any call that reaches the handler."""
    invoked: list = []
    registry.override_handler("write_file", lambda *_a, **args: invoked.append(args) or "spy: nothing written")
    return invoked


def test_plain_install_keeps_the_in_place_contract_and_refuses_explicit_prepare(tmp_path, monkeypatch):
    """True non-repo compatibility: the automatic seam still edits a plain installation
    in place, while explicit prepare keeps its typed refusal."""
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    plain = tmp_path / "wheel-install"
    (plain / "ouroboros").mkdir(parents=True)
    ctx = make_ctx(plain, data, "root-wheel")
    registry = _registry(plain, data, ctx)

    written = registry.execute_result("write_file", {"path": "ouroboros/mod_a.py", "content": "GEN = 'IN_PLACE'\n"})
    assert written.status == "ok", written
    assert (plain / "ouroboros" / "mod_a.py").read_text() == "GEN = 'IN_PLACE'\n"
    assert not body_candidate.is_bound(ctx)

    explicit = registry.execute_result("prepare_self_change", {})
    assert (explicit.status, explicit.code) == ("blocked", "CANDIDATE_UNAVAILABLE"), explicit
    assert "is not a Git working tree" in explicit.text
    assert not body_candidate.is_bound(ctx)


@pytest.mark.parametrize("channel", ["stderr", "stdout"])
def test_an_unreadable_serving_identity_refuses_with_gits_own_words(body, monkeypatch, channel):
    """A serving checkout Git cannot answer about (an unaccepted toolchain licence is
    #4277's class) refuses on both doors with Git's own diagnostic, also when it arrives on
    stdout alone, instead of the old in-place fall-through; the write handler is never reached."""
    serving, data = body
    before = _tree_bytes(serving)
    ctx = make_ctx(serving, data, "root-prereq")
    _probe_answers(monkeypatch, **{channel: "fatal: the toolchain licence was not accepted"})
    registry = _registry(serving, data, ctx)
    invoked = _spy_write_file(registry)

    automatic = registry.execute_result("write_file", {"path": "ouroboros/mod_a.py", "content": "GEN = 'NEVER'\n"})
    assert (automatic.status, automatic.code) == ("blocked", "CANDIDATE_UNAVAILABLE"), automatic
    assert automatic.text.startswith("⚠️ CANDIDATE_UNAVAILABLE: "), automatic
    assert "the toolchain licence was not accepted" in automatic.text

    explicit = registry.execute_result("prepare_self_change", {})
    assert (explicit.status, explicit.code) == ("blocked", "CANDIDATE_UNAVAILABLE"), explicit
    assert "the toolchain licence was not accepted" in explicit.text

    assert invoked == [] and _tree_bytes(serving) == before and not body_candidate.is_bound(ctx)


@pytest.mark.parametrize("layout, words", [
    ("nested", "inside another repository's Git working tree"), ("admin", "outside the Git working tree")])
def test_git_that_is_not_the_bodys_own_working_tree_refuses_rather_than_provisions(body, layout, words):
    """A folder Git places inside another repository's working tree, or one Git answers is
    outside the working tree (here the `.git` admin directory; the refusal names no layout),
    refuses on both doors: it never provisions the enclosing project nor edits in place,
    and the write handler is never reached."""
    serving, data = body
    target = serving / ("ouroboros" if layout == "nested" else ".git")
    ctx = make_ctx(target, data, f"root-{layout}")
    registry = _registry(target, data, ctx)
    invoked = _spy_write_file(registry)
    before = _tree_bytes(target)

    automatic = registry.execute_result("write_file", {"path": "HEAD", "content": "NEVER\n"})
    assert (automatic.status, automatic.code) == ("blocked", "CANDIDATE_UNAVAILABLE"), automatic
    assert words in automatic.text

    explicit = registry.execute_result("prepare_self_change", {})
    assert (explicit.status, explicit.code) == ("blocked", "CANDIDATE_UNAVAILABLE"), explicit
    assert words in explicit.text

    assert invoked == [] and _tree_bytes(target) == before and not body_candidate.is_bound(ctx)


def test_the_identity_probe_takes_its_own_capture_environment(tmp_path, monkeypatch):
    """Git for Windows can reroute a child's whole streams per GIT_REDIRECT_STDOUT/GIT_REDIRECT_STDERR
    (git(1) on Windows: a path, `off`, and `2>&1` on stderr), which would read the canonical
    miss off the captured pipe. The identity read alone spawns in its normal capture environment,
    the redirect settings absent whatever the owner's shell carries, while every other Git
    command keeps the owner's redirection."""
    plain = tmp_path / "wheel-install"
    plain.mkdir()
    captured = {}

    def _spy(cwd, *args, **kwargs):
        captured.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(("git", "rev-parse"), 128, stdout="", stderr=_NO_REPO)

    monkeypatch.setattr(subagent_worktrees, "_git", _spy)
    monkeypatch.setenv("GIT_TRACE", "1")
    monkeypatch.setenv("GIT_REDIRECT_STDERR", "2>&1")
    monkeypatch.setenv("GIT_REDIRECT_STDOUT", r"C:\git-probe-out.log")
    assert body_candidate._serving_worktree(plain) is False
    assert "GIT_REDIRECT_STDERR" not in captured and "GIT_REDIRECT_STDOUT" not in captured
    assert captured["LC_ALL"] == "C" and captured["GIT_TRACE"] == "0"
    assert subagent_worktrees._host_env()["GIT_REDIRECT_STDERR"] == "2>&1"


@pytest.mark.skipif(sys.platform != "win32", reason="GIT_REDIRECT_STDOUT/STDERR are a Git for Windows feature")
def test_an_inherited_redirect_setting_cannot_reclassify_a_plain_install(tmp_path, monkeypatch):
    """Git for Windows honours the reroute variables, so with them inherited from the owner's
    shell a real plain install stays plain only because the identity read spawns without them."""
    plain = tmp_path / "wheel-install"
    plain.mkdir()
    monkeypatch.setenv("GIT_REDIRECT_STDERR", "2>&1")
    monkeypatch.setenv("GIT_REDIRECT_STDOUT", "off")
    assert body_candidate._serving_worktree(plain) is False


@pytest.mark.skipif(sys.platform != "darwin", reason="Git compares a ceiling with the kept spelling on macOS")
def test_a_case_respelled_address_honors_the_configured_ceiling_as_git_does(tmp_path, monkeypatch):
    """Git on macOS reads its cwd as the spelling the volume keeps (getcwd) and compares a
    GIT_CEILING_DIRECTORIES entry against that spelling literally, so the confirmation walks
    in Git's own frame: a case-respelled address does not extend the metadata search past a
    configured ceiling (the repository above it belongs to what the ceiling excludes, and the
    plain install stays plain), while an entry spelled in another case keeps not matching,
    exactly as Git treats it, and Git itself reports that outer repository's working tree."""
    root = tmp_path.resolve()
    keep = root / "alpha" / "app"
    install = keep / "sub"
    install.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
    respelled = keep.with_name("APP") / "SUB"
    entry = keep.with_name("APP")
    if not respelled.exists() or not entry.exists():
        pytest.skip("a case-sensitive volume has one spelling per folder")

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(keep))
    assert body_candidate._serving_worktree(respelled) is False  # .git above the ceiling is excluded

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(entry))
    with pytest.raises(body_candidate.CandidateRefused, match="inside another repository's Git working tree"):
        body_candidate._serving_worktree(install)


@pytest.mark.parametrize("stdout, stderr", [
    ("true\n", _NO_REPO), ("\n", _NO_REPO), ("", _NO_REPO.rstrip("\n")), ("", " " + _NO_REPO)])
def test_a_plain_miss_is_never_read_from_mixed_or_partial_output(tmp_path, monkeypatch, stdout, stderr):
    """Git's miss dies before rev-parse prints a word and ends with exactly one newline: an
    answer beside it on stdout, or a cut or padded miss, is unconfirmed, never a plain install."""
    _probe_answers(monkeypatch, stderr, stdout)
    with pytest.raises(body_candidate.CandidateRefused, match="could not be confirmed"):
        body_candidate._serving_worktree(tmp_path)


def test_the_metadata_search_reads_git_ceiling_directories_as_git_does(tmp_path, monkeypatch):
    """GIT_CEILING_DIRECTORIES as Git's setup.c reads it (canonicalize_ceiling_entry): a relative
    entry is ignored, an entry before an empty one is resolved, one after it is compared
    literally, and the searched directory itself is never excluded (strict ancestor)."""
    from ouroboros.workspace_admission import has_git_metadata

    real = tmp_path.resolve() / "real"
    install = real / "app"
    install.mkdir(parents=True)
    (tmp_path.resolve() / ".git").mkdir()  # metadata above `real`: only a ceiling at `real` hides it
    alias = tmp_path.resolve() / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("this volume or account cannot create a symlink")
    monkeypatch.chdir(tmp_path.resolve())
    for value, found in [
        ("", True),
        (str(real), False),
        ("real", True),  # relative: Git ignores it, whatever the current directory
        (str(alias), False),  # resolved before any empty entry
        (os.pathsep + str(alias), True),  # literal after an empty entry: the alias never matches
        (os.pathsep + real.as_posix(), False),  # literal and exact
        (os.pathsep + real.as_posix() + "/", False),  # one trailing slash is Git's own spelling too
        (os.pathsep + real.parent.as_posix() + "/./real", True),  # otherwise literal: never normalized
        (str(install), True),  # the searched directory itself is never a ceiling
        *([] if os.name == "nt" else [  # a missing middle folder: Git drops the entry, never `real`
            (str(real.parent / "missing" / ".." / "real"), True)]),
    ]:
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", value)
        assert has_git_metadata(install) is found, value


# --------------------------------------------------------------------------- #
# Contacts: the host names the manuals' addresses where the body is touched (BIBLE P1)
# --------------------------------------------------------------------------- #
ROAD = "- **Not a second GEN**, because one owner holds it."
MANUALS = {
    "ouroboros/domains.toml": ('[domains]\nD01 = "Core"\nD02 = "Guards"\n\n[references]\n'
                               'D01 = ["docs/architecture/01-core.md"]\n'
                               'D02 = ["docs/architecture/10-key-invariants.md", "docs/architecture/01-core.md"]\n'
                               '"web/" = ["docs/architecture/03-web.md", "docs/DESIGN.md", "docs/CLIENT.md"]\n'
                               '"skills/" = ["docs/architecture/13-skills.md"]\n"prompts/" = ["docs/development/04-gov.md"]\n'
                               '"build.sh" = ["docs/architecture/08-build.md"]\n\n'
                               '[modules]\n"ouroboros/mod_a.py" = "D01"\n"ouroboros/mod_b.py" = "D02"\n'),
    "docs/ARCHITECTURE.md": ("# Architecture\n\nThe map.\n\n## Chapters\n\n- [Core](architecture/01-core.md)\n"
                             "- [Key Invariants](architecture/10-key-invariants.md)\n"),
    "docs/architecture/01-core.md": "# 1. Core\n\nWhat the core gives me.\n",
    "docs/architecture/10-key-invariants.md": ("# 10. Key Invariants\n\nThe rules every part keeps.\n\n"
                                               f"### 10.2 Closed roads\n\nRoads already closed.\n\n{ROAD}\n"),
    "build.sh": "#!/bin/sh\n", "web/app.js": "// app\n", "web/tests/app.test.js": "// test\n",
    "skills/demo/SKILL.md": "# Demo skill\n", "skills/demo/run.py": "RUN = 1\n", "prompts/SYSTEM.md": "# I am\n",
}
LINE = "[Own body: owning manual `{}`{}; invariants and closed roads: Body Map §10]"
_READS = itertools.count()


def _with_manuals(repo):
    for relative, text in MANUALS.items():
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_bytes(text.encode("utf-8"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "manuals")


def _read_note(registry, ctx, path, **args):
    """A read through the loop's own tool path (``_execute_single_tool`` then ``process_tool_results``): the typed
    result, the host note trailing what the mind is shown, and the trace row."""
    from ouroboros.loop_tool_execution import _execute_single_tool, process_tool_results

    logs = pathlib.Path(ctx.drive_root) / "logs"
    logs.mkdir(exist_ok=True)
    call_id = f"read-{next(_READS)}"
    row = _execute_single_tool(registry, {"id": call_id, "function": {
        "name": "read_file", "arguments": json.dumps({"path": path, **args})}}, logs)
    typed = row["tool_result"]
    assert typed.status == "ok", typed.text
    messages, trace = [], {"tool_calls": [], "reasoning_notes": []}
    process_tool_results([row], messages, trace, lambda *_a: None, registry)
    (shown,) = [message["content"] for message in messages if message.get("role") == "tool"]
    assert shown == row["result"] or shown.startswith(row["result"] + "\n\n"), shown
    return typed, shown[len(row["result"]):].removeprefix("\n\n"), trace["tool_calls"][0]


def test_a_read_of_an_own_body_file_is_followed_by_its_manuals_outside_the_files_text(body, tmp_path):
    """The owner's "ОБЯЗАТЕЛЬНО при работе над телом" needs the address at hand: a read of a body source file ends,
    in what the mind is shown, with one short line naming its owning manual, how many more the manifest declares,
    and §10 of the Body Map. The file's text, its typed result and the loop's producer row carry none of it, and no
    second copy of the read is kept. Own-body identity is where the read resolved, not the file's name; the line
    repeats on every read (no state)."""
    serving, data = body
    _with_manuals(serving)
    ctx = make_ctx(serving, data, "root-reader")
    registry = _registry(serving, data, ctx)
    typed, note, traced = _read_note(registry, ctx, "ouroboros/mod_a.py", root="system_repo")
    assert note == LINE.format("docs/architecture/01-core.md", "")
    assert "Own body" not in typed.text and not typed.host_annotations and "GEN_OLD" in typed.text
    assert "producer_source_ref" not in traced and not list(data.rglob("*.producer-*"))  # no copy of the file
    assert _read_note(registry, ctx, "ouroboros/mod_a.py", root="system_repo")[1] == note  # the same line again
    # The invariants chapter as the owner is not named twice; a build script and a web file are the body too
    # (no filename allowlist); a web file names its first manual and counts the rest.
    assert _read_note(registry, ctx, "ouroboros/mod_b.py", root="system_repo")[1] == (
        "[Own body: owning manual `docs/architecture/10-key-invariants.md` (+1 in domains.toml [references])]")
    assert _read_note(registry, ctx, "build.sh", root="system_repo")[1] == LINE.format("docs/architecture/08-build.md", "")
    assert _read_note(registry, ctx, "web/app.js", root="system_repo")[1] == LINE.format(
        "docs/architecture/03-web.md", " (+2 in domains.toml [references])")
    # A skill's code and a governed prompt keep it; documentation and tests under the same rows stay quiet.
    assert _read_note(registry, ctx, "skills/demo/run.py", root="system_repo")[1] == LINE.format(
        "docs/architecture/13-skills.md", "")
    assert _read_note(registry, ctx, "prompts/SYSTEM.md", root="system_repo")[1] == LINE.format(
        "docs/development/04-gov.md", "")
    for quiet in ("skills/demo/SKILL.md", "web/tests/app.test.js", "docs/architecture/01-core.md", "server.py"):
        assert _read_note(registry, ctx, quiet, root="system_repo")[1] == "", quiet
    assert not list(data.rglob("*.producer-*"))
    # The same read through this task's bound candidate is still the body.
    body_candidate.prepare(ctx)
    assert _read_note(registry, ctx, "ouroboros/mod_a.py")[1] == LINE.format("docs/architecture/01-core.md", "")

    # A foreign project with the same file names is not the body: nothing is named.
    foreign = tmp_path / "foreign"
    for relative in ("ouroboros/mod_a.py", "build.sh", "ouroboros/domains.toml"):
        (foreign / relative).parent.mkdir(parents=True, exist_ok=True)
        (foreign / relative).write_bytes((serving / relative).read_bytes())
    git(foreign, "init", "-q", "-b", "main")
    git(foreign, "add", "-A")
    git(foreign, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "foreign")
    guest = make_ctx(serving, data, "root-guest", workspace_root=str(foreign), workspace_mode="external")
    guest_registry = _registry(serving, data, guest)
    for path in ("ouroboros/mod_a.py", "build.sh"):
        typed, note, _traced = _read_note(guest_registry, guest, path)
        assert note == "" and (foreign / path).read_text(encoding="utf-8") in typed.text


def test_a_copy_or_a_clone_of_the_body_is_read_as_the_body(body, tmp_path):
    """The two roots of ``own_body_root`` past the bound candidate and the serving checkout: a copy the host
    recorded as made from the body (``workspace_copy``) and a task's own folder that is a clone of it."""
    serving, data = body
    _with_manuals(serving)
    expected = LINE.format("docs/architecture/01-core.md", "")
    copy = tmp_path / "copy-of-body"
    shutil.copytree(serving, copy, ignore=shutil.ignore_patterns(".git"))
    git(copy, "init", "-q", "-b", "main")  # its own history: only the host's record makes it the body
    git(copy, "add", "-A")
    git(copy, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "copy")
    for recorded, line in ((True, expected), (False, "")):
        meta = {"root_task_id": "root-x", "delegation_role": "subagent",
                "workspace_copy": {"execution_root": str(copy), "source_is_system_repo": recorded}}
        child = make_ctx(serving, data, f"child-copy-{recorded}", workspace_root=str(copy),
                         workspace_mode="external", task_metadata=meta)
        assert _read_note(_registry(serving, data, child), child, "ouroboros/mod_a.py")[1] == line, recorded

    clone = tmp_path / "clone-of-body"
    git(tmp_path, "clone", "-q", str(serving), str(clone))
    root = make_ctx(serving, data, "root-clone", workspace_root=str(clone), workspace_mode="external")
    assert not body_candidate.is_bound(root)
    assert _read_note(_registry(serving, data, root), root, "ouroboros/mod_a.py")[1] == expected


def test_the_read_line_on_the_real_manifest_names_code_and_leaves_documentation_and_tests_alone(body):
    """The shipped ``ouroboros/domains.toml``: a script, a web implementation file, a module and a governed prompt
    keep the line with one manual address; a skill's SKILL.md, the pull-request template, a web test and a
    vendored manifest page under the same rows stay quiet."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest

    serving, data = body
    _with_manuals(serving)
    (serving / "ouroboros/domains.toml").write_bytes((REPO / "ouroboros/domains.toml").read_bytes())
    real = load_domain_manifest(REPO)
    code = ("scripts/check_domains.py", "web/modules/chat.js", "ouroboros/context.py", "prompts/SYSTEM.md")
    quiet = ("skills/telegram/SKILL.md", ".github/PULL_REQUEST_TEMPLATE.md", "web/tests/host_lifecycle_rows.test.js",
             "web/vendor/VENDOR-MANIFEST.md")
    for relative in code + quiet:
        (serving / relative).parent.mkdir(parents=True, exist_ok=True)
        (serving / relative).write_text("x\n", encoding="utf-8")
    ctx = make_ctx(serving, data, "root-real")
    registry = _registry(serving, data, ctx)
    for relative in code:
        manuals = real.manuals_for(relative)
        note = _read_note(registry, ctx, relative, root="system_repo")[1]
        assert manuals and note.startswith(f"[Own body: owning manual `{manuals[0]}`") and note.count("`") == 2, note
        assert ("(+" in note) is (len(manuals) > 1) and "Body Map §10" in note
    for relative in quiet:
        assert real.manuals_for(relative), relative  # a row covers it: the source role keeps it quiet
        assert _read_note(registry, ctx, relative, root="system_repo")[1] == "", relative


def test_entering_a_body_change_carries_the_closed_roads_once_and_only_for_the_own_body(body, monkeypatch, tmp_path):
    """One decision, the work's first entry into a change of its own body, carries the Closed roads section whole
    with the chapter's address: a candidate prepared fresh, a first explicit resume (by another work too), the
    first automatic bind and Cyber Pro's in-place choice. A later write, a second receipt of a bound task, a
    repeated resume of the candidate it holds, a second in-place choice and a subagent told where its parent put
    it carry none (an acting child in an unrelated workspace once got Ouroboros's closed roads)."""
    import re

    serving, data = body
    _with_manuals(serving)
    explicit = make_ctx(serving, data, "root-explicit")
    registry = _registry(serving, data, explicit)
    receipt = registry.execute("prepare_self_change", {})
    assert receipt.endswith(ROAD) and "### 10.2 Closed roads" in receipt
    assert "`docs/architecture/10-key-invariants.md`" in receipt and "before editing" not in receipt.lower()
    assert "Closed roads" not in registry.execute("prepare_self_change", {})  # already bound: said once
    candidate = re.search(r"body candidate (\S+) is", receipt).group(1)
    for _ in range(2):  # re-entering the candidate this work holds
        again = registry.execute("prepare_self_change", {"resume": candidate})
        assert f"body candidate {candidate} is resumed" in again and "Closed roads" not in again
    assert "Closed roads" not in registry.execute("write_file", {"path": "ouroboros/mod_a.py", "content": "GEN = 'E'\n"})

    _terminal(data, "root-explicit")  # another work takes the ended work's candidate over: its first entry
    taker = make_ctx(serving, data, "root-taker")
    registry = _registry(serving, data, taker)
    taken = registry.execute("prepare_self_change", {"resume": candidate})
    assert f"body candidate {candidate} is resumed" in taken and taken.endswith(ROAD)
    assert "Closed roads" not in registry.execute("prepare_self_change", {"resume": candidate})

    automatic = make_ctx(serving, data, "root-automatic")
    registry = _registry(serving, data, automatic)
    first = registry.execute("write_file", {"path": "ouroboros/mod_b.py", "content": "GEN = 'A'\n"})
    assert ROAD in first and "Closed roads" not in registry.execute(
        "write_file", {"path": "ouroboros/mod_b.py", "content": "GEN = 'B'\n"})

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    in_place = make_ctx(serving, data, "root-in-place")
    registry = _registry(serving, data, in_place)
    assert ROAD in registry.execute("prepare_self_change", {"in_place": True})
    assert "Closed roads" not in registry.execute("prepare_self_change", {"in_place": True})

    workspace = tmp_path / "their-project"
    workspace.mkdir()
    child = make_ctx(serving, data, "child-elsewhere", workspace_root=str(workspace), workspace_mode="external",
                     task_metadata={"root_task_id": "root-x", "delegation_role": "subagent"},
                     task_constraint={"mode": "acting_subagent", "surface": "workspace_in_place",
                                      "write_root": str(workspace), "base_sha": ""})
    from ouroboros.tools.control_runtime import _prepare_self_change

    told = _prepare_self_change(child)  # the handler's own answer, as the Cyber Pro child test reads it
    assert told.startswith("OK: no separate body candidate is prepared for a subagent") and str(workspace) in told
    assert "Closed roads" not in told and ROAD not in told


def test_the_first_write_of_an_install_without_a_git_body_carries_the_closed_roads_once(tmp_path, monkeypatch):
    """A packaged (wheel or image) install writes its body in place; its first body write is the work's entry and
    carries the roads, its later writes none, and each write lands as before."""
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    install = tmp_path / "packaged-install"
    for relative, text in {**MANUALS, "ouroboros/mod_a.py": "START = 1\n"}.items():
        (install / relative).parent.mkdir(parents=True, exist_ok=True)
        (install / relative).write_bytes(text.encode("utf-8"))
    ctx = make_ctx(install, data, "root-packaged")
    registry = _registry(install, data, ctx)
    first = registry.execute_result("write_file", {"path": "ouroboros/mod_a.py", "content": "START = 2\n"})
    assert first.status == "ok" and first.text.endswith(ROAD) and "### 10.2 Closed roads" in first.text
    later = registry.execute_result("write_file", {"path": "ouroboros/mod_a.py", "content": "START = 3\n"})
    assert later.status == "ok" and "Closed roads" not in later.text
    assert (install / "ouroboros/mod_a.py").read_text(encoding="utf-8") == "START = 3\n"
    assert body_candidate.list_candidates() == [] and not body_candidate.is_bound(ctx)


def test_a_body_without_its_invariants_chapter_enters_a_change_without_a_roads_note(body):
    serving, data = body  # the miniature body holds no book: the receipt is the plain one, never a failure
    receipt = _registry(serving, data, make_ctx(serving, data, "root-bookless")).execute("prepare_self_change", {})
    assert receipt.startswith("OK: body candidate") and "Closed roads" not in receipt

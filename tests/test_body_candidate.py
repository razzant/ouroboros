"""Own-body candidates: real Git, real registry dispatch, no serving-tree effect (#1539).

Every scenario builds a miniature serving body under ``tmp_path`` and drives the
product owners themselves — the tool dispatcher, the worktree registry and its
GC, the child-copy admission, the commit and review seams.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from ouroboros import body_candidate, subagent_worktrees
from tests.body_candidate_support import candidate_commit, git, isolate, make_ctx, make_serving


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


@pytest.mark.serial  # real child process
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
        assert result.meta.get("operation_outcome") == "completed_no_effect", result
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


@pytest.mark.serial  # real child process
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


# --------------------------------------------------------------------------- #
# Retention: unique work outlives age; an incomplete capture keeps the original
# --------------------------------------------------------------------------- #
def _expire_all():
    return subagent_worktrees.prune_orphans(retention_days=0)


def test_gc_removes_only_candidates_without_unique_work(body):
    serving, data = body
    clean = body_candidate.prepare(make_ctx(serving, data, "gc-clean"))
    junk = body_candidate.prepare(make_ctx(serving, data, "gc-junk"))
    (pathlib.Path(junk["path"]) / "ouroboros/__pycache__").mkdir()
    (pathlib.Path(junk["path"]) / "ouroboros/__pycache__/mod_a.cpython-311.pyc").write_bytes(b"\0cache")
    env_root = pathlib.Path(junk["path"]).with_name(pathlib.Path(junk["path"]).name + ".env")
    (env_root / "data").mkdir(parents=True)

    kept_fresh = subagent_worktrees.prune_orphans(retention_days=30)
    assert kept_fresh["removed"] == 0 and pathlib.Path(clean["path"]).is_dir()

    # A clean checkout still belongs to its live owner; age cannot remove its working address.
    assert _expire_all() == {"removed": 0, "kept": 2}
    _terminal(data, "gc-clean")
    _terminal(data, "gc-junk")
    report = _expire_all()

    assert report["removed"] == 2
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


def test_gc_counts_from_last_use_and_a_missing_checkout_keeps_its_commits(body):
    serving, data = body
    used = body_candidate.prepare(make_ctx(serving, data, "gc-used"))
    lost = body_candidate.prepare(make_ctx(serving, data, "gc-lost"))
    tip = candidate_commit(pathlib.Path(lost["path"]))
    shutil.rmtree(lost["path"])
    registry_path = data / "state" / "subagent_worktrees.json"
    import json

    rows = json.loads(registry_path.read_text())
    for row in rows["worktrees"]:
        row["created_at"] = 1.0
        row["used_at"] = __import__("time").time() if row["task_id"] == "gc-used" else 1.0
    registry_path.write_text(json.dumps(rows))

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


@pytest.mark.serial  # real child process
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

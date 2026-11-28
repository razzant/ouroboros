"""DEVELOPMENT.md rides whole only as a task's own requirement (BIBLE P1; owner 1.11.2026 and 25.11.2026).

Every mode and focus carries the one resident Body Map; the handbook behind it is read on demand. No binding (a body
worktree, a proven-foreign tree, an unknown one), surface (a delegated child, an API/CLI/scheduled source), task type
(evolution, deep self-review, review) or ``context_requires_self_body_docs`` preloads it: only
``task_contract.context_requires_development is True`` does, in every mode. An explicit false and an absent flag are
both "no handbook" and stay distinct facts in the contract.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ouroboros import context
from ouroboros.context_input_selection import development_required
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.memory import Memory

KEY = "context_requires_development"
HANDBOOK_RULE = "Full engineering discipline."


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})


@pytest.fixture
def trees(tmp_path):
    """The installed body with both books, a worktree of it, a proven-foreign tree and an unknown folder."""
    system = tmp_path / "system"
    for path, text in {
            "prompts/SYSTEM.md": "I am Ouroboros.", "BIBLE.md": "The complete constitution.",
            "docs/ARCHITECTURE.md": "# Body\n\nThe whole body map.\n\n## Chapters\n\n- [Runtime](architecture/runtime.md)\n",
            "docs/architecture/runtime.md": "# Runtime\n\nHow I keep work alive.\n\n## Execution\n\nFull mechanism.\n",
            "docs/DEVELOPMENT.md": "# Engineering\n\nHow I change.\n\n## Chapters\n\n- [Change](development/change.md)\n",
            "docs/development/change.md": f"# Change\n\nHow changes stay coherent.\n\n## Practice\n\n{HANDBOOK_RULE}\n"}.items():
        (system / path).parent.mkdir(parents=True, exist_ok=True)
        (system / path).write_text(text, encoding="utf-8", newline="\n")
    _git(system, "init", "-q")
    _git(system, "add", "-A")
    _git(system, "commit", "-q", "-m", "body")
    _git(system, "worktree", "add", "-q", "--detach", str(tmp_path / "body-worktree"))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    _git(foreign, "init", "-q")
    _git(foreign, "remote", "add", "origin", "https://github.com/someone/elsewhere.git")
    drive = tmp_path / "data"
    memory = Memory(drive_root=drive, repo_dir=system)
    memory.ensure_files()
    env = SimpleNamespace(repo_dir=system, system_repo_dir=system, drive_root=drive,
                          repo_path=lambda p: system / p, drive_path=lambda p: drive / p)
    return env, memory, {"body": tmp_path / "body-worktree", "foreign": foreign, "unknown": tmp_path / "no-such-folder"}


def _core(env, memory, task):
    return context._capture_context_core(env, memory, {"id": "t", "text": "work", **task}, None, None)


def _handbook_free(core):
    assert core.development_required is False and core.development_md == "" and core.required_book_tokens() == 0
    assert core.body_map.startswith("## Body map") and "docs/development/change.md" in core.body_map  # the Map stays
    assert HANDBOOK_RULE not in core.body_map


@pytest.mark.parametrize("tree", ["body", "foreign", "unknown"])
def test_no_binding_preloads_the_handbook_at_the_real_capture(trees, tree):
    """A body worktree, a tree with someone else's remote and a tree whose body fact is unknown all carry the Map
    alone; the contract's own true brings the whole handbook on the same binding."""
    env, memory, paths = trees
    task = {"type": "task", "workspace_root": str(paths[tree]), "workspace_mode": "external"}
    _handbook_free(_core(env, memory, task))
    required = _core(env, memory, {**task, "task_contract": {KEY: True}})
    assert required.development_required is True and HANDBOOK_RULE in required.development_md
    assert required.body_map == _core(env, memory, task).body_map  # one Map whatever the requirement


@pytest.mark.parametrize("task", [
    {"type": "evolution"}, {"type": "deep_self_review"}, {"type": "review"}, {"type": "task"},
    {"type": "task", "_is_direct_chat": True}, {"type": "task", "project_id": "p"},
    {"type": "task", "delegation_role": "subagent"}, {"type": "task", "metadata": {"source": "scheduled_task"}},
    {"type": "task", "actor_id": "cli"},
    {"type": "task", "context_requires_self_body_docs": True},
    {"type": "task", "task_contract": {"context_requires_self_body_docs": True}},
    # A top-level value alone is no requirement: only the contract, which survives restore and Continue, carries it.
    {"type": "task", KEY: True},
], ids=lambda task: "-".join(f"{key}={value}" for key, value in task.items()))
def test_no_task_type_surface_or_subject_marker_preloads_the_handbook(trees, task):
    env, memory, _paths = trees
    _handbook_free(_core(env, memory, task))
    assert development_required(task) is False


def test_only_the_contracts_own_true_requires_it_and_false_and_absence_stay_distinct_facts():
    # start_task authors the key into the new task's own contract (``_promoted`` below pins that door).
    explicit_false = build_task_contract({"id": "no", "task_contract": {KEY: False}})
    absent = build_task_contract({"id": "silent"})
    explicit_true = build_task_contract({"id": "yes", "task_contract": {KEY: True}})
    assert explicit_false[KEY] is False and KEY not in absent and explicit_true[KEY] is True
    assert [development_required({"task_contract": contract}) for contract in (explicit_false, absent, explicit_true)] == [
        False, False, True]
    assert development_required({"task_contract": {KEY: "true"}}) is False  # only a boolean true is the requirement


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_the_real_start_fit_sends_the_whole_handbook_only_as_the_requirement_in_every_mode(trees, mode):
    from ouroboros.context_fit import build_context_fit_plan

    env, memory, paths = trees

    def route(*_a, **_kw):
        return ({"model": "fixture", "provider": "fixture"},
                SimpleNamespace(route_fp="fixture", status="asserted", stale=False, window_tokens=1_000_000))

    for contract, whole in (({KEY: True}, True), ({KEY: False}, False), ({}, False)):
        task = {"id": "t", "text": "work", "type": "evolution", "context_requires_self_body_docs": True,
                "workspace_root": str(paths["body"]), "workspace_mode": "external", "task_contract": contract}
        core = context._capture_context_core(env, memory, task, None, None)
        plan = build_context_fit_plan(env, core, task, preferred_mode=mode, route_resolver=route)
        assert plan.initial_mode == mode
        content = plan.projection(mode).system_message()["content"]
        handbook = [block for block in content if block["text"].startswith("## DEVELOPMENT.md\n\n")]
        assert len(handbook) == int(whole), contract
        assert "## Body map" in content[0]["text"] and HANDBOOK_RULE not in content[0]["text"]
        if whole:
            assert content[1] == handbook[0] and "cache_control" in handbook[0] and HANDBOOK_RULE in handbook[0]["text"]


@pytest.mark.parametrize("flag", [True, False, None])
def test_start_task_carries_an_explicit_answer_to_the_admitted_task_and_absence_stays_absent(tmp_path, monkeypatch, flag):
    from tests.test_root_effort_ingress import _promoted

    task, _row, _workers = _promoted(tmp_path, monkeypatch, **({} if flag is None else {"context_requires_development": flag}))
    assert task.get("context_requires_development", "absent") == ("absent" if flag is None else flag)


# --- What the caller reads ------------------------------------------------------------------------------------------

def test_the_public_schema_says_what_true_costs_and_a_subject_caller_omits_it():
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom
    from ouroboros.tools.control import get_tools

    schema = next(tool for tool in get_tools() if tool.name == "start_task").schema
    description = schema["parameters"]["properties"][KEY]["description"]
    assert "every mode" in description and "body map" in description and "chapter map" not in description
    # start_task reaches the custom codec without parameter descriptions: the lead it sends carries the conditions, once.
    wire = project_function_tools_to_openai_custom([{"type": "function", "function": schema}]).wire_tools()[0]
    lead = (wire.get("custom") or wire)["description"].split("\n")[0]
    clause = ("Set context_requires_development=true only when the task itself needs the whole development handbook in "
              "every mode, never for its subject; a route too small for it refuses the task.")
    assert clause in lead and lead.count("context_requires_development") == 1
    assert "default" not in schema["parameters"]["properties"][KEY] and KEY not in schema["parameters"]["required"]
    self_body = schema["parameters"]["properties"]["context_requires_self_body_docs"]["description"]
    assert "It preloads no manual" in self_body and "full development handbook" not in self_body
    assert "the design bible of docs/DESIGN.md from any folder" in self_body  # item 97, RV1 (Carbon 1.2)


def test_no_public_door_gained_the_requirement():
    from ouroboros.gateway import tasks as api
    from ouroboros.tools import control_subagent_spec

    assert "task_contract" in api._RESERVED_METADATA_KEYS
    assert KEY not in control_subagent_spec.schedule_subagent_properties()

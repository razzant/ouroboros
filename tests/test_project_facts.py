"""Phase 3b: thin per-project facts store — id resolution, knowledge redirect,
selective context load, and the dual-run leak guard."""
from __future__ import annotations

import pathlib
import types

import ouroboros.config as cfg
from ouroboros.project_facts import (
    project_knowledge_dir,
    resolve_project_id,
    sanitize_project_id,
)


class _Ctx:
    def __init__(self, drive_root, project_id=""):
        self.drive_root = pathlib.Path(drive_root)
        self.project_id = project_id
        self.task_id = "t1"

    def drive_path(self, rel):
        return self.drive_root / rel


# --- id resolution (S7) -------------------------------------------------------

def test_resolve_project_id_explicit_wins():
    assert resolve_project_id({"project_id": "My Proj!"}) == sanitize_project_id("My Proj!")


def test_resolve_project_id_hashes_workspace():
    a = resolve_project_id({"workspace_root": "/tmp/x"})
    b = resolve_project_id({"workspace_root": "/tmp/x"})
    assert a.startswith("proj_") and a == b  # stable hash
    assert resolve_project_id({"workspace_root": "/tmp/y"}) != a


def test_resolve_project_id_empty_for_non_workspace():
    # Non-workspace tasks stay on canonical memory (unchanged behavior).
    assert resolve_project_id({}) == ""
    assert resolve_project_id({"type": "task", "memory_mode": "forked"}) == ""


def test_sanitize_project_id_is_path_safe():
    assert "/" not in sanitize_project_id("a/b/../c")
    assert sanitize_project_id("..") == ""


def test_project_id_case_canonicalized():
    from ouroboros.project_facts import explicit_project_id_ok

    assert sanitize_project_id("Proj_X") == "proj_x"      # canonical lowercase
    assert explicit_project_id_ok("Proj_X") is False       # mixed-case explicit rejected
    assert explicit_project_id_ok("proj_x") is True


def test_reserved_device_names_rejected():
    from ouroboros.project_facts import explicit_project_id_ok

    for name in ("con", "CON.md", "nul", "lpt1", "com3"):
        assert sanitize_project_id(name) == ""
        assert explicit_project_id_ok(name) is False


def test_subagent_inherits_explicit_scope_no_workspace_hash():
    # Subagents never re-derive a scope from their (acting) workspace; they inherit.
    assert resolve_project_id({"delegation_role": "subagent", "workspace_root": "/x"}) == ""
    assert resolve_project_id({"delegation_role": "subagent", "project_id": "proj_p"}) == "proj_p"
    # Root workspace tasks still hash.
    assert resolve_project_id({"workspace_root": "/x"}).startswith("proj_")


def test_generic_data_tools_deny_project_store():
    from ouroboros.project_facts import project_store_access_block

    assert project_store_access_block("projects/proj_x/knowledge/index-full.md")
    assert project_store_access_block("projects")
    assert project_store_access_block("/projects/proj_y/k.md")
    # traversal / ./ prefixed forms must not bypass the check
    assert project_store_access_block("./projects/proj_y/k.md")
    assert project_store_access_block("projects/../projects/proj_z/k.md")
    assert project_store_access_block("C:/projects/proj_w/k.md")  # Windows drive-letter form
    assert project_store_access_block("Projects/proj_x/k.md")  # case-insensitive
    assert project_store_access_block("memory/knowledge/x.md") is None
    assert project_store_access_block("a/projects/b") is None  # not the top-level store


def test_generic_tools_deny_project_store_live(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    from ouroboros.tools import core, core_file_tools
    from ouroboros.tools.registry import ToolContext

    (cfg.DATA_DIR / "projects" / "proj_x" / "knowledge").mkdir(parents=True)
    (cfg.DATA_DIR / "projects" / "proj_x" / "knowledge" / "secret.md").write_text("PROJECTSECRET", encoding="utf-8")
    (cfg.DATA_DIR / "memory").mkdir(parents=True)
    (cfg.DATA_DIR / "memory" / "g.md").write_text("GLOBALMARKER", encoding="utf-8")
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=cfg.DATA_DIR)

    # read/list/write deny the project store (incl. ./ and traversal forms)
    assert "ACCESS_DENIED" in core_file_tools._data_read(ctx, "projects/proj_x/knowledge/secret.md")
    assert "ACCESS_DENIED" in core_file_tools._data_read(ctx, "./projects/proj_x/knowledge/secret.md")
    assert "ACCESS_DENIED" in core_file_tools._data_read(ctx, "memory/../projects/proj_x/knowledge/secret.md")
    assert "ACCESS_DENIED" in core_file_tools._data_list(ctx, "projects")
    assert "ACCESS_DENIED" in core._data_write(ctx, "projects/proj_x/k.md", "data")


def test_explicit_project_id_strict_validation():
    from ouroboros.project_facts import explicit_project_id_ok

    assert explicit_project_id_ok("clean-id_1.2") is True
    assert explicit_project_id_ok("foo/bar") is False   # slash would be normalized
    assert explicit_project_id_ok("a b") is False       # space would be normalized
    assert explicit_project_id_ok("..") is False        # empty after sanitize
    assert explicit_project_id_ok("") is False
    assert explicit_project_id_ok(" proj_a") is False    # leading whitespace
    assert explicit_project_id_ok("proj_a\n") is False   # trailing whitespace


# --- knowledge redirect -------------------------------------------------------

def test_knowledge_write_project_scoped_goes_to_project_store(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    from ouroboros.tools import knowledge

    ctx = _Ctx(tmp_path / "child", project_id="proj_abc")
    knowledge._knowledge_write(ctx, "facts", "project fact X", "overwrite")

    proj = project_knowledge_dir("proj_abc") / "facts.md"
    assert proj.exists() and "project fact X" in proj.read_text(encoding="utf-8")
    # Must NOT land in the (forked, discarded) child drive's memory tree.
    assert not (ctx.drive_root / "memory" / "knowledge" / "facts.md").exists()


def test_knowledge_write_canonical_when_no_project(tmp_path):
    from ouroboros.tools import knowledge

    ctx = _Ctx(tmp_path / "drive", project_id="")
    knowledge._knowledge_write(ctx, "facts", "global fact", "overwrite")
    assert (ctx.drive_root / "memory" / "knowledge" / "facts.md").exists()


# --- selective context load ---------------------------------------------------

def test_context_project_scope_replaces_global_knowledge_index(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    from ouroboros import context

    drive = tmp_path / "drive"
    (drive / "memory" / "knowledge").mkdir(parents=True)
    (drive / "memory" / "knowledge" / "index-full.md").write_text("# idx\n- **g**: GLOBAL stuff", encoding="utf-8")
    px = project_knowledge_dir("proj_x")
    px.mkdir(parents=True)
    (px / "index-full.md").write_text("# idx\n- **facts**: project x stuff", encoding="utf-8")
    py = project_knowledge_dir("proj_y")
    py.mkdir(parents=True)
    (py / "index-full.md").write_text("# idx\n- **facts**: project y secret", encoding="utf-8")

    env = types.SimpleNamespace(
        drive_path=lambda rel: drive / rel,
        repo_path=lambda r: tmp_path / r,
        drive_root=drive,
    )
    # project-scoped (explicit param — Env is frozen and is NOT mutated)
    blob = "\n".join(context.build_knowledge_sections(env, project_id="proj_x"))
    assert "project x stuff" in blob
    assert "project y secret" not in blob  # never another project's facts
    assert "GLOBAL stuff" not in blob      # project scope replaces the global index
    # unscoped -> global index
    blob2 = "\n".join(context.build_knowledge_sections(env))
    assert "GLOBAL stuff" in blob2
    assert "Project knowledge" not in blob2


def test_apply_memory_actions_project_scoped_redirects_and_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    from ouroboros.reflection import apply_memory_actions

    drive = tmp_path / "drive"
    env = types.SimpleNamespace(drive_root=drive, repo_dir=tmp_path / "repo")
    applied = apply_memory_actions(env, [
        {"type": "knowledge_write", "topic": "pfacts", "content": "project knowledge X"},
        {"type": "scratchpad_append", "content": "should be skipped"},
        {"type": "identity_update_candidate", "content": "should be skipped"},
    ], project_id="proj_z")
    assert applied == 1  # only knowledge_write applied for a project task
    proj = project_knowledge_dir("proj_z") / "pfacts.md"
    assert proj.exists() and "project knowledge X" in proj.read_text(encoding="utf-8")
    # never canonical: no global knowledge, no scratchpad
    assert not (drive / "memory" / "knowledge" / "pfacts.md").exists()
    assert not (drive / "memory" / "scratchpad_blocks.json").exists()


def test_forked_project_child_skips_global_knowledge_seed(tmp_path):
    from ouroboros.headless import prepare_task_drive

    parent = tmp_path / "data"
    (parent / "memory" / "knowledge").mkdir(parents=True)
    (parent / "memory" / "knowledge" / "g.md").write_text("global recipe", encoding="utf-8")
    (parent / "memory" / "identity.md").write_text("I am Ouroboros", encoding="utf-8")

    # project-scoped forked child: no global knowledge seed, but identity carries (P1)
    child = prepare_task_drive(parent, "projtask1", "forked", project_id="proj_q")
    assert child is not None
    assert (child / "memory" / "identity.md").exists()
    assert not (child / "memory" / "knowledge").exists()  # not seeded from the global tree

    # non-project forked child: global knowledge seeded (unchanged behavior)
    child2 = prepare_task_drive(parent, "plaintask1", "forked")
    assert (child2 / "memory" / "knowledge" / "g.md").exists()


def test_d5_project_scoped_shared_preserves_mode_but_isolates_drive(tmp_path, monkeypatch):
    """D5 (Option A) gateway contract: a project-scoped `shared` task keeps its RECORDED
    memory_mode 'shared', yet the drive is MATERIALIZED isolated (effective mode 'forked')
    so project facts can never leak into global/shared memory. Guards against a future
    'simplification' back to passing memory_mode straight through to prepare_task_drive."""
    import asyncio
    import json
    from types import SimpleNamespace

    from ouroboros.gateway import tasks
    from supervisor import queue
    from supervisor import workers

    async def fake_request_json_or(_request, _default):
        return {"description": "x", "project_id": "proj_x", "memory_mode": "shared"}

    seen_modes: list[str] = []
    captured: dict = {}

    def fake_prepare(drive_root, task_id, mode, project_id=None):
        seen_modes.append(mode)
        child = tmp_path / "child" / task_id
        child.mkdir(parents=True, exist_ok=True)
        return child

    (tmp_path / "data").mkdir()
    (tmp_path / "repo").mkdir()
    monkeypatch.setattr(tasks, "request_json_or", fake_request_json_or)
    monkeypatch.setattr(tasks, "request_drive_root", lambda _r: tmp_path / "data")
    monkeypatch.setattr(tasks, "request_repo_dir", lambda _r: tmp_path / "repo")
    monkeypatch.setattr(tasks, "prepare_task_drive", fake_prepare)
    monkeypatch.setattr(
        queue,
        "enqueue_task",
        lambda task: captured.update(task) or task,
    )
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda *a, **k: True)
    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supervisor_ready_event=None)))
    response = asyncio.run(tasks.api_tasks_create(request))
    body = json.loads(response.body.decode("utf-8"))

    assert body.get("ok") is True, (response.status_code, body)
    assert captured["memory_mode"] == "shared"      # recorded mode preserved (informational)
    assert captured["project_id"] == "proj_x"
    assert seen_modes == ["forked"]                  # but materialized isolated
    assert captured.get("child_drive_root")          # isolated child drive set


def test_scheduled_subagent_task_inherits_project_id():
    # Subagent children inherit the parent's resolved project scope (read consistency).
    from supervisor.events import _build_scheduled_task_payload

    task = _build_scheduled_task_payload({"tid": "c1", "project_id": "proj_x", "delegation_role": "subagent"})
    assert task["project_id"] == "proj_x"
    assert resolve_project_id(task) == "proj_x"


def test_scratchpad_and_identity_tools_noop_for_project_tasks(tmp_path):
    import types

    from ouroboros.tools import control

    ctx = types.SimpleNamespace(drive_root=tmp_path, project_id="proj_p")
    r1 = control._update_scratchpad(ctx, "a meaningful scratchpad note for the task at hand")
    r2 = control._update_identity(ctx, "x" * 60)
    assert "project-scoped" in r1.lower()
    assert ("project-scoped" in r2.lower()) or ("global" in r2.lower())
    # nothing written to canonical memory
    assert not (tmp_path / "memory" / "scratchpad_blocks.json").exists()
    assert not (tmp_path / "memory" / "identity.md").exists()


def test_maybe_promote_skips_project_scoped_task(tmp_path, monkeypatch):
    import types

    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    import ouroboros.post_task_evolution as pte

    env = types.SimpleNamespace(drive_root=tmp_path)
    # explicit project_id -> project-scoped -> never promote global evolution
    assert pte.maybe_promote(env, {"type": "task", "id": "t1", "project_id": "proj_p"}, {"reflection": "x"}) is None


def test_apply_memory_actions_canonical_when_unscoped(tmp_path):
    from ouroboros.reflection import apply_memory_actions

    drive = tmp_path / "drive"
    env = types.SimpleNamespace(drive_root=drive, repo_dir=tmp_path / "repo")
    applied = apply_memory_actions(env, [
        {"type": "knowledge_write", "topic": "gfacts", "content": "global knowledge"},
    ])
    assert applied == 1
    assert (drive / "memory" / "knowledge" / "gfacts.md").exists()

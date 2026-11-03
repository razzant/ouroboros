"""Phase 3b: thin per-project facts store — id resolution, knowledge redirect,
selective context load, and the dual-run leak guard."""
from __future__ import annotations

import pathlib
import types

import pytest

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


def test_resolve_project_id_reads_the_durable_binding_first(tmp_path, monkeypatch):
    """B4=A: a "turn into project" conversion never reaches a running worker, so
    the durable binding outranks whatever the task dict says (or fails to say)."""
    from ouroboros.projects_registry import bind_task_to_project

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    bind_task_to_project(tmp_path, "t-bound", "bound-proj", 5150, origin={"absent": "system"})

    assert resolve_project_id({"id": "t-bound"}) == "bound-proj"
    assert resolve_project_id({"id": "t-bound", "project_id": "stale-proj"}) == "bound-proj"
    assert resolve_project_id({"task_id": "t-bound"}) == "bound-proj"
    # An unbound task keeps today's order untouched.
    assert resolve_project_id({"id": "t-free", "project_id": "explicit"}) == "explicit"
    assert resolve_project_id({"id": "t-free"}) == ""


def test_resolve_project_id_stays_fail_open_on_an_unreadable_bindings_store(tmp_path, monkeypatch):
    """The hot path must never stop the work: an unreadable store reads as "no
    binding", exactly like the workspace registry probe beside it."""
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "project_task_bindings.json").write_text("{ not json", encoding="utf-8")

    assert resolve_project_id({"id": "t-x", "project_id": "explicit"}) == "explicit"
    assert resolve_project_id({"id": "t-x"}) == ""


def test_minted_id_collapses_dash_runs_and_disambiguates_a_lossy_name():
    """I36: dropped characters each became a "-", so a Cyrillic word between two
    Latin ones minted `mlconf--------------------ouroboros`, and two DIFFERENT
    names normalized onto one id and silently shared a project."""
    from ouroboros.project_facts import explicit_project_id_ok, project_id_from_display_name

    first = project_id_from_display_name("ML Conf 2 доклад Ouroboros")
    second = project_id_from_display_name("ML Conf 2 презентация Ouroboros")

    assert "--" not in first and first.startswith("ml-conf-2-ouroboros-")
    assert first != second                       # the silent collision is over
    assert first == project_id_from_display_name("ML Conf 2 доклад Ouroboros")
    for minted in (first, second, project_id_from_display_name("emoji 🎉 name")):
        assert len(minted) <= 64
        assert explicit_project_id_ok(minted)     # round-trips through the normalizer
        assert minted == sanitize_project_id(minted)


def test_minted_id_leaves_clean_ascii_names_and_the_hash_fallback_alone():
    """New ids only: an ASCII name keeps the id it always had, a name with nothing
    usable keeps its proj_<hash> form, and sanitize_project_id is untouched."""
    from ouroboros.project_facts import project_id_from_display_name

    assert project_id_from_display_name("token-atlas") == "token-atlas"
    assert project_id_from_display_name("Cyber Racing") == "cyber-racing"
    assert project_id_from_display_name("airi research") == "airi-research"
    assert project_id_from_display_name("Динозавры").startswith("proj_")
    assert project_id_from_display_name("") == ""
    assert sanitize_project_id("ML Conf 2 доклад Ouroboros") == "ml-conf-2--------ouroboros"


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
    from ouroboros.tools import core
    from ouroboros.tools.registry import ToolContext

    (cfg.DATA_DIR / "projects" / "proj_x" / "knowledge").mkdir(parents=True)
    (cfg.DATA_DIR / "projects" / "proj_x" / "knowledge" / "secret.md").write_text("PROJECTSECRET", encoding="utf-8")
    (cfg.DATA_DIR / "memory").mkdir(parents=True)
    (cfg.DATA_DIR / "memory" / "g.md").write_text("GLOBALMARKER", encoding="utf-8")
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=cfg.DATA_DIR)

    # read/list/write deny the project store (incl. ./ and traversal forms)
    assert "ACCESS_DENIED" in core._data_read(ctx, "projects/proj_x/knowledge/secret.md")
    assert "ACCESS_DENIED" in core._data_read(ctx, "./projects/proj_x/knowledge/secret.md")
    assert "ACCESS_DENIED" in core._data_read(ctx, "memory/../projects/proj_x/knowledge/secret.md")
    assert "ACCESS_DENIED" in core._data_list(ctx, "projects")
    assert "ACCESS_DENIED" in core._data_write(ctx, "projects/proj_x/k.md", "data")


@pytest.mark.parametrize(("path", "hint", "wrong"), [
    ("projects/proj_x/workpad.md", "workpad_read(project_id='proj_x') reads it and "
     "workpad_write(project_id='proj_x', content=...) replaces it", "knowledge_read"),
    ("projects/proj_x/journal.jsonl", "journal_read(project_id='proj_x') reads it and "
     "journal_write(project_id='proj_x', kind=..., text=...) appends to it", "knowledge_read"),
    ("projects/proj_x/logs/task_reflections.jsonl",
     "journal_read(project_id='proj_x', source='reflections') reads it", "it writes it"),
    ("projects/proj_x/knowledge/notes/arch.md",
     "knowledge_read(topic='notes/arch', scope='project:proj_x') reads it and "
     "knowledge_write(topic='notes/arch', scope='project:proj_x', content=..., expected_revision=...) writes it; "
     "expected_revision is the revision knowledge_read returned", "root task"),
    ("projects/proj_x/knowledge/patterns.md", "Topic 'patterns' is global-only", "topic='patterns', scope="),
    ("projects/proj_x/knowledge/overview.md", "never this project copy; knowledge_list(scope='project:proj_x') may "
     "still list it", "topic='overview', scope="),
    ("projects/proj_x/knowledge/improvement-backlog.md", "Topic 'improvement-backlog' is global-only",
     "topic='improvement-backlog', scope="),
    ("projects/proj_x/knowledge/_index.md",
     "not addressable as a knowledge topic; knowledge_list(scope='project:proj_x') lists the shelf", "topic="),
    ("projects/proj_x/knowledge", "knowledge_list(scope='project:proj_x') lists it", "workpad_read"),
    ("projects/proj_x/knowledge/index-full.md", "knowledge_list(scope='project:proj_x') lists it", "topic="),
    ("projects/proj_x/.project.json", "No scoped reader or writer addresses this path", "reads it"),
    ("projects/proj_x/notes/plan.md", "No scoped reader or writer addresses this path", "reads it"),
    ("projects/Not An Id/workpad.md", "No scoped reader or writer addresses this path", "reads it"),
])
def test_project_store_refusal_names_only_tools_that_address_the_path(path, hint, wrong):
    """#1612: the refusal stays, but its next step is the read/write pair whose address
    space holds this path — never the knowledge shelf for a workpad, journal or other file."""
    from ouroboros.project_facts import project_store_access_block

    message = project_store_access_block(path)
    assert message.startswith("⚠️ ACCESS_DENIED: the per-project facts store (projects/<id>/)")
    assert hint in message and wrong not in message


def test_project_store_refusal_hint_reaches_the_target_only_within_existing_authority(tmp_path, monkeypatch):
    """#1612: following the named reader reaches the denied workpad for a root task, and
    a restricted child gets the refusal the hint announces; no access is widened."""
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    from ouroboros.tools import core
    from ouroboros.tools.project_journal import _workpad_read
    from ouroboros.tools.registry import ToolContext

    store = cfg.DATA_DIR / "projects" / "proj_x"
    store.mkdir(parents=True)
    (store / "workpad.md").write_text("WORKPADMARKER", encoding="utf-8")
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=cfg.DATA_DIR)

    refusal = core._data_read(ctx, "projects/proj_x/workpad.md")
    assert "WORKPADMARKER" not in refusal
    assert "knowledge_read" not in refusal
    assert "workpad_read(project_id='proj_x') reads it" in refusal
    assert "Outside its own project a task reads it only as a root task" in refusal
    root = types.SimpleNamespace(task_id="root", drive_root=cfg.DATA_DIR, project_id="mine",
                                 task_metadata={"root_task_id": "root"})
    assert "WORKPADMARKER" in _workpad_read(root, "proj_x")
    child = types.SimpleNamespace(task_id="child", drive_root=cfg.DATA_DIR, project_id="",
                                  task_metadata={"delegation_role": "subagent"})
    assert _workpad_read(child, "proj_x").startswith("⚠️ TOOL_FORBIDDEN")


def test_project_knowledge_hints_match_what_the_knowledge_tools_address(tmp_path, monkeypatch):
    """#1612: replacing a named project note needs the revision a read returns, and a
    reserved topic named with a project scope reaches the global note, not this copy."""
    import json

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    from ouroboros.project_facts import project_store_access_block
    from ouroboros.tools.knowledge import _knowledge_read, _knowledge_write
    from ouroboros.tools.registry import ToolContext

    shelf = cfg.DATA_DIR / "projects" / "proj_x" / "knowledge"
    shelf.mkdir(parents=True)
    (shelf / "arch.md").write_text("PROJECT NOTE\n", encoding="utf-8")
    (shelf / "patterns.md").write_text("PROJECT COPY\n", encoding="utf-8")
    (cfg.DATA_DIR / "memory" / "knowledge").mkdir(parents=True)
    (cfg.DATA_DIR / "memory" / "knowledge" / "patterns.md").write_text("GLOBAL PATTERNS\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=cfg.DATA_DIR)

    assert "expected_revision=" in project_store_access_block("projects/proj_x/knowledge/arch.md")
    assert "revision_required" in _knowledge_write(ctx, "arch", "NEW\n", scope="project:proj_x")
    read = _knowledge_read(ctx, "arch", scope="project:proj_x")
    revision = json.loads(read.split("\n", 1)[0].removeprefix("[Knowledge source] "))["revision"]
    assert _knowledge_write(ctx, "arch", "NEW\n", scope="project:proj_x", expected_revision=revision).startswith("✅")
    assert (shelf / "arch.md").read_text(encoding="utf-8") == "NEW\n"

    assert "global-only" in project_store_access_block("projects/proj_x/knowledge/patterns.md")
    reserved = _knowledge_read(ctx, "patterns", scope="project:proj_x")
    assert "GLOBAL PATTERNS" in reserved and "PROJECT COPY" not in reserved


def test_project_store_write_refusal_names_the_writer_and_its_current_project_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    from ouroboros.tools import core
    from ouroboros.tools.project_journal import _workpad_read, _workpad_write
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=cfg.DATA_DIR)
    refusal = core._data_write(ctx, "projects/proj_x/workpad.md", "OVERWRITE")
    assert "ACCESS_DENIED" in refusal and "knowledge_write" not in refusal
    assert "workpad_write(project_id='proj_x', content=...) replaces it" in refusal
    assert "it writes it only as a root task placed in no project (such as Main)" in refusal
    assert not (cfg.DATA_DIR / "projects" / "proj_x" / "workpad.md").exists()
    # The named writer behaves as announced: its own project and an unplaced root write it;
    # a root placed in another project and a restricted child never do.
    own = types.SimpleNamespace(task_id="t", drive_root=cfg.DATA_DIR, project_id="proj_x", task_metadata={})
    assert _workpad_write(own, "PLAN").startswith("OK:") and _workpad_read(own) == "PLAN"
    main = types.SimpleNamespace(task_id="main", drive_root=cfg.DATA_DIR, project_id="",
                                 task_metadata={"root_task_id": "main"})
    assert _workpad_write(main, "MAIN PLAN", "proj_x").startswith("OK:") and _workpad_read(own) == "MAIN PLAN"
    placed = types.SimpleNamespace(task_id="root", drive_root=cfg.DATA_DIR, project_id="mine",
                                   task_metadata={"root_task_id": "root"})
    child = types.SimpleNamespace(task_id="child", drive_root=cfg.DATA_DIR, project_id="",
                                  task_metadata={"delegation_role": "subagent"})
    assert _workpad_write(placed, "FOREIGN", "proj_x").startswith("⚠️ TOOL_FORBIDDEN")
    assert _workpad_write(child, "FOREIGN", "proj_x").startswith("⚠️ TOOL_FORBIDDEN")
    assert _workpad_read(own) == "MAIN PLAN"


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
    ctx.budget_drive_root = str(tmp_path / "data")  # a split child carries its canonical root (#915)
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

def test_context_project_scope_adds_to_global_knowledge_index(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    from ouroboros import context

    drive = tmp_path / "drive"
    (drive / "memory" / "knowledge").mkdir(parents=True)
    (drive / "memory" / "knowledge" / "index-full.md").write_text("# idx\n- **g**: GLOBAL stuff", encoding="utf-8")
    px = drive / "projects" / "proj_x" / "knowledge"  # the shelf on the context's own root, not DATA_DIR (#915)
    px.mkdir(parents=True)
    (px / "index-full.md").write_text("# idx\n- **facts**: project x stuff", encoding="utf-8")
    py = drive / "projects" / "proj_y" / "knowledge"
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
    assert "GLOBAL stuff" in blob         # project focus retains the shared corpus
    # unscoped -> global index
    blob2 = "\n".join(context.build_knowledge_sections(env))
    assert "GLOBAL stuff" in blob2
    assert "Project knowledge" not in blob2


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
    so the automatic writers can never leak project facts into global/shared memory. Guards against a future
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


def test_scratchpad_and_identity_tools_write_canonical_memory_from_a_project_room(tmp_path):
    import types

    from ouroboros.tools import control

    # One memory, every room: a project-scoped turn writes the same canonical
    # scratchpad and identity files the main chat writes.
    ctx = types.SimpleNamespace(drive_root=tmp_path, project_id="proj_p")
    r1 = control._update_scratchpad(ctx, "a meaningful scratchpad note for the task at hand")
    from hashlib import sha256
    r2 = control._update_identity(ctx, "x" * 60, sha256((tmp_path / "memory/identity.md").read_bytes()).hexdigest())
    assert r1.startswith("OK")
    assert r2.startswith("OK")
    assert (tmp_path / "memory" / "scratchpad_blocks.json").exists()
    assert (tmp_path / "memory" / "identity.md").read_text(encoding="utf-8") == "x" * 60


def test_maybe_promote_skips_project_scoped_task(tmp_path, monkeypatch):
    import types

    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    import ouroboros.post_task_evolution as pte

    env = types.SimpleNamespace(drive_root=tmp_path)
    # explicit project_id -> project-scoped -> never promote global evolution
    assert pte.maybe_promote(env, {"type": "task", "id": "t1", "project_id": "proj_p"}, {"reflection": "x"}) is None


# --- origin-keyed project lookup (one owner message, one Project) -------------

def _owner_ref(client_message_id: str = "msg-1", text: str = "ship it", chat_id: int = 1) -> dict:
    from ouroboros.project_dialogue import build_owner_message_ref

    return build_owner_message_ref(
        chat_id=chat_id, client_message_id=client_message_id,
        ts="2026-09-14T12:15:20+00:00", text=text,
    )


def test_project_id_for_origin_matches_by_value_and_ignores_ts_and_hash(tmp_path):
    """Identity of the owner's message is (chat_id, client_message_id) BY VALUE:
    the binding already validated ts/text_sha256 at write time, so a later reader
    must not re-derive identity from content (DEVELOPMENT.md anti-pattern)."""
    from ouroboros.projects_registry import (
        bind_task_to_project,
        origin_key,
        project_id_for_origin,
    )

    ref = _owner_ref()
    bind_task_to_project(tmp_path, "t-turn", "one-work", 1, origin={"ref": ref})

    assert project_id_for_origin(tmp_path, ref) == "one-work"
    # The same message seen through another producer's copy: same chat + client id,
    # a different capture timestamp and a rewritten text (the LLM-first normal path).
    drifted = dict(ref, ts="2026-09-14T12:37:49+00:00", text_sha256="0" * 64)
    assert project_id_for_origin(tmp_path, drifted) == "one-work"
    assert origin_key(drifted) == (1, "msg-1")
    # Another message, no project; an absent/typed-absence origin asks nothing.
    assert project_id_for_origin(tmp_path, _owner_ref("msg-2")) == ""
    assert project_id_for_origin(tmp_path, None) == ""
    assert project_id_for_origin(tmp_path, {}) == ""
    assert project_id_for_origin(tmp_path, {"chat_id": 1}) == ""
    assert origin_key({"chat_id": "nope", "client_message_id": "m"}) is None


def test_project_id_for_origin_resolves_a_legacy_double_binding_and_discloses_it(
    tmp_path, monkeypatch,
):
    """Legacy state (the measured incident) has ONE origin on TWO projects. The lookup
    never raises - a refusal would either block the owner or re-mint the duplicate it
    exists to end. It prefers the project whose task is still LIVE (that is where the
    work continued), else the LATEST binding, and says so durably."""
    import json

    from ouroboros.projects_registry import (
        bind_task_to_project,
        create_project,
        project_id_for_origin,
    )

    ref = _owner_ref()
    create_project(tmp_path, "first-room", name="First")
    create_project(tmp_path, "second-room", name="Second")
    bind_task_to_project(tmp_path, "t-turn", "first-room", 1, origin={"ref": ref})
    bind_task_to_project(tmp_path, "t-root", "second-room", 1, origin={"ref": ref})
    # bound_at has whole-microsecond resolution; pin the order explicitly.
    path = tmp_path / "state" / "project_task_bindings.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["bindings"]["t-turn"]["bound_at"] = "2026-09-14T12:38:22+00:00"
    data["bindings"]["t-root"]["bound_at"] = "2026-09-14T12:41:23+00:00"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert project_id_for_origin(tmp_path, ref, strict=True) == "second-room"
    rows = [json.loads(line) for line in
            (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[-1]["type"] == "project_origin_ambiguous"
    assert rows[-1]["chosen"] == "second-room"
    assert {row["project_id"] for row in rows[-1]["candidates"]} == {"first-room", "second-room"}
    assert rows[-1]["origin"] == {"chat_id": 1, "client_message_id": "msg-1"}

    # Legacy state is re-read on every promote, admission and convert click, and a
    # fact that has not changed is not news: the same choice discloses once.
    assert project_id_for_origin(tmp_path, ref, strict=True) == "second-room"
    assert len([json.loads(line) for line in
                (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]) == len(rows)

    # The earlier binding's task is still running: the work is there, so it wins.
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "RUNNING", {"t-turn": {"task": {"id": "t-turn"}}})
    monkeypatch.setattr(workers, "PENDING", [])
    assert project_id_for_origin(tmp_path, ref, strict=True) == "first-room"


def test_project_id_for_origin_ignores_a_project_that_can_no_longer_accept_bindings(tmp_path):
    """A deleting/tombstoned project would refuse the bind anyway, so adopting it
    would answer with a room the task can never join."""
    from ouroboros.projects_registry import (
        begin_project_deletion,
        bind_task_to_project,
        create_project,
        project_id_for_origin,
    )

    ref = _owner_ref()
    create_project(tmp_path, "gone-room", name="Gone")
    bind_task_to_project(tmp_path, "t-turn", "gone-room", 1, origin={"ref": ref})
    assert project_id_for_origin(tmp_path, ref, strict=True) == "gone-room"

    begin_project_deletion(tmp_path, "gone-room")
    assert project_id_for_origin(tmp_path, ref, strict=True) == ""


def test_project_id_for_origin_strict_raises_only_for_the_creating_callers(tmp_path):
    """Same split as project_id_for_task: the hot path reads "" from an unreadable
    store, while a caller that would CREATE a project must not do that blind."""
    import pytest

    from ouroboros.projects_registry import project_id_for_origin

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "project_task_bindings.json").write_text("{ not json", encoding="utf-8")

    assert project_id_for_origin(tmp_path, _owner_ref()) == ""
    with pytest.raises(ValueError):
        project_id_for_origin(tmp_path, _owner_ref(), strict=True)

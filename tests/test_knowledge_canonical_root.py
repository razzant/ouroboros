"""Knowledge has one root: ``canonical_data_root`` (task budget → context budget → task drive) (#915).

The knowledge tools, their previous-version sources, the next normal and delegated contexts and the
extension memory reader (Atlas, ``PluginAPI.read_memory``) resolve the same shelf. The configured
``DATA_DIR`` names no knowledge shelf of its own.

Production paths run their real producers: a child admitted through ``schedule_subagent`` and the
supervisor onto the worker's split runner, and a root task on its installation drive (no budget root;
that drive is canonical). Their ``elsewhere`` case is an unrelated configured ``DATA_DIR`` that nothing
may touch: in production it names the installation drive (the ``canonical`` case). The contract shapes
(a budget root only in task metadata or only on the context, or none at all) are synthetic
``ToolContext``s: no supported knowledge-tool producer builds the first two today.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import knowledge as store
from ouroboros.tool_access_paths import canonical_data_root
from ouroboros.tools import knowledge as tools
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._shared import configure_test_subagent
from tests._store_shared import seed_result
from tests.test_subagent_optional_role import _supervisor
from tests.test_subagent_role_texts import _env

pytestmark = pytest.mark.usefixtures("provider_catalog_offline")


def _note(summary: str) -> str:
    return f"---\ntype: note\nsummary: {summary}\n---\n# {summary}\nExact meaning, берег.\r\n"


def _receipt(reply: str) -> dict:
    assert reply.startswith("✅"), reply
    return json.loads(reply.split("\n", 1)[1])


def _read_previous(registry: ToolRegistry, ref: dict) -> str:
    """The advertised reader of a previous version (``read_file`` on ``artifact_store``), as the mind calls it."""
    from ouroboros.tools.core_file_tools import delivered_source_prefix

    reply = registry.execute(ref["read"]["tool"], ref["read"]["arguments"])
    source = delivered_source_prefix(registry._ctx.last_read_view, reply, len(reply))
    return json.loads(source)[ref["field"]]


def _read_previous_in_light(ctx: ToolContext, ref: dict) -> str:
    """The same handle through a Light memory operation's read tools (``consolidator.KnowledgeReadContext``)."""
    from ouroboros.consolidator import KnowledgeReadContext
    from ouroboros.tools.core_file_tools import delivered_source_prefix

    row = KnowledgeReadContext(ctx).read_call({"id": "previous", "function": {
        "name": ref["read"]["tool"], "arguments": json.dumps(ref["read"]["arguments"])}})
    assert row["status"] == "ok", row["result"]
    view = row["result_meta"]["read_file_source"]
    return json.loads(delivered_source_prefix(view, row["result"], len(row["result"])))[ref["field"]]


def _atlas_reads(root: Path, scope: str, topic: str) -> dict:
    """The extension memory reader over the host's data root (``PluginAPI._drive_root``)."""
    from ouroboros.extension_plugin_api import PluginAPIImpl, _PluginAPIConfig

    api = PluginAPIImpl(_PluginAPIConfig(skill_name="atlas", permissions=[], env_allowlist=[],
                                         state_dir=root / "state" / "skills" / "atlas",
                                         settings_reader=dict, drive_root=root))
    return api.read_memory("document", {"address": {"kind": "note", "scope": scope, "topic": topic}})


def _untouched(root: Path) -> bool:
    """No memory or project store, where knowledge shelves and their history live, under ROOT."""
    return not (root / "memory").exists() and not (root / "projects").exists()


def _real_agent(monkeypatch, repo: Path, drive: Path, budget_root: str = ""):
    """The worker's agent for one task drive (``supervisor/worker_process.py``): no model call."""
    from ouroboros import provider_models
    from ouroboros.agent import OuroborosAgent, make_agent

    monkeypatch.setattr(provider_models, "model_has_credentials", lambda _model: True)
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    agent = make_agent(str(repo), str(drive), budget_drive_root=budget_root)
    agent._current_chat_id = 7
    monkeypatch.setattr(agent.llm, "chat", lambda **_kw: pytest.fail("knowledge roots need no model call"))
    return agent


def _next_root_context(monkeypatch, repo: Path, drive: Path, project: str) -> str:
    """The next ordinary root on the installation drive: its real first request."""
    agent = _real_agent(monkeypatch, repo, drive)
    _ctx, messages, _caps = agent._prepare_task_context(
        {"id": "next0001", "type": "task", "text": "Continue", **({"project_id": project} if project else {})})
    return json.dumps(messages, ensure_ascii=False)


def _archive(registry: ToolRegistry, topic: str, scope: str, revision: str) -> dict:
    from ouroboros.loop_tool_execution import _execute_single_tool

    call = {"id": "archive-call", "function": {"name": "knowledge_write", "arguments": json.dumps({
        "topic": topic, "scope": scope, "mode": "archive", "expected_revision": revision,
        "reason": "Superseded by a later lesson."})}}
    executed = _execute_single_tool(registry, call, Path(registry._ctx.drive_root) / "logs", registry._ctx.task_id)
    assert not executed["is_error"], executed
    return _receipt(executed["result"])


def _admitted_child(tmp_path, monkeypatch, project: str, ambient: str):
    """A child admitted for real: ``schedule_subagent`` → supervisor admission → the worker's split runner.

    The parent's drive is canonical; ``ambient`` names the configured DATA_DIR (that drive, or ``elsewhere``)."""
    from ouroboros.tools.control import _schedule_task
    from supervisor import events

    repo, drive, _ = _env(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setattr("ouroboros.config.DATA_DIR", drive if ambient == "canonical" else elsewhere)
    store.write_knowledge_note(store.resolve_knowledge_address(drive, "people/owner"), _note("SEED OWNER"))
    seed_result(drive, "parent", {"status": "running", "type": "task"})  # the running root that delegates
    parent = ToolContext(repo_dir=repo, drive_root=drive, task_id="parent", current_chat_id=7,
                         task_metadata={"root_task_id": "parent"}, project_id=project)
    _schedule_task(parent, subagent_id=configure_test_subagent(monkeypatch), objective="Keep what I learn",
                   expected_output="Notes", memory_mode="forked")
    [event] = parent.pending_events
    supervisor = _supervisor(drive)
    events._handle_schedule_task(event, supervisor)
    [task] = supervisor.PENDING
    agent = _real_agent(monkeypatch, repo, Path(task["drive_root"]), str(task.get("budget_drive_root") or drive))
    ctx, _messages, _caps = agent._prepare_task_context(task)
    return repo, drive, elsewhere, task, agent, ctx


@pytest.mark.serial
@pytest.mark.parametrize("ambient", ["canonical", "elsewhere"])
@pytest.mark.parametrize("project", ["", "demo"])
def test_an_admitted_child_keeps_knowledge_where_every_reader_reads(tmp_path, monkeypatch, project, ambient):
    """``schedule_subagent`` → supervisor admission → the worker's split runner and its own tools; then the
    terminal path removes the execution drive and every reader still finds the notes on the parent's drive."""
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.delegation_knowledge_index import _with_index
    from ouroboros.extension_process_runner import _apply_tool_context_payload, _tool_context_payload
    from ouroboros.headless import copy_child_task_result, remove_subagent_task_drive, retry_child_task_refs

    repo, drive, elsewhere, task, agent, ctx = _admitted_child(tmp_path, monkeypatch, project, ambient)
    child_drive = Path(task["drive_root"])
    registry = agent.tools
    # The real producer: a split execution drive (a forked seed of the global shelf rides on it), with the
    # parent's drive as its canonical root in both the task metadata and the context.
    assert child_drive != drive and (child_drive / "memory" / "knowledge").is_dir() != bool(project)
    assert canonical_data_root(ctx) == drive.resolve()
    assert ctx.task_metadata["budget_drive_root"] == ctx.budget_drive_root == str(drive)

    room = f"project:{project}" if project else "global"
    sources = {topic: _receipt(registry.execute("knowledge_write", {
        "topic": topic, "content": _note(f"SUMMARY {topic}"), **({"scope": scope} if scope else {})}))["knowledge_source"]
        for topic, scope in (("lessons/root-unity", ""), ("people/ally", "global"), ("patterns", room))}
    assert {topic: Path(source["path"]) for topic, source in sources.items()} == {
        "lessons/root-unity": store.resolve_knowledge_address(drive, "lessons/root-unity", room).path,
        "people/ally": store.resolve_knowledge_address(drive, "people/ally", "global").path,
        "patterns": store.resolve_knowledge_address(drive, "patterns", "global").path}
    assert not (child_drive / "memory" / "knowledge" / "people" / "ally.md").exists()
    assert _untouched(elsewhere)
    assert "SUMMARY lessons/root-unity" in registry.execute("knowledge_list", {})
    assert "Exact meaning" in registry.execute("knowledge_read", {"topic": "people/ally", "scope": "global"})
    # An extension call carries the same roots (``extension_process_runner``): the same address.
    carried = _apply_tool_context_payload(ToolContext(repo_dir=repo, drive_root=drive), _tool_context_payload(ctx))
    assert tools._address(carried, "lessons/root-unity") == tools._address(ctx, "lessons/root-unity")

    previous = _archive(registry, "lessons/root-unity", room, sources["lessons/root-unity"]["revision"])
    previous = previous["knowledge_previous_source"]
    # Published on the canonical root before any terminal copy-back exists.
    assert json.loads(read_actor_source_bytes(drive, task["id"], previous))["old_content"] == _note(
        "SUMMARY lessons/root-unity")
    # The terminal path: result copy-back and source promotion, then the one deletion owner.
    _store_task_result(SimpleNamespace(drive_root=child_drive), task, "done", {}, {},
                       loop_outcome={"reason_code": "completed"})
    copy_child_task_result(drive, task)
    retry_child_task_refs(drive, child_drive, task["id"])
    assert remove_subagent_task_drive(drive, task["id"], live=lambda _tid: False)
    assert not child_drive.exists()
    assert _read_previous(registry, previous) == _note("SUMMARY lessons/root-unity")
    assert _read_previous_in_light(ctx, previous) == _note("SUMMARY lessons/root-unity")

    # Next delegated context: the index this child's first delegation would carry.
    index = _with_index({"result": "Subagent request queued."}, ctx)["result"]
    assert "SUMMARY people/ally" in index and "SUMMARY lessons/root-unity" not in index  # archived
    assert "Archived notes: 1" in index and "SEED OWNER" in index
    # Next normal context: the next root on the installation drive.
    following = _next_root_context(monkeypatch, repo, drive, project)
    assert "SUMMARY people/ally" in following and "SUMMARY lessons/root-unity" not in following
    assert "Archived notes: 1" in following
    # Atlas over the host's data root reads the exact revision the tool published.
    atlas = _atlas_reads(drive, "global", "people/ally")
    assert atlas["revision"] == sources["people/ally"]["revision"] and "Exact meaning" in atlas["text"]
    assert _untouched(elsewhere)


NOTE_READER = '''
import json

API = []


def _note(scope, topic):
    return json.dumps(API[0].read_memory("document", {"address": {"kind": "note", "scope": scope, "topic": topic}}))


def register(api):
    API.append(api)
    api.register_tool("note", _note, description="One knowledge note through the memory reader.",
                      schema={"type": "object", "properties": {"scope": {"type": "string"}, "topic": {"type": "string"}}})
'''


@pytest.mark.serial
def test_an_admitted_childs_extension_reads_its_notes_through_the_plugin_api(tmp_path, monkeypatch):
    """The extension producer: ``dispatch_extension_tool_subprocess`` resolves the data root of the admitted
    child's call, the child's own entry (``_child_main``) loads the extension over it and the tool answers
    with ``PluginAPI.read_memory``. Only the spawn is replaced: the entry runs here on the exact payload."""
    from ouroboros import extension_process_runner as runner
    from ouroboros.extension_surface_names import extension_surface_name
    from tests._extension_loader_shared import _prepare_extension
    from tests._shared import clean_extension_runtime_state

    _repo, drive, elsewhere, task, agent, ctx = _admitted_child(tmp_path, monkeypatch, "demo", "elsewhere")
    child_drive = Path(task["drive_root"])
    notes = (("lessons/root-unity", "project:demo"), ("people/ally", "global"))
    sources = {topic: _receipt(agent.tools.execute("knowledge_write", {
        "topic": topic, "scope": scope, "content": _note(f"SUMMARY {topic}")}))["knowledge_source"] for topic, scope in notes}
    for decoy in (child_drive, elsewhere):  # the same addresses on the execution drive and the ambient DATA_DIR
        for topic, scope in notes:
            assert store.write_knowledge_note(store.resolve_knowledge_address(decoy, topic, scope), _note("DECOY")).ok
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    _skill, skills, _ = _prepare_extension(tmp_path, "note_reader", NOTE_READER, permissions=["tool"],
                                           extra_frontmatter='plugin_api: "2.1"\n')
    payloads = []

    def child_entry_here(payload, **_spawn):
        payloads.append(payload)
        call, result = tmp_path / "extension-call.json", tmp_path / "extension-call.result.json"
        call.write_text(json.dumps({**payload, "result_path": str(result)}), encoding="utf-8")
        runner._child_main(str(call))
        outcome = json.loads(result.read_text(encoding="utf-8"))
        assert outcome.get("ok") is True, outcome
        return outcome

    monkeypatch.setattr(runner, "_run_child", child_entry_here)
    ext_tool = {"name": extension_surface_name("note_reader", "note"), "skill": "note_reader",
                "skills_repo_path": str(skills), "timeout_sec": 30}  # the registry entry fields the producer reads
    try:
        for topic, scope in notes:
            answer = json.loads(runner.dispatch_extension_tool_subprocess(ext_tool, ctx, {"scope": scope, "topic": topic}))
            assert answer["revision"] == sources[topic]["revision"], answer
            assert f"SUMMARY {topic}" in answer["text"] and "DECOY" not in answer["text"]
    finally:
        clean_extension_runtime_state()
    assert [payload["drive_root"] for payload in payloads] == [str(drive.resolve())] * len(notes)
    assert {payload["ctx"]["drive_root"] for payload in payloads} == {str(child_drive)}  # the call still names its drive


@pytest.mark.serial
@pytest.mark.parametrize("ambient", ["canonical", "elsewhere"])
@pytest.mark.parametrize("project", ["", "demo"])
def test_a_root_on_its_installation_drive_keeps_knowledge_on_that_drive(tmp_path, monkeypatch, project, ambient):
    """A root on the shared drive carries no budget root (``gateway/tasks.py``, the chat lane): the drive is
    canonical. ``elsewhere`` stands for a configured DATA_DIR that does not name this installation."""
    repo, drive, _ = _env(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setattr("ouroboros.config.DATA_DIR", drive if ambient == "canonical" else elsewhere)
    agent = _real_agent(monkeypatch, repo, drive)
    ctx, _messages, _caps = agent._prepare_task_context(
        {"id": "root0001", "type": "task", "text": "Learn", **({"project_id": project} if project else {})})
    registry = agent.tools
    assert not ctx.budget_drive_root and not ctx.task_metadata.get("budget_drive_root")
    assert canonical_data_root(ctx) == drive.resolve()

    room = f"project:{project}" if project else "global"
    sources = {topic: _receipt(registry.execute("knowledge_write", {
        "topic": topic, "content": _note(f"SUMMARY {topic}"), **({"scope": scope} if scope else {})}))["knowledge_source"]
        for topic, scope in (("lessons/root-unity", ""), ("people/ally", "global"), ("old-lesson", ""))}
    assert {topic: Path(source["path"]) for topic, source in sources.items()} == {
        topic: store.resolve_knowledge_address(drive, topic, "global" if topic == "people/ally" else room).path
        for topic in sources}
    assert _untouched(elsewhere)
    previous = _archive(registry, "old-lesson", room, sources["old-lesson"]["revision"])["knowledge_previous_source"]
    assert _read_previous(registry, previous) == _note("SUMMARY old-lesson")

    following = _next_root_context(monkeypatch, repo, drive, project)
    for summary in ("SUMMARY lessons/root-unity", "SUMMARY people/ally"):
        assert summary in following
    assert "SUMMARY old-lesson" not in following and "Archived notes: 1" in following
    for topic, scope in (("lessons/root-unity", room), ("people/ally", "global")):
        assert _atlas_reads(drive, scope, topic)["revision"] == sources[topic]["revision"]
    assert _untouched(elsewhere)


def _contract_ctx(tmp_path, binding: str):
    """A synthetic context: canonical, execution and ambient roots are three different directories."""
    canonical, execution, ambient = (tmp_path / name for name in ("canonical", "execution", "ambient"))
    for root in (canonical, execution):
        root.mkdir()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=execution, task_id="contract1", project_id="demo",
                      budget_drive_root=str(canonical) if binding in ("context", "both") else "",
                      task_metadata={"budget_drive_root": str(canonical)} if binding in ("metadata", "both") else {})
    return ctx, canonical, execution, ambient


@pytest.mark.parametrize("binding", ["metadata", "context", "both", "none"])
@pytest.mark.parametrize("topic,scope", [("ordinary", ""), ("ordinary", "global"), ("ordinary", "project:demo"),
                                         ("patterns", "project:demo")])
def test_contract_shapes_write_read_list_and_retain_on_the_canonical_root(tmp_path, monkeypatch, binding, topic, scope):
    """Without a budget root (``none``) the execution drive is the installation's only root and is canonical."""
    ctx, canonical, execution, ambient = _contract_ctx(tmp_path, binding)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", ambient)
    root = canonical if binding != "none" else execution
    assert canonical_data_root(ctx) == root.resolve()
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=execution)
    registry.set_context(ctx)
    args = {"topic": topic, **({"scope": scope} if scope else {})}
    written = _receipt(registry.execute("knowledge_write", {**args, "content": _note("CONTRACT SUMMARY")}))
    expected = store.resolve_knowledge_address(
        root, topic, "global" if topic == "patterns" else scope, "" if topic == "patterns" else "demo")
    assert Path(written["knowledge_source"]["path"]) == expected.path
    assert tools._address(ctx, topic, scope) == expected
    assert "Exact meaning" in registry.execute("knowledge_read", args)
    assert "CONTRACT SUMMARY" in registry.execute("knowledge_list", {"scope": expected.scope})
    assert _untouched(ambient)

    previous = _archive(registry, topic, expected.scope, written["knowledge_source"]["revision"])
    previous = previous["knowledge_previous_source"]
    if binding != "none":
        shutil.rmtree(execution)  # the execution drive is gone; the canonical source stays readable
    assert _read_previous(registry, previous) == _note("CONTRACT SUMMARY")


@pytest.mark.parametrize("binding", ["metadata", "context", "none"])
def test_contract_shapes_deliver_the_canonical_index_with_a_first_delegation(tmp_path, monkeypatch, binding):
    from ouroboros.delegation_knowledge_index import _with_index

    ctx, canonical, execution, ambient = _contract_ctx(tmp_path, binding)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", ambient)
    root = canonical if binding != "none" else execution
    for decoy in {execution, ambient} - {root}:  # same topics elsewhere must stay unseen
        for scope in ("global", "project:demo"):
            store.write_knowledge_note(store.resolve_knowledge_address(decoy, "room-plan", scope), _note("DECOY"))
    store.write_knowledge_note(store.resolve_knowledge_address(root, "people/owner"), _note("GLOBAL SUMMARY"))
    store.write_knowledge_note(store.resolve_knowledge_address(root, "room-plan", "project:demo"), _note("ROOM SUMMARY"))

    index = _with_index({"result": "Subagent request queued."}, ctx)["result"]
    assert "GLOBAL SUMMARY" in index and "## Project knowledge (demo)" in index and "ROOM SUMMARY" in index
    assert "DECOY" not in index


def test_navigation_renders_a_project_shelf_from_the_root_it_is_given(tmp_path, monkeypatch):
    from ouroboros.context import knowledge_navigation_sections

    root, ambient = tmp_path / "root", tmp_path / "ambient"
    monkeypatch.setattr("ouroboros.config.DATA_DIR", ambient)
    store.write_knowledge_note(store.resolve_knowledge_address(root, "room-plan", "project:demo"), _note("ROOM SUMMARY"))
    store.write_knowledge_note(store.resolve_knowledge_address(ambient, "decoy", "project:demo"), _note("DECOY"))

    text = "\n\n".join(knowledge_navigation_sections(root, "demo"))
    assert "## Project knowledge (demo)" in text and "ROOM SUMMARY" in text and "DECOY" not in text

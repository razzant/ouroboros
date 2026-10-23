"""Authored memory: the acting mind writes its lessons, register and backlog; the host keeps facts.

Every test drives a real public consumer — the knowledge tool through the registry, the
post-task coordinator, startup recovery and the maintenance pass, the evolution chooser, the
context builder in each mode, ``journal_read`` — and only model transports are fixtures.
What they pin: one writer (revision-checked, history kept) for patterns and the backlog;
no automatic lesson, register rewrite, memory action or backlog effect after any task shape,
including a Pause an earlier version saved; the chooser reads the complete authored backlog;
historical reports and automatic reflections stay readable at their source while contexts
carry dated pointers, and late review settlements stay whole.
"""

from __future__ import annotations

from tests import _dialogue as dialogue_test

import json
import os
import threading
from types import SimpleNamespace

import pytest

from ouroboros import knowledge as store
from ouroboros.context_fit import PATTERN_REGISTER_POINTER
from ouroboros.task_results import load_task_result
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_consolidator_context_fit import fit  # noqa: F401 — offline Light route/window facts
from tests.test_late_phase_pause_cut import (  # noqa: F401
    ROOT, _maintenance, _phase, _resume, _row, _saved_by_an_earlier_version, late,
)

BACKLOG = "improvement-backlog"
LEGACY_BACKLOG = (
    "---\ntype: backlog\ncustom: kept\n---\n# Improvement Backlog\n\n"
    "### ibl-7\n- status: done\n- count: 3\n- fingerprint: fp-7\n- summary: Closed by an earlier host writer.\n")
REGISTER_ROWS = 40
REGISTER = ("# Pattern Register\n\n| Error class | Count | Root cause | Fix | Status |\n|---|---|---|---|---|\n"
            + "".join(f"| class-{n:02d} | {n} | cause {n} | fix {n} | open |\n" for n in range(REGISTER_ROWS))
            + "LAST REGISTER LINE\n")


def _registry(root, **fields):
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry.set_context(ToolContext(repo_dir=root, drive_root=root, task_id=fields.pop("task_id", "mind"), **fields))
    return registry


def _ok(reply):
    assert reply.startswith("✅"), reply
    return json.loads(reply.split("\n", 1)[1])


def _history(root, topic):
    path = root / "memory" / "knowledge_history.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("topic") == topic]


def _address(root, topic):
    return store.resolve_knowledge_address(root, topic, "global")


# --- one writer: the knowledge tool ------------------------------------------------------


def test_the_backlog_is_an_ordinary_note_written_read_and_revised_through_the_tool(tmp_path):
    registry = _registry(tmp_path)
    address = _address(tmp_path, BACKLOG)
    created = _ok(registry.execute("knowledge_write", {"topic": BACKLOG, "content": LEGACY_BACKLOG}))
    assert address.path.read_bytes() == LEGACY_BACKLOG.encode()  # every field and status as written
    read = registry.execute("knowledge_read", {"topic": BACKLOG, "scope": "global"})
    assert created["knowledge_source"]["revision"] in read and "- count: 3" in read
    # An append needs no revision and changes nothing it did not add: no id, count or status is minted.
    _ok(registry.execute("knowledge_write", {"topic": BACKLOG, "mode": "append",
                                             "content": "Freeform: late phases should resume once.\n"}))
    appended = store.read_knowledge_note(address)
    assert appended.raw == LEGACY_BACKLOG.encode() + b"Freeform: late phases should resume once.\n"
    # An edit names the revision it read; unknown metadata survives it.
    _ok(registry.execute("knowledge_write", {"topic": BACKLOG, "mode": "edit", "old_str": "should resume once.",
                                             "content": "resume once, then stop.",
                                             "expected_revision": appended.revision}))
    edited = store.read_knowledge_note(address)
    assert edited.metadata["custom"] == "kept" and edited.text.endswith("resume once, then stop.\n")
    assert edited.text.count("- count: 3") == 1 and edited.text.count("- status: done") == 1
    # A stale revision refuses and hands back the current source; a blind overwrite needs one.
    stale = registry.execute("knowledge_write", {"topic": BACKLOG, "content": "Rewritten.",
                                                 "expected_revision": appended.revision})
    assert "revision_conflict" in stale and "resume once, then stop." in stale
    assert "revision_required" in registry.execute("knowledge_write", {"topic": BACKLOG, "content": "Rewritten."})
    assert store.read_knowledge_note(address).raw == edited.raw
    # Every revision keeps its complete old and new source.
    rows = _history(tmp_path, BACKLOG)
    assert [row["mode"] for row in rows] == ["overwrite", "append", "edit"]
    assert all(row["writer"] == "turn" for row in rows)
    assert rows[0]["old_content"] == "" and "Closed by an earlier host writer." in rows[0]["new_content"]
    assert rows[2]["old_content"].endswith("should resume once.\n")
    assert rows[2]["new_content"].endswith("resume once, then stop.\n")


def test_two_writers_from_one_read_land_once(tmp_path):
    first, second = _registry(tmp_path, task_id="a"), _registry(tmp_path, task_id="b")
    _ok(first.execute("knowledge_write", {"topic": BACKLOG, "content": "- one\n"}))
    revision = store.read_knowledge_note(_address(tmp_path, BACKLOG)).revision
    barrier, replies = threading.Barrier(2), {}

    def write(name, registry, text):
        barrier.wait(10)
        replies[name] = registry.execute("knowledge_write", {"topic": BACKLOG, "content": text,
                                                             "expected_revision": revision})

    threads = [threading.Thread(target=write, args=("a", first, "- A\n")),
               threading.Thread(target=write, args=("b", second, "- B\n"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
        assert not thread.is_alive()
    winners = [name for name, reply in replies.items() if reply.startswith("✅")]
    assert len(winners) == 1, replies
    loser = ({"a", "b"} - set(winners)).pop()
    assert "revision_conflict" in replies[loser]
    text = store.read_knowledge_note(_address(tmp_path, BACKLOG)).text
    assert ("- A" in text) != ("- B" in text)
    assert len(_history(tmp_path, BACKLOG)) == 2  # the create and the one winner


def test_the_pattern_register_is_edited_like_any_note_and_rides_whole(tmp_path):
    from ouroboros.context import build_knowledge_sections

    registry = _registry(tmp_path)
    _ok(registry.execute("knowledge_write", {"topic": "patterns", "content": REGISTER}))
    current = store.read_knowledge_note(_address(tmp_path, "patterns"))
    _ok(registry.execute("knowledge_write", {
        "topic": "patterns", "mode": "edit", "old_str": "| class-39 | 39 | cause 39 | fix 39 | open |",
        "content": "| class-39 | 39 | cause 39 | fixed by its author | closed: verified |",
        "expected_revision": current.revision}))
    text = "\n\n".join(build_knowledge_sections(SimpleNamespace(drive_path=lambda rel: tmp_path / rel)))
    assert "## Known error patterns (Pattern Register)" in text
    assert all(f"| class-{n:02d} |" in text for n in range(REGISTER_ROWS)) and "LAST REGISTER LINE" in text
    assert "fixed by its author" in text
    assert [row["mode"] for row in _history(tmp_path, "patterns")] == ["overwrite", "edit"]


def test_a_correction_keeps_the_attributed_narrative_recoverable(tmp_path):
    """The semantic source witness: correcting an attributed account leaves the original,
    with its attribution, in the note's history — readable through the actor's own reader."""
    from ouroboros.tools.core_file_tools import _read_file

    registry = _registry(tmp_path)
    original = "# Alex\n\nAlex said on 2026-09-01 (chat row 42) that Alpha shipped in June.\n"
    _ok(registry.execute("knowledge_write", {"topic": "people/alex", "scope": "global", "content": original}))
    note = store.read_knowledge_note(_address(tmp_path, "people/alex"))
    _ok(registry.execute("knowledge_write", {
        "topic": "people/alex", "scope": "global", "mode": "edit",
        "old_str": "that Alpha shipped in June.", "content": "that Alpha shipped in June; it shipped in July (release notes).",
        "expected_revision": note.revision}))
    [_, correction] = _history(tmp_path, "people/alex")
    assert "Alex said on 2026-09-01 (chat row 42) that Alpha shipped in June.\n" in correction["old_content"]
    assert "it shipped in July (release notes)." in correction["new_content"]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="later")
    history = _read_file(ctx, path="memory/knowledge_history.jsonl", root="runtime_data")
    assert "(chat row 42) that Alpha shipped in June." in history


def test_a_delegated_child_still_authors_an_explicit_note_on_the_canonical_shelf(tmp_path):
    """Removing the automatic writers bans no explicit author: a child's own knowledge_write
    lands, stamped as a turn, on the canonical shelf its rights already reach."""
    canonical, fork = tmp_path / "canon", tmp_path / "fork"
    canonical.mkdir()
    fork.mkdir()
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=fork)
    registry.set_context(ToolContext(
        repo_dir=tmp_path, drive_root=fork, budget_drive_root=str(canonical), task_id="kid",
        task_metadata={"delegation_role": "subagent", "parent_task_id": "root", "root_task_id": "root",
                       "budget_drive_root": str(canonical)}))
    _ok(registry.execute("knowledge_write", {"topic": "notes/child-finding", "scope": "global",
                                             "content": "A child's explicit finding."}))
    assert store.read_knowledge_note(_address(canonical, "notes/child-finding")).text.endswith("A child's explicit finding.")
    assert [(row["writer"], row["task_id"]) for row in _history(canonical, "notes/child-finding")] == [("turn", "kid")]
    assert not (fork / "memory" / "knowledge").exists()


def test_the_pressure_path_compresses_and_ignores_old_shaped_nominations(tmp_path):
    from ouroboros import consolidator
    from ouroboros.memory import Memory

    memory = Memory(tmp_path)
    for index in range(2):
        memory.append_scratchpad_block(f"block-{index} " + "z" * 500, source=f"source-{index}")

    class Answer:
        def chat(self, **_kwargs):
            return {"content": json.dumps({"knowledge_entries": [{"topic": "lessons/pressure", "content": "Nominated."}],
                                           "compressed_block": "Short."})}, {"cost": 0.01}

    consolidator.consolidate_scratchpad(memory, Answer(), pressure=True)
    assert [block["content"] for block in memory.load_scratchpad_blocks()] == ["Short."]
    assert not (tmp_path / "memory" / "knowledge").exists()
    assert not (tmp_path / "memory" / "knowledge_history.jsonl").exists()


# --- the post-task census: no learning call and no note write after any task shape ----------


class _Transport:
    """Every paid post-task send, answered as its production prompt asks."""

    def __init__(self, *_args, **_kwargs):
        pass

    def chat(self, *, messages, **_kwargs):
        prompt = str(messages[0]["content"])
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.0}
        if "scratchpad working memory" in prompt:
            return {"content": json.dumps({"compressed_block": "Compressed working notes."})}, usage
        if "You decide whether Ouroboros should run ONE reviewed self-improvement" in prompt:
            return {"content": json.dumps({"promote": False, "objective": ""})}, usage
        return {"content": "ok"}, usage


def _offline_light(monkeypatch):
    from ouroboros import consolidator, context_fit
    from ouroboros.capability_evidence import CapabilityEvidence

    monkeypatch.setattr(consolidator, "_consolidation_route", lambda: ("test/model", False))
    monkeypatch.setattr(context_fit, "resolve_context_fit_route", lambda task, *, allow_fetch: (
        {"model": task["model"], "provider": "openrouter"},
        CapabilityEvidence(1_000_000, "confirmed", "test", "route-test", model=task["model"], provider="openrouter")))
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_, **_kw: 1.0)


def _recorded_call_types(monkeypatch):
    from ouroboros import llm_observability

    seen, real = [], llm_observability.chat_observed

    def recording(llm, **kwargs):
        seen.append(kwargs.get("call_type"))
        return real(llm, **kwargs)

    monkeypatch.setattr(llm_observability, "chat_observed", recording)
    return seen


_PRESENCE = {"transport_skill": "slack-bridge", "binding_id": "b" * 32, "event": {"source_event_id": "in-1"}}
_SHAPES = {
    "failed_costly": ({"text": "Fix the flaky deploy"}, {"rounds": 60, "cost": 25.0}, True),
    "workspace": ({"text": "Edit the site", "workspace_root": "/tmp/site", "workspace_mode": "external"},
                  {"rounds": 4, "cost": 0.2}, False),
    "project": ({"text": "Ship the deck", "project_id": "p1"}, {"rounds": 4, "cost": 0.2}, False),
    "presence": ({"text": "hello", "_presence_turn": True, "metadata": {"presence": _PRESENCE}},
                 {"rounds": 2, "cost": 0.0}, False),
    "evolution_task": ({"text": "Improve X", "type": "evolution"}, {"rounds": 9, "cost": 1.0}, False),
    "clean": ({"text": "Say hi"}, {"rounds": 1, "cost": 0.0}, False),
}


@pytest.mark.parametrize("opted_in", [False, True], ids=["evolution_off", "evolution_on"])
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_no_task_shape_buys_a_lesson_or_writes_a_note(tmp_path, monkeypatch, shape, opted_in):
    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.llm as llm_mod
    from ouroboros.memory import Memory

    _offline_light(monkeypatch)
    monkeypatch.setattr(llm_mod, "LLMClient", _Transport)
    call_types = _recorded_call_types(monkeypatch)
    if opted_in:
        monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    else:
        monkeypatch.delenv("OUROBOROS_POST_TASK_EVOLUTION", raising=False)
    fields, usage, errored = _SHAPES[shape]
    (tmp_path / "logs").mkdir()
    memory = Memory(tmp_path)
    for index in range(4):  # past the compression threshold: the one paid upkeep every root keeps
        memory.append_scratchpad_block(f"note-{index}-" + chr(97 + index) * 8_000, source=f"task-{index}")
    trace = {"tool_calls": [{"tool": "run_command", "status": "error", "is_error": True,
                             "result": "TOOL_ERROR: boom"}] * (3 if errored else 0), "reasoning_notes": []}
    task = {"id": f"root-{shape}", "type": "task", "chat_id": 1, "drive_root": str(tmp_path), **fields}
    env = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, drive_path=lambda rel: tmp_path / rel)
    pipeline._store_task_result(env, task, "Done.", usage, trace)
    pipeline._run_post_task_processing_async(env, task, usage, trace, tmp_path / "logs", blocking=True)

    # A workspace binds a project; Project, Presence and evolution tasks never reach the global chooser.
    chooser = opted_in and shape in {"failed_costly", "clean"}
    assert call_types == ["scratchpad_consolidation"] + (["post_task_evolution_decision"] if chooser else []), call_types
    assert not (tmp_path / "memory" / "knowledge_history.jsonl").exists()  # no note, register or backlog write
    assert not (tmp_path / "memory" / "knowledge").exists()
    assert not (tmp_path / "logs" / "task_reflections.jsonl").exists()
    assert memory.load_scratchpad_blocks()[0]["content"] == "Compressed working notes."
    rows = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    assert [row.get("summary_kind") for row in rows if row.get("task_id") == task["id"]].count("host_task_facts") == 1
    checkpoint = pipeline.load_task_result(tmp_path, task["id"])["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "completed" and "post_task_stop_reason" not in checkpoint


def test_a_chooser_that_cannot_be_prepared_degrades_the_phase_with_its_cause_recorded(tmp_path, monkeypatch):
    """No new size rule: a request the route refuses is the stage's ordinary failure, with the
    refusal kept by the call's observability record, not a swallowed abstention."""
    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.llm as llm_mod
    from ouroboros import llm_observability

    class Refusing(_Transport):
        def chat(self, *, messages, **kwargs):
            if "You decide whether Ouroboros should run ONE reviewed self-improvement" in str(messages[0]["content"]):
                raise RuntimeError("context window exceeded: request 2100000 tokens > 1000000")
            return super().chat(messages=messages, **kwargs)

    _offline_light(monkeypatch)
    monkeypatch.setattr(llm_mod, "LLMClient", Refusing)
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    persisted, real_persist = [], llm_observability.persist_observed_call

    def recording(root, **kwargs):
        persisted.append((kwargs.get("call_type"), (kwargs.get("manifest") or {}).get("error", "")))
        return real_persist(root, **kwargs)

    monkeypatch.setattr(llm_observability, "persist_observed_call", recording)
    (tmp_path / "logs").mkdir()
    task = {"id": "root-refused", "type": "task", "chat_id": 1, "text": "Big work", "drive_root": str(tmp_path)}
    env = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, drive_path=lambda rel: tmp_path / rel)
    pipeline._store_task_result(env, task, "Done.", {"rounds": 3}, {"tool_calls": []})
    pipeline._run_post_task_processing_async(env, task, {"rounds": 3}, {"tool_calls": []}, tmp_path / "logs",
                                             blocking=True)

    checkpoint = pipeline.load_task_result(tmp_path, "root-refused")["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded" and "post_task_stop_reason" not in checkpoint
    [(call_type, error)] = [row for row in persisted if row[0].endswith("_error")]
    assert call_type == "post_task_evolution_decision_error" and "context window exceeded" in error


# --- a Pause an earlier version saved: its writes never revive ---------------------------------


def _as_saved_by_an_earlier_version(f, *, stage, completed, remaining, marks, callback=None):
    """The v1 payload an earlier version wrote for its Pause, its retired reflection's nominations and all."""
    entry = {"reflection": "OLD AUTOMATIC LESSON", "memory_actions": [
        {"type": "knowledge_write", "topic": "lessons/old", "content": "OLD LESSON NOTE", "task_id": ROOT},
        {"type": "scratchpad_append", "content": "OLD SCRATCH NOTE", "task_id": ROOT},
        {"type": "identity_update_candidate", "content": "OLD TRAIT", "task_id": ROOT}],
        "backlog_candidates": [{"summary": "OLD CANDIDATE", "category": "process"}]}
    _saved_by_an_earlier_version(f, stage=stage, completed=completed, remaining=remaining, payload={
        "marks": marks, "review_evidence": {"task_id": ROOT, "has_evidence": True},
        "result": {"reflection_entry": entry}, "free_actions_applied": False, "reflection_callback": callback})


def _no_saved_write_took_effect(f):
    from ouroboros.memory import Memory

    memory_text = "".join(path.read_text(encoding="utf-8", errors="replace")
                          for path in (f.root / "memory").rglob("*") if path.is_file())
    assert "OLD LESSON NOTE" not in memory_text and "OLD SCRATCH NOTE" not in memory_text
    assert "OLD TRAIT" not in memory_text and "OLD CANDIDATE" not in memory_text
    assert not (f.root / "memory" / "knowledge" / "lessons").exists()
    assert "OLD" not in json.dumps(Memory(f.root).load_scratchpad_blocks())


def test_a_pause_saved_inside_the_retired_reflection_ends_without_its_writes(late):  # noqa: F811
    f = late
    _as_saved_by_an_earlier_version(f, stage="reflection", completed=["memory_fallback_draft", "scratchpad_consolidation"],
                                    remaining=["reflection", "promotion"], marks=[f"backlog_appended:{f.root}"])
    # Restart runs nothing; the maintenance pass ends it once without running any stage.
    assert f.pipeline.recover_pending_root_post_task_synthesis(f.root, f.root) == 0
    assert _phase(f) == "paused"
    assert _maintenance(f) == 1
    checkpoint = _row(f)["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded"
    assert checkpoint["post_task_stop_reason"] == "owner_pause:skipped=reflection,promotion"
    assert f.light.sends == [] and f.stages == []
    _no_saved_write_took_effect(f)
    assert not _resume(f).get("ok")  # nothing left to resume


def test_a_promotion_an_earlier_version_saved_ends_without_buying_a_step_or_its_nominations(late, monkeypatch):  # noqa: F811
    """An earlier version's split Project callback, paused inside promotion: the maintenance
    pass ends it; neither the chooser nor the global entry is bought, and the saved reflection
    entry and its nominations never take effect."""
    f = late
    seen = []
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote",
                        lambda env, task, sealed, llm: seen.append((str(env.drive_root), task, sealed)))
    callback = {"drive_root": str(f.root), "repo_dir": str(f.root), "task": {"id": ROOT, "type": "task"}}
    _as_saved_by_an_earlier_version(
        f, stage="promotion", completed=["memory_fallback_draft", "scratchpad_consolidation", "reflection"],
        remaining=["promotion"], marks=["promotion:backlog", "promotion:promote", f"backlog_appended:{f.root}"],
        callback=callback)
    assert _maintenance(f) == 1
    assert _row(f)["root_phase_checkpoint"]["post_task_stop_reason"] == "owner_pause:skipped=promotion"
    assert seen == [] and f.light.sends == []
    _no_saved_write_took_effect(f)


def test_a_pause_inside_the_project_entry_ends_it_at_once_and_saves_nothing(late, monkeypatch):  # noqa: F811
    """Today's split Project entry paused mid-chooser: the send on the wire finishes, the next
    is refused, the stage ends with the skip disclosed and nothing is saved to resume."""
    import functools

    from ouroboros.usage_accounting import UsageScope, usage_scope
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_late_phase_pause_cut import ANSWER, _join, _spawned

    f = late
    calls = []

    def maybe_promote(env, task, sealed, client):
        calls.append(((task.get("metadata") or {}).get("globalized_from_project_task", False), sealed))
        if (task.get("metadata") or {}).get("globalized_from_project_task"):
            for _ in range(2):  # a two-call chooser, paused after its first send
                client.chat(messages=[{"role": "user", "content": "choose from the canonical backlog"}], model_role="light")

    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", maybe_promote)
    f.light.hooks["other"] = [lambda: request_owner_pause(ROOT, request_id="entry-pause")]
    sealed = {"final_result_text": ANSWER, "artifact_manifest": [], "completion_observations": {"status": "unavailable"}}
    entry = functools.partial(f.pipeline._run_global_backlog_promotion_only, f.env, {"id": ROOT, "type": "task"})
    scope = UsageScope(drive_root=f.root, task_id=ROOT, root_task_id=ROOT, category="task", source="agent.task",
                       root_limit_usd=3.0, root_limit_source="task_admission")
    with usage_scope(scope):
        _join(_spawned(lambda: f.pipeline._run_post_task_processing_async(
            f.env, f.task, {"rounds": 1}, {"tool_calls": []}, f.root / "logs", on_reflection=entry,
            sealed_final=sealed))[1])
    assert calls == [(False, sealed), (True, sealed)]
    checkpoint = _row(f)["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded" and "post_task_pause" not in checkpoint
    assert checkpoint["post_task_stop_reason"] == "owner_pause:skipped=promotion"
    assert f.light.kinds().count("other") == 1  # the entry's second send is never bought
    assert not _resume(f).get("ok")
    _no_saved_write_took_effect(f)


def _split_project_root(tmp_path, index, text="PROJECT REQUEST", task_type="task"):
    """One finished split Project root through the real finalization: emit → dispatch → late phase."""
    import ouroboros.agent_task_pipeline as pipeline

    canonical, child = tmp_path / "canonical", tmp_path / f"child-{index}"
    for drive in (canonical, child):
        (drive / "logs").mkdir(parents=True, exist_ok=True)
    pending = []
    pipeline.emit_task_results(
        env=SimpleNamespace(repo_dir=tmp_path, drive_root=child, drive_path=lambda rel: child / rel),
        memory=object(), llm=object(), pending_events=pending,
        task={"id": f"proj-root-{index}", "root_task_id": f"proj-root-{index}", "type": task_type, "chat_id": 1,
              "text": f"{text} {index}", "project_id": "p1", "budget_drive_root": str(canonical)},
        text=f"Shipped deck {index}.", usage={"rounds": 3, "cost": 0.2},
        llm_trace={"tool_calls": [], "reasoning_notes": []}, start_time=0.0, drive_logs=child / "logs",
        ctx=SimpleNamespace(pending_restart_reason=""))
    return canonical, child


@pytest.mark.parametrize("opted_in,cadence,task_type,expected", [
    (False, "every_n:1", "task", [0, 0]), (True, "every_n:1", "task", [1, 1]), (True, "every_n:2", "task", [0, 1]),
    (True, "off", "task", [0, 0]), (True, "every_n:1", "evolution", [0, 0])],
    ids=["evolution_off", "every_task", "every_second_task", "cadence_off", "evolution_task_excluded"])
def test_a_split_project_root_reaches_the_canonical_chooser_under_the_same_opt_in_and_cadence(
        tmp_path, monkeypatch, opted_in, cadence, task_type, expected):
    import ouroboros.config as cfg
    import ouroboros.llm as llm_mod

    prompts = []

    class Chooser(_Transport):
        def chat(self, *, messages, **kwargs):
            prompt = str(messages[0]["content"])
            if "You decide whether Ouroboros should run ONE reviewed self-improvement" in prompt:
                prompts.append(prompt)
                return {"content": json.dumps({"promote": True, "objective": "Tighten deck exports",
                                               "backlog_id": ""})}, {"prompt_tokens": 1, "completion_tokens": 1}
            return super().chat(messages=messages, **kwargs)

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "canonical")
    _offline_light(monkeypatch)
    monkeypatch.setattr(llm_mod, "LLMClient", Chooser)
    call_types = _recorded_call_types(monkeypatch)
    if opted_in:
        monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    else:
        monkeypatch.delenv("OUROBOROS_POST_TASK_EVOLUTION", raising=False)
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION_CADENCE", cadence)
    canonical = tmp_path / "canonical"
    (canonical / "memory").mkdir(parents=True)
    assert store.write_knowledge_note(_address(canonical, BACKLOG), "# Backlog\n\nFreeform: exports drift.\n").ok
    backlog = _address(canonical, BACKLOG).path.read_bytes()
    seen = []
    for index in range(2):
        before = len(prompts)
        _canonical, child = _split_project_root(tmp_path, index, task_type=task_type)
        seen.append(len(prompts) - before)
        assert not (child / "state" / "post_task_evolution_counter.json").exists()  # the project gate counts nothing
        checkpoint = load_task_result(canonical, f"proj-root-{index}")["root_phase_checkpoint"]
        assert checkpoint["post_task_synthesis"] == "completed"
    assert seen == expected, (seen, call_types)
    counter = canonical / "state" / "post_task_evolution_counter.json"
    counted = opted_in and cadence.startswith("every_n") and task_type == "task"
    assert (json.loads(counter.read_text(encoding="utf-8"))["n"] if counter.exists() else 0) == (2 if counted else 0)
    request = canonical / "state" / "post_task_evolution_request.json"
    assert request.exists() == bool(sum(expected))
    if prompts:
        # The chooser sees this run's sealed answer and the whole authored backlog, never the
        # project's initial text; its one write is the canonical request with the run's provenance.
        assert "Shipped deck 1." in prompts[-1] and "Freeform: exports drift." in prompts[-1]
        assert "PROJECT REQUEST" not in prompts[-1]
        assert json.loads(request.read_text(encoding="utf-8"))["origin_task_id"] == "proj-root-1"
    assert _address(canonical, BACKLOG).path.read_bytes() == backlog  # authored backlog untouched
    assert not (canonical / "logs" / "task_reflections.jsonl").exists()


# --- the chooser reads the complete authored backlog -------------------------------------------


def _chooser_prompt(tmp_path, monkeypatch, task, sealed=None):
    from ouroboros import post_task_evolution as pte

    captured = {}

    def chat_observed(_llm, **kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return {"content": json.dumps({"promote": False, "objective": ""})}, {}

    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", chat_observed)
    monkeypatch.setattr(pte, "_active_campaign_objective", lambda: "")
    pte._decide_promotion(SimpleNamespace(drive_root=tmp_path), task, sealed, object(), force=False)
    return captured["prompt"]


def test_the_chooser_reads_the_whole_authored_backlog_and_the_exact_initial_text(tmp_path, monkeypatch):
    entries = "".join(f"## Entry {n}\nFreeform reasoning {n}.\n" + ("Done: absorbed in cycle 4; kept as a record.\n"
                                                                    if n == 11 else "") + "\n" for n in range(12))
    note = store.write_knowledge_note(_address(tmp_path, BACKLOG), entries).current
    initial = "Exact initial text " + "x" * 9000 + " END"
    prompt = _chooser_prompt(tmp_path, monkeypatch, {"id": "t", "text": initial})
    assert all(f"## Entry {n}\nFreeform reasoning {n}." in prompt for n in range(12))
    assert "Done: absorbed in cycle 4; kept as a record." in prompt
    assert (f"Source: knowledge_read(topic='{BACKLOG}', scope='global'), revision {note.revision}, "
            f"{len(note.text)} chars, complete.") in prompt
    assert f"Initial text of the run (exact):\n{initial}" in prompt  # whole: no host cut
    assert "Sealed final outcome: not recorded for this run." in prompt


def test_the_chooser_reads_the_sealed_outcome_bounded_with_its_cut_named(tmp_path, monkeypatch):
    from ouroboros.task_finalization import build_sealed_final_package

    answer = "Final answer " + "y" * 9000 + " SEALED-TAIL"
    prompt = _chooser_prompt(tmp_path, monkeypatch, {"id": "t", "text": "x"}, build_sealed_final_package({}, answer))
    assert answer[:4000] in prompt and "SEALED-TAIL" not in prompt
    assert f"OMISSION NOTE: truncated at 4000 chars; original length {len(answer)}" in prompt


def test_a_missing_backlog_is_named_and_an_unreadable_one_stays_unknown(tmp_path, monkeypatch):
    prompt = _chooser_prompt(tmp_path, monkeypatch, {"id": "t", "text": "x"})
    assert "No improvement-backlog note has been written yet" in prompt
    path = _address(tmp_path, BACKLOG).path
    path.mkdir(parents=True)  # unreadable as a file; Windows reports PermissionError
    with pytest.raises(OSError) as failure:
        path.read_text(encoding="utf-8")
    prompt = _chooser_prompt(tmp_path, monkeypatch, {"id": "t", "text": "x"})
    assert (f"Improvement-backlog source unavailable ({type(failure.value).__name__}); "
            "it is unknown, not empty") in prompt


# --- contexts: the full register where it lived, dated pointers, whole late criticism -------------


DEEP_REVIEW = ("<!-- deep-review provenance: generated_at=2026-09-30T10:00:00Z, source_revision=abc123 -->\n"
               "# Deep review\n\nOLD DEEP REVIEW BODY\n")


def _context_world(tmp_path):
    from ouroboros.memory import Memory
    from ouroboros.plan_review_facts import LATE_SETTLEMENT_TASK_TYPE

    repo, drive = tmp_path / "repo", tmp_path / "data"
    for rel, text in {"prompts/SYSTEM.md": "I am Ouroboros.", "BIBLE.md": "The constitution.",
                      "docs/ARCHITECTURE.md": "# Body\n\nThe body map.\n", "docs/DEVELOPMENT.md": "# Engineering\n"}.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    memory = Memory(drive_root=drive, repo_dir=repo)
    memory.ensure_files()
    (drive / "memory" / "identity.md").write_text("My identity.", encoding="utf-8")
    assert store.write_knowledge_note(_address(drive, "patterns"), REGISTER).ok
    (drive / "memory" / "deep_review.md").write_text(DEEP_REVIEW, encoding="utf-8")
    (drive / "logs").mkdir(exist_ok=True)
    rows = [{"ts": "2026-09-01T10:00:00Z", "task_id": "old-1", "task_type": "task", "goal": "old goal",
             "reflection": "OLD AUTOMATIC LESSON BODY"},
            {"ts": "2026-10-12T11:00:00Z", "task_id": "late-1", "type": LATE_SETTLEMENT_TASK_TYPE,
             "task_type": LATE_SETTLEMENT_TASK_TYPE, "reflection": "LATE CRITICISM: reviewers later found the claim unsupported."},
            {"ts": "2026-10-12T12:00:00Z", "task_id": "p-row", "type": "project_reflection_pointer",
             "project_id": "p1", "reflection_path": "/elsewhere/task_reflections.jsonl", "write_failed": True}]
    (drive / "logs" / "task_reflections.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    env = SimpleNamespace(repo_dir=repo, drive_root=drive, repo_path=lambda p: repo / p, drive_path=lambda p: drive / p)
    return env, memory


def _mode_texts(env, memory, task, ctx=None):
    from ouroboros import context
    from ouroboros.context_fit import build_context_fit_plan

    def route(*_args, **_kwargs):
        return ({"model": "fixture", "provider": "fixture"},
                SimpleNamespace(route_fp="fixture", status="unprobeable", stale=False, window_tokens=0))

    core = context._capture_context_core(env, memory, task, None, ctx)
    plan = build_context_fit_plan(env, core, task, preferred_mode="nano", route_resolver=route)
    return {mode: "\n".join(part["text"] for part in plan.projection(mode).system_message()["content"])
            for mode in ("max", "low", "nano")}


_ROLES = {
    "main": ({"_is_direct_chat": True}, True),
    "project": ({"project_id": "p1"}, True),
    "consciousness": ({"_is_direct_chat": True, "metadata": {
        "initiator": "consciousness", "usage_category": "consciousness", "consciousness_autonomy": "act"}}, True),
    "presence": ({"_presence_turn": True, "metadata": {"presence": _PRESENCE}}, True),
    "child": ({"delegation_role": "subagent", "parent_task_id": "root", "root_task_id": "root"}, False),
    "nanny": ({"delegation_role": "subagent", "parent_task_id": "root", "root_task_id": "root",
               "configured_subagent": {"route": {"kind": "agent_session"}}}, False),
}


@pytest.mark.parametrize("role", sorted(_ROLES))
def test_every_role_and_mode_keeps_its_register_and_carries_history_as_dated_pointers(tmp_path, monkeypatch, role):
    import ouroboros.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    env, memory = _context_world(tmp_path)
    fields, integrating = _ROLES[role]
    task = {"id": f"t-{role}", "type": "task", "text": "Work.", **fields}
    actor = (_speaker(env, task["id"]) if role == "presence"
             else _actor(env, task["id"], is_direct_chat=True) if integrating else None)
    for mode, text in _mode_texts(env, memory, task, actor).items():
        where = (role, mode)
        # No automatic historical prose anywhere: not the old lesson, not the old report body.
        assert "OLD AUTOMATIC LESSON BODY" not in text and "OLD DEEP REVIEW BODY" not in text, where
        if integrating:
            # The full current register in Max and Low, exactly where it already lived; Nano carries its
            # reader line there instead (owner, 2026-10-20: the register out of Nano, a link kept).
            shown = [f"| class-{n:02d} |" in text for n in range(REGISTER_ROWS)] + ["LAST REGISTER LINE" in text]
            assert (not any(shown) and text.count(PATTERN_REGISTER_POINTER) == 1) if mode == "nano" else all(shown), where
            assert "Historical automatic reflections: 1 row(s)" in text and "dated 2026-09-01T10:00:00" in text, where
            assert "read_file(root='runtime_data', path='logs/task_reflections.jsonl')" in text, where
            assert "subject to this task's tool/resource restrictions and Presence grants" in text, where
            # The latest late criticism stays whole; a failed project pointer keeps its diagnostic.
            assert "LATE CRITICISM: reviewers later found the claim unsupported." in text, where
            assert "(project write FAILED)" in text, where
            assert ("generated_at=2026-09-30T10:00:00Z, source_revision=abc123 (its provenance header)" in text), where
        else:
            assert "| class-00 |" not in text and "LAST REGISTER LINE" not in text, where
            assert "LATE CRITICISM" not in text and "## Last Deep Self-Review" not in text, where


def test_the_pointers_are_stable_bytes_and_an_undated_report_stays_unknown(tmp_path, monkeypatch):
    import ouroboros.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    env, memory = _context_world(tmp_path)
    task = {"id": "t-main", "type": "task", "text": "Work.", "_is_direct_chat": True}

    def pointers(text):
        return [line for line in text.splitlines()
                if "Historical automatic reflections" in line or "A historical report kept at its source" in line]

    first = pointers(_mode_texts(env, memory, task)["max"])
    for path in (env.drive_root / "memory" / "deep_review.md", env.drive_root / "logs" / "task_reflections.jsonl"):
        os.utime(path, (1_900_000_000, 1_900_000_000))  # a later file time is not provenance
    assert pointers(_mode_texts(env, memory, task)["max"]) == first and len(first) == 2
    (env.drive_root / "memory" / "deep_review.md").write_text("# Report without a header\n", encoding="utf-8")
    text = _mode_texts(env, memory, task)["max"]
    assert "generated_at=unknown, source_revision=unknown (no provenance header)" in text


# --- each pointer names the call THIS actor's own read gates admit, and that call reads -------------


def _actor(env, task_id, drive_root=None, **fields):
    return ToolContext(repo_dir=env.repo_dir, drive_root=drive_root or env.drive_root, task_id=task_id, **fields)


def _speaker(env, task_id, *targets):
    """A Presence speaker whose host-minted ceiling selects exactly ``targets`` beyond its baseline."""
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.frozen_presence_capabilities import PresenceProfileResolution, PresenceSelection, PresenceToolTarget
    from tests.frozen_presence_runtime import ResolvedPresenceRuntime

    selected = targets or (PresenceToolTarget("builtin", "chat_history"),)
    resolution = PresenceProfileResolution(
        active=tuple(PresenceSelection(f"{index + 1:064x}", target) for index, target in enumerate(selected)),
        missing_required=(), missing_optional=(), orphaned=(), runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64, selection_fingerprint="b" * 64, required_selections_present=True)
    ceiling = build_presence_capability_ceiling(skill_name="moderator", skill_content_hash="c" * 64,
                                                state_fingerprint="d" * 64, resolution=resolution)
    return _actor(env, task_id, task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
                  task_metadata={"_presence_turn": True, "presence": _PRESENCE})


def _readers(text):
    """The ``Read:`` clause of every pointer line, in order."""
    return [line.rsplit("Read: ", 1)[1].removesuffix(".").split(" (reader address,", 1)[0]
            for line in text.splitlines()
            if "Read: " in line and ("Historical automatic reflections" in line or "A historical report kept" in line)]


def _run(ctx, call):
    """Dispatch a rendered ``name(key='value', ...)`` call through the actual ToolRegistry as ``ctx``."""
    import ast

    expr = ast.parse(call.split(" (reader address,", 1)[0], mode="eval").body
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry.execute(expr.func.id, {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords})


def test_a_root_on_the_canonical_drive_opens_both_sources_with_the_calls_it_is_shown(tmp_path, monkeypatch):
    import ouroboros.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    env, memory = _context_world(tmp_path)
    root = _actor(env, "t-main", is_direct_chat=True)
    text = _mode_texts(env, memory, {"id": "t-main", "type": "task", "text": "Work."}, root)["max"]
    deep, log = _readers(text)
    assert deep == "read_file(root='runtime_data', path='memory/deep_review.md')"
    assert log == "read_file(root='runtime_data', path='logs/task_reflections.jsonl')"
    assert "OLD DEEP REVIEW BODY" in _run(root, deep) and "OLD AUTOMATIC LESSON BODY" in _run(root, log)


@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
@pytest.mark.parametrize("focus", ["main", "project", "readonly_child", "acting_child"])
@pytest.mark.parametrize("forked", [False, True])
def test_context_readers_open_canonical_history_in_every_mode_and_focus(tmp_path, monkeypatch, mode, focus, forked):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.context import _canonical_runtime_reader

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path / "elsewhere"))
    env, memory = _context_world(tmp_path)
    drive = tmp_path / "fork" if forked else env.drive_root
    drive.mkdir(exist_ok=True)
    workspace = tmp_path / "work"
    workspace.mkdir()
    fields = {"is_direct_chat": focus == "main", "budget_drive_root": str(env.drive_root)}
    meta = {"budget_drive_root": str(env.drive_root), "delegation_role": "root", "root_task_id": "t-focus"}
    if focus != "main":
        fields.update(project_id="p1", workspace_root=workspace, workspace_mode="external")
    if focus.endswith("child"):
        meta.update(delegation_role="subagent", parent_task_id="parent", root_task_id="parent")
        fields["task_constraint"] = TaskConstraint(
            mode="acting_subagent" if focus == "acting_child" else "local_readonly_subagent",
            surface="external_workspace" if focus == "acting_child" else "", write_root=str(workspace))
    actor = _actor(env, "t-focus", drive_root=drive, task_metadata=meta, **fields)
    task = {"id": "t-focus", "type": "task", "text": "Work.", "budget_drive_root": str(env.drive_root)}
    # Use the real task role: integrating roots emit history pointers, children keep
    # their own smaller context and open a supplied canonical address on demand.
    task.update(project_id=fields.get("project_id", ""), **meta)
    texts = _mode_texts(env, memory, task, actor)
    rels = ("memory/deep_review.md", "logs/task_reflections.jsonl")
    expected = [f"read_file(root='runtime_data', path={str((env.drive_root / rel).resolve())!r})"
                if forked else f"read_file(root='runtime_data', path={rel!r})" for rel in rels]
    for context_mode, text in texts.items():
        if focus.endswith("child"):
            assert not _readers(text) and "LAST REGISTER LINE" not in text
        else:
            assert _readers(text) == expected
            for reader, marker in zip(_readers(text), ("OLD DEEP REVIEW BODY", "OLD AUTOMATIC LESSON BODY")):
                assert marker in _run(actor, reader)
            # The register rides whole in Max and Low; Nano's reader line opens the same canonical note.
            assert "LATE CRITICISM" in text and ("LAST REGISTER LINE" in text) == (context_mode != "nano")
            if context_mode == "nano":
                call = PATTERN_REGISTER_POINTER.split(": ", 1)[1].removesuffix(".")
                assert PATTERN_REGISTER_POINTER in text and "LAST REGISTER LINE" in _run(actor, call)
    for rel in (*rels, "settings.json", "memory/knowledge_history.jsonl"):
        source = env.drive_root / rel
        if not source.exists():
            source.write_text('{"synthetic": "CANONICAL_SETTINGS_OR_HISTORY"}\n', encoding="utf-8")
        if forked:
            local = drive / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text("EXECUTION LOCAL BYTES\n", encoding="utf-8")
        reader = _canonical_runtime_reader(actor, env.drive_root, rel)
        assert source.read_text(encoding="utf-8").splitlines()[-1][:40] in _run(actor, reader)
        assert source.read_text(encoding="utf-8").splitlines()[-1][:40] in _run(actor, f"read_file(path={str(source)!r})")
        if forked:
            assert "EXECUTION LOCAL BYTES" in _run(actor, f"read_file(root='runtime_data', path={rel!r})")


def test_a_presence_ceiling_admits_only_what_it_selected(tmp_path, monkeypatch):
    import ouroboros.config as cfg
    from tests.frozen_presence_capabilities import PresenceResourceTarget, PresenceToolTarget

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    env, memory = _context_world(tmp_path)
    task = {"id": "t-speaker", "type": "task", "text": "hi", "_presence_turn": True, "metadata": {"presence": _PRESENCE}}
    # Addresses are qualified, never access promises. Dispatch refuses the emitted calls.
    denied = _speaker(env, "t-speaker")
    for reader in _readers(_mode_texts(env, memory, task, denied)["max"]):
        assert _run(denied, reader).startswith("⚠️ PRESENCE_CAPABILITY_BLOCKED")
    assert _run(denied, "read_file(root='runtime_data', path='memory/deep_review.md')").startswith(
        "⚠️ PRESENCE_CAPABILITY_BLOCKED")
    # A profile that selected read_file over runtime_data/memory opens the report, not the logs.
    granted = _speaker(env, "t-speaker", PresenceToolTarget("builtin", "read_file"),
                       PresenceResourceTarget("runtime_data", ("read",), "memory"))
    deep, log = _readers(_mode_texts(env, memory, task, granted)["max"])
    assert deep == "read_file(root='runtime_data', path='memory/deep_review.md')"
    assert "OLD DEEP REVIEW BODY" in _run(granted, deep)
    assert _run(granted, log).startswith("⚠️ PRESENCE_RESOURCE_BLOCKED")
    assert _run(granted, "read_file(root='runtime_data', path='logs/task_reflections.jsonl')").startswith(
        "⚠️ PRESENCE_RESOURCE_BLOCKED")


def test_project_pointers_name_journal_read_only_where_its_project_authority_admits_it(tmp_path, monkeypatch):
    import ouroboros.config as cfg
    from ouroboros.plan_review_facts import LATE_SETTLEMENT_TASK_TYPE
    from ouroboros.project_facts import project_reflections_path
    from ouroboros.utils import append_jsonl

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    env, memory = _context_world(tmp_path)
    for pid, body in (("p1", "PROJECT OLD LESSON"), ("p2", "OTHER PROJECT LESSON")):
        path = project_reflections_path(pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ts": "2026-08-10T12:00:00Z", "task_id": f"{pid}-1", "reflection": body}) + "\n",
                        encoding="utf-8")
    append_jsonl(project_reflections_path("p1"), {"ts": "2026-10-11T09:00:00Z", "task_id": "pr-2",
                                                  "type": LATE_SETTLEMENT_TASK_TYPE, "task_type": LATE_SETTLEMENT_TASK_TYPE,
                                                  "reflection": "PROJECT LATE CRITICISM"})
    # The canonical log points at p2's store (written there, not failed).
    append_jsonl(env.drive_root / "logs" / "task_reflections.jsonl", {
        "ts": "2026-10-12T13:00:00Z", "task_id": "p2-row", "type": "project_reflection_pointer",
        "project_id": "p2", "reflection_path": str(project_reflections_path("p2"))})

    def pointers(ctx, task):
        text = _mode_texts(env, memory, {"type": "task", "text": "Work.", **task}, ctx)["max"]
        project = text[text.index("## Project execution reflections (this project's own: p1)"):]
        global_row = next(line for line in text.splitlines() if "Full reflection lives on project drive: p2" in line)
        return project, global_row

    root = _actor(env, "t-p", project_id="p1", is_direct_chat=True)
    project, global_row = pointers(root, {"id": "t-p", "project_id": "p1"})
    assert "PROJECT LATE CRITICISM" in project and "PROJECT OLD LESSON" not in project
    assert "dated 2026-08-10T12:00:00" in project
    own = "journal_read(project_id='p1', source='reflections')"
    assert _readers(project) == [own] and "PROJECT OLD LESSON" in _run(root, own)
    # A root may read another project's store; the global pointer row names that call and it reads.
    other = "journal_read(project_id='p2', source='reflections')"
    assert f"; read: {other} (reader address," in global_row and "OTHER PROJECT LESSON" in _run(root, other)
    # A scope-qualified address grants nothing to an actor lacking explicit root identity.
    scoped = _actor(env, "t-s", project_id="p1")
    _project, global_row = pointers(scoped, {"id": "t-s", "project_id": "p1"})
    assert f"; read: {other} (reader address," in global_row and "journal project scope applies" in global_row
    assert _run(scoped, other).startswith("⚠️ TOOL_FORBIDDEN")
    # A Presence speaker's ceiling grants no journal_read at all, even for its own project.
    speaker = _speaker(env, "t-speaker")
    speaker.project_id = "p1"
    project, _row = pointers(speaker, {"id": "t-speaker", "project_id": "p1", "_presence_turn": True,
                                       "metadata": {"presence": _PRESENCE}})
    assert _readers(project) == [own]
    assert _run(speaker, own).startswith("⚠️ PRESENCE_CAPABILITY_BLOCKED")


def test_unestablished_access_stays_unknown_rather_than_denied(tmp_path, monkeypatch):
    import ouroboros.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
    env, memory = _context_world(tmp_path)
    task = {"id": "t-main", "type": "task", "text": "Work."}
    source = (env.drive_root / "memory" / "deep_review.md").resolve()
    # No tool context (the tool-less update letter): no call is promised and none is denied.
    deep, _log = _readers(_mode_texts(env, memory, task)["max"])
    assert deep == f"{source} (no tool context here, so whether this actor can open it is unknown)"



# --- journal_read: the project's reflection rows, through the existing reader -------------------


def _reflection_rows(root, project_id, count):
    from ouroboros.utils import append_jsonl

    path = root / "projects" / project_id / "logs" / "task_reflections.jsonl"
    for index in range(count):
        append_jsonl(path, {"ts": f"2026-08-21T00:00:{index:02d}Z", "task_id": f"r-{index:02d}",
                            "reflection": f"row-{index:02d} " + "body " * 50})
    return path


def _snapshot(text):
    import re

    match = re.search(r"snapshot='([0-9a-f]{64})'", text)
    assert match is not None, text
    return match.group(1)


def test_journal_read_pages_the_reflection_log_whole_and_keeps_its_default(tmp_path, monkeypatch):
    from ouroboros.tools.project_journal import _journal_read
    from ouroboros.utils import append_jsonl

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    for index in range(3):
        append_jsonl(tmp_path / "projects" / "mine" / "journal.jsonl", {
            "ts": f"2026-08-20T00:00:0{index}Z", "kind": "note", "text": f"journal-only-{index}", "task_id": "j"})
    path = _reflection_rows(tmp_path, "mine", 5)
    with path.open("ab") as handle:
        handle.write(b'{"broken":\n')
    ctx = SimpleNamespace(drive_root=tmp_path / "fork", budget_drive_root=str(tmp_path), project_id="mine")
    default = _journal_read(ctx, limit=2)
    assert "journal-only-2" in default and "row-00" not in default and "reflections" not in default
    first = _journal_read(ctx, limit=2, source="reflections")
    assert first.startswith("## Project reflection log (mine)")
    assert "row-03 " + "body " * 49 in first and "row-04" in first and "row-02" not in first  # whole rows
    assert "unreadable=1" in first and "coverage=partial" in first
    assert "journal_read(project_id='mine', source='reflections', limit=2, offset=2, snapshot=" in first
    second = _journal_read(ctx, limit=2, offset=2, snapshot=_snapshot(first), source="reflections")
    assert "row-01" in second and "row-02" in second
    # A cursor belongs to its stream: a reflections snapshot cannot page the journal, nor the reverse.
    assert _journal_read(ctx, limit=2, offset=2, snapshot=_snapshot(first)).startswith("JOURNAL_READ_SNAPSHOT_CHANGED")
    assert _journal_read(ctx, limit=1, offset=1, snapshot=_snapshot(default), source="reflections").startswith(
        "JOURNAL_READ_SNAPSHOT_CHANGED")
    assert "is not one of" in _journal_read(ctx, source="pattern_register")
    assert _journal_read(SimpleNamespace(drive_root=tmp_path, project_id="empty"), source="reflections") == (
        "(no reflection-log rows for project empty)")


def test_journal_read_reflections_keep_the_existing_project_authority(tmp_path, monkeypatch):
    from ouroboros.tools.project_journal import _journal_read

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    _reflection_rows(tmp_path, "other", 1)
    root = SimpleNamespace(drive_root=tmp_path, project_id="mine", task_metadata={"root_task_id": "root"})
    assert "row-00" in _journal_read(root, "other", source="reflections")
    child = SimpleNamespace(drive_root=tmp_path, project_id="mine",
                            task_metadata={"parent_task_id": "root", "delegation_role": "subagent"})
    assert _journal_read(child, "other", source="reflections").startswith("⚠️ TOOL_FORBIDDEN")
    speaker = SimpleNamespace(drive_root=tmp_path, project_id="mine", task_metadata={"presence": _PRESENCE})
    assert _journal_read(speaker, "other", source="reflections").startswith("⚠️ TOOL_FORBIDDEN")
    unscoped = SimpleNamespace(drive_root=tmp_path, project_id="")
    assert _journal_read(unscoped, "other", source="reflections").startswith("⚠️ TOOL_FORBIDDEN")

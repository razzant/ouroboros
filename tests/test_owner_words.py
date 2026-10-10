"""The owner's words that caused a work tree (``ouroboros/owner_words.py``).

Every rule is checked in both directions: the rule fires on its case and stays quiet
on the neighbouring one, so deleting the rule turns a test red.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

from ouroboros import owner_words as ow
from ouroboros.loop_messages import _initialize_owner_directives
from ouroboros.owner_mailbox import KIND_OWNER_TEXT, KIND_TASK_MESSAGE, write_owner_message
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.tools.tool_context import ToolContext

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT = "root0001"
TS = "2026-09-30T05:41:45+00:00"


def _ctx(tmp_path, *, directives=None, metadata=None, task_id=ROOT, task_type="task"):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task_id,
                      task_metadata=dict(metadata or {}), current_task_type=task_type)
    if directives is not None:
        ctx._owner_directives = [dict(row) for row in directives]
    return ctx


def _append(path: pathlib.Path, *rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _in_row(text, client_id, *, ts=TS, chat_id=1, **extra):
    return {"ts": ts, "direction": "in", "chat_id": chat_id, "text": text, "client_message_id": client_id,
            "task_id": "", **extra}


def _chat(tmp_path, *rows):
    _append(tmp_path / "logs" / "chat.jsonl", *rows)


def _annotate(tmp_path, client_id, *, action="promote_chat_to_task", status="scheduled", target=ROOT, token="t1"):
    _append(tmp_path / "logs" / "chat_annotations.jsonl", {
        "ts": TS, "type": "chat_annotation", "client_message_id": client_id, "action": action,
        "target": target, "status": status, "routing_token": token})


def _bind(tmp_path, task_id=ROOT, **origin):
    path = tmp_path / "state" / "project_task_bindings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"bindings": {}}
    data["bindings"][task_id] = {"task_id": task_id, "project_id": "p", "project_chat_id": -7, **origin}
    path.write_text(json.dumps(data), encoding="utf-8")


def _texts(rows):
    return [row["text"] for row in rows]


# --- carried by value ---------------------------------------------------------------------------------------

def test_root_corpus_carries_owner_rows_and_never_initial_text(tmp_path):
    corpus = [{"source": "initial_user", "content": "Build the report"},
              {"source": "owner_mailbox", "content": "Add charts", "msg_id": "m-2"},
              {"source": "initial_text", "content": "A wake or a work order"}]
    words = ow.governing_words_for_schedule(_ctx(tmp_path, directives=corpus))
    assert _texts(words[ow.FIELD]) == ["Build the report", "Add charts"]
    assert {row["carrier"] for row in words[ow.FIELD]} == {"ctx"}
    assert words[ow.FIELD][1]["ref"] == "msg m-2"
    # The other side: a corpus with only the non-owner first turn carries nothing of it.
    only_text = ow.governing_words_for_schedule(_ctx(tmp_path, directives=corpus[2:]))
    assert ow.FIELD not in only_text and only_text[ow.ABSENT_FIELD] == "task_type=task"


def test_child_passes_on_exactly_its_inherited_words(tmp_path):
    inherited = [{"text": "Root words", "source": "initial_user", "carrier": "ctx", "task_id": ROOT,
                  "ts": TS, "ref": "chat 1 / msg-1"}]
    child_meta = {"delegation_role": "subagent", "root_task_id": ROOT, ow.FIELD: inherited}
    child = _ctx(tmp_path, task_id="child01", metadata=child_meta,
                 directives=[{"source": "initial_text", "content": "Parent's assignment"}])
    assert ow.governing_words_for_schedule(child) == {ow.FIELD: inherited}
    absent_child = _ctx(tmp_path, task_id="child02", metadata={"delegation_role": "subagent",
                                                               ow.ABSENT_FIELD: "initiator=consciousness"})
    assert ow.governing_words_for_schedule(absent_child) == {ow.ABSENT_FIELD: "initiator=consciousness"}
    # A root holding the same metadata key rebuilds from its own corpus instead.
    root = _ctx(tmp_path, metadata={ow.FIELD: inherited}, directives=[{"source": "initial_user", "content": "Mine"}])
    assert _texts(ow.governing_words_for_schedule(root)[ow.FIELD]) == ["Mine"]


def test_absence_carries_the_raw_run_origin_marker(tmp_path):
    conscious = _ctx(tmp_path, metadata={"initiator": "consciousness"},
                     directives=[{"source": "initial_text", "content": "wake"}])
    assert ow.governing_words_for_schedule(conscious) == {ow.ABSENT_FIELD: "initiator=consciousness"}
    bare = _ctx(tmp_path, task_type="", directives=[])
    assert ow.governing_words_for_schedule(bare) == {ow.ABSENT_FIELD: ow.NOT_RECORDED}
    spoken = _ctx(tmp_path, metadata={"initiator": "consciousness"},
                  directives=[{"source": "owner_mailbox", "content": "Owner wrote in"}])
    assert _texts(ow.governing_words_for_schedule(spoken)[ow.FIELD]) == ["Owner wrote in"]


def test_main_turn_origin_is_carried_byte_for_byte(tmp_path):
    message = "  Сделай отчёт:\n\tцифры за неделю  \n"
    ref = build_owner_message_ref(chat_id=1, client_message_id="msg-1790746905659-12", ts=TS, text=message)
    ctx = _ctx(tmp_path, metadata={"origin_message_ref": ref, "origin_message_text": message})
    _initialize_owner_directives(ctx, [{"role": "user", "content": message}])
    rows = ow.governing_words_for_schedule(ctx)[ow.FIELD]
    assert rows == [{"text": message, "source": "initial_user", "carrier": "ctx", "task_id": ROOT, "ts": TS,
                     "ref": "chat 1 / msg-1790746905659-12"}]
    unstamped = _ctx(tmp_path)
    _initialize_owner_directives(unstamped, [{"role": "user", "content": message}])
    assert ow.FIELD not in ow.governing_words_for_schedule(unstamped)


def test_project_first_message_is_never_added_to_the_words(tmp_path):
    project_words = "Start the Quotas project"
    _bind(tmp_path, source_text=project_words)
    in_corpus = _ctx(tmp_path, metadata={"project_id": "p"},
                     directives=[{"source": "initial_user", "content": project_words}])
    assert _texts(ow.governing_words_for_schedule(in_corpus)[ow.FIELD]) == [project_words]
    elsewhere = _ctx(tmp_path, metadata={"project_id": "p"},
                     directives=[{"source": "initial_user", "content": "Fix the quota bar"}])
    assert _texts(ow.governing_words_for_schedule(elsewhere)[ow.FIELD]) == ["Fix the quota bar"]


def test_schedule_never_reads_the_chat_chain(tmp_path, monkeypatch):
    import ouroboros.chat_chain as chat_chain
    import ouroboros.project_dialogue as project_dialogue

    def refuse(*_args, **_kwargs):
        raise AssertionError("the chat chain was read")

    for module, name in ((chat_chain, "chat_chain_paths"), (chat_chain, "_ordered_chat_generation_paths"),
                         (project_dialogue, "resolve_owner_message_source")):
        monkeypatch.setattr(module, name, refuse)
    ref = build_owner_message_ref(chat_id=1, client_message_id="msg-9", ts=TS, text="Go")
    ctx = _ctx(tmp_path, metadata={"origin_message_ref": ref},
               directives=[{"source": "initial_user", "content": "Go"}])
    assert _texts(ow.governing_words_for_schedule(ctx)[ow.FIELD]) == ["Go"]
    # The substitution is live: the history resolver does read the chain, and trips it.
    _annotate(tmp_path, "msg-9")
    with pytest.raises(AssertionError, match="chat chain was read"):
        ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})


def test_task_governing_words_reads_both_payload_shapes(tmp_path):
    rows = [{"text": "Words", "source": "initial_user", "carrier": "ctx", "task_id": ROOT, "ts": "", "ref": ""}]
    assert ow.task_governing_words({ow.FIELD: rows}) == (rows, "")
    assert ow.task_governing_words({"metadata": {ow.FIELD: rows}}) == (rows, "")
    assert ow.task_governing_words({"metadata": {ow.ABSENT_FIELD: "task_type=presence"}}) == ([], "task_type=presence")
    assert ow.task_governing_words({"metadata": {}}) == ([], "")


def test_dedup_by_normalized_text_checksum(tmp_path):
    corpus = [{"source": "initial_user", "content": "Do  the\nthing"},
              {"source": "owner_mailbox", "content": "Do the thing "},
              {"source": "owner_mailbox", "content": "Do the other thing"}]
    rows = ow.governing_words_for_schedule(_ctx(tmp_path, directives=corpus))[ow.FIELD]
    assert _texts(rows) == ["Do  the\nthing", "Do the other thing"]


# --- history carriers, in order -----------------------------------------------------------------------------

def _all_carriers(tmp_path):
    origin_ref = build_owner_message_ref(chat_id=1, client_message_id="msg-o", ts=TS, text="origin text")
    task = {"id": ROOT, "origin_message_ref": origin_ref, "origin_message_text": "origin text"}
    _bind(tmp_path, source_text="binding text",
          source_ref=build_owner_message_ref(chat_id=1, client_message_id="msg-b", ts=TS, text="binding text"))
    _chat(tmp_path, _in_row("annotated text", "msg-a"))
    _annotate(tmp_path, "msg-a")
    write_owner_message(tmp_path, "mailbox text", ROOT, msg_id="mb-1", kind=KIND_OWNER_TEXT)
    return task


def test_history_carriers_fall_through_in_order(tmp_path):
    task = _all_carriers(tmp_path)
    rows, absent = ow.task_owner_words(tmp_path, ROOT, task=task)
    assert (_texts(rows), rows[0]["carrier"], absent) == (["origin text"], "origin_message", "")
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})
    assert (_texts(rows), rows[0]["carrier"]) == (["binding text"], "binding")
    (tmp_path / "state" / "project_task_bindings.json").unlink()
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})
    assert (_texts(rows), rows[0]["carrier"], rows[0]["source"]) == (["annotated text"], "annotation",
                                                                     "promote_chat_to_task")
    (tmp_path / "logs" / "chat_annotations.jsonl").unlink()
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})
    assert (_texts(rows), rows[0]["carrier"], rows[0]["ref"]) == (["mailbox text"], "mailbox", "msg mb-1")
    for path in (tmp_path / "memory" / "owner_mailbox").iterdir():
        path.unlink()
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT}) == ([], ow.NOT_RECORDED)
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT, "metadata": {"initiator": "consciousness"}}) == (
        [], "initiator=consciousness")


def test_origin_and_binding_refs_resolve_through_the_chain(tmp_path):
    message = "Ref-only origin"
    ref = build_owner_message_ref(chat_id=1, client_message_id="msg-r", ts=TS, text=message)
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT, "origin_message_ref": ref}) == ([], ow.NOT_RECORDED)
    _chat(tmp_path, _in_row(message, "msg-r"))
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT, "metadata": {"origin_message_ref": ref}})
    assert rows == [{"text": message, "source": "origin_message", "carrier": "origin_message", "task_id": ROOT,
                     "ts": TS, "ref": "chat 1 / msg-r"}]
    _bind(tmp_path, source_ref=ref)
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})[0][0]["carrier"] == "binding"


def test_manual_target_annotation_is_not_a_carrier(tmp_path):
    _chat(tmp_path, _in_row("Which task?", "msg-m"), _in_row("Steer this", "msg-s"))
    _annotate(tmp_path, "msg-m", action="steer_task", status="needs_manual_target")
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT}) == ([], ow.NOT_RECORDED)
    _annotate(tmp_path, "msg-s", action="steer_task", status="delivered", token="t2")
    assert _texts(ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})[0]) == ["Steer this"]


def test_annotation_counts_only_through_an_inbound_row(tmp_path):
    agent_id = "agent-steer:tok-1"
    _annotate(tmp_path, agent_id, action="steer_task", status="delivered")
    _chat(tmp_path, {**_in_row("Ouroboros relayed this", agent_id), "direction": "out"})
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT}) == ([], ow.NOT_RECORDED)
    _chat(tmp_path, _in_row("The owner's own steer", agent_id))
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})
    assert (_texts(rows), rows[0]["source"]) == (["The owner's own steer"], "steer_task")


@pytest.mark.parametrize("injection", [{"system_type": "proactive_message"}, {"presence": {"room": "x"}},
                                       {"source": "skill_repair"}])
def test_non_owner_inbound_rows_are_cut(tmp_path, injection):
    _chat(tmp_path, _in_row("Injected", "msg-i", **injection))
    _annotate(tmp_path, "msg-i")
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT}) == ([], ow.NOT_RECORDED)
    _chat(tmp_path, _in_row("Plain owner row", "msg-p"))
    _annotate(tmp_path, "msg-p", token="t2")
    assert _texts(ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})[0]) == ["Plain owner row"]


def test_mailbox_carrier_takes_owner_text_only(tmp_path):
    write_owner_message(tmp_path, "a sibling's message", ROOT, msg_id="tm-1", kind=KIND_TASK_MESSAGE)
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT}) == ([], ow.NOT_RECORDED)
    write_owner_message(tmp_path, "owner follow-up", ROOT, msg_id="mb-2", kind=KIND_OWNER_TEXT)
    assert _texts(ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})[0]) == ["owner follow-up"]


def test_child_resolves_its_root_unless_it_carries_the_field(tmp_path):
    _bind(tmp_path, source_text="root's origin")
    child = {"id": "child01", "root_task_id": ROOT, "delegation_role": "subagent"}
    rows, _ = ow.task_owner_words(tmp_path, "child01", task=child)
    assert (_texts(rows), rows[0]["task_id"]) == (["root's origin"], ROOT)
    carried = [{"text": "carried", "source": "initial_user", "carrier": "ctx", "task_id": ROOT, "ts": "", "ref": ""}]
    assert ow.task_owner_words(tmp_path, "child01", task={**child, "metadata": {ow.FIELD: carried}}) == (carried, "")


# --- rendering ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("audience", ["child", "session", "reviewer", "plan", "writer"])
def test_render_is_verbatim_and_whole_for_every_audience(audience):
    long_text = ("Слово владельца. " * 700)[:10_250]
    assert len(long_text) == 10_250
    row = {"text": long_text, "source": "initial_user", "carrier": "ctx", "task_id": "9e0a326d", "ts": TS,
           "ref": "chat 1 / msg-1790746905659-12"}
    text = ow.render_owner_words([row], audience=audience, root_task_id="9e0a326d")
    assert long_text in text and text.endswith(long_text)
    assert f"[{TS} · owner · initial_user of task 9e0a326d · chat 1 / msg-1790746905659-12]\n" in text
    assert text.count(long_text) == 1


def test_render_absence_is_one_line_and_unknown_audience_is_refused():
    line = ow.render_owner_words([], "initiator=consciousness", audience="child")
    assert line == "No words of my human are recorded for this work (host marker: initiator=consciousness)."
    assert "\n" not in line
    assert ow.render_owner_words([], "", audience="session") == ""
    with pytest.raises(ValueError):
        ow.render_owner_words([], "x", audience="everyone")


def test_owner_words_text_renders_the_scheduled_words(tmp_path):
    ctx = _ctx(tmp_path, metadata={"root_task_id": ROOT}, directives=[{"source": "initial_user", "content": "Ask"}])
    text = ow.owner_words_text(ctx)
    assert text.startswith("## Words of my human that caused this work (verbatim, host-attested)\n") and "\nAsk" in text
    assert ow.owner_words_text(_ctx(tmp_path, metadata={"initiator": "consciousness"}, directives=[])) == (
        "No words of my human are recorded for this work (host marker: initiator=consciousness).")


# --- no model ----------------------------------------------------------------------------------------------

def _llm_imports(source: str) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(str(node.module or ""))
    return sorted(name for name in names if name.startswith("ouroboros.llm"))


def test_module_imports_no_model_client():
    source = (REPO / "ouroboros" / "owner_words.py").read_text(encoding="utf-8")
    assert _llm_imports(source) == []
    assert _llm_imports("def f():\n    from ouroboros.llm import LLMClient\nimport ouroboros.llm_openai_compatible\n") == [
        "ouroboros.llm", "ouroboros.llm_openai_compatible"]

"""A fresh head that an accepted late quiz answer started reads its asker's kept notes (#2328).

Owner answer A: when the owner answers after the asking task ended, the head the answer starts
sees, right after its assignment, the asker's last authored working notes
(``main_context_authority.WORKING_NOTES_FIELD``: the actor capsules its latest applied fold
kept) as ONE foldable row: dated, source-labelled, possibly intermediate, beside the reader of
the asker's final result. The row is information, never a predecessor: no authority, plan,
review or finish passes. The owner's row stays their words; a live head reading the asker's
thread receives the next answer as mail alone; a task without typed late-answer provenance or
with declared inputs receives no row. The Continue reader's own source selection is unchanged.
"""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest

from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.owner_quiz import quiz_states, reconcile_terminal, record_asked
from tests.test_context_renewal import NOTE, _noted
from tests.test_quiz_answer import _decision_app, _post
from tests.test_quiz_late_answer import LATE_HEAD, _accepted_rows, _intake, _late_answer_head, _late_bridge

ASKER = "task-1"
LATE_ID = f"quiz_late_answer:{ASKER}:q1"
FINAL = "Shipped on sqlite; postgres stays open."
READER = f"get_task_result(task_id={ASKER!r}, include_authority=True)"

pytestmark = pytest.mark.usefixtures("provider_catalog_offline")


def _world(tmp_path):
    """The context-test env and memory; its drive is the canonical root the whole story lives on."""
    from tests.test_doc_context import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    return env, memory, env.drive_root


def _asker_folds_and_ends(root, *, execution_root=None, chat_id=1):
    """The asker asks q1, folds its work with one authored note (the real producer), then ends."""
    from ouroboros.main_context_authority import publish_working_notes
    from ouroboros.task_results import write_task_result

    write_task_result(root, ASKER, "running", result="", chat_id=chat_id)
    record_asked(root, ASKER, quiz_id="q1", question="Which db for the pilot?", options=["sqlite", "postgres"],
                 assumption="sqlite meanwhile", chat_id=chat_id)
    ctx = SimpleNamespace(task_id=ASKER, drive_root=execution_root or root, budget_drive_root=root)
    ref = publish_working_notes(ctx, _noted(root))
    assert ref is not None
    write_task_result(root, ASKER, "completed", result=FINAL, _terminal_observed=True)
    reconcile_terminal(root, ASKER)
    return ref


def _answer_late(root, monkeypatch):
    """The owner presses option 2 on q1 after the asker ended; the answer waits at the bridge intake."""
    import supervisor.message_bus as mb

    bridge, _frames = _late_bridge(root, monkeypatch)
    monkeypatch.setattr(mb, "load_state", lambda: {"session_id": "s-1"})
    response = _post(_decision_app(root, monkeypatch, live_task=None),
                     {"request_id": "r1", "decision_id": f"quiz:{ASKER}:q1", "option_index": 1})
    assert response.status_code == 200 and response.json()["forwarded"] is True, response.text
    return bridge


def _fresh_head(root, monkeypatch, bridge, *, running=None):
    """The real intake starts ONE fresh head; the task its real admission hands the agent."""
    import queue as queue_mod

    import supervisor.workers as workers

    [(args, kwargs)] = _intake(root, monkeypatch, bridge, running=running)
    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    monkeypatch.setattr(workers, "get_event_q", lambda: queue_mod.Queue())

    class _Head:
        task = None

        def handle_task(self, task):
            self.task = task
            return []

    head = _Head()
    workers._run_chat_task(head, args[0], args[1], None, task_constraint=kwargs.get("task_constraint"),
                           task_metadata=kwargs["task_metadata"])
    assert head.task is not None
    return head.task


def _first_request(env, memory, task):
    """The head's first model request, as ``agent.py`` builds it."""
    from ouroboros.context import build_llm_messages

    messages, _ = build_llm_messages(env=env, memory=memory, task=task)
    return messages


def _with_note(messages):
    return [row for row in messages if NOTE in json.dumps(row.get("content"), ensure_ascii=False)]


def _assert_one_asker_row(messages, root, ref):
    """Exactly one notes row, the first after the assignment, whole, dated and source-labelled."""
    rows = _with_note(messages)
    assert len(rows) == 1, "the asker's note reaches the fresh head exactly once"
    from ouroboros.main_context_authority import LATE_ANSWER_NOTES_NAME, WORKING_NOTES_AT_FIELD
    from ouroboros.task_results import load_task_result

    [row] = rows
    assert messages.index(row) == 2, "the first row after the assignment, never the frozen prefix"
    assert set(row) == {"role", "content", HOST_CONTEXT_KIND_KEY} and row["role"] == "user"
    assert row[HOST_CONTEXT_KIND_KEY] == LATE_ANSWER_NOTES_NAME
    text = row["content"]
    assert text.startswith(f"[ASKER_NOTES] Task {ASKER} asked the question this late answer answers.")
    assert text.endswith(NOTE) and text.count(NOTE) == 1
    saved = load_task_result(root, ASKER)
    assert f"named in its result at {saved[WORKING_NOTES_AT_FIELD]}" in text
    assert f"it ended at {saved['terminal_time']['occurred_at']}" in text
    assert "may be intermediate" in text and f"get_task_result(task_id={ASKER!r}) reads that result" in text
    assert "not an owner instruction and not this task's plan, finish or verdict" in text
    assert f"Exact source: {READER} -> working_notes" in text
    assert json.dumps(ref, ensure_ascii=False, sort_keys=True) in text
    assert FINAL not in text, "the final result stays behind its reader"
    assert NOTE not in json.dumps(messages[0], ensure_ascii=False) + json.dumps(messages[1], ensure_ascii=False)
    return row


def _assert_no_inherited_authority(env, task):
    from ouroboros.context import _task_authority_projection

    metadata = task.get("metadata") or {}
    assert "predecessor_authority" not in task and "predecessor_authority_source" not in task
    assert not (task.get("task_contract") or {}).get("predecessor_authority")
    assert "continuation" not in metadata and "objective_author" not in metadata
    projection = _task_authority_projection(env, task)
    assert "predecessor_authority" not in projection and "plan_review_authority" not in projection


def test_the_fresh_head_of_a_late_answer_starts_with_its_askers_notes_once(tmp_path, monkeypatch):
    env, memory, root = _world(tmp_path)
    ref = _asker_folds_and_ends(root)
    task = _fresh_head(root, monkeypatch, _answer_late(root, monkeypatch))
    messages = _first_request(env, memory, task)

    assert task["text"].startswith(LATE_HEAD.format(ASKER) + "\n[Owner quiz answer] quiz q1")
    assert NOTE not in task["text"], "the shared late-answer frame stays the card and the answer"
    _assert_one_asker_row(messages, root, ref)
    _assert_no_inherited_authority(env, task)
    [owner_row] = _accepted_rows(root, LATE_ID)
    assert owner_row["text"] == "2. postgres", "the owner's canonical row stays their own words"


def test_the_next_answer_to_a_live_head_of_the_same_asker_arrives_as_mail_without_the_notes(tmp_path, monkeypatch):
    """Repeated same-asker answers: the head that already holds the notes reads the next answer
    through its mailbox; the live delivery is the unchanged frame, never a second notes copy."""
    import queue as queue_mod

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import drain_owner_entries

    _env, _memory, root = _world(tmp_path)
    _asker_folds_and_ends(root)
    bridge = _answer_late(root, monkeypatch)
    _late_answer_head("head-1", ASKER, 1)
    assert _intake(root, monkeypatch, bridge) == []
    [entry] = drain_owner_entries(root, "head-1", set())
    assert (entry["late_answer"]["task_id"], entry["text"]) == (ASKER, "2. postgres")
    owner_ctx = SimpleNamespace()
    messages = [{"role": "user", "content": "the first late answer"}]
    _drain_incoming_messages(messages, queue_mod.Queue(), root, "head-1", None, set(), owner_ctx=owner_ctx)
    delivered = json.dumps(messages[1:], ensure_ascii=False)
    assert LATE_HEAD.format(ASKER) in delivered and "The owner chose option 2: postgres" in delivered
    assert NOTE not in delivered and "[ASKER_NOTES]" not in delivered


def test_a_head_that_stopped_reading_is_replaced_by_a_fresh_head_with_the_notes(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries

    env, memory, root = _world(tmp_path)
    ref = _asker_folds_and_ends(root)
    bridge = _answer_late(root, monkeypatch)
    _late_answer_head("head-1", ASKER, 1, accepting=False)
    task = _fresh_head(root, monkeypatch, bridge)
    assert drain_owner_entries(root, "head-1", set()) == []
    _assert_one_asker_row(_first_request(env, memory, task), root, ref)


def test_the_notes_survive_quiz_gc_and_the_askers_fork_cleanup_for_a_forked_head(tmp_path, monkeypatch):
    """The accepted snapshot frames the answer after its card left the quiz projection, and a head
    running on its own execution drive reads the asker's notes from the canonical root after the
    asker's execution drive is gone."""
    from ouroboros.owner_quiz import mark_discharged
    from tests.test_doc_context import _make_env_and_memory

    _env, _memory, root = _world(tmp_path / "canonical")
    asker_fork = tmp_path / "asker-fork"
    asker_fork.mkdir()
    ref = _asker_folds_and_ends(root, execution_root=asker_fork)
    bridge = _answer_late(root, monkeypatch)
    mark_discharged(root, ASKER, "q1")
    for index in range(16):  # the real projection cap evicts the settled card
        record_asked(root, ASKER, quiz_id=f"later-{index}", question="Later?", options=["a", "b"])
    assert "q1" not in quiz_states(root, ASKER)
    shutil.rmtree(asker_fork)
    task = {**_fresh_head(root, monkeypatch, bridge), "budget_drive_root": str(root)}
    head_env, head_memory = _make_env_and_memory(tmp_path / "head-fork")
    messages = _first_request(head_env, head_memory, task)

    assert "Question was: Which db for the pilot?" in json.dumps(messages[1], ensure_ascii=False)
    _assert_one_asker_row(messages, root, ref)


def _late_head(**extra):
    return {"id": "fresh-head", "type": "task", "text": "the late answer frame",
            "metadata": {"late_answer": {"task_id": ASKER, "quiz_id": "q1"}}, **extra}


def _continue_of_asker():
    from tests.test_continuation_context_authority import _pull_source

    return {"id": "next", "text": "continue", "predecessor_authority": {"task_id": ASKER, "source": _pull_source(ASKER)}}


@pytest.mark.parametrize("forked", [False, True])
@pytest.mark.parametrize("kind", ["late_answer", "continue"])
def test_the_displayed_notes_reader_resolves_for_the_receiving_task(tmp_path, monkeypatch, forked, kind):
    """Execute the emitted reader, not the asker's unqualified artifact_store recipe."""
    from ouroboros.main_context_authority import late_answer_notes_message, predecessor_notes_message, WORKING_NOTES_FIELD
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    env, _memory, root = _world(tmp_path / "canonical")
    ref = _asker_folds_and_ends(root)
    drive = root / "state" / "headless_tasks" / "receiving-head" / "data" if forked else root
    drive.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=drive, budget_drive_root=str(root), task_id="receiving-head",
                      task_metadata={"delegation_role": "root", "budget_drive_root": str(root)})
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    row = (late_answer_notes_message(_late_head(), drive_root=root) if kind == "late_answer"
           else predecessor_notes_message(_continue_of_asker(), drive_root=root))
    tail = row["content"].split("read from this task with ", 1)[1]
    read, _ = json.JSONDecoder().raw_decode(tail)
    result = registry.execute_result(read["tool"], read["arguments"])
    assert result.status == "ok" and NOTE in result.text, result.text
    assert load_task_result(root, ASKER)[WORKING_NOTES_FIELD] == ref, "display must not rewrite the retained handle"


@pytest.mark.parametrize("field", ["retired", "absent"])
def test_a_cleared_field_stays_cleared_while_a_record_before_the_field_reads_its_legacy_account(tmp_path, field):
    """The asker's latest fold that kept no notes clears the field: no older account returns for the
    late head. A record written before the field serves its stored legacy account, undated. Continue's
    reader keeps its existing reading of the same record."""
    from ouroboros.main_context_authority import (
        LATE_ANSWER_NOTES_NAME, WORKING_NOTES_FIELD, late_answer_notes_message, predecessor_notes_message,
        publish_working_notes,
    )
    from ouroboros.task_results import load_task_result, write_task_result
    from tests.test_continuation_context_authority import _narrative

    legacy = "The account an earlier version stored beside the result."
    write_task_result(tmp_path, ASKER, "completed", result=FINAL, continuation_narrative=_narrative(ASKER, legacy))
    if field == "retired":
        folded, ctx = _noted(tmp_path), SimpleNamespace(task_id=ASKER, drive_root=tmp_path)
        assert publish_working_notes(ctx, folded) is not None
        assert publish_working_notes(ctx, [row for row in folded if not isinstance(row.get("content"), list)]) is None
        assert load_task_result(tmp_path, ASKER)[WORKING_NOTES_FIELD] is None
    else:
        assert WORKING_NOTES_FIELD not in load_task_result(tmp_path, ASKER)
    note = late_answer_notes_message(_late_head(), drive_root=tmp_path)
    if field == "retired":
        assert note is None
    else:
        assert note[HOST_CONTEXT_KIND_KEY] == LATE_ANSWER_NOTES_NAME and note["content"].endswith(legacy)
        assert f"Exact source: {READER} -> continuation_narrative" in note["content"]
        assert "the time they were named in its result was not recorded" in note["content"]
    assert predecessor_notes_message(_continue_of_asker(), drive_root=tmp_path)["content"].endswith(legacy)


@pytest.mark.parametrize("damage", ["source_missing", "source_altered", "field_malformed", "result_unreadable"])
def test_an_unreadable_asker_source_is_a_typed_gap_row_and_nothing_is_inferred(tmp_path, damage):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.main_context_authority import (
        LATE_ANSWER_NOTES_NAME, WORKING_NOTES_FIELD, late_answer_notes_message, publish_working_notes,
    )
    from ouroboros.task_results import write_task_result
    from ouroboros.task_store import patch_document

    write_task_result(tmp_path, ASKER, "running", result="")
    ref = publish_working_notes(SimpleNamespace(task_id=ASKER, drive_root=tmp_path), _noted(tmp_path))
    source = task_artifact_dir_path(tmp_path, ASKER, create=False) / ref["path"]
    if damage == "source_missing":
        source.unlink()
    elif damage == "source_altered":
        source.write_bytes(json.dumps({"task_id": ASKER, "notes": ["forged"]}).encode("utf-8"))
    elif damage == "field_malformed":
        write_task_result(tmp_path, ASKER, "running", strict_existing_dict=True, _field_projector=lambda existing, _new: {
            "status": existing.get("status"), WORKING_NOTES_FIELD: ref["path"]})
    else:
        patch_document(tmp_path, ASKER, lambda current: {**current, "status": ""})
    note = late_answer_notes_message(_late_head(), drive_root=tmp_path)
    assert set(note) == {"role", "content", HOST_CONTEXT_KIND_KEY}
    assert note[HOST_CONTEXT_KIND_KEY] == LATE_ANSWER_NOTES_NAME
    text = note["content"]
    assert text.startswith(f"[ASKER_NOTES] Task {ASKER} asked the question this late answer answers")
    assert NOTE not in text and "forged" not in text and "nothing was inferred" in text and READER in text
    if damage == "result_unreadable":
        assert "whether it kept working notes is unknown" in text
    else:
        assert "their retained source could not be read now" in text


def test_only_typed_late_answer_provenance_selects_an_asker(tmp_path):
    """No row from the text, from a ref missing its quiz, for the asker itself, or twice when the
    asker is also the task's materialized predecessor (its notes already arrive as that row)."""
    from ouroboros.main_context_authority import late_answer_notes_message, predecessor_notes_message
    from ouroboros.task_results import write_task_result

    _env, _memory, root = _world(tmp_path)
    _asker_folds_and_ends(root)
    assert NOTE in late_answer_notes_message(_late_head(), drive_root=root)["content"]
    framed = {"id": "fresh-head", "text": LATE_HEAD.format(ASKER) + "\n[Owner quiz answer] quiz q1"}
    assert late_answer_notes_message(framed, drive_root=root) is None
    assert late_answer_notes_message(_late_head(metadata={"late_answer": {"task_id": ASKER}}), drive_root=root) is None
    assert late_answer_notes_message(_late_head(id=ASKER), drive_root=root) is None
    write_task_result(root, "next", "running", result="")
    both = {**_continue_of_asker(), "metadata": _late_head()["metadata"]}
    assert late_answer_notes_message(both, drive_root=root) is None
    assert NOTE in predecessor_notes_message(both, drive_root=root)["content"]


@pytest.mark.parametrize("selection", ["shared", "declared"])
def test_a_declared_input_late_head_receives_no_asker_notes(tmp_path, selection):
    from ouroboros.context import build_runtime_section

    env, _memory, root = _world(tmp_path)
    _asker_folds_and_ends(root)
    rows = []
    task = _late_head(**({"input_sources": "declared"} if selection == "declared" else {}))
    build_runtime_section(env, task, supplementary_messages_out=rows)
    assert len(_with_note(rows)) == (selection == "shared")


def test_the_producer_dates_the_notes_it_names_and_keeps_that_date_for_the_same_notes(tmp_path):
    from datetime import datetime, timezone

    from ouroboros.main_context_authority import WORKING_NOTES_AT_FIELD, WORKING_NOTES_FIELD, publish_working_notes
    from ouroboros.task_results import load_task_result, write_task_result

    write_task_result(tmp_path, ASKER, "running", result="")
    folded, ctx = _noted(tmp_path), SimpleNamespace(task_id=ASKER, drive_root=tmp_path)
    before = datetime.now(timezone.utc)
    ref = publish_working_notes(ctx, folded)
    named = load_task_result(tmp_path, ASKER)
    assert named[WORKING_NOTES_FIELD] == ref
    assert before <= datetime.fromisoformat(named[WORKING_NOTES_AT_FIELD]) <= datetime.now(timezone.utc)
    assert publish_working_notes(ctx, folded) == ref
    assert load_task_result(tmp_path, ASKER)[WORKING_NOTES_AT_FIELD] == named[WORKING_NOTES_AT_FIELD]

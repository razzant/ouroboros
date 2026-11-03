"""Retained Presence observations use room facts without creating a new terminal or speaker."""

from __future__ import annotations

from tests import _dialogue as dialogue_test

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from ouroboros import consciousness_wake as wake
from ouroboros.dialogue_provenance import render_row_text, row_author, row_class
from ouroboros.presence_recovery import record_unresolved_turn, retained_turn_facts
from ouroboros.presence_runner import _build_task, _notify_unresolved_turn
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_presence_runner import _admission, _event


def _stored(root, status="failed", *, unknown=True, child="", source="Please retain this question."):
    task = _build_task(_admission(), replace(_event(), text=source), drive_root=root, staged_files=())
    metadata = copy.deepcopy(task["metadata"])
    metadata.update(presence_outcome="silent", presence_result_text="")
    if unknown:
        metadata["presence_unknown_outcome"] = {
            "source": "provider_outcome_unknown_no_resend", "error_kind": "provider_outcome_unknown",
            "operation_id": "operation-original",
        }
    if child:
        metadata["presence_work_ref"] = child
    write_task_result(root, task["id"], status, metadata=metadata, chat_id=task["chat_id"],
                      _is_direct_chat=True, source=task["source"],
                      reason_code="provider_unavailable" if unknown else "",
                      terminal_origin="host_notice" if status == "failed" else "")
    return task, load_task_result(root, task["id"])


def _rows(root):
    path = root / "logs" / "chat.jsonl"
    return dialogue_test.log_rows(path)


@pytest.mark.parametrize("carrier", ["metadata", "continuation", "initial", "absent"])
def test_projection_keeps_only_a_real_child_and_no_authored_answer(tmp_path, carrier):
    task, stored = _stored(tmp_path)
    if carrier == "metadata":
        stored["metadata"]["presence_work_ref"] = "actual-child"
    elif carrier == "continuation":
        stored["presence_continuation"] = {"child_work_ref": "actual-child"}
    elif carrier == "initial":
        stored["presence_continuation"] = {"initial": {"work_ref": "actual-child"}}
    before = copy.deepcopy(stored)
    facts = retained_turn_facts(stored, task["id"])
    assert facts["turn_ref"] == task["id"]
    assert facts.get("work_ref") == (None if carrier == "absent" else "actual-child")
    assert facts["uncertainty"]["operation_id"] == "operation-original"
    assert facts["source_event_id"] == _event().source_event_id
    assert facts["recorded_status"] == "failed" and facts["code"] == "presence_attempt_outcome_unknown"
    assert not {"status", "outcome", "text", "ok"} & facts.keys()
    assert stored == before


def test_child_projection_preserves_recorded_precedence(tmp_path):
    task, stored = _stored(tmp_path, child="metadata-child")
    stored["presence_continuation"] = {"child_work_ref": "later-child", "initial": {"work_ref": "initial-child"}}
    assert retained_turn_facts(stored, task["id"])["work_ref"] == "metadata-child"


def test_lost_running_observation_preserves_status_and_reaches_the_regular_wake(tmp_path):
    task, before = _stored(tmp_path, "running", unknown=False)
    assert record_unresolved_turn(tmp_path, task["id"])
    stored = load_task_result(tmp_path, task["id"])
    assert stored["status"] == "running" and stored["metadata"] == before["metadata"]
    assert wake._transition_facts(stored) == {}
    row, = _rows(tmp_path)
    assert row["summary_kind"] == "host_task_facts" and row["outcome_final"] is False
    assert row["outcome_authority"] == "retained_presence_observation" and row["text"] == ""
    assert row["source_text"] == "Please retain this question."
    assert row_author(row)["kind"] == "host" and row_class(row)["lane"] == 2
    assert wake._row_kind(row) == "presence_observation"
    observation = wake.observe_wake(tmp_path, boundary=None, since=0, now=1_900_000_000)
    retained = [line for kind, _offset, line in observation.events if kind == "presence_observation"]
    assert len(retained) == 1 and "recorded_status running" in retained[0]
    assert "Please retain this question." in retained[0]
    assert "no completed answer or absence of external effects is established" in retained[0]
    assert "nothing sent" not in retained[0] and "get_task_result" not in retained[0]
    assert "task_terminal" not in [kind for kind, _offset, _line in observation.events]


def test_concurrent_runner_replays_record_once_without_an_owner_question(tmp_path):
    task, _ = _stored(tmp_path, child="admitted-child")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _index: _notify_unresolved_turn(tmp_path, task["id"]), range(8)))
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["summary_id"] == f"presence-retained:{task['id']}"
    assert rows[0]["presence_recovery"]["work_ref"] == "admitted-child"
    assert rows[0]["chat_id"] == task["chat_id"]
    assert "Please answer in Main" not in render_row_text(rows[0])
    assert not record_unresolved_turn(tmp_path, task["id"])
    assert len(_rows(tmp_path)) == 1


def test_failed_append_is_not_stamped_and_can_be_retried(tmp_path, monkeypatch):
    from ouroboros import project_dialogue

    task, _ = _stored(tmp_path)
    real_append = project_dialogue.append_canonical_task_summary
    monkeypatch.setattr(project_dialogue, "append_canonical_task_summary", lambda *_args: False)
    assert not record_unresolved_turn(tmp_path, task["id"])
    assert "presence_recovery_recorded" not in load_task_result(tmp_path, task["id"])
    monkeypatch.setattr(project_dialogue, "append_canonical_task_summary", real_append)
    assert record_unresolved_turn(tmp_path, task["id"])
    assert len(_rows(tmp_path)) == 1


def test_append_receipt_does_not_revert_a_late_canonical_terminal(tmp_path, monkeypatch):
    from ouroboros import project_dialogue

    task, _ = _stored(tmp_path, "running", unknown=False)
    real_append = project_dialogue.append_canonical_task_summary

    def late_result(root, row):
        appended = real_append(root, row)
        write_task_result(root, task["id"], "completed", terminal_origin="model_final", result="Late selected answer")
        return appended

    monkeypatch.setattr(project_dialogue, "append_canonical_task_summary", late_result)
    assert record_unresolved_turn(tmp_path, task["id"])
    current = load_task_result(tmp_path, task["id"])
    assert current["status"] == "completed" and current["result"] == "Late selected answer"
    assert current["presence_recovery_recorded"]["summary_id"] == f"presence-retained:{task['id']}"
    row, = _rows(tmp_path)
    assert row["status"] == "running" and row["outcome_final"] is False  # historical observation, not a new terminal


def test_unknown_terminal_wake_does_not_rebrand_its_salvaged_outcome_as_silence(tmp_path):
    from ouroboros.dialogue_provenance import RoomLabelResolver

    task, stored = _stored(tmp_path, child="actual-child")
    line = wake._event_line("task_terminal", {"task_id": task["id"]}, {task["id"]: stored},
                            RoomLabelResolver(projects=[]), {}, 1_900_000_000)
    assert "presence_attempt_outcome_unknown" in line and "operation-original" in line
    assert "actual-child" in line and "Please retain this question." in line
    assert "Presence outcome=silent" not in line and "nothing sent" not in line
    # A genuine authored silent result keeps its normal projection.
    del stored["metadata"]["presence_unknown_outcome"]
    line = wake._event_line("task_terminal", {"task_id": task["id"]}, {task["id"]: stored},
                            RoomLabelResolver(projects=[]), {}, 1_900_000_000)
    assert "Presence outcome=silent" in line


def test_retained_facts_survive_later_room_messages_in_existing_memory_lane(tmp_path):
    from ouroboros import dialogue_store
    from ouroboros.utils import utc_now_iso
    from tests.test_memory_view_room import _install, _view

    _install(tmp_path)
    task, _ = _stored(tmp_path, child="admitted-child", source="A source phrase unavailable to another task reader.")
    assert record_unresolved_turn(tmp_path, task["id"])
    dialogue_store.append(tmp_path, "chat", {
        "ts": utc_now_iso(), "direction": "in", "chat_id": task["chat_id"], "text": "A later independent message.",
        "source": task["source"], "transport": {"provider": "telegram"},
    })
    _snapshot, text = _view(tmp_path, task)
    assert "A source phrase unavailable to another task reader." in text
    assert "presence_attempt_outcome_unknown" in text and "admitted-child" in text
    assert "A later independent message." in text
    assert "get_task_result" not in render_row_text(_rows(tmp_path)[-2])


def test_canonical_terminal_keeps_priority_without_erasing_retained_observation(tmp_path):
    from ouroboros.project_dialogue import append_canonical_task_summary
    from ouroboros.utils import utc_now_iso
    from tests.test_memory_view_room import _install, _view

    _install(tmp_path)
    task, _ = _stored(tmp_path, "running", unknown=False, source="Original room question.")
    assert record_unresolved_turn(tmp_path, task["id"])
    observed = _rows(tmp_path)[-1]
    write_task_result(tmp_path, task["id"], "completed", terminal_origin="model_final", result="Late selected result.")
    append_canonical_task_summary(tmp_path, {
        "ts": utc_now_iso(), "direction": "system", "type": "task_summary", "task_id": task["id"],
        "chat_id": task["chat_id"], "status": "completed", "summary_kind": "terminal_root_projection",
        "summary_id": f"task-terminal:{task['id']}", "text": "Canonical completed result.",
    })
    _snapshot, text = _view(tmp_path, task)
    assert "Canonical completed result." in text and "Original room question." in text
    assert "recorded_status running" in text and f"observation at {observed['ts']}" in text
    assert text.index("Canonical completed result.") < text.index("host observation for Presence turn")
    assert load_task_result(tmp_path, task["id"])["status"] == "completed"


def test_ordinary_host_facts_render_unchanged_and_missing_task_is_not_created(tmp_path):
    row = {"type": "task_summary", "task_id": "ordinary", "status": "running", "text": "",
           "result_ref": {"reader": "get_task_result", "task_id": "ordinary"}}
    assert render_row_text(row) == "host facts for ordinary: status=running; result: get_task_result(task_id=ordinary)"
    assert not record_unresolved_turn(tmp_path, "absent-task")
    assert load_task_result(tmp_path, "absent-task") is None and _rows(tmp_path) == []

"""Main mirrors a Project root's final answer as an ordinary Ouroboros message.

The completion row keeps its plain ``text`` byte-identical (Telegram, the
model's own Main context and older clients read only that) and carries ONE
additive typed key, ``completion_answer``, only for a model-authored final
answer of a settled task. These tests pin the gate in both directions, the
unchanged text, and the key's survival live -> durable row -> history replay.
"""

from __future__ import annotations

from tests import _dialogue as dialogue_test

import asyncio
import json
import types
from types import SimpleNamespace

import pytest

import supervisor.message_bus as message_bus
from ouroboros.projects_registry import MIRRORED_ANSWER_MAX_CHARS, mirrored_answer

ANSWER = "**Done: PR #7 merged.**\n\n## What changed\n\n- one\n- two\n"


def _result(**overrides):
    row = {"result": ANSWER, "terminal_origin": "model_final"}
    row.update(overrides)
    return row


@pytest.mark.parametrize("phase", ["done", "warn", "error", "cancelled"])
def test_model_final_answer_is_mirrored_whole_for_every_terminal_phase(phase):
    # The mind's own words ride the row uncut, whatever word the host folded:
    # a Failed task that explains itself is still Ouroboros speaking.
    assert mirrored_answer(_result(), phase) == {"completion_answer": ANSWER.strip()}


@pytest.mark.parametrize("result, phase", [
    (_result(terminal_origin="host_salvage"), "error"),   # preserved bytes are not a final answer
    (_result(terminal_origin="host_notice"), "error"),    # the host's own notice is not his voice
    (_result(terminal_origin=""), "done"),                # unknown authorship is never assumed
    (_result(wake_silent=True), "done"),                  # retained internal words are not speech
    (_result(wake_silent=True, terminal_host_notice="A host notice."), "warn"),
    (_result(result="   "), "done"),                      # nothing to say
    (_result(result={"answer": "x"}), "done"),            # a malformed result is never repr()-ed into his voice
    (_result(), "working"),                               # the outbox freezes the first send
    (_result(), ""),
    (_result(result="x" * (MIRRORED_ANSWER_MAX_CHARS + 1)), "done"),  # over the ceiling: pointer, never a cut
    (None, "done"),
])
def test_everything_else_keeps_the_row_a_pointer(result, phase):
    assert mirrored_answer(result, phase) == {}


def test_the_ceiling_is_inclusive_and_nothing_is_ever_cut():
    exact = "y" * MIRRORED_ANSWER_MAX_CHARS
    assert mirrored_answer(_result(result=exact), "done") == {"completion_answer": exact}


def _enqueue_row(tmp_path, monkeypatch, result):
    from ouroboros.project_dialogue import enqueue_project_completion_summary
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "launch", name="Launch")
    bind_task_to_project(tmp_path, "root-project", project["id"], project["chat_id"],
                         origin={"absent": "system"})
    queued = []
    monkeypatch.setattr("supervisor.terminal_delivery.enqueue_terminal_delivery",
                        lambda _root, event, **_kw: queued.append(dict(event)) or True)
    ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path)
    root = {"id": "root-project", "project_id": "launch", "title": "Ship release",
            "chat_id": project["chat_id"]}
    row = {"task_id": "root-project", "status": "completed", "project_id": "launch",
           "title": "Ship release", **result}
    done = {"status": "completed", "outcome_axes": {"execution": {"status": "ok"}}}
    assert enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, {"status": "completed"}, "root-project", root, row, done) is True
    return queued[0]


def test_the_row_carries_the_answer_and_its_text_does_not_change(tmp_path, monkeypatch):
    mirrored = _enqueue_row(tmp_path / "a", monkeypatch, _result())
    pointer = _enqueue_row(tmp_path / "b", monkeypatch, _result(terminal_origin="host_salvage"))

    assert mirrored["progress_meta"]["completion_answer"] == ANSWER.strip()
    assert mirrored["progress_meta"]["outcome_axes"]["execution"]["status"] == "ok"
    assert "reply_to" not in mirrored["progress_meta"]  # system-origin work has no invented owner quote
    assert mirrored["role"] == "system"
    assert mirrored["system_type"] == "project_completion_summary"
    # Telegram, the model's own context and older clients read `text` alone: the answer never leaks into it,
    # and it invites nothing, since the row's Project reference is its door (owner 17.10, Q2 C).
    assert mirrored["text"] == "Launch › Ship release · Done"
    assert "PR #7" not in mirrored["text"]
    # Without a model-authored answer the key is ABSENT, not empty.
    assert "completion_answer" not in pointer["progress_meta"]


def test_the_answer_survives_live_frame_durable_row_and_history_replay(monkeypatch, tmp_path):
    bridge = message_bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.project_dialogue import build_owner_message_ref
    # The request itself can quote an older message. The answer quotes this request.
    origin = message_bus.log_chat("in", 1, 7, "Ship the release", client_message_id="owner-request",
                                  message_meta={"reply_to": {"address": "task:older", "kind": "work_card"}})
    origin_ref = build_owner_message_ref(chat_id=1, client_message_id="owner-request",
                                         text=origin["text"], ts=origin["ts"])
    event = _enqueue_row(tmp_path, monkeypatch, _result(origin_message_ref=origin_ref,
        outcome_axes={"objective": {"status": "pass", "source": "author_acceptance"}}))
    facts = {key: event["progress_meta"][key] for key in ("outcome_axes", "reply_to")}
    assert facts["reply_to"]["address"] == format_address(row_address(origin))
    assert facts["reply_to"]["snippet"] == "Ship the release"
    message_bus.send_with_budget(
        1, "Launch › Ship release · Done",
        task_id="root-project", role="system", system_type="project_completion_summary",
        progress_meta={**facts, "project_id": "launch", "project_name": "Launch",
                       "target_label": "Launch › Ship release", "status": "completed",
                       "completion_answer": ANSWER,
                       "completion_notice": {"neutral": True, "detail": "", "phase": "warn", "salvage": "",
                                             "cause": "The recorded reason has no description yet.",
                                             "undescribed": ["future_reason"]},
                       "terminal_time": {"v": 1, "source": "executor_terminal", "occurred_at": "2026-09-24T18:18:27Z"}},
    )

    live = next(frame for frame in frames if frame.get("type") == "chat")
    assert live["terminal_time"]["occurred_at"] == "2026-09-24T18:18:27Z"
    assert live["completion_answer"] == ANSWER
    assert live["role"] == "system"   # the wire voice is unchanged: one ending, one ring

    durable = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    stored = next(row for row in durable if row.get("type") == "project_completion_summary")
    assert stored["completion_answer"] == ANSWER
    assert stored["text"] == "Launch › Ship release · Done"

    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "1", "limit": "20"})))
    replayed = next(row for row in json.loads(response.body.decode("utf-8"))["messages"]
                    if row.get("system_type") == "project_completion_summary")
    # Markdown in the answer is NOT normalised away: the read-side plain
    # normalisation of lifecycle rows applies to `text` only.
    assert replayed["completion_answer"] == ANSWER
    assert replayed["text"] == stored["text"]
    assert replayed["terminal_time"] == stored["terminal_time"] == live["terminal_time"]
    for key in ("outcome_axes", "reply_to"):
        assert replayed[key] == stored[key] == live[key] == facts[key]
    assert replayed["completion_notice"] == stored["completion_notice"] == live["completion_notice"] == {
        "neutral": True, "detail": "", "phase": "warn", "salvage": "",
        "cause": "The recorded reason has no description yet.", "undescribed": ["future_reason"]}


def test_a_row_without_the_key_replays_without_it(monkeypatch, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "chat.jsonl").write_text(json.dumps({
        "ts": "2026-08-21T00:00:03Z", "direction": "system", "chat_id": 1,
        "type": "project_completion_summary", "task_id": "root-project", "project_id": "launch",
        "project_name": "Launch", "target_label": "Launch › Ship release", "status": "completed",
        "text": "Launch › Ship release · Completed",
    }) + "\n", encoding="utf-8")

    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "1"})))
    row = json.loads(response.body.decode("utf-8"))["messages"][0]
    assert "completion_answer" not in row


def test_moved_direct_completion_uses_bound_main_request_without_a_result_origin(tmp_path, monkeypatch):
    from ouroboros.chat_references import reference_from_row
    from ouroboros.project_dialogue import build_owner_message_ref, enqueue_project_completion_summary
    from ouroboros.projects_registry import create_project, bind_task_to_project
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    origin = message_bus.log_chat("in", 1, 7, "Move this work", client_message_id="request")
    ref = build_owner_message_ref(chat_id=1, client_message_id="request", text=origin["text"], ts=origin["ts"])
    project = create_project(tmp_path, "room", name="Room")
    bind_task_to_project(tmp_path, "moved", "room", project["chat_id"], origin={"ref": ref, "text": origin["text"]})
    queued = []
    monkeypatch.setattr("supervisor.terminal_delivery.enqueue_terminal_delivery",
                        lambda root, event, **kwargs: queued.append(event) or True)
    assert enqueue_project_completion_summary(tmp_path, {}, "moved",
        {"id": "moved", "_is_direct_chat": True, "project_id": "room", "chat_id": 1},
        {"status": "failed", "project_id": "room", "result": "Explanation", "terminal_origin": "model_final",
         "outcome_axes": {"execution": {"status": "infra_failed"}}}, {})
    assert queued[0]["progress_meta"]["reply_to"] == reference_from_row(origin)
    assert queued[0]["progress_meta"]["outcome_axes"]["execution"]["status"] == "infra_failed"


@pytest.mark.parametrize("reason, salvaged", [("worker_crash_signal", False), ("example_new_reason", True)])
def test_a_no_answer_ending_carries_its_web_facts_beside_its_unchanged_text(tmp_path, monkeypatch, reason, salvaged):
    """Owner 1A/2A (21.10): Main's row for an ending with no authored answer carries the work's own name and a cause
    the web row can say (a typed code with no sentence yet reads neutrally, the exact code beside it for details);
    its text, Telegram's notice detail and the answered-row limitation keep the canonical words and the exact code."""
    from ouroboros import outcomes
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES

    result = {"status": "failed", "reason_code": reason, "outcome_axes": outcomes.terminal_outcome_axes(
        lifecycle="failed", execution=outcomes.EXECUTION_INFRA_FAILED, reason_code=reason, review_trigger="worker_terminal")}
    result.update({"terminal_origin": "host_salvage", "result": "Two update scenarios remain unchecked."} if salvaged
                  else {"result": "❌ worker process crashed (signal 9)."})
    event = _enqueue_row(tmp_path, monkeypatch, result)
    meta, known = event["progress_meta"], reason in TASK_CAUSE_PHRASES
    canonical = f"{TASK_CAUSE_PHRASES[reason]}." if known else f"{reason}."
    salvage = ("Preserved intermediate output (not a final answer): Two update scenarios remain unchecked."
               if salvaged else "")
    assert event["text"] == "Launch › Ship release · Failed\n" + " ".join(p for p in (canonical, salvage) if p)
    assert (meta["task_name"], meta["project_name"]) == ("Ship release", "Launch")
    assert "completion_answer" not in meta and meta["completion_reason"] == canonical
    notice = meta["completion_notice"]
    assert (notice["phase"], notice["detail"], notice["salvage"]) == ("error", canonical, salvage)
    assert notice["cause"] == (canonical if known else TASK_CAUSE_PHRASES["reason_undescribed"])
    assert notice["undescribed"] == ([] if known else [reason])

"""Quiz display path: shared validator, bridge send, delivery handler, replay."""
import json
import types

import pytest

from ouroboros import event_bus
from ouroboros.tools.core import (
    _MAX_QUIZ_OPTIONS,
    QuizValidationError,
    validate_quiz_payload,
)
from supervisor import message_bus
from tests.test_message_bus import _make_bridge


class TestValidateQuizPayload:
    def test_cleans_labels_details_and_string_options(self):
        payload = validate_quiz_payload(
            "  Merge now?  ",
            ["Yes", {"label": " No ", "detail": " wait for CI "}],
            stake=" release timing ",
            assumption=" continuing with the merge ",
        )
        assert payload["question"] == "Merge now?"
        assert payload["options"] == [
            {"label": "Yes"},
            {"label": "No", "detail": "wait for CI"},
        ]
        assert payload["stake"] == "release timing"
        assert payload["assumption"] == "continuing with the merge"

    @pytest.mark.parametrize("options", [
        [], ["only-one"], ["a"] * (_MAX_QUIZ_OPTIONS + 1), "not-a-list",
        [{"detail": "no label"}],
    ])
    def test_bad_options_are_refused_atomically(self, options):
        with pytest.raises(QuizValidationError):
            validate_quiz_payload("q", options, "", "assume")

    def test_assumption_is_required(self):
        # Owner decision 27=A: fire-and-continue must name its default path.
        with pytest.raises(QuizValidationError) as err:
            validate_quiz_payload("q", ["a", "b"], "", "  ")
        assert err.value.code == "QUIZ_ASSUMPTION_REQUIRED"

    def test_empty_or_oversized_question_refused(self):
        with pytest.raises(QuizValidationError):
            validate_quiz_payload("", ["a", "b"], "", "assume")
        with pytest.raises(QuizValidationError):
            validate_quiz_payload("q" * 2001, ["a", "b"], "", "assume")


def test_send_quiz_broadcasts_publishes_and_persists_row(monkeypatch, tmp_path):
    bridge = _make_bridge(monkeypatch)
    frames = []
    events = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda _chat_id: None)
    monkeypatch.setattr(
        message_bus, "publish_event", lambda topic, data: events.append((topic, data)),
    )

    ok, error = bridge.send_quiz(
        123,
        quiz_id="qz-1",
        question="Merge now?",
        options=[{"label": "Yes"}, {"label": "No", "detail": "wait for CI"}],
        stake="release timing",
        assumption="continuing with the merge",
        task_id="task-quiz",
    )

    assert (ok, error) == (True, "ok")
    live = next(frame for frame in frames if frame.get("type") == "quiz")
    assert live["role"] == "assistant"
    assert live["quiz_id"] == "qz-1"
    assert live["question"] == "Merge now?"
    assert live["options"][1] == {"label": "No", "detail": "wait for CI"}
    assert live["state"] == "open"
    assert live["task_id"] == "task-quiz"
    topic, payload = events[-1]
    assert topic == event_bus.CHAT_QUIZ
    assert set(payload) == {
        # task_id joined the topic payload (#Q-2b, closing review note N4):
        # a host subscriber (Telegram) cannot compose the answer address
        # "quiz:{task_id}:{quiz_id}" without it.
        "chat_id", "transport", "quiz_id", "task_id", "question", "options",
        "stake", "assumption", "state", "ts",
    }
    row = json.loads((tmp_path / "logs" / "chat.jsonl").read_text().splitlines()[-1])
    assert row["type"] == "quiz"
    assert row["text"] == "Merge now?"
    assert row["task_id"] == "task-quiz"
    assert row["quiz"]["quiz_id"] == "qz-1"
    assert row["quiz"]["state"] == "open"
    assert row["quiz"]["options"][0] == {"label": "Yes"}


def test_send_quiz_refuses_invalid_payload_and_missing_ids(monkeypatch, tmp_path):
    bridge = _make_bridge(monkeypatch)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    ok, error = bridge.send_quiz(1, quiz_id="", question="q", options=[{"label": "a"}, {"label": "b"}], assumption="x")
    assert not ok and "quiz_id" in error
    # An anonymous quiz cannot deliver its answer anywhere: task_id required.
    ok, error = bridge.send_quiz(1, quiz_id="qz", question="q", options=[{"label": "a"}, {"label": "b"}], assumption="x")
    assert not ok and "task_id" in error
    ok, error = bridge.send_quiz(1, quiz_id="qz", question="q", options=[{"label": "a"}], assumption="x", task_id="t")
    assert not ok
    ok, error = bridge.send_quiz(-5, quiz_id="qz", question="q", options=[{"label": "a"}, {"label": "b"}], assumption="x")
    assert (ok, error) == (True, "ok")  # A2A chats: silent no-op, like links


def test_handle_send_quiz_prefers_bound_project_chat(monkeypatch):
    from supervisor.events_chat_delivery import _handle_send_quiz

    sent = []

    class _Bridge:
        def send_quiz(self, chat_id, **kwargs):
            sent.append((chat_id, kwargs))
            return True, ""

    ctx = types.SimpleNamespace(bridge=_Bridge(), append_jsonl=lambda *a, **k: None,
                                DRIVE_ROOT=None)
    import supervisor.events_chat_delivery as cde

    monkeypatch.setattr(cde, "_bound_project_chat_id", lambda *a, **k: 4242)
    evt = {
        "type": "send_quiz", "chat_id": 1, "task_id": "t1",
        "parent_task_id": "", "root_task_id": "",
        "quiz_id": "qz-2", "question": "Which path?",
        "options": [{"label": "A"}, {"label": "B"}],
        "stake": "", "assumption": "path A meanwhile", "state": "open",
    }
    _handle_send_quiz(evt, ctx)
    assert sent and sent[0][0] == 4242
    assert sent[0][1]["quiz_id"] == "qz-2"
    assert sent[0][1]["assumption"] == "path A meanwhile"

    # No options -> typed drop, no bridge call.
    sent.clear()
    _handle_send_quiz({**evt, "options": []}, ctx)
    assert sent == []

    # Headless exception: an interactive card in the hidden chat-0 panel can
    # never be answered, so it goes to Main instead.
    sent.clear()
    monkeypatch.setattr(cde, "_bound_project_chat_id", lambda *a, **k: None)
    _handle_send_quiz({**evt, "chat_id": 0}, ctx)
    assert sent and sent[0][0] == 1


def test_telegram_manifest_declares_every_plugin_subscription():
    """The loader tears the WHOLE skill down on one undeclared topic: the
    manifest subscribe_events list must cover every plugin subscribe_event."""
    import pathlib as _pathlib
    import re as _re

    root = _pathlib.Path(__file__).resolve().parent.parent / "skills" / "telegram"
    manifest = (root / "SKILL.md").read_text(encoding="utf-8")
    declared = set()
    match = _re.search(r"subscribe_events:\s*\[([^\]]*)\]", manifest)
    assert match, "SKILL.md missing subscribe_events"
    declared = {item.strip() for item in match.group(1).split(",") if item.strip()}
    plugin = (root / "plugin.py").read_text(encoding="utf-8")
    subscribed = set(_re.findall(r"subscribe_event\(\"([^\"]+)\"", plugin))
    assert subscribed <= declared, f"undeclared topics: {sorted(subscribed - declared)}"


def test_chat_quiz_event_reaches_telegram_consumer():
    """Real producer -> real consumer addressing: the exact host event
    send_quiz publishes on the shared bus is what the Telegram quiz handler
    addresses (the chat.links cross-stream wiring, cloned for quiz)."""
    from ouroboros.event_bus import CHAT_QUIZ, VALID_TOPICS, get_global_event_bus
    from skills.telegram import plugin as telegram_plugin

    assert CHAT_QUIZ == "chat.quiz"
    assert CHAT_QUIZ in VALID_TOPICS

    bus = get_global_event_bus()
    received = []
    sub_id = bus.subscribe("telegram-quiz-probe", CHAT_QUIZ, lambda data: received.append(data))
    try:
        bridge = message_bus.LocalChatBridge({})
        bridge._broadcast_fn = lambda payload: None
        bridge._chat_transports[7] = {"kind": "telegram", "conversation_id": 777}
        ok, error = bridge.send_quiz(
            7, quiz_id="qz-e2e", question="Which path?",
            options=[{"label": "A"}, {"label": "B"}],
            assumption="path A meanwhile", task_id="t-e2e",
        )
        assert (ok, error) == (True, "ok")
        assert received, "send_quiz must publish a chat.quiz host event"
        captured = received[-1]
        assert telegram_plugin._target_chat({"TELEGRAM_CHAT_ID": ""}, captured) == 777
        assert captured["question"] == "Which path?"
        assert [o["label"] for o in captured["options"]] == ["A", "B"]
    finally:
        bus.unsubscribe(sub_id)


def test_atif_final_answer_skips_typed_delivery_rows(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "atif_under_test",
        pathlib_root() / "devtools" / "benchmarks" / "terminal_bench" / "atif.py",
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        import pytest as _pytest
        _pytest.skip("atif module has optional deps not present here")
    data = tmp_path / "ouroboros-data" / "logs"
    data.mkdir(parents=True)
    rows = [
        {"direction": "out", "text": "the real final", "ts": "1"},
        {"direction": "out", "text": "Which path?", "type": "quiz", "ts": "2",
         "quiz": {"quiz_id": "q", "options": []}},
    ]
    (data / "chat.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert module._final_answer(tmp_path) == "the real final"


def pathlib_root():
    import pathlib as _pathlib
    return _pathlib.Path(__file__).resolve().parent.parent


def test_history_replays_quiz_row_with_state(tmp_path):
    import asyncio

    from ouroboros.gateway.history import make_chat_history_endpoint

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    row = {
        "ts": "2026-08-31T10:00:00Z", "session_id": "s", "direction": "out",
        "chat_id": 1, "user_id": 7, "text": "Merge now?", "format": "",
        "source": "", "sender_label": "", "sender_session_id": "",
        "client_message_id": "", "transport": {}, "task_id": "task-quiz",
        "type": "quiz",
        "quiz": {
            "quiz_id": "qz-3",
            "options": [{"label": "Yes"}, {"label": "No"}],
            "stake": "", "assumption": "merging meanwhile", "state": "open",
        },
    }
    (logs / "chat.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(types.SimpleNamespace(query_params={"limit": "10"})))
    messages = json.loads(response.body.decode("utf-8"))["messages"]
    quiz_rows = [m for m in messages if m.get("msg_type") == "quiz"]
    assert len(quiz_rows) == 1
    rec = quiz_rows[0]
    assert rec["text"] == "Merge now?"
    assert rec["quiz"]["quiz_id"] == "qz-3"
    assert rec["quiz"]["state"] == "open"
    assert rec["system_type"] == "quiz"  # typed row: replay never reads it as a bare final

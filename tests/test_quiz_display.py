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
        ["a"] * (_MAX_QUIZ_OPTIONS + 1), "not-a-list",
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

    @pytest.mark.parametrize("options", [None, [], ["Confirm"]])
    def test_open_and_single_choice_questions_keep_the_answer_contract(self, options):
        payload = validate_quiz_payload("What should change?", options, "", "", wait_for_answer=True)
        assert payload["options"] == ([] if options is None else [{"label": x} for x in options])
        with pytest.raises(QuizValidationError) as error:
            validate_quiz_payload("What should change?", options, "", "")
        assert error.value.code == "QUIZ_ASSUMPTION_REQUIRED"

    def test_wait_bound_is_whole_minutes_capped_by_the_task_ceiling(self, monkeypatch):
        """The bound belongs to a REQUIRED wait and can never promise more time
        than the absolute wall-clock ceiling already allows (beyond it the
        ceiling ends the task first, so the bound would be a false promise)."""
        monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "21600")  # 360 minutes
        payload = validate_quiz_payload("q", ["a", "b"], "", "",
                                        wait_for_answer=True, max_wait_minutes=30)
        assert payload["max_wait_minutes"] == 30
        # Absent unless asked for: an unbounded wait stays unbounded.
        assert "max_wait_minutes" not in validate_quiz_payload(
            "q", ["a", "b"], "", "", wait_for_answer=True)
        for bad in (0, -5, True, 1.5, "30", 361):
            with pytest.raises(QuizValidationError) as err:
                validate_quiz_payload("q", ["a", "b"], "", "",
                                      wait_for_answer=True, max_wait_minutes=bad)
            assert err.value.code == "QUIZ_WAIT_BOUND_INVALID"
        # On an optional question a bound only names the documented default (no wait), so it
        # takes the omitted path whatever a habit-filled schema put there; the asker's receipt
        # discloses it (tests/test_native_owner_wait.py).
        for named_default in (0, 1, 5, 60, -5, True, "30"):
            assert "max_wait_minutes" not in validate_quiz_payload(
                "q", ["a", "b"], "", "assume", max_wait_minutes=named_default)
        # A required wait keeps the refusal, and every such refusal names the repair.
        for bad in (0, 361):
            with pytest.raises(QuizValidationError) as err:
                validate_quiz_payload("q", ["a", "b"], "", "", wait_for_answer=True, max_wait_minutes=bad)
            assert "omit it for an unbounded wait" in str(err.value)

    def test_question_has_no_length_cap_but_must_be_non_empty(self):
        """The card may be the only explanation its reader gets: a long
        question is accepted whole; only an empty one is refused."""
        long_question = "q" * 2001
        assert validate_quiz_payload(long_question, ["a", "b"], "", "assume")["question"] == long_question
        for empty in ("", "   \n\t "):
            with pytest.raises(QuizValidationError) as err:
                validate_quiz_payload(empty, ["a", "b"], "", "assume")
            assert err.value.code == "QUIZ_QUESTION_INVALID"
            assert str(err.value) == "question must be non-empty."

    def test_over_long_label_is_refused_not_sliced(self):
        at_bound = "L" * 120
        payload = validate_quiz_payload("q", [at_bound, "b"], "", "assume")
        assert payload["options"][0]["label"] == at_bound
        with pytest.raises(QuizValidationError) as err:
            validate_quiz_payload("q", ["L" * 121, "b"], "", "assume")
        assert err.value.code == "QUIZ_OPTIONS_INVALID"
        assert str(err.value) == "option labels must be at most 120 characters."
        # A dict option takes the same bound.
        with pytest.raises(QuizValidationError) as err:
            validate_quiz_payload("q", [{"label": "L" * 121, "detail": "d"}, "b"], "", "assume")
        assert err.value.code == "QUIZ_OPTIONS_INVALID"

    def test_detail_stake_and_assumption_survive_byte_exact(self):
        detail = "детали🙂 " * 75 + "end"  # well past the former 500-char slice
        stake = "stake-" * 100
        assumption = "assumption-" * 60
        assert min(len(detail), len(stake), len(assumption)) > 500
        payload = validate_quiz_payload(
            "q", [{"label": "a", "detail": detail}, "b"], stake, assumption)
        assert payload["options"][0]["detail"] == detail.strip()
        assert payload["stake"] == stake
        assert payload["assumption"] == assumption

    def test_multi_paragraph_unicode_question_survives_into_the_stored_block(self, tmp_path):
        """Validator and the root ask path together: nothing between the tool
        call and the durable owner_quiz block cuts the authored explanation."""
        from ouroboros.owner_quiz import quiz_states
        from tests.test_quiz_answer import _escalate, _tool_ctx

        question = (
            "## Что решаем\n\n"
            + "Абзац с объяснением — «кавычки», emoji 🚀, 𐍈. " * 60
            + "\n\nSecond paragraph in English, with `code` and a list:\n- one\n- two"
        )
        detail = "Что меняется: " + "x" * 700
        assert len(question) > 2000
        assert validate_quiz_payload(question, ["a", "b"], "", "assume")["question"] == question
        ctx = _tool_ctx(tmp_path, role="root")
        out = _escalate(ctx, question=question,
                        options=[{"label": "A", "detail": detail}, {"label": "B"}],
                        stake="s" * 600, assumption="a" * 600)
        assert out.startswith("OK: quiz ")
        event = next(e for e in ctx.pending_events if e.get("type") == "send_quiz")
        assert event["question"] == question
        block = quiz_states(tmp_path, "root-1")[event["quiz_id"]]
        assert block["question"] == question
        assert block["option_details"] == [detail, ""]
        assert block["stake"] == "s" * 600 and block["assumption"] == "a" * 600


@pytest.mark.parametrize("wait_for_answer", [False, True])
def test_send_quiz_broadcasts_publishes_and_persists_row(monkeypatch, tmp_path, wait_for_answer):
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
        assumption="" if wait_for_answer else "continuing with the merge",
        task_id="task-quiz",
        wait_for_answer=wait_for_answer,
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
        "stake", "assumption", "state", "ts", "wait_for_answer",
    }
    row = json.loads((tmp_path / "logs" / "chat.jsonl").read_text().splitlines()[-1])
    assert row["type"] == "quiz"
    assert row["text"] == "Merge now?"
    assert row["task_id"] == "task-quiz"
    assert row["quiz"]["quiz_id"] == "qz-1"
    assert row["quiz"]["state"] == "open"
    assert row["quiz"]["options"][0] == {"label": "Yes"}
    assert live["wait_for_answer"] is payload["wait_for_answer"] is row["quiz"]["wait_for_answer"] is wait_for_answer


def test_send_quiz_accepts_a_long_question_through_the_shared_validator(monkeypatch, tmp_path):
    bridge = _make_bridge(monkeypatch)
    frames, events = [], []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda _chat_id: None)
    monkeypatch.setattr(message_bus, "publish_event", lambda topic, data: events.append((topic, data)))
    question = "Длинное объяснение решения. " * 200
    detail = "consequence " * 60
    ok, error = bridge.send_quiz(
        123, quiz_id="qz-long", question=question,
        options=[{"label": "Yes", "detail": detail}, {"label": "No"}],
        assumption="continuing", task_id="task-quiz",
    )
    assert (ok, error) == (True, "ok")
    live = next(frame for frame in frames if frame.get("type") == "quiz")
    assert live["question"] == question.strip()
    assert live["options"][0]["detail"] == detail.strip()
    assert events[-1][1]["question"] == question.strip()
    row = json.loads((tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["text"] == question.strip()
    # The same validator still refuses an over-long label atomically.
    ok, error = bridge.send_quiz(
        123, quiz_id="qz-label", question="q",
        options=[{"label": "L" * 121}, {"label": "No"}],
        assumption="continuing", task_id="task-quiz",
    )
    assert not ok and "at most 120 characters" in error


def test_send_quiz_refuses_invalid_payload_and_missing_ids(monkeypatch, tmp_path):
    bridge = _make_bridge(monkeypatch)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    ok, error = bridge.send_quiz(1, quiz_id="", question="q", options=[{"label": "a"}, {"label": "b"}], assumption="x")
    assert not ok and "quiz_id" in error
    # An anonymous quiz cannot deliver its answer anywhere: task_id required.
    ok, error = bridge.send_quiz(1, quiz_id="qz", question="q", options=[{"label": "a"}, {"label": "b"}], assumption="x")
    assert not ok and "task_id" in error
    ok, error = bridge.send_quiz(1, quiz_id="qz", question="q", options=[{"label": "a"}], assumption="x", task_id="t")
    assert (ok, error) == (True, "ok")
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

    # An explicitly empty options list is an open question; absence is not.
    sent.clear()
    _handle_send_quiz({**evt, "options": []}, ctx)
    assert sent and sent[0][1]["options"] == []
    sent.clear()
    _handle_send_quiz({key: value for key, value in evt.items() if key != "options"}, ctx)
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


def test_at_most_one_recommended_option_for_both_callers(monkeypatch, tmp_path):
    """Fix cycle 2, 2d: the durable record keeps ONE recommended index, so the shared
    validator refuses a second recommendation for the tool and the bus alike, with the
    same typed shape as its other refusals; one recommendation passes through intact."""
    with pytest.raises(QuizValidationError) as err:
        validate_quiz_payload("q", [{"label": "a", "recommended": True}, {"label": "b", "recommended": True}], "", "assume")
    assert err.value.code == "QUIZ_RECOMMENDED_INVALID" and "at most one" in str(err.value)
    one = validate_quiz_payload("q", [{"label": "a"}, {"label": "b", "recommended": True}], "", "assume")
    assert one["options"] == [{"label": "a"}, {"label": "b", "recommended": True}]
    bridge = _make_bridge(monkeypatch)
    ok, error = bridge.send_quiz(
        1, quiz_id="qz", question="q", task_id="t-1",
        options=[{"label": "a", "recommended": True}, {"label": "b", "recommended": True}], assumption="x")
    assert ok is False and error == "mark at most one option as recommended."

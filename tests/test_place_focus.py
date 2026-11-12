"""Place intake, focus authority, reply boundaries and ordinary completion."""
from dataclasses import replace
from types import SimpleNamespace
import json
import string

import pytest

from ouroboros import places as p
from ouroboros.presence_admission import admit_presence_turn
from ouroboros.presence_runner import PresenceTurnEvent, _build_task, capture_presence_event, presence_turn_replay
from ouroboros.place_intake import decide_attention, settle_heard


@pytest.fixture
def focus(tmp_path, monkeypatch):
    place = p.connect_place(tmp_path, connector="bridge", provider="chat", account_id="a", conversation_id="r", wakes=["mention"])
    event = PresenceTurnEvent("event1", "chat", "a", "r", "", "chat:a:r", {"platform_actor_id": "person"}, {}, {}, "hello", facts=("mention",))
    admission = admit_presence_turn(drive_root=tmp_path, authenticated_transport_skill="bridge", provider="chat", account_id="a", conversation_id="r")
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available", "remaining_usd": 2, "settled_usd": 1})
    monkeypatch.setattr("ouroboros.presence_context._own_work_section", lambda *_args: "Already started: task1")
    return tmp_path, place, admission, event


def task_for(focus):
    root, _place, admission, event = focus
    return _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2}), event, drive_root=root, staged_files=())


def test_focus_no_ceiling_host_actor_prompt(focus):
    from ouroboros.presence_context import frame_focus_content, PLACEHOLDERS
    from pathlib import Path
    task = task_for(focus)
    meta = task["metadata"]
    assert meta["wake_cause"] == "focus" and meta["work_kind"] == "consciousness"
    assert meta["usage_category"] == "consciousness" and meta["runtime_mode_cap"] == "light"
    assert meta["focus"]["actor"]["owner_confirmed"] is False
    assert meta["focus"]["event_identity"] == meta["presence_event_identity"]
    assert "capability_ceiling" not in task["task_contract"]
    assert task["_is_direct_chat"] and task["_presence_turn"]
    # A focus reads places like any wake below Full; its edits are refused on the arguments (consciousness_authority).
    assert meta["root_cost_ceiling_usd"] <= 2 and "toggle_consciousness" in meta["disabled_tools"]
    assert "configure_presence" not in meta["disabled_tools"]
    template = Path("prompts/FOCUS.md").read_text(encoding="utf-8")
    assert {key for _lit, key, _fmt, _conv in string.Formatter().parse(template) if key} == set(PLACEHOLDERS)
    rendered = frame_focus_content(task, "The untrusted event")
    assert "The untrusted event" in rendered and "Already started: task1" in rendered
    assert "connector" in rendered and "folder" in rendered.lower()
    # The speech and authority contracts, said once: this connector declares no in-conversation send tool, so its
    # finish is posted (the speak line says so); an order exists only by a verified cite (the rule in the positive).
    for words in ("Your non-silent finish is posted in chat as your words.", "Silence is recorded as heard.",
                  "Finish with summary: what you heard and did, in your human's language; send_user_message reaches them now.",
                  "Carry your human's order only by citing their exact words received here "
                  "(owner_words of start_task or schedule_followup); names and roles alone prove no authorship."):
        assert words in rendered, words
    for words in ("brief useful first reply", "This turn is yours", "no owner-command authority", "your answer is new speech"):
        assert words not in rendered, words  # owner 04.11; the verdict; the negative rule; the posted-final contract
    # The connector's send and edit times (CREATING_SKILLS ``message.sent_at``/``edited_at``), else unknown.
    assert " at an unrecorded time." in rendered
    root, _place, admission, event = focus
    dated = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2}), replace(event, message={
        "sent_at": "2026-11-05T17:32:00+03:00", "edited_at": "2026-11-05T14:40:00Z", "timestamp": "1"}),
        drive_root=root, staged_files=())
    rendered = frame_focus_content(dated, "The untrusted event")
    assert " at 2026-11-05T17:32:00+03:00, edited at 2026-11-05T14:40:00Z." in rendered and "unrecorded" not in rendered
    dated["metadata"]["presence"]["event"]["message"] = {"sent_at": "2026-11-05T17:32:00"}  # no time zone: not a time
    assert " at an unrecorded time." in frame_focus_content(dated, "The untrusted event")


@pytest.mark.parametrize("state,facts,enabled,allowance,decision", [
    ("active", ("mention",), True, "available", "focus"),
    ("paused", ("mention",), True, "available", "silent"),
    ("active", (), True, "available", "silent"),
    ("active", ("mention",), False, "available", "silent"),
    ("active", ("mention",), True, "exhausted", "retry"),
])
def test_capture_before_attention_and_replay(focus, monkeypatch, state, facts, enabled, allowance, decision):
    from ouroboros import dialogue_store
    root, place, admission, event = focus
    p.edit_place(root, place.place_id, "set_state", author="owner", state=state)
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _r: enabled)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _r: {"status": allowance})
    event = replace(event, facts=facts)
    task = capture_presence_event(admission, event, drive_root=root)
    capture_presence_event(admission, event, drive_root=root)
    rows = list(dialogue_store.iter_rows(root, "chat"))
    assert len(rows) == 1
    assert rows[0].row["text"] == "hello"
    actual, info = decide_attention(root, admission, event)
    assert actual == decision
    if decision == "retry":
        assert info["retry_after"] > 0
    if decision == "silent":
        result = settle_heard(root, task, event)
        assert result.outcome == "silent"
        replay = presence_turn_replay(root, task["id"], event.conversation_key, task["metadata"]["presence_event_identity"])
        assert replay == result


def test_identity_conflict_and_legacy_facts(focus):
    root, place, admission, event = focus
    capture_presence_event(admission, event, drive_root=root)
    with pytest.raises(ValueError, match="identity_conflict"):
        capture_presence_event(admission, replace(event, text="different"), drive_root=root)
    assert decide_attention(root, admission, replace(event, facts=None))[0] == "silent"
    p.edit_place(root, place.place_id, "add_wake", author="owner", fact="legacy_addressed")
    assert decide_attention(root, admission, replace(event, facts=None))[0] == "focus"
    assert decide_attention(root, admission, replace(event, actor={"is_bot": True}))[0] == "silent"


def test_reply_guard_flag_reissue_silent_and_receipted(focus, monkeypatch):
    from ouroboros.place_reply import prepare_focus_reply
    root, place, _admission, _event = focus
    task = task_for(focus)
    ctx = SimpleNamespace(task_metadata=task["metadata"], task_id=task["id"], drive_root=root)
    monkeypatch.setattr("ouroboros.tool_access.canonical_data_root", lambda _ctx: root)
    calls = []
    def check(name, args, **kw):
        calls.append((name, args, kw))
        return (len(calls) > 1, "Reconsider")
    monkeypatch.setattr("ouroboros.safety.check_safety", check)
    assert prepare_focus_reply(ctx, "Hello") == (False, "Reconsider")
    assert not getattr(ctx, "_presence_completion_accepted", False)
    assert prepare_focus_reply(ctx, "Hello")[0]
    assert len(calls) == 2 and calls[0][0] == "place_reply"
    assert ctx._presence_completion["outcome"] == "message"
    prepare_focus_reply(ctx, "Hello")
    assert len(calls) == 2  # final result projection does not recheck accepted identical bytes
    prepare_focus_reply(ctx, "Changed")
    assert len(calls) == 3
    p.edit_place(root, place.place_id, "set_state", author="owner", state="silent")
    assert prepare_focus_reply(ctx, "Changed")[0]
    assert ctx._presence_completion["outcome"] == "silent" and "Reply held" in ctx._focus_reply_hold
    p.edit_place(root, place.place_id, "set_state", author="owner", state="active")
    # A real confirmed tool receipt (here an empty-text reaction) never selects the outcome:
    # new words are new speech under the current check; only the mind's silence says nothing.
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_context import confirmed_delivery_receipts
    from ouroboros.presence_delivery import PresenceDeliveryRecorder
    from ouroboros.task_results import write_task_result
    task["metadata"]["presence"]["event"]["conversation_key"] = conversation_key("chat", "a", "r", "")  # the Host's key
    write_task_result(root, task["id"], "running", metadata=task["metadata"])
    PresenceDeliveryRecorder(root).record("bridge", {
        "schema_version": 1, "delivery_id": "reaction-1", "part_id": "status", "state": "delivered",
        "provider": "chat", "account_id": "a", "conversation_id": "r", "thread_id": "", "text": "",
        "format": "", "message": {"kind": "operation", "method": "setMessageReaction", "reaction": []},
        "origin": {"kind": "tool", "task_id": task["id"], "source_event_id": "event1"}})
    assert confirmed_delivery_receipts(ctx) == frozenset({("reaction-1", "status")})
    assert prepare_focus_reply(ctx, "Final after the reaction")[0]
    assert ctx._presence_completion["outcome"] == "message" and len(calls) == 4
    assert prepare_focus_reply(ctx, "Internal note", quiet=True)[0]
    assert ctx._presence_completion["outcome"] == "silent" and ctx._presence_completion["quiet"] is True
    assert len(calls) == 4  # silence buys no check


def test_owner_verb_authorship_and_atomic_api(focus, monkeypatch):
    from ouroboros.tools.presence import _configure_presence, get_tools
    from ouroboros.gateway.places_api import update_place
    root, place, _admission, _event = focus
    monkeypatch.setattr("ouroboros.tool_access.canonical_data_root", lambda _ctx: root)
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _r: [])
    ctx = SimpleNamespace(task_metadata=task_for(focus)["metadata"], is_direct_chat=True)
    refusal = _configure_presence(ctx, "confirm_owner_account", place.place_id, platform_actor_id="person")
    assert "RESOURCE_CONSTRAINT_BLOCKED" in refusal and not p.get_place(root, place.place_id).owner_accounts
    ctx.task_metadata = {"origin_message_ref": {"client_message_id": "owner1", "chat_id": 1}}
    assert json.loads(_configure_presence(ctx, "confirm_owner_account", place.place_id, platform_actor_id="person"))["ok"]
    assert p.is_owner_account(p.get_place(root, place.place_id), "person", "a")
    saved = update_place(root, place.place_id, {"state": "silent", "add_note": "Owner choice"})["place"]
    assert saved["state"] == "silent" and saved["notes"][0]["author"] == "owner"
    assert saved["notes"][0]["note_id"]
    before = p.get_place(root, place.place_id)
    with pytest.raises(p.PlaceError):
        update_place(root, place.place_id, {"state": "active", "wakes": ["not_a_fact"]})
    assert p.get_place(root, place.place_id) == before
    assert {t.name for t in get_tools()} == {"configure_presence", "presence_cancel_work"}


def test_a_note_is_signed_by_whoever_composed_its_words(focus, monkeypatch):
    """Audit 05.11 item 2: an owner-door turn signed every note the mind composed as my human's. An act stays ordered by
    the turn; a note is my human's only when its whole text is their exact words in an owner-door turn."""
    from ouroboros.tools.presence import _configure_presence
    root, place, _admission, _event = focus
    monkeypatch.setattr("ouroboros.tool_access.canonical_data_root", lambda _ctx: root)
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _r: [])
    card = {"address": "quiz q of task a", "author": "Ouroboros, task a", "text": "Publish the draft here?"}
    door = SimpleNamespace(task_metadata={"origin_message_ref": {"client_message_id": "owner1", "chat_id": 1}},
                           is_direct_chat=True, task_id="t1", _owner_directives=[
                               {"source": "initial_user", "content": "Здесь можно всё обсуждать.\nСекретов от Яна нет."},
                               {"source": "owner_quiz_answer", "content": "1. Publish it", "reply_to": card}])
    for text in ("Секретов от Яна нет.", "Можно обсуждать всё, включая финансы.", "Publish the draft here?", "Publish it"):
        assert json.loads(_configure_presence(door, "add_note", place.place_id, text=text))["ok"]
    focused = SimpleNamespace(task_metadata=task_for(focus)["metadata"], is_direct_chat=True, task_id="t2",
                              _owner_directives=[{"source": "place_owner", "content": "Тут не про работу.", "carried": True}])
    assert json.loads(_configure_presence(focused, "add_note", place.place_id, text="Тут не про работу."))["ok"]
    assert [(n.text, n.author) for n in p.get_place(root, place.place_id).notes] == [
        ("Секретов от Яна нет.", "owner"), ("Можно обсуждать всё, включая финансы.", "ouroboros"),
        ("Publish the draft here?", "ouroboros"), ("Publish it", "owner"), ("Тут не про работу.", "ouroboros")]
    assert json.loads(_configure_presence(door, "set_state", place.place_id, state="active"))["ok"]  # still the turn's act
    assert p.get_place(root, place.place_id).state == "active"


def test_circle_words_and_placements_follow_the_note_rule_and_the_turns_act(focus, monkeypatch):
    """Circle words and a description are my human's only as their exact words in an owner-door turn; a placement is
    the turn's act. A focus (consciousness) may place a place, but its placement certifies no choice of my human's;
    list names each circle's words once and inspect the words a place inherits."""
    from ouroboros.tools.presence import _configure_presence
    root, place, _admission, _event = focus
    monkeypatch.setattr("ouroboros.tool_access.canonical_data_root", lambda _ctx: root)
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _r: [])
    door = SimpleNamespace(task_metadata={"origin_message_ref": {"client_message_id": "owner1", "chat_id": 1}},
                           is_direct_chat=True, task_id="t1", _owner_directives=[
                               {"source": "initial_user", "content": "Друзья: про работу молчи.\nЧат друзей"}])
    focused = SimpleNamespace(task_metadata=task_for(focus)["metadata"], is_direct_chat=True, task_id="t2",
                              _owner_directives=[])
    call = lambda ctx, action, **kw: json.loads(_configure_presence(ctx, action, kw.pop("place_id", ""), **kw))
    owned = call(door, "create_circle", name="Друзья", text="Друзья: про работу молчи.")["circle"]
    composed = call(door, "create_circle", name="Работа", text="Отвечай коротко и по делу.")["circle"]
    assert (owned["text_author"], composed["text_author"]) == ("owner", "ouroboros")
    assert call(focused, "set_circle", place_id=place.place_id, circle_id=owned["circle_id"])["place"]["circle"][
        "placed_by"] == "ouroboros"
    inspected = call(focused, "inspect", place_id=place.place_id)
    assert inspected["circle"]["text"] == "Друзья: про работу молчи." and inspected["circle"]["standing_delegation"] is False
    assert call(door, "set_circle", place_id=place.place_id, circle_id=owned["circle_id"])["place"]["circle"][
        "placed_by"] == "owner"
    assert call(door, "inspect", place_id=place.place_id)["circle"]["standing_delegation"] is True
    described = call(door, "set_description", place_id=place.place_id, description="Чат друзей")["place"]
    assert described["description"]["author"] == "owner"
    assert call(focused, "set_description", place_id=place.place_id, description="Друзья и быт")["place"][
        "description"]["author"] == "ouroboros"
    listed = call(focused, "list")
    assert [c["name"] for c in listed["circles"]] == ["Друзья", "Работа"]
    assert listed["places"][0]["circle"] == {key: listed["places"][0]["circle"][key] for key in (
        "circle_id", "placed_by", "placed_at")}  # the words once, in the circles list
    stale = _configure_presence(focused, "edit_circle", "", circle_id=owned["circle_id"], text="x", expected_revision=0)
    assert stale.startswith("ERROR: Current revision") and p.place_records(root).circle(owned["circle_id"])["text"] != "x"
    gone = call(focused, "delete_circle", circle_id=composed["circle_id"])
    assert gone["circle"]["text"] == "Отвечай коротко и по делу." and gone["detached"] == []
    assert p.get_place(root, place.place_id).state == place.state  # no circle act ever switches a place


def test_author_lending_and_stop(focus):
    from ouroboros.conversation_lease import PresenceTurnGate, live_author, live_authors
    root, _place, _admission, event = focus
    gate = PresenceTurnGate(max_active=2, state_root=root / "state")
    lease = gate.acquire(event.conversation_key)
    try:
        lease.bind_author("author")
        assert live_author(event.conversation_key, drive_root=root) == "author"
        assert live_authors(drive_root=root) == ["author"]
        assert lease.lend()
        assert not live_author(event.conversation_key, drive_root=root)
        assert live_authors(drive_root=root) == ["author"]
        assert not lease.reacquire(lambda: True)
        assert lease.reacquire(lambda: False)
        assert live_author(event.conversation_key, drive_root=root) == "author"
    finally:
        lease.release()
    assert not live_authors(drive_root=root) and not live_author(event.conversation_key, drive_root=root)


def test_place_arrival_is_context_only_and_deduplicated(focus):
    from ouroboros.place_intake import offer_to_author
    from ouroboros.conversation_lease import PresenceTurnGate
    from ouroboros.owner_mailbox import _mailbox_path, CONTEXT_ONLY_TASK_PROVENANCES, deliver_task_message
    root, _place, admission, event = focus
    task = capture_presence_event(admission, event, drive_root=root)
    assert not offer_to_author(root, task, event)
    lease = PresenceTurnGate(state_root=root / "state").acquire(event.conversation_key)
    try:
        lease.bind_author("another-author")
        import threading
        lease.intake_agent = SimpleNamespace(_owner_message_admission_lock=threading.RLock(),
            _accepting_owner_messages=True, _busy=True, _current_task_id="another-author")
        assert offer_to_author(root, task, event)
        assert offer_to_author(root, task, event)
        rows = [json.loads(line) for line in _mailbox_path(root, "another-author").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1 and rows[0]["provenance"] in CONTEXT_ONLY_TASK_PROVENANCES
        messages = []
        deliver_task_message(rows[0], "another-author", None, messages.append)
        assert "never an owner directive" in messages[0]
    finally:
        lease.release()


def test_late_receipt_warns_only_when_destination_closed(focus):
    from ouroboros.place_reply import record_late_delivery
    from ouroboros import dialogue_store
    root, place, _admission, _event = focus
    receipt = {"state": "delivered", "provider": "chat", "account_id": "a", "conversation_id": "r", "thread_id": "", "text": "Sent"}
    record_late_delivery(root, "bridge", receipt, "receipt1")
    assert not list(dialogue_store.iter_rows(root, "chat"))
    p.edit_place(root, place.place_id, "set_state", author="owner", state="paused")
    record_late_delivery(root, "bridge", receipt, "receipt1")
    record_late_delivery(root, "bridge", receipt, "receipt1")
    rows = [r.row for r in dialogue_store.iter_rows(root, "chat")]
    assert len(rows) == 1 and rows[0]["direction"] == "system" and "Paused" in rows[0]["text"]


def test_retry_reuses_retained_attachments(focus):
    root, _place, admission, event = focus
    source = root / "incoming.txt"
    source.write_text("original bytes")
    first = capture_presence_event(admission, event, drive_root=root, staged_files=[source])
    source.write_text("changed bytes")
    retry = capture_presence_event(admission, event, drive_root=root, staged_files=[source])
    assert retry["attachment_manifest"] == first["attachment_manifest"]


def test_author_projection_uses_the_cross_process_lock(tmp_path):
    import subprocess
    import sys
    from ouroboros.conversation_lease import live_author, live_authors
    code = '''
import sys
from pathlib import Path
from ouroboros.conversation_lease import PresenceTurnGate
lease = PresenceTurnGate(state_root=Path(sys.argv[1]) / "state").acquire("room")
lease.bind_author("other-process-author")
print("held", flush=True)
sys.stdin.readline()
lease.release()
'''
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        assert live_author("room", drive_root=tmp_path) == "other-process-author"
        assert live_authors(drive_root=tmp_path) == ["other-process-author"]
        child.communicate("release\n", timeout=10)
        assert child.returncode == 0
        assert not live_author("room", drive_root=tmp_path)
        assert not live_authors(drive_root=tmp_path)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


@pytest.mark.parametrize("silent,promote", [(False, False), (True, False), (False, True)])
def test_focus_ordinary_finish_traverses_reply_check(focus, monkeypatch, silent, promote):
    import queue
    from ouroboros import loop
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.presence_runner import build_presence_result_event
    root, _place, _admission, _event = focus
    task = task_for(focus)
    if promote:
        from tests.test_place_reply_revision import promoted
        task = promoted(task)
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry._ctx.task_metadata = task["metadata"]
    registry._ctx.task_contract = task["task_contract"]
    registry._ctx.is_direct_chat = not promote
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr("ouroboros.presence_context.confirmed_delivery_receipts", lambda _ctx: frozenset())
    checks, calls = [], []
    def check(name, arguments, **kw):
        if name != "place_reply":
            return True, ""
        checks.append(arguments)
        return len(checks) > 1, "Reconsider this reply."
    monkeypatch.setattr("ouroboros.safety.check_safety", check)
    def respond(_llm, messages, *_args, **_kwargs):
        calls.append(messages.copy())
        assert len(calls) <= 2
        return {"content": None, "tool_calls": [{"id": "finish" + str(len(calls)), "type": "function",
            "function": {"name": "finish_task", "arguments": json.dumps({"action": "finish", **({"silent": True} if silent else {"answer": "Hello"})})}}]}, 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    text, usage, _trace = loop.run_llm_loop([{"role": "user", "content": "Hello"}], registry,
        SimpleNamespace(default_model=lambda: "test-model"), root / "logs", lambda *_a, **_k: None,
        queue.Queue(), task_id=task["id"], drive_root=root)
    # The pipeline passes the terminal's own quiet fact (emit_task_results), never a selection the ctx kept.
    result = build_presence_result_event(task, text, registry._ctx, terminal_origin=usage["terminal_origin"],
                                         silent=usage.get("wake_silent") is True)
    if silent:
        assert len(calls) == 1 and not checks
        assert result["outcome"] == "silent" and result["text"] == ""
    else:
        assert len(calls) == 2 and len(checks) == 2
        assert any("Reconsider this reply" in str(row.get("content")) for row in calls[-1])
        assert result["outcome"] == "message" and result["text"] == "Hello"


@pytest.mark.parametrize("state,facts,enabled,available,expected", [
    ("active", ["mention"], True, True, "message"),
    ("paused", ["mention"], True, True, "silent"),
    ("active", ["mention"], True, False, "retry"),
])
def test_new_host_payload_captures_before_attention(focus, monkeypatch, state, facts, enabled, available, expected):
    from starlette.testclient import TestClient
    from ouroboros.gateway import host_service
    from ouroboros import dialogue_store
    root, place, _admission, event = focus
    p.edit_place(root, place.place_id, "set_state", author="owner", state=state)
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _r: enabled)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _r: {
        "status": "available" if available else "exhausted", "remaining_usd": 2 if available else 0})
    async def authenticated(*_args):
        return "bridge", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    calls = []
    def run(**kw):
        rows = list(dialogue_store.iter_rows(root, "chat"))
        assert len(rows) == 1 and rows[0].row["text"] == event.text
        calls.append(kw)
        return SimpleNamespace(outcome="message", text="Hello", task_id="focus", work_ref="")
    body = {"event": {k: v for k, v in event.__dict__.items() if k not in {"delivery_reporting_version", "continuation_version", "identity"}}}
    body["event"]["facts"] = facts
    with TestClient(host_service.create_host_service_app(root, presence_runner=run)) as client:
        response = client.post("/presence/turn", json=body)
        if expected == "retry":
            assert response.status_code == 429
            assert response.json()["code"] == "consciousness_allowance_exhausted"
            assert response.json()["disposition"] == "retry" and response.json()["retry_after"] > 0
        else:
            assert response.status_code == 200, response.text
            assert response.json()["outcome"] == expected
        assert bool(calls) == (expected == "message")
        rows = list(dialogue_store.iter_rows(root, "chat"))
        assert len(rows) == 1
        lookup = client.get("/presence/place", params={"provider": "chat", "account_id": "a", "conversation_id": "r"})
        assert lookup.status_code == 200 and lookup.json()["place_id"] == place.place_id


def test_off_retry_keeps_unknown_attempt_custody(focus, monkeypatch):
    from starlette.testclient import TestClient
    from ouroboros.gateway import host_service
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import run_presence_turn
    from ouroboros.task_results import write_task_result, load_task_result
    root, _place, admission, event = focus
    event = replace(event, conversation_key=conversation_key("chat", "a", "r", ""))
    task = capture_presence_event(admission, event, drive_root=root)
    write_task_result(root, task["id"], "running", metadata=task["metadata"])
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _r: False)
    async def authenticated(*_args):
        return "bridge", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    calls = []
    def factory(**_kw):
        calls.append(True)
        raise AssertionError("an unknown attempt must not generate again")
    def run(**kw):
        return run_presence_turn(repo_dir=root, drive_root=root, agent_factory=factory, **kw)
    body = {"event": {k: v for k, v in event.__dict__.items() if k not in {"delivery_reporting_version", "continuation_version", "identity"}}}
    with TestClient(host_service.create_host_service_app(root, presence_runner=run)) as client:
        response = client.post("/presence/turn", json=body)
    assert response.status_code == 409
    assert response.json()["code"] == "presence_attempt_outcome_unknown"
    assert not calls and load_task_result(root, task["id"])["status"] == "running"
    assert settle_heard(root, task, event) is None
    assert load_task_result(root, task["id"])["status"] == "running"


def test_owner_projection_tracks_heard_events_and_preserves_choices(focus, monkeypatch):
    from ouroboros.place_projection import places_view, connect_declared_place
    root, place, admission, event = focus
    declaration = {"id": "room", "skill": "bridge", "skill_label": "Chat bridge", "provides": "place",
                   "label": "Chat", "destination": "vendor:Chat", "scope": "account",
                   "wakes": ["mention"], "icon": "message-circle",
                   "icon_spec": {"kind": "host", "name": "message-circle"}}
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [declaration])
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window", lambda _root, **kw: {
        "spend_by_cause": [{"cause": "place:" + place.place_id, "usd": 0.25}]})
    task = capture_presence_event(admission, event, drive_root=root)
    view = places_view(root)
    card, = view["places"]
    assert card["unanswered"] is None and card["last_activity_at"]
    assert card["spent_24h_usd"] == 0.25 and card["icon"] == declaration["icon_spec"]
    assert view["connectors"][0]["connected"]
    settle_heard(root, task, event)
    assert places_view(root)["places"][0]["unanswered"] is None  # hearing creates no task store
    connected = connect_declared_place(root, "bridge", account_id="new-account")
    assert connected.provider == "chat" and connected.wakes == (p.WakeItem("mention", "connector"),)
    saved = p.edit_place(root, connected.place_id, "add_note", author="owner", text="Owner choice")
    declaration["wakes"] = ["any_message"]
    assert connect_declared_place(root, "bridge", account_id="new-account") == saved


def test_focus_work_facts_use_ordinary_readers(tmp_path, monkeypatch):
    from ouroboros.presence_context import _own_work_section
    monkeypatch.setattr("ouroboros.tools.recent_tasks.recent_tasks_page", lambda *_a, **_kw: {
        "tasks": [{"task_id": "report", "status": "running", "description": "Prepare the report"}],
        "remaining": 2, "offset": 0, "returned": 1, "snapshot": "snapshot"})
    legacy = {"binding_id": "binding", "event": {"conversation_key": "room"}}
    assert "presence_scope" in _own_work_section(tmp_path, legacy, "current")
    text = _own_work_section(tmp_path, {**legacy, "place": {"place_id": "place"}}, "current")
    assert "report [running]" in text and "get_task_result(task_id)" in text
    assert "presence_scope" not in text and "presence_cancel_work" not in text
    assert "not addressable" not in text


def test_author_projection_cleanup_failure_still_releases_the_gate(tmp_path, monkeypatch):
    from ouroboros.conversation_lease import PresenceTurnGate, live_author, live_authors
    gate = PresenceTurnGate(1, state_root=tmp_path / "state")
    lease = gate.acquire("room")
    lease.bind_author("author")
    assert live_author("room", drive_root=tmp_path) == "author"
    resources = lease._resources
    def unavailable():
        raise OSError("projection cleanup unavailable")
    monkeypatch.setattr(lease, "_clear_author", unavailable)
    try:
        with pytest.raises(OSError, match="projection cleanup"):
            lease.release()
        assert live_author("room", drive_root=tmp_path) == ""
        assert live_authors(drive_root=tmp_path) == []
        with gate.acquire_resources("room", lambda: False):
            pass
    finally:
        resources.close()


# --- the frame states the connector's facts, never a verdict about whom the message is for --------------------------


BOT = {"sends_as": {"id": "B1", "label": "Bot", "kind": "bot"}, "reads_as": []}


def _frame(focus, *, facts, message=None, conversation=None, identity=BOT):
    root, _place, admission, event = focus
    event = replace(event, facts=facts, message=message or {}, conversation=conversation or {}, identity=identity)
    task = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2}), event, drive_root=root, staged_files=())
    from ouroboros.presence_context import frame_focus_content
    return task, frame_focus_content(task, event.text)


@pytest.mark.parametrize("facts,message,identity,expect,absent", [
    # The I1 shape: a reply in a bot-rooted thread that mentions someone else. The verdict is gone; the facts stay.
    (("any_message", "reply_to_me", "my_thread"), {"provider_facts": {"self_user_id": "B1", "mentioned_user_ids": ["U2"]}}, BOT,
     ["The connector observed: a message in a watched conversation; it replies to a message you wrote; it is in a thread you started or wrote in.",
      "It mentions: id U2 (name not supplied; a connector read tool can resolve it) — not you.", "through Bot (B1)."], []),
    # The honest shape after the hub fix: thread membership only.
    (("any_message", "my_thread"), {"provider_facts": {"self_user_id": "B1", "mentioned_user_ids": ["U2"]}}, BOT,
     ["a message in a watched conversation; it is in a thread you started or wrote in."], ["replies to a message you wrote"]),
    # A real mention still says so; connector-resolved names and its own is_self mark are rendered as given.
    (("mention",), {"mentions": [{"id": "B1", "name": "Bot", "is_self": True}, {"id": "U2", "name": "Ann", "is_self": False}]}, BOT,
     ["The connector observed: it mentions your account.", "It mentions: Bot (id B1) — you; Ann (id U2) — not you."], []),
    # No credential identity: the connector's own account id still tells you who you are here.
    (("any_message",), {"provider_facts": {"self_user_id": "B1", "mentioned_user_ids": []}}, None,
     ["Here your account is B1.", "It mentions no one by platform id."], ["You write here"]),
    # A connector that reports no mentions at all: nothing is invented about them.
    (("direct_message",), {}, BOT, ["a direct message to your account."], ["It mentions"]),
])
def test_focus_frame_states_provider_facts_and_mentions_never_a_verdict(focus, facts, message, identity, expect, absent):
    task, rendered = _frame(focus, facts=facts, message=message, identity=identity)
    assert "addressed" not in task["metadata"]["focus"]  # the host computes no verdict about whom a message is for
    for words in ("addressed you", "This turn is yours", "wake list"):
        assert words not in rendered, words
    assert ("A wake is your attention, not an obligation: answer, act or stay silent. An answer may be words, "
            "or a gesture such as a reaction where this place offers one.") in rendered
    for words in expect:
        assert words in rendered, words
    for words in absent:
        assert words not in rendered, words


@pytest.mark.parametrize("shape", ["slack", "generic", "directory"])
def test_focus_frame_renders_the_connector_thread_snapshot_with_marks_and_a_named_omission(focus, monkeypatch, shape):
    from ouroboros import config
    from ouroboros.presence_context import focus_frame_sources
    identity, authors = BOT, {}
    if shape == "slack":
        rows = [{"user": "B1", "ts": "1700000000.1", "bot_profile": {"name": "Bot"}, "text": "root " * 10},
                {"user": "U2", "ts": "1700000001.2", "user_profile": {"real_name": "Ann"}, "text": "a reply " * 10},
                {"user": "U3", "ts": "1700000002.3", "user_profile": {"real_name": "Pat"}, "text": "the waking words"}]
        message = {"message_id": "1700000002.3", "sent_at": "2023-11-14T22:13:22+00:00"}
    elif shape == "generic":
        rows = [{"id": "m1", "actor_id": "B1", "author_name": "Bot", "text": "root " * 10},
                {"id": "m2", "actor_id": "U2", "display_name": "Ann", "text": "a reply " * 10},
                {"id": "m3", "actor_id": "U3", "author": {"display_name": "Pat"}, "text": "the waking words"}]
        message = {"message_id": "m3"}
    else:  # names and the you-mark come from the connector's author directory alone (no identity, bare ids)
        rows = [{"id": "m1", "actor_id": "B1", "text": "root " * 10}, {"id": "m2", "actor_id": "U2", "text": "a reply " * 10},
                {"id": "m3", "actor_id": "U3", "text": "the waking words"}]
        message = {"message_id": "m3", "provider_facts": {"mentioned_user_ids": ["U2", "B1"]}}
        identity, authors = None, {"authors": [{"id": "B1", "name": "Bot", "is_self": True}, {"id": "U2", "name": "Ann", "is_self": False},
                                               {"id": "U3", "name": "Pat", "is_self": False}]}
    conversation = {"thread_history": {"status": "available", "observed_at": "2023-11-14T22:13:23+00:00", **authors,
                                       "data": {"messages": rows, "complete": True}}}
    _task, bare = _frame(focus, facts=("my_thread",), message=message, conversation={}, identity=identity)
    assert "The thread as the connector captured it" not in bare  # no snapshot, no thread section
    task, rendered = _frame(focus, facts=("my_thread",), message=message, conversation=conversation, identity=identity)
    if shape == "directory":  # the directory also names mentioned ids and marks you when provider_facts give only ids
        assert "It mentions: Ann (id U2) — not you; Bot (id B1) — you." in rendered
    thread = rendered.split("The thread as the connector captured it (3 message(s), observed at 2023-11-14T22:13:23+00:00):\n", 1)[1]
    lines = thread.split("\n\nRespond or stay silent first", 1)[0].splitlines()
    assert lines[0].startswith("- Bot (B1) [you] at ") and lines[0].endswith(json.dumps("root " * 10))
    assert lines[1].startswith("- Ann (U2) at ") and "[you]" not in lines[1]
    assert lines[2].startswith("- Pat (U3) [this message] at ") and lines[2].endswith('"the waking words"')
    assert (" at 2023-11-14T22:13:22Z: " in lines[2]) is (shape == "slack")  # a numeric epoch is a time; else unknown
    assert ("earlier message(s)" not in rendered) and ("incomplete" not in rendered)
    # The typed records behind the frame reproduce it byte for byte, snapshot included (the Supervisor's reading).
    assert focus_frame_sources(task["metadata"], rendered)["event"]["conversation"] == conversation
    # The snapshot renders whole, however long: no default cut set to save room (BIBLE P1); a cut is the window's alone.
    long_rows = [{"id": f"m{index}", "actor_id": "U2", "author_name": "Ann", "text": "a line of the thread " * 20}
                 for index in range(300)]
    conversation["thread_history"]["data"]["messages"] = [*long_rows, *rows]
    _task, whole = _frame(focus, facts=("my_thread",), message=message, conversation=conversation, identity=identity)
    assert whole.count("- Ann (U2) at ") >= 300 and "(303 message(s)" in whole and "not shown here" not in whole
    assert not hasattr(config, "FOCUS_THREAD_SNAPSHOT_CHARS")
    conversation["thread_history"]["data"]["messages"] = rows
    conversation["thread_history"]["data"]["complete"] = False
    assert "3 message(s), the connector reports it incomplete, observed at" in _frame(
        focus, facts=("my_thread",), message=message, conversation=conversation, identity=identity)[1]


# --- a refused automatic reply returns with the facts the mind decides on --------------------------------------------


def test_a_refused_focus_reply_returns_destination_sends_and_silence(tmp_path):
    from ouroboros.place_reply import refusal_facts
    from ouroboros.utils import append_jsonl
    key = "chat:a:r:0"
    presence = {"place_id": "p1", "delivery_reporting_version": 1, "place": {"label": "Team chat"},
                "event": {"conversation_key": key, "provider": "chat", "account_id": "a", "conversation_id": "r", "thread_id": ""}}
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="focus", task_metadata={"presence": presence, "focus": {"place_id": "p1"}},
                          _place_message_calls=[{"tool": "ext_send", "at": "2026-11-06T02:18:50+00:00", "conversation_keys": [key],
                                                 "text": "queued"}])
    for row in ({"task_id": "focus", "direction": "in", "text": "hi"},
                {"task_id": "focus", "type": "presence_delivery", "text": "I replied",
                 "transport": {"conversation_key": key, "delivery": {"state": "delivered", "delivery_id": "d", "part_id": "0"}}}):
        append_jsonl(tmp_path / "logs" / "chat.jsonl", row)
    note = refusal_facts(ctx)
    assert note.startswith("\n[FOCUS_REPLY_REFUSED] This answer was not posted.") and "in Team chat (route " in note
    assert '"conversation_id": "r"' in note and 'Confirmed sent there by this turn: "I replied".' in note
    assert 'Your own send calls there without a receipt yet: ext_send at 2026-11-06T02:18:50+00:00: "queued".' in note
    assert 'Silence is valid: finish_task(action="finish", silent=true) keeps the answer as your internal note' in note
    assert refusal_facts(SimpleNamespace(task_metadata={})) == ""  # no reply carrier: no destination, no note


# --- an order exists only where the mind cites my human's exact words and the host verifies their provenance -------


@pytest.mark.parametrize("cite,expected_kind,destination", [
    ("Проверь, пожалуйста, экономику", "owner", "place"), ("", "consciousness", "place"),
    ("Проверь экономику", "refused", "place"),  # a paraphrase of their words is no cite
    ("<@U2> look into this", "refused", "place"),  # a colleague's bytes in the same conversation
    ("Проверь, пожалуйста, экономику", "private", "owner"),  # a private owner start names owner_quote (#2795)
])
def test_start_task_from_a_place_cites_or_starts_its_own_work(focus, monkeypatch, cite, expected_kind, destination):
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.tools import control_routing
    from ouroboros.owner_words import PLACE_CONFIRMED
    root, place, admission, event = focus
    place = p.edit_place(root, place.place_id, "confirm_owner_account", author="owner", platform_actor_id="person")
    words = "Проверь, пожалуйста, экономику"
    task = _build_task(replace(admission, place=place, allowance={"status": "available", "remaining_usd": 2}),
                       replace(event, text=words), drive_root=root, staged_files=())
    assert task["metadata"]["focus"]["actor"]["owner_confirmed"] is True
    ctx = SimpleNamespace(drive_root=root, task_id=task["id"], task_metadata=task["metadata"], task_contract={},
                          current_chat_id=task["chat_id"], _owner_directives=[])
    _initialize_owner_directives(ctx, [{"role": "user", "content": "[FOCUS]"}])
    assert [(row["source"], row["content"], row.get("carried")) for row in ctx._owner_directives] == [
        (PLACE_CONFIRMED, words, True), ("initial_text", "[FOCUS]", None)]
    # a colleague's line in the same thread reached the run too: its bytes are not my human's
    ctx._owner_directives.append({"source": "place_event_context", "content": "<@U2> look into this"})
    emitted = []
    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing", lambda _ctx, evt: emitted.append(evt) or ("live", {"status": "scheduled"}))
    monkeypatch.setattr(control_routing, "_finish_swarm_handoff", lambda _ctx, _evt, response, **_kw: response)
    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    result = control_routing._start_task(ctx, objective="Recheck the economy", title="Economy", destination=destination,
                                         owner_words=cite)
    if expected_kind == "refused":
        assert result.startswith("⚠️ OWNER_WORDS_UNVERIFIED (start_task): owner_words were not found verbatim") and not emitted
        return
    if expected_kind == "private":
        assert result.startswith("⚠️ TOOL_ARG_ERROR (start_task): owner_words cites your human's words") and not emitted
        return
    evt = emitted[0]
    assert evt["work_kind"] == expected_kind
    cited = [row for row in evt.get("owner_corpus") or [] if row.get("source") == "place_owner"]
    if expected_kind == "owner":
        assert cited == [{"source": "place_owner", "content": words, "cited": cite, "basis": "cited", "carried": True,
                          "msg_id": cited[0]["msg_id"]}]  # the whole verified message, the span named
        assert "Started as your human's work on their cited words." in result and "consciousness_cause" not in evt
    else:
        assert cited == [] and evt["consciousness_cause"] == "started"
        assert "Started as your own work (consciousness kind): it draws on the consciousness allowance, and the Off switch stops it." in result
    # the citing run's own corpus never gains the row (its owner_source_sha256 must not move mid-batch)
    assert [row["source"] for row in ctx._owner_directives] == [PLACE_CONFIRMED, "initial_text", "place_event_context"]


# --- the allowance is counts with a reader; the speak line comes from the connector's declarations -------------------


def test_focus_frame_folds_the_allowance_to_counts_behind_an_exact_reader(focus, monkeypatch):
    root, _place, admission, event = focus
    allowance = {"status": "available", "limit_usd": 300.0, "settled_usd": 103.6, "remaining_usd": 196.4,
                 "resets_at": "2026-11-06T10:16:01+00:00", "roots": [f"presence-{index:024x}" for index in range(300)],
                 "spend_by_cause": [{"cause": "own", "usd": 14.1}], "window_rows": 1926}
    task = _build_task(replace(admission, allowance=allowance), event, drive_root=root, staged_files=())
    from ouroboros.presence_context import frame_focus_content
    rendered = frame_focus_content(task, "hello")
    line = next(line for line in rendered.splitlines() if line.startswith("- allowance (last 24 h): "))
    folded = json.loads(line.split(": ", 1)[1])
    assert folded == {"status": "available", "limit_usd": 300.0, "settled_usd": 103.6, "remaining_usd": 196.4,
                      "resets_at": "2026-11-06T10:16:01+00:00", "counted_roots": 300,
                      "root_ids": f'query_data(op="json", kind="task_result", ids=["{task["id"]}"], select="metadata.focus_allowance.roots")'}
    assert "presence-000000" not in rendered and "counted_roots" not in task["metadata"]  # ids live in the record the reader names
    assert task["metadata"]["focus_allowance"]["roots"] == allowance["roots"]


def _publish_bridge_tools(monkeypatch, *declarations):
    """The focus's own connector ``bridge`` with these declared tools, in its registration order, as the dispatcher's
    published registry holds them; their surface names."""
    from ouroboros import extension_registry_state as registry
    from ouroboros.extension_surface_names import extension_surface_name
    manifest = SimpleNamespace(raw_extra={"tools": list(declarations)})
    bundle = registry._ExtensionRegistrations(generation_digest="g1", loaded_skill=SimpleNamespace(manifest=manifest))
    names = []
    for declaration in declarations:
        names.append(extension_surface_name("bridge", declaration["name"]))
        monkeypatch.setitem(registry._tools, names[-1], {"skill": "bridge", "extension_generation": "g1"})
    monkeypatch.setitem(registry._extensions, "bridge", bundle)
    return names


def _publish_bridge_send(monkeypatch, **declaration):
    """One declared tool ``send`` of the connector ``bridge``."""
    return _publish_bridge_tools(monkeypatch, {"name": "send", **declaration})[0]


@pytest.mark.parametrize("declaration,records", [
    ({"effect": "message", "target": ["channel"], "conversation": "channel", "thread": "thread_ts"}, True),
    ({"effect": "message", "target": ["account_id", "chat_id"], "conversation": "chat_id"}, True),
    ({"effect": "message", "target": ["url"], "conversation": "context_id", "thread": "task_id"}, False),  # a peer URL
    ({"effect": "message", "target": ["account_id", "to", "cc", "bcc"], "conversation": "conversation_id"}, False),  # mail
    ({"effect": "write", "target": ["channel"], "conversation": "channel"}, False),
])
def test_speak_line_and_records_only_follow_the_connectors_declarations(focus, monkeypatch, declaration, records):
    from ouroboros.place_reply import records_only, speak_line
    from ouroboros.presence_context import frame_focus_content
    root, place, admission, event = focus
    name = _publish_bridge_send(monkeypatch, **declaration)
    task = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2}),
                       replace(event, thread_id="1712.5", conversation_key="chat:a:r:1712.5"), drive_root=root, staged_files=())
    rendered, line = frame_focus_content(task, "hello"), speak_line(task["metadata"])
    assert records_only(task["metadata"]) is records and line in rendered
    if records:
        thread = f", {declaration['thread']}=1712.5" if declaration.get("thread") else ""
        account = ", account_id=a" if "account_id" in declaration["target"] else ""
        assert line == (f"Words reach chat only when you send them, through {name} ({declaration['conversation']}=r{thread}{account}); "
                        "a send may ride in the same response as your finish. Your finish is your record, never posted.")
        # a Silent place or Advises autonomy keep the held draft, and the line says so
        task["metadata"]["presence"]["place"]["state"] = "silent"
        assert "(chat is Silent: a send is refused and your finish is held for your human, not posted)" in speak_line(task["metadata"])
        task["metadata"]["presence"]["place"]["state"] = "active"
        task["metadata"]["consciousness_autonomy"] = "observe"
        assert "(autonomy is Advises: a send is refused and your finish is held for your human, not posted)" in speak_line(task["metadata"])
    else:
        assert line.startswith("Your non-silent finish is posted in chat as your words")
    assert "bridge" not in line.replace(name, ""), "no connector is named by the host; the tool's own name is"


def test_speak_line_names_every_declared_door_with_its_own_arguments(focus, monkeypatch):
    """Review R1 MAJOR-1, the Slack shape: the connector registers its file upload before its send and declares an
    edit too. The line lists each door with its own arguments, in the connector's order; the host picks none (the first
    registered tool was a false door: the upload wants a file, and its channel argument is not the send's)."""
    from ouroboros.place_reply import conversation_send_tools, speak_line
    root, _place, admission, event = focus
    upload, send, edit = _publish_bridge_tools(monkeypatch,
        {"name": "file_upload", "effect": "message", "target": ["channel_id", "thread_ts"], "conversation": "channel_id", "thread": "thread_ts"},
        {"name": "send", "effect": "message", "target": ["channel_or_user", "thread_ts"], "conversation": "channel_or_user", "thread": "thread_ts"},
        {"name": "message_edit", "effect": "message", "target": ["channel"], "conversation": "channel"})
    task = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2}),
                       replace(event, thread_id="1712.5", conversation_key="chat:a:r:1712.5"), drive_root=root, staged_files=())
    assert [name for name, _effect in conversation_send_tools("bridge")] == [upload, send, edit]
    line = speak_line(task["metadata"])
    assert line == (f"Words reach chat only when you send them, through one of {upload} (channel_id=r, thread_ts=1712.5); "
                    f"{send} (channel_or_user=r, thread_ts=1712.5); {edit} (channel=r); a send may ride in the same response "
                    "as your finish. Your finish is your record, never posted.")
    from ouroboros.presence_context import frame_focus_content
    assert line in frame_focus_content(task, "hello"), "the frame carries the same line"


_REACT = {"name": "react", "effect": "gesture", "target": ["channel"], "conversation": "channel", "message": "ts"}


@pytest.mark.parametrize("message,prefilled", [
    # Only the connector's normalized ``message_id`` is the waking message; a raw ``ts``/``id`` is never guessed.
    ({"message_id": "1712.9", "id": "x-9", "ts": "1700.0"}, "1712.9"),
    ({"id": "x-9", "ts": "1700.0"}, ""),
])
def test_a_gesture_only_connector_keeps_posting_its_finish_and_says_how_a_gesture_stands_alone(
        focus, monkeypatch, message, prefilled):
    """Astra (rl-…181404) critical: with no send door the non-silent finish IS posted (``records_only`` is False and
    the completion is a ``message``); the line says so, names the gesture door and the silent finish for a reaction alone."""
    from ouroboros.place_reply import prepare_focus_reply, records_only, speak_line
    root, _place, admission, event = focus
    [name] = _publish_bridge_tools(monkeypatch, _REACT)
    task = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2}),
                       replace(event, message=message), drive_root=root, staged_files=())
    assert records_only(task["metadata"]) is False
    line = speak_line(task["metadata"])
    target = (" A gesture is a response, not speech. Its prefilled message id is the waking message; another message's "
              "exact id from this conversation is valid too." if prefilled else
              " A gesture is a response, not speech. This event carries no id for the waking message: name an exact "
              "message id you know from this conversation.")
    assert line == (f"Your non-silent finish is posted in chat as your words. A gesture reaches it only when you call it, "
                    f"through gesture: {name} (channel=r, ts={prefilled or '…'}); to answer with the gesture alone, "
                    f"finish with silent=true.{target}")
    assert "1700.0" not in line and "x-9" not in line
    ctx = SimpleNamespace(task_metadata=task["metadata"], task_id=task["id"], drive_root=root)
    monkeypatch.setattr("ouroboros.tool_access.canonical_data_root", lambda _ctx: root)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **kw: (True, ""))
    assert prepare_focus_reply(ctx, "Hello")[0] and ctx._presence_completion["outcome"] == "message", (
        "the line above promised a posted finish")


def test_speak_line_lists_words_and_gesture_doors_and_a_gesture_door_must_name_its_message(focus, monkeypatch):
    from ouroboros.place_reply import conversation_gesture_tools, records_only, speak_line
    root, _place, admission, event = focus
    send, react, _undeclared = _publish_bridge_tools(monkeypatch,
        {"name": "send", "effect": "message", "target": ["channel"], "conversation": "channel", "thread": "thread_ts"},
        _REACT, {"name": "pin", "effect": "gesture", "target": ["channel"], "conversation": "channel"})
    assert [name for name, _effect in conversation_gesture_tools("bridge")] == [react], "no message argument, no door"
    task = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2}),
                       replace(event, thread_id="1712.5", conversation_key="chat:a:r:1712.5",
                               message={"message_id": "1712.9"}), drive_root=root, staged_files=())
    assert records_only(task["metadata"]) is True
    assert speak_line(task["metadata"]) == (
        f"Words or a gesture reach chat only when you call its door, through one of words: {send} (channel=r, "
        f"thread_ts=1712.5); gesture: {react} (channel=r, ts=1712.9); either may ride in the same response as your "
        "finish. Your finish is your record, never posted. A gesture is a response, not speech. Its prefilled message "
        "id is the waking message; another message's exact id from this conversation is valid too.")
    task["metadata"]["presence"]["place"]["state"] = "silent"
    assert "(chat is Silent: a send or gesture is refused and your finish is held" in speak_line(task["metadata"])


# --- a place turn reads any project journal; writes and a child's reads stay forbidden (owner 06.11, 4=A) ----------


def test_a_presence_focus_reads_a_foreign_project_journal_but_never_writes_it(focus):
    from ouroboros.tools.project_journal import _resolve_project_id
    task = task_for(focus)
    speaker = SimpleNamespace(task_id=task["id"], project_id="", drive_root=focus[0], task_metadata=task["metadata"],
                              task_contract={}, _presence_turn=True)
    assert _resolve_project_id(speaker, "nova", write=False) == ("nova", "")
    assert "TOOL_FORBIDDEN" in _resolve_project_id(speaker, "nova", write=True)[1]
    speaker.project_id = "own"  # bound to a room: another room still reads, never writes
    assert _resolve_project_id(speaker, "nova", write=False) == ("nova", "")
    assert "TOOL_FORBIDDEN" in _resolve_project_id(speaker, "nova", write=True)[1]
    child = SimpleNamespace(task_id="child", project_id="", drive_root=focus[0], task_contract={},
                            task_metadata={"parent_task_id": "root-1", "delegation_role": "subagent"})
    assert "TOOL_FORBIDDEN" in _resolve_project_id(child, "nova", write=False)[1]


@pytest.mark.parametrize("place_model,circle_model,model,layer", [
    ("place/chosen", "circle/chosen", "place/chosen", "place"),
    ("", "circle/chosen", "circle/chosen", "circle"),
    ("", "", None, None),
    ("place/chosen", None, "place/chosen", "place"),
    ("", None, None, None),
])
def test_focus_model_inherits_the_admitted_scope(focus, monkeypatch, place_model, circle_model, model, layer):
    from ouroboros.presence_context import frame_focus_content
    root, place, _, event = focus
    p.edit_place(root, place.place_id, "set_model", author="owner", model=place_model)
    if circle_model is not None:
        circle = p.create_circle(root, name="Team", text="Shared words", model=circle_model, author="owner")
        p.edit_place(root, place.place_id, "set_circle", author="owner", circle_id=circle["circle_id"])
    admission = admit_presence_turn(drive_root=root, authenticated_transport_skill="bridge", binding_id=place.place_id)
    if circle_model is not None:
        p.edit_circle(root, circle["circle_id"], author="owner", model="circle/later")
    task = task_for((root, admission.place, admission, event))
    metadata = task["metadata"]
    assert metadata["model_role"] == "consciousness"
    assert metadata.get("lead") == ({"model": model, "source": layer, "use_local": False} if model else None)
    if model is None:
        assert "model" not in metadata and "model_layer" not in metadata
    rendered = frame_focus_content(task, event.text)
    assert "circle/chosen" not in rendered and "circle/later" not in rendered
    if circle_model is not None:
        assert "Shared words" in rendered


@pytest.mark.parametrize("layer", ["place", "circle"])
@pytest.mark.parametrize("alias", ["main", "light"])
@pytest.mark.parametrize("local", [False, True])
def test_focus_aliases_remain_admission_snapshots(focus, monkeypatch, layer, alias, local):
    root, place, _, event = focus
    settings = {"main": "main/first", "light": "light/first", "local": local}
    monkeypatch.setattr("ouroboros.model_slots._main_model", lambda: settings["main"])
    monkeypatch.setattr("ouroboros.config.get_light_model", lambda: settings["light"])
    from ouroboros.config import runtime_setting
    monkeypatch.setattr("ouroboros.config.runtime_setting", lambda key, default="": (
        str(settings["local"]).lower() if key in {"USE_LOCAL_MAIN", "USE_LOCAL_LIGHT"} else runtime_setting(key, default)))
    circle = p.create_circle(root, name="Team", model=alias if layer == "circle" else "circle/model", author="owner")
    p.edit_place(root, place.place_id, "set_circle", author="owner", circle_id=circle["circle_id"])
    if layer == "place":
        p.edit_place(root, place.place_id, "set_model", author="owner", model=alias)
    admission = admit_presence_turn(drive_root=root, authenticated_transport_skill="bridge", binding_id=place.place_id)
    frozen = task_for((root, admission.place, admission, event))["metadata"]
    settings.update(main="main/next", light="light/next", local=not local)
    current = task_for((root, admission.place, admission, event))["metadata"]
    assert frozen["lead"]["model"] == alias + "/first" and current["lead"]["model"] == alias + "/next"
    assert frozen["lead"]["source"] == current["lead"]["source"] == layer
    assert frozen["lead"]["use_local"] is local and current["lead"]["use_local"] is not local
    records = p.place_records(root)
    assert (records.places[place.place_id].model if layer == "place" else records.circle(circle["circle_id"])["model"]) == alias


@pytest.mark.parametrize('layer', ['place', 'circle', 'consciousness'])
def test_an_explicit_place_model_has_its_own_lead_options_and_initial_effort(focus, monkeypatch, layer):
    from ouroboros.agent_dispatch import _initial_effort_for
    from ouroboros.model_slots import task_model_binding, model_role_option, MODEL_CONTEXT_WINDOWS_KEY, task_processing_preference
    from ouroboros.route_choice import bind_task_route, open_route
    from ouroboros.presence_context import focus_metadata
    from ouroboros.tools.registry import ToolContext

    root, place, admission, event = focus
    model = 'claudexor::cursor=chosen-model-high'
    monkeypatch.setenv('OUROBOROS_MODEL_CONSCIOUSNESS', 'claudexor::cursor=conscious-model-low')
    monkeypatch.setenv('OUROBOROS_MODEL_ACCOUNTS', json.dumps({'consciousness': 'chosen-account'}))
    monkeypatch.setenv('OUROBOROS_MODEL_CONTEXT_WINDOWS', json.dumps({'consciousness': 123456}))
    monkeypatch.setenv('OUROBOROS_MODEL_PROCESSING_PREFERENCES', json.dumps({'consciousness': 'fast'}))
    admission = replace(admission, place=replace(place, model=model if layer == 'place' else ''),
                        circle={'model': model} if layer == 'circle' else None)
    metadata = focus_metadata(admission, event, 'turn', 1, root)
    task = {'metadata': metadata, '_presence_turn': True}
    ctx = ToolContext(repo_dir=root, drive_root=root, task_metadata=metadata)
    bind_task_route(ctx, metadata, presence=True)
    role, pin = task_model_binding(task)
    acting, local, _choice = open_route(ctx, None)
    if layer == 'consciousness':
        assert role == 'consciousness' and pin is None
        assert model_role_option('OUROBOROS_MODEL_ACCOUNTS', role) == 'chosen-account'
        assert model_role_option(MODEL_CONTEXT_WINDOWS_KEY, role) == 123456
        assert task_processing_preference(task) == 'fast'
        assert acting == 'claudexor::cursor=conscious-model-low'
        assert _initial_effort_for(task, 'presence') == 'low'
    else:
        assert role == 'lead' and pin == ''
        assert model_role_option(MODEL_CONTEXT_WINDOWS_KEY, role) == 0
        assert task_processing_preference(task) == ''
        assert acting == model and not local
        assert metadata['lead']['source'] == layer
        assert _initial_effort_for(task, 'presence') == 'high'


def test_a_place_lead_keeps_the_wakes_effort_range_when_its_model_has_no_named_level(focus, monkeypatch):
    from ouroboros.agent_dispatch import _initial_effort_for
    from ouroboros.tools.control_runtime import _effort_switch

    root, place, admission, event = focus
    monkeypatch.setenv('OUROBOROS_EFFORT_TASK', 'medium')
    monkeypatch.setenv('OUROBOROS_EFFORT_MAX', 'ultra')
    monkeypatch.setenv('OUROBOROS_MODEL_CONSCIOUSNESS', 'claudexor::cursor=other-model-low')
    task = task_for((root, place, replace(admission, place=replace(place, model='vendor/own-model')), event))
    assert _initial_effort_for(task, 'task') == 'ultra'
    ctx = SimpleNamespace(task_metadata=task['metadata'], current_task_type='task', active_effort='ultra')
    assert _effort_switch(ctx, 'high')[0] == 'high'


@pytest.mark.parametrize('alias', ['main', 'light'])
@pytest.mark.parametrize('local', [False, True])
def test_place_model_aliases_keep_the_resolved_locality_on_the_lead(focus, monkeypatch, alias, local):
    from ouroboros.presence_context import focus_metadata
    from ouroboros.route_choice import bind_task_route, open_route
    from ouroboros.tools.registry import ToolContext

    root, place, admission, event = focus
    monkeypatch.setenv('OUROBOROS_MODEL' + ('_LIGHT' if alias == 'light' else ''), 'chosen-model')
    monkeypatch.setenv('USE_LOCAL_' + alias.upper(), 'true' if local else 'false')
    metadata = focus_metadata(replace(admission, place=replace(place, model=alias)), event, 'turn', 1, root)
    ctx = ToolContext(repo_dir=root, drive_root=root, task_metadata=metadata)
    bind_task_route(ctx, metadata, presence=True)
    assert metadata['model_role'] == 'consciousness'
    assert open_route(ctx, None)[:2] == ('chosen-model', local)

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
    assert meta["root_cost_ceiling_usd"] <= 2 and "configure_presence" in meta["disabled_tools"]
    template = Path("prompts/FOCUS.md").read_text(encoding="utf-8")
    assert {key for _lit, key, _fmt, _conv in string.Formatter().parse(template) if key} == set(PLACEHOLDERS)
    rendered = frame_focus_content(task, "The untrusted event")
    assert "The untrusted event" in rendered and "Already started: task1" in rendered
    assert "connector" in rendered and "folder" in rendered.lower()


@pytest.mark.parametrize("state,facts,enabled,allowance,decision", [
    ("active", ("mention",), True, "available", "focus"),
    ("silent", ("mention",), True, "available", "focus"),
    ("paused", ("mention",), True, "available", "silent"),
    ("active", (), True, "available", "silent"),
    ("active", ("mention",), False, "available", "silent"),
    ("active", ("mention",), True, "exhausted", "retry"),
    ("active", ("mention",), True, "allowance_unknown", "retry"),
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
    ("active", ["mention"], False, True, "silent"),
    ("active", [], True, True, "silent"),
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

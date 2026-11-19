"""REV1 speech regressions, adapted from Astra's executable review probes."""
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import dialogue_store, places, presence_runner as runner
from ouroboros.gateway import host_service
from ouroboros.place_reply import prepare_focus_reply
from ouroboros.presence_continuation import replay_envelope, work_view, turn_response
from ouroboros.presence_recovery import retained_work_view
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_place_focus import focus as focus, task_for


def promoted(parent):
    from ouroboros.consciousness_authority import consciousness_origin_metadata
    from supervisor.workers import _apply_presence_promotion_authority
    task = {"id": "promoted", "metadata": consciousness_origin_metadata(
        {**parent["metadata"], "task_id": parent["id"]}, started_root=True)}
    _apply_presence_promotion_authority(
        {"presence": parent["metadata"]["presence"], "task_contract": parent["task_contract"]},
        task, objective="Do long work", expected_output="A reply")
    assert task["_presence_origin"] and "wake_cause" not in task["metadata"]
    return task


@pytest.mark.parametrize(("legacy", "continuing", "state"), [
    (False, True, "disconnected"),
    (True, False, "silent"),
    (True, True, "paused"),
])
def test_cached_reply_uses_current_state_without_changing_answer(focus, monkeypatch, state, continuing, legacy):
    root, place, admission, event = focus
    task = runner.capture_presence_event(admission, event, drive_root=root)
    metadata = {**task["metadata"], "presence_outcome": "message", "presence_result_text": "Old unsent reply",
                "presence_output_ref": "same-output"}
    if legacy:
        metadata["presence"] = {key: value for key, value in metadata["presence"].items() if key != "place_id"}
        metadata.pop("focus", None)
    identity = metadata["presence_event_identity"]
    continuation = {"event_identity": identity, "initial": {"outcome": "deferred", "text": "Old unsent reply",
                    "output_ref": "same-output", "work_ref": "child"},
                    "outputs": [{"outcome": "message", "text": "Old unsent reply", "output_ref": "same-output"}]}
    write_task_result(root, task["id"], "completed", metadata=metadata, chat_id=task["chat_id"],
                      result="Old unsent reply", terminal_origin="model_final",
                      **({"presence_continuation": continuation} if continuing else {}))
    original = load_task_result(root, task["id"])
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: pytest.fail("replay bought a check"))
    async def authenticated(*_args):
        return "bridge", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    def replay(behind):
        monkeypatch.setattr(runner, "_pointer_behind", lambda *_: behind)
        return runner.run_presence_turn(admission=admission, event=event, repo_dir=root, drive_root=root,
                                       agent_factory=lambda **kw: pytest.fail("replay generated again"))
    with TestClient(host_service.create_host_service_app(root)) as client:
        for restricted in (False, True):
            if restricted:
                places.edit_place(root, place.place_id, "disconnect" if state == "disconnected" else "set_state",
                                  author="owner", **({} if state == "disconnected" else {"state": state}))
            for behind in (False, True):
                result = replay(behind)
                assert result.text == ("" if restricted else "Old unsent reply")
                assert result.outcome == ("silent" if restricted else "message")
                assert result.output_ref == "same-output"
                assert turn_response(result, 1, root)["text"] == result.text
            body = client.get("/presence/work/" + task["id"]).json()
            assert body["text"] == ("" if restricted else "Old unsent reply")
            assert body["outcome"] == ("silent" if restricted else "message")
            if continuing:
                envelope = replay_envelope(root, task["id"], identity)
                assert envelope.text == ("" if restricted else "Old unsent reply")
                assert envelope.output_ref == "same-output" and envelope.work_ref == "child"
                for stored in (original, {**original, "status": "running"}):
                    for _status, poll in (work_view(stored, task["id"], root), retained_work_view(stored, task["id"], root)):
                        assert poll["outputs"][0]["text"] == ("" if restricted else "Old unsent reply")
                        assert poll["outputs"][0]["output_ref"] == "same-output"
        assert load_task_result(root, task["id"]) == original
    notices = [r.row for r in dialogue_store.iter_rows(root, "chat") if r.row.get("type") == "place_notice"]
    assert len(notices) == 1 and "Reply held:" in notices[0]["text"]
    assert notices[0]["held_text"] == "Old unsent reply"
    assert notices[0]["held"] is True and notices[0]["checked"] is False


@pytest.mark.parametrize(("level", "as_root"), [("observe", False), ("act", True)])
def test_reply_advises_hold_and_acts_check_follow_the_carrier(focus, monkeypatch, as_root, level):
    root, place, _, _ = focus
    parent = task_for(focus)
    parent["metadata"]["consciousness_autonomy"] = level
    task = promoted(parent) if as_root else parent
    ctx = SimpleNamespace(task_metadata=task["metadata"], task_id=task["id"], drive_root=root)
    checks = []
    def check(*args, **kwargs):
        checks.append(args[0])
        return (True, "") if len(checks) > 1 else (False, "Reconsider this reply")
    monkeypatch.setattr("ouroboros.safety.check_safety", check)
    monkeypatch.setattr("ouroboros.presence_context.confirmed_delivery_receipts", lambda _: frozenset())
    if level == "act":
        assert prepare_focus_reply(ctx, "Deferred public answer") == (False, "Reconsider this reply")
        assert prepare_focus_reply(ctx, "Deferred public answer")[0]
        assert checks == ["place_reply", "place_reply"]
    else:
        assert prepare_focus_reply(ctx, "Deferred public answer")[0]
        assert ctx._focus_reply_hold == "Reply held: autonomy is Advises" and not checks
    result = runner.build_presence_result_event(task, "Deferred public answer", ctx, terminal_origin="model_final")
    assert result["outcome"] == ("silent" if level == "observe" else "message")
    assert result["text"] == ("" if level == "observe" else "Deferred public answer")
    assert len(checks) == (0 if level == "observe" else 2)  # projection cannot buy a second check
    notices = [r.row for r in dialogue_store.iter_rows(root, "chat") if r.row.get("type") == "place_notice"]
    if level == "observe":
        notice, = notices
        assert notice["held_text"] == "Deferred public answer"
        assert notice["held"] is True and notice["checked"] is False
    else:
        assert notices == []
    places.edit_place(root, place.place_id, "set_state", author="owner", state="silent")
    result = runner.build_presence_result_event(task, "Another answer", ctx, terminal_origin="model_final")
    assert result["outcome"] == "silent" and not result["text"]
    assert len(checks) == (0 if level == "observe" else 2)


@pytest.mark.parametrize("state,level,held", [("silent", "act", True), ("active", "observe", True), ("active", "act", False)])
def test_reply_hold_retains_complete_text_before_safety(focus, monkeypatch, state, level, held):
    root, place, _, _ = focus
    places.edit_place(root, place.place_id, "set_state", author="owner", state=state)
    task = task_for(focus)
    task["metadata"]["consciousness_autonomy"] = level
    ctx = SimpleNamespace(task_metadata=task["metadata"], task_id=task["id"], drive_root=root)
    checks = []
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (checks.append(a) or True, ""))
    monkeypatch.setattr("ouroboros.presence_context.confirmed_delivery_receipts", lambda _: frozenset())
    text = "Full authored reply café\n" * 400
    assert prepare_focus_reply(ctx, text) == (True, "")
    assert len(checks) == (0 if held else 1)
    rows = [r.row for r in dialogue_store.iter_rows(root, "chat")]
    if held:
        notice, = rows
        assert notice["held_text"] == text and notice["held"] is True and notice["checked"] is False
        assert notice["text"] == (f"Reply held: {place.label} is Silent" if state == "silent"
                                  else "Reply held: autonomy is Advises")
    else:
        assert rows == []


def test_conditioned_reply_reaches_current_consumers_without_ordinary_speech(tmp_path, monkeypatch):
    from ouroboros.place_reply import freeze_reply
    from ouroboros.presence_bindings import conversation_key

    place = places.connect_place(tmp_path, connector="telegram-bot", provider="telegram",
                                 account_id="9", conversation_id="55")
    places.edit_place(tmp_path, place.place_id, "confirm_owner_account", author="owner", platform_actor_id="55")
    key = conversation_key("telegram", "9", "55", "")
    destination = {"transport": "telegram", "account_id": "9", "conversation_id": "55", "thread_id": ""}
    event = runner.PresenceTurnEvent("fixture-event", "telegram", "9", "55", "", key, {}, {},
                                    {"reply_targets": []}, "Fixture input")
    identity, task_id = "fixture-identity", "fixture-turn"
    metadata = {"presence_event_identity": identity, "presence": {
        "binding_id": place.place_id, "place_id": place.place_id, "transport_skill": "telegram-bot",
        "direct_reply_key": key, "direct_reply_version": 2,
        "event": {"provider": "telegram", "account_id": "9", "conversation_id": "55", "thread_id": "",
                  "conversation_key": key, "destination": destination, "source_event_id": event.source_event_id,
                  "message": dict(event.message)}}}
    task = {"id": task_id, "chat_id": runner.presence_chat_id(key), "metadata": metadata}
    ctx = SimpleNamespace(task_metadata=metadata, task_id=task_id, drive_root=tmp_path)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    words = "Fixture reply"
    row = runner.build_presence_result_event(task, words, ctx, terminal_origin="model_final")
    frozen = row["conditional_reply"]
    condition = frozen["owner_confirmation"]
    assert (row["outcome"], row["text"], row["message"]) == ("silent", "", "")
    assert frozen == {"outcome": "message", "text": words, "owner_confirmation": condition}
    assert condition["place_id"] == condition["delivery_place"]["place_id"] == place.place_id
    assert metadata["presence_conditional_reply"] == frozen and "presence_owner_confirmation" not in metadata
    stored = write_task_result(tmp_path, task_id, "completed", metadata=metadata, result=words,
                               terminal_origin="model_final", chat_id=task["chat_id"])
    assert (stored["result"], stored["metadata"]["presence_result_text"]) == (words, "")
    result = runner.finish_turn(tmp_path, task, task_id, identity, event, [row], own_start=0)
    assert result == runner.presence_result_from_stored(stored, task_id, tmp_path)
    assert (result.outcome, result.text, result.owner_confirmation) == ("message", words, condition)
    without_policy = runner.presence_result_from_stored(stored, task_id)
    assert (without_policy.outcome, without_policy.text) == ("silent", "")
    assert runner._read_previous_turn(tmp_path, key)["message"] == ""
    assert not [entry for entry in dialogue_store.iter_rows(tmp_path, "chat") if entry.row.get("direction") == "out"]

    initial = {**freeze_reply("deferred", words, condition), "output_ref": "fixture-output", "work_ref": "child"}
    # The current early producer writes empty text; this is the reader's retained-output shape.
    record = {"event_identity": identity, "initial": initial, "outputs": [initial]}
    stored = write_task_result(tmp_path, task_id, "completed", presence_continuation=record)
    replayed = replay_envelope(tmp_path, task_id, identity)
    assert (replayed.outcome, replayed.text, replayed.owner_confirmation) == ("deferred", words, condition)
    status, view = work_view(stored, task_id, tmp_path)
    output, = view["outputs"]
    assert status == 200 and output == {"outcome": "deferred", "text": words, "owner_confirmation": condition,
                                        "output_ref": "fixture-output", "work_ref": "child"}
    assert load_task_result(tmp_path, task_id)["presence_continuation"] == record

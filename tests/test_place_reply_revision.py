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


@pytest.mark.parametrize("state", ["silent", "paused", "disconnected"])
@pytest.mark.parametrize("continuing", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
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


@pytest.mark.parametrize("as_root", [False, True])
@pytest.mark.parametrize("level", ["observe", "act"])
def test_reply_advises_hold_and_acts_check_follow_the_carrier(focus, monkeypatch, as_root, level):
    root, place, _, _ = focus
    parent = task_for(focus)
    parent["metadata"]["consciousness_autonomy"] = level
    task = promoted(parent) if as_root else parent
    ctx = SimpleNamespace(task_metadata=task["metadata"], task_id=task["id"], drive_root=root)
    checks = []
    def check(*args, **kwargs):
        checks.append(args[0])
        return len(checks) > 1, "Reconsider this reply"
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

"""Concurrent offers of one captured event keep its one recipient, even after mailbox retirement.

Admission precedes the first claim inside the mailbox transaction. A refused offer cannot erase a concurrent
successful claim. A stale candidate loses to a later mind decision. Tests pause offers outside every lock and
exercise both orders through the real Host, focus and tell_task, including actual worker consumption and cleanup.
"""
from __future__ import annotations

import threading

import pytest

from ouroboros.conversation_lease import close_author_intake, open_root_intake
from tests.test_place_conversation_work import (
    SILENT, _completed, _focus_ctx, _recipient, attention, call, event_of, place_rows, post, reading, until,
)
from tests.test_place_conversation_work import host as host  # fixture

pytestmark = pytest.mark.usefixtures("provider_catalog_offline")


def _in_offer(monkeypatch):
    """Whether this thread is inside the Host's ``offer_arrival`` (its census, not another reader)."""
    from ouroboros import place_intake

    real, context = place_intake.offer_arrival, threading.local()

    def offering(*args, **kwargs):  # its own re-offers nest
        context.depth = getattr(context, "depth", 0) + 1
        try:
            return real(*args, **kwargs)
        finally:
            context.depth -= 1

    monkeypatch.setattr(place_intake, "offer_arrival", offering)
    return lambda: getattr(context, "depth", 0) > 0


def _first_census_closes(monkeypatch, world, work, offering):
    """The census sees ``work`` read; its final then closes before the append's own read (``offering()`` says whose)."""
    from ouroboros import room_reader

    real, closed = room_reader.work_reads, []

    def reads_then_loop_closes(row, key):
        answer = real(row, key)
        if offering() and not closed:
            closed.append(threading.get_ident())
            assert close_author_intake(world.data, work["id"])
        return answer

    monkeypatch.setattr(room_reader, "work_reads", reads_then_loop_closes)
    return closed


def _pause_after_refusal(monkeypatch, whose, refused, resume):
    """``whose()``'s first refused append returns, then waits outside every lock: the race window."""
    from ouroboros import owner_mailbox

    real = owner_mailbox.write_task_message

    def refused_then_waits(*args, **kwargs):
        written = real(*args, **kwargs)
        if whose() and not written and not refused.is_set():
            refused.set()
            assert resume.wait(30)
        return written

    monkeypatch.setattr(owner_mailbox, "write_task_message", refused_then_waits)


def _offer_thread(world, captured, event, result, name):
    from ouroboros import place_intake

    def body():
        try:
            result[name] = place_intake.offer_arrival(world.data, captured, event, wake=True)
        except BaseException as exc:  # asserted below
            result[name] = exc

    return threading.Thread(target=body, name=name)


def test_a_retry_landing_before_a_refusal_returns_keeps_that_one_holder(host, monkeypatch):
    from ouroboros import place_arrivals, place_intake

    world = host()
    first = reading(world, objective="First work")
    event = event_of(world, "racing-event", text="Exactly one recipient", facts=("any_message",))
    ctx, captured = _focus_ctx(world, event)
    original = lambda: threading.current_thread().name == "original"  # noqa: E731
    refused, resume, result = threading.Event(), threading.Event(), {}
    _first_census_closes(monkeypatch, world, first, original)
    _pause_after_refusal(monkeypatch, original, refused, resume)
    thread = _offer_thread(world, captured, event, result, "original")
    thread.start()
    try:
        assert refused.wait(30), result
        assert open_root_intake(world.data, first)  # its loop reads again (a reopened final), and a retry arrives
        assert place_intake.offer_arrival(world.data, captured, event, wake=True) == (
            place_intake.HANDED_TO_WORK, first["id"])
    finally:
        resume.set()
        thread.join(30)
    assert not thread.is_alive() and result["original"] == (place_intake.HANDED_TO_WORK, first["id"]), result
    assert place_arrivals.arrival_recipient(world.data, captured["id"]) == {
        "to": "work", "task_id": first["id"], "drive": str(world.data)}
    _completed(world, first)
    second = reading(world, objective="Second work")
    assert place_intake.hand_own_arrival(world.data, ctx.task_metadata, second["id"], focus_task_id=ctx.task_id) == (
        "arrival_held_by:" + first["id"])
    assert [len(place_rows(world.data, work["id"])) for work in (first, second)] == [1, 0]


def test_a_retry_with_an_earlier_candidate_lands_nowhere_after_the_mind_takes_the_event(host, monkeypatch):
    from ouroboros import owner_mailbox, place_arrivals, place_intake

    world = host()
    first = reading(world, objective="First work")
    event = event_of(world, "racing-event-2", text="Exactly one recipient", facts=("any_message",))
    ctx, captured = _focus_ctx(world, event)
    name = lambda: threading.current_thread().name  # noqa: E731
    refused, resume, read, append = (threading.Event() for _ in range(4))
    result = {}
    _first_census_closes(monkeypatch, world, first, lambda: name() == "original")
    _pause_after_refusal(monkeypatch, lambda: name() == "original", refused, resume)
    real_write = owner_mailbox.write_task_message

    def retry_reads_then_waits(*args, **kwargs):  # census chose first; its atomic admission/claim has not run
        if name() == "retry" and not read.is_set():
            read.set()
            assert append.wait(30)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(owner_mailbox, "write_task_message", retry_reads_then_waits)
    original, retry = (_offer_thread(world, captured, event, result, which) for which in ("original", "retry"))
    original.start()
    try:
        assert refused.wait(30), result
        assert open_root_intake(world.data, first)
        retry.start()
        assert read.wait(30), result
        assert close_author_intake(world.data, first["id"])  # the original's route changes while the retry waits
        resume.set()
        original.join(30)
        assert result["original"] == (place_intake.ROUTE_TO_MIND, ""), result
        assert open_root_intake(world.data, first)  # reading again as the stale retry appends: it must still land nowhere
    finally:
        resume.set()
        append.set()
        original.join(30)
        retry.join(30)
    assert not original.is_alive() and not retry.is_alive()
    assert result["retry"] == (place_intake.ROUTE_TO_MIND, ""), result
    assert place_rows(world.data, first["id"]) == []
    assert place_arrivals.arrival_recipient(world.data, captured["id"]) == {"to": "mind", "why": "not_reading"}
    second = reading(world, objective="Second work")
    assert place_intake.hand_own_arrival(world.data, ctx.task_metadata, second["id"], focus_task_id=ctx.task_id) == ""
    assert [len(place_rows(world.data, work["id"])) for work in (first, second)] == [0, 1]


def test_two_http_requests_of_one_event_reach_one_work_and_no_focus_hands_it_to_another(host, monkeypatch):
    from ouroboros import place_intake

    world = host()
    first = reading(world, objective="First work")
    settle = place_intake.settle_heard
    closed = _first_census_closes(monkeypatch, world, first, _in_offer(monkeypatch))
    refused, resume, allow_settlement, routed = (threading.Event() for _ in range(4))
    settling, result = [], {}
    _pause_after_refusal(monkeypatch, lambda: closed and threading.get_ident() == closed[0], refused, resume)

    def settles_with_first_wait(*args, **kwargs):
        if kwargs.get("recipient") == first["id"]:
            settling.append(threading.get_ident())
            assert allow_settlement.wait(30)
        return settle(*args, **kwargs)

    monkeypatch.setattr(place_intake, "settle_heard", settles_with_first_wait)

    def request(which):
        try:
            result[which] = post(world, 9900, text="One source, one holder.")
        except BaseException as exc:  # asserted below
            result[which] = exc

    def after_tell(messages):
        result["focus"] = str(messages)
        routed.set()
        return SILENT

    original, retry = (threading.Thread(target=request, args=(which,)) for which in ("original", "retry"))
    original.start()
    try:
        assert refused.wait(30), result
        assert open_root_intake(world.data, first)
        retry.start()
        until(lambda: len(settling) == 1, "the retry's delivery to the first work")
        assert len(place_rows(world.data, first["id"])) == 1
        second = reading(world, objective="Second work")
        world.focus_steps[:] = [call(("tell_task", {"task_id": second["id"], "arrival": True})), after_tell]
        resume.set()  # the refused original returns concurrently with the retry's settlement
        until(lambda: len(settling) == 2 or routed.is_set(), "the original's outcome")
    finally:
        resume.set()
        allow_settlement.set()
        original.join(30)
        if retry.ident is not None:
            retry.join(30)
    assert not original.is_alive() and not retry.is_alive()
    assert [result[which][1].status_code for which in ("original", "retry")] == [200, 200], result
    source = result["original"][0]
    assert not routed.is_set() and world.focus_inputs == [], "no focus started for a message its work holds"
    assert [len(place_rows(world.data, work["id"])) for work in (first, second)] == [1, 0]
    assert attention(world, source)["recipient"] == first["id"]
    assert _recipient(world, source)["task_id"] == first["id"]


def test_an_http_retry_with_an_earlier_candidate_lands_nowhere_while_the_focus_tells_one_work(host, monkeypatch):
    from ouroboros import owner_mailbox

    world = host()
    first = reading(world, objective="First work")
    offering = _in_offer(monkeypatch)
    closed = _first_census_closes(monkeypatch, world, first, offering)
    refused, resume, read, append, routed = (threading.Event() for _ in range(5))
    result = {}
    _pause_after_refusal(monkeypatch, lambda: closed and threading.get_ident() == closed[0], refused, resume)
    real_write = owner_mailbox.write_task_message

    def retry_reads_then_waits(*args, **kwargs):  # wait before the transaction; never block its real mail lock
        if offering() and closed and threading.get_ident() != closed[0] and not read.is_set():
            read.set()
            assert append.wait(60)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(owner_mailbox, "write_task_message", retry_reads_then_waits)

    def request(which):
        try:
            result[which] = post(world, 9940, text="One source, one holder.")
        except BaseException as exc:  # asserted below
            result[which] = exc

    def after_tell(messages):
        result["focus"] = str(messages)
        routed.set()
        return SILENT

    original, retry = (threading.Thread(target=request, args=(which,)) for which in ("original", "retry"))
    original.start()
    try:
        assert refused.wait(30), result
        assert open_root_intake(world.data, first)
        retry.start()
        assert read.wait(30), result
        second = reading(world, objective="Second work")
        world.focus_steps[:] = [call(("tell_task", {"task_id": second["id"], "arrival": True})), after_tell]
        assert close_author_intake(world.data, first["id"])  # routed to the mind while first reads not
        resume.set()
        until(lambda: routed.is_set(), "the focus handing the message to the second work")
        until(lambda: not world.client.app.state.host_service_context.presence_turns.live(), "the focus to end")
        assert open_root_intake(world.data, first)  # first reads again as the stale retry appends
    finally:
        resume.set()
        append.set()
        original.join(60)
        if retry.ident is not None:
            retry.join(60)
    assert not original.is_alive() and not retry.is_alive()
    assert [result[which][1].status_code for which in ("original", "retry")] == [200, 200], result
    assert f"handed to {second['id']} exactly as captured" in result["focus"], "the real tell_task result"
    assert len(world.focus_inputs) == 2, "one focus, two model calls"
    assert [len(place_rows(world.data, work["id"])) for work in (first, second)] == [0, 1]
    assert _recipient(world, result["original"][0])["task_id"] == second["id"]


def test_a_retry_consumed_and_retired_before_the_refusal_returns_keeps_its_recipient(host, monkeypatch):
    """Mailbox absence after normal retirement is not permission to reassign the immutable event."""
    from types import SimpleNamespace
    from ouroboros import place_intake
    from ouroboros.conversation_lease import author_intake_accepting
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.post_task_checkpoint import set_root_post_task_checkpoint
    from ouroboros.task_results import load_task_result
    from supervisor.terminal_delivery import cleanup_settled_owner_mailbox
    from tests.test_place_conversation_work import dispatched, finish, promote, publish, run_root

    world = host()
    first = dispatched(world, promote(world, objective="First work"))
    second = dispatched(world, promote(world, objective="Second work preparing"))
    waited, routed = threading.Event(), threading.Event()
    seen, result = {}, {}

    def wait(_messages):
        waited.set()
        return call(("await_messages", {"timeout_sec": 50}))

    def read_then_finish(messages):
        seen["first_model"] = str(messages)
        return finish("First work considered the followup.")

    box = run_root(world, first, wait, read_then_finish)
    assert waited.wait(30)
    key = first["metadata"]["presence"]["event"]["conversation_key"]
    until(lambda: author_intake_accepting(world.data, first["id"], key), "real first loop's intake")
    closed = _first_census_closes(monkeypatch, world, first, _in_offer(monkeypatch))
    refused, resume, appended, allow_settlement = (threading.Event() for _ in range(4))
    _pause_after_refusal(monkeypatch, lambda: closed and threading.get_ident() == closed[0], refused, resume)
    real_settle = place_intake.settle_heard

    def pause_retry_settlement(*args, **kwargs):
        if kwargs.get("recipient") == first["id"] and threading.get_ident() != closed[0]:
            appended.set()
            assert allow_settlement.wait(30)
        return real_settle(*args, **kwargs)

    monkeypatch.setattr(place_intake, "settle_heard", pause_retry_settlement)

    def request(which):
        try:
            result[which] = post(world, 9990, text="One source must have one recipient.")
        except BaseException as exc:
            result[which] = exc

    def after_tell(messages):
        result["focus"] = str(messages)
        routed.set()
        return SILENT

    original, retry = (threading.Thread(target=request, args=(which,)) for which in ("original", "retry"))
    original.start()
    try:
        assert refused.wait(30), result
        assert open_root_intake(world.data, first)
        world.focus_steps[:] = [call(("tell_task", {"task_id": second["id"], "arrival": True})), after_tell]
        retry.start()
        assert appended.wait(30), result
        until(lambda: "first_model" in seen, "first work actually reading the retry's arrival")
        assert "One source must have one recipient." in seen["first_model"]
        until(lambda: (load_task_result(world.data, first["id"]) or {}).get("status") == "completed", "first final")
        box["thread"].join(30)
        assert "error" not in box
        world.queue.RUNNING.pop(first["id"], None)
        publish(world)
        before = place_rows(world.data, first["id"])
        assert len(before) == 1
        settled = set_root_post_task_checkpoint(SimpleNamespace(drive_root=world.data), first, "completed", strict=True)
        assert settled["root_phase_checkpoint"]["post_task_synthesis"] == "completed"
        cleanup_settled_owner_mailbox(world.data, first["id"], first)
        assert not _mailbox_path(world.data, first["id"]).exists()
        assert open_root_intake(world.data, second)
        resume.set()
        original.join(30)
    finally:
        resume.set()
        allow_settlement.set()
        original.join(30)
        if retry.ident is not None:
            retry.join(30)
    assert not original.is_alive() and not retry.is_alive()
    assert [result[which][1].status_code for which in ("original", "retry")] == [200, 200], result
    until(lambda: not world.client.app.state.host_service_context.presence_turns.live(), "fallback focus to end")
    assert place_rows(world.data, second["id"]) == [], "retirement must not permit a second delivery"
    assert _recipient(world, result["original"][0])["task_id"] == first["id"]
    if routed.is_set():
        assert "arrival_held_by:" + first["id"] in result["focus"]


def test_slow_concurrent_capture_never_makes_host_accept_after_final_drain(host, monkeypatch):
    from tests.test_place_conversation_work import dispatched, finish, promote, run_root
    from ouroboros import place_arrivals, place_intake, place_reply, presence_runner
    from ouroboros.conversation_lease import author_intake_accepting
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.task_results import load_task_result

    world = host()
    work = dispatched(world, promote(world))
    key = work["metadata"]["presence"]["event"]["conversation_key"]
    capture_event = event_of(world, "concurrent-attachment-capture", text="This capture is staging its attachment.")
    admission = admit_presence_turn(drive_root=world.data, authenticated_transport_skill=world.skill,
                                    provider=world.provider, account_id=world.account,
                                    conversation_id="C1", thread_id="1.0")
    upload = world.data / "slow-capture.txt"
    upload.write_text("a provider attachment", encoding="utf-8")
    model_entered, finish_model, offer_ready, staging_entered, release_staging, claim_entered, final_drained, release_final = (
        threading.Event() for _ in range(8)
    )
    observed, outcome = [], {}

    def model(messages):
        observed.append(str(messages))
        model_entered.set()
        assert finish_model.wait(30)
        return finish("Ready to finish.")

    real_reply = place_reply.prepare_focus_reply

    def after_final_drain(ctx, *args, **kwargs):
        if ctx.task_id == work["id"]:
            final_drained.set()
            assert release_final.wait(30)
        return real_reply(ctx, *args, **kwargs)

    monkeypatch.setattr(place_reply, "prepare_focus_reply", after_final_drain)
    real_stage = presence_runner.stage_task_attachments

    def slow_stage(*args, **kwargs):
        if threading.current_thread().name == "concurrent-capture":
            staging_entered.set()
            assert release_staging.wait(30)
        return real_stage(*args, **kwargs)

    monkeypatch.setattr(presence_runner, "stage_task_attachments", slow_stage)
    real_offer = place_intake.offer_arrival

    def after_captured(*args, **kwargs):
        offer_ready.set()
        assert staging_entered.wait(30)
        return real_offer(*args, **kwargs)

    monkeypatch.setattr(place_intake, "offer_arrival", after_captured)
    real_pin = place_arrivals.pin_recipient

    def pin(*args, **kwargs):
        claim_entered.set()
        return real_pin(*args, **kwargs)

    monkeypatch.setattr(place_arrivals, "pin_recipient", pin)

    def capture():
        try:
            outcome["capture"] = presence_runner.capture_presence_event(
                admission, capture_event, drive_root=world.data, staged_files=(upload,))["id"]
        except BaseException as exc:
            outcome["capture_error"] = repr(exc)

    def send():
        try:
            source, response = post(world, 9971, text="Read this before finishing.")
            outcome.update(source=source, status=response.status_code, response=response.json())
        except BaseException as exc:
            outcome["sender_error"] = repr(exc)

    box = run_root(world, work, model)
    capture_thread = threading.Thread(target=capture, name="concurrent-capture")
    sender = threading.Thread(target=send)
    try:
        assert model_entered.wait(30)
        assert author_intake_accepting(world.data, work["id"], key)
        sender.start()
        assert offer_ready.wait(30)
        capture_thread.start()
        assert staging_entered.wait(30)
        assert claim_entered.wait(30)
        finish_model.set()
        assert final_drained.wait(20), "reader never reached its final drain"
        assert not author_intake_accepting(world.data, work["id"], key)
        release_staging.set()
        sender.join(30)
        assert not sender.is_alive(), outcome
        rows = place_rows(world.data, work["id"])
        outcome.update(rows_after_final_drain=len(rows), attention=attention(world, outcome["source"]))
        release_final.set()
        box["thread"].join(30)
        until(lambda: (load_task_result(world.data, work["id"]) or {}).get("status") == "completed", "terminal")
        outcome["model_calls"] = len(observed)
        print(outcome)
        assert "error" not in box and "capture_error" not in outcome and "sender_error" not in outcome
        assert rows == [], "Host claimed a work received an arrival that appended after that work's final drain"
    finally:
        finish_model.set()
        release_staging.set()
        release_final.set()
        for thread in (capture_thread, sender, box["thread"]):
            if thread.ident is not None:
                thread.join(30)


def test_preclaim_lock_failure_does_not_pin_the_holder_named_by_tell(host, monkeypatch):
    import contextlib
    from ouroboros import task_custody
    from ouroboros.place_arrivals import arrival_recipient
    from ouroboros.tools.control_relay import _tell_task

    world = host()
    first = reading(world, objective="First recipient")
    second = reading(world, objective="Alternate recipient")
    ctx, captured = _focus_ctx(world, event_of(world, "exact-preclaim-source", text="These exact words"))
    monkeypatch.setattr("ouroboros.tool_access.canonical_data_root", lambda _ctx: world.data)
    real_lock = task_custody.task_mail_lock
    denied = []

    def deny_once(root, task_id, **kwargs):
        if task_id == first["id"] and not denied:
            denied.append(True)

            @contextlib.contextmanager
            def not_taken():
                yield False

            return not_taken()
        return real_lock(root, task_id, **kwargs)

    monkeypatch.setattr(task_custody, "task_mail_lock", deny_once)
    failed = _tell_task(ctx, first["id"], arrival=True)
    assert "TELL_UNCONFIRMED" in failed
    assert f"it stays with {first['id']} alone" not in failed
    assert "any recorded recipient remains binding" in failed
    assert arrival_recipient(world.data, captured["id"]) == {}
    assert place_rows(world.data, first["id"]) == []

    retried = _tell_task(ctx, second["id"], arrival=True)
    assert "exactly as captured" in retried
    assert arrival_recipient(world.data, captured["id"])["task_id"] == second["id"]
    assert len(place_rows(world.data, second["id"])) == 1
    print({"reported": failed, "actual_pin_after_failure": {}, "alternate_result": retried})

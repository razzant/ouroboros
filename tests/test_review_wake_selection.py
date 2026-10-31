"""D4: explicit alternatives wake a wait, never acquire review or landing authority."""
import copy
import threading
from types import SimpleNamespace

import pytest

from ouroboros import model_sleep, review_waiting as waiting
from ouroboros.task_results import write_task_result
from tests.test_model_sleep import _ctx, _mail
from tests.test_receptive_commit_review import author as author_fixture
from tests.test_receptive_commit_review import _commit, _commit_count, _git, _join, _parent, _produce, until
from tests.test_review_reference_collection import _pool

author = author_fixture


def _pin(operation="op", slot="seat", wave="wave"):
    return {"wave_id": wave, "operation_id": operation, "slot_id": slot}


def _entry(status="responded", *, published=True, additional=False):
    outcome = {"operation_id": "op", "slot_id": "seat", "outcome_id": "op:seat", "status": status,
               "additional": additional, "verdict": "PASS" if status == "responded" else ""}
    entry = SimpleNamespace(operation_id="op", event=threading.Event(),
                            actor=SimpleNamespace(review_wave_id="wave", outcome=outcome))
    if published:
        entry.event.set()
    return entry


@pytest.mark.parametrize("status", ["responded", "parse_failure", "error", "not_dispatched", "interrupted"])
def test_selected_finalized_answer_or_unavailability_wakes_and_peek_never_acks(tmp_path, status):
    ctx, entry = _ctx(tmp_path), _entry(status)
    selected = {"reviewers": [_pin()]}
    state = copy.deepcopy(selected)
    ready = model_sleep.terminal_readiness(ctx, selected, review_entries=[entry])
    assert ready and ready[0]["reason"] == "review_result"
    assert ready[0]["actor_id"] == "review:op:seat"
    retained = {"selection": selected}
    assert waiting.wake_reason(ctx, retained, entries=[entry]) == "review_result"
    assert retained["waking_actor_id"] == "review:op:seat"
    assert model_sleep.terminal_readiness(ctx, selected, review_entries=[entry]) == ready
    assert selected == state and not ctx._loop_mailbox_seen_ids
    assert not model_sleep.terminal_readiness(ctx, selected, [ready[0]["id"]], review_entries=[entry])


@pytest.mark.parametrize("case", ["unselected", "raw", "pending", "additional", "unfinalized", "other_wave", "other_operation", "other_slot"])
def test_noncanonical_or_unselected_review_does_not_wake(tmp_path, case):
    ctx, entry = _ctx(tmp_path), _entry()
    pin = _pin()
    if case == "raw":
        entry.event.clear()
    elif case == "pending":
        entry.actor.outcome["status"] = "pending"
    elif case == "additional":
        entry.actor.outcome["additional"] = True
    elif case == "unfinalized":
        entry.actor.outcome["outcome_id"] = ""
    elif case.startswith("other_"):
        key = {"other_wave": "wave_id", "other_operation": "operation_id", "other_slot": "slot_id"}[case]
        pin[key] = "different"
    chosen = {"reviewers": [] if case == "unselected" else [pin]}
    assert not model_sleep.terminal_readiness(ctx, chosen, review_entries=[entry])
    retained = {"selection": chosen, "waking_actor_id": "review:stale:seat"}
    assert not waiting.wake_reason(ctx, retained, entries=[entry])
    assert "waking_actor_id" not in retained


def test_default_is_whole_wave_and_explicit_helpers_use_readable_ids(tmp_path):
    ctx = _ctx(tmp_path)
    write_task_result(tmp_path, "child", "completed", parent_task_id=ctx.task_id)
    state = waiting.select_helpers(ctx)
    assert not waiting.ready_helpers(ctx, state)
    assert not waiting.wake_reason(ctx, state)
    selected = waiting.select_helpers(ctx, {"tasks": ["child"]})
    assert waiting.wake_reason(ctx, selected) == "task:child:completed"
    ctx._left_review_wait = selected
    resumed = waiting.select_helpers(ctx)
    assert resumed == {key: value for key, value in selected.items() if key != "waking_actor_id"}
    assert "waking_actor_id" not in resumed  # a new wait observes its own waking source


@pytest.mark.parametrize("invalid", [{"tasks": ["all"]}, {"tasks": "child"}, {"runs": ["foreign"]},
                                      {"reviewers": [""]}, {"wake_after_sec": 1}, {"wake_at": "later"}, []])
def test_invalid_selectors_refuse_before_review(tmp_path, invalid):
    ctx = _ctx(tmp_path)
    with pytest.raises(ValueError):
        waiting.select_helpers(ctx, invalid)


def test_counted_row_handles_freeze_all_asked_copies_and_exclude_additional(tmp_path, monkeypatch):
    from ouroboros import reviewer_slot_config as slots

    ctx = _ctx(tmp_path)
    ctx._current_review_retry_key = "wave"
    panel = SimpleNamespace(seats=[slots.PoolSeat(SimpleNamespace(slot_id="seat"), ("change",)),
                                  slots.PoolSeat(SimpleNamespace(slot_id="critic"), ("change",), True)])
    monkeypatch.setattr(slots, "catalog_review_row", lambda _s, name: SimpleNamespace(subagent_id={"@row": "seat"}.get(name, name)))
    waiting.select_helpers(ctx, {"reviewers": ["@row"]})
    waiting.bind_panel(ctx, panel)
    waiting.bind_roster(ctx, [{"operation_id": op, "slot_id": "seat"} for op in ("copy1", "copy2")])
    assert ctx._review_wait_selection["selection"]["reviewers"] == [_pin(op) for op in ("copy1", "copy2")]
    for name in ("critic", "missing"):
        waiting.select_helpers(ctx, {"reviewers": [name]})
        with pytest.raises(ValueError):
            waiting.bind_panel(ctx, panel)


@pytest.mark.parametrize("kind", ["owner_text", "owner_pause", "finalize_now"])
def test_controls_and_owner_input_precede_selected_review_even_behind_mail(tmp_path, kind):
    ctx = _ctx(tmp_path)
    _mail(tmp_path, ctx.task_id, "selected peer", sender="peer", msg_id="peer")
    _mail(tmp_path, ctx.task_id, "owner", kind=kind, msg_id="owner")
    chosen = {"any_mail": True, "reviewers": [_pin()]}
    assert model_sleep.wake_reason(ctx, chosen, review_entries=[_entry()]) == (
        "owner_text" if kind == "owner_text" else f"control:{kind}")
    assert ctx._loop_mailbox_seen_ids == set()
    assert not list(tmp_path.rglob("*.acks.jsonl"))


def test_responsible_question_precedes_selected_terminal_and_unselected_mail_stays_unread(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    run = SimpleNamespace(review_owned=False, settled=False, terminal_state="", pending_interaction_ids=["q"],
                          interaction_source={"ref": "question"})
    monkeypatch.setattr(model_sleep, "_question_runs", lambda _ctx: {"own": run})
    _mail(tmp_path, ctx.task_id, "unselected", sender="elsewhere")
    chosen = {"senders": ["peer"], "reviewers": [_pin()]}
    assert model_sleep.wake_reason(ctx, chosen, review_entries=[_entry()]) == "run_question:own"
    run.review_owned = True
    assert model_sleep.wake_reason(ctx, chosen, review_entries=[_entry()]) == "review_result"
    assert not model_sleep.wake_reason(ctx, {"senders": ["peer"]})
    _mail(tmp_path, ctx.task_id, "selected", sender="peer")
    assert model_sleep.wake_reason(ctx, {"senders": ["peer"]}) == "mail:peer"


@pytest.mark.parametrize("tool", ["change", "commit"])
@pytest.mark.parametrize("mode,enforcement", [("pro", "blocking"), ("pro", "advisory"), ("cyber_pro", "blocking")])
@pytest.mark.serial
def test_selected_review_returns_with_other_seat_live_and_never_lands(author, monkeypatch, tool, mode, enforcement):
    from ouroboros import config
    from ouroboros.tools.review_change import _handle_review_change

    a, output = author, {}
    _pool(monkeypatch)
    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(mode)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    if tool == "commit":
        thread = _commit(a, output, wake_on={"reviewers": ["first"]})
    else:
        _git(a.repo, "add", "change.py")

        def run():
            with _parent(a):
                output["result"] = _handle_review_change(a.ctx, root="system_repo", subject="index",
                                                          wake_on={"reviewers": ["first"]})
        thread = _produce(a, run)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    a.reviewer.gates["m1"].set()
    _join(thread)
    assert _commit_count(a) == 1 and not a.pushes
    assert not a.reviewer.launch_m2.is_set()
    assert any(row.get("reason") == "review_result" for row in a.ctx._review_detached_operations)
    wait = a.ctx._left_review_wait
    assert wait["selection"]["reviewers"][0]["slot_id"] == "first"
    assert wait["selection"]["reviewers"][0]["operation_id"]
    assert "wake_on" not in output["result"] or "TOOL_ARG_ERROR" not in output["result"]
    a.reviewer.release()
    until(lambda: wait.get("done"))
    assert sorted(a.reviewer.rounds) == ["m1", "m2"] and _commit_count(a) == 1


def test_wake_schema_is_unconditional_and_shared():
    from ouroboros.tools import review_change, git

    entries = review_change.get_tools() + git.get_tools()
    schemas = [entry.schema["parameters"]["properties"]["wake_on"] for entry in entries
               if entry.name in {"review_change", "commit_reviewed", "vcs_commit_reviewed"}]
    assert len(schemas) == 3 and all(schema == waiting.wake_on_schema() for schema in schemas)


@pytest.mark.parametrize("install", ["api", "subscription", "mixed"])
@pytest.mark.parametrize("mode,enforcement", [("pro", "advisory"), ("pro", "blocking"), ("cyber_pro", "blocking")])
@pytest.mark.parametrize("unavailable", [False, True])
def test_canonical_cas_wake_is_identical_across_installs_and_review_modes(tmp_path, monkeypatch, install, mode, enforcement, unavailable):
    from ouroboros import config
    from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.reviewer_slot_config import ComposedPanel, PoolSeat
    from ouroboros.review_seat_outcome import retain_collected_outcome
    from ouroboros.tools.commit_gate import bind_commit_wake

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    monkeypatch.setattr(config, "get_runtime_mode", lambda: mode)
    ctx = _ctx(tmp_path)
    ctx._current_review_retry_key = "wave"
    routes = {"api": [ReviewRouteKind.API_CHAT] * 2, "subscription": [ReviewRouteKind.AGENT_SESSION] * 2,
              "mixed": [ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION]}[install]
    slots = [ReviewSlot(f"seat{i}", "model", route=route) for i, route in enumerate(routes)]
    panel = ComposedPanel(tuple(PoolSeat(slot, ("result",)) for slot in slots), True, {})
    assert bind_commit_wake(ctx, {"reviewers": ["seat0", "seat1"]}, panel) is None
    waiting.bind_roster(ctx, [{"operation_id": f"op{i}", "slot_id": slot.slot_id} for i, slot in enumerate(slots)])
    state = copy.deepcopy(ctx._review_wait_selection)
    assert not waiting.ready_helpers(ctx, state)
    for i, slot in enumerate(slots):
        request = ReviewRequest(surface="multi_model_review", task_id=ctx.task_id, goal="check", resolved_wave_id="wave",
                                policy={"parts": ["result"]})
        actor = ReviewActorRecord(slot.slot_id, "model", "error" if unavailable else "ok", operation_id=f"op{i}",
                                  raw_text="" if unavailable else "[]\nNO_FINDINGS", error="unavailable" if unavailable else "")
        assert retain_collected_outcome(tmp_path, request, slot, actor)
    ready = waiting.ready_helpers(ctx, state)
    assert len(ready) == 2 and all(row["reason"] == "review_result" for row in ready)
    assert state == ctx._review_wait_selection, "peeking cannot ACK a result"
    state["seen_outcomes"] = [row["id"] for row in ready]
    restored = _ctx(tmp_path)
    restored._left_review_wait = state
    assert not waiting.ready_helpers(restored, waiting.select_helpers(restored))


def test_time_wake_is_peeked_then_consumed_only_with_the_delivery_cursor(tmp_path):
    ctx = _ctx(tmp_path)
    future = waiting.select_helpers(ctx, {"wake_at": "2999-01-01T00:00:00Z"})
    assert not waiting.wake_reason(ctx, future) and not waiting.delivery_ready(ctx, future)
    state = waiting.select_helpers(ctx, {"wake_at": "2000-01-01T00:00:00Z"})
    assert waiting.wake_reason(ctx, state) == waiting.wake_reason(ctx, state) == "timeout"
    state["seen_helpers"] = [row["id"] for row in waiting.delivery_ready(ctx, state)]
    assert not waiting.wake_reason(ctx, state) and not waiting.delivery_ready(ctx, state)


@pytest.mark.parametrize("complete", [False, True])
def test_cold_peek_requires_completed_producer_receipt(tmp_path, complete):
    from ouroboros.observability import persist_call

    ctx = _ctx(tmp_path)
    actor = _entry().actor
    persist_call(tmp_path, task_id=ctx.task_id, call_id="op_response", call_type="review",
                 payload={"message": {"content": "raw reply"}, "producer_outcome": vars(actor)},
                 manifest={"producer_complete": complete})
    assert bool(model_sleep.terminal_readiness(ctx, {"reviewers": [_pin()]})) is complete


@pytest.mark.parametrize("selected", [False, True])
def test_author_default_keeps_waiting_after_one_seat_finishes(tmp_path, monkeypatch, selected):
    from ouroboros import review_pause

    ctx = _ctx(tmp_path)
    state = waiting.select_helpers(ctx)
    if selected:
        state["selection"]["reviewers"] = [_pin()]
    monkeypatch.setattr(review_pause, "author_paused", lambda _op: False)
    entry = _entry()
    with review_pause.receptive_author(review_pause.receptive_binding(ctx, state=state)):
        assert review_pause.author_detaches(SimpleNamespace(), [entry]) is selected
    if selected:
        assert entry.author_left_for == "review_result"


@pytest.mark.parametrize("review_owned", [False, True])
def test_helper_run_selector_cannot_borrow_review_custody(tmp_path, monkeypatch, review_owned):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(model_sleep, "_owned_runs", lambda *_a, **_kw: {
        "owned": SimpleNamespace(review_owned=review_owned, settled=False, terminal_state="running")})
    if review_owned:
        with pytest.raises(ValueError, match="ordinary owned runs"):
            waiting.select_helpers(ctx, {"runs": ["owned"]})
    else:
        assert waiting.select_helpers(ctx, {"runs": ["owned"]})["selection"]["runs"] == ["owned"]


def test_review_park_rechecks_canonical_state_after_installation_without_ack(tmp_path):
    from ouroboros import owner_wait

    ctx = _ctx(tmp_path)
    write_task_result(tmp_path, "child", "running", parent_task_id=ctx.task_id)
    state = waiting.select_helpers(ctx, {"tasks": ["child"]})
    ctx._left_review_wait = {"binding": "wait", **state}
    checkpoint = {"review_binding": "wait"}
    assert owner_wait._sleep_wake(ctx, checkpoint) == ""
    write_task_result(tmp_path, "child", "failed", parent_task_id=ctx.task_id)
    assert owner_wait._sleep_wake(ctx, checkpoint) == owner_wait._sleep_wake(ctx, checkpoint) == "helper_result"
    assert ctx._left_review_wait["seen_helpers"] == []


def test_unreadable_source_cannot_hide_other_selected_review_or_control(tmp_path, monkeypatch):
    from ouroboros import observability, owner_wait

    ctx = _ctx(tmp_path)
    original = observability.read_call_payload

    def unreadable(root, *, task_id, call_id):
        if call_id.startswith("broken_"):
            raise OSError("unreadable producer")
        return original(root, task_id=task_id, call_id=call_id)
    monkeypatch.setattr(observability, "read_call_payload", unreadable)
    chosen = {"reviewers": [_pin("broken"), _pin()]}
    assert model_sleep.wake_reason(ctx, chosen, review_entries=[_entry()]) == "review_result"
    ctx._left_review_wait = {"binding": "wait", "selection": {"reviewers": [_pin("broken")]}}
    assert owner_wait._sleep_wake(ctx, {"review_binding": "wait"}) == ""
    _mail(tmp_path, ctx.task_id, "stop", kind="finalize_now")
    assert owner_wait._sleep_wake(ctx, {"review_binding": "wait"}) == "control:finalize_now"


@pytest.mark.parametrize("owner", ["task-1", "foreign"])
@pytest.mark.parametrize("status", ["pending", "error"])
def test_retained_ledger_unavailability_uses_exact_owned_canonical_outcome(tmp_path, owner, status):
    from ouroboros import review_ledger as ledger
    from tests.test_review_ledger_settlement import _record

    record = _record(tmp_path)
    canonical = _entry(status).actor.outcome
    ledger.revise_record(tmp_path, record["record_id"], lambda row: {
        **row, "review_wave_id": "wave", "rows": [{"seat_id": "seat", "operation_id": "op", "outcome": canonical}]})
    chosen = {"record_id": record["record_id"], "reviewers": [_pin()]}
    assert bool(model_sleep.terminal_readiness(_ctx(tmp_path, task_id=owner), chosen)) is (owner == "task-1" and status == "error")


@pytest.mark.parametrize("wake_on", [None, {}, {"tasks": [], "reviewers": []}, {"tasks": ["child"]}])
def test_system_report_refuses_unsupported_wake_choice_before_dispatch(tmp_path, monkeypatch, wake_on):
    from ouroboros import preflight_receipts
    from ouroboros.tools import review_change

    ctx, calls = _ctx(tmp_path), []
    write_task_result(tmp_path, "child", "running", parent_task_id=ctx.task_id)
    monkeypatch.setattr(preflight_receipts, "execution_exhibit", lambda _ctx: {"status": "unobserved"})
    def report(*_a, **_kw):
        calls.append(True)
        return {"message": "report"}
    monkeypatch.setattr(review_change, "run_review_change", report)
    text = review_change._handle_review_change(ctx, subject="system", surface="system", wake_on=wake_on)
    selected = bool((wake_on or {}).get("tasks"))
    assert bool(calls) is not selected
    assert ("TOOL_ARG_ERROR" in text and "synchronous report" in text) is selected


def test_selectors_read_the_data_root_only_for_selected_task_ids(tmp_path):
    """A wait without task ids (none, or reviewers only) never touches a data root, so a
    commit ask without wake_on cannot fail on a context that has none; selected task ids
    are still checked against the root."""
    from types import SimpleNamespace
    import pytest
    from ouroboros import model_sleep
    from ouroboros.task_results import write_task_result

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(model_sleep, "_canonical_root",
                        lambda ctx: pytest.fail("a selection without task ids read the data root"))
        assert model_sleep.selectors(SimpleNamespace())["any_mail"] is True
        assert model_sleep.selectors(SimpleNamespace(), reviewers=["seat"])["reviewer_slots"] == ["seat"]
    write_task_result(tmp_path, "known", "completed")
    ctx = SimpleNamespace(drive_root=tmp_path)
    assert model_sleep.selectors(ctx, tasks=["known"])["tasks"] == ["known"]
    with pytest.raises(ValueError, match="unknown"):
        model_sleep.selectors(ctx, tasks=["unknown"])

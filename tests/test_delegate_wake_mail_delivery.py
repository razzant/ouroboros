"""Wake text reaches the real loop once, with durable ACK after incorporation."""
import json
import queue
from dataclasses import replace

import pytest

from tests._run_wait_shared import held_run, publish, scripted_reads
from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines
from tests.test_configured_session_restore import InputReached



@pytest.mark.parametrize("origin", ["owner", "parent", "relay"])
@pytest.mark.parametrize("delivery", ["live_wait", "late_bind"])
def test_wake_mail_reaches_next_real_model_input_once(tmp_path, monkeypatch, origin, delivery):
    from ouroboros import delegate_supervision as ds, loop, loop_round_limits, working_checkpoint as wc
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message, write_task_message
    from tests.test_delegate_pending_recovery import _pending_handoff
    from tests.test_delegate_recovery import _bound
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    registry = _registry(tmp_path, monkeypatch, "nanny", 2)
    ctx = registry._ctx
    text = f"Unique {origin} correction carried by the wake"
    msg_id, run_id = "correction-mail", "bound-run"
    events, directive_calls, stamps = queue.Queue(), [], []
    record = loop._record_owner_directive
    stamp = loop_round_limits._stamp_owner_delivery

    def record_directive(*args, **kwargs):
        if kwargs.get("msg_id") == msg_id:
            directive_calls.append(kwargs)
        return record(*args, **kwargs)

    def stamp_delivery(*args, **kwargs):
        if kwargs.get("msg_id") == msg_id:
            stamps.append(kwargs)
        return stamp(*args, **kwargs)

    monkeypatch.setattr(loop, "_record_owner_directive", record_directive)
    monkeypatch.setattr(loop_round_limits, "_stamp_owner_delivery", stamp_delivery)

    def post_mail():
        if origin == "owner":
            assert write_owner_message(tmp_path, text, ctx.task_id, msg_id=msg_id)
        else:
            assert write_task_message(tmp_path, text, ctx.task_id, msg_id=msg_id,
                                      source_task_id="parent", provenance=(
                                          "ancestor_task" if origin == "parent" else "peer_via_ancestor"),
                                      relayed_from_task_id="sibling" if origin == "relay" else "")

    if delivery == "late_bind":
        _, recovery, task, _, invocation, _, _ = _pending_handoff(tmp_path, ctx.task_id)
        task["_working_recovery"] = {}
        post_mail()
        wake = {"status": "waiting_on_user", "run_id": run_id, "supervision_wake_id": "old-wake",
                "wake_events": ds._addressed_wakes(ctx, {})}
        row = recovery._read(tmp_path, ctx.task_id)
        row["pending_wake"] = {"wake_id": "old-wake", "attempt_key": "1", "payload": wake,
                               "mailbox_ids": [msg_id]}
        recovery._write(tmp_path, row)
        assert recovery.adopt_handoff(ctx, task)["status"] == "adopted"

    def wait_once(_ctx, rid, *_args, **_kwargs):
        assert delivery == "live_wait", "late binding must replay the retained wake before observing"
        post_mail()
        return json.dumps({"status": "terminal", "run_id": rid, "state": "succeeded"})

    # Only the one addressed read is replaced. Registry publication, the wait's readiness (the run's terminal the body
    # recorded), result delivery, next-round drain and checkpoint/ACK are production paths.
    scripted_reads(monkeypatch, wait_once)
    if delivery == "live_wait":
        held_run(tmp_path, run_id, ctx.task_id, ended="succeeded")
    ack = ds.acknowledge_pending_wake
    acknowledged = []

    def acknowledge(actual_ctx, delivered=None):
        saved = wc.read_working(tmp_path, ctx.task_id, 2)[1]
        assert any(text in str(row.get("content")) for row in saved["messages"])
        acknowledged.append(True)
        return ack(actual_ctx, delivered)

    monkeypatch.setattr(ds, "acknowledge_pending_wake", acknowledge)
    drain = loop._drain_incoming_messages
    pending_drains = []

    def drain_messages(*args, **kwargs):
        if ds._load_state(ctx, "").get("pending_wake"):
            assert not acknowledged, "no durable wake ACK before the incorporated boundary"
            assert msg_id in ctx._loop_mailbox_seen_ids
            if delivery == "live_wait":
                assert drain_owner_entries(tmp_path, ctx.task_id, set(), 2), "not durably ACKed yet"
            pending_drains.append(True)
        return drain(*args, **kwargs)

    monkeypatch.setattr(loop, "_drain_incoming_messages", drain_messages)
    rounds = []

    def model_input(call, *_args, **_kwargs):
        rounds.append(call.messages)
        if len(rounds) == 1:
            if delivery == "late_bind":
                assert sum(str(row.get("content") or "").count(text) for row in call.messages) == 1
                _bound(tmp_path, task, invocation)
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": "wait", "type": "function", "function": {
                    "name": "delegate_wait", "arguments": json.dumps({"run_id": run_id})}}]}, 0.0
        assert sum(str(row.get("content") or "").count(text) for row in call.messages) == 1
        assert len(directive_calls) == int(delivery == "late_bind")
        assert len(stamps) == int(delivery == "late_bind" and origin == "owner")
        if directive_calls and origin != "owner":
            assert directive_calls[0]["origin"]["source_task_id"] == "parent"
            assert directive_calls[0]["origin"]["relayed_from_task_id"] == ("sibling" if origin == "relay" else "")
        injected = [event for event in list(events.queue)
                    if event.get("type") == "owner_message_injected" and event.get("text") == text]
        assert len(injected) <= int(delivery == "late_bind")
        assert pending_drains and acknowledged
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", model_input)
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs.update(task_id=ctx.task_id, event_queue=events)
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)
    assert len(rounds) == 2


@pytest.mark.parametrize("delivery", ["tool", "startup"])
def test_only_shown_non_control_mail_is_seen_before_save(tmp_path, monkeypatch, delivery):
    from ouroboros import delegate_supervision as ds
    from ouroboros.owner_mailbox import KIND_HURRY
    from ouroboros.subagent_bootstrap import append_startup_receipt
    from tests.test_sleep_questions import _deliver
    from tests.test_working_checkpoint import _registry

    ctx = _registry(tmp_path, monkeypatch, "nanny", 2)._ctx
    seen = ctx._loop_mailbox_seen_ids = {"already-shown"}
    payload = {"status": "waiting_on_user", "supervision_wake_id": "wake", "wake_events": [
        {"msg_id": "shown", "kind": "owner_text", "text": "shown words"},
        {"msg_id": "omitted", "kind": "owner_text"},
        {"msg_id": "control", "kind": KIND_HURRY, "text": "structural control"}]}
    ds._save_state(ctx, {"run_id": "run", "pending_wake": {
        "wake_id": "wake", "payload": payload, "mailbox_ids": ["shown", "omitted", "control", "absent"]}})
    if delivery == "tool":
        _deliver(ctx, json.dumps(payload), monkeypatch, tool="delegate_wait", meta={"supervision_wake_id": "wake"})
    else:
        append_startup_receipt(ctx, [], json.dumps({"status": "configured_session_continuation", "wake": payload}), seen)
    assert ctx._loop_mailbox_seen_ids is seen
    assert seen == {"already-shown", "shown"}
    assert ds._load_state(ctx, "")["pending_wake"], "only an incorporated save may flush the ACK"


@pytest.mark.parametrize("case", ["incorporated", "unsaved", "unusable_checkpoint"])
def test_crash_handoff_restores_wake_mail_once_and_keeps_later_correction_latest(
        tmp_path, monkeypatch, stub_delegate_engines, case):
    """Real wait, save, freeze, crash handoff, bootstrap and loop: the startup receipt
    renders one wake copy and omits mail the restored transcript already holds."""
    from types import SimpleNamespace

    from ouroboros import delegate_recovery as recovery, delegate_supervision as ds, loop, subagent_bootstrap
    from ouroboros import working_checkpoint as wc
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.owner_mailbox import write_owner_message
    from tests.test_configured_session_restore import _known_custody, _restored_registry, _task
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_sleep_questions import _deliver
    from tests.test_working_checkpoint import _registry

    task_id, run_id = "nanny", "run-one"
    old, correction = "Old instruction: paint it blue", "Later correction: paint it green"
    registry = _restored_registry(tmp_path, monkeypatch, "working")
    ctx = registry._ctx
    task = _task(ctx, "working")
    _known_custody(tmp_path, task, "live")
    messages, trace, usage, seen = [], {"tool_calls": []}, {}, set()
    route = (ctx.active_model, ctx.active_effort, False, "max", 0, ctx.context_fit_plan)
    loop._resume_continuation(registry, ({}, {}, loop._load_working_recovery(ctx, messages)), messages, trace,
                              usage, seen, route, ctx._cost_ceiling, 100.0)
    limit = SimpleNamespace(tools=registry, messages=messages, llm_trace=trace, accumulated_usage=usage,
                            owner_msg_seen=seen, round_idx=8, tool_schemas=[], budget_tail="tool")
    if case == "unsaved":
        wc.save_ready(limit)  # The newest saved state predates the wake below.

    def observe(_ctx, rid, *_args, **_kwargs):
        assert write_owner_message(tmp_path, old, task_id, msg_id="old-mail")
        return json.dumps({"status": "no_progress", "run_id": rid})

    scripted_reads(monkeypatch, observe)
    wake = publish(ctx, run_id)
    messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "wait-call", "type": "function",
                     "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": run_id})}}]})
    messages.extend(_deliver(ctx, wake.text, monkeypatch, tool="delegate_wait", meta=wake.meta))
    if case != "unsaved":
        wc.save_ready(limit)  # Incorporated: the wake and its mail are ACKed only after this save.
        assert ds._load_state(ctx, run_id)["last_acknowledged_wake"]["wake_id"]
        assert write_owner_message(tmp_path, correction, task_id, msg_id="correction")
        loop._drain_incoming_messages(messages, queue.Queue(), tmp_path, task_id, None, seen,
                                      owner_ctx=ctx, defer_content_ack=True)
        wc.save_ready(limit)
    engine_calls = list(stub_delegate_engines)
    # The worker dies: the existing retry freezes saved work, then reserves the exact run.
    successor = {**task, "_attempt": 3}
    assert wc.attach_recovery(tmp_path, successor, source_task_id=task_id, from_attempt=2,
                              cause="worker_crash", prior_task=task)
    if case == "unusable_checkpoint":
        successor["_working_recovery"] = {"source_ref": {"path": "missing"}}
    handoff = recovery.prepare_worker_crash_handoff(tmp_path, successor, old_attempt=2, new_attempt=3,
                                                    worker_id=1, exitcode=1)
    assert handoff["run_id"] == run_id
    replayed = handoff["pending_wake"].get("replay_reason") == "acknowledged_in_dead_transcript"
    assert replayed is (case != "unsaved")
    cold = _registry(tmp_path, monkeypatch, task_id, 3)
    cctx = cold._ctx
    cctx.working_recovery = successor["_working_recovery"]
    cctx._configured_startup_wake = subagent_bootstrap.bootstrap_before_context(
        cctx, successor, resolve_dispatch_axes(successor))
    canonical = json.loads(cctx._configured_startup_wake)
    assert canonical["status"] == "configured_session_continuation"
    assert canonical["wake"]["wake_events"][0]["text"] == old, "the deferred ACK keeps its canonical source"
    assert stub_delegate_engines == engine_calls, "the handoff and the bootstrap observe no engine"

    def first_input(call, *_args, **_kwargs):
        texts = [str(row.get("content") or "") for row in call.messages]
        joined = "\n".join(texts)
        receipt = next(text for text in texts if "[CONFIGURED SESSION STARTUP / WAKE RECEIPT]" in text)
        assert joined.count(old) == 1, "restored or replayed, the old mail reaches the model once"
        assert receipt.count('"supervision_wake_id"') == 1, "one wake copy"
        assert (old in receipt) is (case != "incorporated")
        if case != "unsaved":
            assert joined.rindex(correction) > joined.rindex(old), "the later correction stays latest"
        assert not ds._load_state(cctx, run_id)["pending_wake"], "the ready save ACKed the canonical wake"
        # The one contact since the handoff is the starter's round-top placement of the correction (restored with
        # the transcript, or re-drained from the replayed mailbox) into the run it still holds open; the wake's
        # own mail is a tool result, not a corpus row, so nothing else is owed. The undiscovered engine leaves the
        # row for the next round and this first round goes on to its model call.
        assert stub_delegate_engines == engine_calls + (["gateway"] if case != "unsaved" else [])
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", first_input)
    kwargs = _loop_kwargs(tmp_path, cold, [])
    kwargs["task_id"] = task_id
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)


# --- #2992: a wake the transcript holds only in part ---------------------------------------------------------
# Production chain, only the wake's one read scripted: the registry's ``delegate_wait`` entry (``run_wait``), the
# loop's measured delivery (``process_tool_results``), ``save_ready`` and its deferred ACK, the next wait.

_NANNY, _RUN, _TAIL = "nanny", "run-1", "OWNER-MAIL-TAIL-7c1f"


def _nanny(tmp_path, monkeypatch, observe, runs=(_RUN,)):
    """The nanny's real ``delegate_wait``: its runs ended (the body recorded it), so each wait is ready at once;
    only the one addressed read is scripted (``observe``)."""
    from ouroboros import delegate_output
    from ouroboros.task_results import write_task_result
    from tests.test_working_checkpoint import _registry

    write_task_result(tmp_path, _NANNY, "running")
    registry = _registry(tmp_path, monkeypatch, _NANNY, 1)
    ctx = registry._ctx
    ctx.budget_drive_root, ctx.task_metadata = tmp_path, {"root_task_id": _NANNY}
    ctx._loop_mailbox_seen_ids = set()  # the loop's round-top drain and wake exposure share it
    (tmp_path / "logs").mkdir(exist_ok=True)
    for run_id in runs:
        held_run(tmp_path, run_id, _NANNY, ended="succeeded")
    scripted_reads(monkeypatch, observe)
    monkeypatch.setattr(delegate_output, "_READ_COVERAGE", {})  # a fresh worker's reading ledger
    return registry, ctx


def _observer(tmp_path, polls, mail="", **first):
    """Each read answers a terminal leaf; the first one also posts ``mail`` and carries ``first``."""
    from ouroboros.owner_mailbox import write_owner_message

    def observe(_ctx, rid, *_args, **_kwargs):
        polls.append(rid)
        if len(polls) == 1 and mail:
            assert write_owner_message(tmp_path, mail, _NANNY, msg_id="owner-mail")
        return json.dumps({"status": "terminal", "run_id": rid, "state": "succeeded", "last_seq": len(polls),
                           **(first if len(polls) == 1 else {})})

    return observe


def _call(registry, tmp_path, call_id, name, args):
    from ouroboros.loop_tool_execution import _execute_single_tool

    return _execute_single_tool(registry, {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}, tmp_path / "logs", _NANNY)


def _shown(registry, row, messages, trace, fit):
    from ouroboros.loop_tool_execution import process_tool_results

    messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": row["tool_call_id"], "type": "function",
                     "function": {"name": row["fn_name"], "arguments": json.dumps(row["tool_args"])}}]})
    process_tool_results([row], messages, trace, lambda *_a, **_k: None, registry, fit_candidate=fit, tool_schemas=[])
    return messages[-1]["content"], trace["tool_calls"][-1]


def _ready(registry, messages, trace):
    from types import SimpleNamespace

    from ouroboros.working_checkpoint import save_ready

    save_ready(SimpleNamespace(tools=registry, messages=messages, llm_trace=trace, accumulated_usage={},
                               owner_msg_seen=set(), round_idx=len(trace["tool_calls"]), tool_schemas=[],
                               budget_tail="tool"))


def _wake_state(ctx):
    """(pending wake id, last acknowledged wake id)."""
    from ouroboros import delegate_supervision as ds

    state = ds._load_state(ctx, _RUN)
    return ((state.get("pending_wake") or {}).get("wake_id", ""),
            (state.get("last_acknowledged_wake") or {}).get("wake_id", ""))


def _unread(tmp_path, ctx):
    from ouroboros.owner_mailbox import drain_owner_entries

    return [entry["msg_id"] for entry in drain_owner_entries(tmp_path, _NANNY, set(), str(ctx.task_attempt))]


def _wake_id(row):
    return row["result_meta"]["tool_result_meta"]["supervision_wake_id"]


def _page(ref, start, end):
    return {"root": ref["root"], "path": ref["path"], "start_line": 1, "max_lines": 100_000,
            "start_char": start, "max_chars": end - start}


@pytest.mark.parametrize("shape", ["outer_cut", "inner_spill"])
def test_a_cold_continuation_keeps_pending_wake_sources_and_their_read_coverage(tmp_path, monkeypatch, shape):
    """A resumed worker completes the original source without waiting again, and the ranges the saved transcript
    already holds ride with it (E3): the rest of the source completes it. No mailbox entry can hide a missing source
    map by delivering the obligation through the round-top drain."""
    from ouroboros import owner_wait

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls))
    ctx.task_contract = {"delegation_budget": {"intent_note": "x" * (30_000 if shape == "inner_spill" else 4_000)}}
    fit, messages, trace = _wide() if shape == "inner_spill" else _crowded()
    wake = _call(registry, tmp_path, "wait-before-cold", "delegate_wait", {"run_id": _RUN})
    _shown(registry, wake, messages, trace, fit)
    sources = dict(ctx._wake_sources)
    assert len(sources) == 1
    source = next(iter(sources.values()))
    half, end = source["size"] // 2, source["size"] + 1
    fit = _wide()[0]
    _shown(registry, _call(registry, tmp_path, "read-before-cold", "read_file", _page(source, 0, half)),
           messages, trace, fit)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "")
    state = json.loads(json.dumps(owner_wait.continuation_state(ctx, messages, trace, {}, 1, [], set())))

    fresh, resumed = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls))
    messages, trace = [], {}
    owner_wait.restore_continuation_state(fresh, state, messages, trace, {}, set())
    _shown(fresh, _call(fresh, tmp_path, "read-second-half", "read_file", _page(source, half, end)),
           messages, trace, fit)
    assert _wake_state(resumed) == (_wake_id(wake), ""), "a complete read still waits for checkpoint"
    _ready(fresh, messages, trace)
    assert _wake_state(resumed) == ("", _wake_id(wake)), "the restored coverage and the rest of the source complete it"
    assert resumed._wake_sources == {}
    _call(fresh, tmp_path, "wait-after-cold", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 2, "one fresh poll, no duplicate run"

    resumed._wake_sources = {"stale": {}}
    owner_wait.restore_continuation_state(fresh, {key: value for key, value in state.items() if key != "wake_sources"},
                                         [], {}, {}, set())
    assert resumed._wake_sources == {}, "a legacy state does not inherit another continuation's source map"


def test_a_crowded_frame_cuts_the_wake_and_only_its_source_read_whole_acknowledges_it(tmp_path, monkeypatch):
    """#2992 A/B/C shape: the measured frame cuts the wake's JSON, so its ACK is refused (fail closed); before the
    repair the next wait replayed it forever. A whole read of the exact retained source now acknowledges it at the
    next saved state and the next wait observes the run."""
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from tests._tool_result_delivery_shared import measured_fit
    from tests.test_tool_result_delivery import _view_json

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(
        tmp_path, polls, "Owner correction: " + "keep the reviewed shape; " * 100 + _TAIL))
    fit, filler = measured_fit(window=40_000, reserve=2_000), "filler words for an already long transcript. " * 64
    messages, trace = [{"role": "user", "content": filler}], {"tool_calls": []}
    while estimate_context_prompt_tokens(messages, []) < 40_000 - 2_000 - 900:
        messages[0]["content"] += filler
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    shown, row = _shown(registry, wake, messages, trace, fit)
    assert row.get("result_partial") and _TAIL not in shown, "premise: the frame cut the wake"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and _unread(tmp_path, ctx) == ["owner-mail"]
    omitted = _view_json(shown)["read_omitted"]
    read = _call(registry, tmp_path, "read-1", "read_file", {key: omitted[key] for key in (
        "root", "path", "start_line", "max_lines", "start_char", "max_chars")})
    read_shown, read_row = _shown(registry, read, messages, trace, fit)
    assert not read_row.get("result_partial") and _TAIL in read_shown, "the omitted middle arrived whole"
    assert _wake_state(ctx) == (_wake_id(wake), ""), "the ACK waits for a saved state"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake)) and _unread(tmp_path, ctx) == []
    again = _call(registry, tmp_path, "wait-2", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 2 and _wake_id(again) != _wake_id(wake), "the next wait observes the run"


def test_a_spilled_wake_waits_for_its_source_and_the_next_round_drain_delivers_the_omitted_mail_first(
        tmp_path, monkeypatch):
    """A complete view of a spilled wake is a summary: its cut mail is not seen and nothing is acknowledged. A replay
    before any read keeps that mail, the same canonical source and says it is a replay; the next round's mailbox
    drain delivers the full text, and only then may a replay omit it and acknowledge the wake."""
    from ouroboros import loop
    from tests._tool_result_delivery_shared import measured_fit

    polls, mail = [], "Owner instruction: " + "the exact long specification line; " * 600 + _TAIL
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls, mail))
    fit, messages, trace = measured_fit(window=500_000, reserve=16_000), [{"role": "user", "content": "go"}], {
        "tool_calls": []}
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    envelope = json.loads(wake["result"])
    source = envelope["wake_delivery"]["source"]
    assert envelope["wake_delivery"]["complete"] is False
    shown, row = _shown(registry, wake, messages, trace, fit)
    assert not row.get("result_partial") and _TAIL not in shown, "premise: a complete view of a summary"
    assert "owner-mail" not in (getattr(ctx, "_loop_mailbox_seen_ids", None) or set()), "a cut summary is not seen"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and _unread(tmp_path, ctx) == ["owner-mail"]

    replay = _call(registry, tmp_path, "wait-2", "delegate_wait", {"run_id": _RUN})
    replayed = json.loads(replay["result"])
    assert len(polls) == 1 and _wake_id(replay) == _wake_id(wake), "replayed before any new observation"
    assert replayed["replay_of"]["fresh_observation"] is False and replayed["replay_of"]["published_at"]
    assert replayed["wake_delivery"]["source"] == source, "the same canonical bytes: earlier reading still counts"
    assert [event["msg_id"] for event in replayed["wake_events"]] == ["owner-mail"], "the unseen mail stays"
    _shown(registry, replay, messages, trace, fit)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and _unread(tmp_path, ctx) == ["owner-mail"]

    seen = ctx._loop_mailbox_seen_ids
    loop._drain_incoming_messages(messages, queue.Queue(), tmp_path, _NANNY, None, seen,
                                  owner_ctx=ctx, defer_content_ack=True)
    assert _TAIL in messages[-1]["content"] and "owner-mail" in seen, "the round-top drain delivered it whole"
    _ready(registry, messages, trace)
    assert _unread(tmp_path, ctx) == [] and _wake_state(ctx) == (_wake_id(wake), "")

    narrowed = _call(registry, tmp_path, "wait-3", "delegate_wait", {"run_id": _RUN})
    view = json.loads(narrowed["result"])
    assert len(polls) == 1 and "wake_delivery" not in view and view.get("wake_events", []) == [], \
        "now the replay may omit the delivered mail"
    _shown(registry, narrowed, messages, trace, fit)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake))
    assert len(polls) == 1
    _call(registry, tmp_path, "wait-4", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 2, "an acknowledged wake lets the next wait observe"
    assert wake["result_meta"]["tool_result_meta"]["wake_source"] == source, "the spilled source rides typed meta"


def test_a_spilled_wake_with_mail_and_a_question_counts_only_what_was_read_of_that_exact_source(
        tmp_path, monkeypatch):
    """Mail and a question both wait for the spilled source read whole: half of it shows and acknowledges nothing,
    the same bytes under another name credit nothing, and the remaining half completes both."""
    import shutil

    from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
    from tests._tool_result_delivery_shared import measured_fit

    polls, mail = [], "Owner instruction: " + "the exact long specification line; " * 600 + _TAIL
    question = {"interaction_id": "q1", "questions": [{"id": "choice", "question": "Which path?"}]}
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(
        tmp_path, polls, mail, status="waiting_on_user", state="running", waiting_on_user=True,
        pending_interactions=[question]))
    fit, messages, trace = measured_fit(window=500_000, reserve=16_000), [{"role": "user", "content": "go"}], {
        "tool_calls": []}
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    source = json.loads(wake["result"])["wake_delivery"]["source"]
    _shown(registry, wake, messages, trace, fit)
    total = source["size"]
    store = task_artifact_dir_path(tmp_path, task_id_for_artifacts(ctx))
    shutil.copyfile(store / source["path"], store / "copy.json")
    copy_view = _shown(registry, _call(registry, tmp_path, "read-copy", "read_file", {
        "root": "artifact_store", "path": "copy.json", "start_line": 1, "max_lines": 100_000}), messages, trace, fit)
    assert not copy_view[1].get("result_partial"), "the copy was delivered whole"
    half = _shown(registry, _call(registry, tmp_path, "read-half", "read_file", _page(source, 0, total // 2)),
                  messages, trace, fit)
    assert not half[1].get("result_partial")
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and _unread(tmp_path, ctx) == ["owner-mail"]
    assert not (getattr(ctx, "_sleep_interaction_seen", None) or {}), "the question was not shown yet"
    rest = _call(registry, tmp_path, "read-rest", "read_file", _page(source, total // 2, total + 1))
    _shown(registry, rest, messages, trace, fit)
    assert ctx._sleep_interaction_seen == {_RUN: ["q1"]} and "owner-mail" in ctx._loop_mailbox_seen_ids
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake)) and _unread(tmp_path, ctx) == []


def test_a_refused_ack_after_the_source_read_is_recorded_and_the_replay_acknowledges_without_another_read(
        tmp_path, monkeypatch):
    """The read whole, its saved state, then a mailbox ACK that does not land: the wake stays pending with a typed
    diagnostic and no callback is kept. The replay names the same canonical source, already read in this process,
    and acknowledges at the next saved state."""
    from ouroboros import delegate_custody as custody
    from ouroboros import owner_mailbox
    from tests._tool_result_delivery_shared import measured_fit

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls, "Short owner note " + _TAIL))
    ctx.task_contract = {"delegation_budget": {"intent_note": "preserve-complete-intent:" + "y" * 30_000}}
    fit, messages, trace = measured_fit(window=500_000, reserve=16_000), [{"role": "user", "content": "go"}], {
        "tool_calls": []}
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    source = json.loads(wake["result"])["wake_delivery"]["source"]
    _shown(registry, wake, messages, trace, fit)
    _shown(registry, _call(registry, tmp_path, "read-1", "read_file", _page(source, 0, source["size"] + 1)),
           messages, trace, fit)
    real = owner_mailbox.acknowledge_task_messages
    monkeypatch.setattr(owner_mailbox, "acknowledge_task_messages", lambda *_a, **_k: False)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and not getattr(ctx, "_pending_content_acks", None)
    rows = [json.loads(line) for line in custody.event_log_path(tmp_path).read_text().splitlines()]
    refused = [(row["wake_id"], row["reason"]) for row in rows if row.get("type") == "delegate_supervision_wake_ack_refused"]
    assert refused == [(_wake_id(wake), "mailbox_acknowledgement_unwritten")]
    monkeypatch.setattr(owner_mailbox, "acknowledge_task_messages", real)
    replay = _call(registry, tmp_path, "wait-2", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 1 and json.loads(replay["result"])["wake_delivery"]["source"] == source
    _shown(registry, replay, messages, trace, fit)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake)) and _unread(tmp_path, ctx) == []
    assert replay["result_meta"]["tool_result_meta"]["wake_source"] == source, "found without parsing the view"


def test_a_startup_receipt_carries_an_oversized_recovered_wake_whole_before_its_saved_acknowledgement(
        tmp_path, monkeypatch):
    """The bootstrap's overflow path: a recovered wake longer than any ``delegate_wait`` view is never summarized by
    the startup receipt. The restored transcript gets the canonical wake whole, exposure stages from that text and
    the canonical ACK lands only after a saved state holds it."""
    from ouroboros import delegate_supervision as ds
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.subagent_bootstrap import append_startup_receipt
    from ouroboros.tool_capabilities import tool_result_limit
    from tests.test_working_checkpoint import _registry

    registry = _registry(tmp_path, monkeypatch, _NANNY, 2)
    ctx = registry._ctx
    ctx._loop_mailbox_seen_ids = set()
    assert write_owner_message(tmp_path, "Owner instruction: " + "the exact long specification line; " * 600 + _TAIL,
                               _NANNY, msg_id="owner-mail")
    payload = {"status": "terminal", "run_id": _RUN, "supervision_wake_id": "wake",
               "wake_events": ds._addressed_wakes(ctx, {})}
    assert len(json.dumps(payload, indent=2)) > tool_result_limit("delegate_wait"), "premise: a wait view spills it"
    ds._save_state(ctx, {"run_id": _RUN, "pending_wake": {"wake_id": "wake", "attempt_key": "1", "payload": payload,
                                                          "mailbox_ids": ["owner-mail"]}})
    messages, trace = [], {"tool_calls": []}
    append_startup_receipt(ctx, messages, json.dumps({"status": "configured_session_recovered_wake",
                                                      "recovery": {}, "wake": payload}, indent=2), set())
    assert _TAIL in messages[-1]["content"] and "wake_delivery" not in messages[-1]["content"]
    assert "owner-mail" in ctx._loop_mailbox_seen_ids and _wake_state(ctx) == ("wake", "")
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", "wake") and _unread(tmp_path, ctx) == []


def _crowded(window=40_000, reserve=2_000, free=900):
    """A measured frame and one user turn leaving ``free`` tokens beside the reply reserve: unasked bodies are cut."""
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from tests._tool_result_delivery_shared import measured_fit

    filler = "filler words for an already long transcript. " * 64
    messages = [{"role": "user", "content": filler}]
    while estimate_context_prompt_tokens(messages, []) < window - reserve - free:
        messages[0]["content"] += filler
    return measured_fit(window=window, reserve=reserve), messages, {"tool_calls": []}


def _wide():
    from tests._tool_result_delivery_shared import measured_fit

    return measured_fit(window=500_000, reserve=16_000), [{"role": "user", "content": "go"}], {"tool_calls": []}


def _mail():
    return "Owner instruction: " + "the exact long specification line; " * 600 + _TAIL


def _key(ref):
    return f"{ref['path']}|{ref['sha256']}"


def test_a_spilled_wake_the_frame_also_cuts_waits_for_its_typed_inner_source_not_the_cut_summary(
        tmp_path, monkeypatch):
    """Both shortened: the frame cuts the spilled summary too. The cut text is never parsed for a source; the typed
    inner source is the one obligation, and the outer retained text (only the summary) read whole credits nothing."""
    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls, _mail()))
    fit, messages, trace = _crowded()
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    inner = wake["result_meta"]["tool_result_meta"]["wake_source"]
    shown, row = _shown(registry, wake, messages, trace, fit)
    assert row.get("result_partial") and json.loads(wake["result"])["wake_delivery"]["complete"] is False
    assert list(ctx._wake_sources) == [_key(inner)], "the typed inner source, not one parsed from the cut text"
    outer = row["result_source_ref"]
    summary = _shown(registry, _call(registry, tmp_path, "read-outer", "read_file", _page(outer, 0, outer["size"] + 1)),
                     messages, trace, fit)
    assert not summary[1].get("result_partial") and _TAIL not in summary[0], "the summary arrived whole, mail cut"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and _unread(tmp_path, ctx) == ["owner-mail"]
    whole = _shown(registry, _call(registry, tmp_path, "read-inner", "read_file", _page(inner, 0, inner["size"] + 1)),
                   messages, trace, fit)
    assert not whole[1].get("result_partial") and _TAIL in whole[0]
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake)) and _unread(tmp_path, ctx) == [] and ctx._wake_sources == {}


def test_a_cut_source_read_and_an_unsaved_checkpoint_each_leave_the_wake_pending_until_the_next_saved_state(
        tmp_path, monkeypatch):
    """A read the frame cuts shows a head and tail, so it completes nothing, nor does its repeat. The whole read then
    defers the ACK; a checkpoint whose write fails flushes nothing and keeps it, and the next saved state ACKs."""
    from ouroboros import working_checkpoint as wc
    from tests._tool_result_delivery_shared import measured_fit

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls, _mail()))
    fit, messages, trace = _wide()
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    inner = wake["result_meta"]["tool_result_meta"]["wake_source"]
    _shown(registry, wake, messages, trace, fit)
    for attempt in ("cut-1", "cut-2"):
        cut = _shown(registry, _call(registry, tmp_path, attempt, "read_file", _page(inner, 0, inner["size"] + 1)),
                     messages, trace, measured_fit(window=3_000))
        assert cut[1].get("result_partial"), "the read alone exceeds this frame's usable input"
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and _unread(tmp_path, ctx) == ["owner-mail"]
    _shown(registry, _call(registry, tmp_path, "read-1", "read_file", _page(inner, 0, inner["size"] + 1)),
           messages, trace, fit)
    real = wc._write
    monkeypatch.setattr(wc, "_write", lambda *_a, **_k: (_ for _ in ()).throw(OSError("controlled unsaved checkpoint")))
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == (_wake_id(wake), "") and len(ctx._pending_content_acks) == 1, "kept, not flushed"
    monkeypatch.setattr(wc, "_write", real)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(wake)) and _unread(tmp_path, ctx) == []


def test_a_cut_wake_without_a_retained_source_is_never_acknowledged_and_its_replay_recovers_once_one_is_kept(
        tmp_path, monkeypatch):
    """Source persistence fails: the cut view promises no source, so nothing can complete it and nothing is ACKed.
    The replay is marked and cut again the same way; once a source is kept, reading it whole acknowledges."""
    from ouroboros import tool_result_delivery
    from tests.test_tool_result_delivery import _view_json

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls, "Owner correction: " + "x; " * 900))
    fit, messages, trace = _crowded()
    real = tool_result_delivery.persist_result_source
    monkeypatch.setattr(tool_result_delivery, "persist_result_source", lambda *_a, **_k: {})
    wake = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    shown, row = _shown(registry, wake, messages, trace, fit)
    assert row.get("result_partial") and row.get("result_source_status") == "source_unavailable"
    assert not getattr(ctx, "_wake_sources", None)
    _ready(registry, messages, trace)
    replay = _call(registry, tmp_path, "wait-2", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 1 and json.loads(replay["result"])["replay_of"]["fresh_observation"] is False
    again = _shown(registry, replay, messages, trace, fit)[1]
    _ready(registry, messages, trace)
    assert again.get("result_source_status") == "source_unavailable" and _wake_state(ctx) == (_wake_id(wake), "")
    monkeypatch.setattr(tool_result_delivery, "persist_result_source", real)
    kept, kept_row = _shown(registry, _call(registry, tmp_path, "wait-3", "delegate_wait", {"run_id": _RUN}),
                            messages, trace, fit)
    omitted = _view_json(kept)["read_omitted"]
    _shown(registry, _call(registry, tmp_path, "read-1", "read_file", {key: omitted[key] for key in (
        "root", "path", "start_line", "max_lines", "start_char", "max_chars")}), messages, trace, fit)
    _ready(registry, messages, trace)
    assert kept_row.get("result_source_status") == "ready" and _wake_state(ctx) == ("", _wake_id(wake))


def test_a_stale_source_read_whole_after_its_wake_was_acknowledged_cannot_acknowledge_the_next_wake(
        tmp_path, monkeypatch):
    """No pending source is dropped, so a later read of an older wake's exact source still completes; it names that
    acknowledged wake, never the one pending now, which keeps waiting while the refusal records what was delivered."""
    from ouroboros import delegate_custody as custody
    from ouroboros import loop

    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls, _mail()))
    fit, messages, trace = _wide()
    first = _call(registry, tmp_path, "wait-1", "delegate_wait", {"run_id": _RUN})
    source = first["result_meta"]["tool_result_meta"]["wake_source"]
    _shown(registry, first, messages, trace, fit)
    loop._drain_incoming_messages(messages, queue.Queue(), tmp_path, _NANNY, None, ctx._loop_mailbox_seen_ids,
                                  owner_ctx=ctx, defer_content_ack=True)
    _ready(registry, messages, trace)
    _shown(registry, _call(registry, tmp_path, "wait-2", "delegate_wait", {"run_id": _RUN}), messages, trace, fit)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(first)) and _key(source) in ctx._wake_sources, "acknowledged, kept"
    second = _call(registry, tmp_path, "wait-3", "delegate_wait", {"run_id": _RUN})
    assert len(polls) == 2 and _wake_id(second) != _wake_id(first)
    _shown(registry, _call(registry, tmp_path, "read-old", "read_file", _page(source, 0, source["size"] + 1)),
           messages, trace, fit)
    _ready(registry, messages, trace)
    rows = [json.loads(line) for line in custody.event_log_path(tmp_path).read_text().splitlines()]
    refused = [(row["wake_id"], row["reason"], row["delivered_wake_ids"]) for row in rows
               if row.get("type") == "delegate_supervision_wake_ack_refused"]
    assert refused == [(_wake_id(second), "pending_wake_not_named", [_wake_id(first)])]
    assert _wake_state(ctx) == (_wake_id(second), _wake_id(first)) and ctx._wake_sources == {}
    _shown(registry, second, messages, trace, fit)
    _ready(registry, messages, trace)
    assert _wake_state(ctx) == ("", _wake_id(second))


def test_every_pending_wake_source_stays_completable_beyond_any_count(tmp_path, monkeypatch):
    """Forty runs each hold a spilled wake (their bulk is the shared coordination context, so no mail is cut); the
    oldest exact source, read whole after the other thirty-nine arrived, still acknowledges its own wake only."""
    polls = []
    registry, ctx = _nanny(tmp_path, monkeypatch, _observer(tmp_path, polls), runs=[f"run-{n}" for n in range(40)])
    ctx.task_contract = {"delegation_budget": {"intent_note": "preserve-complete-intent:" + "y" * 30_000}}
    fit, messages, trace = _wide()
    wakes = [_call(registry, tmp_path, f"wait-{n}", "delegate_wait", {"run_id": f"run-{n}"}) for n in range(40)]
    for wake in wakes:
        _shown(registry, wake, messages, trace, fit)
    sources = [wake["result_meta"]["tool_result_meta"]["wake_source"] for wake in wakes]
    assert len({_key(ref) for ref in sources}) == 40 and len(ctx._wake_sources) == 40
    _shown(registry, _call(registry, tmp_path, "read-oldest", "read_file", _page(sources[0], 0, sources[0]["size"] + 1)),
           messages, trace, fit)
    _ready(registry, messages, trace)
    from ouroboros import delegate_supervision as ds

    pending = {n: (ds._load_state(ctx, f"run-{n}").get("pending_wake") or {}).get("wake_id") for n in range(40)}
    assert pending[0] is None and all(pending[n] == _wake_id(wakes[n]) for n in range(1, 40))
    assert len(ctx._wake_sources) == 39


# --- #2992 legacy continuations: a seen-set base 1a0e1e9d saved from a cut summary ----------------------------

def _legacy_spilled_wake(tmp_path, monkeypatch, *, annotated=False):
    """What base 1a0e1e9d saved after a spilled wake carrying a long owner mail: the wake executed and delivered as
    production does (``execute_tool_batch``: its row keeps its invocation record), then the old seam's one difference
    applied by hand, disclosed emulation: its ``_text_mailbox_ids`` counted the cut summary as seen. The review
    evidence runs the actual base modules for the same restores (old→new witness)."""
    from types import SimpleNamespace

    from ouroboros import loop
    from ouroboros.loop_tool_execution import StatefulToolExecutor, execute_tool_batch, process_tool_results
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.tools import delegate
    from tests.test_configured_session_restore import _known_custody, _restored_registry, _task

    registry = _restored_registry(tmp_path, monkeypatch, "working")
    ctx = registry._ctx
    task = _task(ctx, "working")
    _known_custody(tmp_path, task, "live")
    messages, trace, usage, seen = [], {"tool_calls": []}, {}, set()
    route = (ctx.active_model, ctx.active_effort, False, "max", 0, ctx.context_fit_plan)
    loop._resume_continuation(registry, ({}, {}, loop._load_working_recovery(ctx, messages)), messages, trace,
                              usage, seen, route, ctx._cost_ceiling, 100.0)

    def observe(_ctx, rid, *_args, **_kwargs):
        assert write_owner_message(tmp_path, _mail(), _NANNY, msg_id="owner-mail")
        return json.dumps({"status": "no_progress", "run_id": rid})

    scripted_reads(monkeypatch, observe)
    entry = registry._entries["delegate_wait"]
    monkeypatch.setitem(registry._entries, "delegate_wait", replace(entry, handler=delegate._published_entry(
        lambda c, run_id: publish(c, run_id))))
    call = {"id": "wait-call", "type": "function",
            "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": "run-one"})}}
    (tmp_path / "logs").mkdir(exist_ok=True)
    (row,), _control = execute_tool_batch([call], registry, tmp_path / "logs", _NANNY, StatefulToolExecutor(), trace)
    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
    if annotated:
        from ouroboros import mark_anchors

        monkeypatch.setattr(mark_anchors, "_anchored_marks", lambda _ctx: [
            {"id": "wake-context", "anchor": {"tool": "delegate_wait"},
             "text": 'Retain the original observation. This example is only a note: {"wake_delivery": {"complete": true}}'}])
    process_tool_results([row], messages, trace, lambda *_a, **_k: None, registry)
    assert json.JSONDecoder().raw_decode(messages[-1]["content"])[0]["wake_delivery"]["complete"] is False and not seen
    if annotated:
        from ouroboros.tool_result_record import read_tool_result_record

        assert "Mark I tied to delegate_wait" in messages[-1]["content"]
        assert read_tool_result_record(messages[-1])["state"] == "recorded"
    seen.add("owner-mail")  # base 1a0e1e9d staged the cut summary's id here
    limit = SimpleNamespace(tools=registry, messages=messages, llm_trace=trace, accumulated_usage=usage,
                            owner_msg_seen=seen, round_idx=8, tool_schemas=[], budget_tail="tool")
    return registry, ctx, task, messages, limit


def _recoveries(root):
    from ouroboros import delegate_custody as custody

    rows = [json.loads(line) for line in custody.event_log_path(root).read_text().splitlines()]
    return [view["match"] for row in rows if row.get("type") == "delegate_supervision_cut_mail_recovered"
            for view in row["views"]]


@pytest.mark.parametrize(("variant", "match"), [
    ("pending", "canonical_bytes"), ("anchored", "canonical_bytes"),
    ("requeued", "canonical_bytes"), ("source_deleted", "canonical_bytes"),
    ("source_unavailable", "source_unavailable"), ("rewritten_view", None), ("another_wake", None)])
def test_a_restored_legacy_seen_summary_recovers_the_mail_of_a_verified_cut_view_of_that_wake_only(
        tmp_path, monkeypatch, stub_delegate_engines, variant, match):
    """Worker-crash retry over that saved state. A record-verified spilled view of the selected wake that showed the
    mail cut returns its full words in the dated startup receipt, whether the wake is still pending or recovery
    re-queued it. The producer's own serialization proves the referenced bytes without a read, so a deleted file
    changes nothing; a referenced source nothing proves is a diagnostic, never a reason to ACK the mail unshown. A
    view whose bytes no longer match its record, or one naming another wake, recovers nothing (the limit)."""
    import hashlib

    from ouroboros import delegate_recovery as recovery
    from ouroboros import delegate_supervision as ds
    from ouroboros import loop, subagent_bootstrap
    from ouroboros import working_checkpoint as wc
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    registry, ctx, task, messages, limit = _legacy_spilled_wake(tmp_path, monkeypatch, annotated=variant == "anchored")
    tool_row = messages[-1]
    view = json.JSONDecoder().raw_decode(tool_row["content"])[0]

    def restamped(changed):  # other bytes whose own record verifies
        tool_row["content"] = json.dumps(changed, ensure_ascii=False, indent=2)
        tool_row[TOOL_RESULT_RECORD_KEY] = {**tool_row[TOOL_RESULT_RECORD_KEY], "content_sha256": hashlib.sha256(
            tool_row["content"].encode("utf-8")).hexdigest()}

    if variant == "requeued":
        assert ds.acknowledge_pending_wake(ctx, tool_row["content"])  # base's flush ACKed the summary
    elif variant == "source_deleted":
        (task_artifact_dir_path(tmp_path, _NANNY) / view["wake_delivery"]["source"]["path"]).unlink()
    elif variant == "source_unavailable":
        gone = {**view["wake_delivery"]["source"], "path": "source_handles/tool_results/gone.json", "sha256": "0" * 64}
        restamped({**view, "wake_delivery": {**view["wake_delivery"], "source": gone}})
    elif variant == "rewritten_view":
        tool_row["content"] = json.dumps(view, ensure_ascii=False, indent=1)  # same JSON, other bytes
    elif variant == "another_wake":
        restamped({**view, "supervision_wake_id": "another-wake"})
    assert wc.save_round(limit, "ready")  # saved, then the worker dies before any flush
    successor = {**task, "_attempt": 3}
    assert wc.attach_recovery(tmp_path, successor, source_task_id=_NANNY, from_attempt=2, cause="worker_crash",
                              prior_task=task)
    handoff = recovery.prepare_worker_crash_handoff(tmp_path, successor, old_attempt=2, new_attempt=3,
                                                    worker_id=1, exitcode=1)
    assert (handoff["pending_wake"].get("replay_reason") == "acknowledged_in_dead_transcript") is (variant == "requeued")
    cold = _registry(tmp_path, monkeypatch, _NANNY, 3)
    cold._ctx.working_recovery = successor["_working_recovery"]
    cold._ctx._configured_startup_wake = subagent_bootstrap.bootstrap_before_context(
        cold._ctx, successor, resolve_dispatch_axes(successor))
    first = {}

    def first_input(call, *_args, **_kwargs):
        receipt = next(str(row.get("content")) for row in call.messages
                       if "[CONFIGURED SESSION STARTUP / WAKE RECEIPT]" in str(row.get("content")))
        first.update(receipt=receipt, pending=bool(ds._load_state(cold._ctx, "run-one").get("pending_wake")))
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", first_input)
    kwargs = _loop_kwargs(tmp_path, cold, [])
    kwargs["task_id"] = _NANNY
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)
    assert (_TAIL in first["receipt"]) is (match is not None), "the full words travel in the dated receipt"
    assert _recoveries(tmp_path) == ([match] if match else [])
    assert not first["pending"], "the receipt, saved, then acknowledges the canonical wake"


@pytest.mark.parametrize("annotated", [False, True])
def test_a_parked_legacy_seen_summary_keeps_its_replay_pending_until_the_source_is_read_whole(tmp_path, monkeypatch, annotated):
    """The same saved state parked and resumed in the same attempt (a planned restart). The model's replay keeps the
    mail it was shown cut: the spilled view names its exact source and acknowledges nothing until that source is read
    whole and the next state is saved."""
    from ouroboros import delegate_supervision as ds
    from ouroboros import loop
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait
    from ouroboros.task_results import write_task_result
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    _registry_, ctx, _task, messages, limit = _legacy_spilled_wake(tmp_path, monkeypatch, annotated=annotated)
    ctx._owner_wait_requested = "quiz"
    write_task_result(tmp_path, _NANNY, "running")
    park = checkpoint_owner_wait(ctx, messages, limit.llm_trace, limit.accumulated_usage, 9, [], set(limit.owner_msg_seen))
    set_owner_wait(tmp_path, _NANNY, {**park, "state": "waiting"})
    resumed = _registry(tmp_path, monkeypatch, _NANNY, 2)
    resumed._ctx.owner_wait_resume = {**park, "restart_transaction_id": "observed-restart"}
    resumed._ctx.owner_wait_callback = lambda *_: None
    rounds, facts = [], {}

    def tool_row(messages_, call_id):
        return next(str(row["content"]) for row in messages_ if row.get("tool_call_id") == call_id)

    def model(call, *_args, **_kwargs):
        rounds.append(list(call.messages))
        if len(rounds) == 1:
            return {"role": "assistant", "content": "", "tool_calls": [{"id": "wait-again", "type": "function",
                    "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": "run-one"})}}]}, 0.0
        if len(rounds) == 2:
            replay = json.JSONDecoder().raw_decode(tool_row(rounds[-1], "wait-again"))[0]
            facts.update(replay=replay, pending=bool(ds._load_state(resumed._ctx, "run-one").get("pending_wake")),
                         unread=_unread(tmp_path, resumed._ctx))
            return {"role": "assistant", "content": "", "tool_calls": [{"id": "read-source", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps(
                        replay["wake_delivery"]["source"]["read"]["arguments"])}}]}, 0.0
        facts.update(read=tool_row(rounds[-1], "read-source"), unread_after=_unread(tmp_path, resumed._ctx),
                     pending_after=bool(ds._load_state(resumed._ctx, "run-one").get("pending_wake")))
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", model)
    kwargs = _loop_kwargs(tmp_path, resumed, [])
    kwargs["task_id"] = _NANNY
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)
    replay = facts["replay"]
    assert replay["replay_of"]["fresh_observation"] is False
    assert [event["msg_id"] for event in replay["wake_events"] if event.get("text_omitted_chars")] == ["owner-mail"]
    assert facts["pending"] and facts["unread"] == ["owner-mail"], "a cut replay acknowledges nothing"
    assert _TAIL in facts["read"] and not facts["pending_after"] and facts["unread_after"] == []
    assert _recoveries(tmp_path) == ["canonical_bytes"]

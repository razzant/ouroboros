"""A selected finish is ready work under a strict last fit; a held one meets the margin (#3913, ``loop._finish_tool_tail``).

The invariant arrived (commit 7b625da42) inside the module of a reminder about finals nobody reads, because that
module held the priced, place-carrying root; one speech rule retired the reminder and its module, and the invariant
keeps its own home here, its test unchanged. The real Host turn builds the speaker carrier, and the real
``run_llm_loop`` with its budget rail, finalization gates and pipeline run a schedule's root of that conversation; only
the model and the price oracle are scripted. Synthetic data; nothing leaves the process and no provider is contacted.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import extension_registry_state as registry_state
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolEntry, ToolRegistry
from tests.test_presence_explicit_final import (
    NOTE,
    REACT_SCHEMA,
    RECORDS_REACT,
    RECORDS_SEND,
    SEND_SCHEMA,
    finish,
    turn,
)
from tests.test_presence_explicit_final import room as room  # fixture

pytestmark = [pytest.mark.usefixtures("provider_catalog_offline")]

SCHEDULED = {"schedule_id": "s-ready-finish", "occurrence_id": "o-1"}


def open_door(monkeypatch):
    """The connector declares its in-conversation send door and a reaction door (``gesture_room``'s publication)."""
    manifest = SimpleNamespace(raw_extra={"tools": [
        {"name": "send", "effect": "message", "target": ["conversation_id"], "conversation": "conversation_id",
         "thread": "thread_id"},
        {"name": "react", "effect": "gesture", "target": ["account_id", "conversation_id"],
         "conversation": "conversation_id", "message": "message_id"}]})
    bundle = registry_state._ExtensionRegistrations(generation_digest="g1", loaded_skill=SimpleNamespace(manifest=manifest))
    for name in (RECORDS_SEND, RECORDS_REACT):
        monkeypatch.setitem(registry_state._tools, name, {"skill": "bridge", "extension_generation": "g1"})
    monkeypatch.setitem(registry_state._extensions, "bridge", bundle)


def carrier(room):
    """The speaker carrier a real turn of this conversation leaves (the place's door is not yet declared)."""
    room.h.script = [finish(answer="I will look again later.")]
    body = turn(room)
    room.transport.submit("chat:a:1", body)
    return dict(load_task_result(room.data, body["turn_ref"])["metadata"]["presence"])


def run_root(room, monkeypatch, responses, *, presence=None, setup=None):
    """One schedule's root of this conversation in the real loop and pipeline; ``responses`` script its model (a
    callable reads the messages), ``setup`` sets its context up before the loop. Returns its trace, model inputs,
    stored row and pipeline events."""
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros import context, loop
    from tests.test_context_fit_v664 import _plan

    root_id = "ready-finish-root"
    meta = {"presence": presence or carrier(room), "delegation_role": "root", "schedule_occurrence": dict(SCHEDULED)}
    open_door(monkeypatch)
    registry = ToolRegistry(repo_dir=room.h.repo, drive_root=room.data)
    ctx = registry._ctx
    ctx.task_id, ctx.task_metadata, ctx.is_direct_chat, ctx.task_attempt = root_id, meta, False, None
    ctx.active_model, ctx.active_effort, ctx.active_use_local, ctx.active_context_mode = "test-model", "high", False, "max"
    ctx.context_fit_plan, ctx.owner_wait_callback = _plan(window=500_000, known=True), (lambda *_: None)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task["model"], "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="route-a")))
    registry.register(ToolEntry(name=RECORDS_SEND, schema={**SEND_SCHEMA, "name": RECORDS_SEND},
                                handler=room.transport.tool, timeout_sec=30, safety_policy="skip"))
    registry.register(ToolEntry(name=RECORDS_REACT, schema=REACT_SCHEMA, handler=room.transport.react,
                                timeout_sec=30, safety_policy="skip"))
    if setup is not None:
        setup(ctx)
    task = {"id": root_id, "type": "task", "chat_id": 1, "text": "Look again", "metadata": meta,
            "delegation_role": "root", "_skip_post_task_synthesis": True}
    write_task_result(room.data, root_id, "running", metadata=meta, delegation_role="root", root_task_id=root_id,
                      chat_id=1)
    script, inputs = iter(responses), []

    def respond(_llm, messages, *_a, **_kw):
        inputs.append(json.loads(json.dumps(messages)))
        response = next(script, None)
        return (response(messages) if callable(response) else response), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    sent_before, events = list(room.transport.provider), queue.Queue()
    text, usage, trace = loop.run_llm_loop(
        [{"role": "user", "content": "Look again"}], registry, SimpleNamespace(default_model=lambda: "test-model"),
        room.data / "logs", lambda *_a, **_kw: None, queue.Queue(), task_id=root_id, drive_root=room.data,
        event_queue=events)
    pending = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=room.data, repo_dir=room.h.repo), None, None, pending,
                               task, text, usage, trace, 0.0, room.data / "logs", ctx=ctx)
    return SimpleNamespace(id=root_id, trace=trace, inputs=inputs, stored=load_task_result(room.data, root_id),
                           pending=pending, ctx=ctx, meta=meta, usage=usage,
                           sent=room.transport.provider[len(sent_before):], events=list(events.queue))


def _explicit_profile(ctx, authority="explicit"):
    """An active in-task ceiling under an explicit ``cost_hard_stop_pct`` (or a producer allowance), as resolved."""
    from ouroboros import task_pacing

    ctx._cost_ceiling = task_pacing.CostCeiling(state=task_pacing.COST_CEILING_ACTIVE, ceiling_usd=40.0,
                                                root_cap_usd=50.0, basis="test profile")
    ctx._cost_stop_policy = {"authority": getattr(task_pacing, f"COST_STOP_{authority.upper()}"), "profile": {},
                             "root_cap_usd": 50.0, "non_root_member": False, "root_ceiling_usd": 40.0,
                             "producer_allowance_lifted": False}


def _prices(monkeypatch, **initial):
    """The price oracle behind the budget owner's exact probe (``loop_budget._exact_reservations``) and its proxy:
    ``fits``/``two`` answer one and two reservations, ``fail`` makes the request unpriceable, ``wallet`` is the known
    global remainder (``None``: unknown). Every exact request it prices is kept in ``seen``. The comparisons, last-fit
    relaxation and rails that read it are the real ones."""
    from ouroboros import task_pacing

    prices = {"fits": True, "two": True, "fail": False, "wallet": 10.0, "seen": [], **initial}

    def request(*, messages, **_kwargs):
        if prices["fail"]:
            raise RuntimeError("no price for this route")
        prices["seen"].append(json.loads(json.dumps(messages)))
        return {"priced_rows": len(messages)}

    def reservation_fits(*, reservation_count=1, **_kwargs):
        return prices["fits"] if reservation_count == 1 else prices["two"]

    monkeypatch.setattr(task_pacing, "prospective_wrapup_attempt_request", request)
    monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", reservation_fits)
    monkeypatch.setattr("ouroboros.loop_budget._wrapup_global_remaining", lambda: prices["wallet"])
    return prices


@pytest.mark.parametrize("held", [False, True], ids=["ready", "held"])
def test_a_ready_finish_outranks_the_last_fit_margin_and_a_held_one_meets_it(room, monkeypatch, held):
    """A strict last fit (one reservation fits, two do not, no owner relaxation) is a margin for work still to buy.
    A finish selected in a batch is ready: it ends the task as it is, with no wrap-up request, pause or forced
    finalization bought over it. A finish its own gate holds is unfinished work again: the same margin stops the next
    send with the last-fit pause, priced as the stop's own request, before any second model call."""
    from ouroboros.budget_pause import RAIL_WRAPUP_LAST_FIT, BudgetPauseRequested

    prices, presence, script = _prices(monkeypatch, two=False), carrier(room), [finish(silent=True, answer=NOTE)] * 2
    if held:  # after the carrier's own turn: the hold stands for any delivery hold of this root's selected finish
        answers, sent = iter([True]), []
        monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery",
                            lambda *_a, **_k: next(answers, False))
        with pytest.raises(BudgetPauseRequested, match=RAIL_WRAPUP_LAST_FIT):
            run_root(room, monkeypatch, [lambda rows: sent.append(rows) or script[0], script[1]],
                     setup=_explicit_profile, presence=presence)
        assert len(sent) == 1, "the margin stopped the round the held finish would have bought"
        assert any("[BUDGET LIMIT]" in str(rows[-1]["content"]) for rows in prices["seen"])
        return
    run = run_root(room, monkeypatch, script, setup=_explicit_profile, presence=presence)
    assert len(run.inputs) == 1, "no model call after the finish"
    assert "forced_finalization" not in run.trace and not run.stored.get("budget_pause")
    assert not any("[BUDGET LIMIT]" in json.dumps(rows, ensure_ascii=False) for rows in prices["seen"] + run.inputs)
    assert run.stored["result"] == NOTE and run.stored["metadata"]["presence_outcome"] == "silent" and run.sent == []

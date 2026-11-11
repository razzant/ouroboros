"""A live call's tool lane: requests as host facts, effects outside the history lock, whole call/result units.

Effects run one batch at a time on the host's effect lane. History is locked only to record: a
request becomes a host fact before its effect (and in the pre-effect checkpoint), bound to the view
that produced it. The effect and any call tool's handler then run outside the history lock, on the ordinary context
with a staged message list and tool binding, and the completed call/result unit is written whole.
Captions, mail, peer facts and adopted connections keep writing history meanwhile, so a fit taken
during an effect holds its request, never a dangling call.
"""
from __future__ import annotations

import asyncio
import copy
import functools
import json
import logging

from ouroboros.contracts.live_conversation import ToolBatch
from ouroboros.contracts.realtime import RealtimeSetupError, canonical_json, sha256
from ouroboros.realtime_session import joined

log = logging.getLogger(__name__)
REQUEST_KIND = "call_tool_request"
CALL_END = "the call's end"


class CallEnded(Exception):
    """End reached an effect that had not begun: recorded as a control is, but no failure of the call."""


def _recorded_end(host):
    """What the call has itself taken as its end. The task's control reader cannot see it: a drained Stop is excluded as
    already delivered, and End is the page's. Also read on worker threads, by the gates of the work its batch
    started."""
    if host._end_reason is not None:
        return host._end_reason[0]
    return CALL_END if host.hub.closed or host.hub.ended_by_owner else None


async def _dispatch_refusal(host):
    """Why no effect may begin now, in the call's own terms, read where one begins: after every wait for history, a
    checkpoint or an earlier effect. An effect already begun is not stopped: it finishes and is recorded, while each
    later nested start of its body is a new start that its own gate checks."""
    try:
        await host.check()
    except Exception as exc:  # unreadable authority admits no effect either
        return exc
    return CallEnded() if _recorded_end(host) else None


def record_request(host, call):
    """The call's request as one host fact with its complete arguments; the caller holds the history lock.

    Its issuing view is kept by reference: a later connection replaces that observation, never edits
    its rows, so a queued call still answers to the send that produced it.
    """
    from ouroboros.context_budget import bound_host_row
    from ouroboros.utils import utc_now_iso
    key, function = call["id"], call["function"]
    if key in host.executed or key in host.requested:
        return
    host.requested[key] = {"name": function["name"], "state": "requested",
                           "view": getattr(host.ctx, "_last_context_observation", None)}
    host.messages.append(bound_host_row(REQUEST_KIND, "[Host fact; not owner words]\n"
        f"At {utc_now_iso()} you requested {function['name']} (call {key}), arguments: "
        f"{function.get('arguments') or ''}. Its call and result join this history "
        "together when it finishes; until then its outcome is unknown. Do not request it again unless asked."))
    host._context_changed()


def unrun_requests(host):
    """Host facts, once, for requests whose effect never started; the caller holds the history lock at call end."""
    from ouroboros.context_budget import bound_host_row
    rows = []
    for key, request in host.requested.items():
        if request["state"] == "requested":
            request["state"] = "not_run"
            rows.append(bound_host_row(REQUEST_KIND, "[Host fact; not owner words]\n"
                                       f"The requested {request['name']} (call {key}) did not run: the call ended first."))
    return rows


async def tool_batch(host, calls):
    """Deduplicate by call id, then run the fresh calls on the effect lane; return the history changes."""
    from ouroboros.live_context_view import drain_exposure
    from ouroboros.realtime_setup import rows_for_engine
    await drain_exposure(host)
    async with host._effect_lock:
        fresh, cached, proposed = [], [], {}
        for call in calls:
            key, identity = call["id"], canonical_json(call)
            previous = proposed.get(key)
            if previous is not None:
                if previous != identity:
                    raise RealtimeSetupError("tool_identity", "conflicting call ids in one provider batch")
                continue
            proposed[key] = identity
            if key in host.executed:
                if host.executed[key][0] != identity:
                    raise RealtimeSetupError("tool_identity", "provider reused a call id with different arguments")
                cached.extend(copy.deepcopy(host.executed[key][1]))
            else:
                fresh.append(call)
        silent = [call["id"] for call in calls if (host.call_tools.get(call["function"]["name"]) is not None
                   and host.call_tools[call["function"]["name"]].silent)]
        if not fresh:
            return ToolBatch(rows_for_engine(cached, vision=host.model.vision, schemas=host.schemas), silent=silent)
        await host.check()
        job = asyncio.create_task(_effect(host, fresh, cached, silent))
        host.tool_jobs.add(job)
        try:
            return await asyncio.shield(job)
        except asyncio.CancelledError:
            await joined(job)
            raise
        finally:
            if job.done():
                host.tool_jobs.discard(job)


async def _effect(host, fresh, cached, silent):
    from ouroboros.live_context_view import project_call_tool_results
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.loop import _append_or_merge_user_content, _measure_tool_result_batch, _resolve_loop_max_rounds
    from ouroboros.loop_tool_execution import process_tool_results
    from ouroboros.realtime_setup import rows_for_engine
    from ouroboros.tools.tool_discovery import bind_resident_schemas
    async with host._tools_lock:
        host.rounds += 1
        host.ctx._live_tool_step = f"{host.ctx.task_id}:call-tools:{host.rounds}"  # Safety's step (_authoring_call)
        maximum = _resolve_loop_max_rounds(host.ctx)
        if maximum is not None and host.rounds > maximum:
            raise RealtimeSetupError("round_limit", "ordinary tool round limit reached")
        for call in fresh:
            record_request(host, call)
        await joined(asyncio.to_thread(host.checkpoint, "pre_effect"))
        for call in fresh:
            host.requested[call["id"]]["state"] = "started"
        staged, base = copy.deepcopy(host.messages), copy.deepcopy(host.schemas)
        staged_from, schemas = len(staged), copy.deepcopy(base)
        lane = host.tools.copy_for(host.ctx)
        loaded = bind_resident_schemas(lane, schemas)  # enable_tools loads into the staged schemas
        host.ctx.messages = staged  # an effect's own rows (an attached image) follow its unit
        host.ctx._issued_views = {call["id"]: host.requested[call["id"]].pop("view", None) for call in fresh}
    ordinary = [call for call in fresh if call["function"]["name"] not in host.call_tools]
    results, failure, changes = [], None, []
    try:
        if ordinary:  # what ended the batch comes back beside its results and leaves only after they are history
            results, failure = await joined(asyncio.to_thread(_ordinary_batch, host, ordinary, lane))
    finally:
        host.ctx.messages, host.ctx._issued_views = host.messages, {}
    if isinstance(failure, PhysicalDispatchInterrupted):  # refused as it began: say why in the call's own terms
        failure = await _dispatch_refusal(host) or failure
    before, version, replies, ran = None, None, [], []
    if any(call["function"]["name"] in host.call_tools for call in fresh):
        async with host._tools_lock:
            before, version = _receipts(host), host._history_version
        replies, ran, failure = await _call_tools(host, fresh, failure)
    try:
        async with host._tools_lock:
            if schemas != base:  # the effect loaded tools; an adopted connection may have replaced the list since
                if host.schemas == base:
                    host.schemas[:] = schemas
                else:
                    names = {schema["function"]["name"] for schema in host.schemas}
                    host.schemas.extend(schema for schema in schemas if schema["function"]["name"] not in names)
            host.ctx._actor_loaded_tool_names.update(loaded)
            # With no other writer since, what changed beyond the receipts is a handler's own; it follows the unit.
            if before is None or host._history_version != version:
                before = _receipts(host)
            own = host.messages[len(before):]
            del host.messages[len(before):]
            unit = {"role": "assistant", "content": "", "tool_calls": fresh}
            host.messages.append(unit)
            start = len(host.messages)
            if results:  # on current history, so the batch's image eviction and projection see what is there now
                await joined(asyncio.to_thread(process_tool_results, results, host.messages, host.trace,
                    host.ctx.emit_progress_fn, tools=host.tools, tool_schemas=host.schemas,
                    project_results=functools.partial(project_call_tool_results, host), fit_candidate=functools.partial(
                        _measure_tool_result_batch, host.ctx, round_id=host.ctx.task_id + ":call-tools")))
            host.trace["tool_calls"].extend(ran)
            host.messages[start + len(results):start + len(results)] = replies  # adjacent, before trailing rows
            host.messages.extend(own)
            for row in staged[staged_from:]:
                if row.get("role") == "user" and set(row) <= {"role", "content"}:
                    _append_or_merge_user_content(host.messages, row["content"], stamp=host.ctx)
                else:
                    host.messages.append(row)
            host._context_changed()
            changes = [copy.deepcopy(row) for i, row in enumerate(host.messages)
                       if row is not unit and (i >= len(before) or sha256(canonical_json(row)) != before[i])]
            for call in fresh:
                host.executed[call["id"]] = (canonical_json(call), [
                    row for row in changes if row.get("tool_call_id") == call["id"]])
            await joined(asyncio.to_thread(host.checkpoint, "post_batch"))
    except Exception:
        if failure is None:
            raise
        log.warning("A batch ending the call was not fully recorded", exc_info=True)  # what ended it still does
    if failure is not None and not isinstance(failure, CallEnded):
        raise failure  # terminal for the call, after its batch is whole in history
    return ToolBatch(rows_for_engine(cached + changes, vision=host.model.vision, schemas=host.schemas), silent=silent)


def _ordinary_batch(host, calls, lane):
    """The batch's ordinary handlers, on this worker thread. Every context they start is copied from here, so each
    invocation, one the batch stops waiting for included, reads this call's End and drained Stop at its own later gates
    for as long as it lasts, and nothing started elsewhere does."""
    from ouroboros.loop_tool_execution import execute_tool_batch
    from ouroboros.owner_pause import call_end_reader
    with call_end_reader(host.ctx, functools.partial(_recorded_end, host)):
        return execute_tool_batch(calls, lane, host.ctx.drive_root / "logs", host.ctx.task_id, host.executor,
                                  host.trace)


def _receipts(host):
    """Compact content receipts detect even in-place image/view edits, retaining no second conversation."""
    return tuple(sha256(canonical_json(row)) for row in host.messages)


async def _call_tools(host, fresh, failure):
    """Run the batch's call tools in order, outside the history lock: a handler may record facts and captions may land.
    After a failure, ordinary or their own, or once the call has ended, none starts."""
    from ouroboros.tool_call_log import invocation_scope
    replies, ran = [], []
    for call in fresh:
        tool = host.call_tools.get(call["function"]["name"])
        if tool is None:
            continue
        if failure is None:
            failure = await _dispatch_refusal(host)
        if failure is not None:
            ended = isinstance(failure, (CallEnded, RealtimeSetupError))
            why = "the call ended first" if ended else "an earlier call in this batch failed"
            result = f"Call tool did not run: {why}."
        else:
            try:
                args = json.loads(call["function"]["arguments"])
                if not isinstance(args, dict): raise ValueError("arguments must be an object")
                with invocation_scope({"tool_call_id": call["id"]}):
                    result = await tool.handler(host, args)
            except (ValueError, TypeError) as exc:
                result = f"Call tool argument error: {exc}"
            except Exception as exc:
                failure, result = exc, f"Call tool failed: {type(exc).__name__}"
        replies.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})
        ran.append({"name": tool.name, "call_id": call["id"], "result": str(result)})
    return replies, ran, failure

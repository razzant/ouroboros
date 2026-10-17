"""Tool calls of a live call: one serialized batch worker over the unchanged tool execution.

The realtime session asks for tools while it speaks; the turn's tools run exactly
as in a text turn (``handle_tool_calls`` over the turn's own messages, with the
same measured delivery frame), on ONE worker thread so only one batch mutates the
tool context at a time. Every message change a batch made (tool rows, images a
tool viewed, review or dialogue updates) goes back to the session typed: a tool
row as ``ToolResult`` to the session that asked, the pixels a tool attached as
that result's ``images`` (the engine accepts images only on a tool result), and
anything else as a ``ContextItem``. A result whose asking session is gone (a
reconnect) or that the provider withdrew is retained and resent as context. A
provider call id is claimed the moment it is asked for: a second request for the
same id joins the pending or running execution and receives its result, and a
call already executed is answered from its retained result, so no physical
effect happens twice. A Presence call counts its autonomous tool rounds per
correspondent input against the binding's ``inline_max_rounds``; past that
ceiling a request gets a typed tool result instead of running.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import copy
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.gateways.realtime_engine import ContextItem, ToolCall, ToolResult, UpdateTools
from ouroboros.usage_accounting import BudgetExceeded

log = logging.getLogger(__name__)

ROUND_LIMIT_CODE = "TOOL_ROUND_LIMIT"


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
    return "" if content is None else str(content)


def images_of(content: Any) -> Tuple[bytes, ...]:
    images: List[bytes] = []
    for part in content if isinstance(content, list) else ():
        url = str(((part or {}).get("image_url") or {}).get("url") or "") if isinstance(part, dict) else ""
        if url.startswith("data:") and "," in url:
            try:
                images.append(base64.b64decode(url.split(",", 1)[1]))
            except Exception:
                log.debug("undecodable inline image skipped", exc_info=True)
    return tuple(images)


def round_limit_result(call_id: str, limit: int) -> ToolResult:
    """The typed answer a Presence call gives past its binding's autonomous round ceiling."""
    return ToolResult(call_id=call_id, output=(
        f"⚠️ {ROUND_LIMIT_CODE}: this call's binding allows {limit} autonomous tool round(s) per correspondent "
        f"input and that limit is reached. The call did not run this tool. Answer with what you already have; "
        f"the count starts again at the next utterance heard in the call."))


class ToolBatchWorker:
    """The call's tool lane: ``call`` is the live call whose messages, tools and session it serves."""

    def __init__(self, call: Any) -> None:
        self.call = call
        # (session epoch, call, the provider response it came from: one response is one round)
        self.pending: List[Tuple[int, ToolCall, str]] = []
        self.cancelled: set = set()
        self.executed: Dict[str, ToolResult] = {}
        # Every call id asked for and not yet answered, with the latest session epoch
        # that asked: a repeated request joins this execution instead of starting another.
        self.claimed: Dict[str, int] = {}
        self.running = False
        self.batches = 0
        self.tool_calls = 0
        # Autonomous rounds since the last correspondent input (a Presence call's ceiling):
        # the provider responses whose calls ran, so a response's calls share one round
        # however they arrive; calls of no known response count one round per batch.
        self.round_origins: set = set()
        self._takes = 0  # names each take of the pending calls, for calls of no known response
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"call-tools-{call.task_id}")

    def shutdown(self) -> None:
        # Wait for an idle worker; a tool still running is not waited on twice (the
        # call already gave it its settle window).
        self._pool.shutdown(wait=not self.running, cancel_futures=True)

    def enqueue(self, tool_call: ToolCall, epoch: int, origin: str = "") -> None:
        if tool_call.call_id in self.executed:
            # The provider asked again for a call already executed (a resent request
            # after a reconnect): the retained result answers, nothing runs twice.
            asyncio.create_task(self.call._send(self.executed[tool_call.call_id]))
            return
        if tool_call.call_id in self.claimed:
            # Pending or running already: the new session joins that one execution.
            self.claimed[tool_call.call_id] = epoch
            return
        self.claimed[tool_call.call_id] = epoch
        self.pending.append((epoch, tool_call, origin))

    def cancel(self, call_ids: Tuple[str, ...]) -> None:
        self.cancelled.update(call_ids)
        withdrawn = {tc.call_id for _epoch, tc, _origin in self.pending if tc.call_id in call_ids}
        self.pending = [entry for entry in self.pending if entry[1].call_id not in call_ids]
        for call_id in withdrawn:  # never started: the claim is released with the request
            self.claimed.pop(call_id, None)

    def reset_rounds(self) -> None:
        """A correspondent spoke: the autonomous round count starts again."""
        self.round_origins.clear()

    @property
    def rounds_since_input(self) -> int:
        return len(self.round_origins)

    def _round_limit(self) -> Optional[int]:
        if not getattr(self.call.spec, "presence_binding_id", ""):
            return None
        limit = getattr(self.call.ctx, "inline_max_rounds", None)
        return limit if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0 else None

    async def run_forever(self) -> None:
        call = self.call
        while True:
            if not self.pending:
                await asyncio.sleep(0.02)
                continue
            pending, self.pending, self._takes = self.pending, [], self._takes + 1
            limit, unnamed, batch = self._round_limit(), f"take:{self._takes}", []
            for epoch, tc, origin in pending:
                round_key = origin or unnamed
                if limit is not None and round_key not in self.round_origins:
                    if len(self.round_origins) >= limit:
                        result = round_limit_result(tc.call_id, limit)
                        self.executed[tc.call_id] = result
                        asked = self.claimed.pop(tc.call_id, epoch)
                        if asked == call.epoch:
                            await call._send(result)
                        else:  # the session that asked is gone: the answer stays as context
                            await call._send(ContextItem(text=f"[result of tool call {tc.call_id}]\n{result.output}",
                                                         source="tool_result_retained"))
                        continue
                    self.round_origins.add(round_key)
                batch.append((epoch, tc))
            if not batch:
                call._checkpoint()
                continue
            self.running = True  # cleared by the worker thread itself, so a cancelled await never hides a running tool
            call._activity("working", ", ".join(sorted({tc.name for _epoch, tc in batch})))
            try:
                changes, grown = await asyncio.get_running_loop().run_in_executor(
                    self._pool, contextvars.copy_context().run, self._run_batch, [tc for _epoch, tc in batch])
            except BudgetExceeded as exc:
                call.budget_error = exc
                call._begin_close("budget_exceeded")
                return
            except Exception:
                log.warning("live call %s tool batch failed", call.spec.call_id, exc_info=True)
                changes, grown = [], False
            call._activity("idle")
            # The latest session that asked for each call receives its result.
            epochs = {tc.call_id: self.claimed.pop(tc.call_id, epoch) for epoch, tc in batch}
            await self._forward_changes(changes, epochs)
            if grown:
                await call._send(UpdateTools(tuple(call.tool_schemas)))
            call._checkpoint()

    def _run_batch(self, calls: List[ToolCall]) -> Tuple[List[Dict[str, Any]], bool]:
        """On the worker thread: the unchanged tool execution over the turn's own messages."""
        from ouroboros import loop as loop_mod

        call = self.call
        try:
            tool_calls = [{"id": tc.call_id, "type": "function",
                           "function": {"name": tc.name, "arguments": tc.arguments}} for tc in calls]
            call.messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
            before = copy.deepcopy(call.messages)
            names_before = {schema["function"]["name"] for schema in call.tool_schemas}
            self.batches += 1
            self.tool_calls += len(calls)
            loop_mod.handle_tool_calls(
                tool_calls, call.tools, call.drive_logs, call.task_id, call.stateful_executor, call.messages,
                call.llm_trace, call.emit_progress, tool_schemas=call.tool_schemas,
                fit_candidate=functools.partial(loop_mod._measure_tool_result_batch, call.ctx,
                                                round_id=f"{call.task_id}:call:{self.batches}"))
            changes = [msg for index, msg in enumerate(call.messages) if not msg.get("_call_row")
                       and (index >= len(before) or msg != before[index])]
            grown = {schema["function"]["name"] for schema in call.tool_schemas} != names_before
            return changes, grown
        finally:
            self.running = False

    async def _forward_changes(self, changes: List[Dict[str, Any]], epochs: Dict[str, int]) -> None:
        """Every message change the batch made goes back to the session, typed.

        A tool attaches the image it viewed as a user row before its own tool row
        (``vision.attach_local_image_to_context``), so pixels gathered from the
        rows of this batch ride on the next tool result of the batch; pixels after
        the last tool row ride on that last result. With no tool row to carry them
        the omission is named, never silent.
        """
        call = self.call
        carried: Dict[int, List[bytes]] = {}   # index of the tool row -> the pixels it carries
        gathered: List[bytes] = []
        last_tool_index = -1
        for index, msg in enumerate(changes):
            if str(msg.get("role") or "") == "tool":
                carried[index] = [*images_of(msg.get("content")), *gathered]
                gathered, last_tool_index = [], index
            else:
                gathered.extend(images_of(msg.get("content")))
        if gathered and last_tool_index >= 0:
            carried[last_tool_index].extend(gathered)
            gathered = []
        for index, msg in enumerate(changes):
            role, content = str(msg.get("role") or ""), msg.get("content")
            if role == "tool":
                call_id = str(msg.get("tool_call_id") or "")
                result = ToolResult(call_id=call_id, output=text_of(content), images=tuple(carried[index]))
                self.executed[call_id] = result
                if epochs.get(call_id) == call.epoch and call_id not in self.cancelled:
                    await call._send(result)
                else:  # the session that asked is gone or withdrew the call: the result stays as context
                    omitted = (f"\n[{len(result.images)} image(s) of this result cannot travel as context]"
                               if result.images else "")
                    await call._send(ContextItem(text=f"[result of tool call {call_id}]\n{result.output}{omitted}",
                                                 source="tool_result_retained"))
            else:
                text = text_of(content)
                if text.strip():
                    await call._send(ContextItem(text=text, source=f"tool_batch:{role or 'note'}"))
        if gathered:
            await call._send(ContextItem(text=f"[{len(gathered)} image(s) a tool attached could not reach the live "
                                              f"session: no tool result of this batch can carry them]",
                                         source="tool_batch:images_omitted"))

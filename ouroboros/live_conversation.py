"""The ordinary actor's host port for a loaded Live Conversation feature.

The feature moves events; this host port owns fitting, tool execution, canonical
history, spending and Stop. Neither wire packets nor durable task metadata can
mint this process-local turn. Loaded work remains registered through close; Stop joins owned tool work.
"""
from __future__ import annotations

import asyncio
import copy
import functools
import json
import logging
import math
import queue
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import replace

from ouroboros.contracts.live_conversation import LiveConfig, LiveOptions, ToolBatch
from ouroboros.contracts.realtime import RealtimeSetupError, canonical_json, sha256
from ouroboros.realtime_setup import _LoopWork, loaded_provider, rows_for_engine
from ouroboros.realtime_session import joined
from ouroboros.realtime_turn import RealtimeTurn

log = logging.getLogger(__name__)

_active_hosts = weakref.WeakSet()
_active_hosts_lock = threading.Lock()


def end_feature_calls(feature_key, *, reason="calls_disabled", detail="Calls (beta) are off."):
    """Stop this feature's live calls after a confirmed settings write, on every surface."""
    with _active_hosts_lock:
        for host in tuple(_active_hosts):
            if host.turn.feature["key"] == feature_key:
                host._end_reason = (reason, detail)


def loaded_feature(key):
    from ouroboros.extension_registry_state import _conversation_features, _extensions, _lock, _unloading

    with _lock:
        spec = _conversation_features.get(key)
        if spec is None or spec["skill"] in _unloading or spec["skill"] not in _extensions:
            raise RealtimeSetupError("feature_unavailable", str(key))
        return {**spec, "key": key}


@contextmanager
def feature_configuration(key):
    """A configure callback is loaded code too; guard its complete invocation."""
    from ouroboros.extension_registry_state import extension_work_scope

    spec = loaded_feature(key)

    class Read:
        def cancel(self):
            pass  # synchronous reader completes before its work scope exits

    with extension_work_scope(spec, Read()):
        config = spec["configure"]()
        if not isinstance(config, LiveConfig):
            raise RealtimeSetupError("configuration", "the feature must return LiveConfig")
        yield spec, config


def call_offer(config: LiveConfig) -> dict:
    """Feature config projection: enabled, loaded and all credential grants present."""
    if not isinstance(config, LiveConfig):
        raise RealtimeSetupError("configuration", "the feature must return LiveConfig")
    if not config.enabled:
        return {"offered": False, "reason": "calls_disabled", "detail": "Calls (beta) are off."}
    try:
        provider = loaded_provider(config.provider)
        provider["catalog"].model(config.model)
    except RealtimeSetupError as exc:
        return {"offered": False, "reason": exc.code, "detail": str(exc).removeprefix(exc.code + ": ")}
    from ouroboros.extension_registry_state import _extensions, _lock
    with _lock:
        bundle = _extensions.get(provider["skill"])
        apis = tuple(getattr(bundle, "api_instances", ()))
    required = provider["catalog"].credential_settings
    if not any(all(values.get(key) for key in required) for values in (api.get_settings(required) for api in apis)):
        return {"offered": False, "reason": "credential", "detail": "Grant this engine its credential in Skills."}
    return {"offered": True, "reason": "", "detail": ""}


class LiveConversationTurn(RealtimeTurn):
    accepts_owner_messages = True

    def __init__(self, feature, config, spec, *, publish):
        if not isinstance(config, LiveConfig):
            raise RealtimeSetupError("configuration", "the feature must return LiveConfig")
        offer = call_offer(config)
        if not offer["offered"]:
            raise RealtimeSetupError(offer["reason"], offer["detail"])
        self.feature, self.config, self.publish = feature, config, publish
        self.provider = config.provider
        provider = loaded_provider(self.provider)
        self.catalog_model = provider["catalog"].model(config.model)
        self.supports_images = self.catalog_model.vision  # read by the call-open admission reply (gateway/call_ws.py)
        self.route, self._generation = provider["route"], provider["extension_generation"]
        self.native_model = self.catalog_model.id
        self.model = self.route.model_prefix + self.native_model
        self.spec = replace(spec, model=self.native_model)
        # The call's first owner-visible chat row: the cue plus one owner-facing "Engine:" sentence that the
        # room's call card shows as its engine line (owner's choice 27.10). The model's own senses reach it
        # through the call instructions (skills/live_conversation/driver.py).
        self.opening = (("This call is now live: the other side can hear you." if spec.opened_by == "skill"
                         else "The owner has just opened this call and can hear you now.")
                        + f" Engine: {provider['catalog'].label}, {self.catalog_model.label or self.native_model}; "
                        + ("sees the shared screen." if self.catalog_model.vision else "no screen vision."))
        self.request = {"kind": "live_conversation", "feature": feature["key"], "provider": self.provider,
                        "model": self.native_model, "call_id": spec.call_id}
        self.admitted_at = time.time()  # the turn is built at admission, before the task is queued and prepared
        self._lock, self._closed = threading.Lock(), False

    def run(self, ctx, messages, tools, *, task, initial_effort, incoming_messages=None, event_queue=None):
        from ouroboros.call_peers import close_hub, get_hub
        from ouroboros.extension_registry_state import extension_work_scope
        from ouroboros.usage_accounting import BudgetExceeded

        hub = get_hub(self.spec.call_id)
        if hub is None:
            raise RealtimeSetupError("call_not_found", "this call's process-local hub ended")
        host = None

        def close_admission():
            from supervisor.workers import get_direct_chat_agent

            actor = get_direct_chat_agent(ctx.task_id)
            lock = getattr(actor, "_owner_message_admission_lock", None)
            if lock is not None:
                with lock:
                    if (getattr(actor, "_current_task_id", None) == ctx.task_id
                            and getattr(actor, "_realtime_turn", None) is self):
                        actor._accepting_owner_messages = False

        async def owned():
            work = _LoopWork()
            work.task = asyncio.current_task()
            with extension_work_scope(self.feature, work):
                children = ()
                try:
                    # The host is now registered: catch a disable during ordinary
                    # preparation, while the hook covers any subsequent write.
                    try:
                        current_config = self.feature["configure"]()
                    except Exception:
                        current_config = self.config  # failed reads preserve the admitted snapshot
                    if isinstance(current_config, LiveConfig) and not current_config.enabled:
                        raise RealtimeSetupError("calls_disabled", "Calls (beta) are off.")
                    await host.check()
                    runner = asyncio.create_task(self.feature["handler"](host))
                    controls = asyncio.create_task(host.watch_controls())
                    children = (runner, controls)
                    done, _ = await asyncio.wait((runner, controls), return_when=asyncio.FIRST_COMPLETED)
                    if controls in done:
                        await controls
                    return await runner
                finally:
                    # No new mail once the receive loop ends, including while
                    # tool custody and terminal-result post-work still run.
                    close_admission()
                    for child in children:
                        if not child.done():
                            child.cancel()
                    for child in children:
                        try:
                            await joined(child)
                        except (asyncio.CancelledError, Exception):
                            pass
                    # join any already-started tool invocation; cancellation of
                    # its await cannot make an executing effect disappear.
                    await host.finish()

        reason, status, detail = "finished", "ok", ""
        failure = None
        try:
            presence = (task.get("metadata") or {}).get("presence") or {}
            admitted_presence = (self.spec.opened_by == "skill"
                                 and presence.get("binding_id") == self.spec.presence_binding_id)
            if not ctx.is_direct_chat or not (task.get("origin_message_ref") or admitted_presence):
                raise RealtimeSetupError("owner_ingress", "call requires a canonical admitted owner turn")
            host = ConversationHost(self, ctx, messages, tools, task, initial_effort, hub)
            host.incoming, host.event_queue = incoming_messages, event_queue
            host.warm()  # catalog I/O finishes before the audio loop exists
            asyncio.run(owned())
        except BudgetExceeded:
            reason, status = "budget_exhausted", "failed"
            raise
        except (Exception, asyncio.CancelledError) as exc:
            from websockets.exceptions import ConnectionClosed
            if isinstance(exc, ConnectionClosed):
                # The engine closed its socket itself (billing, policy, an internal error): its close
                # code and its own sentence are what the owner needs to read, never a bare number.
                rcvd = getattr(exc, "rcvd", None)
                reason, status = "provider_closed", "infra_failed"
                detail = " ".join(part for part in (str(rcvd.code) if rcvd else "", (rcvd.reason or "") if rcvd else "")
                                  if part).strip()
            else:
                reason = str(getattr(exc, "code", "") or "") or ("cancelled" if isinstance(exc, asyncio.CancelledError) else "call_failed")
                status = ("cancelled" if reason in {"cancelled", "finalize_requested", "owner_stop", "not_connected"}
                          else "infra_failed")
                detail = str(exc).removeprefix(reason + ": ") if isinstance(exc, RealtimeSetupError) else ""
            if reason == "call_failed":
                import traceback
                from ouroboros.observability import redact_projection
                frames = traceback.extract_tb(exc.__traceback__)
                failure = redact_projection({
                    "class": type(exc).__name__, "message": str(exc),
                    "origin": f"{frames[-1].filename}:{frames[-1].lineno} in {frames[-1].name}" if frames else "",
                    "stack": [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                              for frame in frames],
                }).value
                detail = "An unexpected error ended the call."
            if host is not None:
                if failure is not None:
                    host.trace["failure"] = failure
                host.observe("host", canonical_json({"type": "call.ended", "reason": reason,
                    **({"failure": failure} if failure is not None else {})}))
        finally:
            close_admission()  # also covers setup failures before owned() starts
            try:
                if host is not None:
                    host.executor.shutdown(wait=True)
            finally:
                with self._lock:
                    self._closed = True
                try:
                    close_hub(self.spec.call_id, reason, detail=detail)
                    if host is not None:
                        host.observe("host", canonical_json({"type": "call.input_audio",
                            **hub.input_audio_evidence()}))
                finally:
                    if host is not None:
                        host.wire.close()  # the call's last records are on disk before it reports finished
                    self.publish({"type": "conversation_finished", "chat_id": self.spec.chat_id,
                                  "call_id": self.spec.call_id, "task_id": ctx.task_id, "reason": reason})
        if reason == "not_connected":  # nobody joined: one short notice, no transcript to point at
            words = f"The call did not start: {detail.rstrip('.')}."
        else:
            words = ("Call ended. Its transcript is in this room." if status in ("ok", "cancelled") or not detail
                     else f"Call ended: {detail.rstrip('.')}. Its transcript is in this room.")
        return words, {
            "terminal_origin": "host_notice", "execution_status": status, "reason_code": reason,
            **({"failure": failure} if failure is not None else {}),
        }, host.trace if host is not None else {"reasoning_notes": [], "tool_calls": []}


class WireJournal:
    """A call's wire evidence, written by its own thread: the call loop never waits on the disk.

    27.10: an open-append-close for every 20 ms frame on the call thread stood 132 s of a 2.3-minute
    OpenAI call under disk load; the browser's uplink backed up ("too slow to send live audio") and
    barge-ins dropped seconds of the owner's speech. Records keep their order. The writer stops by
    itself once nothing more arrives (a call streams frames until it ends), so no owner has to stop
    it; ``flush`` waits for the records so far, and after ``close`` a record is written at once.
    """

    IDLE_SEC = 0.25  # the writer's linger after the last record, far below a frame gap's cost

    def __init__(self, path):
        self.path = path
        self._queue = queue.SimpleQueue()
        self._thread = None
        self._closed = False
        self._lock = threading.Lock()

    def append(self, record):
        with self._lock:
            if self._closed:
                self._write((record,))
                return
            self._queue.put(record)
            if self._thread is None:
                self._thread = threading.Thread(target=self._drain, name="call-wire-journal", daemon=True)
                self._thread.start()

    def flush(self, timeout=30.0):
        """Wait until every record appended so far is on disk (a reader while the call is open)."""
        with self._lock:
            if self._thread is None:
                return  # nothing queued: the writer stops only on an empty queue
            written = threading.Event()
            self._queue.put(written)
        written.wait(timeout)

    def close(self, timeout=30.0):
        with self._lock:
            self._closed = True
            thread = self._thread
            if thread is not None:
                self._queue.put(None)
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                log.warning("call wire evidence still being written to %s after %.0f s", self.path, timeout)

    def _records(self, batch):
        """The records of ``batch`` in order; each flush marker is set once the records before it are written."""
        records = []
        for item in batch:
            if isinstance(item, threading.Event):
                self._write(records)
                records = []
                item.set()
            elif item is not None:
                records.append(item)
        return records

    def _drain(self):
        while True:
            try:
                batch = [self._queue.get(timeout=self.IDLE_SEC)]
            except queue.Empty:
                with self._lock:
                    if self._queue.empty():
                        self._thread = None
                        return
                continue
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self._write(self._records(batch))
            if None in batch:
                with self._lock:
                    self._thread = None
                return

    def _write(self, records):
        if not records:
            return
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.writelines(canonical_json(record) + "\n" for record in records)
        except OSError:
            log.warning("call wire evidence not written to %s", self.path, exc_info=True)


class ConversationHost:
    """Port given only to an admitted feature; all effects use ordinary owners."""

    def __init__(self, turn, ctx, messages, tools, task, effort, hub):
        from ouroboros.artifacts import task_artifact_dir_path
        from ouroboros.loop import _reset_turn_state, _setup_dynamic_tools, StatefulToolExecutor
        from ouroboros.tool_policy import initial_tool_schemas
        from ouroboros.tools.tool_discovery import update_resident_notice

        self.turn, self.ctx, self.messages, self.tools = turn, ctx, messages, tools
        self.task, self.effort, self.hub = task, effort, hub
        self.config, self.model = turn.config, turn.catalog_model
        self.opening, self.state = turn.opening, {}
        self.trace = {"reasoning_notes": [], "tool_calls": []}
        # Wall-clock facts of the call's start path, kept with the task trace and written to the
        # wire log at the end: where the seconds before the first answer went (owner, 18.10).
        # Admission precedes queueing and the ordinary context preparation; the host starts after them.
        self.timing = self.trace["timing"] = {"admitted_at": getattr(turn, "admitted_at", None) or time.time(),
                                              "host_started_at": time.time(), "connections": []}
        self.plan = ctx.context_fit_plan
        if self.plan.model != turn.model or self.plan.provider != turn.route.accounting_provider:
            raise RealtimeSetupError("route", "call input must use the selected realtime route")
        self.schemas = initial_tool_schemas(tools, context_mode=self.plan.initial_mode,
                                           schema_names=turn.feature.get("initial_tools", ()))
        self._schemas_mode = self.plan.initial_mode
        _reset_turn_state(ctx)
        self.call_tools = {tool.name: tool for tool in turn.feature.get("call_tools", ())}
        collisions = self.call_tools.keys() & {schema["function"]["name"] for schema in tools.schemas()}
        if collisions:
            raise RealtimeSetupError("tool_collision", "Call tools collide with ordinary tools: " + ", ".join(sorted(collisions)))
        self.schemas.extend({"type": "function", "function": {"name": tool.name, "description": tool.description,
            "parameters": dict(tool.parameters)}} for tool in self.call_tools.values())
        self.schemas, _ = _setup_dynamic_tools(tools, self.schemas, messages, context_mode=self.plan.initial_mode)
        # A call starts with its frame's tools, so it names the rest in its own manifest row;
        # a text turn reads them in the connections map.
        update_resident_notice(tools, self.schemas, messages)
        if turn.spec.presence_binding_id:
            from ouroboros.context import build_user_content
            from ouroboros.context_budget import bound_host_row
            # Presence's admitted task is the observed event, so its opening
            # fact joins separately with the ordinary host-fact label.
            messages.append(bound_host_row("call_open", build_user_content({
                "text": self.opening, "metadata": {"host_fact": {"kind": "call_open"}}})))
        ctx.messages, ctx.active_context_mode, ctx.active_effort = messages, self.plan.initial_mode, effort
        ctx._execution_trace = self.trace
        self.executor = StatefulToolExecutor()
        self.tool_jobs = set()
        self.executed, self.captioned = {}, {}
        self.wire_path = task_artifact_dir_path(ctx.drive_root, ctx.task_id, create=True) / "conversation-wire.jsonl"
        self.wire = WireJournal(self.wire_path)
        self.connection_number = 0
        self._connection_sequence = 0
        self._connection_tasks = []
        self._connection_view = None
        self._prepared_successor = None
        self._tools_lock = asyncio.Lock()
        self._rows_lock = asyncio.Lock()
        self.incoming = self.event_queue = None
        self.owner_seen = set()
        self.rounds = 0
        self.quote = None
        self.responses, self.completed = {}, set()
        self._physical_context = None
        self._renewal_window = self._renewal_facts = None
        self._held_input = self._growth = 0
        self._history_at_receipt = 0
        self._serving_history_at_receipt = None
        self._history_version, self._history_cached_version, self._history_tokens = 0, -1, 0
        self._selected_text_estimate = self._selected_raw_estimate = 0
        self._session_instructions = ""
        self._response_text_estimates = {}
        self._density_input = 0
        self._connection_accounting = False
        self._session_opened = False
        self._connection_quote = None
        self._meter_watermark = 0.0
        self._caption_dirty = False
        self._end_reason = None
        with _active_hosts_lock:
            _active_hosts.add(self)

    def _reselect_tools(self, mode):
        """Follow a lower fitted mode, retaining tools explicitly loaded this call."""
        from ouroboros.loop import _setup_dynamic_tools
        from ouroboros.tool_policy import initial_tool_schemas
        from ouroboros.tools.tool_discovery import update_resident_notice

        enabled = set(self.ctx._actor_loaded_tool_names)
        selected = initial_tool_schemas(self.tools, context_mode=mode,
                                        schema_names=self.turn.feature.get("initial_tools", ()))
        names = {schema["function"]["name"] for schema in selected}
        selected.extend(schema for schema in self.schemas
                        if schema["function"]["name"] in (enabled | self.call_tools.keys()) - names)
        self.schemas[:] = selected
        _, rebound = _setup_dynamic_tools(self.tools, self.schemas, self.messages, context_mode=mode)
        update_resident_notice(self.tools, self.schemas, self.messages)
        rebound.update(enabled)
        self._schemas_mode = mode
        self._context_changed()

    def checkpoint(self, boundary):
        from types import SimpleNamespace
        from ouroboros.working_checkpoint import save_round, flush_content_acks

        saved = save_round(SimpleNamespace(tools=self.tools, messages=self.messages, llm_trace=self.trace,
            accumulated_usage={}, round_idx=self.rounds, tool_schemas=self.schemas,
            owner_msg_seen=self.owner_seen), boundary)
        if saved:
            flush_content_acks(self.ctx)

    async def mailbox(self):
        """Use the ordinary typed drain; checkpoint before acknowledging content."""
        from ouroboros.loop_round_limits import _drain_incoming_messages

        if self.incoming is None:
            return []
        async with self._tools_lock:
            rows = []
            def drain():
                # A separate tail prevents merging new words into an item that
                # the provider has already accepted. Authority still comes from
                # the ordinary mailbox drain, including its relays and controls.
                controls = _drain_incoming_messages(rows, self.incoming, self.ctx.drive_root,
                    self.ctx.task_id, self.event_queue, self.owner_seen, owner_ctx=self.ctx,
                    defer_content_ack=True)
                self.messages.extend(rows)
                if rows:
                    # Call-owned before the cancellable connection worker gets
                    # the drain's return value. Canonical words still owe an offer.
                    self._pending_mail_answer = True
                    self._context_changed()
                    self.checkpoint("ready")
                return controls
            controls = await joined(asyncio.to_thread(drain))
            if controls.get("finalize_now") or controls.get("owner_stop"):
                raise RealtimeSetupError("owner_stop", "ordinary owner control ended the call")
            # A live delta reaches the engine as it can take it: an image attached in the room is a caption for a
            # blind engine (R15-1); the canonical rows above keep the image.
            return rows_for_engine(rows, vision=self.model.vision, schemas=self.schemas)

    def take_mail_answer(self):
        pending = getattr(self, '_pending_mail_answer', False)
        self._pending_mail_answer = False
        return pending

    def observe(self, direction, raw):
        self.wire.append({"connection": self.connection_number, "direction": direction, "raw": raw,
                          "at": round(time.time(), 3)})  # observation time, not capture or speech time
        if direction == "usage" and self._connection_accounting:
            from ouroboros.live_usage import observe_live_usage
            observe_live_usage(self, json.loads(raw))

    def _connection_growth(self, usage):
        counts = dict(usage)
        seconds = counts.get("connected_seconds")
        if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
            counts["connected_seconds"] = max(0, seconds - self._meter_watermark)
            self._meter_watermark = max(self._meter_watermark, seconds)
        return counts

    def _settle_connection_sample(self, counts, raw_usage):
        from ouroboros.live_usage import settle_connection_usage
        if counts.get("connected_seconds") == 0:
            return
        args = self._response_args("")
        args.pop("response_key")
        settle_connection_usage(**args, connection_key=str(self.connection_number),
            usage={"connected_seconds": counts["connected_seconds"]}, raw_usage=raw_usage,
            watermark=self._meter_watermark)

    async def check(self):
        from ouroboros.live_usage import check_live_budget
        from ouroboros.store import StoreBusy, acquisition_budget
        from ouroboros.usage_accounting import BudgetExceeded

        waiter = self.ctx.model_wait_context
        if waiter is None:
            raise RealtimeSetupError("host_control", "call has no ordinary model-wait owner")

        def probe():
            gap = None
            try:
                reason = waiter.control_reason(strict_current=True)
                if reason:
                    raise RealtimeSetupError(reason, "ordinary task control ended the call")
            except StoreBusy as exc:
                if exc.reason != "contention":
                    raise
                gap = exc
            # Check money even when the control read has a transient gap. Neither
            # unreadable authority admits an effect; the asynchronous owner waits.
            with acquisition_budget(0):
                check_live_budget(engine=self.turn.route.accounting_provider, model=self.turn.model,
                                  task_id=self.ctx.task_id, drive_root=self.ctx.drive_root)
            if gap is not None:
                raise gap

        while True:
            if self._end_reason is not None:
                raise RealtimeSetupError(*self._end_reason)
            reading = asyncio.create_task(asyncio.to_thread(probe))
            try:
                await asyncio.shield(reading)
                return
            except asyncio.CancelledError:
                try:
                    await joined(reading)
                except Exception:
                    pass  # cancellation still owns this completed read, never a new effect
                raise
            except StoreBusy as exc:
                if exc.reason != "contention":
                    raise
                await asyncio.sleep(.05)
            except BudgetExceeded as exc:
                raise RealtimeSetupError("budget_exhausted", "The shared spending limit has been reached.") from exc

    async def watch_controls(self):
        while True:
            await self.check()
            await asyncio.sleep(.05)

    def warm(self):
        from ouroboros.live_usage import warm_quote
        self.quote = warm_quote(engine=self.turn.route.accounting_provider, model=self.turn.model,
                                drive_root=self.ctx.drive_root)

    def fitted_frame_rows(self):
        return getattr(self, '_fitted_frame_rows', ())

    async def delivery_fits(self, rows, schemas):
        from ouroboros.live_context_view import serving_delivery_fits
        snapshot = copy.copy(self)
        snapshot.messages = copy.deepcopy(self.messages)
        return await asyncio.to_thread(serving_delivery_fits, snapshot, rows, schemas)

    def tool_results_delivered(self, rows):
        from ouroboros.live_context_view import delivered_call_tool_results
        delivered_call_tool_results(self, rows)

    async def tool_result_fallback(self, rows, schemas):
        from ouroboros.live_context_view import fallback_call_tool_results
        return await asyncio.to_thread(fallback_call_tool_results, self, rows, schemas)

    async def prepare_session(self, options: LiveOptions):
        """Open a silent successor from a private ordinary fit; adoption stays explicit."""
        from ouroboros.live_connections import PreparedConnection
        lease = PreparedConnection(self, options, ahead=True)
        try:
            if await asyncio.shield(lease.ready) is None:
                return None  # an older provider cannot silently catch up history
            self._prepared_successor = lease
            return lease
        except BaseException:
            await lease.close()
            raise

    async def session(self, consume, options: LiveOptions):
        from ouroboros.live_connections import PreparedConnection
        self.hub.pause_input()
        lease = PreparedConnection(self, options)
        try:
            result = await lease.run(consume)
            if result or self._prepared_successor is None:
                await joined(lease.task)
            return result
        except BaseException:
            await lease.close()
            raise

    def _response_args(self, response_id):
        return dict(engine=self.turn.route.accounting_provider, model=self.turn.model,
                    call_id=self.turn.spec.call_id, response_key=response_id,
                    task_id=self.ctx.task_id, rendered_mode=self.plan.initial_mode, drive_root=self.ctx.drive_root,
                    quote=self._connection_quote if self._connection_accounting else self.quote,
                    physical_context=self._physical_context)

    def response_started(self, response_id):
        from ouroboros.live_connections import connection_for
        if (view := connection_for(self)) is not None:
            return view.response_started(response_id)
        from ouroboros.live_usage import reserve_live_response, record_refused_live_response
        from ouroboros.usage_accounting import BudgetExceeded
        key = response_id
        if key in self.responses or key in self.completed:
            return
        self.timing.setdefault("first_response_at", time.time())
        # Only the first response has the sealed selection as its known text
        # denominator. Later native audio/transcripts must not become text input
        # witnesses merely because their captions entered canonical history.
        self._response_text_estimates[key], self._density_input = self._density_input, 0
        try:
            self.responses[key] = reserve_live_response(**self._response_args(response_id))
        except BudgetExceeded as exc:
            self.responses[key] = record_refused_live_response(**self._response_args(response_id), reason="budget_exhausted",
                usage={}, raw_usage={}, status="interrupted", pending=True)
            raise RealtimeSetupError("budget_exhausted", "The shared spending limit has been reached.") from exc

    def response_done(self, response_id, *, usage, raw_usage, status):
        from ouroboros.live_connections import connection_for, publish
        if (view := connection_for(self)) is not None:
            view.response_done(response_id, usage=usage, raw_usage=raw_usage, status=status)
            if view is self._connection_view:
                publish(self, view)
            if self._caption_dirty:
                self.checkpoint("ready")
                self._caption_dirty = False
            return
        from ouroboros.live_usage import settle_live_response, record_refused_live_response
        key = response_id
        counts = self._connection_growth(usage)
        if key in self.completed:
            if "connected_seconds" in counts:
                self._settle_connection_sample(counts, raw_usage)
            return
        if key not in self.responses:
            # A tool-only response or a closing receipt may have no start event.
            record_refused_live_response(**self._response_args(response_id), reason="receipt_without_start",
                                        usage=counts, raw_usage=raw_usage, status=status)
        else:
            settle_live_response(self.responses.pop(key), usage=counts, raw_usage=raw_usage, status=status)
        self.completed.add(key)
        self.observe("response", canonical_json({"response_id": response_id, "usage": dict(usage),
                                              "raw_usage": dict(raw_usage), "status": status}))
        from ouroboros.live_context_view import record_response_context
        record_response_context(self, key, usage, raw_usage, status)
        if self._caption_dirty:
            self.checkpoint("ready")
            self._caption_dirty = False

    def _text_estimate(self):
        from ouroboros.realtime_setup import text_input_estimate
        extra = [{"role": "system", "content": self._session_instructions}] if self._session_instructions else []
        return text_input_estimate([*self.messages, *extra], self.schemas)

    def _record_text_density(self, key, usage, raw_usage):
        from ouroboros.capability_evidence import record_token_density, MAIN_DENSITY_BASIS
        from ouroboros.provider_models import normalize_model_identity
        estimate = self._response_text_estimates.pop(key, self._density_input)
        self._density_input = 0
        from ouroboros.live_usage import text_input_witness
        text = text_input_witness(usage, raw_usage)
        # The adapter's text axis is a modality witness; aggregate prompt usage
        # and audio/image/video axes never teach the text estimator a density.
        if (self.model.facts.basis == "provider" and estimate > 0
                and type(text) is int and text > 0):
            record_token_density(self.ctx.drive_root, normalize_model_identity(self.turn.model),
                prompt_chars=estimate * 4, prompt_tokens=text, source="live_text_usage",
                route_fp=self.plan.route_fp, basis=MAIN_DENSITY_BASIS, view_kind=self.plan.view_kind)

    def _context_changed(self):
        self._history_version += 1

    def _serving_history_estimate(self):
        from ouroboros.context_fit import estimate_context_prompt_tokens
        extra = [{"role": "system", "content": self._session_instructions}] if self._session_instructions else []
        rows = rows_for_engine(self.messages, vision=self.model.vision, schemas=self.schemas)
        return estimate_context_prompt_tokens([*rows, *extra], self.schemas)

    def _history_estimate(self):
        from ouroboros.context_fit import estimate_context_prompt_tokens
        owner = getattr(self, "_connection_owner", self)
        version = owner._history_version if self.messages is owner.messages else self._history_version
        if self._history_cached_version != version:
            extra = [{"role": "system", "content": self._session_instructions}] if self._session_instructions else []
            projected = rows_for_engine(self.messages, vision=self.model.vision, fresh=True, owner_mode=self.plan.preferred_mode, schemas=self.schemas)
            self._history_tokens = estimate_context_prompt_tokens([*projected, *extra], self.schemas)
            self._history_cached_version = version
        return self._history_tokens

    def renewal_due(self):
        facts = self.model.facts
        window, reserve = facts.conversation_window_tokens, facts.working_reserve_tokens or 0
        # Session/response boundaries resolve stores. Audio polling only reads
        # that snapshot; a different catalog cannot borrow its resolved window.
        if self._renewal_window is not None and self._renewal_facts == facts:
            window, reserve = self._renewal_window.total_equivalent, self._renewal_window.reserve
        if not window or not self._selected_raw_estimate and not self._held_input:
            return False
        density = (getattr(self._physical_context, "measurement_density", 1.0)
                   if facts.basis == "provider" else 1.0)
        held = self._held_input or math.ceil(self._selected_raw_estimate * density)
        pending = math.ceil(max(0, self._history_estimate() - self._history_at_receipt) * density)
        return held + pending + self._growth + reserve >= window

    def context_view_pending(self):
        from ouroboros.live_context_view import pending_context_view
        return pending_context_view(self)

    async def apply_context_view(self):
        from ouroboros.live_context_view import apply_context_view
        from ouroboros.live_context_view import drain_exposure
        await drain_exposure(self)
        async with self._tools_lock:
            return await joined(asyncio.to_thread(apply_context_view, self))

    async def observe_context(self, row):
        from ouroboros.context_budget import bound_host_row
        async with self._tools_lock:
            if isinstance(row.get('content'), list):
                self.messages.append({**copy.deepcopy(row), '_host_context_kind': 'call_observation'})
            else:
                self.messages.append(bound_host_row("call_observation", "[Call peer observation] " + str(row.get("content", ""))))
            self._context_changed()

    def timing_event(self, event, *, at=None):
        from ouroboros.live_timing import record_timing
        record_timing(self, event, at=at)

    async def retain_frame(self, data, mime, *, source=""):
        """Use the existing content-addressed task media store and its retention."""
        from ouroboros.artifacts import store_chat_media_bytes
        from ouroboros.call_protocol import MAX_VIDEO_FRAME_PAYLOAD, VIDEO_FORMATS
        if mime not in VIDEO_FORMATS or not 0 < len(data) <= MAX_VIDEO_FRAME_PAYLOAD:
            raise RealtimeSetupError("video_format", "invalid retained call frame")
        saved = await joined(asyncio.to_thread(store_chat_media_bytes,
                            self.ctx.drive_root, self.ctx.task_id, data, mime))
        if not saved:
            raise RealtimeSetupError("frame_retention", "call frame could not be retained")
        return saved["path"]

    async def tool_batch(self, calls):
        """Serialize ordinary and feature tools; retain changes to existing rows too."""
        from ouroboros.live_context_view import drain_exposure
        await drain_exposure(self)
        async with self._tools_lock:
            fresh, cached, proposed = [], [], {}
            for call in calls:
                key, identity = call["id"], canonical_json(call)
                previous = proposed.get(key)
                if previous is not None:
                    if previous != identity:
                        raise RealtimeSetupError("tool_identity", "conflicting call ids in one provider batch")
                    continue
                proposed[key] = identity
                if key in self.executed:
                    if self.executed[key][0] != identity:
                        raise RealtimeSetupError("tool_identity", "provider reused a call id with different arguments")
                    cached.extend(copy.deepcopy(self.executed[key][1]))
                else:
                    fresh.append(call)
            silent = [call["id"] for call in calls if (self.call_tools.get(call["function"]["name"]) is not None
                       and self.call_tools[call["function"]["name"]].silent)]
            if not fresh:
                return ToolBatch(rows_for_engine(cached, vision=self.model.vision, schemas=self.schemas), silent=silent)
            await self.check()

            async def run():
                from ouroboros.loop import handle_tool_calls, _measure_tool_result_batch, _resolve_loop_max_rounds
                from ouroboros.live_context_view import project_call_tool_results
                from ouroboros.tool_call_log import invocation_scope
                self.rounds += 1
                maximum = _resolve_loop_max_rounds(self.ctx)
                if maximum is not None and self.rounds > maximum:
                    raise RealtimeSetupError("round_limit", "ordinary tool round limit reached")
                self.messages.append({"role": "assistant", "content": "", "tool_calls": fresh})
                self._context_changed()
                # A compact content receipt detects even in-place image/view edits,
                # without retaining a second copy of the complete conversation.
                before = tuple(sha256(canonical_json(row)) for row in self.messages)
                await joined(asyncio.to_thread(self.checkpoint, "pre_effect"))
                ordinary = [call for call in fresh if call["function"]["name"] not in self.call_tools]
                if ordinary:
                    try:
                        await joined(asyncio.to_thread(handle_tool_calls, ordinary, self.tools,
                            self.ctx.drive_root / "logs", self.ctx.task_id, self.executor, self.messages, self.trace,
                            self.ctx.emit_progress_fn, tool_schemas=self.schemas,
                            project_results=functools.partial(project_call_tool_results, self), fit_candidate=functools.partial(
                                _measure_tool_result_batch, self.ctx, round_id=self.ctx.task_id + ":call-tools")))
                    finally:
                        # Ordinary tools also edit existing rows and enable schemas.
                        self._context_changed()
                for call in fresh:
                    tool = self.call_tools.get(call["function"]["name"])
                    if tool is None:
                        continue
                    try:
                        args = json.loads(call["function"]["arguments"])
                        if not isinstance(args, dict): raise ValueError("arguments must be an object")
                        with invocation_scope({"tool_call_id": call["id"]}):
                            result = await tool.handler(self, args)
                    except (ValueError, TypeError) as exc:
                        result = f"Call tool argument error: {exc}"
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})
                    self._context_changed()
                    self.trace["tool_calls"].append({"name": tool.name, "call_id": call["id"], "result": str(result)})
                changes = [copy.deepcopy(row) for i, row in enumerate(self.messages)
                           if i >= len(before) or sha256(canonical_json(row)) != before[i]]
                for call in fresh:
                    self.executed[call["id"]] = (canonical_json(call), [
                        row for row in changes if row.get("tool_call_id") == call["id"]])
                await joined(asyncio.to_thread(self.checkpoint, "post_batch"))
                return ToolBatch(rows_for_engine(cached + changes, vision=self.model.vision, schemas=self.schemas), silent=silent)

            job = asyncio.create_task(run())
            self.tool_jobs.add(job)
            try:
                return await asyncio.shield(job)
            except asyncio.CancelledError:
                await joined(job)
                raise
            finally:
                if job.done():
                    self.tool_jobs.discard(job)

    async def caption(self, speaker, text, item_id, *, heard=False, played_ms=0, interrupted=False, completed=False,
                      replace=False, speaker_label=""):
        """Final words only; meeting input is never minted as the owner's message."""
        key = (speaker, item_id)
        if not text:
            return
        async with self._rows_lock, self._tools_lock:
            previous = self.captioned.get(key)
            if previous and (not replace or previous["text"] == text):
                return
            retained = {}

            def write():
                from ouroboros import dialogue_store
                from supervisor.message_bus import get_bridge, log_chat
                from ouroboros.chat_references import projection_fields
                from ouroboros.projects_registry import stamp_project_thread

                spec = self.turn.spec
                if previous:
                    row = dialogue_store.revise_text(self.ctx.drive_root, "chat", previous["seq"],
                                                    previous["stored_prefix"] + text)
                elif spec.presence_binding_id:
                    from ouroboros.presence_call import record_call_utterance
                    if not record_call_utterance(self.ctx.drive_root, self.task,
                            direction="out" if speaker == "ouroboros" else "in",
                            text=text, speaker_label=speaker_label, utterance_id=item_id):
                        raise RealtimeSetupError("history_write", "Presence caption could not be retained")
                    event_id = self.task["metadata"]["presence"]["event"]["source_event_id"]
                    name = f"{event_id}:{'spoken' if speaker == 'ouroboros' else 'heard'}:{item_id}"
                    retained["seq"] = dialogue_store.rows_by_client_id(
                        self.ctx.drive_root, name, chat_id=spec.chat_id)[-1].seq
                    return
                elif speaker == "owner" and not heard:
                    message_id = f"call:{spec.call_id}:{item_id}"
                    row = log_chat("in", spec.chat_id, 1, text, source="live_call", sender_label="voice",
                        client_message_id=message_id, task_id=self.ctx.task_id, drive_root=self.ctx.drive_root, require_write=True)
                    if not row:
                        raise RealtimeSetupError("history_write", "owner call caption was not recorded")
                    retained["seq"] = dialogue_store.accepted_row(self.ctx.drive_root, spec.chat_id, message_id).seq
                else:
                    from ouroboros.utils import utc_now_iso
                    spoken = speaker == "ouroboros"
                    meta = {"source": "live_call", "call_id": spec.call_id}
                    if spoken:
                        meta["spoken"] = {"call_id": spec.call_id, "item_id": item_id, "played_ms": played_ms,
                            "status": "interrupted" if interrupted else "played" if completed else "generated"}
                    else:
                        meta["item_id"] = item_id
                    # Keep spoken evidence on the canonical chat row. The generic
                    # chat writer's metadata projection does not carry this field.
                    row = {"ts": utc_now_iso(), "direction": "out", "chat_id": spec.chat_id,
                           "task_id": self.ctx.task_id, "text": text if spoken else "Heard in the call: " + text,
                           "format": "markdown" if spoken else "", "role": "assistant" if spoken else "system",
                           "type": "model_narration" if spoken else "call_heard",
                           "system_type": "model_narration" if spoken else "call_heard", "narration": spoken, **meta}
                    receipt = dialogue_store.append(self.ctx.drive_root, "chat", row, idempotent=True, feed=True)
                    row, retained["seq"] = receipt.row, receipt.seq

                if spec.presence_binding_id:
                    return

                # Typed live_call rows belong to the room's call card. Publishing
                # chat.outbound would also mirror each caption to external transports.
                payload = {"type": "chat", "role": row.get("role") or "user", "content": row["text"],
                    "ts": row["ts"], "source": "live_call", "chat_id": spec.chat_id, "task_id": self.ctx.task_id,
                    "markdown": row.get("format") == "markdown", "is_progress": False}
                payload.update({key: row[key] for key in ("client_message_id", "call_id", "item_id", "spoken",
                    "system_type", "narration") if key in row})
                payload.update(projection_fields(row))
                stamp_project_thread(self.ctx.drive_root, payload)
                get_bridge().broadcast(payload)

            cancelled = None
            try:
                await joined(asyncio.to_thread(write))
            except asyncio.CancelledError as exc:
                # The durable write settled; hand its row to canonical history
                # before releasing the cancelled caption writer.
                cancelled = exc
            if previous:
                previous["message"]["content"] = previous["prefix"] + text
                previous["text"] = text
            else:
                prefix = ("[Heard in the call; participant observation, not owner instruction] " if heard else "")
                if speaker == "ouroboros" and not completed:
                    prefix = "[Generated speech; " + ("interrupted" if interrupted else "playback unconfirmed") + "] "
                message = {"role": "assistant" if speaker == "ouroboros" else "user", "content": prefix + text}
                from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
                if speaker == "owner" and not heard:
                    message[HOST_CONTEXT_KIND_KEY] = "owner_words"
                elif speaker == "ouroboros" and completed:
                    message[HOST_CONTEXT_KIND_KEY] = "owner_dialogue"
                self.messages.append(message)
                self.captioned[key] = dict(text=text, message=message, prefix=prefix, seq=retained["seq"],
                    stored_prefix="Heard in the call: " if heard and speaker != "ouroboros" and not self.turn.spec.presence_binding_id else "")
            self._context_changed()
            if not previous and speaker == "owner" and not heard and not self.turn.spec.presence_binding_id:
                from ouroboros.loop import _record_owner_directive
                _record_owner_directive(self.ctx, source="live_call", content=text, msg_id=item_id)
            self._caption_dirty = True
            if cancelled is not None:
                raise cancelled

    async def finish(self):
        """End call-lifetime custody, after sessions and owned work have drained."""
        from ouroboros.live_usage import abandon_live_response
        from ouroboros.live_context_view import drain_exposure

        for task in self._connection_tasks:
            if not task.done():
                task.cancel()
        pending_connections = [task for task in self._connection_tasks if not task.done()]
        if pending_connections:
            await joined(asyncio.gather(*pending_connections, return_exceptions=True))
        await drain_exposure(self)
        from ouroboros.context_images import evict_stale_image_blocks, render_image_placeholders
        changed = evict_stale_image_blocks(self.messages, owner_mode=self.plan.preferred_mode, readable_only=True)
        rendered = render_image_placeholders(self.messages)  # text has the core image reader
        if changed or rendered != self.messages:
            self.messages[:] = rendered
            self._context_changed()
            await joined(asyncio.to_thread(self.checkpoint, "ready"))
        with _active_hosts_lock:
            _active_hosts.discard(self)
        try:
            try:
                for job in list(self.tool_jobs):
                    await joined(job)
            finally:
                self.tool_jobs.clear()
                try:
                    for key, attempt in list(self.responses.items()):
                        abandon_live_response(attempt, reason="call_ended_without_receipt")
                        del self.responses[key]
                finally:
                    self._response_text_estimates.clear()
                    self._density_input = 0
                    await joined(asyncio.to_thread(self.executor.shutdown, wait=True))
        finally:
            # Evidence last, after every custody step above has run (a cancellation at any of their
            # awaits included): a failed wire-log write never skips custody and never replaces the
            # reason this call ended.
            try:
                self.timing_event("closed")
                self.observe("host", canonical_json({"type": "call.timing", **self.timing}))
            except OSError:
                log.warning("call %s: timing evidence not written", self.turn.spec.call_id, exc_info=True)
            self.wire.close()  # the call's records are on disk; later teardown rows are written at once

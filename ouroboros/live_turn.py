"""Live voice-call driver: one realtime engine session is the turn's model round.

A call is an ordinary turn whose model speaks and listens (ARCHITECTURE §6 "Live
calls"): the agent prepares the text turn's context, this driver opens the engine
with that fitted system text and the room's Nano tool view (``live_call_route``)
and pumps until the call ends: peer audio in, engine audio out through the hub,
captions to the peers and the chat (owner speech as an owner row WITHOUT an owner
turn, Ouroboros speech as the turn's narration with its played mark), tools on one
serialized worker (``live_tool_batch``), the owner's mailbox (Stop, Hurry, typed
words) on a short tick with deferred acknowledgement, and the room's new rows
(``live_awareness``, from the instant the context was captured) as context. Session
caps and transport losses reopen with a history view or the provider's resume
handle; context the call heard is in its transcript and waits while no session is
live. A Stop during the connection cancels the opening; a tool result the transport
refuses is a typed failure the model hears; a session the provider refuses ends
with its own words. The price quote warms on the ledger's thread, never holding the
open. The phases (preparing, connecting, live) and each closing reason with its one
next step come from here and the hub. Returns the loop's ``(text, usage, llm_trace)``
with a host-notice terminal (``terminal_origin=host_notice``) naming the transcript.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import pathlib
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.call_protocol import (
    ACTION_RETRY, ACTION_SETTINGS, PEER_MEETING, PEER_OWNER, STATE_CLOSED, STATE_CLOSING, STATE_CONNECTING,
    STATE_LIVE, STATE_RECONNECTING, CallSpec, PeerAudio, PeerControl, PeerJoined, PeerLeft,
)
from ouroboros.gateways.realtime_engine import (
    SPEAKER_OUROBOROS, SPEAKER_OWNER, USAGE_KEYS,
    AudioIn, AudioOut, Caption, Close, Closed, ContextItem, EngineError, GoAway, Interrupt, PlayedAudio, Ready,
    ResponseDone, ResponseStarted, ResumeCheckpoint, SessionConfig, SpeechStarted, ToolCall, ToolCallCancelled,
    ToolResult, UpdateTools,
)
from ouroboros.gateways.realtime_transport import LiveEngineRefusal
from ouroboros.live_call_route import call_minutes_ceiling, initial_call_schemas, native_search_enabled, resolve_call_route
from ouroboros.live_session_view import (
    calibration_ratio, compression_watermarks, history_view, record_density_witness, text_estimate,
)
from ouroboros.live_tool_batch import ToolBatchWorker, text_of
from ouroboros.usage_accounting import BudgetExceeded

log = logging.getLogger(__name__)

SOURCE_CALL = "call"   # the owner-speech row source (live_awareness skips it)

# Physical/transport bounds, not behaviour: the Stop tick, the ack wait before a spoken row is
# written as generated, the rejoin grace of an empty room, unproven sessions in a row, the first
# (doubling) wait before reopening after one, how long a session stays up after its setup to count
# as working, the margin before a provider's session cap and the wait for a tool batch at close.
MAILBOX_TICK_S = 0.25
SPOKEN_ACK_GRACE_S = 10.0
PEER_GRACE_S = 20.0
RECONNECT_ATTEMPTS = 3
RECONNECT_BACKOFF_S = 0.5
SESSION_PROVEN_S = 10.0
SESSION_CAP_MARGIN_S = 60.0
BATCH_SETTLE_WAIT_S = 5.0
PLANNED_HANDOVERS = ("go_away", "session_cap")  # the provider's or the host's own renewals, not losses

# The one next step the owner's surface offers beside a closed call's reason.
_ACTION_BY_REASON = {
    "missing_credential": ACTION_SETTINGS, "authentication": ACTION_SETTINGS, "engine_refused": ACTION_SETTINGS,
    "rate_limit": ACTION_RETRY, "provider_closed": ACTION_RETRY, "provider_error": ACTION_RETRY,
    "protocol": ACTION_RETRY, "backpressure": ACTION_RETRY,  # a refused request: the same call would repeat it
    "engine_unavailable": ACTION_RETRY, "reconnect_failed": ACTION_RETRY,
}


# ---- seams the lanes meet at (tests replace these) --------------------------------

@dataclass(frozen=True)
class _Acks:
    """A delivery boundary: the content acknowledgements owed for what was posted before it."""

    callbacks: Tuple[Callable[[], Any], ...]


def _hub_for(call_id: str):
    from ouroboros.call_peers import get_hub

    return get_hub(call_id)


def _ledger():
    from ouroboros import live_usage

    return live_usage


def _loop():
    """The tool loop's members, read at call time (tests rebind them there)."""
    from ouroboros import loop

    return loop


def _narration_meta(emit_progress: Callable[..., None]) -> Dict[str, Any]:
    owner = getattr(emit_progress, "__self__", None)
    build = getattr(owner, "_subagent_progress_meta", None)
    try:
        return dict(build("progress")) if callable(build) else {}
    except Exception:
        return {}


@dataclass
class _Spoken:
    text: str = ""
    final_at: float = 0.0
    played_ms: int = 0
    played: bool = False
    interrupted: bool = False
    audio_ms: float = 0
    ack_at: float = 0
    started_at: float = 0.0  # when its first audio went to the peers


class _LiveCall:
    """One call from open to close, on the turn's thread with its own event loop."""

    def __init__(self, *, messages, tools, llm, drive_logs, emit_progress, incoming_messages, task_id, task_type,
                 event_queue, drive_root, spec: CallSpec, task: Optional[Dict[str, Any]] = None,
                 context_captured_at: str = "") -> None:
        self.messages, self.tools, self.llm, self.drive_logs = messages, tools, llm, drive_logs
        self.task = task if isinstance(task, dict) else {}
        self.emit_progress, self.incoming, self.task_id = emit_progress, incoming_messages, task_id
        self.task_type, self.event_queue, self.drive_root, self.spec = task_type, event_queue, drive_root, spec
        # The instant the turn's context was captured: the room feed catches up from
        # there, so a row accepted while the context was being prepared is not lost.
        self.context_captured_at = context_captured_at
        self.ctx = tools._ctx
        self.route = resolve_call_route(spec, drive_root)
        # The PCM rates the engine wire takes and returns (Gemini Live: 16 kHz in, 24 kHz out);
        # audio is labelled with what the session declares, and a Ready that names others wins.
        self.input_rate, self.output_rate = self.route.engine.input_rate, self.route.engine.output_rate
        self.llm_trace: Dict[str, Any] = {"reasoning_notes": ["live_call"], "tool_calls": []}
        self.usage: Dict[str, Any] = {"_task_attempt": getattr(self.ctx, "task_attempt", None),
                                      "live_usage": {key: 0 for key in USAGE_KEYS}, "live_responses": 0}
        self.tool_schemas: List[Dict[str, Any]] = []
        self.hub = None
        self.engine = None
        self.epoch = 0
        self.state = STATE_CONNECTING
        self.started = time.monotonic()
        self.started_at_iso = ""
        self.session_started = self.started
        self.close_reason = ""
        self.close_detail = ""  # the provider's or the host's own sentence beside the reason
        self.closing = False
        self.reconnecting = False
        self.reconnects = 0
        self.unproven_reopens = 0  # sessions in a row that ended, unplanned, before proving themselves
        self.session_proven = False  # this session heard or said something
        self.ready_at: Optional[float] = None  # when this session's setup was answered
        self.resume_handle = ""
        # (session epoch, its opening config): the first response's text count becomes the route's
        # density witness unless other text reached the session before it.
        self.witness: Optional[Tuple[int, SessionConfig]] = None
        self.response_open = False
        self.budget_error: Optional[BudgetExceeded] = None
        self.failure: Optional[Tuple[str, str]] = None
        self.attempts: Dict[str, Future] = {}
        self.current_response = ""  # the response a tool call belongs to: one response is one Presence round
        # Ledger writes run in order on one thread beside the call's loop: a busy store
        # never stalls the audio, and each reservation still precedes its settlement.
        self.records = ThreadPoolExecutor(max_workers=1, thread_name_prefix="call-ledger")
        self.speaking_item = ""
        self.cancelled_items: set = set()
        self.played_ms: Dict[str, int] = {}
        self.playback_acked = False  # whether any peer reports what it played
        self.spoken: Dict[str, _Spoken] = {}
        self.counts = {"owner": 0, "ouroboros": 0, "heard": 0}
        self.muted_peers: set = set()
        # Meeting peers whose microphone feeds the session. While one is present the
        # engine hears a mix, so its input captions are never written as the owner's words.
        self.meeting_mics: set = set()
        # Whether meeting audio reached the engine since its last input caption: the
        # words of that caption came from the mix even if the meeting peer left meanwhile.
        self.meeting_audio_since_caption = False
        # The session's input audio as the provider counts it: where meeting audio sits in
        # it and where each input item starts, so an item's speaker follows the audio that
        # formed it however late its caption, its start or the meeting's departure arrive.
        self.input_ms = 0.0
        self.meeting_spans: List[Tuple[int, float, float]] = []  # (session epoch, start ms, end ms)
        self.input_starts: Dict[str, Tuple[int, float]] = {}
        self.inputs_done: set = set()  # captioned inputs whose start still bounds an earlier pending one
        # Participants whose audio the platform reported flowing since the last heard
        # caption: exactly one names that utterance's speaker, otherwise it stays unnamed.
        self.voices_since_caption: set = set()
        self.empty_since: Optional[float] = None
        self.owner_msg_seen: set = set()
        # Context and results that arrived while no session was live: delivered to a
        # resumed session, dropped when a reopened session replays the history view.
        self._queued: List[Any] = []
        self._closed = asyncio.Event()
        self._hub_events: Optional[asyncio.Queue] = None
        # Context and content acknowledgements leave in order through one delivery task, so a
        # send waiting on the transport never holds the mailbox, the peers or a close decision.
        self._outbox: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reader_stop = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self.tools_lane = ToolBatchWorker(self)
        self.stateful_executor = None
        self.rooms = None
        self.fanout = None

    # ---- lifecycle ---------------------------------------------------------------

    def run(self) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        from ouroboros.utils import utc_now_iso

        self.started_at_iso = utc_now_iso()
        self.hub = _hub_for(self.spec.call_id)
        if self.hub is None:
            return self._terminal("no_hub", failure=("live_call_hub_missing", "the call's peer hub is not open"))
        if self.hub.closed or self.hub.ended_by_owner:
            # Ended while its memory was being prepared: no engine is opened or paid for.
            self.close_reason = "owner_bye"
            self._close_hub()
            return self._terminal("owner_bye", failure=None)
        # The tariff quote is fetched first on the ledger's own thread: the open never waits
        # for it, and every ledger lookup queued behind it reads the warmed catalog.
        self._record(self._settle_safely, lambda: _ledger().warm_quote(engine=self.spec.engine, model=self.spec.model,
                                                                       drive_root=self.drive_root))
        try:
            asyncio.run(self._main())
        finally:
            self.tools_lane.shutdown()
            self.records.shutdown(wait=True)  # every ledger write lands before the terminal
            if self._reader is not None:
                self._reader.join(MAILBOX_TICK_S * 4)  # it leaves within one hub wait of the stop flag
        if self.budget_error is not None:
            raise self.budget_error
        return self._terminal(self.close_reason or "ended", failure=self.failure)

    async def _main(self) -> None:
        from ouroboros.live_awareness import RoomRows, UtteranceFanout
        from ouroboros.loop_budget import _LoopExitContext, _cleanup_loop_resources
        from ouroboros.loop_tool_execution import StatefulToolExecutor

        loop_mod = _loop()
        self._loop = asyncio.get_running_loop()
        self._hub_events, self._outbox = asyncio.Queue(), asyncio.Queue()
        loop_mod._reset_turn_state(self.ctx)
        loop_mod._initialize_owner_directives(self.ctx, self.messages)
        self.tool_schemas, _extra = loop_mod._setup_dynamic_tools(
            self.tools, list(initial_call_schemas(self.tools, self.spec)), self.messages, context_mode="nano")
        self.ctx.event_queue, self.ctx.task_id, self.ctx.messages = self.event_queue, self.task_id, self.messages
        self.ctx._accumulated_usage = self.usage
        self.ctx.active_context_mode = getattr(getattr(self.ctx, "context_fit_plan", None), "initial_mode", "nano")
        self.stateful_executor = StatefulToolExecutor()
        exit_ctx = _LoopExitContext(self.tools, self.drive_root, self.task_id, self.event_queue, self.drive_logs,
                                    self.usage, self.llm_trace, self.ctx, getattr(self.ctx, "_execution_trace", None))
        self.ctx._execution_trace = self.llm_trace
        self.rooms = RoomRows(self.drive_root, self.spec.chat_id, self.task_id, self.spec.call_id,
                              self.context_captured_at or self.started_at_iso)
        self.fanout = UtteranceFanout(self.drive_root, self.spec.chat_id, self.task_id, self.spec.call_id)
        self.rooms.start()
        self._reader = threading.Thread(target=self._read_hub, name=f"call-hub-{self.task_id}", daemon=True)
        self._reader.start()
        # The owner's mailbox, the peers and the tool lane run from the first moment, so a
        # Stop, a join or a control is handled while the provider connection is still pending.
        tasks: List[asyncio.Task] = [asyncio.create_task(self._pump_hub()), asyncio.create_task(self._tick()),
                                     asyncio.create_task(self.tools_lane.run_forever()),
                                     asyncio.create_task(self._deliver_forever())]
        try:
            opening = asyncio.create_task(self._open_engine())
            closed = asyncio.create_task(self._closed.wait())
            await asyncio.wait({opening, closed}, return_when=asyncio.FIRST_COMPLETED)
            if not opening.done():
                # Stop (or another close) during the connection: the opening is cancelled
                # and whatever the provider already opened is closed, so no session is paid for.
                opening.cancel()
            try:
                await opening
            except asyncio.CancelledError:
                if not self.closing:
                    raise
            except LiveEngineRefusal as exc:
                # A typed refusal before any session (a missing key, a malformed route) keeps its own words.
                self.failure = ("live_engine_refused", f"{exc.code}: {exc}")
                self._begin_close("missing_credential" if exc.code == "missing_credential" else "engine_refused",
                                  detail=str(exc))
            except Exception as exc:
                log.warning("live call %s could not open its engine", self.spec.call_id, exc_info=True)
                self.failure = ("live_engine_unavailable", f"{type(exc).__name__}: {exc}")
                self._begin_close("engine_unavailable")
            closed.cancel()
            await self._closed.wait()
        except BudgetExceeded as exc:
            self.budget_error = exc
            self._begin_close("budget_exceeded")
        finally:
            for task in tasks:
                task.cancel()
            await self._shutdown()
            if self.budget_error is not None:
                exit_ctx.attach_exception_evidence(self.budget_error)
            _cleanup_loop_resources(self.stateful_executor, exit_ctx)

    async def _shutdown(self) -> None:
        self.hub.send_state(STATE_CLOSING, self.close_reason)
        if self.engine is not None:
            engine, self.engine = self.engine, None
            self.epoch += 1  # anything the old iterator still yields is stale
            for action in (lambda: engine.send(Close(self.close_reason)), lambda: engine.close(self.close_reason)):
                try:
                    await asyncio.wait_for(action(), timeout=BATCH_SETTLE_WAIT_S)
                except Exception:
                    log.debug("engine close step failed", exc_info=True)
        deadline = time.monotonic() + BATCH_SETTLE_WAIT_S
        while self.tools_lane.running and time.monotonic() < deadline:  # a running tool settles through custody
            await asyncio.sleep(0.05)
        for item_id in list(self.spoken):
            self._write_spoken_row(item_id)
        for key in list(self.attempts):
            self._abandon(self.attempts.pop(key), self.close_reason or "closed")
        try:
            self.fanout.flush()
        except Exception:
            log.debug("call %s final fan-out failed", self.spec.call_id, exc_info=True)
        self._checkpoint()
        self.rooms.stop()
        self._reader_stop.set()
        self._close_hub()

    def _close_hub(self) -> None:
        """The closed state with its words and one next step, then the hub itself."""
        self.state = STATE_CLOSED
        try:
            self.hub.send_state(STATE_CLOSED, self.close_reason, detail=self.close_detail,
                                action=_ACTION_BY_REASON.get(self.close_reason, ""))
            self.hub.close(self.close_reason)
        except Exception:
            log.debug("hub close failed", exc_info=True)

    def _begin_close(self, reason: str, *, detail: str = "") -> None:
        if self.closing:
            return
        self.closing, self.close_reason, self.state = True, reason, STATE_CLOSING
        self.close_detail = detail
        self._closed.set()

    def _activity(self, activity: str, label: str = "") -> None:
        """What the owner's surface shows while a tool batch runs (the tool's own name), and after it."""
        try:
            self.hub.send_activity(activity, label)
        except Exception:
            log.debug("call activity unsent", exc_info=True)

    def _terminal(self, reason: str, *, failure: Optional[Tuple[str, str]]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE

        minutes = round((time.monotonic() - self.started) / 60.0, 1)
        self.usage.update({
            "terminal_origin": TERMINAL_ORIGIN_HOST_NOTICE,
            "live_call": {"call_id": self.spec.call_id, "engine": self.spec.engine, "model": self.spec.model,
                          "ended": reason, "minutes": minutes, "reconnects": self.reconnects,
                          "tool_calls": self.tools_lane.tool_calls, "batches": self.tools_lane.batches, **self.counts},
        })
        label = self.spec.label or self.spec.call_id
        if failure is not None:
            code, detail = failure
            self.usage.update({"execution_status": "infra_failed", "reason_code": code})
            return (f"⚠️ Live call {label} could not run: {detail}", self.usage, self.llm_trace)
        text = (f"Live call {label} ended ({reason}). The transcript is in this chat: "
                f"{self.counts['owner']} owner, {self.counts['heard']} heard and "
                f"{self.counts['ouroboros']} Ouroboros utterances, {self.tools_lane.tool_calls} tool calls, {minutes} min.")
        if self.counts.get("unrecorded"):
            text += f" {self.counts['unrecorded']} utterances could not be written to the conversation record."
        return text, self.usage, self.llm_trace

    # ---- engine session ----------------------------------------------------------

    def _session_config(self, *, history: Tuple[Dict[str, Any], ...] = (), resume_handle: str = "") -> SessionConfig:
        instructions = text_of(self.messages[0].get("content")) if self.messages and self.messages[0].get("role") == "system" else ""
        compress_at, compress_to = compression_watermarks(self.route.fact, instructions, self.tool_schemas,
                                                          calibration_ratio(self.ctx))
        return SessionConfig(route=self.route.engine, instructions=instructions, tools=tuple(self.tool_schemas),
                             history=history, native_search=native_search_enabled(self.spec), resume_handle=resume_handle,
                             compress_at_tokens=compress_at, compress_to_tokens=compress_to)

    async def _open_engine(self, *, resume_handle: str = "") -> None:
        if self.closing or self.hub.ended_by_owner:
            return  # the owner ended the call before its engine opened: nothing is opened or paid for
        if not self.reconnecting:
            self.hub.send_state(STATE_CONNECTING)
        engine = self.llm.open_live(self.route.engine)
        # The history view holds every transcript row up to this count; context recorded
        # after it, while the connection was pending, is delivered once the session is live.
        history, cut = ((), -1) if resume_handle else (self._history_view(), len(self.messages))
        config = self._session_config(history=history, resume_handle=resume_handle)
        try:
            await engine.open(config)
        except asyncio.CancelledError:
            try:  # the call ended while connecting: whatever was opened is closed at once
                await asyncio.wait_for(engine.close("cancelled"), timeout=BATCH_SETTLE_WAIT_S)
            except Exception:
                log.debug("cancelled engine close failed", exc_info=True)
            raise
        self.engine, self.epoch, self.session_started = engine, self.epoch + 1, time.monotonic()
        self.session_proven, self.ready_at = False, None
        self.witness = None if resume_handle else (self.epoch, config)
        self.input_ms = 0.0  # a new session counts its input audio from zero
        self.hub.set_engine_rates(self.input_rate, self.output_rate)
        if self.empty_since is None and not self.hub.peers():
            # A call nobody attached to yet runs the same rejoin grace as one everyone left:
            # a peer whose socket never arrived must not hold the session (or a Presence slot).
            self.empty_since = time.monotonic()
        asyncio.create_task(self._pump_engine(engine, self.epoch))
        queued, self._queued = self._queued, []
        for rows_then, command in queued:
            # A resumed session keeps what it was sent before the loss, so everything that
            # waited is new to it; a replayed history view already holds the rows it covers.
            if resume_handle or rows_then > cut:
                await self._send(command)

    def _history_view(self) -> Tuple[Dict[str, Any], ...]:
        """The newest call rows a fresh session replays (``live_session_view.history_view``)."""
        return history_view(self.route.fact, self.messages, self.tool_schemas, calibration_ratio(self.ctx))

    async def _reconnect(self, reason: str, *, resume: bool) -> None:
        if self.closing or self.reconnecting:
            return
        # A session that heard or said something, or stayed up after its setup, was working, whatever ends it;
        lived = self.ready_at is not None and time.monotonic() - self.ready_at >= SESSION_PROVEN_S
        if self.session_proven or lived:
            self.unproven_reopens = 0
        elif reason not in PLANNED_HANDOVERS:  # an unproven session's planned handover changes no count
            self.unproven_reopens += 1
            if self.unproven_reopens >= RECONNECT_ATTEMPTS:
                self.failure = ("live_engine_unavailable", f"reopened sessions kept ending ({reason})")
                self._begin_close("reconnect_failed", detail=f"the session kept ending right after it opened ({reason})")
                return
        self.reconnecting, self.state = True, STATE_RECONNECTING
        self.hub.send_state(STATE_RECONNECTING, reason)
        old, self.engine = self.engine, None
        self.epoch += 1  # events of the lost session are stale from here
        if old is not None:
            try:
                await asyncio.wait_for(old.close(reason), timeout=BATCH_SETTLE_WAIT_S)
            except Exception:
                log.debug("lost engine close failed", exc_info=True)
        for key in list(self.attempts):
            self._abandon(self.attempts.pop(key), reason)
        self.response_open = False
        if self.unproven_reopens and reason not in PLANNED_HANDOVERS:  # a brief outage keeps its attempts
            await asyncio.sleep(RECONNECT_BACKOFF_S * 2 ** (self.unproven_reopens - 1))
            if self.closing:
                self.reconnecting = False
                return
        handle = self.resume_handle if resume else ""
        for attempt_no in range(RECONNECT_ATTEMPTS):
            try:
                await self._open_engine(resume_handle=handle)
                self.reconnects += 1
                self.reconnecting = False
                return
            except Exception:
                log.warning("live call %s reopen attempt %s failed", self.spec.call_id, attempt_no + 1, exc_info=True)
                await asyncio.sleep(0.5 * (2 ** attempt_no))
        self.reconnecting = False
        self.failure = ("live_engine_unavailable", f"the engine could not be reopened after {reason}")
        self._begin_close("reconnect_failed", detail=f"the session ended ({reason}) and could not be reopened")

    async def _send(self, command: Any) -> None:
        engine = self.engine
        if engine is None:
            if isinstance(command, (ContextItem, ToolResult, UpdateTools)) and not self.closing:
                self._queued.append((len(self.messages), command))  # audio and playback marks are transient; context waits
            return
        if isinstance(command, (ContextItem, ToolResult, UpdateTools)):
            self.witness = None  # the session now holds more text than it was opened with
        try:
            await engine.send(command)
        except LiveEngineRefusal as exc:
            log.debug("engine refused %s (%s)", type(command).__name__, exc.code, exc_info=True)
            if isinstance(command, ToolResult) and exc.code != "closed":
                await self._refused_tool_result(engine, command, exc)
        except Exception:
            log.debug("engine send failed (%s)", type(command).__name__, exc_info=True)

    async def _refused_tool_result(self, engine: Any, command: ToolResult, refusal: LiveEngineRefusal) -> None:
        """The transport could not carry a tool's result: the model learns that, typed, not silence."""
        images = f" and {len(command.images)} image(s), which no history view replays," if command.images else ""
        notice = (f"⚠️ TOOL_RESULT_REFUSED ({refusal.code}): the live session could not carry the result of tool "
                  f"call {command.call_id}: {refusal}. The tool did run; its {len(command.output)}-character text"
                  f"{images} stays in the turn's transcript. Ask again with a narrower request if you need it spoken.")
        try:
            await engine.send(ToolResult(call_id=command.call_id, output=notice))
        except Exception:
            try:  # the call id itself is what the session refuses: the notice travels as context
                await engine.send(ContextItem(text=notice, source="tool_result_refused"))
            except Exception:
                log.warning("live call %s could not report a refused tool result", self.spec.call_id, exc_info=True)

    # ---- pumps -------------------------------------------------------------------

    async def _pump_engine(self, engine: Any, epoch: int) -> None:
        lost = "stream_ended"
        try:
            async for event in engine.events():
                if epoch != self.epoch:
                    return  # a stale session's event
                await self._on_engine_event(event)
        except BudgetExceeded as exc:
            self.budget_error = exc
            self._begin_close("budget_exceeded")
            return
        except Exception:
            log.warning("live call %s engine stream failed", self.spec.call_id, exc_info=True)
            lost = "transport_lost"
        if epoch == self.epoch and not self.closing:  # the session ended without the call ending
            asyncio.create_task(self._reconnect(lost, resume=True))

    def _read_hub(self) -> None:
        while not self._reader_stop.is_set():
            try:
                event = self.hub.next_event(MAILBOX_TICK_S)
            except Exception:
                log.debug("hub read failed", exc_info=True)
                break
            if event is not None and self._loop is not None:
                try:
                    self._loop.call_soon_threadsafe(self._hub_events.put_nowait, event)
                except RuntimeError:
                    break

    async def _pump_hub(self) -> None:
        while True:
            await self._on_hub_event(await self._hub_events.get())

    async def _tick(self) -> None:
        while not self.closing:
            await asyncio.sleep(MAILBOX_TICK_S)
            await self._drain()
            now = time.monotonic()
            ceiling = call_minutes_ceiling(self.spec)
            if ceiling is not None and now - self.started >= ceiling * 60:
                self._begin_close("max_minutes")
            cap = self.route.fact.session_cap_minutes if self.route.fact else None
            if (cap and not self.response_open and not self.reconnecting
                    and now - self.session_started >= cap * 60 - SESSION_CAP_MARGIN_S):
                asyncio.create_task(self._reconnect("session_cap", resume=False))
            for item_id, spoken in list(self.spoken.items()):
                if spoken.final_at and now - max(spoken.final_at, spoken.ack_at) >= SPOKEN_ACK_GRACE_S:
                    self._write_spoken_row(item_id)
            if self.empty_since is not None and now - self.empty_since >= PEER_GRACE_S:
                self._begin_close("peers_left")
            if self.fanout.pending and not self.response_open:
                self.fanout.flush()

    # ---- hub events ---------------------------------------------------------------

    async def _context(self, item: ContextItem) -> None:
        """Context the call heard is part of its transcript: a reopened session replays it."""
        if not item.text.strip():
            return
        self.messages.append({"role": "user", "content": item.text, "_call_row": True, "_context_source": item.source})
        self._post(item)

    def _post(self, item: Any) -> None:
        if self._outbox is not None:
            self._outbox.put_nowait(item)

    async def _deliver_forever(self) -> None:
        while True:
            item = await self._outbox.get()
            if isinstance(item, _Acks):
                # Content is acknowledged only once a live session holds it and the checkpoint
                # is written; otherwise its acknowledgements wait for a later boundary (and
                # unacknowledged entries are redelivered).
                if self.engine is not None and not self.closing and self._checkpoint():
                    for ack in item.callbacks:
                        try:
                            ack()
                        except Exception:
                            log.warning("live call %s mail ACK unwritten; it will be delivered again",
                                        self.spec.call_id, exc_info=True)
                else:
                    self.ctx._pending_content_acks = [*item.callbacks,
                                                      *(getattr(self.ctx, "_pending_content_acks", None) or [])]
                continue
            await self._send(item)

    async def _on_hub_event(self, event: Any) -> None:
        if isinstance(event, PeerAudio):
            if (self.engine is not None and not self.closing and not self.reconnecting
                    and event.peer_id not in self.muted_peers):
                rate = self.input_rate
                duration = len(event.pcm) * 500 / rate
                if event.role == PEER_MEETING:
                    self.meeting_audio_since_caption = True
                    self._note_meeting_audio(self.input_ms, self.input_ms + duration)
                self.input_ms += duration
                await self._send(AudioIn(pcm=event.pcm, rate=rate))
        elif isinstance(event, PeerControl):
            kind = str(event.message.get("type") or "")
            if kind == "played":
                self.playback_acked = True
                item_id, played_ms = str(event.message.get("item_id") or ""), int(event.message.get("played_ms") or 0)
                self.played_ms[item_id] = max(self.played_ms.get(item_id, 0), played_ms)
                if item_id in self.spoken:
                    self.spoken[item_id].played, self.spoken[item_id].played_ms = True, self.played_ms[item_id]
                    self.spoken[item_id].ack_at = time.monotonic()
                    if self.spoken[item_id].final_at and played_ms >= self.spoken[item_id].audio_ms:
                        self._write_spoken_row(item_id)
                await self._send(PlayedAudio(item_id=item_id, played_ms=played_ms))
            elif kind == "mute":
                (self.muted_peers.add if event.message.get("muted") else self.muted_peers.discard)(event.peer_id)
            elif kind == "bye" and event.role == PEER_OWNER:
                self._begin_close("owner_bye")
            elif kind == "context":
                await self._context(ContextItem(text=str(event.message.get("text") or ""),
                                                source=f"peer:{event.role}:{event.message.get('source') or 'observation'}"))
            elif kind == "participant":
                label, speaking = str(event.message.get("label") or "someone"), bool(event.message.get("speaking"))
                if speaking and event.message.get("label"):
                    self.voices_since_caption.add(label)
                await self._context(ContextItem(text=f"{label} {'started' if speaking else 'stopped'} speaking",
                                                source="meeting_participant"))
        elif isinstance(event, PeerJoined):
            self.empty_since = None
            if event.peer.role == PEER_MEETING and event.peer.mic:
                self.meeting_mics.add(event.peer.peer_id)
        elif isinstance(event, PeerLeft):
            self.muted_peers.discard(event.peer_id)
            self.meeting_mics.discard(event.peer_id)
            if not self.hub.peers():
                self.empty_since = time.monotonic()

    # ---- engine events ------------------------------------------------------------

    async def _on_engine_event(self, event: Any) -> None:
        if isinstance(event, (Caption, ResponseDone)):
            self.session_proven = True  # the session heard or said something: it is alive
        if isinstance(event, AudioOut):
            if event.item_id not in self.cancelled_items:
                if event.item_id and event.item_id != self.speaking_item:
                    # Frames carry no item id: an empty, non-final caption names the item
                    # before its first frame, so played acknowledgements bind to it.
                    self.hub.send_caption(SPEAKER_OUROBOROS, "", False, event.item_id)
                self.speaking_item = event.item_id
                spoken = self.spoken.setdefault(event.item_id, _Spoken())
                spoken.started_at = spoken.started_at or time.monotonic()
                spoken.audio_ms += len(event.pcm) * 500 / event.rate
                self.hub.play(event.pcm, event.item_id)
        elif isinstance(event, Caption):
            self.hub.send_caption(event.speaker, event.text, event.final, event.item_id)
            if event.final:
                self._on_final_caption(event)
        elif isinstance(event, SpeechStarted):
            if event.item_id and event.audio_start_ms >= 0:
                self.input_starts[event.item_id] = (self.epoch, float(event.audio_start_ms))
            await self._barge_in()
        elif isinstance(event, ToolCall):
            self.tools_lane.enqueue(event, self.epoch, self.current_response)
        elif isinstance(event, ToolCallCancelled):
            self.tools_lane.cancel(event.call_ids)
        elif isinstance(event, ResponseStarted):
            self.response_open, self.current_response = True, event.response_id or event.item_id
            self._attempt_for(event.response_id or event.item_id)
        elif isinstance(event, ResponseDone):
            self.response_open, self.current_response = False, ""
            self._settle(event)
            await self._drain()
            if self.fanout.pending:
                self.fanout.flush()
        elif isinstance(event, ResumeCheckpoint):
            self.resume_handle = event.handle if event.resumable else ""
        elif isinstance(event, GoAway):
            asyncio.create_task(self._reconnect("go_away", resume=True))
        elif isinstance(event, EngineError):
            if event.retryable:
                asyncio.create_task(self._reconnect(f"engine_error:{event.kind}", resume=True))
            else:
                self.failure = ("live_engine_error", f"{event.kind}: {event.message}")
                self._begin_close(event.kind, detail=event.provider_words)  # only the provider's words are quoted
        elif isinstance(event, Closed):
            if not self.closing:
                asyncio.create_task(self._reconnect(f"transport_lost:{event.reason}", resume=True))
        elif isinstance(event, Ready):
            self.state, self.ready_at = STATE_LIVE, time.monotonic()
            self.hub.send_state(STATE_LIVE)
            effective = dict(event.effective or {})
            if effective.get("input_rate") or effective.get("output_rate"):
                self.input_rate = int(effective.get("input_rate") or self.input_rate)
                self.output_rate = int(effective.get("output_rate") or self.output_rate)
                self.hub.set_engine_rates(self.input_rate, self.output_rate)

    async def _barge_in(self) -> None:
        self.hub.flush()
        # A response can have several audio items queued ahead of playback.
        # Flush/truncate every item with audio still unplayed, an earlier commentary
        # included. A finished item (its final caption arrived) that the peer has played
        # to its end was heard whole: it is left alone, so the owner's next turn never
        # rewrites it as interrupted. An item still generating is cut even when playback
        # has caught up with the frames delivered so far.
        for item in tuple(self.spoken.keys() - self.cancelled_items):
            spoken, played = self.spoken[item], self.played_ms.get(item, 0)
            if not self.playback_acked and spoken.started_at:
                # No peer reports playback (a meeting-only call): its audio plays in real time.
                played = max(played, int(min(spoken.audio_ms, (time.monotonic() - spoken.started_at) * 1000)))
            if spoken.final_at and played >= spoken.audio_ms:
                continue
            self.cancelled_items.add(item)
            spoken.interrupted = True
            # A playback mark can run a few ms past the audio delivered so far; the
            # provider refuses a cut beyond the item's audio, so the cut stays inside it.
            await self._send(Interrupt(item_id=item, played_ms=min(played, int(spoken.audio_ms))))
            if spoken.final_at:
                self._write_spoken_row(item)

    def _on_final_caption(self, event: Caption) -> None:
        text = str(event.text or "").strip()
        if not text:
            if event.speaker == SPEAKER_OWNER:
                self._input_source(event.item_id)  # an input with no words still settles its place
            return
        # Who spoke is read from the audio that formed this input item; an item the provider
        # never placed falls back to what fed the engine since its last input caption: a
        # meeting microphone attached, or meeting audio already sent (its peer may have left).
        source = self._input_source(event.item_id) if event.speaker == SPEAKER_OWNER else None
        heard = self.spec.presence_binding_id or (
            source if source is not None else (self.meeting_mics or self.meeting_audio_since_caption))
        if event.speaker == SPEAKER_OWNER:
            self.meeting_audio_since_caption = False
        if event.speaker == SPEAKER_OWNER and heard:
            self._on_heard(text, event.item_id)
        elif event.speaker == SPEAKER_OWNER:
            self.counts["owner"] += 1
            self.messages.append({"role": "user", "content": text, "_call_row": True})
            self._write_owner_row(text, event.item_id)
            self.fanout.note(SPEAKER_OWNER, text)
        elif event.speaker == SPEAKER_OUROBOROS:
            self.counts["ouroboros"] += 1
            self.messages.append({"role": "assistant", "content": text, "_call_row": True})
            spoken = self.spoken.setdefault(event.item_id, _Spoken())
            spoken.text, spoken.final_at = text, time.monotonic()
            spoken.played_ms = max(spoken.played_ms, self.played_ms.get(event.item_id, 0))
            if spoken.interrupted or (spoken.played and spoken.played_ms >= spoken.audio_ms):
                self._write_spoken_row(event.item_id)
            self.fanout.note(SPEAKER_OUROBOROS, text)

    def _note_meeting_audio(self, start: float, end: float) -> None:
        if self.meeting_spans and self.meeting_spans[-1][0] == self.epoch and self.meeting_spans[-1][2] >= start:
            self.meeting_spans[-1] = (self.epoch, self.meeting_spans[-1][1], end)
        else:
            self.meeting_spans.append((self.epoch, start, end))

    def _input_source(self, item_id: str) -> Optional[bool]:
        """Whether meeting audio fed this input item: from its start to the next item's start
        (or to now while it is the latest); ``None`` when the provider never placed it."""
        placed = self.input_starts.get(item_id)
        if placed is None:
            return None
        epoch, start = placed
        later = [begin for session, begin in self.input_starts.values() if session == epoch and begin > start]
        end = min(later) if later else (self.input_ms if epoch == self.epoch else float("inf"))
        heard = any(session == epoch and begin < end and finish > start
                    for session, begin, finish in self.meeting_spans)
        self.inputs_done.add(item_id)
        # A captioned input's start is retired once no earlier input of its session is pending.
        for key, (session, begin) in list(self.input_starts.items()):
            if key in self.inputs_done and not any(
                    other not in self.inputs_done and other_session == session and other_begin < begin
                    for other, (other_session, other_begin) in self.input_starts.items()):
                del self.input_starts[key]
                self.inputs_done.discard(key)
        return heard

    def _on_heard(self, text: str, item_id: str) -> None:
        """Input speech while meeting audio feeds the session (or in a Presence call).

        Who spoke is a host fact, and the host only knows the mix: the words are
        recorded as heard in the call, never as the owner's message.
        """
        self.counts["heard"] += 1
        self.messages.append({"role": "user", "content": f"[Heard in the call] {text}", "_call_row": True})
        self.fanout.note("heard in the call", text)
        self.tools_lane.reset_rounds()  # a correspondent spoke: a Presence call's round count starts again
        if self.spec.presence_binding_id:
            self._write_presence_heard(text, item_id)
            return
        try:
            from supervisor.message_bus import send_with_budget

            send_with_budget(self.spec.chat_id, f"Heard in the call: {text}", role="system",
                             system_type="call_heard", narration=False,
                             progress_meta={"source": SOURCE_CALL, "call_id": self.spec.call_id,
                                            "item_id": item_id, "call_task_id": self.task_id})
        except Exception:
            log.warning("live call %s could not write a heard row", self.spec.call_id, exc_info=True)

    def _write_presence_heard(self, text: str, item_id: str) -> None:
        """A Presence call's correspondents' words become that conversation's dialogue rows."""
        voices, self.voices_since_caption = self.voices_since_caption, set()
        self._record_presence_utterance("in", text, speaker_label=next(iter(voices)) if len(voices) == 1 else "")

    def _record_presence_utterance(self, direction: str, text: str, *, speaker_label: str = "") -> None:
        try:
            from ouroboros.presence_call import record_call_utterance

            if not record_call_utterance(self.drive_root, self.task, direction=direction, text=text,
                                         speaker_label=speaker_label):
                self.counts["unrecorded"] = self.counts.get("unrecorded", 0) + 1
        except Exception:
            log.warning("live call %s could not record a Presence utterance", self.spec.call_id, exc_info=True)
            self.counts["unrecorded"] = self.counts.get("unrecorded", 0) + 1

    def _write_owner_row(self, text: str, item_id: str) -> None:
        """The owner's words as an owner row of the room, without an owner turn behind it."""
        try:
            from supervisor.message_bus import get_bridge, log_chat

            client_message_id = f"call:{self.spec.call_id}:{item_id or self.counts['owner']}"
            row = log_chat("in", self.spec.chat_id, 1, text, source=SOURCE_CALL, sender_label="voice",
                           client_message_id=client_message_id, task_id=self.task_id, drive_root=self.drive_root)
            if row:
                get_bridge().broadcast({"type": "chat", "role": "user", "content": text, "ts": row.get("ts"),
                                        "source": SOURCE_CALL, "chat_id": self.spec.chat_id, "task_id": self.task_id,
                                        "client_message_id": client_message_id})
        except Exception:
            log.warning("live call %s could not write an owner row", self.spec.call_id, exc_info=True)

    def _write_spoken_row(self, item_id: str) -> None:
        spoken = self.spoken.pop(item_id, None)
        if spoken is None or not spoken.text:
            return
        status = "interrupted" if spoken.interrupted else ("played" if spoken.played else "generated")
        if self.spec.presence_binding_id:
            self._record_presence_utterance("out", spoken.text)
        meta = _narration_meta(self.emit_progress)
        meta["spoken"] = {"call_id": self.spec.call_id, "item_id": item_id, "status": status, "played_ms": spoken.played_ms}
        try:
            self.emit_progress(spoken.text, narration=True, _progress_meta_override=meta)
        except TypeError:
            self.emit_progress(spoken.text, narration=True)

    # ---- the ledger ----------------------------------------------------------------

    def _attempt_for(self, key: str, *, usage: Optional[Dict[str, Any]] = None,
                     raw_usage: Optional[Dict[str, Any]] = None, status: str = "") -> Future:
        if key not in self.attempts:
            mode = str(getattr(self.ctx, "active_context_mode", "") or "nano")
            self.attempts[key] = self._record(self._reserve, key, mode, dict(usage or {}),
                                                     dict(raw_usage or {}), status)
            self.attempts[key].add_done_callback(self._on_reserved)
        return self.attempts[key]

    def _reserve(self, key: str, mode: str, usage: Dict[str, Any], raw_usage: Dict[str, Any], status: str) -> Any:
        try:
            return _ledger().reserve_live_response(
                engine=self.spec.engine, model=self.spec.model, call_id=self.spec.call_id, response_key=key,
                task_id=self.task_id, rendered_mode=mode, drive_root=self.drive_root)
        except BudgetExceeded as exc:
            # The provider already started (or finished) this response: the budget
            # refuses the call, not the provider's charge, which stays on the ledger.
            reason = str(exc)
            self._settle_safely(lambda: _ledger().record_refused_live_response(
                engine=self.spec.engine, model=self.spec.model, call_id=self.spec.call_id, response_key=key,
                task_id=self.task_id, rendered_mode=mode, drive_root=self.drive_root, reason=reason,
                usage=usage, raw_usage=raw_usage, status=status or "budget_refused"))
            raise

    def _on_reserved(self, attempt: Future) -> None:
        exc = attempt.exception()
        if isinstance(exc, BudgetExceeded):
            try:
                self._loop.call_soon_threadsafe(self._refuse_budget, exc)
            except (AttributeError, RuntimeError):  # the loop already ended: run() raises it
                self.budget_error = self.budget_error or exc
        elif exc is not None:
            log.warning("live call %s could not reserve a response", self.spec.call_id, exc_info=exc)

    def _refuse_budget(self, exc: BudgetExceeded) -> None:
        self.budget_error = self.budget_error or exc
        self._begin_close("budget_exceeded")

    def _record(self, action: Callable[..., Any], *args: Any) -> Future:
        """Queue one ledger write in order, inside the turn's context (budget scope, settings)."""
        return self.records.submit(contextvars.copy_context().run, action, *args)

    def _after_reservation(self, attempt: Future, action: Callable[[Any], Any]) -> None:
        """Queue ``action(reservation)`` behind the reservation; a refused one has nothing to settle."""
        def run() -> None:
            try:
                reserved = attempt.result()
            except Exception:
                return
            self._settle_safely(lambda: action(reserved))
        self._record(run)

    def _abandon(self, attempt: Future, reason: str) -> None:
        self._after_reservation(attempt, lambda reserved: _ledger().abandon_live_response(reserved, reason=reason))

    def _settle(self, event: ResponseDone) -> None:
        key = event.response_id or event.item_id
        attempt = self._attempt_for(key, usage=dict(event.usage), raw_usage=dict(event.raw_usage), status=event.status)
        self.attempts.pop(key, None)
        if self.witness and self.witness[0] == self.epoch:  # counted on the ledger thread, off the audio loop
            opened, tokens = self.witness[1], int(event.usage.get("text_input_tokens") or 0)
            self._record(self._settle_safely, lambda: record_density_witness(
                self.route, text_estimate(opened.instructions, opened.history, opened.tools), tokens))
        self.witness = None
        self.usage["live_responses"] += 1
        for name in USAGE_KEYS:
            self.usage["live_usage"][name] += int(event.usage.get(name) or 0)
        usage, raw_usage, status = dict(event.usage), dict(event.raw_usage), event.status
        self._after_reservation(attempt, lambda reserved: _ledger().settle_live_response(
            reserved, usage=usage, raw_usage=raw_usage, status=status))

    def _settle_safely(self, action: Callable[[], Any]) -> None:
        try:
            action()
        except Exception:
            log.warning("live call %s ledger write failed", self.spec.call_id, exc_info=True)

    # ---- the owner's mailbox and the room -------------------------------------------

    async def _drain(self) -> None:
        from ouroboros.loop_round_limits import _drain_incoming_messages

        scratch: List[Dict[str, Any]] = []
        controls = _drain_incoming_messages(scratch, self.incoming, self.drive_root, self.task_id, self.event_queue,
                                            self.owner_msg_seen, owner_ctx=self.ctx, defer_content_ack=True)
        if controls.get("finalize_now"):
            self._begin_close("owner_stop")  # a Stop never waits behind the context sends below
        for msg in scratch:
            self.messages.append({**msg, "_call_row": True})
            item = ContextItem(text=text_of(msg.get("content")), source="mailbox")
            if item.text.strip() and not self.closing:
                self._post(item)
        if not self.closing:
            for item in self.rooms.drain():
                await self._context(item)
            if controls.get("hurry"):
                await self._context(ContextItem(text="The owner pressed Hurry: finish what you are saying and wrap up.",
                                                source="owner_control"))
        # A boundary after everything posted so far owes exactly the acknowledgements taken
        # now; later mail waits for its own boundary.
        pending = list(getattr(self.ctx, "_pending_content_acks", None) or [])
        if (scratch or controls or pending) and self.engine is not None and not self.closing:
            self.ctx._pending_content_acks = []
            self._post(_Acks(tuple(pending)))

    def _checkpoint(self) -> bool:
        """The call's durable state on its task row; content ACKs wait for this write."""
        from ouroboros.task_result_schema import require_writable_task_result_schema, stamp_task_result_schema
        from ouroboros.task_store import patch_document
        from ouroboros.utils import utc_now_iso

        if self.drive_root is None or not self.task_id:
            return False
        block = {"call_id": self.spec.call_id, "consumed_mailbox_ids": sorted(str(x) for x in self.owner_msg_seen),
                 "executed_tool_calls": sorted(self.tools_lane.executed), "batches": self.tools_lane.batches,
                 "utterances": dict(self.counts), "reconnects": self.reconnects, "updated_at": utc_now_iso()}

        def _mutate(current: Dict[str, Any]) -> Dict[str, Any]:
            require_writable_task_result_schema(current)
            return stamp_task_result_schema({**current, "live_call": block})

        try:
            patch_document(pathlib.Path(self.drive_root), self.task_id, _mutate)
            return True
        except Exception:
            log.warning("live call %s checkpoint unwritten; mailbox content stays unacknowledged", self.spec.call_id, exc_info=True)
            return False


def run_live_turn(
    messages: List[Dict[str, Any]], tools: Any, llm: Any, drive_logs: pathlib.Path,
    emit_progress: Callable[..., None], incoming_messages: queue.Queue, task_id: str, task_type: str,
    budget_remaining_usd: Optional[float], event_queue: Optional[queue.Queue], drive_root: Optional[pathlib.Path],
    spec: CallSpec, task: Optional[Dict[str, Any]] = None, context_captured_at: str = "",
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Run one live call to its end; the loop's ordinary triple with a host-authored terminal.

    ``budget_remaining_usd`` is read by the ledger at every response reservation
    (``BudgetExceeded`` propagates after the call is closed); it is accepted here
    so the driver keeps the loop's calling shape. ``context_captured_at`` is the
    instant the agent captured the turn's context; the room feed catches up from it.
    """
    call = _LiveCall(messages=messages, tools=tools, llm=llm, drive_logs=drive_logs, emit_progress=emit_progress,
                     incoming_messages=incoming_messages, task_id=task_id, task_type=task_type,
                     event_queue=event_queue, drive_root=drive_root, spec=spec, task=task,
                     context_captured_at=context_captured_at)
    return call.run()

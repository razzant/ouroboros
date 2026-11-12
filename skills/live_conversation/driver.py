"""The recovered call event loop over the admitted host port.

Audio remains realtime speech-to-speech. Tools run on the ordinary serialized
host lane beside media; a running tool holds no fit or reconnection, since its
request is a host fact in every fit, and only the engine's own open call holds a
live switch. A provider-requested renewal asks the host to fit the whole
transcript again; transport loss continues through that same fitted history.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from websockets.exceptions import ConnectionClosed

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import Audio, CallTool, ContextUpdate, Interrupt, LiveHost, LiveOptions, Respond, ToolOutputs
from .attention import Attention, attend_tool
from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros.realtime_session import joined
from ouroboros.ui_language import normalize_language_tag
from ouroboros.utils import read_json_dict, update_json_locked, utc_now_iso


# A transport bound, not a judgment: a session that keeps dropping before an answer completes or is interrupted is
# resumed this many times, then the call ends with a typed reason instead of reconnecting forever. A close the
# engine announced with `renew` is an expiry, not a drop.
UNPROVEN_RESUMES = 3

CALL_ENDING_GUIDANCE = (
    "Use end_call only when the owner wants the whole call to end: says goodbye or asks to hang up. "
    "An interruption such as 'stop', 'enough' or 'wait' ends the current answer, not the call; "
    "acknowledge briefly and listen for the next request."
)

INITIAL_TOOLS = ("send_user_message", "schedule_subagent", "get_task_result")

# How long a call waits for its peers before ending: nobody joining is `not_connected`, everyone
# leaving is `peers_left` (a transport bound, not a judgment).
EMPTY_CALL_GRACE_S = 15

CALL_INSTRUCTIONS = """## Live calls

In a call my model is a realtime speech model, and I am still myself: the
same core, memory, tools and judgment. I speak in the language of the spoken
conversation whatever the interface language, and call tools myself. I may
speak first when it helps. Silence is a finished choice: `attend` with no
change ends my turn without words. What I see rides my turns; `attend` chooses
how I watch and lets me look now. Work still running is pending, never done; long work becomes a task
through `schedule_subagent` while we keep talking, and its result reaches me in the
call; `get_task_result` reads its retained result. Tools not loaded yet are still mine: `list_available_tools`
names them (a desktop skill, for one, takes screenshots and presses keys) and
`enable_tools` loads them, so I never say I cannot do something before I have
looked. A tool result in another language never changes the language I speak.
My memory, this room and the other rooms are reachable here as in any turn,
through their own tools. A remote shared frame is its own source, not a
screenshot of this machine. Links,
code and lists go to the room's chat through `send_user_message`: saying a
message or a URL aloud does not deliver it, so when asked to send something I use that
tool and confirm its result. Typed messages and other turns' replies in the
room reach me during the call, and my words reach them. In a meeting I speak
when it helps the people there; their voices are correspondents, not my
human. Words cut off by an interruption were not heard, so I do not count
them as said.""" + "\n\n" + CALL_ENDING_GUIDANCE


def engine_sentence(model):
    """The engine and its senses are facts of this call: the owner asks "which model are you on?",
    and a blind engine must not be asked to look at a screenshot (27.10: a Grok call named an
    invented vendor and a screenshot's image closed the call)."""
    senses = ("it hears speech and sees images: the owner's shared screen or camera (Show) and images that tools "
              "attach; with no frame from Show yet, you have not seen the screen"
              if model.vision else
              "it hears speech and does not see images: a remote shared frame or an attached image reaches you as a caption "
              "naming its exact saved file, and vlm_query with that file_path describes it in text. A desktop screenshot "
              "is not the remote shared source")
    caution = (" A silent tool result is not guaranteed silent on this route."
               if not model.silent_tool_results else "")
    return f"This call runs on {model.label or model.id}; {senses}." + caution


def languages_opening_sentence(languages):
    if languages:
        return (f"Speech recognition expects {', '.join(languages)}, as you recorded for these calls; the first is "
                f"the owner's language, so greet in it and stay in the spoken language. When someone "
                "speaks a language that is not listed, record the new list with set_call_languages. Recognition "
                "can still take one language for another: when unsure, ask rather than switch.")
    return ("Speech recognition has no languages recorded for these calls yet and guesses each utterance: assume "
            "the owner's language from this room's correspondence and your memory and record it right away with "
            "set_call_languages, the owner's first; ask when unsure.")


def language_tags(raw):
    if not isinstance(raw, (list, tuple)):
        raise ValueError("languages must be a list of language tags such as ru-RU")
    kept = []
    for item in raw:
        tag = normalize_language_tag(item) if isinstance(item, str) else None
        if not tag:
            raise ValueError(f"not a language tag: {item!r} (use tags such as ru-RU or en-US)")
        if tag not in kept:
            kept.append(tag)
    return tuple(kept)


def recorded_call_languages(state_dir, binding_id=""):
    if state_dir is None:
        return ()
    data = read_json_dict(Path(state_dir) / "call_languages.json") or {}
    bindings = data.get("bindings") if isinstance(data.get("bindings"), dict) else {}
    entry = bindings.get(binding_id) if binding_id else data.get("owner")
    try:
        return language_tags(entry.get("languages")) if isinstance(entry, dict) else ()
    except ValueError:
        return ()


def record_call_languages(state_dir, languages, *, binding_id=""):
    kept = language_tags(languages)
    entry = {"languages": list(kept), "recorded_at": utc_now_iso()}
    def mutate(current):
        if binding_id:
            bindings = current.get("bindings") if isinstance(current.get("bindings"), dict) else {}
            current["bindings"] = {**bindings, binding_id: entry}
        else:
            current["owner"] = entry
        return current
    update_json_locked(Path(state_dir) / "call_languages.json", mutate)
    return kept


def call_tools(state_dir):
    async def end_call(host, arguments):
        if arguments:
            raise ValueError("end_call takes no arguments")
        host.hub.end_by_ouroboros()
        return "The call ends once your goodbye has been heard."

    async def set_languages(host, arguments):
        if not isinstance(arguments, dict) or set(arguments) != {"languages"}:
            raise ValueError("set_call_languages takes a languages list")
        languages = record_call_languages(state_dir, arguments["languages"], binding_id=host.hub.spec.presence_binding_id)
        if languages != host.state.get("languages", ()):
            host.state.update(languages=languages, refit_after_answer=True)
        if not languages:
            return "Recorded: no languages. Speech recognition guesses each utterance again from the next pause."
        return (f"Recorded: speech recognition expects {', '.join(languages)} from the next pause, "
                "and later calls start with it.")

    return (
        CallTool("end_call", CALL_ENDING_GUIDANCE + " Say your goodbye first; the call ends once it has been heard.",
                 {"type": "object", "properties": {}, "additionalProperties": False}, end_call, silent=True),
        CallTool("set_call_languages", "Record the languages speech recognition should expect in calls with these people, the most "
                 "spoken first, as BCP-47 tags such as ru-RU or en-US; an empty list lets recognition guess "
                 "each utterance again. Judge them from the correspondence and your memory, and add a language "
                 "you hear someone use. Recognition takes the list at the next pause and later calls start "
                 "with it. It never decides the language you answer in.",
                 {"type": "object", "properties": {"languages": {"type": "array", "items": {"type": "string"}}},
                  "required": ["languages"], "additionalProperties": False}, set_languages),
        attend_tool(),
    )


def session_options(host, resume_handle=""):
    config, honours = host.config, host.model.honours
    languages = tuple(host.state.get("languages", ()))
    knobs = {key: getattr(config, key) for key in ("pause", "interruptions", "thinking", "search") if key in honours}
    if "thinking" in knobs and knobs["thinking"] not in host.model.thinking_levels:
        knobs["thinking"] = host.model.thinking_default or host.model.thinking_levels[0]
    return LiveOptions(voice=config.voice, resume_handle=resume_handle,
        languages=languages if "languages" in honours else (),
        instructions="\n\n".join(filter(None, (CALL_INSTRUCTIONS, engine_sentence(host.model),
                                                languages_opening_sentence(languages), config.instructions))),
        **knobs)


async def run(host: LiveHost, *, state_dir=None):
    host.state["languages"] = recorded_call_languages(state_dir, host.hub.spec.presence_binding_id)
    await _Call(host).run()


NO_ROOM = "This room's context and working reserve do not fit this engine's window."
ROOM_POINTER = ("[Host fact; not owner words] New room context is in the fitted history. "
                "You may answer, act or stay silent.")


class _Owed:
    """An answer the call still owes (see ``_Call.answer_due``). ``mark`` is where the asked connection's delivery
    record stood; only a row held after it that asks for this answer delivers it, and ``at`` is the first such row."""

    def __init__(self, session, *, ids=(), rows=(), pointers=(), room=False):
        self.ids, self.rows, self.room, self.pointers = tuple(sorted(ids)), tuple(rows), room, list(pointers)
        self.mark = len(getattr(session, 'held_context_rows', None) or ())
        self.at, self.delivered, self.answering = None, False, set()

    def asks(self, row):
        if self.room:  # its own words, or the pointer to them in a fit
            return row.get('content') == ROOM_POINTER or any(row == ask for ask in self.rows)
        return row.get('tool_call_id') in self.ids or not set(self.ids).isdisjoint(row.get('_tool_answer') or ())


class _Call:
    """Media evidence and work belong to the call, including between connections."""

    def __init__(self, host):
        self.host, self.hub = host, host.hub
        self.tool_queue, self.rows = asyncio.Queue(maxsize=32), asyncio.Queue(maxsize=256)
        self.connected = asyncio.Event()
        self.tool_refit = asyncio.Event()
        self.pending_offers = []
        self.attention = host.state["attention"] = Attention(host)
        self.participants, self.meeting_speakers = {}, set()
        self.session = self.last_session = None
        self.played, self.audio_ms, self.pending_spoken = {}, {}, {}
        self.interrupted, self.started_responses, self.active_responses = set(), set(), set()
        self.receipts = {}
        self.tool_responses = set()
        self.response_open = self.tools_running = self.input_open = False
        self.meeting_audio = False
        self.playing_item = ""
        self.empty_since = time.monotonic()
        self.ever_attached = False
        self.peer_info = {}
        self.share_kinds = self.attention.kinds
        self.renewing = self.resume = self.provider_closed = False
        self.connection_resumed = False
        self.renewal_deadline = None
        self.input_rate = self.output_rate = 0
        self.unproven_resumes = 0  # provider drops since the last completed/interrupted answer
        self.offer_wait_event = None
        self.session_ended = asyncio.Event()  # set when the current connection is gone
        self.preparing = None
        self.preparing_options = None
        self.failed_preparation_revision = None
        # What waits for a connection, in order: one connection-owned worker sends it (delivery).
        self.deferred_tool_outputs = []
        self.pending_context_rows = []
        self.delivery_due = asyncio.Event()
        self.delivering = False
        self.delivered_catalog = None
        self.preparing_revision = None
        self.resume_revision = 0
        self.resume_tool_output = None
        # A fresh session starts a new provider conversation; a native resume continues it. Calls not yet
        # answered natively: id -> (conversation that issued it, resume_revision at its arrival).
        self.conversation, self.issued = 0, {}
        # Paused input is retained and replayed; words spoken into a lost connection may not be.
        self.expiry_announced = self.lost_while_speaking = False
        # An ordinary result handed to the session owes its answer (answer_due), and so does a typed room request
        # (room_due). Only a response begun after the adapter delivered it settles that, once it finishes without
        # continuing and, where the adapter records each response's end, holding it there. Speech, transcript tails
        # and receipts carry no answer identity. A response begun before delivery that speaks on is talk-over for
        # OpenAI and Grok (which write results only between responses) and for ordinary Gemini (a WHEN_IDLE result
        # waits for idle); extended thinking, unscheduled, may instead continue the call's own turn. Nothing native
        # tells which, so that answer stays owed: a fresh session's neutral pointer leaves the choice to the mind and
        # its history. That pointer carries the call ids (``_tool_answer``), or is the room pointer, and re-arms the
        # same obligation on its session, so a fresh session lost before answering owes the next one again.
        self.answer_due = self.room_due = None
        self.looked_calls = set()  # a look's fresh pointer asks for pixels its fit may lack, however it comes to be owed
        # The owner's newest words, from their start: the responses begun since (None: nothing owed) and whether what is
        # owed was recognised. One of those responses completing answers them; one cut short does not, so barge-in and a
        # request that crossed new speech stay owed, and a transcript landing after the answer finds nothing to owe.
        # A fresh fit holds recognised words, so a fresh switch owes them as room words; other words stay the gap fact's.
        self.spoken_answers, self.spoken_known = None, False

    async def run(self):
        writer = self.writer = asyncio.create_task(self.save_rows())
        workers = [asyncio.create_task(self.connections()), asyncio.create_task(self.tools())]
        try:
            done, _ = await asyncio.wait([*workers, writer], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
            self.consume_revisit()
            await self.close_preparation()
            self.timing("offer_idle")  # teardown is no longer waiting for a turn
            # A resume never cancels tools. Actual call teardown owns and joins
            # them; the host also joins any already-dispatched side effects.
            for task in workers:
                task.cancel()
            await joined(asyncio.gather(*workers, return_exceptions=True))
            try:
                await joined(self.finalize())
            finally:
                writer.cancel()
                await joined(asyncio.gather(writer, return_exceptions=True))
                self.timing("offer_wait_done")
        if self.hub.ended_by_ouroboros and not self.hub.ended_by_owner and not self.hub.closed:
            self.hub.send_state(cp.STATE_CLOSED, "ouroboros_bye")

    async def connections(self):
        handle = ""
        while True:
            await self.host.check()
            if self.hub.closed or self.hub.ended_by_owner or self.hub.ended_by_ouroboros:
                return
            if self.tool_refit.is_set():
                if not self.resume: handle = ""
                self.tool_refit.clear()
                # The retired native tool obligation cannot hold this new
                # interaction open. Unreceipted reservations remain host-owned.
                self.active_responses.clear()
                self.tool_responses.clear()
                self.response_open = self.input_open = False
            if not handle:
                self.host.state.pop("refit", None)
                self.host.state.pop("refit_after_answer", None)
                self.host.state.pop("resume_handle", None)
            if handle:
                await self.close_preparation()  # native resumption stays sequential
            prepared, self.preparing = self.preparing, None
            lease = None
            if prepared is not None:
                try:
                    lease = await prepared
                except RealtimeSetupError as exc:
                    if exc.code != 'prepared_stale':
                        raise
                except (OSError, ConnectionClosed) as exc:
                    from ouroboros.realtime_session import transport_loss
                    if not transport_loss(exc):
                        raise RealtimeSetupError('provider_closed', 'provider refused the prepared connection') from exc
            if lease is not None and self.preparing_options != session_options(self.host, handle):
                await lease.close()
                lease = None  # promised options changed: one sequential fit with current values
            opening = self.session_ended
            try:
                if lease is not None:
                    try:
                        result = await lease.run(self.consume)
                        if result or self.preparing is None or not self.preparation_ready():
                            await lease.close()  # native resumption/sequential replacement drains the old socket first
                    except RealtimeSetupError as exc:
                        await lease.close()
                        if exc.code != 'prepared_stale':
                            raise
                        lease = None
                if lease is None:
                    result = await self.host.session(self.consume, session_options(self.host, handle))
            except RealtimeSetupError as exc:
                if exc.code != 'prepared_stale' or not handle:
                    raise
                # History moved while the resume opened and its handle is spent: a fresh fit holds the unit.
                self.resume, handle = False, ''
                continue
            except (OSError, asyncio.TimeoutError) as exc:  # includes a stale prepared lease's replacement
                if not self.conversation or self.session_ended is not opening:
                    raise  # a first open, or a connection's own lifetime, keeps its ending
                raise RealtimeSetupError('reconnect_failed', 'the voice connection could not be reopened') from exc
            if result is None:
                return
            handle = result

    async def consume(self, session):
        self.session = self.last_session = session
        self.resume_tool_output = None
        self.expiry_announced = False
        if not self.resume:
            self.delivered_catalog = None
            self.conversation += 1  # calls an earlier conversation issued can no longer be answered natively
        self.connection_resumed = self.resume
        self.session_ended = asyncio.Event()
        if not self.resume:
            # Nothing here answers what the last conversation awaited. Unreceipted reservations remain host-owned.
            self.active_responses.clear()
            self.tool_responses.clear()
            self.response_open = False
            if self.answer_due is not None:  # its result reaches this session only in history, still owed
                ids = self.answer_due.ids
                self.offer_tool_answer(ids, looked=not self.looked_calls.isdisjoint(ids))
            if self.room_due is not None or self.spoken_known:  # so are the room's words and the owner's recognised ones
                self.owe_room_answer()
            self.spoken_answers, self.spoken_known = None, False
            for output, _ in self.deferred_tool_outputs:
                if output is not None:
                    self.retire_calls(output.messages)
            self.deferred_tool_outputs.clear()  # completed effects are in this adopted fresh prefix
            self.pending_context_rows.clear()
            if self.host.model.vision:
                self.attention.seen.clear()  # pixels are transport state; blind frame facts are already in the fit
                self.attention.include_images(getattr(self.host, "fitted_frame_rows", lambda: ())())
            self.refit_offers()
        # Words drained around a cancelled worker, owed like any request. A fresh fit holds them. A native resume continues
        # a conversation that never saw them, so they take the worker's lane, ahead of their pointer.
        cancelled_mail = getattr(self.host, 'take_mail_rows', list)()
        if getattr(self.host, 'take_mail_answer', lambda: False)():
            if self.resume and cancelled_mail:
                await self.context_update(cancelled_mail, request=True)
            else:
                self.owe_room_answer()
        self.renewing = self.resume = self.provider_closed = False
        self.renewal_deadline = None
        self.input_rate = self.output_rate = 0
        self.timed_audio = False
        resumed = replacing = False
        peers = asyncio.create_task(self.peers())
        provider = asyncio.create_task(self.provider())
        children = [provider, peers, asyncio.create_task(self.mailbox()), asyncio.create_task(self.delivery())]
        try:
            while True:
                done, _ = await asyncio.wait(children, return_when=asyncio.FIRST_COMPLETED)
                for child in done:
                    try:
                        await child
                    except ConnectionClosed:
                        if child is provider:
                            raise
                        # The native reader owns the receipts and resumption authority
                        # when a microphone/context write sees the closed socket first.
                        await provider
                        if not self.provider_closed:
                            raise
                await self.rows.join()
                # Joining canonical rows yields. Native input or an answer may
                # have started since the peer loop observed its boundary; retain
                # its reader and receipt instead of cancelling a newly busy turn.
                quiet = self.refit_ready() if self.tool_refit.is_set() else self.quiet()
                if (self.renewing and not self.provider_closed and not quiet
                        and not (self.hub.closed or self.hub.ended_by_owner or self.hub.ended_by_ouroboros)):
                    if peers in done:
                        children.remove(peers)
                        peers = asyncio.create_task(self.peers())
                        children.append(peers)
                    continue
                break
            # Queued and running calls need no wait: their requests are host facts in the fit.
            if self.tool_refit.is_set() and self.hub.ended_by_ouroboros:
                await peers  # a queued end_call still owns goodbye playback
                self.renewing = False
            elif (self.tool_refit.is_set() and not self.resume) or (self.renewing and self.resume and not (
                    self.hub.closed or self.hub.ended_by_owner or self.hub.ended_by_ouroboros) and not self.resume_allowed()):
                # A capacity/catalog obligation needs a fresh fit; so does a handle the reader withdrew during the join.
                self.resume = False
                self.request_renewal()
            replacing = self.renewing
            resumed = replacing and self.resume
            return (self.host.state.get("resume_handle", "") if resumed else "") if self.renewing else None
        finally:
            self.connected.clear()
            self.hub.send_state(cp.STATE_RECONNECTING if replacing else cp.STATE_CLOSING)
            for child in children:
                child.cancel()
            await joined(asyncio.gather(*children, return_exceptions=True))
            self.session = None
            self.session_ended.set()
            if not resumed:
                await joined(self.finalize())

    @staticmethod
    def owed_calls(output):
        return {row.get('tool_call_id') for row in output.messages
                if row.get('role') == 'tool' and row.get('tool_call_id') not in output.silent}

    async def send(self, output):
        """One native write of a tool output with its delivery record. A closed socket raises to the caller, which keeps
        the output in custody: the reader classifies the loss, a resume is owed the same output, and a fresh switch asks
        for its answer with one pointer. The effect ran once."""
        session, current = self.session, self.conversation
        owed = self.owed_calls(output)
        resume_revision = self.resume_revision
        self.arm_answer(owed, session)  # a newer result is answered only after its own delivery; an older debt joins it
        result = await session.send(output)
        self.sent_unrecorded(session, output.messages)
        self.retire_calls(output.messages)  # answered natively
        if self.conversation == current:  # a replacement records its own deliveries
            # Adapters may distinguish native response writes from pixels
            # and queue acceptance. Older sessions keep send-boundary order.
            if getattr(session, 'resume_after_tool_response', False) is not None:
                previous = self.resume_tool_output
                self.resume_tool_output = (resume_revision, bool(previous and previous[1]) or bool(owed))
            self.delivered_catalog = output.tools
            delivered = getattr(self.host, 'tool_results_delivered', None)
            if delivered is not None:
                delivered(getattr(session, "held_context_rows", ()))
        return result

    def retire_calls(self, rows):
        for row in rows:
            self.issued.pop(row.get('tool_call_id'), None)

    def refit_offers(self):
        """A fresh fit already holds every canonical fact; keep transport-only rows and one pointer to the facts."""
        facts = [row for row in self.pending_offers if row.get("_canonical")]
        if facts:
            pointer = {"role": "user", "content": "[Host fact; not owner words] The new facts are in the fitted "
                       "history. You may answer, act or stay silent."}
            if any(row.get("_requested_look") for row in facts):
                pointer["_requested_look"] = True
            self.pending_offers = [row for row in self.pending_offers if not row.get("_canonical")] + [pointer]

    def consume_revisit(self):
        self.attention.revisit_at = None
        self.pending_offers[:] = [row for row in self.pending_offers if not row.get("_requested_look")]

    @staticmethod
    def tool_answer_pointer():
        return {"role": "user", "content":
            "[Host fact; not owner words] The requested tool batch returned; its results are in the fitted history. "
            "You may continue from them, act or stay silent; completed effects stay done."}

    @staticmethod
    def room_answer_pointer():
        return {"role": "user", "content": ROOM_POINTER}

    async def context_update(self, rows, *, request=False):
        # Mailbox and peer rows are already canonical on the host. Check the
        # resulting fit before adding them to this connection; a refit includes
        # them in its selected input instead. A room request stays owed past its write.
        pointer = self.room_answer_pointer()
        self.pending_offers.append(pointer)  # owns the answer before a write can yield or be cancelled
        if request:
            self.owe_room_answer(rows, pointer)
        fit_due = self.host.renewal_due() or self.renewing or self.host.state.get("refit")
        if fit_due or self.pending_context_rows or not self.connected.is_set():
            self.pending_context_rows.extend(rows)  # behind older mail, for the worker or the coming fit
            if self.connected.is_set() and (not fit_due or self.request_renewal() is False):
                self.delivery_due.set()
            return
        # Typed words go at once and only this write holds them. A failed or cancelled one returns them, the pointer still
        # call-owned, before consume can start another connection; the native reader classifies a closed socket.
        session, self.response_open = self.session, True
        try:
            await session.send(ContextUpdate(tuple(rows)))
        except BaseException:
            self.response_open = bool(self.active_responses)
            self.pending_context_rows[:0] = rows
            raise
        self.pending_offers[:] = [row for row in self.pending_offers if row is not pointer]
        self.sent_unrecorded(session, rows)

    def timing(self, phase):
        record = getattr(self.host, "timing_event", None)
        if record is not None:
            record(phase)

    def request_renewal(self, *, resuming=False, within=None):
        if (not resuming and self.failed_preparation_revision is not None
                and self.failed_preparation_revision == getattr(self.host, '_history_version', 0)):
            return False  # only a changed canonical view earns another speculative fit
        if not self.renewing:
            self.timing("requested")
        # A fresh fit wins over a requested resume while the transport is live.
        self.resume = resuming if not self.renewing else self.resume and resuming
        self.renewing = True
        if not resuming:
            self.host.state["refit"] = True
            prepare = getattr(self.host, 'prepare_session', None)
            if (prepare is not None and self.preparing is None
                    and getattr(self.session, 'silent_context_updates', False)):
                self.preparing_revision = getattr(self.host, "_history_version", 0)
                self.preparing_options = session_options(self.host, '')
                self.preparing = asyncio.create_task(prepare(self.preparing_options))
        if within is not None:
            deadline = time.monotonic() + float(within)
            self.renewal_deadline = min(self.renewal_deadline, deadline) if self.renewal_deadline else deadline

    def preparation_ready(self):
        return self.preparing is None or (self.preparing.done() and not self.preparing.cancelled()
                                         and self.preparing.exception() is None)

    async def check_preparation(self):
        task = self.preparing
        if task is None or not task.done() or not task.cancelled() and task.exception() is None:
            return
        self.preparing = None
        try:
            await task
        except RealtimeSetupError as exc:
            if exc.code == 'prepared_stale':
                return  # the existing sequential fallback can now take the quiet boundary
            if exc.code != 'context_capacity':
                raise  # credentials, quota and policy keep their terminal contract
            revision = getattr(exc, 'source_revision', None)
        except (OSError, ConnectionClosed) as exc:
            from ouroboros.realtime_session import transport_loss
            if transport_loss(exc):
                return
            raise
        self.failed_preparation_revision = revision if revision is not None else self.preparing_revision
        self.renewing = self.resume = False
        self.renewal_deadline = None
        current = getattr(self.host, "_history_version", None)
        if revision is not None and current is not None and revision != current:
            self.request_renewal()  # the changed view gets its fit before the old socket can refuse
            return
        self.host.state.pop('refit', None)
        self.tool_refit.clear()
        self.delivery_due.set()

    def serving_catalog(self):
        if self.delivered_catalog is None:
            # External sessions need only the public selected catalog, never .tools.
            self.delivered_catalog = tuple(json.loads(self.session.selected.tools_json))
        return self.delivered_catalog

    async def guard_serving(self, rows):
        check = getattr(self.host, 'delivery_fits', None)
        if check is None or not await check(rows, self.serving_catalog()):
            raise RealtimeSetupError('context_capacity', NO_ROOM)

    async def delivery(self):
        """This connection's one sender of what waited for it. It dies with the connection, so nothing it was measuring
        can land on a successor: what it has not written stays in custody for a native resume, or is held by a fresh fit.
        The reader and audio never wait on its measures. A closed socket raises to consume, whose reader classifies it."""
        while True:
            await self.delivery_due.wait()
            await self.connected.wait()
            self.delivery_due.clear()
            if not self.hub.ended_by_ouroboros:  # a goodbye needs only playback
                await self.release_serving()

    def fit_on_its_way(self):
        return self.renewing and not self.resume

    async def custody_write(self, write):
        # Quiet waits for this: a planned switch must not cancel a write and then have to guess what it did.
        self.delivering = True
        try:
            await write
        finally:
            self.delivering = False

    async def release_serving(self):
        # Only delivery() runs this. Obligations leave custody only after a successful fitting delivery. A refit obligation
        # withholds them from this retiring connection, and a refusal counts only if no fresh fit is coming to hold them.
        while self.deferred_tool_outputs and not self.tool_refit.is_set():
            output, offers = self.deferred_tool_outputs[0]
            if output is None:  # no native lane: only a fit can hold this unit, and none is coming once one was refused
                if self.fit_on_its_way() or self.host.state.get("refit"):
                    return
                raise RealtimeSetupError('context_capacity', NO_ROOM)
            rows = output.messages
            if offers is not None:  # withheld for a fit; an admitted output needs no measure and left no offers
                try:
                    await self.guard_serving(rows)
                except RealtimeSetupError:
                    if self.fit_on_its_way():
                        return
                    fallback = getattr(self.host, 'tool_result_fallback', None)
                    shortened = await fallback(rows, self.serving_catalog()) if fallback else None
                    if shortened is None:
                        raise
                    rows = tuple(shortened)
            await self.connected.wait()
            if self.owed_calls(output):
                self.response_open = True
            tools = output.tools if offers is None else self.serving_catalog()
            await self.custody_write(self.send(ToolOutputs(rows, tools, silent=output.silent)))
            self.deferred_tool_outputs.pop(0)
            self.pending_offers[:] = [row for row in self.pending_offers if all(row is not offer for offer in offers or ())]
        rows = tuple(self.pending_context_rows)
        if rows and not self.tool_refit.is_set():
            try:
                await self.guard_serving(rows)
            except RealtimeSetupError:
                if self.fit_on_its_way():
                    return
                raise
            await self.connected.wait()
            await self.custody_write(self.session.send(ContextUpdate(rows, respond=False)))
            self.sent_unrecorded(self.session, rows)
            self.pending_context_rows[:] = [row for row in self.pending_context_rows if all(row is not old for old in rows)]

    async def close_preparation(self):
        task, self.preparing = self.preparing, None
        if task is None:
            return
        if not task.done():
            task.cancel()
        result = await joined(asyncio.gather(task, return_exceptions=True))
        if not isinstance(result[0], BaseException) and result[0] is not None:
            await result[0].close()

    def refit_ready(self):
        return (self.preparation_ready() and not self.input_unsettled()
                and not (self.active_responses - self.tool_responses)
                and not self.pending_spoken and not self.unheard() and not self.delivering)

    def input_unsettled(self):
        """Words being said or awaiting recognition, by the adapter's own fact where it has one: a VAD edge or an older
        utterance's transcript moves ``input_open``. Read apart from ``renewal_ready``, which an open call also holds."""
        return self.input_open or bool(getattr(self.session, 'input_pending', False))

    def audio_duration(self, item):
        return int(self.audio_ms.get(item, 0))

    def heard(self, item):
        # Playback reports floor at the device rate: a fully played 9,800 ms item reads 9,799 after the
        # 24 -> 48 kHz resample (27.10), so the caption rule's 20 ms tolerance decides here as well.
        return self.played.get(item, 0) > 0 and self.played[item] >= self.audio_duration(item) - 20

    def unheard(self):
        return any(item not in self.interrupted and not self.heard(item) for item in self.audio_ms)

    def quiet(self):
        return not self.quiet_blockers()

    def quiet_blockers(self):
        blocked = [name for name, active in (
            ("response_open", self.response_open), ("input_open", self.input_open), ("delivering", self.delivering),
            ("pending_spoken", bool(self.pending_spoken)), ("unheard", self.unheard())) if active]
        if self.session is None:
            blocked.append("connection")
        elif not self.session.renewal_ready:
            blocked.extend(getattr(self.session, "renewal_blockers", ("session.renewal_ready",)))
        return tuple(blocked)

    def track_offer_wait(self):
        pending = self.pending_offers or self.attention.revisit_due()
        event = "offer_wait:" + ",".join(self.quiet_blockers()) if pending else "offer_idle"
        if event != self.offer_wait_event:
            self.timing(event)
            self.offer_wait_event = event

    async def finish_spoken(self, item, *, interrupted=False, completed=False):
        caption = self.pending_spoken.pop(item, None)
        if caption is not None:
            await self.rows.put(dict(speaker="ouroboros", item_id=item, heard=False,
                played_ms=self.played.get(item, 0), interrupted=interrupted, completed=completed, **caption))

    async def caption(self, data):
        speaker, item = data["speaker"], data["item_id"]
        text = ("[Partial transcript] " if data.get("incomplete") else "") + data["text"]
        if speaker == "owner":
            # A transcript closes its own words, never newer ones its adapter still reports as being said.
            self.input_open = not data["final"] or (self.input_open
                                                    and bool(getattr(self.session, 'input_pending', False)))
            self.spoken_known |= bool(data["final"] and data["text"] and self.spoken_answers is not None)
        self.hub.send_caption(speaker, text, data["final"], item, replace=bool(data.get("replace")))
        if data["final"]:
            heard = speaker == "owner" and (self.meeting_audio or any(
                peer.role == cp.PEER_MEETING and peer.mic for peer in self.hub.peers()))
            if speaker == "ouroboros":
                self.pending_spoken[item] = dict(text=text, **({"replace": True} if data.get("replace") else {}))
                if item in self.interrupted or self.heard(item):
                    await self.finish_spoken(item, interrupted=item in self.interrupted, completed=item not in self.interrupted)
            else:
                await self.rows.put(dict(speaker=speaker, text=text, item_id=item, heard=heard,
                    **({"speaker_label": ", ".join(sorted(self.meeting_speakers))} if heard and self.meeting_speakers else {})))
            if speaker == "owner":
                self.meeting_audio = False
                self.meeting_speakers.clear()

    async def save_rows(self):
        while True:
            row = await self.rows.get()
            try:
                if "tool_request" in row:  # ordered with the captions that led to it
                    requested = getattr(self.host, "tool_requested", None)
                    if requested is not None:
                        await requested(row["tool_request"])
                else:
                    await self.host.caption(**row)
            finally:
                self.rows.task_done()

    async def finalize(self):
        session, self.last_session = self.last_session, None
        if session is not None:
            for event in session.final_captions():
                await self.caption(event.data)
        for item in tuple(self.pending_spoken):
            await self.finish_spoken(item, interrupted=item in self.interrupted)
        drain = asyncio.create_task(self.rows.join())
        try:
            done, _ = await asyncio.wait((drain, self.writer), return_when=asyncio.FIRST_COMPLETED)
            if self.writer in done:
                await self.writer
            await drain
        finally:
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)

    async def tools(self):
        while True:
            calls = [await self.tool_queue.get()]
            while not self.tool_queue.empty():
                calls.append(self.tool_queue.get_nowait())
            self.tools_running = True
            self.hub.send_activity("working", ", ".join(call["function"]["name"] for call in calls))
            try:
                await self.rows.join()
                if self.hub.closed or self.hub.ended_by_owner:
                    continue  # End starts nothing queued; the request stays a host fact and records as not run
                changes = await self.host.tool_batch(calls)
                if self.hub.closed or self.hub.ended_by_owner:
                    continue  # the batch is history, whatever End left unstarted; an ending call delivers nothing more
                images, silent = self.attention.tool_outputs(calls, changes)
                for image in images:
                    await self.host.observe_context(image)  # exact requested pixels survive a fresh engine projection
                completed_in = self.conversation  # a session adopted before its last row lacks it and is refitted below
                if not self.tool_refit.is_set():
                    await self.connected.wait()
                owed = [call["id"] for call in calls if call["id"] not in silent]
                ordinary = bool(owed)
                looked = any(call["function"]["name"] == "attend" and call["id"] not in silent for call in calls)
                if looked:
                    self.looked_calls.update(owed)
                # Native outputs answer only the issuing conversation. A session fitted after this batch
                # completed holds its unit in setup; any other gets a fresh one, never an orphan result.
                native = {self.issued.get(call["id"], (completed_in,))[0] for call in calls} == {self.conversation}
                # 28.10: Gemini cannot change a live session's declarations; a call that enabled desktop
                # tools kept asking for tools it never received. Such an engine renews at this boundary.
                offline_tools = getattr(self.session, "tools_need_renewal", None)
                needs_fit = (not native or self.host.state.get("refit") or self.host.renewal_due()
                             or (offline_tools is not None and offline_tools(tuple(self.host.schemas))))
                if not needs_fit and (self.resume or images):
                    # Fresh projection can collapse a prior frame that this socket
                    # still holds. Price the complete native delivery, images too.
                    needs_fit = not await self.host.delivery_fits(
                        tuple(changes) + tuple(images), self.serving_catalog())
                # Ownership is read after every wait above: a session adopted meanwhile holds the unit in its fit.
                if not native or self.conversation != completed_in:
                    for call in calls:
                        self.issued.pop(call["id"], None)
                if self.conversation != completed_in:
                    if ordinary and not self.hub.ended_by_ouroboros:
                        self.offer_tool_answer(owed, looked=looked)
                    continue
                output = ToolOutputs(tuple(changes) + tuple(images), tuple(self.host.schemas), silent=silent)
                if (self.renewing and not self.resume) or self.tool_refit.is_set() or needs_fit:
                    if not self.hub.ended_by_ouroboros and (native or ordinary):
                        # Results and execution receipts are already canonical.
                        # Waiting for quiet would deadlock on the withheld result.
                        offers = []
                        if ordinary:
                            offers = [{**image, "_canonical": True} for image in images]
                            self.pending_offers.extend(offers)
                            offers.append(self.offer_tool_answer(owed, looked=looked))  # leaves custody with a withheld output
                        if needs_fit or not self.renewing:
                            self.request_renewal()
                        self.tool_refit.set()
                        # A unit no native write can deliver waits for its fit here too, its pointer behind it.
                        self.deferred_tool_outputs.append((output if native else None, offers))
                        if not self.renewing:
                            self.tool_refit.clear()
                            self.delivery_due.set()
                    # A deliberate goodbye needs only playback, no further
                    # native result or connection that could cut it short.
                else:
                    if ordinary:
                        # An ordinary result requests continuation. Between the
                        # write and its native answer, quiet must not reopen the
                        # session and swallow the opening's first spoken words.
                        self.response_open = True
                    await self.write_admitted(output, completed_in)
                if len(silent) == len(calls) and not self.active_responses and self.conversation == completed_in:
                    self.response_open = False
            finally:
                for _ in calls:
                    self.tool_queue.task_done()
                self.tools_running = False
                self.hub.send_activity("idle")

    async def write_admitted(self, output, conversation):
        """An output that needs no fit, written at most once here. If it must wait, behind an earlier output or for a
        connection, or if the socket closes, it joins the queue already admitted (no offers, no measure): the worker of a
        resumed connection writes it; a fresh fit holds it, and consume carries its armed answer."""
        session = self.session
        if self.connected.is_set() and not self.deferred_tool_outputs:
            try:
                await self.send(output)
                return
            except ConnectionClosed:
                if self.conversation != conversation:
                    self.retire_calls(output.messages)
                    return
                if self.session is session:  # its reader is about to classify this; the next ready delivers it
                    self.deferred_tool_outputs.append((output, None))
                    return
        else:
            self.arm_answer(self.owed_calls(output), session)
        self.deferred_tool_outputs.append((output, None))
        self.delivery_due.set()

    async def reader_wrote(self, write):
        """The reader's own write meeting a closed socket never stops it reading: its next receive classifies that loss."""
        try:
            await write
        except ConnectionClosed:
            return False
        return True

    async def interrupt(self):
        self.hub.flush()  # the page drops this audio whatever the socket can still be told
        writable = await self.reader_wrote(self.session.send(Interrupt("", 0)))
        for item in tuple(self.audio_ms):
            if item in self.interrupted or self.heard(item):
                continue
            self.interrupted.add(item)
            if writable:
                played = min(self.played.get(item, 0), int(self.audio_ms[item]))
                writable = await self.reader_wrote(self.session.send(Interrupt(item, played)))
            await self.finish_spoken(item, interrupted=True)

    async def input_frames(self):
        frames = await self.attention.deliver(self.session)
        if frames:
            await self.context_update(frames)

    async def provider(self):
        while True:
            refusal = None
            try:
                events = await self.session.receive()
            except (OSError, ConnectionClosed) as exc:
                from ouroboros.realtime_session import transport_loss
                if not transport_loss(exc):
                    received = getattr(exc, 'rcvd', None)
                    detail = f'{received.code} {received.reason}'.strip() if received else 'provider ended the live connection'
                    raise RealtimeSetupError('provider_closed', detail) from exc
                self.lost_connection(resuming=False)
                return
            # This batch was converted before anything delivered later: only then may a response begin an answer.
            settles = [due for due in (self.answer_due, self.room_due) if self.delivered(due)]
            for event in events:
                data = event.data
                if refusal is not None and not (event.kind == "response_done"
                        or (event.kind == "caption" and data.get("final"))):
                    continue
                if event.kind == 'audio' or event.kind == 'caption' and data['speaker'] == 'ouroboros':
                    # Refit must wait for speech to finish; continuation custody is unchanged.
                    self.tool_responses.difference_update(self.active_responses)
                if event.kind == "ready":
                    if self.tool_refit.is_set():
                        continue  # a late ready cannot release work to the retiring connection
                    self.timing("ready")
                    self.input_rate, self.output_rate = data["input_rate"], data["output_rate"]
                    self.hub.set_engine_rates(self.input_rate, self.output_rate)
                    self.hub.send_state(cp.STATE_LIVE)
                    if not self.host.state.get("greeted"):
                        if not await self.reader_wrote(self.session.send(Respond(self.host.opening))):
                            continue  # never greeted: the replacement does it
                        self.host.state["greeted"] = True
                    self.connected.set()
                    if self.lost_while_speaking and not self.connection_resumed:
                        await self.offer_fact("Someone was speaking when the voice connection was lost; words said just "
                            "before this reconnection may not have been heard, and the transcript holds only what "
                            "was recognised. You may ask for a repeat, continue or stay silent.")
                    self.lost_while_speaking = False
                    self.delivery_due.set()  # what waited for a connection, mail before its answer pointer
                elif event.kind == "audio":
                    if data["rate"] != self.output_rate:
                        raise RealtimeSetupError("audio_rate", "provider changed its output rate")
                    item = data["item_id"]
                    # Some engines cannot make a function result silent. The
                    # goodbye already queued is the last playback of this call.
                    if item in self.interrupted or self.hub.ended_by_ouroboros:
                        continue
                    if self.playing_item != item:
                        self.hub.send_caption("ouroboros", "", False, item)
                        self.playing_item = item
                    self.audio_ms[item] = self.audio_ms.get(item, 0) + len(data["pcm"]) * 500 / self.output_rate
                    self.hub.play(data["pcm"], item)
                    if not getattr(self, "timed_audio", False):
                        self.timed_audio = True
                        self.timing("first_audio")
                elif event.kind == "caption":
                    await self.caption(data)
                elif event.kind == "speech_started":
                    await self.interrupt()
                elif event.kind == "input_started":
                    self.input_open = True
                    self.spoken_answers = {data["response_id"]} if data.get("response_id") else set()
                    await self.reader_wrote(self.input_frames())
                elif event.kind == "input_done":
                    self.input_open = False
                elif event.kind == "response_started":
                    response = data["response_id"]
                    if response not in self.started_responses:
                        try:
                            self.host.response_started(response)
                        except RealtimeSetupError as exc:
                            # This received batch may already contain the bill
                            # and final transcript. Keep those receipts while
                            # admitting no more playback or tool work.
                            refusal = exc
                            continue
                        self.started_responses.add(response)
                        for due in settles:
                            if due is self.answer_due or due is self.room_due:
                                due.answering.add(response)
                        if self.spoken_answers is not None:
                            self.spoken_answers.add(response)
                    if response not in self.receipts:
                        self.active_responses.add(response)
                    self.response_open = bool(self.active_responses) or self.response_open
                    self.hub.send_turn(cp.TURN_ANSWERING)
                elif event.kind == "response_done":
                    response = data["response_id"]
                    if data['status'] in {'completed', 'interrupted'}:
                        self.unproven_resumes = 0
                        if not data.get('continues'):
                            self.answered(response)
                            if data['status'] == 'completed' and response in (self.spoken_answers or ()):
                                self.spoken_answers, self.spoken_known = None, False
                    receipt = {key: data[key] for key in ("usage", "raw_usage", "status")}
                    if self.receipts.get(response) != receipt:
                        self.host.response_done(response, **receipt)
                        self.receipts[response] = receipt
                    self.active_responses.discard(response)
                    self.response_open = bool(self.active_responses) or bool(data.get("continues"))
                    self.hub.send_turn(cp.TURN_AWAITING if self.response_open or self.tools_running else cp.TURN_IDLE)
                elif event.kind == "tool_call":
                    self.tool_responses.update(self.active_responses)
                    call = {"id": data["call_id"], "type": "function", "function": {
                        "name": data["name"], "arguments": data["arguments"]}}
                    self.issued[call["id"]] = (self.conversation, self.resume_revision)
                    await self.rows.put({"tool_request": call})  # a host fact before its effect, queued or not
                    await self.tool_queue.put(call)
                elif event.kind == "renew":
                    self.expiry_announced = True
                    self.request_renewal(resuming=data["resume"], within=data["within_s"])
                elif event.kind == "resumable":
                    self.resume_revision += 1
                    self.host.state["resume_handle"] = data["handle"]
                elif event.kind == "closed":
                    resumable = bool(data.get('resumable') and self.host.state.get('resume_handle'))
                    if not resumable and not data.get('transport_loss'):
                        # The engine's own close words (code and sentence) are the owner's reason (R15-4).
                        raise RealtimeSetupError("provider_closed", data.get("detail") or "provider ended the live connection")
                    self.lost_connection(resuming=resumable and self.resume_allowed())
                    return
            if refusal is not None:
                raise refusal

    def lost_connection(self, *, resuming):
        if not self.expiry_announced:
            self.unproven_resumes += 1
            if self.unproven_resumes > UNPROVEN_RESUMES:
                raise RealtimeSetupError('reconnect_failed', 'the voice connection kept dropping before it answered again')
        self.provider_closed = True
        self.connected.clear()
        self.lost_while_speaking |= self.input_unsettled()
        self.input_open = False  # the dead transport cannot deliver another VAD boundary
        if not self.renewing:
            self.timing('requested')
        self.renewing, self.resume = True, resuming
        if not resuming:
            self.active_responses.clear()
            self.response_open = False
        self.hub.send_state(cp.STATE_RECONNECTING, 'provider_renewal')

    def answer_delivered(self):
        return self.delivered(self.answer_due)

    def delivered(self, due):
        """What asks for this answer reached the provider: its adapter holds those rows, or the pointer asking for their
        answer, only after a successful native write."""
        if due is not None and not due.delivered:
            held = getattr(self.session, 'held_context_rows', None) or ()
            due.at = next((index for index in range(due.mark, len(held)) if due.asks(held[index])), None)
            due.delivered = due.at is not None
        return due is not None and due.delivered

    def sent_unrecorded(self, session, rows):
        """An older session keeps no delivery record: its send boundary stands for one."""
        if session is self.session and getattr(session, 'held_context_rows', None) is None:
            for due in (self.answer_due, self.room_due):
                if due is not None and any(due.asks(row) for row in rows):
                    due.delivered = True

    def answered(self, response):
        """A response begun after delivery has finished. It answers what it also ended holding: an adapter that writes
        queued updates as a turn ends records what each response held before them."""
        ended = getattr(self.session, 'response_context_rows', {}).get(response)
        for due in (self.answer_due, self.room_due):
            if due is None or response not in due.answering or (ended is not None and due.at is not None
                                                                and len(ended) <= due.at):
                continue
            if due is self.answer_due:
                self.answer_due = None
            else:  # already-answered words offer nothing more
                self.room_due = None
                self.pending_offers[:] = [row for row in self.pending_offers if all(row is not p for p in due.pointers)]

    def resume_allowed(self):
        """A native resume needs a handle published after every call this conversation still owes natively
        and after its last native result; a read update is not published until the reader converts it."""
        if not self.host.state.get('resume_handle') or any(
                conversation == self.conversation and revision >= self.resume_revision
                for conversation, revision in self.issued.values()):
            return False
        if self.resume_tool_output is None:
            return True
        return bool(getattr(self.session, 'resume_after_tool_response', self.resume_tool_output[0] != self.resume_revision))

    def offer_tool_answer(self, ids=(), *, looked=False):
        """One pointer, whichever batches owe it. With call ids it carries their answer onto this session: only a
        response begun after the adapter holds that pointer settles it. A look answered "already visible" holds no
        row: its frame was transport state of an earlier socket, so the pointer asks for it again."""
        pointer = self.tool_answer_pointer()
        offered = next((row for row in self.pending_offers if row.get("content") == pointer["content"]), None)
        if offered is None:
            self.pending_offers.append(offered := pointer)
        if looked:
            offered["_requested_look"] = True
        if ids:
            offered["_tool_answer"] = sorted({*offered.get("_tool_answer", ()), *ids})
            self.arm_answer(ids, self.session)
        return offered

    def arm_answer(self, ids, session):
        if ids:
            self.answer_due = _Owed(session or self.last_session, ids=ids)

    def owe_room_answer(self, rows=(), pointer=None):
        """Room words, or the pointer to them, carry their answer onto this session as a result's pointer does. A room
        pointer already pending asks for them too."""
        previous = self.room_due.pointers if self.room_due is not None else ()
        pending = [offer for offer in previous if any(offer is row for row in self.pending_offers)]
        if pointer is None and not pending:
            pointer = next((row for row in self.pending_offers if row.get("content") == ROOM_POINTER), None)
            if pointer is None:
                self.pending_offers.append(pointer := self.room_answer_pointer())
        pointers = [*pending, *([pointer] if pointer is not None else [])]
        self.room_due = _Owed(self.session or self.last_session, rows=rows, pointers=pointers, room=True)

    async def peers(self):
        # The host's independent control watcher owns Stop/Panic/budget reads.
        # Awaiting those store reads here makes owner audio queue and overflow.
        while True:
            self.track_offer_wait()
            await self.check_preparation()
            if self.hub.closed or self.hub.ended_by_owner:
                return
            if self.provider_closed:
                # The ended transport owns this resume transition. Keep any
                # pending refit for the resumed response's quiet boundary.
                return
            if self.hub.peers():
                self.ever_attached = True
                self.empty_since = time.monotonic()
            elif time.monotonic() - self.empty_since >= EMPTY_CALL_GRACE_S:
                if self.ever_attached:
                    raise RealtimeSetupError("peers_left", "no peer is attached")
                raise RealtimeSetupError("not_connected", "no one joined the call")
            if self.hub.ended_by_ouroboros:
                if not self.unheard():
                    # Do not wait for extended thinking, more input, or another
                    # native boundary after our deliberate goodbye.
                    self.renewing = False
                    return
            else:
                if self.tool_refit.is_set() and self.refit_ready():
                    # A withheld result's fit is taken here, between whole peer events: a planned switch never
                    # cancels a microphone write and then guesses what it did. Consume rechecks after joining
                    # its rows, so the page hears of no reconnection yet.
                    return
                if self.host.context_view_pending() and self.quiet():
                    await self.rows.join()
                    if self.quiet() and await self.host.apply_context_view():
                        self.request_renewal()
                if (self.host.renewal_due() or self.host.state.get("refit")
                        or self.host.state.get("refit_after_answer") and self.quiet()):
                    self.request_renewal()
                if self.renewing:
                    if self.quiet() and (self.resume or self.preparation_ready()):
                        self.timing("quiet")
                        if self.resume and not self.resume_allowed():
                            # No current handle covers this conversation: a fresh fit continues it.
                            self.resume = False
                            if self.resume_tool_output is not None and self.resume_tool_output[1]:
                                self.offer_tool_answer()  # always, even if the old socket spoke
                            self.request_renewal()
                            continue  # adopt the fresh result-bearing fit before offering its pointer
                        self.hub.send_state(cp.STATE_RECONNECTING, "provider_renewal")
                        return
                    if self.renewal_deadline is not None and time.monotonic() >= self.renewal_deadline:
                        if self.resume and not self.resume_allowed():
                            self.resume = False
                            self.request_renewal()  # prepare a fresh successor before the engine closes
                        # 28.10: a Gemini call ended mid-sentence when the talk left no quiet moment
                        # within 30 s of an audio-span renewal. The renewal waits for the next quiet
                        # moment; a provider that closes first is resumed from its handle.
                        self.renewal_deadline = None
            # Drain bursts without sleeping per frame, but revisit this call's
            # own maintenance even when continuous PCM keeps the mixer nonempty.
            drain_until = time.monotonic() + .01
            drain_busy = False
            while not (self.hub.closed or self.hub.ended_by_owner or self.provider_closed):
                event = self.hub.next_event(0)
                if event is None:
                    break
                if isinstance(event, cp.PeerAudio) and self.input_rate:
                    self.meeting_audio |= event.role == cp.PEER_MEETING
                    participant = self.participants.get(event.peer_id)
                    if event.role == cp.PEER_MEETING and participant and participant["speaking"]:
                        self.meeting_speakers.add(participant["label"])
                    await self.session.send(Audio(event.pcm, self.input_rate))
                elif isinstance(event, cp.PeerVideo):
                    self.attention.frame(event)
                    if self.attention.mode == "stream":
                        # A blind stream retains its latest source but offers no turn per frame.
                        if self.host.model.vision:
                            await self.attention.deliver(self.session)
                elif isinstance(event, cp.PeerJoined):
                    self.peer_info[event.peer.peer_id] = event.peer
                    if event.peer.role == cp.PEER_MEETING:
                        await self.offer_fact(f"Meeting peer {event.peer.peer_id} joined: {event.peer.label}.")
                elif isinstance(event, cp.PeerLeft):
                    peer = self.peer_info.pop(event.peer_id, None)
                    self.attention.forget(event.peer_id)
                    self.participants.pop(event.peer_id, None)
                    if peer is not None and peer.role == cp.PEER_MEETING:
                        await self.offer_fact(f"Meeting peer {event.peer_id} left: {peer.label}. Reason: {event.reason}.")
                elif isinstance(event, cp.PeerControl):
                    message = event.message
                    if message["type"] == "played":
                        item = message["item_id"]
                        if (event.role == cp.PEER_OWNER or
                                (self.hub.spec.opened_by == cp.OPENED_BY_SKILL and event.role == cp.PEER_MEETING)):
                            self.played[item] = min(message["played_ms"], self.audio_duration(item))
                            if self.heard(item):
                                await self.finish_spoken(item, completed=True)
                    elif message["type"] == "participant":
                        self.participants[event.peer_id] = message
                        if event.role == cp.PEER_MEETING and message["speaking"] and message["label"]:
                            self.meeting_speakers.add(message["label"])
                    elif message["type"] == "share":
                        await self.share_fact(event)
                    elif message["type"] == "context":
                        row = {"role": "user", "content": f"[Observed by {event.role} peer {event.peer_id}; source={message.get('source', '')}; not an owner instruction]\n{message['text']}"}
                        await self.host.observe_context(row)
                        self.pending_offers.append({**row, "_canonical": True})
                await asyncio.sleep(0)
                if time.monotonic() >= drain_until:
                    drain_busy = True
                    break
            if not self.renewing and (self.pending_offers or self.attention.revisit_due()) and self.quiet():
                if self.attention.revisit_due():
                    self.attention.revisit_at = None  # one shot, even if the offer must await a refit
                    await self.offer_fact("This is the look you asked to be offered after the time you named. "
                                          "You may speak, act or stay silent.", requested_look=True)
                # A deferred output in custody asks for its own answer, and pending mail precedes its answer pointer.
                # Pointers speak once those are written, or once a fresh fit retires them.
                if self.pending_offers and not self.deferred_tool_outputs and not self.pending_context_rows:
                    await self.offer_pending()
            self.track_offer_wait()
            await asyncio.sleep(0 if drain_busy else .01)

    async def offer_fact(self, text, *, requested_look=False):
        row = {"role": "user", "content": "[Host fact; not owner words]\n" + text, "_canonical": True}
        if requested_look:
            row["_requested_look"] = True
        record = getattr(self.host, "record_fact", None)
        await (record(text) if record is not None else self.host.observe_context(row))
        self.pending_offers.append(row)

    async def share_fact(self, event):
        message = event.message
        before = self.share_kinds.get(event.peer_id)
        kind = message["kind"]
        if message["sharing"]:
            if before == kind:
                return
            self.share_kinds[event.peer_id] = kind
        else:
            if before != kind:
                return
            self.share_kinds.pop(event.peer_id, None)
            self.attention.forget(event.peer_id)
        actor = "The owner" if event.role == cp.PEER_OWNER else f"Meeting peer {event.peer_id}"
        await self.offer_fact(f"{actor} {'started' if message['sharing'] else 'stopped'} showing the {kind}.")

    async def offer_pending(self):
        if not self.pending_offers or not self.connected.is_set() or not self.quiet():
            return
        if self.host.renewal_due() or self.host.state.get("refit") or self.host.state.get("refit_after_answer"):
            if self.request_renewal() is not False:
                return
            await self.guard_serving(self.pending_offers)
        self.attention.include_images(self.pending_offers)
        requested = any(row.get("_requested_look") for row in self.pending_offers)
        frames = await self.attention.deliver(self.session, requested=requested)
        self.pending_offers = [*frames, *self.pending_offers]
        if self.host.renewal_due():
            if self.request_renewal() is not False:
                return
            await self.guard_serving(self.pending_offers)
        # Persistence or image ACKs may yield to new speech. Keep the facts until
        # quiet is observed again; the frame is already accounted for exactly once.
        if not self.quiet():
            return
        rows, self.pending_offers = self.pending_offers, []
        try:
            await self.session.send(ContextUpdate(tuple(rows)))
        except BaseException:
            self.pending_offers = [*rows, *self.pending_offers]
            raise
        self.sent_unrecorded(self.session, rows)
        self.response_open = True

    async def mailbox(self):
        while True:
            mail = await self.host.mailbox()
            if mail:
                getattr(self.host, 'take_mail_answer', lambda: False)()
                await self.context_update(mail, request=True)
            await asyncio.sleep(.25)

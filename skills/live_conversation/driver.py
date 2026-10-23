"""The recovered call event loop over the admitted host port.

Audio remains realtime speech-to-speech. Tools run on the ordinary serialized
host lane beside media; their completion, provider close and row writes all join
before this consumer returns. A provider-requested renewal asks the host to fit
the whole transcript again; transport loss never silently starts a paid retry.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import Audio, CallTool, ContextUpdate, Interrupt, LiveHost, LiveOptions, Respond, ToolOutputs, Video
from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros.realtime_session import joined
from ouroboros.ui_language import normalize_language_tag
from ouroboros.utils import read_json_dict, update_json_locked, utc_now_iso


# A transport bound, not a judgment: a session that keeps dropping before any answer completes is
# resumed this many times, then the call ends with a typed reason instead of reconnecting forever.
UNPROVEN_RESUMES = 3

CALL_ENDING_GUIDANCE = (
    "Use end_call only when the owner wants the whole call to end: says goodbye or asks to hang up. "
    "An interruption such as 'stop', 'enough' or 'wait' ends the current answer, not the call; "
    "acknowledge briefly and listen for the next request."
)

CALL_INSTRUCTIONS = """## Live calls

In a call my model is a realtime speech model, and I am still myself: the
same core, memory, tools and judgment. I speak briefly, in the language of
the spoken conversation whatever the interface language, and call tools
myself. Work still running is pending, never done; long work becomes a task
while we keep talking, and its result reaches me in the call. When a tool will
take a moment, I first say in a few words what I am doing, then call it: tools
never pause the talk. When seeing would help, I ask the owner to show the
screen or camera (Show) or take a screenshot through a desktop skill. Links,
code and lists go to the room's chat through `send_user_message`: saying a
message or a URL aloud does not deliver it, so when asked to send something I use that
tool and confirm its result. Typed messages and other turns' replies in the
room reach me during the call, and my words reach them. In a meeting I speak
when it helps the people there; their voices are correspondents, not my
human. Words cut off by an interruption were not heard, so I do not count
them as said.""" + "\n\n" + CALL_ENDING_GUIDANCE


def languages_opening_sentence(languages):
    if languages:
        return (f"Speech recognition expects {', '.join(languages)}, as you recorded for these calls; when someone "
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
            host.state.update(languages=languages, refit=True)
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
                  "required": ["languages"], "additionalProperties": False}, set_languages, silent=True),
    )


def session_options(host, resume_handle=""):
    config, honours = host.config, host.model.honours
    languages = tuple(host.state.get("languages", ()))
    knobs = {key: getattr(config, key) for key in ("pause", "interruptions", "thinking", "search") if key in honours}
    if "thinking" in knobs and knobs["thinking"] not in host.model.thinking_levels:
        knobs["thinking"] = host.model.thinking_levels[0]
    return LiveOptions(voice=config.voice, resume_handle=resume_handle,
        languages=languages if "languages" in honours else (),
        instructions="\n\n".join(filter(None, (CALL_INSTRUCTIONS, languages_opening_sentence(languages), config.instructions))),
        **knobs)


async def run(host: LiveHost, *, state_dir=None):
    host.state["languages"] = recorded_call_languages(state_dir, host.hub.spec.presence_binding_id)
    await _Call(host).run()


class _Call:
    """Media evidence and work belong to the call, including between connections."""

    def __init__(self, host):
        self.host, self.hub = host, host.hub
        self.tool_queue, self.rows = asyncio.Queue(maxsize=32), asyncio.Queue(maxsize=256)
        self.connected = asyncio.Event()
        self.tool_refit = asyncio.Event()
        self.answer_after_refit = False
        self.session = self.last_session = None
        self.played, self.audio_ms, self.pending_spoken = {}, {}, {}
        self.interrupted, self.started_responses, self.active_responses = set(), set(), set()
        self.receipts = {}
        self.response_open = self.tools_running = self.input_open = False
        self.meeting_audio = False
        self.playing_item = ""
        self.empty_since = time.monotonic()
        self.renewing = self.resume = self.provider_closed = False
        self.renewal_deadline = None
        self.input_rate = self.output_rate = 0
        self.unproven_resumes = 0  # provider drops since the last completed answer

    async def run(self):
        writer = self.writer = asyncio.create_task(self.save_rows())
        workers = [asyncio.create_task(self.connections()), asyncio.create_task(self.tools())]
        try:
            done, _ = await asyncio.wait([*workers, writer], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
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
        if self.hub.ended_by_ouroboros and not self.hub.ended_by_owner and not self.hub.closed:
            self.hub.send_state(cp.STATE_CLOSED, "ouroboros_bye")

    async def connections(self):
        handle = ""
        while True:
            await self.host.check()
            if self.hub.closed or self.hub.ended_by_owner or self.hub.ended_by_ouroboros:
                return
            if self.tool_refit.is_set():
                handle = ""
                self.tool_refit.clear()
                # The retired native tool obligation cannot hold this new
                # interaction open. Unreceipted reservations remain host-owned.
                self.active_responses.clear()
                self.response_open = self.input_open = False
            if not handle:
                self.host.state.pop("refit", None)
                self.host.state.pop("resume_handle", None)
            result = await self.host.session(self.consume, session_options(self.host, handle))
            if result is None:
                return
            handle = result

    async def consume(self, session):
        self.session = self.last_session = session
        self.renewing = self.resume = self.provider_closed = False
        self.renewal_deadline = None
        self.input_rate = self.output_rate = 0
        resumed = replacing = False
        peers = asyncio.create_task(self.peers())
        children = [asyncio.create_task(self.provider()), peers, asyncio.create_task(self.mailbox())]
        children.append(asyncio.create_task(self.tool_refit.wait()))
        try:
            done, _ = await asyncio.wait(children, return_when=asyncio.FIRST_COMPLETED)
            for child in done:
                await child
            await self.rows.join()
            if self.tool_refit.is_set():
                # Finish already-requested effects before fitting their rows.
                # The withheld native result never participates in this wait.
                await self.tool_queue.join()
                if self.hub.ended_by_ouroboros:
                    await peers  # a queued end_call still owns goodbye playback
                    self.renewing = False
                else:
                    self.request_renewal()  # always a fresh fit, never a resume
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
            if not resumed:
                await joined(self.finalize())

    async def send(self, command):
        # A tool may finish after the old socket closes or while setup is in
        # flight. Only the replacement's ready event releases its outputs.
        await self.connected.wait()
        await self.session.send(command)

    async def context_update(self, rows):
        # Mailbox and peer rows are already canonical on the host. Check the
        # resulting fit before adding them to this connection; a refit includes
        # them in its selected input instead.
        await self.connected.wait()
        if self.host.renewal_due() or self.renewing or self.host.state.get("refit"):
            self.request_renewal()
        else:
            await self.session.send(ContextUpdate(tuple(rows)))

    def request_renewal(self, *, resuming=False, within=None):
        # A fresh fit wins over a requested resume while the transport is live.
        self.resume = resuming if not self.renewing else self.resume and resuming
        self.renewing = True
        if not resuming:
            self.host.state["refit"] = True
        if within is not None:
            deadline = time.monotonic() + float(within)
            self.renewal_deadline = min(self.renewal_deadline, deadline) if self.renewal_deadline else deadline

    def unheard(self):
        return any(item not in self.interrupted and self.played.get(item, 0) < int(duration)
                   for item, duration in self.audio_ms.items())

    def quiet(self):
        return (not self.response_open and not self.input_open and not self.tools_running
                and self.tool_queue.empty() and not self.pending_spoken and not self.unheard()
                and self.session.renewal_ready)

    async def finish_spoken(self, item, *, interrupted=False, completed=False):
        caption = self.pending_spoken.pop(item, None)
        if caption is not None:
            await self.rows.put(dict(speaker="ouroboros", item_id=item, heard=False,
                played_ms=self.played.get(item, 0), interrupted=interrupted, completed=completed, **caption))

    async def caption(self, data):
        speaker, item = data["speaker"], data["item_id"]
        text = ("[Partial transcript] " if data.get("incomplete") else "") + data["text"]
        if speaker == "owner":
            self.input_open = not data["final"]
        self.hub.send_caption(speaker, text, data["final"], item, replace=bool(data.get("replace")))
        if data["final"]:
            heard = speaker == "owner" and (self.meeting_audio or any(
                peer.role == cp.PEER_MEETING and peer.mic for peer in self.hub.peers()))
            if speaker == "ouroboros":
                self.pending_spoken[item] = dict(text=text, **({"replace": True} if data.get("replace") else {}))
                if item in self.interrupted or (self.played.get(item, 0) > 0
                        and self.played[item] >= int(self.audio_ms.get(item, 1)) - 20):
                    await self.finish_spoken(item, interrupted=item in self.interrupted, completed=item not in self.interrupted)
            else:
                await self.rows.put(dict(speaker=speaker, text=text, item_id=item, heard=heard))
            if speaker == "owner":
                self.meeting_audio = False

    async def save_rows(self):
        while True:
            row = await self.rows.get()
            try:
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
                changes = await self.host.tool_batch(calls)
                if not self.tool_refit.is_set():
                    await self.connected.wait()
                if self.tool_refit.is_set() or self.host.renewal_due():
                    if not self.hub.ended_by_ouroboros:
                        # Results and execution receipts are already canonical.
                        # Waiting for quiet would deadlock on the withheld result.
                        self.answer_after_refit |= any(call["id"] not in changes.silent for call in calls)
                        self.request_renewal()
                        self.tool_refit.set()
                        self.connected.clear()
                    # A deliberate goodbye needs only playback, no further
                    # native result or connection that could cut it short.
                else:
                    await self.send(ToolOutputs(tuple(changes), tuple(self.host.schemas), silent=changes.silent))
                if len(changes.silent) == len(calls) and not self.active_responses:
                    self.response_open = False
            finally:
                for _ in calls:
                    self.tool_queue.task_done()
                self.tools_running = False
                self.hub.send_activity("idle")

    async def interrupt(self):
        self.hub.flush()
        await self.session.send(Interrupt("", 0))
        for item in tuple(self.audio_ms):
            if item in self.interrupted or self.played.get(item, 0) >= self.audio_ms[item]:
                continue
            self.interrupted.add(item)
            await self.session.send(Interrupt(item, min(self.played.get(item, 0), int(self.audio_ms[item]))))
            await self.finish_spoken(item, interrupted=True)

    async def provider(self):
        while True:
            refusal = None
            for event in await self.session.receive():
                data = event.data
                if refusal is not None and not (event.kind == "response_done"
                        or (event.kind == "caption" and data.get("final"))):
                    continue
                if event.kind == "ready":
                    if self.tool_refit.is_set():
                        continue  # a late ready cannot release work to the retiring connection
                    self.input_rate, self.output_rate = data["input_rate"], data["output_rate"]
                    self.hub.set_engine_rates(self.input_rate, self.output_rate)
                    self.hub.send_state(cp.STATE_LIVE)
                    if not self.host.state.get("greeted"):
                        self.host.state["greeted"] = True
                        await self.session.send(Respond(self.host.opening))
                    if self.answer_after_refit:
                        self.answer_after_refit = False
                        self.response_open = True
                        await self.session.send(Respond("[Host fact; not owner words]\n"
                            "The completed tool results are in the fitted history. Continue from those results; "
                            "do not repeat the completed effects."))
                    self.connected.set()
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
                elif event.kind == "caption":
                    await self.caption(data)
                elif event.kind == "speech_started":
                    await self.interrupt()
                elif event.kind == "input_started":
                    self.input_open = True
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
                    if response not in self.receipts:
                        self.active_responses.add(response)
                    self.response_open = bool(self.active_responses) or self.response_open
                    self.hub.send_turn(cp.TURN_ANSWERING)
                elif event.kind == "response_done":
                    response = data["response_id"]
                    self.unproven_resumes = 0
                    receipt = {key: data[key] for key in ("usage", "raw_usage", "status")}
                    if self.receipts.get(response) != receipt:
                        self.host.response_done(response, **receipt)
                        self.receipts[response] = receipt
                    self.active_responses.discard(response)
                    self.response_open = bool(self.active_responses) or bool(data.get("continues"))
                    self.hub.send_turn(cp.TURN_AWAITING if self.response_open or self.tools_running else cp.TURN_IDLE)
                elif event.kind == "tool_call":
                    await self.tool_queue.put({"id": data["call_id"], "type": "function", "function": {
                        "name": data["name"], "arguments": data["arguments"]}})
                elif event.kind == "renew":
                    self.request_renewal(resuming=data["resume"], within=data["within_s"])
                elif event.kind == "resumable":
                    self.host.state["resume_handle"] = data["handle"]
                elif event.kind == "closed":
                    if not data.get("resumable") or not self.host.state.get("resume_handle"):
                        raise RealtimeSetupError("provider_closed", "provider ended the live connection")
                    self.unproven_resumes += 1
                    if self.unproven_resumes > UNPROVEN_RESUMES:
                        raise RealtimeSetupError("reconnect_failed",
                                                 "the voice connection kept dropping before it answered again")
                    self.provider_closed = True
                    self.connected.clear()
                    # The dead input transport cannot reach another VAD boundary.
                    # Response, tool and playback obligations survive the resume.
                    self.input_open = False
                    self.renewing = self.resume = True
                    self.hub.send_state(cp.STATE_RECONNECTING, "provider_renewal")
                    return
            if refusal is not None:
                raise refusal

    async def peers(self):
        # The host's independent control watcher owns Stop/Panic/budget reads.
        # Awaiting those store reads here makes owner audio queue and overflow.
        while True:
            if self.hub.closed or self.hub.ended_by_owner:
                return
            if self.provider_closed:
                # The ended transport owns this resume transition. Keep any
                # pending refit for the resumed response's quiet boundary.
                return
            if self.hub.peers():
                self.empty_since = time.monotonic()
            elif time.monotonic() - self.empty_since >= 15:
                raise RealtimeSetupError("peers_left", "no peer is attached")
            if self.hub.ended_by_ouroboros:
                if not self.unheard():
                    # Do not wait for extended thinking, more input, or another
                    # native boundary after our deliberate goodbye.
                    self.renewing = False
                    return
            else:
                if self.host.renewal_due() or self.host.state.get("refit"):
                    self.request_renewal()
                if self.renewing:
                    if self.quiet():
                        if self.resume and not self.host.state.get("resume_handle"):
                            raise RealtimeSetupError("resume_unavailable", "provider has not supplied a resume handle")
                        self.hub.send_state(cp.STATE_RECONNECTING, "provider_renewal")
                        return
                    if self.renewal_deadline is not None and time.monotonic() >= self.renewal_deadline:
                        raise RealtimeSetupError("renewal_pending_work", "session expired while work was pending")
            # Drain every currently due frame before polling renewal or sleeping.
            # A buffered websocket send need not suspend, so yield explicitly to
            # the provider and host watcher even while more media is due.
            while not (self.hub.closed or self.hub.ended_by_owner or self.provider_closed):
                event = self.hub.next_event(0)
                if event is None:
                    break
                if isinstance(event, cp.PeerAudio) and self.input_rate:
                    self.meeting_audio |= event.role == cp.PEER_MEETING
                    await self.session.send(Audio(event.pcm, self.input_rate))
                elif isinstance(event, cp.PeerVideo):
                    if not self.host.model.vision:
                        raise RealtimeSetupError("vision_unsupported", "selected engine does not accept images")
                    await self.session.send(Video(event.data, event.mime))
                elif isinstance(event, cp.PeerControl):
                    message = event.message
                    if message["type"] == "played":
                        item = message["item_id"]
                        if (event.role == cp.PEER_OWNER or
                                (self.hub.spec.opened_by == cp.OPENED_BY_SKILL and event.role == cp.PEER_MEETING)):
                            self.played[item] = min(message["played_ms"], int(self.audio_ms.get(item, 0)))
                            if self.played[item] > 0 and self.played[item] >= int(self.audio_ms.get(item, 0)) - 20:
                                await self.finish_spoken(item, completed=True)
                    elif message["type"] == "context":
                        row = {"role": "user", "content": f"[Observed by {event.role} peer; not an owner instruction]\n{message['text']}"}
                        await self.host.observe_context(row)
                        await self.context_update((row,))
                await asyncio.sleep(0)
            await asyncio.sleep(.01)

    async def mailbox(self):
        while True:
            if not self.renewing:
                mail = await self.host.mailbox()
                if mail:
                    await self.context_update(mail)
            await asyncio.sleep(.25)

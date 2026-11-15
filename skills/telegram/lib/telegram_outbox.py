"""The bridge's outbound journal: what is mirrored to the owner's Telegram chat arrives once, or says why not.

One subscribed event (``chat.outbound``, a photo, video, document, link list, quiz card or quiz lifecycle fact) is
one operation, run by its unchanged handler. Every request that handler makes to create or edit a message
(``telegram_api.JOURNALED_METHODS``) is the operation's next part, numbered in the order the handler asks for it.
Nothing is sent before it is saved under ``<state>/outbox/``: the event exactly as published (its reply, source and
label facts included), the operation's place in the queue, and each part as ``sending`` before its request, with
its outcome after. One operation runs at a time. An event that would have to wait (behind a running or waiting
operation) is saved before it waits: ``chat.outbound`` first runs its handler as a decision that sends nothing
(``_Probe``), so an event its filters drop at publish time is never saved or sent; other topics are saved as they
come. Saved operations run in arrival order while the supervised ``outbox`` task runs; if that task has ended, a
new event runs at once, ahead of what waits (it is still saved first).

A part whose request provably never left (``TelegramTransportError.unsent``: a refused connection or tunnel, a
connect or pool timeout) or that Telegram's flood control refused (429, waiting its ``retry_after``) waits
(``retry_wait``). The ``outbox`` task runs the operation again when it is due, paced by the bridge's shared backoff
(``next_telegram_retry_delay``); a handler never sleeps. That run repeats the handler on the saved event: an
answered part is not sent again, its recorded answer stands in for the request, and a part whose request differs
from the saved one stops the operation instead. An answer that may follow delivery (a read or write timeout, a
connection closed after the request, a 5xx, an unreadable body) makes its part ``unknown``: never sent again, and
the handler goes on as for a message whose id it does not know; an edit counts as made, so no plain-text edit or
fresh message follows it. A part still ``sending`` when the process stopped is ``unknown`` at the next start.
Telegram's other refusals are ``refused``; the handler's own fallback (plain text, a document) is its next part.

The journal fails closed: a part is never sent unless its ``sending`` state is on disk. An event that cannot be
saved is not sent (counted for the Connection form, which says the journal is failing); a saved operation whose
next state cannot be written waits for its retry instead of sending.

States: an operation is ``pending`` (saved, waiting for its turn or a token), ``sending``, ``retry_wait``, or
finished. A finished one is ``confirmed`` (every part it asked for was delivered; its files are removed),
``unknown``, ``refused``, ``failed`` (stopped: expired after ``MAX_AGE_SEC``, changed before its retry, the token
changed after a delivered part, the queue full, its handler failing) or ``superseded`` (on its retry the handler
no longer asked for the part that waited, e.g. progress mirroring was turned off). A saved event whose handler
asks nothing of Telegram when it runs is removed without a trace. The newest ``MAX_FINISHED`` finished ones that
were not confirmed keep metadata only, never words or bytes, for the Connection form. An event published while
the skill is not loaded never reaches it (the host's event bus keeps nothing).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import time
import uuid
import weakref
from typing import Any, Awaitable, Callable, Dict, Optional

from .telegram_api import (
    OUTBOUND_JOURNAL,
    TELEGRAM_RETRY_INITIAL_SEC,
    TelegramBindingChanged,
    TelegramRequestRejected,
    TelegramTokenMissing,
    TelegramTransportError,
    next_telegram_retry_delay,
)
from .telegram_token import binding_for

SCHEMA = 1
MAX_AGE_SEC = 24 * 3600  # still waiting after a day: stopped, never delivered that late
MAX_ACTIVE = 500
MAX_ACTIVE_BYTES = 256 * 1024 * 1024  # UTF-8 bytes of the saved events of waiting operations, media included
MAX_FINISHED = 100
TOKEN_WAIT_SEC = 5.0
_NOT_MODIFIED = "Telegram API editMessageText: message is not modified."
_ACTIVE = frozenset({"pending", "sending", "retry_wait"})
# Delivery manners, not content: Silent mode's sound and the reply anchor may differ between attempts.
_VOLATILE = frozenset({"disable_notification", "reply_to_message_id", "allow_sending_without_reply"})
_SOURCE_KEYS = ("task_id", "subagent_task_id", "row_address", "quiz_id", "card_id", "system_type", "annotation_type")


class OutboxDeferred(BaseException):
    """The operation pauses at this part (it waits to retry, or an earlier operation does).

    A BaseException, like cancellation: a handler's ``except Exception`` fallback never mistakes the pause
    for a failed request and sends something else instead.
    """


class OutboxStopped(BaseException):
    """The operation ends here without sending this part (see the record's ``reason``)."""


class _WouldSend(BaseException):
    """A decision run reached a message request: the event sends something."""


class _Probe:
    """A decision run's journal: the handler's first message request ends the run; nothing is sent."""

    probing = True

    def failed(self, _exc: BaseException) -> None:
        pass

    async def call(self, _client, _method: str, **_request) -> Dict[str, Any]:
        raise _WouldSend()


_PROBE = _Probe()


def _fingerprint(method: str, data: Optional[dict], files: Optional[dict]) -> str:
    """What a part asks for: the method, its fields except delivery manners, and each file's field, name,
    type and size (an operation's own file is fixed by its saved event; a task file by its reference)."""
    digest = hashlib.sha256(method.encode())
    for key in sorted(data or {}):
        if key not in _VOLATILE:
            digest.update(f"\0{key}=".encode() + str(data[key]).encode("utf-8", "surrogatepass"))
    for field in sorted(files or {}):
        spec = files[field]
        name, content, mime = (tuple(spec) + (None, None))[:3] if isinstance(spec, (tuple, list)) else ("", spec, "")
        digest.update(f"\1{field}|{name}|{mime}|{_size(content)}".encode("utf-8", "surrogatepass"))
    return digest.hexdigest()


def _size(content: Any) -> int:
    if isinstance(content, (bytes, bytearray, memoryview)):
        return len(content)
    try:
        return os.fstat(content.fileno()).st_size
    except (AttributeError, OSError, ValueError):
        return -1


def _message_id(payload: Any) -> int:
    result = payload.get("result") if isinstance(payload, dict) else None
    try:
        return int(result.get("message_id") or 0) if isinstance(result, dict) else 0
    except (TypeError, ValueError):
        return 0


def _cause(exc: BaseException) -> str:
    status = getattr(exc, "status_code", 0)
    return getattr(exc, "cause", "") or (f"HTTP {status}" if status else type(exc).__name__)


def _final_state(parts: list) -> str:
    states = {part.get("state") for part in parts}
    for state in ("unknown", "refused", "failed", "superseded"):
        if state in states:
            return state
    return "confirmed"


def _unknown_answer(method: str) -> Dict[str, Any]:
    """What the handler hears for a part whose delivery is unknown: a message without an id, an edit made."""
    return {"ok": True, "result": True if method == "editMessageText" else {}}


class _Operation:
    """One event's run: the journal hook its handler's requests pass through (``TelegramClient.call``)."""

    probing = False

    def __init__(self, outbox: "Outbox", topic: str, event: dict, record: Optional[dict] = None, seq: int = 0):
        self.outbox, self.topic, self.event, self.record = outbox, topic, event, record
        self.seq = int((record or {}).get("seq") or seq)
        self.ordinal = 0
        self.paused = False
        self.stopped = False
        self.failure: Optional[BaseException] = None

    def failed(self, exc: BaseException) -> None:
        """The handler ended in ``exc`` (its ``_report_failure``): unreached waiting parts are not superseded."""
        self.failure = exc

    async def call(self, client, method: str, *, data=None, files=None, timeout: int = 30) -> Dict[str, Any]:
        index, self.ordinal = self.ordinal, self.ordinal + 1
        fingerprint = _fingerprint(method, data, files)
        record = self.record
        part = record["parts"][index] if record is not None and index < len(record["parts"]) else None
        if part is not None:
            if part.get("fingerprint") != fingerprint:
                await self.outbox._stop(self, "changed")
            if record.get("bot") and client.bot_id != record["bot"] and _delivered(record):
                await self._token_changed(part)
            if part["state"] not in _ACTIVE:
                return self._replay(client, part, data)
        else:
            if record is None:
                await self.outbox._admit(self)
                record = self.record
            elif record.get("bot") and client.bot_id != record["bot"] and _delivered(record):
                await self._token_changed(None)
            part = {"n": index, "method": method, "fingerprint": fingerprint, "state": "pending", "attempts": 0}
            record["parts"].append(part)
        return await self._send(client, part, method, data, files, timeout)

    def _replay(self, client, part: dict, data) -> Dict[str, Any]:
        """A finished part's recorded answer, without a request: the handler takes the path it took before."""
        state = part["state"]
        if state == "confirmed":
            client.accepted += 1
            if part.get("not_modified"):
                raise TelegramRequestRejected(_NOT_MODIFIED, status_code=int(part.get("status") or 400))
            message_id = int(part.get("message_id") or 0)
            if message_id:
                return {"ok": True, "result": {"message_id": message_id, "chat": {"id": (data or {}).get("chat_id")}}}
            return {"ok": True, "result": True if part["method"] == "editMessageText" else {}}
        if state == "refused":
            raise TelegramRequestRejected(str(part.get("error") or "Telegram API rejected the request."),
                                          status_code=int(part.get("status") or 0),
                                          plain_retry_safe=bool(part.get("plain_retry_safe")))
        if state == "unknown":
            return _unknown_answer(part["method"])
        raise TelegramTransportError(f"Telegram API delivery of part {part['n'] + 1} is {state}.")

    async def _send(self, client, part: dict, method: str, data, files, timeout) -> Dict[str, Any]:
        record = self.record
        before = part["state"]
        part.update(state="sending", attempts=int(part.get("attempts") or 0) + 1)
        # The bot of its first delivered part owns the operation; until then, the one in force sends it.
        record.update(state="sending", bot=record.get("bot") if _delivered(record) else client.bot_id)
        record["chat_id"] = record.get("chat_id") or _chat(data)
        if not await self.outbox._save(self):  # nothing leaves unless its sending state is on disk
            part["state"] = before
            await self.outbox._unsaved(self)
        try:
            payload = await client.request(method, data=data, files=files, timeout=timeout)
        except TelegramBindingChanged as exc:
            if exc.unsent and not _delivered(record):
                part["state"] = "pending"  # nothing went out with the old bot: the token in force sends it
                await self.outbox._defer(self, 0, "token_changed")
            part.update(state="failed" if exc.unsent else "unknown", reason="token_changed")
            await self.outbox._stop(self, "token_changed", raise_=False)
            raise
        except TelegramTransportError as exc:
            if exc.unsent:
                part["state"] = "retry_wait"
                await self.outbox._defer(self, None, _cause(exc))
            return await self._unknown(part, exc)
        except TelegramRequestRejected as exc:
            if exc.status_code == 429:
                part["state"] = "retry_wait"
                await self.outbox._defer(self, max(1, exc.retry_after), "HTTP 429")
            if not exc.unsent:
                return await self._unknown(part, exc)
            if str(exc) == _NOT_MODIFIED:
                part.update(state="confirmed", not_modified=True, status=exc.status_code)
            else:
                part.update(state="refused", status=exc.status_code, plain_retry_safe=exc.plain_retry_safe,
                            error=str(exc))
            await self.outbox._save(self)
            raise
        except Exception as exc:  # an unexpected failure mid-request may follow delivery too
            return await self._unknown(part, exc)
        part.update(state="confirmed", message_id=_message_id(payload))
        record["retry_delay"] = TELEGRAM_RETRY_INITIAL_SEC
        await self.outbox._save(self)
        return payload

    async def _unknown(self, part: dict, exc: BaseException) -> Dict[str, Any]:
        part.update(state="unknown", reason=_cause(exc))
        await self.outbox._save(self)
        self.outbox._log("warning", f"Telegram {self.topic} part {part['n'] + 1}: delivery unknown ({_cause(exc)}); "
                                    "nothing is sent again for it.")
        return _unknown_answer(part["method"])

    async def _token_changed(self, part: Optional[dict]) -> None:
        """Parts a replaced bot took stay there: nothing more of this operation goes out through another bot."""
        if part is not None and part["state"] in _ACTIVE:
            part.update(state="failed", reason="token_changed")
        await self.outbox._stop(self, "token_changed", raise_=False)
        raise TelegramBindingChanged(sum(1 for item in self.record["parts"] if item.get("state") == "confirmed"))


def _delivered(record: dict) -> bool:
    return any(part.get("state") in {"confirmed", "unknown"} for part in record.get("parts") or ())


def _chat(data) -> int:
    try:
        return int((data or {}).get("chat_id") or 0)
    except (TypeError, ValueError):
        return 0


def _write_private(path: pathlib.Path, content: bytes) -> None:
    """Atomically, owner-only, flushed to disk: a record a restart reads must be the one written."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _unlink(path: pathlib.Path) -> None:
    path.unlink(missing_ok=True)


class Outbox:
    """The journal of one loaded bridge (one PluginAPI): its operations, their queue and its sender task."""

    def __init__(self, api, *, clock: Callable[[], float] = time.time):
        self.api = api
        self.root = pathlib.Path(api.get_state_dir()) / "outbox"
        self.clock = clock
        self._bodies: Dict[str, Callable[[dict], Awaitable[None]]] = {}
        self._ops: Dict[str, dict] = {}
        self._written: set = set()  # ids whose record is on disk: their custody survives a failed later write
        self._reserved: set = set()  # places of events being decided or saved, which nothing later may pass
        self._loaded = False
        self._seq = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # the loop the lock and wake belong to
        self._lock: Optional[asyncio.Lock] = None
        self._wake: Optional[asyncio.Event] = None
        self._waiting_cause = ""
        self._sender_down = False  # the supervised sender ended: nothing would run what waits until a new load
        self._fault = ""  # the last journal write failure, until a write succeeds
        self._not_sent = 0  # events not sent since this load because the journal could not save them
        self._delivered = 0
        self._summary: Dict[str, Any] = {"loaded": False}

    # --- the subscribed handlers and the supervised sender ---

    def subscriber(self, topic: str, body: Callable[[dict], Awaitable[None]], *,
                   decide: bool = False) -> Callable[[dict], Awaitable[None]]:
        """The handler to subscribe for ``topic``. With ``decide`` an event that must wait first runs ``body`` as
        a decision (its filters at publish time); otherwise every such event is saved as it comes."""
        self._bodies[topic] = body

        async def handle(event: Dict[str, Any]) -> None:
            try:
                await self._receive(topic, body, dict(event or {}), decide)
            except Exception as exc:  # the journal itself failed: say so once, never as an unseen task error
                self._log("error", f"Telegram {topic} delivery failed ({type(exc).__name__}).")
        return handle

    async def run(self) -> None:
        """The supervised ``outbox`` task: runs each waiting operation when it is due, oldest first."""
        self._sender_down = False
        try:
            async with self._fifo():
                await self._recover()
            while True:
                record, wait = self._next()
                if record is None:
                    await self._sleep(wait)
                    continue
                await self._resume(record)
        finally:
            self._sender_down = True  # until its restart: a new message no longer queues behind what waits

    def wake(self) -> None:
        if self._wake is not None:
            self._wake.set()

    def summary(self) -> Dict[str, Any]:
        """What the Connection form shows: counts and timing, never words or bytes (any thread)."""
        return dict(self._summary)

    # --- receiving an event ---

    def _bind(self) -> None:
        """The queue's lock and wake belong to the running extension loop; a new loop gets new ones."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop, self._lock, self._wake = loop, asyncio.Lock(), asyncio.Event()

    def _fifo(self) -> asyncio.Lock:
        self._bind()
        return self._lock

    async def _receive(self, topic: str, body, event: dict, decide: bool) -> None:
        try:
            self._load()
        except OSError as exc:  # no journal to hold it: not sent
            await self._refuse(self._new_record(topic, event, self._next_seq(), "pending"), f"journal {type(exc).__name__}")
            return
        lock = self._fifo()
        if not lock.locked() and not self._active() and not self._reserved:
            async with lock:  # free: acquired without waiting, so the event runs now and saves before sending
                op = _Operation(self, topic, event, seq=self._next_seq())
                await self._run(op)
            self.wake()
            return
        # It must wait: saved first, in its place. A decision run says whether the event sends at all.
        seq = self._next_seq()
        self._reserved.add(seq)
        try:
            if decide and not await self._decide(body, event):
                return
            record = await self._custody(topic, event, seq)
        finally:
            self._reserved.discard(seq)
            self.wake()
        if record is None:
            return
        if self._sender_down:  # nothing would run it in order: it goes now, ahead of what waits
            await self._resume(record, ahead=True)

    async def _decide(self, body, event: dict) -> bool:
        """Run ``body`` as a decision: True when it would send. Its filters see the state of publication."""
        context = OUTBOUND_JOURNAL.set(_PROBE)
        try:
            await body(event)
        except _WouldSend:
            return True
        finally:
            OUTBOUND_JOURNAL.reset(context)
        return False

    async def _custody(self, topic: str, event: dict, seq: int) -> Optional[dict]:
        """Save an event that must wait, before it waits; None when the journal cannot hold it (not sent)."""
        record = self._new_record(topic, event, seq, "pending")
        payload = await self._payload(record, event)
        if payload is None:
            return None
        try:
            await asyncio.to_thread(self._write_payload, record, payload)
            await asyncio.to_thread(_write_private, self._record_path(record),
                                    json.dumps(record, ensure_ascii=False).encode("utf-8"))
        except OSError as exc:
            await self._refuse(record, f"journal {type(exc).__name__}")
            return None
        self._ops[record["id"]] = record
        self._written.add(record["id"])
        self._fault = ""
        self._refresh()
        return record

    # --- running one operation ---

    async def _run(self, op: _Operation) -> None:
        """Run ``op``'s handler; the caller holds the queue's lock."""
        context = OUTBOUND_JOURNAL.set(op)
        try:
            await self._bodies[op.topic](op.event)
        except OutboxDeferred:
            op.paused = True
        except OutboxStopped:
            op.stopped = True
        except Exception as exc:  # a handler reports its own failures; this one escaped it
            op.failure = exc
            self._log("error", f"Telegram {op.topic} handler failed ({type(exc).__name__}).")
        finally:
            OUTBOUND_JOURNAL.reset(context)
        await self._settle(op)

    async def _resume(self, record: dict, *, ahead: bool = False) -> None:
        """Run a saved operation; only the queue's due head, unless ``ahead`` (its sender has ended)."""
        async with self._fifo():
            if record.get("id") not in self._ops or record.get("state") not in {"pending", "retry_wait"}:
                return  # finished, or being run, meanwhile
            if not ahead and self._next()[0] is not record:
                return  # an older operation waits, or this one is not due yet
            now = self.clock()
            if now - float(record.get("created_at") or now) > MAX_AGE_SEC:
                record.update(reason="expired")
                for part in record["parts"]:
                    if part.get("state") in _ACTIVE:
                        part.update(state="failed", reason="expired")
                record["state"] = "failed"
                await self._finish(record)
                self._log("warning", f"Telegram {record.get('topic')}: stopped after waiting a day; not sent.")
                return
            binding = binding_for(self.api)
            if binding.state == "pending":
                binding.read_now(self.api)
            if not binding.snapshot()[0]:
                record["next_at"] = now + TOKEN_WAIT_SEC  # no token: a quiet wait, as for every send
                self._refresh()
                return
            body = self._bodies.get(str(record.get("topic") or ""))
            try:
                event = json.loads(await asyncio.to_thread(self._payload_path(record).read_bytes))
            except (OSError, ValueError) as exc:
                event, reason = None, f"event unreadable ({type(exc).__name__})"
            else:
                reason = "no handler" if body is None or not isinstance(event, dict) else ""
            if reason:
                record.update(state="failed", reason=reason)
                await self._finish(record)
                self._log("warning", f"Telegram {record.get('topic')}: not sent ({reason}).")
                return
            await self._run(_Operation(self, str(record["topic"]), event, record))
        self.wake()

    async def _settle(self, op: _Operation) -> None:
        record = op.record
        if record is None or op.paused or record.get("id") not in self._ops:
            return
        if not op.stopped:
            if isinstance(op.failure, TelegramTokenMissing):  # cleared before its first request: wait for one
                record.update(state="pending", next_at=self.clock() + TOKEN_WAIT_SEC)
                await self._save(op)
                return
            if not record["parts"] and op.failure is None:  # its handler asks nothing of Telegram now
                await self._discard(record)
                return
            unreached = "failed" if op.failure is not None else "superseded"
            for part in record["parts"]:
                if part.get("state") in _ACTIVE:
                    part.update(state=unreached, **({"reason": _cause(op.failure)} if op.failure else {}))
            record["state"] = _final_state(record["parts"]) if record["parts"] else "failed"
            if not record["parts"]:
                record["reason"] = _cause(op.failure)
        await self._finish(record)

    # --- the journal ---

    def _new_record(self, topic: str, event: dict, seq: int, state: str) -> dict:
        return {"schema": SCHEMA, "id": f"{seq:012d}-{uuid.uuid4().hex[:8]}", "seq": seq, "topic": topic,
                "created_at": self.clock(), "state": state, "parts": [], "bot": "", "chat_id": 0, "next_at": 0.0,
                "retry_delay": TELEGRAM_RETRY_INITIAL_SEC, "reason": "", "source": _source(event)}

    async def _payload(self, record: dict, event: dict) -> Optional[bytes]:
        """The event as UTF-8 JSON, if the queue can take it; refused (and not sent) otherwise."""
        try:
            payload = (await asyncio.to_thread(json.dumps, event, ensure_ascii=False)).encode("utf-8")
        except (TypeError, ValueError) as exc:
            await self._refuse(record, f"event {type(exc).__name__}")
            return None
        active = [item for item in self._ops.values() if item.get("state") in _ACTIVE]
        if len(active) >= MAX_ACTIVE or sum(int(item.get("payload_bytes") or 0) for item in active) \
                + len(payload) > MAX_ACTIVE_BYTES:
            await self._refuse(record, "full")
            return None
        record["payload_bytes"] = len(payload)
        return payload

    async def _admit(self, op: _Operation) -> None:
        """Save ``op``'s event before its first request; the first record write follows in ``_send``."""
        record = self._new_record(op.topic, op.event, op.seq, "sending")
        payload = await self._payload(record, op.event)
        if payload is None:
            op.stopped = True
            raise OutboxStopped()
        try:
            await asyncio.to_thread(self._write_payload, record, payload)
        except OSError as exc:
            await self._refuse(record, f"journal {type(exc).__name__}")
            op.stopped = True
            raise OutboxStopped() from None
        op.record = record
        self._ops[record["id"]] = record
        older = [item for item in self._ops.values()
                 if item.get("state") in _ACTIVE and int(item.get("seq") or 0) < op.seq]
        if older and not self._sender_down:  # an earlier message waits: this one keeps its place behind it
            record["state"] = "pending"
            if not await self._save(op):
                await self._unsaved(op)
            raise OutboxDeferred()

    async def _unsaved(self, op: _Operation) -> None:
        """A state that could not be written: a saved operation waits to retry; one never saved is not sent."""
        record = op.record
        if record["id"] in self._written:
            delay = float(record.get("retry_delay") or TELEGRAM_RETRY_INITIAL_SEC)
            record.update(state="retry_wait", next_at=self.clock() + delay, reason=f"journal {self._fault}",
                          retry_delay=next_telegram_retry_delay(delay))
            self._refresh()
            raise OutboxDeferred()
        self._ops.pop(record["id"], None)
        await self._refuse(record, f"journal {self._fault}")
        op.stopped = True
        raise OutboxStopped()

    async def _refuse(self, record: dict, reason: str) -> None:
        """Fail closed: an event the journal cannot hold is not sent; the Connection form counts it."""
        self._not_sent += 1
        if reason.startswith("journal"):
            self._fault = reason.split(" ", 1)[1] if " " in reason else reason
        for path in (self._payload_path(record), self._record_path(record)):
            try:
                await asyncio.to_thread(_unlink, path)
            except OSError:
                pass
        self._log("error", f"Telegram {record.get('topic')} was not sent: the outbox could not hold it ({reason}).")
        self._refresh()

    async def _defer(self, op: _Operation, delay: Optional[float], cause: str) -> None:
        """Pause ``op`` until it is due again (its last saved state already names the part ``sending``)."""
        record = op.record
        if delay is None:
            delay = float(record.get("retry_delay") or TELEGRAM_RETRY_INITIAL_SEC)
            record["retry_delay"] = next_telegram_retry_delay(delay)
        record.update(state="retry_wait", next_at=self.clock() + delay, reason=cause)
        await self._save(op)
        waiting = sum(1 for item in self._ops.values() if item.get("state") in _ACTIVE)
        if cause != self._waiting_cause:
            level = "info" if cause == "token_changed" else "warning"
            self._log(level, f"Telegram delivery waits ({cause}): {waiting} message(s) queued, next try in {delay:.0f} s.")
            self._waiting_cause = cause
        raise OutboxDeferred()

    async def _stop(self, op: _Operation, reason: str, *, raise_: bool = True) -> None:
        record = op.record
        for item in record["parts"]:  # what still waited is not sent
            if item.get("state") in _ACTIVE:
                item.update(state="failed", reason=reason)
        record.update(state="failed", reason=reason)
        op.stopped = True
        await self._save(op)
        if reason != "token_changed":
            self._log("warning", f"Telegram {op.topic}: stopped ({reason}); its remaining parts are not sent.")
        if raise_:
            raise OutboxStopped()

    async def _save(self, op: _Operation) -> bool:
        """Write the record; False when it could not be written (the caller decides; nothing is removed)."""
        record = op.record
        record["updated_at"] = self.clock()
        try:
            content = json.dumps(record, ensure_ascii=False).encode("utf-8")
            await asyncio.to_thread(_write_private, self._record_path(record), content)
        except (OSError, TypeError, ValueError) as exc:
            if self._fault != type(exc).__name__:
                self._log("error", f"Telegram outbox could not save {op.topic} ({type(exc).__name__}); "
                                   "nothing is sent without a saved record.")
            self._fault = type(exc).__name__
            self._refresh()
            return False
        self._written.add(record["id"])
        self._fault = ""
        self._refresh()
        return True

    async def _finish(self, record: dict) -> None:
        state = record.get("state")
        record["finished_at"] = self.clock()
        try:
            if state == "confirmed":
                self._ops.pop(record["id"], None)
                self._delivered += 1
                await asyncio.to_thread(self._remove, record)
            else:
                await asyncio.to_thread(_unlink, self._payload_path(record))
                await asyncio.to_thread(_write_private, self._record_path(record),
                                        json.dumps(record, ensure_ascii=False).encode("utf-8"))
                finished = sorted((item for item in self._ops.values() if item.get("state") not in _ACTIVE),
                                  key=lambda item: int(item.get("seq") or 0))
                for stale in finished[:-MAX_FINISHED]:
                    self._ops.pop(stale["id"], None)
                    await asyncio.to_thread(self._remove, stale)
        except (OSError, TypeError, ValueError) as exc:
            # Safe to leave: a part is sent only after its sending state was saved, so a stale record can only
            # replay what was answered or name an interrupted part unknown, never send a delivered part again.
            self._fault = type(exc).__name__
            self._log("warning", f"Telegram outbox files could not be updated ({type(exc).__name__}).")
        if self._waiting_cause and not any(item.get("state") in _ACTIVE for item in self._ops.values()):
            if state == "confirmed":
                self._log("info", f"Telegram delivery resumed after {self._waiting_cause}.")
            self._waiting_cause = ""
        self._refresh()

    async def _discard(self, record: dict) -> None:
        """A saved event its handler no longer sends anything for: removed, counted nowhere."""
        self._ops.pop(record["id"], None)
        try:
            await asyncio.to_thread(self._remove, record)
        except OSError as exc:
            self._fault = type(exc).__name__  # a stale record of it replays to the same nothing
        self._refresh()

    def _load(self) -> None:
        """Read the journal once per load: one short synchronous read, so no two callers ever race on it."""
        if self._loaded:
            return
        for record in self._read_all():
            self._ops[record["id"]] = record
            self._written.add(record["id"])
            self._seq = max(self._seq, int(record.get("seq") or 0))
        self._loaded = True
        self._refresh()

    async def _recover(self) -> None:
        """At the sender's start, holding the queue's lock (so no run is live): settle any run a stop
        interrupted. A part caught ``sending`` may have been delivered."""
        self._load()
        for record in [item for item in self._ops.values() if item.get("state") == "sending"
                       or any(part.get("state") == "sending" for part in item.get("parts") or ())]:
            interrupted = [part for part in record["parts"] if part.get("state") == "sending"]
            for part in interrupted:
                part.update(state="unknown", reason="interrupted")
            if record.get("state") == "sending":
                record["state"] = "pending"
            if interrupted:
                self._log("warning", f"Telegram {record.get('topic')}: a part was in flight at the last stop; "
                                     "its delivery is unknown and it is not sent again.")
                try:
                    await asyncio.to_thread(_write_private, self._record_path(record),
                                            json.dumps(record, ensure_ascii=False).encode("utf-8"))
                except OSError:
                    pass  # the record on disk still says sending: the next load decides the same
        self._refresh()

    def _read_all(self) -> list:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for leftover in self.root.glob(".*.tmp"):  # a write a stop interrupted: its record was never replaced
            leftover.unlink(missing_ok=True)
        records = []
        for path in sorted(self.root.glob("*.json")):
            if path.name.endswith(".event.json"):
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._log("warning", f"Telegram outbox record {path.name} is unreadable; it is left as it is.")
                continue
            if isinstance(record, dict) and record.get("schema") == SCHEMA and isinstance(record.get("parts"), list) \
                    and path.name == f"{record.get('id')}.json":
                records.append(record)
        return sorted(records, key=lambda item: int(item.get("seq") or 0))

    def _write_payload(self, record: dict, payload: bytes) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_private(self._payload_path(record), payload)

    def _remove(self, record: dict) -> None:
        for path in (self._payload_path(record), self._record_path(record)):
            _unlink(path)

    def _record_path(self, record: dict) -> pathlib.Path:
        return self.root / f"{record['id']}.json"

    def _payload_path(self, record: dict) -> pathlib.Path:
        return self.root / f"{record['id']}.event.json"

    # --- the queue ---

    def _active(self) -> bool:
        return any(item.get("state") in _ACTIVE for item in self._ops.values())

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _next(self) -> tuple:
        """The oldest waiting operation if it is due, else how long until it is (None: until woken)."""
        active = sorted((item for item in self._ops.values() if item.get("state") in _ACTIVE),
                        key=lambda item: int(item.get("seq") or 0))
        if not active or active[0].get("state") == "sending" \
                or (self._reserved and min(self._reserved) < int(active[0].get("seq") or 0)):
            return None, None  # nothing waits, a run holds it, or an earlier event is still being saved
        head = active[0]
        wait = float(head.get("next_at") or 0) - self.clock()
        if wait > 0 and self.clock() - float(head.get("created_at") or 0) <= MAX_AGE_SEC:
            return None, wait
        return head, 0.0

    async def _sleep(self, wait: Optional[float]) -> None:
        self._bind()
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=wait)
        except asyncio.TimeoutError:
            pass
        self._wake.clear()

    def _refresh(self) -> None:
        active = [item for item in self._ops.values() if item.get("state") in _ACTIVE]
        finished = [item for item in self._ops.values() if item.get("state") not in _ACTIVE]
        now = self.clock()
        head = min(active, key=lambda item: int(item.get("seq") or 0)) if active else None
        summary: Dict[str, Any] = {"loaded": self._loaded, "waiting": len(active), "delivered": self._delivered,
                                   "fault": self._fault, "not_sent": self._not_sent}
        if head is not None:
            summary.update(oldest_sec=max(0, int(now - float(head.get("created_at") or now))),
                           retry_in_sec=max(0, int(float(head.get("next_at") or now) - now)),
                           cause=str(head.get("reason") or ""))
        for state in ("unknown", "refused", "failed", "superseded"):
            summary[state] = sum(1 for item in finished if item.get("state") == state)
        self._summary = summary

    def _log(self, level: str, message: str) -> None:
        self.api.log(level, message)


def _source(event: dict) -> Dict[str, str]:
    """The event's identities for diagnostics; the words stay in its saved event."""
    source = {key: str(event[key]) for key in _SOURCE_KEYS if event.get(key)}
    reply = event.get("reply_to")
    if isinstance(reply, dict) and reply.get("address"):
        source["reply_to"] = str(reply["address"])
    return source


_OUTBOXES: "weakref.WeakKeyDictionary[object, Outbox]" = weakref.WeakKeyDictionary()
_PINNED: Dict[int, tuple] = {}


def outbox_for(api) -> Outbox:
    """The outbox of this PluginAPI instance: one per loaded generation of the skill, like its token binding."""
    try:
        outbox = _OUTBOXES.get(api)
        if outbox is None:
            outbox = _OUTBOXES[api] = Outbox(api)
        return outbox
    except TypeError:
        pinned = _PINNED.get(id(api))
        if pinned is None or pinned[0] is not api:
            pinned = _PINNED[id(api)] = (api, Outbox(api))
        return pinned[1]


__all__ = ["MAX_AGE_SEC", "Outbox", "OutboxDeferred", "OutboxStopped", "outbox_for"]

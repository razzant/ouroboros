"""Presence outbox delivery helpers (messages, media, albums, edits, moderation and reactions)."""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence, Tuple

from .custody import CALL_STARTED, CustodyStore, OutboxLease, chat_forum, recorded_calls
from .delivery import delivery_report, names_message, reaction_receipt
from .formatting import prepare_caption, prepare_text
from .host import PlaceBindingRevoked
from .transport import (
    MEDIA_METHODS,
    PresenceBotClient,
    TelegramApiError,
    TelegramRequestNotSent,
    _split_text,
)

MAX_OUTBOX_ATTEMPTS = 5
REACTION_METHOD = "setMessageReaction"
REACTION_REFUSALS = frozenset({400, 403})
REACTION_NOT_DONE = frozenset({401, 404, 429})
REACTION_MARKER = "_provider_call_started"
NOT_PERFORMED_CODES = frozenset({400, 401, 403, 404, 429})


class PlaceHeld(RuntimeError):
    """Unsent custody waits for the current place policy, without a provider failure."""


class AlbumIncomplete(TelegramApiError):
    """Telegram accepted an album (ok) but its answer does not name one message per item."""


class UnconfirmedSend(RuntimeError):
    """Telegram answered ok to a send call without naming the message it created."""

    def __init__(self, method: str):
        super().__init__(f"Telegram {method} answered ok without the message it created (no message_id)")


class LeaseLost(RuntimeError):
    """This attempt no longer holds its outbox row because another claim took it."""

    def __init__(self, message: str, message_ids: Sequence[int] = ()):
        super().__init__(message)
        self.message_ids = list(message_ids)


def is_reaction(payload: Dict[str, Any]) -> bool:
    return payload.get("kind") == "operation" and payload.get("method") == REACTION_METHOD


def not_performed(exc: BaseException) -> bool:
    """Telegram's own answer that the call did nothing, or a pre-send check/request proved never to have left."""
    return isinstance(exc, (TelegramRequestNotSent, PlaceHeld, PlaceBindingRevoked)) or (
        isinstance(exc, TelegramApiError) and exc.error_code in NOT_PERFORMED_CODES
    )


def retry_delay(attempts: int) -> float:
    return min(300.0, 2.0 ** min(max(1, attempts), 8))


def format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def send_method(payload: Dict[str, Any]) -> str:
    kind = str(payload.get("kind") or "message")
    return "sendMessage" if kind == "message" else "sendMediaGroup" if kind == "album" else MEDIA_METHODS.get(kind, "")


def album_incomplete(returned: int, named: int, items: int) -> AlbumIncomplete:
    unnamed = f" and {returned - named} entries without a message_id" if returned > named else ""
    return AlbumIncomplete(
        "sendMediaGroup",
        f"Telegram accepted the album and returned {named} messages{unnamed} for "
        f"{items} items; completeness is unknown, and it is not sent again",
    )


def recorded_end(payload: Dict[str, Any]) -> Tuple[str, Any]:
    """What a send row's stored answers already decide, without contacting Telegram."""
    kind = str(payload.get("kind") or "message")
    if kind == "message":
        messages = payload.get("_sent_messages")
        if messages is None:
            return "", None
        if not isinstance(messages, list) or not all(names_message(message) for message in messages):
            return "uncertain", format_error(UnconfirmedSend("sendMessage"))
        chunks = payload.get("_rendered_chunks")
        count = len(chunks) if isinstance(chunks, list) else len(_split_text(str(payload.get("text") or ""), 4000))
        return ("delivered", {"kind": kind, "messages": messages}) if messages and len(messages) >= count else ("", None)
    if kind == "moderation":
        if "_moderation_result" in payload:
            return "delivered", {"kind": kind, "action": payload.get("action"), "result": payload["_moderation_result"]}
        return "", None
    if kind == "operation":
        if "_operation_result" in payload:
            return "delivered", {"kind": kind, "method": payload.get("method"), "result": payload["_operation_result"]}
        return "", None
    if "_sent_media" not in payload:
        return "", None
    stored = payload["_sent_media"]
    if kind == "album":
        items = payload.get("_rendered_items") or payload.get("items") or []
        returned = stored if isinstance(stored, list) else []
        named = [message for message in returned if names_message(message)]
        if len(returned) == len(named) == len(items):
            return "delivered", {"kind": kind, "messages": named}
        return "uncertain", format_error(album_incomplete(len(returned), len(named), len(items)))
    if names_message(stored):
        return "delivered", {"kind": kind, "message": stored}
    return "uncertain", format_error(UnconfirmedSend(send_method(payload) or "send"))


def own_message_row(
    account_id: str, chat_id: str, topic_id: Any, result: Dict[str, Any], words: str, *, album_origin_known: int = 0
) -> Optional[Tuple]:
    message_id = result.get("message_id")
    if not account_id or type(message_id) is not int:
        return None
    chat = result.get("chat") if isinstance(result.get("chat"), dict) else {}
    return (
        account_id,
        chat_id,
        message_id,
        str(topic_id or ""),
        chat_forum(chat.get("type"), chat.get("is_forum"), topic_id),
        account_id,
        "",
        1,
        str(result.get("text") or result.get("caption") or words),
        str(result.get("media_group_id") or ""),
        int(bool(album_origin_known)),
    )


class ProviderCalls:
    """Fenced provider calls for one send attempt, persisting call-started markers first."""

    def __init__(self, store: CustodyStore, lease: OutboxLease):
        self.store, self.lease = store, lease
        self.recorded = recorded_calls(lease.payload)
        self.fence: int = lease.attempts
        self.lease_token: str = lease.lease_token
        self.open: Optional[Dict[str, Any]] = None

    def checkpoint(
        self, payload: Dict[str, Any], *, report=None, sent=None, confirmed: Sequence[int] = ()
    ) -> None:
        stored = self.store.checkpoint_outbox(
            self.lease.delivery_id, payload, report=report, sent=sent,
            attempts=self.fence, lease_token=self.lease_token or None,
        )
        if not stored:
            raise LeaseLost("this attempt no longer holds the outbox row; its record was not stored", confirmed)

    async def make(
        self, payload: Dict[str, Any], part: int, method: str, call: Callable[[], Awaitable[Any]]
    ) -> Any:
        if not self.recorded:
            if not self.store.outbox_lease_valid(self.lease.delivery_id, self.fence, self.lease_token or None):
                raise LeaseLost("this attempt no longer holds the outbox row; nothing was sent")
            return await call()
        marker = {"part": part, "method": method}
        if not self.store.start_outbox_call(
            self.lease.delivery_id, self.lease.attempts, {**payload, CALL_STARTED: marker},
            lease_token=self.lease_token or None,
        ):
            raise LeaseLost("this attempt no longer holds the outbox row; nothing was sent")
        self.open = marker
        try:
            return await call()
        except Exception as exc:
            if not_performed(exc):
                self.checkpoint(payload)
                self.open = None
            raise

    def answered(self) -> None:
        self.open = None


def _with_opt(fn: Any, **kwargs: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import inspect
        params = inspect.signature(fn).parameters
        for k, v in kwargs.items():
            if v is not None and k in params:
                out[k] = v
    except (TypeError, ValueError):
        pass
    return out


async def deliver_outbox(
    client: PresenceBotClient,
    lease: OutboxLease,
    store: CustodyStore,
    *,
    before_send=None,
    account_id: str = "",
    delivered: Callable[[str, Any], None] = lambda _chat, _topic: None,
    calls: Optional[ProviderCalls] = None,
    verified_fds: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    calls = calls or ProviderCalls(store, lease)
    payload = lease.payload
    kind = str(payload.get("kind") or "message")
    chat_id = str(payload.get("chat_id") or "").strip()
    if not chat_id:
        raise ValueError("outbox chat_id is required")
    topic_id = optional_int(payload.get("topic_id"))
    reply_id = optional_int(payload.get("reply_to_message_id"))
    if kind == "message":
        messages = list(payload.get("_sent_messages") or [])
        chunks = payload.get("_rendered_chunks")
        if chunks is None:
            chunks = (
                [{"text": chunk, "parse_mode": ""} for chunk in _split_text(str(payload.get("text") or ""), 4000)]
                if messages
                else prepare_text(str(payload.get("text") or ""), markdown=payload.get("markdown", False))
            )
            payload = {**payload, "_rendered_chunks": chunks}
            calls.checkpoint(payload)
        for index in range(len(messages), len(chunks)):
            chunk = chunks[index]
            if before_send:
                await before_send()
            current = {**payload, "_sent_messages": messages} if messages or "_sent_messages" in payload else payload
            result = await calls.make(
                current,
                index,
                "sendMessage",
                lambda ch=chunk, idx=index: client.send_text_chunk(
                    chat_id,
                    ch["text"],
                    parse_mode=ch["parse_mode"],
                    topic_id=topic_id,
                    reply_to_message_id=reply_id if idx == 0 else None,
                    **_with_opt(client.send_text_chunk, before_send=before_send),
                ),
            )
            if not names_message(result):
                raise UnconfirmedSend("sendMessage")
            report = delivery_report(
                lease.delivery_id,
                payload,
                part_id=index,
                state="delivered",
                receipt=result,
                text=chunk["text"],
                fmt="html" if chunk["parse_mode"] == "HTML" else "plain",
            )
            messages.append({key: value for key, value in result.items() if key != "_delivery"})
            calls.checkpoint(
                {**payload, "_sent_messages": messages},
                report=report,
                sent=own_message_row(account_id, chat_id, topic_id, result, chunk["text"]),
                confirmed=[result["message_id"]],
            )
            calls.answered()
            delivered(chat_id, topic_id)
        return {"kind": kind, "messages": messages}
    if kind == "moderation":
        if "_moderation_result" in payload:
            return {"kind": kind, "action": payload["action"], "result": payload["_moderation_result"]}
        if before_send:
            await before_send()
        result = await calls.make(
            payload, 0, f"moderate:{payload['action']}",
            lambda: client.moderate(str(payload["action"]), dict(payload["parameters"])),
        )
        calls.checkpoint({**payload, "_moderation_result": result})
        calls.answered()
        return {"kind": kind, "action": payload["action"], "result": result}
    if kind == "operation":
        method = str(payload.get("method") or "")
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("Telegram operation parameters are required")
        if "_operation_result" in payload:
            return {"kind": kind, "method": method, "result": payload["_operation_result"]}
        if before_send:
            await before_send()
        result = await calls.make(payload, 0, method, lambda: client.operation(method, dict(parameters)))
        report = delivery_report(
            lease.delivery_id,
            payload,
            part_id="0",
            state="delivered",
            receipt=result,
            text=str(payload.get("text") or ""),
            fmt="plain",
        )
        calls.checkpoint({**payload, "_operation_result": result}, report=report)
        calls.answered()
        return {"kind": kind, "method": method, "result": result}
    if kind == "album":
        return await deliver_album(
            client,
            lease,
            store,
            before_send=before_send,
            account_id=account_id,
            delivered=delivered,
            calls=calls,
            verified_fds=verified_fds,
        )
    if kind not in MEDIA_METHODS:
        raise ValueError(f"unsupported Telegram outbox kind: {kind}")
    caption = payload.get("_rendered_caption")
    if caption is None:
        caption = prepare_caption(str(payload.get("caption") or ""), markdown=payload.get("markdown", False))
        payload = {**payload, "_rendered_caption": caption}
        calls.checkpoint(payload)
    if before_send:
        await before_send()
    result = await calls.make(
        payload,
        0,
        MEDIA_METHODS[kind],
        lambda: client.send_media(
            kind,
            chat_id,
            pathlib.Path(str(payload.get("file_path") or "")),
            prepared_caption=caption,
            topic_id=topic_id,
            reply_to_message_id=reply_id,
            **_with_opt(client.send_media, before_send=before_send, verified_fds=verified_fds),
        ),
    )
    if not names_message(result):
        raise UnconfirmedSend(MEDIA_METHODS[kind])
    report = delivery_report(
        lease.delivery_id,
        payload,
        part_id=0,
        state="delivered",
        receipt=result,
        text=caption["text"],
        fmt="html" if caption["parse_mode"] == "HTML" else "plain",
    )
    result = {key: value for key, value in result.items() if key != "_delivery"}
    calls.checkpoint(
        {**payload, "_sent_media": result},
        report=report,
        sent=own_message_row(account_id, chat_id, topic_id, result, caption["text"]),
        confirmed=[result["message_id"]],
    )
    calls.answered()
    delivered(chat_id, topic_id)
    return {"kind": kind, "message": result}


async def deliver_album(
    client: PresenceBotClient,
    lease: OutboxLease,
    store: CustodyStore,
    *,
    before_send=None,
    account_id: str = "",
    delivered: Callable[[str, Any], None] = lambda _chat, _topic: None,
    calls: Optional[ProviderCalls] = None,
    verified_fds: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    calls = calls or ProviderCalls(store, lease)
    payload = lease.payload
    chat_id = str(payload.get("chat_id") or "").strip()
    topic_id = optional_int(payload.get("topic_id"))
    items = payload.get("_rendered_items")
    if items is None:
        items = [
            {
                "kind": item["kind"],
                "file_path": item["file_path"],
                "caption": prepare_caption(str(item.get("caption") or ""), markdown=payload.get("markdown", False)),
            }
            for item in payload.get("items") or []
        ]
        payload = {**payload, "_rendered_items": items}
        calls.checkpoint(payload)
    if before_send:
        await before_send()
    results = await calls.make(
        payload,
        0,
        "sendMediaGroup",
        lambda: client.send_media_group(
            chat_id, items, topic_id=topic_id, reply_to_message_id=optional_int(payload.get("reply_to_message_id")),
            **_with_opt(client.send_media_group, before_send=before_send, verified_fds=verified_fds),
        ),
    )
    messages, reports, rows = [], [], []
    for index, result in enumerate(results):
        if not names_message(result):
            continue
        item = (
            items[index]
            if index < len(items)
            else {"kind": next((kind for kind in MEDIA_METHODS if kind in result), "unknown"), "file_path": ""}
        )
        wire = result.get("_delivery") or {}
        reports.append(
            delivery_report(
                lease.delivery_id,
                {**payload, "kind": item["kind"], "file_path": item["file_path"]},
                part_id=index,
                state="delivered",
                receipt=result,
                text=wire.get("text", ""),
                fmt=wire.get("format", "plain"),
            )
        )
        message = {key: value for key, value in result.items() if key != "_delivery"}
        messages.append(message)
        rows.append(own_message_row(
            account_id, chat_id, topic_id, message, wire.get("text", ""),
            album_origin_known=1 if index == 0 else 0,
        ))
    complete = len(results) == len(messages) == len(items)
    calls.checkpoint(
        {**payload, "_sent_media": messages, **({} if complete else {CALL_STARTED: calls.open})},
        report=reports,
        sent=rows,
        confirmed=[message["message_id"] for message in messages],
    )
    delivered(chat_id, topic_id)
    if not complete:
        raise album_incomplete(len(results), len(messages), len(items))
    calls.answered()
    return {"kind": "album", "messages": messages}


class ReactionDeliveryHelper:
    """At-most-once execution and durable settlement for ``setMessageReaction`` outbox rows."""

    def __init__(
        self,
        store: CustodyStore,
        *,
        ready_client: Callable[[], Awaitable[Optional[PresenceBotClient]]],
        resolved_reporting: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        check_place: Callable[[Dict[str, Any], str], Awaitable[None]],
        note_error: Callable[[str], None],
        check_account: Callable[[Dict[str, Any]], None] = lambda _payload: None,
        current_gen: Callable[[], int] = lambda: 0,
    ):
        self.store = store
        self._ready_client = ready_client
        self._resolved_reporting = resolved_reporting
        self._check_place = check_place
        self._note_error = note_error
        self._check_account = check_account
        self._current_gen = current_gen
        self._unrecorded: Dict[str, Dict[str, Any]] = {}

    def _fence(self, lease: OutboxLease) -> Dict[str, Any]:
        return {"attempts": lease.attempts, "lease_token": lease.lease_token or None}

    async def process(self, lease: OutboxLease) -> None:
        known = dict(self._unrecorded.pop(lease.delivery_id, None) or lease.payload)
        fence = self._fence(lease)
        if "_operation_result" in known:
            self._finish(lease, known)
            return
        if lease.previous_unanswered:
            known[REACTION_MARKER] = True
        if known.get(REACTION_MARKER):
            await self._settle_unanswered(lease, known)
            return
        client, gen = None, self._current_gen()
        try:
            client = await self._ready_client()
            if client is None:
                raise PlaceHeld("Held: waiting for validated Telegram bot token")
            gen = self._current_gen()
            self._check_account(known)
            known["_reporting"] = await self._resolved_reporting(known)
            if not self.store.checkpoint_outbox(lease.delivery_id, known, **fence):
                raise LeaseLost("reaction attempt lost its outbox lease before sending")
            parameters = known.get("parameters")
            if not str(known.get("chat_id") or "").strip() or not isinstance(parameters, dict):
                raise ValueError("Telegram reaction chat_id and parameters are required")
            await self._check_place(known, str(known.get("topic_id") or ""))
            self._check_account(known)
            if self._current_gen() != gen or not getattr(client, "is_current", lambda: True)():
                raise PlaceHeld("Held: Telegram bot identity or server binding changed")
            if not self.store.checkpoint_outbox(lease.delivery_id, {**known, REACTION_MARKER: True}, **fence):
                raise LeaseLost("reaction attempt lost its outbox lease right before provider call")
            known[REACTION_MARKER] = True
            result = await client.operation(REACTION_METHOD, dict(parameters))
            if result.get("ok") is False:
                raise TelegramApiError(REACTION_METHOD, "Telegram did not confirm the reaction")
            known["_operation_result"] = result
        except LeaseLost as exc:
            self._note_error(format_error(exc))
            return
        except PlaceHeld as exc:
            self.store.hold_outbox(lease.delivery_id, str(exc), **fence)
        except PlaceBindingRevoked as exc:
            self._settle(lease, known, state="failed", error=str(exc))
        except asyncio.CancelledError:
            self.store.release_outbox(lease.delivery_id, reason="worker_cancelled", retry_after_sec=0, **fence)
            raise
        except Exception as exc:
            if isinstance(exc, TelegramRequestNotSent) and (
                self._current_gen() != gen or (client is not None and not getattr(client, "is_current", lambda: True)())
            ):
                known.pop(REACTION_MARKER, None)
                if self.store.checkpoint_outbox(lease.delivery_id, known, **fence):
                    self.store.hold_outbox(lease.delivery_id, f"Held: {format_error(exc)}", **fence)
                return
            self._note_error(format_error(exc))
            try:
                self._after_error(lease, known, exc)
            except Exception as store_error:
                self._note_error(format_error(store_error))
        else:
            self._finish(lease, known)

    def _after_error(self, lease: OutboxLease, known: Dict[str, Any], exc: Exception) -> None:
        fence = self._fence(lease)
        code = exc.error_code if isinstance(exc, TelegramApiError) else None
        if known.get(REACTION_MARKER):
            if code in REACTION_REFUSALS:
                self._settle(lease, known, state="failed", error=format_error(exc))
                return
            if not isinstance(exc, TelegramRequestNotSent) and code not in REACTION_NOT_DONE:
                self._settle(lease, known, state="uncertain", error=format_error(exc))
                return
            del known[REACTION_MARKER]
            if not self.store.checkpoint_outbox(lease.delivery_id, known, **fence):
                return
        wait = max(retry_delay(lease.attempts), float(getattr(exc, "retry_after", None) or 0.0))
        if code == 429:
            self.store.hold_outbox(lease.delivery_id, f"Rate limited: {format_error(exc)}", retry_after_sec=wait, **fence)
            return
        if lease.attempts >= MAX_OUTBOX_ATTEMPTS:
            self._settle(lease, known, state="failed", error=format_error(exc))
        else:
            self.store.release_outbox(lease.delivery_id, reason=format_error(exc), retry_after_sec=wait, **fence)

    async def _settle_unanswered(self, lease: OutboxLease, known: Dict[str, Any]) -> None:
        try:
            if await self._ready_client() is not None:
                known["_reporting"] = await self._resolved_reporting(known)
        except Exception as exc:
            self._note_error(format_error(exc))
        error = format_error(TelegramApiError(REACTION_METHOD, "send_result_unrecorded"))
        try:
            self._settle(lease, known, state="uncertain", error=error)
        except Exception as exc:
            self._note_error(format_error(exc))

    def _settle(self, lease: OutboxLease, known: Dict[str, Any], *, state: str, error: str) -> None:
        fence = self._fence(lease)
        if not self.store.checkpoint_outbox(lease.delivery_id, known, **fence):
            return
        report = delivery_report(lease.delivery_id, known, part_id="status", state=state, error=error)
        if self.store.mark_outbox_failed(lease.delivery_id, reason=error, report=report, state=state, **fence):
            self._note_error(error)

    def _finish(self, lease: OutboxLease, known: Dict[str, Any]) -> None:
        fence = self._fence(lease)
        try:
            report = delivery_report(
                lease.delivery_id,
                known,
                part_id="0",
                state="delivered",
                receipt=known["_operation_result"],
                text=str(known.get("text") or ""),
                fmt="plain",
            )
            if not self.store.checkpoint_outbox(lease.delivery_id, known, report=report, **fence):
                return
            self.store.mark_delivered(lease.delivery_id, provider_receipt=reaction_receipt(known), **fence)
        except Exception as exc:
            self._note_error(format_error(exc))
            self._unrecorded[lease.delivery_id] = known
            try:
                if self.store.checkpoint_outbox(lease.delivery_id, known, **fence):
                    self.store.release_outbox(
                        lease.delivery_id, reason=format_error(exc), retry_after_sec=retry_delay(lease.attempts), **fence
                    )
            except Exception as store_error:
                self._note_error(format_error(store_error))
            else:
                del self._unrecorded[lease.delivery_id]

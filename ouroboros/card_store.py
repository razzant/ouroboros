"""Living-message cards (``show_card``): one durable record per card.

``state/cards/<card_id>.json`` keeps the card's identity, its current body and
revision, and the owner's presses; app bytes are content-addressed at
``state/cards/apps/<sha256>.html``. Pins ("Keep on Widgets", data, no review) live
in ``state/pinned_cards.json``. Every write is a locked read-modify-write
(``update_json_locked``): the tool writes from a worker process, the decision
ingress and the pin endpoints from the server. Reads never write.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import uuid
from typing import Any, Dict, List, Optional

from ouroboros.utils import read_json_dict, update_json_locked, utc_now_iso, write_bytes_atomic

CARDS_DIR = pathlib.Path("state") / "cards"
PINS_FILE = pathlib.Path("state") / "pinned_cards.json"
CARD_KINDS = ("components", "widget", "app")
MAX_SUBMISSIONS = 200  # per card; the oldest are dropped and counted
MAX_PINS = 50
MAX_APP_BYTES = 512 * 1024
_CARD_ID = re.compile(r"card_[0-9a-f]{16}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BODY_KEYS = ("components", "data", "widget", "app")


class CardStoreError(ValueError):
    """A typed refusal (``code``) the tool and the endpoints word for their reader."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def valid_card_id(card_id: Any) -> bool:
    return isinstance(card_id, str) and _CARD_ID.fullmatch(card_id) is not None


def _card_path(root: Any, card_id: str) -> pathlib.Path:
    if not valid_card_id(card_id):
        raise CardStoreError("card_not_found", f"{card_id!r} is not a card id")
    return pathlib.Path(root) / CARDS_DIR / f"{card_id}.json"


def _update(path: pathlib.Path, mutate: Any) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)  # the sidecar lock lives beside the file
    return update_json_locked(path, mutate)


def load_card(root: Any, card_id: Any) -> Optional[Dict[str, Any]]:
    """The stored card, or None when the id is malformed or nothing is stored."""
    return read_json_dict(_card_path(root, card_id)) if valid_card_id(card_id) else None


def store_app_bytes(root: Any, data: bytes) -> Dict[str, Any]:
    """Keep one content-addressed copy of an app's HTML; returns ``{sha256, size}``."""
    digest = hashlib.sha256(data).hexdigest()
    path = pathlib.Path(root) / CARDS_DIR / "apps" / f"{digest}.html"
    if not path.is_file():
        write_bytes_atomic(path, data)
    return {"sha256": digest, "size": len(data)}


def read_app_bytes(root: Any, sha256: Any) -> Optional[bytes]:
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        return None
    path = pathlib.Path(root) / CARDS_DIR / "apps" / f"{sha256}.html"
    return path.read_bytes() if path.is_file() else None


def write_card(root: Any, *, chat_id: int, task_id: str, title: str, fallback_text: str, kind: str,
               body: Dict[str, Any], replace: str = "", same_room: Any = None) -> Dict[str, Any]:
    """Store a new card, or give an earlier card of the same room a new body (same id, revision + 1).

    ``body`` holds the kind's keys (``components`` and optional ``data``, ``widget`` or
    ``app``). A replaced card keeps its presses; the replacing task becomes its author,
    the task that receives later presses. ``same_room(card)`` says a card stored under
    another chat now lives in ``chat_id`` (its work moved into this Project), never a
    card of an unrelated room.
    """
    now = utc_now_iso()
    if not replace:
        card_id = "card_" + uuid.uuid4().hex[:16]
        record = {"card_id": card_id, "chat_id": chat_id, "task_id": task_id, "created_at": now,
                  "updated_at": now, "title": title, "fallback_text": fallback_text, "kind": kind,
                  **body, "submissions": [], "submissions_dropped": 0, "revision": 1}
        return _update(_card_path(root, card_id), lambda _current: record)

    def mutate(current: Dict[str, Any]) -> Dict[str, Any]:
        if not current:
            raise CardStoreError("CARD_REPLACE_NOT_FOUND", f"card {replace} does not exist")
        if current.get("chat_id") != chat_id and not (same_room and same_room(current)):
            raise CardStoreError("CARD_REPLACE_OTHER_CHAT", f"card {replace} belongs to another chat")
        for key in _BODY_KEYS:
            current.pop(key, None)
        current.update(body, title=title, fallback_text=fallback_text, kind=kind, task_id=task_id,
                       updated_at=now, revision=int(current.get("revision") or 0) + 1)
        return current

    if not valid_card_id(replace):
        raise CardStoreError("CARD_REPLACE_NOT_FOUND", f"{replace!r} is not a card id")
    return _update(_card_path(root, replace), mutate)


def record_publication(root: Any, card_id: str, chat_id: int, *, address: str = "") -> Optional[Dict[str, Any]]:
    """The chat the card's row was written to, and once stored that row's address: a late
    press is forwarded there as the owner's reply to the card."""
    def mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not current or (current.get("publication_chat_id") == chat_id
                           and (not address or current.get("publication_address") == address)):
            return None
        current["publication_chat_id"] = chat_id
        if address:
            current["publication_address"] = address
        return current

    return _update(_card_path(root, card_id), mutate) or None


def find_submission(record: Dict[str, Any], request_id: str) -> Optional[Dict[str, Any]]:
    """The press recorded under ``request_id`` (a retry of it delivers that press again)."""
    return next((row for row in record.get("submissions") or []
                 if isinstance(row, dict) and row.get("request_id") == request_id), None)


def record_submission(root: Any, card_id: str, *, request_id: str, action_id: str, label: str,
                      values: Optional[Dict[str, Any]], once: bool, revision: int) -> Dict[str, Any]:
    """Record one owner press of the card drawn at ``revision``, idempotent on ``request_id``.

    Returns ``{"record", "submission", "duplicate"}``; ``{"record", "stale"}`` (the current
    revision) when the card changed since it was drawn, so the press would name another
    control; or ``{"record", "winner"}`` when a ``once`` action was already pressed under
    another request. A press already recorded under ``request_id`` answers as a duplicate
    whatever the card became since.
    """
    outcome: Dict[str, Any] = {}

    def mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not current:
            raise CardStoreError("card_not_found", f"card {card_id} does not exist")
        submissions = current.setdefault("submissions", [])
        prior = find_submission(current, request_id)
        if prior is not None:
            outcome.update(submission=prior, duplicate=True)
            return None
        if int(current.get("revision") or 1) != revision:
            outcome.update(stale=int(current.get("revision") or 1))
            return None
        winner = next((row for row in submissions if row.get("action_id") == action_id), None) if once else None
        if winner is not None:
            outcome.update(winner=winner)
            return None
        # The author and title the press was made on: a retry after a replace still reaches them.
        submission = {"request_id": request_id, "action_id": action_id, "label": label,
                      **({"values": values} if values is not None else {}), "at": utc_now_iso(), "delivered": "owed",
                      "task_id": str(current.get("task_id") or ""), "title": str(current.get("title") or "")}
        submissions.append(submission)
        dropped = max(0, len(submissions) - MAX_SUBMISSIONS)
        del submissions[:dropped]
        current["submissions_dropped"] = int(current.get("submissions_dropped") or 0) + dropped
        outcome.update(submission=submission, duplicate=False)
        return current

    record = _update(_card_path(root, card_id), mutate)
    return {"record": record, **outcome}


def mark_delivery(root: Any, card_id: str, request_id: str, delivered: str, reason: str = "") -> Dict[str, Any]:
    """Settle one press's delivery: ``delivered`` (mailbox), ``forwarded`` (owner message) or ``owed``.
    ``reason`` ``awaiting_recipient_read``: it waits in a paused, held or queued author's mailbox."""
    def mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = find_submission(current, request_id)
        if row is None or (row.get("delivered"), row.get("delivery_reason", "")) == (delivered, reason):
            return None
        row["delivered"] = delivered
        if reason:
            row["delivery_reason"] = reason
        else:
            row.pop("delivery_reason", None)
        return current

    return _update(_card_path(root, card_id), mutate)


def acknowledge_press(root: Any, card_id: str, request_id: str) -> None:
    """Its author read the press (the words entered its transcript): ``read_at`` stays, so a
    reason settled after this read never brings the wait back."""
    def mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = find_submission(current, request_id)
        if row is None or row.get("read_at"):
            return None
        row["read_at"] = utc_now_iso()
        return current

    if valid_card_id(card_id):
        _update(_card_path(root, card_id), mutate)


def late_press_ref(value: Any) -> Optional[Dict[str, str]]:
    """Typed provenance of a press whose card's author no longer holds it: ``{task_id, card_id, request_id}``."""
    if not isinstance(value, dict):
        return None
    ref = {key: str(value.get(key) or "").strip() for key in ("task_id", "card_id", "request_id")}
    return ref if ref["task_id"] and ref["request_id"] and valid_card_id(ref["card_id"]) else None


def press_wait(row: Dict[str, Any]) -> str:
    """``awaiting_recipient_read`` while a delivered press waits for its author to read it, else ``""``."""
    return "" if row.get("read_at") else str(row.get("delivery_reason") or "")


def submissions_summary(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What a surface shows under a card: each press's action, label, time and delivery, and
    its ``request_id`` (an owed press is retried under it, after a reload too); ``reason_code``
    while it waits for its author to read it (``press_wait``)."""
    return [{"action_id": str(row.get("action_id") or ""), "label": str(row.get("label") or row.get("action_id") or ""),
             "at": str(row.get("at") or ""), "delivery": str(row.get("delivered") or "owed"),
             "request_id": str(row.get("request_id") or ""), **({"reason_code": press_wait(row)} if press_wait(row) else {})}
            for row in record.get("submissions") or [] if isinstance(row, dict)]


def pinned_cards(root: Any) -> List[Dict[str, Any]]:
    """Pinned component cards, oldest pin first: ``{card_id, title, components, data?, revision, pinned_at}``."""
    pins = (read_json_dict(pathlib.Path(root) / PINS_FILE) or {}).get("pins")
    return [dict(row) for row in pins if isinstance(row, dict) and valid_card_id(row.get("card_id"))] \
        if isinstance(pins, list) else []


def pin_card(root: Any, card_id: str) -> Dict[str, Any]:
    """Keep a component card on the Widgets page as a snapshot of its current body."""
    record = load_card(root, card_id)
    if record is None:
        raise CardStoreError("card_not_found", f"card {card_id} does not exist")
    if record.get("kind") != "components":
        raise CardStoreError("card_not_pinnable", "only a component card can be kept on Widgets; "
                             "ask Ouroboros to make an app into a skill" if record.get("kind") == "app"
                             else "a skill's widget already lives on the Widgets page")
    pin = {"card_id": card_id, "title": str(record.get("title") or ""), "components": record.get("components") or [],
           **({"data": record["data"]} if isinstance(record.get("data"), dict) else {}),
           "revision": int(record.get("revision") or 1), "pinned_at": utc_now_iso()}

    def mutate(current: Dict[str, Any]) -> Dict[str, Any]:
        pins = [row for row in current.get("pins") or [] if isinstance(row, dict) and row.get("card_id") != card_id]
        if len(pins) >= MAX_PINS:
            raise CardStoreError("pins_full", f"at most {MAX_PINS} cards can be kept; remove one first")
        return {"pins": [*pins, pin]}

    _update(pathlib.Path(root) / PINS_FILE, mutate)
    return pin


def unpin_card(root: Any, card_id: str) -> bool:
    """Remove a pin; False when the card was not pinned."""
    removed: List[bool] = []

    def mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pins = [row for row in current.get("pins") or [] if isinstance(row, dict)]
        kept = [row for row in pins if row.get("card_id") != card_id]
        if len(kept) == len(pins):
            return None
        removed.append(True)
        return {"pins": kept}

    _update(pathlib.Path(root) / PINS_FILE, mutate)
    return bool(removed)


__all__ = [
    "CARD_KINDS", "CardStoreError", "MAX_APP_BYTES", "MAX_PINS", "MAX_SUBMISSIONS", "acknowledge_press",
    "find_submission", "late_press_ref", "load_card", "mark_delivery", "pin_card", "pinned_cards", "press_wait",
    "read_app_bytes", "record_publication", "record_submission", "store_app_bytes", "submissions_summary",
    "unpin_card", "valid_card_id", "write_card",
]

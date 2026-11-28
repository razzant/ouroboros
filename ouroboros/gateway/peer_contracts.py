"""Wire shapes of letters between linked Ouroboros installations, and their one strict parser.

The letter contract is narrow on purpose: words, an optional title, the sender's letter id, an optional cost
ceiling, the chain of askers and, for an answer to this side's question, ``in_reply_to``. Fields that would
name this installation's files or resources (``workspace_root``, ``project_id``, ``executor_ref``, paths) are
not part of it: any other field is ``unexpected_fields``; a non-empty ``attachments`` is reserved and refused.
"""
from __future__ import annotations

import math
import re
from typing import Any, List, Optional

try:
    from typing import Literal, Required, TypedDict
except ImportError:  # pragma: no cover - Python 3.10 compatibility.
    from typing_extensions import Literal, Required, TypedDict

LETTER_ID_CHARS = r"[A-Za-z0-9_.-]{1,64}"  # one path segment of the letter routes (server_auth admits by it)
# ``.`` and ``..`` are no id: an HTTP client removes them from a path, so their status could never be read back.
LETTER_ID_PATTERN = re.compile(r"(?!\.{1,2}$)" + LETTER_ID_CHARS)
MAX_CHAIN = 16
TEXT_MAX_CHARS, TITLE_MAX_CHARS = 100_000, 200
LETTER_FIELDS = frozenset({"letter_id", "text", "title", "cost_ceiling_usd", "protocol", "chain", "in_reply_to",
                           "attachments", "kind", "reply_to"})
REPLY_FORBIDDEN_FIELDS = ("cost_ceiling_usd", "title", "chain", "reply_to")
MESSAGE_FORBIDDEN_FIELDS = ("cost_ceiling_usd", "title", "chain", "in_reply_to")
LETTER_KINDS = ("work", "message")
# 503s say "not known now": the sender keeps its letter and asks again; only the 4xx codes are answers.
REFUSAL_STATUS = {
    "link_unknown": 401, "peer_route_forbidden": 403, "link_paused": 403, "peer_may_not_start": 403,
    "link_silent": 403, "panic_hold": 403, "install_mismatch": 403, "no_budget": 409, "protocol_unsupported": 400, "letter_id_reused": 409,
    "question_closed": 409, "question_not_found": 404, "unexpected_fields": 400, "attachments_unsupported": 400,
    "invalid_letter": 400, "chain_too_long": 400, "letter_not_found": 404,
    "peer_links_unreadable": 503, "store_unavailable": 503,
    "invite_unknown": 401, "invite_transport_mismatch": 400,
    "channel_session_stale": 409, "invalid_session": 400, "peer_channel_refused": 400, "channel_end_missing": 400,
    "peer_door_unsupported": 501, "peer_door_not_running": 503, "peer_door_dir_invalid": 503,
}

# What GET reads of an accepted letter; a refused one was never recorded, so its refusal is the POST's answer.
LetterState = Literal["accepted_running", "needs_input", "waiting_owner", "done", "failed", "cancelled"]
# Why words cleared for this link are held now (``peer_letters.stamp_hold``): not served, and not absent.
LETTER_HOLDS = ("link_paused", "link_silent", "panic_hold", "rules_changed", "unknown")


class LetterChainEntry(TypedDict, total=False):
    install_id: str
    label: str
    task_id: str


class LetterReplyReference(TypedDict):
    letter_id: str
    question_id: str


class LetterCreateRequest(TypedDict, total=False):
    letter_id: Required[str]
    text: Required[str]
    protocol: Required[int]
    kind: Literal["work", "message"]
    title: str
    cost_ceiling_usd: float
    chain: List[LetterChainEntry]
    in_reply_to: LetterReplyReference
    reply_to: str


class PeerMessageReceipt(TypedDict):
    letter_id: str
    kind: Literal["message"]
    state: Literal["received"]


class LetterAnswerReceipt(TypedDict):
    """What POST answers for an answer letter (``in_reply_to``) once its words are proven in the asking root's mailbox
    as the place event ``msg_id``; a repeat of the same id and bytes gets the same receipt and changes nothing.
    Unproven is a 503 naming the letter id: never a receipt, and never the asking letter's ``LetterStatus``."""
    letter_id: str
    in_reply_to: LetterReplyReference
    delivered: Literal[True]
    msg_id: str


class LetterQuestion(TypedDict, total=False):
    question_id: Required[str]
    text: str  # its words, cleared for this link and servable now; else ``held`` names why they are not shown
    held: str  # a ``LETTER_HOLDS`` code, never beside ``text``
    delivery_state: Literal["reserved"]


class LetterStatus(TypedDict, total=False):
    letter_id: Required[str]
    state: Required[LetterState]
    phase: Literal["queued", "running"]
    question: LetterQuestion
    refusal_code: str
    error: str


class LetterStateView(TypedDict, total=False):
    """``GET /api/letters/{id}/state``: what a sender's Stop reads back, the state alone (``view`` confirms it)."""
    letter_id: Required[str]
    state: Required[LetterState]
    view: Required[Literal["state"]]
    phase: Literal["queued", "running"]


class LetterCost(TypedDict):
    known_usd: Optional[float]
    final: bool


class LetterResult(TypedDict, total=False):
    letter_id: Required[str]
    state: Required[Literal["done"]]
    result: Required[str]  # the answer addressed to the sender (``peer_letters.letter_result``); "" when none was kept
    # Whether such an answer is given, an empty one included: never read from ``result``.
    result_available: Required[bool]
    result_held: str  # beside result_available false: an answer cleared for this link is held now (``LETTER_HOLDS``)
    finished_at: Required[str]
    cost: Required[LetterCost]


class LetterRefusal(TypedDict):
    error: str
    refusal_code: str


class LetterRefused(ValueError):
    """A typed refusal of one letter request: ``code`` (a ``REFUSAL_STATUS`` key) and its HTTP status."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code, self.status = code, REFUSAL_STATUS.get(code, 400)


def _string(value: Any, field: str, limit: int, *, required: bool = True) -> str:
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > limit:
        raise LetterRefused("invalid_letter", f"{field} must be a non-empty string of at most {limit} characters")
    return value


def _letter_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not LETTER_ID_PATTERN.fullmatch(value):
        raise LetterRefused("invalid_letter", f"{field} must match [A-Za-z0-9_.-]{{1,64}} and be neither . nor ..")
    return value


def _chain(value: Any) -> list:
    if not isinstance(value, list):
        raise LetterRefused("invalid_letter", "chain must be a list of entries, oldest first")
    if len(value) > MAX_CHAIN:  # never cut: a cut chain loses its facts and lifts the bound on hops
        raise LetterRefused("chain_too_long", f"a chain carries at most {MAX_CHAIN} asking works")
    entries = []
    for item in value:
        if not isinstance(item, dict) or set(item) - {"install_id", "label", "task_id"}:
            raise LetterRefused("invalid_letter", "a chain entry has install_id, label and task_id only")
        entries.append({key: _string(item.get(key, ""), f"chain.{key}", 200, required=False)
                        for key in ("install_id", "label", "task_id") if key in item})
    return entries


def parse_letter(body: Any) -> dict:
    """The accepted letter, exactly as the wire contract allows it; ``LetterRefused`` names the first problem."""
    if not isinstance(body, dict):
        raise LetterRefused("invalid_letter", "the request body must be a JSON object")
    unexpected = sorted(set(body) - LETTER_FIELDS)
    if unexpected:
        raise LetterRefused("unexpected_fields", "fields outside the letter contract: " + ", ".join(unexpected))
    if body.get("attachments") not in (None, []):
        raise LetterRefused("attachments_unsupported", "attachments are reserved and not accepted yet")
    if body.get("protocol") != 1 or isinstance(body.get("protocol"), bool):
        raise LetterRefused("protocol_unsupported", "protocol 1 is the only one this Ouroboros speaks")
    letter: dict[str, Any] = {"letter_id": _letter_id(body.get("letter_id"), "letter_id"),
                              "text": _string(body.get("text"), "text", TEXT_MAX_CHARS), "protocol": 1}
    kind = body.get("kind", "work")
    if not isinstance(kind, str) or kind not in LETTER_KINDS:
        raise LetterRefused("invalid_letter", "kind must be 'work' or 'message'")
    if kind == "message":
        present = sorted(field for field in MESSAGE_FORBIDDEN_FIELDS if field in body)
        if present:
            raise LetterRefused("unexpected_fields", "a message carries no " + ", ".join(present))
        letter["kind"] = "message"
        if "reply_to" in body and body["reply_to"] is not None:
            letter["reply_to"] = _letter_id(body["reply_to"], "reply_to")
        return letter
    if "reply_to" in body:
        raise LetterRefused("unexpected_fields", "only a message carries reply_to")
    reply = body.get("in_reply_to")
    if reply is not None:
        present = sorted(field for field in REPLY_FORBIDDEN_FIELDS if field in body)
        if present:
            raise LetterRefused("unexpected_fields", "an answer carries no " + ", ".join(present))
        if not isinstance(reply, dict) or set(reply) != {"letter_id", "question_id"}:
            raise LetterRefused("invalid_letter", "in_reply_to is {letter_id, question_id}")
        letter["in_reply_to"] = {"letter_id": _letter_id(reply["letter_id"], "in_reply_to.letter_id"),
                                 "question_id": _string(reply["question_id"], "in_reply_to.question_id", 64)}
        return letter
    if "title" in body:
        letter["title"] = _string(body["title"], "title", TITLE_MAX_CHARS)
    if "cost_ceiling_usd" in body:
        ceiling = body["cost_ceiling_usd"]
        if (isinstance(ceiling, bool) or not isinstance(ceiling, (int, float)) or not math.isfinite(ceiling)
                or ceiling <= 0):
            raise LetterRefused("invalid_letter", "cost_ceiling_usd must be a positive number")
        letter["cost_ceiling_usd"] = float(ceiling)
    if "chain" in body:
        letter["chain"] = _chain(body["chain"])
    return letter


def refusal_body(code: str, message: str = "") -> dict:
    """The one refusal shape: ``error`` and ``refusal_code`` both name the code, ``detail`` says it in words."""
    return {"error": code, "refusal_code": code, **({"detail": message} if message and message != code else {})}

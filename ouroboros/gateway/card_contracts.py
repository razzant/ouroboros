"""Living-message cards (``show_card``): the WS frames, the read API and the pin receipt.

``card`` frames are stored as ``msg_type: "card"`` history rows carrying the same ``card``
identity (``ChatHistoryCard``); the body, revision and presses are read from
``GET /api/cards/{card_id}``. Presses use ``POST /api/decisions`` (``card:`` family,
``gateway/decision_contracts.py``). The JSDoc twin is ``web/modules/client_contract_types.js``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ouroboros.gateway.history_contracts import ChatHistoryCard
from ouroboros.gateway.reference_contracts import ChatReferenceFields

try:  # Python 3.11+
    from typing import Literal, NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, TypedDict  # type: ignore[assignment]


class CardSubmissionSummary(TypedDict):
    """One recorded owner press: which control, its label as drawn, when, and its delivery
    (``delivered`` into the authoring task's mailbox, ``forwarded`` as the owner's message
    after that task ended, ``owed`` while neither has happened). ``request_id`` retries an
    owed press, after a reload too. ``reason_code`` ``awaiting_recipient_read``: delivered
    to a paused, held or queued author that has not read it yet."""

    action_id: str
    label: str
    at: str
    delivery: Literal["owed", "delivered", "forwarded"]
    request_id: str
    reason_code: NotRequired[str]


class CardOutbound(ChatReferenceFields):
    """Outbound living card in Ouroboros's message; its history row is ``msg_type: "card"``
    with the same ``card``. ``fallback_text`` is the card's plain version."""

    type: Literal["card"]
    role: Literal["assistant"]
    card: ChatHistoryCard
    fallback_text: str
    ts: str
    chat_id: NotRequired[int]
    task_id: NotRequired[str]
    project_thread: NotRequired[bool]


class CardStateOutbound(TypedDict):
    """A shown card changed: a new body (``card.revision`` grew) or a press recorded or read. Read
    ``GET /api/cards/{card_id}`` again; it is a signal, never a second card."""

    type: Literal["card_state"]
    card: ChatHistoryCard
    submissions_summary: List[CardSubmissionSummary]
    submissions_dropped: int
    ts: str
    chat_id: NotRequired[int]


class CardWidgetRef(TypedDict):
    """An installed skill's widget shown inline by reference (its tab from ``GET /api/widgets``)."""

    skill: str
    tab_id: str
    input: NotRequired[Dict[str, Any]]


class CardAppRef(TypedDict):
    """A stored self-contained HTML app; ``url`` serves its bytes as ``text/plain`` for a sandboxed ``srcdoc``."""

    sha256: str
    size: int
    url: str


class CardResponse(TypedDict):
    """``GET /api/cards/{card_id}``: the card's current body (exactly one of ``components``,
    ``widget``, ``app``) and the owner's presses, the oldest dropped beyond the store bound."""

    card_id: str
    kind: Literal["components", "widget", "app"]
    title: str
    fallback_text: str
    chat_id: int
    task_id: str
    revision: int
    created_at: str
    updated_at: str
    pinned: bool
    components: NotRequired[List[Dict[str, Any]]]
    data: NotRequired[Dict[str, Any]]
    widget: NotRequired[CardWidgetRef]
    app: NotRequired[CardAppRef]
    submissions_summary: List[CardSubmissionSummary]
    submissions_dropped: int


class CardPinResponse(TypedDict):
    """``POST``/``DELETE /api/cards/{card_id}/pin``: whether the card is kept on the Widgets page."""

    ok: bool
    card_id: str
    pinned: bool
    pinned_at: NotRequired[str]


class UiHold(TypedDict):
    """One start the page did not outlive: an app card (``app:<card_id>``, ``rev`` its bytes' sha256), a skill's
    widget (``widget:<skill>:<tab_id>``, ``rev`` its revision) or page module (``module:<contribution key>``)."""

    key: str
    rev: str


class UiHoldsSurface(TypedDict, total=False):
    """Where the page runs, as raw facts: the desktop shell's bridge, inside the Telegram mini app."""

    pywebview: bool
    telegram: bool


class UiHoldsRequest(TypedDict):
    """``POST /api/ui/holds``: the holds one page found when it opened (web/modules/holds.js), once per load."""

    holds: List[UiHold]
    surface: NotRequired[UiHoldsSurface]


class UiHoldsResponse(TypedDict):
    """How many holds were named in a System row now (``0`` when each was named already or is gone)."""

    ok: bool
    named: int


__all__ = [
    "CardAppRef", "CardOutbound", "CardPinResponse", "CardResponse", "CardStateOutbound",
    "CardSubmissionSummary", "CardWidgetRef", "UiHold", "UiHoldsRequest", "UiHoldsResponse", "UiHoldsSurface",
]

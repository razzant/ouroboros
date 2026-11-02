"""Typed owner decision families; runtime validation stays at their ingresses."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from typing import Literal, NotRequired, TypedDict
except ImportError:  # pragma: no cover - Python 3.10 compatibility.
    from typing_extensions import Literal, NotRequired, TypedDict


class DecisionRequest(TypedDict):
    """POST /api/decisions body — the ONE answer ingress for owner decision
    cards (owner decision 1=A). ``decision_id`` is a composed family id:
    ``quiz:{task_id}:{quiz_id}``, ``routing:{client_message_id}:{routing_token}``,
    ``model_wait:{task_id}:{wait_id}``, ``card:{card_id}:{action_id}`` (a press on a
    living card; ``gateway/card_decision.py``), or the reserved
    ``interaction:{task_id}:{run_id}:{interaction_id}`` family.
    ``request_id`` is the idempotency key; a replayed request returns the
    recorded confirmation instead of acting twice. ``comment`` is the owner's
    optional verbatim remark on quiz/routing decisions.

    Quiz ``action=close`` instead requires a non-empty ``reason`` and conveys
    neither an answer nor approval; ``action=dismiss`` carries neither and only removes
    an accepted late answer of unknown delivery from current questions (never delivered,
    never resent). ``option_index`` is optional for answers: an owner who takes none
    of the offered options answers with a non-empty comment and no index.
    Routing requires the integer — its choice IS the option. Model waits
    instead require revision and action, with the matching action fields;
    persist_role defaults false, and an empty credential profile means Auto.
    A card press carries ``revision`` (the card revision it was drawn from; a newer card
    answers 409 ``card_stale`` with the current ``revision``) and ``values`` only for a
    form action (keys are the form's field names; JSON scalars, text at most 4000
    characters, at most 50).
    Each ingress validates its own closed field set before effects.
    """

    request_id: str
    decision_id: str
    option_index: NotRequired[int]
    comment: NotRequired[str]
    revision: NotRequired[int]
    action: NotRequired[Literal["auto_continue", "switch", "retry", "close", "dismiss"]]
    reason: NotRequired[str]  # explicit quiz closure; never an answer/comment
    auto_continue: NotRequired[bool]
    model: NotRequired[str]
    credential_profile_id: NotRequired[str]
    use_local: NotRequired[bool]
    persist_role: NotRequired[bool]
    values: NotRequired[Dict[str, Any]]  # card: form field values


class DecisionResponse(TypedDict, total=False):
    """Answer-ingress reply. 2xx carries the card's lifecycle state
    (quiz: answered or explicitly closed; duplicate marks an idempotent replay).
    Closure carries actor, request identity, reason and time plus its delivery
    disposition; an owed delivery remains canonical until recovery confirms it.
    A LATE quiz answer
    — the card's task already finished — is also 2xx, adding
    answered_after_terminal and forwarded: the answer is recorded and delivered
    into the card's chat as an ordinary owner message, and forwarded=false with
    a reason_code says when that chat is a machine or hidden destination with no
    owner turn to start. 409 carries the winning answered/closed disposition (and
    for a non-root addressee). Routing adds dispatched
    (confirmed durable receipt), task_id (derived promoted id), latest_status
    (superseding status on 409), reason/detail diagnostics, and cause (the
    owner-facing sentence for a refused routing act). A New task picked from a
    card that carried an explicit start adds reasoning_effort, read from the
    admitted root's own row; a steered existing task never reports one.

    Model waits distinguish accepted (202, applied false) from the worker's
    applied_request_id. wait carries the current projection and revision;
    saved says whether the optional permanent role update landed, with null
    reserved for the existing settings-writer timeout's unknown outcome.
    Applied, saved and a refusal or failure stay separate facts: an error after
    the claim still reports the claim's current row and receipt. ``applied:false``
    is no observed application yet, not proof that a pending action cannot apply. The role saves
    only while the task's stop state reads clear — a recorded Stop intent (Wrap
    up included) or Panic saves nothing, nor does an unreadable (unknown) state.

    A card press is recorded per ``request_id`` (each press its own; ``once``
    actions keep the first and answer 409 ``card_action_once`` with ``winner`` (its request_id,
    time and delivery: a winner still owed is retried under its request_id); a card
    that changed since it was drawn answers 409 ``card_stale`` with its ``revision``).
    ``delivery`` says where it went: ``delivered`` into the authoring task's
    mailbox (``reason_code`` ``awaiting_recipient_read`` while a paused, held or
    queued author has not read it), ``forwarded`` as the owner's message after
    that task ended, or ``owed`` (202 with ``reason_code``; resend the same ``request_id``).
    """

    ok: bool
    decision_id: str
    state: str
    answered_index: int
    comment: str
    closed_reason: str
    closed_at: str
    closed_actor: Literal["owner", "model"]
    closed_request_id: str
    question_delivery: Dict[str, Any]
    duplicate: bool
    answered_after_terminal: bool
    forwarded: bool
    error: str
    dispatched: str
    task_id: str
    latest_status: str
    reason: str
    detail: str
    cause: str
    reasoning_effort: str
    request_id: str
    applied: bool
    saved: Optional[bool]
    wait: Dict[str, Any]
    reason_code: str
    delivery: Literal["delivered", "forwarded", "owed"]
    card_id: str
    revision: int
    winner: Dict[str, Any]


class SecretRequestField(TypedDict):
    """One asked field: the manifest's public id and the Settings key it writes."""

    id: str
    setting_key: str


class SecretRequestMeta(TypedDict):
    """An integration-secret request's metadata on its quiz block and card (no value, ever).

    ``plugin_name`` is the installed skill; ``fields`` come from its reviewed ``secret_fields``
    manifest mapping. A save writes only while the skill still maps these ids to these keys."""

    plugin_name: str
    fields: list[SecretRequestField]


class SecretKeyState(TypedDict):
    """Whether an asked key holds a value in Settings now: a boolean, never a value or prefix."""

    id: str
    setting_key: str
    configured: bool


class SecretConsumer(TypedDict):
    """An installed skill whose ``env_from_settings`` reads an asked key, and what its CURRENT
    grant covers (``granted``); ``unsupported`` for a type that receives no keys, ``grant_unknown``
    when the grant could not be read. Facts for the owner, never a grant."""

    name: str
    enabled: bool
    reads: list[str]
    granted: NotRequired[list[str]]
    unsupported: NotRequired[bool]
    grant_unknown: NotRequired[bool]


class SecretOutcome(TypedDict):
    """The host's typed completion of a secret request: ``secret_filled`` or ``secret_replaced``
    (``replaced_fields`` held a value just before the write) after Settings kept the values, or
    ``secret_cancelled``. ``consumers`` is what the readers' grants covered after the save (saving
    grants nothing), or ``consumers_unknown`` when they could not be read, never an empty list;
    ``late`` marks a save after the asking task ended (nothing was resumed); ``interrupted_save``
    marks a cancel after an earlier attempt of unknown result."""

    event: Literal["secret_filled", "secret_replaced", "secret_cancelled"]
    pluginName: str
    fields: list[str]
    at: str
    settings_keys: list[str]
    replaced_fields: NotRequired[list[str]]
    consumers: NotRequired[list[SecretConsumer]]
    consumers_unknown: NotRequired[bool]
    late: NotRequired[bool]
    interrupted_save: NotRequired[bool]


class SecretEvent(TypedDict):
    """The outcome's public event payload: the plugin and field ids, never a value."""

    pluginName: str
    fields: list[str]


class SecretDecisionRequest(TypedDict):
    """POST /api/decisions/secret body — the browser's ``saveSecretToken``.

    ``decision_id`` is the request's question identity ``quiz:{task_id}:{quiz_id}``. A ``save``
    (the default) carries ``values`` for EXACTLY the asked field ids and ``observed``: per field,
    whether the form showed its key as set (from ``status``); it writes only while Settings still
    says exactly that, so Replace is explicit. ``status`` and ``cancel`` carry neither; ``status``
    needs no ``request_id``. ``request_id`` is reused only for the same values."""

    request_id: NotRequired[str]
    decision_id: str
    action: NotRequired[Literal["save", "cancel", "status"]]
    values: NotRequired[Dict[str, str]]
    observed: NotRequired[Dict[str, bool]]


class SecretDecisionResponse(TypedDict, total=False):
    """Secret form reply; every save and cancel reply and every refusal carries ``saved`` on both sides
    of the Settings commit, while a ``status`` read carries the lifecycle and form facts only.

    200 carries the recorded lifecycle: ``secret_filled``, ``secret_replaced`` or
    ``secret_cancelled`` (plugin and field ids), ``secret_outcome`` and the question delivery; a
    ``status`` read and every refusal on a request that still takes a save add ``keys`` and
    ``consumers`` (or ``keys_unknown``/``consumers_unknown``). A refusal says ``saved: false`` with
    ``reason_code`` (``secret_key_state_changed`` when a key was set or cleared since the form read it,
    ``settings_unreadable`` when Settings exist but cannot be read) and the request's current facts,
    so a losing tab settles on the winner. ``saved: true`` with ``secret_save_unverified``,
    ``secret_not_retained`` or ``secret_completion_unrecorded`` wrote Settings without reporting a
    save; ``saved: null`` with ``secret_save_unconfirmed`` is an attempt of unknown outcome, never
    repeated under its own request id, nor is any earlier one. ``save_unconfirmed`` marks such an
    attempt on the block."""

    ok: bool
    saved: Optional[bool]
    decision_id: str
    state: str
    duplicate: bool
    secret_request: SecretRequestMeta
    secret_filled: SecretEvent
    secret_replaced: SecretEvent
    secret_cancelled: SecretEvent
    secret_outcome: SecretOutcome
    keys: list[SecretKeyState]
    keys_unknown: bool
    consumers: list[SecretConsumer]
    consumers_unknown: bool
    save_unconfirmed: bool
    answered_after_terminal: bool
    closed_reason: str
    closed_at: str
    closed_actor: str
    closed_request_id: str
    closed_by_task_id: str
    question_delivery: Dict[str, Any]
    error: str
    reason_code: str

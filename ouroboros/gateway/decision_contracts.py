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
    ``model_wait:{task_id}:{wait_id}``, or the reserved
    ``interaction:{task_id}:{run_id}:{interaction_id}`` family.
    ``request_id`` is the idempotency key; a replayed request returns the
    recorded confirmation instead of acting twice. ``comment`` is the owner's
    optional verbatim remark on quiz/routing decisions.

    ``option_index`` is optional for the quiz family: an owner who takes none
    of the offered options answers with a non-empty comment and no index.
    Routing requires the integer — its choice IS the option. Model waits
    instead require revision and action, with the matching action fields;
    persist_role defaults false, and an empty credential profile means Auto.
    Each ingress validates its own closed field set before effects.
    """

    request_id: str
    decision_id: str
    option_index: NotRequired[int]
    comment: NotRequired[str]
    revision: NotRequired[int]
    action: NotRequired[Literal["auto_continue", "switch", "retry"]]
    auto_continue: NotRequired[bool]
    model: NotRequired[str]
    credential_profile_id: NotRequired[str]
    use_local: NotRequired[bool]
    persist_role: NotRequired[bool]


class DecisionResponse(TypedDict, total=False):
    """Answer-ingress reply. 2xx carries the card's lifecycle state
    (quiz: answered; duplicate marks an idempotent replay). A LATE quiz answer
    — the card's task already finished — is also 2xx, adding
    answered_after_terminal and forwarded: the answer is recorded and delivered
    into the card's chat as an ordinary owner message, and forwarded=false with
    a reason_code says when that chat is a machine or hidden destination with no
    owner turn to start. 409 is left for a card that was already answered (and
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
    """

    ok: bool
    decision_id: str
    state: str
    answered_index: int
    comment: str
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

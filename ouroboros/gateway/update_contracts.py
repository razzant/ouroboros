"""Core update flow envelopes: ``/api/update/preflight``, ``/api/update/apply`` (and the rollback
refusal that shares its error shape), ``/api/update/description`` and the two update WS frames;
re-exported by ``gateway/contracts.py``.
The JSDoc twin is ``web/modules/update_types.js``, re-exported by ``api_types.js``."""

from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    from typing import Literal, NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, TypedDict  # type: ignore[assignment]


class UpdateMergePlan(TypedDict, total=False):
    """Exact, channel-bound merge plan shared by preflight and apply responses."""

    available: bool
    auto_mergeable: bool
    kind: Literal["clean", "conflicting", "current", "unavailable", "unknown"]
    error: str
    remote: str
    remote_branch: str
    target_ref: str
    update_channel: str
    current_branch: str
    base_sha: str
    target_sha: str
    local_dirty_count: int
    local_snapshot: str
    merge_commit: str
    code_conflict_paths: List[str]
    doc_conflict_paths: List[str]
    hot_code_paths: List[str]
    recommended_strategy: Literal["auto_merge", "assisted"]


class UpdatePreflightRequest(TypedDict, total=False):
    """POST /api/update/preflight has an intentionally empty JSON body."""


class UpdatePreflightResponse(TypedDict):
    merge_plan: UpdateMergePlan
    effects: NotRequired[Dict[str, Any]]  # target-bound engine transition, active_work and interpreter facts
    letter: NotRequired[Optional[Dict[str, Any]]]  # prose of exactly these pins; absent or null means unknown
    qualification: NotRequired[Optional[Dict[str, Any]]]  # declared/observed checks of exactly these pins


class UpdateApplyRequest(TypedDict):
    strategy: Literal["auto_merge", "assisted", "manual", "replace"]
    expected_base_sha: NotRequired[str]
    expected_target_sha: NotRequired[str]
    confirm_recovery: NotRequired[bool]


class UpdateApplySuccessResponse(TypedDict):
    status: Literal["ok", "restart_required", "assisted_started", "manual"]
    restarting: NotRequired[bool]
    strategy: NotRequired[Literal["auto_merge", "assisted", "manual", "replace"]]
    task_id: NotRequired[str]
    merge_plan: NotRequired[UpdateMergePlan]
    error: NotRequired[str]
    keep_branch: NotRequired[str]


class UpdateApplyErrorResponse(TypedDict):
    error: str
    outcome: NotRequired[Literal["not_applied", "rolled_back"]]
    index_lock: NotRequired[Dict[str, Any]]  # effective path, exists, inode and size; never ownership
    reason: NotRequired[str]
    abandoned: NotRequired[bool]
    status: NotRequired[Dict[str, Any]]  # preparation observations, not the success discriminator
    update_intent: NotRequired[Dict[str, Any]]
    rescue: NotRequired[Dict[str, Any]]
    keep_branch: NotRequired[str]
    blockers: NotRequired[List[str]]
    rolled_back: NotRequired[bool]
    rollback: NotRequired[str]
    restart_required: NotRequired[bool]
    merge_plan: NotRequired[UpdateMergePlan]
    smoke: NotRequired[Dict[str, Any]]
    # Stash-first prologue disclosures (additive): how the owner's stashed work
    # was unwound on an aborted update. The wave numbers rode the retired
    # wave-floor refusal (#1487 removed it); the optional keys stay in the shape.
    stash_note: NotRequired[str]
    estimated_wave_usd: NotRequired[Optional[float]]
    remaining_usd: NotRequired[Optional[float]]


class UpdateDescriptionRequest(TypedDict):
    """The owner's Write/Rewrite of the update description for one exact range."""
    mode: Literal["write", "rewrite"]
    base_sha: str
    target_sha: str


class UpdateDescriptionResponse(TypedDict):
    """One ordinary Main task was admitted; its status rides ``letter.request`` in the update status."""
    status: Literal["queued"]
    task_id: str
    key: Dict[str, Any]  # {base_sha, target_sha, update_channel, target_ref}


class UpdateDescriptionErrorResponse(TypedDict):
    """400 malformed; 409 a request still running (``task_id``, ``task_status``) or a range nobody
    offered (``legal_ranges``); 503 record busy or the task not admitted (``reason``)."""
    error: str
    task_id: NotRequired[str]
    task_status: NotRequired[str]
    legal_ranges: NotRequired[List[Dict[str, Any]]]
    reason: NotRequired[str]


class UpdateProgressChangedOutbound(TypedDict):
    """Invalidation only; never proof that boot finalization completed."""
    type: Literal["update_progress_changed"]


class UpdateStatusReadyOutbound(TypedDict):
    type: Literal["update_status_ready"]
    available: bool
    check_ok: Optional[bool]


__all__ = [
    "UpdateMergePlan", "UpdatePreflightRequest", "UpdatePreflightResponse", "UpdateApplyRequest",
    "UpdateApplySuccessResponse", "UpdateApplyErrorResponse", "UpdateDescriptionRequest", "UpdateDescriptionResponse",
    "UpdateDescriptionErrorResponse", "UpdateProgressChangedOutbound", "UpdateStatusReadyOutbound",
]

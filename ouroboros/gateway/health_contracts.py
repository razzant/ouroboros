"""``GET /api/health`` and the boot phases it carries, re-exported by gateway.contracts.

The health answer is 200 as soon as the socket serves; its ``boot`` document (the same object as
``StateResponse.boot``) is what ``server_liveness`` publishes for the serving generation. These
types describe, not validate, JSON.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:  # Python 3.11+
    from typing import Literal, NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, TypedDict  # type: ignore[assignment]


class BootStep(TypedDict):
    """One named boot step of the serving generation (``BootDocument.steps``)."""

    name: str
    state: Literal["running", "done", "failed"]
    started_at: str
    ended_at: Optional[str]
    error: Optional[str]


class BootJob(TypedDict):
    """A named background job's durable facts (``maintenance_jobs``), as the health reads them."""

    state: Optional[str]
    done: Optional[int]
    total: Optional[int]


class SupervisorLoadHeadline(TypedDict):
    """The latest ``supervisor_load`` row's headline (``logs/supervisor.jsonl``)."""

    ts: str
    max_event_lag_sec: Optional[float]


class BootDocument(TypedDict):
    """The ``boot`` object of ``GET /api/health`` and ``GET /api/state``: the generation's phase.

    ``phase``: ``starting`` (socket not serving yet), ``serving`` (boot steps run while requests
    are answered), ``ready``, ``failed`` (the generation cannot become ready; ``error`` says why).
    ``dialogue_ready``: an owner message is accepted durably. ``tasks_ready``: pooled task starts
    are admitted. ``steps``: the boot coordinator's and the supervisor init's named steps.
    ``jobs``: the store's named background jobs (``{}`` while the store is unpublished).
    """

    phase: Literal["starting", "serving", "ready", "failed"]
    since: str
    dialogue_ready: bool
    tasks_ready: bool
    steps: List[BootStep]
    jobs: Dict[str, BootJob]
    error: Optional[str]
    supervisor_load: NotRequired[Optional[SupervisorLoadHeadline]]  # None before this generation's first row


class HealthResponse(TypedDict):
    """Shape of ``GET /api/health``: 200 as soon as the socket serves, during boot too."""

    status: Literal["ok"]
    version: str
    runtime_version: str
    app_version: str
    abi_version: NotRequired[str]  # schema.GATEWAY_ABI_VERSION; absent on cores older than GET /api/contract
    boot: NotRequired[BootDocument]  # absent on cores older than the boot phases


__all__ = ["BootStep", "BootJob", "SupervisorLoadHeadline", "BootDocument", "HealthResponse"]

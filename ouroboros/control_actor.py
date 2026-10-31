"""Call-scoped attribution for the existing Pause, Resume and Continue owners.

No authority is stored here. The supervisor binds the requesting mind for one
control call; the existing fence/grant/CAS records retain who requested it.
A call may also carry the caller's named decision to go on past held calls
whose outcome is unknown (``accepting_unknown``); the addressed action's own
census applies it to its tree and records it (``local_custody_repair``).
"""

from contextlib import contextmanager
from contextvars import ContextVar

_ACTOR = ContextVar("lifecycle_control_actor", default=None)
_ACCEPT_UNKNOWN = ContextVar("lifecycle_accept_unknown", default=None)


def control_actor() -> dict:
    return dict(_ACTOR.get() or {})


@contextmanager
def accepting_unknown(ids, text: str = ""):
    """Bind, for this one call, the named held calls (operation or model attempt ids) to go on past."""
    named = sorted({str(value) for value in ids or () if value})
    token = _ACCEPT_UNKNOWN.set({"ids": named, "text": str(text or "").strip()} if named else None)
    try:
        yield
    finally:
        _ACCEPT_UNKNOWN.reset(token)


def accepted_unknown() -> dict:
    """The bound decision, ``{}`` when the call named none."""
    return dict(_ACCEPT_UNKNOWN.get() or {})


PAUSE_AUTHORS = ("owner", "ouroboros")


def pause_author(fence: dict | None) -> dict:
    """Who declared a Pause and why, as its fence recorded it: ``{actor, text, at}``.

    ``actor`` is ``owner`` when the owner's client declared the press, ``ouroboros`` for a mind's ``pause_task``
    (``text`` is its own reason), else ``""``: a press that declared nothing, or a missing or unreadable fence.
    Attribution is never inferred from the door a press came through. ``at`` is the press time, never the
    ``paused_at`` a settling fence restamps. Fences written before ``by`` existed keep their mind record: the
    top-level ``actor``, else the mind's first press on that same fence.
    """
    fence = fence if isinstance(fence, dict) else {}
    at = str(fence.get("requested_at") or "")
    by = fence.get("by") if isinstance(fence.get("by"), dict) else None
    if by is not None and str(by.get("actor") or "") in PAUSE_AUTHORS:
        return {"actor": str(by["actor"]), "text": str(by.get("text") or "").strip(), "at": str(by.get("at") or at)}
    legacy = [fence] + [entry for entry in (fence.get("requests") or {}).values()
                        if isinstance(entry, dict) and entry.get("fence_id") == fence.get("fence_id")]
    for entry in legacy:
        if entry.get("actor") == "mind":
            return {"actor": "ouroboros", "text": str(entry.get("text") or entry.get("reason") or "").strip(), "at": at}
    return {"actor": "", "text": "", "at": at}


@contextmanager
def mind_control(actor_task_id: str, reason: str):
    from ouroboros.utils import utc_now_iso

    token = _ACTOR.set({"actor": "ouroboros", "task_id": actor_task_id, "text": reason, "at": utc_now_iso()})
    try:
        yield
    finally:
        _ACTOR.reset(token)


def observed_control_stop(root, task=None, *, budget=None, owner=None) -> dict:
    """Feed observed work and tree controls to the pure stop reader."""
    from ouroboros.work_stop import current_stop

    return current_stop(task, {"budget": budget or {}, "owner": owner or {}})

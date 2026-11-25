"""Call-scoped attribution for the existing Pause, Resume and Continue owners.

No authority is stored here. The supervisor binds the requesting mind for one
control call; the existing fence/grant/CAS records retain who requested it.
A call may also carry the caller's named decision to go on past held calls
whose outcome is unknown (``accepting_unknown``); the addressed action's own
census applies it to its tree and records it (``local_custody_repair``); a mind's list call
binds each of its rows as such (``list_row``).
The owner's own Resume door binds itself (``owner_door``) for the grant decisions that
need positive owner provenance, which an unbound call never supplies. The host's
own act unbinds all of these (``host_control``), even when a door's call reaches it.
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


_LIST_ROW = ContextVar("lifecycle_list_row", default="")


@contextmanager
def list_row(call_id: str):
    """Bind, for this one call, that it is one row of a mind's list call (its ``list_call``, a general Resume): a
    choice that must be addressed alone (``delegate_resume.selects``) is never taken there."""
    token = _LIST_ROW.set(str(call_id or ""))
    try:
        yield
    finally:
        _LIST_ROW.reset(token)


def list_row_call() -> str:
    """The list call this control call is one row of; ``""`` for a call addressed alone."""
    return str(_LIST_ROW.get() or "")


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


def creating_author(fence: dict | None) -> dict:
    """``pause_author`` of the press that CREATED this fence generation (``requests[request_id]``), read the same
    way: a later press may fill the top-level ``by`` of a fence that declared nobody, never this one. A fence
    without that entry keeps only its own top-level legacy fields (``actor``, ``text``), never a later ``by``."""
    fence = fence if isinstance(fence, dict) else {}
    entry = (fence.get("requests") or {}).get(str(fence.get("request_id") or ""))
    own = {key: fence[key] for key in ("actor", "text", "reason", "requested_at") if key in fence}
    return pause_author({**own, **{key: value for key, value in (entry if isinstance(entry, dict) else {}).items()
                                   if key not in {"fence_id", "requests"}}})


@contextmanager
def mind_control(actor_task_id: str, reason: str):
    from ouroboros.utils import utc_now_iso

    token = _ACTOR.set({"actor": "ouroboros", "task_id": actor_task_id, "text": reason, "at": utc_now_iso()})
    try:
        yield
    finally:
        _ACTOR.reset(token)


_OWNER_DOOR = ContextVar("lifecycle_owner_door", default=None)


@contextmanager
def owner_door(door: str):
    """Bind, for this one call, the owner's own control door it entered through (``api_task_resume``).

    A mind's doors bind ``mind_control`` and automatic paths bind nothing, or unbind it where a door's call reaches
    them (``host_control``), so neither reads as this door.
    Its readers: the owner's choice of 2026-11-02 (#2566), a moved Presence turn's producer allowance; who asked for a
    Resume (``work_stop``); and who granted one (``resume_author``).
    """
    from ouroboros.utils import utc_now_iso

    token = _OWNER_DOOR.set({"actor": "owner", "door": str(door), "at": utc_now_iso()})
    try:
        yield
    finally:
        _OWNER_DOOR.reset(token)


def owner_door_bound() -> dict:
    """The owner door this call entered through: ``{}`` when none is bound, or a mind's control call is."""
    return {} if control_actor() else dict(_OWNER_DOOR.get() or {})


def resume_author() -> dict:
    """Who grants a Resume, recorded as its grant's ``by``: a mind's control, else the owner's own door, else nobody
    (the host's own act); a cold Resume's notice names it (``model_sleep.resumer_words``)."""
    return control_actor() or owner_door_bound()


@contextmanager
def host_control():
    """Bind, for this one call, the host's own act: an automatic path a door's call can reach
    (``continuation_admission.release_settled_continuations``) runs as the host, not as that door, so neither
    the caller's authorship, the owner's door nor the caller's named decision reads through it."""
    tokens = _ACTOR.set(None), _OWNER_DOOR.set(None), _ACCEPT_UNKNOWN.set(None), _LIST_ROW.set("")
    try:
        yield
    finally:
        _LIST_ROW.reset(tokens[3])
        _ACCEPT_UNKNOWN.reset(tokens[2])
        _OWNER_DOOR.reset(tokens[1])
        _ACTOR.reset(tokens[0])


def observed_control_stop(root, task=None, *, budget=None, owner=None) -> dict:
    """Feed observed work and tree controls to the pure stop reader."""
    from ouroboros.work_stop import current_stop

    return current_stop(task, {"budget": budget or {}, "owner": owner or {}})

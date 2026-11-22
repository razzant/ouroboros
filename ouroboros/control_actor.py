"""Call-scoped attribution for the existing Pause, Resume and Continue owners.

No authority is stored here. The supervisor binds the requesting mind for one
control call; the existing fence/grant/CAS records retain who requested it.
A call may also carry the caller's named decision to go on past held calls
whose outcome is unknown (``accepting_unknown``); the addressed action's own
census applies it to its tree and records it (``local_custody_repair``).
The owner's own Resume door binds itself (``owner_door``) for the grant decisions that
need positive owner provenance, which an unbound call never supplies. The host's
own act unbinds all of these (``host_control``), even when a door's call reaches it.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_ACTOR = ContextVar("lifecycle_control_actor", default=None)
_ACCEPT_UNKNOWN = ContextVar("lifecycle_accept_unknown", default=None)
_OWNER_WORD = ContextVar("lifecycle_owner_word", default=None)


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


def creating_author(fence: dict | None) -> dict:
    """``pause_author`` of the press that CREATED this fence generation (``requests[request_id]``), read the same
    way: a later press may fill the top-level ``by`` of a fence that declared nobody, never this one. A fence
    without that entry keeps only its own top-level legacy fields (``actor``, ``text``), never a later ``by``."""
    fence = fence if isinstance(fence, dict) else {}
    entry = (fence.get("requests") or {}).get(str(fence.get("request_id") or ""))
    own = {key: fence[key] for key in ("actor", "text", "reason", "requested_at") if key in fence}
    return pause_author({**own, **{key: value for key, value in (entry if isinstance(entry, dict) else {}).items()
                                   if key not in {"fence_id", "requests"}}})


def read_panic_window(drive_root=None) -> dict:
    """The last Panic as a mind's release must see it, read once OFF any lock where a mind's control enters:
    ``started`` (``state.panic_started_at``, taken at Panic's ingress and recorded by the boot that takes it) and
    ``blocked``: a Panic input not yet taken by a boot, or a state read from its backup copy. Unreadable: blocked."""
    try:
        from ouroboros.installation_stop import installation_panic
        from supervisor.state import load_state

        state = load_state()
        root = drive_root
        if root is None:
            from supervisor import queue as q
            root = q.DRIVE_ROOT
        read = state.get("_state_read") if isinstance(state.get("_state_read"), dict) else {}
        degraded = read.get("source") == "backup" or read.get("quality") == "unavailable"  # never a stale start
        return {"started": str(state.get("panic_started_at") or ""), "blocked": degraded or installation_panic(root)}
    except Exception:
        return {"started": "", "blocked": True}


@contextmanager
def citing_owner(written_at: str = "", window: dict | None = None):
    """Bind, for this one mind control, when the owner wrote the message it cites (``owner_words``, verified with
    its time by ``owner_words.cited_owner_message``) and the last Panic (``read_panic_window``, read here, off the
    queue lock, unless the caller read it): ``work_stop.release_refusal`` compares them without I/O, and what the
    door selects inside this call inherits both."""
    token = _OWNER_WORD.set({"written": str(written_at or ""), "window": window or read_panic_window()})
    try:
        yield
    finally:
        _OWNER_WORD.reset(token)


def mind_release_refusal(stop: dict) -> dict | None:
    """Emergency Stop Invariant, read 21 Nov 2026: after a Panic, a mind releases nothing the Panic stopped (a Panic
    hold of either class) or that already stood when it began (a stop placed before it; an unknown time counts as
    before) without the owner's own message written after it began. Unbound, a Panic hold refuses and older stops
    are not judged; a blocked window (a Panic not yet taken, a backup state) refuses every release."""
    from ouroboros.work_stop import _instant

    if not stop:
        return None  # no stop at all (the installation gate's empty read): nothing to release
    bound = _OWNER_WORD.get() or {}
    window = bound.get("window") or {}
    panic = stop.get("cause") == "panic" or stop.get("door") == "panic"
    started = _instant(window.get("started") or "")
    if not bound:
        return {"ok": False, "error": "panic_requires_owner_resume"} if panic else None
    if window.get("blocked"):
        return {"ok": False, "error": "panic_requires_owner_resume"}
    since = _instant(next((stop[key] for key in ("held_at", "requested_at", "paused_at", "at") if stop.get(key)), ""))
    if not panic and (started is None or (since is not None and since >= started)):
        return None  # no Panic yet, or a stop placed after it: the door's own rules decide
    written = _instant(bound.get("written") or "")
    return None if started is not None and written is not None and written > started else {
        "ok": False, "error": "panic_requires_owner_resume"}


def root_lends_panic_release(drive_root, snapshot: dict, task: dict, stop: dict) -> bool:
    """Whether a member's Panic hold is its root's model's to release (``resume_child_task``, shown as ``release:
    mind`` with the door): the root's release record says an act the owner authorised released it after the last
    Panic began, his press (``by.at``) or a mind's release citing his word (``by.owner_word_at``), as
    ``budget_resume.child_selection_control`` grants it. Display only; the selection checks again."""
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.work_stop import SELECTION_KEY, _instant

    root_id = str(task.get("root_task_id") or "")
    if not (stop.get("cause") == "panic" or stop.get("door") == "panic") or root_id in ("", task.get("id")):
        return False
    window = read_panic_window(drive_root)
    started = _instant(window.get("started") or "")
    if window.get("blocked") or started is None:
        return False
    rows = [row.get("task") if isinstance(row.get("task"), dict) else row
            for lane in ("running", "pending") for row in snapshot.get(lane) or () if isinstance(row, dict)]
    root = next((row for row in rows if str(row.get("id") or "") == root_id), {})
    try:
        grant = (budget_pause_row(drive_root, root_id) or {}).get("grant") or {}
    except Exception:
        grant = {}
    for by in ((root.get(SELECTION_KEY) or {}).get("by"), grant.get("by")):
        by = by if isinstance(by, dict) else {}
        when = _instant(by.get("at") if by.get("actor") == "owner" else by.get("owner_word_at") or "")
        if when is not None and when > started:
            return True
    return False


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
    author = control_actor() or owner_door_bound()
    cited = (_OWNER_WORD.get() or {}).get("written")  # a mind's release citing the owner: its grant keeps when he wrote
    return {**author, "owner_word_at": cited} if author and cited else author


@contextmanager
def host_control():
    """Bind, for this one call, the host's own act: an automatic path a door's call can reach
    (``continuation_admission.release_settled_continuations``) runs as the host, not as that door, so neither
    the caller's authorship, the owner's door nor the caller's named decision reads through it."""
    tokens = _ACTOR.set(None), _OWNER_DOOR.set(None), _ACCEPT_UNKNOWN.set(None)
    try:
        yield
    finally:
        _ACCEPT_UNKNOWN.reset(tokens[2])
        _OWNER_DOOR.reset(tokens[1])
        _ACTOR.reset(tokens[0])


def observed_control_stop(root, task=None, *, budget=None, owner=None) -> dict:
    """Feed observed work and tree controls to the pure stop reader."""
    from ouroboros.work_stop import current_stop

    return current_stop(task, {"budget": budget or {}, "owner": owner or {}})

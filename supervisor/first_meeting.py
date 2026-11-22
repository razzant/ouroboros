"""My human's press of Start opens the first meeting: one host-started root, at most once per state.

Setup's Finish (the wizard's Start, ``POST /api/onboarding/complete``) records ``OUROBOROS_ONBOARDING_COMPLETED_AT``,
and the first state minted after it is born owing the meeting (``state_initialization.first_state`` stores ``owed``
under ``FIRST_MEETING_KEY``). Only such a state owes one: a state minted before a setup, an install without one (Colab,
headless, env-only) and every state an earlier release minted do not. ``admit_first_meeting`` runs at each supervisor
start (``server._run_supervisor``) and, whatever the consciousness switch says, admits ONE ordinary root the way
``gateway.control._admit_description_task`` admits a press's work: in Main, where the press was made (owner item 1),
never in an owner chat bound since (a Telegram chat whose first message arrived before this start). Its objective is the
host's prompt for my human's press of ``setup_start`` (``objective_author`` ``ui_control``): when the press happened,
nothing else. What to say or do is the mind's, from its context (the Runtime ``setup`` facts, ``context_first_hour``).

Order: the key moves ``owed`` -> ``admitting`` with the root's id BEFORE the admission, then -> ``admitted``. A start
that finds ``admitting`` admits the same id again, and the queue refuses an id it holds or has a result for
(``_require_unique_task_id``; the admission writes its durable scheduled receipt in the same transaction), so the
meeting is admitted at most once, and a crash between the write and the admission leaves the record that retries it.
A refused admission stays ``admitting`` with its reason for the next start. While setup's chosen access level waits
for a restart (this process still runs its boot level: ``restart_required``), nothing is admitted: the meeting stays
owed for the restarted process, so it never runs under a level my human did not choose and a desktop restart never
consumes it. A message of mine to my human that already
exists (``dialogue_store.owner_exchange``) settles it as ``not_needed``; a dialogue or state that cannot be read leaves
it as it was. Every attempt writes one ``first_meeting`` row to ``logs/events.jsonl``.
"""

from __future__ import annotations

import logging
import pathlib
import uuid
from typing import Any, Dict, Tuple

from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

FIRST_MEETING_KEY = "first_meeting"
CONTROL = "setup_start"  # the press: setup's Start, the end of the wizard
_OPEN = frozenset({"owed", "admitting"})


class _Settled(Exception):
    """The meeting was settled or claimed with another id meanwhile: nothing to write."""


def owed(pressed_at: str) -> Dict[str, str]:
    """What a first state minted after setup is born with: the meeting its press of Start opens."""
    return {"state": "owed", "pressed_at": str(pressed_at)}


def objective(pressed_at: str) -> str:
    """The host's words for the press: when it happened, never what to do (BIBLE P5). The first line is the name the
    work's card shows until one is coined (``project_naming.admission_names``)."""
    return f"Setup finished\n\nMy human pressed Start at the end of setup at {pressed_at or 'an unrecorded time'}."


def admit_first_meeting(drive_root: Any) -> str:
    """At a supervisor start: admit the meeting this state owes, once. Returns the outcome word, recorded as one row:
    ``none`` (nothing owed), ``not_needed``, ``deferred``, ``refused`` or ``admitted``."""
    from supervisor.state import load_state

    try:  # a start never fails on this, its state read included; the key stays as it was and the next start repeats it
        meeting = load_state().get(FIRST_MEETING_KEY)
        if not isinstance(meeting, dict) or meeting.get("state") not in _OPEN:
            return "none"
        outcome, detail = _admit(drive_root, meeting)
    except Exception as exc:
        log.warning("first meeting not admitted", exc_info=True)
        outcome, detail = "deferred", {"reason": f"{type(exc).__name__}: {exc}"}
    try:
        append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl",
                     {"ts": utc_now_iso(), "type": "first_meeting", "outcome": outcome, **detail})
    except Exception:
        log.debug("first meeting row not recorded", exc_info=True)
    return outcome


def _admit(drive_root: Any, meeting: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    from ouroboros import dialogue_store
    from ouroboros.config import get_runtime_mode, load_settings, normalize_runtime_mode
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.store import StoreError
    from supervisor.queue import enqueue_with_admission_receipt, persist_queue_snapshot
    from supervisor.state import StateUnavailable, update_state

    chosen = normalize_runtime_mode(load_settings().get("OUROBOROS_RUNTIME_MODE"))  # what Finish saved (policy applied)
    if chosen != get_runtime_mode():
        return "deferred", {"reason": f"a restart into the chosen access level ({chosen}) comes first"}
    try:
        _theirs, mine = dialogue_store.owner_exchange(drive_root)
    except StoreError as exc:
        return "deferred", {"reason": f"the dialogue could not be read ({type(exc).__name__})"}
    if mine:
        held = str(meeting.get("task_id") or "")
        update_state(lambda live: _settle(live, "not_needed", held))
        return "not_needed", {"reason": "a message of mine to my human exists", **({"task_id": held} if held else {})}
    chat_id = WEB_UI_CHAT_ID  # the press's own room, whatever owner chat is bound or unread
    task_id = str(meeting.get("task_id") or "") or uuid.uuid4().hex[:8]
    try:
        update_state(lambda live: _claim(live, task_id, chat_id))
    except _Settled:
        return "none", {}
    except StateUnavailable as exc:
        return "deferred", {"reason": f"state not written: {exc.reason}"}
    pressed_at = str(meeting.get("pressed_at") or "")
    admitted = enqueue_with_admission_receipt({
        "id": task_id, "type": "task", "chat_id": chat_id, "root_task_id": task_id, "delegation_role": "root",
        "text": objective(pressed_at),
        "metadata": {"ui_control": {"name": CONTROL, **({"pressed_at": pressed_at} if pressed_at else {})},
                     "objective_author": {"kind": "ui_control", "control": CONTROL}},
        "_require_worker_pool": True, "_require_unique_task_id": True,
    })
    blocked = str(admitted.get("_admission_blocked") or "") if isinstance(admitted, dict) else "refused"
    if blocked and blocked != "duplicate_task_id":  # a duplicate is this id admitted by an earlier start
        reason = str(admitted.get("_worker_pool_disabled_reason") or blocked) if isinstance(admitted, dict) else blocked
        update_state(lambda live: _settle(live, "admitting", task_id, last_refusal=reason))
        return "refused", {"task_id": task_id, "reason": reason}
    persist_queue_snapshot(reason="first_meeting")
    update_state(lambda live: _settle(live, "admitted", task_id))
    return "admitted", {"task_id": task_id, "chat_id": chat_id}


def _claim(live: Dict[str, Any], task_id: str, chat_id: int) -> None:
    """Under the state lock, before the admission: the meeting is this id's, ``admitting``."""
    current = live.get(FIRST_MEETING_KEY)
    if not isinstance(current, dict) or current.get("state") not in _OPEN or current.get("task_id", task_id) != task_id:
        raise _Settled()
    live[FIRST_MEETING_KEY] = {**current, "state": "admitting", "task_id": task_id, "chat_id": chat_id,
                               "admitting_at": utc_now_iso()}


def _settle(live: Dict[str, Any], state: str, task_id: str, **fields: Any) -> None:
    """The meeting's next state, written only over the id this start holds (or an unclaimed one)."""
    current = live.get(FIRST_MEETING_KEY)
    if not isinstance(current, dict) or current.get("state") not in _OPEN or current.get("task_id", task_id) != task_id:
        return
    live[FIRST_MEETING_KEY] = {**current, "state": state, f"{state}_at": utc_now_iso(), **fields}

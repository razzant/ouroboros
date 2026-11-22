"""The Runtime section's ``setup`` key: how this install began and whether my human and I have met, as facts.

Every non-declared context carries it (``context.build_runtime_section``) in the dynamic part beside the body facts,
so the shared prefix never changes. Each line is a reading with its source, or ``unknown: <why>`` when that source
cannot be read; none is a threshold, an instruction or a biography: whether this is a first meeting is the mind's
judgement.

- ``finished_at``: the last recorded setup Finish (``OUROBOROS_ONBOARDING_COMPLETED_AT``, a completion fact only the
  settings document carries).
- ``state``: this state's origin and birth from its initialization record (``state/state.initialized.json``):
  ``first_boot``, ``owner_reset`` (what a Reset deletes: ``gateway.control.api_reset``) or ``legacy_adopted``.
- ``owner_chat``: bound, known empty or unknown, read by the clock's own rule (``consciousness.owner_chat_of``).
- ``exchange``: whether my human and I have written to each other in the rooms they read, from one statement of two
  indexed existence probes (``dialogue_store.owner_exchange``); a read that fails is unknown, never no.
- ``consciousness``: on, off or unknown as the state stands now.
- ``starter_skills``: the starter installer's outcome counts (``state/skill_starter.json``) where it ran.
- ``startup_problems``: what the newest ``startup_verification`` row found wrong, from the events tail the health
  block reads (one read serves both); present only while that row counted issues.
- ``first_meeting``: the one line that is not a reading: the owner's words for a first meeting
  (``prompts/FIRST_MEETING.md``), carried only while a press of Start is on record (the state's ``first_meeting``
  record, which only a state minted after a setup's Finish carries, in a state read that is the current one: a state
  whose initialization record is missing or does not match is unknown, and carries none) and I KNOW I have not written
  to my human (an unknown exchange carries none), and only once that file exists.
"""

from __future__ import annotations

import collections
import datetime
import logging
import pathlib
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)

_EXCHANGE_WORDS = {(False, False): "my human and I have not written to each other yet",
                   (True, False): "my human has written to me; I have not written to them yet",
                   (False, True): "I have written to my human; they have not written to me yet",
                   (True, True): "my human and I have both written"}
# The owner's words for a first meeting: absent until he approves their exact text.
GUIDANCE_REL = pathlib.Path("prompts") / "FIRST_MEETING.md"
_RESET_DELETES = "a Reset deletes memory, chats, logs, task results, uploads and settings"
_STARTER_WORDS = {"fresh": "not run yet", "running": "installing", "pending": "waiting to retry", "done": "done"}
_QUIET = frozenset({"ok", "skipped", "unconfigured"})
# The startup checks that record no status (``agent_startup_checks.verify_system_state``) and the field each needs true.
_STATUSLESS = {"identity": "non_empty", "scratchpad": "exists", "world_profile": "exists"}
_STARTUP_READER = "its startup_verification row in logs/events.jsonl"


def setup_fact(data_root: Any, repo_dir: Any = None) -> Dict[str, str]:
    """The ``setup`` key of ``data_root`` (the canonical data root); never raises, each line answers for itself.
    While a press of Start is on record and I know I have not written to my human, it also carries the owner's
    first-meeting words from ``repo_dir`` if they exist (``GUIDANCE_REL``)."""
    root = pathlib.Path(data_root)
    fact = {"finished_at": _read(_finished_at, root), "state": _read(_state_origin, root)}
    controls = _read(_controls, root, {"owner_chat": "unknown: runtime state unreadable", "consciousness": "unknown"})
    pressed = controls.pop("pressed", False)
    fact.update(controls)
    fact["exchange"], mine = _read(_exchange, root, ("unknown: unreadable", None))
    for key, reader in (("starter_skills", _starter), ("startup_problems", _startup_problems)):
        line = _read(reader, root, "")
        if line:
            fact[key] = line
    guidance = _read(_guidance, repo_dir, "") if pressed and mine is False and repo_dir is not None else ""
    if guidance:
        fact["first_meeting"] = guidance
    return fact


def _read(reader: Callable[[pathlib.Path], Any], root: pathlib.Path, unread: Any = "unknown: unreadable") -> Any:
    try:
        return reader(root)
    except Exception:  # a fact never breaks the Runtime section
        log.debug("setup fact %s unreadable", reader.__name__, exc_info=True)
        return unread


def _minute(stamp: Any) -> str:
    """A recorded stamp to the minute in UTC (``2026-11-21 09:58 UTC``); the text as recorded when it is no stamp."""
    from ouroboros.deadline_utils import parse_deadline_ts

    parsed = parse_deadline_ts(stamp)
    return parsed.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if parsed else str(stamp or "")


def _finished_at(root: pathlib.Path) -> str:
    from ouroboros import config
    from ouroboros.settings_integrity import SettingsDocumentUnknown, read_settings_json_strict
    from ouroboros.settings_setup_contract import ONBOARDING_COMPLETED_KEY

    configured = root.resolve() == pathlib.Path(config.DATA_DIR).resolve()
    try:
        document = read_settings_json_strict(config.SETTINGS_PATH if configured else root / "settings.json")
    except SettingsDocumentUnknown:
        return "unknown: the settings document cannot be read"
    stamp = str((document or {}).get(ONBOARDING_COMPLETED_KEY) or "").strip()
    return _minute(stamp) if stamp else "none recorded"


def _state_origin(root: pathlib.Path) -> str:
    from ouroboros.launch_roots import read_witness

    status, witness = read_witness(root)
    if status != "ok":
        return ("unknown: no initialization record" if status == "missing"
                else f"unknown: the initialization record is {status}")
    origin = str(witness.get("origin") or "unknown origin")
    if witness.get("phase") != "complete":
        return f"{origin}: an initialization pending since {_minute(witness.get('created_at'))}"
    born = _minute(witness.get("completed_at") or witness.get("created_at"))
    if origin == "first_boot":
        return f"first_boot: the first state of this data root, born {born}"
    if origin == "owner_reset":
        reset_at = _minute(witness.get("created_at"))
        return f"owner_reset: a new state born {born} after my human's Reset at {reset_at}; {_RESET_DELETES}"
    if origin == "legacy_adopted":
        return f"legacy_adopted: a state older than its initialization record, adopted {born}"
    return f"{origin}: born {born}"


def _controls(root: pathlib.Path) -> Dict[str, str]:
    """The owner chat and the consciousness switch from one state read, and whether that state records a press of
    Start (``pressed``: its ``first_meeting`` record, which only a state born after a setup's Finish carries; never
    from a read that is not the current state, such as a primary whose initialization record is missing)."""
    from ouroboros.consciousness import owner_chat_of
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from supervisor.first_meeting import FIRST_MEETING_KEY
    from supervisor.state import CURRENT_QUALITIES, control_value, read_state

    read = read_state(root)
    view = read.projection()
    try:
        chat = owner_chat_of(view)
        owner = ("none bound yet: no owner message has bound one" if chat is None
                 else "Main (chat 1)" if chat == WEB_UI_CHAT_ID else f"chat {chat}")
    except LookupError as exc:
        owner = f"unknown: {exc}"
    known, enabled = control_value(view, "bg_consciousness_enabled")
    switch = "on" if known and enabled is True else "off" if known and enabled is False else "unknown"
    pressed = read.quality in CURRENT_QUALITIES and isinstance(view.get(FIRST_MEETING_KEY), dict)
    return {"owner_chat": owner, "consciousness": switch, "pressed": pressed}


def _exchange(root: pathlib.Path) -> Tuple[str, Optional[bool]]:
    """Whether my human and I have written to each other in the rooms they read (``dialogue_store.owner_exchange``),
    and whether I have (``None`` while unknown)."""
    from ouroboros import dialogue_store
    from ouroboros.store import StoreError

    try:
        theirs, mine = dialogue_store.owner_exchange(root)
    except StoreError as exc:
        return f"unknown: the dialogue could not be read ({type(exc).__name__})", None
    return _EXCHANGE_WORDS[(theirs, mine)], mine


def _guidance(repo_dir: Any) -> str:
    """The owner's words for a first meeting (``prompts/FIRST_MEETING.md``) once they exist; "" while they do not."""
    try:
        return (pathlib.Path(repo_dir) / GUIDANCE_REL).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _starter(root: pathlib.Path) -> str:
    """The starter installer's outcome counts; "" where it never ran (no marker, or an install older than it)."""
    from ouroboros.skill_starter import marker_path
    from ouroboros.utils import read_json_dict

    marker = read_json_dict(marker_path(root)) or {}
    state = str(marker.get("state") or "")
    results = marker.get("results") if isinstance(marker.get("results"), dict) else {}
    if state in ("", "upgrade") and not results:
        return ""
    counts = collections.Counter(str(row.get("status") or "unknown")
                                 for row in results.values() if isinstance(row, dict))
    words = [f"{counts.pop(name, 0)} {name}" for name in ("installed", "pending", "failed")]
    words += [f"{count} {name}" for name, count in sorted(counts.items())]
    chosen = marker.get("requested")
    chosen = f"{len(chosen)} chosen; " if isinstance(chosen, list) else ""
    return f"{_STARTER_WORDS.get(state, state or 'unknown')}: {chosen}{', '.join(words)} (state/skill_starter.json)"


def _startup_problems(root: pathlib.Path) -> str:
    """The newest startup verification in the journal's bounded tail, when it counted issues; "" otherwise."""
    from ouroboros.context_health import _RECENT_LOG_BYTES
    from ouroboros.jsonl_tail import cached_tail_objects

    rows = cached_tail_objects(root / "logs" / "events.jsonl", _RECENT_LOG_BYTES)
    row = next((row for row in reversed(rows) if row.get("type") == "startup_verification"), None) or {}
    issues = row.get("issues_count")
    if type(issues) is not int or issues <= 0:
        return ""
    found = []
    for name, check in (row.get("checks") if isinstance(row.get("checks"), dict) else {}).items():
        status = check.get("status") if isinstance(check, dict) else None
        if status is not None and status not in _QUIET:
            found.append(f"{name} {status}")
        elif isinstance(check, dict) and status is None and name in _STATUSLESS and not check.get(_STATUSLESS[name]):
            found.append(f"{name} missing")
        elif isinstance(check, dict) and name == "model" and check.get("configured") == "(not set)":
            found.append("model not set")
    named = f": {', '.join(found)}" if found else ""
    return f"{_minute(row.get('ts'))}, {issues} issue{'' if issues == 1 else 's'}{named} ({_STARTUP_READER})"

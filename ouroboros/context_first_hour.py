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
- ``main``: whether my human has written in Main and whether a message of mine follows their first one there, from
  one indexed range of Main's first rows (``_MAIN_ROWS``, a read bound) that reads no message body.
- ``consciousness``: on, off or unknown as the state stands now.
- ``starter_skills``: the starter installer's outcome counts (``state/skill_starter.json``) where it ran.
- ``startup_problems``: what the newest ``startup_verification`` row found wrong, from the events tail the health
  block reads (one read serves both); present only while that row counted issues.
"""

from __future__ import annotations

import collections
import datetime
import logging
import pathlib
from typing import Any, Callable, Dict

log = logging.getLogger(__name__)

_MAIN_ROWS = 500  # Main's first rows one read covers (an index range, no bodies): a read bound, never a judgement
_RESET_DELETES = "a Reset deletes memory, chats, logs, task results, uploads and settings"
_STARTER_WORDS = {"fresh": "not run yet", "running": "installing", "pending": "waiting to retry", "done": "done"}
_QUIET = frozenset({"ok", "skipped", "unconfigured"})
# The startup checks that record no status (``agent_startup_checks.verify_system_state``) and the field each needs true.
_STATUSLESS = {"identity": "non_empty", "scratchpad": "exists", "world_profile": "exists"}
_STARTUP_READER = "its startup_verification row in logs/events.jsonl"


def setup_fact(data_root: Any) -> Dict[str, str]:
    """The ``setup`` key of ``data_root`` (the canonical data root); never raises, each line answers for itself."""
    root = pathlib.Path(data_root)
    fact = {"finished_at": _read(_finished_at, root), "state": _read(_state_origin, root)}
    fact.update(_read(_controls, root, {"owner_chat": "unknown: runtime state unreadable", "consciousness": "unknown"}))
    fact["main"] = _read(_main, root)
    for key, reader in (("starter_skills", _starter), ("startup_problems", _startup_problems)):
        line = _read(reader, root, "")
        if line:
            fact[key] = line
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
    """The owner chat and the consciousness switch from one state read."""
    from ouroboros.consciousness import owner_chat_of
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from supervisor.state import control_value, read_state

    view = read_state(root).projection()
    try:
        chat = owner_chat_of(view)
        owner = ("none bound yet: no owner message has bound one" if chat is None
                 else "Main (chat 1)" if chat == WEB_UI_CHAT_ID else f"chat {chat}")
    except LookupError as exc:
        owner = f"unknown: {exc}"
    known, enabled = control_value(view, "bg_consciousness_enabled")
    switch = "on" if known and enabled is True else "off" if known and enabled is False else "unknown"
    return {"owner_chat": owner, "consciousness": switch}


def _main(root: pathlib.Path) -> str:
    """My human's first message in Main and the first message of mine after it, among Main's first rows."""
    from ouroboros import dialogue_store
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.store import StoreError

    rows: list = []
    if dialogue_store.present(root):
        try:
            with dialogue_store.transaction(root, write=False) as conn:
                rows = conn.execute("SELECT ts, direction, kind FROM dialogue INDEXED BY dialogue_room "
                                    "WHERE stream='chat' AND chat_id=? ORDER BY seq LIMIT ?",
                                    (WEB_UI_CHAT_ID, _MAIN_ROWS)).fetchall()
        except StoreError as exc:
            return f"unknown: Main could not be read ({type(exc).__name__})"
    elif dialogue_store.holds_dialogue_files(root):
        return "unknown: the dialogue store is not created yet"
    # Untyped inbound rows are my human's words; untyped outbound rows are my messages (host rows are typed or System).
    first = next((at for at, row in enumerate(rows) if row["direction"] == "in" and row["kind"] is None), None)
    start = 0 if first is None else first + 1
    mine = next((row for row in rows[start:] if row["direction"] == "out" and row["kind"] is None), None)
    whole = len(rows) < _MAIN_ROWS
    if first is None:
        if not whole:
            return f"my human's first message is not among Main's first {_MAIN_ROWS} rows"
        return "my human has not written in Main" + (f"; my first message there {_minute(mine['ts'])}" if mine
                                                    else ", nor have I")
    said = f"my human first wrote in Main {_minute(rows[first]['ts'])}"
    if mine is not None:
        return f"{said}; my first message after it {_minute(mine['ts'])}"
    return said + ("; no message of mine after it yet" if whole
                   else f"; none of mine after it in Main's first {_MAIN_ROWS} rows")


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

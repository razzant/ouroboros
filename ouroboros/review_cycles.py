"""Shared review-cycle cap — SSOT for ``OUROBOROS_REVIEW_MAX_CYCLES``.

ONE owner-facing knob bounds how many PAID review cycles each review gate may
spend before it stops and discloses (owner decisions D9/D10/D19/D20/D26-D28).
The setting is a STRING — a positive integer ("1", "2", "5", …) or
``"unlimited"`` (aliases ``inf``/``∞``); anything else fails closed to the
shipped default with one loud log line. It is a string on purpose:
``config._coerce_setting_value`` casts by the default's type, so an ``int``
default would silently swallow ``"unlimited"``. ``review_max_cycles()`` returns
``Optional[int]`` — ``None`` means unlimited.

Per-gate meaning of the ONE number — on every gate it counts PAID cycles, and
identical material is never re-reviewed for pay unless the mind sends it with
answers:

* plan asks are OUTSIDE this cap (owner batch 2, Q3=A): the planning organ
  (``tools/plan_ask``) dispatches an ask whenever the mind asks, and only its
  budget gate and deadline bound it; nothing here counts plan asks;
* result review (``review_change(subject="result")``) — paid result waves per
  ROOT task, counted apart from the task's change-review waves (its own
  ``tool_name`` on the attempt rows), so change reviews never spend it;
* commit gate — paid review-wave cycles per ROOT task (the whole task tree
  shares one ceiling; a manual session is its own task; a follow-up task is a
  fresh root). The paid fact is recorded on the attempt row AT DISPATCH and
  the count is derived from the attempt ledger; the ceiling counts MONEY —
  every dispatched wave counts whatever its terminal, and only UNDISPATCHED
  attempts (free refusals/replays, preflight and assembly failures) stay
  outside the count, which is also the only sense in which "infra retries
  freely" applies to the ceiling. Independently of the number, a
  byte-identical staged diff whose last terminal is a review-VERDICT block is
  refused for FREE from the FIRST block (``identical_diff_refused``, quoting
  the recorded verdict); a rebuttal is content-hashed and a hash new to the
  streak buys exactly ONE paid re-review (a rebuttal is "spent" only when it
  bought a dispatched, verdict-answered wave). Exhaustion under blocking is a
  free typed refusal; under advisory the outcome returns with a typed
  disclosure for explicit author continuation, without another paid dispatch;
* skill review (``skill_review`` → ``review_change(subject="skill")``) — paid skill waves per
  root task for a task-driven review (shared across every skill that task reviews; a
  follow-up starts fresh), counted apart from the task's change and result waves (its own
  ``tool_name`` on the attempt rows); the manual lane (the Skills page, a Hub install) has
  no task, so its ceiling is per CONTENT VERSION of the skill (owner Q23). A byte-identical
  skill under the same rules and panel reuses its settled record at $0 (the ledger's reuse
  key); exhaustion is the same typed event with ``surface="review_skill"``.

Values are read from ``os.environ`` (``config.apply_settings_to_env`` projects
saved settings there) falling back to ``SETTINGS_DEFAULTS`` — no second
mechanism. The deprecated alias ``OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES`` does not bind
at runtime at all: a customized value is MIGRATED into this knob when settings load
(``config.load_settings``, the same rename-alias shape the retention keys use), so the
visible setting is always the authority. Residual, disclosed: a legacy value supplied
only through the environment is not migrated and no longer binds. No new state files; the only other side effect is
``emit_review_cycles_exhausted``, the typed D27 escalation event on the existing
``log_event``/``events.jsonl`` rail.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Optional

from ouroboros.config import SETTINGS_DEFAULTS
# The ONE typed reason/event name for "the shared cap is spent under blocking
# enforcement" (owner D10/D27) — SSOT beside the other outcome reasons.
from ouroboros.outcomes import REASON_REVIEW_CYCLES_EXHAUSTED  # noqa: F401 — re-export
from ouroboros.utils import append_jsonl, emit_log_event, utc_now_iso
from ouroboros.config import runtime_setting
# The canonical "no cap" token (the UI's ∞ saves it), its aliases and the strict parser are the
# ONE optional-bound vocabulary in ``settings_scales``, shared with the task round/lifetime limits.
from ouroboros.settings_scales import UNLIMITED, parse_positive_or_unlimited

log = logging.getLogger(__name__)

REVIEW_MAX_CYCLES_KEY = "OUROBOROS_REVIEW_MAX_CYCLES"
_WARNED: set = set()


def parse_review_max_cycles(raw: Any) -> Optional[int]:
    """Strict parser: positive-integer text → int; an unlimited alias → None.

    Raises ``ValueError`` for anything else (empty, zero, negative, non-integer,
    unknown word) so callers decide between fail-closed default and 400."""
    return parse_positive_or_unlimited(raw)


def is_valid_review_max_cycles(raw: Any) -> bool:
    """SSOT predicate for the settings write boundary (mirrors
    ``config.is_valid_post_task_evolution_cadence``): True iff ``raw`` is a
    positive integer or an unlimited alias."""
    try:
        parse_review_max_cycles(raw)
    except (TypeError, ValueError):
        return False
    return True


def normalize_review_max_cycles(raw: Any) -> str:
    """Canonical persisted form of a VALID value: ``"unlimited"`` for every
    alias, else the integer text. Callers validate first (``is_valid_...``)."""
    parsed = parse_review_max_cycles(raw)
    return UNLIMITED if parsed is None else str(parsed)


def default_review_max_cycles() -> int:
    """The shipped default as an int (the default is bounded by construction)."""
    return int(str(SETTINGS_DEFAULTS[REVIEW_MAX_CYCLES_KEY]))


def review_max_cycles() -> Optional[int]:
    """The shared cap: ``None`` = unlimited, else a positive int.

    Env-or-default like every other getter; a malformed value fails CLOSED to
    the shipped default (bounded) and is reported once per process."""
    default_text = str(SETTINGS_DEFAULTS[REVIEW_MAX_CYCLES_KEY])
    raw = runtime_setting(REVIEW_MAX_CYCLES_KEY, "") or default_text
    try:
        return parse_review_max_cycles(raw)
    except (TypeError, ValueError):
        if f"invalid:{raw!r}" not in _WARNED:
            _WARNED.add(f"invalid:{raw!r}")
            log.warning(
                f"{REVIEW_MAX_CYCLES_KEY}={raw!r} is not a positive integer or "
                f"'unlimited'; using the shipped default {default_text} (bounded).",
            )
        return default_review_max_cycles()


def review_max_cycles_source() -> str:
    """Where the effective cap came from: ``owner_setting`` when the key is
    present in the environment (``config.apply_settings_to_env`` projects saved
    settings there, so an owner edit and an env override are the same fact),
    else ``shipped_default``. Provenance only — never a second parse."""
    return "owner_setting" if runtime_setting(REVIEW_MAX_CYCLES_KEY, "") else "shipped_default"


def emit_review_cycles_exhausted(
    event_queue: Any, drive_root: Any, *, surface: str, task_id: str,
    cycles_paid: int, cap: int, enforcement: str, **extra: Any,
) -> None:
    """The typed escalation event (D27): the durable ``events.jsonl`` append ALWAYS
    lands (the live queue path persists only task_checkpoint rows), and a live
    queue additionally gets the UI push. No new ledger."""
    row = {"type": REASON_REVIEW_CYCLES_EXHAUSTED, "surface": str(surface), "task_id": str(task_id or ""),
           "cycles_paid": int(cycles_paid), "cap": int(cap), "enforcement": str(enforcement or ""), **extra}
    try:
        stamped = {"ts": utc_now_iso(), **row}
        if drive_root:
            append_jsonl(pathlib.Path(str(drive_root)) / "logs" / "events.jsonl", stamped)
        if event_queue is not None:
            emit_log_event(event_queue, stamped, log_label="review cycles")
        else:
            # A queue-less server-process caller (e.g. the HTTP skill-review
            # ctx) still owes the ONE live frame: the raw sink copy of the
            # append above is suppressed, so push the addressed sibling here.
            from supervisor.log_addressing import address_handler_push
            from supervisor.message_bus import try_get_bridge

            bridge = try_get_bridge()
            if bridge is not None:
                bridge.push_log(address_handler_push(
                    pathlib.Path(str(drive_root)) if drive_root else None, dict(stamped)
                ))
    except Exception:
        log.debug("review_cycles_exhausted emission failed for %s", task_id, exc_info=True)

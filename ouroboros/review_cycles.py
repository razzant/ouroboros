"""Shared review-cycle cap — SSOT for ``OUROBOROS_REVIEW_MAX_CYCLES``.

ONE owner-facing knob bounds how many PAID review cycles each review gate may
spend before it stops and discloses (owner decisions D9/D10/D19/D20/D26-D28).
The setting is a STRING — a positive integer ("1", "2", "5", …) or
``"unlimited"`` (aliases ``inf``/``∞``); anything else fails closed to the
shipped default with one loud log line. It is a string on purpose:
``config._coerce_setting_value`` casts by the default's type, so an ``int``
default would silently swallow ``"unlimited"``. ``review_max_cycles()`` returns
``Optional[int]`` — ``None`` means unlimited.

Per-gate meaning of the ONE number — on every gate it counts PAID cycles (and, on
the result gate, the free rewording reminders too); a
usable verdict on identical material is never re-reviewed for pay unless the
mind sends it with answers, while an explicit identical request after a settled
wave formed no verdict is a new paid wave (running or unknown work is collected,
never sent again):

* plan asks are OUTSIDE this cap (owner batch 2, Q3=A): the planning organ
  (``tools/plan_ask``) dispatches an ask whenever the mind asks, and only its
  budget gate and deadline bound it; nothing here counts plan asks;
* result review (``review_change(subject="result")``) — paid result waves per
  ROOT task, counted apart from the task's code-review waves (its own
  ``tool_name`` on the attempt rows), so code reviews never spend it; the same
  number bounds the free reminders asking the author to reword a reviewed answer
  (``tools/review_result._revised``, counted by its saved questions), while an
  unread outcome and newer owner words still reach the author regardless;
* code review — ``review_change`` over code and the commit gate, its named
  preflight look included, spend ONE allowance per (task tree, repository):
  the whole tree shares it and a Continue successor inherits what the roots it
  continues spent, while a manual session, a follow-up task and every other
  independent task is a fresh root; another repository of the same tree has
  its own allowance, and a copy counts with its source
  (``paid_review_pool``). The paid fact is recorded on the
  attempt row AT DISPATCH and the count is derived from the attempt rows; the
  ceiling counts MONEY — every dispatched wave counts whatever its terminal,
  and only UNDISPATCHED attempts (free refusals/replays, preflight and
  assembly failures) stay outside the count, which is also the only sense in
  which "infra retries freely" applies to the ceiling. A rebuttal is
  content-hashed into the round's identity, so a new one buys a new paid wave
  within the ceiling. A commit whose allowance cannot cover both a new named
  look and its panel leaves the wave to the panel and records the look not
  performed. Exhaustion returns the outcome with a typed disclosure for
  explicit author continuation, without another paid dispatch;
* skill review (``skill_review`` → ``review_change(subject="skill")``) — paid skill waves per
  root task for a task-driven review (shared across every skill that task reviews; a
  follow-up starts fresh), counted apart from the task's code and result waves (its own
  ``tool_name`` on the attempt rows); the manual lane (the Skills page, a Hub install) has
  no task, so its ceiling is per CONTENT VERSION of the skill (owner Q23). A byte-identical
  skill under the same rules and panel reuses its settled verdict at $0 (the ledger's reuse
  key); exhaustion is the same typed event with ``surface="review_skill"``.

Values are read from ``os.environ`` (``config.apply_settings_to_env`` projects
saved settings there) falling back to ``SETTINGS_DEFAULTS`` — no second
mechanism. The deprecated alias ``OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES`` does not bind
at runtime at all: a customized value is MIGRATED into this knob when settings load
(``config.load_settings``, the same rename-alias shape the retention keys use), so the
visible setting is always the authority. Residual, disclosed: a legacy value supplied
only through the environment is not migrated and no longer binds. No new state files: the pool is counted from the
existing attempt state (``paid_review_pool``, with ``note_paid_code_wave``'s canonical rows), and the only other
side effect is ``emit_review_cycles_exhausted``, the typed D27 escalation event on the existing
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


CODE_REVIEW_TOOLS = frozenset({"review_change", "commit_reviewed", "vcs_commit_reviewed"})
# What a canonical paid-wave row keeps of its attempt row: whose pool it is and which wave it was.
_PAID_WAVE_FIELDS = ("root_task_id", "review_budget_repo_key", "repo_key", "tool_name", "task_id", "attempt",
                     "review_retry_key", "started_ts")


def note_paid_code_wave(ctx: Any, drive: Any, *, repo_key: str, tool_name: str, task_id: str, attempt: int) -> None:
    """Keep a paid commit-gate wave countable after its drive is gone: called at the paid stamp, it copies the
    accounting facts of the stamped attempt row on a task's OWN drive into the canonical attempt state
    (``AdvisoryReviewState.paid_code_waves``). Idempotent; a wave bought on the canonical drive needs none.
    Accounting never stops the dispatch it counts: a failed write is logged, and the wave stays counted from its
    drive while that drive is retained."""
    try:
        from ouroboros.review_ledger import ledger_root
        from ouroboros.review_state import ReviewStateMutation, load_state, update_state
        from ouroboros.review_state_custody import SEAT_LAUNCH_LOCK_WAIT_SEC

        canonical = ledger_root(ctx)
        if pathlib.Path(drive).resolve(strict=False) == canonical:
            return
        row = load_state(pathlib.Path(drive)).latest_attempt_for(repo_key=repo_key, tool_name=tool_name, task_id=task_id,
                                                                 attempt=int(attempt or 0))
        if row is None or not row.paid:
            return
        wave = {name: getattr(row, name) for name in _PAID_WAVE_FIELDS}

        def keep(state: Any) -> Any:
            known = wave in state.paid_code_waves
            if not known:
                state.paid_code_waves.append(wave)
            return ReviewStateMutation(None, changed=not known)

        update_state(canonical, keep, lock_timeout_sec=SEAT_LAUNCH_LOCK_WAIT_SEC)
    except Exception:
        log.warning("paid review wave of %s was not noted on the canonical root", task_id, exc_info=True)


def paid_review_pool(ctx: Any, root_task_id: str, tool: str) -> tuple[int, tuple[str, ...]]:
    """``(paid waves, the roots whose waves they are)`` of the pool a ``tool`` review of ``ctx.repo_dir`` spends
    from, derived from the existing attempt state (P7: no counter file). The pool counts MONEY: every attempt that
    physically dispatched a wave (``paid``, stamped at dispatch) counts whatever its terminal; free refusals,
    replays and assembly failures (``paid=False``) never do.

    Code review is ONE pool per (task tree, repository): ``review_change`` and the commit gate spend the same
    allowance, and a Continue successor inherits what the roots it continues spent
    (``continuation_admission.continuation_ancestry``); an independent task starts its own. The repository is the
    row's pinned budget owner (``review_budget_repo_key``: a copy's source by host provenance), else its physical
    checkout resolved now. Waves are read from the canonical state (every ``review_change`` row, a canonical-drive
    commit row, and ``note_paid_code_wave``'s row for a commit wave bought on a task's own drive), then, for rows
    older than those notes, from the calling drive and the retained own drives of the tree's roots. One wave is
    counted once: by its review record where its row names one, and never again from a drive whose wave was noted.
    Result and skill waves keep their own allowance per root and kind."""
    root = str(root_task_id or "")
    if not root:
        return 0, ()
    from ouroboros.review_ledger import ledger_root
    from ouroboros.review_state import CommitAttemptRecord, load_state
    from ouroboros.tools.review_change_custody import paid_repo_key, review_state_root

    if tool in ("review_result", "review_skill"):  # a root's allowance, wherever the material lives
        return sum(1 for item in load_state(review_state_root(ctx)).filter_attempts(tool_name=tool)
                   if item.paid and item.root_task_id == root), (root,)
    from ouroboros.review_state_records import _attempt_identity_tuple
    from ouroboros.source_retention import retained_task_roots
    from supervisor.continuation_admission import continuation_ancestry

    canonical = pathlib.Path(review_state_root(ctx) if tool == "review_change" else ledger_root(ctx)).resolve(strict=False)
    roots = continuation_ancestry(canonical, root)
    states = {str(canonical): load_state(canonical)}
    for drive in (pathlib.Path(ctx.drive_root), *(d for member in roots for d in retained_task_roots(canonical, member))):
        states.setdefault(str(drive.resolve(strict=False)), load_state(drive))
    noted = [CommitAttemptRecord(ts="", commit_message="", status="reviewed", paid=True,
                                 **{name: wave[name] for name in _PAID_WAVE_FIELDS if name in wave})
             for wave in states[str(canonical)].paid_code_waves]
    owner = paid_repo_key(ctx, ctx.repo_dir)
    # Unrelated history is filtered before any provenance read; equal physical keys share one resolution.
    owners, waves, held = {str(pathlib.Path(ctx.repo_dir).resolve()): owner}, set(), set()
    for is_note, item in (*((True, item) for item in noted),
                          *((False, item) for state in states.values() for item in state.attempts)):
        if not item.paid or item.root_task_id not in roots or item.tool_name not in CODE_REVIEW_TOOLS:
            continue
        local = (*_attempt_identity_tuple(item), item.review_retry_key)
        if is_note:
            held.add(local)
        elif local in held:
            continue  # its drive's own row of a wave already noted
        key = item.review_budget_repo_key or owners.get(item.repo_key)
        if key is None:
            key = owners[item.repo_key] = paid_repo_key(ctx, item.repo_key)
        if key == owner:  # a copy of a recorded wave is that wave; a reused attempt number is another
            waves.add(("record", item.review_record_id) if item.review_record_id else (*local, item.started_ts))
    return len(waves), roots

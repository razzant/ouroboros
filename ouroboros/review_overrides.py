"""The owner-visible trace of a review that returned FAIL or no verdict, the author's choice to go on over it and the
commits that landed over it (BIBLE P3).

``record_override`` counts the outcome wherever the gate lets work go on past one (critical findings, an unparseable
answer, a failed check, a quorum or infrastructure failure, every other reason ``tools/review.py`` names), once per
outcome: the ``review_advisory_override`` event, the counter ``state/advisory_overrides.json`` (read by
``review_status``) and the work's card fact ``review_overrides`` on its root task's document, a count and the latest,
never a row per decision. A wave still open is no outcome yet: the call that collects it counts how it ended, and
an author who goes on over it unread or unsettled counts it then, once per review record.
``note_author_continuation`` adds its ``continued`` part and ``note_override_landing`` its ``landed`` part. The card
shows each part only when recorded (absent is unknown, not zero), in the app (``web/modules/review_gate_card.js``,
live from the event) and in its Telegram completion message (``card_line``).

Every outcome returns to the author before Git, so a commit usually lands from a later call naming the
reviewed attempt, which keeps the reason (``review_override_reason``); a wave handed back still open stands at what
its review record holds when the author goes on (nothing once it passed). After Git recorded the commit,
``note_override_landing`` adds it to the counter's ``landed`` list (the newest 64, under the install's data root, with
the author's disposition when it continued) when that attempt or this call met such an outcome. ``landed_line`` names those commits in a restart's or
an adoption's notice ("running" only on an adoption record's word), and ``landed_outcome`` one in an evolution cycle's
notice. These rows know only FAIL or no verdict: none says PASS.
"""
from __future__ import annotations

import logging
import pathlib
import subprocess
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from ouroboros.utils import append_jsonl, read_json_dict, update_json_locked, utc_now_iso

log = logging.getLogger(__name__)

WORK_KEY = "review_overrides"  # on the root task's document; it lives and retires with that record
COUNTER_REL = pathlib.Path("state") / "advisory_overrides.json"
LANDED_KEPT = 64
QUOTED = 80  # the head of an author's reason a card line or a notice quotes (``QUOTED`` in review_gate_card.js)
_FAIL_REASONS = frozenset({"critical_findings", "parse_failure"})  # the panel's critical answers, unparseable ones too
_CHECK_REASONS = frozenset({"preflight", "tests_preflight_blocked"})  # the free checks ahead of the panel
_OPEN = "review_late_result_pending"  # a wave still open: no outcome yet (``review_ledger.reduce_verdict``)
_SPACE = " \t\n\r\f\v"  # the cut drops only these, as ``headOf`` does in the app


def _head(text: Any, limit: int = 300) -> str:
    """At most ``limit`` characters, a cut head ending in an ellipsis that says so (``headOf`` in the app)."""
    text = str(text or "")
    return text if len(text) <= limit else text[:limit - 1].rstrip(_SPACE) + "…"


def _more(part: Any, latest: Dict[str, Any]) -> Dict[str, Any]:
    """A counted part of the work's fact (``{count, latest}``) after one more."""
    return {"count": int((part if isinstance(part, dict) else {}).get("count") or 0) + 1, "latest": latest}


def _keep_on_attempt(ctx: Any, reason: str, row: Any = None) -> None:
    """The reason on a durable review attempt, this call's or ``row``, the record a later continuation's validated
    reference resolves to (``commit_gate.resolve_commit_review_reference``), a Continue successor's or a restarted
    worker's alike. A wave still open never overwrites a reason the attempt already keeps (a failed free check)."""
    from ouroboros.review_state import make_repo_key, update_state
    from ouroboros.tools.commit_gate import _current_review_tool_name

    if row is None:
        key = {"repo_key": make_repo_key(pathlib.Path(ctx.repo_dir)), "task_id": str(getattr(ctx, "task_id", "") or ""),
               "tool_name": _current_review_tool_name(ctx), "attempt": getattr(ctx, "_current_review_attempt_number", None)}
    else:
        key = {name: getattr(row, name, None) for name in ("repo_key", "task_id", "tool_name", "attempt")}
    if type(key["attempt"]) is not int or key["attempt"] < 1:
        return  # no reviewed attempt yet: only a commit of this call goes over it

    def stamp(state: Any) -> None:
        if (found := state.latest_attempt_for(**key)) is not None and not (reason == _OPEN and found.review_override_reason):
            found.review_override_reason = reason

    update_state(pathlib.Path(ctx.drive_root), stamp)


def _keep_on_work(ctx: Any, change: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """``(root task id, the work's fact after ``change``)``; the fact is empty when no record could hold it."""
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema
    from ouroboros.task_store import patch_document
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tools.commit_gate import resolve_root_task_id

    work, kept = resolve_root_task_id(ctx), {}

    def keep(current: Dict[str, Any]) -> Any:
        if not current.get("status"):
            return None  # never mint a partial row
        require_writable_task_result_schema(current)
        kept[WORK_KEY] = change(dict(current[WORK_KEY]) if isinstance(current.get(WORK_KEY), dict) else {})
        return stamp_task_result_schema({**current, **kept})

    try:
        if work:
            patch_document(canonical_data_root(ctx), work, keep)
    except Exception:
        log.warning("Review trace not kept on the card of work %s", work, exc_info=True)
        return work, {}
    return work, dict(kept.get(WORK_KEY) or {})


def record_override(ctx: Any, blocked_msg: str) -> None:
    """One review outcome, FAIL or no verdict, the gate let the work go on past. A wave still open counts nothing yet
    (``note_author_continuation`` counts it if the author goes on over it before a collecting call does), and the earlier
    attempt this call collects counted its outcome when it met it; a commit this call lands goes over either all the
    same. Each record is best effort and independent: an unwritable store loses that copy, never the
    others or the decision."""
    reason, message = str(getattr(ctx, "_last_review_block_reason", "") or "unknown"), str(blocked_msg or "")
    # Only while this call collects that attempt: the field outlives the call, a later wave of the task is its own.
    held = getattr(ctx, "_pending_review_attempt", None) if getattr(ctx, "_review_reconcile_only", False) else None
    collected = str(getattr(held, "review_override_reason", "") or "")
    try:
        ctx._review_override_this_call = reason  # a commit this same call lands goes over it
        _keep_on_attempt(ctx, reason)  # the reason a later continuation's attempt keeps, when its record cannot say
        if held is not None:
            _keep_on_attempt(ctx, reason, row=held)  # the attempt collected keeps the outcome counted for it
    except Exception:
        log.warning("A later landing over this decision cannot be joined to it", exc_info=True)
    if reason not in (_OPEN, collected):
        _count(ctx, reason, message)


def _count(ctx: Any, reason: str, message: str, record_id: str = "") -> None:
    """Count one outcome: the event, the install counter and the work's fact. ``record_id`` names the review record
    of an outcome counted when the author goes on over it; the same record and reason as the work's latest count once."""
    from ouroboros.tools.review_helpers import cyber_review_authority

    task_id, at, repeat = str(getattr(ctx, "task_id", "") or ""), utc_now_iso(), []
    latest = {"at": at, "reason": reason, "task_id": task_id, "text": _head(message),  # a cut head says it is one
              **({"record_id": record_id} if record_id else {})}

    def change(prior: Dict[str, Any]) -> Dict[str, Any]:
        last = prior.get("latest") if isinstance(prior.get("latest"), dict) else {}
        if record_id and (last.get("record_id"), last.get("reason")) == (record_id, reason):
            repeat.append(True)
            return prior
        return {**prior, **_more(prior, latest)}

    work, fact = _keep_on_work(ctx, change)
    if repeat:
        return
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": at, "type": "review_advisory_override",
            "decision_authority": "cyber_pro" if cyber_review_authority() else "advisory",
            "block_reason": reason, "message_head": message[:600], "task_id": task_id,
            **({"work_task_id": work, WORK_KEY: fact} if fact else {})})  # the open card's live copy
    except Exception:
        log.debug("Failed to emit review_advisory_override event", exc_info=True)

    def bump(current: Dict[str, Any]) -> Dict[str, Any]:
        recent = [*(current.get("recent") or []), {"ts": at, "block_reason": reason, "message_head": message[:300]}]
        return {**current, "count": int(current.get("count") or 0) + 1, "recent": recent[-10:]}

    try:
        update_json_locked(ctx.drive_root / COUNTER_REL, bump)
    except Exception:
        log.warning("Failed to persist advisory override visibility", exc_info=True)


def _went_on_over(ctx: Any, source: Any) -> str:
    """What the reviewed attempt ``source`` stands at as its author goes on over it, as a reason ("" when it met no
    FAIL or no verdict): the reason the gate kept on it, except where that reason was kept before the panel's outcome
    was known (a wave handed back still open, or a failed free check ahead of the panel): then its review record says
    what it holds now. A panel FAIL on the record, or a counted seat's FAIL on a record still open, is gone over as
    that FAIL; otherwise a failed check stands, and an open wave stands at its record's outcome (nothing once passed)."""
    kept, record_id = (str(getattr(source, name, "") or "") for name in ("review_override_reason", "review_record_id"))
    if not record_id or (kept != _OPEN and kept not in _CHECK_REASONS):
        return kept
    try:
        from ouroboros.review_ledger import ledger_root, load_record

        record = load_record(ledger_root(ctx), record_id) or {}
    except Exception:  # unreadable: the attempt stands where the gate last saw it
        log.debug("The review record %s cannot be read", record_id, exc_info=True)
        return kept
    verdict = record.get("verdict")
    if not (isinstance(verdict, dict) and verdict.get("aggregate")):
        return kept
    reason = str(verdict.get("reason") or "")
    counted = {str(row.get("seat_id") or "") for row in record.get("rows") or []
               if isinstance(row, Mapping) and not row.get("additional")}
    if reason == _OPEN and any(seat in counted and value == "FAIL"
                               for seat, value in (verdict.get("per_row") or {}).items()):
        return "critical_findings"  # still open, yet a counted seat's FAIL is already on the record
    if verdict["aggregate"] == "FAIL":
        return reason or "critical_findings"
    if kept in _CHECK_REASONS:
        return kept  # the panel passed, gave no verdict or is still open: the failed check is what stands
    if verdict["aggregate"] == "PASS":
        return ""
    return reason.partition("gate_block:")[2] or reason or kept


def note_author_continuation(ctx: Any, decision: Any) -> None:
    """The author's explicit continuation over the reviewed attempt its validated reference resolved to, when that
    attempt stands at a FAIL or no verdict: the work's ``continued`` part, with the disposition and the head of its
    reason. Best effort like every record here (``_keep_on_work`` logs a failed write); it never changes the gate."""
    source = getattr(ctx, "_author_commit_source", None)
    reason = _went_on_over(ctx, source)
    if reason:
        record_id = str(getattr(source, "review_record_id", "") or "")
        kept = str(getattr(source, "review_override_reason", "") or "")
        if kept == _OPEN or reason != kept:
            # An outcome no call counted (handed back open and read back later or gone on over before it settled, or
            # a panel FAIL behind a failed check): the decision to go on over it counts it, once per review record.
            _count(ctx, reason, f"The author went on over a review standing at {reason.replace('_', ' ')}.", record_id)
            try:
                _keep_on_attempt(ctx, reason, row=source)
            except Exception:
                log.warning("The counted outcome was not kept on its attempt", exc_info=True)
        decision = decision if isinstance(decision, dict) else {}
        latest = {"at": utc_now_iso(), "disposition": str(decision.get("disposition") or ""),
                  "rationale": _head(decision.get("rationale")), "reason": reason,
                  "record_id": record_id}
        _keep_on_work(ctx, lambda fact: {**fact, "continued": _more(fact.get("continued"), latest)})


def note_override_landing(ctx: Any, commit_sha: str) -> None:
    """After Git recorded ``commit_sha``: keep it in ``landed`` and count it on the work's card when the reviewed
    attempt its author continuation's validated reference resolved to stands at a FAIL or no verdict, or this call
    met one. Git already succeeded: nothing here may fail it."""
    try:
        from ouroboros.tool_access import canonical_data_root

        source = getattr(ctx, "_author_commit_source", None)  # the durable attempt, resolved afresh by each call
        reason = _went_on_over(ctx, source) or getattr(ctx, "_review_override_this_call", "")
        if not (reason and commit_sha):
            return
        author, at = getattr(ctx, "_author_commit_record", None), utc_now_iso()  # this call's continuation, if any
        row = {"sha": str(commit_sha), "reason": str(reason), "task_id": str(getattr(ctx, "task_id", "") or ""),
               "at": at, **({"disposition": str(author["disposition"]), "rationale": _head(author.get("rationale"))}
                            if isinstance(author, dict) and author.get("disposition") else {})}

        def land(current: Dict[str, Any]) -> Dict[str, Any]:
            kept = current.get("landed") if isinstance(current.get("landed"), list) else []
            return {**current, "landed": [*kept, row][-LANDED_KEPT:]}

        try:
            update_json_locked(canonical_data_root(ctx) / COUNTER_REL, land)  # where restart and adoption read it
        finally:  # the work's card counts it whatever happened to the counter
            landing = {"sha": row["sha"], "at": at}
            _keep_on_work(ctx, lambda fact: {**fact, "landed": _more(fact.get("landed"), landing)})
    except Exception:
        log.warning("The landing of %s over a review outcome was not recorded", commit_sha, exc_info=True)


def _counted(part: Any) -> int:
    count = part.get("count") if isinstance(part, dict) else None
    return count if type(count) is int and count > 0 else 0  # an absent or malformed part is unknown: no line shows it


def _decision(latest: Any) -> str:
    """The author's disposition and the head of its reason as a line quotes them ("" when neither is recorded)."""
    latest = latest if isinstance(latest, dict) else {}
    why = _head(latest.get("rationale"), QUOTED)
    return " — ".join(part for part in (str(latest.get("disposition") or ""), f'"{why}"' if why else "") if part)


def card_line(fact: Any) -> str:
    """The work card's compact field, twin of ``reviewGateLine`` in the app: each recorded part, "" without one. A part
    the work's own outcome count does not precede names what it went on over itself."""
    fact = fact if isinstance(fact, dict) else {}
    latest, continued = (fact.get(key) if isinstance(fact.get(key), dict) else {} for key in ("latest", "continued"))
    parts = []
    if _counted(fact):
        reason = str(latest.get("reason") or "unknown").replace("_", " ")
        parts.append(f"FAIL or no verdict {fact['count']}× (latest: {reason})")
    if _counted(continued):
        last = continued.get("latest") if isinstance(continued.get("latest"), dict) else {}
        said, over = _decision(last), "" if parts else str(last.get("reason") or "").replace("_", " ")
        parts.append(f"author continued {continued['count']}×" + (f" over {over}" if over else "")
                     + (f" (latest: {said})" if said else ""))
    if landed := _counted(fact.get("landed")):
        over = "it" if parts else "a FAIL or no verdict"
        parts.append(f"landed over {over}: {landed} commit{'' if landed == 1 else 's'}")
    return "Review: " + " · ".join(parts) if parts else ""


def _outcome(row: Dict[str, Any]) -> str:
    """How a landed commit's review ended, named by its recorded reason, and the author's word when the row keeps it."""
    reason, said = str(row.get("reason") or "unknown"), _decision(row)
    kind = ("review FAIL" if reason in _FAIL_REASONS else "check FAIL" if reason in _CHECK_REASONS
            else "review outcome" if reason == "unknown" else "no review verdict when it landed" if reason == _OPEN
            else "no review verdict")
    return f"{kind}: {reason.replace('_', ' ')}" + (f"; author continued: {said}" if said else "")


def _landed_rows(data_dir: Any) -> Optional[Dict[str, Dict[str, Any]]]:
    """The counter's ``landed`` rows by commit; {} when it holds none, None when it cannot be read."""
    try:
        path = pathlib.Path(data_dir) / COUNTER_REL
        counter = read_json_dict(path)
        if counter is None and path.exists():
            return None  # there, but no counter can be read from it
        rows = (counter or {}).get("landed")
    except Exception:
        log.debug("Commits landed over a review outcome cannot be read", exc_info=True)
        return None
    return {str(row.get("sha")): row for row in rows if isinstance(row, dict)} if isinstance(rows, list) else {}


def landed_outcome(data_dir: Any, commit_sha: str) -> Optional[str]:
    """The review outcome ``commit_sha`` landed over; "" when no landed row names it, so "" never means PASS, and None
    when the rows cannot be read."""
    rows = _landed_rows(data_dir)
    if rows is None:
        return None
    row = rows.get(str(commit_sha)) if commit_sha else None
    return _outcome(row) if row else ""


def landed_line(repo_dir: Any, data_dir: Any, *, base: str, target: str, running: bool) -> str:
    """The commits of ``base..target`` Git recorded over a FAIL or no verdict, each with its outcome, as one line (""
    when none is known). Only an adoption record's word says running; anything else leaves deployment unconfirmed. A
    notice never fails over it: an unreadable counter or Git names nothing."""
    landed = _landed_rows(data_dir)
    if not (landed and base and target):
        return ""
    try:
        listed = subprocess.run(["git", "rev-list", str(target), f"^{base}"], cwd=str(repo_dir),
                                capture_output=True, text=True, check=False, timeout=30)
    except Exception:
        log.debug("Commits landed over a review outcome cannot be named", exc_info=True)
        return ""
    named = [f"{sha[:12]} ({_outcome(landed[sha])})"
             for sha in (listed.stdout.split() if listed.returncode == 0 else []) if sha in landed]
    if not named:
        return ""
    return (f"{'Now running' if running else 'Committed'} over a FAIL or no verdict: {', '.join(named)}"
            + ("." if running else "; deployment unconfirmed, as no record yet binds a running process to them."))

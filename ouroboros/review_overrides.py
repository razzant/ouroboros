"""The owner-visible trace of every review decision Blocking would have stopped (BIBLE P3 "Selected enforcement").

``record_override`` is its one writer. Wherever the gate lets work go on past a signal Blocking enforcement would
have stopped it on (critical findings, a quorum or infrastructure failure, a missing advisory provider and every other
reason ``tools/review.py`` names), it appends the ``review_advisory_override`` event, counts it in
``state/advisory_overrides.json`` (read by ``review_status``) and keeps the work's card fact ``review_overrides`` on
the root task's document: a count and the latest decision, never a row per decision. The work's card shows it in the
app (``web/modules/review_gate_card.js``, live from the event) and in its Telegram completion message (``card_line``).

Advisory and Cyber Pro hand every outcome back before Git, so a commit usually lands from a later call naming the
reviewed attempt, in this process or another. The decision's reason stays on that durable attempt
(``review_override_reason``); after Git recorded the commit, ``note_override_landing`` adds it to the counter file's
``landed`` list (the newest 64) when this call, or the attempt its author
continuation's validated reference resolved to, met such a decision. ``landed_line`` names those commits in a
restart's or an adoption's notice, "running" only on an adoption record's word. Nothing here says merged or accepted:
the gate did not prohibit continuation, which is all it decided.
"""
from __future__ import annotations

import logging
import pathlib
import subprocess
from typing import Any, Dict, Tuple

from ouroboros.utils import append_jsonl, read_json_dict, update_json_locked, utc_now_iso

log = logging.getLogger(__name__)

WORK_KEY = "review_overrides"  # on the root task's document; it lives and retires with that record
COUNTER_REL = pathlib.Path("state") / "advisory_overrides.json"
LANDED_KEPT = 64


def _keep_on_attempt(ctx: Any, reason: str) -> None:
    """The reason on this call's durable review attempt, the record a later continuation's validated reference resolves
    to (``commit_gate.resolve_commit_review_reference``), a Continue successor's or a restarted worker's alike."""
    from ouroboros.review_state import make_repo_key, update_state
    from ouroboros.tools.commit_gate import _current_review_tool_name

    attempt = getattr(ctx, "_current_review_attempt_number", None)
    if type(attempt) is not int or attempt < 1:
        return  # no reviewed attempt yet: only a commit of this call goes over it
    key = {"repo_key": make_repo_key(pathlib.Path(ctx.repo_dir)), "task_id": str(getattr(ctx, "task_id", "") or ""),
           "tool_name": _current_review_tool_name(ctx), "attempt": attempt}

    def stamp(state: Any) -> None:
        if (row := state.latest_attempt_for(**key)) is not None:
            row.review_override_reason = reason

    update_state(pathlib.Path(ctx.drive_root), stamp)


def _keep_on_work(ctx: Any, latest: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """``(root task id, the work's fact after this decision)``; the fact is empty when no record could hold it."""
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema
    from ouroboros.task_store import patch_document
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tools.commit_gate import resolve_root_task_id

    work, kept = resolve_root_task_id(ctx), {}

    def keep(current: Dict[str, Any]) -> Any:
        if not current.get("status"):
            return None  # never mint a partial row
        require_writable_task_result_schema(current)
        prior = current.get(WORK_KEY) if isinstance(current.get(WORK_KEY), dict) else {}
        kept.update(count=int(prior.get("count") or 0) + 1, latest=latest)
        return stamp_task_result_schema({**current, WORK_KEY: dict(kept)})

    try:
        if work:
            patch_document(canonical_data_root(ctx), work, keep)
    except Exception:
        log.warning("Review override not kept on the card of work %s", work, exc_info=True)
        return work, {}
    return work, kept


def record_override(ctx: Any, blocked_msg: str) -> None:
    """One decision the gate let continue where Blocking would have stopped it. Each record is best effort and
    independent: an unwritable store loses that copy, never the others or the decision."""
    from ouroboros.config import get_review_enforcement
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    reason, message = str(getattr(ctx, "_last_review_block_reason", "") or "unknown"), str(blocked_msg or "")
    task_id, at = str(getattr(ctx, "task_id", "") or ""), utc_now_iso()
    try:
        ctx._review_override_this_call = reason  # a commit this same call lands goes over it
        _keep_on_attempt(ctx, reason)  # so does a later continuation naming this attempt
    except Exception:
        log.warning("A later landing over this decision cannot be joined to it", exc_info=True)
    head = message if len(message) <= 300 else message[:299].rstrip() + "…"  # a cut head says it is one
    work, fact = _keep_on_work(ctx, {"at": at, "reason": reason, "task_id": task_id, "text": head})
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": at, "type": "review_advisory_override", "review_enforcement": get_review_enforcement(),
            "decision_authority": "advisory" if review_enforcement_blocks("blocking") else "cyber_pro",
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


def note_override_landing(ctx: Any, commit_sha: str) -> None:
    """After Git recorded ``commit_sha``: keep it in ``landed`` when this call, or the reviewed attempt its author
    continuation's validated reference resolved to, met a decision Blocking would have stopped. Git already
    succeeded: nothing here may fail it."""
    try:
        source = getattr(ctx, "_author_commit_source", None)  # the durable attempt, resolved afresh by each call
        reason = getattr(ctx, "_review_override_this_call", "") or getattr(source, "review_override_reason", "")
        if not (reason and commit_sha):
            return
        row = {"sha": str(commit_sha), "reason": str(reason), "task_id": str(getattr(ctx, "task_id", "") or ""),
               "at": utc_now_iso()}

        def land(current: Dict[str, Any]) -> Dict[str, Any]:
            kept = current.get("landed") if isinstance(current.get("landed"), list) else []
            return {**current, "landed": [*kept, row][-LANDED_KEPT:]}

        update_json_locked(ctx.drive_root / COUNTER_REL, land)
    except Exception:
        log.warning("The landing of %s over a review decision was not recorded", commit_sha, exc_info=True)


def card_line(fact: Any) -> str:
    """The work card's compact field, twin of ``reviewGateLine`` in the app: a count and the latest reason."""
    if not isinstance(fact, dict) or type(fact.get("count")) is not int or fact["count"] < 1:
        return ""
    latest = fact.get("latest") if isinstance(fact.get("latest"), dict) else {}
    reason = str(latest.get("reason") or "unknown").replace("_", " ")
    return f"Review: {fact['count']}× continued where Blocking would have stopped · latest: {reason}"


def landed_line(repo_dir: Any, data_dir: Any, *, base: str, target: str, running: bool) -> str:
    """The commits of ``base..target`` Git recorded over a decision Blocking would have stopped, as one line ("" when
    none is known). Only an adoption record's word says running; anything else leaves deployment unconfirmed. A
    notice never fails over it: an unreadable counter or Git names nothing."""
    try:
        rows = (read_json_dict(pathlib.Path(data_dir) / COUNTER_REL) or {}).get("landed")
        landed = {str(row.get("sha")): row for row in rows if isinstance(row, dict)} if isinstance(rows, list) else {}
        if not (landed and base and target):
            return ""
        listed = subprocess.run(["git", "rev-list", str(target), f"^{base}"], cwd=str(repo_dir),
                                capture_output=True, text=True, check=False, timeout=30)
    except Exception:
        log.debug("Commits landed over a review decision cannot be named", exc_info=True)
        return ""
    named = [f"{sha[:12]} ({str(landed[sha].get('reason') or 'unknown').replace('_', ' ')})"
             for sha in (listed.stdout.split() if listed.returncode == 0 else []) if sha in landed]
    if not named:
        return ""
    return (f"{'Now running' if running else 'Committed'} over a review Blocking would have stopped (the review gate "
            f"did not prohibit continuation): {', '.join(named)}"
            + ("." if running else "; deployment unconfirmed, as no record yet binds a running process to them."))

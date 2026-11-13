"""Late reviewer feedback the host keeps in view in its owning room until the mind releases it.

A paid review that settles after its author can no longer receive it is advice for Ouroboros,
never a decision, a veto or a wake (owner decision on late review delivery, 2026-10-11, #1738):
it waits in the next ordinary turn of the room the work came from, and consciousness reads it
when that room stays quiet: counted in its waiting line and listed on request, since the owner's
09.11.2026 audit decision (item 10, «Брать») replaced one wake pointer per mark. Its record writer (a plan council reply that settled after its task
ended, a left change, result or commit review; marks of the deleted acceptance panel stay as
history) persists the full source first, then calls ``publish_late_review`` once per settled
publication with immutable facts: no live context, no clock.

The publication becomes ONE chronicle ``mark`` signed by the host (``LATE_REVIEW_OPERATION``)
in the room the memory view reads for that work: the room ``dialogue_evidence.own_room_chat``
proves (a project binding, the chat, its ancestors) when it is Main, a registered Project room
or the conversation of Presence-bound work. Another transport chat, the hidden partition or no
chat is never guessed into Main: no mark is placed (``not_activated``) and the full source stays
retained. The mark id is the publication's, so a repeat is ``unchanged``, a released mark stays
released (the journal replays the release after index loss) and the same id with other facts is
a ``conflict`` that writes nothing. The text is the writer's own words, each line and the whole
bounded with a disclosed omission (``utils.truncate_within_limit``); the target is the retained
source itself, read whole from any focus with ``memory_read(source_ref=…)`` unless the writer
names another reader (``source_read``).

``memory_view`` shows its words only in that room; every integrating start counts the others in
its waiting line (``memory_waiting``: how many, in how many rooms, the oldest) and
``memory_read(room_id='late_reviews')`` lists them by room and work. Only the integrating mind
releases it, with a reason (``chronicle_store._rule_mark_release``). Showing it, a tool read or a
mailbox acknowledgement of a notice is not that release.

A left wave's mark also keeps what its record said when it was published (``review.outcome``, from
``outcome_facts``: the verdict word, the findings of each kind, how many asked answers arrived and how
many answers asked of advisers are unreadable, one per adviser and question), and its head says the same in words. The waiting line counts the marks whose
outcome is a ``clean_pass`` and those of a verdict-bearing review that keep no outcome; the list
prints each mark's outcome, reading the review record of an older mark as it stands now
(``mark_outcome``), and what landed of a commit review's change (``mark_landing``). These are facts
beside the words: nothing is released, hidden or ranked by them.

A result review its author finished before (``pending_review=finish``) whose late verdict is a PASS of exactly
the delivered subject also resolves that task's missing-conclusion warning (``resolve_pending_result_review``).

A publication that could not be placed (``failed``) and every duty only the end of its author's
task can discharge stay owed as owner-held rows of the existing obligations table (``OWED_SET``,
immutable facts): a record a running author may still read itself (``owe_late_review``), and a
producer's own duty (``owe_after_end``: a change or commit wave its author left), which the
producer turns into marks once the task's canonical terminal status is recorded. The off-loop
maintenance pass replays those rows (``replay_owed_late_reviews``); a placed, unchanged or
released mark discharges the row, so a replay can neither duplicate nor resurrect it. A plan
reply needs no row of its own: its ask records which seats are still pending, so the maintenance child job
collects ended tasks' pending seats at $0 (``_collect_ended_plan_asks``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import sqlite3
from types import SimpleNamespace
from typing import Any, Dict, Mapping, NamedTuple, Optional

from ouroboros.chronicle_store import LATE_REVIEW_OPERATION, ChronicleStore
from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, WEB_UI_CHAT_ID, is_a2a_chat_id
from ouroboros.utils import truncate_within_limit

log = logging.getLogger(__name__)

# The review a mark names, by its writer's surface; another surface is named as recorded.
SURFACE_NAMES = {"task_acceptance": "result review", "result": "result review", "plan_review": "plan review",
                 "commit_gate": "commit review", "multi_model_review": "change review", "change": "change review",
                 "skill": "skill review"}
LINE_LIMIT = 800  # one line of the writer's words
TEXT_LIMIT = 4000  # the whole mark: the view never degrades a mark, so the bound is set here
OWED_SET = "late_review_marks"  # owner-held rows: id -> {record | left_review, after_end?}


class LateReviewMark(NamedTuple):
    """``saved``, ``unchanged``, ``not_activated`` (the owning room is not proven), ``memory_not_activated``,
    ``conflict``, ``invalid``, ``owed`` (kept for after its author ended) or ``failed`` (an I/O or lock
    failure: the publication stays owed)."""
    status: str
    mark_id: str = ""
    room_id: str = ""
    detail: str = ""


def late_review_mark_id(surface: str, task_id: str, publication_id: str) -> str:
    digest = hashlib.sha256("\0".join((surface, task_id, publication_id)).encode("utf-8")).hexdigest()
    return f"late-review-{digest[:32]}"


def owning_room(drive_root: Any, task_id: str, chat_id: Any) -> Optional[str]:
    """The room the view reads for this work when its ownership is proven, else ``None`` (never a guessed Main)."""
    from ouroboros.dialogue_evidence import own_room_chat, task_room_record  # D15->D06 is lazy-only
    from ouroboros.dialogue_provenance import presence_record_binding
    from ouroboros.memory_inventory import membership_facts

    try:
        own = own_room_chat(SimpleNamespace(task_id=task_id, task_metadata={}, current_chat_id=chat_id), drive_root)
    except (TypeError, ValueError):
        return None
    if own is None:
        return None
    if own == WEB_UI_CHAT_ID or own in membership_facts(drive_root).project_chat_ids:
        return str(own)
    presence = (own != HIDDEN_CHAT_ID and not is_a2a_chat_id(own)
                and presence_record_binding(task_room_record(drive_root, task_id)))
    return str(own) if presence else None


def _facts(record: Any) -> Dict[str, Any]:
    """The record by value, checked: what the writer must name before anything is read or written."""
    if not isinstance(record, Mapping):
        raise ValueError("a late review record is a mapping of immutable facts")
    facts = json.loads(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, default=str))
    for key in ("surface", "task_id", "publication_id"):
        if not isinstance(facts.get(key), str) or not facts[key].strip():
            raise ValueError(f"{key} is a non-empty string")
    lines = facts.get("lines")
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines) or not "".join(lines).strip():
        raise ValueError("lines are the record writer's own settled words, at least one non-empty line")
    ref, read = facts.get("source_ref"), facts.get("source_read")
    if not isinstance(ref, dict) or not ref.get("sha256") or not ref.get("path"):
        raise ValueError("source_ref is the retained full source, with its path and sha256")
    if read is None and ref.get("kind") != "task_source":
        raise ValueError("a source other than a task source names its reader in source_read")
    if read is not None and not (isinstance(read, dict) and read.get("tool")):
        raise ValueError("source_read is a reader call: {tool, arguments}")
    subject = facts.get("subject") or {}
    if not isinstance(subject, dict) or any(isinstance(value, (dict, list)) for value in subject.values()):
        raise ValueError("subject maps names to scalar identity values")
    outcome = facts.get("outcome")
    if outcome is not None and (not isinstance(outcome, dict) or any(isinstance(v, (dict, list)) for v in outcome.values())):
        raise ValueError("outcome maps names to scalar facts of the settled record")
    return {**facts, "subject": subject, "settled_at": str(facts.get("settled_at") or "")}


def _text(facts: Mapping[str, Any]) -> str:
    name = SURFACE_NAMES.get(facts["surface"], facts["surface"])
    settled = f", settled {facts['settled_at']}" if facts["settled_at"] else ""
    head = [f"Late {name} of task {facts['task_id']}{settled}."]
    subject = "; ".join(f"{key}={value}" for key, value in sorted(facts["subject"].items()) if value not in (None, ""))
    if subject:
        head.append(f"Reviewed: {subject}")
    body = [truncate_within_limit(line, LINE_LIMIT) for line in facts["lines"]]
    return truncate_within_limit("\n".join(head + body), TEXT_LIMIT)


def _mark(mark_id: str, room: str, facts: Mapping[str, Any]) -> Dict[str, Any]:
    ref = facts["source_ref"]
    target = {**ref, "task_id": str(ref.get("task_id") or facts["task_id"])}  # readable from any focus
    outcome = facts.get("outcome")
    return {"id": mark_id, "kind": "mark", "room_id": room, "target_ref": target, "text": _text(facts),
            "author": {"kind": "host", "operation": LATE_REVIEW_OPERATION, "surface": facts["surface"],
                       "task_id": facts["task_id"]},
            "scope": "room", "quote": None, "visibility": "full",
            "review": {"surface": facts["surface"], "task_id": facts["task_id"],
                       "publication_id": facts["publication_id"], "settled_at": facts["settled_at"],
                       "subject": facts["subject"], "source_read": facts.get("source_read"),
                       **({"outcome": outcome} if isinstance(outcome, dict) else {})}}


def outcome_facts(record: Any) -> Optional[Dict[str, Any]]:
    """What a review record says at this moment, as the scalars a late mark keeps beside its words: the verdict
    word, how many findings of each kind the record holds, how many of the answers asked of its counted seats
    arrived (one per seat and question; the seat count for a record that keeps no per-question tally) and how
    many answers asked of advisers, heard outside the count, are unreadable (``unheard``: one per adviser and
    question, an adviser that answered one question and not the other included; one per adviser whose row keeps
    no per-question answers). A
    record that is not settled formed no verdict: ``{"verdict": ""}``, never a guessed PASS. A plan council's
    record is asked no verdict question: ``{"advice": True}``, never its missing findings read as none. No record
    at all says nothing (``None``): a mark published without one keeps no outcome, so its list line reads the
    record whenever it can be read."""
    if not isinstance(record, Mapping):
        return None
    if record.get("surface") == "intention":
        return {"advice": True}
    if record.get("state") != "settled":
        return {"verdict": ""}
    from ouroboros.review_ledger import ANSWERED_STATUSES  # D15->D06 is lazy-only

    verdict = record.get("verdict") or {}
    quorum = verdict.get("quorum") or {}
    tallies = [part for part in (quorum.get("parts") or {}).values() if isinstance(part, Mapping)] or [quorum]
    return {"verdict": str(verdict.get("aggregate") or ""),
            **{name: len(verdict.get(f"{name}_findings") or []) for name in ("critical", "advisory", "additional")},
            "asked": sum(int(tally.get("assigned") or 0) for tally in tallies),
            "answered": sum(int(tally.get("responded") or 0) for tally in tallies),
            "unheard": sum(_unheard(row, ANSWERED_STATUSES) for row in record.get("rows") or []
                           if isinstance(row, Mapping) and row.get("additional"))}


def _unheard(row: Mapping[str, Any], answered: Any) -> int:
    """The unreadable answers of one adviser's row: its asked questions without a responded answer, else one
    when the seat itself gave no readable answer (a row that keeps no per-question answers)."""
    parts, answers = row.get("parts") or (), row.get("answers")
    if parts and isinstance(answers, Mapping):
        return sum(1 for part in parts
                   if not (isinstance(answers.get(part), Mapping) and answers[part].get("status") == "responded"))
    return int(row.get("status") not in answered)


def clean_pass(outcome: Any) -> bool:
    """A PASS with nothing left to read: every answer asked of a counted seat arrived, every adviser answered
    and the record holds no finding of any kind. A PASS with notes, a PASS short of an answer, a FAIL and a
    missing verdict are all something else, and so is an outcome kept before advisers were counted: an
    unknown number of unheard advisers is not zero."""
    return (isinstance(outcome, Mapping) and outcome.get("verdict") == "PASS"
            and not any(outcome.get(name) for name in ("critical", "advisory", "additional"))
            and outcome.get("unheard") == 0
            and int(outcome.get("asked") or 0) > 0 and outcome.get("answered") == outcome.get("asked"))


def outcome_words(outcome: Any) -> str:
    """One mark's outcome for the list of late reviews: facts in words, never advice on what to do with it."""
    if not isinstance(outcome, Mapping):
        return "outcome not recorded"
    if outcome.get("advice"):
        return "advice, not a verdict"
    if not outcome.get("verdict"):
        return "no verdict formed"
    kinds = [f"{outcome[name]} {name}" for name in ("critical", "advisory", "additional") if outcome.get(name)]
    short, unheard = int(outcome.get("asked") or 0) - int(outcome.get("answered") or 0), outcome.get("unheard")
    return (str(outcome["verdict"]) + ", " + (", ".join(kinds) if kinds else "no findings")
            + (f", {short} asked answer{'' if short == 1 else 's'} missing" if short > 0 else "")
            + (", adviser answers not recorded" if unheard is None
               else f", {unheard} adviser answer{'' if unheard == 1 else 's'} missing" if unheard else ""))


def mark_outcome(mark: Mapping[str, Any], drive_root: Any = None) -> Optional[Dict[str, Any]]:
    """The outcome of one late-review mark: the facts published with it when they are whole, else (asked with the
    data root) its review record as it reads now. Facts are whole once they say how many advisers went unheard;
    a mark kept before that, or before any outcome was kept, answers from its record, and keeps what it has when
    the record cannot be read. A plan reply is advice by its kind, whatever its council's record holds. ``None``
    when nothing says: the deleted acceptance panel kept no record, and an unreadable record is not a clean one."""
    review = mark.get("review") or {}
    kept = review["outcome"] if isinstance(review.get("outcome"), dict) else None
    if kept is not None and ("unheard" in kept or drive_root is None):
        return kept
    if review.get("surface") == "plan_review":
        return {"advice": True}
    record_id = str((review.get("subject") or {}).get("record_id") or "")
    if drive_root is None or not record_id:
        return kept
    from ouroboros.review_ledger import load_record  # D15->D06 is lazy-only

    try:
        return outcome_facts(load_record(drive_root, record_id)) or kept
    except (OSError, ValueError):
        return kept


def mark_landing(mark: Mapping[str, Any], drive_root: Any) -> Optional[Dict[str, Any]]:
    """What is recorded, as it reads now, of the change a late commit review read: the commit that landed under
    this review (``{"commit": sha, "tree": same|different|unknown}``, the tree's relation to the one reviewed), else
    the pull request a receipt of its task records as merged, named by its address (``{"pr": url}``: a fact about the
    task's receipts, not proof of who merged it or that this change is in it), else the latest receipt of its task
    with its outcome (``{"pr": url, "status": outcome}``: a merge asked for and not recorded as done, ``unknown`` or
    ``queued`` among them; a queued one whose pull request reads otherwise since carries the owner module's
    ``note``), else that neither is recorded (``{}``: no commit under this record and no receipt, which a
    commit under another record of the task or a merge made outside ``pr_merge`` does not show). ``{"unread": True}``
    when the record or the task's result cannot be read: an unread fact is never told as nothing landed. ``None``
    for any other review. Read on request and never kept on the mark: a successor of the task may still land it."""
    review = mark.get("review") or {}
    if review.get("surface") != "commit_gate":
        return None
    from ouroboros.merge_receipts import RECEIPTS_KEY, queued_earlier_note  # the readers below are named lazily only
    from ouroboros.review_doctrine import public_commit_landings
    from ouroboros.review_ledger import load_record
    from ouroboros.task_results import load_task_result

    try:
        record = load_record(drive_root, str((review.get("subject") or {}).get("record_id") or ""))
        if record is None:
            return {"unread": True}
        commits = public_commit_landings(record)
        if commits:
            return {"commit": commits[-1]["commit_sha"], "tree": commits[-1]["reviewed_subject"]}
        task = load_task_result(drive_root, str(review.get("task_id") or ""))
        if not isinstance(task, dict):
            return {"unread": True}
        receipts = [receipt for receipt in task.get(RECEIPTS_KEY) or [] if isinstance(receipt, dict)]
        merged = [receipt for receipt in receipts if (receipt.get("outcome") or {}).get("status") == "merged"]
        latest = (merged or receipts or [None])[-1]
        if latest is None:
            return {}
        address = str((latest.get("repo") or {}).get("url") or f"PR #{latest.get('number')}")
        if merged:
            return {"pr": address}
        note = queued_earlier_note(latest)  # a queued request the pull request's later readback no longer shows
        return {"pr": address, "status": str((latest.get("outcome") or {}).get("status") or "unknown"),
                **({"note": note} if note else {})}
    except Exception:  # the list the mind asked for keeps its other rows when one work's facts cannot be read
        log.warning("Landing facts unavailable for late review mark %s", mark.get("id"), exc_info=True)
        return {"unread": True}


def landing_words(landing: Any) -> str:
    """One late commit review's landing for the list: what is recorded, never a guess about lost work."""
    if not isinstance(landing, Mapping):
        return ""
    if landing.get("unread"):
        return "what landed could not be read"
    if landing.get("commit"):
        relation = {"same": "", "different": "; its tree differs from the reviewed one"}.get(
            str(landing.get("tree") or ""), "; its relation to the reviewed tree is unknown")
        return f"committed as {str(landing['commit'])[:12]} under this review{relation}"
    if landing.get("pr"):
        note = f" ({landing['note']})" if landing.get("note") else ""
        return f"a receipt of its task records {landing['pr']} as {landing.get('status') or 'merged'}{note}"
    return "no commit under this review, no merged receipt of its task"


def publish_late_review(drive_root: Any, record: Mapping[str, Any]) -> LateReviewMark:
    """Keep one settled late review in view in its owning room (module docstring); never raises, never wakes.

    ``record``: ``surface``, ``task_id``, ``publication_id`` (one settled publication: the same id always
    names the same facts), ``chat_id`` (the owning chat, by value), ``lines`` (the writer's settled words:
    its verdict clause, then one line per seat of the whole roster, pending and unknown seats included),
    ``subject`` (scalar identity of what the reviewers read), ``source_ref`` (the retained full source),
    optional ``source_read``, ``settled_at`` and the writer's already-proven ``room_id`` (frozen with its source).
    """
    try:
        facts = _facts(record)
    except ValueError as exc:
        return LateReviewMark("invalid", detail=str(exc))
    mark_id = late_review_mark_id(facts["surface"], facts["task_id"], facts["publication_id"])
    if facts["surface"] == "result" and facts["subject"].get("record_id"):
        try:
            resolve_pending_result_review(drive_root, facts["task_id"], str(facts["subject"]["record_id"]))
        except Exception:  # the mark still carries the verdict; the stored outcome keeps its warning
            log.warning("late result review %s of task %s could not update its outcome", facts["publication_id"],
                        facts["task_id"], exc_info=True)
    try:
        outcome = _publish(pathlib.Path(drive_root), mark_id, facts)
    except Exception as exc:  # a store that cannot serve (StoreError, …): the publication stays owed
        outcome = LateReviewMark("failed", mark_id, "", f"{type(exc).__name__}: {exc}")
    if outcome.status not in {"saved", "unchanged"}:
        log.warning("late review %s of task %s kept no mark in view: %s (%s)", facts["publication_id"],
                    facts["task_id"], outcome.status, outcome.detail)
    _owed(drive_root, mark_id, {"record": facts} if outcome.status == "failed" else None)
    return outcome


# The objective keys ``outcomes._objective_axis`` derives from the review axis; any other key on the objective (its
# ``warning``/``warnings``: ``receipt_absent``, custody) is another producer's independent fact, and so is one of these
# keys another producer set over that derivation (a publication receipt veto, deferred child results, delivery control).
_REVIEW_DERIVED_OBJECTIVE = frozenset({"status", "source", "review_status", "reason", "outcome_tier"})


def _pending_review(row: Mapping[str, Any], record_id: str) -> tuple:
    """``(axes, review, pending)`` of a stored row; ``pending`` is empty unless it names this review."""
    axes = row.get("outcome_axes") if isinstance(row.get("outcome_axes"), dict) else {}
    review = axes.get("review") if isinstance(axes.get("review"), dict) else {}
    pending = review.get("pending_review") if isinstance(review.get("pending_review"), dict) else {}
    return axes, review, pending if pending.get("record_id") == record_id else {}


def resolve_pending_result_review(drive_root: Any, task_id: str, record_id: str) -> bool:
    """A result review its author finished before (``pending_review=finish``) settled after the task ended (owner,
    8 Nov 2026). A PASS of exactly the subject the task delivered supplies the conclusion missing at delivery: the
    ended task's review axis becomes that PASS, with what was stored before kept beside it
    (``pending_finish_resolved``), and the objective takes what ``_objective_axis`` derives from it where it held the
    pending review's derivation; every other objective fact (``receipt_absent``; a key another producer set over that
    derivation, such as a publication receipt veto or deferred child results) is kept. The store's own write applies this
    over the current row, so an outcome write that landed meanwhile is never lost; the existing late-review event
    then carries that current outcome to an open card (``announce_current_outcome``). Any other verdict, or a review
    of another version, leaves the stored outcome and its warning as they were and announces nothing; the late mark
    carries the verdict either way. A replay finds nothing pending and changes nothing. True on a change."""
    from ouroboros import review_ledger as ledger
    from ouroboros.outcomes import _objective_axis
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, write_task_result

    row = ended_task_row(drive_root, task_id) or {}
    if not _pending_review(row, record_id)[2]:
        return False
    record = ledger.load_record(drive_root, record_id) or {}
    subject = str((record.get("subject") or {}).get("diff_sha") or "")
    if str((record.get("verdict") or {}).get("aggregate") or "") != "PASS":
        return False
    changed: list = []

    def resolve(current: Dict[str, Any], _incoming: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        axes, review, pending = _pending_review(current, record_id)
        if str(current.get("status") or "") not in _TRULY_TERMINAL_STATUSES or subject != pending.get("subject_sha"):
            return None  # another version, or resolved or changed meanwhile: nothing is written
        kept = {key: value for key, value in review.items()
                if key not in {"pending_review", "acceptance_decision", "author_finish"}}
        resolved = {**kept, "status": "pass", "aggregate_signals": ["PASS"],
                    "pending_finish_resolved": {key: axes[key] for key in ("review", "objective")
                                                if isinstance(axes.get(key), dict)},
                    "late_verdict": {"record_id": record_id, "aggregate": "PASS", "subject_sha": subject,
                                     "settled_at": str(record.get("ts") or "")}}
        objective = axes.get("objective") if isinstance(axes.get("objective"), dict) else {}
        derived = _objective_axis(review)  # what the pending review gave the objective
        changed.append(True)
        return {"status": current["status"], "outcome_axes": {**axes, "review": resolved, "objective": {
            **{key: value for key, value in objective.items() if key not in _REVIEW_DERIVED_OBJECTIVE},
            **_objective_axis(resolved), **{key: value for key, value in objective.items()
                                             if key in _REVIEW_DERIVED_OBJECTIVE and value != derived.get(key)}}}}

    write_task_result(drive_root, task_id, str(row["status"]), _field_projector=resolve)
    if changed:
        try:
            announce_current_outcome(drive_root, task_id, record_id, subject)
        except Exception:  # the stored outcome stands; an open card reads it at its next history read
            log.warning("late result review %s of task %s: its current outcome was not announced", record_id, task_id,
                        exc_info=True)
    return bool(changed)


# What a finished card's status and cause sentence read of an outcome (``project_dialogue.outcome_phase`` and
# ``_completion_verdict``, their ``log_events.js`` twins ``taskOutcomeSeverity`` and ``taskReasonDetail``): the
# stored axes whole, as the ending row and event carry them (``terminal_projection._project_row``,
# ``agent_task_pipeline``'s ``task_done``), less the late resolution's record of what it replaced; of the row, the
# facts those readers take, of two mappings the one key they read. Never the row's answer, traces or evidence.
_CARD_FACTS = ("task_id", "parent_task_id", "root_task_id", "status", "task_terminal_status", "reason_code",
               "reason_detail", "artifact_status", "review_status", "task_completion", "cancel_origin",
               "terminal_origin", "terminal_plan_review_open", "delegated_runs_unreconciled")
_CARD_PARTS = {"artifact_bundle": "status", "root_phase_checkpoint": "post_task_synthesis"}
_RESOLUTION_FACTS = ("pending_finish_resolved", "late_verdict")


def current_outcome(row: Mapping[str, Any], delivery: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A stored task row's current outcome as its finished card reads it (``_CARD_FACTS``), with an exhausted or
    unknown delivery of its answer merged as the ending row merges it (``terminal_projection._project_row``: a
    row-level warning and the saved-answer pointer, never the task's own axes), the phase ``outcome_phase`` computes
    from them and the row's revision."""
    from ouroboros.outcomes import _merge_objective_warning
    from ouroboros.project_dialogue import outcome_phase  # D15->D17 is lazy-only
    from ouroboros.task_finalization import quiet_finish

    delivery = {key: str(value or "") for key, value in (delivery or {}).items() if key in ("state", "preserved_path")}
    if delivery.get("state") in {"exhausted", "unknown"}:
        stored = dict(row.get("outcome_axes") or {}) if isinstance(row.get("outcome_axes"), dict) else {}
        objective = dict(stored.get("objective") or {}) if isinstance(stored.get("objective"), dict) else {}
        _merge_objective_warning(objective, f"terminal_delivery_{delivery['state']}")
        row = {**row, "outcome_axes": {**stored, "objective": objective}}
    else:
        delivery = {}
    axes = dict(row.get("outcome_axes") or {}) if isinstance(row.get("outcome_axes"), dict) else {}
    if isinstance(axes.get("review"), dict):
        axes["review"] = {key: value for key, value in axes["review"].items() if key not in _RESOLUTION_FACTS}
    answered = (isinstance(row.get("result"), str) and bool(row["result"])) if "result" in row else row.get(
        "_result_text_present") is True  # the answer's existence, as the ending row states it, never its words
    return {**{key: row[key] for key in _CARD_FACTS if row.get(key) not in (None, "")},
            **{key: {part: row[key][part]} for key, part in _CARD_PARTS.items()
               if isinstance(row.get(key), dict) and part in row[key]},
            "_result_text_present": answered, "quiet_effective": quiet_finish(dict(row)), "outcome_axes": axes,
            **({"terminal_delivery": delivery} if delivery else {}),
            "outcome_phase": outcome_phase(dict(row), {}), "revision": str(row.get("updated_at") or "")}


def late_current_outcome(drive_root: Any, row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The current outcome of an ended task a late review revised (``late_verdict``), with its answer's delivery
    as it stands (``terminal_projection._delivery_disposition``), else ``None``: the card's and a reload's one
    reading (``announce_current_outcome``; history's task carriers and replay evidence)."""
    review = ((row or {}).get("outcome_axes") or {}).get("review") if isinstance((row or {}).get("outcome_axes"), dict) else None
    if not isinstance(review, dict) or not isinstance(review.get("late_verdict"), dict):
        return None
    from ouroboros.terminal_projection import _delivery_disposition  # D15->D17 is lazy-only

    try:
        delivery = _delivery_disposition(drive_root, str(row.get("task_id") or ""))
    except Exception:  # an unreadable delivery registry is unknown, as the ending row reads it
        delivery = {"state": "unknown"}
    return current_outcome(row, delivery)


def announce_current_outcome(drive_root: Any, task_id: str, record_id: str, subject: str) -> Dict[str, Any]:
    """The existing late-review event (``review_late_result``) with the task's current outcome after a late PASS
    resolved it, bound to the exact task, record and subject, so an open finished card repaints what a reload of the
    stored row reads. That outcome is the row as the store holds it now, read here, never the resolver's earlier copy:
    a write that landed after the resolution (a custody warning) is what the card repaints. Its ``revision`` is that
    row's, and the card orders it by that revision, never by this event's later ``ts`` (``log_events.js``
    ``lateOutcomeIsCurrent``). An unreadable row raises: nothing is announced, and the card reads the stored outcome
    at its next history read. The durable append lands first; the one live frame is pushed where a bridge exists (the
    server process, as ``post_task_checkpoint`` pushes a finalized cost; the log sink's copy of this type is
    suppressed). Never a chat row, a task summary or a notification."""
    from ouroboros.utils import append_jsonl, utc_now_iso

    row = ended_task_row(drive_root, task_id, strict=True) or {}
    event = {"ts": utc_now_iso(), "type": "review_late_result", "task_id": str(task_id), "surface": "result",
             "record_id": str(record_id), "subject_sha": str(subject), "status": "settled", "aggregate": "PASS",
             # Its audience by value, as a worker's recorded row is stamped: the task left RUNNING long ago.
             **{key: row[key] for key in ("chat_id", "root_task_id", "parent_task_id") if row.get(key) not in (None, "")},
             "current_outcome": late_current_outcome(drive_root, row)}
    append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", event)
    from supervisor.message_bus import try_get_bridge  # D15->D08 is lazy-only

    bridge = try_get_bridge()
    if bridge is not None:
        from supervisor.log_addressing import address_handler_push

        bridge.push_log(address_handler_push(pathlib.Path(drive_root), dict(event)))
    return event


def owe_late_review(drive_root: Any, record: Mapping[str, Any]) -> LateReviewMark:
    """Keep one late review owed until its author's task ended, then publish it (``replay_owed_late_reviews``).

    For feedback a running author may still read itself: reading it is not a release, so the mark
    follows the task's end whatever the author read. An already ended task publishes now.
    """
    try:
        facts = _facts(record)
    except ValueError as exc:
        return LateReviewMark("invalid", detail=str(exc))
    mark_id = late_review_mark_id(facts["surface"], facts["task_id"], facts["publication_id"])
    if ended_task_row(drive_root, facts["task_id"]) is not None:
        return publish_late_review(drive_root, facts)
    return LateReviewMark("owed" if _owed(drive_root, mark_id, {"record": facts, "after_end": True}) else "failed",
                          mark_id)


def replay_settled_late_review(drive_root: Any, mark_id: str) -> Optional[LateReviewMark]:
    """After task end, reuse a publication's frozen source/room instead of composing it from changed facts."""
    from ouroboros import store
    from ouroboros.chronicle_store import is_late_review_mark

    try:
        mark = ChronicleStore(drive_root).get(mark_id)  # includes released marks: a replay cannot resurrect one
        if mark is not None:
            _owed(drive_root, mark_id, None)
            return LateReviewMark("unchanged" if is_late_review_mark(mark) else "conflict", mark_id, mark.get("room_id", ""))
        with store.open_store(drive_root, write=False) as txn:
            owed = txn.get_row(OWED_SET, mark_id) or {}
    except (OSError, ValueError, TimeoutError, sqlite3.Error) as exc:
        return LateReviewMark("failed", mark_id, "", f"frozen settlement unreadable: {type(exc).__name__}: {exc}")
    return publish_late_review(drive_root, owed["record"]) if isinstance(owed.get("record"), dict) else None


def owe_after_end(drive_root: Any, duty_id: str, duty: Mapping[str, Any], *, replace: bool = True) -> bool:
    """Persist one producer duty (``left_review``: immutable facts naming ``task_id``) that only its author's
    ended task can discharge; the replay hands it back to its producer. False when the store refused (the
    caller's own path still holds the duty). ``replace=False`` keeps an existing duty: unread-only feedback
    cannot weaken a left wave's independent room obligation."""
    return _owed(drive_root, duty_id, {**json.loads(json.dumps(dict(duty), default=str)), "after_end": True},
                 replace=replace)


def update_wait_cursors(drive_root: Any, duty_id: str, waiting: Mapping[str, Any]) -> None:
    """Refresh an existing duty's delivery facts; never discharge or recreate it."""
    from ouroboros import store

    with store.open_store(drive_root, write=True) as txn:
        row = txn.get_row(OWED_SET, duty_id)
        if row is None or not isinstance(row.get("left_review"), dict):
            return
        row["left_review"].update({key: waiting[key] for key in
                                  ("selection", "seen_helpers", "seen_outcomes", "review_reference") if key in waiting})
        txn.put_row(OWED_SET, duty_id, row)


def replay_owed_late_reviews(drive_root: Any, *, stop: Any = None) -> Dict[str, int]:
    """The maintenance pass over owed rows: each duty whose author ended is published or handed back to its
    producer; a row that names nothing it can owe is retired. Ended plan asks belong to the
    separate maintenance child job, which uses their pending seats as debt."""
    from ouroboros import store  # D15->D17 is lazy-only

    report = {"settled": 0, "waiting": 0, "retired": 0, "failed": 0}
    try:
        with store.open_store(drive_root, write=False) as txn:
            rows = txn.set_rows(OWED_SET)
    except Exception:
        log.warning("owed late reviews are unreadable; they stay owed", exc_info=True)
        return report
    for duty_id, row in rows.items():
        if stop is not None and stop():
            break
        duty = next((row[key] for key in ("record", "left_review") if isinstance(row.get(key), dict)), None)
        if duty is None:
            report["retired"] += _owed(drive_root, duty_id, None)
        elif row.get("after_end") and ended_task_row(drive_root, str(duty.get("task_id") or "")) is None:
            report["waiting"] += 1
        else:
            report[_settle_owed(drive_root, duty_id, row)] += 1
    return report


def _collect_ended_plan_asks(drive_root: Any, stop: Any, *, candidates=None, on_examined=None) -> int:
    """The crash window of a plan reply: a seat of an ended task whose producer completed while nobody was
    listening settles at $0 through the organ's own collector (``plan_ask.collect_pending``), whose settlement
    keeps the reply in its room. The ask's pending seat is the duty, so nothing else is written for it; a seat
    with no retained outcome stays pending and is never cleaned. Only contention keeps an examination
    owed; deterministic failures rest until the task changes, without discharging any pending seat."""
    from ouroboros.maintenance import _contention  # D15->D17 is lazy-only
    from ouroboros import plan_record  # D15->D06 is lazy-only
    from ouroboros.task_store import plan_collection_candidates  # D15->D17 is lazy-only
    from ouroboros.tools.plan_ask import collect_pending  # D15->D06 is lazy-only

    root, settled = pathlib.Path(drive_root), 0
    try:
        for task_id in plan_collection_candidates(root, stop=stop) if candidates is None else candidates:
            if stop is not None and stop():
                break
            try:
                record = plan_record.load_plan_record(root, task_id)
                pending = {(ask.get("ask_id"), seat.get("seat_id")) for ask, seat in plan_record.pending_participants(record)}
                if pending and ended_task_row(root, task_id, strict=True) is not None:
                    after = collect_pending(None, record, state_root=root, task_id=task_id)
                    settled += sum(1 for ask in after.get("asks") or [] for seat in ask.get("participants") or []
                                   if (ask.get("ask_id"), seat.get("seat_id")) in pending and seat.get("status") != "pending")
            except Exception as exc:
                log.warning("plan record of %s: its pending seats stay pending", task_id, exc_info=True)
                if _contention(exc):
                    continue
            if on_examined is not None:
                on_examined(task_id)
    except Exception:
        log.warning("plan records are unreadable; their pending seats stay pending", exc_info=True)
    return settled


def _settle_owed(drive_root: Any, duty_id: str, row: Mapping[str, Any]) -> str:
    """One owed row whose author ended: ``settled``, ``waiting`` (its producer still owns it) or ``failed``."""
    try:
        if isinstance(row.get("record"), dict):
            status = publish_late_review(drive_root, row["record"]).status
        else:
            from ouroboros.review_pause import settle_left_review  # D15->D06 is lazy-only

            status = settle_left_review(drive_root, row)
    except Exception:
        log.warning("owed late review %s could not be settled; it stays owed", duty_id, exc_info=True)
        return "failed"
    if status == "waiting":
        return "waiting"
    if status == "failed":
        return "failed"
    _owed(drive_root, duty_id, None)  # a placed, unchanged, released or unplaceable outcome discharges it
    return "settled"


def _owed(drive_root: Any, mark_id: str, row: Optional[Dict[str, Any]], *, replace: bool = True) -> bool:
    """Write (``row``) or discharge (``None``) one owed publication; False when the store refused."""
    from ouroboros import store  # D15->D17 is lazy-only

    try:
        if row is None:
            with store.open_store(drive_root, write=False) as txn:
                if txn.get_row(OWED_SET, mark_id) is None:
                    return True
        with store.open_store(drive_root, write=True) as txn:
            if row is None:
                txn.delete_row(OWED_SET, mark_id)
            elif replace or txn.get_row(OWED_SET, mark_id) is None:
                txn.put_row(OWED_SET, mark_id, row)
        return True
    except Exception:
        log.warning("late review %s: its owed row could not be %s", mark_id,
                    "kept; nothing durable owes it now" if row is not None else "discharged", exc_info=True)
        return False


def ended_task_row(drive_root: Any, task_id: str, *, strict: bool = False) -> Optional[Dict[str, Any]]:
    """The canonical row of a task whose terminal status is recorded (a latch), else ``None``; unreadable is
    not ended. Strict readers propagate failures for the caller's retry policy. A writer reads its owning chat here."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result  # D15->D17 is lazy-only

    try:
        row = load_task_result(drive_root, task_id) or {}
    except Exception:
        if strict:
            raise
        log.debug("task %s status unreadable; an owed late review keeps waiting", task_id, exc_info=True)
        return None
    return row if str(row.get("status") or "") in _TRULY_TERMINAL_STATUSES else None


def _publish(root: pathlib.Path, mark_id: str, facts: Mapping[str, Any]) -> LateReviewMark:
    try:
        room = facts["room_id"] if "room_id" in facts else owning_room(root, facts["task_id"], facts.get("chat_id"))
    except (OSError, ValueError, sqlite3.Error) as exc:  # an unreadable binding or Project registry
        return LateReviewMark("failed", mark_id, "", f"owning room unreadable: {type(exc).__name__}: {exc}")
    if room is None:
        return LateReviewMark("not_activated", mark_id, "", (
            f"the room owning chat {facts.get('chat_id')!r} of task {facts['task_id']} is not proven; no mark is "
            f"placed and the full source stays retained at {facts['source_ref']['path']}"))
    try:
        store = ChronicleStore(root)
        receipt = store.ensure_activated()
        if not isinstance(receipt, dict) or receipt.get("kind") != "activation":
            kind = str((receipt or {}).get("kind") or "import_failed")
            return LateReviewMark("failed" if kind == "import_pending" else "memory_not_activated", mark_id, room,
                                  f"{kind}: {(receipt or {}).get('reason') or 'no activation receipt'}")
        result = store.publish([_mark(mark_id, room, facts)])
    except (OSError, TimeoutError, sqlite3.Error) as exc:
        return LateReviewMark("failed", mark_id, room, f"{type(exc).__name__}: {exc}")
    except ValueError as exc:  # the store's own refusal to open its journal
        return LateReviewMark("memory_not_activated", mark_id, room, f"{type(exc).__name__}: {exc}")
    if result.ok:
        return LateReviewMark(result.reason, mark_id, room)
    return LateReviewMark("conflict" if result.detail.startswith("identity collision") else "invalid", mark_id, room,
                          f"{result.reason}: {result.detail}")

"""Author-check facts that never become a review verdict.

``review_replaced_by_subagents`` is the host's record that the author claimed a
helper checked this candidate. The host fills the reference, subject, evidence
digests, actor identities, seat states and time. A missing or unverifiable
report is a gap, never a fabricated verification. The fact is not an input to
``reduce_verdict`` and cannot create PASS or erase a later finding.

Cyber Pro also freezes ``evidence_at_decision`` when the author chooses.
The ``landing`` field (kind ``commit_landing``) is a different fact, written only after Git succeeds,
as is the record's ``commit_landings``: each commit that landed under it, against the reviewed tree,
with the author decision that governed that commit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

FACT_KIND = "review_replaced_by_subagents"
EVIDENCE_KIND = "evidence_at_decision"
LANDING_KEY = "landing"
LANDING_KIND = "commit_landing"
COMMIT_LANDINGS_KEY = "commit_landings"
_LANDING_TEXT = ("commit_sha", "tree_sha", "parent_sha", "author_subject", "author_disposition", "author_reason",
                 "recorded_at")
_RELATIONS = frozenset({"same", "different", "unknown"})
_OBJECT_NAME = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")  # a full SHA-1 or SHA-256 Git object name
_FEEDBACK = frozenset({"none_received", "received", "unavailable", "unknown"})
_GAPS = frozenset({"report_unreadable", "subject_mismatch", "digest_mismatch", "actor_identity_unknown"})
_BILLING = frozenset({"subscription", "measured", "unknown"})
_LIVE = frozenset({"in_flight", "pending_dispatch"})
_ANSWERED = frozenset({"responded", "ok", "empty", "parse_failure"})
_REASON_LIMIT = 8_000
_HEX64 = re.compile(r"[0-9a-f]{64}")
_REF_KEYS = ("task_id", "run_id", "subagent_id", "subject_hash", "report_sha256")

AUTHOR_DISPOSITION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "disposition": {"type": "string", "enum": ["accepted", "rejected", "partial", "deferred"]},
        "rationale": {"type": "string"},
        "subagent_check": {
            "type": "object",
            "additionalProperties": False,
            "description": (
                "Optional author evidence that a helper checked this candidate. "
                "Not a review seat and not a PASS. Blocking still needs a counted PASS."
            ),
            "properties": {
                "reason": {"type": "string"},
                "selection_rationale": {"type": "string"},
                "report_refs": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {key: {"type": "string"} for key in _REF_KEYS},
                }},
            },
            "required": ["reason", "report_refs", "selection_rationale"],
        },
    },
    "required": ["disposition", "rationale"],
    "description": (
        "Explicitly accept the current attributed candidate after reading the referenced outcome. "
        "Never overrides Blocking. Optional subagent_check is author evidence, not a seat and not a PASS."
    ),
}


def _invalid(detail: str) -> ValueError:
    return ValueError(f"AUTHOR_DISPOSITION_INVALID: {detail}")


def _plain(value: Any, *, limit: int = _REASON_LIMIT) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        raise _invalid("a required reason is empty")
    if len(text) > limit:
        raise _invalid("a required reason is too large")
    return text


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def parse_report_ref(value: Any) -> Dict[str, str]:
    """One closed report pointer. At least a task or a run; no extra fields."""
    if not isinstance(value, dict) or set(value) - set(_REF_KEYS):
        raise _invalid("report_refs item fields are invalid")
    ref: Dict[str, str] = {}
    for key in _REF_KEYS:
        if key not in value:
            continue
        text = str(value.get(key) or "").strip()
        if not text:
            continue
        if key == "report_sha256":
            text = text.lower()
            if not _HEX64.fullmatch(text):
                raise _invalid("report_sha256 is not a sha256 digest")
        ref[key] = text
    if not (ref.get("task_id") or ref.get("run_id")):
        raise _invalid("a report ref needs a task_id or a run_id")
    return ref


def parse_subagent_check(value: Any) -> Dict[str, Any]:
    """The author's optional claim. Empty evidence is an invalid envelope."""
    if not isinstance(value, dict) or set(value) - {"reason", "report_refs", "selection_rationale"}:
        raise _invalid("subagent_check fields are invalid")
    refs = value.get("report_refs")
    if not isinstance(refs, list) or not refs:
        raise _invalid("subagent_check needs at least one report ref")
    return {
        "reason": _plain(value.get("reason")),
        "selection_rationale": _plain(value.get("selection_rationale")),
        "report_refs": [parse_report_ref(item) for item in refs],
    }


def iter_seats(actors: Any) -> Iterable[Dict[str, Any]]:
    """Reviewer rows only. A coupling aggregate is not itself a seat."""
    rows = [row for row in actors or [] if isinstance(row, dict)]
    while rows:
        row = rows.pop()
        children = row.get("raw_results")
        if isinstance(children, list) and children and not (row.get("slot_id") or row.get("seat_id")):
            rows.extend(item for item in children if isinstance(item, dict))
            continue
        if row.get("slot_id") or row.get("seat_id") or "operation_state" in row:
            yield row


def seat_state(row: Dict[str, Any]) -> Dict[str, Any]:
    """One seat as it was when the author decided. A live row stays live even with an error."""
    operation = str(row.get("operation_state") or "")
    status = str(row.get("status") or "")
    live = operation in _LIVE
    has_answer = (not live) and (
        row.get("ok") is True or status in _ANSWERED
        or bool(row.get("parsed") or row.get("raw_text") or row.get("answers") or row.get("parsed_items"))
    )
    unavailable = (not live) and (not has_answer) and (
        operation == "custody_lost" or bool(row.get("error")) or bool(row.get("failure_code"))
    )
    ok = row.get("ok")
    return {
        "slot_id": str(row.get("slot_id") or row.get("seat_id") or ""),
        "model": str(row.get("model") or row.get("model_id") or ""),
        "operation_state": operation,
        "status": status,
        "ok": ok if isinstance(ok, bool) else None,
        "has_answer": bool(has_answer),
        "unavailable": bool(unavailable),
    }


def classify_feedback(actors: Any) -> str:
    """Descriptive label for the seats visible at the decision. Not a verdict."""
    seats = [seat_state(row) for row in iter_seats(actors)]
    if not seats:
        return "unknown"
    if any(seat["has_answer"] for seat in seats):
        return "received"
    if any(seat["unavailable"] for seat in seats):
        return "unavailable"
    if all(seat["operation_state"] in _LIVE or seat["status"] == "pending" for seat in seats):
        return "none_received"
    return "unknown"


def _seats_copy(seats: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(seats, list):
        return None
    copied = []
    for seat in seats:
        if not isinstance(seat, dict):
            return None
        try:
            copied.append(_copy(seat))
        except (TypeError, ValueError):
            return None
    return copied


def validate_evidence_snapshot(value: Any, *, subject_hash: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict) or value.get("kind") != EVIDENCE_KIND:
        return None
    feedback = str(value.get("feedback_at_decision") or "")
    subject = str(value.get("subject_hash") or "").strip()
    observed = str(value.get("observed_at") or "").strip()
    seats = _seats_copy(value.get("seats"))
    expected = str(subject_hash or "").strip()
    if feedback not in _FEEDBACK or not subject or not observed or seats is None:
        return None
    if expected and subject != expected:
        return None
    try:
        revision = int(value.get("revision") or 0)
    except (TypeError, ValueError):
        return None
    return {
        "kind": EVIDENCE_KIND,
        "feedback_at_decision": feedback,
        "record_id": str(value.get("record_id") or ""),
        "revision": revision,
        "subject_hash": subject,
        "observed_at": observed,
        "seats": seats,
    }


def reuse_snapshot(prior: Any, subject_hash: str) -> Optional[Dict[str, Any]]:
    """Keep the decision-time label. Later replies do not rewrite it."""
    if not isinstance(prior, dict):
        return None
    checked = validate_evidence_snapshot(prior.get(EVIDENCE_KIND), subject_hash=subject_hash)
    return copy.deepcopy(checked) if checked is not None else None


def capture_evidence_snapshot(actors: Any, *, record_id: str, revision: int, subject_hash: str,
                              observed_at: str, prior: Any = None) -> Dict[str, Any]:
    reused = reuse_snapshot(prior, subject_hash)
    if reused is not None:
        return reused
    return {
        "kind": EVIDENCE_KIND,
        "feedback_at_decision": classify_feedback(actors),
        "record_id": str(record_id or ""),
        "revision": int(revision or 0),
        "subject_hash": str(subject_hash or ""),
        "observed_at": str(observed_at or ""),
        "seats": [seat_state(row) for row in iter_seats(actors)],
    }


def _digest(body: Any) -> str:
    if isinstance(body, str):
        raw = body.encode("utf-8")
    elif isinstance(body, (dict, list)) and body:
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    else:
        return ""
    return hashlib.sha256(raw).hexdigest() if raw else ""


def _billing(row: Dict[str, Any]) -> str:
    label = str(row.get("cost_kind") or row.get("billing") or "").strip().lower()
    return label if label in _BILLING else "unknown"


def _result_subject(row: Dict[str, Any]) -> str:
    direct = str(row.get("subject_hash") or "").strip()
    if direct:
        return direct
    subject = row.get("subject")
    if isinstance(subject, dict):
        return str(subject.get("snapshot_hash") or subject.get("subject_hash") or "").strip()
    return ""


def _load_task(drive_root: Any, task_id: str) -> Any:
    from ouroboros.task_results import load_task_result

    try:
        return load_task_result(drive_root, task_id)
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _gap(ref: Dict[str, str], gap: str, **extra: str) -> Dict[str, Any]:
    body: Dict[str, Any] = {"ref": dict(ref), "status": "gap", "gap": gap}
    body.update(extra)
    return body


def _bind_report(drive_root: Any, ref: Dict[str, str], subject_hash: str) -> Dict[str, Any]:
    task_id = ref.get("task_id") or ""
    row = _load_task(drive_root, task_id) if task_id else None
    if not isinstance(row, dict):
        return _gap(ref, "report_unreadable")
    claimed_run = ref.get("run_id") or ""
    host_run = str(row.get("run_id") or row.get("current_run_id") or "").strip()
    if claimed_run and host_run != claimed_run:
        return _gap(ref, "report_unreadable")
    claimed_subject = ref.get("subject_hash") or ""
    host_subject = _result_subject(row)
    if (claimed_subject and claimed_subject != subject_hash) or (host_subject and host_subject != subject_hash):
        return _gap(ref, "subject_mismatch")
    digest = _digest(row.get("result"))
    if not digest:
        return _gap(ref, "report_unreadable")
    claimed_digest = ref.get("report_sha256") or ""
    if claimed_digest and claimed_digest != digest:
        return _gap(ref, "digest_mismatch", host_sha256=digest)
    configured = row.get("configured_subagent") or {}
    host_actor = str(row.get("subagent_id") or configured.get("selected_subagent_id") or "").strip()
    claimed_actor = ref.get("subagent_id") or ""
    report = {
        "ref": dict(ref),
        "status": "bound",
        "actor_identity": "mismatch" if claimed_actor and claimed_actor != host_actor else "recorded",
        "evidence_sha256": digest,
        "actor": {
            "task_id": str(row.get("task_id") or task_id),
            "run_id": host_run or claimed_run,
            "subagent_id": host_actor,
            "model": str(row.get("model") or row.get("model_id") or ""),
            "billing": _billing(row),
        },
    }
    if not host_actor:
        report.update(status="gap", gap="actor_identity_unknown", actor_identity="unknown")
        report.pop("evidence_sha256")
    return report


def _claim_key(reason: str, selection: str, refs: List[Dict[str, str]]) -> str:
    return json.dumps(
        {"reason": reason, "selection_rationale": selection, "report_refs": refs},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def build_fact(claim: Dict[str, Any], *, subject_hash: str, review_reference: Any, drive_root: Any,
               actors: Any, observed_at: str) -> Dict[str, Any]:
    reference = review_reference if isinstance(review_reference, dict) else {}
    try:
        reference = _copy(reference)
    except (TypeError, ValueError):
        reference = {}
    return {
        "kind": FACT_KIND,
        "reason": claim["reason"],
        "selection_rationale": claim["selection_rationale"],
        "review_reference": reference,
        "subject_hash": str(subject_hash or ""),
        "observed_at": str(observed_at or ""),
        "reports": [_bind_report(drive_root, ref, str(subject_hash or "")) for ref in claim["report_refs"]],
        "seats": [seat_state(row) for row in iter_seats(actors)],
    }


def _report_ok(report: Dict[str, Any], subject_hash: str) -> bool:
    status = report.get("status")
    ref = report.get("ref")
    if not isinstance(ref, dict) or set(ref) - set(_REF_KEYS):
        return False
    claimed = str(ref.get("subject_hash") or "")
    if status == "bound":
        digest = str(report.get("evidence_sha256") or "")
        actor = report.get("actor")
        return bool(_HEX64.fullmatch(digest) and isinstance(actor, dict)
                    and "gap" not in report and (not claimed or claimed == subject_hash))
    if status != "gap" or report.get("gap") not in _GAPS or "evidence_sha256" in report:
        return False
    host = report.get("host_sha256")
    if report.get("gap") == "digest_mismatch":
        return isinstance(host, str) and bool(_HEX64.fullmatch(host))
    return "host_sha256" not in report


def validate_replacement_fact(value: Any, *, subject_hash: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict) or value.get("kind") != FACT_KIND:
        return None
    subject = str(value.get("subject_hash") or "").strip()
    observed = str(value.get("observed_at") or "").strip()
    reports = value.get("reports")
    seats = _seats_copy(value.get("seats"))
    expected = str(subject_hash or "").strip()
    if not subject or not observed or not isinstance(reports, list) or not reports or seats is None:
        return None
    if expected and subject != expected:
        return None
    try:
        reason = _plain(value.get("reason"))
        selection = _plain(value.get("selection_rationale"))
        copied_reports = [_copy(report) for report in reports]
    except (TypeError, ValueError):
        return None
    if any(not isinstance(report, dict) or not _report_ok(report, subject) for report in copied_reports):
        return None
    reference = value.get("review_reference")
    if not isinstance(reference, dict):
        return None
    try:
        reference = _copy(reference)
    except (TypeError, ValueError):
        return None
    return {
        "kind": FACT_KIND, "reason": reason, "selection_rationale": selection,
        "review_reference": reference, "subject_hash": subject, "observed_at": observed,
        "reports": copied_reports, "seats": seats,
    }


def reuse_fact(prior: Any, claim: Dict[str, Any], subject_hash: str) -> Optional[Dict[str, Any]]:
    if not isinstance(prior, dict):
        return None
    checked = validate_replacement_fact(prior.get(FACT_KIND), subject_hash=subject_hash)
    if checked is None:
        return None
    current = _claim_key(claim["reason"], claim["selection_rationale"], claim["report_refs"])
    stored_refs = [report.get("ref") for report in checked["reports"]]
    stored = _claim_key(checked["reason"], checked["selection_rationale"], stored_refs)
    return copy.deepcopy(checked) if current == stored else None


def validate_landing(value: Any, *, subject_hash: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict) or value.get("kind") != LANDING_KIND or value.get("landed") is not True:
        return None
    subject = str(value.get("subject_hash") or "").strip()
    expected = str(subject_hash or "").strip()
    if not subject or (expected and subject != expected):
        return None
    return {
        "kind": LANDING_KIND, "landed": True, "subject_hash": subject,
        "pre_review_fingerprint": str(value.get("pre_review_fingerprint") or ""),
        "post_review_fingerprint": str(value.get("post_review_fingerprint") or ""),
    }


def carry_recorded_doctrine(normalized: Dict[str, Any], record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Keep a valid fact, snapshot and landing. A bad attachment rejects the record."""
    subject = str(normalized.get("subject_hash") or "")
    for key, checker in (
        (FACT_KIND, validate_replacement_fact),
        (EVIDENCE_KIND, validate_evidence_snapshot),
        (LANDING_KEY, validate_landing),
    ):
        if key not in record:
            continue
        checked = checker(record.get(key), subject_hash=subject)
        if checked is None:
            return None
        normalized[key] = checked
    return normalized


def enrich_author_commit(author: Dict[str, Any], *, actors: Any, prior_decision: Any, record_id: str,
                         revision: int, drive_root: Any, cyber: bool, observed_at: str,
                         review_reference: Any) -> Dict[str, Any]:
    """Attach the host facts. The claim itself is not stored as verification."""
    body = dict(author)
    claim = body.pop("subagent_check", None)
    subject = str(body.get("subject_hash") or "")
    if cyber:
        body[EVIDENCE_KIND] = capture_evidence_snapshot(
            actors, record_id=record_id, revision=revision, subject_hash=subject,
            observed_at=observed_at, prior=prior_decision,
        )
    if isinstance(claim, dict):
        reused = reuse_fact(prior_decision, claim, subject)
        body[FACT_KIND] = reused if reused is not None else build_fact(
            claim, subject_hash=subject, review_reference=review_reference, drive_root=drive_root,
            actors=actors, observed_at=observed_at,
        )
    return body


_LEDGER_KEYS = ("subject_hash", "reviewer_signal", "enforcement", "recorded_at", "source", "action")


def decision_for_ledger(decision: Any, record_id: str) -> Dict[str, Any]:
    """The pre-Git decision. Landing is never part of this write."""
    decision = decision if isinstance(decision, dict) else {}
    body: Dict[str, Any] = {
        "disposition": str(decision.get("disposition") or ""),
        "rationale": str(decision.get("rationale") or ""),
        "reused_record_id": str(decision.get("reused_record_id") or record_id),
    }
    for key in _LEDGER_KEYS:
        if decision.get(key) not in ("", None):
            body[key] = decision[key]
    reference = decision.get("review_reference")
    if isinstance(reference, dict):
        try:
            body["review_reference"] = _copy(reference)
        except (TypeError, ValueError):
            pass
    subject = str(decision.get("subject_hash") or "")
    fact = validate_replacement_fact(decision.get(FACT_KIND), subject_hash=subject) if FACT_KIND in decision else None
    if fact is not None:
        body[FACT_KIND] = fact
    snapshot = validate_evidence_snapshot(decision.get(EVIDENCE_KIND), subject_hash=subject) if EVIDENCE_KIND in decision else None
    if snapshot is not None:
        body[EVIDENCE_KIND] = snapshot
    return body


def attach_landing(author: Dict[str, Any], *, pre_review_fingerprint: str, post_review_fingerprint: str) -> Dict[str, Any]:
    """Record that Git succeeded. The evidence snapshot is copied, not recomputed."""
    body = copy.deepcopy(author)
    body[LANDING_KEY] = {
        "kind": LANDING_KIND, "landed": True,
        "subject_hash": str(body.get("subject_hash") or ""),
        "pre_review_fingerprint": str(pre_review_fingerprint or ""),
        "post_review_fingerprint": str(post_review_fingerprint or ""),
    }
    return body


def with_landing(decision: Dict[str, Any], landing: Dict[str, Any]) -> Dict[str, Any]:
    body = {key: copy.deepcopy(value) for key, value in decision.items() if key != "source_ref"}
    body[LANDING_KEY] = copy.deepcopy(landing)
    return body


def commit_landing_fact(subject: Any, commit: Any, *, recorded_at: str, decision: Any = None) -> Optional[Dict[str, str]]:
    """A commit that landed under a record, typed against the tree that record reviewed.

    ``same``/``different`` compare the commit's own tree with the record's ``subject.tree_sha``;
    either one missing is ``unknown``. It speaks for this record only: a different tree never
    says the code went unreviewed, and nothing here is a verdict or a PASS. ``author_subject``
    names the author decision that governed the commit (its ``subject_hash``), ``""`` for none;
    that ``decision``'s disposition and reason are copied when it is the one named, so a later
    decision on the same record never speaks for this commit.
    """
    commit = commit if isinstance(commit, dict) else {}
    sha, tree = (str(commit.get(key) or "").strip().lower() for key in ("commit_sha", "tree_sha"))
    if not _OBJECT_NAME.fullmatch(sha):
        return None
    reviewed = str((subject if isinstance(subject, dict) else {}).get("tree_sha") or "").strip().lower()
    relation = "unknown" if not (tree and reviewed) else "same" if tree == reviewed else "different"
    author = str(commit.get("author_subject") or "")
    governed = decision if isinstance(decision, dict) and author and decision.get("subject_hash") == author else {}
    return {"commit_sha": sha, "tree_sha": tree, "parent_sha": str(commit.get("parent_sha") or ""),
            "reviewed_subject": relation, "author_subject": author,
            "author_disposition": str(governed.get("disposition") or ""), "author_reason": str(governed.get("rationale") or ""),
            "recorded_at": str(recorded_at or "")}


def landed_record(record: Dict[str, Any], landing: Any, commit: Any, *, recorded_at: str,
                  retain: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None
                  ) -> Optional[Dict[str, Any]]:
    """``record`` with what one successful Git command adds, or ``None`` when it holds it already.

    Every bound commit joins ``commit_landings`` (oldest first, every one, one row per commit) with
    the decision the record holds for it as it lands. The author's ``landing`` joins only a record
    that holds the author's decision; ``retain`` keeps the earlier decision's source chain. Critic
    subject, author subject and landed commit stay apart.
    """
    body, changed = dict(record), False
    checked, author = validate_landing(landing), body.get("author_decision")
    if checked is not None and isinstance(author, dict) and author.get(LANDING_KEY) != checked:
        decision = with_landing(author, checked)
        body["author_decision"] = retain(body, decision) if retain else decision
        changed = True
    fact = commit_landing_fact(body.get("subject"), commit, recorded_at=recorded_at, decision=body.get("author_decision"))
    kept = [row for row in body.get(COMMIT_LANDINGS_KEY) or [] if isinstance(row, dict)]
    if fact is not None and all(row.get("commit_sha") != fact["commit_sha"] for row in kept):
        body[COMMIT_LANDINGS_KEY], changed = [*kept, fact], True
    return body if changed else None


def public_commit_landings(record: Any) -> List[Dict[str, str]]:
    """The record's landed commits as a card reads them, oldest first; an unrecognized relation reads ``unknown``."""
    rows = record.get(COMMIT_LANDINGS_KEY) if isinstance(record, dict) else None
    return [{**{key: str(row.get(key) or "") for key in _LANDING_TEXT},
             "reviewed_subject": row.get("reviewed_subject") if row.get("reviewed_subject") in _RELATIONS else "unknown"}
            for row in rows if isinstance(row, dict) and _OBJECT_NAME.fullmatch(str(row.get("commit_sha") or ""))
            ] if isinstance(rows, list) else []


def public_card_decision(decision: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(decision, dict):
        return None
    if not (decision.get("disposition") or decision.get("action") or isinstance(decision.get(FACT_KIND), dict)):
        return None
    body: Dict[str, Any] = {}
    for key in ("disposition", "rationale", "action", "subject_hash", "reviewer_signal",
                "enforcement", "source", "recorded_at", FACT_KIND, EVIDENCE_KIND, LANDING_KEY):
        if decision.get(key) not in ("", None):
            body[key] = copy.deepcopy(decision[key])
    return body or None


def doctrine_result_lines(decision: Any) -> List[str]:
    """Name the fact only on a commit that actually landed."""
    if not isinstance(decision, dict):
        return []
    landing = decision.get(LANDING_KEY)
    if not isinstance(landing, dict) or landing.get("landed") is not True:
        return []
    lines = []
    fact = decision.get(FACT_KIND)
    if isinstance(fact, dict) and fact.get("kind") == FACT_KIND:
        lines.append(
            "review_replaced_by_subagents: subagent reports are author evidence, not review seats, and do not create PASS."
        )
    snapshot = decision.get(EVIDENCE_KIND)
    if isinstance(snapshot, dict) and snapshot.get("feedback_at_decision"):
        lines.append(f"evidence_at_decision: {snapshot['feedback_at_decision']}")
    lines.append("landing: this commit, recorded separately from the evidence at decision.")
    return lines


def history_author_fields(decision: Any) -> Dict[str, Any]:
    decision = decision if isinstance(decision, dict) else {}
    fact = decision.get(FACT_KIND)
    shown: Dict[str, Any] = {
        "remark": "Check replaced by subagents" if isinstance(fact, dict) else "Explicit author decision",
        "status": decision.get("disposition"),
        "reason": decision.get("rationale"),
        "source": decision.get("source_ref"),
    }
    for key in (FACT_KIND, EVIDENCE_KIND, LANDING_KEY):
        if isinstance(decision.get(key), dict):
            shown[key] = copy.deepcopy(decision[key])
    return shown

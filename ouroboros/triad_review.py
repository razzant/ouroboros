"""Shared review primitives: the seat answer contracts and their one parser.

Every review surface (the commit gate, ``review_change`` on any subject — a change, a task
result, a skill payload) asks its seats for JSON findings and reads them here
(``parse_seat_answers``), so no surface re-learns the same truncation and parse-failure
bugs. Contract A is the findings array (``[]`` + ``NO_FINDINGS`` is clean); contract B is
the two-part object of a retrieving change seat. A declared per-item severity policy
(``review_ledger.apply_severity_policy``) is applied while parsing, before any verdict.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ouroboros.utils import append_jsonl, truncate_review_artifact, utc_now_iso


@dataclass
class ReviewActorRecord:
    model_id: str
    status: str
    raw_text: str
    parsed_items: List[Dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: Optional[float] = None
    slot: int = 0
    slot_id: str = ""  # the id the row physically ran under, carried not re-derived
    prompt_ref: Dict[str, Any] = field(default_factory=dict)
    response_ref: Dict[str, Any] = field(default_factory=dict)
    failure_code: str = ""
    reset_at: str = ""
    http_status: Optional[int] = None
    transport_status: str = ""
    reported_cause: str = ""  # the failing owner's reported cause; opaque, "" = none
    failure_details: Dict[str, Any] = field(default_factory=dict)
    operation_id: str = ""
    operation_state: str = "settled"
    late_result_pending: bool = False
    pending_invocation_id: str = ""
    delegated_run_id: str = ""
    recovery_binding: Dict[str, Any] = field(default_factory=dict)
    # One brief, two parts: the parts this seat was asked and its answer per part
    # (``parse_seat_answers``); empty on a record that predates the vector.
    parts: List[str] = field(default_factory=list)
    answers: Dict[str, Any] = field(default_factory=dict)
    coverage: str = ""  # observed read coverage of the required-source manifest ("" = not a retrieving seat)
    context_manifest: Dict[str, Any] = field(default_factory=dict)
    # A durable ``review_model_error`` append already succeeded for this actor (#1578):
    # a positive receipt, never inferred from the status. The wave it ran in.
    error_event_logged: bool = False
    review_wave_id: str = ""
    additional: bool = False
    cost_kind: str = "unknown"
    outcome: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # The durable id is the one the review substrate actually ran this row
        # under, carried on the envelope. Re-deriving it from the record's
        # position is wrong whenever position is not row: an oversized skill
        # review merges C chunk passes over M rows into ONE results list
        # (a chunked wave of an earlier version), so position M+1 was row 1 on its second pass.
        from ouroboros.review_substrate import slot_id_for_row

        return {
            "model_id": self.model_id,
            "status": self.status,
            "raw_text": self.raw_text,
            "parsed_items": list(self.parsed_items),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "cost_kind": self.cost_kind,
            **({"outcome": dict(self.outcome)} if self.outcome else {}),
            "additional": self.additional,
            "slot": self.slot,
            # LEGACY-READ-ONLY fallback: every live producer stamps slot_id (the one
            # dispatch entry is review._handle_multi_model_review — skill_review binds
            # run_review to exactly it, pinned by test), so the positional mint here
            # can only fire when re-serializing an envelope persisted before the carry.
            "slot_id": self.slot_id or (slot_id_for_row(self.slot) if self.slot else ""),
            "prompt_ref": dict(self.prompt_ref),
            "response_ref": dict(self.response_ref),
            "failure_code": self.failure_code,
            "reset_at": self.reset_at,
            "http_status": self.http_status,
            "transport_status": self.transport_status,
            **({"reported_cause": self.reported_cause} if self.reported_cause else {}),
            **({"failure_details": dict(self.failure_details)} if self.failure_details else {}),
            "operation_id": self.operation_id,
            "operation_state": self.operation_state,
            "late_result_pending": self.late_result_pending,
            "pending_invocation_id": self.pending_invocation_id,
            "delegated_run_id": self.delegated_run_id,
            "recovery_binding": dict(self.recovery_binding),
            **({"parts": list(self.parts), "answers": dict(self.answers)} if self.parts else {}),
            **({"coverage": self.coverage} if self.coverage else {}),
            **({"context_manifest": dict(self.context_manifest)} if self.context_manifest else {}),
            "error_event_logged": self.error_event_logged,
            **({"review_wave_id": self.review_wave_id} if self.review_wave_id else {}),
        }


@dataclass
class ParsedTriadReview:
    findings: List[Dict[str, Any]]
    responsive_models: List[str]
    actor_records: List[ReviewActorRecord]
    errors: List[str] = field(default_factory=list)

    @property
    def quorum_met(self) -> bool:
        from ouroboros.review_ledger import reduce_verdict

        rows = [{"seat_id": r.slot_id, "status": r.status, "parts": r.parts, "answers": r.answers}
                for r in self.actor_records if not r.additional]
        quorum = reduce_verdict(rows)["quorum"]
        return quorum["responded"] >= max(1, quorum["required"])

    @property
    def degraded_reasons(self) -> List[str]:
        degraded = [r for r in self.actor_records if r.status in {"error", "parse_failure"}]
        if not degraded or not self.quorum_met:
            return []
        reasons = [f"{r.model_id}={r.status}" for r in degraded]
        return [f"DEGRADED: {', '.join(reasons)} (quorum still met)"]


def _actor_record(
    actor: Dict[str, Any],
    *,
    idx: int,
    model_label: str,
    status: str,
    raw_text: str,
    parsed_items: Optional[List[Dict[str, Any]]] = None,
) -> ReviewActorRecord:
    cost = actor.get("cost_estimate") if actor.get("cost_estimate") is not None else actor.get("cost_usd")
    return ReviewActorRecord(
        model_id=model_label,
        status=status,
        raw_text=raw_text,
        parsed_items=parsed_items or [],
        tokens_in=int(actor.get("tokens_in", 0) or 0),
        tokens_out=int(actor.get("tokens_out", 0) or 0),
        cost_usd=float(cost) if cost is not None else None,
        additional=bool(actor.get("additional")),
        cost_kind=str(actor.get("cost_kind") or ("measured" if cost is not None else "unknown")),
        slot=idx + 1,
        slot_id=str(actor.get("slot_id") or ""),
        prompt_ref=dict(actor.get("prompt_ref") or {}),
        response_ref=dict(actor.get("response_ref") or {}),
        failure_code=str(actor.get("failure_code") or ""),
        reset_at=str(actor.get("reset_at") or ""),
        http_status=(int(actor["http_status"]) if isinstance(actor.get("http_status"), int) else None),
        transport_status=str(actor.get("transport_status") or ""),
        reported_cause=str(actor.get("reported_cause") or ""),
        failure_details=dict(actor.get("failure_details") or {}),
        operation_id=str(actor.get("operation_id") or ""),
        operation_state=str(actor.get("operation_state") or "settled"),
        late_result_pending=bool(actor.get("late_result_pending")),
        pending_invocation_id=str(actor.get("pending_invocation_id") or ""),
        delegated_run_id=str(actor.get("delegated_run_id") or ""),
        recovery_binding=dict(actor.get("recovery_binding") or {}),
        error_event_logged=actor.get("error_event_logged") is True,
        review_wave_id=str(actor.get("review_wave_id") or ""),
    )


def extract_json_array(
    raw: str,
    *,
    normalize: bool = False,
    unwrap_result: bool = False,
    validate_fn: Optional[Callable[[List[Any]], bool]] = None,
) -> Optional[List[Any]]:
    """Best-effort extraction of a JSON array from model output."""
    text = str(raw or "").strip()
    candidates = [text]
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk:
                candidates.append(chunk)

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if unwrap_result and isinstance(obj, dict) and "result" in obj:
                candidate = str(obj["result"]).strip()
                obj = json.loads(candidate)
            if isinstance(obj, list):
                return _accepted_json_array(obj, normalize=normalize, validate_fn=validate_fn)
        except (json.JSONDecodeError, ValueError):
            pass
        except TypeError:
            pass
        ends: List[int] = []
        search_from = 0
        while True:
            pos = candidate.find("]", search_from)
            if pos == -1:
                break
            ends.append(pos)
            search_from = pos + 1
        for end in reversed(ends):
            starts: List[int] = []
            search_from = 0
            while True:
                pos = candidate.find("[", search_from)
                if pos == -1 or pos > end:
                    break
                starts.append(pos)
                search_from = pos + 1
            for start in reversed(starts):
                try:
                    obj = json.loads(candidate[start:end + 1])
                    if isinstance(obj, list):
                        accepted = _accepted_json_array(obj, normalize=normalize, validate_fn=validate_fn)
                        if accepted is not None:
                            return accepted
                except (json.JSONDecodeError, ValueError):
                    continue
    return None


def _accepted_json_array(
    obj: List[Any],
    *,
    normalize: bool,
    validate_fn: Optional[Callable[[List[Any]], bool]],
) -> Optional[List[Any]]:
    if validate_fn is not None and not validate_fn(obj):
        return None
    return _normalize_items(obj) if normalize else obj


def _normalize_items(items: List[Any]) -> List[Any]:
    try:
        from ouroboros.tools.review_helpers import normalize_reviewer_items
        return normalize_reviewer_items(items)
    except Exception:
        return items


def extract_fenced_json(text: str) -> Any:
    """Best-effort parse of a fenced/embedded JSON object or array from model output.

    Reviewers often wrap their verdict in a ```json ... ``` fence; a fenced JSON
    OBJECT (e.g. {"verdict":"PASS","findings":[]}) would otherwise fail json.loads
    and be missed by the array-only extractor, producing a false DEGRADED signal.
    """
    if "```" not in text:
        return None
    for chunk in text.split("```"):
        candidate = chunk.strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            return obj
    return None


# Output SHAPE per surface — the ONE form fact the retrieving-route canonicalizer,
# the session output schema and the strict parser branch on. Shape is FORM only
# (no surface policy lives here: quorum authority stays with its owner). Every
# surface not listed returns the findings ARRAY contract; ``report`` is a
# free-form markdown product that is never canonicalized or extracted.
REVIEW_OUTPUT_SHAPES: Dict[str, str] = {
    "deep_self_review": "report",
    # The commit gate's retrieving seats answer the two-part brief (contract B):
    # one object {change, change_clean, coupling}. Packet seats on the same
    # surface answer contract A over direct chat and are never canonicalized.
    "multi_model_review": "two_part",
    # The planning organ's participants answer ONE reply object {prose, objections,
    # withdrawn} (tools/plan_packet.PLAN_REPLY_CONTRACT). It is no verdict: the host
    # never extracts or aggregates it; the whole text reaches plan_record.parse_reply.
    "plan_review": "plan_reply",
}
TWO_PART_KEYS = ("change", "change_clean", "coupling")

# Default output contract per SHAPE for a retrieving delivery whose surface did
# not hand over its own `policy["output_contract"]`: the prompt must ask for the
# same form the canonicalizer parses, or an obedient reviewer answering another
# surface's contract is read as malformed by its own host.
REVIEW_REPORT_CONTRACT = """\
Deliver the report itself as plain markdown prose — no JSON wrapper, no code
fence around the whole answer. Most critical findings first, each with the
evidence you read; mark anything you could not verify as unverified.
"""


def review_output_shape(surface: str) -> str:
    """``array`` | ``two_part`` | ``report`` | ``plan_reply`` for one review surface."""
    return REVIEW_OUTPUT_SHAPES.get(str(surface or ""), "array")


def request_output_shape(request: Any) -> str:
    """The shape one request's seat answers: its policy's own ``output_shape`` when it names one (a
    result wave rides the commit surface but asks contract A), else its surface's. The ask, the
    session schema and the canonicalizer all read this one fact, so they never disagree."""
    shape = str((getattr(request, "policy", None) or {}).get("output_shape") or "")
    return shape if shape in {"array", "two_part", "report", "plan_reply"} else review_output_shape(
        str(getattr(request, "surface", "") or ""))


def default_output_contract(shape: str) -> str:
    """The prompt contract a retrieving row is asked for when its surface hands
    over no ``policy["output_contract"]`` — keyed by the same shape the
    canonicalizer parses, so the ask and the parse can never disagree."""
    if str(shape or "") == "plan_reply":
        from ouroboros.tools.plan_packet import PLAN_REPLY_CONTRACT

        return PLAN_REPLY_CONTRACT
    return {
        "report": REVIEW_REPORT_CONTRACT, "two_part": REVIEW_TWO_PART_OBJECT_CONTRACT,
    }.get(str(shape or ""), REVIEW_JSON_ARRAY_CONTRACT)


def two_part_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """The two-part answer (contract B) as ONE object, or None when the payload is
    not one. Form only: ``coupling`` must be a list of dicts; ``change`` (when
    present) a list of dicts; ``change_clean`` (when present) a bool. A bare
    ARRAY is accepted as the ``change`` answer alone — the seat spoke but left the
    coupling block out, which the gate records as ``coupling_block_missing``
    rather than as silence."""
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            return None
        return {"change": list(payload), "change_clean": not payload}
    if not isinstance(payload, dict) or not any(key in payload for key in TWO_PART_KEYS):
        return None
    out: Dict[str, Any] = {}
    for key in ("change", "coupling"):
        if key in payload:
            if not isinstance(payload[key], list) or not all(isinstance(item, dict) for item in payload[key]):
                return None
            out[key] = list(payload[key])
    if "change_clean" in payload:
        clean = payload["change_clean"]
        if isinstance(clean, str) and clean.strip().lower() in ("true", "false"):
            clean = clean.strip().lower() == "true"  # the bool spelled as its JSON word
        if not isinstance(clean, bool):
            return None
        out["change_clean"] = clean
    return out


_FENCE_LINE_END = re.compile(r"\r\n|\r|\n")


def _whole_json(raw_text: str) -> Any:
    """The response as ONE JSON value — bare, or inside one code fence that opens
    on the first line and closes on the last; prose around the value is a
    non-response (the same discipline as the array contract). The whole text is
    parsed first, and a fence is recognized by its fence LINES only, so a fence
    spelling inside a JSON string is content, never a split point. Lines end at
    CR/LF only and the body between the fence lines is parsed as written, so a
    U+0085/U+2028/U+2029 inside a JSON string (``str.splitlines`` cuts there) is
    content too."""
    text = str(raw_text or "").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    lines = _FENCE_LINE_END.split(text)
    if len(lines) < 3 or not lines[0].startswith("```") or lines[-1].strip() != "```":
        return None
    tag = lines[0][3:].strip()
    if tag and not tag.isalnum():
        return None
    try:
        return json.loads(text[len(lines[0]):len(text) - len(lines[-1])].strip())
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _change_items(entries: List[Dict[str, Any]], *, model_label: str, slot_id: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for entry in entries:
        item = str(entry.get("item") or "")
        verdict = str(entry.get("verdict") or "").upper()
        if not item or verdict not in {"PASS", "FAIL"}:
            continue
        items.append({
            "item": item, "verdict": verdict,
            "severity": str(entry.get("severity") or "advisory").lower(),
            "reason": str(entry.get("reason") or entry.get("evidence") or "").strip(),
            "model": model_label, "slot_id": slot_id,
            **({"obligation_id": str(entry.get("obligation_id") or "")} if entry.get("obligation_id") else {}),
        })
    return items


def _change_answer(items: List[Dict[str, Any]], severity_policy: Any = None) -> Dict[str, Any]:
    """``findings`` are the FAIL items (what the gate and the record reason over);
    ``items`` is every normalized row, PASS included, for the actor record's
    forensic ``parsed_items`` — stripped before the answer is recorded. A declared
    ``severity_policy`` (a skill wave's checklist rule) decides each FAIL item's
    severity, and so the seat's verdict, before anything reads it."""
    if severity_policy:
        from ouroboros.review_ledger import apply_severity_policy

        items = apply_severity_policy({"status": "", "findings": list(items)}, severity_policy)["findings"]
    failed = [i for i in items if i["verdict"] == "FAIL"]
    critical = [i for i in failed if i["severity"] == "critical"]
    return {"status": "responded", "verdict": "FAIL" if critical else "PASS", "findings": failed,
            "critical": len(critical), "coverage": "n/a", "items": list(items)}


def _unanswered(part: str, error: str = "") -> Dict[str, Any]:
    answer = {"status": "unanswered", "verdict": "", "findings": [], "critical": 0,
              "coverage": "missing" if part == "coupling" else "n/a"}
    if error:
        answer["error"] = error
    return answer


def parse_two_part_answer(raw_text: str, parts: Sequence[str], *, model_label: str = "reviewer",
                          slot_id: str = "") -> Optional[Dict[str, Dict[str, Any]]]:
    """Contract B → ``{part: answer}`` for the parts a seat was asked, or None when
    the response is a non-response (no JSON value, prose around it, wrong form).
    ``change: []`` is a clean answer only with ``change_clean: true``; a
    ``coupling`` block that fails the required matrix (``normalize_scope_items``)
    is ``unanswered`` with its ``error`` (its FAIL rows kept as ``discarded``
    diagnostics, not counted) while ``change`` still counts; a bare
    array from a seat asked both parts answers ``change`` and leaves ``coupling``
    unanswered (``coupling_block_missing``). A seat asked ``coupling`` alone may
    answer the object with only that key or the bare matrix array."""
    asked = [p for p in ("change", "coupling") if p in tuple(parts)]
    value = _whole_json(raw_text)
    if value is None:
        return None
    if isinstance(value, list) and asked == ["coupling"]:
        value = {"coupling": value}
    payload = two_part_payload(value)
    if payload is None:
        return None
    answers: Dict[str, Dict[str, Any]] = {}
    if "change" in asked:
        if "change" not in payload:
            answers["change"] = _unanswered("change", "the change block is missing")
        elif not payload["change"] and not payload.get("change_clean"):
            answers["change"] = _unanswered("change", "an empty change block needs change_clean: true")
        else:
            items = _change_items(payload["change"], model_label=model_label, slot_id=slot_id)
            if payload["change"] and not items:
                # The seat wrote entries the gate cannot read (no ``item``, a verdict
                # outside PASS/FAIL): that is no answer, not a clean one — whatever
                # ``change_clean`` says beside it.
                answers["change"] = _unanswered("change", (
                    f"none of the {len(payload['change'])} change entries has an item and a PASS/FAIL verdict"))
            else:
                answers["change"] = _change_answer(items)
    if "coupling" in asked:
        if "coupling" not in payload:
            answers["coupling"] = _unanswered("coupling", "coupling_block_missing")
        else:
            from ouroboros.tools.scope_review_contract import (
                SCOPE_REQUIRED_ITEMS, classify_scope_findings, normalize_scope_items,
            )

            items, error = normalize_scope_items(payload["coupling"])
            covered = {str(i.get("item") or "") for i in (items or payload["coupling"]) if isinstance(i, dict)}
            n_required = sum(1 for item in SCOPE_REQUIRED_ITEMS if item in covered)
            coverage = "full" if n_required == len(SCOPE_REQUIRED_ITEMS) else "partial" if n_required else "missing"
            critical, advisory = classify_scope_findings(items or [])
            for finding in critical + advisory:  # the projection's placeholder model → this seat
                finding.update(model=model_label, slot_id=slot_id)
            if error or items is None:
                # The matrix is not countable (contract §1.7: ``unanswered`` with
                # its error), but the FAIL rows it did spell out are kept beside
                # the error as ``discarded`` diagnostics — never counted, never
                # silently lost (a critical in a broken matrix stays visible even
                # when the panel's other seats carry the question to PASS).
                answers["coupling"] = {**_unanswered("coupling", error or "coupling matrix invalid"),
                                       "coverage": coverage, "discarded": critical + advisory}
            else:
                answers["coupling"] = {"status": "responded", "verdict": "FAIL" if critical else "PASS",
                                       "findings": critical + advisory, "critical": len(critical), "coverage": coverage,
                                       "items": [dict(i, model=model_label, slot_id=slot_id) for i in items]}
    return answers


def parse_seat_answer(actor: Dict[str, Any], parts: Sequence[str], *, idx: int = 0,
                      severity_policy: Any = None) -> ReviewActorRecord:
    """Read the saved outcome, or finalize one legacy response by its declared parts."""
    from ouroboros.tools.scope_required_sources import coverage_state

    model = str(actor.get("model") or actor.get("request_model") or "").strip()
    raw_text = str(actor.get("text") or "")
    model_label = model or "reviewer"
    slot_id = str(actor.get("slot_id") or "")
    parts = [p for p in ("change", "coupling", "result", "skill") if p in tuple(parts or ("change",))]
    outcome = actor.get("outcome") or {}
    if outcome:
        record = _actor_record(actor, idx=idx, model_label=model_label,
                               status=outcome["status"], raw_text=raw_text,
                               parsed_items=outcome.get("parsed_items") or [])
        record.parts = list(outcome["parts"])
        record.answers = dict(outcome["answers"])
        record.coverage = str(outcome.get("read_coverage") or "")
        record.context_manifest = dict(outcome.get("context_manifest") or {})
        record.additional = bool(outcome.get("additional"))
        record.outcome = dict(outcome)
        return record
    state = str(actor.get("operation_state") or "")
    if (actor.get("late_result_pending") and state != "custody_lost") or state in {"in_flight", "pending_dispatch"}:
        record = _actor_record(actor, idx=idx, model_label=model_label, status="pending", raw_text=raw_text)
    elif actor.get("transport_status") == "interrupted" or actor.get("failure_code") == "review_run_cancelled":
        record = _actor_record(actor, idx=idx, model_label=model_label, status="interrupted", raw_text=raw_text)
    elif state == "not_dispatched":
        record = _actor_record(actor, idx=idx, model_label=model_label, status="not_dispatched", raw_text=raw_text)
    elif str(actor.get("verdict") or "").upper() == "ERROR":
        record = _actor_record(actor, idx=idx, model_label=model_label, status="error", raw_text=raw_text)
    else:
        if len(parts) == 1 and parts[0] != "coupling":
            # Contract A: the change question of a packet seat, and the one question of every
            # single-question seat (a result, a skill payload), answer the findings array.
            parsed = extract_json_array(raw_text, normalize=True)
            if parsed is None or (not parsed and not empty_array_is_verified_clean(raw_text)):
                answers = None
            else:
                items = _change_items([e for e in parsed if isinstance(e, dict)], model_label=model_label,
                                      slot_id=slot_id)
                if parsed and not items:
                    # The seat wrote entries the gate cannot read (no ``item``, a verdict outside
                    # PASS/FAIL): no answer, not a clean one — the rule contract B applies to its
                    # change block, for every contract-A part alike.
                    answers = {parts[0]: _unanswered(parts[0], (
                        f"none of the {len(parsed)} entries has an item and a PASS/FAIL verdict"))}
                else:
                    answers = {parts[0]: _change_answer(items, severity_policy)}
        else:
            answers = parse_two_part_answer(raw_text, parts, model_label=model_label, slot_id=slot_id)
        if answers is None:
            record = _actor_record(actor, idx=idx, model_label=model_label, status="parse_failure", raw_text=raw_text)
        else:
            usable = [p for p in parts if answers[p]["status"] == "responded"]
            items = [f for p in usable for f in answers[p].pop("items", answers[p]["findings"])]
            record = _actor_record(actor, idx=idx, model_label=model_label,
                                   status="responded" if usable else "parse_failure", raw_text=raw_text,
                                   parsed_items=items)
            record.answers = answers

    record.parts = parts
    if "coupling" in parts:
        fact = actor.get("native_read_coverage")
        record.coverage = coverage_state(fact)
        if isinstance(fact, dict):
            record.context_manifest = {"native_read_coverage": fact,
                                       "read_provenance": str(actor.get("read_provenance") or "")}
    return record


def parse_seat_answers(result_json: Dict[str, Any], row_parts: Dict[str, Sequence[str]], *,
                       severity_policy: Any = None) -> ParsedTriadReview:
    """Read ONE producer outcome per seat; legacy rows use the same contract finalizer."""
    findings, responsive, records = [], [], []
    for idx, actor in enumerate(result_json.get("results") or []):
        if not isinstance(actor, dict):
            continue
        slot_id = str(actor.get("slot_id") or "")
        record = parse_seat_answer(actor, row_parts.get(slot_id) or ("change",), idx=idx,
                                   severity_policy=severity_policy)
        records.append(record)
        if record.status == "responded":
            findings.extend(f for answer in record.answers.values() for f in answer.get("findings", []))
            if not record.additional:
                responsive.append(f"{record.model_id} [{slot_id}]" if slot_id else f"{record.model_id}#{idx + 1}")
    return ParsedTriadReview(findings=findings, responsive_models=responsive, actor_records=records)


def parse_review_findings(raw_text: str) -> tuple[Any, List[Dict[str, Any]], str]:
    """Reviewer response -> (parsed, findings, signal), by the object/array ladder."""
    text = str(raw_text or "").strip()
    parsed: Any = None
    findings: List[Dict[str, Any]] = []
    signal = "UNKNOWN"
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = extract_fenced_json(text)
        if parsed is None:
            extracted = extract_json_array(text)
            if extracted is None:
                # Keep non-JSON output untruncated; reviewer raw_text is still useful.
                return None, [], "DEGRADED"
            parsed = extracted
    if isinstance(parsed, dict):
        signal = str(parsed.get("verdict") or parsed.get("status") or "UNKNOWN").upper()
        raw_findings = parsed.get("findings") or []
        if isinstance(raw_findings, list):
            findings = [item for item in raw_findings if isinstance(item, dict)]
    elif isinstance(parsed, list):
        findings = [item for item in parsed if isinstance(item, dict)]
        verdicts = {str(item.get("verdict") or item.get("status") or "").upper() for item in findings}
        if "FAIL" in verdicts:
            signal = "FAIL"
        elif "PASS" in verdicts:
            signal = "PASS"
        elif "DEGRADED" in verdicts:
            signal = "DEGRADED"
        else:
            signal = "UNKNOWN"
    return parsed, findings, signal


# The review output contract lives beside the parser that enforces it: the
# advisory path once shipped a prompt asking for NO_FINDINGS while its own
# parser had no branch for it, so a clean review was recorded as unparseable.
_REVIEW_JSON_ELEMENT_SCHEMA = """\
{
  "item": "<checklist item name>",
  "verdict": "PASS" | "FAIL",
  "severity": "critical" | "advisory",
  "reason": "<for FAIL: file, line/symbol, what is wrong, how to fix>"
}"""

# Findings-only: nothing-to-report is legitimate, so the empty array needs the
# sentinel to be distinguishable from a refusal.
REVIEW_JSON_ARRAY_CONTRACT = f"""\
Return ONLY a JSON array, optionally followed by the sentinel line described
below. Each element:
{_REVIEW_JSON_ELEMENT_SCHEMA}
If you reviewed everything and found NOTHING to report, return the empty array
followed by the sentinel word NO_FINDINGS on its own line, and nothing else.
Prose around the array, or the sentinel without the array, is treated as a
non-response and excluded from quorum.
"""


# Contract B — the two-part brief's answer (one brief, two parts): the change
# findings array of contract A under ``change`` plus the required coupling
# matrix under ``coupling``, in ONE object. ``change_clean`` replaces the
# NO_FINDINGS sentinel inside the object.
REVIEW_TWO_PART_OBJECT_CONTRACT = f"""\
Return ONLY one JSON object with exactly these keys:
{{
  "change": [<Part 1 findings — each element as below>],
  "change_clean": true | false,
  "coupling": [<Part 2 — one entry per required coupling question, each element as below>]
}}
Each element of "change" and "coupling":
{_REVIEW_JSON_ELEMENT_SCHEMA}
"change" lists what you found reviewing the change itself; an empty "change"
array is a clean answer ONLY together with "change_clean": true. "coupling"
must carry one PASS or FAIL entry for EVERY required coupling question named in
Part 2 (a PASS reason is one line: a pointer to what you checked, or "N/A — <reason>";
a FAIL names file:line and the violated invariant; there is no all-clear shortcut
and no empty "coupling"). A seat asked only Part 2 returns
the object with the "coupling" key alone. Prose around the object, a missing
"coupling" block or a malformed matrix is recorded against that part; anything
that is not the object is a non-response excluded from quorum.
"""

_TWO_PART_ELEMENT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "item": {"type": "string"},
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "severity": {"type": "string", "enum": ["critical", "advisory"]},
        "reason": {"type": "string"},
    },
    "required": ["item", "verdict", "reason"],
}
# The session output schema for the ``two_part`` shape: ``coupling`` is the key a
# retrieving seat always owes; ``change`` is owed by the seats asked Part 1 and is
# enforced per seat by the parser, since one surface shares one schema.
TWO_PART_SESSION_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "change": {"type": "array", "items": _TWO_PART_ELEMENT_JSON_SCHEMA},
        "change_clean": {"type": "boolean"},
        "coupling": {"type": "array", "items": _TWO_PART_ELEMENT_JSON_SCHEMA, "minItems": 1},
    },
    "required": ["coupling"],
}


# The sentinel is optional whitespace-separated trailer, not a stricter shape than
# the bare array: emitting it must never make a clean response harder to accept.
_CLEAN_EMPTY_RESPONSE_RE = re.compile(r"^\[\s*\]\s*(?:NO_FINDINGS\s*)?$")


def empty_array_is_verified_clean(raw_text: str) -> bool:
    """True when an EMPTY findings array is a verifiable clean verdict.

    Shared by every consumer of ``REVIEW_JSON_ARRAY_CONTRACT`` so the sentinel
    the prompt asks for is honored identically on each path.

    The WHOLE response must be the clean payload — bare ``[]``, or ``[]``
    followed by a standalone ``NO_FINDINGS`` line — modulo one optional code
    fence. Surrounding prose is refused on purpose: a refusal cannot be told
    from a benign preamble by structure, so "I cannot review this diff.
    []\\nNO_FINDINGS" must not enter quorum as a clean verdict. This matches what
    the contract actually asks for ("Return ONLY a JSON array, optionally
    followed by the sentinel line"); anything looser lets a reviewer opt out of
    the gate with prose.
    """
    text = str(raw_text or "").strip()
    if "```" in text:
        # Unwrap a fenced block: ```json\n[]\n```, optionally with the sentinel
        # AFTER the closing fence — a model that fences its JSON puts it there,
        # and the contract asks for the sentinel on its own line.
        parts = [chunk.strip() for chunk in text.split("```") if chunk.strip()]
        if len(parts) == 2 and parts[1] == "NO_FINDINGS":
            parts = [f"{parts[0]}\nNO_FINDINGS"]
        if len(parts) != 1:
            return False
        text = parts[0]
        tag, _, rest = text.partition("\n")
        if rest.strip() and tag.strip().isalnum():
            text = rest.strip()  # drop a language tag line (json/JSON/text/...)
    return bool(_CLEAN_EMPTY_RESPONSE_RE.match(text))


def _not_model_loss(row: Dict[str, Any]) -> bool:
    """Leaving a live wave or interrupting its host is never a reviewer failure."""
    state = str(row.get("operation_state") or "")
    return (state in {"in_flight", "pending_dispatch", "interrupted"}
            or (row.get("late_result_pending") and state != "custody_lost")
            or row.get("transport_status") == "interrupted"
            or row.get("failure_code") == "review_run_cancelled"
            or bool((row.get("usage") or {}).get("owner_pause_preparing")))


def emit_review_model_error_events(ctx: Any, parsed: ParsedTriadReview, *, source: str, skill_name: str = "") -> None:
    """Parser-level failures (error, parse_failure, partial) through the one durable owner.

    An actor whose run-level failure already has a successful append is skipped.
    """
    for record in parsed.actor_records:
        if record.status not in {"error", "parse_failure", "partial"} or record.error_event_logged:
            continue
        row = asdict(record)
        row["error"] = (record.raw_text if record.status == "error" else
                        "Reviewer response omitted required checklist items." if record.status == "partial"
                        else "Reviewer response did not contain a valid findings array.")
        row["error_note"] = ("Full raw response preserved in review.json raw_actor_records when quorum "
                             "succeeds; otherwise in review_history.jsonl." if source == "skill_review" else
                             "Full raw response preserved in triad_raw_results.")
        record.error_event_logged = _append_review_model_error(ctx, row, source=source, skill_name=skill_name)


def emit_review_run_errors(ctx: Any, request: Any, actors: Sequence[Any], *, drive_root: Any = None) -> None:
    """Run- and collection-level reviewer failures of one review wave (every surface).

    The positive append receipt lands on the actor and rides its run result, and
    reaches this process's custody copies of the same operation, so the downstream
    parser and a later collection or replay of that settlement do not append it
    again; a failed append leaves it unlogged for a later collection. An awaited
    slot is not a failure. No history is read here.
    """
    for actor in actors:
        actor.review_wave_id = str(getattr(actor, "review_wave_id", "") or getattr(request, "resolved_wave_id", "") or "")
        if actor.error_event_logged:
            continue
        if actor.status not in {"error", "not_dispatched", "empty"} and (
                actor.parse_status != "malformed" or REVIEW_OUTPUT_SHAPES.get(request.surface) in ("report", "plan_reply")):
            continue
        row = asdict(actor)
        row["task_id"] = str(getattr(request, "task_id", "") or "")
        row["status"] = actor.status if actor.status in {"error", "not_dispatched"} else "parse_failure"
        row["error"] = actor.error or actor.reason or "Reviewer response was malformed or absent."
        from ouroboros.review_records import review_failure_facts

        row.update(review_failure_facts(row))  # the event row only: the actor record keeps its producer's facts
        meta = getattr(request, "usage_attribution", {}) or {}
        actor.error_event_logged = _append_review_model_error(
            ctx, row, source=str(request.surface or ""), skill_name=str(meta.get("review_skill") or ""),
            drive_root=drive_root)
        if actor.error_event_logged:
            _keep_error_receipt(ctx, actor)


def _keep_error_receipt(ctx: Any, actor: Any) -> None:
    """Carry a successful append to this process's custody copies of the SAME operation
    (#1578): the settled replay copy and a still-registered live entry were taken before
    the coordinator logged. Another operation of the slot never inherits the receipt."""
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK

    identity = (str(actor.slot_id or ""), str(actor.operation_id or ""))
    settled = getattr(ctx, "_review_settled_attempts", None)
    with _ACTIVE_LOCK:
        for held in [*(settled.values() if isinstance(settled, dict) else ()),
                     *(entry.actor for entry in _ACTIVE.values() if entry.actor is not None)]:
            if identity[1] and (str(getattr(held, "slot_id", "") or ""),
                                str(getattr(held, "operation_id", "") or "")) == identity:
                held.error_event_logged = True


def emit_review_execution_error(ctx: Any, assignment: Any, error: Any, *, usage: Dict[str, Any],
                                prompt_ref: Optional[Dict[str, Any]] = None) -> bool:
    """A direct (report) executor's failure, through the same one durable owner as a panel."""
    request, slot = assignment.request, assignment.slot
    row = {
        "task_id": str(request.task_id or ""), "review_wave_id": str(request.resolved_wave_id or ""),
        "model": str(usage.get("resolved_model") or slot.model or ""), "slot_id": slot.slot_id,
        "operation_id": assignment.call_id, "operation_state": str(usage.get("operation_state") or "settled"),
        "usage": usage, "status": "error", "error": str(error),
        "failure_code": str(getattr(error, "code", "") or usage.get("failure_code") or ""),
        "reported_cause": str(getattr(error, "reported_cause", "") or ""),
        "http_status": getattr(error, "status_code", None), "reset_at": str(getattr(error, "reset_at", "") or ""),
        "prompt_ref": prompt_ref or {},
    }
    return _append_review_model_error(ctx, row, source=str(request.surface or ""), drive_root=assignment.custody_root)


def _review_event_log(ctx: Any, drive_root: Any = None) -> pathlib.Path:
    """The CANONICAL events log: the task's budget (canonical) root first; a headless or
    task-drive child drive anchors on the data root that owns it (#1578)."""
    from ouroboros.task_custody import canonical_owner_root

    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    root = next((str(value) for value in (metadata.get("budget_drive_root"), getattr(ctx, "budget_drive_root", ""),
                                          drive_root, getattr(ctx, "drive_root", "")) if str(value or "").strip()), "")
    if not root:  # a context that only exposes its log directory
        root = str(pathlib.Path(ctx.drive_logs()).parent)
    return canonical_owner_root(root) / "logs" / "events.jsonl"


def _append_review_model_error(ctx: Any, row: Dict[str, Any], *, source: str,
                               skill_name: str = "", drive_root: Any = None) -> bool:
    """The one durable ``review_model_error`` owner; True only for a successful append.

    ``append_jsonl``'s log sink owns the one live frame. Known task/root/parent/wave
    identities come from the row, its recovery binding, the task metadata, the bound
    usage scope or the context; the actual failure cause and refs ride as recorded.
    """
    try:
        from ouroboros.usage_accounting import current_usage_scope
        from ouroboros.review_records import review_failure_facts

        if _not_model_loss(row):
            return False
        row = {**row, **review_failure_facts(row)}
        binding, usage = row.get("recovery_binding") or {}, row.get("usage") or {}
        receipt = usage.get("claudexor") if isinstance(usage.get("claudexor"), dict) else {}
        metadata = getattr(ctx, "task_metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        scope = current_usage_scope()
        identity = {key: str(row.get(key) or binding.get(key) or metadata.get(key) or getattr(scope, key, "")
                             or getattr(ctx, key, "") or "")
                    for key in ("task_id", "root_task_id", "parent_task_id", "review_wave_id")}
        payload = {
            "ts": utc_now_iso(), "type": "review_model_error", "source": source, "skill": skill_name,
            **{key: value for key, value in identity.items() if value},
            "model": str(row.get("model_id") or row.get("model") or ""),
            "provider": str(usage.get("provider") or row.get("provider") or ""),
            # A preview with a disclosed cut; the full response stays behind response_ref/the actor record.
            "status": str(row.get("status") or ""), "error": truncate_review_artifact(str(row.get("error") or "")),
            **{key: str(row.get(key) or "") for key in (
                "reported_cause", "failure_code", "transport_status", "reset_at", "slot_id",
                "operation_id", "operation_state")},
            "http_status": row.get("http_status") if isinstance(row.get("http_status"), int) else None,
            "delegated_run_id": str(row.get("delegated_run_id") or usage.get("delegated_run_id")
                                    or receipt.get("run_id") or ""),
            "pending_invocation_id": str(row.get("pending_invocation_id") or usage.get("pending_invocation_id") or ""),
            "prompt_ref": dict(row.get("prompt_ref") or {}), "response_ref": dict(row.get("response_ref") or {}),
            "error_note": str(row.get("error_note") or "See response_ref and the raw actor record for the full response."),
        }
        return append_jsonl(_review_event_log(ctx, drive_root), payload) is True
    except Exception:
        return False


def review_query_error_payload(
    *,
    ctx: Any,
    model: str,
    messages: list,
    slot_id: str,
    error: str,
    slot: Any = None,
) -> dict:
    """Durable failure envelope for one errored multi-model review row.

    The envelope carries the row id too: an errored row is still a row, and
    its durable actor record must name the same one its refs do. Imports stay
    lazy so this module's dependency graph remains the primitives it declares.
    """
    from ouroboros.review_records import review_failure_facts

    facts = review_failure_facts({"status": "error", "error": error})
    payload = {"error": error, **facts, "usage": {}, "slot_id": slot_id, "prompt_ref": {}, "response_ref": {}}
    try:
        from ouroboros.observability import new_call_id, persist_call
        from ouroboros.tools.review_helpers import review_drive_root

        drive_root = review_drive_root(ctx)
        task_id = str(getattr(ctx, "task_id", "") or "multi_model_review") if ctx is not None else "multi_model_review"
        call_id = new_call_id(f"review_multi_model_review_{slot_id}_error")
        prompt_payload = {"messages": messages, "slot_id": slot_id, "model": model}
        if slot is not None:
            # Match the normal substrate prompt receipt. The outer asyncio
            # timeout happens after this ReviewSlot was constructed, so losing
            # it made a present receipt look absent to provenance binders.
            prompt_payload["slot"] = asdict(slot)
        payload["prompt_ref"] = persist_call(
            drive_root,
            task_id=task_id,
            call_id=f"{call_id}_prompt",
            call_type="multi_model_review_prompt",
            payload=prompt_payload,
            manifest={"surface": "multi_model_review", "slot_id": slot_id, "model": model, "synthetic": True},
        )
        payload["response_ref"] = persist_call(
            drive_root,
            task_id=task_id,
            call_id=f"{call_id}_error",
            call_type="multi_model_review_error",
            payload={"error": error, **facts},
            manifest={"surface": "multi_model_review", "slot_id": slot_id, "model": model, "status": "error", "synthetic": True},
        )
    except Exception:
        pass
    return payload


_PROVIDER_OVERSIZE_MARKERS = (
    # Anthropic: "prompt is too long: 1166914 tokens > 1000000 maximum"
    "prompt is too long",
    # Anthropic: "input length and `max_tokens` exceed context limit"
    "exceed context limit",
    # OpenAI error code + message variants
    "context_length_exceeded",
    "maximum context length",
)


def is_provider_oversize_error(error_text: str) -> bool:
    """Mechanical fault classification: does this provider error mean the prompt
    exceeded the model's REAL context window? Deliberately tight markers — any
    other provider/transport error keeps the fail-closed blocking path."""
    low = str(error_text or "").lower()
    return any(marker in low for marker in _PROVIDER_OVERSIZE_MARKERS)

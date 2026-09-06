"""Shared tri-model review primitives.

Both repo commit review and skill review ask multiple reviewer models to
return a JSON array of checklist findings. Keep parsing, quorum accounting,
and observability in one place so future review entrypoints do not re-learn
the same truncation / parse-failure bugs.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ouroboros.utils import append_jsonl, utc_now_iso


@dataclass
class ReviewActorRecord:
    model_id: str
    status: str
    raw_text: str
    parsed_items: List[Dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    slot: int = 0
    slot_id: str = ""  # the id the row physically ran under, carried not re-derived
    prompt_ref: Dict[str, Any] = field(default_factory=dict)
    response_ref: Dict[str, Any] = field(default_factory=dict)
    failure_code: str = ""
    reset_at: str = ""
    http_status: Optional[int] = None
    transport_status: str = ""
    operation_id: str = ""
    operation_state: str = "settled"
    late_result_pending: bool = False
    pending_invocation_id: str = ""
    delegated_run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        # The durable id is the one the review substrate actually ran this row
        # under, carried on the envelope. Re-deriving it from the record's
        # position is wrong whenever position is not row: an oversized skill
        # review merges C chunk passes over M rows into ONE results list
        # (skill_review_passes), so position M+1 is row 1 on its second pass.
        from ouroboros.review_substrate import slot_id_for_row

        return {
            "model_id": self.model_id,
            "status": self.status,
            "raw_text": self.raw_text,
            "parsed_items": list(self.parsed_items),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
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
            "operation_id": self.operation_id,
            "operation_state": self.operation_state,
            "late_result_pending": self.late_result_pending,
            "pending_invocation_id": self.pending_invocation_id,
            "delegated_run_id": self.delegated_run_id,
        }


@dataclass
class ParsedTriadReview:
    findings: List[Dict[str, Any]]
    responsive_models: List[str]
    actor_records: List[ReviewActorRecord]
    errors: List[str] = field(default_factory=list)

    @property
    def quorum_met(self) -> bool:
        from ouroboros.config import adaptive_quorum
        # actor_records holds ALL dispatched reviewers (responsive + errored), so
        # this honors the configured count: configured=1 & responded=1 -> met;
        # configured=3 & responded=1 -> NOT met (loud degraded surfaces).
        return len(self.responsive_models) >= adaptive_quorum(len(self.actor_records))

    @property
    def degraded_reasons(self) -> List[str]:
        degraded = [r for r in self.actor_records if r.status in {"error", "parse_failure", "partial"}]
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
    return ReviewActorRecord(
        model_id=model_label,
        status=status,
        raw_text=raw_text,
        parsed_items=parsed_items or [],
        tokens_in=int(actor.get("tokens_in", 0) or 0),
        tokens_out=int(actor.get("tokens_out", 0) or 0),
        cost_usd=float(actor.get("cost_estimate", 0.0) or 0.0),
        slot=idx + 1,
        slot_id=str(actor.get("slot_id") or ""),
        prompt_ref=dict(actor.get("prompt_ref") or {}),
        response_ref=dict(actor.get("response_ref") or {}),
        failure_code=str(actor.get("failure_code") or ""),
        reset_at=str(actor.get("reset_at") or ""),
        http_status=(int(actor["http_status"]) if isinstance(actor.get("http_status"), int) else None),
        transport_status=str(actor.get("transport_status") or ""),
        operation_id=str(actor.get("operation_id") or ""),
        operation_state=str(actor.get("operation_state") or "settled"),
        late_result_pending=bool(actor.get("late_result_pending")),
        pending_invocation_id=str(actor.get("pending_invocation_id") or ""),
        delegated_run_id=str(actor.get("delegated_run_id") or ""),
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
# (no surface policy lives here: acceptance rules, tier classification and
# quorum authority stay with their owners). Every surface not listed returns
# the findings ARRAY contract; ``object`` is the whole-object acceptance verdict
# (verdict/outcome_tier/criteria_used/dialogue_status/findings/summary), which
# the array-only canonicalizer used to reduce to its findings list; ``report``
# is a free-form markdown product that is never canonicalized or extracted.
REVIEW_OUTPUT_SHAPES: Dict[str, str] = {
    "task_acceptance": "object",
    "deep_self_review": "report",
}
OBJECT_VERDICT_REQUIRED_KEYS = ("verdict",)

# Default output contract per SHAPE for a retrieving delivery whose surface did
# not hand over its own `policy["output_contract"]`: the prompt must ask for the
# same form the canonicalizer parses, or an obedient reviewer answering the
# array contract on an object surface is demoted to malformed by its own host.
REVIEW_JSON_OBJECT_CONTRACT = """\
Return ONLY one JSON object with keys: verdict (PASS|FAIL|DEGRADED), findings
([{severity, item, evidence, recommendation}] — an empty list when nothing is
wrong), and summary, plus any keys the task's contract names. Never a bare
array, never prose around the object: your host parses the object structurally
and treats anything else as a non-response.
"""
REVIEW_REPORT_CONTRACT = """\
Deliver the report itself as plain markdown prose — no JSON wrapper, no code
fence around the whole answer. Most critical findings first, each with the
evidence you read; mark anything you could not verify as unverified.
"""


def review_output_shape(surface: str) -> str:
    """``array`` | ``object`` | ``report`` for one review surface."""
    return REVIEW_OUTPUT_SHAPES.get(str(surface or ""), "array")


def default_output_contract(shape: str) -> str:
    """The prompt contract a retrieving row is asked for when its surface hands
    over no ``policy["output_contract"]`` — keyed by the same shape the
    canonicalizer parses, so the ask and the parse can never disagree."""
    return {
        "object": REVIEW_JSON_OBJECT_CONTRACT, "report": REVIEW_REPORT_CONTRACT,
    }.get(str(shape or ""), REVIEW_JSON_ARRAY_CONTRACT)


def object_verdict_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """The WHOLE object verdict, or None — mirroring `parse_review_findings`'
    object ladder (a verdict signal is required, findings are optional) with
    the array branch's type discipline: a non-empty string verdict and, when
    present, findings as a list of dicts. Shape only: semantic demotion
    (tier/coach/criteria) stays with ``review_actor_aggregation``."""
    if not isinstance(payload, dict) or not all(key in payload for key in OBJECT_VERDICT_REQUIRED_KEYS):
        return None
    if not isinstance(payload.get("verdict"), str) or not payload["verdict"].strip():
        return None
    findings = payload.get("findings")
    if findings is not None and (
        not isinstance(findings, list) or not all(isinstance(item, dict) for item in findings)
    ):
        return None
    return payload


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

# Required-matrix: the parser rejects an empty array as missing coverage, so
# offering the sentinel here would contradict it.
REVIEW_JSON_MATRIX_CONTRACT = f"""\
Return ONLY a JSON array with one entry per required checklist item. Each element:
{_REVIEW_JSON_ELEMENT_SCHEMA}
There is no all-clear shortcut in this mode: an empty array is a non-response.
Report PASS explicitly for every item you reviewed and found clean.
"""


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


def parse_model_review_results(
    result_json: Dict[str, Any],
    *,
    required_items: Optional[Sequence[str]] = None,
) -> ParsedTriadReview:
    """Parse model result envelopes into normalized findings and actor records.

    ``required_items`` enforces the skill-review matrix contract: a reviewer
    that omits a checklist item is non-responsive for quorum.
    """
    findings: List[Dict[str, Any]] = []
    responsive: List[str] = []
    responsive_ids: set[str] = set()
    records: List[ReviewActorRecord] = []
    required = set(required_items or [])
    for idx, actor in enumerate(result_json.get("results") or []):
        if not isinstance(actor, dict):
            continue
        model = str(actor.get("model") or actor.get("request_model") or "").strip()
        raw_text = str(actor.get("text") or "")
        model_label = model or "reviewer"
        actor_slot_id = str(actor.get("slot_id") or "")
        if str(actor.get("operation_state") or "") == "not_dispatched":
            records.append(_actor_record(
                actor, idx=idx, model_label=model_label,
                status="not_dispatched", raw_text=raw_text,
            ))
            continue
        if str(actor.get("verdict") or "").upper() == "ERROR":
            records.append(_actor_record(actor, idx=idx, model_label=model_label, status="error", raw_text=raw_text))
            continue
        parsed = extract_json_array(raw_text, normalize=not required)
        if parsed is None:
            records.append(_actor_record(actor, idx=idx, model_label=model_label, status="parse_failure", raw_text=raw_text))
            continue
        if not required and not parsed and not empty_array_is_verified_clean(raw_text):
            # Anti-refusal coverage contract: an empty array counts as a real
            # "no findings" verdict only with the explicit NO_FINDINGS sentinel
            # (or a bare `[]`-only response). A `[]` buried in refusal prose
            # ("I cannot review this diff... []") must not enter the quorum as
            # a clean PASS.
            records.append(_actor_record(actor, idx=idx, model_label=model_label, status="parse_failure", raw_text=raw_text))
            continue
        actor_findings: List[Dict[str, Any]] = []
        covered_items: set[str] = set()
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            item = str(entry.get("item") or "")
            verdict = str(entry.get("verdict") or "").upper()
            if not item or verdict not in {"PASS", "FAIL"}:
                continue
            covered_items.add(item)
            actor_findings.append({
                "item": item,
                "verdict": verdict,
                "severity": str(entry.get("severity") or "advisory").lower(),
                "reason": str(entry.get("reason") or "").strip(),
                "model": model_label,
                "slot_id": actor_slot_id,
                **({"obligation_id": str(entry.get("obligation_id") or "")} if entry.get("obligation_id") else {}),
            })
        if required and not required.issubset(covered_items):
            records.append(_actor_record(actor, idx=idx, model_label=model_label, status="partial", raw_text=raw_text, parsed_items=actor_findings))
            continue
        findings.extend(actor_findings)
        responsive_id = f"{model_label} [{actor_slot_id}]" if actor_slot_id else f"{model_label}#{idx + 1}"
        response_identity = f"slot:{actor_slot_id}" if actor_slot_id else f"actor:{idx}"
        if response_identity not in responsive_ids:
            responsive.append(responsive_id)
            responsive_ids.add(response_identity)
        records.append(_actor_record(actor, idx=idx, model_label=model_label, status="responded", raw_text=raw_text, parsed_items=actor_findings))
    return ParsedTriadReview(findings=findings, responsive_models=responsive, actor_records=records)


def emit_review_model_error_events(ctx: Any, parsed: ParsedTriadReview, *, source: str, skill_name: str = "") -> None:
    """Persist model error / parse-failure events for observability."""
    try:
        log_path = ctx.drive_logs() / "events.jsonl"
    except Exception:
        return
    for record in parsed.actor_records:
        if record.status not in {"error", "parse_failure", "partial"}:
            continue
        if source == "skill_review":
            note = (
                "Full raw response preserved in review.json raw_actor_records "
                "when quorum succeeds; otherwise in review_history.jsonl."
            )
        else:
            note = "Full raw response preserved in triad_raw_results."
        try:
            append_jsonl(log_path, {
                "ts": utc_now_iso(),
                "type": "review_model_error",
                "source": source,
                "skill": skill_name,
                "model": record.model_id,
                "status": record.status,
                "error_note": note,
            })
        except Exception:
            pass


# ── Reviewer prompt rule texts (byte-stable governance, cache-marked) ─────────
# Moved from review_substrate._render_prompt_parts in v6.74.0: the substrate
# crossed the 1600-line module gate, and these shared reviewer rule texts are
# multi-model review primitives — this module's charter. They MUST stay
# byte-stable across a surface's repeat calls (they live inside the cache-marked
# governance segment).

TIER_CLASSIFICATION_RULES = (
        "outcome_tier classifies the CURRENT deliverable and completion_coach is the single "
        "highest-value change that would move it one tier up. Never classify solved unless the "
        "claimed result is actually verified by the evidence — your veto over false success "
        "claims is the point of this review. A real partial deliverable with honestly marked "
        "gaps is best_effort, not a failure. "
)

ACCEPTANCE_SURFACE_RULES = (
        "For TASK ACCEPTANCE: do not accept a 'solved' claim on assertion alone. Re-derive the "
        "acceptance criteria from the FULL goal/spec narrative yourself (including prose context, "
        "requirements implied outside bullet lists, and any named user-facing interface), enumerate "
        "all affected components/surfaces you expect the deliverable to cover. If "
        "`task_contract.acceptance_claims` is present, treat those "
        "claim/surface/support entries as structured hints, not as an oracle: verify them against "
        "the full goal and use `acceptance_support_refs` as the host-built support map. The claim's "
        "`support` text names expected evidence; it is NOT evidence unless a receipt/tool/artifact/source "
        "ref actually supports it. "
        "Record in criteria_used which claims were supported, missing, partial, or rejected, and why. "
        "Require per-criterion evidence: an EXECUTED check that MIRRORS what the real grader would run "
        "(the actual test/command and its observed output) — not a narrative that it passes. "
        "If the evidence covers only part of the affected surface list, do not classify solved; "
        "name the uncovered criteria/surfaces in findings or completion_coach. "
        "EVIDENCE INDEPENDENCE: a passing test is only credible if it is not graded by the "
        "agent's own hand. From the diff and tool trace, identify which test/check files the "
        "agent CREATED or MODIFIED this turn versus which were pre-existing or grader-owned; if "
        "the only passing evidence comes from tests the agent wrote or edited this same turn, "
        "treat the success claim as UNVERIFIED (outcome_tier at most best_effort, never solved) "
        "and completion_coach must name an independent check (the pre-existing suite, the grader "
        "command, or a behavior the agent did not also author). State in your summary which "
        "evidence you judged independent. "
        "VISIBLE UI EVIDENCE: when the deliverable changes a user-visible interface, require "
        "evidence that at least one relevant real consumer flow was opened in an available browser "
        "and the rendered result was actually inspected with vision. A screenshot file or attachment "
        "without evidence of visual inspection is insufficient for solved. The implementer chooses "
        "states, viewports, and additional engines according to task risk; mobile and WebKit are not "
        "universal requirements, and an unavailable optional engine alone is not degradation. If "
        "visual evidence the implementer judged necessary was unavailable, require an honest "
        "best_effort/degraded result that names the gap. "
        "ENVIRONMENT vs DELIVERABLE: a task_environment_error, round-budget exhaustion, sandbox "
        "auto-evaluation, or provider/runtime fault is NOT itself an agent failure — judge "
        "whether the requested artifact/answer was produced before the environment terminated; "
        "do not FAIL a correct deliverable for an environment-imposed limit, note it as context. "
        "ABSENT-PREMISE / INFEASIBLE DISPOSITION: when the terminal claim is that the task's "
        "premise is absent or the request is infeasible (a required feature, referent, or mode "
        "of operation does not exist on the inspected surfaces, or every remaining route would "
        "breach the task's own stated method restrictions), the deliverable to judge is the "
        "PREMISE ARGUMENT, not the named artifact: verify from the trace which concrete surfaces "
        "were inspected and what each demonstrably cannot do, and whether any in-scope route "
        "remains untried. Under that disposition the named artifact's absence is EXPECTED — do "
        "not instantiate 'the deliverable exists' as a criterion against it (that begs the "
        "question), and never coach continuation whose remaining routes breach the task's stated "
        "restrictions: a route outside 'using only X' is not an actionable route, and demanding "
        "it manufactures an artifact the task forbids. A WEAK premise argument still FAILs on "
        "its own grounds: absence asserted without inspecting the surface that would host the "
        "feature, or an untried in-scope route visible in the packet. "
        "METRIC-GROUNDED EVIDENCE: when the task states an explicit success metric, contains a "
        "worked example, or warns that grading uses held-out cases, treat an existence-only "
        "(artifact_observation) or substring-only verification receipt as INSUFFICIENT for solved — "
        "require evidence that the metric/example is actually met (an exact/exact_line/json_equals "
        "receipt, or the metric value in the check output). ANTI-CHEAT: credible verification uses "
        "ONLY public task info (instruction text, embedded examples, installed oracles, the agent's "
        "own independent checks); if the evidence came from reading a hidden /tests/ dir, "
        "solution.sh, copied verifier code, or an online answer, treat the success claim as "
        "UNVERIFIED. "
        "PROCESS, NOT ONLY OUTCOME: the packet includes a `tool_trajectory` (HOW the task was "
        "solved) and a first-class `verification_summary`. Audit the process — if the agent used "
        "the wrong tool, went the wrong direction, has a red verification never cleared by a green "
        "re-run of the SAME criterion/check "
        "(`verification_summary.unreconciled_red`, or a RED `latest_status`; "
        "`reconciliation_identity` names, per receipt, WHICH authority clears it — "
        "`criterion_id` (an authored criterion name), `check` (the canonical command text), "
        "`artifact_paths` (the observed path SET, compared byte-for-byte, since a leading or "
        "trailing space is a legal filename byte) or `none`; for `check`, `check_rendering` says "
        "which renderer wrote the stored text, and receipts from DIFFERENT renderings are never "
        "the same verification even when the text matches (an `unversioned` receipt predates the "
        "current one and cannot be re-tokenized, so it is unknown, not equal). "
        "`expected_whitespace_normalized` "
        "is set for `check` ONLY, and means the same tokens however spaced still count as the "
        "same check, while a changed ARGUMENT is a different check — so judge the substance and "
        "not the command spelling. It is never set for the other identities, which normalize no "
        "text at all, nor for a masked check, whose reconciliation ignores command text "
        "entirely. The still-red verification is NOT necessarily "
        "the last one — a later green of a DIFFERENT check leaves it standing — so read "
        "`unreconciled_red_identity` for WHICH criterion/check/observed paths are still red, not "
        "`latest_*`), grounded on a check "
        "whose exit code may be MASKED (`verification_summary.check_exit_masking_unreconciled` — a "
        "`| tail`/`grep`/`|| true` pipeline can report exit 0 over a real failure; command text "
        "NEVER decides here, because dropping the masking pipe IS the fix and so necessarily "
        "changes it: a masked check that NAMES a `criterion_id` is cleared only by a later CLEAN "
        "verification naming that SAME id — a later clean receipt that omits its id does not clear "
        "it — and a masked check naming NO criterion is cleared by ANY later clean verification; "
        "so that green is weak evidence until such a clean re-grounding appears), or the "
        "final claim "
        "is not supported by the trajectory, say so: a deliverable that looks superficially "
        "correct but was reached the wrong way, or that contradicts the agent's own checks, is at "
        "most best_effort, and completion_coach must name the process fix. PROVENANCE: every "
        "evidence block is tagged in `__provenance__` (host_attested / agent_supplied / "
        "tool_result / artifact / hidden_or_restricted) — weigh host_attested over agent_supplied, "
        "and NEVER credit a success claim to `hidden_or_restricted` evidence (a benchmark/test leak). "
        "RETRIEVAL: `retrieval` (host_attested, when present) records NATIVE provider web searches "
        "made inside the ANSWERING model's own request — how many, and the URLs it fetched. It is "
        "FACTUAL CONTEXT, not a criterion: its absence means only that no native search was recorded "
        "(native search may not be enabled, or the agent may have retrieved through the `web_search` "
        "/ browser tools, which issue their own calls and are visible in `tool_trajectory`), and "
        "must NOT be treated as a gap; its presence is not credit either — judge the substance of "
        "the evidence as usual. "
        "OBLIGATION REBUTTALS: `acceptance_obligations` (host_attested) lists the id/item/"
        "recommendation of each obligation a prior review round raised; the agent's per-obligation "
        "dispositions and rebuttal reasons arrive under `agent_supplied.agent_decision."
        "obligation_dispositions`, joinable by id. Treat a 'rejected' disposition as a rebuttal to "
        "that finding: if the argument is genuinely valid, do not re-raise the same finding; if it "
        "is not, re-raise the finding and explain why the rebuttal fails. A rebuttal may dismiss or "
        "reframe an obligation, but it is NEVER itself evidence for a criterion — 'supported' still "
        "requires an independent host/tool/artifact receipt and 'solved' still requires the real "
        "grader-mirroring check to pass. "
        "REVIEW REGISTER: you are an outside perspective helping the task land, not a gate of last "
        "resort. Concrete, reachable-now defects block; hypotheticals, style preferences, and hygiene "
        "are advisory findings, not blockers. "
        "REACHABILITY: a requirement that cannot be satisfied in THIS environment (a missing "
        "credential/UI/network/tool, an owner-side dependency) is unavailable here — classify the "
        "tier honestly, name the gap, and do not re-raise it as a blocking finding. "
        "OBLIGATION IDENTITY: every finding carries disposition_kind — \"re_raise\" when it maintains "
        "an obligation already listed in `acceptance_obligations` (then obligation_id MUST be that "
        "exact id), \"new\" otherwise. A re_raise with a missing or unknown obligation_id is recorded "
        "as new, so name ids precisely. When the agent's disposition rejected an obligation, "
        "adjudicate the rebuttal: if the argument is valid, retire the finding — do not re-raise it; "
        "if it is not, re_raise it and state in `evidence` why the argument fails. "
        "DIALOGUE STATUS: dialogue_status is your typed judgement about the review DIALOGUE itself — "
        "\"continue_actionable\" (a concrete, reachable-now improvement exists), \"unreachable_here\" "
        "(the remaining gap cannot be closed in this environment), or \"stable_disagreement\" (both "
        "positions are fully argued and further rounds will not change the outcome). An honest "
        "terminal judgement ends the loop and surfaces both positions to the owner; it is not a "
        "concession. "
)


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
    payload = {"error": error, "usage": {}, "slot_id": slot_id, "prompt_ref": {}, "response_ref": {}}
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
            payload={"error": error},
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

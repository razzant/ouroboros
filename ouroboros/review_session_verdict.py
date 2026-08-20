"""Typed verdict for a delegated review session (D19 / plan 5.4): the session
output schema, its per-surface shaping, and the schema-first / strict-parse /
light-model-extraction canonicalization rail. Extracted from
ouroboros/review_execution.py (v7 L-C split); review_execution.py re-exports
every name."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ouroboros.triad_review import (
    REVIEW_JSON_ARRAY_CONTRACT,
    empty_array_is_verified_clean,
    extract_json_array,
)

# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("review_execution")


# ---------------------------------------------------------------------------
# Typed verdict for a delegated session (D19 / plan 5.4).
# ---------------------------------------------------------------------------

# The ASK: sent as ``outputSchema`` only when the EFFECTIVE route can carry it
# (D19) — judged on the pinned harness's live manifest, never on the static
# adapter flag alone, because the flag describes the adapter and not the
# transport this run actually rides. The run's own reported
# ``outputConformance == "passed"`` is the only thing that lets the structured
# payload be TRUSTED as the verdict (never run success).
REVIEW_SESSION_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["item", "verdict", "severity", "reason"],
                "properties": {
                    "item": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
                    "severity": {"type": "string", "enum": ["critical", "advisory"]},
                    "reason": {"type": "string"},
                    "obligation_id": {"type": "string"},
                },
            },
        },
    },
}


def review_session_output_schema(surface: str) -> Dict[str, Any]:
    """The session verdict schema, shaped to the SURFACE's own clean contract.

    The shared schema admits ``{"findings": []}`` — the honest clean verdict for a
    triad or ordinary advisory reviewer. Scope's coverage contract requires all
    eight checklist rows (PASS included), so its schema demands ``minItems: 1`` —
    a conforming engine refuses the empty answer up front instead of the gate
    discovering a ``parse_failure`` after the run. Advisory keeps the clean-capable
    shared schema (coverage is checked downstream by ``_check_expected_items``).
    """
    if surface == "plan_review":
        # plan review's own element contract (4e133c8a): the generic item/verdict shape
        # would conform-and-launder — an unknown class demotes to a note.
        from ouroboros.tools.plan_spec import PLAN_REVIEW_SESSION_OUTPUT_SCHEMA

        return PLAN_REVIEW_SESSION_OUTPUT_SCHEMA
    if surface != "scope_review":
        return REVIEW_SESSION_OUTPUT_SCHEMA
    shaped = json.loads(json.dumps(REVIEW_SESSION_OUTPUT_SCHEMA))
    shaped["properties"]["findings"]["minItems"] = 1
    return shaped

_UNEXTRACTABLE = "UNEXTRACTABLE"

# The light model canonicalizes NARRATIVE to the review's own output contract —
# bare ``[]`` or a findings array — so a clean verdict from a session survives
# exactly as a findings verdict does (D19 closed the asymmetry where a session
# could BLOCK but never CLEAR). It is the ONE sanctioned second-model use (§8
# item 6 exception): it extracts; it never judges, repairs, summarizes or
# attests, and a transcript that is not a completed review comes back
# UNEXTRACTABLE rather than as an invented verdict.
_SESSION_EXTRACT_PROMPT = (
    "The text below is the final answer of a delegated code-review session.\n"
    "Canonicalize its verdict. Reply with EXACTLY ONE of:\n"
    "1. [] — when the reviewer COMPLETED the review and explicitly reports no findings\n"
    "   (a clean verdict). Never reply [] for a refusal, an error, or an unfinished review.\n"
    "2. ONLY the JSON array of the findings the reviewer reported, copied faithfully\n"
    "   into the review's own output contract (below). Never invent, merge or drop findings.\n"
    f"3. The single word {_UNEXTRACTABLE} — when the text is not a completed review\n"
    "   (a refusal, an error dump, an unfinished session, or anything you cannot map\n"
    "   faithfully onto the contract).\n"
    "No prose, no markdown fences.\n\n"
    "The review's output contract was:\n{contract}\n\n"
    "Session answer to canonicalize:\n{raw_text}\n"
)

# The extraction rail reads the session answer WHOLE — no head/tail window.
# A windowed read is not a smaller extraction, it is a different (fabricated)
# verdict: findings reported mid-transcript vanish, and the light model
# faithfully canonicalizes the cut it was shown into a clean/partial verdict.
# The single physical send still has one hard bound, because a light model's
# context is finite: the engine caps a text artifact at 4 MiB while common
# light-model windows hold ~100k tokens, so past this bound extraction REFUSES
# with the typed ``extraction_incomplete`` disposition — the raw transcript
# survives for forensics and can never read as clean — instead of silently
# shrinking the artifact.
_EXTRACT_MAX_CHARS = 400_000


def _findings_array(payload: Any) -> Optional[List[Dict[str, Any]]]:
    """The findings list inside a structured payload, or None when it has none."""
    if isinstance(payload, dict):
        payload = payload.get("findings")
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    return None


def _strictly_parseable(text: str) -> bool:
    """Would the surfaces' own strict parsers accept this text as a verdict?

    The strict path comes FIRST (D19): a session that already obeyed the output
    contract is passed through byte-identical, and the constitutional
    ``empty_array_is_verified_clean`` predicate stays untouched — its strictness
    is the reason extraction exists, not a defect extraction papers over.

    The WHOLE answer must BE the payload. This used to SCAN with
    ``extract_json_array``, so any JSON array of objects appearing anywhere in a
    transcript made it "strict" — a refusal that quoted the contract's own
    example ("I reviewed NOTHING. The contract asked for entries like
    [{"item": ..., "verdict": "PASS", ...}]") was passed through byte-identical
    as a TRUSTED verdict, and the extraction rail that exists precisely to
    canonicalize a non-verdict never ran. Requiring the whole text removes that
    leniency; it matches the discipline ``empty_array_is_verified_clean``
    already applies, and a narrative falls through to extraction, which is what
    extraction is for.
    """
    body = str(text or "")
    if empty_array_is_verified_clean(body):
        return True
    try:
        parsed = json.loads(body.strip())
    except (TypeError, ValueError):
        return False
    return bool(parsed) and isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed)


def canonicalize_session_verdict(
    raw_text: str,
    *,
    conformance_passed: bool,
    contract: str = "",
    llm: Any = None,
) -> tuple[str, str, Dict[str, Any]]:
    """Return ``(canonical_text, method, extraction_usage)`` for a session answer.

    Order is the owner's (D19): trusted structured output first (gated on the
    run's ``outputConformance == "passed"``, never on run success), then the
    strict parser, then LIGHT-MODEL extraction over the WHOLE answer.
    Extraction is not a review call: it runs under its OWN one-send physical
    rail so it can never consume the reviewing actor's permitted sends, and it
    spends no reviewer slot. An answer too large for the one-send rail is the
    typed ``extraction_incomplete`` — never a windowed read, whose canonical
    form would be a verdict fabricated from the visible cut. ``method`` is one
    of ``schema | strict | light_model_extraction | extraction_incomplete |
    unparsed``.
    """
    text = str(raw_text or "")
    if conformance_passed:
        try:
            payload = json.loads(text.strip())
        except (TypeError, ValueError):
            payload = None
        findings = _findings_array(payload)
        if findings is not None:
            return ("[]" if not findings else json.dumps(findings, ensure_ascii=False)), "schema", {}
        # The engine claimed conformance over a payload that does not carry the
        # contract's shape: fall through to the honest branches, and the caller
        # discloses the delta.
    if _strictly_parseable(text):
        return text, "strict", {}
    if len(text) > _EXTRACT_MAX_CHARS:
        return text, "extraction_incomplete", {}
    canonical, usage = _extract_verdict_via_light_model(text, contract=contract, llm=llm)
    if canonical is not None:
        return canonical, "light_model_extraction", usage
    # `unparsed` is the honest end of THIS layer's knowledge. The coordinator's
    # own fenced scanner may still parse the text downstream; labeling that
    # here would need either a duplicate parser (drift) or a backward import
    # of the coordinator (the one-way seam ARCHITECTURE pins) — both cost more
    # than the telemetry cosmetics are worth. Disclosed residual: a fenced
    # verdict that lands downstream is telemetered `unparsed` at this layer.
    return text, "unparsed", usage


def _extract_verdict_via_light_model(
    raw_text: str, *, contract: str = "", llm: Any = None,
) -> tuple[Optional[str], Dict[str, Any]]:
    """One bounded light-model call canonicalizing narrative to the contract."""
    from ouroboros.config import get_light_model
    from ouroboros.usage_accounting import physical_attempt_limit

    if not str(raw_text or "").strip():
        return None, {}
    model = get_light_model()
    prompt = _SESSION_EXTRACT_PROMPT.format(
        contract=contract or REVIEW_JSON_ARRAY_CONTRACT,
        raw_text=raw_text,  # WHOLE — the caller already bounded the one send
    )
    try:
        if llm is None:
            from ouroboros.llm import LLMClient

            llm = LLMClient()
        from dataclasses import replace as _replace

        from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

        # A FRESH one-send rail: the extraction must not claim a send from the
        # reviewing actor's two-physical-send rail (D19 — not a review call).
        # The ledger row keeps the actor's task/category attribution but is
        # sub-labeled `review_substrate.extraction`, so the small light-model
        # rows beside the $0.00 subscription settlements read as what they are —
        # verdict extraction, not review-slot spend.
        _scope = _replace(current_usage_scope() or UsageScope(),
                          source="review_substrate.extraction")
        with physical_attempt_limit(1), usage_scope(_scope):
            message, usage = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=8192,
                reasoning_effort="low",
                no_proxy=True,
            )
    except Exception as exc:
        log.warning("Review session verdict extraction failed: %s", exc)
        return None, {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, list):
        content = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    body = str(content or "").strip()
    usage = dict(usage or {})
    usage["model"] = model
    if not body or _UNEXTRACTABLE in body.upper()[:80]:
        return None, usage
    if empty_array_is_verified_clean(body):
        return "[]", usage
    findings = _findings_array(extract_json_array(body))
    if findings is None:
        try:
            findings = _findings_array(json.loads(body))
        except (TypeError, ValueError):
            findings = None
    if findings is None:
        return None, usage
    return ("[]" if not findings else json.dumps(findings, ensure_ascii=False)), usage

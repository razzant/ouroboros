"""Canonicalizing a retrieving reviewer's answer to the review output contract.

Split out of ``review_execution`` for module altitude (P7): both the delegated
session route and the native tool-round route feed their collected answers
through this ONE seam — trusted structured output first, then the strict
parser, then bounded light-model extraction over the WHOLE answer. Extraction
extracts; it never judges, repairs, summarizes or attests (D19).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ouroboros.config import get_finalization_grace_sec
from ouroboros.deadline_utils import owner_deadline_exhausted, review_transport_timeout
from ouroboros.triad_review import (
    default_output_contract,
    empty_array_is_verified_clean,
    extract_fenced_json,
    extract_json_array,
    object_verdict_payload,
)

log = logging.getLogger("review_execution")

_UNEXTRACTABLE = "UNEXTRACTABLE"

# The OBJECT variant of the extraction ask (task acceptance): the whole verdict
# object of the contract, never its findings list alone — a findings-only
# canonical form loses verdict/outcome_tier/criteria_used/dialogue_status and
# the coordinator then demotes a completed review to malformed.
_SESSION_EXTRACT_OBJECT_PROMPT = (
    "The text below is the final answer of a delegated review session.\n"
    "Canonicalize its verdict. Reply with EXACTLY ONE of:\n"
    "1. ONLY the JSON object of the reviewer's verdict, copied faithfully into the\n"
    "   review's own output contract (below) — every key the reviewer reported,\n"
    "   findings included. Never invent, merge or drop anything.\n"
    f"2. The single word {_UNEXTRACTABLE} — when the text is not a completed review\n"
    "   (a refusal, an error dump, an unfinished session, or anything you cannot map\n"
    "   faithfully onto the contract).\n"
    "No prose, no markdown fences.\n\n"
    "The review's output contract was:\n{contract}\n\n"
    "Session answer to canonicalize:\n{raw_text}\n"
)

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


def _strictly_parseable(text: str, shape: str = "array") -> bool:
    """Would the surfaces' own strict parsers accept this text as a verdict?

    The strict path comes FIRST (D19): a session that already obeyed the output
    contract is passed through byte-identical, and the constitutional
    ``empty_array_is_verified_clean`` predicate stays untouched — its strictness
    is the reason extraction exists, not a defect extraction papers over.

    ``shape`` gates the predicate: for an OBJECT contract (task acceptance) the
    whole text must BE the verdict object — a bare ``[]`` is not a clean object
    verdict, and letting the array predicate answer for it would route an
    acceptance answer into the array ladder with no verdict/tier/dialogue keys.

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
    if shape == "object":
        try:
            return object_verdict_payload(json.loads(body.strip())) is not None
        except (TypeError, ValueError):
            return False
    if empty_array_is_verified_clean(body):
        return True
    try:
        parsed = json.loads(body.strip())
    except (TypeError, ValueError):
        return False
    return bool(parsed) and isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed)


def _canonical_payload_text(payload: Any, shape: str) -> Optional[str]:
    """Canonical text of a structured payload for ``shape``, or None when it
    does not carry the contract's shape."""
    if shape == "object":
        verdict = object_verdict_payload(payload)
        return None if verdict is None else json.dumps(verdict, ensure_ascii=False)
    findings = _findings_array(payload)
    if findings is None:
        return None
    return "[]" if not findings else json.dumps(findings, ensure_ascii=False)


def canonicalize_session_verdict(
    raw_text: str, *, conformance_passed: bool, contract: str = "", llm: Any = None,
    deadline_at: Any = None, transport_timeout_sec: Any = None, shape: str = "array",
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
    of ``report | schema | strict | light_model_extraction |
    extraction_incomplete | unparsed``.

    ``shape`` (``triad_review.review_output_shape``) is the contract's FORM:
    ``array`` (findings list), ``object`` (the whole acceptance verdict object,
    kept whole on every branch — never reduced to its findings) or ``report``
    (a free-form product that is passed through verbatim; nothing here may
    turn a diagnosis into a findings array).
    """
    text = str(raw_text or "")
    if shape == "report":
        return text, "report", {}
    if conformance_passed:
        try:
            payload = json.loads(text.strip())
        except (TypeError, ValueError):
            payload = None
        canonical = _canonical_payload_text(payload, shape)
        if canonical is not None:
            return canonical, "schema", {}
        # The engine claimed conformance over a payload that does not carry the
        # contract's shape: fall through to the honest branches, and the caller
        # discloses the delta.
    if _strictly_parseable(text, shape):
        return text, "strict", {}
    if len(text) > _EXTRACT_MAX_CHARS:
        return text, "extraction_incomplete", {}
    canonical, usage = _extract_verdict_via_light_model(
        text, contract=contract, llm=llm, deadline_at=deadline_at,
        transport_timeout_sec=transport_timeout_sec, shape=shape)
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
    raw_text: str, *, contract: str = "", llm: Any = None, deadline_at: Any = None,
    transport_timeout_sec: Any = None, shape: str = "array",
) -> tuple[Optional[str], Dict[str, Any]]:
    """One bounded light-model call canonicalizing narrative to the contract."""
    from ouroboros.config import get_light_model
    from ouroboros.usage_accounting import physical_attempt_limit

    if not str(raw_text or "").strip():
        return None, {}
    model = get_light_model()
    if owner_deadline_exhausted(deadline_at=deadline_at, reserve_sec=get_finalization_grace_sec()):
        return None, {"model": model, "reason_code": "deadline_exhausted", "dispatch": "not_dispatched"}
    template = _SESSION_EXTRACT_OBJECT_PROMPT if shape == "object" else _SESSION_EXTRACT_PROMPT
    prompt = template.format(
        contract=contract or default_output_contract(shape),
        raw_text=raw_text,  # WHOLE — the caller already bounded the one send
    )
    # Formatting the extraction prompt can itself be expensive. Re-check at
    # the physical light-model boundary rather than letting an expired owner
    # window reach the transport helper's positive floor.
    if owner_deadline_exhausted(deadline_at=deadline_at, reserve_sec=get_finalization_grace_sec()):
        return None, {"model": model, "reason_code": "deadline_exhausted", "dispatch": "not_dispatched"}
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
        _scope = _replace(current_usage_scope() or UsageScope(), source="review_substrate.extraction")
        chat_kwargs = dict(
            messages=[{"role": "user", "content": prompt}], model=model,
            max_tokens=8192, reasoning_effort="low", no_proxy=True,
        )
        transport = review_transport_timeout(model, transport_timeout_sec, deadline_at)
        if transport is not None:
            chat_kwargs["timeout"] = transport
        with physical_attempt_limit(1), usage_scope(_scope):
            message, usage = llm.chat(**chat_kwargs)
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
    if shape == "object":
        try:
            payload: Any = json.loads(body)
        except (TypeError, ValueError):
            payload = extract_fenced_json(body)
        return _canonical_payload_text(payload, shape), usage
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

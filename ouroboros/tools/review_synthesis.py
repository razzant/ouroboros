"""Shared pure synthesis and normalization for reviewer outputs.

Commit-review claims use the optional LLM deduplicator; the plan-review engine
(``tools/plan_review.py`` + ``plan_spec``/``plan_packet``) keeps only the shared
control-line prefix, the mixed-window input-cap helpers, the cache-friendly
message pair and the usage-emission shim here.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ouroboros.triad_review import extract_json_array
from ouroboros.tools.review_helpers import emit_review_usage

log = logging.getLogger(__name__)

# Bound cost and avoid mixed canonical/raw output on oversized finding sets.
_MAX_CLAIMS_FOR_SYNTHESIS = 30

_MIN_CLAIMS_FOR_SYNTHESIS = 2

_SYNTHESIS_PROMPT_TEMPLATE = (
    "You are a code-review claim synthesizer. You receive a list of raw findings\n"
    "from multiple independent reviewers answering one two-part brief (the change\n"
    "itself, and its coupling to the rest of the repository). Your job is to\n"
    "produce a deduplicated canonical list.\n"
    "\n"
    "## Rules\n"
    "\n"
    "1. Merge claims that share the same **root cause** in the same file/symbol\n"
    "   into ONE canonical entry. Use the most specific/concrete reason text.\n"
    "2. **Do NOT merge** findings about genuinely different bugs, even if they are\n"
    "   in the same file. One root cause = one canonical issue.\n"
    "3. If an incoming claim already carries an `obligation_id` that matches an\n"
    "   open obligation from a previous round (provided below), PRESERVE that\n"
    "   `obligation_id` on the canonical entry. This allows durable obligations\n"
    "   to survive across retries without ID rotation.\n"
    "4. If no existing obligation matches, leave `obligation_id` as \"\" — a new\n"
    "   obligation will be assigned downstream.\n"
    "5. Do NOT invent new findings. Only deduplicate what you have been given.\n"
    "6. For each canonical entry, list `evidence_from_reviewers`: which reviewer(s)\n"
    "   independently flagged this issue (use the `tag` or `model` field if present).\n"
    "7. Output ONLY valid JSON — a JSON array of canonical findings, no markdown fences,\n"
    "   no prose outside the array.\n"
    "\n"
    "## Output format (each element)\n"
    "\n"
    '{"item": "<checklist item name>", "severity": "critical|advisory",\n'
    ' "reason": "<most concrete reason>", "obligation_id": "<existing id or empty>",\n'
    ' "evidence_from_reviewers": ["<tag/model1>", "<tag/model2>"]}\n'
    "\n"
    "## Open obligations from previous rounds (match by item + reason similarity)\n"
    "\n"
    "OPEN_OBLIGATIONS_PLACEHOLDER\n"
    "\n"
    "## Raw reviewer claims to deduplicate\n"
    "\n"
    "CLAIMS_PLACEHOLDER\n"
    "\n"
    "Respond with ONLY the JSON array. No explanation.\n"
)


def _redact(text: str) -> str:
    """Redact secret-like values from a string before including it in an LLM prompt."""
    try:
        from ouroboros.tools.review_helpers import redact_prompt_secrets
        redacted, _ = redact_prompt_secrets(str(text or ""))
        return redacted
    except Exception:
        return ""


def _format_obligations(open_obligations: List[Any]) -> str:
    """Render open obligations as compact secret-redacted JSON."""
    if not open_obligations:
        return "[]"
    from ouroboros.utils import truncate_review_artifact
    items = []
    for o in open_obligations:
        raw_reason = str(getattr(o, "reason", "") or "")
        redacted_reason = _redact(raw_reason)
        items.append({
            "obligation_id": str(getattr(o, "obligation_id", "") or ""),
            "item": str(getattr(o, "item", "") or ""),
            "reason_excerpt": truncate_review_artifact(redacted_reason, limit=500),
        })
    try:
        return json.dumps(items, ensure_ascii=False, indent=2)
    except Exception:
        return "[]"


def _format_claims(findings: List[Dict[str, Any]]) -> str:
    """Render raw findings as compact JSON with secret-redacted reasons."""
    try:
        safe = []
        for f in findings:
            entry = dict(f)
            if "reason" in entry:
                entry["reason"] = _redact(str(entry["reason"] or ""))
            safe.append(entry)
        return json.dumps(safe, ensure_ascii=False, indent=2)
    except Exception:
        return "[]"


def _normalize_evidence(value: Any) -> List[str]:
    """Normalize evidence_from_reviewers without splitting bare strings into chars."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if isinstance(v, str)]
    return []


def _parse_synthesis_output(raw: str) -> Optional[List[Dict[str, Any]]]:
    """Parse the synthesizer's JSON array response. Returns None on failure."""
    if not raw:
        return None
    parsed = extract_json_array(raw)
    if not isinstance(parsed, list):
        return None
    result = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        if not entry.get("item"):
            continue
        canonical = {
            "item": str(entry.get("item", "") or ""),
            # The synthesizer's INPUT is exclusively critical findings, so a
            # missing severity must stay critical — an "advisory" default
            # silently downgraded blocking findings out of the gate.
            "severity": str(entry.get("severity", "critical") or "critical"),
            "reason": str(entry.get("reason", "") or ""),
            "obligation_id": str(entry.get("obligation_id", "") or ""),
            "evidence_from_reviewers": _normalize_evidence(entry.get("evidence_from_reviewers")),
            # FAIL default ensures synthesized findings create obligations downstream.
            "verdict": str(entry.get("verdict", "") or "FAIL"),
        }
        for key in ("tag", "model"):
            if key in entry:
                canonical[key] = entry[key]
        result.append(canonical)
    return result if result else None


def synthesize_to_canonical_issues(
    critical_findings: List[Dict[str, Any]],
    *,
    open_obligations: Optional[List[Any]] = None,
    ctx: Any = None,
) -> List[Dict[str, Any]]:
    """Return deduplicated findings, or original findings on any synthesis failure."""
    if not critical_findings:
        return critical_findings

    if len(critical_findings) < _MIN_CLAIMS_FOR_SYNTHESIS:
        return critical_findings

    # Oversized sets pass through unchanged; no hybrid canonical/raw tail.
    if len(critical_findings) > _MAX_CLAIMS_FOR_SYNTHESIS:
        log.debug(
            "review_synthesis: %d claims exceeds limit %d — skipping synthesis, "
            "returning original findings unchanged",
            len(critical_findings),
            _MAX_CLAIMS_FOR_SYNTHESIS,
        )
        return critical_findings

    obligations = list(open_obligations or [])

    try:
        prompt = (
            _SYNTHESIS_PROMPT_TEMPLATE
            .replace("OPEN_OBLIGATIONS_PLACEHOLDER", _format_obligations(obligations))
            .replace("CLAIMS_PLACEHOLDER", _format_claims(critical_findings))
        )
    except Exception as exc:
        log.warning("review_synthesis: failed to build prompt: %s", exc)
        return critical_findings

    try:
        raw_response = _call_synthesis_llm(prompt, ctx=ctx)
    except Exception as exc:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(exc)
        log.warning("review_synthesis: LLM call raised exception: %s — using original findings", exc)
        return critical_findings

    if raw_response is None:
        log.warning("review_synthesis: LLM call returned None — using original findings")
        return critical_findings

    canonical = _parse_synthesis_output(raw_response)
    if canonical is None:
        log.warning("review_synthesis: failed to parse LLM output — using original findings")
        return critical_findings

    log.debug(
        "review_synthesis: %d raw → %d canonical",
        len(critical_findings),
        len(canonical),
    )
    return canonical


def _call_synthesis_llm(prompt: str, *, ctx: Any = None) -> Optional[str]:
    """Call the light LLM and emit usage so synthesis spend is accounted."""
    try:
        from ouroboros.config import get_light_model
        from ouroboros.llm import LLMClient

        model = get_light_model()

        client = LLMClient()

        # no_proxy avoids macOS fork-safety crashes in worker processes.
        msg, usage = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            model_role="light",
            max_tokens=16384,
            reasoning_effort="low",
            no_proxy=True,
        )

        if _has_billable_usage(usage):
            resolved_model = str((usage or {}).get("resolved_model") or "") or model
            provider = str((usage or {}).get("provider") or "") if isinstance(usage, dict) else ""
            emit_review_usage(
                ctx,
                model=resolved_model,
                usage=usage,
                source="review_synthesis",
                provider=provider,
            )

        if not msg:
            return None
        content = msg.get("content") if isinstance(msg, dict) else None
        if not content:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ]
            return "\n".join(t for t in texts if t) or None
        return str(content) if content else None

    except Exception as exc:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(exc)
        log.warning("review_synthesis: LLM call failed: %s", exc)
        return None


def _has_billable_usage(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    return any(
        usage.get(key)
        for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "cost", "total_cost")
    )


PLAN_REVIEW_CONTROL_PREFIX = "PLAN_REVIEW_CONTROL_JSON: "


def quorum_input_token_limit(models: Any, slot_limits: Any) -> int:
    """Assembly budget for ONE shared prompt fanned across mixed-window slots: the largest cap that
    still leaves a review QUORUM callable, i.e. the quorum-th largest per-slot cap.

    The global minimum let the SMALLEST window dictate the Atlas for everyone — with caps
    [545K, 745K, 745K] and quorum 2 it discarded ~200K of context the two large reviewers could
    have read, and refused an irreducible 600K prompt that those two would have accepted. Here the
    small slot drops OUT of the quorum instead of shrinking it: the caller records it as a typed
    ``preflight_oversize`` result, so a slot that cannot fit is REPORTED as not participating,
    never silently ignored. Quorum is counted over the CONFIGURED slots, so an unavailable or
    uncalibrated slot reads 0 and simply cannot be part of the quorum that justifies a bigger prompt."""
    from ouroboros.config import adaptive_quorum

    limits = dict(slot_limits or {})
    caps = sorted((int(limits.get(str(m), 0) or 0) for m in (models or [])), reverse=True)
    if not caps:
        return 0
    return caps[min(adaptive_quorum(len(caps)), len(caps)) - 1]


def per_slot_input_token_limits(
    models: Any,
    *,
    context_window: Optional[int] = None,
    output_reserve: int,
    tokenizer_margin: int,
    slots: Any = None,
) -> Dict[str, int]:
    """Per-slot calibrated input caps for a prompt fanned across mixed families.

    ``context_window=None`` (the default) resolves each slot's REAL window from
    Capability Evidence and scales the reserves to it, so a sub-1M slot gets a
    fit-sized pack instead of a prompt sized for a window it does not have.
    An explicit window stays honoured for callers that pin one deliberately.
    Frozen ``slots`` return caps keyed by stable slot ID; legacy model-only
    callers retain model keys. Equal models may have different account limits."""
    from ouroboros.reviewer_window import reviewer_context_window, reviewer_window_binding, window_scaled_reserves
    from ouroboros.tools.review_helpers import calibrated_input_token_limit

    limits: Dict[str, int] = {}
    models = list(models or [])
    rows = list(slots) if slots is not None else None
    if rows is not None and len(rows) != len(models):
        raise ValueError("Reviewer capacity rows must align with their frozen models")
    for index, model in enumerate(models):
        row = rows[index] if rows is not None else None
        binding = reviewer_window_binding(row) if row is not None else {}
        key = binding.get("model_role", "").removeprefix("reviewer:") if row is not None else str(model)
        if not key or (key in limits and row is not None):
            raise ValueError("Reviewer capacity requires unique stable slot IDs")
        window = (
            int(context_window)
            if context_window is not None
            else reviewer_context_window(str(model), **binding)
        )
        slot_output_reserve, slot_margin = window_scaled_reserves(
            window, output_reserve=output_reserve, tokenizer_margin=tokenizer_margin,
            model_id=str(model), binding=binding,
        )
        limits[key] = max(0, calibrated_input_token_limit(
            str(model),
            context_window=window,
            output_reserve=slot_output_reserve,
            tokenizer_margin=slot_margin,
        ))
    return limits


COUPLING_QUESTION_IDS = (
    "intent_alignment",
    "forgotten_touchpoints",
    "cross_surface_consistency",
    "regression_surface",
    "prompt_doc_sync",
    "architecture_fit",
    "cross_module_bugs",
    "implicit_contracts",
)


def build_coupling_part(
    *,
    coupling_checklist: str,
    required_sources_section: str,
    repository_index: str,
    history_block: str,
    layer: str = "body",
) -> str:
    """``## Part 2 — Coupling questions`` of the two-part brief: the whole-repository
    reviewer's role frame (the former scope reviewer's), the eight coupling
    questions, the required-source manifest the seat is OWED, the repository
    index it navigates with, and the coupling history of this subject. No
    preamble, calibration, anti-pattern guard, intent, diff or answer format
    here — Part 1 carries each exactly once, and ``## Answer format`` closes the
    brief. ``layer`` is the checklist layer (``review_body_fact.layer_for``).
    """
    questions = "\n".join(f"{i}. {item}" for i, item in enumerate(COUPLING_QUESTION_IDS, start=1))
    body_note = (
        "Apply the `Critical surface whitelist` in `docs/CHECKLISTS.md` for prose-vs-code\n"
        "mismatches." if layer == "body" else
        "The subject is not the Ouroboros body: judge prose-vs-code mismatches against the\n"
        "subject's own documents, which the index below names."
    )
    return f"""\
## Part 2 — Coupling questions

### Your role in this part

You are the whole-repository reviewer, and you REACH the repository with your own
read-only tools. Part 1 covers the change itself line by line; this part covers
what the change is COUPLED to: cross-module contracts, forgotten touchpoints,
hidden regressions, prompt/doc sync, architecture fit, and end-to-end intent
completeness. For each finding name the exact file, symbol, test, prompt, doc,
config, or sibling flow that proves it. Vague concerns without a concrete artifact
reference are advisory, not critical.

### The eight coupling questions

Answer EVERY question below with one entry in the "coupling" block of your answer;
the "item" field carries the identifier verbatim (case-sensitive, no substitutions).
A missing entry means the question was not reviewed.

{questions}

- For FAIL: concrete artifact (file/symbol/line/contract) + what is wrong + how to fix;
  one FAIL entry per distinct root cause, never a compressed summary.
- For PASS: 1–2 sentences stating WHY it passes, naming a concrete artifact or code
  path you checked. A bare "PASS" or a single-word reason is a reviewer failure.
- Do not return duplicate PASS entries, and never PASS a question that also has a
  FAIL — the concrete FAIL is authoritative.
- Severity: critical requires a concrete current artifact and a required change to
  this diff; otherwise advisory. Coupling affects only unchanged code outside the
  diff. {body_note}
- If an open obligation in the coupling history below already names an
  `obligation_id` for a root cause, reuse that exact id; never invent a new id for
  the same root cause.

{coupling_checklist}

{required_sources_section}

{repository_index}

{history_block}
"""


def build_plan_review_messages(
    system_prompt: str,
    user_content: str,
    user_stable_len: int = 0,
) -> List[Dict[str, Any]]:
    """Cache-friendly plan-review message pair.

    The whole system prompt (governance docs + reviewer contract) is byte-stable
    across plan reviews and carries the cache marker; the user content marks its
    evidence/plan boundary so stable repository evidence caches while the
    revised plan does not."""
    from ouroboros.tools.review_helpers import cached_prompt_blocks

    return [
        {"role": "system", "content": cached_prompt_blocks(system_prompt)},
        {
            "role": "user",
            "content": (
                cached_prompt_blocks(
                    user_content[:user_stable_len], user_content[user_stable_len:]
                )
                if 0 < user_stable_len <= len(user_content)
                else user_content
            ),
        },
    ]

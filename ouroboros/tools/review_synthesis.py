"""Reviewer window caps, the coupling part of the brief and plan-review messages.

The mixed-window input-cap helpers the commit panel and the plan-review engine
(``tools/plan_review.py`` + ``plan_spec``/``plan_packet``) size prompts with, the
two-part brief's ``## Part 2 — Coupling questions`` and the cache-friendly
plan-review message pair. Reviewer findings are never rewritten here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ouroboros.tools.scope_review_contract import COUPLING_QUESTION_IDS

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
            context_window
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


def build_coupling_part(
    *,
    coupling_checklist: str,
    required_sources_section: str,
    repository_index: str,
    history_block: str,
    layer: str = "body",
) -> str:
    """``## Part 2 — Coupling questions`` of the two-part brief: the whole-repository
    reviewer's role frame (the former scope reviewer's), the coupling
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

You are the whole-repository reviewer: REACH the subject through your available
tools, using the assigned checking copy if your seat can write. Part 1 covers the change; this part covers
what the change is COUPLED to: cross-module contracts, forgotten touchpoints,
hidden regressions, prompt/doc sync, architecture fit, and end-to-end intent
completeness. For each finding name the exact file, symbol, test, prompt, doc,
config, or sibling flow that proves it. Vague concerns without a concrete artifact
reference are advisory, not critical.

### The coupling questions

Answer EVERY question below with one entry in the "coupling" block of your answer;
the "item" field carries the identifier verbatim (case-sensitive, no substitutions).
A missing entry means the question was not reviewed.

{questions}

- For FAIL: file:line (or the exact artifact) + the invariant it violates + how to fix;
  one FAIL entry per distinct root cause, never a compressed summary.
- For PASS: one line — a pointer to what you checked, `see Part 1 <item>` for a defect
  you already report there (one root defect is one FAIL across both parts), or
  `N/A — <reason>` when the change does not reach the question.
- Do not return duplicate PASS entries, and never PASS a question that also has a
  FAIL — the concrete FAIL is authoritative.
- Severity: critical requires a concrete current artifact and a required change to
  this diff; otherwise advisory. {body_note}

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

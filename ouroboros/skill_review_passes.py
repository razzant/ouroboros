"""Skill-review pass runner (P5): one multi-model review pass normally, or — when an
over-budget skill is split into multiple budget-sized packs — a chunked pass per pack
whose per-model finding arrays are merged into one, so every byte is reviewed without
silent truncation and the existing quorum/aggregation produces one verdict.

Lives outside ``skill_review`` (module-size discipline). The prompt builder and the
multi-model review callable are INJECTED so this module never imports ``skill_review``
(no circular dependency); the agentic alternative is deliberately avoided — each chunk
reuses the SAME hardened review prompt, where skill content stays untrusted DATA.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Tuple

_SINGLE_CONTENT = (
    "Review the skill package whose manifest and payload are included above, using the "
    "Skill Review Checklist. Return ONLY the JSON array described in the output contract."
)


def run_skill_review_passes(
    ctx: Any,
    drive_root: Any,
    skill: Any,
    *,
    evidence: Dict[str, Any],
    file_packs: List[str],
    models: List[str],
    build_prompt: Callable[..., Tuple[str, Dict[str, Any]]],
    run_review: Callable[..., str],
) -> Tuple[str, Dict[str, Any], str, str]:
    """Return ``(prompt, advisory_evidence, result_json_text, infra_error)``. A non-empty
    ``infra_error`` means a pass failed and the caller should fail closed (pending). ``evidence``
    carries the prompt-building inputs: ``manifest_dump``, ``content_hash``, ``history``,
    ``review_rebuttal``, ``required_items``."""
    manifest_dump = evidence["manifest_dump"]
    content_hash = evidence["content_hash"]
    history = evidence["history"]
    review_rebuttal = evidence["review_rebuttal"]
    required_items = evidence["required_items"]
    if len(file_packs) == 1:
        prompt, advisory_evidence = build_prompt(
            ctx, drive_root, skill,
            manifest_dump=manifest_dump, content_hash=content_hash,
            file_pack=file_packs[0], history=history, review_rebuttal=review_rebuttal,
        )
        try:
            result_json_text = run_review(ctx, content=_SINGLE_CONTENT, prompt=prompt, models=models)
        except Exception as exc:  # pragma: no cover — transport failure path
            return prompt, advisory_evidence, "", f"{type(exc).__name__}: {exc}"
        return prompt, advisory_evidence, result_json_text, ""

    # Over-budget skill: review each chunk in a separate pass and merge the per-model
    # records. ``run_review`` returns a JSON OBJECT {"model_count", "results":[...]} (not a
    # bare array), so we union the chunks' ``results`` into ONE such object — the shape the
    # downstream ``parse_model_review_results`` expects (a bare list would crash it).
    prompt = ""
    advisory_evidence: Dict[str, Any] = {}
    merged_results: List[Any] = []
    total = len(file_packs)
    for idx, pack in enumerate(file_packs):
        chunk_prompt, adv = build_prompt(
            ctx, drive_root, skill,
            manifest_dump=manifest_dump, content_hash=content_hash,
            file_pack=pack, history=history, review_rebuttal=review_rebuttal,
        )
        prompt = chunk_prompt
        if idx == 0:
            advisory_evidence = adv
        content = (
            f"This skill is oversized, so its payload is split into {total} parts for "
            f"review; this is PART {idx + 1} of {total}. Review ONLY the files shown in "
            "this part against the Skill Review Checklist — other parts are reviewed "
            "separately, so do NOT flag files absent from this part as missing. Return "
            "ONLY the JSON array described in the output contract."
        )
        try:
            chunk_text = run_review(ctx, content=content, prompt=chunk_prompt, models=models)
            chunk_json = json.loads(chunk_text)
        except Exception as exc:  # pragma: no cover — transport failure path
            return prompt, advisory_evidence, "", f"chunk {idx + 1}/{total}: {type(exc).__name__}: {exc}"
        if isinstance(chunk_json, dict) and "error" in chunk_json:
            return prompt, advisory_evidence, "", f"chunk {idx + 1}/{total} service error: {chunk_json['error']}"
        if not isinstance(chunk_json, dict):
            return prompt, advisory_evidence, "", f"chunk {idx + 1}/{total}: non-object review response"
        # Fail CLOSED unless THIS chunk reached quorum of PARSEABLE reviewers — validated
        # with the SAME parser/required-item contract the single-pass gate uses, so a chunk
        # of malformed/non-JSON actor text cannot pass as "responsive" while the global
        # quorum is satisfied by other chunks (which would leave a portion of the oversized
        # skill under-reviewed — a trust-gate hole). adaptive_quorum matches the single-pass
        # gate (1 reviewer => degraded-but-allowed).
        from ouroboros.config import adaptive_quorum
        from ouroboros.triad_review import parse_model_review_results

        parsed = parse_model_review_results(chunk_json, required_items=required_items)
        required = adaptive_quorum(len(models))
        if len(parsed.responsive_models) < required:
            return (
                prompt, advisory_evidence, "",
                f"chunk {idx + 1}/{total}: only {len(parsed.responsive_models)}/{required} reviewers parsed",
            )
        merged_results.extend(chunk_json.get("results") or [])
    return (
        prompt,
        advisory_evidence,
        json.dumps({"results": merged_results}, ensure_ascii=False),
        "",
    )

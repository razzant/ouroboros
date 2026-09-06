"""Reviewer packet builders for ``plan_task`` — pure string builders, no I/O.

Companion of ``ouroboros.tools.plan_spec`` (schema / findings / aggregate /
evidence): the system prompt carries the findings-only stance, the domain-free
rubric, the blocking rule, the convergence rule (cycle ≥2), the checklist
section verbatim, and the governance pack (W3: BIBLE.md + ARCHITECTURE.md in full for a
self-modification plan, their navigation maps otherwise); the user content carries
TASK OBJECTIVE · SPEC · PLAN PROSE · EVIDENCE (+ OMISSIONS) · ROOT EXPLORATION
LOG · PRIOR CYCLES in that order. String bounds are ``PACKET_*_CHARS`` (plan_spec) via
``utils.truncate_review_artifact`` — visible marker, never silent. The
``PLAN_REVIEW_CONTROL_JSON`` control line is NOT emitted here (Phase C owns it).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ouroboros.tools.plan_spec import (
    PACKET_EXPLORATION_CHARS,
    bounded_json,
    PACKET_OBJECTIVE_CHARS,
    PACKET_PRIOR_CYCLES_CHARS,
    PACKET_PRIOR_FINDING_SUMMARY_CHARS,
    PACKET_PROSE_CHARS,
    PACKET_SPEC_CHARS,
    PLAN_FINDINGS_ARRAY_CONTRACT,
    bounded_text,
    spec_with_ids,
)


class PlanPacketError(ValueError):
    """Typed assembly failure: the packet cannot be built as the constitution requires
    (e.g. a constitutional review without BIBLE.md, D26). The caller must refuse."""


_RUBRIC = (
    "1. Success conditions — is every acceptance claim checkable as stated (by whom, against what)?",
    "2. Load-bearing decisions — are the decisions that would be expensive to reverse explicit, "
    "each with its rejected alternatives and why?",
    "3. Constraints and invariants — are the real constraints named (budget, deadline, safety, "
    "irreversibility, external commitments)?",
    "4. Deferrals — is an expensive-to-reverse decision hiding inside `deferred`?",
    "5. Evidence sufficiency — is the attached evidence enough to judge 1–4? If not, ask for "
    "exactly what is missing with a `need_evidence` finding naming its locator (the host attaches "
    "what its evidence policy allows on the next cycle and names every absence); do not invent a gap.",
)

_BLOCKING_RULE = (
    "A finding is `blocking` iff being wrong about it AFTER the work starts would invalidate work "
    "already done, violate a declared commitment, or make an acceptance claim unverifiable — and it "
    "MUST name `breaks`: the id of the spec element it breaks (goal for the intention as a whole, claim_N, invariant_N, decision_N, "
    "deferred_N). Everything else is a `note`. If missing evidence makes a claim STRUCTURALLY "
    "unverifiable, that is `blocking` with `breaks=<claim id>`, not `need_evidence`."
)

_CONVERGENCE_RULE = (
    "CONVERGENCE RULE (cycle ≥2): a reformulation of a prior finding is not a new finding; a new "
    "`blocking` finding must state why it was invisible in the previous cycle; a prior `blocking` "
    "finding that is still open must be re-emitted (same id, summary starting `still-open:`); a "
    "resolved one is simply not repeated. Spec ids may have SHIFTED between cycles (positional ids "
    "renumber when an element is dropped or reordered): re-target `breaks` against the CURRENT spec "
    "ids using the Spec delta (`renumbered: [{from, to, text}]`), never against the old ids."
)


def build_plan_review_system_prompt(
    *,
    checklist_section: str,
    constitutional: bool,
    bible_text: Optional[str],
    cycle_index: int,
    enforcement: str,
    bible_locator: str = "BIBLE.md",
    bible_nav_map: Optional[str] = None,
    architecture_text: Optional[str] = None,
    architecture_locator: str = "docs/ARCHITECTURE.md",
    architecture_nav_map: Optional[str] = None,
    governance_by_retrieval: bool = False,
) -> str:
    """Findings-only reviewer stance for ONE independent slot (no competing plan,
    no issue quota, no 'your own approach'). Governance pack per the owner-approved
    W3 wording: a self-modification plan (``constitutional``) carries BIBLE.md in
    full AND ARCHITECTURE.md inline; every other plan carries the BIBLE navigation
    map, the ARCHITECTURE navigation map and named on-demand pointers (the caller
    passes RESOLVABLE locators — the absolute system-repo paths). Both documents
    sit in this cache-stable system prompt. Raises ``PlanPacketError`` when
    ``constitutional`` and ``bible_text`` or ``architecture_text`` is empty:
    assembling a constitutional packet without its governance docs is a typed
    failure, never a silent omission (D26, W3). ``governance_by_retrieval=True`` is the
    RETRIEVING (agent_session) reviewer's form of the same pack: the executor's session
    prompt is compact by contract (`review_execution` — a delegated reviewer retrieves with
    its own tools, D12), so a constitutional pack names both documents as MANDATORY full
    reads at their resolvable locators instead of inlining ~500k chars; the documents must
    still exist (same typed failure)."""
    parts = [
        "You are one independent reviewer of an INTENTION — a plan spec — before the work starts. "
        "The work may be code, research, a deliverable, a computer-use flow, or an action in the "
        "world: judge the spec against ITS OWN domain, and use only the evidence in front of you.\n\n"
        "## The one question\n\n"
        "Is this spec sufficient to START the work safely? — not whether everything is specified.\n\n"
        "## Stance — findings only\n\n"
        "Report findings against the spec; do not write a competing plan or propose an alternative "
        "of your own, and do not fill a quota — an empty findings array with NO_FINDINGS is a "
        "legitimate result. Name the exact spec id, locator, or evidence line behind each finding.\n\n"
        "## Rubric (domain-free)\n\n" + "\n".join(_RUBRIC) + "\n",
    ]
    if constitutional:
        parts.append(
            "6. Governance (this plan touches Ouroboros's own body): does the spec contradict "
            "BIBLE.md or a frozen contract? Cite the principle.\n"
        )
    parts.append(f"\n## Blocking rule\n\n{_BLOCKING_RULE}\n")
    # The convergence rule is cycle-dependent, so it lives in the USER content's prior-cycles
    # section — the system prompt stays byte-stable across cycles and its cache block hits
    # (production-gate advisory, 39c3a195).
    parts.append(
        f"\nReview enforcement for this task: {str(enforcement or 'unknown').strip()} (host information; "
        "it does not change what counts as a finding).\n"
        f"\n## Output contract\n\n{PLAN_FINDINGS_ARRAY_CONTRACT}\n---\n"
    )
    if checklist_section:
        parts.append(f"## Plan Review Checklist (verbatim from docs/CHECKLISTS.md)\n\n{checklist_section}\n\n---\n")
    else:
        parts.append("⚠️ OMISSION NOTE: Plan Review Checklist section not supplied by the host.\n\n---\n")
    if constitutional:
        if not bible_text:
            raise PlanPacketError(
                "constitutional plan review requires BIBLE.md in the pack (D26); none supplied"
            )
        if not architecture_text:
            raise PlanPacketError(
                "constitutional plan review requires ARCHITECTURE.md in the pack (W3); none supplied"
            )
        if governance_by_retrieval:
            parts.append(
                "## Governance pack (this plan touches Ouroboros's own body) — MANDATORY FULL READS\n\n"
                "You are a retrieving reviewer: before judging, read BOTH documents in full with your "
                f"own tools — the constitution `{str(bible_locator or 'BIBLE.md').strip()}` and the "
                f"architecture reference `{str(architecture_locator or 'docs/ARCHITECTURE.md').strip()}`. "
                "An api reviewer receives them inline; you receive them by retrieval (same pack, "
                "same authority). Do not judge a self-modification plan without them.\n\n---\n"
            )
        else:
            parts.append(
                f"## BIBLE.md (constitution — this plan touches Ouroboros's own body)\n\n{bible_text}\n\n---\n"
            )
            parts.append(
                "## ARCHITECTURE.md (architecture and data flow — inline, in full, for a self-modification "
                f"plan)\n\n{architecture_text}\n\n---\n"
            )
    else:
        parts.append(
            "This plan does not touch the Ouroboros system repository; the constitutional pack and "
            "the architecture reference are named on-demand pointers — request either as "
            f"`need_evidence` with locator `{str(bible_locator or 'BIBLE.md').strip()}` or "
            f"`{str(architecture_locator or 'docs/ARCHITECTURE.md').strip()}` if you need it, or "
            "with `::lines=A-B` for one section; the host attaches what its evidence policy "
            "allows on the next cycle, names every absence, and cuts an over-bound source "
            "head-first, named `truncated_to_<N>`.\n"
        )
        if bible_nav_map:
            parts.append(f"\n## BIBLE navigation map (pointer, not a copy)\n\n{bible_nav_map}\n")
        else:
            parts.append("\n⚠️ OMISSION NOTE: BIBLE navigation map not supplied by the host.\n")
        if architecture_nav_map:
            parts.append(
                f"\n## ARCHITECTURE navigation map (pointer, not a copy)\n\n{architecture_nav_map}\n"
            )
        else:
            parts.append("\n⚠️ OMISSION NOTE: ARCHITECTURE navigation map not supplied by the host.\n")
    return "\n".join(parts)


def _json_block(payload: Any, limit: int) -> str:
    """Fenced JSON bounded STRUCTURALLY (whole items, disclosed counts + full-set hash) —
    never clipped mid-string (S-B07/S-B08)."""
    text, notes = bounded_json(payload, limit)
    block = f"```json\n{text}\n```"
    if notes:
        block += "\n⚠️ OMISSION NOTE (structural): " + "; ".join(notes)
    return block


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _prior_findings_projection(cycle: Mapping[str, Any]) -> list[dict]:
    """Compact, blocking-FIRST projection of one prior cycle's findings so a prior blocking
    finding can never fall behind the section bound (B-03)."""
    rows = [f for f in (cycle.get("findings") or []) if isinstance(f, Mapping)]
    rows.sort(key=lambda f: 0 if f.get("class") == "blocking" else 1)
    return [{
        "finding_id": f.get("finding_id") or f.get("id") or "", "class": f.get("class") or "",
        "breaks": f.get("breaks") or "", "locator": f.get("locator") or "",
        "summary": bounded_text(f.get("summary"), PACKET_PRIOR_FINDING_SUMMARY_CHARS),
    } for f in rows]


def _render_prior_cycles(prior_cycles: list[dict], dispositions: list[dict], spec_delta: Optional[dict]) -> str:
    lines = [
        "## PRIOR CYCLES (all reviewers' findings from the previous cycle, the agent's "
        "dispositions, and the spec delta)\n",
        f"### Convergence rule\n\n{_CONVERGENCE_RULE}\n",
    ]
    for cycle in prior_cycles:
        cycle = cycle if isinstance(cycle, Mapping) else {}
        lines.append(
            f"### Cycle {cycle.get('cycle_index', '?')} — aggregate {cycle.get('aggregate', '?')}: "
            "findings (blocking first; summaries bounded)\n\n"
            + _json_block(_prior_findings_projection(cycle), PACKET_PRIOR_CYCLES_CHARS) + "\n"
        )
    lines.append("### Agent dispositions\n\n" + _json_block(dispositions or [], PACKET_PRIOR_CYCLES_CHARS) + "\n")
    lines.append("### Spec delta\n\n" + _json_block(spec_delta or {}, PACKET_PRIOR_CYCLES_CHARS) + "\n")
    return "\n".join(lines)


def _render_evidence(manifest: Mapping[str, Any]) -> str:
    declared = list(manifest.get("declared") or [])
    requested = {str(x) for x in (manifest.get("reviewer_requested") or [])}
    requested |= {str(x) for x in (manifest.get("reviewer_requested_dropped") or [])}
    if not declared and not requested and not manifest.get("omissions"):
        return "(no evidence declared)\n"
    lines: list[str] = []
    if not declared:
        lines.append("(no evidence declared by the agent)\n")
    if requested:
        lines.append(
            "Locators marked `[reviewer-requested]` were asked for with `need_evidence` in an earlier "
            "cycle and are attached by the host (W3); the rest were declared by the agent.\n"
        )
    for item in manifest.get("attached") or []:
        tag = " [reviewer-requested]" if str(item.get("locator")) in requested else ""
        redacted = ", secrets_redacted=true" if item.get("secrets_redacted") else ""
        lines.append(
            f"### {item.get('locator')}{tag} (kind={item.get('kind')}, sha256={item.get('sha256')}, "
            f"bytes={item.get('bytes')}, attached_bytes={item.get('attached_bytes')}{redacted})\n"
            f"----- BEGIN {item.get('locator')} -----\n{item.get('text') or ''}\n"
            f"----- END {item.get('locator')} -----\n"
        )
    omissions = list(manifest.get("omissions") or [])
    lines.append("### OMISSIONS (every absence named)\n")
    if omissions:
        lines.append("| locator | reason |\n|---|---|")
        lines.extend(
            f"| {_cell(o.get('locator'))}{' [reviewer-requested]' if str(o.get('locator')) in requested else ''} "
            f"| {_cell(o.get('reason'))} |"
            for o in omissions
        )
        lines.append("")
    else:
        lines.append("none\n")
    return "\n".join(lines)


def plan_user_stable_len(user_content: str) -> int:
    """Byte offset where the cache-stable prefix ends (I-14).

    Everything up to the ROOT EXPLORATION LOG heading — objective, spec, plan prose, attached
    evidence — is byte-identical while the agent revises nothing; the exploration log (the
    agent's live tool-call tail) and the delta history below it change between cycles.
    Passing this to ``build_plan_review_messages`` is what makes a delta cycle re-send the
    evidence CACHED instead of paying for it again."""
    marker = "## ROOT EXPLORATION LOG"
    index = user_content.find(marker)
    return index if index > 0 else 0


def build_plan_review_user_content(
    *,
    objective: str,
    goal: str,
    plan_prose: str,
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    prior_cycles: list[dict],
    dispositions: list[dict],
    spec_delta: Optional[dict],
    root_exploration_log: Optional[str],
    cycle_index: int = 1,
) -> str:
    """Deterministic reviewer packet: TASK OBJECTIVE · SPEC · PLAN PROSE · EVIDENCE
    (+ OMISSIONS) [cache-stable prefix] · ROOT EXPLORATION LOG · PRIOR CYCLES (all reviewers' prior
    findings as a compact blocking-first projection + agent dispositions + spec
    delta on cycle ≥2). String bounds are the ``PACKET_*_CHARS`` constants via
    ``truncate_review_artifact`` (visible marker, never silent); the SPEC JSON
    block is bounded by ``PACKET_SPEC_CHARS`` (worst case ~1M chars otherwise)."""
    view = spec_with_ids(spec)
    if goal and not view.get("goal"):
        view["goal"] = goal
    sections = [
        "## TASK OBJECTIVE\n\n" + (bounded_text(objective, PACKET_OBJECTIVE_CHARS) or "(none declared)") + "\n",
        "## SPEC (ids are the only valid `breaks` targets)\n\n" + _json_block(view, PACKET_SPEC_CHARS) + "\n",
        "## PLAN PROSE\n\n" + (bounded_text(plan_prose, PACKET_PROSE_CHARS) or "(none)") + "\n",
        "## EVIDENCE\n\n" + _render_evidence(manifest),
        "## ROOT EXPLORATION LOG\n\n"
        + (bounded_text(root_exploration_log, PACKET_EXPLORATION_CHARS) or "(not provided by host)") + "\n",
    ]
    if prior_cycles:
        sections.append(_render_prior_cycles(prior_cycles, dispositions, spec_delta))
    elif int(cycle_index or 1) >= 2:
        sections.append(f"## PRIOR CYCLES\n\nCycle {int(cycle_index)}: no prior findings recorded by the host.\n")
    else:
        sections.append("## PRIOR CYCLES\n\nFirst cycle: no prior findings (blind, independent review).\n")
    return "\n".join(sections)

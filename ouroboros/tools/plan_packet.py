"""Reviewer packet builders for ``plan_task`` — pure string builders, no I/O.

Companion of ``ouroboros.tools.plan_spec`` (schema / findings / aggregate /
evidence): the system prompt carries the findings-only stance, the domain-free
rubric, the convergence rule (cycle ≥2), the checklist section verbatim — whose
height rule is what may block; a fallback blocking rule stands in only when the
section is missing — and the governance pack (W3: BIBLE.md + ARCHITECTURE.md in full for a
self-modification plan, their navigation maps otherwise); the user content carries
TASK OBJECTIVE · OWNER WORDS · SPEC · PLAN PROSE · EVIDENCE · OWN ROOM DIALOGUE ·
RELATED ROOM POINTERS · ROOT EXPLORATION LOG · PRIOR CYCLES in that order. The full
redacted dialogue uses task source custody and route-sized projections; only
exploration retains an independent display bound. The complete dispute is
measured with the operative inputs by the delivery layer. The
``PLAN_REVIEW_CONTROL_JSON`` control line is NOT emitted here (Phase C owns it).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from ouroboros.tools.plan_spec import (
    PACKET_EXPLORATION_CHARS,
    bounded_json,
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
    "what its evidence policy allows on the next cycle and names every absence). When what is "
    "missing is the AUTHOR's judgment rather than a document, ask the author: a `need_evidence` "
    "finding whose `breaks` names the spec id the question is about, no locator needed; Ouroboros "
    "answers it in its disposition or escalates it. Do not invent a gap.",
    "6. Subtraction — what could the spec drop (a claim, decision, invariant, deferral, path or "
    "guard) without losing the goal? Say it as a `note` naming the element id; removing is advice "
    "as legitimate as adding.",
)

_BLOCKING_RULE = (
    "A finding is `blocking` iff being wrong about it AFTER the work starts would invalidate work "
    "already done, violate a declared commitment, or make an acceptance claim unverifiable — and it "
    "MUST name `breaks`: the id of the spec element it breaks (goal for the intention as a whole, claim_N, invariant_N, decision_N, "
    "deferred_N). Everything else is a `note`. A claim you cannot check as written is a question to "
    "the author (`need_evidence` with the claim id in `breaks`) or a `note`, not a blocker."
)

_CONVERGENCE_RULE = (
    "CONVERGENCE RULE (cycle ≥2) — read the complete dispute, including earlier rejected "
    "alternatives and their rationale even if you are a new reviewer. Then adjudicate your OWN earlier findings first. They are the rows "
    "below whose finding_id starts with your panel seat (named at the end of this packet). Read the "
    "author's dispositions and the Spec delta as the author's argument, and for each `blocking` "
    "finding and `need_evidence` you raised decide: RESOLVED — the delta or the rationale answers "
    "it: do not repeat it; SUPERSEDED — the element it targeted was removed or replaced: do not "
    "repeat it; STILL OPEN — re-emit it (same id and class, summary starting `still-open:`) naming "
    "the residual the answer does not cover. `still-open` holds only while the goal is unchanged: "
    "when the host says the goal changed, judge the new intention afresh — an earlier finding "
    "survives only where it still breaks a CURRENT element. Then review what changed: a "
    "reformulation of an earlier finding is not a new finding, and a NEW `blocking` finding must "
    "state why it was invisible in the previous cycle. Spec ids may have SHIFTED between cycles "
    "(positional ids renumber when an element is dropped or reordered): re-target `breaks` against "
    "the CURRENT spec ids using the Spec delta (`renumbered: [{from, to, text}]`), never against "
    "the old ids."
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
    """Findings-only reviewer stance with optional brainstorming, not a competing
    plan or an issue quota. Governance pack per the owner-approved
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
        "Report findings against the spec; do not write a compulsory competing plan, and do not "
        "fill a quota — an empty findings array with NO_FINDINGS is a legitimate result. "
        "Name the exact spec id, locator, or evidence line behind each finding.\n\n"
        "Planning review is also an important brainstorming opportunity: challenge the premise "
        "and suggest a simpler or more general alternative when useful. Express this advice as "
        "optional `note` findings; Ouroboros decides whether to adopt it, without a required "
        "disposition. A preference, premise challenge, or repeated suggestion alone is never "
        "a blocker. Independently demonstrated failures still follow the rule below on what may block. "
        "A question the plan leaves open is returned to its author, not filed as advice: "
        "`need_evidence` with the spec id in `breaks` asks Ouroboros, who authors the plan and is "
        "the addressee of everything this review produces, to answer, escalate, or defer it openly "
        "in its disposition.\n\n"
        "## Rubric (domain-free)\n\n" + "\n".join(_RUBRIC) + "\n",
    ]
    if constitutional:
        parts.append(
            "7. Governance (this plan touches Ouroboros's own body): does the spec contradict "
            "BIBLE.md or a frozen contract? Cite the principle.\n"
        )
    if not checklist_section:
        # The checklist's "Height rule" is the one home of what may block; this
        # copy stands in only when the host could not supply it (named below).
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


def _json_block(payload: Any, limit: Optional[int] = None) -> str:
    """Complete fenced JSON, or a disclosed historical projection when bounded."""
    text, notes = (
        bounded_json(payload, limit) if limit is not None
        else (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), [])
    )
    block = f"```json\n{text}\n```"
    if notes:
        block += "\n⚠️ OMISSION NOTE (structural): " + "; ".join(notes)
    return block


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _render_prior_cycles(prior_cycles: list[dict], dispositions: list[dict], spec_delta: Optional[dict],
                         dispute_history: Optional[dict] = None) -> str:
    from ouroboros.tools.plan_review_artifacts import DISPUTE_HISTORY_RULE

    lines = [
        ("## PRIOR CYCLES (selected author view plus exact current index; original verdicts unchanged)\n"
         if (dispute_history or {}).get("authored_view") else
         "## PRIOR CYCLES (complete recorded dispute; original subjects and verdicts)\n"),
        f"### Convergence rule\n\n{_CONVERGENCE_RULE}\n",
        DISPUTE_HISTORY_RULE + "\n",
    ]
    lines.append("### Dispute history\n\n" + _json_block(
        dispute_history if dispute_history is not None else {"rounds": prior_cycles}) + "\n")
    if dispute_history is None:
        lines.append("### Agent dispositions\n\n" + _json_block(dispositions or []) + "\n")
    lines.append("### Spec delta\n\n" + _json_block(spec_delta or {}) + "\n")
    # ONE host fact from the delta the host already computed: `still-open` holds only while the
    # goal is unchanged. `unknown` when the previous frozen spec body was truncated (no delta).
    goal_changed = (spec_delta or {}).get("goal_changed")
    previous = (prior_cycles[-1] if isinstance(prior_cycles[-1], Mapping) else {}).get("cycle_index", "?") if prior_cycles else "?"
    lines.append(f"Goal changed since cycle {previous}: "
                 f"{'unknown' if not isinstance(goal_changed, bool) else 'yes' if goal_changed else 'no'}\n")
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
    dispute_history: Optional[dict] = None,
) -> str:
    """Keep operative inputs complete and attach the exact recorded room source.

    The delivery layer selects a newest source range only when the actual route
    cannot fit the complete dialogue beside governance and the operative plan.
    The full dispute travels through the same route measurement as the spec;
    it has no independent summary, count or character cap.
    The owner's words that caused the work (``manifest["owner_words"]``) follow
    the objective whole, in the cache-stable prefix; no key, no section.
    """
    from ouroboros.tools.plan_dialogue import render_dialogue

    view = spec_with_ids(spec)
    if goal and not view.get("goal"):
        view["goal"] = goal
    owner_words = str(manifest.get("owner_words") or "")
    sections = [
        "## TASK OBJECTIVE\n\n" + (objective or "(none declared)") + "\n",
        *([owner_words + "\n"] if owner_words.strip() else []),
        "## SPEC (ids are the only valid `breaks` targets)\n\n" + _json_block(view) + "\n",
        "## PLAN PROSE\n\n" + (plan_prose or "(none)") + "\n",
        "## EVIDENCE\n\n" + _render_evidence(manifest),
        render_dialogue(manifest),
        "## ROOT EXPLORATION LOG\n\n"
        + (bounded_text(root_exploration_log, PACKET_EXPLORATION_CHARS) or "(not provided by host)") + "\n",
    ]
    if prior_cycles or (dispute_history and (dispute_history.get("gaps") or dispute_history.get("current_author_plan"))):
        sections.append(_render_prior_cycles(prior_cycles, dispositions, spec_delta, dispute_history))
    elif int(cycle_index or 1) >= 2:
        sections.append(f"## PRIOR CYCLES\n\nCycle {int(cycle_index)}: no prior findings recorded by the host.\n")
    else:
        sections.append("## PRIOR CYCLES\n\nFirst cycle: no prior findings (blind, independent review).\n")
    return "\n".join(sections)

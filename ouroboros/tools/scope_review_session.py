"""Session delivery for the commit scope reviewer (phase 5.2/5.6/5.7).

The api pack and the session task are TWO deliveries of ONE review: same role,
checklist, output contract, calibration and intent context. This module owns
only what is session-specific — retrieval pointers instead of assembled
evidence, governance docs as navigation maps, and the forensic (never gating)
coverage manifest. Everything shared is imported from its existing owner, so
the two deliveries cannot drift apart.

Imports from ``scope_review`` are lazy and one-way at call time: this module is
itself imported lazily by ``run_scope_review``, so neither import can cycle at
module load.
"""

from __future__ import annotations

import pathlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ouroboros.tools.review_helpers import (
    CRITICAL_FINDING_CALIBRATION,
    build_goal_section,
    build_rebuttal_section,
    build_scope_section,
    load_checklist_section,
    load_governance_doc,
)
from ouroboros.tools.review_synthesis import build_scope_review_prompt

# What the coverage manifest calls this delivery (D-12). The value names the
# DELIVERY — the reviewer retrieved the surface itself — rather than the wire it
# arrived on; `agent_session` is the transport's own name in `ReviewRouteKind`,
# and reusing it made the manifest describe the route twice and the delivery not
# at all.
#
# D-12 also asked that readers stay compatible with the old spelling. Measured,
# there are NO readers: nothing in `ouroboros/` or `web/` reads this key. The
# manifest is a durable forensic row whose consumer is a person reading it, so
# the rename cannot break a caller, and a compatibility helper here would be
# machinery guarding nothing. Stated rather than built.
AGENTIC_RETRIEVAL_DELIVERY = "agentic_retrieval"


def governance_nav_maps(repo_dir: pathlib.Path, doc_paths: Tuple[str, ...]) -> str:
    """The canonical governance docs as navigation maps (5.7).

    Session delivery replaces the inlined canonical texts with the atlas idea
    reduced to a MAP: every ``##`` through ``####`` heading with its inclusive
    complete-subtree line range, read on demand by the session with its own tools.
    ``generate_doc_nav_map`` is the one existing mapper — no second repository
    scanner (§8 item 8)."""
    from ouroboros.context_layout import book_navigation, generate_doc_nav_map
    from ouroboros.reference_books import BOOK_ENTRYPOINTS, load_reference_book

    parts: list[str] = []
    for rel_path in doc_paths:
        book_id = next((key for key, path in BOOK_ENTRYPOINTS.items() if path == rel_path), None)
        if book_id is not None:
            try:
                book = load_reference_book(repo_dir, book_id)
            except (OSError, ValueError) as exc:
                parts.append(f"Reference book source unavailable: {rel_path}. {exc}. Required coverage is incomplete.")
                continue
            # One map per book, addressed to the chapter the section lives in.
            parts.append(book_navigation(book))
            continue
        else:
            text = load_governance_doc(repo_dir, rel_path, on_missing="placeholder")
        if str(text or "").strip():
            parts.append(generate_doc_nav_map(text, title=rel_path, rel_path=rel_path))
    return (
        "The governance docs are NOT inlined in session delivery. The maps below "
        "index them by line range; the paths are relative to the repository root — "
        "read the sections you need with your own tools.\n\n" + "\n\n---\n\n".join(parts)
    )


@dataclass(frozen=True)
class ScopeIntentContext:
    """The intent/history cluster BOTH scope deliveries take, as one value.

    These five always travel together — the api pack builder and the session task
    builder each need exactly this set — so they ride as one immutable parameter
    object (the ``ReviewAssignment`` pattern) instead of five parallel arguments
    that a caller can silently mis-order.
    """

    goal: str = ""
    scope: str = ""
    review_rebuttal: str = ""
    review_history: Optional[list] = None
    scope_review_history: Optional[list] = None


def build_scope_session_task(
    repo_dir: pathlib.Path,
    commit_message: str,
    intent: ScopeIntentContext,
    drive_root: Optional[pathlib.Path] = None,
    governance_repo_dir: Optional[pathlib.Path] = None,
    managed_subject: Optional[Any] = None,
    task_evidence_section: str = "",
    required_sources: Optional[list] = None,
    required_sources_ref: Optional[dict] = None,
) -> Tuple[str, Dict[str, Any]]:
    """The scope task in SESSION delivery, plus its forensic coverage manifest.

    Same role, checklist, contract, calibration and intent context as the api
    pack (5.3), with retrieval pointers instead of assembled evidence (5.2): no
    touched-file pack, no inlined diff, no atlas. The instruction text is the
    SAME builder the api pack uses, so the two deliveries cannot drift apart.

    The returned manifest is FORENSICS, not a gate (5.6/D16): it records that
    coverage is the session's own retrieval, and that the host did not observe
    which files it opened, as the non-blocking
    ``host_file_read_attestation`` fact. That is a provenance limit on what may
    be CLAIMED about coverage, not a finding that the review was incomplete
    (BIBLE P3, retrieving scope reviewers). The atlas's
    ``excluded_sensitive`` class is preserved on the host side (nothing
    sensitive is assembled at all here); what the harness reads with its own
    tools is not host-filtered, and the manifest says so instead of implying an
    attestation nobody performed."""
    from ouroboros.tools.scope_review import (
        _CANONICAL_CONTEXT_DOCS,
        _build_review_history_section,
        _build_scope_history_section,
    )

    goal, scope, review_rebuttal = intent.goal, intent.scope, intent.review_rebuttal
    review_history, scope_review_history = intent.review_history, intent.scope_review_history

    scope_checklist = load_checklist_section("Intent / Scope Review Checklist")
    if not str(scope_checklist or "").strip():
        raise RuntimeError(
            "Intent / Scope Review Checklist could not be loaded from docs/CHECKLISTS.md — "
            "scope review cannot run without its checklist (fail-closed)."
        )
    goal_section = build_goal_section(goal, scope, commit_message)
    scope_section = build_scope_section(scope)
    rebuttal_section = build_rebuttal_section(review_rebuttal)
    open_obligations = []
    if drive_root is not None:
        try:
            from ouroboros.review_state import load_state, make_repo_key
            state = load_state(pathlib.Path(drive_root))
            open_obligations = state.get_open_obligations(repo_key=make_repo_key(repo_dir))
        except Exception:
            open_obligations = []  # Non-fatal: best-effort hint
    history_section = _build_review_history_section(
        review_history or [], open_obligations=open_obligations,
    )
    scope_history_section = _build_scope_history_section(scope_review_history)
    nav_docs = governance_nav_maps(
        pathlib.Path(governance_repo_dir or repo_dir), _CANONICAL_CONTEXT_DOCS,
    )
    if managed_subject is not None and not managed_subject.fallback_full_diff:
        # Managed resolution (Δ4): the AUTHORITATIVE delta artifact is inlined —
        # a session that retrieved `git diff --cached` itself would re-review the
        # whole two-parent candidate instead of the resolver's work.
        touched_slot = (
            "(managed resolution — the reviewed path set is the resolution delta "
            "plus its conflict anchors: "
            + (", ".join(managed_subject.touched_paths()) or "(none)") + ")"
        )
        diff_slot = (
            "The inlined artifact below is the AUTHORITATIVE review subject for "
            "this managed-update resolution commit. Judge it as rendered; do NOT "
            "substitute your own `git diff --cached` — the staged diff is the "
            "whole two-parent merge candidate and re-renders already-released "
            "code. Read the touched files with your own tools as needed.\n\n"
            + managed_subject.render_prompt_diff()
        )
    else:
        touched_slot = (
            "(not inlined — session delivery: find the touched files yourself, with "
            "`git diff --cached --name-status` if your read-only mode lets you run "
            "commands and by reading the tree if it does not)"
        )
        diff_slot = (
            "(not inlined — session delivery: retrieve the staged change in this "
            "repository root yourself, with `git diff --cached` when your read-only "
            "mode permits commands and by reading the files when it does not)"
        )
        if managed_subject is not None:  # M0 missing: disclose, keep retrieval
            # Session delivery renders no diff body: the header's fallback line
            # must instruct retrieval, never claim a rendering below.
            diff_slot = f"{managed_subject.header(body_rendered=False)}\n\n{diff_slot}"
    task_text, _stable_len = build_scope_review_prompt(
        touched_slot,
        scope_checklist=scope_checklist,
        canonical_docs=nav_docs,
        intent_context=f"{scope_section}\n\n{goal_section}",
        history_block=f"{rebuttal_section}{history_section}{scope_history_section}",
        diff_text=diff_slot,
        repo_pack_placeholder=(
            "(no assembled repository pack in session delivery — the navigation "
            "maps above index the governance docs; retrieve everything else with "
            "your own tools)"
        ),
        critical_calibration=CRITICAL_FINDING_CALIBRATION,
        task_evidence_section=task_evidence_section,
    )
    manifest: Dict[str, Any] = {
        # D-12's ratified spelling. It is deliberately NOT `agent_session`: that
        # is the TRANSPORT's name (`ReviewRouteKind`), and reusing it here made
        # the manifest answer "how was this delivered" with the name of the wire
        # rather than of the delivery. What this field states is that the
        # reviewer RETRIEVED the surface itself. Nothing reads this key, so the
        # rename breaks no caller — see the constant's own note.
        "delivery": AGENTIC_RETRIEVAL_DELIVERY,
        "coverage": "agent_retrieval",
        # Non-blocking by construction (D16): forensics, never a gate, and the
        # fixed_overflow ladder does not apply to sessions (5.7).
        "host_file_read_attestation": "unobserved",
        "coverage_note": (
            "the reviewer session retrieves context with its own tools; the host "
            "assembled no pack and does not observe which files the session opened"
        ),
        "excluded_sensitive": {"policy": "preserved", "host_enforced": False},
        "nav_mapped_docs": list(_CANONICAL_CONTEXT_DOCS),
    }
    if required_sources is not None:
        manifest["native_required_sources"] = required_sources
        manifest["native_required_sources_ref"] = dict(required_sources_ref or {})
        task_text += ("\nThe required source surface is independent from your working window. "
                      "Read it completely in the order you choose; preserve your own conclusions "
                      "and exact source references across focus changes. Missing or unread sources "
                      "remain explicit gaps. Required source manifest: "
                      + json.dumps(required_sources_ref or {}, ensure_ascii=False))
    return task_text, manifest


# Retained only for the existing settings-notice import during coordinated
# integration. This value has no review-authority consumer here; the settings
# owner removes its obsolete acknowledgement notice in the same phase.
SESSION_WINDOW_FLOOR = 200_000


def session_scope_authority(
    critical_findings: list,
    advisory_findings: list,
    *,
    scope_model: str,
    window: Any,
    provenance: str,
    phrase: str,
    result_kwargs: Dict[str, Any],
) -> Tuple[list, list, Any]:
    """Window size does not change a retrieving review's independent findings.

    Native delivery reports exact required-source coverage separately; vendor
    sessions retain unobserved read provenance. The common review owner decides
    completeness/enforcement from those facts, including effective Cyber policy.
    A smaller or unknown window never rewrites a critical finding as advisory.
    """
    return critical_findings, advisory_findings, None

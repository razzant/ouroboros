"""The scope reviewer's brief — one first send for both retrieving deliveries.

Every scope row retrieves (owner decision 2026-09-17), so ONE brief is built here
and the transport only decides how the reviewer reads what the brief does not
carry: an ``api_chat`` row runs a bounded native inspection episode, an
``agent_session`` row a delegated read-only session. The brief carries

* the intent cluster — commit message, goal, scope, rebuttal, review history and
  the open obligations;
* the touched-path manifest and the change-relative required-source manifest
  (``scope_required_sources``) with its MINIMUM sentence;
* the compact repository index (``review_context_atlas.repository_index``):
  every tracked path's class, plus the structural facts and direct importers of
  the touched ones. No file bodies — the reviewer opens any path itself;
* the governance tiers from the ONE SSOT every review surface asks
  (``governance_context``): the rules this change activates inline, the map as
  navigation the reviewer reads on demand. Tier 1 rides the STABLE head, so
  nothing change-relative precedes the cache-marked prefix;
* the staged change itself — inlined when the whole first send lands under the
  row's own bound, and otherwise stored as ONE exact durable source the reviewer
  pages by address.

No size terminal lives here. The builder never refuses a diff for its size, and
a brief a window cannot hold at all is the native executor's typed row refusal
(``native_bound_below_first_send``), never a downgraded finding.

Imports from ``scope_review`` are lazy and one-way at call time: this module is
itself imported lazily by ``run_scope_review``, so neither import can cycle at
module load.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from ouroboros.tools.governance_context import RETRIEVING_DELIVERY, governance_context
from ouroboros.tools.review_helpers import (
    CRITICAL_FINDING_CALIBRATION,
    build_goal_section,
    build_rebuttal_section,
    build_scope_section,
    load_checklist_section,
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

# The checklist section this surface delivers; scope review fails closed without
# it (a reviewer with no checklist cannot answer the required matrix).
SCOPE_CHECKLIST_SECTION = "Intent / Scope Review Checklist"

# A delegated harness owns its own context selection and evidences no window, so
# the host states one conservative ceiling for what it INLINES into a session
# brief instead of inventing a window number for a vendor model. Above it the
# staged diff travels as an exact paged source, which the harness reads in
# ranges from the project.
SESSION_INLINE_DIFF_CEILING_CHARS = 400_000

# Chars per estimated token — the `utils.estimate_tokens` heuristic inverted, so
# the episode's char bound and the governance tiers' token budget meet on one
# scale (the same conversion the deep self-review row uses).
_CHARS_PER_ESTIMATED_TOKEN = 4

# Fallback first-send overhead when the inspection schemas cannot be projected
# for a measurement (no tools, no repository root): the JSON envelope and tool
# schemas that ride every native send, measured once on this build.
_NATIVE_SEND_OVERHEAD_ESTIMATE_CHARS = 20_000

# One durable source id per row: the store is write-once and content-addressed,
# so the same staged diff resolves to the same handle across retries.
DIFF_SOURCE_ID = "scope-staged-diff"


@dataclass(frozen=True)
class ScopeIntentContext:
    """The intent/history cluster the scope brief takes, as one value.

    These six always travel together, so they ride as one immutable parameter
    object (the ``ReviewAssignment`` pattern) instead of six parallel arguments
    that a caller can silently mis-order. ``owner_words`` is the host-attested
    section of the owner's words that caused the work (``owner_words_text``).
    """

    goal: str = ""
    scope: str = ""
    review_rebuttal: str = ""
    review_history: Optional[list] = None
    scope_review_history: Optional[list] = None
    owner_words: str = ""


@dataclass(frozen=True)
class ScopeBriefInputs:
    """Everything ONE scope row's brief is built from, as one value.

    The assembling half (``review_admission.prepare_scope_review``) fills this
    in and the builder reads nothing else: the review subject, the manifests it
    already computed, and the identity of the row the brief is sent to — which
    is what decides the row's bound, its governance share and whether a paged
    source is reachable through the artifact store or through the project.
    """

    commit_message: str = ""
    intent: ScopeIntentContext = field(default_factory=ScopeIntentContext)
    drive_root: Optional[pathlib.Path] = None
    governance_repo_dir: Optional[pathlib.Path] = None
    managed_subject: Optional[Any] = None
    task_evidence_section: str = ""
    required_sources: Optional[list] = None
    required_sources_ref: Optional[dict] = None
    touched_manifest: Optional[list] = None
    # Repo-relative paths of the reviewed change: the index's change-relative
    # rows and the governance tiers' change class are selected from them.
    touched_paths: Tuple[str, ...] = ()
    # The row: `delegated` is the transport, the rest is its route identity.
    delegated: bool = False
    scope_model: str = ""
    slot_id: str = ""
    session_profile: str = ""
    use_local: Optional[bool] = None
    # Where a paged source is stored so the row's OWN reader reaches it: the
    # task whose artifact store a native episode reads, and the data root that
    # store lives under (`policy["native_data_root"]`).
    task_id: str = ""
    source_root: str = ""


# ---------------------------------------------------------------------------
# The row's bounds: what the first send must land under, and the window share
# the governance tiers are taken against.
# ---------------------------------------------------------------------------


def scope_first_send_bound(brief: ScopeBriefInputs) -> int:
    """The row's SEND bound in chars — the one resolver the episode applies.

    ``review_native_transcript_bound`` derives it from the reviewer's window,
    the row's output reserve and the owner transcript ceiling; a route with no
    evidenced window resolves the ceiling, which is the one bound that holds for
    every route (a delegated harness included). The governance tiers take their
    inline share against this same number, converted to tokens, so the brief's
    rules and the brief's diff are sized by one fact.
    """
    from ouroboros.review_native_episode import review_native_transcript_bound
    from ouroboros.tools.scope_review import (
        _SCOPE_SIZING_FALLBACK, _scope_window, _window_scaled_reserves,
    )

    binding = {"model_role": f"reviewer:{brief.slot_id}" if brief.slot_id else "",
               "credential_profile_id": brief.session_profile or None,
               "use_local": brief.use_local}
    output_reserve, _margin = _window_scaled_reserves(
        _scope_window(brief.scope_model, **binding).sizing_window(_SCOPE_SIZING_FALLBACK)
    )
    return review_native_transcript_bound(
        brief.scope_model, output_reserve=output_reserve, **binding)


def _first_send_chars(repo_dir: pathlib.Path, brief: ScopeBriefInputs, brief_text: str) -> int:
    """The wire size of this row's first send, measured the way it is priced.

    A native row is measured by the same builder the paid ledger reserves
    against (``native_first_send_chars``: work order, message objects and the
    inspection schemas that ride every call), so the fit decision and the money
    admission cannot disagree. A route whose schemas cannot be projected here is
    measured as the brief plus this build's send overhead rather than left
    unmeasured.
    """
    if brief.delegated:
        return len(brief_text)
    from ouroboros.review_native_episode import native_first_send_chars
    from ouroboros.reviewer_slot_config import SCOPE_ROLE_HINT
    from ouroboros.tools.scope_review import SCOPE_RETRIEVING_OUTPUT_CONTRACT

    try:
        return native_first_send_chars(
            str(repo_dir), surface="scope_review", role_hint=SCOPE_ROLE_HINT,
            slot_id=brief.slot_id, session_task=brief_text,
            output_contract=SCOPE_RETRIEVING_OUTPUT_CONTRACT, task_id=brief.task_id)
    except (OSError, ValueError, RuntimeError):
        return len(brief_text) + _NATIVE_SEND_OVERHEAD_ESTIMATE_CHARS


# ---------------------------------------------------------------------------
# Diff delivery: inline while the first send holds it, otherwise one exact
# paged source. Never a refusal.
# ---------------------------------------------------------------------------


def _staged_diff(repo_dir: pathlib.Path, brief: ScopeBriefInputs) -> Tuple[str, str, str]:
    """``(header, body, unavailable_reason)`` of the reviewed change.

    A managed resolution states its own subject: the AUTHORITATIVE resolution
    delta against the mechanical merge baseline, never ``git diff --cached`` —
    the staged diff of such a commit is the whole two-parent candidate and
    re-renders already-released code. Without an M0 baseline there is no
    resolution delta to deliver, and the subject's own fallback header instructs
    retrieval instead. An ordinary commit uses the same hardened capture the
    triad's evidence uses; a capture the host cannot perform is DISCLOSED and
    the reviewer retrieves the change itself, exactly as it did before the diff
    was delivered at all.
    """
    subject = brief.managed_subject
    if subject is not None:
        if getattr(subject, "fallback_full_diff", False):
            return "", "", "managed_resolution_without_m0_baseline"
        return subject.header(), subject.diff, ""
    from ouroboros.tools.review_binary_context import StagedDiffUnavailable, capture_staged_diff

    try:
        return "", capture_staged_diff(pathlib.Path(repo_dir)), ""
    except StagedDiffUnavailable as exc:
        return "", "", f"staged_diff_capture_failed: {exc}"


def _store_review_source(
    repo_dir: pathlib.Path, brief: ScopeBriefInputs, raw: bytes, source_id: str,
) -> Dict[str, Any]:
    """Store exact subject bytes at an address the row's own reader reaches.

    Both deliveries write the same content-addressed handle into the task's
    existing actor-readable artifact store — the store a native episode's
    ``read_file(root="artifact_store", …)`` resolves under its data root. A
    delegated harness has no artifact store, only the project, so its brief
    additionally materializes the byte-exact copy inside the review's already
    git-ignored ``.review-drive`` view (the one mechanism the commit-review
    evidence view already uses). A store the host cannot write is not a
    refusal: the caller keeps the inline diff or the unavailable-preimage
    diagnostic, with the reason stated in either case.
    """
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.tools.scope_required_sources import source_text_identity

    root, task_id = str(brief.source_root or "").strip(), str(brief.task_id or "").strip()
    if not root or not task_id:
        return {}
    try:
        identity = source_text_identity(raw)
        ref = store_actor_source_bytes(
            root, task_id, category="context_checkpoints", source_id=source_id,
            data=raw, extension="txt")
    except (OSError, ValueError, TypeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    source: Dict[str, Any] = {**ref, "data_root": root}
    if brief.delegated:
        from ouroboros.review_evidence import materialize_commit_review_session_view

        view = materialize_commit_review_session_view(
            {"source_ref": ref, "task_id": task_id, "data_root": root}, repo_dir)
        if view.get("session_source_status") != "ready":
            return {"error": str(view.get("session_source_error") or "session_view_unavailable")}
        source["session_relative_path"] = view["session_relative_path"]
    source["required_row"] = {
        "root": "session_root" if brief.delegated else "artifact_store",
        "path": source.get("session_relative_path", source["path"]),
        **identity, "coverage_basis": "candidate_blob",
    }
    return source


def _paged_diff_pointer(brief: ScopeBriefInputs, body: str, source: Dict[str, Any]) -> str:
    """The address, size and digest of the paged subject, plus how to read it."""
    noun = "resolution delta" if brief.managed_subject is not None else "staged diff"
    if brief.delegated:
        address = (
            f"- project-relative path: `{source['session_relative_path']}` (inside this "
            "repository root, git-ignored)\n"
            "Read it with your own file tools, in ranges, until you have seen all of it."
        )
    else:
        address = (
            f"- root: `artifact_store`\n- path: `{source['path']}`\n"
            'Read it in ranges with `read_file(root="artifact_store", path="'
            + str(source["path"])
            + '", start_line=A, max_lines=N)`, and keep paging until you have seen all of it.'
        )
    return (
        f"The COMPLETE {noun} is stored as one exact source instead of being "
        "inlined here: the whole first send would not have fit this reviewer's "
        f"window. It is complete and untruncated — {len(body):,} chars, "
        f"sha256 `{source.get('sha256', 'unavailable')}`.\n"
        f"{address}\n"
        "It is the complete change evidence: every added and removed line of this "
        "commit is in it. Read it before you judge the change, and read the touched "
        "files themselves for the context around it."
    )


# The managed subject keeps its authority wording on both deliveries: a reviewer
# that re-derived `git diff --cached` itself would review the whole two-parent
# candidate instead of the resolver's work.
_MANAGED_SUBJECT_AUTHORITY = (
    "The {location} is the AUTHORITATIVE review subject for this managed-update "
    "resolution commit. Judge it as rendered; do NOT substitute your own "
    "`git diff --cached` — the staged diff is the whole two-parent merge candidate "
    "and re-renders already-released code. Read the touched files with your own "
    "tools as needed."
)


def _diff_slot(brief: ScopeBriefInputs, header: str, body: str, pointer: str) -> str:
    """The brief's diff section for one delivery decision."""
    if header:
        location = "paged artifact addressed below" if pointer else "inlined artifact below"
        return (f"{_MANAGED_SUBJECT_AUTHORITY.format(location=location)}\n\n"
                f"{header}\n\n{pointer or body}")
    if pointer:
        return pointer
    if not body.strip():
        return ("(the staged index of this repository root carries no change: nothing "
                "is staged, so there is no diff to render)")
    return body


def _retrieval_slot(brief: ScopeBriefInputs, reason: str) -> str:
    """The diff section when the host delivered no diff, and why.

    The reviewer retrieves the change itself — which every scope reviewer can
    do — and the reason the host could not hand it over is stated instead of a
    silent absence (BIBLE P1).
    """
    subject = brief.managed_subject
    lead = (
        "(not inlined — retrieve the staged change in this repository root yourself, "
        "with `git diff --cached` when your read-only mode permits commands and by "
        f"reading the files when it does not. Host disclosure: {reason})"
    )
    if subject is not None:
        # A body follows no header here: the fallback text must instruct
        # retrieval, never claim a rendering below it.
        return f"{subject.header(body_rendered=False)}\n\n{lead}"
    return lead


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------


def _repository_index(repo_dir: pathlib.Path, touched_paths: Sequence[str]) -> Tuple[str, dict]:
    """The compact index, or a disclosed absence.

    The index is orientation, not authority: a tree it cannot walk (no git
    metadata, an unreadable path) leaves the reviewer its own tools and a stated
    reason rather than failing the review.
    """
    from ouroboros.tools.review_context_atlas import repository_index

    try:
        return repository_index(pathlib.Path(repo_dir), touched_paths=list(touched_paths))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return (
            "## Repository index\n\nThe deterministic repository index could not be "
            f"built for this tree ({reason}). List and read the tree with your own "
            "tools; nothing about it is claimed here.",
            {"strategy": "repository_index", "status": "unavailable", "reason": reason},
        )


def _required_sources_tail(rows: Optional[list], ref: dict) -> str:
    """The required-source manifest and its identity, or ``""``."""
    if rows is None:
        return ""
    from ouroboros.tools.scope_required_sources import render_required_sources

    return (
        "\n\n" + render_required_sources(rows)
        + "\nThe required source surface is independent from your working window. "
        "Read it completely in the order you choose; preserve your own conclusions "
        "and exact source references across focus changes. Missing or unread sources "
        "remain explicit gaps. Required source manifest: "
        + json.dumps(ref, ensure_ascii=False)
    )


def _touched_slot(brief: ScopeBriefInputs) -> str:
    """The touched-path manifest, with what it is and is not."""
    from ouroboros.tools.scope_required_sources import render_touched_manifest

    if brief.managed_subject is not None:
        lead = (
            "(managed resolution — the reviewed path set is the resolution delta "
            "plus its conflict anchors: "
            + (", ".join(brief.managed_subject.touched_paths()) or "(none)") + ")"
        )
    else:
        lead = (
            "(the bodies are NOT inlined: the manifest below names every touched path "
            "with its disposition and its size in the candidate tree, the staged diff "
            "is the complete change evidence, and you open any file itself with your "
            "own tools)"
        )
    manifest = render_touched_manifest(brief.touched_manifest or [])
    return f"{lead}\n\n{manifest}" if manifest else lead


def build_scope_session_task(
    repo_dir: pathlib.Path, brief: ScopeBriefInputs,
) -> Tuple[str, Dict[str, Any]]:
    """The scope brief for ONE row, plus the manifest of what it delivered.

    Same role, checklist, output contract, calibration and intent context for
    both retrieving transports; the transport decides only the reachable address
    of a paged source. The returned manifest is the pre-run disclosure record:
    which governance documents were inlined and which were left as navigation,
    what the repository index accounted for, how the diff was delivered and how
    large each section is, and the read provenance this delivery is EXPECTED to
    produce (a native episode's receipts are host-observed, a delegated
    session's are parsed from its harness journal). The post-run coverage facts
    stay authoritative over this expectation.
    """
    from ouroboros.tools.scope_review import (
        _build_review_history_section, _build_scope_history_section,
    )

    scope_checklist = load_checklist_section(SCOPE_CHECKLIST_SECTION)
    if not str(scope_checklist or "").strip():
        raise RuntimeError(
            f"{SCOPE_CHECKLIST_SECTION} could not be loaded from docs/CHECKLISTS.md — "
            "scope review cannot run without its checklist (fail-closed)."
        )
    repo_dir = pathlib.Path(repo_dir)
    intent = brief.intent
    goal_section = build_goal_section(intent.goal, intent.scope, brief.commit_message, intent.owner_words)
    scope_section = build_scope_section(intent.scope)
    rebuttal_section = build_rebuttal_section(intent.review_rebuttal)
    open_obligations = []
    if brief.drive_root is not None:
        try:
            from ouroboros.review_state import load_state, make_repo_key

            state = load_state(pathlib.Path(brief.drive_root))
            open_obligations = state.get_open_obligations(repo_key=make_repo_key(repo_dir))
        except Exception:
            open_obligations = []  # Non-fatal: the brief states the history it has
    history_section = _build_review_history_section(
        intent.review_history or [], open_obligations=open_obligations)
    # The triad section is handed over so a convergence rule it already carries
    # is not repeated by the scope-only chain's own append.
    scope_history_section = _build_scope_history_section(
        intent.scope_review_history, history_section)

    bound = scope_first_send_bound(brief)
    governance = governance_context(
        pathlib.Path(brief.governance_repo_dir or repo_dir),
        surface="scope",
        touched_paths=brief.touched_paths,
        usable_window_tokens=bound // _CHARS_PER_ESTIMATED_TOKEN,
        delivery=RETRIEVING_DELIVERY,
        checklist_section_text=scope_checklist,
    )
    from ouroboros.tools.scope_required_sources import required_sources_ref, with_inline_sources

    tree_sha = str((brief.required_sources_ref or {}).get("staged_tree_sha") or "")
    required_rows = (with_inline_sources(brief.required_sources, governance.inline_whole_documents)
                     if brief.required_sources is not None else None)
    preimage_sources = []
    for row in required_rows or ():
        if row.get("coverage_basis") != "preimage_unavailable":
            continue
        try:
            raw = subprocess.run(
                ["git", "show", str(row["preimage"])], cwd=repo_dir, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            row["reason"] = f"preimage not delivered: {type(exc).__name__} reading {row['preimage']}"
            continue
        source = _store_review_source(
            repo_dir, brief, raw, "scope-preimage-" + hashlib.sha256(raw).hexdigest()[:12])
        if not source.get("path"):
            row["reason"] = "preimage not delivered: " + str(source.get("error") or "no_source_root")
            continue
        row.update(preimage_of=row["path"], disposition="deleted_preimage", **source["required_row"])
        row.pop("reason", None)
        preimage_sources.append(source)
    required_ref = required_sources_ref(required_rows or [], staged_tree_sha=tree_sha)
    index_text, index_manifest = _repository_index(repo_dir, brief.touched_paths)
    wider = "\n\n".join(part for part in (
        index_text,
        ("### Governance rules this change activates\n\n" + governance.selected_inline
         if governance.selected_inline.strip() else ""),
        governance.navigation,
    ) if str(part or "").strip())
    touched_slot = _touched_slot(brief)
    required_tail = _required_sources_tail(required_rows, required_ref)

    def _assemble(diff_slot: str) -> str:
        task_text, _stable_len = build_scope_review_prompt(
            touched_slot,
            scope_checklist=scope_checklist,
            canonical_docs=governance.stable_inline,
            intent_context=f"{scope_section}\n\n{goal_section}",
            history_block=f"{rebuttal_section}{history_section}{scope_history_section}",
            diff_text=diff_slot,
            repo_pack_placeholder=wider,
            critical_calibration=CRITICAL_FINDING_CALIBRATION,
            task_evidence_section=brief.task_evidence_section,
        )
        return task_text + required_tail

    header, body, unavailable = _staged_diff(repo_dir, brief)
    delivery: Dict[str, Any] = {"diff_chars": len(body)}
    if unavailable:
        diff_slot = _retrieval_slot(brief, unavailable)
        delivery.update(diff_delivery="retrieved_by_reviewer", diff_reason=unavailable)
    else:
        diff_slot = _diff_slot(brief, header, body, "")
        measured = _first_send_chars(repo_dir, brief, _assemble(diff_slot))
        # The first-send size the inlined diff must stay under: a native episode
        # must land BEFORE its landing notice (or the host would open the review
        # by announcing that the window is nearly full); a delegated session has
        # no host-enforced transcript and takes the conservative inline ceiling.
        from ouroboros.review_native_episode import native_landing_at

        ceiling = SESSION_INLINE_DIFF_CEILING_CHARS if brief.delegated else native_landing_at(bound)
        delivery.update(diff_delivery="inline", first_send_chars=measured,
                        first_send_ceiling=ceiling)
        if measured >= ceiling:
            source = _store_review_source(repo_dir, brief, body.encode("utf-8"), DIFF_SOURCE_ID)
            if source.get("path"):
                diff_row = {**source["required_row"], "disposition": "review_subject"}
                if tree_sha:
                    diff_row["candidate_tree"] = tree_sha
                source["required_row"] = diff_row
                required_rows = [*(required_rows or []), diff_row]
                required_ref = required_sources_ref(required_rows, staged_tree_sha=tree_sha)
                required_tail = _required_sources_tail(required_rows, required_ref)
                diff_slot = _diff_slot(
                    brief, header, body, _paged_diff_pointer(brief, body, source))
                delivery.update(diff_delivery="paged", diff_source=source)
            else:
                # An unreachable store is disclosed, never a refusal: the diff
                # stays inline and the episode's own bound decides the send.
                delivery.update(diff_paging="unavailable",
                                diff_paging_reason=str(source.get("error") or "no_source_root"))
    task_text = _assemble(diff_slot)
    if delivery.get("diff_delivery") == "paged":
        delivery["first_send_chars"] = _first_send_chars(repo_dir, brief, task_text)

    manifest: Dict[str, Any] = {
        # D-12's ratified spelling. It is deliberately NOT `agent_session`: that
        # is the TRANSPORT's name (`ReviewRouteKind`), and reusing it here made
        # the manifest answer "how was this delivered" with the name of the wire
        # rather than of the delivery. What this field states is that the
        # reviewer RETRIEVED the surface itself.
        "delivery": AGENTIC_RETRIEVAL_DELIVERY,
        "coverage": "agent_retrieval",
        # The PRE-RUN truth: which provenance this delivery's read receipts will
        # carry. A native episode's reads are executed by the host; a delegated
        # session's are parsed from its harness journal. Neither is a claim that
        # any source WAS read — that is the post-run coverage fact.
        "read_provenance_expected": "harness_observed" if brief.delegated else "host_observed",
        "coverage_note": (
            "the delegated harness retrieves context with its own tools; the host "
            "parses its run journal into harness-observed read receipts and folds "
            "them over the required-source manifest"
            if brief.delegated else
            "the reviewer retrieves context through host-executed read-only tools; "
            "its receipts are folded over the required-source manifest"
        ),
        "excluded_sensitive": {"policy": "preserved", "host_enforced": False},
        "repository_index": index_manifest,
        "governance_manifest": governance.manifest,
        "governance_tokens_estimate": governance.tokens_estimate,
        "preimage_sources": preimage_sources,
        "native_data_root": (str(brief.source_root or "") if preimage_sources
                             or delivery.get("diff_delivery") == "paged" else ""),
        "first_send_bound": bound,
        "brief_chars": len(task_text),
        "brief_sections": {
            "governance_stable_inline": len(governance.stable_inline),
            "governance_selected_inline": len(governance.selected_inline),
            "governance_navigation": len(governance.navigation),
            "scope_checklist": len(scope_checklist),
            "repository_index": len(index_text),
            "intent": len(scope_section) + len(goal_section),
            "history": len(rebuttal_section) + len(history_section) + len(scope_history_section),
            "touched_manifest": len(touched_slot),
            "required_sources": len(required_tail),
            "task_evidence": len(brief.task_evidence_section),
            "diff_slot": len(diff_slot),
        },
        **delivery,
    }
    if required_rows is not None:
        manifest["native_required_sources"] = required_rows
        manifest["native_required_sources_ref"] = required_ref
    return task_text, manifest

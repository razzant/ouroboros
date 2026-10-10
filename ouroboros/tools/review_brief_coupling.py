"""The retrieving seat's brief — one brief, two parts, one first send.

A seat that RETRIEVES (a pool seat, a delegated session, a native inspection
episode) reads the repository with its own tools, so the commit gate asks it
BOTH questions in one brief: ``## Part 1 — The change`` (the same preamble,
calibration, anti-pattern guard, checklist, governance tiers, intent, history
and staged change every packet seat is given) and ``## Part 2 — Coupling
questions`` (the whole-repository reviewer's role frame, the eight coupling
questions, the required-source manifest the seat is OWED, the repository index
it navigates with, and the coupling history of this subject), closed by
``## Answer format`` — contract B (``triad_review.REVIEW_TWO_PART_OBJECT_CONTRACT``).
A ``coupling_only`` seat receives the same brief and is told to answer Part 2
alone. The transport decides only how the seat reads what the brief does not
carry: an ``api_chat`` row runs a bounded native inspection episode, an
``agent_session`` row a delegated read-only session.

The brief carries

* the intent cluster — commit message, goal, scope, rebuttal, review history and
  the open obligations;
* the touched-path manifest and the change-relative required-source manifest
  (``scope_required_sources``) with its MINIMUM sentence;
* the compact repository index (``review_context_atlas.repository_index``):
  every tracked path's class, plus the structural facts and direct importers of
  the touched ones. No file bodies — the reviewer opens any path itself;
* the governance tiers from the ONE SSOT every review surface asks
  (``governance_context``): the rules this change activates inline, the map as
  navigation the reviewer reads on demand;
* the staged change itself — inlined when the whole first send lands under the
  seat's own bound, and otherwise stored as ONE exact durable source the reviewer
  pages by address.

No size terminal lives here. The builder never refuses a diff for its size, and
a brief a window cannot hold at all is the native executor's typed row refusal
(``native_bound_below_first_send``), never a downgraded finding.
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
    _ANTI_THRASHING_RULE_VERDICT,
    _CONVERGENCE_RULE_TEXT,
    _HISTORY_VERIFICATION_ONLY_RULE,
    build_goal_section,
    build_rebuttal_section,
    build_scope_section,
    format_review_history_entry,
    load_checklist_section,
    review_history_with_obligations,
)
from ouroboros.tools.review_synthesis import build_coupling_part

# What the coverage manifest calls this delivery (D-12). The value names the
# DELIVERY — the reviewer retrieved the surface itself — rather than the wire it
# arrived on; `agent_session` is the transport's own name in `ReviewRouteKind`.
AGENTIC_RETRIEVAL_DELIVERY = "agentic_retrieval"

# The checklist section Part 2 delivers; the brief fails closed without it (a
# reviewer with no checklist cannot answer the required coupling matrix).
COUPLING_CHECKLIST_SECTION = "Coupling questions"

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
DIFF_SOURCE_ID = "review-staged-diff"

PART_CHANGE, PART_COUPLING = "change", "coupling"


@dataclass(frozen=True)
class BriefIntent:
    """The intent/history cluster the brief takes, as one value.

    These always travel together, so they ride as one immutable parameter
    object instead of parallel arguments a caller can silently mis-order.
    ``owner_words`` is the host-attested section of the owner's words that
    caused the work (``owner_words_text``); ``coupling_history`` is this
    subject's prior ``per_question.coupling`` rounds.
    """

    goal: str = ""
    scope: str = ""
    review_rebuttal: str = ""
    review_history: Optional[list] = None
    coupling_history: Optional[list] = None
    owner_words: str = ""


@dataclass(frozen=True)
class BriefInputs:
    """Everything ONE retrieving seat's brief is built from, as one value.

    The assembling half (``review_admission``) fills this in and the builder
    reads nothing else: the review subject, the manifests it already computed,
    and the identity of the seat the brief is sent to — which decides the seat's
    bound, its governance share and whether a paged source is reachable through
    the artifact store or through the project.
    """

    commit_message: str = ""
    intent: BriefIntent = field(default_factory=BriefIntent)
    drive_root: Optional[pathlib.Path] = None
    governance_repo_dir: Optional[pathlib.Path] = None
    managed_subject: Optional[Any] = None
    # The reviewed change as the frozen subject renders it (``FrozenSubject
    # .diff_text``); ``None`` captures the reading root's staged index.
    diff_text: Optional[str] = None
    task_evidence_section: str = ""
    required_sources: Optional[list] = None
    required_sources_ref: Optional[dict] = None
    touched_manifest: Optional[list] = None
    # Repo-relative paths of the reviewed change: the index's change-relative
    # rows and the governance tiers' change class are selected from them.
    touched_paths: Tuple[str, ...] = ()
    # The checklist layer of the subject (review_body_fact.layer_for): ``body``
    # runs the body's governance tiers, ``core`` indexes the subject's own
    # documents under ``repo_dir`` and inlines no Ouroboros governance.
    layer: str = "body"
    # The Part-1 checklist section (``review._load_checklist_section(layer)``);
    # empty loads the layer's section itself.
    checklist_section: str = ""
    # The parts this seat is asked: ``("change", "coupling")`` or ``("coupling",)``.
    parts: Tuple[str, ...] = (PART_CHANGE, PART_COUPLING)
    # The seat: `delegated` is the transport, the rest is its route identity.
    delegated: bool = False
    model: str = ""
    slot_id: str = ""
    session_profile: str = ""
    use_local: Optional[bool] = None
    # Where a paged source is stored so the seat's OWN reader reaches it: the
    # task whose artifact store a native episode reads, and the data root that
    # store lives under (`policy["native_data_root"]`).
    task_id: str = ""
    source_root: str = ""


# ---------------------------------------------------------------------------
# The seat's bounds: what the first send must land under, and the window share
# the governance tiers are taken against.
# ---------------------------------------------------------------------------


def first_send_bound(brief: BriefInputs) -> int:
    """The seat's SEND bound in chars — the one resolver the episode applies.

    ``review_native_transcript_bound`` derives it from the reviewer's window,
    the seat's output reserve and the owner transcript ceiling; a route with no
    evidenced window resolves the ceiling, which is the one bound that holds for
    every route (a delegated harness included). The governance tiers take their
    inline share against this same number, converted to tokens, so the brief's
    rules and the brief's diff are sized by one fact.
    """
    from ouroboros.review_native_episode import review_native_transcript_bound
    from ouroboros.reviewer_window import window_scaled_reserves
    from ouroboros.tools.review_multi_model import _review_output_budget
    from ouroboros.tools.scope_window import SCOPE_SIZING_FALLBACK_WINDOW, scope_window

    binding = {"model_role": f"reviewer:{brief.slot_id}" if brief.slot_id else "",
               "credential_profile_id": brief.session_profile or None,
               "use_local": brief.use_local}
    window = scope_window(brief.model, **binding).sizing_window(SCOPE_SIZING_FALLBACK_WINDOW)
    output_reserve, _margin = window_scaled_reserves(
        window, output_reserve=_review_output_budget(), tokenizer_margin=50_000)
    return review_native_transcript_bound(brief.model, output_reserve=output_reserve, **binding)


def _first_send_chars(repo_dir: pathlib.Path, brief: BriefInputs, brief_text: str) -> int:
    """The wire size of this seat's first send, measured the way it is priced.

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
    from ouroboros.tools.review_multi_model import TRIAD_ROLE_HINT
    from ouroboros.triad_review import REVIEW_TWO_PART_OBJECT_CONTRACT

    try:
        return native_first_send_chars(
            str(repo_dir), surface="multi_model_review", role_hint=TRIAD_ROLE_HINT,
            slot_id=brief.slot_id, session_task=brief_text,
            output_contract=REVIEW_TWO_PART_OBJECT_CONTRACT, task_id=brief.task_id)
    except (OSError, ValueError, RuntimeError):
        return len(brief_text) + _NATIVE_SEND_OVERHEAD_ESTIMATE_CHARS


# ---------------------------------------------------------------------------
# Diff delivery: inline while the first send holds it, otherwise one exact
# paged source. Never a refusal.
# ---------------------------------------------------------------------------


def _staged_diff(repo_dir: pathlib.Path, brief: BriefInputs) -> Tuple[str, str, str]:
    """``(header, body, unavailable_reason)`` of the reviewed change.

    A managed resolution states its own subject: the AUTHORITATIVE resolution
    delta against the mechanical merge baseline, never ``git diff --cached`` —
    the staged diff of such a commit is the whole two-parent candidate and
    re-renders already-released code. Without an M0 baseline there is no
    resolution delta to deliver, and the subject's own fallback header instructs
    retrieval instead. A frozen subject hands its rendering in (``diff_text``);
    an ordinary commit uses the same hardened capture the packet's evidence
    uses; a capture the host cannot perform is DISCLOSED and the reviewer
    retrieves the change itself.
    """
    subject = brief.managed_subject
    if subject is not None:
        if getattr(subject, "fallback_full_diff", False):
            return "", "", "managed_resolution_without_m0_baseline"
        return subject.header(), subject.diff, ""
    if brief.diff_text is not None:
        return "", str(brief.diff_text), ""
    from ouroboros.tools.review_binary_context import StagedDiffUnavailable, capture_staged_diff

    try:
        return "", capture_staged_diff(pathlib.Path(repo_dir)), ""
    except StagedDiffUnavailable as exc:
        return "", "", f"staged_diff_capture_failed: {exc}"


def _store_review_source(
    repo_dir: pathlib.Path, brief: BriefInputs, raw: bytes, source_id: str,
) -> Dict[str, Any]:
    """Store exact subject bytes at an address the seat's own reader reaches.

    Both deliveries write the same content-addressed handle into the task's
    existing actor-readable artifact store — the store a native episode's
    ``read_file(root="artifact_store", …)`` resolves under its data root. A
    delegated harness has no artifact store, only the project, so its brief
    additionally materializes the byte-exact copy inside the review's already
    git-ignored ``.review-drive`` view. A store the host cannot write is not a
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


def _paged_diff_pointer(brief: BriefInputs, body: str, source: Dict[str, Any]) -> str:
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


def _diff_slot(brief: BriefInputs, header: str, body: str, pointer: str) -> str:
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


def _retrieval_slot(brief: BriefInputs, reason: str) -> str:
    """The diff section when the host delivered no diff, and why.

    The reviewer retrieves the change itself — which every retrieving seat can
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
# Part 2's own sections
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


def _required_sources_section(rows: Optional[list], ref: dict, *, layer: str = "body") -> str:
    """The required-source manifest and its identity, or ``""``."""
    if rows is None:
        return ""
    from ouroboros.tools.scope_required_sources import render_required_sources

    return (
        render_required_sources(rows, layer=layer)
        + "\nThe required source surface is independent from your working window. "
        "Read it completely in the order you choose; preserve your own conclusions "
        "and exact source references across focus changes. Missing or unread sources "
        "remain explicit gaps. Required source manifest: "
        + json.dumps(ref, ensure_ascii=False)
    )


def _touched_slot(brief: BriefInputs) -> str:
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


def build_coupling_history_section(coupling_history: Optional[list], history_section: str = "") -> str:
    """Prior ``coupling`` rounds of this subject as a brief section.

    A coupling-only retry chain leaves the Part-1 history empty, so the shared
    anti-thrashing block never reaches its convergence rule; from the third
    coupling round on it is appended here instead. ``history_section`` is the
    already-rendered Part-1 history: when it carries the rule, this section does
    not repeat it.
    """
    if not coupling_history:
        return ""
    rounds = []
    for i, entry in enumerate(coupling_history, 1):
        verdict = str(entry.get("verdict") or entry.get("status") or "").strip()
        label = (
            "BLOCKED" if entry.get("blocked")
            else verdict.upper() if verdict and verdict.lower() not in ("responded", "pass")
            else "PASSED"
        )
        parts = [f"Round {i}: {label}"]
        critical_findings = list(entry.get("critical_findings") or [])
        advisory_findings = list(entry.get("advisory_findings") or [])
        if critical_findings:
            parts.append("Critical findings:")
            for finding in critical_findings:
                parts.append(f"- {format_review_history_entry(finding, default_severity='critical')}")
        if advisory_findings:
            parts.append("Advisory findings:")
            for finding in advisory_findings:
                parts.append(f"- {format_review_history_entry(finding)}")
        if not critical_findings and not advisory_findings:
            parts.append(str(entry.get("summary") or "(no summary)"))
        rounds.append("\n".join(parts))
    section = (
        "### Prior coupling rounds (your previous Part-2 answers for this subject)\n\n"
        + "\n\n---\n".join(rounds)
        + "\n\nAddress any previously raised issues. If the same issue persists, "
        "mark it FAIL again with a reference to the prior round.\n"
        f"\nIMPORTANT: {_HISTORY_VERIFICATION_ONLY_RULE}\n"
        f"\nIMPORTANT: {_ANTI_THRASHING_RULE_VERDICT}\n"
    )
    if len(coupling_history) >= 2 and _CONVERGENCE_RULE_TEXT not in str(history_section or ""):
        section = section.rstrip() + f"\n\n**IMPORTANT: {_CONVERGENCE_RULE_TEXT}**\n"
    return section


def answer_format_section(parts: Sequence[str]) -> str:
    """``## Answer format`` — contract B for a seat asked ``coupling``, contract A
    for a seat asked ``change`` alone; a coupling-only seat is told which part
    it owes."""
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT, REVIEW_TWO_PART_OBJECT_CONTRACT

    if PART_COUPLING not in tuple(parts):
        return f"## Answer format\n\n{REVIEW_JSON_ARRAY_CONTRACT}"
    lead = ""
    if PART_CHANGE not in tuple(parts):
        lead = ("This seat is asked Part 2 ONLY: Part 1 above is the context you judge the "
                "coupling against, and your answer is the object with the \"coupling\" key alone.\n\n")
    return f"## Answer format\n\n{lead}{REVIEW_TWO_PART_OBJECT_CONTRACT}"


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------


def build_retrieving_brief(
    repo_dir: pathlib.Path, brief: BriefInputs,
) -> Tuple[str, Dict[str, Any]]:
    """The two-part brief for ONE retrieving seat, plus the manifest of what it
    delivered.

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
    from ouroboros.tools.review_subject import build_triad_session_task

    parts = tuple(p for p in (PART_CHANGE, PART_COUPLING) if p in tuple(brief.parts))
    coupling_checklist = load_checklist_section(COUPLING_CHECKLIST_SECTION)
    if not str(coupling_checklist or "").strip():
        raise RuntimeError(
            f"{COUPLING_CHECKLIST_SECTION} could not be loaded from docs/CHECKLISTS.md — "
            "the coupling question cannot be asked without its checklist (fail-closed)."
        )
    repo_dir = pathlib.Path(repo_dir)
    layer = str(brief.layer or "body")
    checklist_section = brief.checklist_section
    if not str(checklist_section or "").strip():
        from ouroboros.tools.review import _load_checklist_section

        checklist_section = _load_checklist_section(layer)
    intent = brief.intent
    goal_section = build_goal_section(intent.goal, intent.scope, brief.commit_message, intent.owner_words)
    scope_section = build_scope_section(intent.scope)
    rebuttal_section = build_rebuttal_section(intent.review_rebuttal)
    history_section = review_history_with_obligations(intent.review_history, drive_root=brief.drive_root,
                                                      repo_root=repo_dir)
    coupling_history_section = build_coupling_history_section(intent.coupling_history, history_section)

    bound = first_send_bound(brief)
    governance = governance_context(
        pathlib.Path(brief.governance_repo_dir or repo_dir),
        surface="triad",
        touched_paths=brief.touched_paths,
        usable_window_tokens=bound // _CHARS_PER_ESTIMATED_TOKEN,
        delivery=RETRIEVING_DELIVERY,
        checklist_section_text=checklist_section,
        already_inline=("docs/CHECKLISTS_ARCHIVE.md",) if layer == "body" else (),
        layer=layer,
        subject_root=repo_dir if layer != "body" else None,
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
            repo_dir, brief, raw, "review-preimage-" + hashlib.sha256(raw).hexdigest()[:12])
        if not source.get("path"):
            row["reason"] = "preimage not delivered: " + str(source.get("error") or "no_source_root")
            continue
        row.update(preimage_of=row["path"], disposition="deleted_preimage", **source["required_row"])
        row.pop("reason", None)
        preimage_sources.append(source)
    required_ref = required_sources_ref(required_rows or [], staged_tree_sha=tree_sha)
    index_text, index_manifest = _repository_index(repo_dir, brief.touched_paths)
    touched_slot = _touched_slot(brief)
    required_section = _required_sources_section(required_rows, required_ref, layer=layer)
    answer_format = answer_format_section(parts)

    def _assemble(diff_slot: str) -> Tuple[str, str, str]:
        """``(brief_text, part1, part2)``: Part 1 is the packet seat's own task in
        session delivery with the change evidence in its subject slot; Part 2 is
        built only for a seat asked ``coupling``."""
        part1 = "## Part 1 — The change\n\n" + build_triad_session_task(
            goal_section=goal_section, scope_section=scope_section,
            checklist_section=checklist_section, rebuttal_section=rebuttal_section,
            review_history_section=history_section, governance=governance,
            subject=brief.managed_subject, layer=layer,
            subject_root=repo_dir if layer != "body" else None,
            subject_section=f"## Subject — the staged change\n\n{touched_slot}\n\n### Staged diff\n\n{diff_slot}",
        )
        if brief.task_evidence_section.strip():
            part1 += f"\n\n{brief.task_evidence_section}"
        part2 = build_coupling_part(
            coupling_checklist=coupling_checklist, required_sources_section=required_section,
            repository_index=index_text, history_block=coupling_history_section, layer=layer,
        ) if PART_COUPLING in parts else ""
        return "\n\n".join(p for p in (part1, part2, answer_format) if p.strip()), part1, part2

    header, body, unavailable = _staged_diff(repo_dir, brief)
    delivery: Dict[str, Any] = {"diff_chars": len(body)}
    if unavailable:
        diff_slot = _retrieval_slot(brief, unavailable)
        delivery.update(diff_delivery="retrieved_by_reviewer", diff_reason=unavailable)
    else:
        diff_slot = _diff_slot(brief, header, body, "")
        measured = _first_send_chars(repo_dir, brief, _assemble(diff_slot)[0])
        # The first-send size the inlined diff must stay under: a native episode
        # must land BEFORE its landing notice (or the host would open the review
        # by announcing that the window is nearly full); a delegated session has
        # no host-enforced transcript and takes the conservative inline ceiling.
        from ouroboros.review_native_episode import native_landing_at

        ceiling = SESSION_INLINE_DIFF_CEILING_CHARS if brief.delegated else native_landing_at(bound)
        delivery.update(diff_delivery="inline", first_send_chars=measured, first_send_ceiling=ceiling)
        if measured >= ceiling:
            source = _store_review_source(repo_dir, brief, body.encode("utf-8"), DIFF_SOURCE_ID)
            if source.get("path"):
                diff_row = {**source["required_row"], "disposition": "review_subject"}
                if tree_sha:
                    diff_row["candidate_tree"] = tree_sha
                source["required_row"] = diff_row
                required_rows = [*(required_rows or []), diff_row]
                required_ref = required_sources_ref(required_rows, staged_tree_sha=tree_sha)
                required_section = _required_sources_section(required_rows, required_ref, layer=layer)
                diff_slot = _diff_slot(brief, header, body, _paged_diff_pointer(brief, body, source))
                delivery.update(diff_delivery="paged", diff_source=source)
            else:
                # An unreachable store is disclosed, never a refusal: the diff
                # stays inline and the episode's own bound decides the send.
                delivery.update(diff_paging="unavailable",
                                diff_paging_reason=str(source.get("error") or "no_source_root"))
    task_text, part1, part2 = _assemble(diff_slot)
    if delivery.get("diff_delivery") == "paged":
        delivery["first_send_chars"] = _first_send_chars(repo_dir, brief, task_text)

    manifest: Dict[str, Any] = {
        "delivery": AGENTIC_RETRIEVAL_DELIVERY,
        "coverage": "agent_retrieval",
        "parts": list(parts),
        # The PRE-RUN truth: which provenance this delivery's read receipts will
        # carry. Neither is a claim that any source WAS read — that is the
        # post-run coverage fact.
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
        "sha": {"brief": hashlib.sha256(task_text.encode("utf-8")).hexdigest(),
                "change_prompt_sha": hashlib.sha256(part1.encode("utf-8")).hexdigest(),
                "coupling_brief_sha": hashlib.sha256(part2.encode("utf-8")).hexdigest() if part2 else ""},
        "brief_sections": {
            "governance_stable_inline": len(governance.stable_inline),
            "governance_selected_inline": len(governance.selected_inline),
            "governance_navigation": len(governance.navigation),
            "change_checklist": len(checklist_section),
            "coupling_checklist": len(coupling_checklist),
            "repository_index": len(index_text),
            "intent": len(scope_section) + len(goal_section),
            "history": len(rebuttal_section) + len(history_section) + len(coupling_history_section),
            "touched_manifest": len(touched_slot),
            "required_sources": len(required_section),
            "task_evidence": len(brief.task_evidence_section),
            "diff_slot": len(diff_slot),
            "part1": len(part1), "part2": len(part2), "answer_format": len(answer_format),
        },
        **delivery,
    }
    if required_rows is not None:
        manifest["native_required_sources"] = required_rows
        manifest["native_required_sources_ref"] = required_ref
    return task_text, manifest

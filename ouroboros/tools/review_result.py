"""Result review — ``review_change(subject="result")``: does this task's result do what was asked?

One subject of the one review machinery: this module composes the subject (what was asked, the reply, what the
task produced, the host-attested verification) and the result brief; the wave itself (the panel, custody, the paid
stamp, $0 reuse, a rejoin, leaving on an addressed message, settlement, the ledger record) is ``review_change``'s
own. Every seat answers one question, judged on the result, and on the method only where the owner's words or
explicit choice fixed it: the work trace is not part of the subject. The subject is frozen once, at the call: its
exact text is retained and its sha is the wave's identity, so the same reply, material and request reuse the settled
record at $0. What the task produced, with each file's retained exact copy and the excerpts a packet seat reads
inline, is ``result_material``'s; retrieving seats read those copies, never a live path that may have changed since.

At delivery the author decides on the review it actually received (owner, 8 Nov 2026): under Advisory a revised
result is not reviewed again unless the author asks, and delivering it is the author's own acceptance of exactly
that subject beside the reviewers' verdict of the version they saw; Blocking holds any completed review of the
delivered result that did not pass it, whoever called it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.result_material import RESULT_WORDS_SOURCE, answer_text, produced_material

log = logging.getLogger(__name__)

RESULT_LAYER = "result"
RESULT_KIND = "result"
RESULT_ROOT_KIND = "task_result"
RESULT_USER_TURN = "Review the task result against its request, as the instructions above describe."
# The rules a result wave is judged by: this section of docs/CHECKLISTS.md, read from the install as every
# other review surface reads its own section (owner, 18.10: the rubric lives with the other checklists).
RESULT_CHECKLIST_SECTION = "Result Review Checklist"


def refuse_when_off() -> None:
    """Task review Off means no result review at all, whoever asks; change review and ordinary
    child critics stay available. A typed refusal before anything is built or paid."""
    from ouroboros.config import get_task_review_mode

    if get_task_review_mode() == "off":
        from ouroboros.tools.review_change import ReviewChangeArgumentError

        raise ReviewChangeArgumentError(
            "result review is off on this install (task review mode Off): no result review runs; "
            "change review and ordinary child critics stay available")


def result_rubric() -> str:
    """The ``Result Review Checklist`` section every result seat receives."""
    from ouroboros.tools.review_checklist import load_checklist_section

    return load_checklist_section(RESULT_CHECKLIST_SECTION)


def result_rules() -> Dict[str, Any]:
    """The rules a result wave is judged by, in the checklist facts' own shape: the record's provenance, never
    its reuse identity (``result_contract_fingerprint``)."""
    from ouroboros.tools.review_checklist import section_fingerprint

    return section_fingerprint(RESULT_CHECKLIST_SECTION)


# The result brief's own words: how it frames the author's goal and questions, named scope and answer to the
# previous review, in the brief's order, and what a retrieving seat must read and where.
_AUTHOR_WORDS = ("## The author's own goal and questions for this review", "## Scope the author names",
                 "## The author's answer to the previous review of this result")
_READ_SUBJECT = "The subject's exact text is retained at {source}; read it whole. "
_READ_IN_PLACE = ("Every produced file the subject names has the exact copy named beside it: read that copy, which holds "
                  "the bytes this subject names; its live path may have changed since. A symbolic link is its exact link "
                  "text, named there whole. A file without a copy is read at its path, and the subject says why.")


def result_contract_fingerprint() -> str:
    """The review contract of a result wave's reuse identity (``commit_gate.commit_review_contract_fingerprint``
    over ``result_contract_words``)."""
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint

    own, packet, retrieving = result_contract_words()
    return commit_review_contract_fingerprint(own, packet=packet, retrieving=retrieving, layer=RESULT_LAYER, shape="array")


def result_contract_words() -> Tuple[list, list, list]:
    """``(every seat, packet, retrieving)``: the result brief's own text over fixed placeholders in these branches
    (the head with and without the author's words, the subject's frame of what was asked, produced and verified,
    the packet's and a retrieving seat's composition). The rubric (a guide), the request, the reply, the produced
    material and the receipts are data."""
    from ouroboros.owner_words import reviewer_words_material
    from ouroboros.review_words import words_material
    from ouroboros.tools.review_multi_model import triad_stable_head

    heads = [_brief_head(goal=words, scope=words, review_rebuttal=words, rubric="⟨rubric⟩") for words in ("⟨words⟩", "")]
    produced = {"files": [{"path": "⟨path⟩", "kind": "⟨kind⟩", "sha256": "⟨sha⟩", "size": 1, **branch} for branch in (
                    {"retained": "⟨copy⟩"}, {"retained_gap": "⟨why⟩"}, {"link": "⟨link text⟩"})],
                "issues": ["⟨issue⟩"], "effects": [{"tool": "⟨tool⟩", "receipt": "⟨receipt⟩", "host_notes": ["⟨note⟩"]}],
                "words_section": "⟨source words⟩", "contents_section": "⟨excerpts⟩"}
    own = [RESULT_CHECKLIST_SECTION, *heads, _SUBJECT_HEAD, *_ASK_WORDS.values(), *reviewer_words_material(),
           *words_material(RESULT_WORDS_SOURCE), *(answer_text(reply=reply, produced=material, verification="⟨verification⟩")
             for reply, material in (("⟨reply⟩", produced), ("", {})))]
    return (own, [RESULT_USER_TURN, triad_stable_head(RESULT_LAYER, ""), _packet_prompt("⟨head⟩", "⟨subject⟩")],
            [_retrieving_task("⟨head⟩", source, "⟨subject⟩") for source in ("⟨source⟩", None)])


def latest_reply(ctx: Any) -> str:
    """The reply this task would deliver now (its retained delivery candidate), "" when none."""
    return str(getattr(getattr(ctx, "_delivery_candidate", None), "full_text", "") or "")


# How the subject frames what was asked and what was verified, around the host's records of both.
_ASK_WORDS = {"objective": "The objective above: {author}.", "claims": "claims the result is held to ({source}):",
              "support": " (support: {support})", "unavailable": "claims: unavailable ({error})",
              "none": "(the task states no request beyond its reply)",
              "no_receipts": "(no host-attested verification receipts)",
              "unreadable": "(verification receipts unreadable: {error})",
              "readings": "Readings: the marks in force at this ask (the work's room and the global ones), each with its author and date.",
              "reading": "- [mark {id} by {who}, {date}; {scope}{own}] {text}", "own": "; set by the work under review",
              "readings_unreadable": "Readings: the marks in force could not be read here ({error}); memory_read reads a mark."}


def _ask(ctx: Any) -> str:
    """What was asked, host-attested: the task's own request fields (the objective with its author,
    ``owner_words.objective_author_line``), the claims it is held to (an ingress contract's, else the latest plan
    revision's), its owner's words and, beside them, the marks in force at this ask with author and date (a mark
    the work under review set itself says so; an unreadable store is one line; ``marks_in_force``)."""
    from ouroboros.owner_words import objective_author_text, owner_words_text
    from ouroboros.review_evidence_sections import _accept_effective_claims, _accept_task_contract, marks_in_force

    contract = _accept_task_contract(ctx)
    lines: List[str] = []
    for key in ("goal", "objective", "requirements", "interface", "expected_output"):
        value = contract.get(key)
        if value not in (None, "", [], {}):
            lines.append(f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)}")
            if key == "objective":
                lines.append(_ASK_WORDS["objective"].format(author=objective_author_text(ctx)))
    try:
        claims, source, _exhibit = _accept_effective_claims(
            ctx, contract, getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or ""))
    except Exception as exc:
        claims, source = [], ""
        lines.append(_ASK_WORDS["unavailable"].format(error=f"{type(exc).__name__}: {exc}"))
    if claims:
        lines.append(_ASK_WORDS["claims"].format(source=source or "task contract"))
        lines.extend(f"- {claim.get('claim') or claim}" + (_ASK_WORDS["support"].format(support=claim["support"])
                                                            if claim.get("support") else "")
                     for claim in claims if isinstance(claim, dict))
    words = owner_words_text(ctx, audience="reviewer")
    if words.strip():
        lines.append(words.strip())
    try:
        readings = marks_in_force(ctx, str(getattr(ctx, "task_id", "") or "")) if getattr(ctx, "drive_root", None) else []
    except Exception as exc:  # an unreadable store hides nothing else of what was asked
        lines.append(_ASK_WORDS["readings_unreadable"].format(error=f"{type(exc).__name__}: {exc}"))
    else:  # a mark's further lines are indented under its own, as a row's words are (``words_block``)
        lines += [_ASK_WORDS["readings"], *(_ASK_WORDS["reading"].format(**{**item, "own": _ASK_WORDS["own"] if item["own"] else ""})
                                            .replace("\n", "\n  ")
                                            for item in readings)] if readings else []
    return "\n".join(lines) or _ASK_WORDS["none"]


def _verification(ctx: Any) -> str:
    from ouroboros.outcomes import read_context_verification_receipts
    from ouroboros.review_evidence_sections import _accept_receipt_exhibits, _accept_verification_summary

    task_id = str(getattr(ctx, "task_id", "") or "")
    try:
        receipts = read_context_verification_receipts(ctx, task_id, fallback_root=getattr(ctx, "drive_root", None)) if task_id else []
    except Exception as exc:
        return _ASK_WORDS["unreadable"].format(error=f"{type(exc).__name__}: {exc}")
    if not receipts:
        return _ASK_WORDS["no_receipts"]
    return json.dumps({"summary": _accept_verification_summary(receipts), "receipts": _accept_receipt_exhibits(receipts)},
                      ensure_ascii=False, indent=1, default=str)


_SUBJECT_HEAD = "# Task result under review\n\n## What was asked\n\n"


def compose_subject(*, ask: str, reply: str, produced: Dict[str, Any], verification: str) -> str:
    return _SUBJECT_HEAD + ask + "\n\n" + answer_text(reply=reply, produced=produced, verification=verification)


def _retained_subject_text(ctx: Any, digest: str) -> str:
    """The exact retained text of this task's result subject ``digest`` (``subject_source``), else ``""``."""
    from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path

    task_id = str(getattr(ctx, "task_id", "") or "")
    if len(digest) != 64 or not task_id:
        return ""
    path = f"source_handles/tool_results/result-subject-{digest[:16]}-{digest}.txt"  # subject_source's own name
    try:
        size = (task_artifact_dir_path(ctx.drive_root, task_id) / path).stat().st_size
        data = read_actor_source_bytes(ctx.drive_root, task_id, {"kind": "task_source", "root": "artifact_store",
                                                                  "path": path, "sha256": digest, "size": size})
    except (OSError, ValueError, TypeError, KeyError):
        return ""
    return data.decode("utf-8", "replace") if hashlib.sha256(data).hexdigest() == digest else ""


def earlier_request_subject(ctx: Any, subject: Any, rows: Optional[List[Dict[str, Any]]] = None, *,
                            state: str) -> Optional[Tuple[Dict[str, Any], Any]]:
    """This task's newest result record of exactly this answer (the same ``tree_sha``: reply, produced material,
    verification) that judged an earlier request, in ``state`` ("pending": a review still running, collected at
    $0; "settled": a review of this answer before the owner's words changed), with its own frozen subject rebuilt
    from its retained text. A refusal is no review of anything; an unreadable text yields nothing."""
    import dataclasses

    for row in rows if rows is not None else _result_rows(ctx):
        recorded = row.get("subject") or {}
        if (str(recorded.get("tree_sha") or "") != subject.tree_sha or row.get("state") != state
                or str(recorded.get("diff_sha") or "") == subject.diff_sha
                or str((row.get("verdict") or {}).get("aggregate") or "") == "NOT_DISPATCHED"):
            continue
        text = _retained_subject_text(ctx, str(recorded.get("diff_sha") or ""))
        if text.startswith(_SUBJECT_HEAD):
            return row, dataclasses.replace(subject, diff_text=text, diff_sha=hashlib.sha256(text.encode("utf-8")).hexdigest())
    return None


def _words_since(ctx: Any, earlier: str) -> List[str]:
    """My human's rows this run holds now that the earlier subject's request did not carry, each under its stamp:
    told by the row's identity, never by likeness of words (``owner_relay.words_unseen_in``; what my human
    replied to is not theirs; a subject that cannot be read back names every row)."""
    from ouroboros import owner_relay, owner_words

    return [f"{owner_words.words_stamp(row)}\n{row['text']}" for row in owner_relay.words_unseen_in(ctx, earlier)]


def _request_changed_text(ctx: Any, record: Dict[str, Any], *, outcome: bool = False) -> str:
    """Asked once (owner, 18.10): the only review of this exact answer judged an earlier request. ``outcome``:
    its author has not received that review yet, so its findings ride the same question, whole."""
    earlier = _retained_subject_text(ctx, str((record.get("subject") or {}).get("diff_sha") or ""))
    words = _words_since(ctx, earlier)
    aggregate = str((record.get("verdict") or {}).get("aggregate") or "")
    lines = [f"[Result review] This exact result was reviewed (record {record.get('record_id')}: {aggregate}) before "
             "what was asked last changed; its reviewers did not see these words of your human:"]
    lines += [f"- {text}".replace("\n", "\n  ") for text in words] or [  # each whole and dated, every one
        "- (the request changed: its claims, contract or readings; no row of your human is new to that review)"]
    found = _outcome_lines(record) if outcome else []
    if found:
        lines += ["What that review found:", *found]
    lines.append("Do they change what was asked? If they do, revise the result, or ask for a review of the current "
                 "request with review_change(subject=\"result\"). If they do not, deliver again: the earlier review "
                 "stands for this result (its findings and enforcement apply as they are).")
    return "\n".join(lines)


def running_earlier_request(ctx: Any, subject: Any) -> Any:
    """The mind's own call of an unchanged answer whose review is still running under an earlier request
    collects that review at $0 (its frozen subject); else the live subject stands."""
    found = earlier_request_subject(ctx, subject, state="pending")
    return found[1] if found is not None else subject

def result_subject(ctx: Any, request: Any, *, governance_root: Any, answer: Optional[str] = None,
                   produced: Optional[Dict[str, Any]] = None) -> Any:
    """The frozen result subject of this call (``review_subject.FrozenSubject``): its text, its sha as
    the wave's identity, and as root the folder its retrieving seats read (the workspace the task
    changed, else the folder holding the retained subject). ``answer`` is the host's delivery text
    (the Required floor); a mind's call names its reply in ``request.answer`` or reviews its latest."""
    from ouroboros.tools.review_change import ReviewChangeArgumentError
    from ouroboros.tools.review_subject import FrozenSubject, ReviewSubjectSpec

    reply = str(answer if answer is not None else (getattr(request, "answer", "") or latest_reply(ctx)) or "")
    produced = produced if produced is not None else produced_material(ctx)
    if not reply.strip() and not produced["files"] and not produced["effects"] and not produced.get("observed"):
        raise ReviewChangeArgumentError("subject=result has nothing to review: no reply and nothing this task produced")
    answer_part = answer_text(reply=reply, produced=produced, verification=_verification(ctx))
    answer_sha = hashlib.sha256(answer_part.encode("utf-8")).hexdigest()
    text = _SUBJECT_HEAD + _ask(ctx) + "\n\n" + answer_part  # what is asked now (see earlier_request_subject)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source = subject_source(ctx, text)
    root = produced["workspace"] or (str(source.parent) if source else str(pathlib.Path(ctx.drive_root)))
    spec = ReviewSubjectSpec(root_kind=RESULT_ROOT_KIND, root=root, kind=RESULT_KIND, governance_root=str(governance_root),
                             surface="result", body_fact="unknown", body_how="", layer=RESULT_LAYER)
    return FrozenSubject(spec=spec, diff_text=text, diff_sha=digest, tree_sha=answer_sha, parent_sha="",
                         name_status=tuple(("M", str(item["path"])) for item in produced["files"]))


def subject_source(ctx: Any, text: str) -> Optional[pathlib.Path]:
    """The subject's exact text, retained once in the task's artifact store (write-once by digest:
    the same text is the same file); ``None`` when it could not be retained."""
    from ouroboros.artifacts import persist_exact_text_source, task_artifact_dir_path

    root, task_id = getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or "")
    if root is None or not task_id:
        return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    exact, ref, gap = persist_exact_text_source(root, task_id, source_id=f"result-subject-{digest[:16]}", text=text)
    if gap or exact != text or not ref.get("path"):
        log.warning("result subject of %s not retained: %s", task_id, gap.get("reason") if gap else "readback mismatch")
        return None
    return task_artifact_dir_path(root, task_id) / str(ref["path"])


def _brief_head(*, goal: str, scope: str, review_rebuttal: str, rubric: Optional[str] = None) -> str:
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT

    parts = ["## REVIEW INSTRUCTIONS", "", result_rubric() if rubric is None else rubric]
    for heading, words in zip(_AUTHOR_WORDS, (goal, scope, review_rebuttal)):
        if words.strip():
            parts += [heading, "", words.strip(), ""]
    return "\n".join([*parts, "## Your answer", "", REVIEW_JSON_ARRAY_CONTRACT])


def _packet_prompt(head: str, subject_text: str) -> str:
    return f"{head}\n\n---\n\n# THE SUBJECT\n\n{subject_text}"


def _retrieving_task(head: str, source: Any, subject_text: str) -> str:
    where = _READ_SUBJECT.format(source=source) if source else ""
    return f"{head}\n\n{where}{_READ_IN_PLACE}\n\n---\n\n# THE SUBJECT\n\n" + subject_text


def prepare_result_review(ctx: Any, subject: Any, *, goal: str = "", scope: str = "",
                          review_rebuttal: str = "") -> Tuple[Optional[Dict[str, Any]], Optional[str], bool]:
    """The result wave assembled for ``review._dispatch_unified_review`` WITHOUT dispatching anything, in
    ``review._prepare_unified_review``'s return shape: every seat of the composed panel answers the one
    result question; a packet seat receives the subject inline, a retrieving seat its own brief naming the
    retained subject and the produced files to read in place."""
    from ouroboros.review_checking import SUBJECT_WORKTREE, wave_checking_source
    from ouroboros.reviewer_slot_config import commit_triad_delivery, row_plan_retrieves
    from ouroboros.tools.review_admission import seat_vectors
    from ouroboros.tools.review_helpers import review_drive_root
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT

    try:
        row_plan = seat_vectors(commit_triad_delivery())
    except ValueError as exc:
        ctx._last_review_block_reason = "infra_failure"
        return None, f"⚠️ REVIEW_BLOCKED: invalid review pool configuration — {exc}", True
    count = len(row_plan.get("models") or [])
    row_plan["parts"] = [("result",)] * count
    head = _brief_head(goal=goal, scope=scope, review_rebuttal=review_rebuttal)
    source = subject_source(ctx, subject.diff_text)
    tasks, policies, shas = [""] * count, [None] * count, [""] * count
    texts: Dict[str, str] = {}
    manifests: List[Dict[str, Any]] = []
    for index in range(count):
        if not row_plan_retrieves(row_plan, index):
            continue
        task = _retrieving_task(head, source, subject.diff_text)
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
        tasks[index], shas[index], texts[digest] = task, digest, task
        policies[index] = {"output_contract": REVIEW_JSON_ARRAY_CONTRACT, "output_shape": "array"}
        manifests.append({"slot_id": str(row_plan["slot_ids"][index]), "delivery": "retrieving", "parts": ["result"],
                          "brief_chars": len(task), "sha": {"brief": digest}})
    row_plan.update(session_tasks=tasks, session_policies=policies, brief_shas=shas)
    routes = list(row_plan.get("routes") or [])
    accesses = list(row_plan.get("session_accesses") or [""] * len(routes))
    seats = [SimpleNamespace(route=route, session_access=access) for route, access in zip(routes, accesses)]
    packet = [str(row_plan["slot_ids"][index]) for index in range(count) if not row_plan_retrieves(row_plan, index)]
    prompt = _packet_prompt(head, subject.diff_text) if packet else ""
    return {
        "prompt": prompt, "stable_prefix_len": len(head) if packet else 0,
        "models": row_plan["models"], "routes": routes, "row_plan": row_plan,
        "session_task": "", "target_repo": str(subject.spec.root),
        "task_evidence": {}, "preflight_execution": {}, "layer": RESULT_LAYER,
        "governance_manifest": [], "governance_packet_slots": packet,
        "checking_source": wave_checking_source(subject.spec.root, seats, data_root=review_drive_root(ctx),
                                                subject=SUBJECT_WORKTREE),
        "retrieving_manifests": manifests, "brief_texts": texts, "user_turn": RESULT_USER_TURN,
    }, None, False


# --- The delivery step: binding, and the Required floor ------------------------------------------

def _is_root(ctx: Any) -> bool:
    from ouroboros.task_results import resolve_ctx_lineage

    return bool(resolve_ctx_lineage(ctx).get("is_root_task"))


def required_root_wave(ctx: Any) -> bool:
    """Host fact for panel composition: task review is Required and this task is a root.

    Auto, Off and a child stay on the free ``result`` surface. The model cannot
    supply this; ``compose_panel`` reads it from here.
    """
    from ouroboros.config import get_task_review_mode

    return get_task_review_mode() == "required" and _is_root(ctx)


def required_material(ctx: Any, llm_trace: Dict[str, Any], content: str,
                      produced: Optional[Dict[str, Any]] = None) -> str:
    """Why a result carries work the Required floor guarantees a review for (owner, 8 Nov 2026: created results
    plus an explicitly required review of the answer), else "": ``"observed_production"`` — a file or effect the
    task produced, or a host-observed production whose material capture failed (that failure is a named gap,
    never "nothing produced"); ``"answer_required"`` — the task contract's ``review_answer``; ``"answer_protocol"``
    — the answer line of an adapter's answer protocol (a benchmark's answer is its work product). The expected
    result a task was given is intent, and a plain prose request is the mind's own call, never production."""
    from ouroboros.contracts.task_contract import answer_protocol_active, review_answer_required
    from ouroboros.outcomes import selected_final_answer

    produced = produced if produced is not None else produced_material(ctx)
    if produced["files"] or produced["effects"] or produced.get("observed"):
        return "observed_production"
    if review_answer_required(ctx) and str(content or "").strip():
        return "answer_required"
    return "answer_protocol" if answer_protocol_active(ctx) and selected_final_answer(str(content or ""), llm_trace) else ""


def _result_rows(ctx: Any) -> List[Dict[str, Any]]:
    from ouroboros import review_ledger as ledger

    try:
        return ledger.recent_records(ledger.ledger_root(ctx), str(getattr(ctx, "task_id", "") or ""), limit=50,
                                     hot_only=True, surface="result")
    except Exception:
        log.debug("result records of %s unreadable", getattr(ctx, "task_id", ""), exc_info=True)
        return []


def settled_result_record(ctx: Any, subject_sha: str, rows: Optional[List[Dict[str, Any]]] = None, *,
                          refusal: bool = False) -> Optional[Dict[str, Any]]:
    """This task's settled result review of exactly this subject (by the mind or the host), else ``None``; with
    ``refusal``, its retained $0 refusal instead (no allowance could buy it: a review of nothing, yet its outcome)."""
    from ouroboros import review_ledger as ledger

    for row in rows if rows is not None else _result_rows(ctx):
        if (str((row.get("subject") or {}).get("diff_sha") or "") == subject_sha and row.get("state") == "settled"
                and (str((row.get("verdict") or {}).get("aggregate") or "") == "NOT_DISPATCHED") is refusal):
            try:
                return ledger.load_record(ledger.ledger_root(ctx), str(row.get("record_id") or ""))
            except ValueError:
                return None
    return None


def running_result_review(rows: List[Dict[str, Any]], subject_sha: str) -> str:
    """The record id of this task's result review of exactly this subject that is still running, else ``""``."""
    for row in rows:
        if (str((row.get("subject") or {}).get("diff_sha") or "") == subject_sha
                and row.get("state") == "pending" and row.get("record_id")):
            return str(row["record_id"])
    return ""


def _collect_running(ctx: Any, content: str, record_id: str, subject: Any) -> Dict[str, Any]:
    """Rejoin the running result review ``record_id`` at $0: the same brief and frozen subject, under its
    recorded round (the record's identity never moves to material its seats did not see)."""
    from ouroboros import review_ledger as ledger
    from ouroboros.tools.review_change import run_review_change

    from ouroboros.review_continuation import ContinuationRefused

    record = ledger.load_record(ledger.ledger_root(ctx), record_id)
    brief = dict(record.get("brief") or {})
    try:  # a continuation chosen on this round is sent first (``review_collection.repair_round``)
        return run_review_change(ctx, subject="result", answer=str(content or ""), goal=str(brief.get("goal") or ""),
                                 scope=str(brief.get("scope") or ""),
                                 author_questions=list(brief.get("author_questions") or []),
                                 reason="Collect the running review of this result", _frozen_result=subject,
                                 _rejoin_retry_key=str((record.get("fingerprints") or {}).get("retry_key") or ""))
    except ContinuationRefused as exc:  # recorded, not sent now: the review stays pending, never a new wave
        return {"state": "pending", "record_id": record_id, "continuation": exc.payload()}


def supersede_result_runs(llm_trace: Dict[str, Any], marker: str) -> None:
    """A new delivery candidate (a changed answer, or the same answer over changed evidence) is a new subject,
    on every rail: the result runs bound so far speak for an earlier one. Superseded, never deleted."""
    for run in llm_trace.get("review_runs") or []:
        if isinstance(run, dict) and run.get("surface") == RESULT_KIND and not run.get("superseded_by_revision"):
            run["superseded_by_revision"] = marker


def _supersede_other_result_runs(llm_trace: Dict[str, Any], subject_sha: str) -> None:
    """A result run of an earlier answer never speaks for this one: superseded, never deleted."""
    for run in llm_trace.get("review_runs") or []:
        if (isinstance(run, dict) and run.get("surface") == RESULT_KIND and run.get("binding_hash") != subject_sha
                and not run.get("superseded_by_revision")):
            run["superseded_by_revision"] = subject_sha


def _bind_review_run(llm_trace: Dict[str, Any], record: Dict[str, Any], subject_sha: str,
                     enforcement: str, *, earlier_request: bool = False, earlier_revision: bool = False) -> Dict[str, Any]:
    """The result review bound to the delivered answer, in the task's review runs the outcome axes read.
    ``earlier_request``: the mind judged that the owner's words since this review do not change what was
    asked, so the review of this exact answer stands for the current request too. ``earlier_revision``: the
    review judged an earlier version of this result and stands beside the delivered one as that version's
    (``reviewed_subject_sha``), never as a review of the delivered bytes (owner, 8 Nov 2026)."""
    from ouroboros.review_seat_outcome import outcome_identity

    runs = llm_trace.setdefault("review_runs", [])
    record_id = str(record.get("record_id") or "")
    bound = next((run for run in runs if isinstance(run, dict) and run.get("panel_id") == record_id), None)
    verdict, outcomes = dict(record.get("verdict") or {}), outcome_identity(record)
    aggregate = str(verdict.get("aggregate") or "")
    for refused in (run for run in runs if aggregate != "NOT_DISPATCHED" and not earlier_revision and isinstance(run, dict)):
        if refused.get("aggregate") == "NOT_DISPATCHED" and refused.get("binding_hash") == subject_sha:
            refused["superseded_by_revision"] = record_id  # a review that ran answers what its $0 refusal could not
    signal = aggregate if aggregate in {"PASS", "FAIL"} else "DEGRADED"
    facts = {"aggregate_signal": signal, "aggregate": aggregate, "degraded": signal == "DEGRADED",
             "reason": str(verdict.get("reason") or ""), "critical_count": len(verdict.get("critical_findings") or []),
             "dispatch_refusal": str((record.get("dispatch_refusal") or {}).get("kind") or "")}
    reviewed = str((record.get("subject") or {}).get("diff_sha") or "")
    if bound is not None:
        if earlier_request or earlier_revision or reviewed == subject_sha:
            bound["binding_hash"] = subject_sha
        if earlier_request:
            bound["earlier_request"] = True
        if earlier_revision:
            bound.update(earlier_revision=True, reviewed_subject_sha=reviewed)
        elif reviewed == subject_sha:  # the exact subject it reviewed is delivered again
            for key in ("earlier_revision", "reviewed_subject_sha"):
                bound.pop(key, None)
        if bound.get("binding_hash") == subject_sha:
            bound.pop("superseded_by_revision", None)  # the exact subject it reviewed is current again
        if record.get("state") != "pending":
            bound.pop("pending_finish", None)  # the review an earlier delivery left running has settled
        # A genuinely new outcome under the same record (a continued seat, a late answer under the same
        # operation) refreshes the binding; the author's earlier informed finish read the old outcome.
        known = bound.get("outcomes")
        if (known != outcomes) if known is not None else any(bound.get(k) != facts[k] for k in ("aggregate", "critical_count", "reason")):
            bound.update(facts)
            bound.pop("informed_finish", None)
        bound["outcomes"] = outcomes
        return bound
    bound = {"surface": RESULT_KIND, "authority": "host_root", "panel_id": record_id, "record_id": record_id,
             "binding_hash": subject_sha, **facts, "enforcement": enforcement, "outcomes": outcomes,
             **({"earlier_request": True} if earlier_request else {}),
             **({"earlier_revision": True, "reviewed_subject_sha": reviewed} if earlier_revision else {})}
    runs.append(bound)
    return bound


def _blocking_enforcement() -> bool:
    """Blocking left the product (step 4); the branches below are removed with the hold code after 25.11."""
    return False


def pending_review_choice_offered(ctx: Any) -> bool:
    """While a result review of this answer is still running, a nonblocking install lets the mind
    choose ``pending_review=finish``: deliver now, the reviewers settle later."""
    return bool(getattr(ctx, "_result_review_pending", "")) and not _blocking_enforcement()


def _outcome_lines(record: Dict[str, Any]) -> List[str]:
    """A review's outcome, whole: every finding in the record's order (critical, advisory, then an added
    reviewer's, heard and never counted) and why no verdict formed when none did. Nothing is cut."""
    verdict = dict(record.get("verdict") or {})
    aggregate = str(verdict.get("aggregate") or "")
    lines = [f"- {finding.get('severity') or 'advisory'} ({finding.get('seat_id') or 'reviewer'}{note}): "
             f"{finding.get('item') or ''} {finding.get('reason') or ''}".rstrip()
             for name, note in (("critical_findings", ""), ("advisory_findings", ""),
                                ("additional_findings", "; added reviewer, heard, not counted"))
             for finding in verdict.get(name) or [] if isinstance(finding, dict)]
    from ouroboros.review_continuation import continuable_seats

    if offers := continuable_seats(record):  # continued only by the author's explicit choice, in the same wave
        lines.append("Stopped reviewers you may continue: review_change(review_reference="
                     + json.dumps({"record_id": record.get("record_id"), "task_id": record.get("task_id"), "surface": "result"})
                     + ", continue_seats=" + json.dumps([{k: o[k] for k in ("seat_id", "operation_id")} for o in offers]) + ").")
    if aggregate == "NOT_DISPATCHED":
        lines.append(str((record.get("dispatch_refusal") or {}).get("message") or "No review was dispatched."))
    elif aggregate not in {"PASS", "FAIL"}:
        lines.append(f"No verdict was formed ({verdict.get('reason') or 'unknown'}); the record names each seat's state.")
    return lines


def _findings_text(record: Dict[str, Any], *, blocking: bool, required: bool = True, revised: bool = False,
                   raised: Optional[bool] = None) -> str:
    """Hand back the whole outcome and its actual recovery. Advisory leaves another wave to the author;
    Blocking needs PASS, Required alone adds a host wave; a raised cycle cap permits the author's explicit retry."""
    aggregate = str((record.get("verdict") or {}).get("aggregate") or "")
    capped = (record.get("dispatch_refusal") or {}).get("kind") == "review_cycles_exhausted"
    raised = (capped and _ceiling(record) != _ceiling()) if raised is None else bool(raised and capped)
    lines = [(f"[Result review] An earlier version of your result was reviewed (record {record.get('record_id')}): "
              f"{aggregate}. Your current result differs and was not reviewed; this verdict stays that version's."
              if revised else
              f"[Result review] Your result {'was not reviewed' if aggregate == 'NOT_DISPATCHED' else 'was reviewed before delivery'} (record {record.get('record_id')}): {aggregate}."),
             *_outcome_lines(record)]
    lines.append(("Blocking enforcement: no review of this result can be bought until your human raises the review "
                  "ceiling, and it is not delivered unreviewed; stop the task explicitly, or wait for that."
                  if capped and not raised else
                  "Blocking enforcement: this result is not delivered until a review passes it; " + (
                      "the review ceiling was raised since its refusal: ask for that review again" if raised else
                      "resolve the recorded refusal and ask for its review" if aggregate == "NOT_DISPATCHED" else
                      "revise the result" + ("" if required else " and ask for its review"))
                  + ("" if required and not raised else " with review_change(subject=\"result\")")
                  + " or stop the task explicitly.") if blocking else
                 f"You decide: deliver it, or revise it ({'a revision is not delivered until a review passes it' if _blocking_enforcement() else 'a revision is not reviewed again unless you ask'}); nothing is delivered until you answer again."
                 if aggregate == "PASS" and not revised else
                 "You decide: revise the result, deliver it as it is" + ("" if aggregate == "PASS" else " with your reasons")
                 + " (finish_task rationale), or ask for another review with review_change(subject=\"result\"), which "
                 "may spend a paid review; a revised result is not reviewed again unless you ask, and delivering it is your own "
                 "acceptance beside this verdict. Nothing is delivered until you answer again.")
    return "\n".join(lines)


def _outcome_keys(record: Dict[str, Any]) -> set:
    """What a review concluded, by identity: each seat's actual outcome with its status, counted seats and
    added ones alike (``review_waiting.outcome_states``), else the record itself (a $0 refusal has no seat
    outcome). A revision that records cost keeps it; a new outcome under the same record changes it."""
    from ouroboros.review_waiting import outcome_states

    return set(outcome_states(record)) or {str(record.get("record_id") or "")}


def _outcome_received(ctx: Any, llm_trace: Dict[str, Any], record: Dict[str, Any]) -> bool:
    """Whether this exact outcome already reached its author whole: shown by this step's hold (``_result_review_shown``)
    or returned complete by a review tool with its ``review_waiting.publish_delivery`` receipt (no ``result_partial``;
    a $0 refusal, no seat's, is whole in it). A [Selected results] notice or readiness peek is no such delivery."""
    return _outcome_keys(record) <= _received_states(ctx, llm_trace, str(record.get("record_id") or ""))


def _received_states(ctx: Any, llm_trace: Dict[str, Any], record_id: str) -> set:
    seen = set(getattr(ctx, "_result_review_shown", None) or ())
    for call in llm_trace.get("tool_calls") or []:
        meta = call.get("tool_result_meta") if isinstance(call, dict) else None
        receipt = meta.get("review_delivery") if isinstance(meta, dict) else None
        if isinstance(receipt, dict) and not call.get("result_partial") and str(receipt.get("record_id") or "") == record_id:
            seen.update([*map(str, receipt.get("outcome_states") or ()), record_id])
    return seen


def _ready_first(ctx: Any, llm_trace: Dict[str, Any], record_id: str, hold: Any) -> bool:
    """Before a reasoned pending finish (owner, 8 Nov 2026), what reviewers of the running review answered since its
    author last received it is handed back whole, once, and the author decides again; nothing new, no hold."""
    from ouroboros import review_ledger as ledger
    from ouroboros.review_waiting import outcome_states

    try:
        record = ledger.load_record(ledger.ledger_root(ctx), record_id) or {}
    except ValueError:
        record = {}
    seen, closed = _received_states(ctx, llm_trace, record_id), [r for r in record.get("rows") or [] if not ledger.seat_open(r)]
    if not (ready := [row for row, state in zip(closed, outcome_states(record)) if state not in seen]):
        return False
    ctx._acceptance_pending_review_choice = ""  # chosen before these answers: the author chooses again after them
    return _show(ctx, hold, f"[Result review] Reviewers answered while the review of your result still runs (record "
                 f"{record_id}; {len(record.get('rows') or []) - len(closed)} still answering), not yet read by you:\n"
                 + json.dumps(ready, ensure_ascii=False, indent=2, default=str) + "\nFinish again with "
                 "pending_review=finish and your rationale to deliver now (the rest settle later and are kept), or "
                 "wait: your next delivery collects the same review at $0.", record, "pending_ready")


def _show(ctx: Any, hold: Any, text: str, record: Dict[str, Any], cause: str, *, outcome: bool = True) -> bool:
    """Insert the handback, then stage its receipt for the transcript checkpoint, like a whole tool return.
    A question about changed owner words alone certifies no newly rendered outcome."""
    ctx._completion_hold_cause = cause  # each hold names its cause to the delivery owner
    hold(text)
    if outcome:
        from ouroboros.review_waiting import acknowledge_delivery, outcome_states

        shown = getattr(ctx, "_result_review_shown", None) or []
        ctx._result_review_shown = list(dict.fromkeys([*shown, *sorted(_outcome_keys(record))]))
        acknowledge_delivery(ctx, {"record_id": str(record.get("record_id") or ""),
                                   "outcome_states": outcome_states(record)})
    return True


def _ceiling(refusal: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """The review ceiling in force, or the one a $0 refusal met (its ``dispatch_refusal.cap``) when it names it."""
    from ouroboros.review_cycles import review_max_cycles
    return (refusal.get("dispatch_refusal") or {}).get("cap", review_max_cycles()) if refusal else review_max_cycles()


def _acceptance(ctx: Any, subject_sha: str, record: Optional[Dict[str, Any]], *, pending: bool = False) -> Dict[str, Any]:
    """The author's own acceptance of the delivered result (owner, 8 Nov 2026): bound to its exact subject (the ask,
    the reply, the files and effects, the verification) and to the outcome it actually received, with the reason its
    completion gave. Exposure, never agreement; never a reviewer verdict and never carried to another subject."""
    rationale = str((getattr(ctx, "_completion_selected", None) or {}).get("rationale") or "").strip()
    return {"subject_sha": subject_sha, "record_ids": [str(record.get("record_id") or "")] if record else [],
            "outcomes": sorted(_outcome_keys(record)) if record else [],
            **({"rationale": rationale} if rationale else {}), **({"pending": True} if pending else {})}


def _after_refusal(ctx: Any, llm_trace: Dict[str, Any], subject_sha: str) -> bool:
    """A $0 refusal of this result's review (its allowance is spent) was already shown, and the mind delivers: its
    own informed decision. (Blocking holds until a review passes; a raised ceiling lets only Required's floor run.)"""
    run = next((item for item in llm_trace.get("review_runs") or [] if isinstance(item, dict)
                and item.get("surface") == RESULT_KIND and item.get("binding_hash") == subject_sha
                and not item.get("superseded_by_revision")), None)
    if run is not None:
        run["informed_finish"] = _acceptance(ctx, subject_sha, {"record_id": run.get("record_id")})
    return False


def record_review_decision(llm_trace: Dict[str, Any], trigger: str, reason: str = "", *, ctx: Any = None) -> None:
    """Configured mode and trigger, never mode as eligibility: the outcome derives that from bound reviews.
    This delivery replaces the decision, not those reviews or their verdicts."""
    from ouroboros.config import get_task_review_mode

    decision = llm_trace.setdefault("review_decision", {})
    decision.pop("eligibility", None)  # older delivery decisions put the mode in this field
    decision.update(mode=get_task_review_mode(), trigger=trigger)
    if reason:
        decision["reason"] = reason
    else:
        decision.pop("reason", None)


def _other_version(ctx: Any, subject: Any, rows: List[Dict[str, Any]], *, state: str) -> Any:
    """This task's newest result review of another version of its result (another answer or material), in ``state``: a
    pending one as ``(row, its frozen subject)`` from its retained text (collectable at $0), a settled one as its record,
    its $0 refusal too (a revision keeps the review chosen for it, as after a FAIL; root, 9 Nov). Never a child's."""
    import dataclasses

    from ouroboros import review_ledger as ledger

    task_id = str(getattr(ctx, "task_id", "") or "")
    for row in rows:
        recorded = row.get("subject") or {}
        if (row.get("state") != state or str(row.get("task_id") or task_id) != task_id
                or str(recorded.get("diff_sha") or "") == subject.diff_sha):
            continue
        if state == "settled":
            try:
                return ledger.load_record(ledger.ledger_root(ctx), str(row.get("record_id") or ""))
            except ValueError:
                return None
        text = _retained_subject_text(ctx, str(recorded.get("diff_sha") or ""))
        if not text.startswith(_SUBJECT_HEAD):
            return None  # not collectable under its own subject: no duty this delivery can rejoin at $0
        return row, dataclasses.replace(subject, diff_text=text, diff_sha=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                        tree_sha=str(recorded.get("tree_sha") or subject.tree_sha))
    return None


def _finish_pending(ctx: Any, llm_trace: Dict[str, Any], record_id: str, subject: Any) -> bool:
    """A reasoned Advisory finish while the owed review still runs (owner, 8 Nov 2026; never the default): the author's
    acceptance of the exact subject beside no conclusion yet (Done with warnings); the late outcome stays owed."""
    runs = llm_trace.setdefault("review_runs", [])
    run = next((item for item in runs if isinstance(item, dict) and item.get("panel_id") == record_id), None)
    if run is None:
        run = {"surface": RESULT_KIND, "authority": "host_root", "panel_id": record_id, "record_id": record_id,
               "aggregate_signal": "", "aggregate": "", "degraded": False, "enforcement": "advisory"}
        runs.append(run)
    run.update(binding_hash=subject.diff_sha, pending_finish=True)
    run.pop("superseded_by_revision", None)
    run["informed_finish"] = _acceptance(ctx, subject.diff_sha, {"record_id": record_id}, pending=True)
    record_review_decision(llm_trace, "pending_finish", ctx=ctx)
    return False


def _decided(ctx: Any, content: str) -> bool:
    """The author's own reasoned finish of exactly this answer in this delivery (``finish_task`` with a rationale),
    taken after what it had received: a held delivery clears it, and another answer never inherits it."""
    chosen = getattr(ctx, "_completion_selected", None) or {}
    return chosen.get("action") == "finish" and bool(str(chosen.get("rationale") or "").strip()) and (
        chosen.get("answer") == content or chosen.get("answer_sha256") == hashlib.sha256(content.encode("utf-8")).hexdigest())


def _revised(ctx: Any, hold: Any, llm_trace: Dict[str, Any], subject: Any, record: Dict[str, Any], *,
             blocking: bool, required: bool, content: str = "", speech: bool = False) -> bool:
    """Deliver a result whose review judged an earlier version of it (owner, 8 Nov 2026). Advisory: an unreceived
    outcome is shown first, owner words its reviewers did not see are asked about once (owner, 18.10), and a non-PASS
    one is put before the author once more unless its reasoned finish of this answer already chose (``_decided``);
    then that review is bound beside the delivered subject as an earlier version's, with the author's acceptance (an
    earlier PASS so accepted is Done with that label; Presence ``speech`` departed: nothing shown, asked or held).
    Blocking holds until a review passes the delivered version."""
    received, record_id = _outcome_received(ctx, llm_trace, record), str(record.get("record_id") or "")
    if blocking:
        return _show(ctx, hold, _findings_text(record, blocking=True, required=required, revised=True), record,
                     "revision", outcome=not received)
    pair, aggregate = [record_id, subject.diff_sha], str((record.get("verdict") or {}).get("aggregate") or "")
    asked, words_asked = (getattr(ctx, name, None) or [] for name in ("_result_review_revision_asked",
                                                                      "_result_review_request_asked"))
    words = [] if speech or pair in words_asked else _words_since(ctx, _retained_subject_text(ctx, str((record.get(
        "subject") or {}).get("diff_sha") or "")))
    if not speech and (not received or words or (aggregate != "PASS" and pair not in asked and not _decided(ctx, content))):
        ctx._result_review_revision_asked = [*asked, pair]
        ctx._result_review_request_asked = [*words_asked, pair] if words else words_asked
        said = ["Its reviewers did not see these words of your human; do they change what was asked?",
                *(f"- {text}".replace("\n", "\n  ") for text in words)] if words else []
        return _show(ctx, hold, "\n".join([_findings_text(record, blocking=False, revised=True), *said]), record,
                     "revision", outcome=not received)
    run = _bind_review_run(llm_trace, record, subject.diff_sha, "advisory", earlier_revision=True)
    run["informed_finish"] = _acceptance(ctx, subject.diff_sha, record)
    return False


def _presence_turn_speech(ctx: Any) -> bool:
    """A Presence turn delivering its own speech, known before any review: the host built the turn for one
    connector event and stamped its ``presence_event_identity`` (``presence_runner._build_task``; the carrier
    stamps it again). A promoted or follow-up root keeps only ``metadata.presence``
    (``dialogue_provenance.presence_root_carrier``) and is reviewed like any root, whatever completion its
    context holds: a focus records one (``_presence_completion``) only after this step
    (``place_reply.prepare_focus_reply``), for a promoted root of a place too."""
    metadata = getattr(ctx, "task_metadata", None)
    return isinstance(metadata, dict) and bool(metadata.get("presence_event_identity"))


def _presence_result_review(ctx: Any, content: str, llm_trace: Dict[str, Any], subject: Any,
                            record: Optional[Dict[str, Any]], *, owed: bool) -> bool:
    """Presence speech is never held. A Required review of its result still runs: its author departs at
    once (``review_pause.departed_binding``), nobody waits for the seats, and the outcome settles as a late
    mark in the room the work came from (#1738); a review that settled at once, and a $0 refusal (the result
    reviews' paid-cycle allowance is spent), are bound like any other. An outcome that settled unread owes that
    mark itself (``left="unread"``) until a saved whole receipt of it ends the duty
    (``review_waiting.acknowledge_delivery``)."""
    if record is None and owed:
        from ouroboros import review_pause
        from ouroboros.tools.review_change import ReviewChangeArgumentError, run_review_change

        try:
            with review_pause.receptive_author(review_pause.departed_binding(ctx)):
                outcome = run_review_change(ctx, subject="result", answer=str(content or ""), _frozen_result=subject,
                                            reason="Required: the host's review of a Presence result with produced work",
                                            _chosen_by="host")
        except ReviewChangeArgumentError as exc:
            log.warning("Required result review of Presence %s not dispatched: %s", getattr(ctx, "task_id", ""), exc)
            return False
        record = settled_result_record(ctx, subject.diff_sha) or (  # else its $0 refusal, read as delivery reads it
            settled_result_record(ctx, subject.diff_sha, refusal=True) if outcome.get("state") != "pending" else None)
    if record is not None:
        earlier = str((record.get("subject") or {}).get("diff_sha") or "") != subject.diff_sha
        _bind_review_run(llm_trace, record, subject.diff_sha, "blocking" if _blocking_enforcement() else "advisory",
                         earlier_request=earlier)
        if str((record.get("verdict") or {}).get("aggregate") or "") == "NOT_DISPATCHED":
            ctx._result_review_refused = {**(getattr(ctx, "_result_review_refused", None) or {}), subject.diff_sha: {
                "ceiling": _ceiling(record), "record_id": str(record.get("record_id") or "")}}
        elif not _outcome_received(ctx, llm_trace, record):
            # ``waiting`` may hold only cursors. Keep an existing wave duty; otherwise owe its unread outcome.
            from ouroboros import review_pause

            review_pause.owe_unwaited_review(ctx, surface=RESULT_KIND, left="unread",
                                             record_id=str(record.get("record_id") or ""),
                                             retry_key=str((record.get("fingerprints") or {}).get("retry_key") or ""))
    return False


def _owed_wave(ctx: Any, content: str, llm_trace: Dict[str, Any], hold: Any, subject: Any, collect: Any, running: str,
               *, required: str, other: Any) -> Tuple[Optional[bool], Optional[Dict[str, Any]]]:
    """The one wave a delivery owes, through the same machinery as a mind's call: the running review (of this result or
    an earlier version, ``collect``) collected at $0, else Required's floor seated by the host. ``(decision, None)`` when
    the delivery ends here (nothing dispatched; still running unless its author chose ``pending_review=finish``), else
    ``(None, its settled record)``, a $0 refusal included (the allowance is spent): only Blocking holds it."""
    from ouroboros import review_pause
    from ouroboros.tools.review_change import ReviewChangeArgumentError, run_review_change

    def dispatch() -> Dict[str, Any]:
        return run_review_change(ctx, subject="result", answer=str(content or ""), _frozen_result=subject,
                                 reason="Required: the host's review of a result with produced work", _chosen_by="host")

    try:
        with review_pause.receptive_author(review_pause.receptive_binding(ctx)):
            try:
                outcome = _collect_running(ctx, content, running, collect) if running else dispatch()
            except ReviewChangeArgumentError:
                if not running:
                    raise
                # The round settled (or left custody) after the lookup: its record is read again below,
                # and a Required result that no settled review covers gets the host's own review.
                covered = settled_result_record(ctx, collect.diff_sha) is not None
                outcome = dispatch() if required and not covered and other is None else {"state": "settled"}
    except ReviewChangeArgumentError as exc:
        log.warning("Required result review of %s not dispatched: %s", getattr(ctx, "task_id", ""), exc)
        record_review_decision(llm_trace, "not_dispatched", str(exc), ctx=ctx)
        return False, None
    if outcome.get("state") == "pending":
        ctx._result_review_pending = str(outcome.get("record_id") or "pending")
        if (pending_review_choice_offered(ctx)
                and str(getattr(ctx, "_acceptance_pending_review_choice", "") or "") == "finish"):
            return (_ready_first(ctx, llm_trace, ctx._result_review_pending, hold)
                    or _finish_pending(ctx, llm_trace, ctx._result_review_pending, subject)), None
        ctx._completion_hold_cause = "pending"
        unsent = (outcome.get("continuation") or {}).get("message")
        hold((f"[Result review] The continuation you chose for this review is recorded but was not sent ({unsent}). "
              "Your next delivery sends that same recorded choice; nothing is decided again."
              if unsent else "[Result review] The review of your result is still running (an addressed message or a "
              "seat still answering). Read what reached you; your next delivery collects the same review at $0.")
             + (" You may instead finish with pending_review=finish and a rationale; the reviewers then settle later."
                if pending_review_choice_offered(ctx) else ""))
        return True, None
    ctx._result_review_pending = ""
    return None, settled_result_record(ctx, collect.diff_sha) or (
        settled_result_record(ctx, collect.diff_sha, refusal=True) if outcome.get("record_id") else None)


def result_review_at_delivery(ctx: Any, content: str, llm_trace: Dict[str, Any],
                              hold: Any) -> bool:
    """The result review's one host step at a final delivery; ``True`` holds the delivery for another turn.

    Every mode but Off binds a settled result review of exactly this result into the task's review runs, which the
    outcome axes read. Required adds the floor (owner, 8 Nov 2026): a ROOT result with produced work (a file or
    effect, a host-observed production whose capture failed, an answer its contract requires reviewed, an adapter's
    answer line) that nobody reviewed yet is reviewed by the host (``chosen_by="host"``) through the same wave as a
    mind's own call. An outcome the mind has not received whole, a PASS included, is shown once with every finding
    (``hold(text)``, ``_outcome_received``). Then a PASS delivers; Advisory and Cyber Pro deliver on the mind's next
    decision, its acceptance bound to this exact subject; Blocking holds any review of the delivered result short of
    PASS, whoever called it (its $0 refusal too), until one passes it or the task stops. A revised result: ``_revised``.
    The wave a delivery owes (``_owed_wave``): a review still running, of this result or an earlier version, collected
    at $0 unless a nonblocking install's mind chose ``pending_review=finish``, else Required's floor; a raised ceiling
    never buys an author's own refused review again. When the only review of this exact answer judged an earlier
    request, the new words are asked about once (owner, 18.10). A child is never forced a review; Presence speech is
    never held (``_presence_result_review``). Each delivery records why it reviewed or did not (``record_review_decision``)."""
    from ouroboros.config import get_task_review_mode

    if get_task_review_mode() == "off":
        record_review_decision(llm_trace, "mode_off", ctx=ctx)
        return False
    rows = _result_rows(ctx)
    produced = produced_material(ctx) if get_task_review_mode() == "required" and _is_root(ctx) else None
    required = required_material(ctx, llm_trace, content, produced=produced) if produced is not None else ""
    if not required and not rows:
        record_review_decision(llm_trace, "no_production" if produced is not None else "not_requested", ctx=ctx)
        return False
    from ouroboros.body_candidate import serving_repo_dir_for
    from ouroboros.tools.review_change import ReviewChangeArgumentError

    try:
        subject = result_subject(ctx, SimpleNamespace(answer=""), governance_root=pathlib.Path(serving_repo_dir_for(ctx)),
                                 answer=str(content or ""), produced=produced)
    except ReviewChangeArgumentError as exc:
        record_review_decision(llm_trace, "nothing_to_review", str(exc), ctx=ctx)
        return False
    record_review_decision(llm_trace, required or "review_run", ctx=ctx)
    _supersede_other_result_runs(llm_trace, subject.diff_sha)
    blocking, refused = _blocking_enforcement(), getattr(ctx, "_result_review_refused", None) or {}
    record = settled_result_record(ctx, subject.diff_sha, rows)
    # A $0 refusal of exactly this subject not met yet (the mind's own call; a sibling spent the allowance) is bound below
    # like the host's own. A raised ceiling lets Required's floor run; an author's own review it never buys (root, 9 Nov).
    met = None if record or subject.diff_sha in refused else settled_result_record(ctx, subject.diff_sha, rows, refusal=True)
    record = met if met and (not required or _ceiling(met) == _ceiling()) else record
    if record is None and subject.diff_sha in refused:
        standing = refused[subject.diff_sha]["ceiling"] == _ceiling()
        if _presence_turn_speech(ctx):
            if standing:
                record = settled_result_record(ctx, subject.diff_sha, rows, refusal=True)
        elif not blocking:
            return _after_refusal(ctx, llm_trace, subject.diff_sha)
        elif standing or not required:  # still spent, or an author's own review: held until a review passes it
            ctx._completion_hold_cause = "blocking_refusal"  # each hold names its cause to the delivery owner
            hold(_findings_text(settled_result_record(ctx, subject.diff_sha, rows, refusal=True) or {"verdict": {
                "aggregate": "NOT_DISPATCHED"}}, blocking=True, required=bool(required), raised=not standing))
            return True
    # The only review of this exact answer may have judged an earlier request (the owner's words changed since): one
    # running is collected at $0 under its own subject, a settled one asked about once (owner, 18.10). A review of an
    # earlier version still running is the one duty awaited (never a parallel wave); a settled one is ``_revised``'s.
    running = running_result_review(rows, subject.diff_sha) if record is None else ""
    collect, earlier, other = subject, None, None
    if record is None and not running:
        if (pending := earlier_request_subject(ctx, subject, rows, state="pending")) is not None:
            running, collect = str(pending[0].get("record_id") or ""), pending[1]
        elif (settled := earlier_request_subject(ctx, subject, rows, state="settled")) is not None:
            earlier = settled_result_record(ctx, settled[1].diff_sha, rows)
        elif (in_flight := _other_version(ctx, subject, rows, state="pending")) is not None:
            running, collect = str(in_flight[0].get("record_id") or ""), in_flight[1]
        else:
            other = _other_version(ctx, subject, rows, state="settled")
    if _presence_turn_speech(ctx):  # its speech never waits for an outcome: the floor stands, nothing holds it
        if other is not None and not blocking and _outcome_received(ctx, llm_trace, other):  # Advisory: its author's
            return _revised(ctx, hold, llm_trace, subject, other, blocking=False, required=bool(required), speech=True)
        return _presence_result_review(ctx, content, llm_trace, subject, record or earlier,
                                       owed=bool(required) and not running and earlier is None)
    if running:
        ctx._result_review_pending = running
        if (pending_review_choice_offered(ctx)
                and str(getattr(ctx, "_acceptance_pending_review_choice", "") or "") == "finish"):
            return _ready_first(ctx, llm_trace, running, hold) or _finish_pending(ctx, llm_trace, running, subject)
    if record is None and earlier is None and (required and (other is None or blocking) or running):
        decision, record = _owed_wave(ctx, content, llm_trace, hold, subject, collect, running, required=required, other=other)
        if decision is not None:
            return decision
    record = record or earlier
    if record is not None and str((record.get("subject") or {}).get("tree_sha") or subject.tree_sha) != subject.tree_sha:
        other, record = record, None  # the collected review judged another version of this result
    if record is None:
        return (_revised(ctx, hold, llm_trace, subject, other, blocking=blocking, required=bool(required),
                         content=str(content or "")) if other is not None else False)
    stands_for_earlier = str((record.get("subject") or {}).get("diff_sha") or "") not in {"", subject.diff_sha}
    record_id = str(record.get("record_id") or "")
    # What this step put before the mind is plain JSON (ids, [record, subject] pairs, refusals by subject), so a
    # same-ID continuation carries it in its saved state (``owner_wait.continuation_state``).
    if stands_for_earlier:
        asked = getattr(ctx, "_result_review_request_asked", None) or []
        if [record_id, subject.diff_sha] not in asked:
            # The one question carries that review's outcome too when its author has not received it yet.
            outcome = not _outcome_received(ctx, llm_trace, record)
            text = _request_changed_text(ctx, record, outcome=outcome)
            _show(ctx, hold, text, record, "earlier_request", outcome=outcome)
            ctx._result_review_request_asked = [*asked, [record_id, subject.diff_sha]]
            return True
        # Delivered again: the mind judged the new words leave what was asked as it was.
    run = _bind_review_run(llm_trace, record, subject.diff_sha, "blocking" if blocking else "advisory",
                           earlier_request=stands_for_earlier)
    aggregate = str((record.get("verdict") or {}).get("aggregate") or "")
    received = _outcome_received(ctx, llm_trace, record)
    if aggregate == "PASS":  # a PASS its author has not received is handed back once too (owner, 2 Nov)
        return False if received else _show(ctx, hold, _findings_text(record, blocking=False), record, "pass")
    if aggregate == "NOT_DISPATCHED":
        ctx._result_review_refused = {**(getattr(ctx, "_result_review_refused", None) or {}), subject.diff_sha: {
            "ceiling": _ceiling(record), "record_id": record_id}}
    if not received or blocking:  # Blocking holds any completed review of this result that did not pass it
        return _show(ctx, hold, _findings_text(record, blocking=blocking, required=bool(required)), record, "findings")
    run["informed_finish"] = _acceptance(ctx, subject.diff_sha, record)  # the mind received it and delivered
    return False

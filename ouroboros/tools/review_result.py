"""Result review — ``review_change(subject="result")``: does this task's result do what was asked?

One subject of the one review machinery. This module composes the subject — what was asked, the
reply, what the task produced and the host-attested verification — and the result brief; the wave
itself (the panel, custody, the paid stamp, $0 reuse, a rejoin, leaving on an addressed message,
settlement and the ledger record) is ``review_change``'s own. Every seat answers one question,
judged on the result, and on the method only where the owner's words or explicit choice fixed
it: the work trace is not part of the subject.

The subject is frozen once, at the call: its exact text is retained in the task's artifact store
and its sha is the wave's identity, so a second call over the same reply, material and request
reuses the settled record at $0. Retrieving seats read the retained text and the produced files in
place; packet seats receive the text inline. Produced files are named by path and content hash,
never copied.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

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
_READ_IN_PLACE = ("Every produced file the subject names is at its absolute path: read what you need in place, "
                  "nothing is copied for you.")


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
    from ouroboros.tools.review_multi_model import triad_stable_head

    heads = [_brief_head(goal=words, scope=words, review_rebuttal=words, rubric="⟨rubric⟩") for words in ("⟨words⟩", "")]
    produced = {"files": [{"path": "⟨path⟩", "kind": "⟨kind⟩", "sha256": "⟨sha⟩", "size": 1}], "issues": ["⟨issue⟩"],
                "effects": [{"tool": "⟨tool⟩", "receipt": "⟨receipt⟩", "host_notes": ["⟨note⟩"]}]}
    own = [RESULT_CHECKLIST_SECTION, *heads, _SUBJECT_HEAD, *_ASK_WORDS.values(), *reviewer_words_material(),
           *(answer_text(reply=reply, produced=material, verification="⟨verification⟩")
             for reply, material in (("⟨reply⟩", produced), ("", {})))]
    return (own, [RESULT_USER_TURN, triad_stable_head(RESULT_LAYER, ""), _packet_prompt("⟨head⟩", "⟨subject⟩")],
            [_retrieving_task("⟨head⟩", source, "⟨subject⟩") for source in ("⟨source⟩", None)])


def latest_reply(ctx: Any) -> str:
    """The reply this task would deliver now (its retained delivery candidate), "" when none."""
    return str(getattr(getattr(ctx, "_delivery_candidate", None), "full_text", "") or "")


def _file_identity(path: pathlib.Path) -> Dict[str, Any]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return {"size": path.stat().st_size, "sha256": digest.hexdigest()}
    except FileNotFoundError:
        return {"size": 0, "sha256": "deleted"}
    except OSError as exc:
        return {"size": 0, "sha256": f"unreadable ({type(exc).__name__})"}


def produced_material(ctx: Any) -> Dict[str, Any]:
    """What this task produced: its registered artifacts, its workspace changes against the
    task's own baseline, and the effects an extension tool declared (``observed_effect``).
    Inputs, checkpoints and the expected result a task was given are not production. A capture
    that fails is named as a gap, never filled in."""
    out: Dict[str, Any] = {"files": [], "effects": [], "issues": [], "workspace": ""}
    root, task_id = getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or "")
    if root is None or not task_id:
        return out
    try:
        from ouroboros.artifacts import collect_task_artifact_records, registered_task_artifact

        for row in collect_task_artifact_records(root, task_id, measure=False, strict=True, require_registered=True):
            name = str(row.get("relpath") or row.get("name") or "")
            if not name or not row.get("size") or not registered_task_artifact(root, task_id, name):
                continue
            path = pathlib.Path(str(row["path"]))
            out["files"].append({"path": str(path), "kind": "artifact", **_file_identity(path)})
    except Exception as exc:
        out["issues"].append(f"task artifacts could not be listed ({type(exc).__name__}: {exc})")
    try:
        from ouroboros.mutation_attribution import attributed_git_candidates

        getter = getattr(ctx, "active_repo_dir", None)
        repo = getter() if callable(getter) else getattr(ctx, "repo_dir", None)
        if repo:
            snapshot = attributed_git_candidates(getattr(ctx, "budget_drive_root", None) or root, task_id, repo)
            candidates = [str(c.get("path") if isinstance(c, dict) else c) for c in snapshot.get("candidates") or []]
            for relative in candidates:
                path = pathlib.Path(str(repo)) / relative
                out["files"].append({"path": str(path), "kind": "workspace change", **_file_identity(path)})
            if candidates:
                out["workspace"] = str(pathlib.Path(str(repo)).resolve(strict=False))
            elif snapshot.get("blockers") and snapshot.get("blockers") != ["baseline_missing"]:
                out["issues"].append("workspace changes not attributed: " + ", ".join(map(str, snapshot["blockers"])))
    except Exception as exc:
        out["issues"].append(f"workspace changes could not be attributed ({type(exc).__name__}: {exc})")
    out["effects"] = _declared_effects(ctx, root, task_id, out["issues"])
    return out


def _declared_effects(ctx: Any, root: Any, task_id: str, issues: List[str]) -> List[Dict[str, Any]]:
    """Extension tools' own execution receipts (``PluginAPI.register_tool``): read exactly as the producer
    wrote them, never inferred from a method or backend name. A call whose producer reported its own failure
    (``TOOL_REPORTED_FAILURE``) may still declare what ran before it failed; a host-owned outcome (a timeout,
    a refusal, a raised handler) declares nothing. Host notes trail the payload the model saw, so a noted
    call is read from its retained producer payload, verified, and its notes ride beside the receipt."""
    from ouroboros.artifacts import materialize_tool_result_source, read_actor_source_bytes
    from ouroboros.extension_surface_names import parse_extension_surface_name
    from ouroboros.store import StoreError

    effects: List[Dict[str, Any]] = []
    trace = getattr(ctx, "_execution_trace", None) or {}
    for call in trace.get("tool_calls") or []:
        if (not isinstance(call, dict) or not parse_extension_surface_name(str(call.get("tool") or ""))
                or (call.get("is_error") and call.get("tool_result_code") != "TOOL_REPORTED_FAILURE")):
            continue
        tool = str(call.get("tool") or "")
        try:
            if "producer_source_ref" in call:
                if not call["producer_source_ref"]:
                    issues.append(f"effect receipt of {tool} unavailable (its producer payload was not retained)")
                    continue
                text = read_actor_source_bytes(root, task_id, call["producer_source_ref"]).decode("utf-8")
            else:
                text, complete, gap = materialize_tool_result_source(root, task_id, call)
                if not complete:
                    issues.append(f"effect receipt of {tool} unavailable ({gap})")
                    continue
            payload = json.loads(text) if isinstance(text, str) else {}
        except (OSError, ValueError, TypeError, StoreError) as exc:
            issues.append(f"effect receipt of {tool} unreadable ({type(exc).__name__}: {exc})")
            continue
        if isinstance(payload, dict) and payload.get("observed_effect") is True:
            notes = [str(note) for note in call.get("host_annotations") or []]
            effects.append({"tool": tool, "receipt": text, **({"host_notes": notes} if notes else {})})
    return effects


# How the subject frames what was asked and what was verified, around the host's records of both.
_ASK_WORDS = {"objective": "The objective above: {author}.", "claims": "claims the result is held to ({source}):",
              "support": " (support: {support})", "unavailable": "claims: unavailable ({error})",
              "none": "(the task states no request beyond its reply)",
              "no_receipts": "(no host-attested verification receipts)",
              "unreadable": "(verification receipts unreadable: {error})"}


def _ask(ctx: Any) -> str:
    """What was asked, host-attested: the task's own request fields (the objective with its author,
    ``owner_words.objective_author_line``), the claims it is held to (an ingress contract's, else the
    latest plan revision's) and its owner's words."""
    from ouroboros.owner_words import objective_author_text, owner_words_text
    from ouroboros.review_evidence_sections import _accept_effective_claims, _accept_task_contract

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


def answer_text(*, reply: str, produced: Dict[str, Any], verification: str) -> str:
    """The answer half of a result subject (the reply, what the task produced, the verification): its digest
    is the subject's ``tree_sha``, the answer's own identity apart from what was asked."""
    files = [f"- {item['path']} ({item['kind']}; sha256 {item['sha256']}; {item['size']} bytes)"
             for item in produced.get("files") or []]
    effects = [f"- effect declared by {item['tool']}: {item['receipt']}"
               + (f"\n  host notes on this call: {json.dumps(item['host_notes'], ensure_ascii=False)}"
                  if item.get("host_notes") else "")
               for item in produced.get("effects") or []]
    gaps = [f"- not captured: {issue}" for issue in produced.get("issues") or []]
    return "\n".join([
        "## The reply", "", reply.strip() or "(no reply text)", "",
        "## What the task produced", "",
        *(["Effect receipts are the tool producer's reports of execution, not proof of a UI change or goal success.", ""]
          if effects else []),
        *(files + effects + gaps or ["(nothing produced beyond the reply)"]), "",
        "## Host-attested verification", "", verification, "",
    ])


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
    """The owner's words this run holds now that the earlier subject's request did not carry, verbatim."""
    from ouroboros.owner_words import FIELD, governing_words_for_schedule

    rows = governing_words_for_schedule(ctx).get(FIELD) or []
    texts = [str(row.get("text") or "").strip() for row in rows if isinstance(row, dict)]
    return [text for text in texts if text and text not in earlier] if earlier else []


def _request_changed_text(ctx: Any, record: Dict[str, Any]) -> str:
    """Asked once (owner, 18.10): the only review of this exact answer judged an earlier request."""
    earlier = _retained_subject_text(ctx, str((record.get("subject") or {}).get("diff_sha") or ""))
    words = _words_since(ctx, earlier)
    aggregate = str((record.get("verdict") or {}).get("aggregate") or "")
    lines = [f"[Result review] This exact result was reviewed (record {record.get('record_id')}: {aggregate}) before "
             "what was asked last changed; its reviewers did not see these words of your human:"]
    lines += [f"- {text[:600]}" for text in words[:5]] or ["- (the request changed: its claims or contract, or words "
                                                          "the earlier request cannot be read back for)"]
    if len(words) > 5:
        lines.append(f"- ... and {len(words) - 5} more message(s)")
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
    if not reply.strip() and not produced["files"] and not produced["effects"]:
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
    from ouroboros.review_model_routes import get_review_enforcement
    from ouroboros.reviewer_slot_config import commit_triad_delivery, row_plan_retrieves
    from ouroboros.tools.review_admission import seat_vectors
    from ouroboros.tools.review_helpers import review_drive_root, review_enforcement_blocks
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT

    blocking_review = review_enforcement_blocks(get_review_enforcement())
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
        "session_task": "", "target_repo": str(subject.spec.root), "blocking_review": blocking_review,
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
                      produced: Optional[Dict[str, Any]] = None) -> bool:
    """Whether a result carries produced work the Required floor guarantees a review for: a file or an
    effect the task produced, or the answer line of an adapter's answer protocol (a benchmark's answer
    is its work product). The expected result a task was given is intent, never production."""
    from ouroboros.contracts.task_contract import answer_protocol_active
    from ouroboros.outcomes import selected_final_answer

    produced = produced if produced is not None else produced_material(ctx)
    if produced["files"] or produced["effects"]:
        return True
    return bool(answer_protocol_active(ctx) and selected_final_answer(str(content or ""), llm_trace))


def _result_rows(ctx: Any) -> List[Dict[str, Any]]:
    from ouroboros import review_ledger as ledger

    try:
        return ledger.recent_records(ledger.ledger_root(ctx), str(getattr(ctx, "task_id", "") or ""), limit=50,
                                     hot_only=True, surface="result")
    except Exception:
        log.debug("result records of %s unreadable", getattr(ctx, "task_id", ""), exc_info=True)
        return []


def settled_result_record(ctx: Any, subject_sha: str, rows: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """This task's settled result review of exactly this subject (by the mind or the host), else ``None``."""
    from ouroboros import review_ledger as ledger

    for row in rows if rows is not None else _result_rows(ctx):
        if (str((row.get("subject") or {}).get("diff_sha") or "") == subject_sha and row.get("state") == "settled"
                and str((row.get("verdict") or {}).get("aggregate") or "") != "NOT_DISPATCHED"):
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

    record = ledger.load_record(ledger.ledger_root(ctx), record_id)
    brief = dict(record.get("brief") or {})
    return run_review_change(ctx, subject="result", answer=str(content or ""), goal=str(brief.get("goal") or ""),
                             scope=str(brief.get("scope") or ""),
                             author_questions=list(brief.get("author_questions") or []),
                             reason="Collect the running review of this result", _frozen_result=subject,
                             _rejoin_retry_key=str((record.get("fingerprints") or {}).get("retry_key") or ""))


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
                     enforcement: str, *, earlier_request: bool = False) -> Dict[str, Any]:
    """The result review bound to the delivered answer, in the task's review runs the outcome axes read.
    ``earlier_request``: the mind judged that the owner's words since this review do not change what was
    asked, so the review of this exact answer stands for the current request too."""
    runs = llm_trace.setdefault("review_runs", [])
    record_id = str(record.get("record_id") or "")
    bound = next((run for run in runs if isinstance(run, dict) and run.get("panel_id") == record_id), None)
    if bound is not None:
        if earlier_request:
            bound.update(binding_hash=subject_sha, earlier_request=True)
        if bound.get("binding_hash") == subject_sha:
            bound.pop("superseded_by_revision", None)  # the exact subject it reviewed is current again
        return bound
    verdict = dict(record.get("verdict") or {})
    aggregate = str(verdict.get("aggregate") or "")
    signal = aggregate if aggregate in {"PASS", "FAIL"} else "DEGRADED"
    bound = {"surface": RESULT_KIND, "authority": "host_root", "panel_id": record_id, "record_id": record_id,
             "binding_hash": subject_sha, "aggregate_signal": signal, "aggregate": aggregate,
             "degraded": signal == "DEGRADED", "reason": str(verdict.get("reason") or ""),
             "critical_count": len(verdict.get("critical_findings") or []), "enforcement": enforcement,
             "dispatch_refusal": str((record.get("dispatch_refusal") or {}).get("kind") or ""),
             **({"earlier_request": True} if earlier_request else {})}
    runs.append(bound)
    return bound


def _blocking_enforcement() -> bool:
    """Blocking review enforcement as it applies to a result: Cyber Pro never vetoes."""
    from ouroboros.config import get_runtime_mode
    from ouroboros.review_model_routes import get_review_enforcement
    from ouroboros.runtime_mode_policy import runtime_mode_at_least
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    return review_enforcement_blocks(get_review_enforcement()) and not runtime_mode_at_least(get_runtime_mode(), "cyber_pro")


def pending_review_choice_offered(ctx: Any) -> bool:
    """While a result review of this answer is still running, a nonblocking install lets the mind
    choose ``pending_review=finish``: deliver now, the reviewers settle later."""
    return bool(getattr(ctx, "_result_review_pending", "")) and not _blocking_enforcement()


def _findings_text(record: Dict[str, Any], *, blocking: bool) -> str:
    verdict = dict(record.get("verdict") or {})
    aggregate = str(verdict.get("aggregate") or "")
    lines = [f"[Result review] Your result was reviewed before delivery (record {record.get('record_id')}): {aggregate}."]
    for finding in [*(verdict.get("critical_findings") or []), *(verdict.get("advisory_findings") or [])][:12]:
        lines.append(f"- {finding.get('severity') or 'advisory'} ({finding.get('seat_id') or 'reviewer'}): "
                     f"{finding.get('item') or ''} {finding.get('reason') or ''}".rstrip())
    if aggregate == "NOT_DISPATCHED":
        lines.append(str((record.get("dispatch_refusal") or {}).get("message") or "No review was dispatched."))
    elif aggregate not in {"PASS", "FAIL"}:
        lines.append(f"No verdict was formed ({verdict.get('reason') or 'unknown'}); the record names each seat's state.")
    lines.append(("Blocking enforcement: no review of this result can be bought until your human raises the review "
                  "ceiling, and it is not delivered unreviewed; stop the task explicitly, or wait for that."
                  if aggregate == "NOT_DISPATCHED" else
                  "Blocking enforcement: this result is not delivered until a review passes it; revise the result or "
                  "stop the task explicitly.") if blocking else
                 "You decide: revise the result, or deliver it as it is with your reasons; nothing is delivered "
                 "until you answer again.")
    return "\n".join(lines)


def _ceiling() -> Optional[int]:
    from ouroboros.review_cycles import review_max_cycles

    return review_max_cycles()


def _hold_refused(ctx: Any, record_id: str, hold: Any) -> bool:
    """Blocking with an owed review and the ceiling unchanged since its refusal: the refusal shown stands (read
    from its own record, whatever answer the trace's runs speak for now)."""
    from ouroboros import review_ledger as ledger

    try:
        record = ledger.load_record(ledger.ledger_root(ctx), record_id) if record_id else None
    except ValueError:
        record = None
    ctx._completion_hold_cause = "blocking_refusal"  # each hold names its cause to the delivery owner
    hold(_findings_text(record or {"verdict": {"aggregate": "NOT_DISPATCHED"}, "dispatch_refusal": {}}, blocking=True))
    return True


def _after_refusal(llm_trace: Dict[str, Any], subject_sha: str) -> bool:
    """A $0 refusal of this result's review (its allowance is spent) was already shown, and the mind delivers:
    its own informed decision. (Blocking with an owed review never comes here: it holds while the ceiling is
    unchanged, and a raised ceiling lets the owed review run.)"""
    run = next((item for item in llm_trace.get("review_runs") or [] if isinstance(item, dict)
                and item.get("surface") == RESULT_KIND and item.get("binding_hash") == subject_sha
                and not item.get("superseded_by_revision")), None)
    if run is not None:
        run["informed_finish"] = True
    return False


def _presence_result_review(ctx: Any, content: str, llm_trace: Dict[str, Any], subject: Any,
                            record: Optional[Dict[str, Any]], *, owed: bool) -> bool:
    """Presence speech is never held. A Required review of its result still runs: its author departs at
    once (``review_pause.departed_binding``), nobody waits for the seats, and the outcome settles as a late
    mark in the room the work came from (#1738); a review that settled at once is bound like any other."""
    if record is None and owed:
        from ouroboros import review_pause
        from ouroboros.tools.review_change import ReviewChangeArgumentError, run_review_change

        try:
            with review_pause.receptive_author(review_pause.departed_binding(ctx)):
                run_review_change(ctx, subject="result", answer=str(content or ""), _frozen_result=subject,
                                  reason="Required: the host's review of a Presence result with produced work",
                                  _chosen_by="host")
        except ReviewChangeArgumentError as exc:
            log.warning("Required result review of Presence %s not dispatched: %s", getattr(ctx, "task_id", ""), exc)
            return False
        record = settled_result_record(ctx, subject.diff_sha)
    if record is not None:
        earlier = str((record.get("subject") or {}).get("diff_sha") or "") != subject.diff_sha
        _bind_review_run(llm_trace, record, subject.diff_sha, "blocking" if _blocking_enforcement() else "advisory",
                         earlier_request=earlier)
    return False


def result_review_at_delivery(ctx: Any, content: str, llm_trace: Dict[str, Any],
                              hold: Any) -> bool:
    """The result review's one host step at a final delivery; ``True`` holds the delivery for another turn.

    Every mode but Off binds a settled result review of exactly this result into the task's review runs,
    which the outcome axes read. Required adds the floor: a ROOT result with produced material that nobody
    reviewed yet is reviewed by the host, seating the whole pool with ``chosen_by="host"``, through the same
    wave as a mind's own call. The outcome reaches
    the mind before delivery: a PASS delivers; otherwise its findings are shown once (``hold(text)``), then
    Advisory and Cyber Pro deliver on the mind's next decision while Blocking holds until a review passes
    the result or the task stops. A review of this result the mind started that is still running is
    collected the same way: the delivery waits for it unless a nonblocking install's mind chose
    ``pending_review=finish``. When the only review of this exact answer judged an earlier request (its owner's
    words changed since), the host shows the new words once and asks whether they change what was asked; a
    delivery after that keeps that review for this result (owner, 18.10). A child is never forced; Presence
    speech is never held (its owed review runs with nobody waiting and arrives as a late mark,
    ``_presence_result_review``)."""
    from ouroboros.config import get_task_review_mode

    mode = get_task_review_mode()
    if mode == "off":
        return False
    rows = _result_rows(ctx)
    produced = produced_material(ctx) if mode == "required" and _is_root(ctx) else None
    required = produced is not None and required_material(ctx, llm_trace, content, produced=produced)
    if not required and not rows:
        return False
    from ouroboros.body_candidate import serving_repo_dir_for
    from ouroboros.tools.review_change import ReviewChangeArgumentError, run_review_change

    try:
        subject = result_subject(ctx, SimpleNamespace(answer=""), governance_root=pathlib.Path(serving_repo_dir_for(ctx)),
                                 answer=str(content or ""), produced=produced)
    except ReviewChangeArgumentError:
        return False
    _supersede_other_result_runs(llm_trace, subject.diff_sha)
    record = settled_result_record(ctx, subject.diff_sha, rows)
    refused = getattr(ctx, "_result_review_refused", None) or {}
    if record is None and subject.diff_sha in refused:
        if not (required and _blocking_enforcement()):
            return _after_refusal(llm_trace, subject.diff_sha)
        if refused[subject.diff_sha]["ceiling"] == _ceiling():  # still spent: the refusal shown stands, nothing is written
            return _hold_refused(ctx, refused[subject.diff_sha]["record_id"], hold)
        # A raised ceiling lets the owed review run now (below).
    # The only review of this exact answer may have judged an earlier request (the owner's words changed since):
    # one still running is collected at $0 under its own subject, a settled one is asked about once (owner, 18.10).
    running = running_result_review(rows, subject.diff_sha) if record is None else ""
    collect, earlier = subject, None
    if record is None and not running:
        pending = earlier_request_subject(ctx, subject, rows, state="pending")
        if pending is not None:
            running, collect = str(pending[0].get("record_id") or ""), pending[1]
        elif (settled := earlier_request_subject(ctx, subject, rows, state="settled")) is not None:
            earlier = settled_result_record(ctx, settled[1].diff_sha, rows)
    if getattr(ctx, "_presence_completion", None):
        return _presence_result_review(ctx, content, llm_trace, subject, record or earlier,
                                       owed=required and not running and earlier is None)
    if running:
        ctx._result_review_pending = running
        if (pending_review_choice_offered(ctx)
                and str(getattr(ctx, "_acceptance_pending_review_choice", "") or "") == "finish"):
            return False  # the mind chose to finish; the reviewers settle later and leave a mark
    if record is None and earlier is None and (required or running):
        from ouroboros import review_pause

        def dispatch() -> Dict[str, Any]:
            return run_review_change(ctx, subject="result", answer=str(content or ""), _frozen_result=subject,
                                     reason="Required: the host's review of a result with produced work",
                                     _chosen_by="host")

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
                    outcome = dispatch() if required and not covered else {"state": "settled"}
        except ReviewChangeArgumentError as exc:
            log.warning("Required result review of %s not dispatched: %s", getattr(ctx, "task_id", ""), exc)
            return False
        if outcome.get("state") == "pending":
            ctx._result_review_pending = str(outcome.get("record_id") or "pending")
            if (pending_review_choice_offered(ctx)
                    and str(getattr(ctx, "_acceptance_pending_review_choice", "") or "") == "finish"):
                return False  # the mind chose to finish; the reviewers settle later and leave a mark
            ctx._completion_hold_cause = "pending"
            hold("[Result review] The review of your result is still running (an addressed message or a seat "
                 "still answering). Read what reached you; your next delivery collects the same review at $0."
                 + (" You may instead finish with pending_review=finish; the reviewers then settle later."
                    if pending_review_choice_offered(ctx) else ""))
            return True
        ctx._result_review_pending = ""
        record = settled_result_record(ctx, collect.diff_sha)
        if record is None and outcome.get("record_id"):
            # A $0 refusal (the result reviews' paid-cycle allowance is spent) is bound as it is: no review
            # of this result can run, so it is shown once and never holds the delivery again.
            from ouroboros import review_ledger as ledger

            record = ledger.load_record(ledger.ledger_root(ctx), str(outcome["record_id"]))
    record = record or earlier
    if record is None:
        return False
    stands_for_earlier = str((record.get("subject") or {}).get("diff_sha") or "") not in {"", subject.diff_sha}
    if stands_for_earlier:
        asked = getattr(ctx, "_result_review_request_asked", None)
        if not isinstance(asked, set):
            asked = set()
            ctx._result_review_request_asked = asked
        if (str(record.get("record_id") or ""), subject.diff_sha) not in asked:
            asked.add((str(record.get("record_id") or ""), subject.diff_sha))
            ctx._completion_hold_cause = "earlier_request"
            hold(_request_changed_text(ctx, record))
            return True
        # Delivered again: the mind judged the new words leave what was asked as it was.
    enforcement = "blocking" if _blocking_enforcement() else "advisory"
    run = _bind_review_run(llm_trace, record, subject.diff_sha, enforcement, earlier_request=stands_for_earlier)
    if str((record.get("verdict") or {}).get("aggregate") or "") == "PASS":
        return False
    refused = str((record.get("verdict") or {}).get("aggregate") or "") == "NOT_DISPATCHED"
    if refused:
        ctx._result_review_refused = {**(getattr(ctx, "_result_review_refused", None) or {}), subject.diff_sha: {
            "ceiling": _ceiling(), "record_id": str(record.get("record_id") or "")}}
    blocking = required and enforcement == "blocking"
    shown = getattr(ctx, "_result_review_shown", None)
    if not isinstance(shown, set):
        shown = set()
        ctx._result_review_shown = shown
    if str(record.get("record_id") or "") not in shown or blocking:
        shown.add(str(record.get("record_id") or ""))
        ctx._completion_hold_cause = "findings"
        hold(_findings_text(record, blocking=blocking))
        return True
    if enforcement == "advisory":
        run["informed_finish"] = True  # the mind read the findings and delivered: its own decision
    return False

"""Selected author plans in the existing immutable task-source graph.

The hot state keeps one head. Each selection retains its actual stance and
predecessor, independently of which critic wave a later attempt selects.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from ouroboros.review_records import validate_author_disposition


def _read(drive_root: Any, task_id: str, ref: dict) -> dict:
    from ouroboros.artifacts import read_actor_source_bytes
    value = json.loads(read_actor_source_bytes(drive_root, task_id, ref))
    if (not isinstance(value, dict) or value.get("kind") != "plan_author_subject"
            or (not isinstance(value.get("spec"), dict) and not value.get("source_unavailable"))):
        raise ValueError("author plan source identity mismatch")
    return value


def _persist(drive_root: Any, task_id: str, value: dict) -> dict:
    from ouroboros.artifacts import store_actor_source_bytes
    return store_actor_source_bytes(drive_root, task_id, category="context_checkpoints",
        source_id=f"plan-author-{value['fingerprint']}", extension="json",
        data=json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def retain_author_selection(drive_root: Any, task_id: str, state: dict, subject: dict, fingerprint: str) -> dict:
    """Retain the exact available source plus its recorded stance before overwrite.

    Also upgrades a still-reachable legacy selection at the next existing write
    seam. No directory search can recover an already lost legacy stance.
    """
    ref = subject["source_ref"]
    head = state.get("author_history_head") or {}
    if head == ref:
        return copy.deepcopy(subject)
    value = _read(drive_root, task_id, ref)
    author = validate_author_disposition(subject.get("author_disposition"), subject_hash=fingerprint)
    if value.get("fingerprint") != fingerprint or author is None:
        raise ValueError("author selection does not match its source")
    value.update(author_disposition=copy.deepcopy(subject["author_disposition"]),
                 review_fingerprint=subject["review_fingerprint"], selected_source_ref=ref,
                 previous_author_subject=head)
    if "review_wave_artifact" in subject:
        value["review_wave_artifact"] = copy.deepcopy(subject["review_wave_artifact"])
    saved = _persist(drive_root, task_id, value)
    state["author_history_head"] = saved
    return {**copy.deepcopy(subject), "source_ref": saved}


def retain_previous_selection(drive_root: Any, task_id: str, state: dict) -> None:
    """Keep a legacy current selection before another attempt or wave replaces it."""
    attempt = state.get("current_attempt") or {}
    if attempt.get("author_subject"):
        subject = attempt["author_subject"]
        try:
            _read(drive_root, task_id, subject["source_ref"])
            attempt["author_subject"] = retain_author_selection(
                drive_root, task_id, state, subject, attempt["fingerprint"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            # A lost historical plan cannot prevent a NEW attempt or erase the
            # stance still available in hot state. Retain that fact, no fake spec.
            value = {"kind": "plan_author_subject", "fingerprint": attempt["fingerprint"],
                     "review_fingerprint": subject["review_fingerprint"],
                     "author_disposition": copy.deepcopy(subject["author_disposition"]),
                     "selected_source_ref": subject["source_ref"],
                     "previous_author_subject": state.get("author_history_head") or {},
                     "source_unavailable": f"selected plan source unavailable ({type(exc).__name__})"}
            if "review_wave_artifact" in subject:
                value["review_wave_artifact"] = subject["review_wave_artifact"]
            state["author_history_head"] = _persist(drive_root, task_id, value)


def record_attempt(drive_root: Any, task_id: str, *, fingerprint: str, status: str, reason: str,
                   author_subject: dict | None, submitted_subject: dict | None = None) -> dict:
    """The existing locked attempt producer, with author-source retention first."""
    from ouroboros import task_results as results

    if not results._PLAN_REVIEW_HASH_RE.fullmatch(str(fingerprint or "")):
        raise ValueError("PLAN_REVIEW_STATE_INVALID: current attempt fingerprint is invalid")
    if status not in results._PLAN_REVIEW_ATTEMPT_STATUSES:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: current attempt status is invalid")

    submitted = None
    if submitted_subject is not None:
        try:
            submitted = _persist(drive_root, task_id, {"kind": "plan_submitted_subject", "fingerprint": fingerprint,
                "spec": submitted_subject["spec"], "plan_prose": submitted_subject["plan_prose"]})
        except (OSError, ValueError):
            # A failed new source must not leave an old closed plan current.
            record_attempt(drive_root, task_id, fingerprint=fingerprint, status="unavailable",
                           reason="submitted_source_unavailable", author_subject=None)
            raise

    def record(state: dict) -> dict:
        previous = state.get("current_attempt") or {}
        retained = submitted or (previous.get("submitted_subject") if previous.get("fingerprint") == fingerprint else None)
        retain_previous_selection(drive_root, task_id, state)
        selected = retain_author_selection(drive_root, task_id, state, author_subject, fingerprint) if author_subject is not None else None
        state["current_attempt"] = {"fingerprint": fingerprint, "status": status,
                                    "reason": str(reason or "")[:results._PLAN_REVIEW_REASON_MAX_CHARS]}
        if selected is not None:
            state["current_attempt"]["author_subject"] = selected
        elif retained:
            state["current_attempt"]["submitted_subject"] = retained
        return state

    return results._update_plan_review_state(drive_root, task_id, record)


def author_selections(drive_root: Any, task_id: str, state: dict) -> tuple[list, list]:
    """Read distinct immutable selections oldest first; source loss is a named gap."""
    head = state.get("author_history_head") or {}
    attempt = state.get("current_attempt") or {}
    selected = attempt.get("author_subject") or {}
    current_ref = selected.get("source_ref") or {}
    refs = [ref for ref in (current_ref, head) if ref]
    seen, selections, gaps = set(), [], []
    for root in refs:
        ref, visiting, chain = root, set(), []
        while ref:
            if not isinstance(ref, dict):
                gaps.append({"source_ref": {}, "reason": "author selection source reference is malformed"})
                break
            identity = (ref.get("sha256"), ref.get("path"))
            if identity in visiting:
                gaps.append({"source_ref": ref, "reason": "author selection chain contains a cycle"})
                break
            if identity in seen:
                break
            visiting.add(identity)
            try:
                value = _read(drive_root, task_id, ref)
                if ref == current_ref:  # a genuine, still-selected legacy source has no embedded stance
                    value.update(author_disposition=selected.get("author_disposition"),
                                 review_fingerprint=selected.get("review_fingerprint"))
                if validate_author_disposition(value.get("author_disposition"), subject_hash=value.get("fingerprint")) is None:
                    raise ValueError("author source has no bound recorded stance")
                if "review_wave_artifact" not in value:
                    gaps.append({"source_ref": ref, "reason": "legacy author selection has no exact critic wave binding"})
                if value.get("source_unavailable"):
                    gaps.append({"source_ref": value.get("selected_source_ref") or ref, "reason": value["source_unavailable"]})
                chain.append({**value, "source_ref": ref})
                ref = value.get("previous_author_subject") or {}
            except (OSError, KeyError, TypeError, ValueError) as exc:
                gaps.append({"source_ref": ref, "reason": str(exc)})
                if ref == current_ref and validate_author_disposition(
                        selected.get("author_disposition"), subject_hash=attempt.get("fingerprint")) is not None:
                    chain.append({"kind": "plan_author_subject", "fingerprint": attempt["fingerprint"],
                        "review_fingerprint": selected["review_fingerprint"], "source_ref": ref,
                        "author_disposition": copy.deepcopy(selected["author_disposition"]),
                        **({"review_wave_artifact": selected["review_wave_artifact"]} if "review_wave_artifact" in selected else {}),
                        "source_unavailable": "selected plan source unavailable"})
                break
        seen.update(visiting)
        selections.extend(reversed(chain))
    distinct, identities = [], set()
    for selection in selections:
        original = selection.get("selected_source_ref") or selection["source_ref"]
        identity = (original.get("sha256"), original.get("path"), json.dumps(selection["author_disposition"], sort_keys=True))
        if identity not in identities:
            identities.add(identity)
            distinct.append(selection)
    return distinct, gaps


def current_submitted_plan(drive_root: Any, task_id: str, state: dict) -> dict | None:
    """Exact current input even when no reviewer could start; no stance or verdict."""
    from ouroboros.artifacts import read_actor_source_bytes
    attempt = state.get("current_attempt") or {}
    ref = attempt.get("submitted_subject")
    if not ref:
        return None
    value = json.loads(read_actor_source_bytes(drive_root, task_id, ref))
    if (value.get("kind") != "plan_submitted_subject" or value.get("fingerprint") != attempt.get("fingerprint")
            or not isinstance(value.get("spec"), dict) or not isinstance(value.get("plan_prose"), str)):
        raise ValueError("submitted plan source identity mismatch")
    return {**value, "source_ref": ref, "authority": "Submitted for review; no reviewer verdict or author-finish selection."}


def address_runtime_review_mirrors(runtime: dict, history: dict) -> None:
    """De-duplicate known model-only authority carriers against the resident index.

    The source state/gate operands are not modified. This runs before the Runtime
    prefix freezes, so a later authored index view cannot leave a hidden raw copy.
    Unknown identities stay full. Only documented authority-envelope edges recur.
    """
    from ouroboros.review_history_view import _sha, _decision_ref, decision_entries, _decision_alias, _set_existing
    from ouroboros.tools.plan_review_artifacts import plan_decision_aliases

    authors = {_sha(_decision_ref(e["row"]["source"])): e for e in decision_entries(history)
               if e["bound_decision"] and e["row"]["decision_kind"] == "plan_author"}

    def subject(value):
        if not isinstance(value, dict):
            return
        ref = _decision_ref(value.get("source_ref") or value.get("source"))
        entry = authors.get(_sha(ref)) if ref else None
        author = value.get("author_disposition")
        if entry and isinstance(author, dict) and author.get("rationale") == entry["row"].get("reason"):
            author["rationale"] = _decision_alias(entry["bound_decision"])

    def wave_mirrors(value, ref):
        if not isinstance(value, dict) or not ref:
            return
        facade = {"rounds": [{**copy.deepcopy(value), "source": {"source_ref": ref}}]}
        for entry in decision_entries(history):
            if entry["bound_decision"] and _decision_ref(entry["row"].get("source")) == ref:
                for path, alias in plan_decision_aliases(facade, entry):
                    _set_existing(value, path[2:], alias)

    def state(value):
        if not isinstance(value, dict):
            return
        subject((value.get("current_attempt") or {}).get("author_subject"))
        waves = value.get("waves") or []
        for wave in waves:
            if isinstance(wave, dict):
                wave_mirrors(wave, _decision_ref(wave.get("wave_artifact")))
        core = value.get("decision_core")
        if isinstance(core, dict) and waves:
            facade = {"author_disposition": copy.deepcopy(core.get("author_disposition"))}
            for key in ("findings", "dispositions"):
                if isinstance(core.get(key), dict):
                    facade[key] = copy.deepcopy(core[key].get("items") or [])
            wave_mirrors(facade, _decision_ref(waves[-1].get("wave_artifact")))
            for key in facade:
                if key in ("findings", "dispositions"):
                    core[key]["items"] = facade[key]
                elif key in core:
                    core[key] = facade[key]

    plan = runtime.get("plan_review_authority")
    if isinstance(plan, dict):
        subject((plan.get("current_attempt") or {}).get("author_subject"))
    # Continuation carriers are typed by their owning authority projector, not
    # arbitrary user JSON or a recursive sweep for a field named rationale.
    pending = [runtime]
    while pending:
        carrier = pending.pop()
        state(carrier.get("plan_review_state"))
        for key in ("predecessor_authority", "task_contract"):
            child = carrier.get(key)
            if isinstance(child, dict):
                pending.append(child)

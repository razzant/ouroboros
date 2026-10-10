"""Defensive continuation views for Main and delegated helpers.

Root startup envelopes and exact task-result reads stay complete. Main's provider
copy substitutes authored narratives or source-resolvable gaps for oversized raw
answers. Helpers carry the predecessor's answer, contract core and owner words,
with its fingerprint, read source and sized omissions; declared inputs keep only
the reference. Neither projection mutates canonical authority.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any, Dict, MutableSet, Optional

from ouroboros.context_budget import PREDECESSOR_RESULT_INLINE_CHARS

_RAW_AUTHORITY_KEYS = frozenset({"result", "final_answer"})


def project_helper_predecessor_authority(
    authority: Any, *, declared: bool = False,
) -> Dict[str, Any]:
    """Pure, idempotent brief of an envelope, or its declared-input reference.

    The existing contract normalizer handles legacy bodies. Keep its envelope
    kind: a bounded preview's wrapper can exceed the per-field wire allowance,
    and relabelling it would make a rebuild collapse the preview and its digest
    again. Answers/core fields keep that producer's whole-or-pointer semantics.
    Omission sizes count serialized characters (without a string's quotes);
    old name-only omissions have unknown sizes, represented by null.

    A delegated child's contract applies it after the parent contract spread, and
    direct work-order sessions apply it to their own contract, so both receive the
    same brief. The brief does not carry the omitted evidence; its source names the
    reader for the full result.
    """
    if not isinstance(authority, Mapping) or not authority:
        return {}
    from ouroboros.contracts.task_contract import build_task_contract

    envelope = build_task_contract({"predecessor_authority": authority})["predecessor_authority"]
    keep = {"kind", "source", "task_id", "authority_sha256", "authority_chars", "digest_semantics"}
    if not declared:
        keep.update({"status", "execution_status", "reason_code", "terminal_origin", "outcome_axes",
                     "result", "task_contract", "origin_message_text", "origin_message_ref"})
    prior = envelope.get("omitted_fields")
    omitted = {}
    if isinstance(prior, Mapping):
        omitted = copy.deepcopy(dict(prior))
    elif isinstance(prior, list):
        omitted = {key: None for key in prior if isinstance(key, str)}
    for key, value in envelope.items():
        if key not in keep and key != "omitted_fields":
            omitted[key] = len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)) - (
                2 if isinstance(value, str) else 0)
    return {
        **{key: copy.deepcopy(value) for key, value in envelope.items() if key in keep},
        "omitted_fields": dict(sorted(omitted.items())),
    }


def _canonical_result_ref(task_id: str) -> Dict[str, Any]:
    return {"kind": "task_result", "task_id": str(task_id or ""), "reader": "get_task_result"}


def _task_result_source(node: Mapping[str, Any], task_id: str) -> Dict[str, Any]:
    source = node.get("source")
    if isinstance(source, Mapping) and source:
        return copy.deepcopy(dict(source))
    return {
        "kind": "task_result",
        "task_id": str(task_id or ""),
        "reader": "get_task_result",
        "arguments": {
            "task_id": str(task_id or ""),
            "include_authority": True,
        },
    }


def _authority_identity(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    source = value.get("source")
    if isinstance(source, Mapping):
        source_id = str(source.get("task_id") or "").strip()
        if source_id:
            return source_id
        arguments = source.get("arguments")
        if isinstance(arguments, Mapping):
            source_id = str(arguments.get("task_id") or "").strip()
            if source_id:
                return source_id
    return str(value.get("task_id") or "").strip()


def _narrative_for(
    node: Mapping[str, Any], task_id: str, drive_root: Any,
) -> Optional[Dict[str, Any]]:
    from ouroboros.project_dialogue import continuation_narrative_is_valid

    candidate = node.get("continuation_narrative")
    if continuation_narrative_is_valid(candidate, task_id):
        return copy.deepcopy(dict(candidate))
    try:
        from ouroboros.project_dialogue import resolve_legacy_continuation_narrative

        legacy = resolve_legacy_continuation_narrative(
            drive_root, task_id, _canonical_result_ref(task_id),
        )
        if isinstance(legacy, dict) and continuation_narrative_is_valid(legacy, task_id):
            return copy.deepcopy(legacy)
    except Exception:
        # A missing or malformed legacy row is represented by the caller's
        # typed gap.  Main context assembly must not become a second writer.
        return None
    return None


def _narrative_value(
    raw: str,
    *,
    task_id: str,
    node: Mapping[str, Any],
    drive_root: Any,
    seen_narratives: MutableSet[str],
) -> Dict[str, Any]:
    source = _task_result_source(node, task_id)
    narrative = _narrative_for(node, task_id, drive_root)
    narrative_id = f"task-narrative:{task_id}"
    base = {
        "raw_result_resident": False,
        "original_chars": len(raw),
        "omitted_chars": len(raw),
        "source": copy.deepcopy(source),
    }
    if not narrative:
        return {
            **base,
            "status": "unavailable",
            "narrative_status": "unavailable",
            "narrative_gap": {
                "kind": "continuation_narrative_unavailable",
                "reason": "no_exact_authored_summary",
            },
        }
    if narrative_id in seen_narratives:
        return {
            **base,
            "status": "available",
            "narrative_status": "duplicate_reference",
            "narrative_ref": {
                "summary_id": narrative_id,
                "task_id": task_id,
            },
        }
    seen_narratives.add(narrative_id)
    return {
        **base,
        "status": "available",
        "narrative_status": "available",
        "narrative": copy.deepcopy(narrative),
    }


def _project_value(
    value: Any,
    *,
    key: str = "",
    task_id: str = "",
    authority_node: Optional[Mapping[str, Any]] = None,
    drive_root: Any = None,
    seen_narratives: MutableSet[str],
) -> Any:
    if key in _RAW_AUTHORITY_KEYS and isinstance(value, str) and len(value) > PREDECESSOR_RESULT_INLINE_CHARS:
        return _narrative_value(
            value,
            task_id=task_id,
            node=authority_node or {},
            drive_root=drive_root,
            seen_narratives=seen_narratives,
        )
    if isinstance(value, Mapping):
        current_id = str(value.get("task_id") or task_id or "").strip()
        is_authority = key == "predecessor_authority" or (
            "source" in value and "task_id" in value and "task_contract" in value
        )
        if is_authority:
            return _project_authority_node(
                value,
                task_id=current_id,
                drive_root=drive_root,
                seen_narratives=seen_narratives,
            )
        return {
            copy.deepcopy(k): _project_value(
                child,
                key=str(k),
                task_id=current_id,
                authority_node=authority_node,
                drive_root=drive_root,
                seen_narratives=seen_narratives,
            )
            for k, child in value.items()
        }
    if isinstance(value, list):
        return [
            _project_value(
                child,
                task_id=task_id,
                authority_node=authority_node,
                drive_root=drive_root,
                seen_narratives=seen_narratives,
            )
            for child in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _project_value(
                child,
                task_id=task_id,
                authority_node=authority_node,
                drive_root=drive_root,
                seen_narratives=seen_narratives,
            )
            for child in value
        )
    return copy.deepcopy(value)


def _project_authority_node(
    node: Mapping[str, Any],
    *,
    task_id: str,
    drive_root: Any,
    seen_narratives: MutableSet[str],
) -> Dict[str, Any]:
    current_id = str(node.get("task_id") or task_id or "").strip()
    projected = {
        copy.deepcopy(key): _project_value(
            value,
            key=str(key),
            task_id=current_id,
            authority_node=node,
            drive_root=drive_root,
            seen_narratives=seen_narratives,
        )
        for key, value in node.items()
    }
    # A materialized predecessor is already exposed once at the provider
    # projection's top level.  Remove only an equal nested copy; a different
    # identity is the legitimate second hop and remains available.
    contract = projected.get("task_contract")
    nested = contract.get("predecessor_authority") if isinstance(contract, dict) else None
    own = projected.get("predecessor_authority")
    if isinstance(contract, dict) and isinstance(nested, Mapping) and isinstance(own, Mapping):
        if _authority_identity(nested) == _authority_identity(own):
            # Keep the second hop in the complete contract, but avoid carrying
            # a second copy of it beside that contract inside the provider view.
            projected.pop("predecessor_authority", None)
    return projected


def project_main_task_authority(
    task: Mapping[str, Any], *, drive_root: Any = None,
) -> Dict[str, Any]:
    """Build the provider-only authority section without mutating ``task``."""
    seen_narratives: set[str] = set()
    projection: Dict[str, Any] = {}
    task_id = str(task.get("id") or task.get("task_id") or "").strip()
    predecessor = task.get("predecessor_authority")
    contract = task.get("task_contract")
    if isinstance(predecessor, Mapping) and predecessor:
        projection["predecessor_authority"] = _project_authority_node(
            predecessor,
            task_id=str(predecessor.get("task_id") or "").strip(),
            drive_root=drive_root,
            seen_narratives=seen_narratives,
        )
    if isinstance(contract, Mapping):
        projected_contract = _project_value(
            contract,
            task_id=task_id,
            authority_node=predecessor if isinstance(predecessor, Mapping) else {},
            drive_root=drive_root,
            seen_narratives=seen_narratives,
        )
        nested = projected_contract.get("predecessor_authority") if isinstance(projected_contract, dict) else None
        if isinstance(projected_contract, dict) and isinstance(nested, Mapping) and isinstance(predecessor, Mapping):
            if _authority_identity(nested) == _authority_identity(predecessor):
                projected_contract.pop("predecessor_authority", None)
        projection["task_contract"] = projected_contract
    origin_ref = task.get("origin_message_ref")
    origin_text = task.get("origin_message_text")
    if isinstance(origin_ref, Mapping) and origin_ref:
        projection["task_authority_origin"] = {
            "ref": copy.deepcopy(dict(origin_ref)),
            **({"text": str(origin_text)} if isinstance(origin_text, str) and origin_text else {}),
        }
    if isinstance(task.get("authority_historical_gaps"), list):
        projection["authority_historical_gaps"] = copy.deepcopy(task["authority_historical_gaps"])
    return projection


__all__ = ["project_main_task_authority", "project_helper_predecessor_authority"]


def project_predecessor_review_views(runtime: dict, *, drive_root: Any, repo_roots: tuple = ()) -> dict:
    """Materialize an explicitly named predecessor's selected account as history.

    This changes only Main's provider copy. No prior finish/verdict/selection is
    installed on the successor, and no task or history directory is enumerated.
    """
    from types import SimpleNamespace
    from ouroboros import review_history_view as view
    from ouroboros.agent_startup_checks import valid_task_result_authority_source
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.review_history import review_dispute_history
    from ouroboros.task_results import load_task_result

    if not (runtime.get("predecessor_authority") or (runtime.get("task_contract") or {}).get("predecessor_authority")):
        return runtime
    result, pending, seen = copy.deepcopy(runtime), [], set()
    if isinstance(result.get("predecessor_authority"), dict):
        pending.append((result["predecessor_authority"], False))
    contract = result.get("task_contract") or {}
    if isinstance(contract.get("predecessor_authority"), dict):
        pending.append((contract["predecessor_authority"], False))
    for node, inherited in pending:  # Follow only already-admitted, exact predecessor edges.
        source = node.get("source") or {}
        owner = str(source.get("task_id") or "")
        if owner in seen or not valid_task_result_authority_source(source, owner):
            continue
        seen.add(owner)
        try:
            saved = load_task_result(drive_root, owner, strict=True) or {}
            previous = (saved.get("task_contract") or {}).get("predecessor_authority") or {}
            previous_source = previous.get("source") or {}
            previous_id = str(previous_source.get("task_id") or "")
            if previous_id not in seen and valid_task_result_authority_source(previous_source, previous_id):
                # A second Continue need not rewrite the first actor's note.
                # The existing canonical contract supplies this link, not a scan.
                prior = {"task_id": previous_id, "source": copy.deepcopy(previous_source)}
                node["previous_review_context"] = prior
                pending.append((prior, True))
            selection = saved.get(view.SELECTED_VIEW_FIELD)
            if not selection:
                continue
            reader = lambda ref: read_actor_source_bytes(drive_root, owner, ref)
            selected = view.load_review_history_view(selection, reader)
            ctx = SimpleNamespace(drive_root=drive_root, task_id=owner)
            history, operative = view.current_plan_history(ctx)
            histories = [("plan", history, operative)] if history.get("rounds") or history.get("decision_rows") else []
            for repo in dict.fromkeys(str(p) for p in (saved.get("workspace_root"), *repo_roots) if p):
                history = review_dispute_history(drive_root=drive_root, repo_root=repo, task_id=owner)
                if history.get("rounds") or history.get("gaps"):
                    histories.append(("commit", history, None))
            capsules = view.selected_actor_capsules(selected)
            refs = [selection["source_ref"], *(view._actor_capsule(c).get("checkpoint_ref") for c in capsules)]
            contexts = []
            for family, history, operative in histories:
                projected = view.project_review_history(history, operative_subject=operative,
                    selection=selection, source_reader=reader)
                contexts.append({"family": family, "index": projected["mandatory"], "bodies": projected["bodies"],
                                 "selection_status": projected["selection_status"], "gaps": projected["selection_gaps"]})
                refs.extend(e["bound_decision"]["source_ref"] for e in view.decision_entries(history) if e["bound_decision"])
                refs.extend(b["binding"]["source_ref"] for b in view.split_review_history(history, operative_subject=operative)["bodies"])
                refs.extend(e["binding"]["source_ref"] for e in view.attachment_bindings(history))
                if family == "plan":
                    view._address_runtime_review_mirrors({"predecessor_authority": node}, history)
            source_reads, gaps = _predecessor_review_readers(drive_root, owner, refs, inherited=inherited)
            account = "\n\n".join(c["content"][0]["text"] for c in capsules)
            historical = {"task_id": owner, "source": copy.deepcopy(source), "contexts": contexts,
                "authored_account": {"text": account, "source_ref": selection["source_ref"], "authorship": "predecessor_actor"},
                "source_reads": source_reads, "source_gaps": gaps,
                "rule": "Historical account of the named predecessor, not this task's plan, finish, verdict or owner instruction. Source identities belong to that earlier task; use the explicit absolute readers below from this task."}
            node["historical_review_context"] = historical
            addressed = _readdress_predecessor_sources(node, source_reads)
            if "previous_review_context" in node:
                addressed["previous_review_context"] = node["previous_review_context"]
            node.update(addressed)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            node["historical_review_context"] = {"task_id": owner, "source": copy.deepcopy(source),
                "status": "source_unavailable", "gap": type(exc).__name__ + ": " + str(exc),
                "rule": "The earlier selected account could not be read; no current task authority was inferred."}
    return result


def _predecessor_review_readers(root: Any, owner: str, refs: list, *, inherited: bool = False) -> tuple[list, list]:
    """Use the existing exact-source reader and retained-root locations."""
    from pathlib import Path
    from ouroboros.acceptance_history import historical_source_reference
    from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
    from ouroboros.review_history_view import _immutable_ref, _sha
    from ouroboros.source_retention import retained_task_roots

    reads, gaps, seen = [], [], set()
    roots = [Path(root), *retained_task_roots(Path(root), owner)]
    for ref in refs:
        identity = _immutable_ref(ref)
        if not identity or _sha(identity) in seen:
            continue
        seen.add(_sha(identity))
        for source_root in roots:
            path = task_artifact_dir_path(source_root, owner, create=False) / ref["path"]
            if not path.is_file():
                continue
            try:
                read_actor_source_bytes(source_root, owner, ref)
                addressed = historical_source_reference(source_root, owner, ref)
                # Existing lineage reads permit the named predecessor's absolute
                # artifact path, including from a child's different data root.
                if not inherited:
                    addressed["reader"]["arguments"]["root"] = "artifact_store"
                # More distant named predecessors use the helper's existing
                # canonical runtime_data address; they are not direct lineage.
                reads.append({"task_id": owner, "source_ref": identity, "read": addressed["reader"]})
                break
            except (OSError, ValueError, TypeError):
                continue
        else:
            gaps.append({"source_ref": identity, "reason": "predecessor_source_unavailable"})
    return reads, gaps


def _readdress_predecessor_sources(value: Any, readers: list) -> Any:
    """Change reader metadata for exact verified source identities only.

    Text is never parsed or changed. Matching uses the complete typed immutable
    reference, not field names or a path-shaped string in a plan/attachment.
    """
    from ouroboros.review_history_view import _decision_ref, _sha

    by_ref = {_sha(row["source_ref"]): row["read"] for row in readers}
    def visit(item):
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, dict):
            return item
        copied = {key: visit(child) for key, child in item.items()}
        ref = _decision_ref(item)
        read = by_ref.get(_sha(ref)) if ref else None
        if read:
            copied["read"] = copy.deepcopy(read)
            if "reader" in copied:
                copied["reader"] = copy.deepcopy(read)
            if "file" in copied:
                copied["file"] = read["arguments"]["path"]
        return copied
    return visit(value)

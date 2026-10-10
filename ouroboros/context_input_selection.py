"""Task input selection, declared-source composition and historical exposure.

The context facade captures governance once and supplies its runtime/user
renderers. This leaf decides task/document eligibility and builds the selected
core without opening the automatic shared-memory channels. Usable-response
observations retain the first/latest selected inputs as evidence for evaluators,
without changing current task authority or opening today's memory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import pathlib
from typing import Any, Callable, Dict, Optional

from ouroboros.context_fit import ContextCore as _ContextCore
from ouroboros.context_runtime_facts import snapshot_labelled
from ouroboros.contracts.task_contract import normalize_bool
from ouroboros.memory import Memory

log = logging.getLogger("ouroboros.context")

_HISTORICAL_INPUTS = "historical_author_inputs"
_HISTORICAL_COVERAGE = (
    "First and latest usable author inputs only; intermediate views remain in call observability. "
    "Selected logical messages and the recorded physical projection are distinct. Declared or excluded "
    "inputs are not thereby seen. Historical evidence is not current requirements or owner authority. "
    "The host attests capture, not the truth or authority of quoted model, tool or external content. "
    "A source handle is availability, not a reader receipt; packet-only readers see only the labelled preview."
)


def _historical_roots(ctx, drive_root=None):
    return list(dict.fromkeys(pathlib.Path(root) for root in (
        getattr(ctx, "budget_drive_root", None), drive_root, getattr(ctx, "drive_root", None),
    ) if isinstance(root, (str, pathlib.Path)) and str(root)))


def _historical_state(ctx, roots, task_id):
    """Restore only this task's recorded anchors; never look up its room or predecessor."""
    from ouroboros.task_results import load_task_result

    state = getattr(ctx, "_historical_author_inputs", None)
    if isinstance(state, dict) and state.get("task_id") == task_id:
        return state
    for root in roots:
        try:
            row = load_task_result(root, task_id, strict=True) or {}
        except (OSError, ValueError):
            return {"version": 1, "task_id": task_id, "status": "unavailable",
                    "coverage": _HISTORICAL_COVERAGE, "anchors": [], "reason": "record_unavailable"}
        saved = (row.get("review_evidence") or {}).get(_HISTORICAL_INPUTS)
        if isinstance(saved, dict):
            return copy.deepcopy(saved)
    return {"version": 1, "task_id": task_id, "status": "unavailable",
            "coverage": _HISTORICAL_COVERAGE, "anchors": [], "reason": "no_usable_input_captured"}


def _historical_anchor(ctx, observation, roots, task_id):
    """One plain, redacted source, copied into the existing reader stores before publication."""
    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
    from ouroboros.observability import read_blob_ref, read_call_manifest_ref, redact_projection
    from ouroboros.utils import truncate_within_limit

    state = getattr(ctx, "_historical_author_inputs", {}) or {}
    unpublished = None
    for anchor in state.get("anchors", []):
        if (anchor.get("observed_view_revision") == observation.get("revision")
                and anchor.get("physical_attempt_id") == observation.get("physical_attempt_id")):
            if anchor.get("source_ref"):
                return copy.deepcopy(anchor)  # Published bytes are checked, never reconstructed.
            unpublished = anchor  # Retry only this still-retained, never-published observation.
    payload = {"version": 1, "kind": "historical_author_input", "task_id": task_id,
               "task_attempt": observation.get("historical_task_attempt"),
               "observed_at": observation.get("historical_observed_at"),
               "selected_messages": observation["messages"],
               "selected_view_revision": observation.get("revision"),
               "presence_origin": observation.get("historical_presence_origin"),
               "coverage": _HISTORICAL_COVERAGE, "physical_source_status": "unavailable"}
    if observation.get("physical_source_status") == "observed_projection":
        for root in roots:
            try:
                ref = observation["physical_source_ref"]
                manifest = read_call_manifest_ref(root, ref, task_id=task_id)
                physical = read_blob_ref(root, manifest["full_payload_ref"])["messages"]
                if not isinstance(physical, list):
                    continue
                payload.update(physical_messages=physical, physical_source_status="observed_projection",
                               physical_source_identity=ref,
                               physical_projection_seal=manifest.get("model_send_seal"),
                               physical_attempt_id=observation.get("physical_attempt_id"))
                break
            except (OSError, ValueError, KeyError, TypeError):
                continue
    redacted = redact_projection(payload)
    # Call identity differs even for the same selected view. It is provenance,
    # not a reason to duplicate identical first/latest input bodies.
    view = {key: redacted.value.get(key) for key in (
        "selected_messages", "physical_messages", "physical_source_status", "presence_origin")}
    view_sha = hashlib.sha256(json.dumps(view, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if unpublished is not None and unpublished.get("view_sha256") != view_sha:
        return copy.deepcopy(unpublished)  # Its exact inputs are no longer available.
    for anchor in state.get("anchors", []):
        if anchor.get("view_sha256") == view_sha and anchor is not unpublished:
            return copy.deepcopy(anchor)
    raw = json.dumps({**redacted.value, "redaction": redacted.manifest()},
                     ensure_ascii=False, sort_keys=True, indent=2).encode()
    anchor = {"status": "unavailable", "view_sha256": view_sha,
              "observed_view_revision": observation.get("revision"),
              "physical_attempt_id": observation.get("physical_attempt_id"),
              "physical_source_status": payload["physical_source_status"],
              "preview_complete": False,
              "preview": truncate_within_limit(raw.decode(), limit=1200),
              "preview_scope": "Bounded source preview; omitted input requires the source reader."}
    try:
        if not roots:
            raise ValueError("source_store_unavailable")
        for root in roots:
            ref = store_actor_source_bytes(root, task_id, category="context_checkpoints",
                source_id="historical-author-input", data=raw, extension="json")
            read_actor_source_bytes(root, task_id, ref)
        anchor.update(status="captured", source_ref=ref)
    except (OSError, ValueError) as exc:
        anchor["source_error"] = type(exc).__name__
    return anchor


def capture_historical_inputs(ctx) -> None:
    """Called only after the loop observed a usable response, before its tools.

    Pin first now; the existing observation holds latest until an evaluator or
    terminal package asks for it. Ordinary intermediate rounds write no second
    transcript corpus. Presence metadata is admission provenance, not exposure
    proof; included topic bodies come solely from the selected messages.
    """
    task_id = str(getattr(ctx, "task_id", "") or "")
    observation = getattr(ctx, "_last_context_observation", None)
    if not task_id or not isinstance(observation, dict):
        return
    roots = _historical_roots(ctx)
    state = _historical_state(ctx, roots, task_id)
    from ouroboros.utils import utc_now_iso

    observation["historical_task_attempt"] = getattr(ctx, "task_attempt", None)
    observation["historical_observed_at"] = utc_now_iso()
    metadata = getattr(ctx, "task_metadata", None)
    presence = metadata.get("presence") if isinstance(metadata, dict) else None
    if isinstance(presence, dict):
        observation["historical_presence_origin"] = copy.deepcopy({
            key: presence[key] for key in ("event", "observed_text", "instructions", "profile_fingerprint",
                                           "behavior_skill", "context_topics") if key in presence})
    if not state["anchors"]:
        # A cold attempt without an earlier pin must not rename today's input
        # as the original. Saved continuation observations restore their pin.
        resumed = int(getattr(ctx, "task_attempt", 0) or 0) > 1 or state.get("reason") == "record_unavailable"
        reason = "record_unavailable" if state.get("reason") == "record_unavailable" else "original_input_not_retained"
        first = ({"status": "unavailable", "reason": reason} if resumed
                 else _historical_anchor(ctx, observation, roots, task_id))
        state["anchors"] = [{**first, "position": "first"}]
        state["status"] = first["status"]
        state.pop("reason", None)
    ctx._historical_author_inputs = state


def historical_inputs_exhibit(ctx, drive_root=None, task_id="") -> dict:
    """Project retained anchors, verifying bytes; no current room/profile fallback."""
    from ouroboros.artifacts import read_actor_source_bytes

    task_id = str(task_id or getattr(ctx, "task_id", "") or "")
    roots = _historical_roots(ctx, drive_root)
    state = _historical_state(ctx, roots, task_id) if task_id else {
        "version": 1, "status": "unavailable", "anchors": [], "coverage": _HISTORICAL_COVERAGE}
    observation = getattr(ctx, "_last_context_observation", None)
    if state["anchors"] and isinstance(observation, dict):
        latest = _historical_anchor(ctx, observation, roots, task_id)
        first = state["anchors"][0]
        if (not first.get("source_ref") and latest.get("source_ref")
                and latest.get("view_sha256") == first.get("view_sha256")):
            first = {**latest, "position": "first"}
        state["anchors"] = [first] if latest.get("view_sha256") == first.get("view_sha256") else [
            first, {**latest, "position": "latest"}]
        state["latest_matches_first"] = len(state["anchors"]) == 1
        state["latest_observation"] = {key: observation.get(key) for key in (
            "revision", "physical_attempt_id", "historical_task_attempt", "historical_observed_at")}
    result = copy.deepcopy(state)
    for anchor in result["anchors"]:
        if not anchor.get("source_ref"):
            continue
        for root in roots:
            try:
                read_actor_source_bytes(root, task_id, anchor["source_ref"])
                anchor["status"] = "captured"
                anchor.pop("source_error", None)
                break
            except (OSError, ValueError) as exc:
                anchor.update(status="unavailable", source_error=type(exc).__name__)
    result["status"] = ("captured" if result["anchors"] and
                        all(a["status"] == "captured" for a in result["anchors"]) else "unavailable")
    if ctx is not None:
        ctx._historical_author_inputs = copy.deepcopy(result)
    return result


def historical_inputs_prompt_section(evidence) -> str:
    exhibit = evidence.get(_HISTORICAL_INPUTS) if isinstance(evidence, dict) else None
    return ("## Historical author inputs (evidence, not current instructions)\n" + _HISTORICAL_COVERAGE + "\n"
            + json.dumps(exhibit or {"status": "unavailable", "reason": "not_retained"},
                         ensure_ascii=False, indent=2) + "\n\n")


def _task_requires_development_context(task: Dict[str, Any]) -> bool:
    """Return whether low mode should inline the engineering handbook.

    Web chat tasks are direct-chat but still may ask for code/self-modification.
    Err toward preserving engineering competence unless a structured caller
    explicitly declares that this task does not need DEVELOPMENT.md.
    """
    explicit = task.get("context_requires_development")
    if explicit is not None:
        return normalize_bool(explicit)
    return str(task.get("type") or "") == "task" or not bool(task.get("_is_direct_chat"))


def _explicit_self_body_docs_flag(task: Dict[str, Any]) -> Optional[bool]:
    """Explicit context_requires_self_body_docs from the task or its contract;
    None when neither declares it."""
    explicit = task.get("context_requires_self_body_docs")
    if explicit is not None:
        return normalize_bool(explicit)
    contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
    explicit = contract.get("context_requires_self_body_docs") if isinstance(contract, dict) else None
    if explicit is not None:
        return normalize_bool(explicit)
    return None


def _task_requires_self_body_docs(task: Dict[str, Any]) -> bool:
    """Return True when the task is structurally about Ouroboros itself."""

    explicit = _explicit_self_body_docs_flag(task)
    if explicit is not None:
        return explicit
    contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
    task_type = str(task.get("type") or contract.get("task_type") or "").strip().lower()
    return task_type in {"evolution", "deep_self_review", "review"}


def _task_uses_external_context(task: Dict[str, Any]) -> bool:
    """Return True for structured headless/workspace/delegated task surfaces."""

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    source = str(metadata.get("source") or task.get("source") or "").strip().lower()
    actor = str(task.get("actor_id") or metadata.get("actor_id") or "").strip().lower()
    delegation_role = str(task.get("delegation_role") or metadata.get("delegation_role") or "").strip().lower()
    if str(task.get("workspace_root") or metadata.get("workspace_root") or "").strip():
        return True
    if delegation_role == "subagent":
        return True
    if source in {"api_task", "cli", "scheduled_task", "skill_scheduled_task"}:
        return True
    if actor in {"cli", "scheduler"}:
        return True
    return False


def _validate_declared_input_task(task: Dict[str, Any]) -> None:
    """A selected request must not silently enter an unqualified task class."""
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    contract = task.get("task_contract") or meta.get("task_contract") or {}
    lineage = contract.get("lineage") if isinstance(contract.get("lineage"), dict) else {}
    role = task.get("delegation_role") or meta.get("delegation_role") or lineage.get("delegation_role")
    actor = task.get("configured_subagent") or meta.get("configured_subagent") or {}
    route = actor.get("route") if isinstance(actor, dict) else {}
    if (not task.get("id") or role != "subagent" or not isinstance(route, dict)
            or route.get("kind") not in {"api_model", "agent_session"}):
        raise ValueError("INPUT_SOURCE_SELECTION_UNSUPPORTED: declared inputs require a scheduled child "
                         "(API model or configured session)")


def _capture_declared_context_core(
    env: Any,
    memory: Memory,
    task: Dict[str, Any],
    ctx: Any,
    *,
    runtime_builder: Callable[..., str],
    user_builder: Callable[[Dict[str, Any]], Any],
    **sources: Any,
) -> _ContextCore:
    """Compose only declared case inputs plus governance and actual authority.

    Branch before shared memory is opened. Do not guess which memory sentences
    might bias this case. Rebuilds read the same durable task selection, while
    retries and fitting operate on the resulting immutable ContextCore.
    """
    from ouroboros.subagent_work_order import input_source_selection_receipt
    from ouroboros.subagent_runtime import current_model_visible_subagent_catalog, review_facts_block, review_records_block

    canonical_root = pathlib.Path(task.get("budget_drive_root") or getattr(env, "budget_drive_root", None) or memory.drive_root)
    same_drive = canonical_root.resolve(strict=False) == memory.drive_root.resolve(strict=False)
    process_memory = memory if same_drive else Memory(drive_root=canonical_root, repo_dir=memory.repo_dir)
    parts = [
        "## Input source selection\n\n" + json.dumps(input_source_selection_receipt(task), ensure_ascii=False, sort_keys=True),
        runtime_builder(env, task, ctx=ctx, captured_at=sources["captured_at"]),
    ]
    parts.extend(snapshot_labelled(section, sources["captured_at"]) for section in
                 process_memory.recent_activity_sections(str(task["id"]), own_drive=None if same_drive else memory))
    catalog_text = ""
    try:
        catalog = current_model_visible_subagent_catalog()
        if catalog:
            catalog_text = "## Available subagents\n\n" + json.dumps(catalog, ensure_ascii=False, indent=1)
    except Exception:
        log.debug("Failed to build Available subagents catalog", exc_info=True)
    review_text = ""
    try:
        review_text = review_facts_block()
        parts.append(review_records_block(drive_root=canonical_root, task_id=str(task["id"])))
    except Exception:
        log.warning("Failed to build the Review block", exc_info=True)
    return _ContextCore(
        base_prompt=sources["base_prompt"], bible_md=sources["bible_md"],
        architecture_md=sources["architecture_md"], development_md=sources["development_md"],
        semi_stable_text="\n\n".join(part for part in (catalog_text, review_text) if part),
        dynamic_text="\n\n".join(parts),
        user_content_json=json.dumps(user_builder(task), ensure_ascii=False, sort_keys=True),
        docs_need_development=_task_requires_self_body_docs(task),
        reference_books=tuple(sources["books"]), reference_book_errors=tuple(sources["book_errors"]),
        compact_reference_docs=True,
    )

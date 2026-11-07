"""Task input selection, the declared request's omissions and historical exposure.

The context facade captures governance once and supplies its runtime/user
renderers. This leaf decides task/document eligibility and names what a declared
request's view withholds of the automatic shared-memory channels (the one
builder in ``context`` composes it). Usable-response observations retain the
first/latest selected inputs as evidence for evaluators, without changing
current task authority or opening today's memory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.contracts.task_contract import normalize_bool
from ouroboros.reference_books import BOOK_ENTRYPOINTS
from ouroboros.store import StoreError

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
        except (OSError, ValueError, StoreError):
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


def _pointer_surface(task: Dict[str, Any]) -> bool:
    """BIBLE P1's on-demand surfaces: a delegated subagent, an external API/CLI/scheduled source or actor."""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    source = str(metadata.get("source") or task.get("source") or "").strip().lower()
    actor = str(task.get("actor_id") or metadata.get("actor_id") or "").strip().lower()
    delegation_role = str(task.get("delegation_role") or metadata.get("delegation_role") or "").strip().lower()
    return (delegation_role == "subagent" or source in {"api_task", "cli", "scheduled_task", "skill_scheduled_task"}
            or actor in {"cli", "scheduler"})


def _bound_tree_is_body(task: Dict[str, Any], ctx: Any) -> Optional[bool]:
    """Whether the task's bound work tree is Ouroboros's body (a worktree, copy or fork of it): ``True`` when
    ``body_fact`` recognizes it, ``False`` when it proves a foreign tree, ``None`` without a binding or when
    the fact is unknown (no git, no remote, an unreadable tree)."""
    from ouroboros.review_body_fact import BODY_FALSE, BODY_TRUE, body_fact
    from ouroboros.tools.tool_resolution import system_repo_dir_for
    from ouroboros.workspace_copies import same_directory

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    root = str(task.get("workspace_root") or metadata.get("workspace_root") or "").strip()
    if not root:
        return None
    try:
        system = system_repo_dir_for(ctx)
        if same_directory(root, system):
            return True
        data = getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None)
        fact = body_fact(root, system_repo=system, data_dir=data).body
    except Exception:
        log.debug("bound tree body fact unavailable; the subject stays unknown", exc_info=True)
        return None
    return True if fact == BODY_TRUE else False if fact == BODY_FALSE else None


def development_required(task: Dict[str, Any]) -> bool:
    """This task's own direct requirement of the whole handbook in every mode (owner 1.11.2026).

    Only its contract's ``context_requires_development`` is true: ``start_task`` authors that key for the new
    task alone, and no descendant or follow-up inherits it. A top-level value alone is the older answer about
    the subject in Max, and ``context_requires_self_body_docs`` is a subject marker: neither is this requirement.
    """
    contract = task.get("task_contract")
    return isinstance(contract, dict) and contract.get("context_requires_development") is True


def required_book_unavailable(core: Any) -> Dict[str, Any]:
    """The existing source-unavailable refusal when the capture could not read a handbook the task directly requires,
    else ``{}``, under its own typed cause (``required_reference_unavailable``, which Continue may continue; the other
    authority refusals keep theirs). No map or partial book stands in for it; without the requirement the gap stays a
    disclosed one in the view, as before."""
    if not core.development_required or core.development_md.strip():
        return {}
    path = BOOK_ENTRYPOINTS["development"]
    error = next((line for line in core.reference_book_errors if path in line), f"{path} could not be read.")
    return {"reason_code": "required_reference_unavailable", "source": {"kind": "reference_book", "path": path},
            "human_label": f"this task's directly required development handbook ({path})",
            "detail": (f"{error} The whole book is this task's own requirement; no map or partial book stands in. "
                       "Restore the source, then Continue with that requirement; changing it needs an explicit new task."),
            "recovery_choices": ["retry_after_source_recovery", "start_explicit_fresh_task"]}


def development_resident(task: Dict[str, Any], ctx: Any, *, subject_only: bool = False) -> bool:
    """Whether docs/DEVELOPMENT.md stays whole in Max, by the subject of the work (BIBLE P1, owner 6A).

    The task's own ``context_requires_development`` wins either way (absent is not false), read from its row and
    then its contract, which survives restore and Continue; ``subject_only`` asks what the subject alone says. A
    delegated child or an external API/CLI/scheduled surface gets the pointer unless the task itself asks for the
    handbook; self-body work keeps it; a bound work tree ``body_fact`` proves foreign gets the pointer. Anything
    else, an unbound task or a tree whose body fact is unknown, keeps the book: missing facts never narrow it.
    Project membership is not a signal.
    """
    contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
    explicit = None if subject_only else task.get("context_requires_development", contract.get("context_requires_development"))
    if explicit is not None:
        return normalize_bool(explicit)
    if _explicit_self_body_docs_flag(task) is True:
        return True
    if _pointer_surface(task):
        return False
    return _task_requires_self_body_docs(task) or _bound_tree_is_body(task, ctx) is not False


def task_input_pointer(env: Any, task: Dict[str, Any], user_content_json: str) -> Optional[str]:
    """The exact task input as a source pointer (user content JSON), or ``None`` when it cannot be retained.

    The original input stays exact in the captured core and the actor source store; only its Nano delivery
    changes. This is neither the external-assignment compiler nor a summary.
    """
    try:
        from ouroboros.artifacts import store_actor_source_bytes
        source = store_actor_source_bytes(env.drive_root, str(task["id"]), category="context_checkpoints",
            source_id="task_input", data=user_content_json.encode("utf-8"), extension="json")
    except (OSError, ValueError, KeyError):
        log.warning("Exact task input source could not be retained; preserving complete input", exc_info=True)
        return None
    return json.dumps("[Exact task input source]\nThe complete original user input is stored as JSON at this source. "
                      "Read its full content through the given reader in ranges before substantive decisions. "
                      "This pointer is not a summary or a change to the task.\n"
                      + json.dumps(source, ensure_ascii=False, sort_keys=True), ensure_ascii=False)


def _wake_forms(wake: Dict[str, Any]) -> List[str]:
    """A wake's bound deliveries in rescue order (``consciousness_wake.bind_wake_observation``): the projection (my
    human's card answers whole beside the exact pointer), an older wake's middle step when it bound one, then the
    bare pointer when bound apart. The choice among them is only ever their actual fit."""
    keys = ("projection_text", "owner_projection_text", "pointer_text")
    return [text for text in dict.fromkeys(wake.get(key) for key in keys) if text]


def wake_deliveries(wake: Dict[str, Any], full: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The round boundary's choices (``loop_model_call._project_wake_input``): this wake's bound forms as its first
    user message, in rescue order with the bare pointer last, each only when strictly smaller than ``full``."""
    size = len(json.dumps(full, ensure_ascii=False))
    forms = [{"role": "user", "content": text} for text in _wake_forms(wake)]
    return [form for form in forms if len(json.dumps(form, ensure_ascii=False)) < size]


def wake_input_checkpoint(wake: Dict[str, Any], content: Any, *, round_idx: int, after_overflow: bool) -> Dict[str, Any]:
    """The record of the wake delivery a request carried (``content``; ``None``: the whole observation): its
    ``form`` is ``whole``, ``projection`` (my human's card answers inline) or ``pointer`` (the bare pointer,
    nothing inline)."""
    forms = _wake_forms(wake)
    if content is None:
        form = "whole"
    elif content in forms[:-1]:  # any bound form before the last carries my human's words inline
        form = "projection"
    else:  # the last bound form, or a projection with no words of mine: the bare pointer
        form = "pointer"
    return {"checkpoint_kind": "wake_input_inline" if form == "whole" else "wake_input_by_source", "round": round_idx,
            "after_overflow": after_overflow, "form": form, "composition": wake.get("composition"),
            "source_sha256": (wake.get("source") or {}).get("sha256")}


def wake_projection_inputs(task: Dict[str, Any]) -> List[str]:
    """An own-clock wake's bound deliveries as user content JSON (``_wake_forms``), only while its complete source
    reads back exactly.

    ``consciousness_wake.bind_wake_observation`` stores the whole observation and renders these around its pointer;
    the round boundary delivers the same texts (``loop_model_call._project_wake_input``). A source its producer could
    not store, or one that no longer verifies, leaves none: the whole input stays the only delivery and a refused
    start names its frame.
    """
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.consciousness_authority import is_own_clock_wake
    from ouroboros.consciousness_wake import WAKE_OBSERVATION_KEY

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    wake = metadata.get(WAKE_OBSERVATION_KEY)
    if not is_own_clock_wake(metadata) or not isinstance(wake, dict) or not wake.get("projection_text"):
        return []
    source = wake.get("source")
    try:
        read_actor_source_bytes(source["canonical_root"], str(source["task_id"]), source)
    except (KeyError, TypeError, OSError, ValueError) as exc:
        log.warning("Wake %s observation source does not verify (%s: %s); its whole input stays the only delivery",
                    task.get("id"), type(exc).__name__, exc)
        return []
    return [json.dumps(str(text), ensure_ascii=False, sort_keys=True) for text in _wake_forms(wake)]


def rescued_start(env: Any, task: Dict[str, Any], user_content_json: str, unfit: Any, preferred: str,
                  fit: Callable[[str], Tuple[Any, str]]) -> Tuple[Any, str, Dict[str, str]]:
    """The start a physical refusal of the whole input leaves: ``(contents, mode, {mode: delivered input JSON})``.

    Physical rescue only (owner 2026-10-13, #1527): an exact source replaces the whole input only when no view of the
    known window holds it; a room-only shortfall, the directly required handbook, an unusable source or an unfit
    rescue names the start (7A). An own-clock wake's bound deliveries come first, in every mode: its projection, then,
    after that too proves physically unfit, its bare pointer (``wake_projection_inputs``); owner-Nano's exact input
    pointer stays the last rescue (``task_input_pointer``).
    """
    from ouroboros.context_fit import MinimumViewUnfit

    if not unfit.physical or unfit.required_book_tokens:
        raise unfit
    refused = None
    for wake in wake_projection_inputs(task):
        try:
            contents, mode = fit(wake)
        except MinimumViewUnfit as exc:
            refused = exc
            continue
        return contents, mode, dict.fromkeys(contents, wake)
    if refused is not None:
        if preferred != "nano":
            raise refused
        log.info("Wake %s projections cannot fit the owner-Nano start either; the exact input pointer is the last "
                 "rescue", task.get("id"))
    pointer = task_input_pointer(env, task, user_content_json) if preferred == "nano" else None
    if pointer is None:
        raise unfit
    contents, mode = fit(pointer)
    return contents, mode, {"nano": pointer}


def whole_wake_input(message: Dict[str, Any], plan: Any, wake: Dict[str, Any]) -> Dict[str, Any]:
    """The whole wake input the round boundary measures again (``loop_model_call._project_wake_input``).

    A first message that already holds one of this wake's deliveries, its projection or its bare pointer
    (``rescued_start``, or a restored transcript), is that message with its plan's captured whole input; any other
    first message is itself.
    """
    from ouroboros.context_fit import extract_plain_text_from_content

    whole, captured = copy.deepcopy(message), getattr(plan, "user_content_json", None)
    if isinstance(captured, str) and extract_plain_text_from_content(message.get("content")) in _wake_forms(wake):
        whole["content"] = json.loads(captured)
    return whole


def wake_start_checkpoint(task: Dict[str, Any], plan: Any) -> Optional[Dict[str, Any]]:
    """The round boundary's ``wake_input_by_source`` checkpoint for a start that delivered a bound projection or
    bare pointer, naming which (``wake_input_checkpoint``).

    ``round`` 0 is the start itself, before the first round; ``None`` when the start delivered any other input.
    """
    shown = plan.projection(plan.initial_mode).user_content_json
    if shown is None:  # the whole input, as an ordinary start delivers it
        return None
    from ouroboros.consciousness_wake import WAKE_OBSERVATION_KEY

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    wake = metadata.get(WAKE_OBSERVATION_KEY)
    if not isinstance(wake, dict) or json.loads(shown) not in _wake_forms(wake):
        return None
    return wake_input_checkpoint(wake, json.loads(shown), round_idx=0, after_overflow=False)


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


def _declared_omissions(task: Dict[str, Any], spec: Any) -> Any:
    """The one builder's view of a declared request: the role's (a child's or a nanny's) without the story, the
    parent's room page and its open conversation; my human's words ride it as a shared child's unless the task
    was scheduled before ``DECLARED_WORDS_FACT``; the marks in force and the knowledge outline ride it. Identity,
    WORLD, review history, skills and parent context are gated in the builder (``context._capture_context_core``).
    The receipt (``subagent_work_order.input_source_selection_receipt``) names each omission."""
    import dataclasses

    from ouroboros.subagent_work_order import declared_inputs_carry_owner_words  # D03->D07 is lazy-only

    return dataclasses.replace(spec, story=False, room_page=False, room_lanes=False, origin_words=False,
                               other_conversations=False, owner_words=declared_inputs_carry_owner_words(task))


def declared_receipt_section(task: Dict[str, Any]) -> str:
    """``## Input source selection``: the native request's receipt, the one builder's view named in it."""
    from ouroboros.subagent_work_order import input_source_selection_receipt  # D03->D07 is lazy-only

    receipt = input_source_selection_receipt(task, native=True)
    return "## Input source selection\n\n" + json.dumps(receipt, ensure_ascii=False, sort_keys=True)

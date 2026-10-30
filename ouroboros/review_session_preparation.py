"""Session task/request preparation shared by first-send sizing and dispatch.

Route capabilities are negotiated by the caller. Oversized text can be stored
as exact sources; this module never starts, waits or collects an operation.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from ouroboros.deadline_utils import bounded_seconds
from ouroboros.delegate_start_instructions import HOST_FACTS_INSTRUCTION, SHARED_ENVIRONMENT_INSTRUCTION
from ouroboros.triad_review import default_output_contract, request_output_shape


def render_review_session_prompt(request: Any, slot: Any, task: str) -> str:
    """The session wrapper, shared by source preparation's exact first-send measurement."""
    contract = str((request.policy or {}).get("output_contract") or "")
    contract = contract or default_output_contract(request_output_shape(request))
    access = str(getattr(slot, "session_access", "") or "readonly")
    return "\n".join((
        ("You are an independent Ouroboros reviewer slot running as a read-only agent session."
         if access == "readonly" else
         f"You are an independent Ouroboros reviewer slot running as an agent session with {access} "
         "technical access in a private checking folder.")
        + HOST_FACTS_INSTRUCTION,  # an agent session has a shell HOME; the raw API reviewer prompt does not
        SHARED_ENVIRONMENT_INSTRUCTION,
        f"Surface: {request.surface}", f"Role hint: {slot.role_hint or 'general reviewer'}", "", task, "",
        "OUTPUT CONTRACT (your host parses this structurally):",
        contract + "\nThis contract governs the unwrapped substantive deliverable; emit any host-required transport metadata outside it exactly as separately instructed.",
        f"Slot: {slot.slot_id}",
    ))


def session_material(surface: str, contract: str, schema: Any, accesses: Any) -> list:
    """Review-contract material (``commit_gate.commit_review_contract_fingerprint``): an agent-session seat's wrapper
    over a placeholder task for each seated access: the work-order frame with ``contract``, the read-only system
    instructions or the checking copy's, the answer ``schema`` and the paged-turn pointers. The host and environment
    facts every delegated run shares are not review text and are left out."""
    from types import SimpleNamespace

    from ouroboros.review_checking import checking_material
    from ouroboros.review_execution import _REVIEW_SESSION_INSTRUCTIONS
    from ouroboros.tools.review_brief_coupling import paged_review_text
    from ouroboros.tools.review_multi_model import TRIAD_ROLE_HINT

    source = {"session_relative_path": "⟨path⟩", "sha256": "⟨sha⟩"}
    material = [json.dumps(schema, sort_keys=True), *(paged_review_text("⟨text⟩", source, label=f"review {key}",
                                                                        delegated=True) for key in ("prompt", "instructions"))]
    for access in sorted(set(accesses)):
        prompt = render_review_session_prompt(SimpleNamespace(surface=surface, policy={"output_contract": contract}),
                                              SimpleNamespace(role_hint=TRIAD_ROLE_HINT, slot_id="⟨slot⟩", session_access=access),
                                              "⟨task⟩")
        material += [prompt.replace(HOST_FACTS_INSTRUCTION, "").replace(SHARED_ENVIRONMENT_INSTRUCTION, ""),
                     *([_REVIEW_SESSION_INSTRUCTIONS] if access == "readonly" else [
                         *checking_material(), *(_ENVIRONMENT_NOTE["note"].format(where=where) for where in (
                             _ENVIRONMENT_NOTE["scratch"].format(scratch="⟨scratch⟩"), _ENVIRONMENT_NOTE["outside"]))])]
    return material


def seat_service_directories(gateway: Any, route_id: str, shape: Any, data_root: Any, task_id: str,
                             operation_id: str, area: str, thread_seat: str = "") -> Any:
    """What a writable seat may write outside its copy (``workspace_preflight.session_service_directories``):
    one run's short scratch is owned by its review task's drive, keyed by the seat's directory name, and
    granted, or at full access only named. A sticky thread's turns carry only geometry
    (``start_review_thread_turn``), so its seat gets none."""
    from ouroboros.review_checking import seat_dir
    from ouroboros.workspace_preflight import ServiceDirectories, session_service_directories

    if thread_seat:
        return ServiceDirectories(limitation="thread_turns_carry_no_grant: a review thread's turns carry "
                                             "no execution grant" if shape.access == "workspace_write" else "")
    return session_service_directories(
        gateway, route_id, access=shape.access, mode=shape.mode, data_root=data_root, drive_root=data_root,
        task_id=task_id, key=f"seat-{seat_dir(data_root, task_id, operation_id).name}", execution_root=area)


# A writable seat's own note inside the environment facts every delegated run is given.
_ENVIRONMENT_NOTE = {"scratch": "in {scratch}, this run's disposable scratch, never inside this copy",
                     "outside": "outside this copy", "note": "{where}, since its changes are retained as evidence"}


def seat_environment_facts(area: str, source: Any, data_root: Any, service: Any = None) -> str:
    """A writable seat's PROJECT ENVIRONMENT facts from its own frozen copy (#1775).

    Its copy's changes are retained as evidence (``review_checking``), so the facts say
    where an environment may live; a folder read in place has no copy to build in, so
    its seat hears only ``service``: the scratch the host made for it outside its folder. The
    body's documented recipe is named only for a frozen tree of the body."""
    from ouroboros.workspace_preflight import (
        ENVIRONMENT_FACTS_HEADING,
        ServiceDirectories,
        session_environment_facts,
        session_service_facts,
    )

    kind, body, service = str(source.get("kind") or ""), False, service or ServiceDirectories()
    if kind not in ("git_tree", "directory"):  # a folder read in place: only what the seat may write outside
        return (ENVIRONMENT_FACTS_HEADING + session_service_facts(service, data_root).rstrip("\n")
                if service.scratch or service.limitation else "")
    if kind == "git_tree":
        try:
            from ouroboros.config import resolve_repo_dir
            from ouroboros.review_body_fact import BODY_TRUE, body_fact

            body = body_fact(source.get("root"), system_repo=resolve_repo_dir(), data_dir=data_root).body == BODY_TRUE
        except Exception:  # an unrecognized subject is foreign: a named recipe gap, never a guess
            body = False
    where = (_ENVIRONMENT_NOTE["scratch"].format(scratch=service.scratch) if service.scratch
             else _ENVIRONMENT_NOTE["outside"])
    return session_environment_facts(area, data_root=data_root, body=body, service=service,
                                     environment_note=_ENVIRONMENT_NOTE["note"].format(where=where))


def prepare_review_session_request(invocation: Any, route: Any, *,
                                   prompt: str, root: str, thread_id: str,
                                   schema_asked: bool, ephemeral: bool = False, shape: Any = None,
                                   execution: Optional[Dict[str, Any]] = None,
                                   instructions: str = "", input_limit: Optional[dict] = None,
                                   source_root: Any = None, delivery_facts: Optional[dict] = None,
                                   gateway: Any = None, capabilities: Optional[dict] = None) -> Dict[str, Any]:
    """Prepare and measure the exact wire request; no start, wait or recovery effects.

    ``shape`` is the seat's run shape (read-only unless its configured access is
    writable); a writable seat names its private checking copy in ``execution``
    and its own ``instructions``."""
    from ouroboros.effort_evidence import session_prompt_cache_ttl_request
    from ouroboros.review_execution import ReviewRouteUnavailable, _CLAUDEXOR_MAX_SECONDS
    from ouroboros.subagents import delegated_run_shape
    shape = shape or delegated_run_shape(False, str(getattr(invocation, "access", "") or "readonly"))
    seconds = bounded_seconds(invocation.timeout_sec, default=300, maximum=_CLAUDEXOR_MAX_SECONDS)
    use_thread, output_schema = invocation.use_thread, invocation.output_schema
    instructions = instructions or invocation.instructions
    run_request = {
        "prompt": prompt,
        "instructions": instructions,
        "authPreference": "subscription",
        "mode": shape.mode,
        "access": shape.access,
        "scope": {"kind": "project", "root": root, **({"ephemeral": True} if ephemeral else {})},
        # A one-element explicit pool is the pin; primaryHarness is only preference.
        "harnesses": [route.route_id],
        "primaryHarness": route.route_id,
        "maxSeconds": seconds,
    }
    if execution:
        run_request["execution"] = dict(execution)
    if use_thread:
        run_request["_use_thread"] = True
        run_request["_thread_id"] = thread_id
    if route.model:
        run_request["model"] = route.model
    if route.effort:
        run_request["effort"] = route.effort
    if use_thread or getattr(route, "profile_id", ""):
        run_request["credentialProfileId"] = getattr(route, "profile_id", "") or None
    if schema_asked:
        run_request["outputSchema"] = output_schema
    facts = delivery_facts if delivery_facts is not None else invocation.source_delivery
    facts = facts if facts is not None else {}
    facts["prompt_cache_ttl"] = session_prompt_cache_ttl_request(
        run_request, route_id=route.route_id, gateway=gateway, capabilities=capabilities)
    limit = input_limit or facts.get("input_limit") or {}
    budget = limit.get("prompt_budget")
    if type(budget) is int and budget >= 0:
        facts["input_limit"] = dict(limit)
        # This is the engine's declared text budget, not JSON bytes or a model
        # context estimate. Count both caller-controlled text fields.
        text_size = lambda: len(run_request["prompt"]) + len(run_request["instructions"])
        if text_size() > budget:
            from ouroboros.tools.review_brief_coupling import BriefInputs, _store_review_source, paged_review_text

            brief = BriefInputs(delegated=True, task_id=invocation.task_id or "review",
                                source_root=str(source_root or facts.get("data_root") or ""))
            reading_root = Path((execution or {}).get("workspaceRoot") or root)
            facts["original_prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for key in sorted(("prompt", "instructions"), key=lambda k: len(run_request[k]), reverse=True):
                original = run_request[key]
                source = _store_review_source(reading_root, brief, original.encode("utf-8"), f"review-turn-{key}")
                if not source.get("path"):
                    raise ReviewRouteUnavailable(
                        f"Oversized review {key} could not be delivered as a file: {source.get('error') or 'no source root'}",
                        code="degraded_source_unreachable")
                run_request[key] = paged_review_text(original, source, label=f"review {key}", delegated=True)
                facts.setdefault("turn_sources", {})[key] = source
                if text_size() <= budget:
                    break
            if text_size() > budget:
                raise ReviewRouteUnavailable("Review source pointers exceed the declared transport text limit",
                                             code="input_too_large")
        facts["input_limit"].update(prompt_chars=text_size(), fits=text_size() <= budget)
    serialized = json.dumps(run_request, ensure_ascii=False)
    facts.update(first_send_chars=len(serialized), first_send_bytes=len(serialized.encode("utf-8")))
    return run_request

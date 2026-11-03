"""Plan leaves: frozen artifacts, addressed answer turns and starter-authorized implementation.

The ordinary start producer owns every invocation, paid stamp and recovery. This
adapter supplies thread bindings and plan facts; it never sends an unrecorded run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PLAN_FACT_KEYS = ("execution_mode", "effective_access", "thread_id", "authority_ceiling", "promotion")


def plan_facts(record: Any) -> dict:
    return {key: copy.deepcopy(record.get(key) if isinstance(record, dict) else getattr(record, key))
            for key in PLAN_FACT_KEYS
            if (key in record if isinstance(record, dict) else hasattr(record, key))}


def promotion_notice(ctx, entry) -> None:
    """Show the recorded starter's act in its existing task-card timeline/history."""
    progress = getattr(ctx, "emit_progress_fn", None)
    act = entry.promotion
    if not act or not callable(progress) or act.get("authorized_by_task_id") != str(ctx.task_id):
        return
    try:
        progress(f"This task authorized plan implementation with {act['to_access']} access.",
                 card_row="timeline", card_row_id=f"plan-implementation:{act['invocation_id']}")
    except Exception:
        logging.getLogger(__name__).debug("Plan authorization card delivery failed", exc_info=True)


def promotion_permitted(ctx, entry, authority, target_root, fingerprint) -> bool:
    """The only exception to continuation's no-widening floor, including current rights."""
    from ouroboros.delegate_continuation import _ACCESS_RANK

    ceiling = entry.authority_ceiling
    allowed = ceiling.get("allowed_access") if isinstance(ceiling, dict) else None
    return bool(entry.mode == "plan" and entry.thread_id and entry.task_id and not entry.review_owned
                and entry.task_id == str(getattr(ctx, "task_id", "") or "")
                and allowed in _ACCESS_RANK and authority.access in _ACCESS_RANK
                and _ACCESS_RANK[authority.access] <= _ACCESS_RANK[allowed]
                and fingerprint and fingerprint == ceiling.get("task_authority_fingerprint")
                and target_root and target_root == ceiling.get("target_root"))


def inherited_mode(ctx, run_id) -> str | None:
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run

    entry = run(custody.custody_root(ctx), str(run_id))
    return "plan" if entry is not None and entry.mode == "plan" else None


def payload_continuation(ctx, run_id) -> bool:
    """A plan keeps its exact payload selector; fresh resource admission still revalidates it."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run

    entry = run(custody.custody_root(ctx), str(run_id))
    return bool(entry is not None and entry.mode == "plan" and entry.resource_ref.get("root") == "skill_payload")


def bind_thread(gateway, request, route, invocation_id, *, workspace_root="", prior=None) -> dict:
    """Bind before START_REQUESTED; recovery already holds these exact local routing keys."""
    from ouroboros.review_thread_continuity import ensure_execution_thread

    thread_id = str((prior or {}).get("thread_id") or "")
    if not thread_id:
        thread_id = ensure_execution_thread(
            gateway, route=route, root=request["scope"]["root"], mode=request["mode"],
            access=request["access"], workspace_root=workspace_root,
            title="Ouroboros delegated plan", idempotency_key=f"delegate-thread:{invocation_id}")
    request["_thread_id"] = thread_id
    return {"thread_id": thread_id}


def bind_start(gateway, request, route, invocation, continuation, authority, ceiling, actor, target, workspace):
    """Freeze the ceiling before the first read-only turn; successors inherit it verbatim."""
    facts = plan_facts(continuation.custody)
    execution = request.get("execution") or {}
    if facts.get("thread_id") and execution.get("workspaceKind") == "directory" and execution.get("isolation") != "live":
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        raise ClaudexorUnavailable("same_session_promotion_unsupported", "The plan thread is bound to its folder; use a fresh executor for a selected copy.")
    if authority.mode == "plan" and not facts.get("thread_id") and not facts.get("authority_ceiling"):
        facts["authority_ceiling"] = {"task_authority_fingerprint": actor["authority_fingerprint"],
            "allowed_access": ceiling.access, "target_root": target}
    if authority.mode == "plan" or facts.get("thread_id"):
        facts.update(bind_thread(gateway, request, route, invocation, workspace_root=workspace, prior=facts))
    facts.update(execution_mode=authority.mode, effective_access=authority.access)
    return continuation._replace(custody={**continuation.custody, **facts})


def _retain(root, entry, name, data, extension):
    from ouroboros.artifacts import store_actor_source_bytes

    return {**store_actor_source_bytes(root, entry.task_id, category="tool_results",
        source_id=f"delegated-{name}-{entry.run_id}", data=data, extension=extension), "task_id": entry.task_id}


def terminal_plan(entry, gateway, detail=None) -> dict:
    """Read exact engine artifacts; missing/malformed sources remain unverified."""
    receipt = {"plan_run_id": entry.run_id, "thread_id": entry.thread_id,
               "readiness": "unverified", "questions": [], "source_gaps": []}
    if gateway is None or not entry.ledger_root:
        receipt["source_gaps"].append("plan_transport_unavailable")
        return receipt
    root = Path(entry.ledger_root)
    for name, path, extension in (("plan", "final/plan.md", "md"),
                                  ("questions", "final/questions.json", "json")):
        try:
            data = gateway.get_run_artifact(entry.run_id, path)
            ref = _retain(root, entry, name, data, extension)
            receipt[f"{name}_ref"] = ref
            if name == "plan":
                receipt["plan_sha256"] = hashlib.sha256(data).hexdigest()
            else:
                document = json.loads(data)
                questions = document.get("questions")
                if (not isinstance(questions, list) or not all(isinstance(q, dict)
                        and isinstance(q.get("id"), str) and q["id"].strip() for q in questions)
                        or len({q["id"] for q in questions}) != len(questions)
                        or document.get("parse") not in ("found", "none_found")):
                    raise ValueError("unverified questions contract")
        except Exception as exc:
            receipt["source_gaps"].append(f"{name}:{type(exc).__name__}")
    try:
        # Claudexor 4.5.0 schema/plan.ts owns readiness; consume its run projection.
        projection = detail if detail is not None else gateway.get_run(entry.run_id)
        readiness = (projection.get("planReadiness") or {}).get("state")
        questions = projection.get("planQuestions")
        if (readiness not in ("ready", "needs_answers", "unverified") or not isinstance(questions, list)
                or not all(isinstance(q, dict) and isinstance(q.get("id"), str) and q["id"].strip()
                    and q.get("kind") in ("single", "multi", "text")
                    and isinstance(q.get("prompt"), str) and q["prompt"].strip() for q in questions)
                or len({q["id"] for q in questions}) != len(questions)):
            raise ValueError("unverified plan projection")
        receipt.update(questions=questions, readiness=readiness)
    except Exception as exc:
        receipt["source_gaps"].append(f"projection:{type(exc).__name__}")
    if receipt["source_gaps"]:
        receipt["readiness"] = "unverified"
    if receipt["questions"]:
        receipt["reply_to"] = {"starter_task_id": entry.task_id, "run_id": entry.run_id,
            "plan_run_id": entry.run_id, "question_ids": [q["id"] for q in receipt["questions"]]}
    receipt["note"] = ("These are terminal plan artifacts, not pending interactions. The starter may answer "
        "with delegate_answer, escalate this reply_to, clarify with mode=plan and continue_from, or explicitly "
        "implement with mode=agent and continue_from within the recorded authority ceiling.")
    return receipt


def retained_plan(drive, entry, gateway=None):
    from ouroboros.delegate_collection import retained_payload

    retained = retained_payload(drive, entry, require_complete=False) or {}
    plan = retained.get("plan")
    return plan if isinstance(plan, dict) and plan.get("plan_sha256") else terminal_plan(entry, gateway)


def continuation(ctx, drive, entry, gateway, authority, actor, invocation_id, text, *,
                 override=None, answers=None, carrier=None):
    """A terminal plan continues only through the engine's thread contract."""
    from ouroboros.delegate_continuation import ContinuationStart, _adopted_snapshot
    from ouroboros.delegate_custody import held_run_names
    from ouroboros.subagent_route_health import plan_capability
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    catalog = gateway.agent_capabilities()
    route_row = next((r for r in catalog.get("harnesses", []) if r.get("id") == entry.route_id), {})
    if reason := plan_capability(gateway, catalog, route_row, promotion=authority.mode == "agent"):
        raise ClaudexorUnavailable(reason, "This engine cannot continue the frozen plan on its thread; start a fresh executor with the retained plan.")
    if carrier is not None:
        raise ClaudexorUnavailable("plan_carrier_conflict", "Plan turns use their recorded thread, without continue_carrier.")
    if not entry.thread_id or entry.task_id != str(ctx.task_id):
        raise ClaudexorUnavailable("continuation_authority_widened", "Only the recorded starter may continue its plan thread.")
    thread = gateway.get_thread(entry.thread_id)
    head = str((thread.get("thread") or {}).get("headRunId") or "")
    names = ({entry.run_id} if head == entry.run_id else
             held_run_names(gateway.get_run(entry.run_id), entry.run_id))
    if not head or head not in names:
        raise ClaudexorUnavailable("plan_thread_head_mismatch", "The addressed plan is not the thread's current head; no new turn was sent.")
    plan = retained_plan(drive, entry, gateway)
    request = {"_thread_id": entry.thread_id}
    promotion = {}
    if authority.mode == "agent":
        if not plan.get("plan_sha256"):
            raise ClaudexorUnavailable("plan_artifact_unavailable", "The frozen plan could not be retained and hashed.")
        data = gateway.get_run_artifact(entry.run_id, "final/plan.md")
        if not data.strip() or hashlib.sha256(data).hexdigest() != plan["plan_sha256"]:
            raise ClaudexorUnavailable("plan_hash_mismatch", "The engine plan no longer matches the retained plan; no implementation was sent.")
        if plan.get("readiness") != "ready" and override is not True:
            raise ClaudexorUnavailable("plan_readiness_required", "The plan has open or unverified questions; answer them or explicitly set override_plan_readiness=true.")
        # The server freezes and records planHash; no client hash field exists in 4.5.0.
        request.update(planRunId=head, overridePlanReadiness=override is True)
        promotion = {"from_run_id": entry.run_id, "from_mode": entry.mode, "to_mode": authority.mode,
            "from_access": entry.access, "to_access": authority.access, "authorized_by_task_id": ctx.task_id,
            "authority_fingerprint": actor["authority_fingerprint"], "invocation_id": invocation_id,
            "plan_sha256": plan["plan_sha256"], "override_plan_readiness": override is True}
    elif authority.mode == "plan":
        if answers is not None:
            request["answersPlanRunId"] = head
    else:
        raise ClaudexorUnavailable("plan_mode_required", "Continue a plan with mode=plan or explicitly implement with mode=agent.")
    snapshot = None
    if entry.snapshot_id:
        if entry.patch_disposed:
            raise ClaudexorUnavailable("continuation_snapshot_missing", "The plan thread's bound workspace is no longer available.")
        snapshot, code, detail = _adopted_snapshot(entry, SimpleNamespace(access="workspace_write"))
        if code:
            raise ClaudexorUnavailable(code, detail)
    elif authority.access != "readonly" and entry.resource_ref.get("plan_workspace_kind") != "directory":
        raise ClaudexorUnavailable("same_session_promotion_unsupported", "This read-only thread has no private implementation workspace; start a fresh executor with the frozen plan.")
    facts = {"thread_id": entry.thread_id, "execution_mode": authority.mode, "effective_access": authority.access,
             "authority_ceiling": copy.deepcopy(entry.authority_ceiling), "promotion": promotion}
    return ContinuationStart(facts={**facts, "continuation_of": entry.run_id, "resource_ref": copy.deepcopy(entry.resource_ref)}, prompt=text, request=request,
        custody={**facts, "continuation_of": entry.run_id, "capture_id": invocation_id,
                 "snapshot_task_id": entry.snapshot_task_id or entry.task_id}, snapshot=snapshot)


def answer_plan(ctx, entry, answers, reply_to=None):
    """Answer an exact artifact question set by a new ordinary, custodied plan start."""
    from ouroboros.delegate_interactions import _normalized_answers
    from ouroboros.delegate_reply import normalize_reply_to
    from ouroboros.delegate_shared import _fail
    from ouroboros.subagent_runtime import exact_start
    from ouroboros import delegate_custody as custody

    wire, error = _normalized_answers(answers)
    if error is not None:
        return error
    try:
        address = normalize_reply_to(reply_to)
        plan = retained_plan(custody.custody_root(ctx), entry)
        ids = {q["id"] for q in plan["questions"]}
        if (not wire or not ids or not {q["questionId"] for q in wire}.issubset(ids)
                or len({q["questionId"] for q in wire}) != len(wire)
                or address is not None and (address.get("plan_run_id") != entry.run_id
                    or address["run_id"] != entry.run_id or address["starter_task_id"] != entry.task_id
                    or set(address["question_ids"]) != ids)):
            raise ValueError("The answer must name this plan's exact recorded open questions")
    except (ValueError, KeyError) as exc:
        return _fail("delegate_answer", "reply_to_mismatch", str(exc), run_id=entry.run_id)
    snapshot = {"schema": 1, "selected_subagent_id": entry.selected_subagent_id,
        "config_fingerprint": entry.config_fingerprint, "effort": entry.effort,
        "access": entry.authority_ceiling.get("allowed_access") or entry.access,
        "route": {"kind": "agent_session", "target_id": entry.route_id + (f"={entry.model}" if entry.model else ""),
                  "credential_profile_id": entry.profile_id}}
    payload = ({"root": "skill_payload", "bucket": entry.resource_ref["source"],
                "skill_name": entry.resource_ref["skill_name"]} if payload_continuation(ctx, entry.run_id) else {})
    return exact_start(ctx, json.dumps({"answers": wire}, ensure_ascii=False), {
        "snapshot": snapshot, "continue_from": entry.run_id,
        "mode": "plan", "_plan_answers": wire, **payload})

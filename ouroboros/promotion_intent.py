"""Durable new-work intent before transport, and conservative boot reconciliation.

The task result is the existing receipt store. A prepared intent proves admission
has not begun. Once the supervisor claims admission, missing evidence is unknown;
recovery never emits an event or enqueues work.
"""

from __future__ import annotations

import json
from typing import Any

from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import utc_now_iso


def promotion_intent_notice(task_id: str, result: dict) -> str:
    """Expose pending/held/not-started intent through the existing result reader."""
    admission = result.get("promotion_admission") or {}
    status = admission.get("status")
    if status == "not_started":
        intent = result.get("promotion_intent") or {}
        arguments = {key: intent[key] for key in (
            "objective", "title", "expected_output", "project_id", "project_name", "workspace_root",
            "workspace", "source", "context_requires_self_body_docs", "context_requires_development", "reasoning_effort",
        ) if key in intent}
        return (f"Task {task_id}: not_started. No admission began. An explicit retry is available: "
                f"start_task({json.dumps(arguments, ensure_ascii=False)}). Nothing restarts automatically.")
    if result.get("status") != "requested" or status not in {"emitted", "unconfirmed"}:
        return ""
    since = admission.get("prepared_at") or admission.get("emitted_at") or "unknown time"
    return (f"Task {task_id}: admission pending since {since}. "
            + ("The complete intent is durable. " if result.get("promotion_intent") else "This is a legacy receipt. ")
            + ("Admission is held for reconciliation. " if status == "unconfirmed" else "No settled admission receipt exists yet. ")
            + f"Read get_task_result({task_id}) for its recorded outcome before an explicit retry; "
            "elapsed time alone cannot prove this work was not started. A new start_task creates a NEW task id.")


def prepare_promotion_intent(root: Any, event: dict) -> dict:
    """Commit the complete replayable event before any emission; fail closed."""
    from ouroboros.dialogue_provenance import presence_root_carrier

    intent = json.loads(json.dumps(event, ensure_ascii=False, allow_nan=False))
    tid, token = str(intent.get("task_id") or ""), str(intent.get("routing_token") or "")
    if not tid or not token:
        raise ValueError("New work requires a task id and routing token")
    carrier = presence_root_carrier(intent, task_contract=intent.get("task_contract"))
    stored = write_task_result(
        root, tid, "requested", create_only=True, strict_existing_dict=True,
        project_id=str(intent.get("project_id") or ""), description=str(intent.get("objective") or ""),
        expected_output=str(intent.get("expected_output") or ""), suggested_name=str(intent.get("title") or ""),
        promotion_intent=intent,
        promotion_admission={"status": "emitted", "phase": "prepared", "routing_token": token,
                             "prepared_at": utc_now_iso()},
        **({"metadata": carrier, "source": "presence_promote", "delegation_role": "root", "root_task_id": tid}
           if carrier else {}),
    )
    saved = stored.get("promotion_intent") if isinstance(stored, dict) else None
    if (not isinstance(saved, dict) or {k: v for k, v in saved.items() if k != "ts"}
            != {k: v for k, v in intent.items() if k != "ts"}):
        raise RuntimeError("The complete new-work intent was not persisted")
    event.clear()
    event.update(saved)  # An explicit transport retry emits the original timestamp too.
    return stored


def claim_promotion_intent(root: Any, event: dict) -> None:
    """Before admission side effects, retire prepared's never-admitted proof."""
    tid, token = str(event.get("task_id") or ""), str(event.get("routing_token") or "")
    stored = load_task_result(root, tid, strict=True) or {}
    if not stored.get("promotion_intent"):
        return  # Historical host producers keep their existing admission contract.

    def claim(current, _incoming):
        admission = current.get("promotion_admission") or {}
        if (current.get("status") != "requested" or admission.get("status") != "emitted"
                or admission.get("routing_token") != token):
            return None
        return {"status": "requested", "promotion_admission": {
            **admission, "phase": "admitting", "admission_started_at": utc_now_iso(),
        }}

    stored = write_task_result(root, tid, "requested", _field_projector=claim, strict_existing_dict=True)
    admission = stored.get("promotion_admission") or {}
    if admission.get("phase") != "admitting" or admission.get("routing_token") != token:
        raise RuntimeError("The new-work admission claim was not persisted")


def reconcile_promotion_intents(root: Any, *, protected: set[str]) -> dict:
    """At boot with prior workers proved gone: settle only proven unadmitted intent."""
    from ouroboros.task_results import list_task_results

    report = {"not_started": [], "held": []}
    for row in list_task_results(root, statuses=["requested"], strict=True):
        tid = str(row.get("task_id") or row.get("id") or "")
        admission = row.get("promotion_admission") or {}
        if not tid or tid in protected or admission.get("status") != "emitted":
            continue
        token = admission.get("routing_token")

        def reconcile(current, _incoming):
            actual = current.get("promotion_admission") or {}
            if (current.get("status") != "requested" or actual.get("status") != "emitted"
                    or actual.get("routing_token") != token):
                return None
            unadmitted = (actual.get("phase") == "prepared" and bool(current.get("promotion_intent"))
                          and not any(current.get(key) for key in (
                              "admitted_dispatch", "admitted_dispatch_mark", "execution_owner", "started_at")))
            status = "not_started" if unadmitted else "unconfirmed"
            text = ("This work was not started. Retry explicitly with start_task using the saved promotion_intent."
                    if unadmitted else "Admission may have started. This intent is held for reconciliation; no automatic retry.")
            return {"status": "failed" if unadmitted else "requested", "result": text,
                    "reason_code": "promotion_not_started" if unadmitted else "promotion_admission_unknown",
                    "promotion_admission": {**actual, "status": status, "retry_available": unadmitted,
                                            "reconciled_at": utc_now_iso(), "detail": text}}

        stored = write_task_result(root, tid, "requested", _field_projector=reconcile, strict_existing_dict=True)
        settled = (stored.get("promotion_admission") or {}).get("status")
        if settled in {"not_started", "unconfirmed"}:
            report["not_started" if settled == "not_started" else "held"].append(tid)
    return report

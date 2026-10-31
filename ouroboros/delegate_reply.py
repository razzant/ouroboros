"""Exact interaction or plan-question addresses: context, never control authority."""
from __future__ import annotations

import json
from typing import Any


REPLY_TO_SCHEMA = {
    "type": "object",
    "description": "Exact question address from waiting_on_user or a terminal plan receipt. Preserve through escalation and replies; it grants no authority.",
    "properties": {
        "starter_task_id": {"type": "string"},
        "run_id": {"type": "string"},
        "interaction_id": {"type": "string"},
        "plan_run_id": {"type": "string"},
        "question_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["starter_task_id", "run_id", "question_ids"],
    "additionalProperties": False,
}


def normalize_reply_to(value: Any) -> dict | None:
    """Missing stays missing; malformed or unsupported addresses fail explicitly."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("reply_to must be an exact question address")
    selector = "plan_run_id" if "plan_run_id" in value else "interaction_id"
    keys = {"starter_task_id", "run_id", selector, "question_ids"}
    if set(value) != keys or any(
        not isinstance(value.get(key), str) or not value[key].strip()
        for key in keys - {"question_ids"}
    ):
        raise ValueError("reply_to requires starter_task_id, run_id, question_ids and exactly one of interaction_id or plan_run_id")
    questions = value["question_ids"]
    if (not isinstance(questions, list) or not questions
            or any(not isinstance(q, str) or not q.strip() for q in questions)
            or len(set(questions)) != len(questions)):
        raise ValueError("reply_to.question_ids must be distinct non-empty question ids")
    return {**value, "question_ids": list(questions)}


def addressed_questions(starter: str, run_id: str, pending: list) -> list:
    """Stamp each interaction separately; several questions never mean 'latest'."""
    rows = []
    for row in pending:
        address = {"starter_task_id": starter, "run_id": run_id,
                   "interaction_id": row.get("interaction_id"),
                   "question_ids": [q.get("question_id") for q in row.get("questions", [])]}
        try:
            address = normalize_reply_to(address)
        except ValueError:
            address = None  # A malformed engine row supplies no invented address.
        rows.append({**row, **({"reply_to": address} if address else {})})
    return rows


def reply_instruction(address: dict | None) -> str:
    if address is None:
        return ""
    return ("reply_to=" + json.dumps(address, ensure_ascii=False, sort_keys=True)
            + "\nReturn the answer to starter_task_id with this exact reply_to. Only the "
              "starter submits delegate_answer; this address grants no permission. "
              "If the question has resolved, do not substitute another question.")


def validate_reply_target(ctx: Any, address: Any, *, recipient: str = "") -> dict | None:
    """Bind a supplied address to durable custody, without widening message reach."""
    address = normalize_reply_to(address)
    if address is None:
        return None
    from ouroboros.delegate_shared import message_run, delegate_payload

    refusal, entry, _relation = message_run(ctx, "addressed_reply", address["run_id"])
    if refusal is not None or entry is None:
        reason = delegate_payload(refusal)["reason"] if refusal is not None else "run_ownership_unknown"
        raise ValueError(f"reply_to unavailable: {reason}")
    if "plan_run_id" in address and (entry.mode != "plan" or address["plan_run_id"] != entry.run_id):
        raise ValueError("reply_to does not identify this plan run")
    if entry.task_id != address["starter_task_id"]:
        raise ValueError("reply_to starter does not own the addressed run")
    if recipient and recipient != entry.task_id:
        raise ValueError("An addressed answer must return to reply_to.starter_task_id")
    return address


def validate_answer_address(address: Any, starter: str, run_id: str,
                            interaction_id: str, answers: list) -> dict | None:
    address = normalize_reply_to(address)
    if address is not None and (
        address["starter_task_id"] != starter or address["run_id"] != run_id
        or address.get("interaction_id") != interaction_id
        or not {a.get("questionId") for a in answers}.issubset(address["question_ids"])
    ):
        raise ValueError("reply_to does not match this starter, run, interaction and answer set")
    return address


def require_pending_address(address: dict, pending: list) -> None:
    matching = [row for row in pending if row.get("interaction_id") == address["interaction_id"]]
    if (len(matching) != 1
            or matching[0].get("run_id") not in (None, "", address["run_id"])
            or set(address["question_ids"]) != {
        q.get("question_id") for q in matching[0].get("questions", [])
    }):
        raise ValueError("reply_to is stale or does not identify these pending questions")

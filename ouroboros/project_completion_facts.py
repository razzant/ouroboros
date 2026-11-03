"""The Main mirror's outcome and host reference to the request that began the work."""
from __future__ import annotations

from ouroboros.chat_references import origin_reply
from ouroboros.outcomes import normalize_outcome_axes


def completion_facts(root, task_id: str, task: dict, result: dict, event: dict) -> dict:
    """Freeze these facts with the owed delivery, before live or history projection."""
    axes = {key: value for source in (event, result)
            for key, value in (source.get("outcome_axes") or {}).items() if isinstance(value, dict)}
    facts = {"outcome_axes": normalize_outcome_axes({**event, **result, "outcome_axes": axes})}
    reply = origin_reply(root, task_id, {**task, **result}) or origin_reply(root, task_id)
    if not reply:
        from ouroboros.projects_registry import project_binding_for_task
        binding = project_binding_for_task(root, task_id) or {}
        reply = origin_reply(root, task_id, {"origin_message_ref": binding.get("source_ref")})
    if reply:
        facts["reply_to"] = reply
    return facts

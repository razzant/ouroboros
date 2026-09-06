"""Pure supervisor task-payload construction."""

from __future__ import annotations

from typing import Any, Dict

from ouroboros.depth_evidence import parse_task_depth


def build_scheduled_task_payload(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an admitted schedule event into the worker payload shape."""

    tid = str(fields.get("tid") or "")
    chat_id = int(fields.get("chat_id") or 0)
    text = str(fields.get("text") or "")
    desc = str(fields.get("desc") or "")
    expected_output = str(fields.get("expected_output") or "")
    constraints = str(fields.get("constraints") or "")
    role = str(fields.get("role") or "")
    task_context = str(fields.get("task_context") or "")
    depth = parse_task_depth(fields.get("depth"), default=0)
    root_task_id = str(fields.get("root_task_id") or "")
    session_id = str(fields.get("session_id") or "")
    actor_id = str(fields.get("actor_id") or "")
    delegation_role = str(fields.get("delegation_role") or "")
    memory_mode = str(fields.get("memory_mode") or "")
    drive_root = str(fields.get("drive_root") or "")
    child_drive_root = str(fields.get("child_drive_root") or "")
    budget_drive_root = str(fields.get("budget_drive_root") or "")
    root_cost_ceiling_usd = fields.get("root_cost_ceiling_usd")
    task_constraint = fields.get("task_constraint") if isinstance(fields.get("task_constraint"), dict) else None
    required_capabilities = fields.get("required_capabilities") if isinstance(fields.get("required_capabilities"), list) else []
    workspace_root = str(fields.get("workspace_root") or "")
    workspace_mode = str(fields.get("workspace_mode") or "")
    project_id = str(fields.get("project_id") or "")
    allowed_resources = fields.get("allowed_resources") if isinstance(fields.get("allowed_resources"), dict) else {}
    task_contract = fields.get("task_contract") if isinstance(fields.get("task_contract"), dict) else {}
    depth_provenance = fields.get("depth_provenance") if isinstance(fields.get("depth_provenance"), dict) else {}
    parent_id = fields.get("parent_id")
    # Intent only; dispatch derives every effective lane/executor fact later.
    requested_model_lane = str(fields.get("requested_model_lane") or fields.get("model_lane") or "auto")
    parent_model_lane = str(fields.get("parent_model_lane") or "")
    required_model_lane = str(fields.get("required_model_lane") or "")
    requested_executor = str(fields.get("requested_executor") or "").strip().lower() or "auto"
    task_group_id = str(fields.get("task_group_id") or "")
    task_group = fields.get("task_group") if isinstance(fields.get("task_group"), dict) else {}
    subagent_envelope = fields.get("subagent_envelope") if isinstance(fields.get("subagent_envelope"), dict) else {}
    configured_subagent = fields.get("configured_subagent") if isinstance(fields.get("configured_subagent"), dict) else {}
    parent_cognitive_route = fields.get("parent_cognitive_route") if isinstance(fields.get("parent_cognitive_route"), dict) else {}
    task: Dict[str, Any] = {
        "id": tid,
        "type": "task",
        "chat_id": chat_id,
        "text": text,
        "description": desc,
        "objective": desc,
        "expected_output": expected_output,
        "constraints": constraints,
        "role": role,
        "context": task_context,
        "depth": depth,
        "root_task_id": root_task_id,
        "session_id": session_id,
        "actor_id": actor_id,
        "delegation_role": delegation_role,
        "memory_mode": memory_mode,
        "drive_root": drive_root,
        "child_drive_root": child_drive_root,
        "budget_drive_root": budget_drive_root,
        "root_cost_ceiling_usd": root_cost_ceiling_usd,
        "task_constraint": task_constraint,
        "required_capabilities": required_capabilities,
        "workspace_root": workspace_root,
        "workspace_mode": workspace_mode,
        "project_id": project_id,
        "allowed_resources": allowed_resources,
        "task_contract": task_contract,
        "depth_provenance": depth_provenance,
        "model_lane": requested_model_lane,
        "requested_model_lane": requested_model_lane,
        "parent_model_lane": parent_model_lane,
        "required_model_lane": required_model_lane,
        "requested_executor": requested_executor,
        "task_group_id": task_group_id,
        "task_group": task_group,
        "subagent_envelope": subagent_envelope,
        "configured_subagent": configured_subagent,
        "parent_cognitive_route": parent_cognitive_route,
        "metadata": {
            "parent_task_id": parent_id,
            "root_task_id": root_task_id,
            "session_id": session_id,
            "actor_id": actor_id,
            "delegation_role": delegation_role,
            "role": role,
            "memory_mode": memory_mode,
            "task_constraint": task_constraint,
            "required_capabilities": required_capabilities,
            "child_drive_root": child_drive_root,
            "workspace_root": workspace_root,
            "workspace_mode": workspace_mode,
            "allowed_resources": allowed_resources,
            "task_contract": task_contract,
            "depth_provenance": depth_provenance,
            "model_lane": requested_model_lane,
            "requested_model_lane": requested_model_lane,
            "parent_model_lane": parent_model_lane,
            "requested_executor": requested_executor,
            "task_group_id": task_group_id,
            "task_group": task_group,
            "subagent_envelope": subagent_envelope,
            "configured_subagent": configured_subagent,
            "parent_cognitive_route": parent_cognitive_route,
            "root_cost_ceiling_usd": root_cost_ceiling_usd,
        },
    }
    if not drive_root:
        task.pop("drive_root", None)
    if not budget_drive_root:
        task.pop("budget_drive_root", None)
    if task_constraint is None:
        task.pop("task_constraint", None)
        task["metadata"].pop("task_constraint", None)
    if not required_capabilities:
        task.pop("required_capabilities", None)
        task["metadata"].pop("required_capabilities", None)
    if parent_id:
        task["parent_task_id"] = parent_id
    if delegation_role != "subagent":
        # Generic/root schedule events do not have the subagent transition's
        # own exact-id preflight. Ask the queue to perform its strict lookup
        # under the queue lock before this payload can gain custody.
        task["_require_unique_task_id"] = True
    return task

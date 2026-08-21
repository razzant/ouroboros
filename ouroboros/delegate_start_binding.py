"""Fresh delegated-start binding and wire construction.

The facade keeps the nanny verbs small; this seam owns the one-time operation that
turns a host authority target into a snapshot, project registration, idempotency key,
and canonical run-start body. Retries never call it: they replay their stored body.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional

from ouroboros import delegate_custody as custody


class FreshStartBinding(NamedTuple):
    request_body: Dict[str, Any]
    root: str
    execution_root: str
    invocation_id: str
    key: str
    project_id: str
    owned_project_id: str
    project_persistent: bool
    seconds: int
    snapshot_id: str
    target_root: str
    baseline_sha: str
    authority_source: str
    resource_ref: Dict[str, Any]


def start_request(ctx: Any, route: Any, authority: Any, root: str, text: str,
                  seconds: int, instructions: str, *, execution_root: str = "") -> Dict[str, Any]:
    """Build one canonical POST body from the already-derived run shape."""
    request: Dict[str, Any] = {
        "prompt": text,
        "instructions": instructions,
        "authPreference": "subscription",
        "mode": authority.mode,
        "scope": {"kind": "project", "root": root},
        "harnesses": [route.route_id],
        "primaryHarness": route.route_id,
        "access": authority.access,
    }
    if authority.isolation:
        request["execution"] = {
            "isolation": authority.isolation,
            "delegated": authority.delegated,
        }
        if execution_root:
            request["execution"]["workspaceRoot"] = execution_root
    if route.model:
        request["model"] = route.model
    if route.effort:
        request["effort"] = route.effort
    if route.profile_id:
        request["credentialProfileId"] = route.profile_id
    if seconds:
        request["maxSeconds"] = seconds
    return request


def prepare_fresh_start(
    ctx: Any, drive: Any, gateway: Any, route: Any, authority: Any, actor: Dict[str, Any],
    text: str, max_seconds: Optional[int], payload_auth: Optional[Dict[str, Any]],
    workspace_root_supported: bool, *,
    host_instructions: Callable[..., str], assignment_instructions: Callable[..., str],
    bounded_max_seconds: Callable[..., int],
) -> tuple[Optional[FreshStartBinding], str]:
    """Provision and describe a fresh mutating or read-only start."""
    from ouroboros.tools import delegate_integration as integration

    if payload_auth is not None:
        record_auth = payload_auth
    else:
        record_auth, root_error = integration._mutation_authority(ctx, authority)
        if root_error:
            return None, root_error
    invocation_id = custody.new_invocation_id()
    root = str(record_auth["target_root"])
    target_root = ""
    authority_source = ""
    snapshot_id = ""
    execution_root = ""
    baseline_sha = ""
    resource_ref: Dict[str, Any] = {}
    if authority.access == "workspace_write":
        target_root = str(record_auth["target_root"])
        authority_source = str(record_auth["source"])
        if authority_source == "skill_payload":
            snapshot, snap_error = integration._provision_payload_snapshot(
                ctx, drive, record_auth, invocation_id)
        else:
            snapshot, snap_error = integration._provision_snapshot(
                ctx, drive, target_root, invocation_id)
        if snap_error:
            return None, snap_error
        snapshot_id, baseline_sha, execution_root = (
            snapshot.snapshot_id, snapshot.baseline_sha, snapshot.path)
        if not workspace_root_supported:
            root = execution_root
        resource_ref = dict(record_auth.get("resource_ref") or {})
    existing_project = gateway.find_project_id(root)
    project_id = existing_project or gateway.register_project(root)
    owned_project_id = "" if existing_project else project_id
    project_persistent = bool(workspace_root_supported and authority.access == "workspace_write")
    assignment = ("" if bool(actor.get("compiled_work_order"))
                  else assignment_instructions(ctx))
    # The caller supplies the ordinary assignment builder to keep this seam independent
    # of the facade's prompt constants; payload runs still receive their resource name.
    instructions = host_instructions(
        authority, assignment,
        payload_skill=(str((record_auth.get("resource_ref") or {}).get("skill_name") or "")
                       if payload_auth is not None else ""),
    )
    seconds = bounded_max_seconds(ctx, max_seconds)
    key = custody.idempotency_key(
        getattr(ctx, "task_id", ""), route.route_id, authority.access,
        authority.mode, authority.isolation, root,
        execution_root if workspace_root_supported else "", text, instructions,
    )
    request_body = start_request(
        ctx, route, authority, root, text, seconds, instructions,
        execution_root=(execution_root if workspace_root_supported else ""),
    )
    return FreshStartBinding(
        request_body, root, execution_root, invocation_id, key, project_id,
        owned_project_id, project_persistent, seconds, snapshot_id, target_root,
        baseline_sha, authority_source, resource_ref,
    ), ""


__all__ = ["FreshStartBinding", "prepare_fresh_start", "start_request"]

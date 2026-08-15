"""Where a task RUNS: placement admission, sealing, and attachment export.

Split out of `gateway/tasks` because these five answer one question the rest of the
module does not ask — which machine this task's files live on — and they answer it in
a fixed order: read the project's placement, fold in an explicit executor ref,
preflight it, seal it into the task metadata, and only then export attachments to
wherever the answer said. Keeping them together keeps that order readable; the
creation surface just calls the sequence.

The seal matters more than the size: once written, a task's placement is not a
suggestion the queue may re-derive. `workspace_ref.isolate_sealed_placement` deep-copies
it precisely so a creator holding the same dict cannot retarget a task that is already
admitted.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional
from ouroboros.headless import (
    write_workspace_preflight_artifact,
)
from ouroboros.tool_access import path_is_relative_to, paths_overlap_casefold
from ouroboros.workspace_preflight import (
    collect_workspace_preflight,
    summarize_workspace_preflight,
)
from ouroboros.workspace_admission import (
    placement_preflight_summary,
)
from ouroboros.workspace_executor import normalize_executor_ref
from ouroboros.workspace_ref import (
    SEALED_WORKSPACE_REF_KEY,
    LocalWorkspaceRef,
    WorkspaceRef,
    workspace_display_root,
)


def _resolve_workspace_root(
    value: Any,
    *,
    system_repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
) -> Optional[pathlib.Path]:
    """Delegates to the admission SSOT (v6.58.0): the gateway and the promote path
    validate a workspace root through ONE function (workspace_admission), so the two
    surfaces can never drift. WorkspaceRootError subclasses ValueError, so existing
    `except ValueError` call sites keep working unchanged.

    RWS v2: the SSOT now returns the SEALED placement; this door keeps its Home-path
    signature for the local request shape and refuses TYPED for a remote one (a
    remote placement is inherited from the project, never named in the request)."""
    from ouroboros.workspace_admission import local_admitted_path, validate_workspace_root

    return local_admitted_path(
        validate_workspace_root(value, system_repo_dir=system_repo_dir, drive_root=drive_root)
    )

def _normalize_attachments(value: Any) -> List[Dict[str, str]]:
    if not value:
        return []
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            path = str(item.get("path") or "").strip()
            label = str(item.get("label") or item.get("display_name") or pathlib.Path(path).name).strip()
        else:
            path = str(item or "").strip()
            label = pathlib.Path(path).name
        if path:
            out.append({"path": path, "label": label})
    return out


def _project_placement_payload(drive_root: pathlib.Path, project_id: Any) -> Optional[Dict[str, Any]]:
    """The project's serialized REMOTE placement, or ``None`` for a local/absent project.

    The registry read is skipped for a malformed project_id so the existing
    `explicit_project_id_ok` refusal downstream stays the one that reports it.

    The read goes through the registry's own sealed reader, so a durable placement
    this build cannot honor raises (surfaced as a typed 400) instead of reading as
    absent — reading it as absent would run a remote project's task on Home.
    """
    pid = str(project_id or "").strip()
    if not pid:
        return None
    from ouroboros.project_facts import explicit_project_id_ok

    if not explicit_project_id_ok(str(project_id or "")):
        return None
    from ouroboros.projects_registry import get_project, project_placement

    placement = project_placement(get_project(drive_root, pid) or {})
    return None if placement is None else placement.to_payload()


def _fold_request_executor_ref(
    body: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    workspace_ref: Optional[WorkspaceRef],
    workspace_root: Optional[pathlib.Path],
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
) -> str:
    """Validate a request's docker `executor_ref` and fold it into metadata.

    Returns "" on success or the 400 detail. An ssh placement is refused outright:
    ``ExecutorRef`` is a DERIVED projection of the sealed placement (§3.1), so a
    per-task override would be a second placement authority — and only one is stored.
    """
    raw_executor_ref = body.get("executor_ref")
    if not isinstance(raw_executor_ref, dict) or not raw_executor_ref:
        return "executor_ref must be a JSON object"
    if getattr(workspace_ref, "kind", "") == "ssh":
        return "executor_ref is derived from the project's remote placement and cannot be set per task"
    if workspace_root is None:
        return "executor_ref requires an external workspace_root"
    try:
        normalized_executor = normalize_executor_ref(raw_executor_ref)
    except ValueError as exc:
        return str(exc)
    if normalized_executor is None:
        return ""
    for mapping in normalized_executor.mappings:
        for protected_root, label in ((repo_dir, "Ouroboros system repo"), (drive_root, "Ouroboros data drive")):
            if paths_overlap_casefold(mapping.host_path, protected_root):
                return f"executor_ref mapping must not overlap the {label}"
    if not any(path_is_relative_to(workspace_root, mapping.host_path) for mapping in normalized_executor.mappings):
        return "executor_ref mappings must cover workspace_root"
    metadata["executor_ref"] = {
        "type": normalized_executor.kind,
        "id": normalized_executor.executor_id,
        "network": normalized_executor.network,
        "workspace_host_path": str(normalized_executor.mappings[0].host_path),
        "workspace_backend_path": normalized_executor.mappings[0].backend_path,
        "container_name": normalized_executor.container_name,
        "path_mappings": [
            {"host_path": str(mapping.host_path), "backend_path": mapping.backend_path}
            for mapping in normalized_executor.mappings
        ],
    }
    return ""


def _seal_placement_into_metadata(
    metadata: Dict[str, Any],
    workspace_ref: Optional[WorkspaceRef],
    *,
    drive_root: pathlib.Path,
    task_id: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """SEAL the admitted placement into task metadata and collect its preflight.

    The seal is written ONCE (RWS v2, decision D "immutable task placement"): every later
    reader — worker, dispatcher, guards, snapshot restore — READS this ref instead of
    resolving a placement again, which is exactly why a project rebind after admission
    cannot move a task that is already running.

    Returns ``(artifacts, preflight_summary)``.
    """
    artifacts: List[Dict[str, Any]] = []
    if workspace_ref is None:
        return artifacts, {}
    display_root = workspace_display_root(workspace_ref)
    metadata[SEALED_WORKSPACE_REF_KEY] = workspace_ref.to_payload()
    metadata["workspace_root"] = display_root
    if workspace_ref.kind != "local":
        # A remote workspace's preflight is a TARGET fact bundle, not a Home walk — a
        # Home walk here would not merely be slow, it would answer about the wrong
        # filesystem. It goes through the admission SSOT's ONE preflight door: this
        # handler used to carry its own copy of the remote arm, which is exactly how
        # two answers to one question start disagreeing.
        summary = placement_preflight_summary(
            workspace_ref, project_id=str(metadata.get("project_id") or "")
        )
        metadata["workspace_preflight"] = summary
        return artifacts, summary
    workspace_root = workspace_ref.home_path()
    try:
        preflight = collect_workspace_preflight(workspace_root)
        summary = summarize_workspace_preflight(preflight)
        artifacts.append(write_workspace_preflight_artifact(drive_root, task_id, preflight))
    except Exception as exc:
        summary = {
            "schema_version": 1,
            "workspace_root": str(workspace_root),
            "error": f"{type(exc).__name__}: {exc}",
        }
    metadata["workspace_preflight"] = summary
    return artifacts, summary


def _place_attachments(
    attachments: Any,
    workspace_ref: Any,
    *,
    metadata: Dict[str, Any],
    drive_root: Any,
    task_id: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Stage the task's attachments where the task will actually run.

    v6.52.0 (P1): they land in the SAME drive the task reads from at runtime — the
    child drive when forked/empty, else the shared drive — and the returned manifest
    renders READY ``read_file(root='artifact_store', ...)`` lines plus native image
    blocks.

    For a remote placement (D12 channel `attachment_stage`) Home is the SOURCE: the
    export policy runs before any byte leaves, the surviving set is uploaded as
    content-addressed blobs, and the returned manifest carries the ONE target-side
    path a remote `run_command` can open. The remote manifest REPLACES the Home one,
    so the prompt advertises addresses that exist on the host doing the work.

    The third return value is the omission sentence. Both placements produce it from
    the same rows, because "why is my attached file not here" has one answer whether
    Home staging or the export policy dropped it.
    """

    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.remote_task_files import (
        attachment_omission_note,
        export_task_attachments,
        staging_omission_rows,
    )

    omissions: List[Dict[str, str]] = []
    manifest = stage_task_attachments(
        drive_root, task_id, _normalize_attachments(attachments), omitted=omissions,
    )
    note = attachment_omission_note(staging_omission_rows(omissions))
    if getattr(workspace_ref, "kind", "") == "ssh":
        exported = export_task_attachments(
            workspace_ref,
            drive_root=drive_root,
            task_id=task_id,
            manifest=manifest,
            staging_omissions=omissions,
        )
        manifest = list(exported["attachments"])
        note = str(exported.get("note") or "")
        metadata["_remote_attachment_manifest"] = manifest
        if exported["excluded_count"]:
            metadata["_remote_attachment_omissions"] = list(exported["excluded"])
    return manifest, [row for row in manifest if row.get("is_image")], note


def _admit_task_placement(
    body: Dict[str, Any],
    *,
    system_repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
) -> Optional[WorkspaceRef]:
    """Resolve the task's placement ONCE, through the admission SSOT, and return it sealed.

    A remote placement is INHERITED FROM THE PROJECT and can never be chosen per task
    (donor `test_remote_admission.py::test_task_api_inherits_remote_placement_only_from_project`).
    That is not a convenience: the project registry is where `routing_generation` and the
    connection's trust identity live, so a task that picked its own remote target would
    be a placement with no generation to fence and no trust to revalidate. The client
    therefore names a `project_id`; the placement follows from it.

    Raises ``ValueError`` (surfaced as 400) for a malformed request or an unusable
    placement, and the ValueError SUBCLASS ``RemoteWorkspaceUnavailableError`` when a
    remote project's target cannot be reached — refusing is the only alternative to
    running the task on the wrong host. The caller must catch the subclass FIRST: it
    is not a bad request but an unreachable one, and it carries its own status code
    (503), typed code, and next step.
    """
    from ouroboros.workspace_admission import validate_workspace_root

    # BOTH DOORS, one loop. The body was checked here and `metadata` was not, so the
    # rule held on the spelling a caller reads about and not on the spelling a caller
    # guesses: `metadata.workspace_ref` was SILENTLY STRIPPED by
    # `_RESERVED_METADATA_KEYS`, and `metadata.connection_id` was not even in that set,
    # so it was stored and then ignored by everything. Both outcomes are worse than a
    # refusal for the same reason `project_id`-in-metadata is refused three screens
    # down: the owner named a placement, the task ran somewhere else, and nothing said
    # so. `docs/ARCHITECTURE.md` promised the typed 400 for `metadata` too, so the
    # document was right and the code was short of it.
    raw_metadata = body.get("metadata")
    in_metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    for field in ("workspace_ref", "connection_id"):
        if field in body or field in in_metadata:
            raise ValueError(
                f"{field} is not a task field: a remote placement is inherited from "
                "project_id, never chosen per task"
            )
    local_root = _resolve_workspace_root(
        body.get("workspace_root"), system_repo_dir=system_repo_dir, drive_root=drive_root
    )
    project_placement = _project_placement_payload(drive_root, body.get("project_id"))
    if project_placement is None:
        return LocalWorkspaceRef(local_root=str(local_root)) if local_root is not None else None
    if local_root is not None:
        raise ValueError(
            "workspace_root cannot be combined with a remote project: the placement is "
            "inherited from project_id"
        )
    return validate_workspace_root(
        project_placement, system_repo_dir=system_repo_dir, drive_root=drive_root
    )


__all__ = [
    "_project_placement_payload",
    "_fold_request_executor_ref",
    "_seal_placement_into_metadata",
    "_place_attachments",
    "_admit_task_placement",
]

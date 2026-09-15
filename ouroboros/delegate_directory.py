"""Directory work products use the engine's existing capture and delivery owner.

The host retains complete immutable artifact bytes and ordinary run custody. It
does not make a second execution copy, invent a Git baseline, or apply files itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from hashlib import sha256
import uuid

from ouroboros import delegate_custody as custody
from ouroboros.artifacts import stream_artifact_file
from ouroboros.utils import atomic_write_json


DIRECTORY_OPTIONS_NEED_WRITE = (
    "copy and non-empty scope_paths need a write-capable child on an ordinary folder; "
    "a read-only child reads the folder as it is, so omit directory_strategy and scope_paths "
    "(direct with no scope is the same as omitting them) — give the child a write surface only "
    "if it genuinely has to change files."
)


def is_directory_run(entry) -> bool:
    return bool(entry and isinstance(entry.resource_ref, dict)
                and entry.resource_ref.get("workspace_kind") == "directory")


def default_shaped_directory_options(strategy, scope_paths) -> bool:
    """True when the request names only the documented default and nothing else.

    ``direct`` IS the documented meaning of omitting the strategy, and an empty
    ``scope_paths`` selects nothing, so on a shape that never reads these two values
    naming either is the same request as passing neither. It is NOT a general
    identity: a write-capable child's ``[]`` still rides the wire as ``scopePaths:
    []``, its own attested "capture nothing", which only the engine interprets.
    Geometry becomes a REAL request at ``copy`` or a non-empty footprint.
    """
    return strategy in (None, "direct") and not scope_paths


def blocked_geometry_refusal(ctx, authority, selector_root, strategy, scope_paths):
    """Typed pre-POST refusal for geometry this shape can never serve, else ``None``.

    A read-only child and a payload selector never open the ordinary-folder session,
    so a real geometry request is refused before the daemon call — the parent repairs
    it in one move at $0, and the refusal names that repair instead of recommending a
    mutating session for an audit. ``definitely_unrun`` is the producer's own verdict
    (nothing was started), so the host ends the child on the zero-spend terminal path
    rather than waking the model with a startup fault it cannot act on.

    The documented default named explicitly is NOT such a request: it asks for exactly
    what omission asks for, so the caller proceeds as the omitted form. Nothing has to
    be unset for that — the folder branch that reads these two values is reachable only
    from the write-capable non-selector shape this refusal does not touch.

    It lives HERE, not at its one call site, because this module already owns
    ``directory_execution``'s geometry validation and because ``_delegate_start`` sits
    at the 300-line function cap on a shrink-only band path.
    """
    if not (selector_root or getattr(authority, "access", "") != "workspace_write"):
        return None
    if default_shaped_directory_options(strategy, scope_paths):
        return None
    from ouroboros.delegate_evidence import record_start_blocked
    from ouroboros.delegate_shared import _fail

    record_start_blocked(ctx, str(getattr(ctx, "task_id", "") or ""), "directory_execution_unavailable")
    return _fail("delegate_start", "directory_execution_unavailable",
                 DIRECTORY_OPTIONS_NEED_WRITE, definitely_unrun=True)


def directory_execution(gateway, strategy=None, scope_paths=None):
    """Compile the already-authorized folder strategy against engine capability."""
    kinds = gateway.agent_capabilities().get("mutability", {}).get("workspaceKinds", [])
    if "directory" not in kinds:
        raise ValueError("the serving engine does not support directory work products")
    strategy = strategy or "direct"
    if strategy not in {"direct", "copy"}:
        raise ValueError("directory_strategy must be direct or copy")
    if scope_paths is not None and (not isinstance(scope_paths, list) or any(
        not isinstance(path, str) or not path or Path(path).is_absolute()
        or ".." in Path(path).parts for path in scope_paths
    )):
        raise ValueError("scope_paths must contain relative paths within the selected folder")
    if strategy == "copy" and not scope_paths:
        raise ValueError("choose scope_paths for a copy; use '.' for the complete folder")
    execution = {"workspaceKind": "directory", "isolation": "live" if strategy == "direct" else "envelope"}
    if scope_paths is not None:
        execution["scopePaths"] = list(scope_paths)
    reference = {"workspace_kind": "directory", "strategy": strategy,
                 "scopePaths": list(scope_paths or [])}
    return execution, reference


def prepare_directory_execution(ctx, gateway, target_root, authority, strategy, scope_paths):
    """Keep the selected folder's physical validation and engine preparation together."""
    from dataclasses import replace
    from ouroboros.workspace_admission import validate_workspace_root

    validate_workspace_root(target_root, system_repo_dir=ctx.repo_dir, drive_root=custody.custody_root(ctx))
    options, reference = directory_execution(gateway, strategy, scope_paths)
    return replace(authority, isolation=options["isolation"]), options, reference


def _download(gateway, run_id, source, destination, expected=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as sink:
            measured = gateway.stream_run_artifact(run_id, source, sink, expected=expected)
        temporary.replace(destination)
        return {"kind": "workspace_file_output", "name": destination.name,
                "path": str(destination), **measured, "immutable": True}
    finally:
        temporary.unlink(missing_ok=True)


def capture_directory_result(drive, entry, gateway):
    """Capture only a settled run; retain source/actual execution addresses separately."""
    if not entry.settled:
        raise ValueError("the directory run has not settled")
    cap_dir = custody.delegated_capture_dir(drive, entry.task_id, entry.run_id)
    manifest_path = cap_dir / "workspace_patch.json"
    if entry.patch_captured and manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        for record in saved.get("file_outputs", []):
            stream_artifact_file(cap_dir / record["name"], expected=record)
        return saved
    detail = gateway.get_run(entry.run_id)
    product = detail.get("workProduct") or {}
    if product.get("kind") != "files":
        raise ValueError("the completed run has no file work product")
    source = (product.get("files") or {}).get("manifest")
    digest = str((product.get("meta") or {}).get("manifest_sha256") or "").removeprefix("sha256:")
    if not source or len(digest) != 64:
        raise ValueError("the file work product has no verifiable manifest")
    raw_path = cap_dir / "engine-files-manifest.json"
    raw = _download(gateway, entry.run_id, source, raw_path, {"sha256": digest})
    original = json.loads(raw_path.read_text())
    if (original.get("version") != 1 or not original.get("executionRoot")
            or Path(original.get("sourceRoot") or "").resolve() != Path(entry.target_root).resolve()
            or not isinstance(original.get("entries"), list)):
        raise ValueError("engine directory manifest has an invalid source or execution binding")
    direct = entry.resource_ref["strategy"] == "direct"
    if original.get("isolation") != ("live" if direct else "envelope"):
        raise ValueError("engine directory strategy differs from the recorded request")
    if not direct and original.get("complete") is not True:
        raise ValueError("the selected execution copy was not completely captured")
    records = [raw]
    for index, row in enumerate(original["entries"]):
        for side in ("before", "after"):
            state = row.get(side)
            if not isinstance(state, dict) or state.get("kind") != "file":
                continue
            if not state.get("artifactPath"):
                if direct and side == "before":
                    continue
                raise ValueError(f"file {side} has no complete reference: {row.get('path')}")
            record = _download(gateway, entry.run_id, state["artifactPath"],
                               cap_dir / f"{side}-{index}", state)
            records.append({**record, "source_path": row["path"], "side": side})
    manifest = {
        "schema_version": 1, "capture_kind": "engine_directory", "status": "ready",
        "workspace_root": original["executionRoot"], "authority_target_root": entry.target_root,
        "engine_manifest": raw, "engine_manifest_sha256": digest,
        "complete": original.get("complete"), "scope_paths": original.get("scopePaths"),
        "strategy": entry.resource_ref["strategy"], "file_outputs": records,
        "apply_state": (product.get("meta") or {}).get("apply_state"),
        "note": ("Direct effects are already in the source folder; no separate apply or full rollback."
                 if direct else "The engine copy is not applied. Inspect the complete manifest and files before disposition."),
    }
    atomic_write_json(manifest_path, manifest, trailing_newline=True)
    custody.record_patch_captured(drive, entry, status="ready", sha256=digest,
                                 capture_dir=str(cap_dir), execution_root=original["executionRoot"])
    return manifest


def directory_capture_block(drive, entry, manifest):
    cap_dir = custody.delegated_capture_dir(drive, entry.task_id, entry.run_id)
    return {"status": manifest["status"], "capture_kind": "engine_directory",
            "execution_root": manifest["workspace_root"], "authority_target_root": entry.target_root,
            "manifest_artifact": str(cap_dir / "workspace_patch.json"),
            "manifest_read": {"root": "artifact_store", "path": f"delegated_runs/{cap_dir.name}/workspace_patch.json"},
            "file_outputs": manifest["file_outputs"], "patch_artifact": None,
            "complete": manifest["complete"], "scope_paths": manifest["scope_paths"],
            "strategy": manifest["strategy"], "note": manifest["note"]}


def integrate_directory_result(ctx, entry, decision, reason, gateway, *, acknowledge_ambiguous=False,
                               paths=None, orphan=False):
    """Use engine CAS/disposition, retaining unknown effects under the same host intent."""
    drive = custody.custody_root(ctx)
    manifest = capture_directory_result(drive, entry, gateway)
    from ouroboros.delegate_shared import orphan_apply_target_ok
    active, target = Path(ctx.active_repo_dir()).resolve(), Path(entry.target_root).resolve()
    if decision == "apply" and not (orphan_apply_target_ok(target, active) if orphan else active == target):
        return "⚠️ INTEGRATE_DELEGATED_TARGET_MISMATCH: the run belongs to another folder."
    if paths is not None and (not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not path for path in paths)):
        return "⚠️ TOOL_ARG_ERROR (integrate_delegated_patch): paths must be a nonempty list of captured file paths."
    if decision == "reject":
        if manifest["strategy"] == "direct":
            return "⚠️ INTEGRATE_DIRECTORY_ALREADY_APPLIED: direct effects remain in the folder; rejecting a report cannot undo them."
        key = sha256(f"discard:{entry.run_id}:{manifest['engine_manifest_sha256']}".encode()).hexdigest()
        receipt = gateway.decide_run(entry.run_id, {"action": "discard"}, idempotency_key=key)
        if receipt.get("accepted") is not True or receipt.get("status") != "discarded":
            return "⚠️ INTEGRATE_DELEGATED_DISCARD_UNCONFIRMED: engine disposition was not confirmed; all results remain retained."
        disposition = "rejected"
    elif manifest["strategy"] == "direct":
        if manifest.get("apply_state") not in {"applied", "applied_review_blocked"}:
            return "⚠️ INTEGRATE_DIRECTORY_UNCONFIRMED: the engine has not confirmed direct effects; inspect its retained result."
        receipt = {"source": "engine_work_product", "apply_state": manifest["apply_state"]}
        disposition = "applied"
    else:
        if entry.patch_apply_pending and not acknowledge_ambiguous:
            return "⚠️ INTEGRATE_DELEGATED_APPLY_AMBIGUOUS: an earlier apply awaits reconciliation; inspect it before retrying."
        request = {"target": {"kind": "original_project"}, "mode": "apply"}
        if paths is not None:
            request["paths"] = paths
        key = sha256(json.dumps([entry.run_id, manifest["engine_manifest_sha256"], request], sort_keys=True).encode()).hexdigest()
        if entry.patch_apply_pending and getattr(entry, "patch_apply_key", "") != key:
            return "⚠️ INTEGRATE_DELEGATED_APPLY_AMBIGUOUS: retry the original path selection; its apply request still has an unknown outcome."
        if not custody.record_patch_apply_started(drive, entry, target_root=entry.target_root,
                                                  apply_idempotency_key=key):
            return "⚠️ INTEGRATE_INTENT_UNWRITTEN: the apply intent could not be saved; nothing was submitted."
        receipt = gateway.apply_run(entry.run_id, request, idempotency_key=key)
        if receipt.get("applied") is not True or receipt.get("refused") is True:
            if receipt.get("refused") is True:
                custody.record_patch_apply_resolved(drive, entry, reason="engine_refused")
            return "⚠️ INTEGRATE_DELEGATED_APPLY_UNCONFIRMED: " + json.dumps(receipt, ensure_ascii=False)
        disposition = "applied"
        summary = gateway.get_run(entry.run_id).get("summary") or {}
        state = (summary.get("result") or {}).get("applyState")
        if state not in {"applied", "applied_review_blocked"}:
            custody.record_patch_apply_resolved(drive, entry, reason="engine_partial_delivery", engine_receipt=receipt)
            return json.dumps({"status": "partially_applied", "run_id": entry.run_id,
                               "engine_receipt": receipt, "note": "Selected results are applied; remaining results are retained for explicit apply or reject."}, ensure_ascii=False)
    from ouroboros.tools.patch_verdict import write_patch_verdict
    write_patch_verdict(ctx, f"run_{entry.run_id}", outcome=disposition, reason=reason, files=[],
                        manifest=manifest, applied=disposition == "applied", conflicts=[], protected=[],
                        target=entry.target_root)
    if not custody.record_patch_disposed(drive, entry, disposition=disposition,
                                         reason=reason, engine_receipt=receipt):
        return "⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: the engine completed the operation but its host receipt could not be saved."
    return json.dumps({"status": disposition, "run_id": entry.run_id, "target": entry.target_root,
                       "engine_receipt": receipt,
                       "note": "Original effects are unchanged." if disposition == "rejected" else
                       "Results are already in the target folder; this disposition does not commit or publish."}, ensure_ascii=False)

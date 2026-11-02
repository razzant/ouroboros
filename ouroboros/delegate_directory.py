"""Directory work products use the engine's existing capture and delivery owner.

The host retains complete immutable artifact bytes and ordinary run custody. It
does not make a second execution copy, invent a Git baseline, or apply files itself.
An explicit ``workspace="in_place"`` Git writer, which works in its starter's own
folder and so has nothing to capture, is recognized and projected here too.
"""
from __future__ import annotations

import json
from pathlib import Path
from hashlib import sha256
import subprocess
import uuid

from ouroboros import delegate_custody as custody
from ouroboros.configured_subagents import SESSION_ACCESS_PROFILES
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


IN_PLACE = "in_place"  # a Git writer its starter put directly in its own write root (owner 25.10, Q2-A)


def in_place_ref():
    return {"workspace_kind": "git", "strategy": IN_PLACE}


def is_in_place_run(record) -> bool:
    """A custody entry, request row or invocation record of an explicitly in-place Git writer."""
    ref = record.get("resource_ref") if isinstance(record, dict) else getattr(record, "resource_ref", None)
    return isinstance(ref, dict) and ref.get("workspace_kind") == "git" and ref.get("strategy") == IN_PLACE


def in_place_refusal(authority, selector_root, workspace, mode, strategy, scope_paths):
    """The shape rule for ``workspace``: only a Git writer may work in place."""
    from ouroboros.delegate_shared import _fail

    if workspace in (None, "", "copy"):
        return None
    reason = ("workspace must be 'copy' or 'in_place'" if workspace != IN_PLACE else
              "a skill payload is always edited in its private snapshot" if selector_root else
              "an evidence-only investigation or a read-only run never writes your folder"
              if authority.investigation or authority.access == "readonly" or mode in ("ask", "plan") else
              "directory options apply to ordinary folders; use directory_strategy='direct' there"
              if not default_shaped_directory_options(strategy, scope_paths) else "")
    return _fail("delegate_start", "workspace_in_place_unavailable",
                 f"workspace='in_place' was refused: {reason}. Nothing was started.", definitely_unrun=True) if reason else None


def in_place_start(adopted_copy, target_root, service=None):
    """(snapshot, refusal, resource_ref, binding) for an in-place Git writer: no copy is made.
    ``service`` makes its disposable scratch (``subagent_work_order.session_service_directories``),
    called only once the start is accepted, so a refusal leaves nothing behind."""
    from ouroboros.delegate_shared import _fail
    from ouroboros.delegate_start_instructions import in_place_binding_instruction

    if adopted_copy is not None or not (Path(target_root) / ".git").exists():
        detail = ("this continuation resumes an undisposed private copy; continue it there (omit workspace)"
                  if adopted_copy is not None else "the write root is not a Git working tree; ordinary folders "
                  "use directory_strategy='direct'")
        return None, _fail("delegate_start", "workspace_in_place_unavailable",
                           f"workspace='in_place' was refused: {detail}. Nothing was started.", definitely_unrun=True), {}, ""
    return None, "", in_place_ref(), in_place_binding_instruction(target_root, service() if service else None)


def in_place_block(entry):
    """The terminal projection of an in-place run: nothing was captured and nothing is owed."""
    target = str(entry.target_root or entry.execution_root or "")
    return {"status": IN_PLACE, "capture_kind": IN_PLACE, "disposition": IN_PLACE, "execution_root": target,
            "authority_target_root": target, "patch_artifact": None, "manifest_artifact": None,
            "note": (f"This run worked directly in {target} by its starter's choice (workspace='in_place'): its edits "
                     "are already there. There is nothing to apply or reject; inspect them with vcs_read diff and undo "
                     "through git, which also reverts any uncommitted edit of your own there.")}


def started_receipt(payload, resource_ref, target_root, *, durable, investigation):
    """Receipt facts and note of a folder or in-place start (delegate_start's started result)."""
    lead = "CUSTODY IS NOT DURABLE. " if not durable else ""
    if is_in_place_run({"resource_ref": resource_ref}):
        payload.update(workspace=IN_PLACE, authority_target_root=target_root, execution_root=target_root)
        payload["note"] = (lead + f"This run works DIRECTLY in your write root {target_root}, as you chose "
                           "(workspace='in_place'): its edits land there as it makes them. Nothing is captured for apply "
                           "or reject (integrate_delegated_patch refuses); undo through git. Do not edit the same files "
                           "meanwhile and do not start a second in-place writer in this folder: their edits would mix. "
                           "When this folder is the Ouroboros body, its protected paths are not checked for this writer "
                           "(the commit gate still checks them). Use delegate_wait for its result.")
        return
    direct = resource_ref.get("strategy") == "direct"
    payload.update(authority_target_root=target_root,
                   execution_root=target_root if direct else None,
                   directory_strategy=resource_ref["strategy"], scope_paths=resource_ref["scopePaths"])
    payload["note"] = (
        lead
        + ("This run writes directly in the selected folder; effects are already on site, without a full rollback promise. "
           if direct else "The engine prepares a separate copy of the selected inputs; its actual execution directory is not yet observed. ")
        + "Use delegate_wait for its complete file manifest and result. "
        + ("This evidence-only investigation's results are kept as evidence, never applied "
           "(integrate_delegated_patch refuses); never let it commit." if investigation
           else "Copy results use integrate_delegated_patch for apply or reject.")
    )


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


def git_directory_options_refusal(target_root, strategy, scope_paths, *, git_workspace=False):
    """One argument-shape check for scheduling and direct Git session starts."""
    if (git_workspace or (Path(target_root) / ".git").exists()) and not default_shaped_directory_options(strategy, scope_paths):
        return "Directory options apply to ordinary folders; Git workspaces keep their snapshot contract."
    return ""


def snapshot_checkout(ctx, target_root, authority) -> str:
    """The Git worktree a session start snapshots, or "" for the ordinary-folder session.

    A worktree root is its own checkout. An evidence-only investigation (#1609) started
    in a Git SUBDIRECTORY (schedule_subagent's named starting folder) snapshots the
    enclosing worktree, so every checking input is present, and starts in the matching
    folder of that private clone. A writer keeps the worktree-root rule, and a worktree
    overlapping the data drive is never cloned: the folder session then refuses it typed.
    """
    from ouroboros.platform_layer import bootstrap_process_path
    from ouroboros.tool_access import paths_overlap_casefold
    from ouroboros.workspace_admission import has_git_metadata

    root = Path(target_root)
    if (root / ".git").exists():
        return str(root)
    if not getattr(authority, "investigation", False) or not has_git_metadata(root.resolve()):
        return ""
    bootstrap_process_path()
    try:
        probe = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(root),
                               capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    top = Path(probe.stdout.strip()).resolve() if probe.returncode == 0 and probe.stdout.strip() else None
    if top is None or paths_overlap_casefold(top, Path(custody.custody_root(ctx)).resolve()):
        return ""
    return str(top)


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
    if not (selector_root or getattr(authority, "access", "") not in SESSION_ACCESS_PROFILES):
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


def prepare_directory_execution(ctx, gateway, target_root, authority, strategy, scope_paths, staged=""):
    """Keep the selected folder's physical validation and engine preparation together.

    ``staged`` is a folderless Project investigation's per-invocation input copy, made
    from the live scratch by the input owner (``delegate_readonly_inputs``): the engine
    copies THAT host-prepared folder, recorded as ``source_root``, so the scratch itself
    is never the engine's source or a workspace. A folder no private copy can serve
    tells an investigation its one repair, the explicit readonly lowering.
    """
    from dataclasses import replace
    from ouroboros.workspace_admission import validate_workspace_root

    if authority.mode == "plan":
        if not default_shaped_directory_options(strategy, scope_paths):
            raise ValueError("Plan turns read the selected folder; choose directory geometry when starting an executor.")
        return authority, {}, {"plan_workspace_kind": "directory"}
    investigation = getattr(authority, "investigation", False)
    if not staged:
        try:
            validate_workspace_root(target_root, system_repo_dir=ctx.repo_dir, drive_root=custody.custody_root(ctx))
        except ValueError as exc:
            if not investigation:
                raise
            raise ValueError(f"{exc}; an evidence-only investigation has no private copy for this folder, "
                             "so lower this call with access='readonly' to read it in place") from exc
    if investigation:
        # An evidence-only investigation (#1609) never writes the folder itself: it
        # works in the engine's private copy of the whole folder unless it names a scope.
        if strategy == "direct":
            raise ValueError("an investigation without write authority works in a private copy; "
                             "omit directory_strategy or choose copy")
        strategy, scope_paths = "copy", scope_paths or ["."]
    options, reference = directory_execution(gateway, strategy, scope_paths)
    if staged:
        reference["source_root"] = str(staged)
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


def capture_directory_result(drive, entry, gateway, *, publish=True):
    """Capture observed terminal work independently of its pending money record.

    A finished run whose engine reports no file work product is captured ready, never retried
    on later sweeps: as ``no_changes`` after success, otherwise as an absent work product whose
    changes, if any, were not captured (Apply applies nothing; Reject releases the copy).
    One capture fetches each content identity (digest and size) once: every file side keeps
    its own row, and a later side with the same identity names the first verified file.
    Modes stay in the exact engine manifest; no retained file is chmodded.
    """
    if not entry.settled and entry.terminal_state not in custody.TERMINAL_STATES:
        raise ValueError("the directory run has not completed")
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
        # The engine's answer for a finished run is final, never a capture to retry every
        # sweep pass. Only a successful run without a file work product made no changes;
        # after any other end the absence proves nothing, and the note says so.
        state = str((detail.get("summary") or {}).get("state") or entry.terminal_state or "")
        succeeded = state in custody.SUCCEEDED_STATES
        note = ("The finished run produced no file work product: it made no changes to capture." if succeeded else
                f"The run ended {state or 'without success'} and published no file work product: "
                "changes, if any, were not captured."
                + (" In-place effects, if any, are already in the folder." if entry.resource_ref["strategy"] == "direct" else ""))
        manifest = {
            "schema_version": 1, "capture_kind": "engine_directory", "status": "ready", "no_changes": succeeded,
            "work_product": "absent", "workspace_root": "", "authority_target_root": entry.target_root,
            "engine_manifest": None, "engine_manifest_sha256": "", "complete": succeeded, "scope_paths": None,
            "strategy": entry.resource_ref["strategy"], "file_outputs": [], "apply_state": None, "note": note,
        }
        cap_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest_path, manifest, trailing_newline=True)
        if publish:
            custody.record_patch_captured(drive, entry, status="ready", sha256="", capture_dir=str(cap_dir),
                                         execution_root="", no_changes=succeeded)
        return manifest
    source = (product.get("files") or {}).get("manifest")
    digest = str((product.get("meta") or {}).get("manifest_sha256") or "").removeprefix("sha256:")
    if not source or len(digest) != 64:
        raise ValueError("the file work product has no verifiable manifest")
    raw_path = cap_dir / "engine-files-manifest.json"
    raw = _download(gateway, entry.run_id, source, raw_path, {"sha256": digest})
    original = json.loads(raw_path.read_text())
    source_root = entry.resource_ref.get("source_root") or entry.target_root  # a staged folderless input copy
    if (original.get("version") != 1 or not original.get("executionRoot")
            or Path(original.get("sourceRoot") or "").resolve() != Path(source_root).resolve()
            or not isinstance(original.get("entries"), list)):
        raise ValueError("engine directory manifest has an invalid source or execution binding")
    direct = entry.resource_ref["strategy"] == "direct"
    if original.get("isolation") != ("live" if direct else "envelope"):
        raise ValueError("engine directory strategy differs from the recorded request")
    if not direct and original.get("complete") is not True:
        raise ValueError("the selected execution copy was not completely captured")
    records, retained = [raw], {}  # retained: this capture's verified file per content identity
    for index, row in enumerate(original["entries"]):
        for side in ("before", "after"):
            state = row.get(side)
            if not isinstance(state, dict) or state.get("kind") != "file":
                continue
            if not state.get("artifactPath"):
                if direct and side == "before":
                    continue
                raise ValueError(f"file {side} has no complete reference: {row.get('path')}")
            identity = (state.get("sha256"), state.get("sizeBytes"))
            if identity not in retained:
                retained[identity] = _download(gateway, entry.run_id, state["artifactPath"],
                                               cap_dir / f"{side}-{index}", state)
            records.append({**retained[identity], "source_path": row["path"], "side": side})
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
    if publish:
        custody.record_patch_captured(drive, entry, status="ready", sha256=digest,
                                     capture_dir=str(cap_dir), execution_root=original["executionRoot"])
    return manifest


def directory_capture_block(drive, entry, manifest):
    from ouroboros.delegate_registration_policy import INVESTIGATION_NOTE, evidence_only

    cap_dir = custody.delegated_capture_dir(drive, entry.task_id, entry.run_id)
    return {"status": manifest["status"], "capture_kind": "engine_directory",
            "sha256": manifest.get("engine_manifest_sha256", ""),
            **({"no_changes": manifest["no_changes"]} if "no_changes" in manifest else {}),
            "execution_root": manifest["workspace_root"], "authority_target_root": entry.target_root,
            "manifest_artifact": str(cap_dir / "workspace_patch.json"),
            "manifest_read": {"root": "artifact_store", "path": f"delegated_runs/{cap_dir.name}/workspace_patch.json"},
            "file_outputs": manifest["file_outputs"], "patch_artifact": None,
            "complete": manifest["complete"], "scope_paths": manifest["scope_paths"],
            "strategy": manifest["strategy"],
            "note": INVESTIGATION_NOTE if evidence_only(entry) else manifest["note"]}


def integrate_directory_result(ctx, entry, decision, reason, gateway, *, acknowledge_ambiguous=False,
                               paths=None, orphan=False, admission_check=None, prepared=None):
    """Use engine CAS/disposition, retaining unknown effects under the same host intent."""
    drive = custody.custody_root(ctx)
    prepared = not entry.patch_captured if prepared is None else prepared
    try:
        manifest = capture_directory_result(drive, entry, gateway)
        original = None
        if decision == "apply" and manifest.get("engine_manifest") and manifest["strategy"] != "direct":
            cap_dir = custody.delegated_capture_dir(drive, entry.task_id, entry.run_id)
            original = json.loads((cap_dir / manifest["engine_manifest"]["name"]).read_text())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return (f"⚠️ INTEGRATE_MANIFEST_UNREADABLE: {type(exc).__name__}: {exc}. "
                "No request was submitted by this call; retained work and any earlier apply intent remain unresolved.")
    from ouroboros.delegate_shared import orphan_apply_target_ok
    active, target = Path(ctx.active_repo_dir()).resolve(), Path(entry.target_root).resolve()
    if decision == "apply" and not (orphan_apply_target_ok(target, active) if orphan else active == target):
        return "⚠️ INTEGRATE_DELEGATED_TARGET_MISMATCH: the run belongs to another folder."
    if paths is not None and (not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not path for path in paths)):
        return "⚠️ TOOL_ARG_ERROR (integrate_delegated_patch): paths must be a nonempty list of captured file paths."
    if admission_check and (refusal := admission_check()):
        return refusal
    from ouroboros.tools.subagent_integration_delegated import prepared_capture_inspection
    from ouroboros.delegate_collection import capture_projection

    copies = not manifest.get("no_changes") and manifest.get("work_product") != "absent" and manifest["strategy"] != "direct"
    if inspection := prepared_capture_inspection(entry.run_id, decision, copies,
                                                 lambda: capture_projection(drive, entry, reader_task_id=ctx.task_id),
                                                 prepared=prepared):
        return inspection
    receipt: dict = {}
    absent = manifest.get("work_product") == "absent" and not manifest.get("no_changes")
    if absent and decision == "apply":
        # An unsuccessful run's missing work product is not "no changes": nothing is applied
        # and the engine copy is kept; Reject is the disposition that releases it.
        return ("⚠️ INTEGRATE_DELEGATED_NOTHING_CAPTURED: " + manifest["note"]
                + " Nothing was applied and the engine copy is kept; reject the run to release it.")
    if manifest.get("no_changes") or absent:
        # Nothing to apply or undo; an envelope copy is still discarded so the engine releases it.
        if manifest["strategy"] != "direct":
            key = sha256(f"discard:{entry.run_id}:no_changes".encode()).hexdigest()
            receipt = gateway.decide_run(entry.run_id, {"action": "discard"}, idempotency_key=key) or {}
            if receipt.get("accepted") is not True or receipt.get("status") != "discarded":
                return "⚠️ INTEGRATE_DELEGATED_DISCARD_UNCONFIRMED: engine disposition was not confirmed; all results remain retained."
        disposition = "applied" if decision == "apply" else "rejected"
    elif decision == "reject":
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
        from ouroboros.workspace_copies import copy_apply_refusal, target_write_refusal

        touched = [row["path"] for row in original["entries"]
                   if paths is None or row["path"] in paths]
        if refusal := copy_apply_refusal(ctx, target, touched):
            return target_write_refusal(target, refusal)
        if entry.patch_apply_pending and not acknowledge_ambiguous:
            return "⚠️ INTEGRATE_DELEGATED_APPLY_AMBIGUOUS: an earlier apply awaits reconciliation; inspect it before retrying."
        request = {"target": {"kind": "original_project"}, "mode": "apply"}
        if paths is not None:
            request["paths"] = paths
        key = sha256(json.dumps([entry.run_id, manifest["engine_manifest_sha256"], request], sort_keys=True).encode()).hexdigest()
        if entry.patch_apply_pending and getattr(entry, "patch_apply_key", "") != key:
            return "⚠️ INTEGRATE_DELEGATED_APPLY_AMBIGUOUS: retry the original path selection; its apply request still has an unknown outcome."
        replaying_intent = entry.patch_apply_pending
        if not custody.record_patch_apply_started(drive, entry, target_root=entry.target_root,
                                                  apply_idempotency_key=key):
            return "⚠️ INTEGRATE_INTENT_UNWRITTEN: the apply intent could not be saved; nothing was submitted."
        refusal = admission_check() if admission_check else ""
        if not refusal and (write_refusal := copy_apply_refusal(ctx, target, touched)):
            refusal = target_write_refusal(target, write_refusal)
        if refusal:
            if not replaying_intent:
                custody.record_patch_apply_resolved(drive, entry, reason="authority_changed")
            return refusal
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
                                         reason=reason, engine_receipt=receipt,
                                         disposed_by_task_id=str(getattr(ctx, "task_id", "") or "")):
        return "⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: the engine completed the operation but its host receipt could not be saved."
    return json.dumps({"status": disposition, "run_id": entry.run_id, "target": entry.target_root,
                       "engine_receipt": receipt,
                       "note": manifest["note"] if manifest.get("no_changes") else
                       "Original effects are unchanged." if disposition == "rejected" else
                       "Results are already in the target folder; this disposition does not commit or publish."}, ensure_ascii=False)

"""The C1 integration seam of the delegated-run tools (authority → snapshot → capture).

Extracted from ``ouroboros/tools/delegate.py`` when that module crossed its size gate:
this is one coherent concern — how a MUTATING delegated run is bound to the tree it may
change. The host derives the unified authority record (B5), provisions the private
execution snapshot the run edits instead of the shared tree (C1), validates the
byte-identical retry binding, and captures the terminal diff for explicit integration.
``tools.delegate`` re-exports every name here (same objects), so every existing
reference — sibling code, the tests, the convergence census — still finds them there.
"""

from __future__ import annotations

import json
import logging
import pathlib
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple

from ouroboros import delegate_custody as custody
from ouroboros.delegate_custody import RunCustody as _RunCustody
# ONE refusal author for the whole delegate surface: the neutral leaf
# `delegate_shared` (phase B's facade split), never a local twin that could drift.
from ouroboros.delegate_registration_policy import (
    INVESTIGATION_AUTHORITY, INVESTIGATION_NOTE, NO_CHANGES_NOTE, evidence_only, no_write_authority,
    record_persistent as _record_persistent)
from ouroboros.delegate_shared import _fail
from ouroboros.configured_subagents import SESSION_ACCESS_PROFILES
from ouroboros.tools.tool_result import ToolResult
from ouroboros.tools.registry import ToolContext, active_repo_dir_for
from ouroboros.utils import resolve_path_allow_missing

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ouroboros.subagents import DelegatedRunShape

log = logging.getLogger(__name__)


def _resolved(path: Any) -> Optional[pathlib.Path]:
    """Resolve a path, or None when it cannot be resolved at all (null byte, symlink
    loop, unreadable parent). One predicate, so no call site re-enumerates the set."""
    try:
        return resolve_path_allow_missing(pathlib.Path(str(path)))
    except (OSError, ValueError, RuntimeError, TypeError):
        return None


# C1 capture mode: every mutating delegated run executes in a private snapshot and
# returns its diff for explicit integration. The value is recorded (not branched on)
# so the durable record names the regime a run was admitted under.
_CAPTURE_DELEGATED_SNAPSHOT = "delegated_snapshot"


def _mutation_authority(ctx: ToolContext, authority: "DelegatedRunShape") -> tuple[Dict[str, str], Optional[ToolResult]]:
    """The UNIFIED host-derived authority record for one delegated run (B5).

    ``{"target_root", "source", "capture_mode"}``. Two sources, one shape: ``acting_constraint`` (an
    acting child's ``task_constraint.write_root``, which must equal the genuinely ACTIVE workspace root
    — `active_repo_dir_for` falls back to the LIVE repo when `is_workspace_mode()` is false) and
    ``external_workspace_root`` (an ordinary root's selected external workspace or Project room, where it
    already holds write+shell authority). A run without write authority — read-only, or an evidence-only
    investigation (#1609) — gets ``no_write_authority``: the active root as provenance only.
    Disagreement anywhere is a typed refusal, never a best-effort guess.
    """
    root = str(active_repo_dir_for(ctx))
    if authority.access not in SESSION_ACCESS_PROFILES or authority.investigation:
        return no_write_authority(root, investigation=authority.investigation), None
    constraint = getattr(ctx, "task_constraint", None)
    mode = str(
        (constraint.get("mode") if isinstance(constraint, dict)
         else getattr(constraint, "mode", "")) or ""
    ).strip()
    granted = str(
        (constraint.get("write_root") if isinstance(constraint, dict)
         else getattr(constraint, "write_root", "")) or ""
    ).strip()
    # ONE predicate, the one the registry already owns. `workspace_mode_block_reason`
    # returns "" precisely WHEN `workspace_mode` is empty, so "no block reason" is
    # satisfied by the absence of a workspace — the condition was true in exactly the
    # case it was written to refuse. `is_workspace_mode()` is the question actually
    # being asked, and `active_repo_dir()` branches on that same call.
    workspace_active = bool(
        callable(getattr(ctx, "is_workspace_mode", None)) and ctx.is_workspace_mode())
    if mode == "acting_subagent" or granted:
        if not granted:
            return {}, _fail(
                "delegate_start", "write_root_missing",
                "This child is allowed to write, but its task constraint names no write_root, "
                "so there is no directory the host can honestly confine the run to.",
            )
        # AGREEMENT IS NOT ENOUGH: `active_repo_dir_for` falls back to `repo_dir` when
        # workspace mode is off, so a constraint whose write_root happens to name that
        # same directory made the comparison pass and handed a shell the live repository
        # — the very case this guard was written for. Require a genuinely ACTIVE
        # workspace.
        if not workspace_active:
            return {}, _fail(
                "delegate_start", "workspace_not_active",
                "A delegated run may only WRITE inside an ACTIVE workspace, and this task "
                "has none. Refusing rather than falling back to the repository root.",
            )
        # "Can this path be resolved at all" is ONE question, not an exception set to
        # re-enumerate: an embedded null raises ValueError and a symlink loop RuntimeError,
        # and either escaping here would abort delegate_start with a traceback instead of
        # the typed refusal this function exists to produce.
        resolved_root, resolved_grant = _resolved(root), _resolved(granted)
        if resolved_root is None or resolved_root != resolved_grant:
            return {}, _fail(
                "delegate_start", "write_root_mismatch",
                "The active root and the granted write_root disagree, so the run would write "
                "somewhere this task was never given. Refusing rather than guessing.",
                active_root=root, granted_write_root=granted,
            )
        return {"target_root": root, "source": "acting_constraint",
                "capture_mode": _CAPTURE_DELEGATED_SNAPSHOT}, None
    # An ordinary room uses the same selected-target authority without becoming
    # a pooled workspace task or requiring Git merely for native file operations.
    from ouroboros.tool_access import _TOP_LEVEL_PRINCIPAL_PROFILES, active_tool_profile, project_room_lens_dir

    room = (project_room_lens_dir(ctx)
            if active_tool_profile(ctx) in _TOP_LEVEL_PRINCIPAL_PROFILES else None)
    if not workspace_active and room is None:
        return {}, _fail(
            "delegate_start", "workspace_not_active",
            "A delegated run may only WRITE inside an ACTIVE workspace, and this task "
            "has none. Refusing rather than falling back to the repository root.",
        )
    ws_mode = "external" if room is not None else str(getattr(ctx, "workspace_mode", "") or "").strip().lower()
    if ws_mode not in {"external", "external_workspace"}:
        return {}, _fail(
            "delegate_start", "write_root_missing",
            "This task holds a mutating shape but neither an acting write_root nor an "
            "external workspace contract names the tree it may write. Refusing rather "
            "than guessing a target.",
        )
    selected_root = room if room is not None else getattr(ctx, "workspace_root", None)
    declared = _resolved(selected_root)
    resolved_root = _resolved(root)
    if resolved_root is None or declared is None or resolved_root != declared:
        return {}, _fail(
            "delegate_start", "write_root_mismatch",
            "The active root does not resolve to this task's declared external "
            "workspace, so the run would write somewhere this task was never given.",
            active_root=root, declared_workspace_root=str(selected_root or ""),
        )
    return {"target_root": root, "source": "external_workspace_root",
            "capture_mode": _CAPTURE_DELEGATED_SNAPSHOT}, None


def _retry_binding_refusal(record: Dict[str, Any], retry_token: str) -> Optional[ToolResult]:
    """Refuse a MUTATING retry whose stored row carries no C1 isolation binding.

    A PRE-C1 row's recorded body scopes the run at the LIVE target tree, and the
    body is replayed byte-identically — so the binding cannot be minted
    retroactively and replaying would write straight into the shared tree, in the
    in-place regime C1 retired. Returns ``None`` when the full binding is present.
    """
    reference = record.get("resource_ref") or {}
    request = record.get("request") or {}
    if reference.get("workspace_kind") == "directory" and (request.get("execution") or {}).get("workspaceKind") == "directory" and record.get("target_root"):
        return None
    from ouroboros.delegate_directory import is_in_place_run
    if is_in_place_run(record) and record.get("target_root") and str(record.get("execution_root") or "") == str(record["target_root"]):
        return None  # recorded in place by its starter's choice: the replay works there again
    snapshot_id = str(record.get("snapshot_id") or "")
    baseline_sha = str(record.get("baseline_sha") or "")
    target_root = str(record.get("target_root") or "")
    if snapshot_id and str(record.get("execution_root") or "") and baseline_sha and target_root:
        return None
    return _fail(
        "delegate_start", "retry_binding_absent",
        "This retry replays a MUTATING invocation recorded BEFORE private "
        "execution snapshots (no snapshot/baseline binding), so replaying it "
        "would write directly into the shared tree. Start a new run with a "
        "plain delegate_start(subagent_id=..., prompt=...) — it takes its own "
        "snapshot and returns its "
        "diff for explicit integration.",
        retry_of=retry_token, recorded_root=target_root,
        snapshot_id=snapshot_id, baseline_sha=baseline_sha)


def _validated_invocation(drive: Any, retry_token: str, task_id: str,
                          text: str) -> Tuple[Optional[Dict[str, Any]], Optional[ToolResult]]:
    """The stored invocation a retry may replay, or the typed refusal that stops it.

    Six ways a token is not replayable, each answered by name: no record, another
    task's, already bound, definitely refused (its id is retired — replaying wedges a
    permanent 409), no canonical body or prompt (a legacy_gap included), prompt
    disagrees. One author for all six. Empty text asserts nothing, so the recorded
    prompt is replayed (#1614); any other text, whitespace included, must be what the
    original call was given.
    """
    record = custody.invocation_record(drive, retry_token)
    if record is None:
        return None, _fail("delegate_start", "unknown_invocation",
                           "retry_of names an invocation with no durable record on this "
                           "drive. Start a new run with a plain "
                           "delegate_start(subagent_id=..., prompt=...).",
                           retry_of=retry_token)
    # A legacy_gap proves neither its owner nor its body: it stays unrecorded below, never replayed.
    if record["state"] != "legacy_gap" and record["task_id"] != task_id:
        return None, _fail("delegate_start", "invocation_not_owned",
                           "retry_of names another task's invocation. A delegated start "
                           "may only be retried by the task that requested it.",
                           retry_of=retry_token)
    if record["state"] == "started":
        return None, _fail("delegate_start", "invocation_already_started",
                           "That invocation already bound a run — do not re-post it. "
                           "Wait on the existing run instead.",
                           retry_of=retry_token, run_id=record["run_id"])
    if record["state"] == "failed_definite":
        return None, _fail("delegate_start", "invocation_definitely_refused",
                           "That invocation was definitively refused by the daemon; its "
                           "id is retired. Start a new run with a plain "
                           "delegate_start(subagent_id=..., prompt=...).",
                           retry_of=retry_token)
    body = record.get("request")
    prompt = body.get("prompt") if isinstance(body, dict) else None
    if not isinstance(prompt, str) or not prompt:
        return None, _fail("delegate_start", "invocation_request_unrecorded",
                           "That invocation's durable row carries no canonical request "
                           "body, so it cannot be replayed byte-identically. Its outcome "
                           "remains unknown; restore the recorded request or reconcile "
                           "the original invocation before retrying that invocation.",
                           retry_of=retry_token)
    from ouroboros.delegate_continuation import retry_matches_caller_text
    if text and prompt != text and not retry_matches_caller_text(record, text):
        return None, _fail("delegate_start", "retry_prompt_mismatch",
                           "retry_of replays the RECORDED invocation, but the prompt "
                           "you passed differs from the one it sent. Omit prompt (or pass "
                           "'') to replay it, or drop retry_of to start a new run.",
                           retry_of=retry_token)
    return record, None


class _RetryBinding(NamedTuple):
    """Every fact of a replayed invocation, read off its ONE durable record."""

    request_body: Dict[str, Any]
    route: Any          # DelegationRoute
    authority: Any      # DelegatedRunShape
    root: str
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
    processing: Dict[str, Any]
    execution_binding_fingerprint: str
    max_seconds_basis: str
    continuation: Any   # delegate_continuation.ContinuationStart: the recorded lineage
    prompt_cache_ttl: Dict[str, Any]


def _resolve_retry_invocation(ctx: ToolContext, drive: pathlib.Path, retry_token: str, text: str, *,
                              max_seconds: Any = None, geometry: Tuple[Any, Any] = (None, None),
                              ) -> Tuple[Optional[_RetryBinding], Optional[ToolResult]]:
    """Rebuild a retried start from its stored invocation, or refuse it typed.

    The stored invocation is the SINGLE SOURCE of EVERY retry fact (route, shape,
    root, project, key, cap and its basis, continuation lineage), validated BEFORE
    any daemon call so a refused token registers nothing. A MUTATING replay
    additionally re-proves its C1 binding: the row must carry the snapshot/baseline
    binding (pre-C1 rows refused), the task's PRESENT workspace must still resolve
    to the recorded authority target, and the recorded snapshot must still exist on
    disk. Last, the call's own request arguments may not contradict the replay: new
    geometry refuses, and an explicit ``max_seconds`` must equal the recorded cap.
    """
    from ouroboros.delegate_continuation import replayed_custody
    from ouroboros.subagents import DelegatedRunShape, DelegationRoute

    record, refusal = _validated_invocation(
        drive, retry_token, str(getattr(ctx, "task_id", "") or ""), text)
    if refusal:
        return None, refusal
    request_body = record["request"]
    execution = (request_body.get("execution")
                 if isinstance(request_body.get("execution"), dict) else {})
    scope = request_body.get("scope") if isinstance(request_body.get("scope"), dict) else {}
    route = DelegationRoute(route_id=str(request_body.get("primaryHarness") or ""),
                            model=str(request_body.get("model") or ""),
                            effort=str(request_body.get("effort") or ""),
                            # The STORED pin, so a retry's health check judges the
                            # account the replayed body actually names — never the
                            # setting as it reads today (D-U5).
                            profile_id=str(request_body.get("credentialProfileId") or ""))
    authority = DelegatedRunShape(access=str(request_body.get("access") or ""),
                                  mode=str(request_body.get("mode") or ""),
                                  isolation=str(execution.get("isolation") or ""),
                                  delegated=bool(execution.get("delegated")),
                                  investigation=record.get("authority_source") == INVESTIGATION_AUTHORITY)
    scope_root = str(scope.get("root") or "")
    # Claudexor 3.8.1+ separates stable project identity (scope.root) from
    # the one-shot execution snapshot.  Retry still replays the stored body
    # byte-for-byte, but host custody/capture must keep naming the snapshot.
    # On the strict 3.8.0 wire workspaceRoot is absent and scope.root remains
    # the execution root, so this is byte-compatible with the legacy shape.
    root = str((record.get("execution_root") if request_body.get("_thread_id") else "")
               or execution.get("workspaceRoot") or scope_root)
    project_id = str(record.get("project_id") or "")
    # The C1 isolation binding recorded at the original attempt. Pre-C1 rows
    # carry none; their scope.root IS the authority target (in-place regime).
    snapshot_id = str(record.get("snapshot_id") or "")
    target_root = str(record.get("target_root") or "") or scope_root
    if authority.access in SESSION_ACCESS_PROFILES:
        binding_refusal = _retry_binding_refusal(record, retry_token)
        if binding_refusal:
            return None, binding_refusal
        # The replay will WRITE for the recorded AUTHORITY TARGET, so authority
        # is re-asked against the task's PRESENT context — and the answer must
        # be the very target the invocation recorded. A target that moved
        # between the attempts makes the replay a write destined for a tree this
        # task no longer holds, which is a refusal, never a re-derivation. A
        # payload run re-resolves its SEMANTIC reference (bucket + skill name)
        # through a fresh exact binding rather than the workspace contract.
        if str(record.get("authority_source") or "") == "skill_payload":
            _target, _binding, rebind_refusal = _rebind_payload_reference(
                ctx, record.get("resource_ref") or {}, target_root,
                tool="delegate_start", context=f"retry_of={retry_token}")
            if rebind_refusal:
                return None, rebind_refusal
        else:
            record_auth, root_error = _mutation_authority(ctx, authority)
            if root_error:
                return None, root_error
            resolved_current = _resolved(record_auth.get("target_root"))
            resolved_recorded = _resolved(target_root)
            if resolved_current is None or resolved_current != resolved_recorded:
                return None, _fail(
                    "delegate_start", "retry_root_divergence",
                    "This retry replays a MUTATING invocation recorded against a root "
                    "this task no longer holds: the active write root has moved since "
                    "the original attempt. Start a new run for the current root.",
                    retry_of=retry_token, recorded_root=target_root,
                    active_root=record_auth.get("target_root"))
        if snapshot_id:
            # The retry reproduces the EXACT binding — same execution root, same
            # baseline. A snapshot the GC already collected cannot be re-minted
            # (the baseline commit is gone with its ref): typed refusal.
            from ouroboros.subagent_worktrees import find_execution_snapshot
            snap_entry = find_execution_snapshot(snapshot_id)
            if snap_entry is None or not pathlib.Path(str(snap_entry.get("path") or "")).exists():
                return None, _fail(
                    "delegate_start", "execution_snapshot_missing",
                    "This retry replays a MUTATING invocation whose private "
                    "execution snapshot no longer exists on disk, so the recorded "
                    "binding cannot be reproduced. Start a new run with a plain "
                    "delegate_start(subagent_id=..., prompt=...) (it will take a "
                    "fresh snapshot).",
                    retry_of=retry_token, snapshot_id=snapshot_id)
    if geometry != (None, None):
        return None, _fail("delegate_start", "retry_selector_conflict",
                           "A retry replays its recorded directory strategy and scope; omit new geometry arguments.")
    seconds = int(request_body.get("maxSeconds") or 0)
    try:  # read as bounded_max_seconds reads it: zero, negative or malformed means omitted
        asked = max(0, int(max_seconds)) if max_seconds is not None else 0
    except (TypeError, ValueError):
        asked = 0
    if asked and asked != seconds:
        return None, _fail("delegate_start", "retry_selector_conflict",
                           f"A retry replays its recorded max_seconds={seconds}; omit max_seconds to retry.",
                           retry_of=retry_token, recorded_max_seconds=seconds)
    return _RetryBinding(
        request_body=request_body,
        route=route,
        authority=authority,
        root=root,
        key=str(record.get("idempotency_key") or ""),
        project_id=project_id,
        owned_project_id=(project_id if record.get("project_owned") else ""),
        project_persistent=_record_persistent(record),
        seconds=seconds,
        snapshot_id=snapshot_id,
        target_root=target_root,
        baseline_sha=str(record.get("baseline_sha") or ""),
        authority_source=str(record.get("authority_source") or ""),
        resource_ref=(record.get("resource_ref")
                      if isinstance(record.get("resource_ref"), dict) else {}),
        processing=deepcopy(record.get("processing") if isinstance(record.get("processing"), dict) else {}),
        execution_binding_fingerprint=str(record.get("execution_binding_fingerprint") or ""),
        # The replayed body is byte-identical, so its cap basis is the recorded one
        # too, never re-derived from today's clocks.
        max_seconds_basis=str(record.get("max_seconds_basis") or ""),
        continuation=replayed_custody(record),
        prompt_cache_ttl=deepcopy(record.get("prompt_cache_ttl") if isinstance(record.get("prompt_cache_ttl"), dict) else {}),
    ), None


def _provision_snapshot(ctx: ToolContext, drive: pathlib.Path, target_root: str, invocation_id: str, *,
                        start: str = "") -> Tuple[Optional[Any], Optional[ToolResult]]:
    """Provision the C1 private execution snapshot for one mutating run.

    Registered durably (worktree registry) and described durably (baseline manifest
    artifact) BEFORE the caller records any start intent, so a worker death at any
    later point leaves a nameable, GC-reconcilable root — never an orphan directory.
    Returns ``(handle, None)`` or ``(None, typed_refusal)``.

    ``start`` is an investigation's starting folder inside the ``target_root`` worktree
    (``delegate_directory.snapshot_checkout``): the whole worktree is copied, and the
    returned handle's ``path`` is the matching folder, which the run, its custody,
    named evidence and a continuation use; the copy's record keeps the copy root that
    capture and release address.
    """
    from dataclasses import replace

    from ouroboros.subagent_worktrees import provision_execution_snapshot, remove_execution_snapshot

    task_id = str(getattr(ctx, "task_id", "") or "")
    try:
        folder = pathlib.Path(start or target_root).resolve().relative_to(pathlib.Path(target_root).resolve())
        handle = provision_execution_snapshot(target_root=target_root, task_id=task_id, snapshot_id=invocation_id)
    except Exception as exc:
        return None, _fail(
            "delegate_start", "execution_snapshot_failed",
            "A private execution snapshot of the write root could not be provisioned "
            f"({type(exc).__name__}: {exc}). The run was NOT started: a mutating "
            "delegated run executes only in its own snapshot, never in the shared tree.",
            target_root=target_root, definitely_unrun=True)
    if folder.parts and not (pathlib.Path(handle.path) / folder).is_dir():
        # Unclaimed; custody GC retries a failed release once the owning task is proven over.
        try:
            remove_execution_snapshot(handle.snapshot_id)
        except Exception:
            log.warning("Failed to release snapshot %s without its starting folder", handle.snapshot_id, exc_info=True)
        return None, _fail(
            "delegate_start", "execution_snapshot_failed",
            f"The starting folder {folder.as_posix()} is not part of the worktree snapshot (an ignored or "
            "empty folder), so the investigation has no copy to start in. The run was NOT started: start "
            "from the worktree root, or lower this call with access='readonly' to read the folder in place.",
            target_root=target_root, definitely_unrun=True)
    _record_baseline_manifest(drive, task_id, invocation_id, handle,
                              **({"start_folder": folder.as_posix()} if folder.parts else {}))
    return (replace(handle, path=str(pathlib.Path(handle.path) / folder)) if folder.parts else handle), None


_NESTED_REPOSITORIES_SHOWN = 20


def _snapshot_facts(handle: Any) -> Dict[str, Any]:
    """Disclosure on the start receipt: how big the private snapshot is, how long it
    took to provision (#1241) and which nested Git repositories it left out, so an
    empty snapshot says why. Facts only — nothing refuses or truncates on them."""
    if handle is None:
        return {}
    if getattr(handle, "adopted_from", ""):  # a continuation's predecessor snapshot
        facts: Dict[str, Any] = {"reused_from_run": handle.adopted_from}
    else:
        facts = {"entries": int(getattr(handle, "entry_count", 0) or 0),
                 "provisioning_sec": float(getattr(handle, "provisioning_sec", 0.0) or 0.0)}
    rows = [row for row in getattr(handle, "excluded_untracked", ()) or () if isinstance(row, dict)]
    nested = sorted(str(row.get("path") or "") for row in rows if row.get("reason") == "nested_repository")
    unlistable = sorted(str(row.get("path") or "") for row in rows if row.get("reason") == "unlistable_directory")
    if unlistable:
        facts["unlistable_directories_excluded"] = unlistable[:_NESTED_REPOSITORIES_SHOWN]
        facts["unlistable_directories_note"] = ("Folders Git could not open are not in this snapshot, so the run "
                                                "does not see their files; the baseline manifest lists them all.")
    if nested:
        facts["nested_repositories_excluded"] = nested[:_NESTED_REPOSITORIES_SHOWN]
        if len(nested) > _NESTED_REPOSITORIES_SHOWN:
            facts["nested_repositories_excluded_count"] = len(nested)
        facts["nested_repositories_note"] = (
            "Untracked Git repositories (paths relative to the snapshot's repository root) are not in this snapshot: "
            "the run neither works in them nor captures changes there. A child whose workspace is one of them can."
            + (" The full list is in this snapshot's baseline manifest." if len(nested) > _NESTED_REPOSITORIES_SHOWN else ""))
    return facts


def _record_baseline_manifest(drive: pathlib.Path, task_id: str, invocation_id: str,
                              handle: Any, **extra: Any) -> None:
    """The forensic baseline record beside the capture-to-be. Fail-soft: the BINDING
    itself is durable on the custody request row, and the baseline tree stays
    reconstructable for as long as the snapshot lives. One author for the Git and
    the standalone-payload snapshot paths."""
    try:
        from ouroboros.utils import atomic_write_json, utc_now_iso

        cap_dir = custody.delegated_capture_dir(drive, task_id, invocation_id)
        cap_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cap_dir / "baseline_manifest.json", {
            "schema_version": 1,
            "created_at": utc_now_iso(),
            "snapshot_id": handle.snapshot_id,
            "baseline_id": handle.baseline_sha,
            "baseline_tree": handle.baseline_tree,
            "manifest_digest": handle.manifest_digest,
            "entry_count": handle.entry_count,
            "provisioning_sec": float(getattr(handle, "provisioning_sec", 0.0) or 0.0),
            "target_root": handle.target_root,
            "target_head": handle.target_head,
            "execution_root": handle.path,
            "excluded_untracked": list(handle.excluded_untracked),
            **extra,
        }, trailing_newline=True)
    except Exception:
        log.warning("Failed to write delegated baseline manifest for %s", invocation_id,
                    exc_info=True)


def _capture_block(entry: _RunCustody, cap_dir: pathlib.Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The terminal-payload projection of one captured run patch (C1)."""
    from ouroboros.headless import (
        ARTIFACT_STATUS_READY_NO_CHANGES,
        ARTIFACT_STATUS_READY_WITH_CHANGES,
    )
    from ouroboros.subagent_worktrees import UNSAVED_REASONS

    status = str(manifest.get("status") or "")
    patch_path = cap_dir / "workspace.patch"
    has_patch = status == ARTIFACT_STATUS_READY_WITH_CHANGES and patch_path.exists()
    # The tool-surface read handle (CR1-2): the capture lives on the CANONICAL
    # drive, which a split-drive child cannot reach through a raw absolute path —
    # read_file(root='artifact_store', path=...) resolves this prefix against the
    # canonical root for the owning task, so THIS pair is what the agent reads.
    read_prefix = f"delegated_runs/{cap_dir.name}"
    unsaved = [str(row.get("path") or "") for row in manifest.get("untracked_excluded") or []
               if isinstance(row, dict) and row.get("reason") in UNSAVED_REASONS] if not evidence_only(entry) else []
    checkout = ("the copy's checkout itself is released once the capture is saved" if not unsaved else
                f"the copy's checkout is KEPT until then: {len(unsaved)} path(s) in it could not be saved in the "
                f"result (Git repositories or checked-out submodules inside it, or folders Git could not open): "
                f"{', '.join(unsaved[:8])}. A submodule's own commits live in its separate Git directory "
                "(`git -C <submodule> rev-parse --absolute-git-dir`), which goes with the copy, and apply never moves "
                "a submodule pointer: save what you need from execution_root and those directories before you decide")
    block: Dict[str, Any] = {
        "status": status or "missing",
        "baseline_id": entry.baseline_sha,
        "execution_root": entry.execution_root,
        "authority_target_root": entry.target_root,
        "patch_artifact": str(patch_path) if has_patch else None,
        "manifest_artifact": str(cap_dir / "workspace_patch.json"),
        "patch_read": ({"root": "artifact_store", "path": f"{read_prefix}/workspace.patch"}
                       if has_patch else None),
        "manifest_read": {"root": "artifact_store", "path": f"{read_prefix}/workspace_patch.json"},
        "sha256": str(manifest.get("sha256") or ""),
        "patch_size": manifest.get("patch_size"),
        "diffstat": str(manifest.get("diffstat") or ""),
        "file_outputs": list(manifest.get("file_outputs") or []),
        "file_output_count": len(manifest.get("file_output_changes") or []),
        "note": (
            "NOT APPLIED: the run edited its private copy only. Nothing reaches the shared "
            "tree until you explicitly call integrate_delegated_patch(run_id=...) with "
            "decision='apply' or 'reject'. The result is kept until then as the patch and "
            f"its pinned commits (vcs_read diff base=baseline_id head=result_commit); {checkout}. "
            "Read manifest_read, its file artifacts, and patch_read when present."
        ),
        **({"unsaved_paths": unsaved} if unsaved else {}),
    }
    if manifest.get("result_commit"):
        block["result_commit"] = str(manifest["result_commit"])
    if status not in {ARTIFACT_STATUS_READY_WITH_CHANGES, ARTIFACT_STATUS_READY_NO_CHANGES}:
        # A failed manifest's own typed note (unreviewable_metadata_change,
        # non-UTF-8, …) or git's own words are the actionable fact — never boilerplate.
        first = next((row for row in manifest.get("errors") or [] if isinstance(row, dict)), {})
        detail = str(first.get("stderr") or first.get("message") or "").strip()
        block["note"] = str(manifest.get("note") or "") or (f"capture failed: {detail[:600]}" if detail else block["note"])
    elif evidence_only(entry):  # evidence owes no apply/reject, on every read (#1969)
        block["note"] = INVESTIGATION_NOTE
        if getattr(entry, "patch_disposed", "") == custody.EVIDENCE_RETAINED:
            block["disposition"] = custody.EVIDENCE_RETAINED
    elif getattr(entry, "patch_disposed", "") == custody.NO_CHANGES:
        block["disposition"] = custody.NO_CHANGES
        block["note"] = NO_CHANGES_NOTE
    current_head = str(manifest.get("current_head") or "")
    if current_head and entry.baseline_sha and current_head != entry.baseline_sha:
        # Single-writer snapshot: a moved HEAD can only mean the run itself committed.
        # The diff is still measured from the baseline (committed work is captured),
        # but the violation of the no-commit instruction is disclosed, not hidden.
        block["head_moved"] = {
            "baseline": entry.baseline_sha, "current": current_head,
            "note": "the run COMMITTED inside its snapshot despite instructions; its "
                    "committed work is still in the captured diff",
        }
    return block


def _write_result_commit_artifacts(entry: _RunCustody, clone: pathlib.Path, cap_dir: pathlib.Path,
                                   evidence: str) -> Dict[str, Any]:
    """Capture a Git snapshot as its result commit C (parent B) with the ordinary manifest.

    The patch is B..C from git's plumbing; large and binary files ride in it as
    binary diffs, so there are no separate file outputs. New named evidence stays
    out of C (it is retained beside the capture). A git failure is a failed
    manifest that carries git's own words."""
    import os

    from ouroboros.artifact_storage import compress_in_place
    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.headless import ARTIFACT_STATUS_READY_NO_CHANGES, ARTIFACT_STATUS_READY_WITH_CHANGES
    from ouroboros.subagent_worktrees import GitFailed, _git_env, _host_env, _nul_paths, capture_result_commit
    from ouroboros.utils import atomic_write_json, utc_now_iso
    from ouroboros.workspace_patch_rules import NAMED_EVIDENCE_EXCLUSION

    patch_path = cap_dir / "workspace.patch"
    base = entry.baseline_sha
    manifest: Dict[str, Any] = {
        "schema_version": 1, "created_at": utc_now_iso(), "workspace_root": str(clone),
        "patch_name": "workspace.patch", "manifest_name": "workspace_patch.json",
        "base_provenance": "admission_head", "base_ref": base, "base_head": base, "base_is_empty_tree": False,
        "base_explanation": "Application patch from the run's snapshot commit B to its result commit C. "
                            "It does not attribute authorship.",
        "file_outputs": [], "file_output_changes": [], "tracked_excluded": [], "sensitive_blocked": [],
        "diagnostics": [], "errors": []}
    try:
        result = capture_result_commit(entry.snapshot_id, patch_path, held_out=evidence)
        env = _host_env()
        # Raw rows (":mode mode blob blob status", path) also say what B held at each changed path:
        # a later apply proves drift from them once the owner released the pins and Git collected B.
        raw = _git_env(clone, "diff-tree", "-r", "-z", "--no-renames", base, result["result_sha"],
                       env=env).stdout.decode("utf-8", "surrogateescape").split("\0")
        rows = [(meta.split(), path) for meta, path in zip(raw[0::2], raw[1::2]) if meta]
        added = [path for meta, path in rows if meta[4] == "A"]
        changed = [path for meta, path in rows if meta[4] != "A"]
        baseline_entries = {path: f"{meta[0][1:]} {meta[2]}" for meta, path in rows if meta[4] != "A"}
        diffstat = _git_env(clone, "diff-tree", "-r", "--stat", "--no-renames", base, result["result_sha"],
                            env=env).stdout.decode("utf-8", "replace").strip()
        # Each new file held out of C under the evidence folder, in the reason every reader of an
        # exclusion knows (format_patch_exclusions counts it as evidence, never a lost deliverable).
        held = sorted(set(_nul_paths(_git_env(clone, "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                                              "--", evidence, env=env).stdout))
                      - set(_nul_paths(_git_env(clone, "ls-tree", "-r", "-z", "--name-only", base, "--", evidence,
                                                env=env).stdout))) if evidence else []
    except (GitFailed, OSError) as exc:
        manifest.update(status="failed", patch_size=0, sha256="", diffstat="", current_head="",
                        errors=[{"type": "result_commit_failed", "message": str(exc),
                                 "stderr": os.fsdecode(getattr(exc, "stderr", None) or b"")}])
        atomic_write_json(cap_dir / "workspace_patch.json", manifest, trailing_newline=True)
        return manifest
    size, digest = 0, ""
    if result["patch_size"]:
        compress_in_place(patch_path)
        measured = stream_artifact_file(patch_path)
        size, digest = measured["size"], measured["sha256"]
    else:
        patch_path.unlink(missing_ok=True)
    excluded = [row for row in result["excluded"] if row.get("reason") != "named_evidence"] + [
        {"path": path, "reason": NAMED_EVIDENCE_EXCLUSION} for path in held]
    manifest.update(
        status=ARTIFACT_STATUS_READY_WITH_CHANGES if size else ARTIFACT_STATUS_READY_NO_CHANGES,
        current_head=result["head"], patch_size=size, sha256=digest, diffstat=diffstat,
        tracked_changed=changed, untracked_included=added, untracked_excluded=excluded,
        counts={"tracked_changed": len(changed), "tracked_excluded": 0, "untracked_included": len(added),
                "untracked_excluded": len(excluded), "sensitive_blocked": 0},
        result_commit=result["result_sha"], result_tree=result["result_tree"], baseline_entries=baseline_entries,
        result_ref=result["result_ref"], baseline_ref=result["baseline_ref"])
    atomic_write_json(cap_dir / "workspace_patch.json", manifest, trailing_newline=True)
    return manifest


def _capture_terminal_patch(ctx: ToolContext, entry: Optional[_RunCustody], *, gateway=None) -> Optional[Dict[str, Any]]:
    """Capture a terminal mutating run's diff from its execution snapshot, durably.

    Idempotent: the durable ``patch_captured`` flag (replayed) plus the manifest on
    disk mean a re-wait re-reads the existing capture instead of re-diffing. Runs
    only for C1-isolated runs (``execution_root`` recorded); read-only and legacy
    in-place runs return None. Capture failure is disclosed in the block, never
    raised — the settlement and delivery around it must not die on a diff error.
    """
    if entry is None:
        return None
    block = capture_terminal_patch_for_drive(custody.custody_root(ctx), entry, gateway=gateway)
    if block and entry.task_id != str(getattr(ctx, "task_id", "") or ""):
        # Relative handles name the starter's prefix, not the successor's.
        # The capture reader validates this exact canonical product, also after
        # disposition and from a split execution drive.
        for kind in ("patch", "manifest"):
            if block.get(f"{kind}_read") and block.get(f"{kind}_artifact"):
                block[f"{kind}_read"] = {"root": "artifact_store", "path": block[f"{kind}_artifact"]}
    return block


def _restored_from_pin(entry: _RunCustody, record: Dict[str, Any]) -> bool:
    """A gone checkout whose record still pins result commit C is checked out again from it, as a
    continuation's hand-over restores one, so a capture pinned before its manifest landed (or whose
    saved files went) is captured again instead of reported missing (#1651). Only when the pin is
    this run's own: a run owning its copy outright is the only one pinning it until a successor
    supersedes it; a same-tree continuation shares its predecessor's record, so its pin is its own
    only once its PATCH_CAPTURED landed (its capture re-pins before that row) and no successor took over."""
    if (not record.get("result_sha") or record.get("standalone") or entry.superseded_by
            or entry.capture_id and not entry.patch_captured):
        return False
    from ouroboros.subagent_worktrees import recreate_snapshot_checkout

    try:
        recreate_snapshot_checkout(entry.snapshot_id)
    except Exception:
        log.warning("Restoring run %s's checkout from its pinned result failed", entry.run_id, exc_info=True)
        return False
    return pathlib.Path(entry.execution_root).exists()


def capture_terminal_patch_for_drive(drive: Any, entry: _RunCustody, *, gateway=None,
                                     retain_investigation: bool = True, publish: bool = True) -> Optional[Dict[str, Any]]:
    """The drive-rooted core of the terminal capture — same contract, no ToolContext.

    One ordinary capture author for foreground collection, the body and disposition;
    startup/control reconciliation leaves its work pending. Reviewers use their native owner. The
    apply/reject DECISION stays with a live owner, never taken here — an evidence-only investigation has
    none to take, so its ready capture is retained as evidence at once (#1609). ``patch_captured`` MEANS
    "a usable patch artifact exists" (C1-R3): the custody row is minted only over a ready-status
    manifest; a failure manifest is returned as the failed block but leaves the row uncaptured (every
    retry point stays open), and a pre-R3 row over a failed manifest falls through to a fresh capture.
    """
    from ouroboros.delegate_directory import is_directory_run, capture_directory_result, directory_capture_block
    from ouroboros.delegate_capture_attempt import failure_retry
    from ouroboros.tools.subagent_integration_delegated import retain_run_evidence

    if entry is None or entry.review_owned:
        return None  # the review owner compares against its frozen subject, never a product baseline
    from ouroboros.delegate_directory import in_place_block, is_in_place_run
    if is_in_place_run(entry):
        return in_place_block(entry)  # its edits are already in its starter's folder: nothing to capture

    def finish(block, transport=None):
        return retain_run_evidence(drive, entry, block, transport) if retain_investigation and publish else block

    if entry is not None and entry.mode == "plan":
        return None  # A read-only plan retains its workspace for a later implement turn.
    if is_directory_run(entry):
        owned_gateway = None
        try:
            if gateway is None:
                from ouroboros.claudexor_daemon import read_owned_gateway
                owned_gateway = gateway = read_owned_gateway()
            block = directory_capture_block(drive, entry, capture_directory_result(drive, entry, gateway, publish=publish))
            return finish(block, gateway)
        except Exception as exc:
            return {"status": "failed", "capture_kind": "engine_directory", "authority_target_root": entry.target_root,
                    "retry": failure_retry(exc),
                    "note": f"Directory capture unavailable: {type(exc).__name__}: {exc}. The engine result is retained; no new run was started."}
        finally:
            if owned_gateway is not None:
                owned_gateway.close()
    if entry is None or not entry.execution_root or (entry.superseded_by and not entry.patch_captured):
        return None  # a superseded run's snapshot now holds its successor's work
    if entry.patch_result_lost:
        # A later collection reads what survived the decision; it must not turn an unreadable
        # saved manifest into absence or overwrite it while trying to recapture a decided loss.
        from ouroboros.delegate_collection import capture_projection

        return capture_projection(drive, entry)
    from ouroboros.headless import (
        ARTIFACT_STATUS_READY_NO_CHANGES,
        ARTIFACT_STATUS_READY_WITH_CHANGES,
    )

    ready = {ARTIFACT_STATUS_READY_WITH_CHANGES, ARTIFACT_STATUS_READY_NO_CHANGES}
    cap_dir = custody.delegated_capture_dir(drive, entry.task_id, custody.capture_key(entry))
    manifest_path = cap_dir / "workspace_patch.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            manifest = {}
        manifest = manifest if isinstance(manifest, dict) else {}
        if isinstance(manifest.get("authority_drift"), dict) and str(manifest.get("status") or "") == "failed":
            # A legacy failed no-change capture (drift evidence of releases 7.4.1 to
            # this one) is durable forensic custody that never minted PATCH_CAPTURED;
            # it stays rejectable and is never recaptured.
            return _capture_block(entry, cap_dir, manifest)
        if entry.patch_captured and str(manifest.get("status") or "") in ready:
            return finish(_capture_block(entry, cap_dir, manifest))
    from ouroboros.subagent_worktrees import find_execution_snapshot
    snapshot = find_execution_snapshot(entry.snapshot_id) or {}
    exec_root, saved = pathlib.Path(entry.execution_root), {}
    if not exec_root.exists():
        saved = manifest if manifest_path.exists() else {}
        if (str(saved.get("status") or "") in ready and saved.get("result_commit")
                and saved.get("result_commit") == snapshot.get("result_sha")):
            # The copy was released after this capture was saved but its row did not
            # land: the saved result itself is the capture.
            if publish:
                custody.record_patch_captured(drive, entry, status=str(saved.get("status") or ""),
                                              sha256=str(saved.get("sha256") or ""),
                                              patch_size=saved.get("patch_size"), capture_dir=str(cap_dir))
            return finish(_capture_block(entry, cap_dir, saved))
    if not exec_root.exists() and not _restored_from_pin(entry, snapshot):
        # A legacy capture saves file outputs before its diff: a later failure keeps them (#1651).
        kept = len(saved.get("file_output_changes") or []) if saved.get("status") == "failed" else 0
        return {
            **(_capture_block(entry, cap_dir, saved) if kept else {}),
            "status": "missing",
            "baseline_id": entry.baseline_sha,
            "execution_root": entry.execution_root,
            "authority_target_root": entry.target_root,
            "patch_artifact": None,
            "manifest_artifact": str(manifest_path) if kept else None,
            "note": (f"the private copy no longer exists on disk and was not restored from the pinned result "
                     f"{str(snapshot['result_sha'])[:12]}; the pin is kept. Nothing was applied anywhere."
                     if snapshot.get("result_sha") else
                     f"the private copy no longer exists on disk and its earlier capture failed after saving {kept} file "
                     f"change{'s' * (kept != 1)}, which manifest_read names; the run's other changes cannot be captured. "
                     "Nothing was applied anywhere." if kept else
                     "the private copy no longer exists on disk and no saved result names it, so the "
                     "run's changes cannot be captured. Nothing was applied anywhere."),
        }
    try:
        cap_dir.mkdir(parents=True, exist_ok=True)
        if entry.authority_source == "skill_payload":
            # The payload-specific adapter over the skill-loader inventory (R1
            # items 5/6): the generic workspace capture's junk/lockfile filters
            # would silently omit legitimate loader-visible files.
            manifest = _write_payload_patch_artifacts(exec_root, cap_dir, entry)
        else:
            from ouroboros.delegate_start_instructions import EVIDENCE_DIRNAME
            # A Git-subdirectory investigation runs in a folder of its copy; the
            # capture always diffs the whole recorded copy. The run's untracked
            # named evidence is retained beside the capture, never carried in it (#2070).
            clone = pathlib.Path(str(snapshot.get("path") or exec_root))
            try:
                evidence = (exec_root.resolve().relative_to(clone.resolve()) / EVIDENCE_DIRNAME).as_posix()
            except ValueError:
                evidence = ""
            if "file_baseline" in snapshot or "untracked_baseline" in snapshot:
                # A copy provisioned before the one-primitive update keeps its own capture:
                # its file inputs live outside Git and would read as new files in C.
                from ouroboros.headless import write_workspace_patch_artifacts
                _, manifest = write_workspace_patch_artifacts(
                    clone, cap_dir,
                    task={"metadata": {"workspace_preflight": {"git": {"head": entry.baseline_sha}},
                                       "file_baseline": snapshot.get("file_baseline") or {},
                                       "host_evidence": evidence}},
                )
            else:
                manifest = _write_result_commit_artifacts(entry, clone, cap_dir, evidence)
    except Exception as exc:
        log.warning("Delegated run patch capture failed for %s", entry.run_id, exc_info=True)
        return {
            "status": "failed",
            "retry": failure_retry(exc),
            "baseline_id": entry.baseline_sha,
            "execution_root": entry.execution_root,
            "authority_target_root": entry.target_root,
            "patch_artifact": None,
            "manifest_artifact": None,
            "note": f"patch capture failed ({type(exc).__name__}: {exc}); the private copy "
                    "is kept — inspect it directly.",
        }
    if publish and str(manifest.get("status") or "") in ready:
        custody.record_patch_captured(
            drive, entry,
            status=str(manifest.get("status") or ""),
            sha256=str(manifest.get("sha256") or ""),
            patch_size=manifest.get("patch_size"),
            capture_dir=str(cap_dir),
        )
    return finish(_capture_block(entry, cap_dir, manifest))


def capture_stranded_patch(drive_root: Any, run: _RunCustody) -> Dict[str, Any]:
    """Capture a reconciled dead-owner run without deciding its disposition."""

    from ouroboros.delegate_directory import is_directory_run
    if not ((run.execution_root or is_directory_run(run)) and run.settled and not run.patch_disposed):
        return {}
    try:
        block = capture_terminal_patch_for_drive(drive_root, run) or {}
    except Exception:
        log.warning("Reconcile patch capture failed for %s", run.run_id, exc_info=True)
        return {"patch_capture": "failed", "patch_disposition": "pending"}
    return captured_patch_result(block)


def captured_patch_result(block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The existing reconciliation fields, projected without another capture."""
    if not block:
        return {}
    return {
        "patch_capture": str(block.get("status") or ""),
        "patch_artifact": block.get("patch_artifact"),
        "patch_disposition": str(block.get("disposition") or "pending"),  # an investigation's is evidence_retained
    }


def _gone_checkout_holds(repo: pathlib.Path, copy: pathlib.Path, baseline: str) -> str:
    """What Git's administrative entry for a hand-deleted checkout still holds, read-only, and how to
    keep or give it up: a HEAD the child moved off its baseline, changes it staged (its index) or a
    commit its HEAD history still reaches (ORIG_HEAD, the HEAD reflog), any of which goes with the
    entry's prune; "" when none or no entry. The entry names its checkout absolutely or, under
    ``worktree.useRelativePaths``, relative to itself. Git reads an unreadable ORIG_HEAD or reflog
    file as empty, so each is read first. A present one is never absence: ORIG_HEAD must name a
    commit, and as Git skips (exit 0) each reflog line it cannot parse, a final one without its
    newline included, each non-blank line of the reflog must be an entry Git returns. Each way on is a
    run_command argv, not a line for a terminal, so no shell's quoting can split its paths. Raises
    OSError/ValueError."""
    import contextlib

    from ouroboros.subagent_worktrees import _git

    def way(*argv: Any) -> str:
        return f"run_command(cmd={json.dumps([str(arg) for arg in argv], ensure_ascii=False)})"

    common = _git(repo, "rev-parse", "--git-common-dir", check=False)
    if common.returncode:
        return (f"its repository {repo} could not name its Git directory ({common.stderr.strip()[:200]}); "
                "call again once it can")
    root = pathlib.Path(common.stdout.strip())
    try:
        entries = sorted((root if root.is_absolute() else repo / root).joinpath("worktrees").iterdir())
    except FileNotFoundError:
        return ""
    for admin in entries:
        if (admin / (admin / "gitdir").read_text(encoding="utf-8").strip()).parent.resolve(strict=False) != copy:
            continue
        entry, again = f"Git's entry for the gone checkout ({admin})", "; call again once it can be read"
        remove = way("git", "-C", repo, "worktree", "remove", "--force", "--force", copy)
        give_up = f"then give that entry up ({remove}) and call again: reject then records the loss"
        head = _git(repo, f"--git-dir={admin}", "rev-parse", "--verify", "--quiet", "HEAD", check=False)
        if head.returncode:
            return f"{entry} has an unreadable HEAD{again}"
        if (moved := head.stdout.strip()) != baseline:
            return (f"{entry} still names commit {moved[:12]}, not its baseline: keep it if needed "
                    f"({way('git', '-C', repo, 'branch', '<name>', moved)}), {give_up}")
        if (admin / "index").exists():
            staged = _git(repo, f"--git-dir={admin}", "diff-index", "--cached", "--quiet", baseline, check=False)
            if staged.returncode == 1:  # --binary --full-index keeps a staged binary file's bytes, not "differ"
                keep = way("git", f"--git-dir={admin}", "diff", "--cached", "--binary", "--full-index", baseline,
                           "--output=<file>")
                return (f"{entry} still stages changes against its baseline: keep them if needed as a patch git apply "
                        f"restores ({keep}), {give_up}")
            if staged.returncode:
                return f"{entry} could not be compared ({staged.stderr.strip()[:200]}){again}"
        present = {}
        for name in ("ORIG_HEAD", "logs/HEAD"):
            with contextlib.suppress(FileNotFoundError):
                present[name] = (admin / name).read_bytes()
        orig = _git(repo, f"--git-dir={admin}", "rev-parse", "--verify", "--quiet", "ORIG_HEAD", check=False)
        if orig.returncode and "ORIG_HEAD" in present:
            return f"{entry}'s HEAD history could not be read (its ORIG_HEAD names no commit){again}"
        log = _git(repo, f"--git-dir={admin}", "log", "-g", "--no-color", "--no-show-signature", "--format=%H",
                   "HEAD", "--", check=False)  # the common Git directory (``repo``) holds a file named HEAD
        seen = log.stdout.split()
        records = [line for line in present.get("logs/HEAD", b"").split(b"\n") if line.strip()]
        if log.returncode or ("logs/HEAD" in present and len(seen) != len(records)):
            why = log.stderr.strip()[:200] or f"Git reads {len(seen)} of its {len(records)} reflog records"
            return f"{entry}'s HEAD history could not be read ({why}){again}"
        # log -g names only new values. Expiry/deletion without --rewrite can leave a child's commit
        # only in an old-value field, which pruning this entry would also discard.
        old = [line.split(b" ", 1)[0].decode("ascii") for line in records]
        for oid in dict.fromkeys(old):
            if oid in (baseline, "0" * len(baseline)):
                continue
            if (len(oid) != len(baseline) or set(oid) - set("0123456789abcdef")
                    or _git(repo, "cat-file", "-t", oid, check=False).stdout.strip() != "commit"):
                return f"{entry}'s HEAD history could not be read (a reflog old value names no commit){again}"
        held = [oid for oid in dict.fromkeys(orig.stdout.split() + seen + old)
                if oid not in (baseline, "0" * len(baseline))]
        return (f"{entry} still records commit {held[0][:12]}{f' and {len(held) - 1} more' if held[1:] else ''} in "
                f"its HEAD history (ORIG_HEAD or the HEAD reflog): keep it if needed "
                f"({way('git', '-C', repo, 'branch', '<name>', held[0])}), {give_up}" if held else "")
    return ""


def result_loss(drive: Any, entry: _RunCustody) -> Tuple[Dict[str, Any], str]:
    """``(facts, "")`` when a terminal Git writer's result is proven gone for good, else ``({}, why)``
    (#1651; the owner chose an ordinary reject that records the result as lost).

    Asked only after the core above found nothing to capture or restore. Loss needs positive
    absence on every carrier, each read strictly: the private copy (ENOENT), its saved capture
    (none, or a readable failed one, whose file changes saved before a legacy diff failed stay
    and are counted as ``saved_files``), the copy record's result pin, the pinned ref itself and
    whatever Git's own entry for a hand-deleted checkout still holds: a commit the child made,
    changes it staged or its HEAD history (a repository that no longer exists holds none). A read
    failure, a malformed or corrupt record, a saved result or any surviving carrier proves nothing,
    and ``why`` names it with the way on. Skill payloads and engine directories keep their own
    reject contracts."""
    import os

    from ouroboros.delegate_directory import is_directory_run
    from ouroboros.subagent_worktrees import _RESULT_REF_PREFIX, _git, _records, _safe_name
    from ouroboros.workspace_file_outputs import file_output_changes

    if entry.authority_source == "skill_payload" or is_directory_run(entry) or not entry.execution_root:
        return {}, "a skill-payload or engine-directory result keeps its own reject contract"

    def unread(what: str, exc: BaseException) -> Tuple[Dict[str, Any], str]:
        return {}, f"{what} could not be read ({type(exc).__name__}: {exc}); call again once it can be read"

    def kept(what: str) -> Tuple[Dict[str, Any], str]:
        return {}, f"{what}; call again once what it names is restored or readable"

    manifest = custody.delegated_capture_dir(drive, entry.task_id, custody.capture_key(entry)) / "workspace_patch.json"
    try:
        os.lstat(entry.execution_root)
        return kept(f"its private copy {entry.execution_root} is present")
    except FileNotFoundError:
        pass
    except OSError as exc:
        return unread(f"its private copy {entry.execution_root}", exc)
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError(f"a JSON {type(saved).__name__}, not a manifest")
    except FileNotFoundError:
        saved = None
    except (OSError, ValueError) as exc:
        return unread(f"its saved capture {manifest}", exc)
    if saved is not None and not (saved.get("status") == "failed" and not saved.get("result_commit")
                                  and "authority_drift" not in saved):
        return kept(f"its saved capture {manifest} (status {str(saved.get('status') or '')!r}) may hold a result")
    try:  # what a failed legacy capture saved before its diff failed stays; only the rest can be lost
        files = file_output_changes(saved, manifest.parent) if saved else []
    except Exception as exc:
        return unread(f"the file changes its failed capture {manifest} saved", exc)
    try:
        record = next((row for row in _records(op="result_loss", strict=True)
                       if row.get("kind") == "delegated_exec" and row.get("snapshot_id") == entry.snapshot_id), None)
    except Exception as exc:
        return unread("the private-copy records", exc)
    if record and record.get("result_sha") and entry.capture_id and not entry.patch_captured:
        return {}, (f"its copy record pins {str(record['result_sha'])[:12]}, but this continuation never recorded a "
                    f"capture of its own, so that pin may be its predecessor {entry.continuation_of}'s result: it is "
                    "kept, neither restored as this run's result nor released. Closing this run here would discard "
                    "it, so repeating this call fails the same way")
    if record and (record.get("result_sha") or os.path.lexists(str(record.get("path") or ""))):
        return kept(f"its copy record still names result {record.get('result_sha') or record.get('path')}")
    lost = {"execution_root": entry.execution_root, "saved_capture": "failed" if saved else "none",
            **({"saved_files": len(files)} if files else {}), "copy_record": "without_result" if record else "absent"}
    repo = pathlib.Path(str((record or {}).get("git_dir") or entry.target_root or ""))
    ref = f"{_RESULT_REF_PREFIX}{_safe_name(entry.snapshot_id)}"
    if not str(repo).strip("."):
        return kept(f"no repository is recorded to rule out the result pin {ref}")
    try:
        os.lstat(repo)
    except FileNotFoundError:
        return lost, ""
    except OSError as exc:
        return unread(f"its repository {repo}", exc)
    try:  # the pin, then Git's own entry for the gone checkout
        pin = _git(repo, "for-each-ref", "--format=%(objectname)", ref, check=False)
        if pin.returncode:
            return {}, f"its repository {repo} could not be read ({pin.stderr.strip()[:200]}); call again once it can be read"
        if pin.stdout.strip():
            return kept(f"the result pin {ref} still names {pin.stdout.strip()[:12]} in {repo}")
        held = _gone_checkout_holds(repo, pathlib.Path(entry.execution_root).resolve(strict=False), entry.baseline_sha)
    except (OSError, ValueError) as exc:  # no git, the repository went meanwhile, undecodable metadata
        return unread(f"its repository {repo}", exc)
    return ({}, held) if held else (lost, "")


def _saved_words(drive: Any, entry: _RunCustody, count: int) -> str:
    """The file changes a failed legacy capture saved before its diff, named from its manifest."""
    manifest = custody.delegated_capture_dir(drive, entry.task_id, custody.capture_key(entry)) / "workspace_patch.json"
    try:
        names = [str(row["path"]) for row in json.loads(manifest.read_text(encoding="utf-8"))["file_output_changes"]]
    except Exception:  # the count stands without its names
        names = []
    shown = f" ({', '.join(names[:5])}{f' and {len(names) - 5} more' if names[5:] else ''})" if names else ""
    return (f"the {count} file change{'s' * (count != 1)} that failed capture saved{shown} stay{'s' * (count == 1)} "
            "with it, readable through manifest_read and never applied")


def lost_words(drive: Any, entry: _RunCustody, lost: Dict[str, Any]) -> str:
    """What a proven loss took and what stays (#1651), for the reject that records it."""
    if not (count := int(lost.get("saved_files") or 0)):
        return f"its private copy {entry.execution_root} is gone and no saved or pinned result names it, so nothing was kept"
    return (f"its private copy {entry.execution_root} is gone and its earlier capture failed before saving a patch, "
            f"so the changes it did not save are lost; {_saved_words(drive, entry, count)}")


def loss_refusal(drive: Any, entry: _RunCustody, lost: Dict[str, Any], why: str) -> Tuple[str, str]:
    """``(note, step)`` of a refusal over a gone copy: a proven loss names reject as the way out
    (and what it keeps); an unproven one gives ``why`` in full, carrier and way on included."""
    if not lost:
        return "its private copy is gone, but the loss is not proven", f"{why[:1].upper()}{why[1:]}. Nothing was released."
    if not (count := int(lost.get("saved_files") or 0)):
        return (f"its private copy {entry.execution_root} is gone and no saved or pinned result names it: the result "
                "is lost and there is nothing to apply",
                "integrate_delegated_patch(decision='reject') records the loss and closes this obligation.")
    words = _saved_words(drive, entry, count)
    return (f"its private copy {entry.execution_root} is gone and its earlier capture failed before saving a patch: "
            "the changes it did not save are lost and there is nothing to apply",
            f"{words[:1].upper()}{words[1:]}. integrate_delegated_patch(decision='reject') records the loss, keeps "
            "what that capture saved and closes this obligation.")


# -- exact skill-payload delegation (R1) ---------------------------------------
#
# The restored delegated coding target class: an ordinary top-level task selects
# ONE exact user-managed skill payload through the existing ResolvedResourceBinding
# vocabulary (root=skill_payload, bucket, skill_name), Claudexor edits a PRIVATE
# standalone Git snapshot of it, and the parent explicitly applies the captured
# harness-authored diff under a whole-payload content-hash CAS. No fourth write
# surface, no live-payload harness writes, no acting+repair composition.

# The profiles allowed to hold this authority. Mirrors the top-level principal
# set of the tool-access policy: child/acting/repair/ephemeral contexts keep
# their current gates and never acquire delegated payload mutation.
from ouroboros.tool_access import (
    _TOP_LEVEL_PRINCIPAL_PROFILES as _PAYLOAD_PRINCIPAL_PROFILES)


def payload_content_hash(payload_root: Any) -> str:
    """The whole-payload CAS hash over the skill-loader inventory, the bucket's
    per-install notes included: the snapshot copy sits outside data/skills/<bucket>/,
    so it and the live folder hash the same bytes. One definition for snapshot
    baseline, capture result, and apply-time CAS; raises ``SkillPayloadUnreadable``
    (fail-closed) exactly like the loader."""
    from ouroboros.skill_identity import payload_digest

    return payload_digest(pathlib.Path(str(payload_root)), bucket="")


def _payload_selector_refusal(selector_root: str, retry_of: Any, bucket: Any,
                              skill_name: Any) -> Optional[ToolResult]:
    """Argument-shape validation for delegate_start's exact-resource selector."""
    if selector_root and selector_root != "skill_payload":
        return _fail("delegate_start", "unsupported_root",
                     "root supports only 'skill_payload' (an installed user-managed "
                     "skill payload). Ordinary workspace delegation takes no root.")
    if selector_root and str(retry_of or "").strip():
        return _fail("delegate_start", "selector_on_retry",
                     "retry_of replays the RECORDED invocation, whose exact resource "
                     "binding is re-resolved from its durable record — a retry call "
                     "carries no selector. Drop root/bucket/skill_name to retry.")
    if selector_root and (not str(bucket or "").strip() or not str(skill_name or "").strip()):
        return _fail("delegate_start", "payload_selector_incomplete",
                     "root='skill_payload' requires bucket and skill_name.")
    if not selector_root and (str(bucket or "").strip() or str(skill_name or "").strip()):
        return _fail("delegate_start", "payload_selector_incomplete",
                     "bucket/skill_name select a skill payload only together with "
                     "root='skill_payload'.")
    return None


def payload_host_instructions(text: str, skill_name: str) -> str:
    """The payload-run variant of the host instructions (gate fix 3).

    The generic text bans modifying the host's skills — for a payload run that
    blanket ban would contradict the assignment itself, so it is narrowed and an
    explicit, truthful permission block is appended.
    """
    adjusted = text.replace(
        "do not touch the host's runtime controls or memory, or modify its skills",
        "do not touch the host's runtime controls or memory")
    return adjusted + (
        "\nPAYLOAD ASSIGNMENT: this working tree is a PRIVATE standalone copy of "
        f"the installed skill payload '{skill_name}'. Editing its user-authored "
        "files IS your assignment — the prohibition above does not cover this "
        "copy. Lifecycle/control files (manifest sidecars, review/grants/enabled "
        "state, .git internals) remain off-limits and any change to them is "
        "refused whole at apply. Your host reviews the resulting diff and "
        "explicitly applies it to the live payload afterwards; the skill's "
        "review then re-runs before the new content is relied on.")


def claimed_start_request(drive: pathlib.Path, *, actor_ctx: Any = None,
                          initial_bootstrap: bool = False, **request_row: Any) -> Tuple[bool, Dict[str, Any]]:
    """The pre-POST custody decision; each private capture has its own admission."""
    from ouroboros.delegate_start_claims import claimed_start_request as claim

    return claim(drive, actor_ctx=actor_ctx, initial_bootstrap=initial_bootstrap, **request_row)


def _payload_mutation_authority(
    ctx: ToolContext, drive: pathlib.Path, bucket: str, skill_name: str,
    binding: Any, access: str = "workspace_write",
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[ToolResult]]:
    """The payload counterpart of ``_mutation_authority`` (R1 item 1).

    Returns ``(run_shape, authority_record, refusal)``; the refusal is ``None`` on success. The authority is a FRESH
    ``ResolvedResourceBinding`` for ``skill_payload.write`` through the same one
    authorizer the registry uses; the record carries the host-minted semantic
    ``resource_ref`` that custody stores durably and retry/apply re-resolve.
    """
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tool_access import active_tool_profile, build_resolved_resource_binding

    if access == "readonly":
        return None, None, _fail(
            "delegate_start", "payload_delegation_forbidden",
            "The skill-payload selector requires write access; omit the selector for a readonly run.",
            definitely_unrun=True)
    b, s = str(bucket or "").strip(), str(skill_name or "").strip()
    if binding is None:
        # Policy BEFORE lookup (Fable F4): a caller whose profile cannot hold
        # payload-write authority gets an authority denial, not a target-
        # resolution flavor from the binding builder's ValueError.
        profile = str(active_tool_profile(ctx))
        if profile not in _PAYLOAD_PRINCIPAL_PROFILES:
            return None, None, _fail(
                "delegate_start", "payload_delegation_forbidden",
                "This is an AUTHORITY denial, not a lookup failure: only an "
                "ordinary top-level task may delegate skill-payload work. Child, "
                "acting, repair and ephemeral contexts keep their existing "
                "narrower gates.",
                profile=profile)
        try:
            binding = build_resolved_resource_binding(
                ctx, root="skill_payload", operation="write", path=".",
                bucket=b, skill_name=s)
        except ValueError as exc:
            return None, None, _fail(
                "delegate_start", "payload_target_unresolved",
                f"The selected skill payload could not be bound: {exc}. Delegation "
                "targets one EXISTING exact user-managed payload. For a NEW skill, "
                "create its manifest first (write_file root='skill_payload', "
                "bucket='external', path='SKILL.md'), then delegate the "
                "now-existing payload.",
                bucket=b, skill_name=s)
    if getattr(binding, "root", "") != "skill_payload" or getattr(binding, "operation", "") != "write":
        return None, None, _fail(
            "delegate_start", "payload_binding_mismatch",
            "delegate_start(subagent_id=..., prompt=..., root='skill_payload') "
            "requires a skill_payload.write "
            "binding; got "
            f"{getattr(binding, 'root', '')!r}/{getattr(binding, 'operation', '')!r}.")
    if getattr(binding, "profile", "") not in _PAYLOAD_PRINCIPAL_PROFILES:
        return None, None, _fail(
            "delegate_start", "payload_delegation_forbidden",
            "Only an ordinary top-level task may delegate skill-payload work. "
            "Child, acting, repair and ephemeral contexts keep their existing "
            "narrower gates.",
            profile=str(getattr(binding, "profile", "")))
    target = pathlib.Path(binding.base_path)
    if not target.is_dir():
        return None, None, _fail(
            "delegate_start", "payload_target_missing",
            "The selected skill payload does not exist. Create its manifest first "
            "(write_file root='skill_payload', bucket='external', path='SKILL.md'), "
            "then delegate the now-existing payload.",
            bucket=b, skill_name=s)
    if custody.custody_log_unreadable(drive):
        return None, None, _fail(
            "delegate_start", "replacement_custody_unknown",
            "Payload custody is unreadable; no new invocation was admitted.")
    record = {
        "target_root": str(target),
        "source": "skill_payload",
        "capture_mode": _CAPTURE_DELEGATED_SNAPSHOT,
        "state_drive_root": str(binding.state_drive_root),
        # The host-minted semantic reference: replayed by retry, consumed by the
        # owned apply rebind. ``payload_hash`` is filled by snapshot provisioning
        # (the pre-copy loader hash is the CAS baseline).
        "resource_ref": {
            "root": "skill_payload",
            "source": str(binding.source or ""),
            "skill_name": str(binding.skill_name or ""),
            "target_root": str(target),
            "payload_hash": "",
        },
    }
    return delegated_run_shape(True, access), record, None


def _provision_payload_snapshot(
    ctx: ToolContext, drive: pathlib.Path, record: Dict[str, Any], invocation_id: str,
) -> Tuple[Optional[Any], Optional[ToolResult]]:
    """Provision the standalone private Git snapshot for one payload run (R1 §9.2).

    Same durability contract as ``_provision_snapshot``: registered in the
    worktree registry and described by the baseline manifest BEFORE any start
    intent is recorded. On success the record's ``resource_ref.payload_hash``
    is filled with the pre-copy loader hash (the CAS baseline).
    """
    from ouroboros.subagent_worktrees import provision_payload_snapshot

    task_id = str(getattr(ctx, "task_id", "") or "")
    try:
        handle = provision_payload_snapshot(
            target_root=record["target_root"], task_id=task_id,
            snapshot_id=invocation_id)
    except Exception as exc:
        return None, _fail(
            "delegate_start", "execution_snapshot_failed",
            "A private standalone snapshot of the skill payload could not be "
            f"provisioned ({type(exc).__name__}: {exc}). The run was NOT started: "
            "a mutating delegated run executes only in its own snapshot, never "
            "in the live payload.",
            target_root=record["target_root"], definitely_unrun=True)
    record["resource_ref"]["payload_hash"] = handle.payload_hash
    _record_baseline_manifest(drive, task_id, invocation_id, handle,
                              payload_hash=handle.payload_hash,
                              capture_kind="skill_payload")
    return handle, None


def _rebind_payload_reference(
    ctx: ToolContext, resource_ref: Dict[str, Any], recorded_target: str, *,
    tool: str, context: str,
) -> Tuple[Optional[pathlib.Path], Optional[Any], Optional[ToolResult]]:
    """Re-resolve a recorded semantic payload reference against CURRENT authority.
    Retry and the owned apply (R1 item 3) require the fresh binding path to
    equal the recorded target — a moved/collided/removed skill is a typed
    refusal, never a write through a stale physical path. Returns
    ``(live_target, fresh_binding, refusal)``, the refusal ``None`` when it rebinds."""
    from ouroboros.tool_access import build_resolved_resource_binding

    ref = resource_ref if isinstance(resource_ref, dict) else {}
    source = str(ref.get("source") or "")
    skill = str(ref.get("skill_name") or "")
    recorded = _resolved(recorded_target or ref.get("target_root"))
    if not source or not skill or recorded is None:
        return None, None, _fail(
            tool, "payload_reference_incomplete",
            "This run's durable record carries no complete semantic payload "
            "reference, so its target cannot be re-authorized. Start a new "
            "delegation for the current payload.", context=context)
    try:
        binding = build_resolved_resource_binding(
            ctx, root="skill_payload", operation="write", path=".",
            bucket=source, skill_name=skill)
    except ValueError as exc:
        return None, None, _fail(
            tool, "payload_target_unresolved",
            "The recorded skill payload could not be re-bound under current "
            f"authority ({exc}). The skill may have been removed, renamed, or "
            "collided with another location since the run was recorded.",
            source=source, skill_name=skill, context=context)
    fresh = _resolved(binding.base_path)
    if fresh is None or fresh != recorded:
        return None, None, _fail(
            tool, "payload_target_moved",
            "The recorded skill payload no longer resolves to the same physical "
            "target under current authority. Refusing to act through a stale "
            "physical path.",
            recorded_target=str(recorded), current_target=str(fresh or ""),
            context=context)
    return fresh, binding, None


__all__ = [
    "_CAPTURE_DELEGATED_SNAPSHOT",
    "_RetryBinding",
    "_capture_block",
    "_capture_terminal_patch",
    "_fail",
    "_mutation_authority",
    "_provision_snapshot",
    "_resolve_retry_invocation",
    "_resolved",
    "_retry_binding_refusal",
    "_validated_invocation",
    "capture_terminal_patch_for_drive",
    "capture_stranded_patch",
    "claimed_start_request",
    "integrate_payload_patch",
    "payload_content_hash",
    "payload_host_instructions",
]


# v7next F2 (D07): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from ouroboros.tools.delegate_payload_patch import (  # noqa: E402, F401 -- intentional public re-exports
    _candidate_symlink_escapes,
    _finalize_payload_apply,
    _payload_reserved_paths,
    _reserved_payload_rel_path,
    _snapshot_head_textual,
    _write_payload_patch_artifacts,
    integrate_payload_patch,
)

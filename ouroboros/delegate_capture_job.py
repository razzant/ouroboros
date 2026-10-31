"""Addressed capture in the existing maintenance child; decisions remain with the server.

The server requests an immutable binding under the snapshot disposition lock. The child
holds that same cross-process lock while walking/copying/hashing, then publishes only an
artifact reference. The server reacquires the lock and rechecks the current binding and
continuation before accepting it. Neither a child exit nor a stopped observer is a receipt.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import uuid

from ouroboros import delegate_custody as custody, delegate_custody_current as current
from ouroboros.delegate_capture_attempt import GENERATION, POLICY, failure_retry

NAME = "delegate_capture"
READY = {"ready", "ready_with_changes", "ready_no_changes"}


def identity(entry):
    facts = {key: getattr(entry, key) for key in (
        "run_id", "task_id", "invocation_id", "snapshot_id", "capture_id", "snapshot_task_id",
        "continuation_of", "superseded_by", "execution_root", "target_root", "baseline_sha",
        "resource_ref", "authority_source", "mode", "patch_disposed")}
    facts.update(terminal_sha=entry.terminal_source.get("sha256"), policy=POLICY,
                 named_source=entry.named_evidence.get("inputs"))
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()


def _record(root, entry, outcome):
    return custody.emit(root, custody.CAPTURE_JOB_CHANGED, {
        "run_id": entry.run_id, "task_id": entry.task_id,
        "expected_id": entry.capture_job.get("id", ""), "outcome": outcome})


def request_or_result(root, entry, *, generation, recapture=False):
    """Server-only, under its disposition lock. Return prepared bytes or owe one child job."""
    from ouroboros.delegate_collection import _read_source

    job, binding = entry.capture_job, identity(entry)
    recapture |= entry.named_evidence.get("status") == "unreadable"
    if job.get("binding") == binding and (job.get("state") == "ready"
                                         or job.get("state") == "consumed" and not recapture):
        try:
            result = _read_source(root, entry, job["source"])
        except (OSError, ValueError, KeyError):
            pass  # the result receipt was lost: a new attempt is useful
        else:
            if not result.get("capture_interrupted"):
                return result
            accept_result(root, entry, result)
            entry = current.run(root, entry.run_id)
            job = entry.capture_job  # keep the prepared patch; only named retention still waits
    if job.get("binding") != binding or job.get("state") != "pending":
        job = {"id": uuid.uuid4().hex, "binding": binding, "state": "pending",
               "generation": generation, "recapture": recapture}
        if not _record(root, entry, job):
            raise OSError("capture request publication failed")
    # collect_terminal wakes maintenance after releasing its publication/lock boundary.
    return {"status": "pending", "note": "The maintenance child owes this capture; no patch was applied."}


def accept_result(root, entry, result):
    """Server-only, after its fresh continuation check, still under the disposition lock."""
    from ouroboros.delegate_named_evidence import _publish
    from ouroboros.delegate_continuation import disposition_refusal

    entry = current.run(root, entry.run_id)
    if entry.capture_job.get("binding") != identity(entry) or disposition_refusal(root, entry):
        return False
    if entry.capture_job.get("state") == "consumed":
        return True
    if entry.capture_job.get("state") != "ready":
        return False
    if result.get("status") in READY:
        payload = {key: result[key] for key in ("status", "sha256", "patch_size", "execution_root", "no_changes")
                   if key in result}
        payload["capture_dir"] = str(pathlib.Path(result["manifest_artifact"]).parent)
        if not custody.record_patch_captured(root, entry, **payload):
            raise OSError("patch receipt publication failed")
    named = result.get("named_evidence") or {}
    if named:
        _publish(root, entry, pathlib.Path(named["evidence_dir"]).name, named)
    if not _record(root, entry, {**entry.capture_job, "state": "consumed", "binding": identity(entry),
                                "retry": result.get("retry") or named.get("retry", "")}):
        raise OSError("capture result acknowledgement failed")
    return True


def _pending(root, *, child=False):
    pending = "json_extract(facts, '$.capture_job.state')='pending'"
    params = []
    if child:
        pending += (" OR (json_extract(facts, '$.capture_job.state')='consumed'"
                    " AND json_extract(facts, '$.capture_job.retry')='transient'"
                    " AND COALESCE(json_extract(facts, '$.capture_job.child_generation'), '')<>?)")
        params.append(GENERATION)
    return current.held_runs(root, pending, tuple(params))


def due(root, row, facts, child):
    # A transient does not spawn a child; another real child generation may retry it.
    return bool(_pending(root, child=child))


def _capture(root, entry, should_stop):
    from ouroboros.delegate_collection import _retain
    from ouroboros.delegate_continuation import disposition_refusal
    from ouroboros.delegate_named_evidence import run_named_evidence
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    path = custody.disposition_lock_path(root, entry)
    # Only the child may wait: reclaiming a dead owner's name needs another acquisition,
    # and the server that just requested capture may still be publishing its attempt.
    fd = acquire_exclusive_file_lock(path, owner_aware_stale=True)
    if fd is None:
        return False
    try:
        entry = current.run(root, entry.run_id)
        if entry is None or should_stop():
            return False
        request = entry.capture_job
        if (request.get("state") == "consumed" and request.get("retry") == "transient"
                and request.get("child_generation") != GENERATION
                and request.get("binding") == identity(entry) and not disposition_refusal(root, entry)):
            request = {"id": uuid.uuid4().hex, "binding": identity(entry), "state": "pending",
                       "generation": request.get("generation"), "recapture": True}
            if not _record(root, entry, request):
                raise OSError("capture retry publication failed")
            entry.capture_job = request
        if request.get("state") != "pending":
            return False
        if request.get("binding") != identity(entry) or disposition_refusal(root, entry):
            result = {"status": "superseded", "note": "The capture binding changed before collection."}
        else:
            try:
                result = capture_terminal_patch_for_drive(root, entry, retain_investigation=False, publish=False) or {}
                if should_stop():
                    result["capture_interrupted"] = True  # retain the completed patch for revival
                elif result.get("status") in READY:
                    named = run_named_evidence(root, entry, recapture=request.get("recapture", False), publish=False)
                    if named:
                        result["named_evidence"] = named
            except Exception as exc:
                result = {"status": "failed", "note": f"{type(exc).__name__}: {exc}",
                          "retry": failure_retry(exc)}
        source = _retain(root, entry, result, "capture-" + request["id"])
        return _record(root, entry, {**request, "state": "ready", "source": source,
                                    "child_generation": GENERATION})
    finally:
        release_exclusive_file_lock(path, fd)


def run(root, should_stop):
    from ouroboros import maintenance_jobs

    done = 0
    for entry in _pending(root, child=True).values():
        if should_stop():
            break
        done += int(_capture(root, entry, should_stop))
    state = "pending" if _pending(root) else "done"
    if done:
        maintenance_jobs.record(root, NAME, state=state, done=done)
    return {"state": state, "items": done}

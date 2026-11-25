"""Addressed capture in the existing maintenance child; decisions remain with the disposition-lock holder.

``collect_terminal`` requests an immutable binding under the snapshot disposition lock. The
child holds that same cross-process lock while walking/copying/hashing, then publishes only an
artifact reference. A later holder of the lock, ``collect_terminal`` or an explicit Git
disposition (``accept_ready_patch``), rechecks the current binding and continuation before
accepting it. Neither a child exit nor a stopped observer is a receipt.
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


def identity(entry, *, named_evidence=None):
    facts = {key: getattr(entry, key) for key in (
        "run_id", "task_id", "invocation_id", "snapshot_id", "capture_id", "snapshot_task_id",
        "continuation_of", "superseded_by", "execution_root", "target_root", "baseline_sha",
        "resource_ref", "authority_source", "mode", "patch_disposed")}
    named = entry.named_evidence if named_evidence is None else named_evidence
    facts.update(terminal_sha=entry.terminal_source.get("sha256"), policy=POLICY, named_source=named.get("inputs"))
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()


def _bound(entry, binding):
    job = entry.capture_job
    # A failure after publishing this job's own named receipt must not turn its
    # still-ready result into a new job. Other input changes still matter.
    return job.get("binding") == binding or (
        job.get("state") == "ready" and job.get("published_binding") == binding
        and job.get("named_directory") == entry.named_evidence.get("directory"))


def ready_result_owed(entry):
    return entry.capture_job.get("state") == "ready" and _bound(entry, identity(entry))


def ordinary_git(entry):
    """A Git copy's capture, apart from a skill payload's and an engine directory's own contracts."""
    from ouroboros.delegate_directory import is_directory_run

    return entry.authority_source != "skill_payload" and not is_directory_run(entry)


def manifest_recorded(entry, raw):
    """An ordinary Git receipt binds exact manifest bytes; legacy rows, payloads and directories keep the flag."""
    sha = getattr(entry, "patch_manifest_sha", "") if ordinary_git(entry) else ""
    return entry.patch_captured and (not sha or sha == hashlib.sha256(raw).hexdigest())


def _record(root, entry, outcome):
    return custody.emit(root, custody.CAPTURE_JOB_CHANGED, {
        "run_id": entry.run_id, "task_id": entry.task_id,
        "expected_id": entry.capture_job.get("id", ""), "outcome": outcome})


def request_or_result(root, entry, *, generation, recapture=False):
    """Called by ``collect_terminal`` under the run's disposition lock. Return prepared bytes or owe one child job."""
    from ouroboros.delegate_collection import _read_source

    job, binding = entry.capture_job, identity(entry)
    recapture |= entry.named_evidence.get("status") == "unreadable"
    if _bound(entry, binding) and (job.get("state") == "ready"
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


def accept_ready_patch(root, entry, *, custody_held=None):
    """An explicit Git disposition joins a bound, digest-stamped child result before recapturing."""
    from ouroboros.delegate_collection import _read_source

    if not entry.capture_job.get("manifest_sha256") or not ready_result_owed(entry):
        return None
    result = _read_source(root, entry, entry.capture_job["source"])
    if result.get("status") not in READY:
        return None
    if not accept_result(root, entry, result, custody_held=custody_held):
        raise ValueError("prepared capture binding changed before publication")
    return result


def accept_result(root, entry, result, *, custody_held=None):
    """The disposition-lock holder's acceptance (``collect_terminal``, or an explicit Git disposition through
    ``accept_ready_patch``), after its fresh continuation check, still under that lock."""
    from ouroboros.delegate_named_evidence import publish_saved, said_released, saved_named_evidence
    from ouroboros.delegate_continuation import disposition_refusal

    entry = current.run(root, entry.run_id)
    if entry is None or not _bound(entry, identity(entry)) or disposition_refusal(root, entry):
        return False
    named = result.get("named_evidence") or {}
    if named.get("reason") in {"release_state_unavailable", "payload_release_pending"}:
        _, latest = saved_named_evidence(root, entry, observe=False)
        named = result["named_evidence"] = latest or {"status": "partial", "retry": "transient", "files": []}
    elif named and entry.capture_job.get("state") == "consumed":
        named = result["named_evidence"] = said_released(named)
    if entry.capture_job.get("state") == "consumed":
        return True
    if entry.capture_job.get("state") != "ready":
        return False
    if result.get("status") in READY:
        payload = {key: result[key] for key in ("status", "sha256", "patch_size", "execution_root", "no_changes")
                   if key in result}
        path = pathlib.Path(result["manifest_artifact"])
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if entry.capture_job.get("manifest_sha256", sha) != sha:
            raise ValueError("prepared capture manifest changed before publication")
        if ordinary_git(entry) and not manifest_recorded(entry, raw):
            # The same proof a reopened Git capture needs: these bytes are the run's pinned attempt.
            from ouroboros.tools.delegate_integration import _declared_capture_gap

            gap, retry = _declared_capture_gap(entry, path.parent, raw, json.loads(raw.decode("utf-8")), root)
            if gap:
                raise (OSError if retry == "transient" else ValueError)(f"prepared capture not published: {gap}")
        if (not entry.patch_captured or entry.patch_manifest_sha != sha) and not custody.record_patch_captured(
                root, entry, **payload, capture_dir=str(path.parent), manifest_sha256=sha):
            raise OSError("patch receipt publication failed")
    if named.get("evidence_dir") and not named.get("reason"):
        named = result["named_evidence"] = publish_saved(
            root, entry, named, pathlib.Path(named["evidence_dir"]), custody_held=custody_held)
    if named.get("reason"):
        return True  # transport and patch remain usable; no named receipt or consumed acknowledgement
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


# What this process's last child look offered its capture pass: (root, ``ready_requests`` keys). A maintenance child
# runs one pass, so this is that pass's opportunity: process memory, never an installation fact or an attempt claim.
_offered = ("", frozenset())


def _keys(pending):
    return frozenset((run_id, str(entry.capture_job.get("id") or "")) for run_id, entry in pending.items())


def _child_look(root):
    """``_pending(child=True)`` as the child's capture pass is offered it (its due look, then its run)."""
    global _offered
    pending = _pending(root, child=True)
    _offered = (str(root), _keys(pending))
    return pending


def due(root, row, facts, child):
    # A transient does not spawn a child; another real child generation may retry it.
    return bool(_child_look(root) if child else _pending(root))


def ready_requests(root):
    """The requests a maintenance child would take up now (``_pending(child=True)``), as ``(run_id, request id)``."""
    return _keys(_pending(root, child=True))


def offered(root):
    """What the capture pass of this process's child was offered at ROOT (``_child_look``). A later job of that child
    yields only to a ready request outside it: one the pass was offered but could not take up (its disposition lock
    held) is the next child's anyway. Without such a look, every ready request is one the pass never had."""
    return _offered[1] if _offered[0] == str(root) else frozenset()


def _capture(root, entry, should_stop):
    from ouroboros.delegate_collection import _retain
    from ouroboros.delegate_continuation import disposition_refusal
    from ouroboros.delegate_named_evidence import run_named_evidence
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    path = custody.disposition_lock_path(root, entry)
    # Only the child may wait: reclaiming a dead owner's name needs another acquisition,
    # and the collector that just requested capture may still be publishing its attempt.
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
        extra = {}
        if result.get("status") in READY:
            try:
                extra["manifest_sha256"] = hashlib.sha256(pathlib.Path(result["manifest_artifact"]).read_bytes()).hexdigest()
            except OSError as exc:
                result = {"status": "failed", "note": str(exc), "retry": failure_retry(exc)}
            else:
                named = result.get("named_evidence") or {}
                if named.get("evidence_dir") and "inputs" in named:
                    extra.update(published_binding=identity(entry, named_evidence=named),
                                 named_directory=pathlib.Path(named["evidence_dir"]).name)
        source = _retain(root, entry, result, "capture-" + request["id"])
        return _record(root, entry, {**request, "state": "ready", "source": source,
                                    "child_generation": GENERATION, **extra})
    finally:
        release_exclusive_file_lock(path, fd)


def run(root, should_stop):
    from ouroboros import maintenance_jobs

    done = 0
    for entry in _child_look(root).values():
        if should_stop():
            break
        done += int(_capture(root, entry, should_stop))
    state = "pending" if _pending(root) else "done"
    if done:
        maintenance_jobs.record(root, NAME, state=state, done=done)
    return {"state": state, "items": done}

"""Eligibility of mechanical collection, separate from the custody it still owes.

The receipt lives on RunCustody. A terminal revision, binding or observed named-source
change earns work; a clock does not. Transient failures may try once in a new body/child
generation. Readers disclose debt without requesting another attempt.
"""
from __future__ import annotations

import copy
import errno
import uuid

POLICY = "immutable-named-evidence-v1"
GENERATION = uuid.uuid4().hex


def failure_retry(exc):
    """Retry typed transport/IO failures; permission repairs need new input or an explicit attempt."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable, DAEMON_BUSY_CODES

    if isinstance(exc, ClaudexorUnavailable):
        if exc.status_code in (410, 413):
            return "changed_input"  # Reclaimed/over-cap bytes cannot heal with another generation.
        problem = getattr(exc, "problem", {})
        retryable = problem.get("retryable") if isinstance(problem, dict) else None
        if retryable is None:
            retryable = exc.observation_timeout or exc.code in DAEMON_BUSY_CODES
    elif isinstance(exc, OSError):
        retryable = (getattr(exc, "winerror", None) in (32, 33) or not (
            isinstance(exc, PermissionError) or exc.errno in (errno.EACCES, errno.EPERM)))
    else:
        retryable = False
    return "transient" if retryable else "changed_input"


def output_owed(entry):
    return (not entry.output_source or not entry.output_complete
            or entry.output_terminal_sha != entry.terminal_source.get("sha256"))


def capture_required(entry):
    from ouroboros.delegate_directory import is_in_place_run

    return (entry.mode != "plan" and not entry.patch_disposed and not entry.superseded_by
            and not is_in_place_run(entry)  # its edits are already in place: nothing to capture
            and bool(entry.execution_root or entry.resource_ref.get("workspace_kind") == "directory"))


def named_required(entry):
    return (capture_required(entry) and bool(entry.snapshot_id and entry.execution_root)
            and entry.authority_source != "skill_payload"
            and entry.resource_ref.get("workspace_kind") != "directory")


def legacy_cap(entry):
    """A named-evidence receipt made under the retired 64 MiB budget: its gapped files are
    still only in the copy, so it earns one whole retention before the copy may go."""
    return bool((entry.named_evidence.get("inputs") or {}).get("limit_bytes"))


def inputs(entry):
    """Only addressed facts: never stat a snapshot or decode a summary to decide a tick."""
    return {**({"named_legacy_cap": True} if legacy_cap(entry) else {}),
            **{key: copy.deepcopy(getattr(entry, key)) for key in (
        "run_id", "invocation_id", "snapshot_id", "capture_id", "continuation_of", "superseded_by",
        "execution_root", "baseline_sha", "authority_source", "resource_ref", "mode")},
        "terminal_sha": entry.terminal_source.get("sha256"), "policy": POLICY,
        "named_source": copy.deepcopy(entry.named_evidence.get("inputs") or {}),
        "named_status": entry.named_evidence.get("status"),
        "capture_job": {key: entry.capture_job.get(key) for key in ("id", "state")},
        "output_sha": entry.output_sha, "output_complete": entry.output_complete,
        "output_terminal_sha": entry.output_terminal_sha, "patch_captured": entry.patch_captured,
        "patch_disposed": entry.patch_disposed, "ledger_recorded": entry.ledger_recorded, "settled": entry.settled}


def collection_due(entry, *, generation=GENERATION, recapture=False):
    """One rule for scheduling and collection. Missing/malformed outcomes earn one attempt."""
    if entry is None or entry.review_owned or not entry.terminal_source:
        return False
    attempt = entry.collection_attempt
    return (recapture or not isinstance(attempt, dict) or not attempt.get("inputs")
            or attempt["inputs"] != inputs(entry)
            or (attempt.get("retry") == "transient" and attempt.get("generation") != generation))


def record_attempt(root, entry, *, generation=GENERATION, retry="", errors=(), output=None):
    from ouroboros import delegate_custody as custody

    outcome = {"inputs": inputs(entry), "generation": generation, "retry": retry, "errors": list(errors)}
    output = entry.collection_attempt.get("output") if output is None else output
    if output:
        outcome["output"] = output
    if entry.collection_attempt == outcome:
        return True
    return custody.emit(root, custody.COLLECTION_ATTEMPTED,
                        {"run_id": entry.run_id, "task_id": entry.task_id, "outcome": outcome})

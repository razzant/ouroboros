"""Artifact and task lifecycle vocabulary shared by the headless owners.

The artifact-status values a task result may carry, the terminal subset the
pruners and the copy-back gate test against, the lifecycle fields a late child
copy-back must preserve, and the two literals headless mirrors instead of
importing (settled task statuses, the local-readonly subagent mode) because a
module-level import of their SSOT would close an import cycle. Constants only:
every consumer of this vocabulary owns its own behaviour.
"""

from __future__ import annotations


ARTIFACT_STATUS_PENDING = "pending"
ARTIFACT_STATUS_FINALIZING = "finalizing"
ARTIFACT_STATUS_READY = "ready"
ARTIFACT_STATUS_READY_WITH_CHANGES = "ready_with_changes"
ARTIFACT_STATUS_READY_NO_CHANGES = "ready_no_changes"
ARTIFACT_STATUS_MISSING = "missing"
ARTIFACT_STATUS_FAILED = "failed"


ARTIFACT_TERMINAL_STATUSES = {
    ARTIFACT_STATUS_READY,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
    ARTIFACT_STATUS_READY_NO_CHANGES,
    ARTIFACT_STATUS_MISSING,
    ARTIFACT_STATUS_FAILED,
}


# Mirrors task_status.SETTLED_STATUSES; a module-level import would close the
# headless → task_status → outcomes → headless cycle, and the smoke test below
# pins equality so the literal cannot drift from the SSOT.
_FINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "rejected_duplicate"})


# Mirrors tool_capabilities.LOCAL_READONLY_SUBAGENT_MODE; a module-level import would risk
# an import cycle (same rationale as _FINAL_STATUSES above), and the smoke test pins equality
# so the literal cannot drift from this SSOT — the kind of re-derivation drift that stranded
# the reaper's artifact finalization before task_is_readonly_subagent consolidated the gate.
_LOCAL_READONLY_SUBAGENT_MODE = "local_readonly_subagent"


_ARTIFACT_LIFECYCLE_FIELDS = {
    "artifact_status",
    "artifact_error",
    "artifact_bundle",
    "artifact_finalized_at",
}

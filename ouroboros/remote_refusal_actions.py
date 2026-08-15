"""THE register of WHICH OWNER ACTION REMOVES a remote refusal, and nothing else.

`remote_contracts` already says it: "A refusal without an action is a dead end."
This module is the other half of that sentence, and it exists because a refusal
with the WRONG action is worse than one with none. The owner presses what they
were told to press, it succeeds, and the block is still there — so the surface
has taught them that the system's advice is noise.

The defect that produced this module, observed live: a connection whose installed
execd predated a shared-contract change was unselectable in New Project, and the
picker said

    No bootstrapped healthy connections — run Bootstrap (or Test to refresh
    health) in Settings → Connections

`Test` was one of the two offered actions, `Test` returned `ok` with
`health_fresh: true`, and the connection stayed unselectable — because the
contract-set stamp is written by Bootstrap alone. The message had enumerated every
CONCEIVABLE cure instead of naming the one that would move THIS block.

The same shape had already been caught twice in this feature: every typed
`RemoteWorkspaceError` reported `action: "retry"` regardless of code (including
"your execd is too old to talk to"), and `REMOTE_EXECUTION_UNAVAILABLE` printed a
bare `ValueError` with no next step at all. Both were patched at the raise site.
That is why this is a REGISTER and not a third patch: the action belongs to the
CODE, so a refusal cannot be raised without one and a new code cannot be added
without deciding what cures it.

Two things live here and only two.

**Code → action.** :data:`REFUSAL_ACTIONS` maps every refusal code this feature
can emit to the single owner action that removes it.
`workspace_diagnostics.RemoteWorkspaceError` reads it in `__init__`, so ~40 raise
sites gained a correct action without one of them being edited — and a code that
is not in the table is a test failure, not a silent `retry`
(`tests/test_remote_refusal_action_proofs.py`).

The table is also checkable against an INDEPENDENT authority that already exists:
`cli_connections._UNSERVABLE_CODES` is the set the owner CLI maps to exit 4,
meaning "retrying will not help". A code in that set whose action is `retry` is a
self-contradiction the suite can find mechanically, and it found fourteen.

**Which blocker is in front of a connection.** :func:`connection_blocker` answers
"why can this connection not carry a new remote project, and what removes THAT".
One ordered ladder, server-side, so the picker, Settings → Connections and the
CLI cannot each hold a slightly different opinion about selectability — the
browser's `isSelectableRemoteConnection` is now provably the same predicate as
"this row has no blocker" (asserted both ways in the proofs).

Stdlib only, and no imports from either half beyond the two action names
`remote_contracts` already owns, so both Home and execd may cite it — the same
property that lets `remote_contracts` and `export_policy_contract` travel.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ouroboros.remote_contracts import ACTION_BOOTSTRAP, ACTION_REBUILD_BUNDLE

# ── the action vocabulary ────────────────────────────────────────────────────
# Every value is something the OWNER can do, named the way the surface that
# offers it names it. `ACTION_BOOTSTRAP` and `ACTION_REBUILD_BUNDLE` are imported
# rather than restated: two spellings of one action is how a surface starts
# advising a button that does not exist.
ACTION_RETRY = "retry"
ACTION_TEST = "test_connection"
ACTION_RETRUST = "retrust_host"
# Retiring is TERMINAL — there is no un-retire — so whatever wanted a retired
# connection has to be pointed at an active one instead. One action name for both
# concrete routes (add a connection, or `ouroboros projects rebind` an existing
# project onto one), because the gateway used to answer this single condition with
# `rebind_project` at two seams and `choose_active_connection` at a third, and an
# owner told two different things about one fact has to work out that they agree.
ACTION_USE_ACTIVE_CONNECTION = "use_active_connection"
ACTION_ADD_CONNECTION = "add_connection"
ACTION_READMIT_PROJECT = "readmit_project"
ACTION_RECONNECT = "reconnect_connection"
ACTION_RESTART = "restart_ouroboros"
ACTION_START_NEW_TASK = "start_new_task"
ACTION_CHOOSE_GIT_ROOT = "choose_git_worktree_root"
ACTION_CHOOSE_HOST = "choose_supported_host"
ACTION_REPAIR_HOST = "repair_remote_host"
ACTION_CANCEL_TASKS = "cancel_active_tasks"
# NOT a menu, and the distinction is what this whole module is about: a rebind blocked
# by live tasks clears when they finish OR when the owner cancels them, and BOTH
# genuinely remove it. What the class forbids is offering an action that CANNOT move
# the block — not naming two that can.
ACTION_WAIT_OR_CANCEL_TASKS = "wait_or_cancel_tasks"
ACTION_RELOAD_CONNECTIONS = "reload_connections"
ACTION_RELOAD_PROJECTS = "reload_projects"
ACTION_REPAIR_STORE = "repair_connection_store"
ACTION_CONFIRM_RETRUST = "confirm_retrust"
ACTION_LIST_CONNECTIONS = "list_connections"
ACTION_RUN_FROM_TERMINAL = "run_from_controlling_terminal"
# The owner GATE in front of this whole surface (`server_auth.NetworkAuthGate`) had its
# own two names for its own two conditions, and they belong in the same vocabulary as
# everything they gate: a browser that renders `action` cannot tell which middleware
# wrote it. Note that `owner_auth_required` legitimately resolves differently per
# surface — the page unlocks in place, the CLI must be run from a controlling terminal
# so a password can be read at all — so the CLI keeps `ACTION_RUN_FROM_TERMINAL` for its
# OWN local refusal. Two contexts, two actions, one vocabulary.
ACTION_AUTHENTICATE_OWNER = "authenticate_owner"
ACTION_CONFIGURE_NETWORK_PASSWORD = "configure_network_password"
# A refusal the owner CAUSED by declining a confirmation. The honest next step is
# nothing at all, and saying so is different from having nothing to say.
ACTION_NONE = "none"
# The honest answer when the owner can do NOTHING about it. A Home-side wiring
# failure — a transport that does not expose an operation its own contract
# declares, a response that does not match the request it answers — is not cured
# by any button, and telling the owner to retry a structural absence is the dead
# end this module exists to forbid. Naming it as a defect is the only true thing
# to say, and it is deliberately not spelled `retry`.
ACTION_REPORT_DEFECT = "report_defect"

# ── code → the one action that removes it ────────────────────────────────────
# Grouped by WHY, because the grouping is the argument. Every entry is a claim
# that can be checked, and the proofs check the checkable ones by performing the
# action and asserting the block is gone.
#
# Keyed in the WIRE spelling — lower case — because two authorities disagree on
# case by construction (`workspace_admission.REMOTE_TRANSPORT_UNAVAILABLE` and
# `remote_worker_proxy.BROKER_GENERATION_STALE` are Python constants and read as
# upper case; a code the transport raises is already lower). Every lookup
# normalizes with `.strip().lower()`, the same normalization `wire_error_code`
# performs at the HTTP boundary — spelled inline rather than imported, because
# this module travels into the execd bundle and may not reach a gateway.
#
# Codes deliberately ABSENT: the pure 400s (`invalid_request`,
# `invalid_remote_placement`) whose message already IS the instruction — "send
# both halves of a placement" needs no second field naming an action, and giving
# it one would mean choosing between two unrelated causes the code covers.
REFUSAL_ACTIONS: dict[str, str] = {
    # ── Bootstrap is the cure: the installed executor is the stale artifact ──
    # Bootstrap installs the release THIS Home ships and repoints `current`, so it
    # resolves a peer that is behind AND one that is ahead. Test cannot: only
    # Bootstrap writes the contract-set stamp the picker reads.
    "remote_execd_outdated": ACTION_BOOTSTRAP,
    "remote_execd_incompatible": ACTION_BOOTSTRAP,
    "remote_contract_unknown_member": ACTION_BOOTSTRAP,
    "capability_mismatch": ACTION_BOOTSTRAP,
    "execd_artifact_mismatch": ACTION_BOOTSTRAP,
    "incompatible_protocol": ACTION_BOOTSTRAP,
    "execd_preamble_invalid": ACTION_BOOTSTRAP,
    "execd_release_unselected": ACTION_BOOTSTRAP,
    "connection_not_bootstrapped": ACTION_BOOTSTRAP,
    # ── the BUNDLE is the stale artifact, so Bootstrap changes nothing ────────
    "execd_bundle_contract_outdated": ACTION_REBUILD_BUNDLE,
    "execd_bundle_invalid": ACTION_REBUILD_BUNDLE,
    "execd_bundle_unavailable": ACTION_REBUILD_BUNDLE,
    # ── host trust ───────────────────────────────────────────────────────────
    # A different machine is answering the alias. Nothing else may proceed until
    # the owner decides whether that was intentional.
    "host_identity_changed": ACTION_RETRUST,
    "host_identity_mismatch": ACTION_RETRUST,
    "host_identity_confirmation_stale": ACTION_TEST,
    "host_identity_missing": ACTION_BOOTSTRAP,
    # ── the host itself cannot run this executor, and no action here will ─────
    "remote_platform_unsupported": ACTION_CHOOSE_HOST,
    "remote_libc_unsupported": ACTION_CHOOSE_HOST,
    "remote_glibc_too_old": ACTION_CHOOSE_HOST,
    "unsupported_ssh_client": ACTION_REPAIR_HOST,
    # ── owner state: the placement points somewhere it may not ───────────────
    "connection_retired": ACTION_USE_ACTIVE_CONNECTION,
    "connection_not_found": ACTION_ADD_CONNECTION,
    "connection_conflict": ACTION_TEST,
    "connection_store_unavailable": ACTION_REPAIR_STORE,
    "active_lease": ACTION_CANCEL_TASKS,
    "confirmation_required": ACTION_CONFIRM_RETRUST,
    "confirmation_declined": ACTION_NONE,
    "owner_action_required": ACTION_RUN_FROM_TERMINAL,
    "owner_auth_required": ACTION_AUTHENTICATE_OWNER,
    # Configuring the password IS the action; the restart it needs is part of applying
    # it, and `restart_ouroboros` alone would send the owner to reboot into the same
    # unconfigured state.
    "owner_auth_not_configured": ACTION_CONFIGURE_NETWORK_PASSWORD,
    # The project surface's own two, sharing this vocabulary rather than inventing a
    # parallel one: a browser that renders `action` cannot tell which module wrote it.
    "project_has_live_tasks": ACTION_WAIT_OR_CANCEL_TASKS,
    "project_routing_generation_changed": ACTION_RELOAD_PROJECTS,
    # ── the session, not the connection ──────────────────────────────────────
    # Reconnect repairs a session that exists; readmit is for one that does not.
    # These are NOT interchangeable, which is why they are separate actions.
    "remote_session_disconnected": ACTION_RECONNECT,
    "remote_session_absent": ACTION_READMIT_PROJECT,
    "remote_session_ambiguous": ACTION_READMIT_PROJECT,
    "workspace_identity_mismatch": ACTION_READMIT_PROJECT,
    "workspace_not_found": ACTION_READMIT_PROJECT,
    "reconnect_identity_mismatch": ACTION_READMIT_PROJECT,
    "broker_generation_stale": ACTION_RESTART,
    # ── the request named the wrong place ────────────────────────────────────
    "workspace_root_mismatch": ACTION_CHOOSE_GIT_ROOT,
    "workspace_not_git": ACTION_CHOOSE_GIT_ROOT,
    "workspace_not_git_root": ACTION_CHOOSE_GIT_ROOT,
    "invalid_remote_root": ACTION_CHOOSE_GIT_ROOT,
    # ── this task's binding is spent; a retry replays the same binding ───────
    "task_session_unbound": ACTION_START_NEW_TASK,
    "task_session_mismatch": ACTION_START_NEW_TASK,
    "prepared_call_expired": ACTION_START_NEW_TASK,
    "prepared_arguments_mismatch": ACTION_START_NEW_TASK,
    "attachment_manifest_changed": ACTION_START_NEW_TASK,
    "export_arguments_changed": ACTION_START_NEW_TASK,
    # ── no broker/transport in THIS process ──────────────────────────────────
    # Three phase-specific spellings of one condition, kept distinct on purpose so a
    # diagnosis can name WHICH phase refused (admission / prepare / execute). They
    # answer with one action because there is only one: this process has no broker.
    "remote_workspace_unavailable": ACTION_RESTART,
    "remote_service_unavailable": ACTION_RESTART,
    "remote_transport_unavailable": ACTION_RESTART,
    "ssh_executor_unavailable": ACTION_RESTART,
    "broker_closed": ACTION_RESTART,
    "broker_pipe_closed": ACTION_RESTART,
    # A fact the prepare bundle does not carry: the two builds disagree about what a
    # prepare returns, which is a contract-set gap Bootstrap closes.
    "ssh_facts_unavailable": ACTION_BOOTSTRAP,
    # ── Home-side wiring: no owner action moves these ────────────────────────
    "bootstrap_unsupported": ACTION_REPORT_DEFECT,
    "directory_listing_unsupported": ACTION_REPORT_DEFECT,
    "reconnect_unsupported": ACTION_REPORT_DEFECT,
    "broker_pipe_protocol": ACTION_REPORT_DEFECT,
    "prepared_response_invalid": ACTION_REPORT_DEFECT,
    "remote_result_invalid": ACTION_REPORT_DEFECT,
    "remote_blob_invalid": ACTION_REPORT_DEFECT,
    "remote_service_invalid_response": ACTION_REPORT_DEFECT,
    "attachment_execution_path_invalid": ACTION_REPORT_DEFECT,
    # ── genuinely transient: a retry really is the action ────────────────────
    "broker_overloaded": ACTION_RETRY,
    "broker_pipe_timeout": ACTION_RETRY,
    "remote_request_timeout": ACTION_RETRY,
    "admission_cancelled": ACTION_RETRY,
    "remote_service_error": ACTION_RETRY,
    "connection_health_stale": ACTION_TEST,
    "connection_degraded": ACTION_REPAIR_HOST,
    "connection_not_ready": ACTION_TEST,
}

# ── the connection-selectability ladder ─────────────────────────────────────
# Blocker codes in REMOVAL order: earlier entries must be cleared before a later
# one even becomes the answer. Two consequences the surfaces rely on, both
# asserted in the proofs:
#
#   * `connection_blocker` reports the FIRST match, which is the blocker in front
#     of the owner right now rather than the deepest one it can see;
#   * a HIGHER `blocker_rank` therefore means FEWER remaining steps, so a picker
#     choosing which connection to advise about picks the highest rank.
BLOCK_RETIRED = "connection_retired"
BLOCK_HOST_IDENTITY = "host_identity_changed"
BLOCK_NOT_BOOTSTRAPPED = "connection_not_bootstrapped"
BLOCK_EXECD_OUTDATED = "remote_execd_outdated"
BLOCK_DEGRADED = "connection_degraded"
BLOCK_HEALTH_STALE = "connection_health_stale"
BLOCK_NOT_READY = "connection_not_ready"

CONNECTION_BLOCKER_ORDER: tuple[str, ...] = (
    BLOCK_RETIRED,
    BLOCK_HOST_IDENTITY,
    BLOCK_NOT_BOOTSTRAPPED,
    BLOCK_EXECD_OUTDATED,
    BLOCK_DEGRADED,
    BLOCK_HEALTH_STALE,
    BLOCK_NOT_READY,
)

# The owner-facing sentence for each blocker: WHAT is in the way, then the ONE
# action, then — where an owner has already been misled — what will NOT clear it.
# These are the strings the picker and the CLI render verbatim; no surface
# composes its own, which is how "or Test to refresh health" came to be offered
# for a block Test cannot touch.
CONNECTION_BLOCKER_HINTS: dict[str, str] = {
    BLOCK_RETIRED: (
        "This connection is retired and cannot be un-retired — add an active "
        "connection in Settings → Connections, or move an existing project onto "
        "one with “ouroboros projects rebind”."
    ),
    BLOCK_HOST_IDENTITY: (
        "A different host is answering this SSH alias. Use “Retrust host…” in "
        "Settings → Connections to accept the new identity, then Bootstrap."
    ),
    BLOCK_NOT_BOOTSTRAPPED: (
        "This connection has no remote executor installed yet — run Bootstrap in "
        "Settings → Connections."
    ),
    BLOCK_EXECD_OUTDATED: (
        "The executor installed on this host was built against an older "
        "Home↔execd contract set and cannot run a task. Run Bootstrap in "
        "Settings → Connections — Test will report it healthy and change nothing, "
        "because only Bootstrap installs the matching executor."
    ),
    BLOCK_DEGRADED: (
        "The last probe reached this host and found it degraded — repair the host "
        "or the network, then run Test in Settings → Connections."
    ),
    BLOCK_HEALTH_STALE: (
        "This connection has not answered since Ouroboros started — run Test in "
        "Settings → Connections."
    ),
    # The rung this one REPLACED said "still connecting", and the proofs could not
    # bring a connection into it: `_record_runtime_health` never records `connecting`,
    # so a row with fresh health is `ready`, `degraded` or `unknown` and one that has
    # no fresh health is caught a rung earlier. An unreachable rung is a sentence
    # nobody can ever have seen, which is the vacuous-guard family wearing a hint.
    BLOCK_NOT_READY: (
        "The last probe did not report this host as ready — run Test in "
        "Settings → Connections."
    ),
}


def connection_blocker(row: Mapping[str, Any]) -> dict[str, Any]:
    """The ONE thing standing between this connection and a new remote project.

    Returns `{}` when nothing is — which is exactly the definition of selectable,
    so this function and the browser's `isSelectableRemoteConnection` are one
    predicate rather than two that agree by habit. Otherwise returns the blocker
    code, the action that removes it, the owner sentence, and the rank that says
    how deep in the ladder it sits.

    `row` is the PROJECTED connection: the store record, the live broker status,
    and Home's own admission evidence merged in that order
    (`gateway/connections.api_connections_list`). It is read as a mapping and
    nothing is inferred from absence except the conservative answer — a row with
    no evidence at all is blocked, never selectable, because the alternative is
    offering a placement admission would refuse after the owner picked a folder.
    """

    order = CONNECTION_BLOCKER_ORDER
    if str(row.get("lifecycle") or "active") != "active":
        code = BLOCK_RETIRED
    elif str(row.get("error_code") or "") in {
        "host_identity_changed",
        "host_identity_mismatch",
    }:
        code = BLOCK_HOST_IDENTITY
    elif row.get("bootstrap_compatible") is not True:
        code = BLOCK_NOT_BOOTSTRAPPED
    elif row.get("execd_outdated") is True:
        code = BLOCK_EXECD_OUTDATED
    elif str(row.get("status") or "unknown") in {"degraded", "disconnected"}:
        code = BLOCK_DEGRADED
    elif row.get("health_fresh") is not True:
        code = BLOCK_HEALTH_STALE
    elif str(row.get("status") or "unknown") != "ready":
        code = BLOCK_NOT_READY
    else:
        return {}
    return {
        "blocked_by": code,
        "blocker_action": REFUSAL_ACTIONS[code],
        "blocker_hint": CONNECTION_BLOCKER_HINTS[code],
        "blocker_rank": order.index(code),
    }


__all__ = [
    "ACTION_ADD_CONNECTION",
    "ACTION_AUTHENTICATE_OWNER",
    "ACTION_CONFIGURE_NETWORK_PASSWORD",
    "ACTION_CONFIRM_RETRUST",
    "ACTION_BOOTSTRAP",
    "ACTION_CANCEL_TASKS",
    "ACTION_CHOOSE_GIT_ROOT",
    "ACTION_CHOOSE_HOST",
    "ACTION_LIST_CONNECTIONS",
    "ACTION_NONE",
    "ACTION_READMIT_PROJECT",
    "ACTION_REBUILD_BUNDLE",
    "ACTION_RECONNECT",
    "ACTION_RELOAD_CONNECTIONS",
    "ACTION_RELOAD_PROJECTS",
    "ACTION_REPAIR_HOST",
    "ACTION_REPAIR_STORE",
    "ACTION_REPORT_DEFECT",
    "ACTION_RESTART",
    "ACTION_RETRUST",
    "ACTION_RETRY",
    "ACTION_RUN_FROM_TERMINAL",
    "ACTION_START_NEW_TASK",
    "ACTION_TEST",
    "ACTION_USE_ACTIVE_CONNECTION",
    "ACTION_WAIT_OR_CANCEL_TASKS",
    "BLOCK_DEGRADED",
    "BLOCK_EXECD_OUTDATED",
    "BLOCK_HEALTH_STALE",
    "BLOCK_HOST_IDENTITY",
    "BLOCK_NOT_BOOTSTRAPPED",
    "BLOCK_NOT_READY",
    "BLOCK_RETIRED",
    "CONNECTION_BLOCKER_HINTS",
    "CONNECTION_BLOCKER_ORDER",
    "REFUSAL_ACTIONS",
    "connection_blocker",
]

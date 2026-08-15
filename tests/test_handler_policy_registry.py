"""CLASS GATE 2 — a policy refusal inside a handler must say whether it travels.

The defect: a POLICY decision (a refusal that turns on a profile, a runtime mode, a path
class, a placement, a role — never on an IO error) is implemented inside one tool's Home
handler.  The tool later gains a second execution route, and the policy simply does not
run there.  The found instance was a subagent secret-file prohibition.

The audit that produced this file found the class is structural rather than incidental,
and that is the fact worth keeping:  for the 15 tools in ``REMOTE_NATIVE_TOOL_OPERATION``
the second route does not fork INSIDE the handler at all — ``ToolRegistry._dispatch``
REPLACES the handler (``tools/registry.py``, the ``if _prepared.native_routed:`` arm).  So
every policy refusal in those handler bodies is "after the fork" BY CONSTRUCTION, and
"unclassified" and "Home-only" are the same state for a policy, exactly as they once were
for a tool before ``HOME_ONLY_TOOL_NAMES`` was written.

This file is the missing declaration.  Every policy refusal reachable from a dual-route
tool's handler carries one of three verdicts:

* ``travels``            — the same policy is enforced on the other route; the entry names
                           WHERE, so the claim is checkable by a reader.
* ``home_only_vacuous``  — the policy FACT cannot hold on the other route; the entry says
                           why, because "it can't happen" is precisely the reasoning that
                           was wrong about ``execute_prepared``.
* ``escapes``            — a real gap, on the record, with its consequence in words.

``escapes`` is now EMPTY, and ``MAX_ESCAPING_POLICIES`` is 0.  Getting there took two
things, and the difference between them is the whole point of keeping this file:

1. A RECOUNT.  The audit that wrote this registry read the tree at ``539668f``, before the
   subagent-secret lift, the read/list export policy and the remote protected-artifact
   write block landed.  Four of its nine escapes were already closed by the time it was
   read, and two more were miscounted at the leaf: ``extract_video_frames``'s target kernel
   does say it applies no policy, but its CALLER applies both halves, and the target's
   process PREPARE has had two of the four scratch rules all along.  A registry that is
   not re-derived against the tree it claims to describe decays into a list of historical
   worries — which is why ``test_the_registry_describes_refusals_that_still_exist`` exists.
2. Six real closes, in one pass, each structural rather than per-instance: the
   ``run_script`` interpreter allowlist lifted into the placement-blind pipeline beside the
   secret denial; the ``integrate_subagent_patch`` category guard hoisted ABOVE the
   placement fork; the missing two scratch rules added to the target's prepare; the
   ``bytes_equal`` operands run through the bound export document on the target; the marker
   PATTERN half of the restricted-reader denial given a rule field so it can travel as
   data; and the alias (hardlink) probe performed on the target, which holds the filesystem
   the pipeline does not.

BOUNDARY of this gate, stated so its reach is not overestimated:

* Refusals are discovered by the codebase's own marker convention — a ``⚠️ LABEL`` string
  returned or raised from code reachable from the handler within its own module (call
  depth 3).  A refusal that carries no marker is INVISIBLE here.  ``UNLABELLED_KNOWN_GAPS``
  below is where such a case is recorded; both of the two it once held are now closed, and
  the dict keeps them with their closes named rather than being emptied, because "the list
  is empty" and "nobody looked" read identically.
* Reachability is within the handler's own module. A policy reached through another module
  is not followed, which is why the registry is keyed on the LABEL (the unit of policy)
  rather than on a line.
* The gate cannot verify that a ``travels`` claim is TRUE. It only forces the claim to be
  made and located. That is the same trade the platform gate makes, and it is what turns a
  silent omission into a reviewable sentence.  The behavioural proofs of the six closes
  above live in ``tests/test_subagent_placement_parity.py`` and
  ``tests/test_registry_remote_dispatch.py``, next to the routes they exercise.
"""

from __future__ import annotations

import ast
import pathlib
import re
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent

# The codebase's refusal marker. `⚠️ LABEL: message`.
_MARKER = re.compile(r"⚠️\s*([A-Z][A-Z0-9_]{5,})\b")

# A label is a POLICY refusal when it ends this way. IO errors, argument validation,
# resource limits and transport failures all end differently (`_ERROR`, `_TIMEOUT`,
# `_NOT_FOUND`, `_UNAVAILABLE`, `_FAILED`, `_CONFLICT`, `_MISSING`, …) and are excluded —
# they are not decisions about who may do what.
_POLICY_SUFFIXES = ("_BLOCKED", "_FORBIDDEN", "_REFUSED", "_DENIED")

# Policy refusals whose label breaks the suffix convention. Enumerated rather than guessed,
# because a naming convention is not a safety property: `INTEGRATE_SELF_WORKTREE_UNDER_
# WORKSPACE` is one of the live gaps this audit found and it would have been invisible to a
# suffix rule.
EXTRA_POLICY_LABELS: frozenset[str] = frozenset({
    "ROOM_WRITE_VIA_TASK",
    "INTEGRATE_SELF_WORKTREE_UNDER_WORKSPACE",
})

# Policy refusals that carry NO marker and are therefore outside the discovery above.
# Kept with their closes named rather than deleted: an empty dict and an unexamined gate
# read the same, and the next marker-less policy needs somewhere obvious to be written down.
UNLABELLED_KNOWN_GAPS: dict[str, str] = {
    "verify_and_record/bytes_equal_confinement": (
        "CLOSED. tools/verify.py::_bytes_equal_confinement_block refuses with a plain "
        "sentence, so the marker scan cannot see it; it blocks a protected-artifact "
        "`read_bytes` on the Home route because bytes_equal is a byte-read oracle (sizes "
        "plus a hexdump around the first divergence). The target route "
        "(workspace_payload_native.attach_remote_verification_facts) enforced workspace "
        "containment and NOT that check, so a remote bytes_equal could hexdump a black-box "
        "reference binary the identical Home call refuses. It now runs both declared "
        "operands through the operation's own bound export document, whose protected_paths "
        "are the resource policy's protected artifacts in target spellings — the same "
        "authority, the same evaluator, applied at the source."
    ),
    "query_code+search_code/silent_secret_filter": (
        "CLOSED in two steps. tools/query_code.py::_visible_file and tools/core.py's walk "
        "filters DROP secret paths instead of refusing, so there is no marker. The target "
        "filters through the export policy, which used to be profile-BLIND: the rule that "
        "ran was 'nobody sees secrets', not 'subagents do not'. e976cfe made the document "
        "profile-aware (remote_export_policy.restricted_reader_rules adds the secret "
        "components and the owner-control filenames for a restricted reader). What that "
        "left behind was the PATTERN half — `db_password.conf`, `api_key.yaml`, `x.env` are "
        "not enumerable as tokens — which now travels as the `marker_scoped_suffixes` rule "
        "field, matched delimited and suffix-scoped so the predicate is the same one "
        "tools/core._is_subagent_secret_repo_path applies on Home. NAMED RESIDUE: the "
        "`.env.` INFIX form (`staging.env.old`). The document's rule fields are prefix, "
        "suffix, exact-name, component and marker shaped and an infix is none of them, so "
        "closing it would mean inventing a field for one spelling — a rule no reader could "
        "predict. Both ANCHORED forms are covered (`.env*` by prefix, `*.env` by suffix), "
        "and the reach of the residue is a restricted subagent's WALK over a file whose "
        "name carries `.env.` in the middle AND whose trailing extension is not "
        "config-shaped; the named-path doors refuse it on both placements."
    ),
}

# (tool, label) -> (verdict, reason). The single registry the audit found missing.
POLICY_ROUTE_VERDICTS: dict[tuple[str, str], tuple[str, str]] = {
    # ── read side ────────────────────────────────────────────────────────────────
    ("read_file", "REPO_READ_BLOCKED"): (
        "travels",
        "the subagent secret/control-file prohibition — the founding instance, and closed "
        "before this registry was merged. The DECISION was lifted out of the handler body "
        "into tools/dispatch_policy.subagent_secret_path_refusal, which runs BEFORE prepare "
        "over (profile, tool, root, spelling) and therefore cannot be skipped by any "
        "placement; the same spelling predicate additionally tightens the operation's export "
        "document so the target declines the bytes at the source. The ALIAS half (a hardlink "
        "to `.env` under a clean name) is not a spelling and is now probed on the target too, "
        "by export_policy_contract.judged_exclusion, which workspace_native._read_file reaches through the read door (requested spelling, resolved identity, recursive hardlink alias).",
    ),
    ("read_file", "DATA_READ_BLOCKED"): (
        "home_only_vacuous",
        "guards root=runtime_data, which is Home-native under every placement, so this "
        "handler arm is never the one a native route replaces.",
    ),
    ("read_file", "TOOL_ACCESS_BLOCKED"): (
        "home_only_vacuous",
        "the profile x root access matrix. Only active_workspace is target-native, and every "
        "profile that can call read_file at all holds the active_workspace read op — so the "
        "escape is empty. This used to say 'a property of the CURRENT matrix, not a stated "
        "invariant', which is precisely the reasoning shape this file exists to distrust: it "
        "is now DERIVED from _POLICY on every run by test_the_access_matrix_vacuity_is_"
        "CHECKED_and_not_merely_claimed, which also credits the pipeline's tool-NAME "
        "allowlists (they refuse before prepare, so they travel) and forces a new rootless "
        "profile to prove it cannot be a remote task.",
    ),
    ("list_files", "REPO_LIST_BLOCKED"): (
        "travels",
        "the listing half of the same prohibition, closed by the same lift plus two more "
        "pieces: the target's _list_files applies the bound document per entry and emits the "
        "LIST_POLICY_FILTERED count (89a8545), and tools/dispatch_policy.filter_native_listing "
        "runs Home's own redaction over the returned array so both placements produce the "
        "same bytes. The ROOM-LENS branch of tools/core._list_files used to return before "
        "reaching either half and now applies both — that one was a divergence between two "
        "LOCAL calls, which is the same class wearing a different hat.",
    ),
    ("list_files", "DATA_LIST_BLOCKED"): (
        "home_only_vacuous",
        "guards root=runtime_data, which is Home-native under every placement, so the native "
        "route never replaces this arm — the same reasoning as read_file's entry above.",
    ),
    ("list_files", "TOOL_ACCESS_BLOCKED"): (
        "home_only_vacuous",
        "the same profile x root access matrix as read_file, vacuous for the same reason: only "
        "active_workspace routes natively, and every profile reaching list_files holds it.",
    ),
    ("search_code", "TOOL_ACCESS_BLOCKED"): (
        "home_only_vacuous",
        "same access matrix; search_code's secret handling is a silent FILTER rather than a "
        "refusal and is therefore recorded in UNLABELLED_KNOWN_GAPS instead.",
    ),
    ("query_code", "TOOL_ACCESS_BLOCKED"): (
        "home_only_vacuous",
        "same access matrix; the profile-scoped root refusals it also carries concern "
        "system_repo and user_files, which are Home-native and never routed.",
    ),
    # ── write side ───────────────────────────────────────────────────────────────
    ("write_file", "WRITE_FILE_BLOCKED"): (
        "travels",
        "the protected-artifact write block, closed by 591f3fb. Home cannot spell this check "
        "for a target path — it has no Home path to ask `protected_artifacts` about — so the "
        "SOURCE applies it, from the protected_paths the export document already carried for "
        "exactly this purpose (the `mutation` question inside native_mutation_target, typed "
        "REMOTE_PROTECTED_ARTIFACT_BLOCKED, whole batch judged before the first byte). The "
        "pipeline's own protected-write arm is a DIFFERENT authority (runtime-mode protected "
        "paths) and is disabled for workspace tasks on BOTH placements — see the separate "
        "PIPELINE_ONLY_AUTHORITIES note below.",
    ),
    ("write_file", "DATA_WRITE_BLOCKED"): (
        "home_only_vacuous",
        "root=runtime_data and the control-plane paths under it (connection store, skill "
        "owner state, settings.json) are Home-native and never routed to a target.",
    ),
    ("write_file", "WRITE_BLOCKED"): (
        "home_only_vacuous",
        "the runtime_data project-store arm. That root is Home-native under every placement, so "
        "the native route cannot be the one that skips this refusal.",
    ),
    ("write_file", "SKILL_REDIRECT_BLOCKED"): (
        "home_only_vacuous",
        "root=skill_payload is Home-native: reviewed payloads live on Home by contract, so "
        "the redirect rule has no target-side counterpart to need.",
    ),
    ("write_file", "TOOL_ACCESS_BLOCKED"): (
        "home_only_vacuous",
        "the same profile x root access matrix as the read side, vacuous for the same reason: "
        "active_workspace is the only routed root and every calling profile holds its ops.",
    ),
    ("write_file", "ROOM_WRITE_VIA_TASK"): (
        "home_only_vacuous",
        "a project folder-room lens is a Home presentation concept; no lens exists for a "
        "target workspace, so the condition cannot hold on the other route.",
    ),
    **{
        (tool, "ROOM_WRITE_VIA_TASK"): (
            "home_only_vacuous",
            "the same folder-room lens condition as write_file, in the multi-file editors: "
            "a lens is a Home presentation concept and no lens exists for a target "
            "workspace, so the condition cannot hold on the native route.",
        )
        for tool in ("apply_patch", "edit_batch")
    },
    ("edit_text", "EDIT_TEXT_BLOCKED"): (
        "travels",
        "the protected-artifact block for in-place edits, closed by the same source-side "
        "applier as write_file: the target's _edit_text asks the `mutation` question at both doors "
        "against the same bound document before it touches the file, so a remote edit of a "
        "black-box artifact is refused where the bytes are rather than not at all.",
    ),
    ("edit_text", "WRITE_BLOCKED"): (
        "home_only_vacuous",
        "the runtime_data project-store arm, on a root that is Home-native under every "
        "placement; a native route never reaches this branch at all.",
    ),
    ("edit_text", "TOOL_ACCESS_BLOCKED"): (
        "home_only_vacuous",
        "the same profile x root access matrix as the read side; vacuous because a native route "
        "only ever resolves active_workspace, an op every calling profile already holds.",
    ),
    ("edit_text", "ROOM_WRITE_VIA_TASK"): (
        "home_only_vacuous",
        "a project folder-room lens is a Home presentation concept with no target-workspace "
        "counterpart, so the condition this refusal tests cannot hold on the other route.",
    ),
    # ── shell side ───────────────────────────────────────────────────────────────
    ("run_command", "SCRATCH_BLOCKED"): (
        "travels",
        "the declared-scratch safety rule: a throwaway must sit in a git worktree, be "
        "confined to the command cwd, be git-untracked, and not be a directory — four rules, "
        "because scratch is EXCLUDED from the workspace patch and each of them is a way real "
        "work could be kept out of the deliverable. The audit recorded 'the target has no "
        "scratch concept at all'; in fact the target's prepare had the worktree probe and the "
        "git-tracked check from the start, and the recount found the two that were missing "
        "(cwd confinement — it confined to the workspace ROOT — and the directory refusal). "
        "Both are now in workspace_native.prepare_native_operation, so all four run on the "
        "target and a refused call never starts a process.",
    ),
    ("run_command", "SHELL_CWD_BLOCKED"): (
        "travels",
        "cwd resolution is a PREPARE fact, not a handler decision: dispatch_prepare resolves "
        "the cwd on the host that will run the command and projects a refusal as cwd_error, "
        "which the same guard reports on both routes.",
    ),
    ("run_script", "RUN_SCRIPT_BLOCKED"): (
        "travels",
        "the interpreter allowlist, and this one was real: workspace_payload_native."
        "execute_inline_script takes the interpreter VERBATIM into its argv, so an arbitrary "
        "executable name reached the target on a route where the check never ran. The rule is "
        "now tools/dispatch_policy.script_interpreter_refusal, judged on the RAW argument "
        "before prepare — placement-blind by construction, and refused before any target RPC. "
        "The handler keeps its own copy of the check for non-dispatch callers, over the ONE "
        "shared allowlist constant so the two cannot drift.",
    ),
    ("run_script", "SCRATCH_BLOCKED"): (
        "travels",
        "the same declared-scratch rule as run_command and closed by the same four checks in "
        "the target's prepare; run_script reaches them because its operation is in the same "
        "process branch of workspace_native.prepare_native_operation.",
    ),
    ("run_script", "SHELL_CWD_BLOCKED"): (
        "travels",
        "the same prepare-resolved cwd fact as run_command: the target canonicalizes the cwd and "
        "the refusal is projected as cwd_error for the one guard that owns it.",
    ),
    # ── media ────────────────────────────────────────────────────────────────────
    ("extract_video_frames", "PATH_BLOCKED"): (
        "travels",
        "the allowed-file-roots and protected-artifact read guard. The audit read the LEAF and "
        "believed the claim it found there: workspace_media_native.extract_video_frames does "
        "say it performs no confinement and no policy — because its CALLER performs both. "
        "workspace_native.prepare_native_operation relativizes the path and calls "
        "native_target(question=QUESTION_EXPORT, channel='media_frames') against the bound document (protected "
        "artifacts included) before ffmpeg is resolved, and the dispatch resolves the source "
        "through _target(root, …, must_exist=True), which refuses a path escaping the "
        "workspace through a symlink. Home's root set and the target's workspace root are the "
        "two spellings of the same confinement, each for the host that owns the bytes.",
    ),
    # ── the three tools that fork inside their own handler ───────────────────────
    ("claude_code_edit", "CLAUDE_CODE_EDIT_BLOCKED"): (
        "travels",
        "this refusal IS the docker route's answer (it fires when the executor is "
        "docker_exec and covers the cwd), so it is placed on the branch it judges.",
    ),
    ("claude_code_edit", "CLAUDE_CODE_REMOTE_REFUSED"): (
        "travels",
        "the remote branch's own typed refusal, raised inside _remote_claude_code_edit, so it "
        "is that route answering for itself rather than a Home rule that failed to travel.",
    ),
    ("claude_code_edit", "CORE_PROTECTION_BLOCKED"): (
        "home_only_vacuous",
        "conditioned on system_repo_mode, which cannot hold for a task whose workspace is on "
        "another host: the Ouroboros system repo is Home's own tree.",
    ),
    ("claude_code_edit", "SKILL_PAYLOAD_CONTROL_BLOCKED"): (
        "travels",
        "mirrored by argument narrowing: the remote branch refuses bucket/skill_name/outputs "
        "up front in tools/shell.py, so a skill-payload edit cannot reach the target at all.",
    ),
    ("claude_code_edit", "SKILL_REDIRECT_BLOCKED"): (
        "travels",
        "the same mirror: the remote branch refuses skill_name/bucket arguments before touching "
        "the target, so the redirect rule has nothing left to catch on that route.",
    ),
    ("claude_code_edit", "ROOM_WRITE_VIA_TASK"): (
        "travels",
        "_room_default_cwd_edit_block runs BEFORE the placement fork, so both routes get it. "
        "This is the positive example of the pattern the rest of this table is about.",
    ),
    ("integrate_subagent_patch", "INTEGRATE_LINEAGE_FORBIDDEN"): (
        "travels",
        "lineage is checked above the fork; the handler's comment says everything before the "
        "fork is placement-independent, and for this check that is accurate.",
    ),
    ("integrate_subagent_patch", "INTEGRATE_GENESIS_FORBIDDEN"): (
        "travels",
        "also evaluated above the placement fork, together with the lineage checks, so the "
        "remote branch inherits it rather than skipping it.",
    ),
    ("integrate_subagent_patch", "INTEGRATE_TARGET_FORBIDDEN"): (
        "travels",
        "both branches refuse a requested target they do not own; the remote branch checks it "
        "explicitly before touching the mirror.",
    ),
    ("integrate_subagent_patch", "INTEGRATE_REMOTE_REFUSED"): (
        "travels",
        "raised by the remote branch itself inside _integrate_remote_subagent_patch, so it is "
        "the route's own answer rather than a Home rule that failed to travel.",
    ),
    ("integrate_subagent_patch", "INTEGRATE_SELF_WORKTREE_UNDER_WORKSPACE"): (
        "travels",
        "the category guard that stops a self_worktree child's patch (an Ouroboros system-repo "
        "change) from being integrated into a project workspace. It sat BELOW the placement "
        "fork, and the fork's comment claimed only that everything ABOVE it was "
        "placement-independent — a sentence that says nothing about what is below. A remote "
        "task is always workspace mode, so its condition held there by construction and it was "
        "exactly the case being skipped. The guard is now HOISTED above the fork: it is pure "
        "over (child surface, workspace mode, the parent's own surface) and needs no Home path, "
        "so both branches reach it and the local route is unchanged.",
    ),
    ("verify_and_record", "VERIFY_CWD_BLOCKED"): (
        "travels",
        "the cwd class check runs before the route decision, and the remote branch sends the "
        "cwd for the target to canonicalize; the byte-read oracle's confinement is a separate, "
        "marker-less refusal recorded in UNLABELLED_KNOWN_GAPS.",
    ),
    # ── the browser pair: a placement fork that moves no EXECUTION ───────────────
    # These two fork on placement (`_is_remote_placement`) and so are discovered as
    # dual-route, but what forks is the URL, not the executor: the Playwright browser
    # runs on Home under BOTH placements and a remote loopback URL is merely rewritten
    # onto this task's `ssh -L` forward. There is therefore no target-side handler for a
    # policy to fail to reach — which is exactly why each row below can name the one
    # handler as the place the rule runs.
    ("browse_page", "BROWSER_LOCAL_READONLY_BLOCKED"): (
        "travels",
        "the restricted-subagent URL prohibition. It is evaluated on the REQUESTED url before "
        "_resolve_placement_url rewrites anything, and again on the page's SETTLED url after "
        "the goto on every branch (including the infrastructure-retry branch), so a redirect "
        "cannot land the subagent somewhere the first check would have refused. The remote "
        "fork only remaps a loopback origin; the browser itself never leaves Home.",
    ),
    ("browser_action", "BROWSER_LOCAL_READONLY_BLOCKED"): (
        "travels",
        "the same prohibition on the acting side, plus the evaluate ban. It runs inside "
        "_do_action against the live page url on both placements, and the target additionally "
        "cannot reach a foreign origin at all: _remote_foreign_origin_blocked is registered as "
        "a request route for the bridged page, so the remote branch is strictly narrower than "
        "the Home one rather than a route that skipped the rule.",
    ),
    ("browser_action", "CONTEXT_MODE_SELF_LOWERING_BLOCKED"): (
        "travels",
        "an inspection of the JavaScript TEXT, above page.evaluate and above any placement "
        "fork in the handler, so the same predicate judges the same argument on both routes. "
        "The remote branch adds no second evaluate door for it to miss.",
    ),
    ("browser_action", "SCOPE_REVIEW_FLOOR_SELF_LOWERING_BLOCKED"): (
        "travels",
        "the same JavaScript-text inspection, in the same block above page.evaluate, so the "
        "remote branch reaches it on the identical argument before the browser is asked to "
        "run anything.",
    ),
    ("browser_action", "SAFETY_MODE_SELF_LOWERING_BLOCKED"): (
        "travels",
        "likewise a predicate over the JavaScript text evaluated in the handler before "
        "page.evaluate; placement does not select which of these six checks run, so the "
        "remote branch inherits all of them.",
    ),
    ("browser_action", "ELEVATION_BLOCKED"): (
        "travels",
        "the mutative-subagent and post-task-evolution toggles, both refused by text "
        "inspection in the same pre-evaluate block, above the placement fork. The "
        "owner-controlled setting they protect lives on Home, and the browser that would POST "
        "it runs on Home on the remote branch too.",
    ),
    ("browser_action", "OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED"): (
        "travels",
        "the self-attestation ban, also a pre-evaluate text check. Its target is an Ouroboros "
        "API port, and on the remote branch the bridged page's request route blocks every "
        "origin that is not this task's own forward — so the remote route is the narrower of "
        "the two rather than an uncovered one.",
    ),
    # Rebase onto v6.100 — four labels upstream introduced inside dual-route handlers.
    # Both classes are vacuous remotely for the SAME structural reason, and it is a
    # reason the placement model states rather than one this file asserts: the roots
    # they judge are in `workspace_ref.HOME_NATIVE_ROOTS`, so a call naming them is
    # Home-routed BY CONSTRUCTION and the native route never reaches the guard.
    **{
        (tool, "USER_FILES_PATH_BLOCKED"): (
            "home_only_vacuous",
            "the user_files containment guard (upstream's typed UserFilesPathBlockedError). "
            "`user_files` is a HOME_NATIVE_ROOT: it names the owner's own machine, so a "
            "remote task's read of it is answered by the ordinary Home handler and never "
            "crosses the wire. There is no target-side path for this fact to hold on.",
        )
        for tool in ("read_file", "list_files", "search_code")
    },
    ("edit_text", "STR_REPLACE_BLOCKED"): (
        "home_only_vacuous",
        "the skill control-plane write ban (provenance, launcher seed, marketplace and "
        "dependency markers). It is judged against `binding.state_drive_root` and the "
        "skill_payload root, both HOME_NATIVE_ROOTS — the target has no skill control "
        "plane to protect, and an edit naming those roots is Home-routed by construction.",
    ),
}

# Authorities that live in the PIPELINE rather than in a handler, and are therefore not
# discovered above at all — recorded here because one of them was mistaken for a handler
# gap. `protected_paths_in` (runtime_mode_policy) names OUROBOROS' OWN repo files:
# safety-critical, frozen-contract and release-invariant paths of the system repo. The
# pipeline's arm over it is disabled for workspace tasks (`disable_protected = workspace_mode
# and not acting_self_worktree`), which is a property of workspace tasks and NOT of remote
# ones — a local external-workspace task is in exactly the same position. So it is not a
# placement divergence, and the remote case is additionally out of its reach by the root
# matrix: `system_repo` is Home-native under every placement, so a target workspace can never
# BE the tree those paths name.
PIPELINE_ONLY_AUTHORITIES: dict[str, str] = {
    "runtime_mode_policy.protected_paths_in": (
        "the runtime-mode protected core/contract/release list, over the Ouroboros system "
        "repo's own paths. Enforced in registry._dispatch for write_file/edit_text and "
        "disabled for every WORKSPACE task, local and remote alike — so the two placements "
        "agree, and `system_repo` is not a target-native root for it to reach anyway. The "
        "task-contract protected_artifacts authority is the one that governs a workspace "
        "path, and that one does travel (see the write_file / edit_text rows above)."
    ),
}

# The ceiling. A fix may lower it; nothing may raise it without an owner deciding to. It is
# ZERO: every row above is `travels` or `home_only_vacuous`, and a new `escapes` is now a
# gate failure rather than a line item.
MAX_ESCAPING_POLICIES = 0


def _dual_route_handlers() -> dict[str, tuple[str, str]]:
    """tool -> (module, handler function name), for tools with a second route.

    Derived, not listed: a tool is dual-route when the dispatcher can replace its handler
    (it is in ``REMOTE_NATIVE_TOOL_OPERATION``) or when its own reachable body forks on
    placement. Hand-listing would go stale the first time a tool gained a route.
    """

    from ouroboros.tool_capabilities import (
        HYBRID_AFFINITY_TOOL_NAMES,
        REMOTE_NATIVE_TOOL_OPERATION,
    )
    from ouroboros.tools.registry import ToolRegistry

    scratch = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=scratch, drive_root=scratch / "drive")
    out: dict[str, tuple[str, str]] = {}
    for name in sorted(set(REMOTE_NATIVE_TOOL_OPERATION) | set(HYBRID_AFFINITY_TOOL_NAMES)):
        entry = registry._entries.get(name)
        if entry is None:
            continue
        handler = entry.handler
        module = getattr(handler, "__module__", "")
        qualname = getattr(handler, "__qualname__", "").split(".")[0]
        if module and qualname:
            out[name] = (module, qualname)
    return out


def _module_functions(module: str):
    path = REPO.joinpath(*module.split(".")).with_suffix(".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _reachable(node, functions, seen: set[str], depth: int = 0):
    if depth > 3 or node.name in seen:
        return
    seen.add(node.name)
    yield node
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in functions:
                yield from _reachable(functions[name], functions, seen, depth + 1)


def _is_policy_label(label: str) -> bool:
    return label.endswith(_POLICY_SUFFIXES) or label in EXTRA_POLICY_LABELS


def discovered_policy_refusals() -> dict[str, set[str]]:
    """tool -> {policy label} for every dual-route tool's reachable handler code."""

    found: dict[str, set[str]] = {}
    from ouroboros.tool_capabilities import REMOTE_NATIVE_TOOL_OPERATION

    cache: dict[str, dict] = {}
    for tool, (module, handler_name) in _dual_route_handlers().items():
        if module not in cache:
            cache[module] = _module_functions(module)
        functions = cache[module]
        node = functions.get(handler_name)
        if node is None:
            continue
        nodes = list(_reachable(node, functions, set()))
        body = "\n".join(ast.unparse(item) for item in nodes)
        forks = "is_remote_workspace" in body or "_remote_" in body
        if tool not in REMOTE_NATIVE_TOOL_OPERATION and not forks:
            continue
        labels: set[str] = set()
        for item in nodes:
            for sub in ast.walk(item):
                if not isinstance(sub, (ast.Return, ast.Raise)):
                    continue
                for const in ast.walk(sub):
                    if isinstance(const, ast.Constant) and isinstance(const.value, str):
                        labels.update(
                            label
                            for label in _MARKER.findall(const.value)
                            if _is_policy_label(label)
                        )
        if labels:
            found[tool] = labels
    return found


def test_every_dual_route_policy_refusal_is_classified():
    """A policy refusal in a dual-route handler declares whether it survives the route.

    The fix for a failure here is never to weaken the rule: either the policy moves into
    the shared pipeline (or its target-side twin), or it gets a row in
    POLICY_ROUTE_VERDICTS saying which of the three things it is and why.
    """

    unclassified: list[str] = []
    for tool, labels in sorted(discovered_policy_refusals().items()):
        for label in sorted(labels):
            if (tool, label) not in POLICY_ROUTE_VERDICTS:
                unclassified.append(f"{tool}: {label}")
    assert not unclassified, (
        "policy refusal in a handler with a second execution route, with no declared "
        "verdict — 'unclassified' and 'Home-only' are the same state until someone "
        "writes it down:\n" + "\n".join(unclassified)
    )


def test_the_declared_verdicts_are_well_formed():
    """Each verdict is one of the three, and each reason actually explains itself."""

    for (tool, label), (verdict, reason) in POLICY_ROUTE_VERDICTS.items():
        assert verdict in {"travels", "home_only_vacuous", "escapes"}, (
            f"{tool}/{label}: unknown verdict {verdict!r}"
        )
        assert len(reason) > 80, (
            f"{tool}/{label}: a one-word reason is how this class hid the first time — "
            f"say where the policy runs, or why the fact cannot hold. Got {reason!r}"
        )
        if verdict == "travels":
            assert any(
                token in reason
                for token in ("pipeline", "prepare", "target", "branch", "fork", "mirror")
            ), f"{tool}/{label}: a 'travels' claim must name WHERE it travels to"


def test_the_escaping_set_does_not_grow():
    """The live gaps are capped, so the class cannot spread while nobody is looking.

    A ceiling rather than an exact pin: closing one of these must never break the gate.
    """

    escaping = sorted(
        f"{tool}/{label}"
        for (tool, label), (verdict, _reason) in POLICY_ROUTE_VERDICTS.items()
        if verdict == "escapes"
    )
    assert len(escaping) <= MAX_ESCAPING_POLICIES, (
        f"{len(escaping)} policies now escape their second route, over the "
        f"{MAX_ESCAPING_POLICIES} this audit recorded. A new one is a decision, not an "
        "accident — lower the ceiling as they are fixed, never raise it:\n"
        + "\n".join(escaping)
    )


def test_the_registry_describes_refusals_that_still_exist():
    """Stale rows are tolerated; a registry that has WHOLLY rotted is not.

    Tolerated because a fix (moving a policy into the pipeline) removes the label from the
    handler, and a gate that punishes fixes gets deleted. But if almost nothing in the
    table still matches the code, the table has stopped describing this program.
    """

    discovered = discovered_policy_refusals()
    live = sum(
        1
        for (tool, label) in POLICY_ROUTE_VERDICTS
        if label in discovered.get(tool, set())
    )
    assert live >= len(POLICY_ROUTE_VERDICTS) // 2, (
        f"only {live}/{len(POLICY_ROUTE_VERDICTS)} declared rows still match a real "
        "refusal — re-derive this registry before trusting it"
    )


def test_the_marker_less_gaps_stay_named():
    """The gate's own blind spot is written down and non-empty.

    Both entries are now CLOSED, and they stay in the dict with their closes named: an
    empty list and an unexamined gate read identically, and the next marker-less policy
    needs an obvious place to be written down.
    """

    assert UNLABELLED_KNOWN_GAPS, "the marker-less gaps were removed without replacement"
    for key, reason in UNLABELLED_KNOWN_GAPS.items():
        assert "/" in key, f"{key}: name it as tool/policy"
        assert len(reason) > 120, f"{key}: the consequence must be in words, got {reason!r}"
    assert "BOUNDARY" in (__doc__ or ""), "this file must keep stating its own limits"
    for name, reason in PIPELINE_ONLY_AUTHORITIES.items():
        assert "." in name, f"{name}: name the module and the symbol"
        assert len(reason) > 120, f"{name}: say why it is not a placement divergence"


def test_the_access_matrix_vacuity_is_CHECKED_and_not_merely_claimed():
    """The six `TOOL_ACCESS_BLOCKED` rows rest on a fact, so the fact is derived.

    Those rows say the profile x root access matrix cannot escape because
    `active_workspace` is the only target-native root and every profile that can call the
    tool at all already holds that root's operation. The registry's own note admits this
    is "a property of the CURRENT matrix, not a stated invariant" — which is exactly the
    shape of reasoning that was wrong elsewhere in this file. So it is re-derived from the
    matrix on every run: a profile added with `read_file` but without `active_workspace`
    read would make six `home_only_vacuous` verdicts false, and would fail HERE rather
    than being discovered on a remote task.
    """

    from ouroboros.tool_access import _POLICY
    from ouroboros.workspace_ref import SSH_NATIVE_ROOTS

    assert sorted(SSH_NATIVE_ROOTS) == ["active_workspace"], (
        "a second target-native root appeared in the ratified matrix; every "
        f"TOOL_ACCESS_BLOCKED verdict here was written for exactly one: {sorted(SSH_NATIVE_ROOTS)}"
    )
    tools = sorted({tool for (tool, label) in POLICY_ROUTE_VERDICTS if label == "TOOL_ACCESS_BLOCKED"})
    assert tools, "no TOOL_ACCESS_BLOCKED rows left — re-derive this test with them"
    # The op each routed tool asks the matrix for, from the handlers' own `_access_or_block`
    # calls. A routed tool missing from this map is an unclassified one, so the map is
    # asserted complete against the registry rather than merely consulted.
    operations = {
        "read_file": "read", "list_files": "list", "search_code": "search",
        "query_code": "search", "write_file": "write", "edit_text": "edit",
    }
    assert set(tools) <= set(operations), sorted(set(tools) - set(operations))
    # The PIPELINE's own tool-name allowlists (registry._early_dispatch_block, which runs
    # before prepare and is therefore placement-blind). A profile may legitimately lack an
    # op if it cannot call the tool at all — that refusal travels, this one would not.
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )

    name_allowlists = {
        "local_readonly_subagent": LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
        "acting_subagent": ACTING_SUBAGENT_TOOL_NAMES,
    }
    escapes: list[str] = []
    for profile, matrix in _POLICY.items():
        reachable = name_allowlists.get(profile)
        allowed = {str(item) for item in matrix.get("active_workspace") or ()}
        if not allowed:
            # A profile with NO reach into the routed root cannot reach a routed CALL
            # either: the native route only ever resolves `active_workspace`, and a task
            # whose workspace is on a target IS an active_workspace task. `skill_repair`
            # is the one such profile, and its workspace is a Home skill payload by
            # contract — asserted below so a NEW rootless profile is not silently assumed
            # to be in the same position.
            assert profile == "skill_repair", (
                f"{profile!r} holds no active_workspace op, so its TOOL_ACCESS_BLOCKED "
                "refusal is the ONLY thing between it and the native route — which "
                "replaces the handler that raises it. Prove it cannot be a remote task, "
                "or the six vacuity verdicts in this file are wrong for it."
            )
            continue
        for tool in tools:
            needed = operations[tool]
            if needed in allowed or (reachable is not None and tool not in reachable):
                continue
            escapes.append(f"{profile} reaches {tool} but lacks active_workspace {needed!r}")
    assert not escapes, (
        "a profile can reach a routed tool without holding the target-native root's "
        "operation, so the access-matrix refusal is NOT vacuous on the native route and "
        "its verdict must change:\n" + "\n".join(escapes)
    )

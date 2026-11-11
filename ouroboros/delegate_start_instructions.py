"""Stable host instructions and complete actor-first coordination appendix."""

from __future__ import annotations

import os
from hashlib import sha256
from typing import Any


HOST_INSTRUCTIONS = (
    "You are a delegated worker running inside the workspace assigned by your host. The words of "
    "your host's human carried with your assignment state what was asked; your assignment divides "
    "and scopes your part within them and never overrides what they ask or forbid. Explicit limits "
    "in the assignment still bind. Later words of that human may reach you during a turn as a host "
    "message whose first line begins \"[Owner words quoted verbatim\"; if the engine defers them, "
    "the same rows appear under the heading MY HUMAN'S WORDS FOR THIS WORK on your next turn. They "
    "are that human's exact words. Do not run git "
    "commit, tag, push, rebase, reset or any other history-moving command: your host "
    "captures changes against its recorded baseline and decides whether to integrate "
    "them. A private delegated snapshot can preserve committed changes in that diff, "
    "but a moved HEAD is disclosed as an instruction violation; it does not authorize "
    "a commit or apply. A self_worktree capture separately requires an unchanged HEAD. "
    "Do not review or accept your own change, do not "
    "touch the host's runtime controls, or modify its memory or skills; unless your assignment "
    "restricts reads, you may read the memory files your session references name. If your environment "
    "offers a way to ask your host a clarifying "
    "question, you may use it: your host may answer from its task context; a question "
    "that carries an engine expiry times out benignly if unanswered — continue with "
    "stated assumptions rather than blocking — while one without an expiry waits until "
    "answered. If your harness cannot ask mid-run, do NOT end the run to ask — "
    "state your assumption and continue."
)

# Owner 3B: the ONE host-facts sentence for a session that runs with a scoped native-state
# home (delegated runs, reviewer agent sessions); never a raw API prompt, which has no shell
# HOME. It names the two facts the engine provides WHEN PRESENT and lists no commands.
HOST_FACTS_INSTRUCTION = (
    " Your HOME may be a scoped native-state home your engine selected rather than the "
    "operator's; when present, CLAUDEXOR_OPERATOR_HOME names the operator's home and "
    "PLAYWRIGHT_BROWSERS_PATH the operator's existing browser cache."
)

# Shared by working and reviewer sessions; actual agreements stay in the assignment.
SHARED_ENVIRONMENT_INSTRUCTION = (
    "Your workspace may be private while the machine and its resources may be shared. "
    "Follow applicable sharing agreements and choose effective execution with the least "
    "unnecessary interference, preserving the conditions the requested verification needs; "
    "a hidden, headless or isolated stand-in proves only what it preserves. Pass relevant "
    "conditions to your delegates before they start. Raise material unresolved conflicts "
    "through available host dialogue; without an answer, continue unblocked work with stated "
    "assumptions and explicit verification gaps, without assuming the conflict is settled."
)

UNPROVEN_BOUNDARY_INSTRUCTION = (
    " An OS-enforced filesystem boundary was REQUESTED for this run but is NOT guaranteed: "
    "your engine applies one only where it has a mechanism for this host, and your host "
    "reads back from your own attempt records what was actually applied. Work as if there "
    "is no boundary — keep your writes inside this root apart from disposable process state "
    "outside every Git working tree and the shared download cache your host names, do not read the "
    "operator's home directory, credential stores, or the harness runtime tree beyond the paths your "
    "host names, and do NOT describe "
    "yourself in your answer as sandboxed or confined. If your own environment shows you whether "
    "a boundary was in force, say so plainly."
)

# The one name of the folder whose files the host retains as named evidence before it
# releases a copy (`delegate_named_evidence.retain_named_evidence` reads it from here); a session
# is told it, never asked to guess.
EVIDENCE_DIRNAME = ".review-evidence"
# Disposable by purpose, never by name, kind or size (#2642): generated build/dependency output is not
# automatically excluded from a Git copy's capture, and named evidence applies no cache, binary or size filter,
# so a tool's default location can turn this output into a durable payload.
CHECK_BYPRODUCTS = ("build or dependency output that only serves your checks (environments, dependency "
                    "installs, compiled output, tool caches)")
# What the host declares about the scratch it made for one run, before the run uses it.
# Purpose and ownership, never a promise that the contents come back byte for byte.
DISPOSABLE_SCRATCH = (
    "disposable by the host's declaration: never captured; it follows this task, or a reviewer's checking "
    "copy through continuation, until release, unless the producer or owner retains it (the OS "
    "may clear it sooner), so anything left only there may be lost: keep edits in your working tree and "
    "return the results you need; nothing installed there is reinstalled for a later session, which "
    "builds its own from the project's files")


def scratch_fact(*, launcher: bool = False, granted: str = "") -> str:
    """Where disposable process state may go on this host: safe_test's own default parent,
    judged by the launcher's own predicate, so the instruction and its guard cannot disagree.

    ``launcher`` names safe_test, for a copy of the body, whose checkout carries it; any
    other project hears the same place with no command it lacks. The session's own
    temporary directory may still differ. ``granted`` is the exact scratch the host made
    for the session (``workspace_preflight.session_service_directories``), granted by its
    engine or, at full access, merely named: then it is the one place named."""
    if granted:
        return f"{'pass --temp-parent ' if launcher else 'in '}{granted}, this run's disposable scratch"
    from pathlib import Path

    from ouroboros.test_environment import default_temp_parent, git_ancestor

    try:
        parent = Path(default_temp_parent())
        ancestor = git_ancestor(parent.resolve())
    except (OSError, RuntimeError) as exc:  # no usable temp directory: name the gap, never block a start
        return (f"this host's default temp parent is unknown ({type(exc).__name__}): "
                + ("pass safe_test --temp-parent" if launcher else "choose a place") + " outside every Git working tree")
    if ancestor is None:
        return f"safe_test's default temp parent on this host is {parent}" if launcher else f"for example under {parent}"
    return (f"this host's default temp parent {parent} is inside the Git working tree {ancestor}"
            + (", which safe_test refuses: pass --temp-parent" if launcher else ": choose a place")
            + " outside every Git working tree")


def disposable_state_instruction(evidence_root: str, *, investigation: bool = False, granted: str = "") -> str:
    """Where a session's own checks put scratch and evidence (#2070).

    Work product stays in the private copy; disposable process state is the
    session's task-owned scratch, released or retained by its producer outside
    every Git tree, so verification never discovers a checkout by walking up; new
    named evidence goes where the host retains it. Build and dependency output that only
    serves checks is disposable by purpose, never by name or size (#2642): in a Git copy it
    may stay only as new files Git ignores (tracking overrides ignore rules),
    never in the evidence folder, whose retention applies no such filter. ``evidence_root`` "" is a copy
    captured whole: a payload or folder product retains no evidence folder, while an
    evidence-only investigation's copy is itself retained (its PURPOSE line names the
    folder), so it learns to keep that output out and where outside scratch goes. Project-neutral: the body's
    own launcher is named in its environment facts. ``granted`` is the engine-granted
    scratch, named instead of the host's default temp parent (``scratch_fact``).
    """
    scratch = scratch_fact(granted=granted)
    if not evidence_root and investigation:
        return (f"This copy is itself your retained evidence, so keep {CHECK_BYPRODUCTS} out of it; disposable "
                f"process state you keep outside this copy goes outside every Git working tree ({scratch}).")
    if not evidence_root:
        return ("This copy is itself the result: every file you leave in it becomes part of what is "
                f"captured, so keep test scratch, {CHECK_BYPRODUCTS} and evidence outside it and outside every "
                f"Git working tree ({scratch}); no evidence folder is retained for this run, so report "
                "verification results in your answer.")
    return ("Disposable process state you create — test scratch, package-manager temporaries and "
            f"{CHECK_BYPRODUCTS} — is not part of that work: keep it outside every Git working tree ({scratch}) "
            "or, where a tool must keep it in the project tree, in new files Git ignores here (tracking overrides "
            "ignore rules), setting each tool's output and cache location rather than "
            "trusting a default, which may land where it is captured or in the author's checkout; durable package "
            "downloads go only to the shared download cache your host names. Keep verification evidence worth "
            f"returning under {evidence_root}: the host retains new files there before it releases the copy and "
            "never applies them; files the project already tracks there stay ordinary work product. That folder "
            "is for reports, logs, sources, repro scripts and intended binaries of any size; its retention applies "
            "no cache, binary, size or ignore filter, so by-products left there are kept too, and a link or special "
            "entry is only recorded as a gap.")


# Persisted with every run's instructions, which a resumed vendor session may keep from its
# first request (#2513): run-specific facts are scoped to their run, and the current one is
# named by identity (this turn's host facts, the harness's starting directory), never by position.
_ACCESS_PRECEDENCE = (
    "this line governs native process access, while explicit task constraints "
    "and the assigned edit target still bind. This access, the concrete edit, evidence, "
    "scratch and copy-local environment locations, and what a private copy makes read-only "
    "belong to ONE run: a continued run's own host facts in its current turn restate them "
    "within the same owner, task, target and access ceiling, and work orders, bindings or "
    "evidence quoted from an earlier run are history wherever they appear. When a private "
    "delegated snapshot exists, the current run's DELEGATED EXECUTION BINDING is the assigned "
    "edit target and supersedes path fields in the inherited contract. A run's root is the "
    "directory your harness started its current turn in; a path inside an earlier run's copy "
    "means the same relative path from there."
)

READONLY_SOURCE_SCOPE = (
    "The starting folder is a location, not a read boundary. Task-authorized sources remain within the parent's existing read authority, "
    "subject to explicit narrower constraints and this run's tools; naming a path copies "
    "nothing and widens no authority. Credentials and harness runtime state are outside that scope "
    "unless specifically authorized. "
)

ACCESS_INSTRUCTIONS = {
    "readonly": (
        " ACCESS: " + READONLY_SOURCE_SCOPE + "Read-only commands are allowed when the effective tools provide "
        "them. Make no edits or writes; " + _ACCESS_PRECEDENCE
    ),
    "workspace_write": (
        " ACCESS: you may edit inside this root, and outside it write only disposable process state "
        "outside every Git working tree and the shared download cache your host names, never another "
        "project or the operator's files; " + _ACCESS_PRECEDENCE
    ),
    "full": (
        " ACCESS: full native process access is requested for this assignment, "
        "with source edits delivered through the assigned root; effective access "
        "is established by the run receipt, and the private snapshot is not an OS "
        "sandbox; " + _ACCESS_PRECEDENCE
    ),
}


INVESTIGATION_INSTRUCTION = (
    " PURPOSE: this is an evidence-only investigation. Your private copy exists so you can run the "
    "checks you judge useful (tests, scripts, fixtures, temporary files); nothing in it is applied to "
    "the reviewed or researched tree or creates an obligation to apply it. Put any file worth returning "
    "under .review-evidence/: the host retains that folder and the copy's other changes as evidence, with "
    "no cache or binary filter."
)


def access_instruction(access: str, *, investigation: bool = False) -> str:
    """The one instruction for a run's typed access profile, or "".

    A parent's prose ban ("Read-only no edits/commands...") in a work order once
    duplicated and contradicted the profile the host had already derived, and the
    run died unable to reach its own read surface. `DelegatedRunShape.access` is
    the authority: this instruction explains its read/write scope and precedence,
    rather than adding a second access policy in the work order. Naming a source
    does not copy it or widen authority. The parent's prose is never parsed.
    An unrecognized profile renders nothing rather than inventing a rule.
    """
    line = ACCESS_INSTRUCTIONS.get(str(access or "").strip(), "")
    if line and access != "readonly":
        line += (
            " If native permissions block a check, preserve its exact command and diagnostics and "
            "report it as unverified; do not widen access or weaken the check. The integrating "
            "caller may inspect the candidate and run that check separately under its own authority."
        )
        if investigation:
            line += INVESTIGATION_INSTRUCTION
    return line


def execution_binding_fingerprint(execution_root: str, authority_root: str,
                                  binding_kind: str = "snapshot") -> str:
    """Digest the typed execution/authority binding stored with custody."""
    execution = str(execution_root or "").strip()
    authority = str(authority_root or "").strip()
    return sha256(f"{binding_kind}\0{execution}\0{authority}".encode("utf-8")).hexdigest()


def execution_binding_text(execution_root: str, authority_root: str,
                           binding_sha256: str = "", *, evidence_root: str | None = None,
                           granted_scratch: str = "") -> str:
    """The binding names one work-product root, never every process write.

    The contract's stable authority/project root remains the integration identity.
    A private snapshot is the child's write surface, so this separate typed fact
    names both explicitly without rewriting the canonical work order. It is appended after
    that order, so an owner-facing absolute path cannot silently win by omission or by appearing earlier.
    ``evidence_root`` defaults to the copy's retained evidence folder; "" (a skill
    payload copy, whose every file is payload) says no folder is retained.
    ``granted_scratch`` is the scratch the run's request grants, the place it names.
    """
    execution = str(execution_root or "").strip()
    authority = str(authority_root or "").strip()
    if not execution:
        return ""
    digest = binding_sha256 or execution_binding_fingerprint(execution, authority)
    evidence = os.path.join(execution, EVIDENCE_DIRNAME) if evidence_root is None else evidence_root
    return (
        "\n\nDELEGATED EXECUTION BINDING (separate typed host fact; the canonical "
        f"work order remains byte-identical; sha256={digest}): "
        f"the private execution root for this run is {execution}: your work product — every file "
        "and patch change you make — goes only under it, by relative paths or absolute paths under it. "
        + ("This copy is a linked worktree of the author's repository: its branches, tags, stash and Git config "
           "are the author's, so create no branch or tag, do not stash and change no Git config. Files Git "
           "ignores here, and what Git repositories or submodules inside the copy hold, are not part of what is "
           "captured: `git add -f` an ignored file you must deliver. " if evidence_root is None else "")
        + disposable_state_instruction(evidence, granted=granted_scratch) + " The stable authority/project root "
        f"{authority or '(unknown)'} is a read-only identity/reference until the "
        "parent explicitly integrates the captured result. Do not write, patch, "
        "stage, reset, clean, or commit the authority root, another checkout or the operator's "
        "files. Compare only the directory your harness started this run in, physically: another "
        "spelling of it, such as through a symlink, matches, and a later cd changes nothing. If it is "
        "not this root, stop with a typed execution-root mismatch instead of falling back to the "
        "authority root."
    )


def directory_copy_binding_instruction(authority_root: str, binding_sha256: str = "", *,
                                       investigation: bool = False) -> str:
    """Tell a directory-copy child that the engine creates its write root later."""
    authority = str(authority_root or "").strip() or "(unknown)"
    digest = binding_sha256 or execution_binding_fingerprint("", authority, "directory_copy")
    return (
        "\n\nDELEGATED DIRECTORY COPY BINDING (separate typed host fact; the canonical "
        f"work order remains byte-identical; sha256={digest}): the engine will create a private "
        "execution copy for this run. Use only the engine-provided working "
        "directory/cwd for your work product; the selected authority folder "
        f"{authority} is a read-only source reference. Do not write to that "
        "authority folder directly. " + disposable_state_instruction("", investigation=investigation)
    )


def in_place_binding_instruction(root: str, service: Any = None) -> str:
    """Tell an in-place Git writer that its host's own folder is its working tree. ``service`` is
    what ``subagent_work_order.session_service_directories`` made for it: its disposable scratch,
    or the typed reason there is none."""
    scratch, why = getattr(service, "scratch", ""), getattr(service, "limitation", "")
    place = (f"{scratch_fact(granted=scratch)}, {DISPOSABLE_SCRATCH}" if scratch else scratch_fact()
             + (f" (no disposable scratch could be made for this run: {why}; nothing owns or removes that place, "
                "so report the gap)" if why else ""))
    return (
        "\n\nDELEGATED IN-PLACE BINDING (typed host fact): you work directly in "
        f"{str(root or '').strip() or '(unknown)'}, your host task's own folder, as it chose. Every change lands "
        "there as you make it and nothing is captured for a separate apply. Your host may be working there too: "
        "change only what the assignment needs; never reset, clean, stash, check out or commit. Keep test "
        "scratch, environments and caches outside every Git working tree, " + place + "."
    )


def no_copy_binding_instruction(workdir: str) -> str:
    """Name the root of a run no binding renderer covers (read-only, a direct folder): where
    the engine starts it, ``execution.workspaceRoot`` or else ``scope.root``."""
    return f"\n\nThis run has no private copy: its working directory is {str(workdir or '').strip() or '(unknown)'}."


def append_coordination_context(
    base_instructions: str,
    coordination_context: str,
) -> str:
    """Append the exact advisory context without changing instruction roles."""

    context = str(coordination_context or "")
    if not context:
        return base_instructions
    coordination_sha = sha256(context.encode("utf-8")).hexdigest()
    appendix = (
        "\n\nHOST COORDINATION CONTEXT (advisory appendix; canonical work-order "
        f"authority remains unchanged; sha256={coordination_sha}):\n{context}"
    )
    return base_instructions + appendix

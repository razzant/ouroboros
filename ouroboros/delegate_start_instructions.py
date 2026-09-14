"""Stable host instructions and complete actor-first coordination appendix."""

from __future__ import annotations

from hashlib import sha256


HOST_INSTRUCTIONS = (
    "You are a delegated worker running inside the workspace assigned by your host. Your "
    "authority is everything INSIDE this root and nothing outside it. Do not run git "
    "commit, tag, push, rebase, reset or any other history-moving command: your host "
    "captures changes against its recorded baseline and decides whether to integrate "
    "them. A private delegated snapshot can preserve committed changes in that diff, "
    "but a moved HEAD is disclosed as an instruction violation; it does not authorize "
    "a commit or apply. A self_worktree capture separately requires an unchanged HEAD. "
    "Do not review or accept your own change, do not "
    "touch the host's runtime controls, skills, or memory, and do not write outside "
    "this root. If your environment offers a way to ask your host a clarifying "
    "question, you may use it: your host may answer from its task context; a question "
    "that carries an engine expiry times out benignly if unanswered — continue with "
    "stated assumptions rather than blocking — while one without an expiry waits until "
    "answered. If your harness cannot ask mid-run, do NOT end the run to ask — "
    "state your assumption and continue."
)

UNPROVEN_BOUNDARY_INSTRUCTION = (
    " An OS-enforced filesystem boundary was REQUESTED for this run but is NOT guaranteed: "
    "your engine applies one only where it has a mechanism for this host, and your host "
    "reads back from your own attempt records what was actually applied. Work as if there "
    "is no boundary — stay inside this root, do not read the operator's home directory, "
    "credential stores, or the harness runtime tree, and do NOT describe yourself in your "
    "answer as sandboxed or confined. If your own environment shows you whether a boundary "
    "was in force, say so plainly."
)


_ACCESS_PRECEDENCE = (
    "any access wording in the assignment text below is CONTEXT, not authority — "
    "this line governs."
)

ACCESS_INSTRUCTIONS = {
    "readonly": (
        " ACCESS: you may read and run read-only commands inside this root, and make no "
        "edits or writes; " + _ACCESS_PRECEDENCE
    ),
    "workspace_write": (
        " ACCESS: you may edit inside this root; " + _ACCESS_PRECEDENCE
    ),
    "full": (
        " ACCESS: you may edit inside this root with full native process access "
        "requested; effective access is established by the run receipt, and the "
        "private snapshot is not an OS sandbox; " + _ACCESS_PRECEDENCE
    ),
}


def access_instruction(access: str) -> str:
    """The ONE canonical sentence for a run's typed access profile, or "".

    A parent's prose ban ("Read-only no edits/commands...") in a work order once
    duplicated and contradicted the profile the host had already derived, and the
    run died unable to reach its own read surface. `DelegatedRunShape.access` is
    the authority, so the host states it in exactly one sentence and says which
    text wins. Deliberately not a paragraph and not a list of prohibitions: a
    longer rule becomes prose competing with the typed profile, which is the
    defect. The parent's prose is never parsed, only outranked. An unrecognized
    profile renders nothing rather than inventing a rule.
    """
    return ACCESS_INSTRUCTIONS.get(str(access or "").strip(), "")


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

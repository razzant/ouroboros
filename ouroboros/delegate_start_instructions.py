"""Stable host instructions and bounded actor-first coordination appendix."""

from __future__ import annotations

from hashlib import sha256

from ouroboros.delegate_shared import _fail


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


def append_coordination_context(
    base_instructions: str,
    coordination_context: str,
    *,
    instruction_budget_chars: int,
) -> tuple[str, str]:
    """Append the exact advisory context or refuse before physical start."""

    context = str(coordination_context or "").strip()
    if not context:
        return base_instructions, ""
    coordination_sha = sha256(context.encode("utf-8")).hexdigest()
    appendix = (
        "\n\nHOST COORDINATION CONTEXT (advisory appendix; canonical work-order "
        f"authority remains unchanged; sha256={coordination_sha}):\n{context}"
    )
    required_chars = len(base_instructions) + len(appendix)
    if required_chars > instruction_budget_chars:
        return "", _fail(
            "delegate_start",
            "coordination_context_over_budget",
            "The complete coordination appendix does not fit the existing host "
            "instruction-field budget; it was not truncated and the physical leaf "
            "was not started. Retry with a shorter coordination context or preserve "
            "the details in a host artifact/tree note.",
            coordination_context_chars=len(context),
            required_instruction_chars=required_chars,
            instruction_budget_chars=instruction_budget_chars,
            coordination_context_sha256=coordination_sha,
            host_fallback=False,
        )
    return base_instructions + appendix, ""

"""Single source of truth for the LLM context LAYOUT of reference docs.

This module decides, in ONE place, which governance / reference docs enter the
always-on agent context and in what form, per context mode (low / max) and task
kind. Centralizing it here keeps the low/max split from drifting across the
surfaces that consume it (main task context, background consciousness, deep
self-review).

Doc matrix (agent cognition surfaces):

  | doc            | max            | low                                   |
  |----------------|----------------|---------------------------------------|
  | SYSTEM / BIBLE | full (tier-0; caller-owned, never varied here)              |
  | ARCHITECTURE   | full           | navigation map (full sections on demand) |
  | DEVELOPMENT    | full           | full when caller marks/keeps dev context; else pointer |
  | README         | on-demand pointer (removed from always-on for all modes)     |
  | CHECKLISTS     | on-demand pointer (reviewers load their own copy)            |

The TIER-0 protected core (SYSTEM, BIBLE, identity, scratchpad, knowledge index,
recent dialogue) is ALWAYS full in every mode (BIBLE P1 cognitive-horizon / P4)
and is declared here as a data invariant. Memory-section SIZE (not inclusion) is
governed separately by consolidation granularity, not by this layout.

No imports from ``ouroboros.context`` (avoids a circular import); docs are read
directly via ``env.repo_path``.
"""

from __future__ import annotations

from typing import Any, List

# Protected core: always rendered in full, in every context mode. Encoded as
# data so a drift-guard test can assert no future change demotes it.
TIER0_ALWAYS_FULL = frozenset({
    "system",
    "bible",
    "identity",
    "scratchpad",
    "knowledge_index",
    "recent_dialogue",
})


def _read_doc(env: Any, rel_path: str) -> str:
    try:
        return env.repo_path(rel_path).read_text(encoding="utf-8")
    except Exception:
        return ""


def generate_doc_nav_map(text: str, *, title: str, rel_path: str) -> str:
    """Build a compact, fence-aware navigation map of a markdown doc.

    Lists every ``##`` / ``###`` heading with its line range so the agent knows
    what exists and where, and can pull the full section on demand via
    ``read_file(root="system_repo", path=rel_path, start_line=A, max_lines=N)``.
    This is a lossless index
    (P1: no silent truncation) — the single canonical file on disk is unchanged.
    """
    lines = text.splitlines()
    total = len(lines)
    headings: List[tuple[int, str, int]] = []  # (level, title, 1-based line)
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.startswith("## "):
            headings.append((2, line[3:].strip(), i))
        elif line.startswith("### "):
            headings.append((3, line[4:].strip(), i))

    out = [
        f"## {title} (navigation map)",
        "",
        f"Full text is NOT inlined to keep the working context window fit. Read any "
        f"section on demand with `read_file(root=\"system_repo\", path=\"{rel_path}\", "
        f"start_line=A, max_lines=N)` (untruncated). Sections:",
        "",
    ]
    if not headings:
        out.append(f"- (no `##`/`###` headings; read `{rel_path}` directly)")
    for idx, (level, htitle, lineno) in enumerate(headings):
        end = headings[idx + 1][2] - 1 if idx + 1 < len(headings) else total
        indent = "  " if level == 3 else ""
        out.append(f"{indent}- {htitle} — lines {lineno}-{end}")
    return "\n".join(out)


def architecture_context_section(
    env: Any,
    *,
    context_mode: str,
    text: str | None = None,
) -> str:
    """ARCHITECTURE.md: full in max, navigation map in low. Empty if unreadable."""
    if text is None:
        text = _read_doc(env, "docs/ARCHITECTURE.md")
    if not text.strip():
        return ""
    if context_mode == "low":
        return generate_doc_nav_map(
            text, title="ARCHITECTURE.md", rel_path="docs/ARCHITECTURE.md"
        )
    return "## ARCHITECTURE.md\n\n" + text


def reference_doc_sections(
    env: Any,
    *,
    context_mode: str,
    is_code_task: bool,
    architecture_text: str | None = None,
    development_text: str | None = None,
) -> List[str]:
    """Return the reference-doc parts for the always-on static block.

    SYSTEM.md and BIBLE.md are tier-0 and added by the caller; this owns
    ARCHITECTURE / DEVELOPMENT / README / CHECKLISTS per the doc matrix. Anything
    not inlined is named in a single visible on-demand pointer (P1: no silent
    omission).
    """
    low = context_mode == "low"
    parts: List[str] = []
    on_demand: List[str] = []

    arch_section = architecture_context_section(
        env,
        context_mode=context_mode,
        text=architecture_text,
    )
    if arch_section:
        parts.append(arch_section)

    dev_text = (
        development_text
        if development_text is not None
        else _read_doc(env, "docs/DEVELOPMENT.md")
    )
    if dev_text.strip():
        # DEVELOPMENT is the engineering handbook — full in max; in low only
        # when the caller marks/keeps development context, else an on-demand
        # pointer. Direct chat is not a reliable non-code signal by itself.
        if (not low) or is_code_task:
            parts.append("## DEVELOPMENT.md\n\n" + dev_text)
        else:
            on_demand.append("docs/DEVELOPMENT.md")

    # README (user-facing), CHECKLISTS (reviewers load their own copy), and
    # ARCHITECTURE_REFERENCE (detailed module descriptions) are not inlined in
    # the agent context in any mode.
    on_demand.extend(["README.md", "docs/CHECKLISTS.md", "docs/ARCHITECTURE_REFERENCE.md"])

    if on_demand:
        listing = ", ".join(f"`{p}`" for p in on_demand)
        parts.append(
            "## Reference docs available on demand\n\n"
            f"Not inlined in the working context: {listing}. "
            "Read them in full (untruncated) with `read_file(root=\"system_repo\", path=...)` when relevant."
        )
    return parts

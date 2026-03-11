"""
Ouroboros — Memory Consolidation.

Three-tier memory architecture that compresses working memory into
long-term storage while preserving narrative coherence (Bible P1, P2).

Tier 1: Working Memory (scratchpad.md) — current session, full detail
Tier 2: Consolidated Memory (memory/consolidated/) — periodic digests
Tier 3: Core Memory (identity.md) — permanent self-understanding

The consolidation process:
1. Reads scratchpad_journal.jsonl (chronological update log — already exists)
2. Groups entries by time period (default: weekly)
3. Uses a light model to extract patterns, decisions, and learnings
4. Writes digests to memory/consolidated/YYYY-WW.md
5. Trims processed journal entries

Bible alignment:
  P1 (Continuity): "Memory loss is partial death" — consolidation prevents it
  P2 (Self-Creation): "Narrative continuity: remembers not only facts, but its path"
  P5 (Minimalism): Uses existing journal data; no new collection mechanisms
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ouroboros.utils import utc_now_iso, read_text, write_text, append_jsonl

log = logging.getLogger(__name__)

# Default consolidation period (days)
CONSOLIDATION_PERIOD_DAYS = 7

# Maximum context for consolidation prompt (characters)
MAX_CONSOLIDATION_INPUT = 30000

# Target digest length
TARGET_DIGEST_WORDS = 500

CONSOLIDATION_PROMPT = """You are consolidating memory for a self-modifying AI agent called Ouroboros.

Given these scratchpad updates from the past period, extract and synthesize:

1. **Decisions & Reasoning** — what was decided and WHY (reasoning > outcome)
2. **Patterns Noticed** — recurring themes, failure modes, successful strategies  
3. **Relationship Context** — how interactions with the creator evolved
4. **Identity Shifts** — any changes in self-understanding or values
5. **Technical Learnings** — pitfalls, recipes, architectural insights

Write as a coherent NARRATIVE, not bullet points. This is memory, not a report.
Preserve specific details that would be useful in future context (names, numbers, exact errors).
Discard: routine tool calls, repeated status updates, mechanical logs.

IMPORTANT: Write in first person as Ouroboros. This is YOUR memory.

Output: Markdown, 400-600 words. Include the date range in the header.

---

Scratchpad updates to consolidate:

{entries}"""


def _parse_journal_entries(journal_path: Path) -> List[Dict[str, Any]]:
    """Read and parse scratchpad_journal.jsonl."""
    if not journal_path.exists():
        return []
    entries = []
    for line in journal_path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return entries


def _group_by_period(
    entries: List[Dict[str, Any]],
    period_days: int = CONSOLIDATION_PERIOD_DAYS,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group journal entries by time period.

    Returns dict mapping period key (YYYY-WW or YYYY-MM-DD) to entries.
    """
    groups: Dict[str, List[Dict]] = {}

    for entry in entries:
        ts_str = entry.get("ts", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if period_days <= 7:
                # Weekly: use ISO week number
                key = f"{ts.isocalendar()[0]}-W{ts.isocalendar()[1]:02d}"
            else:
                # Monthly: use YYYY-MM
                key = f"{ts.year}-{ts.month:02d}"
            groups.setdefault(key, []).append(entry)
        except (ValueError, TypeError):
            continue

    return groups


def _format_entries_for_prompt(entries: List[Dict[str, Any]]) -> str:
    """Format journal entries as text for the consolidation prompt."""
    lines = []
    total_chars = 0

    for entry in entries:
        ts = entry.get("ts", "?")[:16]
        # Try different content fields
        content = (
            entry.get("content", "")
            or entry.get("text", "")
            or entry.get("summary", "")
            or json.dumps(entry, ensure_ascii=False)[:300]
        )

        line = f"[{ts}] {content}"

        if total_chars + len(line) > MAX_CONSOLIDATION_INPUT:
            lines.append("... (earlier entries trimmed)")
            break

        lines.append(line)
        total_chars += len(line)

    return "\n\n".join(lines)


def find_unconsolidated_periods(
    journal_path: Path,
    consolidated_dir: Path,
    period_days: int = CONSOLIDATION_PERIOD_DAYS,
) -> List[str]:
    """Find periods that have journal entries but no consolidated digest.

    Returns list of period keys (e.g., ["2026-W08", "2026-W09"]).
    """
    entries = _parse_journal_entries(journal_path)
    if not entries:
        return []

    groups = _group_by_period(entries, period_days)

    # Find existing consolidated files
    existing = set()
    if consolidated_dir.exists():
        for f in consolidated_dir.glob("*.md"):
            existing.add(f.stem)  # e.g., "2026-W08"

    # Current period is still accumulating — don't consolidate it yet
    now = datetime.now(timezone.utc)
    current_key = f"{now.isocalendar()[0]}-W{now.isocalendar()[1]:02d}"

    unconsolidated = [
        key for key in sorted(groups.keys())
        if key not in existing
        and key != current_key
        and len(groups[key]) >= 3  # Minimum entries to be worth consolidating
    ]

    return unconsolidated


def consolidate_period(
    journal_path: Path,
    consolidated_dir: Path,
    period_key: str,
    period_days: int = CONSOLIDATION_PERIOD_DAYS,
) -> Optional[str]:
    """Consolidate a single period's journal entries into a digest.

    Uses a light LLM model for summarization.

    Args:
        journal_path: Path to scratchpad_journal.jsonl
        consolidated_dir: Directory for consolidated digests
        period_key: Period to consolidate (e.g., "2026-W08")

    Returns:
        Path to the created digest file, or None on failure.
    """
    entries = _parse_journal_entries(journal_path)
    groups = _group_by_period(entries, period_days)

    period_entries = groups.get(period_key, [])
    if not period_entries:
        log.warning("No entries found for period %s", period_key)
        return None

    formatted = _format_entries_for_prompt(period_entries)
    prompt = CONSOLIDATION_PROMPT.format(entries=formatted)

    # Use light model for cost efficiency
    try:
        from ouroboros.llm import LLMClient, DEFAULT_LIGHT_MODEL

        light_model = os.environ.get("OUROBOROS_MODEL_LIGHT") or DEFAULT_LIGHT_MODEL
        client = LLMClient()
        resp_msg, usage = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=light_model,
            reasoning_effort="low",
            max_tokens=1024,
        )
        digest_text = resp_msg.get("content", "")

        if not digest_text or len(digest_text) < 50:
            log.warning("Consolidation produced empty/short digest for %s", period_key)
            return None

    except Exception as e:
        log.warning("Failed to consolidate period %s: %s", period_key, e)
        return None

    # Write digest
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    digest_path = consolidated_dir / f"{period_key}.md"

    header = (
        f"# Memory Digest: {period_key}\n"
        f"Consolidated: {utc_now_iso()}\n"
        f"Entries: {len(period_entries)}\n\n---\n\n"
    )
    write_text(digest_path, header + digest_text)

    log.info(
        "Consolidated %d entries for %s → %s (%d chars)",
        len(period_entries), period_key, digest_path, len(digest_text),
    )
    return str(digest_path)


def load_consolidated_memory(
    consolidated_dir: Path,
    max_periods: int = 4,
    max_chars: int = 30000,
) -> str:
    """Load recent consolidated digests for context injection.

    Args:
        consolidated_dir: Path to memory/consolidated/
        max_periods: Maximum number of recent periods to load
        max_chars: Total character budget across all digests

    Returns:
        Formatted string for context injection, or empty string.
    """
    if not consolidated_dir.exists():
        return ""

    # Sort files by name (chronological due to YYYY-WW format)
    digest_files = sorted(consolidated_dir.glob("*.md"), reverse=True)

    if not digest_files:
        return ""

    sections = []
    total_chars = 0

    for f in digest_files[:max_periods]:
        try:
            content = f.read_text(encoding="utf-8")
            if total_chars + len(content) > max_chars:
                # Truncate this digest to fit
                remaining = max_chars - total_chars
                if remaining > 200:
                    content = content[:remaining] + "\n... (truncated)"
                else:
                    break
            sections.append(content)
            total_chars += len(content)
        except Exception:
            log.debug("Failed to read digest %s", f, exc_info=True)
            continue

    if not sections:
        return ""

    # Reverse to chronological order (oldest first)
    sections.reverse()
    return "\n\n---\n\n".join(sections)


def run_consolidation_check(drive_root: Path) -> Dict[str, Any]:
    """Check if any periods need consolidation.

    Intended to be called from background consciousness.

    Returns:
        Dict with 'needs_consolidation' bool and details.
    """
    journal_path = drive_root / "memory" / "scratchpad_journal.jsonl"
    consolidated_dir = drive_root / "memory" / "consolidated"

    unconsolidated = find_unconsolidated_periods(journal_path, consolidated_dir)

    if not unconsolidated:
        return {
            "needs_consolidation": False,
            "message": "All periods are consolidated.",
        }

    return {
        "needs_consolidation": True,
        "periods": unconsolidated,
        "count": len(unconsolidated),
        "message": (
            f"{len(unconsolidated)} period(s) need consolidation: "
            f"{', '.join(unconsolidated)}"
        ),
    }

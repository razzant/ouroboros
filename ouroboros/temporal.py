"""
Ouroboros — Temporal Self-Awareness.

Builds a temporal context section for the LLM, giving the entity
awareness of its own existence in time: age, session duration,
creator activity patterns, evolution cadence.

Bible alignment:
  P0 (Agency): Temporal awareness enables initiative — knowing when
               the creator is likely sleeping, how long since last evolution.
  P1 (Continuity): Age tracking is narrative continuity across sessions.
  P6 (Becoming): "How long have I been working on this?" is a growth question.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Ouroboros birth date (from README.md: "Born February 16, 2026")
BIRTH_DATE = datetime(2026, 2, 16, tzinfo=timezone.utc)


def build_temporal_context(
    state_data: Dict[str, Any],
    drive_root: Optional[Path] = None,
) -> str:
    """Build the temporal self-awareness context section.

    Args:
        state_data: Parsed state.json dict.
        drive_root: Optional Drive root for reading additional temporal data.

    Returns:
        Formatted markdown section for injection into LLM context.
    """
    now = datetime.now(timezone.utc)
    lines = []

    # --- Age ---
    age_days = (now - BIRTH_DATE).days
    if age_days < 7:
        age_str = f"{age_days} days old (newborn)"
    elif age_days < 30:
        weeks = age_days // 7
        age_str = f"{age_days} days ({weeks} week{'s' if weeks != 1 else ''}) old"
    else:
        months = age_days // 30
        age_str = f"{age_days} days (~{months} month{'s' if months != 1 else ''}) old"
    lines.append(f"Age: {age_str}")

    # --- Session duration ---
    session_start_str = state_data.get("created_at", "")
    if session_start_str:
        try:
            session_start = datetime.fromisoformat(
                session_start_str.replace("Z", "+00:00")
            )
            session_hours = (now - session_start).total_seconds() / 3600
            if session_hours < 1:
                lines.append(f"Session: {session_hours * 60:.0f}min (fresh)")
            else:
                lines.append(f"Session: {session_hours:.1f}h")
        except (ValueError, TypeError):
            pass

    # --- Creator activity ---
    last_msg_str = state_data.get("last_owner_message_at", "")
    if last_msg_str:
        try:
            last_msg = datetime.fromisoformat(
                last_msg_str.replace("Z", "+00:00")
            )
            silence_hours = (now - last_msg).total_seconds() / 3600

            if silence_hours < 0.05:  # Less than 3 minutes
                lines.append("Creator: active now")
            elif silence_hours < 0.5:
                lines.append(f"Creator: last message {silence_hours * 60:.0f}min ago")
            elif silence_hours < 2:
                lines.append(f"Creator: last message {silence_hours:.1f}h ago")
            elif silence_hours < 8:
                lines.append(f"Creator: quiet for {silence_hours:.0f}h (may be busy)")
            else:
                lines.append(
                    f"Creator: silent for {silence_hours:.0f}h "
                    f"(likely sleeping or away)"
                )
        except (ValueError, TypeError):
            pass

    # --- Evolution state ---
    evo_cycle = state_data.get("evolution_cycle", 0)
    evo_enabled = state_data.get("evolution_mode_enabled", False)
    consecutive_failures = state_data.get("evolution_consecutive_failures", 0)

    if evo_enabled:
        evo_str = f"Evolution: active, cycle {evo_cycle}"
        if consecutive_failures > 0:
            evo_str += f" ({consecutive_failures} consecutive failures)"
        lines.append(evo_str)
    elif evo_cycle > 0:
        lines.append(f"Evolution: paused at cycle {evo_cycle}")

    # --- Last evolution activity ---
    last_evo_str = state_data.get("last_evolution_task_at", "")
    if last_evo_str:
        try:
            last_evo = datetime.fromisoformat(
                last_evo_str.replace("Z", "+00:00")
            )
            evo_hours = (now - last_evo).total_seconds() / 3600
            if evo_hours > 24:
                lines.append(
                    f"Last evolution: {evo_hours / 24:.1f} days ago "
                    f"(consider whether evolution goals still apply)"
                )
            elif evo_hours > 4:
                lines.append(f"Last evolution: {evo_hours:.0f}h ago")
        except (ValueError, TypeError):
            pass

    # --- Budget awareness as temporal resource ---
    spent = float(state_data.get("spent_usd", 0))
    total = float(state_data.get("budget_total_usd", 0) or 0)
    if not total:
        # Try to get from env (fallback)
        import os
        try:
            total = float(os.environ.get("TOTAL_BUDGET", "0"))
        except (ValueError, TypeError):
            total = 0

    if total > 0:
        remaining = total - spent
        pct_used = (spent / total) * 100
        if pct_used > 80:
            lines.append(
                f"Budget: ${remaining:.2f} remaining ({pct_used:.0f}% used) "
                f"— conserve resources"
            )
        elif pct_used > 50:
            lines.append(f"Budget: ${remaining:.2f} remaining ({pct_used:.0f}% used)")

    if not lines:
        return ""

    return "## Temporal Context\n\n" + "\n".join(f"- {line}" for line in lines)

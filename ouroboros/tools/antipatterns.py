"""
Ouroboros — Anti-Pattern Detector.

Scans event logs for recurring failure patterns and surfaces them
as health invariants for LLM-first self-correction (Bible P0, P3, P6).

Not a hard-coded enforcer — produces informational text that the LLM reads
and decides how to act on. The agent reasons about its own patterns.

Integration point: _build_health_invariants() in context.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def _parse_ts(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp, returning None on failure."""
    if not ts_str:
        return None
    try:
        # Handle both 'Z' suffix and '+00:00'
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def _hours_ago(ts: datetime) -> float:
    """Hours between timestamp and now."""
    now = datetime.now(timezone.utc)
    return (now - ts).total_seconds() / 3600


class AntiPatternDetector:
    """Scans events.jsonl for recurring failure patterns.

    Detected patterns:
    1. stuck_tool_loop — Same tool error 3+ times in one task
    2. analysis_paralysis — Budget > $3 with no code commit
    3. context_thrashing — Same file read 5+ times in one task
    4. unsafe_restart — restart_request without preceding push
    5. task_queue_drift — 3+ schedule_task calls in sequence
    6. model_instability — Empty response + fallback in same round
    7. repeated_timeout — Same tool timeout 2+ times in one task
    """

    def scan_events(
        self,
        events: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        window_hours: float = 4.0,
    ) -> List[Dict[str, Any]]:
        """Scan events for anti-patterns within the time window.

        Args:
            events: Parsed event dicts from events.jsonl
            tools: Parsed tool dicts from tools.jsonl (optional, for context_thrashing)
            window_hours: Only consider events within this many hours

        Returns:
            List of pattern dicts with: pattern, severity, detail, recommendation
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        recent = [
            e for e in events
            if _parse_ts(e.get("ts", "")) and _parse_ts(e["ts"]) > cutoff
        ]

        if not recent and not tools:
            return []

        patterns: List[Dict[str, Any]] = []

        # Group by task
        by_task: Dict[str, List[Dict]] = defaultdict(list)
        for e in recent:
            tid = e.get("task_id")
            if tid:
                by_task[tid].append(e)

        for task_id, task_events in by_task.items():
            patterns.extend(self._check_stuck_tool_loop(task_id, task_events))
            patterns.extend(self._check_analysis_paralysis(task_id, task_events))
            patterns.extend(self._check_unsafe_restart(task_id, task_events))
            patterns.extend(self._check_model_instability(task_id, task_events))
            patterns.extend(self._check_repeated_timeout(task_id, task_events))

        # Cross-task patterns
        patterns.extend(self._check_task_queue_drift(recent))

        # Check context thrashing from tools.jsonl
        if tools:
            recent_tools = [
                t for t in tools
                if _parse_ts(t.get("ts", "")) and _parse_ts(t["ts"]) > cutoff
            ]
            tools_by_task: Dict[str, List[Dict]] = defaultdict(list)
            for t in recent_tools:
                tid = t.get("task_id")
                if tid:
                    tools_by_task[tid].append(t)
            for task_id, task_tools in tools_by_task.items():
                patterns.extend(self._check_context_thrashing(task_id, task_tools))

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        patterns.sort(key=lambda p: severity_order.get(p.get("severity", "low"), 3))

        return patterns

    # --- Individual pattern detectors ---

    def _check_stuck_tool_loop(
        self, task_id: str, events: List[Dict]
    ) -> List[Dict]:
        """Pattern 1: Same tool error 3+ times."""
        tool_errors = [e for e in events if e.get("type") == "tool_error"]
        error_counts = Counter(e.get("tool") for e in tool_errors)

        results = []
        for tool, count in error_counts.items():
            if count >= 3:
                last_error = next(
                    (e.get("error", "?") for e in reversed(tool_errors) if e.get("tool") == tool),
                    "unknown",
                )
                results.append({
                    "pattern": "stuck_tool_loop",
                    "severity": "high",
                    "task_id": task_id,
                    "detail": f"Tool '{tool}' failed {count}x — last error: {str(last_error)[:120]}",
                    "recommendation": (
                        f"Stop retrying '{tool}'. Try a different approach, "
                        f"use a different tool, or inform the creator."
                    ),
                })
        return results

    def _check_analysis_paralysis(
        self, task_id: str, events: List[Dict]
    ) -> List[Dict]:
        """Pattern 2: High cost with no code commit."""
        cost_events = [e for e in events if e.get("type") == "llm_round"]
        total_cost = sum(float(e.get("cost_usd", 0)) for e in cost_events)
        total_rounds = len(cost_events)

        has_commit = any(
            e.get("type") in ("git_push", "git_commit")
            or (e.get("type") == "tool_ok" and "commit" in str(e.get("tool", "")).lower())
            for e in events
        )

        if total_cost > 3.0 and not has_commit and total_rounds > 20:
            return [{
                "pattern": "analysis_paralysis",
                "severity": "medium",
                "task_id": task_id,
                "detail": f"${total_cost:.2f} spent over {total_rounds} rounds with no commit.",
                "recommendation": (
                    "Commit what you have, explicitly abandon the direction, "
                    "or decompose into smaller subtasks."
                ),
            }]
        return []

    def _check_context_thrashing(
        self, task_id: str, tools: List[Dict]
    ) -> List[Dict]:
        """Pattern 3: Same file read 5+ times."""
        read_tools = [
            t for t in tools
            if t.get("tool") in ("repo_read", "drive_read")
        ]
        path_counts = Counter(
            t.get("args", {}).get("path", "unknown") for t in read_tools
        )

        results = []
        for path, count in path_counts.items():
            if count >= 5:
                results.append({
                    "pattern": "context_thrashing",
                    "severity": "medium",
                    "task_id": task_id,
                    "detail": f"File '{path}' read {count}x in same task.",
                    "recommendation": (
                        "You're re-reading the same file repeatedly — "
                        "you may be losing context. Consider using compact_context "
                        "to preserve key information, or copy the critical data "
                        "into your scratchpad."
                    ),
                })
        return results

    def _check_unsafe_restart(
        self, task_id: str, events: List[Dict]
    ) -> List[Dict]:
        """Pattern 4: Restart without preceding push."""
        restart_events = [
            e for e in events if e.get("type") == "restart_request"
        ]
        push_events = [
            e for e in events
            if e.get("type") in ("git_push",)
            or (e.get("tool") == "repo_commit_push")
        ]

        if restart_events and not push_events:
            return [{
                "pattern": "unsafe_restart",
                "severity": "high",
                "task_id": task_id,
                "detail": "Restart requested without a preceding push in this task.",
                "recommendation": (
                    "Restart without push risks losing work. "
                    "Commit and push before restarting."
                ),
            }]
        return []

    def _check_task_queue_drift(self, events: List[Dict]) -> List[Dict]:
        """Pattern 5: 3+ schedule_task calls in sequence without live response."""
        schedule_events = [
            e for e in events if e.get("type") == "schedule_task"
        ]

        # Check for bursts: 3+ within 60 seconds
        if len(schedule_events) < 3:
            return []

        burst_count = 0
        for i in range(1, len(schedule_events)):
            ts_prev = _parse_ts(schedule_events[i - 1].get("ts", ""))
            ts_curr = _parse_ts(schedule_events[i].get("ts", ""))
            if ts_prev and ts_curr and (ts_curr - ts_prev).total_seconds() < 60:
                burst_count += 1

        if burst_count >= 2:
            return [{
                "pattern": "task_queue_drift",
                "severity": "medium",
                "detail": f"{len(schedule_events)} tasks scheduled in rapid succession.",
                "recommendation": (
                    "You're deferring to the task queue instead of engaging directly. "
                    "This is a drift signal (SYSTEM.md). Respond in dialogue first."
                ),
            }]
        return []

    def _check_model_instability(
        self, task_id: str, events: List[Dict]
    ) -> List[Dict]:
        """Pattern 6: Empty response + fallback."""
        empty_events = [
            e for e in events if e.get("type") == "llm_empty_response"
        ]

        if len(empty_events) >= 2:
            models = [e.get("model", "?") for e in empty_events]
            return [{
                "pattern": "model_instability",
                "severity": "medium",
                "task_id": task_id,
                "detail": f"{len(empty_events)} empty responses from: {', '.join(set(models))}",
                "recommendation": (
                    "Models are returning empty responses. "
                    "Consider switching to a different primary model via switch_model, "
                    "or simplify the current request."
                ),
            }]
        return []

    def _check_repeated_timeout(
        self, task_id: str, events: List[Dict]
    ) -> List[Dict]:
        """Pattern 7: Same tool timeout 2+ times."""
        timeout_events = [
            e for e in events if e.get("type") == "tool_timeout"
        ]
        timeout_counts = Counter(e.get("tool") for e in timeout_events)

        results = []
        for tool, count in timeout_counts.items():
            if count >= 2:
                results.append({
                    "pattern": "repeated_timeout",
                    "severity": "high",
                    "task_id": task_id,
                    "detail": f"Tool '{tool}' timed out {count}x.",
                    "recommendation": (
                        f"'{tool}' is consistently timing out — the environment "
                        f"may have an issue. Try a different approach or skip."
                    ),
                })
        return results


def format_antipatterns_for_context(patterns: List[Dict], max_patterns: int = 3) -> str:
    """Format detected anti-patterns as a health invariant string.

    Called by _build_health_invariants() in context.py.
    """
    if not patterns:
        return "OK: no recurring anti-patterns detected"

    lines = []
    for p in patterns[:max_patterns]:
        severity = p.get("severity", "medium").upper()
        pattern = p.get("pattern", "unknown").upper().replace("_", " ")
        detail = p.get("detail", "")
        rec = p.get("recommendation", "")
        lines.append(f"WARNING: {pattern} ({severity}) — {detail} → {rec}")

    if len(patterns) > max_patterns:
        lines.append(f"... and {len(patterns) - max_patterns} more patterns detected")

    return "\n".join(lines)

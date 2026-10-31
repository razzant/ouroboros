"""Host wait episodes in the progress stream: accounting and control-storage waits.

A wait is how the host ran, not the turn's work. Its producers write both phases of every episode
under its own ``episode_id`` (``_usage_wait._hold``, ``owner_pause.wait_member_fence``; owner
decision 1A(3), 14 Oct 2026, keeps them whole). Readers fold them: the model's ``## Recent
progress`` shows each run of waits as one line, and neither that window nor a chat history page
spends its quota on a wait row. The browser twin is ``web/modules/task_checkpoints.js``
(``waitEpisode``, ``foldWaitRow``); both word the cases of ``web/tests/fixtures/wait_fold_parity.json``.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple

# kind -> (an open episode, one ended episode, several ended episodes)
WAIT_LABELS = {
    "usage_lock_wait": ("Waiting for accounting access", "Accounting wait ended", "Accounting waits"),
    "control_store_wait": ("Waiting for control storage", "Control storage wait ended", "Control storage waits"),
}


def wait_episode(row: Any) -> Optional[Tuple[str, str, bool, Optional[float]]]:
    """``(kind, episode id, ended, elapsed seconds or None)`` of a host wait row, else None."""
    if not isinstance(row, dict) or row.get("checkpoint_kind") not in WAIT_LABELS or not row.get("episode_id"):
        return None
    elapsed = row.get("elapsed_sec")
    known = isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0
    return row["checkpoint_kind"], str(row["episode_id"]), row.get("phase") == "ended", float(elapsed) if known else None


def _seconds(value: float) -> str:
    # The float's exact value rounded half up, as the browser's toFixed and Math.round read it.
    exact = Decimal(value)
    if value < 10:
        return f"{exact.quantize(Decimal('0.1'), ROUND_HALF_UP)} s"
    whole = int(exact.quantize(Decimal("1"), ROUND_HALF_UP))
    return f"{whole} s" if whole < 60 else f"{whole // 60} min" + (f" {whole % 60} s" if whole % 60 else "")


def fold_line(kind: str, episodes: Dict[str, Tuple[bool, Optional[float]]]) -> str:
    """One line for a kind's episodes (id -> (ended, elapsed)): live while one is open, then
    how many ended and the longest measured one."""
    waiting, ended, plural = WAIT_LABELS[kind]
    closed = [elapsed for done, elapsed in episodes.values() if done]
    known = [elapsed for elapsed in closed if elapsed is not None]
    longest = _seconds(max(known)) if known else ""
    done = (" · ".join(filter(None, (ended, longest))) if len(closed) == 1 else
            f"{plural} ×{len(closed)}" + (f" · longest {longest}" if longest else "") if closed else "")
    return " · ".join(filter(None, (waiting if len(closed) < len(episodes) else "", done)))


def fold_progress(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Progress rows, oldest first, with each run of consecutive wait rows as one entry (the run's
    first ``ts``, ``host_waits`` set; one clause per kind, so alternating kinds stay one line);
    every other row passes unchanged."""
    folded: List[Dict[str, Any]] = []
    run: Optional[Dict[str, Dict[str, Tuple[bool, Optional[float]]]]] = None
    for entry in entries:
        wait = wait_episode(entry)
        if wait is None:
            folded.append(entry)
            run = None
            continue
        if run is None:
            run = {}
            folded.append({"ts": entry.get("ts", ""), "host_waits": True})
        episodes = run.setdefault(wait[0], {})
        seen_ended, seen_elapsed = episodes.get(wait[1], (False, None))
        episodes[wait[1]] = (seen_ended or wait[2], wait[3] if wait[2] else seen_elapsed)
        folded[-1]["text"] = "; ".join(fold_line(kind, kinds) for kind, kinds in run.items())
    return folded

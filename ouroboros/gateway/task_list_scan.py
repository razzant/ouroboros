"""Raw result-name scan for the unfiltered GET /api/tasks slice path.

Split out of ``ouroboros/gateway/tasks.py`` at its module-size ceiling: one
coherent concern — the creation-ts sort scan behind slice-before-projection
(v6.9x P2) plus the ABI-2 admission routing for candidates whose bytes fail
to parse. ``tasks.py`` re-imports these names (same objects), so the endpoint
wiring and tests keep their historical surface.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List

from ouroboros.task_result_schema import (
    quarantine_task_result,
    task_result_schema_refusal,
)
from ouroboros.utils import read_json_dict

# Process-wide {(results_dir, filename) -> raw ts} memo for the unfiltered list
# path. The raw `ts` is CREATION-STABLE (write_task_result sets it on the first
# write; later updates touch only updated_at), so entries never need
# invalidation — only deletions are dropped and new names decoded. Keyed by the
# directory too, so multiple drive roots (tests, child drives) never collide.
# Concurrency note: worst case a race re-reads a file and stores the identical
# creation-stable value; no lock needed.
_RAW_TS_MEMO: Dict[tuple, str] = {}


def _raw_sorted_result_names(results_dir: pathlib.Path) -> tuple[List[str], List[str]]:
    """``(sorted_names, malformed_names)`` for the unfiltered list scan.

    ``sorted_names`` is every parseable result filename, newest-first by RAW
    creation ts (memoized); a row whose file lacks `ts` sorts as
    minus-infinity (oldest), tie-broken by filename for determinism.
    ``malformed_names`` is every candidate whose bytes failed to parse: ABI-2
    forbids silently dropping it — the caller MUST route it through the same
    admission reader as every in-window row (quarantine + the ONE batched
    scan event), even when it would have sorted outside the slice window.
    A malformed name is never memoized, so a genuinely torn CONCURRENT write
    is re-read on the next request (and the quarantine primitive itself
    re-checks under the row's write lock — a row a concurrent writer just
    made admissible is KEPT, never moved)."""
    try:
        with os.scandir(results_dir) as entries:
            names = [entry.name for entry in entries if entry.name.endswith(".json")]
    except OSError:
        return [], []
    dir_key = str(results_dir)
    present = set(names)
    for key in [k for k in list(_RAW_TS_MEMO) if k[0] == dir_key and k[1] not in present]:
        _RAW_TS_MEMO.pop(key, None)
    decorated: List[tuple] = []
    malformed: List[str] = []
    for name in names:
        key = (dir_key, name)
        raw_ts = _RAW_TS_MEMO.get(key)
        if raw_ts is None:
            data = read_json_dict(results_dir / name)
            if data is None:
                malformed.append(name)
                continue
            raw_ts = str(data.get("ts") or "")
            _RAW_TS_MEMO[key] = raw_ts
        decorated.append((raw_ts, name))
    decorated.sort(reverse=True)  # "" (no ts) sorts after every real timestamp
    return [name for _ts, name in decorated], malformed


def _quarantine_malformed_candidates(
    results_dir: pathlib.Path, malformed_names: List[str],
) -> List[Dict[str, str]]:
    """Admission for scan candidates whose bytes failed to parse (ABI-2).

    A malformed candidate is NOT silently dropped — it reaches the same
    admission reader as every projected row, even beyond the slice window;
    only projection is skipped (an inadmissible row is never projected, and a
    row a concurrent write just made admissible re-enters the sort on the
    next request). Returns the quarantined entries for the caller's ONE
    batched scan event."""
    quarantined: List[Dict[str, str]] = []
    for name in malformed_names:
        path = results_dir / name
        raw = read_json_dict(path)
        if raw is None and not path.is_file():
            continue  # vanished between the scan and this read
        refusal = task_result_schema_refusal(raw)
        if not refusal:
            continue  # a concurrent write landed a fresh admissible row
        if quarantine_task_result(path, refusal) == "moved":
            quarantined.append({"task_id": path.stem, "reason": refusal})
    return quarantined

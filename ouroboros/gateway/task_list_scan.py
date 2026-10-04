"""Compact, stat-invalidated result facts for interactive read projections.

The files and their schema readers remain authoritative. The shared memo in
task_result_scan serves child lookup, name ordering, SSE lineage discovery
and Main's newest-result selection; selected full results still pass admission.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, List

from ouroboros.task_result_scan import raw_result_facts as _raw_result_facts
from ouroboros.task_result_schema import (
    quarantine_task_result,
    task_result_schema_refusal,
)
from ouroboros.utils import read_json_dict


def raw_result_facts(results_dir: pathlib.Path, *, reader=None) -> tuple[Dict[str, dict], List[str]]:
    """Gateway read injection over the shared task-result navigation memo."""
    return _raw_result_facts(results_dir, reader=reader or read_json_dict)


def _raw_sorted_result_names(results_dir: pathlib.Path) -> tuple[List[str], List[str]]:
    """``(sorted_names, malformed_names)`` for the unfiltered list scan.

    ``sorted_names`` is every parseable result filename, newest-first by RAW
    raw ts (stat-invalidated); a row whose file lacks `ts` sorts as
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
        rows, malformed = raw_result_facts(results_dir)
    except OSError:
        logging.getLogger(__name__).warning("Task-result directory is unreadable: %s", results_dir, exc_info=True)
        return [], []  # preserve the legacy list caller's fail-soft return
    decorated = [(row["ts"], name) for name, row in rows.items()]
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

"""Compact, stat-invalidated navigation facts for canonical task results.

This process-local memo is shared by child lookup and gateway projections.
It never admits a row or replaces the full schema reader.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List

from ouroboros.task_result_schema import task_result_schema_refusal
from ouroboros.utils import read_json_dict

# Immutable tuple values publish atomically. Failed and torn reads are not cached.
_RAW_TS_MEMO: Dict[tuple, tuple] = {}
_RESULT_FACT_KEYS = (
    "task_id", "id", "ts", "updated_at", "delegation_role", "parent_task_id",
    "project_id", "root_task_id", "child_drive_root", "headless_child_drive_root",
    "superseded_by", "retry_task_id",
)


def _result_stat(path: pathlib.Path) -> tuple:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def raw_result_facts(results_dir: pathlib.Path, *, reader=None) -> tuple[Dict[str, dict], List[str]]:
    """Read changed/new files only; return failed/unstable names for admission.

    A parseable refusal is retained as a fact for each caller's own schema
    reader. The memo never admits or quarantines a row. ``reader`` preserves
    the gateway's injectable read seam.
    """
    from ouroboros.presence_authority import presence_metadata_binding, presence_record_binding

    reader = reader or read_json_dict
    try:
        # Match the canonical glob's platform flavour, including Windows case
        # folding, without glob's suppression of directory-read errors.
        with os.scandir(results_dir) as entries:
            names = sorted(entry.name for entry in entries
                           if pathlib.Path(entry.name).match("*.json"))
    except FileNotFoundError:
        names = []
    dir_key = str(results_dir)
    present = set(names)
    for key in [k for k in list(_RAW_TS_MEMO) if k[0] == dir_key and k[1] not in present]:
        _RAW_TS_MEMO.pop(key, None)
    rows: Dict[str, dict] = {}
    malformed: List[str] = []
    for name in names:
        key = (dir_key, name)
        path = results_dir / name
        try:
            signature = _result_stat(path)
            cached = _RAW_TS_MEMO.get(key)
            if cached is not None and cached[0] == signature:
                rows[name] = dict(cached[1])
                continue
            _RAW_TS_MEMO.pop(key, None)
            data = reader(path)
            if data is None or _result_stat(path) != signature:
                malformed.append(name)
                continue
        except OSError:
            _RAW_TS_MEMO.pop(key, None)
            malformed.append(name)
            continue
        facts = {field: str(data.get(field) or "") for field in _RESULT_FACT_KEYS}
        facts["presence_binding_id"] = presence_record_binding(data)
        metadata = data.get("metadata")
        contract = data.get("task_contract")
        facts["presence_authority_recorded"] = (
            presence_metadata_binding(metadata) is not None
            or isinstance(contract, dict) and "capability_ceiling" in contract
        )
        facts["schema_refusal"] = task_result_schema_refusal(data)
        rows[name] = facts
        _RAW_TS_MEMO[key] = (signature, tuple(facts.items()))
    return rows, malformed

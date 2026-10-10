"""Test helpers over the usage store (``ouroboros/usage_store.py``).

The store keeps ONE current row per attempt (no superseded transition rows),
so a test that used to read ``state/usage_attempts.jsonl`` reads
``ledger_rows`` instead: every attempt's current row in write order (``seq``).
A test that set up history by writing journal lines writes them with
``write_journal`` BEFORE the first money call on that root, and the store's
one-time import picks them up exactly like an upgraded install.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Iterable, List


def ledger_rows(root: pathlib.Path) -> List[Dict[str, Any]]:
    """Every attempt's current row in write order; ``[]`` before any money
    exists (no store and no journal to import). A journal not imported yet is
    imported first, exactly as by any store reader."""
    from ouroboros import usage_store
    from ouroboros.usage_ledger import LEDGER_REL

    root = pathlib.Path(root)
    if not (root / usage_store.STORE_REL).exists() and not (root / LEDGER_REL).exists():
        return []
    return usage_store.read_usage_records(root)


def write_journal(root: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> pathlib.Path:
    """Write a retired-format journal (dense ``seq`` from 1 when absent)."""
    path = pathlib.Path(root) / "state" / "usage_attempts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, row in enumerate(rows, 1):
        lines.append(json.dumps({"seq": index, "ts": "2026-01-01T00:00:00+00:00", **row}, sort_keys=True))
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def write_compacted_journal(root: pathlib.Path, groups: Iterable[Dict[str, Any]],
                            rows: Iterable[Dict[str, Any]] = (), *, epoch: int = 1) -> pathlib.Path:
    """A retired-format journal as the retired compaction left it: one
    ``usage_baseline`` header, its ``usage_baseline_group`` aggregates (each
    with ``folded_attempt_count``, the store's ``weight``) and the retained
    ``rows`` after them. The header's provenance is notional (no archive
    segment is written; nothing ordinary reads it)."""
    groups = [dict(group) for group in groups]
    folded = sum(int(group["folded_attempt_count"]) for group in groups)
    source_rows = 3 * folded
    header = {"kind": "usage_baseline", "attempt_id": "baseline-fixture", "state": "settled",
              "baseline_id": "baseline-fixture", "compaction_epoch": epoch,
              "archive_rel": "archive/usage_ledger/segment_fixture.jsonl", "source_sha256": "0" * 64,
              "source_size_bytes": 1, "source_row_count": source_rows, "source_first_seq": 1,
              "source_last_seq": source_rows, "folded_row_count": source_rows,
              "folded_attempt_count": folded, "group_count": len(groups), "retained_row_count": 0}
    aggregates = [{"kind": "usage_baseline_group", "baseline_id": "baseline-fixture", "state": "settled",
                   **group} for group in groups]
    return write_journal(root, [header, *aggregates, *rows])


def store_row_count(root: pathlib.Path) -> int:
    from ouroboros import usage_store

    with usage_store.read(pathlib.Path(root)) as txn:
        return int(txn.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])


# ---- shared fixtures (moved from the retired writer-view tests) -------------
import pytest  # noqa: E402


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fresh data root. The store is NOT created here: a test that seeds a
    journal does so before its first money call, and the one-time import
    picks the journal up exactly as on an upgraded install."""
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000000")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    yield tmp_path
    from ouroboros import usage_store

    usage_store.forget(tmp_path)


def request(root, **values):
    from ouroboros import usage_accounting as ua

    return ua.AttemptRequest(**{"drive_root": root, "model": "stub", "provider": "local",
                                "task_id": "child", "root_task_id": "dominant",
                                "reservation_usd": 1.0, **values})


def seed(root, count, *, cost="0.0000015"):
    """``count`` settled journal attempts (reserved, dispatched, settled rows)."""
    rows = []
    for index in range(count):
        base = dict(attempt_id=f"seed-{index}", kind="attempt", root_task_id="dominant",
                    provider="local", model="stub", reservation_upper_bound_usd="0.25",
                    pricing_known=True, ts="2020-01-01T00:00:00Z")
        for state in ("reserved", "dispatched", "settled"):
            row = {**base, "state": state, "seq": len(rows) + 1}
            if state == "settled":
                row.update(cost_usd=cost, cost_final=True)
            rows.append(row)
    path = pathlib.Path(root) / "state" / "usage_attempts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def attempt_rows_in_start_order(root: pathlib.Path) -> List[Dict[str, Any]]:
    """Every attempt's current row in the order the attempts STARTED (their
    first transition), for tests that pin an attempt sequence."""
    from ouroboros import usage_store

    if not (pathlib.Path(root) / usage_store.STORE_REL).exists():
        return []
    with usage_store.read(pathlib.Path(root)) as txn:
        return [usage_store._decode(record) for record in
                txn.conn.execute("SELECT * FROM attempts ORDER BY seq_first, seq")]


def dispatched_attempts(root: pathlib.Path) -> List[Dict[str, Any]]:
    """The attempts that were dispatched, in start order: a current row that is
    dispatched or went past it (a dispatch is an attempt's second transition).
    Each keeps the dispatch context (``physical_context``) it was sent with."""
    return [row for row in attempt_rows_in_start_order(root)
            if row.get("state") == "dispatched" or int(row.get("revision") or 0) >= 3]

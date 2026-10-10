"""Build the F1 (#1195) history workload: retained archive segments + a store.

Shared by `tests/test_startup_historical_audit_server.py` (the real-server
readiness-versus-history case) and runnable as a script for measurements.

The shape is an install upgraded from a compacted journal: the retired
compactor left one archive segment per generation under
`archive/usage_ledger/` (the folded attempts' rows, kept as evidence and read
only by the explicit history audit), and the journal it left holds one
aggregate per generation plus the live tail. The server's boot imports that
journal into the usage store; the audit then asks the store and the retained
evidence. The segments are plain retained rows (the audit's reader verifies
no chain and no hashes), so this module writes them directly.

Everything here writes to a caller-supplied synthetic root.  It never reads or
touches the owner's data root.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

if __package__ is None and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_CASH = 0.000123


def _old_ts(age_days: float) -> str:
    moment = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=age_days)
    return moment.isoformat().replace("+00:00", "Z")


def _attempt_row(seq: int, attempt_id: str, task_id: str, ts: str, state: str) -> dict:
    """One valid attempt row (every chain begins reserved)."""
    row = {
        "seq": seq,
        "ts": ts,
        "attempt_id": attempt_id,
        "kind": "attempt",
        "state": state,
        "task_id": task_id,
        "root_task_id": task_id,
        "model": "fixture::model",
        "provider": "fixture",
        "category": "agent",
        "source": "fixture",
    }
    if state in ("reserved", "dispatched"):
        row["reservation_upper_bound_usd"] = 0.002
    elif state == "released":
        row["reason"] = "fixture_release"
    else:
        row["cost_usd"] = _CASH
        row["cost_final"] = True
        row["pricing_known"] = True
        row["prompt_tokens"] = 11
        row["completion_tokens"] = 7
    return row


# reserved -> dispatched -> settled (3 rows), or reserved -> released (2 rows).
# `_chain_shapes` hits an EXACT row count with a mix of the two, which is what
# "N valid attempt rows per generation" means.
def _chain_shapes(rows: int) -> list[tuple[str, ...]]:
    long_chain = ("reserved", "dispatched", "settled")
    short_chain = ("reserved", "released")
    remainder = rows % 3
    shorts = 0 if remainder == 0 else (2 if remainder == 1 else 1)
    longs = (rows - 2 * shorts) // 3
    if longs < 0:
        longs, shorts = 0, rows // 2
    return [long_chain] * longs + [short_chain] * shorts


def build_chain(root: Path, *, generations: int, rows_per_generation: int,
                progress_every: int = 0) -> dict:
    """Write one retained archive segment per generation and return bounded
    counting facts plus (under ``"_aggregates"``) the journal aggregate each
    generation folded into."""
    from ouroboros.usage_ledger import ARCHIVE_SEGMENT_DIR_REL

    directory = root / ARCHIVE_SEGMENT_DIR_REL
    directory.mkdir(parents=True, exist_ok=True)
    archived_ids: list[str] = []
    aggregates: list[dict] = []
    started = time.monotonic()
    for generation in range(1, generations + 1):
        ts = _old_ts(30 + generations - generation)
        lines, seq, settled = [], 0, 0
        shapes = _chain_shapes(rows_per_generation)
        for index, shape in enumerate(shapes):
            attempt_id = f"a{generation:04d}-{index:06d}"
            archived_ids.append(attempt_id)
            settled += shape[-1] == "settled"
            for state in shape:
                seq += 1
                lines.append(json.dumps(
                    _attempt_row(seq, attempt_id, f"task-{index % 17:03d}", ts, state),
                    separators=(",", ":"),
                ))
        (directory / f"segment_{generation:04d}.jsonl").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8")
        aggregates.append({
            "attempt_id": f"fold-{generation:04d}", "task_id": "task-000", "root_task_id": "task-000",
            "model": "fixture::model", "provider": "fixture", "category": "agent", "source": "fixture",
            "folded_attempt_count": len(shapes), "cost_usd": round(settled * _CASH, 12),
            "cost_final": True, "pricing_known": True, "ts": ts,
        })
        if progress_every and generation % progress_every == 0:
            print(f"  generation {generation}/{generations} "
                  f"({time.monotonic() - started:.1f}s)", file=sys.stderr, flush=True)
    segments = sorted(directory.glob("*.jsonl"))
    return {
        "generations": generations,
        "rows_per_generation": rows_per_generation,
        "attempt_chains_per_generation": len(_chain_shapes(rows_per_generation)),
        "rows_written_per_generation": sum(len(shape) for shape in _chain_shapes(rows_per_generation)),
        "archived_attempt_ids_expected": len(archived_ids),
        "segments_on_disk": len(segments),
        "archive_bytes": sum(path.stat().st_size for path in segments),
        "build_seconds": time.monotonic() - started,
        "first_archived_attempt_id": archived_ids[0] if archived_ids else "",
        "last_archived_attempt_id": archived_ids[-1] if archived_ids else "",
        "_aggregates": aggregates,
    }


def live_tail(rows: int) -> tuple[list[str], list[dict]]:
    """Recent attempt chains (never folded) and their ids."""
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    ids, chain_rows = [], []
    for index, shape in enumerate(_chain_shapes(rows)):
        attempt_id = f"live-{index:06d}"
        ids.append(attempt_id)
        for state in shape:
            row = _attempt_row(0, attempt_id, f"task-{index % 17:03d}", ts, state)
            row.pop("seq")
            chain_rows.append(row)
    return ids, chain_rows


def write_seal_manifests(root: Path, *, archived: list[str], live: list[str],
                         missing: int) -> dict:
    """Seal manifests over archived / live / absent attempt identities.

    The archived identities are the ones that force the retained-evidence scan:
    they are absent from the store by construction.
    """
    counts = {"archived": 0, "live": 0, "missing": 0}
    calls = root / "observability" / "calls"

    def _write(attempt_id: str, index: int) -> None:
        directory = calls / f"task-{index % 17:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{attempt_id}.json").write_text(json.dumps({
            "call_id": attempt_id,
            "task_id": f"task-{index % 17:03d}",
            "model_send_seal": {
                "attempt_id": attempt_id,
                "canonical_basis": "model_send_candidate_v1",
                "pre_redaction_sha256": hashlib.sha256(attempt_id.encode()).hexdigest(),
                "size_bytes": len(attempt_id),
            },
        }, separators=(",", ":")), encoding="utf-8")

    index = 0
    for attempt_id in archived:
        _write(attempt_id, index); index += 1; counts["archived"] += 1
    for attempt_id in live:
        _write(attempt_id, index); index += 1; counts["live"] += 1
    for offset in range(missing):
        _write(f"missing-{offset:06d}", index); index += 1; counts["missing"] += 1
    return counts


def prepare_root(root: Path, *, generations: int, rows_per_generation: int,
                 archived_seals: int, live_rows: int, live_seals: int,
                 missing_seals: int, progress_every: int = 0) -> dict:
    """Full F1 fixture: retained segments, the journal the next boot imports
    (aggregates + live tail) and the three seal identities. A store an earlier
    boot of this synthetic root created is removed so the journal is imported
    exactly as on an upgrade."""
    from ouroboros.usage_store import STORE_REL
    from tests._usage_store_testing import write_compacted_journal

    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    for stale in (root / STORE_REL,):
        if stale.exists():
            stale.unlink()
    facts = build_chain(root, generations=generations,
                        rows_per_generation=rows_per_generation,
                        progress_every=progress_every)
    live_ids, live_rows_written = live_tail(live_rows)
    journal = write_compacted_journal(root, facts.pop("_aggregates"), live_rows_written)
    facts["journal_bytes"] = journal.stat().st_size
    # Archived identities are sampled across the whole chain, not just its head.
    per_generation = len(_chain_shapes(rows_per_generation))
    step = max(1, facts["archived_attempt_ids_expected"] // max(1, archived_seals))
    archived_sample = [
        f"a{(1 + (offset * step) // per_generation):04d}-{(offset * step) % per_generation:06d}"
        for offset in range(archived_seals)
    ]
    counts = write_seal_manifests(
        root, archived=archived_sample, live=live_ids[:live_seals], missing=missing_seals,
    )
    facts["seal_manifests"] = counts
    facts["live_tail_rows"] = live_rows
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--archived-seals", type=int, default=8)
    parser.add_argument("--live-rows", type=int, default=16)
    parser.add_argument("--live-seals", type=int, default=8)
    parser.add_argument("--missing-seals", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root)
    os.environ.setdefault("OUROBOROS_DATA_DIR", str(root))
    facts = prepare_root(
        root, generations=args.generations, rows_per_generation=args.rows,
        archived_seals=args.archived_seals, live_rows=args.live_rows,
        live_seals=args.live_seals, missing_seals=args.missing_seals,
        progress_every=args.progress_every,
    )
    print(json.dumps(facts, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

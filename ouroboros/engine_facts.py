"""One engine-and-pool fact: counts and durations, never a verdict.

The row is what the mind reads when it decides whether to schedule, start, or
wait. It reports the primary binding's recent attempts, the newest stored
engine observation, the gateway's already-logged control latency, and the
queue snapshot's project worker pool. It does not call the daemon, poll, or apply a
threshold. A missing source is named, not replaced with a zero or a judgment.
"""

from __future__ import annotations

import ast
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# The owner's named observation windows, not decision thresholds.
_PRIMARY_WINDOW_SEC = 600
_COMPLETIONS_WINDOW_SEC = 3600
_CONTROL_TAIL_BYTES = 65_536
def engine_pool_line(root: Any = None) -> str:
    """The one row. ``root`` is a data directory; the process data root otherwise."""
    drive = _root(root)
    return "; ".join((
        _primary_clause(drive),
        _engine_clause(drive),
        _control_clause(drive),
        _pool_clause(drive),
    ))


def _root(root: Any) -> Path:
    if root is None or not str(root).strip():
        from ouroboros.config import DATA_DIR
        return Path(DATA_DIR)
    return Path(str(root))


def _seconds(value: float) -> str:
    return f"{value:.3f}s" if value < 10 else f"{value:.1f}s"


def _percentile(samples: list[float], fraction: float) -> Optional[float]:
    if not samples:
        return None
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _stamp(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _primary_clause(root: Path) -> str:
    from ouroboros.model_slots import _main_model
    from ouroboros.usage_store import STORE_REL

    model = str(_main_model() or "").strip()
    path = root / STORE_REL
    if not model or not path.is_file():
        return "primary binding: not observed"
    cutoff = time.time() - _PRIMARY_WINDOW_SEC
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.2)
    except sqlite3.Error:
        return "primary binding: not observed"
    try:
        categories = [row[0] for row in conn.execute(
            "SELECT key FROM summaries WHERE scope='category'") if isinstance(row[0], str) and row[0]]
        if not categories:
            present = conn.execute("SELECT 1 FROM attempts LIMIT 1").fetchone()
            if present:
                return "primary binding: not observed"
            counts, durations = {}, []
        else:
            counts, durations = {}, []
            for category in categories:
                rows = conn.execute(
                    "SELECT state, ts_dispatched, ts_final FROM attempts INDEXED BY attempts_category_time "
                    "WHERE category=? AND ts_last_epoch>=? AND model=?",
                    (category, cutoff, model),
                )
                for state, dispatched, final in rows:
                    name = str(state or "")
                    counts[name] = counts.get(name, 0) + 1
                    if name != "settled":
                        continue
                    started, ended = _stamp(dispatched), _stamp(final)
                    if started is not None and ended is not None and ended >= started:
                        durations.append(ended - started)
    except sqlite3.Error:
        return "primary binding: not observed"
    finally:
        conn.close()
    parts = [f"{counts.get('settled', 0)} settled", f"{counts.get('released', 0)} released"]
    for name in ("reserved", "dispatched", "unresolved"):
        if counts.get(name):
            parts.append(f"{counts[name]} {name}")
    for name in sorted(set(counts) - {"settled", "released", "reserved", "dispatched", "unresolved"}):
        if name:
            parts.append(f"{counts[name]} {name}")
    text = f"primary {model} last 10 min: " + ", ".join(parts)
    median, p90 = _percentile(durations, 0.5), _percentile(durations, 0.9)
    if median is not None and p90 is not None:
        text += f"; dispatch→answer median {_seconds(median)}, p90 {_seconds(p90)}"
    return text


def _engine_clause(root: Path) -> str:
    try:
        from ouroboros.claudexor_exit_facts import saved_facts
        row = saved_facts(root).get("capacity")
    except Exception:
        return "engine: not observed"
    if not isinstance(row, dict):
        return "engine: not observed"
    observed = row.get("ts") if isinstance(row.get("ts"), str) and row.get("ts") else ""
    loop = row.get("loop") if isinstance(row.get("loop"), dict) else None
    if not loop:
        return f"engine: not observed since {observed}" if observed else "engine: not observed"
    parts = []
    utilisation = loop.get("utilization")
    if isinstance(utilisation, (int, float)) and not isinstance(utilisation, bool):
        parts.append(f"loop utilisation {utilisation:.2f}")
    delay = loop.get("delay") if isinstance(loop.get("delay"), dict) else {}
    delay_parts = []
    for key, label in (("p50Ms", "p50"), ("p99Ms", "p99")):
        value = delay.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            delay_parts.append(f"{label} {value:g}ms")
    samples = delay.get("samples")
    if isinstance(samples, int) and not isinstance(samples, bool):
        delay_parts.append(f"samples {samples}")
    if delay_parts:
        parts.append("delay " + " ".join(delay_parts))
    when = observed or (loop.get("windowEndedAt") if isinstance(loop.get("windowEndedAt"), str) else "")
    if when:
        parts.append(f"observed {when}")
    return "engine: " + ", ".join(parts) if parts else (
        f"engine: not observed since {observed}" if observed else "engine: not observed")


def _control_clause(root: Path) -> str:
    path = root / "logs" / "server.log"
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - _CONTROL_TAIL_BYTES))
            tail = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return "control latency: not observed"
    newest = None
    for line in tail.splitlines():
        marker = "timings_ms="
        if "claudexor_passive_quota_read" not in line or marker not in line:
            continue
        raw = line.split(marker, 1)[1].strip()
        try:
            timings = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            continue
        if isinstance(timings, dict):
            newest = (line, timings)
    if newest is None:
        return "control latency: not observed"
    line, timings = newest
    parts = []
    for key, value in timings.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{key} {_seconds(float(value) / 1000)}")
    if not parts:
        return "control latency: not observed"
    observed = line[:23] if len(line) >= 23 and line[4] == "-" else ""
    if observed:
        parts.append(f"observed {observed}")
    return "control latency: " + ", ".join(parts)


def _count(document: dict, key: str) -> Optional[int]:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _pool_clause(root: Path) -> str:
    """The published project worker pool (``queue_publication._pool_counts``): the owner's cap on acting tasks
    apart from the processes that exist, slots holding a task (acting, or waiting on their process), pending rows
    apart from those the launch refuses by their own facts, the conversation lane apart, then the supervisor's
    last dated reading of the machine's memory (``_memory_clause``)."""
    try:
        from ouroboros.queue_observation import load_queue_document, pool_observation_status, queue_stamp
        document = load_queue_document(root)
    except Exception:
        document = {"_snapshot_missing": True}
    if not isinstance(document, dict) or document.get("_snapshot_missing") or document.get("_snapshot_invalid"):
        return "pool: not observed; " + _completions_clause(root)
    certified = str(queue_stamp(document).get("ts") or "")
    published = str(document.get("ts") or "")
    when = (f" as of {certified}" if certified else
            f" as published {published}, uncertified" if published else "")
    if pool_observation_status(document) == "unknown":
        return f"project worker pool{when}: unknown (introspection failed); " + _completions_clause(root)

    def said(label: str, key: str) -> str:
        value = _count(document, key)
        return f"{label} {value}" if value is not None else f"{label} not observed"

    pending = _count(document, "pending_count")
    if pending is None and isinstance(document.get("pending"), list):
        pending = len(document["pending"])
    cap = ("owner cap none (the machine's memory is the limit)" if "max_workers" in document
           and document["max_workers"] is None else said("owner cap on acting tasks", "max_workers"))
    parts = [cap, said("total", "worker_total"), said("with a task", "busy_worker_count"),
             said("acting", "acting_worker_count"), said("waiting on their process", "parked_worker_count"),
             said("booting", "booting_count"), said("reaping (incl. booting)", "reaping_count"),
             said("assignable idle", "assignable_idle_workers")]
    parts.append(f"pending rows {pending}" if pending is not None else "pending rows not observed")
    carriers = _count(document, "lane_carrier_total")
    lane = (f"conversation carriers {carriers}, outside the cap and memory admission" if carriers is not None
            else "conversation carriers not observed")
    return (f"project worker pool{when}: " + ", ".join(parts) + "; " + lane + "; "
            + _memory_clause(document.get("memory_admission")) + "; " + _completions_clause(root))


def _memory_clause(reading: Any) -> str:
    """The supervisor's last reading of the machine for new project processes (``pool_admission.facts``)."""
    if not isinstance(reading, dict) or not reading.get("read_at"):
        return "memory admission: not read yet"

    def size(key: str) -> str:
        value = reading.get(key)
        return f"{value / 2 ** 30:.1f} GiB" if isinstance(value, int) and not isinstance(value, bool) else "unknown"

    typical = reading.get("typical_worker_bytes")
    worker = (f"typical worker {typical // 2 ** 20} MiB ({reading.get('typical_basis')})"
              if isinstance(typical, int) and not isinstance(typical, bool) else "typical worker size unknown")
    return (f"memory admission read {reading['read_at']}: available {size('available_bytes')} of "
            f"{size('total_bytes')}, reserve {size('reserve_bytes')}, kernel pressure {reading.get('pressure')}, "
            f"{worker}, starting {reading.get('starting_processes')}, new processes admitted "
            f"{reading.get('starts_admitted')} ({reading.get('reason')})")


def _completions_clause(root: Path) -> str:
    try:
        from ouroboros import store
        state, _tier = store.probe(root)
        if state != "present":
            return "completions last hour: not observed"
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=_COMPLETIONS_WINDOW_SEC)).isoformat()
        with store.open_store(root, write=False) as txn:
            row = txn.conn.execute(
                "SELECT COUNT(*) FROM tasks INDEXED BY tasks_updated "
                "WHERE updated_at>=? AND status IN ('completed', 'failed', 'cancelled', 'infra_failed')",
                (cutoff,),
            ).fetchone()
        return f"completions last hour {int(row[0])}"
    except Exception:
        return "completions last hour: not observed"

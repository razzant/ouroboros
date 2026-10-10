"""F1 on the REAL server: readiness and the gateway do not wait for history.

This boots `server.py` as a real process against an isolated data root that
carries retained usage history (archive segments the retired compactor left,
and the journal the boot imports into the usage store), then watches the
ordering from outside: readiness is served, a trivial gateway request is
answered, and the history audit is still running in another process.

Gated behind `OUROBOROS_RUN_UI_SMOKE=1` like every other real-server case here.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import threading
import time
import urllib.request

import pytest

from tests import _f1_archive_fixture_shared as f1_archive_fixture
from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data

direct_server_with_data = _direct_server_with_data

GENERATIONS = int(os.environ.get("OUROBOROS_1195_AUDIT_GENERATIONS", "300"))
ROWS = int(os.environ.get("OUROBOROS_1195_AUDIT_ROWS", "1000"))


def _audit_records(data_dir: pathlib.Path, since_bytes: int = 0) -> list[dict]:
    """Records appended AFTER `since_bytes`.

    The fixture boots once before the archived fixture exists, and that first
    generation runs its own audit over an empty root. Reading the whole file
    would report that generation's result as this one's.
    """
    path = data_dir / "logs" / "supervisor.jsonl"
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(since_bytes)
        blob = handle.read()
    rows = []
    for line in blob.decode("utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") == "startup_historical_audit":
            rows.append(row)
    return rows


def _terminal(records: list[dict]) -> dict | None:
    return next((row for row in records
                 if row.get("phase") in {"completed", "unknown", "failed", "stopped"}), None)


def _get(url: str, path: str) -> tuple[int, float, dict]:
    started = time.monotonic()
    with urllib.request.urlopen(f"{url}{path}", timeout=10) as response:  # noqa: S310 - local test server
        body = response.read()
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = {}
        return response.status, time.monotonic() - started, payload


def _epoch(text: str) -> float:
    return _dt.datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()


class _BootSampler(threading.Thread):
    """Sample readiness and a trivial request THROUGHOUT the boot, from outside.

    The audit starts after readiness and is finite, so a sampler that only
    begins once `start_server()` returns can miss the whole window. This one
    runs across the boot and timestamps every observation on the same wall
    clock the audit records use.
    """

    def __init__(self, url: str) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.samples: list[dict] = []
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            sample = {"at": time.time()}
            try:
                health_status, health_seconds, _ = _get(self.url, "/api/health")
                sample.update(health_status=health_status, health_seconds=health_seconds)
                state_status, state_seconds, payload = _get(self.url, "/api/state")
                sample.update(state_status=state_status, state_seconds=state_seconds,
                              supervisor_ready=payload.get("supervisor_ready") is True)
            except Exception as exc:
                sample.update(error=type(exc).__name__)
            self.samples.append(sample)
            self.stop.wait(0.05)


@pytest.mark.ui_browser
def test_real_server_serves_readiness_and_requests_while_history_runs_elsewhere(
    direct_server_with_data,
):
    url = direct_server_with_data["url"]
    data_dir: pathlib.Path = direct_server_with_data["data_dir"]

    direct_server_with_data["stop_server"]()   # returns only after a PROVEN container reap
    # The first generation's own children may have been reaped between creating
    # the money name lock file and writing its owner stamp. Such a stampless lock
    # has no pid to prove dead, so the owner-aware acquirer would wait out its age
    # grace and the next boot's journal import would time out. Every process of
    # THIS data root is proven gone by the reap above, so an EMPTY lock here is an
    # orphan by proof.
    from ouroboros.usage_ledger import LOCK_REL

    orphan_lock = data_dir / LOCK_REL
    if orphan_lock.exists() and orphan_lock.stat().st_size == 0:
        orphan_lock.unlink()
    fixture = f1_archive_fixture
    os.environ["OUROBOROS_DATA_DIR"] = str(data_dir)
    build_started = time.monotonic()
    facts = fixture.prepare_root(
        data_dir, generations=GENERATIONS, rows_per_generation=ROWS,
        archived_seals=120, live_rows=40, live_seals=20, missing_seals=4,
    )
    build_seconds = time.monotonic() - build_started
    assert facts["segments_on_disk"] == GENERATIONS, facts
    # What the audit must check is what is actually on disk, counted here.
    expected_manifests = len(list((data_dir / "observability" / "calls").glob("*/*.json")))
    assert expected_manifests == sum(facts["seal_manifests"].values()), facts

    from ouroboros import usage_ledger as ul
    from ouroboros import usage_store

    journal_before = (data_dir / ul.LEDGER_REL).read_bytes()
    archive_before = sorted((path.name, path.stat().st_size)
                            for path in (data_dir / ul.ARCHIVE_SEGMENT_DIR_REL).glob("*.jsonl"))
    supervisor_log = data_dir / "logs" / "supervisor.jsonl"
    before_bytes = supervisor_log.stat().st_size if supervisor_log.exists() else 0

    sampler = _BootSampler(url)
    from ouroboros.startup_historical_audit import HistoricalAudit
    explicit = HistoricalAudit()
    sampler.start()
    try:
        direct_server_with_data["start_server"]()   # returns only once readiness is served
        assert not _audit_records(data_dir, before_bytes), "boot must not launch a historical audit"
        explicit.start(data_dir, pathlib.Path(__file__).resolve().parents[1])
        deadline = time.monotonic() + 120
        terminal = None
        while time.monotonic() < deadline:
            terminal = _terminal(_audit_records(data_dir, before_bytes))
            if terminal is not None:
                break
            time.sleep(0.05)
    finally:
        explicit.stop()
        sampler.stop.set()
        sampler.join(timeout=10)

    records = _audit_records(data_dir, before_bytes)
    started = next((row for row in records if row.get("phase") == "started"), None)
    assert started is not None, f"no audit launch after readiness: {records}"
    assert supervisor_log.stat().st_size > before_bytes
    assert terminal is not None, f"the audit never reached a terminal record: {records}"
    assert terminal["phase"] == "completed", terminal
    assert terminal["exit_code"] == 0
    assert terminal["manifests_checked"] == expected_manifests, (terminal, expected_manifests)
    assert started["pid"] != os.getpid()

    started_at = _epoch(started["ts"])
    terminal_at = _epoch(terminal["ts"])
    assert terminal_at >= started_at

    ready_samples = [row for row in sampler.samples if row.get("supervisor_ready")]
    assert ready_samples, f"readiness was never observed: {sampler.samples[-3:]}"
    first_ready_at = ready_samples[0]["at"]
    # THE ORDERING CLAIM: readiness was already being served before the history
    # pass finished, and it was not waiting for it.
    assert first_ready_at < terminal_at, (first_ready_at, started_at, terminal_at)

    in_window = [row for row in ready_samples if started_at <= row["at"] <= terminal_at]
    assert in_window, (
        "no request landed between the audit launch and its terminal record; "
        "raise OUROBOROS_1195_AUDIT_GENERATIONS to widen the window"
    )
    assert all(row.get("health_status") == 200 and row.get("state_status") == 200
               for row in in_window), in_window[:3]

    # It ran in a different process, recorded in durable custody as such.
    from ouroboros.process_custody import ledger_path

    custody = ledger_path(data_dir)
    assert custody.exists(), f"no durable custody ledger at {custody}"
    entries = []
    for line in custody.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    audits = [row for row in entries if row.get("purpose") == "startup_historical_audit"]
    assert audits, "the audit child never entered durable custody"
    mine = [row for row in audits if int(row.get("pid") or 0) == int(started["pid"])]
    assert mine, (started["pid"], audits)
    assert all(row.get("scope") == "session" for row in mine), mine
    custody_checked = True

    # The boot imported the journal unchanged; the retained evidence the pass
    # read is byte-identical afterwards.
    assert (data_dir / usage_store.LEDGER_REL).read_bytes() == journal_before
    assert sorted((path.name, path.stat().st_size)
                  for path in (data_dir / ul.ARCHIVE_SEGMENT_DIR_REL).glob("*.jsonl")) == archive_before

    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "f1-real-server-ordering.json").write_text(json.dumps({
        "fixture": {key: facts[key] for key in
                    ("generations", "rows_per_generation", "segments_on_disk",
                     "archive_bytes", "archived_attempt_ids_expected", "seal_manifests")},
        "fixture_build_seconds": build_seconds,
        "readiness_first_observed_before_audit_terminal": True,
        "seconds_readiness_before_audit_terminal": terminal_at - first_ready_at,
        "audit_wall_seconds": terminal_at - started_at,
        "requests_answered_inside_the_audit_window": len(in_window),
        "max_health_seconds_inside_window": max(row["health_seconds"] for row in in_window),
        "max_state_seconds_inside_window": max(row["state_seconds"] for row in in_window),
        "seal_manifests_on_disk": expected_manifests,
        "audit_started_record": started,
        "audit_terminal_record": terminal,
        "durable_custody_checked": custody_checked,
        "ledger_unchanged": True,
        "note": "ordering and responsiveness evidence on one machine; not a latency SLA",
    }, indent=1), encoding="utf-8")

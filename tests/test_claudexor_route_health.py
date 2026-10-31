"""Route-health facts: the catalog-negotiated read, the dated supervisor row and the newest row's own
file written on change, the custody sweep's attach-only refresh, and the one line the Runtime vitals
and deep self-review share, which outlives the journal's tail.
No threshold, alarm or advice anywhere: counts the mind weighs (BIBLE P5)."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_exit_facts as facts, jsonl_tail
from ouroboros.context_runtime_facts import vitals_fact, vitals_line
from ouroboros.gateways import claudexor as transport
from ouroboros.gateways.claudexor_route_health import ROUTE_HEALTH_PATH, read_route_health, route_health_facts
from ouroboros.utils import file_identity

REPO = pathlib.Path(__file__).resolve().parent.parent

CATALOG = [{"id": "get:daemon-route-health", "method": "GET", "path": ROUTE_HEALTH_PATH, "parameters": []}]
OLD_CATALOG = [{"id": "get:daemon-status", "method": "GET", "path": "/v2/daemon/status", "parameters": []}]
OBSERVED = "2026-10-28T12:30:00+00:00"


def route(harness, access, runs, failed, *groups, **counts):
    succeeded = runs - failed if type(runs) is int and type(failed) is int else 0
    return {"harnessId": harness, "access": access, "runs": runs, "failed": failed,
            "succeeded": counts.get("succeeded", succeeded), "cancelled": counts.get("cancelled", 0),
            "other": counts.get("other", 0), "topFailures": list(groups)}


def group(category, message, count, code=None):
    return {"category": category, "code": code, "message": message, "count": count}


def body(**overrides):
    base = {
        "windowDays": 7, "from": "2026-10-21T12:00:00.000Z", "generatedAt": "2026-10-28T12:00:00.000Z",
        "scannedRuns": 621, "complete": True, "incompleteReason": None,
        "routes": [
            route("agy", "readonly", 123, 64, group("harness_error", "agy headless tool permission refused: read_file", 38),
                  group("harness_error", "agy headless tool permission refused: grep", 20), cancelled=2),
            route("claude", "readonly", 410, 4, group("quota", "5h window spent", 4, code="subscription_window_exhausted")),
            route("codex", "full", 88, 0),
        ],
    }
    return {**base, **overrides}


class Gateway:
    engine_version = "4.8.0"

    def __init__(self, catalog=CATALOG, answer=None, error=None):
        self.catalog, self.answer, self.error, self.requests, self.closed = catalog, answer, error, [], 0

    def operations(self):
        if isinstance(self.catalog, Exception):
            raise self.catalog
        return self.catalog

    def _request(self, method, path, **kwargs):
        self.requests.append((method, path))
        if self.error is not None:
            raise self.error
        return self.answer

    def close(self):
        self.closed += 1


def route_rows(root):
    path = root / "logs/supervisor.jsonl"
    if not path.exists():
        return []
    return [row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
            if row["type"] == "claudexor_route_health"]


def written(root):
    """The newest row's file: its bytes and identity (an atomic replace is a new identity)."""
    state = root / facts.ROUTE_HEALTH_FILE
    return state.read_bytes(), file_identity(state.stat())


@pytest.fixture(autouse=True)
def fresh_memos(monkeypatch):
    monkeypatch.setattr(jsonl_tail, "_TAILS", {})
    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)


# ---- the reader -------------------------------------------------------------------------------------


def test_reader_asks_only_an_engine_whose_catalog_lists_the_route():
    listed = Gateway(answer=body())
    read = read_route_health(listed)
    assert listed.requests == [("GET", ROUTE_HEALTH_PATH)]
    assert read["window_days"] == 7 and read["scanned_runs"] == 621 and read["complete"] is True
    assert read["generated_at"] == "2026-10-28T12:00:00.000Z" and read["incomplete_reason"] is None
    assert [(r["harness"], r["access"], r["runs"], r["failed"]) for r in read["routes"]] == [
        ("agy", "readonly", 123, 64), ("claude", "readonly", 410, 4), ("codex", "full", 88, 0)]
    assert read["routes"][0]["top_failures"][0] == {
        "category": "harness_error", "code": None, "message": "agy headless tool permission refused: read_file", "count": 38}
    assert read["routes"][1]["top_failures"][0]["code"] == "subscription_window_exhausted"
    assert read["routes"][2]["top_failures"] == [] and read["routes"][0]["cancelled"] == 2

    older = Gateway(catalog=OLD_CATALOG, answer=body())
    assert read_route_health(older) is None and older.requests == []  # not asked, not an empty count


def test_reader_treats_refusals_transport_failures_and_malformed_bodies_as_unavailable():
    for error in (transport.ClaudexorUnavailable("not_found", "old engine", status_code=404),
                  transport.ClaudexorUnavailable("daemon_busy", "later", status_code=503),
                  RuntimeError("socket closed")):
        assert read_route_health(Gateway(error=error)) is None  # never raises into its caller
    assert read_route_health(Gateway(catalog=RuntimeError("catalog unreadable"))) is None
    for malformed in ([], {"windowDays": 7}, {"routes": "x"},
                      body(routes=[route("agy", "readonly", "123", 64)]),
                      body(routes=[route("agy", "readonly", 123, True)]),
                      body(routes=[dict(route("agy", "readonly", 123, 1), topFailures="n/a")]),
                      body(routes=[route("agy", "readonly", 123, 1, {"category": "x", "message": "m", "count": -1})]),
                      body(routes=[route("agy", "readonly", 123, 1), "junk"])):
        assert read_route_health(Gateway(answer=malformed)) is None, malformed


def test_normalization_keeps_unknown_values_unknown_and_bounds_what_the_contract_bounds():
    loose = body(windowDays=True, complete="yes", scannedRuns=-1, routes=[
        dict(route("agy", "readonly", 10, 1, *[group("c", "m" * 200, n) for n in (5, 4, 3, 2)]),
             succeeded=None, cancelled="2", other=None, harnessId=None, access=7)])
    read = route_health_facts(loose)
    assert read["window_days"] is None and read["complete"] is None and read["scanned_runs"] is None
    row = read["routes"][0]
    assert (row["harness"], row["access"]) == ("unknown", "unknown")
    assert row["succeeded"] is None and row["cancelled"] is None and row["other"] is None  # unknown, never zero
    assert [g["count"] for g in row["top_failures"]] == [5, 4, 3] and len(row["top_failures"][0]["message"]) == 160


# ---- the supervisor row -----------------------------------------------------------------------------


def test_row_is_written_on_change_and_never_rewritten_while_the_facts_hold(tmp_path):
    read = route_health_facts(body())
    assert facts.record_route_health(tmp_path, read, engine_version="4.8.0") is True
    (row,) = route_rows(tmp_path)
    assert row["ts"] and row["engine_version"] == "4.8.0" and row["digest"] == facts.route_health_digest(read)
    assert row["routes"] == read["routes"] and row["window_days"] == 7 and row["complete"] is True
    assert facts.latest_route_health(tmp_path) == row  # the newest row, whole, in its own file
    kept = written(tmp_path)
    assert facts.record_route_health(tmp_path, read, engine_version="4.8.0") is False
    later = route_health_facts(body(generatedAt="2026-10-28T12:10:00.000Z", scannedRuns=700))
    assert facts.record_route_health(tmp_path, later, engine_version="4.8.0") is False  # clocks are not a change
    assert len(route_rows(tmp_path)) == 1 and written(tmp_path) == kept  # neither appended nor replaced

    changed = body()
    changed["routes"][1]["failed"] = 5
    assert facts.record_route_health(tmp_path, route_health_facts(changed), engine_version="4.8.0") is True
    first, second = route_rows(tmp_path)
    assert first["digest"] != second["digest"] and second["routes"][1]["failed"] == 5
    assert facts.latest_route_health(tmp_path) == second and written(tmp_path)[1] != kept[1]  # replaced
    incomplete = route_health_facts(body(complete=False, incompleteReason="scan_bound"))
    assert facts.record_route_health(tmp_path, incomplete, engine_version="4.8.0") is True  # completeness is a fact
    assert len(route_rows(tmp_path)) == 3


def test_a_damaged_newest_row_is_no_fact_and_the_next_refresh_writes_it_again(tmp_path):
    read = route_health_facts(body())
    facts.record_route_health(tmp_path, read)
    (tmp_path / facts.ROUTE_HEALTH_FILE).write_text("{torn", encoding="utf-8")
    assert facts.latest_route_health(tmp_path) is None
    assert facts.record_route_health(tmp_path, read) is True and len(route_rows(tmp_path)) == 2
    assert facts.latest_route_health(tmp_path)["digest"] == facts.route_health_digest(read)


def test_malformed_facts_write_no_row(tmp_path):
    for bad in (None, {}, {"routes": None}, "routes"):
        assert facts.record_route_health(tmp_path, bad) is False
    assert not (tmp_path / "logs/supervisor.jsonl").exists() and not (tmp_path / facts.ROUTE_HEALTH_FILE).exists()


# A fresh process: it has none of this process's memory, only the files.
FRESH_PROCESS = """
import json, sys
from ouroboros import claudexor_exit_facts as facts
from ouroboros.context_runtime_facts import vitals_fact
root, later, observed = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
recorded = facts.record_route_health(root, later, engine_version="4.8.0")
print(json.dumps({"recorded": recorded, "route_health": (vitals_fact(root, observed_at=observed) or {}).get("route_health")}))
"""


def test_the_newest_fact_outlives_the_journal_tail_its_rotation_and_the_process(tmp_path):
    facts.record_route_health(tmp_path, route_health_facts(body()), engine_version="4.8.0")
    shown = vitals_fact(tmp_path, observed_at=OBSERVED)["route_health"]
    kept = written(tmp_path)
    journal = tmp_path / "logs/supervisor.jsonl"
    with journal.open("a", encoding="utf-8") as stream:  # busy minutes and other rows, no new facts
        for n in range(400):
            stream.write(json.dumps({"ts": OBSERVED, "type": "supervisor_load", "n": n, "pad": "x" * 800}) + "\n")
    assert journal.stat().st_size > 256_000  # the row left the bounded tail every context reads
    jsonl_tail._TAILS.clear()
    assert vitals_fact(tmp_path, observed_at=OBSERVED)["route_health"] == shown

    later = route_health_facts(body(generatedAt="2026-10-28T12:10:00.000Z", scannedRuns=700))
    child = subprocess.run([sys.executable, "-c", FRESH_PROCESS, str(tmp_path), json.dumps(later), OBSERVED],
                           cwd=REPO, text=True, capture_output=True, timeout=120)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout.splitlines()[-1]) == {"recorded": False, "route_health": shown}
    assert len(route_rows(tmp_path)) == 1 and written(tmp_path) == kept

    journal.rename(tmp_path / "logs/supervisor_rotated.jsonl")  # rotation moves the history aside
    jsonl_tail._TAILS.clear()
    assert vitals_fact(tmp_path, observed_at=OBSERVED)["route_health"] == shown
    assert facts.record_route_health(tmp_path, later) is False and route_rows(tmp_path) == []
    assert written(tmp_path) == kept


# ---- the refresh and its trigger --------------------------------------------------------------------


def test_refresh_records_through_the_gateway_it_opened_and_closes_it(tmp_path, monkeypatch):
    from ouroboros import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    gateway = Gateway(answer=body())
    read = facts.refresh_route_health(lambda: gateway)
    assert read["routes"][0]["failed"] == 64 and gateway.closed == 1
    (row,) = route_rows(tmp_path)
    assert row["engine_version"] == "4.8.0"
    older = Gateway(catalog=OLD_CATALOG)
    assert facts.refresh_route_health(lambda: older) is None and older.closed == 1 and len(route_rows(tmp_path)) == 1


def test_refresh_attaches_read_only_and_never_raises_or_starts_an_engine(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon, config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("a fact is never worth starting an engine")

    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", forbidden)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway",
                        lambda **kw: (_ for _ in ()).throw(transport.ClaudexorUnavailable("daemon_unreachable", "no engine")))
    assert facts.refresh_route_health() is None and route_rows(tmp_path) == []

    def raising():
        raise RuntimeError("factory broke")

    assert facts.refresh_route_health(raising) is None
    attached = Gateway(answer=body())
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **kw: attached)
    assert facts.refresh_route_health()["scanned_runs"] == 621 and attached.closed == 1
    assert len(route_rows(tmp_path)) == 1


@pytest.fixture
def custody_sweep(monkeypatch, tmp_path):
    """The 600 s custody pass with every other step idle; ``calls`` records the steps that ran."""
    from ouroboros import claudexor_daemon, process_custody, process_scope, server_maintenance as sm, terminal_projection

    calls = []
    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(terminal_projection, "reconcile_terminal_projections", lambda *_a: None)
    monkeypatch.setattr(claudexor_daemon, "get_owned_daemon", lambda: SimpleNamespace(
        clear_start_failure_latch=lambda *, cleared_by: False))
    monkeypatch.setattr(process_custody, "reap_orphaned_processes", lambda *a, **k: [])
    monkeypatch.setattr(process_scope, "settle_task_scopes", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    monkeypatch.setattr(sm, "_reconcile_delegated_runs", lambda *a, **k: calls.append("delegated_runs"))
    monkeypatch.setattr(sm, "_cursor_refresh_settled_terminals", lambda *a, **k: None)
    for name in ("_prune_worktrees", "_prune_delegated_snapshots", "_prune_engine_runtimes"):
        monkeypatch.setattr(sm, name, lambda: None)
    monkeypatch.setattr(sm, "_prune_review_checkouts", lambda stop_event=None: None)
    monkeypatch.setattr(facts, "refresh_route_health", lambda *a, **k: calls.append("route_health"))

    def run(stop_event=None):
        latch = threading.Lock()
        latch.acquire()
        sm._run_periodic_custody_sweep(stop_event, latch)
        assert not latch.locked()
        return calls

    return run


def test_the_custody_sweep_refreshes_route_health_after_the_delegated_runs_settled(custody_sweep):
    assert custody_sweep() == ["delegated_runs", "route_health"]


def test_a_closing_generation_skips_the_refresh(custody_sweep):
    stop = threading.Event()
    stop.set()
    assert "route_health" not in custody_sweep(stop)


# ---- the vitals line --------------------------------------------------------------------------------


def test_vitals_render_the_failed_routes_most_failed_first_with_the_top_cause(tmp_path):
    facts.record_route_health(tmp_path, route_health_facts(body()), engine_version="4.8.0")
    fact = vitals_fact(tmp_path, observed_at=OBSERVED)
    assert fact["route_health"] == (
        "12:00: 7-day window, 3 routes, 621 engine runs; failed: agy readonly 64/123 "
        "(harness_error: agy headless tool permission refused: read_file ×38), claude readonly 4/410 "
        "(quota: 5h window spent ×4)")
    line = vitals_line(fact)
    assert "; route health 12:00: 7-day window, 3 routes, 621 engine runs; failed: agy readonly 64/123" in line
    assert line.index("route health") < line.index("runtime python")
    for word in ("WARNING", "CRITICAL", "%", "threshold", "should", "degraded"):
        assert word not in fact["route_health"], word


def test_vitals_bound_the_line_to_a_few_routes_and_a_short_cause_while_the_row_keeps_all(tmp_path):
    many = body(routes=[route(f"h{n}", "readonly", 50, n, group("c", "x" * 120, n)) for n in range(1, 7)])
    facts.record_route_health(tmp_path, route_health_facts(many))
    (row,) = route_rows(tmp_path)
    assert len(row["routes"]) == 6 and len(row["routes"][0]["top_failures"][0]["message"]) == 120
    rendered = vitals_fact(tmp_path, observed_at=OBSERVED)["route_health"]
    assert rendered.endswith(", +2 more routes with failed runs")
    assert "h6 readonly 6/50 (c: " + "x" * 80 + " ×6), h5 readonly 5/50" in rendered and "h2 " not in rendered


def test_vitals_say_none_failed_only_from_a_counted_window_and_name_an_incomplete_scan(tmp_path):
    clean = body(complete=False, incompleteReason="scan_bound",
                 routes=[route(r["harnessId"], r["access"], r["runs"], 0) for r in body()["routes"]])
    facts.record_route_health(tmp_path, route_health_facts(clean))
    fact = vitals_fact(tmp_path, observed_at=OBSERVED)
    assert fact["route_health"] == "12:00: 7-day window, 3 routes, 621 engine runs, none failed; scan incomplete (scan_bound)"


def test_vitals_name_a_counted_window_without_runs_and_its_scan(tmp_path):
    facts.record_route_health(tmp_path, route_health_facts(body(routes=[], scannedRuns=0)))
    assert vitals_fact(tmp_path, observed_at=OBSERVED)["route_health"] == "12:00: 7-day window, no engine runs, scan complete"
    bounded = body(routes=[], scannedRuns=2000, complete=False, incompleteReason="scan_bound")
    facts.record_route_health(tmp_path, route_health_facts(bounded))
    assert vitals_fact(tmp_path, observed_at=OBSERVED)["route_health"] == (
        "12:00: 7-day window, no engine runs; scan incomplete (scan_bound)")
    facts.record_route_health(tmp_path, route_health_facts(body(routes=[], complete="unsaid")))
    assert vitals_fact(tmp_path, observed_at=OBSERVED)["route_health"] == "12:00: 7-day window, no engine runs"


def test_vitals_line_holds_byte_for_byte_while_only_the_clocks_move(tmp_path):
    facts.record_route_health(tmp_path, route_health_facts(body()))
    first = vitals_line(vitals_fact(tmp_path, observed_at=OBSERVED))
    for minute, scanned in (("12:10", 700), ("12:20", 655)):
        later = route_health_facts(body(generatedAt=f"2026-10-28T{minute}:00.000Z", scannedRuns=scanned))
        assert facts.record_route_health(tmp_path, later) is False
        assert vitals_line(vitals_fact(tmp_path, observed_at=OBSERVED)) == first  # the prompt prefix holds
    changed = body(generatedAt="2026-10-28T12:30:00.000Z")
    changed["routes"][0]["failed"] = 63
    facts.record_route_health(tmp_path, route_health_facts(changed))
    line = vitals_line(vitals_fact(tmp_path, observed_at=OBSERVED))
    assert line != first and "12:30: 7-day window" in line and "agy readonly 63/123" in line


def test_vitals_omit_the_key_when_no_engine_answered_or_the_row_is_damaged(tmp_path):
    fact = vitals_fact(tmp_path, observed_at=OBSERVED)  # no row: an older engine, or none attached yet
    assert fact is not None and "route_health" not in fact
    assert "none failed" not in json.dumps(fact) and "no engine runs" not in json.dumps(fact)
    assert "route health" not in vitals_line(fact)
    state = tmp_path / facts.ROUTE_HEALTH_FILE
    state.parent.mkdir(parents=True)
    row = {"ts": OBSERVED, "type": "claudexor_route_health", "digest": "d", "window_days": 7}
    for content in ("{torn", json.dumps([row]), json.dumps({**row, "type": "other", "routes": []}),
                    json.dumps({**row, "routes": None}),
                    json.dumps({**row, "routes": [{"harness": "agy", "access": "readonly", "runs": "x", "failed": 1}]}),
                    json.dumps({**row, "routes": [{"harness": "agy", "access": "readonly", "runs": 3, "failed": 1,
                                                   "top_failures": 5}]}),
                    json.dumps({**row, "routes": [], "pad": "x" * 300_000})):  # past the read bound
        state.write_text(content, encoding="utf-8")
        fact = vitals_fact(tmp_path, observed_at=OBSERVED)
        assert fact is not None and "route_health" not in fact and fact.get("runtime"), content[:60]


def test_vitals_read_the_newest_row_with_its_own_clock(tmp_path):
    facts.record_route_health(tmp_path, route_health_facts(body()))
    newer = body(generatedAt="2026-10-27T23:50:00.000Z")
    newer["routes"][0]["failed"] = 65
    facts.record_route_health(tmp_path, route_health_facts(newer))
    rendered = vitals_fact(tmp_path, observed_at=OBSERVED)["route_health"]
    assert rendered.startswith("10-27 23:50: 7-day window") and "agy readonly 65/123" in rendered

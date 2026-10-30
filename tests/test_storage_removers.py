"""A remover's last outcome is durable and read beside the place it serves: a maintenance job that throws
keeps its state and gains ``last_error`` until it returns normally, the engine place carries the engine's
last retention pass (unknown, never zero, when the engine cannot be asked or keeps no record), the
Storage projection and ``query_data(kind="storage")`` read the same facts, and the runtime ``storage``
fact reads the job rows alone (no engine call while a turn is prepared)."""
from __future__ import annotations

import json
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import maintenance, maintenance_jobs, storage, storage_removers
from ouroboros.context_runtime_facts import storage_fact
from ouroboros.gateways import claudexor as gateway_module
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.gateways.claudexor_maintenance import ClaudexorMaintenanceGateway

HELD = {"outcome": "held", "started_at": "2026-10-27T15:00:00Z", "finished_at": "2026-10-27T15:05:00Z",
        "freed_bytes": 2_000_000, "kept": {"active": 1, "recent": 2, "young": 3, "referenced": 4, "actionable": 0,
                                           "unknown_state": 0},
        "held": {"roots": 357, "native_homes": 2},
        "holds": [{"class": "missing_partition", "count": 357, "example": "/p", "since": "2026-10-18T09:00:00Z"},
                  {"class": "native_home_veto", "count": 2, "example": "/h", "since": "2026-10-20T09:00:00Z"}],
        "errors": {"count": 0, "first": None}}


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    return root


def _engine(data, monkeypatch, answer, *, hung=None):
    """A provisioned engine home whose read answers ANSWER (a record, or an exception raised by the read),
    after HUNG (an event) is set when given."""
    descriptor = data / storage.ENGINE_DIR / "daemon" / "control-api.json"
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    descriptor.write_text("{}", encoding="utf-8")
    asked = []

    class Gateway:
        def __init__(self, endpoint):
            asked.append(endpoint)
            self.closed = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.close()

        def close(self):
            self.closed.set()

        def maintenance_gc_last(self, *, timeout_sec=None):
            asked.append(timeout_sec)
            while hung is not None and not hung.wait(0.01):
                if self.closed.is_set():
                    raise ClaudexorUnavailable("daemon_unreachable", "client closed")
            if isinstance(answer, Exception):
                raise answer
            return {"last": answer}

    monkeypatch.setattr(gateway_module, "discover_daemon_at", lambda home: ("endpoint", pathlib.Path(home)))
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", Gateway)
    return asked


def _readers():
    return [thread for thread in threading.enumerate() if thread.name.startswith("storage-engine-read")]


def _until_no_readers(limit: float = 5.0):
    deadline = time.monotonic() + limit
    while _readers() and time.monotonic() < deadline:
        time.sleep(0.05)
    return _readers()


@pytest.fixture
def stand_in(data):
    """A real engine on loopback behind a real descriptor; ``mode["kind"]``: ``slow`` answers after 0.5 s,
    ``trickle`` sends its body one byte every 50 ms."""
    import http.server

    mode, stop, requests = {"kind": "slow"}, threading.Event(), []
    body = json.dumps({"last": HELD}).encode() + b" " * 2048

    class Engine(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            try:
                if mode["kind"] == "slow" and stop.wait(0.5):
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                for start in range(0, len(body), 1 if mode["kind"] == "trickle" else len(body)):
                    if mode["kind"] == "trickle" and stop.wait(0.05):
                        return
                    self.wfile.write(body[start:start + (1 if mode["kind"] == "trickle" else len(body))])
                    self.wfile.flush()
            except OSError:
                pass  # the reader went away

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    server.daemon_threads = True
    serving = threading.Thread(target=server.serve_forever, name="engine-stand-in", daemon=True)
    serving.start()
    home = data / storage.ENGINE_DIR
    (home / "daemon").mkdir(parents=True)
    (home / "token").write_text("stand-in", encoding="utf-8")
    (home / "daemon" / "control-api.json").write_text(json.dumps(
        {"host": "127.0.0.1", "port": server.server_address[1], "tokenPath": str(home / "token")}), encoding="utf-8")
    try:
        yield SimpleNamespace(mode=mode, requests=requests)
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        serving.join(5)
        _until_no_readers()


def test_a_throwing_job_keeps_its_state_gains_last_error_and_a_normal_return_clears_it(data, monkeypatch):
    from ouroboros import store

    outcomes = iter([RuntimeError("disk said no"), store.StoreBusy("held"), None])

    def run(root, should_stop):
        outcome = next(outcomes)
        if outcome is not None:
            raise outcome
        return {"state": "done", "items": 0}

    monkeypatch.setattr(maintenance, "JOBS", {"custody_trash": (lambda *a: True, run)})
    maintenance_jobs.record(data, "custody_trash", state="pending", stuck={"x": [1]})
    assert maintenance.run(data)["status"] == "failed"
    row = maintenance_jobs.job(data, "custody_trash")
    assert row["state"] == "pending" and row["stuck"] == {"x": [1]}, "the state a due rule reads is untouched"
    assert row["last_error"]["class"] == "RuntimeError" and row["last_error"]["at"]
    [fact] = storage_removers.place_removers(data)["work_folders"]
    assert fact["state"] == "pending" and fact["last_error"]["class"] == "RuntimeError"
    assert storage_removers.not_ok({"work_folders": [fact]}) == [
        f"custody_trash: last run threw RuntimeError at {row['last_error']['at']}"]
    assert maintenance.run(data)["status"] == "completed", "contention is no failure"
    assert maintenance_jobs.job(data, "custody_trash")["last_error"]["class"] == "RuntimeError", "nor a normal return"
    maintenance.run(data)
    assert "last_error" not in maintenance_jobs.job(data, "custody_trash"), "the next normal return clears it"


def test_a_first_run_that_throws_leaves_a_never_finished_job_due(monkeypatch, tmp_path):
    from ouroboros import subagent_metrics

    monkeypatch.setattr(subagent_metrics, "_source_stamp", lambda root: ["events.jsonl", 1])
    assert subagent_metrics.job_due(tmp_path, {}, {}, False)
    assert subagent_metrics.job_due(tmp_path, {"last_error": {"class": "OSError", "at": "now"}}, {}, False)


def test_the_engine_place_carries_the_engines_last_pass(data, monkeypatch):
    asked = _engine(data, monkeypatch, HELD)
    [engine] = storage_removers.place_removers(data)["engine"]
    assert engine == {"id": "engine", "state": "held", "at": "2026-10-27T15:05:00Z", "freed_bytes": 2_000_000,
                      "held": 359, "held_since": "2026-10-18T09:00:00Z", "errors": 0, "first_error": None}
    assert asked[-1] == storage_removers.ENGINE_REUSE_SEC, "no phase bound cuts the read before its own"
    assert storage_removers.not_ok({"engine": [engine]}) == [
        "engine retention: last pass held at 2026-10-27T15:05:00Z; holding 359 folders since 2026-10-18T09:00:00Z"]
    _engine(data, monkeypatch, {**HELD, "outcome": "completed", "held": {"roots": 0, "native_homes": 0}, "holds": []})
    assert storage_removers.not_ok(storage_removers.place_removers(data)) == [], "a clean pass is silent"


@pytest.mark.parametrize("answer,state", [
    (None, "none"),
    (ClaudexorUnavailable("maintenance_unavailable", "old engine", status_code=404), "unsupported"),
    (ClaudexorUnavailable("daemon_unreachable", "down"), "unavailable"),
    ({"outcome": "completed"}, "unavailable"),  # a record of another shape is no record
])
def test_an_engine_that_cannot_answer_is_unknown_never_zero(data, monkeypatch, answer, state):
    _engine(data, monkeypatch, answer)
    [engine] = storage_removers.place_removers(data)["engine"]
    assert engine == {"id": "engine", "state": state, "at": None}, "no freed or held count stands for unknown"
    assert storage_removers.not_ok({"engine": [engine]}) == [f"engine retention: {state}"]


def test_a_hung_engine_is_unavailable_within_the_bound_and_a_poll_reuses_a_recent_read(data, monkeypatch):
    monkeypatch.setattr(storage_removers, "ENGINE_READ_SEC", 0.2)
    hung = threading.Event()
    asked = _engine(data, monkeypatch, HELD, hung=hung)
    try:
        [engine] = storage_removers.place_removers(data)["engine"]
        assert engine == {"id": "engine", "state": "unavailable", "at": None} and not hung.is_set(), \
            "answered while the read still hangs: connect and read together stay within the bound"
    finally:
        hung.set()
    assert _until_no_readers() == []
    now = [time.monotonic() + 3600]  # past every record read so far
    monkeypatch.setattr(storage_removers, "time", SimpleNamespace(monotonic=lambda: now[0]))
    first = storage_removers.place_removers(data, recent=True)["engine"]
    assert first[0]["state"] == "held"
    reads = len(asked)
    now[0] += storage_removers.ENGINE_REUSE_SEC - 1
    assert storage_removers.place_removers(data, recent=True)["engine"] == first and len(asked) == reads, \
        "a projection polled every second reuses the record read seconds ago"
    assert storage_removers.place_removers(data)["engine"] == first and len(asked) > reads, "query_data asks anew"
    reads = len(asked)
    now[0] += storage_removers.ENGINE_REUSE_SEC
    storage_removers.place_removers(data, recent=True)
    assert len(asked) > reads, "an older record is read again"


def test_a_slow_engines_record_arriving_after_the_bound_is_shown_by_the_next_poll(data, monkeypatch, stand_in):
    """Past the answer bound the read goes on until its own: a record arriving then fills the reuse slot,
    so the next poll shows it instead of ``unavailable``."""
    monkeypatch.setattr(storage_removers, "ENGINE_READ_SEC", 0.2)
    monkeypatch.setattr(storage_removers, "ENGINE_REUSE_SEC", 3.0)
    assert storage_removers.place_removers(data, recent=True)["engine"][0]["state"] == "unavailable"
    assert _until_no_readers() == []
    [engine] = storage_removers.place_removers(data, recent=True)["engine"]
    assert engine["state"] == "held" and stand_in.requests == ["/v2/maintenance/gc"], "shown, not read again"


def test_a_trickling_engine_leaves_no_reader_thread(data, monkeypatch, stand_in):
    """Bytes arriving faster than any phase bound keep a read alive; at the read's own bound the waiting
    side closes its client, the engine's connection is torn down and the thread ends."""
    monkeypatch.setattr(storage_removers, "ENGINE_READ_SEC", 0.2)
    monkeypatch.setattr(storage_removers, "ENGINE_REUSE_SEC", 0.5)
    stand_in.mode["kind"] = "trickle"
    [engine] = storage_removers.place_removers(data)["engine"]
    assert engine == {"id": "engine", "state": "unavailable", "at": None}
    assert _until_no_readers() == [], "no reader outlives its bound against a trickling engine"


def test_no_provisioned_engine_has_no_engine_remover(data):
    assert "engine" not in storage_removers.place_removers(data)


def test_the_gateway_reads_the_record_and_types_an_engine_without_the_route(monkeypatch):
    calls = []

    class Gateway(ClaudexorMaintenanceGateway):
        def __init__(self, answer):
            self.answer = answer

        def _request(self, method, path, **kwargs):
            calls.append((method, path, kwargs.get("timeout_sec")))
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

    assert Gateway({"last": HELD}).maintenance_gc_last(timeout_sec=2.0) == {"last": HELD}
    assert calls == [("GET", "/v2/maintenance/gc", 2.0)], "a read, never the POST that starts a pass"
    for status in (404, 405):
        with pytest.raises(ClaudexorUnavailable) as raised:
            Gateway(ClaudexorUnavailable(f"http_{status}", "no route", status_code=status)).maintenance_gc_last()
        assert raised.value.code == "maintenance_unavailable"
    with pytest.raises(ClaudexorUnavailable) as raised:
        Gateway(ClaudexorUnavailable("daemon_unreachable", "down")).maintenance_gc_last()
    assert raised.value.code == "daemon_unreachable", "a down daemon stays unavailable, not unsupported"
    with pytest.raises(ClaudexorUnavailable):
        Gateway({"something": 1}).maintenance_gc_last()


def test_the_runtime_fact_reads_job_rows_alone_and_never_asks_the_engine(data, monkeypatch):
    assert storage_fact(data) is None, "nothing counted, every remover clean"
    maintenance_jobs.record(data, "runtime_retirement", state="done", last_error={"class": "OSError", "at": "T"})
    assert storage_fact(data) == {"removers": "runtime_retirement: last run threw OSError at T"}
    asked = _engine(data, monkeypatch, ClaudexorUnavailable("daemon_unreachable", "down"))
    assert storage_fact(data) == {"removers": "runtime_retirement: last run threw OSError at T"} and asked == [], \
        "no engine call while a turn is prepared"
    (data / "observability").mkdir()
    table = [{"id": "model_calls", "group": "records", "roots": [(str(data), ("observability",))]}]
    storage.save_count(data, storage.measure(data, places=table, sleep=lambda _s: None))
    fact = storage_fact(data)
    assert "counted_at" in fact and fact["removers"] == "runtime_retirement: last run threw OSError at T" and not asked
    assert storage.count.projection(data)["removers"]["engine"] == [{"id": "engine", "state": "unavailable", "at": None}]
    assert asked, "the Storage page still reads the engine's record"


def test_the_projection_and_query_data_read_the_same_removers(data, monkeypatch):
    _engine(data, monkeypatch, HELD)
    maintenance_jobs.record(data, "record_compression", state="done", totals={
        "evidence": {"compressed": 12, "saved_bytes": 4096}, "seats": {"compressed": 3, "saved_bytes": None}})
    projected = storage.count.projection(data)["removers"]
    assert projected == storage_removers.place_removers(data)
    assert set(projected) <= {place_id for place_id, _group, _spec in storage.PLACES}
    [_packing, evidence, retirement] = projected["task_results"]
    assert retirement["id"] == "runtime_retirement", "Task results names the job that retires call journals"
    [_retirement, seats, _trash] = projected["app_state"]
    assert evidence == {"id": "record_compression", "state": "done", "at": evidence["at"],
                        "compressed": 12, "saved_bytes": 4096}, "Task results shows the evidence plane"
    assert seats == {"id": "record_compression", "state": "done", "at": seats["at"],
                     "compressed": 3, "saved_bytes": None}, "App state shows the seat homes, never a sum"

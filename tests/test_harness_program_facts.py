"""Vendor program facts are event observations; context never acquires an engine."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from ouroboros import harness_program_facts as facts

pytestmark = pytest.mark.program_observations


def inventory(date="2026-11-10", *, version="2.1.319", verdict="yes", newest="2.1.321"):
    at = date + "T12:00:00Z"
    return {"observedAt": at, "harnesses": [{"harness": "future-program", "observedAt": at,
        "selection": {"path": "/vendor/program", "version": version}, "updateAvailable": verdict,
        "newest": {"status": "available", "version": newest, "observedAt": "2026-11-10T11:00:00Z"},
        "operation": {"id": "private-receipt", "before": {"version": "old"}}}]}


def saved(root):
    return json.loads((root / facts.OBSERVATION_FILE).read_text(encoding="utf-8"))


@pytest.mark.parametrize("content", [None, "bad json", '{"programs": []}', '{"programs": {"x": null}}'])
def test_nothing_read_or_unreadable_is_explicitly_unobserved(tmp_path, content):
    if content is not None:
        path = tmp_path / facts.OBSERVATION_FILE
        path.parent.mkdir()
        path.write_text(content, encoding="utf-8")
    assert facts.vendor_programs_line(tmp_path) == "Harness programs: not observed."


def test_line_dates_program_observation_and_retained_newest_read(tmp_path):
    first = inventory()
    facts.record_inventory(first, root=tmp_path)
    line = facts.vendor_programs_line(tmp_path)
    assert line == "Harness programs (observed 2026-11-10; newest read 2026-11-10): future-program 2.1.319 (2.1.321 available)."
    later = inventory("2026-12-12")
    facts.record_inventory(later, root=tmp_path)
    assert facts.vendor_programs_line(tmp_path) == line.replace("observed 2026-11-10", "observed 2026-12-12")
    state = saved(tmp_path)
    assert state["completeObservedAt"] == later["observedAt"]
    assert "factsSince" not in state and "completeAt" not in state
    assert state["programs"]["future-program"]["newestObservedAt"] == "2026-11-10T11:00:00Z"
    assert "private-receipt" not in json.dumps(state) and "before" not in json.dumps(state)
    facts.record_inventory(inventory("2026-12-13", version="2.1.321", verdict="no"), root=tmp_path)
    assert facts.vendor_programs_line(tmp_path) == "Harness programs (observed 2026-12-13; newest read 2026-11-10): future-program 2.1.321 (up to date)."


def test_absent_unknown_and_empty_inventories_are_distinct_and_sorted(tmp_path):
    data = inventory(verdict="cannot_tell", version=None)
    data["harnesses"][0]["newest"] = {"status": "unavailable", "observedAt": data["observedAt"]}
    data["harnesses"].append({"harness": "absent", "selection": None, "updateAvailable": "cannot_tell"})
    facts.record_inventory(data, root=tmp_path)
    assert facts.vendor_programs_line(tmp_path).endswith("future-program version unknown (newest unavailable 2026-11-10); not installed: absent.")
    facts.record_inventory({"observedAt": "2026-11-11T00:00:00Z", "harnesses": []}, root=tmp_path)
    assert facts.vendor_programs_line(tmp_path).endswith("none listed.")


def test_partial_reads_merge_and_late_full_reads_cannot_erase_newer_facts(tmp_path):
    facts.record_inventory(inventory(), root=tmp_path)
    newer = inventory("2026-11-12", version="opaque-vendor-build", verdict="cannot_tell")
    newer["harnesses"][0]["harness"] = "another"
    facts.record_inventory(newer, complete=False, root=tmp_path)
    facts.record_inventory(inventory("2026-11-11"), root=tmp_path)
    assert set(saved(tmp_path)["programs"]) == {"future-program", "another"}
    stale = inventory("2026-11-10", version="stale")
    stale["harnesses"][0]["harness"] = "another"
    facts.record_inventory(stale, complete=False, root=tmp_path)
    assert saved(tmp_path)["programs"]["another"]["version"] == "opaque-vendor-build"
    facts.record_inventory(inventory("2026-11-13"), root=tmp_path)
    assert set(saved(tmp_path)["programs"]) == {"future-program"}
    facts.record_inventory(stale, complete=False, root=tmp_path)
    assert set(saved(tmp_path)["programs"]) == {"future-program"}, "old partial reads cannot resurrect removed programs"


def test_differing_observation_dates_are_explicit_per_program(tmp_path):
    data = inventory(verdict="no")
    other = inventory("2026-12-02", verdict="cannot_tell")["harnesses"][0]
    other["harness"] = "other"
    other["newest"]["observedAt"] = "2026-12-01T00:00:00Z"
    data["harnesses"].append(other)
    facts.record_inventory(data, root=tmp_path)
    line = facts.vendor_programs_line(tmp_path)
    assert "up to date, newest read 2026-11-10, observed 2026-11-10" in line
    assert "order unknown, newest read 2026-12-01, observed 2026-12-02" in line
    other["newest"].pop("observedAt")
    facts.record_inventory(data, root=tmp_path)
    assert "newest read date unknown" in facts.vendor_programs_line(tmp_path)


def test_record_failure_cannot_change_the_owner_response(monkeypatch):
    data = inventory()
    def fail(*_a, **_k):
        raise OSError("disk unavailable")
    monkeypatch.setattr(facts, "record_inventory", fail)
    assert facts.observe_inventory(data) is data


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled", "interrupted"])
def test_settlement_event_waits_for_custody_and_deduplicates_the_receipt(monkeypatch, tmp_path, state):
    launched = []
    monkeypatch.setattr(facts, "_pending", set())
    monkeypatch.setattr(facts, "_root", lambda *_: tmp_path)
    monkeypatch.setattr(facts, "_launch", lambda *args: launched.append(args))
    gateway = SimpleNamespace(_endpoint=object())
    operation = {"id": "one", "harness": "future-program", "state": "running", "termination": "unconfirmed"}
    assert facts.operation_observed(gateway, operation) is operation
    operation.update(state=state, finishedAt="2026-11-10T12:00:00Z")
    facts.operation_observed(gateway, operation)
    assert launched == []
    operation["termination"] = "confirmed"
    facts.operation_observed(gateway, operation)
    facts.operation_observed(gateway, deepcopy(operation))
    assert launched == [(gateway._endpoint, tmp_path, ("settled:future-program", facts._fingerprint(("one", operation["finishedAt"]))))]
    operation["id"] = "two"
    facts.operation_observed(gateway, operation)
    assert len(launched) == 2


def test_event_acquisition_is_passive_fresh_and_does_not_check_latest(monkeypatch, tmp_path):
    from ouroboros.gateways import claudexor
    calls = []
    class Gateway:
        def __init__(self, endpoint):
            assert endpoint == "existing-endpoint"
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            calls.append("closed")
        def handshake(self):
            calls.append("handshake")
        def maintenance_harnesses(self, **kwargs):
            calls.append(kwargs)
            return inventory()
    monkeypatch.setattr(claudexor, "ClaudexorGateway", Gateway)
    def forbidden(*_a, **_k):
        pytest.fail("an observation must not wake the engine")
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", forbidden)
    facts._refresh("existing-endpoint", tmp_path, ("settled:future-program", "event"))
    assert calls == ["handshake", {"fresh": True, "check_latest": False}, "closed"]
    assert "2.1.319" in facts.vendor_programs_line(tmp_path)


def test_settlement_notification_does_not_wait_for_a_snapshot_write(monkeypatch, tmp_path):
    import threading

    done = threading.Event()
    monkeypatch.setattr(facts, "_pending", set())
    monkeypatch.setattr(facts, "_root", lambda *_: tmp_path)
    monkeypatch.setattr(facts, "_launch", lambda *_: done.set())
    receipt = {"id": "one", "harness": "future-program", "state": "succeeded", "termination": "confirmed"}
    worker = threading.Thread(target=facts.operation_observed, args=(SimpleNamespace(_endpoint=object()), receipt))
    with facts._observation_lock(tmp_path) as acquired:
        assert acquired
        worker.start()
        notified = done.wait(timeout=1)
    worker.join(timeout=1)
    assert notified and not worker.is_alive()


@pytest.mark.parametrize("fails", [False, True])
def test_eighteen_independent_hosts_share_generation_and_settlement_attempts(monkeypatch, tmp_path, fails):
    from concurrent.futures import ThreadPoolExecutor
    from importlib.util import module_from_spec, spec_from_file_location
    from threading import Barrier, Event
    from ouroboros.gateways import claudexor, claudexor_metadata

    generation = ["engine-one"]
    calls, workers = [], []
    entered, release = Event(), Event()
    class Gateway:
        def __init__(self, endpoint):
            assert endpoint == "existing-endpoint"
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def handshake(self):
            pass
        def maintenance_harnesses(self, **kwargs):
            assert kwargs == {"fresh": True, "check_latest": False}
            calls.append(generation[0])
            entered.set()
            assert release.wait(timeout=3)
            if fails:
                raise OSError("unreadable inventory")
            return inventory()
    monkeypatch.setattr(claudexor, "ClaudexorGateway", Gateway)
    monkeypatch.setattr(claudexor_metadata, "endpoint_generation", lambda _: generation[0])
    for index in range(18):
        spec = spec_from_file_location(f"independent_facts_{index}", facts.__file__)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "_root", lambda *_: tmp_path)
        workers.append(module)
    gateway = SimpleNamespace(_endpoint="existing-endpoint", engine_version="6.0.0")
    def meet(event):
        barrier = Barrier(len(workers))
        def run(module):
            barrier.wait(timeout=3)
            event(module)
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = [pool.submit(run, module) for module in workers]
            assert entered.wait(timeout=2)
            release.set()
            for future in futures:
                future.result(timeout=4)
        drain_observers()
    ready = lambda module: module.engine_ready(gateway)
    meet(ready)
    assert calls == ["engine-one"]
    first_line = facts.vendor_programs_line(tmp_path)
    first_generation = saved(tmp_path)["events"]["engineGeneration"]
    for _ in range(12):
        for worker in workers:
            worker.engine_ready(gateway)
        drain_observers()
    assert calls == ["engine-one"], "later processes find the generation recorded"
    generation[0] = "engine-two"
    meet(ready)
    assert calls == ["engine-one", "engine-two"]
    assert saved(tmp_path)["events"]["engineGeneration"] != first_generation
    receipt = {"id": "private-settlement", "harness": "future-program", "state": "succeeded", "termination": "confirmed"}
    meet(lambda module: module.operation_observed(gateway, receipt))
    meet(lambda module: module.operation_observed(gateway, deepcopy(receipt)))
    meet(ready)
    assert calls == ["engine-one", "engine-two", "engine-two"]
    facts.record_inventory(inventory("2026-12-12"), root=tmp_path)
    meet(ready)
    assert len(calls) == 3, "ordinary inventory publication preserves the event markers"
    if not fails:
        assert facts.vendor_programs_line(tmp_path) == first_line.replace("observed 2026-11-10", "observed 2026-12-12")
    assert set(saved(tmp_path)["events"]) == {"engineGeneration", "settled:future-program"}
    assert "private-settlement" not in json.dumps(saved(tmp_path))
    facts.record_inventory({"observedAt": "2026-12-13T00:00:00Z", "harnesses": []}, root=tmp_path)
    assert set(saved(tmp_path)["events"]) == {"engineGeneration"}, "removed programs leave no event history"


def drain_observers():
    import threading
    for thread in threading.enumerate():
        if thread.name == "vendor-program-observation":
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_owner_observation_publishes_while_background_inventory_waits(monkeypatch, tmp_path):
    import threading
    from ouroboros.gateways import claudexor
    entered, release = threading.Event(), threading.Event()
    class Gateway:
        def __init__(self, _): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def handshake(self): pass
        def maintenance_harnesses(self, **_):
            entered.set()
            assert release.wait(timeout=3)
            return inventory()
    monkeypatch.setattr(claudexor, "ClaudexorGateway", Gateway)
    monkeypatch.setattr(facts, "_root", lambda *_: tmp_path)
    facts._request("endpoint", ("settled:future-program", "one"))
    try:
        assert entered.wait(timeout=2)
        facts.record_inventory(inventory("2026-12-12"), root=tmp_path)
        assert saved(tmp_path)["completeObservedAt"].startswith("2026-12-12")
    finally:
        release.set()
        drain_observers()
    assert saved(tmp_path)["completeObservedAt"].startswith("2026-12-12")


@pytest.mark.parametrize(("verdict", "newest", "phrase"), [
    ("yes", "2.1.321", "2.1.321 available"),
    ("no", "2.1.321", "up to date"),
    ("cannot_tell", "build-nightly", "newest build-nightly, order unknown"),
    ("cannot_tell", None, "newest not read"),
])
def test_line_names_the_fact_and_groups_absent_programs_once(tmp_path, verdict, newest, phrase):
    data = inventory(verdict=verdict, newest=newest)
    data["harnesses"] += [{"harness": name, "selection": None, "updateAvailable": "cannot_tell",
                          "newest": {"status": "available", "version": "123"}}
                         for name in ("z-absent", "a-absent")]
    facts.record_inventory(data, root=tmp_path)
    line = facts.vendor_programs_line(tmp_path)
    age = "observed 2026-11-10" + ("; newest read 2026-11-10" if newest else "")
    assert line == (f"Harness programs ({age}): "
                    f"future-program 2.1.319 ({phrase}); not installed: a-absent, z-absent.")
    assert "\n" not in line and "cannot_tell" not in line and "123" not in line
    facts.record_inventory(data, root=tmp_path)
    assert facts.vendor_programs_line(tmp_path) == line


def crossing_inventory(answer, row_time, names, version):
    data = inventory(answer, version=version)
    row = data["harnesses"][0]
    row["observedAt"] = row_time + "T12:00:00Z"
    data["harnesses"] = [{**deepcopy(row), "harness": name} for name in names]
    return data


@pytest.mark.parametrize("answer", ["2026-11-11", "2026-11-14"])
def test_complete_answers_merge_each_rows_time_not_the_envelope(tmp_path, answer):
    names = [f"program-{n}" for n in range(7)]
    facts.record_inventory(crossing_inventory("2026-11-12", "2026-11-12", names[:1], "2"), complete=False, root=tmp_path)
    facts.record_inventory(crossing_inventory(answer, "2026-11-10", names, "1"), root=tmp_path)
    rows = saved(tmp_path)["programs"]
    assert set(rows) == set(names)
    assert rows[names[0]]["version"] == "2"
    assert all(rows[name]["version"] == "1" for name in names[1:])
    facts.record_inventory(crossing_inventory("2026-11-16", "2026-11-15", names, "3"), root=tmp_path)
    assert all(row["version"] == "3" for row in saved(tmp_path)["programs"].values())
    facts.record_inventory(crossing_inventory("2026-11-19", "2026-11-18", names[:1], "4"), root=tmp_path)
    facts.record_inventory(crossing_inventory("2026-11-20", "2026-11-17", names[1:], "stale"), complete=False, root=tmp_path)
    assert set(saved(tmp_path)["programs"]) == {names[0]}, "a later envelope cannot resurrect older removed rows"
    no_row_time = crossing_inventory("2026-11-21", "2026-11-10", names[1:2], "5")
    no_row_time["harnesses"][0].pop("observedAt")
    facts.record_inventory(no_row_time, complete=False, root=tmp_path)
    assert saved(tmp_path)["programs"][names[1]]["observedAt"] == no_row_time["observedAt"]


def _process_observation(root, data, complete, wait, notify):
    if wait is not None:
        assert wait.wait(15)
    facts.record_inventory(data, complete=complete, root=root)
    if notify is not None:
        notify.set()


@pytest.mark.parametrize("newest_first", [False, True])
@pytest.mark.parametrize("newest_adds", [False, True])
def test_latest_complete_answer_owns_membership_in_either_publication_order(tmp_path, newest_first, newest_adds):
    names = [f"program-{n}" for n in range(7)]
    older_names, newer_names = (names[:1], names) if newest_adds else (names, names[:1])
    older = crossing_inventory("2026-11-19", "2026-11-17", older_names, "older")
    newer = crossing_inventory("2026-11-20", "2026-11-18", newer_names, "newer")
    for answer in ([newer, older] if newest_first else [older, newer]):
        facts.record_inventory(answer, root=tmp_path)
    rows = saved(tmp_path)["programs"]
    assert set(rows) == set(newer_names)
    assert all(row["version"] == "newer" for row in rows.values())
    assert saved(tmp_path)["completeObservedAt"] == newer["observedAt"]
    # A later complete answer admits named cached rows behind the boundary,
    # but never replaces a saved row whose own observation is newer.
    recovery = crossing_inventory("2026-11-21", "2026-11-16", names, "cached")
    facts.record_inventory(recovery, root=tmp_path)
    rows = saved(tmp_path)["programs"]
    assert set(rows) == set(names)
    assert all(rows[name]["version"] == ("newer" if name in newer_names else "cached") for name in names)
    facts.record_inventory(older, root=tmp_path)
    assert set(saved(tmp_path)["programs"]) == set(names), "an older complete answer cannot prune recovery"


def test_two_process_writers_keep_all_programs_and_newer_selection(tmp_path):
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    published = ctx.Event()
    names = [f"program-{n}" for n in range(7)]
    children = [
        ctx.Process(target=_process_observation, args=(tmp_path, crossing_inventory("2026-11-12", "2026-11-12", names[:1], "2"), False, None, published)),
        ctx.Process(target=_process_observation, args=(tmp_path, crossing_inventory("2026-11-11", "2026-11-10", names, "1"), True, published, None)),
    ]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(20)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(5)
    rows = saved(tmp_path)["programs"]
    assert set(rows) == set(names) and rows[names[0]]["version"] == "2"
    assert not (tmp_path / facts.OBSERVATION_FILE).with_suffix(".lock").exists()


def test_partial_coverage_is_explicit_until_a_complete_answer_lands(tmp_path):
    facts.record_inventory(inventory(), complete=False, root=tmp_path)
    assert "other programs not observed" in facts.vendor_programs_line(tmp_path)
    facts.record_inventory(inventory(), root=tmp_path)
    assert "other programs not observed" not in facts.vendor_programs_line(tmp_path)
    facts.record_inventory(inventory("2026-11-12"), complete=False, root=tmp_path)
    assert "other programs not observed" not in facts.vendor_programs_line(tmp_path)


@pytest.mark.parametrize("status", ["not_read", "unavailable", "available"])
def test_newest_gap_has_its_own_status_and_date(tmp_path, status):
    data = inventory(verdict="cannot_tell")
    data["harnesses"][0]["newest"] = {"status": status, "observedAt": "2026-11-09T10:00:00Z", "version": "nightly"}
    facts.record_inventory(data, root=tmp_path)
    line = facts.vendor_programs_line(tmp_path)
    assert ("newest unavailable 2026-11-09" in line) == (status == "unavailable")
    assert ("newest not read" in line) == (status == "not_read")
    assert ("newest nightly, order unknown" in line) == (status == "available")


@pytest.mark.parametrize("mode", ["recovery_only", "normal"])
def test_owned_acquisition_spends_ready_only_after_recovery_ends(monkeypatch, mode):
    from ouroboros import claudexor_daemon as daemon
    from ouroboros.gateways import claudexor as wire
    calls = []
    monkeypatch.setattr(facts, "engine_ready", lambda gateway: calls.append(gateway))
    monkeypatch.setattr(daemon, "attach_home", lambda: None)
    monkeypatch.setattr(wire, "discover_daemon_at", lambda _: wire.DaemonEndpoint("127.0.0.1", 9, "fixture-only"))
    monkeypatch.setattr(wire.ClaudexorGateway, "handshake", lambda self, **_: {"servingMode": mode})
    monkeypatch.setattr(daemon, "get_owned_daemon", lambda: SimpleNamespace(_remember_stop_targets=lambda _: None))
    with daemon.read_owned_gateway() as gateway:
        assert calls == ([] if mode == "recovery_only" else [gateway])

"""Vendor program facts are event observations; context never acquires an engine."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from ouroboros import harness_program_facts as facts
from tests import test_claudexor_status_passive as passive

metadata_engine = passive.metadata_engine


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
    assert facts.vendor_programs_line(tmp_path) == "Vendor programs: not observed."


def test_unchanged_facts_keep_their_date_across_reads_and_only_current_facts_are_saved(tmp_path):
    first = inventory()
    facts.record_inventory(first, root=tmp_path)
    line = facts.vendor_programs_line(tmp_path)
    assert line == "Vendor programs (facts unchanged since 2026-11-10): future-program 2.1.319 (yes; newest 2.1.321)."
    later = inventory("2026-12-12")
    facts.record_inventory(later, root=tmp_path)
    assert facts.vendor_programs_line(tmp_path) == line
    state = saved(tmp_path)
    assert state["observedAt"] == later["observedAt"]
    assert state["programs"]["future-program"]["newestObservedAt"] == "2026-11-10T11:00:00Z"
    assert "private-receipt" not in json.dumps(state) and "before" not in json.dumps(state)
    facts.record_inventory(inventory("2026-12-13", version="2.1.321", verdict="no"), root=tmp_path)
    assert facts.vendor_programs_line(tmp_path) == "Vendor programs (facts unchanged since 2026-12-13): future-program 2.1.321 (no; newest 2.1.321)."


def test_absent_unknown_and_empty_inventories_are_distinct_and_sorted(tmp_path):
    data = inventory(verdict="cannot_tell", version=None)
    data["harnesses"][0]["newest"] = {"status": "unavailable", "observedAt": data["observedAt"]}
    data["harnesses"].append({"harness": "absent", "selection": None, "updateAvailable": "cannot_tell"})
    facts.record_inventory(data, root=tmp_path)
    assert facts.vendor_programs_line(tmp_path).endswith("absent not installed (cannot_tell); future-program version unknown (cannot_tell).")
    facts.record_inventory({"observedAt": "2026-11-11T00:00:00Z", "harnesses": []}, root=tmp_path)
    assert facts.vendor_programs_line(tmp_path).endswith("none listed.")


def test_partial_reads_merge_and_late_full_reads_cannot_erase_newer_facts(tmp_path):
    facts.record_inventory(inventory(), root=tmp_path)
    newer = inventory("2026-11-12", version="opaque-vendor-build", verdict="cannot_tell")
    newer["harnesses"][0]["harness"] = "another"
    # Program file did not change; the inventory's newer read still owns this row.
    newer["harnesses"][0]["observedAt"] = "2026-11-09T00:00:00Z"
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


def test_old_selection_observations_do_not_replace_newer_ones(tmp_path):
    facts.record_inventory(inventory("2026-11-12"), root=tmp_path)
    older = inventory("2026-11-13", version="stale")
    older["harnesses"][0]["observedAt"] = "2026-11-11T00:00:00Z"
    facts.record_inventory(older, root=tmp_path)
    assert saved(tmp_path)["programs"]["future-program"]["version"] == "2.1.319"
    assert saved(tmp_path)["factsSince"] == "2026-11-12"


def test_record_failure_cannot_change_the_owner_response(monkeypatch):
    data = inventory()
    def fail(*_a, **_k):
        raise OSError("disk unavailable")
    monkeypatch.setattr(facts, "record_inventory", fail)
    assert facts.observe_inventory(data) is data


def test_passive_ready_event_is_once_per_generation_and_never_during_recovery(metadata_engine, monkeypatch):
    from ouroboros import claudexor_daemon

    state, calls, _clients, provision, config_dir = metadata_engine
    launched = []
    monkeypatch.setattr(facts, "_pending", set())
    monkeypatch.setattr(facts, "_launch", lambda *args: launched.append(args))
    provision()
    for _ in range(2):
        claudexor_daemon.read_owned_gateway().close()
    assert len(launched) == 1
    assert all(path == "/v2/handshake" for _method, path, _query in calls)
    # Replacing a descriptor admits one new ready observation, not a status timer.
    path = config_dir / "daemon/control-api.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    claudexor_daemon.read_owned_gateway().close()
    assert len(launched) == 2
    state["handshake_error"] = True
    with pytest.raises(Exception):
        claudexor_daemon.read_owned_gateway()
    assert len(launched) == 2
    from ouroboros.gateways.claudexor import ClaudexorGateway
    monkeypatch.setattr(ClaudexorGateway, "handshake", lambda *_a, **_k: {"servingMode": "recovery_only"})
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    claudexor_daemon.read_owned_gateway().close()
    assert len(launched) == 2


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


def test_eighteen_independent_hosts_share_generation_and_settlement_observations(monkeypatch, tmp_path):
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
            assert release.wait(timeout=2)
            return inventory()
    monkeypatch.setattr(claudexor, "ClaudexorGateway", Gateway)
    monkeypatch.setattr(claudexor_metadata, "endpoint_generation", lambda _: generation[0])
    for index in range(18):
        spec = spec_from_file_location(f"independent_facts_{index}", facts.__file__)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "_root", lambda *_: tmp_path)
        # Each worker has independent globals; its dispatch runs on our controlled thread.
        monkeypatch.setattr(module, "_launch", module._refresh)
        workers.append(module)
    gateway = SimpleNamespace(_endpoint="existing-endpoint")
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
    ready = lambda module: module.engine_ready(gateway)
    meet(ready)
    assert calls == ["engine-one"]
    first_line = facts.vendor_programs_line(tmp_path)
    first_generation = saved(tmp_path)["events"]["engineGeneration"]
    for worker in workers:
        worker.engine_ready(gateway)
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
    assert facts.vendor_programs_line(tmp_path) == first_line
    assert set(saved(tmp_path)["events"]) == {"engineGeneration", "settled:future-program"}
    assert "private-settlement" not in json.dumps(saved(tmp_path))
    facts.record_inventory({"observedAt": "2026-12-13T00:00:00Z", "harnesses": []}, root=tmp_path)
    assert set(saved(tmp_path)["events"]) == {"engineGeneration"}, "removed programs leave no event history"


def test_failed_ready_observation_does_not_mark_generation_observed(monkeypatch, tmp_path):
    from ouroboros.gateways import claudexor, claudexor_metadata

    attempts = []
    def unavailable(_):
        attempts.append("attempt")
        raise OSError("engine unavailable")
    monkeypatch.setattr(claudexor, "ClaudexorGateway", unavailable)
    monkeypatch.setattr(claudexor_metadata, "endpoint_generation", lambda _: "generation")
    monkeypatch.setattr(facts, "_root", lambda *_: tmp_path)
    monkeypatch.setattr(facts, "_launch", facts._refresh)
    monkeypatch.setattr(facts, "_pending", set())
    for _ in range(2):
        facts.engine_ready(SimpleNamespace(_endpoint="existing-endpoint"))
        assert facts._read(tmp_path) == {} and not facts._pending
    assert len(attempts) == 2, "another event can retry; failure is not an observation"

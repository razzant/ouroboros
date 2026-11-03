"""Safety's real bookkeeping produces a small durable, verdict-neutral wake projection."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ouroboros import consciousness_view as view, safety_records as records
from ouroboros.task_store import delete_document, read_document
from tests._safety_fixtures import supervisor as supervisor_fixture
from tests._store_shared import seed_result

supervisor = supervisor_fixture


@pytest.mark.parametrize("kind", ["safety_mode_skip", "safety_declared_read_skip", "safety_hold"])
def test_repeated_chip_avoids_store_but_new_chip_and_other_root_write(tmp_path, monkeypatch, kind):
    from ouroboros import task_store

    original = task_store.patch_document
    calls = []
    def patch(*args):
        calls.append(args[:2])
        return original(*args)
    monkeypatch.setattr(task_store, "patch_document", patch)
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="wake", drive_logs=lambda: tmp_path / "logs")
    seed_result(tmp_path, "wake", {"status": "running"})
    records._emit_durable_safety_event(ctx, {"type": kind})
    records._emit_durable_safety_event(ctx, {"type": kind})
    assert len(calls) == 1
    assert len((tmp_path / "logs" / "events.jsonl").read_text().splitlines()) == 2  # Audit is not deduplicated.
    records._emit_durable_safety_event(ctx, {"type": "safety_verdict", "verdicts": [{"status": "SAFE"}]})
    assert len(calls) == 2 and len(read_document(tmp_path, "wake")["safety_checks"]) == 2
    other = tmp_path / "other"
    seed_result(other, "wake", {"status": "running"})
    records._record_task_checks(SimpleNamespace(drive_root=other), {"type": kind, "task_id": "wake"})
    assert len(calls) == 3 and len(read_document(other, "wake")["safety_checks"]) == 1


def test_failed_chip_write_can_retry(tmp_path, monkeypatch):
    from ouroboros import task_store

    original = task_store.patch_document
    attempts = []
    def patch(*args):
        attempts.append(args[:2])
        if len(attempts) == 1:
            raise OSError("temporary storage failure")
        return original(*args)
    monkeypatch.setattr(task_store, "patch_document", patch)
    seed_result(tmp_path, "wake", {"status": "running"})
    ctx = SimpleNamespace(drive_root=tmp_path)
    event = {"type": "safety_mode_skip", "task_id": "wake"}
    with pytest.raises(OSError):
        records._record_task_checks(ctx, event)
    records._record_task_checks(ctx, event)
    records._record_task_checks(ctx, event)
    assert len(attempts) == 2 and read_document(tmp_path, "wake")["safety_checks"] == [{"chip": "not_checked", "tool": ""}]


def test_actual_safety_decisions_produce_panel_chips(supervisor, monkeypatch):
    h = supervisor
    from ouroboros import tool_effects
    from ouroboros.task_results import write_task_result

    h.ctx.task_metadata = {"work_kind": "consciousness", "wake_cause": "own_clock"}
    h.ctx.drive_logs = lambda: h.ctx.drive_root / "logs"
    seed_result(h.ctx.drive_root, "work", {"status": "running", "metadata": h.ctx.task_metadata})
    monkeypatch.setattr(h.module, "_emit_durable_safety_event", records._emit_durable_safety_event)
    monkeypatch.setattr(h.module, "_emit_safety_mode_skip", records._emit_safety_mode_skip)
    monkeypatch.setattr(view, "wake_ids", lambda *args: ["work"])  # the panel selects wakes through the usage ledger
    assert view.recent_wakes(h.ctx.drive_root, [])[0]["checks"] == []
    assert h.module.check_safety("place_reply", {"text": "hello"}, ctx=h.ctx)[0]
    assert view.recent_wakes(h.ctx.drive_root, [])[0]["checks"] == [{"chip": "checked", "tool": "place_reply"}]
    h.replies.append({"status": "DANGEROUS", "reason": "not delegated"})
    assert not h.module.check_safety("place_reply", {"text": "flagged"}, ctx=h.ctx)[0]
    assert h.module.check_safety("place_reply", {"text": "flagged"}, ctx=h.ctx)[0]
    h.replies.append(RuntimeError("checker unavailable"))
    assert not h.module.check_safety("place_reply", {"text": "held"}, ctx=h.ctx)[0]
    h.modes["consciousness"] = "off"
    assert h.module.check_safety("place_reply", {"text": "mode skip"}, ctx=h.ctx)[0]
    h.modes["consciousness"] = "full"
    monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: tool_effects.ToolEffect("read", declared=True))
    assert h.module.check_safety("connector_read", {}, ctx=h.ctx)[0]
    # Lifecycle merges retain the projection; a panel read needs no events.jsonl.
    write_task_result(h.ctx.drive_root, "work", "completed", result="Finished")
    (h.ctx.drive_logs() / "events.jsonl").unlink()
    checks = view.recent_wakes(h.ctx.drive_root, [])[0]["checks"]
    assert [item["chip"] for item in checks] == ["checked", "warned", "sent_despite", "held", "not_checked"]


@pytest.mark.parametrize("event,chips", [
    ({"type": "safety_verdict", "verdicts": [{"status": "SAFE"}]}, ["checked"]),
    ({"type": "safety_verdict", "verdicts": [{"status": "SUSPICIOUS"}]}, ["warned"]),
    ({"type": "safety_verdict", "verdicts": [{"status": "DANGEROUS"}, {"status": "SAFE"}]}, ["warned", "checked"]),
    ({"type": "safety_sent_despite_check"}, ["sent_despite"]),
    ({"type": "safety_hold"}, ["held"]),
    ({"type": "safety_mode_skip"}, ["not_checked"]),
    ({"type": "safety_declared_read_skip"}, ["not_checked"]),
    ({"type": "safety_parse_failed"}, []),
])
def test_each_durable_event_maps_without_inventing_checks(tmp_path, event, chips):
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="wake", drive_logs=lambda: tmp_path / "logs")
    records._emit_durable_safety_event(ctx, {**event, "tool": "connector"})
    assert read_document(tmp_path, "wake") is None  # a check cannot invent a task
    seed_result(tmp_path, "wake", {"status": "running"})
    records._emit_durable_safety_event(ctx, {**event, "tool": "connector"})
    assert read_document(tmp_path, "wake").get("safety_checks", []) == [
        {"chip": chip, "tool": "connector"} for chip in chips]


def test_parallel_checks_are_bounded_and_removed_with_task(tmp_path):
    seed_result(tmp_path, "wake", {"status": "running", "result": "preserved"})
    ctx = SimpleNamespace(drive_root=tmp_path)
    events = [{"type": "safety_verdict", "verdicts": [{"status": "SAFE"}, {"status": "DANGEROUS"}]},
              {"type": "safety_sent_despite_check"}, {"type": "safety_hold"}, {"type": "safety_mode_skip"}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda event: records._record_task_checks(ctx, {**event, "task_id": "wake", "tool": "🙂" * 500}),
                      events * 3))
    document = read_document(tmp_path, "wake")
    checks = document["safety_checks"]
    assert len(checks) == 5 and {item["chip"] for item in checks} == set(records._CHECK_CHIPS)
    assert all(len(item["tool"]) == 160 for item in checks)
    assert len(json.dumps(checks).encode()) < 10 * 1024 and document["result"] == "preserved"
    assert delete_document(tmp_path, "wake") and read_document(tmp_path, "wake") is None


def test_index_failure_does_not_suppress_the_original_durable_audit(tmp_path, monkeypatch):
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="wake", drive_logs=lambda: tmp_path / "logs")
    def unavailable(*args):
        raise OSError("index unavailable")
    monkeypatch.setattr(records, "_record_task_checks", unavailable)
    records._emit_safety_mode_skip(ctx, "place_reply", "off", "check")
    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text())
    assert event["type"] == "safety_mode_skip" and event["task_id"] == "wake"

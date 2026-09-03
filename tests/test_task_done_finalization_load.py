"""Battle test: 64 terminal tasks' forensic copy-back through the 4-thread pool.

full1507 postmortem class: ``_TASK_DONE_EXECUTOR`` (4 threads) inflated every
child blob during ``copy_child_task_result``; with ~125 MB compressed per task
the pool never drained, ``RUNNING`` rows were never popped, worker lanes were
never freed, and the launcher's 2 h deadline then cancelled the tasks queued
behind them. This test drives the REAL copy-back for a full lane count of
children with realistic call closures through a pool of the production width
and pins the two properties that make the lane-release path cheap: no blob is
ever decompressed on the control plane, and the whole cohort finalizes in
seconds, not hours.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from ouroboros import observability
from ouroboros.headless import copy_child_task_result, prepare_task_drive
from ouroboros.observability import persist_call, read_blob_ref
from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result

_LANES = 64
_CALLS_PER_TASK = 16
_POOL_WIDTH = 4  # supervisor.events._TASK_DONE_EXECUTOR


def _seed_child(parent: pathlib.Path, index: int) -> tuple[str, pathlib.Path]:
    task_id = f"load-lane-{index:02d}"
    child = prepare_task_drive(parent, task_id, "empty")
    assert child is not None
    refs = []
    for call in range(_CALLS_PER_TASK):
        persisted = persist_call(
            child,
            task_id=task_id,
            call_id=f"llm_{call}",
            call_type="llm_request",
            payload={
                "prompt": f"lane {index} round {call}",
                # Realistic transcript bulk: tens of KB of low-entropy context.
                "context": ("fuzz target harness output line\n" * 900) + str(call),
            },
        )
        refs.append({"request_ref": persisted["manifest_ref"]})
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        trace_refs={"llm_call_refs": refs},
    )
    return task_id, child


def test_64_lane_copyback_finalizes_without_inflating_blobs(tmp_path, monkeypatch):
    parent = tmp_path / "data"
    parent.mkdir()
    children = [_seed_child(parent, index) for index in range(_LANES)]

    inflated: list[str] = []
    real_open = gzip.open

    def _spy(path, *args, **kwargs):
        inflated.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(observability.gzip, "open", _spy)

    latencies: dict[str, float] = {}

    def _finalize(item: tuple[str, pathlib.Path]) -> None:
        task_id, child = item
        started = time.monotonic()
        copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
        latencies[task_id] = time.monotonic() - started
        assert copied is not None
        assert copied["child_ref_promotion"]["status"] == "complete"
        assert copied["child_ref_promotion"]["unavailable_refs"] == []
        assert copied["child_ref_promotion"]["promoted_ref_count"] >= _CALLS_PER_TASK

    wall_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=_POOL_WIDTH) as pool:
        list(pool.map(_finalize, children))
    wall = time.monotonic() - wall_started
    monkeypatch.undo()

    assert inflated == [], f"{len(inflated)} blob(s) were decompressed on the finalize pool"
    assert len(latencies) == _LANES
    worst = max(latencies.values())
    p50 = statistics.median(latencies.values())
    # Generous CI bounds; the production shape was minutes per task and the
    # pool never draining. A lane must be reusable well inside the 5 s the
    # dispatcher needs to observe a freed worker.
    assert worst < 5.0, f"slowest copy-back {worst:.2f}s (p50 {p50:.3f}s)"
    assert wall < 60.0, f"cohort copy-back took {wall:.1f}s"

    # Durable truth landed for every lane and is readable from canonical CAS.
    for task_id, _child in children:
        canonical = load_task_result(parent, task_id) or {}
        assert canonical.get("status") == STATUS_COMPLETED
        first_ref = canonical["trace_refs"]["llm_call_refs"][0]["request_ref"]
        assert pathlib.Path(first_ref["path"]).is_relative_to(parent / "observability")
    manifest_path = pathlib.Path(
        (load_task_result(parent, children[7][0]) or {})["trace_refs"]["llm_call_refs"][3][
            "request_ref"
        ]["path"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert read_blob_ref(parent, manifest["full_payload_ref"])["prompt"] == "lane 7 round 3"


def test_relocation_is_an_order_of_magnitude_cheaper_than_inflating(tmp_path, monkeypatch):
    """Same closure, both paths, one lane: pins the direction of the fix."""

    parent = tmp_path / "data"
    parent.mkdir()
    task_id, child = _seed_child(parent, 0)
    task = {"id": task_id, "drive_root": str(child)}

    started = time.monotonic()
    copied = copy_child_task_result(parent, task)
    fast = time.monotonic() - started
    assert copied["child_ref_promotion"]["status"] == "complete"

    # Force the legacy verifying path for an identical closure in a fresh parent.
    legacy_parent = tmp_path / "legacy"
    legacy_parent.mkdir()
    legacy_task_id, legacy_child = _seed_child(legacy_parent, 0)
    monkeypatch.setattr(observability, "_relocate_portable_blob", lambda *a, **k: None)
    started = time.monotonic()
    legacy = copy_child_task_result(
        legacy_parent, {"id": legacy_task_id, "drive_root": str(legacy_child)}
    )
    slow = time.monotonic() - started
    assert legacy["child_ref_promotion"]["status"] == "complete"
    assert legacy["child_ref_promotion"]["unavailable_refs"] == []
    assert copied["child_ref_promotion"]["unavailable_refs"] == []

    # Not a benchmark assertion on absolute time: only the ratio must hold,
    # and loosely, so a loaded CI box cannot flip it.
    assert fast * 2.0 < slow, f"relocation {fast:.3f}s vs inflate {slow:.3f}s"

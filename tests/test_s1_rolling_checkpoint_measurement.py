"""Measure real rolling checkpoint I/O; timings are evidence, never thresholds.

Run with pytest -s to retain the S1_CHECKPOINT_MEASUREMENT JSON records. Each
case starts with exactly 100 KB, 1 MB or 5 MB of JSON-encoded mixed transcript
(decimal bytes, UTF-8, production JSON separators) and advances eight rounds
through all four boundaries. The serializer, clock and atomic writer are real.
Storage census and readback happen outside the measured save interval. No
provider, engine, recovery freeze or power-loss durability is exercised here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import pytest

from ouroboros import owner_wait, utils, working_checkpoint as wc
from ouroboros.loop_delivery import DeliveryCandidate
from tests._budget_pause_exact_helpers import _loop_ctx


ROUNDS = 8


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _transcript(target_bytes):
    """Mixed roles, Unicode, tool arguments and results, sized without truncation."""
    messages = [
        {"role": "system", "content": "Preserve complete work and report observed evidence."},
        {"role": "user", "content": "Inspect the project and retain the findings."},
    ]
    results = []
    for index in range(max(1, target_bytes // 8192)):
        call_id = f"historical-{index}"
        result = {"role": "tool", "tool_call_id": call_id,
                  "content": f"File {index}: café / состояние / Δ; measured rows follow.\n"}
        messages.extend([
            {"role": "assistant", "content": f"Inspect module {index} and its callers.",
             "tool_calls": [{"id": call_id, "type": "function", "function": {
                 "name": "read_file", "arguments": json.dumps({"path": f"module_{index}.py"})}}]},
            result,
            {"role": "assistant", "content": f"Module {index} preserves the observed boundary."},
        ])
        results.append(result)
    remaining = target_bytes - len(_json_bytes(messages))
    assert remaining >= 0
    each, extra = divmod(remaining, len(results))
    pattern = "def retain(value): return value; observed=complete; "
    for index, result in enumerate(results):
        length = each + (index < extra)
        # ASCII filler makes the exact UTF-8 byte size independent of Unicode width.
        result["content"] += (pattern * (length // len(pattern) + 1))[:length]
    assert len(_json_bytes(messages)) == target_bytes
    return messages


def _advance(ctx, limit, boundary, index):
    call_id = f"new-{index}"
    if boundary == "ready":
        ctx._delivery_candidate = None
        limit.round_idx = index + 1
        limit.owner_msg_seen.add(f"owner-{index}")
        limit.messages.append({"role": "user", "content": f"Also inspect new module {index}."})
    elif boundary == "pre_effect":
        limit.messages.append({"role": "assistant", "content": "Read the additional module.",
                               "tool_calls": [{"id": call_id, "type": "function", "function": {
                                   "name": "read_file", "arguments": json.dumps({"path": f"new_{index}.py"})}}]})
    elif boundary == "post_batch":
        limit.messages.append({"role": "tool", "tool_call_id": call_id,
                               "content": f"Module {index}: result retained, no pending effect."})
        limit.llm_trace["tool_calls"].append({"id": call_id, "name": "read_file", "status": "success"})
        limit.accumulated_usage["cost"] += 0.01
    else:
        answer = f"Round {index + 1}: inspected the new module; the result is retained."
        limit.messages.append({"role": "assistant", "content": answer})
        ctx._delivery_candidate = DeliveryCandidate(
            full_text=answer, content_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            revision=index + 1, evidence_revision=index + 1,
            evidence_fingerprint=f"evidence-{index}", acceptance_binding={})


def _save_sample(root, limit, boundary):
    previous = dict(limit.accumulated_usage.get("working_checkpoint") or {})
    started = time.perf_counter()
    assert wc.save_round(limit, boundary) is True
    wall_ms = (time.perf_counter() - started) * 1000
    stats = limit.accumulated_usage["working_checkpoint"]
    ctx = limit.tools._ctx
    path = wc.checkpoint_path(root, ctx.task_id, ctx.task_attempt)
    data, state = wc.read_working(root, ctx.task_id, ctx.task_attempt)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    assert files == [path], "ordinary saves must not accumulate immutable snapshots or temporary files"
    assert state["messages"] == limit.messages
    assert state["trace"] == limit.llm_trace
    assert state["seen"] == sorted(limit.owner_msg_seen)
    assert state["working"]["seq"] == stats["saves"]
    assert state["working"]["boundary"] == boundary
    assert state["working"]["pending_tool_call_ids"] == (
        [f"new-{limit.round_idx - 1}"] if boundary == "pre_effect" else [])
    assert "working_checkpoint" not in state["usage"], "measurement counters must not amplify the snapshot"
    assert len(data) == stats["last_bytes"] == path.stat().st_size
    if boundary == "candidate":
        assert state["delivery_candidate"]["full_text"] == ctx._delivery_candidate.full_text
    return {"seq": stats["saves"], "round": limit.round_idx, "boundary": boundary,
            "transcript_bytes": len(_json_bytes(limit.messages)), "checkpoint_bytes": len(data),
            "encode_ms": round(stats["encode_ms"] - previous.get("encode_ms", 0), 3),
            "write_ms": round(stats["write_ms"] - previous.get("write_ms", 0), 3),
            "save_wall_ms": round(wall_ms, 3), "retained_files": len(files),
            "retained_bytes": sum(p.stat().st_size for p in files),
            "sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize("transcript_bytes", [100_000, 1_000_000, 5_000_000],
                         ids=["100KB", "1MB", "5MB"])
def test_repeated_boundaries_measure_real_writer_and_retain_one_snapshot(tmp_path, transcript_bytes):
    ctx, limit = _loop_ctx(tmp_path, "s1-checkpoint-measurement")
    limit.messages = _transcript(transcript_bytes)
    limit.llm_trace = {"tool_calls": [{"id": row["tool_call_id"], "name": "read_file", "status": "success"}
                                     for row in limit.messages if row["role"] == "tool"]}
    limit.tool_schemas = [{"type": "function", "function": {
        "name": "read_file", "description": "Read a project source file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    limit.budget_tail = "tool"
    samples, skipped_ready = [], 0
    for index in range(ROUNDS):
        for boundary in wc.BOUNDARIES:
            _advance(ctx, limit, boundary, index)
            samples.append(_save_sample(tmp_path, limit, boundary))
            if boundary == "ready":
                path = wc.checkpoint_path(tmp_path, ctx.task_id, ctx.task_attempt)
                before = path.read_bytes(), path.stat().st_mtime_ns, dict(limit.accumulated_usage["working_checkpoint"])
                assert wc.save_round(limit, "ready") is False
                assert before == (path.read_bytes(), path.stat().st_mtime_ns,
                                  dict(limit.accumulated_usage["working_checkpoint"]))
                skipped_ready += 1
    stats = limit.accumulated_usage["working_checkpoint"]
    assert stats["saves"] == ROUNDS * len(wc.BOUNDARIES)
    assert len({sample["sha256"] for sample in samples}) == stats["saves"]
    assert stats["max_bytes"] == max(sample["checkpoint_bytes"] for sample in samples)
    by_boundary = {}
    for boundary in wc.BOUNDARIES:
        rows = [sample for sample in samples if sample["boundary"] == boundary]
        by_boundary[boundary] = {key: {"total": round(sum(row[key] for row in rows), 3),
                                      "median": round(statistics.median(row[key] for row in rows), 3),
                                      "max": max(row[key] for row in rows)}
                                 for key in ("checkpoint_bytes", "encode_ms", "write_ms", "save_wall_ms")}
    sources = [Path(module.__file__).resolve() for module in (wc, owner_wait, utils)] + [Path(__file__).resolve()]
    report = {"schema_version": 1, "initial_transcript_bytes": transcript_bytes,
              "rounds": ROUNDS, "saves": stats["saves"], "unchanged_ready_skips": skipped_ready,
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
              "environment": {"platform": platform.platform(), "machine": platform.machine(),
                              "python": sys.version, "executable": sys.executable, "cpu_count": os.cpu_count(),
                              "storage_root": str(tmp_path), "storage_device_id": tmp_path.stat().st_dev},
              "timing_semantics": {"encode_ms": "production continuation_state + JSON UTF-8 encoding; cumulative ms rounded to 0.001",
                                   "write_ms": "production path creation + temp write + atomic replace; no fsync",
                                   "save_wall_ms": "complete save_round; excludes fixture construction, readback and census",
                                   "conditions": "local filesystem; no forced cache flush; first sample included; no performance limits"},
              "total_written_bytes": sum(sample["checkpoint_bytes"] for sample in samples),
              "final_retained_files": samples[-1]["retained_files"],
              "final_retained_bytes": samples[-1]["retained_bytes"],
              "production_stats": stats, "by_boundary": by_boundary, "samples": samples}
    print("S1_CHECKPOINT_MEASUREMENT " + json.dumps(report, ensure_ascii=False, sort_keys=True))

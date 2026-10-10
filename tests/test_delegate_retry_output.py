"""Reader-specific EOF evidence through registered retry consumers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_output
from ouroboros.tool_capabilities import tool_result_limit
from tests.test_delegate_retry_consumers import (
    TerminalGateway, bind_gateway, call, durable_run, retry_context,
    isolated_retry_state as isolated_retry_state,  # re-export the autouse pytest fixture
)


pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def reader_coverage(monkeypatch):
    from types import SimpleNamespace
    from ouroboros import task_status
    from ouroboros.tools import delegate

    # This test has no supervisor refreshing its synthetic queue. Keep only
    # the ownership clock fixed while exercising slow, paged EOF reads; the
    # separate retry-consumer tests still test stale-snapshot rejection.
    clock = task_status.time
    now = clock.time()
    monkeypatch.setattr(task_status, "time", SimpleNamespace(
        time=lambda: now, monotonic=clock.monotonic, sleep=clock.sleep))
    coverage = {}
    monkeypatch.setattr(delegate_output, "_READ_COVERAGE", coverage)
    monkeypatch.setattr(delegate, "_READ_COVERAGE", coverage)


def wait_result(ctx, run_id):
    result = call(ctx, "delegate_wait", run_id=run_id)
    assert result.status == "ok", result.text
    return json.loads(result.text)


def read_whole(ctx, artifact):
    stride = tool_result_limit("read_file") - 5000
    lines = Path(artifact["abs_path"]).read_text(encoding="utf-8").splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        for offset in range(0, max(len(line), 1), stride):
            result = call(ctx, "read_file", root="task_drive", path=artifact["path"],
                          start_line=number, max_lines=1, start_char=offset)
            assert result.status == "ok", result.text


def receipts(ctx):
    return [row for row in custody.custody_rows(custody.custody_root(ctx))
            if row["type"] == custody.OUTPUT_CONSUMED]


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("owner,chain", [
    ("retry-a", ("retry-a", "retry-b")),
    ("retry-a", ("retry-a", "retry-b", "retry-c")),
    ("retry-b", ("retry-a", "retry-b", "retry-c")),
])
def test_predecessor_ack_does_not_count_as_successor_eof(tmp_path, monkeypatch, split, owner, chain):
    ctx = retry_context(tmp_path, monkeypatch, split=split, chain=chain)
    held = durable_run(ctx, owner=owner)
    gateway = TerminalGateway(output="complete work " * 12000)
    bind_gateway(monkeypatch, gateway)
    predecessor = copy.copy(ctx)
    predecessor.task_id = held.task_id
    predecessor.task_metadata = {"root_task_id": held.task_id, "parent_task_id": ""}
    original = wait_result(predecessor, held.run_id)["output_delivery"]["artifact"]
    read_whole(predecessor, original)
    assert custody.replay(custody.custody_root(ctx))[held.run_id].output_consumed

    successor = wait_result(ctx, held.run_id)
    delivery = successor["output_delivery"]
    assert not delivery["consumed"]
    artifact = delivery["artifact"]
    assert artifact["sha256"] == original["sha256"]
    assert Path(artifact["abs_path"]).is_relative_to(ctx.drive_root)
    for start in (1, artifact["lines"]):
        result = call(ctx, "read_file", root="task_drive", path=artifact["path"],
                      start_line=start, max_lines=1)
        assert result.status == "ok", result.text
    assert len(receipts(ctx)) == 1, "head and tail must leave the unread middle unacknowledged"
    assert not wait_result(ctx, held.run_id)["output_delivery"]["consumed"]
    read_whole(ctx, artifact)
    row = receipts(ctx)[-1]
    assert row["task_id"] == held.task_id
    assert row["reader_task_id"] == ctx.task_id
    raw = Path(artifact["abs_path"]).read_bytes()
    assert row["sha256"] == hashlib.sha256(raw).hexdigest() == original["sha256"]
    assert row["bytes"] == len(raw)
    custody._CUSTODY.clear()
    assert wait_result(ctx, held.run_id)["output_delivery"]["consumed"]
    read_whole(ctx, artifact)
    assert len(receipts(ctx)) == 2, "same reader and content has one durable receipt"


def test_successor_receipt_invalidates_for_different_same_length_content(tmp_path, monkeypatch):
    ctx = retry_context(tmp_path, monkeypatch)
    held = durable_run(ctx)
    gateway = TerminalGateway(output="A" * 120000)
    bind_gateway(monkeypatch, gateway)
    first = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    read_whole(ctx, first)
    assert wait_result(ctx, held.run_id)["output_delivery"]["consumed"]
    gateway.output = "B" * 120000
    second = wait_result(ctx, held.run_id)["output_delivery"]
    assert not second["consumed"]
    assert second["artifact"]["bytes"] == first["bytes"]
    assert second["artifact"]["sha256"] != first["sha256"]
    custody._CUSTODY.clear()
    read_whole(ctx, second["artifact"])
    assert wait_result(ctx, held.run_id)["output_delivery"]["consumed"]
    assert len(receipts(ctx)) == 2


def test_old_style_ack_is_starter_receipt_and_duplicate_start_preserves_readers(tmp_path, monkeypatch):
    ctx = retry_context(tmp_path, monkeypatch)
    held = durable_run(ctx)
    bind_gateway(monkeypatch, TerminalGateway(output="V" * 120000))
    artifact = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    data = custody.custody_root(ctx)
    assert custody.emit(data, custody.OUTPUT_CONSUMED, {
        "run_id": held.run_id, "task_id": held.task_id, "sha256": artifact["sha256"],
    })
    custody._CUSTODY.clear()
    assert not wait_result(ctx, held.run_id)["output_delivery"]["consumed"]
    read_whole(ctx, artifact)
    assert custody.record_started(data, held)
    custody._CUSTODY.clear()
    replayed = custody.replay(data)[held.run_id]
    assert dict(replayed.output_reader_receipts) == {
        held.task_id: artifact["sha256"], ctx.task_id: artifact["sha256"],
    }
    assert replayed.task_id == held.task_id


@pytest.mark.parametrize("invalid", ["narrative", "review", "absent"])
def test_foreign_read_never_mints_a_receipt_without_terminal_retry_authority(
    tmp_path, monkeypatch, invalid,
):
    from ouroboros.task_results import write_task_result

    ctx = retry_context(tmp_path, monkeypatch)
    fields = {"source": "review_substrate", "review_slot_id": "slot"} if invalid == "review" else {}
    held = durable_run(ctx, state="" if invalid == "absent" else "succeeded", **fields)
    if invalid == "narrative":
        write_task_result(custody.custody_root(ctx), ctx.task_id, "running",
                          original_task_id="", timeout_retry_from="", supersedes_task_id="")
    artifact = delegate_output._stage_full_output(ctx, held.run_id, "completed body\n")
    assert artifact is not None
    assert custody.emit(custody.custody_root(ctx), custody.OUTPUT_SPILLED, {
        "run_id": held.run_id, "task_id": held.task_id, "artifact": artifact["path"],
        "staged": True, "full_content": True, "sha256": artifact["sha256"],
    })
    read_whole(ctx, artifact)
    assert not receipts(ctx)


def test_small_inline_retry_result_is_actually_complete_without_eof_gate(tmp_path, monkeypatch):
    ctx = retry_context(tmp_path, monkeypatch)
    held = durable_run(ctx, state="failed")
    bind_gateway(monkeypatch, TerminalGateway(state="failed", output="salvaged work"))
    payload = wait_result(ctx, held.run_id)
    assert payload["primary_output"] == "salvaged work"
    assert payload["output_delivery"]["complete"]
    assert payload["output_delivery"]["consumed"]
    assert payload["output_delivery"]["artifact"] is None


def test_native_readonly_retry_can_read_and_ack_full_result(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tool_access import LOCAL_READONLY_SUBAGENT_MODE

    ctx = retry_context(tmp_path, monkeypatch, split=True)
    ctx.task_constraint = TaskConstraint(mode=LOCAL_READONLY_SUBAGENT_MODE)
    held = durable_run(ctx)
    bind_gateway(monkeypatch, TerminalGateway(output="V" * 120000))
    artifact = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    read_whole(ctx, artifact)
    assert wait_result(ctx, held.run_id)["output_delivery"]["consumed"]
    apply = call(ctx, "integrate_delegated_patch", run_id=held.run_id, decision="apply")
    assert apply.status != "ok"


@pytest.mark.parametrize("large", [False, True])
def test_completed_widened_profile_is_readable_with_honest_evidence(tmp_path, monkeypatch, large):
    ctx = retry_context(tmp_path, monkeypatch)
    held = durable_run(ctx)
    bind_gateway(monkeypatch, TerminalGateway(
        output="V" * (120000 if large else 100), effective_access="full",
    ))
    payload = wait_result(ctx, held.run_id)
    assert payload["access_evidence"]["verified"] is False
    assert payload["containment_breach"]["code"] == "access_profile_widened"
    assert payload["containment_breach"]["effective_access"] == "full"
    if large:
        artifact = payload["output_delivery"]["artifact"]
        staged = json.loads(Path(artifact["abs_path"]).read_text(encoding="utf-8"))
        assert staged["access_evidence"]["verified"] is False
        assert staged["containment_breach"] == payload["containment_breach"]
        read_whole(ctx, artifact)
        assert wait_result(ctx, held.run_id)["output_delivery"]["consumed"]
    assert not any(row["type"] == custody.CONTAINMENT_FAULT
                   for row in custody.custody_rows(custody.custody_root(ctx)))

"""#2756: an own task-store file the council sees only in part keeps its whole source.

``plan_task`` attaches a file of the caller's own ``task_drive``/``artifact_store`` as a head cut at
the per-item bound or as a selection, whose sha256/bytes describe the whole file. The exact bytes
that part and its hash were read from are retained at admission in the asking task's artifact store
(``store_actor_source_bytes``), and the ``task_source`` ref travels to the readers: each
participant's delivery, the ask row of the plan record and its source closure. Driven through the
real ``plan_task`` preparation, packet, ledger and record (fake substrate), the real task-drive
prune and child promotion, and the real readers."""
from __future__ import annotations

import hashlib
import os
import time

import pytest

from ouroboros import plan_record
from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
from ouroboros.tools.plan_evidence import EVIDENCE_PER_ITEM_BYTES
from tests.test_plan_task_tool import _own_store, _record, _slots, _user_text, _write
from tests.test_plan_task_tool import harness as _plan_task_harness

harness = _plan_task_harness


def _evidence(size: int) -> bytes:
    """Numbered lines, exactly ``size`` bytes (the reproduction's 56033 by default)."""
    lines = b"".join(f"{n:06d} planning evidence line\n".encode() for n in range(1, size // 30 + 2))
    return lines[:size - 1] + b"\n"


def test_a_part_the_council_saw_keeps_its_whole_source_through_the_task_drive_prune(harness):
    """A folderless task's own ``task_drive`` file of 56033 bytes reaches the council as its head and as a line
    selection, an ``artifact_store`` log as a selection, a small file whole. The head, the selections and the
    hashes were read from bytes now held as one ``task_source`` (one ref for both parts of the same bytes, none
    for the whole file); each delivery, the retained packets and the ask row name it. The agent then rewrites
    the file, the task completes and the ordinary prune removes its drive: the ref still reads the bytes the
    council saw part of, through the byte reader and through ``read_file`` as the delivery says."""
    from ouroboros.headless import prune_task_drives
    from ouroboros.review_ledger import read_source
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry

    ctx = harness.make_ctx(active_workspace=False)
    ctx.project_id = "p1"
    drv, art = _own_store(ctx, "task_drive"), _own_store(ctx, "artifact_store")
    big, log = _evidence(56033), b"first\r\nsecond, the selected line\r\nthird\r\n"
    assert len(big) == 56033 > EVIDENCE_PER_ITEM_BYTES
    (drv / "evidence.md").write_bytes(big)
    (drv / "small.md").write_bytes(b"seen whole\n")
    (art / "run.log").write_bytes(log)
    harness.state["slots"] = _slots(("api1", "m/a"), ("sess1", "cursor=grok", "session"))
    sub = harness.install({})
    locators = ["evidence.md", "evidence.md::lines=1500-1501", "small.md", f"{art / 'run.log'}::lines=2-2"]
    _write(ctx, ask="council", attach=locators)

    ask = _record(harness)["asks"][0]
    rows = {row["locator"]: row for row in ask["attached"]}
    assert list(rows) == locators
    held = rows["evidence.md"]["ref"]
    assert rows["evidence.md::lines=1500-1501"]["ref"] == held and "ref" not in rows["small.md"]
    assert (held["kind"], held["root"], held["size"], held["sha256"]) == (
        "task_source", "artifact_store", len(big), hashlib.sha256(big).hexdigest())
    assert rows["evidence.md"]["sha256"] == held["sha256"] and rows["evidence.md"]["bytes"] == len(big)
    log_ref = rows[f"{art / 'run.log'}::lines=2-2"]["ref"]
    assert (log_ref["size"], log_ref["sha256"]) == (len(log), hashlib.sha256(log).hexdigest())
    assert ask["omissions"] == [{"locator": "evidence.md", "reason": f"truncated_to_{EVIDENCE_PER_ITEM_BYTES}"}]

    request = sub.calls[0]["request"]
    file = task_artifact_dir_path(harness.drive, "task-1") / held["path"]
    for delivery in (_user_text(request.slot_messages["api1"][-1]["content"]), str(request.slot_session_tasks["sess1"])):
        assert big[:EVIDENCE_PER_ITEM_BYTES].decode() in delivery and "001501 planning evidence line" in delivery
        assert "001400 planning evidence line" not in delivery  # past the head, outside the selection
        for ref in (held, log_ref):
            assert f"read_file(root='artifact_store', path='{ref['path']}')" in delivery
        assert f"`{file}`" in delivery
    for participant in ask["participants"]:
        retained = read_source(harness.drive, "task-1", participant["delivery_ref"]).decode("utf-8")
        assert held["path"] in retained and log_ref["path"] in retained

    (drv / "evidence.md").write_bytes(b"rewritten after the ask\n")
    old = time.time() - 8 * 86400
    write_task_result(harness.drive, "task-1", "completed", result="done",
                      ts=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(old)))
    os.utime(drv, (old, old))
    report = prune_task_drives(harness.drive, retention_days=7, now=time.time(), live=lambda _task: False)
    assert [item["task_id"] for item in report["pruned"]] == ["task-1"] and not drv.exists()

    kept = plan_record.load_plan_record(harness.drive, "task-1")["asks"][0]["attached"]
    assert [row.get("ref") for row in kept] == [held, held, None, log_ref]
    assert read_actor_source_bytes(harness.drive, "task-1", held) == big
    assert read_actor_source_bytes(harness.drive, "task-1", log_ref) == log
    registry = ToolRegistry(repo_dir=harness.system, drive_root=harness.drive)
    registry.set_context(ctx)
    shown = registry.execute("read_file", {**held["read"]["arguments"], "start_line": 1400, "max_lines": 2})
    assert "001400 planning evidence line" in shown and "001401 planning evidence line" in shown, shown


@pytest.mark.parametrize("record", ["child_drive", "canonical_drive"])
def test_a_part_seen_in_a_child_drive_keeps_its_whole_source_through_promotion_and_removal(harness, record):
    """A forked child attaches a file of its execution drive's own ``task_drive``. Its plan record lives either in
    that drive (as in ``test_plan_record_sources_survive_real_child_promotion_and_cleanup``), where copy-back
    promotes the source the ask row names with the record's other sources, or on the canonical budget drive,
    where the source is kept beside it from the start. Once the drive is removed the parent reads the whole file
    the council saw the head of."""
    from ouroboros.headless import (copy_child_task_result, prepare_task_drive, remove_subagent_task_drive,
                                    retry_child_task_refs)
    from ouroboros.task_results import write_task_result

    parent = harness.drive
    child = prepare_task_drive(parent, "source", "empty")
    ctx = harness.make_ctx(task_id="source")
    ctx.drive_root = child
    if record == "canonical_drive":
        ctx.budget_drive_root = str(parent)
        ctx.task_metadata["budget_drive_root"] = str(parent)
    home = child if record == "child_drive" else parent
    harness.install({})
    raw = _evidence(EVIDENCE_PER_ITEM_BYTES + 4096)
    drv = _own_store(ctx, "task_drive")
    assert drv.is_relative_to(child)
    (drv / "draft.md").write_bytes(raw)
    _write(ctx, ask="council", attach=[str(drv / "draft.md")])
    [row] = plan_record.load_plan_record(home, "source")["asks"][0]["attached"]
    assert read_actor_source_bytes(home, "source", row["ref"]) == raw
    write_task_result(home, "source", "completed")
    if record == "child_drive":
        copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
        copied = retry_child_task_refs(parent, child, "source")
        assert copied["child_ref_promotion"]["status"] == "complete", copied["child_ref_promotion"]
    assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is True
    assert not child.exists()
    [promoted] = plan_record.load_plan_record(parent, "source")["asks"][0]["attached"]
    assert promoted == row
    assert read_actor_source_bytes(parent, "source", promoted["ref"]) == raw


def test_pending_council_keeps_session_source_loose_until_its_last_reply(harness):
    """A finished author can leave council readers running. Their advertised paths must stay
    openable without an Ouroboros tool; after the last reply normal addressed packing resumes."""
    from ouroboros import evidence_packer as packer, evidence_packs as packs, store
    from ouroboros.review_records import ReviewActorRecord
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import plan_ask

    ctx = harness.make_ctx(active_workspace=False)
    ctx.project_id = "p1"
    drv = _own_store(ctx, "task_drive")
    raw = _evidence(56033)
    (drv / "draft.md").write_bytes(raw)
    slots = _slots(("s1", "cursor=grok", "session"), ("s2", "cursor=grok", "session"))
    harness.state["slots"] = slots
    pending = {"status": "pending", "raw_text": "", "operation_state": "in_flight", "late_result_pending": True}
    sub = harness.install({slot.slot_id: pending for slot in slots})
    _write(ctx, ask="council", attach=["draft.md"])
    request = sub.calls[0]["request"]
    [attached] = _record(harness)["asks"][0]["attached"]
    file = task_artifact_dir_path(harness.drive, ctx.task_id) / attached["ref"]["path"]
    assert all(f"`{file}`" in str(request.slot_session_tasks[slot.slot_id]) for slot in slots)
    assert plan_record.gate_projection(_record(harness), cyber_pro=False)["allow"]
    write_task_result(harness.drive, ctx.task_id, "completed", result="author finished")
    with store.open_store(harness.drive, write=False) as txn:
        assert txn.row(ctx.task_id, "review_collection_pending", "plan_collection_pending") == (0, 1)
    before = packer.task_stamps(harness.drive, ctx.task_id)
    report = packer.run_pass(harness.drive)
    assert file.exists(), report  # plain filesystem read, as an external session reads it
    assert file.read_bytes() == raw and report["reading"] == 1 and report["packed"] == 0
    assert ctx.task_id in packer.keep_rows(harness.drive)

    for index, slot in enumerate(slots):
        since = packer.terminal_revision(harness.drive)
        actor = ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="ok",
                                  raw_text='{"prose":"Read it.","objections":[],"withdrawn":[]}',
                                  operation_id=f"op-{slot.slot_id}-1", operation_state="late_settled")
        plan_ask.settle_released_reply(ctx, request=request, task_id=ctx.task_id, actor=actor,
                                       slot=slot, settled_wave={}, roster_size=2)
        report = packer.run_pass(harness.drive, since=since)
        assert report["faults"] == 0
        if index == 0:
            assert file.exists() and report["packed"] == 0
        else:
            assert packer.task_stamps(harness.drive, ctx.task_id) != before
            assert not file.exists() and report["packed"] == 1
            assert ctx.task_id not in packer.keep_rows(harness.drive)
    assert packs.handle_bytes(harness.drive, ctx.task_id, attached["ref"]["path"]) == raw


def test_a_whole_source_that_could_not_be_kept_is_named_and_never_claimed(harness):
    """When the store refuses the whole source, the part still reaches the council with the full hash, and the
    absence is named (``source_unretained:<error>``) in the delivery's omissions, the ask row and the author's
    read; no ref and no read instruction claim a source the store does not hold."""
    from ouroboros import artifacts

    real = artifacts.store_actor_source_bytes

    def refusing(root, task_id, **kw):
        if kw.get("source_id") == "plan-evidence":
            raise OSError(28, "No space left on device")
        return real(root, task_id, **kw)

    harness.monkeypatch.setattr(artifacts, "store_actor_source_bytes", refusing)
    ctx = harness.make_ctx()
    drv = _own_store(ctx, "task_drive")
    raw = _evidence(EVIDENCE_PER_ITEM_BYTES + 100)
    (drv / "big.md").write_bytes(raw)
    locator = str(drv / "big.md")
    sub = harness.install({})
    text, _result = _write(ctx, ask=["s1"], attach=[locator])
    ask = _record(harness)["asks"][0]
    assert ask["attached"] == [{"locator": locator, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}]
    assert ask["omissions"] == [{"locator": locator, "reason": f"truncated_to_{EVIDENCE_PER_ITEM_BYTES}"},
                                {"locator": locator, "reason": "source_unretained:OSError"}]
    packet = _user_text(sub.calls[0]["request"].slot_messages["s1"][-1]["content"])
    assert raw[:EVIDENCE_PER_ITEM_BYTES].decode() in packet
    assert f"| {locator} | source_unretained:OSError |" in packet and "plan-evidence-" not in packet
    # The part attached; its cut and the unkept whole source stay beside it, never as "not attached".
    assert (f"evidence attached (prepared for the participants): {locator} (in part; source {len(raw)} bytes; "
            f"truncated_to_{EVIDENCE_PER_ITEM_BYTES}; source_unretained:OSError)") in text and "evidence not attached" not in text
    held = task_artifact_dir_path(harness.drive, "task-1") / "source_handles" / "context_checkpoints"
    assert not list(held.glob("plan-evidence-*"))

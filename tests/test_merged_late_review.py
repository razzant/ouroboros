"""Late review facts join actual merge receipts, mailbox delivery and passive card history."""
import asyncio
import copy
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from urllib.parse import quote, urlencode

import pytest

from ouroboros import dialogue_store, merge_receipts, review_ledger, review_pause
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_store import patch_document
from tests.group_run import run_in_group
from ouroboros.tools import github
from tests import test_pr_merge_receipts as merges, test_receptive_commit_review as receptive
from tests.test_review_seat_outcome import _body
from tests.test_task_review_sources import client_for

world = merges.world
author = receptive.author
pytestmark = pytest.mark.serial


def _record(root, task_id, record_id="late-merge-review", *, head=merges.HEAD):
    return review_ledger.write_record(root, review_ledger.ReviewLedgerRecord(
        record_id=record_id, task_id=task_id, state="pending",
        subject={"kind": "base..head", "head": head, "base": merges.BASE},
        verdict={"aggregate": "NOT_PERFORMED"}, panel={"seats": [{"seat_id": "critic"}]}))


def _fail(root, record):
    source = review_ledger.retain_text_source(root, record["task_id"], record_id=record["record_id"],
        seat_id="critic", role="response", part="change", text="# Review\nFAIL: rollback loses data.\nREPORT-END")
    return review_ledger.revise_record(root, record["record_id"], lambda now: {
        **now, "state": "settled", "verdict": {"aggregate": "FAIL", "per_row": {"critic": "FAIL"}},
        "rows": [{"seat_id": "critic", "status": "responded", "source_refs": [source]}]})


@pytest.mark.parametrize("case", ["merged", "changed_head", "queued", "other_record", "absent"])
def test_only_an_observed_merge_under_the_same_record_is_selected_without_writes(world, case):
    record = _record(world.root, "merge-task", head="e" * 40 if case == "changed_head" else merges.HEAD)
    if case != "absent":
        world.gh.merge = "queued" if case == "queued" else "ok"
        merges._merge(world, review_record_id=record["record_id"])
    before = load_task_result(world.root, "merge-task")
    asked = "another-review" if case == "other_record" else record["record_id"]
    found = merge_receipts.merged_under_record(world.root, "merge-task", asked)
    assert len(found) == (1 if case in {"merged", "changed_head"} else 0)
    if found:
        assert found[0]["outcome"]["merge_sha"] == merges.MERGE
        if case == "changed_head":
            words = merge_receipts.review_settlement_words(found, "7", 2)[0]
            assert "changes_after_review" in words and "earlier code" in words
        found[0]["number"] = 999  # returned values cannot mutate the stored receipt
    assert load_task_result(world.root, "merge-task") == before


def _history(root, chat_id=7):
    return json.loads(asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"chat_id": str(chat_id)}))).body)


@pytest.mark.parametrize("case", ["late_fail", "no_source", "pass", "pending", "other_record", "queued", "fail_at_merge"])
def test_card_history_reads_later_fail_with_a_working_saved_report_link_and_no_publication(world, case):
    record = _record(world.root, "merge-task")
    if case == "fail_at_merge":
        record = _fail(world.root, record)
    world.gh.merge = "queued" if case == "queued" else "ok"
    merges._merge(world, review_record_id=record["record_id"])
    event = world.ctx.pending_events[0]
    dialogue_store.append(world.root, "progress", {**event, **event["progress_meta"], "type": "task_progress"})
    before = _history(world.root)
    if case == "other_record":
        _fail(world.root, _record(world.root, "merge-task", "unrelated-review"))
    elif case != "fail_at_merge":
        _fail(world.root, record)
        if case == "no_source":
            review_ledger.revise_record(world.root, record["record_id"], lambda now: {**now,
                "rows": [{**seat, "source_refs": []} for seat in now["rows"]]})
        if case in {"pass", "pending"}:
            review_ledger.revise_record(world.root, record["record_id"], lambda now: {**now,
                "state": "pending" if case == "pending" else "settled", "verdict": {"aggregate": "NOT_PERFORMED" if case == "pending" else "PASS"}})
    receipts, calls, events = copy.deepcopy(merges._receipts(world)), list(world.gh.calls), list(world.ctx.pending_events)
    after = _history(world.root)
    old, = [row for row in before["messages"] if row.get("card_row_id")]
    row, = [row for row in after["messages"] if row.get("card_row_id")]
    assert len(after["messages"]) == len(before["messages"])
    assert all(row[key] == old[key] for key in ("history_id", "card_row_id", "card_row_revision", "ts"))
    if case == "late_fail":
        assert row["text"].startswith(old["text"] + "\nIts later review: FAIL")
        selected = row["late_evidence"]["review_source"]
        assert selected["record_id"] == record["record_id"] and selected["seat_id"] == "critic"
        path = "/".join(quote(selected[key], safe="") for key in ("task_id", "record_id", "seat_id"))
        task, rec, seat = path.split("/")
        url = f"/api/tasks/{task}/reviews/{rec}/{seat}?" + urlencode({key: selected[key] for key in ("role", "part", "sha256")})
        with client_for(world.root) as client:
            response = client.get(url)
        assert response.status_code == 200 and response.text.endswith("REPORT-END")
        fixture = world.root / "late-merge-projection.json"
        fixture.write_text(json.dumps({"before": old, "after": row}), encoding="utf-8")
        result = run_in_group([shutil.which("node"), "--test", "--test-name-pattern=retained merge row refreshes",
                              "web/tests/chat_card_row_placement.test.js"], cwd=Path(__file__).resolve().parents[1],
                             env={**os.environ, "LATE_MERGE_PROJECTION": str(fixture)}, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    elif case == "no_source":
        assert row["text"] == old["text"] + "\nIts later review: FAIL."
        assert "review_source" not in row["late_evidence"]
    else:
        assert row["text"] == old["text"]
    assert merges._receipts(world) == receipts and world.gh.calls == calls and world.ctx.pending_events == events


@pytest.mark.parametrize("ended_first", [False, True])
def test_late_fail_names_the_merge_in_mail_and_the_frozen_room_mark(author, monkeypatch, ended_first):
    a = author
    chat = a.reviewer.chat
    def fail_after_real_dispatch(*args, **kwargs):
        _answer, usage = chat(*args, **kwargs)
        return {"content": _body(True)}, usage
    monkeypatch.setattr(a.reviewer, "chat", fail_after_real_dispatch)
    wait, _attempt = receptive._leave_wave_by_mail(a)
    from ouroboros.projects_registry import get_project
    room = str(get_project(a.data, "alpha")["chat_id"])
    write_task_result(a.data, receptive.TASK, "running", chat_id=int(room))
    # A merge may happen while this paid review still runs. The real tool records its own facts.
    fake = merges.FakeGh(a.data)
    monkeypatch.setattr(github, "_gh_run", fake)
    ctx = SimpleNamespace(task_id=receptive.TASK, drive_root=a.data, task_metadata={}, repo_dir=a.repo,
                          current_chat_id=int(room), pending_events=[])
    assert "merge: merged" in github._pr_merge(ctx, number=7, method="squash", expected_head_sha=merges.HEAD,
                                              review_record_id=wait["record_id"])
    receipt, = merge_receipts.task_receipts(a.data, receptive.TASK)
    chat_rows = lambda room: [(r.get("history_id"), r["text"]) for r in _history(a.data, room)["messages"]]
    before_room, before_main = chat_rows(room), chat_rows(1)
    if ended_first:
        write_task_result(a.data, receptive.TASK, "completed", result="Merged before review settled.")
    a.reviewer.release()
    if not ended_first:
        notice, = receptive.until(lambda: receptive._notices(a))
        assert "PR #7" in notice["text"] and merges.MERGE in notice["text"] and f"Room {room}" in notice["text"]
        assert "reads FAIL" in notice["text"] and "That review call committed or applied nothing" in notice["text"]
        write_task_result(a.data, receptive.TASK, "completed", result="Read the late review.")
        from ouroboros.late_review_marks import replay_owed_late_reviews
        replay_owed_late_reviews(a.data)
    mark, = receptive.until(lambda: receptive._late_marks(a.data))
    assert mark["room_id"] == room and "PR #7" in mark["text"] and "Its record now reads FAIL" in mark["text"]
    frozen = read_actor_source_bytes(a.data, receptive.TASK, mark["target_ref"])
    source = json.loads(frozen)
    assert source["merged_under_record"] == [receipt] and source["room_id"] == room
    assert source["review_revision"] > receipt["review"]["record"]["revision"]
    # Publication/readback enrichment and later routing cannot change this settlement's identity or room.
    patch_document(a.data, receptive.TASK, lambda row: {**row, "chat_id": 1,
        "merge_receipts": [{**r, "revision": r["revision"] + 1, "number": 88} for r in row["merge_receipts"]]})
    assert review_pause.ended_left_review(receptive._left_record(a, wait)) == "unchanged"
    assert receptive._late_marks(a.data) == [mark]
    assert read_actor_source_bytes(a.data, receptive.TASK, mark["target_ref"]) == frozen
    assert chat_rows(room) == before_room and chat_rows(1) == before_main, "the late review adds no chat row"
    assert len(ctx.pending_events) == 1 and len([c for c in fake.calls if c[:2] == ["pr", "merge"]]) == 1


def test_failed_publication_replays_the_frozen_source_and_room_before_reading_new_merge_facts(world, monkeypatch):
    from ouroboros import late_review_marks as late

    record = _record(world.root, "merge-task")
    merges._merge(world, review_record_id=record["record_id"])
    settled = _fail(world.root, record)
    write_task_result(world.root, "merge-task", "completed", chat_id=1)
    left = {"surface": "commit", "task_id": "merge-task", "record_id": record["record_id"], "retry_key": "left",
            "drive_root": str(world.root), "attempt": 1}
    frozen = review_pause._left_review_mark(left, None, [], "", settled)
    with monkeypatch.context() as fail:
        fail.setattr(late, "_publish", lambda root, mark_id, facts: late.LateReviewMark("failed", mark_id))
        assert late.publish_late_review(world.root, frozen).status == "failed"
    patch_document(world.root, "merge-task", lambda row: {**row, "chat_id": 777, "merge_receipts": []})
    monkeypatch.setattr(review_pause, "_left_seats", lambda *_: pytest.fail("a settled publication was recomposed"))
    assert review_pause.ended_left_review(left) == "saved"
    mark, = receptive._late_marks(world.root)
    assert mark["room_id"] == "1" and mark["target_ref"] == {**frozen["source_ref"], "task_id": "merge-task"}
    source = json.loads(read_actor_source_bytes(world.root, "merge-task", mark["target_ref"]))
    assert len(source["merged_under_record"]) == 1

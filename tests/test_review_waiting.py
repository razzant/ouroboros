"""Selected readiness is a peek; delivery and restart preserve exact cursors."""
import threading
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as ledger, review_waiting as waiting
from ouroboros.task_results import write_task_result
from tests.test_review_ledger import _raw, BOTH
from tests.test_review_ledger_settlement import _record


def test_helper_selection_peeks_only_and_pins_ownership(tmp_path, monkeypatch):
    from ouroboros import model_sleep, owner_wait

    ctx = SimpleNamespace(task_id="parent", drive_root=tmp_path, budget_drive_root=None)
    write_task_result(tmp_path, "child", "running", parent_task_id="parent")
    write_task_result(tmp_path, "peer", "running", parent_task_id="other")
    runs = {name: SimpleNamespace(run_id=name, review_owned=review, terminal_state="", settled=False)
            for name, review in (("own-run", False), ("review-run", True))}
    monkeypatch.setattr(model_sleep, "_owned_runs", lambda *_args, **_kw: runs)
    monkeypatch.setattr(owner_wait, "_wait_entries", lambda *_a: pytest.fail("peek materialized mail"))
    default = waiting.select_helpers(ctx)
    assert default["selection"]["tasks"] == default["selection"]["runs"] == []
    state = waiting.select_helpers(ctx, {"tasks": ["child"], "runs": ["own-run"]})
    assert state["selection"]["tasks"] == ["child"] and state["selection"]["runs"] == ["own-run"]
    write_task_result(tmp_path, "peer", "completed", parent_task_id="other")
    write_task_result(tmp_path, "new-child", "completed", parent_task_id="parent")
    assert not waiting.ready_helpers(ctx, state)
    write_task_result(tmp_path, "child", "completed", parent_task_id="parent")
    assert [row["id"] for row in waiting.ready_helpers(ctx, state)] == ["task:child"]
    assert state["seen_helpers"] == []
    assert waiting.ready_helpers(ctx, state)  # readiness never acknowledges
    state["seen_helpers"] = ["task:child"]
    assert not waiting.ready_helpers(ctx, state)
    runs["own-run"].settled = True
    assert waiting.ready_helpers(ctx, state)[0]["id"] == "run:own-run"


def test_delivery_cursor_waits_for_transcript_and_checkpoint(tmp_path, monkeypatch):
    from ouroboros import working_checkpoint
    from ouroboros.late_review_marks import owe_after_end, OWED_SET
    from ouroboros.store import open_store
    from ouroboros.tools.tool_result import _install_tool_result_sidecar, _restore_tool_result_sidecar, _published_tool_result

    record = _record(tmp_path)
    ctx = SimpleNamespace(task_id="task-1", drive_root=tmp_path, budget_drive_root=None,
                          _review_wait_selection={"selection": {}, "seen_helpers": [], "seen_outcomes": [], "duty_id": "owed"})
    owe_after_end(tmp_path, "owed", {"left_review": {"task_id": "task-1", "record_id": record["record_id"]}})
    waiting.save_wait(ctx, record["record_id"], ctx._review_wait_selection)
    record = ledger.settle_from_collected(tmp_path, record["record_id"], [_raw("a", "a", parts=BOTH)])
    token = _install_tool_result_sidecar(ctx, object())
    try:
        waiting.publish_delivery(ctx, "answer", record)
        receipt = _published_tool_result(ctx, None).meta["review_delivery"]
    finally:
        _restore_tool_result_sidecar(token)
    assert ledger.load_record(tmp_path, record["record_id"])["waiting"]["seen_outcomes"] == []
    commits = []
    monkeypatch.setattr(working_checkpoint, "defer_content_ack", lambda _ctx, callback: commits.append(callback))
    ctx._left_review_wait = {"record_id": record["record_id"], "done": True, "noticed": False, "notice": "settled"}
    waiting.acknowledge_delivery(ctx, receipt)
    assert ctx._review_wait_selection["seen_outcomes"]
    assert ctx._left_review_wait["done"] and ctx._left_review_wait["notice"] == "settled"
    assert ledger.load_record(tmp_path, record["record_id"])["waiting"]["seen_outcomes"] == []
    commits[0]()
    saved = ledger.load_record(tmp_path, record["record_id"])["waiting"]
    assert saved["seen_outcomes"] == receipt["outcomes"]
    with open_store(tmp_path, write=True) as txn:
        assert txn.get_row(OWED_SET, "owed")["left_review"]["seen_outcomes"] == receipt["outcomes"]
        txn.delete_row(OWED_SET, "owed")
    commits[0]()
    with open_store(tmp_path, write=False) as txn:
        assert txn.get_row(OWED_SET, "owed") is None, "an ACK cannot recreate a discharged room duty"
    restored = SimpleNamespace(**vars(ctx))
    restored._left_review_wait = saved
    assert waiting.select_helpers(restored)["seen_outcomes"] == receipt["outcomes"]
    restored._left_review_wait = {"record_id": record["record_id"]}
    assert waiting.select_helpers(restored)["selection"] == {}, "legacy waits mean the whole wave"


def _result_record(root, settled=("a", "b")):
    """A result wave's record (seats a and b), with ``settled`` seats answered, as its unread duty names it."""
    record = ledger.revise_record(root, _record(root)["record_id"], lambda current: {**current, "surface": "result"})
    for seat in settled:
        record = ledger.settle_from_collected(root, record["record_id"], [_raw(seat, seat, parts=BOTH, operation_id=seat)])
    return record


def _owes(ctx, record, left="unread"):
    from ouroboros.review_pause import owe_unwaited_review

    assert owe_unwaited_review(ctx, surface="result", retry_key=f"key-{record['record_id']}",
                               record_id=record["record_id"], left=left)


def _duties(root):
    from ouroboros.late_review_marks import OWED_SET
    from ouroboros.store import open_store

    with open_store(root, write=False) as txn:
        return sorted((row["left_review"]["record_id"], row["left_review"]["left"]) for row in txn.set_rows(OWED_SET).values())


def _receipt(ctx, record, rendered_rows=None):
    """The receipt the real publication carries for a response rendering ``rendered_rows``."""
    from ouroboros.tools.tool_result import _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar

    token = _install_tool_result_sidecar(ctx, object())
    try:
        waiting.publish_delivery(ctx, "answer", record, rendered_rows=rendered_rows)
        return _published_tool_result(ctx, None).meta["review_delivery"]
    finally:
        _restore_tool_result_sidecar(token)


def _author(root, task_id="task-1"):
    return SimpleNamespace(task_id=task_id, drive_root=root, budget_drive_root=None)


def test_a_saved_whole_receipt_ends_its_own_records_unread_duty_and_nothing_else(tmp_path):
    from ouroboros.working_checkpoint import flush_content_acks

    ctx = _author(tmp_path)
    own, other, unwaited = _result_record(tmp_path), _result_record(tmp_path), _result_record(tmp_path)
    _owes(ctx, own), _owes(ctx, other), _owes(ctx, unwaited, "unwaited")
    everything = sorted([(own["record_id"], "unread"), (other["record_id"], "unread"), (unwaited["record_id"], "unwaited")])
    waiting.acknowledge_delivery(ctx, _receipt(ctx, own, own["rows"]))
    assert _duties(tmp_path) == everything, "a transcript insertion alone ends nothing: no saved state holds it yet"
    flush_content_acks(ctx)
    assert _duties(tmp_path) == sorted(row for row in everything if row[0] != own["record_id"])
    kept = ledger.load_record(tmp_path, own["record_id"])
    assert (kept["state"], kept["rows"], kept["verdict"]) == ("settled", own["rows"], own["verdict"])  # evidence stays
    assert kept["waiting"]["seen_outcomes"]
    # Another kind of duty keeps its own lifetime, even under a whole receipt of its own record.
    waiting.acknowledge_delivery(ctx, _receipt(ctx, unwaited, unwaited["rows"]))
    flush_content_acks(ctx)
    assert (unwaited["record_id"], "unwaited") in _duties(tmp_path)
    # Delivered again: nothing left to end, and nothing recreated.
    waiting.acknowledge_delivery(ctx, _receipt(ctx, own, own["rows"]))
    flush_content_acks(ctx)
    assert _duties(tmp_path) == sorted(row for row in everything if row[0] != own["record_id"])


@pytest.mark.parametrize("short", ["helper_notice", "report_without_rows", "another_records_receipt"])
def test_a_receipt_short_of_whole_ends_no_unread_duty(tmp_path, short):
    """Cursor ids prove nothing: a helper notice and a report that renders no rows advance them all the same."""
    from ouroboros.working_checkpoint import flush_content_acks

    ctx = _author(tmp_path)
    record, before = _result_record(tmp_path), None
    if short == "a_seat_settled_after_the_render":
        record = _result_record(tmp_path, settled=("a",))
        before = record["rows"]
        record = ledger.settle_from_collected(tmp_path, record["record_id"], [_raw("b", "b", parts=BOTH, operation_id="b")])
    elif short == "wave_still_pending":
        record = _result_record(tmp_path, settled=("a",))
    _owes(ctx, record)
    tokens = [waiting._outcome_token(identity) for identity in waiting.outcome_ids(record)]
    receipt = {
        "helper_notice": {"record_id": record["record_id"], "helpers": [], "outcomes": tokens},
        "report_without_rows": _receipt(ctx, record),
        "a_seat_settled_after_the_render": _receipt(ctx, record, before),
        "wave_still_pending": _receipt(ctx, record, record["rows"]),
        "another_author": _receipt(ctx, record, record["rows"]),
    }.get(short)
    author = _author(tmp_path, "foreign-author") if short == "another_author" else ctx
    if short == "another_records_receipt":
        other = _result_record(tmp_path)
        receipt = _receipt(ctx, other, other["rows"])
    waiting.acknowledge_delivery(author, receipt)
    flush_content_acks(author)
    assert _duties(tmp_path) == [(record["record_id"], "unread")]
    if short in {"helper_notice", "report_without_rows"}:
        assert set(tokens) <= set(ledger.load_record(tmp_path, record["record_id"])["waiting"]["seen_outcomes"])


def test_an_outcome_that_changes_before_the_checkpoint_keeps_the_unread_duty(tmp_path):
    """Whole when acknowledged, short when saved: the check reads the record as the checkpoint finds it."""
    from ouroboros.working_checkpoint import flush_content_acks

    ctx = _author(tmp_path)
    record = _result_record(tmp_path)
    _owes(ctx, record)
    waiting.acknowledge_delivery(ctx, _receipt(ctx, record, record["rows"]))

    def later_answer(current):
        rows = [dict(row) for row in current["rows"]]
        rows[0]["status"] = "continued"  # a new state of that seat's outcome (``outcome_states``)
        return {**current, "rows": rows}

    ledger.revise_record(tmp_path, record["record_id"], later_answer)
    flush_content_acks(ctx)
    assert _duties(tmp_path) == [(record["record_id"], "unread")]


def test_only_a_saved_state_makes_a_whole_receipt_end_the_unread_duty(tmp_path, monkeypatch):
    """The existing owner decides: a failed ready save keeps the acknowledgement pending, a failure exit acknowledges
    nothing, and the next save that holds it ends the duty."""
    from ouroboros import working_checkpoint

    ctx = _author(tmp_path)
    record = _result_record(tmp_path)
    _owes(ctx, record)
    waiting.acknowledge_delivery(ctx, _receipt(ctx, record, record["rows"]))
    limit = SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), messages=[{"role": "tool"}], round_idx=1, owner_msg_seen=set())
    monkeypatch.setattr(working_checkpoint, "save_or_log", lambda *_a: False)  # the write failed; it logged why
    working_checkpoint.save_ready(limit)
    assert _duties(tmp_path) == [(record["record_id"], "unread")] and ctx._pending_content_acks
    monkeypatch.setattr(working_checkpoint, "save_or_log", lambda *_a: True)
    try:
        raise RuntimeError("the loop's failure exit")
    except RuntimeError:
        working_checkpoint.save_terminal(limit)  # as its ``finally`` runs: saves and acknowledges nothing
    assert _duties(tmp_path) == [(record["record_id"], "unread")] and ctx._pending_content_acks
    working_checkpoint.save_terminal(limit)  # a returned result: saved, then acknowledged
    assert _duties(tmp_path) == [] and not ctx._pending_content_acks


def test_unreadable_record_cannot_acknowledge_delivery(tmp_path):
    record = _record(tmp_path)
    ctx = SimpleNamespace(task_id="task-1", drive_root=tmp_path, budget_drive_root=None)
    ledger.record_path(tmp_path, record["record_id"]).write_text("{broken")
    waiting.acknowledge_delivery(ctx, {"record_id": record["record_id"], "outcomes": ["wave:operation:outcome"]})
    assert not getattr(ctx, "_pending_content_acks", None)
    assert not getattr(ctx, "_review_wait_selection", None)


@pytest.mark.parametrize("author", ["task-1", "foreign-author"])
def test_delivery_cursor_accepts_only_its_author(tmp_path, author):
    from ouroboros.working_checkpoint import flush_content_acks

    record = _record(tmp_path)
    ctx = SimpleNamespace(task_id=author, drive_root=tmp_path, budget_drive_root=None)
    waiting.acknowledge_delivery(ctx, {"record_id": record["record_id"], "outcomes": ["wave:operation:outcome"]})
    assert bool(getattr(ctx, "_review_wait_selection", None)) is (author == "task-1")
    assert bool(getattr(ctx, "_pending_content_acks", None)) is (author == "task-1")
    flush_content_acks(ctx)
    saved = ledger.load_record(tmp_path, record["record_id"])
    assert bool(saved.get("waiting")) is (author == "task-1")


@pytest.mark.parametrize("paused", [False, True])
def test_pause_precedes_a_ready_helper(tmp_path, monkeypatch, paused):
    import contextlib
    from ouroboros import review_pause, owner_pause

    entry = SimpleNamespace(author_left_for="", event=threading.Event(), episode={},
                            started_at=1, pause_before_launch=False, retry_state={})
    monkeypatch.setattr(review_pause, "author_paused", lambda _op: paused)
    monkeypatch.setattr(review_pause, "author_hears_mail", lambda: False)
    monkeypatch.setattr(review_pause, "_author_coordinates", lambda _op: (str(tmp_path), "parent", "parent"))
    monkeypatch.setattr(owner_pause, "launch_lock", lambda *_a: contextlib.nullcontext())
    monkeypatch.setattr(waiting, "wake_reason", lambda *_a, **_kw: "task:child:completed")
    with review_pause.receptive_author({"drive_root": tmp_path, "task_id": "parent"}):
        assert review_pause.author_detaches(SimpleNamespace(wait=None), [entry])
    assert entry.pause_before_launch is paused
    assert entry.author_left_for == ("" if paused else "helper_result")


@pytest.mark.parametrize("lost", [False, True])
def test_ended_author_does_not_turn_remote_pending_into_api_loss(tmp_path, lost):
    from ouroboros.review_pause import _settle_left_record

    record = _record(tmp_path)
    seats = [{"slot_id": name, "parts": list(BOTH), "operation_id": name, "status": "error",
              "operation_state": "custody_lost" if lost else "in_flight", "late_result_pending": True}
             for name in ("a", "b")]
    settled = _settle_left_record({"record_id": record["record_id"], "drive_root": str(tmp_path)}, seats, final=True)
    assert settled["state"] == ("settled" if lost else "pending")
    assert settled["verdict"]["aggregate"] != "PASS"


def test_restored_notice_retains_exact_reference_selection_and_seen(tmp_path):
    from ouroboros.review_pause import wake_restored_left_review, PROCESS_TOKEN

    reference = {"record_id": "rl-original", "task_id": "parent", "surface": "change"}
    state = {"owner": "ended-process", "review_reference": reference, "selection": {"tasks": ["child"]},
             "seen_helpers": ["task:child"], "seen_outcomes": ["wave:operation:outcome"], "msg_id": "original"}
    ctx = SimpleNamespace(task_id="parent", drive_root=tmp_path, _left_review_wait=state)
    assert wake_restored_left_review(ctx)
    assert ctx._left_review_wait["review_reference"] == reference
    assert ctx._left_review_wait["seen_helpers"] == ["task:child"]
    assert ctx._left_review_wait["seen_outcomes"] == ["wave:operation:outcome"]
    assert ctx._left_review_wait["owner"] == PROCESS_TOKEN
    assert not wake_restored_left_review(ctx), "same restoration must not notify twice"


def test_successor_cursor_separates_predecessor_history_and_blocks_past_listeners(tmp_path):
    from ouroboros.owner_continue import binding_sha
    from ouroboros.task_store import patch_document
    from ouroboros.working_checkpoint import flush_content_acks

    repo = tmp_path / "repo"
    repo.mkdir()
    other_repo = tmp_path / "other-repo"
    other_repo.mkdir()
    for tid, status in (("task-1", "completed"), ("middle", "completed"), ("successor", "running"), ("stranger", "running")):
        write_task_result(tmp_path, tid, status, chat_id=1)
    for pred, succ in (("task-1", "middle"), ("middle", "successor")):
        binding = {"successor_task_id": succ}
        patch_document(tmp_path, pred, lambda row, b=binding, s=succ: {**row, "continued_by": {
            "successor_task_id": s, "binding": b, "binding_sha256": binding_sha(b), "state": "admitted"}})

    record = ledger.revise_record(tmp_path, _record(tmp_path)["record_id"], lambda cur: {
        **cur, "surface": "commit_gate", "review_wave_id": "wave-1",
        "subject": {**(cur.get("subject") or {}), "root": str(repo)}})
    record = ledger.settle_from_collected(tmp_path, record["record_id"], [
        _raw("a", "a", parts=BOTH, operation_id="a",
             outcome={"outcome_id": "out-a", "status": "responded", "operation_id": "a", "slot_id": "a", "additional": False}),
        _raw("b", "b", parts=BOTH, operation_id="b",
             outcome={"outcome_id": "out-b0", "status": "responded", "operation_id": "b", "slot_id": "b", "additional": False})])
    record_id = record["record_id"]
    pin_a = {"wave_id": record["review_wave_id"], "operation_id": "a", "slot_id": "a"}
    pin_b_old = {"wave_id": record["review_wave_id"], "operation_id": "b", "slot_id": "b"}

    # Legacy wait shape (no waiting.task_id / outcome_states_by_task, raw pre-token seen_outcomes) belongs to record.task_id (P), never S.
    write_task_result(tmp_path, "p-child", "completed", parent_task_id="task-1")
    p_ctx = SimpleNamespace(task_id="task-1", drive_root=tmp_path, budget_drive_root=None, repo_dir=repo)
    s_ctx = SimpleNamespace(task_id="successor", drive_root=tmp_path, budget_drive_root=None, repo_dir=repo)
    ledger.revise_record(tmp_path, record_id, lambda cur: {**cur, "waiting": {
        "record_id": record_id, "duty_id": "p-duty",
        "selection": {"record_id": record_id, "reviewers": [pin_a, pin_b_old], "tasks": ["p-child"]},
        "seen_helpers": ["task:p-child"], "seen_outcomes": waiting.outcome_ids(record),
        "seen_outcome_states": waiting.outcome_states(record)}})
    legacy = ledger.load_record(tmp_path, record_id)
    assert "task_id" not in legacy["waiting"]
    assert not waiting.review_terminals(p_ctx, {"record_id": record_id, "reviewers": [pin_a]}, seen=legacy["waiting"]["seen_outcomes"])
    assert waiting.saved_selection(p_ctx, record_id)["selection"]["tasks"] == ["p-child"]
    assert waiting.saved_selection(p_ctx, record_id)["seen_outcomes"]
    assert waiting.received_whole(legacy, legacy) and waiting.received_whole(legacy, legacy, task_id="task-1")
    assert not waiting.received_whole(legacy, legacy, task_id="successor")

    # Supersede seat b -> b-2: verify S's first call against BOTH the legacy wait as-is (no waiting.task_id) and explicit task_id=P.
    ledger.revise_record(tmp_path, record_id, lambda cur: {
        **cur, "state": "pending",
        "rows": [{**row, "operation_id": "b-2", "status": "pending", "operation_state": "in_flight",
                  "superseded_operation_ids": ["b"]} if row["seat_id"] == "b" else row for row in cur["rows"]]})
    assert "task_id" not in ledger.load_record(tmp_path, record_id)["waiting"]
    saved_for_s = waiting.saved_selection(s_ctx, record_id)
    assert saved_for_s["seen_outcomes"] == [] and saved_for_s["seen_helpers"] == []
    assert saved_for_s["selection"]["tasks"] == []
    assert [pin["operation_id"] for pin in saved_for_s["selection"]["reviewers"]] == ["a", "b-2"]
    for prior_wait in (dict(ledger.load_record(tmp_path, record_id)["waiting"]),
                       {**ledger.load_record(tmp_path, record_id)["waiting"], "task_id": "task-1"}):
        s_ctx._left_review_wait = prior_wait
        selected = waiting.select_helpers(s_ctx)
        assert selected["seen_outcomes"] == [] and selected["seen_helpers"] == []
        assert selected["selection"]["tasks"] == []
        assert [pin["operation_id"] for pin in selected["selection"]["reviewers"]] == ["a", "b-2"]
    s_ctx._left_review_wait = dict(ledger.load_record(tmp_path, record_id)["waiting"])
    assert _receipt(s_ctx, ledger.load_record(tmp_path, record_id), [record["rows"][0]])["helpers"] == []

    # S sees seat a once over the untouched legacy wait, then S's ACK records S's own cursor and archives P's seen_outcome_states.
    ready = waiting.review_terminals(s_ctx, selected["selection"], seen=selected["seen_outcomes"])
    assert [item["slot_id"] for item in ready] == ["a"]
    # Another record, or a pin outside this record's wave, transfers no predecessor lookup.
    assert not waiting.review_terminals(s_ctx, {"record_id": "rl-synthetic-other", "reviewers": [pin_a]})
    assert not waiting.review_terminals(s_ctx, {"record_id": record_id, "reviewers": [{**pin_a, "wave_id": "synthetic-wave"}]})
    waiting.acknowledge_delivery(s_ctx, _receipt(s_ctx, ledger.load_record(tmp_path, record_id), [record["rows"][0]]))
    flush_content_acks(s_ctx)
    after_s = ledger.load_record(tmp_path, record_id)
    assert after_s["waiting"]["task_id"] == "successor" and after_s["waiting"]["attempt_task_id"] == "task-1"
    assert after_s["waiting"]["outcome_states_by_task"]["task-1"] == sorted(legacy["waiting"]["seen_outcome_states"])
    assert "duty_id" not in after_s["waiting"]
    assert not waiting.review_terminals(s_ctx, selected["selection"], seen=after_s["waiting"]["seen_outcomes"])

    # Past listeners (P and M), stranger and another checkout cannot read or overwrite S's cursor.
    m_ctx = SimpleNamespace(task_id="middle", drive_root=tmp_path, budget_drive_root=None, repo_dir=repo)
    stranger = SimpleNamespace(task_id="stranger", drive_root=tmp_path, budget_drive_root=None, repo_dir=repo)
    other_checkout = SimpleNamespace(task_id="successor", drive_root=tmp_path, budget_drive_root=None, repo_dir=other_repo)
    for blocked in (p_ctx, m_ctx, stranger, other_checkout):
        assert not waiting.review_terminals(blocked, {"record_id": record_id, "reviewers": [pin_a]})
        assert waiting.save_wait(blocked, record_id, {"task_id": blocked.task_id, "seen_outcomes": ["tampered"]}) is None
        waiting.acknowledge_delivery(blocked, {"record_id": record_id, "outcomes": ["tampered"]})
        flush_content_acks(blocked)
    assert ledger.load_record(tmp_path, record_id)["waiting"]["task_id"] == "successor"
    assert "tampered" not in ledger.load_record(tmp_path, record_id)["waiting"]["seen_outcomes"]

    # Historical whole-outcome receipt remains per task: what P saw stays P's, and what S saw never counts as P's.
    settled = ledger.settle_from_collected(tmp_path, record_id, [
        _raw("b", "b", parts=BOTH, operation_id="b-2",
             outcome={"outcome_id": "out-b", "status": "responded", "operation_id": "b-2", "slot_id": "b", "additional": False})])
    assert not waiting.received_whole(settled, settled, task_id="task-1")
    waiting.acknowledge_delivery(s_ctx, _receipt(s_ctx, settled, settled["rows"]))
    flush_content_acks(s_ctx)
    settled_after = ledger.load_record(tmp_path, record_id)
    assert waiting.received_whole(settled_after, settled_after, task_id="successor")
    assert not waiting.received_whole(settled_after, settled_after, task_id="task-1")


def _listener_chain(tmp_path):
    """P -> M -> S -> N as the owner's Continue records it, a commit record of P on a repository, and S's context."""
    from ouroboros.owner_continue import binding_sha
    from ouroboros.task_store import patch_document

    repo = tmp_path / "repo"
    repo.mkdir()
    for tid in ("task-1", "middle", "successor", "newer", "stranger"):
        write_task_result(tmp_path, tid, "completed", chat_id=1)
    for pred, succ in (("task-1", "middle"), ("middle", "successor"), ("successor", "newer")):
        binding = {"successor_task_id": succ}
        patch_document(tmp_path, pred, lambda row, b=binding, s=succ: {**row, "continued_by": {
            "successor_task_id": s, "binding": b, "binding_sha256": binding_sha(b), "state": "admitted"}})
    record = {"task_id": "task-1", "surface": "commit_gate", "subject": {"root": str(repo)}}
    ctx = SimpleNamespace(task_id="successor", drive_root=tmp_path, budget_drive_root=None, repo_dir=repo)
    return repo, record, ctx


def _chain_reads(monkeypatch):
    import ouroboros.owner_continue as owner_continue

    reads, real = [], owner_continue.continuation_chain
    monkeypatch.setattr(owner_continue, "continuation_chain", lambda *args: reads.append(args[1:]) or real(*args))
    return reads


@pytest.mark.parametrize(("holder", "listens"), [("", True), ("task-1", True), ("middle", True), ("successor", True),
                                                 ("newer", False), ("stranger", False)])
def test_the_verified_listener_reads_the_recorded_chain_once(tmp_path, monkeypatch, holder, listens):
    """#2901: the record's task, the wait's holder and this task lie on one recorded chain, read once and fresh. A
    holder on it is behind this task or this task itself; a newer listener or a stranger keeps the wait."""
    _repo, record, ctx = _listener_chain(tmp_path)
    reads = _chain_reads(monkeypatch)
    waited = {"task_id": holder, "attempt_task_id": "task-1"} if holder else {}
    assert waiting._current_listener(ctx, {**record, "waiting": waited}) is listens
    assert reads == [("task-1", "successor")]


@pytest.mark.parametrize("failing", ["surface", "attempt_owner", "checkout", "unrecorded_chain"])
def test_the_verified_listener_keeps_every_other_check(tmp_path, monkeypatch, failing):
    """The surface, the attempt's owner and the checkout still refuse before any chain is read; a chain nobody recorded
    refuses after the one read."""
    repo, record, ctx = _listener_chain(tmp_path)
    reads = _chain_reads(monkeypatch)
    waited = {"task_id": "middle", "attempt_task_id": "stranger" if failing == "attempt_owner" else "task-1"}
    record = {**record, "waiting": waited, **({"surface": "result"} if failing == "surface" else {})}
    if failing == "checkout":
        ctx.repo_dir = tmp_path / "other-repo"
        ctx.repo_dir.mkdir()
    if failing == "unrecorded_chain":
        ctx.task_id = "stranger"
    assert waiting._current_listener(ctx, record) is False
    assert reads == ([("task-1", "stranger")] if failing == "unrecorded_chain" else []), "the cheap checks refuse first"
    ctx.task_id, ctx.repo_dir = "successor", repo
    assert waiting._current_listener(ctx, {**record, "surface": "commit_gate", "waiting": {
        "task_id": "middle", "attempt_task_id": "task-1"}}) is True, "the same wait is this task's once nothing fails"


def test_the_verified_listener_fails_closed_when_the_chain_cannot_be_read(tmp_path, monkeypatch):
    import ouroboros.task_results as task_results
    from ouroboros.store import StoreBusy

    _repo, record, ctx = _listener_chain(tmp_path)
    reads = _chain_reads(monkeypatch)

    def busy(*args, **kwargs):
        raise StoreBusy("current store busy", reason="contention")

    monkeypatch.setattr(task_results, "load_task_result", busy)
    for waited in ({}, {"task_id": "middle", "attempt_task_id": "task-1"}):
        assert waiting._current_listener(ctx, {**record, "waiting": waited}) is False, "an unreadable result binds nothing"
    assert reads == [("task-1", "successor")] * 2


def test_open_additional_seat_keeps_counted_settled_wave_short_of_whole_until_adviser_closes(tmp_path):
    from ouroboros.working_checkpoint import flush_content_acks
    from tests.test_review_ledger import CRITICAL, _answer

    ctx, base = _author(tmp_path), _result_record(tmp_path)
    adviser = {**base["rows"][0], "seat_id": "c", "operation_id": "op-c", "status": "pending", "operation_state": "in_flight", "additional": True, "answers": {}}
    record = ledger.note_author_decision(tmp_path, ledger.revise_record(
        tmp_path, base["record_id"], lambda cur: {**cur, "review_wave_id": "w1", "rows": [*cur["rows"], adviser]})["record_id"],
        {"disposition": "accepted", "rationale": "counted pass"})
    _owes(ctx, record)
    waiting.acknowledge_delivery(ctx, _receipt(ctx, record, record["rows"][:2]))
    flush_content_acks(ctx)
    early = ledger.load_record(tmp_path, record["record_id"])
    assert ledger.open_additional_seats(early) and not waiting.received_whole(early, early, task_id="task-1")
    assert early["verdict"]["aggregate"] == "PASS" and _duties(tmp_path) == [(record["record_id"], "unread")]
    closed = ledger.settle_from_collected(tmp_path, record["record_id"], [_raw(
        "c", "c", "responded", parts=BOTH, operation_id="op-c", operation_state="late_settled", review_wave_id="w1",
        answers={"change": _answer("change", "FAIL", [CRITICAL]), "coupling": _answer("coupling", "PASS", [])})])
    assert closed["author_decision"] == early["author_decision"] and not waiting.received_whole(closed, closed, task_id="task-1")
    for _ in range(2):
        waiting.acknowledge_delivery(ctx, _receipt(ctx, closed, closed["rows"]))
        flush_content_acks(ctx)
    assert waiting.received_whole(ledger.load_record(tmp_path, record["record_id"]), task_id="task-1") and _duties(tmp_path) == []

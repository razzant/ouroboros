"""A host-minted direct turn starts with the proven-empty fan-out base, so its finalization reads no
unrelated history; rows without that proof keep their journal repair (#1528).

Everything runs on the real lane of ``tests/test_prestart_storage_wait.py`` (temporary data root, real
store, fake clock): the public door mints and admits the turn to the conversation lane (its RUNNING row,
the ready spare's queue), the carrier (``lane.carry``, the real worker body in this thread) runs the real
agent's preparation (budget gate, room read, the base, the billing pin) and, in the lifecycle cases, the
real loop and pipeline finalize it. Only the provider, context builder, post-task reflection and the
daemon are stubbed.
"""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import store, swarm_rollup, usage_store
from ouroboros.task_results import load_task_result
from tests.test_prestart_storage_wait import _carrier_agent, _host
from tests.test_prestart_storage_wait import lane as lane  # pytest fixture; the alias re-exports it (no F811)
from tests.test_prestart_storage_wait import short_waits as short_waits  # autouse pytest fixture

INITIAL = swarm_rollup.initial_swarm_rollup(None)


def _history(root, n):
    """N unrelated journal rows, half in a rotated archive segment, other roots' fan-out among them."""
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "archive").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        tid = f"o{i:07d}"
        rows.append({"type": "swarm_fanout", "event_id": f"swarm:old{i}", "task_id": tid, "parent_task_id": tid,
                     "root_task_id": tid, "requested_count": 2, "task_ids": [tid + "a", tid + "b"]}
                    if i % 2 == 0 else {"type": "task_received", "task_id": tid, "text": "x" * 120})
    encoded = [json.dumps(row).encode() + b"\n" for row in rows]
    (root / "archive" / "events_20260101T000000.jsonl").write_bytes(b"".join(encoded[: n // 2]))
    with (root / "logs" / "events.jsonl").open("ab") as handle:
        handle.write(b"".join(encoded[n // 2:]))
    return sum(map(len, encoded))


def _journal_reads(monkeypatch):
    reads = {"repairs": 0, "bytes": 0}
    repair, fold = swarm_rollup.repair_swarm_rollup, swarm_rollup.fold_event_suffix

    def counted_repair(*args, **kwargs):
        reads["repairs"] += 1
        return repair(*args, **kwargs)

    def counted_fold(root, through, consume, **kwargs):
        end = fold(root, through, consume, **kwargs)
        reads["bytes"] += end - through
        return end

    monkeypatch.setattr(swarm_rollup, "repair_swarm_rollup", counted_repair)
    monkeypatch.setattr(swarm_rollup, "fold_event_suffix", counted_fold)
    return reads


def _public_entry(lane, monkeypatch, actor, *, minted=True):
    """``handle_chat_direct`` as the outer door drives it; the minted id is ``S``. Returns ``enter(text)``:
    the server admits the turn (nothing of its body runs there), then the carrier — this thread — runs
    the dispatched task with ``actor``; ``None`` when the door admitted nothing to a carrier."""
    from supervisor import worker_chat_lane

    monkeypatch.setattr(worker_chat_lane, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="S")))
    monkeypatch.setattr(worker_chat_lane, "owner_conversation_admitted", lambda *_a: True)

    def enter(text="keep this input", *, between=None):
        worker_chat_lane.handle_chat_direct(7, text, task_metadata={"client_message_id": "m-1"})
        assert lane.body == [] and lane.sleeps == [], "the server only admits"
        if not lane.spare.in_q:
            return None
        if not minted:  # the same admitted turn without the minting fact: what an unproven row looks like
            lane.spare.in_q[0].pop("_fresh_identity", None)
        if between is not None:
            between()
        return lane.carry(actor)

    return enter


def _real_actor(lane, monkeypatch, fanout):
    """The carrier's real actor; the provider answers once and the context builder may emit."""
    from ouroboros import agent as agent_module, agent_task_pipeline
    from ouroboros.agent import OuroborosAgent
    from tests.test_context_fit_v664 import _plan

    repo = lane.root.parent / "repo"
    repo.mkdir(exist_ok=True)
    monkeypatch.setattr(lane.workers, "REPO_DIR", repo)

    class Provider:
        def default_model(self):
            return "openai-compatible::test"

        def chat(self, **kwargs):
            return {"content": "Answer.", "tool_calls": []}, {"cost": 0.0}

    def messages(*, task, ctx, **_kwargs):
        if fanout and not getattr(ctx, "_emitted", False):
            from ouroboros.tools.control_scheduling import _emit_swarm_fanout

            _emit_swarm_fanout(ctx, parent_task_id=str(task["id"]), root_task_id=str(task["id"]), depth=1,
                               task_group_id="grp", task_ids=["child-1"], role="subagent",
                               requested_model_lane="lane-1", objective="one emit", emitted_live=True)
            ctx._emitted = True
        ctx.context_fit_plan = replace(_plan(), model=Provider().default_model(), provider="openai-compatible",
                                       user_content_json=json.dumps(task["text"]))
        return ctx.context_fit_plan.messages_for("max"), {"budget_remaining": 5.0}

    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", messages)
    monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway",
                        lambda *_a, **_k: pytest.fail("no daemon is reached"))
    actor = _carrier_agent(lane, repo)
    actor.llm = Provider()
    return actor


@pytest.mark.parametrize(("minted", "fanout", "history"), [
    pytest.param(True, True, 2000, id='minted-one_emit-2000'),
    pytest.param(False, False, 20, id='unproven-no_fanout-20'),
])
def test_finalization_reads_unrelated_history_only_without_the_minting_proof(
        lane, monkeypatch, provider_catalog_offline, history, fanout, minted):
    history_bytes = _history(lane.root, history)
    actor = _real_actor(lane, monkeypatch, fanout)
    enter = _public_entry(lane, monkeypatch, actor, minted=minted)
    reads = _journal_reads(monkeypatch)
    assert enter() is True
    row = load_task_result(lane.root, "S")
    assert row["status"] == "completed" and not lane.errors
    if minted:
        assert reads == {"repairs": 0, "bytes": 0}
        assert "replaces" not in (row.get("swarm_fanout_basis") or {})
    else:  # the unchanged legacy path: one repair over the whole retained journal
        assert reads["repairs"] == 1 and reads["bytes"] > history_bytes
    truth = swarm_rollup.rollup_from_document(row)
    assert truth["child_ids"] == (["child-1"] if fanout else [])
    assert truth["lanes_requested"] == (["lane-1"] if fanout else [])
    efficiency = row.get("swarm_efficiency")
    assert (efficiency or {}).get("fanout_count") == (1 if fanout else None)


def test_a_minted_turn_does_not_adopt_the_journal_fanout_of_an_older_task_with_its_id(
        lane, monkeypatch, provider_catalog_offline):
    _history(lane.root, 20)
    with (lane.root / "logs" / "events.jsonl").open("a") as handle:  # a deleted task that owned the same 8-hex id
        handle.write(json.dumps({"type": "swarm_fanout", "event_id": "swarm:reused-id", "task_id": "S",
                                 "parent_task_id": "S", "root_task_id": "S", "requested_count": 1,
                                 "task_ids": ["foreign-child"]}) + "\n")
    enter = _public_entry(lane, monkeypatch, _real_actor(lane, monkeypatch, False))
    reads = _journal_reads(monkeypatch)
    assert enter() is True
    row = load_task_result(lane.root, "S")
    assert row["status"] == "completed" and reads == {"repairs": 0, "bytes": 0}
    assert swarm_rollup.rollup_from_document(row)["child_ids"] == [] and row.get("swarm_efficiency") is None


def test_a_busy_store_at_the_first_create_waits_and_writes_the_base_once(lane, monkeypatch):
    """Real contention exactly at the base's create-only write: the preparation's own wait owner sleeps,
    the store frees, the row is created once and the body runs once."""
    from ouroboros import task_results
    from tests.test_prestart_storage_wait import _hold, _release

    task_results.write_task_result(lane.root, "u0000001", "completed", result="unrelated")  # the store exists
    store.close_connections_under(lane.root)
    creates, write = [], task_results.write_task_result

    def busy_at_first_create(root, task_id, status, **kwargs):
        if task_id == "S" and kwargs.get("create_only"):
            creates.append(status)
            if len(creates) == 1:
                _hold(lane, store.store_path(lane.root))
        return write(root, task_id, status, **kwargs)

    monkeypatch.setattr(task_results, "write_task_result", busy_at_first_create)
    lane.on_sleep = lambda _count: _release(lane)
    rows = []
    enter = _public_entry(lane, monkeypatch,
                          _host(lane.root, lambda task: rows.append(load_task_result(lane.root, "S")) or []))
    assert enter() is True
    assert creates == ["requested", "requested"] and len(lane.sleeps) == 1
    assert not lane.errors and not lane.replies
    (row,) = rows
    assert row["swarm_fanout_rollup"] == INITIAL and row["billing_group"]["billing_group_id"] == "S"


def test_a_minted_turn_never_rewrites_a_row_that_already_exists(lane):
    from ouroboros.task_results import stamp_task_result_schema
    from ouroboros.task_store import patch_document

    held = {"version": 2, "fanout_count": 1, "requested_count": 1, "fanout_interval_sec_total": 0,
            "child_ids": ["older-child"], "lanes_requested": [], "source_root": "/elsewhere", "through": 7}
    patch_document(lane.root, "S", lambda _cur: stamp_task_result_schema(
        {"task_id": "S", "status": "requested", "swarm_fanout_rollup": held}))
    lane.task["_fresh_identity"] = True
    rows = []
    assert lane.run(_host(lane.root, lambda task: rows.append(load_task_result(lane.root, "S")) or [])) is True
    (row,) = rows
    assert row["swarm_fanout_rollup"] == held and "swarm_fanout_basis" not in row


def test_the_fresh_base_composes_with_copyback_and_counts_one_emission_once(tmp_path, monkeypatch):
    """Store level, since a direct turn is never split: the preparation's exact create-only write on both
    drives, a replica that finishes the work, and the real copy-back into the canonical store."""
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_finalization import build_swarm_efficiency
    from ouroboros.task_results import write_task_result
    from tests.test_swarm_legacy_transitions import emit

    canonical, replica = tmp_path / "canonical", tmp_path / "replica"
    for root in (canonical, replica):
        write_task_result(root, "parent", "requested", create_only=True)
    emit(replica, "child-1", monkeypatch=monkeypatch)
    reads = _journal_reads(monkeypatch)
    task = {"id": "parent", "root_task_id": "parent", "child_drive_root": str(replica),
            "budget_drive_root": str(canonical)}
    summary = build_swarm_efficiency(SimpleNamespace(drive_root=replica), task)
    write_task_result(replica, "parent", "completed", result="answer", swarm_efficiency=summary)
    for _ in range(2):
        assert copy_child_task_result(canonical, task)["result"] == "answer"
        after = build_swarm_efficiency(SimpleNamespace(drive_root=canonical), task)
        assert after["fanout_count"] == after["subagent_count"] == 1 and after["lanes_requested"] == ["child-1"]
        assert swarm_rollup.load_swarm_rollup(canonical, "parent")["child_ids"] == ["child-1"]
    assert reads == {"repairs": 0, "bytes": 0}


def test_a_legacy_row_without_a_base_still_recovers_its_journal_fanout(lane, monkeypatch):
    from ouroboros.task_results import stamp_task_result_schema
    from ouroboros.task_store import patch_document
    _history(lane.root, 20)
    with (lane.root / "logs" / "events.jsonl").open("a") as handle:  # an old producer: journal only
        handle.write(json.dumps({"type": "swarm_fanout", "event_id": "swarm:legacy-L", "task_id": "L",
                                 "parent_task_id": "L", "root_task_id": "L", "requested_count": 1,
                                 "task_ids": ["L-child"]}) + "\n")
    patch_document(lane.root, "L", lambda _cur: stamp_task_result_schema(
        {"task_id": "L", "status": "completed", "result": "old"}))
    enter = _public_entry(lane, monkeypatch, _host(lane.root, lambda task: lane.body.append(task) or []))
    assert enter("fresh turn") is True
    assert load_task_result(lane.root, "S")["swarm_fanout_rollup"] == INITIAL
    assert "swarm_fanout_rollup" not in load_task_result(lane.root, "L")  # no zero is invented for it
    reads = _journal_reads(monkeypatch)
    assert swarm_rollup.load_swarm_rollup(lane.root, "L")["child_ids"] == ["L-child"]
    assert reads["repairs"] == 1


@pytest.mark.parametrize("extra,seeded", [
    ({"_fresh_identity": True}, True),
    ({}, False),
    ({"_fresh_identity": True, "_attempt": 2}, False),
    ({"_fresh_identity": True, "timeout_retry_from": "R"}, False),
], ids=["minted", "unmarked", "second_attempt", "reissue"])
def test_only_a_minted_first_attempt_gets_the_base_and_the_money_binding_is_unchanged(lane, extra, seeded):
    rows = []
    lane.task.update(extra)
    host = _host(lane.root, lambda task: rows.append(load_task_result(lane.root, "S")) or [])
    assert lane.run(host) is True
    (row,) = rows  # read inside the body: the preparation has finished, the body has not written yet
    assert row["status"] == "requested"
    assert row.get("swarm_fanout_rollup") == (INITIAL if seeded else None)
    assert row["billing_group"]["billing_group_id"] == "S"
    assert row["billing_group"]["billing_group_limit_source"] == "initial_task_admission"


def test_a_budget_refusal_of_a_minted_turn_writes_no_base(lane, monkeypatch):
    """The carrier's strict money gate refuses before the base's create-only write and the money pin: the
    only row the turn leaves is the carrier's exception terminal (the schema's own default rollup, never
    the preparation's base or binding)."""
    from ouroboros import usage_accounting as ua

    usage_store.migrate_from_journal(lane.root)
    reservation = ua.reserve_attempt(ua.AttemptRequest(
        model="m", provider="openai", reservation_usd=10.0, global_limit_usd=10.0,
        drive_root=lane.root, task_id="other", root_task_id="other", source="test"))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, cost_usd=10.0, cost_final=True)
    enter = _public_entry(lane, monkeypatch, _host(lane.root, lambda task: lane.body.append(task) or []))
    assert enter() is False
    assert lane.body == [] and lane.replies == []
    [error] = lane.errors
    assert error.before_task_start is True and error.control_reason == "budget_exhausted"
    stored = load_task_result(lane.root, "S")
    assert stored["status"] == "failed" and stored["reason_code"] == "budget_exhausted"
    assert stored["result"].startswith("🚫 Budget exhausted."), stored["result"]
    assert "billing_group" not in stored and "host_admission" not in stored, "the preparation never pinned or admitted it"


@pytest.mark.parametrize("when", ["before_admission", "after_admission"])
def test_restart_deferral_keeps_its_create_only_receipt_and_the_later_start_reuses_it(lane, monkeypatch, when):
    """A Restart draining at the door (``ready``: its requester finished; a merely requested restart lets the
    owner's turn start) saves the turn on the durable queue with one create-only receipt the later start
    reuses. A drain that opens AFTER the admission finds started work: the admitted turn is not deferred,
    its carrier runs it, and no deferral receipt is written."""
    from ouroboros import server_restart
    from supervisor import queue, task_admission

    receipts = []
    write = task_admission.write_task_result

    def recorded(root, task_id, status, **kwargs):
        stored = write(root, task_id, status, **kwargs)
        receipts.append((kwargs.get("create_only"), (kwargs.get("host_admission") or {}).get("token"), stored))
        return stored

    monkeypatch.setattr(task_admission, "write_task_result", recorded)
    monkeypatch.setattr(queue, "RETURNS_NAMED", {})
    monkeypatch.setitem(server_restart._AGENT_RESTART, "status", "released")
    enter = _public_entry(lane, monkeypatch, _host(lane.root, lambda task: lane.body.append(task) or []))

    def drain_opens():
        server_restart._AGENT_RESTART["status"] = "ready"

    if when == "after_admission":
        assert enter(between=drain_opens) is True  # opened between the admission and the carrier's first step
        server_restart._AGENT_RESTART["status"] = "released"
        assert receipts == [] and lane.workers.PENDING == [], "started work is not deferred"
        assert [row["id"] for row in lane.body] == ["S"] and lane.errors == []
        started = load_task_result(lane.root, "S")
        assert started["swarm_fanout_rollup"] == INITIAL and started["billing_group"]["billing_group_id"] == "S"
        return
    drain_opens()
    assert enter() is None
    server_restart._AGENT_RESTART["status"] = "released"
    (create_only, token, stored), = receipts
    assert create_only is True and stored["host_admission"]["token"] == token and lane.body == []
    saved = load_task_result(lane.root, "S")
    assert saved["status"] == "scheduled" and saved["swarm_fanout_rollup"] == INITIAL
    (pending,) = [task for task in lane.workers.PENDING if task.get("id") == "S"]
    store.close_connections_under(lane.root)
    ran = []
    _host(lane.root, lambda task: ran.append(load_task_result(lane.root, "S")) or []).handle_task(pending)
    (row,) = ran
    assert row["host_admission"]["token"] == token and row["swarm_fanout_rollup"] == INITIAL
    assert row.get("billing_group") == saved.get("billing_group")  # the start's own money path, untouched

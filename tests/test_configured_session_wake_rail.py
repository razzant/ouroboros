"""Focused contract tests for the sleeping configured-session wake rail: what one published wake carries.

``wake_now`` publishes the wake a ready ``delegate_wait`` publishes (its replay first), with only the one addressed
read of the run scripted; the wait's own readiness is ``test_delegate_run_wait``'s."""

from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros import task_tree_ledger
from ouroboros.delegate_supervision import (
    acknowledge_pending_wake,
    delegate_wait_entry,
)
from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
from tests._run_wait_shared import held_run, scripted_reads, wake_now
from tests._shared import write_queue_document
from tests._store_shared import unreadable_result


def _ctx(tmp_path):
    return SimpleNamespace(
        task_id="parent",
        task_attempt=1,
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        task_metadata={"root_task_id": "root", "delegation_role": "subagent"},
    )


def _child(tmp_path, task_id="child", parent_task_id="parent", status=STATUS_RUNNING):
    return write_task_result(
        tmp_path,
        task_id,
        status,
        parent_task_id=parent_task_id,
        root_task_id="root",
        delegation_role="subagent",
    )


def test_sleeping_nanny_wakes_for_only_a_direct_child_beacon(tmp_path):
    _child(tmp_path)
    _child(tmp_path, "sibling", parent_task_id="other")
    ctx = _ctx(tmp_path)

    def wait_once(_ctx, _run_id, _timeout, _cursor):
        assert task_tree_ledger.tree_ledger_append(
            "root", "blocker", "child needs parent input", task_id="child",
            data_root=tmp_path,
        ).startswith("OK:")
        assert task_tree_ledger.tree_ledger_append(
            "root", "question", "sibling asks unrelated question", task_id="sibling",
            data_root=tmp_path,
        ).startswith("OK:")
        return json.dumps({"status": "no_progress", "run_id": "run-1", "last_seq": 1})

    wake = json.loads(wake_now(ctx, "run-1", wait_once=wait_once).text)
    assert wake["status"] == "no_progress"
    assert [event["type"] for event in wake["wake_events"]] == ["child_attention_beacon"]
    assert wake["wake_events"][0]["beacon"]["task_id"] == "child"


def test_child_terminal_transition_coalesces_with_leaf_wake_and_replays_until_ack(tmp_path):
    from ouroboros import delegate_custody as custody

    _child(tmp_path)
    ctx = _ctx(tmp_path)
    calls = []

    def wait_once(_ctx, _run_id, _timeout, _cursor):
        calls.append(1)
        write_task_result(tmp_path, "child", STATUS_COMPLETED, result="verified child artifact")
        return json.dumps({"status": "no_progress", "run_id": "run-1", "last_seq": 2})

    first = json.loads(wake_now(ctx, "run-1", wait_once=wait_once).text)
    terminal = next(event for event in first["wake_events"] if event["type"] == "child_terminal")
    assert terminal["child_task_id"] == "child"
    assert terminal["status"] == STATUS_COMPLETED
    replay = json.loads(wake_now(ctx, "run-1").text)  # replayed before the leaf is read (no read scripted)
    # #2992: a replay says it is one (no fresh observation); its wake facts are the published ones.
    assert replay.pop("replay_of")["fresh_observation"] is False and replay == first
    assert acknowledge_pending_wake(ctx, replay)
    assert calls == [1]
    events = [
        json.loads(line)
        for line in custody.event_log_path(tmp_path).read_text().splitlines()
    ]
    acknowledged = [
        event for event in events
        if event.get("type") == "delegate_supervision_wake_acknowledged"
    ]
    assert acknowledged[-1]["coordination"] is True


def test_physical_terminal_wake_is_not_mislabeled_as_coordination(tmp_path):
    from ouroboros import delegate_custody as custody

    ctx = _ctx(tmp_path)
    wake = wake_now(
        ctx,
        "run-physical",
        wait_once=lambda *_args: json.dumps({
            "status": "completed", "run_id": "run-physical", "last_seq": 1,
        }),
    )

    # The ACK input stays the DELIVERED transcript text, never the result object.
    assert acknowledge_pending_wake(ctx, wake.text)
    events = [
        json.loads(line)
        for line in custody.event_log_path(tmp_path).read_text().splitlines()
    ]
    acknowledged = [
        event for event in events
        if event.get("type") == "delegate_supervision_wake_acknowledged"
    ]
    assert acknowledged[-1]["coordination"] is False


def test_meaningful_wake_carries_live_tree_planning_facts_and_replays_exactly(
    tmp_path, monkeypatch,
):
    from datetime import timedelta

    from ouroboros import usage_accounting
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.deadline_utils import utc_now
    from ouroboros.utils import utc_now_iso

    contract = build_task_contract({
        "delegation_budget": {
            "intent_note": "Prefer one strong critic and preserve time to synthesize.",
        },
    })
    write_task_result(
        tmp_path,
        "root",
        STATUS_RUNNING,
        root_task_id="root",
        delegation_role="root",
        task_contract=contract,
    )
    _child(tmp_path)
    _child(tmp_path, "grandchild", parent_task_id="child")
    now = utc_now()
    (tmp_path / "state").mkdir(exist_ok=True)
    write_queue_document(tmp_path / "state" / "queue_snapshot.json", {
        "ts": utc_now_iso(),
        "pending": [],
        "running": [
            {"id": "child", "task": {
                "id": "child", "parent_task_id": "parent",
                "root_task_id": "root", "delegation_role": "subagent",
            }},
            {"id": "grandchild", "task": {
                "id": "grandchild", "parent_task_id": "child",
                "root_task_id": "root", "delegation_role": "subagent",
            }},
        ],
    })
    monkeypatch.setattr(usage_accounting, "usage_breakdown", lambda *_a, **_k: {
        "settled_usd": 1.25,
        "accounted_usd": 1.75,
        "cost_final": False,
        "unknown_unmetered": 1,
        "integrity_degraded": False,
    })
    ctx = _ctx(tmp_path)
    ctx.task_contract = contract
    ctx.task_metadata.update({
        "task_contract": contract,
        "created_at": (now - timedelta(seconds=30)).isoformat(),
        "deadline_at": (now + timedelta(minutes=10)).isoformat(),
    })

    raw = wake_now(
        ctx,
        "run-live",
        wait_once=lambda *_args: json.dumps({
            "status": "completed", "run_id": "run-live", "last_seq": 1,
        }),
    )
    wake = json.loads(raw.text)
    facts = wake["coordination_context"]
    assert facts["parent_intent"] == {
        "state": "present",
        "authority": "parent_authored_advisory",
        "text": "Prefer one strong critic and preserve time to synthesize.",
    }
    assert facts["time"]["state"] == "known"
    assert 0 < facts["time"]["remaining_sec"] <= 600
    assert facts["settled_spend"]["state"] == "partial"
    assert facts["settled_spend"]["settled_usd"] == 1.25
    assert facts["active_descendants"]["count"] == 2
    assert facts["active_descendants"]["vendor_internal"] == "opaque_not_counted"
    # The acceptance-cycle capacity left with the task-acceptance machine.
    assert "review_capacity" not in facts

    replay = wake_now(ctx, "run-live")  # pending live facts replay without recomputation (no read scripted)
    shown = json.loads(replay.text)  # #2992: the recorded facts exactly, labelled as a replay
    assert shown.pop("replay_of")["fresh_observation"] is False
    assert (replay.status, replay.code, replay.meta, shown) == (raw.status, raw.code, raw.meta, json.loads(raw.text))


def test_child_terminal_before_first_sleep_is_not_lost_as_cursor_baseline(tmp_path):
    from ouroboros.artifacts import copy_file_to_task_artifacts
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    local = tmp_path / "child-drive"
    canonical = tmp_path / "canonical"
    local.mkdir()
    canonical.mkdir()
    source = tmp_path / "child-report.txt"
    source.write_text("exact child artifact", encoding="utf-8")
    copy_file_to_task_artifacts(
        SimpleNamespace(drive_root=canonical, task_id="child"),
        source,
        kind="user_file",
    )
    _child(canonical, status=STATUS_COMPLETED)
    ctx = _ctx(local)
    ctx.budget_drive_root = str(canonical)
    assert task_tree_ledger.tree_ledger_append(
        "root",
        "review_requested",
        "Please independently challenge this evidence before integration.",
        task_id="child",
        payload={"evidence_ref": "artifact:claim-set", "evidence_sha256": "a" * 64},
        data_root=canonical,
    ).startswith("OK:")

    wake = json.loads(wake_now(
        ctx,
        "run-1",
        wait_once=lambda *_args: json.dumps({
            "status": "no_progress", "run_id": "run-1", "last_seq": 1,
        }),
    ).text)

    terminal = next(event for event in wake["wake_events"] if event["type"] == "child_terminal")
    assert terminal["child_task_id"] == "child"
    assert terminal["status"] == STATUS_COMPLETED
    assert terminal["result_sha256"] == _child_result_sha256(
        load_effective_task_result(canonical, "child")
    )
    event = next(item for item in wake["wake_events"] if item["type"] == "child_attention_beacon")
    assert event["beacon"]["kind"] == "review_requested"
    assert event["beacon"]["payload"]["evidence_sha256"] == "a" * 64
    assert not any(path.name.startswith("task_acceptance") for path in canonical.rglob("*"))


def test_oversized_coordination_wake_is_valid_bounded_json_with_exact_source(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit

    ctx = _ctx(tmp_path)
    ctx.task_contract = {
        "delegation_budget": {"intent_note": "preserve-complete-intent:" + "y" * 30_000},
    }
    for index in range(5):
        child_id = f"child-{index}"
        _child(tmp_path, task_id=child_id)
        assert task_tree_ledger.tree_ledger_append(
            "root", "blocker", f"blocker-{index}:" + "x" * 3800,
            task_id=child_id, data_root=tmp_path,
        ).startswith("OK:")

    raw = wake_now(
        ctx,
        "run-large",
        wait_once=lambda *_args: json.dumps({
            "status": "no_progress", "run_id": "run-large", "last_seq": 1,
        }),
    )
    assert len(raw.text) <= tool_result_limit("delegate_wait")
    assert _truncate_tool_result(raw.text, "delegate_wait", {}) == raw.text
    delivered = json.loads(raw.text)
    assert delivered["supervision_wake_id"]
    assert delivered["wake_delivery"]["complete"] is False
    assert delivered["wake_delivery"]["wake_events_total"] == 5
    assert delivered["coordination_context"]["state"] == "available_in_full_wake_source"
    source = delivered["wake_delivery"]["source"]
    full = json.loads(read_actor_source_bytes(tmp_path, "parent", source))
    assert len(full["wake_events"]) == 5
    assert all(len(item["beacon"]["text"]) > 3800 for item in full["wake_events"])
    assert len(full["coordination_context"]["parent_intent"]["text"]) > 30_000
    assert acknowledge_pending_wake(ctx, raw.text)

    after = json.loads(wake_now(
        ctx,
        "run-large",
        wait_once=lambda *_args: json.dumps({
            "status": "completed", "run_id": "run-large", "last_seq": 2,
        }),
    ).text)
    assert not any(
        item.get("type") == "child_attention_beacon"
        for item in after.get("wake_events", [])
    )


def test_live_descendant_fact_rejects_stale_queue_and_ignores_unrelated_corruption(
    tmp_path, monkeypatch,
):
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.utils import utc_now_iso

    ctx = _ctx(tmp_path)
    _child(tmp_path)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    snapshot = {
        "ts": "2000-01-01T00:00:00Z",
        "pending": [],
        "running": [{"id": "child", "task": {
            "id": "child", "parent_task_id": "parent",
            "root_task_id": "root", "delegation_role": "subagent",
        }}],
    }
    write_queue_document(state / "queue_snapshot.json", snapshot)
    stale = coordination_live_context(ctx)["active_descendants"]
    assert stale["state"] == "unknown"
    assert stale["count"] is None

    unreadable_result(monkeypatch, "unrelated-history")
    snapshot["ts"] = utc_now_iso()
    write_queue_document(state / "queue_snapshot.json", snapshot)
    current = coordination_live_context(ctx)["active_descendants"]
    assert current == {
        "state": "known",
        "count": 1,
        "by_status": {"running": 1},
        "scope": "host_visible_descendants",
        "vendor_internal": "opaque_not_counted",
    }


def test_live_descendant_fact_terminal_result_wins_over_stale_running_row(tmp_path):
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.utils import utc_now_iso

    ctx = _ctx(tmp_path)
    _child(tmp_path, status=STATUS_COMPLETED)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    write_queue_document(state / "queue_snapshot.json", {
        "ts": utc_now_iso(),
        "pending": [],
        "running": [{"id": "child", "task": {
            "id": "child", "parent_task_id": "parent",
            "root_task_id": "root", "delegation_role": "subagent",
        }}],
    })

    assert coordination_live_context(ctx)["active_descendants"] == {
        "state": "known",
        "count": 0,
        "by_status": {},
        "scope": "host_visible_descendants",
        "vendor_internal": "opaque_not_counted",
    }


def test_live_descendant_fact_is_unknown_for_an_unreadable_exact_queued_result(tmp_path, monkeypatch):
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.utils import utc_now_iso

    ctx = _ctx(tmp_path)
    _child(tmp_path)
    unreadable_result(monkeypatch, "child")
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    write_queue_document(state / "queue_snapshot.json", {
        "ts": utc_now_iso(), "pending": [],
        "running": [{"id": "child", "task": {
            "id": "child", "parent_task_id": "parent",
            "root_task_id": "root", "delegation_role": "subagent",
        }}],
    })
    fact = coordination_live_context(ctx)["active_descendants"]
    assert fact["state"] == "unknown" and fact["count"] is None
    assert fact["reason"] == "StoreError"


def test_live_descendant_fact_ignores_an_unreadable_unrelated_active_root(tmp_path, monkeypatch):
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.utils import utc_now_iso

    ctx = _ctx(tmp_path)
    _child(tmp_path)
    unreadable_result(monkeypatch, "unrelated-active")
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    write_queue_document(state / "queue_snapshot.json", {
        "ts": utc_now_iso(), "pending": [],
        "running": [
            {"id": "child", "task": {
                "id": "child", "parent_task_id": "parent",
                "root_task_id": "root", "delegation_role": "subagent",
            }},
            {"id": "unrelated-active", "task": {
                "id": "unrelated-active", "parent_task_id": "",
                "root_task_id": "unrelated-active", "delegation_role": "root",
            }},
        ],
    })

    fact = coordination_live_context(ctx)["active_descendants"]
    assert fact["state"] == "known"
    assert fact["count"] == 1 and fact["by_status"] == {"running": 1}


def test_delegate_wait_entry_never_acks_an_undelivered_pending_wake(tmp_path, monkeypatch):
    from ouroboros.tools import delegate as delegate_module

    _child(tmp_path, status=STATUS_COMPLETED)
    ctx = _ctx(tmp_path)
    first = wake_now(
        ctx,
        "run-1",
        wait_once=lambda *_args: json.dumps({
            "status": "no_progress", "run_id": "run-1", "last_seq": 1,
        }),
    )
    assert acknowledge_pending_wake(ctx, "truncated-not-json") is False
    monkeypatch.setattr(
        delegate_module,
        "_read_run",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("pending wake must replay before another physical poll")
        ),
    )

    replay = delegate_wait_entry(ctx, "run-1")
    shown = json.loads(replay.text)  # the same pending wake, labelled as a replay (#2992)
    assert shown.pop("replay_of")["fresh_observation"] is False
    assert (replay.status, replay.code, replay.meta, shown) == (first.status, first.code, first.meta,
                                                               json.loads(first.text))


def test_a_publication_reads_through_one_handshaken_transport_and_closes_it(tmp_path, monkeypatch):
    """A wake makes ONE addressed read of its run through one transport, handshaken once by that read (its
    ``engine_version`` is the receipt) and closed after the publication; nothing keeps a loop's transport."""
    import ouroboros.delegate_custody as custody
    import ouroboros.delegate_supervision as supervision
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway_module
    from tests._delegated_transport_shared import _delegating_ctx

    ctx = _delegating_ctx(tmp_path, acting=False)
    entry = delegate._RunCustody(task_id=ctx.task_id, route_id="fixture", model="fixture",
                                project_id="fixture", project_owned=False, access="readonly")
    monkeypatch.setitem(custody._CUSTODY, "run-loop", entry)
    gateways = []

    class _Gateway:
        def __init__(self):
            self.handshakes, self.reads, self.closed = 0, 0, False
            self._engine_version = ""
            gateways.append(self)

        @property
        def engine_version(self):
            return self._engine_version

        def handshake(self, **_kw):
            self.handshakes += 1
            self._engine_version = "3.10.2"
            return {}

        def get_run(self, rid, *, timeout_sec=None):
            assert rid == "run-loop" and timeout_sec is not None
            self.reads += 1
            return {"lastSeq": 0, "summary": {"state": "running", "effectiveAccess": "readonly"}}

        def close(self):
            self.closed = True

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", _Gateway)
    wake = json.loads(supervision._publish(ctx, {"runs": ["run-loop"]}, "mail:peer").text)
    assert wake["run_id"] == "run-loop" and wake["supervision_wake_id"]
    (gateway,) = gateways
    assert (gateway.handshakes, gateway.reads, gateway.closed) == (1, 1, True)


def test_the_empty_checkpoint_pair_asks_for_no_checkpoint(tmp_path, monkeypatch):
    """Models fill both optional keys with 0 and "" (six live refusals in one day): that
    pair is the schema's empty form. A genuinely half-filled pair is still refused, and
    the refusal names the half that is missing and echoes what it received."""
    _child(tmp_path)
    ctx = _ctx(tmp_path)
    held_run(tmp_path, "run-1", "parent", ended="succeeded")

    def wait_once(_ctx, _run_id, _timeout, _cursor):
        return json.dumps({"status": "completed", "run_id": "run-1", "last_seq": 1})

    scripted_reads(monkeypatch, wait_once)
    wake = json.loads(delegate_wait_entry(ctx, "run-1", checkpoint_after_sec=0, checkpoint_reason="").text)
    assert wake["status"] == "completed" and "error" not in wake
    assert wake["ignored_arguments"] == [
        "checkpoint_after_sec=0 ignored: with an empty checkpoint_reason it asks for no checkpoint"]

    for args, missing in (({"checkpoint_after_sec": 60}, "checkpoint_reason is missing"),
                          ({"checkpoint_reason": "look again"}, "checkpoint_after_sec is missing")):
        result = delegate_wait_entry(ctx, "run-1", **args)
        assert result.status == "error" and result.code == "TOOL_ARG_ERROR"
        assert missing in result.text and "checkpoint_requires_time_and_reason" in result.text

    # 0 seconds WITH a reason genuinely asks for an immediate inspection: on a quiet run (no pending wake to
    # replay) the sleep is armed with that inspection time and its reason rides the owed wait.
    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()
    _child(fresh_root)
    fresh = _ctx(fresh_root)
    fresh.owner_wait_callback = lambda *_a, **_k: "timeout"
    held_run(fresh_root, "run-2", "parent")
    asked = delegate_wait_entry(fresh, "run-2", checkpoint_after_sec=0, checkpoint_reason="look")
    assert asked.status == "ok" and "ignored_arguments" not in json.loads(asked.text)
    assert json.loads(asked.text)["reason"] == "sleep_armed" and fresh._model_sleep["wake_at"]
    assert fresh._pending_task_wait["checkpoint"] == {"reason": "look", "after_sec": 1}

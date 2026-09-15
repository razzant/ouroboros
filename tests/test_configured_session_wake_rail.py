"""Focused contract tests for the sleeping configured-session wake rail."""

from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros import task_tree_ledger
from ouroboros.delegate_supervision import (
    acknowledge_pending_wake,
    delegate_wait_entry,
    supervised_wait,
)
from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result


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

    wake = json.loads(supervised_wait(ctx, "run-1", wait_once=wait_once).text)
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

    first = json.loads(supervised_wait(ctx, "run-1", wait_once=wait_once).text)
    terminal = next(event for event in first["wake_events"] if event["type"] == "child_terminal")
    assert terminal["child_task_id"] == "child"
    assert terminal["status"] == STATUS_COMPLETED
    replay = json.loads(supervised_wait(
        ctx,
        "run-1",
        wait_once=lambda *_args: (_ for _ in ()).throw(
            AssertionError("the unacknowledged child wake must replay before polling the leaf")
        ),
    ).text)
    assert replay == first
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
    wake = supervised_wait(
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
    from ouroboros.review_substrate import review_binding_hash
    from ouroboros.task_results import claim_task_acceptance_review_cycle
    from ouroboros.utils import atomic_write_json, utc_now_iso

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "3")
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
    atomic_write_json(tmp_path / "state" / "queue_snapshot.json", {
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
    components = {
        "candidate_hash": "2" * 64,
        "evidence_revision": "3" * 64,
        "fence_hash": "4" * 64,
    }
    binding = {**components, "binding_hash": review_binding_hash(**components)}
    assert claim_task_acceptance_review_cycle(
        tmp_path,
        "root",
        binding,
        claimed_by_task_id="root",
    )["status"] == "claimed"
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

    raw = supervised_wait(
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
    assert facts["review_capacity"]["claimed_cycles"] == 1
    assert facts["review_capacity"]["remaining_cycles"] == 2

    replay = supervised_wait(
        ctx,
        "run-live",
        wait_once=lambda *_args: (_ for _ in ()).throw(
            AssertionError("pending live facts must replay without recomputation")
        ),
    )
    assert replay == raw


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

    wake = json.loads(supervised_wait(
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

    raw = supervised_wait(
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

    after = json.loads(supervised_wait(
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
    tmp_path,
):
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.utils import atomic_write_json, utc_now_iso

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
    atomic_write_json(state / "queue_snapshot.json", snapshot)
    stale = coordination_live_context(ctx)["active_descendants"]
    assert stale["state"] == "unknown"
    assert stale["count"] is None

    unrelated = tmp_path / "task_results" / "unrelated-history.json"
    unrelated.write_text("{broken", encoding="utf-8")
    snapshot["ts"] = utc_now_iso()
    atomic_write_json(state / "queue_snapshot.json", snapshot)
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
    from ouroboros.utils import atomic_write_json, utc_now_iso

    ctx = _ctx(tmp_path)
    _child(tmp_path, status=STATUS_COMPLETED)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    atomic_write_json(state / "queue_snapshot.json", {
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


def test_live_descendant_fact_is_unknown_for_corrupt_exact_queued_result(tmp_path):
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.utils import atomic_write_json, utc_now_iso

    ctx = _ctx(tmp_path)
    results = tmp_path / "task_results"
    results.mkdir(exist_ok=True)
    (results / "child.json").write_text("{", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    atomic_write_json(state / "queue_snapshot.json", {
        "ts": utc_now_iso(), "pending": [],
        "running": [{"id": "child", "task": {
            "id": "child", "parent_task_id": "parent",
            "root_task_id": "root", "delegation_role": "subagent",
        }}],
    })
    fact = coordination_live_context(ctx)["active_descendants"]
    assert fact["state"] == "unknown" and fact["count"] is None
    assert fact["reason"] == "ValueError"


def test_live_descendant_fact_ignores_corrupt_unrelated_active_root(tmp_path):
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.utils import atomic_write_json, utc_now_iso

    ctx = _ctx(tmp_path)
    _child(tmp_path)
    results = tmp_path / "task_results"
    (results / "unrelated-active.json").write_text("{", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    atomic_write_json(state / "queue_snapshot.json", {
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
    first = supervised_wait(
        ctx,
        "run-1",
        wait_once=lambda *_args: json.dumps({
            "status": "no_progress", "run_id": "run-1", "last_seq": 1,
        }),
    )
    assert acknowledge_pending_wake(ctx, "truncated-not-json") is False
    monkeypatch.setattr(
        delegate_module,
        "_delegate_wait",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("pending wake must replay before another physical poll")
        ),
    )

    assert delegate_wait_entry(ctx, "run-1") == first


def test_supervision_loop_holds_one_handshaken_gateway_and_drops_it_on_a_failed_tick(
    tmp_path, monkeypatch,
):
    """N quiet ticks = ONE handshake: the loop owns one transport and lends it to every
    observing wait, which handshakes it once (``engine_version`` is the receipt). A tick
    that returned no observation (an unresolved read here) drops and closes it, so the
    next tick re-reads the descriptor and re-handshakes; the loop's ``finally`` closes
    whatever it still holds. Drives the REAL observing wait over a fake gateway."""
    import ouroboros.delegate_custody as custody
    import ouroboros.delegate_supervision as supervision
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway_module
    from tests._delegated_transport_shared import _delegating_ctx

    ctx = _delegating_ctx(tmp_path, acting=False)
    entry = delegate._RunCustody(task_id=ctx.task_id, route_id="fixture", model="fixture",
                                project_id="fixture", project_owned=False, access="readonly")
    monkeypatch.setitem(custody._CUSTODY, "run-loop", entry)
    monkeypatch.setattr(supervision.time, "sleep", lambda _sec: None)
    ticks, gateways = [], []

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
            ticks.append(len(gateways))
            if len(ticks) == 4:
                raise gateway_module.ClaudexorUnavailable(
                    "daemon_unreachable", "ConnectError: [Errno 61]", observation_timeout=True)
            return {"lastSeq": 0, "summary": {"state": "running", "effectiveAccess": "readonly"}}

        def close(self):
            self.closed = True

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", _Gateway)
    monkeypatch.setattr(supervision, "_control_wakes",
                        lambda _ctx: [{"type": "deadline"}] if len(ticks) >= 6 else [])
    wake = json.loads(supervision.supervised_wait(ctx, "run-loop").text)
    assert wake["wake_events"] == [{"type": "deadline"}]
    assert ticks == [1, 1, 1, 1, 2, 2], ticks
    first, second = gateways
    assert (first.handshakes, first.reads, first.closed) == (1, 4, True)
    assert (second.handshakes, second.reads, second.closed) == (1, 2, True)

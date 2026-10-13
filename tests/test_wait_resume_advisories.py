"""Resume repairs through native warm/cold and durable terminal consumers."""

import json

import pytest

from ouroboros import model_sleep
from ouroboros import task_tree_ledger as ledger
from ouroboros.owner_mailbox import acknowledged_task_message_ids, write_task_message
from scripts.measure_wait_rounds import measure
from tests.test_model_sleep import _ctx, _mail, _result


def test_childless_default_arms_without_a_timer_and_quiet_mail_never_wakes(tmp_path):
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    assert write_task_message(
        tmp_path, "quiet", "sleeper", source_task_id="peer", provenance="peer_task", msg_id="quiet"
    )
    assert json.loads(_await_messages(ctx))["reason"] == "sleep_armed"
    assert ctx._model_sleep["tasks"] == [] and ctx._model_sleep["wake_at"] == ""
    assert model_sleep.wake_reason(ctx, ctx._model_sleep) == ""
    checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "wait"}], {}, {}, 1, [], set())
    _mail(tmp_path, "sleeper", "answer", kind="owner_text")
    assert direct_owner_wait(ctx, checkpoint) == "owner_text"
    assert acknowledged_task_message_ids(tmp_path, "sleeper") == set()
    ctx.owner_wait_callback = None
    assert "no continuation owner" in _await_messages(ctx)


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("kind", ["owner_text", "quiz_answer", "owner_pause", "hurry", "finalize_now"])
def test_childless_control_and_owner_wakes_preserve_empty_selection(tmp_path, mode, kind):
    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    chosen = model_sleep.selectors(ctx)
    assert model_sleep.request_sleep(ctx, chosen, mode)["reason"] == "sleep_armed"
    assert not chosen["any_mail"] and not chosen["tasks"] and not chosen["wake_at"]
    _mail(tmp_path, "sleeper", "input", kind=kind)
    expected = "owner_text" if kind in {"owner_text", "quiz_answer"} else f"control:{kind}"
    assert model_sleep.wake_reason(ctx, chosen) == expected
    assert acknowledged_task_message_ids(tmp_path, "sleeper") == set()


def test_attention_ledger_unchanged_poll_opens_once_then_reads_an_append(tmp_path, monkeypatch):
    from pathlib import Path

    path = ledger.tree_ledger_path("parent", data_root=tmp_path)
    path.parent.mkdir(parents=True)
    row = {"kind": "question", "task_id": "child", "ts": "same", "text": "first"}
    path.write_text(json.dumps(row) + "\n")
    opened = []
    real = Path.open

    def spy(p, *a, **kw):
        if p == path:
            opened.append(1)
        return real(p, *a, **kw)

    monkeypatch.setattr(Path, "open", spy)
    assert ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path) == [row]
    assert ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path) == [row]
    assert len(opened) == 1
    with real(path, "a") as f:
        f.write(json.dumps({**row, "text": "second"}) + "\n")
    assert len(ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path)) == 2
    assert len(opened) == 2


def test_real_terminal_producer_supplies_lineage_without_transport_keys(tmp_path):
    from tests.test_task_done_durable_vs_transport import KEYS, _run
    from tests.test_wait_measurement import _tool, _usage

    _, _, rows = _run(tmp_path, "child-a", 0)
    row = rows[0]
    assert row["parent_task_id"] == "root-9"
    assert row["root_task_id"] == "root-9" and row["delegation_role"] == "subagent"
    assert all(k not in row for k in KEYS)
    report = measure([_usage(), _tool(), row])
    assert report["coverage_complete"] and report["child_terminal_events"] == 1


def test_measurement_ignores_known_roots_and_interrupted_children_not_unknown_lineage():
    from tests.test_wait_measurement import _tool, _usage

    child = {"type": "task_done", "task_id": "child", "parent_task_id": "parent", "status": "completed"}
    root = {
        "type": "task_done",
        "task_id": "parent",
        "parent_task_id": None,
        "root_task_id": "parent",
        "delegation_role": "root",
        "status": "completed",
    }
    report = measure([_usage(), _tool(), child, root, {**child, "task_id": "retry", "status": "interrupted"}])
    assert report["coverage_complete"] and report["child_terminal_events"] == 1
    unknown = measure([_usage(), _tool(), {"type": "task_done", "task_id": "unknown", "status": "completed"}])
    assert not unknown["coverage_complete"]
    assert all(v is None for v in unknown["targets"].values())


def test_attention_memo_preserves_tail_parity_rewrites_errors_and_cold_fifo(tmp_path, monkeypatch):
    from pathlib import Path

    path = ledger.tree_ledger_path("parent", data_root=tmp_path)
    path.parent.mkdir(parents=True)
    rows = [{"kind": "question", "task_id": "child", "ts": "same", "text": str(i)} for i in range(7)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    first = ledger.tree_ledger_attention_after("parent", "same", seen_ids=set(), data_root=tmp_path)[:5]
    seen = {ledger.tree_ledger_row_id(r) for r in first}
    ledger._ATTENTION_MEMOS.clear()  # cold restoration retains only delivery identities
    assert ledger.tree_ledger_attention_after("parent", "same", seen_ids=seen, data_root=tmp_path) == rows[5:]
    with path.open("ab") as f:
        f.write(b'{"kind":')
    assert ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path) == rows
    with path.open("ab") as f:
        f.write(b'"question","task_id":"child","ts":"tail"}')
    expected = ledger.tree_ledger_rows("parent", data_root=tmp_path)
    assert ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path) == expected
    with path.open("ab") as f:
        f.write(b"\n{bad}\n")
    assert ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path) == expected
    replacement = {**rows[0], "text": "x"}
    path.write_text(json.dumps(replacement) + "\n")  # observable truncation
    assert ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path) == [replacement]
    changed = {**replacement, "text": "y"}
    path.write_text(json.dumps(changed) + "\n")  # same-size changed timestamps
    assert ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path) == [changed]
    path.unlink()
    assert ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path) == []
    path.write_text(json.dumps(replacement) + "\n")  # new inode
    assert ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path) == [replacement]
    with path.open("ab") as f:
        f.write(b"\n")
    real = Path.open

    def refused(p, *a, **kw):
        if p == path:
            raise OSError("reader unavailable")
        return real(p, *a, **kw)

    monkeypatch.setattr(Path, "open", refused)
    with pytest.raises(OSError, match="reader unavailable"):
        ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path)
    assert path not in ledger._ATTENTION_MEMOS
    monkeypatch.setattr(Path, "open", real)
    assert ledger.tree_ledger_attention_after("parent", "", data_root=tmp_path) == [replacement]


def test_terminal_lineage_keeps_explicit_root_over_a_stale_child_frame():
    from supervisor.events_task_done import _terminal_lineage

    assert _terminal_lineage(
        {"parent_task_id": None, "root_task_id": "root", "delegation_role": "root"},
        {"parent_task_id": "forged", "delegation_role": "subagent"},
    ) == {"parent_task_id": None, "root_task_id": "root", "delegation_role": "root"}
    assert _terminal_lineage({}, {}) == {}


@pytest.mark.parametrize("attention", [False, True])
def test_childless_cold_checkpoint_wakes_only_on_meaningful_mail(tmp_path, monkeypatch, attention):
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests.test_model_sleep import _cold_park, _install_queue

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    row = _cold_park(tmp_path, monkeypatch, workers)
    assert row["sleep"]["tasks"] == [] and row["sleep"]["wake_at"] == ""
    assert wake_ready_sleepers(queue) == []
    assert write_task_message(
        tmp_path, "quiet", "sleeper", source_task_id="peer", provenance="peer_task", msg_id="cold-quiet"
    )
    assert wake_ready_sleepers(queue) == []
    if attention:
        assert write_task_message(
            tmp_path,
            "question",
            "sleeper",
            source_task_id="peer",
            provenance="peer_task",
            msg_id="cold-attention",
            attention_kind="question",
        )
    else:
        _mail(tmp_path, "sleeper", "owner answer", kind="owner_text")
    outcome = wake_ready_sleepers(queue)
    assert outcome[0]["ok"] is True
    assert outcome[0]["ready"]["reason"] == ("mail:peer" if attention else "owner_text")
    assert workers.PENDING[0]["_budget_pause_resume"]["selected_by"] == "sleep_wake"
    assert acknowledged_task_message_ids(tmp_path, "sleeper") == set()

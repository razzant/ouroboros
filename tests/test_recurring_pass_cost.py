"""The server's recurring passes cost the same whatever the live set holds (cost law L1, L2, L3, L14).

``tests/test_cost_contract.py`` B7 holds every duty flat against HISTORY. These hold the same passes flat
against the LIVE SET, which is what grew on 01.11.2026: a second pass over an unchanged queue reads no task
body and no usage row, writes nothing, and costs the same at 1 and at many running, pending or paused rows. A
companion beside each law shows the pass still does its work when a real input changes, so a law cannot be
satisfied by a pass that stopped looking. Counting is not timing: no clock decides anything here.
"""
from __future__ import annotations

import pathlib

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._cost_meter import CostMeter, tagged

# The work a pass may not repeat: connections, transactions, statements, decoded JSON (a decoded task
# body is one) and every write. The tag's count of each must be equal at both live-set sizes.
WORK = ("connect", "begins", "statements", "commits", "writes", "json_loads")


def work(counts: dict) -> dict:
    return {key: counts.get(key, 0) for key in WORK}


@pytest.fixture
def meter(tmp_path):
    with CostMeter(tmp_path) as installed:
        yield installed


def counted(meter: CostMeter, tag: str, call) -> dict:
    result = tagged(tag, call)
    meter.counts[tag]["_result"] = 0  # the tag exists even for a pass that did no counted work
    return {"counts": work(meter.snapshot(tag)["counts"]), "result": result,
            "statements": meter.snapshot(tag)["statements"]}


# ---- L3: the pre-assignment pass over PENDING ------------------------------------------------------

def _pending(tmp_path, monkeypatch, name: str, rows: int):
    from ouroboros.task_results import write_task_result

    root = tmp_path / name
    root.mkdir()
    queue, _state, workers = _install_queue(root, monkeypatch)
    for index in range(rows):
        task_id = f"{name}-{index}"
        write_task_result(root, task_id, "scheduled", chat_id=0, text="queued work " * 2000)
        workers.PENDING.append({"id": task_id, "type": "task", "chat_id": 0, "_attempt": 1})
    return root, queue, workers


def test_l3_the_pending_pass_reads_once_at_any_queue_size_and_never_a_body(tmp_path, monkeypatch, meter):
    from ouroboros import task_results

    def one_pass(name, rows):
        _root, queue, workers = _pending(tmp_path, monkeypatch, name, rows)
        monkeypatch.setattr(task_results, "load_task_result",
                            lambda *a, **k: pytest.fail("the pending pass read a task body"))
        with queue._queue_lock:
            outcome = counted(meter, f"L3:{name}", workers._drop_cancelled_pending)
        monkeypatch.undo()
        assert outcome["result"] is True
        return outcome

    small, large = one_pass("one", 1), one_pass("fifty", 50)
    assert small["counts"] == large["counts"], (small["counts"], large["counts"])
    assert small["counts"]["begins"] == 1 and small["counts"]["writes"] == 0, small


def test_l3_a_new_intent_still_settles_its_row_and_the_next_row_reads_the_moved_projection(
        tmp_path, monkeypatch, meter):
    """The companion: the one read per pass still acts, and the row AFTER a settlement sees what it changed."""
    from ouroboros import cancel_intents
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result
    from supervisor import workers as workers_module

    root, queue, workers = _pending(tmp_path, monkeypatch, "live", 3)
    monkeypatch.setattr(workers_module, "_emit_task_done_terminal", lambda *a, **k: True)
    cancel_intents.request_cancel(root, "live-1", reason="owner Stop")
    reads = []
    real = cancel_intents.active_intents
    monkeypatch.setattr(cancel_intents, "active_intents",
                        lambda *a, **k: reads.append(1) or real(*a, **k))
    with queue._queue_lock:
        assert workers._drop_cancelled_pending() is True
    assert [row["id"] for row in workers.PENDING] == ["live-0", "live-2"]
    assert load_task_result(root, "live-1")["status"] == STATUS_CANCELLED
    assert not real(root), "the settled intent left the projection"
    reads.clear()
    with queue._queue_lock:
        assert workers._drop_cancelled_pending() is True
    assert len(reads) == 1, "a pass over rows nothing cancels reads the projection once"


@pytest.mark.parametrize("gap", ["unknown_status", "unreadable_projection"])
def test_l3_unknown_authority_never_dispatches(tmp_path, monkeypatch, meter, gap):
    from ouroboros import cancel_intents, store_hot_facts

    _root, queue, workers = _pending(tmp_path, monkeypatch, "gap", 1)
    if gap == "unknown_status":
        boom = store_hot_facts.statuses

        def failing(root, ids):
            answers = boom(root, ids)
            return {key: RuntimeError("store cannot answer") for key in answers}

        monkeypatch.setattr(store_hot_facts, "statuses", failing)
        with queue._queue_lock:
            assert workers._drop_cancelled_pending() is True
        assert workers.PENDING[0]["_terminalization_retry"]["trigger"] == "pending_result_authority"
        return

    def corrupt(*_a, **_k):
        raise cancel_intents.CancelIntentProjectionCorrupt("unreadable")

    monkeypatch.setattr(cancel_intents, "active_intents", corrupt)
    with queue._queue_lock:
        assert workers._drop_cancelled_pending() is False
    assert [row["id"] for row in workers.PENDING] == ["gap-0"]

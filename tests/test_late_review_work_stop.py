"""A review still running after the answer keeps its work's Stop, and the Stop door takes it.

The answered root has left the queue; its review's delegated run still works. That run is an open run of the
root's tree, so the work offers Stop whether the review operation's pointer is still open or already closed,
and the cascade cancels the run behind the owner's verdict, the one that also speaks for a review panel. While
the operation is live under its controller the Stop stays pending (202): the operation reads the same Stop and
closes itself. Once it has closed, the cascade settles (200). A delegated run outside a review takes the same
door (tests/test_settled_tree_run_stop.py).
"""
import os

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.delegate_custody_reconcile import open_runs
from ouroboros.platform_layer import process_start_time
from ouroboros.review_operation import OPERATION_DISPATCHED, OPERATIONS_FIELD
from ouroboros.task_results import write_task_result
from tests._cancel_intents_shared import qenv as _qenv
from tests.test_settled_tree_run_stop import _controls, _engine, _noop_incidents, _started, _stop

# Requested by name as a test parameter: re-bound through a module attribute (F811 otherwise).
qenv = _qenv

ROOT = "answered-root"


def _answered_root_with_its_review_running(drive, pointer_state):
    write_task_result(drive, ROOT, "completed", result="the answer", root_task_id=ROOT, chat_id=42,
                      **{OPERATIONS_FIELD: {"wave-owner": {
                          "state": pointer_state, "surface": "multi_model_review", "retry_key": "wave",
                          # another live process's controller (this process's own would need a live registration)
                          "controller": {"pid": os.getppid(), "birth": process_start_time(os.getppid()),
                                         "session": "another-process"}}}})
    _started(drive, run_id="review-run", task_id=ROOT, root_task_id=ROOT, source="review_substrate",
             category="task_acceptance_review", review_slot_id="slot-1")


@pytest.mark.parametrize("pointer_state, status", [(OPERATION_DISPATCHED, 202), ("closed", 200)])
def test_a_review_after_the_answer_keeps_its_works_stop_and_the_door_takes_it(qenv, monkeypatch,
                                                                              pointer_state, status):
    _answered_root_with_its_review_running(qenv.drive, pointer_state)
    engine = _engine(monkeypatch)
    assert _controls(qenv.drive) == [[("stop", ROOT)]]
    response = _stop(qenv.drive, ROOT)
    assert response.status_code == status, response.json()
    assert engine.cancels == [("review-run", "owner_task_gone")], "the owner's verdict speaks for the panel"
    assert open_runs(qenv.drive) == []
    # A live operation keeps the Stop open until it closes, and says so; a closed one leaves nothing to wait on.
    assert (ci.active_intent(qenv.drive, ROOT) is None) is (status == 200)
    assert bool(_noop_incidents(qenv.drive, ROOT)) is (status == 202)

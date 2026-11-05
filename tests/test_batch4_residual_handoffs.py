"""Residual Pause replay, pending recovery, and atomic validation consumers."""
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest




def _pending(tmp_path):
    from ouroboros import delegate_custody as dc
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root", "running", root_task_id="root")
    assert dc.record_start_requested(tmp_path, run_id="", task_id="child", root_task_id="root",
        invocation_id="inv-exact", idempotency_key="inv-exact", max_seconds=60,
        request={"prompt": "original full request"}, project_id="", project_owned=False, route="test")
    return dc.pending_invocations(tmp_path)[0]


def test_pending_recovery_handoff_is_excluded_with_pause_and_wait_is_outside_lock(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as dc, owner_pause
    from ouroboros.delegate_custody_reconcile import _recover_pending_invocation

    record = _pending(tmp_path)
    accepted, release = threading.Event(), threading.Event()
    real_submit = ThreadPoolExecutor.submit
    submissions, observed, results, errors = [], [], [], []
    def submit(executor, function, *args, **kwargs):
        def delayed():
            assert release.wait(5)
            return function(*args, **kwargs)
        future = real_submit(executor, delayed)
        submissions.append(True)
        accepted.set()
        return future
    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    def start(body, *, idempotency_key):
        # The handed request may enter its SDK after Pause has been accepted.
        observed.append((body, idempotency_key, owner_pause.read_fence(tmp_path, "root")["state"]))
        return {}  # Unknown accepted handle stays pending, not proven no-effect.
    def recover():
        try:
            results.append(_recover_pending_invocation(tmp_path, SimpleNamespace(start_run=start), record))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=recover)
    thread.start()
    try:
        assert accepted.wait(5), "recovery must use the concrete executor handoff"
        owner_pause.install_fence(tmp_path, "root", request_id="P")
        assert thread.is_alive() and not observed
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert submissions == [True]
    assert record["request"] is None and record["request_ref"]
    assert observed == [({"prompt": "original full request"}, record["invocation_id"], "requested")]
    assert results[0]["action"] == "recovery_pending"
    assert dc.pending_invocations(tmp_path) == [record]


@pytest.mark.parametrize("failure", ["parse", "late_hunk", "missing_delete", "existing_add"])
def test_atomic_patch_validation_refusal_retires_only_its_exact_handoff(tmp_path, monkeypatch, failure):
    from tests.test_batch4_producer_custody import _registry, _consumers
    from ouroboros.task_results import load_task_result

    registry, q, workers = _registry(tmp_path, monkeypatch)
    workspace = registry._ctx.workspace_root
    (workspace / "a.txt").write_text("old\n")
    good = "*** Update File: a.txt\n@@\n-old\n+new\n"
    patches = {
        "parse": "not a patch",
        "late_hunk": good + "*** Update File: a.txt\n@@\n-absent\n+unused\n",
        "missing_delete": good + "*** Delete File: missing.txt\n",
        "existing_add": good + "*** Add File: a.txt\n+duplicate\n",
    }
    result = registry.execute_result("apply_patch", {"patch": patches[failure]})
    assert result.status != "ok", result
    assert result.meta.get("operation_outcome") == "completed_no_effect", result
    assert (workspace / "a.txt").read_text() == "old\n"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    _consumers(tmp_path, registry, q, workers, held=False)

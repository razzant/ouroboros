"""Per-run controls and independent Git captures keep their original floors."""
import json
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from tests._delegated_transport_shared import (  # noqa: F401
    _LiveRunStub, _nanny_ctx, _owned_gateway_uses_each_test_transport,
)


def test_cancel_one_preserves_others_and_root_stop_reaches_late_bound_work(tmp_path, monkeypatch):
    from ouroboros.delegate_custody_reconcile import stop_tree_runs
    from ouroboros.gateways import claudexor
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import delegate

    class Engine(_LiveRunStub):
        def __init__(self):
            self.cancelled = []

        def cancel_run(self, run_id, reason=""):
            self.cancelled.append(run_id)
            return {"accepted": True, "status": "accepted"}

        def get_run(self, run_id, **_kwargs):
            return {"lastSeq": 1, "summary": {"state": "cancelled" if run_id in self.cancelled else "running"}}

        def start_run(self, *_args, **_kwargs):
            pytest.fail("Stop or cleanup replayed a pending POST")

    engine = Engine()
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *_a, **_k: engine)
    ctx = _nanny_ctx(tmp_path)
    for name, task in (("a", "t-a"), ("b", "t-a"), ("child", "child")):
        assert custody.record_started(tmp_path, custody.RunCustody(run_id=name, task_id=task,
            root_task_id="t-a", route_id="r", model="m", access="readonly"))
    assert custody.record_start_requested(tmp_path, task_id="t-a", root_task_id="t-a",
        invocation_id="late", request={"prompt": "already submitted"})
    assert json.loads(delegate._delegate_cancel(ctx, "a").text)["status"] == "confirmed"
    assert engine.cancelled == ["a"]
    assert {r.run_id for r in custody.open_runs(tmp_path)} == {"b", "child"}
    stop_tree_runs(tmp_path, "t-a", gateway_factory=lambda: engine)
    assert set(engine.cancelled) == {"a", "b", "child"}
    assert [r["invocation_id"] for r in custody.pending_invocations(tmp_path)] == ["late"]
    write_task_result(tmp_path, "t-a", status="cancelled", result="Owner stopped the root")
    # The already-submitted reply arrives after Stop; no retry created this run.
    assert custody.record_started(tmp_path, custody.RunCustody(run_id="late-bound", task_id="t-a",
        root_task_id="t-a", invocation_id="late", route_id="r", model="m", access="readonly"))
    custody.reconcile_orphaned_runs(tmp_path, running_task_ids=set(), gateway_factory=lambda: engine)
    assert engine.cancelled[-1] == "late-bound" and not custody.open_runs(tmp_path)


@pytest.mark.parametrize("overlap", [False, True])
def test_independent_git_captures_apply_disjoint_paths_and_refuse_overlap(tmp_path, monkeypatch, overlap):
    from ouroboros.subagent_worktrees import provision_execution_snapshot
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    from tests.test_delegated_run_isolation import _seed_target, _nanny_ctx as git_context, _isolated_entry

    target = _seed_target(tmp_path)
    ctx = git_context(tmp_path, target, monkeypatch)
    custody._CUSTODY.clear()
    rows = []
    for name in ("a", "b"):
        snapshot = provision_execution_snapshot(target_root=target, task_id=ctx.task_id, snapshot_id=name)
        path = "tracked.txt" if name == "a" or overlap else "second.txt"
        (Path(snapshot.path) / path).write_text(f"{name} changed\n")
        entry = _isolated_entry(ctx, target, snapshot, run_id=name)
        capture = _capture_terminal_patch(ctx, entry)
        assert capture["status"] == "ready_with_changes"
        rows.append((entry, capture))
    assert rows[0][0].execution_root != rows[1][0].execution_root
    assert (target / "tracked.txt").read_text() == "one\ntwo\n"
    assert "✅ Integrated" in _integrate_delegated_patch(ctx, "a", "apply", "chosen")
    second = _integrate_delegated_patch(ctx, "b", "apply", "independent work")
    if overlap:
        assert "CONFLICT" in second and not rows[1][0].patch_disposed
        assert Path(rows[1][1]["patch_artifact"]).is_file()
    else:
        assert "✅ Integrated" in second and rows[1][0].patch_disposed == "applied"
        assert (target / "second.txt").read_text() == "b changed\n"
    custody._CUSTODY.clear()

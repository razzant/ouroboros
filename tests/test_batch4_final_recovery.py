"""A refused recovery attempt cannot retire an earlier unknown physical create."""
from pathlib import Path


from tests._delegated_transport_shared import _owned_gateway_uses_each_test_transport  # noqa: F401
from tests.test_delegated_full_access import full_run as full_run



def test_recovering_claim_refusal_keeps_same_pending_invocation_and_snapshot(full_run, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_shared import delegate_payload
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools import delegate

    ctx, _target, facts = full_run
    facts["lost_start"] = True
    first = delegate_payload(delegate._delegate_start(ctx, "Same pending work."))
    invocation = first["pending_invocation_id"]
    drive = custody.custody_root(ctx)
    pending = custody.pending_invocations(drive)
    assert len(pending) == 1 and pending[0]["invocation_id"] == invocation
    snapshot = find_execution_snapshot(invocation)
    assert snapshot and Path(snapshot["path"]).is_dir()
    before = (Path(snapshot["path"]) / "README.md").read_bytes()
    monkeypatch.setattr(custody, "record_start_requested", lambda *_a, **_kw: False)
    refused = delegate_payload(delegate._delegate_start(ctx, "Same pending work.", retry_of=invocation))
    assert refused["reason"] == "start_request_row_unwritable"
    assert not refused.get("definitely_unrun")
    assert refused["project_retired"] is False
    assert len(facts["requests"]) == 1  # no second physical request
    assert custody.pending_invocations(drive)[0]["invocation_id"] == invocation
    assert (Path(snapshot["path"]) / "README.md").read_bytes() == before
    assert find_execution_snapshot(invocation)

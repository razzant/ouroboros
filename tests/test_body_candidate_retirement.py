"""Candidate ownership and GC serialize before the checkout's files are removed."""
from pathlib import Path

import pytest

from ouroboros import body_candidate as body, subagent_worktrees as wt
from ouroboros.task_results import write_task_result
from tests.body_candidate_support import git, isolate, make_ctx, make_serving


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    serving, data = make_serving(tmp_path), isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    bound = body.prepare(make_ctx(serving, data, "predecessor"))
    write_task_result(data, "predecessor", "completed")
    successor = make_ctx(serving, data, "successor", task_contract={
        "predecessor_authority": {"source": {"task_id": "predecessor"}}})
    return data, bound, successor


@pytest.mark.parametrize("explicit", [False, True])
def test_prepare_during_deletion_never_binds_the_retiring_checkout(candidate, monkeypatch, explicit):
    data, bound, successor = candidate
    old_path, remove = Path(bound["path"]), wt._force_rmtree
    prepared = []

    def during_delete(path):
        assert path == old_path and path.is_dir()
        if explicit:
            with pytest.raises(body.CandidateRefused) as refused:
                body.prepare(successor, resume=bound["candidate_id"])
            assert refused.value.code == "CANDIDATE_MISSING"
            assert not body.is_bound(successor)
        prepared.append(body.prepare(successor))
        remove(path)

    monkeypatch.setattr(wt, "_force_rmtree", during_delete)
    assert wt.prune_orphans(retention_days=0) == {"removed": 1, "kept": 1}
    fresh = Path(prepared[0]["path"])
    assert fresh != old_path and fresh.is_dir() and not old_path.exists()
    assert prepared[0]["state"] == "new"
    assert (fresh / "ouroboros/mod_a.py").read_text() == "GEN = 'GEN_OLD'\n"
    assert git(fresh, "status", "--porcelain") == ""
    assert body.find("successor", data)["path"] == str(fresh)


@pytest.mark.parametrize("explicit", [False, True])
def test_transfer_before_the_retirement_decision_keeps_the_checkout(candidate, monkeypatch, explicit):
    data, bound, successor = candidate
    judge = body.retention_verdict

    def transfer_after_capture(row, **kwargs):
        verdict = judge(row, **kwargs)
        assert verdict["remove"]
        inherited = body.prepare(successor, resume=bound["candidate_id"] if explicit else "")
        assert inherited["path"] == bound["path"] and inherited["state"] == "resumed"
        return verdict

    monkeypatch.setattr(body, "retention_verdict", transfer_after_capture)
    assert wt.prune_orphans(retention_days=0) == {"removed": 0, "kept": 1}
    path = Path(bound["path"])
    assert path.is_dir() and git(path, "status", "--porcelain") == ""
    assert body.find("successor", data)["owners"] == ["predecessor", "successor"]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("survives", [False, True])
def test_retirement_wins_after_candidate_lookup(candidate, monkeypatch, explicit, survives):
    data, bound, successor = candidate
    transfer = body._transfer
    if survives:
        monkeypatch.setattr(wt, "_force_rmtree", lambda path: None)

    def retire_before_transfer(ctx, row, owner, **kwargs):
        assert row["path"] == bound["path"] and not row.get("retiring")
        result = wt.prune_orphans(retention_days=0)
        assert result == {"removed": int(not survives), "kept": int(survives)}
        return transfer(ctx, row, owner, **kwargs)

    monkeypatch.setattr(body, "_transfer", retire_before_transfer)
    if explicit:
        with pytest.raises(body.CandidateRefused) as refused:
            body.prepare(successor, resume=bound["candidate_id"])
        assert refused.value.code == "CANDIDATE_MISSING"
        assert not body.is_bound(successor)
    fresh = body.prepare(successor)
    assert fresh["path"] != bound["path"] and Path(fresh["path"]).is_dir()
    assert git(fresh["path"], "status", "--porcelain") == ""
    assert body.find("predecessor", data) is None
    facts = body.context_fact("successor", data)
    assert [fact["path"] for fact in facts] == [fresh["path"]] and facts[0]["yours"]
    if survives:
        [retiring] = [row for row in wt.list_worktrees(data) if row["path"] == bound["path"]]
        assert retiring["retiring"] and retiring["task_id"] == "predecessor"
        assert retiring["removal_failed"] and Path(bound["path"]).is_dir()

"""Hardening round (codex code-review C1-C3): bounded update git plumbing,
the single-batch staged-marker scan, and the merge-write tests-evidence
record. Split out of ``test_update_merge_assisted.py`` at the module-size
gate; the repo fixtures are shared from that suite."""

import supervisor.update_merge as update_merge
from tests.test_update_merge_assisted import (
    _authority_metadata,
    _conflict_repo,
    _materialized_conflict_tx,
    _supervisor_events,
)

# ---------------------------------------------------------------------------
# Hardening round (C1-C3): bounded git plumbing, batched marker scan, and the
# merge-write tests-evidence record.
# ---------------------------------------------------------------------------


def test_marker_check_unmerged_staged_entry_is_an_inspection_error(tmp_path, monkeypatch):
    """C2 parity pin: an UNMERGED index entry has no stage-0 blob — the batch
    resolves it to ``missing`` and the gate reports the same typed inspection
    error the per-path scan reported."""
    repo, head, plan = _conflict_repo(tmp_path, monkeypatch)
    ok_m, msg, _m0 = update_merge.materialize_assisted_merge_live(
        head, plan["local_snapshot"], plan["target_sha"], plan["base_sha"]
    )
    assert ok_m, msg
    # a.txt is unmerged and NOT git-added: no stage-0 entry to inspect.
    ok, err = update_merge.managed_assisted_marker_check()
    assert not ok and "could not inspect staged file a.txt" in err


def test_tests_evidence_on_corrupt_marker_skips_loudly_and_keeps_the_tree(tmp_path, monkeypatch):
    """C3 corrupt branch (F1 semantics): a marker corrupted between the read
    and the write is NEVER overwritten by the forensic evidence — the write is
    skipped with a loud supervisor row and the caller still gets the tree for
    the process-held proof."""
    from supervisor import update_candidate

    repo, head, plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    meta = _authority_metadata(tx)
    marker_path = update_merge._update_tx_marker_path()
    real_snapshot = update_candidate.worktree_snapshot_tree

    def _corrupting_snapshot(base_ref, **kwargs):
        marker_path.write_text("{corrupt", encoding="utf-8")
        return real_snapshot(base_ref, **kwargs)

    monkeypatch.setattr(update_candidate, "worktree_snapshot_tree", _corrupting_snapshot)
    tree = update_merge.record_managed_tests_evidence("resolver", meta)

    assert tree  # the process-held proof still gets its tree
    assert marker_path.read_text(encoding="utf-8") == "{corrupt"  # evidence preserved
    skipped = _supervisor_events(tmp_path, "managed_update_tx_phase_write_skipped_corrupt")
    assert skipped and "tests_evidence" in skipped[-1]["patch_keys"]

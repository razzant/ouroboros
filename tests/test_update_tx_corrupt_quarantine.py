"""Boot-time handling of a CORRUPT managed-update tx marker (#447 F8).

An unparseable marker used to be left in place forever, latching
``repo_writer_admission_closed`` for the whole install with no recovery path.
Boot now quarantines it aside byte-intact (evidence survives, the latch does
not) unless MERGE_HEAD shows the marker may cover a live merge."""

from supervisor import update_candidate, update_merge

from tests.test_update_merge_assisted import _init_repo, _point_at


def test_corrupt_marker_is_quarantined_on_boot_and_reopens_admission(tmp_path, monkeypatch):
    """A marker no flow can parse must not latch pooled/chat admission forever
    (#447): boot quarantines it byte-intact and the durable gate reopens."""
    import supervisor.workers as workers

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    marker = update_merge._update_tx_marker_path()
    marker.write_text("{ not json", encoding="utf-8")
    corrupt_bytes = marker.read_bytes()
    assert workers.repo_writer_admission_closed() == "managed_update_tx:corrupt"

    res = update_merge.finalize_managed_update_on_boot()

    assert res.get("finalized") is False
    assert not marker.exists()
    assert update_merge.read_update_tx_strict()[0] == "absent"
    quarantined = list((repo / ".git").glob("ouroboros-update-tx.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupt_bytes  # evidence preserved byte-intact
    assert workers.repo_writer_admission_closed() == ""


def test_corrupt_marker_over_inflight_merge_stays_fail_closed(tmp_path, monkeypatch):
    """MERGE_HEAD is the one sign a corrupt marker may still cover a genuinely
    active transaction: boot must NOT quarantine it and admission stays closed."""
    import supervisor.workers as workers

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    # The quarantine helper lives in update_candidate and calls ITS module-level
    # _merge_head_sha — patch the callee's module, not update_merge's re-export.
    monkeypatch.setattr(update_candidate, "_merge_head_sha", lambda: "deadbeef")
    marker = update_merge._update_tx_marker_path()
    marker.write_text("{ not json", encoding="utf-8")
    corrupt_bytes = marker.read_bytes()

    res = update_merge.finalize_managed_update_on_boot()

    assert res.get("finalized") is False
    assert marker.read_bytes() == corrupt_bytes
    assert workers.repo_writer_admission_closed() == "managed_update_tx:corrupt"


def test_active_valid_tx_still_closes_admission(tmp_path, monkeypatch):
    """The recoverable-corrupt change must not weaken the real gate: a VALID
    active transaction still closes the durable writer admission."""
    import supervisor.workers as workers

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    update_merge.write_update_tx({"phase": "assisted_resolution", "task_id": "x"})
    assert workers.repo_writer_admission_closed() == "managed_update_tx:assisted_resolution"


def test_unreadable_marker_is_left_in_place_fail_closed(tmp_path, monkeypatch):
    """#447 S2: "corrupt" conflates unreadable with unparseable. A marker that
    cannot be READ may still be a valid live transaction — boot must not
    quarantine it, and admission stays closed."""
    import os

    import supervisor.workers as workers

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    marker = update_merge._update_tx_marker_path()
    marker.write_text('{"phase": "assisted_resolution", "task_id": "x"}', encoding="utf-8")
    os.chmod(marker, 0)
    try:
        res = update_merge.finalize_managed_update_on_boot()
    finally:
        os.chmod(marker, 0o644)

    assert res.get("finalized") is False
    assert "fail closed" in str(res.get("reason") or ""), res
    assert marker.exists()


def test_second_quarantine_in_same_second_does_not_overwrite_first(tmp_path, monkeypatch):
    """#447 S5: POSIX rename replaces its destination and the quarantine name has
    one-second resolution — evidence must be uniquified, not clobbered."""
    import supervisor.workers as workers

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    import time as _time

    monkeypatch.setattr(_time, "time", lambda: 1_756_000_000.0)

    marker = update_merge._update_tx_marker_path()
    marker.write_text("{ first garbage", encoding="utf-8")
    update_merge.finalize_managed_update_on_boot()
    marker.write_text("{ second garbage", encoding="utf-8")
    update_merge.finalize_managed_update_on_boot()

    quarantined = sorted((repo / ".git").glob("ouroboros-update-tx.json.corrupt-*"))
    assert len(quarantined) == 2, quarantined
    contents = {q.read_bytes() for q in quarantined}
    assert contents == {b"{ first garbage", b"{ second garbage"}

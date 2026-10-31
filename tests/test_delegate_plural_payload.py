"""Independent payload captures serialize only the live target mutation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import provision_payload_snapshot
from tests.test_delegated_skill_payload import _payload_ctx, _payload_entry, _seed_skill


def captures(root, monkeypatch):
    from ouroboros.tools.delegate import _capture_terminal_patch

    ctx = _payload_ctx(root, monkeypatch)
    skill = _seed_skill(root / "data")
    rows = []
    for name in ("a", "b"):
        snapshot = provision_payload_snapshot(target_root=skill, task_id=ctx.task_id, snapshot_id=f"snap-{name}")
        (Path(snapshot.path) / "notes.txt").write_text(f"{name}\n")
        entry = _payload_entry(snapshot, skill, run_id=name)
        capture = _capture_terminal_patch(ctx, entry)
        assert capture["status"] == "ready_with_changes"
        rows.append((entry, capture))
    assert rows[0][0].execution_root != rows[1][0].execution_root
    assert (skill / "notes.txt").read_text() == "PENDING\n"
    return ctx, skill, rows


def test_first_payload_apply_wins_and_stale_second_keeps_its_evidence(tmp_path, monkeypatch):
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, rows = captures(tmp_path, monkeypatch)
    first = _integrate_delegated_patch(ctx, "a", "apply", "chosen")
    second = _integrate_delegated_patch(ctx, "b", "apply", "also inspected")
    assert "✅ Integrated" in first
    assert "CONFLICT" in second and "✅" not in second
    assert (skill / "notes.txt").read_text() == "a\n"
    assert rows[0][0].patch_disposed == "applied" and not rows[1][0].patch_disposed
    assert Path(rows[1][1]["patch_artifact"]).is_file()
    assert Path(rows[1][0].execution_root).is_dir()
    custody._CUSTODY.clear()


def test_target_lock_covers_cas_apply_and_publication_across_capture_locks(tmp_path, monkeypatch):
    from ouroboros.tools import delegate_integration as integration
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    from ouroboros import platform_layer

    ctx, skill, _rows = captures(tmp_path, monkeypatch)
    first_at_cas, second_at_lock, release = threading.Event(), threading.Event(), threading.Event()
    original_hash, acquire = integration.payload_content_hash, platform_layer.acquire_exclusive_file_lock
    observations = []

    def locked(path, **kwargs):
        if path.parent.name == "delegate_payload_mutations" and threading.current_thread().name.endswith("_1"):
            second_at_lock.set()
        return acquire(path, **kwargs)

    def content_hash(target):
        value = original_hash(target)
        observations.append((threading.current_thread().name, value))
        if len(observations) == 1:
            first_at_cas.set()
            assert release.wait(30), "test did not release the first CAS"
        return value

    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", locked)
    monkeypatch.setattr(integration, "payload_content_hash", content_hash)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="payload") as pool:
        first = pool.submit(_integrate_delegated_patch, ctx, "a", "apply", "chosen")
        try:
            assert first_at_cas.wait(30)
            second = pool.submit(_integrate_delegated_patch, ctx, "b", "apply", "also inspected")
            assert second_at_lock.wait(30), "second capture never reached the target lock"
            assert len(observations) == 1, "the second capture read CAS outside the target mutation lock"
        finally:
            release.set()
        assert "✅ Integrated" in first.result(timeout=30)
        assert "CONFLICT" in second.result(timeout=30)
    assert (skill / "notes.txt").read_text() == "a\n"
    custody._CUSTODY.clear()


def test_unavailable_target_lock_refuses_without_mutating_either_capture(tmp_path, monkeypatch):
    from ouroboros import platform_layer
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, rows = captures(tmp_path, monkeypatch)
    original = platform_layer.acquire_exclusive_file_lock
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", lambda path, **kw:
        None if path.parent.name == "delegate_payload_mutations" else original(path, **kw))
    result = _integrate_delegated_patch(ctx, "a", "apply", "chosen")
    assert "INTEGRATE_DELEGATED_TARGET_BUSY" in result
    assert (skill / "notes.txt").read_text() == "PENDING\n"
    assert all(not entry.patch_disposed and Path(capture["patch_artifact"]).exists() for entry, capture in rows)
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", original)
    assert "✅ Integrated" in _integrate_delegated_patch(ctx, "a", "apply", "chosen")
    custody._CUSTODY.clear()

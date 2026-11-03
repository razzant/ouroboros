"""A Resume retries only its fence publication under contention."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents, owner_pause
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_admission_contention import database_reader

pytestmark = pytest.mark.serial


def _contend_publication(monkeypatch, *, outside=None, acquire_wait_sec=0.0):
    actual = owner_pause.release_fence
    publications, waits = [], []

    def release(root, task_id, **kwargs):
        if not publications:
            publications.append((root, task_id))
            with database_reader(root, exclusive=True, acquire_wait_sec=acquire_wait_sec):
                return actual(root, task_id, **kwargs)
        return actual(root, task_id, **kwargs)

    def wait(_seconds):
        root, task_id = publications[0]
        with owner_pause.launch_lock(root, task_id, wait=False):
            pass  # A real second owner can acquire the gate before retry.
        waits.append(True)
        if outside is not None and len(waits) == 1:
            outside(root, task_id)

    monkeypatch.setattr(owner_pause, "release_fence", release)
    clock = owner_pause.time
    monkeypatch.setattr(owner_pause, "time", SimpleNamespace(
        monotonic=clock.monotonic, time=clock.time, sleep=wait))
    return publications, waits


@pytest.mark.parametrize("interruption", ["open", "pause", "stop", "different_grant", "read_probe"])
def test_resume_publication_rechecks_exact_current_controls(tmp_path, monkeypatch, interruption):
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    fence, _ = owner_pause.install_fence(tmp_path, "root", request_id="original-pause")
    pause = {"state": "resumed", "grant": {"grant_id": "exact-grant", "authority": "explicit_resume",
             "consumed_at": 1, "owner_pause_fence_id": fence["fence_id"]}}
    write_task_result(tmp_path, "root", "running", budget_pause=pause)

    def change(root, task_id):
        if interruption == "pause":
            owner_pause.install_fence(root, task_id, request_id="newer-pause")
        elif interruption == "stop":
            cancel_intents.request_cancel(root, task_id, source="http_single", reason="owner Stop")
        elif interruption == "different_grant":
            write_task_result(root, task_id, "running", budget_pause={**pause,
                "grant": {**pause["grant"], "grant_id": "another-grant"}})

    publications, waits = _contend_publication(monkeypatch, outside=change)
    if interruption == "read_probe":
        from ouroboros.store import StoreBusy
        original_read, failed = owner_pause.read_fence, []
        def busy_read(*args, **kwargs):
            if waits and not failed:
                failed.append(True)
                raise StoreBusy("read unavailable before mutation", reason="contention")
            return original_read(*args, **kwargs)
        monkeypatch.setattr(owner_pause, "read_fence", busy_read)
    def resume():
        owner_pause.reopen_for_resume(tmp_path, "root", "root", fence_id=fence["fence_id"], grant_id="exact-grant")
    if interruption in {"stop", "different_grant"}:
        with pytest.raises(owner_pause.OwnerPauseRefused, match=(
                "cancelled" if interruption == "stop" else "root_resume_authority_changed")):
            resume()
    else:
        resume()
    assert len(publications) == 1 and waits
    current = load_task_result(tmp_path, "root")["owner_pause"]
    assert owner_pause.fence_closed(current) is (interruption not in {"open", "read_probe"})
    assert (current["fence_id"] != fence["fence_id"]) is (interruption == "pause")


def test_unknown_resume_publication_never_retries_the_write(tmp_path, monkeypatch):
    from ouroboros.store import StoreBusy

    write_task_result(tmp_path, "root", "running", root_task_id="root")
    fence, _ = owner_pause.install_fence(tmp_path, "root", request_id="pause")
    error = StoreBusy("publication rollback is unconfirmed", reason="contention")
    error.rollback_confirmed = False
    writes = []
    def unknown(*_args, **_kwargs):
        writes.append(True)
        raise error
    monkeypatch.setattr(owner_pause, "release_fence", unknown)
    clock = owner_pause.time
    monkeypatch.setattr(owner_pause, "time", SimpleNamespace(
        monotonic=clock.monotonic, sleep=lambda _seconds: pytest.fail("unknown write was retried")))
    with pytest.raises(StoreBusy) as caught:
        owner_pause.reopen_for_resume(tmp_path, "root", "root", fence_id=fence["fence_id"], grant_id="grant")
    assert caught.value is error and writes == [True]
    assert owner_pause.read_fence(tmp_path, "root") == fence

"""Immutable promotion shares storage without sharing a mutable publication."""

import errno
import shutil
from pathlib import Path

import pytest

from ouroboros import headless, observability, source_retention


@pytest.fixture(params=[None, errno.EXDEV, errno.EPERM, errno.EACCES, errno.EOPNOTSUPP])
def link_error(request, monkeypatch):
    if request.param is not None:
        def unsupported(*_args, **_kwargs):
            raise OSError(request.param, "link unavailable in fixture")
        monkeypatch.setattr(source_retention.os, "link", unsupported)
    return request.param


def test_promoted_blob_shares_inode_or_copies_and_survives_child_removal(tmp_path, link_error):
    parent, task = tmp_path / "canonical", "link-task"
    child = headless.prepare_task_drive(parent, task, "empty")
    payload = {"messages": [{"role": "user", "content": "Retained Ω evidence"}]}
    call = observability.persist_call(child, task_id=task, call_id="request", call_type="llm_request", payload=payload)
    original = Path(call["redacted_projection_ref"]["path"])
    compressed, inode = original.read_bytes(), original.stat().st_ino
    _, state = observability.promote_child_task_refs(parent, child, task, {})
    assert state["status"] == "complete" and state["call_inventory_preserved"]
    destination = parent / "observability" / "blobs" / original.name
    assert destination.read_bytes() == compressed
    assert (destination.stat().st_ino == inode) is (link_error is None)
    if observability.posix_private_modes_supported():
        assert destination.stat().st_mode & 0o777 == 0o600
    for root in (parent, child):
        observability.write_blob(root, payload)
    assert original.stat().st_ino == inode and original.read_bytes() == compressed
    assert destination.read_bytes() == compressed
    shutil.rmtree(child)
    _, restored, _ = observability.read_call_payload(parent, task_id=task, call_id="request")
    assert restored == {**payload, "send_messages": payload["messages"]}


def test_failed_blob_publication_retains_child_until_retry(tmp_path, monkeypatch, link_error):
    from ouroboros import utils
    from ouroboros.task_results import write_task_result

    parent, task = tmp_path / "canonical", "failed-placement"
    child = headless.prepare_task_drive(parent, task, "empty")
    payload = {"result": "exact evidence survives failed publication"}
    observability.persist_call(child, task_id=task, call_id="call", call_type="tool", payload=payload)
    write_task_result(child, task, "completed", result="answer", artifact_status="ready")
    headless.copy_child_task_result(parent, {"id": task, "drive_root": str(child)})
    replace, attempts = utils.replace_atomic, []

    def fail_blob_publication(source, destination, *args, **kwargs):
        if destination.parent == parent / "observability" / "blobs":
            original = child / "observability" / "blobs" / destination.name
            assert source.read_bytes() == original.read_bytes()
            assert (source.stat().st_ino == original.stat().st_ino) is (link_error is None)
            attempts.append(source)
            raise OSError("blob publication failed")
        return replace(source, destination, *args, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(utils, "replace_atomic", fail_blob_publication)
        copied = headless.retry_child_task_refs(parent, child, task)
        assert copied["child_ref_promotion"]["status"] == "incomplete"
        assert copied["child_ref_promotion"]["pending_refs"]
        assert headless.remove_subagent_task_drive(parent, task, live=lambda _task: False) is False
        assert child.exists() and attempts and all(not path.exists() for path in attempts)
    copied = headless.retry_child_task_refs(parent, child, task)
    assert copied["child_ref_promotion"]["status"] == "complete"
    assert headless.remove_subagent_task_drive(parent, task, live=lambda _task: False) is True
    assert not child.exists()
    assert observability.read_call_payload(parent, task_id=task, call_id="call")[1] == payload

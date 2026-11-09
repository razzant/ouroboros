"""Immutable promotion shares storage without sharing a mutable publication."""

import errno
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from ouroboros import headless, observability, source_retention
from tests._packed_read_probe import count_packed_reads
from tests._store_shared import remove_store_tree


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
    # A record written whole (a request view is a skeleton placement plus parts: tests/test_shared_views.py).
    call = observability.persist_call(child, task_id=task, call_id="request", call_type="tool_call", payload=payload)
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
    assert restored == payload


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


def test_a_retry_after_the_parent_packed_its_copies_reads_each_member_once(tmp_path, monkeypatch):
    """The maintenance retry re-walks a child whose bytes the parent's packer already holds: each member is
    decoded once, as its node's destination proof or a view's own read, never again by the same node."""
    from ouroboros import evidence_packer, utils
    from ouroboros import evidence_packs as packs
    from ouroboros.task_results import write_task_result

    parent, task = tmp_path / "canonical", "packed-retry"
    child = headless.prepare_task_drive(parent, task, "empty")
    messages = [{"role": "system", "content": "SYSTEM Ω " * 16_000}, {"role": "user", "content": "Inspect it."}]
    payloads = {"request": ("llm_request", {"messages": messages, "send_messages": messages, "model": "m"}),
                "response": ("llm_response", {"message": {"role": "assistant", "content": "reading"}}),
                "tool": ("tool_call", {"tool": "read_file", "result": "x" * (160 * 1024)})}  # above any LZMAFile buffer
    for call_id, (call_type, payload) in payloads.items():
        observability.persist_call(child, task_id=task, call_id=call_id, call_type=call_type, payload=payload)
    write_task_result(child, task, "completed", result="answer", artifact_status="ready")
    headless.copy_child_task_result(parent, {"id": task, "drive_root": str(child)})
    manifest = json.loads(observability.call_manifest_path(child, task, "response").read_text(encoding="utf-8"))
    failing, replace = Path(manifest["full_payload_ref"]["path"]).name, utils.replace_atomic

    def fail_once(source, destination, *args, **kwargs):  # one transient publication keeps the row pending
        if Path(destination).name == failing:
            raise OSError("transient publication failure")
        return replace(source, destination, *args, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(utils, "replace_atomic", fail_once)
        assert headless.retry_child_task_refs(parent, child, task)["child_ref_promotion"]["status"] == "incomplete"
    assert evidence_packer.run_pass(parent)["packed"] == 1  # the parent's copies are pack members only now
    members = {(packs.pack_name(sidecar["task_id"], sidecar["generation"]), member["offset"])
               for _path, sidecar in packs.list_sidecars(parent) for member in sidecar["members"]}
    reads = count_packed_reads(monkeypatch)
    assert headless.retry_child_task_refs(parent, child, task)["child_ref_promotion"]["status"] == "complete"
    # Each member once: a second read rewinds its xz stream; a missing read would be an unverified skip.
    assert reads == Counter(dict.fromkeys(members, 1)), sorted(reads.items())
    remove_store_tree(child)  # the retries opened the child's own store; Windows removes no open file
    for call_id, (_call_type, payload) in payloads.items():
        restored = observability.read_call_payload(parent, task_id=task, call_id=call_id)[1]
        assert restored == json.loads(observability._json_bytes(observability.redact_projection(payload).value))

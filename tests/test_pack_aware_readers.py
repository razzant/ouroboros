"""Readers that announce a retained source test its presence through the pack-aware check, not the
loose file alone: once the packer has moved a terminal task's checkpoint handles into its pack, a
Continue successor still reads its predecessor's exact sources. A handle that is truly gone is a read
failure, never invented bytes. (The late acceptance review's wake line and the acceptance packet's
write-back for a session reader retired with the task-acceptance machine.)"""
from __future__ import annotations

import json

import pytest

from ouroboros import evidence_packer as packer, evidence_packs as packs
from ouroboros.artifacts import store_actor_source_bytes
from tests.test_evidence_packs import _handle_path, _result, root as root  # noqa: F401  (the shared fixture)


def test_a_successor_reads_its_predecessors_packed_sources_and_a_lost_one_fails_to_read(root):
    """The successor path (``main_context_authority.predecessor_notes_message``) reads the
    predecessor's retained sources through ``artifacts.read_actor_source_bytes``: once the packer
    has moved the terminal predecessor's handle into its pack, the same reader still returns the
    exact bytes; a handle that no pack ever held is a read failure, never invented bytes."""
    from ouroboros.artifacts import read_actor_source_bytes

    data = json.dumps({"decision": "keep the protocol"}).encode()
    ref = store_actor_source_bytes(root, "pred", category="context_checkpoints", source_id="plan-author-x",
                                   data=data, extension="json")
    _result(root, "pred", "failed")
    assert read_actor_source_bytes(root, "pred", ref) == data
    assert packer.run_pass(root)["packed"] == 1
    assert not _handle_path(root, "pred", ref).exists() and packs.handle_info(root, "pred", ref["path"])["loose"] is False
    assert read_actor_source_bytes(root, "pred", ref) == data  # a verified pack member: the same exact bytes
    lost = store_actor_source_bytes(root, "pred", category="context_checkpoints", source_id="plan-author-y",
                                    data=b'{"decision": "lost"}', extension="json")
    _handle_path(root, "pred", lost).unlink()  # gone before any pack held it
    with pytest.raises(OSError):  # a missing file, loose and retained alike: a typed read failure
        read_actor_source_bytes(root, "pred", lost)

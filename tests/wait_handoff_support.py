"""Resolve a compact wait's explicit retained source without discarding evidence."""
import hashlib
import json

from ouroboros.artifacts import read_actor_source_bytes


def full_wait_payload(ctx, text):
    assert len(text) <= 15_000
    view = json.loads(text)
    ref = view.get("complete_source")
    if ref:
        assert ref.get("sha256"), "the full-source consumer needs retained authority, not an unavailable index"
        data = read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref)
        assert hashlib.sha256(data).hexdigest() == ref["sha256"]
        assert len(data) == ref["size"]
        return json.loads(data)
    return view

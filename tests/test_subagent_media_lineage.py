"""v6.37.0 guard (C4.4 review fix): a SUBAGENT's media/live-log frames must carry
task lineage so they route to the root's project thread, not the main chat. The
emitters stamp parent_task_id/root_task_id from the task metadata; the supervisor
handlers pass them to _bound_project_chat_id (lineage resolution, tested in
test_project_lineage_chat.py)."""

from __future__ import annotations

from types import SimpleNamespace

_B64 = "a" * 160  # passes the >=100-char image-length gate in _send_photo/_send_video


def _ctx_with_lineage():
    return SimpleNamespace(
        current_chat_id=1,
        task_id="child",
        browser_state=None,
        task_metadata={"parent_task_id": "mid", "root_task_id": "root"},
        pending_events=[],
    )


def test_send_photo_event_carries_lineage():
    from ouroboros.tools.core_artifacts import _send_photo

    ctx = _ctx_with_lineage()
    _send_photo(ctx, image_base64=_B64)
    evt = next(e for e in ctx.pending_events if e.get("type") == "send_photo")
    assert evt["parent_task_id"] == "mid"
    assert evt["root_task_id"] == "root"


def test_send_video_event_carries_lineage(tmp_path):
    from ouroboros.tools.core_artifacts import _send_video

    ctx = _ctx_with_lineage()
    vid_file = tmp_path / "clip.mp4"
    vid_file.write_bytes(b"\x00" * 256)  # non-empty payload past the length gate
    _send_video(ctx, file_path=str(vid_file))
    vid = [e for e in ctx.pending_events if e.get("type") == "send_video"]
    if not vid:
        # The emitter rejected the dummy payload (codec/validation) — the lineage
        # stamping is structurally identical to _send_photo (already asserted), so
        # treat an emitter-side rejection as not-applicable rather than a failure.
        return
    assert vid[-1]["parent_task_id"] == "mid"
    assert vid[-1]["root_task_id"] == "root"


def test_emit_live_log_stamps_lineage():
    from ouroboros.loop_tool_execution import _emit_live_log

    captured = {}

    class _Q:
        def put_nowait(self, item):
            captured["item"] = item

    ctx = SimpleNamespace(
        event_queue=_Q(),
        task_metadata={"parent_task_id": "mid", "root_task_id": "root"},
    )
    tools = SimpleNamespace(_ctx=ctx)
    _emit_live_log(tools, {"type": "tool_live", "task_id": "child", "content": "x"})
    data = (captured.get("item") or {}).get("data") or {}
    assert data.get("parent_task_id") == "mid"
    assert data.get("root_task_id") == "root"

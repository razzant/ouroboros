"""The image auto-attach seam of the loop's tool-result processing.

Split out of ``tests/test_loop_misc.py`` when that module was divided by
theme; every moved block is verbatim.
"""
from __future__ import annotations


def test_tool_results_carrying_auto_attach_image_get_the_image_same_round(tmp_path, monkeypatch):
    """A result whose JSON offers `auto_attach_image` (the unix_computer_use screenshot)
    must have its image attached in the SAME round, through the same implementation
    view_image uses — removing the mandatory second round per observation that consumed
    ~21% of the round budget on computer-use benches (v6.81.0 OSWorld: 3,830 view_image
    rounds after 3,893 screenshots). Failure is strictly non-fatal: a bad path must not
    turn a successful screenshot into a failed tool call."""
    import json as _json
    from types import SimpleNamespace

    from ouroboros.loop_tool_execution import process_tool_results

    attached = []

    def fake_attach(ctx, path):
        attached.append(path)
        ctx.messages.append({"role": "user", "content": [{"type": "image_url"}]})
        return True, "attached"

    import ouroboros.tools.vision as vision
    monkeypatch.setattr(vision, "attach_local_image_to_context", fake_attach)

    messages: list = []
    tools = SimpleNamespace(_ctx=SimpleNamespace(messages=messages, drive_root=str(tmp_path)))
    ok_result = _json.dumps({"ok": True, "path": "/x/shot.png",
                             "auto_attach_image": "/x/shot.png"})
    rows = [
        {"fn_name": "ext_1_r_unix_computer_use_screenshot", "tool_call_id": "c1",
         "result": ok_result, "is_error": False, "args_for_log": {}, "tool_args": {},
         "result_meta": {}},
        # An ERROR result never attaches, even if the field is present.
        {"fn_name": "ext_1_r_unix_computer_use_screenshot", "tool_call_id": "c2",
         "result": ok_result, "is_error": True, "args_for_log": {}, "tool_args": {},
         "result_meta": {}},
        # A result without the field never attaches.
        {"fn_name": "read_file", "tool_call_id": "c3",
         "result": "plain text", "is_error": False, "args_for_log": {}, "tool_args": {},
         "result_meta": {}},
    ]
    # An MCP-shaped result carrying the field must NOT attach: the capability is
    # defined for first-party extension tools; MCP results are untrusted
    # server-supplied data that must not drive automatic context mutation.
    rows.append({"fn_name": "mcp__someserver__screenshot", "tool_call_id": "c9",
                 "result": ok_result, "is_error": False, "args_for_log": {},
                 "tool_args": {}, "result_meta": {}})
    errors = process_tool_results(rows, messages, {"tool_calls": []},
                                  emit_progress=lambda _m: None, tools=tools)
    assert attached == ["/x/shot.png"], "exactly the opted-in successful result attaches"
    assert errors == 1
    # Ordering: in a multi-result round the image lands AFTER the round's complete
    # tool-message block, never between two tool messages answering one assistant
    # turn — contiguity by construction, not by transport repair.
    roles = [m["role"] for m in messages]
    first_image = roles.index("user")
    assert roles[:first_image] == ["tool"] * 4, roles
    # Attachment failure stays non-fatal and the tool result survives untouched.
    monkeypatch.setattr(vision, "attach_local_image_to_context",
                        lambda ctx, path: (_ for _ in ()).throw(RuntimeError("boom")))
    messages2: list = []
    tools2 = SimpleNamespace(_ctx=SimpleNamespace(messages=messages2, drive_root=str(tmp_path)))
    errors2 = process_tool_results(
        [dict(rows[0], tool_call_id="c4")], messages2, {"tool_calls": []},
        emit_progress=lambda _m: None, tools=tools2)
    assert errors2 == 0 and messages2[0]["role"] == "tool"
    # Legacy callers without `tools` keep exactly the old behavior.
    errors3 = process_tool_results(
        [dict(rows[0], tool_call_id="c5")], [], {"tool_calls": []},
        emit_progress=lambda _m: None)
    assert errors3 == 0


def test_undecodable_image_fails_the_attach_not_the_provider_call():
    """A truncated PNG passes header checks; forwarding its bytes used to become a
    non-retryable provider 400 rounds later (5 task deaths in the v6.81.1 OSWorld
    run). The payload builder must raise at build time so the attach seam maps it
    to a tool-visible warning instead."""
    import io
    import pytest
    from PIL import Image

    from ouroboros.tools import vision

    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (1, 2, 3)).save(buf, format="PNG")
    good = buf.getvalue()
    corrupt = good[:40] + b"\x00" * 400

    with pytest.raises(ValueError, match="IMAGE_UNDECODABLE"):
        vision._downscale_image_for_vlm(corrupt, "image/png")
    out, mime = vision._downscale_image_for_vlm(good, "image/png")
    assert out == good and mime == "image/png"

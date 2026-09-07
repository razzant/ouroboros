"""Split-drive skill outputs reach the actual common image reader and send path."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ouroboros import config
from ouroboros.loop_tool_execution import process_tool_results
from ouroboros.tools import vision
from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send


@pytest.fixture
def split_image(tmp_path, monkeypatch):
    home = tmp_path / "home"
    canonical = home / "Ouroboros" / "data"
    child = canonical / "state" / "headless_tasks" / "child" / "data"
    child.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    # The producer's canonical root comes from the task, not this unrelated
    # process default. A home outside the fixture would miss the original bug.
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "unrelated-default")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "unrelated-env"))
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    shot = canonical / "state" / "skills" / "myskill" / "jobs" / "job" / "output" / "shot.png"
    shot.parent.mkdir(parents=True)
    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (23, 84, 140)).save(buf, format="PNG")
    shot.write_bytes(buf.getvalue())
    ctx = SimpleNamespace(
        repo_dir=home / "Ouroboros" / "repo", drive_root=child,
        budget_drive_root=canonical, task_metadata={"budget_drive_root": str(canonical)},
        task_id="child", current_task_type="task", event_queue=None, messages=[],
        active_model="google/gemini-3.5-flash",
    )
    return ctx, shot, buf.getvalue()


@pytest.mark.parametrize("constraint", [None, {"mode": "local_readonly_subagent"},
                                          {"mode": "acting_subagent", "surface": "external_workspace"}])
def test_skill_output_auto_attach_manual_read_and_vision_send(split_image, constraint, monkeypatch):
    ctx, shot, raw = split_image
    ctx.task_constraint = constraint
    result = json.dumps({"ok": True, "path": str(shot), "auto_attach_image": str(shot)})
    rows = [{"fn_name": "ext_1_r_myskill_screenshot", "tool_call_id": "s1", "result": result,
             "is_error": False, "args_for_log": {}, "tool_args": {}, "result_meta": {}}]
    assert process_tool_results(rows, ctx.messages, {"tool_calls": []},
                                lambda _m, *, incident=None: None,
                                tools=SimpleNamespace(_ctx=ctx)) == 0
    sent = prepare_messages_for_send(ctx.messages, routing=VisionRoutingContext(ctx.active_model, object(), {}))
    blocks = [b for m in sent if isinstance(m.get("content"), list) for b in m["content"]
              if b.get("type") == "image_url"]
    assert len(blocks) == 1
    assert base64.b64decode(blocks[0]["image_url"]["url"].split(",", 1)[1]) == raw
    copied = Path(blocks[0]["_source_path"])
    assert copied.is_relative_to(ctx.drive_root / "uploads" / "views")
    assert copied.read_bytes() == raw
    assert "now attached" in vision._view_image(ctx, str(shot))
    assert "now attached" in vision._view_image(ctx, str(copied))
    observed = []
    monkeypatch.setattr(vision, "_get_llm_client", lambda: object())
    monkeypatch.setattr(vision, "_vision_query_with_timeout",
                        lambda _client, **kw: (observed.append(kw) or "visible blue image", {}))
    assert vision._vlm_query(ctx, "inspect", file_path=str(shot), model=ctx.active_model) == "visible blue image"
    assert base64.b64decode(observed[0]["images"][0]["base64"]) == raw


@pytest.mark.parametrize("relative", ["state/skills/myskill/grants.json", "settings.json", "projects/p1/shot.png"])
def test_image_root_admission_preserves_per_path_denials(split_image, relative):
    ctx, _shot, raw = split_image
    ctx.task_constraint = {"mode": "local_readonly_subagent"}
    protected = ctx.budget_drive_root / relative
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_bytes(raw)
    payload, error = vision._load_local_image_payload(ctx, str(protected))
    assert payload is None and error
    assert not ctx.messages

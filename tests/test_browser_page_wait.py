"""Declarative page waits preserve native browser operations and observations."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools import browser, vision
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send


def test_wait_schema_preserves_evaluate_and_exposes_selector_states():
    entries = {entry.name: entry for entry in browser.get_tools()}
    for name in ("browse_page", "browser_action"):
        props = entries[name].schema["parameters"]["properties"]
        assert props["state"]["enum"] == ["attached", "visible", "hidden", "detached"]
    assert {"wait", "evaluate", "click", "screenshot"} <= set(
        entries["browser_action"].schema["parameters"]["properties"]["action"]["enum"])


@pytest.mark.browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_real_current_page_waits_and_timeout_observations(tmp_path, engine):
    pytest.importorskip("playwright.sync_api")
    # Readonly startup must use an installed browser; this test never installs.
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data", workspace_root=str(tmp_path),
                      task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False))
    tools = ToolRegistry(tmp_path, ctx.drive_root)
    tools.set_context(ctx)
    page = tmp_path / "index.html"
    page.write_text('''<!doctype html><html><head><title>Browser wait proof</title></head>
      <body style="font:24px sans-serif;background:#edf4ff;padding:30px">
      <h1>Browser wait proof</h1><button id="show" onclick="setTimeout(() => {
      document.querySelector('#ready').hidden=false; },100)">Show result</button>
      <p id="ready" hidden>Rendered result is ready</p>
      <button id="remove" onclick="document.querySelector('#ready').remove()">Remove result</button>
      </body></html>''', encoding="utf-8")
    try:
        text = tools.execute("browse_page", {"url": page.as_uri(), "engine": engine,
                                              "wait_for": "#ready", "state": "attached"})
        if "not already installed" in text:
            if engine in os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES", "").split(","):
                pytest.fail(text)
            pytest.skip(text)
        assert "Browser wait proof" in text
        original = ctx.browser_state.page
        timed = json.loads(tools.execute("browser_action", {
            "action": "wait", "selector": "#ready", "state": "visible", "timeout": 50}))
        assert timed == {"status": "timeout", "selector": "#ready", "requested_state": "visible",
                         "url": page.as_uri(), "matched_elements": 1, "first_visible": False}
        assert json.loads(tools.execute("browser_action", {
            "action": "wait", "selector": "#ready", "state": "hidden"}))["status"] == "reached"
        assert "Clicked" in tools.execute("browser_action", {"action": "click", "selector": "#show"})
        reached = json.loads(tools.execute("browser_action", {
            "action": "wait", "selector": "#ready", "state": "visible", "timeout": 3000}))
        assert reached["status"] == "reached" and reached["first_visible"] is True
        assert ctx.browser_state.page is original and original.url == page.as_uri()
        assert "Screenshot captured" in tools.execute("browser_action", {"action": "screenshot"})
        png = base64.b64decode(ctx.browser_state.last_screenshot_b64)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        shot = ctx.drive_root / "state/skills/browser-proof/jobs/wait/output/screen.png"
        shot.parent.mkdir(parents=True)
        shot.write_bytes(png)
        ctx.messages = []
        assert vision.attach_local_image_to_context(ctx, str(shot))[0]
        sent = prepare_messages_for_send(ctx.messages, routing=VisionRoutingContext(
            "google/gemini-3.5-flash", object(), {}))
        block = next(b for message in sent if isinstance(message.get("content"), list)
                     for b in message["content"] if b.get("type") == "image_url")
        prepared = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        assert block["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert shot.read_bytes() == png
        assert Path(block["_source_path"]).read_bytes() == prepared
        with Image.open(io.BytesIO(png)) as source, Image.open(io.BytesIO(prepared)) as received:
            assert source.size == (1920, 1080) and source.format == "PNG"
            assert received.size == (1600, 900) and received.format == "JPEG"
            assert all(abs(a - b) < 5 for a, b in zip(received.getpixel((1500, 800)), (237, 244, 255)))
        if out := os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT"):
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / f"page-wait-{engine}.png").write_bytes(png)
            (Path(out) / f"page-wait-{engine}-vision.jpg").write_bytes(prepared)
        assert "cannot run arbitrary" in tools.execute("browser_action", {"action": "evaluate", "value": "1+1"})
        assert "Clicked" in tools.execute("browser_action", {"action": "click", "selector": "#remove"})
        for state in ("hidden", "detached"):
            absent = json.loads(tools.execute("browser_action", {"action": "wait", "selector": "#ready", "state": state}))
            assert absent["status"] == "reached" and absent["matched_elements"] == 0
        # Acting evaluate remains a working operation; waits do not replace it.
        ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(tmp_path))
        assert tools.execute("browser_action", {"action": "evaluate", "value": "() => Promise.resolve(7)"}) == "7"
    finally:
        browser.cleanup_browser(ctx)

"""Generated ordinary artifacts through real Main/Project delivery and SPA replay."""
from __future__ import annotations

from tests import _dialogue as dialogue_test

import base64
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.test_imagegen_tool import configured as configured, _transport, _summary
from tests.test_chat_history_paging_browser import _open, _open_project, _idle, MAIN

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_generated_image_delivery_and_reload_in_main_and_project(direct_server_with_data, monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from playwright.sync_api import sync_playwright
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.utils import append_jsonl
    from supervisor import message_bus
    from supervisor.events_chat_delivery import _handle_send_photo

    server = direct_server_with_data
    root = server["data_dir"]
    project = create_project(root, "image-consumer", name="Image consumer")
    workspace = tmp_path / "image-workspace"; workspace.mkdir()
    bitmap = Image.new("RGB", (640, 360), "#132f42")
    draw = ImageDraw.Draw(bitmap)
    draw.rectangle((80, 70, 560, 290), fill="#4b9b89")
    draw.text((145, 165), "Ordinary generated task artifact", fill="white")
    buf = io.BytesIO(); bitmap.save(buf, format="PNG"); pixels = buf.getvalue()
    calls = []
    def provider(request):
        calls.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(pixels).decode()}]})
    _transport(monkeypatch, provider)
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "publish_event", lambda *a, **k: None)
    bridge = message_bus.LocalChatBridge()
    supervisor = SimpleNamespace(bridge=bridge, DRIVE_ROOT=root, append_jsonl=append_jsonl)
    refs = {}
    for label, chat_id in [("main", 1), ("project", project["chat_id"])]:
        ctx = ToolContext(repo_dir=server["repo_dir"], drive_root=root, task_id=f"image-{label}",
                          workspace_root=workspace, workspace_mode="external", current_chat_id=chat_id)
        registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=root); registry.set_context(ctx)
        result = registry.execute_result("generate_image", {"prompt": "Fixture illustration", "model": "openai::fixture-image"})
        assert result.status == "ok", result
        assert not ctx.pending_events
        image = _summary(result.text)["images"][0]
        assert Path(image["absolute_path"]).read_bytes() == pixels
        sent = registry.execute_result("send_photo", {"file_path": image["absolute_path"], "caption": f"Generated for {label}"})
        assert sent.status == "ok", sent
        assert len(ctx.pending_events) == 1
        _handle_send_photo(ctx.pending_events[0], supervisor)
        refs[label] = image
    assert len(calls) == 2
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", tmp_path / "evidence"))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            for width in (1280, 390):
                page = browser.new_page(viewport={"width": width, "height": 844})
                errors = []; page.on("pageerror", lambda error: errors.append(str(error)))
                try:
                    _open(page, server["url"])
                    for label, feed in [("main", MAIN), ("project", None)]:
                        if label == "project":
                            feed = _open_project(page, project)
                        image = page.locator(f"{feed} img.chat-photo").last
                        image.wait_for(state="visible")
                        page.wait_for_function("selector => { const image = document.querySelector(selector); return image?.complete && image.naturalWidth === 640; }",
                                               arg=f"{feed} img.chat-photo")
                        assert f"Generated for {label}" in page.locator(feed).inner_text()
                        image.scroll_into_view_if_needed()
                        page.screenshot(path=str(evidence / f"image-{label}-{width}.png"))
                        photo = next(row for row in map(json.loads, dialogue_test.log_text(root / "logs" / "chat.jsonl").splitlines())
                                     if row.get("type") == "photo" and row.get("task_id") == f"image-{label}")
                        fetched = page.request.get(server["url"] + photo["download_url"])
                        assert fetched.status == 200 and fetched.body() == pixels
                    page.reload(wait_until="domcontentloaded")
                    _idle(page, MAIN)
                    assert page.locator(f"{MAIN} img.chat-photo").count() == 1
                    assert "Generated for project" not in page.locator(MAIN).inner_text()
                    assert not errors
                finally:
                    page.close()
        finally:
            browser.close()
    assert len(calls) == 2


def test_generated_svg_opens_from_its_file_card_as_an_inert_static_image(direct_server_with_data, monkeypatch, tmp_path):
    """generate_image -> send_file -> the real card's Open on this application origin."""
    from playwright.sync_api import sync_playwright
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.utils import append_jsonl
    from supervisor import message_bus
    from supervisor.events_chat_delivery import _handle_send_document

    server = direct_server_with_data
    root = server["data_dir"]
    workspace = tmp_path / "svg-workspace"; workspace.mkdir()
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" '
           b'fill="#4b9b89"/><script>new Image().src = "/svg-script-ran?origin=" + encodeURIComponent(location.origin)'
           b'</script></svg>')
    _transport(monkeypatch, lambda _request: httpx.Response(200, json={"data": [
        {"b64_json": base64.b64encode(svg).decode(), "media_type": "image/svg+xml"}]}))
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "publish_event", lambda *a, **k: None)
    supervisor = SimpleNamespace(bridge=message_bus.LocalChatBridge(), DRIVE_ROOT=root, append_jsonl=append_jsonl)
    ctx = ToolContext(repo_dir=server["repo_dir"], drive_root=root, task_id="image-svg",
                      workspace_root=workspace, workspace_mode="external", current_chat_id=1)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=root); registry.set_context(ctx)
    result = registry.execute_result("generate_image", {"prompt": "Fixture vector", "model": "openai::fixture-image"})
    assert result.status == "ok", result
    image = _summary(result.text)["images"][0]
    assert image["mime"] == "image/svg+xml" and Path(image["absolute_path"]).read_bytes() == svg
    sent = registry.execute_result("send_file", {"file_path": image["absolute_path"], "caption": "Generated vector"})
    assert sent.status == "ok", sent
    _handle_send_document(ctx.pending_events[-1], supervisor)
    row = next(row for row in map(json.loads, dialogue_test.log_text(root / "logs" / "chat.jsonl").splitlines())
               if row.get("task_id") == "image-svg" and row.get("download_url"))
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", tmp_path / "evidence"))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            context = browser.new_context(viewport={"width": 1280, "height": 844})
            requests, served = [], {}
            context.on("request", lambda request: requests.append(request.url))
            context.on("response", lambda response: served.update(response.headers)
                       if response.url.endswith(row["download_url"]) and response.request.resource_type == "document" else None)
            page = context.new_page()
            errors = []; page.on("pageerror", lambda error: errors.append(str(error)))
            _open(page, server["url"])
            card = page.locator(f"{MAIN} .chat-file-card").last
            card.wait_for(state="visible")
            # Static view: the card's thumbnail still renders the SVG as an image.
            page.wait_for_function("selector => { const image = [...document.querySelectorAll(selector)].pop(); "
                                   "return image?.complete && image.naturalWidth === 320; }", arg=f"{MAIN} img.chat-file-thumb")
            card.click()
            with page.expect_popup() as opened:
                page.locator('dialog.chat-file-dialog [data-file-action="open"]').click()
            popup = opened.value
            popup.wait_for_load_state("load")
            popup.wait_for_timeout(1500)  # a script that ran would have requested its beacon by now
            popup.screenshot(path=str(evidence / "svg-opened-1280.png"))
            observed = {"popup": popup.url, "script_beacons": [url for url in requests if "svg-script-ran" in url],
                        "csp": served.get("content-security-policy"), "nosniff": served.get("x-content-type-options")}
            assert popup.url.endswith(row["download_url"]) and not observed["script_beacons"], observed
            assert (observed["csp"], observed["nosniff"]) == ("sandbox", "nosniff"), observed
            fetched = page.request.get(server["url"] + row["download_url"])
            assert fetched.status == 200 and fetched.body() == svg
            assert fetched.headers["content-type"].startswith("image/svg+xml")
            assert not errors
        finally:
            browser.close()

"""Real upload -> task tool -> supervisor document event -> UI/native download."""
from __future__ import annotations

import ast
import base64
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import urllib.parse
import urllib.request

import pytest
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

pytest_plugins = ("tests.test_ui_smoke_playwright",)


def _digest(path):
    result = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _native_file_api(port, read_sizes):
    """Execute the launcher's real nested API without starting another application."""
    import pathlib
    source = Path(__file__).resolve().parents[1] / "launcher.py"
    names = {"_resolve_bridge_file_url", "_unique_bridge_target", "_fetch_bridge_url_to", "MainApi"}
    selected = [node for node in ast.walk(ast.parse(source.read_text()))
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == names

    def open_recorded(*args, **kwargs):
        response = urllib.request.urlopen(*args, **kwargs)
        original = response.read
        def read(size=-1):
            assert 0 < size <= 1024 * 1024, "native download must stream bounded reads"
            read_sizes.append(size)
            return original(size)
        response.read = read
        return response

    from ouroboros.launcher_background import DesktopApi

    # MainApi's real base, and `background` as a headless launch leaves it (no window, no indicator).
    namespace = {"actual_port": port, "pathlib": pathlib, "shutil": shutil,
                 "tempfile": tempfile, "base64": base64, "log": logging.getLogger(__name__),
                 "DesktopApi": DesktopApi, "background": None,
                 "urllib": SimpleNamespace(parse=urllib.parse, request=SimpleNamespace(urlopen=open_recorded))}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["MainApi"](), sha256(source.read_bytes()).hexdigest()


@pytest.mark.ui_browser
@pytest.mark.serial
def test_large_attachment_returns_through_real_document_handler_and_download(
    direct_server_with_data, monkeypatch, tmp_path,
):
    from playwright.sync_api import sync_playwright
    from tests import fixtures_mock_llm

    root = direct_server_with_data["data_dir"]
    settings_path = root / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["OUROBOROS_FILE_BROWSER_DEFAULT"] = str(root)
    settings_path.write_text(json.dumps(settings))
    direct_server_with_data["restart_server"]()
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "large-delivery.bin"
    with source.open("wb") as handle:
        for _ in range(51):
            handle.write(b"x" * (1024 * 1024))
        handle.write(b"complete last bytes")
    expected = {"size": source.stat().st_size, "sha256": _digest(source)}
    attachments = [source]
    for index in range(27):
        path = tmp_path / f"support-{index:02}.txt"
        path.write_text(f"input {index}")
        attachments.append(path)
    calls, model_calls = [], []
    state = {"sent": False}

    def response_handler(handler):
        request = json.loads(handler.rfile.read(int(handler.headers.get("content-length") or 0)))
        names = {tool.get("function", {}).get("name") for tool in request.get("tools") or []}
        model_calls.append({"tool_names": sorted(name for name in names if name), "file_requested": state["sent"]})
        message = {"role": "assistant", "content": "File delivery is complete."}
        if not state["sent"] and "send_file" in names:
            staged = list((root / "task_results" / "artifacts").glob("*/attachments/*large-delivery.bin"))
            if staged:
                assert len(staged) == 1
                state["sent"] = True
                calls.append({"tool": "send_file", "path": str(staged[0])})
                message = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "large-file-call", "type": "function", "function": {
                        "name": "send_file", "arguments": json.dumps({"file_path": str(staged[0]), "caption": "Complete large dataset"}),
                    },
                }]}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        payload = {"id": "mock-large-file", "object": "chat.completion",
                   "model": request.get("model") or "mock-model",
                   "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        content_type = "application/json"
        if request.get("stream"):
            # The main loop streams every completion: answer in SSE frames with
            # the terminal framing the assembler requires.
            content_type = "text/event-stream"
            delta = dict(message)
            if delta.get("tool_calls"):
                delta["tool_calls"] = [dict(call, index=index)
                                       for index, call in enumerate(delta["tool_calls"])]
            common = {"id": payload["id"], "model": payload["model"], "object": "chat.completion.chunk"}
            frames = [{**common, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
                      {**common, "choices": [], "usage": payload["usage"]}]
            data = ("".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
                    + "data: [DONE]\n\n").encode()
        else:
            data = json.dumps(payload).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", response_handler)
    url = direct_server_with_data["url"]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            document_frames = []
            def websocket(socket):
                def frame(payload):
                    try:
                        value = json.loads(payload)
                    except (ValueError, TypeError):
                        return
                    if value.get("type") == "document":
                        document_frames.append(value)
                socket.on("framereceived", frame)
            page.on("websocket", websocket)
            requests = []
            page.on("request", lambda request: requests.append((request.method, request.url)))
            page.goto(url, wait_until="domcontentloaded")
            page.locator("#chat-input").wait_for(state="visible")
            page.wait_for_function(
                "() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
            page.locator("#chat-file-input").set_input_files([str(path) for path in attachments])
            assert page.locator(".attach-badge").count() == 28
            page.locator("#chat-input").fill("Return the large attached dataset as a downloadable document.")
            page.locator("#chat-send").click()
            # The delivered document, not the owner's own attachment card of the same name.
            card = page.locator(".chat-bubble.assistant .chat-file-card").filter(has_text="large-delivery.bin")
            card.wait_for(state="visible", timeout=60_000)
            assert calls and len(document_frames) == 1
            frame = document_frames[0]
            assert frame["file_base64"] == ""
            assert {key: frame["file_ref"][key] for key in expected} == expected
            assert frame["size_bytes"] == expected["size"]
            from ouroboros.artifacts import resolve_attachment_manifest
            from ouroboros.task_results import load_task_result
            task_id = frame["task_id"]
            result = load_task_result(root, task_id)
            assert result and result["task_contract"]["attachment_manifest_ref"]["count"] == 28
            assert len(resolve_attachment_manifest(root, task_id, result["task_contract"])) == 28
            card.scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / "large-document-desktop.png"))
            card.click()
            with page.expect_download() as download_event:
                page.locator('.chat-file-dialog[open] [data-file-action="download"]').click()
            downloaded = download_event.value
            assert downloaded.failure() is None
            target = tmp_path / "browser-download.bin"
            downloaded.save_as(str(target))
            assert target.stat().st_size == expected["size"] and _digest(target) == expected["sha256"]
            actual_url = url + frame["download_url"]
            assert ("HEAD", actual_url) in requests
            assert downloaded.url == actual_url  # Browser-managed GET is not always a page request event.
            page.locator('.chat-file-dialog[open]').wait_for(state="detached")
            page.set_viewport_size({"width": 390, "height": 844})
            card.scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / "large-document-mobile.png"))
            assert card.evaluate("node => node.getBoundingClientRect().right <= window.innerWidth")
            native_home = tmp_path / "native-home"
            monkeypatch.setenv("HOME", str(native_home))
            monkeypatch.setenv("USERPROFILE", str(native_home))
            read_sizes = []
            api, launcher_hash = _native_file_api(urllib.parse.urlparse(url).port, read_sizes)
            saved = api.download_file_to_downloads(frame["download_url"], "dataset-native.bin")
            assert saved["ok"], saved
            native = Path(saved["path"])
            assert native.is_relative_to(native_home / "Downloads")
            assert native.stat().st_size == expected["size"] and _digest(native) == expected["sha256"]
            history = page.request.get(url + "/api/chat/history?chat_id=1").json()
            history_rows = history if isinstance(history, list) else history.get("messages", [])
            replay = next(row for row in history_rows if row.get("task_id") == task_id and row.get("msg_type") == "document")
            assert replay["download_url_compat"] == frame["download_url_compat"] and replay["download_url_compat"]
            saved_compat = api.download_file_to_downloads(replay["download_url_compat"], "dataset-compat.bin")
            assert saved_compat["ok"] and _digest(Path(saved_compat["path"])) == expected["sha256"]
            saved_blob = api.save_bytes_to_downloads("../owned-blob.txt", base64.b64encode(b"client-owned blob").decode())
            assert saved_blob["ok"] and Path(saved_blob["path"]).parent == native_home / "Downloads"
            assert Path(saved_blob["path"]).read_bytes() == b"client-owned blob"
            assert read_sizes and max(read_sizes) <= 1024 * 1024
            (evidence / "large-document-proof.json").write_text(json.dumps({
                "expected": expected, "frame": frame, "launcher_sha256": launcher_hash,
                "native_read_max": max(read_sizes), "native_read_count": len(read_sizes),
                "browser_download_verified": True, "native_download_verified": True,
                "attachment_count": 28, "actual_handler": "supervisor.events_chat_delivery._handle_send_document",
            }, indent=2) + "\n")
        finally:
            browser.close()
            (evidence / "model-calls.json").write_text(json.dumps(model_calls, indent=2) + "\n")
            (evidence / "browser-requests.json").write_text(json.dumps(requests, indent=2) + "\n")
            for name in ("tools.jsonl", "events.jsonl", "progress.jsonl", "supervisor.jsonl", "server.log"):
                path = root / "logs" / name
                if path.is_file():
                    shutil.copy2(path, evidence / name)

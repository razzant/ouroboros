"""Real Files/Project consumers with deterministic gateway replies and no paid work."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401


pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def _browser(pw, engine):
    try:
        return getattr(pw, engine).launch(headless=True)
    except Exception as exc:
        if "Executable doesn't exist" in str(exc):
            pytest.skip(f"Installed {engine} unavailable: {exc}")
        raise


def _capture(page, name):
    evidence = os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT")
    if evidence:
        destination = Path(evidence)
        destination.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(destination / f"{name}.png"))


def _files_gateway(page):
    page.on("pageerror", lambda error: print(f"Files browser error: {error}"))
    fixture = {
        "writes": [], "pending": [], "hold_write": False,
        "write_error": "", "list_error": "",
        "uploads": 0, "upload_error": "Upload refused.", "directories": [], "deleted": [],
        "hold_navigation": False, "pending_navigation": [],
        "hold_operation": "", "pending_operations": [], "transfers": [], "list_paths": [], "upload_paths": [],
    }
    root = "/workspace/" + "long-project-name-" * 5
    names = ["sample.txt", "other.txt", "unreadable.txt", "picture.png", "paper.pdf", "archive.bin", "large.txt"]

    def reply(route):
        url = urlparse(route.request.url)
        path = parse_qs(url.query).get("path", ["."])[0]
        if url.path.endswith("/list"):
            fixture["list_paths"].append(path)
            if fixture["list_error"]:
                route.fulfill(status=503, json={"error": fixture["list_error"]})
                return
            entries = ([{"name": name, "path": name, "type": "file", "size": 31} for name in names]
                       + [{"name": "docs", "path": "docs", "type": "dir"}]) if path == "." else []
            data = {
                "path": path, "parent_path": ".", "root_path": root,
                "display_path": root if path == "." else f"{root}/{path}",
                "breadcrumb": [{"name": "Workspace", "path": "."}], "entries": entries,
            }
            if fixture["hold_navigation"] and path == "docs":
                fixture["pending_navigation"].append((route, data))
            else:
                route.fulfill(json=data)
        elif url.path.endswith("/read"):
            if path == "unreadable.txt":
                route.fulfill(status=403, json={"error": "File is unavailable."})
                return
            payload = {"path": path, "name": path, "size": 31, "root_path": root,
                       "display_path": f"{root}/{path}", "is_text": True,
                       "content": "Original file contents\nSecond line\n"}
            if path in {"picture.png", "paper.pdf", "archive.bin"}:
                payload.update(is_text=False, content="")
            if path == "picture.png":
                payload.update(is_image=True, media_type="image/png", content_url="/api/files/content?path=picture.png")
            if path == "paper.pdf":
                payload.update(is_pdf=True, content_url="/api/files/content?path=paper.pdf")
            if path == "large.txt":
                payload.update(truncated=True)
            if fixture["hold_navigation"] and path == "other.txt":
                fixture["pending_navigation"].append((route, payload))
            else:
                route.fulfill(json=payload)
        elif url.path.endswith("/write"):
            payload = route.request.post_data_json
            fixture["writes"].append(payload)
            if fixture["hold_write"]:
                fixture["pending"].append(route)
            elif fixture["write_error"]:
                route.fulfill(status=400, json={"error": fixture["write_error"]})
            else:
                route.fulfill(json={"path": payload["path"], "name": payload["path"].split("/")[-1],
                                    "display_path": f'{root}/{payload["path"]}', "size": len(payload["content"])})
        elif url.path.endswith("/mkdir"):
            fixture["directories"].append(route.request.post_data_json)
            route.fulfill(json={"path": "created-dir", "name": "created-dir", "type": "dir"})
        elif url.path.endswith("/upload"):
            fixture["uploads"] += 1
            body = route.request.post_data_buffer.decode("utf-8")
            fixture["upload_paths"].append(re.search(r'name="path"\r\n\r\n([^\r]*)', body).group(1))
            if fixture["upload_error"]:
                route.fulfill(status=400, json={"error": fixture["upload_error"]})
            else:
                names.append("upload.txt")
                data = {"path": "upload.txt", "name": "upload.txt", "size": 11}
                if fixture["hold_operation"] == "upload":
                    fixture["pending_operations"].append((route, data))
                else:
                    route.fulfill(json=data)
        elif url.path.endswith("/transfer"):
            fixture["transfers"].append(route.request.post_data_json)
            data = {"path": "other-copy.txt", "name": "other-copy.txt", "type": "file"}
            if fixture["hold_operation"] == "transfer":
                fixture["pending_operations"].append((route, data))
            else:
                route.fulfill(json=data)
        elif url.path.endswith("/delete"):
            deleted = route.request.post_data_json["path"]
            fixture["deleted"].append(deleted)
            names.remove(deleted)
            data = {"path": deleted, "type": "file", "ok": True}
            if fixture["hold_operation"] == "delete":
                fixture["pending_operations"].append((route, data))
            else:
                route.fulfill(json=data)
        elif url.path.endswith("/content"):
            if path == "picture.png":
                route.fulfill(content_type="image/png", body=base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                ))
            else:
                route.fulfill(status=200, body="", content_type="application/pdf")
        else:
            route.fulfill(status=404, json={"error": "Unexpected file operation in this case."})

    page.route("**/api/files/**", reply)
    return fixture


def _same_draft(page, value, selection=None):
    assert page.evaluate("window.filesEditor === document.querySelector('.files-editor')")
    assert page.locator(".files-editor").input_value() == value
    if selection:
        assert page.locator(".files-editor").evaluate("el => [el.selectionStart, el.selectionEnd]") == selection


def _drop_file(page):
    page.evaluate("""() => {
        const transfer = new DataTransfer();
        transfer.items.add(new File(['upload body'], 'upload.txt', {type:'text/plain'}));
        document.querySelector('.files-layout').dispatchEvent(new DragEvent('drop', {
            bubbles:true, cancelable:true, dataTransfer:transfer,
        }));
    }""")


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_files_retains_document_and_submits_one_write(direct_server_with_data, engine):  # noqa: F811
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = _browser(pw, engine)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            fixture = _files_gateway(page)
            page.goto(direct_server_with_data["url"] + "/#files", wait_until="domcontentloaded")
            page.locator(".files-entry").filter(has_text="sample.txt").click()
            editor = page.get_by_role("textbox", name="File contents", exact=True)
            editor.fill("Draft before save\nKeep this text.")
            editor.evaluate("el => { window.filesEditor = el; el.setSelectionRange(2, 8); }")
            page.locator("#files-refresh").click()
            expect(page.locator(".files-entry.selected")).to_contain_text("sample.txt")
            _same_draft(page, "Draft before save\nKeep this text.", [2, 8])
            expect(page.locator("#files-save")).to_be_enabled()

            for action in ["copy", "move"]:
                page.locator(".files-entry").filter(has_text="other.txt").click(button="right")
                page.locator(f'#files-context-menu [data-action="{action}"]').click()
                expect(page.locator("#files-preview-status")).to_contain_text(f"{action.title()} ready")
                _same_draft(page, "Draft before save\nKeep this text.", [2, 8])
                expect(page.locator(".files-entry.selected")).to_contain_text("sample.txt")
                expect(page.locator("#files-save")).to_be_enabled()

            row = page.locator(".files-entry").filter(has_text="other.txt")
            row.focus()
            row.press("Shift+F10")
            expect(page.get_by_role("menuitem", name="Download", exact=True)).to_be_focused()
            page.keyboard.press("ArrowDown")
            expect(page.get_by_role("menuitem", name="Copy", exact=True)).to_be_focused()
            page.keyboard.press("End")
            expect(page.get_by_role("menuitem", name="Delete", exact=True)).to_be_focused()
            page.keyboard.press("Home")
            expect(page.get_by_role("menuitem", name="Download", exact=True)).to_be_focused()
            page.keyboard.press("Escape")
            expect(page.locator("#files-context-menu")).to_be_hidden()
            expect(row).to_be_focused()
            row.press("Shift+F10")
            page.keyboard.press("End")
            page.keyboard.press("Enter")
            page.get_by_role("button", name="Stay", exact=True).click()
            expect(row).to_be_focused()
            _same_draft(page, "Draft before save\nKeep this text.")

            page.locator("#nav-projects-add").click()
            page.locator("[data-np-name]").fill("Keep name")
            page.locator("[data-np-name]").press("Home")
            page.locator("[data-np-name]").press("Delete")
            expect(page.locator(".confirm-dialog-backdrop")).to_have_count(0)
            expect(page.locator(".new-project-backdrop")).to_be_visible()
            page.keyboard.press("Escape")
            _same_draft(page, "Draft before save\nKeep this text.")

            page.evaluate("""() => {
                window.downloadCalls = [];
                window.pywebview = {api: {download_file_to_downloads: async (...args) => {
                    window.downloadCalls.push(args); return {ok:true, path:'/Downloads/sample.txt'};
                }}};
            }""")
            page.locator("#files-download").click()
            expect(page.locator("#files-preview-status")).to_contain_text("saved to /Downloads/sample.txt")
            page.locator("#files-open-external").click()
            expect(page.locator("#files-preview-status")).to_contain_text("Opened sample.txt externally")
            _same_draft(page, "Draft before save\nKeep this text.", [2, 8])
            assert page.evaluate("window.downloadCalls.map(call => call[2])") == [False, True]

            page.locator("#files-new-dir").click()
            page.locator("[data-confirm-input]").fill("created-dir")
            page.get_by_role("button", name="Create", exact=True).click()
            expect(page.locator("#files-preview-status")).to_contain_text("Directory created-dir created")
            _same_draft(page, "Draft before save\nKeep this text.", [2, 8])
            assert fixture["directories"] == [{"path": ".", "name": "created-dir"}]

            _drop_file(page)
            page.get_by_role("button", name="Discard", exact=True).click()
            expect(page.locator("#files-preview-status")).to_contain_text("Upload refused")
            _same_draft(page, "Draft before save\nKeep this text.")
            assert fixture["uploads"] == 1

            page.locator(".files-entry").filter(has_text="docs").click()
            page.get_by_role("button", name="Stay", exact=True).click()
            _same_draft(page, "Draft before save\nKeep this text.", [2, 8])
            expect(page.locator(".files-entry.selected")).to_contain_text("sample.txt")
            expect(page.locator("#files-save")).to_be_enabled()

            page.locator(".files-entry").filter(has_text="unreadable.txt").click()
            page.get_by_role("button", name="Discard", exact=True).click()
            expect(page.locator("#files-preview-status")).to_contain_text("File is unavailable")
            _same_draft(page, "Draft before save\nKeep this text.")
            expect(page.locator(".files-entry.selected")).to_contain_text("sample.txt")

            page.locator("#files-new-file").click()
            page.get_by_role("button", name="Discard", exact=True).click()
            name = page.get_by_role("textbox", name="File name", exact=True)
            name.fill("missing/notes.txt")
            editor.fill("New file draft")
            editor.evaluate("el => { window.filesEditor = el; el.setSelectionRange(3, 6); }")
            fixture["write_error"] = "Parent directory not found."
            page.locator("#files-save").click()
            expect(page.locator("#files-preview-status")).to_contain_text("Parent directory not found")
            _same_draft(page, "New file draft", [3, 6])
            assert name.input_value() == "missing/notes.txt"
            expect(name).to_be_enabled()
            expect(page.locator("#files-save")).to_be_enabled()
            _capture(page, f"files-{engine}-failed-save")

            fixture["write_error"] = ""
            fixture["hold_write"] = True
            name.fill("notes.txt")
            editor.focus()
            editor.press("Control+s")
            expect(page.locator("#files-save")).to_be_disabled()
            expect(name).to_be_disabled()
            editor.press("Control+s")
            editor.evaluate("el => el.dispatchEvent(new KeyboardEvent('keydown', {key:'s', ctrlKey:true, repeat:true, bubbles:true, cancelable:true}))")
            editor.fill("Newer text typed while saving")
            assert len(fixture["writes"]) == 2
            assert len(fixture["pending"]) == 1
            assert fixture["writes"][-1] == {"path": "notes.txt", "content": "New file draft", "create": True}
            fixture["pending"].pop().fulfill(json={"path": "notes.txt", "name": "notes.txt", "size": 14})
            expect(page.locator("#files-preview-status")).to_contain_text("Newer edits are still unsaved")
            _same_draft(page, "Newer text typed while saving")
            expect(page.locator("#files-save")).to_be_enabled()
            assert name.count() == 0

            fixture["hold_write"] = False
            editor.press("Meta+s")
            expect(page.locator("#files-preview-status")).to_have_text("Saved.")
            expect(page.locator("#files-save")).to_be_disabled()
            assert len(fixture["writes"]) == 3
            assert fixture["writes"][-1] == {"path": "notes.txt", "content": "Newer text typed while saving", "create": False}
            editor.press("Meta+s")
            assert len(fixture["writes"]) == 3
            _same_draft(page, "Newer text typed while saving")

            editor.fill("Draft survives list error")
            fixture["list_error"] = "Directory temporarily unavailable."
            page.locator("#files-refresh").click()
            expect(page.locator("#files-preview-status")).to_contain_text("Directory temporarily unavailable")
            _same_draft(page, "Draft survives list error")
            expect(page.locator("#files-save")).to_be_enabled()
            page.locator('[data-nav-page="dashboard"]').click()
            page.get_by_role("button", name="Stay", exact=True).click()
            expect(page.locator("#page-files")).to_have_class(re.compile(r"\bactive\b"))
            _same_draft(page, "Draft survives list error")

            fixture["list_error"] = ""
            fixture["upload_error"] = ""
            _drop_file(page)
            page.get_by_role("button", name="Discard", exact=True).click()
            expect(page.locator("#files-preview-meta")).to_contain_text("Upload complete")
            uploaded = page.locator(".files-entry").filter(has_text="upload.txt")
            uploaded.click()
            expect(page.locator(".files-editor")).to_have_value("Original file contents\nSecond line\n")
            expect(page.locator("#files-save")).to_be_disabled()
            uploaded.focus()
            uploaded.press("Delete")
            page.get_by_role("button", name="Delete", exact=True).click()
            expect(uploaded).to_have_count(0)
            expect(page.locator("#files-preview-status")).to_contain_text("File deleted")
            expect(page.locator("#files-download")).to_be_hidden()
            expect(page.locator("#files-save")).to_be_hidden()
            assert fixture["uploads"] == 2
            assert fixture["deleted"] == ["upload.txt"]

            page.locator(".files-entry").filter(has_text="sample.txt").click()
            editor.fill("Draft before navigation")
            editor.evaluate("el => { window.filesEditor = el; }")
            fixture["hold_navigation"] = True
            for target in ["other.txt", "docs"]:
                page.locator(".files-entry").filter(has_text=target).click()
                page.get_by_role("button", name="Discard", exact=True).click()
                editor.fill(f"New edits while {target} loads")
                assert len(fixture["pending_navigation"]) == 1
                pending, payload = fixture["pending_navigation"].pop()
                pending.fulfill(json=payload)
                expect(page.locator("#files-preview-status")).to_contain_text("New edits kept")
                _same_draft(page, f"New edits while {target} loads")
                expect(page.locator(".files-entry.selected")).to_contain_text("sample.txt")
                expect(page.locator("#files-save")).to_be_enabled()
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_files_late_mutations_keep_newer_work(direct_server_with_data, engine):  # noqa: F811
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = _browser(pw, engine)
        try:
            for operation in ["upload", "transfer", "delete"]:
                for newer in ["typing", "selection", "directory"]:
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    fixture = _files_gateway(page)
                    fixture.update(hold_operation=operation, upload_error="")
                    page.goto(direct_server_with_data["url"] + "/#files", wait_until="domcontentloaded")
                    page.locator(".files-entry").filter(has_text="sample.txt").click()
                    editor = page.locator(".files-editor")
                    editor.fill("Initial draft")
                    if operation == "upload":
                        _drop_file(page)
                    else:
                        page.locator(".files-entry").filter(has_text="other.txt").click(button="right")
                        if operation == "transfer":
                            page.locator('#files-context-menu [data-action="copy"]').click()
                            page.locator("#files-paste").click()
                        else:
                            page.locator('#files-context-menu [data-action="delete"]').click()
                    page.get_by_role("button", name="Discard", exact=True).click()
                    if operation == "delete":
                        page.get_by_role("button", name="Delete", exact=True).click()
                    expect(page.locator(".confirm-dialog-backdrop")).to_have_count(0)
                    assert len(fixture["pending_operations"]) == 1
                    if newer == "selection":
                        page.locator(".files-entry").filter(has_text="large.txt").click()
                        page.get_by_role("button", name="Discard", exact=True).click()
                        expect(page.locator("#files-preview-path")).to_contain_text("large.txt")
                    elif newer == "directory":
                        page.locator(".files-entry").filter(has_text="docs").click()
                        page.get_by_role("button", name="Discard", exact=True).click()
                        expect(page.locator("#files-preview-path")).to_contain_text("/docs")
                    else:
                        editor.fill("Newer input while action is pending")
                        editor.evaluate("el => {window.filesEditor = el; el.setSelectionRange(4, 9);}")
                    page.locator("#files-preview-content").evaluate("el => {window.pendingPreview = el.firstChild;}")
                    expected_path = page.locator("#files-preview-path").inner_text()
                    list_count = len(fixture["list_paths"])
                    pending, data = fixture["pending_operations"].pop()
                    pending.fulfill(json=data)
                    expected_status = {"upload": "Upload complete", "transfer": "Copied", "delete": "File deleted"}[operation]
                    expect(page.locator("#files-preview-status")).to_contain_text(expected_status)
                    assert page.locator("#files-preview-path").inner_text() == expected_path
                    assert page.evaluate("window.pendingPreview === document.querySelector('#files-preview-content').firstChild")
                    if newer == "typing":
                        _same_draft(page, "Newer input while action is pending", [4, 9])
                        expect(page.locator("#files-save")).to_be_enabled()
                    else:
                        assert len(fixture["list_paths"]) == list_count, "Completion must not cancel/navigate the newer view"
                    if operation == "transfer":
                        assert fixture["transfers"] == [{"source_path": "other.txt", "destination_dir": ".", "mode": "copy"}]
                    if operation == "upload":
                        assert fixture["uploads"] == 1
                        assert fixture["upload_paths"] == ["."]
                    if operation == "delete":
                        assert fixture["deleted"] == ["other.txt"]
                    if operation == "upload" and newer == "typing":
                        _capture(page, f"files-{engine}-late-upload-draft")
                    page.close()
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_files_preview_modes_keep_write_capability_honest(direct_server_with_data, engine):  # noqa: F811
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = _browser(pw, engine)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            fixture = _files_gateway(page)
            page.goto(direct_server_with_data["url"] + "/#files", wait_until="domcontentloaded")
            for filename, selector in [("picture.png", ".files-preview-image"), ("paper.pdf", ".files-preview-frame"),
                                       ("archive.bin", "#files-preview-content"), ("large.txt", "#files-preview-content")]:
                page.locator(".files-entry").filter(has_text=filename).click()
                expect(page.locator("#files-preview-path")).to_contain_text(filename)
                expect(page.locator(selector)).to_be_visible()
                expect(page.locator("#files-save")).to_be_hidden()
                expect(page.locator("#files-download")).to_be_visible()
                expect(page.locator("#files-open-external")).to_be_visible()
                assert page.locator(".files-editor").count() == 0
            assert fixture["writes"] == []
            assert "truncated" in page.locator("#files-preview-meta").inner_text()
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_project_sources_keep_selected_target(direct_server_with_data, engine):  # noqa: F811
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = _browser(pw, engine)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_selector("#nav-projects-add")
            page.evaluate("""async () => {
                window.projectModule = await import('/static/modules/project_create.js');
                window.projectPayloads = [];
                window.projectResult = 'pending';
                window.openTestProject = () => window.projectModule.openNewProjectDialog({
                    apiClient: {
                        fsDirs: async (path) => ({path: path || '/workspace', parent: path && path !== '/workspace' ? '/workspace' : '',
                            dirs: path && path !== '/workspace' ? [] : [{name:'Chosen folder', path:'/workspace/chosen'}, {name:'Other folder', path:'/workspace/other'}]}),
                        projectCreate: async payload => { window.projectPayloads.push(payload); return {project: {id:'created', name:payload.name}}; },
                    },
                }).then(value => { window.projectResult = value; });
            }""")
            for source in ["fileless", "genesis", "attach", "clone"]:
                opener = page.locator("#nav-projects-add")
                opener.focus()
                page.evaluate("window.openTestProject(); undefined")
                expect(page.locator("[data-np-name]")).to_be_focused()
                close = page.locator('.new-project-dialog [aria-label="Close"]')
                close.focus()
                page.keyboard.press("Shift+Tab")
                expect(page.locator("[data-np-create]")).to_be_focused()
                page.keyboard.press("Tab")
                expect(close).to_be_focused()
                page.locator("[data-np-name]").fill("Scoped project")
                page.locator(f'input[name="np-source"][value="{source}"]').check()
                expected = {"name": "Scoped project"}
                if source == "genesis":
                    expected["with_workspace"] = True
                elif source == "attach":
                    page.get_by_role("button", name="Chosen folder", exact=True).click()
                    page.get_by_role("button", name="Select this folder", exact=True).click()
                    page.get_by_role("button", name=".. (up)", exact=True).click()
                    page.get_by_role("button", name="Other folder", exact=True).click()
                    expect(page.locator("[data-np-path]")).to_contain_text("/workspace/other")
                    expect(page.locator("[data-np-selected]")).to_have_text("Selected folder: /workspace/chosen")
                    page.locator("[data-np-initgit]").check()
                    expected.update(path="/workspace/chosen", init_git=True)
                    _capture(page, f"project-{engine}-selected-target")
                elif source == "clone":
                    page.locator("[data-np-giturl]").fill("https://example.com/repository.git")
                    expected["git_url"] = "https://example.com/repository.git"
                page.locator("[data-np-create]").click()
                expect(page.locator(".new-project-backdrop")).to_have_count(0)
                expect(opener).to_be_focused()
                assert page.evaluate("window.projectPayloads.at(-1)") == expected
            assert len(page.evaluate("window.projectPayloads")) == 4
            page.evaluate("window.openTestProject(); undefined")
            page.keyboard.press("Escape")
            expect(page.locator(".new-project-backdrop")).to_have_count(0)
            assert page.evaluate("window.projectResult") is None
            expect(opener).to_be_focused()
            assert len(page.evaluate("window.projectPayloads")) == 4
            page.evaluate("""() => window.projectModule.openProjectRowMenu({id:'example', name:'Scoped project'}, {
                anchorEl: document.querySelector('#nav-projects-add'), apiClient: {},
            })""")
            # The menu opens on its first item, Pin since the sidebar pins (#2006); the keyboard goes on to Rename.
            expect(page.get_by_role("menuitem", name="Pin", exact=True)).to_be_focused()
            page.keyboard.press("ArrowDown")
            expect(page.get_by_role("menuitem", name="Rename…", exact=True)).to_be_focused()
            page.keyboard.press("Enter")
            expect(page.locator("[data-confirm-input]")).to_be_focused()
            page.keyboard.press("Escape")
            expect(opener).to_be_focused()
        finally:
            browser.close()


def _hit_target(page, selector):
    return page.locator(selector).evaluate("""el => {
        const rect = el.getBoundingClientRect();
        const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
        return {visible: rect.x >= 0 && rect.y >= 0 && rect.right <= innerWidth && rect.bottom <= innerHeight,
            hit: !!hit && (hit === el || el.contains(hit)), rect: {x:rect.x, y:rect.y, width:rect.width, height:rect.height}};
    }""")


def _resize(page, width, height):
    page.set_viewport_size({"width": width, "height": height})
    page.evaluate("""async () => {
        await new Promise(requestAnimationFrame);
        await Promise.all(document.getAnimations().filter(animation =>
            animation.effect?.getComputedTiming().iterations !== Infinity
        ).map(animation => animation.finished.catch(() => {})));
    }""")


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_files_project_actions_fit_available_viewport(direct_server_with_data, engine):  # noqa: F811
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = _browser(pw, engine)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            _files_gateway(page)
            page.goto(direct_server_with_data["url"] + "/#files", wait_until="domcontentloaded")
            page.locator(".files-entry").filter(has_text="sample.txt").click()
            page.locator(".files-editor").fill("Editable content with a long file path")
            failures = []
            for width, height in [(320, 640), (390, 844), (640, 360), (641, 640), (760, 640),
                                  (768, 640), (980, 700), (981, 700), (1280, 800)]:
                _resize(page, width, height)
                for selector in ["#files-save", "#files-download", "#files-open-external"]:
                    result = _hit_target(page, selector)
                    if not result["visible"] or not result["hit"]:
                        failures.append({"surface": "Files", "width": width, "height": height, "selector": selector, **result})
                editable = page.locator('.files-editor').evaluate("""el => {
                    let rect = el.getBoundingClientRect();
                    let top = Math.max(0, rect.top), bottom = Math.min(innerHeight, rect.bottom);
                    for (let parent = el.parentElement; parent; parent = parent.parentElement) {
                        if (getComputedStyle(parent).overflowY !== 'visible') {
                            const box = parent.getBoundingClientRect();
                            top = Math.max(top, box.top); bottom = Math.min(bottom, box.bottom);
                        }
                    }
                    return {visibleHeight: bottom - top, lineHeight: parseFloat(getComputedStyle(el).lineHeight)};
                }""")
                if editable['visibleHeight'] < editable['lineHeight']:
                    failures.append({"surface": "Files editor", "width": width, "height": height, **editable})
                if width in {320, 390}:
                    _capture(page, f"files-{engine}-{width}-actions")
            # Dialog content can scroll; the dialog frame and its actions stay in view.
            page.evaluate("""async () => {
                const module = await import('/static/modules/project_create.js');
                window.openGeometryProject = () => module.openNewProjectDialog({apiClient:{
                    fsDirs: async () => ({path:'/workspace/' + 'long-folder-name-'.repeat(8), dirs:[]}),
                }});
            }""")
            for width, height in [(320, 640), (390, 844), (640, 360)]:
                _resize(page, width, height)
                page.evaluate("window.openGeometryProject(); undefined")
                page.locator('[name="np-source"][value="attach"]').check()
                expect(page.locator("[data-np-path]")).to_contain_text("long-folder-name")
                for selector in [".new-project-dialog", "[data-np-create]", ".new-project-dialog .marketplace-modal-actions [data-np-cancel]"]:
                    result = _hit_target(page, selector)
                    if not result["visible"] or (selector != ".new-project-dialog" and not result["hit"]):
                        failures.append({"surface": "Project", "width": width, "height": height, "selector": selector, **result})
                _capture(page, f"project-{engine}-{width}x{height}-actions")
                page.keyboard.press("Escape")
            assert not failures, json.dumps(failures, indent=2)
        finally:
            browser.close()


_PANES = """() => {
    const shown = (el) => Boolean(el && el.getClientRects().length && getComputedStyle(el).visibility !== 'hidden');
    const list = document.querySelector('#files-list'), active = document.activeElement;
    return {view: document.querySelector('.files-layout').dataset.filesView,
        list: shown(document.querySelector('.files-sidebar')), file: shown(document.querySelector('.files-preview')),
        back: shown(document.querySelector('[data-files-back]')), listHeight: list.getBoundingClientRect().height,
        status: shown(document.querySelector('#files-preview-status')) ? document.querySelector('#files-preview-status').textContent : '',
        statusAboveList: Boolean(document.querySelector('#files-preview-status').compareDocumentPosition(list)
            & Node.DOCUMENT_POSITION_FOLLOWING) && list.parentElement.contains(document.querySelector('#files-preview-status')),
        unsaved: [...document.querySelectorAll('.files-entry')].filter((row) => shown(row.querySelector('.files-entry-unsaved')))
            .map((row) => row.querySelector('.files-entry-name, .files-entry-untitled').textContent),
        focus: active === document.body ? 'body' : (active.matches('[data-files-back]') ? 'back'
            : active.closest('.files-entry')?.querySelector('.files-entry-name, .files-entry-untitled')?.textContent
            || active.getAttribute('aria-label') || active.id || active.className),
        focusShown: shown(active), sideways: document.scrollingElement.scrollWidth > innerWidth + 1,
        layers: window.__ouroLayers.ids()};
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_narrow_files_page_shows_the_list_first_and_back_keeps_the_draft(direct_server_with_data, engine):  # noqa: F811
    """#7 on a phone: the list owns the page, a chosen file replaces it with Back to folder above it, Back keeps an
    unsaved draft that the file's row (or a new file's row) reopens, leaving it still asks, and the desk keeps both."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = _browser(pw, engine)
        try:
            page = browser.new_page(viewport={"width": 390, "height": 844}, has_touch=True)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            fixture = _files_gateway(page)
            page.goto(direct_server_with_data["url"] + "/#files", wait_until="domcontentloaded")
            row = page.locator(".files-entry").filter(has_text="sample.txt")
            row.wait_for()
            panes = page.evaluate(_PANES)
            assert (panes["view"], panes["list"], panes["file"]) == ("list", True, False), panes
            assert panes["listHeight"] > 844 * 0.6 and not panes["sideways"], panes
            _capture(page, f"files-{engine}-390-list")

            row.click()
            editor = page.get_by_role("textbox", name="File contents", exact=True)
            expect(editor).to_be_visible()
            panes = page.evaluate(_PANES)
            assert (panes["view"], panes["list"], panes["file"], panes["back"], panes["focus"], panes["layers"]) == (
                "file", False, True, True, "back", ["files-file"]), panes
            editor.fill("Draft typed on a phone")
            editor.evaluate("el => { window.filesEditor = el; }")
            expect(page.locator("#files-save")).to_be_enabled()

            # Back keeps the draft, says so on the file's row and returns focus there; nothing is asked or written.
            page.get_by_role("button", name="Back to folder", exact=True).click()
            panes = page.evaluate(_PANES)
            assert (panes["view"], panes["list"], panes["file"], panes["focus"], panes["unsaved"], panes["layers"]) == (
                "list", True, False, "sample.txt", ["sample.txt"], []), panes
            expect(page.locator(".confirm-dialog")).to_have_count(0)
            assert fixture["writes"] == []
            row.click()
            expect(editor).to_be_visible()
            _same_draft(page, "Draft typed on a phone")
            assert page.evaluate(_PANES)["focus"] == "back"
            page.get_by_role("button", name="Back to folder", exact=True).click()

            # Leaving the draft for another file still asks; Stay keeps the list, the draft and its mark.
            page.locator(".files-entry").filter(has_text="other.txt").click()
            page.get_by_role("button", name="Stay", exact=True).click()
            panes = page.evaluate(_PANES)
            assert (panes["view"], panes["unsaved"], panes["focusShown"]) == ("list", ["sample.txt"], True), panes
            _same_draft(page, "Draft typed on a phone")
            _capture(page, f"files-{engine}-390-back-unsaved")

            # A read that fails is said over the list it came from; the draft survives it.
            page.locator(".files-entry").filter(has_text="unreadable.txt").click()
            page.get_by_role("button", name="Discard", exact=True).click()
            expect(page.locator("#files-preview-status")).to_contain_text("File is unavailable")
            panes = page.evaluate(_PANES)
            assert panes["view"] == "list" and panes["statusAboveList"] and "File is unavailable" in panes["status"], panes
            _same_draft(page, "Draft typed on a phone")

            # A new file's draft has no folder row yet: Back gives it one, at the top, which reopens it.
            page.locator("#files-new-file").click()
            page.get_by_role("button", name="Discard", exact=True).click()
            name = page.get_by_role("textbox", name="File name", exact=True)
            expect(name).to_be_focused()
            name.fill("phone-notes.md")
            editor.fill("A new draft")
            editor.evaluate("el => { window.filesEditor = el; }")
            page.get_by_role("button", name="Back to folder", exact=True).click()
            panes = page.evaluate(_PANES)
            assert (panes["view"], panes["focus"], panes["unsaved"]) == ("list", "phone-notes.md", ["phone-notes.md"]), panes
            expect(page.locator(".files-entry").first).to_contain_text("phone-notes.md")
            page.locator(".files-entry.is-draft").click()
            assert name.input_value() == "phone-notes.md"
            _same_draft(page, "A new draft")

            # The desk shows both again, the draft rows and Back gone, and focus in sight stays put; the phone width
            # returns to the same file. Focus on Back when both panes come back lands on a shown control.
            editor.focus()
            _resize(page, 1280, 800)
            panes = page.evaluate(_PANES)
            assert (panes["list"], panes["file"], panes["back"], panes["unsaved"], panes["focusShown"], panes["layers"]) == (
                True, True, False, [], True, []), panes
            assert page.evaluate("document.activeElement === window.filesEditor")
            _resize(page, 390, 844)
            panes = page.evaluate(_PANES)
            assert (panes["view"], panes["list"], panes["file"], panes["back"]) == ("file", False, True, True), panes
            assert page.evaluate("document.activeElement === window.filesEditor")
            _same_draft(page, "A new draft")
            page.get_by_role("button", name="Back to folder", exact=True).focus()
            _resize(page, 1280, 800)
            panes = page.evaluate(_PANES)
            assert panes["focusShown"] and panes["focus"] != "body", panes
            _resize(page, 390, 844)
            panes = page.evaluate(_PANES)
            assert (panes["view"], panes["focus"], panes["focusShown"]) == ("file", "back", True), panes

            # Without a draft, an image opens straight away and Back returns to its row.
            page.get_by_role("button", name="Back to folder", exact=True).click()
            page.locator(".files-entry").filter(has_text="picture.png").click()
            page.get_by_role("button", name="Discard", exact=True).click()
            expect(page.locator(".files-preview-image")).to_be_visible()
            assert page.evaluate(_PANES)["focus"] == "back"
            _capture(page, f"files-{engine}-390-image")
            # Another page put in front drops the file's layer with the page's watch; Files shown again re-reads its
            # panes and the file alone is a layer once more.
            page.evaluate("document.querySelector('[data-nav-page=\"skills\"]').click()")
            page.wait_for_function("document.querySelector('#page-skills').classList.contains('active')")
            assert page.evaluate("window.__ouroLayers.ids()") == []
            page.evaluate("document.querySelector('[data-nav-page=\"files\"]').click()")
            page.wait_for_function("window.__ouroLayers.ids().includes('files-file')")
            panes = page.evaluate(_PANES)
            assert (panes["view"], panes["list"], panes["file"], panes["back"]) == ("file", False, True, True), panes
            page.get_by_role("button", name="Back to folder", exact=True).click()
            panes = page.evaluate(_PANES)
            assert (panes["view"], panes["focus"], panes["unsaved"], panes["sideways"]) == (
                "list", "picture.png", [], False), panes
            assert not errors, errors
        finally:
            browser.close()


_STACKED = """() => {
    const preview = document.querySelector('.files-preview'), drawer = document.getElementById('primary-sidebar');
    const rect = drawer.getBoundingClientRect(), hit = document.elementFromPoint(rect.left + 24, innerHeight / 2);
    return {layers: window.__ouroLayers.ids(), view: document.querySelector('.files-layout').dataset.filesView,
        fileInert: Boolean(preview.closest('[inert]')), drawerInert: Boolean(drawer.closest('[inert]')),
        drawerOnTop: rect.width > 0 && rect.right > 0 && Boolean(hit) && drawer.contains(hit)};
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_file_read_landing_behind_the_phone_drawer_stays_under_it(direct_server_with_data, engine):  # noqa: F811
    """#7 with a slow read: the chosen file arrives after the person opened the drawer. The file takes its layer under
    the drawer, which stays the top layer over an inert room: Escape closes the drawer and leaves the file shown, the
    next Escape returns the file to the list as Back to folder does."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = _browser(pw, engine)
        try:
            page = browser.new_page(viewport={"width": 390, "height": 844}, has_touch=True)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            fixture = _files_gateway(page)
            page.goto(direct_server_with_data["url"] + "/#files", wait_until="domcontentloaded")
            row = page.locator(".files-entry").filter(has_text="other.txt")
            row.wait_for()
            fixture["hold_navigation"] = True
            row.click()
            for _ in range(100):
                if fixture["pending_navigation"]:
                    break
                page.wait_for_timeout(50)
            assert len(fixture["pending_navigation"]) == 1, fixture["pending_navigation"]
            page.locator("[data-mobile-nav-toggle]:visible").first.click()
            page.wait_for_function("window.__ouroLayers.top() === 'drawer'")
            pending, payload = fixture["pending_navigation"].pop()
            fixture["hold_navigation"] = False
            pending.fulfill(json=payload)
            page.wait_for_function("window.__ouroLayers.ids().includes('files-file')")
            _resize(page, 390, 844)
            stacked = page.evaluate(_STACKED)
            _capture(page, f"files-{engine}-390-late-read-under-drawer")
            assert stacked == {"layers": ["files-file", "drawer"], "view": "file", "fileInert": True,
                               "drawerInert": False, "drawerOnTop": True}, stacked

            page.keyboard.press("Escape")
            page.wait_for_function("!window.__ouroLayers.ids().includes('drawer')")
            _resize(page, 390, 844)
            stacked = page.evaluate(_STACKED)
            assert (stacked["layers"], stacked["view"], stacked["fileInert"]) == (["files-file"], "file", False), stacked
            page.keyboard.press("Escape")
            page.wait_for_function("document.querySelector('.files-layout').dataset.filesView === 'list'")
            panes = page.evaluate(_PANES)
            assert (panes["layers"], panes["list"], panes["file"], panes["focus"]) == ([], True, False, "other.txt"), panes
            assert not errors, errors
        finally:
            browser.close()

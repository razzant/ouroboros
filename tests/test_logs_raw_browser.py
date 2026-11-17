"""Real browser consumers of the compact/Raw contract; HTTP is explicitly stubbed here.

The separate backend/producer integration is the parent's check. These tests run the exact served modules,
including the large received-event worker, and inspect byte cursors, download bytes and cancelled continuations.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

pytestmark = [pytest.mark.ui_browser]


@pytest.fixture
def raw_server(ui_candidate_checkout):
    """Loopback contract stub, with actual HTTP downloads (browser routing cannot fulfill every download)."""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ui_candidate_checkout.path / "web"), **kwargs)

        def log_message(self, *_args):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/static/"):
                self.path = self.path.removeprefix("/static")
                return super().do_GET()
            query = parse_qs(parsed.query)
            server.requests.append((parsed.path, query))
            payload = server.payload
            download = query.get("download") == ["1"]
            body = payload if download else json.dumps(_chunk(payload, int(query["offset"][0])) if "/raw" in parsed.path else {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if download:
                self.send_header("Content-Disposition", 'attachment; filename="exact.json"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A browser cancelled its own request.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.payload, server.requests = b"{}\n", []
    server.url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, name="raw-contract-stub")
    thread.start()
    try:
        yield server
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=10)
        assert not thread.is_alive()


def _source(payload):
    return {"v": 1, "kind": "log_row", "placement": "canonical", "task_id": "", "log": "events",
            "start": 0, "length": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _entry(payload, **fields):
    return {"type": "raw_contract_sentinel", "ts": "2026-11-04T01:00:00Z", "note": "Compact preview",
            "_raw": {"ref": "opaque +/= & source", "source": _source(payload), "omitted_fields": ["payload"],
                     "omitted_field_count": 1, "public_bytes": len(payload), "representation": "stored"}, **fields}


def _chunk(payload, offset):
    end = min(len(payload), offset + 65536)
    while end < len(payload) and payload[end] & 0xC0 == 0x80:
        end -= 1
    return {"source": _source(payload), "representation": "stored", "sha256": hashlib.sha256(payload).hexdigest(),
            "total_bytes": len(payload), "offset": offset, "next_offset": end, "has_more": end < len(payload),
            "text": payload[offset:end].decode()}


def _mount(browser, url, rows=()):
    page = browser.new_page(viewport={"width": 390, "height": 780}, accept_downloads=True)
    page.goto(url + "/api/health")
    page.set_content('''<link rel="stylesheet" href="/static/ui.css"><link rel="stylesheet" href="/static/style.css">
        <style>body{display:block;overflow:auto}#raw-mount{height:760px;padding:12px}.settings-logs-panel{height:100%}</style>
        <div id="raw-mount"></div>''')
    calls = []
    def history(route):
        calls.append(route.request.url)
        name = urlparse(route.request.url).path.rsplit("/", 1)[-1]
        assert parse_qs(urlparse(route.request.url).query)["view"] == ["compact"]
        route.fulfill(json={"name": name, "entries": list(rows) if name == "events" else [], "view": "compact",
                            "gaps": [], "skipped_rows": 0})
    page.route("**/api/logs/*?view=compact*", history)
    page.evaluate('''async () => {
        const {initLogs} = await import('/static/modules/logs.js');
        window.state = {activePage:'dashboard', dashboardActiveSubtab:'logs'};
        const handlers = new Map();
        window.emit = data => handlers.get('log')({data});
        window.reopen = () => handlers.get('open')();  // a reconnect: the recent history is read again
        window.expose = shown => { state.activePage = shown ? 'dashboard' : 'chat';
            dispatchEvent(new CustomEvent('ouro:page-shown', {detail:{page:state.activePage}})); };
        initLogs({mount:document.getElementById('raw-mount'), state, ws:{on:(type,fn)=>handlers.set(type,fn)}});
    }''')
    page.wait_for_function("() => !document.querySelector('.logs-history-status').textContent.includes('Loading')")
    return page, calls


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_raw_byte_pages_stay_bounded_and_full_download_keeps_exact_bytes(raw_server, engine):
    from playwright.sync_api import expect, sync_playwright

    payload = (json.dumps({"payload": "🙂Ж" * 600000, "sentinel": "EXACT-END"}, ensure_ascii=False) + "\n").encode()
    row = _entry(payload)
    raw_server.payload = payload
    offsets = []
    evidence = Path(os.environ["OUROBOROS_UI_EVIDENCE_DIR"])
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, calls = _mount(browser, raw_server.url, [row])
            def requested(request):
                if "/raw?" not in request.url:
                    return
                query = parse_qs(urlparse(request.url).query)
                assert query["ref"] == [row["_raw"]["ref"]]
                if "download" in query:
                    assert "offset" not in query and "limit" not in query
                else:
                    assert query["limit"] == ["65536"]
                    offset = int(query["offset"][0]); offsets.append(offset)
            page.on("request", requested)
            page.get_by_role("button", name="Raw", exact=True).click()
            status = page.locator(".log-raw-status")
            expect(status).to_contain_text("Source SHA-256")
            page.screenshot(path=str(evidence / f"raw-page-{engine}.png"))
            for _ in range(48):
                page.get_by_role("button", name="Next page", exact=True).click()
                expect(status).to_contain_text("Source SHA-256")
                assert page.locator(".log-raw").evaluate("p => new TextEncoder().encode(p.textContent).length") <= 65536
            assert offsets == [0] + [_chunk(payload, offset)["next_offset"] for offset in offsets[:-1]]
            assert offsets[1] != len(_chunk(payload, 0)["text"]), "UTF-8 bytes differ from both Python and JS character counts"
            with page.expect_download() as pending:
                page.get_by_role("link", name="Download full").click()
            download = pending.value
            path = evidence / f"raw-download-{engine}.json"
            download.save_as(path)
            assert path.read_bytes() == payload
            assert len(calls) == 4
            (evidence / f"raw-pages-{engine}.json").write_text(json.dumps({"contract_stub": True, "offsets": offsets,
                "total_bytes": len(payload), "download_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_raw_refusals_identity_and_cancelled_continuations_never_substitute_rows(raw_server, engine):
    from playwright.sync_api import expect, sync_playwright

    payload = b'{"payload":"' + b'a' * 150000 + b'"}\n'
    row = _entry(payload)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _calls = _mount(browser, raw_server.url, [row])
            page.evaluate('''() => {
                const original = fetch;
                window.gates = [];
                window.fetch = (url, init) => String(url).includes('/raw?')
                    ? new Promise(resolve => gates.push({url, signal:init.signal, resolve})) : original(url, init);
                window.answer = data => gates.shift().resolve(new Response(JSON.stringify(data.body), {status:data.status}));
            }''')
            raw = page.locator(".log-raw")
            status = page.locator(".log-raw-status")
            for code, reason in [(404, "log_source_unavailable"), (409, "log_source_changed"), (416, "raw_range_invalid"), (503, "log_source_unavailable")]:
                page.get_by_role("button", name="Raw", exact=True).click()
                page.wait_for_function("() => gates.length === 1")
                page.evaluate("answer", {"status": code, "body": {"error": "Exact source refused", "reason_code": reason}})
                expect(status).to_contain_text(f"HTTP {code}, {reason}")
                assert raw.inner_text() == ""
                page.get_by_role("button", name="Hide raw").click()
            page.get_by_role("button", name="Raw", exact=True).click()
            page.wait_for_function("() => gates.length === 1")
            page.evaluate("answer", {"status": 200, "body": _chunk(payload, 0)})
            expect(status).to_contain_text("Source SHA-256")
            first = raw.inner_text()
            page.get_by_role("button", name="Next page").click()
            page.wait_for_function("() => gates.length === 1")
            page.evaluate("() => expose(false)")
            assert page.evaluate("() => gates[0].signal.aborted")
            page.evaluate("answer", {"status": 200, "body": _chunk(payload, 65536)})
            assert raw.inner_text() == first, "a hidden response cannot paint"
            page.evaluate("() => expose(true)")
            page.wait_for_function("() => gates.length === 1")
            changed = _chunk(payload, 65536); changed["sha256"] = "0" * 64
            page.evaluate("answer", {"status": 200, "body": changed})
            expect(status).to_contain_text("Raw source changed")
            assert raw.inner_text() == first
            page.get_by_role("button", name="Retry", exact=True).click()
            page.wait_for_function("() => gates.length === 1")
            page.get_by_role("button", name="Hide raw").click()
            assert page.evaluate("() => gates[0].signal.aborted")
            page.evaluate("answer", {"status": 200, "body": _chunk(payload, 65536)})
            assert raw.inner_text() == ""
            page.get_by_role("button", name="Raw", exact=True).click()
            page.wait_for_function("() => gates.length === 1")
            page.get_by_role("button", name="System", exact=True).click()
            assert page.evaluate("() => gates[0].signal.aborted")
            page.evaluate("answer", {"status": 200, "body": _chunk(payload, 0)})
            page.get_by_role("button", name="System", exact=True).click()
            page.wait_for_function("() => gates.length === 1")
            page.get_by_role("button", name="Clear", exact=True).click()
            assert page.evaluate("() => gates[0].signal.aborted")
            page.evaluate("answer", {"status": 200, "body": _chunk(payload, 0)})
            assert page.locator(".log-entry").count() == 0
            page.evaluate("emit", {**row, "type": "task_started", "task_id": "collapse"})
            page.locator("summary").click()
            page.get_by_role("button", name="Raw", exact=True).click()
            page.wait_for_function("() => gates.length === 1")
            page.locator("summary").click()
            page.wait_for_function("() => gates[0].signal.aborted")
            page.evaluate("answer", {"status": 200, "body": _chunk(payload, 0)})
            assert page.locator(".log-task-event").count() == 0
            page.evaluate("emit", {**row, "type": "trim_raw"})
            page.get_by_role("button", name="Raw", exact=True).click()
            page.wait_for_function("() => gates.length === 1")
            page.evaluate("() => {for (let i=0;i<501;i++) emit({type:'trim_'+i, ts:'2026-11-04T03:00:00Z'});}")
            assert page.evaluate("() => gates[0].signal.aborted")
            page.evaluate("answer", {"status": 200, "body": _chunk(payload, 0)})
            assert page.locator(".log-raw-tools:not([hidden])").count() == 0
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_large_received_raw_uses_worker_and_repeat_keeps_selected_text(raw_server, engine):
    from playwright.sync_api import expect, sync_playwright

    evidence = Path(os.environ["OUROBOROS_UI_EVIDENCE_DIR"])
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _calls = _mount(browser, raw_server.url)
            page.evaluate('''() => {
                const Native = Worker; window.workerCount = 0;
                window.Worker = class extends Native { constructor(...args) {super(...args); workerCount++;}
                    terminate() { if (!this.ended) {this.ended=true; workerCount--;} super.terminate();} };
                emit({type:'worker_sha_verify', is_progress:true, task_id:'large', ts:'2026-11-04T01:00:00Z', payload:'x'.repeat(17500000), sentinel:'RECEIVED-END'});
            }''')
            page.locator("summary").click()
            elapsed = page.locator(".log-raw-toggle").evaluate("b => {const at=performance.now(); b.click(); return performance.now()-at;}")
            expect(page.locator(".log-raw-status")).to_contain_text("Received event")
            assert page.evaluate("workerCount") == 0
            assert len(page.locator(".log-raw").inner_text()) <= 65536
            with page.expect_download() as pending:
                page.get_by_role("link", name="Download full").click()
            path = evidence / f"received-download-{engine}.json"
            pending.value.save_as(path)
            data = json.loads(path.read_bytes())
            assert len(data["payload"]) == 17500000 and data["sentinel"] == "RECEIVED-END"
            selected = page.locator(".log-task-event .log-headline").evaluate('''node => {
                window.keptHeadline = node.firstChild;
                const range=document.createRange(); range.selectNodeContents(node);
                getSelection().removeAllRanges(); getSelection().addRange(range); return getSelection().toString();
            }''')
            page.evaluate("() => emit({type:'worker_sha_verify', is_progress:true, task_id:'large', ts:'2026-11-04T01:00:01Z'})")
            assert page.evaluate("() => getSelection().toString()") == selected
            assert page.locator(".log-task-event .log-headline").evaluate("n => n.firstChild === keptHeadline")
            expect(page.locator(".log-raw-status")).to_contain_text("entry changed")
            page.screenshot(path=str(evidence / f"raw-repeat-{engine}.png"))
            (evidence / f"received-raw-{engine}.json").write_text(json.dumps({"contract_stub": True,
                "open_handler_ms": elapsed, "download_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_raw_desktop_bytes_bridge_verifies_download_and_cancels_late_save(raw_server, engine):
    """Existing bridge stub only: no native app, filesystem save or GUI is started."""
    from playwright.sync_api import expect, sync_playwright
    import base64

    payload = (json.dumps({"payload": "🙂Ж" * 30000}, ensure_ascii=False) + "\n").encode()
    raw_server.payload = payload
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _ = _mount(browser, raw_server.url, [_entry(payload)])
            page.evaluate("""async () => {
                window.saved=[];
                window.pywebview={api:{save_bytes_to_downloads:async (name,b64)=>{saved.push({name,b64});return {ok:true};},
                    download_file_to_downloads:()=>{throw Error('Raw URL must not enter the old launcher allowlist');}}};
                (await import('/static/modules/ui_helpers.js')).installDesktopShellLinkInterceptor();
            }""")
            page.get_by_role("button", name="Raw", exact=True).click()
            expect(page.locator(".log-raw-status")).to_contain_text("Source SHA-256")
            page.get_by_role("link", name="Download full").click()
            expect(page.locator(".log-raw-status")).to_have_text("Full Raw saved.")
            assert base64.b64decode(page.evaluate("saved[0].b64")) == payload
            raw_server.payload = b'{"newer":"must not substitute"}'
            page.get_by_role("link", name="Download full").click()
            expect(page.locator(".log-raw-status")).to_contain_text("Raw source changed")
            assert page.evaluate("saved.length") == 1
            raw_server.payload = payload
            page.evaluate("""() => {
                const prior=fetch;
                window.fetch=(url,init)=>String(url).includes('download=1')
                    ? new Promise(resolve=>window.heldDownload={resolve,signal:init.signal}) : prior(url,init);
            }""")
            page.get_by_role("link", name="Download full").click()
            page.wait_for_function("() => window.heldDownload")
            page.evaluate("() => expose(false)")
            assert page.evaluate("heldDownload.signal.aborted")
            page.evaluate("text => heldDownload.resolve(new Response(text))", payload.decode())
            page.wait_for_timeout(150)
            assert page.evaluate("saved.length") == 1
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_backfill_keeps_an_open_timeline_in_time_order_and_discloses_skipped_records_and_left_out_fields(raw_server, engine):
    """R1: rows older than an open Timeline's take their places (shown nodes stay); a compact read says what it skipped
    (that read only) and a row what its view left out."""
    from playwright.sync_api import expect, sync_playwright

    evidence = Path(os.environ["OUROBOROS_UI_EVIDENCE_DIR"])
    evidence.mkdir(parents=True, exist_ok=True)
    row = lambda second, tool: {"type": "tool_call_finished", "tool": tool, "is_error": False, "task_id": "ordered",
                                "ts": f"2026-11-04T01:00:{second:02d}Z"}
    payload = json.dumps({"type": "llm_api_error", "error": "provider said no " * 400}).encode()
    shortened = _entry(payload, type="llm_api_error", ts="2026-11-04T01:00:40Z")
    shortened["_raw"].update(omitted_fields=["error"], omitted_field_count=1)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _calls = _mount(browser, raw_server.url)
            page.evaluate("row => emit(row)", row(30, "third_tool"))
            card = page.locator('[data-task-group="ordered"]')
            card.locator("summary").click()
            card.locator(".log-task-event").first.evaluate("node => { window.thirdRow = node; }")

            def history(route):
                name = urlparse(route.request.url).path.rsplit("/", 1)[-1]
                entries = [row(10, "first_tool"), row(20, "second_tool"), shortened] if name == "events" else []
                damaged = name == "events"
                route.fulfill(json={"name": name, "entries": entries, "view": "compact",
                                    "gaps": ["malformed_jsonl"] if damaged else [], "skipped_rows": 1 if damaged else 0})
            page.unroute("**/api/logs/*?view=compact*")
            page.route("**/api/logs/*?view=compact*", history)
            page.evaluate("() => reopen()")
            status = page.locator(".logs-history-status")
            expect(status).to_have_text("Damaged records skipped in the recent history read: events 1 (malformed line).")
            headlines = card.locator(".log-task-event .log-headline")
            expect(headlines).to_have_text(["first_tool finished", "second_tool finished", "third_tool finished"])
            assert card.locator(".log-task-event").nth(2).evaluate("node => node === window.thirdRow"), "the shown row keeps its node"
            note = page.locator(".log-entry:not(.log-task-card) [data-log-omitted]")
            expect(note).to_have_text("Left out of this view: error. Raw reads the whole row.")
            page.screenshot(path=str(evidence / f"logs-order-gaps-omitted-{engine}.png"), full_page=True)
        finally:
            browser.close()

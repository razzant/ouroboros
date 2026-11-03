"""Real Available-subagents binder/markup over synthetic engine facts, no daemon."""
from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.ui_browser
WEB = Path(__file__).resolve().parents[1] / "web"

HTML = """<!doctype html><html lang="en" class="ouro-ui" data-theme="light"><head>
<meta charset="utf-8"><link rel="stylesheet" href="/static/ui.css">
<link rel="stylesheet" href="/static/style.css"><link rel="stylesheet" href="/static/settings.css">
<link rel="stylesheet" href="/static/model_roles.css"><style>body{display:block;padding:28px}main{max-width:900px;margin:auto}</style></head>
<body><main><h2>Available subagents · exact quota scope</h2>
<p id="scenario">Generic engine read: exact model availability is unknown.</p>
<button id="scoped">Use exact engine scope</button><button id="generic">Use generic engine scope</button>
<div id="available-subagents-editor"></div></main><script type="module">
import { createAvailableSubagentsEditor } from '/static/modules/subagents_settings.js';
const store = {error:'',facet:()=> 'ok',subscribe:()=>()=>{},refresh:async()=>{},snapshot:{
  harnesses:[{id:'claude',enabled:true,status:'ok',models:['synthetic-model']}],
  profiles:{profiles:[{profile:{harness_id:'claude',profile_id:'chosen',enabled:true},status:{verification:'passed'}}],harnessAccounts:[]},
  quota:[{subject:{harness:'claude',subject_id:'chosen'},freshness:'fresh',
    availability:{state:'available'},constraints:[{used_ratio:0.5}]}],
  availability_scope:{model:null,requestKind:'text',basis:'quota_block'}
}};
const editor = createAvailableSubagentsEditor({doc:document,win:null,store});
editor.load({enabled:true,items:[{subagent_id:'fixture',access:'readonly',recommended_use:'Synthetic account and model facts remain available.',
  route:{kind:'agent_session',target_id:'claude=synthetic-model',credential_profile_id:'chosen'}}]});
await editor.reloadStatus();
for (const [id,model] of [['scoped','synthetic-model'],['generic',null]]) {
  document.getElementById(id).onclick = async()=>{
    store.snapshot = structuredClone(store.snapshot);
    store.snapshot.availability_scope.model = model;
    await editor.reloadStatus();
    document.getElementById('scenario').textContent = model ? 'Exact engine read: quota available; execution is not guaranteed.' : 'Generic engine read: exact model availability is unknown.';
  };
}
window.fixtureReady = true;
</script></body></html>"""


@pytest.mark.parametrize("base_only", [False, True], ids=["exact-catalog", "1m-base-catalog"])
def test_actual_available_subagents_row_needs_honored_scope(tmp_path, base_only):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("UI browser explicitly selected")
    from playwright.sync_api import sync_playwright
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    selected = "synthetic-model[1M]" if base_only else "synthetic-model"
    html = HTML.replace("target_id:'claude=synthetic-model'", f"target_id:'claude={selected}'")
    html = html.replace("['scoped','synthetic-model']", f"['scoped','{selected}']")
    suffix = "-1m" if base_only else ""
    requests = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            errors = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            def serve(route):
                path = urlparse(route.request.url).path
                requests.append(path)
                if path == "/":
                    route.fulfill(content_type="text/html", body=html)
                elif path.startswith("/static/"):
                    source = (WEB / path.removeprefix("/static/")).resolve()
                    if source.is_relative_to(WEB) and source.is_file():
                        route.fulfill(content_type=mimetypes.guess_type(source)[0] or "text/plain", body=source.read_bytes())
                    else:
                        route.fulfill(status=404, body="No fixture asset")
                else:
                    pytest.fail(f"Unexpected browser request: {path}")
            page.route("**/*", serve)
            page.goto("https://quota-fixture.invalid/")
            page.wait_for_function("window.fixtureReady === true")
            status = page.locator("[data-subagent-status]")
            assert status.inner_text() == "Not checked"
            assert "pinned account ready" in status.get_attribute("title")
            assert page.locator('[data-subagent-field="model"]').input_value() == selected
            assert page.locator('[data-subagent-field="account"]').input_value() == "chosen"
            assert page.locator('[data-subagent-field="access"]').input_value() == "readonly"
            page.screenshot(path=str(evidence / f"available-row-generic{suffix}.png"), full_page=True)
            qualifier = page.locator('[data-subagent-meta][data-availability-qualifier]')
            if base_only:
                assert qualifier.is_visible()
                assert "Base model listed" in qualifier.inner_text()
            page.locator("#scoped").click()
            page.wait_for_function("document.querySelector('[data-subagent-status]').textContent === 'Available'")
            if base_only:
                assert qualifier.is_visible()
            page.screenshot(path=str(evidence / f"available-row-scoped{suffix}.png"), full_page=True)
            page.locator("#generic").click()
            page.wait_for_function("document.querySelector('[data-subagent-status]').textContent === 'Not checked'")
            if base_only:
                assert qualifier.is_visible()
                assert page.locator('[data-subagent-field="model"]').input_value() == selected
            assert errors == []
            assert not any(path.startswith("/api/") for path in requests)
            (evidence / f"browser-row-observation{suffix}.json").write_text(json.dumps({
                "consumer": "createAvailableSubagentsEditor + real DOM + production CSS", "engine": "Chromium",
                "observed_statuses": ["Not checked", "Available", "Not checked"], "requests": requests,
                "selected_model": selected, "base_only_qualifier_visible": base_only,
                "page_errors": errors, "no_daemon_or_provider": True}, indent=2))
        finally:
            browser.close()

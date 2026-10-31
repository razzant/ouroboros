"""Real candidate SPA, real gateway projection and durable Main notice replay.

Only unrelated Settings APIs use the established subscription UI fixtures.
Capability/profile JSON is exact recorded engine wire; notices pass through the
production supervisor, message bus and history endpoint before the browser sees them.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from tests.candidate_checkout import candidate_checkout
from tests.test_claudexor_capability_replay import replay as replay
from tests.test_claudexor_capability_engine import call, notices, TREE

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def navigate(page, name):
    if page.viewport_size["width"] <= 640:
        page.locator(".page.active [data-mobile-nav-toggle]").click()
    page.locator(f'#primary-sidebar [data-nav-page="{name}"]').click()
    page.locator(f"#page-{name}.active").wait_for()


@pytest.fixture
def capability_ui(tmp_path, monkeypatch, replay):
    from ouroboros import claudexor_daemon as owned, llm_claudexor as transport
    from ouroboros.gateways import claudexor as wire
    from ouroboros.gateway.claudexor_accounts import _read_status_payload
    from ouroboros.gateway.history import make_chat_history_endpoint
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from tests import test_subscription_setup_browser as pages
    f = replay("both")
    with pytest.raises(transport.ClaudexorModelError):
        call(f)
    history = notices(f, monkeypatch)
    assert len(history) == 1
    monkeypatch.setattr(owned, "owned_config_dir", lambda: f.root)
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(status_dict=lambda: {"state": "running"}))
    monkeypatch.setattr(wire, "discover_daemon_at", lambda *_: wire.DaemonEndpoint("127.0.0.1", 1, "offline"))
    monkeypatch.setattr(owned.OwnedClaudexorDaemon, "ensure_running", lambda *_: pytest.fail("browser woke engine"))
    status = _read_status_payload(False)
    assert status["reads"]["accounts"] == "ok"
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", tmp_path / "evidence"))
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "browser-consumed.json").write_text(json.dumps({
        "engine_tree": TREE, "status": status, "history": history,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_RUN_UI_SMOKE", "1")
    if os.environ.get("CAPABILITY_BROWSER_EXECUTABLE"):
        monkeypatch.setenv("OUROBOROS_UI_BROWSER_EXECUTABLE", os.environ["CAPABILITY_BROWSER_EXECUTABLE"])
    source = Path(__file__).resolve().parents[1]
    app = Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(f.root))])
    with TestClient(app) as history_client, \
            candidate_checkout(source, tmp_path / "candidate", origin_proof=True) as candidate:
        monkeypatch.setattr(pages, "WEB", candidate.path / "web")
        candidate.hold()
        generator = pages.subscription_ui.__wrapped__()
        try:
            ui = next(generator)
            # Catalog, quota and account mutations aren't the subject of this
            # page check; their existing fixtures prevent unrelated failed reads.
            ui["fixture"]["status"].update(profiles=status["profiles"])
            def history_response(route):
                from urllib.parse import urlsplit
                url = urlsplit(route.request.url)
                response = history_client.get(url.path + ("?" + url.query if url.query else ""))
                route.fulfill(status=response.status_code, content_type="application/json", body=response.text)
            ui["page"].route("**/api/chat/history*", history_response)
            with urlopen(ui["url"] + "/static/candidate-sentinel.txt") as response:
                assert response.read() == candidate.sentinel_bytes
            hashes = {}
            for name in ("harness_accounts", "account_resources", "claudexor_status_store"):
                path = f"web/modules/{name}.js"
                with urlopen(ui["url"] + f"/static/modules/{name}.js") as response:
                    body = response.read()
                assert body == candidate.state.files[path][0]
                hashes[path] = hashlib.sha256(body).hexdigest()
            (evidence / "browser-source.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
            yield ui, evidence
        finally:
            generator.close()
            candidate.release()


@pytest.mark.parametrize("width,height,theme", [(1360, 900, "light"), (390, 844, "dark")])
def test_capability_line_and_main_notice(capability_ui, width, height, theme):
    from playwright.sync_api import expect

    ui, evidence = capability_ui
    page = ui["page"]
    mark = ui["fixture"]["status"]["profiles"]["routing_evidence"]["capabilities"][0]
    # Preserve the recorded engine wire; exercise both sides of its expiry with a fixed browser clock.
    page.clock.set_fixed_time(mark["observed_at"])
    page.set_viewport_size({"width": width, "height": height})
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
    page.goto(ui["url"] + "/#settings")
    page.click('[data-settings-tab="providers"]')
    lines = page.locator("[data-model-capability]")
    lines.first.wait_for()
    assert lines.count() == 2
    text = lines.first.inner_text()
    assert "gpt-6.1-sol" in text and "not supported" in text
    assert "temporary mark set" in text and "may be retried" in text
    lines.first.scroll_into_view_if_needed()
    geometry = lines.evaluate_all("""nodes => nodes.map(node => {
        const row = node.closest('.harness-account-row'), actions = row.querySelector('.harness-account-actions');
        const r = row.getBoundingClientRect(), n = node.getBoundingClientRect(), a = actions.getBoundingClientRect();
        const style = getComputedStyle(node);
        return {row: r.toJSON(), line: n.toJSON(), actions: a.toJSON(), font: style.fontSize,
                color: style.color, background: getComputedStyle(row).backgroundColor,
                scrollWidth: row.scrollWidth, clientWidth: row.clientWidth};
    })""")
    for measure in geometry:
        r, n, a = measure["row"], measure["line"], measure["actions"]
        assert n["left"] >= r["left"] and n["right"] <= r["right"] + 1
        assert n["bottom"] <= r["bottom"] and n["top"] >= r["top"]
        assert n["right"] <= a["left"] or n["bottom"] <= a["top"]
        assert measure["scrollWidth"] <= measure["clientWidth"] + 1
        assert measure["font"] == "12px"
    stem = f"{width}-{theme}"
    page.screenshot(path=str(evidence / f"accounts-{stem}.png"), animations="disabled")
    (evidence / f"geometry-{stem}.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    page.clock.set_fixed_time(mark["expires_at"])
    page.reload()
    page.click('[data-settings-tab="providers"]')
    expect(lines).to_have_count(2)
    expect(lines.first).to_contain_text("temporary mark set")
    expect(lines.first).to_contain_text("the next request may recheck it")
    expect(lines.first).not_to_contain_text("may be retried")
    navigate(page, "chat")
    notice = page.locator('#chat-messages .chat-bubble[data-system-type="model_capability_notice"]')
    notice.wait_for()
    assert notice.count() == 1
    assert all(word in notice.inner_text() for word in ("No account can serve gpt-6.1-sol", "a:", "b:",
                                                      "not supported", "Check the requested model",
                                                      "First seen in Helper call."))
    notice.scroll_into_view_if_needed()
    page.screenshot(path=str(evidence / f"chat-{stem}.png"), animations="disabled")
    # A stale cached roster can retain evidence bytes, but cannot show them as
    # a current mark after its account read fails.
    ui["fixture"]["status"]["reads"]["accounts"] = "failed"
    navigate(page, "settings")
    page.reload()
    page.click('[data-settings-tab="providers"]')
    page.locator('.harness-account-row[data-profile="a"]').wait_for()
    assert page.locator("[data-model-capability]").count() == 0
    ui["fixture"]["status"]["reads"]["accounts"] = "ok"
    # A failed marks read is ONE dated note for the whole section, never "no marks"
    # and never a sentence repeated on every account row.
    profiles = ui["fixture"]["status"]["profiles"]
    marks = profiles.pop("routing_evidence")
    profiles["routing_evidence_read"] = {"state": "failed", "observed_at": marks["capabilities"][0]["observed_at"]}
    page.reload()
    page.click('[data-settings-tab="providers"]')
    page.locator('.harness-account-row[data-profile="a"]').wait_for()
    assert page.locator("[data-model-capability]").count() == 0
    note = page.locator("#harness-model-marks-note")
    assert note.is_visible() and note.inner_text().startswith("Temporary model marks could not be read")
    page.screenshot(path=str(evidence / f"accounts-unread-{stem}.png"), animations="disabled")
    # An older engine advertises no capability rows: the host does not ask and shows nothing.
    profiles.pop("routing_evidence_read")
    page.reload()
    page.click('[data-settings-tab="providers"]')
    page.locator('.harness-account-row[data-profile="a"]').wait_for()
    assert page.locator("[data-model-capability]").count() == 0
    assert not page.locator("#harness-model-marks-note").is_visible()
    assert not page.locator("#harness-accounts-error").is_visible()
    assert not ui["posts"], "passive browser checks never write Settings or accounts"

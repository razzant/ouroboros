"""Effort facts in the real Logs SPA, live and after history reload, never in Chat.

The provider and engine results are synthetic fixtures (the shared effort
resolution fixtures), not vendor captures. Production code builds and persists
the facts: ``effort_request_facts`` for API sends, ``llm_claudexor._usage`` for
managed model operations, ``final_attempt_facts`` plus the custody writer for a
delegated session, the supervisor ``llm_usage`` handler, and ``/api/logs``.
Only the WebSocket transport and bootstrap/state envelope are fixtures; there
is no supervisor process.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import delegate_custody
from ouroboros.gateway.logs import api_logs_tail
from ouroboros.gateways.claudexor import final_attempt_facts
from ouroboros.llm_attempt import effort_request_facts
from ouroboros.llm_claudexor import _usage as managed_usage
from supervisor.events_budget import _handle_llm_usage
from tests.test_delegated_activity_browser import engine_ui as engine_ui
from tests.test_subscription_setup_browser import capture
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui
from tests._usage_store_testing import root as root

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PAIRED = json.loads((FIXTURES / "claudexor_effort_resolution.json").read_text(encoding="utf-8"))
VARIANTS = json.loads((FIXTURES / "claudexor_effort_resolution_variants.json").read_text(encoding="utf-8"))

# Exact Logs bodies: every requested/sent/reported fact is explicit or "unknown".
EXPECTED = {
    "effort-api-explicit": 'Effort requested ultra · host sent {"extra_body.reasoning":'
                           '{"effort":"max","exclude":false}} · reported unknown',
    "effort-api-omitted": "Effort requested medium · host sent omitted · reported unknown",
    "effort-managed-prepared": 'Effort requested ultra · host sent {"options.reasoningEffort":"ultra"}'
                               " · prepared reasoning.effort=xhigh (downward; account_catalog) · reported unknown",
    "effort-managed-reported": 'Effort requested ultra · host sent {"options.reasoningEffort":"ultra"}'
                               " · prepared reasoning.effort=xhigh (downward; account_catalog)"
                               " · reported xhigh (codex.responses.reasoning.effort)",
    "effort-managed-omitted": 'Effort requested ultra · host sent {"options.reasoningEffort":"ultra"}'
                              " · prepared omitted (omitted; adapter) · reported unknown",
    "effort-session": "Engine requested effort ultra · host sent unknown"
                      " · prepared reasoning.effort=low (floor; account_catalog) · reported unknown",
}
DISCLOSURE_MARKERS = ("Effort requested", "Engine requested effort", "host sent", "prepared ")


def _usage_events(root: Path) -> list[dict]:
    api = {"provider": "openrouter", "resolved_model": "vendor/future", "requested_reasoning_effort": "ultra"}
    managed = {"provider": "claudexor", "requested_reasoning_effort": "ultra"}
    explicit = {"model": "vendor/future", "extra_body": {"reasoning": {"effort": "max", "exclude": False}}}
    managed_facts = effort_request_facts(managed, {"options": {"reasoningEffort": "ultra"}})
    # The engine's one model reporter mirrors the provider's appliedOptions echo
    # into observed/observedSource, so a reported result carries both, equal.
    reported = {"appliedOptions": {"reasoningEffort": "xhigh"}, "effortResolution": {
        **PAIRED, "observed": "xhigh", "observedSource": "codex.responses.reasoning.effort"}}
    return [
        ("effort-api-explicit", "openrouter", {"effort": effort_request_facts(api, explicit)}),
        ("effort-api-omitted", "openai",
         {"effort": effort_request_facts({**api, "requested_reasoning_effort": "medium"}, {"model": "future"})}),
        ("effort-managed-prepared", "claudexor", managed_usage({"effortResolution": PAIRED}, managed_facts)[0]),
        ("effort-managed-reported", "claudexor", managed_usage(reported, managed_facts)[0]),
        ("effort-managed-omitted", "claudexor",
         managed_usage({"effortResolution": VARIANTS["known_parameter_omitted"]}, managed_facts)[0]),
    ]


def _settle_session(root: Path) -> dict:
    run_dir = root / "runs" / "run-effort" / "final"
    run_dir.mkdir(parents=True)
    (run_dir / "telemetry.yaml").write_text(json.dumps({
        "run_id": "run-effort", "final_attempt_id": "attempt-2",
        "attempts": [{"attempt_id": "attempt-1", "observed_model": "vendor/old"},
                     {"attempt_id": "attempt-2", "harness_id": "codex", "observed_model": "vendor/future",
                      "profile_id": "work", "effort_resolution": VARIANTS["engine_order_floor"]}],
    }), encoding="utf-8")
    observed = final_attempt_facts({"summary": {"runDir": str(run_dir.parent)}}, "run-effort")
    assert observed["effort_resolution"] == VARIANTS["engine_order_floor"]
    assert delegate_custody.emit(root, delegate_custody.SETTLED, {
        "run_id": "run-effort", "task_id": "effort-session", "route": "codex",
        "source": "subagent", "category": "task", "model": observed["model"],
        "observed_attempt": observed, "state": "succeeded",
    })
    return observed


def _disclosed(page, selector: str) -> list[str]:
    text = page.locator(selector).inner_text() if page.locator(selector).count() else ""
    return [marker for marker in DISCLOSURE_MARKERS if marker in text]


def _navigate(page, target: str):
    if page.viewport_size["width"] < 700:  # The rail is an off-canvas drawer on phones.
        page.locator('[data-mobile-nav-toggle]:visible').first.click()
    page.click(f'[data-nav-page="{target}"]')


def _open_logs(page):
    _navigate(page, "dashboard")
    page.wait_for_selector('#page-logs', state="visible")
    for details in page.locator('#log-entries .log-task-details').all():
        if details.get_attribute('open') is None:
            details.locator(':scope > summary').click()


def _assert_logs(page, bodies: dict[str, str]):
    for task, body in bodies.items():
        page.locator('#log-entries').get_by_text(body, exact=True).first.wait_for(state="attached", timeout=10000)
        assert page.locator('#log-entries').get_by_text(body, exact=True).count() == 1, task


@pytest.mark.parametrize("width", [1360, 390])
def test_effort_facts_render_only_in_logs_live_and_after_reload(engine_ui, root, monkeypatch, width):
    ui, page, sockets, published = engine_ui, engine_ui["page"], [], []
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
                    or Path(os.environ["OUROBOROS_TEST_TEMP_ROOT"]) / "effort-logs-browser")
    engine = page.context.browser.browser_type.name
    assert engine == os.environ["OUROBOROS_UI_BROWSER_ENGINE"]  # Evidence names the engine that ran.
    monkeypatch.setenv("OUROBOROS_UI_EVIDENCE_DIR", str(evidence))
    page.set_viewport_size({"width": width, "height": 900})
    running = {task: {"task": {"id": task, "chat_id": 1, "type": "task"}} for task in EXPECTED}
    ctx = SimpleNamespace(RUNNING=running, DRIVE_ROOT=root, bridge=SimpleNamespace(push_log=published.append))
    page.route_web_socket("**/ws", lambda socket: sockets.append(socket))
    app = Starlette(routes=[Route("/api/logs/{name}", api_logs_tail)])
    app.state.drive_root = root
    with TestClient(app) as client:
        def forward(route):
            parsed = urlparse(route.request.url)
            response = client.get(parsed.path + ("?" + parsed.query if parsed.query else ""))
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

        page.route("**/api/logs/**", forward)
        page.goto(ui["url"] + "/")
        page.wait_for_selector('#chat-messages', state="attached")
        for _ in range(100):
            if sockets:
                break
            page.wait_for_timeout(50)
        assert sockets
        for task, provider, usage in _usage_events(root):
            _handle_llm_usage({"type": "llm_usage", "task_id": task, "model": f"{provider}/fixture",
                               "provider": provider, "source": "loop", "usage": usage}, ctx)
        assert [row["task_id"] for row in published] == list(EXPECTED)[:-1]
        assert all(row["chat_id"] == 1 for row in published)
        for row in published:
            sockets[-1].send(json.dumps({"type": "log", "chat_id": 1, "data": row}))
        # Positive control after the effort frames: this socket does reach Chat.
        sockets[-1].send(json.dumps({"type": "chat", "role": "assistant", "chat_id": 1,
                                     "content": "Control reply after effort frames"}))
        page.locator('#chat-messages').get_by_text("Control reply after effort frames").wait_for()
        assert _disclosed(page, '#chat-messages') == []
        assert page.locator('#toast-stack .toast').count() == 0
        capture(page, f"{engine}-{width}-chat-after-live-effort")
        _open_logs(page)
        live = {task: body for task, body in EXPECTED.items() if task != "effort-session"}
        _assert_logs(page, live)
        capture(page, f"{engine}-{width}-logs-live")
        observed = _settle_session(root)
        page.reload()
        page.wait_for_selector('#chat-messages', state="attached")
        assert _disclosed(page, '#chat-messages') == []
        _open_logs(page)
        _assert_logs(page, EXPECTED)
        assert page.locator('#toast-stack .toast').count() == 0
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator('#log-entries').get_by_text(EXPECTED["effort-session"], exact=True).scroll_into_view_if_needed()
        capture(page, f"{engine}-{width}-logs-history-reloaded")
        _navigate(page, "chat")
        assert _disclosed(page, '#chat-messages') == []
        history = client.get("/api/logs/events?limit=50").json()["entries"]
    persisted = {row["task_id"]: row for row in history if row.get("task_id") in EXPECTED}
    assert set(persisted) == set(EXPECTED)
    assert persisted["effort-session"]["observed_attempt"] == observed
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / f"{engine}-{width}-events.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

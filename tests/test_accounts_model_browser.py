"""#244 in the real Accounts, login and onboarding consumers.

Every status body below is produced by the real producer (`_status_payload`) over
the real `ClaudexorGateway`, with only the engine's HTTP answers faked, and then
rendered by the real SPA served from the candidate checkout. Accounts and names
are synthetic.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests.test_subscription_setup_browser import subscription_ui as subscription_ui  # noqa: F401  (fixture)

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

NOW = "2026-10-27T12:00:00Z"
UNSUPPORTED_TEXT = ("Claudexor 3.5.2 keeps accounts in an older format that this version of Ouroboros "
                    "no longer reads, so they can’t be listed or managed here. They are unchanged. "
                    "Claudexor 4.7.0 is installed and takes over when the agent daemon next restarts.")
NOT_LISTED = "Not listed — needs a newer Claudexor"
SIZES = ((1110, 750), (390, 844))


def _at(hours: float) -> str:
    moment = datetime.fromisoformat(NOW.replace("Z", "+00:00")) + timedelta(hours=hours)
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _named(harness, profile_id, *, name="", email="", plan="", verification="passed", source="vendor",
           enabled=True, availability="available"):
    return {"profile": {"harness_id": harness, "profile_id": profile_id, "display_name": name,
                        "credential_kind": "config_dir_login", "enabled": enabled},
            "status": {"harness_id": harness, "profile_id": profile_id, "availability": availability,
                       "verification": verification, "verification_source": source, "last_verified_at": _at(-0.4)},
            "identity": {"email": email, "plan": plan} if email or plan else None}


def _window(harness, subject, used, reset_hours):
    return {"subject": {"harness": harness, "subject_id": subject}, "freshness": "fresh", "observed_at": _at(-0.1),
            "constraints": [{"id": "five_hour", "label": "5h", "used_ratio": used, "resets_at": _at(reset_hours),
                             "freshness": "fresh"}],
            "availability": {"state": "exhausted" if used >= 1 else "available",
                             "resets_at": _at(reset_hours) if used >= 1 else None}}


def _pools(claude, codex, cursor):
    return [{"harness_id": "claude", "next_up": claude}, {"harness_id": "codex", "next_up": codex},
            {"harness_id": "cursor", "next_up": cursor}]


CATALOG = [{"id": harness, "displayName": label, "status": "ok", "enabled": True,
            "accessProfilesSupported": ["readonly", "workspace_write", "full"]}
           for harness, label in (("claude", "Claude Code"), ("codex", "Codex"), ("cursor", "Cursor"))]
READY = {"state": "ready", "version": "4.7.0", "target_version": "4.7.0", "source": "release"}
STAGED = {"state": "update_staged", "version": "3.5.2", "target_version": "4.7.0", "staged_version": "4.7.0",
          "source": "release"}
ALEKSANDRA = "aleksandra.konstantinopolskaya-research@example-lab.invalid"
OVERFLOW = dict(name="Work · research overflow (Q4 2026, shared seat)",
                email="research-overflow+q4@example-company.invalid", plan="Team Premium")
UNIFIED = [
    _named("claude", "claude-default", email=ALEKSANDRA, plan="Max 20x"),
    _named("claude", "work-research-q4-overflow", enabled=False, **OVERFLOW),
    _named("codex", "codex-default", email="ops-night-shift@example.invalid", plan="Pro"),
    _named("codex", "codex-backup-2", verification="not_run", source="local_store", availability="unknown"),
]
NONE = {"kind": "none", "reason": "no account"}
# The mock's five scenes (.review-evidence/mock-recovered/scenes.mjs) as ENGINE answers.
SCENES = {
    "modern": dict(engine="4.7.0", runtime=READY, quota=[_window("claude", "claude-default", 0.38, 2),
                                                         _window("codex", "codex-default", 0.12, 4)],
                   roster={"profiles": UNIFIED, "harnessAccounts": [], "accountPools": _pools(
                       {"kind": "profile", "profileId": "claude-default"},
                       {"kind": "profile", "profileId": "codex-default"}, NONE)}),
    "legacy": dict(engine="3.5.2", runtime=STAGED, quota=[_window("claude", None, 0.61, 3)],
                   roster={"profiles": [_named("claude", "work-research-q4-overflow", **OVERFLOW)], "harnessAccounts": [
                       {"harness_id": "claude", "native_credentials_enabled": True, "native_login_detected": True,
                        "identity": {"email": ALEKSANDRA, "plan": "Max 20x"},
                        "next_up": {"kind": "native", "route": "local_session"}},
                       {"harness_id": "codex", "native_credentials_enabled": True, "native_login_detected": True,
                        "identity": {"email": "ops-night-shift@example.invalid", "plan": "Pro"},
                        "next_up": {"kind": "native", "route": "local_session"}},
                       {"harness_id": "cursor", "native_credentials_enabled": True, "native_login_detected": False,
                        "identity": None, "next_up": {"kind": "none", "reason": "not signed in"}}]}),
    "failed": dict(engine="4.7.0", runtime=READY, quota=[], roster=httpx.ConnectError("roster read refused")),
    "empty": dict(engine="4.7.0", runtime=READY, quota=[],
                  roster={"profiles": [], "harnessAccounts": [], "accountPools": _pools(NONE, NONE, NONE)}),
    "unprofiled": dict(engine="4.7.0", runtime=READY,
                       quota=[_window("claude", "claude-default", 0.38, 2), _window("codex", None, 1, 3)],
                       roster={"profiles": UNIFIED, "harnessAccounts": [], "accountPools": _pools(
                           {"kind": "profile", "profileId": "claude-default"}, {"kind": "api_key_route"}, NONE)}),
}


def produced_status(tmp_path: Path, name: str, scene: dict) -> dict:
    """The real full-status producer's answer to one engine state."""
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway import claudexor_accounts
    from ouroboros.gateways import claudexor as wire

    config_dir = tmp_path / f"engine-{name}"
    (config_dir / "daemon").mkdir(parents=True)
    token = config_dir / "daemon" / "token"
    token.write_text("fixture-token", encoding="utf-8")
    (config_dir / "daemon" / "control-api.json").write_text(
        json.dumps({"host": "127.0.0.1", "port": 1, "tokenPath": str(token)}), encoding="utf-8")
    answers = {
        "/v2/handshake": {"compatible": True, "protocolMajor": wire.CLAUDEXOR_PROTOCOL_MAJOR,
                          "engine": {"version": scene["engine"]}},
        "/v2/agent-capabilities": {"harnesses": CATALOG},
        "/v2/harnesses": {"harnesses": []},
        "/v2/operations": {"operations": []},
        "/v2/credential-profiles": scene["roster"],
        "/v2/quota": {"snapshots": scene["quota"], "absences": []},
    }

    def respond(request):
        answer = answers[request.url.path]
        if isinstance(answer, Exception):
            raise answer
        return httpx.Response(200, json=answer)

    original_init = wire.ClaudexorGateway.__init__

    def initialize(gateway, endpoint):
        original_init(gateway, endpoint)
        headers, timeout = dict(gateway._client.headers), gateway._client.timeout
        gateway._client.close()
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1", headers=headers, timeout=timeout,
                                       trust_env=False, transport=httpx.MockTransport(respond))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owned, "owned_config_dir", lambda: config_dir)
        patch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(
            status_dict=lambda: {"state": "running", "runtime": dict(scene["runtime"])}))
        patch.setattr(wire.ClaudexorGateway, "__init__", initialize)
        return json.loads(json.dumps(claudexor_accounts._status_payload(False, join=False)))


def _evidence(name: str) -> Path | None:
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if not root:
        return None
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _open_accounts(page, url, width):
    # The isolated server has no engine for the vendor-CLI maintenance panel; its
    # empty inventory keeps that unrelated panel out of these frames.
    page.route("**/api/claudexor/maintenance/harnesses**", _fulfill({"harnesses": []}))
    page.goto(url, wait_until="domcontentloaded")
    if width < 980:
        page.click("[data-mobile-nav-toggle]")
        page.wait_for_selector("#primary-sidebar.open")
    page.click('[data-nav-page="settings"]')
    page.wait_for_selector(".settings-shell")
    page.click('[data-settings-tab="providers"]')
    page.wait_for_selector("#harness-accounts-groups .agent-family-card")


def _shoot(page, name, width, height):
    first = _evidence(f"{name}__{width}x{height}.png")
    if first is None:
        return
    page.screenshot(path=str(first), animations="disabled")
    page.set_viewport_size({"width": width, "height": 2400})  # the sticky Save bar sits below the subject
    page.wait_for_timeout(250)
    box = page.evaluate("""() => {
        const a = document.querySelector('.agents-service').getBoundingClientRect();
        const b = document.querySelector('#harness-accounts-section').getBoundingClientRect();
        return {x: Math.max(0, Math.min(a.left, b.left) - 8), y: Math.max(0, a.top - 8),
                width: Math.max(a.right, b.right) - Math.min(a.left, b.left) + 16, height: b.bottom - a.top + 16};
    }""")
    page.screenshot(path=str(_evidence(f"{name}__{width}x{height}__subject.png")), clip=box, animations="disabled")
    page.set_viewport_size({"width": width, "height": height})


def _fulfill(payload):
    """A one-argument handler: Playwright hands (route, request) to a two-parameter one."""
    return lambda route: route.fulfill(json=payload)


def _login_answer(route, state, after):
    """The engine's setup-job answers: it resolves the empty family's login onto
    `claude-default-2`, then settles on a verify-race reason the rows must judge."""
    if route.request.method == "POST" and route.request.url.endswith("/api/claudexor/login"):
        state["creates"].append(json.loads(route.request.post_data or "{}"))
        state["status"] = after
        return route.fulfill(json={"job_id": "job-244", "job": {
            "id": "job-244", "state": "running", "profileId": "claude-default-2"},
            "disclosure_native": True, "setup_login_source": "per_harness"})
    if route.request.method == "DELETE":
        return route.fulfill(json={"job": {"id": "job-244", "state": "cancelled"}})
    return route.fulfill(json={"job": {"id": "job-244", "state": "failed", "profileId": "claude-default-2",
                                       "outcome": {"reason": "auth_not_ready"}}})


def _login_transport(state, after):
    return lambda route: _login_answer(route, state, after)


def _browser(pw):
    engine = os.environ.get("OUROBOROS_UI_BROWSER_ENGINE", "chromium")
    return getattr(pw, engine).launch(headless=True), engine


def test_accounts_render_every_account_model_state_and_recover_on_refresh(direct_server_with_data, tmp_path):
    from playwright.sync_api import sync_playwright

    payloads = {name: produced_status(tmp_path, name, scene) for name, scene in SCENES.items()}
    # The producer's own verdicts (transport 3.2/protocol 3 still admits a 3.5.2 engine).
    legacy = payloads["legacy"]
    assert legacy["daemon"]["state"] == "running" and legacy["daemon"]["engine_version"] == "3.5.2"
    assert legacy["reads"] == {"catalog": "ok", "accounts": "failed", "quota": "ok"}
    assert legacy["facets"]["accounts"]["error"] == "account_model_unsupported" and legacy["profiles"] == {}
    assert payloads["failed"]["facets"]["accounts"]["error"] not in ("account_model_unsupported", None)
    assert payloads["empty"]["reads"]["accounts"] == "ok" and payloads["empty"]["unified_accounts"] is True
    report = {}
    with sync_playwright() as pw:
        browser, engine = _browser(pw)
        try:
            for theme in ("light", "dark"):
                for width, height in SIZES:
                    for name, payload in payloads.items():
                        page = browser.new_page(viewport={"width": width, "height": height})
                        errors = []
                        page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
                        page.add_init_script(f"Date.now = () => Date.parse({json.dumps(NOW)});"
                                             f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)});")
                        page.route("**/api/claudexor/status**", _fulfill(payload))
                        _open_accounts(page, direct_server_with_data["url"], width)
                        banner = page.locator("#agents-service-banner")
                        page.wait_for_function("() => !document.querySelector('#agents-service-banner')"
                                               ".textContent.startsWith('Checking')")
                        groups = page.locator("#harness-accounts-groups")
                        if name == "legacy":
                            assert banner.inner_text() == UNSUPPORTED_TEXT
                            assert banner.get_attribute("data-tone") == "warn"
                            assert groups.locator(".harness-account-row").count() == 0
                            assert groups.locator("[data-family-add], [data-harness-login], [data-next-up]").count() == 0
                            assert groups.inner_text().count(NOT_LISTED) == 3
                        elif name == "failed":
                            assert "could not be read" in banner.inner_text()
                            assert groups.locator("[data-family-add]").count() == 3  # Connect stays reachable
                        elif name == "empty":
                            assert banner.inner_text().startswith("Claudexor ready")
                            assert [b.inner_text() for b in groups.locator("[data-family-add]").all()] == ["Connect"] * 3
                        elif name == "unprofiled":
                            codex = groups.locator('.harness-account-row[data-profile="codex-default"]')
                            assert "Usage unavailable" in codex.inner_text()
                            assert "harness-exhausted" not in (codex.get_attribute("class") or "")
                        else:
                            claude = groups.locator('.harness-account-row[data-profile="claude-default"]')
                            assert "38% used · resets in 2h" in claude.inner_text()
                            assert claude.locator("[data-harness-remove]").count() == 1
                        report[f"{engine}/{theme}/{width}x{height}/{name}"] = {
                            "banner": banner.inner_text(), "rows": groups.locator(".harness-account-row").count()}
                        _shoot(page, f"built__{name}__{theme}", width, height)
                        assert not errors, errors
                        page.close()

            # Retry: the owner updates the engine; Refresh (and the poll) read the new roster.
            page = browser.new_page(viewport={"width": 1110, "height": 750})
            current = {"payload": payloads["legacy"]}
            page.add_init_script(f"Date.now = () => Date.parse({json.dumps(NOW)});")
            page.route("**/api/claudexor/status**", lambda route: route.fulfill(json=current["payload"]))
            page.route("**/api/claudexor/quota/refresh", lambda route: route.fulfill(
                json={"snapshots": SCENES["modern"]["quota"], "absences": []}))
            _open_accounts(page, direct_server_with_data["url"], 1110)
            page.wait_for_function("(text) => document.querySelector('#agents-service-banner').textContent === text",
                                   arg=UNSUPPORTED_TEXT)
            current["payload"] = payloads["modern"]
            page.click("#btn-harness-refresh")
            page.wait_for_selector('.harness-account-row[data-profile="codex-default"]')
            assert page.locator("#agents-service-banner").inner_text().startswith("Claudexor ready")
            assert page.locator("#harness-accounts-groups [data-family-add]").count() == 3
            page.close()
        finally:
            browser.close()
    target = _evidence(f"accounts-report-{engine}.json")
    if target:
        target.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


def test_an_empty_family_login_confirms_only_the_engine_resolved_row(direct_server_with_data, tmp_path):
    """Connect on an empty family sends no profile; the engine resolves `claude-default-2`. A verified
    SIBLING (`claude-default`) never confirms it; only that exact row does."""
    from playwright.sync_api import sync_playwright

    empty = produced_status(tmp_path, "login-empty", SCENES["empty"])
    sibling = produced_status(tmp_path, "login-sibling", {**SCENES["empty"], "roster": {
        "profiles": [_named("claude", "claude-default")], "harnessAccounts": [], "accountPools": _pools(NONE, NONE, NONE)}})
    resolved = produced_status(tmp_path, "login-resolved", {**SCENES["empty"], "roster": {
        "profiles": [_named("claude", "claude-default"), _named("claude", "claude-default-2")],
        "harnessAccounts": [], "accountPools": _pools(NONE, NONE, NONE)}})
    outcomes = {}
    with sync_playwright() as pw:
        browser, _engine = _browser(pw)
        try:
            for after, expected in ((sibling, "unconfirmed"), (resolved, "success")):
                page = browser.new_page(viewport={"width": 1110, "height": 750})
                state = {"status": empty, "creates": []}
                page.route("**/api/claudexor/status**", lambda route: route.fulfill(json=state["status"]))

                page.route("**/api/claudexor/login**", _login_transport(state, after))
                _open_accounts(page, direct_server_with_data["url"], 1110)
                page.locator('[data-family="claude"] [data-family-add]').click()
                page.wait_for_function("() => /Signed in\\.|Could not confirm the sign-in yet/.test("
                                       "document.querySelector('[data-family-login=\"claude\"]').textContent)",
                                       timeout=45_000)
                card = page.locator('[data-family-login="claude"]').inner_text()
                outcomes[expected] = card
                assert state["creates"] and state["creates"][0].get("profile_id", "") == ""
                if expected == "success":
                    assert "Signed in." in card
                else:
                    assert "Could not confirm the sign-in yet" in card and "Signed in." not in card
                shot = _evidence(f"built__login-{expected}__light__1110x750.png")
                if shot:
                    page.screenshot(path=str(shot), animations="disabled")
                page.close()
        finally:
            browser.close()
    assert set(outcomes) == {"unconfirmed", "success"}


def test_onboarding_accounts_step_explains_a_pre_unified_engine(subscription_ui, tmp_path):
    ui, page = subscription_ui, subscription_ui["page"]
    ui["fixture"]["status"] = produced_status(tmp_path, "onboarding-legacy", SCENES["legacy"])
    page.goto(ui["url"] + "/onboarding")
    page.wait_for_function("(text) => document.querySelector('#agents-status-note')?.textContent === text",
                           arg=UNSUPPORTED_TEXT)
    rows = page.locator("#agents-family-list")
    assert rows.locator("[data-agent-connect]").count() == 0
    assert rows.inner_text().count(NOT_LISTED) == 4
    assert "newer Claudexor" in page.locator("#agents-outcome").inner_text()
    shot = _evidence("built__onboarding-legacy__light__1360x900.png")
    if shot:
        page.screenshot(path=str(shot), animations="disabled")

"""Settings › Accounts and keys › Remote engines, a remote Main and remote Agents rows, in a real browser.

The SPA is this checkout's own, on the row-level Settings (each page with its own Save). ``GET``/``POST
/api/settings`` and ``POST /api/owner/safety-model`` reach the real handlers over a temporary data folder
(``test_main_connection_choice_browser.real_settings``), and every ``/api/engine-connections`` request reaches the real
handlers (``ouroboros.gateway.engine_connections``) over the real attach path (``remote_engines``). Only OpenSSH and
the engine are stood in (``tests._remote_engine_stand``): a stand-in ``ssh`` on a real process boundary and a
bearer-checked loopback engine. No real host, account or token. Frames are taken at the owner's width (1512 x 949)
and a phone's (390), light and dark, when ``OUROBOROS_UI_EVIDENCE_DIR`` names a folder.
"""
from __future__ import annotations

import json
from urllib.parse import urlparse

import pytest

from tests import test_main_connection_choice_browser as main_choice
from tests import test_subscription_role_routes_browser as roles
from tests._remote_engine_stand import TOKEN
from tests.settings_browser_helpers import open_settings, press_editor_save, settings_page

pytest_plugins = ("tests._remote_engine_stand",)
pytestmark = [pytest.mark.ui_browser]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui
real_settings = main_choice.real_settings
REMOTE_MAIN = "claudexor@vps::codex=gpt-6-remote"
KEY = "OUROBOROS_ENGINE_CONNECTIONS"
OWNER = {"width": 1512, "height": 949}   # the owner's Mac window
PHONE = {"width": 390, "height": 844}
ROSTER = {"profiles": [
    {"profile": {"harness_id": "codex", "profile_id": "remote-main", "display_name": "Remote main", "enabled": True},
     "status": {"verification": "passed", "availability": "available"}},
    {"profile": {"harness_id": "codex", "profile_id": "remote-spare", "enabled": False},
     "status": {"verification": "not_run", "availability": "unknown"}}], "accountPools": []}


@pytest.fixture
def engines_ui(real_settings, host, monkeypatch):
    """``real_settings`` whose ``/api/engine-connections`` requests reach the real handlers; every answer is kept.
    The rows live in the saved document each test seeds (a route key is read live from it), never in this process's
    environment, where the stand puts them for its own tests: a Settings read would show them over the document."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros import remote_engines
    from ouroboros.gateway.engine_connections import engine_connection_routes

    monkeypatch.delenv(remote_engines.ENGINE_CONNECTIONS_KEY, raising=False)
    answers = []
    with TestClient(Starlette(routes=engine_connection_routes())) as client:
        def forward(route):
            request = route.request
            url = urlparse(request.url)
            response = client.request(request.method, url.path + (f"?{url.query}" if url.query else ""))
            answers.append((request.method, url.path, response.status_code, response.text))
            route.fulfill(status=response.status_code, content_type="application/json", body=response.text)

        page = real_settings["page"]
        page.set_viewport_size(OWNER)
        page.route("**/api/engine-connections**", forward)
        # No engine serves the vendor-CLI maintenance panel here; its empty inventory keeps it out of the frames.
        page.route("**/api/claudexor/maintenance/harnesses**", lambda route: route.fulfill(json={"harnesses": []}))
        try:
            yield {**real_settings, "host": host, "answers": answers}
        finally:
            remote_engines.retire("build-box")  # the stand retires its own names; this one the owner added here
        page_text = page.content()
        assert TOKEN not in page_text, "the page never shows the engine's token"
        assert not [answer for answer in answers if TOKEN in answer[3]], "no answer ever carries the token"
        assert not [exchange for exchange in real_settings["exchanges"] if TOKEN in json.dumps(exchange)]


def _frames(page, subject, name):
    """The owner's width and a phone's, light and dark: nothing wider than the window."""
    for size in (OWNER, PHONE):
        page.set_viewport_size(size)
        for scheme in ("light", "dark"):
            page.emulate_media(color_scheme=scheme)
            subject.scroll_into_view_if_needed()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (name, size)
            roles.capture(page, f"{name}-{scheme}-{size['width']}")
    page.emulate_media(color_scheme="light")
    page.set_viewport_size(OWNER)


def _card(page):
    """Accounts and keys' Remote engines section, on the list the shell shows."""
    return settings_page(page, "accounts").locator(f'section[data-setting-row="{KEY}"]')


def _rows(ui):
    return main_choice._disk(ui).get(KEY)


def test_an_engine_is_added_saved_checked_and_disconnected_on_its_own_page(engines_ui):
    """The list contacts no host; a new engine's page refuses without a name, then writes one entry and returns to the
    list, which names it. Its page: Check attaches (one forward; only descriptor and token reads) and lists that
    engine's own accounts; an edit sets Check aside until it is saved; a reload reads it again; Disconnect closes this
    computer's tunnel only; Turn off takes it out of new work."""
    from playwright.sync_api import expect

    from ouroboros import remote_engines

    ui, host = engines_ui, engines_ui["host"]
    host.state["profiles"] = ROSTER
    main_choice._seed(ui, {KEY: []})
    page = main_choice._open(ui, tab="accounts")
    card = _card(page)
    expect(card.locator("[data-engine-rows]")).to_have_text("No remote engine yet.")
    expect(card.locator("[data-engine-more] summary")).to_have_text("More: what runs where")
    assert not host.ssh_calls(), "opening Settings contacts no host"
    _frames(page, card, "remote-engines-empty")

    card.locator('[data-open-path="accounts/engine/new"]').click()
    form = settings_page(page, "accounts/engine/new")
    expect(form.locator("[data-engine-status]")).to_have_text("Not saved")
    expect(form.locator("[data-engine-check]")).to_have_count(0)  # a form never saved has nothing to check
    form.locator("[data-editor-save]").click()
    expect(form.locator("[data-editor-result]")).to_have_text("Not saved. Name the new remote engine.")
    expect(form.locator('[data-engine-field="label"]')).to_be_focused()
    fields = {"label": "Build box", "ssh_host": "ouroboros@vds", "engine_home": str(host.engine_home),
              "remote_root": str(host.remote_root), "workspaces": str(host.workspace)}
    for field, value in fields.items():
        form.locator(f'[data-engine-field="{field}"]').fill(value)
    _frames(page, form, "remote-engine-new")
    form.locator("[data-editor-save]").click()
    expect(card.locator("[data-engine-result]")).to_have_text("Added Build box.")
    head = card.locator('[data-engine-name="build-box"]')
    expect(head).to_contain_text("Build box")
    expect(head).to_contain_text("ouroboros@vds")
    assert _rows(ui) == [{"name": "build-box", "label": "Build box", "ssh_host": "ouroboros@vds",
                          "engine_home": str(host.engine_home), "remote_root": str(host.remote_root),
                          "workspaces": [str(host.workspace)]}], "one entry, under a name minted from its label"
    assert not host.ssh_calls(), "saving an engine contacts no host"
    _frames(page, card, "remote-engines-added")

    head.locator("[data-entry-open]").click()
    engine = settings_page(page, "accounts/engine/build-box")
    status, meta = engine.locator("[data-engine-status]"), engine.locator("[data-engine-meta]")
    expect(engine.locator('[data-engine-field="ssh_host"]')).to_have_value("ouroboros@vds")
    expect(status).to_have_text("Not checked")
    engine.locator("[data-engine-check]").click()
    expect(status).to_have_text("Connected")
    expect(status).to_have_attribute("data-tone", "ok")
    expect(meta).to_contain_text("engine 5.1.0")
    expect(engine.locator("[data-editor-result]")).to_have_text("Connected: codex · agents: codex")
    expect(engine.locator("[data-engine-accounts]")).to_have_text(
        "Accounts there: Remote main (codex) · remote-spare (codex, disabled, not verified)")
    commands = [" ".join(call["command"]) for call in host.ssh_calls() if not call["forward"]]
    forwards = [call for call in host.ssh_calls() if call["forward"]]
    assert len(forwards) == 1 and forwards[0]["no_command"], "one -N -L forward, and nothing started on the host"
    assert commands and all("control-api.json" in command or "cat " in command for command in commands), commands
    _frames(page, engine, "remote-engine-connected")

    # Check speaks for the SAVED engine: an edit sets it aside, and Save brings it back.
    engine.locator('[data-engine-field="remote_root"]').fill(f"{host.remote_root}-next")
    expect(engine.locator("[data-engine-check]")).to_be_disabled()
    expect(status).to_have_text("Unsaved changes")
    expect(meta).to_have_text("· Saved engine: connected · Save, then Check reads the new fields")
    expect(engine.locator("[data-engine-accounts]")).to_be_hidden()
    _frames(page, engine, "remote-engine-unsaved-edit")
    assert press_editor_save(page) == "Saved · applies from the next model call"
    assert _rows(ui)[0]["remote_root"] == f"{host.remote_root}-next"
    expect(engine.locator("[data-engine-check]")).to_be_enabled()

    page = main_choice._open(ui, tab="accounts/engine/build-box")  # read again from the document on disk
    engine = settings_page(page, "accounts/engine/build-box")
    status = engine.locator("[data-engine-status]")
    expect(engine.locator('[data-engine-field="remote_root"]')).to_have_value(f"{host.remote_root}-next")
    expect(status).to_have_text("Connected")  # this installation's live tunnel, re-read
    engine.locator("[data-engine-disconnect]").click()
    expect(engine.locator("[data-editor-result]")).to_have_text(
        "Disconnected: this computer's tunnel is closed; the engine and its runs on the host continue.")
    expect(status).to_have_text("Not connected")
    assert not remote_engines.published("build-box"), "Disconnect stops this installation's forward"
    engine.locator("[data-engine-toggle]").click()
    expect(engine.locator("[data-editor-result]")).to_have_text("Turned off · applies from the next model call")
    assert _rows(ui)[0].get("enabled") is False
    expect(engine.locator("[data-engine-toggle]")).to_have_text("Turn on")
    open_settings(page, "accounts")
    expect(_card(page).locator('[data-engine-name="build-box"]')).to_contain_text("Off: takes no new work")


def _items(document):
    """Each Agents row as the document on disk stores the roster (a JSON text or an object), by its id."""
    roster = document["OUROBOROS_SUBAGENTS"]
    roster = json.loads(roster) if isinstance(roster, str) else roster
    return {item["subagent_id"]: item for item in roster["items"]}


def _routes(document):
    """Each row's route, an empty pin read as none: both are Auto, and the server spells a session row's Auto as
    ``credential_profile_id: ""`` whenever it writes the roster (``route_spec.route_spec_dict``, local rows alike)."""
    return {key: {field: value for field, value in item["route"].items() if field != "credential_profile_id" or value}
            for key, item in _items(document).items()}


def _session(key):
    return {"subagent_id": key, "recommended_use": f"remote {key}", "access": "full",
            "route": {"kind": "agent_session", "target_id": "codex=gpt-6-remote", "engine": "vps"}}


def test_main_and_an_agent_pin_that_engines_own_accounts_through_their_own_saves(engines_ui):
    """Models › Main: the remote source, its model from that engine and an account from that engine's roster, saved
    through Main's own Save, which writes Main and its pin and nothing else. Agents: one remote session row works in a
    declared folder on the host under a remote account, saved through its page, the other rows, Main and the engine
    rows as they were. A closed tunnel keeps both saved choices, said as not checked."""
    from playwright.sync_api import expect

    from ouroboros import remote_engines

    ui, host = engines_ui, engines_ui["host"]
    host.state["profiles"] = ROSTER
    workspace = str(host.workspace)
    # A remote session row may review; one marked Reviewer keeps the review pool from asking to go empty.
    roster = {"enabled": True, "items": [{**_session("copy"), "review_eligible": True}, _session("inplace")]}
    main_choice._seed(ui, {KEY: host.rows, "OUROBOROS_SUBAGENTS": json.dumps(roster)})
    assert remote_engines.check("vps")["ok"], "the tunnel the Accounts page's Check opens (proved above)"
    page = main_choice._open(ui, tab="models/role/main")
    main = main_choice._row(page, "main")
    source = main.locator("select[data-model-role-source]")
    option = source.locator('optgroup[label="Remote engines"] option[value="remote:vps:codex"]')
    expect(option).to_have_text("On Build box · Codex (model)")
    source.select_option("remote:vps:codex")
    model = main.locator("[data-model-role-model]")
    model.click()
    suggestion = page.locator(f'#{model.get_attribute("aria-controls")} [data-model-value="gpt-6-remote"]')
    expect(suggestion).to_be_visible()
    suggestion.click()
    expect(model).to_have_value("gpt-6-remote")
    account = main.locator("select[data-model-role-account]")
    expect(account).to_be_visible()
    assert account.locator("option").evaluate_all("nodes => nodes.map(n => n.value)") == ["", "remote-main", "remote-spare"], \
        "that engine's accounts, never this computer's"
    account.select_option("remote-main")
    expect(main.locator("[data-model-role-status]")).to_have_text(
        "Runs on Build box: its requests go to that host under that engine's accounts while the tools it calls run on "
        "this computer. It is never moved to another engine, this computer's included (Fallback models you set still "
        "apply as for any route); an Emergency Stop here may leave a request running there until it ends.")
    _frames(page, main, "remote-main-chosen")
    before = main_choice._disk(ui)
    posts = main_choice._save(ui)
    assert posts and all(status == 200 for _m, _p, _b, status, _r in posts), posts
    disk = main_choice._disk(ui)
    assert main_choice._lanes(ui) == {"OUROBOROS_MODEL": REMOTE_MAIN}, "Main alone: no other role was authored"
    pins = disk.get("OUROBOROS_MODEL_ACCOUNTS") or {}
    assert (json.loads(pins) if isinstance(pins, str) else pins) == {"main": "remote-main"}
    assert disk[KEY] == before[KEY] == host.rows, "an engine row is never rewritten by a role's Save"
    assert _routes(disk) == {"copy": _session("copy")["route"], "inplace": _session("inplace")["route"]}, "nor an Agents row"

    form = roles.open_agent(ui, "inplace")
    pins = form.locator('[data-subagent-field="account"] option').evaluate_all("nodes => nodes.map(n => n.value)")
    assert pins == ["", "remote-main", "remote-spare"], pins
    form.locator('[data-subagent-field="account"]').select_option("remote-main")
    places = form.locator('[data-subagent-field="workspace"] option').evaluate_all(
        "nodes => nodes.map(n => [n.value, n.textContent])")
    assert places == [["", "A copy of the local project, staged on Build box"], [workspace, f"{workspace} on Build box, in place"]]
    form.locator('[data-subagent-field="workspace"]').select_option(workspace)
    expect(form.locator("[data-subagent-remote-note]")).to_have_text(
        "Its tools and files are on that host. An Emergency Stop here may leave it working until this computer "
        "reconnects.")
    _frames(page, form, "remote-agent-chosen")
    press_editor_save(page)
    disk = main_choice._disk(ui)
    routes = _routes(disk)
    assert routes == {"copy": _session("copy")["route"], "inplace": {
        "kind": "agent_session", "target_id": "codex=gpt-6-remote", "engine": "vps", "credential_profile_id": "remote-main",
        "workspace": workspace}}, routes
    assert _items(disk)["copy"].get("review_eligible") is True, "a remote reviewer row keeps its mark"
    assert disk["OUROBOROS_MODEL"] == REMOTE_MAIN and disk[KEY] == host.rows, "an Agents Save touches no role or engine row"

    remote_engines.retire("vps")  # the tunnel closes: both saved choices stay, said as not checked
    page = main_choice._open(ui, tab="models/role/main")
    main = main_choice._row(page, "main")
    # one engine is named one way on the page, connected or not: its saved name and the source's own word
    expect(main.locator("select[data-model-role-source] option:checked")).to_have_text(
        "On Build box · Codex (model) · not checked")
    expect(main.locator("select[data-model-role-account] option:checked")).to_have_text("remote-main · not checked")
    _frames(page, main, "remote-main-not-connected")


def test_owed_stops_a_removal_and_an_unread_status_are_said_on_the_list_and_the_page(engines_ui):
    """A stop Emergency Stop still owes on an engine is its head's chip and its page's line; Remove asks first and says
    those stops stay undelivered; the removed connection's owed stop stays on the list with how it can be delivered;
    a status read that fails is said, never painted as not checked."""
    from playwright.sync_api import expect

    from ouroboros import config, delegate_custody as custody, remote_stops

    ui, host = engines_ui, engines_ui["host"]
    main_choice._seed(ui, {KEY: host.rows})
    data = config.DATA_DIR
    owed_ref = {"kind": "remote", "connection": "vps", "ssh_host": host.rows[0]["ssh_host"],
                "engine_home": host.rows[0]["engine_home"], "store": host.state["store"]}
    run = custody.RunCustody(run_id="run-owed", task_id="t-owed", invocation_id="inv-owed", route_id="codex", model="m",
                             access="full", mode="ask", ledger_root=str(data), engine_ref=owed_ref,
                             daemon_store_id=host.state["store"])
    assert custody.record_started(data, run)
    remote_stops.record_owed_stops(data)
    page = main_choice._open(ui, tab="accounts")
    head = _card(page).locator('[data-engine-name="vps"]')
    expect(head).to_contain_text("1 stop owed")
    _frames(page, _card(page), "remote-engines-owed")
    head.locator("[data-entry-open]").click()
    engine = settings_page(page, "accounts/engine/vps")
    expect(engine.locator("[data-engine-owed]")).to_have_text(
        "Emergency Stop: 1 run is not yet confirmed stopped on this engine; the stop is delivered when it answers this "
        "computer, and the work may continue there until then")
    expect(engine.locator("[data-engine-status]")).to_have_attribute("data-tone", "warn")
    _frames(page, engine, "remote-engine-owed")

    engine.locator("[data-editor-remove]").click()
    dialog = page.locator(".confirm-dialog-backdrop")
    expect(dialog).to_contain_text("Remove Build box?")
    expect(dialog).to_contain_text("Work already running on that host continues there")
    expect(dialog).to_contain_text("removing it leaves those stops undelivered until it is saved again")
    _frames(page, dialog, "remote-engine-remove-asks")
    dialog.locator("[data-confirm-ok]").click()
    card = _card(page)
    expect(card.locator("[data-engine-result]")).to_have_text(  # the server's own sentence of what the removal did
        "Removed Build box. Remote engine Build box: this computer's tunnel to it is closed; work already running there "
        "continues on that host.")
    assert _rows(ui) == []
    expect(card.locator("[data-engine-owed]")).to_have_text(
        "Emergency Stop: 1 run is not yet confirmed stopped on vps, which is no longer saved; the stop is delivered "
        "only after a connection named vps with the same SSH destination and engine home is saved again, and the work "
        "may continue there until then.")
    _frames(page, card, "remote-engines-removed-owed")

    main_choice._seed(ui, {KEY: host.rows})
    page.route("**/api/engine-connections", lambda route: route.fulfill(status=500, json={"error": "store busy"}))
    page = main_choice._open(ui, tab="accounts")
    expect(_card(page).locator('[data-engine-name="vps"]')).to_contain_text("Status unknown")
    open_settings(page, "accounts/engine/vps")
    engine = settings_page(page, "accounts/engine/vps")
    expect(engine.locator("[data-engine-status]")).to_have_text("Status unknown")
    expect(engine.locator("[data-engine-owed]")).to_contain_text("could not be read")
    _frames(page, engine, "remote-engine-status-unread")
    page.unroute("**/api/engine-connections")

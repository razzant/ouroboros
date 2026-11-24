"""A skill signs in to the service it declares from its card and sheet (no running backend).

The Skills page runs against stubbed ``/api/sign-in*`` answers shaped as ``ouroboros/gateway/sign_in.py`` gives
them. The card's meta line opens the skill's sheet at Sign-ins; Add account opens the sign-in editor in place with the
only client chosen and, on this non-loopback page, the floor sentence; Open the sign-in page posts ``begin`` and waits.
The provider's return is the status the return page leaves behind, read when the window regains focus: the account's
row appears with its result line. The paste path finishes on the same route with the address the browser stopped
on, and Revoke asks in place first, naming the sibling sign-in it may end.
"""
from __future__ import annotations

import pytest

from tests._skills_page_harness import URL, capture, open_skills, skills_browser as _skills_browser

pytestmark = [pytest.mark.ui_browser]
skills_browser = _skills_browser
STATUS, SIGN_IN = "/api/sign-in/status", "/api/sign-in"
RETURN = "http://127.0.0.1:8765/onboarding/static/provider-return.html"
ISSUER = "https://accounts.example.test"
REG, CONSENT = "r" * 64, "k" * 32  # a saved client's registration and an account's consent, as the core's status carries them
CALENDAR = {
    "name": "calendar", "source": "external", "type": "script", "version": "2.1.0", "description": "Calendar events.",
    "payload_root": "skills/external/calendar", "enabled": True, "content_hash": "c" * 64, "permissions": [],
    "grants": {"all_granted": True}, "owner_attestable": True, "review_status": "clean", "review_stale": False,
    "executable_review": True, "review_profile": "", "removal": "delete",
    "review_gate": {"status": "clean", "stale": False, "executable_review": True},
}
CLIENT = {"id": "c-1", "label": "Personal", "issuer": ISSUER, "client_id_tail": "a1b2", "return": "loopback",
          "source": "owner", "secret_set": False, "registration": REG, "used_by": []}


def account(key, label, since, **extra):
    return {"key": key, "label": label, "subject": key, "verified": True, "state": "signed_in", "since": since,
            "scope": ["openid", "email"], "missing_required": [], "not_allowed": [],
            "client": {"id": "c-1", "label": "Personal"}, "via": "card", "siblings": [], "consent": CONSENT, **extra}


def status(accounts=(), pending=None):
    service = {"skill": "calendar", "service": "google", "label": "Google", "issuer": ISSUER, "scopes": ["openid", "email"],
               "scope_labels": {}, "optional_scopes": [], "accounts_mode": "many", "optional": True,
               "notes": {"refused": "admin_policy_enforced: ask your administrator."}, "accounts": list(accounts),
               "pending": pending, "clients": [CLIENT]}
    return {"ok": True, "services": [service], "return_address": RETURN}


def _open_sheet(page):
    page.goto(URL, wait_until="networkidle")
    meta = page.locator('.skills-card[data-skill="calendar"] [data-skill-sign-ins]')
    meta.get_by_role("button").click()
    return meta, page.locator('dialog.skills-detail-sheet [data-sign-in-group="google"]')


def _begin(skills, group, attempt, started):
    """The pending status and the begin receipt name the same ordinary attempt."""
    skills.answers[STATUS] = status(skills.answers[STATUS]["services"][0]["accounts"],
                                   {"started": started, "client": "Personal", "attempt_id": attempt})
    group.locator('[data-sign-in-editor="sign_in"]').get_by_role("button", name="Open the sign-in page").click()
    return group.locator('[data-sign-in-editor="waiting"]')


def test_add_account_waits_at_the_provider_then_shows_the_row_on_return_and_by_paste(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.add_init_script("window.open = (...args) => { window.opened = args; return null; };")
        skills = open_skills(page, extensions=[CALENDAR])
        required = status()
        required["services"][0]["optional"] = False
        skills.answers[STATUS] = required
        replies = {"begin": {"ok": True, "url": f"{ISSUER}/authorize?state=s1", "redirect_uri": RETURN,
                             "client": {"id": "c-1", "label": "Personal"}, "scope": "openid email", "attempt_id": "attempt-one"}}
        skills.answers[SIGN_IN] = lambda request: (200, replies[request.post_data_json["action"]])
        meta, group = _open_sheet(page)
        expect(meta).to_have_text("Not signed in to Google")
        expect(page.locator('.skills-card[data-skill="calendar"][data-needs-you]')).to_have_count(1)
        expect(page.locator('[data-installed-filter="needs"] .skills-category-count')).to_have_text("1")
        expect(group).to_contain_text("Sign-ins · Google")
        group.get_by_role("button", name="Add account").click()
        editor = group.locator('[data-sign-in-editor="sign_in"]')
        expect(editor.get_by_role("radio")).to_be_checked()  # the only client is chosen before the provider
        expect(editor).to_contain_text("This sign-in comes back to the computer where Ouroboros runs.")  # skills.test is no loopback host
        capture(page, "chromium", 1280, "sign-in-editor")

        waiting = _begin(skills, group, "attempt-one", "2026-11-22T10:00:00+00:00")
        expect(waiting).to_contain_text("Waiting for you to finish at Google…")
        expect(waiting).to_contain_text("admin_policy_enforced: ask your administrator.")
        assert skills.posted(SIGN_IN)[-1] == {"action": "begin", "skill": "calendar", "service": "google", "optional_scopes": [],
                                              "scope_words": ["openid", "email"], "login_hint": "", "client": "c-1", "registration": REG}
        assert page.evaluate("window.opened[0]") == f"{ISSUER}/authorize?state=s1"
        expect(waiting.locator("details")).to_have_attribute("open", "")  # here the browser cannot come back: the paste is open

        # The return page completed the sign-in; coming back to this window reads the status again.
        first = account("a-1", "owner@example.test", "2026-11-22T10:01:00+00:00", attempt_id="attempt-one")
        other = account("a-neighbour", "neighbour@example.test", "2026-11-22T10:02:00+00:00", attempt_id="attempt-neighbour")
        skills.answers[STATUS] = status([first, other])
        page.evaluate("window.dispatchEvent(new Event('focus'))")
        row = group.locator('[data-sign-in-row="a-1"]')
        expect(row.locator('[data-sign-in-result="a-1"]')).to_have_text("Signed in as owner@example.test via Personal.")
        expect(row).to_contain_text("owner@example.test via Personal")
        expect(waiting).to_have_count(0)
        expect(meta).to_have_text("Signed in to Google: owner@example.test · neighbour@example.test")
        expect(group.locator('[data-sign-in-result="a-neighbour"]')).to_be_empty()
        capture(page, "chromium", 1280, "sign-in-row")

        # The paste path: the address the browser stopped on finishes the same sign-in route.
        group.get_by_role("button", name="Add account").click()
        replies["begin"] = {**replies["begin"], "url": f"{ISSUER}/authorize?state=s2", "attempt_id": "attempt-two"}
        waiting = _begin(skills, group, "attempt-two", "2026-11-22T10:05:00+00:00")
        replies["complete"] = {"ok": True, "skill": "calendar", "service": "google", "label": "Google",
                               "account": {"key": "a-2", "label": "second@example.test"}, "scope": "openid email",
                               "missing_required": [], "state": "signed_in", "attempt_id": "attempt-two"}
        skills.answers[STATUS] = status([first, other, account("a-2", "second@example.test", "2026-11-22T10:06:00+00:00")])
        waiting.get_by_label("If your browser did not come back here, paste the address it stopped on").fill(f"{RETURN}?state=s2&code=c2")
        waiting.get_by_role("button", name="Finish sign-in").click()
        expect(group.locator('[data-sign-in-result="a-2"]')).to_have_text("Signed in as second@example.test via Personal.")
        assert skills.posted(SIGN_IN)[-1] == {"action": "complete", "returned_url": f"{RETURN}?state=s2&code=c2",
                                              "attempt_id": "attempt-two"}
        expect(group.locator('[data-sign-in-editor="waiting"]')).to_have_count(0)

        # A paste that got no usable answer (a 502, then a kept answer): one status read by this attempt, the one next step, one send per press.
        group.get_by_role("button", name="Add account").click()
        replies["begin"] = {**replies["begin"], "url": f"{ISSUER}/authorize?state=s3", "attempt_id": "attempt-three"}
        waiting = _begin(skills, group, "attempt-three", "2026-11-22T10:10:00+00:00")
        outcomes = iter([(502, {"error": "Bad gateway"}),
                         (200, {"ok": False, "code": "SIGN_IN_REQUIRED", "reason": "answer_saved", "attempt_id": "attempt-three", "error": "kept"}),
                         (200, {**replies["complete"], "account": {"key": "a-3", "label": "third@example.test"}, "attempt_id": "attempt-three"})])
        skills.answers[SIGN_IN] = lambda request: (next(outcomes) if request.post_data_json["action"] == "complete"
                                                   else (200, replies[request.post_data_json["action"]]))
        waiting.get_by_label("If your browser did not come back here, paste the address it stopped on").fill(f"{RETURN}?state=s3&code=c3")
        for said in ("sends each one-time code only once", "Choose Finish sign-in again; its one-time code isn’t sent a second time."):
            waiting.get_by_role("button", name="Finish sign-in").click()
            expect(waiting.locator('[data-sign-in-result="editor"]')).to_contain_text(said)
            expect(waiting).to_have_count(1)  # no row bears this attempt: still waiting
        skills.answers[STATUS] = status([first, other, account("a-3", "third@example.test", "2026-11-22T10:12:00+00:00", attempt_id="attempt-three")])
        waiting.get_by_role("button", name="Finish sign-in").click()
        expect(group.locator('[data-sign-in-result="a-3"]')).to_have_text("Signed in as third@example.test via Personal.")
        assert [body["returned_url"] for body in skills.posted(SIGN_IN) if body["action"] == "complete" and "state=s3" in body["returned_url"]] \
            == [f"{RETURN}?state=s3&code=c3"] * 3  # one send per press, never an automatic one
    finally:
        skills_page = locals().get("skills")
        if skills_page is not None:
            skills_page.close(browser)
        else:
            browser.close()


def test_sign_in_uses_the_selected_it_client_for_the_work_account(skills_browser):
    from playwright.sync_api import expect

    work_hint = "work@example.test"
    work_return = "https://sign-in.example.test/onboarding/static/provider-return.html"
    work_client = {**CLIENT, "id": "c-2", "label": "Work (IT-supplied)", "client_id_tail": "c3d4",
                   "return": f"exact:{work_return}", "registration": "w" * 64}
    read_events = "https://www.googleapis.com/auth/calendar.events.readonly"
    authorize_url = f"{ISSUER}/authorize?state=work"
    sign_in_status = status()
    personal = {**CLIENT, "used_by": [{"skill": "mail", "account": "personal@example.test"}]}
    alias = "https://issuer.example.test/auth/userinfo.email"
    sign_in_status["services"][0].update(clients=[personal, work_client], scope_spellings={"email": [alias]}, optional_scopes=[
        {"scope": read_events, "label": "Read calendar events"},
        {"scope": "https://www.googleapis.com/auth/calendar.events", "label": "Edit calendar events", "default": True},
    ])

    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.add_init_script("window.open = (...args) => { window.opened = args; return null; };")
        skills = open_skills(page, extensions=[CALENDAR])
        skills.answers[STATUS] = sign_in_status
        skills.answers[SIGN_IN] = {"ok": True, "url": authorize_url, "redirect_uri": work_return,
                                   "client": {"id": work_client["id"], "label": work_client["label"]},
                                   "scope": f"openid email {read_events}", "attempt_id": "attempt-work"}
        _meta, group = _open_sheet(page)
        group.get_by_role("button", name="Add account").click()
        editor = group.locator('[data-sign-in-editor="sign_in"]')
        expect(editor.locator('[data-sign-in-permissions]')).to_contain_text(f'email (email, {alias})')
        clients = editor.get_by_role("radiogroup", name="Client")
        expect(clients.get_by_role("radio")).to_have_count(2)
        note = editor.locator("[data-sign-in-return-note]")
        account_hint = editor.get_by_label("Account", exact=True)
        personal_radio = clients.get_by_role("radio", name="Personal")
        account_hint.fill("personal@example.test")
        expect(personal_radio).to_be_checked()
        expect(note).to_contain_text("comes back to the computer where Ouroboros runs")
        account_hint.fill(work_hint)
        expect(personal_radio).not_to_be_checked()
        personal_radio.check()
        account_hint.fill("another@example.test")
        account_hint.fill(work_hint)
        expect(personal_radio).to_be_checked()  # a manual choice survives hint edits
        editor.get_by_role("checkbox", name="Read calendar events").check()
        editor.get_by_role("checkbox", name="Edit calendar events").uncheck()
        clients.get_by_role("radio", name="Work (IT-supplied)").check()
        expect(note).to_be_visible()
        expect(note).to_contain_text(work_return)
        expect(note).not_to_contain_text("comes back to the computer where Ouroboros runs")
        note.scroll_into_view_if_needed()
        capture(page, "chromium", 1280, "sign-in-it-client")

        editor.get_by_role("button", name="Open the sign-in page").click()
        waiting = group.locator('[data-sign-in-editor="waiting"]')
        expect(waiting).to_be_visible()
        expect(waiting.locator("details")).to_have_js_property("open", False)
        page.wait_for_function("url => window.opened?.[0] === url", arg=authorize_url)
        assert skills.posted(SIGN_IN) == [{"action": "begin", "skill": "calendar", "service": "google",
                                          "optional_scopes": [read_events], "scope_words": ["openid", "email", alias, read_events],
                                          "login_hint": work_hint, "client": "c-2", "registration": "w" * 64}]
    finally:
        if skills is not None:
            skills.close(browser)
        else:
            browser.close()


def test_save_client_approves_the_complete_shown_map_without_starting_sign_in(skills_browser):
    from playwright.sync_api import expect

    endpoints = {"authorization": f"{ISSUER}/authorize", "token": "https://token.example.test/token",
                 "identity": "https://identity.example.test/profile", "revocation": "https://token.example.test/revoke"}
    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.add_init_script("window.open = (...args) => { window.opened = args; return null; };")
        skills = open_skills(page, extensions=[CALENDAR])
        listed = status()
        listed["services"][0].update(clients=[], endpoints=endpoints)
        skills.answers[STATUS] = listed
        skills.answers[SIGN_IN] = {"ok": True, "client": {**CLIENT, "label": "Work (IT-supplied)"}}
        _meta, group = _open_sheet(page)
        group.get_by_role("button", name="Add account").click()
        group.get_by_role("button", name="Add a client…").click()
        form = group.locator("[data-sign-in-client-form]")
        binding = form.locator("[data-sign-in-binding]")
        for address in (ISSUER, *endpoints.values()):
            expect(binding).to_contain_text(address)
        form.locator('[data-client-field="label"]').fill("Work (IT-supplied)")
        form.locator('[data-client-field="client_id"]').fill("fixture-it-client")
        form.get_by_role("button", name="Save").click()
        expect(group.locator('[data-sign-in-result="clients"]')).to_have_text("Saved Work (IT-supplied).")
        assert skills.posted(SIGN_IN) == [{"action": "client_save", "issuer": ISSUER, "endpoints": endpoints,
                                          "client": {"label": "Work (IT-supplied)", "client_id": "fixture-it-client", "return": "loopback"}}]
        assert page.evaluate("window.opened === undefined")
    finally:
        if skills is not None:
            skills.close(browser)
        else:
            browser.close()


def test_a_client_the_host_registers_names_its_provider_and_keeps_the_note_for_a_loopback_return(skills_browser):
    from playwright.sync_api import expect

    saved = {**CLIENT, "id": "c-9", "label": "Registered by Ouroboros", "source": "registered", "return": f"exact:{RETURN}",
             "registration": "g" * 64}
    replies = {"begin": {"ok": True, "url": f"{ISSUER}/authorize?state=dcr", "redirect_uri": RETURN,
                         "client": {"id": "c-9", "label": "Registered by Ouroboros"}, "scope": "openid email", "attempt_id": "attempt-dcr"},
               "client_remove": {"ok": True, "removed": False}}
    browser = skills_browser.chromium.launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.add_init_script("window.open = (...args) => { window.opened = args; return null; };")
        skills = open_skills(page, extensions=[CALENDAR])
        listed = status()
        listed["services"][0]["clients"] = []
        skills.answers[STATUS] = listed
        skills.answers[SIGN_IN] = lambda request: (200, replies[request.post_data_json["action"]])
        _meta, group = _open_sheet(page)
        group.get_by_role("button", name="Add account").click()
        editor = group.locator('[data-sign-in-editor="sign_in"]')
        expect(editor).to_contain_text(f"This asks Google ({ISSUER}) to register a client")  # the provider, by the address it will be asked at
        editor.get_by_role("button", name="Register a client and sign in").click()
        expect(group.locator('[data-sign-in-editor="waiting"]')).to_be_visible()
        assert skills.posted(SIGN_IN) == [{"action": "begin", "skill": "calendar", "service": "google", "optional_scopes": [],
                                          "scope_words": ["openid", "email"], "login_hint": "", "register_client": True, "issuer": ISSUER}]

        # Once saved, the client returns to the loopback address; a page elsewhere still explains that, and Remove says what it did.
        listed = status()
        listed["services"][0]["clients"] = [saved]
        skills.answers[STATUS] = listed
        _meta, group = _open_sheet(page)
        group.get_by_role("button", name="Add account").click()
        note = group.locator("[data-sign-in-return-note]")
        expect(note).to_contain_text(f"Google sends you back to {RETURN}; this client must list exactly this address.")
        expect(note).to_contain_text("comes back to the computer where Ouroboros runs")
        group.get_by_role("button", name="Clients…").click()
        group.get_by_role("button", name="Remove…").click()
        group.get_by_role("button", name="Remove", exact=True).click()
        expect(group.locator('[data-sign-in-result="clients"]')).to_contain_text("was already gone, so this removed nothing")
    finally:
        if skills is not None:
            skills.close(browser)
        else:
            browser.close()


def test_revoke_asks_in_place_first_and_names_the_sibling_sign_in(skills_browser):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        skills = open_skills(page, extensions=[CALENDAR])
        skills.answers[STATUS] = status([account("a-1", "owner@example.test", "2026-11-14T10:00:00+00:00",
                                                 siblings=[{"skill": "mail", "service": "google"}])])
        skills.answers[SIGN_IN] = {"ok": True, "revoked": True}
        _meta, group = _open_sheet(page)
        group.get_by_role("button", name="More for owner@example.test").click()
        page.get_by_role("menuitem", name="Revoke at Google…").click()
        confirm = group.locator('[data-sign-in-confirm="a-1"]')
        expect(confirm).to_contain_text("may also end other sign-ins made with the same client for owner@example.test: "
                                        "mail (owner@example.test)")
        expect(confirm).to_contain_text("Skills that keep their own sign-in may lose access too.")
        assert skills.posted(SIGN_IN) == []  # asked first: nothing is revoked yet
        capture(page, "chromium", 1280, "revoke-confirm")

        skills.answers[STATUS] = status()
        confirm.get_by_role("button", name="Revoke at Google").click()
        expect(group.locator('[data-sign-in-result=""]')).to_have_text("Google confirmed the revocation.")
        assert skills.posted(SIGN_IN) == [{"action": "revoke", "skill": "calendar", "service": "google", "account": "a-1", "consent": CONSENT}]
        expect(group.locator("[data-sign-in-row]")).to_have_count(0)
    finally:
        skills_page = locals().get("skills")
        if skills_page is not None:
            skills_page.close(browser)
        else:
            browser.close()

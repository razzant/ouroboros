"""Permissions and checks against a real server: Access and the result review save themselves, keep the truth about
the restart, and the Access levels say what each does to reviews."""

from __future__ import annotations

import json
import os
import pathlib
import time
from datetime import datetime, timezone
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest

from tests import test_ui_smoke_playwright as ui_smoke
from tests.settings_browser_helpers import commit_row, open_settings, setting_row, settings_page

direct_server_with_data = ui_smoke.direct_server_with_data

META = "() => document.querySelector('#page-settings [data-setting-row=\"{row}\"] [data-setting-meta]').textContent"


def _meta_says(page, row, words):
    page.wait_for_function(f"words => ({META.format(row=row)})().includes(words)", arg=words)
    return setting_row(page, row).locator("[data-setting-meta]").inner_text()


def _open_permissions(page, url):
    page.goto(url + "/#settings", wait_until="domcontentloaded")
    level = open_settings(page, "permissions")
    # The page reads the settings when it is shown: the chosen Access level is drawn from that read.
    level.locator('[data-setting-row="OUROBOROS_RUNTIME_MODE"] [role="radio"][aria-checked="true"]').wait_for(state="visible")
    return level


@pytest.mark.ui_browser
def test_access_and_review_round_trip_without_losing_restart_truth(direct_server_with_data):  # noqa: F811
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(fixture["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        try:
            _open_permissions(page, fixture["url"])
            access = setting_row(page, "OUROBOROS_RUNTIME_MODE")
            # Raising Access asks first, then writes at once through its owner endpoint: no page-wide Save.
            access.locator('[data-setting-value="cyber_pro"]').click()
            page.locator('[data-confirm-ok]').click()
            said = _meta_says(page, "OUROBOROS_RUNTIME_MODE", "after restart: Cyber Pro")
            assert said == "Running: Light · after restart: Cyber Pro", said  # the level in force is the server's own answer
            running = page.evaluate("async () => (await (await fetch('/api/settings')).json())._meta.restart_state.running")
            assert running["OUROBOROS_RUNTIME_MODE"] == "light", running
            # The server's receipt and timing, word for word: a restart row says no "was" (its meta says both values)
            # and Access offers no Undo (choosing again may ask); a delayed strip says what it replaced, with Undo.
            assert access.locator("[data-setting-result]").inner_text() == "Saved · applies after restart"
            assert commit_row(page, "OUROBOROS_TASK_REVIEW_MODE", "required") == "Saved · applies to new tasks · was Auto · Undo"
            stored = json.loads((fixture["data_dir"] / "settings.json").read_text(encoding="utf-8"))
            assert stored["OUROBOROS_RUNTIME_MODE"] == "cyber_pro"
            assert stored["OUROBOROS_TASK_REVIEW_MODE"] == "required"

            # A reload keeps both facts: the saved level is chosen, the level in force is named beside it.
            _open_permissions(page, fixture["url"])
            access = setting_row(page, "OUROBOROS_RUNTIME_MODE")
            assert access.locator('[role="radio"][aria-checked="true"]').get_attribute("data-setting-value") == "cyber_pro"
            assert _meta_says(page, "OUROBOROS_RUNTIME_MODE", "Running:") == "Running: Light · after restart: Cyber Pro"
            marks = dict(zip(access.locator(".settings-choice").evaluate_all("nodes => nodes.map((n) => n.dataset.settingValue)"),
                             access.locator("[data-setting-mark]").all_inner_texts()))
            assert marks == {"light": "Now", "advanced": "", "pro": "", "cyber_pro": "After restart"}
            review = setting_row(page, "OUROBOROS_TASK_REVIEW_MODE")
            assert review.locator('[data-setting-value="required"]').get_attribute("aria-pressed") == "true"
            assert page.locator('[data-setting-row="OUROBOROS_REVIEW_ENFORCEMENT"]').count() == 0  # the setting is retired
            access.scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / "settings-policy-reload-desktop.png"))
        finally:
            browser.close()


@pytest.mark.ui_browser
def test_mobile_onboarding_keeps_all_four_access_choices_usable(direct_server_with_data):  # noqa: F811
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(fixture["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        try:
            page.goto(fixture["url"] + "/onboarding", wait_until="domcontentloaded")
            page.locator("#next-btn").click()
            for field in page.locator("[data-model-role-model]").all():
                if not field.input_value():
                    field.fill("mock-model")
            page.locator("#next-btn").click()
            choices = page.locator("[data-runtime-mode]")
            choices.last.wait_for(state="visible")
            assert choices.count() == 4
            geometry = choices.evaluate_all("nodes => nodes.map(n => {const r=n.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width};})")
            assert max(row["x"] for row in geometry) - min(row["x"] for row in geometry) < 2
            assert all(row["w"] > 250 for row in geometry), geometry
            page.locator('[data-runtime-mode="cyber_pro"]').click()
            assert page.locator('[data-runtime-mode="cyber_pro"]').get_attribute("aria-pressed") == "true"
            assert page.locator("[data-review-mode]").count() == 0  # the step chooses Access only
            page.screenshot(path=str(evidence / "onboarding-cyber-mobile.png"), full_page=True)
        finally:
            browser.close()


@pytest.mark.ui_browser
def test_cyber_pro_says_what_it_does_to_reviews_and_lowering_it_asks_nothing(direct_server_with_data):  # noqa: F811
    """A leftover Blocking value in the file draws nothing; the Cyber Pro level says what it does to reviews in its own
    line, and a lower Access saved for the next start is a distinct fact from the level in force."""
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(fixture["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    settings_path = fixture["data_dir"] / "settings.json"
    fixture["stop_server"]()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(OUROBOROS_RUNTIME_MODE="cyber_pro", OUROBOROS_REVIEW_ENFORCEMENT="blocking")
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    fixture["start_server"]()  # this process boots under Cyber Pro
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 390)):
                page = browser.new_page(viewport={"width": width, "height": 1000})
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
                _open_permissions(page, fixture["url"])
                assert page.evaluate("document.documentElement.dataset.theme") == theme
                access = setting_row(page, "OUROBOROS_RUNTIME_MODE")
                assert page.locator('[data-setting-row="OUROBOROS_REVIEW_ENFORCEMENT"]').count() == 0
                # The level itself says what it does to reviews, in its own line.
                assert access.locator('[data-setting-value="cyber_pro"] .settings-choice-note').inner_text() == \
                    "Full say over itself and this computer. Asks for reviews when it judges."
                assert access.locator('[data-setting-value="pro"] .settings-choice-note').inner_text() == \
                    "May change its protected core. Asks for a review before committing new code."
                assert "nothing blocks" not in page.locator('[data-settings-page="permissions"]').inner_text()
                access.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f"settings-cyber-{theme}-{width}.png"))
                # A lower Access saved for the next start is a distinct fact from the level in force: lowering asks
                # nothing and writes at once.
                pro = access.locator('[data-setting-value="pro"]')
                pro.focus()
                page.keyboard.press("Enter")
                assert _meta_says(page, "OUROBOROS_RUNTIME_MODE", "after restart") == "Running: Cyber Pro · after restart: Pro"
                assert access.locator("[data-setting-result]").inner_text() == "Saved · applies after restart"
                assert not page.locator('[data-confirm-ok]').is_visible()  # lowering Access is never asked about
                page.wait_for_function("() => document.activeElement?.dataset.settingValue === 'pro'")  # focus stays put
                assert json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_RUNTIME_MODE"] == "pro"
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(evidence / f"settings-cyber-saved-pro-{theme}-{width}.png"))
                # Back to the state this loop starts from: raising Access asks first.
                access.locator('[data-setting-value="cyber_pro"]').click()
                page.locator('[data-confirm-ok]').click()
                _meta_says(page, "OUROBOROS_RUNTIME_MODE", "How far it reaches")  # nothing waits for a restart any more
                page.close()
        finally:
            browser.close()
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["OUROBOROS_RUNTIME_MODE"] == "cyber_pro"


def _seed_reset_note(data_dir):
    from ouroboros.projects_registry import create_project

    work = data_dir.parent / "reset-working-folder"
    work.mkdir()
    (work / "example.txt").write_bytes(b"A synthetic working file\n")
    # A shelf on its own cannot prove which working folders this profile owns.
    assert create_project(data_dir, "alpha", name="Alpha project", working_dir=str(work))["created"]
    note = data_dir / "projects" / "alpha" / "knowledge" / "one.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# A single synthetic note\n", encoding="utf-8")


@pytest.fixture()
def builtin_skills_only(monkeypatch):
    """Launch input, so it outlives the Reset that deletes ``settings.json``: the case's server starts with a source
    list naming only the built-in skills (the form tests/test_onboarding_consciousness_browser.py ``OFFLINE`` uses), so
    the successor's Setup has no hub row to read or fetch. A Reset removes only what the document mirrored into the
    environment (``settings_integrity.drop_projection``), not an operator's own variable. No provider is added: it
    stays only in the document, and the successor still asks for Setup."""
    original = ui_smoke.isolated_environment
    only = json.dumps([{"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted",
                        "confirmed": True}])
    monkeypatch.setattr(ui_smoke, "isolated_environment",
                        lambda *args, **kwargs: {**original(*args, **kwargs), "OUROBOROS_SKILL_SOURCES": only})


def _path(event) -> str:
    return urlsplit(event.url).path


def _copy(root: pathlib.Path):
    """The serving installation's primary state copy through the reader boot and Setup share
    (``supervisor.state.read_state_copy``), never written: its object once it reads whole, else ``None``."""
    from supervisor.state import read_state_copy

    status, copy, _detail = read_state_copy(root / "state" / "state.json")
    return copy if status == "ok" else None


def _eventually(read, what: str, timeout: float = 60.0):
    """``read()``'s first truthy answer while the serving installation writes. A read met between two of its
    writes (on Windows a file being replaced refuses a reader for a moment) is read again; the clock is only the
    transport bound, and a source that stays unreadable fails here, never passing as an answer."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if found := read():
                return found
        except (OSError, ValueError):
            pass
        if time.monotonic() > deadline:
            pytest.fail(f"{what}: not observed within {timeout:.0f}s")
        time.sleep(0.25)


def _proven_absent(path: pathlib.Path) -> bool:
    """Whether ``path`` is gone, proven as boot proves absence (``installation_stop.confirm_absent``): a missing
    entry below a real directory. An entry this process cannot examine raises, so unknown is never absent."""
    from ouroboros.installation_stop import confirm_absent

    try:
        os.lstat(path)
    except FileNotFoundError:
        confirm_absent(path)
        return True
    return False


def _history(address) -> list:
    """The note's rows in its journal with their claims commits, newest first (``knowledge_versions.NoteHistory``
    with ``commits``), read to the journal's beginning: a gap (a line that is no JSON, a journal not read whole)
    fails, so an unreadable history never reads as an empty one."""
    from ouroboros import knowledge_versions

    history = knowledge_versions.NoteHistory(address, commits=True)
    rows = [row for _offset, row in history]
    assert not history.gaps(), history.gaps()
    return rows


def _seed_published(root: pathlib.Path, address, identity: str) -> None:
    """The seed note as its one writer publishes it for the initialization ``identity`` (``own_time_seed.settle``):
    its exact bytes, the wake claim committed for it, and a journal holding exactly that capture and that commit."""
    from ouroboros import knowledge, knowledge_versions
    from ouroboros import own_time_seed as seed

    assert address.path.read_bytes() == seed.SEED_NOTE.encode("utf-8")
    assert knowledge.active_claims(root).get("wake") == [address]
    rows = _history(address)
    assert len(rows) == 2, rows
    commit, capture = rows
    assert knowledge_versions.is_claim_commit(commit) and commit["active_claims"] == ["wake"]
    assert not knowledge_versions.is_claim_commit(capture)
    assert capture["writer_input_ref"]["initialization_id"] == identity
    for row in rows:
        assert (row["writer"], row["new_sha256"]) == (knowledge.SEED_WRITER, seed.SEED_REVISION), row


def _the_owners_history(page, url: str, root: pathlib.Path, own_time) -> str:
    """Before the Reset, this installation's own history, read and never written; returns its state's identity. A
    state exists, so Setup has no Consciousness step and no first state is due. The owner's Start publishes this
    initialization's note with its committed wake claim, the owner's Stop is saved as the owner's, Main is the bound
    owner chat, and the owner's last command is my human's last word there."""
    from ouroboros import dialogue_store
    from ouroboros import own_time_seed as seed
    from ouroboros.consciousness import OWNER_STOP, STOP_SOURCE_KEY
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.store import close_connections_under
    from supervisor.state_initialization import first_state_pending

    identity = _eventually(lambda: _copy(root), "the state before the Reset")["initialization_id"]
    offer = page.request.get(url + "/api/onboarding/bootstrap").json()["bootstrap"]
    assert offer["consciousnessOffer"]["offered"] is False and "consciousness" not in offer["stepOrder"]
    assert first_state_pending(root) is False
    assert page.request.post(url + "/api/command", data={"cmd": "/bg start"}).ok
    _eventually(lambda: (_copy(root) or {}).get(seed.STATE_KEY) == seed.WRITTEN,
                "the owner's Start publishing its note")
    _seed_published(root, own_time, identity)
    assert page.request.post(url + "/api/command", data={"cmd": "/bg stop"}).ok

    def stopped():
        copy = _copy(root) or {}
        return copy if copy.get(STOP_SOURCE_KEY) == OWNER_STOP else None

    state = _eventually(stopped, "the owner's saved Stop")
    assert state.get("bg_consciousness_enabled") is False and state.get("owner_chat_id") == WEB_UI_CHAT_ID
    try:
        said = dialogue_store.last_inbound(root, WEB_UI_CHAT_ID)
    finally:
        close_connections_under(root)  # no handle of this process stays in the profile the Reset clears
    assert said is not None and (said.row["direction"], said.row["text"]) == ("in", "/bg stop"), said
    return identity


def _the_successor_before_setup(page, url: str, root: pathlib.Path, own_time, accepted: dict, earlier: str) -> None:
    """After the Reset, before Setup. This case's provider was only in the settings document the Reset deletes, so
    the successor answers with Setup (200; 204, or a held Reset, fails with its diagnostic) and with the Reset's own
    receipt: from the Reset's ordinary producer, the one word that it completed. The profile it left is then only
    observed: the earlier note, its journal rows and its wake claim are gone; the Reset's pending witness names its
    identity; both state copies are proven missing; and Setup's reader of a first state to come
    (``state_initialization.first_state_pending``), asked from this process, agrees with the page that offers the
    Consciousness step. Asking writes nothing.

    On disk the copies beside that witness are only missing. What the predicate makes of a copy that is there is shown
    by answers supplied to its copy reader in this process alone: reader inputs, not files, events or requests to the
    server. An unreadable copy is refused and one with the Reset's own identity is adopted, so no first state is due;
    a copy with the earlier identity, read from the first profile before the Reset, is the one boot sets aside, so a
    first state is still due."""
    from ouroboros import knowledge
    from supervisor.state import read_state_copy
    from supervisor.state_initialization import first_state_pending, pending_reset_identity

    ready = page.request.get(url + "/api/onboarding", max_redirects=0)
    assert ready.status != 204, "Unexpected provider readiness after clearing a file-only mock configuration"
    code = ready.json().get("reason_code") if ready.status == 409 else None
    if code == "reset_interrupted":
        # The refusal itself is tests/test_boot_readiness.py
        # ``test_interrupted_reset_serves_recovery_without_opening_the_old_profile``; here it is only reported.
        # Read the existing outcome, without requesting another Reset.
        response = page.request.get(url + "/api/reset", max_redirects=0, timeout=5_000)
        detail = response.json() if response.ok else {}
        if detail.get("reset_id") == accepted["reset_id"] and detail.get("status") == "interrupted":
            pytest.fail(f"Successor held the accepted Reset: {str(detail.get('error', 'unknown'))[:600]}")
        pytest.fail(f"Successor held Reset; its receipt was not confirmed (HTTP {response.status}).")
    assert ready.status == 200, f"Successor readiness: HTTP {ready.status}, reason={code}"
    # Confirm the accepted operation at the real successor before attempting Setup; About must still obtain and
    # render the receipt by itself.
    outcome = page.request.get(url + "/api/reset", max_redirects=0, timeout=5_000)
    assert outcome.ok, f"Successor Reset status: HTTP {outcome.status}"
    completed = outcome.json()
    assert completed.get("reset_id") == accepted["reset_id"] and completed.get("status") == "complete"
    assert "memory" in completed["deleted"]
    assert _proven_absent(own_time.path) and not _history(own_time) and "wake" not in knowledge.active_claims(root)
    witness = (root / "state" / "state.initialized.json").read_bytes()
    copies = [root / "state" / name for name in ("state.json", "state.last_good.json")]
    assert pending_reset_identity(root) == accepted["reset_id"]
    assert [read_state_copy(path)[0] for path in copies] == ["missing", "missing"]
    assert first_state_pending(root) is True
    due = page.request.get(url + "/api/onboarding/bootstrap").json()["bootstrap"]
    assert (due["consciousnessOffer"]["offered"], due["consciousnessOffer"]["enabled"]) == (True, None)
    assert "consciousness" in due["stepOrder"]
    for observed, pending in ((("unreadable", None, ""), False),
                              (("ok", {"initialization_id": accepted["reset_id"]}, ""), False),
                              (("ok", {"initialization_id": earlier}, ""), True)):
        with patch("supervisor.state.read_state_copy", return_value=observed):
            assert first_state_pending(root) is pending, observed
    assert (root / "state" / "state.initialized.json").read_bytes() == witness, "asking wrote the witness"
    assert [read_state_copy(path)[0] for path in copies] == ["missing", "missing"], "asking wrote a state copy"


def _setup_through_the_wizard(page, mock_url: str) -> None:
    """The successor's Setup, driven through its form to Start with the offered Consciousness switch left on."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    iframe = "#onboarding-overlay iframe.onboarding-frame"
    page.locator(iframe).wait_for(state="visible", timeout=20_000)
    wizard = page.frame_locator(iframe)
    wizard.locator('[data-collapse="api-access"] > summary').click(timeout=20_000)
    wizard.locator('[data-provider-card="compatible"] [data-provider-toggle]').click()
    wizard.locator("#s-openai-compatible-base-url").fill(mock_url)
    wizard.locator("#next-btn").click()
    wizard.locator('[data-model-role="main"] [data-model-role-model]').fill("openai-compatible::mock-model")
    previews, completes, failures = [], [], []

    def observed(request):
        endpoint = _path(request)
        if request.method == "POST" and endpoint == "/api/onboarding/subagents/preview":
            previews.append(request)
        if request.method == "POST" and endpoint == "/api/onboarding/complete":
            completes.append(request)

    def failed(request):
        endpoint = _path(request)
        if endpoint in {"/api/onboarding/subagents/preview", "/api/onboarding/complete"}:
            failures.append((endpoint, str(request.failure or "unknown")[:120]))

    def wizard_diagnostic():
        message = page.evaluate("""() => {
            const frame = document.querySelector('#onboarding-overlay iframe.onboarding-frame');
            const status = frame?.contentDocument?.querySelector('.wizard-error');
            return status?.textContent?.trim().slice(0, 300) || '';
        }""")
        return f"preview POSTs={len(previews)}, Start POSTs={len(completes)}, failures={failures[-3:]}, wizard={message!r}"

    page.on("request", observed)
    page.on("requestfailed", failed)
    try:
        steps_left = wizard.locator(".wizard-step").count() - 2
        assert steps_left >= 1
        for _ in range(steps_left - 1):
            wizard.locator("#next-btn").click()
        before_summary = tuple(previews)
        # The last Continue starts a refresh for the actual current draft before painting Summary.
        with page.expect_request_finished(
            lambda r: r in previews and r not in before_summary, timeout=20_000
        ) as preview:
            wizard.locator("#next-btn").click()
        preview_response = preview.value.response()
        assert preview_response is not None and preview_response.ok, wizard_diagnostic()
        assert preview.value == previews[-1], "A newer Setup preview was still pending"
        assert preview_response.body()
        wizard.locator(".summary-card").first.wait_for(timeout=10_000)
        assert not completes, wizard_diagnostic()
        with page.expect_response(
            lambda r: r.request.method == "POST" and _path(r) == "/api/onboarding/complete", timeout=30_000
        ) as configured:
            wizard.locator("#next-btn").click()
        assert configured.value.ok, f"Start HTTP {configured.value.status}; {wizard_diagnostic()}"
        assert len(completes) == 1, wizard_diagnostic()
        assert completes[0].post_data_json.get("consciousness") is True, "Setup's Start did not send the switch on"
    except PlaywrightTimeout:
        pytest.fail(f"Setup did not complete: {wizard_diagnostic()}", pytrace=False)
    finally:
        page.remove_listener("request", observed)
        page.remove_listener("requestfailed", failed)


def _the_profile_setup_made(root: pathlib.Path, own_time, identity: str, earlier: str) -> None:
    """After Setup's Start: the first state is born from the Reset's identity, on by Setup's choice and with no Stop
    carried over, and once its witness completes no first state is due. Its own note is published anew for this
    identity with its committed wake claim. The Runtime section's setup facts read the Reset's origin and what it
    deletes, and neither the earlier owner chat binding nor my human's earlier words: nothing is bound and my human
    has not written. A greeting of mine may already stand, from the first meeting this Start opens."""
    from ouroboros import own_time_seed as seed
    from ouroboros.consciousness import STOP_SOURCE_KEY
    from ouroboros.context_first_hour import setup_fact
    from ouroboros.store import close_connections_under
    from supervisor.first_meeting import FIRST_MEETING_KEY
    from supervisor.state_initialization import authority_reason, first_state_pending

    state = _eventually(lambda: _copy(root), "the successor's first state")
    assert state["initialization_id"] == identity != earlier
    assert state.get("bg_consciousness_enabled") is True and state.get(STOP_SOURCE_KEY) is None
    pressed = (state.get(FIRST_MEETING_KEY) or {}).get("pressed_at")  # the press of Start it was born owing a meeting
    assert pressed, state.get(FIRST_MEETING_KEY)
    _eventually(lambda: not authority_reason(root, identity), "the completion of the Reset identity's initialization")
    assert first_state_pending(root) is False
    _eventually(lambda: (_copy(root) or {}).get(seed.STATE_KEY) == seed.WRITTEN, "the new profile's own note")
    _seed_published(root, own_time, identity)

    def setup_lines():
        fact = setup_fact(root)
        lines = ("finished_at", "state", "owner_chat", "consciousness", "exchange")
        return None if any(fact[line].startswith("unknown") for line in lines) else fact

    try:
        fact = _eventually(setup_lines, "the Runtime section's setup facts, each read from its source")
    finally:
        close_connections_under(root)  # this process's own store handles, before the server stops
    minute = datetime.fromisoformat(pressed).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    assert fact["finished_at"] == minute  # the setup Finish this state was born from, as the settings document holds it
    assert fact["state"].startswith("owner_reset: a new state born")
    assert "a Reset deletes memory, chats" in fact["state"]
    assert fact["consciousness"] == "on"
    assert fact["owner_chat"] == "none bound yet: no owner message has bound one"
    assert fact["exchange"] in ("my human and I have not written to each other yet",
                                "I have written to my human; they have not written to me yet"), fact["exchange"]


@pytest.mark.ui_browser
@pytest.mark.usefixtures("builtin_skills_only")
@pytest.mark.parametrize("direct_server_with_data", [_seed_reset_note], indirect=True)
def test_reset_from_about_shows_the_successors_completed_receipt_in_the_same_tab(direct_server_with_data):
    """One confirmed Reset through the ordinary isolated server, its successor and the same tab, in the order it
    happens: this installation's history before it, the successor before Setup, Setup's Start, the profile that
    Start makes from the Reset's identity, and About showing the Reset's receipt in the same tab. The test acts only
    through the server's own doors (the owner's commands, About's Reset, Setup's Start); its direct reads of the
    profile write nothing, stand in for none of the Reset's records and race none of its writers."""
    from playwright.sync_api import expect, sync_playwright

    from ouroboros import knowledge
    from ouroboros import own_time_seed as seed

    fixture = direct_server_with_data
    url, root = fixture["url"], pathlib.Path(fixture["data_dir"]).resolve()
    mock_url = json.loads((root / "settings.json").read_text(encoding="utf-8"))["OPENAI_COMPATIBLE_BASE_URL"]
    assert urlsplit(url).hostname == urlsplit(mock_url).hostname == "127.0.0.1"
    # The provider lives only in the settings document the Reset deletes, never in launch input: the successor
    # can only ask for Setup.
    assert "OPENAI_COMPATIBLE_BASE_URL" not in fixture["env"]
    own_time = knowledge.resolve_knowledge_address(root, seed.TOPIC, "global")
    note = pathlib.Path("alpha/knowledge/one.md")
    original = (root / "projects" / note).read_bytes()
    meta = 'meta[name="ouroboros-chat-preview-scope"]'
    posts, reads = [], []

    def observe_request(request):
        if request.method == "POST" and _path(request) in {"/api/reset", "/api/reset/v1"}:
            posts.append(_path(request))

    def observe_response(response):
        if response.ok and response.request.method == "GET" and _path(response) == "/api/reset":
            reads.append(response)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on("request", observe_request)
        page.on("response", observe_response)
        try:
            page.goto(url + "/#settings", wait_until="domcontentloaded")
            page.locator("#onboarding-overlay").wait_for(state="detached")
            old_scope = page.locator(meta).get_attribute("content")
            assert old_scope
            earlier = _the_owners_history(page, url, root, own_time)

            # The Reset: one confirmed operation in About; its replacement's new document then takes this tab.
            open_settings(page, "computer/about").locator("[data-settings-fold] > summary").click()
            setting_row(page, "computer:reset").locator("[data-reset-all]").click()
            with page.expect_response(
                lambda r: r.ok and r.request.method == "POST" and _path(r) == "/api/reset/v1"
            ) as reply:
                page.get_by_role("dialog").get_by_role("button", name="Delete and restart").click()
            accepted = reply.value.json()
            assert accepted["status"] == "requested" and accepted["restarting"] and accepted["reset_id"]
            page.wait_for_function(
                "([selector, old]) => { const scope = document.querySelector(selector)?.content; return scope && scope !== old; }",
                arg=[meta, old_scope], timeout=60_000)

            _the_successor_before_setup(page, url, root, own_time, accepted, earlier)
            # The launch list reached the successor itself: its Setup offers the built-in source and no hub row.
            offered = page.request.get(url + "/api/onboarding/starter-skills").json()
            assert [row["id"] for row in offered["sources"]] == ["builtin"], offered["sources"]
            _setup_through_the_wizard(page, mock_url)
            _the_profile_setup_made(root, own_time, accepted["reset_id"], earlier)
            page.locator("#onboarding-overlay").wait_for(state="detached", timeout=60_000)

            # About in the same tab: no test-driven reload, second confirmation, navigation or unfolding of its result.
            about = settings_page(page, "computer/about")
            shown = about.locator("[data-reset-notes-path]")
            expect(shown).to_be_visible(timeout=30_000)
            expect(about.locator("[data-setting-result]").last).to_contain_text("Reset complete.")
            folder = pathlib.Path(shown.inner_text().strip()).resolve()
            assert _path(reads[-1]) == "/api/reset"
            receipt = reads[-1].json()
            assert receipt["status"] == "complete" and receipt["reset_id"] == accepted["reset_id"]
            assert folder == pathlib.Path(receipt["projects_set_aside_path"]).resolve()
            assert folder.parent == root / "projects" and folder.name.startswith(".before-reset-")
            assert not (root / "projects" / note).exists() and (folder / note).read_bytes() == original
            assert posts == ["/api/reset/v1"]
            assert (root.parent / "reset-working-folder" / "example.txt").read_bytes() == b"A synthetic working file\n"
            evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(root.parent)))
            evidence.mkdir(parents=True, exist_ok=True)
            shown.scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / "reset-about-same-tab.png"))
        finally:
            browser.close()

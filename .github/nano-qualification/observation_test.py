"""QA overlay, never part of the product: the Settings and first-run states G7 adds for a private compatible server, in real Chromium.

Synthetic data only, written to the product's own screenshot destination (``OUROBOROS_UI_EVIDENCE_DIR``) as each state is
reached: a later red does not hide an earlier observation, and a state that was not reached is absent, never green.

* The product's stubbed-API Settings fixture (``subscription_ui``) serves the real ``web/`` files: every read is a payload built here.
* ONE first-run journey on the product's real server, started here from nothing (a task-owned home and data root, an environment
  built from scratch, a loopback server that names no program), walking the wizard's real Test, Back and Save.

A row records what the page, the server and the files SHOWED, stamped frames included; this file never judges it. What a row must
show is declared once (``STATE_EXPECT``, ``JOURNEY``), read into the plan, and judged by the reader (``qualification_evidence.judge``).
Only what the plan cannot hold is checked here: the words of a note, a Test line, a withheld Test. Failures are published as a node,
a state or step name and a code from ``FAILURE_CODES`` plus ``failed-<state or step>-page.png``: never a message, a log or a directory.

SHAPE: the selectors, sentences, fields and the server's start that follow the product's Settings, wizard and launch are what a change
of those has to re-bind; ``render_github_linux.OBSERVATION_HOOKS`` names each, so a missing one is a render error, never a red run.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import socket
import struct
import subprocess
import sys
import time
import zlib
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import open_settings

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

OUT = Path(os.environ["OUROBOROS_UI_EVIDENCE_DIR"])
REPO = Path(__file__).resolve().parents[1]  # the overlay sits in tests/ of the checkout it observes
KEY_PAGE = "accounts/key/openai-compatible"
# A private name by the product's own rule (``.lan``), and a RESERVED public one (RFC 2606): the product reads nothing from
# the network for the second and holds no fact about it; the browser must not reach it either.
PRIVATE_ADDRESS = "http://lab.lan:8080/v1"
PUBLIC_ADDRESS = "https://provider.example/v1"
PUBLIC_HOST = "provider.example"
CAUTION = "Server type not confirmed: Ouroboros cannot tell whether it keeps the whole history."
RISK = "A server that does not show which program it runs may shorten a long conversation without saying so."
CHECKED = {"type": "unknown", "observed_at": "2026-11-19T10:00:00Z"}
LLAMA = {"type": "llama.cpp", "observed_at": "2026-11-19T10:00:00Z"}
SEEN = {"type": "ollama", "version": "0.40.16", "observed_at": "2026-11-18T10:00:00Z"}
EDITED_ADDRESS = "http://other.lan:8080/v1"  # another private address, typed over the saved one
OLD_OLLAMA = {"type": "ollama", "version": "0.12.5", "observed_at": "2026-11-19T10:00:00Z", "update_to": "0.12.6"}
# What a Test answers when the guard withheld the request: the product's ``ProviderTestResponse.code`` and its own words.
GUARD_REFUSAL = {"ok": False, "code": "ollama_guard_consent", "error": (
    "The server at http://lab.lan:8080 does not show which program it runs, so Ouroboros cannot confirm that it keeps the "
    "whole history; nothing was sent. To allow it, turn on the consent for this connection in Settings.")}
DAY = r"\S+ \d+"
HELD = (rf"^Ollama 0\.40\.16 was seen at this address on {DAY}\. Server type not confirmed at the last check on {DAY}\. "
        r"Nothing is sent there until this key's server identifies itself: let its metadata be read "
        r"\(GET /api/version\) or use another key\.$")
MODEL = "synthetic-model"

# What the page must show in each Settings state, in STATE_FIELDS order (None: recorded, not asserted). Read by the checker
# through the plan: a recorded row that differs is red whatever the test said of itself. SHAPE: the last three words are the UI's
# own (``data-server-state`` unchecked | unconfirmed | identified | update | held | draft, ``data-admission`` consent | prior,
# ``data-consent-offer`` facts | test | ticked).
STATE_FIELDS = ("consent_shown", "consent_ticked", "note_shown", "address", "server_state", "admission", "consent_offer")
STATE_EXPECT = {
    "never-checked": (True, False, True, "private", "unchecked", "", "facts"),
    "unconfirmed": (True, False, True, "private", "unconfirmed", "", "facts"),
    "consent-stands": (True, True, True, "private", "unconfirmed", "consent", None),
    "prior-route": (True, False, True, "private", "unconfirmed", "prior", "facts"),
    "identified-llama-cpp": (False, False, True, "private", "identified", "", None),
    "held-ollama-seen": (False, False, True, "private", "held", "", None),
    "held-with-standing-consent": (True, True, True, "private", "held", "consent", "ticked"),
    "update-needed-ollama": (False, False, True, "private", "update", "", None),
    "saved-address-edited": (False, False, True, "private", "draft", "", None),
    "public-address-no-fact": (False, False, False, "public", None, None, None),
    "private-draft-no-fact": (False, False, False, "private", None, None, None),
    "private-draft-asks-consent": (True, False, False, "private", None, None, "test"),
    "held-with-standing-consent-phone": (True, True, True, "private", "held", "consent", "ticked"),
}
# The first eight states: the stub's ``unidentified`` fact and the note that must show.
STATES = {
    "never-checked": ({"admission": ""}, rf"^{re.escape(RISK)} A request goes to one only with your consent\.$"),
    "unconfirmed": ({"admission": "", "checked": CHECKED},
                    rf"^Server type not confirmed at the last check on {DAY}\. .*A request goes to one only with your consent\.$"),
    "consent-stands": ({"admission": "consent", "checked": CHECKED}, rf"{re.escape(RISK)}$"),
    "prior-route": ({"admission": "prior", "checked": CHECKED},
                    rf"^Server type not confirmed at the last check on {DAY}\. {re.escape(RISK)} "
                    r"This key may send to one: its server is on record answering a request before this check existed\.$"),
    "identified-llama-cpp": ({"admission": "", "checked": LLAMA}, rf"^Identified as llama\.cpp at the last check on {DAY}\.$"),
    "held-ollama-seen": ({"admission": "", "ollama": SEEN, "checked": CHECKED}, HELD),
    "held-with-standing-consent": ({"admission": "consent", "ollama": SEEN, "checked": CHECKED}, HELD),
    # SHAPE (provisional): the backend's own reading of an Ollama older than the native path needs names the version to update
    # to (``update_to``); the key page then says so instead of asking for metadata that cannot help.
    "update-needed-ollama": ({"admission": "", "checked": OLD_OLLAMA},
                             rf"^Identified as Ollama 0\.12\.5 at the last check on {DAY}\. It can silently truncate history: update "
                             r"Ollama to 0\.12\.6 or later\. Nothing is sent until then\.$"),
}
DRAFT_NOTE = r"^This address has not been checked: what was read at the saved one does not apply here\. Test checks it\.$"
assert list(STATES) == list(STATE_EXPECT)[:len(STATES)]
# The Test lines of the key page: the stub's answer, the words the owner reads, the tone they carry.
LINES = {
    "passed": ({"ok": True, "caution": CAUTION}, f"Works. {CAUTION}", "warn"),
    "failed": ({"ok": False, "error": "Invalid key", "caution": CAUTION}, f"Not ready — Invalid key. {CAUTION}", "danger"),
    "plain": ({"ok": True}, "Works", "ok"),
}
# What every Test line of the key page must also show: the consent its request carried (the typed address, nothing else).
LINE_EXPECT = {"consent_sent": ["address"]}
# The first-run journey, in order. Each step: whether it keeps frames, and the facts it must show, OBSERVED on the page, the
# server and the files (a list is the kinds of the consents a request carried: ``address`` is the typed address). The first
# step is the fresh start, read from the running server; Test, Back and Save are the wizard's own controls.
JOURNEY = {
    "address-typed": {"frames": True, "expect": {"files_before_boot": 0, "secret_env": [], "served_secrets": [],
                                                  "address_typed": True, "consent_shown": False, "consent_ticked": False}},
    "unticked-test-refused": {"frames": True, "expect": {
        "refusal_code": "ollama_guard_consent", "refusal_tone": "warn", "refusal_names_reason": True, "chat_posts_seen": 0,
        "consent_sent": [], "settings_writes": 0, "consent_shown": True, "consent_ticked": False, "consent_offer": "test"}},
    "ticked-test-works": {"frames": True, "expect": {
        "result_tone": "warn", "result_has_caution": True, "chat_posts_seen": 1, "consent_sent": ["address"],
        "settings_writes": 0, "consent_shown": True, "consent_ticked": True}},
    "ticked-test-works-phone": {"frames": True, "expect": {"consent_shown": True, "consent_ticked": True}},
    "back-keeps-input": {"frames": True, "expect": {"address_kept": True, "consent_shown": True, "consent_ticked": True}},
    "saved": {"frames": False, "expect": {"status": 200, "consent_sent": ["address"], "address_saved": True,
                                          "consent_saved": "endpoint_id"}},
    "key-page-after": {"frames": True, "expect": {"note_shown": True, "note_names_risk": True, "consent_shown": True,
                                                  "consent_ticked": True}},
}
# What a failure may say: the node, where it was, and one of these. Nothing else leaves the runner.
FAILURE_CODES = ("selector_missing", "text_differs", "classifier_disagrees", "form_not_drawn", "timeout", "assertion", "browser_error", "other")
CODE_BY_TYPE = {"TimeoutError": "timeout", "AssertionError": "assertion", "Error": "browser_error", "Failed": "timeout"}

SETTINGS_ROOT = "#page-settings [data-settings-page]"
WIZARD_ROOT = ".wizard-content"
ADDRESS_FIELD = '[data-connection-field="base_url"]'
# What the owner must read or press, by the name the checker may publish -> where it is drawn (SHAPE).
SETTINGS_HOOKS = {"consent": ".accounts-key-consent", "note": "[data-connection-unidentified]", "address": ADDRESS_FIELD,
                  "save": "[data-editor-save]", "test": "[data-connection-test]", "result": "[data-editor-result]"}
# The wizard's row holds the box, Test and its result, not the key page's note (no saved facts exist there).
WIZARD_HOOKS = {"consent": ".accounts-key-consent", "address": "#s-openai-compatible-base-url",
                "test": "[data-connection-test]", "result": "[data-connection-test-status]"}
# What a first run starts from: a name that looks like a credential, or one the journey needs empty, must hold nothing.
SECRET_NAME = re.compile(r"(_KEY|_TOKEN|_SECRET|_PASSWORD)$")
NEEDED_EMPTY = ("OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED",
                "OUROBOROS_CONNECTIONS")

# Runs in the page. The active level of the scroll container (the last visible match of ``root``) is measured against its own
# client width, the document against the viewport, and every hooked element that is drawn against the viewport and each
# ancestor that clips it (a horizontal overflow other than ``visible``; a vertical ``hidden`` or ``clip``).
MEASURE_JS = """([root, hooks]) => {
    const doc = document.documentElement;
    const levels = [...document.querySelectorAll(root)].filter((el) => !el.hidden && el.getClientRects().length);
    const level = levels.length ? levels[levels.length - 1] : null;
    const out = {viewport: [innerWidth, innerHeight], document: {scroll_width: doc.scrollWidth, inner_width: innerWidth},
                 container: level ? {scroll_width: level.scrollWidth, client_width: level.clientWidth} : null, clipped: []};
    if (!level) return out;
    for (const [name, selector] of Object.entries(hooks)) {
        for (const el of level.querySelectorAll(selector)) {
            if (!el.getClientRects().length) continue;
            const r = el.getBoundingClientRect();
            let clipped = r.width < 1 || r.height < 1;
            let left = 0, right = innerWidth, top = -Infinity, bottom = Infinity;
            for (let up = el.parentElement; up; up = up.parentElement) {
                const style = getComputedStyle(up);
                const box = up.getBoundingClientRect();
                if (style.overflowX !== 'visible') { left = Math.max(left, box.left); right = Math.min(right, box.right); }
                if (style.overflowY === 'hidden' || style.overflowY === 'clip') { top = Math.max(top, box.top); bottom = Math.min(bottom, box.bottom); }
            }
            if (r.left < left - 1 || r.right > right + 1 || r.top < top - 1 || r.bottom > bottom + 1) clipped = true;
            if (el.scrollWidth > el.clientWidth + 1 && getComputedStyle(el).overflowX !== 'visible') clipped = true;
            if (clipped && !out.clipped.includes(name)) out.clipped.push(name);
        }
    }
    return out;
}"""


class Refused(AssertionError):
    """A named observation that did not hold; ``code`` is one of ``FAILURE_CODES``."""

    def __init__(self, code, detail=""):
        assert code in FAILURE_CODES, code
        super().__init__(f"{code} {detail}".strip())
        self.code = code


def need(condition, code, detail=""):
    if not condition:
        raise Refused(code, detail)


def attempt(code, action):
    """Run a Playwright action: whatever fails in it is this code; the product's words stay in pytest's own output."""
    try:
        return action()
    except Refused:
        raise
    except Exception as error:
        raise Refused(code, type(error).__name__) from error


def frame_name(owner, kind):
    return f"{owner}-{kind}.png"


def stamp(png, label):
    """The frame carries its owner in a tEXt chunk, so a copied or reused file is told from the name it is kept under."""
    body = b"qa-owner\0" + label.encode("ascii")
    chunk = struct.pack(">I", len(body)) + b"tEXt" + body + struct.pack(">I", zlib.crc32(b"tEXt" + body))
    return png[:-12] + chunk + png[-12:]  # the 12 bytes of IEND stay last


def norm(address):
    return str(address).rstrip("/")


def consent_kinds(values, address):
    """The kinds of the consents a request carried, each once: ``address`` for the typed address, ``other`` for anything else
    (a request that names its consent under two keys still carries one kind)."""
    return sorted({"address" if norm(value) == norm(address) else "other" for value in values})


def consent_kind(consent, address, endpoint, scope):
    """What a saved consent is, in one word: ``none``, ``text``, ``ids_not_told_apart``, ``scope_id``, ``other_endpoint`` or
    ``endpoint_id``. ``endpoint`` and ``scope`` are the product's own ``endpoint_id`` and ``scope_id`` of that address, handed in:
    this decides on four strings and runs no product code (the self-test runs it on a fabricated address). The product names
    one address twice and never one for the other: the endpoint (userinfo, the whole path, the query) is what a permission to
    send covers; the address id is the broad name of a hold on Ollama and permits nothing, so it is told from the endpoint's,
    and two ids this cannot tell apart would let it pass as one."""
    if not consent:
        return "none"
    if re.fullmatch(r"[0-9a-f]{24}", consent) is None or address in consent:
        return "text"
    if not endpoint or endpoint == scope:
        return "ids_not_told_apart"
    return "endpoint_id" if consent == endpoint else "scope_id" if consent == scope else "other_endpoint"


def first_run_facts(files_before, env, served):
    """What the first run started from, as observed (names only, never a value): the files in the task-owned root before the
    boot, and the names, in the server's environment and in the settings it then served, that hold a credential or a setting
    the journey needs empty."""
    def holding(items):
        return sorted(key for key, value in items if (SECRET_NAME.search(key) or key in NEEDED_EMPTY)
                      and value not in ("", None, False, 0, [], {}))
    return {"files_before_boot": files_before, "secret_env": holding(env.items()), "served_secrets": holding(served.items())}


def _write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / (name + ".partial")
    temp.write_text(json.dumps(body, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    temp.replace(OUT / name)


def _read(name, default):
    try:
        return json.loads((OUT / name).read_text(encoding="ascii"))
    except (OSError, ValueError):
        return default


class Where:
    """Which node is running and where in it. A failure leaves one record (node, state or step, code) and, when the page can
    still be photographed, ``failed-<where>-page.png``; the exception then goes on to pytest unchanged. ``inside()`` records
    while the browser is still open (a browser that was closed first cannot be photographed)."""

    def __init__(self, node):
        self.node, self.at, self.page, self.recorded = node, "setup", None, False

    def __enter__(self):
        return self

    def __exit__(self, _kind, error, _tb):
        if error is not None and not self.recorded:
            self.record(error)
        return False

    @contextmanager
    def inside(self):
        try:
            yield
        except BaseException as error:
            self.record(error)
            raise

    def record(self, error):
        self.recorded = True
        code = getattr(error, "code", None) or CODE_BY_TYPE.get(type(error).__name__, "other")
        row = {"node": self.node, "state": self.at, "code": code if code in FAILURE_CODES else "other", "frame": None}
        if self.page is not None:
            try:
                name = f"failed-{self.at}-page.png"
                data = stamp(self.page.screenshot(animations="disabled"), name[:-4])
                (OUT / name).write_bytes(data)
                row["frame"] = {"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            except Exception:
                pass  # a failure frame is a courtesy; the record stands without it
        _write("observation-failure.json", {"failures": [*_read("observation-failure.json", {}).get("failures", []), row]})


def checkout_identity():
    """The tree this test runs on, read by the test itself: the checker holds it to the planned head."""
    done = subprocess.run(["git", "rev-parse", "HEAD", "HEAD^{tree}"], capture_output=True, text=True, timeout=30, check=True)
    commit, tree = done.stdout.split()
    return {"commit": commit, "tree": tree}


def kind_of(address):
    """Private or public by the product's own rule, on the tree under test: no network, no DNS."""
    try:
        from ouroboros.server_window import private_host
    except ImportError as error:
        raise Refused("classifier_disagrees", "private_host") from error
    return "private" if private_host(address) else "public"


def _shot(page, locator, owner):
    """The viewport and the card as stamped PNGs; their sizes and digests are the observation's receipt (the checker judges them)."""
    files = {}
    for kind, take in (("page", page.screenshot), ("card", locator.screenshot)):
        data = stamp(take(animations="disabled"), f"{owner}-{kind}")
        (OUT / frame_name(owner, kind)).write_bytes(data)
        files[kind] = {"file": frame_name(owner, kind), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return files


def measured(page, root, hooks):
    """The scroll container's and the document's widths and what is clipped, as numbers; the checker judges them."""
    return page.evaluate(MEASURE_JS, [root, hooks])


def settled(locator, shown):
    """Give the page the time Playwright gives an expectation, then say what is drawn: the wait is not a verdict, the answer is."""
    from playwright.sync_api import expect
    try:
        expect(locator).to_be_visible() if shown else expect(locator).to_be_hidden()
    except AssertionError:
        pass
    return locator.is_visible()


def box_state(label, box, shown):
    """The consent box as drawn once the page has had its time to draw it as expected: is it shown, and is it ticked."""
    drawn = settled(label, shown)
    return {"consent_shown": drawn, "consent_ticked": drawn and box.is_checked()}


def _attribute(locator, name):
    """An attribute the page draws on one element, '' when it draws none (recorded; a bounded token or nothing)."""
    try:
        value = locator.first.get_attribute(name, timeout=2000)
    except Exception:
        value = None
    return re.sub(r"[^a-z0-9_-]", "", str(value or "").lower())[:40]


def _words(locator):
    return " ".join(locator.inner_text().split())


def _wizard_connection(page):
    """Reach the built-in compatible row by the two controls a person can press, on entry and after Back."""
    api = page.locator('details[data-collapse="api-access"]')
    api.wait_for(state="visible")
    if api.get_attribute("open") is None:
        api.locator(":scope > summary").click()
    card = api.locator('[data-provider-card="compatible"]')
    toggle = card.locator("[data-provider-toggle]")
    toggle.wait_for(state="visible")
    if toggle.get_attribute("aria-expanded") == "false":
        toggle.click()
    row = card.locator('[data-connection-row][data-builtin="1"]')
    row.locator(ADDRESS_FIELD).wait_for(state="visible")
    return row


@pytest.fixture
def compat_ui(subscription_ui):
    ui, page = subscription_ui, subscription_ui["page"]
    OUT.mkdir(parents=True, exist_ok=True)
    ui["settings"].update(OPENAI_COMPATIBLE_BASE_URL=PRIVATE_ADDRESS, OPENAI_COMPATIBLE_API_KEY="synthetic-mask...",
                          OUROBOROS_CONNECTIONS=[])
    ui["settings"]["_meta"].update(custom_secret_keys=[], tokens={"OUROBOROS_CONNECTIONS": "list-1"},
        effects={key: "next_call" for key in ("OUROBOROS_CONNECTIONS", "OPENAI_COMPATIBLE_BASE_URL",
                                              "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED")})
    facts = {"fact": None, "configured": True, "tests": [], "reply": {"ok": True}, "saves": [], "hosts": []}

    def settings(route):
        entry = {"kind": "openai-compatible", "id": "openai-compatible", "name": "", "builtin": True, "enabled": True,
                 "configured": facts["configured"], "source": "settings" if facts["configured"] else "none",
                 "display": "OpenAI Compatible"}
        if facts["fact"] is not None:
            entry["unidentified"] = facts["fact"]
        ui["settings"]["_meta"]["connections"] = [entry]
        if route.request.method == "GET":
            route.fulfill(json=ui["settings"])
            return
        facts["saves"].append(route.request.post_data_json)
        route.fulfill(json={"status": "saved", "saved": True, "changes": {}})

    def tested(route):
        facts["tests"].append(route.request.post_data_json)
        route.fulfill(json=facts["reply"])

    page.on("request", lambda request: facts["hosts"].append((urlsplit(request.url).hostname or "").lower()))
    # The reserved public name is never reached: a request to it is counted, and refused if it is made.
    page.route(re.compile(r"^https?://provider\.example(:\d+)?/"), lambda route: route.abort())
    page.route("**/api/settings", settings)
    page.route("**/api/providers/test", tested)
    # The same quiet answers the product's own Accounts suite gives the neighbours of a key page.
    page.route("**/api/extensions", lambda route: route.fulfill(json={"skills": [], "live": {"settings_sections": []}}))
    page.route("**/api/claudexor/maintenance/harnesses*", lambda route: route.fulfill(json={"harnesses": []}))
    page.route("**/api/claudexor/credential-profiles/**", lambda route: route.fulfill(json={"ok": True}))
    return {**ui, "facts": facts}


def _key_page(ui, fact, address=PRIVATE_ADDRESS, configured=True):
    page = ui["page"]
    ui["facts"].update(fact=fact, configured=configured)
    ui["settings"]["OPENAI_COMPATIBLE_BASE_URL"] = address if configured else ""
    ui["settings"]["OPENAI_COMPATIBLE_API_KEY"] = "synthetic-mask..." if configured else ""
    ui["errors"].watch(page)  # the page is navigated here (tests/ui_console_errors.py)
    page.goto(ui["url"] + "/#settings", wait_until="domcontentloaded")
    page.reload(wait_until="domcontentloaded")
    level = attempt("selector_missing", lambda: open_settings(page, KEY_PAGE))
    attempt("selector_missing", lambda: level.locator("[data-connection-row]").wait_for(state="visible"))
    return level


def _row(name, page, level, *, note_pattern):
    """What one Settings state SHOWS: the facts, the note, the measure, the frames. ``ok`` says only that the note reads as it should."""
    note, label, box = (level.locator(sel) for sel in ("[data-connection-unidentified]", ".accounts-key-consent",
                                                       "[data-connection-consent]"))
    want = dict(zip(STATE_FIELDS, STATE_EXPECT[name]))
    note_shown, drawn = settled(note, want["note_shown"]), box_state(label, box, want["consent_shown"])
    seen = {**drawn, "note_shown": note_shown, "address": kind_of(level.locator(ADDRESS_FIELD).input_value()),
            "server_state": _attribute(note, "data-server-state") if note_shown else "",
            "admission": _attribute(note, "data-admission") if note_shown else "",
            "consent_offer": _attribute(label, "data-consent-offer") if drawn["consent_shown"] else ""}
    text = _words(note) if note_shown else ""
    return {"state": name, **seen, "note": text, "measure": measured(page, SETTINGS_ROOT, SETTINGS_HOOKS),
            "frames": _shot(page, level.locator("[data-connection-row]"), name),
            "ok": note_pattern is None or re.search(note_pattern, text, re.S) is not None}  # the words: the plan holds no sentence


def test_each_state_of_an_unidentified_compatible_key_in_a_real_browser(compat_ui):
    from playwright.sync_api import expect
    ui, page = compat_ui, compat_ui["page"]
    report = {"engine": "chromium", "browser_version": page.context.browser.version, "checkout": checkout_identity(),
              "states": [], "reserved_host_requests": 0, "complete": False}

    def keep(row):
        report["states"].append(row)
        report["reserved_host_requests"] = ui["facts"]["hosts"].count(PUBLIC_HOST)
        _write("observation.json", report)
        need(row["ok"], "text_differs", row["state"])

    with Where("test_each_state_of_an_unidentified_compatible_key_in_a_real_browser") as where:
        where.page = page
        for name, (fact, note_pattern) in STATES.items():
            where.at = name
            keep(_row(name, page, _key_page(ui, fact), note_pattern=note_pattern))
        # A saved private key whose address is edited: what was read at the saved address does not speak of the typed one,
        # so the page says so and offers no box until a Test of the draft finds a reason to (SHAPE, provisional).
        where.at = "saved-address-edited"
        level = _key_page(ui, {"admission": "consent", "checked": CHECKED})
        attempt("selector_missing", lambda: level.locator(ADDRESS_FIELD).fill(EDITED_ADDRESS))
        keep(_row("saved-address-edited", page, level, note_pattern=DRAFT_NOTE))
        # The control: a saved PUBLIC address (a reserved name) has no fact in the read, so the page draws neither a note nor
        # the box, and the form as before. The product reads nothing from the network for it, and neither may the browser.
        where.at = "public-address-no-fact"
        level = _key_page(ui, None, PUBLIC_ADDRESS)
        attempt("form_not_drawn", lambda: [level.locator(hook).wait_for(state="visible") for hook in (
            ADDRESS_FIELD, "[data-editor-save]", "[data-connection-test]")])
        need(level.locator(ADDRESS_FIELD).input_value() == PUBLIC_ADDRESS, "form_not_drawn")
        # A Test of it (answered by the stub; nothing is sent anywhere) still offers no consent and says nothing of one: the
        # control is observed after it.
        ui["facts"]["reply"] = {"ok": True}
        attempt("selector_missing", lambda: level.locator("[data-connection-test]").click())
        attempt("text_differs", lambda: expect(level.locator("[data-editor-result]")).to_have_text("Works"))
        keep(_row("public-address-no-fact", page, level, note_pattern=None))
        # A genuinely private DRAFT: nothing saved, a private address typed. No fact exists for it and the server has not
        # spoken (no Test yet), so the page offers no box and no note, exactly as for the public control; what tells the two
        # apart is the address (the product's own rule calls it private) and what a Test of it brings out below.
        where.at = "private-draft-no-fact"
        level = _key_page(ui, None, "", configured=False)
        attempt("selector_missing", lambda: level.locator(ADDRESS_FIELD).fill(PRIVATE_ADDRESS))
        keep(_row("private-draft-no-fact", page, level, note_pattern=None))
        # The same draft after a Test the guard withheld for want of a consent: the box is offered, unticked, and the result
        # carries the code. The stub answers with the product's own ``code``; the real refusal is in the journey below.
        where.at = "private-draft-asks-consent"
        ui["facts"]["reply"] = GUARD_REFUSAL
        attempt("selector_missing", lambda: level.locator("[data-connection-test]").click())
        result = level.locator("[data-editor-result]")
        attempt("text_differs", lambda: expect(result).to_have_text(re.compile(r"^Not ready")))
        need(_attribute(result, "data-test-code") == "ollama_guard_consent", "text_differs", "the code of a withheld Test")
        need(_attribute(result, "data-tone") == "warn", "text_differs", "tone of a withheld Test")
        keep(_row("private-draft-asks-consent", page, level, note_pattern=None))
        # The standing consent under a hold on a phone: the longest note and the label on the narrowest page.
        where.at = "held-with-standing-consent-phone"
        page.set_viewport_size({"width": 390, "height": 844})
        fact, note_pattern = STATES["held-with-standing-consent"]
        keep(_row("held-with-standing-consent-phone", page, _key_page(ui, fact), note_pattern=note_pattern))
        need([row["state"] for row in report["states"]] == list(STATE_EXPECT), "other", "states out of order")
        report["complete"] = True
        _write("observation.json", report)


def test_a_ticked_consent_travels_with_test_and_the_caution_shows_beside_the_outcome(compat_ui):
    from playwright.sync_api import expect
    ui, page = compat_ui, compat_ui["page"]
    report = {"engine": "chromium", "checkout": checkout_identity(), "lines": [], "saves_during_test": 0, "complete": False}
    with Where("test_a_ticked_consent_travels_with_test_and_the_caution_shows_beside_the_outcome") as where:
        where.page = page
        where.at = "unconfirmed"
        level = _key_page(ui, {"admission": "", "checked": CHECKED})
        attempt("selector_missing", lambda: level.locator("[data-connection-consent]").check())
        result = level.locator("[data-editor-result]")
        for outcome, (reply, text, tone) in LINES.items():
            ui["facts"]["reply"] = reply
            before = len(ui["facts"]["tests"])
            attempt("selector_missing", lambda: level.locator("[data-connection-test]").click())
            try:
                expect(result).to_have_text(text)  # the wait; what is drawn is read below
            except AssertionError:
                pass
            sent = ui["facts"]["tests"][before:]
            row = {"outcome": outcome, "text": _words(result), "tone": _attribute(result, "data-tone"),
                   "consent_sent": consent_kinds([value for post in sent for value in _consent_values(post)], PRIVATE_ADDRESS),
                   "measure": measured(page, SETTINGS_ROOT, SETTINGS_HOOKS),
                   "frames": _shot(page, level.locator("[data-connection-row]"), f"test-{outcome}")}
            row["ok"] = row["text"] == text and row["tone"] == tone and len(sent) == 1  # the words, the tone and one request
            report["lines"].append(row)
            report["saves_during_test"] = len(ui["facts"]["saves"])
            _write("observation-lines.json", report)
            need(row["ok"], "text_differs", outcome)
        report["complete"] = True
        _write("observation-lines.json", report)


@contextmanager
def _unnamed_server():
    """An inert OpenAI-compatible server on loopback that names no program anywhere: every metadata path answers 404 and
    only the model list and the chat answer. It records each request, so what a Test sent can be counted."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _answer(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            calls.append(("GET", path))
            if path == "/v1/models":
                return self._answer(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
            self._answer(404, {"error": {"message": "not found"}})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            path = self.path.split("?", 1)[0]
            calls.append(("POST", path))
            if path == "/v1/chat/completions":
                return self._answer(200, {"id": "synthetic", "object": "chat.completion", "model": MODEL,
                                          "choices": [{"index": 0, "finish_reason": "stop",
                                                       "message": {"role": "assistant", "content": "OK"}}],
                                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
            self._answer(404, {"error": {"message": "not found"}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _chat_posts(calls):
    return sum(1 for method, path in calls if method == "POST" and path.endswith("/chat/completions"))


def _consent_values(post):
    """What a request carries as a consent: the non-empty values of every field that names it, wherever it sits."""
    found = []

    def walk_value(value, named):
        if isinstance(value, dict):
            for key, inner in value.items():
                walk_value(inner, named or "CONTINUE_UNCONFIRMED" in str(key).upper())
        elif named and value not in ("", None, False):
            found.append(value)

    walk_value(post or {}, False)
    return found


def _get(port, path):
    """One JSON document from a server on loopback (no proxy, no browser)."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", path)
        return json.loads(connection.getresponse().read())
    finally:
        connection.close()


@contextmanager
def first_run_server(root):
    """The product's own server, started here from nothing: a task-owned home and data root that held no file, an environment
    built from scratch (nothing is inherited, so no credential, consent or setting of the runner can reach it) and a free
    loopback port. Yields the address, the settings file it will write and the facts of the start (``first_run_facts``, read
    from the files and from what the running server serves). SHAPE: the launch contract (``server.py --host --port``,
    ``OUROBOROS_*`` roots, ``/api/health`` ready when ``boot.tasks_ready``) is the product's; the render names each of its words."""
    dirs = {name: root / name for name in ("home", "tmp", "data", "worktrees", "projects", "deliverables", "files", "bench", "xdg")}
    xdg = {f"XDG_{kind}_HOME": str(dirs["xdg"] / kind.lower()) for kind in ("CONFIG", "CACHE", "DATA", "STATE")}
    for path in (*dirs.values(), dirs["data"] / "state", *map(Path, xdg.values())):
        path.mkdir(parents=True, exist_ok=True)
    probes = [socket.socket(), socket.socket()]  # the app and its host service, two different free ports
    try:
        for probe in probes:
            probe.bind(("127.0.0.1", 0))
        ports = [probe.getsockname()[1] for probe in probes]
    finally:
        for probe in probes:
            probe.close()
    paths = {"APP_ROOT": root, "REPO_DIR": REPO, "DATA_DIR": dirs["data"], "SETTINGS_PATH": dirs["data"] / "settings.json",
             "PID_FILE": dirs["data"] / "state/server.pid", "PORT_FILE": dirs["data"] / "state/server_port",
             "SUBAGENT_WORKTREE_ROOT": dirs["worktrees"], "SUBAGENT_PROJECTS_ROOT": dirs["projects"],
             "DELIVERABLES_ROOT": dirs["deliverables"], "USER_FILES_ROOT": dirs["files"], "BENCH_RUNS_ROOT": dirs["bench"]}
    env = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ["PATH"]]), "LANG": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "HOME": str(dirs["home"]), "TMPDIR": str(dirs["tmp"]),
           **xdg, **{f"OUROBOROS_{key}": str(value) for key, value in paths.items()},
           "OUROBOROS_SERVER_HOST": "127.0.0.1", "OUROBOROS_SERVER_PORT": str(ports[0]),
           "OUROBOROS_HOST_SERVICE_PORT": str(ports[1]), "OUROBOROS_PYTEST_ACTIVE": "1"}
    files = sum(1 for path in root.rglob("*") if path.is_file())
    with (root / "server.log").open("xb") as log:
        proc = subprocess.Popen([sys.executable, "-B", str(REPO / "server.py"), "--host", "127.0.0.1", "--port", str(ports[0])],
                                cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 240  # the boot of a first run, as long as the earlier lane allowed it
            while True:
                need(proc.poll() is None, "other", "the server exited before it was ready")
                try:
                    if _get(ports[0], "/api/health").get("boot", {}).get("tasks_ready"):
                        break
                except (OSError, ValueError, AttributeError):
                    pass
                need(time.monotonic() < deadline, "timeout", "the server was not ready")
                time.sleep(0.5)
            yield (f"http://127.0.0.1:{ports[0]}", paths["SETTINGS_PATH"],
                   first_run_facts(files, env, _get(ports[0], "/api/settings")))
        finally:  # only the process this started, and only by its own handle
            proc.terminate()
            try:
                proc.wait(timeout=35)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


@pytest.mark.timeout(600)
def test_first_run_from_nothing_asks_then_tests_goes_back_and_saves(tmp_path_factory):
    """FIRST RUN, from nothing: the product's real server is started here over a task-owned home that held no file and an
    environment built from scratch; the wizard is the real one, against a loopback server that names no program. The start is
    READ (the files, the environment, what the server serves). Then the owner's steps: the address typed, Test with nothing ticked
    (refused, nothing sent), the box ticked, Test (works, with its caution), Next and Back, Save, and the key page. The permission
    exists only because the box was clicked. SHAPE (provisional): the box opens after a Test answered with the consent code
    (``data-test-code`` on the result, ``data-consent-offer`` on its label), the typed address is the consent a Test carries."""
    from playwright.sync_api import expect, sync_playwright
    from tests._wizard_walk import walk

    steps, writes = [], []  # the rows so far; every POST the page made to a settings writer
    report = {"engine": "chromium", "checkout": checkout_identity(), "steps": steps, "complete": False}

    with Where("test_first_run_from_nothing_asks_then_tests_goes_back_and_saves") as where:
        with _unnamed_server() as (url, calls), \
                first_run_server(tmp_path_factory.mktemp("first-run")) as (base, settings_file, start), sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_context(viewport={"width": 1360, "height": 900}).new_page()
                where.page = page

                def step(name, locator=None, root=WIZARD_ROOT, hooks=WIZARD_HOOKS, **seen):
                    """Record what one step showed (for a step that keeps frames, its measure and its frames): the reader judges it."""
                    row = {"step": name, **seen}
                    if JOURNEY[name]["frames"]:
                        row.update(measure=measured(page, root, hooks), frames=_shot(page, locator, name))
                    steps.append(row)
                    _write("observation-journey.json", report)

                with where.inside():
                    from ouroboros.llm_ollama_guard import TEST_CAUTION, endpoint_id, scope_id
                    page.on("request", lambda request: writes.append(request.url) if request.method == "POST" and urlsplit(
                        request.url).path in ("/api/settings", "/api/onboarding/complete") else None)
                    report["browser_version"] = browser.version
                    need(kind_of(url) == "private", "classifier_disagrees", "the journey server")
                    where.at = "address-typed"
                    attempt("timeout", lambda: page.goto(base + "/onboarding", wait_until="networkidle"))
                    row = attempt("selector_missing", lambda: _wizard_connection(page))
                    field, label, box = (row.locator(ADDRESS_FIELD), row.locator(".accounts-key-consent"),
                                         row.locator("[data-connection-consent]"))
                    status, test = row.locator("[data-connection-test-status]"), row.locator("[data-connection-test]")

                    def pressed(done):
                        """Press Test, wait for its words, and read what it did: the words, the conversation and settings writes so far,
                        and the kinds of consent its request carried."""
                        with page.expect_response("**/api/providers/test") as sent:
                            attempt("selector_missing", lambda: test.click())
                        try:
                            expect(status).to_have_text(done)
                        except AssertionError:
                            pass
                        return {"result": _words(status), "chat_posts_seen": _chat_posts(calls), "settings_writes": len(writes),
                                "consent_sent": consent_kinds(_consent_values(sent.value.request.post_data_json), url)}

                    attempt("selector_missing", lambda: field.fill(url))
                    step("address-typed", row, **start, address_typed=norm(field.input_value()) == norm(url),
                         **box_state(label, box, False))  # nothing has said a consent matters yet

                    where.at = "unticked-test-refused"
                    seen, shown = pressed(re.compile(r"^Not ready")), box_state(label, box, True)  # the refusal offers the box, unticked
                    step("unticked-test-refused", row, **seen, **shown, refusal_code=_attribute(status, "data-test-code"),
                         refusal_tone=_attribute(status, "data-tone"),
                         refusal_names_reason="does not show which program it runs" in seen["result"],
                         consent_offer=_attribute(label, "data-consent-offer") if shown["consent_shown"] else "")

                    where.at = "ticked-test-works"
                    attempt("selector_missing", lambda: box.check())
                    seen = pressed(f"Works. {TEST_CAUTION}")
                    step("ticked-test-works", row, **seen, **box_state(label, box, True), result_tone=_attribute(status, "data-tone"),
                         result_has_caution=seen["result"] == " ".join(f"Works. {TEST_CAUTION}".split()))
                    where.at = "ticked-test-works-phone"
                    page.set_viewport_size({"width": 390, "height": 844})
                    row.scroll_into_view_if_needed()
                    step("ticked-test-works-phone", row, **box_state(label, box, True))
                    page.set_viewport_size({"width": 1360, "height": 900})

                    # Next and Back, the way the owner would: what was typed and ticked is still there, and is what Save will carry.
                    where.at = "back-keeps-input"
                    walk(page, "accounts", "models")
                    page.wait_for_selector('[data-model-role="main"]')
                    walk(page, "models", "accounts")
                    attempt("selector_missing", lambda: _wizard_connection(page))  # read the remounted draft without changing it
                    step("back-keeps-input", row, address_kept=norm(field.input_value()) == norm(url), **box_state(label, box, True))

                    # Finish the wizard the way the product's own first-run journeys do (tests/test_onboarding_connections_browser.py).
                    where.at = "saved"
                    walk(page, "accounts", "models")
                    page.wait_for_selector('[data-model-role="main"]')
                    for role in ("main", "light"):
                        group = page.locator(f'[data-model-role="{role}"]')
                        options = group.locator("[data-model-role-source] option").evaluate_all("nodes => nodes.map(n => n.value)")
                        chosen = next((value for value in options if value.startswith("api:openai-compatible")), "")
                        need(bool(chosen), "selector_missing", f"{role} source")
                        group.locator("[data-model-role-source]").select_option(chosen)
                        group.locator("[data-model-role-model]").fill(MODEL)
                    page.click("#next-btn")  # Review
                    page.locator('[data-runtime-mode="light"]').click()
                    walk(page, "review_mode", "summary")
                    page.locator("[data-starter-toggle]").click()  # no starter installation: this journey makes no public request
                    page.locator('[data-starter-all="0"]').click()
                    with page.expect_response("**/api/onboarding/complete") as completed:
                        page.click("#next-btn")
                    reply = completed.value.json()
                    need(completed.value.status == 200 and reply.get("ok") is True
                         and type(reply.get("restart_required")) is bool, "other", "completion not confirmed")
                    saved = json.loads(settings_file.read_text(encoding="utf-8"))
                    step("saved", status=completed.value.status,
                         consent_sent=consent_kinds(_consent_values(completed.value.request.post_data_json), url),
                         address_saved=norm(saved.get("OPENAI_COMPATIBLE_BASE_URL", "")) == norm(url),
                         consent_saved=consent_kind(str(saved.get("OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED", "")), url,
                                                    endpoint_id(url), scope_id(url)))

                    where.at = "key-page-after"
                    # A bare browser tab can offer Open now if the chosen Access needs a restart. Read Settings via
                    # that real handoff; do not bypass the restart screen or claim the new runtime mode applied.
                    if reply["restart_required"]:
                        attempt("selector_missing", lambda: page.locator("#open-app-btn").click())
                    page.wait_for_url(base + "/", wait_until="domcontentloaded")
                    level = attempt("selector_missing", lambda: open_settings(page, KEY_PAGE))
                    keyed = level.locator("[data-connection-row]")
                    attempt("selector_missing", lambda: keyed.wait_for(state="visible"))
                    after_note, after_label = level.locator("[data-connection-unidentified]"), level.locator(".accounts-key-consent")
                    note_shown = settled(after_note, True)
                    words = _words(after_note) if note_shown else ""
                    step("key-page-after", keyed, root=SETTINGS_ROOT, hooks=SETTINGS_HOOKS, note=words, note_shown=note_shown,
                         note_names_risk=words.endswith(RISK), server_state=_attribute(after_note, "data-server-state"),
                         **box_state(after_label, level.locator("[data-connection-consent]"), True))
                    need([row["step"] for row in steps] == list(JOURNEY), "other", "steps out of order")
                    report["complete"] = True
                    _write("observation-journey.json", report)
            finally:
                browser.close()

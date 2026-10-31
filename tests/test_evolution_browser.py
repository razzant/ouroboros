"""Dashboard → Evolution in a real browser over a production-shaped history (`/api/evolution-data` mocked).

A burst of early tags, months without one, then a release every few days: one measure is drawn, milestones are dots,
every version a tick, and the chosen version's own row reads beneath. The campaign object never waits for history.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

ROOT = Path(__file__).resolve().parents[1]
STATE = {"sha": "evolution-browser-fixture", "supervisor_ready": True, "active_chat_activities": [], "projects": [], "runtime_mode": "advanced",
         "evolution_enabled": False, "evolution_state": {"status": "disabled", "enabled": False, "detail": "Evolution mode is off."},
         "bg_consciousness_enabled": True, "bg_consciousness_state": {"status": "sleeping", "spent_24h_usd": 1.25, "daily_usd": 10}}


def history(*, refresh="idle", versions=True, error=""):
    """60 versions: twelve in three February days (4.0.0–6.3.0), then 6.10.0–7.9.2 from June to October; four MAJOR
    releases, 24 MINOR ones and one row its author titled (7.6.0): 28 milestones, the server's rule (owner 21.10, A)."""
    rows, start = [], datetime(2026, 2, 16, 10, tzinfo=timezone.utc)
    dates = [start + timedelta(hours=6 * i) for i in range(12)]
    dates += [datetime(2026, 6, 1, 9, tzinfo=timezone.utc) + timedelta(days=2.5 * i) for i in range(48)]
    for i, date in enumerate(dates):
        if i < 12:
            major, minor, patch = 4 + i // 4, i % 4, 0
        else:
            major, step = (6, i - 12) if i < 30 else (7, i - 30)
            minor, patch = step // 3 + (10 if major == 6 else 0), step % 3
        version = f"{major}.{minor}.{patch}" + ("-rc.1" if i in (20, 40) else "")
        level = "pre" if "rc" in version else "major" if (minor, patch) == (0, 0) else "minor" if patch == 0 else "patch"
        titled = version == "7.6.0"
        rows.append({
            "tag": f"v{version}", "aliases": [f"remote-v{version}"] if i < 12 else [], "version": version,
            "date": date.isoformat(), "commit": f"{i:040x}", "level": level, "milestone": titled or level in ("major", "minor"),
            "code_lines": int(7000 * (1.085 ** i)), "bible_bytes": 14000 + 700 * i,
            "system_bytes": 9000 + 1500 * i if i < 30 else 24000 + 200 * (i - 30),
            "release": {"source": "readme" if i >= 12 else "tag", "title": "Memory as one ladder" if titled else "",
                        "kind": "feat" if patch == 0 else "fix",
                        "heading": f"release {version} keeps its own words, long enough to wrap on a phone screen.",
                        "body": "Memory keeps an open conversation verbatim and turns a closed matter into an authored page. " * 4},
        })
    identity = [None] * 14 + [4000 + 400 * i for i in range(46)]
    return {"versions": rows if versions else [], "saved_at": datetime.now(timezone.utc).isoformat(),
            "refresh": {"state": refresh, "started_at": "", "finished_at": "", "error": error, "unmeasured": 0},
            "install": {"identity": {"at_versions": identity if versions else [], "first_at": dates[14].isoformat(),
                                     "latest": {"at": datetime.now(timezone.utc).isoformat(), "chars": 26500}},
                        "scratchpad": {"at_versions": [None] * len(rows) if versions else [], "first_at": "", "latest": None}},
            "checkpoints": [], "generated_at": datetime.now(timezone.utc).isoformat()}


def open_evolution(ui, answer, *, width=1100, height=750, scheme="dark", state=STATE, asked=None):
    page = ui["page"]
    page.set_viewport_size({"width": width, "height": height})
    page.emulate_media(color_scheme=scheme)
    origin = urlparse(ui["url"]).netloc
    page.route("**/*", lambda route: route.fallback() if urlparse(route.request.url).netloc == origin else route.abort())

    def data(route):
        if asked is not None:
            asked.append(parse_qs(urlparse(route.request.url).query))
        route.fulfill(json=answer() if callable(answer) else answer)

    page.route("**/api/evolution-data*", data)
    page.route("**/api/state", lambda route: route.fulfill(json=state() if callable(state) else state))
    page.goto(ui["url"] + "/#dashboard")
    page.locator('[data-dashboard-tab="evolution"]').click()
    page.locator("#evo-campaign-chip").wait_for(state="visible")
    return page


def overflow(page):
    return page.evaluate("""() => {
        const box = document.querySelector('#page-evolution .costs-scroll');
        return {page: document.documentElement.scrollWidth - innerWidth, panel: box.scrollWidth - box.clientWidth};
    }""")


@pytest.mark.parametrize("width", [1100, 390])
def test_one_measure_milestones_and_the_chosen_version_beneath(subscription_ui, width):
    page = open_evolution(subscription_ui, history(), width=width, height=844 if width < 500 else 750)
    detail = page.locator(".evo-detail-title")
    detail.wait_for()
    assert detail.inner_text() == "Release 7.9.2 keeps its own words, long enough to wrap on a phone screen", "the newest is chosen"
    assert page.locator(".evo-ch-rug").count() == 60
    assert page.locator(".evo-ch-rug.is-milestone").count() == 28, "MAJOR and MINOR releases and the titled row"
    assert page.locator(".evo-ch-pin").count() == 28, "every milestone keeps its dot, labelled or not"
    boxes = page.evaluate("""() => [...document.querySelectorAll('.evo-ch-mlabel, .evo-ch-label, .evo-ch-strong')]
        .map((n) => { const b = n.getBBox(); return [b.x, b.y, b.x + b.width, b.y + b.height, n.textContent]; })""")
    named = page.locator(".evo-ch-mlabel").count()
    assert named >= (1 if width > 500 else 0), "the chart names milestones where there is room clear of the line"
    assert page.locator(".evo-ch-label").count() == 1, "the chosen version always names itself"
    clashes = [(a[4], b[4]) for i, a in enumerate(boxes) for b in boxes[i + 1:]
               if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]]
    assert not clashes, clashes
    covered = page.evaluate("""() => { const line = document.querySelector('.evo-ch-line'), out = [];
        const labels = [...document.querySelectorAll('.evo-ch-mlabel')].map((n) => [n.getBBox(), n.textContent]);
        for (let s = 0; s <= line.getTotalLength(); s += 1) { const p = line.getPointAtLength(s);
            for (const [b, text] of labels) if (p.x > b.x && p.x < b.x + b.width && p.y > b.y && p.y < b.y + b.height) out.push(text); }
        return [...new Set(out)]; }""")
    assert covered == [], f"the line runs through these milestone labels: {covered}"
    assert page.locator(".evo-chart text", has_text="no tags").count() == (1 if width > 500 else 0), "the quiet months are a gap"
    assert page.locator("#evo-campaign-chip").inner_text() == "Off"
    assert page.locator("#evo-consciousness").inner_text().startswith("Consciousness sleeping")
    assert page.locator("#evo-consciousness-allowance").inner_text() == "Allowance (24 h): $1.25 / $10.00"
    # The line and its area stop at the axis break; only the dashed bridge crosses the quiet months.
    runs = page.evaluate("""() => document.querySelector('.evo-ch-line').getAttribute('d').split('M').filter(Boolean)
        .map((run) => run.split('L').map((pair) => Number(pair.split(',')[0])))""")
    gap = page.evaluate("""() => { const t = [...document.querySelectorAll('.evo-chart text')].find((n) => n.textContent === 'no tags');
        return t ? Number(t.getAttribute('x')) : null; }""")
    assert len(runs) == 2 and (gap is None or all(max(r) < gap or min(r) > gap for r in runs)), (runs, gap)
    assert overflow(page) == {"page": 0, "panel": 0}
    setup_browser.capture(page, f"evolution-history-{width}")

    hit = page.locator(".evo-chart-hit")
    hit.focus()
    page.keyboard.press("ArrowLeft")
    assert page.locator(".evo-detail-version").inner_text() == "7.9.1"
    for _ in range(4):  # Page Up steps milestones: 7.9.0, 7.8.0, 7.7.0, 7.6.0
        page.keyboard.press("PageUp")
    assert page.locator(".evo-detail-version").inner_text() == "7.6.0"
    assert detail.inner_text() == "Memory as one ladder"
    assert page.locator(".evo-detail-lead").inner_text().startswith("Release 7.6.0 keeps its own words")
    assert hit.get_attribute("aria-valuetext").startswith("7.6.0, ")
    page.locator('.evo-detail-nav [data-step="-1"]').click()
    assert page.locator(".evo-detail-version").inner_text() == "7.5.2", "the buttons step every version, not just milestones"
    more = page.locator("[data-full]")
    if more.is_visible():
        more.click()
        assert "is-full" in page.locator(".evo-detail-body").get_attribute("class")

    page.locator('[data-measure="identity"]').click()
    assert page.locator('[data-measure="identity"]').get_attribute("aria-pressed") == "true"
    assert page.locator('[data-measure="code_lines"]').get_attribute("aria-pressed") == "false"
    assert page.locator(".evo-caption").inner_text().startswith("This installation · Identity: characters of identity.md")
    assert page.locator(".evo-chart .evo-ch-strong").text_content() == "26.5K chars", "the newest journal record ends the line"
    assert overflow(page) == {"page": 0, "panel": 0}
    setup_browser.capture(page, f"evolution-identity-{width}")


def test_history_states_never_hold_the_campaign_or_drop_a_saved_copy(subscription_ui):
    answers = {"now": history(refresh="running", versions=False)}
    asked = []
    page = open_evolution(subscription_ui, lambda: answers["now"], asked=asked)
    page.locator(".evo-empty-line", has_text="first measurement on this install").wait_for()
    assert page.locator(".evo-story-status").inner_text().endswith("Measuring every version…")
    assert page.locator("#evo-start").is_enabled(), "the campaign answers without the history"
    assert page.locator(".evo-story").bounding_box()["height"] < 220, "no empty canvas while the first copy is measured"
    setup_browser.capture(page, "evolution-first-measurement")

    answers["now"] = history(refresh="failed", error="RuntimeError: git cat-file failed (rc=128)")
    page.locator(".evo-detail-title").wait_for(timeout=10000)  # the poll brings the saved copy with the failure
    status = page.locator(".evo-story-status")
    assert "Couldn’t check for new versions" in status.inner_text()
    answers["now"] = history(refresh="running")
    status.locator("[data-retry]").click()
    page.wait_for_function("() => document.querySelector('.evo-story-status').textContent.includes('Checking for new versions')")
    assert asked[-1] == {"force": ["1"]}, "Try again after a failed refresh re-measures"
    assert page.locator(".evo-ch-rug").count() == 60, "the copy on screen stays while it runs"
    setup_browser.capture(page, "evolution-checking")

    answers["now"] = history(refresh="failed", versions=False, error="HistoryUnavailable: git for-each-ref failed (rc=128)")
    page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', {detail: {tab: 'evolution'}}))")
    page.locator(".evo-empty-line", has_text="no saved copy on this install yet").wait_for()
    assert page.locator(".evo-empty .evo-meta").inner_text().startswith("HistoryUnavailable")
    assert page.locator("#evo-start").is_enabled()
    setup_browser.capture(page, "evolution-unavailable")


def test_a_partly_measured_history_offers_try_again_and_keeps_its_copy(subscription_ui):
    partial = history()
    partial["refresh"].update(unmeasured=2, unmeasured_error="v7.9.2: 1 Python blob(s) of fc8cd8146a1b missing from the object store")
    partial["versions"][-1].update(code_lines=None, bible_bytes=None, system_bytes=None)
    answers, asked = {"now": partial}, []
    page = open_evolution(subscription_ui, lambda: answers["now"], asked=asked)
    status = page.locator(".evo-story-status")
    status.locator(".ui-status", has_text="2 versions couldn’t be measured: v7.9.2: 1 Python blob(s)").wait_for()
    assert page.locator(".evo-fact-value").first.inner_text() == "Not measured", "the newest version says so, never 0"
    assert page.locator(".evo-ch-rug").count() == 60, "the measured history stays on screen"
    setup_browser.capture(page, "evolution-partly-measured")
    answers["now"] = history(refresh="running")
    status.locator("[data-retry]").click()
    page.wait_for_function("() => document.querySelector('.evo-story-status').textContent.includes('Checking for new versions')")
    assert asked[-1] == {"force": ["1"]}, "Try again re-measures what failed"


def test_detail_controls_keep_keyboard_focus_across_repaints(subscription_ui):
    page = open_evolution(subscription_ui, history())
    page.locator(".evo-detail-title").wait_for()
    page.locator(".evo-chart-hit").focus()
    page.keyboard.press("PageUp")  # a milestone with neighbours on both sides
    back = page.locator('.evo-detail-nav [data-step="-1"]')
    back.focus()
    before = page.locator(".evo-detail-version").inner_text()
    page.keyboard.press("Enter")
    page.wait_for_function("v => document.querySelector('.evo-detail-version').textContent !== v", arg=before)
    assert page.evaluate("() => document.activeElement?.getAttribute('aria-label')") == "Previous version", \
        "the pressed control keeps keyboard focus after the detail repaints"
    page.keyboard.press("Enter")  # and a second press keeps stepping without reaching for the mouse
    page.locator(".evo-detail-body").wait_for()
    more = page.locator("[data-full]")
    more.focus()
    page.keyboard.press("Enter")
    page.wait_for_function("() => document.querySelector('[data-full]')?.textContent === 'Show less'")
    assert page.evaluate("() => document.activeElement?.hasAttribute('data-full')"), "Show more keeps focus as Show less"


def test_show_more_appears_when_a_narrower_window_starts_clipping_the_body(subscription_ui):
    short = history()
    short["versions"][-1]["release"]["body"] = ("Memory keeps an open conversation verbatim and turns a closed matter into an "
                                                "authored page whose quotations are checked.")
    page = open_evolution(subscription_ui, short, width=1100)
    body = page.locator(".evo-detail-body")
    body.wait_for()
    assert page.locator("[data-full]").is_hidden(), "two lines fit: nothing to show more of"
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_function("() => { const b = document.querySelector('.evo-detail-body'); return b.scrollHeight > b.clientHeight + 1; }")
    page.locator("[data-full]").wait_for(state="visible", timeout=5000)


def test_milestone_labels_follow_the_room_and_never_drop_a_version(subscription_ui):
    page = open_evolution(subscription_ui, history(), width=1360, height=800)
    page.locator(".evo-detail-title").wait_for()
    wide = page.locator(".evo-ch-mlabel").count()
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_function("n => document.querySelectorAll('.evo-ch-mlabel').length < n", arg=wide, timeout=5000)
    narrow = page.locator(".evo-ch-mlabel").count()
    assert wide > narrow, (wide, narrow)
    assert page.locator(".evo-ch-pin").count() == 28 and page.locator(".evo-ch-rug").count() == 60, "no event leaves"
    hit = page.locator(".evo-chart-hit")
    assert hit.get_attribute("aria-valuemax") == "60"
    hit.focus()
    page.keyboard.press("Home")
    assert page.locator(".evo-detail-version").inner_text() == "4.0.0"
    page.keyboard.press("ArrowRight")
    assert page.locator(".evo-detail-version").inner_text() == "4.1.0", "every version stays selectable by keyboard"


def test_touch_navigation_reaches_every_version_even_at_identical_dates(subscription_ui):
    """Dots can overlap; the existing step controls must still expose every release without a keyboard."""
    page = subscription_ui['new_page'](has_touch=True)
    try:
        ui = {**subscription_ui, 'page': page}
        answer = history()
        for row in answer['versions']:
            row['date'] = answer['versions'][0]['date']
        open_evolution(ui, answer, width=390, height=844)
        page.locator('.evo-detail-title').wait_for()
        for row in reversed(answer['versions'][:-1]):
            page.locator('[data-step="-1"]').tap()
            assert page.locator('.evo-detail-version').inner_text() == row['version']
        assert page.locator('[data-step="-1"]').is_disabled()
        for row in answer['versions'][1:]:
            page.locator('[data-step="1"]').tap()
            assert page.locator('.evo-detail-version').inner_text() == row['version']
        assert page.locator('[data-step="1"]').is_disabled()
    finally:
        page.close()


RUNNING = {**STATE, "evolution_enabled": True,
           "evolution_state": {"status": "running", "enabled": True, "detail": "Evolution task is running now."}}


def until(page, predicate, rounds=200):
    """Route callbacks run while Playwright is called; round-trip until one has recorded what the test needs."""
    for _ in range(rounds):
        if predicate():
            return
        page.evaluate("0")
    raise AssertionError("the route callback never recorded the request")


def test_a_start_the_runtime_applies_late_keeps_pause_and_arrives_with_the_page_poll(subscription_ui):
    """/api/command only queues a command, so the state read right after a Start can still say off. Pause stays
    offered while the Start is unconfirmed, and the page's own periodic state read brings what the runtime applied."""
    page = subscription_ui["page"]
    answers, commands, reads = {"state": STATE, "command": "queue"}, [], []

    def state():
        reads.append(answers["state"]["evolution_state"]["status"])
        return answers["state"]

    def command(route):
        commands.append(route.request.post_data_json["cmd"])
        if answers["command"] == "refuse":
            route.fulfill(status=409, json={"error": "Complete provider setup before sending this command."})
        elif answers["command"] == "lose":
            route.abort("connectionreset")
        else:
            route.fulfill(json={"status": "ok"})

    page.route("**/api/command", command)
    page.clock.install()
    open_evolution(subscription_ui, history(), state=state)
    start, pause, chip = page.locator("#evo-start"), page.locator("#evo-stop"), page.locator("#evo-campaign-chip")
    assert (chip.inner_text(), start.is_visible(), pause.is_visible()) == ("Off", True, False)

    def send(button, *, confirm=True):
        sent, read = len(commands), len(reads)
        button.click()
        if confirm:
            page.locator("[data-confirm-ok]").click()  # an empty objective: the backend's own
        until(page, lambda: len(commands) > sent and len(reads) > read)  # the command, then the state read after it
        page.wait_for_function("() => !document.querySelector('#evo-stop').disabled")
        page.evaluate("() => new Promise((done) => requestAnimationFrame(() => setTimeout(done, 0)))")

    def poll(now):
        answers["state"] = now
        read = len(reads)
        page.clock.fast_forward(21_000)  # app.js reads /api/state every 20 s for the whole page
        until(page, lambda: len(reads) > read)

    answers["command"] = "refuse"
    send(start)
    assert commands == ["/evolve on"] and page.locator(".toast", has_text="Could not send /evolve on").count() == 1
    assert (start.is_visible(), pause.is_visible()) == (True, False), "a refused Start was never queued"

    answers["command"] = "queue"
    send(start)
    assert chip.inner_text() == "Off", "the read right after the queued Start still says off"
    assert pause.is_visible() and pause.is_enabled(), "Pause stays offered while the Start is unconfirmed"
    assert start.is_visible()
    setup_browser.capture(page, "evolution-start-unconfirmed")

    poll(RUNNING)
    page.wait_for_function("() => document.querySelector('#evo-campaign-chip').textContent === 'Running'", timeout=5000)
    assert (start.inner_text(), start.is_visible(), pause.is_visible()) == ("New objective…", True, True), \
        "the page's read brought the applied Start without removing the existing retarget action"
    sent = len(commands)
    start.click()
    page.locator("[data-confirm-ok]").click()
    page.wait_for_function("() => !document.querySelector('[data-confirm-ok]')")
    assert len(commands) == sent, "an empty new objective keeps the running campaign unchanged"
    start.click()
    page.locator("[data-confirm-input]").fill("Keep history responsive")
    page.locator("[data-confirm-ok]").click()
    until(page, lambda: len(commands) > sent)
    page.wait_for_function("() => !document.querySelector('#evo-stop').disabled")
    assert commands[-1] == "/evolve on Keep history responsive"
    setup_browser.capture(page, "evolution-running-retargeted")

    send(pause, confirm=False)
    assert commands[-1] == "/evolve off" and chip.inner_text() == "Running", "the Pause is only queued"
    poll(STATE)
    page.wait_for_function("() => document.querySelector('#evo-campaign-chip').textContent === 'Off'", timeout=5000)
    assert (start.is_visible(), pause.is_visible()) == (True, False), "off again, and nothing left unconfirmed"

    answers["command"] = "lose"
    send(start)
    assert page.locator(".toast", has_text="Could not send /evolve on").count() >= 1
    assert pause.is_visible() and pause.is_enabled(), "no answer: the Start may have been queued, so Pause stays"


def released_history():
    """The production-shaped history up to 7.7.1, every version this README carries speaking with its real row."""
    from ouroboros import evolution_history as eh

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    answer = history()
    end = next(i for i, row in enumerate(answer["versions"]) if row["version"] == "7.7.1")
    answer["versions"] = answer["versions"][:end + 1]
    for row in answer["versions"]:
        found = eh.release_row(readme, row["version"])
        if found:
            row["release"] = eh.release_words({"version": row["version"], "row": found})
            row["milestone"] = bool(row["release"]["title"]) or row["level"] in ("major", "minor")
    return answer


def test_release_words_stay_as_written_in_a_translated_interface(subscription_ui):
    """The overlay translates the page's own words; a release author's words (title, lead, body, chart labels, the
    tooltip line) stay as written, even where the memory holds a translation, and are never reported as missing."""
    page = subscription_ui["page"]
    answer = released_history()
    newest = answer["versions"][-1]["release"]
    assert newest["source"] == "readme" and len(newest["body"]) > 1000, "the real 7.7.1 row"
    heading = newest["heading"].strip().removesuffix(".")
    title = heading[:1].upper() + heading[1:]
    misses = []
    memory = {"language": "ru", "english": False, "revision": 3, "entries": {
        "Previous version": {"text": "Предыдущая версия"}, "Code": {"text": "Код"},
        "How Ouroboros changed": {"text": "Как менялся Ouroboros"},
        title: {"text": "Всё, что влито после 7.7.0"}}}  # an authored title the memory happens to hold

    def report(route):
        misses.extend(item.get("key", "") for item in (route.request.post_data_json or {}).get("items", []))
        route.fulfill(json={"ok": True})

    page.route("**/api/ui/i18n", lambda route: route.fulfill(json=memory))
    page.route("**/api/ui/i18n/missing", report)
    open_evolution(subscription_ui, answer)
    page.wait_for_function("() => document.documentElement.lang === 'ru'")
    page.locator(".evo-detail-title").wait_for()
    page.wait_for_function("() => document.querySelector('[data-measure=\"code_lines\"]').textContent === 'Код'")
    box = page.locator(".evo-chart-hit").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + 60)
    page.locator(".evo-tip").wait_for(state="visible")
    page.evaluate("() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))")
    shown = page.evaluate("""async () => {
        const i18n = await import('/static/modules/i18n.js');
        await i18n.flushMisses();
        return [...document.querySelectorAll('#page-evolution :is(.evo-detail-title, .evo-detail-lead, .evo-detail-body,'
            + ' .evo-ch-mlabel, .evo-ch-label, .evo-tip-head)')].map((node) => [node.className.baseVal ?? node.className,
                node.textContent, i18n.missKeyFor(node.textContent)]);
    }""")
    kinds = {kind.split()[0] for kind, _text, _key in shown}
    assert {"evo-detail-title", "evo-detail-body", "evo-ch-label", "evo-ch-mlabel", "evo-tip-head"} <= kinds, kinds
    assert page.locator(".evo-detail-title").inner_text() == title, "authored words the memory knows stay as written"
    assert page.locator(".evo-detail-body").text_content() == newest["body"]
    leaked = sorted({key for key in misses} & {key for _kind, _text, key in shown})
    assert not leaked, leaked
    # The page's own words stay the interface language's: translated where the memory has them, reported where not.
    assert page.locator('.evo-detail-nav [data-step="-1"]').get_attribute("aria-label") == "Предыдущая версия"
    assert page.locator("#evo-story-title").inner_text() == "Как менялся Ouroboros"
    assert {"Next version", "Show more", "Constitution"} <= set(misses), sorted(misses)[:40]
    setup_browser.capture(page, "evolution-translated-interface")


# v6.2.0's annotation as git keeps it: its version has no README row, so these are its words.
V620 = ("v6.2.0: Critical Bugfixes + LLM-First Dedup\n\nCritical fixes:\n"
        "- Fix worker_id==0 hard-timeout bug (caused double task execution)\n"
        "- Fix double budget accounting (~2x drift eliminated)\n- Fix compact_context broken handler signature\n\n"
        "New features:\n- LLM-first task dedup (replaces hardcoded keyword similarity)\n"
        "- LLM-driven context compaction via light model\n\nOther fixes:\n- Health invariant #5 now functional\n"
        "- shlex.split for shell cmd parsing\n- Retry gets new task_id with lineage tracking\n"
        "- claude_code_edit timeout aligned to 300s\n- Direct-chat schedule_task audit warning")


def test_a_tag_annotation_reads_beneath_with_its_own_lines(subscription_ui):
    from ouroboros import evolution_history as eh

    answer = history()
    newest = answer["versions"][-1]
    newest["release"] = eh.release_words({"version": "6.2.0", "annotation": V620})
    page = open_evolution(subscription_ui, answer)
    assert page.locator(".evo-detail-title").inner_text() == "Critical Bugfixes + LLM-First Dedup"
    more = page.locator("[data-full]")
    more.wait_for(state="visible")
    more.click()
    lines = [line for line in page.locator(".evo-detail-body").inner_text().splitlines() if line.strip()]
    assert lines == [line for line in V620.split("\n\n", 1)[1].splitlines() if line.strip()], lines
    setup_browser.capture(page, "evolution-tag-annotation")

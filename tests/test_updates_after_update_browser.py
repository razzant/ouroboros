"""After-update samples inside the existing Last attempt of the real Updates panel (owner 25.10: A).

The real SPA (index.html, app.js, dashboard.js, updates.js, update_status.js and their CSS) runs in Chromium and
WebKit against the ``subscription_ui`` static server. ``/api/update/status`` answers with representative records,
so each state is exact: populated samples of the shown attempt, zero, missing and long values, a record of another
operation, an attempt without samples and one still running. A frame whose data hides the rows is a negative
control: it is kept as evidence and never counts as coverage of them. The update notice beside the version is read
with the panel: it closes with its own button and a confirmed success never wears it, so the panel keeps no Dismiss
and Restore is the attempt's one button.

Screenshots (1100x750, 560x800 and 390x844, light and dark, closed and open) and the facts read in the page go to
``OUROBOROS_UI_EVIDENCE_DIR`` when it is set.
"""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests import test_room_work_browser as rw
from tests.test_dark_voice_material_contrast_browser import COMPOSITE
from tests.ui_chat_viewport_smoke import wait_drawer_closed

subscription_ui = rw.subscription_ui
pytestmark = [pytest.mark.ui_browser]

OPERATION = "5be1c0a7e2d94f18b3a6c9d0e7f21345"
BASE, TARGET = "3f0c9a1e" * 5, "8d2b6e4f" * 5
GIT_LOG = {"branch": "ouroboros", "sha": TARGET[:12], "tags": [],
           "commits": [{"sha": TARGET, "short_sha": TARGET[:8], "date": "2026-10-21T08:40:00Z",
                        "message": "Merge the reviewed update"}]}
LABELS = ["Source", "Observed", "RSS peak", "CPU time, process lifetime", "CPU share, sample interval",
          "Duty overruns", "Direct message to first model call"]


def sample(generation, source, observed_at, **values):
    return {"generation": generation, "source_sha": source, "observed_at": observed_at, "pid": 41812,
            "scope": "server_process", "rss_bytes": 1153433600, "rss_measure": "peak", "cpu_seconds": 842.611204,
            "cpu_percent": 21.402913, "cpu_interval_seconds": 1800.508117, "duty_overruns": None,
            "direct_message_first_dispatch": {"task_id": "c41f9e02", "attempt_id": "turn_5a1d2c3b4e5f",
                                              "accepted_at": 90411.25, "dispatched_at": 90413.75, "seconds": 2.5},
            **values}


BEFORE = sample("a91c07f2d4b64e1f8c3d5e7f9a1b3c5d", BASE, "2026-10-21T08:31:02.114233+00:00")
AFTER = sample("d54e68d9ea1c43f9a617c9b226f3ffde", TARGET, "2026-10-21T09:12:05.798644+00:00",
               rss_bytes=1331691520, cpu_seconds=1137.432162, cpu_percent=58.38595954311026,
               cpu_interval_seconds=1428.2196088329947, duty_overruns={"reconcile_owner_bindings": 1},
               direct_message_first_dispatch={"task_id": "4e15961e", "attempt_id": "turn_602bcfa57861",
                                              "accepted_at": 148783.524252125, "dispatched_at": 148798.762589166,
                                              "seconds": 15.238337040995248})


def record(operation=OPERATION, before=BEFORE, after=AFTER, outcome="applied"):
    """``after_update`` exactly as the status endpoint projects it (``update_generation_facts.projection``)."""
    return {"operation_id": operation, "outcome": outcome, "base_sha": BASE, "target_sha": TARGET,
            "before": copy.deepcopy(before), "after": copy.deepcopy(after)}


def applied(operation=OPERATION, **extra):
    """The retained record as the restarted server serves it: concluded at boot, begun by a generation now gone."""
    return {"operation_id": operation, "generation": "a91c07f2", "current_generation": False, "active": False,
            "result": "restart_requested", "outcome": "applied", "outcome_at": "2026-10-21T09:05:41+00:00",
            "concluded_by": "boot", "stage": "restart_requested", "base_sha": BASE, "target_sha": TARGET, **extra}


def status(after_update, progress=None):
    now = datetime.now(timezone.utc)
    return {"managed": True, "check_ok": True, "available": False, "safe_to_apply": True, "from_cache": True,
            "checked_at": (now - timedelta(hours=2)).isoformat(), "warnings": [], "update_tx": {"active": False},
            "official_repo_url": "https://github.com/razzant/ouroboros", "target_ref": "managed/ouroboros",
            "update_channel": "development", "current_version": "7.8.0", "latest_version": "7.8.0",
            "current_short_sha": TARGET[:8], "latest_short_sha": TARGET[:8], "running_sha": TARGET,
            "update_progress": applied() if progress is None else progress, "after_update": after_update,
            "official_tags": [], "letter": {
                "state": "ready", "relation": "applied", "description_current": True,
                "key": {"base_sha": BASE, "target_sha": TARGET}, "author_version": "7.8.0", "target_version": "7.8.0",
                "written_at": "2026-10-21T07:58:00+00:00",
                "text": "This update makes opening projects faster and keeps the chat responsive while delegated "
                        "work copies its evidence. Reviews, schedules and pausing were reworked."}}


@pytest.fixture(params=["chromium", "webkit"])
def updates_ui(request, monkeypatch):
    """``subscription_ui`` in each engine; evidence per engine under ``updates-after-update``."""
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        monkeypatch.setenv("OUROBOROS_UI_EVIDENCE_DIR", str(Path(root) / "updates-after-update" / request.param))
    return {**request.getfixturevalue("subscription_ui"), "engine": request.param}


# What the owner can see of the panel and of the update notice, read in one frame. ``pairs`` are the rows the
# samples add: everything in the list from its first group heading on, counted only when actually painted (a closed
# disclosure paints nothing). The notice is read from its own state, since a phone keeps it in the closed drawer.
FACTS = """() => {
    const text = (el) => (el?.textContent || '').replace(/\\s+/g, ' ').trim();
    const shown = (el) => Boolean(el) && el.checkVisibility() && el.getBoundingClientRect().height > 0;
    const box = (el) => { const r = el.getBoundingClientRect(); return {x: r.x, y: r.y, right: r.right, bottom: r.bottom}; };
    const card = document.querySelector('#updates-status-card');
    const details = document.querySelector('#updates-attempt-details');
    const foot = details.parentElement;
    const notice = document.querySelector('.update-notice'), pill = document.getElementById('update-pill');
    const close = document.querySelector('.update-notice-close');
    const list = details.querySelector('#updates-attempt-facts');
    const raw = details.querySelector('#updates-attempt-raw');
    const pre = raw.querySelector('pre');
    const items = [...list.children], first = items.findIndex((el) => el.classList.contains('updates-facts-head'));
    const observed = (first < 0 ? [] : items.slice(first)).filter(shown);
    const heads = observed.filter((el) => el.classList.contains('updates-facts-head'));
    const labels = observed.filter((el) => el.tagName === 'DT' && !heads.includes(el));
    const pairs = labels.map((dt) => ({label: text(dt), value: text(dt.nextElementSibling),
        exact: text(dt.nextElementSibling.querySelector('time')), labelBox: box(dt), valueBox: box(dt.nextElementSibling),
        valueOverflow: dt.nextElementSibling.scrollWidth - dt.nextElementSibling.clientWidth}));
    const of = (el) => (shown(el) ? window.__contrast.of(el).ratio : null);
    const control = (selector) => { const el = document.querySelector(selector); return shown(el) ? box(el) : null; };
    return {
        headline: text(document.querySelector('#updates-summary')), action: text(document.querySelector('#btn-update-primary')),
        cardText: card.innerText, attemptShown: shown(document.querySelector('#updates-attempt')),
        attemptHeadline: text(document.querySelector('#updates-attempt-headline')),
        footButtons: [...foot.querySelectorAll(':scope > button')].filter(shown).map(text),
        notice: {shown: Boolean(notice) && !notice.hidden, text: pill && !pill.hidden ? text(pill) : '',
                 close: Boolean(close) && !close.hidden, tone: pill?.dataset.tone || ''},
        open: details.open, rawOpen: raw.open, rawShown: shown(raw), rawLabel: text(raw.querySelector('summary')),
        rawText: shown(pre) ? pre.textContent : null, rawOverflow: pre.scrollWidth - pre.clientWidth,
        heads: heads.map(text), pairs,
        strayBlocks: document.querySelectorAll('#updates-generation-facts, #updates-status-card h4:not([class])').length,
        cardOverflow: card.scrollWidth - card.clientWidth,
        pageOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        card: box(card), foot: box(foot), summary: box(details.querySelector('summary')), details: box(details),
        restore: control('#updates-attempt-restore'),
        meta: box(document.querySelector('#updates-meta')), status: box(document.querySelector('#updates-summary')),
        contrast: {label: of(labels[0]), value: of(labels[0]?.nextElementSibling), head: of(heads[0]), raw: of(pre),
                   summary: of(details.querySelector('summary'))},
    };
}"""


def facts(page):
    return page.evaluate(FACTS)


def evidence(name, value):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def capture(page, name, reveal=None, block="start"):
    """The screen at the viewer's size, scrolled to ``reveal``: what the owner sees there. An element shot would
    blank whatever the panel's own scroller holds outside its view, so the changed part is brought into view instead."""
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        if reveal:
            page.locator(reveal).evaluate("(el, block) => el.scrollIntoView({block})", block)
            rw.settle_frames(page)
        Path(root).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(root) / f"{name}.png"), animations="disabled")


def wait_reads(page, reads, count):
    for _ in range(100):
        if len(reads) >= count:
            break
        page.wait_for_timeout(40)
    assert len(reads) >= count, f"expected {count} status reads, saw {len(reads)}"
    page.wait_for_timeout(120)
    rw.settle_frames(page)


def open_updates(ui, current, *, theme="dark", width=1100, height=750):
    """The real app, Dashboard → Updates, answering every status read from ``current["status"]``."""
    page = ui["page"]
    transport = rw.Transport(page)
    reads = []

    def answer(route):
        reads.append(route.request.url)
        route.fulfill(json=current["status"])

    page.route("**/api/update/status**", answer)
    page.route("**/api/git/log**", lambda route: route.fulfill(json=GIT_LOG))
    rw.open_app(ui, transport, theme=theme, viewport={"width": width, "height": height})
    page.evaluate(COMPOSITE)
    if width < 980:
        page.locator("[data-mobile-nav-toggle]:visible").first.click()
    page.click('[data-nav-page="dashboard"]')
    if width < 980:
        wait_drawer_closed(page)
    count = len(reads)
    page.click('[data-dashboard-tab="updates"]')
    wait_reads(page, reads, count + 1)
    return page, transport, reads


def refresh(page, transport, reads):
    """The panel's own refresh path: one ``update_progress_changed`` frame, read by the panel and by the indicator."""
    count = len(reads)
    transport.send({"type": "update_progress_changed"})
    wait_reads(page, reads, count + 2)


def test_action_row_preserves_empty_notes_and_hidden_primary_buttons(updates_ui):
    from playwright.sync_api import expect

    current = {"status": {**status(None, {}), "checked_at": ""}}
    page, transport, reads = open_updates(updates_ui, current)
    note, button = page.locator("#updates-action-note"), page.locator("#btn-update-primary")
    row = note.locator("..")
    expect(button).to_be_visible()
    expect(note).to_be_empty()
    expect(note).not_to_be_visible()
    # The one next action stands at the headline's end (the tab body's hero), whatever the meta lines say.
    row_box, action_box = row.bounding_box(), button.bounding_box()
    assert abs(action_box["x"] + action_box["width"] - row_box["x"] - row_box["width"]) <= 1
    capture(page, "action-with-empty-note")
    # An unmanaged checkout can still say when it was checked, without offering
    # an update. Layout must not revive that hidden primary action.
    current["status"].update(managed=False, checked_at=datetime.now(timezone.utc).isoformat())
    refresh(page, transport, reads)
    expect(note).to_be_visible()
    expect(button).not_to_be_visible()
    capture(page, "checked-with-no-primary-action")
    current["status"]["checked_at"] = ""
    refresh(page, transport, reads)
    expect(note).not_to_be_visible()
    expect(button).not_to_be_visible()


def toggle(page, selector, key):
    page.locator(selector).focus()
    page.keyboard.press(key)
    rw.settle_frames(page)


def restore_keeps_its_side(current, width):
    """Restore, the attempt's one button (the notice has its own close), sits where approved A placed it: at the end
    of the row on a wide card, at the start under the summary in a narrow one. Closed, it shares the summary's row
    on a wide card; open, it is on the row below the list. It stays inside the attempt, under the headline's facts."""
    restore, foot, summary = current["restore"], current["foot"], current["summary"]
    if not restore:
        return False
    side = abs(restore["right"] - foot["right"]) < 1 if width >= 640 else abs(restore["x"] - foot["x"]) < 1
    if current["open"]:
        row = restore["y"] >= current["details"]["bottom"]
    else:
        row = abs(restore["y"] - summary["y"]) < 12 if width >= 640 else summary["bottom"] <= restore["y"]
    return side and row and restore["y"] >= current["meta"]["bottom"]


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width,height", [(1100, 750), (390, 844)])
def test_samples_are_rows_of_their_last_attempt_and_survive_refreshes(updates_ui, theme, width, height):
    current = {"status": status(record())}
    page, transport, reads = open_updates(updates_ui, current, theme=theme, width=width, height=height)
    prefix = f"populated-{theme}-{width}"
    closed = facts(page)
    # The card leads with the installation's state and its action; the samples add no block or sentence to it.
    assert closed["headline"] == "Up to date · 7.8.0" and closed["action"] == "Check again"
    assert closed["attemptShown"] and not closed["open"] and closed["heads"] == [] and closed["strayBlocks"] == 0
    for word in ("RSS", "CPU", "Duty", "first model call", TARGET, "4e15961e", "Raw record"):
        assert word not in closed["cardText"], word
    # A confirmed success wears no notice: its record and samples are the panel's alone, which has no Dismiss.
    assert not closed["notice"]["shown"] and closed["footButtons"] == ["Restore previous version"], closed
    assert closed["status"]["y"] < closed["meta"]["y"] < closed["summary"]["y"], closed
    assert restore_keeps_its_side(closed, width), closed
    capture(page, prefix + "-closed")

    page.locator("#updates-attempt-details > summary").click()
    rw.settle_frames(page)
    opened = facts(page)
    assert opened["heads"] == ["Before the update (recorded sample, server process)",
                               "After the update (recorded sample, server process)"], opened["heads"]
    # A value the sample did not record leaves its row (the raw record keeps the null): Before has no Duty overruns.
    before_labels = [label for label in LABELS if label != "Duty overruns"]
    assert [pair["label"] for pair in opened["pairs"]] == before_labels + LABELS, opened["pairs"]
    before, after = ({pair["label"]: pair for pair in side} for side in (opened["pairs"][:6], opened["pairs"][6:]))
    assert after["Source"]["value"] == TARGET and before["Source"]["value"] == BASE  # full technical values, whole
    # The local minute to read; the exact recorded stamp is the raw record's alone.
    assert after["Observed"]["exact"] == "" and "UTC" in after["Observed"]["value"]
    # Sizes and durations read as the other Dashboard tabs write them.
    assert after["RSS peak"]["value"] == "1.3 GB"
    assert after["CPU time, process lifetime"]["value"] == "19 min"
    assert after["CPU share, sample interval"]["value"] == "58.4 % over 24 min"
    assert after["Duty overruns"]["value"] == "reconcile owner bindings ×1"
    assert after["Direct message to first model call"]["value"] == "15 s from acceptance · task 4e15961e"
    assert "Duty overruns" not in before  # null is never shown as 0
    assert opened["rawShown"] and opened["rawLabel"] == "Raw record" and opened["rawText"] is None
    capture(page, prefix + "-open", "#updates-attempt-details > summary")
    capture(page, prefix + "-open-after", "#updates-attempt-facts .updates-facts-head >> nth=1")
    # Open, the list takes the whole row; Restore moves down to the row below it, never across, inside the card. Its
    # side is read against the attempt, whose width a classic scrollbar (WebKit) narrows once the panel scrolls.
    assert restore_keeps_its_side(opened, width) and opened["restore"]["right"] <= opened["card"]["right"], opened
    for pair in opened["pairs"]:
        assert pair["valueOverflow"] <= 1, pair
        assert (abs(pair["valueBox"]["x"] - pair["labelBox"]["x"]) < 1) == (width < 640), (width, pair)

    toggle(page, "#updates-attempt-raw > summary", "Enter")
    shown = facts(page)
    assert shown["rawOpen"] and json.loads(shown["rawText"]) == current["status"]["after_update"]
    assert '"duty_overruns": null' in shown["rawText"]  # the exact record: null stays null
    assert shown["rawOverflow"] <= 1 and shown["cardOverflow"] <= 1 and shown["pageOverflow"] <= 1, shown
    assert all(ratio >= 4.5 for ratio in shown["contrast"].values()), shown["contrast"]
    capture(page, prefix + "-raw", "#updates-attempt-raw-text", "center")

    # A refresh, a tab switch and a changed record repaint in place; both disclosures stay as the owner left them.
    refresh(page, transport, reads)
    page.click('[data-dashboard-tab="logs"]')
    count = len(reads)
    page.click('[data-dashboard-tab="updates"]')
    wait_reads(page, reads, count + 1)
    kept = facts(page)
    assert kept["open"] and kept["rawOpen"] and kept["rawText"] == shown["rawText"]
    assert [pair["value"] for pair in kept["pairs"]] == [pair["value"] for pair in shown["pairs"]]
    current["status"]["after_update"]["after"]["direct_message_first_dispatch"]["seconds"] = 3.75
    refresh(page, transport, reads)
    moved = facts(page)
    assert moved["open"] and moved["rawOpen"] and '"seconds": 3.75' in moved["rawText"]
    assert moved["pairs"][-1]["value"] == "3.8 s from acceptance · task 4e15961e"

    toggle(page, "#updates-attempt-raw > summary", "Space")
    toggle(page, "#updates-attempt-details > summary", "Enter")
    again = facts(page)
    assert not again["open"] and not again["rawOpen"] and again["heads"] == []
    assert restore_keeps_its_side(again, width), again
    evidence(prefix, {"closed": closed, "open": shown, "refreshed": kept, "changed": moved, "closedAgain": again,
                      "statusReads": len(reads), "console": transport.console})


@pytest.mark.parametrize("width,height", [(1100, 750), (390, 844)])
def test_zero_missing_and_long_values_keep_their_words_and_fit(updates_ui, width, height):
    duties = {f"reconcile_{name}_after_restart_owner_binding_check": count
              for name, count in (("project_workspaces", 3), ("delegated_review_custody", 1),
                                  ("scheduled_task_ledger", 12), ("subscription_quota_snapshots", 2))}
    after = sample("f" * 32 + "-generation-with-a-long-custody-session-identifier", TARGET,
                   "2026-10-21T09:12:05.798644+00:00", rss_bytes=0, rss_measure="current", cpu_seconds=0.0,
                   cpu_percent=None, cpu_interval_seconds=None, duty_overruns=duties,
                   direct_message_first_dispatch=None)
    current = {"status": status(record(before=None, after=after))}
    page, _transport, _reads = open_updates(updates_ui, current, width=width, height=height)
    page.locator("#updates-attempt-details > summary").click()
    toggle(page, "#updates-attempt-raw > summary", "Enter")
    shown = facts(page)
    assert shown["heads"] == ["Before the update", "After the update (recorded sample, server process)"]
    values = {pair["label"]: pair["value"] for pair in shown["pairs"]}
    assert values["Sample"] == "unobserved"
    assert values["RSS"] == "0 bytes" and values["CPU time, process lifetime"] == "0.0 s"  # zero is measured
    # What the sample did not record leaves its row; the raw record below keeps the nulls.
    assert "CPU share, sample interval" not in values and "Direct message to first model call" not in values
    assert values["Duty overruns"] == ", ".join(f"{duty.replace('_', ' ')} ×{count}" for duty, count in duties.items())
    assert json.loads(shown["rawText"]) == current["status"]["after_update"] and '"before": null' in shown["rawText"]
    assert all(pair["valueOverflow"] <= 1 for pair in shown["pairs"]), shown["pairs"]
    assert shown["rawOverflow"] <= 1 and shown["cardOverflow"] <= 1 and shown["pageOverflow"] <= 1, shown
    capture(page, f"long-zero-missing-{width}-rows", "#updates-attempt-facts .updates-facts-head >> nth=1")
    capture(page, f"long-zero-missing-{width}-raw", "#updates-attempt-raw-text", "center")
    evidence(f"long-zero-missing-{width}", shown)


def test_samples_wait_for_their_attempt_and_never_join_another(updates_ui):
    running = {**applied(), "active": True, "current_generation": True, "outcome": "", "result": "",
               "concluded_by": "", "stage": "checking", "stage_started_at": datetime.now(timezone.utc).isoformat()}
    current = {"status": status(record(after=None), progress=running)}
    page, transport, reads = open_updates(updates_ui, current)
    # While its update runs, the existing Last attempt is hidden and its samples with it (the accepted cost of A).
    during = facts(page)
    assert during["headline"] == "Checking the update and dependencies…" and not during["attemptShown"]
    assert "RSS" not in during["cardText"] and during["strayBlocks"] == 0
    # The notice names the stage and offers no close: the attempt itself ends it.
    assert during["notice"] == {"shown": True, "text": "Checking update…", "close": False, "tone": "neutral"}, during
    capture(page, "running")

    current["status"] = status(record())
    refresh(page, transport, reads)
    page.locator("#updates-attempt-details > summary").click()
    rw.settle_frames(page)
    done = facts(page)
    assert done["attemptShown"] and len(done["heads"]) == 2 and done["rawLabel"] == "Raw record"
    assert not done["notice"]["shown"], done["notice"]  # a confirmed success wears no notice

    # This server's next attempt could not be saved: the retained record still names the earlier operation.
    unsaved = {**applied("e0d1c2b3a4f5061728394a5b6c7d8e9f"), "current_generation": True, "outcome": "not_applied",
               "concluded_by": "executor", "reason": "release_moved", "result": "failed", "retention_unconfirmed": True}
    current["status"] = status(record(), progress=unsaved)
    refresh(page, transport, reads)
    toggle(page, "#updates-attempt-raw > summary", "Enter")
    other = facts(page)
    assert "could not be saved" in other["cardText"] and other["heads"] == []
    assert other["rawLabel"] == "Raw record (not matched to this attempt)"
    # Its failure keeps a closable notice; the panel offers no button of its own for it.
    assert other["notice"] == {"shown": True, "text": "Update not applied", "close": True, "tone": "error"}, other
    assert other["footButtons"] == [], other["footButtons"]
    assert json.loads(other["rawText"]) == current["status"]["after_update"]
    capture(page, "another-operation", "#updates-attempt")

    # An attempt retained without samples (a record older than them) shows its own rows and no record.
    current["status"] = status(None, progress=applied("7a6b5c4d3e2f10293847566574839201"))
    refresh(page, transport, reads)
    legacy = facts(page)
    assert legacy["open"] and legacy["heads"] == [] and not legacy["rawShown"]
    evidence("lifecycle", {"running": during, "concluded": done, "anotherOperation": other, "withoutSamples": legacy,
                           "console": transport.console})


def drawer(page, width, opened):
    """A phone keeps the notice in the navigation drawer: open it to reach the notice, close it to read the panel."""
    if width >= 980:
        return
    if opened:
        page.locator("[data-mobile-nav-toggle]:visible").first.click()
        page.wait_for_function("document.getElementById('content').inert === true")
    else:
        page.keyboard.press("Escape")
        wait_drawer_closed(page)
    page.evaluate("() => Promise.all(document.getAnimations().filter(a => a.effect?.getTiming().iterations !== Infinity)"
                  ".map(a => a.finished.catch(() => {})))")
    rw.settle_frames(page)


@pytest.mark.parametrize("theme,width,height", [("dark", 1100, 750), ("light", 390, 844)])
def test_the_notice_closes_itself_and_leaves_the_last_attempt_whole(updates_ui, theme, width, height):
    """An unconfirmed outcome keeps its notice until the notice's own close. Closing it by keyboard hides that notice
    and nothing else: the Last attempt keeps its rows, its one recorded sample and its exact record, and no request
    is sent. The next attempt, a confirmed success, wears no notice; its samples show in the panel alone."""
    unconfirmed = {**applied(), "outcome": "unknown", "reason": "startup_unconfirmed"}
    current = {"status": status(record(after=None, outcome="unknown"), progress=unconfirmed)}
    page, transport, reads = open_updates(updates_ui, current, theme=theme, width=width, height=height)
    page.locator("#updates-attempt-details > summary").click()
    toggle(page, "#updates-attempt-raw > summary", "Enter")
    shown = facts(page)
    assert shown["notice"] == {"shown": True, "text": "Update outcome unconfirmed", "close": True, "tone": "neutral"}
    assert shown["attemptHeadline"] == "The last update outcome is unconfirmed." and shown["footButtons"] == []
    assert shown["heads"] == ["Before the update (recorded sample, server process)", "After the update"], shown["heads"]
    assert [pair["label"] for pair in shown["pairs"]] == [*(label for label in LABELS if label != "Duty overruns"), "Sample"], shown["pairs"]
    assert shown["pairs"][0]["value"] == BASE and shown["pairs"][-1]["value"] == "unobserved"
    assert shown["rawLabel"] == "Raw record" and json.loads(shown["rawText"]) == current["status"]["after_update"]
    capture(page, f"notice-{theme}-{width}-unconfirmed", "#updates-attempt-details > summary")

    drawer(page, width, True)
    capture(page, f"notice-{theme}-{width}-notice")
    sent = len(transport.requests)
    toggle(page, ".update-notice-close", "Enter")
    drawer(page, width, False)
    hidden = facts(page)
    # Only the notice went, in this browser: the attempt, its rows and its exact record stay as they were.
    assert not hidden["notice"]["shown"] and not hidden["notice"]["close"], hidden["notice"]
    assert hidden["open"] and hidden["rawOpen"] and hidden["rawText"] == shown["rawText"]
    assert [(p["label"], p["value"]) for p in hidden["pairs"]] == [(p["label"], p["value"]) for p in shown["pairs"]]
    assert hidden["attemptHeadline"] == shown["attemptHeadline"]
    assert [request for request in transport.requests[sent:] if request[0] != "GET"] == [], transport.requests[sent:]
    refresh(page, transport, reads)
    kept = facts(page)
    # Six recorded rows before the update (its unrecorded duty overruns leave their row) and the after side's one line.
    assert not kept["notice"]["shown"] and kept["rawText"] == shown["rawText"] and len(kept["pairs"]) == 7, kept
    capture(page, f"notice-{theme}-{width}-closed", "#updates-attempt-details > summary")

    # The next attempt is a confirmed success: nothing to close, and its samples are the panel's.
    following = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
    current["status"] = status(record(operation=following), progress=applied(following))
    refresh(page, transport, reads)
    success = facts(page)
    assert not success["notice"]["shown"] and success["attemptHeadline"] == "Update completed."
    # Both sides recorded; the before side has no duty overruns, so its row leaves (13 rows, not 14).
    assert len(success["heads"]) == 2 and len(success["pairs"]) == 13 and success["rawLabel"] == "Raw record"
    evidence(f"notice-{theme}-{width}", {"unconfirmed": shown, "noticeClosed": hidden, "refreshed": kept,
                                         "nextSuccess": success, "console": transport.console})


def test_restore_moves_only_down_when_the_list_opens_in_a_middle_width_window(updates_ui):
    """Between a phone and a desk (a 560px window) the attempt is narrow, so Restore sits under the summary at its
    start; opening the list moves it down to the row below and never across."""
    page, _transport, _reads = open_updates(updates_ui, {"status": status(record())}, theme="light", width=560,
                                            height=800)
    closed = facts(page)
    capture(page, "middle-light-560-closed")
    page.locator("#updates-attempt-details > summary").click()
    rw.settle_frames(page)
    opened = facts(page)
    capture(page, "middle-light-560-open", "#updates-attempt-raw", "center")
    evidence("middle-light-560", {"closed": closed, "open": opened})
    assert restore_keeps_its_side(closed, 560) and restore_keeps_its_side(opened, 560), (closed, opened)
    assert all(pair["valueOverflow"] <= 1 for pair in opened["pairs"]) and opened["cardOverflow"] <= 1, opened


def test_a_frame_without_the_record_is_kept_as_a_negative_control_not_coverage(updates_ui):
    """The fixture of the earlier panel test omitted ``after_update``: its frames show no samples at all."""
    current = {"status": {key: value for key, value in status(record()).items() if key != "after_update"}}
    page, transport, reads = open_updates(updates_ui, current)
    page.locator("#updates-attempt-details > summary").click()
    rw.settle_frames(page)
    without = facts(page)
    assert without["open"] and without["heads"] == [] and without["pairs"] == [] and not without["rawShown"]
    capture(page, "negative-control-without-after-update", "#updates-attempt-details > summary")
    current["status"] = status(record())
    refresh(page, transport, reads)
    populated = facts(page)
    assert len(populated["heads"]) == 2 and len(populated["pairs"]) == 13  # the before side records no duty overruns
    evidence("negative-control", {
        "without_after_update": {"rows_painted": len(without["pairs"]), "counts_as_coverage": False, "facts": without},
        "with_after_update": {"rows_painted": len(populated["pairs"]), "counts_as_coverage": True}})


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_disclosure_focus_and_raw_tray_use_shared_design_tokens(updates_ui, theme):
    page, _transport, _reads = open_updates(updates_ui, {"status": status(record())}, theme=theme)
    toggle(page, "#updates-attempt-details > summary", "Enter")
    toggle(page, "#updates-attempt-raw > summary", "Enter")
    selectors = ["#updates-attempt-details > summary", "#updates-attempt-raw > summary",
                 ".updates-letter-details > summary", ".updates-recovery > summary"]
    observed = []
    for selector in selectors:
        page.keyboard.press("Tab")  # establish keyboard modality before selecting the exact control
        page.locator(selector).focus()
        value = page.locator(selector).evaluate("""el => {
            const style = getComputedStyle(el);
            const probe = document.createElement('span');
            probe.style.color = 'var(--focus-accent-border)';
            el.append(probe);
            const expectedColor = getComputedStyle(probe).color;
            probe.remove();
            return {visible: el.matches(':focus-visible'), color: style.outlineColor,
                    width: style.outlineWidth, kind: style.outlineStyle, offset: style.outlineOffset, expectedColor};
        }""")
        observed.append({"selector": selector, **value})
    radius = page.locator("#updates-attempt-raw-text").evaluate("""el => {
        const style = getComputedStyle(el);
        return {actual: style.borderTopLeftRadius, expected: style.getPropertyValue('--radius-inner').trim()};
    }""")
    evidence(f"design-tokens-{theme}", {"focus": observed, "radius": radius})
    for value in observed:
        assert value["visible"] and value["color"] == value["expectedColor"], value
        assert (value["width"], value["kind"], value["offset"]) == ("2px", "solid", "2px"), value
    assert radius["actual"] == radius["expected"], radius
    page.keyboard.press("Tab")
    page.locator("#updates-attempt-raw > summary").focus()
    capture(page, f"design-tokens-{theme}", "#updates-attempt-raw > summary", "center")


@pytest.mark.parametrize("theme,width,height", [("dark", 1100, 750), ("light", 390, 844)])
def test_assisted_apply_keeps_server_binding_and_failed_recovery_instruction(updates_ui, theme, width, height):
    """A late apply reply cannot replace a bound transaction with a client-invented one."""
    from playwright.sync_api import expect

    current = {"status": {**status(None, progress={}), "available": True, "latest_sha": TARGET}}
    page, transport, reads = open_updates(updates_ui, current, theme=theme, width=width, height=height)
    plan = {"available": True, "kind": "conflicting", "base_sha": BASE, "target_sha": TARGET,
            "local_dirty_count": 0, "code_conflict_paths": [], "doc_conflict_paths": [], "hot_code_paths": []}
    page.route("**/api/update/preflight", lambda route: route.fulfill(json={"merge_plan": plan}))
    pending = []
    page.route("**/api/update/apply", lambda route: pending.append(route))
    try:
        page.click("#btn-update-primary")
        with page.expect_request("**/api/update/apply"):
            page.click("[data-confirm-ok]")
        expect(page.locator("#updates-summary")).to_have_text("Applying the update…")
        progress = {"operation_id": OPERATION, "generation": "server-a", "current_generation": True,
                    "active": False, "result": "assisted_started", "stage": "resolving",
                    "tx_attempt_id": "bound-transaction", "base_sha": BASE, "target_sha": TARGET}
        current["status"].update(update_progress=progress, update_tx={
            "active": True, "attempt_id": "bound-transaction", "phase": "assisted_resolution",
            "task_id": "resolver-task"})
        refresh(page, transport, reads)  # the authoritative status arrives before the apply response
        assert pending
        pending.pop().fulfill(json={"status": "assisted_started", "task_id": "resolver-task"})
        expect(page.locator("#updates-summary")).to_have_text("A conflicting update is being resolved under review.")
        assert page.locator("#updates-attempt-headline").inner_text() == "A reviewed resolution owns this update."
        resolving = facts(page)
        # Said of the admission: the later failed and finished-rollback states below never contradict it.
        page.get_by_text("Ouroboros started a reviewed resolution of the update merge. Watch progress in chat.",
                         exact=True).wait_for(state="hidden", timeout=10000)
        capture(page, f"assisted-binding-{theme}-{width}")

        current["status"]["update_tx"]["phase"] = "rolling_back"
        current["status"]["update_progress"] = {
            **progress, "current_generation": False, "outcome": "failed", "concluded_by": "boot",
            "reason": "rollback_incomplete", "error": "The update's original failure.",
            "stash_note": "The rollback could not save its intent.", "keep_branch": "recovery/local-work",
            "retention_unconfirmed": True}
        refresh(page, transport, reads)
        expect(page.locator("#updates-summary")).to_have_text(
            "The update failed, and restoring the previous version did not finish.")
        retry = "Quit and reopen the desktop app, or restart the server process for a web deployment, to retry the recovery."
        closed = facts(page)
        assert closed["cardText"].count(retry) == 1, closed["cardText"]
        assert "recovery/local-work" in closed["cardText"] and "could not be saved" in closed["cardText"]
        assert not closed["restore"] and closed["cardOverflow"] <= 1 and closed["pageOverflow"] <= 1
        capture(page, f"failed-recovery-{theme}-{width}-closed", "#updates-summary")
        page.locator("#updates-attempt-details > summary").click()
        opened = facts(page)
        assert "The update's original failure." in opened["cardText"]
        assert "The rollback could not save its intent." in opened["cardText"]
        assert opened["cardText"].count(retry) == 1, opened["cardText"]
        capture(page, f"failed-recovery-{theme}-{width}-details", "#updates-attempt-details > summary")

        # The same report under pending_boot_smoke keeps the in-app Restart that phase allows.
        current["status"]["update_tx"]["phase"] = "pending_boot_smoke"
        refresh(page, transport, reads)
        expect(page.locator("#btn-update-primary")).to_have_text("Restart now")
        smoke = facts(page)
        assert smoke["cardText"].count(retry) == 1, smoke["cardText"]

        # A later boot finished the rollback and awaits its restart (rollback_restart_pending): the local-work verdict
        # gives the one instruction, and Last attempt no longer says restoring "did not finish".
        current["status"]["update_tx"].update(phase="marker_cleanup_retry", local_work_recovery=True)
        refresh(page, transport, reads)
        expect(page.locator("#updates-summary")).to_have_text("Local changes still need recovery confirmation.")
        finished = facts(page)
        assert finished["cardText"].count("Quit and reopen") == 1, finished["cardText"]
        assert "did not finish" not in finished["cardText"]
        assert finished["attemptHeadline"] == "The update did not complete."
        assert "The rollback could not save its intent." in finished["cardText"]  # history stays in Details
        assert finished["cardOverflow"] <= 1 and finished["pageOverflow"] <= 1
        capture(page, f"rollback-finished-{theme}-{width}", "#updates-summary")
        evidence(f"assisted-recovery-{theme}-{width}", {"resolving": resolving, "closed": closed, "open": opened,
                                                      "smoke": smoke, "finished": finished,
                                                      "statusReads": len(reads), "console": transport.console})
    finally:
        for route in pending:
            route.abort()

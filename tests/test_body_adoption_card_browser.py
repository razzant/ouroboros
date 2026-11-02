"""#2242 in real browsers: isolated server -> ledger -> task result -> adoption row -> history -> Main card.

The review records are built and revised by the ledger's own writers (``build_commit_gate_record``,
``note_author_decision``, ``note_commit_landing``, in the commit gate's order), the task results by the task
pipeline's writers, and each adoption row by the real boot publisher (``body_adoption._publish_on_task``) and
the supervisor's own send handler. The page is the isolated, origin-proof candidate server
(``direct_server_with_data``); nothing in the browser is mocked. Scenes follow the chosen demonstration (owner
29.10, A): PASS on the exact bytes, FAIL with the author continuing, PASS of an earlier version, no verdict, an
unreadable record, and a row without its card. Narrow pages (390 and 320 px) add the longest relations: a commit
adopted as part of a later one, two commits under one record with their own decisions, an unknown tree. Last,
with the page open, an adoption arrives for a card that says ``Committed`` (it turns ``Adopted`` without a
reload) and one whose task fact is lost on the way (its row arrives; its Reviews keep saying ``Committed``). Then a
history read the server answered while the adoption was still unconfirmed lands after the card read the adoption: the
opened row keeps its node, its disclosure and ``Adopted``. ``EVIDENCE`` is a module seam a QA wrapper may set (a
directory for the frames and measurements).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as rl
from tests.test_review_record_card_projection import CRITICAL, _answer
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _emit_ws_frame
from tests.ui_console_errors import PageErrors

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
EVIDENCE: Path | None = None
GOAL = "Return focus to the composer after a message is sent, on both the Enter and the Send button paths"
NAME = "Фокус в поле ввода после отправки"
SUBJECT, LATER = "f" * 64, "e" * 64
# The chosen demonstration's own author words (build/scenes.py RATIONALE), one long enough to need its Record facts.
REASONS = {
    "rejected": "The Plus menu closes on send before the reply can land, so the sequence the reviewer describes cannot "
                "occur; the new test covers both send paths. The menu's own close handler runs on pointerdown, before "
                "the send, and the reply can only arrive after the round trip, so the composer never takes focus away "
                "from an open menu. END-OF-REASON",
    "revised": "Applied the reviewer's note on the test name and tightened one assertion after the review.",
    "noverdict": "Two reviewers did not answer (a time limit and a quota); the third passed it. The change is two lines "
                 "in one module and covered by the new test.",
    "later": "Applied the reviewer's naming note after the review; this commit was not adopted.",
}
SCENES = ("pass", "rejected", "revised", "noverdict", "unavailable", "nocard")
NARROW = ("range", "rangefail", "ab", "unknown", "late", "lost")  # 390 and 320 only; late/lost adopt with the page open
LATE = ("delayed",)  # unconfirmed at the first boot; it adopts with the page open, after a history read was answered
WAITING = "Body {cand} is running, but the native host did not confirm its artifact; the adoption stays unconfirmed."
# scene: (reviewed commit, verdict, [(landed commit, decision subject, disposition, reason key)], adopted (head, previous))
FACTS = {
    "pass": ("cand", "PASS", [("cand", "", "", "")], ("cand", "early")),
    "rejected": ("cand", "FAIL", [("cand", SUBJECT, "rejected", "rejected")], ("cand", "early")),
    "revised": ("early", "PASS", [("cand", SUBJECT, "accepted", "revised")], ("cand", "early")),
    "noverdict": ("cand", "", [("cand", SUBJECT, "accepted", "noverdict")], ("cand", "early")),
    "unavailable": ("cand", "PASS", [("cand", "", "", "")], ("cand", "early")),
    "range": ("early", "", [("early", SUBJECT, "accepted", "noverdict")], ("cand", "base")),
    "rangefail": ("early", "FAIL", [("early", SUBJECT, "rejected", "rejected")], ("cand", "base")),
    "ab": ("early", "FAIL", [("early", SUBJECT, "rejected", "rejected"), ("cand", LATER, "accepted", "later")], ("early", "base")),
    "unknown": ("cand", "PASS", [("cand", "", "", "")], ("cand", "early")),
    "late": ("cand", "PASS", [("cand", "", "", "")], None),
    "lost": ("cand", "PASS", [("cand", "", "", "")], None),
    "delayed": ("cand", "PASS", [("cand", "", "", "")], None),
}
COMBOS = (("webkit", "light", 1100, 735), ("webkit", "dark", 1100, 735), ("webkit", "light", 390, 844),
          ("webkit", "dark", 390, 844), ("chromium", "light", 1100, 735), ("chromium", "light", 390, 844),
          ("webkit", "light", 320, 700), ("chromium", "light", 320, 700))
RELATION = {
    "pass": "Adopted {cand}: the exact bytes this review passed.",
    "rejected": "Adopted {cand} despite this rejection, by Ouroboros's own decision.",
    "revised": "This review was for earlier code; it does not cover adopted {cand}.",
    "noverdict": "Adopted {cand} with no verdict from this review, by Ouroboros's own decision.",
    "range": "Adopted {early} as part of {cand}, with no verdict from this review, by Ouroboros's own decision.",
    "rangefail": "Adopted {early} as part of {cand} despite this rejection, by Ouroboros's own decision.",
    "ab": "Adopted {early} despite this rejection, by Ouroboros's own decision.",
    "unknown": "Adopted {cand}; whether this review saw these bytes is not recorded.",
    "late": "Committed {cand}: the exact bytes this review passed.",
    "lost": "Committed {cand}: the exact bytes this review passed.",
    "delayed": "Committed {cand}: the exact bytes this review passed.",
}
STATE = {"pass": "passed", "rejected": "rejected", "revised": "passed", "noverdict": "no verdict", "range": "no verdict",
         "rangefail": "rejected", "ab": "rejected", "unknown": "passed", "late": "passed", "lost": "passed",
         "delayed": "passed"}
SETTLE = """() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"""
# The chosen frames hold the opened card's Reviews door near the top of the feed; a page frame does the same.
ANCHOR = """el => { const feed = el.closest('#chat-messages'); if (feed) feed.scrollTop += el.getBoundingClientRect().top - 110; }"""
# Each chat-history read the page has applied: its JSON was consumed and the task that applies it has run.
COUNT_HISTORY_READS = """() => {
    window.__historyReadsApplied = 0;
    const nativeFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
        const read = nativeFetch(input, init);
        const raw = typeof input === 'string' ? input : input?.url || '';
        if (new URL(raw, location.href).pathname !== '/api/chat/history') return read;
        return read.then((response) => {
            const json = response.json.bind(response);
            response.json = () => json().finally(() => setTimeout(() => { window.__historyReadsApplied += 1; }, 0));
            return response;
        });
    };
}"""
# A frame on the page's current socket (a reconnect replaces the first one).
EMIT = """frame => window.__testSockets.at(-1).dispatchEvent(new MessageEvent('message', {data: JSON.stringify(frame)}))"""
# The relation's own box against its text's line boxes: a clamp or a clip shows as text past the box.
GEOMETRY = """sel => { const node = document.querySelector(sel); if (!node) return null;
    const style = getComputedStyle(node); const box = node.getBoundingClientRect(); const range = document.createRange();
    range.selectNodeContents(node); const lines = [...range.getClientRects()].filter(rect => rect.width > 0);
    return {text: node.textContent.trim(), clientHeight: node.clientHeight, scrollHeight: node.scrollHeight,
        lineHeight: parseFloat(style.lineHeight), fontSize: style.fontSize, color: style.color, display: style.display,
        clamp: style.webkitLineClamp || style.lineClamp || '', overflow: style.overflow, width: box.width,
        lines: new Set(lines.map(rect => Math.round(rect.top))).size,
        textBottom: Math.max(...lines.map(rect => rect.bottom)), boxBottom: box.bottom}; }"""


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _seat(slot, model, verdict, *, parts=("change", "coupling"), error=False, findings=()):
    if error:
        return {"slot_id": slot, "model_id": model, "status": "error", "parts": list(parts),
                "error": "The reviewer did not answer within 600 s."}
    return {"slot_id": slot, "model_id": model, "status": "responded", "parts": list(parts),
            "answers": {part: _answer(part, verdict if part == "change" else "PASS", findings if part == "change" else ())
                        for part in parts}}


def _record(task_id, verdict, tree, parent):
    """One commit-gate wave on ``tree`` from the ledger's own builder: PASS, FAIL, or ``""`` for no verdict."""
    seats = {
        "PASS": [_seat("s1", "openai/gpt-5.5", "PASS"), _seat("s2", "anthropic/claude-opus-5-5", "PASS"),
                 _seat("s3", "google/gemini-3.1-pro", "PASS")],
        "FAIL": [_seat("s1", "openai/gpt-5.5", "FAIL", findings=[CRITICAL]),
                 _seat("s2", "anthropic/claude-opus-5-5", "PASS"), _seat("s3", "google/gemini-3.1-pro", "PASS")],
        "": [_seat("s1", "openai/gpt-5.5", "PASS"), _seat("s2", "anthropic/claude-opus-5-5", "", error=True),
             _seat("s3", "google/gemini-3.1-pro", "", error=True)],
    }[verdict]
    plans = [{"slot_id": row["slot_id"], "model": row["model_id"], "route": "api_chat", "parts": row["parts"],
              "retrieves": True} for row in seats]
    record = rl.build_commit_gate_record({
        "task_id": task_id, "root_task_id": task_id, "enforcement": "blocking" if verdict == "PASS" else "advisory",
        "enforcement_blocks": verdict == "PASS", "composition": "full_pool", "chosen_by": "owner",
        "triad_raw": seats, "structured": {"rows": plans, "started_ts": "2026-10-28T14:05:31+00:00"},
        "binding": {"tree_sha": tree, "parents": [parent], "diff_sha256": "d" * 64},
    })
    record.brief["goal"] = GOAL
    return record


def _history_repo(path):
    """Three real commits: the serving base, an earlier candidate commit and the candidate built on it."""
    path.mkdir(parents=True)
    for args in (("init", "-q"), ("config", "user.name", "Test"), ("config", "user.email", "test@example.invalid")):
        _git(path, *args)
    for text in ("value = 1\n", "value = 2\n", "value = 3\n"):
        (path / "change.py").write_text(text, encoding="utf-8")
        _git(path, "add", "change.py")
        _git(path, "commit", "-q", "-m", text.strip())
    return {name: {"sha": _git(path, "rev-parse", rev), "tree": _git(path, "rev-parse", f"{rev}^{{tree}}")}
            for name, rev in (("base", "HEAD~2"), ("early", "HEAD~1"), ("cand", "HEAD"))}


def _note(cand, prev, pushed):
    return (f"Adopted candidate commit {cand[:12]} on ouroboros; previous commit {prev[:12]}.\n"
            + ("Publication: Pushed ouroboros to origin." if pushed else "Publication withheld: this adoption rests on "
               "selected test evidence and the automatic post-adoption push needs a full run, so nothing was pushed."))


def seed(fixture, monkeypatch, scratch):
    """Every scene's record, task result and adoption row, written while the server is stopped."""
    from collections import deque

    from ouroboros import body_adoption
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.post_task_synthesis import _record_task_facts
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_SCHEDULED, write_task_result
    from supervisor import events_chat_delivery as delivery
    from supervisor import message_bus, state, workers

    root, repo = fixture["data_dir"], fixture["repo_dir"]
    fixture["stop_server"]()
    history = scratch / "adopted-history"
    commits = _history_repo(history)
    sha = {name: commit["sha"] for name, commit in commits.items()}
    events = []
    monkeypatch.setattr(workers, "get_event_q", lambda **_kw: SimpleNamespace(put=events.append))
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": 1})
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=1000))
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "publish_event", lambda *a, **k: None)
    bridge = message_bus.LocalChatBridge({})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    frames, errors = [], []
    bridge._broadcast_fn = frames.append  # the live frames the server's socket would have carried
    ctx = SimpleNamespace(DRIVE_ROOT=root, task_registry={}, send_with_budget=message_bus.send_with_budget,
                          append_jsonl=lambda *a: errors.append(a))
    env = SimpleNamespace(drive_root=root, repo_dir=repo)
    published = []
    for scene in SCENES + NARROW + LATE:
        task_id = f"adopt-{scene}"
        if scene == "nocard":  # an older task: its record names Main, its card is not in this page's history
            task_id = "adopt-nocard-older"
            write_task_result(root, task_id, STATUS_COMPLETED, chat_id=1, root_task_id=task_id, result="Готово.")
        else:
            reviewed, verdict, landings, _adopted = FACTS[scene]
            parent = sha["base"] if reviewed == "early" else sha["early"]
            rid = rl.write_record(root, _record(task_id, verdict, commits[reviewed]["tree"], parent))["record_id"]
            for landed, subject, disposition, reason in landings:
                if subject:  # the author's decision is noted before Git, its landing after Git succeeded
                    rl.note_author_decision(root, rid, {
                        "disposition": disposition, "rationale": REASONS[reason], "subject_hash": subject,
                        "reviewer_signal": rl.load_record(root, rid)["verdict"]["aggregate"], "enforcement": "advisory",
                        "source": "author", "recorded_at": "2026-10-28T14:09:02+00:00"})
                rl.note_commit_landing(root, rid, {"kind": "commit_landing", "landed": True, "subject_hash": subject,
                                                   "pre_review_fingerprint": subject, "post_review_fingerprint": subject}
                                       if subject else None,
                                       {"commit_sha": sha[landed], "tree_sha": "" if scene == "unknown" else commits[landed]["tree"],
                                        "parent_sha": sha["base"] if landed == "early" else sha["early"],
                                        "author_subject": subject})
            if scene == "unavailable":
                rl.record_path(root, rid).unlink()  # the task's index names a record whose file is gone
            task = {"id": task_id, "type": "task", "chat_id": 1}
            write_task_result(root, task_id, STATUS_SCHEDULED, chat_id=1, root_task_id=task_id,
                              suggested_name=NAME, result="Task accepted and scheduled.")
            _store_task_result(env, task, f"Готово: фокус возвращается в поле ввода ({scene}).",
                               {"rounds": 31, "cost": 0}, {"tool_calls": []})
            _record_task_facts(env, task, {"rounds": 31}, {"tool_calls": []}, root / "logs")
        adopted = ("cand", "early") if scene == "nocard" else FACTS[scene][3]
        if scene in LATE:  # its first boot settled the adoption unconfirmed: the row is the card's revision 1
            message_bus.send_with_budget(1, f"♻️ Restart requested by agent: adopt reviewed body candidate {sha['cand'][:12]}",
                                         role="system", system_type="restart_notice")
            body_adoption._publish_on_task(root, history, {
                "task_id": task_id, "adoption_id": f"adoption-{scene}", "outcome": "unconfirmed", "candidate": sha["cand"],
                "previous": sha["early"], "branch": "ouroboros", "card_row_revision": 1}, WAITING.format(cand=sha["cand"][:12]))
            published.append(scene)
            continue
        if adopted is None:
            continue  # adopted only once the page is open
        head, prev = (sha[name] for name in adopted)
        message_bus.send_with_budget(1, f"♻️ Restart requested by agent: adopt reviewed body candidate {head[:12]}",
                                     role="system", system_type="restart_notice")
        body_adoption._publish_on_task(root, history, {
            "task_id": task_id, "adoption_id": f"adoption-{scene}", "outcome": "adopted", "candidate": head,
            "previous": prev, "branch": "ouroboros"}, _note(head, prev, scene in ("pass", "revised", "unavailable", "nocard")))
        published.append(scene)
    for event in events:
        delivery._handle_send_message(dict(event), ctx)
    assert not errors and len(events) == len(published)
    fixture["start_server"]()
    live = [frame for frame in frames if frame.get("type") == "chat" and frame.get("card_row_id")]
    assert len(live) == len(published), live
    return {"sha": sha, "events": events, "frames": frames, "live": live, "published": published, "ctx": ctx,
            "history": history, "root": root}


def _publish_live(seeded, page, monkeypatch, scene, *, lose_fact=False, revision=None):
    """The next boot's adoption of ``scene``'s commit, through the real publisher and send handler, with the page open.

    ``lose_fact`` makes the task-result write fail while the outbox publication still succeeds; ``revision`` is the
    card row's revision that boot's handoff gave it, after a wait it kept. The frames the bridge broadcast reach the
    page's own socket; the page then reads the task again from the running server.
    """
    from ouroboros import body_adoption, task_store
    from supervisor import events_chat_delivery as delivery

    events, frames, sha = seeded["events"], seeded["frames"], seeded["sha"]
    events.clear()
    before, real = len(frames), task_store.patch_document

    def lost(*_a, **_kw):
        raise OSError("task store write lost")

    if lose_fact:
        monkeypatch.setattr(task_store, "patch_document", lost)
    try:
        body_adoption._publish_on_task(seeded["root"], seeded["history"], {
            "task_id": f"adopt-{scene}", "adoption_id": f"adoption-{scene}", "outcome": "adopted",
            "candidate": sha["cand"], "previous": sha["early"], "branch": "ouroboros",
            **({"card_row_revision": revision} if revision else {})}, _note(sha["cand"], sha["early"], False))
    finally:
        monkeypatch.setattr(task_store, "patch_document", real)
    assert len(events) == 1, events
    delivery._handle_send_message(dict(events[0]), seeded["ctx"])
    sent = [frame for frame in frames[before:] if frame.get("type") == "chat"]
    assert [frame.get("card_row_id") for frame in sent] == [f"body-adoption:adoption-{scene}"], sent
    with page.expect_response(lambda response: response.url.endswith(f"/api/tasks/adopt-{scene}")
                              and response.request.method == "GET", timeout=15000):
        for frame in sent:
            page.evaluate(EMIT, frame)
    page.evaluate(SETTLE)
    page.wait_for_timeout(500)  # the read's answer is applied after it arrives; a lost fact changes nothing to wait for
    return sent


def _open_reviews(page, scene, words):
    """Open the card's Reviews door and its one row; wait until the relation says ``words``."""
    card = page.locator(f'.chat-live-card[data-task-id="adopt-{scene}"]')
    card.scroll_into_view_if_needed()
    door = card.locator("[data-live-review-summary]")
    door.wait_for(state="visible", timeout=15000)
    door.click()
    group = card.locator("[data-review-group]").first
    group.wait_for(state="visible")
    toggle = group.locator("[data-review-group-toggle]")
    toggle.click()
    _wait_relation(page, scene, words)
    return card, door, group, toggle


def _wait_relation(page, scene, words):
    page.wait_for_function("""([task, words]) => document.querySelector(
        `.chat-live-card[data-task-id="${task}"] [data-review-landing]`)?.textContent.includes(words)""",
                           arg=[f"adopt-{scene}", words], timeout=15000)


def _relation_geometry(page, scene):
    """The relation is read whole: no clamp and every line of it inside its box; its reason may be cut."""
    base = f'.chat-live-card[data-task-id="adopt-{scene}"] [data-review-landing]'
    relation = page.evaluate(GEOMETRY, f"{base} .review-line-whole")
    assert relation and relation["clamp"] in ("none", "") and relation["display"] == "block", relation
    assert relation["scrollHeight"] <= relation["clientHeight"] + 1, relation  # never cut
    assert relation["textBottom"] <= relation["boxBottom"] + 1, relation  # no line past its own box
    return {"relation": relation, "reason": page.evaluate(GEOMETRY, f"{base} .review-line-quiet")}


def test_adoption_lives_in_its_task_card_and_the_review_row_says_the_relation(direct_server_with_data, monkeypatch,
                                                                              tmp_path):
    from playwright.sync_api import sync_playwright

    seeded = seed(direct_server_with_data, monkeypatch, tmp_path)
    shas = {name: value[:12] for name, value in seeded["sha"].items()}
    url = direct_server_with_data["url"]
    evidence = EVIDENCE or Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "adoption-card-evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as pw:
        for engine, theme, width, height in COMBOS:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=theme)
                errors = PageErrors(None, engine)
                errors.watch(page, navigates=False)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.goto(url, wait_until="domcontentloaded")
                page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
                tag = f"{engine}-{width}-{theme}"
                feed = page.locator("#chat-messages")
                page.locator('.chat-live-card[data-task-id="adopt-pass"]').wait_for(state="visible", timeout=30000)
                # No standalone adoption bubble beside a loaded card; the task-unbound restart request stays.
                assert "Body adoption" not in feed.text_content()
                restarts = feed.locator('.chat-bubble.system[data-system-type="restart_notice"]')
                # One task-unbound restart request per seeded adoption stays a System row (far rows sleep: textContent).
                restart_words = restarts.all_text_contents()
                assert len(restart_words) == len(seeded["published"]), restart_words
                # The card-less row stands alone as a System row; no card is invented for it, and no task whose
                # card is loaded has a standalone adoption bubble beside it.
                assert page.locator('.chat-live-card[data-task-id="adopt-nocard-older"]').count() == 0
                alone = feed.locator('.chat-bubble.system[data-system-type="body_adoption"]')
                assert alone.count() == 1 and alone.get_attribute("data-task-id") == "adopt-nocard-older", alone.count()
                assert f"Adopted candidate commit {shas['cand']}" in alone.text_content()  # far rows sleep: innerText is ''
                alone.scroll_into_view_if_needed()
                page.evaluate(SETTLE)
                page.screenshot(path=str(evidence / f"{tag}-nocard-feed.png"))
                facts, geometry = {}, {}
                narrow = width <= 390
                for scene in SCENES[:-1] + (NARROW if narrow else ()):
                    card = page.locator(f'.chat-live-card[data-task-id="adopt-{scene}"]')
                    card.scroll_into_view_if_needed()
                    assert card.get_attribute("data-expanded") == "0"
                    door = card.locator("[data-live-review-summary]")
                    door.wait_for(state="visible", timeout=15000)
                    if scene == "unavailable":
                        door.click()
                        card.locator("[data-review-group]").first.wait_for(state="visible")
                        head = card.locator("[data-review-group-toggle]").first.inner_text()
                        assert "could not be read" in head and card.locator("[data-review-landing]").count() == 0
                    else:
                        # The relation needs the task's adoption read beside the record: wait for the hydrated words.
                        card, door, group, toggle = _open_reviews(page, scene, RELATION[scene].format(**shas))
                        head = toggle.inner_text()
                        assert ("earlier version" in head) == (scene == "revised"), head
                        assert STATE[scene] in head, head
                        geometry[scene] = _relation_geometry(page, scene)
                        reason = card.locator("[data-review-landing] .review-line-quiet")
                        _landed, governed, _disposition, key = FACTS[scene][2][0]  # the landing the relation names
                        assert reason.count() == (1 if governed else 0), scene
                        if governed:  # that landing's own decision, never a later one on the same record
                            assert reason.text_content().startswith(f"Reason: {REASONS[key][:60]}"), scene
                        page.evaluate(SETTLE)
                        card.screenshot(path=str(evidence / f"{tag}-{scene}-reviews.png"))
                        door.evaluate(ANCHOR)
                        page.evaluate(SETTLE)
                        page.screenshot(path=str(evidence / f"{tag}-{scene}-reviews-page.png"))
                        if reason.count():
                            group.locator("[data-review-facts] summary").click()
                            facts_text = " ".join(group.locator(".review-facts-body").inner_text().split())
                            for _landed, subject, _disposition, key in FACTS[scene][2]:
                                assert not subject or REASONS[key] in facts_text, (scene, key)  # each landing's whole reason
                            facts[scene] = {"facts_has_every_reason": True}
                    if scene in SCENES:
                        # Details: the adoption is the task's own row in its card.
                        card.locator("[data-live-disclosure]").click()
                        row = card.locator(".chat-live-line", has_text=f"Adopted candidate commit {shas['cand']}")
                        row.wait_for(state="visible", timeout=10000)
                        assert row.count() == 1
                        row.scroll_into_view_if_needed()
                        page.evaluate(SETTLE)
                        card.screenshot(path=str(evidence / f"{tag}-{scene}-details.png"))
                        assert card.evaluate("el => el.scrollWidth <= el.clientWidth + 1"), scene
                        card.locator("[data-live-disclosure]").click()
                    door.click()
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
                assert feed.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
                if (engine, theme, width) == COMBOS[0][:3]:
                    # Live: the very frame the bridge broadcast, replayed after history painted, joins its keyed row.
                    for frame in seeded["live"]:
                        _emit_ws_frame(page, frame)
                    page.evaluate(SETTLE)
                    assert "Body adoption" not in feed.text_content()
                    assert feed.locator('.chat-bubble[data-system-type="body_adoption"]').count() == 1
                    for scene in SCENES[:-1]:
                        card = page.locator(f'.chat-live-card[data-task-id="adopt-{scene}"]')
                        card.locator("[data-live-disclosure]").click()
                        assert card.locator(".chat-live-line", has_text=f"Adopted candidate commit {shas['cand']}").count() == 1, scene
                        card.locator("[data-live-disclosure]").click()
                    # Cold reload: the same history rebuilds the same card, relation and single row.
                    errors.watch(page)  # Only this deliberate reload admits the existing WebKit exception.
                    page.reload(wait_until="domcontentloaded")
                    page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
                    page.locator('.chat-live-card[data-task-id="adopt-rejected"]').wait_for(state="visible", timeout=30000)
                    _open_reviews(page, "rejected", RELATION["rejected"].format(**shas))
                    assert "Body adoption" not in feed.text_content()
                assert not errors.unexpected(), errors.report()
                results.append({"engine": engine, "theme": theme, "width": width, "height": height, "facts": facts,
                                "geometry": geometry, "restart_rows": restart_words, "page_errors": list(errors),
                                "page_error_classification": errors.report()})
                page.close()
            finally:
                browser.close()
        results.append({"incremental": _incremental(pw, seeded, monkeypatch, url, shas, evidence)})
    (evidence / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"VISUAL_EVIDENCE {evidence}")


def _answered_now(route):
    """The server's answer to the page's request at the moment it was asked, kept to be delivered later."""
    response = route.fetch()
    return route, {"status": response.status, "headers": response.headers, "body": response.body()}


def _late_history(seeded, page, monkeypatch, evidence, committed, adopted):
    """A history read answered while the adoption was unconfirmed lands after the card read the adoption (the boot
    serves before it settles). The read is the page's own, after a reconnect, answered by the running server when
    asked and delivered only after the adoption's row arrived and the card re-read its task."""
    card = _open_reviews(page, "delayed", committed)[0]
    relation, toggle = card.locator("[data-review-landing]"), card.locator("[data-review-group-toggle]").first
    relation.evaluate("el => { el.__kept = true; }")  # the opened row's own node, through every read below
    holding, held = [True], []

    def hold(route):
        if holding[0]:
            held.append(_answered_now(route))
        else:
            route.continue_()

    page.route("**/api/chat/history**", hold)
    sockets = page.evaluate("() => window.__testSockets.length")
    page.evaluate("() => window.__testSockets.at(-1).close()")
    page.wait_for_function("n => window.__testSockets.length > n && window.__testSockets.at(-1).readyState === 1",
                           arg=sockets, timeout=60000)
    for _ in range(300):  # the reconnect's own read reaches the server before the adoption is published
        if held:
            break
        page.wait_for_timeout(100)
    assert held, "the page's reconnect read never reached the server"
    _publish_live(seeded, page, monkeypatch, "delayed", revision=2)
    _wait_relation(page, "delayed", adopted)
    holding[0], late = False, list(held)
    applied = page.evaluate("() => window.__historyReadsApplied")
    for route, answer in late:
        route.fulfill(**answer)
    page.wait_for_function("n => window.__historyReadsApplied >= n", arg=applied + len(late), timeout=15000)
    page.unroute("**/api/chat/history**", hold)
    page.evaluate(SETTLE)
    page.wait_for_timeout(300)
    words = relation.text_content()
    kept = relation.evaluate("el => el.__kept === true && el.isConnected")
    page.evaluate(SETTLE)
    card.screenshot(path=str(evidence / "webkit-390-light-delayed-after-late-history.png"))
    assert words.startswith(adopted) and kept and toggle.get_attribute("aria-expanded") == "true", (words, kept)
    return {"late_reads": len(late), "words": words, "same_node": kept}


def _incremental(pw, seeded, monkeypatch, url, shas, evidence):
    """An opened card that says ``Committed`` turns ``Adopted`` when the adoption arrives, without a reload; one
    whose task fact was lost gets its row and keeps ``Committed``; a late history read cannot take an adopted one
    back. A cold reload then says the same."""
    browser = pw.webkit.launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844}, color_scheme="light")
        errors = PageErrors(None, "webkit")
        errors.watch(page, navigates=False)
        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
        page.add_init_script(f"({COUNT_HISTORY_READS})()")
        page.goto(url, wait_until="domcontentloaded")
        page.locator('.chat-live-card[data-task-id="adopt-late"]').wait_for(state="visible", timeout=30000)
        committed, adopted = RELATION["late"].format(**shas), f"Adopted {shas['cand']}: the exact bytes this review passed."
        cards = {scene: _open_reviews(page, scene, committed)[0] for scene in ("late", "lost")}
        page.evaluate("window.__sameDocument = true")  # a reload would drop it
        cards["late"].screenshot(path=str(evidence / "webkit-390-light-late-before.png"))

        _publish_live(seeded, page, monkeypatch, "late")
        _wait_relation(page, "late", adopted)  # the opened row says it in place, no reload
        geometry = _relation_geometry(page, "late")
        assert cards["late"].locator("[data-review-group-toggle]").first.get_attribute("aria-expanded") == "true"
        page.evaluate(SETTLE)
        cards["late"].screenshot(path=str(evidence / "webkit-390-light-late-after.png"))

        _publish_live(seeded, page, monkeypatch, "lost", lose_fact=True)
        lost_words = page.locator('.chat-live-card[data-task-id="adopt-lost"] [data-review-landing]').text_content()
        assert lost_words.startswith(committed) and "Adopted" not in lost_words, lost_words  # nothing recorded, nothing named
        cards["lost"].locator("[data-live-disclosure]").click()
        row = cards["lost"].locator(".chat-live-line", has_text=f"Adopted candidate commit {shas['cand']}")
        row.wait_for(state="visible", timeout=10000)
        page.evaluate(SETTLE)
        cards["lost"].screenshot(path=str(evidence / "webkit-390-light-lost-after.png"))
        assert page.evaluate("window.__sameDocument === true") and not errors.unexpected(), errors.report()
        delayed = _late_history(seeded, page, monkeypatch, evidence, committed, adopted)
        assert page.evaluate("window.__sameDocument === true") and not errors.unexpected(), errors.report()

        errors.watch(page)
        page.reload(wait_until="domcontentloaded")
        page.locator('.chat-live-card[data-task-id="adopt-late"]').wait_for(state="visible", timeout=30000)
        _open_reviews(page, "late", adopted)
        _open_reviews(page, "lost", committed)
        _open_reviews(page, "delayed", adopted)
        cold = page.locator('.chat-live-card[data-task-id="adopt-lost"] [data-review-landing]').text_content()
        assert "Adopted" not in cold, cold
        page.evaluate(SETTLE)
        page.locator('.chat-live-card[data-task-id="adopt-late"]').screenshot(
            path=str(evidence / "webkit-390-light-late-reload.png"))
        assert not errors.unexpected(), errors.report()
        page.locator('.chat-live-card[data-task-id="adopt-delayed"]').screenshot(
            path=str(evidence / "webkit-390-light-delayed-reload.png"))
        return {"late": adopted, "lost": lost_words, "lost_after_reload": cold, "geometry": geometry, "delayed": delayed,
                "page_errors": list(errors), "page_error_classification": errors.report()}
    finally:
        browser.close()

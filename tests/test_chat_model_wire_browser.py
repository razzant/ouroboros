"""A room's model, Main's own write and a work's own model: from the page to the provider's wire and back.

The real server on its own port and data root, the real page, and a loopback provider that stands for the models
(``fixtures_mock_llm``): no paid call and no account login is possible here. Nothing of the server is answered by
the test. The provider lists its models on ``GET /v1/models``, so the real ``/api/model-catalog`` suggests them, and
it can hold a marked request, so a work stays running (and another queued behind it) while the page is read.

One row is written by the test, through the dialogue store's own writer: a task created over ``POST /api/tasks``
says nothing in its room, so a queued one has no card there. In the product a queued root's card comes from the
frame the work that scheduled it sends (``events_schedule_task``: its line and the ``cancelable`` marker); the test
appends that row for the task the server admitted, and everything about the task itself is the server's.

What is proved end to end (owner 10.11.2026: a room's model stays with its room; a pick in the Main chat writes
Main itself after a confirming sheet; a queued work's model is changed before it starts; the recent models are the
server's):

- a pick in a Project room's capsule is saved for that room, the next Send reaches the provider on that model, the
  running card says whose model it is, and a reload keeps the room's choice;
- the Main chat's capsule changes Main through the sheet, the settings document holds it, and the next work runs
  on it;
- a queued work is moved from its card before it starts (``at_start``) and then runs on the model it was given;
- the recent models are what the picks were, newest first, the same on a second page load.
"""
from __future__ import annotations

import json
import os
import pathlib
import re

import pytest

from tests import fixtures_mock_llm
from tests._model_choice_browser import (
    _CAPSULE, _CARD, AT_START, MAIN, OTHER, PICKED, QUEUED, SOURCE, Provider,
    _announce, _answered, _asked, _capsule_ready, _close_project, _idle, _open_project,
    _post, _preferences, _search_pick, _send, _state, _work_of,
)
from tests.test_effort_range_browser import _launch
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 -- fixture

pytestmark = [pytest.mark.ui_browser]  # its server, data root and provider are its own: nothing on the host is shared

def test_a_rooms_model_mains_write_and_a_queued_works_model_reach_the_provider(direct_server_with_data, monkeypatch, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    server["stop_server"]()
    settings_path = server["data_dir"] / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(OUROBOROS_MODEL=MAIN, OUROBOROS_MODEL_LIGHT=f"{SOURCE}::mock-light",
                    OUROBOROS_MODEL_FALLBACKS=f"{SOURCE}::mock-fallback",
                    OUROBOROS_TASK_REVIEW_MODE="off", OUROBOROS_SAFETY_MODE="off",
                    OUROBOROS_SUBAGENTS=json.dumps({"enabled": False, "items": []}))
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    provider = Provider()
    # Plain functions, so the handler class binds each request's handler as their argument.
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", lambda handler: provider.post(handler))
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_GET", lambda handler: provider.get(handler), raising=False)
    server["start_server"]()
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    seen: dict = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, "webkit", viewport={"width": 1512, "height": 949})
        writes: list = []
        page.on("request", lambda request: writes.append((request.url.split("/api/", 1)[1], json.loads(request.post_data or "{}")))
                if request.method == "POST" and ("/api/owner/chat-model" in request.url or re.search(r"/api/tasks/[^/]+/model$", request.url)) else None)
        try:
            page.goto(server["url"], wait_until="domcontentloaded")
            _capsule_ready(page, "#chat-model-mode")
            assert _state(page)["main_route"] == {"model": MAIN, "use_local": False}
            assert _preferences(page)["model_recents"] == [], "no model was chosen yet"

            # --- 1. A Project room: its capsule follows Main until it chooses; a pick is saved for the room at once.
            alpha = _post(page, "/api/projects", {"name": "Wire alpha"})["project"]
            beta = _post(page, "/api/projects", {"name": "Wire beta"})["project"]
            prefix = _open_project(page, alpha)
            capsule = f"#{prefix}-model-mode"
            closed = page.evaluate(_CAPSULE, capsule)
            assert (closed["name"], closed["value"], closed["own"]) == ("Main", "mock-main", "false"), closed
            _search_pick(page, capsule, PICKED)
            page.wait_for_function("(sel) => document.querySelector(sel).dataset.own === 'true'", arg=capsule, timeout=20_000)
            picked = page.evaluate(_CAPSULE, capsule)
            assert picked["value"] == "mock-picked" and picked["name"] != "Main", picked
            assert writes[-1] == ("owner/chat-model", {"kind": "model", "model": PICKED, "use_local": False, "project_id": alpha["id"]}), writes
            state = _state(page)
            rooms = {row["id"]: row.get("model_choice") for row in state["projects"]}
            assert rooms[alpha["id"]] == {"kind": "model", "model": PICKED, "use_local": False}, rooms
            assert rooms[beta["id"]] == {"kind": "main"}, "a choice stays with its room"
            assert state["main_route"]["model"] == MAIN, "a room's pick never writes Main"

            # --- 2. Send: the provider is asked for the room's model; the running card says whose model it is.
            provider.hold("ROOM_WIRE_ALPHA")
            _send(page, prefix, "ROOM_WIRE_ALPHA")
            card = f"#{prefix}-messages .chat-live-card"
            assert provider.held("ROOM_WIRE_ALPHA"), "the room's work never reached its held call"
            page.wait_for_function("(sel) => document.querySelector(sel + ' .card-model-fact')?.innerText.includes('mock-picked')",
                                   arg=card, timeout=60_000)
            running = page.evaluate(_CARD, card)
            seen["running"] = running
            assert provider.models_of("ROOM_WIRE_ALPHA")[:1] == ["mock-picked"], provider.models_of("ROOM_WIRE_ALPHA")
            assert running["why"] == "this room's model" and running["basis"] == "own", running
            assert running["fact"].endswith("· mock-picked"), running
            state = _state(page)
            work = _work_of(state, running["task"])
            seen["running_work"] = work
            assert work and work["lead"] == {"model": PICKED, "use_local": False, "basis": "room"}, work
            assert {"action": "model", "target": running["task"]} in work["controls"], work["controls"]
            assert running["tag"] == "BUTTON", "the server offers the control: the fact is pressable"

            # --- 3. A queued work is moved from its card before it starts. Project work is bounded by the one worker:
            #        a held task keeps it, so the next one stays queued behind it.
            provider.hold("ROOM_WIRE_GAMMA")
            _post(page, "/api/tasks", {"description": "ROOM_WIRE_GAMMA hold the worker", "project_id": beta["id"], "title": "Gamma"})
            assert provider.held("ROOM_WIRE_GAMMA"), "the first task never reached its held call"
            delta = _post(page, "/api/tasks", {"description": "ROOM_WIRE_DELTA wait for the worker", "project_id": beta["id"], "title": "Delta"})["task_id"]
            page.wait_for_function("async (id) => { const s = await (await fetch('/api/state', { cache: 'no-store' })).json();"
                                   " return s.works.some((w) => w.episode.task_id === id && w.activity === 'queued'); }", arg=delta, timeout=60_000)
            queued_work = _work_of(_state(page), delta)
            seen["queued_work"] = queued_work
            assert queued_work["lead"] == {"model": MAIN, "use_local": False, "basis": "main"}, queued_work
            assert {"action": "model", "target": delta} in queued_work["controls"], queued_work["controls"]
            _announce(page, server["data_dir"], _state(page), delta, "Queued: waiting for the worker")
            beta_prefix = _open_project(page, beta)
            page.reload(wait_until="domcontentloaded")
            _capsule_ready(page, "#chat-model-mode")
            beta_prefix = _open_project(page, beta)
            beta_card = f'#{beta_prefix}-messages .chat-live-card[data-task-id="{delta}"]'
            page.wait_for_function("(sel) => Boolean(document.querySelector(sel + ' button.card-model-fact'))", arg=beta_card, timeout=60_000)
            queued = page.evaluate(_CARD, beta_card)
            seen["queued"] = queued
            assert queued["chip"] == "Queued" and queued["fact"].endswith("· mock-main") and queued["why"] == "", queued
            assert provider.models_of("ROOM_WIRE_DELTA") == [], "a queued work has asked no model yet"
            page.click(f"{beta_card} button.card-model-fact")
            page.wait_for_selector(".card-model-pop .chat-model-row", timeout=20_000)
            popover = page.evaluate("""() => ({ title: document.querySelector('.card-model-pop-title').innerText,
                acts: document.querySelectorAll('.card-model-pop .card-model-act').length,
                main: document.querySelector('.card-model-pop .chat-model-row').innerText.replace(/\\s+/g, ' ').trim(),
                rule: document.querySelector('.card-model-pop .chat-model-rule').innerText })""")
            seen["queued_popover"] = popover
            assert popover["title"] == "Model for this work · follows Main" and popover["acts"] == 0, popover
            assert popover["main"] == "mock-main Main this work follows Main" and popover["rule"] == AT_START, popover
            page.click(f'.card-model-pop .chat-model-row[data-row="route|{QUEUED}"]')
            page.wait_for_function("(words) => [...document.querySelectorAll('.toast')].some((toast) => toast.innerText.includes(words))",
                                   arg=AT_START, timeout=20_000)
            task_write = writes[-1]
            assert task_write[0] == f"tasks/{delta}/model", writes
            assert {key: value for key, value in task_write[1].items() if key != "request_id"} == {"action": "set", "model": QUEUED, "use_local": False}
            assert task_write[1]["request_id"], task_write
            page.wait_for_function("(sel) => document.querySelector(sel + ' .card-model-fact').innerText.includes('mock-queued')",
                                   arg=beta_card, timeout=20_000)
            moved = page.evaluate(_CARD, beta_card)
            assert moved["why"] == "you chose it" and not moved["pending"], moved
            moved_work = _work_of(_state(page), delta)
            assert moved_work["lead"] == {"model": QUEUED, "use_local": False, "basis": "owner"}, moved_work

            # The held works finish on their own models; the queued one starts on the model it was given.
            seen["before_release"] = [(w["name"], w["activity"], w["live"]) for w in _state(page)["works"]]
            provider.release("ROOM_WIRE_ALPHA")
            provider.release("ROOM_WIRE_GAMMA")
            _asked(page, provider, "ROOM_WIRE_DELTA")
            _idle(page)
            assert provider.models_of("ROOM_WIRE_DELTA") == ["mock-queued"], provider.models_of("ROOM_WIRE_DELTA")
            assert set(provider.models_of("ROOM_WIRE_GAMMA")) == {"mock-main"}, "a follower ran on Main"
            assert set(provider.models_of("ROOM_WIRE_ALPHA")) == {"mock-picked"}, provider.models_of("ROOM_WIRE_ALPHA")
            assert _state(page)["projects"] and {row["id"]: row.get("model_choice") for row in _state(page)["projects"]}[beta["id"]] == {"kind": "main"}, \
                "a work's own model is the work's: its room still follows Main"

            # --- 4. Reload: the room keeps its choice, its finished card names the model it ran on, the next Send too.
            page.reload(wait_until="domcontentloaded")
            _capsule_ready(page, "#chat-model-mode")
            prefix = _open_project(page, alpha)
            capsule = f"#{prefix}-model-mode"
            page.wait_for_function("(sel) => document.querySelector(sel).dataset.own === 'true'", arg=capsule, timeout=20_000)
            assert page.evaluate(_CAPSULE, capsule)["value"] == "mock-picked"
            page.wait_for_function("(sel) => document.querySelector(sel + ' [data-live-meta]')?.innerText.includes('mock-picked')",
                                   arg=f"#{prefix}-messages .chat-live-card", timeout=30_000)
            _send(page, prefix, "ROOM_WIRE_RELOADED")
            _answered(page, prefix, "ROOM_WIRE_RELOADED")
            _idle(page)
            assert provider.models_of("ROOM_WIRE_RELOADED") == ["mock-picked"]

            # --- 5. The Main chat's capsule is Main itself: a pick asks first, then writes Main; the next work runs on it.
            _close_project(page)
            _search_pick(page, "#chat-model-mode", OTHER)
            page.wait_for_selector(".confirm-dialog [data-main-followers]", timeout=20_000)
            sheet = page.evaluate("""() => { const d = document.querySelector('.confirm-dialog'); const text = (n) => n.innerText.replace(/\\s+/g, ' ').trim();
                return { title: text(d.querySelector('h3')), row: text(d.querySelector('.chat-model-sheet-row')), count: text(d.querySelector('[data-main-followers]')),
                    rooms: [...d.querySelectorAll('.chat-model-off')].map((row) => [text(row.querySelector('.chat-model-off-name')), row.querySelector('.ui-checkbox')?.checked]),
                    ok: text(d.querySelector('[data-confirm-ok]')) }; }""")
            seen["sheet"] = sheet
            assert sheet["title"] == "Change Main" and "mock-other" in sheet["row"] and "was mock-main" in sheet["row"], sheet
            assert sheet["count"] == "No running work follows Main now.", sheet
            assert sheet["rooms"] == [["Wire alpha", False]] and sheet["ok"] == "Change Main", "the room with its own model is listed, unticked"
            assert json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_MODEL"] == MAIN, "nothing is written before he confirms"
            page.click(".confirm-dialog [data-confirm-ok]")
            page.wait_for_function("() => document.querySelector('#chat-model-mode .chat-capsule-value').textContent === 'mock-other'", timeout=20_000)
            assert writes[-1] == ("owner/chat-model", {"kind": "model", "model": OTHER, "use_local": False}), writes
            assert _state(page)["main_route"] == {"model": OTHER, "use_local": False}
            saved = json.loads(settings_path.read_text(encoding="utf-8"))
            assert saved["OUROBOROS_MODEL"] == OTHER, saved["OUROBOROS_MODEL"]
            assert (saved["OUROBOROS_MODEL_LIGHT"], saved["OUROBOROS_MODEL_FALLBACKS"]) == \
                (settings["OUROBOROS_MODEL_LIGHT"], settings["OUROBOROS_MODEL_FALLBACKS"]), "only Main was written"
            _send(page, "chat", "ROOM_WIRE_MAIN")
            _answered(page, "chat", "ROOM_WIRE_MAIN")
            _idle(page)
            assert provider.models_of("ROOM_WIRE_MAIN") == ["mock-other"], provider.models_of("ROOM_WIRE_MAIN")
            # A follower follows the new Main; the room with its own model does not.
            beta_prefix = _open_project(page, beta)
            _send(page, beta_prefix, "ROOM_WIRE_FOLLOWER")
            _answered(page, beta_prefix, "ROOM_WIRE_FOLLOWER")
            _idle(page)
            assert provider.models_of("ROOM_WIRE_FOLLOWER") == ["mock-other"]

            # --- 6. The recent models are the server's: what was picked, newest first, the same on a second page load.
            recents = _preferences(page)["model_recents"]
            seen["recents"] = recents
            assert recents == [{"model": OTHER, "use_local": False}, {"model": QUEUED, "use_local": False},
                               {"model": PICKED, "use_local": False}], recents
            _close_project(page)
            for _ in range(2):
                page.reload(wait_until="domcontentloaded")
                _capsule_ready(page, "#chat-model-mode")
                page.click("#chat-model-mode .chat-model-head")
                page.wait_for_function("() => document.querySelectorAll('#chat-model-mode .chat-model-strip .chat-seg').length >= 4", timeout=20_000)
                strip = page.evaluate(_CAPSULE, "#chat-model-mode")
                assert strip["strip"] == ["mock-other", "mock-queued", "mock-picked", "Search"], strip
                assert strip["checked"] == ["mock-other"], strip
                assert _preferences(page)["model_recents"] == recents
        finally:
            provider.release_all()
            (evidence / "wire-seen.json").write_text(json.dumps(seen, indent=2, default=str), encoding="utf-8")
            (evidence / "wire-models.json").write_text(json.dumps(
                [{"model": body.get("model"), "marker": provider.marker_of(body)} for body in provider.received]), encoding="utf-8")
            (evidence / "wire-writes.json").write_text(json.dumps(writes, indent=2), encoding="utf-8")
            for name in ("events.jsonl", "supervisor.jsonl"):
                log = server["data_dir"] / "logs" / name
                if log.exists():
                    (evidence / f"wire-{name}").write_text("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]), encoding="utf-8")
            try:
                (evidence / "wire-state.json").write_text(json.dumps(_state(page), indent=1, default=str), encoding="utf-8")
                (evidence / "wire-page.txt").write_text(page.locator("body").inner_text(), encoding="utf-8")
                page.screenshot(path=str(evidence / "wire-last.png"))
            except Exception:  # noqa: BLE001 - diagnostics only: the page may be gone
                pass
            browser.close()

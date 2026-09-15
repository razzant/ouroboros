"""Browser proof for #895: two Main root cards born from ONE owner message convert into ONE Project.

Seeds two completed managed roots (T1, T2) carrying the same ingress origin, as Main root
cards. Converting T1 creates the Project; converting T2 must ADOPT it (same chip name, one
registry row), never mint a second one. A direct conversation turn keeps no conversion
control (docs/DESIGN.md "Conversation activity block": conversion is the model's addressing
call, never a card button), so both cards here are managed roots — the case the origin key
exists for.

Run: OUROBOROS_RUN_UI_SMOKE=1 OUROBOROS_DATA_DIR=$(mktemp -d) python -m pytest \
  tests/test_ui_smoke_origin_adopt.py -o addopts="" -m ui_browser -q
Screenshots land in $OUROBOROS_UI_SHOTS (default: the test's tmp dir).
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401

pytestmark = pytest.mark.ui_browser

ORIGIN_TEXT = "token-usage - submit and merge\ncontext-lens - merge"


def _wait_status(page, expected, timeout=10_000):
    page.wait_for_function(
        "(exp) => (document.querySelector('#chat-status')?.textContent || '').trim() === exp",
        arg=expected, timeout=timeout,
    )


def _seed(data_dir: pathlib.Path) -> dict:
    from ouroboros.project_dialogue import _text_sha256
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl

    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ref = {
        "chat_id": 1, "client_message_id": "msg-origin-1",
        "ts": "2026-09-14T12:15:20.535950+00:00", "text_sha256": _text_sha256(ORIGIN_TEXT),
    }
    append_jsonl(logs / "chat.jsonl", {
        "ts": ref["ts"], "direction": "in", "chat_id": 1, "user_id": 1,
        "text": ORIGIN_TEXT, "client_message_id": ref["client_message_id"],
    })
    # T1 and T2: two completed managed roots admitted from the same owner message.
    for tid, title, first in (("t1direct", "Draft the plan", True), ("t2promoted", "Publish seven skills", False)):
        write_task_result(
            data_dir, tid, "completed", result="done", description=ORIGIN_TEXT, objective=ORIGIN_TEXT,
            chat_id=1, title=title, origin_message_ref=dict(ref), origin_message_text=ORIGIN_TEXT,
            delegation_role="root", root_task_id=tid,
        )
        append_jsonl(logs / "progress.jsonl", {
            "ts": "2026-09-14T12:30:00+00:00" if first else "2026-09-14T12:38:00+00:00",
            "type": "send_message", "direction": "out", "chat_id": 1, "user_id": 1,
            "task_id": tid, "is_progress": True, "content": f"Working on {title}", "text": f"Working on {title}",
        })
        append_jsonl(logs / "chat.jsonl", {
            "ts": "2026-09-14T12:39:31+00:00" if first else "2026-09-14T12:41:00+00:00",
            "direction": "system", "chat_id": 1, "user_id": 1, "type": "task_summary",
            "summary_kind": "terminal_root_projection", "summary_id": f"task-terminal:{tid}",
            "task_id": tid, "root_task_id": tid, "status": "completed", "outcome": "Done",
            "outcome_phase": "done", "outcome_final": True, "text": f"Done. {title}",
            "tool_calls": 1, "rounds": 2,
        })
    return ref


def test_ui_two_cards_one_origin_convert_into_one_project(direct_server_with_data, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = pathlib.Path(direct_server_with_data["data_dir"])
    _seed(data_dir)
    shots = pathlib.Path(os.environ.get("OUROBOROS_UI_SHOTS") or tmp_path)
    shots.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            _wait_status(page, "Online", timeout=30_000)
            c1 = page.locator('#chat-messages .chat-live-card[data-task-id="t1direct"]')
            c2 = page.locator('#chat-messages .chat-live-card[data-task-id="t2promoted"]')
            c1.wait_for(state="visible", timeout=30_000)
            c2.wait_for(state="visible", timeout=30_000)
            assert c1.locator("[data-turn-into-project]").count() == 1
            assert c2.locator("[data-turn-into-project]").count() == 1
            page.screenshot(path=str(shots / "01_two_cards_two_buttons.png"), full_page=True)

            c1.locator("[data-turn-into-project]").first.evaluate("b => b.click()")
            c1.locator(".chat-live-project-card-btn").first.wait_for(
                state="visible", timeout=20_000,
            )
            page.screenshot(path=str(shots / "02_first_converted.png"), full_page=True)
            projects = page.request.get(f"{url}/api/projects").json()["projects"]
            assert len(projects) == 1, projects
            first_name = c1.inner_text().strip().splitlines()[0]

            # The sibling card: its button is still there (this task is completed, so
            # the sibling claim never bound it), and converting it must ADOPT.
            btn = c2.locator("[data-turn-into-project]")
            assert btn.count() == 1
            # JS click: a toast from the first conversion can sit over the button.
            btn.first.evaluate("b => b.click()")
            c2.locator(".chat-live-project-card-btn").first.wait_for(
                state="visible", timeout=20_000,
            )
            page.screenshot(path=str(shots / "03_second_adopted.png"), full_page=True)
            projects = page.request.get(f"{url}/api/projects").json()["projects"]
            assert len(projects) == 1, [p.get("id") for p in projects]
            second_name = c2.inner_text().strip().splitlines()[0]
            assert first_name == second_name, (first_name, second_name)

            # A card that is already bound can STILL have its button on screen (a phone
            # that missed the refresh, the Telegram mini app, a second tab). Replay the
            # exact request the client sends — api_client.js::projectFromTask POSTs
            # /api/projects/from-task with the default id chat_activity.js derives — and
            # the server adopts that Project instead of minting one or answering 4xx.
            replay = page.request.post(f"{url}/api/projects/from-task", data={
                "task_id": "t2promoted", "id": "task-t2promoted",
                "name": "", "objective_hint": ORIGIN_TEXT,
            })
            assert replay.status == 200, replay.text()
            replayed = replay.json()
            assert replayed["adopted"] is True, replayed
            assert replayed["project"]["id"] == projects[0]["id"]
            assert len(page.request.get(f"{url}/api/projects").json()["projects"]) == 1

            bindings = json.loads((data_dir / "state" / "project_task_bindings.json").read_text())["bindings"]
            assert bindings["t1direct"]["project_id"] == bindings["t2promoted"]["project_id"]

            page.reload(wait_until="domcontentloaded")
            _wait_status(page, "Online", timeout=30_000)
            page.screenshot(path=str(shots / "04_after_reload.png"), full_page=True)
            assert page.locator('#chat-messages [data-turn-into-project]').count() == 0
        finally:
            browser.close()


def _seed_live_sibling(data_dir: pathlib.Path) -> None:
    """T1 = a completed managed root (card in Main); T2 = a LIVE queued root with the same origin,
    parked by a replay-safe budget pause so assignment never dispatches it (the same row shape
    ``test_budget_pause_v664.py`` restores). Both render as Main root cards with the convert button."""
    import datetime as _dt

    from ouroboros.project_dialogue import _text_sha256
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl

    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (data_dir / "state").mkdir(parents=True, exist_ok=True)
    ref = {
        "chat_id": 1, "client_message_id": "msg-origin-live",
        "ts": "2026-09-14T12:15:20.535950+00:00", "text_sha256": _text_sha256(ORIGIN_TEXT),
    }
    append_jsonl(logs / "chat.jsonl", {
        "ts": ref["ts"], "direction": "in", "chat_id": 1, "user_id": 1,
        "text": ORIGIN_TEXT, "client_message_id": ref["client_message_id"],
    })
    write_task_result(
        data_dir, "l1direct", "completed", result="done", description=ORIGIN_TEXT, objective=ORIGIN_TEXT,
        chat_id=1, title="Draft the plan", origin_message_ref=dict(ref), origin_message_text=ORIGIN_TEXT,
        delegation_role="root", root_task_id="l1direct",
    )
    append_jsonl(logs / "progress.jsonl", {
        "ts": "2026-09-14T12:30:00+00:00", "type": "send_message", "direction": "out", "chat_id": 1,
        "user_id": 1, "task_id": "l1direct", "is_progress": True, "content": "Working on Draft the plan",
        "text": "Working on Draft the plan",
    })
    append_jsonl(logs / "chat.jsonl", {
        "ts": "2026-09-14T12:39:31+00:00", "direction": "system", "chat_id": 1, "user_id": 1,
        "type": "task_summary", "summary_kind": "terminal_root_projection", "summary_id": "task-terminal:l1direct",
        "task_id": "l1direct", "root_task_id": "l1direct", "status": "completed", "outcome": "Done",
        "outcome_phase": "done", "outcome_final": True, "text": "Done. Draft the plan", "tool_calls": 1, "rounds": 2,
    })
    live_task = {
        "id": "l2root", "type": "task", "chat_id": 1, "priority": 0, "text": ORIGIN_TEXT,
        "description": ORIGIN_TEXT, "objective": ORIGIN_TEXT, "title": "Publish seven skills",
        "suggested_name": "Publish seven skills", "root_task_id": "l2root", "delegation_role": "root",
        "origin_message_ref": dict(ref), "origin_message_text": ORIGIN_TEXT,
        "_budget_pause": {"status": "paused_before_dispatch", "physical_calls": 0,
                          "replay_safe": True, "auto_resume": False},
    }
    write_task_result(
        data_dir, "l2root", "scheduled", result="Task is queued.", description=ORIGIN_TEXT, objective=ORIGIN_TEXT,
        chat_id=1, title="Publish seven skills", origin_message_ref=dict(ref), origin_message_text=ORIGIN_TEXT,
        delegation_role="root", root_task_id="l2root",
    )
    append_jsonl(logs / "progress.jsonl", {
        "ts": "2026-09-14T12:38:00+00:00", "type": "send_message", "direction": "out", "chat_id": 1,
        "user_id": 1, "task_id": "l2root", "is_progress": True, "content": "Queued: Publish seven skills",
        "text": "Queued: Publish seven skills",
    })
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    (data_dir / "state" / "queue_snapshot.json").write_text(json.dumps({
        "_schema_version": 1, "ts": now, "reason": "ui_smoke_seed",
        "pending_count": 1, "running_count": 0, "reaping_count": 0,
        "acceptance_fences": [], "budget_root_fences": [],
        "pending": [{"id": "l2root", "type": "task", "priority": 0, "attempt": 1, "queued_at": now,
                     "queue_seq": 1, "task": live_task}],
        "running": [],
    }), encoding="utf-8")


def test_ui_live_sibling_loses_button_without_reload(direct_server_with_data, tmp_path):  # noqa: F811
    """Converting the completed root's card binds the LIVE sibling root too: its Main card drops
    the convert button and shows the Project pointer after the state refresh, with no reload."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = pathlib.Path(direct_server_with_data["data_dir"])
    # Seed while no server runs: the live main loop rewrites state/queue_snapshot.json every
    # tick, so a snapshot written under a running server is gone before the next boot restores it.
    direct_server_with_data["stop_server"]()
    _seed_live_sibling(data_dir)
    direct_server_with_data["start_server"]()  # the queue snapshot is restored at boot
    shots = pathlib.Path(os.environ.get("OUROBOROS_UI_SHOTS") or tmp_path)
    shots.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            # No "Online" wait here: a live budget-paused root keeps the chat status busy; the
            # two cards are the gate.
            c1 = page.locator('#chat-messages .chat-live-card[data-task-id="l1direct"]')
            c2 = page.locator('#chat-messages .chat-live-card[data-task-id="l2root"]')
            c1.wait_for(state="visible", timeout=30_000)
            c2.wait_for(state="visible", timeout=30_000)
            assert c1.locator("[data-turn-into-project]").count() == 1
            assert c2.locator("[data-turn-into-project]").count() == 1
            page.screenshot(path=str(shots / "11_live_sibling_two_buttons.png"), full_page=True)

            c1.locator("[data-turn-into-project]").first.evaluate("b => b.click()")
            c1.locator(".chat-live-project-card-btn").first.wait_for(state="visible", timeout=20_000)
            # No reload: the sibling's button must go and a pointer must appear on the next
            # /api/state refresh (projects_changed triggers it; the poll is the backstop).
            try:
                page.wait_for_function(
                    "() => { const c = document.querySelector('#chat-messages .chat-live-card[data-task-id=\"l2root\"]');"
                    " return !!c && !c.querySelector('[data-turn-into-project]') && !!c.querySelector('.chat-live-project-card-btn'); }",
                    timeout=45_000,
                )
            except Exception:
                page.screenshot(path=str(shots / "12_FAIL_live_sibling.png"), full_page=True)
                import shutil
                for rel in ("logs/supervisor.jsonl", "logs/events.jsonl", "state/queue_snapshot.json"):
                    src = data_dir / rel
                    if src.exists():
                        shutil.copy(src, shots / ("12_FAIL_" + rel.replace("/", "_")))
                (shots / "12_FAIL_state.json").write_text(json.dumps({
                    "state": page.request.get(f"{url}/api/state").json(),
                    "card_html": page.evaluate("() => (document.querySelector('#chat-messages .chat-live-card[data-task-id=\"l2root\"]') || {}).outerHTML || null"),
                }, indent=1, default=str)[:20000])
                raise
            page.screenshot(path=str(shots / "12_live_sibling_pointer_no_reload.png"), full_page=True)
            projects = page.request.get(f"{url}/api/projects").json()["projects"]
            assert len(projects) == 1, [p.get("id") for p in projects]
            bindings = json.loads((data_dir / "state" / "project_task_bindings.json").read_text())["bindings"]
            assert bindings["l1direct"]["project_id"] == bindings["l2root"]["project_id"]
            assert not page.locator('.toast.error, .toast-error, [data-toast-tone="error"]').count()
        finally:
            browser.close()

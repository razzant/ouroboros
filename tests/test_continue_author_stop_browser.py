"""An explicit author stop offers the EXISTING Continue, through the real server, worker pool and SPA.

The shared ``direct_server_with_data`` fixture (a copied candidate checkout, a disposable data root,
the in-process mock model) runs one seeded root. The scripted model answers its first main call with
``finish_task(action="stop")``, so the stop, its stored terminal and the card are the product's own.
The settled card shows the ordinary Continue button and nothing starts by itself. The owner's press
admits one successor in the same chat, and after a reload the card points to it. A clean finish on
the same stand offers nothing. Screenshots go to ``OUROBOROS_UI_EVIDENCE_DIR``.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from tests import test_batch4_owner_controls_browser as b4

direct_server_with_data = b4.direct_server_with_data
_keep_checkout_across_restart = b4._keep_checkout_across_restart

pytestmark = [pytest.mark.ui_browser]

STOPPED, FINISHED = "b4-echo", "b4-foxtrot"
RATIONALE = "Not ready: the export endpoint still fails on empty input."


class _StopModel:
    """Each root's main call: ECHO stops with unfinished work, FOXTROT finishes, a successor answers OK."""

    def __init__(self, monkeypatch):
        from tests import fixtures_mock_llm

        self.lock = threading.Lock()
        self.calls: list[tuple[str, bool]] = []
        monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", lambda handler: self.handle(handler))

    def main_calls(self, marker):
        with self.lock:
            return self.calls.count((marker, True))

    def handle(self, handler):
        length = int(handler.headers.get("Content-Length", 0) or 0)
        payload = json.loads(handler.rfile.read(length) or b"{}")
        text = "\n".join(b4._user_text(m) for m in payload.get("messages") or []
                         if isinstance(m, dict) and m.get("role") == "user")
        marker = ("SUCCESSOR" if "[CONTINUE]" in text else
                  next((name for name in ("ECHO", "FOXTROT") if f"{name}-7Q" in text), "OTHER"))
        main = bool(payload.get("tools"))
        with self.lock:
            self.calls.append((marker, main))
        names = {(tool.get("function") or {}).get("name") for tool in payload.get("tools") or [] if isinstance(tool, dict)}
        message, finish = {"role": "assistant", "content": "OK"}, "stop"
        if main and marker in {"ECHO", "FOXTROT"} and "finish_task" in names:
            arguments = ({"action": "stop", "answer": RATIONALE + " Stopping with the work unfinished.",
                          "rationale": RATIONALE} if marker == "ECHO" else {"action": "finish", "answer": "All done."})
            message, finish = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{marker.lower()}", "type": "function",
                "function": {"name": "finish_task", "arguments": json.dumps(arguments)}}]}, "tool_calls"
        b4._reply(handler, payload, message, finish)


def test_an_author_stop_offers_the_existing_continue_and_one_press_admits_one_successor(direct_server_with_data,
                                                                                        monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = b4._evidence_dir(data_dir, "continue-after-author-stop")
    monkeypatch.setitem(b4._TEXT, STOPPED, "ECHO-7Q: ship the export endpoint.")
    monkeypatch.setitem(b4._TEXT, FINISHED, "FOXTROT-7Q: tidy the notes.")
    model = _StopModel(monkeypatch)
    server["stop_server"]()
    b4._seed_roots(data_dir, (STOPPED, FINISHED))
    server["start_server"]()
    b4._resume_after_app_stop(url, STOPPED, FINISHED)
    record: dict = {}
    try:
        with sync_playwright() as pw:
            browser, page, errors = b4._launch(pw)

            def shot(name):
                page.screenshot(path=str(evidence / f"{name}.png"), full_page=False)

            try:
                b4._open_chat(page, url)
                stopped = b4._wait(lambda: (lambda t: t if t.get("status") == "completed" else None)(
                    b4._task(page, url, STOPPED)), 180, "ECHO settled by its author's stop")
                finished = b4._wait(lambda: (lambda t: t if t.get("status") == "completed" else None)(
                    b4._task(page, url, FINISHED)), 180, "FOXTROT settled by its author's finish")
                axes = stopped.get("outcome_axes") or {}
                record["stopped"] = {key: stopped.get(key) for key in ("status", "reason_code", "continuation_offer")}
                record["stopped"]["task_completion"] = (axes.get("execution") or {}).get("task_completion")
                record["finished"] = {key: finished.get(key) for key in ("status", "reason_code", "continuation_offer")}
                assert record["stopped"]["task_completion"]["action"] == "stop", record
                assert record["stopped"]["task_completion"]["rationale"] == RATIONALE
                assert stopped["continuation_offer"] == {"eligible": True, "cause": "author_stop", "refusal": ""}
                assert finished["continuation_offer"]["refusal"] == "author_finished", record

                card = b4._card(page, STOPPED)
                expect(card.locator(":scope > [data-live-summary-button] [data-live-phase]")).to_have_text("Stopped")
                button = card.locator("[data-continue-task]")
                expect(button).to_have_text("Continue", timeout=60_000)
                expect(button).to_have_attribute("title", "Start a new task that continues this unfinished work")
                expect(card).to_contain_text("Ouroboros stopped with unfinished work")
                expect(b4._card(page, FINISHED).locator("[data-continue-task]")).to_have_count(0)
                card.scroll_into_view_if_needed()
                shot("01-stop-offers-continue")
                # The offer starts nothing by itself: no successor, no admission, no model call for one.
                time.sleep(3)
                assert not b4._events(data_dir, "owner_continue_admitted")
                assert model.main_calls("SUCCESSOR") == 0

                page.reload(wait_until="domcontentloaded")
                page.wait_for_selector("#page-chat", timeout=30_000)
                card = b4._card(page, STOPPED)
                button = card.locator("[data-continue-task]")
                expect(button).to_have_text("Continue", timeout=60_000)  # replayed history carries the offer
                expect(b4._card(page, FINISHED).locator("[data-continue-task]")).to_have_count(0)
                shot("02-offer-after-reload")

                button.click()
                expect(button).to_have_text("Continued", timeout=30_000)
                successor = button.get_attribute("data-continue-successor")
                assert successor and successor.startswith(f"{STOPPED}-c"), successor
                shot("03-continued")
                b4._wait(lambda: model.main_calls("SUCCESSOR") >= 1, 180, "the successor's first model call")
                succ = b4._task(page, url, successor)
                record["successor"] = {key: succ.get(key) for key in ("status", "chat_id", "root_task_id")}
                record["successor"]["cause"] = ((succ.get("metadata") or {}).get("continuation") or {}).get("cause")
                assert succ["chat_id"] == 1 and succ["root_task_id"] == successor
                admitted = [row["task_id"] for row in b4._events(data_dir, "owner_continue_admitted")
                            if row.get("predecessor_task_id") == STOPPED]
                assert admitted == [successor], admitted

                page.reload(wait_until="domcontentloaded")
                page.wait_for_selector("#page-chat", timeout=30_000)
                button = b4._card(page, STOPPED).locator("[data-continue-task]")
                expect(button).to_have_text("Continued", timeout=60_000)
                expect(button).to_have_attribute("data-continue-successor", successor)
                expect(button).to_be_disabled()
                shot("04-continued-after-reload")
                assert errors == []
            except Exception:
                shot("failure")
                raise
            finally:
                record["page_errors"] = list(errors)
                browser.close()
    finally:
        record["model_calls"] = list(model.calls)
        (evidence / "record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

"""#949: a speechless turn's whole card stays before an intervening owner message.

One real SPA page receives live frames and reloads the same retained dialogue through
the real gateway. Fixture words/times are synthetic; no owner runtime or model is used.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from ouroboros import dialogue_store
from ouroboros.projects_registry import create_project
from ouroboros.task_results import write_task_result
from ouroboros.terminal_projection import _project_row
from ouroboros.tool_call_log import append_call_row
from supervisor.message_ingress import log_chat
from tests._shared import producer_evidence_dir
from tests.test_chat_history_paging_browser import _FRAMES, MAIN, _idle, _open, _reads
from tests.test_history_bookmark_browser import _bookmark_reconnect
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import (
    _OBSERVE_STATE_READS,
    _TEST_SOCKET_OPEN,
    _emit_ws_frame,
    _wait_state_reads_quiescent,
)
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
TASK = "order-949-turn"
PORTAL = "order-949-portal"


def _at(clock):
    return f"2026-10-25T{clock}Z"


_ORDER = """() => [...document.querySelectorAll('#chat-messages > *')]
    .filter(node => node.matches('.chat-live-card, .chat-bubble:not(.typing-bubble)'))
    .map(node => ({kind: node.matches('.chat-live-card') ? node.dataset.taskId
        : node.classList.contains('user') ? 'owner' : 'answer',
        ts: new Date(Number(node.dataset.ts)).toISOString(), text: node.innerText,
        top: node.getBoundingClientRect().top, bottom: node.getBoundingClientRect().bottom}))"""


def test_speechless_card_keeps_source_time_across_reload_and_reconnect(direct_server_with_data, tmp_path):
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    root = server["data_dir"]
    evidence = producer_evidence_dir(tmp_path / "order-949", "order-949")
    project = create_project(root, "order-949-project", name="Mirror speeds", origin="agent")
    log_chat("system", 1, 1, "Mirror speeds · Started", ts=_at("06:58:20"), task_id=PORTAL,
             record_type="project_started", require_write=True, drive_root=root,
             message_meta={"project_id": project["id"], "project_name": "Mirror speeds",
                           "task_name": "Compare mirror speeds", "delivery_id": "project-start:order-949"})
    source = {"candidate_identity": server["candidate_identity"], "rows": [], "tools": []}
    observed = {}

    with sync_playwright() as pw:
        browser = pw.webkit.launch(headless=True)
        try:
            source["browser"] = {"engine": "webkit", "version": browser.version, "width": 1280, "height": 850}
            page = browser.new_page(viewport={"width": 1280, "height": 850}, timezone_id="UTC")
            errors = PageErrors(page, "webkit")
            page.add_init_script(f"({_OBSERVE_STATE_READS})()")
            _open(page, server["url"])
            _wait_state_reads_quiescent(page)
            # The fixture verifies the served static/Python sentinel at boot; retain the
            # exact changed module as well, independently of the worktree's path.
            served = page.request.get(server["url"] + "/static/modules/chat.js")
            assert served.ok
            source["served_chat_sha256"] = hashlib.sha256(served.body()).hexdigest()
            assert served.body() == (server["repo_dir"] / "web/modules/chat.js").read_bytes()

            def speech(role, clock, text):
                row = {"role": role, "ts": _at(clock), "text": text}
                if role == "assistant":
                    row["task_id"] = TASK
                source["rows"].append(row)
                log_chat("in" if role == "user" else "out", 1, 1, text, ts=row["ts"],
                         task_id=row.get("task_id", ""), require_write=True, drive_root=root)
                _emit_ws_frame(page, {**row, "type": "chat", "chat_id": 1, "content": text})

            speech("user", "07:12:05", "Read the last three loader logs and check for retries.")
            identity = {"task_id": TASK, "invocation_id": "order-949-read", "tool_call_id": "call_order949",
                        "tool": "read_file", "args": {"path": "runs/loader.log"}}
            for kind, clock in [("tool_call_started", "07:12:09"), ("tool_call", "07:12:10")]:
                row = {**identity, "type": kind, "ts": _at(clock)}
                if kind == "tool_call":
                    row.update(status="ok", is_error=False, result_preview="Loader log read")
                source["tools"].append(row)
                assert append_call_row({}, root / "logs", row)["task_log"]
                _emit_ws_frame(page, {"type": "log", "chat_id": 1, "data": {**row, "_live_tool_frame": True}})
            speech("user", "07:12:21", "Check the older log too, if it is still there.")
            speech("assistant", "07:12:41", "Two logs show retries. The older log is still available.")
            result = write_task_result(root, TASK, "completed", chat_id=1, root_task_id=TASK,
                                       delegation_role="root", _is_direct_chat=True, tool_calls=1,
                                       result=source["rows"][-1]["text"])
            summary = {**_project_row(TASK, result, {}, {"chat_id": 1}), "ts": _at("07:12:42")}
            dialogue_store.append(root, "chat", summary)
            source["summary"] = summary
            _emit_ws_frame(page, {**summary, "type": "chat", "role": "system", "system_type": "task_summary",
                                  "content": summary["text"]})

            expected = [(PORTAL, _at("06:58:20")), ("owner", _at("07:12:05")), (TASK, _at("07:12:09")),
                        ("owner", _at("07:12:21")), ("answer", _at("07:12:41"))]

            def capture(name):
                page.evaluate(_FRAMES)
                observed[name] = page.evaluate(_ORDER)
                page.screenshot(path=str(evidence / f"{name}.png"), animations="disabled")
                (evidence / f"{name}.html").write_text(page.content(), encoding="utf-8")
                return [(row["kind"], row["ts"].replace(".000Z", "Z")) for row in observed[name]]

            assert capture("live") == expected
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(_TEST_SOCKET_OPEN)
            _wait_state_reads_quiescent(page)
            _idle(page, MAIN)
            replay = capture("replay")
            source["history"] = _reads(page)
            messages = source["history"][-1]["body"]["messages"]
            carriers = [row for row in messages if row.get("task_id") == TASK and row.get("tool_evidence")]
            assert carriers and all(row["ts"] >= _at("07:12:41") for row in carriers)
            assert min(item["ts"] for row in carriers for item in row["tool_evidence"]["observations"]) == _at("07:12:09")
            assert not any(row.get("is_progress") for row in messages if row.get("task_id") == TASK)
            assert replay == expected, "the card's first tool precedes the owner's follow-up on saved replay"

            # Repeated history reaches the same mounted consumer; no second card or
            # bubble is allowed, and the existing portal keeps its own earlier anchor.
            for attempt in range(2):
                before = len(_reads(page))
                sockets, states = page.evaluate("() => [window.__testSockets.length, window.__stateReadsSettled]")
                _bookmark_reconnect(page)
                # The initial state read establishes the served SHA. Closing before
                # it settles legitimately reloads the document as an unknown version.
                page.wait_for_function("""([sockets, states]) => window.__testSockets.length > sockets
                    && window.__testSockets.at(-1).readyState === WebSocket.OPEN
                    && window.__stateReadsSettled > states && window.__stateReadsInFlight === 0""", arg=[sockets, states])
                page.wait_for_function("n => window.__historyReads.length > n && window.__historyReads.at(-1).done", arg=before)
                _idle(page, MAIN)
                assert capture(f"reconnect-{attempt + 1}") == expected
            source["history"] = _reads(page)
            assert not errors.unexpected()
        finally:
            source["page_errors"] = list(errors) if "errors" in locals() else []
            (evidence / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
            (evidence / "observed.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
            browser.close()

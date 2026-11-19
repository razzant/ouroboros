"""A plan seat's room reads through the REAL server, worker, task API and SPA, in real WebKit.

The owner's words go through the real composer into a real task. A scripted local provider answers every request
the real server sends and nothing inside the server is patched: the main model calls the real ``plan_task``, native
seats call the real ``read_file``, and each script acts only on what its own request carries (the address its
delivery named, the text a read returned). Two models publish a small window through ``GET /models``, so their
seats are paged by the real layout: the brief is their required source and the room stays an optional observation.

What a reader of this test should see hold: a file read of the room is a fact beside the answer, never a capability
delta, an objection, a verdict or a quorum change; a required brief keeps what it owes and never reads as the room;
Record facts survive a page reload and a server restart; Read opens the seat's saved answer. ``EVIDENCE`` is a module
seam a QA wrapper may set; ``OUROBOROS_UI_EVIDENCE_DIR`` names where the frames go otherwise. The short test name keeps
``tmp_path`` (the fixture server's TMPDIR) short enough for its multiprocessing socket under macOS's 104-byte limit.

After that real scenario is judged, a separate phase writes SYNTHETIC saved states the scripted provider does not
produce (a definite no-send, an unknown outcome, a seat still in flight) through the production plan record writer
while the server is stopped, as the fixture seeds. They are states a record may hold, never observed outcomes. The
same server and SPA must show their prepared counts as never delivered or not yet known to be, and the real seats'
measured counts exactly as before.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import pytest

from ouroboros import dialogue_store
from ouroboros.configured_subagents import SUBAGENTS_SETTING
from tests.test_batch4_owner_controls_browser import _reply, _user_text, _wait
from tests.ui_console_errors import PageErrors

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser]
OWNER_WORDS = "ROOMREAD-7Q: plan the room read facts and ask the council."
SEATS = ("part", "whole", "zero", "other", "sol", "brief", "skip")
PAGED = ("brief", "skip")
SMALL_WINDOW, LARGE_WINDOW = 8_000, 1_000_000
CONVERSATION_ROWS, PROGRESS_ROWS = 19, 481
PLAN = {"goal": "Ship the room read facts", "plan": "Keep the existing facts structure and its primitives.",
        "done_when": [{"id": "c1", "claim": "reads reach Record facts", "support": "open the card"}],
        "affected_paths": [], "ask": "council", "message": "Is it right to start?"}
ROOM_ADDRESS = re.compile(r"read_file\(root='artifact_store', path='([^']+)'\)")
BRIEF_ADDRESS = re.compile(r'read_file\(root="artifact_store", path="([^"]+)"\)')
COMBOS = (("light", 1148, 808), ("dark", 1148, 808), ("light", 390, 844), ("dark", 390, 844))
EVIDENCE: Path | None = None
SERVED_MODULES = ("plan_record_presentation.js", "review_presentation.js", "review_opinions.js")
SAVED_MESSAGE = "SYNTHETIC saved states written by this test through plan_record.append_ask; none is an observed outcome"
SAVED = {  # seat -> the dispatch facts its saved row holds; each also carries a real seat's prepared counts
    "nosend": {"status": "not_dispatched", "dispatched": False, "not_dispatched_reason": "SYNTHETIC refusal before any send"},
    "unknown": {"status": "pending", "dispatched": True, "unresolved": True, "error": "SYNTHETIC dispatch outcome unknown"},
    "inflight": {"status": "pending", "dispatched": True, "unresolved": False},
}
MAY_NOT_HAVE_GONE = "counted for a delivery that may not have been sent"


def _call(name, arguments, ordinal):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{name}_{ordinal}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}, "tool_calls"


def _last_line(result):
    numbers = re.findall(r"^\s*(\d+)\t", result, flags=re.MULTILINE)
    return int(numbers[-1]) if numbers else 0


class Provider:
    """The local OpenAI-compatible provider the real server talks to; its log is evidence of what each seat was sent."""

    def __init__(self, monkeypatch):
        from tests import fixtures_mock_llm

        self.lock, self.answered, self.seats = threading.Lock(), set(), {}
        self.main_tools, self.plan_results = [], []
        monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", lambda handler: self.post(handler))
        monkeypatch.setattr(fixtures_mock_llm._Handler, "do_GET", lambda handler: self.get(handler), raising=False)

    def get(self, handler):
        if handler.path != "/v1/models":
            handler.send_error(404)
            return
        rows = [{"id": f"mock-{seat}", "context_length": SMALL_WINDOW if seat in PAGED else LARGE_WINDOW} for seat in SEATS]
        body = json.dumps({"object": "list", "data": [*rows, {"id": "mock-model", "context_length": LARGE_WINDOW}]}).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def post(self, handler):
        payload = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0) or 0)) or b"{}")
        seat = str(payload.get("model") or "").rpartition("::")[2].removeprefix("mock-")
        message, finish = self.seat(seat, payload) if seat in SEATS else self.main(payload)
        _reply(handler, payload, message, finish)
        if seat in SEATS and finish == "stop":
            with self.lock:
                self.answered.add(seat)  # its answer is written; the host has yet to settle it

    def main(self, payload):
        names = [tool["function"]["name"] for tool in payload.get("tools") or []]
        if "plan_task" not in names:
            return {"role": "assistant", "content": "OK"}, "stop"
        called = {call["id"] for row in payload["messages"] for call in row.get("tool_calls") or []
                  if call["function"]["name"] == "plan_task"}
        results = [str(row.get("content") or "") for row in payload["messages"]
                   if row.get("role") == "tool" and row.get("tool_call_id") in called]
        with self.lock:
            self.main_tools, self.plan_results = names, results
        if not results:
            return _call("plan_task", PLAN, 0)
        if len(results) == 1 or ": pending" in results[-1]:
            # Answers arrive in the mailbox; a free read shows them once the host has settled them, so a read that still
            # names a pending seat reads again. The pause only paces those reads; it proves nothing about settlement.
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline and self.answered != set(SEATS):
                time.sleep(0.5)
            time.sleep(2)
            return _call("plan_task", {}, len(results))
        return {"role": "assistant", "content": "The council answered; the plan record has its facts."}, "stop"

    def seat(self, seat, payload):
        sent = "\n".join(_user_text(row) for row in payload["messages"] if row.get("role") == "user")
        results = [str(row.get("content") or "") for row in payload["messages"] if row.get("role") == "tool"]
        opened = [json.loads(call["function"]["arguments"]).get("path") for row in payload["messages"] for call in row.get("tool_calls") or []]
        read = self.next_read(seat, sent, results, opened)
        with self.lock:
            log = self.seats.setdefault(seat, {
                "tools": bool(payload.get("tools")), "first_send_chars": len(sent), "reads": [], "read_heads": [],
                "paged_first_send": "is not inlined because of the initial delivery size" in sent,
                "first_send_has_conversation": "Conversation row 0" in sent, "first_send_has_plan": PLAN["plan"] in sent})
            log["read_heads"] = [result[:160] for result in results]
            if read:
                log["reads"].append(read)
        if read:
            return _call("read_file", read, len(results))
        answer = {"prose": f"REPLY-{seat}: the reading of the request holds.", "objections": [], "withdrawn": []}
        return {"role": "assistant", "content": json.dumps(answer)}, "stop"

    @staticmethod
    def next_read(seat, sent, results, opened):
        """The next ``read_file`` this seat makes, from what it has actually received; ``None`` answers."""
        room = ROOM_ADDRESS.search(sent)
        target = {"root": "artifact_store", "path": room.group(1)} if room else {}
        if seat == "part":
            return None if results else {**target, "max_lines": 40}
        if seat == "other":
            return None if results else {"path": "README.md"}
        if seat == "whole":
            if not results:
                return target
            total = int(re.search(r"(\d+) snapshot lines", sent).group(1))
            line = _last_line(results[-1])
            return None if line >= total and "RESULT TRUNCATED" not in results[-1] else {**target, "start_line": max(line, 1)}
        if seat == "brief":  # only the pointer arrived: the brief first, then the room address the brief itself names
            brief = BRIEF_ADDRESS.search(sent).group(1)
            named = ROOM_ADDRESS.search("\n".join(results))
            if not named:
                return {"root": "artifact_store", "path": brief, **({"start_line": max(_last_line(results[-1]), 1)} if results else {})}
            return None if named.group(1) in opened else {"root": "artifact_store", "path": named.group(1), "max_lines": 5}
        return None


def seed(fixture):
    """The room's earlier rows and a reviewer catalog, written while the fixture's server is stopped."""
    root = fixture["data_dir"]
    fixture["stop_server"]()
    settings_path = root / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings[SUBAGENTS_SETTING] = json.dumps({"enabled": True, "items": [
        {"subagent_id": seat, "recommended_use": f"{seat} reviewer", "effort": "high", "review_eligible": True,
         "route": {"kind": "api_model", "target_id": f"openai-compatible::mock-{seat}"},
         **({"delivery": "packet"} if seat == "sol" else {})} for seat in SEATS]})
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    for n in range(CONVERSATION_ROWS):
        dialogue_store.append(root, "chat", {"chat_id": 1, "direction": "in" if n % 2 == 0 else "out",
                                             "text": f"Conversation row {n}: agreeing the outline and its limits."})
    for n in range(PROGRESS_ROWS):
        dialogue_store.append(root, "progress", {"chat_id": 1, "content": f"Progress row {n}: read the module, ran a check. " * 3})
    fixture["start_server"]()


def owner_starts_the_work(browser, url, evidence):
    """The owner's words through the real composer; the started root's id comes from the card the SPA renders."""
    page = browser.new_page(viewport={"width": 1148, "height": 808})
    try:
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_selector("#chat-input", timeout=60_000)
        page.fill("#chat-input", OWNER_WORDS)
        page.click("#chat-send")
        card = page.locator(".chat-live-card[data-task-id]").last
        card.wait_for(state="visible", timeout=120_000)
        page.screenshot(path=str(evidence / "started.png"))
        return card.get_attribute("data-task-id")
    finally:
        page.close()


def read_detail(request_context, url, task_id):
    response = request_context.get(f"{url}/api/tasks/{task_id}")
    assert response.ok, response.status
    return response.json()


def open_plan(page, task_id, theme):
    page.evaluate("theme => { document.documentElement.dataset.theme = theme; }", theme)
    card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
    card.wait_for(state="visible", timeout=60_000)
    door = card.locator("[data-live-review-summary]")
    door.wait_for(state="visible", timeout=30_000)
    door.click()
    group = card.locator(f'[data-review-group="plan:{task_id}"]')
    group.locator("[data-review-group-toggle]").click()
    group.locator(".review-check-body").wait_for(state="visible")
    page.wait_for_timeout(250)
    return card, group


def record_facts(group, ready=()):
    """Record facts as rendered, read in one pass once the named seats' own status rows (``Participant <seat>``) exist."""
    facts = group.locator("[data-review-facts]")
    facts.locator("summary").click()
    body = facts.locator(".review-facts-body")
    body.wait_for(state="visible")
    for seat in ready:
        body.locator("dt", has_text=re.compile(rf"^Participant {seat}$")).wait_for(state="attached")
    rows = body.evaluate("""el => [...el.querySelectorAll('dt')].map(dt => [dt.textContent, dt.nextElementSibling.textContent,
        getComputedStyle(dt.nextElementSibling).color])""")
    return body, rows


def room_rows(rows):
    return {label: value for label, value, _color in rows if re.fullmatch(r"Participant \w+ room", label)}


def room_line(inline, read=None):
    """The room value Record facts show for a seat that was sent: its inline counts, then what its file reads covered."""
    line = f"{inline['conversation_inline_rows']:,} of {inline['conversation_rows']:,} conversation rows inline"
    line += f", {inline['other_rows']:,} other rows only in the snapshot file" if inline["other_rows"] > 0 else ""
    if read:
        line += f" · file reads covered {read['covered_chars']:,} of {read['complete_chars']:,} characters · observed by the host"
    return line


def served_identity(api, url, root):
    """Each module the plan's Record facts come from, as the server serves it and as the source checkout holds it."""
    return {name: {"served": hashlib.sha256(api.get(f"{url}/static/modules/{name}").body()).hexdigest(),
                   "source": hashlib.sha256(Path(root, "web", "modules", name).read_bytes()).hexdigest()} for name in SERVED_MODULES}


def chain_facts(detail, root, task_id):
    """The record, its exact snapshot and the seats' own retained history, as the host left them."""
    from ouroboros import review_ledger
    from ouroboros.artifacts import task_artifact_dir_path

    [ask] = detail["plan_record"]["asks"]
    store = task_artifact_dir_path(root, task_id, create=False)
    source = (store / ask["dialogue_ref"]["ref"]["path"]).read_bytes()
    histories = []
    for path in sorted(store.glob("source_handles/context_checkpoints/*-history-*.json")):
        history = json.loads(path.read_text(encoding="utf-8"))
        histories.append({"file": path.name, "coverage": history.get("coverage"), "observed_read_coverage": history.get("observed_read_coverage"),
                          "required_sources": history.get("required_sources"), "read_receipts": [
                              {key: receipt.get(key) for key in ("tool", "opened_root", "opened_path", "source_revision", "source_start_char",
                                                                 "source_end_char", "complete_chars", "delivered")}
                              for receipt in history.get("read_receipts") or []]})
    record = review_ledger.load_record(root, ask["record_id"])
    return {"ask": ask, "snapshot_sha256": hashlib.sha256(source).hexdigest(), "snapshot_chars": len(source.decode("utf-8")),
            "histories": histories, "ledger_verdict": record["verdict"],
            "ledger_status": {row["seat_id"]: row.get("status") for row in record["rows"]},
            "ledger_receipts": {row["seat_id"]: row.get("receipts") or {} for row in record["rows"]},
            "plan_objections": detail.get("plan_objections")}


def settled_chain(api, url, root, task_id, deadline):
    """The chain ``check_chain`` judges, read again until no seat is pending in the plan record or the ask's ledger record.

    The host settles a seat's plan row before its ledger row, and the task can end between the two, so neither the end
    nor a pause proves the ledger caught up. Only those durable rows decide readiness; at the bound the chain as it then
    stands is judged, and every poll stays with the evidence."""
    polls = []
    while True:
        detail = read_detail(api, url, task_id)
        chain = chain_facts(detail, root, task_id)
        pending = {"plan": sorted(row["seat_id"] for row in chain["ask"]["participants"] if row.get("status") == "pending"),
                   "ledger": sorted(seat for seat, status in chain["ledger_status"].items() if status == "pending")}
        polls.append({"monotonic": round(time.monotonic(), 3), **pending})
        if not (pending["plan"] or pending["ledger"]) or time.monotonic() >= deadline:
            return detail, chain, polls
        time.sleep(0.5)


def check_chain(chain):
    ask, total = chain["ask"], chain["snapshot_chars"]
    seats = {row["seat_id"]: row for row in ask["participants"]}
    assert set(seats) == set(SEATS) and all(row["status"] == "answered" for row in seats.values()), seats
    assert ask["dialogue_ref"]["sha256"] == chain["snapshot_sha256"] == ask["dialogue_ref"]["ref"]["sha256"]
    read = {seat: row.get("room_read_coverage") for seat, row in seats.items()}
    host = {"provenance": "host_observed", "complete_chars": total}
    assert read["sol"] is None, read["sol"]  # a packet seat has no tools: nothing was read, and nothing is claimed
    assert read["whole"] == {**host, "status": "complete", "covered_chars": total}
    for seat in ("zero", "other", "skip"):
        assert read[seat] == {**host, "status": "incomplete", "covered_chars": 0}, seat
    for seat in ("part", "brief"):
        assert read[seat]["status"] == "incomplete" and 0 < read[seat]["covered_chars"] < total and read[seat]["complete_chars"] == total, seat
    # An optional read that stopped short changes nothing; only an unread required brief owes a delta.
    assert {seat: [delta["reason"] for delta in row["capability_delta"]] for seat, row in seats.items() if row.get("capability_delta")} == {
        "skip": ["native_required_source_coverage_incomplete"]}
    assert chain["plan_objections"]["open_objections"] == {}
    verdict = chain["ledger_verdict"]
    assert (verdict["aggregate"], verdict["reason"]) == ("NOT_PERFORMED", "intention_no_verdict_question")
    assert verdict["quorum"] == {"required": 0, "responded": len(SEATS), "assigned": len(SEATS)}
    for seat, coverage in read.items():
        assert chain["ledger_receipts"][seat].get("room_read_coverage") == coverage, seat
    # Every native read of the room addressed the exact declared snapshot; a brief stays its own required source.
    path = ask["dialogue_ref"]["ref"]["path"]
    room_reads = [receipt for history in chain["histories"] for receipt in history["read_receipts"] if receipt["opened_path"] == path]
    assert room_reads and all(receipt["source_revision"] == chain["snapshot_sha256"] for receipt in room_reads)
    paged = [history for history in chain["histories"] if history["observed_read_coverage"]]
    assert len(paged) == 2, [history["file"] for history in chain["histories"]]
    for history in paged:
        [required] = history["coverage"]["sources"]
        [observed] = history["observed_read_coverage"]["sources"]
        assert required["path"] != path and required["coverage_basis"] == "candidate_blob" and observed["path"] == path


def check_screen(view, chain):
    seats = {row["seat_id"]: row for row in chain["ask"]["participants"]}
    assert view["head"] == "no open objection" and view["head_tone"] != "warn", view["head"]
    assert view["seat_states"] == ["answered, no objection"] * len(SEATS), view["seat_states"]
    assert f"{len(SEATS)} of {len(SEATS)} counted answered" in view["group_heads"][1], view["group_heads"]
    assert not view["errors"].unexpected(), view["errors"].report()
    assert not view["overflow"], view
    shown = view["room_rows"]
    assert set(shown) == {f"Participant {seat} room" for seat in SEATS} and "%" not in json.dumps(shown)
    for seat, row in seats.items():
        inline, value = row["room_delivery"], shown[f"Participant {seat} room"]
        assert f"{inline['conversation_inline_rows']} of {inline['conversation_rows']} conversation rows inline" in value, value
        read = row.get("room_read_coverage")
        expected = f"file reads covered {read['covered_chars']:,} of {read['complete_chars']:,} characters · observed by the host" if read else "file reads"
        assert (expected in value) is bool(read), value
        assert value == room_line(inline, read), value  # an answered seat's measured counts, with no qualifier


def watch_page(page):
    """What a window reports beside its pixels, kept as evidence: page errors (strict until the caller declares a
    reload), console errors, every failed response."""
    seen = {"errors": PageErrors(None, page.context.browser.browser_type.name), "console_errors": [], "failed_responses": []}
    seen["errors"].watch(page, navigates=False)
    page.on("console", lambda message: seen["console_errors"].append(message.text) if message.type == "error" else None)
    page.on("response", lambda response: seen["failed_responses"].append([response.status, response.url]) if response.status >= 400 else None)
    return seen


def look(page, url, task_id, theme, tag, evidence, chain):
    """One owner window: the Plan row, Read of the paged seat, Record facts, and the same facts after a page reload."""
    seen = watch_page(page)
    page.goto(url, wait_until="domcontentloaded")
    card, group = open_plan(page, task_id, theme)
    page.screenshot(path=str(evidence / f"plan-{tag}.png"))
    view = {"theme": theme, "tag": tag, **seen, "user_agent": page.evaluate("navigator.userAgent"),
            "groups": card.locator("[data-review-group]").evaluate_all("rows => rows.map(row => row.dataset.reviewGroup)"),
            "group_heads": [text.strip() for text in card.locator("[data-review-group] .review-check-head").all_inner_texts()],
            "head": group.locator(".review-check-head .review-state").inner_text().strip(),
            "head_tone": group.locator(".review-check-head .review-state").get_attribute("data-tone"),
            "seat_states": [text.strip() for text in group.locator("[data-review-opinion] .review-state").all_inner_texts()]}
    # Read opens the seat's saved answer: the paged seat's reply, never the brief it read or the room.
    row = group.locator("[data-review-opinion]", has_text="mock-brief")
    row.evaluate("el => el.scrollIntoView({block: 'center'})")
    page.wait_for_timeout(150)
    page.screenshot(path=str(evidence / f"seats-{tag}.png"))
    read = row.locator("[data-review-read-url]")
    source = read.get_attribute("data-review-read-url")
    exported = page.request.get(url + source)
    assert exported.ok, exported.status
    view["read_url"], view["read_body"] = source, exported.text()
    assert "REPLY-brief" in view["read_body"] and "YOUR PANEL SEAT" not in view["read_body"], view["read_body"][:400]
    read.focus()
    page.keyboard.press("Enter")
    reader = page.locator("dialog[open]:has(.document-reader-panel)")
    reader.wait_for(state="visible", timeout=30_000)
    page.wait_for_function("() => document.querySelector('dialog[open] .document-reader-body')?.textContent.includes('REPLY-brief')")
    page.screenshot(path=str(evidence / f"reader-{tag}.png"))
    page.keyboard.press("Escape")
    reader.wait_for(state="hidden")
    body, rows = record_facts(group)
    view["room_rows"] = room_rows(rows)
    view["facts_color"] = sorted({color for label, _value, color in rows if label.endswith(" room")})
    label = body.locator("dt", has_text=re.compile(r"^Participant zero room$"))
    label.evaluate("el => el.scrollIntoView({block: 'center'})")
    page.wait_for_timeout(200)
    page.screenshot(path=str(evidence / f"facts-{tag}.png"))
    view["overflow"] = card.evaluate("el => el.scrollWidth > el.clientWidth + 1") or page.evaluate(
        "document.documentElement.scrollWidth > window.innerWidth + 1") or body.evaluate("el => el.scrollWidth > el.clientWidth + 1")
    # The ask's own ledger record beside the plan: the quorum the owner sees, whatever the seats read of the file.
    record = card.locator('[data-review-group^="review_record:"]')
    record.locator("[data-review-group-toggle]").click()
    record.locator(".review-check-body").wait_for(state="visible")
    record.evaluate("el => el.scrollIntoView({block: 'center'})")
    page.wait_for_timeout(200)
    page.screenshot(path=str(evidence / f"record-{tag}.png"))
    view["record_text"] = record.inner_text()
    seen["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload(wait_until="domcontentloaded")
    _card, again = open_plan(page, task_id, theme)
    body, reloaded = record_facts(again)
    view["facts_same_after_reload"] = reloaded == rows
    body.locator("dt", has_text=re.compile(r"^Participant zero room$")).evaluate("el => el.scrollIntoView({block: 'center'})")
    page.wait_for_timeout(200)
    page.screenshot(path=str(evidence / f"facts-reloaded-{tag}.png"))
    view["facts"] = rows
    return view


def write_saved_states(fixture, task_id, real_ask):
    """The SYNTHETIC ask, written while the server is stopped by the production record writer. Each seat carries the
    real preparation's own inline counts (seat ``zero``'s); none carries an operation, so no collector can settle it."""
    from ouroboros import plan_record
    from ouroboros.utils import utc_now_iso

    inline = next(row["room_delivery"] for row in real_ask["participants"] if row["seat_id"] == "zero")
    ask = {"ask_id": "a2", "ts": utc_now_iso(), "revision": real_ask["revision"], "message": SAVED_MESSAGE, "attach": [],
           "attached": [], "omissions": [], "host_added": [], "retry_key": "synthetic-saved-states",
           "participants": [{"seat_id": seat, "model": f"synthetic/{seat}", "route": "api_chat", "retrieves": False,
                             "owner_marked": True, "host_added": False, "continued": False, "room_delivery": dict(inline), **facts}
                            for seat, facts in SAVED.items()]}
    fixture["stop_server"]()
    plan_record.append_ask(fixture["data_dir"], task_id, ask)
    fixture["start_server"]()
    return ask


def stored_saved(detail):
    """The saved seats' dispatch facts and counts as the restarted server's task detail holds them."""
    [ask] = [row for row in detail["plan_record"]["asks"] if row["ask_id"] == "a2"]
    keys = ("status", "dispatched", "unresolved", "not_dispatched_reason", "error", "room_delivery")
    return {row["seat_id"]: {key: row[key] for key in keys if key in row} for row in ask["participants"]}


def look_saved(page, url, task_id, theme, tag, evidence):
    """The saved ask's seats in Record facts at one window, and the same facts after a page reload."""
    seen = watch_page(page)
    page.goto(url, wait_until="domcontentloaded")
    card, group = open_plan(page, task_id, theme)
    body, rows = record_facts(group, ready=SAVED)
    body.locator("dd.review-facts-note", has_text="SYNTHETIC saved states").evaluate("el => el.scrollIntoView({block: 'start'})")
    page.screenshot(path=str(evidence / f"saved-{tag}.png"))
    body.locator("dt", has_text=re.compile(r"^Participant inflight room$")).evaluate("el => el.scrollIntoView({block: 'center'})")
    page.screenshot(path=str(evidence / f"saved-rows-{tag}.png"))
    seats = tuple(f"{seat} (synthetic/{seat})" for seat in SAVED)
    view = {"theme": theme, "tag": tag, **seen, "facts": rows, "room_rows": room_rows(rows),
            "seat_lines": {label: value for label, value, _color in rows if label in seats},
            "saved_colors": sorted({color for label, _value, color in rows if label.startswith(("Participant unknown", "Participant inflight"))}),
            "overflow": card.evaluate("el => el.scrollWidth > el.clientWidth + 1") or page.evaluate(
                "document.documentElement.scrollWidth > window.innerWidth + 1") or body.evaluate("el => el.scrollWidth > el.clientWidth + 1")}
    seen["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload(wait_until="domcontentloaded")
    _card, again = open_plan(page, task_id, theme)
    body, reloaded = record_facts(again, ready=SAVED)
    view["facts_same_after_reload"] = reloaded == rows
    body.locator("dt", has_text=re.compile(r"^Participant inflight room$")).evaluate("el => el.scrollIntoView({block: 'center'})")
    page.screenshot(path=str(evidence / f"saved-rows-reloaded-{tag}.png"))
    return view


def check_saved(view, ask, real_rooms):
    """A confirmed no-send shows no delivered-count line; a pending seat's prepared counts say they may not have gone;
    the real answered seats' measured counts read exactly as the real phase showed them."""
    shown, prepared = view["room_rows"], room_line(ask["participants"][0]["room_delivery"])
    assert {label: shown.get(label) for label in real_rooms} == real_rooms, shown
    assert set(shown) == {*real_rooms, "Participant unknown room", "Participant inflight room"}, shown  # nosend has none
    for seat in ("unknown", "inflight"):
        assert shown[f"Participant {seat} room"] == f"{prepared} · {MAY_NOT_HAVE_GONE}", shown
    status = {label: value for label, value, _color in view["facts"] if label in {f"Participant {seat}" for seat in SAVED}}
    assert status == {f"Participant {seat}": f"status={facts['status']} · model=synthetic/{seat}" for seat, facts in SAVED.items()}, status
    assert view["seat_lines"]["nosend (synthetic/nosend)"] == f"not sent — {SAVED['nosend']['not_dispatched_reason']}", view["seat_lines"]
    assert not view["errors"].unexpected(), view["errors"].report()
    assert not view["overflow"] and view["facts_same_after_reload"], view


def test_room_read(direct_server_with_data, monkeypatch, tmp_path):
    from playwright.sync_api import sync_playwright

    import tests.test_ui_smoke_playwright as smoke

    ready = smoke._wait_supervisor_ready
    monkeypatch.setattr(smoke, "_wait_supervisor_ready", lambda url, timeout_sec=180: ready(url, timeout_sec))  # a shared machine's restart
    provider, fixture = Provider(monkeypatch), direct_server_with_data
    seed(fixture)
    url, root = fixture["url"], fixture["data_dir"]
    evidence = EVIDENCE or Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "room-read-evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.webkit.launch(headless=True)
        results = {"browser": {"name": browser.browser_type.name, "version": browser.version, "executable": browser.browser_type.executable_path}}
        try:
            task_id = owner_starts_the_work(browser, url, evidence)
            api = pw.request.new_context()
            served = api.get(f"{url}/static/modules/plan_record_presentation.js").body()
            results["source_identity"] = {"candidate_version": fixture["candidate_version"], "served_js_sha256": hashlib.sha256(served).hexdigest(),
                                          "source_js_sha256": hashlib.sha256(Path(smoke.REPO_ROOT, "web/modules/plan_record_presentation.js").read_bytes()).hexdigest(),
                                          "modules": served_identity(api, url, smoke.REPO_ROOT)}
            assert all(row["served"] == row["source"] for row in results["source_identity"]["modules"].values()), results["source_identity"]
            settled_by = time.monotonic() + 420  # one bound for the task's end and its chain settling
            _wait(lambda: (lambda row: None if row.get("status") in {"", "scheduled", "queued", "pending", "running"} else row)(read_detail(api, url, task_id)),
                  420, f"{task_id} to end")
            detail, chain, polls = settled_chain(api, url, root, task_id, settled_by)
            results.update(task_id=task_id, status=detail.get("status"), provider_seats=provider.seats, main_tools=provider.main_tools,
                           plan_task_results=provider.plan_results, chain_polls=polls)
            (evidence / "provider.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
            (evidence / "task-detail.json").write_text(json.dumps(detail, indent=1, ensure_ascii=False), encoding="utf-8")
            (evidence / "chain.json").write_text(json.dumps(chain, indent=1, ensure_ascii=False), encoding="utf-8")
            check_chain(chain)
            # What the seats were actually sent: the ordinary delivery carries rows inline, a paged one only its pointer.
            assert provider.seats["brief"]["paged_first_send"] and provider.seats["skip"]["paged_first_send"]
            assert not provider.seats["brief"]["first_send_has_plan"] and provider.seats["zero"]["first_send_has_plan"]
            assert provider.seats["zero"]["first_send_has_conversation"] and not provider.seats["sol"]["tools"]
            # The mind's own read in the real worker: the same facts, apart.
            assert "conversation rows inline" in provider.plan_results[-1] and "(host_observed)" in provider.plan_results[-1]
            views = []
            for theme, width, height in COMBOS:
                page = browser.new_page(viewport={"width": width, "height": height})
                try:
                    views.append(look(page, url, task_id, theme, f"{theme}-{width}x{height}", evidence, chain))
                finally:
                    page.close()
                check_screen(views[-1], chain)
                assert views[-1]["facts_same_after_reload"], views[-1]["tag"]
            results["views"] = views
            fixture["restart_server"]()
            page = browser.new_page(viewport={"width": 1148, "height": 808})
            try:
                page.goto(url, wait_until="domcontentloaded")
                _card, group = open_plan(page, task_id, "light")
                _body, rows = record_facts(group)
                results["facts_same_after_restart"] = rows == views[0]["facts"]
                page.screenshot(path=str(evidence / "facts-after-restart.png"))
            finally:
                page.close()
            results["record_same_after_restart"] = read_detail(api, url, task_id)["plan_record"] == detail["plan_record"]
            assert results["facts_same_after_restart"] and results["record_same_after_restart"]
            # SYNTHETIC SAVED STATES: a separate phase once the real scenario above is judged and its evidence written.
            ask = write_saved_states(fixture, task_id, chain["ask"])
            saved = results["saved_states"] = {
                "label": "synthetic: written by this test through plan_record.append_ask while the server was stopped; "
                         "never an observed vendor or engine outcome", "written": ask,
                "stored_after_restart": stored_saved(read_detail(api, url, task_id)),
                "source_identity": served_identity(api, url, smoke.REPO_ROOT), "views": []}
            assert saved["stored_after_restart"] == stored_saved({"plan_record": {"asks": [ask]}}), saved["stored_after_restart"]
            assert all(row["served"] == row["source"] for row in saved["source_identity"].values()), saved["source_identity"]
            for theme, width, height in COMBOS:
                page = browser.new_page(viewport={"width": width, "height": height})
                try:
                    saved["views"].append(look_saved(page, url, task_id, theme, f"{theme}-{width}x{height}", evidence))
                finally:
                    page.close()
                check_saved(saved["views"][-1], ask, views[0]["room_rows"])
            saved["stored_after_views"] = stored_saved(read_detail(api, url, task_id))
            assert saved["stored_after_views"] == saved["stored_after_restart"], saved["stored_after_views"]
        finally:
            browser.close()
            (evidence / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

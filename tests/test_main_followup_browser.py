"""My human's next Main message reaches the turn already answering them, seen in the real SPA.

Two ordinary composer sends against an attested candidate server (keyless, a loopback model,
no transport configured, so nothing leaves the machine): the first starts a Main turn whose model
round the scenario holds; the second, with a screenshot, lands in that same turn's mailbox and its
bubble says so ("Message saved for task", never the turn's name or a placeholder; never read). Released, the same author reads it at its
final drain and answers both in one reply; stopped, one Stop ends the work both messages reached and
the second message's words and file stay with that work, unread. Screenshots, DOM and the facts read
go to the managed evidence channel; OUROBOROS_BROWSER_EVIDENCE_OUT overrides the destination.
"""

import json
import re
import shutil
import uuid

import pytest

from tests._shared import producer_evidence_dir
from tests.system_e2e.harness import (
    ArtifactOracle,
    KeylessIsolatedServer,
    ModelGate,
    ScriptedStubModel,
    keyless_settings,
    message_text,
    wait_durable_result,
    wait_until,
    write_settings_file,
)
from tests._browser_candidate import wait_clone as wait_clone
pytestmark = [pytest.mark.ui_browser]

def _screenshot_png(width=240, height=135) -> bytes:
    """The picture the second message carries: an opaque PNG, its rows striped like a chat list."""
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + (b"\xee\xee\xf5" if (y // 15) % 2 else b"\x2a\x55\xa8") * width for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class _Owner:
    """The loopback model's two actors: the first message's turn, and any turn a follow-up would wrongly start."""

    def __init__(self, marker, steps):
        self.marker, self.steps, self.turn_calls, self.other_calls = marker, steps, [], []

    def __call__(self, body):
        users = [message_text(m) for m in body.get("messages", []) if m.get("role") == "user"]
        if not any(self.marker in text for text in users):
            self.other_calls.append(body)
            return {"final": "A second author answered."}
        index = len(self.turn_calls)
        self.turn_calls.append(body)
        return self.steps[min(index, len(self.steps) - 1)](body)


def _send(page, text, files=None):
    if files:
        page.locator("#chat-file-input").set_input_files(files)
    page.locator("#chat-input").fill(text)
    page.locator("#chat-send").click()


def _bubble(page, text):
    return page.locator(".chat-bubble", has_text=text).first


def _receipt(page, text):
    return _bubble(page, text).locator(".msg-routing-annotation")


SAVED = re.compile(r"^Message saved for task$")  # saved, never read, never the turn's name (owner 12.11)


def _turn_ids(oracle):
    rows = oracle.queue_snapshot().get("running") or []
    return [row["id"] for row in rows if (row.get("task") or {}).get("_is_direct_chat")]


@pytest.mark.parametrize("ending", ["read", "stop"])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_clarification_reaches_the_turn_answering_my_human_with_one_reply_or_one_stop(
        wait_clone, tmp_path, ending, engine):
    from playwright.sync_api import expect, sync_playwright

    marker = "FOLLOWUP_" + uuid.uuid4().hex[:10]
    first = f"{marker}: send the release note to our team channel"
    second = "This chat, the screenshot shows it"
    answer = "Sent to the team channel shown in your screenshot."
    def turn_round(body):  # the first turn's own agent rounds (a namer's call carries the words but no tools)
        return bool(body.get("tools")) and any(marker in message_text(m) for m in body.get("messages", [])
                                               if m.get("role") == "user")

    gate = ModelGate(turn_round, timeout=240)
    if ending == "read":  # held in its only round: released, its final drain reads the clarification
        steps = [lambda body: {"final": "Sent the release note to the DM."},
                 lambda body: {"final": answer if second in json.dumps(body, ensure_ascii=False) else "Missed it."}]
    else:  # a first round that narrates (its card offers Stop), then a held second round
        steps = [lambda body: {"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"}}]
        gate.match = lambda body: turn_round(body) and len(owner.turn_calls) >= 1
    owner = _Owner(marker, steps)
    evidence = producer_evidence_dir(tmp_path / "evidence", "main-followup",
                                     env="OUROBOROS_BROWSER_EVIDENCE_OUT") / f"main-followup-{ending}"
    evidence.mkdir(parents=True, exist_ok=True)
    shot = tmp_path / "team-channel.png"
    shot.write_bytes(_screenshot_png())
    facts = {"ending": ending, "marker": marker, "boundary": "keyless candidate server, loopback model, no "
             "transport configured: no Telegram, Slack or provider send is possible", "reached": []}
    # Calls with no tools (the turn namer's) get a title; the turn's own rounds follow ``owner``.
    with ScriptedStubModel([owner] * 12, gate=gate, final_answer="Release note for the team channel") as stub:
        root = tmp_path / "instance" / "data"
        root.mkdir(parents=True)
        settings_path = root / "settings.json"
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, root, settings_path)
        oracle = ArtifactOracle(root)
        server.start(ready_timeout=180)
        try:
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": 1100, "height": 750}, reduced_motion="reduce")
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.locator("#chat-input").wait_for(state="visible", timeout=60_000)
                    _send(page, first)
                    assert gate.arrived.wait(120), "the first turn never reached its held model round"
                    facts["reached"].append("first turn holds a model round")
                    (turn,) = wait_until(lambda: _turn_ids(oracle) or None, 60)
                    facts["turn"] = turn
                    _send(page, second, files=[str(shot)])
                    receipt = _receipt(page, second)
                    expect(receipt).to_have_text(SAVED, timeout=60_000)
                    assert receipt.get_attribute("data-annotation-status") == "delivered"
                    assert receipt.get_attribute("data-annotation-action") == "mailbox_delivery"
                    facts["receipt_text"] = receipt.inner_text()
                    expect(_bubble(page, second).locator(".chat-attachments")).to_be_visible()
                    facts["reached"].append("second message saved for the same turn")
                    annotation = next(row for row in oracle._jsonl("logs/chat_annotations.jsonl")
                                      if row.get("action") == "mailbox_delivery")
                    facts["annotation"] = annotation
                    assert annotation["target"] == turn and annotation["status"] == "delivered"
                    assert [row["label"] for row in annotation.get("attachment_manifest") or []] == ["team-channel.png"]
                    assert _turn_ids(oracle) == [turn] and not owner.other_calls  # one author
                    expect(_bubble(page, first)).to_be_visible()  # both messages stay my human's own bubbles
                    page.screenshot(path=str(evidence / "joined.png"), full_page=True, animations="disabled")

                    if ending == "read":
                        gate.release.set()
                        _bubble(page, answer).wait_for(timeout=120_000)
                        result = wait_durable_result(oracle, turn, timeout=120)
                        facts["result_status"] = result["status"]
                        assert result["status"] == "completed"
                        assert len(owner.turn_calls) == 2 and second in json.dumps(owner.turn_calls[1], ensure_ascii=False)
                        assert page.locator(".chat-bubble", has_text=answer).count() == 1
                        facts["reached"].append("the same author answered both in one reply")
                    else:
                        card = page.locator(f'.chat-live-card[data-task-id="{turn}"]')
                        trigger = card.locator("[data-cancel-run]")
                        expect(trigger).to_have_text("Stop…", timeout=60_000)
                        title = card.locator("[data-live-title]").inner_text().strip()
                        expect(_receipt(page, second)).to_have_text(SAVED)  # the card keeps its name; the line has none
                        facts["card_title"] = title
                        trigger.click()
                        menu = page.locator("body > .task-control-menu")
                        menu.wait_for(state="visible", timeout=10_000)
                        page.screenshot(path=str(evidence / "stop-menu.png"), full_page=True, animations="disabled")
                        menu.locator('.task-control-item[data-task-control="stop_now"]').click()
                        result = wait_durable_result(oracle, turn, timeout=120)
                        facts["result_status"] = result["status"]
                        assert result["status"] == "cancelled"
                        unread = [json.loads(row) for row in (result.get("unread_mailbox") or {}).get("rows") or []]
                        facts["unread_after_stop"] = [row.get("text", "")[:80] for row in unread]
                        assert any(row.get("text", "").startswith(second) for row in unread), unread
                        expect(_receipt(page, second)).to_have_text(SAVED)
                        page.wait_for_function("id => document.querySelector(`.chat-live-card[data-task-id=\"${id}\"]`)"
                                               "?.dataset.finished === '1'", arg=turn, timeout=60_000)
                        assert wait_until(lambda: _turn_ids(oracle) == [], 60)  # the published queue follows its tick
                        assert not owner.other_calls
                        facts["reached"].append("one Stop ended the work both messages reached")
                    page.screenshot(path=str(evidence / f"{ending}.png"), full_page=True, animations="disabled")
                    assert not errors, errors
                    facts["passed"] = True
                except Exception:
                    page.screenshot(path=str(evidence / "failure.png"), full_page=True, animations="disabled")
                    raise
                finally:
                    (evidence / "dom.html").write_text(page.content(), encoding="utf-8")
                    facts["page_errors"] = errors
                    facts["model_calls"] = stub.kinds()
                    browser.close()
        finally:
            gate.release.set()
            server.stop()
            if (root / "logs").exists():
                shutil.copytree(root / "logs", evidence / "logs", dirs_exist_ok=True)
            (evidence / "facts.json").write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    assert server.proc.poll() is not None


def test_finished_main_turn_waits_for_a_model_title_before_it_becomes_a_project(wait_clone, tmp_path):
    from threading import Event

    from playwright.sync_api import expect, sync_playwright

    original = "Plot the northern streams and label the reservoir."
    answer = "The reservoir is marked in the sketch."
    title = "Northern stream map"
    accept_title = Event()

    class NamingStub(ScriptedStubModel):
        def _answer(self, body, seq):
            if any(message_text(item).startswith("Name this project from the owner's request")
                   for item in body.get("messages", [])):
                return "naming", {"role": "assistant", "content": title if accept_title.is_set() else '""'}
            return super()._answer(body, seq)

    evidence = producer_evidence_dir(tmp_path / "evidence", "project-model-title",
                                     env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    facts = {"original": original, "title": title}
    with NamingStub([{"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"}}],
                    final_answer=answer) as stub:
        root = tmp_path / "instance" / "data"
        root.mkdir(parents=True)
        settings = root / "settings.json"
        write_settings_file(settings, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, root, settings)
        oracle = ArtifactOracle(root)
        server.start(ready_timeout=90)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                try:
                    page = browser.new_page(viewport={"width": 1100, "height": 750}, reduced_motion="reduce")
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    _send(page, original)
                    expect(_bubble(page, answer)).to_be_visible(timeout=70_000)
                    card = page.locator('#chat-messages .chat-live-card[data-task-id]').last
                    tid = card.get_attribute("data-task-id")
                    result = wait_durable_result(oracle, tid, timeout=45)
                    assert result["status"] == "completed" and result.get("origin_message_ref")
                    assert result.get("origin_message_text") == original
                    assert wait_until(lambda: tid not in _turn_ids(oracle), 30)
                    assert wait_until(lambda: "naming" in stub.kinds(), 30)  # await the separate first-work title
                    facts["task_id"] = tid
                    page.reload(wait_until="domcontentloaded")
                    card = page.locator(f'#chat-messages .chat-live-card[data-task-id="{tid}"]')
                    button = card.locator('[data-turn-into-project]')
                    before = stub.kinds().count("naming")
                    facts["before_clicks"] = before
                    with page.expect_response(lambda response: response.url.endswith("/api/projects/from-task")) as refused:
                        button.click()
                    assert refused.value.status == 409
                    assert refused.value.json()["reason_code"] == "project_name_unavailable"
                    expect(page.get_by_text("The naming model gave no title, so the work stayed in Main. "
                                            "Press Turn into project to try again.", exact=True)).to_be_visible()
                    expect(button).to_be_enabled()
                    assert not page.request.get(f"{server.base_url}/api/projects").json()["projects"]
                    assert stub.kinds().count("naming") == before + 1
                    page.screenshot(path=str(evidence / "refused.png"), full_page=True, animations="disabled")

                    accept_title.set()
                    page.get_by_role("button", name="Dismiss notice").click()
                    with page.expect_response(lambda response: response.url.endswith("/api/projects/from-task")) as created:
                        button.click()
                    assert created.value.status == 200
                    project = created.value.json()["project"]
                    assert project["name"] == title and stub.kinds().count("naming") == before + 2
                    expect(card.locator(".chat-live-project-name")).to_have_text(title)
                    assert len(page.request.get(f"{server.base_url}/api/projects").json()["projects"]) == 1
                    for cid in (1, project["chat_id"]):
                        messages = page.request.get(f"{server.base_url}/api/chat/history?chat_id={cid}").json()["messages"]
                        assert sum(row.get("role") == "user" and row.get("text") == original for row in messages) == 1
                    sent = [data for kind, data in stub.calls if kind == "naming"]
                    assert all(original in message_text(row["messages"][-1]) for row in sent[-2:])
                    page.reload(wait_until="domcontentloaded")
                    expect(card.locator(".chat-live-project-name")).to_have_text(title)
                    phase = card.locator("[data-live-phase]")
                    expect(phase).to_have_text("Done", timeout=30_000)
                    expect(phase).to_be_visible()
                    facts["reloaded_phase"] = phase.inner_text()
                    assert stub.kinds().count("naming") == before + 2
                    page.screenshot(path=str(evidence / "converted.png"), full_page=True, animations="disabled")
                    facts["passed"] = True
                finally:
                    browser.close()
        finally:
            server.stop()
            facts["model_calls"] = stub.kinds()
            (evidence / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    assert server.proc.poll() is not None

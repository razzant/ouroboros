"""One saved task's actual Main card, current-overview choice and Resume.

The setup reuses S49's keyless server, local formatter/count stub and real
Pause/Restart. Only setup uses the task API to make the first window hold;
both subsequent requests must come from the product's card through Chromium.
"""
from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlsplit

import pytest
from tests._shared import producer_evidence_dir
from tests.system_e2e import test_cold_return_window_park as cold
from tests.system_e2e.harness import (
    LANE_MOCK,
    ArtifactOracle,
    require_lane,
    start_server,
    wait_durable_result,
    wait_until,
)
from tests.system_e2e.test_cold_return_window_park import (
    serving_identity,  # noqa: F401 -- shared fixture
)
from tests.system_e2e.test_presence_resilience import (
    s33_candidate,  # noqa: F401 -- shared fixture
)

VIEWPORT = (1440, 900)
SCENE = "cold-return-window-park"
HELD_MARKERS, CURRENT_MARKERS, ROOMY_IDENTITY = 6, 1, 30
RENDER_TIMEOUT = 60_000
CHOSEN_REASON = "Your chosen overview is in place. The request fit at the last check. Press Resume."
CHOSEN_TITLE = "Your chosen overview is in place"
CHOSEN_STEP = "Resume with the chosen overview"


def _main_action(page, task_id):
    return page.locator(f'#page-chat [data-owner-action="{task_id}"] .owner-action-body[data-task-id="{task_id}"]')


def _screen(page, evidence, name):
    page.screenshot(path=str(evidence / name), full_page=False, animations="disabled")


def _path_matches(exchange, method, path):
    return exchange.request.method == method and urlsplit(exchange.url).path == path


def _open_hold(page, card, expect, task_path, shown):
    """Use the real Resume control and the exact browser GET, not the fixture's read."""
    button = card.locator("[data-resume-run]")
    expect(button).to_be_enabled(timeout=RENDER_TIMEOUT)
    with page.expect_response(lambda response: _path_matches(response, "GET", task_path), timeout=RENDER_TIMEOUT) as read:
        button.click()
    response = read.value
    assert response.status == 200
    stop = response.json()["current_stop"]
    assert {**stop["window"], "stop_identity": stop["identity"]} == shown
    dialog = page.get_by_role("dialog").filter(has=page.locator("[data-confirm-input]"))
    expect(dialog).to_be_visible(timeout=RENDER_TIMEOUT)
    choices = dialog.get_by_role("combobox")
    expect(choices).to_be_visible()
    return dialog, choices


def _confirm(page, dialog, expect, resume_path):
    with page.expect_response(
        lambda response: _path_matches(response, "POST", resume_path), timeout=RENDER_TIMEOUT,
    ) as posted:
        dialog.locator("[data-confirm-ok]").click()
    response = posted.value
    assert response.status == 200
    assert response.json().get("exact_continuation"), response.json()
    expect(dialog).to_be_hidden(timeout=RENDER_TIMEOUT)


def _pause_main_work(candidate, root, stub, settings):
    """S49's first generation, but only this work's result is addressed to Main."""
    from ouroboros.budget_pause import STATE_PAUSED
    from ouroboros.server_process import record_service_binding

    data = root / "data"
    (data / "state").mkdir(parents=True)
    record_service_binding(data, "local_model", "127.0.0.1", stub.port, pid=stub.pid)
    server = start_server(candidate, root, settings)
    oracle = ArtifactOracle(data)
    try:
        cold._write_memory(server, stub, oracle, "BROWSER_OVERVIEW_HELD", cold._overview(data, HELD_MARKERS, "first"))
        held = cold._overview_revision(data)
        # S49's submit defaults to hidden; a hidden result cannot witness frame 06 in Main.
        created = cold._post(server, "/api/tasks", {
            "description": f"Keep listing the repository files. [{cold.MARK}] {cold._markers(cold.WORK, cold.WORDS)}",
            "source": "web", "chat_id": 1, "memory_mode": "forked", "timeout_sec": 1800,
        })
        task_id = created.get("body", {}).get("task_id")
        assert task_id and created["status"] in (200, 201, 202), created
        assert wait_until(lambda: any(data.rglob(cold.EFFECT["arguments"]["path"])), 240)
        assert cold.BACKGROUND + cold.PRICE * (cold.WORDS + HELD_MARKERS) in stub.counts
        assert cold._post(server, f"/api/tasks/{task_id}/pause", {"request_id": "browser-cold-return-pause"})["status"] in (200, 202)
        assert wait_until(lambda: cold._pause_row(data, task_id).get("state") == STATE_PAUSED
                          and task_id not in oracle.running_ids(), 240)
        owner_pause = cold._pause_row(data, task_id)
        assert owner_pause.get("reason") == "owner", owner_pause
    finally:
        server.stop()
    rounds = len(stub.work_rounds())
    assert rounds >= 1
    assert f"Revision: {held}" in cold._system(stub.work_rounds()[0])
    cold._one_effect(data, stub)
    return task_id, owner_pause, rounds, held


def _prepare_offer(server, stub, task_id, owner_pause, held_revision):
    data, oracle = server.data_root, ArtifactOracle(server.data_root)
    cold._restored_queue(oracle, task_id)
    # S49 already tests the first park where neither overview fits; this browser
    # journey starts with the offer-eligible state. The writer still has to admit
    # this real change to the active overview without changing the saved pin.
    cold._write_memory(server, stub, oracle, "BROWSER_CURRENT_AND_ROOMY",
                       cold._overview(data, CURRENT_MARKERS, "second"), cold._identity(data, ROOMY_IDENTITY))
    current_revision = cold._overview_revision(data)
    assert current_revision != held_revision
    before = cold._settled_posts(stub, oracle)
    assert cold._resume(server, task_id)["status"] == 200
    parked = cold._parked_after(data, task_id, owner_pause["pause_id"])
    assert parked and cold._waits_in_queue(oracle, task_id)
    assert stub.posts() == before, stub.posts_since(before)
    shown = cold._shown_hold(server, task_id)
    with_held = cold.BACKGROUND + cold.PRICE * (cold.WORDS + HELD_MARKERS + ROOMY_IDENTITY)
    with_current = cold.BACKGROUND + cold.PRICE * (cold.WORDS + CURRENT_MARKERS + ROOMY_IDENTITY)
    assert {key: value for key, value in shown.items() if key != "stop_identity"} == {
        "pause_id": parked["pause_id"], "pause_generation": parked["pause_generation"],
        "input_tokens": with_held, "window_tokens": cold.WINDOW, "reply_floor_tokens": cold.REPLY_FLOOR,
        "over_tokens": with_held + cold.REPLY_FLOOR - cold.WINDOW,
        "overview": {"held_revision": held_revision, "current_revision": current_revision, "input_tokens": with_current},
    }
    assert isinstance(shown["stop_identity"], str) and shown["stop_identity"], shown
    return oracle, shown


def _assert_continued(oracle, stub, data, task_id, initial_rounds, shown, counts_before):
    result = wait_durable_result(oracle, task_id, timeout=600)
    assert result.get("status") == "completed" and cold.FINAL in str(result.get("result") or ""), result
    from ouroboros.budget_pause import STATE_RESUMED

    consumed = cold._pause_row(data, task_id)
    assert consumed.get("state") == STATE_RESUMED and consumed.get("pause_id") == shown["pause_id"], consumed
    rounds = stub.work_rounds()[initial_rounds:]
    assert rounds, "Resume after the switch never reached the model stub"
    system = cold._system(rounds[0])
    overview = shown["overview"]
    assert f'Revision: {overview["current_revision"]}' in system and f'Revision: {overview["held_revision"]}' not in system
    assert "second edition" in system and "first edition" not in system
    assert any(row.get("role") in {"tool", "assistant"} for row in rounds[0]["messages"] if isinstance(row, dict))
    assert overview["input_tokens"] in stub.counts[counts_before:]
    assert shown["input_tokens"] not in stub.counts[counts_before:]
    terminal = wait_until(lambda: cold._task_done(oracle, task_id), 60)
    assert terminal and len(terminal) == 1 and terminal[0].get("status") == "completed", terminal
    cold._one_effect(data, stub)


def _use_card(page, expect, server, oracle, stub, task_id, shown, initial_rounds, evidence):
    task_path = f"/api/tasks/{task_id}"
    resume_path = f"{task_path}/resume"
    sent = []

    def record(request):
        if request.method == "POST" and urlsplit(request.url).path == resume_path:
            sent.append(request.post_data_json or {})

    page.on("request", record)
    page.goto(server.base_url, wait_until="domcontentloaded")
    card = _main_action(page, task_id)
    expect(card).to_be_visible(timeout=RENDER_TIMEOUT)
    card.scroll_into_view_if_needed()
    _screen(page, evidence, "01-held-does-not-fit.png")
    dialog, choices = _open_hold(page, card, expect, task_path, shown)
    assert choices.locator("option").evaluate_all("nodes => nodes.map(node => node.value)") == ["again", "overview", "end"]
    for count in (81_500, 79_000, 80_000):
        expect(dialog).to_contain_text(f"{count:,}")
    _screen(page, evidence, "02-dialog-three-steps.png")
    before = cold._settled_posts(stub, oracle)
    overview = shown["overview"]
    choices.select_option(value="overview")
    _confirm(page, dialog, expect, resume_path)
    assert sent == [{"context_choice": cold._named(shown, overview_revision=overview["current_revision"])}], sent
    data = server.data_root
    parked = cold._parked_after(data, task_id, shown["pause_id"])
    assert parked and cold._waits_in_queue(oracle, task_id)
    next_hold = cold._shown_hold(server, task_id)
    assert (next_hold["pause_id"], next_hold["pause_generation"]) == (parked["pause_id"], parked["pause_generation"])
    assert next_hold["overview"] == {**overview, "taken": True}, next_hold
    assert isinstance(next_hold["stop_identity"], str) and next_hold["stop_identity"], next_hold
    assert next_hold["stop_identity"] != shown["stop_identity"], next_hold
    assert next_hold["overview"]["input_tokens"] + next_hold["reply_floor_tokens"] == 79_000
    assert next_hold["window_tokens"] == 80_000 and stub.posts() == before
    assert len(stub.work_rounds()) == initial_rounds and not cold._task_done(oracle, task_id)
    # Wait for the card to repaint; a successful POST or a transient toast
    # cannot establish that its new step is visible.
    reason = card.get_by_text(CHOSEN_REASON, exact=True)
    expect(reason).to_be_visible(timeout=RENDER_TIMEOUT)
    reason.scroll_into_view_if_needed()
    _screen(page, evidence, "03-choice-sent.png")

    page.reload(wait_until="domcontentloaded")
    expect(reason).to_be_visible(timeout=RENDER_TIMEOUT)
    reason.scroll_into_view_if_needed()
    _screen(page, evidence, "04-current-overview-held-after-reload.png")
    dialog, choices = _open_hold(page, card, expect, task_path, next_hold)
    assert choices.locator("option").evaluate_all("nodes => nodes.map(node => node.value)") == ["again", "end"]
    expect(dialog.get_by_text(CHOSEN_TITLE, exact=True)).to_be_visible()
    expect(choices.locator('option[value="again"]')).to_have_text(CHOSEN_STEP)
    expect(dialog).to_contain_text("79,000")
    expect(dialog).to_contain_text("80,000")
    expect(dialog).not_to_contain_text(re.compile(r"make room|free (?:up )?space|raise the window", re.I))
    assert stub.posts() == before and len(sent) == 1  # reading and reload must not resubmit the choice
    _screen(page, evidence, "05-dialog-can-continue.png")
    stub.finish, counts_before = True, len(stub.counts)
    choices.select_option(value="again")
    _confirm(page, dialog, expect, resume_path)
    assert sent == [{"context_choice": cold._named(shown, overview_revision=overview["current_revision"])},
                     {"context_choice": cold._named(next_hold, try_again=True)}], sent
    _assert_continued(oracle, stub, data, task_id, initial_rounds, next_hold, counts_before)
    finished = page.locator("#page-chat").get_by_text(cold.FINAL, exact=True)
    expect(finished).to_be_visible(timeout=RENDER_TIMEOUT)
    finished.scroll_into_view_if_needed()
    _screen(page, evidence, "06-continued.png")


@pytest.mark.browser
@pytest.mark.timeout(cold.LONG_SCENARIO_TIMEOUT_SEC)
def test_saved_work_card_uses_the_current_overview_then_resumes(s33_candidate, serving_identity, tmp_path):  # noqa: F811
    require_lane(LANE_MOCK)
    playwright = pytest.importorskip("playwright.sync_api")
    root = tmp_path / "instance"
    evidence = producer_evidence_dir(tmp_path / "evidence", SCENE, env="OUROBOROS_BROWSER_EVIDENCE_OUT") / SCENE
    evidence.mkdir(parents=True)
    with cold._LocalLaneStub() as stub:
        stub.pid = serving_identity(stub.port)
        settings = cold._settings(stub)
        task_id, owner_pause, rounds, held = _pause_main_work(s33_candidate, root, stub, settings)
        server = start_server(s33_candidate, root, settings)
        try:
            oracle, shown = _prepare_offer(server, stub, task_id, owner_pause, held)
            with playwright.sync_playwright() as pw:
                browser = pw.chromium.launch()
                try:
                    page = browser.new_page(viewport=dict(zip(("width", "height"), VIEWPORT)), locale="en-US")
                    _use_card(page, playwright.expect, server, oracle, stub, task_id, shown, rounds, evidence)
                finally:
                    browser.close()
        finally:
            server.stop()


def _observe_composer(page, playwright, server, text):
    sent = []

    def observe(socket):
        if urlsplit(socket.url).path != "/ws":
            return

        def frame(payload):
            row = json.loads(payload)
            if isinstance(row, dict) and row.get("type") == "chat" and row.get("content") == text:
                sent.append(row)

        socket.on("framesent", frame)

    page.on("websocket", observe)
    page.goto(server.base_url, wait_until="domcontentloaded")
    page.locator("#chat-input").fill(text)
    button = page.locator("#chat-send")
    playwright.expect(button).to_be_enabled(timeout=60_000)
    button.click()
    playwright.expect(page.locator("#page-chat").get_by_text(cold.FINAL, exact=True)).to_be_visible(timeout=120_000)
    [outbound] = sent
    assert outbound.get("profile_scope") and outbound.get("client_message_id")
    return outbound["client_message_id"]


@pytest.mark.browser
@pytest.mark.timeout(240)
def test_web_main_composer_keeps_one_saved_origin_and_first_prepared_request(s33_candidate, serving_identity, tmp_path):  # noqa: F811
    from ouroboros.local_model_server import input_fingerprint
    from ouroboros.send_clock import CLOCK_NOTE_PREFIX
    from ouroboros.server_process import record_service_binding

    require_lane(LANE_MOCK)
    playwright = pytest.importorskip("playwright.sync_api")
    text = "Ordinary Main browser submission. " + cold._markers(cold.WORK, cold.WORDS) + " DONE_MARKER"
    root = tmp_path / "instance"
    data = root / "data"
    (data / "state").mkdir(parents=True)
    with cold._LocalLaneStub() as stub:
        stub.pid, stub.finish = serving_identity(stub.port), True
        record_service_binding(data, "local_model", "127.0.0.1", stub.port, pid=stub.pid)
        server = start_server(s33_candidate, root, cold._settings(stub))
        try:
            with playwright.sync_playwright() as pw:
                browser = pw.chromium.launch()
                try:
                    cmid = _observe_composer(browser.new_page(locale="en-US"), playwright, server, text)
                finally:
                    browser.close()
            oracle = ArtifactOracle(server.data_root)
            log = [json.loads(line) for line in oracle.chat_bytes().splitlines()]
            [saved] = [row for row in log if row.get("direction") == "in" and row.get("text") == text]
            assert saved.get("chat_id") == 1 and saved.get("source") == "web" and saved.get("client_message_id") == cmid
            seen = wait_until(lambda: [event["task"]["metadata"] for event in oracle.events("task_received")
                if event.get("task", {}).get("metadata", {}).get("client_message_id") == cmid], 30)
            [metadata] = seen
            origin = metadata["origin_message_ref"]
            assert (origin["chat_id"], origin["client_message_id"], origin["ts"], origin["text_sha256"]) == (
                1, cmid, saved["ts"], hashlib.sha256(text.encode()).hexdigest())
            assert not metadata.get("origin_suppressed")
            [first] = stub.work_rounds()
            assert sum(str(row.get("content", "")).count(text) for row in first["messages"] if row["role"] == "user") == 1
            clocks = [row["content"] for row in first["messages"] if row.get("role") == "user"
                      and isinstance(row.get("content"), str) and row["content"].startswith(CLOCK_NOTE_PREFIX)]
            [clock] = clocks
            assert first["messages"][-1] == {"role": "user", "content": clock}
            assert any(input_fingerprint(first) == input_fingerprint(payload)
                       and payload["messages"][-1] == first["messages"][-1] for payload in stub.measured)
        finally:
            server.stop()

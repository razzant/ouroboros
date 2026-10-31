"""TZ-2 B1: a comment-only question (an ``escalate`` with zero options) reaches the real
web UI as a quiz card with no option buttons and the free-answer box, and the owner's
typed answer travels verbatim through the real gateway: the durable quiz block records
no chosen option and the exact text, and the card shows it as the owner's answer — live
and again from history after a reload. Chromium and WebKit, on a 375px phone and a
wide (>=980px) desktop layout; an engine that is not installed skips by name.

Both message fields are typed by the keyboard (docs/DESIGN.md "Controls and editable
choices"): the Main composer and the own answer take Shift+Enter as a line break, a
composing Enter sends nothing, and Enter sends — held down on the answer, it still
records one. One desktop run presses the Send buttons instead. Playwright's keys are
synthetic: neither a native IME nor a phone keyboard is certified here.

The same keys carry a Project question with options: the request is typed into the
Project composer, and the owner's own words answer from the Project form or from its
Main copy; both settle, the Main copy leaves, and the Project replays the record after
a reload. A Files editor in the same app keeps Enter as a line break and sends nothing.

Only model judgment is a fixture (the scripted stub asks, then finishes); the server,
the worker, the WebSocket ingress and the decision endpoint are the production ones.
"""
import json
from urllib.parse import urlparse

import pytest

from tests._shared import producer_evidence_dir
from devtools.benchmarks.common.server_runner import _api
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.system_e2e.harness import (
    ArtifactOracle, KeylessIsolatedServer, ScriptedStubModel, keyless_settings,
    wait_durable_result, wait_until, write_settings_file,
)

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.browser]

QUESTION = "What deadline should the report state on its cover page?"
REQUEST = ("Ask me for the deadline,", "then wait for my answer.")
ANSWER = "Friday, 3 October —\nand say it is provisional."
PROJECT_REQUEST = ("Ask which evidence the report should use,", "then wait for my answer.")
PROJECT_QUESTION = "Which evidence should the report use?"
OPTIONS = [{"label": "Primary sources", "detail": "Use the complete original measurements."},
           {"label": "Both sources", "detail": "Include the independent replication too."}]
OWN_ANSWER = "Neither —\nuse the archived 2019 survey."
# The Main copy shows the recorded words for its five-second settlement, then leaves.
MAIN_COPY_ANSWERED = """([selector, answer]) => {
    const card = document.querySelector(selector);
    return card?.dataset.state === 'answered' && card.querySelector('.chat-quiz-answer')?.textContent === answer;
}"""

def _evidence_dir(tmp_path):
    return producer_evidence_dir(tmp_path / "screenshots", "quiz-comment-only",
                                 env="OUROBOROS_BROWSER_EVIDENCE_OUT")


# The two IME orders, dispatched in the real engine: Chromium/Firefox mark the composing
# keydown; WebKit ends the composition first and marks the committing Enter only with 229.
COMPOSING_ENTER = """el => {
    const enter = (init) => {
        const event = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true, ...init });
        el.dispatchEvent(event);
        return [event.defaultPrevented, event.keyCode];
    };
    el.dispatchEvent(new CompositionEvent('compositionstart', { data: '' }));
    const composing = enter({ isComposing: true, keyCode: 229 });
    el.dispatchEvent(new CompositionEvent('compositionend', { data: '' }));
    return [composing, enter({ keyCode: 229 }), el.value];
}"""


def _type_lines(page, field, text):
    field.focus()
    for index, line in enumerate(text.split("\n")):
        if index:
            page.keyboard.press("Shift+Enter")
        page.keyboard.type(line)
    assert field.input_value() == text
    assert field.evaluate(COMPOSING_ENTER) == [[False, 229], [False, 229], text]


def _settled_answer_card(card, answer=ANSWER, options=0):
    """The answered card shows the owner's words, no chosen option and no second answer."""
    card.locator('.chat-quiz-answer').filter(has_text=answer.split("\n")[0]).wait_for(timeout=30000)
    wait_until(lambda: card.get_attribute("data-state") == "answered" or None, 30)
    assert card.locator('.chat-quiz-answer').evaluate("el => el.textContent") == f"Owner's answer: {answer}"
    assert card.locator('.chat-quiz-option').count() == options
    assert card.locator('.chat-quiz-option.chosen').count() == 0
    assert card.locator('.chat-quiz-comment').count() == 0


def _chat_rows(oracle, client_message_id):
    rows = (json.loads(line) for line in oracle.chat_bytes().decode("utf-8").splitlines() if line.strip())
    return [row for row in rows if row.get("client_message_id") == client_message_id]


def _task_waiting(oracle, message_id):
    """The task the typed message started, once its question is waiting for the owner."""
    task = wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
        if (row.get("task", {}).get("metadata", {}).get("origin_message_ref") or {}).get("client_message_id") == message_id), None), 90)
    assert task
    wait = wait_until(lambda: (block if (block := oracle.task_result(task["id"]).get("owner_wait", {})).get("state") == "waiting" else None), 90)
    assert wait and wait.get("quiz_id"), wait
    return task, wait


def _page(pw, engine, width):
    """The installed engine (an absent one skips by name) on a phone or a desktop layout."""
    try:
        browser = getattr(pw, engine).launch()
    except Exception as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(f"Installed {engine} unavailable: {exc}")
        raise
    mobile = width < 980
    return browser, browser.new_page(viewport={"width": width, "height": 812 if mobile else 900},
                                     is_mobile=mobile, has_touch=mobile, reduced_motion="reduce")


def _outbound(page):
    """What the page itself sends: chat frames on its real socket, and every POST."""
    sent = {"chat": [], "posts": []}

    def frame(payload):
        try:
            message = json.loads(payload)
        except (TypeError, ValueError):
            return
        if isinstance(message, dict) and message.get("type") == "chat":
            sent["chat"].append(message)
    page.on("websocket", lambda socket: socket.on("framesent", frame))
    page.on("request", lambda request: sent["posts"].append((urlparse(request.url).path, request.post_data))
            if request.method == "POST" else None)
    return sent


def _open_project(page, project, task_id="", quiz_id=""):
    """The navigation event the Project references and the side navigation dispatch."""
    page.evaluate("""([project, task_id, quiz_id]) => window.dispatchEvent(new CustomEvent('ouro:open-project',
        {detail: {project, task_id, quiz_id}}))""", [project, task_id, quiz_id])
    panel = page.locator('#project-panel')
    panel.wait_for(state="visible")
    return panel


@pytest.mark.parametrize(("engine", "width", "send_by"), [
    ("chromium", 375, "enter"), ("chromium", 1280, "enter"),
    ("webkit", 375, "enter"), ("webkit", 1280, "enter"), ("chromium", 1280, "button"),
])
def test_comment_only_question_renders_and_takes_a_free_text_answer(wait_clone, tmp_path, engine, width, send_by):
    from playwright.sync_api import sync_playwright

    root = tmp_path / "instance" / "data"
    root.mkdir(parents=True)
    screenshots = _evidence_dir(tmp_path)
    shot = f"{engine}-{width}-{send_by}-comment-only"
    steps = [
        {"tool": "escalate", "arguments": {"question": QUESTION, "options": [],
            "stake": "The deadline is printed on the cover page.", "wait_for_answer": True}},
        {"final": "The report states the deadline you gave."},
    ]
    with sync_playwright() as pw:
        browser, page = _page(pw, engine, width)
        decisions = []
        page.on("request", lambda request: decisions.append(request.post_data)
                if request.method == "POST" and request.url.endswith("/api/decisions") else None)
        try:
            with ScriptedStubModel(steps) as stub:
                settings_path = root / "settings.json"
                write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
                server = KeylessIsolatedServer(wait_clone, root, settings_path)
                server.start(ready_timeout=120)
                oracle = ArtifactOracle(root)
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.locator('#chat-status').filter(has_text="Online").wait_for(timeout=60000)
                    composer = page.locator('#chat-input')
                    assert composer.get_attribute("enterkeyhint") == "send"
                    request = "\n".join(REQUEST)
                    _type_lines(page, composer, request)
                    page.locator('#chat-input-area').screenshot(animations="disabled", path=str(screenshots / f"{shot}-composer.png"))
                    if send_by == "enter":
                        composer.press("Enter")
                    else:
                        page.locator('#chat-send').click()
                    wait_until(lambda: composer.input_value() == "" or None, 30)
                    sent = page.locator('#chat-messages .chat-bubble.user[data-client-message-id]')
                    sent.first.wait_for(timeout=30000)
                    assert sent.count() == 1
                    message_id = sent.first.get_attribute("data-client-message-id")
                    row = wait_until(lambda: next(iter(_chat_rows(oracle, message_id)), None), 60)
                    assert row and row["text"] == request, row
                    task, wait = _task_waiting(oracle, message_id)
                    selector = f'#chat-messages .chat-quiz-card[data-task-id="{task["id"]}"][data-quiz-id="{wait["quiz_id"]}"]'
                    card = page.locator(selector)
                    card.get_by_text(QUESTION, exact=True).wait_for(timeout=30000)
                    field = card.locator('.chat-quiz-comment')
                    field.wait_for(timeout=30000)
                    # Zero options: no option button at all, only the free-answer box and its send.
                    assert card.locator('.chat-quiz-option').count() == 0
                    assert card.locator('.chat-quiz-comment').count() == 1
                    assert card.locator('.chat-quiz-send').is_disabled()
                    assert field.get_attribute("enterkeyhint") == "send"
                    # The card fits the viewport: no horizontal overflow on the phone layout.
                    box = card.bounding_box()
                    assert box and box["x"] >= 0 and box["x"] + box["width"] <= width + 1, box
                    card.scroll_into_view_if_needed()
                    card.screenshot(animations="disabled", path=str(screenshots / f"{shot}-question.png"))
                    field.press("Enter")
                    assert field.input_value() == "", "an empty answer neither sends nor becomes a line"
                    _type_lines(page, field, ANSWER)
                    assert card.locator('.chat-quiz-send').is_enabled()
                    card.screenshot(animations="disabled", path=str(screenshots / f"{shot}-draft.png"))
                    assert decisions == [] and card.get_attribute("data-state") == "open"
                    if send_by == "enter":
                        # Held down: the first press answers, the repeats add nothing.
                        for _ in range(3):
                            page.keyboard.down("Enter")
                        page.keyboard.up("Enter")
                    else:
                        card.locator('.chat-quiz-send').click()
                    result = wait_durable_result(oracle, task["id"], timeout=120)
                    block = result["owner_quiz"][wait["quiz_id"]]
                    assert block.get("answered_index") is None and block["comment"] == ANSWER, block
                    assert result["status"] == "completed", result.get("status")
                    _settled_answer_card(card)
                    assert [json.loads(body)["comment"] for body in decisions] == [ANSWER]
                    card.screenshot(animations="disabled", path=str(screenshots / f"{shot}-answered.png"))
                    # History replay: a fresh page rebuilds the same settled card from the durable record.
                    page.reload(wait_until="domcontentloaded")
                    card = page.locator(selector)
                    card.get_by_text(QUESTION, exact=True).wait_for(timeout=30000)
                    _settled_answer_card(card)
                    card.scroll_into_view_if_needed()
                    card.screenshot(animations="disabled", path=str(screenshots / f"{shot}-history.png"))
                    (screenshots / f"{shot}.json").write_text(json.dumps({
                        "engine": engine, "width": width, "send_by": send_by, "task_id": task["id"],
                        "quiz_id": wait["quiz_id"], "chat_row": row, "owner_quiz": block}, indent=2))
                finally:
                    server.stop()
        finally:
            browser.close()


@pytest.mark.parametrize(("engine", "width", "answer_in"), [
    ("chromium", 1280, "project"), ("webkit", 375, "project"),
    ("webkit", 1280, "main"), ("chromium", 375, "main"),
])
def test_project_question_takes_own_words_on_enter_in_the_project_or_its_main_copy(
        wait_clone, tmp_path, engine, width, answer_in):
    from playwright.sync_api import sync_playwright

    root = tmp_path / "instance" / "data"
    root.mkdir(parents=True)
    files = tmp_path / "files"
    files.mkdir()
    screenshots = _evidence_dir(tmp_path)
    shot = f"{engine}-{width}-project-answered-in-{answer_in}"
    steps = [
        {"tool": "escalate", "arguments": {"question": PROJECT_QUESTION, "options": OPTIONS,
            "stake": "The answer determines the report's evidence.", "wait_for_answer": True}},
        {"final": "The report uses the evidence you named."},
    ]
    with sync_playwright() as pw:
        browser, page = _page(pw, engine, width)
        sent = _outbound(page)
        try:
            with ScriptedStubModel(steps) as stub:
                settings_path = root / "settings.json"
                write_settings_file(settings_path, keyless_settings(
                    stub, OUROBOROS_MAX_WORKERS=1, OUROBOROS_FILE_BROWSER_DEFAULT=str(files)))
                server = KeylessIsolatedServer(wait_clone, root, settings_path)
                server.start(ready_timeout=120)
                oracle = ArtifactOracle(root)
                try:
                    project = _api(server.base_url, "POST", "/api/projects", {"name": "Evidence review"})["project"]
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.locator('#chat-status').filter(has_text="Online").wait_for(timeout=60000)
                    panel = _open_project(page, project)
                    composer = panel.locator('.chat-text-row textarea')
                    assert composer.get_attribute("enterkeyhint") == "send"
                    request = "\n".join(PROJECT_REQUEST)
                    _type_lines(page, composer, request)
                    composer.press("Enter")
                    wait_until(lambda: composer.input_value() == "" or None, 30)
                    # One frame, to the Project's own room, with the typed line break.
                    assert [(frame["content"], frame.get("project_id"), int(frame.get("chat_id") or 1))
                            for frame in sent["chat"]] == [(request, project["id"], project["chat_id"])]
                    message_id = sent["chat"][0]["client_message_id"]
                    row = wait_until(lambda: next(iter(_chat_rows(oracle, message_id)), None), 60)
                    assert row["text"] == request and row["chat_id"] == project["chat_id"], row
                    task, wait = _task_waiting(oracle, message_id)
                    key = f'[data-task-id="{task["id"]}"][data-quiz-id="{wait["quiz_id"]}"]'
                    in_project = panel.locator(f'.chat-quiz-card{key}:not(.project-question-card)')
                    main_copy = f'#chat-messages .project-question-card{key}'
                    in_main = page.locator(main_copy)
                    in_project.get_by_text(PROJECT_QUESTION, exact=True).wait_for(timeout=30000)
                    in_main.wait_for(state="attached", timeout=30000)
                    # The Project form and its Main copy: the same options, the same message field.
                    for copy in (in_project, in_main):
                        assert copy.locator('.chat-quiz-option').count() == len(OPTIONS)
                        assert copy.locator('.chat-quiz-comment').get_attribute("enterkeyhint") == "send"
                    if answer_in == "main":
                        page.locator('#project-panel-close').click()
                        panel.wait_for(state="hidden")
                    card = in_project if answer_in == "project" else in_main
                    field = card.locator('.chat-quiz-comment')
                    card.scroll_into_view_if_needed()
                    box = card.bounding_box()
                    assert box and box["x"] >= 0 and box["x"] + box["width"] <= width + 1, box
                    card.screenshot(animations="disabled", path=str(screenshots / f"{shot}-question.png"))
                    field.press("Enter")
                    assert field.input_value() == "", "an empty answer neither sends nor becomes a line"
                    _type_lines(page, field, OWN_ANSWER)
                    assert card.locator('.chat-quiz-send').is_enabled()
                    card.screenshot(animations="disabled", path=str(screenshots / f"{shot}-draft.png"))
                    assert not [path for path, _ in sent["posts"] if path == "/api/decisions"]
                    # Held down: the first press answers, the repeats add nothing.
                    for _ in range(3):
                        page.keyboard.down("Enter")
                    page.keyboard.up("Enter")
                    # Wherever the words were typed, the Main copy settles into them, then leaves.
                    page.wait_for_function(MAIN_COPY_ANSWERED, arg=[main_copy, f"Owner's answer: {OWN_ANSWER}"], timeout=30000)
                    _settled_answer_card(card, OWN_ANSWER, options=len(OPTIONS))
                    card.screenshot(animations="disabled", path=str(screenshots / f"{shot}-answered.png"))
                    in_main.wait_for(state="detached", timeout=15000)
                    decisions = [json.loads(body) for path, body in sent["posts"] if path == "/api/decisions"]
                    assert [(body["decision_id"], body["comment"], "option_index" in body) for body in decisions] == [
                        (f'quiz:{task["id"]}:{wait["quiz_id"]}', OWN_ANSWER, False)]
                    result = wait_durable_result(oracle, task["id"], timeout=120)
                    block = result["owner_quiz"][wait["quiz_id"]]
                    assert block.get("answered_index") is None and block["comment"] == OWN_ANSWER, block
                    assert result["status"] == "completed", result.get("status")
                    # After a reload the answered question stays out of Main; its Project replays the record.
                    page.reload(wait_until="domcontentloaded")
                    page.locator('#chat-status').filter(has_text="Online").wait_for(timeout=60000)
                    _open_project(page, project, task["id"], wait["quiz_id"])
                    _settled_answer_card(in_project, OWN_ANSWER, options=len(OPTIONS))
                    in_project.scroll_into_view_if_needed()
                    in_project.screenshot(animations="disabled", path=str(screenshots / f"{shot}-history.png"))
                    # Main painted its whole history, which serves this answered pointer, and mounted no copy.
                    page.locator('#chat-messages .chat-load-older').filter(
                        has_text="Beginning of saved history").wait_for(state="attached", timeout=30000)
                    pointers = [(row["quiz_id"], row["quiz_state"]) for row in _api(
                        server.base_url, "GET", "/api/chat/history?chat_id=1")["messages"]
                        if row.get("system_type") == "project_question_pointer"]
                    assert pointers == [(wait["quiz_id"], "answered")] and in_main.count() == 0
                    # Every other multiline field keeps Enter as a line break: a Files editor sends nothing.
                    # The page is read from the address only on load, so Files opens in a fresh document.
                    page.goto("about:blank")
                    page.goto(server.base_url + "/#files", wait_until="domcontentloaded")
                    page.locator('#files-new-file').click()
                    editor = page.locator('.files-editor')
                    editor.click()
                    page.keyboard.type("first line")
                    page.keyboard.press("Enter")
                    page.keyboard.type("second line")
                    assert editor.input_value() == "first line\nsecond line"
                    assert editor.get_attribute("enterkeyhint") is None
                    page.locator('.files-editor-shell').screenshot(animations="disabled", path=str(screenshots / f"{shot}-files-editor.png"))
                    assert len(sent["chat"]) == 1
                    assert [path for path, _ in sent["posts"] if path.startswith(("/api/files", "/api/decisions"))] == [
                        "/api/decisions"]
                    assert list(files.iterdir()) == []
                    (screenshots / f"{shot}.json").write_text(json.dumps({
                        "engine": engine, "width": width, "answer_in": answer_in, "project": project,
                        "task_id": task["id"], "quiz_id": wait["quiz_id"], "chat_frame": sent["chat"][0],
                        "decision": decisions[0], "chat_row": row, "owner_quiz": block}, indent=2))
                finally:
                    server.stop()
        finally:
            browser.close()

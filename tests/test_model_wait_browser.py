"""Actual chat, model editor and Accounts against controlled wait responses."""
from __future__ import annotations

import copy
import json

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_reconnect_feed_browser import HISTORY_LANDED

subscription_ui = setup_browser.subscription_ui
capture = setup_browser.capture

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
TASK = "analysis-task"
# The wait picker offers an API lane per provider whose credential is stored.
API_LANE = "api:openai"


def _chip_look(card):
    """The card chip's status mark (docs/DESIGN.md "Status marks"): its body tone, the mark's shape and hue, and
    whether any of its dots animates."""
    return card.locator('[data-live-phase]').evaluate("""el => { const mark = el.querySelector(':scope > .status-mark');
        const dots = [...mark.querySelectorAll('i')].map(dot => getComputedStyle(dot).animationName);
        return {tone: el.dataset.chipTone, shape: mark.dataset.shape, hue: mark.dataset.hue,
            animation: dots.find(name => name !== 'none') || 'none'}; }""")


@pytest.fixture
def waiting_ui(subscription_ui):
    ui = subscription_ui
    page = ui["page"]
    ui["settings"]["OPENAI_API_KEY"] = "***set***"
    rows = {
        key: {"wait_id": key, "revision": 1, "task_attempt": 1, "role": role,
              "model": "claudexor::codex=gpt-test", "source": "codex",
              "credential_profile_id": account, "credential_harness": "codex",
              "reason": "quota", "reset_at": None, "auto_continue": True,
              "state": "waiting", "worker_slot_held": True}
        for key, role, account in [("main-wait", "main", "personal"), ("light-wait", "light", "work")]
    }
    history = [{"type": "task_model_wait", "task_id": TASK, "chat_id": 1,
                "ts": "2026-09-06T22:00:00Z", **copy.deepcopy(row)} for row in rows.values()]
    sockets, controls, logins = [], [], []
    acknowledgements = {}
    status = {"terminal": False, "fail_next": False, "stale_next": False}
    page.route_web_socket("**/ws", lambda socket: sockets.append(socket))
    page.add_init_script("""window.waitFrames = 0;
        window.WebSocket = class extends window.WebSocket {
            constructor(...args) { super(...args); this.addEventListener('message', () => { window.waitFrames += 1; }); }
        };""")

    def reply(route, body, code=200):
        route.fulfill(status=code, content_type="application/json", body=json.dumps(body))

    page.route("**/api/state", lambda route: reply(route, {
        "sha": "browser-fixture", "supervisor_ready": True, "projects": [], "active_chat_activities": [] if status["terminal"] else [{
            "activity_id": TASK, "kind": "managed_task", "chat_id": 1, "phase": "working",
            "model_waits": rows,
        }],
    }))
    def history_reply(route):
        messages = []
        for event in history:
            if event['type'] == 'task_done':
                messages.append({**event, 'role': 'system', 'text': '', 'system_type': 'task_summary', 'tool_calls': 1})
            else:
                messages.append({'role': 'system', 'text': '', 'system_type': 'task_model_wait',
                                 'task_id': TASK, 'ts': event['ts'],
                                 'model_waits': {event['wait_id']: event},
                                 **({'task_terminal_status': 'completed'} if status['terminal'] else {})})
        reply(route, {"messages": [*ui.get('wait_chat_messages', []), *messages]})

    page.route("**/api/chat/history*", history_reply)
    page.route(f"**/api/tasks/{TASK}", lambda route: reply(route, {
        "task_id": TASK, "status": "completed" if status["terminal"] else "running", "model_waits": rows,
    }))

    def decide(route):
        body = route.request.post_data_json
        controls.append(body)
        if status["fail_next"]:
            status["fail_next"] = False
            reply(route, {"error": "Fixture connection failed before confirmation"}, 503)
            return
        key = body["decision_id"].split(":", 2)[2]
        row = rows[key]
        if status['stale_next']:
            # The competing host update happens at dispatch, after the view's
            # observation, so a bootstrap state read cannot remove this race.
            status['stale_next'] = False
            row['revision'] = 5
        if body["request_id"] in acknowledgements:
            reply(route, {**acknowledgements[body["request_id"]], "duplicate": True, "wait": row})
            return
        if body["revision"] != row["revision"] or row["state"] != "waiting":
            reply(route, {"ok": False, "reason_code": "stale_model_wait", "error": "This wait has changed.",
                          "state": row["state"], "wait": row}, 409)
            return
        row["pending_action"] = copy.deepcopy(body)
        ack = {"ok": True, "decision_id": body["decision_id"], "request_id": body["request_id"],
               "state": "waiting", "duplicate": False, "applied": False,
               "wait": copy.deepcopy(row), "saved": body.get("persist_role") is True}
        acknowledgements[body["request_id"]] = ack
        reply(route, ack, 202)

    page.route("**/api/decisions", decide)

    def login(route):
        logins.append(route.request.post_data_json)
        reply(route, {"job_id": "fixture-login", "job": {"state": "waiting_for_input", "phase": "awaiting_user"},
                      "sequence": 1, "cursor": "fixture",
                      "deviceCode": {"flow": "chatgptDeviceCode", "verificationUrl": "https://auth.example/device", "userCode": "ABCD-1234"}})

    page.route("**/api/claudexor/login", login)
    page.route("**/api/claudexor/login/fixture-login", lambda route: reply(route, {
        "job": {"state": "waiting_for_input", "phase": "awaiting_user"}, "sequence": 1,
        "deviceCode": {"flow": "chatgptDeviceCode", "verificationUrl": "https://auth.example/device", "userCode": "ABCD-1234"},
    }))

    def emit(row):
        event = {"type": "task_model_wait", "task_id": TASK, "chat_id": 1,
                 "ts": "2026-09-06T22:01:00Z", **copy.deepcopy(row)}
        history.append(event)
        before = page.evaluate('window.waitFrames')
        sockets[-1].send(json.dumps({"type": "log", "chat_id": 1, "data": event}))
        page.wait_for_function('(before) => window.waitFrames > before', arg=before)

    def apply(key):
        row = rows[key]
        command = row.pop("pending_action")
        row["revision"] += 1
        row["applied_request_id"] = command["request_id"]
        if command["action"] == "auto_continue":
            row["auto_continue"] = command["auto_continue"]
        else:
            row["state"] = "resolved"
            row["resolution"] = "model_switched" if command["action"] == "switch" else "retry_requested"
        emit(row)

    ui.update(rows=rows, controls=controls, logins=logins, emit=emit, apply=apply, wait_status=status,
              sockets=sockets, history=history)
    # The state snapshot can render waits before initial history rebuilds the
    # card. Anchor the fixture in an actual history row before measuring DOM
    # identity, focus or computed styles on that presentation generation.
    ui["wait_chat_messages"] = [{"role": "assistant", "is_progress": True, "task_id": TASK,
                                 "text": "Preparing the model request", "ts": "2026-09-06T21:59:59Z"}]
    page.goto(ui["url"] + "/")
    page.wait_for_selector(f'.chat-live-card[data-task-id="{TASK}"][data-ts]')
    page.wait_for_selector('[data-wait-id="light-wait"]')
    yield ui


def test_multiple_waits_toggle_and_exact_role_switch_wait_for_application(waiting_ui):
    ui, page = waiting_ui, waiting_ui["page"]
    main = page.locator('[data-wait-id="main-wait"]')
    light = page.locator('[data-wait-id="light-wait"]')
    card = page.locator(f'.chat-live-card[data-task-id="{TASK}"]')
    assert page.locator('.model-wait-row').count() == 2
    assert card.locator('[data-live-phase]').inner_text() == 'Waiting for access'
    # Quota that resumes by itself: a neutral chip with still dots, nothing that moves and no owner's amber.
    assert _chip_look(card) == {'tone': 'neutral', 'shape': 'still', 'hue': 'ink', 'animation': 'none'}
    assert card.get_attribute('data-expanded') == '0'
    assert 'worker slot' in card.locator('[data-wait-slot]').inner_text()
    assert light.locator('[data-wait-auto]').is_checked()
    assert 'Account: personal' in main.inner_text()
    assert 'Account: work' in light.inner_text()
    capture(page, 'waiting-two-roles')
    light.locator('[data-wait-auto]').uncheck()
    page.wait_for_function("() => document.querySelector('[data-wait-id=light-wait] [data-wait-notice]').textContent.includes('Request accepted')")
    assert ui['controls'][-1]['decision_id'] == f'model_wait:{TASK}:light-wait'
    assert ui['controls'][-1]['revision'] == 1
    assert ui['rows']['light-wait']['auto_continue'] is True
    ui['apply']('light-wait')
    page.wait_for_selector('[data-wait-id="light-wait"] [data-wait-change]:not([disabled])')
    assert not light.locator('[data-wait-auto]').is_checked()
    light.locator('[data-wait-change]').click()
    light.locator('[data-model-role-source]').select_option(API_LANE)
    light.locator('[data-model-role-model]').fill('owner-model')
    assert not light.locator('[data-wait-persist]').is_checked()
    light.locator('[data-wait-apply]').click()
    page.wait_for_function("() => document.querySelector('[data-wait-id=light-wait] [data-wait-notice]').textContent.includes('Request accepted')")
    command = ui['controls'][-1]
    assert command['model'] == 'openai::owner-model'
    assert command['credential_profile_id'] == ''
    assert command['persist_role'] is False
    assert command['decision_id'].endswith(':light-wait')
    assert page.locator('.model-wait-row').count() == 2, 'accepted is not applied'
    ui['apply']('light-wait')
    page.wait_for_selector('[data-wait-id="light-wait"]', state='detached')
    assert main.count() == 1
    stale = {**ui['rows']['light-wait'], 'state': 'waiting', 'revision': 1}
    ui['emit'](stale)
    assert light.count() == 0


def test_edit_focus_persist_choice_and_narrow_layout_survive_sibling_updates(waiting_ui):
    ui, page = waiting_ui, waiting_ui['page']
    main = page.locator('[data-wait-id="main-wait"]')
    main.locator('[data-wait-change]').click()
    field = main.locator('[data-model-role-model]')
    field.fill('chosen-model')
    field.evaluate('(el) => { el.focus(); el.setSelectionRange(2, 5); window.waitInput = el; }')
    ui['rows']['light-wait']['revision'] += 1
    ui['rows']['light-wait']['reset_at'] = '2026-09-07T01:00:00Z'
    ui['emit'](ui['rows']['light-wait'])
    page.wait_for_function("() => document.querySelector('[data-wait-id=light-wait] [data-wait-reset]').textContent.includes('Quota resets')")
    assert field.input_value() == 'chosen-model'
    assert field.evaluate('(el) => el === window.waitInput && document.activeElement === el && el.selectionStart === 2 && el.selectionEnd === 5')
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    capture(page, 'waiting-model-picker-narrow')
    main.locator('[data-wait-persist]').check()
    main.locator('[data-wait-apply]').click()
    page.wait_for_function("() => document.querySelector('[data-wait-id=main-wait] [data-wait-notice]').textContent.includes('Settings saved')")
    assert ui['controls'][-1]['persist_role'] is True
    assert ui['controls'][-1]['decision_id'].endswith(':main-wait')


def test_auth_opens_existing_accounts_login_for_the_exact_profile(waiting_ui):
    ui, page = waiting_ui, waiting_ui['page']
    row = ui['rows']['light-wait']
    row.update(revision=2, reason='auth', worker_slot_held=False)
    ui['emit'](row)
    auth = page.locator('[data-wait-id="light-wait"]')
    page.wait_for_function("() => document.querySelector('[data-wait-id=light-wait] [data-wait-reason]').textContent === 'Sign-in required'")
    assert not auth.locator('[data-wait-auto]').is_visible()
    capture(page, 'waiting-auth')
    auth.locator('[data-wait-login]').click()
    page.wait_for_selector('[data-login-card]')
    page.wait_for_function("() => document.querySelector('[data-login-card]').textContent.includes('ABCD-1234')")
    assert page.locator('[data-settings-tab="providers"]').get_attribute('aria-selected') == 'true'
    assert ui['logins'][0]['harness'] == 'codex'
    assert ui['logins'][0]['profile_id'] == 'work'
    capture(page, 'waiting-auth-accounts')


def test_network_retry_reuses_request_and_terminal_history_suppresses_old_waits(waiting_ui):
    ui, page = waiting_ui, waiting_ui['page']
    ui['wait_status']['fail_next'] = True
    main = page.locator('[data-wait-id="main-wait"]')
    main.locator('[data-wait-retry]').click()
    main.locator('[data-wait-repeat]').wait_for(state='visible')
    request = copy.deepcopy(ui['controls'][-1])
    main.locator('[data-wait-repeat]').click()
    page.wait_for_function("() => document.querySelector('[data-wait-id=main-wait] [data-wait-notice]').textContent.includes('Request accepted')")
    assert ui['controls'][-1] == request
    terminal = {'type': 'task_done', 'task_id': TASK, 'status': 'completed',
                'artifact_status': 'ready', 'ts': '2026-09-06T22:02:00Z'}
    ui['wait_status']['terminal'] = True
    ui['history'].append(terminal)
    for socket in ui['sockets']:
        socket.send(json.dumps({'type': 'log', 'chat_id': 1, 'data': terminal}))
    page.wait_for_selector('.model-wait-row', state='detached')
    ui['emit']({**ui['rows']['light-wait'], 'revision': 99})
    assert page.locator('.model-wait-row').count() == 0
    ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.wait_for_selector(f'.chat-live-card[data-task-id="{TASK}"][data-finished="1"]')
    assert page.locator('.model-wait-row').count() == 0
    capture(page, 'waiting-terminal-replay')


def test_stale_revision_uses_latest_host_state_before_a_new_action(waiting_ui):
    ui, page = waiting_ui, waiting_ui['page']
    ui['wait_status']['stale_next'] = True
    main = page.locator('[data-wait-id="main-wait"]')
    main.locator('[data-wait-retry]').click()
    page.wait_for_function("() => document.querySelector('[data-wait-id=main-wait] [data-wait-notice]').textContent.includes('This wait has changed')")
    assert main.locator('[data-wait-repeat]').is_hidden()
    previous = ui['controls'][-1]
    main.locator('[data-wait-retry]').click()
    page.wait_for_function("() => document.querySelector('[data-wait-id=main-wait] [data-wait-notice]').textContent.includes('Request accepted')")
    assert ui['controls'][-1]['revision'] == 5
    assert ui['controls'][-1]['request_id'] != previous['request_id']


def test_wait_updates_preserve_reading_position_away_from_the_live_edge(waiting_ui):
    ui, page = waiting_ui, waiting_ui['page']
    ui['wait_chat_messages'] = [{
        'role': 'user', 'text': f'Earlier message {index}: ' + 'Context retained for the task. ' * 12,
        'client_message_id': f'earlier-{index}', 'ts': f'2026-09-06T21:{index:02d}:00Z',
    } for index in range(24)]
    ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.wait_for_selector('.chat-bubble.user')
    page.wait_for_selector('[data-wait-id="light-wait"]')
    page.wait_for_selector('#chat-messages[data-history-hydrated="true"]')
    scroll = page.locator('#chat-messages')
    scroll.evaluate('(el) => { el.style.overflowAnchor = "none"; }')
    # A real upward gesture cancels the initial history-follow frames and
    # establishes reading intent; assigning scrollTop alone does neither.
    page.wait_for_function("() => document.querySelector('#chat-messages').scrollTop > 0")
    scroll.hover()
    page.mouse.wheel(0, 300 - scroll.evaluate('(el) => el.scrollTop'))
    page.wait_for_function("() => Math.abs(document.querySelector('#chat-messages').scrollTop - 300) <= 1")
    assert scroll.evaluate('(el) => el.scrollHeight - el.clientHeight - el.scrollTop > 48')
    anchor = page.locator('.chat-bubble.user').nth(2)
    before = anchor.bounding_box()['y']
    ui['rows']['light-wait'].update(revision=2, reason='auth')
    ui['emit'](ui['rows']['light-wait'])
    page.wait_for_function("() => document.querySelector('[data-wait-id=light-wait] [data-wait-reason]').textContent === 'Sign-in required'")
    assert abs(anchor.bounding_box()['y'] - before) <= 1
    capture(page, 'waiting-scroll-preserved')


def test_mixed_access_wait_replays_and_opens_accounts_without_automatic_login(waiting_ui):
    ui, page = waiting_ui, waiting_ui['page']
    ui['rows']['light-wait'].update(revision=2, reason='auth_quota', credential_profile_id='')
    ui['emit'](ui['rows']['light-wait'])
    mixed = page.locator('[data-wait-id="light-wait"]')
    page.wait_for_function("() => document.querySelector('[data-wait-id=light-wait] [data-wait-reason]').textContent === 'Waiting for access'")
    assert mixed.locator('[data-wait-login]').is_hidden()
    assert mixed.locator('[data-wait-auto]').is_checked()
    assert mixed.locator('[data-wait-auto-label]').inner_text() == 'Continue automatically when access is restored'
    assert 'Some accounts need sign-in; others are waiting for quota.' in mixed.inner_text()
    assert 'Auto rotation' in mixed.inner_text() and 'Account:' not in mixed.inner_text()
    ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.wait_for_selector('[data-wait-id="light-wait"]')
    assert mixed.locator('[data-wait-reason]').inner_text() == 'Waiting for access'
    mixed.locator('[data-wait-auto]').uncheck()
    page.wait_for_function("() => document.querySelector('[data-wait-id=light-wait] [data-wait-notice]').textContent.includes('Request accepted')")
    ui['apply']('light-wait')
    page.wait_for_selector('[data-wait-id="light-wait"] [data-wait-change]:not([disabled])')
    assert not mixed.locator('[data-wait-auto]').is_checked()
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    capture(page, 'waiting-mixed-access-narrow')
    mixed.locator('[data-wait-settings]').click()
    page.wait_for_selector('[data-settings-tab="providers"][aria-selected="true"]')
    assert ui['logins'] == []


def test_dated_unavailable_pool_waits_visibly_without_a_sign_in_prompt(waiting_ui):
    """An engine-dated pool refusal is neither quota nor sign-in: the card says so, names the
    engine's reset, keeps automatic continuation and never offers a login."""
    ui, page = waiting_ui, waiting_ui['page']
    ui['rows']['light-wait'].update(revision=2, reason='unavailable', credential_profile_id='',
                                    reset_at='2026-09-07T01:00:00Z')
    ui['emit'](ui['rows']['light-wait'])
    pool = page.locator('[data-wait-id="light-wait"]')
    page.wait_for_function("() => document.querySelector('[data-wait-id=light-wait] [data-wait-reason]').textContent === 'Waiting for an account'")
    assert pool.locator('[data-wait-login]').is_hidden()
    assert pool.locator('[data-wait-auto]').is_checked()
    assert pool.locator('[data-wait-auto-label]').inner_text() == 'Continue automatically when access is restored'
    reset = pool.locator('[data-wait-reset]').inner_text()
    assert 'No account can serve this model now.' in reset and 'availability is unconfirmed' in reset
    assert 'Quota' not in reset and 'sign-in' not in pool.inner_text().lower()
    capture(page, 'waiting-unavailable-pool')
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    capture(page, 'waiting-unavailable-pool-narrow')
    ui['rows']['light-wait'].update(revision=3, reset_at=None)
    ui['emit'](ui['rows']['light-wait'])
    page.wait_for_function("() => document.querySelector('[data-wait-id=light-wait] [data-wait-reset]').textContent.includes('The reset time is not known.')")
    ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.wait_for_selector('[data-wait-id="light-wait"]')
    assert pool.locator('[data-wait-reason]').inner_text() == 'Waiting for an account'
    assert ui['logins'] == []


def test_direct_api_wait_never_offers_subscription_login_or_auto_rotation(waiting_ui):
    ui, page = waiting_ui, waiting_ui['page']
    ui['rows']['light-wait'].update(revision=2, reason='auth', model='openai::primary-test', source='',
        credential_profile_id='', credential_harness='', auto_continue=False, availability_observation='unavailable')
    ui['emit'](ui['rows']['light-wait'])
    card = page.locator('[data-wait-id="light-wait"]')
    page.wait_for_function("() => document.querySelector('[data-wait-id=light-wait] [data-wait-reason]').textContent === 'Waiting for provider access'")
    assert card.locator('[data-wait-login]').is_hidden() and card.locator('[data-wait-auto]').is_hidden()
    assert 'Configured API access' in card.inner_text() and 'Auto rotation' not in card.inner_text()
    assert 'Automatic access checks are unavailable.' in card.inner_text()
    capture(page, 'waiting-direct-api-access')
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    capture(page, 'waiting-direct-api-access-narrow')
    ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.wait_for_selector('[data-wait-id="light-wait"]')
    assert card.locator('[data-wait-login]').is_hidden()
    assert ui['logins'] == []


def test_wait_reconnect_preserves_unsubmitted_form_without_context_or_animation(waiting_ui):
    ui, page = waiting_ui, waiting_ui["page"]
    row = page.locator('[data-wait-id="main-wait"]')
    page.evaluate("() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))")
    waiting_card = page.locator('.chat-live-card[data-model-waiting="1"]')
    # An automatic wait is neutral and still; the same wait once only the owner can end it is the owner's move, amber.
    assert _chip_look(waiting_card) == {'tone': 'neutral', 'shape': 'still', 'hue': 'ink', 'animation': 'none'}
    ui['rows']['main-wait'].update(revision=2, auto_continue=False, requires_owner_action=True)
    ui['emit'](ui['rows']['main-wait'])
    page.wait_for_function("() => document.querySelector('.chat-live-card[data-model-waiting=\"1\"] .chat-live-phase')"
                           "?.dataset.chipTone === 'owner'")
    assert _chip_look(waiting_card) == {'tone': 'owner', 'shape': 'dot', 'hue': 'owner', 'animation': 'none'}
    row.locator('[data-wait-change]').click()
    field = row.locator('[data-model-role-model]')
    field.wait_for()
    assert row.locator('.model-role-details').count() == 0
    row.locator('[data-model-role-source]').select_option(API_LANE)
    assert row.locator('.model-role-details').count() == 0
    field.fill('unfinished-owner-model')
    row.locator('[data-wait-persist]').check()
    field.focus()
    field.evaluate("el => el.setSelectionRange(2, 8)")
    document_id = page.evaluate("window.waitDocumentId = crypto.randomUUID()")
    page.evaluate(HISTORY_LANDED)
    with page.expect_response('**/api/chat/history*'):
        ui["sockets"][-1].close()
    page.wait_for_function("() => window.historyLanded > 0")  # response received; assert the retained form below
    assert page.locator('[data-system-type="reconnect"]').count() == 0, "a reconnect adds no row"
    assert page.evaluate("window.waitDocumentId") == document_id
    assert field.input_value() == 'unfinished-owner-model'
    assert field.evaluate("el => document.activeElement === el")
    assert field.evaluate("el => [el.selectionStart, el.selectionEnd]") == [2, 8]
    assert row.locator('[data-wait-persist]').is_checked()
    assert row.locator('.model-role-details').count() == 0
    row.locator('[data-wait-apply]').click()
    page.wait_for_function("""() => document.querySelector('[data-wait-id=main-wait] [data-wait-notice]')
        .textContent.includes('Request accepted')""")
    assert ui["controls"][-1]["model"] == "openai::unfinished-owner-model"
    assert ui["controls"][-1]["persist_role"] is True
    assert set(ui["controls"][-1]) == {
        "request_id", "decision_id", "revision", "action", "model",
        "credential_profile_id", "use_local", "persist_role",
    }
    capture(page, "waiting-reconnected-draft-preserved")


def test_native_progress_keeps_its_wait_card_and_settles_only_its_controls(waiting_ui):
    ui, page = waiting_ui, waiting_ui["page"]
    task = "native-choice"

    def emit(frame):
        before = page.evaluate("window.waitFrames")
        ui["sockets"][-1].send(json.dumps(frame))
        page.wait_for_function("before => window.waitFrames > before", arg=before)

    progress = {"type": "chat", "role": "assistant", "chat_id": 1, "task_id": task,
                "is_progress": True, "content": "Checking the request",
                "ts": "2026-09-06T22:02:00Z"}
    emit(progress)
    card = page.locator(f'.chat-live-card[data-task-id="{task}"]')
    card.wait_for(state="visible")
    card.evaluate("el => { window.nativeWaitCard = el; }")
    wait = {**ui["rows"]["light-wait"], "wait_id": "native-wait", "worker_slot_held": False}
    event = {"type": "task_model_wait", "task_id": task, "chat_id": 1,
             "ts": "2026-09-06T22:02:01Z", **wait}
    emit({"type": "log", "chat_id": 1, "data": event})
    card.locator('[data-wait-id="native-wait"]').wait_for()
    emit({**progress, "content": "The same request is still waiting", "ts": "2026-09-06T22:02:02Z"})
    assert card.evaluate("el => el === window.nativeWaitCard")
    assert card.locator('[data-wait-id="native-wait"]').is_visible()
    assert card.locator('.model-wait-row').count() == 1
    assert page.locator(f'.chat-live-card[data-task-id="{TASK}"] .model-wait-row').count() == 2
    capture(page, "native-with-model-wait")
    terminal = {"type": "task_done", "task_id": task, "status": "completed",
                "ts": "2026-09-06T22:02:03Z"}
    emit({"type": "log", "chat_id": 1, "data": terminal})
    page.wait_for_selector(f'.chat-live-card[data-task-id="{task}"][data-finished="1"]')
    assert card.locator('.model-wait-row').count() == 0
    emit({"type": "log", "chat_id": 1, "data": {**event, "revision": 99}})
    assert card.locator('.model-wait-row').count() == 0
    assert page.locator(f'.chat-live-card[data-task-id="{TASK}"] .model-wait-row').count() == 2
    capture(page, "native-settled-sibling-waits-retained")


@pytest.mark.parametrize('action', ['switch', 'retry'])
def test_paid_review_controls_survive_author_terminal_and_reload(waiting_ui, tmp_path, monkeypatch, action):
    ui, page = waiting_ui, waiting_ui['page']
    monkeypatch.setenv('OUROBOROS_UI_EVIDENCE_DIR', str(tmp_path))
    print(f'PAID_REVIEW_UI_EVIDENCE {tmp_path}')
    row = ui['rows']['light-wait']
    row.update(revision=2, role='reviewer:one', worker_slot_held=False,
               model_wait_owner_id='review-operation-one',
               review_operation={'owner_id': 'review-operation-one', 'surface': 'task_acceptance',
                                 'retry_key': 'paid-panel', 'slot_id': 'one'})
    ui['emit'](row)
    waiter = page.locator('[data-wait-id="light-wait"]')
    waiter.locator('[data-wait-role]').filter(has_text='Reviewer').wait_for()
    capture(page, 'paid-review-before-author-terminal')
    terminal = {'type': 'task_done', 'task_id': TASK, 'status': 'completed',
                'artifact_status': 'ready', 'ts': '2026-09-06T22:02:00Z'}
    ui['wait_status']['terminal'] = True
    ui['history'].append(terminal)
    ui['sockets'][-1].send(json.dumps({'type': 'log', 'chat_id': 1, 'data': terminal}))
    card = page.locator(f'.chat-live-card[data-task-id="{TASK}"][data-finished="1"]')
    card.wait_for()
    assert waiter.is_visible(), 'paid review outlives the terminal author'
    assert page.locator('[data-wait-id="main-wait"]').count() == 0
    assert _chip_look(card) == {'tone': 'ok', 'shape': 'dot', 'hue': 'ok', 'animation': 'none'}
    waiter.locator('[data-wait-auto]').uncheck()
    waiter.locator('[data-wait-notice]').filter(has_text='Request accepted').wait_for()
    assert ui['controls'][-1]['decision_id'] == f'model_wait:{TASK}:light-wait'
    ui['apply']('light-wait')
    ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    card.wait_for()
    waiter.wait_for()
    assert not waiter.locator('[data-wait-auto]').is_checked()
    assert card.locator('[data-live-phase]').inner_text() == 'Done'
    capture(page, 'paid-review-after-author-terminal-reload')
    if action == 'switch':
        waiter.locator('[data-wait-change]').click()
        waiter.locator('[data-model-role-source]').select_option(API_LANE)
        waiter.locator('[data-model-role-model]').fill('replacement-reviewer')
        waiter.locator('[data-wait-apply]').click()
    else:
        waiter.locator('[data-wait-retry]').click()
    waiter.locator('[data-wait-notice]').filter(has_text='Request accepted').wait_for()
    assert ui['controls'][-1]['action'] == action
    assert ui['controls'][-1]['decision_id'] == f'model_wait:{TASK}:light-wait'
    if action == 'switch':
        assert ui['controls'][-1]['model'] == 'openai::replacement-reviewer'
    ui['apply']('light-wait')
    waiter.wait_for(state='detached')
    ui['emit']({**row, 'revision': 99, 'state': 'waiting'})
    assert waiter.count() == 0, 'resolved operation cannot be revived by a stale waiting row'
    assert card.get_attribute('data-finished') == '1'


def test_a_checked_role_save_that_did_not_land_is_told_once_its_wait_settles(waiting_ui):
    """"Also save this role in Settings" is a separate fact from the task change. While the row
    waits, the card says it inline; once the row leaves because it resolved, one toast tells a
    save that did not land (a failure carrying the applied receipt, an unknown save whose row
    already resolved, a same-request duplicate). A card closed without resolution says nothing."""
    ui, page = waiting_ui, waiting_ui['page']
    page.evaluate("""() => { window.__toasts = [];
        new MutationObserver((records) => { for (const record of records) for (const node of record.addedNodes)
            if (node.classList?.contains('toast')) window.__toasts.push(node.textContent); })
            .observe(document.body, { childList: true, subtree: true }); }""")
    toasts = lambda: page.evaluate('window.__toasts')  # noqa: E731
    answers, held = {}, []

    def decide(route):
        body = route.request.post_data_json
        ui['controls'].append(body)
        answer = answers[body['decision_id'].rsplit(':', 1)[1]].pop(0)
        if answer is None:
            held.append(route)  # answered later, after its row already resolved
            return
        code, payload = answer(body)
        route.fulfill(status=code, content_type='application/json', body=json.dumps(payload))

    page.route('**/api/decisions', decide)

    def resolved(key, request_id):
        row = ui['rows'][key]
        row.pop('pending_action', None)
        row.update(revision=row['revision'] + 1, state='resolved', resolution='model_switched',
                   applied_request_id=request_id)
        return copy.deepcopy(row)

    def pending(key, body):
        return {**copy.deepcopy(ui['rows'][key]),
                'pending_action': {k: v for k, v in body.items() if k != 'decision_id'}}

    def choose(key, model):
        view = page.locator(f'[data-wait-id="{key}"]')
        view.locator('[data-wait-change]').click()
        view.locator('[data-model-role-source]').select_option(API_LANE)
        view.locator('[data-model-role-model]').fill(model)
        view.locator('[data-wait-persist]').check()
        view.locator('[data-wait-apply]').click()
        return view

    # A failure that carries the applied receipt: the switch ran, the Settings save did not.
    answers['main-wait'] = [lambda body: (503, {
        'ok': False, 'reason_code': 'model_wait_decision_failed',
        'error': 'another settings write is still in progress after 1s; nothing was written — try again',
        'applied': True, 'saved': False, 'wait': resolved('main-wait', body['request_id'])})]
    choose('main-wait', 'busy-model').wait_for(state='detached')
    page.wait_for_function('() => window.__toasts.length === 1')
    assert toasts() == ['Main: model change applied; it was not saved in Settings.']
    assert ui['controls'][-1]['persist_role'] is True
    capture(page, 'role-save-applied-not-saved-toast')

    # The row resolves (the controller applied it) before the save answers; the save is unknown.
    answers['light-wait'] = [None]
    light = choose('light-wait', 'slow-model')
    for _ in range(500):
        if held:
            break
        page.wait_for_timeout(20)
    assert held, 'the request reached the host'
    ui['emit'](resolved('light-wait', ui['controls'][-1]['request_id']))
    light.wait_for(state='detached')
    assert len(toasts()) == 1, 'nothing is told before the Settings outcome is known'
    held.pop().fulfill(status=503, content_type='application/json', body=json.dumps({
        'ok': False, 'code': 'settings_save_timeout', 'saved': None,
        'error': 'the settings save is still running in the server and was left to finish on its own'}))
    page.wait_for_function('() => window.__toasts.length === 2')
    assert toasts()[1] == 'Light: model change applied; whether it was saved in Settings is unconfirmed.'
    capture(page, 'role-save-unconfirmed-after-resolution')

    # A same-request duplicate: still pending, the card says it inline; resolution tells it once.
    repeat = {**copy.deepcopy(ui['rows']['main-wait']), 'wait_id': 'repeat-wait', 'revision': 1, 'state': 'waiting'}
    for key in ('applied_request_id', 'resolution'):
        repeat.pop(key, None)
    ui['rows']['repeat-wait'] = repeat
    ui['emit'](repeat)
    answers['repeat-wait'] = [
        lambda body: (503, {'ok': False, 'reason_code': 'model_wait_decision_failed',
                            'error': 'Settings could not be read; nothing was written',
                            'applied': False, 'saved': False, 'wait': pending('repeat-wait', body)}),
        lambda body: (200, {'ok': True, 'decision_id': body['decision_id'], 'request_id': body['request_id'],
                            'state': 'waiting', 'duplicate': True, 'applied': False, 'saved': False,
                            'wait': pending('repeat-wait', body)})]
    view = choose('repeat-wait', 'repeat-model')
    view.locator('[data-wait-repeat]').wait_for(state='visible')
    first = copy.deepcopy(ui['controls'][-1])
    view.locator('[data-wait-repeat]').click()
    view.locator('[data-wait-notice]').filter(
        has_text='Settings were not saved. Request accepted; waiting for the task to apply it.').wait_for()
    assert ui['controls'][-1] == first, 'the same request, never a new one'
    capture(page, 'role-save-not-saved-pending-duplicate')
    assert len(toasts()) == 2
    ui['emit'](resolved('repeat-wait', first['request_id']))
    view.wait_for(state='detached')
    page.wait_for_function('() => window.__toasts.length === 3')
    assert toasts()[2] == 'Main: model change applied; it was not saved in Settings.'
    ui['emit'](ui['rows']['repeat-wait'])  # the same resolved frame again
    page.wait_for_timeout(300)
    assert len(toasts()) == 3, 'told once'

    # A card closed without resolution: the change was never applied, so nothing is told.
    task = 'closing-task'

    def frame(data):
        before = page.evaluate('window.waitFrames')
        ui['sockets'][-1].send(json.dumps(data))
        page.wait_for_function('before => window.waitFrames > before', arg=before)

    progress = {'type': 'chat', 'role': 'assistant', 'chat_id': 1, 'task_id': task, 'is_progress': True,
                'content': 'Checking the request', 'ts': '2026-09-06T22:03:00Z'}
    frame(progress)
    closing = {**copy.deepcopy(repeat), 'wait_id': 'closing-wait', 'role': 'light', 'revision': 1, 'state': 'waiting'}
    for key in ('applied_request_id', 'resolution'):
        closing.pop(key, None)
    # This wait is closing-task's own: the census (which the frame below reads again) lists it under no other task.
    frame({'type': 'log', 'chat_id': 1, 'data': {'type': 'task_model_wait', 'task_id': task, 'chat_id': 1,
                                                 'ts': '2026-09-06T22:03:01Z', **closing}})
    answers['closing-wait'] = [lambda body: (202, {
        'ok': True, 'decision_id': body['decision_id'], 'request_id': body['request_id'], 'state': 'waiting',
        'duplicate': False, 'applied': False, 'saved': False,
        'wait': {**copy.deepcopy(closing), 'pending_action': {k: v for k, v in body.items() if k != 'decision_id'}}})]
    closing_view = choose('closing-wait', 'closing-model')
    closing_view.locator('[data-wait-notice]').filter(has_text='Settings were not saved.').wait_for()
    frame({'type': 'log', 'chat_id': 1, 'data': {'type': 'task_done', 'task_id': task, 'status': 'completed',
                                                 'ts': '2026-09-06T22:03:02Z'}})
    closing_view.wait_for(state='detached')
    page.wait_for_timeout(300)
    assert len(toasts()) == 3, 'a card closed without resolution tells no Settings outcome'
    capture(page, 'role-save-closed-without-resolution')

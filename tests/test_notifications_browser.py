"""Notifications through the real SPA, driven by real socket frames.

What this proves: the shipped client, mounted by the real `app.js`, decides on
frames delivered through the real `web/modules/ws.js` dispatch — one
notification per logical event, none from a reload, hidden text by default, a
child's conclusion kept away from the owner, and a click that actually moves the
application to the source. The socket is a stub only as a TRANSPORT: frames
enter the same `ws.on` path the server's frames take, so nothing here bypasses
the client's own routing. An earlier version of this file called the module
directly and therefore could not see that the subscription was wired inside a
room; it now goes through the dispatch.

What it deliberately does NOT prove: that the OS displayed a banner. A real
system notification is not observable from the page, so `window.Notification`
is a recording stub; the OS surface, its sound and Do Not Disturb stay outside
any automated claim.
"""
from __future__ import annotations

import json

import pytest
from tests.test_subscription_setup_browser import capture, subscription_ui as _subscription_ui

subscription_ui = _subscription_ui  # re-exported pytest fixture (requested by name below)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

# Records notifications, and replaces the socket transport with one this test
# can push frames into. `ws.js` keeps its own dispatch, reconnect and queue.
CLIENT_STUB = """
window.__notifications = [];
class RecordingNotification {
    static permission = 'granted';
    static requestPermission() { return Promise.resolve('granted'); }
    constructor(title, options) {
        this.title = title;
        this.options = options || {};
        window.__notifications.push({ title, options: this.options, node: this });
    }
    close() { this.closed = true; }
}
window.Notification = RecordingNotification;

window.__sockets = [];
class TestSocket {
    constructor(url) {
        this.url = url;
        this.readyState = 1;
        window.__sockets.push(this);
        setTimeout(() => this.onopen && this.onopen({}), 0);
    }
    send() {}
    close() { this.readyState = 3; if (this.onclose) this.onclose({ code: 1000 }); }
}
TestSocket.OPEN = 1;
window.WebSocket = TestSocket;
window.__deliver = (frame) => {
    const socket = window.__sockets[window.__sockets.length - 1];
    socket.onmessage({ data: JSON.stringify(frame) });
};
"""

CATEGORY_ENABLED = "() => !document.querySelector('[data-notify-pref=task_done]').disabled"
SHOW_TEXT_SAVED = (
    "() => JSON.parse(localStorage.getItem('ouroboros.notifications') || '{}').show_text === true"
)
SOCKET_READY = "() => (window.__sockets || []).length > 0"

MAIN = 1
PROJECT = 4242


def open_notifications(page):
    page.locator('[data-nav-page="settings"]').click()
    page.wait_for_selector('#page-settings.active')
    page.locator('[data-settings-tab="appearance"]').click()
    page.wait_for_selector('[data-settings-panel="appearance"].active')
    return page.locator('[data-settings-panel="appearance"] [data-notify-settings]')


def pref(page, key):
    return page.locator(f'[data-settings-panel="appearance"] [data-notify-pref="{key}"]')


def boot(page, ui):
    page.add_init_script(CLIENT_STUB)
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    page.wait_for_function(SOCKET_READY)


def deliver(page, frame):
    page.evaluate('frame => window.__deliver(frame)', frame)
    page.wait_for_timeout(60)


def deliver_together(page, frames):
    """Dispatch several frames without yielding between them.

    `projects_changed` extends the client's room set SYNCHRONOUSLY and then
    kicks off an async refresh that replaces it from the server. Delivering the
    room announcement and the room's own frame together is how the client
    itself behaves when a new Project speaks immediately.
    """
    page.evaluate('frames => frames.forEach((frame) => window.__deliver(frame))', frames)
    page.wait_for_timeout(60)


def recorded(page):
    return page.evaluate('window.__notifications.map(n => ({ title: n.title, options: n.options }))')


def titles(page):
    return [note['title'] for note in recorded(page)]


def enable(page):
    open_notifications(page)
    pref(page, 'enabled').check()
    page.wait_for_function(CATEGORY_ENABLED)


def test_notifications_are_off_until_asked_and_then_ring_once(subscription_ui):
    ui = subscription_ui
    page = ui['page']
    boot(page, ui)

    # A new client is quiet: nothing is requested from this system.
    block = open_notifications(page)
    assert block.count() == 1
    assert pref(page, 'enabled').is_checked() is False
    assert pref(page, 'main_reply').is_checked() is False, 'ordinary Main replies start off'
    assert pref(page, 'show_text').is_checked() is False, 'message text starts hidden'
    assert pref(page, 'task_done').is_disabled() is True
    status = page.locator('[data-settings-panel="appearance"] [data-notify-status]')
    assert 'off' in status.inner_text().lower()
    capture(page, 'notifications-default-off')

    # A real terminal frame while disabled must not ring.
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 't-off', 'status': 'completed'}})
    assert recorded(page) == []

    pref(page, 'enabled').check()
    page.wait_for_function(CATEGORY_ENABLED)
    assert 'enabled' in status.inner_text().lower()
    capture(page, 'notifications-enabled')

    # The test button is the owner's only honest way to see what this system
    # does with a notification, so it must become usable and actually deliver.
    test_button = page.locator('[data-settings-panel="appearance"] [data-notify-test]')
    assert test_button.is_enabled() is True
    test_button.click()
    assert titles(page) == ['Ouroboros notifications are working']
    page.evaluate('window.__notifications.length = 0')

    # The shape a finished MANAGED root actually arrives in, then the authored
    # summary of the SAME task: one conclusion, one notification.
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 't-1', 'status': 'completed'}})
    assert titles(page) == ['Task finished']
    deliver(page, {'type': 'chat', 'role': 'system', 'system_type': 'task_summary',
                   'task_id': 't-1', 'chat_id': MAIN, 'content': 'The report is ready.'})
    assert titles(page) == ['Task finished'], 'the same conclusion must not ring twice'

    # Progress is not an event.
    deliver(page, {'type': 'chat', 'role': 'assistant', 'is_progress': True,
                   'chat_id': MAIN, 'content': 'working'})
    assert len(recorded(page)) == 1

    # A conversation turn ends with the same frame a managed task does. With
    # the ordinary-reply toggle off, that ending must stay silent rather than
    # claiming a task finished.
    deliver(page, {'type': 'chat', 'role': 'assistant', 'chat_id': MAIN,
                   'content': 'Here it is.', 'client_message_id': 'cm-direct'})
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 't-direct',
                            'status': 'completed', '_is_direct_chat': True}})
    assert len(recorded(page)) == 1, 'an ordinary reply must not ring as a finished task'


def test_a_child_conclusion_stays_with_its_parent(subscription_ui):
    ui = subscription_ui
    page = ui['page']
    boot(page, ui)
    enable(page)

    # A subagent's ordinary traffic declares the lineage its terminal omits.
    deliver(page, {'type': 'chat', 'role': 'assistant', 'is_progress': True, 'chat_id': MAIN,
                   'delegation_role': 'subagent', 'parent_task_id': 'root-1',
                   'subagent_task_id': 'child-1', 'content': 'child working'})
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 'child-1', 'status': 'completed'}})
    assert recorded(page) == [], 'a child escalates to its parent; it does not ring'

    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 'root-1', 'status': 'completed'}})
    assert titles(page) == ['Task finished']


def test_an_unopened_project_room_notifies_and_machine_traffic_does_not(subscription_ui):
    """The room gate is app.js's own project set, and no panel is involved.

    The unit suite proves `attach()` itself for a room with no chat instance
    (web/tests/notifications_attach.test.js); this proves the app-level wiring:
    the gate app.js passes really does admit a Project room the owner never
    opened, and really does refuse machine partitions.
    """
    ui = subscription_ui
    page = ui['page']
    boot(page, ui)
    enable(page)
    assert page.locator('#project-panel.open').count() == 0

    deliver_together(page, [
        {'type': 'projects_changed', 'chat_id': PROJECT},
        {'type': 'quiz', 'task_id': 't-p', 'chat_id': PROJECT,
         'quiz': {'quiz_id': 'q-p', 'state': 'open', 'wait_for_answer': True,
                  'question': 'Merge the pull request?'}},
    ])
    assert titles(page) == ['Ouroboros is waiting for your answer']
    capture(page, 'notifications-unopened-project')

    # An answered question never rings, even if its frame arrives late.
    deliver(page, {'type': 'quiz', 'task_id': 't-p2', 'chat_id': MAIN,
                   'quiz': {'quiz_id': 'q-p2', 'state': 'answered', 'wait_for_answer': True}})
    assert len(recorded(page)) == 1

    # Machine traffic is not the owner's business: the hidden partition and A2A
    # ids are refused. An ordinary positive chat this client has not registered
    # as a Project IS the owner's, exactly as the Main thread reads it.
    for chat_id in (0, -7):
        deliver(page, {'type': 'log', 'chat_id': chat_id,
                       'data': {'type': 'task_done', 'task_id': f'x{chat_id}',
                                'status': 'completed'}})
    assert len(recorded(page)) == 1
    deliver(page, {'type': 'log', 'chat_id': 994321,
                   'data': {'type': 'task_done', 'task_id': 'external', 'status': 'completed'}})
    assert len(recorded(page)) == 2, 'an external owner transport still reaches the owner'

    # An update/restart teardown reports interrupted and requeues the task: not
    # an ending, and it must not consume that task's key.
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 't-requeued',
                            'status': 'interrupted'}})
    assert len(recorded(page)) == 2
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 't-requeued',
                            'status': 'completed'}})
    assert len(recorded(page)) == 3, 'the real completion of that task still rings'

    # Click-to-source: the notifier hands the target to app.js's navigation.
    assert page.locator('#page-settings.active').count() == 1
    page.evaluate('window.__notifications.at(-1).node.onclick()')
    page.wait_for_selector('#page-chat.active')
    capture(page, 'notifications-click-to-source')


def test_choices_survive_reload_and_reload_itself_never_notifies(subscription_ui):
    ui = subscription_ui
    page = ui['page']
    # A reload replays history; seed that history so the replay has real rows
    # to render, including a finished task and an answered question.
    page.route('**/api/chat/history*', lambda route: route.fulfill(
        content_type='application/json',
        body=json.dumps({'messages': [
            {'role': 'user', 'text': 'Run the report', 'ts': '2026-09-19T10:00:00Z'},
            {'role': 'assistant', 'text': 'Done.', 'ts': '2026-09-19T10:00:01Z', 'task_id': 't-hist'},
            {'role': 'system', 'system_type': 'task_summary', 'task_id': 't-hist',
             'text': 'Task finished', 'ts': '2026-09-19T10:00:02Z'},
        ], 'progress': []})))
    boot(page, ui)
    enable(page)
    pref(page, 'show_text').check()
    page.wait_for_function(SHOW_TEXT_SAVED)

    page.reload()
    page.wait_for_selector('#chat-input')
    page.wait_for_function(SOCKET_READY)
    page.wait_for_timeout(300)
    assert recorded(page) == [], 'rendering history must stay silent'

    open_notifications(page)
    assert pref(page, 'enabled').is_checked() is True, 'the choice is client-local and persisted'
    assert pref(page, 'show_text').is_checked() is True

    # With text shown, the body carries the message.
    deliver(page, {'type': 'chat', 'role': 'system', 'system_type': 'task_summary',
                   'task_id': 't-after-reload', 'chat_id': MAIN, 'content': 'The report is ready.'})
    notes = recorded(page)
    assert len(notes) == 1
    assert notes[0]['options']['body'] == 'The report is ready.'
    capture(page, 'notifications-after-reload')


# The desktop app's bridge as `launcher_background.DesktopApi` answers it, recording each call: the
# real SPA decides from these answers exactly as inside the packaged window. `window.__nextNative`
# scripts the next `show_native_notification` answers; nothing here reaches an operating system.
DESKTOP_BRIDGE = """
window.__native = [];
window.__asks = 0;
window.__nextNative = [];
window.pywebview = { api: {
    shell_info: async () => ({ shell_version: '7.7.0', persistent_storage: true,
        native_notifications: { available: true, status: 'not_determined', platform: 'macos', reason: '' } }),
    request_native_notifications: async () => {
        window.__asks += 1;
        return { available: true, status: 'authorized', platform: 'macos', reason: '' };
    },
    show_native_notification: async (...args) => {
        window.__native.push(args);
        return window.__nextNative.shift() || { ok: true, status: 'submitted', platform: 'macos', sound: 'os' };
    },
    request_attention: async () => ({ ok: true, status: 'native_sound', sound_played: true }),
    notify_owner: async () => ({ ok: true, status: 'native_sound', sound_played: true }),
} };
"""
TOASTS = "() => [...document.querySelectorAll('#toast-stack .toast')].map((node) => node.textContent)"


def test_the_desktop_app_keeps_permission_and_the_last_hand_off_apart(subscription_ui):
    """Through the shipped Settings: switching on is the gesture that asks; an unconfirmed hand-off shows
    no second alert and says so; a failed one falls back and says so without withdrawing the permission."""
    ui = subscription_ui
    page = ui['page']
    page.add_init_script(DESKTOP_BRIDGE)
    boot(page, ui)
    enable(page)
    status = page.locator('[data-settings-panel="appearance"] [data-notify-status]')
    page.wait_for_function("() => window.__asks === 1")
    page.wait_for_function("() => /System notifications are on/.test("
                           "document.querySelector('[data-settings-panel=\"appearance\"] [data-notify-status]').textContent)")
    assert 'shows each alert' not in status.inner_text()
    capture(page, 'notifications-desktop-on')

    page.evaluate("window.__nextNative.push({ ok: null, status: 'unknown', platform: 'macos', reason: 'no_completion' })")
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 't-unknown', 'status': 'completed'}})
    page.wait_for_function("() => window.__native.length === 1")
    page.wait_for_timeout(120)
    assert page.evaluate(TOASTS) == [], 'it may still appear: no copy in the app'
    assert recorded(page) == [], 'and no browser banner'
    assert 'did not confirm the last alert' in status.inner_text()
    capture(page, 'notifications-desktop-unconfirmed')

    page.evaluate("window.__nextNative.push({ ok: false, status: 'failed', platform: 'macos', reason: 'no_center' })")
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 't-failed', 'status': 'completed'}})
    page.wait_for_function("() => window.__native.length === 2")
    page.wait_for_function("() => window.__notifications.length === 1")
    assert titles(page) == ['Task finished'], 'never reached the system: the browser banner this client allows'
    assert page.evaluate(TOASTS) == []
    assert 'could not be handed to the system (no_center)' in status.inner_text()
    assert 'so it fell back to a browser banner or the app' in status.inner_text(), 'never claimed as the app'
    assert page.evaluate("window.__asks") == 1, 'a live alert never asks for permission'
    capture(page, 'notifications-desktop-fell-back')


# The bridge of a desktop app before system notifications, as 7.6.0's `launcher.py` exposed it: no
# `shell_info` or `show_native_notification`, and `notify_owner` IS `request_attention`, which, with the
# window hidden in background mode, shows a Windows balloon with Windows' sound whatever `sound` says.
OLD_DESKTOP_BRIDGE = """
window.__attention = [];
const attention = async (...args) => {
    window.__attention.push(args);
    return { ok: true, status: 'background', banner: true, sound_played: false };
};
window.pywebview = { api: { request_attention: attention, notify_owner: attention } };
"""


def test_sound_off_never_calls_an_older_desktop_app(subscription_ui):
    """That app's hidden-window balloon would sound: Sound off keeps to a silent browser banner, Sound on
    still reaches the app, and Settings never calls the banner fallback the app."""
    ui = subscription_ui
    page = ui['page']
    page.add_init_script(OLD_DESKTOP_BRIDGE)
    boot(page, ui)
    enable(page)
    pref(page, 'sound').uncheck()
    page.wait_for_function("() => JSON.parse(localStorage.getItem('ouroboros.notifications') || '{}').sound === false")
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 't-old-silent', 'status': 'completed'}})
    page.wait_for_function("() => window.__notifications.length === 1")
    assert page.evaluate("window.__attention") == [], 'never asked: its balloon would play a sound'
    assert recorded(page)[0]['options']['silent'] is True
    status = page.locator('[data-settings-panel="appearance"] [data-notify-status]')
    assert 'predates them, so alerts fall back to a browser banner or the app' in status.inner_text()

    pref(page, 'sound').check()
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 't-old-loud', 'status': 'completed'}})
    page.wait_for_function("() => window.__attention.length === 1")
    assert page.evaluate("window.__attention[0][0]") is True, 'Sound on keeps the earlier path'
    capture(page, 'notifications-old-app-sound-off')

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
import os
from pathlib import Path

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

    ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
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
window.__systemPermission = 'not_determined';
window.pywebview = { api: {
    shell_info: async () => ({ shell_version: '7.7.0', persistent_storage: true,
        native_notifications: { available: true, status: window.__systemPermission, platform: 'macos', reason: '' } }),
    request_native_notifications: async () => {
        window.__asks += 1;
        // A granted request changes what the next fresh system read observes.
        window.__systemPermission = 'authorized';
        return { available: true, status: window.__systemPermission, platform: 'macos', reason: '' };
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


def test_an_older_desktop_app_is_never_called_sound_on_or_off(subscription_ui):
    """That app's hidden-window balloon would sound, and an older launcher may ignore `sound=false`: its alerts
    are a browser banner under the Sound choice, and Settings never calls the banner fallback the app."""
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
    assert recorded(page)[0]['options']['silent'] is True
    status = page.locator('[data-settings-panel="appearance"] [data-notify-status]')
    assert 'predates them, so alerts fall back to a browser banner or the app' in status.inner_text()

    pref(page, 'sound').check()
    deliver(page, {'type': 'log', 'chat_id': MAIN,
                   'data': {'type': 'task_done', 'task_id': 't-old-sound-on', 'status': 'completed'}})
    page.wait_for_function("() => window.__notifications.length === 2")
    assert recorded(page)[1]['options']['silent'] is False, 'the banner\'s sound is the system\'s'
    assert page.evaluate("window.__attention") == [], 'never asked, Sound on or off'
    capture(page, 'notifications-old-app-never-called')


# A recording Web Audio double: nothing reaches a speaker. Each started oscillator is kept with its frequency
# and the peak its gain ramps to, so the test reads what WOULD sound.
AUDIO_STUB = """
window.__audio = { contexts: 0, tones: [] };
class RecordingAudioContext {
    constructor() { this.currentTime = 0; this.state = 'running'; this.destination = {}; window.__audio.contexts += 1; }
    resume() { return Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
    createOscillator() {
        return {
            type: '', frequency: { value: 0 },
            connect(node) { this.output = node; return node; },
            start(at) { window.__audio.tones.push({ hz: this.frequency.value, type: this.type, at, peak: this.output && this.output.peak }); },
            stop(at) { this.stoppedAt = at; },
        };
    }
    createGain() {
        const node = { connect(next) { return next; } };
        node.gain = {
            value: 1,
            setValueAtTime() {},
            linearRampToValueAtTime(value) { node.peak = value; },
            exponentialRampToValueAtTime() {},
        };
        return node;
    }
}
window.AudioContext = RecordingAudioContext;
window.webkitAudioContext = RecordingAudioContext;
"""
TONES = "() => window.__audio.tones"


def volume(page):
    return page.locator('[data-settings-panel="appearance"] [data-notify-volume]')


def capture_block(locator, name):
    """The Notifications block alone, as rendered, beside the full-page captures."""
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        target = Path(root)
        target.mkdir(parents=True, exist_ok=True)
        locator.screenshot(path=str(target / f"{name}.png"), animations="disabled")


def test_app_tone_volume_previews_only_on_press_and_survives_reload(subscription_ui):
    """The approved row under Sound: a level stored by this client, a slider that plays nothing, and the
    existing small capsule button that plays the soft tone once. Light, dark and narrow, as rendered."""
    ui = subscription_ui
    page = ui['page']
    page.add_init_script(AUDIO_STUB)
    boot(page, ui)
    block = open_notifications(page)
    row = block.locator('.notify-tone-volume')
    assert row.count() == 1
    # Directly under Sound, inside the same card as the checkboxes.
    assert page.evaluate("""() => {
        const sound = document.querySelector('[data-settings-panel="appearance"] [data-notify-pref="sound"]').closest('label');
        return sound.nextElementSibling === document.querySelector('[data-settings-panel="appearance"] .notify-tone-volume');
    }""")
    assert row.locator('label[for="notify-tone-volume"]').inner_text() == 'App tone volume'
    preview = row.locator('[data-notify-preview]')
    assert preview.get_attribute('class') == 'btn btn-default btn-sm'
    assert preview.inner_text() == 'Preview tone'
    assert 'not the system notification volume' in row.locator('.settings-inline-note').inner_text()
    value = row.locator('[data-notify-volume-value]')
    assert value.inner_text() == '40%'
    assert volume(page).is_disabled() and preview.is_disabled(), 'inert while notifications are off'

    enable(page)
    assert volume(page).is_enabled() and preview.is_enabled()
    for level in ('70', '0', '25'):
        volume(page).fill(level)
    page.wait_for_function("() => JSON.parse(localStorage.getItem('ouroboros.notifications') || '{}').tone_volume === 25")
    assert value.inner_text() == '25%'
    page.wait_for_timeout(150)
    assert page.evaluate("window.__audio.contexts") == 0, 'moving the slider opens no audio device'
    assert page.evaluate(TONES) == []

    page.locator('[data-theme-choice="light"]').click()
    row.scroll_into_view_if_needed()
    capture(page, 'notifications-tone-light')
    capture_block(block, 'notifications-tone-light-block')
    page.locator('[data-theme-choice="dark"]').click()
    assert page.locator('html').get_attribute('data-theme') == 'dark'
    row.scroll_into_view_if_needed()
    capture(page, 'notifications-tone-dark')
    capture_block(block, 'notifications-tone-dark-block')

    preview.click()
    page.wait_for_function("() => window.__audio.tones.length === 1")
    (tone,) = page.evaluate(TONES)
    assert tone['hz'] == 523.25 and tone['type'] == 'sine'
    assert abs(tone['peak'] - 0.05 * 0.25 ** 2) < 1e-12, tone

    pref(page, 'sound').uncheck()
    assert volume(page).is_disabled() and preview.is_disabled(), 'inert while Sound is off'
    row.scroll_into_view_if_needed()
    capture_block(block, 'notifications-tone-sound-off-block')
    pref(page, 'sound').check()

    ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.wait_for_selector('#chat-input')
    block = open_notifications(page)
    assert volume(page).input_value() == '25', 'the level is client-local and persisted'
    assert block.locator('[data-notify-volume-value]').inner_text() == '25%'
    assert page.evaluate(TONES) == [], 'a reload plays nothing'

    page.set_viewport_size({'width': 390, 'height': 844})
    page.locator('[data-theme-choice="light"]').click()
    block.locator('.notify-tone-volume').scroll_into_view_if_needed()
    width = page.evaluate("""() => {
        const row = document.querySelector('[data-settings-panel="appearance"] .notify-tone-volume');
        const card = row.closest('.settings-effort-card');
        return { row: row.scrollWidth, card: card.clientWidth };
    }""")
    assert width['row'] <= width['card'], width
    capture(page, 'notifications-tone-narrow')
    capture_block(block, 'notifications-tone-narrow-block')


# A current desktop app whose `show_native_notification` answer the test holds, so the owner's choices can
# change while it is pending (research c7845ef5). Nothing reaches an operating system.
PENDING_DESKTOP_BRIDGE = """
window.__toasts = [];
new MutationObserver((records) => {
    for (const record of records) {
        for (const node of record.addedNodes) {
            if (node.nodeType === 1 && node.classList.contains('toast')) window.__toasts.push(node.textContent);
        }
    }
}).observe(document, { childList: true, subtree: true });
window.__native = [];
window.__attention = [];
window.__release = [];
window.pywebview = { api: {
    // The preparation read, plus a fresh read after a definite failure;
    // this fixture deliberately withholds a positive sound setting.
    shell_info: async () => ({ shell_version: '7.7.0', persistent_storage: true,
        native_notifications: window.__systemRead || { available: true, status: 'authorized', platform: 'macos', reason: '' } }),
    request_native_notifications: async () => ({ available: true, status: 'authorized', platform: 'macos', reason: '' }),
    show_native_notification: (...args) => {
        window.__native.push(args);
        return new Promise((resolve) => window.__release.push(resolve));
    },
    // Model a sound-capable Windows/legacy bridge; the page must request it silently.
    notify_owner: async (sound = true, title = '', body = '', cueWhenVisible = true) => {
        window.__attention.push([sound, title, body]);
        if (!cueWhenVisible) return { ok: false, status: 'visible' };
        return sound ? { ok: true, status: 'native_sound', sound_played: true } : { ok: true, status: 'window_only', sound_played: false };
    },
    request_attention: async (sound = true) => (sound ? { ok: true, status: 'native_sound', sound_played: true }
        : { ok: true, status: 'window_only', sound_played: false }),
} };
"""
REFUSAL = {'ok': False, 'status': 'failed', 'platform': 'macos', 'reason': 'no_center'}
APP_BUNDLE_REFUSAL = {'ok': False, 'status': 'unavailable', 'platform': 'macos', 'reason': 'not_an_app_bundle'}
DENIED_REFUSAL = {'ok': False, 'status': 'denied', 'platform': 'macos', 'reason': ''}
STORED = "() => JSON.parse(localStorage.getItem('ouroboros.notifications') || '{}')"


def release(page, answer):
    page.evaluate("answer => window.__release.shift()(answer)", answer)
    page.wait_for_timeout(120)


def finish(page, task_id, count):
    deliver(page, {'type': 'log', 'chat_id': MAIN, 'data': {'type': 'task_done', 'task_id': task_id, 'status': 'completed'}})
    page.wait_for_function("n => window.__native.length === n", arg=count)


def test_a_choice_changed_while_the_system_answer_is_pending_holds(subscription_ui):
    """Through the shipped Settings: Sound off or notifications off while the desktop app's answer is pending
    holds for the fallback that had not started; the launcher is only ever asked silently; the page plays no
    tone without a positive read taken after a definite failure; a client without native permission/sound
    evidence stays quiet, and an explicit denial is never read past."""
    ui = subscription_ui
    page = ui['page']
    page.add_init_script(CLIENT_STUB)
    page.add_init_script(AUDIO_STUB)
    page.add_init_script(PENDING_DESKTOP_BRIDGE)
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    page.wait_for_function(SOCKET_READY)
    enable(page)

    # Control: unchanged choices. The refusal falls back to the browser banner, which keeps the system's sound.
    finish(page, 't-control', 1)
    assert page.evaluate("window.__native[0][2]") is True, 'the system is asked for its own sound'
    release(page, REFUSAL)
    page.wait_for_function("() => window.__notifications.length === 1")
    assert recorded(page)[0]['options']['silent'] is False
    assert page.evaluate("window.__attention") == [[False, 'Task finished', '']], 'never the launcher\'s own sound'

    # Sound off while pending: the same fallback, silent.
    finish(page, 't-muted', 2)
    pref(page, 'sound').uncheck()
    page.wait_for_function(STORED + ".sound === false")
    release(page, REFUSAL)
    page.wait_for_function("() => window.__notifications.length === 2")
    assert recorded(page)[1]['options']['silent'] is True
    pref(page, 'sound').check()
    page.wait_for_function(STORED + ".sound === true")

    # Notifications off while pending: nothing at all.
    finish(page, 't-off', 3)
    asked = len(page.evaluate("window.__attention"))
    pref(page, 'enabled').uncheck()
    page.wait_for_function(STORED + ".enabled === false")
    release(page, REFUSAL)
    assert len(recorded(page)) == 2, 'no browser banner'
    assert len(page.evaluate("window.__attention")) == asked, 'no attention cue'
    assert page.evaluate("window.__toasts") == [], 'no in-app alert'

    # Today's launcher outside an app bundle, and a page without banner permission: the in-app alert, the
    # attention cue asked silently, and no page sound (the permission was never read: unknown is not a grant).
    pref(page, 'enabled').check()
    page.wait_for_function(CATEGORY_ENABLED)
    page.evaluate("window.Notification.permission = 'default'")
    finish(page, 't-bundle', 4)
    release(page, APP_BUNDLE_REFUSAL)
    page.wait_for_function("() => window.__toasts.length === 1")
    assert page.evaluate("window.__attention.at(-1)") == [False, 'Task finished', '']
    assert page.evaluate("window.__audio.contexts") == 0, 'the page opened no audio device for any alert'

    # The app now reads authorized with its sound on (the launcher seam): an explicit denial at the hand-off
    # stays quiet, and only a definite failure, read again after it, plays the one soft tone.
    page.evaluate("window.__systemRead = { available: true, status: 'authorized', platform: 'macos', reason: '',"
                  " sound_setting: 'enabled' }")
    finish(page, 't-denied', 5)
    release(page, DENIED_REFUSAL)
    page.wait_for_function("() => window.__toasts.length === 2")
    assert page.evaluate(TONES) == [], 'an explicit denial is never read past'
    finish(page, 't-failed', 6)
    release(page, REFUSAL)
    page.wait_for_function("() => window.__audio.tones.length === 1")
    assert page.evaluate("window.__audio.tones[0].hz") == 523.25
    assert page.evaluate("window.__toasts.length") == 3
    capture(page, 'notifications-desktop-late-mute')

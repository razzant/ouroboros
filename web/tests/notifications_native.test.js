/* System notifications through the desktop app (DESIGN §9; owner decisions 1A, 2A, 3A).

   The bridge is a recording double of launcher_background.DesktopApi: nothing here
   reaches an operating system, plays a sound or shows a banner. */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
    DEFAULT_NOTIFY_PREFS,
    NOTIFY_PREFS_KEY,
    createNotifier,
    getNotifier,
    nativeStatusText,
    resetNotifier,
} from '../modules/notifications.js';
import { loadDesktopShell, mountDesktopShell } from '../modules/desktop_shell.js';
import { recordingAudio, settle } from './notification_audio_fixture.js';

const ON = { ...DEFAULT_NOTIFY_PREFS, enabled: true };
const FINISHED = { role: 'system', system_type: 'task_summary', task_id: 't1', chat_id: 7, content: 'Report ready' };
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function storage(prefs = ON) {
    const map = new Map([[NOTIFY_PREFS_KEY, JSON.stringify(prefs)]]);
    return { map, getItem: (k) => (map.has(k) ? map.get(k) : null), setItem: (k, v) => { map.set(k, String(v)); } };
}

function nodes(selectors = {}) {
    return {
        addEventListener() {},
        removeEventListener() {},
        querySelectorAll: (selector) => selectors[selector] || [],
    };
}

/** A notifier whose desktop app answers `answer` (a value or a function of the call). */
function fixture({ answer, prefs = ON, granted = false, extra = {}, documentRef = nodes() } = {}) {
    const calls = [];
    const banners = [];
    const toasts = [];
    const activated = [];
    let browserAsks = 0;
    class Banner {
        static permission = granted ? 'granted' : 'default';
        static async requestPermission() { browserAsks += 1; return 'granted'; }
        constructor(title, options) { banners.push({ title, options }); }
        close() {}
    }
    const audio = recordingAudio();
    const hostApi = {
        show_native_notification: (...args) => {
            calls.push(['show_native_notification', ...args]);
            return typeof answer === 'function' ? answer(...args) : answer;
        },
        // Model a sound-capable Windows/legacy bridge; the page must request it silently.
        request_attention: (...args) => {
            calls.push(['request_attention', ...args]);
            return args[0] ? { ok: true, status: 'native_sound', sound_played: true } : { ok: true, status: 'window_only', sound_played: false };
        },
        ...extra,
    };
    const notifier = createNotifier({
        storage: storage(prefs),
        notificationCtor: Banner,
        audioContextCtor: audio.ctor,
        showToast: (line) => { toasts.push(line); return null; },
        onActivate: (target) => activated.push(target),
        documentRef,
        hostApi,
    });
    return { notifier, calls, banners, toasts, activated, audio, tones: () => audio.tones.length, browserAsks: () => browserAsks };
}

test('a submitted system notification is the one surface: no browser banner, no toast, no page tone', async () => {
    const fx = fixture({ answer: { ok: true, status: 'submitted', sound: 'os' }, granted: true });
    const out = fx.notifier.handleFrame(FINISHED, { kind: 'chat', isMain: true });
    await tick();
    assert.equal(out.surface, 'native');
    assert.equal(fx.calls.length, 1);
    const [name, title, body, sound, token] = fx.calls[0];
    assert.equal(name, 'show_native_notification');
    assert.equal(title, 'Task finished');
    assert.equal(body, '', 'message text stays private unless the owner turned it on');
    assert.equal(sound, true, 'the OS plays its own sound (1A)');
    assert.match(token, /^[A-Za-z0-9_.:-]{1,96}$/, 'the token is the launcher\'s accepted shape');
    assert.equal(fx.banners.length, 0);
    assert.deepEqual(fx.toasts, []);
    assert.equal(fx.tones(), 0);
    // The same event again is still one notification.
    assert.equal(fx.notifier.handleFrame(FINISHED, { kind: 'chat', isMain: true }), null);
    fx.notifier.destroy();
});

test('Sound off asks the system for a silent notification', async () => {
    const fx = fixture({ answer: { ok: true, status: 'submitted', sound: 'off' }, prefs: { ...ON, sound: false, show_text: true } });
    fx.notifier.handleFrame(FINISHED, { kind: 'chat', isMain: true });
    await tick();
    assert.equal(fx.calls[0][2], 'Report ready', 'text shown only when the owner turned it on');
    assert.equal(fx.calls[0][3], false);
    assert.equal(fx.tones(), 0);
    fx.notifier.destroy();
});

test('a click on the system notification opens the same source a banner would', async () => {
    const fx = fixture({ answer: { ok: true, status: 'submitted', sound: 'os' } });
    fx.notifier.handleFrame(
        { task_id: 't2', chat_id: 9, quiz: { quiz_id: 'q1', state: 'open', wait_for_answer: true } },
        { kind: 'quiz' },
    );
    await tick();
    const token = fx.calls[0][4];
    assert.equal(fx.notifier.activateNative(token), true);
    assert.deepEqual(fx.activated, [{ chatId: 9, taskId: 't2', quizId: 'q1' }]);
    assert.equal(fx.notifier.activateNative(token), false, 'one click, one navigation');
    assert.equal(fx.notifier.activateNative('n99-unknown'), false, 'an unknown token (after a reload) only opens the window');
    fx.notifier.destroy();
});

/* A refused alert still reaches the owner, through the attention cue asked WITHOUT the launcher's sound,
   and adds no page sound: a refusal is no evidence that the system lets Ouroboros sound (owner decision,
   a2199222). A permission not yet asked, a denial, one the system could not read (`unavailable`, or its
   own error to the question) and a failure stay quiet, and so does a `sound_setting` on the refusal: after
   a definite failure the tone reads only the app's `shell_info()`, asked after that failure
   (notifications_sound.test.js), which this fixture's app does not offer. */
for (const [status, extra] of [
    ['not_determined', {}],
    ['not_determined', { reason: 'authorization_error: UNErrorDomain 1: x' }],
    ['denied', {}],
    ['unavailable', { reason: 'not_an_app_bundle' }],
    ['failed', { reason: 'no_center' }],
    ['failed', { reason: 'no_center', sound_setting: 'enabled' }],
]) {
    test(`a ${status} refusal ${JSON.stringify(extra)} falls back with no sound`, async () => {
        const fx = fixture({ answer: { ok: false, status, platform: 'macos', reason: '', ...extra } });
        assert.equal(fx.notifier.handleFrame(FINISHED, { kind: 'chat', isMain: true }).surface, 'native');
        await settle();
        assert.deepEqual(fx.toasts, ['Task finished'], 'the alert still reaches the owner');
        assert.deepEqual(fx.calls.filter(([name]) => name === 'request_attention'), [['request_attention', false]],
            'the attention cue, never the launcher\'s own sound');
        assert.equal(fx.audio.contexts.length, 0, 'no page tone: no audio device is even opened');
        fx.notifier.destroy();
    });
}

test('a call that throws before sending falls back silently; a failed or unreadable answer after it is unknown', async () => {
    const thrown = fixture({ answer: () => { throw new Error('bridge gone'); } });
    thrown.notifier.handleFrame(FINISHED, { kind: 'chat', isMain: true });
    await settle();
    assert.deepEqual(thrown.toasts, ['Task finished'], 'nothing was sent: the alert falls back');
    assert.equal(thrown.tones(), 0, 'no permission fact came back: quiet');
    thrown.notifier.destroy();
    for (const answer of [() => Promise.reject(new Error('closed')), () => undefined, () => ({ ok: false }),
        () => ({ status: 'denied' }), () => ({ ok: true, status: 'weird' })]) {
        const status = { textContent: '' };
        const fx = fixture({ answer, granted: true, documentRef: nodes({ '[data-notify-status]': [status] }) });
        fx.notifier.configure({ shell: { desktop: true, version: '7.7.0', native: { status: 'authorized' } } });
        fx.notifier.handleFrame(FINISHED, { kind: 'chat', isMain: true });
        await settle();
        assert.equal(fx.banners.length, 0, 'it may have reached the system: no second alert');
        assert.deepEqual(fx.toasts, []);
        assert.equal(fx.tones(), 0);
        assert.equal(fx.calls.some(([name]) => name === 'request_attention'), false);
        assert.match(status.textContent, /did not confirm the last alert/);
        assert.equal(fx.notifier.activateNative(fx.calls[0][4]), true, 'its click still opens the source');
        fx.notifier.destroy();
    }
});

test('an app without the native method falls back to the page, asking only its silent window cue (3A)', async () => {
    // request_attention without notify_owner: its `false` only raises the window (notifications.test.js).
    const fx = fixture({ answer: null, extra: { show_native_notification: undefined } });
    assert.equal(fx.notifier.handleFrame(FINISHED, { kind: 'chat', isMain: true }).surface, 'in_app');
    await settle();
    assert.deepEqual(fx.calls, [['request_attention', false]], 'never the system method it lacks, never its own sound');
    assert.deepEqual(fx.toasts, ['Task finished']);
    assert.equal(fx.tones(), 0);
    fx.notifier.destroy();
});

/* The bridge of an app before system notifications, as 7.6.0's launcher.py exposed it: `notify_owner`
   IS `request_attention`, and while the window is hidden in background mode Background.attention hands
   the indicator only the text, so WindowsTray queues a balloon that plays Windows' sound whatever
   `sound` says. That launcher is frozen into the installed app; only the page can keep Sound off. */
function oldWindowsBridge({ hidden = true } = {}) {
    const balloons = [];
    const attention = (sound = true, title = '', body = '', cueWhenVisible = true) => {
        if (hidden) {
            balloons.push({ title, body, requested: Boolean(sound) });
            return { ok: true, status: 'background', banner: true, sound_played: false };
        }
        if (!cueWhenVisible) return { ok: false, status: 'visible' };
        return { ok: true, status: 'native_sound', sound_played: Boolean(sound) };
    };
    return { balloons, api: { show_native_notification: undefined, request_attention: attention, notify_owner: attention } };
}

test('an older app\'s audible hidden-window balloon is never reached, Sound on or off', async () => {
    for (const sound of [false, true]) {
        for (const granted of [false, true]) {
            const old = oldWindowsBridge();
            const fx = fixture({ answer: null, granted, prefs: { ...ON, sound }, extra: old.api });
            fx.notifier.handleFrame(FINISHED, { kind: 'chat', isMain: true });
            await settle();
            assert.deepEqual(old.balloons, [], 'no balloon: that launcher would sound whatever it is asked');
            assert.equal(fx.tones(), 0);
            if (granted) {
                assert.equal(fx.banners.length, 1, 'the browser banner this client allows');
                assert.equal(fx.banners[0].options.silent, !sound, 'its sound is the system\'s, under the Sound choice');
            } else {
                assert.deepEqual(fx.toasts, ['Task finished'], 'the alert still reaches the owner, silently');
            }
            fx.notifier.destroy();
        }
    }
});

test('Sound off still asks a current app whose system notification fell back: its balloon can be silent', async () => {
    const asked = [];
    const fx = fixture({
        answer: { ok: false, status: 'failed', reason: 'balloon_not_submitted' },
        prefs: { ...ON, sound: false },
        extra: { notify_owner: (...args) => { asked.push(args); return { ok: true, status: 'background', banner: true }; } },
    });
    fx.notifier.handleFrame(FINISHED, { kind: 'chat', isMain: true });
    await tick();
    await tick();
    assert.equal(asked.length, 1);
    assert.equal(asked[0][0], false, 'the current launcher sends it with NIIF_NOSOUND or refuses it');
    fx.notifier.destroy();
});

test('enabling asks the desktop app, not the browser, and the answer reaches Settings', async () => {
    const status = { textContent: '' };
    const attention = { textContent: 'stale' };
    let asked = 0;
    const fx = fixture({
        prefs: DEFAULT_NOTIFY_PREFS,
        answer: { ok: true, status: 'submitted' },
        documentRef: nodes({ '[data-notify-status]': [status], '[data-notify-attention-status]': [attention] }),
        extra: { request_native_notifications: () => { asked += 1; return { available: true, status: 'authorized', platform: 'macos' }; } },
    });
    fx.notifier.configure({ shell: { desktop: true, version: '7.7.0', native: { status: 'not_determined' } } });
    assert.match(status.textContent, /off/);
    await fx.notifier.setPref('enabled', true);
    assert.equal(asked, 1);
    assert.equal(fx.browserAsks(), 0);
    assert.match(status.textContent, /System notifications are on/);
    assert.equal(attention.textContent, '', 'no second line while alerts go to the system');
    fx.notifier.destroy();
});

test('the status line names what the desktop app can do, and the old app honestly', () => {
    const shell = { desktop: true, version: '7.7.0' };
    const on = { native: true, shell };
    assert.equal(nativeStatusText({ ...on, enabled: false, capability: { status: 'authorized' } }), '');
    assert.equal(nativeStatusText({ enabled: true, native: true, shell: { desktop: false } }), '', 'a browser keeps the browser line');
    const authorized = nativeStatusText({ ...on, capability: { status: 'authorized' } });
    assert.match(authorized, /alerts go to the system, whose notification settings, Focus or Do Not Disturb decide/);
    assert.doesNotMatch(authorized, /notification server/, 'no limit reported, none named');
    const linux = nativeStatusText({ ...on, capability: { status: 'authorized', limits: ['no_sound', 'no_click'] } });
    assert.match(linux, /notification server plays no sound for them and cannot open their source when clicked\.$/);
    assert.doesNotMatch(authorized, /shows each alert|plays its own sound/, 'submission is not a promise of a seen banner');
    assert.match(nativeStatusText({ ...on, capability: { status: 'not_determined' } }), /not been asked yet.*Test asks it/);
    const failedAsk = nativeStatusText({ ...on, capability: { status: 'not_determined', reason: 'authorization_error: UNErrorDomain 1: x' } });
    assert.match(failedAsk, /could not ask whether Ouroboros may show notifications \(authorization_error: UNErrorDomain 1: x\), so alerts fall back to the app\./);
    assert.doesNotMatch(failedAsk, /denied|not been asked yet/, 'the system\'s error is neither a denial nor an unasked question');
    assert.match(nativeStatusText({ ...on, capability: { status: 'denied' } }), /denied.*fall back to the app\..*notification settings/);
    assert.match(nativeStatusText({ ...on, banner: true, capability: { status: 'denied' } }),
        /fall back to a browser banner or the app/, 'a banner this client allows is never called the app');
    assert.doesNotMatch(nativeStatusText({ ...on, banner: true, capability: { status: 'unavailable' } }), /in the app instead/);
    assert.match(nativeStatusText({ ...on, capability: { status: 'unavailable', reason: 'not_an_app_bundle' } }),
        /unavailable to this desktop app \(7\.7\.0\) here \(not_an_app_bundle\)/);
    const legacy = nativeStatusText({ native: false, shell: { desktop: true, version: '6.82.0' } });
    assert.match(legacy, /This desktop app \(6\.82\.0\) has no system-notification method here, so alerts fall back/);
    assert.doesNotMatch(legacy, /predates|current Ouroboros desktop app|Update/, 'a missing method proves no remedy');
});

test('permission and the last hand-off are two facts: a failure never reads as a withdrawn permission', () => {
    const on = { native: true, shell: { desktop: true }, capability: { status: 'authorized' } };
    assert.match(nativeStatusText({ ...on, outcome: { status: 'failed', reason: 'balloon_not_submitted' } }),
        /last alert could not be handed to the system \(balloon_not_submitted\), so it fell back to the app\./);
    assert.match(nativeStatusText({ ...on, banner: true, outcome: { status: 'failed' } }),
        /so it fell back to a browser banner or the app\./);
    assert.match(nativeStatusText({ ...on, outcome: { status: 'unknown' } }),
        /did not confirm the last alert in time; it may still appear there, so the app did not repeat it/);
    assert.match(nativeStatusText({ ...on, outcome: { status: 'submitted' } }), /System notifications are on/);
    assert.match(nativeStatusText({ ...on, capability: { status: 'denied' }, outcome: { status: 'failed' } }), /denied/,
        'a refused permission names the permission, not the hand-off');
});

test('an unanswered hand-off is unknown: no second alert or sound, and its late click still opens the source', async () => {
    const status = { textContent: '' };
    const fx = fixture({
        answer: { ok: null, status: 'unknown', platform: 'macos', reason: 'no_completion' },
        granted: true,
        documentRef: nodes({ '[data-notify-status]': [status] }),
    });
    fx.notifier.configure({ shell: { desktop: true, version: '7.7.0', native: { status: 'authorized' } } });
    fx.notifier.handleFrame(
        { task_id: 't2', chat_id: 9, quiz: { quiz_id: 'q1', state: 'open', wait_for_answer: true } }, { kind: 'quiz' },
    );
    await tick();
    await tick();
    assert.deepEqual(fx.toasts, [], 'it may still appear: no in-app copy');
    assert.equal(fx.banners.length, 0, 'no browser banner either');
    assert.equal(fx.tones(), 0);
    assert.equal(fx.calls.filter(([name]) => name === 'request_attention').length, 0, 'and no attention sound');
    assert.match(status.textContent, /did not confirm the last alert/);
    // The system shows it late and the owner clicks it.
    assert.equal(fx.notifier.activateNative(fx.calls[0][4]), true);
    assert.deepEqual(fx.activated, [{ chatId: 9, taskId: 't2', quizId: 'q1' }]);
    fx.notifier.destroy();
});

test('A submitted, B refused: B falls back and loses its token, A keeps its own', async () => {
    const answers = [{ ok: true, status: 'submitted', sound: 'os' }, { ok: false, status: 'failed', reason: 'balloon_not_submitted' }];
    const status = { textContent: '' };
    const fx = fixture({ answer: () => answers.shift(), documentRef: nodes({ '[data-notify-status]': [status] }) });
    fx.notifier.configure({ shell: { desktop: true, native: { status: 'authorized' } } });
    fx.notifier.handleFrame({ ...FINISHED, task_id: 'tA', chat_id: 3 }, { kind: 'chat', isMain: true });
    await tick();
    fx.notifier.handleFrame({ ...FINISHED, task_id: 'tB', chat_id: 4 }, { kind: 'chat', isMain: true });
    await tick();
    await tick();
    const [tokenA, tokenB] = fx.calls.filter(([name]) => name === 'show_native_notification').map((call) => call[4]);
    assert.deepEqual(fx.toasts, ['Task finished'], 'only B fell back');
    assert.match(status.textContent, /last alert could not be handed to the system/);
    assert.equal(fx.notifier.activateNative(tokenB), false, 'the fallback owns B now');
    assert.equal(fx.notifier.activateNative(tokenA), true);
    assert.deepEqual(fx.activated, [{ chatId: 3, taskId: 'tA', isMain: true }]);
    fx.notifier.destroy();
});

test('Test asks the system first; a live alert never asks', async () => {
    const order = [];
    const fx = fixture({
        answer: () => { order.push('show'); return { ok: true, status: 'submitted', sound: 'os' }; },
        extra: { request_native_notifications: async () => { order.push('ask'); return { status: 'authorized' }; } },
    });
    fx.notifier.handleFrame(FINISHED, { kind: 'chat', isMain: true });
    await tick();
    assert.deepEqual(order, ['show'], 'a live alert goes straight to delivery');
    assert.equal(fx.notifier.test().surface, 'native');
    await tick();
    await tick();
    assert.deepEqual(order, ['show', 'ask', 'show'], 'the Test click is the gesture that may ask');
    assert.equal(fx.calls.at(-1)[1], 'Ouroboros notifications are working');
    fx.notifier.destroy();
});

test('a current app whose shell_info failed is never called old (desktop_shell -> Settings line)', async () => {
    const status = { textContent: '' };
    const fx = fixture({ answer: { ok: true, status: 'submitted' }, documentRef: nodes({ '[data-notify-status]': [status] }) });
    const api = { shell_info: async () => { throw new Error('bridge gone'); }, show_native_notification() {} };
    fx.notifier.configure({ shell: await loadDesktopShell({ win: { pywebview: { api } }, appVersion: async () => '7.7.0' }) });
    assert.doesNotMatch(status.textContent, /no system-notification method/);
    const old = await loadDesktopShell({ win: { pywebview: { api: { request_attention() {} } } }, appVersion: async () => '6.82.0' });
    const legacy = fixture({ answer: null, extra: { show_native_notification: undefined },
        documentRef: nodes({ '[data-notify-status]': [status] }) });
    legacy.notifier.configure({ shell: old });
    assert.match(status.textContent, /This desktop app \(6\.82\.0\) has no system-notification method here/,
        'an app without the method says so (3A), and nothing more');
    fx.notifier.destroy();
    legacy.notifier.destroy();
});

/* settings.js hands each shell to the notifier: `mountDesktopShell(page, (shell) => getNotifier().configure({ shell }))`.
   A shell a failed read kept is not a new observation, so the permission the owner granted since stands. */
test('a shell kept through a failed read never undoes a permission granted meanwhile (Settings wiring)', async () => {
    let reads = 0;
    let failSlowRead;
    const status = { textContent: '' };
    const asking = {
        shell_info: () => {
            reads += 1;
            if (reads === 1) return Promise.resolve({ shell_version: '7.7.0', persistent_storage: true,
                native_notifications: { available: true, status: 'not_determined', platform: 'macos', reason: '' } });
            return new Promise((_resolve, reject) => { failSlowRead = reject; });
        },
        request_native_notifications: async () => ({ available: true, status: 'authorized', platform: 'macos', reason: '' }),
    };
    const fx = fixture({ answer: { ok: true, status: 'submitted' }, extra: asking,
        documentRef: nodes({ '[data-notify-status]': [status] }) });
    const listeners = new Map();
    const win = { pywebview: { api: asking }, addEventListener: (type, fn) => listeners.set(type, fn),
        removeEventListener: (type) => listeners.delete(type) };
    let health = 0;
    const dispose = mountDesktopShell({ querySelectorAll: () => [] }, (shell) => fx.notifier.configure({ shell }), {
        win, load: (options) => loadDesktopShell({ ...options, appVersion: async () => { health += 1; return ''; } }),
    });
    try {
        await settle();
        assert.match(status.textContent, /has not been asked yet/);
        listeners.get('pywebviewready')();  // its read is slow
        await settle();
        await fx.notifier.requestPermission();  // the owner allows notifications meanwhile
        assert.match(status.textContent, /^System notifications are on/);
        failSlowRead(new Error('the slow read failed'));
        await settle();
        assert.match(status.textContent, /^System notifications are on/, 'the kept snapshot is older than the grant');
        assert.equal(reads, 2);
        assert.equal(health, 0, 'no extra read of the server');
    } finally {
        dispose();
        fx.notifier.destroy();
    }
});

test('a new successful shell answer still reaches the notifier (Settings wiring)', async () => {
    let reads = 0;
    const status = { textContent: '' };
    const answering = {
        shell_info: async () => {
            reads += 1;
            return { shell_version: '7.7.0', persistent_storage: true, native_notifications: {
                available: true, status: reads === 1 ? 'not_determined' : 'denied', platform: 'macos', reason: '' } };
        },
    };
    const fx = fixture({ answer: { ok: true, status: 'submitted' }, extra: answering,
        documentRef: nodes({ '[data-notify-status]': [status] }) });
    const listeners = new Map();
    const win = { pywebview: { api: answering }, addEventListener: (type, fn) => listeners.set(type, fn),
        removeEventListener: (type) => listeners.delete(type) };
    const dispose = mountDesktopShell({ querySelectorAll: () => [] }, (shell) => fx.notifier.configure({ shell }), { win });
    try {
        await settle();
        assert.match(status.textContent, /has not been asked yet/);
        listeners.get('pywebviewready')();
        await settle();
        assert.match(status.textContent, /^The system denied notifications/, 'the window\'s newer answer is a new fact');
    } finally {
        dispose();
        fx.notifier.destroy();
    }
});

test('the client-level notifier answers the desktop app\'s click by token', async () => {
    resetNotifier();
    const notifier = getNotifier({ storage: storage(), documentRef: nodes(), notificationCtor: undefined, audioContextCtor: null });
    assert.equal(typeof globalThis.ouroNotifications?.activate, 'function');
    assert.equal(globalThis.ouroNotifications.activate('n1-none'), false);
    assert.equal(notifier, getNotifier());
    resetNotifier();
    assert.equal(globalThis.ouroNotifications, undefined);
});

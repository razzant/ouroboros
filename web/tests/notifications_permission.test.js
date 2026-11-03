/* macOS can accept an alert with nil error AFTER permission was revoked (native QA P9).
   Exercise the real page consumer, with recording bridges/audio only: no OS effects. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createNotifier, DEFAULT_NOTIFY_PREFS, NOTIFY_PREFS_KEY } from '../modules/notifications.js';
import { deferred, recordingAudio, settle } from './notification_audio_fixture.js';

const ALLOWED = { status: 'authorized', sound_setting: 'enabled' };
const SUBMITTED = { ok: true, status: 'submitted' };
const FRAME = { role: 'system', system_type: 'task_summary', task_id: 'result', chat_id: 7 };

function fixture({ read = ALLOWED, submitted = SUBMITTED, prefs = {}, ask = ALLOWED, cached = ALLOWED } = {}) {
    const calls = [], toasts = [], activated = [];
    const status = { textContent: '' };
    const audio = recordingAudio();
    const notifier = createNotifier({
        storage: { getItem: (key) => key === NOTIFY_PREFS_KEY
            ? JSON.stringify({ ...DEFAULT_NOTIFY_PREFS, enabled: true, ...prefs }) : null, setItem() {} },
        notificationCtor: undefined, audioContextCtor: audio.ctor,
        documentRef: { querySelectorAll: (selector) => selector === '[data-notify-status]' ? [status] : [] },
        showToast: (line) => { toasts.push(line); }, onActivate: (target) => activated.push(target),
        hostApi: {
            shell_info: async () => {
                calls.push(['read']);
                return { native_notifications: typeof read === 'function' ? await read() : read };
            },
            request_native_notifications: () => { calls.push(['ask']); return ask; },
            show_native_notification: (...args) => {
                calls.push(['send', ...args]);
                return typeof submitted === 'function' ? submitted() : submitted;
            },
        },
    });
    notifier.configure({ shell: { desktop: true, native: cached } }); // the last Settings read
    return { notifier, calls, toasts, status, audio, activated,
        alert: () => notifier.handleFrame(FRAME, { kind: 'chat', isMain: true }) };
}

test('revoked/unasked/unavailable: a fresh read prevents the nil-error submission and keeps the fallback quiet', async () => {
    for (const permission of ['denied', 'not_determined', 'unavailable']) {
        for (const gesture of [false, true]) {
            const fact = { status: permission, sound_setting: 'enabled' };
            const fx = fixture({ read: fact, ask: fact });
            if (gesture) fx.notifier.test(); else fx.alert();
            await settle();
            assert.deepEqual(fx.calls.map(([name]) => name), gesture ? ['ask', 'read'] : ['read']);
            assert.equal(fx.toasts.length, 1, `${permission}: the alert still reaches the owner`);
            assert.equal(fx.audio.contexts.length, 0, 'denial cannot license an app tone even with sound enabled');
            assert.match(fx.status.textContent, /denied|not been asked|unavailable/);
            assert.doesNotMatch(fx.status.textContent, /System notifications are on/);
            fx.notifier.destroy();
        }
    }
});

test('fresh authorized read -> direct handoff, one surface and the original click target', async () => {
    const fx = fixture();
    fx.alert();
    await settle();
    assert.deepEqual(fx.calls.map(([name]) => name), ['read', 'send']);
    assert.equal(fx.calls[1][3], true);
    assert.deepEqual(fx.toasts, []);
    assert.equal(fx.audio.contexts.length, 0);
    assert.equal(fx.notifier.activateNative(fx.calls[1][4]), true);
    assert.deepEqual(fx.activated, [{ chatId: 7, taskId: 'result', isMain: true }]);
    fx.notifier.destroy();
});

test('the final preference check follows the preparation wait (including Test)', async () => {
    for (const gesture of [false, true]) {
        for (const change of [null, ['sound', false], ['enabled', false], ['task_done', false], 'destroy']) {
            const pending = deferred();
            const fx = fixture({ read: () => pending.promise });
            if (gesture) fx.notifier.test(); else fx.alert();
            await settle();
            assert.equal(fx.calls.some(([name]) => name === 'send'), false, 'preparation has not finished');
            if (change === 'destroy') fx.notifier.destroy();
            else if (change) await fx.notifier.setPref(...change);
            pending.resolve(ALLOWED);
            await settle();
            const sends = fx.calls.filter(([name]) => name === 'send');
            const cancelled = change === 'destroy' || change?.[0] === 'enabled'
                || (!gesture && change?.[0] === 'task_done');
            assert.equal(sends.length, cancelled ? 0 : 1);
            if (sends.length) assert.equal(sends[0][3], change?.[0] !== 'sound');
            assert.deepEqual(fx.toasts, []);
            fx.notifier.destroy();
        }
    }
    const pending = deferred();
    const fx = fixture({ read: () => pending.promise, prefs: { sound: false } });
    fx.alert();
    await fx.notifier.setPref('sound', true);
    pending.resolve(ALLOWED);
    await settle();
    assert.equal(fx.calls.find(([name]) => name === 'send')[3], false, 'a silent decision stays silent');
    fx.notifier.destroy();
});

test('a later successful completion cannot promote a newer denial or repeat the accepted alert', async () => {
    const sent = deferred();
    const fx = fixture({ submitted: () => sent.promise });
    fx.alert();
    await settle();
    const send = fx.calls.find(([name]) => name === 'send');
    assert.ok(send);
    fx.notifier.configure({ shell: { desktop: true, native: { status: 'denied' } } });
    sent.resolve(SUBMITTED);
    await settle();
    assert.match(fx.status.textContent, /denied/, 'submission is not proof of permission');
    assert.deepEqual(fx.toasts, [], 'accepted is not reclassified as unsent by a later status');
    assert.equal(fx.audio.contexts.length, 0);
    assert.equal(fx.notifier.activateNative(send[4]), true);
    fx.notifier.destroy();
});

test('an unreadable preparation still permits native delivery but cannot manufacture permission', async () => {
    for (const read of [null, () => { throw new Error('no readback'); }]) {
        const fx = fixture({ read, cached: null });
        fx.alert();
        await settle();
        assert.deepEqual(fx.toasts, []);
        assert.match(fx.status.textContent, /accepted the last alert/);
        assert.doesNotMatch(fx.status.textContent, /System notifications are on/);
        assert.deepEqual(fx.calls.map(([name]) => name), ['read', 'send']);
        fx.notifier.destroy();
    }
});

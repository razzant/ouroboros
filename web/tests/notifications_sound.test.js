/* The owner's sound choices hold after the alert was decided.

   A decision is a snapshot; every effect that starts after an await — Test's permission question,
   the system's answer, the launcher's answer, the read after a failure, the audio device waking —
   asks the CURRENT choices again.
   The pre-change notifier produced three late sounds: Sound off and
   notifications off while the system's answer was pending, and Sound off while the launcher's
   answer was pending. Each is pinned here with the matching control that still sounds or shows.
   Every bridge and audio device is a recording double: nothing reaches an operating system or a
   speaker. */
import test from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_NOTIFY_PREFS, NOTIFY_PREFS_KEY, createNotifier, normalizeNotifyPrefs, readNotifyPrefs }
    from '../modules/notifications.js';
import { deferred, recordingAudio, settle, tonePeak } from './notification_audio_fixture.js';

const ON = { ...DEFAULT_NOTIFY_PREFS, enabled: true };
const FINISHED = { role: 'system', system_type: 'task_summary', task_id: 't1', chat_id: 1, content: 'Report ready' };
const CONTEXT = { kind: 'chat', isMain: true };
/* The system never took the alert (desktop_notifications.refused): the page falls back. */
const REFUSED = { ok: false, status: 'failed', platform: 'macos', reason: 'no_center' };

function store(prefs = ON) {
    const map = new Map([[NOTIFY_PREFS_KEY, JSON.stringify(prefs)]]);
    return { map, getItem: (k) => (map.has(k) ? map.get(k) : null), setItem: (k, v) => { map.set(k, String(v)); } };
}

function page(selectors = {}) {
    const handlers = new Map();
    return {
        handlers,
        addEventListener(type, fn) { handlers.set(type, fn); },
        removeEventListener(type) { handlers.delete(type); },
        querySelectorAll: (selector) => selectors[selector] || [],
        fire(type, target) { handlers.get(type)?.({ target }); },
    };
}

/* A control double: `closest` answers for its own data attribute, as a real element would. */
function control(attribute, props = {}) {
    const node = { ...props, getAttribute: (name) => (name === attribute ? props[attribute] ?? '' : null) };
    node.closest = (selector) => (selector === `[${attribute}]` ? node : null);
    return node;
}

/** A current desktop app whose system answer the test settles; this page may show browser banners. */
function desktop({ prefs = ON, permission = 'granted', native = 'pending' } = {}) {
    const calls = [];
    const toasts = [];
    const banners = [];
    const answer = deferred();
    const audio = recordingAudio();
    const hostApi = {
        show_native_notification: (...args) => {
            calls.push(['show_native_notification', ...args]);
            return native === 'pending' ? answer.promise : native;
        },
        // Model a sound-capable Windows/legacy bridge; the page must request it silently.
        request_attention: (...args) => {
            calls.push(['request_attention', ...args]);
            return args[0] ? { ok: true, status: 'native_sound', sound_played: true } : { ok: true, status: 'window_only', sound_played: false };
        },
    };
    class Banner {
        static permission = permission;
        constructor(title, options) { banners.push({ title, options }); }
    }
    const notifier = createNotifier({
        storage: store(prefs), hostApi, notificationCtor: Banner, audioContextCtor: audio.ctor, documentRef: page(),
        showToast: (line) => { toasts.push(line); return null; },
    });
    return { notifier, calls, toasts, banners, audio, answer };
}

test('Sound off while the system answer is pending: the fallback banner is silent', async () => {
    const run = async (mute) => {
        const fx = desktop();
        fx.notifier.handleFrame(FINISHED, CONTEXT);
        assert.equal(fx.calls[0][3], true, 'the system was asked for its own sound before any change');
        if (mute) await fx.notifier.setPref('sound', false);
        fx.answer.resolve(REFUSED);
        await settle();
        fx.notifier.destroy();
        assert.equal(fx.banners.length, 1, 'the alert still reaches the owner');
        assert.equal(fx.audio.contexts.length, 0, 'and the page plays nothing of its own');
        return fx.banners[0].options.silent;
    };
    assert.equal(await run(false), false, 'control: the banner keeps the system\'s sound');
    assert.equal(await run(true), true);
});

test('Sound off while the system answer is pending: the in-app fallback asks the launcher silently', async () => {
    const fx = desktop({ permission: 'default' });
    fx.notifier.handleFrame(FINISHED, CONTEXT);
    await fx.notifier.setPref('sound', false);
    fx.answer.resolve(REFUSED);
    await settle();
    assert.deepEqual(fx.toasts, ['Task finished']);
    assert.deepEqual(fx.calls.slice(1), [['request_attention', false]], 'never the launcher\'s Glass');
    assert.equal(fx.audio.contexts.length, 0);
    fx.notifier.destroy();
});

test('notifications or the category switched off while the system answer is pending: no fallback at all', async () => {
    for (const [key, value] of [[null, null], ['enabled', false], ['task_done', false]]) {
        for (const permission of ['granted', 'default']) {
            const fx = desktop({ permission });
            fx.notifier.handleFrame(FINISHED, CONTEXT);
            if (key) await fx.notifier.setPref(key, value);
            fx.answer.resolve(REFUSED);
            await settle();
            const shown = fx.banners.length + fx.toasts.length;
            if (!key) {
                assert.equal(shown, 1, `control (${permission}): the fallback reaches the owner`);
            } else {
                assert.deepEqual(fx.calls.map(([name]) => name), ['show_native_notification'], `${key} off: no attention cue`);
                assert.equal(shown, 0, `${key} off (${permission}): no banner and no in-app alert`);
            }
            assert.equal(fx.audio.contexts.length, 0);
            fx.notifier.destroy();
        }
    }
});

test('an alert decided silent never gains a sound when Sound comes back on', async () => {
    const fx = desktop({ prefs: { ...ON, sound: false } });
    fx.notifier.handleFrame(FINISHED, CONTEXT);
    assert.equal(fx.calls[0][3], false, 'the system is asked for a silent notification');
    await fx.notifier.setPref('sound', true);
    fx.answer.resolve(REFUSED);
    await settle();
    assert.equal(fx.banners.length, 1);
    assert.equal(fx.banners[0].options.silent, true, 'the fallback banner stays silent too');
    assert.equal(fx.audio.contexts.length, 0);
    fx.notifier.destroy();
});

test('the launcher\'s answer before the browser banner: notifications off meanwhile shows nothing', async () => {
    const run = async (off) => {
        const pending = deferred();
        const banners = [];
        const toasts = [];
        const notifier = createNotifier({
            storage: store(),
            notificationCtor: class { static permission = 'granted'; constructor(t) { banners.push(t); } },
            audioContextCtor: recordingAudio().ctor, documentRef: page(), showToast: (line) => { toasts.push(line); },
            hostApi: { show_native_notification: () => REFUSED, request_attention: () => ({ ok: true }),
                notify_owner: () => pending.promise },
        });
        notifier.handleFrame(FINISHED, CONTEXT);
        await settle();
        if (off) await notifier.setPref('enabled', false);
        pending.resolve({ ok: false, status: 'visible' });
        await settle();
        notifier.destroy();
        return [...banners, ...toasts];
    };
    assert.deepEqual(await run(false), ['Task finished'], 'control: the visible window gets the browser banner');
    assert.deepEqual(await run(true), []);
});

test('Test switched off while the system asks its question: nothing is delivered', async () => {
    const run = async (off) => {
        const asked = deferred();
        const sent = [];
        const notifier = createNotifier({
            storage: store({ ...ON, important: false }), notificationCtor: undefined, audioContextCtor: null, documentRef: page(),
            hostApi: { request_native_notifications: () => asked.promise,
                show_native_notification: (...args) => { sent.push(args[0]); return { ok: true, status: 'submitted' }; } },
        });
        notifier.test();
        if (off) await notifier.setPref('enabled', false);
        asked.resolve({ status: 'authorized' });
        await settle();
        notifier.destroy();
        return sent;
    };
    assert.deepEqual(await run(false), ['Ouroboros notifications are working'], 'control: Test ignores the categories');
    assert.deepEqual(await run(true), []);
});

test('Test through an app without shell_info: the hand-off asks the choices of the moment it is sent', async () => {
    // The permission question is the one await before a hand-off; the dispatch itself asks again.
    const run = async ({ prefs = { ...ON, important: false }, change = null } = {}) => {
        const asked = deferred();
        const sent = [];
        const notifier = createNotifier({
            storage: store(prefs), notificationCtor: undefined, audioContextCtor: null, documentRef: page(),
            hostApi: { request_native_notifications: () => asked.promise,
                show_native_notification: (...args) => { sent.push(args); return { ok: true, status: 'submitted' }; } },
        });
        notifier.test();
        if (change) await notifier.setPref(...change);
        asked.resolve({ status: 'authorized' });
        await settle();
        notifier.destroy();
        return sent.map((args) => [args[0], args[2]]);
    };
    const title = 'Ouroboros notifications are working';
    assert.deepEqual(await run(), [[title, true]], 'control: Test ignores the categories and keeps the system\'s sound');
    assert.deepEqual(await run({ change: ['sound', false] }), [[title, false]], 'Sound off while the system asks: sent silent');
    assert.deepEqual(await run({ prefs: { ...ON, sound: false }, change: ['sound', true] }), [[title, false]],
        'a Test decided silent never gains a sound');
    assert.deepEqual(await run({ change: ['enabled', false] }), [], 'notifications off while the system asks: nothing is sent');
});

/* The evidence the fallback tone may use: the app's own
   `shell_info().native_notifications`, read AFTER the system failed to take the alert, saying it authorizes
   Ouroboros with its sound on. Only the native macOS adapter reports `sound_setting`; absence is unknown. */
const READ_SOUND_ON = { available: true, status: 'authorized', platform: 'macos', reason: '', sound_setting: 'enabled' };

/** A current desktop app answering the hand-off and the later read as told; this page has no banner permission. */
function afterFailure({ read = READ_SOUND_ON, final = REFUSED, prefs = ON, resume } = {}) {
    const calls = [];
    const toasts = [];
    const audio = recordingAudio({ resume });
    const hostApi = {
        shell_info: () => {
            calls.push(['shell_info']);
            // Preparation is already authorized here; these tests vary the independent
            // read AFTER a definite failure. Preparation races have their own consumer tests.
            if (!calls.some(([name]) => name === 'show_native_notification')) {
                return { native_notifications: READ_SOUND_ON };
            }
            return typeof read === 'function' ? read() : { shell_version: '7.7.0', native_notifications: read };
        },
        show_native_notification: (...args) => {
            calls.push(['show_native_notification', ...args]);
            return typeof final === 'function' ? final() : final;
        },
        request_native_notifications: () => { calls.push(['request_native_notifications']); return { status: 'authorized' }; },
        request_attention: (...args) => {
            calls.push(['request_attention', ...args]);
            return args[0] ? { ok: true, status: 'native_sound', sound_played: true } : { ok: true, status: 'window_only', sound_played: false };
        },
    };
    const notifier = createNotifier({
        storage: store(prefs), hostApi, notificationCtor: class { static permission = 'default'; }, audioContextCtor: audio.ctor,
        documentRef: page(), showToast: (line) => { toasts.push(line); return null; },
    });
    return { notifier, calls, toasts, audio };
}

test('a definite failure, then a positive read after it: one soft tone; refusals and unknown get no later read', async () => {
    const run = async (final) => {
        const fx = afterFailure({ final });
        fx.notifier.handleFrame(FINISHED, CONTEXT);
        await settle();
        fx.notifier.destroy();
        return fx;
    };
    const loud = await run(REFUSED);
    assert.deepEqual(loud.calls.map(([name]) => name), ['shell_info', 'show_native_notification', 'request_attention', 'shell_info'],
        'preparation precedes dispatch; only the separate read after failure licenses a tone');
    assert.equal(loud.calls[2][1], false, 'never the launcher\'s Glass');
    assert.deepEqual(loud.toasts, ['Task finished']);
    assert.equal(loud.audio.tones.length, 1, 'control: failed, and authorized with its sound on after the failure');
    assert.equal(loud.audio.tones[0].frequency.value, 523.25);
    for (const status of ['denied', 'not_determined', 'unavailable']) {
        const refused = await run({ ok: false, status, platform: 'macos', reason: '' });
        assert.deepEqual(refused.toasts, ['Task finished'], `${status}: the alert still reaches the owner`);
        assert.deepEqual(refused.calls.map(([name]) => name), ['shell_info', 'show_native_notification', 'request_attention'],
            `${status}: quiet, with no later read or permission question to overturn the refusal`);
        assert.equal(refused.audio.contexts.length, 0);
    }
    for (const final of [{ ok: null, status: 'unknown', reason: 'no_completion' }, () => Promise.reject(new Error('gone'))]) {
        const unknown = await run(final);
        assert.deepEqual(unknown.toasts, [], 'an unknown hand-off may still appear: no fallback');
        assert.deepEqual(unknown.calls.map(([name]) => name), ['shell_info', 'show_native_notification']);
        assert.equal(unknown.audio.contexts.length, 0);
    }
});

test('only the read after the failure counts: sound switched off or permission withdrawn before it stays quiet', async () => {
    // The system answers READ_SOUND_ON until the alert is handed over and `after` from then on: the owner
    // switched the system's sound off, or withdrew the permission, between an earlier read and the failure.
    const tones = async (after) => {
        let handedOver = false;
        const fx = afterFailure({
            final: () => { handedOver = true; return REFUSED; },
            read: () => {
                if (after instanceof Error) throw after;
                return { shell_version: '7.7.0', native_notifications: handedOver ? after : READ_SOUND_ON };
            },
        });
        // A grant read earlier (the Settings mount) is no evidence for this failure either.
        fx.notifier.configure({ shell: { desktop: true, version: '7.7.0', native: READ_SOUND_ON } });
        fx.notifier.handleFrame(FINISHED, CONTEXT);
        await settle();
        fx.notifier.destroy();
        assert.deepEqual(fx.toasts, ['Task finished'], 'the read never blocks the alert');
        assert.equal(fx.calls.filter(([name]) => name === 'shell_info').length, 2);
        assert.equal(fx.calls.some(([name]) => name === 'request_native_notifications'), false);
        return fx.audio.tones.length;
    };
    assert.equal(await tones(READ_SOUND_ON), 1, 'control');
    assert.equal(await tones({ ...READ_SOUND_ON, sound_setting: 'disabled' }), 0, 'the system\'s sound switched off');
    assert.equal(await tones({ ...READ_SOUND_ON, status: 'denied' }), 0, 'the permission withdrawn');
    assert.equal(await tones({ ...READ_SOUND_ON, status: 'not_determined' }), 0);
    assert.equal(await tones({ ...READ_SOUND_ON, sound_setting: 'unknown' }), 0);
    assert.equal(await tones({ available: true, status: 'authorized', platform: 'macos', reason: '' }), 0,
        'today\'s launcher reports no sound setting');
    assert.equal(await tones({ available: false, status: 'unavailable', platform: 'macos', reason: 'not_an_app_bundle' }), 0);
    assert.equal(await tones(null), 0, 'a read with no answer');
    assert.equal(await tones(new Error('bridge gone')), 0, 'a read that fails');
});

test('a choice changed while the read after the failure is pending: no tone', async () => {
    const run = async (change) => {
        const read = deferred();
        const fx = afterFailure({ read: () => read.promise });
        fx.notifier.handleFrame(FINISHED, CONTEXT);
        await settle();
        assert.deepEqual(fx.toasts, ['Task finished'], 'the alert is shown before the read returns');
        if (change) await fx.notifier.setPref(...change);
        read.resolve({ shell_version: '7.7.0', native_notifications: READ_SOUND_ON });
        await settle();
        fx.notifier.destroy();
        return fx.audio.tones.length;
    };
    assert.equal(await run(null), 1, 'control');
    assert.equal(await run(['sound', false]), 0);
    assert.equal(await run(['enabled', false]), 0);
    assert.equal(await run(['task_done', false]), 0);
    assert.equal(await run(['tone_volume', 0]), 0);
});

test('after the failure and a positive read, a choice changed while the audio device wakes wins', async () => {
    const run = async (change) => {
        const waking = deferred();
        const fx = afterFailure({ resume: () => waking.promise });
        fx.notifier.handleFrame(FINISHED, CONTEXT);
        await settle();
        assert.equal(fx.audio.contexts.length, 1, 'the tone waits for the device');
        if (change) await fx.notifier.setPref(...change);
        waking.resolve();
        await settle();
        fx.notifier.destroy();
        return fx.audio.tones;
    };
    assert.equal((await run(null)).length, 1, 'control');
    assert.equal((await run(['sound', false])).length, 0);
    assert.equal((await run(['enabled', false])).length, 0);
    assert.equal((await run(['task_done', false])).length, 0);
    assert.equal((await run(['tone_volume', 0])).length, 0);
    assert.equal(tonePeak((await run(['tone_volume', 10]))[0]), 0.05 * 0.1 ** 2, 'a lower level applies at once');
    assert.equal(tonePeak((await run(['tone_volume', 100]))[0]), 0.05 * 0.4 ** 2, 'a higher one never makes it louder');
});

test('failed alerts waiting together for the audio device sound once, never as a louder sum', async () => {
    const waking = deferred();
    const fx = afterFailure({ resume: () => waking.promise });
    for (const id of ['a', 'b', 'c']) fx.notifier.handleFrame({ ...FINISHED, task_id: id }, CONTEXT);
    await settle();
    assert.equal(fx.toasts.length, 3, 'each alert still reaches the owner');
    waking.resolve();
    await settle();
    assert.equal(fx.audio.contexts.length, 1);
    assert.equal(fx.audio.tones.length, 1);
    fx.notifier.destroy();
});

test('Preview tone plays the soft tone: C5, a 25 ms rise, a fade, 0.05 × (level/100)² at its peak', async () => {
    const audio = recordingAudio();
    const notifier = createNotifier({ storage: store(), notificationCtor: undefined, audioContextCtor: audio.ctor, documentRef: page() });
    assert.equal(await notifier.previewTone(), true);
    assert.equal(audio.tones.length, 1);
    const [tone] = audio.tones;
    assert.equal(tone.type, 'sine');
    assert.equal(tone.frequency.value, 523.25, 'not the earlier 660 Hz beep');
    const peak = 0.05 * (40 / 100) ** 2;
    assert.deepEqual(tone.gain.automation, [['set', 0, 0], ['linear', peak, 0.025], ['exponential', 0.00001, 0.3]],
        'it rises and fades instead of the earlier flat 0.05 switched on and off');
    assert.equal(tone.startedAt, 0);
    assert.equal(tone.stoppedAt, 0.31);
    assert.equal(tone.gain.output, audio.contexts[0].destination);
    notifier.destroy();
});

test('a choice changed while the audio device wakes wins: muted, off or quieter before the tone starts', async () => {
    const run = async (change) => {
        const waking = deferred();
        const audio = recordingAudio({ resume: () => waking.promise });
        const notifier = createNotifier({ storage: store(), notificationCtor: undefined, audioContextCtor: audio.ctor, documentRef: page() });
        const pressed = notifier.previewTone();
        assert.equal(audio.contexts.length, 1, 'the tone waits for the device');
        if (change) await notifier.setPref(...change);
        waking.resolve();
        await pressed;
        notifier.destroy();
        return audio.tones;
    };
    assert.equal((await run(null)).length, 1, 'control');
    assert.equal((await run(['sound', false])).length, 0);
    assert.equal((await run(['enabled', false])).length, 0);
    assert.equal((await run(['tone_volume', 0])).length, 0);
    assert.equal(tonePeak((await run(['tone_volume', 10]))[0]), 0.05 * 0.1 ** 2, 'a lower level applies at once');
    assert.equal(tonePeak((await run(['tone_volume', 100]))[0]), 0.05 * 0.4 ** 2, 'a higher one never makes it louder');
});

test('presses waiting together for the audio device sound once, never as a louder sum', async () => {
    const waking = deferred();
    const audio = recordingAudio({ resume: () => waking.promise });
    const notifier = createNotifier({ storage: store(), notificationCtor: undefined, audioContextCtor: audio.ctor, documentRef: page() });
    const presses = [notifier.previewTone(), notifier.previewTone(), notifier.previewTone()];
    waking.resolve();
    assert.deepEqual(await Promise.all(presses), [false, false, true], 'the newest press plays');
    assert.equal(audio.contexts.length, 1);
    assert.equal(audio.tones.length, 1);
    assert.equal(await notifier.previewTone(), false, 'nor over a tone still sounding');
    audio.contexts[0].currentTime = 0.31;
    assert.equal(await notifier.previewTone(), true, 'once it has ended, the next press plays');
    notifier.destroy();
});

test('the App tone volume is a client-local whole number from 0 to 100 that survives a new client', async () => {
    assert.equal(DEFAULT_NOTIFY_PREFS.tone_volume, 40);
    assert.equal(normalizeNotifyPrefs({}).tone_volume, 40);
    assert.equal(normalizeNotifyPrefs({ tone_volume: 72.6 }).tone_volume, 73);
    assert.equal(normalizeNotifyPrefs({ tone_volume: 250 }).tone_volume, 100);
    assert.equal(normalizeNotifyPrefs({ tone_volume: -3 }).tone_volume, 0);
    for (const malformed of ['55', null, true, Number.NaN, Number.POSITIVE_INFINITY]) {
        assert.equal(normalizeNotifyPrefs({ tone_volume: malformed }).tone_volume, 40, String(malformed));
    }
    // A client that stored its choices before the level existed keeps every choice and gets the default level.
    const older = { enabled: true, needs_answer: false, task_done: true, important: true, main_reply: true, sound: false, show_text: true };
    assert.deepEqual(readNotifyPrefs(store(older)), { ...older, tone_volume: 40 });

    const storage = store();
    const first = createNotifier({ storage, notificationCtor: undefined, audioContextCtor: null, documentRef: page() });
    await first.setPref('tone_volume', '25');
    assert.equal(JSON.parse(storage.map.get(NOTIFY_PREFS_KEY)).tone_volume, 25);
    await first.setPref('tone_volume', '');
    await first.setPref('tone_volume', 'loud');
    assert.equal(first.prefs.tone_volume, 25, 'a value the control cannot express keeps the current level');
    assert.equal(first.prefs.sound, true, 'the level is not the Sound switch');
    first.destroy();
    const second = createNotifier({ storage, notificationCtor: undefined, audioContextCtor: null, documentRef: page() });
    assert.equal(second.prefs.tone_volume, 25);
    second.destroy();
});

test('moving the slider stores and paints the level and plays nothing; only Preview tone does', async () => {
    const slider = control('data-notify-volume', { value: '40', disabled: false });
    const value = { textContent: '' };
    const preview = control('data-notify-preview', { disabled: false });
    const sound = control('data-notify-pref', { 'data-notify-pref': 'sound', checked: true, disabled: false });
    const doc = page({
        '[data-notify-volume]': [slider], '[data-notify-volume-value]': [value],
        '[data-notify-preview]': [preview], '[data-notify-pref]': [sound],
    });
    const storage = store();
    const audio = recordingAudio();
    const notifier = createNotifier({ storage, notificationCtor: undefined, audioContextCtor: audio.ctor, documentRef: doc });
    notifier.mountSettings(doc);
    assert.equal(value.textContent, '40%');
    assert.equal(preview.disabled, false);

    for (const level of ['10', '65', '0', '30']) {
        slider.value = level;
        doc.fire('input', slider);
    }
    doc.fire('change', slider);
    await settle();
    assert.equal(audio.contexts.length, 0, 'the slider never opens an audio device');
    assert.equal(JSON.parse(storage.map.get(NOTIFY_PREFS_KEY)).tone_volume, 30);
    assert.equal(value.textContent, '30%');

    doc.fire('click', preview);
    await settle();
    assert.equal(audio.tones.length, 1, 'the explicit press plays one tone');
    assert.equal(tonePeak(audio.tones[0]), 0.05 * 0.3 ** 2);

    sound.checked = false;
    doc.fire('change', sound);
    await settle();
    assert.equal(slider.disabled, true, 'the level is inert while Sound is off');
    assert.equal(preview.disabled, true);
    assert.equal(await notifier.previewTone(), false, 'and a press that reaches it anyway plays nothing');

    sound.checked = true;
    doc.fire('change', sound);
    slider.value = '0';
    doc.fire('input', slider);
    await settle();
    assert.equal(preview.disabled, true, 'nothing to preview at level 0');
    assert.equal(audio.tones.length, 1);
    notifier.destroy();
    assert.equal(doc.handlers.size, 0, 'destroy releases the input listener too');
});

test('Preview tone ignores the categories and the system\'s permission, never notifications or Sound', async () => {
    const quiet = { ...ON, needs_answer: false, task_done: false, important: false, main_reply: false };
    const run = async (prefs, notificationCtor = class { static permission = 'denied'; }) => {
        const audio = recordingAudio();
        const notifier = createNotifier({ storage: store(prefs), notificationCtor, audioContextCtor: audio.ctor, documentRef: page() });
        const played = await notifier.previewTone();
        notifier.destroy();
        return [played, audio.tones.length];
    };
    assert.deepEqual(await run(quiet), [true, 1], 'every category off and a denial: the owner still hears the preview');
    assert.deepEqual(await run(quiet, undefined), [true, 1]);
    assert.deepEqual(await run({ ...quiet, enabled: false }), [false, 0]);
    assert.deepEqual(await run({ ...quiet, sound: false }), [false, 0]);
    assert.deepEqual(await run({ ...quiet, tone_volume: 0 }), [false, 0]);
});

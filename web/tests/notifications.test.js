import test from 'node:test';
import assert from 'node:assert/strict';

import {
    DEFAULT_NOTIFY_PREFS,
    NOTIFY_PREFS_KEY,
    classifyLiveFrame,
    createNotificationActivation,
    createNotifier,
    decideNotification,
    attentionStatusText,
    noticeTitle,
    normalizeNotifyPrefs,
    notifyStatusText,
    readNotifyPrefs,
    writeNotifyPrefs,
} from '../modules/notifications.js';
import { recordingAudio, settle } from './notification_audio_fixture.js';

const ON = { ...DEFAULT_NOTIFY_PREFS, enabled: true };

test('activation resolves an addressed Project independently of the sidebar and preserves question identity', async () => {
    const calls = [];
    const project = { id: 'older-than-sidebar', chat_id: -77, lifecycle: 'active' };
    const activate = createNotificationActivation({
        resolveProject: async chatId => { calls.push(['resolve', chatId]); return project; },
        openProject: (...args) => calls.push(['project', ...args]),
        openMain: () => calls.push(['main']),
        unavailable: () => calls.push(['unavailable']),
    });
    await activate({ chatId: -77, taskId: 'author', quizId: 'question' });
    assert.deepEqual(calls, [['resolve', -77], ['project', project,
        { openOnly: true, taskId: 'author', quizId: 'question' }]]);
    calls.length = 0;
    await activate({ chatId: 1 });
    assert.deepEqual(calls, [['main']], 'only an explicitly Main-addressed notification opens Main');
});

test('unreadable, missing, inactive and mismatched notification destinations say unavailable without navigation', async () => {
    for (const resolveProject of [async () => { throw new Error('503'); }, async () => null,
        async () => ({ id: 'p', chat_id: -77, lifecycle: 'tombstoned' }),
        async () => ({ id: 'p', chat_id: -78, lifecycle: 'active' })]) {
        const calls = [];
        const activate = createNotificationActivation({ resolveProject,
            openProject: () => calls.push('project'), openMain: () => calls.push('main'),
            unavailable: () => calls.push('unavailable') });
        await activate({ chatId: -77 });
        assert.deepEqual(calls, ['unavailable']);
    }
});

test('a later Main activation supersedes an older unresolved Project notification', async () => {
    let complete;
    const calls = [];
    const activate = createNotificationActivation({
        resolveProject: () => new Promise(resolve => { complete = resolve; }),
        openProject: () => calls.push('project'), openMain: () => calls.push('main'),
        unavailable: () => calls.push('unavailable'),
    });
    const earlier = activate({ chatId: -77 });
    await activate({ chatId: 1 });
    complete({ id: 'p', chat_id: -77, lifecycle: 'active' });
    await earlier;
    assert.deepEqual(calls, ['main']);
});

function fakeStorage(initial = {}) {
    const map = new Map(Object.entries(initial));
    return {
        map,
        getItem: (k) => (map.has(k) ? map.get(k) : null),
        setItem: (k, v) => { map.set(k, String(v)); },
    };
}

/* A minimal document double: the notifier only adds/removes listeners and
   walks [data-notify-*] nodes, so this is enough to exercise the shell. */
function fakeDocument(nodes = {}) {
    return {
        handlers: new Map(),
        addEventListener(type, fn) { this.handlers.set(type, fn); },
        removeEventListener(type) { this.handlers.delete(type); },
        querySelectorAll(selector) { return nodes[selector] || []; },
    };
}

test('shipped defaults keep the quiet, private starting point', () => {
    assert.equal(DEFAULT_NOTIFY_PREFS.enabled, false);
    assert.equal(DEFAULT_NOTIFY_PREFS.main_reply, false, 'ordinary Main replies start off');
    assert.equal(DEFAULT_NOTIFY_PREFS.show_text, false, 'message text starts hidden');
    assert.equal(DEFAULT_NOTIFY_PREFS.needs_answer, true);
    assert.equal(DEFAULT_NOTIFY_PREFS.task_done, true);
});

test('preferences are read tolerantly and per key', () => {
    assert.deepEqual(normalizeNotifyPrefs(null), { ...DEFAULT_NOTIFY_PREFS });
    // An unknown key is dropped; a malformed value keeps only ITS default.
    assert.deepEqual(
        normalizeNotifyPrefs({ enabled: true, sound: 'yes', nope: 1 }),
        { ...DEFAULT_NOTIFY_PREFS, enabled: true },
    );
    const storage = fakeStorage({ [NOTIFY_PREFS_KEY]: '{not json' });
    assert.deepEqual(readNotifyPrefs(storage), { ...DEFAULT_NOTIFY_PREFS });
    const broken = { getItem() { throw new Error('blocked'); } };
    assert.deepEqual(readNotifyPrefs(broken), { ...DEFAULT_NOTIFY_PREFS });
    assert.equal(writeNotifyPrefs({ setItem() { throw new Error('blocked'); } }, ON), false);
    const ok = fakeStorage();
    assert.equal(writeNotifyPrefs(ok, { ...ON, nope: true }), true);
    assert.deepEqual(JSON.parse(ok.map.get(NOTIFY_PREFS_KEY)), { ...ON });
});

test('a waiting question is required, an optional one is model-chosen', () => {
    const waiting = classifyLiveFrame(
        { task_id: 't1', quiz: { quiz_id: 'q1', state: 'open', wait_for_answer: true, question: 'Merge it?' } },
        { kind: 'quiz' },
    );
    assert.equal(waiting.category, 'needs_answer');
    assert.equal(waiting.key, 'quiz:t1:q1');
    assert.equal(waiting.body, 'Merge it?');

    // Flat shape (the frame carries quiz fields directly) is the same fact.
    const flat = classifyLiveFrame(
        { task_id: 't1', quiz_id: 'q2', wait_for_answer: true }, { kind: 'quiz' },
    );
    assert.equal(flat.category, 'needs_answer');

    const optional = classifyLiveFrame(
        { task_id: 't1', quiz: { quiz_id: 'q3', state: 'open' } }, { kind: 'quiz' },
    );
    assert.equal(optional.category, 'important', 'no positive wait fact ⇒ optional ask');
});

test('an answered or closed question never rings', () => {
    for (const state of ['answered', 'closed', 'expired_terminal', 'superseded']) {
        assert.equal(classifyLiveFrame(
            { task_id: 't1', quiz: { quiz_id: 'q', state, wait_for_answer: true } }, { kind: 'quiz' },
        ), null, state);
    }
    assert.equal(classifyLiveFrame({ task_id: 't1', quiz: {} }, { kind: 'quiz' }), null);
});

test('importance rides the existing proactive discriminator', () => {
    const candidate = classifyLiveFrame({
        role: 'assistant', system_type: 'proactive_message', task_id: 't2',
        client_message_id: 'cm-9', content: 'The provider is down; I paused.',
    }, { kind: 'chat', isMain: true });
    assert.equal(candidate.category, 'important');
    assert.equal(candidate.key, 'important:cm-9');
});

test('a Project root notice addressed to Main is important and opens Main', () => {
    const notice = { role: 'assistant', system_type: 'main_notice', task_id: 'project-root', chat_id: 1,
        client_message_id: 'cm-main', content: 'The Cursor account has no included usage left.' };
    const candidate = classifyLiveFrame(notice, { kind: 'chat', isMain: true });
    assert.equal(candidate.category, 'important');
    assert.equal(candidate.target.chatId, 1);
    // The same frame from a known child never rings.
    assert.equal(classifyLiveFrame(notice, { kind: 'chat', isMain: true, isRoot: false }), null);
});

test('a managed root concludes as a log frame, and both shapes share one key', () => {
    const log = classifyLiveFrame(
        { type: 'task_done', status: 'completed', chat_id: 1 },
        { kind: 'log', isMain: true, isRoot: true, taskId: 't-log' },
    );
    assert.ok(log, 'a settled terminal is an ending');
    assert.equal(log.category, 'task_done');
    assert.equal(log.key, 'conclusion:t-log', 'the same key a chat-row terminal would produce');
    assert.equal(log.body, '', 'a log frame carries diagnostics, never owner prose');

    // Other log traffic is not an event.
    assert.equal(classifyLiveFrame({ type: 'task_metrics_event', status: 'completed' },
        { kind: 'log', isRoot: true, taskId: 't-log' }), null);
    // Without a resolved task identity there is nothing to key or open.
    assert.equal(classifyLiveFrame({ type: 'task_done', status: 'completed' },
        { kind: 'log', isRoot: true }), null);
    // A child's conclusion has the same wire shape; only lineage separates them.
    assert.equal(classifyLiveFrame({ type: 'task_done', status: 'completed' },
        { kind: 'log', isRoot: false, taskId: 't-child' }), null);
    // An update/restart teardown pushes a NON-settled terminal and requeues the
    // same id: it must neither ring nor consume the task's key.
    for (const status of ['interrupted', '', 'running']) {
        assert.equal(classifyLiveFrame({ type: 'task_done', status },
            { kind: 'log', isRoot: true, taskId: 't-log' }), null, status);
    }

    // One event, one delivery, whichever shape arrives first.
    const seen = new Set();
    const first = decideNotification(log, ON, seen);
    assert.equal(first.deliver, true);
    seen.add(first.key);
    const viaChatRow = classifyLiveFrame(
        { role: 'system', system_type: 'task_summary', task_id: 't-log' },
        { kind: 'chat', isMain: true, isRoot: true },
    );
    assert.equal(decideNotification(viaChatRow, ON, seen).reason, 'duplicate');
});

test('only a ROOT terminal notifies; a child escalates instead', () => {
    const root = classifyLiveFrame(
        { role: 'system', system_type: 'task_summary', task_id: 't3', content: 'Done.' },
        { kind: 'chat', isMain: true },
    );
    assert.equal(root.category, 'task_done');
    assert.equal(root.key, 'conclusion:t3');

    assert.equal(classifyLiveFrame({
        role: 'system', system_type: 'task_summary', task_id: 't4', delegation_role: 'subagent',
    }, { kind: 'chat', isMain: true }), null, 'a subagent terminal is not an owner event');

    assert.equal(classifyLiveFrame({
        role: 'assistant', task_id: 't5', delegation_role: 'subagent',
        subagent_event: 'completed',
    }, { kind: 'chat', isMain: true }), null);

    // A bare `task_terminal_status` row is set only for a DIRECT turn, so it is
    // that conversation's ending — never "a task finished".
    const stopped = classifyLiveFrame(
        { role: 'assistant', task_id: 't-direct', task_terminal_status: 'failed',
          content: 'Stopped.' },
        { kind: 'chat', isMain: true },
    );
    assert.equal(stopped.category, 'main_reply');
    assert.equal(stopped.title, 'Ouroboros replied');

    // A chat-row terminal whose lineage says child is equally not an owner event.
    assert.equal(classifyLiveFrame(
        { role: 'system', system_type: 'task_summary', task_id: 't6' },
        { kind: 'chat', isMain: true, isRoot: false },
    ), null);
});

test('one turn ending is one notification, not a reply plus a terminal', () => {
    const seen = new Set();
    const reply = classifyLiveFrame(
        { role: 'assistant', task_id: 't-turn', content: 'Here it is.' },
        { kind: 'chat', isMain: true },
    );
    assert.equal(reply.category, 'main_reply');
    assert.equal(reply.key, 'conclusion:t-turn');
    seen.add(decideNotification(reply, { ...ON, main_reply: true }, seen).key);
    const terminal = classifyLiveFrame(
        { type: 'task_done', status: 'completed' },
        { kind: 'log', isRoot: true, taskId: 't-turn' },
    );
    assert.equal(decideNotification(terminal, { ...ON, main_reply: true }, seen).reason, 'duplicate');
});

test('a Main turn left to run in the background rings as a finished task; an inline turn still does not', () => {
    const ending = (owner) => classifyLiveFrame(
        { type: 'task_done', status: 'completed', chat_id: 1, _is_direct_chat: true, ...(owner ? { execution_owner: owner } : {}) },
        { kind: 'log', isMain: true, isRoot: true, taskId: 't-bg' },
    );
    const pooled = ending({ kind: 'pooled' });
    assert.equal(pooled?.category, 'task_done', 'the pool ran its last attempt: the owner hears that the work ended');
    assert.equal(pooled.key, 'conclusion:t-bg', 'the same key its final answer row would use, so one banner');
    assert.equal(ending({ kind: 'direct' }), null, 'an inline conversation turn ends as a reply, not a task');
    assert.equal(ending(null), null, 'without a carrier fact the historical origin still decides');
});

test('ordinary replies are Main-only, and progress/user frames are never events', () => {
    const main = classifyLiveFrame(
        { role: 'assistant', content: 'Here it is.', client_message_id: 'cm-1' },
        { kind: 'chat', isMain: true },
    );
    assert.equal(main.category, 'main_reply');
    assert.equal(main.key, 'main_reply:cm-1', 'a turn without a task id keys on its row');
    assert.equal(classifyLiveFrame(
        { role: 'assistant', content: 'Here it is.' }, { kind: 'chat', isMain: false },
    ), null, 'a Project room reply is not the Main-reply category');
    assert.equal(classifyLiveFrame(
        { role: 'assistant', is_progress: true, content: 'thinking' }, { kind: 'chat', isMain: true },
    ), null);
    assert.equal(classifyLiveFrame(
        { role: 'user', content: 'hi' }, { kind: 'chat', isMain: true },
    ), null);
    assert.equal(classifyLiveFrame({ role: 'assistant' }, { kind: 'log', isMain: true }), null);
});

test('the gate answers disabled, category-off, duplicate and text policy', () => {
    const candidate = classifyLiveFrame(
        { task_id: 't', quiz: { quiz_id: 'q', state: 'open', wait_for_answer: true, question: 'Which one?' } },
        { kind: 'quiz' },
    );
    assert.deepEqual(decideNotification(candidate, DEFAULT_NOTIFY_PREFS).deliver, false);
    assert.equal(decideNotification(candidate, DEFAULT_NOTIFY_PREFS).reason, 'disabled');
    assert.equal(decideNotification(candidate, { ...ON, needs_answer: false }).reason, 'category_off');
    assert.equal(decideNotification(null, ON).reason, 'not_notifiable');
    assert.equal(decideNotification({ ...candidate, category: 'weird' }, ON).reason, 'unknown_category');

    const hidden = decideNotification(candidate, ON, new Set());
    assert.equal(hidden.deliver, true);
    assert.equal(hidden.body, '', 'text stays hidden unless show_text is on');
    assert.equal(decideNotification(candidate, { ...ON, show_text: true }, new Set()).body, 'Which one?');
    assert.equal(decideNotification(candidate, ON, new Set([candidate.key])).reason, 'duplicate');
});

test('the status line states what this client can actually do', () => {
    assert.match(notifyStatusText({ enabled: false }), /off/);
    assert.match(notifyStatusText({ enabled: true, supported: false }), /no system notifications/);
    assert.match(notifyStatusText({ enabled: true, supported: true, permission: 'denied' }), /denied/);
    assert.match(notifyStatusText({ enabled: true, supported: true, permission: 'default' }), /not been granted/);
    assert.match(notifyStatusText({ enabled: true, supported: true, permission: 'granted' }), /enabled/);
    assert.match(notifyStatusText({ storageAvailable: false }), /blocks storage/);
});

test('attention status distinguishes native desktop attention from browser support', () => {
    assert.match(attentionStatusText({ nativeAttention: true }), /only visual attention/);
    assert.match(attentionStatusText({ status: 'native_sound' }), /reported a sound/);
    assert.match(attentionStatusText({ supported: true }), /Browser notifications/);
    assert.match(attentionStatusText({ bridge: true }), /will be confirmed/);
    assert.match(attentionStatusText(), /no native attention bridge/);
});

function notifierFixture({ permission = 'granted', prefs = { ...ON, task_done: true } } = {}) {
    const built = [];
    class FakeNotification {
        static permission = permission;
        static async requestPermission() { return permission; }
        constructor(title, options) {
            this.title = title;
            this.options = options;
            built.push(this);
        }
        close() { this.closed = true; }
    }
    const toasts = [];
    const activated = [];
    let focused = 0;
    const documentRef = fakeDocument();
    const notifier = createNotifier({
        storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify(prefs) }),
        notificationCtor: FakeNotification,
        audioContextCtor: null,
        showToast: (line) => toasts.push(line),
        onActivate: (target) => activated.push(target),
        focusWindow: () => { focused += 1; },
        documentRef,
    });
    return { notifier, built, toasts, activated, documentRef, focusedCount: () => focused };
}

test('a granted client gets one banner per logical event, tagged for collapse', () => {
    const fx = notifierFixture();
    const frame = { role: 'system', system_type: 'task_summary', task_id: 't9', content: 'Finished.' };
    const first = fx.notifier.handleFrame(frame, { kind: 'chat', isMain: true });
    assert.equal(first.surface, 'banner');
    assert.equal(fx.built.length, 1);
    assert.equal(fx.built[0].title, 'Task finished');
    assert.equal(fx.built[0].options.tag, 'conclusion:t9');
    assert.equal(fx.built[0].options.body, undefined, 'hidden text sends no body');
    assert.equal(fx.built[0].options.silent, false, 'sound on ⇒ the OS may play its own');

    // The same event delivered again (Main and its Project room, a repeated
    // targeted refresh) must not ring twice.
    assert.equal(fx.notifier.handleFrame(frame, { kind: 'chat', isMain: true }), null);
    assert.equal(fx.built.length, 1);
    assert.equal(fx.toasts.length, 0);
});

/* A current app's system notification that did not reach the system (desktop_notifications.refused). */
const REFUSED = { ok: false, status: 'failed', platform: 'windows', reason: 'balloon_not_submitted' };
/* A refusal carrying a sound setting is no evidence by itself: after a definite failure the fallback tone
   reads only the app's `shell_info()`, asked after that failure (notifications_sound.test.js), which these
   launchers do not offer. */
const SOUND_ON_REFUSED = { ok: false, status: 'failed', platform: 'macos', reason: 'no_center', sound_setting: 'enabled' };
const FINISHED_ROW = { role: 'system', system_type: 'task_summary', task_id: 'native-1' };

test('a current app whose system notification fell back asks its attention cue once, without its sound', async () => {
    const calls = [];
    const notifier = createNotifier({
        storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify(ON) }),
        notificationCtor: undefined,
        audioContextCtor: null,
        showToast: () => {},
        documentRef: fakeDocument(),
        hostApi: {
            show_native_notification: () => REFUSED,
            request_attention: (...args) => { calls.push(args); return { ok: true, status: 'window_only' }; },
        },
    });
    assert.equal(notifier.handleFrame(FINISHED_ROW, { kind: 'chat', isMain: true }).surface, 'native');
    await settle();
    assert.deepEqual(calls, [[false]]);
    notifier.destroy();
});

test('an app whose notify_owner predates system notifications is never asked, Sound on or off: the browser banner or the toast', async () => {
    for (const sound of [true, false]) {
        for (const [permission, surface] of [['granted', 'banner'], [undefined, 'in_app']]) {
            const calls = [];
            const banners = [];
            const toasts = [];
            const attention = { textContent: '' };
            const audio = recordingAudio();
            const notifier = createNotifier({
                storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify({ ...ON, sound }) }),
                notificationCtor: permission
                    ? class { static permission = permission; constructor(title, options) { banners.push(options); } }
                    : undefined,
                audioContextCtor: audio.ctor,
                showToast: (line) => { toasts.push(line); },
                documentRef: fakeDocument({ '[data-notify-attention-status]': [attention] }),
                hostApi: {
                    request_attention: (...args) => { calls.push(['request_attention', ...args]); return { ok: true }; },
                    notify_owner: (...args) => { calls.push(['notify_owner', ...args]); return { ok: true, status: 'background', banner: true }; },
                },
            });
            assert.equal(notifier.handleFrame(FINISHED_ROW, { kind: 'chat', isMain: true }).surface, surface);
            await settle();
            assert.deepEqual(calls, [], 'an older launcher may sound whatever it is asked: it is not called');
            assert.equal(audio.tones.length, 0, 'no system evidence that sound is allowed: no page tone either');
            if (permission) assert.equal(banners[0].silent, !sound, 'the browser banner keeps the Sound choice');
            else assert.deepEqual(toasts, ['Task finished']);
            notifier.mountSettings();
            assert.equal(attention.textContent, '', 'no attention cue to describe');
            notifier.destroy();
        }
    }
});

test('the oldest app, request_attention without notify_owner, gets only its silent window cue, Sound on or off', async () => {
    for (const sound of [true, false]) {
        const calls = [];
        const toasts = [];
        const attention = { textContent: '' };
        const audio = recordingAudio();
        const notifier = createNotifier({
            storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify({ ...ON, sound }) }),
            notificationCtor: undefined,
            audioContextCtor: audio.ctor,
            showToast: (line) => { toasts.push(line); },
            documentRef: fakeDocument({ '[data-notify-attention-status]': [attention] }),
            // Its `false` only shows the window, on every platform (platform_layer.request_native_attention).
            hostApi: { request_attention: (...args) => { calls.push(args); return { ok: true, status: 'window_only', sound_played: false }; } },
        });
        assert.equal(notifier.handleFrame(FINISHED_ROW, { kind: 'chat', isMain: true }).surface, 'in_app');
        await settle();
        assert.deepEqual(calls, [[false]], 'never its own sound');
        assert.equal(audio.contexts.length + audio.tones.length, 0, 'it reports no system permission: no page tone');
        assert.deepEqual(toasts, ['Task finished']);
        assert.match(attention.textContent, /raise this window without a sound of its own/);
        notifier.destroy();
    }
});

test('with banner permission, a current app that can hide its window answers before any browser banner', async () => {
    const calls = [];
    const banners = [];
    const toasts = [];
    let oscillators = 0;
    class GrantedNotification {
        static permission = 'granted';
        constructor(title, options) { banners.push({ title, options }); }
        close() {}
    }
    class FakeAudioContext {
        constructor() { this.currentTime = 0; this.destination = {}; }
        resume() {}
        createOscillator() { oscillators += 1; return { connect() {}, start() {}, stop() {} }; }
        createGain() { return { gain: { value: 0 }, connect() {} }; }
    }
    const make = (notifyOwner) => createNotifier({
        storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify(ON) }),
        notificationCtor: GrantedNotification,
        audioContextCtor: FakeAudioContext,
        showToast: (line) => { toasts.push(line); },
        documentRef: fakeDocument(),
        hostApi: {
            show_native_notification: () => REFUSED,
            request_attention: (...args) => { calls.push(['request_attention', ...args]); return { ok: true }; },
            notify_owner: (...args) => { calls.push(['notify_owner', ...args]); return notifyOwner(); },
        },
    });
    const deliver = async (notifier, id) => {
        const out = notifier.handleFrame({ role: 'system', system_type: 'task_summary', task_id: id }, { kind: 'chat', isMain: true });
        await settle();
        notifier.destroy();
        return out.surface;
    };
    // Never the launcher's own sound (macOS Glass, a Windows beep): it is asked silently.
    const asked = [['notify_owner', false, 'Task finished', '', false]];

    // Hidden on purpose: the launcher's own banner (it opens the window) and no browser banner,
    // whose focus() cannot undo a hide; the queued banner owns the sound, so no page tone.
    assert.equal(await deliver(make(() => ({ ok: true, status: 'background', banner: true, sound_played: false })), 'h'), 'native');
    assert.deepEqual(calls, asked);
    assert.equal(banners.length, 0);
    assert.equal(oscillators, 0);
    assert.deepEqual(toasts, ['Task finished'], 'the alert waits in the window too');

    // Visible: the launcher does nothing (no raise, no system sound); the browser banner owns the sound.
    calls.length = 0;
    toasts.length = 0;
    assert.equal(await deliver(make(() => ({ ok: false, status: 'visible' })), 'v'), 'native');
    assert.deepEqual(calls, asked, 'never the raise-and-sound cue for a visible window');
    assert.equal(banners.length, 1);
    assert.equal(banners[0].options.silent, false);
    assert.equal(oscillators, 0, 'one sound: the banner\'s');
    assert.equal(toasts.length, 0);

    // A launcher that cannot answer: the browser banner, as before.
    calls.length = 0;
    assert.equal(await deliver(make(() => Promise.reject(new Error('bridge gone'))), 'r'), 'native');
    assert.equal(banners.length, 2);
});

test('a current app takes the alert text, asked without its sound; the oldest one gets no text', async () => {
    const calls = [];
    const host = {
        show_native_notification: () => REFUSED,
        request_attention: (...args) => { calls.push(['request_attention', ...args]); return { ok: true }; },
        notify_owner: (...args) => { calls.push(['notify_owner', ...args]); return { ok: true, status: 'background' }; },
    };
    const make = (hostApi, prefs = ON) => createNotifier({
        storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify(prefs) }),
        notificationCtor: undefined,
        audioContextCtor: null,
        showToast: () => {},
        documentRef: fakeDocument(),
        hostApi,
    });
    const frame = (id) => ({ role: 'system', system_type: 'task_summary', task_id: id, content: 'Report ready' });
    const newer = make(host, { ...ON, show_text: true });
    newer.handleFrame(frame('text-1'), { kind: 'chat', isMain: true });
    await settle();
    assert.equal(calls.length, 1);
    assert.equal(calls[0][0], 'notify_owner');
    assert.equal(calls[0][1], false, 'never the launcher\'s own sound (macOS Glass, a Windows beep)');
    assert.equal(typeof calls[0][2], 'string');
    assert.equal(calls[0][3], 'Report ready');
    newer.destroy();

    calls.length = 0;
    const private_ = make(host);
    private_.handleFrame(frame('text-2'), { kind: 'chat', isMain: true });
    await settle();
    assert.equal(calls[0][3], '', 'message text stays private unless the owner turned it on');
    private_.destroy();

    calls.length = 0;
    const older = make({ request_attention: host.request_attention }, { ...ON, show_text: true });
    older.handleFrame(frame('text-3'), { kind: 'chat', isMain: true });
    await settle();
    assert.deepEqual(calls, [['request_attention', false]], 'its silent window cue only, never the text');
    older.destroy();

    calls.length = 0;
    const alias = make({ request_attention: host.request_attention, notify_owner: host.notify_owner });
    alias.handleFrame(frame('text-4'), { kind: 'chat', isMain: true });
    await settle();
    assert.deepEqual(calls, [], 'its notify_owner may show a balloon Windows sounds: never asked');
    alias.destroy();
    assert.match(attentionStatusText({ status: 'background' }), /instead of opening this window/);
});

test('a browser\'s permission is no evidence that sound is allowed: its in-app fallback stays quiet', async () => {
    const tonesWith = async (notificationCtor) => {
        const audio = recordingAudio();
        const toasts = [];
        const notifier = createNotifier({
            storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify(ON) }),
            notificationCtor,
            audioContextCtor: audio.ctor,
            showToast: (line) => { toasts.push(line); },
            documentRef: fakeDocument(),
        });
        notifier.handleFrame({ role: 'system', system_type: 'task_summary', task_id: 'browser' }, { kind: 'chat', isMain: true });
        await settle();
        notifier.destroy();
        assert.deepEqual(toasts, ['Task finished'], 'the alert still reaches the owner');
        return audio.tones.length;
    };
    assert.equal(await tonesWith(class { static permission = 'default'; }), 0, 'not yet asked is not a grant');
    assert.equal(await tonesWith(class { static permission = 'denied'; }), 0);
    assert.equal(await tonesWith(undefined), 0, 'nothing to read');
    assert.equal(await tonesWith(class { static permission = 'granted'; constructor() { throw new Error('no banner here'); } }), 0,
        'a granted banner that could not be shown says nothing of sound');
});

test('whatever the launcher answers, the page adds no tone of its own', async () => {
    const tonesFor = async (answer) => {
        const audio = recordingAudio();
        const notifier = createNotifier({
            storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify(ON) }),
            notificationCtor: undefined,
            audioContextCtor: audio.ctor,
            hostApi: { show_native_notification: () => SOUND_ON_REFUSED, request_attention: () => answer, notify_owner: () => answer },
            showToast: () => {},
            documentRef: fakeDocument(),
        });
        notifier.handleFrame({ role: 'system', system_type: 'task_summary', task_id: 'b' }, { kind: 'chat', isMain: true });
        await settle();
        notifier.destroy();
        return audio.contexts.length + audio.tones.length;
    };
    // Windows, window hidden: the balloon is queued and the OS plays its sound.
    assert.equal(await tonesFor({ ok: true, status: 'background', banner: true, sound_played: false }), 0);
    // macOS, window hidden: a Dock badge; an older launcher's own sound, or none at all.
    assert.equal(await tonesFor({ ok: true, status: 'background', banner: false, sound_played: true }), 0);
    assert.equal(await tonesFor({ ok: true, status: 'background', banner: false, sound_played: false }), 0,
        'no evidence the system lets Ouroboros sound: the earlier page beep is gone');
    assert.equal(await tonesFor({ ok: true, status: 'window_only', sound_played: false }), 0);
    assert.equal(await tonesFor(Promise.reject(new Error('bridge gone'))), 0);
});

test('a banner click focuses the window and hands the target to navigation', () => {
    const fx = notifierFixture();
    fx.notifier.handleFrame(
        { task_id: 't1', chat_id: 7, quiz: { quiz_id: 'q1', state: 'open', wait_for_answer: true } },
        { kind: 'quiz' },
    );
    fx.built[0].onclick();
    assert.equal(fx.focusedCount(), 1);
    assert.deepEqual(fx.activated, [{ chatId: 7, taskId: 't1', quizId: 'q1' }]);
    assert.equal(fx.built[0].closed, true);
});

test('denied or missing permission degrades to the in-app surface, never silence', () => {
    const denied = notifierFixture({ permission: 'denied' });
    const out = denied.notifier.handleFrame(
        { role: 'system', system_type: 'task_summary', task_id: 't1' }, { kind: 'chat', isMain: true },
    );
    assert.equal(out.surface, 'in_app');
    assert.equal(denied.built.length, 0);
    assert.deepEqual(denied.toasts, ['Task finished']);

    const unsupported = createNotifier({
        storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify(ON) }),
        notificationCtor: undefined,
        audioContextCtor: null,
        showToast: () => {},
        documentRef: fakeDocument(),
    });
    assert.equal(unsupported.supported(), false);
    assert.equal(unsupported.permission(), 'unsupported');
});

test('disabled preferences deliver nothing at all', () => {
    const fx = notifierFixture({ prefs: DEFAULT_NOTIFY_PREFS });
    assert.equal(fx.notifier.handleFrame(
        { role: 'system', system_type: 'task_summary', task_id: 't1' }, { kind: 'chat', isMain: true },
    ), null);
    assert.equal(fx.built.length, 0);
    assert.equal(fx.toasts.length, 0);
    assert.equal(fx.notifier.test(), null, 'the test button stays inert while off');
});

test('the test notification always shows its own text, and destroy releases listeners', () => {
    const fx = notifierFixture();
    const result = fx.notifier.test();
    assert.equal(result.surface, 'banner');
    assert.equal(fx.built.at(-1).options.body, 'This is a test notification.');
    assert.ok(fx.documentRef.handlers.has('change'));
    fx.notifier.destroy();
    assert.equal(fx.documentRef.handlers.size, 0);
    assert.equal(fx.notifier.handleFrame(
        { role: 'system', system_type: 'task_summary', task_id: 'later' }, { kind: 'chat', isMain: true },
    ), null, 'a destroyed notifier is inert');
});

test('settings controls paint current state and gate on the master switch', async () => {
    const inputs = [
        { getAttribute: () => 'enabled', checked: false, disabled: false },
        { getAttribute: () => 'task_done', checked: false, disabled: false },
    ];
    const status = { textContent: '' };
    const button = { disabled: false };
    const documentRef = fakeDocument({
        '[data-notify-pref]': inputs,
        '[data-notify-status]': [status],
        '[data-notify-test]': [button],
    });
    const storage = fakeStorage();
    const notifier = createNotifier({
        storage,
        notificationCtor: undefined,
        audioContextCtor: null,
        documentRef,
    });
    notifier.mountSettings();
    assert.equal(inputs[0].checked, false);
    assert.equal(inputs[1].disabled, true, 'categories are inert while notifications are off');
    assert.equal(button.disabled, true);
    assert.match(status.textContent, /off/);

    await notifier.setPref('enabled', true);
    assert.equal(inputs[0].checked, true);
    assert.equal(inputs[1].disabled, false);
    assert.equal(button.disabled, false);
    assert.equal(notifier.prefs.enabled, true);
    assert.deepEqual(JSON.parse(storage.map.get(NOTIFY_PREFS_KEY)), { ...ON });
    // An unknown key is refused rather than stored.
    await notifier.setPref('nope', true);
    assert.equal('nope' in notifier.prefs, false);
});

test('a reminder and a skill notice ring under the messages toggle, titled by their author', () => {
    const reminder = {
        type: 'chat', role: 'system', system_type: 'reminder', chat_id: 1, task_id: '', source: 'Ouroboros',
        ts: '2026-10-03T12:00:01.000001+00:00',
        content: 'Reminder · Ouroboros · written Oct 3 14:05 · for Oct 3 15:00 (UTC+3)\nCall mother',
    };
    const hit = classifyLiveFrame(reminder, { kind: 'chat', isMain: true });
    assert.equal(hit.category, 'important', 'no new category and no new toggle');
    assert.equal(hit.title, 'Reminder from Ouroboros');
    assert.equal(hit.body, 'Call mother', 'the host signature line stays out of the banner body');
    assert.equal(hit.key, 'important::2026-10-03T12:00:01.000001+00:00');
    const seen = new Set();
    assert.equal(decideNotification(hit, ON, seen).body, '', 'the words stay private until text is on');
    assert.equal(decideNotification(hit, { ...ON, show_text: true }, seen).body, 'Call mother');
    assert.equal(decideNotification(hit, { ...ON, important: false }, seen).reason, 'category_off');
    const notice = classifyLiveFrame({ ...reminder, system_type: 'skill_notice', source: 'calendar',
        content: 'Notice · calendar\nMeeting with Ivan in 15 min' }, { kind: 'chat', isMain: true });
    assert.equal(notice.title, 'Notice from calendar');
    assert.equal(notice.body, 'Meeting with Ivan in 15 min');
    assert.equal(noticeTitle('skill_notice', ''), 'Notice from a skill');
    assert.equal(noticeTitle('reminder', 'owner'), 'Reminder from owner', 'a row the owner set is signed by the owner');
});

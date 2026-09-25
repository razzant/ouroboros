import test from 'node:test';
import assert from 'node:assert/strict';

import {
    DEFAULT_NOTIFY_PREFS,
    NOTIFY_PREFS_KEY,
    classifyLiveFrame,
    createNotifier,
    decideNotification,
    attentionStatusText,
    normalizeNotifyPrefs,
    notifyStatusText,
    readNotifyPrefs,
    writeNotifyPrefs,
} from '../modules/notifications.js';

const ON = { ...DEFAULT_NOTIFY_PREFS, enabled: true };

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
    for (const state of ['answered', 'expired_terminal', 'superseded']) {
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
    assert.match(attentionStatusText({ nativeAttention: true }), /system sound/);
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

test('desktop host attention is requested once without replacing banner delivery', () => {
    let calls = 0;
    const notifier = createNotifier({
        storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify(ON) }),
        notificationCtor: undefined,
        audioContextCtor: null,
        showToast: () => {},
        documentRef: fakeDocument(),
        hostApi: { request_attention: () => { calls += 1; return { ok: true, status: 'native_sound' }; } },
    });
    const result = notifier.handleFrame(
        { role: 'system', system_type: 'task_summary', task_id: 'native-1' },
        { kind: 'chat', isMain: true },
    );
    assert.equal(result.surface, 'in_app');
    assert.equal(calls, 1);
    notifier.destroy();
});

test('banner owns sound, while silent in-app delivery still raises the window', () => {
    let calls = 0;
    class FakeNotification {
        static permission = 'granted';
        constructor() {}
    }
    const bannerNotifier = createNotifier({
        storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify(ON) }),
        notificationCtor: FakeNotification,
        audioContextCtor: null,
        hostApi: { request_attention: () => { calls += 1; return { ok: true }; } },
        documentRef: fakeDocument(),
    });
    assert.equal(bannerNotifier.handleFrame(
        { role: 'system', system_type: 'task_summary', task_id: 'banner-1' },
        { kind: 'chat', isMain: true },
    ).surface, 'banner');
    assert.equal(calls, 0, 'banner sound stays with the Notification API');
    bannerNotifier.destroy();

    let soundValue = null;
    const silentNotifier = createNotifier({
        storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify({ ...ON, sound: false }) }),
        notificationCtor: undefined,
        audioContextCtor: null,
        hostApi: { request_attention: (sound) => { soundValue = sound; return { ok: true }; } },
        showToast: () => {},
        documentRef: fakeDocument(),
    });
    assert.equal(silentNotifier.handleFrame(
        { role: 'system', system_type: 'task_summary', task_id: 'silent-1' },
        { kind: 'chat', isMain: true },
    ).surface, 'in_app');
    assert.equal(soundValue, false);
    silentNotifier.destroy();
});

test('in-app fallback tone is used when native window attention cannot play sound', async () => {
    let oscillators = 0;
    class FakeAudioContext {
        constructor() { this.currentTime = 0; this.destination = {}; }
        resume() {}
        close() {}
        createOscillator() { oscillators += 1; return { connect() {}, start() {}, stop() {} }; }
        createGain() { return { gain: { value: 0 }, connect() {} }; }
    }
    const notifier = createNotifier({
        storage: fakeStorage({ [NOTIFY_PREFS_KEY]: JSON.stringify(ON) }),
        notificationCtor: undefined,
        audioContextCtor: FakeAudioContext,
        hostApi: { request_attention: () => ({ ok: true, status: 'window_only', sound_played: false }) },
        showToast: () => {},
        documentRef: fakeDocument(),
    });
    notifier.handleFrame(
        { role: 'system', system_type: 'task_summary', task_id: 'window-only' },
        { kind: 'chat', isMain: true },
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(oscillators, 1);
    notifier.destroy();
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

test('an owner notification log frame is the skill-first category, titled by its source', () => {
    const frame = {
        type: 'owner_notification', category: 'notice', text: 'Meeting with Ivan in 15 min',
        source: 'skill:calendar', key: 'cal:evt-1', ts: '2026-09-25T14:30:00+00:00', chat_id: 1,
    };
    const hit = classifyLiveFrame(frame, { kind: 'log' });
    assert.equal(hit.category, 'notice');
    assert.equal(hit.title, 'Reminder from calendar');
    assert.equal(hit.body, 'Meeting with Ivan in 15 min');
    assert.equal(hit.key, 'notice:skill:calendar:cal:evt-1', 'the producer key collapses a redelivery');
    assert.deepEqual(hit.target, { chatId: 1 });
    // A model-free scheduler row of the agent's own is a reminder from Ouroboros;
    // without a producer key the instant is the identity.
    const own = classifyLiveFrame({ ...frame, source: 'task_followup', key: '' }, { kind: 'log' });
    assert.equal(own.title, 'Reminder from Ouroboros');
    assert.equal(own.key, 'notice:task_followup:2026-09-25T14:30:00+00:00');
    // The category is its own toggle and the sentence stays behind show_text.
    const seen = new Set();
    assert.equal(decideNotification(hit, { ...ON, notice: false }, seen).reason, 'category_off');
    const decision = decideNotification(hit, ON, seen);
    assert.equal(decision.deliver, true);
    assert.equal(decision.body, '', 'title only until the owner turns text on');
    assert.equal(decideNotification(hit, { ...ON, show_text: true }, seen).body, 'Meeting with Ivan in 15 min');
    assert.equal(DEFAULT_NOTIFY_PREFS.notice, true, 'on by default once notifications are on');
});

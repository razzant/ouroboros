// В9: until the host states `supervisor_ready: true`, the Main and Project
// headers say Starting… where they used to say Online — literally, so a
// readiness that is down with a named `supervisor_error` is still Starting….
// Work facts keep their words (Sending..., Queued..., Working...), a disconnect
// is still Reconnecting..., and the error text is never the header's.
import assert from 'node:assert/strict';
import test from 'node:test';
import {
    computeDerivedChatStatus,
    createStateSnapshotSequencer,
    supervisorReady,
} from '../modules/chat_activity.js';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

test('the reducer says Starting… only where it would say Online', () => {
    assert.deepEqual(computeDerivedChatStatus({ supervisorStarting: true }),
        { kind: 'starting', text: 'Starting…', showDots: false });
    assert.equal(computeDerivedChatStatus({ supervisorStarting: false }).text, 'Online');
    assert.equal(computeDerivedChatStatus({}).text, 'Online', 'callers that state nothing keep Online');
    const outranks = [
        [{ isConnected: false }, 'Reconnecting...'],
        [{ hasActiveLiveCard: true }, 'Working...'],
        [{ activeManagedCount: 1 }, 'Working...'],
        [{ activeDirectCount: 1 }, 'Thinking...'],
        [{ pendingSubmissionsCount: 1 }, 'Sending...'],
        [{ queuedManagedCount: 1 }, 'Queued...'],
        [{ waitingModelCount: 1 }, 'Waiting for access'],
        [{ pausedManagedCount: 1 }, 'Paused (budget)'],
    ];
    for (const [input, text] of outranks) {
        assert.equal(computeDerivedChatStatus({ ...input, supervisorStarting: true }).text, text,
            `${JSON.stringify(input)} is a fact about work, not about the start`);
    }
});

const CRASH = 'Supervisor loop died after 3 consecutive crashes: boom';

test('readiness is the typed boolean alone; an error beside it changes nothing', () => {
    assert.equal(supervisorReady({ supervisor_ready: true }), true);
    assert.equal(supervisorReady({ supervisor_ready: true, supervisor_error: 'Supervisor init failed: x' }), true);
    assert.equal(supervisorReady({ supervisor_ready: false, supervisor_error: null }), false);
    assert.equal(supervisorReady({ supervisor_ready: false, supervisor_error: CRASH }), false,
        'a named error is not readiness');
    for (const body of [null, undefined, 'x', {}, { supervisor_error: CRASH }, { supervisor_ready: 'true' }, { supervisor_ready: 1 }]) {
        assert.equal(supervisorReady(body), null, `${JSON.stringify(body)} states nothing`);
    }
});

test('the page sequencer keeps its newest applied body until a read is unavailable', () => {
    const applied = [];
    const snapshots = createStateSnapshotSequencer((data) => applied.push(data));
    assert.equal(snapshots.latest(), null);
    const older = snapshots.begin();
    const newer = snapshots.begin();
    assert.equal(snapshots.apply(newer, { supervisor_ready: true }), true);
    assert.deepEqual(snapshots.latest(), { supervisor_ready: true });
    assert.equal(snapshots.apply(older, { supervisor_ready: false }), false, 'a stale body never becomes the newest');
    assert.deepEqual(snapshots.latest(), { supervisor_ready: true });
    snapshots.fail(snapshots.begin());
    assert.equal(snapshots.latest(), null, 'a disconnect episode retires what the page knew');
    assert.equal(applied.length, 1);
});

function mountChat({ asPanel = false, chatId = asPanel ? 7 : 1, latest = null, connected = true, snapshots = null } = {}) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: [], window: { complete: true } }
            : { active_chat_activities: [], active_chat_activities_complete: true } }));
    const handlers = new Map();
    let generation = 0;
    let isConnected = connected;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => isConnected, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(asPanel ? [chatId] : []), unreadCount: 0 },
        updateUnreadBadge() {},
        // Fetched bodies are ignored: each test hands the instance its snapshot.
        stateSnapshots: snapshots || { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {}, fail() {}, latest: () => latest },
        chatId, idPrefix: 'chat', mountEl: env.mount, asPanel,
    });
    const badge = () => globalThis.document.byId.get('chat-status');
    return {
        instance, handlers,
        status: () => badge()?.textContent,
        kind: () => badge()?.className,
        snapshot: (patch) => instance.hydrateStateSnapshot({
            active_chat_activities: [], active_chat_activities_complete: true, ...patch,
        }, Infinity, ++generation),
        setConnected(value) { isConnected = value; handlers.get(value ? 'open' : 'close')?.({ previouslyConnected: true }); },
        settle: async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); },
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

test('Main: Starting… until supervisor_ready is true, error or not, then Online', () => {
    const chat = mountChat();
    try {
        chat.snapshot({ supervisor_ready: false, supervisor_error: null });
        assert.equal(chat.status(), 'Starting…');
        assert.equal(chat.kind(), 'status-badge starting', 'the existing amber starting tone, not the green one');
        chat.snapshot({ supervisor_ready: true });
        assert.equal(chat.status(), 'Online');
        assert.equal(chat.kind(), 'status-badge online');
        chat.snapshot({ supervisor_ready: false, supervisor_error: CRASH });
        assert.equal(chat.status(), 'Starting…', 'readiness went down: never Online, whatever the error says');
        assert.equal(chat.kind(), 'status-badge starting');
        chat.snapshot({ supervisor_ready: false, supervisor_error: CRASH,
            active_chat_activities: [{ activity_id: 'q-1', chat_id: 1, kind: 'managed_task', phase: 'queued' }] });
        assert.equal(chat.status(), 'Queued...', 'known work keeps its own word beside the error');
        chat.snapshot({ supervisor_ready: true, supervisor_error: CRASH });
        assert.equal(chat.status(), 'Online', 'only the typed readiness moves the pill; the error lives elsewhere');
    } finally { chat.close(); }
});

test('Main: a failed supervisor init is not ready — Starting…, never Online (TZ-1 host contract)', () => {
    // The host's failure rail (server.py `_run_supervisor`) publishes a failed
    // init as `supervisor_ready: false` beside its `supervisor_error`; the typed
    // boolean alone drives the pill, so the header never paints Online over it.
    const failed = { supervisor_ready: false, supervisor_error: 'Supervisor init failed: boot dependency refused' };
    assert.equal(supervisorReady(failed), false);
    const chat = mountChat();
    try {
        chat.snapshot(failed);
        assert.equal(chat.status(), 'Starting…');
        assert.equal(chat.kind(), 'status-badge starting');
        chat.snapshot(failed);
        assert.notEqual(chat.status(), 'Online', 'repeated failed-init bodies never settle into Online');
        chat.instance.hydrateStateSnapshot({ supervisor_error: failed.supervisor_error }, Infinity, 98);
        assert.equal(chat.status(), 'Starting…', 'an error without the readiness fact states nothing new');
    } finally { chat.close(); }
});

test('Main: a readiness-only body re-derives the header, and a body without the fact changes nothing', () => {
    const chat = mountChat();
    try {
        chat.snapshot({ supervisor_ready: true });
        assert.equal(chat.status(), 'Online');
        chat.instance.hydrateStateSnapshot({ supervisor_ready: false }, Infinity, 99);
        assert.equal(chat.status(), 'Starting…', 'no activity list is needed to state the start');
        chat.instance.hydrateStateSnapshot({ active_chat_activities: [], active_chat_activities_complete: true }, Infinity, 100);
        assert.equal(chat.status(), 'Starting…', 'silence is not readiness');
    } finally { chat.close(); }
});

test('Main: a disconnect forgets readiness, so a reconnect never flashes a stale Online', async () => {
    const chat = mountChat();
    try {
        chat.snapshot({ supervisor_ready: true });
        assert.equal(chat.status(), 'Online');
        chat.setConnected(false);
        assert.equal(chat.status(), 'Reconnecting...');
        chat.setConnected(true);
        assert.equal(chat.status(), 'Starting…', 'the host may have restarted: readiness is proven again');
        chat.snapshot({ supervisor_ready: true });
        assert.equal(chat.status(), 'Online');
        await chat.settle();
    } finally { chat.close(); }
});

test('Main: work during the start keeps its own word; only a ready census clears it', () => {
    const chat = mountChat();
    try {
        chat.snapshot({ supervisor_ready: false,
            active_chat_activities: [{ activity_id: 'q-1', chat_id: 1, kind: 'managed_task', phase: 'queued' }] });
        assert.equal(chat.status(), 'Queued...', 'a restored queue row is a fact about work, not the start');
        chat.snapshot({ supervisor_ready: false });
        assert.equal(chat.status(), 'Queued...', 'a not-ready census cannot prove the row gone');
        chat.snapshot({ supervisor_ready: true });
        assert.equal(chat.status(), 'Online');
    } finally { chat.close(); }
});

test('Project panel: seeds from the page snapshot instead of a literal Online', () => {
    for (const [latest, expected] of [
        [{ supervisor_ready: true }, 'Online'],
        [{ supervisor_ready: false }, 'Starting…'],
        [{ supervisor_ready: false, supervisor_error: CRASH }, 'Starting…'],
        [{ supervisor_error: CRASH }, 'Starting…'],
        [null, 'Starting…'],
    ]) {
        const panel = mountChat({ asPanel: true, latest });
        try {
            assert.equal(panel.status(), expected, `latest=${JSON.stringify(latest)}`);
            panel.snapshot({ supervisor_ready: true });
            assert.equal(panel.status(), 'Online', 'the fan-out snapshot settles the panel');
        } finally { panel.close(); }
    }
    const offline = mountChat({ asPanel: true, latest: { supervisor_ready: true }, connected: false });
    try {
        // The markup keeps its Connecting... text (the stub does not parse it).
        assert.ok(!['Online', 'Starting…'].includes(offline.status()), 'no socket yet: the seed claims nothing');
    } finally { offline.close(); }
});

test('Project panel: a late mount after a reconnect never seeds from the body before the drop', async () => {
    // The page wiring (web/app.js): the one sequencer fans every applied body to
    // the open chats and retires its newest body on each socket close and open.
    const chats = [];
    const page = createStateSnapshotSequencer((data, requestedAt, generation) => {
        for (const chat of chats) chat.instance.hydrateStateSnapshot(data, requestedAt, generation);
    });
    const beforeDrop = page.begin();
    page.apply(page.begin(), { supervisor_ready: true });
    assert.equal(page.latest()?.supervisor_ready, true);
    page.fail(page.begin()); // socket close
    page.fail(page.begin()); // socket open
    const panel = mountChat({ asPanel: true, snapshots: page });
    chats.push(panel);
    try {
        assert.equal(panel.status(), 'Starting…', 'the pre-drop ready body is gone, so the seed is not Online');
        assert.equal(page.apply(beforeDrop, { supervisor_ready: true }), false, 'a read begun before the drop is stale');
        assert.equal(panel.status(), 'Starting…');
        page.apply(page.begin(), { supervisor_ready: false, supervisor_error: CRASH });
        assert.equal(panel.status(), 'Starting…');
        page.apply(page.begin(), { supervisor_ready: true });
        assert.equal(panel.status(), 'Online', 'a post-reconnect ready body settles it');
        panel.setConnected(false);
        assert.equal(panel.status(), 'Reconnecting...');
        panel.setConnected(true);
        assert.equal(panel.status(), 'Starting…', 'the panel re-proves readiness after its own reconnect');
        await panel.settle();
        assert.equal(panel.status(), 'Starting…', "the reconnect's census read states no readiness");
    } finally { panel.close(); }
});

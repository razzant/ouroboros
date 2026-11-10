// Issue #1102, defect 3: app.js used to discard a rejected history paint with a
// bare `catch {}`. app.js is a boot script (top-level DOM wiring), so the ACK
// function is lifted out of its source and run against stubbed module state.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import * as readAhead from '../modules/read_ahead.js';
import { createRoomRetirement } from '../modules/room_retire.js';
import { forgetProjectPreviews } from '../modules/chat_preview.js';

// The span below is delimited by line breaks; normalize CRLF so a Windows
// checkout (core.autocrlf) reads the same bytes the delimiters expect.
const source = readFileSync(new URL('../app.js', import.meta.url), 'utf8').replace(/\r\n?/g, '\n');
const start = source.indexOf('async function acknowledgeProjectAfterPaint(');
const end = source.indexOf('\n}\n', start);
assert.ok(start >= 0 && end > start, 'acknowledgeProjectAfterPaint is a top-level function in app.js');
const ackSource = source.slice(start, end + 2);

function harness({ refreshHistory, page = {} }) {
    const acked = [], errors = [];
    const inst = {
        page: { hidden: false, isConnected: true, ...page },
        cancelHistoryPaint() {}, refreshHistory,
    };
    const context = vm.createContext({
        ...readAhead,  // app.js imports these (DESIGN "Unread": a message read in another room)
        navState: { activeProjectId: 'p1' },
        projectInstances: new Map([['p1', inst]]),
        projectPaintRequests: new Map(),
        projectReveals: new Map(),
        state: { projectSeenRevision: {} },
        markProjectViewed: async (id, revision) => { acked.push([id, revision]); },
        console: { error: (...args) => errors.push(args) },
    });
    vm.runInContext(`${ackSource}\nglobalThis.ack = acknowledgeProjectAfterPaint;`, context);
    return { acked, errors, inst, context, open: () => context.ack({ id: 'p1', visible_revision: 5 }, inst) };
}

test('app.js no longer swallows a rejected history paint silently', () => {
    assert.doesNotMatch(ackSource, /catch\s*\{\s*\}/, 'no bare catch around the history paint');
    assert.match(ackSource, /paint\?\.read/, 'the read receipt (painted at the newest messages) gates the ACK');
});

test('a rejected history paint is reported and never acknowledged', async () => {
    const failure = new Error('history paint exploded');
    const h = harness({ refreshHistory: async () => { throw failure; } });
    await h.open();
    assert.deepEqual(h.acked, []);
    assert.equal(h.errors.length, 1, 'the rejection reaches the console instead of vanishing');
    assert.equal(h.errors[0].at(-1), failure);
    assert.equal(h.context.projectPaintRequests.size, 0, 'the failed request is released, so Retry can run again');
});

test('a failed history read resolves unpainted and is never acknowledged', async () => {
    const h = harness({ refreshHistory: async ({ revision }) => ({ painted: false, revision }) });
    await h.open();
    assert.deepEqual(h.acked, []);
    assert.deepEqual(h.errors, [], 'an unpainted receipt is an ordinary outcome, not a defect');
});

for (const [name, mutate] of [
    ['hidden while the read was in flight', inst => { inst.page.hidden = true; }],
    ['destroyed (detached) while the read was in flight', inst => { inst.page.isConnected = false; }],
]) test(`a paint on a panel ${name} acknowledges nothing`, async () => {
    // The receipt says read: only the panel's own lifecycle guard can refuse it.
    const h = harness({ refreshHistory: async ({ revision }) => {
        mutate(h.inst);
        return { painted: true, read: true, revision };
    } });
    await h.open();
    assert.deepEqual(h.acked, [], 'a revision nobody saw is never acknowledged');
});

test('a painted, visible, connected panel acknowledges exactly the painted revision', async () => {
    const h = harness({ refreshHistory: async ({ revision }) => ({ painted: true, read: true, revision }) });
    await h.open();
    assert.deepEqual(h.acked, [['p1', 5]]);
});

test('a painted room whose reader is not at the newest messages acknowledges nothing', async () => {
    const h = harness({ refreshHistory: async ({ revision }) => ({ painted: true, read: false, revision }) });
    await h.open();
    assert.deepEqual(h.acked, [], 'opening or refreshing while reading older content is not reading');
    assert.equal(h.context.projectPaintRequests.size, 0, 'the next arrival at the newest messages can retry');
});

test('a question paint never rides an acknowledging request in flight; the reverse decides nothing', async () => {
    const pending = [];
    const h = harness({ refreshHistory: ({ revision }) => new Promise((resolve) => {
        pending.push(() => resolve({ painted: true, read: true, revision }));
    }) });
    const ordinary = h.open();
    const paintOnly = h.context.ack({ id: 'p1', visible_revision: 5 }, h.inst, { forcePaint: true, paintOnly: true });
    assert.equal(pending.length, 2, 'the question paint does not inherit the acknowledging request');
    const during = h.open();
    assert.equal(pending.length, 2, 'a request during the question paint rides it');
    pending[1]();
    await Promise.all([paintOnly, during]);
    assert.deepEqual(h.acked, [], 'the question paint, and whatever rode it, acknowledges nothing');
    pending[0]();
    await ordinary;
});

test('a paint-only open (a question revealed from Main) never acknowledges', async () => {
    const h = harness({ refreshHistory: async ({ revision }) => ({ painted: true, read: true, revision }) });
    await h.context.ack({ id: 'p1', visible_revision: 5 }, h.inst, { forcePaint: true, paintOnly: true });
    assert.deepEqual(h.acked, []);
});

// The shared cursor helpers are lifted the same way: markProjectViewed keeps the
// server's answer, and a re-read of the shared cursors only ever clears a dot.
function lift(name) {
    const from = source.indexOf(`function ${name}(`);
    const until = source.indexOf('\n}\n', from);
    assert.ok(from >= 0 && until > from, `${name} is a top-level function in app.js`);
    return source.slice(source.lastIndexOf('\n', from) + 1, until + 2);
}

function cursorHarness({ seen = {}, rows = [], answer = () => ({}) } = {}) {
    const posts = [], paints = [];
    const context = vm.createContext({
        ...readAhead,  // app.js imports these (DESIGN "Unread": a message read in another room)
        state: { projectSeenRevision: { ...seen } },
        lastProjectRows: rows,
        paintProjectsNav: () => paints.push(rows.map((row) => [row.id, row._unread])),
        paintProjectsCount: () => {}, patchProjectActivityMarkers: () => {}, mergeMainSeenRevision: () => false,
        fetchJson: async (url, init = {}) => { posts.push([url, init.method || 'GET', init.body || '']); return answer(init); },
    });
    vm.runInContext(`${lift('markProjectViewed')}\n${lift('mergeProjectSeenRevisions')}
        let sharedSeenRead = null;\n${lift('refreshSharedProjectSeen')}
        globalThis.api = { markProjectViewed, mergeProjectSeenRevisions, refreshSharedProjectSeen,
            pending: () => sharedSeenRead };`, context);
    return { context, posts, paints, api: context.api };
}

test('an ACK keeps the server-confirmed cursor, not the requested revision', async () => {
    const rows = [{ id: 'p1', visible_revision: 7, _unread: true }];
    // The server clamps a request of 7 to what it had when the ACK landed (6).
    const h = cursorHarness({ rows, answer: () => ({ ok: true, project_seen_revision: { p1: 6 } }) });
    assert.equal(await h.api.markProjectViewed('p1', 7), true);
    assert.equal(h.context.state.projectSeenRevision.p1, 6);
    assert.equal(rows[0]._unread, true, 'revision 7 stays unread until a read covers it');
});

test('shared cursors from another client clear a stale dot and never revive a read one', async () => {
    const rows = [{ id: 'p1', visible_revision: 4, _unread: true }, { id: 'p2', visible_revision: 2, _unread: false }];
    const h = cursorHarness({ seen: { p2: 2 }, rows,
        answer: () => ({ project_seen_revision: { p1: 4, p2: 1 } }) });
    h.api.refreshSharedProjectSeen();
    h.api.refreshSharedProjectSeen();
    assert.equal(h.posts.length, 1, 'one shared-cursor read at a time');
    await h.api.pending();
    assert.deepEqual(rows.map((row) => row._unread), [false, false]);
    assert.equal(h.context.state.projectSeenRevision.p2, 2, 'an older cursor never moves this one back');
    assert.deepEqual(h.paints, [[['p1', false], ['p2', false]]]);
    assert.equal(h.api.pending(), null, 'the next snapshot may read again');
    h.api.mergeProjectSeenRevisions({ p1: 3 });
    assert.equal(rows[0]._unread, false, 'a stale answer cannot revive the dot');
});

test('a failed ACK or cursor read changes nothing', async () => {
    const rows = [{ id: 'p1', visible_revision: 4, _unread: true }];
    const h = cursorHarness({ rows, answer: () => { throw new Error('offline'); } });
    assert.equal(await h.api.markProjectViewed('p1', 4), false);
    h.api.refreshSharedProjectSeen();
    await h.api.pending();
    assert.equal(rows[0]._unread, true);
    assert.deepEqual(h.context.state.projectSeenRevision, {});
});

test('only the Project room paint path posts a read cursor', () => {
    const callers = [...source.matchAll(/markProjectViewed\(/g)].length;
    assert.equal(callers, 2, 'its definition and the one call inside acknowledgeProjectAfterPaint');
    assert.match(ackSource, /await markProjectViewed\(project\.id, revision\)/);
});

// A question opened from Main (DESIGN "Unread"): landing on it is not
// reading the newer messages below it. The reveal is one transaction — the question
// owns the viewport before history I/O, its paint never joins an acknowledging
// request, and nothing is acknowledged until it ends.
function revealHarness() {
    const acked = [], calls = [], reveals = [];
    const pending = (list) => { let settle; const promise = new Promise((resolve) => { settle = resolve; }); list.push({ settle }); return promise; };
    const inst = {
        page: { hidden: false, isConnected: true }, generation: 0,
        cancelHistoryPaint() { this.generation += 1; },
        // Resolves when the test settles it; a cancelled paint lands unpainted, as chat.js's does.
        refreshHistory({ revision }) {
            const own = this.generation;
            const call = { revision, result: null };
            calls.push(call);
            return new Promise((resolve) => { call.settle = (result) => resolve(
                own === inst.generation ? { ...result, revision } : { painted: false, revision }); });
        },
        revealQuestion: () => pending(reveals),
    };
    const project = { id: 'p1', visible_revision: 5 };
    const context = vm.createContext({
        ...readAhead,  // app.js imports these (DESIGN "Unread": a message read in another room)
        navState: { activeProjectId: 'p1' },
        projectInstances: new Map([['p1', inst]]),
        projectPaintRequests: new Map(), projectReveals: new Map(),
        lastProjectRows: [project],
        state: { projectSeenRevision: {} },
        markProjectViewed: async (id, revision) => { acked.push([id, revision]); },
        console: { error() {} },
    });
    vm.runInContext(`${lift('freshProjectRow')}\n${ackSource}
        ${lift('revealProjectQuestion')}
        globalThis.api = { ack: acknowledgeProjectAfterPaint,
            reveal: (inst) => revealProjectQuestion(lastProjectRows[0], inst, 't1', 'q1') };`, context);
    return { acked, calls, reveals, inst, project, context, api: context.api };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

test('opening a question supersedes an acknowledgement in flight instead of joining it', async () => {
    const h = revealHarness();
    const ordinary = h.api.ack(h.project, h.inst);
    const revealed = h.api.reveal(h.inst);
    assert.equal(h.reveals.length, 1, 'the question owns the viewport before either history or detail I/O');
    await flush();
    assert.equal(h.calls.length, 2, 'the question paint is its own request, not the one in flight');
    h.calls[0].settle({ painted: true, read: true });
    await ordinary;
    assert.deepEqual(h.acked, [], 'the superseded request acknowledges nothing');
    h.calls[1].settle({ painted: true, read: true });
    await flush();
    assert.equal(h.calls.length, 2, 'no read decision while the question is still being revealed');
    h.reveals[0].settle(true);
    await flush();
    h.calls[2].settle({ painted: true, read: false });
    await revealed;
    assert.deepEqual(h.acked, [], 'landing on the question is not reading the newer messages');
});

test('no acknowledgement decides while a question reveal is in progress; the reveal decides after it', async () => {
    const h = revealHarness();
    const revealed = h.api.reveal(h.inst);
    await flush();
    h.calls[0].settle({ painted: true, read: true });
    await flush();
    assert.equal(h.reveals.length, 1);
    // A state poll or an arrival edge during the reveal still paints, but reads nothing.
    const during = h.api.ack(h.project, h.inst);
    await flush();
    h.calls[1].settle({ painted: true, read: true });
    await during;
    assert.deepEqual(h.acked, [], 'the reader has not been placed yet');
    h.reveals[0].settle(true);
    await flush();
    assert.equal(h.calls.length, 3, 'the reveal ends with its own read decision');
    h.calls[2].settle({ painted: true, read: true });
    await revealed;
    assert.deepEqual(h.acked, [['p1', 5]], 'a reader at the newest messages after the reveal has read them');
});

// A room left is kept away (room_retire.js), hidden, and its paint is cancelled. Back on screen it must start a
// read of its own: the cancelled request can only land unpainted, so joining it would leave the dot until the next
// revision or scroll.
test('a room back from away never joins the paint cancelled when it left', async () => {
    const h = revealHarness();
    h.inst.hasPendingWork = () => true;
    h.inst.page.dataset = {};
    h.context.createRoomRetirement = createRoomRetirement;
    vm.runInContext(`${lift('cancelProjectPaint')}
        const rooms = createRoomRetirement({ instances: projectInstances, cancelPaint: cancelProjectPaint });
        globalThis.api.close = (pid) => { navState.activeProjectId = null; rooms.putAway(pid); };`, h.context);
    const before = h.api.ack(h.project, h.inst);
    await flush();
    h.api.close('p1');
    assert.equal(h.inst.page.hidden, true, 'the room is away: hidden, not destroyed');
    h.context.navState.activeProjectId = 'p1';
    h.inst.page.hidden = false;
    const reopened = h.api.ack(h.project, h.inst);
    assert.equal(h.calls.length, 2, 'the reopened panel reads again');
    h.calls[0].settle({ painted: true, read: true });
    await before;
    assert.deepEqual(h.acked, [], 'the cancelled paint acknowledges nothing');
    h.calls[1].settle({ painted: true, read: true });
    await reopened;
    assert.deepEqual(h.acked, [['p1', 5]]);
});

// A state snapshot that still shows the open room unread offers the read decision
// again through the real renderProjectsNav → acknowledgeProjectAfterPaint path: a
// failed POST is retried by the next poll even when the snapshot is unchanged, and
// the retry still posts only a read (painted, shown, the reader at the newest message).
function pollHarness({ read = true, hidden = false, failures = 1 } = {}) {
    const posts = [], refreshes = [];
    let failing = failures;
    const inst = {
        page: { hidden, isConnected: true }, cancelHistoryPaint() {},
        refreshHistory: async ({ revision }) => { refreshes.push(revision); return { painted: true, read, revision }; },
    };
    const context = vm.createContext({
        ...readAhead,  // app.js imports these (DESIGN "Unread": a message read in another room)
        forgetProjectPreviews,
        state: { projectSeenRevision: {} }, navState: { activeProjectId: 'p1' },
        projectInstances: new Map([['p1', inst]]), projectPaintRequests: new Map(), projectReveals: new Map(),
        projectActivityIndex: null, knownProjectsJson: '', lastProjectRows: [], projectsNav: { unreadRows: () => [] }, rooms: { forget() {} },
        closeProjectPanel() {}, paintProjectsNav() {}, syncNavigationState() {}, patchProjectActivityMarkers() {},
        refreshSharedProjectSeen() {},
        fetchJson: async (_url, init = {}) => {
            const body = JSON.parse(init.body);
            posts.push(body.project_seen_revision);
            if (failing-- > 0) throw new Error('offline');
            return { ok: true, project_seen_revision: body.project_seen_revision };
        },
        console: { error() {} },
    });
    vm.runInContext(`${ackSource}\n${lift('markProjectViewed')}\n${lift('mergeProjectSeenRevisions')}
        ${lift('renderProjectsNav')}
        globalThis.poll = () => renderProjectsNav([{ id: 'p1', name: 'P', chat_id: 9, lifecycle: 'active',
            visible_revision: 5 }], [9]);`, context);
    const poll = async () => { context.poll(); for (let i = 0; i < 5; i++) await flush(); };
    return { context, posts, refreshes, poll, unread: () => context.lastProjectRows[0]?._unread };
}

test('a failed read acknowledgement is retried by the next unchanged state poll', async () => {
    const h = pollHarness();
    await h.poll();
    assert.deepEqual(h.posts, [{ p1: 5 }], 'the reader at the newest message is acknowledged');
    assert.equal(h.unread(), true, 'the failed POST leaves the room unread');
    await h.poll();
    assert.deepEqual(h.posts, [{ p1: 5 }, { p1: 5 }], 'the same snapshot retries the same revision');
    assert.equal(h.context.state.projectSeenRevision.p1, 5);
    assert.equal(h.unread(), false, 'the confirmed cursor clears the dot');
    await h.poll();
    assert.equal(h.posts.length, 2, 'a read room posts nothing more');
});

for (const [name, options, refreshes] of [
    ['whose reader is not at the newest message', { read: false }, 3],
    ['that is hidden', { hidden: true }, 0],
]) test(`state polls never acknowledge a room ${name}`, async () => {
    const h = pollHarness({ ...options, failures: 0 });
    for (let i = 0; i < 3; i++) await h.poll();
    assert.deepEqual(h.posts, [], 'retrying the decision is not acknowledging');
    assert.equal(h.refreshes.length, refreshes, 'a shown room re-reads through the existing receipt; a hidden one does nothing');
    assert.equal(h.unread(), true);
});

// A question reveal belongs to the navigation that started it. Closing the room while
// its detail read is held, then reopening it ordinarily, is a new navigation: the old
// reveal neither blocks nor decides that showing's read, whether the reopen reuses a
// hidden pending-work survivor (its draft and staged files) or builds a new instance.
function navigationHarness({ pendingWork, exitAtOnce = true, stack = false, covers = stack }) {
    const acked = [], built = [], held = [];
    const instance = () => {
        const inst = {
            page: { hidden: false, isConnected: true, dataset: {} }, generation: 0, draft: 'Yes, after the tag', leaving: false,
            // chat.js `leave`: its reads stop (a room leaving the screen); `closeTransient` closes the document it opened.
            leave() { this.leaving = true; this.generation += 1; }, stay() { this.leaving = false; },
            refreshes: 0, destroyed: false, latestShown: 0, transientCloses: 0, arrivals: [],
            hasPendingWork: () => pendingWork, hasPaintedHistory: () => true,
            // A return (chat.js `resume`): this room was left following its newest message.
            resume(arrive) { this.latestShown += 1; this.arrivals.push(arrive === true); },
            ws: { paused: false, pause() { this.paused = inst.away = true; }, resume() { this.paused = inst.away = false; } }, away: false,
            closeTransient() { this.transientCloses += 1; },
            cancelHistoryPaint() { this.generation += 1; },
            destroy() { this.destroyed = true; this.page.isConnected = false; },
            refreshHistory({ revision }) {
                const own = ++this.refreshes && this.generation;
                return Promise.resolve().then(() => (own === this.generation && !this.destroyed
                    ? { painted: true, read: true, revision } : { painted: false, revision }));
            },
            // A reveal naming a question awaits its detail read until the test releases it.
            revealQuestion: (taskId, quizId) => (taskId && quizId
                ? new Promise((resolve) => held.push(resolve)) : Promise.resolve(false)),
        };
        built.push(inst);
        return inst;
    };
    const project = { id: 'p1', name: 'P', chat_id: 9, lifecycle: 'active', visible_revision: 5 };
    const context = vm.createContext({
        ...readAhead,  // app.js imports these (DESIGN "Unread": a message read in another room)
        navState: { activeProjectId: null, mobileDrawerOpen: false },
        projectInstances: new Map(), projectPaintRequests: new Map(), projectReveals: new Map(),
        lastProjectRows: [project], state: { projectSeenRevision: {} }, mainChat: null,
        projectPanelTitle: {}, projectPanelBody: {}, ctx: {},
        // The panel's exit has run by the next sync here (as under reduced motion): a retired room is put away.
        showPage: async () => true, isStackLayout: () => stack, roomCoversMain: () => covers, createChatInstance: instance,
        document: { hidden: false },
        syncNavigationState() {
            if (context.mainChat) context.syncMainLifetime();
            if (exitAtOnce) for (const [pid, inst] of [...context.rooms.retiring]) context.rooms.settle(pid, inst);
        },
        markProjectViewed: async (id, revision) => { acked.push([id, revision]); },
        mainReads: 0, acknowledgeMainAfterPaint() { context.mainReads += 1; },  // Main's read decision (its own tests)
        mainUnread: 0, mainUnreadCount: () => context.mainUnread,
        console: { error() {} }, createRoomRetirement, roomSocket: (socket) => socket, ws: {},
    });
    vm.runInContext(`let projectNavigationGeneration = 0, projectPanelOpeningSince = 0;
        ${['mainOnScreen', 'enterMain', 'syncMainLifetime', 'cancelProjectPaint',
        'closeProjectPanel', 'openProjectPanel', 'resolveProjectRoom', 'freshProjectRow', 'revealProjectQuestion'].map(lift).join('\n')}\n${ackSource}
        const rooms = createRoomRetirement({ instances: projectInstances, cancelPaint: cancelProjectPaint });
        Object.assign(globalThis, { rooms, openProjectPanel });
        globalThis.api = { open: (options) => openProjectPanel(lastProjectRows[0], options), close: () => closeProjectPanel(),
            openAs: (project, options) => openProjectPanel(project, options) };`,
    context);
    return { acked, built, held, project, context, api: context.api };
}

// A document a chat opened (the reader, the file dialog) is a modal over the whole
// app: when its room leaves the screen it closes, and only it; the room is kept for its return.
// A room shown closes the document Main had open in either layout. Main itself leaves only where the room
// covers it (the stack layout); docked beside a Project it stays on screen and live (the owner's rule: Main is
// live while any of it is visible).
for (const stack of [true, false]) for (const pendingWork of [true, false]) {
    test(`a room ${pendingWork ? 'with' : 'without'} pending work leaving the screen leaves no document open over the next view (${stack ? 'stack' : 'docked'})`, async () => {
        const h = navigationHarness({ pendingWork, stack });
        // chat.js `leave` stops the room's reads and draws; `closeTransient` closes only the document it opened.
        const main = { page: { hidden: false, dataset: {} }, transientCloses: 0, leaves: 0,
            leave() { this.leaves += 1; }, closeTransient() { this.transientCloses += 1; }, resume() {},
            ws: { paused: false, pause() { this.paused = true; }, resume() { this.paused = false; } } };
        Object.assign(h.context, { mainChat: main });
        h.context.state.activePage = 'chat';
        await h.api.open();
        assert.equal(main.transientCloses, 1, 'a room shown over Main closes the document Main had open');
        assert.deepEqual([main.leaves, main.ws.paused], stack ? [1, true] : [0, false],
            stack ? 'covered, Main leaves and stops hearing its frames' : 'docked beside the room, Main stays live');
        const room = h.built[0];
        assert.equal(room.transientCloses, 0, 'the room shown keeps its own');
        h.api.close();
        assert.ok(room.transientCloses > 0, 'the kept room closes its document before it is hidden');
        assert.deepEqual([room.destroyed, room.page.hidden, room.away, room.page.dataset.pendingWork],
            [false, true, true, pendingWork ? '1' : undefined], 'and is kept away, marked when it holds staged files');
        assert.equal(room.draft, 'Yes, after the tag');
        await h.api.open();
        assert.equal(h.built.at(-1), room, 'the kept room is the one shown again');
        assert.deepEqual([room.page.hidden, room.away, room.page.dataset.pendingWork], [false, false, undefined]);
        assert.equal(main.transientCloses, 2);
        assert.equal(main.leaves, stack ? 2 : 0);
    });
}

// Who may say which room opens (CLIENT_CONTRACT §6): the caller's own answer, or the registry asked by the id.
test('a row that names its room opens as it stands; a newer answer is never replaced by the older feed row', async () => {
    const h = navigationHarness({ pendingWork: false });
    await h.api.openAs({ ...h.project, name: 'Renamed since that feed' });
    assert.equal(h.context.projectPanelTitle.textContent, 'Renamed since that feed');
    assert.equal(h.built.length, 1);
});

for (const [answer, says] of [[{ projects: [] }, 'gone'], [{ projects: [{ id: 'p1', name: 'P', chat_id: 9, lifecycle: 'deleting' }] }, 'deleting'],
    [{}, 'unreadable']]) {
    test(`a reference reads its room by its id: an older feed row opens nothing the registry answers ${says}`, async () => {
        const h = navigationHarness({ pendingWork: false });
        const reads = [], toasts = [];
        Object.assign(h.context, { fmt: (text) => text, showToast: (text) => toasts.push(text),
            apiClient: { projectById: async (id) => { reads.push(id); return answer; } } });
        await h.api.openAs({ id: 'p1', name: 'P' }, { openOnly: true });
        assert.deepEqual(reads, ['p1'], 'the feed names p1 active, and the registry is asked all the same');
        assert.equal(h.built.length, 0, 'no room is built');
        assert.equal(h.context.navState.activeProjectId, null, 'the view stays');
        assert.equal(toasts.length, 1, 'and says why');
    });
}

test('a reference opens what the registry answers for its id, a superseded read nothing', async () => {
    const h = navigationHarness({ pendingWork: false });
    const held = [];
    Object.assign(h.context, { apiClient: { projectById: () => new Promise((resolve) => held.push(resolve)) } });
    const first = h.api.openAs({ id: 'p1', name: 'Name the reference was drawn with' }, { openOnly: true });
    h.api.close();
    held.shift()({ projects: [{ ...h.project, name: 'Late' }] });
    await first;
    assert.equal(h.built.length, 0, 'a close supersedes the read');
    const second = h.api.openAs({ id: 'p1', name: 'Name the reference was drawn with' }, { openOnly: true });
    await flush();
    held.shift()({ projects: [{ ...h.project, name: 'The name it has now' }] });
    await second;
    assert.equal(h.context.projectPanelTitle.textContent, 'The name it has now');
    assert.equal(h.built.length, 1);
});

// Leaving a Project for Main is a return to Main (app.js `enterMain`, chat.js `resume`, which keeps a place left in
// older history; this Main was left following its newest message). Covered (a phone, a finger's docked column) Main
// comes back from away; docked beside the room it stayed live. Either way what came meanwhile is offered its read and
// lands (owner 06.11).
for (const [stack, covers, layout] of [[true, true, 'stack'], [false, true, 'docked under a finger'], [false, false, 'docked']]) {
    test(`leaving a Project for Main is a return to Main (${layout})`, async () => {
        const h = navigationHarness({ pendingWork: false, stack, covers });
        const main = { page: { hidden: false, dataset: {} }, returns: 0, arrivals: [],
            resume(arrive) { this.returns += 1; this.arrivals.push(arrive === true); },
            ws: { paused: false, pause() { this.paused = true; }, resume() { this.paused = false; } } };
        Object.assign(h.context, { mainChat: main });
        h.context.state.activePage = 'chat';
        h.context.mainUnread = 2;  // Main's news came while the Project was open
        await h.api.open();
        assert.equal(main.ws.paused, covers, covers ? 'covered, Main is away' : 'beside the room, Main stays live');
        h.api.close();
        assert.equal(main.returns, 1, 'the panel closed over Main: Main returns (chat.js `resume`)');
        assert.deepEqual(main.arrivals, [true], 'unread: it lands on what came, wherever later rows put it (DESIGN "Unread")');
        assert.equal(main.ws.paused, false);
        assert.equal(h.context.mainReads, 1, 'and what came while the Project was open is offered its read (owner 06.11)');
        h.api.close();
        assert.equal(main.returns, 1, 'closing with no Project open moves nothing');
        assert.equal(h.context.mainReads, 1);
        h.context.mainUnread = 0;
        await h.api.open();
        h.api.close();
        assert.deepEqual(main.arrivals, [true, false], 'a read Main opens at the bottom');
        h.context.state.activePage = 'settings';
        await h.api.open();
        assert.equal(main.ws.paused, true, 'on another page Main is away in either layout');
        h.api.close();
        assert.equal(main.returns, 2, 'on another page Main returns when Chat is shown again, not before');
    });
}

for (const pendingWork of [true, false]) {
    test(`an ordinary reopen of a kept room ${pendingWork ? 'with' : 'without'} pending work decides its own read while an old question reveal is held`, async () => {
        const h = navigationHarness({ pendingWork });
        const revealed = h.api.open({ openOnly: true, taskId: 't1', quizId: 'q1' });
        await flush();
        assert.equal(h.held.length, 1, 'the question detail read is held');
        assert.deepEqual(h.acked, [], 'landing on the question is not reading');
        assert.equal(h.built[0].latestShown, 0, 'a question opened from Main lands on the question');
        h.api.close();
        await h.api.open();
        const [first, reopened] = [h.built[0], h.built.at(-1)];
        assert.equal(reopened.latestShown, 1, 'an ordinary reopen is a return (left following: the newest message)');
        assert.equal(reopened, first, 'the kept room is reused');
        assert.equal(first.destroyed, false);
        assert.deepEqual(h.acked, [['p1', 5]], 'the reader at the newest message has read, the reveal still held');
        const refreshes = first.refreshes;
        h.held[0](true);
        await revealed;
        await flush();
        assert.deepEqual(h.acked, [['p1', 5]], 'the retired reveal decides nothing when it ends');
        assert.equal(first.refreshes, refreshes, 'nor reads for an instance its navigation left');
        assert.equal(h.context.projectReveals.size, 0);
        assert.equal(reopened.draft, 'Yes, after the tag', 'the reopened room keeps its draft');
    });
}

test('only an unread room opened from its row lands on what arrived last', async () => {
    const h = navigationHarness({ pendingWork: true });
    await h.api.open();
    const room = h.built[0];
    assert.deepEqual(room.arrivals, [true], 'unread: the opening lands on its newest arrival (owner decision 2026-10-11)');
    h.api.close();
    h.context.state.projectSeenRevision.p1 = 5;
    await h.api.open();
    assert.deepEqual(room.arrivals, [true, false], 'a read room reopens at its newest message, as before');
    h.api.close();
    h.context.state.projectSeenRevision.p1 = 4;
    const revealed = h.api.open({ openOnly: true, taskId: 't1', quizId: 'q1' });
    await flush();
    assert.deepEqual(room.arrivals, [true, false], 'a question opened from Main lands on the question');
    h.held[0](true);
    await revealed;
});

test('a return follows the place the room was left at; a notification outranks it and lands on what arrived', async () => {
    const h = navigationHarness({ pendingWork: false });
    await h.api.open();
    const room = h.built[0];
    room.showLatest = (options) => { room.latest = options; };
    h.api.close();
    await h.context.openProjectPanel(h.context.lastProjectRows[0], { openOnly: true, latest: true });
    assert.equal(room.arrivals.length, 1, 'no return decides the place');
    assert.equal(room.latest?.arrive, true, 'unread: the notification lands on what arrived last');
});

test('a question reveal still in progress keeps withholding the read of its own showing', async () => {
    const h = navigationHarness({ pendingWork: false });
    const revealed = h.api.open({ openOnly: true, taskId: 't1', quizId: 'q1' });
    await flush();
    const inst = h.built[0];
    await h.context.acknowledgeProjectAfterPaint(h.project, inst);
    assert.deepEqual(h.acked, [], 'an arrival or a poll during the reveal paints but decides nothing');
    h.held[0](true);
    await revealed;
    assert.deepEqual(h.acked, [['p1', 5]], 'the reveal decides once it ends');
});

// A Project room that leaves the screen keeps its content for its exit (owner 20.10, 4A; the desktop panel no longer
// fades out empty): it is retired, not destroyed, until the exit has run; its reads stop at once.
test('a closed room is retired for its exit, reopened as the same room, and a late read of it acknowledges nothing', async () => {
    const h = navigationHarness({ pendingWork: false, exitAtOnce: false });
    h.context.state.activePage = 'chat';
    await h.api.open();
    const room = h.built[0];
    const reads = room.refreshes;
    h.api.close();
    assert.deepEqual([room.destroyed, room.leaving, room.page.inert, 'leaving' in room.page.dataset], [false, true, true, true],
        'the room stays whole and inert while it leaves, its reads stopped');
    assert.equal(h.context.projectInstances.has('p1'), false, 'it is no longer a live instance');
    await h.api.open();
    assert.equal(h.built.length, 1, 'reopened during its exit, it is the same room');
    assert.deepEqual([room.leaving, room.page.inert, 'leaving' in room.page.dataset], [false, false, false]);
    assert.ok(room.refreshes >= reads, 'and it reads its history as any reopen does');
    // Closed again and put away when the exit has run: kept away, the same room for the next return.
    h.api.close();
    h.context.rooms.settle('p1', room);
    assert.deepEqual([room.destroyed, room.page.hidden, room.away], [false, true, true]);
    assert.equal(h.context.rooms.retiring.size, 0);
    await h.api.open();
    assert.equal(h.built.length, 1, 'the return shows the room kept away');
    h.context.rooms.settle('p1', room);
    assert.equal(room.destroyed, false, 'a late settle of an exit already ended changes nothing');
});

test('the acknowledgement owner refuses a paint from a room that is no longer the live one', async () => {
    const h = harness({ refreshHistory: async ({ revision }) => {
        h.context.projectInstances.set('p1', { page: { hidden: false, isConnected: true } });  // replaced meanwhile
        return { painted: true, read: true, revision };
    } });
    await h.open();
    assert.deepEqual(h.acked, [], 'a retired room\'s late read never acknowledges its replacement\'s revision');
});

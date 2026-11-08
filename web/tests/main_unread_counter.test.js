// Both sidebar numbers count the same thing (owner Q1 = B, 19.10.2026; DESIGN "Unread"): unread standalone
// messages, Main's from its room revision against the shared read cursor, the Projects' summed over the rows.
// app.js is a boot script, so the functions are lifted out of its source and run against stubbed module state.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

import * as readAhead from '../modules/read_ahead.js';

const source = readFileSync(new URL('../app.js', import.meta.url), 'utf8').replace(/\r\n?/g, '\n');

function lift(name) {
    const from = source.search(new RegExp(`\\n(async )?function ${name}\\(`));
    const until = source.indexOf('\n}\n', from + 1);
    assert.ok(from >= 0 && until > from, `${name} is a top-level function in app.js`);
    return source.slice(from + 1, until + 2);
}

// One top-level block of app.js that ends at a line `});` (a listener it registers), by how it starts.
function liftBlock(start) {
    const from = source.indexOf(`\n${start}`);
    assert.ok(from >= 0, `${start}… is a top-level block in app.js`);
    return source.slice(from + 1, source.indexOf('\n});\n', from + 1) + 5);
}

// One top-level statement of app.js, by how it starts (a declaration or a listener it registers).
function liftLine(start) {
    const from = source.indexOf(`\n${start}`);
    assert.ok(from >= 0, `${start}… is a top-level line in app.js`);
    return source.slice(from + 1, source.indexOf('\n', from + 1) + 1);
}

// What the Projects list says of the rows it holds (projects_nav.js `unreadRows`): here the feed's, or those a case adds.
const heldRows = (feed, beyond = []) => ({ unreadRows: () => [...(feed() || []).filter((row) => row._unread), ...beyond] });

function harness({ revision = null, seen = null, loaded = true, page = 'files', project = null, answer = () => ({}) } = {}) {
    const posts = [], badges = [], acks = [], landings = [];
    const moves = {};  // what Main's feed (#chat-messages) and its ↓ (#chat-scroll-bottom) listen to
    const context = vm.createContext({
        ...readAhead,  // app.js imports these: a message read in full elsewhere (DESIGN "Unread")
        state: { mainVisibleRevision: revision, mainSeenRevision: seen, mainSeenLoaded: loaded, activePage: page },
        navState: { activeProjectId: project },  // a Project open over Main (owner 06.11)
        updateUnreadBadge: () => badges.push(context.mainUnreadCount()),
        refreshSharedProjectSeen: () => posts.push(['shared']),
        acknowledgeMainAfterPaint: () => acks.push(context.state.mainVisibleRevision),
        mainChat: { showLatest: (options) => landings.push(options?.arrive === true) },
        document: { getElementById: (id) => ({ addEventListener(type, move) { moves[`${id} ${type}`] = move; } }) },
        fetchJson: async (url, init = {}) => {
            posts.push([url, init.method || 'GET', JSON.parse(init.body || '{}')]);
            return answer(JSON.parse(init.body || '{}'));
        },
    });
    vm.runInContext(liftLine('let mainOpening') + ['mainOnScreen', 'mainUnreadCount', 'observeMainRevision', 'landMain',
        'mergeMainSeenRevision', 'markMainViewed'].map(lift).join('\n') + liftLine("for (const id of ['chat-messages'")
        + '\nglobalThis.mainUnreadCount = mainUnreadCount;'
        + ' globalThis.api = { observeMainRevision, mergeMainSeenRevision, markMainViewed, mainUnreadCount };', context);
    return { context, posts, badges, acks, landings, moves, api: context.api };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

test("Main's number is its room revision less its read cursor; unknown counts none", () => {
    assert.equal(harness({ revision: 42, seen: 40 }).api.mainUnreadCount(), 2);
    assert.equal(harness({ revision: 40, seen: 42 }).api.mainUnreadCount(), 0);
    assert.equal(harness({ revision: null, seen: 40 }).api.mainUnreadCount(), 0);
    assert.equal(harness({ revision: 42, seen: null }).api.mainUnreadCount(), 0);
});

test('a cursor the server never held is set once to what Main holds, never per arrival', async () => {
    const h = harness({ seen: null, answer: (body) => ({ main_seen_revision: body.main_seen_revision }) });
    h.api.observeMainRevision(161);
    await flush();
    assert.deepEqual(h.posts, [['/api/ui/preferences', 'POST', { main_seen_revision: 161 }]]);
    assert.equal(h.context.state.mainSeenRevision, 161);
    assert.equal(h.api.mainUnreadCount(), 0, 'the history before this version reads as read');
    h.api.observeMainRevision(163);
    await flush();
    assert.equal(h.api.mainUnreadCount(), 2, 'later messages count');
    assert.equal(h.posts.filter(([url]) => url === '/api/ui/preferences').length, 1, 'no second initial write');
});

test('before the shared cursors are read nothing is written and nothing counts; they are read again', async () => {
    const h = harness({ seen: null, loaded: false });
    h.api.observeMainRevision(7);
    await flush();
    assert.deepEqual(h.posts, [['shared']], 'a read, never a write');
    assert.equal(h.api.mainUnreadCount(), 0);
});

test('a failed cursor read initializes nothing and is read again; an answer with the key then counts', () => {
    const h = harness({ seen: null, loaded: false });
    for (const failed of [null, undefined, {}, { error: 'busy' }]) {
        assert.equal(h.api.mergeMainSeenRevision(failed), false);
        assert.equal(h.context.state.mainSeenLoaded, false, 'an answer without the key proves nothing');
    }
    h.api.observeMainRevision(5);
    assert.deepEqual(h.posts, [['shared']], 'the cursor is read again; no write sets it to what Main holds');
    assert.equal(h.api.mainUnreadCount(), 0);
    assert.equal(h.api.mergeMainSeenRevision({ main_seen_revision: 2 }), true);
    assert.equal(h.api.mainUnreadCount(), 3, 'the stored cursor, not the current revision');
});

test('an unreadable revision keeps the known one', () => {
    const h = harness({ revision: 9, seen: 5 });
    h.api.observeMainRevision(null);
    assert.equal(h.context.state.mainVisibleRevision, 9);
    assert.equal(h.api.mainUnreadCount(), 4);
});

test('a newer revision re-reads the shared cursors, and offers the read decision only while Main is shown', () => {
    const away = harness({ revision: 3, seen: 3, page: 'files' });
    away.api.observeMainRevision(5);
    assert.deepEqual(away.posts, [['shared']]);
    assert.deepEqual(away.acks, []);
    const shown = harness({ revision: 3, seen: 3, page: 'chat' });
    shown.api.observeMainRevision(5);
    assert.deepEqual(shown.acks, [5]);
    // A Project open over Main (a finger's column, the desk's drawer, the phone's card): Main is not the room on screen,
    // so its new message counts and nothing reads it until Main is back (owner 06.11; DESIGN "Unread").
    const covered = harness({ revision: 3, seen: 3, page: 'chat', project: 'p1' });
    covered.api.observeMainRevision(5);
    assert.deepEqual(covered.acks, []);
    assert.equal(covered.api.mainUnreadCount(), 2);
});

// The app's start opens Main (DESIGN "Unread", owner 06.11): unread, it lands on what came, once; news that arrives
// while the owner is on Main never moves the reader, and a read Main opens at its bottom.
test('the app start lands an unread Main once; later arrivals never move it', () => {
    const opened = harness({ revision: 3, seen: 3, page: 'chat' });
    opened.api.observeMainRevision(5);
    opened.api.observeMainRevision(6);
    assert.deepEqual(opened.landings, [true], 'the first loaded snapshot is the opening, the next one only counts');
    assert.deepEqual(opened.acks, [5, 6]);
    const read = harness({ revision: 3, seen: 3, page: 'chat' });
    read.api.observeMainRevision(3);
    read.api.observeMainRevision(4);
    assert.deepEqual(read.landings, [], 'opened read: the bottom, and a later arrival does not jump');
    const unknown = harness({ revision: 3, seen: null, loaded: false, page: 'chat' });
    unknown.api.observeMainRevision(5);
    assert.deepEqual(unknown.landings, [], 'a cursor not yet read opens nothing yet');
    unknown.api.mergeMainSeenRevision({ main_seen_revision: 3 });
    unknown.api.observeMainRevision(5);
    assert.deepEqual(unknown.landings, [true], 'the opening waits for the cursor, then lands');
});

test("the reader's own first move in Main ends the start's opening; Main is still offered its read", () => {
    const moves = Object.keys(harness().moves);
    assert.deepEqual(moves.sort(), ['chat-messages', 'chat-scroll-bottom'].flatMap((id) => ['keydown', 'pointerdown',
        'touchstart', 'wheel'].map((type) => `${id} ${type}`)), "the feed's keys, wheel, touch and press, and its ↓");
    for (const move of moves) {
        const moved = harness({ revision: 3, seen: null, loaded: false, page: 'chat' });
        moved.api.observeMainRevision(5);  // a slow cursor read: the opening still waits
        moved.moves[move]();
        moved.api.mergeMainSeenRevision({ main_seen_revision: 3 });
        moved.api.observeMainRevision(5);
        assert.deepEqual([moved.landings, moved.acks], [[], [5]], `${move}: no landing over the reader's place`);
    }
});

// The read itself: a revision is written only after a paint made while Main is the room on screen. Main's reader
// reaching its newest message calls it as well, so the gate is the read's own, not only its callers'.
function paintHarness({ project = null, hold = false } = {}) {
    const paints = [], writes = [];
    let release = () => {};
    const context = vm.createContext({
        ...readAhead,
        console,
        state: { mainVisibleRevision: 5, mainSeenRevision: 3, mainSeenLoaded: true, activePage: 'chat' },
        navState: { activeProjectId: project },
        updateUnreadBadge: () => {},
        mainChat: { refreshHistory: ({ revision }) => {
            paints.push(revision);
            const painted = { read: true, revision };
            return hold ? new Promise((resolve) => { release = () => resolve(painted); }) : Promise.resolve(painted);
        } },
        fetchJson: async (url, init = {}) => {
            const body = JSON.parse(init.body || '{}');
            writes.push(body.main_seen_revision);
            return body;
        },
    });
    vm.runInContext(`let mainPaintRequest = null;\n${['mainOnScreen', 'mergeMainSeenRevision', 'markMainViewed',
        'acknowledgeMainAfterPaint'].map(lift).join('\n')}\nglobalThis.read = acknowledgeMainAfterPaint;`, context);
    return { context, paints, writes, release: () => release() };
}

test('Main is read only by a paint made while it is on screen; a Project over it, or opened meanwhile, takes none', async () => {
    const shown = paintHarness();
    await shown.context.read();
    assert.deepEqual([shown.paints, shown.writes], [[5], [5]], 'Main on screen: its newest revision is read');
    const covered = paintHarness({ project: 'p1' });
    await covered.context.read();
    assert.deepEqual([covered.paints, covered.writes], [[], []], 'a Project over Main: no paint for a read, no write');
    const opened = paintHarness({ hold: true });
    const reading = opened.context.read();
    opened.context.navState.activeProjectId = 'p1';  // a Project opens while Main's paint is under way
    opened.release();
    await reading;
    assert.deepEqual([opened.paints, opened.writes], [[5], []], 'the paint ends under the Project: the read is withheld');
});

test('the shared cursor only moves forward, and a write keeps what the server answers', async () => {
    const h = harness({ revision: 10, seen: 6, answer: () => ({ main_seen_revision: 8 }) });
    assert.equal(h.api.mergeMainSeenRevision({ main_seen_revision: 4 }), false, 'a stale answer never revives a read number');
    assert.equal(h.context.state.mainSeenRevision, 6);
    assert.equal(await h.api.markMainViewed(10), true);
    assert.equal(h.context.state.mainSeenRevision, 8, 'the clamped cursor the server confirmed');
    assert.equal(h.api.mainUnreadCount(), 2);
});

test('a failed write changes nothing', async () => {
    const h = harness({ revision: 10, seen: 6, answer: () => { throw new Error('offline'); } });
    assert.equal(await h.api.markMainViewed(10), false);
    assert.equal(h.context.state.mainSeenRevision, 6);
});

test('the Projects number sums the unread messages of the unread rows', () => {
    const count = { textContent: '', title: '', attrs: {},
        setAttribute(name, value) { this.attrs[name] = value; }, removeAttribute(name) { delete this.attrs[name]; } };
    const context = vm.createContext({
        ...readAhead,
        navProjectsCount: count,
        projectsNav: heldRows(() => context.lastProjectRows),
        state: { projectSeenRevision: { a: 1, b: 4, c: 0 } },
        lastProjectRows: [{ id: 'a', visible_revision: 4, _unread: true }, { id: 'b', visible_revision: 4, _unread: false },
            { id: 'c', visible_revision: 2, _unread: true }, { id: 'd', visible_revision: 3, _unread: false }],
    });
    vm.runInContext(`${lift('paintProjectsCount')}\nglobalThis.paint = paintProjectsCount;`, context);
    context.paint();
    assert.equal(count.textContent, '5');
    assert.equal(count.attrs['aria-label'], '5 unread messages in Projects');
    context.lastProjectRows = [];
    context.paint();
    assert.equal(count.textContent, '');
    assert.equal(count.attrs['aria-label'], undefined);
});

test('a message read in full in another room leaves the counts without moving a cursor', () => {
    readAhead.noteReadAhead({ read_ahead_revisions: { 1: [[42, 42]], 5551: [[4, 4]] } });
    try {
        const main = harness({ revision: 42, seen: 40 });
        assert.equal(main.api.mainUnreadCount(), 1);  // 41 still unread; 42 was read in its Project
        assert.equal(main.context.state.mainSeenRevision, 40);
        const count = { textContent: '', title: '', attrs: {}, setAttribute() {}, removeAttribute() {} };
        const context = vm.createContext({ ...readAhead, navProjectsCount: count, state: { projectSeenRevision: { nova: 1 } },
            projectsNav: heldRows(() => context.lastProjectRows),
            lastProjectRows: [{ id: 'nova', chat_id: 5551, visible_revision: 4, _unread: true }] });
        vm.runInContext(`${lift('paintProjectsCount')}\nglobalThis.paint = paintProjectsCount;`, context);
        context.paint();
        assert.equal(count.textContent, '2');  // p1 and p2 unread; m3 read in Main
    } finally {
        readAhead.noteReadAhead({ read_ahead_revisions: {} });  // answers join (read_ahead.js): this resets nothing, later cases use other revisions
    }
});

test("another client's read reaches this one at its next state snapshot, a late older answer taking nothing back", async () => {
    // The host as both clients share it: one client posts its shared read there; this client only polls state.
    const host = { main_seen_revision: 70, project_seen_revision: { nova: 1 }, read_ahead_revisions: {} };
    const answers = [];  // this client's preferences reads, answered when the test says (the network's order)
    const count = { textContent: '', title: '', attrs: {}, setAttribute() {}, removeAttribute() {} };
    const context = vm.createContext({
        ...readAhead,
        navProjectsCount: count,
        projectsNav: heldRows(() => context.lastProjectRows),
        state: { mainVisibleRevision: 72, mainSeenRevision: 70, mainSeenLoaded: true, activePage: 'files',
            projectSeenRevision: { nova: 1 } },
        navState: { activeProjectId: null },
        lastProjectRows: [{ id: 'nova', chat_id: 7302, visible_revision: 4, _unread: true }],
        updateUnreadBadge() {}, paintProjectsNav() {}, patchProjectActivityMarkers() {}, acknowledgeMainAfterPaint() {},
        fetchJson: (url, init = {}) => new Promise((resolve) => {
            assert.equal(url, '/api/ui/preferences');
            assert.equal(init.method || 'GET', 'GET');
            answers.push(() => resolve(JSON.parse(JSON.stringify(host))));  // the host's state when it answers
        }),
    });
    vm.runInContext(liftLine('let mainOpening') + ['mainOnScreen', 'mainUnreadCount', 'observeMainRevision', 'landMain',
        'mergeMainSeenRevision', 'mergeProjectSeenRevisions', 'refreshSharedProjectSeen', 'paintProjectsCount'].map(lift).join('\n')
        + '\nlet sharedSeenRead = null;\nglobalThis.api = { observeMainRevision, mainUnreadCount, paintProjectsCount };', context);
    context.api.paintProjectsCount();
    assert.deepEqual([context.api.mainUnreadCount(), count.textContent], [2, '3']);
    const stale = JSON.parse(JSON.stringify(host));  // what an earlier read saw, before the other client read m3
    host.read_ahead_revisions = { 1: [[72, 72]], 7302: [[4, 4]] };  // the other client read m3 (Main 72, Nova 4)
    context.api.observeMainRevision(72);  // an ordinary snapshot: nothing new, Main still shows unread
    assert.equal(answers.length, 1, 'a snapshot that still shows unread reads the shared cursors again');
    answers.shift()();
    await flush();
    assert.deepEqual([context.api.mainUnreadCount(), count.textContent], [1, '2'], 'm3 read once; p1, p2 and 71 stay');
    readAhead.noteReadAhead({ ...stale, read_ahead_revisions: {} });  // an older answer arriving last
    context.api.paintProjectsCount();
    assert.deepEqual([context.api.mainUnreadCount(), count.textContent], [1, '2']);
});

function snapshotHarness({ host, beyond = [], seen }) {
    const answers = [];
    const count = { textContent: '', title: '', attrs: {}, setAttribute() {}, removeAttribute() {} };
    const context = vm.createContext({
        ...readAhead,
        navProjectsCount: count,
        heldRows, beyond,
        navState: { activeProjectId: null },
        state: { mainVisibleRevision: 90, mainSeenRevision: 90, mainSeenLoaded: true, activePage: 'files',
            projectSeenRevision: seen },
        closeProjectPanel() {}, patchProjectActivityMarkers() {}, syncNavigationState() {}, acknowledgeProjectAfterPaint() {},
        updateUnreadBadge() {},
        fetchJson: () => new Promise((resolve) => { answers.push(() => resolve(JSON.parse(JSON.stringify(host)))); }),
    });
    vm.runInContext(['renderProjectsNav', 'mergeMainSeenRevision', 'mergeProjectSeenRevisions', 'refreshSharedProjectSeen',
        'paintProjectsCount'].map(lift).join('\n')
        + '\nlet sharedSeenRead = null, knownProjectsJson = null, lastProjectRows = null, projectActivityIndex = {};'
        + '\nconst projectsNav = heldRows(() => lastProjectRows, beyond);'
        + '\nfunction paintProjectsNav() { paintProjectsCount(); }\nglobalThis.api = { renderProjectsNav };', context);
    return { context, count, answers };
}

test("a snapshot showing only a Project unread reads another client's record as well", async () => {
    const host = { main_seen_revision: 90, project_seen_revision: { vega: 2 }, read_ahead_revisions: { 7303: [[5, 5]] } };
    const { context, count, answers } = snapshotHarness({ host, seen: { vega: 2 } });
    context.api.renderProjectsNav([{ id: 'vega', chat_id: 7303, visible_revision: 5, lifecycle: 'active' }], [7303]);
    assert.equal(count.textContent, '3');  // 3, 4 and 5: the other client read 5 in Main, this one does not know yet
    assert.equal(answers.length, 1, 'Main shows nothing unread: the Project row alone reads the shared cursors again');
    answers.shift()();
    await flush();
    assert.equal(count.textContent, '2');
});

test("a room beyond the feed counts and asks after another client's read though the feed shows nothing unread", async () => {
    const host = { main_seen_revision: 90, project_seen_revision: { old: 3 }, read_ahead_revisions: {} };
    const old = { id: 'old', chat_id: 7304, visible_revision: 3 };
    const { context, count, answers } = snapshotHarness({ host, beyond: [old], seen: { old: 1 } });
    context.api.renderProjectsNav([{ id: 'vega', chat_id: 7303, visible_revision: 0, lifecycle: 'active' }], [7303, 7304]);
    assert.equal(count.textContent, '2', 'revisions 2 and 3 of the room a complete list showed');
    assert.equal(answers.length, 1, 'that room alone reads the shared cursors again');
    context.beyond.length = 0;  // the room reads as read once its cursor has moved, as the list module would say
    answers.shift()();
    await flush();
    assert.equal(count.textContent, '');
});

// Main shown again from another page (DESIGN "Unread"): the room's own page-shown listener claims a plain return to its
// newest message during the event; while Main is unread the shell's claim after it lands on what came, wherever later
// rows put it. A Project opening over Main lands on its own room instead.
function pageHarness({ unread = 0 } = {}) {
    const claims = [];
    const context = vm.createContext({
        state: { activePage: 'files' },
        navState: { activeProjectId: null, mobileDrawerOpen: false },
        beforePageLeaveHandlers: [],
        mainChat: { showLatest(options) { claims.push(options?.arrive === true); }, closeTransient() {} },
        document: { querySelectorAll: () => [], getElementById: () => null },
        window: { dispatchEvent(event) { if (event.detail?.page === 'chat') context.mainChat.showLatest(); } },
        CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init?.detail; } },
        updateUnreadBadge() {}, acknowledgeMainAfterPaint() {}, syncNavigationState() {},
        // It lands Main itself, unless it closes for a reveal (a task's place), which brings its own.
        closeProjectPanel(options) { if (context.navState.activeProjectId && !options?.reveal) claims.push('closed'); },
        mainUnreadCount: () => unread,
    });
    vm.runInContext(`${['showPage', 'mainOnScreen', 'landMain'].map(lift).join('\n')}\nglobalThis.show = showPage;`, context);
    return { context, claims };
}

test('Main shown again from another page lands on what came while unread; read, or under an opening Project, it does not', async () => {
    const unread = pageHarness({ unread: 2 });
    await unread.context.show('chat');
    assert.deepEqual(unread.claims, [false, true], 'the landing is the last claim, so it supersedes the plain return');
    const read = pageHarness();
    await read.context.show('chat');
    assert.deepEqual(read.claims, [false], 'a read Main returns to its bottom');
    const opening = pageHarness({ unread: 2 });
    await opening.context.show('chat', { closeProject: false });
    assert.deepEqual(opening.claims, [false], 'a Project opening over Main: Main lands when it is shown again');
    const leaving = pageHarness({ unread: 2 });
    leaving.context.state.activePage = 'chat';
    await leaving.context.show('settings');
    assert.deepEqual(leaving.claims, [], 'another page shown while Main has news claims nothing for Main');
});

test("Main pressed again while shown (its row, a notification) lands what came; read, or under a Project, it does not", async () => {
    const pressed = pageHarness({ unread: 2 });
    pressed.context.state.activePage = 'chat';
    await pressed.context.show('chat');
    assert.deepEqual(pressed.claims, [true], 'no page change, so no plain return: the landing alone');
    const read = pageHarness();
    read.context.state.activePage = 'chat';
    await read.context.show('chat');
    assert.deepEqual(read.claims, [], 'a read Main keeps its reader where it is');
    const covered = pageHarness({ unread: 2 });
    Object.assign(covered.context, { state: { activePage: 'chat' }, navState: { activeProjectId: 'p1' } });
    await covered.context.show('chat');
    assert.deepEqual(covered.claims, ['closed'], 'the Project closes and lands Main: one claim, not two');
    const task = pageHarness({ unread: 2 });
    await task.context.show('chat', { reveal: true });
    assert.deepEqual(task.claims, [false], "a task's place (ouro:open-task) is its own destination: Main's news waits");
    const taskOverProject = pageHarness({ unread: 2 });
    Object.assign(taskOverProject.context, { state: { activePage: 'chat' }, navState: { activeProjectId: 'p1' } });
    await taskOverProject.context.show('chat', { reveal: true });
    assert.deepEqual(taskOverProject.claims, [], 'the Project closes for the reveal: it lands nothing of Main either');
});

test("a task opened in Main (ouro:open-task) shows Main as a reveal, never as an opening", async () => {
    const listeners = {}, shown = [];
    const context = vm.createContext({
        window: { addEventListener: (type, listener) => { listeners[type] = listener; } },
        showPage: async (...args) => { shown.push(args); return false; },  // stop there: the reveal is another test's
        state: { activePage: 'chat' }, revealQuotedSource: async () => ({ state: 'found' }), showToast() {}, tr: (_k, s) => s,
    });
    vm.runInContext(liftBlock("window.addEventListener('ouro:open-task'"), context);
    await listeners['ouro:open-task']({ detail: { task_id: 't1', chat_id: 1 } });
    assert.deepEqual(JSON.parse(JSON.stringify(shown)), [['chat', { reveal: true }]]);
});

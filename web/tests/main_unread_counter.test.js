// Both sidebar numbers count the same thing (owner Q1 = B, 19.10.2026; DESIGN "Unread"): unread standalone
// messages, Main's from its room revision against the shared read cursor, the Projects' summed over the rows.
// app.js is a boot script, so the functions are lifted out of its source and run against stubbed module state.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

import * as readAhead from '../modules/read_ahead.js';
import { forgetProjectPreviews } from '../modules/chat_preview.js';

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
        mainChat: { showLatest: (options) => landings.push(options?.arrive === true), resume: (arrive) => landings.push(['resume', arrive]) },
        document: { getElementById: (id) => ({ addEventListener(type, move) { moves[`${id} ${type}`] = move; } }) },
        fetchJson: async (url, init = {}) => {
            posts.push([url, init.method || 'GET', JSON.parse(init.body || '{}')]);
            return answer(JSON.parse(init.body || '{}'));
        },
    });
    vm.runInContext(liftLine('let mainOpening') + ['mainOnScreen', 'mainUnreadCount', 'observeMainRevision', 'enterMain',
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
    vm.runInContext(liftLine('let mainOpening') + ['mainOnScreen', 'mainUnreadCount', 'observeMainRevision', 'enterMain',
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
        heldRows, beyond, forgetProjectPreviews,
        navState: { activeProjectId: null },
        state: { mainVisibleRevision: 90, mainSeenRevision: 90, mainSeenLoaded: true, activePage: 'files',
            projectSeenRevision: seen },
        closeProjectPanel() {}, patchProjectActivityMarkers() {}, syncNavigationState() {}, acknowledgeProjectAfterPaint() {},
        updateUnreadBadge() {}, rooms: { forget() {} },  // every read names the projects going away (room_retire.js)
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

// Main shown from elsewhere, or pressed again (DESIGN "Unread"; #2911 composed with the return rule of 3 Nov 2026).
// Shown from another page Main comes back from away in the navigation sync (`syncMainLifetime`), and that is a return:
// chat.js `resume` puts it as it was left (following: lands on what came; in older history: stays, ↓ marked, unread).
// Pressed again while it is the page shown, it is an opening that lands an unread Main. A Project opening over Main
// lands its own room.
function pageHarness({ unread = 0, page = 'files', project = null } = {}) {
    const claims = [];
    const context = vm.createContext({
        state: { activePage: page },
        navState: { activeProjectId: project, mobileDrawerOpen: false },
        beforePageLeaveHandlers: [],
        // chat.js: `showLatest` goes to the present (with `arrive`, onto what came); `resume` is a return, as left.
        mainChat: { showLatest(options) { claims.push(options?.arrive === true ? 'land' : 'latest'); },
            resume(arrive) { claims.push(['resume', arrive]); }, closeTransient() {}, ws: { paused: page !== 'chat' } },
        rooms: { suspend(room) { room.ws.paused = true; }, restore(room) { room.ws.paused = false; },
            forgetShownSelections(others, options) { context.forgottenSelections += 1; context.forgetOptions = options; } },
        forgottenSelections: 0,
        roomCoversMain: () => true,
        document: { hidden: false, querySelectorAll: () => [], getElementById: () => null },
        window: { dispatchEvent() {} },
        CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init?.detail; } },
        updateUnreadBadge() {}, acknowledgeMainAfterPaint() {},
        syncNavigationState() { context.syncMainLifetime(); },
        // Its own return to Main is project_history_ack.test.js's; here it says only that it closed.
        closeProjectPanel(options) {
            if (context.navState.activeProjectId && !options?.reveal) claims.push('closed');
            context.navState.activeProjectId = null;
        },
        mainUnreadCount: () => unread,
    });
    vm.runInContext(`${['showPage', 'mainOnScreen', 'enterMain', 'syncMainLifetime'].map(lift).join('\n')}
        Object.assign(globalThis, { show: showPage, syncMainLifetime, enterMain });`, context);
    return { context, claims };
}

test('Main shown from another page is a return: chat.js `resume` puts it as left, unread on what came', async () => {
    const unread = pageHarness({ unread: 2 });
    await unread.context.show('chat');
    assert.deepEqual(unread.claims, [['resume', true]], 'one claim, the return: no landing over a place left in older history');
    assert.equal(unread.context.mainChat.ws.paused, false, 'and Main is back from away');
    const read = pageHarness();
    await read.context.show('chat');
    assert.deepEqual(read.claims, [['resume', false]], 'a read Main returns as left');
    const opening = pageHarness({ unread: 2 });
    await opening.context.show('chat', { closeProject: false, sync: false });  // as openProjectPanel shows the page
    assert.deepEqual(opening.claims, [], 'a Project opening over Main: Main returns when it is shown again');
    const leaving = pageHarness({ unread: 2, page: 'chat' });
    await leaving.context.show('settings');
    assert.deepEqual(leaving.claims, [], 'another page shown while Main has news claims nothing for Main');
    assert.equal(leaving.context.mainChat.ws.paused, true, 'Main behind another page is away');
});

test("Main pressed again while shown (its row) lands what came; read, under a Project, or for a task's place it does not", async () => {
    const pressed = pageHarness({ unread: 2, page: 'chat' });
    await pressed.context.show('chat');
    assert.deepEqual(pressed.claims, ['land'], 'no page change, so no return: the landing alone');
    const read = pageHarness({ page: 'chat' });
    await read.context.show('chat');
    assert.deepEqual(read.claims, [], 'a read Main keeps its reader where it is');
    const covered = pageHarness({ unread: 2, page: 'chat', project: 'p1' });
    await covered.context.show('chat');
    assert.deepEqual(covered.claims, ['closed'], 'the Project closes and Main returns from under it, not an opening');
    const task = pageHarness({ unread: 2 });
    await task.context.show('chat', { reveal: true });
    assert.deepEqual(task.claims, [['resume', true]],
        "a task's place (ouro:open-task) lands nothing: Main comes back as left, and the reveal's own claim follows");
    const taskOverProject = pageHarness({ unread: 2, page: 'chat', project: 'p1' });
    await taskOverProject.context.show('chat', { reveal: true });
    assert.deepEqual(taskOverProject.claims, [], 'the Project closes for the reveal: it lands nothing of Main either');
});

// Every way onto Main but a reveal passes `enterMain`: the app's start and Main pressed again (`land`) and a
// notification land an unread Main on what came; a return (`return`, `window`) is chat.js `resume`, as it was left.
test('an opening and a notification land an unread Main; a return and a window shown again resume it as left', () => {
    const cases = [
        ['land', 2, ['land']], ['land', 0, []],
        ['notification', 2, ['land']], ['notification', 0, ['latest']],  // a notification leads to the present in any case
        ['return', 2, [['resume', true]]], ['return', 0, [['resume', false]]],
        ['window', 2, [['resume', true]]], ['window', 0, [['resume', false]]],
    ];
    for (const [how, unread, claims] of cases) {
        const h = pageHarness({ unread, page: 'chat' });
        let reads = 0;
        h.context.acknowledgeMainAfterPaint = () => { reads += 1; };
        h.context.enterMain(how);
        assert.deepEqual([h.claims, reads], [claims, 1], `${how}, ${unread ? 'unread' : 'read'}: offered its read too`);
    }
    const covered = pageHarness({ unread: 2, page: 'chat', project: 'p1' });
    let coveredReads = 0;
    covered.context.acknowledgeMainAfterPaint = () => { coveredReads += 1; };
    covered.context.enterMain('land');
    covered.context.enterMain();
    assert.deepEqual([covered.claims, coveredReads], [[['resume', false]], 0],
        'beside a Project Main is not the room on screen: nothing lands and nothing is read');
});

test('a window shown again brings Main back as a window, which keeps its place, and a hidden one sends it away', () => {
    const h = pageHarness({ unread: 2, page: 'chat' });
    const hows = [];
    const enter = h.context.enterMain;
    h.context.enterMain = (how) => { hows.push(how); enter(how); };
    h.context.document.hidden = true;
    vm.runInContext(liftBlock("document.addEventListener('visibilitychange'"), Object.assign(h.context, {
        document: Object.assign(h.context.document, { addEventListener: (type, listener) => { h.context.shown = listener; } }),
        projectInstances: new Map(),
    }));
    h.context.shown();
    assert.equal(h.context.mainChat.ws.paused, true, 'hidden, Main is away');
    h.context.document.hidden = false;
    h.context.shown();
    assert.deepEqual([hows, h.claims], [['window'], [['resume', true]]], 'shown again: a window return, as left');
    assert.equal(h.context.forgottenSelections, 1, 'shown again, the copies kept while hidden go: the live selection is its own');
    assert.equal(h.context.forgetOptions?.keepCovered, true, 'a room a layer still covers keeps its copy');
});

// A notification is an explicit destination (DESIGN "History edges"): Main opens on what came, a Project at its latest.
test('a notification opens Main on what came and a Project at its latest, outranking where each was left', async () => {
    const shown = [], entered = [], opened = [];
    let activation = null;
    const context = vm.createContext({
        getNotifier: () => ({ configure: ({ onActivate }) => { activation = onActivate; } }),
        createNotificationActivation: (handlers) => handlers,
        showToast() {}, fetchJson: async () => ({}),
        showPage: async (...args) => { shown.push(args); return true; },
        enterMain: (how) => entered.push(how),
        openProjectPanel: (project, options) => opened.push([project.id, options]),
        closeNotificationReaders() {},
    });
    vm.runInContext(liftBlock('getNotifier().configure('), context);
    await activation.openMain();
    assert.deepEqual([shown, entered], [[['chat']], ['notification']]);
    await activation.openProject({ id: 'p1' }, { taskId: 't1' });
    assert.deepEqual(JSON.parse(JSON.stringify(opened)), [['p1', { taskId: 't1', latest: true }]]);
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

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

function harness({ revision = null, seen = null, loaded = true, page = 'files', answer = () => ({}) } = {}) {
    const posts = [], badges = [], acks = [];
    const context = vm.createContext({
        ...readAhead,  // app.js imports these: a message read in full elsewhere (DESIGN "Unread")
        state: { mainVisibleRevision: revision, mainSeenRevision: seen, mainSeenLoaded: loaded, activePage: page },
        updateUnreadBadge: () => badges.push(context.mainUnreadCount()),
        refreshSharedProjectSeen: () => posts.push(['shared']),
        acknowledgeMainAfterPaint: () => acks.push(context.state.mainVisibleRevision),
        fetchJson: async (url, init = {}) => {
            posts.push([url, init.method || 'GET', JSON.parse(init.body || '{}')]);
            return answer(JSON.parse(init.body || '{}'));
        },
    });
    vm.runInContext(['mainUnreadCount', 'observeMainRevision', 'mergeMainSeenRevision', 'markMainViewed']
        .map(lift).join('\n') + '\nglobalThis.mainUnreadCount = mainUnreadCount; globalThis.api = { observeMainRevision,'
        + ' mergeMainSeenRevision, markMainViewed, mainUnreadCount };', context);
    return { context, posts, badges, acks, api: context.api };
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
        state: { mainVisibleRevision: 72, mainSeenRevision: 70, mainSeenLoaded: true, activePage: 'files',
            projectSeenRevision: { nova: 1 } },
        lastProjectRows: [{ id: 'nova', chat_id: 7302, visible_revision: 4, _unread: true }],
        updateUnreadBadge() {}, paintProjectsNav() {}, patchProjectActivityMarkers() {}, acknowledgeMainAfterPaint() {},
        fetchJson: (url, init = {}) => new Promise((resolve) => {
            assert.equal(url, '/api/ui/preferences');
            assert.equal(init.method || 'GET', 'GET');
            answers.push(() => resolve(JSON.parse(JSON.stringify(host))));  // the host's state when it answers
        }),
    });
    vm.runInContext(['mainUnreadCount', 'observeMainRevision', 'mergeMainSeenRevision', 'mergeProjectSeenRevisions',
        'refreshSharedProjectSeen', 'paintProjectsCount'].map(lift).join('\n')
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

test("a snapshot showing only a Project unread reads another client's record as well", async () => {
    const host = { main_seen_revision: 90, project_seen_revision: { vega: 2 }, read_ahead_revisions: { 7303: [[5, 5]] } };
    const answers = [];
    const count = { textContent: '', title: '', attrs: {}, setAttribute() {}, removeAttribute() {} };
    const context = vm.createContext({
        ...readAhead,
        navProjectsCount: count,
        navState: { activeProjectId: null },
        state: { mainVisibleRevision: 90, mainSeenRevision: 90, mainSeenLoaded: true, activePage: 'files',
            projectSeenRevision: { vega: 2 } },
        closeProjectPanel() {}, patchProjectActivityMarkers() {}, syncNavigationState() {}, acknowledgeProjectAfterPaint() {},
        updateUnreadBadge() {},
        fetchJson: () => new Promise((resolve) => { answers.push(() => resolve(JSON.parse(JSON.stringify(host)))); }),
    });
    vm.runInContext(['renderProjectsNav', 'mergeMainSeenRevision', 'mergeProjectSeenRevisions', 'refreshSharedProjectSeen',
        'paintProjectsCount'].map(lift).join('\n')
        + '\nlet sharedSeenRead = null, knownProjectsJson = null, lastProjectRows = null, projectActivityIndex = {};'
        + '\nfunction paintProjectsNav() { paintProjectsCount(); }\nglobalThis.api = { renderProjectsNav };', context);
    context.api.renderProjectsNav([{ id: 'vega', chat_id: 7303, visible_revision: 5, lifecycle: 'active' }], [7303]);
    assert.equal(count.textContent, '3');  // 3, 4 and 5: the other client read 5 in Main, this one does not know yet
    assert.equal(answers.length, 1, 'Main shows nothing unread: the Project row alone reads the shared cursors again');
    answers.shift()();
    await flush();
    assert.equal(count.textContent, '2');
});

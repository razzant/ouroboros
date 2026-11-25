// The pager's frozen continuation and the read receipt's newest admitted target
// may belong to different snapshots. Exercise their real createChatInstance seam.
import assert from 'node:assert/strict';
import test from 'node:test';
import { ElementStub, PREVIEW_SCOPE } from './chat_dom_fixture.js';
import { row, room, until, quiet } from './project_read_state_fixture.js';

const note = n => ({ ...row(n), role: 'user', text: `owner note ${n}` });
const coverage = (upper, from, { view = 'room', chain = 'chat-file' } = {}) => ({ v: 1, view, upper: { chat: upper, progress: 0 },
    spans: { chat: { from, to: upper, chain, gaps: [] },
        progress: { from: 0, to: 0, chain: 'empty', gaps: [] } } });
const named = id => ({ complete: false, truncated_by: ['quota'],
    latest_message: { history_id: `chat:${id}`, out_of_order: false } });
// history_paging.latest_arrival: a recent read whose bounded search ran out at
// `before` names nothing; its quiet chain's page holding the target names it.
const searchedWindow = before => ({ complete: false, truncated_by: ['quota'], latest_message: null, latest_before: before });
const snapshot = (name, upper, target, { before = null, unknown = false, ...bytes } = {}) => {
    const rows = [note(upper - 80), note(upper - 70)];
    const quiet = before !== null;
    const pages = Array.from({ length: 6 }, (_, i) => ({
        messages: i === 5 ? [row(target)] : [note(upper - 200 - i * 100)],
        coverage: coverage(upper, i === 5 ? 0 : upper - 200 - i * 100, bytes),
        window: quiet && i === 5 ? { latest_message: { history_id: `chat:${target}`, out_of_order: true } } : {},
    }));
    const window = unknown ? { complete: false, truncated_by: ['quota'], latest_message: null }
        : quiet ? searchedWindow(before) : named(target);
    return { name, target: `chat:${target}`, recent: { messages: rows, window, coverage: coverage(upper, upper - 100, bytes) }, pages };
};

function raceRoom(t, { old = snapshot('old', 1000, 9), fresh = snapshot('fresh', 2000, 1100) } = {}) {
    const snapshots = { old, fresh };
    const server = { current: snapshots.old, recentGates: [], pageGates: new Map(), reads: [], pages: [] };
    const deferred = () => { let release; const promise = new Promise(resolve => { release = resolve; }); return { promise, release }; };
    const holdPage = key => { const gate = deferred(); server.pageGates.set(key, gate.promise); return gate.release; };
    const holdRecent = () => { const gate = deferred(); server.recentGates.push(gate.promise); return gate.release; };
    let arrivals = 0;
    const r = room(t, { projectId: 'snapshot-race', onReadingLatest: () => { arrivals += 1; }, fetchHistory: async url => {
        const cursor = new URL(url, 'http://local').searchParams.get('cursor');
        let data, gate;
        if (cursor) {
            const [name, number] = cursor.split(':'), index = Number(number);
            const frozen = snapshots[name];
            assert.ok(frozen, `cursor belongs to an immutable snapshot: ${cursor}`);
            server.pages.push(cursor);
            data = { ...frozen.pages[index - 1], page_cursor: cursor,
                next_cursor: index < frozen.pages.length ? `${name}:${index + 1}` : null,
                has_more: index < frozen.pages.length };
            gate = server.pageGates.get(cursor);
        } else {
            const fresh = server.current;
            server.reads.push(fresh.name);
            data = { ...fresh.recent, page_cursor: `${fresh.name}:0`, next_cursor: `${fresh.name}:1`, has_more: true };
            gate = server.recentGates.shift();
        }
        await gate;
        return { ok: true, json: async () => ({ profile_scope: PREVIEW_SCOPE, ...data }) };
    } });
    const rect = ElementStub.prototype.getBoundingClientRect;
    ElementStub.prototype.getBoundingClientRect = function () {
        const id = this.dataset?.historyId;
        if (Object.values(snapshots).some(frozen => frozen.target === id)) {
            return r.box(-r.messages.scrollTop, 20 - r.messages.scrollTop);
        }
        if (id) return r.box(-500, -480); // walked pages are off screen and can be evicted
        return rect.call(this);
    };
    t.after(() => { ElementStub.prototype.getBoundingClientRect = rect; });
    const open = () => {
        void r.instance.showLatest({ arrive: true });
        r.instance.cancelHistoryPaint();
        return r.instance.refreshHistory({ revision: 4 });
    };
    const newer = async (next = snapshots.fresh, revision = 5) => {
        server.current = next;
        const receipt = await r.instance.refreshHistory({ revision });
        assert.notEqual(receipt.read, true, 'the newly admitted target is offpage, so its revision cannot be read');
    };
    return { r, server, snapshots, holdPage, holdRecent, open, newer, arrivals: () => arrivals,
        trace: () => ({ reads: server.reads, pages: server.pages, shown: r.shown(),
            scrollTop: r.messages.scrollTop, arrivals }) };
}

test('an opening rebases its frozen cursor when a later recent read names a different offpage target', async t => {
    const f = raceRoom(t), releaseOld = f.holdPage('old:5'), releaseFresh = f.holdPage('fresh:2');
    await f.open();
    await until(() => f.server.pages.includes('old:5'));
    assert.ok(!f.r.shown().includes('chat:800'), 'the walk has passed pages the pager keeps');
    await f.newer();
    releaseOld();
    await quiet();
    t.diagnostic(JSON.stringify(f.trace()));
    assert.ok(f.server.pages.includes('fresh:2'), 'the opening must leave the old frozen snapshot and reach the admitted target');
    assert.deepEqual(f.server.pages, ['old:1', 'old:2', 'old:3', 'old:4', 'old:5', 'fresh:1', 'fresh:2'],
        'no stale old-chain continuation after the in-flight page settles');
    // More traffic can grow the room without replacing the named target.
    f.server.current = f.snapshots.poll = snapshot('poll', 3000, 1100);
    await f.r.instance.refreshHistory({ revision: 6 });
    releaseFresh();
    await until(() => f.r.shown().includes('chat:1100') && f.r.messages.scrollTop === 0);
    assert.deepEqual(f.server.pages, ['old:1', 'old:2', 'old:3', 'old:4', 'old:5', ...[1, 2, 3, 4, 5, 6].map(i => `fresh:${i}`)]);
    assert.deepEqual(f.server.reads, ['old', 'fresh', 'fresh', 'poll'], 'one rebase; a same-target poll does not restart the walk');
    assert.equal(f.arrivals(), 1, 'the owned opening lands once');
    assert.equal((await f.r.instance.refreshHistory({ revision: 6 })).read, true, 'only the visible target can be read');
    const pages = [...f.server.pages], reads = f.server.reads.length;
    f.r.readLatest();
    const top = f.r.messages.scrollTop;
    f.server.current = f.snapshots.later = snapshot('later', 4000, 3999);
    assert.equal((await f.r.instance.refreshHistory({ revision: 7 })).read, false);
    await quiet();
    assert.deepEqual(f.server.pages, pages, 'a later arrival inherits no opening and starts no page or return loop');
    assert.equal(f.server.reads.length, reads + 1);
    assert.equal(f.r.messages.scrollTop, top, 'a later arrival cannot jump the reader');
    t.diagnostic(JSON.stringify(f.trace()));
});

test('a rebase response superseded in flight cannot lend its cursor to the newest admitted target', async t => {
    const f = raceRoom(t), releaseOld = f.holdPage('old:5');
    await f.open();
    await until(() => f.server.pages.includes('old:5'));
    await f.newer();
    const releaseRebase = f.holdRecent();
    releaseOld();
    await quiet();
    assert.deepEqual(f.server.reads, ['old', 'fresh', 'fresh'], 'the rebase is in flight on fresh snapshot');
    f.server.current = f.snapshots.newest = snapshot('newest', 3000, 2100);
    assert.equal((await f.r.instance.refreshHistory({ revision: 6 })).read, false, 'an even newer target is admitted offpage');
    releaseRebase();
    await until(() => f.r.shown().includes('chat:2100') && f.r.messages.scrollTop === 0);
    assert.deepEqual(f.server.reads, ['old', 'fresh', 'fresh', 'newest', 'newest'], 'the still-owned opening rebases for each changed target');
    assert.deepEqual(f.server.pages, ['old:1', 'old:2', 'old:3', 'old:4', 'old:5', ...[1, 2, 3, 4, 5, 6].map(i => `newest:${i}`)],
        'neither superseded chain can continue after the latest target is admitted');
    assert.equal(f.arrivals(), 1);
    assert.equal((await f.r.instance.refreshHistory({ revision: 6 })).read, true);
    t.diagnostic(JSON.stringify(f.trace()));
});

const chain = (name, length) => Array.from({ length }, (_, i) => `${name}:${i + 1}`);
// The opening walks the old chain to its fifth page; `next` is admitted while that page is held.
async function admitDuringWalk(f, next, revision = 5) {
    const releaseOld = f.holdPage('old:5');
    await f.open();
    await until(() => f.server.pages.includes('old:5'));
    await f.newer(next, revision);
    releaseOld();
    await quiet();
}
const landedOn = (f, id) => until(() => f.r.shown().includes(id) && f.r.messages.scrollTop === 0);

test('a reply past the bounded search replaces the frozen named target: the opening rebases before walking on', async t => {
    // The retained Chromium counterexample: the frozen chain named the old reply; the fresh
    // read's search ran out above it (latest_before) and named nothing.
    const f = raceRoom(t, { fresh: snapshot('fresh', 2000, 1100, { before: 1500 }) });
    await admitDuringWalk(f, f.snapshots.fresh);
    await landedOn(f, 'chat:1100');
    t.diagnostic(JSON.stringify(f.trace()));
    assert.deepEqual(f.server.reads, ['old', 'fresh', 'fresh'], 'one return to the searched snapshot');
    assert.deepEqual(f.server.pages, [...chain('old', 5), ...chain('fresh', 6)], 'its quiet chain carries the search to the reply');
    assert.equal(f.arrivals(), 1);
    assert.equal((await f.r.instance.refreshHistory({ revision: 5 })).read, true);
});

test('quiet growth whose search joins the frozen quiet chain keeps walking it', async t => {
    const f = raceRoom(t, { old: snapshot('old', 1000, 9, { before: 500 }) });
    f.snapshots.poll = snapshot('poll', 1500, 9, { before: 900 });
    await admitDuringWalk(f, f.snapshots.poll);
    await landedOn(f, 'chat:9');
    t.diagnostic(JSON.stringify(f.trace()));
    assert.deepEqual(f.server.reads, ['old', 'poll'], 'no rebase: nothing between the two searches is unsearched');
    assert.deepEqual(f.server.pages, chain('old', 6));
    assert.equal(f.arrivals(), 1);
    assert.equal((await f.r.instance.refreshHistory({ revision: 5 })).read, true);
});

test('growth admitted while the first walked page is in flight restarts nothing: the walk goes on', async t => {
    const f = raceRoom(t);
    f.snapshots.poll = snapshot('poll', 1500, 9);
    const release = f.holdPage('old:1');
    await f.open();
    await until(() => f.server.pages.includes('old:1'));
    await f.newer(f.snapshots.poll);
    release();
    await landedOn(f, 'chat:9');
    t.diagnostic(JSON.stringify(f.trace()));
    assert.deepEqual(f.server.reads, ['old', 'poll'], 'the newest page is still drawn, yet the chain is not re-anchored mid-walk');
    assert.deepEqual(f.server.pages, chain('old', 6));
    assert.equal(f.arrivals(), 1);
});

test('quiet growth searched only above the frozen chain rebases: the bytes between are unsearched', async t => {
    const f = raceRoom(t, { old: snapshot('old', 1000, 9, { before: 500 }), fresh: snapshot('fresh', 2000, 1100, { before: 1500 }) });
    await admitDuringWalk(f, f.snapshots.fresh);
    await landedOn(f, 'chat:1100');
    t.diagnostic(JSON.stringify(f.trace()));
    assert.deepEqual(f.server.reads, ['old', 'fresh', 'fresh']);
    assert.deepEqual(f.server.pages, [...chain('old', 5), ...chain('fresh', 6)]);
    assert.equal(f.arrivals(), 1);
});

// A history_id and a search bound are byte offsets: another room view or a replaced
// chain makes them other bytes, even where the same id is named.
for (const [change, bytes] of [['room view changed', { view: 'room-2' }], ['chain replaced', { chain: 'replaced' }]]) {
    for (const kind of ['named', 'searched']) {
        test(`a ${kind} target on a ${change} is not the frozen chain's: the opening rebases`, async t => {
            const f = kind === 'named'
                ? raceRoom(t, { fresh: snapshot('fresh', 2000, 9, bytes) })
                : raceRoom(t, { old: snapshot('old', 1000, 9, { before: 500 }), fresh: snapshot('fresh', 1500, 9, { before: 900, ...bytes }) });
            await admitDuringWalk(f, f.snapshots.fresh);
            await landedOn(f, 'chat:9');
            t.diagnostic(JSON.stringify(f.trace()));
            assert.deepEqual(f.server.reads, ['old', 'fresh', 'fresh']);
            assert.deepEqual(f.server.pages, [...chain('old', 5), ...chain('fresh', 6)], 'no old-chain page after the evidence changed');
            assert.equal(f.arrivals(), 1);
        });
    }
}

test('a newest arrival turned unknown mid-walk ends the opening: no search, no landing, no later jump', async t => {
    const f = raceRoom(t, { fresh: snapshot('fresh', 2000, 1100, { unknown: true }) });
    await admitDuringWalk(f, f.snapshots.fresh);
    t.diagnostic(JSON.stringify(f.trace()));
    assert.deepEqual(f.server.reads, ['old', 'fresh']);
    assert.deepEqual(f.server.pages, chain('old', 5), 'an unknown target is never searched for');
    assert.equal(f.arrivals(), 0, 'nothing is landed on');
    // The source heals and names the reply: the spent opening neither pages nor lands
    // (a reader left following the bottom still follows appended rows; that is no landing).
    f.server.current = f.snapshots.healed = snapshot('healed', 2100, 1100);
    assert.equal((await f.r.instance.refreshHistory({ revision: 6 })).read, false);
    await quiet();
    t.diagnostic(JSON.stringify(f.trace()));
    assert.deepEqual(f.server.pages, chain('old', 5));
    assert.deepEqual(f.server.reads, ['old', 'fresh', 'healed']);
    assert.ok(!f.r.shown().includes('chat:1100') && f.arrivals() === 0);
});

for (const kind of ['named', 'searched']) for (const cancel of ['gesture', 'question', 'hidden']) {
    test(`a ${cancel} during ${kind} snapshot rebasing ends the same opening before any new-chain page`, async t => {
        const f = raceRoom(t, kind === 'searched' ? { fresh: snapshot('fresh', 2000, 1100, { before: 1500 }) } : {});
        const releaseOld = f.holdPage('old:5');
        await f.open();
        await until(() => f.server.pages.includes('old:5'));
        await f.newer();
        const releaseRebase = f.holdRecent();
        releaseOld();
        await quiet();
        assert.equal(f.server.reads.length, 3, 'the next recent read is the pending snapshot rebase');
        if (cancel === 'gesture') {
            for (const handler of f.r.messages.listeners.get('wheel') || []) {
                handler({ type: 'wheel', deltaY: 1, target: f.r.messages, timeStamp: 0 });
            }
        } else if (cancel === 'question') {
            globalThis.document.body = globalThis.document.createElement('body');
            void f.r.instance.revealQuestion('explicit-task', 'explicit-question');
        } else f.r.instance.page.hidden = true;
        const top = f.r.messages.scrollTop;
        releaseRebase();
        await quiet();
        assert.deepEqual(f.server.pages, ['old:1', 'old:2', 'old:3', 'old:4', 'old:5'], 'neither frozen chain continues after cancellation');
        assert.equal(f.arrivals(), 0);
        assert.equal(f.r.messages.scrollTop, top);
        f.r.instance.page.hidden = false;
        assert.notEqual((await f.r.instance.refreshHistory({ revision: 6 })).read, true);
        await quiet();
        assert.deepEqual(f.server.pages, ['old:1', 'old:2', 'old:3', 'old:4', 'old:5'], 'showing or refreshing later cannot revive the spent opening');
        assert.equal(f.r.messages.scrollTop, top);
        t.diagnostic(JSON.stringify(f.trace()));
    });
}

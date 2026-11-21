import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatHistoryPager } from '../modules/chat_history.js';

function page(index, { last = 4, messages, chain = 'A' } = {}) {
    return {
        messages: messages || [{ history_id: `chat:${index}`, text: `Page ${chain}-${index}` }],
        page_cursor: `replay:${chain}:${index}`,
        next_cursor: index < last ? `before:${chain}:${index + 1}` : null,
        has_more: index < last,
        window: { complete: false },
    };
}

function harness({ fetch, maxPages = 3 } = {}) {
    const calls = [], applied = [], released = [], states = [], protectedIds = new Set();
    let live = true;
    const pager = createChatHistoryPager({
        fetchPage: (cursor, options) => {
            calls.push({ cursor, ...options });
            return fetch ? fetch(cursor, options)
                : page(Number(cursor.split(':').at(-1)), { chain: cursor.split(':')[1] });
        },
        applyPage: (messages, descriptor) => applied.push({ messages, ...descriptor }),
        releasePage: descriptor => released.push(descriptor),
        isPageProtected: descriptor => protectedIds.has(descriptor.id),
        isAlive: () => live,
        onState: state => states.push(state),
        maxPages,
    });
    return { pager, calls, applied, released, states, protectedIds, setAlive: value => { live = value; } };
}

for (const jump of ['around', 'revisit']) test(`partial latest rechains a ${jump} to present-based older paging`, async () => {
    const partial = { ...page(0, { chain: 'present' }),
        window: { latest_message: null, truncated_by: ['chat_projection_failed'] } };
    const h = harness({ fetch: (cursor, options) => options.around || cursor === 'jump'
        ? page(3, { chain: 'jump' }) : cursor ? page(1, { chain: 'present' }) : partial });
    h.pager.acceptRecent(page(0));
    await (jump === 'around' ? h.pager.around('task:old', () => true) : h.pager.revisit('jump', () => true));
    assert.equal(h.pager.getState().away, true);
    assert.equal((await h.pager.newer()).status, 'unavailable', 'a jumped chain cannot walk newer');
    await h.pager.latest(() => true);
    assert.equal(h.pager.getState().away, false);
    assert.equal(h.pager.getState().firstPage.requestCursor, partial.page_cursor);
    assert.equal(h.pager.getState().error, null, 'the partial response remains a useful page');
    await h.pager.older();
    assert.equal(h.calls.at(-1).cursor, 'before:present:1');
});

test('first recent read owns the frozen chain; a later live refresh does not replace it', () => {
    const h = harness();
    const recent = page(0);
    assert.equal(h.pager.acceptRecent(recent).status, 'applied');
    assert.equal(h.applied.length, 1);
    assert.equal(h.applied[0].direction, 'recent');
    assert.equal(h.applied[0].messages, recent.messages);
    assert.equal(h.pager.acceptRecent(page(0, { chain: 'new-live' })).status, 'ignored');
    const state = h.pager.getState();
    assert.equal(h.applied.length, 1);
    assert.equal(state.firstPage.requestCursor, recent.page_cursor);
    assert.equal(state.firstPage.nextCursor, recent.next_cursor);
    assert.equal(state.canOlder, true);
    assert.equal(state.canNewer, false);
    assert.equal(state.cachedPages.length, 1);
    assert.doesNotMatch(JSON.stringify(state), /Page A-0/, 'state/descriptors do not retain message bodies');
});

test('a successful sparse page advances its cursor even with no visible messages', async () => {
    const h = harness({ fetch: cursor => cursor === 'before:A:1'
        ? page(1, { messages: [] }) : page(2, { last: 2 }) });
    h.pager.acceptRecent(page(0));
    assert.equal((await h.pager.older()).status, 'applied');
    assert.deepEqual(h.applied.at(-1).messages, []);
    assert.equal(h.pager.getState().olderExhausted, false);
    await h.pager.older();
    assert.deepEqual(h.calls.map(call => call.cursor), ['before:A:1', 'before:A:2']);
    assert.equal(h.pager.getState().olderExhausted, true);
    assert.equal((await h.pager.older()).status, 'unavailable');
    assert.equal(h.calls.length, 2);
});

test('empty pages are walked over, never becoming a reading position or a newer gap', async () => {
    const h = harness({ fetch: cursor => {
        const index = Number(cursor.split(':').at(-1));
        return index < 4 ? page(index, { messages: [] }) : page(4);
    } });
    h.pager.acceptRecent(page(0));
    for (let n = 0; n < 3; n += 1) {
        await h.pager.older();
        const step = h.pager.getState();
        assert.equal(step.canNewer, false, 'an empty page above the window is not a gap');
        assert.equal(step.olderExhausted, false);
    }
    assert.deepEqual(h.pager.getState().cachedPages.map(item => item.index), [0],
        'an empty page leaves the cache at once: its descriptor alone holds the window');
    assert.deepEqual(h.released.map(item => item.index), [1, 2, 3]);
    await h.pager.older();
    const state = h.pager.getState();
    assert.deepEqual(state.cachedPages.map(item => [item.index, item.rows]), [[0, 1], [4, 1]]);
    assert.equal(state.pageCount, 5, 'walked-over empty pages keep their descriptors');
    assert.equal(state.canNewer, false);
    assert.equal(state.olderExhausted, true);
    const spent = h.calls.length;
    assert.equal((await h.pager.newer()).status, 'unavailable');
    assert.equal(h.calls.length, spent, 'walking back across empty pages costs no request');
});

test('a re-read page keeps its frozen boundaries while its row count is refreshed', async () => {
    let zero = false;
    const h = harness({ fetch: cursor => cursor === 'replay:A:0'
        ? page(0, { messages: zero ? [] : undefined })
        : page(Number(cursor.split(':').at(-1))) });
    h.pager.acceptRecent(page(0));
    for (let n = 0; n < 3; n += 1) await h.pager.older();
    assert.equal(h.pager.getState().canNewer, true, 'a released NON-empty page is still a real gap');
    zero = true;
    await h.pager.newer();
    // The refreshed descriptor lives in the page list; its now-empty cache entry
    // is released at once, like any other empty page.
    assert.deepEqual({ ...h.pager.getState().firstPage },
        { id: 'history-page-1-0', chain: 1, index: 0, requestCursor: 'replay:A:0',
            nextCursor: 'before:A:1', hasMore: true, rows: 0, loaded: true, coverage: null });
    assert.equal(h.pager.getState().cachedPages.some(item => item.index === 0), false);
    assert.equal(h.pager.getState().pageCount, 4, 'a refreshed row count mints no extra descriptor');
    assert.equal(h.pager.getState().canNewer, false);
});

test('evicted newer and older pages replay their exact handles, including page zero', async () => {
    const h = harness();
    h.pager.acceptRecent(page(0));
    for (let n = 0; n < 3; n += 1) await h.pager.older();
    let state = h.pager.getState();
    assert.equal(state.cachedPages.length, 3);
    assert.deepEqual(state.cachedPages.map(item => item.index), [1, 2, 3]);
    assert.equal(state.canNewer, true);
    assert.equal(h.released[0].index, 0);
    await h.pager.newer();
    assert.equal(h.calls.at(-1).cursor, 'replay:A:0');
    state = h.pager.getState();
    assert.deepEqual(state.cachedPages.map(item => item.index).sort(), [0, 1, 2]);
    assert.equal(state.canNewer, false);
    await h.pager.older();
    assert.equal(h.calls.at(-1).cursor, 'replay:A:3');
    assert.equal(h.pager.getState().pageCount, 4, 'rereading does not mint another descriptor');
});

test('a protected page stays mounted without blocking navigation across evicted gaps', async () => {
    const h = harness();
    const initial = h.pager.acceptRecent(page(0));
    h.protectedIds.add(initial.page.id);
    for (let n = 0; n < 3; n += 1) await h.pager.older();
    assert.deepEqual(h.pager.getState().cachedPages.map(item => item.index), [0, 2, 3]);
    assert.equal(h.pager.getState().firstPage.index, 2);
    await h.pager.newer();
    assert.equal(h.calls.at(-1).cursor, 'replay:A:1');
    assert.ok(!h.released.some(item => item.id === initial.page.id));
    assert.deepEqual(h.pager.getState().cachedPages.map(item => item.index).sort(), [0, 1, 2]);
});

test('selection spanning pages may exceed the cache until explicit unpin and trim', async () => {
    const h = harness();
    h.protectedIds.add(h.pager.acceptRecent(page(0)).page.id);
    for (let n = 0; n < 4; n += 1) {
        const next = h.pager.getState().pageCount;
        h.protectedIds.add(`history-page-1-${next}`);
        await h.pager.older();
    }
    assert.equal(h.pager.getState().cachedPages.length, 5);
    assert.equal(h.released.length, 0);
    h.protectedIds.clear();
    assert.equal(h.pager.trim().length, 2);
    assert.deepEqual(h.pager.getState().cachedPages.map(item => item.index), [2, 3, 4]);
});

test('a failed read keeps content and its exact cursor for explicit retry', async () => {
    const error = Object.assign(new Error('The Project binding changed'), {
        status: 409, payload: { reason_code: 'history_view_changed' },
    });
    let fail = true;
    const h = harness({ fetch: () => { if (fail) throw error; return page(1); } });
    h.pager.acceptRecent(page(0));
    const result = await h.pager.older();
    assert.equal(result.error, error);
    assert.equal(result.status, 'error');
    assert.equal(h.pager.getState().error, error);
    assert.equal(h.pager.getState().retryDirection, 'older');
    assert.equal(h.pager.getState().retryCursor, 'before:A:1');
    assert.equal(h.pager.getState().loading, '');
    assert.equal(h.applied.length, 1);
    assert.equal(h.released.length, 0);
    h.pager.acceptRecent(page(0, { chain: 'different-live-snapshot' }));
    fail = false;
    assert.equal((await h.pager.retry()).status, 'applied');
    assert.deepEqual(h.calls.map(call => call.cursor), ['before:A:1', 'before:A:1']);
    assert.equal(h.pager.getState().error, null);
});

test('missing continuation facts fail visibly instead of inventing EOF', async () => {
    const h = harness({ fetch: () => ({ messages: [], page_cursor: 'replay:A:1' }) });
    h.pager.acceptRecent(page(0));
    assert.equal((await h.pager.older()).status, 'error');
    assert.equal(h.pager.getState().olderExhausted, false);
    assert.equal(h.pager.getState().retryCursor, 'before:A:1');
    assert.equal(h.applied.length, 1);
});

test('one in-flight navigation coalesces repeats and reports its loading direction', async () => {
    let resolve;
    const h = harness({ fetch: () => new Promise(done => { resolve = done; }) });
    h.pager.acceptRecent(page(0));
    const first = h.pager.older();
    assert.equal(h.pager.older(), first);
    assert.equal(h.pager.getState().loading, 'older');
    assert.equal((await h.pager.latest()).status, 'busy');
    await Promise.resolve();
    assert.equal(h.calls.length, 1);
    resolve(page(1));
    await first;
    assert.equal(h.pager.getState().loading, '');
    assert.equal(h.applied.length, 2);
});

test('latest starts a new frozen chain after success, retaining old protected content until unpin', async () => {
    const h = harness({ fetch: cursor => cursor === null ? page(0, { chain: 'B' })
        : page(1, { chain: cursor.split(':')[1] }) });
    const initial = h.pager.acceptRecent(page(0));
    await h.pager.older();
    h.protectedIds.add(initial.page.id);
    const latest = await h.pager.latest();
    assert.equal(h.calls.at(-1).cursor, null);
    assert.equal(latest.page.requestCursor, 'replay:B:0');
    assert.equal(h.applied.at(-1).direction, 'latest');
    assert.equal(h.pager.getState().pageCount, 1);
    assert.deepEqual(h.pager.getState().cachedPages.map(item => item.chain), [1, 2]);
    assert.ok(h.released.some(item => item.index === 1 && item.chain === 1));
    h.protectedIds.clear();
    assert.deepEqual(h.pager.trim(), [initial.page.id]);
    await h.pager.older();
    assert.equal(h.calls.at(-1).cursor, 'before:B:1');
});

test('a protected old page stays mounted after latest rebases the chain', async () => {
    const h = harness({ fetch: cursor => cursor === null ? page(0, { chain: 'B' })
        : page(Number(cursor.split(':').at(-1)), { chain: cursor.split(':')[1] }) });
    h.pager.acceptRecent(page(0));
    const old = await h.pager.older();
    h.protectedIds.add(old.page.id);
    await h.pager.latest();
    assert.equal(h.pager.getState().firstPage.requestCursor, 'replay:B:0');
    assert.ok(h.pager.getState().cachedPages.some(item => item.id === old.page.id),
        'the reader still sees it until they leave it');
});

test('a failed latest read leaves the old chain available and retries the same request', async () => {
    let fail = true;
    const error = new Error('History source unavailable');
    const h = harness({ fetch: () => { if (fail) throw error; return page(0, { chain: 'B' }); } });
    const initial = h.pager.acceptRecent(page(0));
    await h.pager.latest();
    assert.equal(h.pager.getState().firstPage.id, initial.page.id);
    assert.equal(h.released.length, 0);
    assert.equal(h.pager.getState().retryDirection, 'latest');
    fail = false;
    await h.pager.retry();
    assert.deepEqual(h.calls.map(call => call.cursor), [null, null]);
    assert.equal(h.pager.getState().firstPage.requestCursor, 'replay:B:0');
});

// ↓ returns to the present (DESIGN "History edges"): its `latest` carries its claim.
test('↓ makes an older page still loading yield before it lands: the window stays at the present it reads', async () => {
    const answers = new Map();
    const h = harness({ fetch: cursor => new Promise(resolve => answers.set(cursor, resolve)) });
    h.pager.acceptRecent(page(0));
    const older = h.pager.older();
    await Promise.resolve();
    const latest = h.pager.latest(() => true);
    assert.equal(h.calls[0].signal.aborted, true, 'the older read is stopped at once');
    assert.equal(h.pager.getState().loading, 'latest');
    await Promise.resolve();
    answers.get('before:A:1')(page(1));
    answers.get(null)(page(0, { chain: 'B' }));
    assert.equal((await older).status, 'disposed', 'a late older page never lands');
    assert.equal((await latest).status, 'applied');
    assert.deepEqual(h.applied.map(item => item.direction), ['recent', 'latest']);
    const state = h.pager.getState();
    assert.equal(state.away, false);
    assert.equal(state.firstPage.requestCursor, 'replay:B:0');
    h.pager.toPresent();
    assert.equal(h.pager.getState().loading, '', 'with nothing loading there is nothing to stop');
});

test('a read of the present already out serves every later asker: it lands while its latest claim holds', async () => {
    const answers = [];
    const h = harness({ fetch: () => new Promise(resolve => answers.push(resolve)) });
    h.pager.acceptRecent(page(0));
    let first = true, second = true;
    const one = h.pager.latest(() => first);
    await Promise.resolve();
    const two = h.pager.latest(() => second);
    assert.equal(two, one, 'a second ↓ joins the read already out');
    first = false;
    answers[0](page(0, { chain: 'B' }));
    assert.equal((await two).status, 'applied', 'the newer claim still holds: it lands');
    assert.equal(h.calls.length, 1);
    // A claim the reader left drops the answer, unless an asker without a claim (Refresh history) joined it.
    second = false;
    const three = h.pager.latest(() => second);
    await Promise.resolve();
    assert.equal(h.pager.latest(), three);
    answers[1](page(0, { chain: 'C' }));
    assert.equal((await three).status, 'applied');
    assert.equal(h.pager.getState().firstPage.requestCursor, 'replay:C:0');
    const four = h.pager.latest(() => second);
    await Promise.resolve();
    answers[2](page(0, { chain: 'D' }));
    assert.equal((await four).status, 'superseded', 'one the reader moved on from never replaces their chain');
    assert.equal(h.pager.getState().firstPage.requestCursor, 'replay:C:0');
    assert.equal(h.calls.length, 3);
});

test('a failed read of the present is the window\'s Retry while ↓ wants it; one the reader left leaves no failure', async () => {
    let wanted = true;
    const h = harness({ fetch: () => { throw new TypeError('offline'); } });
    h.pager.acceptRecent(page(0));
    assert.equal((await h.pager.latest(() => wanted)).status, 'error');
    assert.equal(h.pager.getState().retryDirection, 'latest');
    wanted = false;
    assert.equal((await h.pager.latest(() => wanted)).status, 'error');
    assert.equal(h.pager.getState().retryDirection, '', 'the reader moved on: nothing to retry');
});

test('only ↓\'s own failed read of the present is retried as ↓; an asker without a claim retries where it is', async () => {
    const h = harness({ fetch: () => { throw new TypeError('offline'); } });
    h.pager.acceptRecent(page(0));
    const retry = () => [h.pager.getState().retryDirection, h.pager.getState().retryReturn];
    assert.equal((await h.pager.latest(() => true)).status, 'error');
    assert.deepEqual(retry(), ['latest', true]);
    assert.equal((await h.pager.latest()).status, 'error');
    assert.deepEqual(retry(), ['latest', false], 'a refresh or a rebase reading the present is not the reader\'s ↓');
    assert.equal((await h.pager.older()).status, 'error');
    assert.deepEqual(retry(), ['older', false]);
});

test('↓ leaves a navigation\'s chain for the present: a navigation still loading yields, and the window is no longer away', async () => {
    const answers = [];
    const h = harness({ fetch: (cursor, { around }) => (around ? new Promise(resolve => answers.push(resolve))
        : page(0, { chain: 'B' })) });
    h.pager.acceptRecent(page(0));
    const navigation = h.pager.around('task:t-1', () => true);
    await Promise.resolve();
    assert.equal(h.pager.getState().loading, 'around');
    const latest = h.pager.latest(() => true);
    assert.equal(h.calls[0].signal.aborted, true);
    answers[0]({ ...page(5), anchor: { state: 'found' } });
    assert.equal((await navigation).status, 'disposed');
    assert.equal((await latest).status, 'applied');
    assert.equal(h.pager.getState().away, false);
    const jumped = harness({ fetch: (cursor, { around }) => (around ? { ...page(5), anchor: { state: 'found' } }
        : page(0, { chain: 'B' })) });
    jumped.pager.acceptRecent(page(0));
    assert.equal((await jumped.pager.around('task:t-1', () => true)).status, 'applied');
    assert.equal(jumped.pager.getState().away, true, 'a navigation began the chain away from the present');
    assert.equal((await jumped.pager.latest(() => true)).status, 'applied');
    assert.equal(jumped.pager.getState().away, false, '↓\'s latest begins the present\'s chain again');
});

test('suspension aborts a page without losing the chain, and an old answer cannot replace a new visit', async () => {
    const answers = [];
    const h = harness({ fetch: () => new Promise(resolve => answers.push(resolve)) });
    h.pager.acceptRecent(page(0));
    const old = h.pager.older();
    await Promise.resolve();
    h.pager.suspend();
    assert.equal(h.calls[0].signal.aborted, true);
    assert.equal(h.pager.getState().cachedPages.length, 1);
    assert.equal(h.released.length, 0);
    const next = h.pager.older();
    await Promise.resolve();
    answers[0](page(1, { chain: 'obsolete' }));
    assert.equal((await old).status, 'disposed');
    answers[1](page(1));
    assert.equal((await next).status, 'applied');
    assert.deepEqual(h.applied.map(p => p.messages[0].text), ['Page A-0', 'Page A-1']);
});

test('destroy aborts the read, releases cached pages once, and rejects late applies', async () => {
    let resolve;
    const h = harness({ fetch: () => new Promise(done => { resolve = done; }) });
    h.pager.acceptRecent(page(0));
    const loading = h.pager.older();
    await Promise.resolve();
    h.pager.destroy();
    h.pager.destroy();
    const states = h.states.length;
    assert.equal(h.calls[0].signal.aborted, true);
    assert.equal(h.released.length, 1);
    resolve(page(1));
    assert.equal((await loading).status, 'disposed');
    assert.equal(h.applied.length, 1);
    assert.equal(h.states.length, states);
    assert.deepEqual(h.pager.getState().cachedPages, []);
    assert.equal(h.pager.getState().initialized, false);
});

test('a dead chat instance cannot apply a completed read before explicit disposal', async () => {
    let resolve;
    const h = harness({ fetch: () => new Promise(done => { resolve = done; }) });
    h.pager.acceptRecent(page(0));
    const loading = h.pager.older();
    await Promise.resolve();
    h.setAlive(false);
    resolve(page(1));
    assert.equal((await loading).status, 'disposed');
    assert.equal(h.applied.length, 1);
    h.pager.destroy();
    assert.equal(h.released.length, 1);
});

for (const direction of ['recent', 'older', 'newer', 'around', 'revisit']) test(`partial projection remains a page: ${direction}`, async () => {
    const partial = data => ({ ...data, window: { latest_message: null, complete: false, truncated_by: ['chat_projection_failed'] } });
    let fault = direction !== 'newer';
    const h = harness({ fetch: cursor => { const data = page(Number(cursor?.split(':').at(-1)) || 0); return fault ? partial(data) : data; } });
    if (direction !== 'recent') h.pager.acceptRecent(page(0));
    if (direction === 'newer') { for (let n = 0; n < 3; n++) await h.pager.older(); fault = true; }
    const result = direction === 'recent' ? h.pager.acceptRecent(partial(page(0)))
        : direction === 'around' ? await h.pager.around('task:available', () => true)
        : direction === 'revisit' ? await h.pager.revisit('replay:A:1', () => true)
        : await h.pager[direction]();
    assert.equal(result.status, 'applied');
    assert.equal(h.pager.getState().error, null);
    assert.ok(h.applied.at(-1).window.truncated_by.includes('chat_projection_failed'));
    assert.ok(h.pager.getState().cachedPages.length <= 3);
});

import assert from 'node:assert/strict';
import test from 'node:test';
import { apiClient } from '../modules/api_client.js';
import { initStorage } from '../modules/storage.js';
import { setup } from './dashboard_dom_fixture.js';

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const settle = () => new Promise((resolve) => setImmediate(resolve));
const storageUrl = '/api/storage';
const storageVolume = { free_bytes: 170e9, total_bytes: 4e12, observed_at: '2026-10-05T10:00:00Z' };
const storageBody = (overrides = {}) => ({ count: null, counting: null, error: null, volume: storageVolume, retention_days: 7, ...overrides });
const deferred = () => {
    const handle = {};
    handle.promise = new Promise((resolve, reject) => Object.assign(handle, { resolve, reject }));
    return handle;
};

/** Timers by hand: the poll is a scheduled callback the test can see, fire or see cleared. */
function manualTimers(t) {
    const pending = new Map();
    let next = 1;
    const realClear = globalThis.clearTimeout;
    globalThis.setTimeout = (fn, ms) => { pending.set(next, { fn, ms }); return next++; };
    globalThis.clearTimeout = (id) => { pending.delete(id); };
    t.after(() => { globalThis.clearTimeout = realClear; });
    return {
        polls: () => [...pending.values()].filter((timer) => timer.ms === 1000).length,
        async fire() {
            // Only the poll: a toast's dismissal is not this flow's to fire.
            const due = [...pending].filter(([, timer]) => timer.ms === 1000);
            for (const [id] of due) pending.delete(id);
            for (const [, timer] of due) await timer.fn();
        },
    };
}

function openStorage(t) {
    const fixture = setup(t);
    const timers = manualTimers(t);
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'storage' };
    const show = (tab) => {
        state.dashboardActiveSubtab = tab;
        fixture.window.dispatchEvent({ type: 'ouro:dashboard-subtab-shown', detail: { tab } });
    };
    return { ...fixture, timers, state, show, start: () => initStorage({ state, mount: fixture.mount }) };
}

// ---------------------------------------------------------------------------
// Dashboard -> Storage -> Free temporary space: the flow's requests through the real controller.
// ---------------------------------------------------------------------------

const previewUrl = '/api/storage/free/preview';
const freeUrl = '/api/storage/free';
const freeCancelUrl = '/api/storage/free/cancel';
const itemsUrl = (view, { plan = 'p1', cursor = 0, limit = view === 'pool' ? 500 : 100, version = '' } = {}) => (
    `/api/storage/free/items?plan=${plan}&view=${view}&cursor=${cursor}&limit=${limit}${version ? `&version=${version}` : ''}`);
const freeGroup = (id, bytes) => ({ id, count: 3, inside_wait: 2, in_use: 1, unreadable: 0, bytes, bytes_complete: true,
    items: [{ id: `${id}:a`, name: 'Run the QA wave', bytes }] });
const poolEntry = (id, overrides = {}) => ({ id, group: 'work_folders', kind: 'work_folder', name: `Task ${id}`, path: `/data/${id}`, task_id: id, bytes: 1e9,
    bytes_complete: true, eligible_after: null, reason: null, outcome: null, ...overrides });
const poolFacts = (count, overrides = {}) => ({ count, bytes: count * 1e9, bytes_complete: true, kinds: { work_folder: count }, excluded: [], ...overrides });
const freeBody = (state, overrides = {}) => storageBody({ free: { state, plan: 'p1', started_at: '2026-10-05T11:00:00Z', finished_at: null,
    done: 0, total: 0, groups: [], pool: poolFacts(0), run: null, error: null, ...overrides } });
const readyBody = (count = 2, overrides = {}) => freeBody('ready', { pool: poolFacts(count), ...overrides });
const page = (view, items, overrides = {}) => response({ plan: 'p1', view, version: 'p1', total: items.length, counts: {}, next_cursor: null, items, ...overrides });
const freeRun = (overrides = {}) => ({ groups: [], selected: 2, done: 2, removed: 2, removed_bytes: 2e9, deleting: 0, deleting_bytes: 0, stuck: 0, failed: 0, not_started: 0,
    kept: [], free_before: 140e9, free_after: 142e9, stopped: false, error: null, outcomes_ready: true, ...overrides });
const storageTray = (mount) => mount.querySelector('[data-storage-tray]');
const trayButton = (mount, name) => storageTray(mount).querySelector(`[${name}]`);
const pressTray = (mount, name) => storageTray(mount).fire('click', { target: trayButton(mount, name) });
const toggleOpen = (mount, key) => {
    const details = storageTray(mount).querySelector(`[data-free-disclosure="${key}"]`);
    details.setAttribute('open', '');
    return storageTray(mount).fire('toggle', { target: details });
};

/** Open the tab on a ready plan whose pool the controller has read whole. */
async function openReady(t, items = [poolEntry('a'), poolEntry('b')], overrides = {}) {
    const opened = openStorage(t);
    opened.routes.set(storageUrl, response(readyBody(items.length, overrides)));
    opened.routes.set(itemsUrl('pool'), page('pool', items));
    opened.storage = opened.start();
    await settle();
    return opened;
}

test('Storage flow previews on its door, polls while listing, reads the whole pool once ready and enables Remove only then', async (t) => {
    const { mount, routes, calls, timers, start, document } = openStorage(t);
    routes.set(storageUrl, response(storageBody({ free: { state: 'idle', plan: null, started_at: null, finished_at: null, done: 0, total: 0, groups: [], pool: null, run: null, error: null } })));
    start();
    await settle();
    const door = mount.querySelector('[data-storage-door]');
    assert.equal(door.hidden, false);
    assert.equal(door.textContent, 'Free temporary space…');
    assert.equal(storageTray(mount).textContent, '', 'at rest the flow is empty');
    routes.set(previewUrl, (_url, init) => {
        assert.equal(init.method, 'POST');
        return response(freeBody('listing', { total: 2, pool: poolFacts(1, { bytes: null }), started: true }), 202);
    });
    door.focus();
    await door.fire('click', { target: door });
    await settle();
    assert.deepEqual(calls, [storageUrl, previewUrl]);
    assert.match(storageTray(mount).textContent, /Checking what can go · 0 of 2 folders/);
    assert.equal(door.hidden, true, 'no second start while it lists');
    assert.equal(mount.querySelector('[data-storage-action="measure"]').textContent, 'Measure now', 'a preview POST never reads as a count');
    assert.equal(document.activeElement?.getAttribute('data-key'), 'data-free-stop', 'focus follows the door into the listing');
    assert.equal(timers.polls(), 1, 'a listing is polled');

    // Three items over two pages: Remove waits for the last id, and the page in between claims no whole list.
    const held = deferred();
    routes.set(storageUrl, response(readyBody(3)));
    routes.set(itemsUrl('pool'), page('pool', [poolEntry('a'), poolEntry('b')], { total: 3, next_cursor: 2 }));
    routes.set(itemsUrl('pool', { cursor: 2, version: 'p1' }), () => held.promise);
    await timers.fire();
    await settle();
    assert.match(storageTray(mount).textContent, /3 items · 3\.0 GB counted[\s\S]*Reading the list · 2 of 3/);
    const waiting = trayButton(mount, 'data-free-go');
    assert.equal(waiting.getAttribute('aria-disabled'), 'true');
    assert.equal(document.activeElement?.getAttribute('data-key'), 'data-free-go', 'focus follows Stop to the action it waits for');
    let posted = 0;
    routes.set(freeUrl, () => { posted += 1; return response(readyBody(3)); });
    await storageTray(mount).fire('click', { target: waiting });
    await settle();
    assert.equal(posted, 0, 'a partial list sends nothing');
    assert.doesNotMatch(storageTray(mount).textContent, /Remove 3 items|Show all/);
    held.resolve(page('pool', [poolEntry('c')], { total: 3 }));
    await settle();
    assert.match(storageTray(mount).textContent, /Show all 3 and choose[\s\S]*Remove 3 items/);
    assert.equal(trayButton(mount, 'data-free-go').getAttribute('aria-disabled'), null);
    assert.equal(timers.polls(), 0, 'a ready list is not polled');
    assert.deepEqual(calls.slice(2), [storageUrl, itemsUrl('pool'), itemsUrl('pool', { cursor: 2, version: 'p1' })], 'each page once, the version on the next');

    // Cancel hides the list without a request and the door returns with the focus.
    const before = calls.length;
    trayButton(mount, 'data-free-cancel').focus();
    await pressTray(mount, 'data-free-cancel');
    assert.equal(storageTray(mount).textContent, '');
    assert.equal(door.hidden, false);
    assert.equal(document.activeElement, door, 'focus lands on the door');
    assert.equal(calls.length, before);
});

test('Storage flow sends the real ids of exactly what is chosen, across every page, then shows the result with no competing start', async (t) => {
    const items = Array.from({ length: 150 }, (_, index) => poolEntry(`i${index}`));
    const { mount, routes, calls, timers, show, document } = await openReady(t, items);
    const rows = () => storageTray(mount).querySelectorAll('[data-free-pick]');
    assert.equal(rows().length, 100, 'rows are shown a hundred at a time');
    assert.match(storageTray(mount).textContent, /150 of 150 chosen/);
    // A row not on screen stays chosen; clearing all reaches it too.
    const first = rows()[0];
    first.checked = false;
    await storageTray(mount).fire('change', { target: first });
    assert.match(storageTray(mount).textContent, /149 of 150 chosen[\s\S]*Remove 149 items/);
    await pressTray(mount, 'data-free-none');
    assert.match(storageTray(mount).textContent, /0 of 150 chosen[\s\S]*Nothing selected/);
    assert.equal(trayButton(mount, 'data-free-go'), null);
    await pressTray(mount, 'data-free-all');
    assert.match(storageTray(mount).textContent, /150 of 150 chosen[\s\S]*Remove 150 items/);
    await pressTray(mount, 'data-free-more-picks');
    assert.equal(rows().length, 150);
    const last = rows()[149];
    last.checked = false;
    await storageTray(mount).fire('change', { target: last });
    assert.match(storageTray(mount).textContent, /Remove 149 items/);

    let posted = null;
    routes.set(freeUrl, async (_url, init) => {
        posted = JSON.parse(init.body);
        return response(freeBody('running', { total: 150, pool: poolFacts(150), run: freeRun({ selected: 149, done: 2, removed: 0, removed_bytes: 0, deleting: 2, deleting_bytes: 2e9 }), started: true }), 202);
    });
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.equal(document.body.querySelector('[role="dialog"]'), null, 'the review was the one confirmation');
    assert.deepEqual(Object.keys(posted).sort(), ['items', 'plan']);
    assert.equal(posted.plan, 'p1');
    assert.deepEqual(posted.items, items.slice(0, 149).map((item) => item.id));
    assert.match(storageTray(mount).textContent, /Removing · 2 of 149 · 0 bytes of their counted size gone so far/);
    assert.match(storageTray(mount).textContent, /2 are still being deleted/);
    assert.equal(document.activeElement?.getAttribute('data-key'), 'data-free-stop', 'focus moves to the running flow\'s Stop');
    assert.equal(mount.querySelector('[data-storage-door]').hidden, true);
    assert.equal(timers.polls(), 1);

    const done = (run) => freeBody('done', { pool: poolFacts(150), run: freeRun({ selected: 149, done: 149, ...run }) });
    routes.set(storageUrl, response(done({ removed: 147, removed_bytes: 147e9, deleting: 1, deleting_bytes: 1e9, failed: 1 })));
    await timers.fire();
    await settle();
    const result = storageTray(mount).textContent;
    assert.ok(result.indexOf('1 could not be deleted.') < result.indexOf('Removed 147 items.'), 'what went wrong first');
    assert.match(result, /Their counted size was 147 GB\. Free now: 170 GB, read [^.]*\. 1 is still being deleted\./);
    assert.equal(mount.querySelector('[data-storage-door]').hidden, true, 'no competing start before Done');
    assert.equal(trayButton(mount, 'data-free-preview'), null);
    assert.equal(document.activeElement?.getAttribute('data-key'), 'data-free-done');
    assert.equal(timers.polls(), 1, 'the maintenance child is still deleting');
    routes.set(storageUrl, response(done({ removed: 147, removed_bytes: 147e9, stuck: 1, failed: 1 })));
    await timers.fire();
    await settle();
    assert.match(storageTray(mount).textContent, /1 could not be deleted yet; the app retries at its next maintenance run\./);
    assert.equal(timers.polls(), 0, 'an entry the child reported stuck is not polled for');
    assert.equal(mount.querySelector('[data-storage-live]').textContent, 'Not counted yet. 1 could not be deleted. 1 could not be deleted yet; the app retries at its next maintenance run. Removed 147 items.', 'the live region carries the phase, errors first');
    routes.set(storageUrl, response(done({ removed: 147, removed_bytes: 147e9, deleting: 1, deleting_bytes: 1e9, failed: 1 })));
    show('storage');
    await settle();
    assert.equal(timers.polls(), 1, 'the entry changed, the tab was shown again: it is deleting again');
    const reads = calls.filter((url) => url.startsWith('/api/storage/free/items'));
    assert.deepEqual(reads, [itemsUrl('pool')], 'one read of the pool, no outcome read until asked for');

    await pressTray(mount, 'data-free-done');
    assert.equal(storageTray(mount).textContent, '');
    assert.equal(mount.querySelector('[data-storage-door]').hidden, false, 'Done gives the door back');
    assert.equal(calls.filter((url) => url === freeUrl).length, 1, 'one POST');
});

test('Storage flow says why a run did not start, stops offering a stale list, and never repeats the POST', async (t) => {
    const { mount, routes, calls, toasts } = await openReady(t);
    routes.set(freeUrl, response({ ...readyBody(2), started: false, reason: 'preview_outdated' }, 202));
    const old = trayButton(mount, 'data-free-go');
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.deepEqual(toasts(), ['Nothing was removed. The list changed; check again.']);
    assert.equal(trayButton(mount, 'data-free-go'), null, 'a refused plan offers no Remove');
    await storageTray(mount).fire('click', { target: old });  // a press that reaches the replaced node
    await settle();
    assert.match(storageTray(mount).textContent, /The list changed; check again\. Nothing was removed\./);
    assert.equal(calls.filter((url) => url === freeUrl).length, 1);
    // Check again is the owner's: a new preview, a new plan, a whole new read and choice.
    routes.set(previewUrl, response(freeBody('ready', { plan: 'p2', pool: poolFacts(1), started: true }), 202));
    routes.set(itemsUrl('pool', { plan: 'p2' }), response({ plan: 'p2', view: 'pool', version: 'p2', total: 1, counts: {}, next_cursor: null, items: [poolEntry('z')] }));
    await pressTray(mount, 'data-free-preview');
    await settle();
    assert.match(storageTray(mount).textContent, /Remove 1 item/);
    assert.doesNotMatch(storageTray(mount).textContent, /The list changed|the list you reviewed/i, 'the owner\'s own new start has read the refusal');
    assert.ok(calls.includes(itemsUrl('pool', { plan: 'p2' })));
});

for (const reason of ['unknown_item', 'not_offered']) {
    test(`Storage flow treats a ${reason} refusal as a list that changed`, async (t) => {
        const { mount, routes, toasts } = await openReady(t);
        routes.set(freeUrl, response({ ...readyBody(2), started: false, reason }, 202));
        await pressTray(mount, 'data-free-go');
        await settle();
        assert.deepEqual(toasts(), ['Nothing was removed. The list changed; check again.']);
        assert.equal(trayButton(mount, 'data-free-go'), null);
    });
}

test('Storage flow keeps Remove on a busy refusal: nothing about the list changed', async (t) => {
    const { mount, routes, toasts } = await openReady(t);
    routes.set(freeUrl, response({ ...readyBody(2), started: false, reason: 'busy' }, 202));
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.deepEqual(toasts(), ['Nothing was removed. Something else is still running.']);
    assert.ok(trayButton(mount, 'data-free-go'));
});

test('Storage flow after a dropped POST reads what happened once and never sends it again', async (t) => {
    const { mount, routes, calls, toasts, timers } = await openReady(t);
    routes.set(freeUrl, new Error('network down'));
    const held = deferred();
    routes.set(storageUrl, () => held.promise);
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.equal(calls.filter((url) => url === freeUrl).length, 1);
    assert.deepEqual(toasts(), ['The answer to Remove was lost; nothing is sent again.'], 'neither the browser\'s words nor "nothing was removed"');
    assert.match(storageTray(mount).textContent, /The answer to Remove was lost\. Reading what the server says now; nothing is sent again\./);
    const waiting = trayButton(mount, 'data-free-go');
    assert.equal(waiting.getAttribute('aria-disabled'), 'true');
    await storageTray(mount).fire('click', { target: waiting });
    await settle();
    assert.equal(calls.filter((url) => url === freeUrl).length, 1, 'no second send while the outcome is unknown');
    // The run did begin: its truth takes over.
    held.resolve(response(freeBody('running', { total: 2, pool: poolFacts(2), run: freeRun({ done: 1, removed: 1, removed_bytes: 1e9 }) })));
    await settle();
    assert.match(storageTray(mount).textContent, /Removing · 1 of 2/);
    assert.equal(calls.filter((url) => url === freeUrl).length, 1);
    assert.equal(timers.polls(), 1);
});

test('Storage flow keeps reading after a dropped POST while the read fails too, and stays inert until one answers', async (t) => {
    const { mount, routes, calls, timers } = await openReady(t);
    routes.set(freeUrl, new Error('network down'));
    routes.set(storageUrl, response({ error: 'synthetic read refusal' }, 503));
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.equal(trayButton(mount, 'data-free-go').getAttribute('aria-disabled'), 'true');
    assert.equal(timers.polls(), 1, 'the unknown outcome is read for, not forgotten');
    routes.set(storageUrl, response(readyBody(2)));
    await timers.fire();
    await settle();
    // The plan still stands, but the read cannot say the lost request will never land: no Remove on it again.
    assert.equal(trayButton(mount, 'data-free-go'), null);
    assert.ok(trayButton(mount, 'data-free-preview'), 'a new list is the way on');
    assert.equal(timers.polls(), 0);
    assert.equal(calls.filter((url) => url === freeUrl).length, 1);
});

test('Storage flow keeps the owner\'s press after a server that answered with a refusal status', async (t) => {
    const { mount, routes, toasts } = await openReady(t);
    routes.set(freeUrl, response({ error: 'The request was malformed.' }, 400));
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.deepEqual(toasts(), ['The request was malformed.']);
    assert.doesNotMatch(storageTray(mount).textContent, /was lost/, 'a status means it was answered');
    assert.equal(trayButton(mount, 'data-free-go').getAttribute('aria-disabled'), null);
});

// The server admits the run, then its answer fails (a projection or serialization error after start): HTTP 5xx says
// nothing about whether the run began, so it is as unknown as a dropped connection.
for (const status of [500, 502, 503, 504]) {
    test(`Storage flow treats a ${status} answer to Remove as unknown and never sends it again`, async (t) => {
        const { mount, routes, calls, toasts } = await openReady(t);
        routes.set(freeUrl, response({ error: 'Internal Server Error' }, status));
        routes.set(storageUrl, response(readyBody(2)));
        await pressTray(mount, 'data-free-go');
        await settle();
        assert.deepEqual(toasts(), ['The answer to Remove was lost; nothing is sent again.'], 'never the server\'s words, never "nothing was removed"');
        const tray = storageTray(mount).textContent;
        assert.match(tray, /The answer to Remove was lost\. It may still start, so nothing is sent again from this list\./);
        assert.doesNotMatch(tray, /Nothing was removed/);
        assert.equal(trayButton(mount, 'data-free-go'), null, 'no Remove on this plan again, even after a ready-looking read');
        assert.equal(calls.filter((url) => url === freeUrl).length, 1);
    });
}

// A 202 whose body was cut (ERR_CONTENT_LENGTH_MISMATCH) was admitted by the server: the same unknown as a dropped
// connection. The client's placeholder for an unreadable body once passed as an answer and the tray vanished.
const unreadable = (status = 202) => ({ ok: true, status, json: async () => { throw new TypeError('network error'); } });

test('Storage flow treats an unreadable 202 body for Remove as unknown, keeps the tray and never sends it again', async (t) => {
    const { mount, routes, calls, toasts } = await openReady(t);
    routes.set(freeUrl, unreadable());
    routes.set(storageUrl, response(readyBody(2)));
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.deepEqual(toasts(), ['The answer to Remove was lost; nothing is sent again.']);
    const tray = storageTray(mount).textContent;
    assert.match(tray, /The answer to Remove was lost\. It may still start, so nothing is sent again from this list\./);
    assert.doesNotMatch(tray, /Nothing was removed/);
    assert.equal(trayButton(mount, 'data-free-go'), null);
    assert.equal(calls.filter((url) => url === freeUrl).length, 1);
    assert.ok(calls.filter((url) => url === storageUrl).length >= 2, 'a read says what happened');
});

test('Storage flow says an unreadable 202 body for a count in its own words and does not take it for a state', async (t) => {
    const { routes, toasts, storage } = await openReady(t);
    routes.set('/api/storage/measure', unreadable());
    await (await storage).measure();
    await settle();
    assert.deepEqual(toasts(), ['Storage could not be measured.']);
});

test('Storage flow keeps a migrating 503 as a definite refusal: the gate answered before any handler ran', async (t) => {
    const { mount, routes, toasts } = await openReady(t);
    routes.set(freeUrl, response({ error: 'The store is migrating.', reason_code: 'migrating' }, 503));
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.doesNotMatch(storageTray(mount).textContent, /was lost/);
    assert.equal(toasts().length, 1);
    assert.equal(trayButton(mount, 'data-free-go').getAttribute('aria-disabled'), null);
});

test('Storage flow refuses a pool page it cannot trust, offers Try again, and never fills the gap with groups', async (t) => {
    const { mount, routes, calls, start } = openStorage(t);
    routes.set(storageUrl, response(readyBody(3, { groups: [freeGroup('work_folders', 3e9)] })));
    routes.set(itemsUrl('pool'), page('pool', [poolEntry('a'), poolEntry('a')], { total: 3 }));
    start();
    await settle();
    assert.match(storageTray(mount).textContent, /The whole list could not be read, so nothing can be removed from here yet\./);
    assert.equal(trayButton(mount, 'data-free-go'), null);
    assert.equal(calls.filter((url) => url === itemsUrl('pool')).length, 1, 'no retry by itself');
    routes.set(itemsUrl('pool'), page('pool', [poolEntry('a'), poolEntry('b'), poolEntry('c')]));
    // Try again starts the same read over; an id already seen in the refused page must not count against it.
    const retry = trayButton(mount, 'data-free-retry-pool');
    await storageTray(mount).fire('click', { target: retry });
    await settle();
    assert.match(storageTray(mount).textContent, /Remove 3 items/);
});

test('Storage flow names a pool read that failed, with no loop, and an outdated plan by what to do next', async (t) => {
    const { mount, routes, calls, start } = openStorage(t);
    routes.set(storageUrl, response(readyBody(2)));
    routes.set(itemsUrl('pool'), response({ error: 'synthetic refusal' }, 503));
    start();
    await settle();
    assert.match(storageTray(mount).textContent, /The whole list could not be read/);
    assert.equal(calls.filter((url) => url === itemsUrl('pool')).length, 1);
    routes.set(itemsUrl('pool'), response({ error: 'The plan is gone.', reason_code: 'preview_outdated' }, 409));
    await pressTray(mount, 'data-free-retry-pool');
    await settle();
    assert.match(storageTray(mount).textContent, /The list changed; check again\./);
    assert.doesNotMatch(storageTray(mount).textContent, /Nothing was removed/, 'a refused read proves nothing about removals');
    assert.equal(trayButton(mount, 'data-free-go'), null);
    assert.ok(trayButton(mount, 'data-free-preview'));
});

test('Storage flow reads outcomes only when asked, restarts from the first page when the version moves, and reads again as the trash settles', async (t) => {
    const { mount, routes, calls, timers, start, show } = openStorage(t);
    routes.set(storageUrl, response(freeBody('done', { pool: poolFacts(102), run: freeRun({ selected: 102, removed: 1, removed_bytes: 1e9, deleting: 1, failed: 1 }) })));
    start();
    await settle();
    assert.equal(calls.filter((url) => url.includes('/free/items')).length, 0, 'closed: nothing read');
    const row = (id, outcome) => poolEntry(id, { outcome });
    const hundred = (outcome) => Array.from({ length: 100 }, (_, index) => row(`r${index}`, outcome));
    const first = (rows, version = 'v1') => response({ plan: 'p1', view: 'outcomes', version, total: 102, counts: {}, next_cursor: 100, items: rows });
    routes.set(itemsUrl('outcomes'), first([row('c', 'failed'), row('b', 'deleting'), ...hundred('removed').slice(2)]));
    await toggleOpen(mount, 'each');
    await settle();
    assert.match(storageTray(mount).textContent, /Each item \(102\)[\s\S]*Task c[\s\S]*Could not be deleted\.[\s\S]*Task b[\s\S]*Moved aside; still being deleted\.[\s\S]*Show 2 more/);
    assert.equal(calls.filter((url) => url.includes('/free/items')).length, 1, 'one page for one opening');

    // The next page finds the version moved: the rows on screen go and the list is read again, never spliced.
    routes.set(itemsUrl('outcomes', { cursor: 100, version: 'v1' }), response({ error: 'The list moved.', reason_code: 'view_changed' }, 409));
    routes.set(itemsUrl('outcomes'), first([row('c', 'failed'), ...hundred('removed').slice(1)], 'v2'));
    routes.set(itemsUrl('outcomes', { cursor: 100, version: 'v2' }), response({ plan: 'p1', view: 'outcomes', version: 'v2', total: 102, counts: {}, next_cursor: null, items: [row('b', 'removed'), row('a', 'removed')] }));
    await pressTray(mount, 'data-free-more-outcomes');
    await settle();
    const seen = storageTray(mount).querySelectorAll('[data-outcome]').map((node) => node.getAttribute('data-outcome'));
    assert.equal(seen.length, 102, 'no row twice, none lost');
    assert.equal(seen.filter((outcome) => outcome === 'deleting').length, 0, 'the old rows are gone');
    assert.equal(trayButton(mount, 'data-free-more-outcomes'), null);

    // The run's own counts moved at a poll: the open list is read again from the top.
    const reads = () => calls.filter((url) => url === itemsUrl('outcomes')).length;
    const before = reads();
    const settled = response(freeBody('done', { pool: poolFacts(102), run: freeRun({ selected: 102, removed: 2, removed_bytes: 2e9, deleting: 0, failed: 1 }) }));
    routes.set(storageUrl, settled);
    routes.set(itemsUrl('outcomes'), first([row('c', 'failed'), ...hundred('removed').slice(1)], 'v3'));
    routes.set(itemsUrl('outcomes', { cursor: 100, version: 'v3' }), response({ plan: 'p1', view: 'outcomes', version: 'v3', total: 102, counts: {}, next_cursor: null, items: [row('b', 'removed'), row('a', 'removed')] }));
    await timers.fire();
    await settle();
    assert.equal(reads(), before + 1);
    assert.equal(storageTray(mount).querySelectorAll('[data-outcome]').length, 102, 'as many rows as the owner had asked for');
    assert.equal(timers.polls(), 0);

    // A list that never stops moving ends in a sentence, not in a loop.
    routes.set(itemsUrl('outcomes'), response({ plan: 'p1', view: 'outcomes', version: 'v9', total: 102, counts: {}, next_cursor: 100, items: hundred('removed') }));
    routes.set(itemsUrl('outcomes', { cursor: 100, version: 'v9' }), response({ error: 'The list moved.', reason_code: 'view_changed' }, 409));
    routes.set(storageUrl, response(freeBody('done', { pool: poolFacts(102), run: freeRun({ selected: 102, removed: 3, removed_bytes: 3e9, deleting: 0, failed: 0 }) })));
    const again = reads();
    show('storage');
    await settle();
    await settle();
    assert.ok(reads() - again <= 4, 'bounded restarts');
    assert.match(storageTray(mount).textContent, /Each item’s result could not be read in full\.[\s\S]*Try again/);
    assert.equal(timers.polls(), 0);
});

test('Storage flow lists what stays only when opened, page by page', async (t) => {
    const stays = Array.from({ length: 100 }, (_, index) => poolEntry(`x${index}`, { reason: index ? 'in_use' : 'capture_unproven', bytes: null }));
    const { mount, routes, calls } = await openReady(t, [poolEntry('a')], { pool: poolFacts(1, { excluded: [{ reason: 'capture_unproven', count: 1 }, { reason: 'in_use', count: 101 }] }) });
    assert.equal(calls.filter((url) => url.includes('view=excluded')).length, 0);
    routes.set(itemsUrl('excluded'), page('excluded', stays, { total: 102, next_cursor: 100 }));
    await toggleOpen(mount, 'left');
    await settle();
    assert.match(storageTray(mount).textContent, /1: unique work not verified saved · 101: still in use[\s\S]*Task x0[\s\S]*Task x99[\s\S]*Show 2 more/);
    // A repaint re-opens the disclosure and its toggle reads nothing more.
    await toggleOpen(mount, 'left');
    await settle();
    assert.equal(calls.filter((url) => url.includes('view=excluded')).length, 1);
    routes.set(itemsUrl('excluded', { cursor: 100, version: 'p1' }), page('excluded', [poolEntry('y', { reason: 'in_use', bytes: null }), poolEntry('z', { reason: 'in_use', bytes: null })], { total: 102 }));
    await pressTray(mount, 'data-free-more-excluded');
    await settle();
    assert.equal(storageTray(mount).querySelectorAll('[data-free-excluded]').length, 102);
    assert.equal(trayButton(mount, 'data-free-more-excluded'), null);
});

test('Storage flow reaches the saved groups on purpose: nothing checked, the shared confirm, one POST of whole groups', async (t) => {
    const groups = [freeGroup('work_folders', 6.8e9), freeGroup('review_evidence', 43e9), freeGroup('result_pins', 2e5)];
    const { mount, routes, calls, document } = await openReady(t, [poolEntry('a')], { groups });
    assert.match(storageTray(mount).textContent, /Saved, unknown or undecided: this button never removes them/);
    assert.equal(storageTray(mount).querySelector('[data-free-group]'), null, 'read-only until asked for');
    await pressTray(mount, 'data-free-saved');
    assert.equal(document.activeElement?.getAttribute('data-key'), 'data-free-saved-back', 'focus moves into the checklist');
    assert.match(storageTray(mount).textContent, /These hold saved work, so nothing here starts checked\.[\s\S]*Nothing selected/);
    assert.equal(trayButton(mount, 'data-free-go'), null, 'no ordinary Remove competes here');
    assert.equal(trayButton(mount, 'data-free-saved-go'), null);
    const box = storageTray(mount).querySelector('[data-free-group="review_evidence"]');
    box.checked = true;
    await storageTray(mount).fire('change', { target: box });
    const go = trayButton(mount, 'data-free-saved-go');
    assert.ok(go.className.includes('btn-danger'));
    assert.equal(go.textContent, 'Free 43 GB');

    // Declining the dialog sends nothing.
    let click = storageTray(mount).fire('click', { target: go });
    await settle();
    let dialog = document.body.querySelector('[role="dialog"]');
    assert.match(dialog.textContent, /Free temporary space\?[\s\S]*This removes 43 GB for good\. Saved work goes too: Reviewers’ saved files of finished tasks\./);
    await dialog.parentElement.fire('click', { target: dialog.querySelector('[data-confirm-cancel]') });
    await click;
    await settle();
    assert.equal(calls.filter((url) => url === freeUrl).length, 0);

    let posted = null;
    routes.set(freeUrl, async (_url, init) => {
        posted = JSON.parse(init.body);
        return response(freeBody('running', { total: 3, pool: poolFacts(1), groups, run: freeRun({ groups: posted.groups, selected: 3, done: 1, removed: 0, removed_bytes: 0 }), started: true }), 202);
    });
    click = storageTray(mount).fire('click', { target: trayButton(mount, 'data-free-saved-go') });
    await settle();
    dialog = document.body.querySelector('[role="dialog"]');
    await dialog.parentElement.fire('click', { target: dialog.querySelector('[data-confirm-ok]') });
    await click;
    await settle();
    assert.deepEqual(posted, { plan: 'p1', groups: ['review_evidence'] });
    assert.match(storageTray(mount).textContent, /Removing · 1 of 3/);
});

for (const reason of ['preview_outdated', 'unknown_group', 'not_offered']) {
    test(`Storage flow takes the destructive capsule away from saved groups after a ${reason} refusal and never sends again`, async (t) => {
        const groups = [freeGroup('review_evidence', 43e9)];
        const { mount, routes, calls, toasts, document } = await openReady(t, [poolEntry('a')], { groups });
        await pressTray(mount, 'data-free-saved');
        const box = storageTray(mount).querySelector('[data-free-group="review_evidence"]');
        box.checked = true;
        await storageTray(mount).fire('change', { target: box });
        routes.set(freeUrl, response({ ...readyBody(1, { groups }), started: false, reason }, 202));
        const old = trayButton(mount, 'data-free-saved-go');
        const click = storageTray(mount).fire('click', { target: old });
        await settle();
        const dialog = document.body.querySelector('[role="dialog"]');
        await dialog.parentElement.fire('click', { target: dialog.querySelector('[data-confirm-ok]') });
        await click;
        await settle();
        assert.deepEqual(toasts(), ['Nothing was removed. The list changed; check again.']);
        assert.match(storageTray(mount).textContent, /Saved groups[\s\S]*The list changed; check again\. Nothing was removed\./);
        assert.equal(trayButton(mount, 'data-free-saved-go'), null, 'no destructive capsule on a refused plan');
        assert.equal(storageTray(mount).querySelector('.btn-danger'), null);
        assert.ok(trayButton(mount, 'data-free-preview'), 'Check again is offered');
        // A press that reaches the replaced capsule opens no dialog and sends nothing.
        document.body.querySelector('[role="dialog"]')?.remove();
        await storageTray(mount).fire('click', { target: old });
        await settle();
        assert.equal(document.body.querySelector('[role="dialog"]'), null);
        assert.equal(calls.filter((url) => url === freeUrl).length, 1, 'one POST');
        // Back to the review: it says the same and offers no Remove either.
        await pressTray(mount, 'data-free-saved-back');
        assert.equal(trayButton(mount, 'data-free-go'), null);
        assert.ok(trayButton(mount, 'data-free-preview'));
    });
}

test('Storage flow sends no row hidden below the shown hundred that the owner left unchosen', async (t) => {
    const items = Array.from({ length: 150 }, (_, index) => poolEntry(`i${index}`));
    const { mount, routes } = await openReady(t, items);
    assert.match(storageTray(mount).textContent, /The 50 not shown keep their choice: 50 chosen, and Remove takes those too\./);
    await pressTray(mount, 'data-free-none');
    const first = storageTray(mount).querySelectorAll('[data-free-pick]')[0];
    first.checked = true;
    await storageTray(mount).fire('change', { target: first });
    assert.match(storageTray(mount).textContent, /1 of 150 chosen[\s\S]*The 50 not shown keep their choice: none chosen\.[\s\S]*Remove 1 item/);
    let posted = null;
    routes.set(freeUrl, async (_url, init) => {
        posted = JSON.parse(init.body);
        return response(freeBody('running', { total: 1, pool: poolFacts(150), run: freeRun({ selected: 1, done: 0, removed: 0, removed_bytes: 0 }), started: true }), 202);
    });
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.deepEqual(posted, { plan: 'p1', items: ['i0'] });
});

test('Storage flow stops a run on Stop and keeps what was stopped in the result', async (t) => {
    const { mount, routes, calls, start } = openStorage(t);
    routes.set(storageUrl, response(freeBody('running', { total: 3, run: freeRun({ selected: 3, done: 1, removed: 0, removed_bytes: 0, deleting: 1, deleting_bytes: 1e9 }) })));
    const storage = start();
    await settle();
    routes.set(freeCancelUrl, (_url, init) => {
        assert.equal(init.method, 'POST');
        return response(freeBody('stopped', { run: freeRun({ selected: 3, done: 1, removed: 0, removed_bytes: 0, deleting: 1, not_started: 2, stopped: true }) }), 202);
    });
    await pressTray(mount, 'data-free-stop');
    await settle();
    assert.ok(calls.includes(freeCancelUrl));
    assert.match(storageTray(mount).textContent, /Removed 0 items, then stopped\. The rest was left as it was\./);
    assert.match(storageTray(mount).textContent, /2 were not started and stay as they were\./);
    storage.destroy();
});

test('Storage flow stops reading when its tab is left, resumes the same read on return, and a new plan starts clean', async (t) => {
    const { mount, routes, calls, start, show } = openStorage(t);
    const held = deferred();
    let signal = null;
    routes.set(storageUrl, response(readyBody(3)));
    routes.set(itemsUrl('pool'), page('pool', [poolEntry('a'), poolEntry('b')], { total: 3, next_cursor: 2 }));
    routes.set(itemsUrl('pool', { cursor: 2, version: 'p1' }), (_url, init) => { signal = init.signal; return held.promise; });
    start();
    await settle();
    show('logs');
    assert.equal(signal.aborted, true, 'leaving the tab aborts its read');
    held.resolve(page('pool', [poolEntry('c')], { total: 3 }));  // a server that answered anyway
    await settle();
    assert.doesNotMatch(storageTray(mount).textContent, /Remove 3 items/, 'a late page counts for nothing');

    routes.set(itemsUrl('pool', { cursor: 2, version: 'p1' }), page('pool', [poolEntry('c')], { total: 3 }));
    show('storage');
    await settle();
    assert.match(storageTray(mount).textContent, /Remove 3 items/);
    assert.equal(calls.filter((url) => url === itemsUrl('pool')).length, 1, 'the page already read is not read again');

    // Another tab previewed: this flow's choices and rows mean nothing under the new plan.
    await pressTray(mount, 'data-free-none');
    routes.set(storageUrl, response(freeBody('ready', { plan: 'p2', pool: poolFacts(1) })));
    routes.set(itemsUrl('pool', { plan: 'p2' }), response({ plan: 'p2', view: 'pool', version: 'p2', total: 1, counts: {}, next_cursor: null, items: [poolEntry('n')] }));
    show('logs');
    show('storage');
    await settle();
    assert.match(storageTray(mount).textContent, /1 of 1 chosen[\s\S]*Remove 1 item/);
    const box = storageTray(mount).querySelector('[data-free-pick]');
    assert.equal(box.hasAttribute('checked'), true, 'nothing carried over');
});

test('Storage destroy aborts a pool read and ignores its late page and every later press', async (t) => {
    const { mount, routes, calls, start } = openStorage(t);
    const held = deferred();
    let signal = null;
    routes.set(storageUrl, response(readyBody(2)));
    routes.set(itemsUrl('pool'), (_url, init) => { signal = init.signal; return held.promise; });
    const storage = start();
    await settle();
    storage.destroy();
    assert.equal(signal.aborted, true);
    held.resolve(page('pool', [poolEntry('a'), poolEntry('b')]));
    await settle();
    assert.doesNotMatch(storageTray(mount).textContent, /Remove 2 items/);
    const before = calls.length;
    await storageTray(mount).fire('click', { target: trayButton(mount, 'data-free-cancel') });
    await settle();
    assert.equal(calls.length, before);
});

// ---------------------------------------------------------------------------
// The owner's attempt beside the server's current facts: a refusal or a lost answer is said where the flow now stands,
// and nothing the server no longer holds is offered as sendable. Envelopes are the real server's: a `preview_outdated`
// answer carries the projection as it is now (no plan after a restart, a listing or another plan after a new preview).
// ---------------------------------------------------------------------------

const idleFree = { state: 'idle', plan: null, started_at: null, finished_at: null, done: 0, total: 0, groups: [], pool: null, run: null, error: null };
const listingFree = { ...idleFree, state: 'listing', started_at: '2026-10-05T12:00:00Z', total: 4 };
const p2Ready = (count = 1) => freeBody('ready', { plan: 'p2', pool: poolFacts(count) }).free;
const p2Page = (items) => response({ plan: 'p2', view: 'pool', version: 'p2', total: items.length, counts: {}, next_cursor: null, items });
const counting = { started_at: '2026-10-05T10:59:00Z', place: 'engine', current: 'projects', places_done: 1, places_total: 12, bytes: 1e9, files: 10, done: [] };

test('Storage flow names saved-group items it leaves out by their own words, never as "another reason" that cannot go', async (t) => {
    const groups = [freeGroup('review_evidence', 4e9)];
    const excluded = [{ reason: 'advanced_group', count: 2 }, { reason: 'in_use', count: 1 }];
    const { mount, routes } = await openReady(t, [poolEntry('a')], { groups, pool: poolFacts(1, { excluded }) });
    // The server sorts by reason: the saved-group items come first.
    routes.set(itemsUrl('excluded'), page('excluded', [
        poolEntry('s1', { group: 'review_evidence', kind: 'other', reason: 'advanced_group', bytes: null }),
        poolEntry('s2', { group: 'review_evidence', kind: 'other', reason: 'advanced_group', bytes: null }),
        poolEntry('u1', { reason: 'in_use', bytes: null }),
    ]));
    await toggleOpen(mount, 'left');
    await settle();
    const tray = storageTray(mount).textContent;
    assert.doesNotMatch(tray, /another reason/);
    assert.match(tray, /Can’t go right now[\s\S]*1: still in use/);
    assert.doesNotMatch(tray, /Can’t go right now[^]*?2: /, 'the saved groups say their own count');
    const rows = storageTray(mount).querySelectorAll('[data-free-excluded]');
    assert.equal(rows.length, 3);
    assert.match(rows[0].textContent, /only its saved group can remove it/);
});

for (const [name, now, expectRest] of [['no plan after a restart', idleFree, true], ['a listing started elsewhere', listingFree, false]]) {
    test(`Storage flow keeps a preview_outdated refusal said when the answer holds ${name}`, async (t) => {
        const { mount, routes, calls, toasts } = await openReady(t);
        routes.set(freeUrl, response({ ...storageBody({ free: now }), started: false, reason: 'preview_outdated' }, 202));
        await pressTray(mount, 'data-free-go');
        await settle();
        assert.deepEqual(toasts(), ['Nothing was removed. The list changed; check again.']);
        assert.match(storageTray(mount).textContent, /Nothing was removed: the list you reviewed had changed\./, 'the attempt is still said in place');
        assert.equal(trayButton(mount, 'data-free-go'), null);
        if (expectRest) assert.equal(mount.querySelector('[data-storage-door]').hidden, false, 'the server holds no plan: the door is the next step');
        else assert.match(storageTray(mount).textContent, /Checking what can go/, 'the server\'s own phase stands beside it');
        assert.equal(calls.filter((url) => url === freeUrl).length, 1);
        if (!expectRest) return;
        // The owner's next press starts over and the old sentence goes.
        routes.set(previewUrl, response({ ...storageBody({ free: listingFree }), started: true }, 202));
        await mount.querySelector('[data-storage-door]').fire('click', { target: mount.querySelector('[data-storage-door]') });
        await settle();
        assert.doesNotMatch(storageTray(mount).textContent, /the list you reviewed/);
    });
}

test('Storage flow shows the plan that replaced the refused one as a new review, with the refusal said and nothing carried over', async (t) => {
    const { mount, routes } = await openReady(t);
    await pressTray(mount, 'data-free-none');
    routes.set(freeUrl, response({ ...storageBody({ free: p2Ready(1) }), started: false, reason: 'preview_outdated' }, 202));
    routes.set(itemsUrl('pool', { plan: 'p2' }), p2Page([poolEntry('n')]));
    const first = storageTray(mount).querySelector('[data-free-pick]');
    first.checked = true;
    await storageTray(mount).fire('change', { target: first });
    await pressTray(mount, 'data-free-go');
    await settle();
    const tray = storageTray(mount).textContent;
    assert.match(tray, /Nothing was removed: the list you reviewed had changed\./);
    assert.match(tray, /1 of 1 chosen/, 'the new plan\'s own list, read whole');
    assert.doesNotMatch(tray, /Task a|Task b/);
});

test('Storage flow keeps an outdated read said when the refreshed answer holds no plan, instead of emptying the tray', async (t) => {
    const { mount, routes } = await openReady(t, [poolEntry('a')], { pool: poolFacts(1, { excluded: [{ reason: 'in_use', count: 1 }] }) });
    routes.set(itemsUrl('excluded'), response({ error: 'preview_outdated', reason_code: 'preview_outdated' }, 409));
    routes.set(storageUrl, response(storageBody({ free: idleFree })));
    await toggleOpen(mount, 'left');
    await settle();
    await settle();
    assert.match(storageTray(mount).textContent, /The list you were reviewing changed\./);
    assert.equal(mount.querySelector('[data-storage-door]').hidden, false);
});

test('Storage flow clears its poll while a POST is out, so the POST\'s own answer is the next truth', async (t) => {
    const { mount, routes, calls, timers, start } = openStorage(t);
    routes.set(storageUrl, response({ ...readyBody(2), counting }));
    routes.set(itemsUrl('pool'), page('pool', [poolEntry('a'), poolEntry('b')]));
    start();
    await settle();
    assert.equal(timers.polls(), 1, 'a running count is polled');
    const held = deferred();
    routes.set(freeUrl, () => held.promise);
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.equal(timers.polls(), 0, 'no poll races the POST');
    const reads = calls.filter((url) => url === storageUrl).length;
    await timers.fire();
    await settle();
    assert.equal(calls.filter((url) => url === storageUrl).length, reads);
    held.resolve(response({ ...freeBody('running', { total: 2, pool: poolFacts(2), run: freeRun({ done: 0, removed: 0, removed_bytes: 0 }) }), counting, started: true }, 202));
    await settle();
    assert.match(storageTray(mount).textContent, /Removing · 0 of 2/);
    assert.equal(calls.filter((url) => url === storageUrl).length, reads, 'the POST answer was applied, not dropped for a re-read');
    assert.equal(timers.polls(), 1);
});

test('Storage flow holds the review inert while Remove is out: no Cancel, no new choice, the press reads as sent', async (t) => {
    const { mount, routes } = await openReady(t);
    const held = deferred();
    let posted = null;
    routes.set(freeUrl, (_url, init) => { posted = JSON.parse(init.body); return held.promise; });
    await pressTray(mount, 'data-free-go');
    await settle();
    const go = trayButton(mount, 'data-free-go');
    assert.equal(go.getAttribute('aria-disabled'), 'true');
    assert.equal(go.textContent, 'Sending…');
    await pressTray(mount, 'data-free-cancel');
    await pressTray(mount, 'data-free-none');
    const box = storageTray(mount).querySelector('[data-free-pick]');
    assert.equal(box.hasAttribute('disabled'), true);
    box.checked = false;
    await storageTray(mount).fire('change', { target: box });
    assert.match(storageTray(mount).textContent, /2 of 2 chosen/, 'what is shown is what was sent');
    assert.notEqual(storageTray(mount).textContent, '', 'Cancel did not hide an unanswered Remove');
    assert.deepEqual(posted.items, ['a', 'b']);
    held.resolve(response({ ...readyBody(2), started: false, reason: 'busy' }, 202));
    await settle();
    assert.equal(trayButton(mount, 'data-free-go').getAttribute('aria-disabled'), null, 'answered: the review is the owner\'s again');
});

test('Storage flow after a lost Remove answer and a read that shows the same plan never offers that list again', async (t) => {
    const { mount, routes, calls, timers, document } = await openReady(t);
    routes.set(freeUrl, new Error('network down'));
    routes.set(storageUrl, response(readyBody(2)));
    await pressTray(mount, 'data-free-go');
    await settle();
    // The read says the plan still stands. It cannot say the request never reached the server: it may land later.
    const tray = storageTray(mount).textContent;
    assert.match(tray, /The answer to Remove was lost\. It may still start, so nothing is sent again from this list\./);
    assert.doesNotMatch(tray, /Nothing was removed/, 'no certainty the read cannot give');
    assert.equal(trayButton(mount, 'data-free-go'), null, 'no Remove on this plan again');
    assert.equal(trayButton(mount, 'data-free-cancel'), null, 'nothing hides the unknown outcome');
    assert.equal(mount.querySelector('[data-storage-door]').hidden, true);
    const box = storageTray(mount).querySelector('[data-free-pick]');
    assert.equal(box.hasAttribute('disabled'), true, 'no new selection on a list that may still be running');
    await pressTray(mount, 'data-free-none');
    assert.match(storageTray(mount).textContent, /2 of 2 chosen/);
    assert.equal(timers.polls(), 0, 'one read answered: no poll of its own');
    assert.equal(calls.filter((url) => url === freeUrl).length, 1);
    // The original request lands late and starts; the owner's Check again finds the run and shows it.
    routes.set(previewUrl, response({ ...freeBody('running', { total: 2, pool: poolFacts(2), run: freeRun({ done: 1, removed: 1, removed_bytes: 1e9 }) }), started: false }, 202));
    trayButton(mount, 'data-free-preview').focus();
    await pressTray(mount, 'data-free-preview');
    await settle();
    assert.match(storageTray(mount).textContent, /Removing · 1 of 2/);
    assert.doesNotMatch(storageTray(mount).textContent, /was lost/);
    assert.equal(calls.filter((url) => url === freeUrl).length, 1, 'never resent');
    assert.equal(document.activeElement?.getAttribute('data-key'), 'data-free-stop');
});

test('Storage flow after a lost Remove answer and a replaced plan says the outcome is unknown, beside the new list', async (t) => {
    const { mount, routes, calls } = await openReady(t);
    routes.set(freeUrl, new Error('network down'));
    routes.set(storageUrl, response(storageBody({ free: p2Ready(1) })));
    routes.set(itemsUrl('pool', { plan: 'p2' }), p2Page([poolEntry('n')]));
    await pressTray(mount, 'data-free-go');
    await settle();
    const tray = storageTray(mount).textContent;
    assert.match(tray, /The answer to Remove was lost and the list has changed since; whether it removed anything is not known here\. Nothing is sent again\./);
    assert.match(tray, /Remove 1 item/, 'the new plan is the owner\'s: the lost request can no longer start under it');
    assert.equal(calls.filter((url) => url === freeUrl).length, 1);
    // The owner's next Remove is a new attempt: the old sentence goes with it.
    routes.set(freeUrl, response({ ...freeBody('running', { plan: 'p2', total: 1, pool: poolFacts(1), run: freeRun({ selected: 1, done: 0, removed: 0, removed_bytes: 0 }) }), started: true }, 202));
    await pressTray(mount, 'data-free-go');
    await settle();
    assert.match(storageTray(mount).textContent, /Removing · 0 of 1/);
    assert.doesNotMatch(storageTray(mount).textContent, /was lost/);
});

test('Storage flow holds the saved groups while their Free is out: no Back, no new check', async (t) => {
    const groups = [freeGroup('review_evidence', 43e9), freeGroup('result_pins', 2e5)];
    const { mount, routes, document } = await openReady(t, [poolEntry('a')], { groups });
    await pressTray(mount, 'data-free-saved');
    const box = storageTray(mount).querySelector('[data-free-group="review_evidence"]');
    box.checked = true;
    await storageTray(mount).fire('change', { target: box });
    const held = deferred();
    routes.set(freeUrl, () => held.promise);
    const click = storageTray(mount).fire('click', { target: trayButton(mount, 'data-free-saved-go') });
    await settle();
    const dialog = document.body.querySelector('[role="dialog"]');
    await dialog.parentElement.fire('click', { target: dialog.querySelector('[data-confirm-ok]') });
    await settle();
    await pressTray(mount, 'data-free-saved-back');
    assert.match(storageTray(mount).textContent, /These hold saved work/, 'Back is inert while Free is out');
    const other = storageTray(mount).querySelector('[data-free-group="result_pins"]');
    assert.equal(other.hasAttribute('disabled'), true);
    other.checked = true;
    await storageTray(mount).fire('change', { target: other });
    assert.equal(storageTray(mount).querySelector('[data-free-group="result_pins"]').hasAttribute('checked'), false);
    held.resolve(response({ ...freeBody('running', { total: 3, pool: poolFacts(1), groups, run: freeRun({ groups: ['review_evidence'], selected: 3, done: 0, removed: 0, removed_bytes: 0 }) }), started: true }, 202));
    await click;
    await settle();
    assert.match(storageTray(mount).textContent, /Removing · 0 of 3/);
});

test('Storage flow resumes an interrupted pool read on tab return even when the status read fails', async (t) => {
    const { mount, routes, calls, start, show } = openStorage(t);
    const held = deferred();
    routes.set(storageUrl, response(readyBody(3)));
    routes.set(itemsUrl('pool'), page('pool', [poolEntry('a'), poolEntry('b')], { total: 3, next_cursor: 2 }));
    routes.set(itemsUrl('pool', { cursor: 2, version: 'p1' }), () => held.promise);
    start();
    await settle();
    show('logs');
    routes.set(storageUrl, response({ error: 'synthetic read refusal' }, 503));
    routes.set(itemsUrl('pool', { cursor: 2, version: 'p1' }), page('pool', [poolEntry('c')], { total: 3 }));
    show('storage');
    await settle();
    await settle();
    assert.match(storageTray(mount).textContent, /Storage status could not be read|Remove 3 items/);
    assert.match(storageTray(mount).textContent, /Remove 3 items/, 'the held plan\'s list is read on, not left at "Reading the list…"');
    assert.equal(calls.filter((url) => url === itemsUrl('pool', { cursor: 2, version: 'p1' })).length, 2);
});

test('Storage flow says a list read refused with preview_outdated changed the list, never that nothing was removed, beside a lost Remove', async (t) => {
    const { mount, routes, calls } = await openReady(t, [poolEntry('a')], { pool: poolFacts(1, { excluded: [{ reason: 'in_use', count: 1 }] }) });
    routes.set(freeUrl, new Error('network down'));
    routes.set(storageUrl, response({ error: 'synthetic read refusal' }, 503));
    await pressTray(mount, 'data-free-go');
    await settle();
    // The plan is still the one held: the status read fails, so no newer plan ever replaces it.
    routes.set(itemsUrl('excluded'), response({ error: 'preview_outdated', reason_code: 'preview_outdated' }, 409));
    await toggleOpen(mount, 'left');
    await settle();
    await settle();
    assert.ok(calls.includes(itemsUrl('excluded')));
    const tray = storageTray(mount).textContent;
    assert.match(tray, /The answer to Remove was lost\./);
    assert.match(tray, /The list changed; check again\./);
    assert.doesNotMatch(tray, /Nothing was removed/, 'a refused read and a lost answer cannot say nothing was removed');
    assert.equal(calls.filter((url) => url === freeUrl).length, 1);
});

test('Storage flow keeps the same words for a list read refused while the status read fails, with nothing sent', async (t) => {
    const { mount, routes } = await openReady(t, [poolEntry('a')], { pool: poolFacts(1, { excluded: [{ reason: 'in_use', count: 1 }] }) });
    routes.set(storageUrl, response({ error: 'synthetic read refusal' }, 503));
    routes.set(itemsUrl('excluded'), response({ error: 'preview_outdated', reason_code: 'preview_outdated' }, 409));
    await toggleOpen(mount, 'left');
    await settle();
    await settle();
    const tray = storageTray(mount).textContent;
    assert.match(tray, /The list changed; check again\./);
    assert.doesNotMatch(tray, /Nothing was removed/, 'another window may have removed things under that list');
    assert.equal(trayButton(mount, 'data-free-go'), null);
    assert.ok(trayButton(mount, 'data-free-preview'));
});

// ---------------------------------------------------------------------------
// Finished copies not verified saved, discarded by hand: the controller against synthetic v2-shaped answers
// (the seam of backend 75de071b; a real combined run is a separate check)
// ---------------------------------------------------------------------------

const unsaved = (overrides = {}) => ({ not_proven_saved_count: 3, not_proven_saved_bytes: 4.2e6,
    not_proven_saved: [{ path: 'notes/draft.md', bytes: 3e6 }, { path: 'a.txt', bytes: 2e5 }], omitted_count: 0, keeps_commits: false, ...overrides });
const copyEntry = (id, overrides = {}) => poolEntry(id, { group: 'working_copies', kind: 'working_copy', reason: 'capture_unproven', availability: 'manual_discard',
    bytes: 2e9, manual: unsaved(), ...overrides });
const manualUrl = (options = {}) => itemsUrl('manual', { limit: 50, ...options });
const manualPage = (items, overrides = {}) => page('manual', items, overrides);
const manualFree = (n, extra = {}) => ({ pool: poolFacts(2, { manual: { count: n, bytes: n * 2e9, bytes_complete: true },
    excluded: [{ reason: 'capture_unproven', count: n, manual_count: n }, { reason: 'in_use', count: 1 }] }), ...extra });
const copiesOf = (n) => Array.from({ length: n }, (_, index) => copyEntry(`m${index}`));
const pick = async (mount, index, on = true) => {
    const box = storageTray(mount).querySelector(`[data-free-manual-pick="${index}"]`);
    box.checked = on;
    await storageTray(mount).fire('change', { target: box });
};
const confirmDialog = async (document, how) => {
    const dialog = document.body.querySelector('[role="dialog"]');
    await dialog.parentElement.fire('click', { target: dialog.querySelector(how === 'ok' ? '[data-confirm-ok]' : '[data-confirm-cancel]') });
};
const posts = (calls) => calls.filter((url) => url === freeUrl).length;

/** A ready plan with copies by hand, the manual screen open and its list read whole. */
async function openCopies(t, n = 3, { items = copiesOf(n), body = {} } = {}) {
    const opened = await openReady(t, [poolEntry('a'), poolEntry('b')], manualFree(n, body));
    opened.routes.set(manualUrl(), manualPage(items));
    await pressTray(opened.mount, 'data-free-manual-open');
    await settle();
    return { ...opened, items };
}

test('Storage flow opens copies by hand on purpose, reads their list whole and starts with nothing chosen', async (t) => {
    const { mount, routes, calls, document } = await openReady(t, [poolEntry('a'), poolEntry('b')], manualFree(3));
    assert.match(storageTray(mount).textContent, /Finished copies not verified saved/);
    assert.equal(calls.filter((url) => url.includes('view=manual')).length, 0, 'closed: nothing read');
    assert.match(storageTray(mount).textContent, /Remove 2 items/, 'the ordinary pool never counts them');
    routes.set(manualUrl(), manualPage(copiesOf(3)));
    await pressTray(mount, 'data-free-manual-open');
    await settle();
    assert.deepEqual(calls.filter((url) => url.includes('view=manual')), [manualUrl()], 'one read, 50 to a page');
    assert.equal(document.activeElement?.getAttribute('data-key'), 'data-free-manual-back', 'focus moves into the screen');
    const tray = storageTray(mount).textContent;
    assert.match(tray, /Copies not verified saved[\s\S]*Nothing here starts chosen\.[\s\S]*0 of 3 chosen/);
    assert.equal(storageTray(mount).querySelectorAll('[data-free-manual-pick]').length, 3);
    assert.equal(storageTray(mount).querySelectorAll('[data-free-manual-pick]').filter((box) => box.hasAttribute('checked')).length, 0, 'default unchecked');
    assert.equal(trayButton(mount, 'data-free-manual-go'), null);
    assert.equal(trayButton(mount, 'data-free-go'), null, 'no ordinary Remove competes here');
    assert.equal(storageTray(mount).querySelector('.btn-danger'), null);
    assert.match(tray, /Nothing selected/);
    assert.equal(calls.filter((url) => url === freeUrl).length, 0);
    // Back returns to the review, the pool untouched; opening again keeps no choice that was not made.
    await pressTray(mount, 'data-free-manual-back');
    assert.match(storageTray(mount).textContent, /Remove 2 items/);
});

test('Storage flow sends one POST of exactly the chosen copy ids with the acknowledgement, only after the dialog is confirmed', async (t) => {
    const { mount, routes, calls, document, items } = await openCopies(t, 4);
    await pick(mount, 1);
    await pick(mount, 3);
    assert.match(storageTray(mount).textContent, /2 of 4 chosen/);
    const go = trayButton(mount, 'data-free-manual-go');
    assert.ok(go.className.includes('btn-danger'));
    assert.equal(go.textContent, 'Discard 2 copies');
    // Declining sends nothing, and the choice stays.
    let click = storageTray(mount).fire('click', { target: go });
    await settle();
    const dialog = document.body.querySelector('[role="dialog"]');
    assert.match(dialog.textContent, /Discard these copies\?[\s\S]*This deletes 2 copies whole\. 6 files in them are not verified saved \(8\.4 MB\) and can be lost for good\./);
    await confirmDialog(document, 'cancel');
    await click;
    await settle();
    assert.equal(posts(calls), 0);
    assert.match(storageTray(mount).textContent, /2 of 4 chosen/);

    let body = null;
    routes.set(freeUrl, async (_url, init) => {
        body = JSON.parse(init.body);
        return response(freeBody('running', { total: 4, pool: poolFacts(2), run: freeRun({ selected: 2, done: 0, removed: 0, removed_bytes: 0 }), started: true }), 202);
    });
    click = storageTray(mount).fire('click', { target: trayButton(mount, 'data-free-manual-go') });
    await settle();
    await confirmDialog(document, 'ok');
    await click;
    await settle();
    assert.equal(posts(calls), 1);
    assert.deepEqual(Object.keys(body).sort(), ['accept_loss', 'manual', 'plan'], 'its own form: no items, no groups');
    assert.deepEqual(body, { plan: 'p1', manual: [items[1].id, items[3].id], accept_loss: true });
    assert.match(storageTray(mount).textContent, /Removing · 0 of 2/);
    assert.equal(document.activeElement?.getAttribute('data-key'), 'data-free-stop');
});

test('Storage flow never lets the acknowledgement stand in for a choice: nothing chosen, a choice taken back or an unoffered row sends nothing', async (t) => {
    const items = [...copiesOf(2), copyEntry('stranger', { availability: 'none' })];
    const { mount, calls, document, routes } = await openCopies(t, 3, { items });
    // A press that reaches a capsule that is not there (a stale node) opens no dialog and sends nothing.
    const stale = { hasAttribute: (name) => name === 'data-free-manual-go', getAttribute: () => null, closest: () => stale, disabled: false };
    await storageTray(mount).fire('click', { target: stale });
    await settle();
    assert.equal(document.body.querySelector('[role="dialog"]'), null);
    // Choose, then take it back: the capsule goes and so does the request.
    await pick(mount, 0);
    const old = trayButton(mount, 'data-free-manual-go');
    await pick(mount, 0, false);
    assert.equal(trayButton(mount, 'data-free-manual-go'), null);
    await storageTray(mount).fire('click', { target: old });
    await settle();
    assert.equal(document.body.querySelector('[role="dialog"]'), null, 'no dialog for a choice that was taken back');
    // The row the server did not offer by hand cannot be chosen at all.
    const stranger = storageTray(mount).querySelector('[data-free-manual-pick="2"]');
    assert.ok(stranger.hasAttribute('disabled'));
    await pick(mount, 2);
    assert.equal(trayButton(mount, 'data-free-manual-go'), null);
    assert.match(storageTray(mount).textContent, /0 of 3 chosen/);
    // Choose none clears every choice, then the one real choice is the only id sent.
    await pick(mount, 1);
    await pressTray(mount, 'data-free-manual-none');
    assert.match(storageTray(mount).textContent, /0 of 3 chosen/);
    assert.equal(posts(calls), 0);
    let body = null;
    routes.set(freeUrl, (_url, init) => { body = JSON.parse(init.body); return response(freeBody('running', { total: 3, pool: poolFacts(2), run: freeRun({ selected: 1 }), started: true }), 202); });
    await pick(mount, 0);
    const click = pressTray(mount, 'data-free-manual-go');
    await settle();
    await confirmDialog(document, 'ok');
    await click;
    await settle();
    assert.deepEqual(body.manual, ['m0']);
});

test('Storage flow keeps each form exclusive: pool Remove and saved groups never carry manual or the acknowledgement', async (t) => {
    const groups = [freeGroup('review_evidence', 43e9)];
    const { mount, routes, document } = await openReady(t, [poolEntry('a'), poolEntry('b')], { ...manualFree(2), groups });
    const bodies = [];
    routes.set(freeUrl, (_url, init) => { bodies.push(JSON.parse(init.body)); return response({ ...readyBody(2, { ...manualFree(2), groups }), started: false, reason: 'busy' }, 202); });
    await pressTray(mount, 'data-free-go');
    await settle();
    await pressTray(mount, 'data-free-saved');
    const box = storageTray(mount).querySelector('[data-free-group="review_evidence"]');
    box.checked = true;
    await storageTray(mount).fire('change', { target: box });
    const click = pressTray(mount, 'data-free-saved-go');
    await settle();
    await confirmDialog(document, 'ok');
    await click;
    await settle();
    assert.deepEqual(bodies, [{ plan: 'p1', items: ['a', 'b'] }, { plan: 'p1', groups: ['review_evidence'] }]);
});

test('Storage flow keeps the capsule off while the list is partial or failed, and a retry reads it again from the start', async (t) => {
    const { mount, routes, calls } = await openReady(t, [poolEntry('a')], manualFree(3));
    const held = deferred();
    routes.set(manualUrl(), manualPage(copiesOf(2), { total: 3, next_cursor: 2 }));
    routes.set(manualUrl({ cursor: 2, version: 'p1' }), () => held.promise);
    await pressTray(mount, 'data-free-manual-open');
    await settle();
    assert.match(storageTray(mount).textContent, /Reading the list · 2 of 3/);
    assert.equal(storageTray(mount).querySelectorAll('[data-free-manual-pick]').length, 0, 'a part of the list is not offered');
    const waiting = trayButton(mount, 'data-free-manual-go');
    assert.equal(waiting.getAttribute('aria-disabled'), 'true');
    assert.equal(storageTray(mount).querySelector('.btn-danger'), null);
    await storageTray(mount).fire('click', { target: waiting });
    await settle();
    assert.equal(posts(calls), 0);
    held.resolve(response({ error: 'Internal Server Error' }, 500));
    await settle();
    assert.match(storageTray(mount).textContent, /The whole list could not be read, so nothing can be discarded from here yet\./);
    assert.equal(trayButton(mount, 'data-free-manual-go'), null);
    assert.equal(storageTray(mount).querySelector('.btn-danger'), null);
    // A page that cannot be trusted (the total is not the pool's) starts the list over on Try again.
    routes.set(manualUrl({ cursor: 2, version: 'p1' }), manualPage([copyEntry('m2')], { total: 3 }));
    await pressTray(mount, 'data-free-retry-manual');
    await settle();
    assert.equal(storageTray(mount).querySelectorAll('[data-free-manual-pick]').length, 3, 'a failed request resumes where it stopped');
    assert.match(storageTray(mount).textContent, /0 of 3 chosen/);
    const wrong = openStorage(t);
    wrong.routes.set(storageUrl, response(readyBody(1, manualFree(3))));
    wrong.routes.set(itemsUrl('pool'), page('pool', [poolEntry('a')]));
    wrong.routes.set(manualUrl(), manualPage(copiesOf(2)));
    wrong.start();
    await settle();
    await pressTray(wrong.mount, 'data-free-manual-open');
    await settle();
    assert.match(storageTray(wrong.mount).textContent, /The whole list could not be read/, 'two copies for three offered is not the offer');
    assert.equal(trayButton(wrong.mount, 'data-free-manual-go'), null);
});

test('Storage flow counts every chosen copy, shown or not, in the capsule and the request', async (t) => {
    const items = copiesOf(30);
    const { mount, routes, document } = await openCopies(t, 30, { items });
    assert.equal(storageTray(mount).querySelectorAll('[data-free-manual-pick]').length, 25, 'a screenful at a time');
    assert.match(storageTray(mount).textContent, /Showing 25 of 30\. The 5 not shown keep their choice: none chosen\./);
    await pick(mount, 24);
    await pressTray(mount, 'data-free-more-manual');
    assert.equal(storageTray(mount).querySelectorAll('[data-free-manual-pick]').length, 30);
    await pick(mount, 29);
    await pick(mount, 0);
    assert.match(storageTray(mount).textContent, /3 of 30 chosen/);
    assert.equal(trayButton(mount, 'data-free-manual-go').textContent, 'Discard 3 copies');
    let body = null;
    routes.set(freeUrl, (_url, init) => { body = JSON.parse(init.body); return response(freeBody('running', { total: 30, pool: poolFacts(2), run: freeRun({ selected: 3 }), started: true }), 202); });
    const click = pressTray(mount, 'data-free-manual-go');
    await settle();
    assert.match(document.body.querySelector('[role="dialog"]').textContent, /This deletes 3 copies whole\./);
    await confirmDialog(document, 'ok');
    await click;
    await settle();
    assert.deepEqual(body.manual, ['m0', 'm24', 'm29'], 'in the list\'s order, every one chosen and no other');
});

test('Storage flow says a refused acknowledgement changed nothing, and an outdated plan takes the whole screen away', async (t) => {
    const { mount, routes, calls, toasts, document } = await openCopies(t, 2);
    await pick(mount, 0);
    routes.set(freeUrl, response({ ...readyBody(2, manualFree(2)), started: false, reason: 'loss_not_accepted' }, 202));
    let click = pressTray(mount, 'data-free-manual-go');
    await settle();
    await confirmDialog(document, 'ok');
    await click;
    await settle();
    assert.deepEqual(toasts(), ['Nothing was removed. The loss was not confirmed.']);
    assert.ok(trayButton(mount, 'data-free-manual-go'), 'the list is as it was');
    assert.match(storageTray(mount).textContent, /1 of 2 chosen/);
    assert.equal(posts(calls), 1);
    routes.set(freeUrl, response({ ...readyBody(2, manualFree(2)), started: false, reason: 'not_offered' }, 202));
    click = pressTray(mount, 'data-free-manual-go');
    await settle();
    await confirmDialog(document, 'ok');
    await click;
    await settle();
    assert.match(toasts().join(' '), /The list changed; check again\./);
    const tray = storageTray(mount).textContent;
    assert.match(tray, /The list changed; check again\. Nothing was removed\./);
    assert.equal(storageTray(mount).querySelectorAll('[data-free-manual-pick]').length, 0);
    assert.equal(trayButton(mount, 'data-free-manual-go'), null);
    assert.equal(storageTray(mount).querySelector('.btn-danger'), null);
    assert.ok(trayButton(mount, 'data-free-preview'));
    assert.equal(posts(calls), 2, 'never repeated on its own');
});

for (const [label, answer] of [['an unreadable 202 body', () => unreadable()], ['a 5xx answer', () => response({ error: 'Internal Server Error' }, 502)], ['a dropped connection', () => new Error('network down')]]) {
    test(`Storage flow treats ${label} for Discard as unknown, reads once and never sends it or its dialog again`, async (t) => {
        const { mount, routes, calls, toasts, document } = await openCopies(t, 2);
        await pick(mount, 0);
        routes.set(freeUrl, answer());
        routes.set(storageUrl, response(readyBody(2, manualFree(2))));
        const old = trayButton(mount, 'data-free-manual-go');
        const click = pressTray(mount, 'data-free-manual-go');
        await settle();
        await confirmDialog(document, 'ok');
        await click;
        await settle();
        // A press that reaches the replaced capsule opens no dialog and sends nothing.
        await storageTray(mount).fire('click', { target: old });
        await settle();
        assert.equal(document.body.querySelector('[role="dialog"]'), null, 'no second dialog for an answer that is unknown');
        assert.deepEqual(toasts(), ['The answer to Discard was lost; nothing is sent again.'], 'neither the server\'s words nor "nothing was removed"');
        const tray = storageTray(mount).textContent;
        assert.match(tray, /The answer to Discard was lost\. It may still start, so nothing is sent again from this list\. Check again for a new one\./);
        assert.doesNotMatch(tray, /Nothing was removed/);
        assert.equal(trayButton(mount, 'data-free-manual-go'), null);
        assert.equal(storageTray(mount).querySelector('.btn-danger'), null);
        assert.equal(storageTray(mount).querySelectorAll('[data-free-manual-pick]').filter((box) => !box.hasAttribute('disabled')).length, 0, 'no new choice');
        assert.ok(trayButton(mount, 'data-free-preview'));
        assert.equal(posts(calls), 1);
        assert.ok(calls.filter((url) => url === storageUrl).length >= 2, 'a read says what happened');
    });
}

test('Storage flow forgets the copies screen and its choices with the plan, and Done or Cancel closes it', async (t) => {
    const { mount, routes } = await openCopies(t, 2);
    await pick(mount, 0);
    routes.set(previewUrl, response(freeBody('ready', { plan: 'p2', ...manualFree(2), started: true }), 202));
    routes.set(itemsUrl('pool', { plan: 'p2' }), response({ plan: 'p2', view: 'pool', version: 'p2', total: 2, counts: {}, next_cursor: null, items: [poolEntry('a'), poolEntry('b')] }));
    // The server replaces the plan under the owner: the choice means nothing on the new list.
    routes.set(storageUrl, response(freeBody('ready', { plan: 'p2', ...manualFree(2) })));
    await mount.querySelector('[data-storage-door]').fire('click', { target: mount.querySelector('[data-storage-door]') });
    await settle();
    await pressTray(mount, 'data-free-preview').catch(() => {});
    await settle();
    const review = storageTray(mount).textContent;
    assert.match(review, /Remove 2 items/);
    assert.equal(storageTray(mount).querySelectorAll('[data-free-manual-pick]').length, 0, 'the screen is closed on a new plan');
    assert.ok(trayButton(mount, 'data-free-manual-open'), 'the door is there again, nothing chosen');
    routes.set(manualUrl({ plan: 'p2' }), response({ plan: 'p2', view: 'manual', version: 'p2', total: 2, counts: {}, next_cursor: null, items: copiesOf(2) }));
    await pressTray(mount, 'data-free-manual-open');
    await settle();
    assert.match(storageTray(mount).textContent, /0 of 2 chosen/, 'nothing carried over');
    await pressTray(mount, 'data-free-manual-back');
    await pressTray(mount, 'data-free-cancel');
    assert.equal(storageTray(mount).textContent, '');
    assert.equal(mount.querySelector('[data-storage-door]').hidden, false);
});

test('the copies-by-hand request carries its own form and an acknowledgement that is exactly true or false, never inferred', async (t) => {
    const { routes } = openStorage(t);
    const bodies = [];
    routes.set(freeUrl, (_url, init) => { bodies.push(JSON.parse(init.body)); return response(storageBody(), 202); });
    await apiClient.freeStorageManual('p1', ['a', 'b'], true);
    await apiClient.freeStorageManual('p1', ['a'], 'yes');
    await apiClient.freeStorageManual('p1', ['a']);
    assert.deepEqual(bodies, [
        { plan: 'p1', manual: ['a', 'b'], accept_loss: true },
        { plan: 'p1', manual: ['a'], accept_loss: false },
        { plan: 'p1', manual: ['a'], accept_loss: false },
    ]);
});

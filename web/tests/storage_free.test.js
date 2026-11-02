import assert from 'node:assert/strict';
import test from 'node:test';
import { EMPTY_LIST, KEPT_REASONS, MANUAL_ROWS, OUTDATES_PLAN, acceptPage, confirmCopy, doneSentences, doorOffered, freeSum, goLabel, manualConfirmCopy, manualFacts, manualKnown, manualOf, notStarted, resultSentences, savedGroups, sumComplete, trayHtml, trayState } from '../modules/storage_free.js';

// Text is escaped through document.createElement, as a browser's innerHTML does.
globalThis.document = globalThis.document || {
    createElement: () => ({
        set textContent(value) {
            this.innerHTML = String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        },
    }),
};

const GB = 1e9;
const NOW = new Date('2026-10-17T18:00:00Z');
const volume = { free_bytes: 148 * GB, total_bytes: 4000 * GB, observed_at: '2026-10-17T15:40:00Z' };
const group = (id, overrides = {}) => ({
    id, count: 1629, inside_wait: 1620, in_use: 11, unreadable: 0, bytes: 6.84 * GB, bytes_complete: true,
    items: [{ id: 'direct:cde295b6', name: 'Run the QA wave on branch…', bytes: 1.966 * GB }], ...overrides,
});
const entry = (id, overrides = {}) => ({
    id, group: 'work_folders', kind: 'work_folder', name: `Task ${id}`, path: `/data/state/headless_tasks/${id}`, task_id: id, bytes: 1 * GB, bytes_complete: true,
    eligible_after: '2026-11-03T08:00:00Z', reason: null, outcome: null, ...overrides,
});
const rows = () => [entry('a'), entry('b', { kind: 'working_copy', bytes: 2 * GB }), entry('c', { kind: 'environment', bytes: 0.5 * GB, eligible_after: null })];
const pool = (overrides = {}) => ({ count: 3, bytes: 3.5 * GB, bytes_complete: true, kinds: { work_folder: 1, working_copy: 1, environment: 1 },
    excluded: [{ reason: 'capture_unproven', count: 2 }, { reason: 'in_use', count: 1 }], ...overrides });
const free = (state, overrides = {}) => ({
    state, plan: 'a1b2', started_at: '2026-10-17T16:10:03Z', finished_at: null, done: 0, total: 0, groups: [], pool: pool(), run: null, error: null,
    ...overrides,
});
const both = () => [group('work_folders'), group('working_copies', { count: 284, inside_wait: 0, in_use: 3, bytes: 2.1 * GB,
    items: [{ id: 'plain:t9', name: 'Port the header', bytes: 2.6 * GB }] })];
const saved = () => [group('review_evidence', { count: 40, inside_wait: 0, in_use: 2, bytes: 43 * GB }),
    group('result_pins', { count: 3, inside_wait: 0, in_use: 0, bytes: 2e5, items: [] })];
const run = (overrides = {}) => ({
    groups: [], selected: 1601, done: 1601, removed: 1584, removed_bytes: 8.2 * GB, deleting: 3,
    deleting_bytes: 0.3 * GB, stuck: 0, failed: 0, not_started: 0, kept: [{ reason: 'results_pending', count: 11 }, { reason: 'in_use', count: 3 }],
    free_before: 140 * GB, free_after: 148 * GB, stopped: false, error: null, outcomes_ready: true, ...overrides,
});
/** A flow whose pool was read whole: every id chosen unless named. */
const read = (items = rows(), { chosen = items.map((item) => item.id), ...rest } = {}) => ({
    pool: { ...EMPTY_LIST(), items, total: items.length, complete: true, next: null, want: Infinity }, chosen: new Set(chosen), checked: new Set(), dismissed: false, ...rest,
});
const AVOIDED = /At least|Excluded|System data|Junk|Clean\b|Prune|%|[Pp]ackage/;

test('at rest the flow is empty and the door stands alone; every phase up to Done holds the door back', () => {
    assert.equal(trayHtml(free('idle', { plan: null, pool: null }), read()), '');
    assert.equal(trayState(free('idle')), 'rest');
    assert.deepEqual(['rest', 'empty', 'listing', 'ready', 'running', 'done', 'failed'].filter(doorOffered), ['rest', 'empty']);
    const listing = trayHtml(free('listing', { done: 1240, total: 1661, pool: pool({ bytes: 5.1 * GB, bytes_complete: false }) }), read());
    assert.match(listing, /Checking what can go · 1,240 of 1,661 folders · 5\.1 GB so far\. Nothing is removed\./);
    assert.match(listing, /data-free-stop[^>]*>Stop</);
    assert.doesNotMatch(listing, /type="checkbox"|data-free-go/, 'no list of technical groups while it lists');
    assert.match(trayHtml(free('listing', { done: 1, total: 2, pool: null, groups: [group('work_folders', { bytes: 3 * GB })] }), read()), /3\.0 GB so far/);
    const noSizeYet = trayHtml(free('listing', { done: 1, total: 2, pool: pool({ bytes: null, bytes_complete: false }) }), read());
    assert.match(noSizeYet, /Checking what can go · 1 of 2 folders\. Nothing is removed\./);
    assert.doesNotMatch(noSizeYet, /0 bytes|so far/, 'a pool with no size yet is never zero');
});

test('ready names the exact composition and its observed size, one Remove last, the details folded', () => {
    const ready = trayHtml(free('ready', { groups: [...both(), ...saved()] }), read(), 7, { now: NOW });
    assert.match(ready, /Ready to remove[\s\S]*3 items · 3\.5 GB counted/);
    assert.match(ready, /Leftovers of finished work whose results are saved: 1 work folder, 1 working copy, 1 temporary environment\./);
    assert.match(ready, /Freed space can differ from the sizes\./);
    assert.match(ready, /Temporary environments go whole, with whatever was installed in them\. Later work gets an empty one or says why it cannot; nothing is reinstalled by itself\. Freed space can differ from the sizes\./);
    assert.match(ready, /<summary[^>]*>Show all 3 and choose<\/summary>/);
    assert.match(ready, /<summary[^>]*>Not included, and why<\/summary>/);
    assert.match(ready, /data-free-cancel[^>]*>Cancel<[\s\S]*btn-primary[^>]*data-free-go[^>]*>Remove 3 items</);
    assert.equal((ready.match(/btn-danger/g) || []).length, 0, 'the ordinary action is no danger capsule');
    assert.ok(ready.lastIndexOf('data-free-go') > ready.lastIndexOf('data-free-cancel'));
    assert.doesNotMatch(ready, AVOIDED);
    const noEnvironment = trayHtml(free('ready', { pool: pool({ count: 2, bytes: 3 * GB, kinds: { work_folder: 1, working_copy: 1 } }) }), read(rows().slice(0, 2)));
    assert.doesNotMatch(noEnvironment, /temporary environment/, 'the sentence only when one is chosen');
});

test('the review never claims or sends a whole list it has not read: reading, failing and an incomplete pool', () => {
    const reading = trayHtml(free('ready'), { pool: { ...EMPTY_LIST(), items: rows().slice(0, 1), total: 3, loading: true, want: Infinity }, chosen: new Set(), checked: new Set() });
    assert.match(reading, /3 items · 3\.5 GB counted/, 'the pool’s own facts are shown while its ids are read');
    assert.match(reading, /Reading the list · 1 of 3/);
    assert.match(reading, /data-free-go[^>]*aria-disabled="true"[^>]*>Reading the list…</);
    assert.doesNotMatch(reading, /Remove 3 items|Show all 3/);
    const failed = trayHtml(free('ready'), { pool: { ...EMPTY_LIST(), items: rows().slice(0, 1), total: 3, error: 'read', want: Infinity }, chosen: new Set(), checked: new Set() });
    assert.match(failed, /data-tone="warn"[^>]*>The whole list could not be read, so nothing can be removed from here yet\./);
    assert.match(failed, /data-free-retry-pool[^>]*>Try again</);
    assert.doesNotMatch(failed, /data-free-go/);
    const outdated = trayHtml(free('ready'), read(rows(), { outdated: 'refused' }));
    assert.match(outdated, /The list changed; check again\. Nothing was removed\./);
    const readRefused = trayHtml(free('ready'), read(rows(), { outdated: 'read', postUnknown: 'Remove' }));
    assert.match(readRefused, /The list changed; check again\./);
    assert.doesNotMatch(readRefused, /Nothing was removed/, 'only the server\'s refusal of a send says nothing was removed');
    assert.match(outdated, /data-free-preview[^>]*>Check again</);
    assert.doesNotMatch(outdated, /data-free-go/);
    const lost = trayHtml(free('ready'), read(rows(), { postUnknown: true, confirming: true }));
    assert.match(lost, /The answer to Remove was lost\. Reading what the server says now; nothing is sent again\./);
    assert.match(lost, /data-free-go[^>]*aria-disabled="true"/);
    const noPool = trayHtml(free('ready', { pool: null, groups: both() }), read([]));
    assert.match(noPool, /The ordinary list is not available/);
    assert.doesNotMatch(noPool, /data-free-go|Free 8\.9/, 'groups never stand in for the pool');
    assert.deepEqual([trayState(free('ready', { pool: null, groups: [group('work_folders', { count: 0 })] })), trayState(free('ready', { pool: null, groups: both() }))], ['empty', 'ready']);
});

test('Show all carries every row’s real facts, the counts follow the choice, and a long list says what is not shown', () => {
    const ready = trayHtml(free('ready'), read(rows(), { chosen: ['b'] }), 7, { now: NOW });
    assert.match(ready, /1 of 3 chosen/);
    assert.match(ready, /Choose all 3[\s\S]*Choose none/);
    assert.equal((ready.match(/type="checkbox"/g) || []).length, 3);
    assert.equal((ready.match(/ checked/g) || []).length, 1);
    assert.match(ready, /1 item · 2\.0 GB counted/);
    assert.match(ready, /whose results are saved: 1 working copy\./, 'the line follows what is chosen');
    assert.match(ready, />Remove 1 item</);
    assert.match(ready, /Task a[\s\S]*1\.0 GB[\s\S]*Work folder · the app may clear this by itself from 3 Nov at \d\d:\d\d[\s\S]*\/data\/state\/headless_tasks\/a/);
    assert.match(ready, /Task c[\s\S]*Temporary environment · the app does not clear this by itself/);
    const none = trayHtml(free('ready'), read(rows(), { chosen: [] }));
    assert.match(none, /Nothing is chosen\./);
    assert.match(none, /Nothing selected/);
    assert.doesNotMatch(none, /data-free-go/);
    const many = Array.from({ length: 250 }, (_, index) => entry(`i${index}`));
    const long = trayHtml(free('ready', { pool: pool({ count: 250, kinds: { work_folder: 250 } }) }), read(many));
    assert.equal((long.match(/type="checkbox"/g) || []).length, 100);
    assert.match(long, /250 of 250 chosen/);
    assert.match(long, /Showing 100 of 250\. The 150 not shown keep their choice: 150 chosen, and Remove takes those too\./);
    // Rows off screen keep exactly their own choice: the line counts the chosen ones among them, the action all of them.
    const mixed = trayHtml(free('ready', { pool: pool({ count: 250 }) }), read(many, { chosen: many.map((item) => item.id).filter((id) => !['i3', 'i150', 'i200'].includes(id)) }));
    assert.match(mixed, /247 of 250 chosen/);
    assert.match(mixed, /The 150 not shown keep their choice: 148 chosen, and Remove takes those too\./);
    assert.match(mixed, />Remove 247 items</);
    const onlyShown = trayHtml(free('ready', { pool: pool({ count: 250 }) }), read(many, { chosen: ['i0', 'i99'] }));
    assert.match(onlyShown, /The 150 not shown keep their choice: none chosen\./);
    assert.match(onlyShown, />Remove 2 items</);
    assert.doesNotMatch(long + mixed + onlyShown, /stay as they are/, 'nothing implies that a chosen row off screen is kept');
    assert.match(long, /data-free-more-picks[^>]*>Show 100 more</);
    assert.equal((trayHtml(free('ready', { pool: pool({ count: 250 }) }), read(many, { poolShown: 300 })).match(/type="checkbox"/g) || []).length, 250);
});

test('an unknown or incomplete size is never zero: the row, the sum, the saved groups and the dialog say so', () => {
    const unknown = trayHtml(free('ready', { pool: pool({ count: 1, bytes: null, bytes_complete: false, kinds: { work_folder: 1 } }) }), read([entry('a', { bytes: null, bytes_complete: false })]));
    assert.match(unknown, /1 item · size unknown/);
    assert.match(unknown, /storage-free-size is-qualified"[^>]*>size unknown</);
    assert.match(unknown, />Remove 1 item</);
    assert.doesNotMatch(unknown, /0 bytes/);
    const partial = trayHtml(free('ready', { pool: pool({ count: 2, bytes: 1 * GB, bytes_complete: false }) }), read([entry('a'), entry('b', { bytes: null })]));
    assert.match(partial, /2 items · 1\.0 GB counted, some sizes unknown/);
    const incomplete = trayHtml(free('ready'), read([entry('a', { bytes_complete: false })]));
    assert.match(incomplete, /1\.0 GB \(some could not be read\)/);

    const pins = group('result_pins', { count: 4, inside_wait: 0, in_use: 0, bytes: null, bytes_complete: false, items: [] });
    const evidence = group('review_evidence', { bytes: 5.1 * GB, bytes_complete: false });
    const flow = (checked) => ({ ...read(), saved: true, checked: new Set(checked) });
    const alone = trayHtml(free('ready', { groups: [pins] }), flow(['result_pins']));
    assert.match(alone, /storage-free-size is-qualified"[^>]*>size unknown</);
    assert.match(alone, /btn-danger[^>]*data-free-saved-go[^>]*data-i18n-fmt="storage\.free\.saved\.go"[^>]*>Free, size unknown</, 'the groups are selected: the capsule stays');
    assert.doesNotMatch(alone, /Nothing selected|0 bytes/);
    const mixed = trayHtml(free('ready', { groups: [evidence, pins] }), flow(['review_evidence', 'result_pins']));
    assert.match(mixed, /5\.1 GB \(some could not be read\)/);
    assert.match(mixed, /data-free-saved-go[^>]*>Free 5\.1 GB or more</);
    assert.equal(freeSum(free('ready', { groups: [evidence, pins] })), 5.1 * GB);
    assert.equal(sumComplete(free('ready', { groups: [evidence, pins] })), false);
    assert.equal(sumComplete(free('ready', { groups: both() })), true);
    assert.equal(goLabel(free('ready', { groups: both() }), new Set(['work_folders'])), 'Free 6.8 GB');
    const copy = confirmCopy(free('ready', { groups: [evidence, pins] }), new Set(['review_evidence', 'result_pins']));
    assert.match(copy.body, /^This removes the checked folders for good; some of their sizes could not be read\. Saved work goes too:/);
    assert.deepEqual(copy.details.rows.map((row) => row.value), ['5.1 GB (some could not be read) · 1,629 reviewer folders', 'size unknown · 4 results']);
    assert.equal(copy.confirmLabel, 'Free 5.1 GB or more');
});

test('a page that would leave an item missing or counted twice is refused, a moved version says so', () => {
    const fresh = () => EMPTY_LIST();
    const page = (overrides = {}) => ({ plan: 'p', view: 'pool', version: 'p', total: 3, counts: {}, next_cursor: 2, items: [entry('a'), entry('b')], ...overrides });
    const scope = { plan: 'p', view: 'pool', expected: 3 };
    const list = fresh();
    assert.equal(acceptPage(list, page(), scope), '');
    assert.equal(list.complete, false);
    assert.equal(list.next, 2);
    assert.equal(acceptPage(list, page({ next_cursor: null, items: [entry('c')] }), scope), '');
    assert.equal(list.complete, true);
    assert.deepEqual(list.items.map((item) => item.id), ['a', 'b', 'c']);
    for (const [label, bad] of [
        ['another plan', page({ plan: 'q' })], ['another view', page({ view: 'excluded' })], ['no items', page({ items: null })],
        ['total differs from the pool', page({ total: 4 })], ['ends short', page({ next_cursor: null })],
        ['a cursor that does not move', page({ next_cursor: 0 })], ['an empty page that goes on', page({ items: [] })],
        ['a repeated id', page({ items: [entry('a'), entry('a')] })], ['an empty id', page({ items: [entry('')] })],
    ]) assert.equal(acceptPage(fresh(), bad, scope), 'incomplete', label);
    const second = fresh();
    acceptPage(second, page(), scope);
    assert.equal(acceptPage(second, page({ version: 'other', items: [entry('c')] }), scope), 'moved');
    const growing = fresh();
    acceptPage(growing, page(), scope);
    assert.equal(acceptPage(growing, page({ total: 5, items: [entry('c')] }), { plan: 'p', view: 'pool' }), 'incomplete');
});

test('counts of one read as one: in use, kept, deleting, stuck, failed and not started', () => {
    const one = trayHtml(free('ready', { groups: [group('review_evidence', { count: 2, inside_wait: 0, in_use: 1, unreadable: 1 })] }), { ...read(), saved: true });
    assert.match(one, /1 is in use and stays\. 1 stays: its task record could not be read\./);
    const many = trayHtml(free('ready', { groups: [group('review_evidence', { count: 5, inside_wait: 0, in_use: 2, unreadable: 3 })] }), { ...read(), saved: true });
    assert.match(many, /2 are in use and stay\. 3 stay: their task record could not be read\./);
    assert.deepEqual(doneSentences(free('done', { run: run({ removed: 1, removed_bytes: 1e9, deleting: 1, deleting_bytes: 1e6, stuck: 1, failed: 1, not_started: 1,
        kept: [{ reason: 'in_use', count: 1 }] }) })), [
        '1 could not be deleted.', '1 could not be deleted yet; the app retries at its next maintenance run.',
        'Removed 1 item.', 'Their counted size was 1.0 GB.', 'Free space could not be read.', '1 is still being deleted.', '1 was kept: still in use.',
        '1 was not started and stays as it was.', 'The next count will show the new total.',
    ]);
    assert.deepEqual(resultSentences(free('done', { run: run({ deleting: 0, failed: 0, stuck: 2, kept: [] }) })).problems,
        ['2 could not be deleted yet; the app retries at its next maintenance run.']);
});

test('an empty preview says so and the door stays; a pool of nothing still shows what stays and why', () => {
    const emptyPool = pool({ count: 0, bytes: 0, kinds: {}, excluded: [] });
    const empty = free('ready', { pool: emptyPool, groups: [group('work_folders', { count: 0, inside_wait: 0, in_use: 0, bytes: 0, items: [] })] });
    assert.equal(trayState(empty), 'empty');
    assert.match(trayHtml(empty, read([])), /Nothing temporary is waiting\.[\s\S]*Most of Ouroboros is kept on purpose\./);
    const staysOnly = free('ready', { pool: pool({ count: 0, bytes: 0, kinds: {}, excluded: [{ reason: 'in_use', count: 2 }] }) });
    assert.equal(trayState(staysOnly), 'ready');
    const html = trayHtml(staysOnly, read([]));
    assert.match(html, /Nothing ready to remove[\s\S]*Not included, and why[\s\S]*data-free-cancel[^>]*>Close</);
    assert.doesNotMatch(html, /data-free-go|Show all/);
});

test('running, done, stopped and failed read in the owner\'s words; the result leads with what went wrong and has one way out', () => {
    const running = trayHtml(free('running', { total: 1630, groups: both(), run: run({ selected: 1630, done: 412, removed: 0, removed_bytes: 0, deleting: 412, deleting_bytes: 2.1 * GB, kept: [] }) }), read());
    assert.match(running, /Removing · 412 of 1,630 · 0 bytes of their counted size gone so far\. You can leave this tab\. 412 are still being deleted\./);
    assert.match(running, /data-free-stop[^>]*>Stop</);
    assert.doesNotMatch(running, /btn-danger|type="checkbox"|data-free-go/);
    const done = trayHtml(free('done', { groups: both(), run: run() }), read(), 7, { volume, now: NOW });
    assert.doesNotMatch(done, /data-tone="warn"/, 'a kept item is a fact, not a failure');
    assert.match(done, /Removed 1,584 items\./);
    assert.match(done, /Their counted size was 8\.2 GB\. Free now: 148 GB, read today at \d\d:\d\d\. 3 are still being deleted\. 11 were kept: their results are not copied yet\. 3 were kept: still in use\. The next count will show the new total\./);
    assert.doesNotMatch(done, /freed|reclaimed|data-free-preview/, 'no promise of space, no second start before Done');
    assert.match(done, /data-free-done[^>]*>Done</);
    assert.deepEqual(resultSentences(free('done', { run: run({ removed: 1, removed_bytes: 1e9, deleting: 0, kept: [] }) }), volume, NOW).rest.slice(0, 2),
        ['Their counted size was 1.0 GB.', `Free now: 148 GB, read today at ${new Date(volume.observed_at).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}.`]);
    const problems = trayHtml(free('done', { run: run({ failed: 2, stuck: 1 }) }), read());
    assert.ok(problems.indexOf('data-tone="warn"') < problems.indexOf('Removed 1,584 items.'), 'what went wrong comes first');
    assert.match(problems, /2 could not be deleted\. 1 could not be deleted yet/);
    const stopped = trayHtml(free('stopped', { groups: both(), run: run({ stopped: true, removed: 2, not_started: 7, deleting: 2 }) }), read());
    assert.match(stopped, /Removed 2 items, then stopped\. The rest was left as it was\./);
    assert.match(stopped, /2 are still being deleted\./);
    assert.match(stopped, /7 were not started and stay as they were\./);
    assert.equal(trayState(free('stopped', { run: null })), 'rest', 'a stopped listing leaves nothing to report');
    assert.equal(trayState(free('done', { run: run() }), true), 'rest', 'Done dismisses the result');
    const failed = trayHtml(free('failed', { error: 'The sizes could not be read.', pool: null }), read());
    assert.match(failed, /data-tone="warn"[^>]*>Temporary space could not be listed: The sizes could not be read\. Nothing was removed\./);
    assert.match(failed, /data-free-preview[^>]*>Try again</);
    assert.equal(notStarted('preview_outdated'), 'Nothing was removed. The list changed; check again.');
    for (const reason of ['unknown_item', 'not_offered', 'unknown_group']) assert.equal(notStarted(reason), 'Nothing was removed. The list changed; check again.');
    assert.equal(notStarted('busy'), 'Nothing was removed. Something else is still running.');
    assert.equal(notStarted('selection_conflict'), 'Nothing was removed. Try again.');
    for (const html of [running, done, stopped, failed]) assert.doesNotMatch(html, AVOIDED);
});

test('a failed free run preserves partial removals and reports its error first', () => {
    const partial = trayHtml(free('failed', { run: run({ error: 'OSError' }) }), read());
    assert.match(partial, /data-tone="warn"[^>]*>Freeing stopped: OSError\./);
    assert.match(partial, /Removed 1,584 items\./);
    assert.ok(partial.indexOf('Freeing stopped') < partial.indexOf('Removed 1,584'));
    assert.match(partial, /3 are still being deleted\./);
    assert.doesNotMatch(partial, /Nothing was removed|could not be listed/);
    const running = trayHtml(free('running', { total: 1, run: run({ selected: 1, done: 1, removed: 1, removed_bytes: 1e9, deleting: 0, deleting_bytes: 0 }) }), read());
    assert.match(running, /1\.0 GB of their counted size gone so far/);
    assert.doesNotMatch(running, /still being deleted/);
});

test('each item keeps its own outcome and reason, in the server’s order; a lost plan keeps the counts and says so', () => {
    const outcomes = { ...EMPTY_LIST(), total: 4, version: 'v', want: 100, complete: true, items: [
        entry('a', { outcome: 'failed' }), entry('b', { outcome: 'kept', reason: 'changed', kind: 'working_copy' }),
        entry('c', { outcome: 'not_started', kind: 'environment' }), entry('d', { outcome: 'deleting' }), entry('e', { outcome: 'removed' }), entry('f', { outcome: 'stuck' }),
    ] };
    const html = trayHtml(free('done', { run: run({ selected: 6 }) }), { ...read(), outcomes });
    assert.match(html, /<summary[^>]*>Each item \(4\)<\/summary>/);
    assert.match(html, /data-outcome="failed"[\s\S]*Work folder · Could not be deleted\./);
    assert.match(html, /data-outcome="kept"[\s\S]*Working copy · Kept: changed after the check\./);
    assert.match(html, /data-outcome="not_started"[\s\S]*Temporary environment · Not started: left as it was\./);
    assert.match(html, /data-outcome="deleting"[\s\S]*Moved aside; still being deleted\./);
    assert.match(html, /data-outcome="stuck"[\s\S]*Not deleted yet; the app retries at its next maintenance run\./);
    assert.doesNotMatch(html, /data-free-more-outcomes/);
    const paged = trayHtml(free('done', { run: run() }), { ...read(), outcomes: { ...outcomes, total: 450, complete: false, next: 6 } });
    assert.match(paged, /Each item \(450\)/);
    assert.match(paged, /data-free-more-outcomes[^>]*>Show 100 more</);
    const reading = trayHtml(free('done', { run: run() }), { ...read(), outcomes: { ...EMPTY_LIST(), want: 100, loading: true } });
    assert.match(reading, /Each item \(1,601\)[\s\S]*Reading each item’s result…/);
    const broken = trayHtml(free('done', { run: run() }), { ...read(), outcomes: { ...EMPTY_LIST(), want: 100, error: 'read' } });
    assert.match(broken, /Each item’s result could not be read in full\.[\s\S]*data-free-retry-outcomes/);
    const lost = trayHtml(free('done', { run: run({ outcomes_ready: false }) }), read());
    assert.match(lost, /Each item’s result is not available for this run; the counts above stand\./);
    assert.doesNotMatch(lost, /Each item \(/);
});

test('what the pool leaves out is named by reason, item by item, and the saved groups are read-only until asked for', () => {
    const excluded = { ...EMPTY_LIST(), total: 3, complete: true, want: 100, items: [
        entry('x', { reason: 'capture_unproven', bytes: null, bytes_complete: false, kind: 'working_copy' }),
        entry('y', { reason: 'in_use', bytes: null }), entry('z.json', { reason: 'unreadable_record', bytes: null, kind: 'other' }),
    ] };
    const html = trayHtml(free('ready', { groups: [...both(), ...saved()] }), { ...read(), excluded });
    assert.match(html, /Can’t go right now[\s\S]*2: unique work not verified saved · 1: still in use/);
    const reasons = trayHtml(free('ready', { pool: pool({ excluded: [{ reason: 'unproven_excluded_inputs', count: 1 }, { reason: 'never_seen_before', count: 2 }] }) }), read());
    assert.match(reasons, /1: files only a patch rule calls disposable · 2: for another reason/, 'a reason this client lacks words for is still counted');
    assert.match(html, /data-free-excluded="capture_unproven"[\s\S]*Working copy · unique work not verified saved/);
    assert.match(html, /data-free-excluded="unreadable_record"[\s\S]*Other item · their task record could not be read[\s\S]*\/data\/state\/headless_tasks\/z\.json/);
    assert.doesNotMatch(html.split('data-free-excluded')[1], /size unknown/, 'an item the pool leaves out is not sized, which says nothing');
    assert.match(html, /Saved, unknown or undecided: this button never removes them/);
    assert.match(html, /data-kept-group="review_evidence"[\s\S]*40 reviewer folders\. Test records, logs and screenshots behind review verdicts\. Saved work\./);
    assert.doesNotMatch(html, /data-free-group|data-kept-group="work_folders"|data-kept-group="working_copies"/, 'no checkbox and no ordinary group here');
    assert.match(html, /data-free-saved[^>]*>Choose saved groups to remove…</);
    assert.match(html, /Projects you keep working in, results, history, memory, skills, services and the app itself are never part of this\./);
    assert.deepEqual(savedGroups(free('ready', { groups: [...both(), group('unknown_kind', { count: 2 }), group('lost_copies', { count: 0 })] })).map((row) => row.id), ['unknown_kind']);
    const paged = trayHtml(free('ready'), { ...read(), excluded: { ...excluded, total: 250, complete: false, next: 3 } });
    assert.match(paged, /data-free-more-excluded[^>]*>Show 100 more</);
    const reading = trayHtml(free('ready'), { ...read(), excluded: { ...EMPTY_LIST(), want: 100, loading: true } });
    assert.match(reading, /Reading the list…/);
    const failed = trayHtml(free('ready'), { ...read(), excluded: { ...EMPTY_LIST(), want: 100, error: 'read' } });
    assert.match(failed, /The list of what stays could not be read in full\.[\s\S]*data-free-retry-excluded/);
});

test('the saved groups: reached on purpose, nothing checked, a danger capsule last and the loss sentences', () => {
    const flow = (checked = []) => ({ ...read(), saved: true, checked: new Set(checked) });
    const groups = [...both(), ...saved()];
    const none = trayHtml(free('ready', { groups }), flow());
    assert.match(none, /Saved groups[\s\S]*These hold saved work, so nothing here starts checked\. Each says what goes\./);
    assert.equal((none.match(/type="checkbox"/g) || []).length, 2);
    assert.doesNotMatch(none, /data-free-group="work_folders"|data-free-group="working_copies"/, 'the pool covers those two');
    assert.doesNotMatch(none, / checked>/);
    assert.match(none, /Nothing selected/);
    assert.doesNotMatch(none, /btn-danger|data-free-go/);
    assert.match(none, /Reviewers’ saved files of finished tasks[\s\S]*43 GB[\s\S]*Their verdicts stay in the review results\./);
    assert.match(none, /Git marks of results never applied[\s\S]*can still be applied as a plain patch/);
    assert.match(none, /Gone for good\. Sizes are what the listed folders occupy; the space freed can be smaller\./);
    assert.match(none, /Largest items \(1 of 43\)[\s\S]*Freeing removes every checked group, not only these\.[\s\S]*Run the QA wave on branch…[\s\S]*2\.0 GB/);
    const chosenOne = trayHtml(free('ready', { groups }), flow(['review_evidence']));
    assert.equal((chosenOne.match(/btn-danger/g) || []).length, 1);
    assert.match(chosenOne, /data-free-saved-back[^>]*>Back<[\s\S]*btn-danger[^>]*data-free-saved-go[^>]*>Free 43 GB</);
    const lost = trayHtml(free('ready', { groups }), { ...flow(['review_evidence']), postUnknown: 'Free' });
    assert.match(lost, /The answer to Free was lost\./);
    assert.doesNotMatch(lost, /btn-danger/);
    assert.doesNotMatch(trayHtml(free('ready', { groups: both() }), flow()), /Saved groups/, 'no such groups: the review stays');
    const stale = trayHtml(free('ready', { groups }), { ...flow(['review_evidence']), outdated: 'refused' });
    assert.match(stale, /data-tone="warn"[^>]*>The list changed; check again\. Nothing was removed\./);
    assert.match(stale, /data-free-saved-back[^>]*>Back<[\s\S]*data-free-preview[^>]*>Check again</);
    assert.doesNotMatch(stale, /btn-danger|data-free-saved-go/, 'a refused plan offers nothing destructive');
});

test('the confirm dialog carries the sum, the loss sentence and one detail row per checked group', () => {
    const groups = [...both(), ...saved()];
    const copy = confirmCopy(free('ready', { groups }), new Set(['review_evidence', 'result_pins']));
    assert.equal(copy.title, 'Free temporary space?');
    assert.equal(copy.body, 'This removes 43 GB for good. Saved work goes too: Reviewers’ saved files of finished tasks; Git marks of results never applied.');
    assert.doesNotMatch(copy.body, /are kept/);
    assert.deepEqual(copy.details.rows, [
        { label: 'Reviewers’ saved files of finished tasks', value: '43 GB · 40 reviewer folders' },
        { label: 'Git marks of results never applied', value: '200 KB · 3 results' },
    ]);
    assert.deepEqual([copy.confirmLabel, copy.danger], ['Free 43 GB', true]);
    const one = confirmCopy(free('ready', { groups }), new Set(['result_pins']));
    assert.equal(one.details.rows.length, 1);
    assert.equal(one.confirmLabel, 'Free 200 KB');
    const ordinary = confirmCopy(free('ready', { groups }), new Set(['work_folders']));
    assert.match(ordinary.body, /Results, messages and the exact model history are kept\./);
});

test('names from the server are escaped text; titles and accessible names are fixed chrome: nothing minted per preview', () => {
    const hostile = [entry('a', { name: '<img src=x onerror=1>', path: '/tmp/"><script>' })];
    const html = trayHtml(free('ready', { pool: pool({ count: 1, kinds: { work_folder: 1 } }) }), read(hostile));
    assert.doesNotMatch(html, /<img|<script/);
    const attributes = (body) => [...body.matchAll(/\s(title|aria-label)="([^"]*)"/g)].map(([, name, value]) => `${name}=${value}`);
    assert.deepEqual(attributes(trayHtml(free('ready', { groups: both() }), read())), []);
    assert.deepEqual(attributes(trayHtml(free('done', { groups: both(), run: run() }), read())), []);
});

// Revision 4 of the backend contract (sections 2 and 3): every typed reason a page, a pool or a run's kept rows can carry.
const CONTRACT_REASONS = ['results_pending', 'in_use', 'busy', 'unreadable_record', 'restart', 'other', 'capture_unproven', 'retained_evidence',
    'undeclared_environment', 'owner_keep', 'changed', 'unproven_excluded_inputs', 'advanced_group'];

test('every reason the server can send has its own words; none falls through to "another reason"', () => {
    for (const reason of CONTRACT_REASONS.filter((name) => name !== 'other')) {
        assert.ok(KEPT_REASONS[reason], reason);
        assert.notEqual(KEPT_REASONS[reason], KEPT_REASONS.other, reason);
    }
    assert.deepEqual(Object.keys(KEPT_REASONS).sort(), [...CONTRACT_REASONS].sort(), 'the words name exactly the contract\'s reasons');
});

test('saved-group items the pool leaves out are counted by the saved groups, not under what cannot go', () => {
    const html = trayHtml(free('ready', { groups: saved(), pool: pool({ excluded: [{ reason: 'advanced_group', count: 43 }, { reason: 'in_use', count: 1 }] }) }), {
        ...read(), excluded: { ...EMPTY_LIST(), total: 44, complete: false, next: 1, want: 100, items: [entry('s', { group: 'review_evidence', kind: 'other', reason: 'advanced_group', bytes: null })] },
    });
    assert.match(html, /Can’t go right now[\s\S]*>1: still in use</);
    assert.match(html, /data-free-excluded="advanced_group"[\s\S]*Other item · saved work: only its saved group can remove it/);
    const savedOnly = trayHtml(free('ready', { groups: saved(), pool: pool({ excluded: [{ reason: 'advanced_group', count: 43 }] }) }), read());
    assert.doesNotMatch(savedOnly, /Can’t go right now/, 'nothing is blocked: only saved work is left out');
});

test('a list cut short after its first page still says it is being read', () => {
    const html = trayHtml(free('ready'), { pool: { ...EMPTY_LIST(), items: rows().slice(0, 2), total: 3, next: 2, want: Infinity }, chosen: new Set(), checked: new Set() });
    assert.match(html, /Reading the list · 2 of 3/);
});

test('while Remove is out or its answer is lost the review takes no new choice and offers no way to hide it', () => {
    const sending = trayHtml(free('ready'), read(rows(), { sending: true }));
    assert.match(sending, /data-free-go[^>]*aria-disabled="true"[^>]*>Sending…</);
    assert.match(sending, /data-free-cancel[^>]*aria-disabled="true"/);
    assert.match(sending, /data-free-all[^>]*aria-disabled="true"/);
    assert.equal((sending.match(/data-free-pick="\d+"[^>]*disabled/g) || []).length, 3);
    const confirming = trayHtml(free('ready'), read(rows(), { postUnknown: true, confirming: true }));
    assert.match(confirming, /The answer to Remove was lost\. Reading what the server says now; nothing is sent again\./);
    assert.match(confirming, /data-free-go[^>]*aria-disabled="true"[^>]*>Checking what happened…</);
    assert.doesNotMatch(confirming, /data-free-cancel/);
    const held = trayHtml(free('ready'), read(rows(), { postUnknown: true }));
    assert.match(held, /The answer to Remove was lost\. It may still start, so nothing is sent again from this list\. Check again for a new one\./);
    assert.match(held, /data-free-preview[^>]*>Check again</);
    assert.doesNotMatch(held, /data-free-go|data-free-cancel|Nothing was removed/);
    const savedLost = trayHtml(free('ready', { groups: saved() }), read(rows(), { saved: true, postUnknown: 'Free', checked: new Set(['result_pins']) }));
    assert.match(savedLost, /The answer to Free was lost\. It may still start, so nothing is sent again from this list\. Check again for a new one\./);
    assert.doesNotMatch(savedLost, /data-free-saved-back|data-free-saved-go/);
    assert.match(savedLost, /data-free-group="result_pins"[^>]*disabled/);
});

test('what happened to the owner\'s attempt is said beside whatever the server holds now', () => {
    const idle = { ...free('idle'), plan: null, pool: null };
    assert.match(trayHtml(idle, { notice: 'refused' }), /Nothing was removed: the list you reviewed had changed\./);
    assert.match(trayHtml(idle, { notice: 'read' }), /The list you were reviewing changed\./);
    assert.match(trayHtml(free('listing', { plan: null }), { notice: 'lostFree' }), /The answer to Free was lost and the list has changed since/);
    assert.match(trayHtml(free('listing', { plan: null }), { notice: 'lost' }), /The answer to Remove was lost and the list has changed since; whether it removed anything is not known here\. Nothing is sent again\.[\s\S]*Checking what can go/);
    assert.match(trayHtml(free('ready', { plan: 'p2' }), read(rows(), { notice: 'refused' })), /Nothing was removed: the list you reviewed had changed\.[\s\S]*Remove 3 items/);
    assert.equal(trayHtml(idle, { notice: 'refused', dismissed: true }), '', 'a dismissed flow says nothing');
    assert.equal(trayHtml(idle, {}), '');
});

// ---------------------------------------------------------------------------
// Finished copies not verified saved, discarded by hand (backend seam v2, synthetic responses)
// ---------------------------------------------------------------------------

const unsaved = (overrides = {}) => ({ not_proven_saved_count: 3, not_proven_saved_bytes: 4.2e6, not_proven_saved: [
    { path: 'notes/draft.md', bytes: 3e6 }, { path: '.cache/x.bin', bytes: 1e6 }, { path: 'a.txt', bytes: 2e5 }], omitted_count: 0, keeps_commits: false, ...overrides });
const copy = (id, overrides = {}) => entry(id, { group: 'working_copies', kind: 'working_copy', bytes: 2 * GB, eligible_after: null, reason: 'capture_unproven',
    availability: 'manual_discard', manual: unsaved(), ...overrides });
const copies = (n = 3) => Array.from({ length: n }, (_, index) => copy(`m${index}`));
const manualPool = (count = 3, overrides = {}) => pool({ manual: { count, bytes: count * 2 * GB, bytes_complete: true }, excluded: [{ reason: 'capture_unproven', count, manual_count: count }], ...overrides });
/** A flow on the copies screen whose list was read whole: nothing chosen unless named. */
const byHand = (items = copies(), { chosen = [], ...rest } = {}) => read(rows(), {
    manualOpen: true, manual: { ...EMPTY_LIST(), items, total: items.length, complete: true, next: null, want: Infinity }, manualChosen: new Set(chosen), manualShown: MANUAL_ROWS, ...rest,
});

test('the door to copies by hand names them from the pool\'s own fact; each reason row gives up only its own manual share', () => {
    const excluded = [{ reason: 'capture_unproven', count: 5, manual_count: 3 }, { reason: 'retained_evidence', count: 4 }, { reason: 'busy', count: 2, manual_count: 2 },
        { reason: 'in_use', count: 1 }, { reason: 'advanced_group', count: 6 }];
    const html = trayHtml(free('ready', { pool: pool({ manual: { count: 5, bytes: 9.5 * GB, bytes_complete: true }, excluded }) }), read());
    // 5 − 3 and not 5 − 5 (the global manual count); a row with no manual share keeps its whole count; a row that is all manual is gone.
    assert.match(html, /Can’t go right now[\s\S]*2: unique work not verified saved · 4: marked as evidence, or could not be read through · 1: still in use</);
    assert.doesNotMatch(html.match(/Can’t go right now[\s\S]*?<\/p>/)[0], /busy|saved work|5: |6: /);
    assert.match(html, /Finished copies not verified saved/);
    assert.match(html, /5 copies of finished work stay: some files in them are not verified saved\. 9\.5 GB counted\. You can discard them one by one\./);
    assert.match(html, /btn-ghost[^>]*data-free-manual-open[^>]*>Choose copies to discard…</);
    assert.doesNotMatch(html, /btn-danger|data-free-manual-go|type="checkbox"[^>]*data-free-manual-pick/, 'the door chooses nothing and discards nothing');
    assert.match(html, /Remove 3 items/, 'the ordinary pool is untouched');
    const one = trayHtml(free('ready', { pool: pool({ manual: { count: 1, bytes: null, bytes_complete: false }, excluded: [{ reason: 'capture_unproven', count: 1, manual_count: 1 }] }) }), read());
    assert.match(one, /1 copy of finished work stays: some files in it are not verified saved\. size unknown\./);
    assert.doesNotMatch(one, /Can’t go right now/, 'everything left out can be discarded by hand: nothing is blocked');
    const none = trayHtml(free('ready', { pool: pool({ manual: { count: 0, bytes: 0, bytes_complete: true } }) }), read());
    assert.doesNotMatch(none, /data-free-manual-open|Finished copies not verified saved/);
    assert.doesNotMatch(trayHtml(free('ready', { pool: pool() }), read()), /data-free-manual-open/, 'an older answer without the field offers nothing');
    assert.equal(manualOf(free('ready', { pool: pool() })), null);
    assert.equal(manualOf(free('ready', { pool: manualPool(2) })).count, 2);
    const held = trayHtml(free('ready', { pool: manualPool(2) }), read(rows(), { sending: true }));
    assert.match(held, /data-free-manual-open[^>]*aria-disabled="true"/);
});

test('a plan holding only copies by hand is still a review, and an item left out by hand never reads as "Can’t go"', () => {
    assert.equal(trayState(free('ready', { pool: pool({ count: 0, kinds: {}, excluded: [], manual: { count: 2, bytes: 4 * GB, bytes_complete: true } }) })), 'ready');
    assert.equal(trayState(free('ready', { pool: pool({ count: 0, kinds: {}, excluded: [] }) })), 'empty');
    const listed = { ...EMPTY_LIST(), total: 3, complete: true, want: 100, items: [
        copy('x'), entry('y', { reason: 'in_use', availability: 'none', bytes: null }), entry('z', { reason: 'advanced_group', availability: 'saved_group', kind: 'other', bytes: null }) ] };
    const html = trayHtml(free('ready', { pool: pool({ manual: { count: 1, bytes: 2 * GB, bytes_complete: true }, excluded: [
        { reason: 'capture_unproven', count: 1, manual_count: 1 }, { reason: 'in_use', count: 1 }, { reason: 'advanced_group', count: 1 }] }) }), { ...read(), excluded: listed });
    assert.match(html, /data-free-excluded="capture_unproven" data-availability="manual_discard"[\s\S]*Working copy · unique work not verified saved · can be discarded by hand/);
    assert.doesNotMatch(html.match(/data-free-excluded="in_use"[\s\S]*?<\/li>/)[0], /discarded by hand/);
    assert.doesNotMatch(html.match(/data-free-excluded="advanced_group"[\s\S]*?<\/li>/)[0], /discarded by hand/);
    assert.match(html, /Can’t go right now[\s\S]*?>1: still in use</);
});

test('the copies screen lists every copy unchosen, with what it would lose said by name and the whole folder going', () => {
    const items = [copy('m0'), copy('m1', { bytes: null, bytes_complete: false, manual: unsaved({ not_proven_saved_count: 61, not_proven_saved_bytes: 9e6, omitted_count: 11,
        not_proven_saved: Array.from({ length: 50 }, (_, index) => ({ path: `big/f${index}.bin`, bytes: 1e5 })) }) }),
    copy('m2', { manual: unsaved({ not_proven_saved_count: 0, not_proven_saved_bytes: 0, not_proven_saved: [], keeps_commits: true }) })];
    const html = trayHtml(free('ready', { pool: manualPool(3) }), byHand(items));
    assert.match(html, /<h3[^>]*>Copies not verified saved<\/h3>/);
    assert.doesNotMatch(html, /storage-tray-sum/, 'nothing chosen: no sum');
    assert.match(html, /Finished work copies the app’s own cleanup leaves alone, because some files in them are not verified saved\. Nothing here starts chosen\./);
    assert.match(html, /data-tone="warn"[^>]*>Discarding a copy deletes its whole folder, not only the files listed\. What is not verified saved can be lost for good\. Task results, history and commits on a branch that stays are kept\./);
    assert.equal((html.match(/type="checkbox"/g) || []).length, 3);
    assert.doesNotMatch(html, / checked/, 'nothing starts chosen');
    assert.match(html, /0 of 3 chosen/);
    assert.doesNotMatch(html, /Choose none/, 'nothing to clear yet');
    assert.match(trayHtml(free('ready', { pool: manualPool(3) }), byHand(items, { chosen: ['m0'] })), /1 of 3 chosen[\s\S]*data-free-manual-none[^>]*>Choose none</);
    assert.doesNotMatch(html, /Choose all/, 'there is no select-all on this screen');
    assert.match(html, /Nothing selected/);
    assert.doesNotMatch(html, /btn-danger|data-free-manual-go|data-free-go|data-free-saved-go/);
    assert.match(html, /Task m0[\s\S]*2\.0 GB[\s\S]*Working copy · the whole folder goes[\s\S]*\/data\/state\/headless_tasks\/m0[\s\S]*3 files not verified saved · 4\.2 MB\./);
    assert.match(html, /Show the 3 files[\s\S]*notes\/draft\.md[\s\S]*3\.0 MB[\s\S]*\.cache\/x\.bin/);
    assert.match(html, /Task m1[\s\S]*size unknown[\s\S]*61 files not verified saved · 9\.0 MB\./);
    assert.match(html, /Show the 50 largest of 61 files[\s\S]*11 more are not listed\./, 'the sample is disclosed as a sample');
    assert.match(html, /Task m2[\s\S]*No file in it is unsaved\. Its branch and the commits on it stay\./);
    assert.equal((html.match(/<summary[^>]*>Show the /g) || []).length, 2, 'a copy with nothing to list has no list');
    assert.doesNotMatch(html, /only copy|only version|unique|nothing else|will be lost/i, 'not verified saved is the claim, never that it is the only one');
    assert.doesNotMatch(html, AVOIDED);
    assert.match(html, /data-free-manual-back[^>]*>Back</);
    assert.match(html, /Gone for good\. Sizes are what the whole folders occupy; the space freed can be smaller\./);
});

test('what is chosen drives the count, the size and the capsule; a choice not drawn on screen still counts', () => {
    const many = Array.from({ length: 30 }, (_, index) => copy(`c${index}`));
    const html = trayHtml(free('ready', { pool: manualPool(30) }), byHand(many, { chosen: ['c0', 'c27'] }));
    assert.match(html, /storage-tray-sum[^>]*>2 chosen · 4\.0 GB counted</);
    assert.equal((html.match(/data-free-manual-pick="\d+"/g) || []).length, MANUAL_ROWS, 'a screenful at a time');
    assert.equal((html.match(/ checked/g) || []).length, 1, 'only the drawn choice is drawn checked');
    assert.match(html, /2 of 30 chosen/);
    assert.match(html, /Showing 25 of 30\. The 5 not shown keep their choice: 1 chosen, and Discard takes those too\./);
    assert.match(html, /data-free-more-manual[^>]*>Show 5 more</);
    assert.match(html, /btn-danger[^>]*data-free-manual-go[^>]*data-i18n-fmt="storage\.free\.manual\.go"[^>]*>Discard 2 copies</);
    const unseen = trayHtml(free('ready', { pool: manualPool(30) }), byHand(many, { chosen: ['c0'] }));
    assert.match(unseen, /Showing 25 of 30\. The 5 not shown keep their choice: none chosen\./);
    assert.match(unseen, />Discard 1 copy</);
    assert.deepEqual(manualFacts(byHand(many).manual, new Set(['c0', 'c27', 'not-in-the-list'])).rows.map((row) => row.id), ['c0', 'c27'], 'an id the list does not hold is no choice');
    const sized = manualFacts(byHand([copy('a'), copy('b', { bytes: null }), copy('c', { bytes_complete: false })]).manual, new Set(['a', 'b', 'c']));
    assert.equal(sized.bytes, 4 * GB);
    assert.equal(sized.whole, false);
    assert.match(trayHtml(free('ready', { pool: manualPool(3) }), byHand([copy('a'), copy('b', { bytes: null })], { chosen: ['a', 'b'] })), /2 chosen · 2\.0 GB counted, some sizes unknown/);
});

test('a copy is chosen only when the server offered it by hand and described its loss; the rest are shown, never sendable', () => {
    const odd = [copy('ok'), copy('no-facts', { manual: undefined }), copy('partial', { manual: { not_proven_saved_count: 2 } }), copy('ordinary', { availability: 'ordinary' }),
        copy('saved', { availability: 'saved_group' }), copy('none', { availability: 'none' })];
    assert.deepEqual(odd.map(manualKnown), [true, false, false, false, false, false]);
    const html = trayHtml(free('ready', { pool: manualPool(6) }), byHand(odd, { chosen: odd.map((item) => item.id) }));
    assert.equal((html.match(/data-free-manual-pick="\d+"[^>]*disabled/g) || []).length, 5, 'five rows cannot be chosen');
    assert.equal((html.match(/data-free-manual-pick="\d+"[^>]*checked/g) || []).length, 1, 'and their ids, even if present in the state, are not drawn chosen');
    assert.match(html, /What it would lose could not be read, so it cannot be chosen here\./);
    assert.deepEqual(manualFacts(byHand(odd).manual, new Set(odd.map((item) => item.id))).rows.map((row) => row.id), ['ok'], 'no unoffered id reaches a request');
    assert.match(html, />Discard 1 copy</);
});

test('the capsule waits for the whole list: reading, a failed page and an outdated plan leave nothing to press', () => {
    const half = (extra = {}) => ({ ...EMPTY_LIST(), items: copies(2), total: 3, next: 2, want: Infinity, ...extra });
    const reading = trayHtml(free('ready', { pool: manualPool(3) }), byHand(copies(2), { manual: half({ loading: true }), chosen: ['m0'] }));
    assert.match(reading, /Reading the list · 2 of 3/);
    assert.match(reading, /btn-default[^>]*data-free-manual-go[^>]*aria-disabled="true"[^>]*>Reading the list…</);
    assert.doesNotMatch(reading, /btn-danger|type="checkbox"|Discard 1 copy|storage-tray-sum/, 'a part of the list is no list to choose from');
    const failed = trayHtml(free('ready', { pool: manualPool(3) }), byHand(copies(2), { manual: half({ error: 'read' }), chosen: ['m0'] }));
    assert.match(failed, /data-tone="warn"[^>]*>The whole list could not be read, so nothing can be discarded from here yet\./);
    assert.match(failed, /data-free-retry-manual[^>]*>Try again</);
    assert.doesNotMatch(failed, /data-free-manual-go|btn-danger|type="checkbox"/);
    for (const [label, flow] of [['refused', { outdated: 'refused' }], ['read', { outdated: 'read' }]]) {
        const stale = trayHtml(free('ready', { pool: manualPool(3) }), byHand(copies(3), { chosen: ['m0'], ...flow }));
        assert.match(stale, /The list changed; check again\./, label);
        assert.match(stale, /data-free-manual-back[^>]*>Back<[\s\S]*data-free-preview[^>]*>Check again</, label);
        assert.doesNotMatch(stale, /btn-danger|data-free-manual-go|type="checkbox"/, label);
    }
    assert.match(trayHtml(free('ready', { pool: manualPool(3) }), byHand(copies(3), { chosen: ['m0'], outdated: 'refused' })), /Nothing was removed\./);
});

test('while Discard is out or its answer is lost the screen takes no new choice and says which answer was lost', () => {
    const sending = trayHtml(free('ready', { pool: manualPool(3) }), byHand(copies(3), { chosen: ['m0'], sending: true }));
    assert.match(sending, /btn-danger[^>]*data-free-manual-go[^>]*aria-disabled="true"[^>]*>Sending…</);
    assert.match(sending, /data-free-manual-back[^>]*aria-disabled="true"/);
    assert.match(sending, /data-free-manual-none[^>]*aria-disabled="true"/);
    assert.equal((sending.match(/data-free-manual-pick="\d+"[^>]*disabled/g) || []).length, 3);
    const confirming = trayHtml(free('ready', { pool: manualPool(3) }), byHand(copies(3), { chosen: ['m0'], postUnknown: 'Discard', confirming: true }));
    assert.match(confirming, /The answer to Discard was lost\. Reading what the server says now; nothing is sent again\./);
    assert.match(confirming, /Checking what happened…/);
    assert.doesNotMatch(confirming, /data-free-manual-back|btn-danger|data-free-manual-go|type="checkbox"/, 'a list nothing can be sent from is not shown');
    const held = trayHtml(free('ready', { pool: manualPool(3) }), byHand(copies(3), { chosen: ['m0'], postUnknown: 'Discard' }));
    assert.match(held, /The answer to Discard was lost\. It may still start, so nothing is sent again from this list\. Check again for a new one\./);
    assert.match(held, /data-free-preview[^>]*>Check again</);
    assert.doesNotMatch(held, /btn-danger|data-free-manual-go|data-free-manual-back|type="checkbox"/);
    assert.match(trayHtml(free('listing', { plan: null }), { notice: 'lostDiscard' }), /The answer to Discard was lost and the list has changed since; whether it removed anything is not known here\. Nothing is sent again\./);
});

test('the dialog says the whole copy goes with what is not verified saved in it, once per chosen copy', () => {
    const items = [copy('a'), copy('b', { manual: unsaved({ not_proven_saved_count: 61, not_proven_saved_bytes: 9e6, omitted_count: 11 }) }), copy('c', { bytes: null, manual: unsaved({ not_proven_saved_count: 0, not_proven_saved_bytes: 0, not_proven_saved: [], keeps_commits: true }) }), copy('d')];
    const dialog = manualConfirmCopy(byHand(items).manual, new Set(['a', 'b', 'c']));
    assert.equal(dialog.title, 'Discard these copies?');
    assert.equal(dialog.body, 'This deletes 3 copies whole. 64 files in them are not verified saved (13 MB) and can be lost for good. Task results, history and commits on a branch that stays are kept.');
    assert.equal(dialog.danger, true);
    assert.equal(dialog.confirmLabel, 'Discard 3 copies');
    assert.deepEqual(dialog.details.rows, [
        { label: 'Task a', value: '2.0 GB · 3 files not verified saved' }, { label: 'Task b', value: '2.0 GB · 61 files not verified saved' },
        { label: 'Task c', value: 'size unknown · no file unsaved' }]);
    const single = manualConfirmCopy(byHand([copy('a', { manual: unsaved({ not_proven_saved_count: 0, not_proven_saved_bytes: 0, not_proven_saved: [], keeps_commits: true }) })]).manual, new Set(['a']));
    assert.equal(single.body, 'This deletes 1 copy whole. No file in them is unsaved. Task results, history and commits on a branch that stays are kept.');
    assert.equal(single.confirmLabel, 'Discard 1 copy');
});

test('a manual run reports in the existing outcome words; a copy whose work could not be saved first says it stayed whole', () => {
    const outcomes = { ...EMPTY_LIST(), total: 4, version: 'v', want: 100, complete: true, items: [
        copy('a', { outcome: 'removed', reason: null }), copy('b', { outcome: 'kept', reason: 'capture_unproven' }), copy('c', { outcome: 'kept', reason: 'changed' }),
        copy('d', { outcome: 'kept', reason: 'in_use' }), copy('e', { outcome: 'failed', reason: null }) ] };
    const html = trayHtml(free('done', { run: run({ selected: 5, removed: 1 }) }), { ...read(), outcomes });
    assert.match(html, /data-outcome="removed"[\s\S]*Working copy · Removed\./);
    assert.match(html, /data-outcome="kept"[\s\S]*Working copy · Kept: its work could not be saved first, so nothing was removed\./);
    assert.match(html, /Working copy · Kept: changed after the check\./);
    assert.match(html, /Working copy · Kept: still in use\./);
    assert.match(html, /data-outcome="failed"[\s\S]*Could not be deleted\./);
    // The pool's own kept item keeps the pool's words.
    const pooled = trayHtml(free('done', { run: run() }), { ...read(), outcomes: { ...outcomes, items: [entry('p', { outcome: 'kept', reason: 'capture_unproven', availability: 'none' })] } });
    assert.match(pooled, /Kept: unique work not verified saved\./);
    assert.equal(notStarted('loss_not_accepted'), 'Nothing was removed. The loss was not confirmed.');
    assert.equal(OUTDATES_PLAN.has('loss_not_accepted'), false, 'a refused acknowledgement leaves the list as it was');
    assert.ok(OUTDATES_PLAN.has('not_offered') && OUTDATES_PLAN.has('unknown_item'));
});

test('names and paths from the server are escaped text on the copies screen', () => {
    const hostile = copy('h', { name: '<img src=x onerror=1>', path: '/data/<b>x</b>', manual: unsaved({ not_proven_saved: [{ path: '"><script>.js', bytes: 5 }] }) });
    const html = trayHtml(free('ready', { pool: manualPool(1) }), byHand([hostile]));
    assert.doesNotMatch(html, /<img|<b>|<script/);
    assert.match(html, /&lt;img src=x onerror=1&gt;/);
});

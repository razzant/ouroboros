import assert from 'node:assert/strict';
import test from 'node:test';
import { CAUSES, EMPTY_LIST, MANUAL_ROWS, manualFacts, manualKnown, trayHtml } from '../modules/storage_free.js';
import { payloadConfirmCopy, payloadFacts, payloadKnown, payloadOf } from '../modules/storage_payloads.js';

globalThis.document = globalThis.document || {
    createElement: () => ({
        set textContent(value) {
            this.innerHTML = String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        },
    }),
};

const GB = 1e9;
const proposal = (id, overrides = {}) => ({
    id: `payload:${id}+payload-p1`, group: 'stored_payloads', kind: 'payload', name: `Compare sample catalogs ${id}`,
    path: `/data/task_results/artifacts/${id}`, task_id: id, bytes: 7.4 * GB, bytes_complete: true, eligible_after: null,
    reason: 'payload_offer', outcome: null, availability: 'manual_discard',
    manual: {
        not_proven_saved_count: 0, not_proven_saved_bytes: 0, not_proven_saved: [], omitted_count: 0, keeps_commits: false,
        payload_release: true, consequence: 'The sample snapshot cannot be measured again on the same bytes.',
        proposed_at: '2024-02-03T12:00:00Z', root_task_id: id,
        files: [{ path: 'delegated_runs/r/evidence/files/catalog.sqlite', bytes: 7.4 * GB, role: 'frozen copy of catalog.sqlite', loss: 'no rerun on the same bytes' }],
        retained: [{ role: 'report', path: 'delegated_runs/r/evidence/files/REPORT.md', bytes: 900 },
            { role: 'method', path: 'delegated_runs/r/evidence/files/measure.py', bytes: 400 }, { role: 'result', path: 'task_result', bytes: null }],
    },
    ...overrides,
});
const copy = (id) => ({ id: `plain:${id}`, group: 'working_copies', kind: 'working_copy', name: `Copy ${id}`, path: `/c/${id}`, task_id: id,
    bytes: 1 * GB, bytes_complete: true, eligible_after: null, reason: 'capture_unproven', outcome: null, availability: 'manual_discard',
    manual: { not_proven_saved_count: 1, not_proven_saved_bytes: 10, not_proven_saved: [{ path: 'a', bytes: 10 }], omitted_count: 0, keeps_commits: false } });
const free = (manual) => ({ state: 'ready', plan: 'p1', started_at: '2024-02-03T12:00:00Z', finished_at: null, done: 0, total: 0, groups: [],
    pool: { count: 0, bytes: 0, bytes_complete: true, kinds: {}, excluded: [{ reason: 'payload_offer', count: manual.payloads, manual_count: manual.payloads }], manual }, run: null, error: null });
const ui = (items, rest = {}) => ({ pool: EMPTY_LIST(), chosen: new Set(), checked: new Set(), dismissed: false, manualOpen: true, manualKind: 'payload',
    manual: { ...EMPTY_LIST(), items, total: items.length, complete: true, next: null, want: Infinity }, manualChosen: new Set(), payloadChosen: new Set(),
    manualShown: MANUAL_ROWS, ...rest });

test('a proposal is its own row in the decisions, apart from copies, and opens its own screen', () => {
    const door = trayHtml(free({ count: 2, bytes: 8.4 * GB, bytes_complete: true, retained: 0, payloads: 1 }), { ...ui([]), manualOpen: false });
    assert.match(door, /1 proposal to release heavy files of finished work[\s\S]*data-free-payload-open/);
    assert.match(door, /1 finished copy not verified saved[\s\S]*data-free-manual-open/);
});

test('nothing starts chosen; every file, its role and loss and every ref that stays are on the row', () => {
    const html = trayHtml(free({ count: 1, bytes: 7.4 * GB, bytes_complete: true, retained: 0, payloads: 1 }), ui([proposal('t1'), copy('c1')]));
    assert.match(html, /<h3[^>]*>Heavy files of finished work<\/h3>/);
    assert.doesNotMatch(html, / checked/);
    assert.match(html, /catalog\.sqlite[\s\S]*frozen copy of catalog\.sqlite · no rerun on the same bytes/);
    assert.match(html, /What stays[\s\S]*REPORT\.md[\s\S]*measure\.py[\s\S]*the task’s own answer/);
    assert.match(html, /cannot be measured again/);
    assert.doesNotMatch(html, /Copy c1/, 'copies stay on their own screen');
    assert.match(html, /data-free-manual-pick="0"/, 'the row keeps its index in the whole list');
});

test('a proposal is never a copy\'s choice, and its loss words are complete before it can be chosen', () => {
    const items = [proposal('t1'), copy('c1')];
    const list = { complete: true, items };
    assert.ok(payloadOf(items[0]) && payloadKnown(items[0]) && manualKnown(items[0]));
    assert.equal(manualFacts(list, new Set(items.map((item) => item.id))).count, 1, 'manual facts count the copy only');
    assert.deepEqual(payloadFacts(list, new Set([items[0].id])).files, 1);
    const unnamed = proposal('t2', { manual: { ...proposal('t2').manual, consequence: '' } });
    assert.ok(!payloadKnown(unnamed));
});

test('the confirm dialog names exactly what goes and what stays', () => {
    const items = [proposal('t1')];
    const copyText = payloadConfirmCopy({ complete: true, items }, new Set([items[0].id]));
    assert.equal(copyText.title, 'Release these files?');
    assert.match(copyText.body, /1 file \(7\.4 GB\) of 1 finished task for good[\s\S]*Reports, methods, results and history stay/);
    assert.equal(copyText.confirmLabel, 'Release 1 file');
    assert.ok(copyText.danger);
});

test('every hold of a proposal reads its own words', () => {
    for (const cause of ['payload_changed', 'payload_missing', 'payload_shared', 'payload_linked', 'payload_not_regular', 'retained_changed',
        'release_partial', 'release_pending', 'retained_by_other_offer', 'payload_protected', 'payload_unreadable', 'release_unexplained']) {
        assert.ok(CAUSES[cause], cause);
    }
});

// Dashboard › Costs, "By project and day" (cost_project_days.js) over GET /api/cost-breakdown?view=project_days: Main,
// each catalog project (closed ones too) and the records outside both (other rooms, none known) are separate rooms; every UTC day of
// [since, until) is a row, newest first, an `until` at midnight ending the day before; confirmed and estimated money
// stay apart, an estimate is said as one, calls with no outcome as open and calls that ended unreconciled as such; an
// unknown-only day is Unknown (never $0) while a measured zero is $0.00; the span's total adds the known money and counts
// the unknown prices; a failed view says why; the owner's room survives a catalog that cannot be read, a deleted one
// without its name, and a later choice drops an older answer; each phrase is said in the interface's language, again
// from the accepted view when the language changes.
import assert from 'node:assert/strict';
import test from 'node:test';

import { apiClient } from '../modules/api_client.js';
import { dayWords, mountCostProjectDays, projectDayRooms, projectDaysPresentation, spanDays } from '../modules/cost_project_days.js';
import { applyPayload, currentPayload, flushMisses, setMissTransport } from '../modules/i18n.js';
import { installCardDom } from './helpers/card_dom.js';

const flush = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const row = (day, extra = {}) => ({ day, settled_usd: 0, confirmed_usd: 0, estimated_usd: 0, unknown_unmetered: 0, non_final_rows: 0,
    accounting_open_rows: 0, open_rows: 0, unresolved_rows: 0, physical_calls: 0, subscription_sessions: 0,
    price_basis: extra.settled_usd === null ? 'unknown' : 'confirmed', ...extra });
const totalOf = (rows) => {
    const { day: _day, price_basis: _basis, ...counts } = row('');
    const measured = rows.filter((part) => part.settled_usd !== null);
    const total = Object.fromEntries(Object.keys(counts).map((key) => [key, rows.reduce((sum, part) => sum + (part[key] ?? 0), 0)]));
    return { ...total, ...(measured.length ? {} : { settled_usd: null, confirmed_usd: null, estimated_usd: null }),
        price_basis: measured.some((part) => part.price_basis === 'mixed')
            || new Set(measured.map((part) => part.price_basis)).size > 1 ? 'mixed' : (measured[0]?.price_basis ?? 'unknown') };
};
const view = (rows, extra = {}) => ({ available: true, selector: 'project:sample-project', days: 7, timezone: 'UTC',
    since: '2026-11-17T00:00:00Z', until: '2026-11-23T09:30:00Z', rows, total: totalOf(rows), integrity_degraded: false, ...extra });
const CATALOG = [
    { id: 'sample-project', name: 'Sample project', lifecycle: 'active' },
    { id: 'old', name: 'Settings', closed: { at: '2026-11-01T00:00:00Z', by: 'owner' }, lifecycle: 'active' },
    { id: 'gone', name: 'Gone', lifecycle: 'tombstoned' },
    { id: 'long', name: 'A room whose name is far longer than the narrowest phone can show in one line, on purpose', lifecycle: 'deleting' },
];

test('the rooms are Main, the catalog’s projects in its order (closed said so), then the records outside both', () => {
    assert.deepEqual(projectDayRooms(CATALOG).map((room) => [room.value, room.label, Boolean(room.closed)]), [
        ['main', 'Main', false],
        ['project:sample-project', 'Sample project', false],
        ['project:old', 'Settings', true],
        ['project:long', 'A room whose name is far longer than the narrowest phone can show in one line, on purpose', false],
        ['unassigned', 'Other or unknown room', false],
    ]);
    assert.deepEqual(projectDayRooms(null).map((room) => room.value), ['main', 'unassigned'], 'no catalog: Main and the unassigned still');
});

test('a day keeps confirmed and estimated apart, an unknown-only day is Unknown, a measured zero is $0.00', () => {
    assert.equal(dayWords(row('2026-11-20', { settled_usd: 12, confirmed_usd: 12 })).text, '$12.00');
    const mixed = dayWords(row('2026-11-21', { settled_usd: 32.5, confirmed_usd: 20, estimated_usd: 12.5, price_basis: 'mixed' }));
    assert.equal(mixed.text, '$32.50');
    assert.equal(mixed.detail, '$12.50 of it estimated');
    assert.equal(dayWords(row('2026-11-21', { settled_usd: 2, estimated_usd: 2, price_basis: 'estimated' })).detail, 'estimated');
    assert.equal(dayWords(row('2026-11-22', { physical_calls: 3 })).text, '$0.00', 'measured: 0 is a price');
    const unknown = dayWords(row('2026-11-22', { settled_usd: null, confirmed_usd: null, estimated_usd: null, unknown_unmetered: 2 }));
    assert.equal(unknown.settled, null);
    assert.equal(unknown.text, 'Unknown');
    assert.equal(unknown.detail, '2 without a full price');
    const partial = dayWords(row('2026-11-22', { settled_usd: 5, confirmed_usd: 5, unknown_unmetered: 1 }));
    assert.equal(partial.text, 'Tracked: $5.00');
    assert.equal(partial.detail, '1 without a full price');
    const partialSession = dayWords(row('2026-11-22', { settled_usd: 1.25, confirmed_usd: 1.25, unknown_unmetered: 1, subscription_sessions: 1 }));
    assert.equal(partialSession.text, 'Tracked: $1.25');
    assert.equal(partialSession.detail, '1 without a full price · 1 subscription session', 'partially priced sessions have a known subtotal and an unpriced remainder');
    const open = dayWords(row('2026-11-23', { settled_usd: 0, accounting_open_rows: 1, non_final_rows: 1, open_rows: 1 }));
    assert.equal(open.text, '$0.00', 'a measured zero stays one beside a call still open');
    assert.equal(open.detail, '1 call still open');
    assert.equal(open.open, true);
    const unknownOpen = dayWords(row('2026-11-23', { settled_usd: null, accounting_open_rows: 2, non_final_rows: 2, open_rows: 2 }));
    assert.equal(unknownOpen.text, 'Unknown');
    assert.equal(unknownOpen.detail, '2 calls still open', 'open and non-final rows overlap: the calls in flight are counted once');
    // A settled estimate keeps the ledger's accounting open, yet it is no call in flight: it is said as an estimate alone.
    const recorded = dayWords(row('2026-11-23', { settled_usd: 2, estimated_usd: 2, price_basis: 'estimated', accounting_open_rows: 1, non_final_rows: 1 }));
    assert.equal(recorded.detail, 'estimated');
    assert.equal(recorded.open, false);
    assert.match(dayWords(row('2026-11-23', { settled_usd: null, unknown_unmetered: 1, subscription_sessions: 1 })).detail, /1 subscription session$/);
});

test('estimate presence and tiny amounts survive even when a day or an estimated share is zero', () => {
    const zero = row('2026-11-22', { price_basis: 'estimated', accounting_open_rows: 2, open_rows: 1, subscription_sessions: 1 });
    const tiny = row('2026-11-23', { settled_usd: 0.00000025, confirmed_usd: 0.00000025, price_basis: 'mixed' });
    const shown = projectDaysPresentation(view([zero, tiny]));
    assert.equal(shown.days[0].words.text, '<$0.0001');
    assert.equal(shown.days[0].words.detail, '$0.00 of it estimated');
    assert.equal(shown.days[1].words.text, '$0.00');
    assert.equal(shown.days[1].words.detail, 'estimated · 1 call still open · 1 subscription session');
    assert.equal(shown.total.text, '<$0.0001');
    assert.match(shown.total.detail, /^\$0\.00 of it estimated/);
    assert.equal(projectDaysPresentation(view([zero])).total.detail, 'estimated · 1 call still open · 1 subscription session');
});

test('a measured $0 stays $0.00 beside records without a full price or still open; only null is Unknown', () => {
    const zeroUnpriced = dayWords(row('2026-11-23', { settled_usd: 0, unknown_unmetered: 2 }));
    assert.equal(zeroUnpriced.text, 'Tracked: $0.00', 'never "Cost not priced": the zero is measured');
    assert.equal(zeroUnpriced.detail, '2 without a full price');
    const zeroBoth = dayWords(row('2026-11-23', { settled_usd: 0, unknown_unmetered: 1, accounting_open_rows: 1, non_final_rows: 2, open_rows: 1 }));
    assert.equal(zeroBoth.text, 'Tracked: $0.00');
    assert.equal(zeroBoth.detail, '1 call still open · 1 without a full price', 'open and non-final rows overlap: not added up');
    const unknownBoth = dayWords(row('2026-11-23', { settled_usd: null, confirmed_usd: null, estimated_usd: null, unknown_unmetered: 3,
        accounting_open_rows: 1, open_rows: 1 }));
    assert.equal(unknownBoth.text, 'Unknown');
    assert.equal(unknownBoth.detail, '1 call still open · 3 without a full price');
    // A span whose known money adds up to zero keeps it too.
    const total = projectDaysPresentation(view([row('2026-11-23', { settled_usd: 0, accounting_open_rows: 1, open_rows: 1 }),
        row('2026-11-22', { settled_usd: null, unknown_unmetered: 1 })])).total;
    assert.equal(total.text, 'Tracked: $0.00');
    assert.equal(total.detail, '1 call still open · 1 without a full price');
});

test('a call that ended unreconciled is said apart from a call with no outcome, never as one still running', () => {
    // Its answer may have arrived before its accounting failed (usage_accounting._account_response): unresolved, not open.
    const ended = dayWords(row('2026-11-23', { settled_usd: null, confirmed_usd: null, estimated_usd: null, unresolved_rows: 1,
        unknown_unmetered: 1, accounting_open_rows: 1, non_final_rows: 1, physical_calls: 1 }));
    assert.equal(ended.text, 'Unknown');
    assert.equal(ended.detail, '1 call not reconciled · 1 without a full price');
    assert.equal(ended.open, false, 'no call of it is said to be running');
    const both = dayWords(row('2026-11-23', { settled_usd: 1.5, confirmed_usd: 1.5, open_rows: 2, unresolved_rows: 3,
        accounting_open_rows: 5, non_final_rows: 5 }));
    assert.equal(both.text, '$1.50');
    assert.equal(both.detail, '2 calls still open · 3 calls not reconciled');
    assert.equal(both.open, true);
    const total = projectDaysPresentation(view([row('2026-11-23', { settled_usd: 0, unresolved_rows: 1, accounting_open_rows: 1 }),
        row('2026-11-22', { settled_usd: 2, confirmed_usd: 2, open_rows: 1, accounting_open_rows: 1 })])).total;
    assert.equal(total.text, '$2.00');
    assert.equal(total.detail, '1 call still open · 1 call not reconciled');
});

test('the span is every UTC day of [since, until), newest first; an until at midnight ends the day before', () => {
    assert.deepEqual(spanDays('2026-11-20T00:00:00Z', '2026-11-23T09:30:00Z'), ['2026-11-23', '2026-11-22', '2026-11-21', '2026-11-20']);
    assert.deepEqual(spanDays('2026-11-20T00:00:00Z', '2026-11-23T00:00:00Z'), ['2026-11-22', '2026-11-21', '2026-11-20']);
    assert.deepEqual(spanDays('2026-11-20T00:00:00Z', '2026-11-20T00:00:00Z'), []);
    assert.deepEqual(spanDays('', '2026-11-23T09:30:00Z'), []);
});

test('every day of the span is a row, a day without records said so, the day still counted marked; bars against the largest day', () => {
    const shown = projectDaysPresentation(view([
        row('2026-11-23', { settled_usd: 1, confirmed_usd: 1, physical_calls: 2 }),
        row('2026-11-21', { settled_usd: 20, confirmed_usd: 15, estimated_usd: 5, price_basis: 'mixed', physical_calls: 9 }),
        row('2026-11-17', { settled_usd: 10, confirmed_usd: 10, physical_calls: 4 }),
    ]));
    assert.equal(shown.state, 'available');
    assert.deepEqual(shown.days.map((item) => item.day), ['2026-11-23', '2026-11-22', '2026-11-21', '2026-11-20', '2026-11-19', '2026-11-18', '2026-11-17']);
    assert.equal(shown.days[0].counting, true, 'until is not a midnight: its day is still being counted');
    assert.equal(shown.days.filter((item) => item.counting).length, 1);
    assert.equal(shown.days[1].quiet, true);
    assert.equal(Math.round(shown.days[2].confirmedShare), 75);
    assert.equal(Math.round(shown.days[2].estimatedShare), 25);
    assert.equal(Math.round(shown.days[6].confirmedShare), 50);
    assert.equal(shown.total.text, '$31.00');
    assert.equal(shown.total.calls, '15 model calls');
    assert.equal(shown.note, 'UTC days · only records with a known date.', 'never a claim of the whole history');
    const atMidnight = projectDaysPresentation(view([row('2026-11-22', { settled_usd: 1, confirmed_usd: 1 })], { until: '2026-11-23T00:00:00Z' }));
    assert.equal(atMidnight.days[0].day, '2026-11-22');
    assert.equal(atMidnight.days.some((item) => item.counting), false);
});

test('the total adds the known money and counts unknown prices; it is Unknown only when no day is priced', () => {
    const mixed = projectDaysPresentation(view([
        row('2026-11-22', { settled_usd: null, confirmed_usd: null, estimated_usd: null, unknown_unmetered: 1 }),
        row('2026-11-21', { settled_usd: 3, confirmed_usd: 3 }),
    ]));
    assert.equal(mixed.total.text, 'Tracked: $3.00', 'known money is neither zeroed nor hidden');
    assert.equal(mixed.total.detail, '1 without a full price');
    assert.equal(mixed.days.find((item) => item.day === '2026-11-22').words.text, 'Unknown');
    const none = projectDaysPresentation(view([row('2026-11-22', { settled_usd: null, unknown_unmetered: 4 })]));
    assert.equal(none.total.text, 'Unknown');
    assert.equal(none.total.detail, '4 without a full price');
    const zero = projectDaysPresentation(view([row('2026-11-22', { physical_calls: 1 })]));
    assert.equal(zero.total.text, '$0.00');
});

test('a refused or failed view says why; no records is its own state; the unassigned room says what it holds; degraded integrity is disclosed on empty and non-empty slices', () => {
    assert.match(projectDaysPresentation({ available: false, selector: 'project:x', error_code: 'unknown_selector' }).note, /could not be found in the project registry/);
    assert.equal(projectDaysPresentation({ available: false, selector: 'main', error_code: 'unknown_days' }).note, 'This span cannot be counted.');
    assert.match(projectDaysPresentation({ available: false, selector: 'main', error_code: 'ledger_unavailable' }).note, /usage ledger could not be read/);
    assert.match(projectDaysPresentation({ available: false, selector: 'main', error_code: 'project_source_unavailable' }).note, /Room assignments could not be read/);
    assert.equal(projectDaysPresentation(null).state, 'unavailable');
    assert.equal(projectDaysPresentation(null).days.length, 0, 'never a table of zeros');
    const empty = projectDaysPresentation(view([], { days: 30 }));
    assert.equal(empty.state, 'empty');
    assert.equal(empty.note, 'No records with a known date in the last 30 UTC days.');
    const degradedEmpty = projectDaysPresentation(view([], { days: 7, integrity_degraded: true }));
    assert.equal(degradedEmpty.state, 'empty');
    assert.equal(degradedEmpty.note, 'No available records with a known date in the last 7 UTC days. Some earlier ledger rows were quarantined, so this view may be incomplete.');
    const degradedShown = projectDaysPresentation(view([row('2026-11-20', { settled_usd: 4.5, confirmed_usd: 4.5 })], { integrity_degraded: true }));
    assert.equal(degradedShown.state, 'available');
    assert.equal(degradedShown.total.text, 'Tracked: $4.50', 'known sum stays intact under Tracked when ledger integrity is degraded');
    assert.equal(degradedShown.total.detail, 'ledger incomplete');
    assert.equal(degradedShown.days.find((item) => item.day === '2026-11-20').words.text, '$4.50', 'day amount stays concrete without repeating the quarantine caveat');
    assert.equal(degradedShown.note, 'UTC days · only records with a known date. Some earlier ledger rows were quarantined, so this view may be incomplete.');
    const degradedZero = projectDaysPresentation(view([row('2026-11-20', { settled_usd: 0, physical_calls: 1 })], { integrity_degraded: true }));
    assert.equal(degradedZero.total.text, 'Tracked: $0.00', 'measured zero stays $0.00 under Tracked when ledger integrity is degraded');
    assert.equal(degradedZero.total.detail, 'ledger incomplete');
    assert.equal(degradedZero.days.find((item) => item.day === '2026-11-20').words.text, '$0.00');
    const degradedUnknown = projectDaysPresentation(view([row('2026-11-20', { settled_usd: null, unknown_unmetered: 2 })], { integrity_degraded: true }));
    assert.equal(degradedUnknown.total.text, 'Unknown');
    assert.equal(degradedUnknown.total.detail, '2 without a full price · ledger incomplete');
    const unassigned = projectDaysPresentation(view([row('2026-11-20', { settled_usd: 1, confirmed_usd: 1 })], { selector: 'unassigned' }));
    assert.match(unassigned.note, /neither Main nor a project: another room, such as a place, or none known/);
    assert.match(unassigned.note, /A deleted project’s records are in the totals above, not in this room\./, 'only what holds: in the totals, not here');
    assert.doesNotMatch(projectDaysPresentation(view([row('2026-11-20', { settled_usd: 1 })])).note, /deleted|quarantined/);
});

function stand(t, { days = () => Promise.resolve(view([row('2026-11-21', { settled_usd: 3, confirmed_usd: 3 })])), rooms = () => Promise.resolve(CATALOG) } = {}) {
    const dom = installCardDom();
    t.after(() => dom.restore());
    const host = document.createElement('div');
    document.body.append(host);
    const asked = { days: [], rooms: 0 };
    const card = mountCostProjectDays(host, {
        load: (selector, span, signal) => { asked.days.push(`${selector}/${span}`); return days(selector, span, signal); },
        rooms: (signal) => { asked.rooms += 1; return rooms(signal); },
    });
    const select = host.querySelector('[data-cost-room]');
    const pick = (value) => { select.value = value; select.dispatch('change'); };
    return { host, card, asked, select, pick };
}

test('the card reads the catalog and the room on reload; a room or span choice reads the days alone', async (t) => {
    const { host, card, asked, select, pick } = stand(t);
    assert.deepEqual(asked, { days: [], rooms: 0 }, 'nothing is read before the page asks');
    await card.reload();
    assert.deepEqual(asked, { days: ['main/7'], rooms: 1 }, 'a week first');
    assert.equal(select.querySelectorAll('option').length, 5);
    assert.equal(select.querySelector('optgroup').getAttribute('label'), 'Projects');
    const closed = select.querySelector('option[value="project:old"]');
    assert.equal(closed.textContent, 'Settings (closed)');
    assert.ok(closed.hasAttribute('data-i18n-authored'));
    assert.match(host.querySelector('.costs-rooms-total').textContent, /\$3\.00/);
    pick('project:old');
    await flush();
    host.querySelector('[data-cost-room-span="30"]').click();
    await flush();
    assert.deepEqual(asked, { days: ['main/7', 'project:old/7', 'project:old/30'], rooms: 1 }, 'a month is the owner\u2019s explicit choice');
    assert.equal(host.querySelector('[data-cost-room-span="30"]').getAttribute('aria-pressed'), 'true');
});

test('the group of projects is named in the interface’s language, a project’s own name never', async (t) => {
    const prior = currentPayload();
    setMissTransport(() => null);
    // The memory is in force before the card paints; the page overlay (absent here) never reads a `label` attribute.
    applyPayload({ language: 'ru', english: false, entries: { Projects: { text: 'Проекты' }, 'Sample project': { text: 'Образец' } } });
    try {
        const { select, card } = stand(t);
        await card.reload();
        assert.equal(select.querySelector('optgroup').getAttribute('label'), 'Проекты');
        const own = select.querySelector('option[value="project:sample-project"]');
        assert.equal(own.textContent, 'Sample project');
        assert.ok(own.hasAttribute('data-i18n-authored'));
    } finally {
        await flushMisses();
        applyPayload(prior);
    }
});

test('a later room’s answer wins: the earlier one, arriving late, paints nothing', async (t) => {
    const waiting = new Map();
    const { host, card, pick } = stand(t, { days: (selector) => new Promise((resolve) => waiting.set(selector, resolve)) });
    const first = card.reload();
    await flush();
    waiting.get('main')(view([row('2026-11-21', { settled_usd: 3, confirmed_usd: 3 })], { selector: 'main' }));
    await first;
    pick('project:sample-project');
    assert.ok(host.querySelector('.costs-rooms-total').hidden);
    assert.equal(host.querySelector('.costs-rooms-days').innerHTML, '', 'no Main data below the new room');
    pick('project:old');
    await flush();
    waiting.get('project:old')(view([row('2026-11-21', { settled_usd: 7, confirmed_usd: 7 })], { selector: 'project:old' }));
    await flush();
    waiting.get('project:sample-project')(view([row('2026-11-21', { settled_usd: 99, confirmed_usd: 99 })], { selector: 'project:sample-project' }));
    await flush();
    assert.match(host.querySelector('.costs-rooms-total').textContent, /\$7\.00/);
    assert.doesNotMatch(host.querySelector('.costs-rooms-total').textContent, /99/);
});

test('changing the span before the first catalog returns cancels only the figures; both results paint independently', async (t) => {
    const days = new Map();
    let finishRooms, catalogSignal;
    const { host, card, select, asked } = stand(t, {
        days: (_selector, span, signal) => new Promise((resolve) => days.set(span, { resolve, signal })),
        rooms: (signal) => new Promise((resolve) => { finishRooms = resolve; catalogSignal = signal; }),
    });
    const first = card.reload();
    host.querySelector('[data-cost-room-span="30"]').click();
    assert.ok(days.get(7).signal.aborted);
    assert.equal(catalogSignal.aborted, false);
    days.get(30).resolve(view([row('2026-11-21', { settled_usd: 15, confirmed_usd: 15 })], { selector: 'main', days: 30 }));
    await flush();
    assert.match(host.querySelector('.costs-rooms-total').textContent, /\$15\.00/, 'slow catalog does not delay the figures');
    assert.equal(select.querySelectorAll('option').length, 2);
    finishRooms(CATALOG);
    await flush();
    assert.equal(select.querySelectorAll('option').length, 5, 'late catalog is still usable');
    assert.equal(asked.rooms, 1);
    days.get(7).resolve(view([]));
    await first;
    assert.match(host.querySelector('.costs-rooms-total').textContent, /\$15\.00/, 'old span cannot paint');
});

test('a catalog that cannot be read keeps the owner’s room and the rooms last listed, and says so', async (t) => {
    let fail = false;
    const { host, card, select, pick, asked } = stand(t, { rooms: () => (fail ? Promise.reject(new Error('HTTP 503')) : Promise.resolve(CATALOG)) });
    await card.reload();
    pick('project:sample-project');
    await flush();
    fail = true;
    await card.reload();
    assert.equal(select.value, 'project:sample-project', 'never replaced by Main');
    assert.equal(select.querySelectorAll('option').length, 5);
    assert.equal(asked.days.at(-1), 'project:sample-project/7');
    assert.match(host.querySelector('[data-cost-room-catalog-note]').textContent, /project list could not be read/);
    assert.match(host.querySelector('.costs-rooms-total').textContent, /\$3\.00/, 'figures do not fail with the catalog');
    fail = false;
    await card.reload();
    assert.equal(host.querySelector('[data-cost-room-catalog-note]').textContent, '');
});

test('a project absent from a refreshed list stays the choice under a neutral label; leaving drops an answer still out', async (t) => {
    const prior = currentPayload();
    setMissTransport(() => null);
    const language = (payload) => { const kept = globalThis.document; globalThis.document = undefined; try { applyPayload(payload); } finally { globalThis.document = kept; } };
    t.after(async () => { await flushMisses(); language(prior); });
    let catalog = CATALOG;
    let release = null;
    const { host, card, select, pick, asked } = stand(t, { rooms: () => Promise.resolve(catalog) });
    await card.reload();
    pick('project:sample-project');
    await flush();
    catalog = CATALOG.filter((item) => item.id !== 'sample-project');
    await card.reload();
    assert.equal(select.value, 'project:sample-project', 'its days stay its own, never another room’s');
    assert.equal(asked.days.at(-1), 'project:sample-project/7');
    const kept = select.querySelector('option[value="project:sample-project"]');
    assert.equal(kept.textContent, 'Project not in list', 'neutral label: absence from list does not prove deletion and does not bring back its name');
    assert.doesNotMatch(kept.textContent, /deleted/i);
    assert.equal(kept.hasAttribute('data-i18n-authored'), false, 'the words are the interface’s, translated as such');
    assert.equal(select.textContent.includes('Sample project'), false);
    assert.equal(select.textContent.includes('Gone'), false, 'a tombstoned row is never offered');
    language({ language: 'ru', english: false, revision: 5, entries: { 'Project not in list': { text: 'Проект не в списке' } } });
    card.repaint();
    assert.equal(select.value, 'project:sample-project');
    assert.equal(select.querySelector('option[value="project:sample-project"]').textContent, 'Проект не в списке');
    assert.equal(select.textContent.includes('Sample project'), false);
    language(prior);
    const late = stand(t, { days: () => new Promise((resolve) => { release = resolve; }) });
    const pending = late.card.reload();
    await flush();
    assert.equal(late.host.querySelector('.costs-rooms-status').textContent, 'Counting…');
    assert.equal(late.card.leave(), true);
    release(view([]));
    await pending;
    assert.equal(late.host.querySelector('.costs-rooms-note').textContent, '', 'a left read paints nothing');
    assert.equal(late.host.querySelector('.costs-rooms-status').textContent, '');
    assert.ok(host);
});

test('a day with records but no price draws no bar: an empty track would read as $0', async (t) => {
    const { host, card } = stand(t, { days: () => Promise.resolve(view([
        row('2026-11-22', { settled_usd: null, confirmed_usd: null, estimated_usd: null, unknown_unmetered: 1 }),
        row('2026-11-21', { settled_usd: 0, physical_calls: 2 }),
    ])) });
    await card.reload();
    const days = host.querySelectorAll('.costs-room-day');
    const unknown = days.find((node) => node.textContent.includes('Unknown'));
    const zero = days.find((node) => node.textContent.includes('$0.00'));
    assert.equal(unknown.querySelector('.costs-room-bar'), null);
    assert.match(unknown.textContent, /1 without a full price/);
    assert.ok(zero.querySelector('.costs-room-bar'), 'a measured zero keeps its (empty) track');
});

test('each phrase and UTC date is formatted in the interface’s language; a language change says the accepted view again, asking nothing', async (t) => {
    const prior = currentPayload();
    setMissTransport(() => null);
    // The words alone: without a document i18n.js leaves the double's page unwalked (no overlay runs in this test).
    const language = (payload) => { const kept = globalThis.document; globalThis.document = undefined; try { applyPayload(payload); } finally { globalThis.document = kept; } };
    const dateOpts = { weekday: 'short', month: 'short', day: 'numeric', timeZone: 'UTC' };
    const enDate = new Intl.DateTimeFormat('en-US', dateOpts).format(new Date('2026-11-23T00:00:00Z'));
    const enGbDate = new Intl.DateTimeFormat('en-GB', dateOpts).format(new Date('2026-11-23T00:00:00Z'));
    const ruDate = new Intl.DateTimeFormat('ru', dateOpts).format(new Date('2026-11-23T00:00:00Z'));
    assert.notEqual(enDate, enGbDate, 'regional English tag must differ from en-US so region handling is tested');
    t.after(async () => { await flushMisses(); language(prior); });
    const accepted = view([
        row('2026-11-23', { settled_usd: 32.5, confirmed_usd: 20, estimated_usd: 12.5, price_basis: 'mixed', unknown_unmetered: 2,
            open_rows: 1, unresolved_rows: 3, physical_calls: 9 }),
        row('2026-11-21', { settled_usd: null, confirmed_usd: null, estimated_usd: null, unknown_unmetered: 1 }),
    ], { integrity_degraded: true });
    const { host, card, asked } = stand(t, { days: () => Promise.resolve(accepted) });
    await card.reload();
    const read = asked.days.length;
    const roomsRead = asked.rooms;
    const total = () => host.querySelector('.costs-rooms-total').textContent;
    assert.equal(total(), 'Tracked: $32.50 $12.50 of it estimated · 1 call still open · 3 calls not reconciled · 3 without a full price · ledger incomplete 9 model calls');
    assert.equal(host.querySelectorAll('.costs-room-date')[0].textContent, `${enDate} so far`);
    language({ language: 'en-GB', english: true, revision: 6, entries: {} });
    card.repaint();
    assert.equal(asked.days.length, read, 'regional UI switch asks nothing');
    assert.equal(asked.rooms, roomsRead);
    assert.equal(host.querySelectorAll('.costs-room-date')[0].textContent, `${enGbDate} so far`);
    language({ language: 'ru', english: false, revision: 7, entries: {
        'Tracked: {amount}': { text: 'Учтено: {amount}' },
        '{amount} of it estimated': { text: 'из них по оценке {amount}' },
        '{n} call still open': { text: '{n} вызов ещё открыт' },
        '{count} calls not reconciled': { text: '{count} вызова не сверены' },
        '{count} without a full price': { text: '{count} без полной цены' },
        '{count} model calls': { text: '{count} вызовов модели' },
        'ledger incomplete': { text: 'журнал неполный' },
        Unknown: { text: 'Неизвестно' }, 'No records': { text: 'Нет записей' }, 'so far': { text: 'пока' },
        'UTC days · only records with a known date.': { text: 'Дни по UTC · только записи с известной датой.' },
        'Some earlier ledger rows were quarantined, so this view may be incomplete.': { text: 'Часть ранних записей журнала была изолирована, поэтому данные могут быть неполными.' },
    } });
    card.repaint();  // what Costs does on `ouro:language-changed`
    assert.equal(asked.days.length, read, 'the accepted view is said again: nothing is asked for it');
    assert.equal(asked.rooms, roomsRead);
    assert.equal(host.querySelectorAll('.costs-room-date')[0].textContent, `${ruDate} пока`);
    assert.equal(total(), 'Учтено: $32.50 из них по оценке $12.50 · 1 вызов ещё открыт · 3 вызова не сверены · 3 без полной цены · журнал неполный 9 вызовов модели');
    const lines = host.querySelectorAll('.costs-room-day');
    assert.equal(lines.length, 7);
    assert.match(lines[0].textContent, /пока/);
    assert.match(lines[1].textContent, /Нет записей/);
    assert.match(lines[2].textContent, /Неизвестно/);
    assert.match(lines[2].textContent, /1 без полной цены/);
    assert.equal(
        host.querySelector('.costs-rooms-note').textContent,
        'Дни по UTC · только записи с известной датой. Часть ранних записей журнала была изолирована, поэтому данные могут быть неполными.',
    );
    for (const composed of ['.costs-rooms-total', '.costs-rooms-days', '.costs-rooms-note']) {
        assert.ok(host.querySelector(composed).hasAttribute('data-i18n-fmt'), `${composed}: composed here, so the overlay leaves it alone`);
    }
    assert.equal(lines.some((node) => /so far|No records|Unknown/.test(node.textContent)), false, 'no English word is left behind');
    // An unknown or malformed language tag falls back safely to English instead of the browser default locale.
    language({ language: 'xx-ZZ', english: false, revision: 8, entries: {} });
    card.repaint();
    assert.equal(asked.days.length, read);
    assert.equal(host.querySelectorAll('.costs-room-date')[0].textContent, `${enDate} so far`);
    language({ language: 'not_a_locale', english: false, revision: 9, entries: {} });
    card.repaint();
    assert.equal(asked.days.length, read);
    assert.equal(host.querySelectorAll('.costs-room-date')[0].textContent, `${enDate} so far`);
    // A choice clears its figures until its own answer: a language change in between says nothing old.
    let answer;
    const waiting = stand(t, { days: () => new Promise((resolve) => { answer = resolve; }) });
    const first = waiting.card.reload();
    await flush();
    answer(accepted);
    await first;
    waiting.pick('unassigned');
    waiting.card.repaint();
    assert.equal(waiting.host.querySelector('.costs-rooms-days').innerHTML, '', 'the cleared room shows no earlier figures');
});

test('the card asks the breakdown for its room view alone: a room or span choice never reads the page’s totals again', async (t) => {
    const dom = installCardDom();
    t.after(() => dom.restore());
    const prior = globalThis.fetch;
    const urls = [];
    globalThis.fetch = async (url) => {
        urls.push(String(url));
        const days = String(url).startsWith('/api/cost-breakdown') ? { project_days: view([row('2026-11-21', { settled_usd: 3, confirmed_usd: 3 })]) } : null;
        return { ok: true, status: 200, json: async () => structuredClone(days || { projects: CATALOG }) };
    };
    t.after(() => { globalThis.fetch = prior; });
    const host = document.createElement('div');
    document.body.append(host);
    const card = mountCostProjectDays(host);  // the module's own readers
    await card.reload();
    const select = host.querySelector('[data-cost-room]');
    select.value = 'project:sample-project';
    select.dispatch('change');
    await flush();
    host.querySelector('[data-cost-room-span="30"]').click();
    await flush();
    const spend = urls.filter((url) => url.startsWith('/api/cost-breakdown'));
    assert.deepEqual(spend, [
        '/api/cost-breakdown?view=project_days&project_days=main&days=7',
        '/api/cost-breakdown?view=project_days&project_days=project%3Asample-project&days=7',
        '/api/cost-breakdown?view=project_days&project_days=project%3Asample-project&days=30',
    ]);
    assert.match(host.querySelector('.costs-rooms-total').textContent, /\$3\.00/);
    assert.equal((await apiClient.costProjectDays('unassigned', 7)).project_days.available, true);
});

test('on the Costs page a language change says the room’s accepted figures again; nothing is read for it', async (t) => {
    const { setup } = await import('./dashboard_dom_fixture.js');
    const { initCosts } = await import('../modules/costs.js');
    const prior = currentPayload();
    setMissTransport(() => null);
    const language = (payload) => { const kept = globalThis.document; globalThis.document = undefined; try { applyPayload(payload); } finally { globalThis.document = kept; } };
    t.after(async () => { await flushMisses(); language(prior); });
    const dateOpts = { weekday: 'short', month: 'short', day: 'numeric', timeZone: 'UTC' };
    const enDate = new Intl.DateTimeFormat('en-US', dateOpts).format(new Date('2026-11-21T00:00:00Z'));
    const enGbDate = new Intl.DateTimeFormat('en-GB', dateOpts).format(new Date('2026-11-21T00:00:00Z'));
    const ruDate = new Intl.DateTimeFormat('ru', dateOpts).format(new Date('2026-11-21T00:00:00Z'));
    const env = setup(t);
    const room = '/api/cost-breakdown?view=project_days&project_days=main&days=7';
    const answer = { project_days: view([row('2026-11-21', { settled_usd: 3, confirmed_usd: 3, open_rows: 1 })], { selector: 'main' }) };
    env.routes.set(room, { ok: true, status: 200, json: async () => structuredClone(answer) });
    const page = initCosts({ mount: env.mount, state: { activePage: 'dashboard' } });
    t.after(() => page.destroy());
    env.window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'costs' } }));
    await flush();
    const reads = () => env.calls.filter((url) => url === room).length;
    assert.equal(reads(), 1);
    const total = () => env.mount.querySelector('.costs-rooms-total').textContent;
    const dates = () => env.mount.querySelectorAll('.costs-room-date').map((node) => node.textContent);
    assert.match(total(), /\$3\.00\s*1 call still open/);
    assert.ok(dates().includes(enDate));
    language({ language: 'en-GB', english: true, revision: 8, entries: {} });
    env.window.dispatchEvent(new CustomEvent('ouro:language-changed'));
    assert.ok(dates().includes(enGbDate));
    assert.equal(reads(), 1, 'a regional language change asks nothing');
    language({ language: 'ru', english: false, revision: 9, entries: { '{n} call still open': { text: '{n} вызов ещё открыт' } } });
    env.window.dispatchEvent(new CustomEvent('ouro:language-changed'));
    assert.match(total(), /\$3\.00\s*1 вызов ещё открыт/);
    assert.ok(dates().includes(ruDate));
    assert.equal(reads(), 1, 'a language change asks nothing');
});

// Dashboard › Logs is data first: a hidden panel keeps every entry current and writes no node, reads no history;
// shown, it catches up newest first a chunk per frame, Raw is the received event written only when opened, and an
// open Timeline grows in place. Runs the real module on the shared dashboard DOM fixture.
import assert from 'node:assert/strict';
import test from 'node:test';

import { initLogs } from '../modules/logs.js';
import { setup } from './dashboard_dom_fixture.js';

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const logUrl = (name) => `/api/logs/${name}?view=compact&limit=150`;
const settle = () => new Promise((resolve) => setImmediate(resolve));
const STREAMS = ['events', 'tools', 'progress', 'supervisor'];

function mountLogs(t, { page = 'dashboard', history = {} } = {}) {
    const fixture = setup(t);
    for (const name of STREAMS) fixture.routes.set(logUrl(name), response({ entries: history[name] || [] }));
    const state = { activePage: page, dashboardActiveSubtab: 'logs' };
    initLogs({ mount: fixture.mount, ws: fixture.ws, state });
    const list = fixture.mount.querySelector('#log-entries');
    return {
        ...fixture, state, list,
        emit: (data) => fixture.ws.emit('log', { data }),
        show: () => {
            state.activePage = 'dashboard';
            fixture.window.dispatchEvent({ type: 'ouro:page-shown', detail: { page: 'dashboard' } });
        },
        card: (id) => list.children.find((node) => node.dataset.taskGroup === id),
        click: (target) => list.dispatchEvent({ type: 'click', target }),
        toggle: (details, open) => { details.open = open; list.dispatchEvent({ type: 'toggle', target: details }); },
    };
}

const at = (second) => `2026-11-04T01:00:${String(second).padStart(2, '0')}Z`;

test('Clear cancels a pending backfill without accepting its late rows or leaving Loading behind', async (t) => {
    const { mount, routes, ws } = setup(t);
    for (const name of STREAMS) routes.set(logUrl(name), response({ entries: [] }));
    const pending = [];
    routes.set(logUrl('events'), () => new Promise(resolve => pending.push(resolve)));
    initLogs({ mount, ws, state: { activePage: 'dashboard', dashboardActiveSubtab: 'logs' } });
    assert.equal(pending.length, 1);
    await mount.querySelector('#btn-clear-logs').fire('click');
    const status = mount.querySelector('.logs-history-status'), list = mount.querySelector('#log-entries');
    assert.match(status.textContent, /History read cancelled/);
    assert.equal(status.hidden, false);
    pending[0](response({ entries: [{ type: 'late_cleared_history', ts: at(1) }] }));
    await settle();
    assert.equal(list.children.length, 0);
    assert.match(status.textContent, /History read cancelled/);
    ws.emit('log', { data: { type: 'live_after_clear', ts: at(2) } });
    assert.equal(list.children.length, 1);
    ws.emit('open');
    pending[1](response({ entries: [] }));
    await settle();
    assert.equal(status.hidden, true, 'only a successful reread clears the missing-history status');
});

test('a hidden panel writes no node and reads no history; shown, it catches up and reads the owed history once', async (t) => {
    const earlier = { type: 'test_backfilled_event', ts: at(0), message: 'earlier' };
    const logs = mountLogs(t, { page: 'chat', history: { events: [earlier] } });
    await settle();
    assert.equal(logs.calls.length, 0, 'no history is read for a panel nobody sees');
    logs.emit({ type: 'tool_call_finished', tool: 'run_command', is_error: true, task_id: 'busy', ts: at(1) });
    logs.emit({ type: 'task_heartbeat', phase: 'running', task_id: 'busy', ts: at(2) });
    logs.emit({ type: 'test_standalone_event', ts: at(3) });
    logs.ws.emit('open');  // a reconnect while hidden owes the same one read
    await settle();
    assert.equal(logs.list.children.length, 0, 'hidden, the entries stay data');
    assert.equal(logs.calls.length, 0);

    logs.show();
    await settle();
    assert.equal(logs.calls.length, STREAMS.length, 'shown, the owed history is read once');
    assert.equal(logs.list.children.length, 3);
    const busy = logs.card('busy');
    assert.equal(busy.dataset.category, 'errors', 'the failure earned while hidden still files the card under Errors');
    assert.equal(busy.querySelector('[data-task-headline]').textContent, 'Still working');
    assert.equal(busy.querySelector('[data-task-count]').textContent, 'x2');
    assert.match(logs.list.textContent, /test_backfilled_event|earlier/i);

    // Away and back: nothing new, nothing read again.
    logs.state.activePage = 'settings';
    logs.emit({ type: 'task_heartbeat', phase: 'running', task_id: 'busy', ts: at(4) });
    assert.equal(busy.querySelector('[data-task-count]').textContent, 'x2', 'a hidden card is not redrawn');
    logs.show();
    await settle();
    assert.equal(busy.querySelector('[data-task-count]').textContent, 'x3');
    assert.equal(logs.calls.length, STREAMS.length);
});

test('the first showing after a burst draws the newest entries at once and the rest a chunk per frame, in order', async (t) => {
    const logs = mountLogs(t, { page: 'chat' });
    const frames = [];
    globalThis.requestAnimationFrame = (fn) => { frames.push(fn); return frames.length; };
    const runFrame = () => { const due = frames.splice(0); due.forEach((fn) => fn()); };
    for (let index = 0; index < 100; index += 1) logs.emit({ type: `burst_event_${index}`, ts: at(index % 60) });
    assert.equal(logs.list.children.length, 0);

    logs.show();
    // An event's type is printed only where its headline does not already say it (U028): the headline names it.
    const types = () => logs.list.children.map((node) => node.querySelector('.log-headline').textContent);
    assert.equal(logs.list.children.length, 40, 'the newest chunk is on screen at once');
    assert.equal(types().at(-1), 'burst_event_99');
    runFrame();
    assert.equal(logs.list.children.length, 80);
    logs.emit({ type: 'live_during_catch_up', ts: at(59) });  // a live entry meanwhile is newest
    runFrame();
    runFrame();
    assert.equal(logs.list.children.length, 101);
    assert.deepEqual(types(), [...Array.from({ length: 100 }, (_, index) => `burst_event_${index}`), 'live_during_catch_up']);
    await settle();  // the history owed since the panel was hidden has been read; nothing outlives the test
    for (let index = 0; index < 4 && frames.length; index += 1) runFrame();
});

test('Raw pins the received event while open, discloses a repeat, and releases its page when closed', async (t) => {
    const logs = mountLogs(t);
    await settle();
    const first = { type: 'deps_sync_ok', reason: 'boot', source: 'lock', ts: at(1), detail: { n: 1 } };
    logs.emit(first);
    const entry = logs.list.children[0];
    const raw = entry.querySelector('.log-raw');
    const button = entry.querySelector('.log-raw-toggle');
    assert.equal(raw.textContent, '', 'no Raw text exists before it is asked for');
    logs.click(button);
    await settle();
    assert.equal(raw.hidden, false);
    assert.equal(raw.textContent, JSON.stringify(first, null, 2));
    assert.equal(button.textContent, 'Hide raw');
    const repeat = { ...first, ts: at(2), detail: { n: 2 } };
    logs.emit(repeat);  // collapses into the same entry (x2)
    assert.equal(logs.list.children.length, 1);
    assert.equal(entry.querySelector('.log-repeat').textContent, 'x2');
    assert.equal(raw.textContent, JSON.stringify(first, null, 2), 'an open Raw keeps the source the owner asked for');
    assert.match(entry.querySelector('.log-raw-status').textContent, /entry changed/);
    logs.click(button);
    assert.equal(raw.hidden, true);
    assert.equal(raw.textContent, '', 'closed, the text is let go');

    logs.emit({ type: 'unserializable_event', ts: at(3), size: 1n });
    const odd = logs.list.children.at(-1);
    logs.click(odd.querySelector('.log-raw-toggle'));
    assert.match(odd.querySelector('.log-raw-status').textContent, /^Raw unavailable: /);
});

test('an open Timeline grows in place, keeps MAX rows, and closed lets its rows go', async (t) => {
    const logs = mountLogs(t);
    await settle();
    logs.emit({ type: 'task_started', task_id: 't', ts: at(1) });
    logs.emit({ type: 'tool_call_finished', tool: 'read_file', is_error: false, task_id: 't', ts: at(2) });
    const card = logs.card('t');
    const details = card.querySelector('details');
    const rows = () => card.querySelector('[data-task-timeline]').querySelectorAll('.log-task-event');
    assert.equal(rows().length, 0, 'a closed Timeline holds no rows');
    logs.toggle(details, true);
    const [one, two] = rows();
    assert.equal(rows().length, 2);
    logs.emit({ type: 'task_heartbeat', phase: 'running', task_id: 't', ts: at(3) });
    assert.equal(rows().length, 3);
    assert.equal(rows()[0], one, 'earlier rows keep their nodes');
    assert.equal(rows()[1], two);
    for (let second = 4; second < 40; second += 1) logs.emit({ type: 'task_heartbeat', phase: 'running', task_id: 't', ts: at(second) });
    assert.equal(rows().length, 30);
    assert.equal(rows().includes(one), false, 'the oldest row leaves past the bound');
    logs.toggle(details, false);
    assert.equal(rows().length, 0);
    // Opened by its summary, a Timeline draws its rows before the browser opens it.
    logs.click(details.querySelector('summary'));
    assert.equal(rows().length, 30);
});

test('a history owed while hidden merges by time: old rows go before the live ones, and a card keeps its newest state', async (t) => {
    const older = { type: 'test_older_row', ts: at(1) };
    const card = { type: 'task_heartbeat', phase: 'older-phase', task_id: 'busy', ts: at(2) };
    const logs = mountLogs(t, { page: 'chat', history: { events: [older, card] } });
    await settle();
    logs.emit({ type: 'tool_call_finished', tool: 'read_file', is_error: false, task_id: 'busy', ts: at(30) });
    logs.emit({ type: 'test_live_row', ts: at(31) });
    logs.show();
    await settle();
    const list = logs.list.children;
    assert.equal(list.length, 3);
    assert.equal(list[0].textContent.includes('older'), true, 'the oldest row leads');
    assert.equal(list[1].dataset.taskGroup, 'busy');
    assert.notEqual(list.at(-1).dataset.taskGroup, 'busy', 'the newest live row stays last');
    const busy = logs.card('busy');
    assert.equal(busy.querySelector('[data-task-count]').textContent, 'x2');
    assert.equal(busy.querySelector('[data-task-headline]').textContent, 'read_file finished', 'the older row does not take over the card');
    logs.toggle(busy.querySelector('details'), true);
    const rows = busy.querySelectorAll('.log-task-event').map((row) => row.querySelector('.log-headline').textContent);
    assert.equal(rows.at(-1), 'read_file finished', 'the timeline is in time order');
});

test('past the bound a merge lets the oldest go by time, never the entries that arrived live', async (t) => {
    const history = Array.from({ length: 150 }, (_, index) => ({ type: `old_a_${index}`, ts: `2026-11-03T0${index % 3}:00:${String(index % 60).padStart(2, '0')}.${index}Z` }));
    const logs = mountLogs(t, { page: 'chat', history: { events: history, tools: history.map((row) => ({ ...row, type: `old_b_${row.type}` })),
        progress: history.map((row) => ({ ...row, type: `old_c_${row.type}` })), supervisor: history.map((row) => ({ ...row, type: `old_d_${row.type}` })) } });
    await settle();
    logs.emit({ type: 'test_live_keeps_its_place', ts: at(40) });
    logs.show();
    await settle();
    assert.equal(logs.list.children.length, 500);
    assert.match(logs.list.children.at(-1).textContent, /test_live_keeps_its_place|Event/);
    assert.equal(logs.list.children.at(-1).querySelector('.log-raw-toggle') !== null, true);
    logs.click(logs.list.children.at(-1).querySelector('.log-raw-toggle'));
    await settle();
    assert.match(logs.list.children.at(-1).querySelector('.log-raw').textContent, /test_live_keeps_its_place/);
});

test('trimming hidden entries defers DOM removal until shown and remains bounded', async (t) => {
    const logs = mountLogs(t);
    await settle();
    for (let i = 0; i < 500; i++) logs.emit({ type: `before_${i}`, ts: at(i) });
    const first = logs.list.children[0];
    logs.state.activePage = 'chat';
    for (let i = 0; i < 600; i++) logs.emit({ type: `after_${i}`, ts: at(i) });
    assert.equal(logs.list.children.length, 500);
    assert.equal(logs.list.children[0], first, 'no hidden removal');
    logs.show();
    assert.equal(logs.list.children.length, 500);
    assert.equal(logs.list.children.includes(first), false);
});

// R1 review repairs (owner-visible order, reading place, honest compact view, bounded adoption).
const compactRef = (row, extra = {}) => ({ ...row, _raw: { ref: `ref-${row.ts}`, representation: 'stored',
    source: { v: 1, kind: 'log_row', placement: 'root', task_id: '', log: 'events', start: 0, length: 10, sha256: 'a'.repeat(64) },
    omitted_fields: [], omitted_field_count: 0, public_bytes: 10, ...extra } });

test('backfilled rows older than an open Timeline\'s rows take their places in time order, keeping the shown nodes', async (t) => {
    const logs = mountLogs(t);
    await settle();
    const row = (second, tool) => ({ type: 'tool_call_finished', tool, is_error: false, task_id: 't', ts: at(second) });
    logs.emit(row(30, 'third_tool'));
    logs.toggle(logs.card('t').querySelector('details'), true);
    const timeline = logs.card('t').querySelector('[data-task-timeline]');
    const [third] = timeline.querySelectorAll('.log-task-event');
    logs.routes.set(logUrl('events'), response({ entries: [row(10, 'first_tool'), row(20, 'second_tool')] }));
    logs.ws.emit('open');  // a reconnect reads the recent history again
    await settle();
    const rows = timeline.querySelectorAll('.log-task-event');
    assert.deepEqual(rows.map((node) => node.querySelector('.log-headline').textContent),
        ['first_tool finished', 'second_tool finished', 'third_tool finished']);
    assert.equal(rows[2], third, 'the row already shown keeps its node');
});

test('Raw on a received event shown again keeps its page, its scroll and its text node; only its download is rebuilt', async (t) => {
    const logs = mountLogs(t);
    await settle();
    logs.emit({ type: 'deps_sync_ok', reason: 'boot', ts: at(1), detail: 'x'.repeat(200) });
    const entry = logs.list.children[0];
    logs.click(entry.querySelector('.log-raw-toggle'));
    await settle();
    await settle();
    const pre = entry.querySelector('.log-raw');
    const page = pre.textContent;
    assert.match(page, /deps_sync_ok/);
    pre.scrollTop = 200;  // the reader scrolled inside the page
    let writes = 0;
    const own = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(pre), 'textContent');
    Object.defineProperty(pre, 'textContent', { get() { return own.get.call(this); }, set(value) { writes += 1; own.set.call(this, value); } });
    logs.state.activePage = 'settings';
    logs.window.dispatchEvent({ type: 'ouro:page-shown', detail: { page: 'settings' } });
    logs.show();
    await settle();
    await settle();
    assert.equal(pre.textContent, page);
    assert.equal(writes, 0, 'the page on screen is not written again');
    assert.equal(pre.scrollTop, 200, 'the reading place inside the page is kept');
    const download = entry.querySelector('.log-raw-download');
    assert.equal(download.hidden, false, 'the full download is offered again');
    assert.match(String(download.href || download.getAttribute('href') || ''), /^blob:/);
    // An explicit page move starts at that page's top: tests/test_logs_raw_browser.py (a worker-sized event, real engines).
});

test('a compact row names the fields its view left out and points at Raw; an adopted live twin keeps its own view', async (t) => {
    const logs = mountLogs(t, { history: { events: [compactRef({ type: 'llm_api_error', ts: at(1) },
        { omitted_fields: ['error', 'response_body'], omitted_field_count: 3 })] } });
    await settle();
    const [entry] = logs.list.children;
    const note = entry.querySelector('[data-log-omitted]');
    assert.ok(note, 'the entry says what is missing');
    assert.match(note.textContent, /error, response_body and 1 more/);
    assert.match(note.textContent, /Raw/);
    logs.emit({ type: 'task_heartbeat', phase: 'running', task_id: 'g', ts: at(2), note: 'live' });
    logs.toggle(logs.card('g').querySelector('details'), true);
    logs.routes.set(logUrl('events'), response({ entries: [compactRef({ type: 'task_heartbeat', phase: 'running', task_id: 'g', ts: at(2) },
        { omitted_fields: ['note'], omitted_field_count: 1 })] }));
    logs.ws.emit('open');
    await settle();
    const [row] = logs.card('g').querySelectorAll('.log-task-event');
    const rowNote = row.querySelector('[data-log-omitted]');
    assert.ok(!rowNote || rowNote.hidden || rowNote.textContent === '', 'the live twin was drawn whole: nothing is said missing');
});

// A context view checkpoint reads as its reconstruction; a receipt Raw could not use is said, never passed off as whole.
for (const [unavailable, reason, said] of [[[], '', null], [['source_refs', 'capsule_refs'], 'receipt_unavailable',
    / · receipt unavailable: 2 declared fields missing · /], [['fit'], 'receipt_mismatch', / · receipt does not match: 1 declared field missing · /]]) {
    test(`a reconstructed Raw page ${said ? `says its ${reason}` : 'that is whole says nothing missing'}`, async (t) => {
        const row = compactRef({ type: 'task_checkpoint', checkpoint_kind: 'context_view', ts: at(1) },
            { representation: 'reconstructed', public_bytes: null, omitted_fields: ['source_refs', 'capsule_refs', 'fit'], omitted_field_count: 3 });
        const logs = mountLogs(t, { history: { events: [row] } });
        await settle();
        const text = '{"checkpoint_kind":"context_view"}\n';
        const sha = 'b'.repeat(64);
        logs.routes.set(`/api/logs/events/raw?${new URLSearchParams({ ref: row._raw.ref, offset: 0, limit: 65536 })}`, response({
            source: row._raw.source, representation: 'reconstructed', sha256: sha, total_bytes: text.length, offset: 0,
            next_offset: text.length, has_more: false, text, unavailable_fields: unavailable, unavailable_reason: reason }));
        const [entry] = logs.list.children;
        logs.click(entry.querySelector('.log-raw-toggle'));
        await settle();
        await settle();
        const status = entry.querySelector('.log-raw-status').textContent;
        assert.match(status, /^0–\d+ of \d+ bytes · reconstructed/);
        if (said) assert.match(status, said);
        else assert.doesNotMatch(status, /missing/);
        assert.equal(entry.querySelector('.log-raw').textContent, text);
    });
}

test('the compact read\'s skipped damaged records are disclosed for that read only, beside a failed stream', async (t) => {
    const logs = mountLogs(t);
    logs.routes.set(logUrl('events'), response({ name: 'events', view: 'compact', entries: [compactRef({ type: 'ok_row', ts: at(1) })],
        gaps: ['malformed_jsonl', 'torn_archive_line'], skipped_rows: 2 }));
    logs.routes.set(logUrl('tools'), response({ name: 'tools', view: 'compact', entries: [], gaps: [], skipped_rows: 0 }));
    logs.routes.set(logUrl('supervisor'), response({ reason_code: 'log_source_unavailable' }, 503));
    logs.ws.emit('open');
    await settle();
    await settle();
    const status = logs.mount.querySelector('.logs-history-status');
    assert.equal(status.hidden, false);
    assert.match(status.textContent, /supervisor could not be loaded/);
    assert.match(status.textContent, /events 2 \(malformed line, torn archive line\)/);
    assert.match(status.textContent, /recent history read/);
    assert.doesNotMatch(status.textContent, /tools/, 'a clean read claims nothing');
    assert.equal(logs.list.children.length, 1, 'the valid row is shown');
    logs.routes.set(logUrl('events'), response({ entries: [], gaps: [], skipped_rows: 0 }));
    logs.routes.set(logUrl('supervisor'), response({ entries: [], gaps: [], skipped_rows: 0 }));
    logs.ws.emit('open');
    await settle();
    await settle();
    assert.equal(status.hidden, true, 'a later read without damage clears it');
});

test('adopting exact sources for live twins costs one pass over the entries, not one per backfilled row', async (t) => {
    const logs = mountLogs(t);
    await settle();
    let reads = 0;
    const counted = (row) => {
        const evt = { ...row };
        Object.defineProperty(evt, 'ts', { enumerable: true, get() { reads += 1; return row.ts; } });
        return evt;
    };
    const live = [];
    for (let group = 0; group < 120; group += 1) {
        for (let index = 0; index < 30; index += 1) {
            const second = String(index).padStart(2, '0');
            live.push({ type: 'task_heartbeat', phase: 'running', task_id: `g${group}`, ts: `2026-11-04T02:${String(group % 60).padStart(2, '0')}:${second}.${group}Z` });
        }
    }
    for (const row of live) logs.emit(counted(row));
    const twins = live.slice(-400).map((row) => compactRef(row));
    logs.routes.set(logUrl('events'), response({ entries: twins, gaps: [], skipped_rows: 0 }));
    reads = 0;
    logs.ws.emit('open');
    await settle();
    await settle();
    const entriesHeld = live.length;  // 120 groups x 30 rows
    assert.ok(reads <= entriesHeld * 3, `adoption read live keys ${reads} times for ${twins.length} twins over ${entriesHeld} held rows`);
});

// One compact time line (owner 17.10, Q7 A; docs/DESIGN.md "Project completion mirror"): a completion row's own
// time says when its task ended too. One clock when both facts fall in one minute, both only when they differ, a day
// only off today; the full instants (zone names where two minutes read alike) stay its tooltip and spoken text.
import assert from 'node:assert/strict';
import test from 'node:test';
import { formatMsgTime as bubbleClock } from '../modules/chat_activity.js';
import { decorateProjectRow } from '../modules/project_answer.js';
import { formatMsgTime, savedTerminalRowNote, terminalTimeNote } from '../modules/utils.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

const ended = (occurred_at) => ({ v: 1, source: 'executor_terminal', occurred_at, attempt: {} });
const at = (value, timeZoneName) => new Date(value).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZoneName });

function inZone(t, zone) {
    const prior = process.env.TZ;
    t.after(() => { if (prior === undefined) delete process.env.TZ; else process.env.TZ = prior; });
    process.env.TZ = zone;
}

test('one clock for one minute, both facts only when they differ, a day only off today', (t) => {
    inZone(t, 'UTC');
    const now = new Date('2026-10-17T19:00:00Z');
    const same = terminalTimeNote(ended('2026-10-17T18:46:10Z'), '2026-10-17T18:46:50Z', now);
    assert.equal(same.short, '18:46', 'the row reads like every other message: its own time');
    assert.equal(same.full, `Task ended ${at('2026-10-17T18:46:10Z')} · Notification added ${at('2026-10-17T18:46:50Z')}`);
    assert.equal(terminalTimeNote(ended('2026-10-17T18:41:00Z'), '2026-10-17T18:46:00Z', now).short, 'Ended 18:41 · added 18:46');
    assert.equal(terminalTimeNote(ended('2026-10-16T23:58:00Z'), '2026-10-17T00:03:00Z', now).short,
        'Ended Oct 16, 23:58 · added 00:03');
    assert.equal(terminalTimeNote(ended('2025-12-31T23:58:00Z'), '2026-01-01T00:03:00Z', now).short,
        'Ended Dec 31, 23:58 · added Jan 1, 00:03', 'a date is the short form, never the year');
    // Yesterday's single clock is the bubble's own wording.
    assert.equal(terminalTimeNote(ended('2026-10-16T18:46:10Z'), '2026-10-16T18:46:30Z', now).short, 'Yesterday, 18:46');
});

test('an unknown end still says so, and is never the notification time', (t) => {
    inZone(t, 'UTC');
    const now = new Date('2026-10-17T19:00:00Z');
    for (const fact of [{ source: 'unknown', occurred_at: null }, null, undefined, { source: 'executor_terminal', occurred_at: 'nope' }]) {
        const note = terminalTimeNote(fact, '2026-10-17T18:46:00Z', now);
        assert.equal(note.short, 'End time not recorded · added 18:46');
        assert.equal(note.full, `Task end time not recorded · Notification added ${at('2026-10-17T18:46:00Z')}`);
    }
    assert.deepEqual(terminalTimeNote(ended('2026-10-17T18:41:00Z'), '', now),
        { short: 'Ended 18:41', full: `Task ended ${at('2026-10-17T18:41:00Z')}` }, 'an unreadable row time adds nothing');
});

test('two different minutes that read alike across a DST fallback carry their zone names', (t) => {
    inZone(t, 'America/New_York');
    const now = new Date('2026-11-01T12:00:00Z');
    const [end, added] = ['2026-11-01T05:30:10Z', '2026-11-01T06:30:40Z'];
    const note = terminalTimeNote(ended(end), added, now);
    assert.equal(note.short, 'Ended 01:30 EDT · added 01:30 EST');
    assert.equal(note.full, `Task ended ${at(end, 'short')} · Notification added ${at(added, 'short')}`);
});

test('the bubble clock is one function, and the room timeline keeps the full line', (t) => {
    inZone(t, 'UTC');
    assert.equal(bubbleClock, formatMsgTime, 'chat_activity re-exports the shared clock');
    const row = { task_id: 'late', system_type: 'task_summary', status: 'failed', ts: '2026-09-26T19:21:43Z',
        terminal_time: ended('2026-09-24T18:18:27Z') };
    assert.equal(savedTerminalRowNote(row), terminalTimeNote(row.terminal_time, row.ts).full);
    assert.equal(savedTerminalRowNote({ ...row, ts: '2026-09-24T18:18:40Z' }), '', 'the same minute adds nothing');
});

test('a completion row writes its one time line into its own time, with no second note', (t) => {
    inZone(t, 'UTC');
    const env = installDom();
    try {
        for (const role of ['assistant', 'system']) {
            const bubble = document.createElement('div');
            bubble.innerHTML = '<div class="message"></div><div class="msg-time" title="old">old</div>';
            env.mount.append(bubble);
            const [end, addedAt] = ['2026-09-24T18:18:27Z', '2026-09-26T19:21:43Z'];
            decorateProjectRow(bubble, { role, projectId: 'p', projectName: 'Build', systemType: 'project_completion_summary',
                terminalTime: ended(end), addedAt, limitation: 'Windows is not verified.' });
            const time = bubble.querySelector('.msg-time');
            const note = terminalTimeNote(ended(end), addedAt);
            assert.equal(note.short, 'Ended Sep 24, 18:18 · added Sep 26, 19:21');
            assert.equal(time.title, note.full);
            assert.deepEqual(time.children.map((node) => [node.className, node.attributes.get('aria-hidden') || '', node.textContent]),
                [['', 'true', note.short], ['msg-time-full', '', note.full]]);
            assert.deepEqual(bubble.children.filter((node) => node.classList.contains('msg-provenance')).map((node) => node.className),
                role === 'assistant' ? ['msg-provenance project-answer-limitation'] : [], 'no second time note');
        }
    } finally { restoreDom(env.prior); }
});

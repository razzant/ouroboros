import assert from 'node:assert/strict';
import test from 'node:test';

// The DOM-free helper; the rendered rows are exercised by dashboard_read_states.test.js and the browser lane.
globalThis.document = globalThis.document || { createElement: () => ({}) };
const { elapsedWords, liveActivityRowHtml } = await import('../modules/activity.js');

test('a running time reads in words: seconds below a minute, then minutes, then hours and minutes', () => {
    for (const [seconds, words] of [
        [0, '0s'], [9, '9s'], [45, '45s'], [59.4, '59s'],
        [59.6, '1 min'], [60, '1 min'], [89, '1 min'], [90, '2 min'], [560, '9 min'], [1700, '28 min'],
        [3599, '1 h'], [3600, '1 h'], [5400, '1 h 30 min'], [7200, '2 h'], [7260, '2 h 1 min'],
    ]) assert.equal(elapsedWords(seconds), words, `${seconds} s`);
    assert.equal(elapsedWords('1700'), '28 min', 'a numeric string is a number');
    assert.equal(elapsedWords(-5), '0s', 'a clock that ran backwards is not a negative duration');
});

test('a duration the row does not carry is no words, never 0s', () => {
    for (const missing of [null, undefined, '', 'abc', NaN, Infinity]) assert.equal(elapsedWords(missing), '', String(missing));
});

test('a live turn row says its time in words and says nothing when it has no start', () => {
    const now = 1_700_000_000_000 + 5_400_000;
    assert.match(liveActivityRowHtml({ activity_id: 'turn', kind: 'direct_chat', phase: 'working', started_at: 1_700_000_000 }, now),
        /<span data-i18n-skip>Working<\/span> · 1 h 30 min<\/span>/);
    const unstarted = liveActivityRowHtml({ activity_id: 'turn', kind: 'direct_chat', phase: 'working' }, now);
    assert.doesNotMatch(unstarted, /· \d|0s/, 'no start time, no invented duration');
});

test('a live inline Presence row says its work, a recorded Stop among it, and offers a control only from its first durable write', () => {
    const now = 1_700_000_000_000;
    const running = { activity_id: 'turn', kind: 'presence', phase: 'working', cancelable: true, started_at: 1_700_000_000 };
    const work = { name: 'Community <research>', activity: 'working', confirmed: true };
    const named = liveActivityRowHtml(running, now, null, false, work);
    assert.match(named, /<span class="ui-row-title activity-name" data-i18n-authored>Community &lt;research&gt;<\/span>/);
    assert.match(named, /data-act="task-control" data-id="turn"/);
    assert.match(named, /<span data-i18n-skip>Working<\/span> · 0s<\/span>/, 'nothing beside the word while no Stop is recorded');
    const stopping = liveActivityRowHtml(running, now, null, false, { ...work,
        because: [{ code: 'stop_requested', text: 'Stop requested: it ends at its next step' }] });
    assert.match(stopping, /<span data-i18n-skip>Working · Stop requested: it ends at its next step<\/span> · 0s<\/span>/);
    assert.match(liveActivityRowHtml(running, now), /<span class="ui-row-title activity-name">Live turn<\/span>/);
    // Before its first write no door takes it: the row says it is preparing and offers nothing.
    const preparing = liveActivityRowHtml({ activity_id: 'turn', kind: 'presence', phase: 'preparing', cancelable: false }, now);
    assert.match(preparing, /<span data-i18n-skip>Preparing<\/span>/);
    assert.doesNotMatch(preparing, /data-act="task-control"/);
    // Another kind keeps its own label and control, whatever it is called or whichever way its flag reads.
    const direct = liveActivityRowHtml({ activity_id: 'd', kind: 'direct_chat', phase: 'working', cancelable: false }, now, null, false, { name: 'A name',
        because: [{ code: 'held', text: 'Held' }] });
    assert.match(direct, /<span class="ui-row-title activity-name">Direct turn<\/span>/);
    assert.doesNotMatch(direct, /Held/);
    assert.match(direct, /data-act="task-control" data-id="d"/);
});

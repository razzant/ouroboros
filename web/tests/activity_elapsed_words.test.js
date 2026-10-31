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

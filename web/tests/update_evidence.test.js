import assert from 'node:assert/strict';
import test from 'node:test';
import { afterUpdateRows, bindPreflightDescription, preflightDescription, preflightEvidenceRows } from '../modules/update_status.js';
import { updateLetterView } from '../modules/updates.js';

const plan = { base_sha: 'aaaa', target_sha: 'bbbb' };
const letter = { key: plan, text: '## Exact range\n\n- one change', description_current: true, state: 'ready', relation: 'pending' };

test('a new preflight removes the older description and preserves the exact one', () => {
    const status = { available: true, managed: true, check_ok: true, letter };
    const matching = bindPreflightDescription(status, { merge_plan: plan, letter });
    assert.match(updateLetterView(matching).markdown, /Exact range/);
    const moved = bindPreflightDescription(status, { merge_plan: { ...plan, target_sha: 'cccc' }, letter });
    assert.equal(moved.letter, null);
    assert.equal(updateLetterView(moved).markdown, '');
    assert.equal(preflightDescription({ merge_plan: moved.preflight_range, letter }), '');
    // The exact range's text reaches the dialog as Markdown to render, never flattened into the body.
    assert.equal(preflightDescription({ merge_plan: plan, letter }), letter.text);
    assert.equal(preflightDescription({ merge_plan: plan, letter: { ...letter, description_current: false } }), '');
});

test('the preflight pins the prose only: the live request keeps its task followed', () => {
    // Write was pressed, then Apply opened before anything was saved: the plan has no text to show,
    // yet the request's task is still the one whose save and end the panel must hear.
    const request = { task_id: 'd3adb33f', mode: 'write', key: plan, status: 'running', saved: false };
    const writing = { available: true, managed: true, check_ok: true, running_sha: 'aaaa',
        letter: { state: 'none', text: '', checked_key: plan, request } };
    const pinned = bindPreflightDescription(writing, { merge_plan: plan, letter: { ...letter, key: {}, text: '' } });
    assert.equal(pinned.letter.request.task_id, 'd3adb33f');
    assert.equal(updateLetterView(pinned, 'preflighting').note, 'Ouroboros is writing the description (task d3adb33f).');
    // A refresh while the dialog is open binds the fresh status: its request, never the old pin's.
    const saved = { ...writing, letter: { ...letter, request: { ...request, status: 'completed', saved: true } } };
    const refreshed = bindPreflightDescription(saved, { merge_plan: plan, letter: pinned.letter });
    assert.deepEqual(refreshed.letter.request, saved.letter.request);
    assert.equal(updateLetterView(refreshed, 'preflighting').markdown, '', 'the dialog\'s pin still decides the prose');
    // A matching pin carries the live request beside its own text; no request, no letter.
    assert.equal(bindPreflightDescription(saved, { merge_plan: plan, letter }).letter.request.saved, true);
    assert.equal(bindPreflightDescription({}, { merge_plan: { ...plan, target_sha: 'cccc' }, letter }).letter, null);
});

const rowsText = (rows) => rows.map((row) => `${row.label}: ${row.value}`).join('\n');

test('confirmation details name failed, absent and green evidence without converting declarations to proof', () => {
    const receipt = { checks: [{ name: 'full_battery', status: 'failed', source: 'author_declared', detail: 'two FAILs accepted' }],
        review: { observed: 'FAIL', declared: 'AUTHOR ACCEPTED' } };
    const qualification = { ...plan, commits: [{ sha: 'bbbb', subject: '#123', receipt }, { sha: 'missing', subject: '#122' }],
        github: { status: 'observed', runs: [{ sha: 'bbbb', conclusion: 'failure', url: 'https://github.com/run/1' }] } };
    const rows = preflightEvidenceRows({ merge_plan: plan, letter, qualification });
    assert.ok(rows.every((row) => typeof row.label === 'string' && typeof row.value === 'string'));
    const text = rowsText(rows);
    for (const part of ['bbbb: #123', 'failed (author_declared)', 'two FAILs accepted', 'missing: #122\nNo qualification receipt',
        'post-merge: failure', 'Review observed FAIL, author declared AUTHOR ACCEPTED']) {
        assert.ok(text.includes(part), part);
    }
    assert.doesNotMatch(text, /Exact range/, 'the description never repeats inside the evidence');
    qualification.commits = [{ sha: 'bbbb', subject: 'green', receipt: { checks: [{ name: 'full_battery', status: 'passed', source: 'author_declared' }], review: { observed: 'PASS', declared: 'PASS' } } }];
    qualification.github.runs[0].conclusion = 'success';
    const green = rowsText(preflightEvidenceRows({ merge_plan: plan, letter, qualification }));
    assert.match(green, /full_battery passed/);
    assert.doesNotMatch(green, /failed|failure|No qualification receipt|FAIL/);
    assert.doesNotMatch(green, /also covers/);
    qualification.github.runs[0].covered_shas = ['earlier-merge'];
    assert.match(rowsText(preflightEvidenceRows({ merge_plan: plan, letter, qualification })), /This run also covers: earlier-/);
    // Facts of another range are unknown for this one, never borrowed.
    assert.deepEqual(preflightEvidenceRows({ merge_plan: { ...plan, target_sha: 'cccc' }, letter, qualification }),
        [{ label: 'Checks', value: 'Qualification for this range is unknown.' }]);
});

test('each commit keeps its own verdict, record state and request-time relation; a legacy receipt invents neither', () => {
    const at = (sha, review) => ({ sha, subject: sha, receipt: { checks: [], review } });
    const qualification = { ...plan, github: { status: 'observed', runs: [] }, commits: [
        at('pass-different', { observed: 'PASS', state: 'settled', relation: 'changes_after_review', declared: 'PASS' }),
        at('fail-same', { observed: 'FAIL', state: 'settled', relation: 'covers_head', declared: 'PASS' }),
        at('legacy', { observed: 'PASS', declared: 'PASS' }),
        at('future', { observed: 'PASS', state: 'pending', relation: 'covers_head', declared: 'unknown', landing: 'same' }),
    ] };
    const value = Object.fromEntries(preflightEvidenceRows({ merge_plan: plan, letter, qualification })
        .map((row) => [row.label, row.value]));
    assert.ok(value['pass-dif'].endsWith(
        'Review observed PASS, author declared PASS; record settled, subject relation at the merge request changes_after_review.'));
    assert.ok(value['fail-sam'].endsWith(
        'Review observed FAIL, author declared PASS; record settled, subject relation at the merge request covers_head.'));
    assert.ok(value.legacy.endsWith(
        'Review observed PASS, author declared PASS; record state and subject relation at the merge request not recorded.'));
    assert.doesNotMatch(value.legacy, /settled|covers_head/);
    assert.ok(value.future.endsWith('record pending, subject relation at the merge request covers_head; landing same.'));
});

const sides = (rows) => {
    const at = rows.findIndex((row, index) => index > 0 && row.head);
    return [rows.slice(0, at), rows.slice(at)];
};
const values = (rows) => Object.fromEntries(rows.filter((row) => !row.head).map((row) => [row.label, row.value]));

test('after-update rows keep peak RSS, lifetime CPU, interval share and dispatch apart, each in full', () => {
    assert.deepEqual(afterUpdateRows(null), []);
    const source = 'c0ffee'.repeat(6) + 'abcd';
    const after = { source_sha: source, observed_at: '2026-10-21T09:12:05.123456+00:00', scope: 'server_process',
        rss_bytes: 734003200, rss_measure: 'peak', cpu_seconds: 412.5, cpu_percent: 12.3456, cpu_interval_seconds: 600,
        duty_overruns: { reconcile: 2 }, direct_message_first_dispatch: { task_id: 'turn-task', seconds: 4.567 } };
    const [before, rows] = sides(afterUpdateRows({ operation_id: 'op', before: null, after }));
    assert.deepEqual(before, [{ head: 'Before the update' }, { label: 'Sample', value: 'unobserved' }]);
    assert.equal(rows[0].head, 'After the update (recorded sample, server process)');
    assert.deepEqual(rows.slice(1).map((row) => row.label), ['Source', 'Observed', 'RSS peak', 'CPU time, process lifetime',
        'CPU share, sample interval', 'Duty overruns', 'Direct message to first model call']);
    // Sizes and durations read as the other Dashboard tabs write them; the duties in words, each with its count.
    assert.deepEqual(values(rows), {
        Source: source, Observed: rows[2].value, 'RSS peak': '734 MB', 'CPU time, process lifetime': '7 min',
        'CPU share, sample interval': '12.3 % over 10 min', 'Duty overruns': 'reconcile ×2',
        'Direct message to first model call': '4.6 s from acceptance · task turn-task',
    });
    assert.equal(values(afterUpdateRows({ after: { ...after, cpu_seconds: 17690.78, rss_bytes: 2363.23 * 1048576,
        duty_overruns: { 'custody-maintenance': 1, owner_pause_settlement: 1 } } }).slice(3))['CPU time, process lifetime'], '4 h 55 min');
    assert.equal(values(afterUpdateRows({ after: { ...after, duty_overruns: { 'custody-maintenance': 1, owner_pause_settlement: 1 } } })
        .slice(3))['Duty overruns'], 'custody maintenance ×1, owner pause settlement ×1');
    // The local minute to read; the exact recorded stamp is the raw record's alone.
    assert.match(rows[2].value, /UTC[+-]\d/);
    assert.ok(!JSON.stringify(rows).includes(after.observed_at));
    const text = JSON.stringify(rows);
    assert.doesNotMatch(text, /better|worse|regress|improv|healthy|median|average|latency|response time|current/i);
});

test('a zero is a measurement and a missing one stays unobserved; a current RSS drops the peak word', () => {
    const zero = { source_sha: 'a'.repeat(40), observed_at: '', rss_bytes: 0, rss_measure: 'current', cpu_seconds: 0,
        cpu_percent: 0, cpu_interval_seconds: 0, duty_overruns: {}, direct_message_first_dispatch: { task_id: '', seconds: 0 } };
    const missing = { source_sha: '', rss_bytes: null, rss_measure: 'unobserved', cpu_seconds: null, cpu_percent: null,
        cpu_interval_seconds: null, duty_overruns: null, direct_message_first_dispatch: null };
    const [before, after] = sides(afterUpdateRows({ before: zero, after: missing }));
    // A zero is a measurement and says so; a value the sample did not record leaves its row (the raw record keeps the
    // null), never a zero in its place.
    assert.deepEqual(values(before), {
        Source: 'a'.repeat(40), Observed: 'unknown time', RSS: '0 bytes', 'CPU time, process lifetime': '0.0 s',
        'CPU share, sample interval': '0.0 % over 0.0 s', 'Duty overruns': 'none',
        'Direct message to first model call': '0.0 s from acceptance',
    });
    assert.deepEqual(values(after), { Source: 'unknown source', Observed: 'unknown time' });
    assert.doesNotMatch(JSON.stringify(after), /unobserved|\b0\b/);
    // A side with no sample at all still says so once.
    assert.deepEqual(sides(afterUpdateRows({ before: null, after: zero }))[0],
        [{ head: 'Before the update' }, { label: 'Sample', value: 'unobserved' }]);
    // One measured half of the share still says which half is missing.
    const half = sides(afterUpdateRows({ before: { ...missing, cpu_percent: 5 }, after: null }))[0];
    assert.equal(values(half)['CPU share, sample interval'], '5.0 % over unobserved');
});

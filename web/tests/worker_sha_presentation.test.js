import assert from 'node:assert/strict';
import test from 'node:test';
import { summarizeLogEvent, duplicateLogEventKey, categorizeLogEvent } from '../modules/log_events.js';
import { applyPayload, CODE_PREFIX } from '../modules/i18n.js';

const observation = {
    type: 'worker_sha_verify', expected_sha: 'a'.repeat(40), observed_sha: 'b'.repeat(40),
    ok: false, worker_pid: 123,
};

test('Logs renders relation facts without turning inequality into a warning', () => {
    const cases = [
        ['equal', 'ok', 'Worker checkout matches baseline'],
        ['descendant', 'info', 'Worker checkout descends from baseline'],
        ['non_descendant', 'warn', 'Worker checkout is not a descendant of baseline'],
        ['unavailable', 'warn', 'Worker checkout comparison unavailable'],
        ['not_applicable', 'info', 'Worker checkout comparison skipped'],
    ];
    for (const [relation, phase, headline] of cases) {
        const event = { ...observation, relation };
        if (relation === 'equal') { event.observed_sha = event.expected_sha; event.ok = true; }
        if (relation === 'not_applicable') { event.expected_sha = ''; event.ok = null; }
        const view = summarizeLogEvent(event);
        assert.equal(view.phase, phase, relation);
        assert.equal(view.headline, headline, relation);
        assert.equal(categorizeLogEvent(event), 'system');
        assert.ok(view.meta.includes('pid 123'));
        assert.equal(event.ok, relation === 'equal' ? true : relation === 'not_applicable' ? null : false);
    }
});

test('historical rows retain only their recorded equality facts', () => {
    const different = summarizeLogEvent(observation);
    assert.equal(different.phase, 'info');
    assert.equal(different.headline, 'Worker checkout differs from baseline');
    assert.match(different.body, /Ancestry was not recorded/);
    assert.deepEqual(different.meta, ['baseline aaaaaaaa', 'observed bbbbbbbb', 'pid 123']);
    const equal = summarizeLogEvent({ ...observation, observed_sha: observation.expected_sha, ok: true });
    assert.equal(equal.phase, 'ok');
    assert.equal(equal.headline, 'Worker checkout matches baseline');
    const skipped = summarizeLogEvent({ ...observation, expected_sha: '', ok: null });
    assert.equal(skipped.phase, 'info');
    assert.equal(skipped.headline, 'Worker checkout comparison skipped');
    assert.equal(skipped.body, 'No managed baseline recorded.');
    assert.equal(summarizeLogEvent({ ...observation, observed_sha: '' }).phase, 'warn');
});

test('repeat grouping cannot hide a changed or missing ancestry observation', () => {
    const variants = [observation, ...['descendant', 'non_descendant', 'unavailable'].map(
        (relation) => ({ ...observation, relation }),
    )];
    assert.equal(new Set(variants.map(duplicateLogEventKey)).size, variants.length);
    const descendant = { ...observation, relation: 'descendant' };
    assert.equal(duplicateLogEventKey(descendant), duplicateLogEventKey({ ...descendant, worker_pid: 456 }));
    assert.notEqual(duplicateLogEventKey(descendant), duplicateLogEventKey({ ...descendant, observed_sha: 'c'.repeat(40) }));
});

test('new host labels use the translation memory and leave SHA values intact', () => {
    applyPayload({ language: 'ru', english: false, revision: 1, entries: {
        [CODE_PREFIX + 'worker.sha.descendant']: { text: 'Коммит рабочего процесса продолжает базовую историю' },
        'baseline {sha}': { text: 'база {sha}' },
        'observed {sha}': { text: 'наблюдается {sha}' },
    } });
    try {
        const event = { ...observation, relation: 'descendant' };
        const before = duplicateLogEventKey(event);
        const view = summarizeLogEvent(event);
        assert.equal(view.headline, 'Коммит рабочего процесса продолжает базовую историю');
        assert.deepEqual(view.meta, ['база aaaaaaaa', 'наблюдается bbbbbbbb', 'pid 123']);
        assert.equal(duplicateLogEventKey(event), before);
    } finally {
        applyPayload({ language: '', english: true, revision: 0, entries: {} });
    }
});

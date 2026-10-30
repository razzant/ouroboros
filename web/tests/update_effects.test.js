import assert from 'node:assert/strict';
import test from 'node:test';
import { updateEffectsText } from '../modules/update_effects.js';

const preflight = (engine, active = { status: 'known', model: 2, non_model: 3 }) => ({
    merge_plan: { target_sha: 'target' },
    effects: { target_sha: 'target', active_work: active,
        engine: { serving: { version: '4.5.0' }, target: { version: '4.6.2' }, ...engine },
        replacement_engine: { serving: { version: '4.5.0' }, target: { version: '4.6.2' }, action: 'replace' },
        runtime: { installed: { python: '3.10.19', sqlite: '3.50.4' }, bundle: { python: '3.10.22', sqlite: '3.53.1' } },
    },
});

test('Apply shows shared-engine interruption, current work, and the unchanged interpreter', () => {
    const text = updateEffectsText(preflight({ action: 'replace', direction: 'forward' }));
    for (const phrase of ['4.5.0 → 4.6.2', 'end all its active runs', 'including other work',
        '2 model call(s), 3 other job(s)', 'forward-only', 'Python 3.10.19, SQLite 3.50.4',
        'Python 3.10.22, SQLite 3.53.1', 'keeps the interpreter and SQLite']) assert.ok(text.includes(phrase), phrase);
});

test('a backward pin explains its cold-start limit and replacement uses its own pin', () => {
    const data = preflight({ action: 'keep', direction: 'backward', target: { version: '4.4.0' } });
    assert.match(updateEffectsText(data), /keeps the serving engine running/);
    assert.match(updateEffectsText(data), /cold start needs a body pinning 4.5.0 or newer/);
    assert.doesNotMatch(updateEffectsText(data), /end all its active runs/);
    assert.match(updateEffectsText(data, { replace: true }), /end all its active runs/);
});

test('unreadable or differently bound observations never become zero work', () => {
    for (const data of [undefined, {}, preflight({ action: 'unknown' }, { status: 'unknown', total: null }),
        { ...preflight({ action: 'replace' }), merge_plan: { target_sha: 'moved' } }]) {
        const text = updateEffectsText(data);
        assert.match(text, /Currently active engine work: unknown/);
        assert.doesNotMatch(text, /0 model call|0 other job/);
    }
    assert.match(updateEffectsText(preflight({ action: 'keep' }, { status: 'known', model: 0, non_model: 0 })), /0 model call/);
});

// Restart to update: the effects read for one prepared candidate (update_impact.candidate_effects).
const prepared = (runtime, fields = {}) => ({
    merge_plan: { target_sha: 'target' },
    effects: { candidate_id: 'update_a', target_sha: 'target', provisional: false,
        active_work: { status: 'known', model: 1, non_model: 0 },
        engine: { serving: { version: '4.6.0' }, target: { version: '4.6.10' }, action: 'replace', direction: 'forward' },
        replacement_engine: { serving: { version: '4.6.0' }, target: { version: '4.6.0' }, action: 'keep', direction: 'unchanged' },
        runtime: { installed: { python: '3.10.19', sqlite: '3.50.4' }, bundle: { python: '3.10.22', sqlite: '3.53.1' }, ...runtime },
        ...fields },
});

test('Restart to update words its candidate: the composition pin, the work and the runtime its preparation selected', () => {
    const changing = updateEffectsText(prepared({ selected: { python: '3.10.22', sqlite: '3.53.1' }, changed_by_apply: true }),
        { candidate: 'update_a' });
    for (const phrase of ['4.6.0 → 4.6.10', 'end all its active runs', '1 model call(s), 0 other job(s)', 'forward-only',
        'This install: Python 3.10.19, SQLite 3.50.4', 'the new start runs Python 3.10.22, SQLite 3.53.1']) {
        assert.ok(changing.includes(phrase), phrase);
    }
    assert.doesNotMatch(changing, /keeps the interpreter|final pin may change/);
    const keeping = prepared({ selected: { python: '3.10.19', sqlite: '3.50.4' }, changed_by_apply: false });
    assert.match(updateEffectsText(keeping, { candidate: 'update_a' }), /Restart to update keeps the interpreter and SQLite/);
});

test("another candidate's effects, a target-bound read and an unrecorded runtime stay unknown for Restart to update", () => {
    const unrecorded = prepared({ selected: { python: null, sqlite: null }, changed_by_apply: null });
    assert.match(updateEffectsText(unrecorded, { candidate: 'update_a' }), /Whether Restart to update changes the interpreter/);
    const known = prepared({ changed_by_apply: false });
    const legacy = { ...known, effects: { ...known.effects, candidate_id: undefined } };
    for (const [data, id] of [[known, 'update_b'], [known, ''], [legacy, 'update_a'], [undefined, 'update_a']]) {
        const text = updateEffectsText(data, { candidate: id });
        assert.match(text, /Currently active engine work: unknown/);
        assert.match(text, /engine transition could not be confirmed/);
        assert.match(text, /Whether Restart to update changes the interpreter and SQLite is unknown/);
        assert.doesNotMatch(text, /model call|keeps the interpreter|4\.6\.10/);
    }
});

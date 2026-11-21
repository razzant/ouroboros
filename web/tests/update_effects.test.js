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

test('every Update and Replace leads with what the restart does to tasks, whatever the engine facts are', () => {
    const facts = [/^Running tasks stop and continue by themselves after the restart, from their last saved step/,
        /waiting and sleeping tasks and the runnable queue return too, and schedules are kept/,
        /no need to pause anything first\.\n\nTasks already paused or held stay so until they are resumed\./,
        /Not continued: a place conversation cut mid-turn, a task with nothing saved to continue from, and services/,
        /A task the restart holds instead of continuing waits for Resume\.\n\nClaudexor: /];
    // A fixed effect, not an observation: an unreadable or re-bound preflight says it too.
    for (const data of [undefined, {}, preflight({ action: 'keep' }), preflight({ action: 'replace' }),
        preflight({ action: 'unknown' }, { status: 'unknown', total: null })]) {
        for (const options of [undefined, { replace: true }]) {
            const text = updateEffectsText(data, options);
            for (const fact of facts) assert.match(text, fact);
        }
    }
    // The engine's admission count is not this installation's task count, and says so beside that paragraph.
    const known = updateEffectsText(preflight({ action: 'keep' }));
    assert.match(known, /Currently active engine work: 2 model call\(s\), 3 other job\(s\)\./);
    assert.doesNotMatch(known, /Currently active: /);
});

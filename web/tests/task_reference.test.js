// The one door to a Main task's card (docs/DESIGN.md "References and actions"): its markup, the one
// navigation it raises, and that no other module raises it.

import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import test from 'node:test';

import { openTaskReference, taskReferenceHtml } from '../modules/task_reference.js';

test('the door is the sentence-link words for one task, and nothing without one', () => {
    assert.equal(taskReferenceHtml('d3adb33f'),
        '<button type="button" class="consciousness-inline-link ui-hit-area" data-open-task="d3adb33f">Open in Main ↗</button>');
    assert.equal(taskReferenceHtml(''), '');
    assert.match(taskReferenceHtml('a"b<c'), /data-open-task="a&quot;b&lt;c"/);
});

test('a press on a door opens its task in Main; any other press is not the door\'s', () => {
    const raised = [];
    globalThis.window = new EventTarget();
    window.addEventListener('ouro:open-task', (event) => raised.push(event.detail));
    try {
        const door = { dataset: { openTask: 'c0ffee42' } };
        assert.equal(openTaskReference({ target: { closest: (selector) => (selector === '[data-open-task]' ? door : null) } }), true);
        assert.equal(openTaskReference({ target: { closest: () => null } }), false);
        assert.equal(openTaskReference({ target: {} }), false);
        assert.deepEqual(raised, [{ task_id: 'c0ffee42', chat_id: 1 }]);
    } finally {
        delete globalThis.window;
    }
});

test('only the door module raises ouro:open-task; Consciousness and Updates draw it from there', () => {
    const modules = new URL('../modules/', import.meta.url);
    const raisers = readdirSync(modules).filter((name) => name.endsWith('.js')
        && readFileSync(new URL(name, modules), 'utf8').includes("new CustomEvent('ouro:open-task'"));
    assert.deepEqual(raisers, ['task_reference.js']);
    assert.doesNotMatch(readFileSync(new URL('../app.js', import.meta.url), 'utf8'), /new CustomEvent\('ouro:open-task'/);
    for (const name of ['consciousness_dashboard.js', 'updates.js']) {
        const source = readFileSync(new URL(name, modules), 'utf8');
        assert.match(source, /import \{ openTaskReference, taskReferenceHtml \} from '\.\/task_reference\.js';/, name);
        assert.doesNotMatch(source, /data-cx-open-task|Open in Main/, name);
    }
});

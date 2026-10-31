// The Costs budget save is feedback about an action the owner just took (DESIGN §4 "Status and chips"): a plain
// "saved" ends by itself; a restart still required, a warning and an error stay until the next save replaces them,
// and one timer owner means an older acknowledgement never erases a newer save's words or its tone.
import assert from 'node:assert/strict';
import test from 'node:test';
import { budgetSavePresentation, initCosts } from '../modules/costs.js';
import { setup } from './dashboard_dom_fixture.js';

const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const settle = () => new Promise((resolve) => setImmediate(resolve));
const ACK = 4000;

test('a save says what happened and only a plain acknowledgement may end by itself', () => {
    const cases = [
        [{ no_changes: true }, 'No changes.', 'ok', true],
        [{ immediate_changed: true }, 'Saved. Took effect immediately.', 'ok', true],
        [{ next_task_changed: true }, 'Saved. Applies on the next task.', 'ok', true],
        [{ immediate_changed: true, next_task_changed: true },
            'Saved. Some changes took effect immediately; others apply on the next task.', 'ok', true],
        [{ restart_required: true }, 'Saved. Restart required.', 'warn', false],
        // Nothing changed, but a setting saved earlier still waits for a restart (gateway/settings.py restart_state).
        [{ no_changes: true, restart_required: true }, 'No changes. Restart still required.', 'warn', false],
        [{ immediate_changed: true, warnings: ['Limit is below known spend.', 'Cap lowered.'] },
            'Saved. Took effect immediately. Warnings: Limit is below known spend. | Cap lowered.', 'warn', false],
    ];
    for (const [data, text, tone, transient] of cases) {
        assert.deepEqual(budgetSavePresentation(data), { text, tone, transient }, JSON.stringify(data));
    }
});

// The real panel, its real click handler and the shared HTTP client on the dashboard DOM fixture, with mocked time.
function costsPanel(t, answers) {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const env = setup(t);
    const posts = [];
    env.routes.set('/api/settings', async (_url, init) => {
        posts.push(JSON.parse(init.body));
        const next = answers.shift();
        if (next instanceof Error) throw next;
        return next;
    });
    initCosts({ mount: env.mount, state: {} });
    const status = env.mount.querySelector('#budget-save-status');
    const save = env.mount.querySelector('#btn-save-budget');
    env.mount.querySelector('#s-budget').value = '200';
    env.mount.querySelector('#s-per-task-cost').value = '50';
    const press = async () => { await save.fire('click', { currentTarget: save }); await settle(); };
    const words = () => [status.textContent, status.dataset.tone ?? ''];
    return { ...env, status, save, posts, press, words, tick: (ms) => t.mock.timers.tick(ms) };
}

test('a plain save acknowledges, then clears its words and its tone after the acknowledgement time', async (t) => {
    const panel = costsPanel(t, [reply({ immediate_changed: true })]);
    await panel.press();
    assert.deepEqual(panel.words(), ['Saved. Took effect immediately.', 'ok']);
    panel.tick(ACK - 1);
    assert.deepEqual(panel.words(), ['Saved. Took effect immediately.', 'ok'], 'still there just before it ends');
    panel.tick(1);
    assert.deepEqual(panel.words(), ['', ''], 'the acknowledgement ended by itself, its tone with it');
    assert.equal(panel.save.disabled, false);
});

test('a restart the save requires, a warning and an error are never erased by time', async (t) => {
    const panel = costsPanel(t, [reply({ restart_required: true }), reply({ no_changes: true, restart_required: true }),
        reply({ immediate_changed: true, warnings: ['Limit is below known spend.'] }),
        reply({ error: 'Settings file is read-only.' }, 500)]);
    await panel.press();
    assert.deepEqual(panel.words(), ['Saved. Restart required.', 'warn']);
    panel.tick(10 * ACK);
    assert.deepEqual(panel.words(), ['Saved. Restart required.', 'warn'], 'the restart stays required, so its words stay');
    await panel.press();  // the same values again: nothing changes, and the restart is still owed
    panel.tick(10 * ACK);
    assert.deepEqual(panel.words(), ['No changes. Restart still required.', 'warn'], 'it says why it stays');
    await panel.press();
    panel.tick(10 * ACK);
    assert.deepEqual(panel.words(), ['Saved. Took effect immediately. Warnings: Limit is below known spend.', 'warn']);
    await panel.press();
    panel.tick(10 * ACK);
    assert.deepEqual(panel.words(), ['Error: Settings file is read-only.', 'error'], 'a failed save is not an acknowledgement');
});

test('one timer owner: an older acknowledgement never erases what a newer save says', async (t) => {
    const panel = costsPanel(t, [reply({ immediate_changed: true }), reply({ error: 'Offline.' }, 503),
        reply({ immediate_changed: true }), reply({ restart_required: true })]);
    await panel.press();
    panel.tick(ACK - 1000);
    await panel.press();  // a failing save inside the window: the first timer must not take it away
    panel.tick(ACK);
    assert.deepEqual(panel.words(), ['Error: Offline.', 'error']);
    // Two plain saves: the second one's own full time counts, and the first one's timer is gone.
    await panel.press();
    panel.tick(ACK - 1000);
    await panel.press();  // answered by the restart-required reply
    panel.tick(ACK);
    assert.deepEqual(panel.words(), ['Saved. Restart required.', 'warn']);
});

test('a validation message after an acknowledgement is not erased by the acknowledgement\'s timer either', async (t) => {
    const panel = costsPanel(t, [reply({ immediate_changed: true })]);
    await panel.press();
    panel.mount.querySelector('#s-budget').value = '0';
    panel.tick(ACK - 1);
    await panel.press();
    panel.tick(ACK);
    assert.deepEqual(panel.words(), ['Budget values must be at least 0.01.', 'error'], 'never in the acknowledgement\'s tone');
    assert.equal(panel.posts.length, 1, 'an invalid value is never sent');
});

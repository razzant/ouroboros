import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizeConsciousness } from '../modules/consciousness_api.js';
import { deadlineWords, splitWakes } from '../modules/consciousness_words.js';
import { applyPayload, setMissTransport } from '../modules/i18n.js';
import { initActivity } from '../modules/activity.js';
import { setup } from './dashboard_dom_fixture.js';

const now = Date.parse('2027-01-15T12:00:00Z');
const facts = { enabled: true, status: 'due', next_wake_at: '2027-01-15T11:50:00Z', appointment_at: '2027-01-15T11:30:00Z' };

for (const [zone, past, future] of [['UTC', '11:30', '13:00'], ['Europe/Moscow', '14:30', '16:00']]) {
    test(`deadline words use the appointment and the viewer's clock in ${zone}`, (t) => {
        const prior = process.env.TZ;
        process.env.TZ = zone;
        t.after(() => { if (prior === undefined) delete process.env.TZ; else process.env.TZ = prior; });
        const snapshot = normalizeConsciousness(facts);
        assert.equal(snapshot.appointmentAt, facts.appointment_at);
        assert.equal(snapshot.liveWakeHeld, false);
        assert.equal(deadlineWords(snapshot, now), `since ${past}`);
        assert.equal(deadlineWords({ ...snapshot, nextWakeAt: new Date(now).toISOString() }, now), `since ${past}`);
        assert.equal(deadlineWords({ ...snapshot, nextWakeAt: '2027-01-15T13:00:00Z' }, now),
            future, 'a future retry is never since its old appointment');
        assert.equal(deadlineWords({ ...snapshot, appointmentAt: '' }, now), 'due');
        for (const override of [{ liveWakeTaskId: 'own' }, { liveWakeHeld: true }, { enabled: false }, { enabled: null }, { nextWakeAt: '' }]) {
            assert.equal(deadlineWords({ ...snapshot, ...override }, now), '');
        }
        for (const status of ['wake_failed', 'wake_rejected', 'wake_outcome_unknown', 'allowance_unknown', 'wake_paused', 'sleeping']) {
            assert.equal(deadlineWords({ ...snapshot, status }, now), '', 'a passed explained outcome adds no time');
            assert.equal(deadlineWords({ ...snapshot, status, nextWakeAt: '2027-01-15T13:00:00Z' }, now), future);
        }
        for (const allowanceResetsAt of ['2027-01-15T11:00:00Z', '2027-01-15T12:00:00Z', '2027-01-15T13:00:00Z']) {
            assert.equal(deadlineWords({ ...snapshot, status: 'allowance_exhausted', allowanceResetsAt }, now),
                allowanceResetsAt.includes('13:00') ? future : '', 'an expired reset is not a coming check');
        }
        const localMidnight = Date.parse(zone === 'UTC' ? '2027-01-15T00:01:00Z' : '2027-01-14T21:01:00Z');
        const appointmentAt = new Date(localMidnight - 120000).toISOString();
        const previousDay = new Date(appointmentAt).toLocaleDateString([], { month: 'short', day: 'numeric' });
        assert.equal(deadlineWords({ ...snapshot, appointmentAt, nextWakeAt: appointmentAt }, localMidnight),
            `since ${previousDay} 23:59`);
        const tomorrow = new Date(localMidnight + 86400000).toISOString();
        const nextDay = new Date(tomorrow).toLocaleDateString([], { month: 'short', day: 'numeric' });
        assert.equal(deadlineWords({ ...snapshot, nextWakeAt: tomorrow }, localMidnight), `${nextDay} 00:01`);
        assert.equal(deadlineWords({ ...snapshot, status: 'allowance_exhausted', allowanceResetsAt: tomorrow }, localMidnight), `${nextDay} 00:01`);
    });
}

test('split wakes preserves each source order, with absent legacy cause treated as own', () => {
    const wakes = [{ taskId: 'place1', cause: 'focus' }, { taskId: 'own1', cause: 'own' },
        { taskId: 'legacy' }, { taskId: 'place2', cause: 'focus' }];
    assert.deepEqual(splitWakes(wakes), { own: [wakes[1], wakes[2]], places: [wakes[0], wakes[3]] });
    assert.deepEqual(splitWakes(null), { own: [], places: [] });
    assert.deepEqual(splitWakes([wakes[1]]), { own: [wakes[1]], places: [] });
    assert.deepEqual(splitWakes([wakes[0]]), { own: [], places: [wakes[0]] });
});

test('chip deadline words are registered with i18n', (t) => {
    setMissTransport(() => Promise.resolve(null));
    applyPayload({ language: 'ru', english: false, entries: {
        'since {time}': { text: 'с {time}' }, 'code:consciousness.due': { text: 'пора' },
    } });
    t.after(() => applyPayload({ language: '', english: true, entries: {} }));
    assert.match(deadlineWords(normalizeConsciousness(facts), now), /^с /);
    assert.equal(deadlineWords(normalizeConsciousness({ ...facts, appointment_at: '' }), now), 'пора');
});

test('Activity refresh keeps the host sentence as its only clock and names own tasks', async (t) => {
    const prior = process.env.TZ;
    process.env.TZ = 'Europe/Moscow';
    t.after(() => { if (prior === undefined) delete process.env.TZ; else process.env.TZ = prior; });
    const cleanups = [];
    const { mount, routes, ws } = setup({ after: (cleanup) => cleanups.push(cleanup) });
    t.mock.method(Date, 'now', () => now);
    const response = (data) => ({ ok: true, status: 200, json: async () => data });
    routes.set('/api/tasks?queue_only=1', response({ queue: { running: [], pending: [] } }));
    routes.set('/api/schedules', response({ tasks: [] }));
    const activity = initActivity({ mount, ws });
    t.after(() => { activity.destroy(); cleanups.forEach((cleanup) => cleanup()); });
    for (const override of [
        { detail: 'Due since 11:30; the wake-up has not started yet.' },
        { live_wake_task_id: 'own', detail: 'Wake-up own is running.' },
        { live_wake_held: true, detail: 'Wake-up own is paused or waiting in the queue; its card shows why. The next wake-up follows it.' },
        { appointment_at: '', detail: 'Due; the wake-up has not started yet.' },
        { next_wake_at: '2027-01-15T13:00:00Z', detail: 'Sleeping until 13:00.' },
    ]) {
        const bg = { ...facts, tasks_running: 1, max_tasks: 2, ...override };
        routes.set('/api/state', response({ bg_consciousness_enabled: true, bg_consciousness_state: bg }));
        await activity.refresh();
        const text = mount.querySelector('[data-activity-section="background"]').textContent;
        assert.equal(text.split(override.detail).length - 1, 1);
        assert.doesNotMatch(text.replace(override.detail, ''), /\d\d:\d\d|[Dd]ue|since|[Nn]ext wake/);
        assert.match(text, /own tasks 1\/2/);
    }
});

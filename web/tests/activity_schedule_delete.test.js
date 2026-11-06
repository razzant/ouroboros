import assert from 'node:assert/strict';
import test from 'node:test';

// The DOM-free Activity helpers the rendered list and the lifecycle toast use.
globalThis.document = globalThis.document || { createElement: () => ({}) };
const { isRetainedSchedule, scheduleOutcomeToast, scheduleRowHtml, scheduleStatus } = await import('../modules/activity.js');

const consumedPending = {
    id: 'fu-1', name: 'Check the results', status: 'delete_pending', delete_pending: true,
    relation: 'related', enabled: false, trigger: { type: 'once', run_at: '2027-01-15T09:00:00Z' },
    completed_at: '2027-01-15T09:00:05Z', delete_requested_at: '2027-01-15T10:00:00Z',
    withdrawal: { work_in_flight: true, restorable: true },
};

test('an accepted deferred delete reads as pending, never as a refusal', () => {
    const outcome = { ok: true, changed: true, status: 'delete_deferred', audit: 'recorded',
        detail: 'an accepted run is still owed; the row will be removed once that run starts' };
    assert.deepEqual(scheduleOutcomeToast('delete', outcome),
        [`Schedule deletion is pending: ${outcome.detail}`, 'info']);
    // A consumed one-shot still refuses rearming, even when deletion is pending.
    const restore = { ok: false, changed: false, status: 'consumed_not_rearmed', audit: 'recorded',
        detail: 'already fired' };
    assert.deepEqual(scheduleOutcomeToast('restore', restore),
        ['Schedule restore did not change anything: already fired', 'error']);
    // Bookkeeping that changed while refusing a release still reads as refused.
    assert.equal(scheduleOutcomeToast('restore', { changed: true, status: 'stale_hold', audit: 'recorded' })[1], 'error');
    assert.equal(scheduleOutcomeToast('delete', { ok: true, changed: true, status: 'deleted', audit: 'recorded',
        running_or_queued: false }), null);
    const withdrawn = { ok: true, changed: true, status: 'withdrawn', audit: 'recorded', running_or_queued: false };
    assert.deepEqual(scheduleOutcomeToast('delete', withdrawn), ['Schedule withdrawn.', 'info']);
    assert.match(scheduleOutcomeToast('delete', { ...withdrawn, running_or_queued: true })[0], /was not cancelled/);
    assert.match(scheduleOutcomeToast('delete', { ...withdrawn, running_or_queued: null })[0], /is unknown/);
    assert.match(scheduleOutcomeToast('delete', { ...withdrawn, audit: 'incomplete' })[0], /audit record is incomplete/);
    assert.match(scheduleOutcomeToast('delete', { ...withdrawn, changed: false })[0], /did not change anything/);
});

test('a deleted row still owing work stays in the standing list after reload', () => {
    assert.equal(scheduleStatus(consumedPending), 'delete_pending');
    // Older payload without the server word: the durable request still wins over "consumed".
    const legacy = { ...consumedPending };
    delete legacy.status;
    delete legacy.withdrawal;
    assert.equal(scheduleStatus(legacy), 'delete_pending');
    assert.equal(isRetainedSchedule(legacy), false);
    assert.match(scheduleRowHtml(legacy), / · deletion pending<\/span>/);
    assert.match(scheduleRowHtml(legacy), /<p class="activity-facts">relation related/, 'the relation is a fact of its Details');
    assert.doesNotMatch(scheduleRowHtml(legacy), /work in flight|data-act="schedule-toggle"/);
    assert.equal(isRetainedSchedule(consumedPending), false, 'not collapsed into History');
    assert.equal(isRetainedSchedule({ ...consumedPending, status: 'consumed', delete_requested_at: '' }), true);
    const html = scheduleRowHtml(consumedPending);
    assert.match(html, / · deletion pending · work in flight<\/span>/);
    assert.match(html, /<p class="activity-facts">relation related · already fired<\/p>/);
    assert.doesNotMatch(html.split('activity-row-actions')[1], /work in flight/, 'wrapping detail stays with the description');
    assert.match(html, /data-action="restore">Restore</, 'restore cancels withdrawal without rearming a consumed one-shot');
    assert.match(html, /data-act="schedule-delete" data-id="fu-1"/);
    const independent = scheduleRowHtml({ ...consumedPending, relation: 'independent', trigger: { type: 'cron', expr: '0 9 * * *' },
        completed_at: '', next_run_at: '2027-01-16T09:00:00Z' });
    assert.match(independent, /deletion pending · work in flight/);
    assert.match(independent, /data-action="restore">Restore</);
    assert.doesNotMatch(independent, />Enable</, 'cancelling Delete must be explicit');
    assert.doesNotMatch(independent, / · next /, 'a deleted row announces no next run');
});

test('a held withdrawal offers only the exact hold release until the hold is gone', () => {
    const held = { ...consumedPending, followup_hold: { hold_id: 'h-1', reason: 'origin_stopped' }, hold_persisted: true };
    const html = scheduleRowHtml(held);
    assert.match(html, /data-action="restore" data-hold-id="h-1">Restore hold</);
    assert.doesNotMatch(html, /data-action="restore">Restore</, 'a generic Restore over a hold could only be refused');
    assert.doesNotMatch(html, />Enable</);
    assert.match(scheduleRowHtml(consumedPending), /data-action="restore">Restore</, 'released, the withdrawal is restorable');
    const unknown = scheduleRowHtml({ ...held, relation: 'unknown' });
    assert.match(unknown, /resolve relationship in conversation/);
    assert.doesNotMatch(unknown, /data-hold-id=/);
    assert.doesNotMatch(unknown, /data-action="restore">Restore</);
});

test('withdrawal history and Restore follow explicit facts, including unknown work in flight', () => {
    for (const flight of [false, null]) {
        for (const restorable of [true, false]) {
            const row = { ...consumedPending, withdrawal: { work_in_flight: flight, restorable,
                by: 'owner', reason: 'No longer <needed>', at: '2027-01-15T10:00:00Z' } };
            const html = scheduleRowHtml(row);
            assert.equal(isRetainedSchedule(row), true);
            assert.match(html, / · withdrawn/);
            assert.equal(html.includes('withdrawn · restorable'), restorable);
            assert.equal(html.includes('data-action="restore">Restore<'), restorable);
            assert.match(html, /by owner · No longer &lt;needed&gt;/);
            assert.doesNotMatch(html, /work in flight|settled|still running| · next /);
        }
    }
});

test('a legacy note keeps its name, shows its words and offers the same lifecycle', () => {
    const note = {
        id: 'followup-t1-abc123', name: 'Reminder of task t1', kind: 'notify', status: 'active', enabled: true,
        relation: 'independent', trigger: { type: 'once', run_at: '2027-01-15T09:00:00Z' },
        notification: { text: 'Call <mother>', set_at: '2027-01-15T08:00:00Z' },
    };
    const html = scheduleRowHtml(note);
    assert.match(html, /<span class="ui-row-title activity-name" data-i18n-authored>Reminder of task t1<\/span>/);
    // Its words are behind the row's one Details, never sliced into a second title.
    assert.match(html, /<summary data-act="schedule-text" data-id="followup-t1-abc123">Details<\/summary>\s*<p class="activity-order" data-i18n-authored>Call &lt;mother&gt;<\/p>/);
    assert.doesNotMatch(html, /title="Call/);
    assert.match(html, /<span class="activity-sub">Reminder · Once on <time /);
    // The server's display name is the note's own first line: then the title says it and Details holds no copy.
    const named = scheduleRowHtml({ ...note, display_name: 'Call <mother>' });
    assert.match(named, /activity-name" data-i18n-authored>Call &lt;mother&gt;<\/span>/);
    assert.doesNotMatch(named, /activity-order/);
    assert.match(html, /data-act="schedule-toggle" data-id="followup-t1-abc123" data-action="disable">Disable</);
    assert.match(html, /data-act="schedule-delete" data-id="followup-t1-abc123" data-note="1">Delete</);
    const task = scheduleRowHtml({ ...note, kind: undefined });
    assert.match(task, /<span class="ui-row-title activity-name" data-i18n-authored>Reminder of task t1<\/span>/, 'a task row keeps its schedule name');
    assert.match(task, /data-note="">Delete</);
});

test('a row shows its stored error: a consumed note whose chat write was not confirmed says so', () => {
    const note = {
        id: 'followup-t1-abc123', name: 'Reminder of task t1', kind: 'notify', status: 'consumed', enabled: false,
        relation: 'independent', trigger: { type: 'once', run_at: '2027-01-15T09:00:00Z' },
        completed_at: '2027-01-15T09:00:05Z', notification: { text: 'Call mother' },
    };
    const failed = scheduleRowHtml({ ...note, last_error: 'delivery <not> confirmed' });
    assert.match(failed, /<span class="activity-sub">last error: delivery &lt;not&gt; confirmed<\/span>/);
    assert.equal(failed.split('delivery &lt;not&gt; confirmed').length, 2, 'said once');
    assert.doesNotMatch(failed, /activity-tag">last error/, 'never a second, italic copy in the meta line');
    // The run's own line already says it: the stored error is not printed again.
    const said = scheduleRowHtml({ ...note, last_error: 'Not queued', outcome: { state: 'failed', line: 'Not queued' } });
    assert.equal(said.split('Not queued').length, 2);
    assert.match(said, /last run: failed · <span data-i18n-authored>Not queued<\/span>/);
    assert.match(failed, /consumed once · history/, 'still history: no retry control appears');
    assert.doesNotMatch(failed, /data-act="schedule-toggle"/);
    for (const quiet of [note, { ...note, last_error: '' }]) {
        assert.doesNotMatch(scheduleRowHtml(quiet), /last error/, 'a row with no stored error says none');
    }
});

test('a switched-off one-shot whose moment passed is finished; one still ahead stays standing', () => {
    const once = { id: 'o', status: 'disabled', enabled: false, trigger: { type: 'once', run_at: '2026-09-25T12:00:00Z' } };
    assert.equal(isRetainedSchedule({ ...once, time_passed_on: '2026-09-25T12:00:00+00:00' }), true);
    assert.equal(isRetainedSchedule(once), false, 'the server, never the browser clock, says the moment passed');
    assert.equal(isRetainedSchedule({ ...once, status: 'held', time_passed_on: '2026-09-25T12:00:00+00:00' }), false,
        'a held row keeps its control in view');
    assert.equal(isRetainedSchedule({ ...once, status: 'active', enabled: true }), false);
});

test('the outcome speaks the word of the pressed button', () => {
    const ok = { ok: true, changed: true, status: 'updated', audit: 'recorded', running_or_queued: true };
    assert.match(scheduleOutcomeToast('restore', ok, { word: 'enable', done: 'enabled' })[0], /^Schedule enabled\. A task/);
    assert.match(scheduleOutcomeToast('restore', ok, { word: 'restore', done: 'restored' })[0], /^Schedule restored\. /);
    assert.equal(scheduleOutcomeToast('restore', { changed: true, status: 'stale_hold', audit: 'recorded' }, { word: 'enable' })[0],
        'Schedule enable refused: stale_hold');
    assert.equal(scheduleOutcomeToast('restore', { changed: true }, { word: 'enable' })[0],
        'Schedule enable refused: no status reported');
    assert.equal(scheduleOutcomeToast('restore', { ...ok, running_or_queued: false }, { word: 'enable', done: 'enabled' }), null,
        'a clean change says nothing: the row itself shows it');
});

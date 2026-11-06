import assert from 'node:assert/strict';
import test from 'node:test';

// An answered root under the owner's Pause is a census-only Activity row whose census phase only guesses at a park
// (#2669). The confirmed work whose live episode it is says what is true (the server's work projection, shaped as
// `gateway/work_projection.works_snapshot` sends it): its word and line, the row's own outcome still leading, and only
// the controls that work lists. The browser suite tests/test_owner_pause_ended_members_browser.py renders the same rule
// over a backend-generated projection.
globalThis.document = globalThis.document || { createElement: () => ({}) };
const { liveActivityRowHtml } = await import('../modules/activity.js');

const census = (phase, extra = {}) => ({ activity_id: 'root-1', kind: 'managed_task', phase, pause_cause: 'owner',
    status: 'completed', cancelable: true, can_resume: false, started_at: 0, ...extra });
const paused = { code: 'paused_by_owner', text: 'You paused it', since: '2026-11-04T01:00:00+00:00',
    by: { actor: 'owner', text: '', at: '2026-11-04T01:00:00+00:00' } };
const work = (activity, transition, tail, controls) => ({ work_id: 'root-1', room: 1,
    episode: { task_id: 'root-1', live: true }, activity, transition, live: activity === 'working', confirmed: true,
    because: [paused], tails: [tail], controls, release: activity === 'unknown' ? 'none' : 'mind' });
const unconfirmed = (id) => ({ code: 'external_runs_unconfirmed', text: '1 external run may still be running', count: 1,
    ids: [id], kind: 'run', state: 'unknown' });
const running = { code: 'delegated_running', text: '1 external run still working', count: 1, ids: ['run-leaf'],
    kind: 'run', state: 'live' };
const STOP = [{ action: 'stop', target: 'root-1', scope: 'cascade' }];
// The caller's outcome slot, made legible: the chip's word and its quieter second fact.
const slot = (view) => `<chip>${view.text}|${view.secondary || ''}</chip>`;

test('a parked census row says its confirmed work, the outcome leading, never the census guess', () => {
    const row = liveActivityRowHtml(census('budget_paused'), 0, slot, false,
        work('unknown', 'none', unconfirmed('run-leaf'), STOP));
    assert.match(row, /<chip>Done\|Activity unconfirmed<\/chip> · <span data-i18n-skip>You paused it · [^·<]+ · 1 external run may still be running<\/span>/);
    assert.doesNotMatch(row, /Paused|Pausing/);
    assert.match(row, /data-act="task-control" data-id="root-1"/, 'its work lists Stop: the row offers it');
    // The run seen running: the work says the Pause is still settling.
    const live = liveActivityRowHtml(census('budget_pausing', { live: true }), 0, slot, false,
        work('working', 'pausing', running, [{ action: 'pause', target: 'root-1' }, ...STOP]));
    assert.match(live, /<chip>Done\|Pausing…<\/chip> · <span data-i18n-skip>You paused it · [^·<]+ · 1 external run still working<\/span>/);
});

test('a work that lists no control offers none: no trigger, so no generic Stop or Resume', () => {
    const row = liveActivityRowHtml(census('budget_paused'), 0, slot, false,
        work('unknown', 'none', unconfirmed('inv-1'), []));
    assert.match(row, /Activity unconfirmed/);
    assert.doesNotMatch(row, /data-act="task-control"/);
    // Positive control: without a confirmed work the row keeps its census word and the shared control.
    const unread = liveActivityRowHtml(census('budget_paused', { status: undefined }), 0, null);
    assert.match(unread, /<span data-i18n-skip>Paused<\/span>/);
    assert.match(unread, /data-act="task-control" data-id="root-1" data-root="1" data-budget-paused="1"/);
});

test('a late row takes its state and line from the work; rows that are not parked keep their census words', () => {
    const late = { title: 'Audit the release', answered_at: '' };
    const unknown = liveActivityRowHtml(census('budget_paused', { late_work: late }), 0, slot, false,
        work('unknown', 'none', unconfirmed('run-leaf'), STOP));
    assert.match(unknown, /Remaining work status unknown · answer delivered · You paused it · [^·<]+ · 1 external run may still be running/);
    assert.doesNotMatch(unknown, /Remaining work paused/);
    const live = liveActivityRowHtml(census('budget_pausing', { late_work: late }), 0, slot, false,
        work('working', 'pausing', running, STOP));
    assert.match(live, /Remaining work pausing · answer delivered · You paused it · [^·<]+ · 1 external run still working/);
    // A working direct turn is no parked guess: its census words stand, its work beside it unchanged.
    const turn = liveActivityRowHtml({ activity_id: 'turn', kind: 'direct_chat', phase: 'working', started_at: 0 }, 0,
        null, false, { ...work('working', 'none', running, STOP), episode: { task_id: 'turn', live: true } });
    assert.match(turn, /<span class="ui-row-meta activity-sub"><span class="status-mark activity-mark"[^>]*>(?:<i><\/i>){3}<\/span><span data-i18n-skip>Working<\/span><\/span>/);
});

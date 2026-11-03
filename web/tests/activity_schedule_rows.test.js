import assert from 'node:assert/strict';
import test from 'node:test';

globalThis.document = globalThis.document || { createElement: () => ({}) };
const { scheduleInstantHtml, scheduleRowHtml } = await import('../modules/activity.js');

const standing = { id: 'daily', name: 'Daily check', enabled: true,
    trigger: { type: 'cron', expr: '0 9 * * *' }, relation: 'independent' };

test('schedule headings prefer the projected name and retain legacy fallbacks', () => {
    for (const [row, name] of [
        [{ ...standing, display_name: 'Check <results>' }, 'Check &lt;results&gt;'],
        [standing, 'Daily check'], [{ id: 'bare' }, 'bare'],
    ]) {
        assert.ok(scheduleRowHtml(row).includes(`<span class="activity-name" data-i18n-authored>${name}</span>`));
    }
});

test('task and note text is readable with a native disclosure for the complete instruction', () => {
    const text = 'Check <results>\nKeep the second line & its exact words.';
    for (const fields of [{ kind: 'task', task: { text } }, { kind: 'note', notification: { text } }]) {
        const html = scheduleRowHtml({ ...standing, ...fields });
        assert.match(html, /<details[^>]+data-schedule-text="daily">/);
        assert.match(html, /<summary[^>]*>Check &lt;results&gt;…<\/summary>/);
        assert.match(html, /Check &lt;results&gt;<br>Keep the second line &amp; its exact words\./);
        assert.doesNotMatch(html, /title=/);
        assert.match(scheduleRowHtml({ ...standing, ...fields }, true), /data-schedule-text="daily" open>/);
    }
    const short = scheduleRowHtml({ ...standing, task: { text: 'Read the report' } });
    assert.match(short, /data-i18n-authored>Read the report<\/span>/);
    assert.doesNotMatch(short, /<details/);
    assert.doesNotMatch(scheduleRowHtml(standing), /data-schedule-text/);
    const long = 'Read '.repeat(30);
    assert.match(scheduleRowHtml({ ...standing, description: long }), /<summary[^>]*>Read [^<]+…<\/summary>/);
    assert.ok(scheduleRowHtml({ ...standing, description: long }).includes(long.trim()));
});

test('waiting details are inline for the projected and legacy rows', () => {
    for (const fields of [
        { waiting_detail: { reason: 'workspace_unusable', detail: 'Choose <folder>' } },
        { waiting_detail: 'workspace_unusable: Choose <folder>' },
        { hold: { reason: 'workspace_unusable', detail: 'Choose <folder>' } },
    ]) {
        const html = scheduleRowHtml({ ...standing, ...fields });
        assert.match(html, /waiting: workspace_unusable: Choose &lt;folder&gt;<\/span>/);
        assert.doesNotMatch(html, /title=/);
    }
    assert.doesNotMatch(scheduleRowHtml(standing), /waiting:/);
    assert.match(scheduleRowHtml({ ...standing, followup_hold: { reason: 'origin_stopped', detail: 'Owner paused it' } }),
        /Original work stopped or restarted: Owner paused it/);
    // A follow-up waiting on its original work, as the projection sends it (no occurrence hold).
    const paused = { ...standing, waiting_detail: { reason: '', detail: '', followup_wait: 'origin_owner_paused' } };
    assert.match(scheduleRowHtml(paused), / · Original work is paused/);
    assert.doesNotMatch(scheduleRowHtml(paused), /waiting:/, 'a follow-up wait is not an occurrence hold');
    assert.doesNotMatch(scheduleRowHtml({ ...paused, waiting_detail: { reason: '', detail: '', followup_wait: '' } }),
        /Original work/);
});

test('kind, zone and destination are visible without inventing missing addresses', () => {
    const html = scheduleRowHtml({ ...standing, kind: 'task', timezone: 'Europe/Moscow',
        destination: { project_id: 'p<1>', chat_id: 0 } });
    assert.match(html, /task · 0 9 \* \* \* \(Europe\/Moscow\)/);
    assert.match(html, /project p&lt;1&gt; · chat 0/);
    assert.match(scheduleRowHtml({ ...standing, task: { chat_id: 4 } }), / · chat 4/);
    assert.doesNotMatch(scheduleRowHtml(standing), / · (project|chat) /);
});

test('last-run outcomes keep settled words and use the canonical end time', () => {
    const ended = '2027-01-15T09:00:00Z';
    for (const state of ['completed', 'failed', 'cancelled', 'handed_off']) {
        const html = scheduleRowHtml({ ...standing, last_run_at: '2027-01-14T08:00:00Z',
            outcome: { task_id: 't1', state, ended_at: ended, line: 'Result <checked>' } });
        assert.ok(html.includes(`last run: ${state} · ${scheduleInstantHtml(ended, { includeYear: true })}`));
        assert.match(html, /Result &lt;checked&gt;/);
        assert.doesNotMatch(html, /2027-01-14|running/);
    }
    const untimed = scheduleRowHtml({ ...standing, last_run_at: ended,
        outcome: { state: 'completed', ended_at: null } });
    assert.match(untimed, /last run: completed<\/span>/);
    assert.doesNotMatch(untimed, /<time/);
});

test('a quiet completion shows the run\'s own sentence after ran · no message; a notice stays the host\'s', () => {
    // The server decides what the line is (a notice, the run's sentence, an answer's end); the row shows it as authored words.
    const quiet = { state: 'completed', quiet: true, notice: false, line: 'Checked: <nothing> changed.' };
    const html = scheduleRowHtml({ ...standing, outcome: quiet });
    assert.match(html, /last run: ran · no message · <span data-i18n-authored>Checked: &lt;nothing&gt; changed\.<\/span><\/span>/);
    assert.match(scheduleRowHtml({ ...standing, outcome: { ...quiet, line: '' } }), /last run: ran · no message<\/span>/);
    const notice = scheduleRowHtml({ ...standing,
        outcome: { ...quiet, quiet: false, notice: true, line: 'Unsettled <review>' } });
    assert.match(notice, /last run: completed · notice: Unsettled &lt;review&gt;<\/span>/);
    assert.doesNotMatch(notice, /no message|delivered|data-i18n-authored>Unsettled/);
    const failed = scheduleRowHtml({ ...standing, outcome: { ...quiet, state: 'failed' } });
    assert.match(failed, /last run: failed/);
    assert.doesNotMatch(failed, /ran · no message/);
});

test('unobserved outcomes never imply liveness; only the explicit open-run fact says it is open', () => {
    const row = { ...standing, last_task_id: 't1', outcome: { state: 'pending_observation', task_id: 't1' } };
    assert.match(scheduleRowHtml(row), /last run: outcome not observed yet/);
    assert.doesNotMatch(scheduleRowHtml(row), /running|previous run still open/);
    assert.match(scheduleRowHtml({ ...row, previous_run_open: { task_id: 't<1>', since: '2027-01-15T09:00:00Z' } }),
        /previous run still open: t&lt;1&gt;/);
    assert.doesNotMatch(scheduleRowHtml({ ...standing, last_task_id: 't1' }), /last run:|running|previous run still open/);
});

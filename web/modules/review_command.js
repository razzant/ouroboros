import { apiClient } from './api_client.js';
import { PROCESSING_PREFERENCE_KEY, rosterHandles } from './route_editor_primitives.js';

/** `/review` runs on one enabled catalog row (absent `enabled` = on), named as its Settings card; none = the Main model. */
export async function chooseAndSendReview({ openConfirmDialog, ws, readSettings = apiClient.settings }) {
    let settings, rows;
    try {
        settings = await readSettings();
        const roster = settings?.OUROBOROS_SUBAGENTS;
        rows = roster ? (typeof roster === 'string' ? JSON.parse(roster) : roster).items : [];
    } catch {}
    if (!Array.isArray(rows)) rows = null;
    const handles = rosterHandles(rows, settings?.[PROCESSING_PREFERENCE_KEY]);
    const answer = await openConfirmDialog({
        title: 'Deep self-review', input: true, confirmLabel: 'Queue review',
        body: `Who reviews the whole system?${rows ? '' : '\nThe subagent catalog could not be read, so only Main is available.'}`,
        choices: [{ value: '', label: 'Main model (default)' }, ...(rows || []).flatMap((row, index) =>
            row?.enabled !== false && row?.subagent_id
                ? [{ value: row.subagent_id, label: `Subagent ${index + 1} — ${handles.get(row.subagent_id) || row.subagent_id}` }] : [])],
    });
    if (!answer?.confirmed) return false;
    ws.send({ type: 'command', cmd: `/review ${answer.value || ''}`.trim() });
    return true;
}

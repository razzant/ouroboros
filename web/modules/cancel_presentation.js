import { plainCauseText } from './utils.js';
import { fmt, tr } from './i18n.js';

// Twin of supervisor.cancel_publication.cancel_cause_clauses. IDs stay in
// cancel_origin/lineage details; a compact sentence never guesses an actor.
export const CANCEL_SOURCE_PHRASES = {
    "server_shutdown": "Server shutdown",
    http_single: 'Stopped from the app (Stop now)',
    http_cascade: 'Stopped from the app (Stop now)',
    http_graceful: 'Stopped from the app (Wrap up)',
    cascade_descendant: 'Stopped with the task tree it belongs to',
    owner_restart: 'Stopped by the Restart command',
    snapshot_restore: 'The server stopped while this task was still running',
    agent_tool: 'Stopped by Ouroboros',
    silence_resolved: 'Silence check withdrawn: the observed work no longer needed it',
};
// The English control word each app-stop phrase names (owner decision: these five words stay English).
export const CANCEL_CONTROL_WORDS = { http_single: 'Stop now', http_cascade: 'Stop now', http_graceful: 'Wrap up' };
// Twin of CANCEL_SOURCE_LABELS: a producer's own fixed label restates its phrase.
export const CANCEL_SOURCE_LABELS = {
    http_graceful: 'owner requested finalize-then-stop',
    owner_restart: 'Owner restart',
    snapshot_restore: 'server_shutdown',
};
// Twin of CANCEL_REASON_MAX_CHARS / CANCEL_REASON_PREVIEW_NOTE: the record keeps
// the whole reason, so a shortened clause names itself a preview.
export const CANCEL_REASON_MAX_CHARS = 160;
export const CANCEL_REASON_PREVIEW_NOTE = ' (preview; the full reason is kept with the task)';

function reasonPreview(value) {
    const whole = plainCauseText(value, 0);
    const shown = plainCauseText(value, CANCEL_REASON_MAX_CHARS);
    return shown === whole ? shown : `${shown} ${tr('cancel.reason_preview_note', CANCEL_REASON_PREVIEW_NOTE.trim())}`;
}

export function cancelCauseClauses(origin, record = {}) {
    const source = String(origin.source || '');
    // Ouroboros's own cancel_task stamps the run that asked, not a swept root.
    const asker = source === 'agent_tool';
    const asked = String(origin.requested_by || '');
    const self = String(record.task_id || record.id || record.subagent_task_id || '');
    const parent = String(record.parent_task_id || '');
    const root = String(record.root_task_id || '');
    const actor = origin.request_origin?.kind === 'agent_task' && origin.request_origin.task_id;
    const relation = asked && asked !== self && asked === parent
        ? (asker ? tr('cancel.relation.requested_by_parent', 'Requested by its parent task') : tr('cancel.relation.stopped_with_parent', 'Stopped with its parent task'))
        : asked && asked !== self && parent && asked === root
            ? (asker ? tr('cancel.relation.requested_by_ancestor', 'Requested by an ancestor task') : tr('cancel.relation.stopped_with_ancestor', 'Stopped with an ancestor task')) : '';
    const stated = String(origin.reason || '');
    const label = Object.hasOwn(CANCEL_SOURCE_LABELS, source) ? CANCEL_SOURCE_LABELS[source] : null;
    const control = origin.stop_policy === 'immediate' ? 'Stop now'
        : origin.stop_policy === 'finalize_then_cancel' ? 'Wrap up' : CANCEL_CONTROL_WORDS[source];
    // The English phrases stay the source (twin of the Python table); an install language reads
    // them by code (web/modules/i18n.js). The five task-control words stay English by decision:
    // a phrase that names one carries it as an operand the translation cannot touch. A source
    // with no phrase stays raw in every language.
    return [
        Object.hasOwn(CANCEL_CONTROL_WORDS, source)
            ? fmt('Stopped from the app ({control})', { control })
            : Object.hasOwn(CANCEL_SOURCE_PHRASES, source) ? tr(`cancel.source.${source}`, CANCEL_SOURCE_PHRASES[source]) : source,
        stated.split(/\s+/).filter(Boolean).join(' ') === label ? '' : reasonPreview(stated),
        origin.scope === 'cascade' ? tr('cancel.scope_cascade', 'this task and its sub-tasks') : '',
        relation,
        actor && !(asker && relation && String(actor) === asked) ? tr('cancel.relation.requested_by_task', 'Requested by a task') : '',
    ];
}

// [command, ending] of a record of the owner-stop rail; twin of owner_stop.owner_stop_ending. The
// command is the typed stop its terminal carries (cancel_origin.stop_policy); the ending is 'stop'
// for Stop now and for a Wrap up without a final answer, 'summary' otherwise (older records too).
export function ownerStopEnding(record = {}) {
    const policy = String(record.cancel_origin?.stop_policy || '');
    const command = policy === 'immediate' ? 'stop_now' : policy === 'finalize_then_cancel' ? 'wrap_up' : '';
    const reason = String(record?.reason_code || '');
    if (reason === 'provider_unavailable' && command === 'stop_now'
        && record.cancel_origin?.control_reason === 'owner_stopped_direct_chat_turn') return [command, 'stop'];
    if (reason !== 'owner_requested_finalization') return ['', ''];
    const unanswered = command === 'wrap_up' && String(record.outcome_axes?.execution?.status || '') === 'failed';
    return [command, command === 'stop_now' || unanswered ? 'stop' : 'summary'];
}

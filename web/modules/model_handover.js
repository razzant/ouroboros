import { tr } from './i18n.js';
import { escapeHtmlText } from './utils.js';
import { compactModel, displayApiModelTarget } from './harness_presentation.js';

const selectionWords = {
    answer_sha256: () => tr('task.handover.selected.answer_sha256', 'confirmed a saved answer'),
    answer: () => tr('task.handover.selected.answer', 'submitted an answer'),
    silent: () => tr('task.handover.selected.silent', 'finished without a final message'),
    stop: () => tr('task.handover.selected.stop', 'chose to stop'),
};

// Older route pairs include intended routes, not necessarily replies. A typed selection
// or successful tool call also proves an act; an incomplete status alone does not. A helper's card (`child`) names
// each endpoint as its solve row does, without the local API connection; the stored routes keep it.
export function modelHandoverLines(rows, child = false) {
    const shown = (route) => (child ? displayApiModelTarget(route) || compactModel(route, false) : route);
    return (Array.isArray(rows) ? rows : []).filter(row => row?.from_model && row?.to_model).map(row => {
        const selected = row.status === 'selected' && typeof row.selection === 'string' && Object.hasOwn(selectionWords, row.selection);
        // The same response can work and then choose (`accept_handover_selection`): every fact a row states is said,
        // and one never hides another. A row that names no work says nothing about work; `recovered` alone proves none.
        const worked = row.work_observed === true ? tr('task.handover.tool_succeeded', 'completed a tool call') : '';
        const usedTool = row.status === 'recovered' && row.work_observed === true;
        const observed = row.reply_observed === true || selected || usedTool;
        const stated = selected
            ? selectionWords[row.selection]()
            : row.status === 'incomplete' ? tr('task.handover.incomplete', 'handover recorded as incomplete')
            : row.status === 'pending' && observed ? tr('task.handover.pending', 'no choice recorded') : '';
        const act = [worked, stated].filter(Boolean).join(' · ');
        const unknown = !observed && tr('task.handover.reply_unrecorded', 'successor reply not recorded');
        // Details keeps the route prefixes: they may be all that changed.
        return [`${shown(row.from_model)} → ${shown(row.to_model)}`, unknown, act].filter(Boolean).join(' · ');
    });
}

export function modelHandoverFact(rows, child = false) {
    return [tr('task.fact.model_handover', 'Model handover'), modelHandoverLines(rows, child).map(escapeHtmlText).join('<br>'), true];
}

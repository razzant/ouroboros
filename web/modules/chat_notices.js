import { showToast } from './toast.js';

// Shared by every Main/Project chat instance on the page: a Project incident is
// mirrored into Main, but must still produce exactly one toast.
const shownIncidentToastKeys = new Set();

export function showTaskIncidentToast(msg) {
    const incident = String(msg?.task_incident || '').trim();
    if (!incident) return;
    const key = String(msg?.toast_once || `${msg?.task_id || ''}:${incident}`).trim();
    if (!key || shownIncidentToastKeys.has(key)) return;
    shownIncidentToastKeys.add(key);
    if (shownIncidentToastKeys.size > 500) {
        const oldest = shownIncidentToastKeys.values().next().value;
        shownIncidentToastKeys.delete(oldest);
    }
    showToast(String(msg?.content || msg?.text || incident), 'error');
}

export function showContextFitToast(evt) {
    if (evt?.checkpoint_kind !== 'context_fit_low_retry') return;
    const key = `context-fit:${String(evt?.toast_once || `${evt?.task_id || ''}:${evt?.round || ''}`)}`;
    if (shownIncidentToastKeys.has(key)) return;
    shownIncidentToastKeys.add(key);
    if (shownIncidentToastKeys.size > 500) {
        const oldest = shownIncidentToastKeys.values().next().value;
        shownIncidentToastKeys.delete(oldest);
    }
    showToast('Context exceeded this route. Retrying the same model once with the task-local Low view.', 'warn');
}

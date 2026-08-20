// Pure chat-card disclosure and lifecycle projections. This module owns no
// listeners, timers, or DOM creation; chat.js remains the instance lifecycle
// and render orchestrator.
//
// Ownership is intentionally narrow:
// - disclosure maps a click to an existing live-line key;
// - sticky state resets or projects fields on a caller-owned card record;
// - the terminal helper classifies host-provided lifecycle facts.
//
// Reusable-card identity and cancel eligibility deliberately live in
// task_control_menu.js, which owns the shared stop/hurry control: one owner,
// one REUSABLE_TASK_IDS Set.
//
// The functions accept DOM-shaped values because that is the established
// public contract, but they never query global document/window state. This
// keeps the owner dependency-free and directly executable in the Node suite.
// chat.js imports and re-exports these bindings directly, so external imports
// keep the same identity across the facade.

// Row-surface disclosure guard (v6.71.0), pure for node tests: returns the
// lineKey to toggle for a click landing on `target`, or '' when the click must
// NOT toggle (nested interactive element, or an active text selection inside
// the line).
export function liveLineRowToggleKey(target, selection = null) {
    const line = target?.closest?.('.chat-live-line.expandable');
    if (!line) return '';
    if (target.closest('button, a, input, textarea, select, label, summary, [contenteditable="true"]')) return '';
    if (selection && !selection.isCollapsed && line.contains(selection.anchorNode)) return '';
    return (line.dataset && line.dataset.liveLineKey) || '';
}

/**
 * Reset the sticky presentation state (collapsed activity + cost projection)
 * introduced in v6.82 P1. Used by resetLiveCardRecord; pure over the record
 * shape so dependency-free node tests can exercise the recycle path.
 */
export function clearStickyCardState(record) {
    if (!record) return record;
    record.collapsedActivity = '';
    record.costMeta = null;
    // A recycled slot must not inherit the previous cycle's finalizing hold.
    record.finalizingHold = false;
    // The activity clock is cycle state too: a
    // recycled slot ('bg-consciousness', 'active') would otherwise open showing
    // the previous cycle's "Latest" time.
    record.latestActivityTs = '';
    if (record.activityEl) {
        record.activityEl.textContent = '';
        record.activityEl.removeAttribute('title');
    }
    return record;
}

/**
 * Decide the collapsed activity line text (v6.82 P1), shared by root and
 * subagent cards. Root cards show the latest activity headline ONLY when a
 * coined name occupies the title — an unnamed card's title already shows the
 * activity, so the line is suppressed to avoid duplication. Subagent titles
 * keep the role·model·id identity, so their routed progress body always feeds
 * the line. A frame without new activity keeps `previous`, so finishing a card
 * never blanks its last activity. Geometry is owned by the two-line CSS clamp;
 * this character ceiling is only a defensive DOM/accessibility bound.
 */
export const COLLAPSED_ACTIVITY_MAX = 240;

export function boundActivityPreview(value = '') {
    const candidate = String(value || '').replace(/\s+/g, ' ').trim();
    if (candidate.length <= COLLAPSED_ACTIVITY_MAX) return candidate;
    return candidate.slice(0, COLLAPSED_ACTIVITY_MAX - 1).trimEnd() + '…';
}

export function projectCollapsedActivity({
    isSubagent = false, suggestedName = '', headline = '', body = '', previous = '',
} = {}) {
    const current = boundActivityPreview(isSubagent ? body : headline);
    const candidate = current || boundActivityPreview(previous);
    if (!isSubagent && !String(suggestedName || '').trim()) return '';
    return candidate;
}

// v6.82 (P5): terminal card phases. 'cancelled' is a first-class terminal phase
// so a force-cancelled root resolves its card instead of re-inflating.
export function isTerminalTaskPhase(phase = '', terminal = false) {
    return Boolean(terminal) || ['done', 'lifecycle_error', 'cancelled'].includes(phase);
}

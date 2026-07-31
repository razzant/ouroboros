import { escapeHtmlAttr, escapeHtmlText as escapeHtml, formatUsdWhole, renderMarkdown } from './utils.js';
import { renderPageHeader } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';
import { showToast } from './toast.js';
import { downloadViaHostBridge, openViaHostBridge } from './ui_helpers.js';
import { apiClient, apiFetch, cancelTask } from './api_client.js';
import {
    compactModel,
    formatReviewProjection,
    getLogTaskGroupId,
    isGroupedTaskEvent,
    normalizeLogTs,
    summarizeChatLiveEvent,
    taskOutcomeSeverity,
    taskTerminalPhase,
} from './log_events.js';
import { openConfirmDialog } from './confirm_dialog.js';

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

/** Convert a raw source timestamp to sortable epoch milliseconds. */
export function rawTimestampEpoch(raw) {
    if (raw == null || raw === '') return NaN;
    const epoch = typeof raw === 'number' ? raw : Date.parse(String(raw));
    return Number.isFinite(epoch) ? epoch : NaN;
}

/**
 * Insert a top-level timeline node chronologically while keeping typing last.
 * Equal timestamps preserve arrival order; timestamp-free nodes append.
 */
export function insertTimelineNode(messages, node, typing = null, { stickToBottom = false } = {}) {
    const previousScrollTop = Number(messages?.scrollTop) || 0;
    const previousScrollHeight = Number(messages?.scrollHeight) || 0;
    const rawNodeTs = node?.dataset?.ts;
    const nodeTs = rawNodeTs == null || rawNodeTs === '' ? NaN : Number(rawNodeTs);
    let before = null;
    if (Number.isFinite(nodeTs)) {
        for (const child of Array.from(messages?.children || [])) {
            if (child === node || child === typing) continue;
            const rawChildTs = child?.dataset?.ts;
            const childTs = rawChildTs == null || rawChildTs === '' ? NaN : Number(rawChildTs);
            if (Number.isFinite(childTs) && childTs > nodeTs) {
                before = child;
                break;
            }
        }
    }
    let insertedAboveViewport = false;
    if (
        before
        && !stickToBottom
        && typeof before.getBoundingClientRect === 'function'
        && typeof messages.getBoundingClientRect === 'function'
    ) {
        insertedAboveViewport = before.getBoundingClientRect().top <= messages.getBoundingClientRect().top;
    }
    if (before) messages.insertBefore(node, before);
    else if (typing && typing.parentNode === messages) messages.insertBefore(node, typing);
    else messages.appendChild(node);

    const nextScrollHeight = Number(messages?.scrollHeight) || 0;
    if (stickToBottom) {
        messages.scrollTop = nextScrollHeight;
    } else if (insertedAboveViewport) {
        messages.scrollTop = previousScrollTop + Math.max(0, nextScrollHeight - previousScrollHeight);
    }
    return { before, insertedAboveViewport };
}

const CHAT_STORAGE_KEY = 'ouro_chat';
const CHAT_INPUT_HISTORY_KEY = 'ouro_chat_input_history';
const CHAT_SESSION_ID_KEY = 'ouro_chat_session_id';
const MAX_PENDING_ATTACHMENTS = 10;
const MAX_ATTACHMENT_FILE_BYTES = 50 * 1024 * 1024;
const MAX_PENDING_ATTACHMENT_BYTES = 100 * 1024 * 1024;
// Shared by every Main/Project chat instance on the page: a Project incident is
// mirrored into Main, but must still produce exactly one toast.
const shownIncidentToastKeys = new Set();

function optionalFiniteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

/** Pure presentation projection used by the header and dependency-free tests. */
export function headerBudgetPresentation(data) {
    if (!data || data.accounting_loading === true) {
        return { state: 'loading', label: 'Loading…', fillPct: 0 };
    }
    if (data?.accounting?.available === false) {
        return { state: 'unavailable', label: 'Unavailable', fillPct: 0 };
    }
    // Older state shapes did not carry accounting.available.  Keep accepting
    // them when they contain a real numeric projection, but never coerce null
    // (ledger failure in the new shape) into a convincing $0.
    const spent = optionalFiniteNumber(data.spent_usd);
    if (spent === null) {
        return { state: 'unavailable', label: 'Unavailable', fillPct: 0 };
    }
    const rawLimit = optionalFiniteNumber(data.budget_limit);
    const limit = rawLimit !== null && rawLimit > 0 ? rawLimit : 0;
    const label = typeof data.budget_text === 'string' && data.budget_text.trim()
        ? data.budget_text
        : `${formatUsdWhole(spent)} / ${limit > 0 ? formatUsdWhole(limit) : '∞'}`;
    return {
        state: 'available',
        label,
        fillPct: limit > 0 ? Math.min(100, Math.max(0, (spent / limit) * 100)) : 0,
    };
}

/**
 * Render task money without conflating unknown/non-final values with a final
 * zero.  The returned strings are card metadata, not another cost authority.
 */
export function taskCostMeta(payload = {}) {
    const has = (key) => Object.prototype.hasOwnProperty.call(payload, key);
    // Task-scope accounting evidence only (v6.82 P1): a bare `cost_usd` is NOT
    // enough — llm_round_finished carries a per-round delta under that key, and
    // rendering it as task cost lied on the card. Subagent progress_meta and
    // task_done/task_cost_finalized frames carry cost_accounting_status /
    // cost_final alongside cost_usd, so honest task-scope frames still qualify.
    const hasAccountingEvidence = [
        'cost_accounting_status', 'cost_final',
        'cost_usd_with_children', 'cost_with_children_partial',
        'reserved_usd', 'unresolved_upper_bound_usd', 'unknown_unmetered',
    ].some(has);
    if (!hasAccountingEvidence) return [];
    if (payload.cost_accounting_status === 'unavailable') return ['cost unavailable'];

    const own = optionalFiniteNumber(payload.cost_usd);
    const finalKnown = payload.cost_final === true;
    const pendingKnown = payload.cost_final === false
        || payload.cost_with_children_partial === true
        || payload.cost_accounting_status === 'available' && !has('cost_final');
    const meta = [];
    if (own === null) {
        meta.push('cost pending');
    } else if (finalKnown || pendingKnown || own !== 0) {
        meta.push(`cost=$${own.toFixed(2)}${pendingKnown && !finalKnown ? ' (pending)' : ''}`);
    }

    const subtree = optionalFiniteNumber(payload.cost_usd_with_children);
    if (subtree !== null && (
        own === null || subtree !== own || payload.cost_with_children_partial === true
    )) {
        const partial = payload.cost_with_children_partial === true || !finalKnown;
        meta.push(`subtree=$${subtree.toFixed(2)}${partial ? ' (pending)' : ''}`);
    }
    const reserved = optionalFiniteNumber(payload.reserved_usd);
    if (reserved !== null && reserved > 0) meta.push(`reserved=$${reserved.toFixed(2)}`);
    const unresolved = optionalFiniteNumber(payload.unresolved_upper_bound_usd);
    if (unresolved !== null && unresolved > 0) meta.push(`unresolved≤$${unresolved.toFixed(2)}`);
    const unknown = optionalFiniteNumber(payload.unknown_unmetered);
    if (unknown !== null && unknown > 0) meta.push(`unmetered=${Math.trunc(unknown)}`);
    return meta;
}

/**
 * Project one frame's task-scope cost evidence into the sticky structured form
 * `{meta, ts, final}` (v6.82 P1). Returns null when the frame carries NO
 * task-scope accounting evidence (e.g. an llm_round_finished per-round delta)
 * — such frames must never touch a card's cost.
 */
export function taskCostProjection(payload = {}, rawTs = '') {
    const meta = taskCostMeta(payload);
    if (!meta.length) return null;
    const unavailable = payload.cost_accounting_status === 'unavailable';
    return {
        meta,
        ts: rawTimestampEpoch(rawTs),
        // Only a SETTLED ledger value is final. "unavailable" is an honest
        // unknown, not a settled truth: marking it final let one transient
        // ledger-read failure outrank every later real reading.
        final: payload.cost_final === true,
        unavailable,
    };
}

/**
 * Sticky per-card cost precedence (v6.82 P1). Rank unavailable < pending < final:
 * an honest reading always outranks an unknown (one transient ledger-read failure
 * must not pin the card for the whole run) and a settled value outranks both.
 * Among equal rank the newer raw source timestamp wins, so an older history replay
 * can never overwrite newer evidence; frames without evidence (null `next`) keep
 * the previous projection, so an unavailable snapshot is still sticky.
 */
export function mergeStickyCostMeta(previous, next) {
    if (!next || !Array.isArray(next.meta) || !next.meta.length) return previous || null;
    if (!previous || !Array.isArray(previous.meta) || !previous.meta.length) return next;
    // Rank: unavailable < pending < final. An `unavailable` snapshot is sticky (a
    // costless frame must not erase it) but must NOT outrank a later HONEST reading:
    // one transient ledger-read failure would otherwise pin the card to "cost
    // unavailable" for the rest of the run.
    const rank = (p) => (p.final ? 2 : (p.unavailable ? 0 : 1));
    const prevRank = rank(previous);
    const nextRank = rank(next);
    if (prevRank !== nextRank) return nextRank > prevRank ? next : previous;
    const prevTs = Number(previous.ts);
    const nextTs = Number(next.ts);
    if (Number.isFinite(prevTs) && Number.isFinite(nextTs) && nextTs < prevTs) return previous;
    // A frame whose source timestamp is unreadable must not defeat a
    // timestamped previous value of equal finality.
    if (Number.isFinite(prevTs) && !Number.isFinite(nextTs)) return previous;
    return next;
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
    // The activity CLOCK and the full-text reference are cycle state too: a
    // recycled slot ('bg-consciousness', 'active') would otherwise open showing
    // the previous cycle's "Latest" time and its tooltip.
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
 * never blanks its last activity.
 */
export const COLLAPSED_ACTIVITY_MAX = 400;

export function projectCollapsedActivity({
    isSubagent = false, suggestedName = '', headline = '', body = '', previous = '',
} = {}) {
    const candidate = String((isSubagent ? body : headline) || '').trim()
        || String(previous || '').trim();
    if (!isSubagent && !String(suggestedName || '').trim()) return '';
    // BOUNDED, and the bound is DISCLOSED. The line is fed the uncut text (never a
    // timeline preview), but a single card line is not a place to render an
    // unbounded narration: it is clipped at a generous ceiling with an explicit
    // ellipsis, and applyLiveCardState keeps the complete text on the element's
    // title so the full value stays reachable rather than destroyed (BIBLE P1 —
    // truncation is allowed when it is visible and the whole text survives).
    if (candidate.length <= COLLAPSED_ACTIVITY_MAX) return candidate;
    return candidate.slice(0, COLLAPSED_ACTIVITY_MAX - 1).trimEnd() + '…';
}

// Logical slots that may host multiple independent cycles (v6.82: hoisted to
// module scope so cancelRunEligibility shares the same truth as the card layer).
export const REUSABLE_TASK_IDS = new Set(['bg-consciousness', 'active']);

// v6.82 (P5): terminal card phases. 'cancelled' is a first-class terminal phase
// so a force-cancelled root resolves its card instead of re-inflating.
export function isTerminalTaskPhase(phase = '', terminal = false) {
    return Boolean(terminal) || ['done', 'lifecycle_error', 'cancelled'].includes(phase);
}

/**
 * v6.82 (P5): may this live card offer the "Cancel run" action?
 * Card shape alone cannot answer it — an in-process direct-chat turn mints an
 * ordinary non-reusable, non-subagent card (supervisor/workers.py builds it a
 * real uuid task id) yet has no queue entry to cancel. So eligibility requires
 * the supervisor's host-attested `cancelable` progress-meta marker on top of
 * the structural gates: a ROOT (non-subagent) pooled card, not a reusable slot,
 * not finished, not converted into a project chip.
 */
export function cancelRunEligibility({
    groupId = '', isSubagent = false, finished = false, cancelable = false, converted = false,
} = {}) {
    return Boolean(cancelable)
        && !isSubagent
        && !finished
        && !converted
        && Boolean(String(groupId || '').trim())
        && !REUSABLE_TASK_IDS.has(String(groupId || ''));
}

function withTaskCostMeta(summary, payload, { replace = false, rawTs = '' } = {}) {
    const projection = taskCostProjection(payload, rawTs);
    // `replace` frames (task_done/task_cost_finalized) never keep the
    // summarizer's own meta strings — even without accounting evidence a
    // terminal frame must not render an ungated bare cost string.
    // Cost renders from the card's STICKY record.costMeta (applyLiveCardState),
    // never from this frame's meta list: the sticky projection is the SINGLE
    // cost renderer. Summarizer-built `cost=` strings are dropped
    // UNCONDITIONALLY — a frame whose payload carries no task-scope accounting
    // evidence must show no money at all, not a bare per-call number.
    const base = replace ? { ...summary, meta: [] } : summary;
    const out = projection ? { ...base, costProjection: projection } : { ...base };
    if (Array.isArray(out.meta) && out.meta.length) {
        out.meta = out.meta.filter((entry) => !String(entry || '').startsWith('cost='));
    }
    return out;
}

function showTaskIncidentToast(msg) {
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

function showContextFitToast(evt) {
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

function getOrCreateChatSessionId() {
    try {
        const existing = sessionStorage.getItem(CHAT_SESSION_ID_KEY);
        if (existing) return existing;
        const created = (globalThis.crypto && typeof crypto.randomUUID === 'function')
            ? crypto.randomUUID()
            : `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        sessionStorage.setItem(CHAT_SESSION_ID_KEY, created);
        return created;
    } catch {
        return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
}

function projectIdFromTask(taskId = '') {
    const seed = String(taskId || '')
        .toLowerCase()
        .replace(/[^a-z0-9_.-]+/g, '-')
        .replace(/^-+|-+$/g, '');
    return (seed ? `task-${seed}` : `task-${Date.now().toString(36)}`).slice(0, 64);
}

function loadInputHistory() {
    try {
        const raw = JSON.parse(sessionStorage.getItem(CHAT_INPUT_HISTORY_KEY) || '[]');
        return Array.isArray(raw) ? raw.filter(Boolean).slice(-50) : [];
    } catch {
        return [];
    }
}

function saveInputHistory(entries) {
    try {
        sessionStorage.setItem(CHAT_INPUT_HISTORY_KEY, JSON.stringify(entries.slice(-50)));
    } catch {}
}

export function initChat(ctx) {
    // Back-compat main-chat entry: one full-page instance bound to chat 1.
    return createChatInstance(ctx);
}

export function createChatInstance({
    ws, state, updateUnreadBadge, openSettingsTab, openDashboardTab,
    chatId = 1, projectId = '', idPrefix = 'chat', mountEl = null,
    asPanel = false, title = 'Chat',
}) {
    const container = mountEl || document.getElementById('content');
    const chatSessionId = getOrCreateChatSessionId();
    const isMain = chatId === 1;
    // Per-thread storage so a project thread never bleeds into the main chat.
    const storeKey = (base) => (isMain ? base : `${base}:${chatId}`);

    const page = document.createElement('div');
    page.id = asPanel ? `panel-${idPrefix}` : 'page-chat';
    page.className = asPanel ? 'chat-instance-panel' : 'page active';
    // A project panel reuses the lean `.project-panel-bar` (title + close) from
    // index.html, so it renders a minimal status-only header — NOT the overlay
    // page header (that would duplicate the title and drag in the GLOBAL
    // Evolve/Restart/Panic/budget chrome, which belongs to the one agent, not a
    // single project thread). The main chat keeps the full overlay header.
    const headerHtml = asPanel
        ? `<div class="chat-panel-statusbar"><span id="chat-status" class="status-badge offline">Connecting...</span></div>`
        : renderPageHeader({
            title: title,
            icon: PAGE_ICONS.chat,
            variant: 'overlay',
            className: 'chat-page-header',
            actionsHtml: `
                <div class="chat-header-actions" id="chat-header-actions">
                    <button class="chat-header-btn" type="button" data-chat-command="restart" title="Restart agent">Restart</button>
                    <button class="chat-header-btn danger" type="button" data-chat-command="panic" title="Stop all workers">Panic</button>
                    <details class="chat-header-more">
                        <summary class="chat-header-btn" title="More agent controls">More</summary>
                        <div class="chat-header-menu">
                            <button class="chat-header-menu-item" type="button" data-chat-command="bg" title="Toggle background consciousness">Consciousness</button>
                            <button class="chat-header-menu-item" type="button" data-chat-command="evolve" title="Toggle evolution mode">Evolve</button>
                            <button class="chat-header-menu-item" type="button" data-chat-command="review" title="Run review now">Review</button>
                        </div>
                    </details>
                </div>
                <button class="chat-budget-pill" id="chat-budget-pill" type="button" title="Open budget controls" aria-label="Open budget controls">
                    <span class="chat-budget-text" id="chat-budget-text">Loading…</span>
                    <div class="chat-budget-bar">
                        <div class="chat-budget-bar-fill" id="chat-budget-bar-fill"></div>
                    </div>
                </button>
                <span id="chat-status" class="status-badge offline">Connecting...</span>
            `,
        });
    page.innerHTML = `
        ${headerHtml}
        <div id="chat-messages"></div>
        <div id="chat-input-area">
            <div id="chat-attachment-preview" class="chat-attachment-preview"></div>
            <div class="chat-input-wrap">
                <div class="chat-toolbar-row">
                    <div class="chat-composer-pills" id="chat-composer-pills">
                        <button class="chat-swarm" id="chat-swarm" type="button" data-armed="false" title="Swarm: arm a one-shot deep plan + multi-subagent fan-out (plan_task + web search, then delegate) for your next message. Auto-disarms after sending.">Swarm</button>
                        <div class="chat-context-mode" id="chat-context-mode" data-context-mode="max" role="group" aria-label="Context size mode" title="Context mode (owner setting). Low fits ~200K / local models; Max is full. Saves immediately; lowering to Low requires Ouroboros to be idle.">
                            <button class="chat-seg" type="button" data-mode="low">Low</button>
                            <button class="chat-seg" type="button" data-mode="max">Max</button>
                        </div>
                    </div>
                </div>
                <div class="chat-text-row">
                    <button class="chat-attach-btn" id="chat-attach" type="button" title="Attach file">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
                    </button>
                    <input type="file" id="chat-file-input" class="chat-file-input-hidden" accept="*/*" multiple>
                    <textarea id="chat-input" placeholder="Message Ouroboros..." rows="1" autocorrect="off" autocapitalize="off" spellcheck="false"></textarea>
                    <div class="chat-send-group">
                        <button class="chat-scroll-bottom-btn" id="chat-scroll-bottom" type="button" aria-label="Scroll to latest message" title="Scroll to latest message">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="M19 12l-7 7-7-7"/></svg>
                        </button>
                        <button class="chat-send-inline" id="chat-send" title="Send message">Send</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    if (idPrefix !== 'chat') {
        // Instance-namespaced ids + mirror classes so the shared #chat-* CSS
        // (extended with .chat-* twins) keeps styling secondary instances.
        page.querySelectorAll('[id]').forEach((el) => {
            if (el.id.startsWith('chat-')) {
                el.classList.add(el.id);
                el.id = idPrefix + '-' + el.id.slice(5);
            }
        });
    }
    container.appendChild(page);

    const byId = (suffix) => page.querySelector(`[id="${idPrefix}-${suffix}"]`);
    const messagesDiv = byId('messages');
    const input = byId('input');
    const inputArea = byId('input-area');
    const sendBtn = byId('send');
    const statusBadge = byId('status');
    const headerActions = byId('header-actions');
    const pageHeader = page.querySelector('.chat-page-header');
    const budgetPill = byId('budget-pill');
    const attachBtn = byId('attach');
    const fileInput = byId('file-input');
    const attachmentPreview = byId('attachment-preview');
    const scrollBottomBtn = byId('scroll-bottom');
    let pendingAttachments = [];
    let attachmentsUploading = false;
    let nestedSubagentsExpanded = false;

    async function loadUiPreferences() {
        try {
            const prefs = await apiClient.uiPreferences();
            nestedSubagentsExpanded = prefs?.nested_subagents_expanded === true;
        } catch {
            nestedSubagentsExpanded = false;
        }
    }

    function pendingAttachmentBytes(items = pendingAttachments) {
        return items.reduce((total, item) => total + Number(item.file?.size || 0), 0);
    }

    function updateAttachmentPreview() {
        if (!pendingAttachments.length) {
            attachmentPreview.classList.remove('visible');
            attachmentPreview.innerHTML = '';
            requestAnimationFrame(() => updateMessagesPadding({ preserveStickiness: false }));
            return;
        }
        attachmentPreview.classList.add('visible');
        attachmentPreview.innerHTML = pendingAttachments.map((item) => `
            <span class="attach-badge" data-attachment-id="${escapeHtmlAttr(item.id)}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>
                <span class="attach-name" title="${escapeHtmlAttr(item.display_name)}">${escapeHtml(item.display_name)}</span>
                <button class="attach-remove" type="button" title="Remove" aria-label="Remove attachment ${escapeHtmlAttr(item.display_name)}" data-attachment-remove="${escapeHtmlAttr(item.id)}" ${attachmentsUploading ? 'disabled aria-disabled="true"' : ''}>×</button>
            </span>
        `).join('');
        requestAnimationFrame(() => updateMessagesPadding({ preserveStickiness: false }));
        attachmentPreview.querySelectorAll('[data-attachment-remove]').forEach((button) => {
            button.addEventListener('click', () => {
                if (attachmentsUploading) return;
                const removeId = button.getAttribute('data-attachment-remove') || '';
                pendingAttachments = pendingAttachments.filter((item) => item.id !== removeId);
                updateAttachmentPreview();
            });
        });
    }

    // Shared paperclip/paste stager; upload still happens only on Send.
    function stagePendingFiles(files) {
        const incoming = Array.from(files || []).filter(Boolean);
        if (!incoming.length) return;
        if (attachmentsUploading) {
            showToast('Wait for the current upload to finish before changing attachments.', 'error');
            return;
        }
        if (pendingAttachments.length + incoming.length > MAX_PENDING_ATTACHMENTS) {
            showToast(`Attach up to ${MAX_PENDING_ATTACHMENTS} files per message.`, 'error');
            return;
        }
        const oversized = incoming.find((file) => Number(file.size || 0) > MAX_ATTACHMENT_FILE_BYTES);
        if (oversized) {
            showToast(`Each attachment must be ${Math.round(MAX_ATTACHMENT_FILE_BYTES / (1024 * 1024))} MB or smaller.`, 'error');
            return;
        }
        const incomingBytes = incoming.reduce((total, file) => total + Number(file.size || 0), 0);
        if (pendingAttachmentBytes() + incomingBytes > MAX_PENDING_ATTACHMENT_BYTES) {
            const limitMb = Math.round(MAX_PENDING_ATTACHMENT_BYTES / (1024 * 1024));
            showToast(`Attachments are limited to ${limitMb} MB total per message.`, 'error');
            return;
        }
        pendingAttachments = pendingAttachments.concat(incoming.map((file) => ({
            id: (globalThis.crypto && typeof crypto.randomUUID === 'function')
                ? crypto.randomUUID()
                : `attachment-${Date.now()}-${Math.random().toString(16).slice(2)}`,
            file,
            display_name: file.name || 'upload',
        })));
        updateAttachmentPreview();
    }

    async function cleanupUploadedAttachments(uploaded) {
        const filenames = uploaded
            .map((item) => item.filename)
            .filter(Boolean);
        if (!filenames.length) return;
        const results = await Promise.allSettled(filenames.map(async (filename) => {
            const resp = await apiFetch('/api/chat/upload', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename }),
            });
            if (!resp.ok) throw new Error(`DELETE ${filename} failed with HTTP ${resp.status}`);
        }));
        const failed = results.filter((result) => result.status === 'rejected');
        if (failed.length) {
            console.warn('Failed to clean up uploaded chat attachments after send failure', failed);
        }
    }

    function setAttachmentUploadState(uploading) {
        attachmentsUploading = uploading;
        attachBtn.disabled = uploading;
        attachBtn.classList.toggle('uploading', uploading);
        fileInput.disabled = uploading;
        input.disabled = uploading;
        updateAttachmentPreview();
    }

    attachBtn.addEventListener('click', () => fileInput.click());

    // Local-only staging avoids orphan uploads and fast-send races.
    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files || []);
        fileInput.value = '';
        stagePendingFiles(files);
    });

    // Image paste uses the same stager; only image matches call preventDefault().
    // Timestamped names keep repeated clipboard images distinct.
    input.addEventListener('paste', (e) => {
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        const pastedImages = [];
        for (let i = 0; i < items.length; i += 1) {
            const item = items[i];
            if (item && item.kind === 'file' && typeof item.type === 'string' && item.type.startsWith('image/')) {
                const blob = item.getAsFile();
                if (!blob) continue;
                const ext = (item.type.split('/')[1] || 'png').split(';')[0].trim() || 'png';
                const ts = Date.now() + i;
                const safeBlob = blob instanceof File
                    ? new File([blob], `clipboard-${ts}.${ext}`, { type: blob.type })
                    : new File([blob], `clipboard-${ts}.${ext}`, { type: item.type });
                pastedImages.push(safeBlob);
            }
        }
        if (!pastedImages.length) return;
        e.preventDefault();
        stagePendingFiles(pastedImages);
    });

    let fileDragDepth = 0;
    function isFileDrag(event) {
        return Array.from(event.dataTransfer?.types || []).includes('Files');
    }
    function setFileDragActive(active) {
        inputArea.classList.toggle('drag-active', Boolean(active));
    }
    page.addEventListener('dragenter', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        fileDragDepth += 1;
        setFileDragActive(true);
    });
    page.addEventListener('dragover', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
        setFileDragActive(true);
    });
    page.addEventListener('dragleave', (event) => {
        if (!isFileDrag(event)) return;
        fileDragDepth = Math.max(0, fileDragDepth - 1);
        if (fileDragDepth === 0) setFileDragActive(false);
    });
    page.addEventListener('drop', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        fileDragDepth = 0;
        setFileDragActive(false);
        stagePendingFiles(event.dataTransfer?.files || []);
    });

    // Pass 1 builds live cards in memory; pass 2 inserts them in transcript order.
    let _syncPass1Active = false;

    const persistedHistory = [];
    const seenMessageKeys = new Set();
    const messageKeyOrder = [];
    const pendingUserBubbles = new Map();
    const inputHistory = loadInputHistory();
    let inputHistoryIndex = inputHistory.length;
    let inputDraft = '';
    let historyLoaded = false;
    let inputHistorySeededFromServer = false; // set true only after a successful server-side recall seed
    let historySyncPromise = null;
    let lastHistorySyncSucceeded = false;
    let historyPaintGeneration = 0;
    let welcomeShown = false;
    const liveCardRecords = new Map();
    // Reusable slots (bg-consciousness, active) destroy+recreate their card on every
    // new cycle and auto-collapse on each cycle finish. Remember the owner's explicit
    // expand per slot so cycle churn restores it instead of snapping the card shut.
    const stickyExpandedSlots = new Set();
    // Cluster B: a proactively-coined name (task_named) can arrive BEFORE the card's
    // liveCardRecords entry exists (the namer broadcasts as the task starts). Buffer it
    // here so createLiveCardRecord can apply it when the card appears (no lost title).
    const pendingSuggestedNames = new Map();
    const taskUiStates = new Map();
    // Busy-chat decision turns reuse the normal agent/event path for ordering and
    // observability, but they are not user tasks. Their structural backend marker
    // suppresses the transient card while preserving the typed routing annotation
    // and any non-empty final conversational answer as separate UI roles.
    const ephemeralDecisionTaskIds = new Set();
    // Finished task ids hidden from routine syncs until reload/reconnect rebuilds history.
    const retiredTaskIds = new Set();
    // The owner's last main-chat request, handed to the next live card it spawns so a
    // "turn into project" conversion can name the project from it (P1).
    let _pendingCardObjective = '';
    let activeLiveGroupId = '';
    let historySyncTimer = null;
    let pendingReconnectSync = false;  // Set when a fromReconnect sync arrives while one is already in-flight.
    let pendingReconnectBannerText = readPendingReconnectBanner();

    function registerEphemeralDecisionFrame(frame) {
        const taskId = String(frame?.task_id || '').trim();
        if (!taskId) return false;
        if (frame?.ephemeral_decision) {
            ephemeralDecisionTaskIds.add(taskId);
            const taskState = taskUiStates.get(taskId);
            if (taskState?.cleanupTimer) clearTimeout(taskState.cleanupTimer);
            taskUiStates.delete(taskId);
            const record = liveCardRecords.get(taskId);
            if (record) {
                record.root?.remove();
                liveCardRecords.delete(taskId);
            }
            pendingSuggestedNames.delete(taskId);
            if (activeLiveGroupId === taskId) activeLiveGroupId = '';
        }
        return ephemeralDecisionTaskIds.has(taskId);
    }

    function buildMessageKey(role, text, timestamp, opts = {}) {
        if (opts.clientMessageId) return `client|${opts.clientMessageId}`;
        if (role !== 'user' && !opts.isProgress && opts.taskId) {
            return [
                'task',
                role,
                opts.systemType || '',
                opts.source || '',
                opts.taskId,
                text,
            ].join('|');
        }
        if (!timestamp) return '';
        return [
            role,
            opts.isProgress ? '1' : '0',
            opts.systemType || '',
            opts.source || '',
            opts.senderLabel || '',
            opts.senderSessionId || '',
            opts.taskId || '',
            timestamp,
            text,
        ].join('|');
    }

    function reconnectBannerText(reason = '') {
        if (reason === 'sha-change') return '♻️ Restart complete';
        if (reason) return '♻️ Reconnected';
        return '';
    }

    function readPendingReconnectBanner() {
        try {
            const url = new URL(window.location.href);
            return reconnectBannerText(url.searchParams.get('_ouro_reason') || '');
        } catch {
            return '';
        }
    }

    function clearPendingReconnectBanner() {
        try {
            const url = new URL(window.location.href);
            if (!url.searchParams.has('_ouro_reason') && !url.searchParams.has('_ouro_refresh')) return;
            url.searchParams.delete('_ouro_reason');
            url.searchParams.delete('_ouro_refresh');
            window.history.replaceState({}, '', url);
        } catch {}
    }

    function rememberMessageKey(key) {
        if (!key || seenMessageKeys.has(key)) return;
        seenMessageKeys.add(key);
        messageKeyOrder.push(key);
        if (messageKeyOrder.length > 2000) {
            const oldest = messageKeyOrder.shift();
            if (oldest) seenMessageKeys.delete(oldest);
        }
    }

    function formatMsgTime(isoStr) {
        if (!isoStr) return null;
        try {
            const d = new Date(isoStr);
            if (isNaN(d)) return null;
            const now = new Date();
            const pad = n => String(n).padStart(2, '0');
            const hhmm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            const todayStr = now.toDateString();
            const yesterday = new Date(now);
            yesterday.setDate(now.getDate() - 1);
            let short;
            if (d.toDateString() === todayStr) short = hhmm;
            else if (d.toDateString() === yesterday.toDateString()) short = `Yesterday, ${hhmm}`;
            else short = `${months[d.getMonth()]} ${d.getDate()}, ${hhmm}`;
            const full = `${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()} at ${hhmm}`;
            return { short, full };
        } catch {
            return null;
        }
    }

    function stampNodeTimestamp(node, raw, { anchor = false } = {}) {
        if (!node) return false;
        const epoch = rawTimestampEpoch(raw);
        if (!Number.isFinite(epoch)) return false;
        if (anchor && node.dataset.ts) {
            const current = Number(node.dataset.ts);
            const next = Number.isFinite(current) ? Math.min(current, epoch) : epoch;
            node.dataset.ts = String(next);
            return Number.isFinite(current) && next < current;
        } else {
            node.dataset.ts = String(epoch);
        }
        return false;
    }

    function getSenderLabel(role, isProgress = false, systemType = '', opts = {}) {
        if (role === 'user') {
            if (opts.source === 'telegram') return opts.senderLabel || 'Telegram';
            if (opts.senderSessionId && opts.senderSessionId !== chatSessionId) {
                return `WebUI (${opts.senderSessionId.slice(0, 8)})`;
            }
            return opts.senderLabel || 'You';
        }
        if (role === 'system') {
            if (systemType === 'task_summary') return '📋 Task Summary';
            if (systemType === 'skill_review') return '📋 Skill Review';
            return '📋 System';
        }
        if (isProgress) return '💬 Thought';
        return 'Ouroboros';
    }

    function summarizeSkillReviewMessage(text) {
        const raw = String(text || '');
        const lines = raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
        const headline = lines[0] || 'Skill review';
        const hashLine = lines.find((line) => line.startsWith('content_hash=')) || '';
        const reviewersLine = lines.find((line) => line.startsWith('Reviewers:')) || '';
        const findingsLine = lines.find((line) => /^##\s+Findings/.test(line)) || '';
        const meta = [hashLine, reviewersLine.replace(/^Reviewers:\s*/, ''), findingsLine.replace(/^##\s*/, '')]
            .filter(Boolean)
            .map((line) => escapeHtml(line.length > 140 ? `${line.slice(0, 137)}...` : line))
            .join(' · ');
        return {
            headline: escapeHtml(headline.replace(/^#+\s*/, '')),
            meta,
        };
    }

    function renderSkillReviewDisclosure(text) {
        const summary = summarizeSkillReviewMessage(text);
        return `
            <div class="skill-review-disclosure" data-skill-review-disclosure data-expanded="0">
                <button type="button" class="skill-review-summary-button" data-skill-review-toggle aria-expanded="false">
                    <span class="skill-review-summary-main">${summary.headline}</span>
                    <span class="skill-review-summary-side">
                        <span class="skill-review-meta">${summary.meta}</span>
                        <span class="skill-review-toggle-label">Show review</span>
                    </span>
                </button>
                <div class="skill-review-full" data-skill-review-full hidden>${renderMarkdown(text)}</div>
            </div>
        `;
    }

    function setStatus(kind, text) {
        if (!statusBadge) return;
        statusBadge.className = `status-badge ${kind}`;
        statusBadge.textContent = text;
    }

    function syncHeaderControlState(data) {
        headerActions?.querySelectorAll('[data-chat-command]').forEach((button) => {
            const cmd = button.dataset.chatCommand;
            if (cmd === 'evolve') {
                button.classList.toggle('on', !!data?.evolution_enabled);
                if (data?.evolution_state?.detail) button.title = data.evolution_state.detail;
            } else if (cmd === 'bg') {
                button.classList.toggle('on', !!data?.bg_consciousness_enabled);
                if (data?.bg_consciousness_state?.detail) button.title = data.bg_consciousness_state.detail;
            }
        });
        // Evolve/Consciousness now live inside the More menu; surface a small dot
        // on the More summary so an active mode stays visible without opening it.
        const moreSummary = headerActions?.querySelector('.chat-header-more > summary');
        if (moreSummary) {
            const anyActive = !!data?.evolution_enabled || !!data?.bg_consciousness_enabled;
            moreSummary.classList.toggle('has-active', anyActive);
        }
        const ctxBtn = byId('context-mode');
        if (ctxBtn && typeof data?.context_mode === 'string') {
            ctxBtn.dataset.contextMode = data.context_mode === 'low' ? 'low' : 'max';
            ctxBtn.dataset.contextModeAutoLow = data.context_mode_auto_low ? 'true' : 'false';
        }
        const budget = headerBudgetPresentation(data);
        const budgetText = byId('budget-text');
        const budgetFill = byId('budget-bar-fill');
        if (budgetText) budgetText.textContent = budget.label;
        if (budgetFill) budgetFill.style.width = `${budget.fillPct}%`;
    }

    async function refreshHeaderControlState(force = false) {
        if (!force && state.activePage !== 'chat') return;
        try {
            const resp = await apiFetch('/api/state', { cache: 'no-store' });
            if (!resp.ok) {
                syncHeaderControlState({ accounting: { available: false } });
                return;
            }
            syncHeaderControlState(await resp.json());
        } catch {
            syncHeaderControlState({ accounting: { available: false } });
        }
    }

    function persistVisibleHistory() {
        try {
            sessionStorage.setItem(storeKey(CHAT_STORAGE_KEY), JSON.stringify(persistedHistory.slice(-200)));
        } catch {}
    }

    const NEAR_BOTTOM_THRESHOLD_PX = 160;

    function isNearBottom(threshold = NEAR_BOTTOM_THRESHOLD_PX) {
        const remaining = messagesDiv.scrollHeight - messagesDiv.scrollTop - messagesDiv.clientHeight;
        return remaining <= threshold;
    }

    function captureVisibleTimelineAnchor(excludeNode = null) {
        const nodes = Array.from(messagesDiv.children).filter(
            (node) => node !== excludeNode && !node.classList.contains('typing-bubble')
        );
        const messagesTop = messagesDiv.getBoundingClientRect().top;
        const node = nodes.find((item) => item.getBoundingClientRect().bottom > messagesTop) || null;
        if (!node) return null;
        const ts = node.dataset?.ts || '';
        return {
            node,
            taskId: node.classList.contains('chat-live-card') ? (node.dataset?.taskId || '') : '',
            clientMessageId: node.dataset?.clientMessageId || '',
            ts,
            ordinal: ts ? nodes.filter((item) => item.dataset?.ts === ts).indexOf(node) : -1,
            offset: node.getBoundingClientRect().top - messagesTop,
        };
    }

    function restoreVisibleTimelineAnchor(anchor) {
        if (!anchor) return false;
        let node = anchor.node?.isConnected ? anchor.node : null;
        if (!node && anchor.taskId) {
            node = Array.from(messagesDiv.children).find(
                (item) => item.classList.contains('chat-live-card')
                    && item.dataset?.taskId === anchor.taskId
            ) || null;
        }
        if (!node && anchor.clientMessageId) {
            node = Array.from(messagesDiv.children).find(
                (item) => item.dataset?.clientMessageId === anchor.clientMessageId
            ) || null;
        }
        if (!node && anchor.ts) {
            const matches = Array.from(messagesDiv.children).filter(
                (item) => item.dataset?.ts === anchor.ts
            );
            node = matches[anchor.ordinal] || matches[0] || null;
        }
        if (!node) return false;
        const currentOffset = node.getBoundingClientRect().top
            - messagesDiv.getBoundingClientRect().top;
        messagesDiv.scrollTop += currentOffset - anchor.offset;
        return true;
    }

    function insertMessageNode(node, options = {}) {
        if (!node) return;
        const shouldStick = Boolean(options.forceStick) || isNearBottom();
        const isMounted = node.parentNode === messagesDiv;
        if (isMounted && !options.reorderExisting) {
            if (shouldStick) messagesDiv.scrollTop = messagesDiv.scrollHeight;
            updateScrollButton();
            return;
        }
        const reorderAnchor = isMounted && !shouldStick
            ? captureVisibleTimelineAnchor(node)
            : null;
        // Scope to THIS instance's column — a global id lookup would resolve to
        // the first panel's typing node and misplace project-thread messages.
        const typing = messagesDiv.querySelector('.typing-bubble');
        insertTimelineNode(messagesDiv, node, typing, { stickToBottom: shouldStick });
        if (reorderAnchor) restoreVisibleTimelineAnchor(reorderAnchor);
        // A new message arriving while the user is scrolled up reveals the
        // jump-to-newest button instead of silently piling up off-screen.
        updateScrollButton();
    }

    function isBackgroundTaskId(taskId = '') {
        return taskId === 'bg-consciousness';
    }

    function shouldAlwaysShowTaskCard(taskId = '') {
        return isBackgroundTaskId(taskId);
    }

    function isForegroundLiveCard(record) {
        return Boolean(record?.root?.isConnected && !record.finished && !isBackgroundTaskId(record.groupId));
    }

    function createTaskUiState(taskId) {
        if (!taskId) return null;
        const taskState = {
            taskId,
            toolCalls: 0,
            forceCard: false,
            cardVisible: false,
            completed: false,
            completedPhase: '',
            bufferedLiveUpdates: [],
            cleanupTimer: null,
        };
        taskUiStates.set(taskId, taskState);
        return taskState;
    }

    function getTaskUiState(taskId = '', createIfMissing = true) {
        if (!taskId) return null;
        if (taskUiStates.has(taskId)) return taskUiStates.get(taskId);
        return createIfMissing ? createTaskUiState(taskId) : null;
    }

    function scheduleTaskUiCleanup(taskState, delayMs = 120000) {
        if (!taskState) return;
        if (taskState.cleanupTimer) clearTimeout(taskState.cleanupTimer);
        taskState.cleanupTimer = setTimeout(() => {
            taskUiStates.delete(taskState.taskId);
            // Keep the finished card interactive, but mark it retired so routine
            // syncs do not rebuild duplicates. Reload/reconnect clears this set.
            if (!REUSABLE_TASK_IDS.has(taskState.taskId) && taskState.taskId !== '') {
                retiredTaskIds.add(taskState.taskId);
            }
        }, delayMs);
    }

    function bufferLiveUpdate(taskState, summary, ts, dedupeKey = '', rawTs = '') {
        if (!taskState || !summary) return;
        taskState.bufferedLiveUpdates.push({
            summary,
            ts,
            rawTs,
            dedupeKey: dedupeKey || summary.dedupeKey || '',
        });

    }

    function reanchorTaskCard(
        record,
        rawTs,
        { suppressDomInsert = false } = {},
        seen = new Set(),
    ) {
        if (!record || seen.has(record.groupId)) return false;
        seen.add(record.groupId);
        const movedEarlier = stampNodeTimestamp(record.root, rawTs, { anchor: true });
        if (record.isSubagent) {
            const parent = liveCardRecords.get(record.parentGroupId);
            const parentMoved = reanchorTaskCard(
                parent, rawTs, { suppressDomInsert }, seen
            );
            return movedEarlier || parentMoved;
        }
        if (!movedEarlier) return false;
        if (suppressDomInsert || _syncPass1Active) {
            record._anchorOrderDirty = true;
            return true;
        }
        insertMessageNode(record.root, { reorderExisting: true });
        record._anchorOrderDirty = false;
        return true;
    }

    function reanchorVisibleTaskCard(taskState, rawTs, options = {}) {
        if (!taskState?.cardVisible) return false;
        return reanchorTaskCard(liveCardRecords.get(taskState.taskId), rawTs, options);
    }

    function revealBufferedCardIfNeeded(taskState, { suppressDomInsert = false, rawTs = '' } = {}) {
        if (!taskState) return;
        if (taskState.cardVisible) {
            reanchorVisibleTaskCard(taskState, rawTs, { suppressDomInsert });
            return;
        }
        if (!(taskState.forceCard || taskState.toolCalls > 0 || shouldAlwaysShowTaskCard(taskState.taskId))) {
            return;
        }
        taskState.cardVisible = true;
        activeLiveGroupId = taskState.taskId;
        const subagentInfo = subagentChildParents.get(taskState.taskId);
        const record = subagentInfo
            ? getSubagentCardRecord(
                taskState.taskId,
                subagentInfo.parentId,
                subagentInfo.role,
            )
            : getLiveCardRecord(taskState.taskId);
        let anchorMovedEarlier = false;
        if (!record.isSubagent) {
            anchorMovedEarlier = stampNodeTimestamp(record.root, rawTs, { anchor: true });
            for (const update of taskState.bufferedLiveUpdates) {
                anchorMovedEarlier = stampNodeTimestamp(
                    record.root, update.rawTs, { anchor: true }
                ) || anchorMovedEarlier;
            }
        }
        ensureLiveCardVisible(record, { suppressDomInsert, reorderExisting: anchorMovedEarlier });
        const bufferedUpdates = [...taskState.bufferedLiveUpdates];
        taskState.bufferedLiveUpdates = [];
        for (const update of bufferedUpdates) {
            applyLiveCardState(update.summary, taskState.taskId, update.ts, update.dedupeKey, {
                suppressDomInsert,
                rawTs: update.rawTs,
            });
        }
        if (taskState.completed) {
            finishLiveCard(taskState.taskId, taskState.completedPhase || 'done');
        }
    }

    function markTaskToolCall(taskId, count = 1, minimumOnly = false, rawTs = '') {
        const taskState = getTaskUiState(taskId, true);
        if (!taskState) return null;
        const safeCount = Math.max(0, Number(count) || 0);
        if (minimumOnly) {
            taskState.toolCalls = Math.max(taskState.toolCalls, safeCount);
        } else {
            taskState.toolCalls += safeCount;
        }
        revealBufferedCardIfNeeded(taskState, { rawTs });
        return taskState;
    }

    function forceTaskCard(taskId, rawTs = '') {
        const taskState = getTaskUiState(taskId, true);
        if (!taskState) return null;
        taskState.forceCard = true;
        revealBufferedCardIfNeeded(taskState, { rawTs });
        return taskState;
    }

    function markAssistantReply(taskId = '') {
        const resolvedTaskId = taskId || '';
        if (!resolvedTaskId) return;
        const taskState = getTaskUiState(resolvedTaskId, false);
        if (!taskState) return;
        taskState.completed = true;
        taskState.completedPhase = taskState.completedPhase || 'done';
        if (!taskState.cardVisible) {
            scheduleTaskUiCleanup(taskState, 30000);
            return;
        }
        scheduleTaskUiCleanup(taskState);
    }

    function markTaskComplete(taskId = '', phase = '') {
        const taskState = getTaskUiState(taskId, false);
        if (!taskState) return;
        taskState.completed = true;
        if (phase) taskState.completedPhase = phase;
    }

    // v6.82 (P5): task ids whose progress carried the supervisor's host-attested
    // `cancelable` marker (queue tasks the cancel endpoint can genuinely reach).
    // Learned from live WS frames and history replay alike, possibly before the
    // card exists, so it lives beside the card records rather than on them.
    const cancelableTaskIds = new Set();

    function queueTaskLiveUpdate(summary, taskId, ts, dedupeKey = '', rawTs = '') {
        const resolvedTaskId = taskId || activeLiveGroupId || '';
        if (!resolvedTaskId) return;
        const taskState = getTaskUiState(resolvedTaskId, true);
        if (!taskState) return;
        // Even an already-completed card must absorb an earlier historical/nested
        // event into its chronology anchor before lifecycle policy ignores the
        // event's content.
        reanchorVisibleTaskCard(taskState, rawTs);
        if (taskState.completed && !isTerminalTaskPhase(summary.phase || '', summary.terminal)) {
            // A non-terminal event on a reusable id starts a fresh visible cycle.
            if (REUSABLE_TASK_IDS.has(resolvedTaskId)) {
                if (taskState.cleanupTimer) clearTimeout(taskState.cleanupTimer);
                taskState.completed = false;
                taskState.completedPhase = '';
                taskState.cardVisible = false;
                taskState.bufferedLiveUpdates = [];
                taskState.toolCalls = 0;
                taskState.forceCard = false;
                const oldRec = liveCardRecords.get(resolvedTaskId);
                if (oldRec) {
                    oldRec.root?.remove();
                    liveCardRecords.delete(resolvedTaskId);
                }
                retiredTaskIds.delete(resolvedTaskId);
            } else {
                return;
            }
        }
        if (summary.phase === 'error' || summary.phase === 'timeout' || (summary.terminal && summary.phase === 'warn')) {
            taskState.forceCard = true;
        }
        if (!taskState.cardVisible) {
            bufferLiveUpdate(taskState, summary, ts, dedupeKey, rawTs);
            revealBufferedCardIfNeeded(taskState, { rawTs });
            return;
        }
        applyLiveCardState(summary, resolvedTaskId, ts, dedupeKey, { rawTs });
    }

    async function turnTaskIntoProject(record) {
        if (!record || record.root?.dataset?.projectCreating === '1' || record.root?.dataset?.projectCreated === '1') return;
        const taskId = String(record.groupId || '').trim();
        const projectId = projectIdFromTask(taskId);
        record.root.dataset.projectCreating = '1';
        const actions = record.turnProjectBtn?.parentElement || record.root.querySelector('.chat-live-actions');
        if (actions) {
            actions.innerHTML = '<button type="button" class="chat-live-project-btn" disabled>Creating project…</button>';
            record.cancelRunBtn = null;
        }
        try {
            // One-click convert (owner P1): no name prompt, no extra LLM call.
            // The SERVER derives the project name (gateway/projects.py
            // _derive_project_name: title -> objective -> queue snapshot). We also
            // hand it the owner's original request as a fallback hint so a still
            // in-progress DIRECT chat task — which has no server-side title/objective
            // yet — is named from what the owner asked, not "New project".
            const payload = await apiClient.projectFromTask(taskId, projectId, '', record.objectiveHint || '');
            const project = payload.project || { id: projectId, name: projectId };
            showToast(`Project created: ${project.name || project.id}`, 'ok');
            window.dispatchEvent(new CustomEvent('ouro:project-created', { detail: { project } }));
            markCardConverted(record, project);
        } catch (exc) {
            showToast(`Project creation failed: ${exc.message || exc}`, 'error');
            delete record.root.dataset.projectCreating;
            if (actions) {
                actions.innerHTML = '<button type="button" class="chat-live-project-btn" data-turn-into-project>Turn into project</button>';
                record.turnProjectBtn = actions.querySelector('[data-turn-into-project]');
                // Re-wire the click handler — innerHTML replaced the original node,
                // so without this the restored button would be dead after a
                // transient failure (T5).
                record.turnProjectBtn?.addEventListener('click', (event) => {
                    event.stopPropagation();
                    turnTaskIntoProject(record);
                });
                // P5: innerHTML also dropped a rendered "Cancel run" — restore it.
                record.cancelRunBtn = null;
                syncCancelRunButton(record);
            }
        }
    }

    // v6.82 (P5): "Cancel run" on live pooled ROOT cards. Forced cancel of the
    // selected task AND its live subtree (explicit cascade — the endpoint's
    // default stays single-task for headless callers). Gated on the supervisor's
    // host-attested `cancelable` marker so a direct-chat turn (which mints a
    // card of the same shape but has no queue entry) never shows a dead button.
    function ensureLiveActionsEl(record) {
        if (!record?.root
            || record.root.dataset.projectCreated === '1'
            || record.root.dataset.projectCreating === '1') return null;
        let actions = record.root.querySelector('.chat-live-actions');
        if (!actions) {
            actions = document.createElement('div');
            actions.className = 'chat-live-actions';
            const timeline = record.timelineEl && record.timelineEl.parentElement === record.root
                ? record.timelineEl
                : null;
            record.root.insertBefore(actions, timeline);
        }
        return actions;
    }

    function syncCancelRunButton(record) {
        if (!record?.root) return;
        const eligible = cancelRunEligibility({
            groupId: record.groupId,
            isSubagent: record.isSubagent,
            finished: record.finished,
            cancelable: cancelableTaskIds.has(record.groupId),
            converted: record.root.dataset.projectCreated === '1',
        });
        const existing = record.root.querySelector('[data-cancel-run]');
        if (!eligible) {
            existing?.remove();
            record.cancelRunBtn = null;
            return;
        }
        if (existing) {
            record.cancelRunBtn = existing;
            return;
        }
        const actions = ensureLiveActionsEl(record);
        if (!actions) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-xs btn-danger';
        btn.dataset.cancelRun = '1';
        btn.textContent = 'Cancel run';
        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            cancelRunFromCard(record);
        });
        actions.appendChild(btn);
        record.cancelRunBtn = btn;
    }

    async function cancelRunFromCard(record) {
        const taskId = String(record?.groupId || '').trim();
        if (!taskId || record.finished) return;
        const confirmed = await openConfirmDialog({
            title: 'Cancel this run?',
            body: 'Cancel this run and all its subagents? Unfinished results and artifacts may be lost.',
            confirmLabel: 'Cancel run',
            cancelLabel: 'Keep running',
            danger: true,
        });
        if (!confirmed) return;
        // Completion-wins race: the task may have finished while the dialog was
        // open — its task_done already resolved the card, nothing to cancel.
        if (record.finished) return;
        const btn = record.cancelRunBtn;
        if (btn) btn.disabled = true;
        try {
            // Answered only after the teardown finished, so a resolved promise
            // means the run is really down; a refusal throws and is toasted below.
            await cancelTask(taskId, { cascade: true });
            // Backend publication is fail-soft past the durable boundary, so a 200
            // can arrive with the task_done event lost. Reconcile from the durable
            // record through the same terminal seam replay uses — idempotent with
            // a later event, so double resolution is harmless.
            try {
                const stored = await apiFetch(`/api/tasks/${encodeURIComponent(taskId)}`).then(
                    (resp) => (resp && typeof resp.json === 'function' && resp.ok !== false) ? resp.json() : null,
                );
                const status = String(stored?.status || '');
                if (!record.finished && ['completed', 'failed', 'cancelled', 'cancel_requested', 'rejected_duplicate'].includes(status)) {
                    finishLiveCard(taskId, taskTerminalPhase(stored));
                }
            } catch {
                // The card still resolves on its own frame if one arrives.
            }
            // The card resolves via the existing task_done{status:"cancelled"}
            // frames; keep the button disabled until that (or removal) happens.
        } catch (exc) {
            // 404 = nothing live anymore (natural completion beat the cancel):
            // graceful no-op, the card resolves on its own terminal frame.
            if (exc?.status === 404 || record.finished) {
                // Completion-wins race: the run finished while the request was in
                // flight, so there is nothing to cancel. RESYNC rather than leave a
                // dead disabled button — the card resolves on its own terminal
                // frame, and until then the action is simply no longer offered.
                // `cancelableTaskIds` is the eligibility AUTHORITY — clearing the
                // record flag alone left the button mounted and merely re-enabled.
                cancelableTaskIds.delete(taskId);
                record.cancelable = false;
                syncCancelRunButton(record);
                // REAL resync, not just button removal: 404 says the task is no
                // longer live, but if its terminal frame was lost this card would
                // sit "Working" forever. Ask the durable record and resolve the
                // card through the same terminal seam replay uses.
                try {
                    const stored = await apiFetch(`/api/tasks/${encodeURIComponent(taskId)}`).then(
                        (resp) => (resp && typeof resp.json === 'function' && resp.ok !== false) ? resp.json() : null,
                    );
                    const status = String(stored?.status || '');
                    if (record.finished || !status) return;
                    if (['completed', 'failed', 'cancelled', 'cancel_requested', 'rejected_duplicate'].includes(status)) {
                        finishLiveCard(taskId, taskTerminalPhase(stored));
                    }
                } catch {
                    // The card still resolves on its own frame if one arrives;
                    // nothing worse than the pre-resync behavior.
                }
                return;
            }
            showToast(`Cancel failed: ${exc?.message || exc}`, 'error');
            if (btn) btn.disabled = false;
        }
    }

    function markTaskCancelable(taskId = '') {
        const id = String(taskId || '').trim();
        if (!id || cancelableTaskIds.has(id)) return;
        cancelableTaskIds.add(id);
        const record = liveCardRecords.get(id);
        if (record) syncCancelRunButton(record);
    }

    // One-way conversion (P3): the WHOLE card becomes a calm "project identity"
    // chip. The live task is now owned by the project panel (it's bound there),
    // so the main chat is freed — the card stops being a busy red task and
    // recolors to the project fuchsia. Plain wording (no "ack"); click opens the panel.
    function markCardConverted(record, project) {
        delete record.root.dataset.projectCreating;
        record.root.dataset.projectCreated = '1';
        record.root.dataset.projectId = project.id || '';
        const name = String(project.name || project.id || 'Project').trim();
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'chat-live-project-card-btn';
        const icon = document.createElement('span');
        icon.className = 'chat-live-project-icon';
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = '📁';
        const nameEl = document.createElement('span');
        nameEl.className = 'chat-live-project-name';
        nameEl.textContent = name;  // textContent — no HTML injection from a project name
        const status = document.createElement('span');
        status.className = 'chat-live-project-status';
        status.textContent = 'running in background ↗';
        chip.append(icon, nameEl, status);
        chip.addEventListener('click', () => {
            window.dispatchEvent(new CustomEvent('ouro:open-project', { detail: { project } }));
        });
        // Atomic detach-and-reparent (C4.5): replaceChildren swaps the whole live
        // timeline (subagent cards, working bubble) for the chip in one paint — no
        // innerHTML reparse, no torn intermediate state. The task tree re-homes to
        // the project thread on the backend (lineage classification + the owner
        // request mirrored into the project chat), so the main chat keeps only this
        // calm pointer; the project panel re-renders the full tree from history.
        record.root.replaceChildren(chip);
        record.turnProjectBtn = null;
        record.cancelRunBtn = null;
        // The task now lives in the project panel, so this card must stop counting
        // as a foreground ACTIVE task in the main chat — otherwise isForegroundLiveCard
        // keeps suppressing the typing indicator / status-badge clear until the
        // background task ends. Mark it finished; the converted-card guards in
        // applyLiveCardState/finishLiveCard then ignore any late terminal frame.
        // (The detached element refs are LEFT intact — nulling them made other
        // terminal paths like finishLiveCard throw on a post-conversion frame.)
        record.finished = true;
        // Recolor on the next frame so the 250ms fuchsia fade actually animates
        // (the class can't be added in the same paint as the content swap).
        requestAnimationFrame(() => record.root.classList.add('is-project'));
        signalChatFreed();  // subtle "this chat is free again" composer cue
    }

    // A brief composer brighten when a task leaves the main chat for a project —
    // a calm "you're free to start something else" signal (P3). Self-clearing.
    let _chatFreedTimer = null;
    function signalChatFreed() {
        const row = page.querySelector('.chat-text-row');
        if (!row) return;
        row.classList.add('chat-freed');
        if (_chatFreedTimer) clearTimeout(_chatFreedTimer);
        _chatFreedTimer = setTimeout(() => row.classList.remove('chat-freed'), 900);
    }

    function createLiveCardRecord(groupId = '', options = {}) {
        const normalizedGroupId = groupId || `task-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const timelineId = `chat-live-timeline-${normalizedGroupId.replace(/[^A-Za-z0-9_-]/g, '-')}`;
        const root = document.createElement('div');
        root.className = 'chat-live-card';
        root.dataset.taskId = normalizedGroupId;
        if (options.isSubagent) {
            root.classList.add('subagent');
            root.dataset.subagent = '1';
            root.dataset.parentTaskId = String(options.parentGroupId || '');
            root.dataset.subagentRole = String(options.role || '');
        }
        root.dataset.finished = '0';
        root.dataset.expanded = ((options.isSubagent && nestedSubagentsExpanded) || stickyExpandedSlots.has(normalizedGroupId)) ? '1' : '0';
        // No "Turn into project" for: subagent cards, non-main panels, or a task that
        // is ALREADY bound to a project (a project-chat follow-up) — see task_bindings
        // from /api/state, surfaced on window.__ouroTaskBindings (P2).
        const alreadyBound = !!(window.__ouroTaskBindings || {})[normalizedGroupId];
        const projectActionHtml = (
            isMain
            && !options.isSubagent
            && !alreadyBound
            && !ephemeralDecisionTaskIds.has(normalizedGroupId)
        )
            ? `<div class="chat-live-actions"><button type="button" class="chat-live-project-btn" data-turn-into-project>Turn into project</button></div>`
            : '';
        root.innerHTML = `
            <button type="button" class="chat-live-summary-button" data-live-summary-button aria-expanded="false" aria-controls="${escapeHtmlAttr(timelineId)}">
                <div class="chat-live-summary">
                    <div class="chat-live-summary-main">
                        <span class="chat-live-phase working" data-live-phase>Working</span>
                        <div class="chat-live-typing" data-live-typing aria-hidden="true">
                            <span></span><span></span><span></span>
                        </div>
                        <span class="chat-live-title" data-live-title>Waiting for work</span>
                    </div>
                    <div class="chat-live-summary-side">
                        <span class="chat-live-count" data-live-count hidden>2 notes</span>
                        <span class="chat-live-toggle" data-live-toggle>Show details</span>
                        <svg class="chat-live-chevron" width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true">
                            <path d="M5 7.5 10 12.5 15 7.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                    </div>
                </div>
                <div class="chat-live-activity" data-live-activity></div>
                <div class="chat-live-meta" data-live-meta></div>
            </button>
            ${projectActionHtml}
            <div class="chat-live-timeline" data-live-timeline id="${escapeHtmlAttr(timelineId)}"></div>
        `;
        const record = {
            groupId: normalizedGroupId,
            root,
            summaryButtonEl: root.querySelector('[data-live-summary-button]'),
            phaseEl: root.querySelector('[data-live-phase]'),
            inlineTypingEl: root.querySelector('[data-live-typing]'),
            titleEl: root.querySelector('[data-live-title]'),
            activityEl: root.querySelector('[data-live-activity]'),
            countEl: root.querySelector('[data-live-count]'),
            metaEl: root.querySelector('[data-live-meta]'),
            toggleEl: root.querySelector('[data-live-toggle]'),
            turnProjectBtn: root.querySelector('[data-turn-into-project]'),
            // P5: "Cancel run" button element (rendered lazily by syncCancelRunButton
            // once the host-attested cancelable marker is known for this task).
            cancelRunBtn: null,
            timelineEl: root.querySelector('[data-live-timeline]'),
            updates: 0,
            finished: false,
            items: [],
            lastHumanHeadline: '',
            expandedLineKeys: new Set(),
            isSubagent: Boolean(options.isSubagent),
            parentGroupId: String(options.parentGroupId || ''),
            subagentRole: String(options.role || ''),
            subagentsEl: null,
            _anchorOrderDirty: false,
            // Hidden-page layout sync is deferred until page/visibility returns.
            _needsLayoutSync: false,
            // The owner's request that spawned this card (main, non-subagent only),
            // used to name a project on "turn into project" when the server has no
            // title/objective yet (P1, direct-chat conversion). One-shot handoff.
            objectiveHint: (isMain && !options.isSubagent) ? _pendingCardObjective : '',
            // Cluster B: the proactively-coined LLM project name; when set it becomes
            // the card title (the activity headline keeps rendering in the lines below).
            suggestedName: '',
            // P1 (v6.82): last raw activity candidate (remembered even while the
            // collapsed line is suppressed on unnamed root cards) + sticky cost.
            collapsedActivity: '',
            costMeta: null,
        };
        if (isMain && !options.isSubagent) _pendingCardObjective = '';
        record.summaryButtonEl?.addEventListener('click', () => {
            const nowExpanded = record.root.dataset.expanded !== '1';
            setLiveCardExpanded(record, nowExpanded);
            if (REUSABLE_TASK_IDS.has(record.groupId)) {
                if (nowExpanded) stickyExpandedSlots.add(record.groupId);
                else stickyExpandedSlots.delete(record.groupId);
            }
        });
        record.turnProjectBtn?.addEventListener('click', (event) => {
            event.stopPropagation();
            turnTaskIntoProject(record);
        });
        record.timelineEl?.addEventListener('click', (event) => {
            const button = event.target.closest('[data-live-line-toggle]');
            // Row-surface disclosure (v6.71.0): any click on the line's
            // NON-interactive surface toggles it (guards live in the pure
            // helper: nested interactive elements, active text selection).
            const lineKey = button
                ? (button.dataset.liveLineToggle || '')
                : liveLineRowToggleKey(event.target, window.getSelection?.());
            if (!lineKey) return;
            const nowExpanded = !record.expandedLineKeys.has(lineKey);
            if (nowExpanded) record.expandedLineKeys.add(lineKey);
            else record.expandedLineKeys.delete(lineKey);
            renderLiveCardTimeline(record);
            syncLiveCardLayout(record);
            // Keyboard/AT continuity: focus the rebuilt line's REAL toggle button.
            record.timelineEl
                ?.querySelector(`[data-live-line-toggle="${(window.CSS && CSS.escape) ? CSS.escape(lineKey) : lineKey}"]`)
                ?.focus?.({ preventScroll: true });
            // P3: on expand, lazily fetch the genuinely-full output for a server-truncated
            // line (the WS preview was capped at 4000); cached on the item so a re-render
            // keeps it. Best-effort — the capped preview stays on failure.
            if (nowExpanded) {
                const item = record.items.find((it) => it.lineKey === lineKey);
                if (item && item.truncated && item.fullRef && !item.fetchedFull && !item._fetchingFull) {
                    fetchFullLineOutput(item, record);
                }
            }
        });
        liveCardRecords.set(normalizedGroupId, record);
        // Cluster B: apply a name that arrived (task_named) before this card existed.
        const _pendingName = pendingSuggestedNames.get(normalizedGroupId);
        if (_pendingName && !record.isSubagent) {
            pendingSuggestedNames.delete(normalizedGroupId);
            record.suggestedName = _pendingName;
            if (record.titleEl) record.titleEl.textContent = _pendingName;
        }
        resetLiveCardRecord(record);
        // P5: the cancelable marker may have arrived (scheduled progress frame /
        // history replay) before this card was minted.
        syncCancelRunButton(record);
        return record;
    }

    function getLiveCardRecord(groupId = '') {
        const normalizedGroupId = groupId || activeLiveGroupId || 'chat';
        return liveCardRecords.get(normalizedGroupId) || createLiveCardRecord(normalizedGroupId);
    }

    // Cluster B: apply the proactively-coined project name to a main card already on
    // screen (live `task_named` event or history replay). A main card's groupId IS its
    // task_id, so the lookup is direct. No-op until the card exists / without a name.
    function applySuggestedName(taskId, name) {
        const tid = String(taskId || '').trim();
        const nm = String(name || '').trim();
        if (!tid || !nm) return;
        const record = liveCardRecords.get(tid);
        if (!record) {
            // Card not created yet (the namer raced ahead of the first progress event).
            // Buffer so createLiveCardRecord applies it when the card appears.
            pendingSuggestedNames.set(tid, nm);
            return;
        }
        if (record.isSubagent) return;
        record.suggestedName = nm;
        if (record.titleEl) record.titleEl.textContent = nm;
        // P1 (v6.82): the collapsed activity line was suppressed while the card
        // was unnamed; populate it now from the remembered candidate so the live
        // task_named direct-DOM path does not depend on the next
        // applyLiveCardState frame. (The pendingSuggestedNames drain path in
        // createLiveCardRecord relies on the creating frame that immediately
        // follows record creation — a fresh record has no remembered activity
        // yet, so there is nothing for the drain itself to populate.)
        renderCollapsedActivity(record, projectCollapsedActivity({
            suggestedName: nm,
            previous: record.collapsedActivity,
        }));
    }

    // ONE renderer for the collapsed activity line: text plus the complete-text
    // reference. Two call sites (frame updates and the late task_named path) drifted
    // apart before, so a late-named card showed clipped text with no way back to the
    // full narration; a suppressed line must not leak through a tooltip either.
    function renderCollapsedActivity(record, text) {
        if (!record?.activityEl) return;
        record.activityEl.textContent = text;
        const full = String(record.collapsedActivity || '');
        if (text && full && full !== text) record.activityEl.title = full;
        else record.activityEl.removeAttribute('title');
    }

    function ensureSubagentContainer(parentId = '') {
        if (!parentId) return null;
        const parentRecord = getLiveCardRecord(parentId);
        let container = parentRecord.subagentsEl;
        if (!container) {
            container = document.createElement('div');
            parentRecord.subagentsEl = container;
        }
        container.className = 'chat-subagents';
        container.dataset.subagentsFor = parentId;
        if (container.parentNode !== parentRecord.root || container.previousElementSibling !== parentRecord.timelineEl) {
            parentRecord.timelineEl?.insertAdjacentElement('afterend', container);
        }
        return container;
    }

    function getSubagentCardRecord(childId = '', parentId = '', role = '') {
        if (!childId || !parentId) return null;
        const existing = liveCardRecords.get(childId);
        const wasSubagent = existing?.isSubagent === true || existing?.root?.classList.contains('subagent');
        const record = existing || createLiveCardRecord(childId, {
            isSubagent: true,
            parentGroupId: parentId,
            role,
        });
        record.isSubagent = true;
        record.parentGroupId = parentId;
        record.subagentRole = role || record.subagentRole || '';
        record.root.classList.add('subagent');
        record.root.dataset.subagent = '1';
        record.root.dataset.parentTaskId = parentId;
        record.root.dataset.subagentRole = record.subagentRole;
        const container = ensureSubagentContainer(parentId);
        if (container && record.root.parentNode !== container) {
            container.appendChild(record.root);
        }
        const parentRecord = liveCardRecords.get(parentId);
        if (parentRecord) updateLiveCardCount(parentRecord);
        if (!wasSubagent) setLiveCardExpanded(record, nestedSubagentsExpanded);
        return record;
    }

    function setLiveCardTypingVisible(record, visible) {
        if (!record?.inlineTypingEl) return;
        record.inlineTypingEl.style.display = visible ? '' : 'none';
    }

    function resetLiveCardRecord(record) {
        record.updates = 0;
        record.finished = false;
        record.items = [];
        record.lastHumanHeadline = '';
        record.expandedLineKeys.clear();
        record._anchorOrderDirty = false;
        clearStickyCardState(record);
        record.titleEl.textContent = 'Working...';
        record.phaseEl.dataset.phase = 'working';
        record.phaseEl.textContent = 'Working';
        record.phaseEl.className = 'chat-live-phase working';
        record.countEl.hidden = true;
        record.countEl.textContent = '0 notes';
        record.metaEl.innerHTML = '';
        record.timelineEl.innerHTML = '';
        record.root.dataset.finished = '0';
        setLiveCardTypingVisible(record, true);
        setLiveCardExpanded(record, (record.isSubagent && nestedSubagentsExpanded) || stickyExpandedSlots.has(record.groupId));
    }

    function ensureLiveCardVisible(
        record,
        { suppressDomInsert = false, reorderExisting = false } = {},
    ) {
        if (record?.isSubagent && record.parentGroupId) {
            if (!suppressDomInsert && !_syncPass1Active) {
                const parentRecord = getLiveCardRecord(record.parentGroupId);
                if (parentRecord.isSubagent && parentRecord.parentGroupId) {
                    ensureLiveCardVisible(parentRecord);
                } else {
                    insertMessageNode(parentRecord.root);
                }
                const container = ensureSubagentContainer(record.parentGroupId);
                if (container && record.root.parentNode !== container) {
                    container.appendChild(record.root);
                }
                updateLiveCardCount(parentRecord);
            }
            return;
        }
        if (!record.isSubagent && !suppressDomInsert && !_syncPass1Active) {
            insertMessageNode(record.root, { reorderExisting });
        }
    }

    function formatLiveCardPhaseLabel(phase) {
        if (phase === 'thinking') return 'Thinking';
        if (phase === 'working') return 'Working';
        if (phase === 'done') return 'Done';
        if (phase === 'cancelled') return 'Cancelled';
        if (phase === 'warn') return 'Notice';
        if (phase === 'error' || phase === 'timeout' || phase === 'lifecycle_error') return 'Issue';
        if (!phase) return 'Working';
        return phase.charAt(0).toUpperCase() + phase.slice(1);
    }

    function setLiveCardExpanded(record, expanded) {
        if (!record?.root) return;
        record.root.dataset.expanded = expanded ? '1' : '0';
        syncLiveCardToggle(record);
        if (record.root.isConnected) {
            requestAnimationFrame(() => syncLiveCardLayout(record));
        }
    }

    function isLiveLineExpandable(item) {
        return Boolean(
            (item.fullHeadline && item.fullHeadline !== item.headline)
            || (item.fullBody && item.fullBody !== item.body)
            // P3: even when the preview equals the capped body, a server-truncated line
            // with a fetch ref has MORE to show (the genuinely-full output on demand).
            || (item.truncated && item.fullRef)
        );
    }

    function syncLiveCardToggle(record) {
        if (!record?.toggleEl) return;
        const expanded = record.root.dataset.expanded === '1';
        record.toggleEl.textContent = expanded ? 'Hide details' : 'Show details';
        record.summaryButtonEl?.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    }

    function directSubagentCount(record) {
        return record?.subagentsEl?.querySelectorAll(':scope > .chat-live-card.subagent').length || 0;
    }

    function updateLiveCardCount(record) {
        if (!record?.countEl) return;
        const bits = [];
        if (record.items.length >= 2) bits.push(`${record.items.length} notes`);
        const children = directSubagentCount(record);
        if (children) bits.push(`${children} ${children === 1 ? 'child' : 'children'}`);
        record.countEl.hidden = bits.length === 0;
        record.countEl.textContent = bits.join(' · ');
    }

    function syncLiveCardLayout(record) {
        if (!record?.root) return;
        // Hidden SPA/browser tabs report zero geometry; defer to avoid collapsed
        // cards. Generalized to panel instances: any visible host counts.
        const activePage = record.root.closest('.page.active');
        const panelHost = record.root.closest('.chat-instance-panel');
        // A panel counts as visible only when it is actually shown (not a
        // hidden/display:none secondary instance) — zero geometry otherwise.
        const visibleHost = activePage || (panelHost && panelHost.offsetParent !== null);
        if (!visibleHost || document.hidden) {
            record._needsLayoutSync = true;
            return;
        }
        record._needsLayoutSync = false;
        if (record.isSubagent && record.parentGroupId) {
            const parentRecord = liveCardRecords.get(record.parentGroupId);
            if (parentRecord?.root?.isConnected) {
                requestAnimationFrame(() => syncLiveCardLayout(parentRecord));
            }
        }
    }

    // Re-sync cards after SPA return or browser tab visibility restore, then put
    // the thread back where the user left it (P7) instead of at the very top.
    window.addEventListener('ouro:page-shown', (event) => {
        if (event?.detail?.page !== 'chat') return;
        for (const record of liveCardRecords.values()) {
            if (record?.root?.isConnected) syncLiveCardLayout(record);
        }
        restoreScrollPosition();  // no-op for hidden panel instances
    });
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) return;
        if (state.activePage !== 'chat') return;
        for (const record of liveCardRecords.values()) {
            if (record?.root?.isConnected && record._needsLayoutSync) syncLiveCardLayout(record);
        }
    });

    function buildTimelineItemHtml(item, record) {
        const expandable = isLiveLineExpandable(item);
        const expanded = expandable && record.expandedLineKeys.has(item.lineKey);
        const displayHeadline = expanded && item.fullHeadline ? item.fullHeadline : item.headline;
        // P3: when expanded, prefer the genuinely-full fetched output, then the capped
        // fullBody, then the preview body. A server-truncated line shows the fetched full
        // text in a bounded-scroll box so a huge research output never grows the chat.
        const displayBody = expanded ? (item.fetchedFull || item.fullBody || item.body) : item.body;
        const showingFetched = expanded && Boolean(item.fetchedFull);
        const loadingFull = expanded && Boolean(item.truncated && item.fullRef && !item.fetchedFull);
        const isProgressLine = item.phase === 'working' || item.phase === 'thinking';
        const bodyId = `chat-live-line-body-${String(record.groupId || 'task').replace(/[^A-Za-z0-9_-]/g, '-')}-${String(item.lineKey || '').replace(/[^A-Za-z0-9_-]/g, '-')}`;
        const headContent = `
            <span class="chat-live-line-title">${isProgressLine ? renderMarkdown(displayHeadline) : escapeHtml(displayHeadline)}</span>
            <span class="chat-live-line-repeat" ${item.count > 1 ? '' : 'hidden'}>${item.count > 1 ? `${item.count}x` : ''}</span>
            ${item.ts ? `<span class="chat-live-line-time">${escapeHtml(item.ts)}</span>` : ''}
        `;
        const headHtml = expandable
            ? `
                <button
                    type="button"
                    class="chat-live-line-toggle"
                    data-live-line-toggle="${escapeHtmlAttr(item.lineKey)}"
                    aria-expanded="${expanded ? 'true' : 'false'}"
                    ${displayBody ? `aria-controls="${escapeHtmlAttr(bodyId)}"` : ''}
                >
                    <span class="chat-live-line-head">${headContent}</span>
                    <span class="chat-live-line-expand-label">${expanded ? 'Collapse' : ((item.truncated && item.fullRef) ? 'Show full' : 'Expand')}</span>
                </button>
            `
            : `<div class="chat-live-line-head">${headContent}</div>`;
        return `
            <div
                class="chat-live-line ${item.phase || 'working'}${expandable ? ' expandable' : ''}"
                data-live-line-key="${escapeHtmlAttr(item.lineKey || '')}"
                data-expanded="${expanded ? '1' : '0'}"
            >
                ${headHtml}
                ${displayBody ? `<div class="chat-live-line-body${showingFetched ? ' chat-live-line-body-full' : ''}" id="${escapeHtmlAttr(bodyId)}">${renderMarkdown(displayBody)}${loadingFull ? '<div class="chat-live-line-loading">Loading full output…</div>' : ''}</div>` : ''}
            </div>
        `;
    }

    function isTimelinePinnedToBottom(record) {
        const el = record?.timelineEl;
        if (!el) return true;
        return el.scrollHeight - el.scrollTop - el.clientHeight <= 24;
    }

    // Full rebuild for initial render and expand/collapse toggles.
    function renderLiveCardTimeline(record) {
        const el = record.timelineEl;
        const pinned = isTimelinePinnedToBottom(record);
        const prevTop = el.scrollTop;
        el.innerHTML = record.items.map((item) => buildTimelineItemHtml(item, record)).join('');
        el.scrollTop = pinned ? el.scrollHeight : prevTop;
    }

    // P3: fetch the genuinely-full output for a server-truncated timeline line (the WS
    // preview was capped at 4000 chars), cache it on the item, then re-render if the line
    // is still expanded. The full text is fetched on demand (not pushed over the socket)
    // and shown in a bounded-scroll box. Best-effort — the capped preview stays on failure.
    async function fetchFullLineOutput(item, record) {
        item._fetchingFull = true;
        try {
            const resp = await apiFetch(`/api/tasks/${encodeURIComponent(item.fullRef)}`, { cache: 'no-store' });
            const data = resp && typeof resp.json === 'function' ? await resp.json() : resp;
            // Compose ALL available full fields — a subagent line can carry both a result AND a
            // (separately truncated) trace_summary, so `result || trace_summary` would hide the
            // full trace. Label each section when both are present.
            const result = String((data && data.result) || '').trim();
            const trace = String((data && data.trace_summary) || '').trim();
            let full = '';
            if (result && trace) full = `[RESULT]\n${result}\n\n[TRACE]\n${trace}`;
            else full = result || trace;
            if (full) item.fetchedFull = full;
        } catch {
            // best-effort: leave the capped preview on failure
        } finally {
            item._fetchingFull = false;
            if (record.expandedLineKeys.has(item.lineKey)) {
                const hadFocus = Boolean(
                    document.activeElement?.closest?.(`[data-live-line-key="${(window.CSS && CSS.escape) ? CSS.escape(item.lineKey) : item.lineKey}"]`),
                );
                renderLiveCardTimeline(record);
                syncLiveCardLayout(record);
                if (hadFocus) {
                    record.timelineEl
                        ?.querySelector(`[data-live-line-toggle="${(window.CSS && CSS.escape) ? CSS.escape(item.lineKey) : item.lineKey}"]`)
                        ?.focus?.({ preventScroll: true });
                }
            }
        }
    }

    // Append without disturbing existing DOM nodes.
    function appendTimelineItem(item, record) {
        const pinned = isTimelinePinnedToBottom(record);
        const wrapper = document.createElement('div');
        wrapper.innerHTML = buildTimelineItemHtml(item, record).trim();
        const node = wrapper.firstElementChild;
        if (node) {
            record.timelineEl.appendChild(node);
            if (record.root.dataset.expanded === '1' && pinned) {
                record.timelineEl.scrollTop = record.timelineEl.scrollHeight;
            }
        }
    }

    // Patch the last DOM node for dedup/count bumps.
    function patchLastTimelineItem(item, record) {
        const lastEl = record.timelineEl.lastElementChild;
        if (!lastEl) return renderLiveCardTimeline(record);
        const wrapper = document.createElement('div');
        wrapper.innerHTML = buildTimelineItemHtml(item, record).trim();
        const newNode = wrapper.firstElementChild;
        if (newNode) record.timelineEl.replaceChild(newNode, lastEl);
    }

    // Patch a specific timeline node in place (evolving subagent dashboard rows).
    function patchTimelineItemAt(item, record) {
        const key = String(item.lineKey || '').replace(/[^A-Za-z0-9_-]/g, '');
        const el = key ? record.timelineEl.querySelector(`[data-live-line-key="${key}"]`) : null;
        if (!el) return renderLiveCardTimeline(record);
        const wrapper = document.createElement('div');
        wrapper.innerHTML = buildTimelineItemHtml(item, record).trim();
        const newNode = wrapper.firstElementChild;
        if (newNode) record.timelineEl.replaceChild(newNode, el);
    }

    function scheduleHistorySync() {
        if (historySyncTimer) clearTimeout(historySyncTimer);
        historySyncTimer = setTimeout(() => {
            historySyncTimer = null;
            syncHistory({ includeUser: false }).catch(() => {});
        }, 700);
    }

    function applyLiveCardState(summary, groupId, ts, dedupeKey = '', { suppressDomInsert = false, rawTs = '' } = {}) {
        const nextGroupId = groupId || activeLiveGroupId || 'active';
        const record = getLiveCardRecord(nextGroupId);
        // A converted card is now a terminal project chip — its task is owned by the
        // project panel. Ignore ALL further frames (incl. terminal) so they neither
        // overwrite the chip nor dereference the nulled element refs (P3).
        if (record.root?.dataset?.projectCreated === '1') return;
        const nextPhase = summary.phase || '';
        if (record.finished && !isTerminalTaskPhase(nextPhase, summary.terminal)) {
            return;
        }

        if (!record.isSubagent) {
            activeLiveGroupId = nextGroupId;
            reanchorTaskCard(record, rawTs, { suppressDomInsert });
        }
        ensureLiveCardVisible(record, { suppressDomInsert });
        record.updates += 1;
        const wasFinished = record.finished;
        // Prefer the last meaningful headline when an update carries none (e.g. a
        // structured terminal marker), so finishing a card doesn't blank its title.
        const headline = summary.headline || record.lastHumanHeadline || 'Working...';
        const syntheticKey = summary.dedupeKey || dedupeKey || `${summary.phase || 'working'}|${headline}|${summary.body || ''}`;
        const isLegacyParentSubagentKey = syntheticKey.startsWith('parent-subagent:');
        const inPlaceByKey = isLegacyParentSubagentKey
            || syntheticKey.startsWith('subagent-lifecycle:')
            || syntheticKey.startsWith('subagent-progress:')
            || syntheticKey.startsWith('subagent-result:')
            || syntheticKey.startsWith('task_done|');
        if (!isLegacyParentSubagentKey) {
            record.finished = isTerminalTaskPhase(nextPhase, summary.terminal);
        }
        record.root.dataset.finished = record.finished ? '1' : '0';
        if (summary.human && headline) {
            record.lastHumanHeadline = headline;
        }

        const shouldPromote =
            Boolean(summary.promote)
            || !record.lastHumanHeadline
            || record.finished;
        const activeHeadline = shouldPromote
            ? headline
            : (record.lastHumanHeadline || headline);
        const activePhase = record.finished
            ? (summary.phase || 'done')
            : (shouldPromote ? (summary.phase || 'working') : (record.phaseEl.dataset.phase || 'working'));

        record.phaseEl.dataset.phase = activePhase;
        record.phaseEl.textContent = formatLiveCardPhaseLabel(activePhase);
        record.phaseEl.className = `chat-live-phase ${activePhase}`;
        // Cluster B: a coined project name takes the title slot; the live activity
        // headline still renders in the timeline lines below. Falls back to the
        // activity headline until the proactive namer has produced a name.
        record.titleEl.textContent = record.suggestedName || activeHeadline;
        // P1 (v6.82): collapsed activity line. The raw candidate is remembered
        // even while the line is suppressed (unnamed root card) so a late coined
        // name can populate it without waiting for the next frame. Root cards
        // take candidates from HUMAN frames only — terminal/lifecycle fallbacks
        // ("Done") are phase chips, not activity, and must not overwrite or
        // duplicate the last meaningful action.
        // The line takes the FULL companion (fullHeadline/fullBody) whenever the
        // summarizer carried one: `headline`/`body` are timeline PREVIEWS capped
        // for the bounded row, and the wrapping activity line has no such bound,
        // so reusing the preview here would be an undisclosed cut of owner-facing
        // cognitive text (BIBLE P1 / DEVELOPMENT no-silent-truncation).
        const rootActivityHeadline = (!record.isSubagent && summary.human)
            ? String(summary.fullHeadline || activeHeadline || '') : '';
        const childActivityBody = record.isSubagent
            ? String(summary.fullBody || summary.body || '') : '';
        const activityCandidate = String(
            (record.isSubagent ? childActivityBody : rootActivityHeadline) || '',
        ).trim();
        const activityText = projectCollapsedActivity({
            isSubagent: record.isSubagent,
            suggestedName: record.suggestedName,
            headline: record.isSubagent ? '' : rootActivityHeadline,
            body: childActivityBody,
            previous: record.collapsedActivity,
        });
        if (activityCandidate) record.collapsedActivity = activityCandidate;
        renderCollapsedActivity(record, activityText);

        const shouldRenderLine = summary.visible !== false && Boolean(headline || summary.body);
        // Legacy parent-subagent rows update in place if replayed from old
        // history. Child-card lifecycle/progress rows also evolve in place.
        let timelineUpdate = 'none';
        let patchIndex = -1;
        if (shouldRenderLine) {
            const lastIdx = record.items.length - 1;
            // Full-array dedup (Variant A): match the incoming line's key ANYWHERE in
            // the card, not only against the last item. Otherwise a background
            // syncHistory(rebuildAll=false) re-feeds historical progress lines whose
            // key != the last item, and each gets re-appended → the "Notes" count
            // grows without bound on every sync/reconnect.
            const existingIdx = record.items.findIndex((it) => it.dedupeKey === syntheticKey);
            if (existingIdx !== -1 && inPlaceByKey) {
                const it = record.items[existingIdx];
                it.phase = summary.phase || it.phase;
                it.headline = headline || it.headline;
                it.fullHeadline = summary.fullHeadline || headline || it.fullHeadline;
                it.body = summary.body || '';
                it.fullBody = summary.fullBody || summary.body || it.fullBody || '';
                it.fullRef = summary.fullRef || it.fullRef || '';
                it.truncated = summary.truncated || it.truncated || false;
                it.ts = ts || it.ts;
                patchIndex = existingIdx;
                timelineUpdate = 'patch-at';
            } else if (existingIdx === lastIdx && existingIdx !== -1) {
                // Consecutive live duplicate of the most recent line → coalesce count.
                const it = record.items[existingIdx];
                it.count += 1;
                it.ts = ts || it.ts;
                it.fullHeadline = summary.fullHeadline || it.fullHeadline || it.headline;
                it.fullBody = summary.fullBody || it.fullBody || it.body;
                it.fullRef = summary.fullRef || it.fullRef || '';
                it.truncated = summary.truncated || it.truncated || false;
                timelineUpdate = 'patch-last';
            } else if (existingIdx !== -1) {
                // Already rendered earlier in this card (e.g. a historical progress line
                // re-fed by a background sync). Do NOT re-append (the unbounded "Notes"
                // growth) and do NOT bump its count — just keep its timestamp fresh.
                const it = record.items[existingIdx];
                it.ts = ts || it.ts;
                timelineUpdate = 'duplicate-skip';
            } else {
                const lineKey = `line-${Date.now()}-${Math.random().toString(16).slice(2)}`;
                record.items.push({
                    phase: summary.phase || 'working',
                    headline: headline || 'Update',
                    fullHeadline: summary.fullHeadline || headline || 'Update',
                    body: summary.body || '',
                    fullBody: summary.fullBody || summary.body || '',
                    fullRef: summary.fullRef || '',
                    truncated: summary.truncated || false,
                    ts: ts || '',
                    count: 1,
                    dedupeKey: syntheticKey,
                    lineKey,
                });
                timelineUpdate = 'append';
            }
        }
        updateLiveCardCount(record);
        // "Latest" is an ACTIVITY clock, not a bookkeeping clock: a cost-only frame
        // (task_cost_finalized and friends carry no human narration) must not make a
        // silent card look freshly active. Only frames that actually said something
        // move it.
        if (ts && (summary.human || activityCandidate)) record.latestActivityTs = ts;
        // P1 (v6.82): sticky cost — only frames carrying task-scope accounting
        // evidence attach a costProjection; a costless frame re-renders the
        // previous projection instead of erasing it.
        if (summary.costProjection) {
            record.costMeta = mergeStickyCostMeta(record.costMeta, summary.costProjection);
        }
        record.metaEl.innerHTML = [
            nextGroupId === 'bg-consciousness' ? 'Background thinking' : '',
            ...(Array.isArray(summary.meta) ? summary.meta : []),
            ...((record.costMeta && Array.isArray(record.costMeta.meta)) ? record.costMeta.meta : []),
            record.latestActivityTs ? `Latest ${record.latestActivityTs}` : '',
        ].filter(Boolean).map((item) => `<span class="chat-live-meta-text">${escapeHtml(item)}</span>`).join('');
        // Incremental updates; full rebuilds stay limited to toggles.
        const lastItem = record.items[record.items.length - 1];
        if (timelineUpdate === 'append' && lastItem) {
            appendTimelineItem(lastItem, record);
        } else if (timelineUpdate === 'patch-last' && lastItem) {
            patchLastTimelineItem(lastItem, record);
        } else if (timelineUpdate === 'patch-at' && patchIndex !== -1) {
            patchTimelineItemAt(record.items[patchIndex], record);
        }
        ensureLiveCardVisible(record, { suppressDomInsert });
        syncLiveCardLayout(record);
        hideTypingIndicatorOnly();
        const justFinished = record.finished && !wasFinished;
        const drivesComposerStatus = !isBackgroundTaskId(nextGroupId);
        // P5: a finished card must not keep offering "Cancel run".
        if (justFinished) syncCancelRunButton(record);
        if (record.finished) {
            setLiveCardTypingVisible(record, false);
            markTaskComplete(nextGroupId, summary.phase || 'done');
            if (justFinished) {
                if (!stickyExpandedSlots.has(record.groupId)) {
                    setLiveCardExpanded(record, record.isSubagent && nestedSubagentsExpanded);
                }
                scheduleHistorySync();
            }
            syncLiveCardToggle(record);
            if (drivesComposerStatus) {
                setStatus(summary.phase === 'error' || summary.phase === 'timeout' ? 'error' : 'online', summary.phase === 'error' || summary.phase === 'timeout' ? 'Attention' : 'Online');
            }
        } else {
            setLiveCardTypingVisible(record, true);
            if (drivesComposerStatus) {
                setStatus('thinking', 'Working...');
            } else if (!hasActiveLiveCard() && statusBadge && ['Thinking...', 'Working...'].includes(statusBadge.textContent)) {
                setStatus('online', 'Online');
            }
        }
        if (summary.expandByDefault) {
            setLiveCardExpanded(record, true);
        }
    }

    function finishLiveCard(groupId = '', phase = '') {
        const record = groupId
            ? liveCardRecords.get(groupId)
            : (activeLiveGroupId ? liveCardRecords.get(activeLiveGroupId) : null);
        if (!record) return;
        // A converted card is a terminal project chip now — ignore late terminal
        // frames so they neither overwrite the chip nor touch its element refs (T4).
        if (record.root?.dataset?.projectCreated === '1') return;
        const wasFinished = record.finished;
        record.finished = true;
        record.root.dataset.finished = '1';
        syncCancelRunButton(record);
        const activePhase = ['error', 'timeout', 'warn', 'cancelled'].includes(phase) ? phase : 'done';
        record.phaseEl.dataset.phase = activePhase;
        record.phaseEl.textContent = formatLiveCardPhaseLabel(activePhase);
        record.phaseEl.className = `chat-live-phase ${activePhase}`;
        setLiveCardTypingVisible(record, false);
        markTaskComplete(record.groupId, activePhase);
        if (!wasFinished) {
            if (!stickyExpandedSlots.has(record.groupId)) {
                setLiveCardExpanded(record, record.isSubagent && nestedSubagentsExpanded);
            }
            scheduleHistorySync();
        }
        syncLiveCardToggle(record);
        if (activeLiveGroupId === record.groupId) activeLiveGroupId = '';
        if (!hasActiveLiveCard()) {
            setStatus(activePhase === 'error' || activePhase === 'timeout' ? 'error' : 'online',
                      activePhase === 'error' || activePhase === 'timeout' ? 'Attention' : 'Online');
        }
    }

    function appendTaskSummaryToLiveCard(msg, { suppressDomInsert = false } = {}) {
        const taskId = msg?.task_id || activeLiveGroupId || '';
        const rawTs = msg?.ts || new Date().toISOString();
        if (registerEphemeralDecisionFrame(msg)) return;
        if (!taskId) {
            finishLiveCard(taskId, 'done');
            return;
        }
        // Cluster B: a card (re)built from a task_summary row also carries the coined name
        // on reload (history attaches suggested_name to summary rows too) — apply it so the
        // title survives even when no progress row was retained.
        if (msg?.suggested_name) applySuggestedName(taskId, msg.suggested_name);
        const reviewDetails = formatReviewProjection(msg?.review_projection);
        const taskState = getTaskUiState(taskId, Boolean(reviewDetails));
        if (!taskState) {
            finishLiveCard(taskId, 'done');
            return;
        }
        if (reviewDetails) taskState.forceCard = true;
        revealBufferedCardIfNeeded(taskState, { suppressDomInsert, rawTs });
        if (!taskState.cardVisible) {
            markAssistantReply(taskId);
            return;
        }
        const record = liveCardRecords.get(taskId);
        const reasonCode = msg?.reason_code ? String(msg.reason_code) : '';
        const severity = taskOutcomeSeverity(msg || {});
        const terminalPhase = taskTerminalPhase(msg || {});
        const failedResult = severity === 'error';
        // P5: a cancelled root says "Cancelled", never a generic "Done" headline.
        const doneHeadline = severity === 'cancelled'
            ? 'Cancelled'
            : (failedResult && reasonCode
                ? `Done: ${reasonCode}`
                : (severity === 'warn'
                    ? (reasonCode ? `Finished with warnings: ${reasonCode}` : 'Finished with warnings')
                    : ((record && record.lastHumanHeadline) || 'Done')));
        applyLiveCardState(
            {
                phase: terminalPhase,
                headline: doneHeadline,
                body: reviewDetails,
                visible: Boolean(reviewDetails),
                human: false,
                promote: true,
                terminal: true,
                expandByDefault: Boolean(reviewDetails),
                costProjection: taskCostProjection(msg, rawTs),
            },
            taskId,
            normalizeLogTs(rawTs),
            `task_done|${taskId}`,
            { suppressDomInsert, rawTs },
        );
        finishLiveCard(taskId, terminalPhase);
        scheduleTaskUiCleanup(taskState);
    }

    // child task_id -> { parentId, role }, learned from subagent lifecycle pings.
    // Child cards are mounted under the parent card, but their phase/terminal
    // state is independent so a finished child cannot mark the parent done.
    const subagentChildParents = new Map();
    // Children whose card has reached a terminal phase — late non-lifecycle
    // progress for these must NOT revive it back to "working".
    const subagentTerminalChildren = new Set();

    const SUBAGENT_EVENT_PHASE = {
        scheduled: 'start', running: 'working', completed: 'done', completed_warn: 'warn',
        // 'cancelled' is a first-class terminal phase (v6.82 P5) — a cancelled
        // child must read as Cancelled, not as a generic amber "Notice".
        failed: 'error', rejected: 'warn', cancelled: 'cancelled', interrupted: 'warn',
    };
    const SUBAGENT_EVENT_LABEL = {
        scheduled: 'scheduled', running: 'running', completed: 'done', completed_warn: 'done with warnings',
        failed: 'failed', rejected: 'rejected', cancelled: 'cancelled', interrupted: 'interrupted',
    };

    // E2 (v6.39 UI): merge a subagent's parent/role/model, PRESERVING a previously-seen model
    // when a later (model-less) event — e.g. a synthesized terminal — updates the entry, so the
    // "role · model" headline survives the child's lifecycle.
    function setSubagentParent(childId, { parentId = '', role = '', model = '' } = {}) {
        const prev = subagentChildParents.get(childId) || {};
        subagentChildParents.set(childId, {
            parentId: parentId || prev.parentId || '',
            role: role || prev.role || '',
            model: String(model || '').trim() || prev.model || '',
        });
    }

    function formatSubagentHeadline(childId = '', role = '', label = '', model = '') {
        const shortChild = String(childId || '').slice(0, 8);
        const cleanRole = String(role || '').trim();
        const suffix = label ? ` — ${label}` : '';
        // Show the resolved model compactly NEXT TO the role (e.g. "planning-scout · gemini-3.5-flash").
        const modelPart = compactModel(model) ? ` · ${compactModel(model)}` : '';
        if (cleanRole) {
            return `${cleanRole}${modelPart}${shortChild ? ` (${shortChild})` : ''}${suffix}`;
        }
        return `Subagent ${shortChild || 'child'}${modelPart}${suffix}`;
    }

    function updateLiveCardFromProgressMessage(msg) {
        const taskId = msg?.task_id || activeLiveGroupId || '';
        const rawTs = msg?.ts || new Date().toISOString();
        if (registerEphemeralDecisionFrame(msg)) return;
        if (!taskId) return;
        // P5: host-attested cancelable marker (live WS frames AND history replay —
        // progress rows persist it through _PROGRESS_META_FIELDS). The supervisor
        // stamps it ONLY on lineage-resolved non-subagent ROOTS (with the RUNNING
        // row's authoritative lineage on the same frame), so the marker itself is
        // the truth — re-deriving rootness from frame shape here would wrongly
        // reject a timeout-retry root, whose root_task_id names the ORIGINAL task
        // while the endpoint can cancel its current id. A direct-chat turn never
        // carries the marker.
        if (msg?.cancelable === true && msg?.task_id) markTaskCancelable(String(msg.task_id));
        // Subagent lifecycle pings render as child cards linked to the parent;
        // they must not update the parent card's terminal state.
        const lifecycleParent = String(msg?.parent_task_id || '').trim();
        if (msg?.subagent_event && lifecycleParent && lifecycleParent !== taskId) {
            updateSubagentCardFromEvent(msg, rawTs);
            return;
        }
        // A known subagent child's own (non-lifecycle) progress stays on the child
        // card so parallel work remains visible without expanding the parent.
        if (subagentChildParents.has(taskId)) {
            routeSubagentProgressToCard(taskId, msg);
            return;
        }
        // Progress messages are visible status; do not force-open completed replay.
        const taskState = getTaskUiState(taskId, true);
        if (taskState && !taskState.completed) taskState.forceCard = true;
        const summary = summarizeChatLiveEvent({
            type: 'send_message',
            is_progress: true,
            content: msg?.content || msg?.text || '',
            text: msg?.content || msg?.text || '',
            task_id: taskId,
            subagent_event: msg?.subagent_event || '',
            subagent_task_id: msg?.subagent_task_id || '',
            root_task_id: msg?.root_task_id || '',
            parent_task_id: msg?.parent_task_id || '',
            delegation_role: msg?.delegation_role || '',
            subagent_role: msg?.subagent_role || '',
            status: msg?.status || '',
            cost_usd: msg?.cost_usd,
            cost_accounting_status: msg?.cost_accounting_status,
            cost_accounting_error: msg?.cost_accounting_error,
            cost_final: msg?.cost_final,
            cost_usd_with_children: msg?.cost_usd_with_children,
            cost_with_children_partial: msg?.cost_with_children_partial,
            reserved_usd: msg?.reserved_usd,
            unresolved_upper_bound_usd: msg?.unresolved_upper_bound_usd,
            unknown_unmetered: msg?.unknown_unmetered,
            result: msg?.result || '',
            trace_summary: msg?.trace_summary || '',
            error: msg?.error || '',
            artifact_status: msg?.artifact_status || '',
            lifecycle: msg?.lifecycle || null,
        });
        if (!summary) return;
        const presented = withTaskCostMeta(summary, msg, { rawTs });
        queueTaskLiveUpdate(presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs);
        // Cluster B: history progress recs carry the coined name (live progress does
        // not — the live path uses the separate `task_named` event). Apply it after the
        // card exists so a reload shows the same title.
        if (msg?.suggested_name) applySuggestedName(taskId, msg.suggested_name);
        // History projects authoritative terminal truth onto the latest progress
        // anchor when the best-effort task_summary row is absent. Apply that truth
        // before a later ordinary assistant row closes the card, so replay cannot
        // freeze a degraded review as a green completion.
        if (
            msg?.task_terminal_status
            && (msg?.outcome_axes || msg?.review_projection || msg?.reason_code)
        ) {
            appendTaskSummaryToLiveCard(msg);
        }
    }

    function updateSubagentCardFromEvent(evt, tsValue) {
        if (!evt || String(evt.delegation_role || '').toLowerCase() !== 'subagent') return false;
        const parentId = String(evt.parent_task_id || '').trim();
        const childId = String(evt.subagent_task_id || evt.task_id || '').trim();
        if (!parentId || !childId || parentId === childId) return false;
        const rawEvent = String(evt.subagent_event || 'update').toLowerCase();
        const completionSeverity = rawEvent === 'completed' ? taskOutcomeSeverity(evt) : 'done';
        const event = rawEvent === 'completed'
            ? (completionSeverity === 'error' ? 'failed' : (completionSeverity === 'warn' ? 'completed_warn' : 'completed'))
            : rawEvent;
        const role = String(evt.subagent_role || '').trim();
        setSubagentParent(childId, { parentId, role, model: evt.model });
        const { model } = subagentChildParents.get(childId) || {};
        // NOTE: 'interrupted' is intentionally excluded — it is retryable
        // (written before requeue), so the child resumes and its later progress
        // must still flow to its card. Only true terminals lock it.
        if (['completed', 'completed_warn', 'failed', 'cancelled', 'rejected'].includes(event)) {
            subagentTerminalChildren.add(childId);  // lock the child card terminal
        }
        const phase = SUBAGENT_EVENT_PHASE[event] || 'working';
        const label = SUBAGENT_EVENT_LABEL[event] || event;
        const shortChild = childId.slice(0, 8);
        const headline = formatSubagentHeadline(childId, role, label, model);
        // Surface the child's handoff (result/trace/error) as expandable detail
        // on the child card.
        const detailParts = [];
        const reviewDetails = formatReviewProjection(evt.review_projection);
        if (reviewDetails) detailParts.push(`[REVIEW]\n${reviewDetails}`);
        if (evt.result) detailParts.push(`[RESULT]\n${String(evt.result)}`);
        if (evt.trace_summary) detailParts.push(`[TRACE]\n${String(evt.trace_summary)}`);
        if (evt.error) detailParts.push(`[ERROR]\n${String(evt.error)}`);
        const metaBits = [`child=${shortChild}`];
        if (role) metaBits.push(`role=${role}`);
        forceTaskCard(parentId, tsValue);
        const childState = getTaskUiState(childId, true);
        if (childState && !childState.completed) childState.forceCard = true;
        getSubagentCardRecord(childId, parentId, role);
        queueTaskLiveUpdate({
            phase,
            headline,
            body: reviewDetails,
            fullBody: detailParts.join('\n\n'),
            visible: true,
            promote: true,
            expandByDefault: Boolean(reviewDetails),
            meta: metaBits,
            costProjection: taskCostProjection(evt, tsValue),
            dedupeKey: `subagent-lifecycle:${childId}`,
            terminal: ['completed', 'completed_warn', 'failed', 'cancelled', 'rejected'].includes(event),
        }, childId, normalizeLogTs(tsValue || new Date().toISOString()), `subagent-lifecycle:${childId}`, tsValue);
        return true;
    }

    // Record a terminal child's latest narration as its collapsed activity
    // without touching phase, timeline or finished state (P1, v6.82).
    function rememberSubagentActivity(childId, msg) {
        const record = liveCardRecords.get(childId);
        if (!record || !record.activityEl) return;
        const line = String(msg?.content || msg?.text || '').trim().split('\n').filter(Boolean).pop() || '';
        if (!line) return;
        // The complete line is kept: this is owner-facing cognitive text, and the
        // CSS wraps it (`overflow-wrap:anywhere`), so a hardcoded slice here would
        // be an undisclosed cut (BIBLE P1 / DEVELOPMENT no-silent-truncation).
        record.collapsedActivity = line;
        renderCollapsedActivity(record, projectCollapsedActivity({
            isSubagent: true,
            body: record.collapsedActivity,
            previous: record.collapsedActivity,
        }));
    }

    // A known child's own (non-lifecycle) progress updates the linked child card.
    function routeSubagentProgressToCard(childId, msg) {
        const info = subagentChildParents.get(childId);
        if (!info) return;
        if (subagentTerminalChildren.has(childId)) {
            // Never revive a finished child's phase/timeline — but its last
            // narration is still its collapsed activity (P1, v6.82): on replay
            // terminal children are pre-marked before pass 1, so without this a
            // finished child card would show an empty activity line.
            rememberSubagentActivity(childId, msg);
            return;
        }
        const { parentId, role, model } = info;
        const shortChild = String(childId).slice(0, 8);
        const line = String(msg?.content || msg?.text || '').trim().split('\n').filter(Boolean).pop() || '';
        const headline = formatSubagentHeadline(childId, role, 'running', model);
        const rawTs = msg?.ts || new Date().toISOString();
        forceTaskCard(parentId, rawTs);
        const childState = getTaskUiState(childId, true);
        if (childState && !childState.completed) childState.forceCard = true;
        getSubagentCardRecord(childId, parentId, role);
        const meta = [`child=${shortChild}`];
        if (role) meta.push(`role=${role}`);
        queueTaskLiveUpdate({
            phase: 'working',
            headline,
            body: line.slice(0, 200),
            // The timeline row stays a bounded preview; the collapsed activity
            // line reads this uncut companion instead (P1: no silent cut).
            fullBody: line,
            visible: true,
            promote: true,
            meta,
            dedupeKey: `subagent-progress:${childId}`,
        }, childId, normalizeLogTs(rawTs), `subagent-progress:${childId}`, rawTs);
    }

    function routeSubagentFinalMessageToCard(taskId, msg) {
        const childId = String(taskId || '').trim();
        const info = subagentChildParents.get(childId);
        if (!childId || !info) return false;
        const { parentId, role, model } = info;
        const shortChild = childId.slice(0, 8);
        const text = String(msg?.content || msg?.text || '').trim();
        const rawTs = msg?.ts || new Date().toISOString();
        forceTaskCard(parentId, rawTs);
        const record = getSubagentCardRecord(childId, parentId, role);
        const priorTerminalPhase = record?.finished ? String(record.phaseEl?.dataset?.phase || '') : '';
        const resultPhase = ['warn', 'error', 'cancelled'].includes(priorTerminalPhase) ? priorTerminalPhase : 'done';
        const meta = [`child=${shortChild}`];
        if (role) meta.push(`role=${role}`);
        queueTaskLiveUpdate({
            phase: resultPhase,
            headline: formatSubagentHeadline(childId, role, 'result', model),
            body: text.slice(0, 200),
            fullBody: text,
            visible: true,
            promote: true,
            meta,
            dedupeKey: `subagent-result:${childId}`,
            terminal: true,
        }, childId, normalizeLogTs(rawTs), `subagent-result:${childId}`, rawTs);
        return true;
    }

    // Resolve a child's card from the child's terminal task_done
    // (which arrives on the log channel without subagent metadata).
    function routeSubagentTerminalToCard(childId, evt) {
        const info = subagentChildParents.get(childId);
        if (!info) return false;
        const status = String(evt.status || '').toLowerCase();
        const severity = taskOutcomeSeverity(evt);
        const failed = severity === 'error' || status === 'failed';
        const cancelled = status === 'cancelled' || status === 'cancel_requested';
        const rejected = status === 'rejected_duplicate';
        const event = failed ? 'failed' : cancelled ? 'cancelled' : rejected ? 'rejected' : (severity === 'warn' ? 'completed_warn' : 'completed');
        updateSubagentCardFromEvent({
            delegation_role: 'subagent',
            parent_task_id: info.parentId,
            subagent_task_id: childId,
            subagent_role: info.role,
            subagent_event: event,
            model: info.model || '',
            review_projection: evt.review_projection,
            result: evt.result || '',
            error: evt.error || '',
            cost_usd: evt.cost_usd,
            cost_accounting_status: evt.cost_accounting_status,
            cost_accounting_error: evt.cost_accounting_error,
            cost_final: evt.cost_final,
            cost_usd_with_children: evt.cost_usd_with_children,
            cost_with_children_partial: evt.cost_with_children_partial,
            reserved_usd: evt.reserved_usd,
            unresolved_upper_bound_usd: evt.unresolved_upper_bound_usd,
            unknown_unmetered: evt.unknown_unmetered,
        }, evt.ts || evt.timestamp || new Date().toISOString());
        return true;
    }

    function updateLiveCardFromLogEvent(evt) {
        if (!evt || !isGroupedTaskEvent(evt)) return;
        showContextFitToast(evt);
        if (registerEphemeralDecisionFrame(evt)) return;
        const taskId = getLogTaskGroupId(evt) || activeLiveGroupId || '';
        if (!taskId) return;
        const eventType = evt.type || evt.event || '';
        const rawTs = evt.ts || evt.timestamp || new Date().toISOString();
        // A known subagent child's log events update its linked child card.
        if (subagentChildParents.has(taskId)) {
            if (eventType === 'task_done') {
                routeSubagentTerminalToCard(taskId, evt);
                return;
            }
            if (subagentTerminalChildren.has(taskId)) return;
            if (eventType === 'tool_call_started') {
                markTaskToolCall(taskId, 1, false, rawTs);
            } else if ((eventType === 'task_metrics_event' || eventType === 'task_eval') && Number.isFinite(Number(evt.tool_calls))) {
                markTaskToolCall(taskId, Number(evt.tool_calls), true, rawTs);
            } else if (
                eventType === 'tool_call_timeout'
                || eventType === 'tool_timeout'
                || eventType === 'llm_round_error'
                || eventType === 'llm_api_error'
                || (eventType === 'tool_call_finished' && evt.is_error)
            ) {
                forceTaskCard(taskId, rawTs);
            }
            const summary = summarizeChatLiveEvent(evt);
            if (!summary) return;
            const info = subagentChildParents.get(taskId);
            if (info) getSubagentCardRecord(taskId, info.parentId, info.role);
            const presented = withTaskCostMeta(summary, evt, {
                replace: eventType === 'task_done' || eventType === 'task_cost_finalized',
                rawTs,
            });
            queueTaskLiveUpdate(presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs);
            return;
        }
        if (eventType === 'tool_call_started') {
            markTaskToolCall(taskId, 1, false, rawTs);
        } else if ((eventType === 'task_metrics_event' || eventType === 'task_eval') && Number.isFinite(Number(evt.tool_calls))) {
            markTaskToolCall(taskId, Number(evt.tool_calls), true, rawTs);
        } else if (
            eventType === 'tool_call_timeout'
            || eventType === 'tool_timeout'
            || eventType === 'llm_round_error'
            || eventType === 'llm_api_error'
            || (eventType === 'tool_call_finished' && evt.is_error)
        ) {
            forceTaskCard(taskId, rawTs);
        }
        if (eventType === 'task_done' && formatReviewProjection(evt.review_projection)) {
            forceTaskCard(taskId, rawTs);
        }
        const summary = summarizeChatLiveEvent(evt);
        if (!summary) return;
        const presented = withTaskCostMeta(summary, evt, {
            replace: eventType === 'task_done' || eventType === 'task_cost_finalized',
            rawTs,
        });
        queueTaskLiveUpdate(presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs);
        updateSubagentCardFromEvent(evt, rawTs);
        if (eventType === 'task_done') {
            const taskState = getTaskUiState(taskId, false);
            revealBufferedCardIfNeeded(taskState, { rawTs });
        }
    }

    function addMessage(text, role, markdown = false, timestamp = null, isProgress = false, opts = {}) {
        const pending = !!opts.pending;
        const ephemeral = !!opts.ephemeral;
        const clientMessageId = opts.clientMessageId || '';
        const senderLabel = opts.senderLabel || '';
        const senderSessionId = opts.senderSessionId || '';
        const source = opts.source || '';
        const systemType = opts.systemType || '';
        const taskId = opts.taskId || '';
        const ts = timestamp || new Date().toISOString();
        const messageKey = buildMessageKey(role, text, ts, {
            clientMessageId,
            systemType,
            isProgress,
            source,
            senderLabel,
            senderSessionId,
            taskId,
        });
        if (messageKey && seenMessageKeys.has(messageKey)) return null;

        if (!isProgress && !ephemeral) {
            persistedHistory.push({
                text,
                role,
                ts,
                markdown: !!markdown,
                systemType,
                source,
                senderLabel,
                senderSessionId,
                clientMessageId,
                taskId,
            });
            persistVisibleHistory();
        }

        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}` + (isProgress ? ' progress' : '');
        if (pending) bubble.classList.add('pending');
        if (ephemeral) bubble.dataset.ephemeral = '1';
        if (clientMessageId) bubble.dataset.clientMessageId = clientMessageId;
        if (systemType) bubble.dataset.systemType = systemType;
        if (senderSessionId) bubble.dataset.senderSessionId = senderSessionId;
        if (taskId) bubble.dataset.taskId = taskId;

        const sender = getSenderLabel(role, isProgress, systemType, { source, senderLabel, senderSessionId });
        const rendered = role === 'user'
            ? escapeHtml(text)
            : (role === 'system' && systemType === 'skill_review'
                ? renderSkillReviewDisclosure(text)
                : renderMarkdown(text));
        const timeFmt = formatMsgTime(ts);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const pendingHtml = pending ? `<div class="msg-pending">Queued until reconnect</div>` : '';
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(sender)}</div>
            <div class="message">${rendered}</div>
            ${pendingHtml}
            ${timeHtml}
        `;
        const skillReviewToggle = bubble.querySelector('[data-skill-review-toggle]');
        if (skillReviewToggle) {
            skillReviewToggle.addEventListener('click', () => {
                const disclosure = bubble.querySelector('[data-skill-review-disclosure]');
                const full = bubble.querySelector('[data-skill-review-full]');
                const label = bubble.querySelector('.skill-review-toggle-label');
                const expanded = disclosure?.dataset.expanded === '1';
                if (!disclosure || !full) return;
                disclosure.dataset.expanded = expanded ? '0' : '1';
                full.hidden = expanded;
                skillReviewToggle.setAttribute('aria-expanded', expanded ? 'false' : 'true');
                if (label) label.textContent = expanded ? 'Show review' : 'Hide review';
                requestAnimationFrame(() => updateMessagesPadding({ preserveStickiness: true }));
            });
        }
        stampNodeTimestamp(bubble, ts);
        insertMessageNode(bubble, { forceStick: !!opts.forceStick });
        renderRoutingAnnotation(bubble, opts.chatAnnotation);
        rememberMessageKey(messageKey);
        if (pending && clientMessageId) pendingUserBubbles.set(clientMessageId, bubble);
        return bubble;
    }

    function routingAnnotationText(annotation) {
        if (!annotation || typeof annotation !== 'object') return '';
        const action = String(annotation.action || '');
        const status = String(annotation.status || '');
        const target = String(annotation.target || '');
        if (status === 'pending') return 'Choosing the right destination…';
        if (status === 'needs_manual_target') {
            const optionLabels = (Array.isArray(annotation.options) ? annotation.options : [])
                .map(option => {
                    if (!option || typeof option !== 'object') return '';
                    if (option.label) return String(option.label);
                    if (option.action === 'new_task_in_project') {
                        return `New task in ${String(option.project_name || 'Project')}`;
                    }
                    return String(option.title || option.task_id || option.project_name || option.project_id || '');
                })
                .filter(Boolean);
            if (optionLabels.length) return `Choose a target · ${optionLabels.join(' / ')}`;
            return target ? `Choose a target · ${target}` : 'Choose a target';
        }
        if (status === 'project_unavailable') return 'Project is unavailable';
        const labels = {
            mailbox_delivery: 'Delivered to task',
            steer_task: 'Steered task',
            promote_chat_to_task: 'Started task',
            route_to_project: 'Routed to project',
            project_route: 'Project routing',
        };
        const label = labels[action] || status.replaceAll('_', ' ') || action.replaceAll('_', ' ');
        return target && label ? `${label} · ${target}` : label;
    }

    function renderRoutingAnnotation(bubble, annotation) {
        if (!bubble) return false;
        const text = routingAnnotationText(annotation);
        let note = bubble.querySelector('.msg-routing-annotation');
        if (!text) {
            note?.remove();
            delete bubble.dataset.chatAnnotationStatus;
            return false;
        }
        if (!note) {
            note = document.createElement('div');
            note.className = 'msg-routing-annotation';
            const time = bubble.querySelector('.msg-time');
            if (time) time.before(note);
            else bubble.append(note);
        }
        const status = String(annotation.status || '');
        note.textContent = text;
        note.dataset.annotationStatus = status;
        bubble.dataset.chatAnnotationStatus = status;
        return true;
    }

    function updateMessageAnnotation(clientMessageId, annotation) {
        const messageId = String(clientMessageId || '');
        if (!messageId) return false;
        const bubble = Array.from(messagesDiv.querySelectorAll('.chat-bubble.user[data-client-message-id]'))
            .find((candidate) => candidate.dataset.clientMessageId === messageId);
        return renderRoutingAnnotation(bubble, annotation);
    }

    function clearTransientRoutingAnnotations() {
        for (const note of messagesDiv.querySelectorAll(
            '.msg-routing-annotation[data-annotation-status="pending"]',
        )) {
            const bubble = note.closest('.chat-bubble');
            if (bubble) delete bubble.dataset.chatAnnotationStatus;
            note.remove();
        }
    }

    function markPendingDelivered(clientMessageId) {
        const bubble = pendingUserBubbles.get(clientMessageId || '');
        if (!bubble) return;
        bubble.classList.remove('pending');
        bubble.querySelector('.msg-pending')?.remove();
        pendingUserBubbles.delete(clientMessageId);
    }

    function ensureWelcomeMessage() {
        if (!isMain) return;
        if (welcomeShown) return;
        const hasRealBubbles = Array.from(messagesDiv.querySelectorAll('.chat-bubble')).some(
            bubble => !bubble.classList.contains('typing-bubble')
        );
        if (hasRealBubbles) return;
        welcomeShown = true;
        addMessage('Ouroboros has awakened', 'assistant', false, null, false, { ephemeral: true });
    }

    async function syncHistory({ includeUser = false, fromReconnect = false } = {}) {
        if (historySyncPromise) {
            // Preserve reconnect intent so retiredTaskIds is cleared after this sync.
            if (fromReconnect) {
                pendingReconnectSync = true;
                return historySyncPromise.then(() => {
                    // The first reconnect waiter consumes the queued rebuild; any
                    // concurrent waiter sees and awaits the newly installed global
                    // promise. No caller may render its Reconnected banner against
                    // the intermediate (non-rebuilt) DOM.
                    if (pendingReconnectSync) {
                        pendingReconnectSync = false;
                        return syncHistory({ includeUser: false, fromReconnect: true });
                    }
                    return historySyncPromise || lastHistorySyncSucceeded;
                });
            }
            return historySyncPromise;
        }
        historySyncPromise = (async () => {
            try {
                const resp = await apiFetch(`/api/chat/history?limit=1000${isMain ? '' : `&chat_id=${chatId}`}`, { cache: 'no-store' });
                if (!resp.ok) {
                    lastHistorySyncSucceeded = false;
                    return false;
                }
                const data = await resp.json();
                const messages = Array.isArray(data.messages) ? data.messages : [];
                const scrollBeforeSync = {
                    top: messagesDiv.scrollTop,
                    nearBottom: isNearBottom(),
                    anchor: captureVisibleTimelineAnchor(),
                };

                // First load/reconnect trusts server history and fully rebuilds the
                // feed; routine post-completion syncs only fold in new task cards.
                const rebuildAll = !historyLoaded || fromReconnect;
                // On a soft reconnect the module (and its dedupe set) survives, so a
                // plain re-sync would skip user messages and dedupe-drop every
                // assistant bubble — the conversation would vanish. Restore user text
                // and rebuild from durable history whenever we rebuild.
                const renderUser = includeUser || fromReconnect;
                if (!historyLoaded || fromReconnect) retiredTaskIds.clear();
                if (rebuildAll) {
                    for (const record of liveCardRecords.values()) record.root?.remove();
                    liveCardRecords.clear();
                    taskUiStates.clear();
                    ephemeralDecisionTaskIds.clear();
                    activeLiveGroupId = '';
                    // Atomically drop the standalone message bubbles and the dedupe
                    // state so the rebuild below cannot produce duplicates even if
                    // stale bubbles lingered in the DOM. Keep the typing indicator.
                    for (const bubble of Array.from(messagesDiv.querySelectorAll('.chat-bubble'))) {
                        if (!bubble.classList.contains('typing-bubble')) bubble.remove();
                    }
                    seenMessageKeys.clear();
                    messageKeyOrder.length = 0;
                    // Subagent lineage + terminal state live only in memory. Clear and
                    // rebuild them from durable history BEFORE the card passes, so a
                    // finished child card finalizes regardless of replay order or which
                    // event carried the terminal signal (a subagent 'completed' event OR
                    // a server task_terminal_status). Otherwise finished children stick
                    // on "working" and get revived by parent heartbeats on reload.
                    subagentChildParents.clear();
                    subagentTerminalChildren.clear();
                    for (const msg of messages) {
                        if (String(msg.delegation_role || '').toLowerCase() !== 'subagent') continue;
                        const parentId = String(msg.parent_task_id || '').trim();
                        const childId = String(msg.subagent_task_id || msg.task_id || '').trim();
                        if (!parentId || !childId || parentId === childId) continue;
                        if (!subagentChildParents.has(childId)) {
                            setSubagentParent(childId, { parentId, role: String(msg.subagent_role || '').trim(), model: msg.model });
                        }
                        const ev = String(msg.subagent_event || '').toLowerCase();
                        if (msg.task_terminal_status || ['completed', 'completed_warn', 'failed', 'cancelled', 'rejected'].includes(ev)) {
                            subagentTerminalChildren.add(childId);
                        }
                    }
                }

                // Two passes ensure cards exist before finishLiveCard() marks them done.

                // Pass 1 builds timelines with DOM insertion suppressed.
                _syncPass1Active = true;
                try { for (const msg of messages) {
                    const taskId = msg.task_id || '';
                    if (!taskId) continue;
                    if (retiredTaskIds.has(taskId)) continue;
                    if (msg.is_progress) {
                        updateLiveCardFromProgressMessage(msg);
                        continue;
                    }
                    if (msg.system_type === 'task_summary') {
                        // Historical cards only for non-trivial tasks.
                        const hadToolCalls = (msg.tool_calls || 0) > 0;
                        const hadMultipleRounds = (msg.rounds || 0) > 1;
                        const severity = taskOutcomeSeverity(msg);
                        const needsVisibleTerminal = severity === 'error' || severity === 'warn' || severity === 'cancelled';
                        if (hadToolCalls || hadMultipleRounds || needsVisibleTerminal) {
                            const taskState = getTaskUiState(taskId, true);
                            if (taskState) taskState.forceCard = true;
                        }
                        // Pass 2 inserts this in the right transcript position.
                        appendTaskSummaryToLiveCard(msg, { suppressDomInsert: true });
                    }
                } } finally { _syncPass1Active = false; }

                // Pass 2 inserts cards at the first visible task message, then finishes them.
                const insertedCardTaskIds = new Set();
                function reorderDirtyCardIfNeeded(rec) {
                    if (!rec?._anchorOrderDirty || rec.isSubagent || !rec.root?.isConnected) return;
                    insertMessageNode(rec.root, { reorderExisting: true });
                    rec._anchorOrderDirty = false;
                }
                function insertCardIfNeeded(taskId) {
                    if (!taskId || insertedCardTaskIds.has(taskId)) return;
                    insertedCardTaskIds.add(taskId);
                    const rec = liveCardRecords.get(taskId);
                    reorderDirtyCardIfNeeded(rec);
                    if (rec && rec.root && !rec.root.isConnected) {
                        if (rec.isSubagent) ensureLiveCardVisible(rec);
                        else insertMessageNode(rec.root);
                    }
                }
                for (const msg of messages) {
                    const taskId = msg.task_id || '';
                    if (!renderUser && msg.role === 'user') continue;
                    if (msg.is_progress) {
                        // Progress-only/failed tasks still anchor at their first event.
                        insertCardIfNeeded(taskId);
                        continue;
                    }
                    if (msg.system_type === 'task_summary') continue;
                    // A delivered document is a media bubble, not a task-final
                    // message — render it BEFORE the taskId/finishLiveCard block so
                    // a mid-task file delivery replayed while its task is still
                    // running does not falsely finalize that task's live card.
                    if (msg.msg_type === 'document') {
                        appendDocumentBubble(msg);
                        continue;
                    }
                    if (taskId && (msg.role === 'assistant' || msg.role === 'system')) {
                        if (subagentChildParents.has(taskId)) {
                            insertCardIfNeeded(taskId);
                            routeSubagentFinalMessageToCard(taskId, msg);
                            const taskState = getTaskUiState(taskId, false);
                            const record = liveCardRecords.get(taskId);
                            const preservedPhase = taskState?.completedPhase || record?.phaseEl?.dataset?.phase || 'done';
                            finishLiveCard(taskId, preservedPhase);
                            continue;
                        }
                        insertCardIfNeeded(taskId);
                        const taskState = getTaskUiState(taskId, false);
                        const record = liveCardRecords.get(taskId);
                        const preservedPhase = taskState?.completedPhase || record?.phaseEl?.dataset?.phase || 'done';
                        finishLiveCard(taskId, preservedPhase);
                    }
                    addMessage(msg.text, msg.role, !!msg.markdown, msg.ts || null, false, {
                        systemType: msg.system_type || '',
                        source: msg.source || '',
                        senderLabel: msg.sender_label || '',
                        senderSessionId: msg.sender_session_id || '',
                        clientMessageId: msg.client_message_id || '',
                        taskId,
                        chatAnnotation: msg.chat_annotation || null,
                    });
                }
                // Resolve cards whose task is already terminal on the server
                // (crash storm / hard timeout / cancellation write a terminal
                // status but no task_summary). Without this their progress-only
                // cards re-inflate as "Working" forever on reload/reconnect.
                const terminalTaskRecords = new Map();
                for (const msg of messages) {
                    const tid = msg.task_id || '';
                    if (tid && msg.task_terminal_status) {
                        terminalTaskRecords.set(tid, {
                            ...msg,
                            status: String(msg.task_terminal_status),
                        });
                    }
                }
                for (const [tid, terminalRecord] of terminalTaskRecords) {
                    const status = String(terminalRecord.status || '');
                    // Subagent terminal status resolves the child card, not the
                    // parent. Otherwise reload can revive a crashed/cancelled child.
                    if (subagentChildParents.has(tid)) {
                        routeSubagentTerminalToCard(tid, terminalRecord);
                        continue;
                    }
                    const rec = liveCardRecords.get(tid);
                    if (rec && !rec.finished) {
                        insertCardIfNeeded(tid);
                        if (terminalRecord.outcome_axes || terminalRecord.review_projection || terminalRecord.reason_code) {
                            appendTaskSummaryToLiveCard(terminalRecord);
                        } else {
                            // P5: shared terminal mapping — a cancelled root replays
                            // as "Cancelled", never as a generic "Done".
                            finishLiveCard(tid, taskTerminalPhase(terminalRecord));
                        }
                    }
                }

                // Append disconnected visible cards after mid-task reload; skip trivial placeholders.
                for (const [tid, rec] of liveCardRecords) {
                    reorderDirtyCardIfNeeded(rec);
                    if (rec && rec.root && !rec.root.isConnected && !retiredTaskIds.has(tid)) {
                        const ts = taskUiStates.get(tid);
                        if (ts && !ts.cardVisible && ts.completed) continue;
                        if (rec.isSubagent) ensureLiveCardVisible(rec);
                        else insertMessageNode(rec.root);
                    }
                }

                // After first load, unfinished foreground cards still show typing.
                if (!historyLoaded) {
                    const hasOngoingTask = Array.from(liveCardRecords.values()).some(isForegroundLiveCard);
                    if (hasOngoingTask) showTyping();
                }

                // One-shot server recall seed includes other clients without resetting
                // ArrowUp during reconnect. Merge [server..., local...], newest wins.
                if (!inputHistorySeededFromServer) {
                    const serverTexts = [];
                    for (const msg of messages) {
                        if (msg.role !== 'user') continue;
                        let text = (msg.text || '').trim();
                        if (text) serverTexts.push(text);
                    }
                    const combined = [...serverTexts, ...inputHistory];
                    const deduped = [];
                    const seen = new Set();
                    for (let i = combined.length - 1; i >= 0; i--) {
                        if (!seen.has(combined[i])) {
                            deduped.unshift(combined[i]);
                            seen.add(combined[i]);
                        }
                    }
                    inputHistory.length = 0;
                    inputHistory.push(...deduped.slice(-50));
                    saveInputHistory(inputHistory);
                    inputHistoryIndex = inputHistory.length;
                    inputHistorySeededFromServer = true;
                }

                const wasFirstLoad = !historyLoaded;
                historyLoaded = true;
                lastHistorySyncSucceeded = true;
                // First load jumps to latest; reconnect preserves older-message reading.
                if (wasFirstLoad || (fromReconnect ? scrollBeforeSync.nearBottom : isNearBottom())) {
                    updateMessagesPadding({ preserveStickiness: false });
                    scrollToBottomAfterLayout();
                } else if (fromReconnect) {
                    // Rebuild may add old rows ABOVE and new rows BELOW the viewport
                    // simultaneously. Total scrollHeight delta cannot distinguish the
                    // two and over-scrolls readers by the height of new bottom content.
                    // Restore the first visible timestamped node to its prior visual
                    // offset instead; equal-ts ordinals preserve arrival-order identity.
                    // Live-card expansion and responsive layout settle on RAF; restore
                    // after two frames so asynchronous card heights above the anchor
                    // cannot move the reader immediately after this function resolves.
                    await new Promise((resolve) => requestAnimationFrame(
                        () => requestAnimationFrame(resolve)
                    ));
                    const restoredFromAnchor = restoreVisibleTimelineAnchor(scrollBeforeSync.anchor);
                    if (!restoredFromAnchor) messagesDiv.scrollTop = scrollBeforeSync.top;
                    updateScrollButton();
                }
                return messages.length > 0;
            } catch (err) {
                lastHistorySyncSucceeded = false;
                const socketState = ws?.ws?.readyState;
                const expectedDisconnect = socketState !== WebSocket.OPEN;
                if (expectedDisconnect && err instanceof TypeError) {
                    return false;
                }
                console.error('Failed to load chat history:', err);
                return false;
            } finally {
                historySyncPromise = null;
                // A reconnect caller waiting on the active promise owns replay of
                // pendingReconnectSync above, so its own promise resolves only after
                // the authoritative rebuild.
            }
        })();
        return historySyncPromise;
    }

    function cancelHistoryPaint() {
        historyPaintGeneration += 1;
    }

    async function refreshHistory({ revision = 0 } = {}) {
        const generation = ++historyPaintGeneration;
        await syncHistory({ includeUser: true });
        if (!lastHistorySyncSucceeded || generation !== historyPaintGeneration || page.hidden) {
            return { painted: false, revision: Number(revision) || 0 };
        }
        // A successful fetch is not a read acknowledgement until the rebuilt
        // DOM has crossed an actual browser paint while this Project remains
        // visible. Two frames cover layout followed by paint/composite.
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        return {
            painted: generation === historyPaintGeneration && !page.hidden,
            revision: Math.max(0, Number(revision) || 0),
        };
    }

    (async () => {
        await loadUiPreferences();
        if (await syncHistory({ includeUser: true })) return;
        try {
            const saved = JSON.parse(sessionStorage.getItem(storeKey(CHAT_STORAGE_KEY)) || '[]');
            for (const msg of saved) {
                addMessage(msg.text, msg.role, !!msg.markdown, msg.ts || null, false, {
                    systemType: msg.systemType || '',
                    source: msg.source || '',
                    senderLabel: msg.senderLabel || '',
                    senderSessionId: msg.senderSessionId || '',
                    clientMessageId: msg.clientMessageId || '',
                    taskId: msg.taskId || '',
                });
            }
        } catch {}
        historyLoaded = true;
        ensureWelcomeMessage();
    })();

    function rememberInput(text) {
        if (!text) return;
        if (inputHistory[inputHistory.length - 1] !== text) inputHistory.push(text);
        saveInputHistory(inputHistory);
        inputHistoryIndex = inputHistory.length;
        inputDraft = '';
    }

    function resizeChatInput({ preserveStickiness = false } = {}) {
        const caretAtEnd = input.selectionEnd >= input.value.length - 1;
        const previousScrollTop = input.scrollTop;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        input.scrollTop = caretAtEnd ? input.scrollHeight : previousScrollTop;
        updateMessagesPadding({ preserveStickiness });
    }

    function restoreInputHistory(step) {
        if (!inputHistory.length) return;
        if (step < 0) {
            if (input.selectionStart !== 0 || input.selectionEnd !== 0) return;
            if (inputHistoryIndex === inputHistory.length) inputDraft = input.value;
            inputHistoryIndex = Math.max(0, inputHistoryIndex - 1);
            input.value = inputHistory[inputHistoryIndex] || '';
        } else {
            if (input.selectionStart !== input.value.length || input.selectionEnd !== input.value.length) return;
            inputHistoryIndex = Math.min(inputHistory.length, inputHistoryIndex + 1);
            input.value = inputHistoryIndex === inputHistory.length ? inputDraft : (inputHistory[inputHistoryIndex] || '');
        }
        resizeChatInput({ preserveStickiness: false });
        const cursor = input.value.length;
        input.setSelectionRange(cursor, cursor);
    }

    async function sendMessage(planMode = false) {
        if (sendBtn.disabled) return;  // guard against Enter re-entry during async upload
        let text = input.value.trim();
        // The owner's pure typed request (before attachment lines) — captured so a
        // live card spawned by this message can name a project from it on a "turn
        // into project" conversion even before the task records its objective (P1,
        // direct-chat case: the server has no title/objective/queue source yet).
        const objectiveText = text;
        const hasAttachments = pendingAttachments.length > 0;
        let uploadedAttachments = [];
        let attachmentMeta = [];
        if (!text && !pendingAttachments.length) return;
        if (pendingAttachments.length) {
            // Upload immediately before send; offline queueing would orphan files.
            if (ws.ws?.readyState !== WebSocket.OPEN) {
                showToast('Cannot attach file while offline. Reconnect and try again.', 'error');
                return;
            }
            const staged = [...pendingAttachments];
            const uploaded = [];
            setAttachmentUploadState(true);
            setSendBusy(true, staged.length > 1 ? 'Uploading files' : 'Uploading');
            try {
                for (const stagedItem of staged) {
                    if (ws.ws?.readyState !== WebSocket.OPEN) throw new Error('Connection closed during upload. Reconnect and try again.');
                    const formData = new FormData();
                    formData.append('file', stagedItem.file);
                    const resp = await apiFetch('/api/chat/upload', { method: 'POST', body: formData });
                    const data = await resp.json().catch(() => ({}));
                    if (!resp.ok || !data.ok) {
                        throw new Error(data.error || resp.statusText);
                    }
                    uploaded.push({
                        filename: data.filename || '',
                        path: data.path || '',
                        display_name: data.display_name || stagedItem.display_name,
                        mime: data.mime || stagedItem.file?.type || '',
                    });
                }
                if (ws.ws?.readyState !== WebSocket.OPEN) throw new Error('Connection closed after upload. Reconnect and try again.');
                uploadedAttachments = uploaded;
                const attachmentLines = uploaded
                    .map((item) => `[Attached file: ${item.display_name} saved to ${item.path}]`)
                    .join('\n');
                text += (text ? '\n\n' : '') + attachmentLines;
                // Structured attachment metadata rides the WS frame so the
                // gateway can hand image uploads to the model as NATIVE image
                // blocks (vision models) instead of only a path label.
                attachmentMeta = uploaded.map((item) => ({
                    filename: item.filename,
                    display_name: item.display_name,
                    mime: item.mime || '',
                }));
            } catch (e) {
                await cleanupUploadedAttachments(uploaded);
                showToast('Upload error: ' + e.message, 'error');
                return;  // pending attachments and preview remain so the user can retry
            } finally {
                setAttachmentUploadState(false);
                setSendBusy(false);
            }
        }
        if (!text) return;
        const forcePlan = !!planMode && !text.startsWith('/');
        const result = ws.send({
            type: 'chat',
            content: text,
            sender_session_id: chatSessionId,
            force_plan: forcePlan,
            ...(isMain ? {} : { chat_id: chatId }),
            ...(projectId ? { project_id: projectId } : {}),
            ...(attachmentMeta.length ? { attachments: attachmentMeta } : {}),
        }, hasAttachments ? { queue: false } : undefined);
        if (hasAttachments && result?.status !== 'sent') {
            await cleanupUploadedAttachments(uploadedAttachments);
            showToast('Connection lost before send. Reconnect and try again.', 'error');
            return;
        }
        // One-shot: disarm Swarm now that the message is sent.
        if (planMode) setSwarm(false);
        // Hand the objective to the NEXT main-chat live card this message spawns.
        if (isMain && objectiveText) _pendingCardObjective = objectiveText;
        if (hasAttachments) {
            pendingAttachments = [];
            updateAttachmentPreview();
        }
        rememberInput(text);
        input.value = '';
        addMessage(text, 'user', false, null, false, {
            pending: result?.status === 'queued',
            source: 'web',
            senderSessionId: chatSessionId,
            clientMessageId: result?.clientMessageId || '',
            forceStick: true,
        });
        resizeChatInput({ preserveStickiness: false });
        scrollToBottomAfterLayout();
    }

    // Send mode lives on DOM so CSS and click/Enter share one source.
    const sendGroup = page.querySelector('.chat-send-group');

    // Swarm is a one-shot arm: the next send goes through plan_task multi-model
    // brainstorm/planning, then the pill auto-disarms so it never sticks.
    const swarmBtn = byId('swarm');
    function swarmArmed() {
        return swarmBtn?.dataset.armed === 'true';
    }
    function setSwarm(armed) {
        if (swarmBtn) swarmBtn.dataset.armed = armed ? 'true' : 'false';
    }

    function setSendBusy(busy, label = '') {
        sendGroup.dataset.busy = busy ? '1' : '0';
        sendBtn.disabled = busy;
        if (busy) {
            sendBtn.textContent = label || 'Sending';
            sendBtn.title = label || 'Sending';
        } else {
            sendBtn.textContent = 'Send';
            sendBtn.title = 'Send message';
        }
    }

    swarmBtn?.addEventListener('click', () => setSwarm(!swarmArmed()));

    // Context-mode quick toggle: the owner endpoint hot-applies the setting
    // without a restart; Max -> Low is accepted only while Ouroboros is idle.
    const contextModeBtn = byId('context-mode');
    contextModeBtn?.addEventListener('click', async (event) => {
        const seg = event.target.closest('.chat-seg');
        if (!seg || contextModeBtn.dataset.disabled === 'true') return;
        const next = seg.dataset.mode === 'low' ? 'low' : 'max';
        const current = contextModeBtn.dataset.contextMode === 'low' ? 'low' : 'max';
        // A displayed `low` that is a system AUTO-DOWNGRADE is not an owner selection:
        // re-picking Low must still POST (the endpoint is idempotent and clears the
        // derived flag), or an unconfirmable-window install stays wedged with scope
        // review blocking every commit and no reachable way to declare Low.
        const derivedLow = contextModeBtn.dataset.contextModeAutoLow === 'true';
        if (next === current && !(next === 'low' && derivedLow)) return;
        contextModeBtn.dataset.disabled = 'true';
        const postMode = (mode) => apiFetch('/api/owner/context-mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode }),
        });
        try {
            let resp = await postMode(next);
            if (!resp.ok) {
                let payload = {};
                try { payload = await resp.json(); } catch {}
                // Max context mode needs the active model's 1M-token window confirmed.
                // Offer a plain, model-scoped confirmation (kept until the model changes).
                const ack = payload?.needs_ack;
                if (next === 'max' && ack && ack.model) {
                    const ok = window.confirm(
                        `${payload.error || 'Max context mode needs a confirmed 1M-token window.'}\n\n` +
                        `Confirm that this model supports a 1,000,000-token context window?\n` +
                        `  provider: ${ack.provider || '(default)'}\n  model: ${ack.model}\n` +
                        `  base_url: ${ack.base_url || '(default)'}\n\n` +
                        `This applies only to this exact model/provider and is removed if you change it.`
                    );
                    if (ok) {
                        const ackResp = await apiFetch('/api/owner/capability-ack', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                provider: ack.provider, model: ack.model,
                                base_url: ack.base_url, window_tokens: 1000000,
                                note: 'owner-confirmed via context-mode toggle',
                            }),
                        });
                        if (ackResp.ok) {
                            resp = await postMode(next);  // retry with the confirmation in place
                        } else {
                            showToast('Could not save the confirmation.', 'error');
                        }
                    }
                }
            }
            if (resp.ok) {
                contextModeBtn.dataset.contextMode = next;
            } else {
                let message = 'Could not change context mode.';
                try { const p = await resp.json(); if (p?.error) message = p.error; } catch {}
                showToast(message, 'error');
            }
        } catch (e) {
            showToast(`Could not change context mode: ${e.message || e}`, 'error');
            /* leave the current value; /api/state refresh will resync */
        } finally {
            contextModeBtn.dataset.disabled = 'false';
            refreshHeaderControlState(true);
        }
    });

    // Arrow wrappers avoid MouseEvent leaking into sendMessage(planMode).
    sendBtn.addEventListener('click', () => sendMessage(swarmArmed()));
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(swarmArmed());
            return;
        }
        if (e.key === 'ArrowUp' && !e.shiftKey) {
            restoreInputHistory(-1);
        } else if (e.key === 'ArrowDown' && !e.shiftKey) {
            restoreInputHistory(1);
        }
    });
    // Dynamic CSS reserve keeps the absolute composer from covering messages.
    function scrollToBottom() {
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function scrollToBottomAfterLayout() {
        requestAnimationFrame(() => {
            scrollToBottom();
            requestAnimationFrame(scrollToBottom);
        });
    }

    // P7 — per-instance scroll memory. Switching tabs/opening a project panel used
    // to drop this thread back to the very top (the browser zeroes a hidden
    // column's scrollTop, and toggling .page display can reset it too). We
    // remember where the user was and restore it on show: pinned to the latest
    // message in the common case, or the exact spot they'd scrolled back to.
    let _savedScrollTop = 0;
    let _savedStick = true;  // a fresh thread starts pinned to the newest message
    let _restoring = false;  // suppress saved-state writes during a restore pass
    const isInstanceVisible = () =>
        Boolean(messagesDiv) && messagesDiv.offsetParent !== null && !document.hidden;
    messagesDiv?.addEventListener('scroll', () => {
        // Ignore the spurious scrollTop=0 a browser emits while the column is
        // hidden — that would erase the real position we want to restore.
        if (!isInstanceVisible()) return;
        // WebKit fires a scrollTop=0 event when a hidden column is re-shown, and
        // our own re-pin loop writes scrollTop too — neither is a real user
        // scroll, so during a restore pass we only refresh the button, never the
        // saved position (which would corrupt a mid-history restore to the top).
        if (_restoring) { updateScrollButton(); return; }
        _savedScrollTop = messagesDiv.scrollTop;
        _savedStick = isNearBottom();
        updateScrollButton();
    }, { passive: true });

    // Round glass "jump to newest" affordance — shown only when the user has
    // scrolled up away from the bottom, for both the main chat and panels.
    function updateScrollButton() {
        if (!scrollBottomBtn) return;
        scrollBottomBtn.classList.toggle('visible', isInstanceVisible() && !isNearBottom());
    }
    scrollBottomBtn?.addEventListener('click', () => {
        _savedStick = true;
        scrollToBottomAfterLayout();
        updateScrollButton();
    });

    function restoreScrollPosition() {
        if (!isInstanceVisible()) return;  // hidden column has no geometry yet
        // WebKit (the desktop WKWebView) leaves a freshly un-hidden flex column's
        // scrollTop pinned at 0 for a frame or two after the page is shown, so a
        // single/double rAF re-pin (which is enough in Chromium) lands the user at
        // the very top. Re-apply the target position across several frames until
        // the late relayout settles, then keep the button state in sync.
        _restoring = true;
        const targetStick = _savedStick;
        const targetTop = _savedScrollTop;
        let frames = 0;
        const apply = () => {
            if (!isInstanceVisible()) { _restoring = false; return; }
            // scrollHeight is re-read each frame so a sticky thread tracks late
            // card-layout growth; a restored mid-history spot re-pins to the exact
            // saved offset (idempotent, so it isn't overridden).
            messagesDiv.scrollTop = targetStick ? messagesDiv.scrollHeight : targetTop;
            updateScrollButton();
            if (++frames < 12) requestAnimationFrame(apply);
            else _restoring = false;
        };
        requestAnimationFrame(apply);
    }

    function updateMessagesPadding(options = {}) {
        const preserveStickiness = options.preserveStickiness !== false;
        const shouldStick = preserveStickiness && isNearBottom();
        if (pageHeader && messagesDiv) {
            // The main header wraps to two rows on narrow viewports. Reserve its
            // REAL rendered height so scrollTop=0 never hides the first message
            // behind the absolute overlay; project panels have no overlay header.
            const headerReserve = Math.max(56, Math.ceil(pageHeader.offsetHeight || 0));
            page.style.setProperty('--chat-header-reserve', `${headerReserve}px`);
        }
        if (inputArea && messagesDiv) {
            const reserve = Math.max(92, Math.ceil(inputArea.offsetHeight || 0) + 16);
            // Set on the instance page root so it cascades to #chat-messages
            // (padding) AND the sibling scroll-to-bottom button (bottom offset).
            page.style.setProperty('--chat-input-reserve', `${reserve}px`);
        }
        if (shouldStick) scrollToBottomAfterLayout();
        updateScrollButton();
    }

    function installChatResizeObservers() {
        if (typeof ResizeObserver !== 'function') return;
        let queued = false;
        const schedule = () => {
            if (queued) return;
            queued = true;
            requestAnimationFrame(() => {
                queued = false;
                updateMessagesPadding({ preserveStickiness: true });
            });
        };
        const observer = new ResizeObserver(schedule);
        if (pageHeader) observer.observe(pageHeader);
        if (inputArea) observer.observe(inputArea);
        if (messagesDiv) observer.observe(messagesDiv);
    }

    installChatResizeObservers();

    input.addEventListener('input', () => {
        if (inputHistoryIndex === inputHistory.length) inputDraft = input.value;
        resizeChatInput({ preserveStickiness: false });
    });

    headerActions?.addEventListener('click', (event) => {
        const button = event.target.closest('[data-chat-command]');
        if (!button) return;
        button.closest('details')?.removeAttribute('open');
        const command = button.dataset.chatCommand;
        if (command === 'evolve') {
            const next = !button.classList.contains('on');
            button.classList.toggle('on', next);
            ws.send({ type: 'command', cmd: `/evolve ${next ? 'start' : 'stop'}` });
            return;
        }
        if (command === 'bg') {
            const next = !button.classList.contains('on');
            button.classList.toggle('on', next);
            ws.send({ type: 'command', cmd: `/bg ${next ? 'start' : 'stop'}` });
            return;
        }
        if (command === 'review') {
            ws.send({ type: 'command', cmd: '/review' });
            return;
        }
        if (command === 'restart') {
            ws.send({ type: 'command', cmd: '/restart' });
            return;
        }
        if (command === 'panic' && confirm('Kill all workers immediately?')) {
            ws.send({ type: 'command', cmd: '/panic' });
        }
    });

    // The More menu is a native <details> (no auto-dismiss): collapse it when a
    // click/tap lands outside it, or on Escape, so it never stays stuck open.
    if (!asPanel) {
        const collapseHeaderMenus = (predicate) => {
            page.querySelectorAll('details.chat-header-more[open]').forEach((details) => {
                if (predicate(details)) details.removeAttribute('open');
            });
        };
        document.addEventListener('click', (event) => {
            collapseHeaderMenus((details) => !details.contains(event.target));
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') collapseHeaderMenus(() => true);
        });
    }

    budgetPill?.addEventListener('click', () => {
        if (typeof openDashboardTab === 'function') openDashboardTab('costs');
        else if (typeof openSettingsTab === 'function') openSettingsTab('costs');
    });

    if (asPanel) {
        // The panel has no global controls/budget to poll; seed the status from
        // the live socket so a late-created panel never gets stuck on
        // "Connecting…" (the one-shot WS `open` already fired before it existed;
        // future reconnects still update it via the shared `open` handler).
        if (ws.isConnected?.()) setStatus('online', 'Online');
    } else {
        refreshHeaderControlState(true);
        setInterval(refreshHeaderControlState, 3000);
    }

    const typingEl = document.createElement('div');
    // Per-instance id (main stays 'typing-indicator'; panels get a unique id) so
    // multiple open chat columns never collide on a duplicate DOM id.
    typingEl.id = idPrefix === 'chat' ? 'typing-indicator' : `${idPrefix}-typing-indicator`;
    typingEl.className = 'chat-bubble assistant typing-bubble';
    typingEl.style.display = 'none';
    typingEl.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
    messagesDiv.appendChild(typingEl);

    function hasActiveLiveCard() {
        return Array.from(liveCardRecords.values()).some(isForegroundLiveCard);
    }

    function showTyping() {
        if (!hasActiveLiveCard()) {
            typingEl.style.display = '';
            if (isNearBottom()) messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        setStatus('thinking', 'Thinking...');
    }

    function hideTypingIndicatorOnly() {
        typingEl.style.display = 'none';
    }

    function hideTyping() {
        hideTypingIndicatorOnly();
        if (statusBadge && ['Thinking...', 'Working...'].includes(statusBadge.textContent)) {
            setStatus('online', 'Online');
        }
    }

    const isKnownProjectFrame = (msg) => {
        const cid = Number(msg?.chat_id ?? 1);
        return state.projectChatIds instanceof Set && state.projectChatIds.has(cid);
    };

    function incrementUnreadIfNeeded(msg) {
        if (!isMain) return;  // the global unread badge tracks the main chat
        // Project visible_revision is the sole unread authority for a Project.
        // Main may mirror its summary/progress/log into the штаб live card, but
        // that presentation mirror must not create a second Main unread.
        if (isKnownProjectFrame(msg)) return;
        if (state.activePage === 'chat') return;
        state.unreadCount++;
        updateUnreadBadge();
    }

    ws.on('typing', (msg) => {
        if (!isMyThread(msg)) return;  // each column shows typing only for its own thread
        showTyping();
    });

    // One socket, client-side fan-out: project instances take only their own
    // thread. The MAIN instance keeps ordinary non-project traffic AND mirrors
    // project progress/digests/logs as the "штаб", but never raw project chat
    // user/assistant messages.
    const isProjectMirrorFrame = (msg) => {
        if (!msg) return false;
        if (msg.type === 'log') return true;
        if (msg.is_progress) return true;
        if (msg.system_type === 'task_summary' || msg.system_type === 'project_digest') return true;
        return false;
    };

    const isMyThread = (msg, { mirrorProject = false } = {}) => {
        const cid = Number(msg?.chat_id ?? 1);
        if (isMain) {
            if (isKnownProjectFrame(msg)) {
                return mirrorProject && isProjectMirrorFrame(msg);
            }
            return true;
        }
        return cid === chatId;
    };

    ws.on('chat', (msg) => {
        if (!isMyThread(msg, { mirrorProject: true })) return;
        if (msg.role === 'user') {
            const clientMessageId = msg.client_message_id || '';
            const senderSessionId = msg.sender_session_id || '';
            if (senderSessionId === chatSessionId && clientMessageId) {
                markPendingDelivered(clientMessageId);
                return;
            }
            addMessage(msg.content, 'user', false, msg.ts || null, false, {
                source: msg.source || '',
                senderLabel: msg.sender_label || '',
                senderSessionId,
                clientMessageId,
                taskId: msg.task_id || '',
            });
            incrementUnreadIfNeeded(msg);
            return;
        }

        if (msg.role === 'assistant' || msg.role === 'system') {
            hideTyping();
            const explicitTaskId = msg.task_id || '';
            const ephemeralDecision = registerEphemeralDecisionFrame(msg);
            if (msg.is_progress) {
                showTaskIncidentToast(msg);
                if (ephemeralDecision) return;
                updateLiveCardFromProgressMessage(msg);
                return;
            }
            if (msg.system_type === 'task_summary') {
                appendTaskSummaryToLiveCard(msg);
                markAssistantReply(explicitTaskId);
                incrementUnreadIfNeeded(msg);
                return;
            }
            if (explicitTaskId && subagentChildParents.has(explicitTaskId)) {
                routeSubagentFinalMessageToCard(explicitTaskId, msg);
                markAssistantReply(explicitTaskId);
                incrementUnreadIfNeeded(msg);
                return;
            }
            if (explicitTaskId) finishLiveCard(explicitTaskId);
            markAssistantReply(explicitTaskId);
            clearTransientRoutingAnnotations();
            addMessage(msg.content, msg.role, msg.markdown, msg.ts || null, false, {
                systemType: msg.system_type || '',
                source: msg.source || '',
                taskId: explicitTaskId,
            });
            incrementUnreadIfNeeded(msg);
        }
    });

    ws.on('message_annotation', (msg) => {
        if (!isMyThread(msg)) return;
        if (msg.annotation_type !== 'routing_ack') return;
        updateMessageAnnotation(msg.client_message_id || '', msg);
    });

    ws.on('log', (msg) => {
        if (!msg?.data) return;
        // Log frames now carry the task's chat_id (backend stamps it), so the
        // per-thread fan-out routes the full live card to its own column: a
        // project panel builds/animates/finalizes ITS card, while the main
        // chat mirrors project progress as штаб. Legacy frames without chat_id
        // default to the main chat.
        if (!isMyThread(msg, { mirrorProject: true })) return;
        updateLiveCardFromLogEvent(msg.data);
    });

    // Cluster B: the proactive namer coined a project name for a fresh card — show it
    // as the card title up front (turn-into-project then reuses the same name). Not
    // thread-gated on chat_id: the broadcast carries only task_id, and applySuggestedName
    // no-ops unless THIS thread already holds that card.
    ws.on('task_named', (msg) => {
        applySuggestedName(msg?.task_id || '', msg?.suggested_name || '');
    });

    ws.on('outbound_sent', (evt) => {
        markPendingDelivered(evt?.clientMessageId || '');
    });

    ws.on('photo', (msg) => {
        if (!isMyThread(msg)) return;
        hideTyping();
        const role = msg.role === 'user' ? 'user' : 'assistant';
        const sender = role === 'user'
            ? getSenderLabel('user', false, '', {
                source: msg.source || '',
                senderLabel: msg.sender_label || '',
                senderSessionId: msg.sender_session_id || '',
            })
            : 'Ouroboros';
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}`;
        const rawTs = msg.ts || new Date().toISOString();
        const timeFmt = formatMsgTime(rawTs);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const captionHtml = msg.caption ? `<div class="message">${escapeHtml(msg.caption)}</div>` : '';
        const mime = /^image\/[a-z0-9.+-]+$/i.test(String(msg.mime || '')) ? String(msg.mime) : 'image/png';
        const imageBase64 = /^[A-Za-z0-9+/=\s]+$/.test(String(msg.image_base64 || ''))
            ? String(msg.image_base64 || '').replace(/\s+/g, '')
            : '';
        const imageUrl = imageBase64 ? `data:${mime};base64,${imageBase64}` : '';
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(sender)}</div>
            ${captionHtml}
            <div class="message"><img class="chat-photo" src="${escapeHtmlAttr(imageUrl)}" alt="Photo attachment"></div>
            ${timeHtml}
        `;
        const img = bubble.querySelector('.chat-photo');
        if (img && imageUrl) {
            img.addEventListener('click', () => window.open(imageUrl, '_blank'));
        }
        stampNodeTimestamp(bubble, rawTs);
        insertMessageNode(bubble);
        incrementUnreadIfNeeded(msg);
    });

    ws.on('video', (msg) => {
        if (!isMyThread(msg)) return;
        hideTyping();
        const role = msg.role === 'user' ? 'user' : 'assistant';
        const sender = role === 'user'
            ? getSenderLabel('user', false, '', {
                source: msg.source || '',
                senderLabel: msg.sender_label || '',
                senderSessionId: msg.sender_session_id || '',
            })
            : 'Ouroboros';
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}`;
        const rawTs = msg.ts || new Date().toISOString();
        const timeFmt = formatMsgTime(rawTs);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const captionHtml = msg.caption ? `<div class="message">${escapeHtml(msg.caption)}</div>` : '';
        const mime = /^video\/[a-z0-9.+-]+$/i.test(String(msg.mime || '')) ? String(msg.mime) : 'video/mp4';
        const videoBase64 = /^[A-Za-z0-9+/=\s]+$/.test(String(msg.video_base64 || ''))
            ? String(msg.video_base64 || '').replace(/\s+/g, '')
            : '';
        const videoUrl = videoBase64 ? `data:${mime};base64,${videoBase64}` : '';
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(sender)}</div>
            ${captionHtml}
            <div class="message"><video class="chat-video" src="${escapeHtmlAttr(videoUrl)}" controls></video></div>
            ${timeHtml}
        `;
        stampNodeTimestamp(bubble, rawTs);
        insertMessageNode(bubble);
        incrementUnreadIfNeeded(msg);
    });

    // Shared document-bubble builder for both live WS frames and history replay.
    // Download priority: a durable server download_url routed through
    // downloadViaHostBridge (desktop host-bridge saves to Downloads instead of
    // navigating the WKWebView fullscreen; browser falls back to fetch+blob),
    // else an in-memory base64 blob (live-only), else a disabled label.
    function buildDocumentBubble(msg) {
        const role = msg.role === 'user' ? 'user' : 'assistant';
        const sender = role === 'user'
            ? getSenderLabel('user', false, '', {
                source: msg.source || '',
                senderLabel: msg.sender_label || '',
                senderSessionId: msg.sender_session_id || '',
            })
            : 'Ouroboros';
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}`;
        const rawTs = msg.ts || new Date().toISOString();
        const timeFmt = formatMsgTime(rawTs);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const captionHtml = msg.caption ? `<div class="message">${escapeHtml(msg.caption)}</div>` : '';
        const mime = /^[A-Za-z0-9!#$&^_.+-]+\/[A-Za-z0-9!#$&^_.+-]+$/.test(String(msg.mime || ''))
            ? String(msg.mime)
            : 'application/octet-stream';
        const fileBase64 = /^[A-Za-z0-9+/=\s]+$/.test(String(msg.file_base64 || ''))
            ? String(msg.file_base64 || '').replace(/\s+/g, '')
            : '';
        const downloadUrl = /^\/api\/files\/download\?/.test(String(msg.download_url || ''))
            ? String(msg.download_url)
            : '';
        const filename = String(msg.filename || 'file').replace(/[\r\n]+/g, ' ').slice(0, 200);
        const canDownload = Boolean(downloadUrl || fileBase64);
        // Body click = open in default OS app (external window); a separate ↓
        // button saves to ~/Downloads. Both degrade to a base64 blob when only
        // the live payload is present (no durable server URL to hand the bridge).
        const openHtml = canDownload
            ? `<button type="button" class="chat-file" data-open="1">📎 ${escapeHtml(filename)}</button>`
            : `<span class="chat-file chat-file-empty">📎 ${escapeHtml(filename)}</span>`;
        const downloadHtml = canDownload
            ? `<button type="button" class="chat-file-download" data-download="1" title="Download" aria-label="Download">↓</button>`
            : '';
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(sender)}</div>
            ${captionHtml}
            <div class="message"><div class="chat-file-row">${openHtml}${downloadHtml}</div></div>
            ${timeHtml}
        `;
        const saveBlobFallback = () => {
            const bytes = Uint8Array.from(atob(fileBase64), (c) => c.charCodeAt(0));
            const blobUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
            const tmp = document.createElement('a');
            Object.assign(tmp, { href: blobUrl, download: filename, rel: 'noopener' });
            document.body.appendChild(tmp);
            tmp.click();
            tmp.remove();
            setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
        };
        const openBtn = bubble.querySelector('.chat-file[data-open]');
        if (openBtn && canDownload) {
            openBtn.addEventListener('click', async () => {
                try {
                    if (downloadUrl) {
                        await openViaHostBridge(downloadUrl, filename);
                        return;
                    }
                    saveBlobFallback();
                } catch (err) {
                    showToast(`Could not open file: ${err && err.message ? err.message : err}`, 'error');
                }
            });
        }
        const dlBtn = bubble.querySelector('.chat-file-download[data-download]');
        if (dlBtn && canDownload) {
            dlBtn.addEventListener('click', async () => {
                try {
                    if (downloadUrl) {
                        await downloadViaHostBridge(downloadUrl, filename);
                        return;
                    }
                    saveBlobFallback();
                } catch (err) {
                    showToast(`Could not download file: ${err && err.message ? err.message : err}`, 'error');
                }
            });
        }
        stampNodeTimestamp(bubble, rawTs);
        return bubble;
    }

    // Dedup key shared by the live WS insert and history replay of the SAME
    // document (send_document uses one ts for both the frame and the persisted
    // row), so a routine background sync (rebuildAll=false, bubbles not cleared)
    // does not re-insert an already-rendered file bubble.
    function documentMessageKey(msg) {
        return [
            'document',
            String(msg.ts || ''),
            String(msg.download_url || ''),
            String(msg.filename || ''),
            String(msg.caption || ''),
        ].join('|');
    }

    function appendDocumentBubble(msg) {
        const key = documentMessageKey(msg);
        if (key && seenMessageKeys.has(key)) return false;
        rememberMessageKey(key);
        insertMessageNode(buildDocumentBubble(msg));
        return true;
    }

    ws.on('document', (msg) => {
        if (!isMyThread(msg)) return;
        hideTyping();
        if (appendDocumentBubble(msg)) incrementUnreadIfNeeded(msg);
    });

    let wsHasConnectedOnce = false;

    ws.on('open', () => {
        setStatus('online', 'Online');
        refreshHeaderControlState(true);
        const reconnectBanner =
            pendingReconnectBannerText
            || (wsHasConnectedOnce ? '♻️ Reconnected' : '');
        const shouldClearReconnectParams = Boolean(pendingReconnectBannerText);
        pendingReconnectBannerText = '';
        const isReconnect = wsHasConnectedOnce;
        wsHasConnectedOnce = true;
        updateMessagesPadding();
        loadUiPreferences()
            .then(() => syncHistory({ includeUser: !historyLoaded, fromReconnect: isReconnect }))
            .then((hasMessages) => {
                if (!hasMessages) ensureWelcomeMessage();
                if (reconnectBanner) {
                    addMessage(reconnectBanner, 'system', false, null, false, { ephemeral: true, systemType: 'reconnect' });
                    if (shouldClearReconnectParams) clearPendingReconnectBanner();
                }
            })
            .catch(() => {
                if (reconnectBanner) {
                    addMessage(reconnectBanner, 'system', false, null, false, { ephemeral: true, systemType: 'reconnect' });
                    if (shouldClearReconnectParams) clearPendingReconnectBanner();
                }
            });
    });

    ws.on('close', () => {
        hideTyping();
        setStatus('offline', 'Reconnecting...');
        syncHeaderControlState({ accounting: { available: false } });
    });

    return {
        page,
        chatId,
        projectId,
        // Called by app.js when this instance's panel is (re)shown so a project
        // thread restores its scroll position instead of jumping to the top (P7).
        restoreScrollPosition,
        refreshHistory,
        cancelHistoryPaint,
        destroy() {
            cancelHistoryPaint();
            try { page.remove(); } catch {}
        },
    };
}

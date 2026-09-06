// Pure chat-activity helpers shared by chat.js and dependency-free node tests:
// live-card presentation projections (moved verbatim from chat.js) plus the
// in-flight direct/ephemeral turn status reducer and snapshot hydration.
import { compactModel } from './log_events.js';
import { joinMarkdownHeadings } from './utils.js';
import { REUSABLE_TASK_IDS } from './task_control_menu.js';
import {
    accountedUpperBound,
    accountedUpperBoundWithChildren,
    escapeHtmlAttr,
    escapeHtmlText as escapeHtml,
    formatUsdWhole,
    renderMarkdown,
} from './utils.js';

export function withTaskCostMeta(summary, payload, { replace = false, rawTs = '' } = {}) {
    const projection = taskCostProjection(payload, rawTs);
    // `replace` frames (task_done/task_cost_finalized) never keep the
    // summarizer's own meta strings. Cost renders ONLY from the card's sticky
    // record.costMeta (applyLiveCardState); summarizer-built `cost=` strings
    // are dropped UNCONDITIONALLY — a frame without task-scope accounting
    // evidence must show no money at all, not a bare per-call number.
    const base = replace ? { ...summary, meta: [] } : summary;
    const out = projection ? { ...base, costProjection: projection } : { ...base };
    if (Array.isArray(out.meta) && out.meta.length) {
        out.meta = out.meta.filter((entry) => !String(entry || '').startsWith('cost='));
    }
    return out;
}

export function senderLabel(role, isProgress = false, systemType = '', opts = {}, chatSessionId = '') {
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

export function isLiveLineExpandable(item) {
    return Boolean(
        (item.fullHeadline && item.fullHeadline !== item.headline)
        || (item.fullBody && item.fullBody !== item.body)
        // P3: even when the preview equals the capped body, a server-truncated line
        // with a fetch ref has MORE to show (the genuinely-full output on demand).
        || (item.truncated && item.fullRef)
    );
}

export function buildTimelineItemHtml(item, record) {
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
        <span class="chat-live-line-title"${isProgressLine ? ' data-chat-markdown-enhanced' : ''}>${isProgressLine ? renderMarkdown(displayHeadline) : escapeHtml(displayHeadline)}</span>
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

// Sortable data-ts stamping for timeline nodes; anchor mode only ever moves a
// node's effective timestamp earlier so replay cannot teleport it downward.
export function stampNodeTimestamp(node, raw, { anchor = false } = {}) {
    if (!node) return false;
    const epoch = rawTimestampEpoch(raw);
    if (!Number.isFinite(epoch)) return false;
    if (anchor && node.dataset.ts) {
        const current = Number(node.dataset.ts);
        const next = Number.isFinite(current) ? Math.min(current, epoch) : epoch;
        if (node.dataset.ts !== String(next)) node.dataset.ts = String(next);
        return Number.isFinite(current) && next < current;
    }
    if (node.dataset.ts !== String(epoch)) node.dataset.ts = String(epoch);
    return false;
}

export function durableChatMediaUrl(value) {
    const url = String(value || '');
    return /^\/api\/tasks\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\/artifacts\/chat-media-[0-9a-f]{64}\.(png|jpg|gif|webp|mp4|webm)$/.test(url) ? url : '';
}

export function chatMediaMessageKey(msg) {
    return [
        msg.msg_type || msg.type,
        String(msg.task_id || ''),
        String(msg.ts || ''),
        String(msg.caption || ''),
        String(msg.mime || ''),
    ].join('|');
}

export function documentMessageKey(msg) {
    return [
        'document',
        String(msg.task_id || ''),
        String(msg.ts || ''),
        String(msg.download_url || ''),
        String(msg.filename || ''),
        String(msg.caption || ''),
    ].join('|');
}

export function pendingAttachmentBytes(items = []) {
    return items.reduce((total, item) => total + Number(item.file?.size || 0), 0);
}

export function isFileDrag(event) {
    return Array.from(event.dataTransfer?.types || []).includes('Files');
}

export function isNonTerminalMediaHistoryRow(msg) {
    return msg.system_type === 'photo' || msg.system_type === 'video';
}

export function isBackgroundTaskId(taskId = '') {
    return taskId === 'bg-consciousness';
}

export function shouldAlwaysShowTaskCard(taskId = '') {
    return isBackgroundTaskId(taskId);
}

export function isForegroundLiveCard(record) {
    return Boolean(
        record?.root?.isConnected && !record.finished && !record.reviewAnchor
        && !isBackgroundTaskId(record.groupId)
    );
}

export function shouldFirePanic(dialogResult) {
    return dialogResult === true;
}

export async function confirmAndSendPanic(deps) {
    const decision = await deps.openConfirmDialog({
        title: 'Panic — stop all workers',
        body: 'Kill all workers immediately?',
        confirmLabel: 'Kill all workers',
        cancelLabel: 'Keep running',
        danger: true,
    });
    if (shouldFirePanic(decision)) {
        deps.ws.send({ type: 'command', cmd: '/panic' });
        return true;
    }
    return false;
}

export function getOrCreateChatSessionId(storage, cryptoImpl, now = Date.now, random = Math.random) {
    try {
        const existing = storage.getItem('ouro_chat_session_id');
        if (existing) return existing;
        const created = cryptoImpl && typeof cryptoImpl.randomUUID === 'function'
            ? cryptoImpl.randomUUID()
            : `chat-${now()}-${random().toString(16).slice(2)}`;
        storage.setItem('ouro_chat_session_id', created);
        return created;
    } catch {
        return `chat-${now()}-${random().toString(16).slice(2)}`;
    }
}

export function projectIdFromTask(taskId = '', now = Date.now) {
    const seed = String(taskId || '')
        .toLowerCase()
        .replace(/[^a-z0-9_.-]+/g, '-')
        .replace(/^-+|-+$/g, '');
    return (seed ? `task-${seed}` : `task-${now().toString(36)}`).slice(0, 64);
}

export function loadChatInputHistory(storage, key) {
    try {
        const raw = JSON.parse(storage.getItem(key) || '[]');
        return Array.isArray(raw) ? raw.filter(Boolean).slice(-50) : [];
    } catch {
        return [];
    }
}

export function saveChatInputHistory(storage, key, entries) {
    try {
        storage.setItem(key, JSON.stringify(entries.slice(-50)));
    } catch {}
}

// Row-surface disclosure guard (v6.71.0), pure for node tests: returns the
// lineKey to toggle for a click landing on `target`, or '' when the click must
// NOT toggle (nested interactive element, or an active text selection inside
// the line).
export function liveLineRowToggleKey(target, selection = null) {
    const line = target?.closest?.('.chat-live-line.expandable');
    if (!line) return '';
    if (target.closest('button, a, input, textarea, select, label, summary, [contenteditable="true"]')) return '';
    if (selectionInside(line, selection)) return '';
    return (line.dataset && line.dataset.liveLineKey) || '';
}

/**
 * Two children of one parent whose compact headlines would read the same are
 * twins: the card then keeps the short task id to tell them apart. The key is
 * the DISPLAYED identity — the role (or its `Subagent` fallback) and the
 * compact model name — so equivalent spellings (`openai/gpt-5.6-sol`,
 * `openai::gpt-5.6-sol`, `gpt-5.6-sol`) collide exactly when the headlines do.
 */
export function subagentIdentityKey({ parentId = '', role = '', model = '' } = {}) {
    return `${parentId}\u0000${subagentIdentityTitle({ role, model })}`;
}

/** The child card's title: `role · model` (`Subagent · model` without a role), never an activity label. */
export function subagentIdentityTitle({ role = '', model = '' } = {}) {
    const name = String(role || '').trim() || 'Subagent';
    const short = compactModel(model);
    return short ? `${name} · ${short}` : name;
}

export function subagentTwin(children, childId) {
    const own = children.get(childId);
    if (!own) return false;
    const key = subagentIdentityKey(own);
    let n = 0;
    for (const c of children.values()) if (subagentIdentityKey(c) === key) n += 1;
    return n > 1;
}

/**
 * A non-collapsed text selection anchored inside `el`: the reader is copying,
 * not clicking, so a click-to-toggle surface must not fire (DESIGN.md §5).
 */
export function selectionInside(el, selection = globalThis.getSelection?.()) {
    return Boolean(selection && !selection.isCollapsed && el?.contains?.(selection.anchorNode));
}

/**
 * A click-to-toggle surface whose content stays selectable (DESIGN.md §5): a
 * pointer click whose drag left a selection inside it does nothing, and Enter /
 * Space activate it like a native button. The surface is a `div[role=button]`
 * because WebKit never lets text inside a real <button> be selected.
 */
export function bindContentButton(el, onActivate) {
    if (!el) return;
    el.addEventListener('click', (event) => {
        if (event.detail && selectionInside(el)) return;
        onActivate(event);
    });
    el.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            el.click();
        }
    });
}

/** Convert a raw source timestamp to sortable epoch milliseconds. */
export function rawTimestampEpoch(raw) {
    if (raw == null || raw === '') return NaN;
    const epoch = typeof raw === 'number' ? raw : Date.parse(String(raw));
    return Number.isFinite(epoch) ? epoch : NaN;
}

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
    // Presence means a VALUE, exactly as in `resolveCostPair`: a browser
    // producer literal (chat.js `costMetaKeys`) materializes every cost name it
    // knows, so a bare own property proves nothing about the frame. Counting
    // those as evidence made an evidence-free terminal frame project
    // "cost pending" and outrank a live ceiling on recency alone — the very
    // thing this projection promises never to do.
    const has = (key) => Object.prototype.hasOwnProperty.call(payload, key)
        && payload[key] !== undefined;
    // Task-scope accounting evidence only (v6.82 P1): a bare `cost_usd` is NOT
    // enough — llm_round_finished carries a per-round delta under that key, and
    // rendering it as task cost lied on the card. Subagent progress_meta and
    // task_done/task_cost_finalized frames carry cost_accounting_status /
    // cost_final alongside cost_usd, so honest task-scope frames still qualify.
    const hasAccountingEvidence = [
        'cost_accounting_status', 'cost_final',
        'cost_usd_with_children', 'cost_with_children_partial',
        'accounted_upper_bound_usd', 'accounted_upper_bound_usd_with_children',
        'reserved_usd', 'unresolved_upper_bound_usd', 'unknown_unmetered',
    ].some(has);
    if (!hasAccountingEvidence) return [];
    if (payload.cost_accounting_status === 'unavailable') return ['cost unavailable'];

    // C2/F12: ONE precedence resolver, shared with the Python seams and with
    // log_events — the deprecated alias wins a diverged pair, so the read side
    // and the write side never pick opposite winners for the same record.
    const own = accountedUpperBound(payload);
    // Compact cards show one complete amount. Prefer the subtree projection
    // when the producer has one; leaf/legacy frames still fall back to own.
    const total = accountedUpperBoundWithChildren(payload) ?? own;
    const finalKnown = payload.cost_final === true
        && payload.cost_with_children_partial !== true;
    const pendingKnown = payload.cost_final === false
        || payload.cost_with_children_partial === true
        || payload.cost_accounting_status === 'available' && !has('cost_final');
    // ONE amount (owner decisions, 2026-09-02): the accounted upper bound already
    // contains settled + reserved + unresolved (cost_projection.py), so the card
    // states that number once and lets its wording carry the openness — a ceiling
    // (`up to`) while the ledger is open, a plain amount once final. Calls with no
    // known price are not named here (owner: no separate counter); component
    // breakdowns and unmetered counts stay on Costs, Logs and task detail.
    if (total === null) return ['cost pending'];
    if (!(finalKnown || pendingKnown || total !== 0)) return [];
    const amount = `$${total.toFixed(2)}`;
    return [finalKnown ? amount : `up to ${amount}`];
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
        final: payload.cost_final === true
            && payload.cost_with_children_partial !== true,
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
    // The executor chip is cycle state like the cost projection: a recycled
    // slot must not claim the previous cycle's delegated route as its own.
    record.executorChip = null;
    // A recycled slot must not inherit the previous cycle's finalizing hold.
    record.finalizingHold = false;
    // The activity clock is cycle state too: a
    // recycled slot ('bg-consciousness', 'active') would otherwise open showing
    // the previous cycle's "updated" time.
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
 * keep the role · model identity (the id only for twins), so their routed progress body always feeds
 * the line. A frame without new activity keeps `previous`, so finishing a card
 * never blanks its last activity. Geometry is owned by the two-line CSS clamp;
 * this character ceiling is only a defensive DOM/accessibility bound.
 */
export const COLLAPSED_ACTIVITY_MAX = 240;

/**
 * The collapsed activity line is plain text: the expanded timeline renders the
 * same headline through `renderMarkdown`, so the compact projection strips that
 * renderer's marker inventory (utils.js) — fences, inline code, bold, emphasis,
 * strikethrough, headings, bullets, links, table pipes. It strips line by line
 * without the renderer's block context, so a stray pipe row or list marker the
 * timeline would show literally is dropped here too: over-stripping is the
 * accepted side of that trade, a leaked marker is not. Headings follow the
 * renderer's own rule (`joinMarkdownHeadings`: markers off, ` — ` before the
 * text under a real heading). A headline that is nothing but markers keeps its
 * source text: an empty projection would flip the reserved activity band's
 * `:empty` rules.
 */
export function plainActivityText(text = '') {
    const source = String(text || '');
    const plain = joinMarkdownHeadings(source)
        .replace(/```\w*\n([\s\S]*?)```/g, '$1')
        .replace(/(``|`)(.+?)\1/g, '$2')
        .replace(/\*\*(.+?)\*\*/g, '$1')
        .replace(/\*(.+?)\*/g, '$1')
        .replace(/~~(.+?)~~/g, '$1')
        .replace(/^- (.+)$/gm, '$1')
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
        .replace(/^\|(.+)\|$/gm, (_, row) => row.split('|').map((cell) => cell.trim()).join(' '))
        .replace(/^[\s\-:|]+$/gm, '');
    const trimmed = plain.trim();
    return trimmed || source;
}

export function boundActivityPreview(value = '') {
    const candidate = plainActivityText(value).replace(/\s+/g, ' ').trim();
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

// ---------------------------------------------------------------------------
// In-flight chat activity status (owner decisions 1A-5A; managed continuity).
// ---------------------------------------------------------------------------

// Snapshot-authoritative activity kinds: only kinds the server's activity
// snapshot actually enumerates may be deleted by /api/state hydration.
// "managed_task" rows come from the supervisor queue (PENDING/RUNNING roots),
// so their absence from a snapshot is authoritative conclusion. Typing frames
// without a kind stamp (legacy frames, subagents) stay exempt — they are
// concluded by their own final/summary frames, as before.
const SNAPSHOT_AUTHORITATIVE_KINDS = new Set(['direct_chat', 'ephemeral_decision', 'managed_task']);

/**
 * One request/apply clock for every /api/state consumer on a page. Responses
 * may finish in either order; once generation N applies, an older generation
 * can no longer mutate any projection. requestedAt stays tied to request start
 * so activity hydration keeps its WS-arrival barrier.
 */
export function createStateSnapshotSequencer(onApply, now = () => Date.now()) {
    let requestedGeneration = 0;
    let appliedGeneration = 0;
    return {
        begin() {
            return { generation: ++requestedGeneration, requestedAt: now() };
        },
        apply(request, data) {
            const generation = Number(request?.generation) || 0;
            if (!generation || generation <= appliedGeneration) return false;
            appliedGeneration = generation;
            onApply(data, request.requestedAt, generation);
            return true;
        },
        isCurrent(request) {
            return (Number(request?.generation) || 0) > appliedGeneration;
        },
    };
}

/**
 * Main-thread fan-out gate for a live WS frame.
 *
 * Main adopts a frame only when the server did NOT stamp it as a Project
 * thread AND its chat_id is not a project the client already knows. The
 * server stamp (`project_thread`, set at the message_bus broadcast choke from
 * the registry) closes the race where a fresh project's frames arrive before
 * `projectChatIds` learns the project — previously Main adopted them and
 * minted an empty "Working..." card. Frames without the stamp (main, legacy
 * missing, external transports such as Telegram) route exactly as before;
 * explicit chat_id=0 remains the internal Skill Review partition. No
 * numeric-range heuristic is involved.
 */
export function mainThreadAccepts(msg, projectChatIds) {
    if (msg && msg.project_thread) return false;
    const cid = Number(msg?.chat_id ?? 1);
    // chat_id=0 is the internal Skill Review/panel partition. An explicit zero
    // is never a Main conversation. Legacy LOG frames whose inner payload did
    // not carry chat_id are handled separately by mainLogFrameAccepts().
    // Negative ids are reserved for synthetic A2A traffic and never enter a
    // human-facing browser stream.
    if (cid <= 0) return false;
    return !(projectChatIds instanceof Set && projectChatIds.has(cid));
}

/** Main routing for the legacy LocalChatBridge log envelope. */
export function mainLogFrameAccepts(msg, projectChatIds) {
    const data = msg?.data;
    if (data && typeof data === 'object' && Object.prototype.hasOwnProperty.call(data, 'chat_id')) {
        return mainThreadAccepts({ ...data, ...msg, chat_id: data.chat_id }, projectChatIds);
    }
    // Older bridges stamped absent inner identity as outer zero. This is the
    // one compatibility case; a real inner zero above remains panel-only.
    if (Number(msg?.chat_id) === 0) return !msg?.project_thread;
    return mainThreadAccepts(msg, projectChatIds);
}

/** Route one ordinary Chat frame to the current Main or Project instance. */
export function chatThreadAccepts(msg, isMain, chatId, projectChatIds) {
    if (isMain) return mainThreadAccepts(msg, projectChatIds);
    return Number(msg?.chat_id ?? 1) === chatId;
}

/**
 * Route one LocalChatBridge log envelope to the current Chat instance while
 * keeping an explicit inner chat_id=0 in the hidden panel partition.
 */
export function chatLogThreadAccepts(msg, isMain, chatId, projectChatIds) {
    if (isMain) return mainLogFrameAccepts(msg, projectChatIds);
    const data = msg?.data;
    if (data && typeof data === 'object' && Object.prototype.hasOwnProperty.call(data, 'chat_id')) {
        return chatThreadAccepts({ ...msg, ...data, chat_id: data.chat_id }, false, chatId, projectChatIds);
    }
    // An absent inner identity historically arrives as outer zero. Project
    // instances do not adopt that unowned compatibility frame.
    if (Number(msg?.chat_id) === 0) return false;
    return chatThreadAccepts(msg, false, chatId, projectChatIds);
}

const TERMINAL_TASK_STATUSES = new Set([
    'completed', 'failed', 'cancelled', 'rejected_duplicate',
]);
const TERMINAL_SUBAGENT_EVENTS = new Set([
    'completed', 'completed_warn', 'failed', 'cancelled', 'rejected',
]);

/**
 * Positive typed task-terminal truth shared by history and live Chat rows.
 * Role + task_id is deliberately insufficient: review references, lifecycle
 * receipts, annotations and media can all carry a real task id mid-run.
 */
export function positiveTaskTerminalFact(row) {
    if (!row || typeof row !== 'object') return false;
    if (String(row.system_type || '') === 'task_summary') return true;
    if (TERMINAL_TASK_STATUSES.has(String(row.task_terminal_status || '').toLowerCase())) return true;
    return String(row.delegation_role || '').toLowerCase() === 'subagent'
        && TERMINAL_SUBAGENT_EVENTS.has(String(row.subagent_event || '').toLowerCase());
}

/**
 * Single status reducer for the chat header (owner decisions 2A/5A; managed
 * activities added by the project-continuity contract). Priority: disconnected
 * > background live card (Working...) > admitted managed work (Working...) >
 * server-confirmed direct/ephemeral turns (Thinking...) > local pending
 * submissions (Sending...) > queue-admitted but unstarted managed work
 * (Queued...) > idle. A queued task ranks below
 * Sending... because an unacknowledged local submission is the more actionable
 * state. Pure over its inputs for dependency-free node tests.
 */
export function computeDerivedChatStatus({
    isConnected = true,
    hasActiveLiveCard = false,
    activeDirectCount = 0,
    activeManagedCount = 0,
    queuedManagedCount = 0,
    pausedManagedCount = 0,
    pendingSubmissionsCount = 0,
} = {}) {
    if (!isConnected) {
        return { kind: 'offline', text: 'Reconnecting...', showDots: false };
    }
    if (hasActiveLiveCard) {
        return { kind: 'thinking', text: 'Working...', showDots: false };
    }
    if (activeManagedCount > 0) {
        return { kind: 'thinking', text: 'Working...', showDots: true };
    }
    if (activeDirectCount > 0) {
        return { kind: 'thinking', text: 'Thinking...', showDots: true };
    }
    if (pendingSubmissionsCount > 0) {
        return { kind: 'thinking', text: 'Sending...', showDots: true };
    }
    if (queuedManagedCount > 0) {
        return { kind: 'thinking', text: 'Queued...', showDots: true };
    }
    if (pausedManagedCount > 0) {
        // Budget-paused work is NOT running and will not start by itself:
        // never dress it up as Working or Queued.
        return { kind: 'online', text: 'Paused (budget)', showDots: false };
    }
    return { kind: 'online', text: 'Online', showDots: false };
}

/**
 * Local-echo continuity: split the bounded journal of locally-sent owner rows
 * against ONE fetched history response. Entries whose client_message_id
 * appears in the response are CONFIRMED durable (server history is the
 * authority; the local copy retires). The rest are UNCONFIRMED and must
 * survive a full feed rebuild: a stale history snapshot — fetched before the
 * send was logged — has no authority to erase a message the owner just sent.
 * Pure over its inputs for dependency-free node tests.
 */
export function partitionLocalEchoJournal(journal, serverClientMessageIds) {
    const confirmed = [];
    const unconfirmed = [];
    const entries = journal instanceof Map ? journal.values() : (journal || []);
    for (const entry of entries) {
        const cmid = String(entry?.clientMessageId || '');
        if (!cmid) continue;
        if (serverClientMessageIds && serverClientMessageIds.has(cmid)) confirmed.push(entry);
        else unconfirmed.push(entry);
    }
    return { confirmed, unconfirmed };
}

// ---------------------------------------------------------------------------
// Pure message-presentation helpers (moved verbatim from chat.js — that
// module sits at its byte ceiling).
// ---------------------------------------------------------------------------

/** Dedupe key for one rendered chat row; client_message_id wins when present. */
export function buildMessageKey(role, text, timestamp, opts = {}) {
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

export function reconnectBannerText(reason = '') {
    if (reason === 'sha-change') return '♻️ Restart complete';
    if (reason) return '♻️ Reconnected';
    return '';
}

/** {short, full} presentation of a message timestamp, or null when unreadable. */
export function formatMsgTime(isoStr) {
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

/** Human label for ONE manual-routing option row (shared with the picker card). */
export function routingOptionLabel(option) {
    if (!option || typeof option !== 'object') return '';
    if (option.label) return String(option.label);
    if (option.action === 'new_task_in_project') {
        return `New task in ${String(option.project_name || 'Project')}`;
    }
    if (option.title || option.project_name) {
        return String(option.title || option.project_name);
    }
    return option.project_id && !option.task_id ? 'Project' : 'Task';
}

/** Human text for a typed routing annotation ('' hides the line). */
export function routingAnnotationText(annotation) {
    if (!annotation || typeof annotation !== 'object') return '';
    const action = String(annotation.action || '');
    const status = String(annotation.status || '');
    const target = String(annotation.target || '');
    const targetLabel = String(annotation.target_label || '')
        || (target ? (action === 'project_route' ? 'Project' : 'Task') : '');
    if (status === 'pending') return 'Choosing the right destination…';
    if (status === 'needs_manual_target') {
        const optionLabels = (Array.isArray(annotation.options) ? annotation.options : [])
            .map(routingOptionLabel)
            .filter(Boolean);
        if (optionLabels.length) return `Choose a target · ${optionLabels.join(' / ')}`;
        return targetLabel ? `Choose a target · ${targetLabel}` : 'Choose a target';
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
    return targetLabel && label ? `${label} · ${targetLabel}` : label;
}

/**
 * Reconcile the client's active-activity map against one /api/state snapshot
 * (owner decision 1A). The snapshot is authoritative ONLY over kinds it
 * enumerates (direct/ephemeral registry turns and queue-listed managed roots)
 * that existed before it was requested; it must never delete (a) an activity
 * registered by a WS typing frame AFTER the request started (the barrier), or
 * (b) a kind-less typing entry (legacy frames, subagents), which no snapshot
 * source tracks.
 *
 * `concludedIds` (Set/Map with .has) is the client-side conclusion ledger: a
 * turn already concluded by its keyed final must never be re-inserted by a
 * snapshot that was captured while it still ran (activity ids are unique task
 * ids and never restart, so conclusion is final). Without this, a one-shot
 * hydration (project panels) could resurrect a finished turn indefinitely.
 */
export function computeHydratedDirectActivities(
    existingMap,
    turnsList,
    chatId,
    snapshotBarrierMs = Infinity,
    concludedIds = null,
    snapshotGeneration = 0,
) {
    const nextMap = new Map(existingMap || []);
    if (!Array.isArray(turnsList)) return nextMap;
    const currentSnapshotGeneration = Number(snapshotGeneration) || 0;
    const currentChatTurns = turnsList.filter((t) => Number(t?.chat_id ?? 1) === chatId);
    const activeIdsInSnapshot = new Set();
    for (const turn of currentChatTurns) {
        const aid = String(turn?.activity_id || '').trim();
        if (!aid) continue;
        if (concludedIds && concludedIds.has(aid)) continue;
        activeIdsInSnapshot.add(aid);
        const hadExisting = nextMap.has(aid);
        const existing = nextMap.get(aid) || {};
        const hydrated = {
            activityId: aid,
            kind: turn.kind || 'direct_chat',
            phase: turn.phase || 'thinking',
            clientMessageId: turn.client_message_id || existing.clientMessageId || '',
            // Strictly CLIENT-clock "first observed" time: the snapshot's
            // server-clock started_at must never enter the barrier comparison
            // below (clock skew would let finished activities linger).
            startedAt: existing.startedAt || Date.now(),
        };
        // Snapshot-only rows carry HTTP provenance. A live frame overwrites
        // the row without this marker and keeps request-start barrier authority.
        if (existing.snapshotGeneration || (!hadExisting && currentSnapshotGeneration)) {
            hydrated.snapshotGeneration = currentSnapshotGeneration || existing.snapshotGeneration;
        }
        nextMap.set(aid, hydrated);
    }
    for (const [aid, entry] of nextMap.entries()) {
        if (activeIdsInSnapshot.has(aid)) continue;
        // Deletion authority is scoped to snapshot-enumerated kinds: a
        // kind-less typing entry is invisible to every snapshot source and is
        // concluded by its own final/summary frame instead.
        if (!SNAPSHOT_AUTHORITATIVE_KINDS.has(String(entry?.kind || ''))) continue;
        const entrySnapshotGeneration = Number(entry?.snapshotGeneration) || 0;
        if (
            currentSnapshotGeneration
            && entrySnapshotGeneration
            && entrySnapshotGeneration < currentSnapshotGeneration
        ) {
            nextMap.delete(aid);
            continue;
        }
        const startedAt = Number(entry?.startedAt) || 0;
        if (startedAt >= snapshotBarrierMs) continue;
        nextMap.delete(aid);
    }
    return nextMap;
}

/**
 * Hydrate one authoritative activity snapshot and identify the narrower event
 * that can wake durable task-detail convergence: a host-stamped managed root
 * observed before this request, now absent from the GLOBAL snapshot. A root
 * still listed under another chat merely departed locally. Direct/ephemeral
 * removals carry no task-detail/card authority, but they ARE conclusions
 * (#369): the caller records them so a late frame cannot resurrect the
 * turn, and clears the linked Sending... submission.
 */
export function reconcileHydratedDirectActivities(
    existingMap,
    turnsList,
    chatId,
    snapshotBarrierMs = Infinity,
    concludedIds = null,
    snapshotGeneration = 0,
) {
    const activities = computeHydratedDirectActivities(
        existingMap, turnsList, chatId, snapshotBarrierMs, concludedIds, snapshotGeneration,
    );
    const globallyActiveActivityIds = new Set();
    for (const turn of Array.isArray(turnsList) ? turnsList : []) {
        const activityId = String(turn?.activity_id || '').trim();
        if (activityId) globallyActiveActivityIds.add(activityId);
    }
    const departedManagedTaskIds = [];
    const disappearedManagedTaskIds = [];
    const concludedDirectActivities = [];
    for (const [activityId, entry] of existingMap || []) {
        if (activities.has(activityId)) continue;
        if (concludedIds?.has(activityId)) continue;
        if (String(entry?.kind || '') !== 'managed_task') {
            // A direct/ephemeral row the authoritative snapshot no longer
            // lists is settled: its live final was missed (ephemeral
            // task_done frames never reach the card layer), so the snapshot
            // is the conclusion of record.
            if (!globallyActiveActivityIds.has(activityId)) {
                concludedDirectActivities.push({
                    activityId,
                    clientMessageId: String(entry?.clientMessageId || ''),
                });
            }
            continue;
        }
        departedManagedTaskIds.push(activityId);
        if (globallyActiveActivityIds.has(activityId)) continue;
        disappearedManagedTaskIds.push(activityId);
    }
    return {
        activities,
        departedManagedTaskIds,
        disappearedManagedTaskIds,
        concludedDirectActivities,
        globallyActiveActivityIds,
    };
}

/**
 * Card-set durable-truth reconcile (stuck "Working..." pill class). The header
 * reducer reads hasActiveLiveCard — a pure DOM scan of mounted unfinished
 * foreground cards — so a card minted by a replayed frame whose own terminal
 * row never reached this client (lost task_done, lineage-only subagent final
 * re-minting a finished parent) kept the pill on "Working..." forever: every
 * existing terminal path keys on the card's OWN id reaching a snapshot or
 * frame first. This selector closes the gap from the card side: given the
 * compact card projection `{id, finished, isSubagent, connected}` and the set
 * of ids the GLOBAL /api/state snapshot confirms live, it returns the mounted
 * unfinished foreground card ids the snapshot does NOT vouch for. Each one is
 * handed to observeMissingManagedTask, whose durable task-detail read finishes
 * the card ONLY on a proven terminal status (`log_events.js::isTerminalTaskDetail`); a 404 or
 * nonterminal detail keeps the id and retries on the next snapshot (owner
 * Q3=A: no timers, no id-shape heuristics, no fabricated terminal).
 *
 * Skipped here: finished cards, detached roots (not part of the reducer's
 * scan), subagent cards (their parent owns the lineage; observe filters them
 * too), reusable slots ('bg-consciousness', 'active' — many cycles per id, no
 * single durable result) and the 'chat' fallback group id. Pure for node tests.
 */
export function unconfirmedForegroundCardIds(cards, activeIds) {
    const out = [];
    for (const card of Array.isArray(cards) ? cards : []) {
        const id = String(card?.id || '');
        if (!id || id === 'chat' || REUSABLE_TASK_IDS.has(id)) continue;
        if (card.finished || card.isSubagent || !card.connected) continue;
        if (activeIds?.has(id)) continue;
        out.push(id);
    }
    return out;
}

// Extracted from chat.js (byte-ratchet payment): the DOM half of the routing
// acknowledgement, kept beside its text builder above.
export function renderRoutingAnnotation(bubble, annotation) {
    if (!bubble) return false;
    const text = routingAnnotationText(annotation);
    let note = bubble.querySelector('.msg-routing-annotation');
    if (!text) {
        const hasStatus = bubble.dataset.chatAnnotationStatus !== undefined;
        if (!note && !hasStatus) return false;
        note?.remove();
        if (hasStatus) delete bubble.dataset.chatAnnotationStatus;
        return true;
    }
    const status = String(annotation.status || '');
    const changed = !note || note.textContent !== text
        || note.dataset.annotationStatus !== status
        || bubble.dataset.chatAnnotationStatus !== status;
    if (!note) {
        note = document.createElement('div');
        note.className = 'msg-routing-annotation';
        const time = bubble.querySelector('.msg-time');
        if (time) time.before(note);
        else bubble.append(note);
    }
    if (note.textContent !== text) note.textContent = text;
    if (note.dataset.annotationStatus !== status) note.dataset.annotationStatus = status;
    if (bubble.dataset.chatAnnotationStatus !== status) bubble.dataset.chatAnnotationStatus = status;
    return changed;
}

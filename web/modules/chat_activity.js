// Pure chat-activity helpers shared by chat.js and dependency-free node tests:
// live-card presentation projections (moved verbatim from chat.js) plus the
// census snapshot hydration. The room's status word is the server's (work_status.js).
import { META_PART_SEPARATOR, executorIdentityMarkup, executorModelText, formatCoordinatorProvenance, harnessPresentation } from './harness_presentation.js';
import { effortChipMarkup } from './effort_chip.js';
import { resultFilesItemHtml } from './result_files.js';
import { taskSourceDownloadUrl, apiFetch } from './api_client.js';
import { compactModel, formatLogDuration, modelExecutionView } from './log_events.js';
import { connectionLabel, routeConnectionId } from './connection_labels.js';
import { fitText, fullText } from './text_fit.js';
import { delegatedActivityBodyHtml, delegatedHeadline, delegatedLineView } from './delegated_activity.js';
import { REUSABLE_TASK_IDS } from './task_control_menu.js';
import { ownLiveDetail, paintLiveDetail } from './task_phase_chip.js';
import { fmt, tr, tx } from './i18n.js';
import { formatMsgTime } from './utils.js';
import { hostRowDisplayText } from './host_notice_text.js';
export { bootPhase, startingStatus, supervisorReady } from './host_starting_status.js';
import {
    projectMarkdownLines, MARKDOWN_FENCED_CODE, accountedUpperBound,
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
    if (payload?.model_execution && typeof payload.model_execution === 'object') {
        out.modelExecution = payload.model_execution;
    }
    if (Array.isArray(out.meta) && out.meta.length) {
        out.meta = out.meta.filter((entry) => !String(entry || '').startsWith('cost='));
    }
    return out;
}

export function applyHistoricalModelExecution(record, historical) {
    if (!record || record.modelExecution || !historical?.model_execution) return false;
    record.modelExecution = historical.model_execution;
    return true;
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
        if (systemType === 'task_summary') return 'Task Summary';
        if (systemType === 'skill_review') return 'Skill Review';
        return 'System';
    }
    if (isProgress) return 'Thought';
    // A self-initiated turn (a consciousness wake-up) signs its final bubble
    // through the same sender line the user bubble uses for its source.
    if (opts.initiator === 'consciousness') return 'Ouroboros · Consciousness';
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

// A late-review row links its exact applied review record through the task artifact
// route (#1369) — an absent or unsupported `late_evidence.source_ref` offers no link
// rather than a guessed one. The card's timeline and a card-less System row share it.
export function cardRowEvidenceRef(msg) {
    const evidence = msg?.late_evidence && typeof msg.late_evidence === 'object'
        ? taskSourceDownloadUrl(String(msg.task_id || '').trim(), msg.late_evidence.source_ref) : '';
    return evidence ? { href: evidence, label: 'Download the review record' } : null;
}

// The stored record link as one download anchor, or nothing. A value restored from
// the session snapshot is held to the task artifact route it was minted on. It wears
// the chat link ink (`md-link`), as the result-file downloads beside it do.
export function evidenceLinkHtml(evidenceRef) {
    const href = typeof evidenceRef?.href === 'string' && evidenceRef.href.startsWith('/api/tasks/') ? evidenceRef.href : '';
    return href
        ? `<p class="chat-live-line-evidence"><a class="md-link" href="${escapeHtmlAttr(href)}" download data-live-line-evidence>${escapeHtml(evidenceRef.label || 'Download the review record')}</a></p>`
        : '';
}

// A placed row's first line heads; card_row_id keeps identity, and late reviews
// retain their record link. Only typed host pause copy enters the translator.
export function cardRowSummary(msg, phase, rawTs = '') {
    const text = String(msg.text ?? msg.content ?? '');
    const lines = hostRowDisplayText(msg, msg.role === 'system' && msg.system_type === 'task_pause_notice' ? tx(text) : text).split('\n');
    const rowId = String(msg.card_row_id || '').trim() || `${String(msg.system_type || '').trim()}|${rawTs}`;
    return {
        phase, headline: lines[0].trim(), body: lines.slice(1).join('\n').trim(), dedupeKey: `cardrow|${rowId}`,
        cardRowRevision: msg.card_row_revision, evidenceRef: cardRowEvidenceRef(msg),
    };
}

export function buildTimelineItemHtml(item, record) {
    if (item.resultArtifacts) return resultFilesItemHtml(item);
    // A delegated observation renders its per-seq projection; one wholly shown by
    // an earlier row, or folded into a silent stretch, keeps only its keyed slot.
    const delegated = item.activity ? delegatedLineView(item) : null;
    if (delegated?.hidden) return `<div class="chat-live-line" data-live-line-key="${escapeHtmlAttr(item.lineKey || '')}" data-delegated-folded hidden></div>`;
    const expandable = Boolean(delegated) || isLiveLineExpandable(item);
    const expanded = expandable && record.expandedLineKeys.has(item.lineKey);
    const displayHeadline = delegated ? delegatedHeadline(delegated)
        : expanded && item.fullHeadline ? item.fullHeadline : item.headline;
    // P3: when expanded, prefer the genuinely-full fetched output, then the capped
    // fullBody, then the preview body. A server-truncated line shows the fetched full
    // text in a bounded-scroll box so a huge research output never grows the chat.
    const displayBody = expanded ? (item.fullReadError || item.fetchedFull || item.fullBody || item.body) : item.body;
    const showingFetched = expanded && Boolean(item.fetchedFull);
    const loadingFull = expanded && Boolean(item.truncated && item.fullRef && !item.fetchedFull && !item.fullReadError);
    // A late-review row offers its exact applied review record (`cardRowSummary`).
    const evidenceHtml = evidenceLinkHtml(item.evidenceRef);
    const bodyHtml = !displayBody ? '' : item.bodyFormat === 'text'
        ? escapeHtml(displayBody) : renderMarkdown(displayBody, { inlineHeadingBreaks: true });
    const isProgressLine = item.phase === 'working' || item.phase === 'thinking';
    const bodyId = `chat-live-line-body-${String(record.groupId || 'task').replace(/[^A-Za-z0-9_-]/g, '-')}-${String(item.lineKey || '').replace(/[^A-Za-z0-9_-]/g, '-')}`;
    const headContent = `
        <span class="chat-live-line-title"${isProgressLine && !delegated ? ' data-chat-markdown-enhanced' : ''}>${isProgressLine && !delegated ? renderMarkdown(displayHeadline, { inlineHeadingBreaks: true }) : escapeHtml(displayHeadline)}</span>
        <span class="chat-live-line-repeat" ${item.count > 1 ? '' : 'hidden'}>${item.count > 1 ? `${item.count}x` : ''}</span>
        ${item.ts ? `<span class="chat-live-line-time">${escapeHtml(item.ts)}</span>` : ''}
    `;
    const headHtml = expandable
        ? `
            <div
                role="button" tabindex="0"
                class="chat-live-line-toggle"
                data-live-line-toggle="${escapeHtmlAttr(item.lineKey)}"
                aria-expanded="${expanded ? 'true' : 'false'}"
                ${displayBody ? `aria-controls="${escapeHtmlAttr(bodyId)}"` : ''}
            >
                <span class="chat-live-line-head">${headContent}</span>
                <span class="chat-live-line-expand-label">${expanded ? 'Collapse' : ((item.truncated && item.fullRef) ? 'Show full' : 'Expand')}</span>
            </div>
        `
        : `<div class="chat-live-line-head">${headContent}</div>`;
    return `
        <div
            class="chat-live-line ${item.phase || 'working'}${expandable ? ' expandable' : ''}"
            data-live-line-key="${escapeHtmlAttr(item.lineKey || '')}"
            ${item.historyId ? `data-history-id="${escapeHtmlAttr(item.historyId)}"`
        : item.sourceHistoryId ? `data-source-history-id="${escapeHtmlAttr(item.sourceHistoryId)}"` : ''}
            data-expanded="${expanded ? '1' : '0'}"
        >
            ${headHtml}
            ${delegated ? `<div class="chat-live-line-body chat-delegated-activity" id="${escapeHtmlAttr(bodyId)}">${delegatedActivityBodyHtml(delegated, { expanded })}</div>`
        : displayBody || evidenceHtml ? `<div class="chat-live-line-body${showingFetched ? ' chat-live-line-body-full' : ''}"${item.bodyFormat === 'text' ? ' data-body-format="text"' : ''} id="${escapeHtmlAttr(bodyId)}">${bodyHtml}${evidenceHtml}${loadingFull ? '<div class="chat-live-line-loading">Loading full output…</div>' : ''}</div>` : ''}
        </div>
    `;
}

// ---------------------------------------------------------------------------
// Folded tool evidence (docs/DESIGN.md "Conversation activity block").
// ---------------------------------------------------------------------------

/**
 * A turn's routine execution is ONE row, whatever a burst costs in calls, so
 * the narration around it stays readable. The accumulator is a state map keyed
 * by invocation rather than a pair of counters: duplicate, reordered and
 * concurrent frames all settle on the same totals, and the host's own numbers
 * replace them without either side double counting.
 */
function ensureToolFold(record) {
    if (!record.toolFold) record.toolFold = { calls: new Map(), host: null };
    return record.toolFold;
}

/**
 * One frame's fact about one invocation: {key, status, receipt, tool}. Status
 * preserves independent start/wait/settlement facts. Reordered starts never reopen
 * a settled call; a true result replaces only a provisional host-error settlement.
 */
export function noteToolCall(record, observation) {
    const key = observation?.key;
    if (!record || !key) return record;
    const { calls } = ensureToolFold(record);
    const prev = calls.get(key);
    const fact = observation.fact || (['ok', 'error'].includes(observation.status) ? 'settled' : 'started');
    const next = { ...prev,
        receipt: prev?.receipt ?? Boolean(observation.receipt),
        tool: observation.tool || prev?.tool || '',
    };
    if (fact === 'settled') {
        if (!next.settlement || (next.settlement.hostError && !observation.hostError)) {
            next.settlement = { status: observation.status, hostError: Boolean(observation.hostError) };
            next.receipt = Boolean(observation.receipt);
        }
    } else if (fact === 'wait_ended') next.waitEnded = true;
    else next.started = true;
    next.live = !record.finished && (next.live || observation.live === true || (!observation.fact && observation.status === 'calling'));
    next.status = next.settlement?.status || (next.waitEnded ? 'wait_ended' : next.live ? 'calling' : 'unknown');
    calls.set(key, next);
    return record;
}

/**
 * The host's totals for the turn, merged FIELD-WISE onto what the host already
 * stated. An ABSENT field (null/undefined) stays absent and keeps the previous
 * known value; invocation totals never decrease on replay. A partial snapshot carrying `tool_calls` must not read
 * as "no addressing calls" and turn a block that only addressed work into
 * content, and it must not erase an error or routing count a complete snapshot
 * already gave. `counts` is known only as a NON-EMPTY object, so an empty or
 * absent `tool_call_counts` keeps the live map's names and the row behind Expand
 * is never explicitly emptied while the turn counts calls.
 */
export function noteToolHostMetrics(record, host) {
    for (const observation of host?.evidence?.observations || []) applyToolObservation(record, observation);
    const fold = ensureToolFold(record);
    if (host?.evidence?.coverage) { fold.coverage = host.evidence.coverage; fold.legacy = host.evidence.legacy; }
    if (record.finished) for (const call of fold.calls.values()) {
        call.live = false;
        call.status = call.settlement?.status || (call.waitEnded ? 'wait_ended' : 'unknown');
    }
    const known = fold.host || {};
    const carry = (next, before) => (next === null || next === undefined ? (before ?? null) : next);
    const counts = host?.counts && typeof host.counts === 'object' && Object.keys(host.counts).length > 0
        ? host.counts : (known.counts ?? null);
    fold.host = {
        calls: host?.calls == null ? (known.calls ?? null) : Math.max(host.calls, known.calls ?? 0),
        errors: carry(host?.errors, known.errors),
        routing: host?.routing == null ? (known.routing ?? null) : Math.max(host.routing, known.routing ?? 0),
        completion: host?.completion == null ? (known.completion ?? null) : Math.max(host.completion, known.completion ?? 0),
        counts,
    };
    return toolEvidenceView(record.toolFold);
}

/**
 * One frame about one invocation, applied to the block's fold: the map first,
 * then the row it owns, rebuilt from the map and the host's totals together.
 * The meta counts follow the same reading, so the header never disagrees with
 * the row while a turn runs.
 */
export function applyToolObservation(record, observation) {
    noteToolCall(record, observation);
    const view = toolEvidenceView(record.toolFold);
    record.toolCalls = view.calls;
    record.toolErrors = view.errors;
    // Successful settlement retires an earlier provisional error/wait notice;
    // the fold retains the independent wait fact, including after task terminal.
    if (record.toolFold.calls.get(observation.key)?.settlement?.status === 'ok' && record.items) {
        const count = record.items.length;
        record.items = record.items.filter(item => item.dedupeKey !== observation.key);
        view.clearedNotice = count !== record.items.length;
    }
    return view;
}

const perToolLine = (entries) => entries
    .filter(([name, n]) => name && n > 0)
    .map(([name, n]) => (n > 1 ? `${name} ×${n}` : name)).join(' · ');

export function toolEvidenceIncomplete(coverage) {
    return Boolean(coverage && (coverage.gaps?.length || coverage.matched > coverage.shown
        || coverage.source && !Number.isFinite(coverage.live_size)));
}

// One tool row combines host counts with invocation facts; names and read scope
// live behind Expand. A bounded read is not an unknown invocation outcome.
export function toolEvidenceView(fold = null) {
    const live = fold?.calls instanceof Map ? [...fold.calls.values()] : [];
    const host = fold?.host || null;
    const observed = live.length + (fold?.legacy?.calls || 0);
    const calls = Math.max(Number.isInteger(host?.calls) ? host.calls : 0, observed);
    // Frozen totals count model wait errors. Canonical evidence reports operation
    // outcomes; a bounded partial read discloses its gap instead of reviving waits.
    const partial = toolEvidenceIncomplete(fold?.coverage) || Boolean(fold?.coverage) && observed < calls;
    const bounded = fold?.coverage && (fold.coverage.archives_bounded || fold.coverage.archives_available > fold.coverage.archives
        || fold.coverage.live_size > fold.coverage.live_window);
    // Only settled, individually identified calls can supersede an aggregate
    // host error. Partial replay and legacy start-only rows have no such proof.
    const outcomesKnown = calls > 0 && live.length === calls && !(fold?.legacy?.calls)
        && live.every(call => call.settlement);
    const observedErrors = live.filter(call => call.status === 'error').length + (fold?.legacy?.errors || 0);
    const errors = outcomesKnown ? observedErrors
        : Math.max(Number.isInteger(host?.errors) ? host.errors : 0, observedErrors);
    const liveCounts = new Map();
    for (const call of live) liveCounts.set(call.tool, (liveCounts.get(call.tool) || 0) + 1);
    const headline = !calls && partial ? tr('task.tools.history_incomplete', 'Tool history incomplete')
        : fmt(calls === 1 ? '{n} tool call' : '{n} tool calls', { n: calls });
    return {
        phase: errors > 0 ? 'warn'
            : ((!host && live.some((call) => call.status === 'calling')) ? 'calling' : 'result'),
        headline: headline + (errors > 0 ? ` · ${fmt(errors === 1 ? '{n} error' : '{n} errors', { n: errors })}` : '')
            + (live.some(call => call.waitEnded) || fold?.legacy?.wait_ended ? ` · ${tr('task.tools.wait_ended', 'wait ended')}` : '')
            + (calls && (live.some(call => ['unknown', 'wait_ended'].includes(call.status)) || fold?.legacy?.unknown || fold?.coverage && observed < calls)
                ? ` · ${tr('task.tools.outcome_unknown', 'outcome unknown')}` : ''),
        body: '',
        fullBody: (partial || bounded ? (partial ? tr('task.tools.incomplete', 'Invocation evidence is incomplete.')
            : tr('task.tools.bounded_window', 'Only recent tool history was read.'))
            + (fold?.coverage?.source ? ` ${fmt('Source: {source}.', { source: fold.coverage.source })}` : '') + ' ' : '')
            + perToolLine(host?.counts && typeof host.counts === 'object'
            ? Object.entries(host.counts) : [...liveCounts]),
        visible: true,
        // Host-stamped routing/completion acts are receipts, never work. Missing
        // aggregate fields do not erase complete per-invocation receipt evidence.
        receipt: calls > 0 && errors <= 0 && (Number(host?.routing || 0) + Number(host?.completion || 0) >= calls
            || !partial && live.length >= calls && live.every((call) => call.receipt)),
        calls,
        errors,
    };
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

/**
 * A history row that carries replay evidence only (a recorded quiz answer, a
 * hidden terminal projection) and mounts nothing: the one owner of that
 * distinction for the replay passes and the pager's page-row count.
 */
export function isReplayEvidenceRow(row) {
    return ['quiz_answer', 'quiz_closed', 'task_evidence'].includes(row?.system_type) || Boolean(row?.summary_kind && row?.historical_terminal);
}

export function isForegroundLiveCard(record) {
    return Boolean(
        record?.root?.isConnected && !record.finished && !record.reviewAnchor && !record.historicalUnavailable && !record.historicalUnconfirmed
    );
}

export function sendPanic({ ws, frame = { type: 'command', cmd: '/panic' }, onAccepted }) {
    const result = ws.send(frame, { onPanicAccepted: onAccepted });
    // Send first, then stop the page's own work (skills' page modules).
    globalThis.dispatchEvent?.(new CustomEvent('ouro:panic'));
    return result;
}

// A task still settling its pause (the durable row says ``pausing``): its
// checkpoint may not be saved yet, so a Restart can interrupt it.
const PAUSING_PHASES = new Set(['budget_pausing', 'pausing']);

/**
 * The truthful body of the ONE Restart confirmation (owner quiz 285597): what
 * the server's owner Restart actually does to running, paused, queued and
 * still-pausing work (supervisor/restart_retention.py). ``activities`` is the
 * live census; ``null`` means it could not be read, and the body says so
 * instead of promising there is nothing still pausing.
 * @param {Array<{phase?: string}>|null} activities
 * @returns {string}
 */
export function restartConfirmBody(activities) {
    const lines = [
        'Running tasks stop, then eligible saved work resumes after the restart. Tasks already paused stay paused.',
        'Runnable queued tasks return to the queue. Existing holds and limits still apply.',
        'Saved settings apply after the restart: work that follows a model setting resumes on the current setting; task-specific choices stay.',
    ];
    if (!Array.isArray(activities) || activities.some((row) => row?.phase === 'unknown')) {
        lines.push('Pause status could not be read: a task that is still pausing would be interrupted instead of staying paused.');
        if (!Array.isArray(activities)) return lines.join('\n');
    }
    const pausing = activities.filter((row) => PAUSING_PHASES.has(String(row?.phase || ''))).length;
    if (pausing) {
        lines.push(`${pausing} task${pausing === 1 ? ' is' : 's are'} still pausing: a pause not saved when the restart `
            + 'stops it is interrupted instead of staying paused.');
    }
    return lines.join('\n');
}

async function readLiveActivities() {
    const resp = await apiFetch('/api/state', { cache: 'no-store' });
    const data = resp?.ok ? await resp.json() : null;
    if (!Array.isArray(data?.active_chat_activities) || data.active_chat_activities_complete !== true
        || data.works_complete === false) {
        throw new Error('census unavailable');
    }
    // The frozen census lists roots. A parked root can still have a child
    // saving its Pause: Restart must warn about that member too, once per id.
    const activities = new Map(data.active_chat_activities.map((row) => [row.activity_id, row]));
    for (const work of data.works || []) {
        for (const [id, state] of Object.entries(work.members?.states || {})) {
            if (state.confirmed === false) activities.set(id, { phase: 'unknown' });
            else if (state.transition === 'pausing') activities.set(id, { phase: 'pausing' });
        }
    }
    return [...activities.values()];
}

/**
 * The ONE Restart confirm-and-send both UI Restart buttons use (the chat
 * header and Settings "Restart now"; owner quiz 285597: one shared
 * confirmation, deliberately added to the formerly immediate header button,
 * never two dialogs). Telegram `/restart` and Panic are separate commands and
 * keep their own contracts. `queue:false`: a disconnected page never queues a
 * destructive command for a later reconnect.
 */
export async function confirmAndSendRestart({ openConfirmDialog, ws, readActivities = readLiveActivities }) {
    let activities = null;
    try {
        activities = await readActivities();
    } catch {
        activities = null;
    }
    const confirmed = await openConfirmDialog({
        title: 'Restart agent',
        body: restartConfirmBody(activities),
        confirmLabel: 'Restart',
        danger: true,
    });
    if (!confirmed) return 'cancelled';
    const result = ws?.send?.({ type: 'command', cmd: '/restart' }, { queue: false });
    return result?.status === 'sent' ? 'sent' : 'not_connected';
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
    const control = target.closest('button, a, input, textarea, select, label, summary, [contenteditable="true"], [role="button"]');
    if (control && !control.matches?.('[data-live-line-toggle]')) return '';
    if (selectionInside(line, selection)) return '';
    return (line.dataset && line.dataset.liveLineKey) || '';
}

/** One listener owner survives keyed timeline patches and older-page replay. */
export function bindLiveCardTimeline(el, onActivate) {
    if (!el) return () => {};
    const onClick = (event) => {
        const lineKey = liveLineRowToggleKey(event.target, el.ownerDocument?.getSelection?.() || globalThis.getSelection?.());
        if (!lineKey) return;
        event.stopPropagation();
        onActivate(lineKey, event);
    };
    const onKeydown = (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        const header = event.target?.closest?.('[data-live-line-toggle]');
        if (!header || !el.contains(header)) return;
        const lineKey = liveLineRowToggleKey(event.target);
        if (!lineKey) return;
        event.preventDefault();
        event.stopPropagation();
        if (!event.repeat) onActivate(lineKey, event);
    };
    el.addEventListener('click', onClick);
    el.addEventListener('keydown', onKeydown);
    let released = false;
    return () => {
        if (released) return;
        released = true;
        el.removeEventListener('click', onClick);
        el.removeEventListener('keydown', onKeydown);
    };
}

/** Twins share a displayed role; their model is a separately labelled fact. */
export function subagentIdentityKey({ parentId = '', role = '' } = {}) {
    return `${parentId}\u0000${subagentIdentityTitle({ role })}`;
}

export function subagentIdentityTitle({ role = '' } = {}) {
    return String(role || '').trim() || 'Subagent';
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
 * A non-collapsed text selection touching `el`: the reader is copying,
 * not clicking, so a click-to-toggle surface must not fire (DESIGN.md §5).
 */
export function selectionInside(el, selection = globalThis.getSelection?.()) {
    if (!el || !selection || selection.isCollapsed) return false;
    if (el.contains?.(selection.anchorNode) || el.contains?.(selection.focusNode)) return true;
    // Both endpoints can be outside a header while the selected range crosses it.
    for (let i = 0; i < (selection.rangeCount || 0); i += 1) {
        if (selection.getRangeAt(i).intersectsNode(el)) return true;
    }
    return false;
}

/**
 * A click-to-toggle surface whose content stays selectable (DESIGN.md §5): a
 * pointer click whose drag left a selection inside it does nothing, and Enter /
 * Space activate it like a native button. The surface is a `div[role=button]`
 * because WebKit never lets text inside a real <button> be selected. A surface that is a
 * control only in some of its states passes `isActive`: while it says no, clicks and keys
 * pass through untouched (nothing is stopped, so document-level handlers still see them).
 */
export function bindContentButton(el, onActivate, isActive = () => true) {
    if (!el) return;
    const nestedControl = (event) => {
        const control = event.target?.closest?.('button, a, input, textarea, select, label, summary, [contenteditable="true"], [role="button"]');
        return control && control !== el;
    };
    el.addEventListener('click', (event) => {
        if (!isActive() || nestedControl(event) || (event.detail && selectionInside(el))) return;
        event.stopPropagation?.();
        onActivate(event);
    });
    el.addEventListener('keydown', (event) => {
        if (isActive() && !nestedControl(event) && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault();
            event.stopPropagation?.();
            if (!event.repeat) el.click();
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

/** Pure presentation projection used by the header and dependency-free tests. `attention` is the owner's move: a
 *  task the spending limit stopped waits for the owner (a Main attention row, main_decisions.py), else the pill is
 *  a neutral fact. */
export function headerBudgetPresentation(data) {
    const attention = Array.isArray(data?.main_attention)
        && data.main_attention.some((row) => row?.attention_reason === 'budget');
    if (!data || data.accounting_loading === true) {
        return { state: 'loading', label: 'Loading…', fillPct: 0, attention };
    }
    if (data?.accounting?.available === false) {
        return { state: 'unavailable', label: 'Unavailable', fillPct: 0, attention };
    }
    // Older state shapes did not carry accounting.available.  Keep accepting
    // them when they contain a real numeric projection, but never coerce null
    // (ledger failure in the new shape) into a convincing $0.
    const spent = optionalFiniteNumber(data.spent_usd);
    if (spent === null) {
        return { state: 'unavailable', label: 'Unavailable', fillPct: 0, attention };
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
        attention,
    };
}

/**
 * Project the producer's scoped carrier (#498) into card meta: strings, not another cost
 * authority, never conflating unknown/non-final values with a final zero. The carrier
 * answers the whole question in one fact — which scope the number describes,
 * whether a zero is EVIDENCED, whether anything in that scope is unpriced — so
 * a frame that has one never re-derives it from two half-matching fields.
 *
 * `null` means "no carrier here, use the legacy derivation below".
 */
export function costPresentationMeta(presentation) {
    if (!presentation || typeof presentation !== 'object') return null;
    if (!presentation.has_rows) return presentation.accounting_open ? ['Cost unknown'] : [];
    const amount = presentation.tracked_amount;
    if (amount === null || amount === undefined || !Number.isFinite(Number(amount))) {
        // No priced or bounded row evidenced anything: an empty ledger and a
        // ledger of exclusively unpriced calls both sum to 0.0, and neither is a
        // measured zero. The card says so instead of inventing a free result.
        return ['Cost unknown'];
    }
    const money = `$${Number(amount).toFixed(2)}`;
    if (presentation.has_unpriced) return [`Tracked: ${presentation.tracked_final ? money : `up to ${money}`}`];
    return [presentation.tracked_final ? money : `up to ${money}`];
}

/** Why a mixed subtotal reads `Tracked:`: said in the card's details, never a hover-only title. */
export const costPresentationDetail = (presentation) => (costPresentationMeta(presentation)?.[0]?.startsWith('Tracked: ')
    ? 'Cost: some steps have no price, so Tracked counts only the priced steps.' : '');

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
        'cost_accounting_status', 'cost_final', 'cost_presentation',
        'cost_usd_with_children', 'cost_with_children_partial',
        'accounted_upper_bound_usd', 'accounted_upper_bound_usd_with_children',
        'reserved_usd', 'unresolved_upper_bound_usd', 'unknown_unmetered',
    ].some(has);
    if (!hasAccountingEvidence) return [];
    if (payload.cost_accounting_status === 'unavailable') return ['cost unavailable'];
    // #498: a producer that had a readable ledger but NO same-scope facts for
    // this frame (a nested child's foreign subtree rollup) sends an explicit
    // null carrier. That amount is unknown to this card, not unreadable: the
    // owner vocabulary for an unknown amount is "Cost unknown" (DESIGN), while
    // "cost unavailable" stays reserved for a ledger that could not be read.
    // The projection below still ranks it `unavailable` so a narrower own zero
    // cannot outrank it (mergeStickyCostMeta).
    if (has('cost_presentation') && payload.cost_presentation === null) return ['Cost unknown'];

    // #498: the producer's own carrier wins, because it was built from the exact
    // ledger bucket it describes. The derivation below stays for legacy frames
    // and is deliberately conservative: it may not know a zero is unevidenced.
    const presented = costPresentationMeta(payload.cost_presentation);
    if (presented) return presented;

    // C2/F12: ONE precedence resolver, shared with the Python seams and with
    // log_events — the deprecated alias wins a diverged pair, so the read side
    // and the write side never pick opposite winners for the same record.
    const own = accountedUpperBound(payload);
    // Compact cards show one complete amount. Prefer the subtree projection
    // when the producer has one; leaf/legacy frames still fall back to own.
    const hasSubtree = has('accounted_upper_bound_usd_with_children') || has('cost_usd_with_children');
    const total = hasSubtree ? accountedUpperBoundWithChildren(payload) : own;
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
    if (total === null) return ['Cost unknown'];
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
    const detail = costPresentationDetail(payload.cost_presentation);
    const unavailable = payload.cost_accounting_status === 'unavailable' || payload.cost_presentation === null;
    const presentation = payload.cost_presentation;
    const legacyRollup = payload.cost_presentation === null && (
        payload.accounted_upper_bound_usd_with_children !== undefined || payload.cost_usd_with_children !== undefined);
    return {
        meta,
        ...(detail ? { detail } : {}),
        ts: rawTimestampEpoch(rawTs),
        ...(presentation?.scope ? { scope: presentation.scope } : legacyRollup ? { scope: 'rollup' } : {}),
        // Only a SETTLED ledger value is final. "unavailable" is an honest
        // unknown, not a settled truth: marking it final let one transient
        // ledger-read failure outrank every later real reading. The scoped
        // carrier answers this for its own scope when the frame has one (#498).
        final: presentation && typeof presentation === 'object'
            ? !unavailable && presentation.tracked_final === true
                && presentation.has_unpriced === false && presentation.accounting_open === false
                && payload.cost_final !== false && payload.cost_with_children_partial !== true
            : !unavailable && !meta.includes('Cost unknown') && payload.cost_final === true
                && payload.cost_with_children_partial !== true && !(Number(payload.unknown_unmetered) > 0)
                && !(Number(payload.non_final_rows) > 0) && !payload.ledger_integrity_degraded,
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
    if (previous.scope === 'root_tree' && next.scope !== 'root_tree') return previous;
    if (next.scope === 'root_tree' && previous.scope !== 'root_tree') return next;
    if (previous.scope === 'rollup' && next.scope === 'own') return previous;
    if (next.scope === 'rollup' && previous.scope === 'own') return next;
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
    record.collapsedActivityLead = '';
    record.causeHeld = false;  // an outcome's cause owns the line from then on; equal narration words never take it
    record.costMeta = null;
    // The executor chip is cycle state like the cost projection: a recycled
    // slot must not claim the previous cycle's delegated route as its own.
    record.executorChip = null;
    // A recycled slot must not inherit the previous cycle's finalizing hold —
    // nor the outcome observed under it (#1110), which would otherwise paint the
    // new cycle's chip with the old cycle's Failed.
    Object.assign(record, { finalizingHold: false, censusPhase: '', observedOutcome: '', workView: null });
    // The activity clock is cycle state too: a recycled slot ('active') would
    // otherwise open showing the previous cycle's "updated" time.
    record.latestActivityTs = '';
    record.latestActivityRaw = '';
    record.startedAt = 0;
    if (record.activityEl) {
        fitText(record.activityEl, '', 0);  // also drops a portal line's word-cut fit
        record.activityEl.removeAttribute('title');
        if (record.activityEl.dataset) delete record.activityEl.dataset.lead;
    }
    record.modelExecution = null;
    record.effort = null;
    record.toolCalls = null;
    record.rounds = null;
    record.toolErrors = null;
    // The folded evidence is cycle state too: a recycled slot must not count
    // the previous cycle's invocations.
    record.toolFold = null;
    record.waitFolds = null;
    record.durationSec = null;
    record.historicalUnavailable = false;
    record.historicalUnconfirmed = false;
    record.historicalTerminal = null;
    record.lastLiveObservedAt = 0;
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
 * accepted side of that trade, a leaked marker is not. The line-anchored markers
 * (headings with ` — ` before the text under a real heading, bullets, table rows)
 * go through `projectMarkdownLines`, the one projection `describeText` also runs
 * before a preview loses its newlines. A headline that is nothing but markers keeps
 * its source text: an empty projection would flip the reserved activity band's
 * `:empty` rules.
 */
export function plainActivityText(text = '') {
    const source = String(text || '');
    const plain = projectMarkdownLines(source)
        .replace(MARKDOWN_FENCED_CODE, '$1')
        .replace(/(``|`)(.+?)\1/g, '$2')
        .replace(/\*\*(.+?)\*\*/g, '$1')
        .replace(/\*(.+?)\*/g, '$1')
        .replace(/~~(.+?)~~/g, '$1')
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1');
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
    if (!isSubagent && candidate === boundActivityPreview(suggestedName || headline)) return '';
    return candidate;
}

// v6.82 (P5): terminal card phases. 'cancelled' is a first-class terminal phase
// so a force-cancelled root resolves its card instead of re-inflating.
export function isTerminalTaskPhase(phase = '', terminal = false) {
    return Boolean(terminal) || ['done', 'lifecycle_error', 'cancelled'].includes(phase);
}

// In-flight chat activity status (owner decisions 1A-5A; managed continuity).

/**
 * One request/apply clock for every /api/state consumer on a page. Responses
 * may finish in either order; once generation N applies, an older generation
 * can no longer mutate any projection. requestedAt stays tied to request start
 * and is the barrier for the CARD scan (`lastLiveObservedAt`) only — activity
 * hydration is a plain projection of the census and has no barrier.
 *
 * `gate(force)` is the page-wide single-flight admission for the readers: it
 * resolves to a request when the caller may read now. A periodic tick that
 * lands while a read is in flight is never queued: it resolves to null once
 * that read settles, so a caller that only needs some fresh read to have
 * landed (the boot prefetch before the socket opens) may await it. A forced
 * caller that lands mid-flight is coalesced with every other forced caller
 * into ONE follow-up read that starts when the in-flight read settles — the
 * first forced caller receives that request, the others resolve to null once
 * the follow-up has applied or failed. Gated requests settle via `apply`/`fail` even if a callback throws.
 * Only `fail` fans out `onUnavailable`; socket changes and confirmed mutations `retire()` older bodies.
 *
 * Each request names its `path`: once a full `task_bindings` map arrived with its revision, the
 * next read asks `bindings_since=<revision>`, and a body answering `task_bindings_unchanged` is
 * filled with that held map for that exact revision only; any other answer drops the held map,
 * so the read after it asks for the whole map again.
 */
export function createStateSnapshotSequencer(onApply, now = () => Date.now(), onUnavailable = () => {}) {
    let requestedGeneration = 0;
    let appliedGeneration = 0;
    // Newest applied body until a failed read or retire() clears it (late-mount seed).
    let latest = null;
    let held = null;  // {revision, bindings}: the last full task_bindings map this page applied
    let inflight = null;
    let settled = null;
    let followUp = null;
    const deferred = () => { let resolve; const promise = new Promise((r) => { resolve = r; }); return { promise, resolve }; };
    const begin = () => ({ generation: ++requestedGeneration, requestedAt: now(),
        path: held ? `/api/state?bindings_since=${encodeURIComponent(held.revision)}` : '/api/state' });
    const withBindings = (data) => {
        if (!data || typeof data !== 'object') return data;
        const revision = typeof data.task_bindings_revision === 'string' ? data.task_bindings_revision : '';
        if (data.task_bindings_unchanged === true && held && revision === held.revision) {
            return { ...data, task_bindings: held.bindings };
        }
        held = revision && data.task_bindings && typeof data.task_bindings === 'object'
            ? { revision, bindings: data.task_bindings } : null;
        return data;
    };
    const open = () => { settled = deferred(); inflight = begin(); return inflight; };
    const settle = (request) => {
        if (!inflight || request !== inflight) return;
        const done = settled;
        const next = followUp;
        inflight = settled = followUp = null;
        if (next) next.resolve({ request: open(), done: settled.promise });
        done.resolve();
    };
    return {
        begin,
        retire() { appliedGeneration = begin().generation; latest = null; held = null; },
        gate(force = false) {
            if (!inflight) return Promise.resolve(open());
            if (!force) return settled.promise.then(() => null);
            if (!followUp) { followUp = deferred(); return followUp.promise.then((f) => f.request); }
            return followUp.promise.then((f) => f.done).then(() => null);
        },
        apply(request, data) {
            try {
                const generation = Number(request?.generation) || 0;
                if (!generation || generation <= appliedGeneration) return false;
                appliedGeneration = generation;
                latest = data = withBindings(data);
                onApply(data, request.requestedAt, generation);
                return true;
            } finally { settle(request); }
        },
        isCurrent(request) {
            return (Number(request?.generation) || 0) > appliedGeneration;
        },
        fail(request) {
            try {
                const generation = Number(request?.generation) || 0;
                if (!generation || generation <= appliedGeneration) return false;
                appliedGeneration = generation;
                latest = null;
                onUnavailable(request.requestedAt, generation);
                return true;
            } finally { settle(request); }
        },
        latest: () => latest,
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

export const TERMINAL_TASK_STATUSES = new Set([
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
 * Whether an unkeyed live row ends the unscoped Main turn: only a legacy model final does.
 * Voice confers no finality (DESIGN), so a System row concludes nothing; one answering a
 * submission names its `client_message_id`, and only that `Sending` settles.
 */
export function unkeyedFrameEndsTurn(row) {
    return String(row?.role || '') !== 'system';
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

// Only a reload onto changed code earns a feed notice; any other reload reason or reconnect is silent.
export function restartNoticeText(reason = '') {
    return reason === 'sha-change' ? 'Restart complete' : '';
}

// The message clock is the chat's one compact time format, shared with the terminal time line (utils.js).
export { formatMsgTime };

/**
 * Project one /api/state activity census onto the client's active-activity map.
 *
 * This map is a PROJECTION of the census: the client never inserts into it from
 * a WS frame (a typing frame is a submission receipt, not liveness), so the
 * census is its only inserter (finals and the census delete). Deletion
 * authority follows from that — on a
 * `complete` snapshot every id the census does not list is gone, whatever its
 * kind, with no wall-clock barrier and no generation marker. An incomplete
 * snapshot retains omitted rows as unconfirmed; only its positive rows can
 * affirm current activity.
 *
 * `concludedIds` (Set/Map with .has) is the client-side conclusion ledger: a
 * turn already concluded by its keyed final must never be re-inserted by a
 * snapshot captured while it still ran (activity ids are unique task ids and
 * never restart, so conclusion is final).
 */
export function computeHydratedDirectActivities(existingMap, turnsList, chatId, concludedIds = null, complete = true) {
    const nextMap = new Map(Array.from(existingMap || [], ([id, row]) => [id,
        complete ? row : { ...row, _activityUnconfirmed: true }]));
    if (!Array.isArray(turnsList)) return nextMap;
    const listed = new Set();
    for (const turn of turnsList) {
        if (Number(turn?.chat_id ?? 1) !== chatId) continue;
        const aid = String(turn?.activity_id || '').trim();
        if (!aid || (concludedIds && concludedIds.has(aid))) continue;
        listed.add(aid);
        nextMap.set(aid, {
            ...turn, _activityUnconfirmed: false,
            activityId: aid,
            kind: turn.kind || 'direct_chat',
            phase: turn.phase || 'thinking',
            clientMessageId: turn.client_message_id || nextMap.get(aid)?.clientMessageId || '',
        });
    }
    // The census is the only inserter into this map, so on a complete snapshot
    // every id it does not list is gone, whatever its kind. An incomplete
    // snapshot (supervisor not ready / a source failed) deletes nothing.
    if (complete) for (const aid of nextMap.keys()) if (!listed.has(aid)) nextMap.delete(aid);
    return nextMap;
}

/**
 * Hydrate one authoritative activity census (see
 * computeHydratedDirectActivities for the projection contract) and identify the
 * narrower event that can wake durable task-detail convergence: a host-stamped
 * managed root the client held, now absent from the GLOBAL snapshot. A root
 * still listed under another chat merely departed locally. Direct/ephemeral
 * removals carry no task-detail/card authority, but they ARE conclusions
 * (#369): the caller records them so a late frame cannot resurrect the
 * turn, and clears the linked Sending... submission.
 */
export function reconcileHydratedDirectActivities(
    existingMap,
    turnsList,
    chatId,
    concludedIds = null,
    complete = true,
) {
    const activities = computeHydratedDirectActivities(
        existingMap, turnsList, chatId, concludedIds, complete,
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
 * Card-set durable-truth reconcile (stuck "Working..." card class). A card
 * minted by a replayed frame whose own terminal row never reached this client
 * (lost task_done, lineage-only subagent final re-minting a finished parent)
 * kept its chip on "Working..." forever: every existing terminal path keys on
 * the card's OWN id reaching a snapshot or frame first, and such a card never
 * sets the room's status (the server's work projection does). This selector
 * closes the gap from the card side: given the
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
 * too), reusable slots ('active' — many cycles per id, no single durable
 * result) and the 'chat' fallback group id. Pure for node tests.
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

// В9: the host stamps a typed `ingress_accepted: true` on an owner echo only after the durable chat write, so that
// client_message_id's bubble says `Input saved` — never that work began. `ingress_pending` keeps that note
// provisional until dispatch is confirmed or the ended host's delivery becomes unknown.
export function markIngressSaved(root, row) {
    const cmid = String(row?.client_message_id || ''), state = row?.ingress_pending === true ? 'pending' : '';
    if (row?.role !== 'user' || row.ingress_accepted !== true || !cmid) return false;
    const bubble = [...root.querySelectorAll('.chat-bubble.user[data-client-message-id]')]
        .find((node) => node.dataset.clientMessageId === cmid), prior = bubble?.querySelector('[data-ingress-saved]');
    if (!bubble || (prior && (prior.dataset.ingressSaved !== 'pending' || prior.dataset.ingressSaved === state))) return false;  // `Input saved` and a kept send's delivery doubt are final
    const note = Object.assign(document.createElement('div'), { className: 'msg-pending', textContent: 'Input saved' });
    note.dataset.ingressSaved = state;
    for (const doubt of [prior, bubble.querySelector('[data-ingress-unconfirmed]')]) doubt?.remove();  // the saved row settles a "Send again" doubt
    bubble.insertBefore(note, bubble.querySelector('.msg-time'));
    return true;
}

// Full narration stays in the timeline, not a mouse-only title.
// How many lines a COLLAPSED card's activity line shows (the CSS clamps say the same: a child 1, a Main portal
// 2, a root card 3); an expanded card shows the whole line.
export function collapsedActivityLines(record) {
    const root = record?.root;
    if (!root || root.dataset?.expanded === '1') return 0;
    if (record.isSubagent || root.dataset?.subagent === '1') return 1;
    return root.dataset?.portal === '1' ? 2 : 3;
}

/** Re-fit the activity line after the card's form changed (expanded, collapsed, bound as a portal). */
export function refitCollapsedActivity(record) {
    const el = record?.activityEl;
    if (!el) return false;
    fitText(el, fullText(el), collapsedActivityLines(record));
    return true;
}

export function renderCollapsedActivity(record, text) {
    if (!record?.activityEl) return false;
    const el = record.activityEl;
    // A child's report names its reader before its words: the label is drawn by CSS from
    // `data-lead`, so the fit below counts its room and a copy of the line keeps only the words.
    const lead = text ? String(record.collapsedActivityLead || '') : '';
    const leadChanged = (el.dataset?.lead || '') !== lead;
    if (leadChanged && el.dataset) {
        if (lead) el.dataset.lead = lead;
        else delete el.dataset.lead;
    }
    // A collapsed card's line is cut on a word with a clean ending (never `.…`; text_fit.js).
    const shown = fullText(el);
    const changed = shown !== text || leadChanged || Boolean(el.hasAttribute?.('title'));
    if (shown !== text || leadChanged) fitText(el, text, collapsedActivityLines(record));
    if (el.hasAttribute?.('title')) el.removeAttribute('title');
    return Boolean(changed && el.isConnected);
}

// The terminal observation names only this task's loose files. Keep paths and
// HEAD in the existing Details timeline; the compact line lives in taskReasonDetail.
export function readCandidateRemainder(item, detail, ok = true) {
    const body = ok ? candidateRemainderBody(detail, item.remainderFact) : null;
    item.fullReadError = body !== null ? '' : ok
        ? 'The saved file list is unavailable for this observation. Close and reopen this detail to retry.'
        : 'The saved file list could not be loaded. Close and reopen this detail to retry.';
    return body;
}

export function candidateRemainderBody(detail, fact) {
    const snapshot = detail?.mutation_evidence?.terminal_candidate_snapshot;
    const exact = fact?.snapshot_at && snapshot?.captured_at === fact.snapshot_at
        && snapshot?.baseline_hash === fact.baseline_hash && Array.isArray(snapshot.surfaces);
    // Older terminal rows carried the exact list inline. A newer observation
    // never substitutes a different saved snapshot for the one the summary names.
    const paths = exact ? snapshot.surfaces.filter(row => Array.isArray(row.uncommitted_paths)).map(row => ({
        root: row.canonical_root, commit: row.base_commit, paths: row.uncommitted_paths,
    })) : Array.isArray(fact?.surfaces) ? fact.surfaces : null;
    if (!paths) return null;
    return [fact.state === 'unknown' ? 'The uncommitted remainder could not be checked.' : '',
        ...paths.filter(row => Array.isArray(row.paths) && row.paths.length).map(row =>
            `Candidate: ${row.root || 'Saved candidate'}\nHEAD: ${row.commit || 'unknown'}\n\nFiles:\n${row.paths.join('\n')}`)]
        .filter(Boolean).join('\n\n');
}

export function syncCandidateRemainderItem(record, detail) {
    const fact = detail?.outcome_axes?.execution?.mutation_attribution?.uncommitted_remainder;
    if (!record || !fact) return false;
    const key = `candidate-remainder|${record.groupId}`;
    const index = record.items.findIndex(item => item.dedupeKey === key);
    const previous = record.items[index];
    const same = previous && JSON.stringify(previous.remainderFact) === JSON.stringify(fact);
    const full = candidateRemainderBody(detail, fact);
    const fullRef = fact.snapshot_at ? String(fact.task_id || detail.task_id || record.groupId) : '';
    const body = fact.state === 'known' && fact.count === 0 ? '' : full
        ?? (same && !previous.truncated ? previous.body : null)
        ?? [fact.state === 'unknown' ? 'The uncommitted remainder could not be checked.' : '',
        fact.paths_omitted > 0 ? `${fact.paths_omitted} recorded file path${fact.paths_omitted === 1 ? ' is' : 's are'} not loaded.` : '',
        fullRef ? 'Expand to read the saved file list.' : 'The saved file list is unavailable.']
        .filter(Boolean).join('\n\n');
    if (!body) {
        if (index < 0) return false;
        record.items.splice(index, 1);
        return true;
    }
    if (same && previous.body === body) return false;
    const item = { phase: 'info', headline: 'Uncommitted files', fullHeadline: 'Uncommitted files',
        body, fullBody: body, bodyFormat: 'text', receipt: true, count: 1, ts: '', sourceTs: String(detail.ts || ''),
        dedupeKey: key, lineKey: `candidate-remainder-${String(record.groupId).replace(/[^A-Za-z0-9_-]/g, '-')}`,
        remainderFact: fact, fullField: 'candidate_remainder', fullRef,
        truncated: full === null && Boolean(fullRef) && !(same && !previous.truncated),
        fetchedFull: same ? previous.fetchedFull : '', fullReadError: same ? previous.fullReadError : '' };
    if (index < 0) record.items.push(item);
    else Object.assign(record.items[index], item);
    return true;
}

// The 13 cost-meta keys shared by both subagent whitelists (the delegation
// trio stays inline in each literal — the wire test scans those literals).
const COST_META_KEYS = [
    'cost_usd', 'accounted_upper_bound_usd', 'accounted_upper_bound_usd_with_children',
    'cost_accounting_status', 'cost_accounting_error', 'cost_final', 'cost_usd_with_children',
    'cost_with_children_partial', 'reserved_usd', 'unresolved_upper_bound_usd',
    'unknown_unmetered', 'non_final_rows', 'cost_presentation',
];
export function costMetaKeys(src) {
    return Object.fromEntries(COST_META_KEYS.map((key) => [key, src?.[key]]));
}

const CARD_META_KEYS = [
    ...COST_META_KEYS, 'executor_route', 'execution_evidence', 'actual_substrate',
    'executor_observation', 'model_execution', 'tool_calls', 'model', 'ts', 'initiator', 'cancel_origin',
    'delegated_activity', 'outcome_axes', 'task_completion', 'effort_level', 'effort_requested', 'effort_source',
];
export function cardMetaKeys(src) {
    return Object.fromEntries(CARD_META_KEYS.map((key) => [key, src?.[key]]));
}

// the ONE meta-line renderer, fed entirely from record state,
// so a replay batch renders it exactly once per card.
// "updated" names the turn's own last narration in the chat's one time format (formatMsgTime: the
// same words as a bubble's clock), never the host locale's long date.
function updatedTime(record) {
    const time = record.latestActivityRaw ? formatMsgTime(record.latestActivityRaw) : null;
    return time?.short || record.latestActivityTs || '';
}

function updatedLabel(record) {
    const text = updatedTime(record);
    return text ? `updated ${text}` : '';
}

// An unfinished card always carries a time on its meta line: its last narration's, else when the work
// started (the census start, else the card's first row). The line is never empty beside a control.
function clockLabel(record) {
    const updated = updatedLabel(record);
    if (updated || record.finished) return updated;
    const started = Number(record.startedAt) || Number(record.root?.dataset?.ts) || 0;
    const time = started > 0 ? formatMsgTime(new Date(started).toISOString()) : null;
    return time?.short ? `started ${time.short}` : '';
}

// The meta line keeps its real ` · ` text separators (a copied line reads as it looks), but each
// separator rides inside the part before it and holds on to that part's last word (a no-break space
// before the dot): a wrapped line may end with a dot, never start with one or keep a dot alone.
const META_SEP_HTML = `<span class="chat-live-meta-sep">${META_PART_SEPARATOR.replace(/^ /, '&nbsp;')}</span>`;
// The dot goes inside the part's last fact when that fact is a plain meta span, so the fact (kept
// whole, as today) carries its dot with it; otherwise it follows the part.
function withTrailingSeparator(part) {
    const lastFact = part.lastIndexOf('<span class="chat-live-meta-text');
    const tail = lastFact >= 0 ? part.slice(lastFact + 1) : '';
    if (lastFact >= 0 && /<\/span>\s*$/.test(part) && !tail.includes('<span') && (tail.match(/<\/span>/g) || []).length === 1) {
        return part.replace(/<\/span>\s*$/, `${META_SEP_HTML}</span>`);
    }
    return part + META_SEP_HTML;
}

function metaLineHtml(parts) {
    const kept = parts.filter(Boolean);
    return kept.map((part, index) => `<span class="chat-live-meta-part">${index < kept.length - 1
        ? withTrailingSeparator(part) : part}</span>`).join('');
}

// The card's facts beyond its compact line: labelled rows of its own details block (a nested child
// carries its own), opened by the card's one disclosure. A row exists only for a known fact; the
// markup equals its serialization, so an unchanged list is never rebuilt under a reader's selection.
function paintLiveFacts(record, rows) {
    const el = ownLiveDetail(record, 'liveFacts');
    if (!el) return false;
    if (record.wakeSummary) rows.unshift([tr('task.fact.wake_summary', 'In its words'), `<span data-i18n-authored>${escapeHtml(record.wakeSummary)}</span>`]);
    const html = rows.filter(([, value]) => value).map(([label, value]) =>
        `<div class="chat-live-fact"><dt>${escapeHtml(label)}</dt><dd>${value}</dd></div>`).join('');
    if (el.innerHTML === html && el.hidden === !html) return false;
    if (el.innerHTML !== html) el.innerHTML = html;
    el.hidden = !html;
    return true;
}

export function renderLiveCardMeta(record, { agentModel = record?.agentModel || '' } = {}) {
    if (!record?.metaEl) return false;
    const toolCalls = Number.isInteger(record.toolCalls) ? `${record.toolCalls} tool ${record.toolCalls === 1 ? "call" : "calls"}` : '';
    const rounds = Number.isInteger(record.rounds) && record.rounds >= 0 ? `${record.rounds} round${record.rounds === 1 ? '' : 's'}` : '';
    const cost = (record.costMeta && Array.isArray(record.costMeta.meta)) ? record.costMeta.meta : [];
    // A Main portal is the collapsed form of a work that lives in its Project room: tool calls ·
    // cost · updated, plus the facts that change what the owner may trust. Executor, model and
    // effort details stay on the room's card.
    const portal = record.root?.dataset?.portal === '1';
    const text = (item) => `<span class="chat-live-meta-text">${escapeHtml(item)}</span>`;
    // The compact line says who ran the work, on what, how long and for how much (DESIGN §5 "A task card's
    // summary outranks its details"). A delegated child names its executor and only the executor's own model
    // evidence; any other card the model a usable solve response observed, else its agent model, and an
    // unobserved execution in its own words. Effort is the selected leaf's dispatch level, not the nanny's
    // reasoning level or a provider echo (subagent_runtime.resolve_configured_actor_dispatch).
    // Session counters belong to the supervising agent, never the foreign executor.
    const chip = record.executorChip;
    const supervision = chip && harnessPresentation(chip.harness).kind === 'harness';
    const execution = modelExecutionView(record.modelExecution);
    const coordinator = compactModel(agentModel);
    const provenance = chip ? formatCoordinatorProvenance(
        record?.coordinatorProvenance || chip?.coordinatorProvenance || chip?.snapshot || '',
    ) : '';
    const model = chip ? executorModelText(chip, compactModel)
        : execution && !execution.observed ? execution.value : execution?.model || coordinator;
    // A key added under Accounts follows its model ("claude-opus-5.5 · Claude credits"); the built-in key adds nothing.
    const keyRoute = chip || (execution && !execution.observed) ? '' : execution ? record.modelExecution?.used_model : agentModel;
    const key = connectionLabel(routeConnectionId(keyRoute));
    const updated = record.finished ? updatedTime(record) : '';
    // The executor block is ONE part of this line, joined by the same text
    // separator as the rest: concatenating it left the chip and the first fact
    // touching in a copied line and running together for a screen reader,
    // because only the flex gap separated them.
    const html = portal ? metaLineHtml([
        record.historicalUnavailable ? 'Outcome unavailable' : (record.historicalUnconfirmed ? 'Activity unconfirmed' : ''),
        !record.finished && record.projectHoldDetail || '',
        // A settled portal needs no activity clock: its chip states how the work ended and the author's
        // final (with its own end time) arrives as a message.
        toolCalls, ...cost, record.finished ? '' : clockLabel(record),
    ].filter(Boolean).map(text)) : metaLineHtml([
        chip ? executorIdentityMarkup(chip, { model }) : (model ? text(key ? `${model} · ${key}` : model) : ''),
        effortChipMarkup(record.effort),
        ...[
            record.historicalUnavailable ? 'Outcome unavailable' : (record.historicalUnconfirmed ? 'Activity unconfirmed' : ''),
            !record.finished && record.projectHoldDetail || '',
            record.historyRetentionProblem || '',
            record.toolErrors > 0 ? `${record.toolErrors} error${record.toolErrors === 1 ? '' : 's'}` : '',
            Number.isFinite(record.durationSec) ? formatLogDuration(record.durationSec) : '',
            !supervision && record.toolCalls > 0 ? toolCalls : '',
            ...(Array.isArray(record._lastFrameMeta) ? record._lastFrameMeta : []),
            ...cost,
            record.finished ? '' : clockLabel(record),
        ].filter(Boolean).map(text),
    ]);
    const factsChanged = paintLiveFacts(record, portal ? [] : [
        ['Executor', chip?.title ? escapeHtml(chip.title) : ''],
        // Provenance is a recorded fact of the schedule (chosen_by / coordinator_provenance), never inferred here.
        [chip ? 'Coordinator' : 'Agent model', coordinator ? escapeHtml(provenance ? `${coordinator} (${provenance})` : coordinator) : ''],
        // An unobserved execution of a card without an executor is its compact model fact already.
        [execution?.label || 'Model execution', execution && (chip || execution.observed) ? escapeHtml(execution.value) : ''],
        ['Started by', record.scheduleName ? `${tr('task.fact.started_by.schedule', 'Schedule')} · <span data-i18n-authored>${escapeHtml(record.scheduleName)}</span>` : record.initiator === 'consciousness' ? 'Consciousness' : ''],
        ['Tool calls', !supervision && record.toolCalls === 0 ? '0' : ''],
        ['Rounds', !supervision && rounds ? `${record.rounds} · usable solve response${record.rounds === 1 ? '' : 's'}` : ''],
        ['Supervision', supervision && (rounds || toolCalls)
            ? `${[rounds, toolCalls].filter(Boolean).join(META_PART_SEPARATOR)} (supervising agent; executor session counts are not reported)` : ''],
        ['Updated', escapeHtml(updated)],
    ]);
    const detailChanged = paintLiveDetail(record, 'liveCostDetail', String(record.costMeta?.detail || ''));
    if (record.metaEl.innerHTML === html) return (detailChanged || factsChanged) && Boolean(record.metaEl.isConnected);
    record.metaEl.innerHTML = html;
    return Boolean(record.metaEl.isConnected);
}

// Only host-attested cancelable queue roots receive this control.
// A card's OWN actions row: a nested child card carries a row of its own, and a
// descendant lookup would hand a root without one its child's.
// The row lives at the end of the card's own foot, under its meta line (right-docked, after the quiet
// Reviews count or a portal's Project reference), and the summary above it keeps its own inset. A nested
// child's foot is its own, so a descendant lookup is never used.
function ownFootEnd(record) {
    const foot = record?.footEl;
    const end = record?.footEndEl;
    return foot && end && foot.parentElement === record.root && end.parentElement === foot ? end : null;
}

export function ownLiveActionsEl(record) {
    const holder = ownFootEnd(record) || record?.root;
    return [...(holder?.children || [])]
        .find((node) => node.classList?.contains('chat-live-actions')) || null;
}

export function ensureLiveActionsEl(record) {
    if (!record?.root
        || record.root.dataset.projectCreated === '1'
        || record.root.dataset.projectCreating === '1') return null;
    let actions = ownLiveActionsEl(record);
    if (!actions) {
        actions = document.createElement('div');
        actions.className = 'chat-live-actions';
        const end = ownFootEnd(record);
        if (end) {
            end.appendChild(actions);
        } else {
            const timeline = record.timelineEl && record.timelineEl.parentElement === record.root
                ? record.timelineEl
                : null;
            record.root.insertBefore(actions, timeline);
        }
    }
    return actions;
}

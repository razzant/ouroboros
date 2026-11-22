/** Historical rows enrich the existing timeline without acquiring live authority. */
import { foldWaitRow } from './task_checkpoints.js';
import { childLineCause } from './log_events.js';

export function compareHistoryPosition(left, right) {
    if (!left || !right) return 0;
    const stream = String(left.source || '').localeCompare(String(right.source || ''));
    return stream || Number(left.offset) - Number(right.offset);
}

export function historyRowIds(row) {
    return [row.history_id, ...(row.review_group?.attempts || []).map(attempt => attempt.history_id)].filter(Boolean);
}

/** A stored owner row carries its physical room in its address. Old rows in their own feed can lack it. */
export function ownerRowRoom(row, fallback) {
    const address = /^row:(-?\d+)@/.exec(row?.row_address || '');
    return address ? Number(address[1]) : Number(row?.chat_id ?? fallback);
}

/** Find the source bubble for a host receipt. A pending echo still belongs to its submitting room. */
export function ownerMessageBubble(feed, row, chatId) {
    if (!row?.client_message_id) return null;
    const room = ownerRowRoom(row, chatId);
    const bubble = [...(feed?.querySelectorAll('.chat-bubble.user[data-client-message-id]') || [])].find(({ dataset }) => {
        if (dataset.clientMessageId !== row.client_message_id) return false;
        if (row.origin_projected || dataset.originProjected) return Boolean(row.origin_projected && row.origin_id && row.origin_id === dataset.originId);
        if (dataset.rowAddress && row.row_address) return dataset.rowAddress === row.row_address;
        return ownerRowRoom({ row_address: dataset.rowAddress }, dataset.ownerChatId ?? chatId) === room;
    }) || null;
    if (bubble && row.origin_projected && row.opening_input_kind) {
        const explanation = row.opening_input_kind === 'owner_press'
            ? 'Button pressed without words. The label is the asking task’s text.'
            : row.opening_input_kind === 'owner_card_input'
                ? 'Card input received. Any button label is the asking task’s text, not the owner’s words.'
                : 'Card answer. Any button label is the asking task’s text, not part of the answer.';
        bubble.setAttribute?.('aria-description', explanation);
        if (!bubble.title) bubble.title = explanation;
    }
    return bubble;
}

/**
 * The room a mounted owner bubble's words were written in, as the host stated it: the physical address of the row it
 * was drawn from, else the original's room a Project's retained copy names (`data-source-chat-id`). `null` when the
 * host stated neither (a local echo, a copy from before copies named it): never the room on screen.
 */
export function statedOwnerRoom(node) {
    const address = /^row:(-?\d+)@/.exec(node?.dataset?.rowAddress || '');
    if (address) return Number(address[1]);
    const copy = node?.dataset?.sourceChatId;
    return copy === undefined || copy === '' ? null : Number(copy);
}

/** A receipt whose host stated that no row wears it (`source_row_address: ""`, live and saved alike): no mounted bubble
 * is its row, whatever its client id, and a unique match is no authority. */
export function statesNoRow(frame) {
    return frame?.source_row_address === '';
}

/**
 * In a Project, the one mounted owner bubble another room's routing receipt paints: the bubble the host showed as
 * exactly that message, by the physical address the host stated (the row's own, or its retained copy's) and its stated
 * room. No bubble is made for it, a bubble of this room or one whose room was never stated takes nothing from it, and
 * a hidden panel or a synthetic room has no such receipt. A frame that names no room or no address (an older one) is
 * no other room's: a mounted client id is no placement, so only Main's own path ever read it as Main's.
 */
export function foreignReceiptBubble(feed, frame, chatId) {
    if (!Number.isInteger(frame?.chat_id) || !frame.source_row_address) return null;
    const room = frame.chat_id;
    const id = String(frame.client_message_id || '');
    if (!id || chatId <= 0 || room <= 0 || room === chatId) return null;
    const found = [...(feed?.querySelectorAll('.chat-bubble.user[data-client-message-id]') || [])]
        .filter((node) => node.dataset.clientMessageId === id && statedOwnerRoom(node) === room
            && (node.dataset.rowAddress || node.dataset.sourceRowAddress) === frame.source_row_address);
    return found.length === 1 ? found[0] : null;
}

/** An exact receipt cannot be painted on a pending echo that has not learned its physical address. */
export function receiptBubble(feed, frame, chatId) {
    if (statesNoRow(frame)) return null;
    const room = ownerRowRoom(frame, chatId);
    const matches = [...feed.querySelectorAll('.chat-bubble.user[data-client-message-id]')].filter((node) => {
        const data = node.dataset;
        if (data.clientMessageId !== frame.client_message_id) return false;
        if (frame.source_row_address) return (data.rowAddress || data.sourceRowAddress) === frame.source_row_address;
        return !data.originProjected && (statedOwnerRoom(node) ?? Number(data.ownerChatId ?? chatId)) === room;
    });
    return matches.length === 1 ? matches[0] : null;
}

/** Whether a canonical history row adopts this mounted bubble as its own: the bubble is still a retained copy
 * (`data-origin-projected`, before the row's adoption clears it) of that exact origin identity, never matched by words
 * or time, and names no other row's address (a resolvable copy already wears this row's own). */
export function adoptsRetainedCopy(node, opts) {
    const data = node?.dataset || {};
    if (opts.originProjected || !opts.originId || data.originProjected !== '1' || data.originId !== opts.originId) return false;
    return !data.rowAddress || data.rowAddress === String(opts.reply?.row_address || '');
}

export function matchesHistoryBubble(node, opts, legacyKey, chatId) {
    const data = node.dataset;
    if (opts.originId && data.originId === opts.originId) return true;
    if (!opts.clientMessageId) return !data.historyId && data.messageKey === legacyKey;
    if (opts.originProjected || data.originProjected || data.clientMessageId !== opts.clientMessageId) return false;
    if (data.historyId && data.historyId !== opts.historyId) return false;
    if (data.rowAddress && opts.reply?.row_address) return data.rowAddress === opts.reply.row_address;
    return ownerRowRoom({ row_address: data.rowAddress }, data.ownerChatId ?? chatId) === ownerRowRoom(opts.reply, chatId);
}

/** Saved openings keep their own placement key; the old key is only for echo correlation. */
export function messagePlacementKey(opts, legacyKey) {
    if (opts.historyId) return `history:${opts.historyId}`;
    if (opts.originId) return `origin|${opts.originId}`;
    const address = opts.reply?.role === 'user' && opts.reply.row_address;
    return address ? `row|${address}` : legacyKey;
}

/** Reuse a mounted node for its live echo without changing the attribution of a retained copy. */
export function livePriorBubble(bubbles, opts, keys, legacyKey, chatId) {
    if (opts.originId) return (!keys || keys.origin.has(opts.originId))
        ? bubbles().find(node => node.dataset.originId === opts.originId) : null;
    const row = opts.reply;
    const address = row?.role === 'user' ? row.row_address : '';
    const room = ownerRowRoom(row, chatId);
    if (keys && !address && !keys.legacy.has(legacyKey)) return null;
    return bubbles().find(({ dataset: data }) => {
        if (address && data.rowAddress) return data.rowAddress === address;
        if (data.messageKey !== legacyKey) return false;
        if (row?.role !== 'user') return !data.originProjected;
        if (data.originProjected && (address || data.rowAddress)) return false;
        return ownerRowRoom({ row_address: data.rowAddress }, data.ownerChatId ?? chatId) === room;
    });
}

/** During a page replay, every identity key a mounted bubble carries (a superset: keys are only added, `note`): a row
 * looks for its prior bubble only when one could be mounted, not by scanning the feed per row (O(rows²) a page). */
export function createReplayKeys() {
    const keys = { history: new Set(), client: new Set(), origin: new Set(), legacy: new Set() };
    keys.note = (node) => {
        if (!node) return;
        const nodes = node.classList?.contains('chat-bubble') ? [node] : [];
        for (const bubble of [...nodes, ...(node.querySelectorAll?.('.chat-bubble') || [])]) {
            const { historyId, clientMessageId, originId, messageKey } = bubble.dataset || {};
            if (historyId) keys.history.add(historyId);
            if (clientMessageId) keys.client.add(clientMessageId);
            if (originId) keys.origin.add(originId);
            if (messageKey) keys.legacy.add(messageKey);
        }
    };
    return keys;
}

export function stampHistoryNode(node, id, position) {
    if (!node || !id) return;
    node.dataset.historyId = id;
    if (position) {
        node.dataset.historySource = position.source;
        node.dataset.historyOffset = String(position.offset);
    }
}

// Cause facts only a frame that states them replaces: an absent key keeps the last stated value, an explicit
// empty debt list clears it. A reason survives only while the frame also states no typed axes or review —
// those are the new authoritative cause and drop the old reason with it.
const CARRIED_CAUSE_FACTS = ['_result_text_present', 'delegated_runs_unreconciled'];
const REASON_FACTS = ['reason_code', 'reason_detail'];
const REASON_AXES = ['outcome_axes', 'terminal_plan_review_open', 'review_status', 'task_completion'];

function mergeChildCause(stated, before) {
    if (!before) return stated;
    let merged = stated;
    const carry = (key) => {
        if (merged[key] === undefined && before[key] !== undefined) merged = { ...merged, [key]: before[key] };
    };
    for (const key of CARRIED_CAUSE_FACTS) carry(key);
    if (!REASON_FACTS.some((key) => merged[key] !== undefined) && !REASON_AXES.some((key) => merged[key] !== undefined)) {
        for (const key of REASON_FACTS) carry(key);
    }
    return merged;
}

/** What a line keeps across the frames that rewrite it. A child's ending carries two independent facts: its report and
 * trace (`fullOutput` with their `outputPreview`, free of cause words, from the newest frame that carried them) and its
 * cause facts (`childCause`, the newest frame's, with facts it does not restate carried). A frame of the outcome alone
 * (task_done, a late review's current outcome) carries no report: the line keeps the one it holds and words that
 * frame's cause beside it, an emptied cause included. The answer's presence stays as last stated until a frame states it again. */
export function timelineReadFields(summary, previous = {}) {
    const stated = summary.childCause, before = previous.childCause;
    const cause = stated ? mergeChildCause(stated, before) : stated;
    const read = {
        fullRef: summary.fullRef || previous.fullRef || '',
        fullField: summary.fullField || previous.fullField,
        truncated: summary.truncated || previous.truncated || false,
        ...(summary.serviceStop ? { serviceStop: summary.serviceStop } : {}),
        childCause: cause && before && JSON.stringify(cause) === JSON.stringify(before) ? before : cause,
    };
    if (!stated) return previous.fullOutput ? { ...read, fullOutput: '', outputPreview: '' } : read;
    if (summary.fullOutput) return { ...read, fullOutput: summary.fullOutput, outputPreview: summary.outputPreview || '' };
    const fullOutput = previous.fullOutput || '', outputPreview = previous.outputPreview || '';
    // A status-only frame's fullBody is the cause detail composed from its own facts; word it from the
    // carried facts instead, so an absent current debt does not drop a known open warning.
    const detail = childLineCause(cause).detail;
    return { ...read, fullOutput, outputPreview, body: summary.body || outputPreview,
        fullBody: [fullOutput, detail].filter(Boolean).join('\n\n') };
}

export function mergeHistoricalTimelineItem(record, summary, row, ts) {
    if (summary.visible === false || !(summary.headline || summary.body)) return false;
    const identity = String(row?.history_id || '');
    if (!identity) return false;
    if (summary.waitEpisode) {  // the same one row per wait kind as live
        const { changed, index, moved } = foldWaitRow(record, summary, { ts, sourceTs: row.ts || '', earliest: true });
        if (index === -1 || moved) sortBySource(record);
        return changed;
    }
    // A child's lifecycle is one evolving status, just as it is live. Its
    // authored progress keeps every source record, even when text and time match;
    // so does each delegated observation (`activity`), projected per seq at render.
    // The task's outcome (`task_done|`) is one line too, as live: a read inside the
    // post-task window projects it still finalizing, which is that line Working,
    // never a line of work, and a finished card takes nothing from it (live: closedPhase).
    const outcome = String(summary.dedupeKey || '').startsWith('task_done|');
    if (outcome && !summary.terminal && record.finished) return false;
    const evolving = summary.terminal || outcome || String(summary.dedupeKey || '').startsWith('subagent-lifecycle:')
        || String(summary.dedupeKey || '').startsWith('cardrow|');
    const activity = summary.activity ? { activity: summary.activity } : {};
    const key = evolving ? summary.dedupeKey : `history:${identity}`;
    // A child's words and a thought keep their live occurrence alias beside the physical history
    // identity: history may precede live delivery, or adopt a line before it is redelivered.
    const occurrence = /^(subagent-line|reasoning):/.test(String(summary.dedupeKey || ''))
        ? { occurrenceKey: summary.dedupeKey } : {};
    const source = { sourceHistoryId: identity, sourceHistoryRevision: summary.cardRowRevision,
        historyPosition: row.history_position };
    let item = record.items.find((entry) => entry.dedupeKey === key);
    if (!item && !evolving) {
        item = record.items.find((entry) => !entry.historyId
            && (entry.occurrenceKey ?? entry.dedupeKey) === summary.dedupeKey && entry.sourceTs === row.ts);
    }
    if (item && evolving) {
        // Source revision advances independently of live content: an older
        // fallback can yield to its canonical current row without rolling back
        // content, while stale sources cannot redirect an already newer locator.
        const existingPosition = item.historyPosition;
        const adoptedSource = !item.sourceHistoryId || Number.isSafeInteger(summary.cardRowRevision)
            && (!Number.isSafeInteger(item.sourceHistoryRevision) || summary.cardRowRevision > item.sourceHistoryRevision);
        if (adoptedSource) Object.assign(item, source);
        const incomingTime = Date.parse(row.ts), existingTime = Date.parse(item.sourceTs || '');
        if (Number.isSafeInteger(item.cardRowRevision)) {
            // Only a newer revision of this merge's review can enrich an equal receipt revision.
            const newerReview = summary.cardReview?.record_id && Number.isSafeInteger(summary.cardReview.revision)
                && (!item.cardReview || summary.cardReview.record_id === item.cardReview.record_id
                    && summary.cardReview.revision > item.cardReview.revision);
            if (!Number.isSafeInteger(summary.cardRowRevision) || summary.cardRowRevision < item.cardRowRevision
                || summary.cardRowRevision === item.cardRowRevision && !newerReview) return adoptedSource;
        } else if (!(outcome && summary.currentOutcome) && !Number.isSafeInteger(summary.cardRowRevision) && (incomingTime < existingTime || incomingTime === existingTime
                && compareHistoryPosition(row.history_position, existingPosition) < 0)) {
            // An older row still brings the report a newer frame of the outcome alone never carried: the line takes
            // the report and keeps its newer cause (without a report its body and fullBody are that cause's words).
            if (!(summary.childCause && summary.fullOutput && item.childCause && !item.fullOutput)) return adoptedSource;
            Object.assign(item, timelineReadFields({ ...item, fullRef: summary.fullRef, fullField: summary.fullField, truncated: summary.truncated },
                { ...item, childCause: summary.childCause, fullOutput: summary.fullOutput, outputPreview: summary.outputPreview }));
            return true;
        }
        const update = { headline: summary.headline || item.headline,
            cardRowRevision: summary.cardRowRevision,
            cardReview: summary.cardReview || null, evidenceRef: summary.evidenceRef || null,
            fullHeadline: summary.fullHeadline || summary.headline || item.fullHeadline,
            body: summary.body || '', fullBody: summary.fullBody || summary.body || '',
            ...timelineReadFields(summary, item),
            phase: summary.phase || item.phase, sourceTs: row.ts || item.sourceTs,
            ts, ...source };
        if (Object.entries(update).every(([key, value]) => key === 'historyPosition'
            ? JSON.stringify(item[key]) === JSON.stringify(value) : item[key] === value)) return adoptedSource;
        Object.assign(item, update);
        return true;
    }
    if (item?.historyId === identity) return false;
    if (item) {
        item.historyId = identity;
        item.historyPosition = row.history_position;
        item.dedupeKey = key;
        Object.assign(item, occurrence);
        item.count = 1;
        if (summary.evidenceRef && !item.evidenceRef) item.evidenceRef = summary.evidenceRef;
    } else {
        record.items.push({
            cardRowRevision: summary.cardRowRevision,
            cardReview: summary.cardReview || null,
            phase: summary.phase || 'working', headline: summary.headline || 'Update',
            fullHeadline: summary.fullHeadline || summary.headline || 'Update',
            body: summary.body || '', fullBody: summary.fullBody || summary.body || '',
            ...timelineReadFields(summary),
            evidenceRef: summary.evidenceRef || null,
            ts: ts || '', sourceTs: row.ts || '', count: 1, dedupeKey: key, ...activity, ...occurrence,
            ...(evolving ? source : { historyId: identity, historyPosition: row.history_position }),
            lineKey: evolving && !String(key).startsWith('cardrow|') ? `terminal-${String(key).replace(/[^A-Za-z0-9_-]/g, '-')}`
                : `history-${identity.replace(/[^A-Za-z0-9_-]/g, '-')}`,
        });
    }
    sortBySource(record);
    return true;
}

function sortBySource(record) {
    record.items.sort((a, b) => {
        const first = Date.parse(a.sourceTs || ''), second = Date.parse(b.sourceTs || '');
        return (Number.isFinite(first) && Number.isFinite(second) ? first - second : 0)
            || compareHistoryPosition(a.historyPosition, b.historyPosition);
    });
}

/** Each mounted node a released id removes, as [id, node]: the rows it renders
 * and the evolving lines that only locate their page by it (never a row stamp). */
export const historyStamps = root => [['[data-history-id]', 'historyId'], ['[data-source-history-id]', 'sourceHistoryId']]
    .flatMap(([selector, key]) => Array.from(root.querySelectorAll(selector), node => [node.dataset[key], node]));

/** A page can leave the rendered window only outside reading, focus and selection. */
export function historyNodeIsProtected(node, viewport, selection = globalThis.getSelection?.()) {
    if (!node?.isConnected) return false;
    const active = node.ownerDocument?.activeElement;
    if (active && node.contains(active)) return true;
    if (selection && !selection.isCollapsed) {
        for (let index = 0; index < selection.rangeCount; index += 1) {
            try { if (selection.getRangeAt(index).intersectsNode(node)) return true; } catch {}
        }
    }
    return historyNodeOnScreen(node, viewport);
}

/** A rendered node intersecting the feed's viewport (or a band of it); a collapsed one has no boxes. A row asleep
 * (chat_row_sleep.js) shows nothing outside its own box, which stays laid out while what it holds is not: off the
 * screen, it answers for everything in it without laying that out. */
export function historyNodeOnScreen(node, viewport, bounds = viewport.getBoundingClientRect()) {
    if (!node?.isConnected) return false;
    const row = node.closest?.('[data-laid]');
    if (row && row !== node && !historyNodeOnScreen(row, viewport, bounds)) return false;
    const rect = node.getBoundingClientRect();
    return Boolean(node.getClientRects().length && rect.bottom > bounds.top && rect.top < bounds.bottom);
}

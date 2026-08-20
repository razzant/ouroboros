import {
    escapeHtmlAttr,
    escapeHtmlText as escapeHtml,
    renderMarkdown,
} from './utils.js';
import { projectCollapsedActivity } from './chat_card_state.js';

// Live-card presentation for ONE chat instance: what a card SHOWS. The coined
// project name and the bounded collapsed-activity line, the phase label, the
// expand/collapse disclosure and its lazily-materialized timeline, the per-line
// HTML with its expand affordance, the incremental append/patch writers, the
// subagent container, and the single meta-line renderer fed from record state.
// Everything here writes through a card record; the record map, the name buffer,
// the viewport wrapper, the record factory and the layout sync are handed over
// explicitly, so a Main chat and a Project panel render only their own cards.
export function createLiveCardView({
    liveCardRecords,
    pendingSuggestedNames,
    withStableViewport,
    getLiveCardRecord,
    syncLiveCardLayout,
}) {
    // Cluster B: apply the proactively-coined project name to a main card already on
    // screen (live `task_named` event or history replay). A main card's groupId IS its
    // task_id, so the lookup is direct. No-op until the card exists / without a name.
    function applySuggestedName(taskId, name) {
        return withStableViewport(() => applySuggestedNameMutation(taskId, name));
    }

    function applySuggestedNameMutation(taskId, name) {
        const tid = String(taskId || '').trim();
        const nm = String(name || '').trim();
        if (!tid || !nm) return;
        const record = liveCardRecords.get(tid);
        if (!record) {
            // Card not created yet (the namer raced ahead of the first progress event).
            // Buffer so createLiveCardRecord applies it when the card appears.
            // FIFO cap: `task_named` is the one broadcast without a thread gate,
            // so every instance buffers every task's name — bound the buffer so
            // a long-lived instance cannot grow it without limit (P3).
            pendingSuggestedNames.set(tid, nm);
            if (pendingSuggestedNames.size > 100) {
                const oldest = pendingSuggestedNames.keys().next().value;
                pendingSuggestedNames.delete(oldest);
            }
            return;
        }
        if (record.isSubagent) return;
        record.suggestedName = nm;
        if (record.titleEl) record.titleEl.textContent = nm;
        // P1 (v6.82): the collapsed activity line was suppressed while the card
        // was unnamed; populate it now from the remembered candidate so the live
        // task_named direct-DOM path does not depend on the next frame.
        renderCollapsedActivity(record, projectCollapsedActivity({
            suggestedName: nm,
            headline: record.collapsedActivity,
            previous: record.collapsedActivity,
        }));
    }

    // One renderer for the bounded collapsed projection. Full narration is
    // owned by timeline disclosure, never by a mouse-only title attribute.
    function renderCollapsedActivity(record, text) {
        if (!record?.activityEl) return;
        record.activityEl.textContent = text;
        record.activityEl.removeAttribute('title');
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

    function setLiveCardTypingVisible(record, visible) {
        if (!record?.inlineTypingEl) return;
        record.inlineTypingEl.style.display = visible ? '' : 'none';
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
        const mutate = () => {
            if (!record?.root) return;
            record.root.dataset.expanded = expanded ? '1' : '0';
            // perf2 P4.4: first expand materializes a lazily-deferred timeline
            // (its DOM was skipped while the card was collapsed/display:none).
            if (expanded && record._timelineDirty) renderLiveCardTimeline(record);
            syncLiveCardToggle(record);
            if (record.root.isConnected) {
                requestAnimationFrame(() => syncLiveCardLayout(record));
            }
        };
        return record?.root?.isConnected ? withStableViewport(mutate) : mutate();
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

    // perf2 P4.4 (lazy LINEAGE bodies): a collapsed SUBAGENT timeline is
    // display:none, so building its DOM during a bulk replay is pure waste.
    // Data stays complete in record.items; DOM writers defer through this
    // guard while collapsed, and the first setLiveCardExpanded(true)
    // materializes the timeline. TOP-LEVEL cards render eagerly: their
    // collapsed timeline text is part of the feed DOM contract (ui-smoke
    // asserts it), and the deep-lineage fan-out lives in subagent children.
    function deferCollapsedTimeline(record) {
        if (!record) return true;
        if (!record.isSubagent) return false;
        if (record.root?.dataset?.expanded === '1') return false;
        record._timelineDirty = true;
        return true;
    }

    // Full rebuild for initial render and expand/collapse toggles.
    function renderLiveCardTimeline(record) {
        if (deferCollapsedTimeline(record)) return undefined;
        record._timelineDirty = false;
        return withStableViewport(() => {
            const el = record.timelineEl;
            const pinned = isTimelinePinnedToBottom(record);
            const prevTop = el.scrollTop;
            el.innerHTML = record.items.map((item) => buildTimelineItemHtml(item, record)).join('');
            el.scrollTop = pinned ? el.scrollHeight : prevTop;
        });
    }

    // Append without disturbing existing DOM nodes.
    function appendTimelineItem(item, record) {
        if (deferCollapsedTimeline(record)) return;
        // Expanded but stale (dirty was set while collapsed): patching the
        // stale DOM would target wrong nodes — materialize from items instead.
        if (record._timelineDirty) return renderLiveCardTimeline(record);
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
        // perf2 P4.4 [GPT#15]: collapsed → mark dirty and leave; stale
        // expanded DOM → full materialization instead of a mismatched patch.
        if (deferCollapsedTimeline(record)) return;
        if (record._timelineDirty) return renderLiveCardTimeline(record);
        const lastEl = record.timelineEl.lastElementChild;
        if (!lastEl) return renderLiveCardTimeline(record);
        const wrapper = document.createElement('div');
        wrapper.innerHTML = buildTimelineItemHtml(item, record).trim();
        const newNode = wrapper.firstElementChild;
        if (newNode) record.timelineEl.replaceChild(newNode, lastEl);
    }

    // Patch a specific timeline node in place (evolving subagent dashboard rows).
    function patchTimelineItemAt(item, record) {
        // perf2 P4.4 [GPT#15]: same dirty/collapsed discipline as patch-last.
        if (deferCollapsedTimeline(record)) return;
        if (record._timelineDirty) return renderLiveCardTimeline(record);
        const key = String(item.lineKey || '').replace(/[^A-Za-z0-9_-]/g, '');
        const el = key ? record.timelineEl.querySelector(`[data-live-line-key="${key}"]`) : null;
        if (!el) return renderLiveCardTimeline(record);
        const wrapper = document.createElement('div');
        wrapper.innerHTML = buildTimelineItemHtml(item, record).trim();
        const newNode = wrapper.firstElementChild;
        if (newNode) record.timelineEl.replaceChild(newNode, el);
    }

    // perf2 P4.3: the ONE meta-line renderer, fed entirely from record state
    // (sticky executor chip, last frame's meta strings, sticky cost, activity
    // clock) so a replay batch can render it exactly once per card.
    function renderLiveCardMeta(record) {
        if (!record?.metaEl) return;
        const executorChipHtml = record.executorChip
            ? `<span class="harness-chip chat-live-executor-chip" title="${escapeHtml(record.executorChip.title || '')}">`
              + `<span aria-hidden="true">${escapeHtml(record.executorChip.icon || '')}</span> `
              + `${escapeHtml(record.executorChip.label || '')}</span>`
            : '';
        record.metaEl.innerHTML = executorChipHtml + [
            record.groupId === 'bg-consciousness' ? 'Background thinking' : '',
            ...(Array.isArray(record._lastFrameMeta) ? record._lastFrameMeta : []),
            ...((record.costMeta && Array.isArray(record.costMeta.meta)) ? record.costMeta.meta : []),
            record.latestActivityTs ? `Latest ${record.latestActivityTs}` : '',
        ].filter(Boolean).map((item) => `<span class="chat-live-meta-text">${escapeHtml(item)}</span>`).join('');
    }

    return {
        applySuggestedName,
        renderCollapsedActivity,
        ensureSubagentContainer,
        setLiveCardTypingVisible,
        formatLiveCardPhaseLabel,
        setLiveCardExpanded,
        isLiveLineExpandable,
        syncLiveCardToggle,
        directSubagentCount,
        buildTimelineItemHtml,
        isTimelinePinnedToBottom,
        deferCollapsedTimeline,
        renderLiveCardTimeline,
        appendTimelineItem,
        patchLastTimelineItem,
        patchTimelineItemAt,
        renderLiveCardMeta,
    };
}

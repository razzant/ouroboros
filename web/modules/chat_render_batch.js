// perf2 P4 (RENDER-BATCH): pure helpers behind chat.js's rebuildAll replay
// batch and the "Load older" window escalation. Dependency-free at import time
// so node tests can exercise them directly.

/**
 * The Main chat's live-card bound (issue #135). Main never runs destroy(), so its
 * live task cards would accumulate for the whole session. Past `cap` cards BEYOND
 * the population the last full rebuild produced, the next history sync replays
 * durable history instead of folding into the existing cards - the transaction a
 * reconnect already runs. The bound is RELATIVE because a history window mints
 * cards itself (summary rows, progress rows, lineage rows) and can exceed the cap
 * on its own; an absolute cap would rebuild on every later sync.
 *
 * Only the sync that STARTED with the arm up may consume it. A window fetched
 * before the arm went up cannot answer for the cards that raised it: rebuilding
 * from it would drop the newest cards and clear the arm without ever replaying a
 * window that contains them. The three moments a sync passes through are therefore
 * distinct: begin() before its fetch, beginReplay() when its SYNCHRONOUS replay
 * starts (no live frame can interleave after that), and settle() when it lands.
 * An arm raised between the first two came from live cards this window never saw
 * and survives the rebuild; one raised during the replay came from the window's own
 * rows and is answered by the floor that same rebuild sets.
 *
 * Accepted residual, shared with every rebuild the chat has always run (reconnect,
 * first load, Load older): cards minted after a rebuild's fetch left are replaced by
 * that rebuild and come back with their task's next frame or the next sync. A sync
 * that already started armed does not re-arm on them: carrying a dirty bit across an
 * armed rebuild would rebuild again after every busy rebuild, a storm by time where
 * the relative floor removed the storm by size.
 */
export function createLiveCardBound(cap) {
    let armed = false;
    let floor = 0;
    let inherited = false;
    let raisedInFlight = false;
    return {
        isArmed: () => armed,
        /** The offline bootstrap painted sessionStorage: the next sync must rebuild. */
        arm() { armed = true; },
        observe(size) { if (size > floor + cap) armed = true; },
        begin() {
            inherited = armed;
            raisedInFlight = false;
            return inherited;
        },
        beginReplay() { raisedInFlight = armed && !inherited; },
        settle({ rebuilt, size }) {
            if (!rebuilt) return;
            armed = raisedInFlight;
            floor = size;
        },
    };
}

/**
 * perf2 P4 follow-up (double-fetch fix): the debounced post-completion resync
 * behind chat.js's scheduleHistorySync. Finished transitions REPLAYED by
 * syncHistory itself (pass 1 suppressed task summaries, pass 2 / terminal-
 * resolution finishLiveCard) must NOT schedule the resync: those rows just
 * arrived from the canonical history response, so the 700ms refetch was
 * re-downloading the whole window after EVERY history load (Main bootstrap,
 * project open, Load-older, reconnect rebuild). A LIVE completion — a WS
 * frame arriving outside any replay — must keep scheduling a REAL fetch
 * [GPT#12]: a lost task_done is healed only by refetching.
 */
export function createHistoryResyncScheduler({
    isReplayActive,
    run,
    debounceMs = 700,
    setTimer = (fn, ms) => setTimeout(fn, ms),
    clearTimer = (id) => clearTimeout(id),
}) {
    let timer = null;
    return {
        /**
         * @param {boolean} keepPending an armed full rebuild is a DEADLINE, not a
         * best-effort refetch: it is the transaction a reconnect runs, and the live-card
         * bound (issue #135) is only a bound if it actually happens. While one is armed,
         * a later completion must not push the pending run out again — completions
         * arriving faster than the debounce would otherwise starve it for as long as the
         * traffic lasts, which is exactly the busy session the bound exists for. The
         * deadline has a second hole the caller closes: a run that lands while an older
         * history request is still in flight only JOINS it, spending the timer on a
         * window fetched before the arm, so the caller re-arms when such a run settles
         * with the bound still armed.
         */
        schedule(keepPending = false) {
            if (isReplayActive()) return false;
            if (timer != null) {
                if (keepPending) return true;
                clearTimer(timer);
            }
            timer = setTimer(() => {
                timer = null;
                run();
            }, debounceMs);
            return true;
        },
        cancel() {
            if (timer == null) return;
            clearTimer(timer);
            timer = null;
        },
    };
}

/**
 * Insert a top-level timeline node chronologically while keeping typing last.
 * Equal timestamps preserve arrival order; timestamp-free nodes append.
 * (Moved verbatim from chat.js — that module sits at its byte ceiling.)
 */
export function insertTimelineNode(messages, node, typing = null) {
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
    if (before) messages.insertBefore(node, before);
    else if (typing && typing.parentNode === messages) messages.insertBefore(node, typing);
    else messages.appendChild(node);
    return { before };
}

/**
 * Sort key for a top-level timeline node: its stamped `data-ts` epoch, or
 * +Infinity for timestamp-free nodes so they keep the historical "append at
 * the end (before typing)" placement of insertTimelineNode.
 */
export function timelineNodeSortKey(node) {
    const raw = node?.dataset?.ts;
    const ts = raw == null || raw === '' ? NaN : Number(raw);
    return Number.isFinite(ts) ? ts : Infinity;
}

/**
 * Stable chronological order for a batch of collected timeline nodes: key is
 * the stamped ts, tie-break is collection (arrival) order. This reproduces
 * insertTimelineNode's semantics over an initially-empty feed — equal
 * timestamps preserve arrival order (chat_chronology pin), undated nodes land
 * at the end in arrival order.
 */
export function orderBatchNodes(nodes) {
    return nodes
        .map((node, index) => ({ node, index, key: timelineNodeSortKey(node) }))
        .sort((a, b) => {
            if (a.key === b.key) return a.index - b.index;
            return a.key < b.key ? -1 : 1;
        })
        .map((entry) => entry.node);
}

/**
 * One rebuildAll replay batch: collects top-level nodes destined for the feed
 * (instead of per-row live-DOM chronological insertion), remembers which live
 * cards need their one final meta/count/layout pass, and defers the per-frame
 * typing-indicator write to a single application after the mount (the header
 * badge is written only by chat.js's status reducer, once after the batch).
 *
 * mount() performs the ONE DOM insertion of the whole replay: a stable sort,
 * one detached fragment, one insertBefore ahead of the typing indicator
 * (typing stays last). No awaits happen between the feed clearing and this
 * mount — the caller keeps the whole section synchronous [GPT#14].
 */
export function createRebuildBatch(doc = null) {
    const nodes = [];
    const seen = new Set();
    const touched = new Set();
    let holding = null;
    return {
        touched,
        typingHidden: false,
        collect(node) {
            if (!node || seen.has(node)) return;
            seen.add(node);
            nodes.push(node);
            // Parent collected nodes in arrival (chronological) order inside a
            // detached holding fragment so adjacency-sensitive consumers
            // (chat_media gallery grouping) observe the same feed shape during
            // a rebuild replay as on the live feed. mount() reparents them.
            const ownerDoc = doc || node.ownerDocument || globalThis.document;
            if (ownerDoc?.createDocumentFragment) {
                holding = holding || ownerDoc.createDocumentFragment();
                holding.appendChild(node);
            }
        },
        touch(record) {
            if (record) touched.add(record);
        },
        mount(messages, typing = null) {
            if (!messages) return;
            const ordered = orderBatchNodes(nodes);
            const ownerDoc = doc || messages.ownerDocument || globalThis.document;
            const fragment = ownerDoc.createDocumentFragment();
            for (const node of ordered) fragment.appendChild(node);
            if (typing && typing.parentNode === messages) messages.insertBefore(fragment, typing);
            else messages.appendChild(fragment);
        },
    };
}

// Small, bounded projection used to decide whether an existing live-card
// mutation actually changed its connected presentation. Timeline rows and
// review groups report their own changes, so this never serializes a whole card.
export function captureLiveCardProjection(record) {
    const root = record?.root;
    if (!root?.isConnected) return null;
    return [
        root.parentNode, root.previousElementSibling, root.className,
        root.dataset?.finished, root.dataset?.expanded, root.dataset?.subagentRole,
        record.phaseEl?.hidden, record.phaseEl?.className, record.phaseEl?.textContent,
        record.titleEl?.textContent, record.activityEl?.textContent,
        record.metaEl?.innerHTML, record.countEl?.hidden, record.countEl?.textContent,
        record.inlineTypingEl?.style?.display, record.toggleEl?.textContent,
        record.summaryButtonEl?.getAttribute?.('aria-expanded'),
        root.querySelector?.('.chat-live-actions')?.innerHTML || '',
    ];
}

export function liveCardProjectionChanged(before, record) {
    const after = captureLiveCardProjection(record);
    if (!before || !after) return before !== after;
    return before.some((value, index) => value !== after[index]);
}

export function syncLiveCardToggle(record) {
    if (!record?.toggleEl) return;
    const expanded = record.root.dataset.expanded === '1';
    const text = expanded ? 'Hide details' : 'Show details';
    const ariaExpanded = expanded ? 'true' : 'false';
    if (record.toggleEl.textContent !== text) record.toggleEl.textContent = text;
    if (record.summaryButtonEl?.getAttribute('aria-expanded') !== ariaExpanded) {
        record.summaryButtonEl?.setAttribute('aria-expanded', ariaExpanded);
    }
}

// Incremental timeline DOM writes share the Chat viewport boundary but own no
// scroll state. Keeping them here also keeps the byte-capped instance factory
// focused on event projection rather than HTML replacement mechanics.
export function createLiveCardTimelineRenderer({ withStableViewport, buildTimelineItemHtml }) {
    const defer = (record) => {
        if (!record?.isSubagent || record.root?.dataset?.expanded === '1') return false;
        record._timelineDirty = true;
        return true;
    };
    const render = (record) => {
        if (defer(record)) return false;
        record._timelineDirty = false;
        return withStableViewport(() => {
            const el = record.timelineEl;
            const html = record.items.map((item) => buildTimelineItemHtml(item, record)).join('');
            if (el.innerHTML === html) return false;
            const pinned = el.scrollHeight - el.scrollTop - el.clientHeight <= 24;
            const prevTop = el.scrollTop;
            el.innerHTML = html;
            el.scrollTop = pinned ? el.scrollHeight : prevTop;
            return Boolean(el.isConnected);
        });
    };
    const nodeFor = (item, record) => {
        const doc = record.timelineEl?.ownerDocument || globalThis.document;
        const wrapper = doc.createElement('div');
        wrapper.innerHTML = buildTimelineItemHtml(item, record).trim();
        return wrapper.firstElementChild;
    };
    const append = (item, record) => {
        if (defer(record)) return false;
        if (record._timelineDirty) return render(record);
        const pinned = record.timelineEl.scrollHeight
            - record.timelineEl.scrollTop - record.timelineEl.clientHeight <= 24;
        const node = nodeFor(item, record);
        if (!node) return false;
        record.timelineEl.appendChild(node);
        if (record.root.dataset.expanded === '1' && pinned) {
            record.timelineEl.scrollTop = record.timelineEl.scrollHeight;
        }
        return Boolean(record.timelineEl.isConnected);
    };
    const replace = (item, record, current) => {
        if (defer(record)) return false;
        if (record._timelineDirty || !current) return render(record);
        const node = nodeFor(item, record);
        if (!node || node.outerHTML === current.outerHTML) return false;
        record.timelineEl.replaceChild(node, current);
        return Boolean(record.timelineEl.isConnected);
    };
    return {
        renderLiveCardTimeline: render,
        appendTimelineItem: append,
        patchLastTimelineItem: (item, record) => replace(
            item, record, record.timelineEl.lastElementChild,
        ),
        patchTimelineItemAt: (item, record) => {
            const key = String(item.lineKey || '').replace(/[^A-Za-z0-9_-]/g, '');
            const current = key
                ? record.timelineEl.querySelector(`[data-live-line-key="${key}"]`) : null;
            return replace(item, record, current);
        },
    };
}

// "Load older" quota escalation ladder (perf2 P4.5): the DEFAULT request sends
// no quota params (the server window governs); each click asks for explicitly
// larger n_human/n_progress until the server-side caps (1500/600).
export const LOAD_OLDER_QUOTA_STEPS = [
    { n_human: 400, n_progress: 240 },
    { n_human: 1500, n_progress: 600 },
];

/** Next explicit-quota step above `current` (null = server default window). */
export function nextQuotaEscalation(current = null) {
    const currentHuman = Number(current?.n_human) || 0;
    return LOAD_OLDER_QUOTA_STEPS.find((step) => step.n_human > currentHuman) || null;
}

/**
 * Presentation state for the top-of-feed "Load older" control, driven by the
 * SERVER's window truncation verdict (window.complete / window.truncated_by,
 * perf2 P3.2 [Fable#2]) — never by a client guess:
 * - hidden: the window is complete (or the server predates the field);
 * - button: quota-truncated AND a larger explicit quota is still available;
 * - notice: nothing more can be loaded from here — the honest boundary text
 *   names BOTH the on-disk archive floor and the subagent lineage cap
 *   [GPT#11], so a short-history user is never told about phantom archives.
 */
export function loadOlderControlState(windowInfo = null, quota = null) {
    if (!windowInfo || typeof windowInfo !== 'object' || windowInfo.complete === true) {
        return { mode: 'hidden', label: '' };
    }
    const causes = Array.isArray(windowInfo.truncated_by)
        ? windowInfo.truncated_by.map(String)
        : [];
    if (causes.includes('quota') && nextQuotaEscalation(quota)) {
        return { mode: 'button', label: 'Load older messages' };
    }
    return {
        mode: 'notice',
        label: 'Older messages stay in on-disk archives, and deep subagent lineage '
            + 'is capped per window — this view is at its maximum depth.',
    };
}

/**
 * Timeline viewport anchors (extracted verbatim from chat.js at the byte
 * ratchet): capture the first visible timestamped node and restore its exact
 * offset after a mutation. Pure over the passed feed element.
 */
export function createTimelineAnchors({ messagesDiv, liveCardRecords }) {
    function captureVisibleTimelineAnchor(excludeNode = null) {
        // The Load-older control is excluded like .typing-bubble [GPT#13]:
        // anchoring must land on the first visible TIMESTAMPED node, or a
        // Load-older restore would pin the button itself and drift the view.
        const nodes = Array.from(messagesDiv.children).filter(
            (node) => node !== excludeNode
                && !excludeNode?.contains?.(node)
                && !node.classList.contains('typing-bubble')
                && !node.classList.contains('chat-load-older')
        );
        const messagesRect = messagesDiv.getBoundingClientRect();
        const topNode = nodes.find((item) => {
            const rect = item.getBoundingClientRect();
            return rect.bottom > messagesRect.top && rect.top < messagesRect.bottom;
        }) || null;
        if (!topNode) return null;

        // A live-card can span several screens while the reader is inside a
        // child summary or timeline line. Preserve that visible boundary, not
        // merely the root card whose own top may be far above the viewport.
        let node = topNode;
        if (topNode.classList.contains('chat-live-card')) {
            const selector = [
                '.chat-live-card',
                '[data-live-summary-button]',
                '[data-live-title]',
                '[data-live-activity]',
                '[data-live-meta]',
                '.chat-live-actions',
                '.chat-live-line',
                '[data-review-section]',
                '[data-review-section-toggle]',
                '[data-review-hydrate-status]',
                '[data-review-group]',
                '[data-review-attempt]',
                '[data-review-attempt-detail]',
                '.chat-live-project-card-btn',
            ].join(',');
            const candidates = [topNode, ...topNode.querySelectorAll(selector)]
                .map((candidate) => {
                    let depth = 0;
                    let parent = candidate === topNode ? null : candidate.parentElement;
                    while (parent && topNode.contains(parent) && parent !== topNode) {
                        depth += 1;
                        parent = parent.parentElement;
                    }
                    return { node: candidate, rect: candidate.getBoundingClientRect(), depth };
                })
                .filter(({ node: candidate, rect }) => candidate.getClientRects().length
                    && rect.width > 0
                    && rect.height > 0
                    && rect.bottom > messagesRect.top
                    && rect.top < messagesRect.bottom);
            const belowTop = candidates
                .filter(({ rect }) => rect.top >= messagesRect.top)
                .sort((a, b) => (a.rect.top - b.rect.top) || (b.depth - a.depth));
            const crossing = candidates
                .filter(({ rect }) => rect.top <= messagesRect.top && rect.bottom > messagesRect.top)
                .sort((a, b) => b.depth - a.depth);
            node = belowTop[0]?.node || crossing[0]?.node || topNode;
        }

        const cardChain = [];
        let card = node.classList.contains('chat-live-card')
            ? node
            : node.closest?.('.chat-live-card');
        while (card && messagesDiv.contains(card)) {
            cardChain.push({
                node: card,
                taskId: card.dataset?.taskId || '',
                offset: card.getBoundingClientRect().top - messagesRect.top,
            });
            card = card.parentElement?.closest?.('.chat-live-card') || null;
        }

        const ts = topNode.dataset?.ts || '';
        const anchorRole = [
            '[data-live-summary-button]',
            '[data-live-title]',
            '[data-live-activity]',
            '[data-live-meta]',
            '.chat-live-actions',
            '.chat-live-project-card-btn',
        ].find((candidate) => node.matches?.(candidate)) || '';
        return {
            node,
            cardChain,
            lineKey: node.matches?.('.chat-live-line') ? (node.dataset?.liveLineKey || '') : '',
            anchorRole,
            topNode,
            clientMessageId: topNode.dataset?.clientMessageId || '',
            ts,
            ordinal: ts ? nodes.filter((item) => item.dataset?.ts === ts).indexOf(topNode) : -1,
            offset: node.getBoundingClientRect().top - messagesRect.top,
            topOffset: topNode.getBoundingClientRect().top - messagesRect.top,
        };
    }

    function restoreVisibleTimelineAnchor(anchor) {
        if (!anchor) return false;
        const isRendered = (node) => {
            if (!node?.isConnected || !messagesDiv.contains(node)) return false;
            const rect = node.getBoundingClientRect();
            return node.getClientRects().length > 0 && rect.width > 0 && rect.height > 0;
        };
        const restoreNode = (node, offset) => {
            if (!isRendered(node)) return false;
            const currentOffset = node.getBoundingClientRect().top
                - messagesDiv.getBoundingClientRect().top;
            messagesDiv.scrollTop += currentOffset - offset;
            return true;
        };

        if (restoreNode(anchor.node, anchor.offset)) return true;

        const cardChain = Array.isArray(anchor.cardChain) && anchor.cardChain.length
            ? anchor.cardChain
            : [];
        const resolveCard = (entry) => {
            if (isRendered(entry?.node)) return entry.node;
            if (!entry?.taskId) return null;
            const record = liveCardRecords.get(entry.taskId);
            return isRendered(record?.root) ? record.root : null;
        };
        const ownerCard = resolveCard(cardChain[0]);
        if (ownerCard && anchor.lineKey) {
            const line = Array.from(ownerCard.querySelectorAll('.chat-live-line'))
                .find((candidate) => candidate.dataset?.liveLineKey === anchor.lineKey
                    && candidate.closest('.chat-live-card') === ownerCard);
            if (restoreNode(line, anchor.offset)) return true;
        }
        if (ownerCard && anchor.anchorRole) {
            const roleNode = Array.from(ownerCard.querySelectorAll(anchor.anchorRole))
                .find((candidate) => candidate.closest('.chat-live-card') === ownerCard);
            if (restoreNode(roleNode, anchor.offset)) return true;
        }
        for (const entry of cardChain) {
            if (restoreNode(resolveCard(entry), entry.offset)) return true;
        }

        let node = isRendered(anchor.topNode) ? anchor.topNode : null;
        if (!node && anchor.clientMessageId) {
            node = Array.from(messagesDiv.children).find(
                (item) => item.dataset?.clientMessageId === anchor.clientMessageId
            ) || null;
        }
        if (!node && anchor.ts) {
            const matches = Array.from(messagesDiv.children).filter((item) => item.dataset?.ts === anchor.ts);
            node = matches[anchor.ordinal] || matches[0] || null;
        }
        return restoreNode(node, anchor.topOffset ?? anchor.offset);
    }

    return { captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor };
}

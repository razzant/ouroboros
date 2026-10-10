// Chat timeline ordering, keyed item reconciliation and reading anchors.
// These helpers own no history source, navigation or task authority.
import { compareHistoryPosition } from './chat_history_replay.js';
import { appendDelegatedItem, reconcileDelegatedItems } from './delegated_activity.js';

const nodePosition = node => node?.dataset?.historySource
    ? { source: node.dataset.historySource, offset: Number(node.dataset.historyOffset) } : null;

// A feed holding only chrome has no conversation: the history control, the typing
// indicator, Main's empty-state greeting and an ephemeral notice (the reconnect
// banner). The loading state and the greeting share this one rule.
export const feedIsEmpty = messages => Array.from(messages.children).every(node =>
    node.classList.contains('chat-load-older') || node.classList.contains('typing-bubble')
    || node.classList.contains('chat-empty-welcome') || node.dataset.ephemeral === '1');

/**
 * History chrome only; the chat instance retains navigation and reading state.
 *
 * `Load more history` only ever loads OLDER messages and shows only while there
 * are older ones, including those a newer read revealed behind the reader's chain
 * (owner decisions 2026-09-14, 2026-10-05). Whether a newer page is
 * cached is a fact about the bounded page cache, not about what the reader can
 * see, so it never drives this button: a released newer page returns through a
 * positive gesture at an unambiguous island edge, and the floating scroll-to-latest
 * button remains the only return-to-present control.
 */
export function createHistoryControls(messagesDiv, statusHost = null) {
    const doc = messagesDiv.ownerDocument;
    const root = doc.createElement('div');
    root.className = 'chat-load-older';
    const button = doc.createElement('button');
    button.type = 'button';
    button.className = 'chat-load-older-btn';
    const note = doc.createElement('span');
    note.className = 'chat-load-older-note';
    root.append(button, note);
    // The recent-window read is the chat's own request, not a pager page, so its
    // in-flight and failed states are carried here and drawn by the same control
    // (issue #1102: an empty feed under a green header read as a dead app, and a
    // failed read looked identical to a slow one).
    let recent = null;
    return {
        olderButton: button,
        // Only an EMPTY feed (or a failure already on screen) gets the loading
        // state: an ordinary refresh never puts chrome over a painted transcript.
        // Returns whether there is a state to draw, so a painted feed costs nothing.
        beginRecent() {
            if (recent?.error || feedIsEmpty(messagesDiv)) recent = { loading: true };
            return Boolean(recent);
        },
        // A failure is shown where the loading state was; elsewhere the reader
        // keeps the transcript they have and the next sync reconciles it.
        endRecent(error = null) { recent = error ? { error } : null; },
        recentFailed: () => Boolean(recent?.error),
        render(snapshot, coverage = {}, approximate = false) {
            const error = recent?.error || snapshot.error;
            const hydrating = Boolean(recent?.loading);
            const loading = hydrating || Boolean(snapshot.loading);
            const changedView = error?.body?.reason_code === 'history_view_changed';
            const incomplete = coverage.gaps === true;
            let noteText = error ? 'Some saved history could not be loaded.'
                : hydrating && feedIsEmpty(messagesDiv) ? ''
                : incomplete ? 'Some saved history is not loaded. Shown messages may have gaps.'
                : !hydrating && coverage.complete ? 'Beginning of saved history' : '';
            if (approximate) noteText += `${noteText ? ' ' : ''}Saved position could not be restored exactly.`;
            // One note moves into persistent chrome when it describes the reading
            // window; the ordinary beginning marker belongs at the feed's start.
            const host = statusHost && noteText && (error || incomplete || approximate) ? statusHost : root;
            if (note.parentNode !== host) host.appendChild(note);
            note.classList.toggle('chat-history-status', host === statusHost);
            const buttonHidden = !error && !snapshot.canOlder && !(coverage.horizonGap && !snapshot.canNewer) && !hydrating;
            const fields = [
                [button, { textContent: hydrating ? 'Loading saved history…' : loading ? 'Loading…'
                    : changedView ? 'Refresh history' : error ? 'Retry loading messages' : 'Load more history',
                    disabled: loading, hidden: buttonHidden }],
                [note, { textContent: noteText, hidden: !noteText }],
                [root, { hidden: buttonHidden && (host !== root || !noteText) }],
            ];
            for (const [node, values] of fields) {
                for (const [key, value] of Object.entries(values)) if (node[key] !== value) node[key] = value;
            }
            if (hydrating) root.setAttribute('aria-busy', 'true'); else root.removeAttribute('aria-busy');
            if ((snapshot.initialized || error || hydrating) && !root.isConnected) messagesDiv.prepend(root);
            root.classList.toggle('has-gaps', incomplete || Boolean(error));
            return { ...coverage, complete: Boolean(coverage.complete && !error),
                error: Boolean(error), note: noteText, truncated_by: incomplete ? ['unloaded'] : [] };
        },
    };
}

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
// A focused text control keeps its own caret; Chromium mirrors it into the
// document Selection, so clearing and rebuilding document ranges around a
// timeline move collapses that caret (a typed wait-picker draft lost its
// selection on every reconnect). While such a control is focused, the document
// ranges are that mirror: leave them alone and restore the control's caret.
function textControlCaret(active) {
    const tag = String(active?.tagName || active?.nodeName || '').toUpperCase();
    if (tag !== 'INPUT' && tag !== 'TEXTAREA') return null;
    try {
        const { selectionStart, selectionEnd, selectionDirection } = active;
        if (selectionStart == null || selectionEnd == null) return null;
        return { selectionStart, selectionEnd, selectionDirection: selectionDirection || 'none' };
    } catch {
        return null; // input types without a caret (number, email, ...) throw on read
    }
}

function restoreTextControlCaret(active, caret) {
    if (!caret || typeof active?.setSelectionRange !== 'function') return;
    try {
        active.setSelectionRange(caret.selectionStart, caret.selectionEnd, caret.selectionDirection);
    } catch {
        // The control changed type or lost its value between capture and restore.
    }
}

export function insertTimelineNode(messages, node, typing = null) {
    const rawNodeTs = node?.dataset?.ts;
    const nodeTs = rawNodeTs == null || rawNodeTs === '' ? NaN : Number(rawNodeTs);
    let before = null;
    if (Number.isFinite(nodeTs)) {
        for (const child of Array.from(messages?.children || [])) {
            if (child === node || child === typing) continue;
            const rawChildTs = child?.dataset?.ts;
            const childTs = rawChildTs == null || rawChildTs === '' ? NaN : Number(rawChildTs);
            if (Number.isFinite(childTs) && (childTs > nodeTs
                    || childTs === nodeTs && compareHistoryPosition(nodePosition(child), nodePosition(node)) > 0)) {
                before = child;
                break;
            }
        }
    }
    const target = before || (typing?.parentNode === messages ? typing : null);
    if (node.parentNode === messages && node.nextElementSibling === target) return { before };
    const doc = messages.ownerDocument;
    const active = doc?.activeElement;
    const caret = textControlCaret(active);
    const selection = caret ? null : doc?.getSelection?.();
    const ranges = Array.from({ length: selection?.rangeCount || 0 }, (_, index) => {
        const range = selection.getRangeAt(index);
        return [range.startContainer, range.startOffset, range.endContainer, range.endOffset];
    });
    if (target) messages.insertBefore(node, target);
    else messages.appendChild(node);
    if (active && node.contains?.(active) && doc.activeElement !== active) active.focus({ preventScroll: true });
    if (caret) restoreTextControlCaret(active, caret);
    if (ranges.length && ranges.every(([start, , end]) => start.isConnected && end.isConnected)) {
        selection.removeAllRanges();
        for (const [start, startOffset, end, endOffset] of ranges) {
            const range = doc.createRange();
            range.setStart(start, startOffset); range.setEnd(end, endOffset); selection.addRange(range);
        }
    }
    return { before };
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
function timelineItemForAnchor(record, anchor) {
    if (!anchor?.lineKey) return null;
    const owner = anchor.cardChain?.[0]?.taskId;
    if (owner && record.groupId && owner !== record.groupId) return null;
    // Mounted keys preserve focus/selection in one session; the row or evolving
    // lifecycle identity survives a fresh card built from saved history.
    if (anchor.lineHistoryId) return record.items.find(item => item.historyId === anchor.lineHistoryId) || null;
    if (anchor.lineLifecycleKey) return record.items.find(item => item.dedupeKey === anchor.lineLifecycleKey) || null;
    return record.items.find(item => item.lineKey === anchor.lineKey) || null;
}

export function createLiveCardTimelineRenderer({ withStableViewport, buildTimelineItemHtml, isReplayActive = () => false }) {
    // Remember generated markup, not the enhanced DOM: a timestamp update must
    // not undo markdown controls or replace a body the reader has selected.
    const rendered = new WeakMap();
    const markup = (node) => node.outerHTML ?? node.nodeValue;
    const remember = (node) => {
        rendered.set(node, markup(node));
        for (const child of Array.from(node.childNodes || [])) remember(child);
        return node;
    };
    const patchNode = (current, next) => {
        const source = markup(next);
        if (rendered.get(current) === source) return false;
        if (markup(current) === source) {
            remember(current);
            return false;
        }
        if (current.nodeType !== next.nodeType || current.nodeName !== next.nodeName) {
            current.parentNode.replaceChild(remember(next), current);
            return true;
        }
        if (current.nodeType !== 1) {
            current.nodeValue = next.nodeValue;
        } else {
            for (const attr of Array.from(current.attributes)) {
                if (!next.hasAttribute(attr.name)) current.removeAttribute(attr.name);
            }
            for (const attr of Array.from(next.attributes)) {
                if (current.getAttribute(attr.name) !== attr.value) current.setAttribute(attr.name, attr.value);
            }
            const children = Array.from(next.childNodes);
            children.forEach((child, index) => {
                const own = current.childNodes[index];
                if (own) patchNode(own, child);
                else current.appendChild(remember(child));
            });
            while (current.childNodes.length > children.length) current.lastChild.remove();
        }
        rendered.set(current, source);
        return true;
    };
    const defer = (record) => {
        if (!isReplayActive() && (!record?.isSubagent || record.root?.dataset?.expanded === '1')) return false;
        record._timelineDirty = true;
        return true;
    };
    const nodeFor = (item, record) => {
        const doc = record.timelineEl?.ownerDocument || globalThis.document;
        const wrapper = doc.createElement('div');
        wrapper.innerHTML = buildTimelineItemHtml(item, record).trim();
        return wrapper.firstElementChild;
    };
    const render = (record) => {
        if (defer(record)) return false;
        record._timelineDirty = false;
        reconcileDelegatedItems(record); // replay, page release and reorders re-project per seq
        return withStableViewport(() => {
            const el = record.timelineEl;
            const pinned = el.scrollHeight - el.scrollTop - el.clientHeight <= 24;
            const prevTop = el.scrollTop, newest = el.lastElementChild;
            const byKey = new Map(Array.from(el.children).map((node) => [node.dataset.liveLineKey, node]));
            const active = el.ownerDocument?.activeElement;
            const caret = textControlCaret(active);
            const selection = caret ? null : el.ownerDocument?.getSelection?.();
            const ranges = Array.from({ length: selection?.rangeCount || 0 }, (_, index) => {
                const range = selection.getRangeAt(index);
                return [range.startContainer, range.startOffset, range.endContainer, range.endOffset];
            });
            let changed = false;
            let moved = false;
            record.items.forEach((item, index) => {
                const next = nodeFor(item, record);
                if (!next) return;
                const current = byKey.get(String(item.lineKey || ''));
                const node = current || remember(next);
                if (current) {
                    byKey.delete(String(item.lineKey || ''));
                    changed = patchNode(current, next) || changed;
                }
                if (el.children[index] !== node) {
                    moved = moved || node.parentNode === el;
                    el.insertBefore(node, el.children[index] || null);
                    changed = true;
                }
            });
            for (const node of byKey.values()) {
                node.remove();
                changed = true;
            }
            if (moved) {
                if (el.contains(active) && el.ownerDocument.activeElement !== active) active.focus({ preventScroll: true });
                if (caret) restoreTextControlCaret(active, caret);
                const intact = ranges.filter(([start, , end]) => start.isConnected && end.isConnected);
                if (intact.length) {
                    selection.removeAllRanges();
                    for (const [start, startOffset, end, endOffset] of intact) {
                        const range = el.ownerDocument.createRange();
                        range.setStart(start, startOffset);
                        range.setEnd(end, endOffset);
                        selection.addRange(range);
                    }
                }
            }
            // Only a new newest line keeps a pinned timeline at its end; a disclosure
            // or late full output keeps the line being read where it is.
            if (changed) el.scrollTop = pinned && el.lastElementChild !== newest ? el.scrollHeight : prevTop;
            return changed && Boolean(el.isConnected);
        });
    };
    const append = (item, record) => {
        if (defer(record)) return false;
        if (record._timelineDirty) return render(record);
        const pinned = record.timelineEl.scrollHeight
            - record.timelineEl.scrollTop - record.timelineEl.clientHeight <= 24;
        const node = nodeFor(item, record);
        if (!node) return false;
        record.timelineEl.appendChild(remember(node));
        if (record.root.dataset.expanded === '1' && pinned) {
            record.timelineEl.scrollTop = record.timelineEl.scrollHeight;
        }
        return Boolean(record.timelineEl.isConnected);
    };
    const replace = (item, record, current) => {
        if (defer(record)) return false;
        if (record._timelineDirty || !current) return render(record);
        const node = nodeFor(item, record);
        if (!node || !patchNode(current, node)) return false;
        return Boolean(record.timelineEl.isConnected);
    };
    return {
        renderLiveCardTimeline: render,
        appendTimelineItem: append,
        patchLastTimelineItem: (item, record) => replace(
            item, record, record.timelineEl.lastElementChild,
        ),
        patchTimelineItemAt: (item, record) => {
            const current = Array.from(record.timelineEl.children)
                .find((node) => node.dataset.liveLineKey === String(item.lineKey || ''));
            return replace(item, record, current);
        },
    };
}

// A live card's header chrome: a saved role inside its owner card.
const CARD_HEADER = [
    '[data-live-summary-button]',
    '[data-live-title]',
    '[data-live-activity]',
    '[data-live-meta]',
    '.chat-live-actions',
    '.chat-live-project-card-btn',
];

// The feed's bounded, separately scrolling boxes: a card timeline, a fetched
// full output, a Review attempt detail. Each clips what it holds.
const BOUNDED_BOX = '[data-live-timeline], .chat-live-line-body-full, [data-review-attempt-detail]';

// A node's own bounded reading box: an expanded line's fetched full output or
// a Review attempt detail. The owner card timeline is found from the node.
function innerBox(node) {
    if (node?.matches?.('[data-review-attempt-detail]')) return node;
    return node?.matches?.('.chat-live-line') ? node.querySelector?.(':scope > .chat-live-line-body-full') || null : null;
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
        let topNode = nodes.find((item) => {
            const rect = item.getBoundingClientRect();
            return rect.bottom > messagesRect.top && rect.top < messagesRect.bottom;
        }) || null;
        if (!topNode) return null;

        // A live-card can span several screens while the reader is inside a
        // child summary or timeline line. Preserve that visible boundary, not
        // merely the root card whose own top may be far above the viewport.
        let node = topNode;
        if (!topNode.dataset?.historyId && !topNode.classList.contains('chat-live-card')) {
            // Media owns a layout wrapper around keyed bubbles. Save the
            // physical child being read, not an unidentifiable wrapper.
            node = [...topNode.querySelectorAll('[data-history-id]')]
                .find(child => child.getBoundingClientRect().bottom > messagesRect.top) || topNode;
        }
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
            // Header chrome heads what it discloses: when it is all that precedes
            // an expanded line, that line is being read, and anchoring it lets a
            // reopen expand it again instead of saving the header alone.
            const header = candidate => candidate.classList.contains('chat-live-card') || CARD_HEADER.some(role => candidate.matches(role));
            const content = header(node) ? belowTop.find(({ node: candidate }) => !header(candidate))?.node : null;
            if (content?.matches('.chat-live-line') && content.dataset?.expanded === '1') node = content;
            if (node === topNode && topNode.getBoundingClientRect().top < messagesRect.top) {
                // The card's visible part holds nothing anchorable (a wait row, a
                // block without work): keep the reader's view of what FOLLOWS the
                // card. Pinning the card's own top, far above the viewport, would let
                // the card's shrink or growth move the content the reader is on.
                const following = nodes.find((item) => {
                    if (item === topNode) return false;
                    const rect = item.getBoundingClientRect();
                    return rect.top >= messagesRect.top && rect.top < messagesRect.bottom;
                });
                if (following) { topNode = following; node = following; }
            }
        }

        // A visible bounded full output or Review detail the reader entered
        // (the top edge is inside it) or scrolled is the place being read; its
        // line or detail anchors, so reopening expands and scrolls it again.
        // Visible means one common part inside the feed and every bounded box
        // around it: overlapping each clip at different places shows nothing.
        const shown = (box, rect) => {
            let top = Math.max(rect.top, messagesRect.top), bottom = Math.min(rect.bottom, messagesRect.bottom);
            for (let clip = box.parentElement?.closest?.(BOUNDED_BOX); clip && messagesDiv.contains(clip);
                clip = clip.parentElement?.closest?.(BOUNDED_BOX)) {
                const edge = clip.getBoundingClientRect();
                top = Math.max(top, edge.top); bottom = Math.min(bottom, edge.bottom);
            }
            return top < bottom;
        };
        const reading = Array.from(messagesDiv.querySelectorAll?.('.chat-live-line-body-full, [data-review-attempt-detail]') || [])
            .map(box => ({ box, rect: box.getBoundingClientRect() }))
            .filter(({ box, rect }) => box.getClientRects().length && shown(box, rect)
                && (box.scrollTop > 0 || rect.top < messagesRect.top))
            .sort((a, b) => a.rect.top - b.rect.top)
            .map(({ box }) => (box.matches('[data-review-attempt-detail]') ? box : box.closest('.chat-live-line')))
            .find(owner => owner && nodes.some(item => item.contains(owner)));
        if (reading) {
            node = reading;
            topNode = nodes.find(item => item.contains(reading));
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
        const anchorRole = CARD_HEADER.find((candidate) => node.matches?.(candidate)) || '';
        const lineKey = node.matches?.('.chat-live-line') ? (node.dataset?.liveLineKey || '') : '';
        const lineItem = lineKey ? (liveCardRecords.get(cardChain[0]?.taskId)?.items || [])
            .find(item => item.lineKey === lineKey) : null;
        const lifecycleKey = String(lineItem?.dedupeKey || '');
        return {
            node,
            // The line's physical source selects its page; its immutable row or
            // lifecycle key below selects the line inside that page's mixed card.
            historyId: lineItem?.historyId || lineItem?.sourceHistoryId
                || node.closest?.('[data-history-id]')?.dataset?.historyId
                || topNode.dataset?.historyId || '',
            reviewKey: ['reviewAttemptDetail', 'reviewAttempt', 'reviewGroup', 'reviewSection']
                .find(key => node.dataset?.[key]) || '',
            reviewValue: node.dataset?.reviewAttemptDetail || node.dataset?.reviewAttempt
                || node.dataset?.reviewGroup || node.dataset?.reviewSection || '',
            cardChain,
            lineKey,
            lineHistoryId: lineItem?.historyId || '',
            lineLifecycleKey: (lineItem?.sourceHistoryId
                || lifecycleKey.startsWith('subagent-lifecycle:')
                || lifecycleKey.startsWith('task_done|')
                || lifecycleKey.startsWith('cardrow|')) ? lifecycleKey : '',
            lineExpanded: node.matches?.('.chat-live-line') && node.dataset?.expanded === '1',
            anchorRole,
            topNode,
            clientMessageId: topNode.dataset?.clientMessageId || '',
            ts,
            ordinal: ts ? nodes.filter((item) => item.dataset?.ts === ts).indexOf(topNode) : -1,
            offset: node.getBoundingClientRect().top - messagesRect.top,
            topOffset: topNode.getBoundingClientRect().top - messagesRect.top,
        };
    }

    function restoreVisibleTimelineAnchor(anchor, { exact = false, cardOnly = false } = {}) {
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
        // The saved node found: its owner timeline puts it back at its offset
        // there (that scroll moves the node), then its own box scrolls. A box
        // that is missing, or an own box that cannot reach its scroll, is not
        // exact. A timeline at its scroll limit has done what it can: its node
        // still takes the exact feed offset, and a reflowed earlier row only
        // moves the box around it.
        const restoreSaved = (node) => {
            const timeline = node.closest?.('[data-live-timeline]'), inner = innerBox(node);
            let settled = true;
            if (Number.isFinite(anchor.timelineOffset)) {
                const offset = () => node.getBoundingClientRect().top - timeline.getBoundingClientRect().top;
                if (timeline) timeline.scrollTop += offset() - anchor.timelineOffset;
                const miss = timeline ? offset() - anchor.timelineOffset : NaN;
                settled = Boolean(timeline) && (Math.abs(miss) <= 1 || (miss > 0
                    ? timeline.scrollTop >= timeline.scrollHeight - timeline.clientHeight - 1
                    : timeline.scrollTop <= 0));
            }
            if (Number.isFinite(anchor.innerTop)) {
                if (inner) inner.scrollTop = anchor.innerTop;
                settled = settled && Boolean(inner) && Math.abs(inner.scrollTop - anchor.innerTop) <= 1;
            }
            return restoreNode(node, anchor.offset) && (settled || !exact);
        };

        const historyId = anchor.historyId || anchor.id;
        if (historyId && !anchor.lineKey && !anchor.reviewKey && !anchor.anchorRole) {
            const node = Array.from(messagesDiv.querySelectorAll('[data-history-id]'))
                .find(item => item.dataset.historyId === historyId);
            if (isRendered(node)) return restoreSaved(node);
        }

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
        if (ownerCard && anchor.reviewKey) {
            const node = Array.from(ownerCard.querySelectorAll('[data-review-section], [data-review-group], [data-review-attempt], [data-review-attempt-detail]'))
                .find(item => item.dataset?.[anchor.reviewKey] === anchor.reviewValue);
            if (isRendered(node)) return restoreSaved(node);
        }
        if (ownerCard && anchor.lineKey) {
            const item = timelineItemForAnchor(liveCardRecords.get(cardChain[0]?.taskId) || { items: [] }, anchor);
            const lineKey = item?.lineKey || (anchor.lineHistoryId || anchor.lineLifecycleKey ? '' : anchor.lineKey);
            const line = Array.from(ownerCard.querySelectorAll('.chat-live-line'))
                .find((candidate) => candidate.dataset?.liveLineKey === lineKey
                    && candidate.closest('.chat-live-card') === ownerCard);
            if (isRendered(line)) return restoreSaved(line);
        }
        if (ownerCard && anchor.anchorRole) {
            const roleNode = Array.from(ownerCard.querySelectorAll(anchor.anchorRole))
                .find((candidate) => candidate.closest('.chat-live-card') === ownerCard);
            if (isRendered(roleNode)) return restoreSaved(roleNode);
        }
        if (exact && (anchor.lineKey || anchor.reviewKey || anchor.anchorRole)) return false;
        if (exact && cardChain.length && restoreNode(ownerCard, anchor.offset)) return true;
        if (exact && !anchor.clientMessageId) return false;
        for (const entry of cardChain) {
            if (restoreNode(resolveCard(entry), entry.offset)) return true;
        }
        if (cardOnly) return false;

        let node = isRendered(anchor.topNode) ? anchor.topNode : null;
        if (!node && anchor.clientMessageId) {
            node = Array.from(messagesDiv.children).find(
                (item) => item.dataset?.clientMessageId === anchor.clientMessageId
            ) || null;
        }
        if (!node && anchor.ts && !exact) {
            const matches = Array.from(messagesDiv.children).filter((item) => item.dataset?.ts === anchor.ts);
            node = matches[anchor.ordinal] || matches[0] || null;
        }
        return restoreNode(node, anchor.topOffset ?? anchor.offset);
    }

    function serializeTimelineAnchor(anchor = captureVisibleTimelineAnchor()) {
        if (!anchor) return null;
        const { node, topNode: _topNode, cardChain, ...fields } = anchor;
        // A bounded box scrolls on its own: the node keeps its offset inside its
        // owner timeline, and its own full output or Review detail its scroll.
        const timeline = node?.closest?.('[data-live-timeline]'), inner = node && innerBox(node);
        return { ...fields, cardChain: (cardChain || []).map(({ taskId, offset }) => ({ taskId, offset })),
            ...(timeline && messagesDiv.contains(timeline) ? { timelineOffset:
                node.getBoundingClientRect().top - timeline.getBoundingClientRect().top } : {}),
            ...(inner ? { innerTop: inner.scrollTop } : {}) };
    }

    /** Positioning waits for the data owners of the saved place: the Review
     * detail it names and the full output an expanded line is still fetching. */
    function anchorOwnersReady(anchor, reviewReady) {
        const taskId = anchor?.cardChain?.[0]?.taskId;
        if (anchor?.reviewKey && !reviewReady(taskId)) return false;
        const record = anchor?.lineExpanded && liveCardRecords.get(taskId);
        return !(record && timelineItemForAnchor(record, anchor)?._fetchingFull);
    }
    return { captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor, serializeTimelineAnchor, anchorOwnersReady };
}

/** Update one live timeline item using the same keys the card's producer owns.
 * Historical source-addressed insertions use mergeHistoricalTimelineItem; live
 * lifecycle notes keep their existing in-place semantics and disclosure key.
 */
export function updateLiveTimelineItem(record, summary, { ts, rawTs, syntheticKey, headline, inPlaceByKey }) {
    // A delegated observation is its own source record; its projection is per seq.
    if (summary.activity) return appendDelegatedItem(record, summary, { ts, rawTs, syntheticKey, headline });
    let timelineUpdate = 'none', patchIndex = -1;
    const lastIdx = record.items.length - 1;
    // Full-array dedup keeps routine history syncs from growing Notes.
    const existingIdx = record.items.findIndex((it) => it.dedupeKey === syntheticKey);
    if (existingIdx !== -1 && inPlaceByKey) {
        const it = record.items[existingIdx];
        if (Number.isSafeInteger(it.cardRowRevision)
            && (!Number.isSafeInteger(summary.cardRowRevision) || summary.cardRowRevision <= it.cardRowRevision)) {
            return { timelineUpdate: 'duplicate-skip', patchIndex };
        }
        const patch = {
            cardRowRevision: summary.cardRowRevision,
            phase: summary.phase || it.phase,
            headline: headline || it.headline,
            fullHeadline: summary.fullHeadline || headline || it.fullHeadline,
            body: summary.body || '',
            fullBody: summary.fullBody || summary.body || it.fullBody || '',
            fullRef: summary.fullRef || it.fullRef || '',
            truncated: summary.truncated || it.truncated || false,
            evidenceRef: summary.evidenceRef || it.evidenceRef || null,
            // A call's failure frame replaces its receipt start: the row is
            // content again once it reports an error.
            receipt: Boolean(summary.receipt),
            ts: ts || it.ts,
            // Replay compares the child's current status with older pages.
            // Its source time advances with the live status, not with narration.
            ...(syntheticKey.startsWith('subagent-lifecycle:') && rawTs ? { sourceTs: rawTs } : {}),
        };
        if (Object.entries(patch).some(([key, value]) => it[key] !== value)) {
            Object.assign(it, patch);
            patchIndex = existingIdx;
            timelineUpdate = 'patch-at';
        } else {
            timelineUpdate = 'duplicate-skip';
        }
    } else if (existingIdx === lastIdx && existingIdx !== -1) {
        const it = record.items[existingIdx];
        const patch = {
            ts: ts || it.ts,
            fullHeadline: summary.fullHeadline || it.fullHeadline || it.headline,
            fullBody: summary.fullBody || it.fullBody || it.body,
            fullRef: summary.fullRef || it.fullRef || '',
            truncated: summary.truncated || it.truncated || false,
        };
        if (Object.entries(patch).every(([key, value]) => it[key] === value)) {
            timelineUpdate = 'duplicate-skip';
        } else {
            Object.assign(it, patch);
            it.count += 1;
            timelineUpdate = 'patch-last';
        }
    } else if (existingIdx !== -1) {
        // An older duplicate only refreshes its timestamp.
        const it = record.items[existingIdx];
        it.ts = ts || it.ts;
        timelineUpdate = 'duplicate-skip';
    } else {
        const lineKey = `line-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        record.items.push({
            cardRowRevision: summary.cardRowRevision,
            phase: summary.phase || 'working',
            headline: headline || 'Update',
            fullHeadline: summary.fullHeadline || headline || 'Update',
            body: summary.body || '',
            fullBody: summary.fullBody || summary.body || '',
            fullRef: summary.fullRef || '',
            truncated: summary.truncated || false,
            // A late-review row's exact record link (#1369); null on every other row.
            evidenceRef: summary.evidenceRef || null,
            receipt: Boolean(summary.receipt),
            ts: ts || '',
            sourceTs: rawTs,
            count: 1,
            dedupeKey: syntheticKey,
            lineKey,
        });
        timelineUpdate = 'append';
    }
    return { timelineUpdate, patchIndex };
}

/** The block's folded tool evidence row. Live frames and the host's metrics
 * reach it through the same keyed in-place upsert, so neither route can mint a
 * second row, and the row keeps the position and timestamp of its first frame.
 */
export function upsertToolFoldRow(record, view, ts, rawTs) {
    const syntheticKey = `tools|${record.groupId}`;
    // Stationary: the row keeps the place and the time of the first frame it
    // counted, so a burst of calls never walks it down the timeline.
    const first = !record.items.some((item) => item.dedupeKey === syntheticKey);
    return updateLiveTimelineItem(record, view, {
        ts: first ? ts : '', rawTs, syntheticKey, headline: view.headline, inPlaceByKey: true,
    });
}

// perf2 P4 (RENDER-BATCH): pure helpers behind chat.js's rebuildAll replay
// batch and the "Load older" window escalation. Dependency-free at import time
// so node tests can exercise them directly.

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
        schedule() {
            if (isReplayActive()) return false;
            if (timer != null) clearTimer(timer);
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
 * typing/status writes to a single application after the mount.
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
    return {
        touched,
        status: null,
        typingHidden: false,
        collect(node) {
            if (!node || seen.has(node)) return;
            seen.add(node);
            nodes.push(node);
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
 *
 * `ancestry_depth` (project threads, T0) is a DIFFERENT boundary: part of a
 * FORKED thread's shared past was not read at all — the fork chain hit its
 * depth cap, closed in a cycle, or named an ancestor with no project binding.
 * A larger quota cannot recover it, and the archive/lineage wording would
 * misname where the missing conversation is, so it gets its own sentence
 * (plan A3b: a shared past out of reach is disclosed, never a silent gap).
 * `lens_unavailable` is narrower still: the lens could not be BUILT (the registry
 * was unreadable), so whether this thread HAS a shared past is unknown. It rides
 * alongside `ancestry_depth` and adds its own clause.
 * Causes accumulate: a notice carries ONE sentence per present cause, because
 * a window can be cut by the fork chain and the archive floor at once.
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
    // Causes ACCUMULATE: a forked thread can hit its ancestry cap AND the
    // on-disk archive floor in the same window. Stopping at the first match
    // named one boundary and silently swallowed the other, which is the same
    // silent gap the disclosure exists to prevent — so every present cause
    // contributes its own sentence.
    const sentences = [];
    if (causes.includes('ancestry_depth')) {
        sentences.push(
            'Part of this thread’s shared past could not be read: the fork '
            + 'chain is too deep, or one of its parent threads is unavailable.',
        );
    }
    // A DIFFERENT fact from `ancestry_depth`, and it must not borrow the
    // archive/lineage wording: the lens could not be BUILT at all, so whether this
    // thread even HAS a shared past is unknown rather than known-and-cut. The
    // server sets both causes together, so this refines the sentence.
    if (causes.includes('lens_unavailable')) {
        sentences.push(
            'Its fork history could not be looked up just now, so a shared past '
            + 'may be missing from this view rather than absent.',
        );
    }
    const others = causes.filter(
        (cause) => cause !== 'ancestry_depth' && cause !== 'lens_unavailable',
    );
    if (others.length || !sentences.length) {
        sentences.push(
            'Older messages stay in on-disk archives, and deep subagent lineage '
            + 'is capped per window — this view is at its maximum depth.',
        );
    }
    return { mode: 'notice', label: sentences.join(' ') };
}

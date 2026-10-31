/**
 * One explicit navigation to one place in a room's history (DESIGN "History edges", "Room work status"):
 * the page around a task, or the page a reader left read again, through the chat instance's own history
 * pager and reading intent, and the node such a landing focuses. It keeps no pager, cache, poller or timer
 * of its own; a navigation the reader moved on from (a gesture, another navigation, the room hidden) is
 * dropped before it changes anything.
 */

/**
 * @param {{
 *   pager: { around: Function, revisit: Function, pageOf: (id: string) => ?string },
 *   reading: { claim: () => () => boolean, stick: boolean, nearBottom: () => boolean },
 *   feed: Element,
 *   records: Map<string, { root?: Element, items?: Array<Object>, parentGroupId?: string }>,
 *   present: () => boolean,  // the room is still shown and alive
 *   expand?: (record: Object) => void,  // existing card disclosure/materialization owner
 * }} owners
 */
export function createHistoryReveal({ pager, reading, feed, records, present, expand = () => {} }) {
    // One history read for a navigation (the pager lets a window read in flight finish first): null once
    // the reader has moved on.
    async function read(isCurrent, load, claim = true) {
        const claimed = claim ? reading.claim() : () => true;
        const wanted = () => present() && isCurrent() && claimed();
        const result = await load(wanted).catch((error) => ({ status: 'error', error }));
        return wanted() ? result : null;
    }

    // Where a landing for a task goes: its card, else its own row of the conversation, else the exact
    // row the read named; null when the page holds nothing of it to show.
    function landing(taskId, anchor) {
        const card = records.get(taskId)?.root;
        if (card?.isConnected) return card;
        const own = [...feed.querySelectorAll('[data-task-id]')].find((node) => node.dataset.taskId === taskId);
        if (own) return own;
        const id = anchor?.history_id;
        return id ? [...feed.querySelectorAll('[data-history-id], [data-source-history-id]')]
            .find((node) => node.dataset.historyId === id || node.dataset.sourceHistoryId === id) || null : null;
    }

    // Exact references may name a line in a collapsed child card. Expand through the same card owner
    // that handles its disclosure, then require the actual addressed node. A portal, retained context,
    // or a folded repeat with no original line is not an exact landing.
    function exact(address, id = '') {
        const find = () => [...feed.querySelectorAll('[data-row-address], [data-history-id]')].find(node =>
            (node.dataset.rowAddress === address || id && node.dataset.historyId === id)
            && node.dataset.originProjected !== '1'
            && (!address.startsWith('row:') || !node.closest?.('[data-portal="1"]'))) || null;
        let target = find();
        if (!target && id) {
            const record = [...records.values()].find(record => record.items?.some(item => item.historyId === id));
            const seen = new Set();
            for (let owner = record; owner && !seen.has(owner); owner = records.get(owner.parentGroupId)) {
                if (owner.root?.dataset.portal === '1') return null;
                seen.add(owner); expand(owner);
                if (owner.episodeDisclosure) owner.episodeDisclosure.open = true;
            }
            target = find();
        }
        for (let node = target?.parentElement; node && node !== feed; node = node.parentElement) {
            if (node.matches?.('details')) node.open = true;
            const record = records.get(node.dataset.taskId);
            if (node.matches?.('.chat-live-card') && record) expand(record);
        }
        // A relay deck owns this row's visibility: its caller opens the deck before positioning/focusing it.
        return target && !target.closest?.('[hidden]')
            && (target.getClientRects?.().length !== 0 || target.dataset.deckTucked === '1') ? target : null;
    }

    return {
        exact,
        // Reply's caller claims once before any I/O and carries that intent across destination lookup.
        // Task references keep their work-card semantics; only row references require an exact node.
        // A failed read keeps its typed error (the host's reason code); `left`: a page was drawn without it.
        async reference(address, isCurrent, prepared = null) {
            const result = await read(isCurrent, wanted => pager.around(address, wanted, prepared), false);
            if (!result) return null;
            if (result.status === 'anchor') return result.anchor;
            if (result.status === 'error') return { state: 'unknown', error: result.error };
            if (result.status !== 'applied') return { state: 'unknown' };
            const target = address.startsWith('task:') ? landing(address.slice(5), result.anchor)
                : result.anchor?.row_address === address ? exact(address, result.anchor.history_id) : null;
            return target ? { target } : { state: 'unavailable', left: true };
        },
        // Where a return goes: the passage on screen, the page of the reader's chain that holds it (read again
        // if it was let go meanwhile) and whether the reader was following the present: held to the newest
        // message and still at it (a view moved without a gesture, by focus or find in page, is a passage).
        departure(anchor) {
            const present = reading.stick === true && reading.nearBottom() === true;
            // A card's header names no row of its own: the card stands where its earliest row is (chat.js stamps it).
            const card = anchor?.cardChain?.find((link) => link.node?.dataset?.historySource)?.node.dataset;
            const id = anchor?.historyId || (card ? `${card.historySource}:${card.historyOffset}` : '');
            return { anchor, cursor: (id && pager.pageOf?.(id)) || null, present };
        },
        // The page around a task: `{ target }` to land on; `{ state }` 'absent' (no message of it in this
        // room's history), 'moved' (another room's) or 'unknown' (the read could not tell, or its drawn page
        // shows nothing of it: `left`, the reader's chain was replaced); null when dropped.
        async task(taskId, isCurrent) {
            const result = await read(isCurrent, (wanted) => pager.around(`task:${taskId}`, wanted));
            if (!result) return null;
            const target = result.status === 'applied' ? landing(taskId, result.anchor) : null;
            if (target) return { target };
            if (result.status === 'applied') return { state: 'unknown', left: true };
            return { state: result.status === 'anchor' ? result.anchor.state : 'unknown' };
        },
        // The page a reader left, read again by its cursor: true once it is drawn, false when the read failed
        // for now, 'gone' when it cannot be read again (the room's view changed), null when dropped.
        async revisit(cursor, isCurrent) {
            const result = await read(isCurrent, (wanted) => pager.revisit(cursor, wanted));
            if (!result || result.status === 'applied') return result && true;
            return result.error?.status >= 400 && result.error.status < 500 ? 'gone' : false;
        },
    };
}

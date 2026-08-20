// Visible-timeline anchoring for ONE chat instance. `captureVisibleTimelineAnchor`
// records the boundary the reader is looking at (including the child row inside a
// live card that spans several screens) and `restoreVisibleTimelineAnchor` puts it
// back after a DOM mutation; `isNearBottom` is the follow-the-tail predicate they
// share with the scroll owner. The instance hands over its transcript element and
// live-card records explicitly, so a Main chat and a Project panel anchor
// independently and neither can read the other's nodes.
export function createTimelineAnchors({ messagesDiv, liveCardRecords }) {
    const NEAR_BOTTOM_THRESHOLD_PX = 160;

    function isNearBottom(threshold = NEAR_BOTTOM_THRESHOLD_PX) {
        const remaining = messagesDiv.scrollHeight - messagesDiv.scrollTop - messagesDiv.clientHeight;
        return remaining <= threshold;
    }

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

    return { isNearBottom, captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor };
}

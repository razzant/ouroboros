function reviewNodeKey(node) {
    const dataset = node?.dataset || {};
    if (Object.hasOwn(dataset, 'reviewSection')) return 'section';
    if (Object.hasOwn(dataset, 'reviewHydrateStatus')) return 'hydrate-status';
    if (Object.hasOwn(dataset, 'reviewAuthorDecision')) return 'author-decision';
    if (dataset.reviewOpinion) return `opinion:${dataset.reviewOpinion}`;
    if (dataset.reviewFacts) return `facts:${dataset.reviewFacts}`;
    if (dataset.reviewReadUrl) return `read:${dataset.reviewReadUrl}`;
    if (dataset.reviewGroup) return `group:${dataset.reviewGroup}`;
    if (dataset.reviewGroupToggle) return `group-toggle:${dataset.reviewGroupToggle}`;
    if (dataset.reviewAttempt) return `attempt:${dataset.reviewAttempt}`;
    if (dataset.reviewAttemptDetail) return `attempt-detail:${dataset.reviewAttemptDetail}`;
    if (Object.hasOwn(dataset, 'reviewHistory')) return 'history';
    for (const className of ['chat-review-groups', 'chat-review-attempts']) {
        if (node?.classList?.contains?.(className)) return `class:${className}`;
    }
    return '';
}

const isText = (node) => node?.nodeType === 3;
// The nodes a parent shows: every element and text node (whitespace included, it is
// a space between inline words). A stub without childNodes has elements only.
const shownNodes = (parent) => Array.from(parent?.childNodes || parent?.children || [])
    .filter((node) => node?.nodeType === 1 || isText(node) || node?.tagName);

function syncReviewAttributes(current, desired, preserve = new Set()) {
    for (const attr of Array.from(current.attributes || [])) {
        if (!preserve.has(attr.name) && !desired.hasAttribute(attr.name)) {
            current.removeAttribute(attr.name);
        }
    }
    for (const attr of Array.from(desired.attributes || [])) {
        if (!preserve.has(attr.name) && current.getAttribute?.(attr.name) !== attr.value) current.setAttribute(attr.name, attr.value);
    }
}

/**
 * Make `current`'s child nodes equal `desired`'s in order. Elements keep their identity
 * (a key, else the next unkeyed element of the same tag), so focus, an open disclosure,
 * a scroll position and anything a click is reading survive; text nodes are matched in
 * order and rewritten in place, so a word beside an element is never left stale.
 */
function replaceReviewChildren(current, desired) {
    const wanted = shownNodes(desired);
    if (!wanted.some((node) => !isText(node))) {
        if (current.innerHTML !== desired.innerHTML) current.innerHTML = desired.innerHTML;
        return;
    }
    const available = shownNodes(current);
    const elements = available.filter((node) => !isText(node));
    const texts = available.filter(isText);
    const used = new Set();
    const placed = [];
    for (const want of wanted) {
        let node;
        if (isText(want)) {
            node = texts.find((candidate) => !used.has(candidate));
            if (node) { if (node.data !== want.data) node.data = want.data; } else node = want.cloneNode(false);
        } else {
            const key = reviewNodeKey(want);
            node = elements.find((candidate) => !used.has(candidate) && candidate.tagName === want.tagName
                && reviewNodeKey(candidate) === key);
            if (node) patchReviewElement(node, want); else node = want.cloneNode(true);
        }
        used.add(node);
        placed.push(node);
    }
    // Insert in order, moving an existing node only when it is out of place: a node
    // taken out of the document loses focus, so nothing already in order is touched.
    let cursor = current.firstChild || null;
    for (const node of placed) {
        while (cursor && !used.has(cursor)) cursor = cursor.nextSibling || null;
        if (node === cursor) { cursor = cursor.nextSibling || null; continue; }
        current.insertBefore(node, cursor);
    }
    for (const node of shownNodes(current)) if (!used.has(node)) node.remove();
}

function patchReviewElement(current, desired) {
    // A saved-detail box keeps its reading position across the rewrite of its text.
    const isDetail = Boolean(current?.dataset?.reviewAttemptDetail);
    const scrollTop = isDetail ? current.scrollTop : null;
    // An opened disclosure belongs to the owner: an update rewrites it, never closes it.
    const preserve = current.tagName === 'DETAILS' ? new Set(['open']) : new Set();
    syncReviewAttributes(current, desired, preserve);
    replaceReviewChildren(current, desired);
    if (isDetail && Number.isFinite(scrollTop) && current.scrollTop !== scrollTop) current.scrollTop = scrollTop;
}

export function reconcileReviewElementTree(current, desired) {
    if (!current || !desired || current.tagName !== desired.tagName) return false;
    patchReviewElement(current, desired);
    return true;
}

/**
 * Reconcile the keyed Reviews subtree in place. Stable group/opinion/facts
 * nodes retain focus, an open disclosure and a reading position; only genuine
 * structural additions/removals allocate nodes.
 */
export function reconcileReviewMarkup(host, html) {
    const doc = host?.ownerDocument;
    const template = doc?.createElement?.('template');
    if (!host || !template || !template.content) {
        if (host) host.innerHTML = html;
        return false;
    }
    template.innerHTML = String(html || '').trim();
    const desired = template.content.firstElementChild;
    const current = Array.from(host.children || [])
        .find((child) => Object.hasOwn(child.dataset || {}, 'reviewSection')) || null;
    if (!desired) {
        if (current) current.remove();
        return true;
    }
    if (!current) {
        host.replaceChildren(desired);
        return true;
    }
    reconcileReviewElementTree(current, desired);
    for (const child of Array.from(host.children || [])) {
        if (child !== current) child.remove();
    }
    return true;
}

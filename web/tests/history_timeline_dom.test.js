import assert from 'node:assert/strict';
import test from 'node:test';
import { bindLiveCardTimeline, buildTimelineItemHtml, selectionInside } from '../modules/chat_activity.js';
import { createLiveCardTimelineRenderer, createTimelineAnchors, updateLiveTimelineItem } from '../modules/chat_render_batch.js';
import { historyStamps, mergeHistoricalTimelineItem } from '../modules/chat_history_replay.js';

// A real tree (including text nodes), with explicit removal effects on focus.
// The injected renderer builds JSON trees so this fixture needs no HTML parser.
class Node {
    constructor(doc, [name, attrs = {}, content = []]) {
        this.ownerDocument = doc;
        this.nodeName = name;
        this.nodeType = name === '#text' ? 3 : 1;
        this.attrs = { ...attrs };
        this.nodeValue = this.nodeType === 3 ? content : null;
        this.childNodes = this.nodeType === 3 ? [] : content.map((item) => new Node(doc, item));
        this.childNodes.forEach((child) => { child.parentNode = this; });
        this.parentNode = null;
        this.scrollTop = 0;
        this.clientHeight = 30;
    }
    get children() { return this.childNodes.filter((child) => child.nodeType === 1); }
    get firstElementChild() { return this.children[0]; }
    get lastElementChild() { return this.children.at(-1); }
    get lastChild() { return this.childNodes.at(-1); }
    get attributes() { return Object.entries(this.attrs).map(([name, value]) => ({ name, value })); }
    get dataset() { return { liveLineKey: this.attrs['data-live-line-key'], expanded: this.attrs['data-expanded'] }; }
    get isConnected() { return this === this.ownerDocument.root || Boolean(this.parentNode?.isConnected); }
    get scrollHeight() { return this.children.length * 20; }
    hasAttribute(name) { return name in this.attrs; }
    getAttribute(name) { return this.attrs[name] ?? null; }
    setAttribute(name, value) { this.attrs[name] = value; }
    removeAttribute(name) { delete this.attrs[name]; }
    contains(node) { return node === this || this.childNodes.some((child) => child.contains(node)); }
    focus() { this.ownerDocument.activeElement = this; }
    tuple() { return [this.nodeName, this.attrs, this.nodeType === 3 ? this.nodeValue : this.childNodes.map((child) => child.tuple())]; }
    get outerHTML() { return this.nodeType === 1 ? JSON.stringify(this.tuple()) : undefined; }
    set innerHTML(value) { this.childNodes = [new Node(this.ownerDocument, JSON.parse(value))]; this.childNodes[0].parentNode = this; }
    appendChild(node) { return this.insertBefore(node, null); }
    insertBefore(node, next) {
        node.remove();
        const index = this.childNodes.indexOf(next);
        this.childNodes.splice(index < 0 ? this.childNodes.length : index, 0, node);
        node.parentNode = this;
        return node;
    }
    replaceChild(next, current) { this.insertBefore(next, current); current.remove(); }
    remove() {
        if (!this.parentNode) return;
        if (this.contains(this.ownerDocument.activeElement)) this.ownerDocument.activeElement = null;
        this.parentNode.childNodes.splice(this.parentNode.childNodes.indexOf(this), 1);
        this.parentNode = null;
    }
    querySelector(selector) {
        const key = selector.match(/data-live-line-key="([^"]+)"/)?.[1];
        return key ? this.children.find((child) => child.dataset.liveLineKey === key) : null;
    }
}

const text = (value) => ['#text', {}, value];
function rendererFixture(options = {}) {
    const doc = { activeElement: null, createElement: (name) => new Node(doc, [name]) };
    const timelineEl = new Node(doc, ['DIV']);
    doc.root = timelineEl;
    const record = { timelineEl, root: { dataset: { expanded: '1' } }, expandedLineKeys: new Set(), items: [] };
    const build = (item) => JSON.stringify(['DIV', { 'data-live-line-key': item.lineKey,
        'data-expanded': record.expandedLineKeys.has(item.lineKey) ? '1' : '0' }, [
        ['DIV', { role: 'button', 'aria-expanded': String(record.expandedLineKeys.has(item.lineKey)) }, [
            ['SPAN', { class: 'title' }, [text(item.title || item.lineKey)]],
            ['SPAN', { class: 'time' }, [text(item.ts || '')]],
        ]],
        ['DIV', { class: 'body' }, [['P', {}, [text(item.body || 'Long narration')]]]],
    ]]);
    const renderer = createLiveCardTimelineRenderer({ withStableViewport: (fn) => fn(), buildTimelineItemHtml: build, ...options });
    return { doc, record, ...renderer };
}

for (const kind of ['receipt', 'lifecycle', 'terminal', 'activity']) {
test(`a live ${kind} retains its physical page and exact row through canonical adoption`, () => {
    const evolving = kind !== 'activity';
    const prefix = { receipt: 'cardrow|', lifecycle: 'subagent-lifecycle:', terminal: 'task_done|', activity: 'progress:' }[kind];
    const summary = id => ({ headline: 'PR #7 merge: merged', body: 'Exact merge evidence. '.repeat(20),
        phase: 'done', dedupeKey: `${prefix}${id}`,
        ...(kind === 'receipt' ? { cardRowRevision: 3 } : {}), terminal: kind === 'terminal' });
    const row = offset => ({ history_id: `chat:${offset}`,
        history_position: { source: 'chat', offset }, ts: '2026-09-12T12:00:00Z' });
    const live = rendererFixture();
    for (const id of ['other-receipt', 'receipt']) updateLiveTimelineItem(live.record, summary(id), {
        ts: '12:00', rawTs: row(41).ts, syntheticKey: summary(id).dedupeKey,
        headline: summary(id).headline, inPlaceByKey: true,
    });
    const item = live.record.items[1], liveKey = item.lineKey;
    // Component-only line disclosure state: production receipts expose card
    // expansion, which the backend/browser regression exercises separately.
    live.record.expandedLineKeys.add(liveKey);
    live.renderLiveCardTimeline(live.record);

    // Real renderer nodes, with deterministic feed geometry. A cold row changes
    // both its DOM key and offset; an identical neighbour must not be restored.
    const box = (top, height) => ({ top, bottom: top + height, left: 0, right: 600, width: 600, height });
    const mount = (f, lineTops) => {
        const messages = new Node(f.doc, ['DIV']);
        const card = new Node(f.doc, ['DIV']);
        f.doc.root = messages;
        messages.appendChild(card); card.appendChild(f.record.timelineEl);
        messages.scrollTop = 200;
        messages.getBoundingClientRect = () => box(0, 400);
        // Node's dataset is synthesized from attributes; task identity belongs
        // to the card fixture, while the actual line datasets stay rendered.
        Object.defineProperty(card, 'dataset', { value: { taskId: 'owner', expanded: '1' } });
        card.classList = { contains: value => value === 'chat-live-card' };
        card.matches = () => false;
        card.getBoundingClientRect = () => box(-messages.scrollTop, 1200);
        card.getClientRects = () => [card.getBoundingClientRect()];
        card.querySelectorAll = () => f.record.timelineEl.children;
        card.parentElement = messages;
        f.record.timelineEl.parentElement = card;
        for (const [index, line] of f.record.timelineEl.children.entries()) {
            line.parentElement = f.record.timelineEl;
            line.classList = { contains: value => value === 'chat-live-line' };
            line.matches = selector => selector === '.chat-live-line';
            line.closest = selector => selector === '.chat-live-card' ? card : null;
            line.getBoundingClientRect = () => box(lineTops[index] - messages.scrollTop, 80);
            line.getClientRects = () => [line.getBoundingClientRect()];
        }
        f.record.root = card; f.record.groupId = 'owner';
        const anchors = createTimelineAnchors({ messagesDiv: messages,
            liveCardRecords: new Map([['owner', f.record]]) });
        return { messages, card, anchors };
    };
    const mounted = mount(live, [100, 220]);
    const beforeReplay = mounted.anchors.serializeTimelineAnchor();

    const mountedLine = live.record.timelineEl.lastElementChild;
    const mountedHeader = mountedLine.firstElementChild;
    mountedHeader.focus();
    assert.equal(mergeHistoricalTimelineItem(live.record, summary('receipt'), row(41), '12:00'), true);
    live.renderLiveCardTimeline(live.record);
    assert.equal(live.record.items[1], item);
    assert.equal(item.lineKey, liveKey);
    assert.equal(item[evolving ? 'sourceHistoryId' : 'historyId'], 'chat:41');
    assert.equal(item[evolving ? 'historyId' : 'sourceHistoryId'], undefined, 'immutable and evolving source identities remain distinct');
    assert.equal(live.record.timelineEl.lastElementChild, mountedLine);
    assert.equal(live.doc.activeElement, mountedHeader);
    const saved = mounted.anchors.serializeTimelineAnchor();
    assert.equal(saved.lineExpanded, true);
    assert.equal(saved.offset, 20);
    assert.equal(saved.lineHistoryId, evolving ? '' : 'chat:41');
    assert.equal(saved.historyId, 'chat:41', 'the nested line must supply its own physical page, not a card-wide source');
    mounted.card.remove();

    // A rebuilt card (fresh DOM keys) finds the saved line by its row identity.
    const cold = rendererFixture();
    cold.record.groupId = 'owner';
    for (const [id, offset] of [['other-receipt', 40], ['receipt', 41]]) {
        mergeHistoricalTimelineItem(cold.record, summary(id), row(offset), '12:00');
    }
    cold.renderLiveCardTimeline(cold.record);
    const coldItem = cold.record.items[1];
    assert.notEqual(coldItem.lineKey, liveKey);
    assert.equal(coldItem[evolving ? 'sourceHistoryId' : 'historyId'], 'chat:41');
    const reopened = mount(cold, [280, 320]);
    assert.equal(reopened.anchors.restoreVisibleTimelineAnchor(saved, { exact: true }), true);
    assert.equal(reopened.messages.scrollTop, 300);
    assert.equal(cold.record.timelineEl.lastElementChild.getBoundingClientRect().top, saved.offset);
    assert.equal(reopened.anchors.serializeTimelineAnchor().historyId, 'chat:41');
    assert.equal(reopened.anchors.serializeTimelineAnchor().lineLifecycleKey, evolving ? `${prefix}receipt` : '');
    assert.equal(beforeReplay.lineLifecycleKey, evolving ? `${prefix}receipt` : '');
    assert.equal(saved.lineLifecycleKey, evolving ? `${prefix}receipt` : '');
});
}

test('older rows and timestamp patches preserve the mounted row, focused header and selected body', () => {
    const f = rendererFixture();
    f.record.items = [{ lineKey: 'newer', ts: '12:00' }];
    assert.equal(f.renderLiveCardTimeline(f.record), true);
    const row = f.record.timelineEl.firstElementChild;
    const [header, body] = row.children;
    const title = header.firstElementChild;
    const selectedText = body.firstElementChild.firstElementChild || body.firstElementChild.childNodes[0];
    header.focus();
    f.record.expandedLineKeys.add('newer');
    f.record.items.unshift({ lineKey: 'older' });
    f.record.items[1].ts = '12:01';
    assert.equal(f.renderLiveCardTimeline(f.record), true);
    assert.equal(f.record.timelineEl.children[1], row);
    assert.equal(row.firstElementChild, header);
    assert.equal(header.firstElementChild, title);
    assert.equal(row.children[1], body);
    assert.equal(body.contains(selectedText), true);
    assert.equal(f.doc.activeElement, header);
    assert.equal(header.getAttribute('aria-expanded'), 'true');
    assert.equal(f.renderLiveCardTimeline(f.record), false);
});

test('patching a timestamp keeps DOM additions made by markdown enhancement', () => {
    const f = rendererFixture();
    const item = { lineKey: 'one', ts: '12:00' };
    f.record.items = [item];
    f.appendTimelineItem(item, f.record);
    const row = f.record.timelineEl.firstElementChild;
    const body = row.children[1];
    const copy = new Node(f.doc, ['BUTTON', { 'aria-label': 'Copy' }, [text('Copy')]]);
    body.appendChild(copy);
    copy.focus();
    item.ts = '12:01';
    assert.equal(f.patchLastTimelineItem(item, f.record), true);
    assert.equal(body.lastElementChild, copy);
    assert.equal(f.doc.activeElement, copy);
    assert.equal(f.patchTimelineItemAt(item, f.record), false);
});

test('keyed reorder preserves header identity and restores focus; removed keys leave the timeline', () => {
    const f = rendererFixture();
    const a = { lineKey: 'a' }, b = { lineKey: 'b' };
    f.record.items = [a, b];
    f.renderLiveCardTimeline(f.record);
    const second = f.record.timelineEl.children[1];
    const header = second.firstElementChild;
    header.focus();
    f.record.items = [b, a];
    f.renderLiveCardTimeline(f.record);
    assert.equal(f.record.timelineEl.firstElementChild, second);
    assert.equal(f.doc.activeElement, header);
    f.record.items = [b];
    f.renderLiveCardTimeline(f.record);
    assert.equal(f.record.timelineEl.children.length, 1);
    assert.equal(f.doc.activeElement, header);
});

test('collapsed subagent defers DOM writes and reconciles when opened', () => {
    const f = rendererFixture();
    f.record.isSubagent = true;
    f.record.root.dataset.expanded = '0';
    f.record.items = [{ lineKey: 'one' }];
    assert.equal(f.renderLiveCardTimeline(f.record), false);
    assert.equal(f.record._timelineDirty, true);
    f.record.root.dataset.expanded = '1';
    assert.equal(f.renderLiveCardTimeline(f.record), true);
    assert.equal(f.record._timelineDirty, false);
});

test('timeline markup uses a selectable accessible header and exact history attribution', (t) => {
    const prior = globalThis.document;
    globalThis.document = { createElement: () => ({ textContent: '', get innerHTML() { return this.textContent; } }) };
    t.after(() => { globalThis.document = prior; });
    const html = buildTimelineItemHtml({ lineKey: 'one', historyId: 'source"1', phase: 'working', headline: '## Title\nNarration', fullHeadline: 'Full title', body: '## Body\nDetails' }, { expandedLineKeys: new Set(), groupId: 'task' });
    assert.match(html, /<div\s+role="button" tabindex="0"/);
    assert.doesNotMatch(html, /<button/);
    assert.match(html, /data-history-id="source&quot;1"/);
    assert.match(html, /aria-controls="chat-live-line-body-task-one"/);
    assert.match(html, /Title<\/strong><br>/);
    assert.match(html, /Body<\/strong><br>/);
    // An evolving line is released with its current source row, so visibility
    // protection must see it; that row may render elsewhere, so it is a locator.
    const receipt = buildTimelineItemHtml({ lineKey: 'terminal-receipt', dedupeKey: 'cardrow|merge-receipt:r',
        sourceHistoryId: 'progress:9', phase: 'result', headline: 'PR #7 merge: merged' }, { expandedLineKeys: new Set(), groupId: 'task' });
    assert.match(receipt, /data-source-history-id="progress:9"/);
    assert.doesNotMatch(receipt, /data-history-id/);
    const live = buildTimelineItemHtml({ lineKey: 'live', phase: 'working', headline: 'Live only' }, { expandedLineKeys: new Set(), groupId: 'task' });
    assert.doesNotMatch(live, /history-id/);
});

test('history stamps pair each row and source-only locator with its node', () => {
    const row = { dataset: { historyId: 'progress:4' } }, terminal = { dataset: { sourceHistoryId: 'progress:4' } };
    const root = { querySelectorAll: selector => ({ '[data-history-id]': [row], '[data-source-history-id]': [terminal] })[selector] };
    assert.deepEqual(historyStamps(root), [['progress:4', row], ['progress:4', terminal]]);
});

test('selection crossing a header is protected even with both endpoints outside', () => {
    const el = { contains: () => false };
    assert.equal(selectionInside(el, { isCollapsed: false, rangeCount: 1, getRangeAt: () => ({ intersectsNode: (node) => node === el }) }), true);
});

test('delegated header activation respects selection, nested controls and one keyboard activation', () => {
    const handlers = {};
    let selection = null, calls = 0, stopped = 0, prevented = 0;
    const line = { dataset: { liveLineKey: 'one' }, contains: (node) => node === inside };
    const inside = {};
    const header = { matches: () => true };
    const target = {
        closest: (selector) => selector === '.chat-live-line.expandable' ? line
            : selector === '[data-live-line-toggle]' ? header : header,
    };
    const owner = { contains: (node) => node === header, ownerDocument: { getSelection: () => selection }, addEventListener: (name, fn) => { handlers[name] = fn; } };
    bindLiveCardTimeline(owner, (key) => { assert.equal(key, 'one'); calls += 1; });
    const event = { target, stopPropagation: () => { stopped += 1; }, preventDefault: () => { prevented += 1; } };
    selection = { isCollapsed: false, anchorNode: inside };
    handlers.click(event);
    assert.equal(calls, 0);
    selection = null;
    handlers.click(event);
    handlers.keydown({ ...event, key: 'Enter' });
    handlers.keydown({ ...event, key: ' ' });
    handlers.keydown({ ...event, key: ' ', repeat: true });
    assert.deepEqual([calls, stopped, prevented], [3, 4, 3]);
    const link = { closest: (selector) => selector === '.chat-live-line.expandable' ? line : selector === '[data-live-line-toggle]' ? header : { matches: () => false } };
    handlers.click({ ...event, target: link });
    handlers.keydown({ ...event, target: link, key: 'Enter' });
    assert.equal(calls, 3);
});

test('a line disclosure or late full output keeps a pinned timeline where it is; a new newest line is followed', () => {
    const f = rendererFixture();
    f.record.items = [{ lineKey: 'one' }, { lineKey: 'two' }];
    f.renderLiveCardTimeline(f.record);
    assert.equal(f.record.timelineEl.scrollTop, 40, 'a fresh timeline opens at its newest line');
    f.record.timelineEl.scrollTop = 10; // still within the pinned band
    f.record.expandedLineKeys.add('one');
    assert.equal(f.renderLiveCardTimeline(f.record), true);
    assert.equal(f.record.timelineEl.scrollTop, 10, 'expanding the line being read does not scroll it away');
    f.record.items[0].body = 'The fetched full output';
    assert.equal(f.renderLiveCardTimeline(f.record), true);
    assert.equal(f.record.timelineEl.scrollTop, 10);
    f.record.items.push({ lineKey: 'three' });
    f.renderLiveCardTimeline(f.record);
    assert.equal(f.record.timelineEl.scrollTop, 60, 'a new newest line is followed');
});

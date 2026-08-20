// Behavioural characterization of the live-card presentation owner, exercised
// where the code now lives. The renderers only build strings and write into a
// card record's elements, so a small element model (plus the detached div that
// escapeHtmlText round-trips through) reaches every branch: the coined title
// and its collapsed line, the phase label, the expand affordance and the lazily
// materialized timeline, the incremental append/patch writers, and the one
// meta-line renderer.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createLiveCardView } from '../modules/chat_live_card_view.js';

const HTML_ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;' };

function makeElement(tag = 'div') {
    let html = '';
    const el = {
        tagName: tag.toUpperCase(),
        className: '',
        hidden: false,
        scrollTop: 0,
        scrollHeight: 1000,
        clientHeight: 100,
        dataset: {},
        attributes: {},
        style: {},
        children: [],
        removedAttributes: [],
        set textContent(value) { html = String(value ?? '').replace(/[&<>]/g, (c) => HTML_ESCAPES[c]); },
        get textContent() { return html; },
        appendChild(child) { el.children.push(child); child.parentNode = el; return child; },
        replaceChild(next, previous) {
            const at = el.children.indexOf(previous);
            if (at === -1) el.children.push(next);
            else el.children[at] = next;
            next.parentNode = el;
            return previous;
        },
        insertAdjacentElement(_position, node) { el.children.push(node); node.parentNode = el; return node; },
        removeAttribute(name) { el.removedAttributes.push(name); delete el.attributes[name]; },
        setAttribute(name, value) { el.attributes[name] = String(value); },
        querySelector(selector) { return el.stubbedNodes?.[selector] ?? null; },
        querySelectorAll(selector) { return el.stubbedLists?.[selector] ?? []; },
        get lastElementChild() { return el.children[el.children.length - 1] ?? null; },
        get firstElementChild() { return el.children[0] ?? null; },
    };
    // The builders write ONE root element per line, then read it back through
    // firstElementChild — so an innerHTML write has to materialize that node.
    Object.defineProperty(el, 'innerHTML', {
        get() { return html; },
        set(value) {
            html = String(value ?? '');
            el.children = html.trim() ? [Object.assign(makeElement('div'), { outerHTML: html })] : [];
        },
    });
    return el;
}

function makeRecord(overrides = {}) {
    return {
        groupId: 'task-1',
        isSubagent: false,
        items: [],
        expandedLineKeys: new Set(),
        collapsedActivity: '',
        suggestedName: '',
        _timelineDirty: false,
        _lastFrameMeta: [],
        costMeta: null,
        root: makeElement('div'),
        titleEl: makeElement('span'),
        activityEl: makeElement('div'),
        metaEl: makeElement('div'),
        toggleEl: makeElement('span'),
        summaryButtonEl: makeElement('button'),
        inlineTypingEl: makeElement('div'),
        timelineEl: makeElement('div'),
        subagentsEl: null,
        ...overrides,
    };
}

function view({ records = new Map(), names = new Map() } = {}) {
    const layouts = [];
    const priorDocument = globalThis.document;
    // escapeHtmlText round-trips through a detached div's textContent/innerHTML,
    // and every builder below only ever creates elements.
    globalThis.document = { createElement: (tag) => makeElement(tag) };
    const api = createLiveCardView({
        liveCardRecords: records,
        pendingSuggestedNames: names,
        // The viewport wrapper is chat.js's; here it must simply run the mutation.
        withStableViewport: (mutate) => mutate(),
        getLiveCardRecord: (groupId) => {
            if (!records.has(groupId)) records.set(groupId, makeRecord({ groupId }));
            return records.get(groupId);
        },
        syncLiveCardLayout: (record) => layouts.push(record),
    });
    return {
        ...api,
        records,
        names,
        layouts,
        restore() { globalThis.document = priorDocument; },
    };
}

test('a coined name that arrives before its card is buffered under a bounded FIFO', () => {
    const v = view();
    for (let i = 0; i < 120; i += 1) v.applySuggestedName(`task-${i}`, `name-${i}`);
    assert.equal(v.names.size, 100, 'the unthreaded task_named broadcast must not grow without limit');
    assert.equal(v.names.has('task-0'), false, 'the oldest buffered name is evicted first');
    assert.equal(v.names.get('task-119'), 'name-119');
    v.restore();
});

test('a coined name titles a main card and repopulates its collapsed line', () => {
    const v = view();
    const record = makeRecord({ collapsedActivity: 'Reading the dataset' });
    v.records.set('task-1', record);
    v.applySuggestedName('task-1', 'Data Analysis');
    assert.equal(record.suggestedName, 'Data Analysis');
    assert.equal(record.titleEl.textContent, 'Data Analysis');
    assert.equal(record.activityEl.textContent, 'Reading the dataset');
    assert.deepEqual(record.activityEl.removedAttributes, ['title'], 'narration is never a mouse-only title');
    assert.equal(v.names.has('task-1'), false, 'an applied name is not also buffered');
    v.restore();
});

test('a subagent card never takes the coined project name, and blanks are ignored', () => {
    const v = view();
    const child = makeRecord({ isSubagent: true });
    v.records.set('child-1', child);
    v.applySuggestedName('child-1', 'Data Analysis');
    assert.equal(child.suggestedName, '');
    v.applySuggestedName('', 'Data Analysis');
    v.applySuggestedName('task-9', '   ');
    assert.equal(v.names.size, 0);
    v.restore();
});

test('the phase label maps every known phase and capitalizes the rest', () => {
    const v = view();
    assert.equal(v.formatLiveCardPhaseLabel('thinking'), 'Thinking');
    assert.equal(v.formatLiveCardPhaseLabel('done'), 'Done');
    assert.equal(v.formatLiveCardPhaseLabel('cancelled'), 'Cancelled');
    assert.equal(v.formatLiveCardPhaseLabel('warn'), 'Notice');
    for (const phase of ['error', 'timeout', 'lifecycle_error']) {
        assert.equal(v.formatLiveCardPhaseLabel(phase), 'Issue');
    }
    assert.equal(v.formatLiveCardPhaseLabel(''), 'Working');
    assert.equal(v.formatLiveCardPhaseLabel('queued'), 'Queued');
    v.restore();
});

test('a line is expandable on divergent full text, or on a truncated line with a fetch ref', () => {
    const v = view();
    assert.equal(v.isLiveLineExpandable({ headline: 'a', fullHeadline: 'a', body: 'b', fullBody: 'b' }), false);
    assert.equal(v.isLiveLineExpandable({ headline: 'a', fullHeadline: 'aa' }), true);
    assert.equal(v.isLiveLineExpandable({ body: 'b', fullBody: 'bb' }), true);
    // The preview can equal the capped body and STILL have more to show.
    assert.equal(v.isLiveLineExpandable({ body: 'b', fullBody: 'b', truncated: true, fullRef: 'task-2' }), true);
    assert.equal(v.isLiveLineExpandable({ body: 'b', fullBody: 'b', truncated: true }), false);
    v.restore();
});

test('the disclosure label and aria state follow the card', () => {
    const v = view();
    const record = makeRecord();
    v.syncLiveCardToggle(record);
    assert.equal(record.toggleEl.textContent, 'Show details');
    assert.equal(record.summaryButtonEl.attributes['aria-expanded'], 'false');
    record.root.dataset.expanded = '1';
    v.syncLiveCardToggle(record);
    assert.equal(record.toggleEl.textContent, 'Hide details');
    assert.equal(record.summaryButtonEl.attributes['aria-expanded'], 'true');
    v.restore();
});

test('the inline typing dots are shown and hidden without touching anything else', () => {
    const v = view();
    const record = makeRecord();
    v.setLiveCardTypingVisible(record, false);
    assert.equal(record.inlineTypingEl.style.display, 'none');
    v.setLiveCardTypingVisible(record, true);
    assert.equal(record.inlineTypingEl.style.display, '');
    v.setLiveCardTypingVisible({}, true);
    v.restore();
});

test('the subagent container is created once and kept directly after the timeline', () => {
    const v = view();
    const first = v.ensureSubagentContainer('parent-1');
    assert.equal(first.className, 'chat-subagents');
    assert.equal(first.dataset.subagentsFor, 'parent-1');
    const parent = v.records.get('parent-1');
    assert.equal(parent.subagentsEl, first);
    assert.equal(v.ensureSubagentContainer('parent-1'), first, 'the container is reused, never re-minted');
    assert.equal(v.ensureSubagentContainer(''), null);
    // Only DIRECT children count towards the card's "N children" badge.
    parent.subagentsEl.stubbedLists = { ':scope > .chat-live-card.subagent': [1, 2] };
    assert.equal(v.directSubagentCount(parent), 2);
    assert.equal(v.directSubagentCount({}), 0);
    v.restore();
});

test('a timeline line renders its expand affordance only when there is more to show', () => {
    const v = view();
    const record = makeRecord();
    const plain = v.buildTimelineItemHtml(
        { lineKey: 'k1', headline: 'Ran tests', body: '', count: 1, ts: '10:00', phase: 'done' },
        record,
    );
    assert.match(plain, /<div class="chat-live-line-head">/);
    assert.doesNotMatch(plain, /chat-live-line-toggle/);
    const expandable = v.buildTimelineItemHtml(
        { lineKey: 'k2', headline: 'Ran tests', fullHeadline: 'Ran the whole suite', body: 'ok', count: 3, ts: '', phase: 'done' },
        record,
    );
    assert.match(expandable, /data-live-line-toggle="k2"/);
    assert.match(expandable, /aria-expanded="false"/);
    assert.match(expandable, /<span class="chat-live-line-expand-label">Expand<\/span>/);
    assert.match(expandable, /3x/);
    v.restore();
});

test('an expanded truncated line offers "Show full", then prefers the fetched output', () => {
    const v = view();
    const record = makeRecord();
    record.expandedLineKeys.add('k1');
    const item = { lineKey: 'k1', headline: 'Research', body: 'capped', fullBody: 'capped', truncated: true, fullRef: 'task-2', count: 1, phase: 'working' };
    // Collapsed, the affordance advertises that there is genuinely more.
    record.expandedLineKeys.delete('k1');
    assert.match(v.buildTimelineItemHtml(item, record), /Show full/);
    record.expandedLineKeys.add('k1');
    const loading = v.buildTimelineItemHtml(item, record);
    assert.match(loading, /Collapse/);
    assert.match(loading, /Loading full output…/);
    const fetched = v.buildTimelineItemHtml({ ...item, fetchedFull: 'the whole thing' }, record);
    assert.match(fetched, /chat-live-line-body-full/);
    assert.match(fetched, /the whole thing/);
    assert.doesNotMatch(fetched, /Loading full output…/);
    v.restore();
});

test('a collapsed SUBAGENT timeline defers its DOM; a top-level card renders eagerly', () => {
    const v = view();
    const top = makeRecord();
    assert.equal(v.deferCollapsedTimeline(top), false);
    const child = makeRecord({ isSubagent: true });
    assert.equal(v.deferCollapsedTimeline(child), true);
    assert.equal(child._timelineDirty, true, 'the deferred DOM is marked stale, the data is not lost');
    child.root.dataset.expanded = '1';
    child._timelineDirty = false;
    assert.equal(v.deferCollapsedTimeline(child), false);
    assert.equal(v.deferCollapsedTimeline(null), true);
    v.restore();
});

test('a full timeline rebuild follows the tail only while it was pinned', () => {
    const v = view();
    const record = makeRecord();
    record.items = [{ lineKey: 'k1', headline: 'One', body: '', count: 1, phase: 'done' }];
    record.timelineEl.scrollTop = 0;
    record.timelineEl.scrollHeight = 500;
    record.timelineEl.clientHeight = 100;
    v.renderLiveCardTimeline(record);
    assert.equal(record.timelineEl.scrollTop, 0, 'a reader scrolled up is not yanked to the tail');
    assert.match(record.timelineEl.innerHTML, /One/);
    record.timelineEl.scrollTop = 400;  // pinned: 500 - 400 - 100 <= 24
    v.renderLiveCardTimeline(record);
    assert.equal(record.timelineEl.scrollTop, 500);
    v.restore();
});

test('append and patch materialize a stale timeline instead of writing into wrong nodes', () => {
    const v = view();
    const record = makeRecord();
    record.items = [{ lineKey: 'k1', headline: 'One', body: '', count: 1, phase: 'done' }];
    record._timelineDirty = true;
    v.appendTimelineItem(record.items[0], record);
    assert.equal(record._timelineDirty, false);
    assert.match(record.timelineEl.innerHTML, /One/, 'a stale timeline is rebuilt from items, not appended to');
    // With a clean timeline the append is incremental.
    v.appendTimelineItem(record.items[0], record);
    assert.equal(record.timelineEl.children.length, 2);
    // patch-last replaces the last node in place rather than growing the list.
    v.patchLastTimelineItem({ ...record.items[0], headline: 'Two' }, record);
    assert.equal(record.timelineEl.children.length, 2);
    assert.match(record.timelineEl.lastElementChild.outerHTML, /Two/);
    // A collapsed subagent card takes neither path: its DOM stays deferred.
    const child = makeRecord({ isSubagent: true });
    v.appendTimelineItem(record.items[0], child);
    assert.equal(child.timelineEl.innerHTML, '');
    assert.equal(child._timelineDirty, true);
    v.restore();
});

test('patch-at rebuilds when the addressed node is gone', () => {
    const v = view();
    const record = makeRecord();
    record.items = [{ lineKey: 'k1', headline: 'One', body: '', count: 1, phase: 'done' }];
    v.patchTimelineItemAt(record.items[0], record);
    assert.match(record.timelineEl.innerHTML, /One/, 'a missing node falls back to the full rebuild');
    v.restore();
});

test('the meta line renders from record state: sticky chip, frame meta, cost, activity clock', () => {
    const v = view();
    const record = makeRecord({ groupId: 'bg-consciousness' });
    record.executorChip = { icon: '*', label: 'Claudexor', title: 'routed' };
    record._lastFrameMeta = ['rounds=3'];
    record.costMeta = { meta: ['cost=$1.00'] };
    record.latestActivityTs = '10:05';
    v.renderLiveCardMeta(record);
    const html = record.metaEl.innerHTML;
    assert.match(html, /harness-chip chat-live-executor-chip/);
    assert.match(html, /Background thinking/);
    assert.match(html, /rounds=3/);
    assert.match(html, /cost=\$1\.00/);
    assert.match(html, /Latest 10:05/);
    // No chip fact, no placeholder chip.
    const plain = makeRecord();
    v.renderLiveCardMeta(plain);
    assert.equal(plain.metaEl.innerHTML, '');
    v.restore();
});

test('the first expand materializes a timeline that was deferred while collapsed', () => {
    const v = view();
    const record = makeRecord({ isSubagent: true });
    record.items = [{ lineKey: 'k1', headline: 'One', body: '', count: 1, phase: 'done' }];
    record._timelineDirty = true;
    record.root.isConnected = false;
    v.setLiveCardExpanded(record, true);
    assert.equal(record.root.dataset.expanded, '1');
    assert.equal(record._timelineDirty, false);
    assert.match(record.timelineEl.innerHTML, /One/);
    assert.equal(record.toggleEl.textContent, 'Hide details');
    v.restore();
});

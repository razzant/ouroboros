import test from 'node:test';
import assert from 'node:assert/strict';
import { reconcileReviewElementTree } from '../modules/review_dom_patch.js';

// A small DOM with real text nodes, child order and moves counted: enough to see what the keyed
// Reviews patch does to the nodes a reader is looking at.
class Text {
    constructor(data) { this.nodeType = 3; this.data = String(data); this.parentNode = null; }
    cloneNode() { return new Text(this.data); }
    get nextSibling() { const list = this.parentNode?.childNodes || []; return list[list.indexOf(this) + 1] || null; }
    remove() { this.parentNode?._detach(this); }
}
class El {
    constructor(tag, attrs = {}, children = []) {
        this.nodeType = 1;
        this.tagName = tag.toUpperCase();
        this.childNodes = [];
        this.parentNode = null;
        this.moves = 0;
        this.scrollTop = 0;
        this._attrs = new Map();
        for (const [name, value] of Object.entries(attrs)) this.setAttribute(name, value);
        for (const child of children) this.insertBefore(typeof child === 'string' ? new Text(child) : child, null);
    }
    get dataset() {
        return Object.fromEntries([...this._attrs].filter(([name]) => name.startsWith('data-'))
            .map(([name, value]) => [name.slice(5).replace(/-([a-z])/g, (_all, char) => char.toUpperCase()), value]));
    }
    get classList() { return { contains: (name) => String(this._attrs.get('class') || '').split(/\s+/).includes(name) }; }
    get attributes() { return [...this._attrs].map(([name, value]) => ({ name, value })); }
    get children() { return this.childNodes.filter((node) => node.nodeType === 1); }
    get firstChild() { return this.childNodes[0] || null; }
    get nextSibling() { const list = this.parentNode?.childNodes || []; return list[list.indexOf(this) + 1] || null; }
    get innerHTML() { return this.childNodes.map((node) => (node.nodeType === 3 ? node.data : `<${node.tagName.toLowerCase()}>${node.innerHTML}</${node.tagName.toLowerCase()}>`)).join(''); }
    set innerHTML(value) { this.childNodes.forEach((node) => { node.parentNode = null; }); this.childNodes = value ? [new Text(value)] : []; this.childNodes.forEach((node) => { node.parentNode = this; }); }
    get textContent() { return this.childNodes.map((node) => (node.nodeType === 3 ? node.data : node.textContent)).join(''); }
    hasAttribute(name) { return this._attrs.has(name); }
    getAttribute(name) { return this._attrs.has(name) ? this._attrs.get(name) : null; }
    setAttribute(name, value) { this._attrs.set(name, String(value)); }
    removeAttribute(name) { this._attrs.delete(name); }
    _detach(node) { const index = this.childNodes.indexOf(node); if (index >= 0) this.childNodes.splice(index, 1); node.parentNode = null; }
    insertBefore(node, before) {
        if (node.parentNode) { node.parentNode._detach(node); node.moves = (node.moves || 0) + 1; }
        const index = before ? this.childNodes.indexOf(before) : -1;
        if (index >= 0) this.childNodes.splice(index, 0, node); else this.childNodes.push(node);
        node.parentNode = this;
        return node;
    }
    remove() { this.parentNode?._detach(this); }
    cloneNode(deep) {
        return new El(this.tagName.toLowerCase(), Object.fromEntries(this._attrs), deep ? this.childNodes.map((node) => node.cloneNode(true)) : []);
    }
}
const el = (tag, attrs, ...children) => new El(tag, attrs, children);

test('a word beside an element is rewritten, while the element and a focused control keep their nodes', () => {
    // The R1 failure: `<span><i></i>Awaiting answer</span>` patched to FAIL kept the old words.
    const dot = el('i', { 'aria-hidden': 'true' });
    const read = el('button', { 'data-review-read-url': '/a' }, 'Read');
    const current = el('div', { 'data-review-opinion': 'seat' }, el('span', { class: 'state' }, dot, 'Awaiting answer'), 'note', read);
    const desired = el('div', { 'data-review-opinion': 'seat' }, el('span', { class: 'state' }, el('i', { 'aria-hidden': 'true' }), 'rejected it'), 'note', el('button', { 'data-review-read-url': '/a' }, 'Read'));
    assert.equal(reconcileReviewElementTree(current, desired), true);
    assert.equal(current.textContent, 'rejected itnoteRead');
    assert.equal(current.children[0].childNodes[0], dot, 'the decorative element is the same node');
    assert.equal(current.children[1], read, 'a focused Read stays the node the reader holds');
    assert.equal(read.moves, 0, 'and is never taken out of the document');
});

test('a state that changes anatomy leaves no stale words in either direction', () => {
    const current = el('div', {}, el('span', { class: 'review-state ui-status', 'data-tone': 'ok' }, 'passed it'));
    const live = el('div', {}, el('span', { class: 'review-state', 'data-tone': 'working' }, el('span', { class: 'status-mark' }), el('span', { class: 'review-state-word' }, 'awaiting')));
    reconcileReviewElementTree(current, live);
    assert.equal(current.textContent, 'awaiting');
    assert.equal(current.children[0].getAttribute('data-tone'), 'working');
    const settled = el('div', {}, el('span', { class: 'review-state ui-status', 'data-tone': 'error' }, 'rejected it'));
    reconcileReviewElementTree(current, settled);
    assert.equal(current.textContent, 'rejected it');
    assert.equal(current.children[0].getAttribute('class'), 'review-state ui-status');
    assert.equal(current.children[0].children.length, 0);
});

test('an opened Record facts keeps open and its reading position while its text is rewritten', () => {
    const body = el('div', { class: 'review-facts-body', 'data-review-attempt-detail': 'g:facts' }, el('dl', {}, el('dt', {}, 'Record'), el('dd', {}, 'old')));
    body.scrollTop = 120;
    const facts = el('details', { 'data-review-facts': 'g', open: '' }, el('summary', {}, 'Record facts'), body);
    const current = el('div', {}, facts);
    const desired = el('div', {}, el('details', { 'data-review-facts': 'g' }, el('summary', {}, 'Record facts'),
        el('div', { class: 'review-facts-body', 'data-review-attempt-detail': 'g:facts' }, el('dl', {}, el('dt', {}, 'Record'), el('dd', {}, 'new')))));
    reconcileReviewElementTree(current, desired);
    assert.equal(current.children[0], facts);
    assert.equal(facts.hasAttribute('open'), true, 'the owner opened it; an update never closes it');
    assert.equal(facts.children[1], body);
    assert.equal(body.scrollTop, 120);
    assert.match(body.textContent, /Recordnew/);
});

test('keyed rows reorder by identity and an unkeyed sibling is matched in order', () => {
    const one = el('div', { 'data-review-opinion': 'one' }, 'first');
    const two = el('div', { 'data-review-opinion': 'two' }, 'second');
    const current = el('div', {}, one, two, el('p', {}, 'tail'));
    const desired = el('div', {}, el('div', { 'data-review-opinion': 'two' }, 'second'), el('div', { 'data-review-opinion': 'one' }, 'first!'), el('p', {}, 'tail'));
    reconcileReviewElementTree(current, desired);
    assert.deepEqual(current.children.slice(0, 2), [two, one]);
    assert.equal(one.textContent, 'first!');
    assert.equal(current.textContent, 'secondfirst!tail');
});

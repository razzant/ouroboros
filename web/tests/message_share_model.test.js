// What Share passes on (message_share_model.js): the blocks of one rendered message, its Text (link words with the
// whole address, code exactly as written in a fence, lists close, a table and a diagram as their words and source),
// word counts, a chosen part and exact words inside it, and what a card cannot draw.
import assert from 'node:assert/strict';
import test from 'node:test';
import {
    blockWords, cardUnsupported, pickBlocks, shareBlocks, shareHref, shareText, sliceBlocks, textDirection, wordCount,
} from '../modules/message_share_model.js';

// A rendered message as the walker reads it: text nodes and elements with tag, attributes and children.
const text = (value) => ({ nodeType: 3, nodeValue: value, get textContent() { return value; } });
function el(tag, attrs = {}, ...childNodes) {
    return {
        nodeType: 1, tagName: tag.toUpperCase(), childNodes, className: attrs.class || '',
        getAttribute: (name) => (name in attrs ? String(attrs[name]) : null),
        get textContent() { return childNodes.map((child) => child.textContent).join(''); },
    };
}

const ROUTE = 'https://example.org/route?day=1&language=en';
const message = () => el('div', {},
    el('p', {}, text('Stay by the '), el('strong', {}, text('river')), text(':\n flat, '),
        el('a', { href: ROUTE, class: 'md-link' }, text('route')), text('. Use '), el('code', {}, text('npm  test')), text('.')),
    el('h2', { class: 'md-h2' }, text('The route')),
    el('ol', { start: '3' },
        el('li', {}, text('Praça do Comércio, '), el('em', {}, text('9:30')), text('.')),
        el('li', {}, el('p', {}, text('Loose item')), el('ul', {}, el('li', {}, text('nested'))))),
    el('div', { class: 'md-code-block' }, el('span', { class: 'md-code-language' }, text('js')),
        el('button', {}, text('Copy')),
        el('pre', {}, el('code', { class: 'language-js' }, text('  const a = 1;\n\n  return a;\n\n\n')))),
    el('blockquote', { class: 'md-quote' }, el('p', {}, text('A quoted line'))),
    el('hr'),
    el('ul', {}, el('li', {}, el('span', { class: 'md-checkbox is-checked', 'aria-hidden': 'true' }), text(' Done item'))),
    el('div', { class: 'md-table-wrap' }, el('table', {},
        el('thead', {}, el('tr', {}, el('th', {}, text('A')), el('th', {}, text('B')))),
        el('tbody', {}, el('tr', {}, el('td', {}, text('1')), el('td', {}, el('a', { href: '/api/files/x' }, text('two'))))))),
    el('div', { class: 'md-mermaid' }, text('graph TD;\nA-->B\n')),
    el('p', {}, text('שלום עולם')),
    el('p', {}, text('Area is $$x^2$$ here')));

test('Text keeps the words, each link with its whole address, code exactly in a fence, and lists close', () => {
    const blocks = shareBlocks(message());
    assert.deepEqual(blocks.map((item) => item.kind), ['paragraph', 'heading', 'item', 'item', 'item', 'code', 'paragraph',
        'rule', 'item', 'table', 'diagram', 'paragraph', 'paragraph']);
    assert.equal(shareText(blocks), [
        `Stay by the river: flat, route (${ROUTE}). Use npm  test.`,  // a source line break is a space
        '',
        'The route',
        '',
        '3. Praça do Comércio, 9:30.',
        '4. Loose item',
        '   • nested',
        '',
        '```js',
        '  const a = 1;',
        '',
        '  return a;',
        '',
        '',
        '```',
        '',
        '> A quoted line',
        '',
        '———',
        '',
        '[x] Done item',
        '',
        'A | B',
        '1 | two',
        '',
        '```mermaid',
        'graph TD;',
        'A-->B',
        '```',
        '',
        'שלום עולם',
        '',
        'Area is $$x^2$$ here',
    ].join('\n'));
    assert.ok(!shareText(blocks).includes('Copy'), 'the code block’s own button is not a word of the message');
});

test('a fence outgrows the code’s own backticks, and an address is an absolute web or mail one', () => {
    const blocks = shareBlocks(el('div', {}, el('pre', {}, el('code', {}, text('```\ninner\n```\n')))));
    assert.equal(shareText(blocks), '````\n```\ninner\n```\n````');
    assert.equal(shareHref(' https://x.test/a '), 'https://x.test/a');
    assert.equal(shareHref('mailto:a@b.test'), 'mailto:a@b.test');
    assert.equal(shareHref('/api/files/x'), '');
    assert.equal(shareHref('javascript:alert(1)'), '');
    const mail = shareBlocks(el('p', {}, el('a', { href: 'mailto:a@b.test' }, text('a@b.test'))));
    assert.equal(shareText(mail), 'a@b.test', 'words that are the address carry it already');
});

test('counts, a chosen part, exact words inside parts and what a card cannot draw', () => {
    const blocks = shareBlocks(message());
    assert.equal(wordCount('Praça do Comércio, ferry.'), 4);
    assert.equal(wordCount(blockWords(blocks.slice(0, 2))), wordCount('Stay by the river flat route Use npm test The route'));
    const parts = pickBlocks(blocks, new Set([1, 2]));
    assert.equal(shareText(parts), 'The route\n\n3. Praça do Comércio, 9:30.');
    parts[0].runs[0].text = 'changed';
    assert.equal(blocks[1].runs[0].text, 'The route', 'a choice is its own copy');
    // "river: flat," across a bold run, "do Comércio," inside an item, and code with its exact spaces.
    const pieces = [{ block: 0, run: 1, start: 0, end: 5 }, { block: 0, run: 2, start: 0, end: 7 },
        { block: 2, run: 0, start: 6, end: 18 }, { block: 5, run: 0, start: 0, end: 14 }];
    assert.equal(shareText(sliceBlocks(blocks, pieces)), 'river: flat,\n\n3. do Comércio,\n\n```js\n  const a = 1;\n```');
    assert.deepEqual(cardUnsupported(blocks), ['table', 'math', 'diagram']);
    assert.deepEqual(cardUnsupported(pickBlocks(blocks, new Set([0, 1, 2]))), []);
    assert.equal(blocks[11].rtl, true);
    assert.equal(textDirection('Hello שלום'), 'ltr', 'the first letter sets the direction');
});

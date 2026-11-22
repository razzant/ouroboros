// The Card of Share (message_share_card.js): the words laid out inside the card at full size (a word wider than the
// card breaks between graphemes, never shrinks), a link's site beside its words, the brand row, and the honest
// refusals: what a card cannot draw, a text taller than one card, an engine with no canvas.
import assert from 'node:assert/strict';
import test from 'node:test';
import { CARD_MAX_HEIGHT, CARD_WIDTH, layoutCard, renderShareCard } from '../modules/message_share_card.js';
import { shareBlocks, shareText } from '../modules/message_share_model.js';

const THEME = { background: '#fff', tint: 'pink', text: '#000', meta: '#555', link: '#b00', brand: '#b00', rule: '#ddd',
    wash: '#eee', sans: 'system-ui' };
const measure = (value) => [...String(value)].length * 10;  // every character 10 px wide
const paragraph = (index, ...runs) => ({ index, kind: 'paragraph', depth: 0, quote: 0, runs });

test('every word stays at its size inside the card: long words break between letters, links show their site', () => {
    const blocks = [
        { index: 0, kind: 'heading', level: 2, depth: 0, quote: 0, runs: [{ text: 'The route' }] },
        paragraph(1, { text: 'See ' }, { text: 'the guide', link: 1, href: 'https://www.example.org/guide' },
            { text: ` and ${'a'.repeat(100)} end.` }),
        { index: 2, kind: 'item', depth: 0, quote: 0, marker: '1.', runs: [{ text: 'First stop', bold: true }] },
        { index: 3, kind: 'code', depth: 0, quote: 0, runs: [{ text: '  indented\nline', code: true }] },
    ];
    const plan = layoutCard(blocks, { theme: THEME, measure, excerpt: true, logo: true });
    const pieces = plan.lines.flatMap((line) => line.pieces.map((piece) => ({ ...piece, y: line.y })));
    for (const piece of pieces) assert.ok(piece.x >= 0 && piece.x + piece.width <= CARD_WIDTH - 29.5, piece);
    const words = pieces.map((piece) => piece.text).join('');
    assert.ok(words.includes(' (example.org)'), 'the link’s site in the quiet ink');
    assert.equal(words.replace(/[^a]/g, '').length >= 100, true, 'no letter of the long word is lost');
    const code = pieces.filter((piece) => /indented/.test(piece.text));
    assert.equal(code[0].text, '\u00a0\u00a0indented', 'code keeps its leading spaces');
    assert.deepEqual(plan.brand.pieces.map((piece) => piece.text), ['Ouroboros', ' · excerpt']);
    assert.ok(plan.brand.mark, 'the app’s own mark');
    assert.ok(plan.height > 0 && plan.height <= CARD_MAX_HEIGHT);
});

test('a deeply nested excerpt stays whole as Text and refuses a card before drawing outside its width', async () => {
    const text = (value) => ({ nodeType: 3, nodeValue: value, textContent: value });
    const el = (tag, ...childNodes) => ({ nodeType: 1, tagName: tag, childNodes, getAttribute: () => null,
        get textContent() { return childNodes.map((child) => child.textContent).join(''); } });
    let source = el('PRE', el('CODE', text('deep payload\n')));
    for (let depth = 0; depth < 20; depth += 1) source = el('UL', el('LI', text('Parent'), source));
    const excerpt = shareBlocks(el('DIV', source)).filter((item) => item.kind === 'code');
    assert.equal(excerpt[0].depth, 20, 'depth comes from the production Markdown block reader');
    assert.ok(shareText(excerpt).includes('deep payload'), 'Text keeps the chosen leaf');
    let painted = false;
    const ctx = { measureText: (value) => ({ width: measure(value) }), scale() { painted = true; },
        beginPath() {}, roundRect() {}, rect() {}, fill() {}, fillText() {}, fillRect() {} };
    const canvas = { getContext: () => ctx, toBlob(done) { painted = true; done(new Blob(['png'])); } };
    const doc = { createElement: (tag) => tag === 'canvas' ? canvas : {}, documentElement: {}, body: {} };
    assert.deepEqual(await renderShareCard(excerpt, { doc, win: {}, logo: '' }), { refused: 'too_wide', kinds: [] });
    assert.equal(painted, false);
    // Width is a drawing fact, not a fixed nesting limit: a shallow oversized marker also refuses.
    const wide = [{ index: 0, kind: 'item', depth: 0, quote: 0, marker: '9'.repeat(50) + '.', runs: [{ text: 'leaf' }] }];
    assert.deepEqual(await renderShareCard(wide, { doc, win: {}, logo: '' }), { refused: 'too_wide', kinds: [] });
    const rtl = [{ ...paragraph(0, { text: 'שלום' }), depth: 20, rtl: true }];
    assert.deepEqual(await renderShareCard(rtl, { doc, win: {}, logo: '' }), { refused: 'too_wide', kinds: [] });
});

test('a card refuses what it cannot draw, a text taller than one card, and an engine without a canvas', async () => {
    const table = [{ index: 0, kind: 'table', depth: 0, quote: 0, runs: [{ text: 'A | B', verbatim: true }] }];
    assert.deepEqual(await renderShareCard(table, { doc: {} }), { refused: 'unsupported', kinds: ['table'] });
    const formula = [{ ...paragraph(0, { text: 'Area $$x^2$$' }), math: true }];
    assert.deepEqual(await renderShareCard(formula, { doc: {} }), { refused: 'unsupported', kinds: ['math'] });
    const plain = [paragraph(0, { text: 'Hello' })];
    assert.deepEqual(await renderShareCard(plain, { doc: { createElement: () => ({}) } }), { refused: 'unavailable', kinds: [] });
    const tall = Array.from({ length: 60 }, (_, index) => paragraph(index, { text: `Line ${index}` }));
    assert.ok(layoutCard(tall, { theme: THEME, measure }).height > CARD_MAX_HEIGHT, 'past one card, never squeezed');
    // The same tall text through a canvas: refused before anything is drawn.
    let drawn = false;
    const ctx = { measureText: (value) => ({ width: measure(value) }), scale() { drawn = true; } };
    const canvas = { getContext: () => ctx, toBlob() { drawn = true; } };
    const doc = { createElement: (tag) => (tag === 'canvas' ? canvas : {}), documentElement: {}, body: {} };
    assert.deepEqual(await renderShareCard(tall, { doc, win: {}, logo: '' }), { refused: 'too_long', kinds: [] });
    assert.equal(drawn, false);
});

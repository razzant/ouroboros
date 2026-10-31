// The quote panel (chat_reply_quote.js): ONE component for a reply (one line, a button that jumps), the strip above
// the composer (with cancel) and a relay (the owner's complete words, clamped, `Show all` only while it hides text).
// Layout is not measured here (no engine); the structure, the roles, the words-as-text and the toggle are.
import assert from 'node:assert/strict';
import test from 'node:test';
import { clampWithToggle, createRelayQuote, createReplyQuote, normalizeReplyRef, replySnippet } from '../modules/chat_reply_quote.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

function withDom(run) {
    const env = installDom();
    try { return run(); } finally { restoreDom(env.prior); }
}
const fixedTime = () => ({ short: '00:37', full: 'Oct 12, 00:37' });

test('a reply quote is one jump button in the QUOTED author\'s voice, words set as text', () => withDom(() => {
    const jumps = [];
    const panel = createReplyQuote({ address: 'row:1@a#1', role: 'user', snippet: '<b>not markup</b>', ts: 't' },
        { onJump: (ref) => jumps.push(ref.address), formatTime: fixedTime });
    assert.equal(panel.dataset.author, 'user');
    assert.equal(panel.dataset.mode, 'reply');
    const [main] = panel.children;
    assert.equal(main.tagName, 'BUTTON');
    const [head, text] = main.children;
    assert.equal(head.children[0].textContent, 'You');
    assert.equal(head.children[1].textContent, '· 00:37');
    assert.equal(text.textContent, '<b>not markup</b>');
    assert.match(main.getAttribute('aria-label'), /^Reply to You, 00:37: /);
    main.click();
    assert.deepEqual(jumps, ['row:1@a#1']);
}));

test('the composer strip is the same panel with a lead word and a cancel control', () => withDom(() => {
    let cancelled = 0;
    const strip = createReplyQuote({ address: 'row:1@a#1', role: 'assistant', snippet: 'answer' },
        { composer: true, onCancel: () => { cancelled += 1; }, formatTime: () => null });
    assert.equal(strip.getAttribute('role'), 'group');
    assert.equal(strip.dataset.mode, 'composer');
    const [main, cancel] = strip.children;
    assert.equal(main.children[0].children[0].textContent, 'Replying to');
    assert.equal(cancel.getAttribute('aria-label'), 'Cancel reply');
    cancel.click();
    assert.equal(cancelled, 1);
}));

test('a reference with nothing to show draws nothing; a snippet is the first readable line cut at a word', () => withDom(() => {
    assert.equal(createReplyQuote(null), null);
    assert.equal(createReplyQuote({ role: 'user' }), null);
    assert.equal(normalizeReplyRef({ role: 'robot', snippet: 'x' }).author_label, '');
    assert.equal(replySnippet('# Title\nbody'), 'Title');
    assert.equal(replySnippet('one two three four', 9), 'one two…');
}));

test('a relay quote names the owner, where and when he wrote, carries his complete words, and only then offers Show all', () => withDom(() => {
    const words = 'проведи аудит\nи верни проекты к работе';
    const panel = createRelayQuote({ text: words, source: { label: 'Main', ts: 't' }, relay_id: 'r1' }, { formatTime: fixedTime });
    assert.equal(panel.dataset.mode, 'relay');
    assert.equal(panel.dataset.author, 'user');
    assert.equal(panel.getAttribute('role'), 'group');
    assert.equal(panel.getAttribute('aria-label'), 'Quoted message, You, Main, 00:37');
    const [head, text] = panel.children;
    assert.equal(head.children[0].textContent, 'You');
    assert.equal(head.children[1].textContent, '· Main · 00:37');
    assert.equal(text.textContent, words, 'the complete words, not a snippet');
    assert.ok(text.classList.contains('chat-clamp'));
    const toggle = head.children.find((node) => node.tagName === 'BUTTON');
    assert.ok(toggle?.hidden, 'the toggle stays hidden until a measure finds hidden text');
    assert.equal(createRelayQuote({ text: '   ' }), null, 'an empty relay draws nothing');
}));

test('a captionless photo or file relayed into a work still draws its panel: who, where, when and the files', () => withDom(() => {
    const panel = createRelayQuote({ text: '', attachments: [{ name: 'one.png', upload: 'abc' }],
        source: { label: 'Main', ts: 't' }, relay_id: 'r2' }, { formatTime: fixedTime });
    assert.ok(panel, 'his files are his message even without words');
    const [head, files, ...rest] = panel.children;
    assert.equal(head.children[0].textContent, 'You');
    assert.equal(head.children[1].textContent, '· Main · 00:37');
    assert.equal(files.className, 'chat-quote-files');
    assert.equal(files.textContent, 'Attached: one.png');
    assert.equal(rest.length, 0, 'no empty text line and no host placeholder');
    assert.equal(head.children.find((node) => node.tagName === 'BUTTON'), undefined, 'nothing to clamp, no Show all');
    assert.equal(createRelayQuote({ text: '', attachments: [{ upload: 'abc' }] }), null, 'a nameless manifest names nothing');
}));

test('a relay of several owner messages shows each of them, its own where, when, words and files, in the host order', () => withDom(() => {
    const first = { text: 'Please deploy', attachments: [], source: { label: 'Main', ts: 't' } };
    const second = { text: 'Use staging only', attachments: [{ name: 'config.json' }], source: { label: 'Alpha', ts: 't' } };
    // The envelope repeats its first quote in the singular fields for older readers; `quotes` is the whole truth.
    const group = createRelayQuote({ relay_id: 'r3', ...first, quotes: [first, second] }, { formatTime: fixedTime });
    assert.equal(group.className, 'chat-relay-quotes');
    const panels = group.children;
    assert.deepEqual(panels.map((panel) => panel.dataset.mode), ['relay', 'relay']);
    assert.deepEqual(panels.map((panel) => panel.querySelector('.chat-quote-text').textContent), ['Please deploy', 'Use staging only']);
    assert.deepEqual(panels.map((panel) => panel.children[0].children[1].textContent), ['· Main · 00:37', '· Alpha · 00:37']);
    assert.equal(panels[0].querySelector('.chat-quote-files'), null);
    assert.equal(panels[1].querySelector('.chat-quote-files').textContent, 'Attached: config.json');
    assert.ok(panels.every((panel) => panel.dataset.relayId === 'r3'));
    const one = createRelayQuote({ relay_id: 'r4', ...first, quotes: [first] }, { formatTime: fixedTime });
    assert.equal(one.dataset.mode, 'relay', 'one quoted message is one panel, no extra block around it');
    assert.equal(createRelayQuote({ relay_id: 'r5', text: '', quotes: [] }), null, 'a note without quoted words draws no panel');
}));

test('Show all opens the clamp, Show less closes it, through the caller\'s stable-viewport seam', () => withDom(() => {
    const writes = [];
    const panel = createRelayQuote({ text: 'long words', source: { label: 'Main' } },
        { onDomWrite: (write) => { writes.push(1); return write(); }, formatTime: () => null });
    const [head, text] = panel.children;
    const toggle = head.children.find((node) => node.tagName === 'BUTTON');
    assert.equal(toggle.getAttribute('aria-controls'), text.id);
    assert.equal(toggle.textContent, 'Show all');
    toggle.click();
    assert.equal(text.dataset.expanded, '1');
    assert.equal(toggle.getAttribute('aria-expanded'), 'true');
    assert.equal(toggle.textContent, 'Show less');
    assert.equal(toggle.hidden, false, 'an open quote can always be closed');
    toggle.click();
    assert.equal(text.dataset.expanded, undefined);
    assert.equal(toggle.textContent, 'Show all');
    assert.equal(writes.length, 2);
}));

test('the clamp puts its toggle into the row it is given, never a row of its own', () => withDom(() => {
    const row = document.createElement('div');
    const text = document.createElement('div');
    const toggle = clampWithToggle(text, row);
    assert.ok(text.classList.contains('chat-clamp'));
    assert.deepEqual(row.children, [toggle]);
    assert.equal(clampWithToggle(null, row), null);
}));

test('a quoted photo leads the panel with its thumbnail; only a safe upload URL becomes one', () => withDom(() => {
    const url = '/api/files/download?upload=0e3dabd5a6bf487ba054deef93921914_shot.png';
    const panel = createReplyQuote({ address: 'row:1@a#1', role: 'user', snippet: 'заведи проект', ts: 't',
        attachment: { name: 'shot.png', kind: 'image', mime: 'image/png', available: true, url } }, { formatTime: fixedTime });
    const [main] = panel.children;
    assert.ok(main.classList.contains('has-thumb'));
    const [thumb] = main.children;
    assert.equal(thumb.tagName, 'IMG');
    assert.equal(thumb.src, url);
    assert.equal(thumb.alt, '', 'decorative: the panel name already says what is quoted');
    const foreign = createReplyQuote({ address: 'row:1@a#2', role: 'user', snippet: 'x',
        attachment: { kind: 'image', available: true, url: 'https://example.com/a.png' } });
    assert.equal(foreign.children[0].classList.contains('has-thumb'), false, 'an arbitrary address never becomes an image');
    const photoOnly = createReplyQuote({ address: 'row:1@a#3', role: 'user',
        attachment: { kind: 'image', available: true, url } });
    assert.equal(photoOnly.children[0].children.at(-1).textContent, 'Photo', 'a photo with no words says Photo');
}));

test('a snippet cut never ends in punctuation before its ellipsis', () => {
    assert.equal(replySnippet('Берём A: сохраняю нынешний пузырь. Старые и новые фразы', 36), 'Берём A: сохраняю нынешний пузырь…');
});

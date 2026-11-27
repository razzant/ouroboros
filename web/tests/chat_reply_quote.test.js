// The quote panel (chat_reply_quote.js): ONE component for a reply (one line, a button that jumps), the strip above
// the composer (with cancel) and a relay (the owner's complete words, clamped, `Show all` only while it hides text).
// Layout is not measured here (no engine); the structure, the roles, the words-as-text and the toggle are.
import assert from 'node:assert/strict';
import test from 'node:test';
import { clampWithToggle, createRelayQuote, createReplyQuote, normalizeReplyRef, quoteSignatureOf, releaseClampFade, replyQuoteSignature,
    replySnippet } from '../modules/chat_reply_quote.js';
import { applyPayload, CODE_PREFIX, pendingMisses, setMissTransport } from '../modules/i18n.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

function withDom(run) {
    const env = installDom();
    try { return run(); } finally { restoreDom(env.prior); }
}
const fixedTime = () => ({ short: '00:37', full: 'Oct 12, 00:37' });

test('a sender selection names its source separately in the bubble and composer', () => withDom(() => {
    for (const composer of [true, false]) {
        const panel = createReplyQuote({ address: 'row:1@a#1', role: 'assistant',
            excerpt: { text: 'North quay', start_utf16: 12 } }, { composer });
        const main = panel.querySelector('.chat-quote-main');
        const head = main.querySelector('.chat-quote-head');
        const words = panel.querySelector('.chat-quote-text');
        assert.equal(main.tagName, 'BUTTON');
        assert.equal(main.querySelector('.chat-quote-text'), null);
        assert.equal(head.children[0].textContent, 'Selected by sender · Source:');
        assert.equal(head.children[1].textContent, 'Ouroboros');
        assert.equal(words.textContent, 'North quay');
        assert.equal(main.getAttribute('aria-label'), 'Selected by sender · Source: Ouroboros: North quay');
    }
}));

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

test('a quoted row in another room names that room and ends its head with the arrow a press follows', () => withDom(() => {
    const jumps = [];
    const raw = { address: 'row:1@q#1', role: 'assistant', kind: 'question', snippet: 'Which report format?', ts: 't' };
    const away = createReplyQuote({ ...raw, room_label: 'Main' }, { onJump: (ref) => jumps.push(ref.address), formatTime: fixedTime });
    const [main] = away.children;
    const head = main.children[0];
    assert.deepEqual([...head.children].map((part) => part.className),
        ['chat-quote-author', 'chat-quote-room', 'chat-quote-when', 'chat-quote-away']);
    assert.equal(head.children[1].textContent, '· Main');
    assert.equal(head.children[3].textContent, '↗');
    assert.equal(head.children[3].getAttribute('aria-hidden'), 'true');
    assert.equal(main.getAttribute('aria-label'), 'Reply to Ouroboros, opens in Main, 00:37: Which report format?');
    main.click();
    assert.deepEqual(jumps, ['row:1@q#1']);
    // The same reference without a room is the ordinary quote: nothing names a room or leaves.
    const here = createReplyQuote(raw, { formatTime: fixedTime }).children[0].children[0];
    assert.deepEqual([...here.children].map((part) => part.className), ['chat-quote-author', 'chat-quote-when']);
}));

test('a recorded choice has owner ink and its jump keeps the system row identity', () => withDom(() => {
    const raw = { address: 'row:1@a#1', role: 'system', author_label: 'You', snippet: 'First' };
    const jumps = [];
    const panel = createReplyQuote({ ...raw, quote_role: 'user' }, { onJump: ref => jumps.push(ref) });
    assert.equal(panel.dataset.author, 'user');
    panel.children[0].click();
    assert.equal(jumps[0].role, 'system');
    assert.equal(jumps[0].address, raw.address);
    assert.equal(createReplyQuote(raw).dataset.author, 'system');
    assert.equal(createReplyQuote({ ...raw, quote_role: 'unknown' }).dataset.author, 'system');
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

test('a CSS-only fit keeps authored child nodes intact through Show all and Show less', () => withDom(() => {
    const host = document.createElement('div');
    const line = document.createElement('p');
    const authored = document.createElement('span');
    authored.textContent = '  My EXACT words\n<as typed> 🙂  ';
    authored.dataset.i18nAuthored = '';
    line.append(authored);
    host.append(line);
    const fits = [];
    const toggle = clampWithToggle(line, host, { fit: (node, _text, lines, onFit) => {
        fits.push(lines);
        assert.equal(node.children[0], authored);
        onFit(lines > 0);
    } });
    assert.equal(toggle.hidden, false);
    toggle.click();
    toggle.click();
    assert.deepEqual(fits, [6, 0, 6]);
    assert.equal(authored.textContent, '  My EXACT words\n<as typed> 🙂  ');
    assert.equal(authored.dataset.i18nAuthored, '');
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

// The quote panel's one lifecycle: an expanded clamp wears the shared cue only while it hides text, and folding —
// or the owner's own release (releaseClampFade) — drops it. The shared binding's mechanics and its disposal are
// measured in scroll_fade.test.js.
test('an expanded clamp wears the shared cue only while open, and the owner release drops it', () => withDom(() => {
    const host = document.createElement('div');
    const text = document.createElement('div');
    const toggle = clampWithToggle(text, host, { fit: (_n, _t, _lines, onFit) => onFit(true), scrollRegion: 'Quotation' });
    assert.equal(text.classList.contains('scroll-fade-y'), false, 'a folded clamp carries no cue');
    toggle.click();  // open
    assert.equal(text.dataset.expanded, '1');
    assert.equal(text.classList.contains('scroll-fade-y'), true, 'the open quotation wears the cue');
    toggle.click();  // fold
    assert.equal(text.dataset.expanded, undefined);
    assert.equal(text.classList.contains('scroll-fade-y'), false, 'folding releases the cue');
    toggle.click();  // open again, then the owner removes the panel while it is open
    assert.equal(text.classList.contains('scroll-fade-y'), true);
    releaseClampFade(text);
    assert.equal(text.classList.contains('scroll-fade-y'), false, 'the owner release drops the cue');
}));

test('derived_question marks only automatically derived question quotes and tracks their signature', () => withDom(() => {
    const base = { address: 'row:1@q#1', role: 'assistant', kind: 'question', author_label: 'Ouroboros',
        snippet: 'Which format should I ship?', room_label: 'Main', ts: '2026-11-23T10:00:00Z' };
    const explicit = createReplyQuote(base, { formatTime: fixedTime });
    assert.equal(explicit.dataset.derivedQuestion, undefined, 'an explicit Reply carries no derived-question mark');
    assert.equal(normalizeReplyRef(base).derived_question, undefined);
    assert.equal(quoteSignatureOf(explicit), '', 'an ordinary quote is not signed: only the derived one is reconciled');

    const derived = createReplyQuote({ ...base, room_label_kind: 'main', derived_question: true }, { formatTime: fixedTime });
    assert.equal(derived.dataset.derivedQuestion, '1');
    assert.equal(derived.dataset.away, '1', 'cross-room derived question quote marks itself away for truncation rules');
    assert.equal(derived.dataset.derived, undefined, 'distinct from assistant host-derived quotes');
    assert.equal(quoteSignatureOf(derived), replyQuoteSignature({ ...base, room_label_kind: 'main', derived_question: true }));
    assert.notEqual(quoteSignatureOf(derived), replyQuoteSignature({ ...base, derived_question: true }),
        'host Main and a user project literally named Main have distinct signatures');
    assert.notEqual(quoteSignatureOf(derived), replyQuoteSignature({ ...base, room_label: 'Archive', derived_question: true }));
    assert.notEqual(quoteSignatureOf(derived), replyQuoteSignature({ ...base, room_label: '', derived_question: true }));
    assert.notEqual(quoteSignatureOf(derived), replyQuoteSignature({ ...base, snippet: 'Other line', derived_question: true }));

    const sameRoom = createReplyQuote({ ...base, room_label: '', derived_question: true }, { formatTime: fixedTime });
    assert.equal(sameRoom.dataset.derivedQuestion, '1');
    assert.equal(sameRoom.dataset.away, undefined, 'same-room derived question quote is not marked away');
    assert.equal(explicit.dataset.away, undefined, 'ordinary quotes are never marked away');

    const composer = createReplyQuote({ ...base, room_label_kind: 'main', derived_question: true }, { composer: true, formatTime: fixedTime });
    assert.equal(composer.dataset.derivedQuestion, undefined, 'composer strips are never marked as derived questions');
    assert.equal(composer.dataset.away, undefined, 'composer strips are never marked away');
    assert.equal(quoteSignatureOf(composer), '');
}));

test('a normalized ref round-trips through normalizeReplyRef without losing its room, kind or thumbnail', () => {
    const url = '/api/files/download?upload=0e3dabd5a6bf487ba054deef93921914_shot.png';
    const raw = { address: 'row:1@q#1', role: 'assistant', kind: 'question', author_label: 'Ouroboros',
        snippet: 'Which format?', room_label: 'Main', room_label_kind: 'main', ts: '2026-11-23T10:00:00Z',
        derived_question: true,
        attachment: { name: 'shot.png', kind: 'image', mime: 'image/png', available: true, url } };
    const once = normalizeReplyRef(raw);
    const twice = normalizeReplyRef(once);
    assert.equal(twice.room, 'Main', 'the room a rename compares survives a restored staged reply or echo re-read');
    assert.equal(twice.room_label_kind, 'main', 'the built-in room kind survives normalization round-trips');
    assert.equal(twice.media?.url, url, 'the quoted photo keeps its thumbnail through the same re-reads');
    assert.deepEqual(twice, once, 'every effective field of the automatically shown question is one shape');
    assert.equal(normalizeReplyRef({ ...raw, room_label_kind: 'unknown' }).room_label_kind, undefined,
        'unknown room_label_kind is ignored so it never translates as host chrome');
});

test('built-in Main and Project destination labels translate while identically spelled user projects stay verbatim', () => withDom(() => {
    setMissTransport(() => Promise.resolve(null));
    const EN = { language: '', english: true, revision: 0, entries: {} };
    const RU = {
        language: 'ru', english: false, revision: 3,
        entries: {
            Main: { text: 'Основной' },
            [CODE_PREFIX + 'project.reference.generic']: { text: 'Проект' },
            [CODE_PREFIX + 'chat.reply.reply_to']: { text: 'Ответ на' },
            'opens in {room}': { text: 'открывается в {room}' },
        },
    };
    const base = { address: 'row:1@q#1', role: 'assistant', kind: 'question', author_label: 'Ouroboros',
        snippet: 'Which format?', ts: 't', derived_question: true };
    try {
        applyPayload(RU);
        const cases = [
            { input: { ...base, room_label: 'Main', room_label_kind: 'main' }, shown: 'Основной' },
            { input: { ...base, room_label: 'Project', room_label_kind: 'project' }, shown: 'Проект' },
            { input: { ...base, room_label: 'Main' }, shown: 'Main' },
            { input: { ...base, room_label: 'Project' }, shown: 'Project' },
            { input: { ...base, room_label: 'Weekly report' }, shown: 'Weekly report' },
        ];
        for (const { input, shown } of cases) {
            const panel = createReplyQuote(input, { formatTime: fixedTime });
            const main = panel.querySelector('.chat-quote-main');
            assert.equal(panel.dataset.away, '1');
            assert.equal(main.dataset.i18nAuthored, '', 'mixed chrome/authored quote stays outside the DOM overlay');
            assert.equal(panel.querySelector('.chat-quote-room').textContent, `· ${shown}`);
            assert.equal(main.getAttribute('aria-label'),
                `Ответ на Ouroboros, открывается в ${shown}, 00:37: Which format?`,
                'visible and accessible destination labels agree');
        }
        assert.deepEqual(pendingMisses(), [], 'user-supplied project names never register translation misses');
        assert.notEqual(
            replyQuoteSignature({ ...base, room_label: 'Project', room_label_kind: 'project' }),
            replyQuoteSignature({ ...base, room_label: 'Project' }),
        );
        const ruAuthoredSig = replyQuoteSignature({ ...base, room_label: 'Main' });
        applyPayload(EN);
        assert.notEqual(ruAuthoredSig, replyQuoteSignature({ ...base, room_label: 'Main' }),
            'language/dictionary changes invalidate derived quote signatures so chrome words re-render too');
    } finally {
        applyPayload(EN);
    }
}));

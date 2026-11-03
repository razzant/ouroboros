// Read aloud (read_aloud.js): the reply's plain words (code announced and skipped), pieces at sentence ends, one reply
// at a time with Stop on the one being read, Panic and leaving the page stop it, no silent substitute for a chosen
// engine, the voice menu's honest destination per voice, and the voice door kept to the row menu.
import assert from 'node:assert/strict';
import test from 'node:test';
import { bindBubbleActions, createMessageActions } from '../modules/message_actions.js';
import { READ_ALOUD, readableText, registerReadAloud, speechChunks } from '../modules/read_aloud.js';
import { installDomDouble } from './dom_double.js';

// A rendered reply as the walker sees it: text nodes and elements with tag, classes and attributes.
const text = (value) => ({ nodeType: 3, textContent: value });
function node(tag, attrs, ...children) {
    const classes = String(attrs.class || '').split(/\s+/).filter(Boolean);
    return {
        nodeType: 1, tagName: tag.toUpperCase(), childNodes: children,
        matches(selector) {
            return selector.split(',').map((part) => part.trim()).some((part) => (part.startsWith('.')
                ? classes.includes(part.slice(1)) : part.startsWith('[') ? attrs[part.slice(1, part.indexOf('='))] === 'true'
                    : part === tag));
        },
    };
}

test('a reply reads as its words: code announced and skipped, links by their words, controls silent', () => {
    const reply = node('div', {},
        node('p', {}, text('Use '), node('code', {}, text('npm test')), text(' first.')),
        node('div', { class: 'md-code-block' }, node('button', {}, text('Copy')), node('pre', {}, text('npm run build'))),
        node('ul', {}, node('li', {}, text('See '), node('a', {}, text('the guide')), text('.')), node('li', {}, text('Done.'))),
        node('span', { 'aria-hidden': 'true' }, text('decor')),
        node('pre', {}, text('raw')));
    assert.equal(readableText(reply), 'Use npm test first.\nCode.\nSee the guide.\nDone.\nCode.');
});

test('speech pieces break at sentence ends and never exceed the bound', () => {
    assert.deepEqual(speechChunks('One. Two!\nThree?', 8), ['One.', 'Two!', 'Three?']);
    assert.deepEqual(speechChunks('One. Two!\nThree?', 9), ['One. Two!', 'Three?'], 'sentences share a piece that fits');
    assert.deepEqual(speechChunks('One. Two. Three.', 240), ['One. Two. Three.']);
    const long = `${'word '.repeat(120)}end.`;
    const pieces = speechChunks(long, 60);
    assert.ok(pieces.every((piece) => piece.length <= 60 && piece), pieces);
    assert.equal(pieces.join(' '), long.replace(/\s+/g, ' ').trim(), 'no word lost or doubled');
    assert.deepEqual(speechChunks('', 60), []);
});

function engine() {
    const said = [];
    const impl = {
        stops: 0,
        speak(words, options) {
            let finish;
            const done = new Promise((resolve) => { finish = resolve; });
            said.push({ words, options, finish });
            return done;
        },
        stop() { impl.stops += 1; },
        voices: () => [{ id: 'v-local', label: 'Milena', lang: 'ru-RU', local: true },
            { id: 'v-net', label: 'Google русский', lang: 'ru-RU', local: false }],
    };
    return { impl, said };
}

function hub(provider) {
    const listeners = new Set();
    return {
        provider, voice: '', chosenVoices: [],
        readAloud() { return this.provider; },
        voiceId() { return this.voice; },
        list: () => [],
        onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },
        changed() { for (const fn of [...listeners]) fn(); },
        async chooseVoice(id) { this.chosenVoices.push(id); this.voice = id; },
        async choose() {},
    };
}

const row = (address, words = 'First sentence. Second one.') => ({ address, role: 'assistant', text: words,
    content: node('div', {}, text(words)) });

test('Read aloud speaks one reply at a time, shows Stop on it, and stops on a second press, Panic or page hide', async () => {
    const env = installDomDouble();
    try {
        const registry = createMessageActions();
        const { impl, said } = engine();
        const speech = hub({ key: 'device_voice:device_voice', available: true, destination: 'device', impl });
        speech.voice = 'v-local';
        const off = registerReadAloud({ speech, registry });
        const action = registry.actionsFor(row('a')).find((item) => item.key === READ_ALOUD);
        assert.ok(action, 'offered on Ouroboros\'s replies');
        assert.equal(registry.actionsFor({ ...row('u'), role: 'user' }).length, 0, 'not on the owner\'s own rows');
        const first = action.run(row('a'));
        assert.equal(said.length, 1, 'the first piece is spoken inside the press');
        assert.deepEqual(said[0].words, 'First sentence. Second one.');
        assert.equal(said[0].options.voice, 'v-local', 'the chosen voice');
        assert.deepEqual(action.state(row('a')), { label: 'Stop', icon: 'stop', title: 'Stop reading aloud', pressed: true });
        assert.equal(action.state(row('b')), null);
        assert.equal(off.speaking(), 'a');
        // Another reply stops the first.
        void action.run(row('b', 'Other.'));
        assert.equal(said[0].options.signal.aborted, true);
        assert.equal(impl.stops, 1);
        assert.equal(off.speaking(), 'b');
        said[0].finish();
        await first;
        assert.equal(off.speaking(), 'b', 'the first ending does not clear the second');
        // A second press on the reply being read stops it.
        await action.run(row('b'));
        assert.equal(off.speaking(), '');
        assert.equal(impl.stops, 2);
        void action.run(row('c'));
        globalThis.window.emit('ouro:panic');
        assert.equal(off.speaking(), '', 'Panic stops reading');
        void action.run(row('d'));
        globalThis.window.emit('pagehide');
        assert.equal(off.speaking(), '', 'leaving the page stops reading');
        // Finishing every piece clears the state by itself.
        const done = action.run(row('e', 'One. Two.'));
        said.at(-1).finish();
        await Promise.resolve();
        off();
        await done;
        assert.equal(off.speaking(), '');
        assert.equal(registry.actionsFor(row('a')).length, 0, 'unregistered');
    } finally { env.restore(); }
});

test('a chosen engine that cannot run says why; none at all offers no action', async () => {
    const env = installDomDouble();
    try {
        const registry = createMessageActions();
        const speech = hub({ key: 'gone:tts', available: false, reason: 'The chosen engine is disabled, removed or awaiting review.' });
        const off = registerReadAloud({ speech, registry });
        const action = registry.actionsFor(row('a')).find((item) => item.key === READ_ALOUD);
        await assert.rejects(action.run(row('a')), /disabled, removed or awaiting review/);
        speech.provider = null;
        assert.equal(registry.actionsFor(row('a')).length, 0);
        off();
    } finally { env.restore(); }
});

test('the voice door is in the row menu only; its menu names each voice On this device or Online voice', async () => {
    const env = installDomDouble();
    try {
        const { doc, el } = env;
        const registry = createMessageActions();
        const { impl } = engine();
        const speech = hub({ key: 'device_voice:device_voice', available: true, destination: 'device', impl });
        const off = registerReadAloud({ speech, registry });
        const feed = el('div');
        doc.body.append(feed);
        const bubble = el('div', { className: 'chat-bubble assistant', dataset: { rowAddress: 'row:1@a' } },
            el('div', { className: 'message', text: 'Hello.' }),
            el('div', { className: 'chat-message-actions' }, el('button', { className: 'chat-message-reply' }),
                el('button', { className: 'chat-message-copy' })));
        feed.append(bubble);
        const bound = bindBubbleActions({ feed, registry });
        bound.sync(bubble);
        const strip = bubble.children[1];
        assert.deepEqual(strip.children.map((child) => child.dataset.messageAction || child.className),
            ['chat-message-reply', READ_ALOUD, 'chat-message-copy'], 'the bubble keeps one control for reading');
        const keys = registry.actionsFor(row('a')).map((item) => [item.key, item.label]);
        assert.deepEqual(keys, [[READ_ALOUD, 'Read aloud'], [`${READ_ALOUD}-voice`, 'Read aloud voice…']]);
        const door = registry.actionsFor(row('a'))[1];
        door.run(row('a'), { anchor: bubble });
        const menu = doc.body.querySelector('.voice-menu');
        const items = menu.children.map((item) => [item.children.map((part) => part.textContent), item.getAttribute('aria-checked')]);
        assert.deepEqual(items, [
            [['Automatic', 'A voice on this device'], 'true'],
            [['Milena', 'On this device · ru-RU'], 'false'],
            [['Google русский', 'Online voice · ru-RU'], 'false'],
        ], 'never local unless the browser says so');
        menu.children[2].click();
        await Promise.resolve();
        assert.deepEqual(speech.chosenVoices, ['v-net']);
        bound.destroy();
        off();
    } finally { env.restore(); }
});

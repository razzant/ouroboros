// Read aloud (docs/DESIGN.md "Replies"): a saved Ouroboros reply's action, beside Copy and in the row menu, that speaks
// the reply with the chosen text-to-speech engine (none chosen: a live one on this device). One reply at a time:
// starting another stops the first, the reply being read shows Stop in the action's place, and Panic or leaving the
// page stops it. The row menu's Read aloud voice… lists the engine's voices, each "On this device" only when the
// browser marks it local, otherwise "Online voice". The words are the reply's plain text: code blocks are announced
// as "code" and skipped, links read as their words.
import { tr } from './i18n.js';
import { bindMenu } from './ui_interactions.js';
import { getMessageActions } from './message_actions.js';
import { getSpeech, voiceDestination } from './speech_providers.js';

export const READ_ALOUD = 'read-aloud';
const BLOCKS = new Set(['P', 'DIV', 'LI', 'UL', 'OL', 'TR', 'TABLE', 'BLOCKQUOTE', 'BR', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6']);
const SKIPPED = 'button, svg, img, script, style, [aria-hidden="true"]';
const CODE = '.md-code-block, pre, .md-mermaid, .md-chart';

/** The words a listener should hear from a rendered reply `node`. */
export function readableText(node) {
    const parts = [];
    const walk = (current) => {
        for (const child of current?.childNodes || []) {
            if (child.nodeType === 3) parts.push(child.textContent);
            if (child.nodeType !== 1 || child.matches?.(SKIPPED)) continue;
            if (child.matches?.(CODE)) { parts.push(`\n${tr('read_aloud.code', 'Code.')}\n`); continue; }
            walk(child);
            if (BLOCKS.has(child.tagName)) parts.push('\n');
        }
    };
    walk(node);
    return parts.join('').replace(/[ \t ]+/g, ' ').replace(/ *\n[\s]*/g, '\n').trim();
}

/** `text` in pieces of at most `max` characters, at sentence ends where it can: a long utterance is cut short by
 * some engines' online voices. */
export function speechChunks(text, max = 240) {
    const chunks = [];
    let current = '';
    const add = (piece) => {
        if (current && current.length + 1 + piece.length > max) { chunks.push(current); current = ''; }
        current = current ? `${current} ${piece}` : piece;
    };
    for (const line of String(text || '').split(/\n+/)) {
        for (let sentence of line.match(/[^.!?…]+[.!?…]*/g) || []) {
            sentence = sentence.trim();
            while (sentence.length > max) {
                const cut = sentence.lastIndexOf(' ', max) > 0 ? sentence.lastIndexOf(' ', max) : max;
                add(sentence.slice(0, cut).trim());
                sentence = sentence.slice(cut).trim();
            }
            if (sentence) add(sentence);
        }
        if (current) { chunks.push(current); current = ''; }
    }
    return chunks;
}

/**
 * The Read aloud actions over `speech` (speech_providers.js hub) in `registry` (message_actions.js). Returns the
 * unregister; `stop()` and `speaking()` are on the returned function for the page and tests.
 */
export function registerReadAloud({ speech = getSpeech(), registry = getMessageActions(), win = globalThis.window,
    doc = globalThis.document } = {}) {
    let current = null;  // { address, abort, provider }

    function stop() {
        if (!current) return;
        const run = current;
        current = null;
        run.abort.abort();
        try { run.provider.impl?.stop?.(); } catch { /* the engine already ended */ }
        registry.changed(READ_ALOUD);
    }

    async function read(row) {
        if (current?.address === row.address) { stop(); return; }
        stop();
        const provider = speech.readAloud();
        if (!provider) throw new Error(tr('read_aloud.none', 'No voice is available on this device.'));
        if (!provider.available) throw new Error(provider.reason);
        const chunks = speechChunks(readableText(row.content) || row.text);
        if (!chunks.length) return;
        const run = { address: row.address, abort: new AbortController(), provider };
        current = run;
        registry.changed(READ_ALOUD);
        try {
            // The first piece is spoken inside the press (some engines need the gesture); the rest follow it.
            for (const chunk of chunks) {
                if (current !== run) return;
                await provider.impl.speak(chunk, { voice: speech.voiceId() || undefined, signal: run.abort.signal });
            }
        } finally {
            if (current === run) { current = null; registry.changed(READ_ALOUD); }
        }
    }

    function voiceMenu(anchor) {
        const provider = speech.readAloud();
        const menu = doc.createElement('div');
        menu.className = 'ui-popup voice-menu';
        menu.setAttribute('role', 'menu');
        menu.setAttribute('aria-label', tr('read_aloud.voice_menu', 'Read aloud voice'));
        let binding = null;
        const item = (label, line, checked, onSelect) => {
            const node = doc.createElement('button');
            node.type = 'button';
            node.className = 'voice-menu-item';
            node.setAttribute('role', 'menuitemradio');
            node.setAttribute('aria-checked', String(checked));
            for (const [className, text] of [['voice-menu-name', label], ['voice-menu-line', line]]) {
                const part = doc.createElement('span');
                part.className = className;
                part.textContent = text;
                node.append(part);
            }
            node.addEventListener('click', () => {
                binding?.close({ restoreFocus: true });
                void onSelect();
            });
            menu.append(node);
        };
        const engines = speech.list('tts').filter((row) => row.available);
        if (engines.length > 1) {
            for (const row of engines) item(row.label, row.destinationLabel, row.key === provider?.key,
                () => speech.choose('tts', row.key));
        }
        let voices = [];
        try { voices = provider?.available ? provider.impl.voices() : []; } catch { voices = []; }
        const chosen = speech.voiceId();
        item(tr('read_aloud.automatic', 'Automatic'), tr('read_aloud.automatic_line', 'A voice on this device'),
            !chosen, () => speech.chooseVoice(''));
        for (const voice of voices) {
            item(voice.label, [voiceDestination(voice), voice.lang].filter(Boolean).join(' · '), voice.id === chosen,
                () => speech.chooseVoice(voice.id));
        }
        if (!voices.length) {
            const note = doc.createElement('div');
            note.className = 'voice-menu-note';
            note.textContent = provider?.available ? tr('read_aloud.no_voices', 'No voices are available yet. Try again in a moment.')
                : (provider?.reason || tr('read_aloud.none', 'No voice is available on this device.'));
            menu.append(note);
        }
        doc.body.append(menu);
        binding = bindMenu(menu, { anchor, onClose: () => menu.remove() });
    }

    const offered = (row) => row?.role === 'assistant' && Boolean(speech.readAloud());
    const offs = [
        registry.register({ key: READ_ALOUD, label: tr('read_aloud.label', 'Read aloud'), icon: 'volume', order: 10,
            applies: offered, run: (row) => read(row),
            state: (row) => (current?.address === row.address ? { label: tr('read_aloud.stop', 'Stop'), icon: 'stop',
                title: tr('read_aloud.stop_title', 'Stop reading aloud'), pressed: true } : null) }),
        // Only in the row menu: the bubble keeps one control for reading.
        registry.register({ key: `${READ_ALOUD}-voice`, label: tr('read_aloud.voice', 'Read aloud voice…'), order: 11,
            surface: 'menu', applies: offered, run: (_row, ctx) => voiceMenu(ctx.anchor) }),
        speech.onChange(() => registry.changed(READ_ALOUD)),
    ];
    win?.addEventListener?.('ouro:panic', stop);
    win?.addEventListener?.('pagehide', stop);
    const unregister = () => {
        stop();
        for (const off of offs) off();
        win?.removeEventListener?.('ouro:panic', stop);
        win?.removeEventListener?.('pagehide', stop);
    };
    unregister.stop = stop;
    unregister.speaking = () => current?.address || '';
    return unregister;
}

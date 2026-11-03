// The dictation mic in a room's composer (voice_controls.js): present only while the chosen engine can run here, the
// recording face, words at the caret saved at Stop or an Insert/Copy offer when the draft moved, Esc / Panic / the
// room hiding / the engine leaving, the engine menu, "+" → Add dictation… and the gesture order with the real
// dictation controller.
import assert from 'node:assert/strict';
import test from 'node:test';
import { bindComposerPlus } from '../modules/composer_plus.js';
import { createDictation } from '../modules/dictation.js';
import { bindVoiceControls, clockText, spacedWords, startVoice } from '../modules/voice_controls.js';
import { installDomDouble } from './dom_double.js';
import { environment, stream } from './helpers/voice_env.js';

const LIVE = { key: 'whisper-voice:stt', skill: 'whisper-voice', label: 'Whisper', runtime: 'server', available: true,
    destinationLabel: 'On this computer', route_url: '/api/extensions/whisper-voice/transcribe' };
const BROWSER = { key: 'browser_speech:speech', skill: 'browser_speech', label: 'Browser speech', runtime: 'client',
    available: false, reason: 'Browser speech is unavailable in this browser or desktop app.', destinationLabel: 'Goes to Google' };

function room() {
    const env = installDomDouble();
    const { doc, el } = env;
    const input = el('textarea');
    Object.assign(input, { value: '', selectionStart: 0, selectionEnd: 0,
        setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; } });
    const slot = el('span', { className: 'chat-voice-slot', attrs: { 'data-composer-voice': '' } });
    slot.hidden = true;
    const plus = el('button', { attrs: { 'data-composer-plus': '' } });
    const wrap = el('div', { className: 'chat-input-wrap' }, el('div', { className: 'chat-text-row' }, plus, input, slot));
    doc.body.append(wrap);
    const composer = { input, voiceSlot: slot, insertAtCursor(text) {
        const { selectionStart: start, selectionEnd: end } = input;
        input.value = input.value.slice(0, start) + text + input.value.slice(end);
        input.selectionStart = input.selectionEnd = start + text.length;
    } };
    return { ...env, input, slot, wrap, composer };
}

function speechHub(rows, key = '') {
    const listeners = new Set();
    const hub = {
        rows, key, chosen: [], hosts: null,
        list: () => hub.rows,
        selected: () => hub.rows.find((row) => row.key === hub.key)
            || (hub.key ? { key: hub.key, available: false, reason: 'The chosen engine is disabled, removed or awaiting review.' }
                : { key: '', available: false, code: 'not_selected', reason: '' }),
        onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },
        changed() { for (const fn of [...listeners]) fn(); },
        async choose(kind, next) { hub.chosen.push([kind, next]); hub.key = next; hub.changed(); },
        attachHosts(hosts) { hub.hosts = hosts; },
        readAloud: () => null, voiceId: () => '',
    };
    return hub;
}

function fakeDictation() {
    const fake = { calls: [], opts: null };
    fake.make = (opts) => {
        fake.opts = opts;
        const idle = (reason) => opts.onState({ state: 'idle', reason });
        return {
            start(provider, options) { fake.calls.push(['start', provider.key, options.owner]); return Promise.resolve({ ok: true }); },
            stop() { fake.calls.push(['stop']); return Promise.resolve(); },
            cancel(reason) { fake.calls.push(['cancel', reason]); idle(reason); },
            suspend() { fake.calls.push(['suspend']); idle('hidden'); },
            destroy() { fake.calls.push(['destroy']); },
        };
    };
    return fake;
}

const labels = (slot) => slot.children.map((node) => node.getAttribute?.('aria-label') || node.textContent);

test('clock and spacing helpers', () => {
    assert.equal(clockText(0), '0:00');
    assert.equal(clockText(65_400), '1:05');
    assert.equal(spacedWords('Buy milk', 3, 'and eggs'), ' and eggs', 'a word before gets a space; the space after stays');
    assert.equal(spacedWords('Buy', 3, 'milk'), ' milk');
    assert.equal(spacedWords('', 0, 'Hello'), 'Hello');
    assert.equal(spacedWords('milk', 0, 'Buy'), 'Buy ');
});

test('the mic stands in its slot only while the chosen engine can run here; its menu lists engines with where audio goes', async () => {
    const r = room();
    try {
        const hub = speechHub([LIVE, BROWSER], '');
        const chooser = [];
        const fake = fakeDictation();
        const dispose = bindVoiceControls({ composer: r.composer, speech: hub, makeDictation: fake.make,
            chooser: (options) => chooser.push(options) });
        assert.equal(r.slot.hidden, true, 'no engine chosen: no mic (1A)');
        assert.equal(r.slot.children.length, 0);
        hub.key = BROWSER.key; hub.changed();
        assert.equal(r.slot.hidden, true, 'a chosen engine that cannot run here leaves no mic either');
        hub.key = LIVE.key; hub.changed();
        assert.equal(r.slot.hidden, false);
        assert.deepEqual(labels(r.slot), ['Dictate', 'Dictation engine']);
        const [mic, chevron] = r.slot.children;
        assert.equal(mic.title, 'Dictate · On this computer');
        assert.match(mic.innerHTML, /^<svg /);
        chevron.click();
        const menu = r.doc.body.querySelector('.voice-menu');
        assert.ok(menu, 'the chevron opens the engine menu');
        const items = menu.children.map((node) => [node.getAttribute('role'), node.children.map((part) => part.textContent),
            node.getAttribute('aria-checked'), node.getAttribute('aria-disabled')]);
        assert.deepEqual(items, [
            ['menuitemradio', ['Whisper', 'On this computer'], 'true', null],
            ['menuitemradio', ['Browser speech', BROWSER.reason], 'false', 'true'],
            ['menuitem', ['Get another engine…'], null, null],
            ['menuitem', ['Turn off dictation'], null, null],
        ], 'speech-to-text engines only, each with its destination or why it cannot run');
        menu.children[1].click();
        assert.deepEqual(hub.chosen, [], 'an engine that cannot run is not chosen');
        menu.children[2].click();
        assert.equal(chooser.length, 1, 'Get another engine… opens the chooser');
        assert.equal(chooser[0].composer, r.composer);
        r.slot.children[1].click();
        r.doc.body.querySelector('.voice-menu').children[3].click();
        await Promise.resolve();
        assert.deepEqual(hub.chosen, [['stt', '']], 'Turn off dictation clears the choice');
        assert.equal(r.slot.hidden, true);
        dispose();
        assert.ok(fake.calls.some(([name]) => name === 'destroy'));
    } finally { r.restore(); }
});

test('recording face, Stop, words at the saved caret or an Insert/Copy offer; nothing is sent', async () => {
    const r = room();
    try {
        const hub = speechHub([LIVE], LIVE.key);
        const fake = fakeDictation();
        const copied = [];
        bindVoiceControls({ composer: r.composer, speech: hub, makeDictation: fake.make, copy: (text) => copied.push(text) });
        r.slot.children[0].click();
        assert.deepEqual(fake.calls[0], ['start', LIVE.key, 'dictation:whisper-voice:stt']);
        fake.opts.onState({ state: 'permission' });
        assert.deepEqual(labels(r.slot), ['Recording', 'Stop and transcribe', 'Cancel dictation']);
        assert.equal(r.slot.children[1].disabled, true, 'Stop waits for the microphone');
        fake.opts.onState({ state: 'recording', startedAt: Date.now() });
        assert.equal(r.slot.dataset.voiceState, 'recording');
        const [time, stop] = r.slot.children;
        assert.equal(time.getAttribute('role'), 'timer');
        assert.equal(time.children[1].textContent, '0:00');
        assert.equal(stop.disabled, false);
        assert.match(stop.innerHTML, /^<svg /, 'Stop stands in the mic\'s place');
        stop.click();
        assert.deepEqual(fake.calls.at(-1), ['stop']);
        fake.opts.onState({ state: 'transcribing' });
        assert.deepEqual(labels(r.slot), ['Transcribing…', 'Cancel dictation']);
        // The draft did not change since Stop: the words go in at the saved caret, spaced from their neighbours.
        r.input.value = 'Buy milk';
        fake.opts.onState({ state: 'idle' });
        fake.opts.onTranscript({ kind: 'insert', text: 'and eggs', start: 8, end: 8 });
        assert.equal(r.input.value, 'Buy milk and eggs');
        assert.deepEqual(labels(r.slot), ['Dictate', 'Dictation engine'], 'the mic returns');
        // The draft moved meanwhile: an offer holds the words; Insert puts them at the caret now.
        fake.opts.onTranscript({ kind: 'offer', text: 'call mum', actions: ['Insert', 'Copy'] });
        const offer = r.wrap.querySelector('.voice-offer');
        assert.ok(offer, 'the offer waits above the composer');
        assert.equal(r.input.value, 'Buy milk and eggs', 'the draft is not touched by an offer');
        assert.equal(offer.querySelector('.voice-offer-text').textContent, 'call mum');
        assert.deepEqual(offer.querySelector('.voice-offer-actions').children.map((node) => node.getAttribute('aria-label')),
            ['Insert', 'Copy', 'Discard dictated text']);
        r.input.setSelectionRange(0, 0);
        offer.querySelector('.voice-offer-actions').children[0].click();
        assert.equal(r.input.value, 'call mum Buy milk and eggs');
        assert.equal(r.wrap.querySelector('.voice-offer'), null);
        fake.opts.onTranscript({ kind: 'offer', text: 'second thought' });
        r.wrap.querySelector('.voice-offer-actions').children[1].click();
        assert.deepEqual(copied, ['second thought'], 'Copy copies and leaves the draft');
        assert.equal(r.input.value, 'call mum Buy milk and eggs');
        fake.opts.onTranscript({ kind: 'offer', text: 'gone' });
        r.wrap.querySelector('.voice-offer').emit('keydown', { key: 'Escape' });
        assert.equal(r.wrap.querySelector('.voice-offer'), null, 'Esc discards an offer');
    } finally { r.restore(); }
});

test('Esc, Panic, the room hiding and the engine leaving end a recording', () => {
    const r = room();
    try {
        const hub = speechHub([LIVE], LIVE.key);
        const fake = fakeDictation();
        const notices = [];
        bindVoiceControls({ composer: r.composer, speech: hub, makeDictation: fake.make,
            notice: (text, tone) => notices.push([text, tone]) });
        const record = () => { r.slot.children[0].click(); fake.opts.onState({ state: 'recording', startedAt: Date.now() }); };
        record();
        const esc = r.input.emit('keydown', { key: 'Escape' });
        assert.deepEqual(fake.calls.at(-1), ['cancel', 'cancelled']);
        assert.equal(esc.defaultPrevented, true);
        r.input.emit('keydown', { key: 'Escape' });
        assert.equal(fake.calls.filter(([name]) => name === 'cancel').length, 1, 'Esc while idle is not the mic\'s');
        record();
        globalThis.window.emit('ouro:panic');
        assert.deepEqual(fake.calls.at(-1), ['cancel', 'panic']);
        record();
        globalThis.window.emit('ouro:page-shown');
        assert.notDeepEqual(fake.calls.at(-1), ['suspend'], 'a visible room keeps recording');
        r.wrap.hidden = true;
        globalThis.window.emit('ouro:page-shown');
        assert.deepEqual(fake.calls.at(-1), ['suspend'], 'a hidden room releases the microphone');
        r.wrap.hidden = false;
        record();
        hub.rows = [];
        hub.changed();
        assert.deepEqual(fake.calls.at(-1), ['cancel', 'engine']);
        assert.deepEqual(notices.at(-1), ['The chosen engine is disabled, removed or awaiting review.', 'warn'],
            'the engine leaving says why; nothing else is chosen');
        assert.equal(r.slot.hidden, true);
    } finally { r.restore(); }
});

test('the mic press starts capture inside the gesture and the transcript lands at the caret (real controller)', async () => {
    const r = room();
    try {
        const env = environment();
        let acquiredInGesture = false;
        let inGesture = false;
        const capture = {
            acquire() {
                acquiredInGesture = inGesture;
                const handle = { stream: stream(), signal: new AbortController().signal, release() {} };
                return Object.assign(Promise.resolve(handle), { release() {} });
            },
        };
        const recorder = { start: () => ({ result: new Promise(() => {}), stop: async () => new Blob(['x'], { type: 'audio/webm' }) }),
            cancel() {} };
        const makeDictation = (opts) => createDictation({ ...opts, env, capture, makeRecorder: () => recorder,
            transcribe: async () => ({ text: 'and bread' }) });
        const hub = speechHub([LIVE], LIVE.key);
        bindVoiceControls({ composer: r.composer, speech: hub, makeDictation });
        r.input.value = 'Milk';
        r.input.setSelectionRange(4, 4);
        inGesture = true;
        r.slot.children[0].click();
        inGesture = false;
        assert.equal(acquiredInGesture, true, 'getUserMedia (the lease) is asked before the press handler returns');
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(r.slot.dataset.voiceState, 'recording');
        r.slot.children[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(r.input.value, 'Milk and bread');
        assert.equal(r.slot.dataset.voiceState, 'idle');
    } finally { r.restore(); }
});

test('"+" offers Add dictation… while no engine can dictate here, and stops once one can', async () => {
    const r = room();
    try {
        const hub = speechHub([{ ...LIVE, available: false, reason: 'not yet' }], '');
        const sections = [];
        const store = { byPlacement: () => [], voice: () => null, onChange: () => () => {} };
        const composer = { ...r.composer, plusMenu: { setSections: (next) => sections.push(next) } };
        const offPlus = bindComposerPlus({ composer, store });
        const hosts = { clientProviders: () => [], onProvidersChange: () => () => {} };
        const stop = startVoice({ hosts, speech: hub });
        assert.equal(hub.hosts, hosts, 'page modules\' providers join the speech hub');
        const keys = () => sections.at(-1).at(-1).items.map((item) => item.key);
        assert.deepEqual(keys(), ['dictation', 'make-button']);
        hub.key = LIVE.key; hub.changed();
        assert.deepEqual(keys(), ['dictation', 'make-button'], 'chosen but not runnable: the way to another engine stays');
        hub.rows = [LIVE]; hub.changed();
        assert.deepEqual(keys(), ['make-button'], 'an engine that dictates here: the mic is the way in');
        stop();
        offPlus();
    } finally { r.restore(); }
});

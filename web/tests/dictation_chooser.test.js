// "+" → Add dictation… (dictation_chooser.js) and the page's speech hub (speech_providers.js createSpeechHub): live
// engines are chosen at once, an installed built-in engine that is off is turned on through the ordinary Skills toggle
// and then chosen, a built-in not installed yet and Whisper are reached through the Skills page (the installed entry, or
// the Browse search where they install), a failed start chooses nothing, and the hub's Read aloud engine never
// substitutes for a chosen one.
import assert from 'node:assert/strict';
import test from 'node:test';
import { HUB_STT_ENGINES, chooseOption, chooserOptions, openDictationChooser } from '../modules/dictation_chooser.js';
import { createSpeechHub } from '../modules/speech_providers.js';
import { installDomDouble } from './dom_double.js';
import { environment } from './helpers/voice_env.js';

const LOCAL = { hostname: 'localhost', host: 'localhost:8765' };
const BROWSER_OFFER = { skill: 'browser_speech', skill_label: 'Browser speech', id: 'speech', label: 'Browser speech',
    title: 'Free. Your browser recognizes speech: Chrome sends audio to Google.', destination: 'vendor:Browser speech service',
    install_size: '', enabled: false };

function hubReading({ live = [], offers = [], chosen = { key: '', available: false, reason: '' } } = {}) {
    const calls = [];
    return {
        calls,
        list: () => live,
        offers: () => offers,
        selected: () => chosen,
        choose: async (kind, key) => { calls.push(['choose', kind, key]); },
        enableOffer: async (offer) => { calls.push(['enable', offer.skill]); },
    };
}

test('the chooser lists live engines, bundled engines that are off and Whisper, each with where audio goes', () => {
    const speech = hubReading({
        live: [{ key: 'demo:speech', skill: 'demo', label: 'Demo', available: true, destinationLabel: 'On this computer' },
            { key: 'gone:speech', skill: 'gone', label: 'Gone', available: false, reason: 'Not here.' }],
        offers: [BROWSER_OFFER, { ...BROWSER_OFFER, skill: 'demo', id: 'speech' }],
        chosen: { key: 'demo:speech', available: true },
    });
    const { rows } = chooserOptions({ speech, installed: [], location: LOCAL });
    assert.deepEqual(rows.map((row) => [row.kind, row.label, row.line, row.hint, Boolean(row.disabled)]), [
        ['live', 'Demo', 'On this computer', 'Chosen', false],
        ['live', 'Gone', 'Not here.', 'Use', true],
        ['offer', 'Browser speech', BROWSER_OFFER.title, 'Turn on and use', false],
        ['hub', 'Whisper', 'On this computer · ~500 MB · OuroborosHub', 'Find Whisper in Skills', false],
    ], 'an offer that is already live is not offered twice; the browser path is the skill\'s own words');
    const installed = chooserOptions({ speech, installed: [{ name: 'whisper-voice', enabled: false }], location: LOCAL });
    assert.equal(installed.rows.at(-1).hint, 'Open Whisper in Skills', 'installed but off: its Skills entry');
    const remote = chooserOptions({ speech, installed: null, location: { hostname: 'box.lan', host: 'box.lan:8765' } });
    assert.equal(remote.rows.at(-1).line, 'On your Ouroboros server (box.lan:8765) · ~500 MB · OuroborosHub');
    const live = hubReading({ live: [{ key: 'whisper-voice:stt', skill: 'whisper-voice', label: 'Whisper', available: true,
        destinationLabel: 'On this computer' }] });
    assert.deepEqual(chooserOptions({ speech: live, location: LOCAL }).rows.map((row) => row.kind), ['live'],
        'a live Whisper is an engine, not an install offer');
    assert.equal(HUB_STT_ENGINES[0].slug, 'whisper-voice');
});

test('choosing: live → saved; bundled → turned on then saved; Whisper → the Skills page', async () => {
    const speech = hubReading({ offers: [BROWSER_OFFER] });
    const opened = [];
    const open = (detail) => opened.push(detail);
    assert.match(await chooseOption({ kind: 'live', key: 'demo:speech', label: 'Demo' }, { speech, open }), /Press the mic/);
    assert.match(await chooseOption({ kind: 'offer', label: 'Browser speech', offer: BROWSER_OFFER }, { speech, open }), /is on/);
    assert.deepEqual(speech.calls, [['choose', 'stt', 'demo:speech'], ['enable', 'browser_speech']]);
    const whisper = { kind: 'hub', engine: HUB_STT_ENGINES[0] };
    assert.equal(await chooseOption({ ...whisper, installed: false }, { speech, open }), '');
    await chooseOption({ ...whisper, installed: true }, { speech, open });
    assert.deepEqual(opened, [{ query: 'whisper-voice' }, { skill: 'whisper-voice' }]);
});

test('a built-in engine this install has not installed is found in Browse, never toggled', async () => {
    const absent = { ...BROWSER_OFFER, installed: false };
    const speech = hubReading({ offers: [absent] });
    const [row] = chooserOptions({ speech, installed: [], location: LOCAL }).rows;
    assert.deepEqual([row.kind, row.hint], ['offer', 'Find Browser speech in Skills']);
    const opened = [];
    assert.equal(await chooseOption(row, { speech, open: (detail) => opened.push(detail) }), '');
    assert.deepEqual(opened, [{ query: 'browser_speech' }]);
    assert.deepEqual(speech.calls, [], 'a toggle cannot turn on a skill that is not installed');
    const installedOff = chooserOptions({ speech: hubReading({ offers: [BROWSER_OFFER] }), installed: [], location: LOCAL });
    assert.equal(installedOff.rows[0].hint, 'Turn on and use', 'an installed built-in that is off is turned on');
});

test('the chooser sheet says why a chosen engine cannot dictate and turns Browser speech on from a press', async () => {
    const env = installDomDouble();
    try {
        const { doc, el } = env;
        const input = el('textarea');
        const plus = el('button', { attrs: { 'data-composer-plus': '' } });
        doc.body.append(el('div', { className: 'chat-text-row' }, plus, input));
        const speech = hubReading({ offers: [BROWSER_OFFER], chosen: { key: 'gone:speech', available: false, reason: 'Not here.' } });
        const notices = [];
        await openDictationChooser({ composer: { input }, speech, listing: async () => { throw new Error('offline'); },
            notice: (text, tone) => notices.push([text, tone]) });
        const menu = doc.body.querySelector('.voice-chooser');
        assert.equal(menu.getAttribute('role'), 'menu');
        assert.deepEqual(menu.children.slice(0, 2).map((node) => node.textContent),
            ['Choose how to dictate', 'Your chosen engine cannot dictate here: Not here.']);
        const options = menu.querySelectorAll('[data-voice-option]');
        assert.deepEqual(options.map((node) => node.dataset.voiceOption), ['browser_speech:speech', 'hub:whisper-voice'],
            'a failed Skills read still offers the Browse search');
        options[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(speech.calls, [['enable', 'browser_speech']]);
        assert.deepEqual(notices, [['Browser speech is on. Press the mic to dictate.', 'ok']]);
        assert.equal(doc.body.querySelector('.voice-chooser'), null, 'the sheet closes on a choice');
    } finally { env.restore(); }
});

function storeDouble({ providers = [], offers = [], voice = {} } = {}) {
    const listeners = new Set();
    const store = {
        reads: 0, rows: providers, prefs: { voice: { stt: '', tts: '', tts_voice: '', ...voice } },
        providers: (kind) => store.rows.filter((row) => row.provides === kind),
        fault: () => '',
        offers: () => offers,
        preferences: () => store.prefs,
        onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },
        async refresh() { store.reads += 1; for (const fn of [...listeners]) fn(store); return true; },
    };
    return store;
}

test('the hub turns a bundled engine on through the Skills toggle, then chooses it; a failed start chooses nothing', async () => {
    const store = storeDouble({ offers: [BROWSER_OFFER] });
    const posts = [], saves = [], emitted = [];
    let answer = { skill: 'browser_speech', enabled: true, extension_action: 'extension_loaded' };
    const speech = createSpeechHub({ store, env: environment(), post: async (url, body) => { posts.push([url, body]); return answer; },
        save: async (payload) => { saves.push(payload); store.prefs = { voice: { ...store.prefs.voice, ...payload.voice } }; },
        emit: (name, result) => emitted.push([name, result.extension_action]) });
    let notices = 0;
    speech.onChange(() => { notices += 1; });
    await speech.enableOffer(BROWSER_OFFER);
    assert.deepEqual(posts, [['/api/skills/browser_speech/toggle', { enabled: true }]]);
    assert.deepEqual(emitted, [['browser_speech', 'extension_loaded']], 'the Skills page and the store hear it');
    assert.deepEqual(saves, [{ voice: { stt: 'browser_speech:speech' } }]);
    assert.ok(store.reads >= 1 && notices >= 1, 'the projection is read again and every control hears it');
    answer = { skill: 'browser_speech', enabled: true, extension_action: 'extension_load_error', extension_reason: 'boom' };
    await assert.rejects(speech.enableOffer(BROWSER_OFFER), /boom/);
    assert.equal(saves.length, 1, 'an engine that did not start is not chosen');
});

test('the hub merges page-module providers, and Read aloud uses the chosen engine or, with none chosen, a device one', async () => {
    const device = { key: 'device_voice:device_voice', skill: 'device_voice', provides: 'tts', runtime: 'client',
        label: 'Device voice', destination: 'device' };
    const online = { ...device, key: 'cloud:tts', skill: 'cloud', label: 'Cloud', destination: 'vendor:Cloud' };
    const store = storeDouble({ providers: [online, device] });
    const saves = [];
    const speech = createSpeechHub({ store, env: environment(), save: async (payload) => {
        saves.push(payload); store.prefs = { voice: { ...store.prefs.voice, ...payload.voice } };
    } });
    assert.equal(speech.readAloud(), null, 'no page module registered yet: nothing can read');
    const providerListeners = new Set();
    const impl = { label: 'Device voice', voices: () => [], speak: async () => {}, stop() {} };
    let registered = [];
    const hosts = { clientProviders: (kind) => registered.filter((row) => row.kind === kind),
        onProvidersChange(fn) { providerListeners.add(fn); return () => providerListeners.delete(fn); } };
    let notices = 0;
    speech.onChange(() => { notices += 1; });
    speech.attachHosts(hosts);
    registered = [{ key: device.key, kind: 'tts', impl }, { key: online.key, kind: 'tts', impl: { ...impl, label: 'Cloud' } }];
    for (const fn of providerListeners) fn();
    assert.ok(notices >= 2, 'a page module registering is a change');
    assert.equal(speech.readAloud().key, device.key, 'nothing chosen: a live engine on this device');
    registered = [{ key: online.key, kind: 'tts', impl: { ...impl, label: 'Cloud' } }];
    for (const fn of providerListeners) fn();
    assert.equal(speech.readAloud(), null, 'nothing chosen and only an engine that sends words away: none');
    registered = [{ key: device.key, kind: 'tts', impl }, { key: online.key, kind: 'tts', impl: { ...impl, label: 'Cloud' } }];
    for (const fn of providerListeners) fn();
    await speech.chooseVoice('v-1');
    assert.equal(speech.voiceId(), 'v-1');
    await speech.choose('tts', online.key);
    assert.deepEqual(saves.at(-1), { voice: { tts: online.key, tts_voice: '' } }, 'another engine starts from its own voice pick');
    assert.equal(speech.readAloud().key, online.key);
    store.rows = [device];
    const gone = speech.readAloud();
    assert.equal(gone.key, online.key, 'a chosen engine that left is not replaced');
    assert.equal(gone.available, false);
    assert.match(gone.reason, /disabled, removed or awaiting review/);
});

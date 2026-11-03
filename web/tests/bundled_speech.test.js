import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { environment } from './helpers/voice_env.js';

// Host modules are served as ESM outside web's package.json too. Load the exact
// shipped source as ESM, without adding Node-only files to the skill payload.
async function moduleAt(path) {
    const source = await readFile(new URL(path, import.meta.url), 'utf8');
    return import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
}
const stt = await moduleAt('../../skills/browser_speech/ui/stt.js');
const tts = await moduleAt('../../skills/device_voice/ui/tts.js');

class Recognition {
    static instances = [];
    constructor() { Recognition.instances.push(this); this.aborts = 0; this.stops = 0; }
    start() { this.started = true; }
    stop() { this.stops++; }
    abort() { this.aborts++; }
    words(text, index = 0) {
        const results = Array.from({ length: index + 1 }, () => Object.assign([{ transcript: '' }], { isFinal: false }));
        results[index] = Object.assign([{ transcript: text }], { isFinal: true });
        this.onresult({ resultIndex: index, results });
    }
}

test('browser destinations distinguish Edge, Chrome, Safari and unknown engines', () => {
    assert.equal(stt.recognitionDestination('Chrome/100 Safari/500 Edg/100'), 'vendor:Microsoft');
    assert.equal(stt.recognitionDestination('Chrome/100 Safari/500'), 'vendor:Google');
    assert.equal(stt.recognitionDestination('Version/18 Safari/605'), 'vendor:Apple (on device when available)');
    assert.equal(stt.recognitionDestination('other'), 'vendor:Browser speech service');
    assert.equal(stt.createRecognition({}).available().code, 'no_speech_api');
});

test('recognition starts immediately, keeps final results in order, stops once and removes signal handlers', async () => {
    const env = environment(); env.SpeechRecognition = Recognition;
    const provider = stt.createRecognition(env), abort = new AbortController();
    const session = provider.start({ lang: 'ru', signal: abort.signal });
    const native = Recognition.instances.at(-1);
    assert.equal(native.started, true); assert.equal(native.lang, 'ru');
    native.words('two', 1); native.words('one', 0);
    const result = session.stop(); session.stop();
    assert.equal(native.stops, 1);
    native.onend();
    assert.deepEqual(await result, { text: 'one two' });
    abort.abort(); assert.equal(native.aborts, 0); assert.equal(env.timers.size, 0);
});

test('no_speech is empty, permission/network failure rejects, cancellation blocks late recognition', async () => {
    const env = environment(); env.webkitSpeechRecognition = Recognition;
    const provider = stt.createRecognition(env);
    const empty = provider.start(); Recognition.instances.at(-1).onerror({ error: 'no-speech' });
    assert.deepEqual(await empty.stop(), { no_speech: true });
    for (const code of ['not-allowed', 'network']) {
        const failed = provider.start(); Recognition.instances.at(-1).onerror({ error: code });
        await assert.rejects(failed.stop(), error => error.code === code);
    }
    const cancelled = provider.start(), native = Recognition.instances.at(-1);
    cancelled.cancel();
    assert.equal(native.onresult, null); assert.equal(native.aborts, 1);
    await assert.rejects(cancelled.stop(), { name: 'AbortError' });
    const okay = provider.start(); Recognition.instances.at(-1).words('next'); Recognition.instances.at(-1).onend();
    assert.deepEqual(await okay.stop(), { text: 'next' });
});

test('unresponsive browser stop times out and aborts its native session', async () => {
    const env = environment(); env.SpeechRecognition = Recognition;
    const provider = stt.createRecognition(env), session = provider.start();
    const result = session.stop(); env.fireTimer(10000);
    await assert.rejects(result, /did not finish/);
    assert.equal(Recognition.instances.at(-1).aborts, 1); assert.equal(env.timers.size, 0);
});

function voiceEnv() {
    const env = environment();
    let voices = [
        { voiceURI: 'online', name: 'Online', lang: 'en', localService: false },
        { voiceURI: 'local', name: 'Local', lang: 'ru', localService: true },
    ];
    const spoken = [];
    let cancels = 0;
    env.speechSynthesis = { getVoices: () => voices, speak: value => spoken.push(value), cancel: () => cancels++ };
    env.SpeechSynthesisUtterance = class { constructor(text) { this.text = text; } };
    return { env, spoken, setVoices: value => { voices = value; }, cancels: () => cancels };
}

test('device voice prefers local speech and marks online voices honestly', async () => {
    const fixture = voiceEnv(), provider = tts.createDeviceVoice(fixture.env);
    assert.deepEqual(provider.voices().map(voice => [voice.id, voice.local]), [['local', true], ['online', false]]);
    const first = provider.speak('hello');
    assert.equal(fixture.spoken[0].voice.voiceURI, 'local');
    fixture.spoken[0].onend(); await first;
    const second = provider.speak('hello online', { voice: 'online' });
    assert.equal(fixture.spoken[1].voice.localService, false);
    provider.stop(); await second;
    assert.equal(fixture.cancels(), 1);
});

test('one reading at a time, abort and dispose settle even if native cancel has no callback', async () => {
    const fixture = voiceEnv(), provider = tts.createDeviceVoice(fixture.env), abort = new AbortController();
    const first = provider.speak('first', { signal: abort.signal });
    const oldEnd = fixture.spoken[0].onend;
    const second = provider.speak('second');
    await first;
    abort.abort(); oldEnd();
    assert.equal(fixture.cancels(), 1, 'old signal and callback cannot cancel the newer utterance');
    provider.stop(); await second;
    assert.equal(fixture.cancels(), 2);
});

test('missing API and removed chosen voice are explicit; voices can appear after mounting', async () => {
    assert.equal(tts.createDeviceVoice({}).available().code, 'no_voice_api');
    const fixture = voiceEnv(), provider = tts.createDeviceVoice(fixture.env);
    await assert.rejects(provider.speak('text', { voice: 'gone' }), /no longer available/);
    assert.equal(fixture.spoken.length, 0);
    fixture.setVoices([]); await assert.rejects(provider.speak('text'), /No voices/);
    fixture.setVoices([{ voiceURI: 'new', name: 'New', localService: false }]);
    assert.equal(provider.voices()[0].local, false);
    await assert.rejects(provider.speak('text'), /No voice on this device/, 'only online voices: none is picked unasked');
    assert.equal(fixture.spoken.length, 0);
    const result = provider.speak('text', { voice: 'new' }); provider.stop(); await result;
    assert.equal(fixture.spoken[0].voice.voiceURI, 'new', 'a chosen online voice still speaks');
});

test('both host modules use API 1 registration and idempotent abort/dispose only', () => {
    for (const [module, kind] of [[stt, 'stt'], [tts, 'tts']]) {
        const abort = new AbortController(), registrations = [];
        let unregistered = 0;
        const mounted = module.mount({ api: 1, skill: 'fixture', key: 'fixture:voice', signal: abort.signal,
            providers: { register: (...args) => { registrations.push(args); return () => unregistered++; } } });
        assert.equal(registrations[0][0], kind);
        abort.abort(); mounted.dispose();
        assert.equal(unregistered, 1);
        assert.throws(() => module.mount({ api: 2 }), /API 1/);
    }
});

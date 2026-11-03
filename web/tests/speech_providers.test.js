import test from 'node:test';
import assert from 'node:assert/strict';
import { createSpeechProviders, destinationLabel, voiceDestination, transcribeAudio } from '../modules/speech_providers.js';
import { Recorder, environment } from './helpers/voice_env.js';

const server = { key: 'local:speech', provides: 'stt', runtime: 'server', label: 'Local',
    route_url: '/api/extensions/local/transcribe', destination: 'host', formats: ['audio/webm'], max_bytes: 30 };
const client = { key: 'browser:speech', provides: 'stt', runtime: 'client', destination: 'vendor:Browser', label: 'Browser' };
const impl = { label: 'Browser speech', destination: 'vendor:Google', available: () => ({ ok: true }), start() {} };

test('live projections merge exact client keys; chosen missing/disabled engines never silently fall back', () => {
    const env = environment(); env.MediaRecorder = Recorder;
    let rows = [server, client], prefs = { voice: { stt: client.key } };
    const registry = createSpeechProviders({ source: () => rows, preferences: () => prefs, env });
    assert.equal(registry.selected('stt').code, 'not_loaded');
    const off = registry.register(client.key, 'stt', impl);
    assert.equal(registry.selected('stt').available, true);
    assert.equal(registry.selected('stt').destinationLabel, 'Goes to Google');
    assert.equal(registry.list('stt').length, 2);
    rows = [server];
    assert.equal(registry.selected('stt').code, 'missing');
    assert.equal(registry.selected('stt').key, client.key);
    rows = [server, client]; off();
    assert.equal(registry.selected('stt').code, 'not_loaded');
    prefs = { voice: { stt: '' } };
    assert.equal(registry.selected('stt').code, 'not_selected');
});

test('provider updates and unregister retain the latest implementation only', () => {
    const env = environment();
    const registry = createSpeechProviders({ source: () => [client], env });
    const old = registry.register(client.key, 'stt', impl);
    const next = registry.register(client.key, 'stt', { ...impl, available: () => ({ ok: false, code: 'no_speech_api' }) });
    old();
    assert.equal(registry.list('stt')[0].code, 'no_speech_api');
    assert.match(registry.list('stt')[0].reason, /unavailable/);
    next(); assert.equal(registry.list('stt')[0].code, 'not_loaded');
});

test('external host registrations still require a live same-key contribution and matching kind', () => {
    const env = environment();
    const registry = createSpeechProviders({ source: () => [client], clients: () => [
        { key: client.key, kind: 'tts', impl }, { key: 'disabled:stt', kind: 'stt', impl }], env });
    assert.equal(registry.list('stt').length, 1);
    assert.equal(registry.list('stt')[0].available, false);
    assert.deepEqual(registry.list('tts'), []);
});

test('capability loss gives a reason without changing the saved choice', () => {
    const env = environment(); env.MediaRecorder = Recorder;
    const registry = createSpeechProviders({ source: () => [server], preferences: () => ({ voice: { stt: server.key } }), env });
    assert.equal(registry.selected('stt').available, true);
    env.isSecureContext = false; env.location.hostname = 'lan.example';
    assert.equal(registry.selected('stt').code, 'insecure_origin');
    assert.equal(registry.selected('stt').key, server.key);
    env.isSecureContext = true; env.MediaRecorder = undefined;
    assert.equal(registry.selected('stt').code, 'unsupported_format');
});

test('selection writes only the named preference and failed saves do not claim success', async () => {
    const writes = [];
    const registry = createSpeechProviders({ saveUiPreferences: async patch => { writes.push(patch); throw new Error('offline'); } });
    await assert.rejects(registry.choose('stt', client.key), /offline/);
    assert.deepEqual(writes, [{ voice: { stt: client.key } }]);
    assert.equal(registry.selected('stt').key, '');
});

test('destination follows this client origin and actual voice localService, never the server label', () => {
    assert.equal(destinationLabel('device'), 'On this device');
    for (const hostname of ['localhost', '127.0.0.1', '[::1]'])
        assert.equal(destinationLabel('host', { hostname }), 'On this computer');
    assert.equal(destinationLabel('host', { hostname: 'ouro.example', host: 'ouro.example:443' }), 'On your Ouroboros server (ouro.example:443)');
    assert.equal(destinationLabel('vendor:Google'), 'Goes to Google');
    assert.equal(voiceDestination({ local: true }), 'On this device');
    assert.equal(voiceDestination({ local: false }), 'Online voice');
    assert.equal(voiceDestination({}), 'Online voice');
});

test('raw speech upload preserves MIME/signal, no_speech leaves text absent and structured errors stay legible', async () => {
    const audio = new Blob(['audio'], { type: 'audio/mp4' }), controller = new AbortController(), calls = [];
    let body = { text: ' words ' }, ok = true;
    const fetchImpl = async (...args) => { calls.push(args); return { ok, status: ok ? 200 : 503, json: async () => body }; };
    assert.deepEqual(await transcribeAudio(server, audio, { signal: controller.signal, fetchImpl }), { text: 'words' });
    const [url, init] = calls[0];
    assert.equal(url, server.route_url); assert.equal(init.body, audio); assert.equal(init.signal, controller.signal);
    assert.equal(init.headers['Content-Type'], 'audio/mp4'); assert.equal(init.method, 'POST');
    body = { text: '', status: 'no_speech' };
    assert.deepEqual(await transcribeAudio(server, audio, { fetchImpl }), { no_speech: true });
    body = { error: { code: 'unavailable', message: 'Model is unavailable' } }; ok = false;
    await assert.rejects(transcribeAudio(server, audio, { fetchImpl }), /Model is unavailable/);
    await assert.rejects(transcribeAudio(server, new Blob(['x'.repeat(31)]), { fetchImpl }), error => error.code === 'too_large');
    assert.equal(calls.length, 3, 'oversized audio never reaches the route');
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { apiClient } from '../modules/api_client.js';
import { SETTINGS_INPUT_FIELDS, SECRET_KEYS, renderSettingsPage } from '../modules/settings_ui.js';
import { CallSession, callAvailability, mediaFailure, mountRoomCall } from '../modules/call.js';
import { decodeFrame, encodeFrame, followLevel, levelFromRms, LevelMeter, newerEpoch, pcm16, PlaybackBuffer,
    playedMilliseconds, LEVEL_CEILING_DB, LEVEL_FLOOR_DB, MAX_FRAME_PAYLOAD, MAX_BUFFER_SECONDS } from '../modules/call_worklet.js';

test('PCM uses the frozen 16-byte little-endian header, signed mono samples and wrapping counters', () => {
    const wire = encodeFrame({ kind: 2, epoch: 0x1234, seq: 0xabcdef12, rate: 48000,
        stream: 0x5678, pcm: new Int16Array([-32768, -1, 0, 32767]) });
    assert.deepEqual([...new Uint8Array(wire)], [79, 67, 1, 2, 52, 18, 18, 239, 205, 171,
        128, 187, 0, 0, 120, 86, 0, 128, 255, 255, 0, 0, 255, 127]);
    assert.deepEqual(decodeFrame(wire), { kind: 2, epoch: 0x1234, seq: 0xabcdef12, rate: 48000,
        stream: 0x5678, pcm: new Int16Array([-32768, -1, 0, 32767]) });
    const wrapped = decodeFrame(encodeFrame({ epoch: 65536, seq: 2 ** 32, rate: 44100, pcm: new Int16Array() }));
    assert.equal(wrapped.epoch, 0); assert.equal(wrapped.seq, 0); assert.equal(wrapped.rate, 44100);
});

test('Live calls in Settings: one engine (Gemini Live), its model and voice, and the Google AI key right there', () => {
    const html = renderSettingsPage();
    const fields = SETTINGS_INPUT_FIELDS.filter(([, key]) => key.includes('CALL_'));
    assert.deepEqual(fields.map(([, key]) => key), ['OUROBOROS_CALL_MODEL_GOOGLE', 'OUROBOROS_CALL_VOICE_GOOGLE',
        'OUROBOROS_CALL_NATIVE_SEARCH', 'OUROBOROS_CALL_MAX_MINUTES', 'OUROBOROS_PRESENCE_CALL_MAX_MINUTES']);
    for (const [id, key, fallback] of fields) {
        assert.equal(html.split(`id="${id}"`).length - 1, 1, key);
        if (key.endsWith('MAX_MINUTES')) {
            assert.equal(fallback, undefined);
            assert.match(html, new RegExp(`id="${id}"[^>]*min="1"[^>]*step="1"`));
        }
        if (key.includes('_MODEL_') || key.includes('_VOICE_'))
            assert.match(html, new RegExp(`<input id="${id}"[^>]*type="text"`));
    }
    // No engine to choose and nothing about another engine: calls are Gemini Live.
    for (const gone of ['s-call-engine', 's-call-model-openai', 's-call-voice-openai']) assert.equal(html.includes(`id="${gone}"`), false, gone);
    // The Google AI key lives where every provider key lives (Accounts, and the Secrets list); Live calls
    // names it there instead of a third field for the same secret.
    assert.ok(SECRET_KEYS.some(([key]) => key === 'GOOGLE_API_KEY'));
    assert.equal(html.split('data-secret-setting="GOOGLE_API_KEY"').length - 1, 2);
    const live = html.slice(html.indexOf('<h3>Live calls</h3>'), html.indexOf('s-call-duration-help'));
    assert.doesNotMatch(live, /data-secret-setting/);
    assert.match(live, /Gemini Live with the Google AI key in Accounts → Google/);
    assert.doesNotMatch(live, /OpenAI/);
});

test('framing rejects bad magic, version, kind, rate, lengths and oversized audio', () => {
    for (const [index, value] of [[0, 0], [1, 0], [2, 2], [3, 0]]) {
        const wire = encodeFrame({ rate: 1, pcm: new Int16Array([1]) });
        new DataView(wire).setUint8(index, value);
        assert.throws(() => decodeFrame(wire));
    }
    assert.throws(() => decodeFrame(new ArrayBuffer(15)));
    assert.throws(() => decodeFrame(new ArrayBuffer(17)));
    assert.throws(() => encodeFrame({ rate: 0, pcm: new Int16Array() }));
    assert.throws(() => encodeFrame({ kind: 7, rate: 1, pcm: new Int16Array() }));
    assert.throws(() => encodeFrame({ rate: 1, pcm: new Int16Array(MAX_FRAME_PAYLOAD / 2 + 1) }));
    assert.equal(decodeFrame(encodeFrame({ rate: 1, pcm: new Int16Array(MAX_FRAME_PAYLOAD / 2) })).pcm.byteLength,
        MAX_FRAME_PAYLOAD);
    const zero = encodeFrame({ rate: 1, pcm: new Int16Array() });
    new DataView(zero).setUint32(10, 0, true);
    assert.throws(() => decodeFrame(zero), /sample rate/);
});

test('float capture saturates correctly and played time never rounds up unheard samples', () => {
    assert.deepEqual([-2, -1, -0.5, 0, 0.5, 1, 2, NaN].map(pcm16), [-32768, -32768, -16384, 0, 16384, 32767, 32767, 0]);
    assert.equal(playedMilliseconds(47, 48000), 0);
    assert.equal(playedMilliseconds(480, 48000), 10);
    assert.equal(playedMilliseconds(441, 44100), 10);
    assert.equal(newerEpoch(0, 65535), true);
    assert.equal(newerEpoch(65535, 0), false);
    assert.equal(newerEpoch(7, 7), false);
});

test('jitter buffer counts rendered samples per item and flush preserves only heard progress', () => {
    const reports = [];
    const buffer = new PlaybackBuffer(1000, data => reports.push(data));
    const frame = (item_id, count, epoch = 0) => ({ item_id, epoch, rate: 1000, pcm: new Int16Array(count).fill(16384) });
    buffer.push(frame('one', 20));
    const output = new Float32Array(10);
    buffer.render(output);
    assert.deepEqual([...output], Array(10).fill(0));
    assert.deepEqual(reports, []);
    buffer.push(frame('one', 30));
    buffer.render(output);
    assert.deepEqual([...output], Array(10).fill(0.5));
    buffer.flush(1);
    assert.deepEqual(reports, [{ type: 'played', item_id: 'one', played_ms: 10 }]);
    buffer.render(output);
    assert.deepEqual([...output], Array(10).fill(0));
    assert.equal(buffer.push(frame('old', 100, 0)), false);
    buffer.push(frame('two', 40, 1));
    buffer.push(frame('three', 40, 1));
    buffer.render(new Float32Array(80));
    assert.deepEqual(reports.slice(1), [{ type: 'played', item_id: 'two', played_ms: 40 },
        { type: 'played', item_id: 'three', played_ms: 40 }]);
    buffer.flush(0);
    assert.equal(buffer.epoch, 1);
});

test('short final audio drains after bounded jitter wait, and physical overflow/rate mismatch are explicit', () => {
    const reports = [];
    const buffer = new PlaybackBuffer(1000, report => reports.push(report));
    const frame = { item_id: 'short', epoch: 0, rate: 1000, pcm: new Int16Array(5).fill(32767) };
    buffer.push(frame);
    for (let i = 0; i < 4; i++) buffer.render(new Float32Array(10));
    assert.deepEqual(reports, [{ type: 'played', item_id: 'short', played_ms: 5 }]);
    assert.throws(() => buffer.push({ ...frame, rate: 16000 }), /rate/);
    buffer.push({ ...frame, pcm: new Int16Array(1000 * MAX_BUFFER_SECONDS) });
    assert.throws(() => buffer.push(frame), /buffer is full/);
});

test('a long answer arriving faster than it plays is held, not a failed call', () => {
    const buffer = new PlaybackBuffer(1000, () => {});
    const second = { item_id: 'long', epoch: 0, rate: 1000, pcm: new Int16Array(1000) };
    for (let i = 0; i < 30; i++) assert.equal(buffer.push(second), true);  // thirty seconds in one burst
    assert.equal(buffer.queued, 30000);
});

test('levels map RMS to a perceptual 0..1 with silence exactly still, and the envelope ignores the frame rate', () => {
    assert.equal(levelFromRms(0), 0);
    assert.equal(levelFromRms(NaN), 0);
    assert.equal(levelFromRms(10 ** (LEVEL_FLOOR_DB / 20)), 0);           // the floor is silence
    assert.equal(levelFromRms(10 ** ((LEVEL_FLOOR_DB - 6) / 20)), 0);     // room noise below it too
    assert.equal(levelFromRms(10 ** (LEVEL_CEILING_DB / 20)), 1);
    assert.equal(levelFromRms(1), 1);
    const speech = levelFromRms(10 ** (-30 / 20));
    assert.ok(speech > 0.6 && speech < 0.8, speech);  // quiet speech still moves the light (the curve)
    assert.ok(levelFromRms(0.01) < levelFromRms(0.05));
    // The same 200 ms at 60, 120 and 30 frames a second lands on the same envelope.
    const run = (hz) => { let value = 0; for (let i = 0; i < hz / 5; i++) value = followLevel(value, 1, 1 / hz); return value; };
    for (const hz of [120, 30]) assert.ok(Math.abs(run(hz) - run(60)) < 1e-9, hz);
    // A fast attack and a slow release.
    assert.ok(followLevel(0, 1, 0.05) > 0.6);
    assert.ok(followLevel(1, 0, 0.05) > 0.8);
    assert.equal(followLevel(0.5, 0.2, -1), 0.5);
});

test('the meter reports one window of captured and rendered RMS and stays quiet through silence', () => {
    const reports = [];
    const meter = new LevelMeter(100, report => reports.push(report));
    const quantum = (value, n = 50) => new Float32Array(n).fill(value);
    meter.measure(quantum(0.5), quantum(0), false, 1);
    assert.equal(reports.length, 0);  // half a window
    meter.measure(quantum(0.5), quantum(0.25), true, 2);
    assert.equal(reports.length, 1);
    assert.equal(reports[0].type, 'level');
    assert.ok(Math.abs(reports[0].mic - 0.5) < 1e-6);
    assert.ok(Math.abs(reports[0].out - Math.sqrt(0.25 * 0.25 / 2)) < 1e-6);
    assert.deepEqual([reports[0].playing, reports[0].t], [true, 2]);
    for (let i = 0; i < 6; i++) meter.measure(quantum(0), quantum(0), false, 3);
    assert.equal(reports.length, 2);  // the first silent window says the sound stopped, the rest say nothing
    assert.deepEqual([reports[1].mic, reports[1].out, reports[1].playing], [0, 0, false]);
    meter.measure(null, quantum(0.1, 100), true, 4);  // no microphone input: playback alone still reports
    assert.equal(reports.length, 3);
    assert.equal(reports[2].mic, 0);
});

function fixture({ open } = {}) {
    const tracks = [{ enabled: true, stopped: false, stop() { this.stopped = true; } }];
    const media = { getTracks: () => tracks, getAudioTracks: () => tracks };
    const contexts = [], nodes = [], sockets = [], requests = [], views = [], captions = [], levels = [], settled = [];
    const timers = new Map();
    let timerId = 0;
    class AudioContext {
        constructor() { this.sampleRate = 48000; this.destination = {}; this.outputLatency = 0.12; contexts.push(this); }
        get audioWorklet() { return { addModule: async () => {} }; }
        resume() { return Promise.resolve(); }
        close() { this.closed = true; return Promise.resolve(); }
        createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    }
    class AudioWorkletNode {
        constructor() {
            this.messages = [];
            this.port = { postMessage: data => this.messages.push(data), close() {} };
            nodes.push(this);
        }
        connect() {}
        disconnect() { this.disconnected = true; }
    }
    const env = { isSecureContext: true, navigator: { mediaDevices: { getUserMedia: async () => media } },
        AudioContext, AudioWorkletNode, WebSocket: class {}, crypto: { randomUUID: () => 'call-request' },
        setTimeout: (fn, ms) => { timers.set(++timerId, { fn, ms }); return timerId; },
        clearTimeout: id => timers.delete(id) };
    const socket = path => {
        const sock = { path, readyState: 0, bufferedAmount: 0, sent: [],
            send(data) { this.sent.push(typeof data === 'string' ? JSON.parse(data) : decodeFrame(data)); },
            close() { this.readyState = 3; this.closed = true; this.onclose?.({ code: 1000, reason: '' }); },
            connect() { this.readyState = 1; this.onopen(); } };
        sockets.push(sock); return sock;
    };
    const session = new CallSession({ chatId: 71, env, socket, open: async body => {
        requests.push(body); return open ? open(body) : { call_id: 'c', ws_path: '/ws/call/c' };
    }, changed: view => views.push(view), caption: msg => captions.push(msg), level: data => levels.push(data),
    settled: view => settled.push(view) });
    // Server messages arrive the way the page meets them: through the call socket.
    const state = msg => sockets.at(-1).onmessage({ data: JSON.stringify({ type: 'state', ...msg }) });
    return { session, env, tracks, contexts, nodes, sockets, requests, views, captions, levels, settled, timers, state };
}

test('feature detection names why a surface cannot call, and Telegram hides Talk', () => {
    const { env } = fixture();
    assert.deepEqual(callAvailability(env), { supported: true, code: '' });
    for (const [override, code] of [[{ isSecureContext: false }, 'insecure'], [{ navigator: {} }, 'no_microphone_api'],
        [{ AudioWorkletNode: null }, 'no_live_audio'], [{ WebSocket: null }, 'no_call_socket'],
        // The desktop app's WebView without the microphone: the old app, not "this browser".
        [{ navigator: {}, pywebview: { api: {} } }, 'old_app']]) {
        const result = callAvailability({ ...env, ...override });
        assert.equal(result.supported, false); assert.equal(result.code, code); assert.equal(result.hidden, undefined);
    }
    for (const extra of [{ Telegram: { WebApp: {} } }, { document: { documentElement: { dataset: { ouroborosHost: 'telegram' } } } },
        { parent: { Telegram: { WebApp: {} } } }]) assert.equal(callAvailability({ ...env, ...extra }).hidden, true);
    assert.equal(mediaFailure({ name: 'NotAllowedError' }), 'permission_denied');
    assert.equal(mediaFailure({ name: 'NotFoundError' }), 'no_microphone');
    assert.equal(mediaFailure({ name: 'NotReadableError' }), 'microphone_unreadable');
    assert.equal(mediaFailure(new Error('anything else')), 'call_failed');
});

test('a call shows its own phases: permission, admission, then every server phase it reports', async () => {
    const f = fixture();
    try {
        await f.session.start();
        assert.deepEqual(f.views.map(view => view.state), ['permission', 'admission']);
        assert.deepEqual(f.requests, [{ chat_id: 71, client_message_id: 'call-request' }]);  // no per-call fields
        const socket = f.sockets[0]; socket.connect();
        assert.deepEqual(socket.sent[0], { type: 'hello', protocol: 1, role: 'owner', mic: true, speaker: true, rate: 48000, label: 'Owner' });
        f.state({ state: 'preparing', reason: '', phase: 'memory', engine: 'google', detail: '', action: '' });
        assert.deepEqual([f.session.state, f.views.at(-1).phase], ['preparing', 'memory']);
        assert.equal(f.tracks[0].enabled, false);  // nothing leaves the microphone before the call is live
        f.state({ state: 'connecting', reason: '', phase: '', engine: 'google' });
        assert.deepEqual([f.session.state, f.views.at(-1).phase], ['connecting', '']);
        f.state({ state: 'live' });
        assert.equal(f.session.state, 'live');
        assert.ok(f.views.at(-1).liveSince > 0);
        assert.equal(f.tracks[0].enabled, true);
        // An old server sends only state and reason.
        f.state({ state: 'reconnecting', reason: 'session_cap' });
        assert.deepEqual([f.views.at(-1).state, f.views.at(-1).code], ['reconnecting', 'session_cap']);
        f.state({ state: 'live' });
        assert.equal(f.views.at(-1).code, '');
        f.session.receive(JSON.stringify({ type: 'activity', activity: 'working', label: 'Searching the web' }));
        assert.deepEqual(f.views.at(-1).activity, { working: true, label: 'Searching the web' });
        f.session.receive(JSON.stringify({ type: 'activity', activity: 'idle', label: '' }));
        assert.deepEqual(f.views.at(-1).activity, { working: false, label: '' });
        // A newer server's phase or message changes nothing here; the call goes on.
        f.state({ state: 'warming_up' });
        f.session.receive(JSON.stringify({ type: 'something_new', value: 1 }));
        assert.equal(f.session.state, 'live');
        f.state({ state: 'closed', reason: 'provider_closed', detail: 'Session ended by the provider', engine: 'google', action: 'retry' });
        assert.deepEqual([f.session.state, f.session.code, f.session.detail, f.session.action],
            ['closed', 'provider_closed', 'Session ended by the provider', 'retry']);
        assert.equal(f.settled.length, 1);
        assert.equal(f.tracks[0].stopped, true); assert.equal(f.contexts[0].closed, true);
        assert.equal(socket.closed, true); assert.equal(f.nodes[0].disconnected, true);
        assert.equal(f.session.active, false);
    } finally { f.session.stop(); }
});

test('only a socket that never answers is bounded; a phase that reports itself has no clock', async () => {
    const silent = fixture();
    try {
        await silent.session.start();
        silent.sockets[0].connect();
        const [[, bound]] = [...silent.timers];
        assert.equal(bound.ms, 15000);
        bound.fn();  // no message at all
        assert.deepEqual([silent.session.state, silent.session.code], ['failed', 'socket_silent']);
        assert.equal(silent.settled.length, 1);
    } finally { silent.session.stop(); }
    const slow = fixture();
    try {
        await slow.session.start();
        slow.sockets[0].connect();
        assert.equal(slow.timers.size, 1);
        slow.state({ state: 'preparing', reason: '', phase: 'memory' });
        assert.equal(slow.timers.size, 0);  // any message ends the transport bound; preparing may take its time
        assert.equal(slow.session.state, 'preparing');
    } finally { slow.session.stop(); }
});

test('an unexplained close ends visibly with its own words, not "finished" and not a retry loop', async () => {
    const f = fixture();
    try {
        await f.session.start();
        f.sockets[0].connect();
        f.state({ state: 'live' });
        f.sockets[0].onclose({ code: 1011, reason: '' });
        assert.deepEqual([f.session.state, f.session.code, f.session.detail], ['failed', 'socket_closed', '']);
        assert.equal(f.requests.length, 1);  // nothing reopened on its own
    } finally { f.session.stop(); }
    const g = fixture();
    try {
        await g.session.start();
        g.sockets[0].connect();
        g.sockets[0].onclose({ code: 1008, reason: 'Policy violation' });
        assert.deepEqual([g.session.code, g.session.detail], ['socket_closed', 'Policy violation']);
    } finally { g.session.stop(); }
});

test('admission refusals keep the server code and words; a call asks for no engine (there is one)', async () => {
    const refused = fixture({ open: () => { const error = new Error('No live model is set'); error.status = 400;
        error.body = { ok: false, code: 'call_model_missing', error: 'No live model is set for this engine in Settings (Models, Live calls).' };
        throw error; } });
    try {
        await refused.session.start();
        assert.deepEqual([refused.session.state, refused.session.code], ['failed', 'call_model_missing']);
        assert.match(refused.session.words, /No live model is set for this engine/);
        assert.equal(refused.tracks[0].stopped, true);
    } finally { refused.session.stop(); }
    const plain = fixture();
    try {
        await plain.session.start({ engine: 'a-removed-engine' });  // a stale caller cannot choose an engine
        assert.deepEqual(Object.keys(plain.requests[0]).sort(), ['chat_id', 'client_message_id']);
    } finally { plain.session.stop(); }
});

test('mic frames, mute, levels with output latency, and the owner End', async () => {
    const f = fixture();
    try {
        await f.session.start();
        const socket = f.sockets[0]; socket.connect();
        f.state({ state: 'live' });
        const audio = () => ({ type: 'capture', generation: f.session.captureGeneration, rate: 48000, pcm: new Int16Array([100]) });
        f.session.onAudio(audio());
        assert.equal(socket.sent.at(-1).rate, 48000); assert.equal(socket.sent.at(-1).kind, 1);
        const old = audio();
        f.session.setMuted(true);
        assert.equal(f.tracks[0].enabled, false);
        assert.deepEqual(socket.sent.at(-1), { type: 'mute', muted: true });
        const sent = socket.sent.length; f.session.onAudio(audio());
        assert.equal(socket.sent.length, sent);
        // Muted, the light still hears Ouroboros: levels keep coming, delayed by the output latency.
        f.session.onAudio({ type: 'level', mic: 0, out: 10 ** (-30 / 20), playing: true, t: 1 });
        assert.equal(f.levels.length, 1);
        assert.equal(f.levels[0].mic, 0);
        assert.ok(f.levels[0].out > 0.6);
        assert.deepEqual([f.levels[0].playing, f.levels[0].latency], [true, 0.12]);
        f.session.setMuted(false); f.session.onAudio(old);
        assert.equal(socket.sent.at(-1).type, 'mute');  // a capture from before the toggle is never sent
        f.session.end();
        assert.deepEqual([f.session.state, f.session.code], ['ended', 'owner_end']);
        assert.deepEqual(socket.sent.at(-1), { type: 'bye', reason: 'left' });
        assert.equal(f.settled.length, 1);
        f.session.dismiss();
        assert.equal(f.session.state, 'idle');
    } finally { f.session.stop(); }
});

test('item binding, final captions, playback acknowledgements and stale epochs never claim queued audio', async () => {
    const f = fixture();
    try {
        await f.session.start(); f.sockets[0].connect();
        const caption = (item_id, extra = {}) => f.session.receive(JSON.stringify({ type: 'caption', speaker: 'ouroboros', item_id, text: '<b>literal</b>', final: false, ...extra }));
        const audio = epoch => encodeFrame({ kind: 2, epoch, rate: 48000, pcm: new Int16Array([100]) });
        assert.throws(() => f.session.receive(audio(0)), /without its speech item/);
        caption('one'); f.session.receive(audio(0));
        assert.equal(f.nodes[0].messages.at(-1).item_id, 'one');
        assert.equal(f.sockets[0].sent.filter(m => m.type === 'played').length, 0);
        caption('two'); caption('one', { final: true }); f.session.receive(audio(0));
        assert.equal(f.nodes[0].messages.at(-1).item_id, 'two');
        f.session.onAudio({ type: 'played', item_id: 'one', played_ms: 17 });
        f.session.onAudio({ type: 'played', item_id: 'one', played_ms: 10 });
        assert.deepEqual(f.sockets[0].sent.at(-1), { type: 'played', item_id: 'one', played_ms: 17 });
        f.session.receive(JSON.stringify({ type: 'flush', epoch: 1 }));
        const count = f.nodes[0].messages.length;
        f.session.receive(audio(0));
        assert.equal(f.nodes[0].messages.length, count);
        caption('three', { epoch: 1 });
        f.session.receive(JSON.stringify({ type: 'flush', epoch: 0 }));
        f.session.receive(audio(1));
        assert.equal(f.nodes[0].messages.at(-1).item_id, 'three');
        assert.equal(f.captions[0].text, '<b>literal</b>');
        assert.equal(f.captions[0].call_id, 'c');  // captions carry their call for reconciliation
    } finally { f.session.stop(); }
});

test('permission and transport failures release everything; the microphone has one owner, then the next', async () => {
    const f = fixture(), other = fixture();
    try {
        await f.session.start();
        await other.session.start();
        assert.deepEqual([other.session.state, other.session.code], ['failed', 'microphone_busy']);
        assert.equal(other.requests.length, 0);
        f.sockets[0].onerror(); assert.equal(f.session.state, 'failed');
        await other.session.start(); other.sockets[0].connect();
        other.state({ state: 'live' });
        other.sockets[0].bufferedAmount = 48000 * 2 + 1;
        other.session.onAudio({ type: 'capture', generation: other.session.captureGeneration, rate: 48000, pcm: new Int16Array([1]) });
        assert.equal(other.session.code, 'slow_uplink'); assert.equal(other.tracks[0].stopped, true);
    } finally { f.session.stop(); other.session.stop(); }
    const denied = fixture();
    denied.env.navigator.mediaDevices.getUserMedia = async () => { throw { name: 'NotAllowedError' }; };
    try {
        await denied.session.start();
        assert.deepEqual([denied.session.state, denied.session.code], ['failed', 'permission_denied']);
        assert.equal(denied.contexts[0].closed, true);
        assert.equal(denied.requests.length, 0);
    } finally { denied.session.stop(); }
});

test('ending while permission or admission is pending cleans up late success and opens nothing paid', async () => {
    let resolveMedia, resolveOpen;
    const f = fixture();
    f.env.navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { resolveMedia = resolve; });
    try {
        const pending = f.session.start();
        assert.equal(f.session.state, 'permission');
        f.session.end();
        assert.equal(f.session.state, 'ended');
        resolveMedia({ getTracks: () => f.tracks }); await pending;
        assert.equal(f.tracks[0].stopped, true); assert.equal(f.requests.length, 0);
    } finally { f.session.stop(); }
    const g = fixture({ open: () => new Promise(resolve => { resolveOpen = resolve; }) });
    try {
        const admission = g.session.start();
        while (!resolveOpen) await new Promise(resolve => setImmediate(resolve));
        assert.equal(g.session.state, 'admission');
        g.session.end(); resolveOpen({ call_id: 'late', ws_path: '/ws/call/late' }); await admission;
        g.sockets[0].connect();
        assert.equal(g.sockets[0].sent[0].mic, false);
        assert.deepEqual(g.sockets[0].sent[1], { type: 'bye', reason: 'left' });
        assert.equal(g.sockets[0].closed, true);
    } finally { g.session.stop(); }
});

test('gateway client uses the call contract and refuses cross-origin or arbitrary sockets', async () => {
    const old = globalThis.fetch;
    globalThis.fetch = async (url, init) => {
        assert.equal(url, '/api/call/open'); assert.equal(init.method, 'POST');
        assert.deepEqual(JSON.parse(init.body), { chat_id: 9, client_message_id: 'one' });
        return { ok: true, json: async () => ({ call_id: 'one', ws_path: '/ws/call/one' }) };
    };
    try { assert.equal((await apiClient.openCall({ chat_id: 9, client_message_id: 'one' })).call_id, 'one'); }
    finally { globalThis.fetch = old; }
    const options = { location: { href: 'https://app.test/', origin: 'https://app.test' }, WebSocket: class { constructor(url) { this.url = url; } } };
    assert.equal(apiClient.callSocket('/ws/call/id', options).url, 'wss://app.test/ws/call/id');
    for (const path of ['https://other.test/ws/call/id', '/ws', '/ws/call/id?token=secret', '//other.test/ws/call/id'])
        assert.throws(() => apiClient.callSocket(path, options), /invalid call socket/);
});

test('the registered AudioWorklet captures at device rate, never echoes, and meters while muted', async () => {
    const saved = Object.fromEntries(['AudioWorkletProcessor', 'registerProcessor', 'sampleRate', 'currentTime']
        .map(key => [key, globalThis[key]]));
    let Processor;
    globalThis.sampleRate = 48000;
    globalThis.currentTime = 3.5;
    globalThis.AudioWorkletProcessor = class {
        // A real port detaches what it transfers; so does this one.
        constructor() { this.sent = []; this.port = { postMessage: (msg, transfer) => this.sent.push(transfer ? structuredClone(msg, { transfer }) : msg) }; }
    };
    globalThis.registerProcessor = (name, cls) => { assert.equal(name, 'ouroboros-call-audio'); Processor = cls; };
    try {
        await import('../modules/call_worklet.js?processor-test');
        const processor = new Processor();
        const kinds = () => processor.sent.map(msg => msg.type);
        processor.port.onmessage({ data: { type: 'capture', enabled: true, generation: 7 } });
        const output = new Float32Array(128);
        for (let i = 0; i < 16; i++) processor.process([[new Float32Array(128).fill(0.5)]], [[output]]);
        assert.deepEqual([...output], Array(128).fill(0));
        // Every window is posted, not only the first: a transferred buffer is replaced by a whole new one.
        const posted = processor.sent.filter(msg => msg.type === 'capture');
        assert.equal(posted.length, 2);
        assert.deepEqual(posted.map(msg => msg.pcm.length), [960, 960]);
        const captured = processor.sent.find(msg => msg.type === 'capture');
        assert.equal(captured.pcm.length, 960); assert.equal(captured.rate, 48000);
        assert.equal(captured.generation, 7); assert.equal(captured.pcm[0], 16384);
        const level = processor.sent.find(msg => msg.type === 'level');
        assert.ok(Math.abs(level.mic - 0.5) < 1e-6);
        assert.deepEqual([level.out, level.playing, level.t], [0, false, 3.5]);
        // Muted (capture off): nothing is captured, the meter still reports what plays.
        processor.port.onmessage({ data: { type: 'capture', enabled: false, generation: 8 } });
        const captures = kinds().filter(kind => kind === 'capture').length;
        processor.port.onmessage({ data: { type: 'audio', epoch: 1, rate: 48000,
            item_id: 'spoken', pcm: new Int16Array(4800).fill(16384) } });
        for (let i = 0; i < 16; i++) processor.process([[new Float32Array(128)]], [[output]]);
        assert.equal(kinds().filter(kind => kind === 'capture').length, captures);
        const playing = processor.sent.filter(msg => msg.type === 'level').at(-1);
        assert.equal(playing.mic, 0);
        assert.ok(playing.out > 0.4 && playing.playing, JSON.stringify(playing));
        assert.equal(output[0], 0.5);
        processor.port.onmessage({ data: { type: 'flush', epoch: 2 } });
        assert.equal(processor.sent.at(-1).type, 'played');
        processor.process([[]], [[output]]); assert.equal(output[0], 0);
    } finally {
        for (const [key, value] of Object.entries(saved)) {
            if (value === undefined) delete globalThis[key]; else globalThis[key] = value;
        }
    }
});

test('a room call holds its room for the whole call and lets go when it ends; destroying the room hangs up', async () => {
    const calls = [];
    let retains = false;
    let settle = null;
    const room = { hasPendingWork: () => false, closeTransient: () => calls.push('room-leave'), destroy: () => calls.push('room-destroy') };
    mountRoomCall({ room, release: () => calls.push('release'), mount: ({ onSettled }) => {
        settle = onSettled;
        return { retains: () => retains, leave: () => calls.push('call-leave'), destroy: () => calls.push('call-destroy') };
    } });
    assert.equal(room.hasPendingWork(), false);  // no call: the room's own work decides
    retains = true;
    assert.equal(room.hasPendingWork(), true);   // permission, admission, preparation and live all hold it
    room.closeTransient();
    assert.deepEqual(calls, ['room-leave']);     // leaving the screen keeps the call
    retains = false;
    settle({ state: 'closed' });
    await new Promise(resolve => queueMicrotask(resolve));
    assert.deepEqual(calls, ['room-leave', 'release']);
    assert.equal(room.hasPendingWork(), false);
    room.destroy();
    assert.deepEqual(calls, ['room-leave', 'release', 'call-destroy', 'room-destroy']);
    // The room's own pending work still holds it without a call.
    const busy = { hasPendingWork: () => true, destroy() {} };
    mountRoomCall({ room: busy, mount: () => ({ retains: () => false, destroy() {} }) });
    assert.equal(busy.hasPendingWork(), true);
});

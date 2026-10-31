import test from 'node:test';
import assert from 'node:assert/strict';
import { prepareRecorder, recordingFormat, wavBlob } from '../modules/voice_recorder.js';
import { Recorder, environment, stream } from './helpers/voice_env.js';

const provider = { formats: ['audio/webm', 'audio/mp4', 'audio/wav'], max_bytes: 100, max_seconds: 5 };

test('codec negotiation prefers Opus, accepts Safari AAC, respects provider-only MIME and falls back to WAV', () => {
    assert.equal(recordingFormat(provider.formats, { MediaRecorder: Recorder }), 'audio/webm;codecs=opus');
    const safari = { MediaRecorder: { isTypeSupported: type => type === 'audio/mp4' } };
    assert.equal(recordingFormat(provider.formats, safari), 'audio/mp4');
    assert.equal(recordingFormat(['audio/webm'], safari), '');
    assert.equal(recordingFormat(['audio/wav'], { AudioContext: class {} }), 'audio/wav');
    assert.equal(recordingFormat(['audio/mp4'], { AudioContext: class {} }), '');
});

test('MediaRecorder uses 32 kbps and raw MIME, owns timers and resolves final audio', async () => {
    const env = environment(); env.MediaRecorder = Recorder;
    const recording = prepareRecorder(provider, { env }).start(stream());
    const native = Recorder.instances.at(-1);
    assert.equal(native.options.audioBitsPerSecond, 32000);
    assert.equal(native.slice, 250);
    native.data(20);
    const audio = await recording.stop();
    assert.equal(audio.size, 20); assert.equal(audio.type, 'audio/webm;codecs=opus');
    assert.equal(env.timers.size, 0);
    assert.equal(native.ondataavailable, null);
});

test('byte and time limits stop recording; an oversized final blob is refused, never sliced', async () => {
    const env = environment(), notices = []; env.MediaRecorder = Recorder;
    const recording = prepareRecorder(provider, { env, onLimit: code => notices.push(code) }).start(stream());
    const native = Recorder.instances.at(-1);
    native.data(101);
    await assert.rejects(recording.result, error => error.code === 'too_large');
    assert.equal(native.state, 'inactive'); assert.deepEqual(notices, ['max_bytes']);
    const next = prepareRecorder(provider, { env, onLimit: code => notices.push(code) }).start(stream());
    Recorder.instances.at(-1).data(10);
    env.fireTimer(5000);
    assert.equal((await next.result).size, 10);
    assert.deepEqual(notices, ['max_bytes', 'max_seconds']);
});

test('cancel suppresses late data and errors, settles the promise and stops the native recorder', async () => {
    const env = environment(); env.MediaRecorder = Recorder;
    const recording = prepareRecorder(provider, { env }).start(stream());
    const native = Recorder.instances.at(-1), lateData = native.ondataavailable;
    recording.cancel(); lateData({ data: new Blob(['late']) });
    await assert.rejects(recording.result, { name: 'AbortError' });
    assert.equal(native.state, 'inactive'); assert.equal(env.timers.size, 0);
    const next = prepareRecorder(provider, { env }).start(stream());
    Recorder.instances.at(-1).data(12);
    assert.equal((await next.stop()).size, 12);
});

test('WAV encodes actual mono PCM, bounded sample count and clipping', async () => {
    const audio = wavBlob([new Float32Array([-2, 0, 2])], 3, 16000);
    const view = new DataView(await audio.arrayBuffer());
    assert.equal(audio.type, 'audio/wav'); assert.equal(audio.size, 50);
    assert.equal(view.getUint32(24, true), 16000); assert.equal(view.getUint16(22, true), 1);
    assert.equal(view.getInt16(44, true), -32768); assert.equal(view.getInt16(48, true), 32767);
});

test('WAV resumes in the gesture, records finite samples and disconnects every audio node', async () => {
    const env = environment(); let context, resumes = 0, closes = 0;
    const node = () => ({ connect() {}, disconnect() { this.disconnected = true; } });
    env.AudioContext = class {
        constructor() { context = this; this.sampleRate = 16000; }
        resume() { resumes++; return Promise.resolve(); }
        close() { closes++; return Promise.resolve(); }
        createMediaStreamSource() { return this.source = node(); }
        createScriptProcessor() { return this.processor = node(); }
        createGain() { return this.gain = { ...node(), gain: { value: 1 } }; }
    };
    const limits = [], prepared = prepareRecorder(provider, { env, onLimit: code => limits.push(code) });
    assert.equal(resumes, 1);
    const recording = prepared.start(stream());
    context.processor.onaudioprocess({ inputBuffer: { getChannelData: () => new Float32Array(1000).fill(.25) } });
    const audio = await recording.result;
    assert.equal(audio.size, 100); assert.deepEqual(limits, ['max_bytes']);
    assert.equal(closes, 1); assert.equal(env.timers.size, 0);
    assert.ok(context.source.disconnected && context.processor.disconnected && context.gain.disconnected);
});

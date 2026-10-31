import test from 'node:test';
import assert from 'node:assert/strict';
import { createDictation, transcriptInsertion } from '../modules/dictation.js';
import { createCaptureLease } from '../modules/media_capture.js';
import { Recorder, deferred, environment, stream } from './helpers/voice_env.js';

const provider = { key: 'local:speech', runtime: 'server', formats: ['audio/webm'], max_seconds: 5 };
function setup() {
    const env = environment(); env.MediaRecorder = Recorder;
    const capture = createCaptureLease(env), request = deferred(), plans = [], notices = [], states = [], sent = [];
    let draft = { value: 'start end', revision: 1, start: 6, end: 6 };
    const controller = createDictation({ env, capture, readDraft: () => draft,
        onState: value => states.push(value), onTranscript: value => plans.push(value), notice: value => notices.push(value),
        transcribe: (...args) => { sent.push(args); return request.promise; } });
    return { env, capture, request, plans, notices, states, sent, controller, draft: value => { draft = value; } };
}

async function startAndStop(fixture) {
    await fixture.controller.start(provider);
    Recorder.instances.at(-1).data(15);
    const stopped = fixture.controller.stop();
    await Promise.resolve(); await Promise.resolve();
    return { stopped };
}

test('transcript inserts at Stop caret only if unchanged, otherwise Insert/Copy; restoring text still counts as editing', () => {
    const before = { value: 'ab', revision: 2, start: 1, end: 1 };
    assert.deepEqual(transcriptInsertion(before, { ...before, start: 2 }, ' c '), { kind: 'insert', text: 'c', start: 1, end: 1 });
    assert.equal(transcriptInsertion(before, { ...before, revision: 3 }, 'c').kind, 'offer');
    assert.deepEqual(transcriptInsertion(before, { ...before, value: 'new' }, 'c').actions, ['Insert', 'Copy']);
    assert.equal(transcriptInsertion(before, before, ' ').kind, 'empty');
});

test('dictation snapshots at Stop, releases audio before upload, and hands back one insertion without sending', async () => {
    const fixture = setup();
    await fixture.controller.start(provider);
    fixture.draft({ value: 'edited during recording', revision: 2, start: 3, end: 8 });
    Recorder.instances.at(-1).data(15);
    const stopped = fixture.controller.stop();
    await Promise.resolve(); await Promise.resolve();
    assert.equal(fixture.capture.currentHolder(), null);
    assert.equal(fixture.sent.length, 1);
    fixture.request.resolve({ text: 'hello' }); await stopped;
    assert.deepEqual(fixture.plans, [{ kind: 'insert', text: 'hello', start: 3, end: 8 }]);
    assert.equal(fixture.controller.state, 'idle'); assert.equal(fixture.env.timers.size, 0);
});

test('editing while transcription is pending offers an explicit choice and preserves the current draft', async () => {
    const fixture = setup(), { stopped } = await startAndStop(fixture);
    fixture.draft({ value: 'owner typed', revision: 2, start: 11, end: 11 });
    fixture.request.resolve({ text: 'hello' }); await stopped;
    assert.deepEqual(fixture.plans, [{ kind: 'offer', text: 'hello', actions: ['Insert', 'Copy'] }]);
});

test('Cancel, suspend, destroy and Panic suppress late transcripts and abort pending transport', async () => {
    for (const action of ['cancel', 'suspend', 'destroy']) {
        const fixture = setup(), { stopped } = await startAndStop(fixture);
        fixture.controller[action]('panic');
        assert.equal(fixture.sent[0][2].signal.aborted, true);
        fixture.request.resolve({ text: 'late' }); await stopped;
        assert.deepEqual(fixture.plans, []); assert.equal(fixture.controller.state, 'idle');
        assert.equal(fixture.env.timers.size, 0);
    }
    const fixture = setup();
    await fixture.controller.start(provider);
    fixture.capture.releaseAll('panic');
    assert.equal(fixture.controller.state, 'idle');
    assert.equal(Recorder.instances.at(-1).state, 'inactive');
});

test('suspend during permission prompt frees the lease now and stops the late microphone', async () => {
    const fixture = setup(), permission = deferred(), mic = stream();
    fixture.env.navigator.mediaDevices.getUserMedia = () => permission.promise;
    const pending = fixture.controller.start(provider);
    fixture.controller.suspend();
    assert.equal(fixture.capture.currentHolder(), null);
    permission.resolve(mic); await pending;
    assert.equal(mic.track.stops, 1); assert.equal(fixture.controller.state, 'idle');
});

test('no speech reports honestly; a later successful recording still inserts', async () => {
    const fixture = setup(), { stopped } = await startAndStop(fixture);
    fixture.request.resolve({ no_speech: true }); await stopped;
    assert.deepEqual(fixture.plans, []); assert.match(fixture.notices[0], /No speech/);
    const next = setup(), { stopped: nextStop } = await startAndStop(next);
    next.request.resolve({ text: 'yes' }); await nextStop;
    assert.equal(next.plans[0].text, 'yes');
});

test('client recognition starts in the gesture under a logical lease without getUserMedia', async () => {
    const fixture = setup(); let starts = 0, cancels = 0;
    fixture.env.navigator.mediaDevices.getUserMedia = () => { throw new Error('must not open a second microphone'); };
    const pending = fixture.controller.start({ runtime: 'client', impl: { start() {
        starts++; assert.equal(fixture.capture.currentHolder().label, 'Dictation');
        return { stop: async () => ({ text: 'recognized' }), cancel() { cancels++; } };
    } } });
    assert.equal(starts, 1);
    await pending; await fixture.controller.stop();
    assert.equal(fixture.plans[0].text, 'recognized'); assert.equal(cancels, 0);
    assert.equal(fixture.capture.currentHolder(), null);
});

test('another capture holder refuses dictation without releasing that owner', async () => {
    const fixture = setup(), lease = await fixture.capture.acquire({ owner: 'session', label: 'Voice session' });
    assert.equal((await fixture.controller.start(provider)).code, 'in_use');
    assert.equal(fixture.capture.currentHolder().owner, 'session');
    assert.match(fixture.notices[0], /Voice session/);
    lease.release();
});

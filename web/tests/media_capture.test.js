import test from 'node:test';
import assert from 'node:assert/strict';
import { createCaptureLease, captureAvailability, captureReason, isLoopback } from '../modules/media_capture.js';
import { deferred, environment, stream } from './helpers/voice_env.js';

const error = name => Object.assign(new Error(name), { name });

test('acquisition calls getUserMedia before the first microtask and excludes a second holder', async () => {
    const env = environment(), opened = deferred(), mic = stream(), calls = [];
    env.navigator.mediaDevices.getUserMedia = constraints => { calls.push(constraints); return opened.promise; };
    const capture = createCaptureLease(env);
    const pending = capture.acquire({ owner: 'room', label: 'Dictation' });
    assert.equal(calls.length, 1, 'no await before the native request');
    assert.deepEqual(calls[0], { audio: {}, video: false });
    assert.deepEqual(capture.acquire({ owner: 'other' }), { code: 'in_use', label: 'Dictation', owner: 'room' });
    assert.equal(capture.currentHolder().owner, 'room');
    opened.resolve(mic);
    const lease = await pending;
    assert.equal(lease.stream, mic);
    lease.release(); lease.release();
    assert.equal(mic.track.stops, 1);
    assert.equal(capture.currentHolder(), null);
    env.navigator.mediaDevices.getUserMedia = () => Promise.resolve(stream());
    const next = await capture.acquire({ owner: 'other' });
    next.release();
});

test('permission cancellation releases immediately and stops a late stream without touching the next owner', async () => {
    const env = environment(), opened = deferred(), late = stream();
    env.navigator.mediaDevices.getUserMedia = () => opened.promise;
    const capture = createCaptureLease(env);
    const pending = capture.acquire({ owner: 'first' });
    pending.release();
    const next = capture.acquire({ owner: 'client', capture: false });
    opened.resolve(late);
    assert.equal((await pending).code, 'lost');
    assert.equal(late.track.stops, 1);
    assert.equal(capture.currentHolder().owner, 'client');
    next.release();
});

test('typed physical refusal releases reservation and a subsequent healthy capture works', async () => {
    for (const [name, code] of [['NotAllowedError', 'denied'], ['NotFoundError', 'no_device'],
        ['NotReadableError', 'lost'], ['SecurityError', 'denied']]) {
        const env = environment(), capture = createCaptureLease(env);
        env.navigator.mediaDevices.getUserMedia = () => Promise.reject(error(name));
        assert.deepEqual(await capture.acquire({ owner: 'one' }), { code });
        assert.equal(capture.currentHolder(), null);
        assert.ok(captureReason(code));
        env.navigator.mediaDevices.getUserMedia = () => Promise.resolve(stream());
        (await capture.acquire({ owner: 'two' })).release();
    }
});

test('http LAN refuses explicitly; loopback works; no API and reserved kinds stay typed', () => {
    const env = environment();
    env.isSecureContext = false; env.location.hostname = '192.168.1.20';
    assert.equal(captureAvailability(env).code, 'insecure_origin');
    for (const hostname of ['localhost', '127.0.0.1', '127.1.2.3', '[::1]']) {
        env.location.hostname = hostname;
        assert.equal(captureAvailability(env).ok, true);
    }
    for (const hostname of ['localhost.example', '127.0.0.1.evil', '127.0.0.999']) assert.equal(isLoopback(hostname), false);
    env.navigator.mediaDevices = undefined;
    assert.equal(captureAvailability(env).code, 'no_api');
    assert.equal(captureAvailability(env, { capture: false }).ok, true);
    assert.equal(createCaptureLease(env).acquire({ kind: 'camera' }).code, 'no_api');
});

test('chosen missing device falls back once preserving processing constraints; denied never retries', async () => {
    for (const name of ['OverconstrainedError', 'NotFoundError', 'NotAllowedError']) {
        const env = environment(), calls = [], mic = stream(), capture = createCaptureLease(env);
        env.navigator.mediaDevices.getUserMedia = constraints => {
            calls.push(constraints);
            return calls.length === 1 ? Promise.reject(error(name)) : Promise.resolve(mic);
        };
        const lease = await capture.acquire({ constraints: { deviceId: { exact: 'gone' }, noiseSuppression: false } });
        if (name === 'NotAllowedError') { assert.equal(lease.code, 'denied'); assert.equal(calls.length, 1); }
        else {
            assert.equal(lease.fellBack, true);
            assert.deepEqual(calls[1], { audio: { noiseSuppression: false }, video: false });
            lease.release();
        }
    }
});

test('same holder replaces its stream, failed replacement keeps old audio, latest concurrent replacement wins', async () => {
    const env = environment(), original = stream(), newer = stream(), stale = stream(), calls = [];
    env.navigator.mediaDevices.getUserMedia = () => { const call = deferred(); calls.push(call); return call.promise; };
    const capture = createCaptureLease(env), pending = capture.acquire({ owner: 'call' });
    calls[0].resolve(original);
    const lease = await pending;
    const failed = lease.replace(); calls[1].reject(error('NotAllowedError'));
    assert.equal((await failed).code, 'denied'); assert.equal(lease.stream, original);
    const a = lease.replace(), b = lease.replace();
    assert.equal(calls.length, 4);
    calls[3].resolve(newer); assert.equal(await b, newer);
    calls[2].resolve(stale); assert.equal((await a).code, 'lost');
    assert.equal(original.track.stops, 1); assert.equal(stale.track.stops, 1);
    original.track.dispatchEvent(new Event('ended'));
    assert.equal(capture.currentHolder().owner, 'call');
    newer.track.dispatchEvent(new Event('ended'));
    assert.equal(capture.currentHolder(), null); assert.equal(lease.signal.reason, 'lost');
});

test('page hide, visibility hide and Panic release physical and logical holders', async () => {
    for (const event of ['pagehide', 'visibilitychange', 'panic']) {
        const env = environment(), capture = createCaptureLease(env);
        const lease = await capture.acquire({ owner: 'room' });
        const mic = lease.stream;
        if (event === 'panic') capture.releaseAll('panic');
        else if (event === 'visibilitychange') { env.document.hidden = true; env.document.dispatchEvent(new Event(event)); }
        else env.dispatchEvent(new Event(event));
        assert.equal(mic.track.stops, 1); assert.equal(capture.currentHolder(), null);
        const client = capture.acquire({ owner: 'recognizer', capture: false });
        assert.equal(client.stream, null); client.release();
    }
});

test('Panic releases the holder at once and stops its tracks; a later capture still works', async () => {
    const env = environment(), mic = stream();
    env.navigator.mediaDevices.getUserMedia = () => Promise.resolve(mic);
    const capture = createCaptureLease(env);
    const lease = await capture.acquire({ owner: 'dictation', label: 'Dictation' });
    env.dispatchEvent(new Event('ouro:panic'));
    assert.equal(capture.currentHolder(), null, 'Panic ends the capture');
    assert.equal(mic.track.stops, 1);
    assert.equal(lease.signal.reason, 'panic', 'the holder learns why');
    env.dispatchEvent(new Event('ouro:panic'));
    env.navigator.mediaDevices.getUserMedia = () => Promise.resolve(stream());
    const next = await capture.acquire({ owner: 'dictation' });
    assert.equal(capture.currentHolder().owner, 'dictation');
    next.release();
});

test('only an opted-in holder keeps capture while hidden, including pending permission and replacement', async () => {
    const env = environment(), opened = deferred(), capture = createCaptureLease(env);
    env.navigator.mediaDevices.getUserMedia = () => opened.promise;
    const pending = capture.acquire({ owner: 'call', keepWhenHidden: true });
    env.document.hidden = true;
    env.document.dispatchEvent(new Event('visibilitychange'));
    assert.equal(capture.currentHolder()?.owner, 'call');
    opened.resolve(stream());
    const lease = await pending;
    const replacement = stream();
    env.navigator.mediaDevices.getUserMedia = () => Promise.resolve(replacement);
    await lease.replace();
    env.document.dispatchEvent(new Event('visibilitychange'));
    assert.equal(lease.signal.aborted, false);
    assert.equal(replacement.track.stops, 0);
    lease.release();
    env.navigator.mediaDevices.getUserMedia = () => Promise.resolve(stream());
    const dictation = await capture.acquire({ owner: 'dictation' });
    env.document.dispatchEvent(new Event('visibilitychange'));
    assert.equal(dictation.signal.reason, 'hidden', 'opt-in does not leak to the next holder');
    assert.equal(capture.currentHolder(), null);
});

test('background opt-in still releases on pagehide, Panic, device loss and explicit End', async () => {
    for (const cause of ['pagehide', 'ouro:panic', 'lost', 'end']) {
        const env = environment(), capture = createCaptureLease(env);
        const lease = await capture.acquire({ owner: 'call', keepWhenHidden: true });
        const mic = lease.stream;
        if (cause === 'lost') mic.track.dispatchEvent(new Event('ended'));
        else if (cause === 'end') lease.release('ended');
        else env.dispatchEvent(new Event(cause));
        assert.equal(lease.signal.aborted, true, cause);
        assert.equal(mic.track.stops, 1, cause);
        assert.equal(capture.currentHolder(), null, cause);
    }
});

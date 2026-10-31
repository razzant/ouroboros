import './fixtures/call_modules.js';
import test from 'node:test';
import assert from 'node:assert/strict';
const { DEFAULT_DEVICE_PREFS, DEVICES_CHANGED, listDevices, loadDevicePrefs, microphoneChanged, microphoneConstraints,
    openCamera, openMicrophone, routeSpeaker, saveDevicePrefs, speakerChoice } = await import('../../skills/live_conversation/ui/call_devices.js');

function client(extra = {}) {
    const map = new Map();
    const events = [];
    return { localStorage: { getItem: key => (map.has(key) ? map.get(key) : null), setItem: (key, value) => map.set(key, String(value)) },
        CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init?.detail; } },
        dispatchEvent: event => events.push(event), events, ...extra };
}

const missing = name => Object.assign(new Error(name), { name });

test('a device choice stays on this client, merges, is announced, and a client without storage keeps it per page', () => {
    const env = client();
    assert.deepEqual(loadDevicePrefs(env), DEFAULT_DEVICE_PREFS);
    saveDevicePrefs({ mic: 'usb-mic' }, env);
    saveDevicePrefs({ noiseSuppression: false }, env);
    assert.deepEqual(loadDevicePrefs(env), { ...DEFAULT_DEVICE_PREFS, mic: 'usb-mic', noiseSuppression: false });
    assert.equal(env.events.at(-1).type, DEVICES_CHANGED);
    assert.equal(env.events.at(-1).detail.mic, 'usb-mic');
    // Values of the wrong kind never become a choice.
    env.localStorage.setItem('ouroboros.call.devices', JSON.stringify({ mic: 7, echoCancellation: 'no' }));
    assert.deepEqual(loadDevicePrefs(env), DEFAULT_DEVICE_PREFS);
    const blocked = client({ localStorage: { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } } });
    saveDevicePrefs({ camera: 'cam-2' }, blocked);
    assert.equal(loadDevicePrefs(blocked).camera, 'cam-2');
    assert.equal(loadDevicePrefs(client()).camera, '');  // another window's choice is not this one's
});

test('the microphone and camera open on the chosen device, or the system one when the chosen is gone', async () => {
    const requests = [];
    const stream = { getTracks: () => [] };
    const env = { navigator: { mediaDevices: { getUserMedia: async (constraints) => {
        requests.push(constraints);
        if (constraints.audio?.deviceId || constraints.video?.deviceId) throw missing('OverconstrainedError');
        return stream;
    } } } };
    const prefs = { ...DEFAULT_DEVICE_PREFS, mic: 'usb-mic', camera: 'front', autoGainControl: false };
    assert.deepEqual(microphoneConstraints(prefs).audio, { channelCount: 1, echoCancellation: true, noiseSuppression: true,
        autoGainControl: false, deviceId: { exact: 'usb-mic' } });
    const opened = await openMicrophone(env, prefs);
    assert.equal(opened.fellBack, true);
    assert.equal(requests.length, 2);
    assert.equal(requests[1].audio.deviceId, undefined);
    assert.equal(requests[1].audio.autoGainControl, false);  // the cleaning survives the fallback
    const camera = await openCamera(env, prefs);
    assert.deepEqual([camera.fellBack, requests.at(-1).video], [true, true]);
    // The system's own devices need no second request.
    requests.length = 0;
    const plain = await openMicrophone(env, DEFAULT_DEVICE_PREFS);
    assert.deepEqual([plain.fellBack, requests.length], [false, 1]);
    // A refusal is not a missing device: it is never retried on another one.
    const refused = { navigator: { mediaDevices: { getUserMedia: async () => { throw missing('NotAllowedError'); } } } };
    await assert.rejects(openMicrophone(refused, prefs), { name: 'NotAllowedError' });
});

test('the speaker follows the choice where the engine allows it and falls back to the system one', async () => {
    assert.equal(speakerChoice({ AudioContext: class {} }), false);
    class Routed {
        sinkId = '';
        async setSinkId(id) { (this.ids ||= []).push(id); if (id === 'gone') throw missing('NotFoundError'); this.sinkId = id; }
    }
    assert.equal(speakerChoice({ AudioContext: Routed }), true);
    assert.equal(await routeSpeaker({}, { speaker: 'headphones' }), 'system');  // this engine plays where the system says
    const context = new Routed();
    assert.equal(await routeSpeaker(context, { speaker: '' }), 'system');
    assert.equal(context.ids, undefined);
    assert.equal(await routeSpeaker(context, { speaker: 'headphones' }), 'routed');
    assert.equal(await routeSpeaker(context, { speaker: 'gone' }), 'fell_back');
    assert.deepEqual(context.ids, ['headphones', 'gone', '']);
    // Back to the system's own speaker during a call: a routed context returns to it; one never routed is left alone.
    assert.equal(await routeSpeaker(context, { speaker: 'headphones' }), 'routed');
    assert.equal(await routeSpeaker(context, { speaker: 'default' }), 'system');
    assert.deepEqual(context.ids.slice(-2), ['headphones', '']);
    assert.equal(context.sinkId, '');
    const untouched = new Routed();
    assert.equal(await routeSpeaker(untouched, { speaker: 'default' }), 'system');
    assert.equal(untouched.ids, undefined);
});

test('device lists leave out the browser aliases and say whether the devices are named yet', async () => {
    const env = { navigator: { mediaDevices: { enumerateDevices: async () => [
        { kind: 'audioinput', deviceId: 'default', label: 'Default' }, { kind: 'audioinput', deviceId: 'a1', label: '' },
        { kind: 'audiooutput', deviceId: 'communications', label: '' }, { kind: 'videoinput', deviceId: 'v1', label: '' }] } } };
    assert.deepEqual(await listDevices(env), { audioinput: [{ id: 'a1', label: '' }], audiooutput: [],
        videoinput: [{ id: 'v1', label: '' }], named: false });
    env.navigator.mediaDevices.enumerateDevices = async () => [{ kind: 'audioinput', deviceId: 'a1', label: 'USB mic' }];
    assert.equal((await listDevices(env)).named, true);
    assert.deepEqual(await listDevices({ navigator: {} }), { audioinput: [], audiooutput: [], videoinput: [], named: false });
    assert.equal(microphoneChanged(DEFAULT_DEVICE_PREFS, { ...DEFAULT_DEVICE_PREFS, speaker: 'x', camera: 'y' }), false);
    assert.equal(microphoneChanged(DEFAULT_DEVICE_PREFS, { ...DEFAULT_DEVICE_PREFS, echoCancellation: false }), true);
    assert.equal(microphoneChanged(DEFAULT_DEVICE_PREFS, { ...DEFAULT_DEVICE_PREFS, mic: 'usb-mic' }), true);
});

import './fixtures/call_modules.js';
import test from 'node:test';
import assert from 'node:assert/strict';
const { CallVision } = await import('../../skills/live_conversation/ui/call_video.js');

function fixture() {
    let bytes = new Uint8Array([1, 2, 3]);
    const frames = [], controls = [];
    const session = { state: 'live', active: true, epoch: 0,
        socket: { readyState: 1, bufferedAmount: 0, send: value => frames.push(value) },
        send: value => controls.push(value), changed() {}, view() {} };
    const env = { clearInterval() {}, document: { createElement: () => ({
        getContext: () => ({ drawImage() {} }),
        toBlob: callback => callback(new Blob([bytes], {type:'image/jpeg'})),
    }) } };
    const vision = new CallVision(session, env);
    vision.kind = 'screen';
    vision.video = { videoWidth:1280, videoHeight:720, pause() {} };
    return { vision, session, frames, controls, change: value => { bytes = new Uint8Array(value); } };
}

test('only exact JPEG duplicates are skipped; sharing starts after a sent frame and stops once', async () => {
    const f = fixture();
    await f.vision.tick();
    assert.equal(f.frames.length, 1);
    assert.deepEqual(f.controls, [{type:'share', sharing:true, kind:'screen'}]);
    f.vision.lastSentAt = 0;
    await f.vision.tick();
    assert.equal(f.frames.length, 1, 'elapsed time never forces an identical delivery');
    f.change([1, 2, 4]);
    await f.vision.tick();
    assert.equal(f.frames.length, 2, 'one changed byte must be sent; no similarity threshold');
    assert.equal(f.controls.length, 1, 'a frame does not announce another share');
    f.vision.stop(); f.vision.stop();
    assert.deepEqual(f.controls[1], {type:'share', sharing:false, kind:'screen'});
    assert.equal(f.controls.length, 2);
});

test('cadence and one frame in flight remain physical bounds; an unsent share is never announced', async () => {
    const f = fixture();
    f.vision.pendingBytes = f.session.socket.bufferedAmount = 20000;
    await f.vision.tick();
    assert.equal(f.frames.length, 0); assert.equal(f.controls.length, 0);
    f.session.socket.bufferedAmount = 16000;
    await f.vision.tick();
    assert.equal(f.frames.length, 1);
    f.change([9]); await f.vision.tick();
    assert.equal(f.frames.length, 1, 'one-second cadence survives byte differences');
    const unsent = fixture(); unsent.vision.stop();
    assert.equal(unsent.controls.length, 0);
});

test('an identical JPEG is retried in a new epoch, then suppressed in that epoch', async () => {
    const f = fixture();
    await f.vision.tick();
    f.session.epoch += 1; // the first send raced a flush and may have been rejected
    f.vision.lastSentAt = 0;
    await f.vision.tick();
    assert.equal(f.frames.length, 2, 'a rejected epoch is not a delivered duplicate');
    f.vision.lastSentAt = 0;
    await f.vision.tick();
    assert.equal(f.frames.length, 2, 'same-epoch exact duplicates stay suppressed');
    assert.equal(f.controls.length, 1, 'a retry is not a new share');
});

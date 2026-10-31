// Controller qualification only: media methods are controlled here, not a decoder or native WebView.
// The real browser lane is tests/test_chat_media_recovery_browser.py.
import assert from 'node:assert/strict';
import test from 'node:test';
import { fixture } from './helpers/media_dom.js';

function player(t) {
    const fx = fixture(); // builds the production createChatMedia controller
    document.getElementById = (id) => fx.tracker.created.find((node) => node.id === id) || null;
    t.mock.timers.enable({ apis: ['setTimeout'] }); // advance toast expiry without wall-clock sleeps
    const bubble = fx.controller.buildMediaBubble({
        type: 'video', role: 'assistant', download_url: '/api/files/recovery.mp4', mime: 'video/mp4',
    });
    document.body.appendChild(bubble);
    const media = bubble.querySelector('video');
    const button = (action) => bubble.querySelector(`[data-media-action="${action}"]`);
    const calls = [];
    media.load = () => {
        calls.push('load');
        // Resource selection resets the position/rate/error; mute is an element preference.
        media.currentTime = 0;
        media.playbackRate = 1;
        media.error = null;
    };
    media.play = () => { calls.push('play'); media.paused = false; return Promise.resolve(); };
    const emit = (type) => {
        for (const listener of media.listeners.get(type) || []) listener({ currentTarget: media });
    };
    const feedback = () => ({
        label: button('play').getAttribute('aria-label'),
        icon: button('play').innerHTML,
        toasts: document.body.querySelectorAll('.toast').map((node) => node.textContent),
    });
    t.after(() => { fx.controller.destroy(); fx.restore(); });
    return { ...fx, bubble, media, button, calls, emit, feedback };
}

function deferred() {
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}

for (const action of ['play', 'surface', 'repeat']) {
    for (const muted of [false, true]) {
        test(`${action} reloads failed media synchronously, preserving rate/mute and ${action === 'repeat' ? 'starting at zero' : 'position'} (muted=${muted})`, async (t) => {
            const fx = player(t);
            const { media, calls, button } = fx;
            media.error = { code: 4 };
            media.currentTime = 37.25;
            media.duration = 120;
            media.muted = muted;
            fx.emit('error');
            fx.emit('loadedmetadata');
            fx.emit('timeupdate');
            await button('speed').click();
            assert.equal(media.playbackRate, 1.5);
            assert.deepEqual(calls, [], 'failure, passive events and speed changes never start recovery');

            const gate = deferred();
            media.play = () => {
                calls.push('play');
                assert.equal(media.currentTime, action === 'repeat' ? 0 : 37.25);
                assert.equal(media.playbackRate, 1.5);
                assert.equal(media.muted, muted);
                media.paused = false;
                return gate.promise;
            };
            const pressing = (action === 'surface' ? media : button(action)).click();
            // No await has occurred: load AND play must still belong to this gesture's stack.
            assert.deepEqual(calls, ['load', 'play']);
            gate.resolve();
            await pressing;
            fx.emit('loadedmetadata');
            assert.equal(media.currentTime, action === 'repeat' ? 0 : 37.25);
            assert.equal(button('play').getAttribute('aria-label'), 'Pause');
            assert.deepEqual(fx.feedback().toasts, []);
            assert.deepEqual(calls, ['load', 'play']);
        });
    }
}

test('healthy pause/resume and Repeat use the same source; Repeat starts at zero', async (t) => {
    const { media, button, calls } = player(t);
    media.currentTime = 37.25;
    await button('play').click();
    await button('play').click();
    assert.equal(media.paused, true);
    assert.equal(media.currentTime, 37.25);
    await media.click();
    assert.equal(media.paused, false);
    assert.equal(media.currentTime, 37.25);
    await button('repeat').click();
    assert.equal(media.currentTime, 0);
    assert.deepEqual(calls, ['play', 'play', 'play']);
});

test('a permanently failed source gives one honest toast per intentional attempt and never retries itself', async (t) => {
    const fx = player(t);
    fx.media.error = { code: 4 };
    fx.media.play = () => {
        fx.calls.push('play');
        fx.media.error = { code: 4 };
        fx.media.paused = false; // WebKit can leave this set after a failed play.
        return Promise.reject(new Error('decoder unavailable'));
    };
    await fx.button('play').click();
    assert.deepEqual(fx.feedback().toasts, ['Could not play media: decoder unavailable']);
    assert.equal(fx.media.paused, true);
    assert.equal(fx.button('play').getAttribute('aria-label'), 'Play');
    fx.emit('error');
    fx.emit('pause');
    t.mock.timers.tick(60_000);
    await Promise.resolve();
    assert.deepEqual(fx.calls, ['load', 'play'], 'error events and elapsed time do not retry');
});

for (const newer of ['pause', 'repeat', 'release']) {
    test(`a rejected old play cannot change feedback after newer ${newer}`, async (t) => {
        const fx = player(t);
        const old = deferred();
        fx.media.error = { code: 4 };
        fx.media.play = () => { fx.calls.push('play'); fx.media.paused = false; return old.promise; };
        const pressing = fx.button('play').click();
        assert.deepEqual(fx.calls, ['load', 'play']);
        if (newer === 'pause') {
            await fx.button('play').click();
            assert.equal(fx.media.paused, true);
            assert.equal(fx.button('play').getAttribute('aria-label'), 'Play');
        } else if (newer === 'repeat') {
            fx.media.currentTime = 37.25;
            fx.media.play = () => { fx.calls.push('play'); return Promise.resolve(); };
            await fx.button('repeat').click();
            assert.equal(fx.media.currentTime, 0);
            assert.equal(fx.button('play').getAttribute('aria-label'), 'Pause');
        } else {
            fx.controller.release(fx.bubble);
            assert.equal(fx.media.paused, true);
            assert.equal(fx.button('play').listeners.get('click').size, 0);
        }
        const feedback = fx.feedback();
        const calls = [...fx.calls];
        old.reject(new Error('obsolete decoder failure'));
        await pressing;
        assert.deepEqual(fx.feedback(), feedback, 'neither a stale toast nor a control repaint');
        assert.deepEqual(fx.feedback().toasts, []);
        assert.deepEqual(fx.calls, calls, 'no stale recovery attempt');
    });
}

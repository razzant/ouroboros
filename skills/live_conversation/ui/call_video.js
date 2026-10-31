/** Browser camera and screen frames for a live call. Capture stays local until the owner
 * starts sharing. One frame in flight, a physical cadence and exact-duplicate suppression.
 */
import { encodeVideoFrame, MAX_VIDEO_FRAME_PAYLOAD } from './call_worklet.js';
import { loadDevicePrefs, cameraConstraints } from './call_devices.js';

import { acquire } from '/static/modules/media_capture.js';

const FRAME_MS = 1000;
// Audio keeps streaming beside a frame: a socket that never quite empties still lets the next frame go once
// what remains of the previous one is at most this much (about 170 ms of 48 kHz speech; a transport bound).
const QUEUED_AUDIO_SLACK_BYTES = 16 * 1024;

export function visionAvailability(env = globalThis) {
    const media = env.navigator?.mediaDevices;
    const shell = Boolean(env.pywebview) && /Linux/i.test(env.navigator?.userAgent || '');
    const telegram = Boolean(env.Telegram?.WebApp)
        || env.document?.documentElement?.dataset?.ouroborosHost === 'telegram';
    if (telegram) return { screen: false, camera: false, reason: 'Video sharing is unavailable in the Telegram mini app.' };
    if (shell) return { screen: false, camera: false, reason: 'Video sharing is unavailable in the Linux desktop app. Open this room in a browser.' };
    return { screen: typeof media?.getDisplayMedia === 'function',
        camera: typeof media?.getUserMedia === 'function',
        reason: 'Screen sharing is unavailable in this browser. Open this room in a browser with screen capture.',
        cameraReason: 'Camera sharing is unavailable in this browser.' };
}

function jpeg(canvas, quality) {
    return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
}

export class CallVision {
    constructor(session, env = globalThis) {
        this.session = session;
        this.env = env;
        this.kind = '';
        this.error = '';
        this.token = 0;
        this.seq = 0;
        this.lastSentAt = 0;
        this.lastBytes = null;
        this.lastEpoch = null;
        this.busy = false;
        this.cameraId = '';
        this.announced = '';
        this.pendingBytes = 0;  // the last frame's bytes still queued in the call socket (upper bound)
    }

    /** Video bytes still in the socket: never more than the socket holds, zero once it drained. */
    pending(bufferedAmount) {
        this.pendingBytes = Math.min(this.pendingBytes, bufferedAmount);
        return this.pendingBytes;
    }

    notify() { this.session.changed(this.session.view()); }

    async toggle(kind) {
        if (this.kind === kind) { this.stop(); return; }
        this.stop();
        const available = visionAvailability(this.env);
        if (!available[kind]) { this.error = available[`${kind}Reason`] || available.reason; this.notify(); return; }
        const token = this.token;
        let stream;
        try {
            const prefs = loadDevicePrefs(this.env);
            const pending = acquire({ kind, owner: this.session, label: 'Live Conversation', keepWhenHidden: true,
                constraints: kind === 'screen' ? true : cameraConstraints(prefs).video });
            this.pendingLease = pending;
            const lease = await pending;
            if (lease.code) throw new Error(lease.code);
            if (token !== this.token) { lease.release('cancelled'); return; }
            this.lease = lease;
            this.pendingLease = null;
            lease.signal.addEventListener('abort', () => this.stop(), { once: true });
            stream = lease.stream;
            if (kind === 'camera') this.cameraId = prefs.camera;
            if (token !== this.token || !this.session.active) { lease.release('cancelled'); return; }
            const video = this.env.document.createElement('video');
            video.muted = true; video.playsInline = true; video.srcObject = stream;
            await video.play();
            if (token !== this.token || !this.session.active) { lease.release('cancelled'); return; }
            this.video = video;
            this.stream = stream;
            this.kind = kind;
            this.error = '';
            this.lastSentAt = 0;
            for (const track of stream.getVideoTracks()) track.onended = () => { if (token === this.token) this.stop(); };
            this.timer = this.env.setInterval(() => { void this.tick(); }, FRAME_MS);
            this.notify();
            void this.tick();
        } catch (error) {
            stream?.getTracks().forEach(track => track.stop());
            if (token !== this.token) return;
            this.stop(false);
            this.error = error?.name === 'NotAllowedError' ? 'Video sharing was not allowed.'
                : 'Video sharing could not start.';
            this.notify();
        }
    }

    /** A shown camera starts again on the device chosen now. */
    restart() {
        const kind = this.kind;
        if (!kind) return Promise.resolve();
        this.stop(false);
        return this.toggle(kind);
    }

    stop(notify = true) {
        this.token++;
        if (this.timer != null) this.env.clearInterval(this.timer);
        this.timer = null;
        for (const track of this.stream?.getTracks() || []) { track.onended = null; track.stop(); }
        this.stream = null;
        const lease = this.lease, pending = this.pendingLease;
        this.lease = this.pendingLease = null;
        lease?.release('cancelled'); pending?.release?.('cancelled');
        if (this.video) { this.video.pause(); this.video.srcObject = null; this.video = null; }
        const changed = Boolean(this.kind || this.error);
        if (this.announced) this.session.send({ type: 'share', sharing: false, kind: this.announced });
        this.announced = '';
        this.kind = this.error = '';
        this.lastBytes = null;
        this.lastEpoch = null;
        // The last frame may still be queued in the call socket: its bytes stay counted until the socket drains
        // them (`pending` decays with bufferedAmount), or stopping a share would read as a stalled uplink (R15-2).
        if (changed && notify) this.notify();
    }

    async tick() {
        const token = this.token;
        const session = this.session;
        if (this.busy || !this.kind || session.state !== 'live' || session.socket?.readyState !== 1
            || Date.now() - this.lastSentAt < FRAME_MS) return;
        // One frame in flight at a time: a frame still queued in the socket is not followed by another,
        // so video never piles up behind live speech on a slow uplink.
        if (this.pending(session.socket.bufferedAmount) > QUEUED_AUDIO_SLACK_BYTES) return;
        const width = this.video?.videoWidth || 0, height = this.video?.videoHeight || 0;
        if (!width || !height) return;
        this.busy = true;
        try {
            const maxSide = this.kind === 'screen' ? 1280 : 768;
            let blob = null;
            for (const side of [maxSide, Math.round(maxSide * 0.75), Math.round(maxSide * 0.5)]) {
                const scale = Math.min(1, side / Math.max(width, height));
                const canvas = this.env.document.createElement('canvas');
                canvas.width = Math.max(1, Math.round(width * scale));
                canvas.height = Math.max(1, Math.round(height * scale));
                canvas.getContext('2d').drawImage(this.video, 0, 0, canvas.width, canvas.height);
                for (const quality of [0.7, 0.55, 0.4]) {
                    blob = await jpeg(canvas, quality);
                    if (blob?.size && blob.size <= MAX_VIDEO_FRAME_PAYLOAD) break;
                }
                if (blob?.size && blob.size <= MAX_VIDEO_FRAME_PAYLOAD) break;
            }
            if (token !== this.token || session.state !== 'live' || session.socket?.readyState !== 1) return;
            if (!blob?.size || blob.size > MAX_VIDEO_FRAME_PAYLOAD) throw new Error('JPEG frame unavailable');
            const data = new Uint8Array(await blob.arrayBuffer());
            if (token !== this.token || session.state !== 'live' || session.socket?.readyState !== 1) return;
            const epoch = session.epoch;
            if (this.lastEpoch === epoch && this.lastBytes?.length === data.length
                && data.every((byte, i) => byte === this.lastBytes[i])) return;
            session.socket.send(encodeVideoFrame({ epoch, seq: this.seq++, mime: 'image/jpeg', data }));
            this.pendingBytes = data.byteLength;
            this.lastSentAt = Date.now();
            this.lastBytes = data;
            this.lastEpoch = epoch;
            if (!this.announced) {
                this.announced = this.kind;
                session.send({ type: 'share', sharing: true, kind: this.kind });
            }
        } catch {
            if (token !== this.token) return;
            this.stop(false);
            this.error = 'Video capture paused.';
            this.notify();
        } finally { this.busy = false; }
    }
}

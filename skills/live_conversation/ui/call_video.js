/** Browser camera and screen frames for a live call. Capture stays local until the owner
 * starts sharing; a small changed thumbnail and a slow keep-alive limit billed frames.
 */
import { encodeVideoFrame, MAX_VIDEO_FRAME_PAYLOAD } from './call_worklet.js';
import { loadDevicePrefs, cameraConstraints } from './call_devices.js';

import { acquire } from '/static/modules/media_capture.js';

const FRAME_MS = 1000;
const KEEPALIVE_MS = 15000;
const SAMPLE_WIDTH = 16;
const SAMPLE_HEIGHT = 9;

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

function changedEnough(previous, next) {
    if (!previous) return true;
    let difference = 0;
    for (let i = 0; i < next.length; i += 4) {
        difference += Math.abs(next[i] - previous[i]) + Math.abs(next[i + 1] - previous[i + 1])
            + Math.abs(next[i + 2] - previous[i + 2]);
    }
    return difference / (next.length * 0.75) > 6;
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
        this.sample = null;
        this.busy = false;
        this.cameraId = '';
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
        this.kind = this.error = '';
        this.sample = null;
        if (changed && notify) this.notify();
    }

    async tick() {
        const token = this.token;
        const session = this.session;
        if (this.busy || !this.kind || session.state !== 'live' || session.socket?.readyState !== 1
            || session.socket.bufferedAmount > 1024 * 1024 || Date.now() - this.lastSentAt < FRAME_MS) return;
        const width = this.video?.videoWidth || 0, height = this.video?.videoHeight || 0;
        if (!width || !height) return;
        this.busy = true;
        try {
            const sample = this.env.document.createElement('canvas');
            sample.width = SAMPLE_WIDTH; sample.height = SAMPLE_HEIGHT;
            const sampleContext = sample.getContext('2d', { willReadFrequently: true });
            sampleContext.drawImage(this.video, 0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT);
            const pixels = sampleContext.getImageData(0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT).data;
            if (!changedEnough(this.sample, pixels) && Date.now() - this.lastSentAt < KEEPALIVE_MS) return;
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
            session.socket.send(encodeVideoFrame({ epoch: session.epoch, seq: this.seq++, mime: 'image/jpeg', data }));
            this.lastSentAt = Date.now();
            this.sample = new Uint8ClampedArray(pixels);
        } catch {
            if (token !== this.token) return;
            this.stop(false);
            this.error = 'Video capture paused.';
            this.notify();
        } finally { this.busy = false; }
    }
}

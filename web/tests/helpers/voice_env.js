export function deferred() {
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}

export function stream() {
    const track = new EventTarget();
    track.readyState = 'live';
    track.stops = 0;
    track.stop = () => { track.stops++; track.readyState = 'ended'; };
    return { getTracks: () => [track], getAudioTracks: () => [track], track };
}

export function environment() {
    const env = new EventTarget();
    env.document = new EventTarget();
    env.document.hidden = false;
    env.isSecureContext = true;
    env.location = { hostname: 'localhost', host: 'localhost:8765' };
    env.navigator = { mediaDevices: { getUserMedia: () => Promise.resolve(stream()) } };
    env.timers = new Map();
    let tick = 0;
    env.setTimeout = (fn, ms) => { env.timers.set(++tick, { fn, ms }); return tick; };
    env.clearTimeout = key => env.timers.delete(key);
    env.fireTimer = ms => {
        for (const [key, timer] of [...env.timers]) if (timer.ms === ms) {
            env.timers.delete(key); timer.fn();
        }
    };
    return env;
}

export class Recorder {
    static instances = [];
    static isTypeSupported(type) { return type === 'audio/webm;codecs=opus'; }
    constructor(stream, options) {
        this.stream = stream; this.options = options; this.mimeType = options.mimeType;
        this.state = 'inactive'; Recorder.instances.push(this);
    }
    start(ms) { this.slice = ms; this.state = 'recording'; }
    data(size) { this.ondataavailable?.({ data: new Blob([new Uint8Array(size)], { type: this.mimeType }) }); }
    stop() { this.state = 'inactive'; queueMicrotask(() => this.onstop?.()); }
}

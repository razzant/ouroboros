/* A recording stand-in for the Web Audio API (notifications' soft app tone): nothing here reaches an
   audio device. Every oscillator keeps its frequency, gain automation and start/stop times, so a test
   reads what WOULD sound; `tones` holds only the oscillators that were actually started.

   `resume(context)` decides what `AudioContext.resume()` returns: a value (the device is running at
   once) or a promise the test settles, to change the owner's choices while the device wakes. */
export function recordingAudio({ resume = () => undefined, state } = {}) {
    const contexts = [];
    const tones = [];
    class RecordingAudioContext {
        constructor() {
            this.currentTime = 0;
            this.destination = { node: 'destination' };
            if (state !== undefined) this.state = state;
            contexts.push(this);
        }

        resume() { return resume(this); }

        close() { this.state = 'closed'; }

        createOscillator() {
            return {
                type: '',
                frequency: { value: 0 },
                gain: null,
                connect(node) { this.gain = node; return node; },
                start(at) { this.startedAt = at; tones.push(this); },
                stop(at) { this.stoppedAt = at; },
            };
        }

        createGain() {
            const automation = [];
            const record = (kind) => (value, at) => { automation.push([kind, value, at]); };
            return {
                automation,
                gain: {
                    value: 1,
                    setValueAtTime: record('set'),
                    linearRampToValueAtTime: record('linear'),
                    exponentialRampToValueAtTime: record('exponential'),
                },
                connect(node) { this.output = node; return node; },
            };
        }
    }
    return { ctor: RecordingAudioContext, contexts, tones };
}

/** The peak a tone reaches (its linear attack target), or null. */
export const tonePeak = (osc) => osc?.gain?.automation.find(([kind]) => kind === 'linear')?.[1] ?? null;

/** A promise the test settles by hand. */
export function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
}

export const settle = async (rounds = 4) => {
    for (let i = 0; i < rounds; i += 1) await new Promise((done) => setTimeout(done, 0));
};

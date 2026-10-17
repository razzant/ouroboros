/** The call's light follows what is heard: the microphone at once, playback when it leaves the
 * speaker (`outputLatency`, else `baseLatency`, later than it was rendered). Colour comes from the
 * call's state (the capsule's `mode` and whose `voice` may show); amplitude comes from here, as two
 * numbers on the composer (`--call-mic`, `--call-out`, 0..1) and one `draw` call per frame for the
 * visual. One frame loop runs only while the room shows a call and something moves: a voice, its
 * fading envelope, or a phase that moves by itself (preparing, listening, thinking). Reduced
 * motion gets calm steps the stylesheet fades and still drawings, never a swell or a travelling light.
 */
import { followLevel } from './call_worklet.js';

// A level above this is somebody's voice; it has stopped below half of it (hysteresis).
const VOICE_GATE = 0.12;
// What reduced motion shows for a voice: a fixed, calm brightness, faded by the stylesheet.
const REDUCED_LEVEL = 0.6;
// Playback levels waiting for their moment: 32 windows of 20 ms outlast any output latency.
const MAX_PENDING = 32;
const MOVING_MODES = new Set(['trace', 'listen', 'think']);

export function createCallLight({ env = globalThis, visible = () => true, heard = () => {}, voice = () => {},
    draw = () => {} } = {}) {
    let target = null;
    let active = false;
    let mode = 'still';
    let speaker = '';
    let mic = 0;
    let out = 0;
    let micTarget = 0;
    let outTarget = 0;
    let playing = false;
    let ownerVoice = false;
    let frame = 0;
    let last = 0;
    const pending = [];
    const now = () => env.performance?.now?.() ?? Date.now();
    const reduced = () => Boolean(env.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches);

    function write() {
        target?.style.setProperty('--call-mic', mic.toFixed(3));
        target?.style.setProperty('--call-out', out.toFixed(3));
    }

    function hear(next) {
        if (next === playing) return;
        playing = next;
        heard(playing);
    }

    function due(time) {
        while (pending.length && pending[0].at <= time) {
            const next = pending.shift();
            outTarget = next.out;
            hear(next.playing);
        }
    }

    function schedule() {
        if (!frame && active && visible()) frame = env.requestAnimationFrame(tick);
    }

    function tick(time) {
        frame = 0;
        if (!active || !visible()) {
            // Off screen nothing glows; the light starts from stillness when the room returns.
            mic = out = 0;
            last = 0;
            write();
            return;
        }
        due(time);
        // A woken loop has no previous frame: it takes one frame's time, so the first frame moves.
        const dt = last ? Math.min(0.1, (time - last) / 1000) : 1 / 60;
        last = time;
        const calm = reduced();
        if (calm) {
            mic = micTarget > VOICE_GATE ? REDUCED_LEVEL : 0;
            out = outTarget > VOICE_GATE ? REDUCED_LEVEL : 0;
        } else {
            mic = followLevel(mic, micTarget, dt);
            out = followLevel(out, outTarget, dt);
            if (mic < 0.002) mic = 0;
            if (out < 0.002) out = 0;
        }
        write();
        draw({ mic, out, dt, time: time / 1000, mode, voice: speaker, owner: ownerVoice, reduced: calm });
        const moving = pending.length || (!calm && (mic || out || micTarget || outTarget || MOVING_MODES.has(mode)));
        if (moving) schedule();
        else last = 0;
    }

    function level({ mic: nextMic = 0, out: nextOut = 0, playing: nextPlaying = false, latency = 0 }) {
        if (!active) return;
        micTarget = nextMic;
        if (!ownerVoice && nextMic > VOICE_GATE && !playing) { ownerVoice = true; voice(); }
        else if (ownerVoice && nextMic < VOICE_GATE / 2) ownerVoice = false;
        pending.push({ at: now() + latency * 1000, out: nextOut, playing: nextPlaying });
        if (pending.length > MAX_PENDING) pending.splice(0, pending.length - MAX_PENDING);
        if (visible()) schedule();
        else due(now());  // hidden: `heard` stays true to the speaker for the words and the mark
    }

    function reset() {
        if (frame) env.cancelAnimationFrame?.(frame);
        frame = 0;
        last = 0;
        mic = out = micTarget = outTarget = 0;
        pending.length = 0;
        ownerVoice = false;
        write();
        hear(false);
    }

    const wake = () => { if (env.document?.visibilityState !== 'hidden') schedule(); };
    env.document?.addEventListener?.('visibilitychange', wake);
    return {
        attach(element) { target = element; write(); },
        setActive(next) {
            if (next === active) return;
            active = next;
            if (!active) reset();
        },
        /** The capsule's light state and whose voice may show; a change draws the next frame. */
        setMode(nextMode, nextVoice = '') {
            if (nextMode === mode && nextVoice === speaker) return;
            mode = nextMode;
            speaker = nextVoice;
            // A call that ended draws its still state once; a live one draws on its next frame.
            if (!active) draw({ mic: 0, out: 0, dt: 0, time: now() / 1000, mode, voice: speaker, owner: false, reduced: reduced() });
            else schedule();
        },
        level,
        wake,
        reset,
        heard: () => playing,
        levels: () => ({ mic, out }),
        destroy() {
            reset();
            active = false;
            target = null;
            env.document?.removeEventListener?.('visibilitychange', wake);
        },
    };
}

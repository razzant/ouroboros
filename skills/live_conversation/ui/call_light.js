/** The call's light follows what is heard: the microphone at once, playback when it leaves the
 * speaker (`outputLatency`, else `baseLatency`, later than it was rendered). Colour comes from the
 * call's state (the capsule's `mode` and whose `voice` may show); amplitude comes from here, as two
 * numbers on the composer (`--call-mic`, `--call-out`, 0..1) and one `draw` call per frame for the
 * visual. One frame loop runs only while the room shows a call and something moves: a voice, its
 * fading envelope, or a phase that moves by itself (preparing, listening, thinking). Reduced
 * motion gets calm steps the stylesheet fades and still drawings, never a swell or a travelling light.
 */
import { followLevel } from './call_worklet.js';

// A level this far above the room is somebody's voice; it has stopped below half of it (hysteresis).
const VOICE_GATE = 0.12;
// A call is full duplex, so the owner's voice is a state that holds, not a flicker of the meter: it begins
// once the voice has held a quarter of a second and ends after most of a second below the gate, so a pause
// between words, a clatter or a cough moves the wave without changing the words or the colour.
const OPEN_HOLD_S = 0.25;
const CLOSE_HOLD_S = 0.9;
// The room's own level falls with every quiet moment at once and creeps up this fast (level per second)
// while the room stays loud: a voice keeps dipping between its words, a fan, a street or a far conversation
// does not, so within a few seconds a steady background lifts the gate above itself.
const ROOM_RISE_PER_S = 0.08;
// What reduced motion shows for a voice: a fixed, calm brightness, faded by the stylesheet.
const REDUCED_LEVEL = 0.6;
// Playback levels waiting for their moment: 32 windows of 20 ms outlast any output latency.
const MAX_PENDING = 32;
const MOVING_MODES = new Set(['trace', 'listen', 'think']);

export function createCallLight({ env = globalThis, visible = () => true, heard = () => {}, voice = () => {},
    owner = () => {}, draw = () => {} } = {}) {
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
    let aboveSince = null;  // when the level last rose over the gate, while it stays there
    let belowSince = null;  // when it last fell under the release, while it stays there
    let room = 0;  // the room's own level, under the owner's voice
    let roomAt = null;
    let quiet = 0;  // the hold that still closes the owner's voice after the meter fell silent
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

    function roomLevel(at, heardLevel) {
        // A call starts in a quiet room; the level so far is the room's until this moment joins it.
        const before = room;
        const dt = roomAt === null ? 0 : Math.max(0, at - roomAt) / 1000;
        roomAt = at;
        room = heardLevel <= room ? heardLevel : Math.min(heardLevel, room + ROOM_RISE_PER_S * dt);
        return before;
    }

    function gate(at, voiced) {
        // The words follow the gate (`owner`), never a caption; the gate holds through short gaps either way.
        const over = voiced > VOICE_GATE && !playing;
        const under = voiced < VOICE_GATE / 2;
        aboveSince = over ? (aboveSince ?? at) : null;
        belowSince = under ? (belowSince ?? at) : null;
        if (!ownerVoice && over && at - aboveSince >= OPEN_HOLD_S * 1000) { ownerVoice = true; voice(); owner(true); }
        else if (ownerVoice && under && at - belowSince >= CLOSE_HOLD_S * 1000) { ownerVoice = false; owner(false); }
    }

    function level({ mic: nextMic = 0, out: nextOut = 0, playing: nextPlaying = false, latency = 0 }) {
        if (!active) return;
        const at = now();
        micTarget = nextMic;
        // The gate reads the voice above the room, so a steady background keeps the words and the colour calm;
        // the wave itself still shows what the microphone hears.
        gate(at, Math.max(0, nextMic - roomLevel(at, nextMic)));
        // The meter reports the first window of digital silence and then nothing until sound returns
        // (call_worklet.js LevelMeter), so the owner's voice still closes when its hold runs out.
        env.clearTimeout?.(quiet);
        quiet = !nextMic && !nextOut && !nextPlaying && ownerVoice
            ? env.setTimeout?.(() => { quiet = 0; if (active) gate(now(), 0); }, CLOSE_HOLD_S * 1000 + 20) || 0 : 0;
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
        room = 0;
        roomAt = null;
        env.clearTimeout?.(quiet);
        quiet = 0;
        aboveSince = belowSince = null;
        if (ownerVoice) { ownerVoice = false; owner(false); }
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
        speaking: () => ownerVoice,
        levels: () => ({ mic, out }),
        destroy() {
            reset();
            active = false;
            target = null;
            env.document?.removeEventListener?.('visibilitychange', wake);
        },
    };
}

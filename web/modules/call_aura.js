/** Visual B, the living capsule: the call's light runs along the composer capsule itself. A light
 * in the speaker's colour family travels the capsule's rim (the owner's blue, Ouroboros's crimson:
 * two layers cross-faded by opacity), wider and faster as the heard voice rises; behind the
 * capsule's frosted glass a soft glow lights it from within and spills just around it, the room
 * untouched; a small voice wave at its left end carries the voice itself. Preparing runs a
 * neutral light round the rim with a travelling pulse, thinking a soft crimson pulse, an error
 * holds still in amber, and a muted call is still while Ouroboros is silent.
 *
 * A visual is one contract, so another replaces this module and its call.css block and nothing
 * else: `mount` fills the capsule's slot, may add layers to the row and behind it, and draws one
 * frame when the light asks (`draw`); `mark` is the small static sign of Talk and of the shell's
 * mark. The light writes `--call-mic`/`--call-out` on the composer and the capsule writes
 * `data-call-light|voice|tone`; the stylesheet does the rest with opacity and transform.
 */
const WAVE_WIDTH = 64;
const WAVE_HEIGHT = 30;
// Three strands of the wave: [opacity, cycles across the wave, phase].
const STRANDS = [[1, 1.9, 0], [0.6, 2.7, 1.3], [0.35, 3.6, 2.4]];

/** The wave's height for one strand at `u` (0..1 across), in pixels from the middle. */
export function waveOffset(u, [alpha, cycles, phase], { time = 0, amp = 0, pulse = false } = {}) {
    const envelope = Math.sin(Math.PI * u) ** 2;
    const reach = envelope * alpha * WAVE_HEIGHT * 0.42;
    if (pulse) return Math.sin(u * Math.PI * 2 * cycles - time * 7 + phase) * reach
        * (0.12 + 0.18 * Math.max(0, Math.sin(u * 6 - time * 5)));
    return Math.sin(u * Math.PI * 2 * cycles + time * (4 + 6 * amp) + phase) * reach * amp;
}

/** What the wave shows for the light's state: whose colour, how high, a pulse or a voice. */
export function waveShape({ mic = 0, out = 0, time = 0, mode = 'listen', voice = 'owner', owner = false, reduced = false }) {
    const heardOuroboros = voice === 'ouroboros' && out > 0.02;
    const colour = mode === 'error' ? 'error'
        : heardOuroboros || mode === 'think' ? 'ouroboros'
            : mode !== 'muted' && (owner || (mic > 0.08 && mic >= out)) ? 'owner' : 'idle';
    if (reduced || mode === 'error' || mode === 'still') return { colour, amp: 0, pulse: false };
    if (mode === 'muted') return { colour, amp: heardOuroboros ? out : 0, pulse: false };
    if (mode === 'think' || mode === 'trace') return { colour, amp: 0, pulse: true };
    const idle = mode === 'listen' ? 0.07 + 0.03 * Math.sin(time * 2.2) : 0;
    return { colour, amp: Math.max(voice === 'ouroboros' ? out : Math.max(mic, out), idle), pulse: false };
}

export const CALL_VISUAL = Object.freeze({
    name: 'aura',
    mount({ slot, row, wrap, doc, env = globalThis }) {
        const canvas = doc.createElement('canvas');
        canvas.className = 'call-wave';
        canvas.setAttribute('aria-hidden', 'true');
        const ratio = Math.min(2, Math.max(1, Number(env.devicePixelRatio) || 1));
        canvas.width = WAVE_WIDTH * ratio;
        canvas.height = WAVE_HEIGHT * ratio;
        slot.replaceChildren(canvas);
        const context = canvas.getContext?.('2d') || null;
        const rim = doc.createElement('span');
        rim.className = 'call-rim';
        rim.setAttribute('aria-hidden', 'true');
        rim.innerHTML = '<i class="call-rim-trace"></i><i class="call-rim-listen"></i>'
            + '<i class="call-rim-owner"></i><i class="call-rim-ouroboros"></i>';
        row.append(rim);
        // The glow lives behind the capsule, so its frosted glass is lit from behind.
        const aura = doc.createElement('span');
        aura.className = 'call-aura';
        aura.setAttribute('aria-hidden', 'true');
        aura.innerHTML = '<i class="call-aura-owner"></i><i class="call-aura-ouroboros"></i>';
        wrap.append(aura);
        const fit = () => aura.style.setProperty('--call-row-height', `${row.offsetHeight}px`);
        const sizing = typeof env.ResizeObserver === 'function' ? new env.ResizeObserver(fit) : null;
        sizing?.observe(row);
        fit();
        let colours = null;  // read from the stylesheet once per theme, never per frame
        const recolour = () => { colours = null; };
        env.addEventListener?.('ouro:theme-changed', recolour);
        let spin = 0;

        function palette() {
            if (!colours) {
                const style = env.getComputedStyle?.(wrap);
                const read = name => style?.getPropertyValue(name).trim() || 'currentColor';
                colours = { owner: read('--call-wave-owner'), ouroboros: read('--call-wave-ouroboros'),
                    idle: read('--call-wave-idle'), error: read('--call-wave-error') };
            }
            return colours;
        }

        function drawWave(shape, time) {
            if (!context) return;
            context.setTransform(ratio, 0, 0, ratio, 0, 0);
            context.clearRect(0, 0, WAVE_WIDTH, WAVE_HEIGHT);
            context.lineCap = 'round';
            context.strokeStyle = palette()[shape.colour];
            for (const strand of STRANDS) {
                context.beginPath();
                for (let x = 0; x <= WAVE_WIDTH; x += 2) {
                    const y = WAVE_HEIGHT / 2 + waveOffset(x / WAVE_WIDTH, strand, { time, amp: shape.amp, pulse: shape.pulse });
                    if (x) context.lineTo(x, y); else context.moveTo(x, y);
                }
                context.globalAlpha = strand[0] * (shape.colour === 'idle' ? 0.55 : 0.95);
                context.lineWidth = 3.2 * strand[0] + 1;
                context.stroke();
            }
            context.globalAlpha = 1;
        }

        return {
            draw(state) {
                const { mic = 0, out = 0, dt = 0, time = 0, mode = 'listen', reduced = false } = state;
                const lit = Math.max(mic, out, mode === 'think' ? 0.32 + 0.14 * Math.sin(time * 5) : 0,
                    mode === 'trace' ? 0.18 : 0, mode === 'listen' ? 0.1 : 0);
                // The rim's light goes round faster and wider as the voice rises; reduced motion holds it.
                if (!reduced && !['muted', 'error', 'still'].includes(mode)) spin = (spin + dt * (36 + 220 * lit)) % 360;
                rim.style.setProperty('--call-spin', `${spin.toFixed(1)}deg`);
                rim.style.setProperty('--call-rim-level', (reduced ? 0 : lit).toFixed(3));
                drawWave(waveShape(state), time);
            },
            destroy() {
                sizing?.disconnect();
                env.removeEventListener?.('ouro:theme-changed', recolour);
                slot.replaceChildren();
                rim.remove();
                aura.remove();
            },
        };
    },
    mark(doc) {
        const mark = doc.createElement('span');
        mark.className = 'call-entry-mark';
        mark.setAttribute('aria-hidden', 'true');
        mark.innerHTML = '<i></i><i></i><i></i><i></i><i></i>';
        return mark;
    },
});

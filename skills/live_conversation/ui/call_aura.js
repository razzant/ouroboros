/** The wave in the glass: the call's light is the owner's wave, drawn inside the composer capsule's
 * frosted glass. Three strands of a sine whose envelope is zero at both ends (`STRANDS`, `waveShape`)
 * run as tapered ribbons across the capsule's width, after the words (a phone's chip has no room there: its
 * wave is the words' second line, under the state word, while a voice speaks or the line is empty).
 * The strands and a faint tint of the glass take the speaker's hue (owner blue, Ouroboros crimson) and
 * change it by a slow direct cross-fade: the new hue eases in while the old one eases out, so for about a
 * second both are lit. Whose hue is a held state (the light's gate and the words' voice), never the meter:
 * a call is full duplex, and a noisy room or a pause between words must not flick it. The glass's outline
 * breathes with the heard level, a spring of a few pixels that never moves the layout. Thinking is a soft
 * travelling pulse that the wave eases into and out of; working draws the strands in to the room's three
 * dots and back; muted is one dashed neutral line; an error holds still in amber. In the big view
 * (`data-call-view="big"`) the same glass and wave are larger.
 *
 * A visual is one contract, so another replaces this module and its call.css block and nothing
 * else: `mount` fills the capsule's slot (here the wave's canvas and the work dots), may add layers
 * to the row (here the glass behind it) and draws one frame when the light asks (`draw`); `mark` is
 * the small static sign of Talk and of the shell's mark. The light writes `--call-mic`/`--call-out`
 * on the composer and the capsule writes `data-call-light|voice|tone|view|work|line`.
 */
import { LEVEL_CEILING_DB, LEVEL_CURVE, LEVEL_FLOOR_DB } from './call_worklet.js';

// Three strands of the wave: [opacity, cycles across the wave, phase].
export const STRANDS = [[1, 1.9, 0], [0.6, 2.7, 1.3], [0.35, 3.6, 2.4]];
// The level's last half second, travelling outward from the wave's middle.
const HISTORY = 26;
const HISTORY_STEP = 0.024;
// After a change of size or a still state the visual paints its own frames this long, so springs settle.
const SETTLE_SECONDS = 0.9;
// A change of hue: the new one eases in while the old one eases out, with this time constant (seconds).
const BLEND_SECONDS = 0.45;
// Listening's wave and thinking's pulse ease into each other with this time constant (seconds).
const MODE_SECONDS = 0.35;
const HUES = ['idle', 'owner', 'ouroboros', 'error'];
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const ease = (current, target, dt, up, down) => current + (target - current) * (1 - Math.exp(-Math.max(0, dt) / (target > current ? up : down)));
const rgba = (c, alpha) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${clamp(alpha, 0, 1).toFixed(3)})`;
const triplet = (text, fallback) => {
    const numbers = String(text || '').match(/[\d.]+/g);
    return numbers && numbers.length >= 3 ? numbers.slice(0, 3).map(Number) : fallback;
};

/** What the wave shows for the light's state: whose colour, how high, a pulse or a voice. */
// The wave reads a voice as if its level filled at -18 dBFS (the owner's choice of 18.10): an ordinary voice
// swings wider and loud syllables reach the top. The level itself, and the voice gate it feeds, keep their
// -12 dB ceiling (call_worklet.js), so background noise is no more "speaking" than before.
const WAVE_CEILING_DB = -18;
const WAVE_GAIN = ((LEVEL_CEILING_DB - LEVEL_FLOOR_DB) / (WAVE_CEILING_DB - LEVEL_FLOOR_DB)) ** LEVEL_CURVE;
const loud = level => Math.min(1, level * WAVE_GAIN);

export function waveShape({ mic = 0, out = 0, time = 0, mode = 'listen', voice = 'owner', owner = false, reduced = false }) {
    const heardOuroboros = voice === 'ouroboros' && out > 0.02;
    // Ouroboros keeps its hue through the pauses of an answer (the words' voice); the owner has theirs while the
    // light's gate holds their voice. The meter alone, a room's noise included, only moves the wave.
    const colour = mode === 'error' ? 'error'
        : voice === 'ouroboros' || mode === 'think' ? 'ouroboros'
            : mode !== 'muted' && owner ? 'owner' : 'idle';
    if (reduced || mode === 'error' || mode === 'still') return { colour, amp: 0, pulse: false };
    if (mode === 'muted') return { colour, amp: heardOuroboros ? loud(out) : 0, pulse: false };
    if (mode === 'think' || mode === 'trace') return { colour, amp: 0, pulse: true };
    const idle = mode === 'listen' ? 0.07 + 0.03 * Math.sin(time * 2.2) : 0;
    return { colour, amp: Math.max(loud(voice === 'ouroboros' ? out : Math.max(mic, out)), idle), pulse: false };
}

/** Talk's sign: the three strands in miniature, still. */
function waveMark(doc) {
    const mark = doc.createElement('span');
    mark.className = 'call-entry-mark';
    mark.setAttribute('aria-hidden', 'true');
    const path = ([alpha, cycles, phase]) => {
        let d = '';
        for (let k = 0; k <= 36; k += 1) {
            const u = k / 36;
            const y = 7 + 4.8 * alpha * Math.sin(Math.PI * u) ** 2 * Math.sin(2 * Math.PI * cycles * u + phase);
            d += `${k ? 'L' : 'M'}${(1.2 + u * 17.6).toFixed(2)} ${y.toFixed(2)}`;
        }
        return d;
    };
    mark.innerHTML = '<svg width="20" height="14" viewBox="0 0 20 14" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">'
        + `<path d="${path(STRANDS[2])}" stroke-width="1" opacity="0.4"/><path d="${path(STRANDS[1])}" stroke-width="1.3" opacity="0.65"/>`
        + `<path d="${path(STRANDS[0])}" stroke-width="1.8"/></svg>`;
    return mark;
}

function mount({ slot, row, wrap, doc, env = globalThis }) {
    // The glass: the row's own surface while a call is live, tinted by the speaker's hue, breathing outward.
    const glass = doc.createElement('span');
    glass.className = 'call-glass';
    glass.setAttribute('aria-hidden', 'true');
    glass.innerHTML = '<span class="call-glass-tint call-glass-owner"></span><span class="call-glass-tint call-glass-ouroboros"></span>';
    row.append(glass);
    const canvas = doc.createElement('canvas');
    canvas.className = 'call-wave';
    canvas.setAttribute('aria-hidden', 'true');
    const work = doc.createElement('span');
    work.className = 'call-work chat-live-typing';
    work.setAttribute('aria-hidden', 'true');
    work.innerHTML = '<span></span><span></span><span></span>';
    slot.replaceChildren(canvas, work);
    const context = canvas.getContext?.('2d') || null;
    const ratio = Math.min(2, Math.max(1, Number(env.devicePixelRatio) || 1));
    const big = () => wrap.dataset.callView === 'big';

    // ---- where the wave lives: measured on a change of size, never per frame ----
    let width = 300;
    let height = 52;
    let x0 = 20;
    let x1 = 240;
    let centre = 0.5;
    let band = 190;  // the big view's free half-height; a phone's small wave, its reach
    let place = '';  // where a phone's chip keeps its small wave (call.css --call-wave-place), else after the words
    let mounted = false;
    function fit() {
        width = Math.max(24, slot.clientWidth || 0);
        height = Math.max(16, slot.clientHeight || 0);
        if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio);
        if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio);
        const edge = slot.getBoundingClientRect?.() || { left: 0 };
        const buttons = row.querySelector('.call-buttons')?.getBoundingClientRect?.();
        const words = row.querySelector('.call-words');
        place = String(env.getComputedStyle?.(slot)?.getPropertyValue?.('--call-wave-place') || '').trim();
        centre = big() && (env.innerWidth || 0) >= 980 ? 0.54 : 0.5;
        band = 190;
        if (wrap.dataset.callView === 'sheet') { x0 = 3; x1 = width - 3; } else if (big()) {
            x0 = 28;
            x1 = width - 28;
            // The wave lives in the free band between the words above it and the controls below it.
            const top = words?.getBoundingClientRect?.().bottom || 0;
            const bottom = buttons?.top || 0;
            if (bottom > top + 80) {
                centre = ((top + bottom) / 2 - (edge.top || 0)) / height;
                band = (bottom - top) / 2 - 12;
            }
        } else {
            x0 = 20;
            x1 = buttons?.width ? Math.max(x0 + 40, buttons.left - edge.left - 10) : width - 20;
            const box = words?.getBoundingClientRect?.();
            const state = row.querySelector('.call-state')?.getBoundingClientRect?.();
            if (place === 'line' && box?.width && state?.height) {
                // A phone's chip: the wave is the words' second line. It runs from the words' left edge to the
                // controls, between the state word's line and the glass's lower edge, so it never touches a
                // letter; the line holds words instead when there is something to say (call_capsule.js).
                x0 = box.left - edge.left;
                const top = state.bottom - (edge.top || 0) + 1;
                const bottom = height - 1.5;
                centre = (top + bottom) / 2 / height;
                band = Math.max(2, (bottom - top) / 2 - 3);  // the reach, less a speaking strand's soft band
            } else {
                place = '';
                // The words end where their right-most visible line ends: the state and its time share one row,
                // so the widest child alone falls short by the time's width.
                const right = Math.max(0, ...[...(words?.children || [])]
                    .filter(child => !child.hidden && child.getClientRects?.().length)
                    .map(child => child.getBoundingClientRect().right));
                if (right) {
                    // The wave starts after the words, never under them; a row with no room for a whole wave
                    // there draws none (the glass's colour and breath still say who speaks).
                    const end = right - edge.left + 8;
                    x0 = end + (x1 - end >= 190 ? 20 : 4);
                    if (x1 - x0 < 44) x1 = x0;
                }
            }
        }
        work.style.setProperty('--call-work-x', `${Math.round((x0 + x1) / 2)}px`);
        work.style.setProperty('--call-work-y', `${(centre * 100).toFixed(1)}%`);
        settle();
        if (mounted) frame();  // a resized canvas is blank: paint it before the browser does
    }
    const sizing = typeof env.ResizeObserver === 'function' ? new env.ResizeObserver(fit) : null;
    for (const element of [slot, row, row.querySelector('.call-buttons'), row.querySelector('.call-words')])
        if (element) sizing?.observe(element);

    let colours = null;  // read from the stylesheet once per theme, never per frame
    const recolour = () => { colours = null; frame(); };
    env.addEventListener?.('ouro:theme-changed', recolour);
    function palette() {
        if (!colours) {
            const style = env.getComputedStyle?.(wrap);
            const read = (name, fallback) => triplet(style?.getPropertyValue(name), fallback);
            colours = { owner: read('--call-owner-rgb', [47, 107, 214]), ouroboros: read('--call-ouro-rgb', [201, 53, 69]),
                error: read('--call-amber-rgb', [180, 83, 9]), neutral: read('--neutral-rgb', [28, 33, 51]),
                rest: Number(style?.getPropertyValue('--call-wave-rest')) || 0.5,
                band: Number(style?.getPropertyValue('--call-wave-band')) || 0.12 };
        }
        return colours;
    }

    // ---- the voice, as the wave carries it ----
    const history = new Float32Array(HISTORY);
    let historyClock = 0;
    const heardAt = (distance) => {
        const x = clamp(distance, 0, 1) * (HISTORY - 1);
        const i = Math.floor(x);
        return history[i] + (history[Math.min(HISTORY - 1, i + 1)] - history[i]) * (x - i);
    };
    let amp = 0;          // the level's spring
    let velocity = 0;
    let phase = 0;        // the travelling phase
    let busy = 0;         // working: the strands draw in to the three dots
    let pulsing = 0;      // thinking's pulse, eased in and out of listening's wave
    // Each colour's share of the strands and the glass: the one in use eases to 1, the others to 0.
    const weight = { idle: 1, owner: 0, ouroboros: 0, error: 0 };
    let glow = 0;         // the glass's tint level, shared by the voices in use
    let breath = 0;       // the outline's spring
    let shown = 1;        // a phone's wave on the words' second line: 0 while the line holds words
    let breathVelocity = 0;
    const tint = { owner: 0, ouroboros: 0 };
    const written = new Map();
    const put = (target, name, value) => {
        if (written.get(name) === value) return;
        written.set(name, value);
        target.style.setProperty(name, value);
    };
    const now = () => (env.performance?.now?.() ?? Date.now()) / 1000;
    let input = null;
    let clock = 0;
    let settling = 0;
    let settleUntil = 0;

    const points = [];
    function line([alpha, cycles, offset], pulseShare, cy, reach, spanScale, time) {
        // One strand's centre line across the span, sampled every ~3 px.
        const span = x1 - x0;
        const n = Math.max(24, Math.min(260, Math.round(span / 3)));
        const k = clamp(span / 300, 1, big() ? 2.4 : 2);
        points.length = 0;
        for (let i = 0; i <= n; i += 1) {
            const u = i / n;
            const uu = (u - 0.5) / spanScale + 0.5;  // working draws the span in to its middle
            const envelope = uu <= 0 || uu >= 1 ? 0 : Math.sin(Math.PI * uu) ** 2;
            let y = 0;
            if (pulseShare > 0.001) {
                y += pulseShare * Math.sin(uu * Math.PI * 2 * cycles * k - time * 7 + offset) * reach * envelope * alpha
                    * (0.14 + 0.2 * Math.max(0, Math.sin(uu * 6 - time * 5)));
            }
            if (pulseShare < 0.999) {
                const local = amp * 0.42 + 0.58 * heardAt(Math.abs(uu - 0.5) * 2) * (amp > 0.005 ? 1 : 0);
                y += (1 - pulseShare) * Math.sin(uu * Math.PI * 2 * cycles * k + phase + offset) * reach * envelope * alpha * local;
            }
            points.push(x0 + u * span, cy - y);
        }
        return points;
    }

    function ribbon(p, colour, alpha, body, tip) {
        // A tapered ribbon along the line: widest in the middle, round caps at the ends.
        const n = p.length / 2;
        context.beginPath();
        for (let pass = 0; pass < 2; pass += 1) {
            for (let k = 0; k < n; k += 1) {
                const i = pass ? n - 1 - k : k;
                const a = Math.max(0, i - 1);
                const b = Math.min(n - 1, i + 1);
                const tx = p[b * 2] - p[a * 2];
                const ty = p[b * 2 + 1] - p[a * 2 + 1];
                const length = Math.hypot(tx, ty) || 1;
                const half = (tip + (body - tip) * Math.sin(Math.PI * (i / (n - 1))) ** 0.6) / 2;
                const side = pass ? -half : half;
                const px = p[i * 2] - (ty / length) * side;
                const py = p[i * 2 + 1] + (tx / length) * side;
                if (!pass && !k) context.moveTo(px, py); else context.lineTo(px, py);
            }
        }
        context.closePath();
        context.fillStyle = rgba(colour, alpha);
        context.fill();
        for (const i of [0, n - 1]) {
            context.beginPath();
            context.arc(p[i * 2], p[i * 2 + 1], tip / 2, 0, Math.PI * 2);
            context.fill();
        }
    }

    function frame() {
        if (!input) return;
        const t = now();
        const dt = clock ? Math.min(0.1, Math.max(0, t - clock)) : 1 / 60;
        clock = t;
        const shape = waveShape(input);
        const reduced = Boolean(input.reduced);
        const large = big();
        const level = shape.amp;
        const working = input.mode === 'think' && wrap.dataset.callWork === '1';

        historyClock += dt;
        while (historyClock >= HISTORY_STEP) { historyClock -= HISTORY_STEP; history.copyWithin(1, 0, HISTORY - 1); }
        history[0] = reduced ? 0 : level;
        // A spring on the level: it rises within a few frames, barely overshoots and lets go gently.
        velocity += (140 * (level - amp) - 22 * velocity) * dt;
        amp = reduced ? level : Math.max(0, amp + velocity * dt);
        phase += reduced ? 0 : dt * (4 + 4.5 * amp);
        busy = reduced ? (working ? 1 : 0) : ease(busy, working ? 1 : 0, dt, 0.28, 0.22);
        pulsing = reduced ? Number(shape.pulse) : ease(pulsing, shape.pulse ? 1 : 0, dt, MODE_SECONDS, MODE_SECONDS);

        // A direct cross-fade: the new speaker's hue eases in while the old one eases out at the same pace,
        // so for a moment the strands and the glass carry both voices' colours.
        for (const name of HUES) weight[name] = reduced ? Number(name === shape.colour)
            : ease(weight[name], name === shape.colour ? 1 : 0, dt, BLEND_SECONDS, BLEND_SECONDS);
        const colours = palette();
        const hues = { idle: colours.neutral, owner: colours.owner, ouroboros: colours.ouroboros, error: colours.error };
        const ink = [0, 1, 2].map(i => HUES.reduce((sum, name) => sum + hues[name][i] * weight[name], 0));
        const rest = weight.idle;  // how far the strands have settled to the room's quiet ink, eased with the hue

        // The glass: tinted by the voices in use, its outline breathing with the level.
        const lit = clamp(amp * 1.4, 0, 1) * (working ? 0.5 : 1) * (1 - weight.idle);
        glow = reduced ? 0.5 : ease(glow, working ? 0.2 : 0.18 + 0.82 * lit, dt, 0.4, 0.35);
        for (const name of ['owner', 'ouroboros']) tint[name] = glow * weight[name];
        const reach = reduced ? 0 : (large ? 8 : 4.5) * clamp(amp * 1.25, 0, 1);
        breathVelocity += (170 * (reach - breath) - 19 * breathVelocity) * dt;
        breath = reduced ? 0 : Math.max(0, breath + breathVelocity * dt);
        put(row, '--call-breath', breath.toFixed(2));
        put(row, '--call-tint-owner', tint.owner.toFixed(3));
        put(row, '--call-tint-ouroboros', tint.ouroboros.toFixed(3));
        // On a phone's chip the words' second line is the wave's only while it has no words to show.
        const free = place !== 'line' || wrap.dataset.callLine !== 'text';
        shown = reduced ? (free ? 1 : 0) : ease(shown, free ? 1 : 0, dt, 0.08, 0.05);
        const dots = busy > 0.55 && x1 > x0 && shown > 0.5 ? '1' : '';  // the dots stand only where the wave has room
        if ((slot.dataset.work || '') !== dots) slot.dataset.work = dots;
        const sheet = wrap.dataset.callView === 'sheet';
        if (!context || !(wrap.dataset.call === 'live' || (sheet && input.mode === 'error'))) return;

        // ---- the three strands ----
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, width, height);
        if (x1 - x0 < 1 || shown < 0.01) return;  // no room, or the words' second line holds words
        context.globalAlpha = shown;
        const cy = height * centre;
        const span = x1 - x0;
        // The chip's canvas is the whole glass: 0.4 of it is 0.6 of the capsule inside the glass's padding.
        const reachPx = large ? Math.min(height * 0.4, band) : place ? band : height * 0.4;
        const tip = large ? 0.9 : 0.7;
        if (input.mode === 'error') {
            ribbon(line(STRANDS[0], 0, cy, 0, 1, t), colours.error, 0.95, large ? 3.2 : 2.2, tip);
            return;
        }
        if (input.mode === 'muted' && shape.amp === 0) {
            // Muted: one quiet dashed line in the room's ink.
            context.beginPath();
            context.moveTo(x0, cy);
            context.lineTo(x1, cy);
            context.strokeStyle = rgba(colours.neutral, colours.rest * 0.9);
            context.lineWidth = large ? 2.4 : 1.8;
            context.lineCap = 'round';
            context.setLineDash?.([0.1, large ? 8 : 6]);
            context.stroke();
            context.setLineDash?.([]);
            return;
        }
        if (busy > 0.98) return;  // the three dots stand where the strands drew in
        const gone = 1 - busy * 0.97;
        const spanScale = 1 - 0.92 * busy;
        const restBoost = reduced && weight.idle < 0.5 ? 1.5 : 1;
        for (let s = STRANDS.length - 1; s >= 0; s -= 1) {
            const strand = STRANDS[s];
            const p = line(strand, pulsing, cy, reachPx, spanScale, t);
            const alpha = strand[0] * (colours.rest * rest + 0.96 * (1 - rest)) * gone;
            // A phone's small wave draws slimmer strands with a narrower soft band, so it keeps clear of the words.
            const body = ((large ? 2.1 : 1.35) + (large ? 3.6 : 2) * strand[0]) * restBoost * (1 - 0.1 * rest)
                * (place ? 0.72 : 1);
            const lift = (1 - rest) * (1 - pulsing);  // a voice's soft light, faded in and out with its hue
            if (s === 0 && lift > 0.01) {
                // The glass's own light: a soft body between the main strand and its mirror.
                context.beginPath();
                for (let i = 0; i < p.length; i += 2) { if (i) context.lineTo(p[i], p[i + 1]); else context.moveTo(p[i], p[i + 1]); }
                for (let i = p.length - 2; i >= 0; i -= 2) context.lineTo(p[i], cy + (cy - p[i + 1]));
                context.closePath();
                context.fillStyle = rgba(ink, 0.07 * lit * gone * lift);
                context.fill();
            }
            if (rest < 0.99) ribbon(p, ink, alpha * colours.band * 1.6 * (1 - rest), body * (place ? 2.4 : 3.6), tip * 2);
            ribbon(p, ink, alpha, body, tip);
        }
    }

    function settleFrame() {
        settling = 0;
        if (now() > settleUntil) return;
        frame();
        settling = env.requestAnimationFrame?.(settleFrame) || 0;
    }
    function settle() {
        settleUntil = Math.max(settleUntil, now() + SETTLE_SECONDS);
        if (!settling && input) settling = env.requestAnimationFrame?.(settleFrame) || 0;
    }

    fit();
    mounted = true;
    return {
        draw(state) {
            input = state;
            frame();
            // Still states stop the light's own frames: the springs finish easing on a few of ours.
            if (['muted', 'error', 'still'].includes(state.mode) || wrap.dataset.callMove === '1') settle();
        },
        destroy() {
            if (settling) env.cancelAnimationFrame?.(settling);
            sizing?.disconnect();
            env.removeEventListener?.('ouro:theme-changed', recolour);
            slot.replaceChildren();
            delete slot.dataset.work;
            glass.remove();
            for (const name of ['--call-breath', '--call-tint-owner', '--call-tint-ouroboros']) row.style.removeProperty?.(name);
        },
    };
}

export const CALL_VISUAL = Object.freeze({ name: 'glasswave', mount, mark: waveMark });

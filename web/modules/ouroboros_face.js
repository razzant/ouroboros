// The face of an Ouroboros of the owner's own on another computer (DESIGN "Circles and Add", Your Ouroboroses): the app
// icon's snake, byte for byte (assets/icon.svg), in one colour of the free palette below. Same seed, same face: the
// seed is the other installation's install id, hashed onto one combination; a stranger's seed never rolls the app
// icon, which stays this installation's own face. A leaf: strings only, no DOM, no network.

const BODY = 'M589.99 252.1A8 8 0 0 1 598.78 242.47A267.08 267.08 0 0 1 757.16 320.68C801.52 365.95 822.7 419.83 832.3 463.36A328 328 0 0 1 255.12 737.96C245.26 725.54 232.08 707.09 215.94 678.26C201.86 653.11 184.64 618.47 169.64 577.52C150.3 524.72 143.13 484.58 143.13 448.22C143.13 401.88 154.77 348.74 188.14 302.02C205.09 278.29 238.92 240.44 301.52 213.68C334.55 199.56 368.47 191.57 405.94 185.98C420.08 183.87 433.73 182.25 443.23 181.33C451.69 180.51 459.49 179.99 466.9 180.05C475.2 180.12 483.01 180.92 490.81 182.72A98 98 0 0 1 542.47 342.81C512.01 377.56 465.28 381.32 425.15 409.26A152 152 0 1 0 661.69 507.61C657.37 483.11 641.06 425.78 566.84 367.01A8 8 0 0 1 565.56 355.74A124 124 0 0 0 589.99 252.1Z';
const TILE = '#e5dcdc', INK = '#130d11', RED = '#e0200a', LIGHT_EYE = '#f6efe6';

// Wedges from the ring's centre (measured from BODY): the head, the tail's tip and the darker tail, trimmed to the body.
const CX = 512, CY = 534, R0 = 100, R1 = 480;
const round = (value) => String(Math.round(value * 10) / 10);
const point = (deg, r) => { const a = deg * Math.PI / 180; return `${round(CX + r * Math.sin(a))} ${round(CY - r * Math.cos(a))}`; };
const sector = (a, b) => {
    const big = b - a > 180 ? 1 : 0;
    return `M${point(a, R1)}A${R1} ${R1} 0 ${big} 1 ${point(b, R1)}L${point(b, R0)}A${R0} ${R0} 0 ${big} 0 ${point(a, R0)}Z`;
};
const HEAD = sector(250, 372), TIP = sector(14, 50), DARK_TAIL = sector(14, 150);

/** CIE Lab of a #rrggbb colour (the filter and the eye compare colours by ΔE76). */
function lab(hex) {
    const [r, g, b] = [1, 3, 5].map((i) => {
        const c = parseInt(hex.slice(i, i + 2), 16) / 255;
        return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    });
    const x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047, y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    const z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
    const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
    return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}
const deltaE = (a, b) => { const p = lab(a), q = lab(b); return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]); };

// The free palette: tones all round the wheel on the icon's own tile (no lilac, orchid or coral), each with four ways
// of painting the snake.
const SETS = [
    ['cobalt', '#2a4fd6'], ['azure', '#1b86cf'], ['turquoise', '#14a3a0'], ['emerald', '#0c8656'],
    ['ochre', '#c4880f'], ['vermilion', '#d6431c'], ['garnet', '#8e1b2e'], ['graphite', '#5f6670'],
];
const PAINTS = { none: { layers: [] }, head: { layers: [HEAD], head: true }, cap: { layers: [HEAD + TIP], head: true },
    darktail: { layers: [DARK_TAIL] } };

/** Why a body and paint are not both clear and contrasting, or '' when they are. */
function dropped(body, paint) {
    if (deltaE(body, TILE) < 40) return 'body merges with the tile';
    if (deltaE(body, INK) < 30) return 'an almost black body reads as the app icon';
    if (paint !== 'none' && deltaE(INK, body) < 45) return 'black paint merges with a dark body';
    return '';
}

/** Every face a stranger's seed may roll, in a fixed order. */
export const FACES = Object.freeze(SETS.flatMap(([id, body]) => Object.keys(PAINTS)
    .filter((paint) => !dropped(body, paint)).map((paint) => Object.freeze({ id: paint === 'none' ? id : `${id}.${paint}`, body, paint }))));

/** The eye: the icon's red where it shows, else its ink, a light eye only on a dark body. */
function eyeFor(body) {
    const e = lab(RED), b = lab(body), dl = Math.abs(e[0] - b[0]), chroma = Math.hypot(b[1], b[2]);
    let dh = Math.abs(Math.atan2(e[2], e[1]) - Math.atan2(b[2], b[1])) * 180 / Math.PI;
    if (dh > 180) dh = 360 - dh;
    const inkOrLight = Math.abs(lab(INK)[0] - b[0]) >= 25 ? INK : LIGHT_EYE;
    if (dh < 40 && dl < 25) return inkOrLight;
    if (dl >= 15 || (dh >= 120 && chroma >= 35) || (dh >= 110 && chroma >= 40 && b[0] >= 45)) return RED;
    return inkOrLight;
}

/** FNV-1a: one seed, one face, on every device. */
function hash(text) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
}

/** The face a seed rolls: a face id rolls itself, any other text is hashed onto one. */
export function faceOf(seed) {
    const text = String(seed ?? '');
    return FACES.find((face) => face.id === text) || FACES[hash(text) % FACES.length];
}

const svgOpen = (size) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="${size}" height="${size}" focusable="false">`
    + `<rect width="1024" height="1024" fill="${TILE}"/>`;
const eyeHtml = (fill) => `<circle fill="${fill}" cx="354.2" cy="276.1" r="58"/>`;

/** A square SVG of the face (the host crops it round): `own` is this installation's own face, the app icon itself. */
export function faceSvg(seed, size = 36, { own = false } = {}) {
    const px = Math.max(12, Math.min(96, Math.round(Number(size) || 36)));
    if (own) return `${svgOpen(px)}<path fill="${INK}" d="${BODY}"/>${eyeHtml(RED)}</svg>`;
    const face = faceOf(seed), paint = PAINTS[face.paint];
    let out = `${svgOpen(px)}<path fill="${face.body}" d="${BODY}"/>`;
    if (paint.layers.length) {
        out += `<path fill="${INK}" d="${paint.layers.join('')}"/>`
            + `<path fill="${TILE}" fill-rule="evenodd" d="M0 0H1024V1024H0Z${BODY}"/>`;
    }
    return `${out}${eyeHtml(paint.head ? RED : eyeFor(face.body))}</svg>`;
}

/** The face as a decorative picture beside a name: `size` is 20, 28 or 36 px (the places that show one). */
export function faceHtml(seed, size = 36, { own = false } = {}) {
    return `<span class="peer-face" data-size="${Number(size) || 36}" aria-hidden="true">${faceSvg(seed, size, { own })}</span>`;
}

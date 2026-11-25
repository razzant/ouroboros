// A QR code as inline SVG, drawn on this device by the vendored encoder (Project Nayuki QR-Code-generator v1.5.0, see
// web/vendor/VENDOR-MANIFEST.md): no network at runtime. Dark modules on a light square in every theme, so a phone's
// camera reads it, with the four-module quiet zone the standard asks for.
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';

const ENCODER_URL = new URL('../vendor/qrcodegen/qrcodegen.js', import.meta.url).href;
let loading = null;

/** The encoder, loaded once as a classic script (it defines the global `qrcodegen`); a failed load may be retried. */
export function qrEncoder(doc = globalThis.document) {
    if (globalThis.qrcodegen?.QrCode) return Promise.resolve(globalThis.qrcodegen);
    loading ||= new Promise((resolve, reject) => {
        const script = doc.createElement('script');
        script.src = ENCODER_URL;
        script.onload = () => (globalThis.qrcodegen?.QrCode ? resolve(globalThis.qrcodegen) : reject(new Error('the QR encoder did not load')));
        script.onerror = () => { loading = null; reject(new Error('the QR encoder did not load')); };
        doc.head.append(script);
    });
    return loading;
}

/** `text` as an SVG string sized by its container, or "" while the encoder is not loaded. */
export function qrSvg(text, { label = '', border = 4 } = {}) {
    const encoder = globalThis.qrcodegen;
    if (!encoder?.QrCode) return '';
    const code = encoder.QrCode.encodeText(String(text), encoder.QrCode.Ecc.MEDIUM);
    const size = code.size + border * 2;
    const path = [];
    for (let y = 0; y < code.size; y += 1) {
        for (let x = 0; x < code.size; x += 1) if (code.getModule(x, y)) path.push(`M${x + border},${y + border}h1v1h-1z`);
    }
    // Fixed black on white, not theme inks: a scanner needs the contrast whatever the page around it is.
    return `<svg class="qr-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" shape-rendering="crispEdges" role="img" aria-label="${escapeHtml(label)}"><rect width="${size}" height="${size}" fill="#fff"/><path d="${path.join('')}" fill="#000"/></svg>`;
}

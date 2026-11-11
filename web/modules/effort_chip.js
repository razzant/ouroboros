// The child card's effort chip (DECISIONS v3 §5; DESIGN "Child-card anatomy"): one text part
// of the meta line, composed from the frame's three scalars — the APPLIED level, the level
// Ouroboros asked for when it differs, and the source of the decision (a pin, the model
// name, Cyber Pro, or the owner's range). A frame without `effort_level` paints nothing;
// an old frame is never back-filled. The words read through tr()/fmt(): a live card is never
// overlay-translated.
import { fmt } from './i18n.js';
import { effortText } from './effort_levels.js';
import { PIN_BODY } from './page_icons.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

const PIN_SVG = `<svg class="chat-live-effort-pin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PIN_BODY}</svg>`;

/** `{level, requested, source}` from a chat frame, or null when the frame carries no effort level. */
export function effortFact(frame) {
    const level = String(frame?.effort_level || '').trim().toLowerCase();
    if (!level) return null;
    return {
        level,
        requested: String(frame?.effort_requested || '').trim().toLowerCase(),
        source: String(frame?.effort_source || '').trim().toLowerCase(),
    };
}

/** The chip's words: "Effort High", "Effort High (asked Ultra)", a pin's bare "High",
 *  "X-High · model name", "Effort Ultra (Cyber Pro)". */
export function effortChipText(fact) {
    if (!fact?.level) return '';
    const level = effortText(fact.level);
    const asked = fact.requested && fact.requested !== fact.level
        ? fmt(' (asked {requested})', { requested: effortText(fact.requested) }) : '';
    if (fact.source === 'pin') return `${level}${asked}`;
    if (fact.source === 'model_name') return fmt('{level} · model name', { level: `${level}${asked}` });
    if (fact.source === 'cyber') return fmt('Effort {level} (Cyber Pro)', { level: `${level}${asked}` });
    return fmt('Effort {level}', { level: `${level}${asked}` });
}

/** The meta-line part: the words, with the pin glyph before a pinned level. */
export function effortChipMarkup(fact) {
    const text = effortChipText(fact);
    if (!text) return '';
    const pinned = fact.source === 'pin';
    const label = pinned ? ` aria-label="${escapeHtml(fmt('Pinned effort {level}', { level: text }))}"` : '';
    return `<span class="chat-live-meta-text chat-live-effort" data-effort-source="${escapeHtml(fact.source)}"${label}>`
        + `${pinned ? PIN_SVG : ''}${escapeHtml(text)}</span>`;
}

// The mark of the source a model is reached through (the Model capsule's glyph, a suggestion's row). Brand marks are
// monochrome and inherit currentColor. OpenAI's is the harness SSOT's (harness_presentation.js, the same knot as
// web/providers/openai.svg); Anthropic's and OpenRouter's are Simple Icons' (CC0; OpenRouter's from simple-icons
// 16.36.0, sourced from openrouter.ai, the split arrows of web/providers/openrouter.ico); the OpenAI-compatible mark
// is the stroke of web/providers/openai-compatible.svg without its tile. A source without a mark of its own shows
// its initial, never an invented logo.
import { escapeHtmlAttr as esc } from './ui_primitives.js';
import { HARNESS_MARKS, harnessPresentation } from './harness_presentation.js';

const MARKS = Object.freeze({
    openai: { viewBox: HARNESS_MARKS.codex.viewBox, body: `<path fill="currentColor" d="${HARNESS_MARKS.codex.path}"/>` },
    anthropic: { viewBox: '0 0 24 24', body: '<path fill="currentColor" d="M17.3041 3.541h-3.6718l6.696 16.918H24Zm-10.6082 0L0 20.459h3.7442l1.3693-3.5527h7.0052l1.3693 3.5528h3.7442L10.5363 3.5409Zm-.3712 10.2232 2.2914-5.9456 2.2914 5.9456Z"/>' },
    openrouter: { viewBox: '0 0 24 24', body: '<path fill="currentColor" d="M16.778 1.844v1.919q-.569-.026-1.138-.032-.708-.008-1.415.037c-1.93.126-4.023.728-6.149 2.237-2.911 2.066-2.731 1.95-4.14 2.75-.396.223-1.342.574-2.185.798-.841.225-1.753.333-1.751.333v4.229s.768.108 1.61.333c.842.224 1.789.575 2.185.799 1.41.798 1.228.683 4.14 2.75 2.126 1.509 4.22 2.11 6.148 2.236.88.058 1.716.041 2.555.005v1.918l7.222-4.168-7.222-4.17v2.176c-.86.038-1.611.065-2.278.021-1.364-.09-2.417-.357-3.979-1.465-2.244-1.593-2.866-2.027-3.68-2.508.889-.518 1.449-.906 3.822-2.59 1.56-1.109 2.614-1.377 3.978-1.466.667-.044 1.418-.017 2.278.02v2.176L24 6.014Z"/>' },
    'openai-compatible': { viewBox: '0 0 64 64', body: '<path d="M20 18h24a10 10 0 0 1 0 20H32a8 8 0 0 0 0 16h12M18 46h14" fill="none" stroke="currentColor" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round"/>' },
});
const svg = (viewBox, body) => `<svg class="model-source-mark" viewBox="${viewBox}" aria-hidden="true" focusable="false">${body}</svg>`;
const initial = (label) => `<span class="model-source-mark model-source-initial" aria-hidden="true">${esc((String(label || '').trim()[0] || '·').toUpperCase())}</span>`;

/**
 * A source's mark as markup: its harness's (a subscription), its provider's (an API key), else its initial.
 * @param {?{lane?: string, harness?: string, kind?: string, label?: string, sourceLabel?: string}} source
 */
export function sourceMark(source) {
    const label = source?.label || source?.sourceLabel || '';
    if (source?.lane === 'subscription') {
        const mark = harnessPresentation(source.harness);
        return mark.generic ? initial(label) : svg(mark.viewBox, `<path fill="currentColor" d="${mark.path}"/>`);
    }
    const mark = MARKS[source?.kind];
    return mark ? svg(mark.viewBox, mark.body) : initial(label);
}

/** Portable UI primitives shared by first-party forms and optional author pages.
 * No host, network, document-global or framework dependency. Callers own data,
 * validation and lifecycle; renderers escape their own HTML/attribute contexts.
 */

export function escapeHtmlAttr(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/`/g, '&#96;');
}

const TONES = new Set(['ok', 'danger', 'warn', 'muted', 'info']);
const TONE_ALIASES = Object.freeze({
    error: 'danger',
    success: 'ok',
    warning: 'warn',
    neutral: 'muted',
});
const SAFE_FIELD_TYPES = new Set(['text', 'number', 'url', 'email', 'password', 'textarea', 'select', 'checkbox']);

function safeFieldType(value) {
    const type = String(value || 'text').toLowerCase();
    return SAFE_FIELD_TYPES.has(type) ? type : 'text';
}

function safeNumericAttribute(name, value) {
    if (value === '' || value === null || value === undefined || !Number.isFinite(Number(value))) return '';
    return ` ${name}="${escapeHtmlAttr(value)}"`;
}

/** Render the narrow host-owned field contract shared by Widgets and Settings. */
export function renderSafeField(field = {}, savedValues = {}, options = {}) {
    const rawName = String(field.name || '');
    const name = escapeHtmlAttr(rawName);
    const label = escapeHtmlAttr(field.label || rawName);
    const type = safeFieldType(field.type);
    const accessible = ` aria-label="${label}"`
        + (field.help ? ` aria-description="${escapeHtmlAttr(field.help)}"` : '');
    const hasSaved = type !== 'password' && Object.prototype.hasOwnProperty.call(savedValues || {}, rawName);
    const saved = type === 'password' ? '' : (hasSaved ? savedValues[rawName] : field.default);
    const value = escapeHtmlAttr(saved ?? '');
    const placeholder = field.placeholder ? ` placeholder="${escapeHtmlAttr(field.placeholder)}"` : '';
    const required = field.required ? ' required' : '';
    const disabled = field.disabled || options.disabled ? ' disabled' : '';
    const fieldClass = escapeHtmlAttr(`ui-field ${options.fieldClass || 'widget-field'}`);
    const inlineClass = escapeHtmlAttr(`ui-field ui-field-inline ${options.inlineClass || `${options.fieldClass || 'widget-field'} widget-field-inline`}`);
    const helpClass = escapeHtmlAttr(`ui-field-help ${options.helpClass || 'widget-field-help'}`);
    const maxSpan = Math.max(1, Math.min(4, Number(options.maxSpan) || 4));
    const span = Math.max(1, Math.min(maxSpan, Number(field.span) || 1));
    const spanClass = options.spanClassPrefix ? ` ${escapeHtmlAttr(options.spanClassPrefix)}${span}` : '';
    const help = field.help ? `<small class="${helpClass}">${escapeHtmlAttr(field.help)}</small>` : '';
    if (type === 'textarea' && field.disabled) {
        // A multiline value the schema declares read-only (a status) is text, never an inert control: whole at any width,
        // selectable and reachable by keyboard, in the value's ink. `<output>` keeps its name in form.elements and its
        // text as `.value`, so collecting the form and refreshing the field in place work as for the control.
        return `<label class="${fieldClass}${spanClass}"><span>${label}</span><output class="ui-control ui-control-readonly" name="${name}"${accessible} tabindex="0">${value}</output>${help}</label>`;
    }
    if (type === 'textarea') {
        return `<label class="${fieldClass}${spanClass}"><span>${label}</span><textarea class="ui-control" name="${name}"${accessible}${placeholder}${required}${disabled}>${value}</textarea>${help}</label>`;
    }
    if (type === 'select') {
        const optionsHtml = (Array.isArray(field.options) ? field.options : []).map((option) => {
            const optionValue = typeof option === 'object' && option !== null ? option.value : option;
            const optionLabel = typeof option === 'object' && option !== null ? (option.label ?? option.value) : option;
            const selected = String(optionValue ?? '') === String(saved ?? '') ? ' selected' : '';
            return `<option value="${escapeHtmlAttr(optionValue ?? '')}"${selected}>${escapeHtmlAttr(optionLabel ?? '')}</option>`;
        }).join('');
        return `<label class="${fieldClass}${spanClass}"><span>${label}</span><select class="ui-control" name="${name}"${accessible}${required}${disabled}>${optionsHtml}</select>${help}</label>`;
    }
    if (type === 'checkbox') {
        return `<label class="${inlineClass}${spanClass}"><input class="ui-checkbox" type="checkbox" name="${name}"${accessible}${saved ? ' checked' : ''}${required}${disabled}> <span>${label}</span>${help}</label>`;
    }
    const numeric = type === 'number'
        ? `${safeNumericAttribute('min', field.min)}${safeNumericAttribute('max', field.max)}${safeNumericAttribute('step', field.step)}`
        : '';
    const autocomplete = type === 'password' ? ' autocomplete="new-password"' : '';
    return `<label class="${fieldClass}${spanClass}"><span>${label}</span><input class="ui-control" type="${type}" name="${name}"${accessible} value="${value}"${placeholder}${numeric}${required}${disabled}${autocomplete}>${help}</label>`;
}

/** Collect values according to the same closed field contract used for rendering. */
export function collectSafeFieldValues(form, fields = [], { includePasswords = true } = {}) {
    const values = {};
    for (const field of Array.isArray(fields) ? fields : []) {
        const name = String(field?.name || '');
        const type = safeFieldType(field?.type);
        if (!name || (type === 'password' && !includePasswords)) continue;
        const input = form?.elements?.namedItem
            ? form.elements.namedItem(name)
            : form?.elements?.[name];
        if (!input) continue;
        values[name] = type === 'checkbox' ? Boolean(input.checked) : input.value;
    }
    return values;
}

export function normalizeTone(tone = 'muted', fallback = 'muted') {
    const canonical = (value) => {
        const clean = String(value || '').toLowerCase();
        const normalized = TONE_ALIASES[clean] || clean;
        return TONES.has(normalized) ? normalized : '';
    };
    return canonical(tone) || canonical(fallback) || 'muted';
}

export function setInlineStatus(el, text, tone = 'muted') {
    if (!el) return;
    const next = text || '';
    if (el.textContent !== next) el.textContent = next;
    el.dataset.tone = normalizeTone(tone);
}

/* ---- The result of an action (DESIGN "Status and chips"; ui.css .ui-action-row / .ui-result) -----------------------
   The caller runs the action and owns its words; these helpers only write the result line and hold its lifetime. A
   plain acknowledgement may end by itself after SAVE_ACK_MS; a warning, a needed restart or a failure stays until the
   next result or an edit in the same subject replaces it, so `ends` is honoured only for an ok or neutral tone. One
   timer owner per node: an older acknowledgement never erases a newer result. Nothing here runs a click. */

export const SAVE_ACK_MS = 4000;
const RESULT_TIMERS = new WeakMap();
const ENDING_TONES = new Set(['ok', 'muted']);

function stopResultTimer(node) {
    const timer = RESULT_TIMERS.get(node);
    if (timer !== undefined) {
        clearTimeout(timer);
        RESULT_TIMERS.delete(node);
    }
}

/** Write a result into `node`: its words, its tone, the raw text behind one Details disclosure; `ends` lets a plain
 * acknowledgement clear itself after SAVE_ACK_MS. An empty result clears the node. */
export function setResult(node, { text = '', tone = 'muted', details = '', ends = false } = {}) {
    if (!node) return;
    stopResultTimer(node);
    const words = String(text ?? '');
    const raw = String(details ?? '');
    node.textContent = words;
    if (raw) {
        const doc = node.ownerDocument;
        const disclosure = doc.createElement('details');
        disclosure.className = 'ui-disclosure';
        const summary = doc.createElement('summary');
        summary.textContent = 'Details';
        const body = doc.createElement('pre');
        body.className = 'ui-result-details';
        body.textContent = raw;
        disclosure.append(summary, body);
        node.append(disclosure);
    }
    if (!words && !raw) {
        delete node.dataset.tone;
        return;
    }
    const normalized = normalizeTone(tone);
    node.dataset.tone = normalized;
    if (ends && ENDING_TONES.has(normalized)) {
        const timer = setTimeout(() => {
            if (RESULT_TIMERS.get(node) !== timer) return;
            RESULT_TIMERS.delete(node);
            node.textContent = '';
            delete node.dataset.tone;
        }, SAVE_ACK_MS);
        RESULT_TIMERS.set(node, timer);
    }
}

/** An edit in the same subject (the form or section the action belongs to) clears its result; returns the disposer. */
export function clearResultOnEdit(subject, node) {
    if (!subject?.addEventListener || !node) return () => {};
    const clear = (event) => {
        const target = event?.target;
        if (target && (target === node || node.contains?.(target))) return;
        setResult(node, {});
    };
    subject.addEventListener('input', clear);
    subject.addEventListener('change', clear);
    return () => {
        subject.removeEventListener('input', clear);
        subject.removeEventListener('change', clear);
    };
}

function rawErrorText(err) {
    const body = err?.body;
    if (body && typeof body === 'object') {
        const said = body.error ?? body.detail ?? body.message;
        const line = [body.reason_code, said && typeof said === 'object' ? JSON.stringify(said) : said].filter(Boolean).join(': ');
        if (line) return line;
        try { return JSON.stringify(body); } catch { /* an unserializable body is just an error */ }
    }
    if (typeof body === 'string' && body) return body;
    return String(err?.message || err || '');
}

// The sentence the server wrote for a person in its answer, read where fetchJson (api_client.js) reads an error's
// text: the body's `error` (a string, or a structured `{message|detail}`), then its `message`. A bare code
// (`card_stale`), a line that only repeats the answer's own code (`continue refused: queue_not_restored`) and the
// placeholder fetchJson writes for a body it could not read are not that sentence. Shown on its own, the sentence
// starts with a capital and ends with one stop.
const BARE_CODE = /^[a-z][a-z0-9]*(?:[_.:-][a-z0-9]+)*$/;
function serverSentence(body) {
    if (!body || typeof body !== 'object') return '';
    const codes = [body.reason_code, body.error_code, body.code].filter((code) => typeof code === 'string' && code.length > 2);
    const said = body.error && typeof body.error === 'object' ? [body.error.message, body.error.detail] : [body.error];
    for (const value of [...said, body.message]) {
        const text = typeof value === 'string' ? value.trim() : '';
        if (!text || BARE_CODE.test(text) || codes.some((code) => text.includes(code))
            || /^non-json response \(HTTP \d+\)$/.test(text)) continue;
        const capital = text.charAt(0).toUpperCase() + text.slice(1);
        return /[.!?…]$/.test(capital) ? capital : `${capital}.`;
    }
    return '';
}

/** Owner words for a failed action: when Ouroboros answered with a reason for a person (its body's sentence), that
 * reason; generic words, keyed on typed fields only (a TypeError without a response, `err.status`,
 * `err.body.reason_code`), when it could not be reached or answered without one; never words read out of an error's
 * own message. The raw text comes back as `details`. */
export function errorWords(err) {
    const status = Number(err?.status);
    const code = err?.body?.reason_code;
    const said = serverSentence(err?.body);
    let text;
    if (err instanceof TypeError && err.response === undefined) text = "Couldn't reach Ouroboros.";
    else if (said) text = said;
    else if (status >= 500) text = "Ouroboros couldn't do this.";
    else if (code) text = 'Ouroboros refused this.';
    else if (status === 401 || status === 403) text = 'Not allowed here.';
    else if (status === 404) text = "Ouroboros couldn't find this.";
    else if (status >= 400) text = "Ouroboros didn't accept this.";
    else text = "Didn't work.";
    return { text, tone: 'danger', details: rawErrorText(err) };
}

import { normalizeTone } from './ui_helpers.js';

function getStack() {
    let stack = document.getElementById('toast-stack');
    if (!stack) {
        stack = document.createElement('div');
        stack.id = 'toast-stack';
        stack.className = 'toast-stack';
        stack.setAttribute('aria-live', 'polite');
        stack.setAttribute('aria-relevant', 'additions');
        document.body.appendChild(stack);
    }
    return stack;
}

// Tone needs a channel that is not colour. The tinted border is a 0.15-alpha
// edge, so ok / warn / danger also carry a leading glyph while the message text
// itself stays at --text-primary (see the `.toast-*` rules in web/style.css).
// The glyph is DECORATION for assistive tech — danger is already announced
// through role="alert" — so it is aria-hidden rather than read as punctuation.
const TONE_GLYPHS = { ok: '✓', warn: '!', danger: '✕' };

export function showToast(message, tone = 'info', { ttl = 6000 } = {}) {
    const stack = getStack();
    const toast = document.createElement('div');
    const cleanTone = normalizeTone(tone || 'info', 'info');
    toast.className = `toast toast-${cleanTone}`;
    toast.setAttribute('role', cleanTone === 'danger' ? 'alert' : 'status');
    const glyph = TONE_GLYPHS[cleanTone];
    if (glyph) {
        const mark = document.createElement('span');
        mark.className = 'toast-glyph';
        mark.setAttribute('aria-hidden', 'true');
        mark.textContent = glyph;
        toast.appendChild(mark);
    }
    const text = document.createElement('span');
    text.className = 'toast-text';
    text.textContent = message || '';
    toast.appendChild(text);
    stack.appendChild(toast);
    const dismiss = () => {
        toast.classList.add('is-hiding');
        setTimeout(() => toast.remove(), 180);
    };
    if (ttl > 0) setTimeout(dismiss, ttl);
    toast.addEventListener('click', dismiss);
    return toast;
}

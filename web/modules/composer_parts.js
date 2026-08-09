/**
 * Ordered composer parts + the reversible `[context:]` marker codec.
 *
 * ONE module owns the context-capture grammar so every producer (chat composer,
 * Changes dock, Files dock) and every consumer (chat rendering, history replay)
 * read the same bytes. The marker is a TEXT convention: nothing about the chat
 * transport, contracts.py, or the agent's system prompt changes. The agent sees
 * self-describing natural language it can act on with the tools it already has.
 *
 * Grammar
 * -------
 *   [context: <path> L<start>-L<end>]
 *   ```
 *   <the selected lines, verbatim>
 *   ```
 *   [context: <path>]
 *
 * A selection of at most MAX_CHIP_LINES lines is inlined VERBATIM in a fenced
 * block right after its marker: the agent has the exact bytes with zero extra
 * tool rounds, and the marker names the exact referent. The fence length is the
 * longest backtick run in the content plus one (minimum 3), so content that
 * itself contains fences can never terminate the block early. A selection over
 * the cap, or a whole-file chip, serializes as the bare `[context: <path>]`
 * marker and the agent reads the file itself.
 *
 * Reversibility is the invariant: `parseContent(serializeParts(parts))`
 * re-serializes to the identical string. A path that cannot round-trip (it
 * contains a newline or `]`, or is empty) is REFUSED at capture time —
 * `makeChipPart` returns null — rather than silently producing a marker that
 * parses back as prose. Anything that is not an exactly-formed marker line
 * (`[context:foo]`, leading spaces, trailing text) stays plain text.
 *
 * Producer contract (`makeChipPart` content)
 * ------------------------------------------
 * `content` is the selected text VERBATIM, LF-separated with no trailing newline;
 * CRLF and ONE trailing newline — what a raw editor selection or file slice
 * usually hands over — are normalized HERE so no producer repeats that
 * arithmetic. The range and the bytes must not CONTRADICT each other, which is a
 * FLOOR, not an equality: the fence must carry at least as many lines as the
 * range names.
 *
 * A file capture meets it exactly (`slice(start-1, end)` of the line list). A
 * DIFF capture legitimately carries MORE: decision 9 inlines the selected diff
 * lines verbatim, and an interleaved `-` line is real selected text that the
 * new-side range cannot number. Both are honest, so both are allowed — while
 * content that spans FEWER lines than the range names is dropped, because there
 * the range would overclaim what the fence can show.
 *
 * What makes the FLOOR safe is that `chipLabel` counts the bytes the chip really
 * INLINES, never the range: a chip folds its fence away behind "N lines" that is
 * always provable from the payload it hides, so no extra line can ride along
 * unannounced. Over the cap there IS no fence to hide, so the label mirrors the
 * serializer and counts the range — the one place where bytes are not the answer.
 *
 * This module is pure with respect to the network: no fetch, no transport, no
 * send logic. `createComposerParts` is a thin DOM mount over the same core.
 */

import { escapeHtml, escapeHtmlAttr } from './utils.js';

/** Selections longer than this are handed to the agent as a bare marker. */
export const MAX_CHIP_LINES = 200;

const MARKER_RE = /^\[context: ([^\]\n]+?)(?: L(\d+)-L(\d+))?\]$/;
const FENCE_RE = /^(`{3,})$/;

function isPositiveInt(value) {
    return Number.isInteger(value) && value > 0;
}

/** A path is representable only if the marker grammar can round-trip it. */
export function chipPathIsRepresentable(path) {
    const raw = typeof path === 'string' ? path : '';
    if (!raw.trim()) return false;
    if (raw !== raw.trim()) return false;
    if (/[\n\r\]]/.test(raw)) return false;
    // A path whose own tail looks like the line-range suffix would parse back as
    // a different (path, range) pair — refuse it rather than mislabel the bytes.
    return !/ L\d+-L\d+$/.test(raw);
}

/** Line count of chip bytes — the ONE place a fence's size is measured. */
export function chipContentLines(content) {
    return typeof content === 'string' && content !== '' ? content.split('\n').length : 0;
}

/**
 * Hold a chip's shape to the grammar, keeping only what round-trips. Shared by
 * `makeChipPart` (which normalizes bytes first) and `normalizeParts` (which must
 * not touch bytes, so a hand-written fence still re-serializes verbatim), so a
 * pre-built chip object cannot smuggle a range past the same checks.
 *
 * An unusable range (inverted, zero, fractional, absent) does not discard the
 * capture: the chip degrades to the whole-file form, which is still true. Bytes
 * go with it, because only a range can locate them.
 */
function sanitizeChip(chip) {
    if (!chipPathIsRepresentable(chip?.path)) return null;
    const start = isPositiveInt(chip.lineStart) ? chip.lineStart : null;
    const end = isPositiveInt(chip.lineEnd) ? chip.lineEnd : null;
    const out = { type: 'chip', path: chip.path };
    if (start === null || end === null || end < start) return out;
    out.lineStart = start;
    out.lineEnd = end;
    // The FLOOR (see the producer contract above): a fence may carry more lines
    // than the range names (a diff selection's `-` rows), never fewer.
    if (chipContentLines(chip.content) >= end - start + 1) out.content = chip.content;
    return out;
}

/**
 * Build a chip part, or return null when the path cannot round-trip. Callers
 * disclose the refusal to the owner instead of capturing a lossy marker.
 *
 * `content` is normalized to the producer contract documented at the top of this
 * module — CRLF becomes LF, ONE trailing newline is stripped — and then held to
 * the same floor as any other chip.
 */
export function makeChipPart({ path, lineStart = null, lineEnd = null, content = null } = {}) {
    const bytes = typeof content === 'string' && content !== ''
        ? content.replace(/\r\n/g, '\n').replace(/\n$/, '')
        : null;
    return sanitizeChip({ type: 'chip', path, lineStart, lineEnd, content: bytes });
}

export function makeTextPart(text) {
    const value = typeof text === 'string' ? text : '';
    return value ? { type: 'text', text: value } : null;
}

/**
 * Human label for a chip: `name · N lines`, or `name` for a whole file.
 *
 * N counts whatever the chip actually HIDES, which means mirroring
 * `serializeChip`'s cap decision rather than the raw byte count:
 *
 *   • bytes that INLINE (a range, and at most `MAX_CHIP_LINES` of them) are what
 *     the fold conceals, so they are what the label counts — a diff chip whose
 *     2-line range inlines 3 verbatim rows says "3 lines", never the range's 2;
 *   • OVER the cap the serializer drops the fence and keeps the range, so the label
 *     counts the RANGE too. Counting the dropped bytes there would put the dock
 *     label at odds with the wire and with the transcript (which re-reads the bare
 *     marker and has only the range) for exactly the >200-row Changes selection
 *     that reaches this path;
 *   • with no usable range nothing can locate bytes, the serializer emits the
 *     whole-file marker, and the honest label is the bare name.
 */
export function chipLabel(chip) {
    const path = String(chip?.path || '');
    const name = path.split('/').filter(Boolean).pop() || path;
    const start = Number(chip?.lineStart);
    const end = Number(chip?.lineEnd);
    const hasRange = Number.isFinite(start) && Number.isFinite(end);
    const bytes = chipContentLines(chip?.content);
    if (hasRange && bytes > 0 && bytes <= MAX_CHIP_LINES) {
        return `${name} · ${bytes} line${bytes === 1 ? '' : 's'}`;
    }
    if (!hasRange) return name;
    const count = Math.max(1, end - start + 1);
    return `${name} · ${count} line${count === 1 ? '' : 's'}`;
}

// ---------------------------------------------------------------------------
// Parts reducer (pure; every op returns a NEW list)
// ---------------------------------------------------------------------------

/**
 * Adjacent text parts are merged, because the serialized form cannot tell them
 * apart — keeping the list normalized is what makes the codec reversible.
 */
export function normalizeParts(parts) {
    const out = [];
    for (const part of Array.isArray(parts) ? parts : []) {
        if (!part || (part.type !== 'text' && part.type !== 'chip')) continue;
        if (part.type === 'chip') {
            // Every chip goes through the same gate, however it was built: a
            // hand-assembled `{type:'chip', lineStart:5, lineEnd:2}` would
            // otherwise serialize as the un-parseable claim `L5-L2`.
            const chip = sanitizeChip(part);
            if (chip) out.push(chip);
            continue;
        }
        const text = typeof part.text === 'string' ? part.text : '';
        if (!text) continue;
        const last = out[out.length - 1];
        if (last && last.type === 'text') out[out.length - 1] = { type: 'text', text: `${last.text}\n${text}` };
        else out.push({ type: 'text', text });
    }
    return out;
}

export function pushText(parts, text) {
    const part = makeTextPart(text);
    if (!part) return normalizeParts(parts);
    return normalizeParts([...(parts || []), part]);
}

export function pushChip(parts, chip) {
    const part = chip && chip.type === 'chip' ? chip : makeChipPart(chip || {});
    if (!part) return normalizeParts(parts);
    return normalizeParts([...(parts || []), part]);
}

/** Backspace-in-empty-input semantics: drop the trailing part. */
export function popLast(parts) {
    const list = normalizeParts(parts);
    list.pop();
    return list;
}

export function clearParts() {
    return [];
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------

function longestBacktickRun(text) {
    let best = 0;
    for (const match of String(text).matchAll(/`+/g)) {
        if (match[0].length > best) best = match[0].length;
    }
    return best;
}

export function fenceFor(content) {
    return '`'.repeat(Math.max(3, longestBacktickRun(content) + 1));
}

function serializeChip(chip) {
    const path = String(chip.path);
    const hasRange = isPositiveInt(chip.lineStart) && isPositiveInt(chip.lineEnd);
    const content = hasRange && typeof chip.content === 'string' && chip.content !== '' ? chip.content : null;
    if (content !== null) {
        const lineCount = content.split('\n').length;
        if (hasRange && lineCount <= MAX_CHIP_LINES) {
            const fence = fenceFor(content);
            return `[context: ${path} L${chip.lineStart}-L${chip.lineEnd}]\n${fence}\n${content}\n${fence}`;
        }
        // Over the inline cap: keep the (true) range, drop the bytes — the agent
        // reads exactly that span itself instead of getting a truncated excerpt.
        return `[context: ${path} L${chip.lineStart}-L${chip.lineEnd}]`;
    }
    if (hasRange) return `[context: ${path} L${chip.lineStart}-L${chip.lineEnd}]`;
    return `[context: ${path}]`;
}

/** Ordered parts -> the exact content string that is sent, stored and replayed. */
export function serializeParts(parts) {
    return normalizeParts(parts)
        .map((part) => (part.type === 'chip' ? serializeChip(part) : part.text))
        .join('\n');
}

/**
 * The exact inverse of `serializeParts`. Unrecognized or malformed marker-like
 * lines are returned as plain text, so prose that merely resembles a marker
 * survives untouched.
 */
export function parseContent(text) {
    const raw = typeof text === 'string' ? text : '';
    if (!raw) return [];
    const lines = raw.split('\n');
    const parts = [];
    let pending = [];
    const flushText = () => {
        if (!pending.length) return;
        const joined = pending.join('\n');
        pending = [];
        if (joined) parts.push({ type: 'text', text: joined });
    };
    for (let i = 0; i < lines.length; i += 1) {
        const match = MARKER_RE.exec(lines[i]);
        if (!match || !chipPathIsRepresentable(match[1])) {
            pending.push(lines[i]);
            continue;
        }
        const chip = makeChipPart({
            path: match[1],
            lineStart: match[2] ? Number(match[2]) : null,
            lineEnd: match[3] ? Number(match[3]) : null,
        });
        if (!chip) {
            pending.push(lines[i]);
            continue;
        }
        // A fenced block on the NEXT line belongs to this marker when it closes.
        // Only a RANGED marker can own inlined bytes (that is the only form the
        // serializer emits a fence for), so a fence after a whole-file marker
        // stays ordinary text and the string still round-trips.
        const fence = chip.lineStart ? FENCE_RE.exec(lines[i + 1] || '') : null;
        if (fence) {
            let close = -1;
            for (let j = i + 2; j < lines.length; j += 1) {
                if (lines[j] === fence[1]) { close = j; break; }
            }
            if (close > i + 1) {
                chip.content = lines.slice(i + 2, close).join('\n');
                i = close;
            }
        }
        flushText();
        parts.push(chip);
    }
    flushText();
    return normalizeParts(parts);
}

// ---------------------------------------------------------------------------
// Thin DOM mount (no transport, no send logic)
// ---------------------------------------------------------------------------

/**
 * Render `parts` as inline chips before a live input inside `container`.
 *
 * @param {object} options
 * @param {HTMLElement} options.container host element (gets `.composer-parts`)
 * @param {HTMLElement} options.input     the live input/textarea, kept last
 * @param {Function} [options.onChange]   called with the new parts list
 */
export function createComposerParts({ container, input, onChange = null } = {}) {
    if (!container || !input) throw new Error('createComposerParts needs container + input');
    let parts = [];
    container.classList.add('composer-parts', 'composer-parts-host');

    const emit = () => { if (typeof onChange === 'function') onChange(getParts()); };

    function paint() {
        container.querySelectorAll('[data-composer-part]').forEach((node) => node.remove());
        parts.forEach((part, index) => {
            const node = document.createElement('span');
            node.dataset.composerPart = String(index);
            if (part.type === 'chip') {
                const label = chipLabel(part);
                node.className = 'composer-part-chip';
                node.title = part.path;
                node.innerHTML = `<span class="composer-part-chip-label">${escapeHtml(label)}</span>`
                    + `<button type="button" class="composer-part-remove"`
                    + ` data-composer-part-remove="${escapeHtmlAttr(String(index))}"`
                    + ` title="Remove ${escapeHtmlAttr(label)}"`
                    + ` aria-label="Remove ${escapeHtmlAttr(label)}">×</button>`;
            } else {
                node.className = 'composer-part-text';
                node.textContent = part.text;
            }
            container.insertBefore(node, input);
        });
    }

    function setParts(next) {
        parts = normalizeParts(next);
        paint();
        return getParts();
    }

    function getParts() {
        return parts.map((part) => ({ ...part }));
    }

    /** The typed draft becomes a text part, so chip/comment order is preserved. */
    function commitDraft() {
        const draft = input.value;
        if (!draft) return getParts();
        input.value = '';
        parts = pushText(parts, draft);
        paint();
        return getParts();
    }

    function addChip(chip) {
        commitDraft();
        parts = pushChip(parts, chip);
        paint();
        emit();
        focus();
        return getParts();
    }

    function clear() {
        parts = clearParts();
        input.value = '';
        paint();
        emit();
        return getParts();
    }

    function focus() {
        try { input.focus(); } catch {}
    }

    const onRemoveClick = (event) => {
        const button = event.target.closest?.('[data-composer-part-remove]');
        if (!button || !container.contains(button)) return;
        const index = Number(button.getAttribute('data-composer-part-remove'));
        if (!Number.isInteger(index) || index < 0 || index >= parts.length) return;
        parts = normalizeParts(parts.filter((_, i) => i !== index));
        paint();
        emit();
        focus();
    };

    const onKeyDown = (event) => {
        if (event.key !== 'Backspace') return;
        if (input.value !== '' || !parts.length) return;
        event.preventDefault();
        parts = popLast(parts);
        paint();
        emit();
    };

    container.addEventListener('click', onRemoveClick);
    input.addEventListener('keydown', onKeyDown);

    return {
        getParts,
        setParts,
        addChip,
        commitDraft,
        clear,
        focus,
        /** Everything currently in the field, typed draft included. */
        serialize() {
            const draft = input.value;
            const all = draft ? pushText(parts, draft) : parts;
            return serializeParts(all);
        },
        destroy() {
            container.removeEventListener('click', onRemoveClick);
            input.removeEventListener('keydown', onKeyDown);
            container.querySelectorAll('[data-composer-part]').forEach((node) => node.remove());
            container.classList.remove('composer-parts', 'composer-parts-host');
        },
    };
}

// Height of the Main/Project message field (DESIGN "message field"): it grows with
// its wrapped text, or an empty field's placeholder, up to the stylesheet's one
// max-height, then scrolls, and shrinks again. `data-sizing` is the one selected
// mode, read by CSS and script alike:
//   content — the engine sizes the field (`field-sizing: content`); no height writes.
//   fit / capped — otherwise: the measured height in `--chat-input-height`, below the
//     cap (`fit`, overflow hidden, because scrollHeight is a rounded integer and must
//     not leave a sub-pixel scrollbar) or at it (`capped`, the text scrolls).

// Content sizing is used only where it sizes a placeholder as it sizes the same text,
// padding and max-height included. WebKit 26 does neither for a wrapped placeholder
// (its last line is cut, a long one passes the cap), so it measures instead.
const placeholderSizing = new WeakMap();

function sizesPlaceholderAsText(doc) {
    if (!placeholderSizing.has(doc)) {
        const probe = doc.createElement('textarea');
        // A hidden probe, one word per line, removed before anything paints.
        probe.style.cssText = 'position:absolute;visibility:hidden;field-sizing:content;box-sizing:border-box;'
            + 'width:0;min-height:0;max-height:none;padding:2px 0;border:0;font:10px/10px sans-serif';
        probe.placeholder = 'a a';
        doc.body.append(probe);
        const placeholder = probe.offsetHeight;
        probe.style.maxHeight = '5px';
        const capped = probe.offsetHeight;
        probe.style.maxHeight = 'none';
        probe.value = probe.placeholder;
        placeholderSizing.set(doc, placeholder === probe.offsetHeight && capped === 5);
        probe.remove();
    }
    return placeholderSizing.get(doc);
}

export function chatInputSizingMode(css = globalThis.CSS, doc = globalThis.document) {
    return css?.supports?.('field-sizing', 'content') && doc?.body && sizesPlaceholderAsText(doc)
        ? 'content' : 'measured';
}

const px = (value) => parseFloat(value) || 0;

function caretViewport(input) {
    const { ownerDocument, value, selectionStart, selectionEnd } = input;
    if (ownerDocument?.activeElement !== input || !value || selectionStart !== selectionEnd || !input.clientWidth) return null;
    const used = px(getComputedStyle(input).width);
    // clientWidth excludes the scrollbar; recover the fractional part of the used width for wrapping.
    return { value, at: selectionEnd, width: input.clientWidth + used - Math.round(used), height: input.clientHeight,
        top: input.scrollTop, scrollHeight: input.scrollHeight };
}

const samePlace = (before, now) => before && now && before.value === now.value && before.at === now.at
    && Math.abs(now.top - before.top) <= 1;

// A textarea exposes selection offsets, not their line boxes. Lay out just text, only for a middle caret on a shrink;
// the tail has its offset in scrollHeight. Copy wrap metrics, not the field, its styling machinery or its listeners.
function caretLineTop(input, at, width, style, line) {
    const doc = input.ownerDocument;
    const probe = doc.createElement('div');
    probe.style.cssText = 'position:fixed;top:0;left:0;visibility:hidden;pointer-events:none;box-sizing:content-box;';
    probe.style.width = `${width - px(style.paddingLeft) - px(style.paddingRight)}px`;
    for (const key of ['direction', 'fontFamily', 'fontSize', 'fontStyle', 'fontWeight', 'letterSpacing', 'lineHeight',
        'overflowWrap', 'tabSize', 'textIndent', 'whiteSpace', 'wordBreak', 'wordSpacing']) probe.style[key] = style[key];
    const text = doc.createTextNode(input.value);
    probe.append(text);
    doc.body.append(probe);
    try {
        const range = doc.createRange();
        range.setStart(text, at);
        range.collapse(true);
        const y = range.getBoundingClientRect().top - probe.getBoundingClientRect().top;
        return px(style.paddingTop) + Math.round(y / line) * line;
    } finally {
        probe.remove();
    }
}

function keepVisibleCaret(input, before, now) {
    if (!samePlace(before, now)) return;
    const { height, width, top, scrollHeight } = now;
    if (!height || (height >= before.height && width >= before.width && scrollHeight <= before.scrollHeight)) return;
    const style = getComputedStyle(input), pad = px(style.paddingBottom);
    if (before.at === input.value.length) {
        // Pin the tail only if it was visible; reading higher in a long draft leaves the same caret offset.
        if (before.scrollHeight - before.top - before.height <= pad + 1
            && scrollHeight - top - height > pad + 1) input.scrollTop = scrollHeight;
        return;
    }
    const line = px(style.lineHeight);
    const y = caretLineTop(input, before.at, width, style, line);
    const oldY = width === before.width ? y : caretLineTop(input, before.at, before.width, style, line);
    if (oldY < before.top - 1 || oldY + line > before.top + before.height + 1) return;
    if (y < top - 1) input.scrollTop = y - px(style.paddingTop);
    else if (y + line > top + height + 1) input.scrollTop = Math.ceil(y + line + pad - height);
}

export function attachChatInputSizing(input, { followCaret = false } = {}) {
    const mode = chatInputSizingMode();
    const measured = mode !== 'content';
    let disposed = false;
    // The width the last measurement wrapped at, unrounded: clientWidth is whole pixels,
    // so a sub-pixel change (a window resized at 125%) could rewrap unseen. 0: unmeasured.
    let width = 0;
    // Something other than the width changed the wrapping: a loaded font, a placeholder.
    let stale = false;
    // The max-height the last measurement capped at: a phone keyboard moves it (`keyboard_viewport.js`, `--chat-viewport-input-cap`).
    let capAt = '';
    let queued = false;
    let frame = 0;
    // The last field geometry, after any correction. A later CSS wrap can hide the same caret without shrinking the box.
    let caretLayout = null;
    const readCaret = () => followCaret ? caretViewport(input) : null;
    input.dataset.sizing = measured ? 'fit' : 'content';

    // Where an edit leaves the text: a caret at the end keeps the last line in view, an
    // empty field shows its placeholder from the top, any other caret stays where it was.
    const scrollAfterEdit = (previous) => !input.value ? 0
        : input.selectionEnd >= input.value.length - 1 ? input.scrollHeight : previous;
    const usedWidth = () => px(getComputedStyle(input).width);

    function measure(edited) {
        // Hidden (a Project panel is built before it is shown): no width to wrap at,
        // so nothing is pinned now and the reveal measures.
        if (!(input.clientWidth > 0)) { width = 0; return; }
        stale = false;
        const style = getComputedStyle(input);
        capAt = style.maxHeight;
        const cap = parseFloat(capAt);  // NaN for `none`: never capped
        const box = style.boxSizing === 'border-box'
            ? px(style.borderTopWidth) + px(style.borderBottomWidth)
            : -(px(style.paddingTop) + px(style.paddingBottom));
        const scrollTop = input.scrollTop;
        // Measured at its natural one-row height without a scrollbar, which would
        // narrow the text it measures. The row around it keeps its height meanwhile:
        // where the composer is in flow (keyboard up outside Telegram), a one-row field
        // for that instant would lengthen the feed, and WebKit clamps the feed's
        // scroll to that length, leaving the newest message under the field.
        const holder = input.parentElement;
        const held = holder?.style.getPropertyValue('min-height') ?? '';
        if (holder) holder.style.setProperty('min-height', `${holder.offsetHeight}px`);
        input.dataset.sizing = 'fit';
        input.style.removeProperty('--chat-input-height');
        const needed = input.scrollHeight + box;
        const capped = needed > cap;
        input.style.setProperty('--chat-input-height', `${capped ? cap : needed}px`);
        input.dataset.sizing = capped ? 'capped' : 'fit';
        if (holder) {
            if (held) holder.style.setProperty('min-height', held);
            else holder.style.removeProperty('min-height');
        }
        input.scrollTop = !capped ? 0 : edited ? scrollAfterEdit(scrollTop) : scrollTop;
        width = usedWidth();
    }

    // Without an edit, only a changed wrapping width (a resize, the busy Send label, a
    // reveal), a loaded font, a new placeholder or a moved cap (a viewport resize, the body's
    // keyboard-open class) re-measures: the height this module writes reports back and is
    // ignored. The write waits for the next frame, because inside the observer it would
    // resize the composer that chat.js observes higher in the same tree.
    const remeasure = () => {
        if (queued) return;
        queued = true;
        frame = requestAnimationFrame(() => {
            queued = false;
            if (disposed) return;
            const style = measured ? getComputedStyle(input) : null;
            if (style && (stale || px(style.width) !== width || style.maxHeight !== capAt)) measure(false);
            const now = readCaret();
            keepVisibleCaret(input, caretLayout, now);
            if (now) now.top = input.scrollTop;
            caretLayout = now;
        });
    };
    const markStale = () => {
        if (disposed) return;
        stale = true;
        remeasure();
    };
    // `loading` starts a loading period in every engine (WebKit sends no `loadingdone`);
    // `ready` settles once its faces are in.
    const fonts = measured ? globalThis.document?.fonts : null;
    const fontsLoading = () => fonts.ready.then(markStale);
    let observer = null;
    let placeholder = null;
    let keyboard = null;
    if ((measured || followCaret) && typeof ResizeObserver === 'function') {
        observer = new ResizeObserver(remeasure);
        observer.observe(input);
    }
    // A placeholder rewritten in place: the interface language's dictionary arriving.
    if (measured && typeof MutationObserver === 'function') {
        placeholder = new MutationObserver(markStale);
        placeholder.observe(input, { attributes: true, attributeFilter: ['placeholder'] });
        // A class change can restore the resting cap without resizing the field. Telegram keeps its typing cap
        // through keyboard-closing; a capped draft's written height will not itself notify the ResizeObserver.
        const body = input.ownerDocument?.body || globalThis.document?.body;
        if (body) {
            keyboard = new MutationObserver(remeasure);
            keyboard.observe(body, { attributes: true, attributeFilter: ['class'] });
        }
    }
    // A viewport resize can move the cap without changing the width; one that moves neither writes nothing.
    const viewports = measured ? [globalThis.window?.visualViewport, globalThis.window].filter(Boolean) : [];
    for (const target of viewports) target.addEventListener?.('resize', remeasure);
    fonts?.addEventListener('loading', fontsLoading);
    // A loading period already running sent its `loading` before this listener.
    if (fonts?.status === 'loading') fontsLoading();
    // The first measurement waits for no report, which can be late or missing for a field already laid out at
    // attach (on Linux WebKit, Main's field stayed at its unmeasured 33.2 px until the model strip opened). It is
    // asked after the caller's synchronous build (a microtask), so the width it reads is the built page's and a
    // stand that runs frames at once does not re-enter the caller, and only where frames exist and the field is
    // laid out then: a hidden one (or a host without either) measures on its first report or edit, as before.
    if (measured) {
        queueMicrotask(() => {
            if (!disposed && typeof requestAnimationFrame === 'function' && input.clientWidth > 0) remeasure();
        });
    }

    return {
        mode,
        // Rebase on a changed editing place, not on a box the fold has already started to narrow.
        remeasureViewport(write) {
            if (!disposed) {
                const now = readCaret();
                if (!samePlace(caretLayout, now)) caretLayout = now;
            }
            write();
            if (!disposed && (measured || caretLayout)) remeasure();
        },
        // After a value change: typing, paste, a restored draft, a recalled or sent message.
        sync() {
            if (disposed) return;
            if (measured) measure(true);
            else input.scrollTop = scrollAfterEdit(input.scrollTop);
            caretLayout = readCaret();
        },
        dispose() {
            disposed = true;
            caretLayout = null;
            observer?.disconnect();
            observer = null;
            placeholder?.disconnect();
            placeholder = null;
            keyboard?.disconnect();
            keyboard = null;
            for (const target of viewports) target.removeEventListener?.('resize', remeasure);
            fonts?.removeEventListener('loading', fontsLoading);
            if (queued) cancelAnimationFrame(frame);
            queued = false;
        },
    };
}

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

export function attachChatInputSizing(input) {
    const mode = chatInputSizingMode();
    const measured = mode !== 'content';
    let disposed = false;
    // The width the last measurement wrapped at, unrounded: clientWidth is whole pixels,
    // so a sub-pixel change (a window resized at 125%) could rewrap unseen. 0: unmeasured.
    let width = 0;
    // Something other than the width changed the wrapping: a loaded font, a placeholder.
    let stale = false;
    let queued = false;
    let frame = 0;
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
        const cap = parseFloat(style.maxHeight);  // NaN for `none`: never capped
        const box = style.boxSizing === 'border-box'
            ? px(style.borderTopWidth) + px(style.borderBottomWidth)
            : -(px(style.paddingTop) + px(style.paddingBottom));
        const scrollTop = input.scrollTop;
        // Measured at its natural one-row height without a scrollbar, which would
        // narrow the text it measures. The row around it keeps its height meanwhile:
        // where the composer is in flow (the phone's keyboard is up) a one-row field
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
    // reveal), a loaded font or a new placeholder re-measures: the height this module
    // writes reports back and is ignored. The write waits for the next frame, because
    // inside the observer it would resize the composer that chat.js observes higher in
    // the same tree.
    const remeasure = () => {
        if (queued) return;
        queued = true;
        frame = requestAnimationFrame(() => {
            queued = false;
            if (!disposed && (stale || usedWidth() !== width)) measure(false);
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
    if (measured && typeof ResizeObserver === 'function') {
        observer = new ResizeObserver(remeasure);
        observer.observe(input);
    }
    // A placeholder rewritten in place: the interface language's dictionary arriving.
    if (measured && typeof MutationObserver === 'function') {
        placeholder = new MutationObserver(markStale);
        placeholder.observe(input, { attributes: true, attributeFilter: ['placeholder'] });
    }
    fonts?.addEventListener('loading', fontsLoading);
    // A loading period already running sent its `loading` before this listener.
    if (fonts?.status === 'loading') fontsLoading();

    return {
        mode,
        // After a value change: typing, paste, a restored draft, a recalled or sent message.
        sync() {
            if (disposed) return;
            if (measured) measure(true);
            else input.scrollTop = scrollAfterEdit(input.scrollTop);
        },
        dispose() {
            disposed = true;
            observer?.disconnect();
            observer = null;
            placeholder?.disconnect();
            placeholder = null;
            fonts?.removeEventListener('loading', fontsLoading);
            if (queued) cancelAnimationFrame(frame);
            queued = false;
        },
    };
}

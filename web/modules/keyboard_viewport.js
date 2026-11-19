import { isStackLayout, onLayoutChange, pinchMagnified } from './layout.js';

// A disappearing keyboard can settle after the first paint without another resize. Check early, then sparingly
// through the native dismissal tail; each blur gets only this burst, even if the viewport never grows.
const BLUR_CHECK_AT_MS = [32, 100, 240, 460, 750];
const sameHeight = (a, b) => Math.abs(a - b) <= 1;

// Mobile soft-keyboard handling: viewport shrink counts only while an editable
// owns focus. Drawer opening clears that state explicitly before navigation is
// rendered, so stale WebView geometry cannot hide an otherwise-open sidebar.
export function bindKeyboardViewport({ telegramHost, shownRoom }) {
    const vvhStyle = document.createElement('style');
    vvhStyle.id = 'runtime-vvh';
    document.head.appendChild(vvhStyle);

    const keyboardEditableSelector = [
        'textarea',
        'select',
        'input:not([type])',
        'input[type="text"]',
        'input[type="search"]',
        'input[type="email"]',
        'input[type="url"]',
        'input[type="tel"]',
        'input[type="password"]',
        'input[type="number"]',
        '[contenteditable]:not([contenteditable="false"])',
    ].join(',');

    let keyboardOpen = false;
    let keyboardTouchStartY = 0;
    let stableViewportHeight = 0;
    let restingLayoutHeight = 0;
    let restingLayoutWidth = 0;
    let focusBaselineHeight = 0;
    let resizeKind = '';
    let baselineFrame = 0;
    let lastLayoutHeight = 0;
    let lastPairedHeight = 0;
    let closingLayoutHeight = 0;
    let viewportCap = 0;
    let blurFrame = 0;
    let blurTimer = 0;

    function keyboardEditable(node) {
        if (!(node instanceof Element)) return null;
        const editable = node.closest(keyboardEditableSelector);
        if (!editable) return null;
        if (editable.matches('input, textarea, select') && (editable.disabled || editable.readOnly)) {
            return null;
        }
        return editable;
    }

    function focusedKeyboardEditable() {
        return keyboardEditable(document.activeElement);
    }

    function viewportHeight(layoutHeight) {
        const visual = window.visualViewport?.height;
        if (Number.isFinite(visual) && visual > 0) return visual;
        if (Number.isFinite(layoutHeight) && layoutHeight > 0) return layoutHeight;
        return document.documentElement.clientHeight;
    }

    function cancelBlurChecks() {
        if (blurFrame) cancelAnimationFrame(blurFrame);
        if (blurTimer) clearTimeout(blurTimer);
        blurFrame = blurTimer = closingLayoutHeight = 0;
    }

    function setRestingHeight(height, layoutHeight, layoutWidth) {
        stableViewportHeight = height;
        restingLayoutHeight = layoutHeight;
        restingLayoutWidth = layoutWidth;
        focusBaselineHeight = 0;
        resizeKind = '';
    }

    function cancelBaselineFrame() {
        if (baselineFrame) cancelAnimationFrame(baselineFrame);
        baselineFrame = 0;
    }

    function cancelViewportChecks() {
        cancelBaselineFrame();
        cancelBlurChecks();
    }

    function measureBaselineNextFrame(height) {
        cancelBaselineFrame();
        baselineFrame = requestAnimationFrame(() => {
            baselineFrame = 0;
            updateVvh(undefined, height);
        });
    }

    function findScrollableKeyboardNode(target) {
        let el = target;
        while (el && el !== document.body) {
            const overflow = getComputedStyle(el).overflowY;
            if (['auto', 'scroll'].includes(overflow) && el.scrollHeight > el.clientHeight) return el;
            el = el.parentElement;
        }
        return null;
    }

    function lockTouchStart(e) {
        if (e.touches && e.touches.length) keyboardTouchStartY = e.touches[0].clientY;
    }

    function lockBoundaryTouch(e) {
        if (e.touches?.length !== 1 || pinchMagnified()) return;
        const touch = e.touches[0];
        const scrollable = findScrollableKeyboardNode(e.target);
        if (scrollable && touch) {
            const dy = touch.clientY - keyboardTouchStartY;
            const atTop = scrollable.scrollTop <= 0;
            const atBottom = Math.ceil(scrollable.scrollTop + scrollable.clientHeight) >= scrollable.scrollHeight;
            if ((!atTop && dy > 0) || (!atBottom && dy < 0)) return;
        }
        e.preventDefault();
    }

    function applyKeyboardState(visible) {
        const nextVisible = Boolean(visible);
        if (nextVisible && !keyboardOpen) {
            window.scrollTo(0, 0);
            document.addEventListener('touchstart', lockTouchStart, { passive: true });
            document.addEventListener('touchmove', lockBoundaryTouch, { passive: false });
        } else if (!nextVisible && keyboardOpen) {
            document.removeEventListener('touchstart', lockTouchStart);
            document.removeEventListener('touchmove', lockBoundaryTouch);
        }
        document.documentElement.classList.toggle('keyboard-open', nextVisible);
        document.body.classList.toggle('keyboard-open', nextVisible);
        keyboardOpen = nextVisible;
    }

    const updateVvh = (event, settledHeight = 0) => {
        const inner = window.innerHeight;
        const h = viewportHeight(inner);
        const layoutHeight = inner || h;
        const layoutWidth = window.innerWidth;
        const focused = focusedKeyboardEditable();
        if (!isStackLayout()) {
            cancelViewportChecks();
            applyKeyboardState(false);
            // A docked leg never supplies the phone's reference height. Keep it only across the same focus.
            if (!focused) setRestingHeight(0, 0, 0);
            document.body.classList.remove('keyboard-closing');
            lastPairedHeight = lastLayoutHeight = 0;
            viewportCap = 0;
            vvhStyle.textContent = ':root{--vvh:100dvh}';
            return false;
        }
        if (!h || pinchMagnified()) {
            if (h) cancelViewportChecks(); // a pinch keeps its own pan and needs no deferred reads
            if (!focused && keyboardOpen) applyKeyboardState(false);
            return false;
        }
        const paired = sameHeight(layoutHeight, h);
        const atRest = !focused && !blurFrame && !blurTimer;
        const resize = event?.type === 'resize';
        const freshPairedResize = paired && !sameHeight(lastPairedHeight, layoutHeight)
            && (resize || event?.type === 'ouro:host-viewport');
        const layoutResized = resize && event.currentTarget === window && lastLayoutHeight
            && !sameHeight(lastLayoutHeight, layoutHeight);
        lastLayoutHeight = layoutHeight;
        if (paired) lastPairedHeight = layoutHeight;
        // A window resize can announce a new layout before visual catches up. Remember the announced height,
        // not the intermediate visual sample; only a matching pair at the end of the blur burst can confirm it.
        if (!focused && !atRest && (freshPairedResize || layoutResized)) closingLayoutHeight = layoutHeight;
        const restingHeight = Math.min(h, layoutHeight);
        if (!stableViewportHeight) setRestingHeight(restingHeight, layoutHeight, layoutWidth);
        if (focused && !focusBaselineHeight) focusBaselineHeight = stableViewportHeight;
        const typingViewport = Boolean(focusBaselineHeight);

        // A delayed layout shrink at the same width can still be the keyboard. Only use a different layout as full
        // while focused when both dimensions show a turn and this focus already showed a visual-only keyboard.
        const turned = layoutWidth !== restingLayoutWidth && !sameHeight(layoutHeight, restingLayoutHeight)
            && (layoutWidth > layoutHeight) !== (restingLayoutWidth > restingLayoutHeight);
        if (typingViewport && paired && !turned && layoutHeight < focusBaselineHeight - 1) resizeKind = 'paired';
        if (typingViewport && resizeKind !== 'paired' && layoutHeight > h + 1) {
            if (sameHeight(layoutHeight, restingLayoutHeight)) {
                if (h < focusBaselineHeight - 1) resizeKind = 'visual';
            } else if (focused && resizeKind === 'visual' && turned) {
                // Window resize may lead the shell's media-query notification. Let it paint before learning a
                // new compact height: a docked sample will cancel the frame instead of replacing the reference.
                if (settledHeight && sameHeight(settledHeight, layoutHeight)) {
                    focusBaselineHeight = stableViewportHeight = restingLayoutHeight = layoutHeight;
                    restingLayoutWidth = layoutWidth;
                } else if (!settledHeight && !baselineFrame) measureBaselineNextFrame(layoutHeight);
            }
        }
        if (atRest && settledHeight && sameHeight(settledHeight, restingHeight) && (!typingViewport || paired)) {
            setRestingHeight(restingHeight, layoutHeight, layoutWidth);
        }
        const shrink = Math.max(0, focusBaselineHeight - h);
        const awaitingExpansion = shrink > 1;
        const shortened = shrink > Math.max(120, focusBaselineHeight * 0.25);
        const open = Boolean(focused && shortened);
        // Losing focus releases the keyboard controls immediately. Its short height is still
        // the room's height, but cannot become the baseline for the next editable's focus.
        if (!focused && !awaitingExpansion) {
            cancelBlurChecks();
            focusBaselineHeight = 0;
            resizeKind = '';
        }
        // Paint immediately, but let focus claim a shrink before treating it as rest. During blur we
        // only reconsider a fresh paired resize at the end of the existing bounded dismissal burst.
        if (!focused && !settledHeight && (!awaitingExpansion || (atRest && freshPairedResize))) {
            cancelBaselineFrame();
            if (restingHeight !== stableViewportHeight || awaitingExpansion) {
                measureBaselineNextFrame(restingHeight);
            }
        }

        const telegram = telegramHost.bound();
        const closing = telegram && !focused && awaitingExpansion && shortened;
        const closingChanged = document.body.classList.contains('keyboard-closing') !== closing;
        const overlap = telegram && (focused || awaitingExpansion) ? Math.round(shrink) : 0;
        const quarterCap = Math.max(28, Math.min(140, Math.floor(h / 4)));
        // Telegram's one floating composer starts from its resting cap, loses at most one pixel per pixel of
        // shrink, reaching the quarter cap before keyboard-open. The stylesheet still gives a tight room its own fit.
        const cap = telegram ? Math.max(quarterCap, 140 - overlap) : quarterCap;
        const capChanged = cap !== viewportCap;
        viewportCap = cap;
        const css = ':root{--vvh:' + Math.ceil(h) + 'px;--chat-viewport-input-cap:'
            + cap + 'px;--chat-keyboard-overlap:' + overlap + 'px}';
        const resized = css !== vvhStyle.textContent;
        const changed = open !== keyboardOpen;
        const room = telegram && (capChanged || (typingViewport && (resized || changed || closingChanged))) ? shownRoom() : null;
        const writeViewport = () => {
            if (closingChanged) document.body.classList.toggle('keyboard-closing', closing);
            if (resized) vvhStyle.textContent = css;
            if (changed) applyKeyboardState(open);
        };
        if (room) room.composer.remeasureInput(writeViewport);
        else writeViewport();
        // WebKit may scroll the document to the field after the first resize. Only the
        // room scrolls here. A pinch or a dialog's focused field keeps its own pan.
        if (telegram && open && focused.matches('textarea.chat-input, #chat-input') && window.scrollY) window.scrollTo(0, 0);
        if (typingViewport) room?.reflowViewport();
        return !focused && awaitingExpansion;
    };

    document.addEventListener('focusin', (event) => {
        if (!keyboardEditable(event.target)) return;
        cancelViewportChecks();
        updateVvh();
    });
    document.addEventListener('focusout', (event) => {
        if (!keyboardEditable(event.target)) return;
        cancelViewportChecks();
        const blurredAt = performance.now();
        const check = (index) => {
            const elapsed = performance.now() - blurredAt;
            while (BLUR_CHECK_AT_MS[index] <= elapsed) index++;
            const done = index === BLUR_CHECK_AT_MS.length;
            const height = done ? closingLayoutHeight : 0;
            if (done) closingLayoutHeight = 0;
            if (updateVvh(undefined, height) && !done) {
                blurTimer = setTimeout(() => { blurTimer = 0; check(index + 1); }, BLUR_CHECK_AT_MS[index] - elapsed);
            }
        };
        blurFrame = requestAnimationFrame(() => { blurFrame = 0; check(0); });
    });
    window.addEventListener('pagehide', cancelViewportChecks);

    function releaseKeyboardForDrawer() {
        const editable = focusedKeyboardEditable();
        if (editable && typeof editable.blur === 'function') editable.blur();
        applyKeyboardState(false);
    }

    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', updateVvh);
        window.visualViewport.addEventListener('scroll', updateVvh);
    }
    window.addEventListener('resize', updateVvh);
    window.addEventListener('ouro:host-viewport', updateVvh);
    onLayoutChange(() => updateVvh());
    window.addEventListener('scroll', () => { if (telegramHost.bound()) updateVvh(); }, { passive: true });
    updateVvh();
    return releaseKeyboardForDrawer;
}

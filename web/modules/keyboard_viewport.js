import { syncShortRoom } from './composer_fold.js';
import { currentPointer, isStackLayout, onLayoutChange, pinchMagnified } from './layout.js';

const SETTLE_MS = 250;
const QUICK_REFOCUS_MS = 1000;
const POST_BLUR_READS = ['pointerdown', 'touchstart', 'scroll'];
const EDITABLE = [
    'textarea', 'select', 'input:not([type])', 'input[type="text"]', 'input[type="search"]',
    'input[type="email"]', 'input[type="url"]', 'input[type="tel"]', 'input[type="password"]',
    'input[type="number"]', '[contenteditable]:not([contenteditable="false"])',
].join(',');
const keyboardShort = (base, height) => base - height > Math.max(120, base * 0.25);
const hostMode = () => document.documentElement.dataset?.telegramFullscreen || '';
const nowMs = () => (typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now());

// Phone viewport and soft-keyboard contract:
// - Room height (--vvh) is always visualViewport.height in stack layout (100dvh when docked, untouched while
//   pinch-zoomed), with document scroll (window.scrollY) kept at 0 while the room is fixed.
// - keyboard-open is cosmetic only (composer air, short-room fold, look-B chrome), derived from coarse focus + shrink,
//   held across in-session geometry resets and quick refocus while the viewport stays keyboard-short, and cleared on
//   session end or keyboard-sized viewport growth.
export function bindKeyboardViewport({ shownRoom = () => null } = {}) {
    const vvhStyle = document.createElement('style');
    vvhStyle.id = 'runtime-vvh';
    document.head.appendChild(vvhStyle);
    const vv = window.visualViewport;

    let keyboardOpen = false, touchStartY = 0, focusHeight = 0, heldFloor = 0, recentOpen = null;
    let lastWidth = window.innerWidth, lastHeight = 0, lastMode = hostMode();
    let raf1 = 0, raf2 = 0, settleTimer = 0, watchingAfterBlur = false;

    function keyboardEditable(node) {
        if (!(node instanceof Element)) return null;
        const el = node.closest(EDITABLE);
        return el && !(el.matches('input, textarea, select') && (el.disabled || el.readOnly)) ? el : null;
    }

    const focusedEditable = () => keyboardEditable(document.activeElement);

    function viewportHeight() {
        const visual = window.visualViewport?.height;
        if (Number.isFinite(visual) && visual > 0) return visual;
        return Number.isFinite(window.innerHeight) && window.innerHeight > 0
            ? window.innerHeight
            : document.documentElement.clientHeight;
    }

    function scrollsThisWay(target, dy) {
        for (let el = target; dy && el && el !== document.body; el = el.parentElement) {
            if (!['auto', 'scroll'].includes(getComputedStyle(el).overflowY) || el.scrollHeight <= el.clientHeight) continue;
            if (dy > 0 ? el.scrollTop > 0 : Math.ceil(el.scrollTop + el.clientHeight) < el.scrollHeight) return true;
        }
        return false;
    }

    const lockTouchStart = (e) => {
        if (e.touches?.length) touchStartY = e.touches[0].clientY;
    };

    const lockBoundaryTouch = (e) => {
        if (e.touches?.length === 1 && !pinchMagnified() && !scrollsThisWay(e.target, e.touches[0].clientY - touchStartY)) {
            e.preventDefault();
        }
    };

    function applyKeyboardState(visible) {
        const next = Boolean(visible);
        if (next && !keyboardOpen) {
            window.scrollTo(0, 0);
            document.addEventListener('touchstart', lockTouchStart, { passive: true });
            document.addEventListener('touchmove', lockBoundaryTouch, { passive: false });
        } else if (!next && keyboardOpen) {
            document.removeEventListener('touchstart', lockTouchStart);
            document.removeEventListener('touchmove', lockBoundaryTouch);
        }
        document.documentElement.classList.toggle('keyboard-open', next);
        document.body.classList.toggle('keyboard-open', next);
        keyboardOpen = next;
    }

    const onPostBlurInput = (event) => {
        if (event?.type !== 'scroll') watchAfterBlur(false);
        if (updateVvh() && event?.type === 'scroll') watchAfterBlur(false);
    };

    function watchAfterBlur(on) {
        if (on === watchingAfterBlur) return;
        watchingAfterBlur = on;
        const method = on ? 'addEventListener' : 'removeEventListener';
        for (const type of POST_BLUR_READS) document[method](type, onPostBlurInput, { capture: true, passive: true });
    }

    function cancelScheduled() {
        if (raf1) cancelAnimationFrame(raf1);
        if (raf2) cancelAnimationFrame(raf2);
        if (settleTimer) clearTimeout(settleTimer);
        raf1 = raf2 = settleTimer = 0;
    }

    function cancelChecks() {
        cancelScheduled();
        watchAfterBlur(false);
    }

    function commitRoomChange(write) {
        const room = shownRoom();
        if (room) {
            room.composer.remeasureInput(() => {
                write();
                syncShortRoom(room.page);
            });
            room.reflowViewport();
        } else {
            write();
        }
    }

    function updateVvh(event) {
        const h = viewportHeight();
        const layoutHeight = window.innerHeight || h;
        const width = window.innerWidth, mode = hostMode(), focused = focusedEditable();
        if (!isStackLayout()) {
            cancelChecks();
            focusHeight = heldFloor = 0;
            recentOpen = null;
            lastWidth = width;
            lastHeight = h || lastHeight;
            lastMode = mode;
            const css = ':root{--vvh:100dvh}';
            const resized = css !== vvhStyle.textContent, changed = keyboardOpen;
            if (resized || changed) {
                commitRoomChange(() => {
                    if (resized) vvhStyle.textContent = css;
                    if (changed) applyKeyboardState(false);
                });
            }
            return resized;
        }
        if (!h || pinchMagnified()) {
            if (h) cancelScheduled();
            if (!focused) {
                focusHeight = heldFloor = 0;
                if (keyboardOpen) commitRoomChange(() => applyKeyboardState(false));
            }
            return false;
        }
        if (window.scrollY) window.scrollTo(0, 0);

        const soft = currentPointer() !== 'fine';
        const prevWidth = lastWidth, prevHeight = lastHeight || h;
        const geoReset = width !== lastWidth || mode !== lastMode || event?.type === 'pageshow';
        lastWidth = width;
        lastHeight = h;
        lastMode = mode;

        if (recentOpen && (
            !soft
            || nowMs() - recentOpen.at > QUICK_REFOCUS_MS
            || width !== recentOpen.width
            || mode !== recentOpen.mode
            || keyboardShort(h, recentOpen.h)
        )) {
            recentOpen = null;
        } else if (recentOpen && !focused) {
            recentOpen.h = Math.min(recentOpen.h, h);
        }

        if (!focused || !soft) {
            focusHeight = heldFloor = 0;
        } else if (recentOpen) {
            recentOpen = null;
            focusHeight = heldFloor = h;
        } else if (geoReset) {
            const closedOnReset = Math.abs(width - prevWidth) > 100
                ? !keyboardShort(prevWidth, h)
                : keyboardShort(h, heldFloor || prevHeight);
            heldFloor = keyboardOpen && !closedOnReset ? h : 0;
            focusHeight = h;
        } else {
            if (heldFloor) heldFloor = keyboardShort(h, heldFloor) ? 0 : Math.min(heldFloor, h);
            focusHeight = Math.max(focusHeight || h, h);
        }

        const open = Boolean(focused && soft && (heldFloor > 0 || keyboardShort(layoutHeight, h) || keyboardShort(focusHeight, h)));
        const cap = Math.max(28, Math.min(140, Math.floor(h / 4)));
        const css = `:root{--vvh:${Math.ceil(h)}px;--chat-viewport-input-cap:${cap}px}`;
        const resized = css !== vvhStyle.textContent, changed = open !== keyboardOpen;
        if (resized || changed) {
            commitRoomChange(() => {
                if (resized) vvhStyle.textContent = css;
                if (changed) applyKeyboardState(open);
            });
        }
        return resized;
    }

    function scheduleSettle(afterBlur = false) {
        cancelScheduled();
        raf1 = requestAnimationFrame(() => {
            raf1 = 0;
            updateVvh();
            raf2 = requestAnimationFrame(() => {
                raf2 = 0;
                updateVvh();
                settleTimer = setTimeout(() => {
                    settleTimer = 0;
                    updateVvh();
                    if (afterBlur && isStackLayout() && !focusedEditable()) watchAfterBlur(true);
                }, SETTLE_MS);
            });
        });
    }

    function onFocusIn(event) {
        if (!keyboardEditable(event.target)) return;
        watchAfterBlur(false);
        updateVvh(event);
        if (isStackLayout()) scheduleSettle(false);
    }

    function onFocusOut(event) {
        if (!keyboardEditable(event.target) || keyboardEditable(event.relatedTarget)) return;
        if (keyboardOpen && isStackLayout() && currentPointer() !== 'fine') {
            const h = viewportHeight();
            if (h > 0) recentOpen = { h, width: window.innerWidth, mode: hostMode(), at: nowMs() };
        }
        // Defer clearing keyboard-open to frame 1 so tapping Send or "+" lands its click before padding shifts.
        if (isStackLayout()) {
            watchAfterBlur(true);
            scheduleSettle(true);
        } else {
            focusHeight = heldFloor = 0;
            recentOpen = null;
            updateVvh(event);
        }
    }

    function releaseKeyboardForDrawer() {
        focusedEditable()?.blur?.();
        recentOpen = null;
        focusHeight = heldFloor = 0;
        if (keyboardOpen) commitRoomChange(() => applyKeyboardState(false));
        else applyKeyboardState(false);
    }

    const winEvents = ['resize', 'orientationchange', 'ouro:host-viewport', 'ouro:host-chrome'];
    document.addEventListener('focusin', onFocusIn);
    document.addEventListener('focusout', onFocusOut);
    window.addEventListener('pagehide', cancelChecks);
    window.addEventListener('pageshow', updateVvh);
    window.addEventListener('scroll', updateVvh, { passive: true });
    for (const type of winEvents) window.addEventListener(type, updateVvh);
    vv?.addEventListener('resize', updateVvh);
    vv?.addEventListener('scroll', updateVvh);
    const stopLayoutWatch = onLayoutChange(() => updateVvh());

    function dispose() {
        cancelChecks();
        applyKeyboardState(false);
        syncShortRoom(shownRoom()?.page);
        document.removeEventListener('focusin', onFocusIn);
        document.removeEventListener('focusout', onFocusOut);
        window.removeEventListener('pagehide', cancelChecks);
        window.removeEventListener('pageshow', updateVvh);
        window.removeEventListener('scroll', updateVvh);
        for (const type of winEvents) window.removeEventListener(type, updateVvh);
        vv?.removeEventListener('resize', updateVvh);
        vv?.removeEventListener('scroll', updateVvh);
        stopLayoutWatch();
        vvhStyle.remove();
    }

    updateVvh();
    return Object.assign(releaseKeyboardForDrawer, { dispose });
}

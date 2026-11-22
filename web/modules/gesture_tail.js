/** The tail of a finished touch gesture (docs/DESIGN.md "Replies", "Shell"): the click, and where its owner asks the
 * mousedown, that a host may still fire for the finger that swiped a row, dragged the drawer or held a long press. It
 * lands on whatever lies under the lifted finger and is not a tap on it.
 *
 * The tail belongs to its gesture, not to a stretch of time. It is swallowed until it arrives or until the next
 * independent action begins: a new touch or pointer of any device, on any target, or a key. A tap right after a
 * gesture therefore acts, once, even where the host sent the gesture no click at all (a finger that moved). A click no
 * press made (a key's Enter or Space, an assistive technology's, a script's) is never a finger's tail. Nothing listens
 * before a gesture ends or after its tail: no timer and no standing document layer. Each gesture owner keeps its own
 * and releases it with its disposer. */

const FRESH = ['pointerdown', 'touchstart', 'keydown'];
const FRESH_OPTIONS = { capture: true, passive: true };

/** `doc` is the document of the gesture's finger; `mousedown`: its tail's mousedown is swallowed too (it would take the
 * focus out of a menu the gesture just opened). Returns `{ arm, disarm, armed }`: `arm` as the gesture ends. */
export function createGestureTail(doc, { mousedown = false } = {}) {
    let armed = false;

    // Whether `event` is the finger's and so swallowed: a script's click, or one no press made, is not.
    function swallow(event) {
        if (event.isTrusted === false || event.detail === 0) return false;
        event.preventDefault();
        event.stopPropagation();
        return true;
    }
    function onClick(event) { if (swallow(event)) disarm(); }  // the click is the tail's last event

    function arm() {
        if (armed || typeof doc?.addEventListener !== 'function') return;
        armed = true;
        if (mousedown) doc.addEventListener('mousedown', swallow, true);
        doc.addEventListener('click', onClick, true);
        for (const type of FRESH) doc.addEventListener(type, disarm, FRESH_OPTIONS);
    }

    function disarm() {
        if (!armed) return;
        armed = false;
        if (mousedown) doc.removeEventListener('mousedown', swallow, true);
        doc.removeEventListener('click', onClick, true);
        for (const type of FRESH) doc.removeEventListener(type, disarm, FRESH_OPTIONS);
    }

    return { arm, disarm, get armed() { return armed; } };
}

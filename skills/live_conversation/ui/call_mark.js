/** The shell's "In call" mark. A call outlives its room leaving the screen (Settings, another
 * Project, a closed panel), so while one lasts and its room is not on screen, one capsule at the
 * top of the window names the room and the call's time: pressing it returns to the room, End ends
 * the call from wherever the owner is. It reads only the rooms' own facts: their sessions, and
 * `onScreen`, which app.js answers from its navigation state. Navigation announces itself
 * (`ouro:page-shown`, the Project panel's body class, a room page hidden or shown); the clock ticks
 * once a second only while the mark is shown.
 */
const entries = new Set();
let mounted = null;  // { root, room, time, timer, observer, watched, env }

function clock(since, now) {
    if (!since) return '';
    const seconds = Math.max(0, Math.floor((now - since) / 1000));
    const hours = Math.floor(seconds / 3600);
    const minutes = String(Math.floor(seconds / 60) % 60).padStart(2, '0');
    const rest = String(seconds % 60).padStart(2, '0');
    return hours ? `${hours}:${minutes}:${rest}` : `${minutes}:${rest}`;
}

/** The entry the mark stands for: the one live call, when its room is not on screen. */
export function markedCall(list = entries) {
    const call = [...list].find(entry => entry.session.active);
    return call && !call.onScreen() ? call : null;
}

function mount(env, sign) {
    const doc = env.document;
    const root = doc.createElement('div');
    root.className = 'call-mark';
    root.setAttribute('role', 'region');
    root.setAttribute('aria-label', 'Call in progress');
    root.hidden = true;
    root.innerHTML = `<button type="button" class="call-mark-open"><span class="call-mark-label">In call</span><span class="call-mark-room" data-i18n-skip></span><span class="call-mark-time" data-i18n-skip></span></button>
        <button type="button" class="call-mark-end">End</button>`;
    // The visual's own small sign leads the mark, as it leads Talk.
    if (sign) root.querySelector('.call-mark-open').prepend(sign());
    doc.body.append(root);
    const state = { root, env, room: root.querySelector('.call-mark-room'), time: root.querySelector('.call-mark-time'),
        timer: 0, watched: null, observer: null, current: null };
    root.querySelector('.call-mark-open').addEventListener('click', () => state.current?.reveal());
    root.querySelector('.call-mark-end').addEventListener('click', () => state.current?.session.end());
    // Navigation changed what is on screen: the mark follows, and a room shown again resumes its light.
    const resync = () => { for (const entry of entries) entry.wake?.(); sync(env); };
    state.resync = resync;
    env.addEventListener?.('ouro:page-shown', resync);
    state.observer = typeof env.MutationObserver === 'function' ? new env.MutationObserver(resync) : null;
    state.observer?.observe(doc.body, { attributes: true, attributeFilter: ['class'] });
    mounted = state;
    return state;
}

function unmount() {
    if (!mounted) return;
    const { root, env, timer, observer, resync } = mounted;
    env.clearInterval(timer);
    observer?.disconnect();
    env.removeEventListener?.('ouro:page-shown', resync);
    root.remove();
    mounted = null;
}

export function sync(env = globalThis) {
    const call = markedCall();
    // Mounted (hidden) from a call's first phase, so navigation away from its room is heard.
    if (!mounted && ![...entries].some(entry => entry.session.active)) return;
    if (!entries.size) { unmount(); return; }
    const state = mounted || mount(env, [...entries][0]?.sign);
    state.current = call;
    // A room page shown or hidden by navigation changes what is on screen.
    const page = [...entries].find(entry => entry.session.active)?.page || null;
    if (state.watched !== page) {
        state.observer?.disconnect();
        state.observer?.observe(env.document.body, { attributes: true, attributeFilter: ['class'] });
        if (page) state.observer?.observe(page, { attributes: true, attributeFilter: ['hidden', 'class'] });
        state.watched = page;
    }
    state.root.hidden = !call;
    if (!call) { env.clearInterval(state.timer); state.timer = 0; return; }
    const paint = () => {
        const room = String(call.label() || '');
        if (state.room.textContent !== room) state.room.textContent = room;
        const time = clock(call.session.liveSince, Date.now());
        if (state.time.textContent !== time) state.time.textContent = time;
        state.time.hidden = !time;
    };
    paint();
    if (!state.timer) state.timer = env.setInterval(() => (markedCall() ? paint() : sync(env)), 1000);
}

/** One room call joins the mark's census; its entry is told about changes and leaves on destroy. */
export function trackCallMark({ session, label, onScreen, reveal, page, env = globalThis, sign = null, wake = null }) {
    const entry = { session, label, onScreen, reveal, page, sign, wake,
        changed: () => sync(env),
        destroy() { entries.delete(entry); sync(env); } };
    entries.add(entry);
    return entry;
}

export { clock as callClock };

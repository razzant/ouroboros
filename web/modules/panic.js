/** Immediate controls never enter an offline queue or a saved resend store. */
import { showToast } from './toast.js';
import { desktopRoomStorage } from './desktop_room_session.js';

// A native door takes the stop when it receives the message: its reply only tells the page, a stop can end the app before
// that reply is sent, and a call the launcher declines never answers (pywebview's promise has no timeout of its own). The
// reply is waited for this long, then the press asks its captured socket too.
const DOOR_REPLY_MS = 1000;
const SOCKET_ACK_MS = 2000;

export function hasDesktopPanic() {
    return Boolean(desktopRoomStorage()?.panic) || typeof globalThis.pywebview?.api?.panic_stop === 'function';
}

export function isPanicFrame(frame) {
    return ['chat', 'command'].includes(frame?.type)
        && String(frame.content || frame.cmd || '').trim().toLowerCase().startsWith('/panic');
}

export async function panicWithoutSocket(surface, socket = null, onAccepted) {
    // The desktop launcher owns its own stop, even if the server cannot answer.
    const native = desktopRoomStorage();
    const bridge = globalThis.pywebview?.api?.panic_stop;
    // One press settles once: the first door that confirms the stop, else the failure once every door is out. A late
    // reply, a second door's answer or a closed socket after that changes nothing.
    let settled = false, stopSocket = () => {};
    const settle = (accepted) => {
        if (settled) return;
        settled = true;
        stopSocket();
        if (accepted) { onAccepted?.(); return; }
        showToast(hasDesktopPanic()
            ? 'Panic was not confirmed. Use Panic in the tray or menu bar.'
            : 'Panic was not confirmed. Check Ouroboros on the host computer.', 'error', { ttl: 0 });
    };
    // A WebSocket belongs to one server connection. HTTP can reconnect to a successor; this door must not. The only
    // socket this press may use is the connection it holds now, open and unbacked, and never a later one.
    const captured = socket?.readyState === 1 && !(socket.bufferedAmount > 0) ? socket : null;
    // The stamped WebKit document has a door before pywebview injection. pywebview builds each bridge function from the
    // published Python parameters (`new Function(params, body)`, every argument forwarded), so `length` is the launcher's
    // own parameter count: 1, the document's ticket, for a launcher that verifies it; 0 for an older one, which gets none.
    const ticket = typeof bridge === 'function' && bridge.length && native?.panicTicket;
    const bound = native?.panic ? () => native.panic() : ticket ? () => bridge(ticket) : null;
    const older = !bound && typeof bridge === 'function' ? () => bridge() : null;
    if (bound || older) {
        // Asked at once, before anything else. Its refusal, a lost reply or none within the bound falls through to the
        // captured socket: /panic is idempotent, so a stop the door took already is only asked for again, and a stop
        // it never took still reaches the server. A reply after the bound still settles a press that has not. An older
        // launcher's refusal, which it never tied to a ticket, ends the press as it always has, but only while it is
        // still waited for: once the socket has been asked, that one answers.
        await new Promise((resolve) => {
            let timer, waiting = true;
            const answered = (reply, resolved) => {
                clearTimeout(timer);
                if (reply?.status === 'stopping') settle(true);
                else if (resolved && older && waiting) settle(false);
                resolve();
            };
            try { Promise.resolve((bound || older)()).then((reply) => answered(reply, true), () => answered(null, false)); }
            catch { answered(null, false); return; }
            timer = setTimeout(() => { waiting = false; resolve(); }, DOOR_REPLY_MS);
        });
    }
    if (settled) return;
    // Bytes already sent may arrive later on this same connection: an ACK proves
    // acceptance, neither physical exit nor a bounded network delivery time.
    if (!captured || captured.readyState !== 1 || captured.bufferedAmount > 0) {
        settle(false);
        return;
    }
    let timer;
    stopSocket = () => {
        clearTimeout(timer);
        captured.removeEventListener('message', received);
        captured.removeEventListener('close', closed);
    };
    const closed = () => settle(false);
    const received = (event) => {
        let message;
        try { message = JSON.parse(event.data); } catch { return; }
        if (message?.type === 'panic_accepted') settle(true);
    };
    captured.addEventListener('message', received);
    captured.addEventListener('close', closed);
    timer = setTimeout(() => settle(false), SOCKET_ACK_MS);
    try {
        captured.send(JSON.stringify({ type: 'command', cmd: '/panic', client_surface: surface || { channel: 'web' } }));
    } catch {
        settle(false);
    }
}

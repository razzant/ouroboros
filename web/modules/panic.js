/** Immediate controls never enter an offline queue or a saved resend store. */
import { showToast } from './toast.js';
import { desktopRoomStorage } from './desktop_room_session.js';

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
    const failed = () => showToast(hasDesktopPanic()
        ? 'Panic was not confirmed. Use Panic in the tray or menu bar.'
        : 'Panic was not confirmed. Check Ouroboros on the host computer.', 'error', { ttl: 0 });
    if (native?.panic) {
        // The stamped WebKit document has a door before pywebview injection. A lost reply
        // could follow an accepted stop: this gesture never attempts a second transport.
        let accepted;
        try { accepted = (await native.panic())?.status === 'stopping'; } catch { /* Outcome unconfirmed. */ }
        if (accepted) onAccepted?.(); else failed();
        return;
    }
    if (typeof bridge === 'function') {
        let resolved = false, accepted = false;
        // pywebview 5.4 generates the function with the published Python parameters.
        // Older launchers take zero arguments; both use this single immediate native call.
        const ticket = bridge.length && native?.panicTicket;
        try { accepted = (await (ticket ? bridge(ticket) : bridge()))?.status === 'stopping'; resolved = true; } catch { /* Bound delivery may be unconfirmed. */ }
        if (resolved || ticket) {
            if (accepted) onAccepted?.(); else failed();
            return;
        }
    }
    // A WebSocket belongs to one server connection. HTTP can reconnect to a
    // successor; this door must not. No client/server clock comparison is needed.
    // Bytes already sent may arrive later on this same connection: an ACK proves
    // acceptance, neither physical exit nor a bounded network delivery time.
    if (!socket || socket.readyState !== 1 || socket.bufferedAmount > 0) {
        failed();
        return;
    }
    let timer;
    const dispose = () => {
        clearTimeout(timer);
        socket.removeEventListener('message', received);
        socket.removeEventListener('close', closed);
    };
    const closed = () => { dispose(); failed(); };
    const received = event => {
        let message;
        try { message = JSON.parse(event.data); } catch { return; }
        if (message?.type === 'panic_accepted') { dispose(); onAccepted?.(); }
    };
    socket.addEventListener('message', received);
    socket.addEventListener('close', closed);
    timer = setTimeout(() => { dispose(); failed(); }, 2000);
    try {
        socket.send(JSON.stringify({ type: 'command', cmd: '/panic', client_surface: surface || { channel: 'web' } }));
    } catch {
        dispose();
        failed();
    }
}

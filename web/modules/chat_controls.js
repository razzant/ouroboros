// Pure chat header controls. Dependency-free at import time: the /panic flow
// takes its dialog and websocket seams as injected deps, so the Node suite
// drives the REAL production path without a DOM.

/**
 * /panic gate (v6.90.3, CRITICAL CONTROL): pure decision helper between the
 * confirm dialog's resolution and sending the panic command. Panic fires on an
 * EXPLICIT boolean-true confirm and on nothing else — cancel, backdrop,
 * Escape, and a dialog API drift that starts resolving objects (the input
 * mode's `{confirmed, value}` shape) all read as "do not fire". Node-tested.
 */
export function shouldFirePanic(dialogResult) {
    return dialogResult === true;
}

/**
 * /panic action (v6.90.3, CRITICAL CONTROL): the COMPLETE confirm-and-send
 * flow behind the header's Panic button, with injectable deps so the node
 * suite drives the REAL production path — dialog options, the strict
 * shouldFirePanic gate, and the exact outbound command — not just the boolean
 * helper. The header action passes the real openConfirmDialog and ws; a
 * broken await, option drift, or command typo here fails the node test
 * instead of leaving the live button silently inert.
 * Fires exactly one {type:'command', cmd:'/panic'} on an explicit confirm;
 * cancel/backdrop/Escape (false) send NOTHING.
 */
export async function confirmAndSendPanic(deps) {
    const decision = await deps.openConfirmDialog({
        title: 'Panic — stop all workers',
        body: 'Kill all workers immediately?',
        confirmLabel: 'Kill all workers',
        cancelLabel: 'Keep running',
        danger: true,
    });
    if (shouldFirePanic(decision)) {
        deps.ws.send({ type: 'command', cmd: '/panic' });
        return true;
    }
    return false;
}

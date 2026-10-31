const status = document.querySelector('#connection');
const send = document.querySelector('#send');
const reconnect = document.querySelector('#reconnect');
let socket;

async function connect() {
    reconnect.hidden = true;
    send.disabled = true;
    try {
        const response = await fetch('/api/onboarding/bootstrap', {cache: 'no-store'});
        if (response.status === 401) {
            const next = location.pathname + location.search + location.hash;
            location.assign('/auth/login?next=' + encodeURIComponent(next));
            return;
        }
        if (!response.ok) throw new Error(`Readiness check returned HTTP ${response.status}`);
        if (!(await response.json()).ready) {
            location.assign('/onboarding');
            return;
        }
        const state = await fetch('/api/state', {cache: 'no-store'});
        if (!state.ok) throw new Error(`State read returned HTTP ${state.status}`);
        document.body.dataset.servedUi = (await state.json()).served_ui?.source || 'unknown';
        socket?.close();
        const url = new URL('/ws', location.href);
        url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(url);
        socket.addEventListener('open', () => {
            status.textContent = 'Connected';
            send.disabled = false;
        });
        socket.addEventListener('close', () => {
            status.textContent = 'Disconnected. Reconnect to read current state; messages are not resent.';
            send.disabled = true;
            reconnect.hidden = false;
        });
        socket.addEventListener('message', event => {
            const message = JSON.parse(event.data);
            if (message.type !== 'chat' || (message.chat_id != null && message.chat_id !== 1)) return;
            const row = document.createElement('li');
            row.dataset.role = message.role;
            row.dataset.accepted = String(message.ingress_accepted === true);
            row.textContent = `${message.role}: ${message.content}`;
            document.querySelector('#messages').appendChild(row);
            if (message.ingress_accepted === true) status.textContent = 'Accepted by the core';
        });
    } catch (error) {
        status.textContent = error.message;
        reconnect.hidden = false;
    }
}

document.querySelector('#chat').addEventListener('submit', event => {
    event.preventDefault();
    if (socket?.readyState !== WebSocket.OPEN) return;
    const field = document.querySelector('#message');
    // getRandomValues also works on ordinary HTTP LAN origins.
    const id = Array.from(crypto.getRandomValues(new Uint8Array(16)),
        byte => byte.toString(16).padStart(2, '0')).join('');
    socket.send(JSON.stringify({type: 'chat', content: field.value, chat_id: 1,
        client_message_id: id}));
    field.value = '';
    status.textContent = 'Sent to socket; waiting for the core to accept it.';
});
reconnect.addEventListener('click', connect);
connect();

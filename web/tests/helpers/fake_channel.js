// A synchronous MessageChannel double: a message posted on one port reaches the other port's onmessage at
// once, so bridge tests can assert what crossed the channel without awaiting the event loop.
export class FakeMessageChannel {
    constructor() {
        const port = (name) => ({
            name, onmessage: null, closed: false, other: null,
            postMessage(data, transfer = []) {
                if (this.closed || this.other.closed) return;
                this.other.onmessage?.({ data, ports: [], transfer });
            },
            close() { this.closed = true; this.other.closed = true; },
            start() {},
        });
        this.port1 = port('port1');
        this.port2 = port('port2');
        this.port1.other = this.port2;
        this.port2.other = this.port1;
    }
}

export const srcdocNonce = (srcdoc) => (String(srcdoc || '').match(/"nonce":"([^"]+)"/) || [])[1];

// The frame side of a mounted bridge in a DOM double: it says hello as the bootstrap does (the srcdoc's nonce
// and a fresh channel), then `send` posts on that channel and `replies` records what the parent answered there
// (its ready notice aside). `emit` delivers a window message event to the page.
export function joinFrame(frame, emit, { nonce = srcdocNonce(frame.srcdoc) } = {}) {
    const channel = new FakeMessageChannel();
    const replies = [];
    channel.port1.onmessage = (event) => { if (event.data?.type !== 'ouro-widget-ready') replies.push(event.data); };
    emit({ source: frame.contentWindow, data: { type: 'ouro-widget-hello', nonce }, ports: [channel.port2] });
    return { nonce, replies, channel, send: (data) => channel.port1.postMessage({ nonce, ...data }) };
}

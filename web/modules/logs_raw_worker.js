// A live event without a file reference has only the received snapshot. Serialize it on explicit Raw opening,
// outside the UI thread; the controller terminates this worker on completion or cancellation and pages the Blob.
globalThis.onmessage = ({ data }) => {
    try {
        const text = JSON.stringify(data, null, 2);
        if (typeof text !== 'string') throw new Error('this event has no JSON form');
        globalThis.postMessage({ blob: new Blob([text], { type: 'application/json' }) });
    } catch (error) { globalThis.postMessage({ error: error.message }); }
};

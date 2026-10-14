import test from 'node:test';
import assert from 'node:assert/strict';
import { mountVoiceInput, voiceCandidate } from '../modules/voice_input.js';

const row = {
    name: 'whisper-voice', enabled: true, executable_review: true,
    desired_live: true, live_loaded: true, content_hash: 'sha',
    voice_input: { version: 1, route: 'transcribe', route_url: '/api/extensions/whisper-voice/transcribe',
        mime_types: ['audio/webm'], max_bytes: 393216 },
};

class Node {
    constructor() { this.attrs = {}; this.handlers = {}; this.dataset = {}; this.textContent = ''; }
    setAttribute(k, v) { this.attrs[k] = v; }
    addEventListener(k, fn) { this.handlers[k] = fn; }
    removeEventListener(k) { delete this.handlers[k]; }
    before(...children) { this.children = children; }
    remove() { this.removed = true; }
    click() { this.handlers.click?.(); }
}

class Recorder {
    static isTypeSupported(type) { return type === 'audio/webm'; }
    constructor(stream) { this.stream = stream; this.state = 'inactive'; }
    start() { this.state = 'recording'; }
    stop() {
        this.state = 'inactive';
        this.ondataavailable?.({ data: new Blob([new Uint8Array([0x1a, 0x45, 0xdf, 0xa3, 1])]) });
        this.onstop?.();
    }
}

async function tick() { await new Promise(resolve => setTimeout(resolve, 0)); }

function fixture({ index = () => ({ skills: [row] }), result = () => ({ text: 'hello world' }),
    deny = false, RecorderClass = Recorder } = {}) {
    const previous = globalThis.document;
    const document = new Node();
    document.createElement = () => new Node();
    globalThis.document = document;
    const input = { value: 'old draft' };
    document.activeElement = input;
    const anchor = new Node();
    let stops = 0;
    const media = { getUserMedia: async () => {
        if (deny) throw new Error('NotAllowedError');
        return { getTracks: () => [{ stop() { stops++; } }] };
    } };
    let requests = 0;
    const fetcher = async (url) => {
        requests++;
        if (url === '/api/extensions') return { ok: true, json: async () => index() };
        return { ok: true, json: async () => result() };
    };
    const drafts = [];
    const dispose = mountVoiceInput({ row, input, anchor, media, Recorder: RecorderClass, fetcher,
        onDraft: (...args) => drafts.push(args) });
    return { dispose: () => { dispose(); globalThis.document = previous; },
        suspend: () => dispose.suspend?.(), input, anchor, drafts,
        document, stops: () => stops, requests: () => requests };
}

test('only a live reviewed enabled row is selected', () => {
    assert.equal(voiceCandidate({ skills: [{ ...row, enabled: false }] }), null);
    assert.equal(voiceCandidate({ skills: [{ ...row, executable_review: false }] }), null);
    assert.equal(voiceCandidate({ skills: [{ ...row, live_loaded: false }] }), null);
    assert.equal(voiceCandidate({ skills: [row] }).name, 'whisper-voice');
});

test('a first incompatible skill does not hide a later recordable voice skill', () => {
    const mp4 = { ...row, name: 'mp4-voice', voice_input: {
        ...row.voice_input, route_url: '/api/extensions/mp4-voice/transcribe', mime_types: ['audio/mp4'],
    } };
    const candidates = { skills: [row, mp4] };
    assert.equal(voiceCandidate(candidates, type => type === 'audio/mp4')?.name, 'mp4-voice');
    assert.equal(voiceCandidate(candidates, type => type === 'audio/webm')?.name, 'whisper-voice');
    assert.equal(voiceCandidate(candidates, () => false), null);
});

test('explicit stop yields a draft only; disposal stops tracks and keyboard', async () => {
    const f = fixture();
    try {
        const button = f.anchor.children[0];
        button.click();
        await tick();
        assert.equal(button.dataset.busy, 'true');
        button.click();
        await tick();
        assert.deepEqual(f.drafts, [['hello world', 'old draft']]);
        assert.equal(f.stops(), 1);
        assert.equal(f.requests(), 3); // preflight, POST, postflight
        assert.equal(typeof f.document.handlers.keydown, 'function');
    } finally { f.dispose(); }
    assert.equal(f.document.handlers.keydown, undefined);
});

test('revocation before result or microphone denial never mutates draft', async () => {
    let current = { skills: [row] };
    const f = fixture({ index: () => current });
    try {
        f.anchor.children[0].click(); await tick();
        current = { skills: [] };
        f.anchor.children[0].click(); await tick();
        assert.equal(f.drafts.length, 0);
        assert.equal(f.input.value, 'old draft');
    } finally { f.dispose(); }
    const denied = fixture({ deny: true });
    try {
        denied.anchor.children[0].click(); await tick();
        assert.equal(denied.drafts.length, 0);
        assert.equal(denied.input.value, 'old draft');
    } finally { denied.dispose(); }
});


test('silent recording reports no speech without changing the draft', async () => {
    const f = fixture({ result: () => ({ text: '', status: 'no_speech' }) });
    try {
        f.anchor.children[0].click(); await tick();
        f.anchor.children[0].click(); await tick();
        assert.equal(f.input.value, 'old draft');
        assert.deepEqual(f.drafts, []);
        assert.match(f.anchor.children[1].textContent, /No speech detected/);
    } finally { f.dispose(); }
});


test('hide suspends microphone without destroying the draft or affordance', async () => {
    const f = fixture();
    try {
        const button = f.anchor.children[0];
        button.click(); await tick();
        assert.equal(button.dataset.busy, 'true');
        f.suspend();
        assert.equal(f.stops(), 1);
        assert.equal(button.dataset.busy, 'false');
        assert.equal(f.input.value, 'old draft');
        assert.equal(f.drafts.length, 0);
        button.click(); await tick(); // same mounted control can start again
        assert.equal(button.dataset.busy, 'true');
    } finally { f.dispose(); }
});

test('late onstop from cancelled recording cannot release the next microphone', async () => {
    class DelayedRecorder extends Recorder {
        static instances = [];
        constructor(stream) { super(stream); DelayedRecorder.instances.push(this); }
        stop() {
            this.state = 'inactive';
            this.ondataavailable?.({ data: new Blob([new Uint8Array([0x1a, 0x45, 0xdf, 0xa3, 1])]) });
            this.deliverStop = this.onstop;
        }
    }
    const f = fixture({ RecorderClass: DelayedRecorder });
    try {
        const button = f.anchor.children[0];
        button.click(); await tick();
        button.click();
        const old = DelayedRecorder.instances[0];
        button.click(); // cancel while waiting for onstop
        button.click(); await tick();
        const next = DelayedRecorder.instances[1];
        old.deliverStop(); await tick();
        assert.equal(next.state, 'recording');
        assert.equal(button.dataset.busy, 'true');
        assert.equal(f.drafts.length, 0);
    } finally { f.dispose(); }
});

test('late no-speech reply after cancel cannot repaint a new recording', async () => {
    let fulfill;
    const f = fixture({ result: () => new Promise(resolve => { fulfill = resolve; }) });
    try {
        const button = f.anchor.children[0];
        button.click(); await tick();
        button.click(); await tick();
        assert.equal(typeof fulfill, 'function');
        button.click(); // cancel pending transcription
        button.click(); await tick(); // new recording
        fulfill({ text: '', status: 'no_speech' }); await tick();
        assert.equal(button.dataset.busy, 'true');
        assert.match(f.anchor.children[1].textContent, /Recording/);
        assert.equal(f.input.value, 'old draft');
    } finally { f.dispose(); }
});

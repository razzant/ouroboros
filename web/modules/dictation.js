/** Dictation lifecycle without DOM. The owner of this controller cancels it on
 * room hide/destroy, Panic and provider removal. Only onTranscript may offer a
 * draft edit; the controller never sends a message.
 */
import * as pageCapture from './media_capture.js';
import { prepareRecorder } from './voice_recorder.js';
import { transcribeAudio } from './speech_providers.js';

export function transcriptInsertion(before, current, text) {
    if (!text?.trim()) return { kind: 'empty' };
    // Revision catches editing and then restoring the same words while STT runs.
    const unchanged = before.value === current.value && before.revision === current.revision;
    return unchanged ? { kind: 'insert', text: text.trim(), start: before.start, end: before.end }
        : { kind: 'offer', text: text.trim(), actions: ['Insert', 'Copy'] };
}

export function createDictation({ capture = pageCapture, env = globalThis, readDraft,
    onState = () => {}, onTranscript = () => {}, notice = () => {}, fetchImpl,
    makeRecorder = prepareRecorder, transcribe = transcribeAudio } = {}) {
    let active = null;
    let state = 'idle';
    function present(next, facts = {}) { state = next; onState({ state, ...facts }); }
    function release(run) {
        run.lease?.signal.removeEventListener('abort', run.lost);
        run.lease?.release();
        run.lease = null;
    }
    function cancel(reason = 'cancelled') {
        const run = active;
        if (!run) return;
        active = null;
        run.abort.abort(reason);
        env.clearTimeout(run.timeout);
        release(run);
        run.pending?.release?.(reason);
        run.client?.cancel();
        run.prepared?.cancel();
        present('idle', { reason });
    }
    function failed(run, error) {
        if (active !== run) return;
        cancel(error.code || 'failed');
        notice(error.message || String(error));
    }
    function start(provider, { lang, owner = 'dictation' } = {}) {
        if (active) return Promise.resolve({ code: 'in_use' });
        const run = { provider, abort: new AbortController(), lost: null, stopping: false };
        active = run;
        present('permission');
        let pending;
        try {
            if (provider.runtime === 'server') run.prepared = makeRecorder(provider, { env,
                onLimit(code) {
                    if (active !== run) return;
                    notice(code === 'max_seconds' ? 'Recording stopped at the engine’s time limit.'
                        : 'Recording stopped at the engine’s size limit.');
                    void stop();
                } });
            pending = capture.acquire({ kind: 'audio', owner, label: 'Dictation', capture: provider.runtime !== 'client' });
            run.pending = pending;
            if (pending.code) {
                cancel(pending.code);
                notice(pageCapture.captureReason(pending.code, pending.label));
                return Promise.resolve(pending);
            }
            // The logical lease is synchronous: recognition.start() also needs this gesture.
            if (provider.runtime === 'client') {
                run.lease = pending;
                run.client = provider.impl.start({ lang, signal: run.abort.signal });
            }
        } catch (error) { failed(run, error); return Promise.resolve({ code: error.code || 'failed' }); }
        return Promise.resolve(pending).then(lease => {
            if (active !== run) { lease.release?.(); return { code: 'cancelled' }; }
            if (lease.code) {
                cancel(lease.code);
                notice(pageCapture.captureReason(lease.code, lease.label));
                return lease;
            }
            run.lease = lease;
            run.lost = () => {
                const reason = lease.signal.reason;
                cancel(reason);
                if (reason === 'lost') notice(pageCapture.captureReason('lost'));
            };
            lease.signal.addEventListener('abort', run.lost, { once: true });
            if (lease.signal.aborted) { run.lost(); return { code: 'lost' }; }
            if (lease.fellBack) notice('The chosen microphone is unavailable. Using the system microphone.');
            if (run.prepared) {
                run.recorder = run.prepared.start(lease.stream);
                run.recorder.result.catch(error => { if (!run.stopping) failed(run, error); });
            }
            // The stream (absent for a client recognizer) feeds the face's level dot; the face never keeps it.
            present('recording', { startedAt: Date.now(), stream: lease.stream || null });
            return { ok: true };
        }).catch(error => { failed(run, error); return { code: error.code || 'failed' }; });
    }
    async function stop() {
        const run = active;
        if (!run || run.stopping || state !== 'recording') return;
        run.stopping = true;
        const snapshot = { ...readDraft() };
        present('transcribing');
        let timer;
        try {
            let result;
            if (run.client) {
                result = await run.client.stop();
                release(run);
            } else {
                const audio = await run.recorder.stop();
                release(run);
                if (active !== run) return;
                timer = run.timeout = env.setTimeout(() => run.abort.abort('timeout'), 120000);
                result = await transcribe(run.provider, audio, { signal: run.abort.signal, fetchImpl });
            }
            if (active !== run || run.abort.signal.aborted) return;
            active = null;
            present('idle');
            if (result.no_speech) notice('No speech detected. The draft is unchanged.');
            else onTranscript(transcriptInsertion(snapshot, readDraft(), result.text));
        } catch (error) {
            if (run.abort.signal.reason === 'timeout') error = new Error('Transcription took too long. Try again.');
            failed(run, error);
        } finally { env.clearTimeout(timer); }
    }
    return { start, stop, cancel, suspend: () => cancel('hidden'), destroy: () => cancel('destroyed'),
        get state() { return state; } };
}

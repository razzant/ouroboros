// Optional composer input, with no controls or listeners while no live skill
// advertises voice_input. The server rechecks the executable route on dispatch.
const MAX_RECORD_SECONDS = 60;
const AUDIO_BITS_PER_SECOND = 32000;

export function voiceCandidate(index, canRecord = () => true) {
    const rows = Array.isArray(index?.skills) ? index.skills : Array.isArray(index?.extensions) ? index.extensions : [];
    return rows.find(row => row.voice_input && row.enabled && row.executable_review
        && row.desired_live && row.live_loaded && row.voice_input.route_url
        && row.voice_input.mime_types?.some(type => canRecord(type))) || null;
}

// The catalog is a passive view of reviewed live registration. Missing or
// unreadable status withdraws the affordance; dispatch checks it again.
export function bindVoiceInput({ input, anchor, fetcher, onWs, onChange, showToast }) {
    let alive = true;
    let dispose = null;
    let identity = '';
    let read = 0;
    async function refresh() {
        const ticket = ++read;
        let selected = null;
        try {
            const response = await fetcher('/api/extensions', { cache: 'no-store' });
            if (response.ok) selected = voiceCandidate(await response.json(),
                type => globalThis.MediaRecorder?.isTypeSupported?.(type));
        } catch { /* An unavailable capability is not permission to record. */ }
        if (!alive || ticket !== read) return;
        const next = selected ? `${selected.name}:${selected.content_hash}:${JSON.stringify(selected.voice_input)}` : '';
        if (identity === next) return;
        dispose?.();
        dispose = null;
        identity = next;
        if (selected) dispose = mountVoiceInput({
            row: selected, input, anchor, fetcher, onChange,
            onDraft(text, original) {
                if (!alive) return;
                input.value = input.value
                    ? `${input.value}${input.value.endsWith('\n') ? '' : '\n'}${text}`
                    : text;
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.focus();
                if (input.value !== `${original}${text}`)
                    showToast('Voice text added to the current draft; review before sending.', 'info');
            },
        });
    }
    onWs('extension_lifecycle', refresh);
    window.addEventListener('ouro:page-shown', refresh);
    void refresh();
    const teardown = () => {
        alive = false;
        read++;
        dispose?.();
        window.removeEventListener('ouro:page-shown', refresh);
    };
    teardown.suspend = () => dispose?.suspend?.();
    return teardown;
}

export function mountVoiceInput({ row, input, anchor, fetcher = fetch, media = navigator.mediaDevices,
    Recorder = globalThis.MediaRecorder, onDraft, onChange = () => {} }) {
    if (!row?.voice_input || !anchor || !input || !media?.getUserMedia || !Recorder) return () => {};
    const voice = row.voice_input;
    const mime = voice.mime_types.find(type => Recorder.isTypeSupported?.(type));
    if (!mime) return () => {};
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chat-voice-btn';
    button.title = 'Record voice (Ctrl+Shift+M)';
    button.setAttribute('aria-label', button.title);
    button.innerHTML = '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 10a7 7 0 0 0 14 0M12 17v5m-4 0h8"/></svg>';
    const status = document.createElement('span');
    status.className = 'chat-voice-status';
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    anchor.before(button, status);
    let alive = true;
    let generation = 0;
    let recorder = null;
    let stream = null;
    let abort = null;
    let requesting = false;
    let transcribing = false;
    let timer = null;
    let chunks = [];
    let recordedBytes = 0;
    let initialDraft = '';
    let startedAt = 0;

    function present(text, busy = false) {
        if (!alive) return;
        status.textContent = text;
        button.dataset.busy = busy ? 'true' : 'false';
        button.setAttribute('aria-label', busy
            ? transcribing ? 'Cancel voice input (Escape)'
                : recorder?.state === 'recording' ? 'Stop voice recording (Ctrl+Shift+M)' : 'Cancel voice input (Escape)'
            : 'Record voice (Ctrl+Shift+M)');
        onChange();
    }
    function release() {
        if (timer) clearTimeout(timer);
        timer = null;
        stream?.getTracks().forEach(track => track.stop());
        stream = null;
        recorder = null;
        chunks = [];
        recordedBytes = 0;
    }
    function cancel(reason = 'Voice recording cancelled.') {
        generation++;
        abort?.abort();
        abort = null;
        requesting = false;
        transcribing = false;
        const old = recorder;
        if (old?.state === 'recording') {
            old.onstop = null;
            try { old.stop(); } catch {}
        }
        release();
        present(reason);
    }
    async function start() {
        const ticket = ++generation;
        requesting = true;
        initialDraft = input.value;
        present('Requesting microphone…', true);
        try {
            // Never infer authority from the original page load. A disabled or
            // stale skill cannot start a new recording on a cached button.
            const fresh = await fetcher('/api/extensions', { cache: 'no-store' });
            if (!fresh.ok) throw new Error('Voice skill status unavailable');
            const selected = voiceCandidate(await fresh.json(), type => Recorder.isTypeSupported?.(type));
            if (!selected || selected.name !== row.name || selected.content_hash !== row.content_hash
                || selected.voice_input.route_url !== voice.route_url) {
                throw new Error('Voice skill no longer available');
            }
            if (ticket !== generation || !alive) return;
            const opened = await media.getUserMedia({ audio: true });
            if (ticket !== generation || !alive) {
                opened.getTracks().forEach(track => track.stop());
                return;
            }
            stream = opened;
            requesting = false;
            const current = new Recorder(opened, { mimeType: mime, audioBitsPerSecond: AUDIO_BITS_PER_SECOND });
            recorder = current;
            chunks = [];
            recordedBytes = 0;
            current.ondataavailable = event => {
                if (ticket !== generation || !alive || !event.data?.size) return;
                recordedBytes += event.data.size;
                chunks.push(event.data);
                if (recordedBytes > voice.max_bytes) cancel('Recording too large. Try a shorter phrase.');
            };
            current.onerror = () => { if (ticket === generation) cancel(); };
            current.onstop = () => { void transcribe(ticket); };
            current.start(250);
            startedAt = Date.now();
            timer = setTimeout(() => stop(), MAX_RECORD_SECONDS * 1000);
            present('Recording… Mic or Ctrl+Shift+M to stop; Escape cancels.', true);
        } catch (error) {
            if (ticket !== generation || !alive) return;
            requesting = false;
            release();
            present(`Voice unavailable: ${error?.message || error}`);
        }
    }
    function stop() {
        if (!recorder || recorder.state !== 'recording') return;
        if (timer) clearTimeout(timer);
        timer = null;
        transcribing = true;
        present('Transcribing locally…', true);
        recorder.stop();
    }
    async function transcribe(ticket) {
        // MediaRecorder.onstop can arrive after Escape or a new recording.
        // The old callback must not release the new stream or its timer.
        if (!alive || ticket !== generation) return;
        const total = recordedBytes;
        const blob = new Blob(chunks, { type: mime });
        release();
        if (!total || total > voice.max_bytes || Date.now() - startedAt > (MAX_RECORD_SECONDS + 2) * 1000) {
            transcribing = false;
            present('Recording too long or too large. Try a shorter phrase.');
            return;
        }
        const controller = new AbortController();
        abort = controller;
        const deadline = setTimeout(() => controller.abort(), 120000);
        try {
            const response = await fetcher(voice.route_url, {
                method: 'POST', headers: { 'Content-Type': mime }, body: blob,
                signal: controller.signal, cache: 'no-store',
            });
            const result = await response.json();
            if (!alive || ticket !== generation) return;
            if (response.ok && result?.status === 'no_speech' && result.text === '') {
                present('No speech detected. The draft is unchanged.');
                return;
            }
            if (!response.ok || typeof result?.text !== 'string' || !result.text.trim()) {
                throw new Error(result?.error?.message || 'No speech recognized');
            }
            const check = await fetcher('/api/extensions', { cache: 'no-store' });
            if (!check.ok) throw new Error('Could not confirm voice skill remains enabled');
            const live = voiceCandidate(await check.json(), type => Recorder.isTypeSupported?.(type));
            if (!live || live.name !== row.name || live.content_hash !== row.content_hash || !alive
                || ticket !== generation) throw new Error('Voice skill changed during transcription');
            onDraft(result.text.trim(), initialDraft);
            present('Transcript added to draft. Review it before sending.');
        } catch (error) {
            if (ticket === generation && alive) present(`Voice transcription failed: ${error?.message || error}`);
        } finally {
            clearTimeout(deadline);
            if (ticket === generation) transcribing = false;
            if (abort === controller) abort = null;
        }
    }
    function toggle() {
        if (recorder?.state === 'recording') stop();
        else if (abort || requesting || transcribing) cancel();
        else void start();
    }
    function hotkey(event) {
        if (document.activeElement !== input && document.activeElement !== button) return;
        if (!alive || event.isComposing || event.repeat) return;
        if (event.key === 'Escape' && (requesting || recorder?.state === 'recording' || transcribing)) {
            event.preventDefault();
            cancel();
            return;
        }
        if (!event.ctrlKey || !event.shiftKey || event.key.toLowerCase() !== 'm' || event.altKey) return;
        event.preventDefault();
        toggle();
    }
    button.addEventListener('click', toggle);
    document.addEventListener('keydown', hotkey);
    const teardown = () => {
        if (!alive) return;
        cancel();
        alive = false;
        document.removeEventListener('keydown', hotkey);
        button.removeEventListener('click', toggle);
        button.remove();
        status.remove();
        onChange();
    };
    teardown.suspend = () => {
        if (requesting || recorder?.state === 'recording' || transcribing || abort)
            cancel('Voice input cancelled when chat was hidden.');
    };
    return teardown;
}

/** Settings › Models › Local model (docs/DESIGN.md "Settings"): what Start launches, as rows that save themselves, and
 *  the local model's own Start, Stop, Test and Install. A saved change applies when the local model is stopped and
 *  started again (its row says so); nothing here asks for a restart of Ouroboros. The status is read while this page
 *  is on screen. */
import { apiFetch, fetchJson } from './api_client.js';
import { bindSettingRows, setButtonBusy, settingsRow } from './settings_controls.js';
import { registerSettingsPage } from './settings_pages.js';
import { checkerUses, deploymentOf } from './settings_policy.js';
import { rowsOf } from './settings_rows.js';
import { setResult } from './ui_primitives.js';

const PAGE = 'models/local';
const POLL_MS = 3000;
// The fields a deployment's checker binds when it answers from this model (`deployment_policy.local_launch_refusal`).
const FIXED_KEYS = ['LOCAL_MODEL_SOURCE', 'LOCAL_MODEL_FILENAME', 'LOCAL_MODEL_PORT', 'LOCAL_MODEL_CHAT_FORMAT'];

/** A deployment checker answering locally binds this server's identity: those fields read only, its note shown
 *  (the Start endpoint refuses any other launch). */
export function applyLocalModelPolicy(root, fixed) {
    for (const key of FIXED_KEYS) {
        const input = root.querySelector(`[data-setting-row="${key}"] [data-setting-control]`);
        if (input) input.readOnly = fixed;
    }
    const note = root.querySelector('[data-local-model-fixed]');
    if (note) note.hidden = !fixed;
}

/** The local model's state in words: `{ text, tone }` from one status answer. */
export function localModelStatusWords(d = {}) {
    const status = String(d.status || 'offline');
    const installing = d.runtime_status === 'installing';
    let text = status.charAt(0).toUpperCase() + status.slice(1);
    if (status === 'ready' && d.context_length) text += ` · context ${d.context_length}`;
    if (status === 'downloading' && d.download_progress != null) text += ` · ${Math.round(d.download_progress * 100)}%`;
    if (installing) text = 'Installing the local runtime…';
    if (d.runtime_status === 'install_ok') text += ' · runtime installed';
    if (d.runtime_status === 'install_error') text += ' · the runtime install failed';
    if (d.error && !installing) text += ` · ${d.error}`;
    return { text, tone: status === 'ready' ? 'ok' : status === 'error' || d.runtime_status === 'install_error' ? 'danger' : 'muted' };
}

function mountLocalModel(container, { path, ctx }) {
    const rows = (fold) => rowsOf(PAGE, { fold }).map((row) => settingsRow(row)).join('');
    container.innerHTML = `<p class="settings-page-lead">A model file that runs on this computer. Fill this in only if you want one.</p>
        <section class="ui-card settings-card">
            <div class="settings-rows">${rows(false)}</div>
            <p class="settings-card-note" data-local-model-fixed hidden>Your organization’s Safety Supervisor answers from this model: its model, file, port and chat format are set by your organization. Start and Stop stay yours.</p>
        </section>
        <section class="ui-card settings-card">
            <h3 class="settings-card-title">Run it</h3>
            <p class="ui-status local-model-state" id="local-model-status" role="status" aria-live="polite" data-tone="muted">Offline</p>
            <p class="settings-card-lead" data-local-model-pending hidden></p>
            <div id="local-model-progress-wrap" class="local-model-progress-wrap" role="progressbar" aria-label="Download" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" hidden>
                <div id="local-model-progress-bar" class="local-model-progress-bar"></div></div>
            <div class="ui-action-row">
                <button type="button" class="btn" data-primary id="btn-local-start">Start</button>
                <button type="button" class="btn btn-default" id="btn-local-stop" disabled>Stop</button>
                <button type="button" class="btn btn-default" id="btn-local-test" disabled>Test tool calling</button>
                <button type="button" class="btn btn-default" id="btn-local-install-runtime" hidden>Install the local runtime</button>
                <span class="ui-result" id="local-model-action-status" role="status" aria-live="polite" aria-atomic="true"></span>
            </div>
            <pre class="ui-result-details" id="local-model-test-result" hidden></pre>
        </section>
        <section class="ui-card settings-card"><details class="ui-disclosure" data-settings-fold>
            <summary>More: GPU layers, port, chat format</summary>
            <div class="settings-rows">${rows(true)}</div></details></section>`;
    const el = (id) => container.querySelector(`#${id}`);
    const bound = bindSettingRows(container, { ask: ctx.ask });
    const [start, stop, test, install] = ['btn-local-start', 'btn-local-stop', 'btn-local-test', 'btn-local-install-runtime'].map(el);
    const action = (text, tone = 'muted') => setResult(el('local-model-action-status'), text ? { text, tone } : {});
    const testResult = (text) => Object.assign(el('local-model-test-result'), { textContent: text, hidden: !text });
    let alive = true;
    let stopPending = false;
    let ready = false;
    let startAfterInstall = false;
    let sequence = 0;
    const onScreen = () => alive && ctx.state?.activePage === 'settings' && ctx.state?.settingsPath === path && !globalThis.document?.hidden;
    // What Start launches: the fields as they stand now (a field commits when it is left, so these are the saved ones).
    const typed = (key) => String(container.querySelector(`[data-setting-row="${key}"] [data-setting-control]`)?.value ?? '').trim();
    const launchBody = () => ({
        source: typed('LOCAL_MODEL_SOURCE'), filename: typed('LOCAL_MODEL_FILENAME'), port: parseInt(typed('LOCAL_MODEL_PORT'), 10) || 8766,
        n_gpu_layers: parseInt(typed('LOCAL_MODEL_N_GPU_LAYERS'), 10), n_ctx: parseInt(typed('LOCAL_MODEL_CONTEXT_LENGTH'), 10) || 65536,
        chat_format: typed('LOCAL_MODEL_CHAT_FORMAT'),
    });

    async function readStatus() {
        if (!onScreen()) return;
        const mine = ++sequence;
        let d;
        try { d = await fetchJson('/api/local-model/status', { cache: 'no-store' }); } catch { return; }
        if (!alive || mine !== sequence) return;
        const installing = d.runtime_status === 'installing';
        const downloading = d.status === 'downloading';
        ready = d.status === 'ready';
        const words = localModelStatusWords(d);
        Object.assign(el('local-model-status'), { textContent: words.text }).dataset.tone = words.tone;
        // Saved settings the running model does not have yet, and facts it did not report: the server's own sentence.
        const pending = el('local-model-status').parentElement.querySelector('[data-local-model-pending]');
        Object.assign(pending, { textContent: d.settings_application?.summary || '', hidden: !d.settings_application?.summary });
        const wrap = el('local-model-progress-wrap');
        wrap.hidden = !(downloading && d.download_progress != null);
        if (!wrap.hidden) {
            const share = Math.round(d.download_progress * 100);
            el('local-model-progress-bar').style.width = `${share}%`;
            wrap.setAttribute('aria-valuenow', String(share));
        }
        stop.disabled = stopPending || !ready;
        test.disabled = !ready;
        start.disabled = installing || downloading;
        install.hidden = !(d.runtime_status === 'missing' || d.runtime_status === 'install_error');
        install.disabled = installing;
        if (d.runtime_status === 'install_error' && d.runtime_install_log) testResult(`Install failed:\n${d.runtime_install_log}`);
        if (d.runtime_status === 'install_ok' && startAfterInstall) { startAfterInstall = false; void startModel(); }
    }

    async function startModel() {
        const body = launchBody();
        if (!body.source) { action('Name a model first: a Hugging Face repository or a file on this computer.', 'danger'); return; }
        testResult('');
        action('');
        try {
            const resp = await apiFetch('/api/local-model/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            const data = await resp.json();
            if (resp.status === 412 && data.error === 'runtime_missing') {
                install.hidden = false;
                action('The local runtime is not installed. Install it, and the model starts by itself.', 'danger');
                testResult(`llama-cpp-python is not installed.\nBy hand: ${data.hint || 'pip install llama-cpp-python[server]'}`);
            } else if (data.error) action(`Not started: ${data.error}`, 'danger');
            else void readStatus();
        } catch (e) { action(`Not started: ${e.message}`, 'danger'); }
    }

    start.addEventListener('click', () => void startModel());
    stop.addEventListener('click', async () => {
        if (!alive || stopPending) return;
        stopPending = true;
        setButtonBusy(stop, true);
        action('Stopping…');
        try {
            const result = await fetchJson('/api/local-model/stop', { method: 'POST' }, { rejectOkFalse: true });
            if (!alive) return;
            if (result?.status !== 'stopped' || result?.error) throw new Error(result?.error || 'The server did not confirm Stop.');
            action('The local model stopped.', 'ok');
            await readStatus();
        } catch (e) { if (alive) action(`Stop failed: ${e.message}. You can press Stop again.`, 'danger'); }
        stopPending = false;
        if (alive) { setButtonBusy(stop, false); stop.disabled = !ready; }
    });
    test.addEventListener('click', async () => {
        action('Testing…');
        testResult('');
        try {
            const r = await (await apiFetch('/api/local-model/test', { method: 'POST' })).json();
            if (r.error) { action(`Test failed: ${r.error}`, 'danger'); return; }
            action(r.success ? 'Chat and tool calling work.' : 'The test did not pass.', r.success ? 'ok' : 'warn');
            testResult([`${r.chat_ok ? 'Passed' : 'Failed'}: basic chat${r.tokens_per_sec ? ` (${r.tokens_per_sec} tokens a second)` : ''}`,
                `${r.tool_call_ok ? 'Passed' : 'Failed'}: tool calling`, r.details && !r.success ? r.details : ''].filter(Boolean).join('\n'));
        } catch (e) { action(`Test failed: ${e.message}`, 'danger'); }
    });
    install.addEventListener('click', async () => {
        install.disabled = true;
        action('Installing the local runtime. This may take a few minutes…');
        startAfterInstall = Boolean(launchBody().source);
        try {
            const d = await (await apiFetch('/api/local-model/install-runtime', { method: 'POST' })).json();
            if (d.error) throw new Error(d.error);
        } catch (e) {
            action(`Install failed: ${e.message}`, 'danger');
            install.disabled = false;
        }
    });

    const read = async () => {
        try {
            const settings = await ctx.readSettings();
            if (alive) applyLocalModelPolicy(container, checkerUses(deploymentOf(settings?._meta), 'local'));
        } catch { /* the rows keep what they show */ }
        if (alive) bound.paint();
        void readStatus();
    };
    const onShown = (event) => { if (event.detail?.path === path) void read(); };
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    void read();
    const interval = setInterval(readStatus, POLL_MS);
    return { unmount() {
        alive = false;
        clearInterval(interval);
        globalThis.window?.removeEventListener('ouro:settings-page-shown', onShown);
        bound.dispose();
    } };
}

registerSettingsPage(PAGE, { title: 'Local model', mount: mountLocalModel });

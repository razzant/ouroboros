const status = document.querySelector('#status');
const save = document.querySelector('#save');
const check = document.querySelector('#check');

async function readSetup() {
    const response = await fetch('/api/onboarding/bootstrap', {cache: 'no-store'});
    if (!response.ok) throw new Error(`Setup read returned HTTP ${response.status}`);
    return response.json();
}

readSetup().then(({ready, bootstrap}) => {
    const {contract, initialState} = bootstrap;
    const endpoint = contract.providerFields.find(row => row.settingKey === 'OPENAI_COMPATIBLE_BASE_URL');
    const model = contract.modelSlots.find(row => row.settingKey === 'OUROBOROS_MODEL');
    document.querySelector('#endpoint').value = initialState[endpoint?.stateKey] || '';
    const selected = initialState[model?.stateKey] || '';
    document.querySelector('#model').value = selected.startsWith('openai-compatible::')
        ? selected.slice('openai-compatible::'.length) : '';
    status.textContent = ready ? 'This core is already configured. Saving changes its settings.'
        : 'Ready to configure. Reading this page has saved nothing.';
    save.disabled = false;
}).catch(error => { status.textContent = error.message; });

document.querySelector('#setup').addEventListener('submit', async event => {
    event.preventDefault();
    save.disabled = true;
    try {
        const response = await fetch('/api/onboarding/complete', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                OPENAI_COMPATIBLE_BASE_URL: document.querySelector('#endpoint').value.trim(),
                OPENAI_COMPATIBLE_API_KEY: '',
                OUROBOROS_MODEL: 'openai-compatible::' + document.querySelector('#model').value.trim(),
                skipSubscriptionPresets: true, allow_empty_review_pool: true,
            }),
        });
        const receipt = await response.json();
        if (!response.ok || receipt.ok !== true) {
            status.textContent = `${receipt.error || `HTTP ${response.status}`}. ` +
                (receipt.saved === false ? 'Nothing was saved.' : receipt.saved === true
                    ? 'Settings were saved; a later step failed.' : 'Whether settings were saved is unknown.');
            save.disabled = receipt.saved !== false;
            check.hidden = false;
            return;
        }
        if (receipt.status !== 'saved' || typeof receipt.restart_required !== 'boolean' ||
            typeof receipt.runtime_mode !== 'string') {
            throw new Error('Setup returned an incomplete completion receipt');
        }
        status.textContent = 'Settings saved.';
        // Carry the actual core receipt; a window closing is not a save receipt.
        if (window.parent !== window) {
            // Convention of the builtin/copied parent. Other parents choose their own handoff.
            window.parent.postMessage({type: 'ouroboros:onboarding-complete',
                restart_required: receipt.restart_required, runtime_mode: receipt.runtime_mode}, location.origin);
        } else if (window.pywebview?.api?.onboarding_finished) {
            await window.pywebview.api.onboarding_finished({ok: true,
                restart_required: receipt.restart_required, runtime_mode: receipt.runtime_mode});
        } else if (receipt.restart_required) {
            document.querySelector('#restart').hidden = false;
        } else {
            location.assign('/');
        }
    } catch (error) {
        status.textContent = `${error.message}. Save outcome is unknown; inspect current status before another save.`;
        check.hidden = false;
    }
});

check.addEventListener('click', async () => {
    try {
        const state = await readSetup();
        status.textContent = state.ready
            ? 'The core is configured. Inspect its settings before saving again; readiness is not a receipt for this attempt.'
            : 'The core still needs setup. No save is retried automatically.';
    } catch (error) { status.textContent = error.message; }
});

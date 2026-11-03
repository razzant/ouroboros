import assert from 'node:assert/strict';
import test from 'node:test';
import { installDomDouble } from './dom_double.js';
import { initEvolution, runtimeTone } from '../modules/evolution.js';
import { applyPayload } from '../modules/i18n.js';
import { WS } from '../modules/ws.js';

const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(setImmediate); };

test('enabled consciousness keeps historical pause and unknown wake outcomes visible instead of off', async () => {
    const env = installDomDouble();
    // The labels come from i18n's entries; its page overlay (a walker and observer over real text nodes) stays out of the double.
    const translate = (payload) => { const body = env.doc.body; env.doc.body = null; try { applyPayload(payload); } finally { env.doc.body = body; } };
    const savedFetch = globalThis.fetch;
    let status = 'wake_paused';
    const details = {
        wake_paused: 'The last wake-up returned while paused; its task card shows the current state. Next wake check at 18:00.',
        wake_outcome_unknown: "The last wake-up's outcome is unconfirmed; next check at 18:00.",
    };
    globalThis.fetch = async (url) => Response.json(String(url).includes('evolution-data')
        ? { versions: [], refresh: { state: 'idle' }, install: {} }
        : { bg_consciousness_enabled: status !== 'disabled', bg_consciousness_state: { status, detail: details[status] || '' } });
    let dispose;
    try {
        const mount = env.doc.createElement('div');
        env.doc.body.append(mount);
        const ws = new WS('ws://unused');
        ({ dispose } = initEvolution({ ws, mount, state: { activePage: 'dashboard', dashboardActiveSubtab: 'evolution' } }));
        const line = () => env.doc.getElementById('evo-consciousness');
        // A paused or unknown wake-up goes on by itself: the neutral sentence, never the owner's amber.
        for (const [next, label, tone] of [['wake_paused', 'paused', 'neutral'],
            ['wake_outcome_unknown', 'unknown', 'neutral'], ['sleeping', 'sleeping', 'ok'], ['due', 'due', 'ok'],
            ['future_status', 'off', 'neutral'], ['disabled', 'off', 'neutral']]) {
            status = next;
            ws.emit('open');
            await flush();
            assert.equal(line().textContent, `Consciousness ${label}`);
            assert.equal(line().getAttribute('data-tone'), tone);
            assert.equal(env.doc.getElementById('evo-consciousness-detail').textContent, details[status] || '', 'historical wording stays intact');
        }
        translate({ language: 'ru', english: false, entries: {
            'code:consciousness.wake_paused': { text: 'приостановлено', provenance: 'imported' },
            'code:consciousness.wake_outcome_unknown': { text: 'исход неизвестен', provenance: 'imported' },
            'code:consciousness.due': { text: 'пора', provenance: 'imported' },
        } });
        for (const [next, label] of [['wake_paused', 'приостановлено'], ['wake_outcome_unknown', 'исход неизвестен'], ['due', 'пора']]) {
            status = next; ws.emit('open'); await flush();
            assert.equal(line().textContent, `Consciousness ${label}`);
        }
    } finally {
        dispose?.();
        translate({ language: 'en', english: true });
        globalThis.fetch = savedFetch;
        env.restore();
    }
});

test("a runtime tone is amber only for the owner's move: a first conversation, the owner's chat, a spending stop", () => {
    const tones = {
        ok: ['running', 'queued', 'idle_ready', 'thinking', 'sleeping', 'due'],
        warn: ['waiting_for_first_conversation', 'waiting_for_owner_chat', 'budget_blocked', 'budget_stopped'],
        error: ['paused_failures', 'error_backoff', 'allowance_exhausted', 'allowance_unknown', 'wake_failed', 'wake_rejected'],
        // What goes on by itself, or is off, keeps the neutral tone.
        neutral: ['waiting_for_idle', 'waiting_for_restart_verify', 'paused', 'starting', 'state_unknown', 'unknown',
            'wake_paused', 'wake_outcome_unknown', 'disabled', 'stopped', 'accounting_unavailable', ''],
    };
    for (const [tone, statuses] of Object.entries(tones)) {
        for (const status of statuses) assert.equal(runtimeTone(status), tone, status);
    }
});

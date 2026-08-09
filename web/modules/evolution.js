import { chartColorAlpha, escapeHtmlText, formatUsd2, readThemeTokens } from './utils.js';
import { apiFetch } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';

/**
 * Ask for the evolution-campaign objective (pure decision helper, node-tested
 * with an injected dialog). window.prompt is dead on the macOS desktop shell
 * (pywebview/WKWebView answers null silently), which made Start campaign a
 * one-click PAID launch with a hidden-empty objective and no way to cancel.
 * Semantics (owner-approved Б1-8): cancel/Escape/backdrop = do NOT start;
 * confirmed-empty = start with the default autonomous objective downstream
 * (evolution_lifecycle.py); confirmed-text = start with that objective.
 */
export async function promptCampaignObjective({ dialogImpl = openConfirmDialog } = {}) {
    const answer = await dialogImpl({
        title: 'Start evolution campaign',
        body: 'Evolution campaign objective (optional). Leave empty to let Ouroboros pick its own autonomous objective.',
        input: true,
        confirmLabel: 'Start campaign',
    });
    if (!answer?.confirmed) return { confirmed: false, objective: '' };
    return { confirmed: true, objective: String(answer.value || '').trim() };
}

// Chart.js paints on a canvas and cannot resolve CSS variables, so every colour
// and the mono stack are READ from the :root design tokens (web/style.css)
// through the shared `readThemeTokens` seam (web/modules/utils.js) — never
// pasted here as hexes. Every name below must be DEFINED in `:root`: the seam
// returns computed values (an alias resolves through fine), but an undefined
// token comes back as `''` and a canvas handed `''` drops the series silently.
// Six series need six DISTINGUISHABLE lanes, so the ramp names ROLE tokens
// (task red, owner blue, ok green, notice amber, project teal, accent) instead
// of shades of one hue — a neutral text ladder collapses into "three grays" at
// 2px stroke width on a canvas.
const CHART_TOKENS = {
    seriesRamp: [
        '--accent-light',
        '--user',
        '--green',
        '--amber',
        '--project',
        '--accent',
    ],
    axis: '--text-muted',
    axisTitle: '--text-secondary',
    grid: '--divider',
    surface: '--bg-elevated',
    border: '--surface-border',
    strong: '--text-primary',
    mono: '--font-mono',
};

const CHART_SCALAR_KEYS = ['axis', 'axisTitle', 'grid', 'surface', 'border', 'strong', 'mono'];

/** Resolve the evolution chart theme from the live design tokens. */
export function evolutionChartTheme(root = document.documentElement) {
    const ramp = CHART_TOKENS.seriesRamp;
    const values = readThemeTokens(
        [...ramp, ...CHART_SCALAR_KEYS.map((key) => CHART_TOKENS[key])],
        root,
    );
    const theme = { series: values.slice(0, ramp.length) };
    CHART_SCALAR_KEYS.forEach((key, idx) => {
        theme[key] = values[ramp.length + idx];
    });
    return theme;
}

export function initEvolution({ ws, state, mount }) {
    const page = document.createElement('div');
    page.id = 'page-evolution';
    page.className = 'settings-embedded-content settings-evolution-panel';
    page.innerHTML = `
        <div id="evo-chart-content" class="evolution-container">
            <div class="evo-runtime-card">
                <div class="evo-runtime-head">
                    <div>
                        <div class="section-title">Runtime Status</div>
                        <div id="evo-runtime-detail" class="evo-runtime-detail">Loading evolution and consciousness state...</div>
                    </div>
                    <div class="evo-runtime-pills">
                        <span id="evo-mode-pill" class="evo-runtime-pill">Evolution</span>
                        <span id="evo-bg-pill" class="evo-runtime-pill">Consciousness</span>
                    </div>
                    <div class="evo-runtime-pills evo-runtime-controls">
                        <button id="evo-start" class="btn btn-primary btn-sm evo-refresh-btn" type="button">Start campaign</button>
                        <button id="evo-stop" class="btn btn-default btn-sm evo-refresh-btn" type="button">Pause</button>
                        <button id="evo-refresh" class="btn btn-default btn-sm evo-refresh-btn" type="button">Refresh</button>
                        <span id="evo-status" class="status-badge">Loading...</span>
                    </div>
                </div>
                <div id="evo-runtime-meta" class="evo-runtime-meta"></div>
                <div id="evo-campaign-detail" class="evo-runtime-detail"></div>
            </div>
            <div class="evo-chart-wrap">
                <canvas id="evo-chart"></canvas>
            </div>
            <div id="evo-tags-list" class="evo-tags-list"></div>
        </div>
    `;
    mount.appendChild(page);

    function isEvolutionVisible() {
        return state.activePage === 'dashboard' && state.dashboardActiveSubtab === 'evolution';
    }

    // -----------------------------------------------------------------------
    // Evolution chart + runtime state
    // -----------------------------------------------------------------------
    let evoChart = null;
    let loadSequence = 0;
    let chartLoaded = false;
    const refreshBtn = document.getElementById('evo-refresh');
    const startBtn = document.getElementById('evo-start');
    const stopBtn = document.getElementById('evo-stop');
    const statusBadge = document.getElementById('evo-status');
    const runtimeDetail = document.getElementById('evo-runtime-detail');
    const runtimeMeta = document.getElementById('evo-runtime-meta');
    const campaignDetail = document.getElementById('evo-campaign-detail');
    const evolutionPill = document.getElementById('evo-mode-pill');
    const consciousnessPill = document.getElementById('evo-bg-pill');
    const tagsList = document.getElementById('evo-tags-list');

    // Dataset order is the identity of a series (the tooltip resolves a key by
    // dataset index), so the keys stay fixed and only the colour ramp is themed.
    const SERIES_KEYS = [
        'code_lines',
        'bible_kb',
        'system_kb',
        'identity_kb',
        'scratchpad_kb',
        'memory_kb',
    ];
    const LABELS = {
        code_lines: 'Code (lines)',
        bible_kb:   'BIBLE.md (KB)',
        system_kb:  'SYSTEM.md (KB)',
        identity_kb:'identity.md (KB)',
        scratchpad_kb: 'Scratchpad (KB)',
        memory_kb:  'Memory (KB)',
    };

    function setBadge(kind, text) {
        if (!statusBadge) return;
        statusBadge.textContent = text;
        statusBadge.className = `status-badge ${kind}`;
    }

    function formatTs(value) {
        if (!value) return '';
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) return '';
        return parsed.toLocaleString([], {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    }

    function pillTone(status) {
        if (['running', 'queued', 'idle_ready'].includes(status)) return 'online';
        if (['waiting_for_idle', 'waiting_for_owner_chat', 'waiting_for_restart_verify', 'paused', 'starting'].includes(status)) return 'starting';
        if (['budget_blocked', 'budget_stopped', 'paused_failures', 'error_backoff'].includes(status)) return 'error';
        return 'offline';
    }

    function shortStatusLabel(status, fallback = 'off') {
        if (status === 'running') return 'running';
        if (status === 'queued') return 'queued';
        if (status === 'idle_ready') return 'idle';
        if (status === 'waiting_for_idle') return 'waiting';
        if (status === 'waiting_for_owner_chat') return 'needs owner';
        if (status === 'waiting_for_restart_verify') return 'restart verify';
        if (status === 'paused' || status === 'paused_failures') return 'paused';
        if (status === 'budget_blocked' || status === 'budget_stopped') return 'budget';
        if (status === 'error_backoff') return 'retrying';
        if (status === 'stopped') return 'stopped';
        return fallback;
    }

    function runtimeChip(label, value) {
        if (value === null || value === undefined || value === '') return '';
        return `<span class="evo-runtime-chip"><strong>${label}:</strong> ${value}</span>`;
    }

    function renderRuntimeState(runtime = {}, generatedAt = '') {
        const evolution = runtime.evolution_state || {};
        const campaign = evolution.campaign || {};
        const consciousness = runtime.bg_consciousness_state || {};
        const evolutionStatus = evolution.status || (runtime.evolution_enabled ? 'idle_ready' : 'disabled');
        const consciousnessStatus = consciousness.status || (runtime.bg_consciousness_enabled ? 'running' : 'disabled');

        evolutionPill.className = `evo-runtime-pill ${pillTone(evolutionStatus)}`;
        evolutionPill.textContent = `Evolution ${shortStatusLabel(evolutionStatus, 'off')}`;

        consciousnessPill.className = `evo-runtime-pill ${pillTone(consciousnessStatus)}`;
        consciousnessPill.textContent = `Consciousness ${shortStatusLabel(consciousnessStatus, 'off')}`;

        const lines = [];
        if (evolution.detail) lines.push(evolution.detail);
        if (consciousness.detail) lines.push(`Consciousness: ${consciousness.detail}`);
        runtimeDetail.textContent = lines.filter(Boolean).join(' ');

        runtimeMeta.innerHTML = [
            runtimeChip('Cycle', evolution.cycle || 0),
            runtimeChip('Campaign', campaign.id || ''),
            runtimeChip('Campaign attempts', campaign.cycles_done || ''),
            runtimeChip('Absorbed cycles', campaign.absorbed_cycles_done || 0),
            runtimeChip('Queue', `${evolution.pending_count || 0} pending / ${evolution.running_count || 0} running`),
            runtimeChip('Failures', evolution.consecutive_failures || 0),
            runtimeChip('Budget left', Number.isFinite(Number(evolution.budget_remaining_usd)) ? formatUsd2(evolution.budget_remaining_usd) : ''),
            runtimeChip('Last evolution', formatTs(evolution.last_task_at)),
            runtimeChip('Next wakeup', consciousness.next_wakeup_sec ? `${consciousness.next_wakeup_sec}s` : ''),
            runtimeChip('Last background cycle', formatTs(consciousness.last_cycle_finished_at || consciousness.last_cycle_started_at)),
            runtimeChip('Updated', formatTs(generatedAt)),
        ].filter(Boolean).join('');
        if (campaignDetail) {
            const objective = campaign.objective || '';
            const progress = campaign.progress_notes || '';
            campaignDetail.textContent = [objective ? `Objective: ${objective}` : '', progress ? `Progress: ${progress}` : ''].filter(Boolean).join(' ');
        }
        // Evolution is self-modification work, so it is hard-blocked in light mode.
        const isLightMode = String(runtime.runtime_mode || '') === 'light';
        if (startBtn) {
            startBtn.disabled = isLightMode;
            startBtn.title = isLightMode
                ? "Evolution campaigns need runtime mode 'advanced' or 'pro' (currently 'light')."
                : '';
        }
    }

    function renderEmptyState(message) {
        if (evoChart) {
            evoChart.destroy();
            evoChart = null;
        }
        tagsList.innerHTML = `<div class="evo-empty">${message}</div>`;
    }

    async function loadEvolution(force = false) {
        chartLoaded = true;
        const requestId = ++loadSequence;
        refreshBtn.disabled = true;
        setBadge('starting', force ? 'Refreshing...' : 'Loading...');
        try {
            const suffix = force ? '?force=1' : '';
            const [stateResp, evoResp] = await Promise.all([
                apiFetch('/api/state', { cache: 'no-store' }),
                apiFetch(`/api/evolution-data${suffix}`, { cache: 'no-store' }),
            ]);
            if (!stateResp.ok) throw new Error('State API error ' + stateResp.status);
            if (!evoResp.ok) throw new Error('Evolution API error ' + evoResp.status);
            const runtime = await stateResp.json();
            const data = await evoResp.json();
            if (requestId !== loadSequence) return;
            renderRuntimeState(runtime, data.generated_at || '');
            const points = data.points || [];
            if (points.length === 0) {
                renderEmptyState('No evolution tags yet. When evolution commits start landing, the chart will appear here.');
                setBadge('offline', 'No data');
                return;
            }
            setBadge('online', data.cached ? `${points.length} tags (cached)` : `${points.length} tags`);
            renderChart(points);
            renderTagsList(points);
        } catch (err) {
            console.error('Evolution load error:', err);
            if (requestId !== loadSequence) return;
            renderEmptyState('Failed to load evolution data. Use Refresh to try again.');
            setBadge('error', 'Error');
            runtimeDetail.textContent = 'Failed to load evolution state. Try Refresh or wait for the runtime to reconnect.';
            runtimeMeta.innerHTML = '';
        } finally {
            if (requestId === loadSequence) refreshBtn.disabled = false;
        }
    }

    function ensureEvolutionLoaded(force = false) {
        if (!force && chartLoaded) {
            loadEvolution(false);
            return;
        }
        loadEvolution(force);
    }

    function renderChart(points) {
        const theme = evolutionChartTheme();
        const labels = points.map(p => p.tag);
        const datasets = SERIES_KEYS.map((key, idx) => {
            const color = theme.series[idx];
            return {
                label: LABELS[key],
                data: points.map(p => p[key] ?? null),
                borderColor: color,
                backgroundColor: chartColorAlpha(color, 0.13),
                borderWidth: 2,
                pointRadius: 3,
                pointHoverRadius: 5,
                tension: 0.3,
                fill: false,
                yAxisID: key === 'code_lines' ? 'y' : 'y1',
            };
        });
        const ctx = document.getElementById('evo-chart').getContext('2d');
        if (evoChart) evoChart.destroy();
        evoChart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: theme.axisTitle,
                            usePointStyle: true,
                            pointStyle: 'circle',
                            padding: 16,
                            font: { size: 11, family: theme.mono },
                        },
                    },
                    tooltip: {
                        backgroundColor: theme.surface,
                        titleColor: theme.strong,
                        bodyColor: theme.axisTitle,
                        borderColor: theme.border,
                        borderWidth: 1,
                        titleFont: { family: theme.mono, size: 11 },
                        bodyFont: { family: theme.mono, size: 11 },
                        callbacks: {
                            title: function(items) {
                                if (!items.length) return '';
                                const p = points[items[0].dataIndex];
                                return p.tag + ' (' + new Date(p.date).toLocaleDateString() + ')';
                            },
                            label: function(ctx) {
                                const val = ctx.parsed.y;
                                if (val === null || val === undefined) return null;
                                const key = SERIES_KEYS[ctx.datasetIndex];
                                if (key === 'code_lines') return ' ' + ctx.dataset.label + ': ' + val.toLocaleString() + ' lines';
                                return ' ' + ctx.dataset.label + ': ' + val.toFixed(1) + ' KB';
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        ticks: { color: theme.axis, font: { size: 10, family: theme.mono }, maxRotation: 45 },
                        grid: { color: theme.grid },
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        title: { display: true, text: 'Lines of Code', color: theme.series[0], font: { size: 11 } },
                        ticks: { color: theme.axis, font: { size: 10, family: theme.mono } },
                        grid: { color: theme.grid },
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        title: { display: true, text: 'Size (KB)', color: theme.axisTitle, font: { size: 11 } },
                        ticks: { color: theme.axis, font: { size: 10, family: theme.mono } },
                        grid: { drawOnChartArea: false },
                    },
                },
            },
        });
    }

    function renderTagsList(points) {
        const rows = points.map(p => {
            const d = new Date(p.date);
            const dateStr = d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
            const codeLines = Number(p.code_lines || 0);
            const bibleKb = Number(p.bible_kb || 0);
            const systemKb = Number(p.system_kb || 0);
            const identityKb = Number(p.identity_kb || 0);
            const scratchpadKb = Number(p.scratchpad_kb || 0);
            const memoryKb = Number(p.memory_kb || 0);
            return `<tr>
                <td><code>${escapeHtmlText(p.tag || '')}</code></td>
                <td>${escapeHtmlText(dateStr)}</td>
                <td>${Number.isFinite(codeLines) ? codeLines.toLocaleString() : '0'}</td>
                <td>${Number.isFinite(bibleKb) ? bibleKb.toFixed(1) : '0.0'}</td>
                <td>${Number.isFinite(systemKb) ? systemKb.toFixed(1) : '0.0'}</td>
                <td>${Number.isFinite(identityKb) ? identityKb.toFixed(1) : '0.0'}</td>
                <td>${Number.isFinite(scratchpadKb) ? scratchpadKb.toFixed(1) : '0.0'}</td>
                <td>${Number.isFinite(memoryKb) ? memoryKb.toFixed(1) : '0.0'}</td>
            </tr>`;
        }).reverse().join('');
        tagsList.innerHTML = `
            <table class="cost-table">
                <thead><tr>
                    <th>Tag</th><th>Date</th><th>Code Lines</th>
                    <th>BIBLE (KB)</th><th>SYSTEM (KB)</th>
                    <th>Identity (KB)</th><th>Scratchpad (KB)</th><th>Memory (KB)</th>
                </tr></thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    }

    // -----------------------------------------------------------------------
    // Refresh button + event listeners
    // -----------------------------------------------------------------------
    refreshBtn.addEventListener('click', () => {
        loadEvolution(true);
    });
    startBtn?.addEventListener('click', async () => {
        const asked = await promptCampaignObjective();
        // Cancel = no campaign start (semantic fix, owner-approved Б1-8: the
        // old `prompt() || ''` started a paid campaign even on Cancel/null).
        if (!asked.confirmed) return;
        await apiFetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cmd: `/evolve on${asked.objective ? ` ${asked.objective}` : ''}` }),
        });
        loadEvolution(true);
    });
    stopBtn?.addEventListener('click', async () => {
        await apiFetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cmd: '/evolve off' }),
        });
        loadEvolution(true);
    });

    ws.on('open', () => {
        if (isEvolutionVisible()) {
            ensureEvolutionLoaded(false);
        }
    });

    window.addEventListener('ouro:page-shown', (event) => {
        if (event?.detail?.page === 'dashboard' && state.dashboardActiveSubtab === 'evolution') {
            ensureEvolutionLoaded(false);
        }
    });
    window.addEventListener('ouro:dashboard-subtab-shown', (event) => {
        if (event?.detail?.tab === 'evolution') ensureEvolutionLoaded(false);
    });

    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && isEvolutionVisible()) {
            if (chartLoaded) loadEvolution(false);
        }
    });
}

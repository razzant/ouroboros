// Main-screen Update affordance (P2): a compact pill that appears when a managed update
// is available (status is populated by the boot-time check-on-restart), opening a staged
// choice dialog (Auto-update / Ouroboros-assisted / Manual) backed by a fresh merge
// preflight. The full merge/smoke/rollback happens server-side; this is the thin,
// transparent control surface. Non-invasive: the detailed Dashboard -> Updates panel
// stays the place for recovery/details.

import { apiClient, updateStrategyForPlan } from './api_client.js';

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (c) => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}

// Fail-soft wrapper around the api_client update helpers (the pill must never throw the app).
async function safe(fn) {
    try {
        return await fn();
    } catch {
        return null;
    }
}

export function updatePillText(status = {}) {
    const currentVersion = String(status.current_version || '');
    const latestVersion = String(status.latest_version || '');
    const currentSha = String(status.current_short_sha || status.current_sha || '').slice(0, 8);
    const latestSha = String(status.latest_short_sha || status.latest_sha || '').slice(0, 8);
    if (currentVersion && latestVersion && currentVersion === latestVersion) {
        return currentSha && latestSha
            ? `Update ${currentSha} → ${latestSha}`
            : `Update available${latestSha ? ` · ${latestSha}` : ''}`;
    }
    const current = currentVersion || currentSha;
    const latest = latestVersion || latestSha;
    return current && latest ? `Update ${current} → ${latest}` : 'Update available';
}

export function verifiedUpdatePlan(preflight) {
    const plan = preflight?.merge_plan;
    if (!plan || typeof plan !== 'object') return null;
    const strategy = updateStrategyForPlan(plan);
    if (
        !strategy
        || !plan.base_sha
        || !plan.target_sha
        || !Number.isInteger(plan.local_dirty_count)
        || plan.local_dirty_count < 0
    ) return null;
    return { plan, strategy };
}

export function initUpdateStatus({ showPage, openDashboardTab, ws } = {}) {
    function ensurePill() {
        let pill = document.getElementById('update-pill');
        if (!pill) {
            pill = document.createElement('button');
            pill.id = 'update-pill';
            pill.type = 'button';
            pill.className = 'update-pill';
            pill.hidden = true;
            pill.addEventListener('click', openUpdateDialog);
            // Dedicated sidebar footer slot: the pill is a full row of its own, so
            // it cannot inflate the compact brand sub line that carries the
            // version + liveness dot.
            const slot = document.getElementById('nav-update-slot');
            if (slot) {
                slot.appendChild(pill);
            } else {
                document.body.appendChild(pill);
            }
        }
        return pill;
    }

    function renderPill(status) {
        const pill = ensurePill();
        if (!status || !status.available) {
            pill.hidden = true;
            return;
        }
        pill.textContent = updatePillText(status);
        pill.classList.toggle('has-local', Boolean(status.dirty || status.ahead));
        pill.hidden = false;
    }

    async function refresh() {
        renderPill(await safe(() => apiClient.updateStatus()));
    }

    async function openUpdateDialog() {
        const overlay = document.createElement('div');
        overlay.className = 'update-dialog-overlay';
        overlay.innerHTML = '<div class="update-dialog"><div class="update-dialog-status">Checking update…</div></div>';
        document.body.appendChild(overlay);

        const pre = await safe(() => apiClient.updatePreflight());
        const verified = verifiedUpdatePlan(pre);
        if (!verified) {
            overlay.querySelector('.update-dialog').innerHTML = `
                <h3 class="update-dialog-title">Update plan unavailable</h3>
                <div class="update-dialog-meta">The update could not be verified. No files were changed.</div>
                <div class="update-dialog-actions">
                    <button data-retry class="btn btn-primary">Retry</button>
                    <button data-open-details class="btn btn-default">Open details</button>
                    <button data-close class="btn btn-default">Cancel</button>
                </div>`;
            overlay.addEventListener('click', (event) => {
                const t = event.target;
                if (t === overlay || t.hasAttribute?.('data-close')) overlay.remove();
                if (t.hasAttribute?.('data-retry')) {
                    overlay.remove();
                    openUpdateDialog();
                }
                if (t.hasAttribute?.('data-open-details')) {
                    overlay.remove();
                    showPage?.('dashboard');
                    openDashboardTab?.('updates');
                }
            });
            return;
        }
        const { plan, strategy } = verified;
        const hot = new Set(plan.hot_code_paths || []);
        const conflicts = [
            ...((plan.code_conflict_paths || []).map((p) => (hot.has(p) ? `Code (hot): ${p}` : `Code: ${p}`))),
            ...((plan.doc_conflict_paths || []).map((p) => `Docs: ${p}`)),
        ];
        const base = plan.base_sha ? String(plan.base_sha).slice(0, 8) : '';
        const target = plan.target_sha ? String(plan.target_sha).slice(0, 8) : '';
        const primary = strategy === 'auto_merge'
            ? '<button data-strategy="auto_merge" class="btn btn-primary">Update now</button>'
            : (strategy === 'assisted'
                ? '<button data-strategy="assisted" class="btn btn-primary">Update with Ouroboros</button>'
                : '<button class="btn btn-primary" disabled>Update unavailable</button>');

        overlay.querySelector('.update-dialog').innerHTML = `
            <h3 class="update-dialog-title">Update ${escapeHtml(base)} → ${escapeHtml(target)}</h3>
            <div class="update-dialog-meta">${plan.local_dirty_count} local change(s)${conflicts.length ? ` · ${conflicts.length} conflict(s)` : (plan.merge_commit && plan.merge_commit !== plan.target_sha ? ' · automatic Git merge' : ' · direct fast-forward')}</div>
            ${conflicts.length ? `<ul class="update-dialog-conflicts">${conflicts.map((r) => `<li>${escapeHtml(r)}</li>`).join('')}</ul>` : ''}
            <div class="update-dialog-note">Git handles clean updates directly. Ouroboros joins only when Git reports a real conflict. A rescue snapshot and rollback protect the current checkout; uncommitted local edits are stashed and restored as uncommitted work after the update.</div>
            <div class="update-dialog-actions">
                ${primary}
                <button data-open-details class="btn btn-default">Open details</button>
                <button data-close class="btn btn-default">Cancel</button>
            </div>
            <div class="update-dialog-status" hidden></div>`;

        const statusEl = overlay.querySelector('.update-dialog-status');
        overlay.addEventListener('click', async (event) => {
            const t = event.target;
            if (t === overlay || t.hasAttribute?.('data-close')) {
                overlay.remove();
                return;
            }
            if (t.hasAttribute?.('data-open-details')) {
                overlay.remove();
                showPage?.('dashboard');
                openDashboardTab?.('updates');
                return;
            }
            const strat = t.dataset?.strategy;
            if (!strat) return;
            statusEl.hidden = false;
            statusEl.textContent = 'Applying update…';
            const data = await apiClient.updateApply(strat, plan).catch((e) => ({
                error: String((e && e.message) || e),
                restart_required: Boolean(e?.body?.restart_required),
            }));
            if (data && data.status === 'ok' && data.restarting) {
                statusEl.textContent = 'Update applied; smoke-test passed; restarting…';
            } else if (data && data.status === 'assisted_started') {
                statusEl.textContent = 'Ouroboros is resolving the merge under review — watch progress in chat.';
            } else if (data && data.status === 'restart_required') {
                statusEl.textContent = 'The update landed, but automatic restart failed. Restart Ouroboros to finish.';
            } else if (data && data.restart_required) {
                statusEl.textContent = `Did not complete: ${data.error}. Runtime shutdown was incomplete; restart Ouroboros before retrying.`;
            } else {
                statusEl.textContent = (data && data.error) ? `Did not complete: ${data.error}` : 'Update did not complete.';
            }
        });
    }

    refresh();
    ws?.on?.('update_status_ready', refresh);
    window.addEventListener('ouro:page-shown', (event) => {
        if (event?.detail?.page === 'chat') refresh();
    });

    return { refresh };
}

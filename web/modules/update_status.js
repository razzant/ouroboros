// Main-screen Update affordance (P2): a compact pill that appears when a managed update
// is available (status is populated by the boot-time check-on-restart), opening a staged
// choice dialog (Auto-update / Ouroboros-assisted / Manual) backed by a fresh merge
// preflight. The full merge/smoke/rollback happens server-side; this is the thin,
// transparent control surface. Non-invasive: the detailed Dashboard -> Updates panel
// stays the place for recovery/details.

import { apiClient } from './api_client.js';

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

export function initUpdateStatus({ showPage, openDashboardTab } = {}) {
    function ensurePill() {
        let pill = document.getElementById('update-pill');
        if (!pill) {
            pill = document.createElement('button');
            pill.id = 'update-pill';
            pill.type = 'button';
            pill.className = 'update-pill';
            pill.hidden = true;
            pill.addEventListener('click', openUpdateDialog);
            const anchor = document.getElementById('nav-version');
            if (anchor && anchor.parentNode) {
                anchor.parentNode.insertBefore(pill, anchor.nextSibling);
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
        const cur = status.current_version || (status.current_sha ? String(status.current_sha).slice(0, 8) : '');
        const next = status.latest_version || (status.latest_sha ? String(status.latest_sha).slice(0, 8) : '');
        pill.textContent = (cur && next) ? `Update ${cur} → ${next}` : 'Update available';
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
        const plan = (pre && pre.merge_plan) || {};
        const kind = plan.kind || 'unknown';
        // Auto-update only for a clean merge AND a clean working tree: the backend routes a
        // clean merge with uncommitted local work to the reviewed assisted task (never an
        // unreviewed auto-commit), so the dialog must offer assisted there, not Auto-update.
        const clean = kind === 'clean' && Number(plan.local_dirty_count || 0) === 0;
        const hot = new Set(plan.hot_code_paths || []);
        const conflicts = [
            ...((plan.protected_conflict_paths || []).map((p) => `Protected: ${p}`)),
            ...((plan.code_conflict_paths || []).map((p) => (hot.has(p) ? `Code (hot): ${p}` : `Code: ${p}`))),
            ...((plan.doc_conflict_paths || []).map((p) => `Docs: ${p}`)),
        ];
        const base = plan.base_sha ? String(plan.base_sha).slice(0, 8) : '';
        const target = plan.target_sha ? String(plan.target_sha).slice(0, 8) : '';
        const primary = clean
            ? '<button data-strategy="auto_merge" class="btn btn-primary">Auto-update</button>'
            : '<button data-strategy="assisted" class="btn btn-primary">Ouroboros-assisted update</button>';

        overlay.querySelector('.update-dialog').innerHTML = `
            <h3 class="update-dialog-title">Update ${escapeHtml(base)} → ${escapeHtml(target)}</h3>
            <div class="update-dialog-meta">${plan.local_dirty_count || 0} local change(s)${conflicts.length ? ` · ${conflicts.length} conflict(s)` : ' · clean merge'}</div>
            ${conflicts.length ? `<ul class="update-dialog-conflicts">${conflicts.map((r) => `<li>${escapeHtml(r)}</li>`).join('')}</ul>` : ''}
            <div class="update-dialog-note">Your local work is preserved in a rescue snapshot first; a smoke test runs before the restart is accepted, and a failed update auto-rolls-back to the current version.</div>
            <div class="update-dialog-actions">
                ${primary}
                <button data-strategy="manual" class="btn btn-default">Open details</button>
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
            const strat = t.dataset?.strategy;
            if (!strat) return;
            if (strat === 'manual') {
                overlay.remove();
                showPage?.('dashboard');
                openDashboardTab?.('updates');
                return;
            }
            statusEl.hidden = false;
            statusEl.textContent = 'Applying update…';
            const data = await apiClient.updateApply(strat).catch((e) => ({ error: String((e && e.message) || e) }));
            if (data && data.status === 'ok' && data.restarting) {
                statusEl.textContent = 'Update applied; smoke-test passed; restarting…';
            } else if (data && data.status === 'assisted_started') {
                statusEl.textContent = 'Ouroboros is resolving the merge under review — watch progress in chat.';
            } else if (data && data.status === 'manual') {
                // The backend routed this update to MANUAL (e.g. it touches protected paths like
                // BIBLE/CHECKLISTS/SAFETY) — surface that handoff, don't show a generic failure.
                const prot = Array.isArray(data.protected_paths) && data.protected_paths.length
                    ? ` (protected: ${data.protected_paths.slice(0, 6).map(escapeHtml).join(', ')})`
                    : '';
                statusEl.textContent = `This update needs manual handling${prot} — opening the detailed Updates panel…`;
                setTimeout(() => { overlay.remove(); showPage?.('dashboard'); openDashboardTab?.('updates'); }, 1500);
            } else {
                statusEl.textContent = (data && data.error) ? `Did not complete: ${data.error}` : 'Update did not complete.';
            }
        });
    }

    refresh();
    window.addEventListener('ouro:page-shown', (event) => {
        if (event?.detail?.page === 'chat') refresh();
    });

    return { refresh };
}

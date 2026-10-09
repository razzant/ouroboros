import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { bindDialogFocus } from './ui_interactions.js';

let activeDialog = null;
let activeClose = null;

/**
 * Render one caller-supplied typed detail disclosure. Values are always text;
 * callers cannot inject markup into the shared dialog.
 * @param {{summary?: string, rows?: Array<{label?: string, value?: string}>}|null} details
 */
export function renderConfirmDialogDetails(details) {
    if (!details || typeof details !== 'object') return '';
    const rows = Array.isArray(details.rows)
        ? details.rows.filter((row) => row && typeof row === 'object')
        : [];
    if (!rows.length) return '';
    const renderedRows = rows.map((row) => `
        <div class="confirm-dialog-detail-row">
            <dt>${escapeHtml(String(row.label ?? 'Detail'))}</dt>
            <dd>${escapeHtml(String(row.value ?? ''))}</dd>
        </div>`).join('');
    return `<details class="confirm-dialog-details ui-rich-content">
        <summary>${escapeHtml(String(details.summary ?? 'Show details'))}</summary>
        <dl class="confirm-dialog-detail-list">${renderedRows}
        </dl>
    </details>`;
}

export function openConfirmDialog({
    title,
    body,
    details = null,
    input = false,
    initialValue = '',
    // [{value, label}] makes the input a <select>; the result shape is unchanged.
    choices = null,
    // Alert mode (v6.90.3): one OK-style button, no cancel button. Escape,
    // backdrop, and the header Close still resolve the promise (false), same
    // as a cancelled confirm — callers treat any resolution as "seen".
    alert = false,
    confirmLabel = alert ? 'OK' : 'Continue',
    cancelLabel = 'Cancel',
    danger = false,
} = {}) {
    if (activeClose) activeClose(false);
    return new Promise((resolve) => {
        const backdrop = document.createElement('div');
        backdrop.className = 'marketplace-modal-backdrop confirm-dialog-backdrop';
        backdrop.innerHTML = `
            <div class="marketplace-modal confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-dialog-title">
                <div class="marketplace-modal-head">
                    <h3 id="confirm-dialog-title">${escapeHtml(title || 'Confirm action')}</h3>
                    <button type="button" class="btn btn-default btn-sm" data-confirm-cancel aria-label="Close">Close</button>
                </div>
                <div class="marketplace-modal-body">
                    <p>${escapeHtml(body || 'Continue?')}</p>
                    ${renderConfirmDialogDetails(details)}
                    ${!input ? '' : Array.isArray(choices)
                        ? `<select class="ui-control files-modal-input confirm-dialog-input" data-confirm-input aria-labelledby="confirm-dialog-title">${choices.map((choice) => `<option value="${escapeHtml(String(choice?.value ?? ''))}">${escapeHtml(String(choice?.label ?? choice?.value ?? ''))}</option>`).join('')}</select>`
                        : `<input class="ui-control files-modal-input confirm-dialog-input" data-confirm-input type="text" aria-labelledby="confirm-dialog-title" value="${escapeHtml(initialValue)}">`}
                </div>
                <div class="marketplace-modal-actions">
                    ${alert ? '' : `<button type="button" class="btn btn-default" data-confirm-cancel>${escapeHtml(cancelLabel)}</button>`}
                    <button type="button" class="btn ${danger ? 'btn-danger' : 'btn-primary'}" data-confirm-ok>${escapeHtml(confirmLabel)}</button>
                </div>
            </div>
        `;
        let settled = false;
        let disposeFocus = null;
        const finish = (value) => {
            if (settled) return;
            settled = true;
            document.removeEventListener('keydown', onKey);
            if (activeDialog === backdrop) activeDialog = null;
            if (activeClose === cancel) activeClose = null;
            disposeFocus?.();
            backdrop.remove();
            resolve(value);
        };
        const result = (confirmed) => input
            ? { confirmed, value: confirmed ? (backdrop.querySelector('[data-confirm-input]')?.value || '') : '' }
            : confirmed;
        // Supersession must honor the MODE's contract: a newer dialog cancelling
        // this one resolves the same shape a user cancel would ({confirmed:false,
        // value:''} in input mode), never a bare false the docs do not promise.
        const cancel = () => finish(result(false));
        backdrop.addEventListener('click', (event) => {
            if (event.target === backdrop || event.target.closest('[data-confirm-cancel]')) {
                finish(result(false));
            } else if (event.target.closest('[data-confirm-ok]')) {
                finish(result(true));
            }
        });
        const onKey = (event) => {
            if (activeDialog !== backdrop || event.defaultPrevented || event.isComposing) return;
            if (event.key === 'Escape') {
                event.preventDefault();
                finish(result(false));
            } else if (input && event.key === 'Enter' && event.target?.matches?.('[data-confirm-input]')) {
                event.preventDefault();
                finish(result(true));
            }
        };
        document.addEventListener('keydown', onKey);
        document.body.appendChild(backdrop);
        activeDialog = backdrop;
        activeClose = cancel;
        disposeFocus = bindDialogFocus(backdrop.querySelector('[role="dialog"]'), {
            initialFocus: backdrop.querySelector(input ? '[data-confirm-input]' : '[data-confirm-ok]'),
            onEscape: cancel,
        });
        backdrop.querySelector('[data-confirm-input]')?.select?.();
    });
}

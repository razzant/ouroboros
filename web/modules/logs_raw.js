import { apiFetch, fetchJson } from './api_client.js';
import { downloadBlobViaHostBridge, shellBridgeApi } from './ui_helpers.js';
import { machineWords } from './utils.js';

// One explicit opening owns one page. The source is fixed until it is closed: a repeated event must never
// silently substitute newer bytes. File rows use the gateway's verified reader; unreferenced live events say
// "Received event" and serialize large snapshots off the UI thread. Neither path accumulates pages in the DOM.
// The record is the last section of the event's one fold (logs.js): opening the fold reads it, closing it lets the
// page go. Paging shows only for a record of more than one page. The tools are small neutral capsules, as every other
// Retry is (UI_CONTROL_TEXT_INVENTORY.md).
export const LOG_RAW_HTML = `<pre class="log-raw" hidden></pre>
    <div class="log-raw-tools" hidden><div class="log-raw-status ui-status" role="status"></div>
        <div class="log-raw-pages"><button class="btn btn-default btn-sm" data-raw-action="first" type="button" hidden>First page</button>
        <button class="btn btn-default btn-sm" data-raw-action="next" type="button" hidden>Next page</button>
        <button class="btn btn-default btn-sm" data-raw-action="retry" type="button" hidden>Retry</button>
        <a class="btn btn-default btn-sm log-raw-download" download="log-row.json" hidden>Download full</a></div></div>`;
const PAGE_BYTES = 65536;
const SOURCE_KEYS = ['v', 'kind', 'placement', 'task_id', 'log', 'start', 'length', 'sha256'];
const sameSource = (a, b) => a && b && SOURCE_KEYS.every(key => a[key] === b[key]);
const rawUrl = (raw, params) => `/api/logs/${encodeURIComponent(raw.source.log)}/raw?${new URLSearchParams({ ref: raw.ref, ...params })}`;
// A reconstruction whose receipt could not be used is the stored event served whole: its status names what it lacks.
const RECEIPT_GAPS = { receipt_unavailable: 'receipt unavailable', receipt_mismatch: 'receipt does not match' };
function receiptGap(data) {
    const missing = data.unavailable_fields?.length || 0;
    return missing ? ` · ${RECEIPT_GAPS[data.unavailable_reason] || 'receipt not used'}: ${missing} declared ${missing === 1 ? 'field' : 'fields'} missing` : '';
}
const digest = (sha) => `${String(sha).slice(0, 12)}…`;  // a glance compares it; a download is checked whole
// A record of one page reads whole: no paging, no byte range, one JSON value indented. A longer one's page is as read.
const whole = (data) => data.offset === 0 && !data.has_more;
function wholeText(data) {
    if (!whole(data)) return data.text;
    try { return JSON.stringify(JSON.parse(data.text), null, 2); } catch { return data.text; }
}

// Only a demonstrably small JSON value is serialized on the UI thread. The budget includes escaping and nesting;
// the worker handles the rest, including a received 17 MB checkpoint with no canonical file reference yet.
function smallText(value) {
    let budget = 8000;
    const seen = new Set();
    const fits = (item, depth = 0) => {
        budget -= 4 + depth * 2;
        if (budget < 0 || depth > 32) return false;
        if (typeof item === 'string') budget -= item.length;
        else if (item && typeof item === 'object') {
            if (seen.has(item)) return false;
            seen.add(item);
            for (const key of Object.keys(item)) {
                budget -= key.length + 4;
                if (!fits(item[key], depth + 1)) return false;
            }
            seen.delete(item);
        }
        return budget >= 0;
    };
    if (!fits(value)) return null;
    const text = JSON.stringify(value, null, 2);
    if (typeof text !== 'string') throw new Error('this event has no JSON form');
    return new TextEncoder().encode(text).length <= PAGE_BYTES ? text : null;
}

export function createLogRaw({ visible }) {
    const bindings = new WeakMap(), opened = new Set();
    function stop(view) {
        view.turn += 1;
        view.abort?.abort(); view.abort = null;
        view.downloadAbort?.abort(); view.downloadAbort = null;
        view.worker?.terminate(); view.worker = null;
        view.rejectWorker?.(new DOMException('Raw closed', 'AbortError')); view.rejectWorker = null;
    }
    function release(view) {
        stop(view);
        if (view.url) { URL.revokeObjectURL(view.url); view.download.removeAttribute('href'); view.download.hidden = true; }
        view.url = null; view.blob = null;
    }
    const shown = view => visible() && !view.fold.closest?.('[hidden]');
    const current = (view, turn) => opened.has(view) && view.turn === turn && shown(view);
    // Another page takes focus after Logs leaves; WebKit clears its selection even though Raw's text stays.
    function keepSelection(view) {
        const selection = view.pre.ownerDocument?.getSelection?.();
        view.ranges = selection?.rangeCount && !selection.isCollapsed && view.pre.contains(selection.anchorNode)
            && view.pre.contains(selection.focusNode)
            ? Array.from({ length: selection.rangeCount }, (_, index) => selection.getRangeAt(index).cloneRange()) : null;
    }
    function restoreSelection(view) {
        const selection = view.pre.ownerDocument?.getSelection?.(), ranges = view.ranges;
        view.ranges = null;
        if (!selection || !ranges?.length || !ranges.every(range => view.pre.contains(range.startContainer) && view.pre.contains(range.endContainer))) return;
        selection.removeAllRanges();
        ranges.forEach(range => selection.addRange(range));
    }
    function status(view, text) { view.status.textContent = text; }
    function showPage(view, data) {
        view.ranges = null;
        view.next = data.next_offset;
        view.shown = data.offset;
        view.pre.textContent = wholeText(data);
        view.pre.scrollTop = 0;
        view.first.disabled = data.offset === 0;
        view.nextButton.disabled = !data.has_more;
        view.first.hidden = view.nextButton.hidden = whole(data);
        view.retry.hidden = true;
        view.download.hidden = false;
        const identity = view.raw ? `${machineWords(data.representation)}${receiptGap(data)} · Source SHA-256 ${digest(data.source.sha256)}` : 'Received event';
        status(view, whole(data) ? identity : `${data.offset}–${data.next_offset} of ${data.total_bytes} bytes · ${identity}`);
        view.loading = false;
    }
    async function read(view, offset = 0) {
        stop(view);
        const turn = view.turn;
        // Shown again, a received event's page is still on screen: only its download is built again, the page is not
        // written, so its scroll and a selection in it stay. Moving to a page shows that page from its top.
        const again = !view.raw && view.shown === offset;
        view.offset = offset; view.loading = true; view.owed = false;
        if (!again) {
            view.first.disabled = view.nextButton.disabled = true;
            view.retry.hidden = true;
            status(view, 'Loading Raw…');
        }
        try {
            if (view.raw) {
                const raw = view.raw;
                if (!['events', 'tools', 'supervisor'].includes(raw.source?.log) || typeof raw.ref !== 'string') throw new Error('Invalid Raw reference');
                view.abort = new AbortController();
                const data = await fetchJson(rawUrl(raw, { offset, limit: PAGE_BYTES }), { cache: 'no-store', signal: view.abort.signal });
                if (!current(view, turn)) return;
                if (!sameSource(raw.source, data.source) || data.representation !== raw.representation
                    || (view.payloadHash && (view.payloadHash !== data.sha256 || view.total !== data.total_bytes))) {
                    throw new Error('Raw source changed; this page was not substituted');
                }
                const bytes = typeof data.text === 'string' ? new TextEncoder().encode(data.text).length : -1;
                if (data.offset !== offset || !Number.isSafeInteger(data.next_offset) || !Number.isSafeInteger(data.total_bytes)
                    || !/^[a-f0-9]{64}$/.test(data.sha256 || '') || bytes < 0 || bytes > PAGE_BYTES
                    || data.next_offset !== offset + bytes || data.next_offset > data.total_bytes
                    || data.has_more !== (data.next_offset < data.total_bytes) || (data.has_more && !bytes)
                    || (data.unavailable_fields !== undefined && (!Array.isArray(data.unavailable_fields)
                        || data.unavailable_fields.some(name => typeof name !== 'string')))) throw new Error('Invalid Raw page');
                view.payloadHash = data.sha256; view.total = data.total_bytes;
                view.download.href = rawUrl(raw, { download: 1 });
                showPage(view, data);
            } else {
                if (!view.blob) {
                    const small = smallText(view.event);
                    const blob = small !== null ? new Blob([small], { type: 'application/json' }) : await new Promise((resolve, reject) => {
                        view.rejectWorker = reject;
                        const worker = view.worker = new Worker(new URL('./logs_raw_worker.js', import.meta.url), { type: 'module' });
                        worker.onmessage = ({ data }) => data.error ? reject(new Error(data.error)) : resolve(data.blob);
                        worker.onerror = () => reject(new Error('Received event could not be serialized'));
                        worker.postMessage(view.event);
                    });
                    if (!current(view, turn)) return;
                    view.rejectWorker = null;
                    view.blob = blob;
                }
                if (!current(view, turn)) return;
                view.worker?.terminate(); view.worker = null;
                if (!view.url) view.url = URL.createObjectURL(view.blob);
                view.download.href = view.url;
                if (again) { view.download.hidden = false; view.loading = false; restoreSelection(view); return; }
                // Withhold an unfinished UTF-8 code point, just as the gateway does. No character-count cursor.
                const bytes = new Uint8Array(await view.blob.slice(offset, offset + PAGE_BYTES + 1).arrayBuffer());
                let end = Math.min(PAGE_BYTES, bytes.length);
                if (end < bytes.length) while (end && (bytes[end] & 0xc0) === 0x80) end -= 1;
                const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(0, end));
                if (!current(view, turn)) return;
                showPage(view, { offset, next_offset: offset + end, total_bytes: view.blob.size, has_more: offset + end < view.blob.size, text });
            }
        } catch (error) {
            if (!current(view, turn)) return;
            view.worker?.terminate(); view.worker = null;
            view.loading = false;
            view.retry.hidden = false;
            view.download.hidden = true;
            const code = [error.status && `HTTP ${error.status}`, error.body?.reason_code].filter(Boolean).join(', ');
            status(view, `Raw unavailable${code ? ` (${code})` : ''}: ${error.message}`);
        }
    }
    function close(view) {
        opened.delete(view); release(view);
        view.pre.hidden = true; view.pre.textContent = ''; view.tools.hidden = true;
    }
    async function download(view) {
        // Packaged launchers accept only their older file routes. An explicit download uses their existing bytes
        // bridge; ordinary browsers stream the attachment URL. Neither path puts the full value in the DOM.
        const turn = view.turn, abort = view.downloadAbort = new AbortController();
        status(view, 'Downloading full Raw…');
        try {
            const response = await apiFetch(rawUrl(view.raw, { download: 1 }), { signal: abort.signal, cache: 'no-store' });
            if (!response.ok) {
                const body = await response.json().catch(() => ({}));
                throw new Error(`HTTP ${response.status}${body.reason_code ? `, ${body.reason_code}` : ''}`);
            }
            const blob = await response.blob();
            if (!current(view, turn) || abort.signal.aborted) return;
            const bytes = await blob.arrayBuffer();
            const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(n => n.toString(16).padStart(2, '0')).join('');
            if (!current(view, turn) || abort.signal.aborted) return;
            if (blob.size !== view.total || hash !== view.payloadHash) throw new Error('Raw source changed; download was not substituted');
            const result = await downloadBlobViaHostBridge(blob, 'log-row.json');
            if (current(view, turn)) status(view, result.ok ? 'Full Raw saved.' : `Raw download unavailable: ${result.error || 'save refused'}`);
        } catch (error) {
            if (current(view, turn) && !abort.signal.aborted) status(view, `Raw download unavailable: ${error.message}`);
        } finally { if (view.downloadAbort === abort) view.downloadAbort = null; }
    }
    return {
        rememberSelection() { for (const view of opened) if (shown(view)) keepSelection(view); },
        bind(fold, source) { bindings.set(fold, source); },
        changed(root) {
            for (const view of opened) {
                if (!root.contains(view.fold)) continue;
                const next = view.source();
                if (next === view.event || (view.raw?.ref && next?._raw?.ref === view.raw.ref && sameSource(next._raw.source, view.raw.source))) continue;
                release(view); view.invalid = true;
                view.first.disabled = view.nextButton.disabled = true; view.retry.hidden = view.download.hidden = true;
                status(view, 'This entry changed. Close and reopen its record to read its current source.');
            }
        },
        remove(root) { for (const view of opened) if (root.contains(view.fold)) { opened.delete(view); release(view); } },
        exposure() {
            for (const view of opened) {
                if (view.invalid) continue;
                if (!shown(view)) { view.owed ||= view.loading || !view.raw; release(view); }
                else if (view.owed) void read(view, view.offset);
            }
        },
        // The record follows its fold: opened, its first page is read; closed, the page is let go.
        toggled(fold) {
            const source = bindings.get(fold);
            if (!source) return false;
            const existing = [...opened].find(view => view.fold === fold);
            if (!fold.open) { if (existing) close(existing); return true; }
            if (existing) return true;
            const event = source();
            const view = { source, fold, event, raw: event?._raw, turn: 0,
                pre: fold.querySelector('.log-raw'), tools: fold.querySelector('.log-raw-tools'), status: fold.querySelector('.log-raw-status'),
                first: fold.querySelector('[data-raw-action="first"]'), nextButton: fold.querySelector('[data-raw-action="next"]'),
                retry: fold.querySelector('[data-raw-action="retry"]'), download: fold.querySelector('.log-raw-download') };
            opened.add(view); view.pre.hidden = view.tools.hidden = false;
            void read(view);
            return true;
        },
        click(target, event) {
            const link = target?.closest?.('.log-raw-download');
            if (link && shellBridgeApi(window)) {
                const view = [...opened].find(item => item.download === link);
                if (view?.raw) {
                    event?.preventDefault();
                    if (!view.invalid && !view.downloadAbort) void download(view);
                    return true;
                }
            }
            const action = target?.closest?.('[data-raw-action]');
            if (!action) return false;
            const view = [...opened].find(item => item.fold.contains(action));
            if (view && !view.invalid && !action.disabled) void read(view, action.dataset.rawAction === 'first' ? 0 : action.dataset.rawAction === 'next' ? view.next : view.offset);
            return true;
        },
    };
}

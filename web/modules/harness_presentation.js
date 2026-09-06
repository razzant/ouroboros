// Harness identity presentation SSOT.
//
// Brand vector provenance (paths and viewBoxes preserved byte-for-byte from
// Claudexor's HarnessLogoData.swift): Claude, Cursor and OpenCode are from
// Simple Icons; Codex/OpenAI is from SVGL. The marks are monochrome and inherit
// currentColor. Product names and marks remain the property of their owners.
//
// This module presents identity only. It does not decide route availability,
// status, requested/effective/executed truth, or review authority.

import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const HARNESS_MARKS = Object.freeze({
    codex: Object.freeze({
        label: 'Codex',
        viewBox: '0 0 256 260',
        path: 'M239.184 106.203a64.716 64.716 0 0 0-5.576-53.103C219.452 28.459 191 15.784 163.213 21.74A65.586 65.586 0 0 0 52.096 45.22a64.716 64.716 0 0 0-43.23 31.36c-14.31 24.602-11.061 55.634 8.033 76.74a64.665 64.665 0 0 0 5.525 53.102c14.174 24.65 42.644 37.324 70.446 31.36a64.72 64.72 0 0 0 48.754 21.744c28.481.025 53.714-18.361 62.414-45.481a64.767 64.767 0 0 0 43.229-31.36c14.137-24.558 10.875-55.423-8.083-76.483Zm-97.56 136.338a48.397 48.397 0 0 1-31.105-11.255l1.535-.87 51.67-29.825a8.595 8.595 0 0 0 4.247-7.367v-72.85l21.845 12.636c.218.111.37.32.409.563v60.367c-.056 26.818-21.783 48.545-48.601 48.601Zm-104.466-44.61a48.345 48.345 0 0 1-5.781-32.589l1.534.921 51.722 29.826a8.339 8.339 0 0 0 8.441 0l63.181-36.425v25.221a.87.87 0 0 1-.358.665l-52.335 30.184c-23.257 13.398-52.97 5.431-66.404-17.803ZM23.549 85.38a48.499 48.499 0 0 1 25.58-21.333v61.39a8.288 8.288 0 0 0 4.195 7.316l62.874 36.272-21.845 12.636a.819.819 0 0 1-.767 0L41.353 151.53c-23.211-13.454-31.171-43.144-17.804-66.405v.256Zm179.466 41.695-63.08-36.63L161.73 77.86a.819.819 0 0 1 .768 0l52.233 30.184a48.6 48.6 0 0 1-7.316 87.635v-61.391a8.544 8.544 0 0 0-4.4-7.213Zm21.742-32.69-1.535-.922-51.619-30.081a8.39 8.39 0 0 0-8.492 0L99.98 99.808V74.587a.716.716 0 0 1 .307-.665l52.233-30.133a48.652 48.652 0 0 1 72.236 50.391v.205ZM88.061 139.097l-21.845-12.585a.87.87 0 0 1-.41-.614V65.685a48.652 48.652 0 0 1 79.757-37.346l-1.535.87-51.67 29.825a8.595 8.595 0 0 0-4.246 7.367l-.051 72.697Zm11.868-25.58 28.138-16.217 28.188 16.218v32.434l-28.086 16.218-28.188-16.218-.052-32.434Z',
    }),
    claude: Object.freeze({
        label: 'Claude Code',
        viewBox: '0 0 24 24',
        path: 'm4.7144 15.9555 4.7174-2.6471.079-.2307-.079-.1275h-.2307l-.7893-.0486-2.6956-.0729-2.3375-.0971-2.2646-.1214-.5707-.1215-.5343-.7042.0546-.3522.4797-.3218.686.0608 1.5179.1032 2.2767.1578 1.6514.0972 2.4468.255h.3886l.0546-.1579-.1336-.0971-.1032-.0972L6.973 9.8356l-2.55-1.6879-1.3356-.9714-.7225-.4918-.3643-.4614-.1578-1.0078.6557-.7225.8803.0607.2246.0607.8925.686 1.9064 1.4754 2.4893 1.8336.3643.3035.1457-.1032.0182-.0728-.164-.2733-1.3539-2.4467-1.445-2.4893-.6435-1.032-.17-.6194c-.0607-.255-.1032-.4674-.1032-.7285L6.287.1335 6.6997 0l.9957.1336.419.3642.6192 1.4147 1.0018 2.2282 1.5543 3.0296.4553.8985.2429.8318.091.255h.1579v-.1457l.1275-1.706.2368-2.0947.2307-2.6957.0789-.7589.3764-.9107.7468-.4918.5828.2793.4797.686-.0668.4433-.2853 1.8517-.5586 2.9021-.3643 1.9429h.2125l.2429-.2429.9835-1.3053 1.6514-2.0643.7286-.8196.85-.9046.5464-.4311h1.0321l.759 1.1293-.34 1.1657-1.0625 1.3478-.8804 1.1414-1.2628 1.7-.7893 1.36.0729.1093.1882-.0183 2.8535-.607 1.5421-.2794 1.8396-.3157.8318.3886.091.3946-.3278.8075-1.967.4857-2.3072.4614-3.4364.8136-.0425.0304.0486.0607 1.5482.1457.6618.0364h1.621l3.0175.2247.7892.522.4736.6376-.079.4857-1.2142.6193-1.6393-.3886-3.825-.9107-1.3113-.3279h-.1822v.1093l1.0929 1.0686 2.0035 1.8092 2.5075 2.3314.1275.5768-.3218.4554-.34-.0486-2.2039-1.6575-.85-.7468-1.9246-1.621h-.1275v.17l.4432.6496 2.3436 3.5214.1214 1.0807-.17.3521-.6071.2125-.6679-.1214-1.3721-1.9246L14.38 17.959l-1.1414-1.9428-.1397.079-.674 7.2552-.3156.3703-.7286.2793-.6071-.4614-.3218-.7468.3218-1.4753.3886-1.9246.3157-1.53.2853-1.9004.17-.6314-.0121-.0425-.1397.0182-1.4328 1.9672-2.1796 2.9446-1.7243 1.8456-.4128.164-.7164-.3704.0667-.6618.4008-.5889 2.386-3.0357 1.4389-1.882.929-1.0868-.0062-.1579h-.0546l-6.3385 4.1164-1.1293.1457-.4857-.4554.0608-.7467.2307-.2429 1.9064-1.3114Z',
    }),
    cursor: Object.freeze({
        label: 'Cursor',
        viewBox: '0 0 24 24',
        path: 'M11.503.131 1.891 5.678a.84.84 0 0 0-.42.726v11.188c0 .3.162.575.42.724l9.609 5.55a1 1 0 0 0 .998 0l9.61-5.55a.84.84 0 0 0 .42-.724V6.404a.84.84 0 0 0-.42-.726L12.497.131a1.01 1.01 0 0 0-.996 0M2.657 6.338h18.55c.263 0 .43.287.297.515L12.23 22.918c-.062.107-.229.064-.229-.06V12.335a.59.59 0 0 0-.295-.51l-9.11-5.257c-.109-.063-.064-.23.061-.23',
    }),
    opencode: Object.freeze({
        label: 'OpenCode',
        viewBox: '0 0 24 24',
        path: 'M22 24H2V0h20zM17 4.8H7v14.4h10z',
    }),
});

export const GENERIC_HARNESS_MARK = Object.freeze({
    viewBox: '0 0 24 24',
    path: 'M12 2 22 12 12 22 2 12Z',
});

const GENERIC_LABELS = Object.freeze({ agy: 'Antigravity' });
const API_IDS = new Set(['api', 'api_chat', 'api_model', 'native']);

/** Return presentation facts only; callers retain status and evidence ownership. */
export function harnessPresentation(harnessId, { label = '', channel = '' } = {}) {
    const rawId = String(harnessId ?? '').trim();
    const normalized = rawId.toLowerCase();
    const isApi = channel === 'api' || API_IDS.has(normalized);
    if (isApi) {
        return Object.freeze({
            kind: 'channel',
            id: 'api',
            harnessId: null,
            label: String(label || '').trim() || 'API',
            known: true,
            generic: true,
            ...GENERIC_HARNESS_MARK,
        });
    }
    const branded = HARNESS_MARKS[normalized];
    const visibleLabel = String(label || '').trim()
        || branded?.label || GENERIC_LABELS[normalized] || rawId || 'Unknown harness';
    return Object.freeze({
        ...(branded || GENERIC_HARNESS_MARK),
        kind: 'harness',
        id: normalized || rawId,
        harnessId: rawId,
        label: visibleLabel,
        known: Boolean(branded || GENERIC_LABELS[normalized]),
        generic: !branded,
    });
}

/** Decorative currentColor mark plus adjacent, always-visible escaped text. */
export function harnessIdentityMarkup(harnessId, {
    label = '', channel = '', className = '',
} = {}) {
    const presentation = harnessPresentation(harnessId, { label, channel });
    const classes = ['harness-identity', className].filter(Boolean).join(' ');
    const dataIdentity = presentation.kind === 'channel' ? 'api' : presentation.harnessId;
    return `<span class="${escapeHtml(classes)}" data-presentation-kind="${presentation.kind}" data-harness-identity="${escapeHtml(dataIdentity || '')}">`
        + `<svg class="harness-identity-mark" viewBox="${presentation.viewBox}" width="1em" height="1em" fill="currentColor" aria-hidden="true" focusable="false"><path d="${presentation.path}"></path></svg>`
        + `<span class="harness-identity-label">${escapeHtml(presentation.label)}</span></span>`;
}

/** Shared visible identity for one optional named harness account. */
export function harnessAccountIdentityMarkup(harnessId, {
    label = '', profile = '', className = '',
} = {}) {
    const identity = harnessIdentityMarkup(harnessId, { label, className });
    return identity + (profile ? ` (${escapeHtml(profile)})` : '');
}

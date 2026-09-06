// One line of truth for an extension "action" whose response carries no
// `message`: every top-level object with a `state` string is summarised as
// `key: state (reason)`, so a degraded bridge is reported as what it is
// instead of the generic "Saved." (#376). Summaries take the neutral tone —
// the text is a status report, not a success claim.
export function extensionActionStatus(data, fallback = 'Saved.') {
    if (!data || typeof data !== 'object' || Array.isArray(data)) return { text: fallback, tone: 'ok' };
    if (typeof data.message === 'string' && data.message.trim()) return { text: data.message, tone: 'ok' };
    const parts = [];
    for (const [key, value] of Object.entries(data)) {
        if (!value || typeof value !== 'object' || Array.isArray(value)) continue;
        if (typeof value.state !== 'string' || !value.state.trim()) continue;
        const reason = [value.reason_code, value.message].find((item) => typeof item === 'string' && item.trim());
        parts.push(reason ? `${key}: ${value.state} (${reason})` : `${key}: ${value.state}`);
    }
    return parts.length ? { text: parts.join(' · '), tone: 'muted' } : { text: fallback, tone: 'ok' };
}

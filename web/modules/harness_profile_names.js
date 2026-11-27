// Profile-name contract — the ONE pure owner of the engine's registration slug
// rule and of the loop-until-stable submission form.
//
// Two forms live here: `normalizeProfileName` (the slug itself) and
// `profileNameSubmission` (what a typed name becomes when the user submits
// it). The login card applies the second; the Add-account prompt applies the
// first. Both are pure and node-tested without a DOM.

export function normalizeProfileName(raw) {
    // The ENGINE's registration contract is the authority: a profile id must
    // match ^[a-z0-9][a-z0-9_-]{0,63}$ (Claudexor profile-registration slug).
    // Lowercase, map every other character to '-', strip separators the slug
    // may not START with, cap at the engine's 64, and return '' for anything
    // that still cannot satisfy the contract (e.g. a name with no ASCII
    // alphanumerics at all) — an empty result makes the dialog/card ask again
    // instead of submitting a name the engine would refuse. Lives with the
    // profile-name contract; `harness_login_cards.js` (and, through it,
    // `harness_accounts.js`) re-exports the established path.
    const mapped = String(raw || '').trim().toLowerCase().replace(/[^a-z0-9_-]/g, '-');
    const slug = mapped.replace(/^[-_]+/, '').slice(0, 64);
    return /^[a-z0-9][a-z0-9_-]{0,63}$/.test(slug) ? slug : '';
}

export function profileNameSubmission(raw) {
    // The same loop-until-stable rule `promptProfileName` (Add account)
    // applies: a submitted name STARTS a login only when it already IS its
    // normalized form; otherwise the normalized spelling is handed back,
    // editable, with a note — never rewritten silently. Pure for node tests.
    const typed = String(raw || '').trim();
    const normalized = normalizeProfileName(typed);
    if (!normalized) {
        return { profile: '', normalized: '',
            note: 'Enter a name that starts with a lowercase letter or digit — '
                + 'letters, digits, "-" and "_", at most 64 characters.' };
    }
    if (normalized !== typed) {
        return { profile: '', normalized,
            note: `"${typed}" will be saved as "${normalized}" — edit the name or press the button again to continue.` };
    }
    return { profile: normalized, normalized, note: '' };
}

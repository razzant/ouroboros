// What a private compatible key says of the server at its address, and when it offers the consent box: one place for
// both hosts of a key's form (Settings' key page, the first-run wizard's rows). The words are composed from typed facts:
// the saved ones (`_meta.connections[].unidentified`, ouroboros/llm_ollama_guard.py `connection_facts`) and a Test's
// own answer (`code`). Nothing here compares versions: the backend says which Ollama is too old (`update_to`).
// docs/DESIGN.md §7 Accounts; docs/CLIENT_CONTRACT.md §4.

const RISK = 'A server that does not show which program it runs may shorten a long conversation without saying so.';
const HOLD = "Nothing is sent there until this key's server identifies itself: let its metadata be read (GET /api/version) or use another key.";
const NEEDS = 'A request goes to one only with your consent.';
const PRIOR = 'This key may send to one: its server is on record answering a request before this check existed.';
const DRAFT = 'This address has not been checked: what was read at the saved one does not apply here. Test checks it.';
const UNREADABLE = 'What earlier checks stored about this address could not be read; its stored facts are unknown until it can be.';
/** The stored word for a consent explicitly withdrawn where a launch environment holds one; every reader
 *  takes it for no consent (pairs `ouroboros/llm_ollama_guard.py` `CONSENT_REVOKED`). */
export const CONSENT_REVOKED = 'revoked';

/** The Test answers (`ProviderTestResponse.code`) that name why the check sent nothing, with the chip the key's head wears. */
const WITHHELD = {
    ollama_guard_consent: 'Needs your consent',
    ollama_guard_held: 'Ollama was seen at this address',
    ollama_guard_unavailable: 'Ollama needs an update',
};
export const CONSENT_CODE = 'ollama_guard_consent';
export const guardCode = (result) => (Object.hasOwn(WITHHELD, result?.code) ? result.code : '');
export const guardChip = (code) => WITHHELD[code] || '';

const day = (iso) => {
    const when = new Date(iso || '');
    return Number.isFinite(when.getTime()) ? ` on ${when.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}` : '';
};
const named = (fact) => Boolean(fact?.checked?.type) && fact.checked.type !== 'unknown';

/**
 * A private key's note: Ollama read at its address (any key's, so it never says whose server it is), what this key's own
 * last check identified, then what follows. `state` names the server as known: `unchecked`, `unconfirmed` (no program
 * named), `identified`, `update` (this check named an Ollama the backend says is too old), `held` (Ollama was seen at the
 * address and this key's server did not identify itself: nothing is sent, whatever is ticked or set up before), `draft`
 * (`moved`: the address typed is not the one the facts speak of). '' for a key the check does not concern.
 */
export function serverNote(entry, { moved = false } = {}) {
    const fact = entry?.unidentified;
    if (!fact) return { state: '', text: '', admission: '' };
    if (moved) return { state: 'draft', text: DRAFT, admission: '' };
    if (fact.evidence === 'unreadable') {  // the stored facts defer; what the settings themselves prove still shows
        const standing = fact.admission || '';
        return { state: 'unchecked', admission: standing,
            text: `${UNREADABLE} ${standing === 'consent' ? RISK : `${RISK} ${NEEDS}`}` };
    }
    const { checked, ollama } = fact;
    const last = checked?.type;
    const own = last === 'ollama';
    const shown = (row) => (row?.version ? ` ${row.version}` : '');
    const admission = fact.admission || '';
    if (own && checked.update_to) {
        return { state: 'update', admission, text: `Identified as Ollama${shown(checked)} at the last check${day(checked.observed_at)}. It can silently truncate history: update Ollama to ${checked.update_to} or later. Nothing is sent until then.` };
    }
    const seen = ollama && !own ? `Ollama${shown(ollama)} was seen at this address${day(ollama.observed_at)}.${ollama.update_to
        ? ` It can silently truncate history until updated to ${ollama.update_to} or later.` : ''}` : '';
    const type = named(fact) ? `Identified as ${own ? 'Ollama' : last}${shown(checked)} at the last check${day(checked.observed_at)}.`
        : last ? `Server type not confirmed at the last check${day(checked.observed_at)}.` : '';
    const next = named(fact) ? '' : ollama ? HOLD
        : admission === 'consent' ? RISK : admission === 'prior' ? `${RISK} ${PRIOR}` : `${RISK} ${NEEDS}`;
    const state = named(fact) ? 'identified' : ollama ? 'held' : last ? 'unconfirmed' : 'unchecked';
    return { state, admission, text: [seen, type, next].filter(Boolean).join(' ') };
}

/** Whether a consent can matter to the facts' server: private, naming no program, and no Ollama seen at its address. */
export const consentMatters = (fact) => Boolean(fact) && !named(fact) && !fact.ollama;

/**
 * The key's consent box and note, painted from what is known now: `entry` the saved facts of the key, `moved` that the
 * address typed differs from the one they speak of, `outcome` what a Test of this draft was withheld for (`guardCode`;
 * it stays until the address changes). The box is offered by that Test when it asked for the consent, else by the
 * facts, and while it is ticked, to be withdrawn; a Test that found a hold or an old Ollama takes the offer back, since
 * no consent lifts either.
 */
export function paintServerCheck(scope, { entry = null, moved = false, outcome = '' } = {}) {
    const label = scope.querySelector('.accounts-key-consent');
    const box = scope.querySelector('[data-connection-consent]');
    // While an applied consent is being withdrawn on the unmoved address the box stays visible: a pending
    // withdrawal must be seen, and reversible, until Save lands it (the reload flips `applied` to '0').
    const withdrawing = !outcome && !moved && box?.dataset.applied === '1';
    const reason = outcome === CONSENT_CODE ? 'test' : !outcome && !moved && consentMatters(entry?.unidentified) ? 'facts'
        : box?.checked || withdrawing ? 'ticked' : '';
    if (label) {
        label.hidden = !reason;
        if (reason) label.dataset.consentOffer = reason; else delete label.dataset.consentOffer;
    }
    const note = scope.querySelector('[data-connection-unidentified]');
    if (!note) return;
    const { state, text, admission } = serverNote(entry, { moved });
    Object.assign(note, { textContent: text, hidden: !text });
    for (const [name, value] of [['serverState', state], ['admission', admission]]) {
        if (text && value) note.dataset[name] = value; else delete note.dataset[name];
    }
}

/** A key's consent box as read: ticked while the stored consent covers the address it was read with. */
export function restoreConsent(box, { stands, url }) {
    box.checked = Boolean(stands);
    Object.assign(box.dataset, { applied: stands ? '1' : '0', appliedUrl: url });
}

/** A new address was typed: the box stands for the address shown, so it is ticked only while that is the stored one. */
export function retargetConsent(box, url) {
    box.checked = box.dataset.applied === '1' && url === box.dataset.appliedUrl;
}

/** The box as an edit, `null` while it stands as read. Ticked, it names the address shown (`url`, as it is saved); cleared,
 *  a consent that stood is withdrawn (`''`). */
export function consentEdit(box, url) {
    if (!box) return null;
    const stood = box.dataset.applied === '1';
    if (box.checked) return stood && url === box.dataset.appliedUrl ? null : url;
    return stood ? '' : null;
}

/** A test's line: its outcome, then the `caution` of a request that reached `/v1` on a server which named no program, whatever the outcome. */
export function providerTestStatusText(result = {}) {
    const caution = String(result?.caution || '').trim();
    if (result?.ok === true) return caution ? `Works. ${caution}` : 'Works';
    const reason = String(result?.error || '').trim();
    const failed = reason ? `Not ready — ${reason}` : 'Not ready';
    return caution ? `${failed}${/[.!?]$/.test(failed) ? '' : '.'} ${caution}` : failed;
}

/** Its tone: a passed test that reached a server which named no program is a warning, not a plain success; a request the
 *  check withheld (`code`) is a decision for its owner, not a provider failure. */
export const providerTestTone = (result) => (result?.ok === true ? (result.caution ? 'warn' : 'ok') : guardCode(result) ? 'warn' : 'danger');

/** A failed request reads "Not ready"; only the server's typed refusal adds its own controlled reason. */
export function providerTestNetworkErrorStatus(error) {
    const body = error?.status === 400 && error?.body;
    return body?.code === 'masked_key_endpoint_changed' ? providerTestStatusText(body) : 'Not ready';
}

/** A Test's result line as a host keeps it: words, tone and the typed code the check withheld for (''). */
export const testVerdict = (data) => ({ text: providerTestStatusText(data), tone: providerTestTone(data), code: guardCode(data) });

/** What a draft's Test says of its box next: the code it was withheld for, none when it passed on a server that named its
 *  program, and what stood before when it failed otherwise or passed on a caution (the consent still matters). */
export const nextOutcome = (answered, before = '') => guardCode(answered) || (answered?.ok === true && !answered.caution ? '' : before);

/** The code on the result node: the stable hook of what the check withheld for. */
export function markTestCode(node, code) {
    if (!node) return;
    if (code) node.dataset.testCode = code; else delete node.dataset.testCode;
}

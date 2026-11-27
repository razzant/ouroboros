// What the owner is shown is what a sign-in, a Disconnect and a Revoke act on (docs/DESIGN.md "Sign-ins on a skill's card"): the
// words of a skill that may not start a sign-in, of a Revoke the provider did not confirm or was never asked, of a changed consent
// and of the two confirmations. DOM-free: the renderers are pure over the status shape.
import assert from 'node:assert/strict';
import test from 'node:test';

import { admissionWords, confirmSentence, failureWords, signInRowsHtml, signInTray } from '../modules/skill_sign_in.js';

const text = (html) => String(html).replace(/<[^>]+>/g, '').replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim();
const account = (overrides = {}) => ({ key: 'a-1', label: 'personal@example.test', subject: '1', verified: true, state: 'signed_in',
    since: '2026-11-14T12:00:00+00:00', scope: ['openid', 'email'], missing_required: [], not_allowed: [],
    client: { id: 'c-1', label: 'Personal' }, via: 'card', siblings: [], consent: 'k'.repeat(32), ...overrides });
const service = (overrides = {}) => ({ skill: 'calendar', service: 'google', label: 'Google', issuer: 'https://accounts.google.com',
    scopes: ['openid', 'email'], accounts_mode: 'many', optional: false, accounts: [], pending: null, clients: [], ...overrides });
const status = (...services) => ({ ok: true, services, return_address: 'http://127.0.0.1:8765/onboarding/static/provider-return.html' });
const REFUSED = { state: 'sign_in_required', reason: 'refresh_rejected', since: '2026-11-21T12:00:00+00:00' };

test('a skill that may not act says why in place of Sign in, and its accounts keep Disconnect and Revoke', () => {
    assert.equal(text(admissionWords(service({ admission: 'disabled' }))), 'calendar is off. Turn it on, then sign in.');
    assert.equal(text(admissionWords(service({ admission: 'not_reviewed' }))), 'calendar’s current files have no review that lets them run. Review them, then sign in.');
    assert.equal(text(admissionWords(service({ admission: 'not_trusted' }))), 'This installation doesn’t run calendar’s code, so it can’t sign in here.');
    const off = signInRowsHtml(service({ admission: 'disabled', accounts: [account(REFUSED)] }));
    assert.match(off, /data-sign-in-admission/);
    assert.match(text(off), /calendar is off\. Turn it on, then sign in\./);
    assert.doesNotMatch(off, /data-sign-in-act="begin"/, 'no Sign in again while the skill may not start one');
    assert.match(off, /aria-label="More for personal@example\.test" data-sign-in-act="menu"/, 'the menu with Disconnect and Revoke stays');
    assert.match(signInRowsHtml(service({ accounts: [account(REFUSED)] })), /data-sign-in-act="begin" data-sign-in-account="a-1">Sign in again</);
    assert.doesNotMatch(signInRowsHtml(service({ accounts: [account()] })), /data-sign-in-admission/);
    assert.equal(signInTray('calendar', status(service({ admission: 'not_reviewed', accounts: [account(REFUSED)] }))), '', 'no tray for a sign-in nobody can start');
    assert.equal(signInTray('calendar', status(service({ admission: 'disabled' }))), '');
    assert.match(signInTray('calendar', status(service())), /data-sign-in-account="" data-sign-in-begin>Sign in</);
});

test('a Revoke the provider did not confirm or was never asked keeps the sign-in and sends the owner to the provider, never to repeat it', () => {
    const google = service();
    const words = (result) => failureWords(google, result, { account: 'personal@example.test' });
    assert.equal(words({ reason: 'revocation_not_confirmed', revocation_reason: 'not_confirmed' }),
        'Google didn’t confirm the revocation, and the sign-in for personal@example.test is kept here. Check your Google account and remove the permission there if it is still listed, then choose Disconnect here.');
    assert.equal(words({ reason: 'revocation_not_asked', revocation_reason: 'no_endpoint' }),
        'Google publishes no revocation address, so it wasn’t asked and the sign-in for personal@example.test is kept here. Remove the permission in your Google account, then choose Disconnect here.');
    assert.equal(words({ reason: 'revocation_not_asked', revocation_reason: 'no_tokens' }),
        'No token is saved for personal@example.test, so there was nothing to revoke at Google; the sign-in is kept here. Remove the permission in your Google account if it is still listed, then choose Disconnect here.');
    assert.equal(words({ reason: 'revocation_not_asked', revocation_reason: 'provider_approval_required' }),
        'Google wasn’t asked: its saved addresses can’t be confirmed. The sign-in for personal@example.test is kept here. Remove the permission in your Google account, then choose Disconnect here.');
    for (const why of ['not_confirmed', 'no_endpoint', 'no_tokens', 'provider_approval_required']) {
        const reason = why === 'not_confirmed' ? 'revocation_not_confirmed' : 'revocation_not_asked';
        assert.doesNotMatch(words({ reason, revocation_reason: why }), /again|retry|repeat/i, why);
    }
    assert.equal(words({ reason: 'consent_changed' }), 'personal@example.test was signed in again after you opened this. Review it and choose again.');
    assert.equal(words({ reason: 'skill_disabled' }), 'calendar is off. Turn it on, then sign in.');
    assert.equal(words({ reason: 'skill_not_reviewed' }), 'calendar’s current files have no review that lets them run. Review them, then sign in.');
    assert.equal(words({ reason: 'skill_not_trusted' }), 'This installation doesn’t run calendar’s code, so it can’t sign in here.');
});

test('Disconnect no longer promises a Revoke the row cannot offer, and Revoke says when the sign-in is forgotten', () => {
    const google = service();
    const disconnect = text(confirmSentence(google, account(), 'disconnect'));
    assert.equal(disconnect, 'calendar stops using personal@example.test and this installation forgets the sign-in. Google keeps the permission until you remove it in your Google account; to end it there now, cancel and choose Revoke instead.');
    assert.doesNotMatch(disconnect, /revoke it, here/);
    assert.match(text(confirmSentence(google, account(), 'revoke')), /forgotten here only once Google confirms\.$/);
});

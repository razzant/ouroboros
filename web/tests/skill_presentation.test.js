// Words shared by every Skills surface: ages, versions, popularity (unknown is
// said, never 0), who lines, trust sentences, permissions and the Official promise.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    catalogWhoHtml, chipHtml, declaredPermissionWords, factsText, officialMarkHtml, officialPromise, permissionWords, popularityText, relativeAge,
    routeExternalLink, sourceAboutText, statusHtml, trayHtml, trustSentence, trustValueText, versionText,
} from '../modules/skill_presentation.js';

const NOW = Date.parse('2026-10-17T15:00:00Z');

test('ages read as people say them', () => {
    assert.equal(relativeAge(NOW - 30_000, NOW), 'just now');
    assert.equal(relativeAge(NOW - 12 * 60_000, NOW), '12 min ago');
    assert.equal(relativeAge(NOW - 5 * 3600_000, NOW), '5 h ago');
    assert.equal(relativeAge(NOW - 3 * 86400_000, NOW), '3 days ago');
    assert.equal(relativeAge(NOW - 49 * 86400_000, NOW), '7 weeks ago');
    assert.equal(relativeAge(NOW - 120 * 86400_000, NOW), '4 months ago');
    assert.equal(relativeAge('', NOW), '');
    assert.equal(relativeAge('not a date', NOW), '');
});

test('version labels are opaque and never double-prefixed', () => {
    assert.equal(versionText('0.3.2'), 'v0.3.2');
    assert.equal(versionText('v1.0'), 'v1.0');
    assert.equal(versionText(''), '');
});

test('popularity: one number with its noun; counted zero, uncounted and unknown are different words', () => {
    assert.equal(popularityText({ signals: { votes: 25 } }), '▲ 25 votes');
    assert.equal(popularityText({ signals: { votes: 1 } }), '▲ 1 vote');
    assert.equal(popularityText({ signals: { votes: 0 } }), 'No votes yet');
    assert.equal(popularityText({ signals: { comments: 3 } }), 'Votes not counted yet');
    assert.equal(popularityText({ signals: null }), 'No data');
    assert.equal(popularityText({ source_id: 'builtin', signals: null }), '', 'built-ins have no discussion to count');
    // A registry hit carries the registry's own counts (stars, downloads), never votes.
    assert.equal(popularityText({ signals: null, registry_stats: { stars: 429, downloads: 171800, comments: 6, installs: 6520 } }), '★ 429 · 171.8k downloads · 6 comments');
    assert.equal(popularityText({ signals: null, registry_stats: { stars: 0, downloads: 887 } }), 'No stars yet · 887 downloads');
    assert.equal(popularityText({ signals: null, registry_stats: {} }), 'No data', 'a registry hit without counts is unknown');
    assert.equal(popularityText({ signals: { stars: 4 } }), 'Votes not counted yet', 'signals carry votes; stars there are not a vote count');
    assert.equal(factsText({ signals: { votes: 20 }, needs_keys: ['A', 'B'] }), '▲ 20 votes · needs 2 keys');
    assert.equal(factsText({ signals: { votes: 2 }, needs_keys: ['A'] }), '▲ 2 votes · needs a key');
});

test('the who line names source, author, the author repository and the update age', () => {
    // A pointer entry's `author_repo` is the author's repository; `repo_url` is the source's own and stays out of the line.
    const html = catalogWhoHtml({ source_name: 'Ouroboros Community', owner: 'l1nkberry', repo_url: 'https://github.com/razzant/OuroborosHub',
        author_repo: 'https://github.com/CrossRead/scholion', updated_at: new Date(Date.now() - 2 * 86400_000).toISOString() });
    assert.match(html, /^Ouroboros Community · by l1nkberry · <a class="skills-link" href="https:\/\/github\.com\/CrossRead\/scholion" target="_blank" rel="noopener noreferrer" data-skill-external>CrossRead\/scholion ↗<\/a> · updated 2 days ago$/);
    assert.equal(catalogWhoHtml({ source_name: 'Ouroboros', owner: 'razzant', repo_url: 'https://github.com/razzant/OuroborosHub' }), 'Ouroboros · by razzant',
        'a hosted entry has no author repository; no date is said when the host gives none');
    assert.equal(catalogWhoHtml({ source_name: 'Ouroboros', owner: '' }), 'Ouroboros', 'an unknown author is omitted, never «by unknown»');
    assert.equal(catalogWhoHtml({ source_id: 'builtin' }), 'Comes with the app · updates with Ouroboros');
    assert.doesNotMatch(catalogWhoHtml({ source_name: 'X', author_repo: 'javascript:alert(1)' }), /href/);
});

test('the consequence of Install is the source\'s setting as it was read, said only while it holds; an unconfirmed one is the suggested one', () => {
    const hub = { name: 'Ouroboros', trust: 'trusted' };
    const community = { name: 'Ouroboros Community', trust: 'review' };
    // Every sentence is conditional: the host reads the source again when Install is pressed and installs by that.
    assert.equal(trustSentence({ ...hub, confirmed: true }), 'Installs without review while you trust Ouroboros');
    assert.equal(trustSentence({ ...community, confirmed: true }), 'Reviewed before it runs while that is the setting for Ouroboros Community');
    // The host installs by the row's trust whether or not the owner confirmed it (a new install's hub rows).
    assert.equal(trustSentence({ ...hub, confirmed: false }), 'Installs without review while that is the suggested setting for Ouroboros');
    assert.equal(trustSentence({ ...community, confirmed: false }), 'Reviewed before it runs while that is the suggested setting for Ouroboros Community');
    // «You trust» is the owner's confirmed choice: an entry whose source row is not in its catalog claims none.
    assert.equal(trustSentence(hub), 'Installs without review while that is the setting for Ouroboros');
    assert.equal(trustSentence(community), 'Reviewed before it runs while that is the setting for Ouroboros Community');
    assert.equal(trustSentence({ trust: 'trusted', confirmed: true }), 'Installs without review while you trust its source');
    // Where a skill comes from is not how it is reviewed: the built-in source installs by its own setting.
    assert.equal(trustSentence({ kind: 'builtin', name: 'Built in', trust: 'review', confirmed: true }), 'Reviewed before it runs while that is the setting for Built in');
    assert.equal(trustSentence({ kind: 'builtin', name: 'Built in', trust: 'trusted', confirmed: true }), 'Installs without review while you trust Built in');
    // No trust fact, no promise of a review or of its absence.
    for (const unknown of [null, {}, { name: 'Ouroboros', trust: 'unexpected', confirmed: true }]) {
        assert.equal(trustSentence(unknown), 'Its source’s setting decides the review');
        assert.equal(trustSentence(unknown, { detail: true }), 'Whether Ouroboros reviews it before it runs is its source’s setting in Sources.');
    }
    // The sheet has room to say «the X source»: the product and a source can share the name Ouroboros.
    assert.equal(trustSentence({ ...hub, confirmed: true }, { detail: true }), 'Installs without review while you trust the Ouroboros source.');
    assert.equal(trustSentence({ ...hub, confirmed: false }, { detail: true }), 'Installs without review while that is the suggested setting for the Ouroboros source.');
    assert.equal(trustSentence({ ...community, confirmed: true }, { detail: true }),
        'Ouroboros reviews it on this computer before it runs while that is the setting for the Ouroboros Community source. You can skip the review.');
    assert.equal(trustSentence({ ...community, confirmed: false }, { detail: true }),
        'Ouroboros reviews it on this computer before it runs while that is the suggested setting for the Ouroboros Community source. You can skip the review.');
    assert.equal(trustSentence({ trust: 'review' }, { detail: true }), 'Ouroboros reviews it on this computer before it runs while that is the setting for its source. You can skip the review.');
    // None of them is a bare promise of what a later press does.
    for (const policy of [hub, community].flatMap((row) => [true, false, undefined].map((confirmed) => ({ ...row, confirmed })))) {
        for (const detail of [false, true]) assert.match(trustSentence(policy, { detail }), / while /);
    }
    assert.equal(trustValueText({ trust: 'trusted' }), 'Installs without review');
    assert.equal(trustValueText({ trust: 'review' }), 'Reviews each skill');
    assert.match(sourceAboutText({ kind: 'hub', part: 'official' }), /marked Official by its maintainer/);
    assert.match(sourceAboutText({ kind: 'registry' }), /they appear when you search/);
});

test('the Official promise vouches for the version; how a new install goes is said only of the source the mark is bound to, while its setting holds', () => {
    const lead = 'Official — the maintainer of Ouroboros vouches for this exact version.';
    const hub = { name: 'Ouroboros', url: 'https://github.com/razzant/OuroborosHub', trust: 'trusted', confirmed: true };
    const from = 'It comes from the Ouroboros source (github.com/razzant/OuroborosHub).';
    assert.equal(officialPromise(hub), `${lead} ${from} A new install skips the review while you trust the Ouroboros source.`);
    assert.equal(officialPromise({ ...hub, trust: 'review' }), `${lead} ${from} A new install is still reviewed while that is the setting for the Ouroboros source.`);
    // An unconfirmed row still decides the install: its setting is said, as the suggested one.
    assert.equal(officialPromise({ ...hub, confirmed: false }), `${lead} ${from} A new install skips the review while that is the suggested setting for the Ouroboros source.`);
    assert.equal(officialPromise({ ...hub, trust: 'review', confirmed: false }),
        `${lead} ${from} A new install is still reviewed while that is the suggested setting for the Ouroboros source.`);
    assert.equal(officialPromise({ name: 'Built in', trust: 'trusted', confirmed: true }),
        `${lead} It comes from the Built in source. A new install skips the review while you trust the Built in source.`);
    // No row of its own: no claim of the owner's choice. No name: none invented.
    assert.equal(officialPromise({ name: 'Ouroboros', trust: 'trusted' }),
        `${lead} It comes from the Ouroboros source. A new install skips the review while that is the setting for the Ouroboros source.`);
    assert.equal(officialPromise({ trust: 'review', confirmed: true }),
        `${lead} It comes from its source. A new install is still reviewed while that is the setting for its source.`);
    // The sentence is about a new install while the setting holds, never about a copy installed earlier under
    // another setting, and never a bare promise of what a later press does.
    for (const promise of [true, false, undefined].flatMap((confirmed) => ['trusted', 'review'].map((trust) => officialPromise({ ...hub, trust, confirmed })))) {
        assert.match(promise, /A new install (skips the review|is still reviewed) while /);
        assert.doesNotMatch(promise, /it installs without review|still reviews it|because you chose/);
    }
    // An unbound mark, or a source whose setting is unknown, borrows no source and promises nothing.
    for (const unbound of [null, { name: 'Ouroboros' }, { name: 'Ouroboros', trust: 'unexpected' }]) {
        assert.equal(officialPromise(unbound), `${lead} The mark doesn’t decide the review: a new install follows its source’s setting in Sources.`);
    }
    assert.match(officialMarkHtml(), /^<button type="button" class="skills-mark ui-hit-area" data-mark="official" data-official-mark aria-haspopup="dialog">/);
    assert.match(officialMarkHtml({ control: false }), /^<span class="skills-mark"/);
    assert.match(officialMarkHtml({ sourceId: 'builtin' }), /data-official-source="builtin"/);
    assert.match(officialMarkHtml({ sourceId: 'x"<y' }), /data-official-source="x&quot;&lt;y"/);
});

test('permissions read as plain words, once each', () => {
    assert.deepEqual(permissionWords(['net', 'tool', 'subscribe_event:chat.outbound', 'subscribe_event:chat.typing', 'unknown']),
        ['Internet', 'New tools for Ouroboros', 'Sees app events']);
});

test('declared permissions keep their id when no friendlier word exists, and invent none', () => {
    assert.deepEqual(declaredPermissionWords(['page_module', 'net', 'subscribe_event:chat.typing', 'net', ' page_module ', 'odd:id']),
        ['page_module', 'Internet', 'Sees app events', 'odd:id']);
    // Only a non-empty string is an id: nothing is made of a malformed entry, and a prototype key is an id, not a word.
    assert.deepEqual(declaredPermissionWords(['', ' ', null, undefined, 7, {}, ['net'], 'constructor']), ['constructor']);
    assert.deepEqual(declaredPermissionWords(undefined), []);
    assert.deepEqual(declaredPermissionWords('net'), []);
    // Browse's and the grant dialogs' words are unchanged: an id with no word is left out there.
    assert.deepEqual(permissionWords(['page_module', 'net']), ['Internet']);
});

test('status sentences and trays are markup the page styles, with escaped words', () => {
    // Live work is the shared moving mark (web/modules/status_mark.js) in the sentence's or chip's own ink.
    const moving = '<span class="status-mark" aria-hidden="true" data-shape="moving" data-hue="ink"><i></i><i></i><i></i></span>';
    assert.equal(statusHtml('Installing…', 'muted', { work: true }),
        `<span class="ui-status skills-status" data-tone="muted" data-work>${moving}<span class="skills-status-text">Installing…</span></span>`);
    assert.equal(chipHtml('Reviewing…', 'neutral', { work: true }),
        `<span class="ui-chip skills-chip" data-tone="neutral">${moving}Reviewing…</span>`);
    assert.doesNotMatch(statusHtml('On · v1', 'ok') + chipHtml('On', 'ok'), /status-mark|data-work/, 'at rest a sentence keeps its own dot');
    assert.equal(trayHtml({ sentence: '' }), '');
    assert.match(trayHtml({ sentence: 'One sentence.', note: 'A note.', tone: 'danger' }), /data-tone="danger"[\s\S]*One sentence\.<div class="skills-tray-note">A note\.<\/div>/);
});

test('an external link press goes through the host opener during the gesture', () => {
    const opened = [];
    let prevented = false;
    const link = { getAttribute: () => 'https://github.com/x/y' };
    const event = { target: { closest: (sel) => (sel === 'a[data-skill-external]' ? link : null) }, preventDefault() { prevented = true; } };
    assert.equal(routeExternalLink(event, (url) => { opened.push(url); }), true);
    assert.equal(prevented, true);
    assert.deepEqual(opened, ['https://github.com/x/y']);
    assert.equal(routeExternalLink({ target: { closest: () => null } }), false);
});

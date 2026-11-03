// Words shared by every Skills surface: ages, versions, popularity (unknown is
// said, never 0), who lines, trust sentences, permissions and the Official promise.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    catalogWhoHtml, chipHtml, factsText, officialMarkHtml, officialPromise, permissionWords, popularityText, relativeAge,
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

test('trust sentences and values say the consequence, from the row the owner set', () => {
    assert.equal(trustSentence({ name: 'Ouroboros', trust: 'trusted' }), 'Installs without review — you trust Ouroboros');
    assert.equal(trustSentence({ name: 'Ouroboros Community', trust: 'review' }), 'Ouroboros reviews it before it runs');
    assert.equal(trustSentence(null), 'Ouroboros reviews it before it runs', 'an unknown source is reviewed');
    assert.equal(trustValueText({ trust: 'trusted' }), 'Installs without review');
    assert.equal(trustValueText({ trust: 'review' }), 'Reviews each skill');
    assert.match(sourceAboutText({ kind: 'hub', part: 'official' }), /marked Official by its maintainer/);
    assert.match(sourceAboutText({ kind: 'registry' }), /they appear when you search/);
});

test('the Official promise is bound to the source that set it and follows its trust', () => {
    const trusted = officialPromise({ name: 'Ouroboros', url: 'https://github.com/razzant/OuroborosHub', trust: 'trusted' });
    assert.equal(trusted, 'Official — the maintainer of Ouroboros vouches for this exact version. It comes from the Ouroboros source (github.com/razzant/OuroborosHub), so it installs without review while you trust that source.');
    assert.match(officialPromise({ name: 'Ouroboros', trust: 'review' }), /Ouroboros still reviews it, because you chose to review each skill/);
    assert.match(officialMarkHtml(), /^<button type="button" class="skills-mark" data-mark="official" data-official-mark aria-haspopup="dialog">/);
    assert.match(officialMarkHtml({ control: false }), /^<span class="skills-mark"/);
    assert.match(officialMarkHtml({ sourceId: 'builtin' }), /data-official-source="builtin"/);
    assert.match(officialMarkHtml({ sourceId: 'x"<y' }), /data-official-source="x&quot;&lt;y"/);
    assert.match(officialPromise({ name: 'Built in', trust: 'trusted' }), /the Built in source, so it installs without review/);
});

test('permissions read as plain words, once each', () => {
    assert.deepEqual(permissionWords(['net', 'tool', 'subscribe_event:chat.outbound', 'subscribe_event:chat.typing', 'unknown']),
        ['Internet', 'New tools for Ouroboros', 'Sees app events']);
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

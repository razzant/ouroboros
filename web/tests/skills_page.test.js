// The Skills page controller's pure decisions: lifecycle merge, the Installed
// filter strip, «Update all», publish destinations and the repair prompt.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { createMySkillsDestination } from '../modules/skill_dialogs.js';
import { publicationDisclosure } from '../modules/skill_presentation.js';

import {
    buildHealPrompt, destinationFromChoice, inFlightCount, installedFilterCounts, lifecycleEventsFromQueue,
    matchesInstalledFilter, mergeLifecycleEvents, publishDestinations, updateAllCandidates,
} from '../modules/skills.js';

const LISTING = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/extensions_v4.json', import.meta.url), 'utf8'));
const SOURCES = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/skill_sources.json', import.meta.url), 'utf8')).sources;
const rows = () => structuredClone(LISTING.skills);
const byName = (name) => rows().find((skill) => skill.name === name);

test('the active lifecycle job joins the event stream once', () => {
    const active = { id: 'j2', target: 'b', status: 'running', kind: 'install' };
    assert.deepEqual(lifecycleEventsFromQueue({ events: [{ id: 'j1', target: 'a', status: 'failed' }, active], active }),
        [{ id: 'j1', target: 'a', status: 'failed' }, active]);
    assert.deepEqual(lifecycleEventsFromQueue(null), []);
});

test('an in-flight job annotates the existing card; an unknown target becomes a virtual row; the newest event wins', () => {
    const merged = mergeLifecycleEvents([{ name: 'a', enabled: true }], [
        { target: 'a', status: 'running', kind: 'disable' },
        { target: 'n', status: 'queued', kind: 'install', message: 'Installing n' },
    ]);
    assert.equal(merged.length, 2);
    const a = merged.find((skill) => skill.name === 'a');
    assert.equal(a.lifecycle_pending, true);
    assert.equal(a.lifecycle_kind, 'disable');
    const n = merged.find((skill) => skill.name === 'n');
    assert.equal(n.lifecycle_virtual, true);
    assert.equal(n.description, 'Installing n');
    const settled = mergeLifecycleEvents([{ name: 'a' }], [
        { target: 'a', status: 'running', kind: 'enable' },
        { target: 'a', status: 'succeeded', kind: 'enable' },
    ]);
    assert.equal(settled[0].lifecycle_pending, undefined, 'a newer terminal event clears the older pending one');
    assert.equal(inFlightCount([{ target: 'a', status: 'running' }, { target: 'a', status: 'succeeded' }, { target: 'b', status: 'queued' }]), 1);
});

test('the filter strip counts facts of the loaded list', () => {
    const counts = installedFilterCounts(rows());
    assert.deepEqual(counts, { all: 9, needs: 4, updates: 3, off: 5, made: 2 });
    assert.equal(matchesInstalledFilter(byName('zoom_meeting'), 'needs'), true);
    assert.equal(matchesInstalledFilter(byName('telegram'), 'needs'), false);
    assert.equal(matchesInstalledFilter(byName('weather'), 'updates'), true);
    assert.equal(matchesInstalledFilter(byName('old_tool'), 'updates'), true, 'withdrawn counts as an update to act on');
    assert.equal(matchesInstalledFilter(byName('minecraft-widget'), 'made'), true);
    assert.equal(matchesInstalledFilter({ name: 'v', lifecycle_virtual: true, enabled: false }, 'off'), false);
});

test('«Update all» covers only unmodified updates installed through trust from a row still trusted', () => {
    const list = rows();
    assert.deepEqual(updateAllCandidates(list, SOURCES).map((skill) => skill.name), ['weather']);
    const second = structuredClone(byName('weather'));
    second.name = 'weather2';
    assert.deepEqual(updateAllCandidates([...list, second], SOURCES).map((skill) => skill.name), ['weather', 'weather2']);
    const reviewNow = SOURCES.map((source) => (source.id === 'ouroboros' ? { ...source, trust: 'review' } : source));
    assert.deepEqual(updateAllCandidates([...list, second], reviewNow), [], 'a row switched to review drops out');
    assert.deepEqual(updateAllCandidates([...list, second], null).length, 2, 'without the sources read, the install basis decides');
    const unknownBase = structuredClone(second);
    unknownBase.install.base_known = false;
    assert.deepEqual(updateAllCandidates([unknownBase], SOURCES), [], 'an install whose original is unknown asks what to keep');
});

test('publish destinations: back to the origin, the owner’s git source, the community catalog', () => {
    const withGit = [...SOURCES, { id: 'mine', name: 'My skills repo', kind: 'git', trust: 'trusted' }];
    const madeHere = publishDestinations(byName('minecraft-widget'), withGit);
    assert.deepEqual(madeHere.map((choice) => choice.value), ['source:mine', 'source:ouroboros-community']);
    assert.match(madeHere[0].label, /^My skills repo — your own source$/);
    assert.match(madeHere[1].detail, /New skills and new authors are read by the maintainer first\./);
    // The hub's two rows are one repository: a skill from it goes back as its origin, or beside it as a new skill.
    const fromCommunity = publishDestinations(byName('anime_studio'), withGit);
    assert.deepEqual(fromCommunity.map((choice) => choice.value), ['origin:ouroboros-community', 'source:mine', 'new:ouroboros-community']);
    assert.equal(fromCommunity[0].label, 'Back to Ouroboros Community — where it came from');
    assert.equal(fromCommunity[2].label, 'Ouroboros Community — as a new skill');
    const fromOfficial = publishDestinations(byName('weather'), SOURCES);
    assert.deepEqual(fromOfficial.map((choice) => choice.value), ['origin:ouroboros', 'new:ouroboros-community'],
        'from either half of the hub: never «share with everyone» into the same repository');
    const pointer = byName('anime_studio');
    pointer.install.release.repo = 'https://github.com/CrossRead/anime';
    assert.equal(publishDestinations(pointer, withGit)[0].label, 'Back to CrossRead/anime — its author’s repository',
        'a hub entry that points at its author’s repository goes back there');
    // The destination the host takes ({source_id, mode}: origin | entry | new_entry).
    assert.deepEqual(destinationFromChoice('origin:ouroboros-community'), { source_id: 'ouroboros-community', mode: 'origin' });
    assert.deepEqual(destinationFromChoice('new:ouroboros-community'), { source_id: 'ouroboros-community', mode: 'new_entry' });
    assert.deepEqual(destinationFromChoice('source:mine'), { source_id: 'mine', mode: 'entry' });
    assert.equal(destinationFromChoice(''), null);
    assert.equal(destinationFromChoice('default'), null, 'no connected destination keeps the host default, never an invented source id');
    assert.deepEqual(publishDestinations(byName('telegram'), SOURCES).map((choice) => choice.value), ['source:ouroboros-community']);
    assert.deepEqual(publishDestinations(byName('telegram'), SOURCES.filter((row) => row.kind !== 'hub')), [],
        'nothing to choose: the host publishes to its default');
});

test('the repair prompt carries bounded diagnostics as data', () => {
    const prompt = buildHealPrompt({ ...byName('minecraft-widget'), load_error: 'x'.repeat(5000) });
    assert.match(prompt, /Repair and run the installed Ouroboros skill selected in the Skills UI\./);
    assert.match(prompt, /"name": "minecraft-widget"/);
    assert.match(prompt, /widget_safety/);
    assert.ok(prompt.length < 9000, 'the load error is bounded');
});

test('saved My skills and git origins keep unknown visibility after a private connection', async () => {
    const answer = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/my_skills.json', import.meta.url), 'utf8')).private;
    const connected = await createMySkillsDestination(async () => answer);
    assert.match(publicationDisclosure(connected.private, connected.repository), /your private repository/);
    const saved = publishDestinations(byName('minecraft-widget'), [answer.source])[0];
    assert.equal(saved.repository, answer.repository);
    assert.equal(saved.private, undefined, 'the saved source contains no visibility fact');
    assert.equal(publicationDisclosure(saved.private, saved.repository),
        "Everything in the skill's folder goes to owner/ouroboros-skills; whoever can read that repository can see it.");
    for (const kind of ['git', 'hub']) {
        const source = { ...answer.source, kind };
        const origin = publishDestinations({ install: { source } }, [source])[0];
        assert.equal(origin.repository, answer.repository);
        assert.match(publicationDisclosure(origin.private, origin.repository), /whoever can read that repository/);
    }
    const pointer = byName('anime_studio');
    pointer.install.release.repo = 'https://github.com/author/private-skill';
    const choice = publishDestinations(pointer, SOURCES)[0];
    assert.equal(choice.repository, 'author/private-skill');
    assert.match(publicationDisclosure(choice.private, choice.repository), /goes to author\/private-skill; whoever can read/);
});

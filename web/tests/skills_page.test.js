// The Skills page controller: lifecycle merge, the Installed filter strip,
// «Update all», publish destinations and the repair prompt.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createMySkillsDestination } from '../modules/skill_dialogs.js';
import { opensDetail, publicationDisclosure } from '../modules/skill_presentation.js';
import { browseCardState } from '../modules/skill_browse.js';
import { installedSkillModel, installedSkillRowKey, renderInstalledSkillCard } from '../modules/skill_card_renderer.js';
import { settle, Stub, world } from './helpers/settings_dom.js';

import {
    buildHealPrompt, carryAnsweredFailures, destinationFromChoice, inFlightCount, initSkills, installedFilterCounts, lifecycleEventsFromQueue,
    listedReadyOn, matchesInstalledFilter, mergeLifecycleEvents, publishDestinations, readAnsweredFailures, tickFailures,
    updateAllCandidates, skillsViewsTemplate, visibleJobIds, writeAnsweredFailures,
    SKILLS_ACTIONS,
} from '../modules/skills.js';

const LISTING = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/extensions_v4.json', import.meta.url), 'utf8'));
const SOURCES = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/skill_sources.json', import.meta.url), 'utf8')).sources;
const rows = () => structuredClone(LISTING.skills);
const byName = (name) => rows().find((skill) => skill.name === name);

test('search and chips stay above the one scroller; the views\' actions belong to the band', () => {
    const page = new Stub('div');
    page.innerHTML = skillsViewsTemplate();
    const scroll = page.querySelector('.skills-scroll');
    for (const selector of ['#skills-pane-installed', '#skills-pane-browse', '#skills-pane-sources']) {
        assert.ok(scroll.contains(page.querySelector(selector)), selector);
    }
    // Each view's search row stands over the scroller, never inside it (skills.js: "each view's search
    // row, then their one scroller").
    for (const selector of ['#skills-installed-search', '#skills-installed-filters', '#skills-browse-chrome']) {
        const chrome = page.querySelector(selector);
        assert.ok(chrome, selector);
        assert.equal(scroll.contains(chrome), false, `${selector} stays above the scroller`);
    }
    // Update all and Refresh are the band's end (extensions.js draws them there from SKILLS_ACTIONS);
    // the views block keeps no action of its own, and no tab strip of the retired page survives.
    assert.match(SKILLS_ACTIONS, /id="skills-update-all"/);
    assert.match(SKILLS_ACTIONS, /id="skills-refresh"/);
    for (const selector of ['#skills-update-all', '#skills-refresh', '.skills-tabs']) {
        assert.equal(page.querySelector(selector), null, `${selector} is not part of the views`);
    }
});

test('the chip rails start at the column edge, and on a phone run to the page\'s edges as Logs\' rail does', () => {
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    assert.doesNotMatch(css, /\.skills-(filter|category)-strip \{[^}]*margin(-left)?: [^;}]*calc\(-1 \* var\(--space-2\)\)/,
        'no chip rail starts 8px left of the column');
    assert.match(css, /\.skills-filter-strip \{ flex-basis: 100%; \}/);
    assert.match(css, /html\[data-layout="stack"\] \.skills-filter-strip \{ flex-basis: calc\(100% \+ 2 \* var\(--page-header-padding-x\)\); \}/);
    assert.match(css, /html\[data-layout="stack"\] :is\(\.skills-filter-strip, \.skills-category-strip\) \{[^}]*margin-inline: calc\(-1 \* var\(--page-header-padding-x\)\);\s*padding-inline: var\(--page-header-padding-x\);\s*scroll-padding-inline: var\(--page-header-padding-x\);/);
});

test('Update counts trusted candidates briefly in the band and keeps the full scope in its accessible name', () => {
    const source = readFileSync(new URL('../modules/skills.js', import.meta.url), 'utf8');
    const button = { setAttribute(key, value) { this[key] = value; } };
    const context = vm.createContext({ snapshot: { rawSkills: true }, projected: () => [], paintFilters() {},
        updateAllCandidates: () => Array.from({ length: context.count }), cache: {}, activeTab: 'installed',
        updateAllBtn: button, document: { getElementById: () => null }, lifecycleEventsFromQueue: () => [], inFlightCount: () => 0,
        section: { paintAnswers() {} } });
    vm.runInContext(source.match(/    function paintChrome\(\) \{[\s\S]+?\n    \}/)[0], context);
    for (const count of [4, 0]) {
        context.count = count; context.paintChrome();
        assert.equal(button.textContent, `Update ${count}`);
        assert.equal(button['aria-label'], `Update ${count} trusted skills`);
        assert.equal(button.hidden, count < 2);
    }
});

test('a skill opens Widgets as a jump; another view is the section\'s page, never a private tab', async () => {
    const source = readFileSync(new URL('../modules/skills.js', import.meta.url), 'utf8');
    const shown = [], gone = [];
    const opener = {};
    const context = vm.createContext({ ctx: { showPage: (...args) => shown.push(args) }, activeTab: 'installed', visible: true,
        section: { async open(view, options) { gone.push([view, options.opener]); context.activeTab = view; return true; } },
        browse: { searchFor: (name) => gone.push(['search', name]) } });
    vm.runInContext(source.match(/    async function runInstalledAction\([\s\S]+?\n    \}/)[0], context);
    await context.runInstalledAction('example', 'open_widgets');
    assert.deepEqual(shown, [['widgets']], 'the action does not press a primary map row');
    await context.runInstalledAction('example', 'open_widgets', { opener });
    assert.equal(shown[1][0], 'widgets');
    assert.equal(shown[1][1].opener, opener, 'the sheet hands the same jump its underlying Browse control');
    await context.runInstalledAction('example', 'find_in_browse', { node: opener });
    assert.deepEqual(gone, [['browse', opener], ['search', 'example']], 'Find in Browse jumps from its pressed action');
});

test('the page polls while a review of a skill runs, by the review\'s own facts, never by its chip\'s words', () => {
    const source = readFileSync(new URL('../modules/skills.js', import.meta.url), 'utf8');
    const polled = (row) => {
        const timers = [];
        const context = vm.createContext({
            pollTimer: 0, clearTimeout() {}, setTimeout: (fn, ms) => { timers.push(ms); return 1; }, destroyed: false, visible: true,
            document: { hidden: false }, listingReadState: 'ok', snapshot: { rawSkills: [row], queue: null }, pendingListingHint: false,
            inFlightCount: () => 0, lifecycleEventsFromQueue: () => [], projected: () => [row], reviewingSkills: new Set(),
            installedSkillModel, pageModuleFacts: () => ({}), awaitsNewerListing: () => false, WORK_POLL_MS: 3000, REVIEW_POLL_MS: 15000,
        });
        vm.runInContext(source.match(/    function schedulePoll\(\) \{[\s\S]+?\n    \}/)[0], context);
        context.schedulePoll();
        return timers.length > 0;
    };
    const running = byName('whoop-health');  // its review job runs
    assert.equal(installedSkillModel(running).reviewing, true);
    assert.equal(polled(running), true);
    const update = { ...byName('weather'), install: { ...(byName('weather').install || {}), pending_update: { version: '2.0.0' } } };
    assert.match(renderInstalledSkillCard(update), />Reviewing…</);
    assert.deepEqual([installedSkillModel(update).reviewing, polled(update)], [true, true], 'an update waits for its review');
    const idle = byName('weather');
    assert.deepEqual([installedSkillModel(idle).reviewing, polled(idle)], [false, false]);
    // A chip that only says the word (here a job's own message) is no review: the words decide nothing.
    const wordy = { ...idle, lifecycle_pending: true, lifecycle_kind: 'enable', lifecycle_message: 'Reviewing the files before turning on…' };
    assert.match(renderInstalledSkillCard(wordy), />Reviewing the files before turning on…</);
    assert.deepEqual([installedSkillModel(wordy).reviewing, polled(wordy)], [false, false]);
    // A queue-only card (two folders share the name, so no listed row owns the job) answers the same fact.
    const virtual = (kind) => mergeLifecycleEvents([{ name: 'notes' }, { name: 'notes' }], [{ target: 'notes', status: 'running', kind }])
        .find((row) => row.lifecycle_virtual);
    assert.match(renderInstalledSkillCard(virtual('review')), />Reviewing…</);
    assert.deepEqual([installedSkillModel(virtual('review')).reviewing, installedSkillModel(virtual('install')).reviewing], [true, false]);
});

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

// An installed extension the host lists On, loaded and `ready`: the one row that can outlive an earlier failed Enable.
const READY_EXTENSION = { name: 'mail', type: 'extension', enabled: true, live_loaded: true, live_reason: 'ready', review_status: 'clean', grants: { all_granted: true } };

test('only an affirmatively ready On extension can outlive a failed Enable', () => {
    assert.equal(listedReadyOn(READY_EXTENSION), true);
    const proves = (changes) => listedReadyOn({ ...READY_EXTENSION, ...changes });
    for (const [why, changes] of Object.entries({
        'not an extension': { type: 'script' }, 'an instruction': { type: 'instruction' }, 'no type': { type: undefined },
        'Off': { enabled: false }, 'a load error': { load_error: 'Port busy' }, 'not loaded': { live_loaded: false }, 'load unknown': { live_loaded: undefined },
        'no live reason': { live_reason: undefined }, 'not_extension': { live_reason: 'not_extension' }, 'an unfamiliar reason': { live_reason: 'something_new' },
        'health unavailable': { live_reason: 'health_unavailable' }, 'restart pending': { live_reason: 'companion_refresh_pending' },
        'retained unresolved': { live_reason: 'retained_unresolved' }, 'a process still exiting': { retained_companions: { daemon: 'stop requested' } },
        'missing access': { grants: { all_granted: false, missing_keys: ['KEY'] } }, 'a colliding identity': { identity_collision: true },
        'tools still listing': { mcp_servers: [{ id: 'm', configured: true, listed: false }] },
        'tools failing': { mcp_servers: [{ id: 'm', configured: true, listed: true, error: 'refused' }] },
    })) assert.equal(proves(changes), false, why);
    assert.equal(listedReadyOn(undefined), false);
    assert.equal(listedReadyOn({ ...READY_EXTENSION, lifecycle_virtual: true }), false);
});

test('an answered failed Enable leaves a ready On card; every other failure and the history stay', () => {
    const failed = { id: 'j1', target: 'mail', status: 'failed', kind: 'enable', error: 'Port busy' };
    const answered = () => true;
    const row = (skills, events, options) => mergeLifecycleEvents(skills, events, options).filter((item) => item.name === 'mail');
    const [card] = row([READY_EXTENSION], [failed], { superseded: answered });
    assert.equal(card.lifecycle_status, undefined, 'the answered Enable stops annotating the card');
    assert.equal(card.lifecycle_error, undefined);
    assert.equal(installedSkillModel(card).needs, false);
    const [open] = row([READY_EXTENSION], [failed], { superseded: () => false });
    assert.deepEqual([open.lifecycle_status, open.lifecycle_error, open.lifecycle_job_id, installedSkillModel(open).needs], ['failed', 'Port busy', 'j1', true]);
    // No answer at all, and an unknown one, are no recovery.
    assert.equal(row([READY_EXTENSION], [failed])[0].lifecycle_status, 'failed', 'no proof offered');
    assert.equal(row([READY_EXTENSION], [failed], { superseded: () => undefined })[0].lifecycle_status, 'failed', 'an unknown answer');
    const running = { id: 'j2', target: 'mail', status: 'running', kind: 'enable' };
    assert.equal(row([READY_EXTENSION], [failed, running], { superseded: answered })[0].lifecycle_pending, true, 'the newest running event wins over an answered older failure');
    assert.equal(row([READY_EXTENSION], [running, failed], { superseded: answered })[0].lifecycle_status, undefined, 'a failure newer than the running job is the one answered');
    assert.equal(mergeLifecycleEvents([READY_EXTENSION], [{ ...failed, target: 'ghost' }], { superseded: answered })
        .find((item) => item.name === 'ghost')?.lifecycle_virtual, true, 'a failed install with no listed row stays a row');
});

test('a name the listing carries twice is never answered; its job stands beside the directories, on none of them', () => {
    const failed = { id: 'j1', target: 'mail', status: 'failed', kind: 'enable', error: 'Port busy' };
    const asked = [];
    const answer = (event, listed) => { asked.push(listed.collision_row_id || 'unmarked'); return true; };
    const twins = [{ ...READY_EXTENSION, identity_collision: true, collision_row_id: 'a' }, { ...READY_EXTENSION, identity_collision: true, collision_row_id: 'b' }];
    for (const pair of [twins, [...twins].reverse(), [{ ...READY_EXTENSION }, { ...READY_EXTENSION }]]) {
        const merged = mergeLifecycleEvents(pair, [failed], { superseded: answer });
        assert.deepEqual(merged.filter((item) => !item.lifecycle_virtual).map((item) => [item.collision_row_id, item.lifecycle_status, item.lifecycle_error, item.lifecycle_job_id]),
            pair.map((item) => [item.collision_row_id, undefined, undefined, undefined]), 'both directories keep their identity, and neither carries the job');
        const attempts = merged.filter((item) => item.lifecycle_virtual);
        assert.deepEqual(attempts.map((item) => [item.name, item.lifecycle_status, item.lifecycle_error, item.lifecycle_pending, item.collision_row_id]),
            [['mail', 'failed', 'Port busy', false, undefined]], 'one name-level attempt carries the actual error');
        assert.match(attempts[0].description, /this is not one of them/, 'it says it is an attempt, not another directory');
    }
    assert.deepEqual(asked, [], 'the caller is not even asked for a duplicated name');
    // Its stable row identity is its own (no directory has it, and the one directory left of a name is keyed by the name), so
    // the card patches in place whichever order the listing gives.
    const [attempt, ...directories] = mergeLifecycleEvents(twins, [failed]);
    assert.deepEqual([installedSkillRowKey(attempt), ...directories.map(installedSkillRowKey)], ['attempt:mail', 'collision:a', 'collision:b']);
    const card = renderInstalledSkillCard(attempt);
    assert.match(card, /data-lifecycle-virtual/);
    assert.match(card, /data-skill-row="attempt:mail"/);
    assert.doesNotMatch(card, /skills-toggle|skills-card-menu|<details|collision:|Nothing was installed/,
        'no switch, menu or Details, no directory identity, and no claim that nothing was installed beside installed folders');
    assert.deepEqual([...card.matchAll(/data-skill-action="([^"]+)"/g)].map((match) => match[1]), ['find_in_browse'],
        'the attempt can only Find in Browse: no operation on a directory');
    // A job in flight is the same: it moves no directory's switch, and a settled one makes no row at all.
    for (const status of ['queued', 'running']) {
        const [pending, ...rest] = mergeLifecycleEvents(twins, [{ id: 'j9', target: 'mail', status, kind: 'enable' }]);
        assert.deepEqual([pending.lifecycle_virtual, pending.lifecycle_pending, ...rest.map((item) => item.lifecycle_pending)], [true, true, undefined, undefined], status);
    }
    assert.equal(mergeLifecycleEvents(twins, [{ id: 'j9', target: 'mail', status: 'succeeded', kind: 'enable' }]).length, 2);
    const [single] = mergeLifecycleEvents([READY_EXTENSION, { ...READY_EXTENSION, name: 'other' }], [failed], { superseded: answer });
    assert.equal(single.lifecycle_status, undefined, 'a name listed once beside another is answered');
    assert.deepEqual(asked, ['unmarked']);
});

test('an answered job is settled by its queue id before the join by name: a doubled name, in either order, carries none of it', () => {
    const failed = { id: 'j1', target: 'mail', status: 'failed', kind: 'enable', error: 'Port busy' };
    const twins = [{ ...READY_EXTENSION, identity_collision: true, collision_row_id: 'a' }, { ...READY_EXTENSION, identity_collision: true, collision_row_id: 'b' }];
    const offered = [];
    const options = { answered: (event) => event.id === 'j1', superseded: (event) => { offered.push(event.id); return true; } };
    const events = [failed];
    for (const order of [twins, [...twins].reverse()]) {
        const merged = mergeLifecycleEvents(order, events, options);
        assert.deepEqual(merged.map((item) => item.collision_row_id), order.map((item) => item.collision_row_id));
        assert.deepEqual(merged.map((item) => item.lifecycle_status), [undefined, undefined], 'neither row carries an answered job');
    }
    assert.deepEqual(offered, [], 'a doubled name is never offered a new answer');
    assert.deepEqual(events, [failed], 'the history entry is the queue\'s, untouched');
    // A newer job of the name nobody answered is shown, as one row of its own: neither directory takes it, in either order,
    // and the answered older job's words are nowhere.
    const other = { ...failed, id: 'j2', error: 'Port 9090 busy' };
    for (const order of [twins, [...twins].reverse()]) {
        const merged = mergeLifecycleEvents(order, [failed, other], options);
        assert.deepEqual(merged.filter((item) => item.lifecycle_virtual).map((item) => [item.name, item.lifecycle_error]), [['mail', 'Port 9090 busy']]);
        assert.deepEqual(merged.filter((item) => !item.lifecycle_virtual).map((item) => [item.collision_row_id, item.lifecycle_status, item.lifecycle_error]),
            order.map((item) => [item.collision_row_id, undefined, undefined]), 'both directories keep their identity and carry nothing');
        assert.ok(!JSON.stringify(merged).includes('Port busy'), 'the answered job is not exposed behind the newer one');
    }
    assert.deepEqual(offered, [], 'the newer job of a doubled name is no more offered a new answer');
    // The newest job of a name speaks: an answered one does not expose an older failure, nor hide a newer running job.
    // (Nothing here proves a new answer, so the older failure could only be hidden by the answered newest one.)
    const alone = { answered: options.answered };
    const older = { ...failed, id: 'j0', error: 'Older' };
    assert.equal(mergeLifecycleEvents([READY_EXTENSION], [older, failed], alone)[0].lifecycle_status, undefined);
    assert.equal(mergeLifecycleEvents([READY_EXTENSION], [older], alone)[0].lifecycle_status, 'failed', 'the older one is not itself answered');
    const running = { id: 'j3', target: 'mail', status: 'running', kind: 'enable' };
    assert.equal(mergeLifecycleEvents([READY_EXTENSION], [failed, running], alone)[0].lifecycle_pending, true);
    const asked = [];
    mergeLifecycleEvents([READY_EXTENSION], [failed, running], { answered: (event) => { asked.push(event.id); return true; } });
    assert.deepEqual(asked, [], 'only a failed job is ever asked about');
    // No answer given, or an unknown one: the failure stays.
    assert.equal(mergeLifecycleEvents([READY_EXTENSION], [failed], { answered: () => undefined })[0].lifecycle_status, 'failed');
});

// Two folders of one name, then fewer: the name listed once and Off, or by no folder at all.
const TWO_FOLDERS = [{ ...READY_EXTENSION, enabled: false, identity_collision: true, collision_row_id: 'a' },
    { ...READY_EXTENSION, enabled: false, identity_collision: true, collision_row_id: 'b' }];
const ONE_LEFT = { ...READY_EXTENSION, enabled: false };
// What the page keeps (skills.js): a failed job it reads while its name is listed more than once stays name-level.
const memoryOfSharedNames = (seen = new Set()) => ({ seen, nameLevel: (event, shared) => { if (shared) seen.add(event.id); return seen.has(event.id); } });
const failedEnable = (id, error) => ({ id, target: 'mail', status: 'failed', kind: 'enable', error });

test('a failed job read beside two folders stays an attempt of its own when fewer are listed, and the folder left carries none of it', () => {
    const job = failedEnable('j2', 'Port 9090 busy');
    const events = [job];
    const memory = memoryOfSharedNames();
    const shared = mergeLifecycleEvents(TWO_FOLDERS, events, memory);
    assert.deepEqual([...memory.seen], ['j2'], 'read while the name was listed twice');
    assert.match(shared[0].description, /this is not one of them/);
    const merged = mergeLifecycleEvents([ONE_LEFT], events, memory);
    assert.equal(merged.length, 2);
    const [attempt, folder] = merged;
    assert.deepEqual([attempt.name, attempt.lifecycle_virtual, attempt.lifecycle_name_level, attempt.lifecycle_status, attempt.lifecycle_error, attempt.lifecycle_pending],
        ['mail', true, true, 'failed', 'Port 9090 busy', false], 'the job stays the attempt it was, with its own error');
    assert.deepEqual([folder.name, folder.enabled, folder.lifecycle_virtual, folder.lifecycle_status, folder.lifecycle_error, folder.lifecycle_job_id, folder.lifecycle_pending],
        ['mail', false, undefined, undefined, undefined, undefined, undefined], 'the folder keeps its own state, Off, and nothing of the job');
    assert.deepEqual(merged.map(installedSkillRowKey), ['attempt:mail', 'mail'], 'two rows, two identities: neither is patched or looked up as the other');
    assert.doesNotMatch(attempt.description, /More than one installed folder uses it|not one of them/, 'it no longer says two folders exist');
    assert.match(attempt.description, /seen while more than one installed folder shared the name/);
    assert.match(attempt.description, /can.t say which folder it was for/);
    assert.deepEqual(events, [job], 'the queue\'s history is untouched');
    // The same words stay in any later read of one folder; two folders again says what is true then.
    assert.equal(mergeLifecycleEvents([ONE_LEFT], events, memory)[0].description, attempt.description);
    assert.match(mergeLifecycleEvents([...TWO_FOLDERS].reverse(), events, memory)[0].description, /this is not one of them/);
    // The row is a placeholder: no switch, menu or Details, nothing that operates on the folder of its name; its text is safe.
    const card = renderInstalledSkillCard(attempt);
    assert.match(card, /data-lifecycle-virtual/);
    assert.match(card, /data-skill-row="attempt:mail"/);
    assert.doesNotMatch(card, /skills-toggle|skills-card-menu|<details|collision:|Nothing was installed|More than one installed folder uses it/);
    assert.deepEqual([...card.matchAll(/data-skill-action="([^"]+)"/g)].map((match) => match[1]), ['find_in_browse']);
    assert.match(renderInstalledSkillCard({ ...attempt, name: 'a"b' }), /data-skill-row="attempt:a&quot;b"/);
    assert.match(renderInstalledSkillCard({ ...attempt, lifecycle_error: '<b>busy</b>' }), /&lt;b&gt;busy&lt;\/b&gt;/);
    // Its tray id is in the attempt namespace (as a collision row's is in its own, with a colon); a real skill named
    // `attempt-mail` keeps the plain id, so the two never meet.
    assert.match(card, /id="skill-trays-attempt:mail"/);
    assert.match(renderInstalledSkillCard({ ...byName('weather'), name: 'attempt-mail' }), /id="skill-trays-attempt-mail"/);
});

test('a job first seen in flight beside two folders is remembered too: it stays its own attempt while it runs and after it fails', () => {
    const memory = memoryOfSharedNames();
    const live = { id: 'j9', target: 'mail', status: 'running', kind: 'enable', message: 'Turning on mail' };
    const shared = mergeLifecycleEvents(TWO_FOLDERS, [live], memory);
    assert.deepEqual([...memory.seen], ['j9'], 'read while the name was listed twice, before any failure');
    assert.deepEqual([shared[0].lifecycle_virtual, shared[0].lifecycle_pending], [true, true]);
    assert.match(shared[0].description, /this is not one of them/);
    assert.deepEqual(shared.slice(1).map((item) => item.lifecycle_status), [undefined, undefined], 'neither folder carries it while both are listed');
    // One folder is left while the job still runs: the survivor's toggle, busy state and words stay its own.
    const [attempt, folder] = mergeLifecycleEvents([ONE_LEFT], [live], memory);
    assert.deepEqual([attempt.lifecycle_virtual, attempt.lifecycle_name_level, attempt.lifecycle_status, attempt.lifecycle_pending],
        [true, true, 'running', true]);
    assert.match(attempt.description, /seen while more than one installed folder shared the name/);
    assert.deepEqual([folder.lifecycle_status, folder.lifecycle_pending, folder.lifecycle_job_id, folder.lifecycle_message, folder.lifecycle_error],
        [undefined, undefined, undefined, undefined, undefined], 'the job in flight never moves onto the survivor');
    // It then fails: still its own attempt, with its own error; the survivor still carries none of it.
    const [failedAttempt, survivor] = mergeLifecycleEvents([ONE_LEFT], [{ ...live, status: 'failed', error: 'Port 9090 busy' }], memory);
    assert.deepEqual([failedAttempt.lifecycle_virtual, failedAttempt.lifecycle_name_level, failedAttempt.lifecycle_status, failedAttempt.lifecycle_error],
        [true, true, 'failed', 'Port 9090 busy']);
    assert.match(failedAttempt.description, /seen while more than one installed folder shared the name/);
    assert.deepEqual([survivor.lifecycle_status, survivor.lifecycle_error, survivor.lifecycle_job_id], [undefined, undefined, undefined]);
    // A new job first seen beside the one folder keeps the ordinary behavior: on that row, remembered as nothing.
    const fresh = memoryOfSharedNames();
    const [row] = mergeLifecycleEvents([ONE_LEFT], [{ ...live, id: 'j10' }], fresh);
    assert.deepEqual([row.lifecycle_virtual, row.lifecycle_job_id, row.lifecycle_pending, row.lifecycle_message],
        [undefined, 'j10', true, 'Turning on mail']);
    assert.deepEqual([...fresh.seen], []);
});

test('the active job shadows a queued same-name job in display, not in ambiguity memory', () => {
    const active = { id: 'active', target: 'mail', status: 'running', kind: 'enable' };
    const queued = { ...active, id: 'queued', status: 'queued' };
    const older = failedEnable('older', 'Older failure');
    const memory = memoryOfSharedNames();
    const read = (skills, events, active = null) => mergeLifecycleEvents(skills, lifecycleEventsFromQueue({ events, active }), memory);
    const [first, ...folders] = read(TWO_FOLDERS, [older, active, queued], active);
    assert.equal(first.lifecycle_status, 'running');
    assert.deepEqual(folders, TWO_FOLDERS);
    assert.deepEqual(memory.seen, new Set(['active', 'queued', 'older']));
    for (const job of [queued, { ...queued, status: 'running' }, failedEnable(queued.id, 'Queued job failed')]) {
        const [attempt, survivor] = read([ONE_LEFT], [{ ...active, status: 'succeeded' }, job], job.status === 'running' ? job : null);
        assert.deepEqual([attempt.lifecycle_name_level, attempt.lifecycle_status, survivor], [true, job.status, ONE_LEFT]);
    }
    const [fresh] = read([ONE_LEFT], [queued, { ...active, id: 'fresh' }]);
    assert.deepEqual([fresh.lifecycle_virtual, fresh.lifecycle_job_id, fresh.lifecycle_pending], [undefined, 'fresh', true]);
});

test('a job the page never read beside two folders can only be read as the one listed folder\'s: the first view is all there is', () => {
    // The queue names no directory and is read apart from the listing: with no memory (a new tab, or storage that refuses)
    // nothing says which folder the job was for, so a name listed once shows it on that row, as every failed job of such a name does.
    for (const options of [memoryOfSharedNames(), {}]) {
        const [row, ...rest] = mergeLifecycleEvents([ONE_LEFT], [failedEnable('j1', 'Port 8080 busy')], options);
        assert.deepEqual([rest.length, row.lifecycle_virtual, row.lifecycle_name_level, row.lifecycle_status, row.lifecycle_job_id, row.lifecycle_error],
            [0, undefined, undefined, 'failed', 'j1', 'Port 8080 busy']);
    }
    const memory = memoryOfSharedNames();
    mergeLifecycleEvents([ONE_LEFT], [failedEnable('j1', 'Port 8080 busy')], memory);
    assert.deepEqual([...memory.seen], [], 'a listing of one folder remembers nothing');
    // With no memory at all, two folders still show the job as an attempt for as long as they are listed.
    assert.equal(mergeLifecycleEvents(TWO_FOLDERS, [failedEnable('j1', 'Port 8080 busy')], {})[0].lifecycle_name_level, true);
});

test('a name no folder carries: a job remembered beside several keeps its reading, any other is the ordinary placeholder', () => {
    const none = (id) => mergeLifecycleEvents([], [failedEnable(id, 'Port 8080 busy')], memoryOfSharedNames(new Set(['j7'])));
    const [ordinary] = none('j1');
    assert.deepEqual([ordinary.lifecycle_virtual, ordinary.lifecycle_name_level, installedSkillRowKey(ordinary), ordinary.description], [true, undefined, 'mail', 'Port 8080 busy']);
    assert.match(renderInstalledSkillCard(ordinary), /Nothing was installed\./, 'a missing skill\'s failed job keeps its own words and note');
    const [named] = none('j7');
    assert.deepEqual([named.lifecycle_virtual, named.lifecycle_name_level, installedSkillRowKey(named)], [true, true, 'attempt:mail']);
    assert.equal(named.description, mergeLifecycleEvents([ONE_LEFT], [failedEnable('j7', 'x')], memoryOfSharedNames(new Set(['j7'])))[0].description,
        'the same words and identity as with one folder listed, so the card patches in place when the last one goes');
    assert.doesNotMatch(renderInstalledSkillCard(named), /Nothing was installed/);
});

test('after the folder is left alone, a newer job is an ordinary one and no older job comes back', () => {
    const memory = memoryOfSharedNames(new Set(['j2']));
    const older = failedEnable('j1', 'Port 8080 busy');
    const attemptJob = failedEnable('j2', 'Port 9090 busy');
    const newer = failedEnable('j3', 'Port 7070 busy');
    const events = [older, attemptJob, newer];
    const history = structuredClone(events);
    // A newer failed Enable is on the one folder as on any single folder; the attempt is no longer the newest job of the name.
    const merged = mergeLifecycleEvents([ONE_LEFT], events, memory);
    assert.equal(merged.length, 1);
    assert.deepEqual([merged[0].lifecycle_virtual, merged[0].lifecycle_status, merged[0].lifecycle_job_id, merged[0].lifecycle_error],
        [undefined, 'failed', 'j3', 'Port 7070 busy']);
    assert.ok(!JSON.stringify(merged).includes('9090') && !JSON.stringify(merged).includes('8080'));
    // The newest status speaks for the name: a job in flight annotates the folder; a settled one leaves nothing, and neither
    // brings the attempt back.
    const running = { id: 'j4', target: 'mail', status: 'running', kind: 'enable' };
    const [pending, ...rest] = mergeLifecycleEvents([ONE_LEFT], [attemptJob, running], memory);
    assert.deepEqual([rest.length, pending.lifecycle_virtual, pending.lifecycle_pending, pending.lifecycle_job_id], [0, undefined, true, 'j4']);
    for (const status of ['succeeded', 'cancelled']) {
        const [only, ...more] = mergeLifecycleEvents([ONE_LEFT], [attemptJob, { id: 'j5', target: 'mail', status, kind: 'enable' }], memory);
        assert.deepEqual([more.length, only.lifecycle_virtual, only.lifecycle_status, only.lifecycle_error], [0, undefined, undefined, undefined], status);
    }
    // Answered stays answered: the attempt is not shown again, nor an older failure exposed behind an answered newest job.
    const answeredAttempt = { answered: (event) => event.id === 'j2', nameLevel: memory.nameLevel };
    assert.deepEqual(mergeLifecycleEvents([ONE_LEFT], [attemptJob], answeredAttempt), [ONE_LEFT]);
    assert.deepEqual(mergeLifecycleEvents([ONE_LEFT], [older, attemptJob], answeredAttempt), [ONE_LEFT]);
    const answeredNewest = { answered: (event) => event.id === 'j3', nameLevel: memory.nameLevel };
    assert.deepEqual(mergeLifecycleEvents([ONE_LEFT], [attemptJob, newer], answeredNewest), [ONE_LEFT], 'an answered newest job takes the name from the attempt');
    assert.deepEqual(events, history, 'the history is the queue\'s, untouched');
});

test('no listed row answers an attempt read beside several folders; an ordinary failed Enable of one row is answered as before', () => {
    const attemptJob = failedEnable('j2', 'Port 9090 busy');
    const ordinary = failedEnable('j3', 'Port 7070 busy');
    const asked = [];
    // The caller proves every row it is asked about ready and On.
    const options = { ...memoryOfSharedNames(new Set(['j2'])), superseded: (event, row) => { asked.push([event.id, row.name]); return true; } };
    // The one folder left is proven ready, and cannot say it was the one the job was for: nobody is even asked, and it stays.
    const [held, folder] = mergeLifecycleEvents([ONE_LEFT], [attemptJob], options);
    assert.deepEqual([asked, held.lifecycle_virtual, held.lifecycle_error, folder.lifecycle_status], [[], true, 'Port 9090 busy', undefined]);
    assert.equal(mergeLifecycleEvents([], [attemptJob], options)[0].lifecycle_name_level, true, 'nor does an empty listing answer it');
    assert.deepEqual(asked, []);
    // An ordinary failed Enable of the same single row is asked, and answered: the card stops carrying it.
    assert.deepEqual(mergeLifecycleEvents([ONE_LEFT], [ordinary], options), [ONE_LEFT]);
    assert.deepEqual(asked, [['j3', 'mail']]);
    // The attempt leaves with a newer job of its name, or with the queue's own eviction of it.
    assert.deepEqual(mergeLifecycleEvents([ONE_LEFT], [attemptJob, { id: 'j4', target: 'mail', status: 'succeeded', kind: 'enable' }], options), [ONE_LEFT]);
    assert.deepEqual(mergeLifecycleEvents([ONE_LEFT], [], options), [ONE_LEFT]);
    assert.deepEqual(asked, [['j3', 'mail']]);
});

test('the tab keeps name-level job ids apart from the answered ones, bounded and best effort the same way', async () => {
    const { readNameLevelFailures, writeNameLevelFailures } = await import('../modules/skills.js');
    const data = new Map();
    const store = { getItem: (key) => data.get(key) ?? null, setItem: (key, value) => data.set(key, String(value)), removeItem: (key) => data.delete(key) };
    writeAnsweredFailures(new Set(['a']), store);
    writeNameLevelFailures(new Set(['n1', 'n2']), store);
    assert.deepEqual([...readAnsweredFailures(store)], ['a'], 'the answered ones are not the name-level ones');
    assert.deepEqual([...readNameLevelFailures(store)], ['n1', 'n2']);
    assert.ok([...data.values()].every((value) => JSON.parse(value).every((id) => typeof id === 'string')), 'ids and nothing else');
    // Bounded as the answered ones are: what an accepted queue read no longer lists as failed goes, and no read takes none.
    assert.deepEqual([...carryAnsweredFailures(readNameLevelFailures(store), new Map([['n2', 3]]))], ['n2']);
    assert.deepEqual([...carryAnsweredFailures(readNameLevelFailures(store), undefined)], ['n1', 'n2']);
    writeNameLevelFailures(new Set(), store);
    assert.deepEqual([...readNameLevelFailures(store)], []);
    assert.deepEqual([...readAnsweredFailures(store)], ['a'], 'clearing one leaves the other');
    const refusing = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('full'); }, removeItem() { throw new Error('denied'); } };
    for (const unusable of [refusing, null]) {
        assert.deepEqual([...readNameLevelFailures(unusable)], []);
        assert.doesNotThrow(() => writeNameLevelFailures(new Set(['j1']), unusable));
        assert.doesNotThrow(() => writeNameLevelFailures(new Set(), unusable));
    }
});

test('answered ids stay until an accepted queue read no longer lists their job as failed', () => {
    const answered = new Set(['a', 'b']);
    assert.deepEqual([...carryAnsweredFailures(answered, undefined)], ['a', 'b'], 'no accepted read yet: none is dropped');
    assert.deepEqual([...carryAnsweredFailures(answered, new Map([['b', 3], ['c', 4]]))], ['b'],
        'a job still listed as failed stays; one no longer listed goes; one never answered is not made');
    assert.deepEqual([...carryAnsweredFailures(answered, new Map())], [], 'an accepted queue with no failed job takes them all');
    assert.deepEqual([...answered], ['a', 'b'], 'the earlier snapshot keeps its own');
});

test('name-level ids are pruned by any visible queue record of their job, not only a failed one', () => {
    const events = [{ id: 'run', target: 'mail', status: 'running', kind: 'enable' },
        { id: 'wait', target: 'mail', status: 'queued', kind: 'enable' },
        { id: 'done', target: 'mail', status: 'succeeded', kind: 'enable' },
        { id: 'lost', target: 'mail', status: 'failed', kind: 'enable' },
        { id: '', target: 'mail', status: 'running', kind: 'enable' }, { target: 'mail', status: 'running' },
        { id: 7, target: 'mail', status: 'failed', kind: 'enable' }];
    const visible = visibleJobIds(events);
    assert.deepEqual([...visible].sort(), ['lost', 'run', 'wait'], 'every visibly listed job with a string id, of any visible status');
    assert.deepEqual([...carryAnsweredFailures(new Set(['run', 'done', 'lost', 'gone']), visible)].sort(), ['lost', 'run'],
        'a job still queued or running keeps its ambiguity id; a settled or unlisted one goes');
    assert.deepEqual([...carryAnsweredFailures(new Set(['run']), undefined)], ['run'], 'no accepted read yet: none is dropped');
    assert.deepEqual([...carryAnsweredFailures(new Set(['run']), visibleJobIds([]))], [], 'a queue with nothing visible takes them all');
});

test('this tab keeps only the answered job ids, and a store that refuses costs the page nothing', () => {
    const tab = () => {
        const data = new Map();
        return { data, getItem: (key) => data.get(key) ?? null, setItem: (key, value) => data.set(key, String(value)), removeItem: (key) => data.delete(key) };
    };
    const store = tab();
    writeAnsweredFailures(new Set(['skill-job-1', 'skill-job-2']), store);
    const [key] = store.data.keys();
    assert.equal(key, 'ouroboros.skills.answeredFailures');
    assert.deepEqual(JSON.parse(store.data.get(key)), ['skill-job-1', 'skill-job-2'], 'ids and nothing else');
    assert.deepEqual([...readAnsweredFailures(store)], ['skill-job-1', 'skill-job-2']);
    writeAnsweredFailures(new Set(), store);
    assert.equal(store.data.size, 0, 'no ids, no key');
    for (const garbage of ['', '{', 'null', '[]', '{"a":1}', '"j1"', '[7,null,"",{"a":1}]']) {
        store.data.set(key, garbage);
        assert.deepEqual([...readAnsweredFailures(store)], [], `kept as ${JSON.stringify(garbage)}`);
    }
    store.data.set(key, '["j1",7,"","j2",null]');
    assert.deepEqual([...readAnsweredFailures(store)], ['j1', 'j2'], 'what is no id is dropped');
    const refusing = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('full'); }, removeItem() { throw new Error('denied'); } };
    for (const unusable of [refusing, null]) {
        assert.deepEqual([...readAnsweredFailures(unusable)], []);
        assert.doesNotThrow(() => writeAnsweredFailures(new Set(['j1']), unusable));
        assert.doesNotThrow(() => writeAnsweredFailures(new Set(), unusable));
    }
    // The tab's own store is sessionStorage, and one whose very access is refused is no store.
    const original = Object.getOwnPropertyDescriptor(globalThis, 'sessionStorage');
    try {
        const session = tab();
        Object.defineProperty(globalThis, 'sessionStorage', { configurable: true, value: session });
        writeAnsweredFailures(new Set(['j1']));
        assert.deepEqual([...readAnsweredFailures()], ['j1']);
        assert.equal(session.data.size, 1);
        Object.defineProperty(globalThis, 'sessionStorage', { configurable: true, get() { throw new Error('SecurityError'); } });
        assert.deepEqual([...readAnsweredFailures()], []);
        assert.doesNotThrow(() => writeAnsweredFailures(new Set(['j1'])));
    } finally {
        if (original) Object.defineProperty(globalThis, 'sessionStorage', original);
        else delete globalThis.sessionStorage;
    }
});

test('failure marks follow the accepted queue read: carried over, stamped once, dropped with the job, never invented', () => {
    let tick = 10;
    const next = () => ++tick;
    const failedJob = (id, kind = 'enable') => ({ id, target: 'mail', kind, status: 'failed' });
    const first = tickFailures([failedJob('a'), failedJob('b'), { id: 'c', target: 'mail', kind: 'enable', status: 'running' },
        { target: 'mail', kind: 'enable', status: 'failed' }, { id: '', target: 'mail', kind: 'enable', status: 'failed' }, { id: 7, target: 'mail', kind: 'enable', status: 'failed' }], undefined, next);
    assert.deepEqual([...first], [['a', 11], ['b', 12]], 'only failed jobs with a string id; a running one and an id-less one take none');
    const earlier = new Map(first);
    const second = tickFailures([failedJob('b'), failedJob('d'), { id: 'a', target: 'mail', kind: 'enable', status: 'succeeded' }], earlier, next);
    assert.deepEqual([...second], [['b', 12], ['d', 13]], 'b keeps its mark, d is new, a is gone with its failure');
    assert.deepEqual([...earlier], [['a', 11], ['b', 12]], 'the marks of the visible snapshot are not touched');
    assert.deepEqual([...tickFailures([], earlier, next)], [], 'a queue that lists nothing keeps nothing');
    const events = Array.from({ length: 80 }, (_, index) => failedJob(`job-${index}`));
    const kept = tickFailures(events, tickFailures(events, undefined, next), next);
    assert.equal(kept.size, 80, 'bounded by the queue\'s own cap, however many reads have passed');
    assert.equal(tickFailures([failedJob('active')], undefined, next).size, 1);
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

test('Sources Publish one filters the retained Installed cards even while the refresh is pending and if it fails', async (t) => {
    const { env, state, layers, showPage } = world(t);
    const reply = (body, status = 200) => ({ ok: status === 200, status, json: async () => body });
    const initial = reply({ skills: [
        { name: 'local-example', source: 'external' },
        { name: 'downloaded-example', source: 'external', install: { source: { id: 'catalog-example' } } },
    ] });
    let finish;
    env.routes.set('/api/extensions', () => new Promise((resolve) => { finish = resolve; }));
    env.routes.set('/api/state', reply({ github_token_configured: false }));
    env.routes.set('/api/skills/lifecycle-queue', reply({ events: [] }));
    env.routes.set('/api/skill-sources', reply({ sources: [] }));
    env.routes.set('/api/skill-catalog', reply({ items: [], sources: [] }));
    const controls = initSkills({ state, layers, showPage });
    try {
        const { document } = env;
        document.getElementById('skills-installed-search').value = '';
        const list = document.getElementById('skills-list');
        const query = list.querySelectorAll.bind(list);
        // Supply this one direct-child selector to the Node DOM fixture.
        list.querySelectorAll = (selector) => selector === ':scope > .skills-card'
            ? list.children.filter((node) => node.matches('.skills-card')) : query(selector);
        await controls.open('installed');
        await settle(); // Let optional facts arrive before the first list; this fixture does not model template enrichment.
        finish(initial);
        await settle();
        const answer = document.getElementById('page-skills').querySelector('[data-extensions-open="installed"]')
            .querySelector('[data-extensions-answer]');
        const good = answer.textContent;
        assert.match(good, /^2 installed/);
        const local = list.querySelector('[data-skill="local-example"]');
        const downloaded = list.querySelector('[data-skill="downloaded-example"]');
        assert.equal(local.hidden, false);
        assert.equal(downloaded.hidden, false);

        await controls.open('sources');
        await settle();
        finish = undefined;
        let finishState;
        env.routes.set('/api/state', () => new Promise(resolve => { finishState = resolve; }));
        const pane = document.getElementById('skills-pane-sources');
        await pane.fire('click', { target: pane.querySelector('[data-source-action="made_here"]') });
        assert.equal(typeof finish, 'function', 'the fresh listing has started but has not answered');
        assert.equal(document.getElementById('skills-pane-installed').hidden, false);
        assert.equal(document.getElementById('skills-installed-filters').querySelector('[data-installed-filter="made"]')
            .getAttribute('aria-pressed'), 'true');
        assert.equal(local.hidden, false);
        assert.equal(downloaded.hidden, true, 'the retained cards already match Made here');

        finish(reply({ error: 'fixture unavailable' }, 503));
        await settle();
        finishState(reply({ github_token_configured: false }));
        await settle();
        assert.match(document.getElementById('skills-status').textContent, /Showing the previous list/);
        assert.equal(answer.textContent, `Last known: ${good}`);
        assert.equal(list.querySelector('[data-skill="local-example"]'), local);
        assert.equal(list.querySelector('[data-skill="downloaded-example"]'), downloaded);
        assert.equal(local.hidden, false);
        assert.equal(downloaded.hidden, true);
        const retry = document.getElementById('skills-refresh').fire('click');
        finishState(reply({ github_token_configured: false }));
        finish(reply({ skills: [] }));
        await retry;
        await settle();
        assert.equal(answer.textContent, 'None yet');
    } finally {
        controls.destroy();
    }
});

test('Sources arrival and Refresh use one authoritative list for both the rail and page', async (t) => {
    const { env, state, layers, showPage } = world(t);
    const reply = (body, status = 200) => ({ ok: status === 200, status, json: async () => body });
    const reads = [];
    env.routes.set('/api/extensions', reply({ skills: [] }));
    env.routes.set('/api/state', reply({ github_token_configured: false }));
    env.routes.set('/api/skills/lifecycle-queue', reply({ events: [] }));
    env.routes.set('/api/skill-sources', () => new Promise(resolve => reads.push(resolve)));
    const catalog = reply({ items: [], sources: [] });
    env.routes.set('/api/skill-catalog', catalog);
    let finishCatalog;
    env.routes.set('/api/skill-catalog?refresh=1', () => new Promise(resolve => { finishCatalog = resolve; }));
    const controls = initSkills({ state, layers, showPage });
    try {
        env.document.getElementById('skills-installed-search').value = '';
        const page = env.document.getElementById('page-skills');
        const answer = page.querySelector('[data-extensions-open="sources"]').querySelector('[data-extensions-answer]');
        const pane = env.document.getElementById('skills-pane-sources');
        const refresh = env.document.getElementById('skills-refresh');
        const status = () => pane.querySelector('.skills-sources-id')?.textContent || '';
        const settleRead = async (finish, result) => { finish(result); await settle(); };
        const fail = reply({ error: 'fixture unavailable' }, 503);
        await controls.open('sources');
        await settle();
        assert.equal(reads.length, 1, 'the arriving page and place share one request');
        assert.equal(answer.textContent, '');
        assert.doesNotMatch(status(), /0 sources/);
        await settleRead(reads[0], fail);
        assert.equal(answer.textContent, 'Could not read');
        assert.match(status(), /couldn’t be read:/);
        assert.equal(reads.length, 1, 'a first-start check does not retry a failed arrival');

        await showPage('chat');
        await controls.open('sources');
        await settle();
        assert.equal(reads.length, 2, 'the return starts one fresh read');
        const pending = refresh.fire('click');
        await settle();
        assert.equal(reads.length, 3, 'an explicit Refresh supersedes the arrival read');
        await settleRead(reads[2], reply({ sources: [{ id: 'one', kind: 'git', name: 'Fresh list' }] }));
        assert.equal(answer.textContent, '1 connected');
        assert.match(status(), /1 source connected/);
        assert.match(pane.querySelector('[data-source-row="one"]').textContent, /Fresh list/);
        await settleRead(reads[1], fail);
        assert.equal(answer.textContent, '1 connected');
        assert.match(status(), /1 source connected/, 'an obsolete failure cannot repaint the view, even with catalog health pending');
        finishCatalog(catalog);
        await pending;

        const failed = refresh.fire('click');
        await settle();
        assert.equal(reads.length, 4);
        await settleRead(reads[3], fail);
        assert.equal(answer.textContent, 'Last known: 1 connected');
        assert.match(status(), /couldn’t be read:/);
        assert.match(pane.querySelector('[data-source-row="one"]').textContent, /Fresh list/);
        finishCatalog(catalog);
        await failed;
        assert.equal(reads.length, 4);
    } finally {
        controls.destroy();
    }
});

test('the tab shown again and the socket reopened are each read once, for the list of pages and the open view together', async (t) => {
    const { env, state, layers, showPage } = world(t);
    const reply = (body) => ({ ok: true, status: 200, json: async () => body });
    env.routes.set('/api/extensions', reply({ skills: [] }));
    env.routes.set('/api/state', reply({ github_token_configured: false }));
    env.routes.set('/api/skills/lifecycle-queue', reply({ events: [] }));
    env.routes.set('/api/skill-sources', reply({ sources: [] }));
    env.routes.set('/api/skill-catalog', reply({ items: [], sources: [] }));
    // As with the real socket, the section and the views hear the same event object, in registration order.
    const heard = new Map();
    const ws = { ...env.ws, on(type, fn) {
        if (!heard.has(type)) heard.set(type, new Set());
        heard.get(type).add(fn);
        return () => heard.get(type).delete(fn);
    } };
    const emit = (type, event = {}) => [...(heard.get(type) || [])].forEach((fn) => fn(event));
    const controls = initSkills({ state, layers, showPage, ws });
    try {
        const { document } = env;
        document.getElementById('skills-installed-search').value = '';
        document.getElementById('page-skills').querySelector('.section-rail').getClientRects = () => [{}];  // beside the view
        await controls.open('installed');
        const listings = () => env.calls.filter((url) => url === '/api/extensions').length;
        // Past the 300 ms a hint waits for its one read, and that read's answer.
        const quiet = async () => { await settle(); await new Promise((resolve) => setTimeout(resolve, 400)); await settle(); };
        await quiet();
        const arrived = listings();
        assert.ok(arrived >= 1, 'the arriving view read the listing');

        document.hidden = true;
        document.dispatchEvent({ type: 'visibilitychange' });
        await quiet();
        assert.equal(listings(), arrived, 'a hidden tab reads nothing');
        document.hidden = false;
        document.dispatchEvent({ type: 'visibilitychange' });
        await quiet();
        assert.equal(listings(), arrived + 1, 'the return starts one read, not one for the rows and another for the hint');

        emit('open', { previouslyConnected: true });
        await quiet();
        assert.equal(listings(), arrived + 2, 'the reopened socket starts one read');

        emit('extension_health_changed');
        emit('extension_lifecycle', { action: 'enable', skill: 'example' });
        await quiet();
        assert.equal(listings(), arrived + 3, 'hints of one burst share a read');
    } finally {
        env.document.hidden = false;
        controls.destroy();
    }
});

// Browse through the real page: the place's shared reads answered by the test, the board's layout inert (its
// observers and inline styles are absent from this fixture and nothing here measures).
function browsePage(t) {
    const page = world(t);
    class Observer { observe() {} disconnect() {} }
    const absent = Object.entries({ ResizeObserver: Observer, MutationObserver: Observer, CSS: { escape: String } })
        .filter(([name]) => globalThis[name] === undefined);
    for (const [name, value] of absent) globalThis[name] = value;
    const inert = { removeProperty() {}, setProperty() {}, getPropertyValue: () => '' };
    const styled = 'style' in Stub.prototype;
    if (!styled) Object.defineProperty(Stub.prototype, 'style', { configurable: true, get: () => inert });
    t.after(() => {
        for (const [name] of absent) delete globalThis[name];
        if (!styled) delete Stub.prototype.style;
    });
    const { document } = page.env;
    const reply = (body, status = 200) => ({ ok: status === 200, status, json: async () => body });
    page.env.routes.set('/api/state', reply({ github_token_configured: false }));
    page.env.routes.set('/api/skills/lifecycle-queue', reply({ events: [] }));
    const pane = () => document.getElementById('skills-pane-browse');
    const card = (id) => pane().querySelectorAll('.marketplace-card').find((node) => node.dataset.item === id);
    return { ...page, document, reply, card,
        says: (id) => card(id).querySelector('.skills-status-text').textContent,
        // A press as the page hears it: every listener from the pressed node up, the document's last.
        async press(node) {
            for (let at = node; at; at = at.parentElement) await at.fire('click', { target: node, preventDefault() {} });
            await settle();
        },
        // An entry as the host's catalog gives it: its source's id, name, address and trust ride on the entry itself.
        entry: (source, name, extra = {}) => ({ id: `${source.id}/${name}`, name, description: `About ${name}.`, version: '1.0.0',
            source_id: source.id, source_name: source.name, trust: source.trust, repo_url: source.url, ...extra }) };
}

test('Browse says how an entry installs only while its source’s setting holds: a later source list changes no card or open detail, a saved choice re-reads the catalog', async (t) => {
    const { env, state, layers, showPage, document, reply, card, says, press, entry } = browsePage(t);
    // A new install: the hub's settings are suggested, not confirmed. The host installs by them all the same.
    const hub = { id: 'hub', name: 'Example hub', kind: 'hub', part: 'official', url: 'https://example.test/hub', trust: 'trusted', confirmed: false };
    const shared = { id: 'shared', name: 'Example shared', kind: 'git', url: 'https://example.test/shared', trust: 'review', confirmed: true };
    const catalogOf = (...sources) => ({ categories: [], sources: sources.map((source) => ({ ...source, state: 'ready' })),
        items: [entry(sources[0], 'tiles', { official: true }), entry(sources[1], 'labels')] });
    let catalog = catalogOf(hub, shared);
    const sourceReads = [];
    const posts = [];
    env.routes.set('/api/extensions', reply({ skills: [] }));
    env.routes.set('/api/skill-sources', () => new Promise((resolve) => sourceReads.push(resolve)));
    env.routes.set('/api/skill-catalog', async () => reply(catalog));
    env.routes.set('/api/skill-catalog/item/shared/labels', () => new Promise(() => {}));
    env.routes.set('/api/skill-sources/hub', async (_url, init) => { posts.push(JSON.parse(init.body)); return reply({ ok: true }); });
    const controls = initSkills({ state, layers, showPage });
    const settleSources = async (...rows) => { sourceReads.at(-1)(reply({ sources: rows })); await settle(); };
    const firstStart = () => document.body.querySelector('.skills-dialog');
    const reviewOf = (sheet) => sheet.querySelectorAll('.skills-sheet-section')
        .find((section) => section.querySelector('h3')?.textContent === 'Review').querySelector('p');
    try {
        document.getElementById('skills-installed-search').value = '';
        await controls.open('browse');
        await settle();
        await settleSources(hub, shared);
        assert.equal(says('hub/tiles'), 'Installs without review while that is the suggested setting for Example hub',
            'the effective default, labelled as suggested');
        assert.equal(says('shared/labels'), 'Reviewed before it runs while that is the setting for Example shared');
        // Arriving on a new install asks once how the hub installs. The owner chooses to review its skills: the
        // write is followed by one read of the sources and one of the catalog under Browse, whose cards then say so.
        assert.ok(firstStart(), 'the first-start question opened over Browse');
        const chosen = { ...hub, trust: 'review', confirmed: true };
        catalog = catalogOf(chosen, shared);
        const row = firstStart().querySelectorAll('[data-row]').find((node) => node.dataset.row === 'hub');
        await press(row.querySelectorAll('[data-trust]').find((node) => node.dataset.trust === 'review'));
        await press(firstStart().querySelector('[data-first-ok]'));
        assert.deepEqual(posts, [{ trust: 'review', confirmed: true }]);
        assert.equal(sourceReads.length, 2, 'the saved choice read the sources again');
        await settleSources(chosen, shared);
        assert.equal(firstStart(), null);
        assert.equal(says('hub/tiles'), 'Reviewed before it runs while that is the setting for Example hub');

        // The place is shown again: its fresh source list is still on its way when a detail is opened.
        await showPage('chat');
        await controls.open('browse');
        await settle();
        assert.equal(sourceReads.length, 3, 'the return started one fresh read of the sources');
        const tiles = card('hub/tiles');
        await press(card('shared/labels').querySelector('.skills-card-name'));
        const sheet = document.body.querySelector('.skills-detail-sheet');
        const review = reviewOf(sheet);
        const shown = 'Ouroboros reviews it on this computer before it runs while that is the setting for the Example shared source. You can skip the review.';
        assert.equal(review.textContent, shown);
        const install = sheet.querySelector('[data-sheet-action="install"]');
        install.focus();
        // That list answers: Example shared was set to install without review somewhere else, same id and address.
        // An Install pressed now would go without review, since the host reads the source at the press. Nothing on
        // screen says otherwise: each sentence holds only while the setting it names does, and none is repainted
        // from a list that is no fact of the shown entries.
        await settleSources(chosen, { ...shared, trust: 'trusted' });
        assert.equal(card('hub/tiles'), tiles, 'no card was rebuilt');
        assert.equal(says('shared/labels'), 'Reviewed before it runs while that is the setting for Example shared');
        assert.equal(reviewOf(sheet), review);
        assert.equal(review.textContent, shown);
        assert.equal(sheet.querySelector('[data-sheet-action="install"]'), install, 'the open detail keeps its action');
        assert.equal(document.activeElement, install, 'and the keyboard its place');
        // The Official mark of a card is explained from that entry too, under the same condition.
        await press(tiles.querySelector('[data-official-mark]'));
        const popup = document.body.querySelector('.skills-mark-popup');
        assert.equal(popup.querySelector('p').textContent, 'Official — the maintainer of Ouroboros vouches for this exact version. '
            + 'It comes from the Example hub source (example.test/hub). A new install is still reviewed while that is the setting for the Example hub source.');
        await press(popup.querySelector('[data-popup-close]'));
        await press(sheet.querySelector('[data-sheet-close]'));
        assert.deepEqual(posts, [{ trust: 'review', confirmed: true }], 'nothing else was written');
    } finally {
        controls.destroy();
    }
});

test('Compare… promises no review and no absence of one: the replacement installs as its source is set at that moment', async (t) => {
    const { env, state, layers, showPage, document, reply, card, says, press, entry } = browsePage(t);
    const shared = { id: 'shared', name: 'Example shared', kind: 'git', url: 'https://example.test/shared', trust: 'review', confirmed: true };
    // The name is taken by a copy from a source that is not connected now: the listing names no id for it.
    const mine = { name: 'notes', version: '0.9.0', enabled: true, description: 'Mine.', install: { source: { name: 'Elsewhere', kind: 'git' } } };
    let finishListing, finishSources;
    env.routes.set('/api/extensions', () => new Promise((resolve) => { finishListing = resolve; }));
    env.routes.set('/api/skill-sources', () => new Promise((resolve) => { finishSources = resolve; }));
    env.routes.set('/api/skill-catalog', reply({ categories: [], sources: [{ ...shared, state: 'ready' }], items: [entry(shared, 'notes')] }));
    const controls = initSkills({ state, layers, showPage });
    try {
        document.getElementById('skills-installed-search').value = '';
        await controls.open('browse');
        await settle();  // the optional facts first: this fixture draws the installed list once and patches no card
        finishListing(reply({ skills: [mine] }));
        await settle();
        finishSources(reply({ sources: [shared] }));  // then the board is drawn, with the installed listing known
        await settle();
        assert.equal(says('shared/notes'), 'You have another “notes” (from Elsewhere)');
        await press(card('shared/notes').querySelector('[data-browse-action="compare"]'));
        const dialog = document.body.querySelector('.skills-dialog');
        assert.match(dialog.querySelector('h3').textContent, /Two skills are called “notes”/);
        const note = dialog.querySelector('.skills-dialog-note').textContent;
        assert.equal(note, 'Keeping yours changes nothing. Using the other one replaces yours and keeps a backup of it; '
            + 'its source’s setting in Sources decides whether it is reviewed first.');
        assert.doesNotMatch(dialog.textContent, /reviewed before it runs|without review/, 'the entry’s setting as read is no promise of the install');
        await press(dialog.querySelector('[data-dialog-cancel]'));
        assert.equal(document.body.querySelector('.skills-dialog'), null);
    } finally {
        controls.destroy();
    }
});

test('a queue-only install names no origin, and only a queued or running one opens details; a folder with no read manifest states no access', () => {
    const job = (status, extra = {}) => ({ id: 'j', target: 'calendar', kind: 'install', status, source: 'skills', message: 'Installing calendar', error: 'x', ...extra });
    const [pending] = mergeLifecycleEvents([], [job('running')]);
    const [failed] = mergeLifecycleEvents([], [job('failed')]);
    assert.equal(opensDetail(pending), true);
    assert.equal(opensDetail(failed), false, 'a terminal install names no installation to describe');
    assert.equal(opensDetail({ ...pending, lifecycle_status: 'succeeded' }), false, 'only an explicitly queued or running install opens');
    for (const virtual of [pending, failed]) {
        for (const invented of ['source', 'version', 'type', 'enabled', 'review_status', 'review_stale', 'permissions', 'grants']) {
            assert.equal(invented in virtual, false, `a queue record carries no ${invented}`);
        }
    }
    for (const virtual of [pending, failed]) {
        assert.equal(installedSkillModel(virtual).sourceName, '', 'the queue source is the caller, not the package origin');
        assert.doesNotMatch(renderInstalledSkillCard(virtual), /Made here|Built in/);
        assert.equal(matchesInstalledFilter(virtual, 'made'), false);
    }
    assert.match(renderInstalledSkillCard(failed), /<span class="skills-card-name" tabindex="-1">/);
    assert.match(renderInstalledSkillCard(failed), /Find in Browse/);
    const real = { name: 'folder', type: 'instruction', enabled: false, source: 'external', payload_root: 'skills/external/folder',
        review_status: 'pending', permissions: [], grants: {}, load_error: 'manifest parse error' };
    assert.doesNotMatch(renderInstalledSkillCard({ ...real, manifest_file: '' }), />Can use</);
    assert.match(renderInstalledSkillCard({ ...real, manifest_file: 'SKILL.md' }), /Can use[\s\S]*Nothing beyond its own folder/);
    assert.match(renderInstalledSkillCard(real), />Can use</, 'an older host sends no field: its rows read as before');
});

test('Needs you counts an enabled, permitted skill whose declared secret is not set, and nothing else for that secret', () => {
    const secret = (state) => ({ secret_config: { fields: [{ id: 'botToken', key: 'TELEGRAM_BOT_TOKEN', state }] } });
    const withTelegram = (overrides) => rows().map((skill) => (skill.name === 'telegram' ? { ...skill, ...overrides } : skill));
    const base = installedFilterCounts(rows());
    const missing = withTelegram(secret('missing'));
    assert.deepEqual(installedFilterCounts(missing), { ...base, needs: base.needs + 1 });
    assert.equal(matchesInstalledFilter(missing.find((skill) => skill.name === 'telegram'), 'needs'), true);
    for (const state of ['unknown', 'configured']) assert.deepEqual(installedFilterCounts(withTelegram(secret(state))), base, state);
    assert.deepEqual(installedFilterCounts(withTelegram({ ...secret('missing'), enabled: false })), { ...base, off: base.off + 1 },
        'an off skill is under Off; its secret asks nothing');
    const ungranted = { ...secret('missing'), grants: { all_granted: false, missing_keys: ['TELEGRAM_BOT_TOKEN'] } };
    assert.deepEqual(installedFilterCounts(withTelegram(ungranted)), { ...base, needs: base.needs + 1 }, 'Needs access, counted once');
});

test('page-module faults participate in Installed filters and Browse state', () => {
    const telegram = byName('telegram');
    const pageModules = { faults: [{ skill: telegram.name, current: true, reason: 'fixture failed' }], held: [] };
    assert.equal(installedSkillModel(telegram, { pageModules }).needs, true);
    assert.equal(matchesInstalledFilter(telegram, 'needs', installedSkillModel(telegram, { pageModules })), true);
    const counts = installedFilterCounts(rows(), pageModules);
    assert.equal(counts.needs, 5, 'a page-local fault must be reflected in Needs you');
    const browse = browseCardState(
        { id: 'telegram-row', name: telegram.name, source_id: telegram.install?.source?.id, source_name: 'built-in', installed: true },
        { installed: telegram, listed: true, pageModules },
    );
    assert.match(browse.status, /needs you/);
});

test('a held page module after Panic explains reload instead of offering a dead Start it action', () => {
    const telegram = byName('telegram');
    const model = installedSkillModel(telegram, {
        pageModules: { faults: [], held: [{ key: 'module:telegram', skill: telegram.name, reason: 'start' }], stopped: true },
    });
    const html = model.trays.join('');
    assert.equal(model.needs, true);
    assert.match(html, /Reload this page to restore it/);
    assert.doesNotMatch(html, /start_host_module/);
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
    for (const fixed of [{ deployment: { name: 'weather' } }, { trust: { state: 'trusted', administered: true } },
        { trust: { state: 'untrusted', administered: false } }]) {
        const managed = { ...byName('weather'), ...fixed };
        assert.equal(installedSkillModel(managed).update, false);
        assert.deepEqual(updateAllCandidates([managed], SOURCES), [], 'bulk update follows the card’s organization policy');
    }
    const permitted = { ...byName('weather'), trust: { state: 'trusted', administered: false } };
    assert.equal(installedSkillModel(permitted).update, true);
    assert.deepEqual(updateAllCandidates([permitted], SOURCES), [permitted]);
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
    // A hub read from a folder of its repository takes no entries; its pointer entries still go back to their authors.
    const folderHub = { id: 'team-hub', name: 'Team hub', kind: 'hub', url: 'https://github.com/team/tools', path: 'catalog' };
    const fromFolderHub = byName('anime_studio');
    fromFolderHub.install.source = { id: 'team-hub', kind: 'hub', name: 'Team hub' };
    assert.deepEqual(publishDestinations(fromFolderHub, [...withGit, folderHub]).map((choice) => choice.value),
        ['source:mine', 'source:ouroboros-community']);
    fromFolderHub.install.release.repo = 'https://github.com/CrossRead/anime';
    assert.equal(publishDestinations(fromFolderHub, [...withGit, folderHub])[0].value, 'origin:team-hub');
    // A view pinned to a commit never reads a publication: not a destination of its own.
    const pinned = { id: 'pinned', name: 'Pinned', kind: 'git', url: 'https://github.com/alice/skills', commit: 'a'.repeat(40) };
    assert.deepEqual(publishDestinations(byName('minecraft-widget'), [...withGit, pinned]).map((choice) => choice.value),
        ['source:mine', 'source:ouroboros-community']);
    // Neither does a view of a tag named in full; a branch named in full is an ordinary destination.
    const tagged = { ...pinned, id: 'tagged', commit: undefined, ref: 'refs/tags/v1.0.0' };
    const branch = { ...pinned, id: 'branch', commit: undefined, ref: 'refs/heads/main' };
    assert.deepEqual(publishDestinations(byName('minecraft-widget'), [...withGit, tagged, branch]).map((choice) => choice.value),
        ['source:mine', 'source:branch', 'source:ouroboros-community']);
    assert.deepEqual(publishDestinations(byName('telegram'), SOURCES).map((choice) => choice.value), ['source:ouroboros-community']);
    assert.deepEqual(publishDestinations(byName('telegram'), SOURCES.filter((row) => row.kind !== 'hub')), [],
        'nothing to choose: the host publishes to its default');
});

test('a copy whose view is connected again under another id follows the source the host matches now', () => {
    // Team now reads another view of the repository; Shared reads the view the copy came from. The host names
    // Shared in the listing, under the name the copy was installed with.
    const team = { id: 'team', name: 'Team', kind: 'git', url: 'https://github.com/example/skills', ref: 'next', path: 'pack', trust: 'trusted' };
    const shared = { ...team, id: 'shared', name: 'Shared', ref: 'stable', trust: 'review' };
    const alpha = byName('weather');
    alpha.install.source = { id: 'shared', name: 'Team', kind: 'git' };
    assert.deepEqual(publishDestinations(alpha, [team, shared]).map((choice) => choice.value), ['origin:shared', 'source:team']);
    assert.deepEqual(updateAllCandidates([alpha], [team, shared]), [], 'its updates come from Shared, a review row');
    assert.deepEqual(updateAllCandidates([alpha], [team, { ...shared, trust: 'trusted' }]).map((skill) => skill.name), ['weather']);
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

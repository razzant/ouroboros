// Search in Settings (docs/DESIGN.md "Settings"): a literal, case-folded match over the words on screen and the words
// they replaced. Every row is found by its name, by an old name and by its key, in English and as a translated
// install shows it; the tabs Settings had still lead somewhere; and the last line hands the words to Ouroboros.
import assert from 'node:assert/strict';
import test from 'node:test';

import { resultsHtml, searchEntries, searchSettings } from '../modules/settings_home.js';
import { HAND_WRITTEN, HOME_ROWS, ROWS } from '../modules/settings_rows.js';

const ids = (query, ...rest) => searchSettings(query, ...rest).map((entry) => entry.id);

test('every row is found by its name, an old name and its key; every home row by its name and each alias', () => {
    const entries = searchEntries({ forms: [] });
    for (const row of [...ROWS, ...HAND_WRITTEN]) {
        for (const words of [row.name, ...(row.aliases || []), ...(row.keys || [])].filter(Boolean)) {
            assert.ok(ids(words, entries).includes(row.id), `${row.id} by "${words}"`);
            assert.ok(ids(String(words).toUpperCase(), entries).includes(row.id), `${row.id} by "${words}" in capitals`);
        }
        const hit = searchSettings(row.keys?.[0] || row.id, entries).find((entry) => entry.id === row.id);
        assert.deepEqual([hit.to, hit.row], [row.page, row.id], 'a result opens the row\'s page at the row');
    }
    for (const row of HOME_ROWS) {
        for (const words of [row.name, ...row.aliases]) assert.ok(ids(words, entries).includes(row.id), `${row.id} by "${words}"`);
    }
    assert.deepEqual(ids('', entries), [], 'no words, no results');
    assert.deepEqual(ids('zzz-no-setting-has-this', entries), []);
});

test('the tabs Settings had are aliases, and a row is found by its page\'s title', () => {
    const entries = searchEntries({ forms: [] });
    const dead = { Behavior: 'home:permissions', Advanced: 'home:computer', Secrets: 'home:accounts', Providers: 'home:accounts' };
    for (const [tab, home] of Object.entries(dead)) assert.ok(ids(tab, entries).includes(home), tab);
    // A Costs row lives behind the door `dashboard:costs`; its page title is the home row that opens it.
    const costs = searchSettings('costs', entries);
    assert.ok(costs.some((entry) => entry.id === 'TOTAL_BUDGET' && entry.where === 'Costs'));
    assert.ok(costs.some((entry) => entry.id === 'home:costs'));
});

test('a translated install finds a row by the words on its screen, and still by the English', () => {
    const memory = { 'Total budget': 'Общий бюджет', 'Paid work stops when known spend reaches it.': 'Платная работа останавливается на этой сумме.',
        Costs: 'Расходы', 'Skills folder': 'Папка навыков' };
    const translate = (text) => memory[text] ?? text;
    const entries = searchEntries({ forms: [] });
    assert.ok(ids('общий бюджет', entries, translate).includes('TOTAL_BUDGET'), 'by the translated name');
    assert.ok(ids('ОСТАНАВЛИВАЕТСЯ', entries, translate).includes('TOTAL_BUDGET'), 'by the translated meta, case-folded');
    assert.ok(ids('расходы', entries, translate).includes('OUROBOROS_PER_TASK_COST_USD'), 'by the translated page title');
    assert.ok(ids('папка', entries, translate).includes('OUROBOROS_SKILLS_REPO_PATH'));
    assert.ok(ids('total budget', entries, translate).includes('TOTAL_BUDGET'), 'the English words still find it');
    assert.deepEqual(ids('общий бюджет', entries, (text) => text), [], 'without the memory those words are nobody\'s');
});

test('a form a skill registered is found by its title and its skill, and opens that skill\'s page', () => {
    const forms = [{ skill: 'live_conversation', name: 'Calls', sections: [{ section_id: 'config', title: 'Calls (beta)' }] }];
    const entries = searchEntries({ forms });
    const [hit] = searchSettings('calls (beta)', entries);
    assert.deepEqual([hit.id, hit.to, hit.where], ['skill:live_conversation:config', 'skill/live_conversation', 'Calls']);
    assert.ok(ids('live_conversation', entries).includes('skill:live_conversation:config'));
    assert.ok(!ids('calls (beta)', searchEntries({ forms: [] })).length, 'only a registered form is listed');
});

test('the match is literal: no intents and no synonyms stand in for the words', () => {
    const entries = searchEntries({ forms: [] });
    // "spend less" names no row, alias or key, so it finds nothing; the Ask line carries it instead.
    assert.deepEqual(ids('spend less', entries), []);
    assert.deepEqual(ids('quieter', entries), []);
});

test('the last line asks Ouroboros with the words, whatever was found', () => {
    const none = resultsHtml('never email my clients', []);
    assert.match(none, /No setting has these words\./);
    assert.match(none, /data-settings-ask/);
    assert.match(none, /Ask Ouroboros: “never email my clients”/);
    const entries = searchEntries({ forms: [] });
    const some = resultsHtml(' Total budget ', searchSettings('Total budget', entries));
    assert.match(some, /data-settings-result="TOTAL_BUDGET"/);
    assert.match(some, /Ask Ouroboros: “Total budget”/, 'the words are carried trimmed');
    assert.ok(some.lastIndexOf('data-settings-ask') > some.lastIndexOf('data-settings-result'), 'the Ask line is the last line');
    assert.doesNotMatch(some, /No setting has these words/);
});

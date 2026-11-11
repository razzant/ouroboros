import assert from 'node:assert/strict';
import test from 'node:test';
import { buildModelIndex } from '../modules/chat_model_routes.js';
import { createModelList } from '../modules/model_list.js';
import { ElementStub } from './chat_dom_fixture.js';

const route = (model) => ({ model, use_local: false });
const MAIN = route('openai::model-main');
const RECENT = route('openai::model-recent');
const item = (id, provider = 'openai') => ({ id, provider_id: provider,
    value: provider === 'openrouter' ? id : `${provider}::${id}` });

function listMarkup({ mainRow = true, multiple = false } = {}) {
    const index = buildModelIndex({ catalog: { items: [item('model-main'), item('model-recent'),
        ...(multiple ? [item('model-other', 'openrouter')] : [])] } });
    const el = new ElementStub('div');
    const list = createModelList({ el, input: new ElementStub('input'), id: 'models',
        store: { index: () => index, status: () => ({ read: true }) },
        main: () => MAIN, current: () => MAIN, recents: () => [RECENT], mainRow, onRoute() {} });
    list.paint();
    list.destroy();
    return el.innerHTML;
}

function sections(options) {
    const groups = [{ heading: '', rows: [] }];
    for (const match of listMarkup(options).matchAll(/<p class="chat-model-section"[^>]*>(.*?)<\/p>|<div class="chat-model-row[^>]* data-row="([^"]*)" aria-selected="(true|false)">(.*?)<\/div>/g)) {
        if (match[1] !== undefined) groups.push({ heading: match[1], rows: [] });
        else groups.at(-1).rows.push({ key: match[2], selected: match[3] === 'true',
            main: match[4].includes('<span class="chat-model-row-tag">Main</span>') });
    }
    return groups;
}

const main = { key: `route|${MAIN.model}`, selected: true, main: true };
const recent = { key: `route|${RECENT.model}`, selected: false, main: false };

test('one source names its models after Recent and lists Main only at the top', () => {
    assert.deepEqual(sections(), [
        { heading: '', rows: [main] },
        { heading: 'Recent', rows: [recent] },
        { heading: 'OpenAI', rows: [recent] },
    ]);
    assert.match(listMarkup(), /<p class="chat-model-section" role="presentation" data-i18n-skip>OpenAI<\/p>/,
        'the source name keeps the same spelling and translation treatment as its source row');
});

test('without a top Main row its model stays under the source, tagged Main', () => {
    assert.deepEqual(sections({ mainRow: false }), [
        { heading: '', rows: [] },
        { heading: 'Recent', rows: [recent] },
        { heading: 'OpenAI', rows: [main, recent] },
    ]);
});

test('several sources keep the Sources heading and source rows after Recent', () => {
    assert.deepEqual(sections({ multiple: true }), [
        { heading: '', rows: [main] },
        { heading: 'Recent', rows: [recent] },
        { heading: 'Sources', rows: [
            { key: 'source:openrouter', selected: false, main: false },
            { key: 'source:openai', selected: false, main: false },
        ] },
    ]);
});

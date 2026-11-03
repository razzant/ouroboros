// The Skills page's requests and the shapes it reads, held to the host contract:
// every skill-source call sends exactly what `gateway/skill_sources.py` takes, a
// typed refusal reaches the caller with its code, and every fixture the page's
// tests render is a field-for-field instance of the JSDoc twin of
// `ouroboros/gateway/skill_source_contracts.py` (skill_source_contract_types.js).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { skillSourcesApi } from '../modules/api_client.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/skills_v4/${name}`, import.meta.url), 'utf8'));
const TYPES = readFileSync(new URL('../modules/skill_source_contract_types.js', import.meta.url), 'utf8');

/** {name: {required: Set, optional: Set}} from the twin's `@typedef {Object}` blocks. */
function typedefs(text) {
    const out = {};
    for (const block of text.matchAll(/@typedef \{Object\} (\w+)([\s\S]*?)\*\//g)) {
        const shape = { required: new Set(), optional: new Set() };
        for (const prop of block[2].matchAll(/@property \{([^}]*)\} (\w+)/g)) {
            (prop[1].trim().endsWith('=') ? shape.optional : shape.required).add(prop[2]);
        }
        out[block[1]] = shape;
    }
    return out;
}
const SHAPES = typedefs(TYPES);

function assertShape(value, name, where) {
    const shape = SHAPES[name];
    assert.ok(shape, `the twin defines ${name}`);
    const keys = new Set(Object.keys(value));
    for (const key of shape.required) assert.ok(keys.has(key), `${where}: ${name} requires ${key}`);
    for (const key of keys) assert.ok(shape.required.has(key) || shape.optional.has(key), `${where}: ${key} is not a ${name} field`);
}

test('every fixture is an instance of the host contract it stands for', () => {
    const sources = fixture('skill_sources.json');
    assertShape(sources, 'SkillSourcesResponse', 'skill_sources.json');
    sources.sources.forEach((row) => assertShape(row, 'SkillSourceRow', `source ${row.id}`));

    for (const [key, probe] of Object.entries(fixture('skill_source_probe.json'))) assertShape(probe, 'SkillSourceProbeResponse', `probe ${key}`);

    const catalog = fixture('skill_catalog.json');
    assertShape(catalog, 'SkillCatalogResponse', 'skill_catalog.json');
    catalog.items.forEach((item) => assertShape(item, 'SkillCatalogItem', item.id));
    catalog.sources.forEach((row) => assertShape(row, 'SkillSourceRow', `catalog source ${row.id}`));

    const search = fixture('skill_catalog_search.json');
    assertShape(search, 'SkillCatalogSearchResponse', 'skill_catalog_search.json');
    search.items.forEach((item) => assertShape(item, 'SkillCatalogItem', `search ${item.id}`));
    assert.ok(search.items.filter((item) => item.registry_slug).every((item) => item.installed === null),
        'the search never joins registry hits with installs');

    const detail = fixture('skill_catalog_item.json');
    assertShape(detail, 'SkillCatalogItemDetailResponse', 'skill_catalog_item.json');
    assertShape(detail.item, 'SkillCatalogItem', 'detail item');

    for (const [key, answer] of Object.entries(fixture('skill_install.json'))) assertShape(answer, 'SkillCatalogInstallResponse', `install ${key}`);

    const starter = fixture('onboarding_starter_skills.json');
    assertShape(starter, 'StarterSkillsResponse', 'onboarding_starter_skills.json');
    starter.sources.forEach((row) => assertShape(row, 'SkillSourceRow', `starter source ${row.id}`));
});

test('the shape check refuses a field the contract does not have, and a missing required one', () => {
    const row = fixture('skill_sources.json').sources[0];
    assert.throws(() => assertShape({ ...row, managed_by: 'x' }, 'SkillSourceRow', 'row'), /managed_by is not a SkillSourceRow field/);
    const { trust: _trust, ...partial } = row;
    assert.throws(() => assertShape(partial, 'SkillSourceRow', 'row'), /requires trust/);
});

test('digests are 64 lowercase hex, the release identity the host compares', () => {
    const digests = [
        ...fixture('skill_catalog.json').items.map((item) => item.digest),
        ...fixture('extensions_v4.json').skills.flatMap((skill) => [skill.content_hash, skill.install?.release?.digest, skill.install?.update_available?.digest]),
    ].filter(Boolean);
    assert.ok(digests.length > 10);
    for (const digest of digests) assert.match(digest, /^[0-9a-f]{64}$/);
});

function recordingFetch(answer = { ok: true }, status = 200) {
    const calls = [];
    globalThis.fetch = async (url, init = {}) => {
        calls.push({ url, method: init.method || 'GET', body: init.body === undefined ? undefined : JSON.parse(init.body) });
        return { ok: status < 400, status, json: async () => answer };
    };
    return calls;
}

test('each call sends exactly what the host takes', async () => {
    const original = globalThis.fetch;
    try {
        const calls = recordingFetch();
        await skillSourcesApi.list();
        await skillSourcesApi.probe({ url: 'https://github.com/someone/skills' });
        await skillSourcesApi.add({ url: 'https://github.com/someone/skills', name: 'Some skills', trust: 'review' });
        await skillSourcesApi.update('my folder', { trust: 'trusted', confirmed: true });
        await skillSourcesApi.remove('my folder');
        await skillSourcesApi.refresh('ouroboros');
        await skillSourcesApi.catalog();
        await skillSourcesApi.catalog({ refresh: true });
        await skillSourcesApi.search({ q: 'weather', source: 'clawhub' });
        await skillSourcesApi.item('ouroboros-community', 'scholion');
        await skillSourcesApi.install('ouroboros/weather');
        await skillSourcesApi.install('clawhub/steipete/weather', { replace: true });
        await skillSourcesApi.updateSkill('anime_studio', 'merge', 'a'.repeat(64), 'b'.repeat(64));
        await skillSourcesApi.updateSkill('weather', 'replace', 'c'.repeat(64));
        await skillSourcesApi.uninstallSkill('weather');
        await skillSourcesApi.starterSkills();
        assert.deepEqual(calls, [
            { url: '/api/skill-sources', method: 'GET', body: undefined },
            { url: '/api/skill-sources/probe', method: 'POST', body: { url: 'https://github.com/someone/skills' } },
            { url: '/api/skill-sources', method: 'POST', body: { url: 'https://github.com/someone/skills', name: 'Some skills', trust: 'review' } },
            { url: '/api/skill-sources/my%20folder', method: 'POST', body: { trust: 'trusted', confirmed: true } },
            { url: '/api/skill-sources/my%20folder/remove', method: 'POST', body: {} },
            { url: '/api/skill-sources/ouroboros/refresh', method: 'POST', body: {} },
            { url: '/api/skill-catalog', method: 'GET', body: undefined },
            { url: '/api/skill-catalog?refresh=1', method: 'GET', body: undefined },
            { url: '/api/skill-catalog/search?q=weather&source=clawhub', method: 'GET', body: undefined },
            { url: '/api/skill-catalog/item/ouroboros-community/scholion', method: 'GET', body: undefined },
            { url: '/api/skill-catalog/install', method: 'POST', body: { id: 'ouroboros/weather', turn_on_when_ready: true } },
            { url: '/api/skill-catalog/install', method: 'POST', body: { id: 'clawhub/steipete/weather', turn_on_when_ready: true, replace: true } },
            { url: '/api/skills/anime_studio/update', method: 'POST', body: { mode: 'merge', digest: 'a'.repeat(64), local_digest: 'b'.repeat(64) } },
            { url: '/api/skills/weather/update', method: 'POST', body: { mode: 'replace', digest: 'c'.repeat(64) } },
            { url: '/api/skills/weather/uninstall', method: 'POST', body: {} },
            { url: '/api/onboarding/starter-skills', method: 'GET', body: undefined },
        ]);
    } finally {
        globalThis.fetch = original;
    }
});

test('a typed refusal rejects with its code and facts; a probe that cannot read answers in its body', async () => {
    const original = globalThis.fetch;
    try {
        const conflict = fixture('skill_install.json').conflict;
        recordingFetch(conflict, 409);
        const refused = await skillSourcesApi.install('clawhub/steipete/weather').then(() => null, (error) => error);
        assert.equal(refused.message, conflict.error);
        assert.equal(refused.body.code, 'name_taken');
        assert.deepEqual(refused.body.occupant, { name: 'weather', source: 'Ouroboros', locations: ['ouroboroshub'] });
        recordingFetch({ ok: false, code: 'invalid_source', error: 'that name is taken' });
        await assert.rejects(skillSourcesApi.update('ouroboros', { name: 'Ouroboros Community' }), /that name is taken/,
            'an ok:false write is a refusal even with a 2xx status');
        const unreadable = fixture('skill_source_probe.json').unreadable;
        recordingFetch(unreadable);
        assert.deepEqual(await skillSourcesApi.probe({ url: 'https://github.com/someone/missing' }), unreadable);
    } finally {
        globalThis.fetch = original;
    }
});

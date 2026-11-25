// The Skills page's requests, held to the host contract: every skill-source
// call sends exactly what `gateway/skill_sources.py` takes, and a typed refusal
// reaches the caller with its code. The wire shapes are defined in
// `gateway/skill_source_contracts.py`.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { skillSourcesApi } from '../modules/api_client.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/skills_v4/${name}`, import.meta.url), 'utf8'));

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
        await skillSourcesApi.search({ source: 'clawhub', sort: 'updated', cursor: 'next page' });
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
            { url: '/api/skill-catalog/search?q=&source=clawhub&sort=updated&cursor=next+page', method: 'GET', body: undefined },
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

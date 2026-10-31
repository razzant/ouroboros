import test from 'node:test';
import assert from 'node:assert/strict';
import { renderSkillIcon } from '../modules/page_icons.js';

test('named and missing icons keep monochrome house geometry and decorative semantics', () => {
    assert.match(renderSkillIcon('mail'), /currentColor/);
    assert.match(renderSkillIcon({ kind: 'host', name: 'mail' }), /aria-hidden="true"/);
    assert.notEqual(renderSkillIcon('mail'), renderSkillIcon({}));
    assert.equal(renderSkillIcon('missing'), renderSkillIcon({}));
    assert.match(renderSkillIcon({}, { className: '" onload="x', size: 18 }), /class="&quot; onload=&quot;x"/);
    assert.match(renderSkillIcon({ kind: 'svg', svg: '<svg onload="x"/>' }), /<path/);
});

test('host-sanitized SVG paints without a second client sanitizer and keeps requested size', async () => {
    const { normalizeIcon } = await import('../modules/consciousness_api.js');
    const { skillIconHtml } = await import('../modules/skill_presentation.js');
    const svg = '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="9"/></svg>';
    globalThis.DOMPurify = { sanitize() { throw new Error('must not sanitize twice'); } };
    try {
        const icon = normalizeIcon({ kind: 'svg', svg });
        assert.equal(icon.kind, 'svg');
        assert.match(skillIconHtml(icon, 27), /width="27" height="27"/);
        assert.match(skillIconHtml(icon), /<circle cx="12"/);
        assert.doesNotMatch(skillIconHtml(normalizeIcon({}), 21), /<circle/);
        assert.match(skillIconHtml({}, 21), /width="21" height="21"/);
        for (const spec of [{ kind: 'host', name: 'mail' }, { name: 'mail' }]) {
            assert.match(skillIconHtml(spec), /<rect x="2" y="4"/);
            assert.doesNotMatch(skillIconHtml(spec), /M9 3v5/);
        }
        assert.match(skillIconHtml({}), /M9 3v5/);
        assert.equal(skillIconHtml({ svg }), skillIconHtml({}));
    } finally { delete globalThis.DOMPurify; }
});

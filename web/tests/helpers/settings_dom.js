// A DOM and a server for the node tests of Settings: the dashboard fixture's document with an element that parses
// markup as a browser nests it, the home's passive reads answered, and the reads that start something left out.
import { forgetSettingRows } from '../../modules/settings_controls.js';
import { NodeStub, setup } from '../dashboard_dom_fixture.js';

// The fixture's element, reading a self-closed tag (an SVG `<path … />` of a glyph) as childless, so a row drawn
// after a glyph is that glyph's sibling here as it is in a browser.
export class Stub extends NodeStub {
    get innerHTML() { return super.innerHTML; }
    set innerHTML(html) {
        this.textContent = '';
        const stack = [this];
        for (const token of String(html).match(/<[^>]+>|[^<]+/g) || []) {
            if (token.startsWith('</')) { stack.pop(); continue; }
            if (!token.startsWith('<')) { stack.at(-1)._text += token; continue; }
            const tag = token.match(/^<([\w-]+)/)?.[1];
            if (!tag) continue;
            const node = new Stub(tag);
            for (const attr of token.slice(tag.length + 1, -1).matchAll(/([\w-]+)(?:="([^"]*)")?/g)) node.setAttribute(attr[1], attr[2] ?? '');
            stack.at(-1).appendChild(node);
            if (!['input', 'br', 'hr', 'img'].includes(tag) && !token.endsWith('/>')) stack.push(node);
        }
    }
}

export const ANSWERS = {
    '/api/settings': { OUROBOROS_MODEL: 'openrouter::z-ai/glm-5.3', OPENROUTER_API_KEY: 'sk-...', OUROBOROS_SAFETY_OWNER_RULES: '', _meta: {} },
    '/api/mcp/status': { enabled: true, servers: [] },
    '/api/extensions': { skills: [{ name: 'one' }], live: { settings_sections: [] } },
    '/api/places': { places: [], connectors: [] },
    '/api/update/status': { managed: true, check_ok: true, current_version: '7.10.2' },
    '/api/storage': { count: null },
    '/api/desktop/autostart': { state: 'unavailable' },
};
export const STARTS = ['/api/model-catalog', '/api/review-pool', '/api/claudexor/wake', '/api/claudexor/status', '/api/mcp/refresh', '/api/providers/test',
    '/api/update/check', '/api/storage/measure'];
export const settle = async () => { for (let round = 0; round < 12; round += 1) await new Promise((resolve) => setImmediate(resolve)); };

export function world(t) {
    forgetSettingRows();
    const env = setup(t);
    const posts = [];
    for (const [url, body] of Object.entries(ANSWERS)) {
        env.routes.set(url, async (_url, init = {}) => {
            if (init.method && init.method !== 'GET') posts.push(url);
            return { ok: true, status: 200, json: async () => body };
        });
    }
    env.document.createElement = (tag) => new Stub(tag);
    const content = new Stub('div');
    content.id = 'content';
    env.document.body.appendChild(content);
    const state = { activePage: 'chat' };
    const shown = [];
    const showPage = async (name, ...rest) => { shown.push([name, ...rest]); state.activePage = name; return true; };
    return { env, posts, content, state, shown, showPage };
}


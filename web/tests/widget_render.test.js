// The declarative widget renderer (widget_render.js) renders one declaration for the place that passes its
// context: the Widgets page resolves media through the card's own skill routes; a place without a skill route
// (a message card) accepts only inline data. Mounting, state and dispatch stay with widgets.js.
import assert from 'node:assert/strict';
import test from 'node:test';
import { extensionRoutePath, extensionRoutePrefix } from '../modules/api_client.js';
import { componentIdentity, filenameFromWidgetUrl, indexComponentTree, renderComponent } from '../modules/widget_render.js';

const DATA_PNG = 'data:image/png;base64,iVBORw0KGgo=';
const view = (state = {}) => ({ state, status: {}, componentState: {}, formValues: {}, pendingActions: new Set(),
    visibleKeys: new Set() });
const widgetsPage = (skill) => ({ routePath: (route, params) => extensionRoutePath(skill, route, params),
    routePrefix: extensionRoutePrefix(skill) });
const noSkillRoute = { routePath: () => '', routePrefix: '' };
const imageSrc = (html) => html.match(/<img src="([^"]*)"/)?.[1] ?? null;

function withOrigin(fn) {
    const prior = globalThis.window;
    globalThis.window = { location: { origin: 'http://local' } };
    try { return fn(); } finally { globalThis.window = prior; }
}

test('the Widgets page context resolves a media route through the card\'s own skill', () => withOrigin(() => {
    const html = renderComponent(widgetsPage('weather'), { type: 'image', route: 'radar', query: { day: 'mon' }, alt: 'Radar' },
        view(), 'components.0');
    assert.equal(imageSrc(html), '/api/extensions/weather/radar?day=mon');
    assert.match(html, /^<figure data-widget-component="path:components\.0" class="widget-media">/);
}));

test('a URL from data is accepted only under the context\'s own route prefix; a place with no skill route takes none', () => withOrigin(() => {
    const state = { result: { own: '/api/extensions/weather/files/a.png', other: '/api/extensions/mail/files/a.png', inline: DATA_PNG } };
    const image = (ctx, path) => imageSrc(renderComponent(ctx, { type: 'image', path }, view(state), 'components.0'));
    assert.equal(image(widgetsPage('weather'), 'own'), '/api/extensions/weather/files/a.png');
    assert.equal(image(widgetsPage('weather'), 'other'), null, 'another skill\'s route is refused');
    assert.equal(image(widgetsPage('weather'), 'inline'), DATA_PNG);
    assert.equal(image(noSkillRoute, 'own'), null, 'no route prefix: no extension URL at all');
    assert.equal(image(noSkillRoute, 'inline'), DATA_PNG, 'inline data still renders');
    const routed = renderComponent(noSkillRoute, { type: 'image', route: 'radar', label: 'Radar' }, view(), 'components.0');
    assert.equal(imageSrc(routed), null);
    assert.match(routed, /Radar: no safe media source\./);
}));

test('a component\'s own inline src draws where the state is an object; a path into the state still wins', () => withOrigin(() => {
    const image = (spec, state) => imageSrc(renderComponent(noSkillRoute, { type: 'image', ...spec }, view(state), 'components.0'));
    assert.equal(image({ src: DATA_PNG }, { result: {} }), DATA_PNG, 'a card: its data is an object and the image names no path');
    assert.equal(image({ src: DATA_PNG }), DATA_PNG, 'no state yet');
    assert.equal(image({ src: 'shots/dock.png' }, { result: {} }), null, 'a file name is not a source');
    assert.equal(image({ path: 'shot', src: 'shots/dock.png' }, { result: { shot: DATA_PNG } }), DATA_PNG);
    assert.equal(image({ path: 'missing', src: DATA_PNG }, { result: {} }), DATA_PNG, 'a path that reads nothing falls back to src');
    assert.equal(image({ src: 'shots/dock.png' }, { result: DATA_PNG }), DATA_PNG, 'a state that is the media itself');
    assert.equal(image({ src: 'shots/dock.png' }, { result: [DATA_PNG] }), DATA_PNG, 'a one-item list reads as its item, as before');
    const own = renderComponent(widgetsPage('weather'), { type: 'image', src: 'placeholder.png' },
        view({ result: ['/api/extensions/weather/radar.png'] }), 'components.0');
    assert.equal(imageSrc(own), '/api/extensions/weather/radar.png', 'a skill\'s result that is its own route still draws');
    assert.equal(image({ path: 'shot', src: DATA_PNG }, { result: { shot: 'not-media' } }), null, 'a path that reads a value does not fall back');
    const figure = (spec) => renderComponent(noSkillRoute, { type: 'image', src: DATA_PNG, ...spec }, view({ result: {} }), 'components.0');
    assert.match(figure({ alt: 'The dock' }), /alt="The dock"><figcaption>The dock<\/figcaption>/);
    assert.match(figure({}), /alt="image"><\/figure>$/, 'an image with no label or alt text has no caption');
}));

test('every rendered root carries its identity and the view learns which components are visible', () => withOrigin(() => {
    const v = view({ result: { temp: 21 } });
    const html = renderComponent(widgetsPage('weather'), { type: 'group', components: [
        { type: 'metric', id: 'temp', label: 'Temperature', path: 'temp', unit: '°C' },
        { type: 'callout', text: 'Dry all week' },
    ] }, v, 'components.0');
    assert.match(html, /^<section data-widget-component="path:components\.0" class="widget-group widget-group-stack">/);
    assert.match(html, /<div data-widget-component="id:temp" class="widget-metric" data-tone="muted"><span>Temperature<\/span><strong>21 <small>°C<\/small><\/strong><\/div>/);
    assert.match(html, /data-widget-component="path:components\.0\.components\.1"/);
    assert.deepEqual([...v.visibleKeys], ['path:components.0', 'id:temp', 'path:components.0.components.1']);
    assert.equal(componentIdentity({ id: 'temp' }, 'components.3'), 'id:temp');
    const { entries } = indexComponentTree([{ type: 'group', components: [{ type: 'callout' }] }]);
    assert.deepEqual(entries.map((entry) => entry.key), ['path:components.0', 'path:components.0.components.0']);
    assert.throws(() => indexComponentTree([{ id: 'x' }, { id: 'x' }]), /duplicate declarative widget component identity/);
    assert.equal(filenameFromWidgetUrl('/api/extensions/weather/download?filename=report.pdf'), 'report.pdf');
}));

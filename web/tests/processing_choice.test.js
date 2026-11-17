// A processing tier beside its model (processing_choice.js; DESIGN "Composer owner controls", "A work's model on its
// card"): the stored words (`economy`, `standard`, `fast`; '' is Default) in the owner's (Flex, Standard, Fast), what
// the catalog says each route serves, which tier a scope takes without a choice of its own, what a choice is and what
// a call was served as, and the strip at the foot of the Model sheet. The saves are composer_model.test.js and
// card_model.test.js; the strip's markup is built into nodes here (dom_double.js `parseHtml`).
import assert from 'node:assert/strict';
import test from 'node:test';
import { buildModelIndex } from '../modules/chat_model_routes.js';
import {
    TIERS, createTierStrip, offersUltrafast, receiptWords, resolvedTier, servedWords, tierExplanation, tierNamed, tierSupport,
    tierWord,
} from '../modules/processing_choice.js';
import { domDouble } from './dom_double.js';

const route = (model) => ({ model, use_local: false });
const ASTRA = route('claudexor::codex=gpt-6-astra');
const SOL = route('claudexor::codex=gpt-6.1-sol');
const OLD = route('claudexor::codex=gpt-5-old');
const GLM = route('z-ai/glm-5.3');
const codex = (id, modes) => ({ provider_id: 'claudexor', provider: 'Codex', source_id: 'codex', id, name: id,
    value: `claudexor::codex=${id}`, credential_profile_id: 'home', ...(modes ? { processing: { modes } } : {}) });

test('the tiers in the strip\'s order and in the owner\'s words; every tier but Standard is named beside a model', () => {
    assert.deepEqual(TIERS, ['economy', 'standard', 'fast', 'ultrafast'], 'cheaper and slower first');
    assert.ok(Object.isFrozen(TIERS));
    assert.deepEqual(['economy', 'standard', 'fast', 'ultrafast', '', 'priority', undefined].map(tierWord),
        ['Flex', 'Standard', 'Fast', 'Ultrafast', 'Default', 'Default', 'Default'], 'anything unknown reads as Default');
    assert.deepEqual(['economy', 'standard', 'fast', 'ultrafast', '', 'priority'].map(tierNamed), [true, false, true, true, false, false]);
});

test('support is the catalog\'s word: listed, unlisted or unknown, and Standard is every route\'s', () => {
    const catalog = (processingPreferences) => ({
        model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex', ...(processingPreferences ? { processingPreferences } : {}) }],
        items: [codex('gpt-6-astra', ['standard', 'fast', 'ultrafast']), codex('gpt-6.1-sol', ['standard']), codex('gpt-5-old'),
            { provider_id: 'openrouter', id: GLM.model, name: GLM.model, value: GLM.model }],
    });
    // In the strip's order: Flex, Standard, Fast, Ultrafast.
    const advertised = buildModelIndex({ catalog: catalog(['standard', 'fast', 'ultrafast']) });
    const support = (index, seen) => TIERS.map((tier) => tierSupport(index, seen, tier));
    assert.deepEqual(support(advertised, ASTRA), ['unlisted', 'listed', 'listed', 'listed'], 'its transport and its accounts both list Fast and Ultrafast');
    assert.deepEqual(support(advertised, SOL), ['unlisted', 'listed', 'unlisted', 'unlisted'], 'its accounts do not list them for it');
    assert.deepEqual(support(advertised, OLD), ['unlisted', 'listed', 'unknown', 'unknown'], 'no account says: unconfirmed');
    // A transport that does not advertise a tier leaves it out, whatever an account lists.
    const narrow = buildModelIndex({ catalog: catalog(['standard', 'fast']) });
    assert.deepEqual(support(narrow, ASTRA), ['unlisted', 'listed', 'listed', 'unlisted']);
    // No transport list: the accounts' list decides, and nothing says nothing. Ultrafast alone needs the
    // transport's own word too (13.11): an account's list without it leaves Ultrafast unconfirmed.
    const unsaid = buildModelIndex({ catalog: catalog(null) });
    assert.deepEqual(support(unsaid, ASTRA), ['unlisted', 'listed', 'listed', 'unknown']);
    assert.deepEqual(support(unsaid, OLD), ['unknown', 'listed', 'unknown', 'unknown']);
    assert.deepEqual(support(unsaid, GLM), ['unknown', 'listed', 'unknown', 'unknown'], 'a key\'s source says nothing of tiers');
    assert.deepEqual(support(unsaid, route('vendor/unlisted')), ['unknown', 'listed', 'unknown', 'unknown'], 'a route the catalog does not list');
    assert.equal(tierSupport(advertised, null, 'fast'), 'listed', 'no route: nothing to warn about');
    assert.equal(tierSupport(advertised, SOL, ''), 'listed', 'Default is no tier of its own');
    assert.equal(tierSupport(null, SOL, 'fast'), 'unknown', 'no catalog read yet');
});

for (const [source, label] of [['claude', 'Claude Code'], ['cursor', 'Cursor']]) {
    test(`a source that accepts Ultrafast with an ordinary fallback (${label}) does not offer it without its model's word`, () => {
        // Claudexor declares the tiers a harness ACCEPTS, Ultrafast included for Claude and Cursor; acceptance is not
        // service: such a call may run as Standard. The strip offers Ultrafast only where the route's model says it too.
        const item = (id, modes) => ({ provider_id: 'claudexor', provider: label, source_id: source, id, name: id,
            value: `claudexor::${source}=${id}`, credential_profile_id: 'home', ...(modes ? { processing: { modes } } : {}) });
        const index = buildModelIndex({ catalog: { model_sources: [{ id: source, label, credentialHarness: source,
            processingPreferences: ['standard', 'fast', 'economy', 'ultrafast'] }],
            items: [item('model-a'), item('model-b', ['standard', 'fast']), item('model-c', ['standard', 'fast', 'ultrafast'])] } });
        const support = (id) => tierSupport(index, route(`claudexor::${source}=${id}`), 'ultrafast');
        assert.deepEqual(['model-a', 'model-b', 'model-c'].map(support), ['unknown', 'unlisted', 'listed']);
        assert.deepEqual(['model-a', 'model-b'].map((id) => offersUltrafast({ support: support(id) })), [false, false],
            'the declaration alone advertises no Ultrafast');
        assert.equal(offersUltrafast({ support: support('model-c') }), true, 'transport support and the model\'s service evidence');
        assert.equal(offersUltrafast({ support: support('model-c'), main: true }), false, 'never for Main');
    });
}

test('a scope without a tier of its own takes Main\'s while it follows Main, else the default one; Standard is a choice', () => {
    const tiers = { main: 'fast', default: 'economy' };
    assert.equal(resolvedTier({ choice: 'standard', follows: true, tiers }), 'standard', 'an explicit Standard is never Default');
    assert.equal(resolvedTier({ choice: '', follows: true, tiers }), 'fast');
    assert.equal(resolvedTier({ choice: '', follows: false, tiers }), 'economy');
    assert.equal(resolvedTier({ follows: true, tiers: { main: '', default: 'economy' } }), 'economy', 'Main without its own tier takes the default');
    assert.equal(resolvedTier({ follows: true, tiers: null }), '', 'nothing read: nothing claimed');
    assert.equal(resolvedTier(), '');
});

test('before a press the strip says what a tier means here, and that an unlisted or unconfirmed one may go as Standard', () => {
    assert.equal(tierExplanation('standard', { support: 'unlisted', route: ASTRA }), 'The ordinary service at the catalog price.');
    assert.equal(tierExplanation('', { route: ASTRA }), 'The ordinary service at the catalog price.', 'Default is said as Standard\'s service');
    assert.equal(tierExplanation('fast', { support: 'listed', route: ASTRA }), 'Faster replies at a higher price.');
    assert.equal(tierExplanation('economy', { support: 'unlisted', route: ASTRA, sourceLabel: 'Codex' }),
        'Lower price, slower; a provider may be out of capacity for it. Codex does not list Flex for gpt-6-astra: its calls go as Standard.');
    assert.equal(tierExplanation('fast', { support: 'unlisted', route: SOL }),
        'Faster replies at a higher price. This source does not list Fast for gpt-6.1-sol: its calls go as Standard.');
    assert.equal(tierExplanation('fast', { support: 'unknown', route: OLD }),
        'Faster replies at a higher price. Not confirmed for gpt-5-old here: a provider may serve it as Standard.');
    assert.equal(tierExplanation('ultrafast', { support: 'listed', route: ASTRA }), 'The fastest replies at the highest price, where a source offers it.');
    assert.equal(tierExplanation('ultrafast', { support: 'unlisted', route: SOL, sourceLabel: 'Codex' }),
        'The fastest replies at the highest price, where a source offers it. Codex does not list Ultrafast for gpt-6.1-sol: its calls go as Standard.');
});

test('the card says what the newest call was served as only where it differs from what was asked', () => {
    assert.equal(servedWords(null), '');
    assert.equal(servedWords('fast'), '');
    assert.equal(servedWords({}), '', 'nothing asked, nothing seen');
    assert.equal(servedWords({ requested: 'fast', submitted: 'fast', observed: 'fast' }), '', 'served as asked');
    assert.equal(servedWords({ requested: 'fast', submitted: 'fast', observed: 'standard' }), 'Fast asked · served Standard');
    assert.equal(servedWords({ requested: 'economy', submitted: 'economy', observed: 'standard' }), 'Flex asked · served Standard');
    assert.equal(servedWords({ requested: 'ultrafast', submitted: 'ultrafast', observed: 'fast' }), 'Ultrafast asked · served Fast');
    assert.equal(servedWords({ requested: 'fast', submitted: null, observed: 'unknown', reason: 'processing_not_submitted' }),
        'Fast asked · sent as Standard', 'a route that cannot carry the tier sent none');
    assert.equal(servedWords({ requested: 'fast', submitted: 'fast', observed: 'unknown' }), 'Fast asked · not confirmed');
    assert.equal(servedWords({ requested: 'standard', submitted: 'standard', observed: 'unknown' }), '', 'an unreported Standard is no news');
    assert.equal(servedWords({ requested: 'standard', submitted: null, observed: 'unknown', reason: 'processing_not_submitted' }), '',
        'an explicit Standard sent with no tier is the ordinary service');
    // A Fast or Flex the engine refused before generating, which the host sent again as Standard (owner 13.11): its
    // receipt keeps what the call asked (llm_claudexor.py), so the deviation is said.
    for (const [tier, word] of [['fast', 'Fast'], ['economy', 'Flex']]) {
        const retried = { requested: tier, submitted: 'standard', submittedNative: 'default', observed: 'standard',
            observedNative: ['default'], reason: 'submitted_mode_differs' };
        assert.equal(servedWords(retried), `${word} asked · served Standard`);
        assert.equal(receiptWords(retried), `asked ${word} · sent Standard · served Standard`);
        // The same continuation where the engine sent no receipt of its own: what was sent is still known.
        const unreported = { requested: tier, submitted: 'standard', observed: 'unknown', reason: 'submitted_mode_differs' };
        assert.equal(servedWords(unreported), `${word} asked · sent as Standard`);
    }
});

test('the receipt in full says what was asked, sent and served, and a provider\'s own word that is not the tier\'s', () => {
    assert.equal(receiptWords(undefined), '');
    assert.equal(receiptWords({ requested: 'fast', submitted: 'fast', observed: 'fast', observedNative: ['priority'] }),
        'asked Fast · sent Fast · served Fast', 'priority is Fast in the provider\'s words: nothing more to say');
    assert.equal(receiptWords({ requested: 'fast', submitted: 'fast', observed: 'fast', observedNative: ['ultrafast'] }),
        'asked Fast · sent Fast · served Fast (ultrafast)', 'an older Codex says Ultrafast where it served Fast');
    assert.equal(receiptWords({ requested: 'ultrafast', submitted: 'ultrafast', observed: 'ultrafast', observedNative: ['ultrafast'] }),
        'asked Ultrafast · sent Ultrafast · served Ultrafast');
    assert.equal(receiptWords({ requested: 'economy', submitted: 'economy', observed: 'standard', observedNative: ['default', 7, 'flex'] }),
        'asked Flex · sent Flex · served Standard (default, flex)', 'a word that does not fold into what was seen is said with the others');
    assert.equal(receiptWords({ requested: 'fast', submitted: null, observed: 'unknown', reason: 'processing_not_submitted' }),
        'asked Fast · sent with no tier · served as not reported');
    assert.equal(receiptWords({ submitted: 'standard', observed: 'standard' }), 'sent Standard · served Standard', 'nothing asked: no asked part');
});

/* The strip on a page, painted as its sheet paints it, with every choice it hands back. */
const MEANS = { economy: 'Lower price, slower; a provider may be out of capacity for it.',
    standard: 'The ordinary service at the catalog price.', fast: 'Faster replies at a higher price.' };
function mountStrip(view = {}) {
    const { doc } = domDouble({ parseHtml: true });
    const chosen = [];
    const strip = createTierStrip({ doc, id: 'sheet-1-tier', onChoose: (tier) => chosen.push(tier) });
    doc.body.append(strip.el);
    strip.paint({ choice: '', resolved: '', scope: 'Default · Standard · for new work here',
        explain: (tier) => tierExplanation(tier, { support: 'listed' }), ...view });
    const seg = (tier) => strip.segs.find((made) => made.dataset.tier === tier);
    return {
        doc, strip, chosen, seg,
        checked: () => strip.segs.filter((made) => made.getAttribute('aria-checked') === 'true').map((made) => made.dataset.tier),
        ring: () => strip.segs.filter((made) => made.dataset.resolved === 'true').map((made) => made.dataset.tier),
        stop: () => strip.segs.filter((made) => made.tabIndex === 0).map((made) => made.dataset.tier),
        scope: () => strip.el.querySelector('.model-tier-scope').textContent,
        means: () => strip.el.querySelector('.model-tier-means').textContent,
    };
}

test('the strip is one track of the tiers and a Default beside it, with its scope and its meaning under them', () => {
    const s = mountStrip();
    const track = s.strip.el.querySelector('.model-tier-track');
    assert.ok(track.classList.contains('ui-segment-group'), 'the shared segment strip');
    assert.deepEqual([track.getAttribute('role'), track.getAttribute('aria-label'), track.getAttribute('aria-describedby')],
        ['radiogroup', 'Processing tier', 'sheet-1-tier-scope sheet-1-tier-means']);
    assert.deepEqual(s.strip.segs.map((made) => [made.dataset.tier, made.textContent, made.getAttribute('role'), made.className]), [
        ['economy', 'Flex', 'radio', 'ui-segment model-tier-seg'], ['standard', 'Standard', 'radio', 'ui-segment model-tier-seg'],
        ['fast', 'Fast', 'radio', 'ui-segment model-tier-seg'], ['ultrafast', 'Ultrafast', 'radio', 'ui-segment model-tier-seg']]);
    assert.deepEqual([s.strip.reset.dataset.tier, s.strip.reset.textContent, s.strip.reset.hidden], ['', 'Default', true],
        'Default stands only while there is a choice to return from');
    assert.equal(s.strip.el.querySelector('.model-tier-means').getAttribute('aria-live'), 'polite');
    assert.deepEqual([s.scope(), s.means()], ['Default · Standard · for new work here', MEANS.standard],
        'without a choice and a tier read, the tier taken is said as Standard\'s service');
});

test('a choice is filled; without one the tier the scope takes wears the ring; an explicit Standard is a choice too', () => {
    const s = mountStrip({ resolved: 'fast', supportOf: (tier) => (tier === 'economy' ? 'unlisted' : 'listed') });
    assert.deepEqual([s.checked(), s.ring(), s.stop(), s.strip.reset.hidden, s.means()], [[], ['fast'], ['fast'], true, MEANS.fast]);
    assert.deepEqual(s.strip.segs.map((made) => made.dataset.support), ['unlisted', 'listed', 'listed', 'listed']);
    s.strip.paint({ choice: 'economy', resolved: 'economy', scope: 'Flex · for new work here' });
    assert.deepEqual([s.checked(), s.ring(), s.stop(), s.strip.reset.hidden, s.scope(), s.means()],
        [['economy'], [], ['economy'], false, 'Flex · for new work here', MEANS.economy]);
    s.strip.paint({ choice: 'standard', resolved: 'standard' });
    assert.deepEqual([s.checked(), s.ring(), s.strip.reset.hidden], [['standard'], [], false], 'Standard chosen is not Default');
    s.strip.paint({ choice: '', resolved: '' });
    assert.deepEqual([s.checked(), s.ring(), s.stop(), s.strip.reset.hidden], [[], ['standard'], ['standard'], true]);
});

test('a press chooses at once, Default included; the chosen tier and a busy strip choose nothing', () => {
    const s = mountStrip({ choice: 'economy', resolved: 'economy' });
    s.seg('fast').click();
    s.seg('economy').click();
    s.strip.reset.click();
    assert.deepEqual(s.chosen, ['fast', ''], 'the chosen tier again is no act');
    s.strip.paint({ busy: true });
    // A save in flight is said (aria-disabled) and its presses do nothing; a disabled button would drop the focus.
    assert.ok(s.strip.segs.every((made) => made.getAttribute('aria-disabled') === 'true' && !made.disabled)
        && s.strip.reset.getAttribute('aria-disabled') === 'true', 'a save in flight takes no press and keeps the focus');
    s.strip.el.emit('click', { target: s.seg('standard') });
    assert.deepEqual(s.chosen, ['fast', ''], 'even a press that reaches the strip');
    s.strip.paint({ busy: false });
    assert.equal(s.seg('standard').getAttribute('aria-disabled'), 'false');
    s.seg('standard').click();
    assert.deepEqual(s.chosen, ['fast', '', 'standard']);
});

test('the pointer or the focus on a tier says what it means before a press; arrows move among tiers without choosing', () => {
    const s = mountStrip({ resolved: 'standard', ultrafast: true });
    s.seg('fast').emit('pointerover');
    assert.equal(s.means(), MEANS.fast);
    s.strip.el.emit('pointerleave');
    assert.equal(s.means(), MEANS.standard, 'the tier taken again');
    s.seg('economy').emit('focusin');
    assert.equal(s.means(), MEANS.economy);
    s.seg('economy').emit('focusout');
    assert.equal(s.means(), MEANS.standard);
    const right = s.seg('standard').emit('keydown', { key: 'ArrowRight' });
    assert.deepEqual([right.defaultPrevented, s.doc.activeElement, s.stop(), s.chosen], [true, s.seg('fast'), ['fast'], []]);
    s.seg('fast').emit('keydown', { key: 'ArrowDown' });
    assert.equal(s.doc.activeElement, s.seg('ultrafast'));
    s.seg('ultrafast').emit('keydown', { key: 'ArrowRight' });
    assert.equal(s.doc.activeElement, s.seg('ultrafast'), 'the last tier is the last');
    s.seg('ultrafast').emit('keydown', { key: 'Home' });
    assert.deepEqual([s.doc.activeElement, s.stop()], [s.seg('economy'), ['economy']]);
    s.seg('economy').emit('keydown', { key: 'ArrowLeft' });
    assert.equal(s.doc.activeElement, s.seg('economy'), 'the first is the first');
    s.seg('economy').emit('keydown', { key: 'End' });
    assert.equal(s.doc.activeElement, s.seg('ultrafast'));
    s.seg('ultrafast').emit('keydown', { key: 'ArrowUp' });
    assert.equal(s.doc.activeElement, s.seg('fast'));
    assert.equal(s.seg('fast').emit('keydown', { key: 'Enter' }).defaultPrevented, undefined, 'Enter is the button\'s own press');
    assert.deepEqual(s.chosen, [], 'moving chose nothing');
    s.strip.destroy();
    assert.equal(s.strip.el.listenerCount(), 0, 'destroy takes its listeners');
});

test('a finger previews nothing, its tap chooses; the meaning\'s box is as tall as its tallest explanation', () => {
    const s = mountStrip({ resolved: 'standard', ultrafast: false });
    s.seg('fast').emit('pointerover', { pointerType: 'touch' });
    assert.equal(s.means(), MEANS.standard, 'a finger on a tier changes no line under it before its tap lands');
    s.seg('fast').emit('pointerover', { pointerType: 'mouse' });
    assert.equal(s.means(), MEANS.fast, 'the mouse previews');
    s.strip.el.emit('pointerleave', { pointerType: 'mouse' });
    // Every explanation a shown tier may say, each in a hidden copy in the box's one cell: a preview never grows it.
    const box = s.strip.el.querySelector('.model-tier-meansbox');
    const sizer = box.querySelector('.model-tier-sizer');
    assert.deepEqual([box.children.map((node) => node.className), sizer.getAttribute('aria-hidden')],
        [['model-tier-means', 'model-tier-sizer'], 'true']);
    assert.deepEqual(sizer.children.map((line) => line.textContent), [MEANS.economy, MEANS.standard, MEANS.fast]);
    const kept = sizer.children[0];
    s.strip.paint({ scope: 'Fast · for new work here' });
    assert.ok(sizer.children[0] === kept, 'the same explanations are not written again');
    s.strip.paint({ ultrafast: true });
    assert.equal(sizer.children.length, 4, 'an offered Ultrafast has its copy too');
});

test('Default pressed by the keyboard leaves with its choice and hands the focus to the tier the scope takes', () => {
    const s = mountStrip({ choice: 'fast', resolved: 'fast' });
    s.strip.reset.focus();
    s.strip.reset.click();
    assert.deepEqual(s.chosen, ['']);
    s.strip.paint({ choice: '', resolved: 'standard' });  // its save answered
    assert.deepEqual([s.strip.reset.hidden, s.doc.activeElement], [true, s.seg('standard')], 'never lost to the page');
});

test('Ultrafast is offered only where its route lists it or it is already the choice, and never for Main', () => {
    assert.equal(offersUltrafast({ support: 'listed' }), true);
    assert.equal(offersUltrafast({ support: 'unknown' }), false, 'a native mode id alone confirms nothing');
    assert.equal(offersUltrafast({ support: 'unlisted', choice: 'ultrafast' }), true, 'a chosen Ultrafast stays to be seen and left');
    assert.equal(offersUltrafast({ main: true, support: 'listed', choice: 'ultrafast' }), false, 'Main\'s tier is a Settings value');
    const hidden = mountStrip({ ultrafast: false });
    assert.equal(hidden.seg('ultrafast').hidden, true);
    assert.deepEqual(TIERS.filter((tier) => !hidden.seg(tier).hidden), ['economy', 'standard', 'fast']);
    const offered = mountStrip({ ultrafast: true, choice: 'ultrafast' });
    assert.equal(offered.seg('ultrafast').hidden, false);
    assert.deepEqual(offered.checked(), ['ultrafast']);
});

test('a hidden Ultrafast is never a keyboard stop: End and arrows stop at Fast', () => {
    const s = mountStrip({ resolved: 'standard', ultrafast: false });
    s.seg('standard').emit('keydown', { key: 'End' });
    assert.equal(s.doc.activeElement, s.seg('fast'));
    s.seg('fast').emit('keydown', { key: 'ArrowRight' });
    assert.equal(s.doc.activeElement, s.seg('fast'), 'Fast is the last shown tier');
});

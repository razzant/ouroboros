import './fixtures/call_modules.js';
import test from 'node:test';
import assert from 'node:assert/strict';
const { callSettingsHtml, scheduleSave, autosaves } = await import('../../skills/live_conversation/ui/call_settings.js');

const catalog = { id:'fixture:voice', label:'Fixture engine', privacy:'Synthetic audio stays here.', models:[
    {id:'first', label:'First model', default:true, vision:false, voices:[{id:'quiet',label:'Quiet'}], honours:['pause'],
        notes:[['thinking','This model has no thinking setting.'],['search','This model cannot search.']],
        facts:{conversation_window_tokens:12345,source:'Measured synthetic route.'}},
    {id:'second',label:'Second model',voices:[{id:'bright',label:'Bright'}],honours:['thinking','search','interruptions'],
        thinking_levels:['small','large'],facts:{conversation_window_tokens:23456,source:'Not yet measured.'}},
] };

test('settings choices, hidden knobs and notes come from the selected catalog', () => {
    const config = { provider:catalog.id, catalogs:[catalog], enabled:true, model:'', voice:'quiet' };
    const first = callSettingsHtml(config);
    assert.match(first, /Fixture engine/);
    assert.match(first, /First model/);
    assert.match(first, /Quiet/);
    assert.match(first, /name="pause"/);
    assert.doesNotMatch(first, /name="thinking"|name="search"/);
    assert.match(first, /This model has no thinking setting/);
    assert.match(first, /Synthetic audio stays here/);
    assert.match(first, /12,345/);
    const second = callSettingsHtml({...config,model:'second'});
    assert.match(second, /name="thinking"/);
    assert.match(second, /value="large"/);
    assert.match(second, /Bright/);
    assert.doesNotMatch(second, /name="pause"/);
    assert.match(second, /Not yet measured/);
    assert.doesNotMatch(second, /Gemini|Grok|OpenAI/);
});

test('an unavailable selected engine is named rather than silently replaced', () => {
    const html = callSettingsHtml({provider:'missing:voice',catalogs:[catalog],model:'lost-model',voice:'lost-voice'});
    assert.match(html, /missing:voice/);
    assert.match(html, /not available/);
    assert.match(html, /value="missing:voice" selected/);
});

test('the Gemini engine choice carries its regional availability note only for Gemini', () => {
    const note = /Google does not offer Gemini Live in Russia or Belarus\./;
    const gemini = { ...catalog, id:'gemini_live:gemini', label:'Gemini' };
    assert.match(callSettingsHtml({provider:gemini.id, catalogs:[gemini,catalog]}), note);
    assert.doesNotMatch(callSettingsHtml({provider:catalog.id, catalogs:[gemini,catalog]}), note);
});

test('memory disclosures keep the catalog window unit and measurement source', () => {
    const render = basis => callSettingsHtml({provider:catalog.id, catalogs:[{...catalog, models:[{
        ...catalog.models[0], facts:{conversation_window_tokens:80000, basis, source:'Measured route.'},
    }]}]});
    assert.match(render('estimator'), /80,000 tokens by Ouroboros(?:’|&#39;)s own estimate per session, then it renews\./);
    assert.match(render('estimator'), /Measured route\./);
    assert.doesNotMatch(render('estimator'), /provider tokens/);
    assert.match(render('provider'), /80,000 provider tokens per session, then it renews\./);
    assert.doesNotMatch(render('provider'), /own estimate/);
    assert.match(callSettingsHtml({provider:'missing',catalogs:[]}), /memory window for this selection is not known/);
});

test('thinking labels its default level honestly and keeps explicit levels and other defaults', () => {
    const thinking = {...catalog.models[1], default:true, notes:[['thinking', 'Fixture thinking note.']]};
    const config = {provider:catalog.id, catalogs:[{...catalog,models:[thinking]}]};
    assert.match(callSettingsHtml({...config, thinking:'obsolete'}), /value="" selected>Default \(small\)<\/option>/);
    const named = {provider:catalog.id, catalogs:[{...catalog,models:[{...thinking, thinking_default:'large'}]}]};
    assert.match(callSettingsHtml({...named, thinking:''}), /value="" selected>Default \(large\)<\/option>/);
    const blank = callSettingsHtml({...config, thinking:''});
    assert.match(blank, /value="" selected>Default \(small\)<\/option>/);
    assert.doesNotMatch(blank, />Engine’s default<\/option>/);
    assert.match(blank, /Engine’s default model/);
    assert.match(blank, /Engine’s default voice/);
    assert.match(blank, /data-call-knob-note="thinking">Fixture thinking note\./);
    assert.match(callSettingsHtml({...config, thinking:'large'}), /value="large" selected/);
    assert.doesNotMatch(callSettingsHtml({provider:catalog.id,catalogs:[catalog]}), /name="thinking"/);
});

test('an unavailable engine offers status guidance without making manual grants the setup flow', () => {
    const available = callSettingsHtml({provider:catalog.id, catalogs:[catalog]});
    const missing = callSettingsHtml({provider:'missing:voice', catalogs:[catalog]});
    assert.match(missing, /Choose another engine or check its status in Skills\./);
    assert.doesNotMatch(missing, /Enable an engine skill|grant its credential/);
    assert.doesNotMatch(available, /check its status in Skills/);
    assert.match(available, /ordinary memory and tools/);
});

test('a call setting saves itself through the host Save, once per burst and after a Save in flight', () => {
    const timers = new Map(); let next = 0;
    const env = { setTimeout: (fn) => { timers.set(++next, fn); return next; }, clearTimeout: id => timers.delete(id) };
    const run = () => { const pending = [...timers.values()]; timers.clear(); pending.forEach(fn => fn()); };
    const save = { disabled: false }, submitted = [];
    const form = { requestSubmit: button => submitted.push(button) };
    const autosave = scheduleSave(form, save, env);
    autosave.change(); autosave.change(); autosave.change();
    assert.equal(timers.size, 1);
    run();
    assert.deepEqual(submitted, [save]);
    save.disabled = true;              // the host is still saving
    autosave.change(); run();
    assert.equal(submitted.length, 1);
    save.disabled = false; run();
    assert.equal(submitted.length, 2);
    autosave.change(); autosave.cancel(); run();
    assert.equal(submitted.length, 2);
});

test('only the declared call fields save themselves; device choices stay local', () => {
    for (const name of ['enabled', 'provider', 'model', 'voice', 'instructions', 'pause', 'interruptions', 'thinking', 'search']) {
        assert.equal(autosaves(name), true, name);
    }
    assert.equal(autosaves(''), false);
    assert.equal(autosaves('microphone'), false);
});

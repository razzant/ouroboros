import test from 'node:test';
import assert from 'node:assert/strict';
import { matchingModelOptions, modelChooserHtml, modelOptions } from '../modules/model_chooser.js';

test('model suggestions filter values and labels without a result-count ceiling', () => {
    const items = Array.from({ length: 40 }, (_, i) => ({ value: `provider/model-${i}`, label: `Reasoner ${i}` }));
    assert.equal(matchingModelOptions(items, 'provider/').length, 40);
    assert.equal(matchingModelOptions(items, 'reasoner 12')[0].value, 'provider/model-12');
    assert.deepEqual(matchingModelOptions(items, 'new-owner-model'), []);
    assert.equal(items.length, 40);
});

test('model choice markup escapes catalog text and retains arbitrary current IDs', () => {
    const html = modelChooserHtml('aria-label="Main model"', 'owner/"<new>', 'main-options', [
        { value: '"<script>', label: '<img src=x>' },
    ]);
    assert.match(html, /value="owner\/&quot;&lt;new&gt;"/);
    assert.doesNotMatch(html, /<script>|<img/);
    assert.match(html, /role="combobox"/);
    assert.match(html, /aria-controls="main-options"/);
    assert.doesNotMatch(html, /datalist/);
    assert.deepEqual(modelOptions(['a', 'a', { id: 'b', name: 'B' }, { value: '', label: 'Engine default' }]), [
        { value: 'a', label: 'a' }, { value: 'b', label: 'B' }, { value: '', label: 'Engine default' },
    ]);
});
test('the supplying-program footer is a sibling of the listbox and placement is swappable', () => {
    const programs = [{harness: 'future', label: 'Future', programs: [{path: '/vendor/future', version: '7'}]}];
    for (const programPlacement of ['list', 'line']) {
        const html = modelChooserHtml('', 'typed', 'models', [], {programs, programPlacement});
        assert.match(html, /role="group" aria-label="Supplying program"/);
        assert.match(html, /<div id="models" role="listbox"><\/div>/);
        assert.match(html, /data-model-custom/);
        assert.match(html, new RegExp(`data-program-placement="${programPlacement}"`));
        assert.equal((html.match(/data-model-footer/g) || []).length, 1);
    }
});

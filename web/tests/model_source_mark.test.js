// The mark of a model's source (model_source_mark.js): a subscription draws its harness's mark, an API key its
// provider's sourced one, and a source without a mark of its own shows its initial, never an invented logo.
import assert from 'node:assert/strict';
import test from 'node:test';
import { sourceMark } from '../modules/model_source_mark.js';

test('a source draws its own sourced mark; one without a mark shows its initial, never an invented logo', () => {
    const api = (kind, label) => ({ lane: 'api', kind, label });
    const openrouter = sourceMark(api('openrouter', 'OpenRouter'));
    assert.match(openrouter, /^<svg class="model-source-mark" viewBox="0 0 24 24" aria-hidden="true"/);
    assert.match(openrouter, /d="M16\.778 1\.844v1\.919/, 'OpenRouter\'s split arrows (simple-icons 16.36.0), not a boxed O');
    assert.match(sourceMark(api('openai', 'OpenAI')), /^<svg class="model-source-mark" viewBox="0 0 256 260"/, 'the harness SSOT\'s knot');
    assert.match(sourceMark({ lane: 'subscription', harness: 'codex', label: 'Codex' }), /^<svg class="model-source-mark" viewBox="0 0 256 260"/);
    assert.match(sourceMark({ lane: 'subscription', harness: 'claude', label: 'Claude Code' }), /viewBox="0 0 24 24"/);
    assert.match(sourceMark(api('zai', 'Z.ai (GLM)')), /model-source-initial" aria-hidden="true">Z</);
    assert.match(sourceMark({ lane: 'subscription', harness: 'newcomer', label: 'newcomer' }), /model-source-initial" aria-hidden="true">N</);
    assert.match(sourceMark({ lane: 'local', sourceLabel: 'Local' }), /model-source-initial" aria-hidden="true">L</);
    assert.match(sourceMark(null), /model-source-initial" aria-hidden="true">·</, 'nothing read yet: no letter is invented');
    assert.match(sourceMark(api('zai', '<b>')), /aria-hidden="true">&lt;</, 'a label is text');
});

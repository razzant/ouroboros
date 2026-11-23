// The mark of a model's source (model_source_mark.js) on the path a route takes to it (chat_model_routes.describeRoute):
// a subscription draws the mark of the program that presents it — the one visual answer of
// harness_presentation.sourcePresentation: a held record decides, and without one the saved source keeps its own
// exact known mark (owner 22.11: both named sources stay recognizable). An API key its provider's sourced one, and
// any other source the generic harness mark: never a letter, never a logo read from the spelling of an id.
import assert from 'node:assert/strict';
import test from 'node:test';
import { buildModelIndex, describeRoute } from '../modules/chat_model_routes.js';
import { GENERIC_HARNESS_MARK, HARNESS_MARKS } from '../modules/harness_presentation.js';
import { sourceMark } from '../modules/model_source_mark.js';

const CODEX = { id: 'codex', label: 'Codex', credentialHarness: 'codex' };
// The core's answer when the engine did not name its sources in time (gateway/models.py `_subscription_timeout`).
const TIMED_OUT = { items: [], model_sources: [], partial: true,
    errors: [{ provider_id: 'claudexor', stage: 'timeout', error: 'Subscription catalogs did not answer in time.' }] };
const COLD = [['no index', null], ['not read', {}], ['timed out', TIMED_OUT]];
const held = (...model_sources) => ({ items: [], model_sources });
// A valid read that no longer lists the saved source's own route: the source stays recognizable (owner 22.11).
const STATES = [...COLD, ['read, this source unlisted', held(CODEX)]];
const pathOf = (html) => /\bd="([^"]+)"/.exec(html)?.[1];
/** The path of the mark `model` wears in a catalogue state (null: a route described with no index at all). */
const drawn = (model, catalog, use_local = false) =>
    pathOf(sourceMark(describeRoute(catalog && buildModelIndex({ catalog }), { model, use_local })));

test('both saved named sources keep their own vectors before, without and after a read that no longer lists them; a held record decides', () => {
    for (const [state, catalog] of STATES) {
        assert.equal(drawn('claudexor::codex=gpt-6-astra', catalog), HARNESS_MARKS.codex.path, `codex, ${state}`);
        assert.equal(drawn('claudexor::claude=claude-fable-5-2', catalog), HARNESS_MARKS.claude.path, `claude, ${state}`);
        assert.equal(drawn('claudexor::agy=claude-x', catalog), HARNESS_MARKS.agy.path, `agy, ${state}`);
    }
    // The record wins over the saved mark: one that names no program, one that names another, one that binds an opaque id.
    assert.equal(drawn('claudexor::codex=gpt-6-astra', held({ ...CODEX, credentialHarness: '' })), GENERIC_HARNESS_MARK.path,
        'an explicitly unbound record names no program');
    assert.equal(drawn('claudexor::codex=gpt-6-astra', held({ ...CODEX, credentialHarness: 'claude' })), HARNESS_MARKS.claude.path,
        'a record may bind another program');
    assert.equal(drawn('claudexor::claude=claude-fable-5-2', held({ id: 'claude', label: 'Claude Code', credentialHarness: 'codex' })),
        HARNESS_MARKS.codex.path);
    assert.equal(drawn('claudexor::opaque=gpt-x', held({ id: 'opaque', label: 'Model service', credentialHarness: 'codex' })),
        HARNESS_MARKS.codex.path, 'an opaque id its record binds');
    assert.equal(drawn('claudexor::agy-models=x', held({ id: 'agy-models', label: 'Antigravity models', credentialHarness: 'agy' })),
        HARNESS_MARKS.agy.path);
});

test('an id with no mark entry of its own wears the generic mark in every state; keys keep their providers\' sourced marks', () => {
    for (const [state, catalog] of STATES) {
        // A case variant and a reserved channel word are spellings, not programs: the drawn vector stays neutral.
        for (const model of ['claudexor::future=x', 'claudexor::CLAUDE=x', 'claudexor::native=x', 'zai::glm-5.3']) {
            assert.equal(drawn(model, catalog), GENERIC_HARNESS_MARK.path, `${model}, ${state}`);
        }
        assert.equal(drawn('qwen3.gguf', catalog, true), GENERIC_HARNESS_MARK.path, `local, ${state}`);
        // A key keeps its provider's sourced mark: the same Claude through OpenRouter is OpenRouter's, never a program's.
        assert.match(drawn('anthropic/claude-opus-5.5', catalog), /^M16\.778 1\.844v1\.919/, 'OpenRouter\'s split arrows');
        assert.match(drawn('anthropic::claude-opus-5-5', catalog), /^M17\.3041 3\.541/, 'Anthropic\'s');
        assert.equal(drawn('openai::gpt-6', catalog), HARNESS_MARKS.codex.path, 'OpenAI\'s knot, the harness SSOT\'s');
    }
    assert.equal(pathOf(sourceMark(null)), GENERIC_HARNESS_MARK.path, 'nothing read yet');
    assert.match(sourceMark(null), /^<svg class="model-source-mark" viewBox="0 0 24 24" aria-hidden="true" focusable="false">/);
    assert.match(sourceMark({ lane: 'api', kind: 'openai-compatible' }), /stroke="currentColor"/, 'the compatible endpoint\'s stroke');
});

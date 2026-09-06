// The ONE test that crosses the endpoint -> module WIRE.
//
// Every other web test in this range exercises a PURE PROJECTOR: it hands a
// hand-written object to a render function and checks the string. That is why
// three separate defects shipped at once and none of them was caught —
// `non_final_rows` was read by the client and never emitted by the server,
// `executor_route` was emitted by the server and never forwarded by the live
// chat path, and the reviewer-slot profile index read a shape the wire does not
// send. A projector test cannot see any of those, because it never asks whether
// the object it was handed is the object the server actually produces.
//
// So this file asserts the SEAM instead of either side of it: the field names
// the client reads off a server payload must be field names the server emits,
// derived from the real Python projection tuples and the real client source
// rather than restated by hand in the test.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { accountRows } from '../modules/harness_accounts.js';
import { nextUpAccount } from '../modules/claudexor_status_store.js';
import { indexProfilesByHarness } from '../modules/reviewer_slots.js';

// Source pins below delimit across line breaks; normalize CRLF so a Windows
// checkout (core.autocrlf) reads the same bytes the delimiters were written for.
const repoFile = (rel) => readFileSync(
    fileURLToPath(new URL(`../../${rel}`, import.meta.url)), 'utf-8',
).replace(/\r\n?/g, '\n');
const moduleFile = (rel) => readFileSync(
    fileURLToPath(new URL(`../modules/${rel}`, import.meta.url)), 'utf-8',
).replace(/\r\n?/g, '\n');

/** Names inside a Python tuple literal assigned to `name = (...)`. */
function pythonTupleNames(source, name) {
    const start = source.indexOf(`${name} = (`);
    assert.notEqual(start, -1, `${name} not found — the wire contract moved, update this test`);
    const end = source.indexOf('\n)\n', start);
    // A missing closer must fail loudly: a slice to EOF over-fills the name set
    // and lets the subset assertions below pass vacuously.
    assert.notEqual(end, -1, `${name} tuple closer not found — update this test`);
    const body = source.slice(start, end);
    return new Set([...body.matchAll(/"([a-z_]+)"/g)].map((m) => m[1]));
}

test('every accounting field the costs page reads is a field the history endpoint emits', () => {
    // v7 moved the cost-breakdown endpoint (and its field tuple) out of
    // gateway/history.py into gateway/cost_breakdown.py; history.py re-exports it.
    const costBreakdown = repoFile('ouroboros/gateway/cost_breakdown.py');
    const taskResults = repoFile('ouroboros/task_results.py');
    // What the server puts on the wire: the projected ledger fields plus the four
    // keys the endpoint attaches itself right after the projection.
    const emitted = pythonTupleNames(costBreakdown, '_ACCOUNTING_SUMMARY_FIELDS');
    for (const literal of ['available', 'authority', 'limit_usd', 'remaining_known_usd']) {
        emitted.add(literal);
    }
    // What the client reads off it, taken from the client source.
    const read = new Set(
        [...moduleFile('costs.js').matchAll(/\baccounting\.([a-z_]+)/g)].map((m) => m[1]),
    );
    assert.ok(read.size >= 8, 'no accounting reads found — the regex or the module moved');
    const missing = [...read].filter((field) => !emitted.has(field));
    assert.deepEqual(missing, [], `costs.js reads server fields that are never emitted: ${missing}`);
    // `cost_final`'s disclosed cause must travel WITH the flag, in both directions.
    assert.ok(emitted.has('non_final_rows') && read.has('non_final_rows'));
    // The durable/replay carry list is no longer a hand-typed copy: it is DERIVED
    // from the cost SSOT (C12), so assert the SSOT owns the name and that
    // task_results really derives from it — a re-typed literal there is the drift
    // this test exists to catch.
    const openness = pythonTupleNames(repoFile('ouroboros/cost_projection.py'), 'COST_OPENNESS_FIELDS');
    assert.ok(openness.has('non_final_rows') && openness.has('ledger_integrity_degraded'));
    assert.match(taskResults, /TASK_COST_META_FIELDS = tuple\(/);
    assert.match(taskResults, /COST_OPENNESS_FIELDS/);
});

test('the update letter typedef promises exactly the fields the projection emits', () => {
    const typedef = moduleFile('api_types.js');
    const start = typedef.indexOf('@typedef {Object} UpdateLetter');
    assert.notEqual(start, -1, 'UpdateLetter typedef not found — update this test');
    const end = typedef.indexOf('*/', start);
    assert.notEqual(end, -1, 'UpdateLetter typedef closer not found — update this test');
    const promised = new Set(
        [...typedef.slice(start, end).matchAll(/@property \{[^\n]*?\} ([a-z_]+)/g)].map((m) => m[1]),
    );
    // What the server emits: the keys of project_letter's return dict.
    const py = repoFile('ouroboros/update_letter.py');
    const fn = py.indexOf('def project_letter(');
    assert.notEqual(fn, -1, 'project_letter not found — update this test');
    const ret = py.indexOf('return {', fn);
    const retEnd = py.indexOf('\n    }\n', ret);
    assert.ok(ret > -1 && retEnd > -1, 'project_letter return dict not found — update this test');
    const emitted = new Set([...py.slice(ret, retEnd).matchAll(/^\s+"([a-z_]+)":/gm)].map((m) => m[1]));
    assert.ok(promised.size >= 8 && emitted.size >= 8, 'too few fields found — the regex or the source moved');
    assert.deepEqual([...promised].sort(), [...emitted].sort());
    // …and the panel reads nothing the typedef does not promise.
    const read = new Set([...moduleFile('updates.js').matchAll(/\bletter\.([a-z_]+)/g)].map((m) => m[1]));
    assert.ok(read.size >= 6, 'no letter reads found — the regex or the module moved');
    const unknown = [...read].filter((field) => !promised.has(field));
    assert.deepEqual(unknown, [], `updates.js reads letter fields the typedef does not promise: ${unknown}`);
});

test('the live progress path forwards every progress field the endpoint emits and the chat UI consumes', () => {
    const emitted = pythonTupleNames(repoFile('ouroboros/gateway/history.py'), '_PROGRESS_META_FIELDS');
    for (const field of ['executor_route', 'model_lane', 'status', 'subagent_event',
        'execution_evidence', 'actual_substrate']) {
        assert.ok(emitted.has(field), `${field} is no longer emitted by the history endpoint`);
    }
    // executor_route drives the executor chip in log_events.js; it must reach the
    // live summarizer, not only the replayed history row. The evidence/substrate
    // pair upgrades the same chip at terminal, so the whole trio travels together.
    assert.match(moduleFile('log_events.js'), /evt\?\.executor_route/);
    assert.match(moduleFile('log_events.js'), /evt\.execution_evidence/);
    assert.match(moduleFile('log_events.js'), /evt\?\.actual_substrate/);
    // Every LIVE call site that ENUMERATES fields instead of spreading the frame is
    // a whitelist, and a whitelist silently drops whatever it forgot — which is how
    // a chip came back on reload and was missing while the task ran.
    const chat = moduleFile('chat.js');
    const DELEGATION_KEYS = ['executor_route', 'execution_evidence', 'actual_substrate'];
    const whitelists = chat.split('summarizeChatLiveEvent({').slice(1)
        .map((chunk) => chunk.slice(0, chunk.indexOf('});')))
        .filter((chunk) => !chunk.includes('...evt'));
    assert.ok(whitelists.length > 0, 'no enumerated live call site found — update this test');
    for (const chunk of whitelists) {
        const forwarded = new Set([...chunk.matchAll(/^\s+([a-z_]+):/gm)].map((m) => m[1]));
        for (const key of DELEGATION_KEYS) {
            assert.ok(forwarded.has(key),
                `a chat.js live whitelist drops ${key}: the chip only tells the truth after a reload`);
        }
    }
    // The SECOND whitelist of the same class: routeSubagentTerminalToCard
    // synthesizes a terminal from the log-channel task_done through an
    // enumerated updateSubagentCardFromEvent({...}) literal. Forgetting the
    // delegation keys there keeps a log-channel-only card at "no run yet"
    // forever — the exact defect family this file was written for.
    const terminalWhitelists = chat.split('updateSubagentCardFromEvent({').slice(1)
        .map((chunk) => {
            // The literal closes with `}, ts)` — cut at its closing brace line so
            // the scanned chunk is exactly the enumerated field list.
            const end = chunk.search(/^\s*\},/m);
            return end === -1 ? chunk : chunk.slice(0, end);
        });
    assert.ok(terminalWhitelists.length > 0,
        'no enumerated updateSubagentCardFromEvent call site found — update this test');
    for (const chunk of terminalWhitelists) {
        const forwarded = new Set([...chunk.matchAll(/^\s+([a-z_]+):/gm)].map((m) => m[1]));
        for (const key of DELEGATION_KEYS) {
            assert.ok(forwarded.has(key),
                `the synthesized subagent terminal drops ${key}: a log-channel-only terminal cannot upgrade the chip`);
        }
    }
});

test('both consumers of the credential-profiles wire read the SAME shape', () => {
    // One golden body, two independent readers. The reviewer-slot index used to
    // read flat camelCase off the `{profile,status,identity}` wrapper and matched
    // nothing, so every session row had an empty profile picker while the harness
    // account list beside it rendered the same accounts correctly.
    const payload = { profiles: JSON.parse(readFileSync(
        fileURLToPath(new URL('./fixtures/credential_profiles_response.json', import.meta.url)),
        'utf-8',
    )) };
    const fromAccounts = accountRows(payload)
        .filter((row) => row.kind === 'profile')
        .map((row) => `${row.harness}/${row.profile_id}`)
        .sort();
    const index = indexProfilesByHarness(payload);
    const fromSlots = Object.entries(index)
        .flatMap(([harness, entries]) => entries.map((entry) => `${harness}/${entry.id}`))
        .sort();
    assert.ok(fromAccounts.length > 0, 'fixture carries no profile rows');
    assert.deepEqual(fromSlots, fromAccounts);
});

test('the UNIFIED wire shape feeds the same readers: all rows named, pools carry routing', () => {
    // The second golden body — a unified engine's ControlCredentialProfilesResponse
    // (frozen contract §L.1): the migrated default login is an ordinary
    // registry row (reserved `<harness>-default` id), `harnessAccounts` is the
    // empty compatibility key, and the routing verdict lives in the additive
    // `accountPools`. Same two readers, no unified-specific branch in either.
    const payload = {
        unified_accounts: true,
        profiles: JSON.parse(readFileSync(
            fileURLToPath(new URL('./fixtures/credential_profiles_response_unified.json', import.meta.url)),
            'utf-8',
        )),
    };
    const rows = accountRows(payload);
    assert.ok(rows.length > 0, 'fixture carries no rows');
    assert.ok(rows.every((row) => row.kind === 'profile'),
        'a unified payload synthesizes no native pseudo-rows');
    // The migrated default is pinnable through the same index the reviewer
    // rows read — the reviewer-side half of the unification.
    const index = indexProfilesByHarness(payload);
    assert.ok((index.codex || []).some((entry) => entry.id === 'codex-default'));
    // The enabled projection reaches both consumers from one reader.
    const byId = Object.fromEntries(rows.map((row) => [row.profile_id, row.enabled]));
    assert.deepEqual(byId, { 'codex-default': true, koshak: false });
    // …and the index carries the same fact, so the pin selects can label a
    // disabled account instead of offering it bare.
    const indexEnabled = Object.fromEntries(Object.values(index).flat()
        .map((entry) => [entry.id, entry.enabled]));
    assert.deepEqual(indexEnabled, byId);
    // Routing rides the pool, read through the store's one dual-wire reader.
    assert.deepEqual(nextUpAccount(payload, 'codex'),
        { kind: 'profile', profileId: 'codex-default' });
});

test('delivered photo, video, and document rows keep their replay wire fields', () => {
    const history = repoFile('ouroboros/gateway/history.py');
    const chat = moduleFile('chat.js');

    const documentBranch = history.slice(
        history.indexOf('if entry.get("type") == "document":'),
        history.indexOf('elif entry.get("type") in {"photo", "video"}',
            history.indexOf('if entry.get("type") == "document":')),
    );
    for (const field of ['msg_type', 'filename', 'mime', 'download_url', 'caption', 'size_bytes']) {
        assert.match(documentBranch, new RegExp(`rec\\["${field}"\\]`),
            `document replay no longer emits ${field}`);
    }

    const mediaBranchStart = history.indexOf('elif entry.get("type") in {"photo", "video"}');
    const mediaBranch = history.slice(mediaBranchStart, history.indexOf('\n            if ', mediaBranchStart));
    // Same keyword-argument idiom as the links branch below.
    for (const field of ['msg_type=', 'mime=', 'download_url=', 'download_url_compat=', 'caption=']) {
        assert.ok(mediaBranch.includes(field), `photo/video replay no longer emits ${field}`);
    }

    const linksBranchStart = history.indexOf('elif entry.get("type") == "links"');
    const linksBranch = history.slice(linksBranchStart, history.indexOf('\n            if ', linksBranchStart));
    for (const field of ['msg_type="links"', 'actions=', 'title=']) {
        assert.ok(linksBranch.includes(field), `links replay no longer emits ${field}`);
    }

    assert.match(chat, /\['document', 'photo', 'video', 'links', 'quiz'\]\.includes\(msg\.msg_type\)/);
    assert.match(chat, /if \(msg\.msg_type === 'document'\) appendDocumentBubble\(msg\);/);
    assert.match(chat, /else if \(msg\.msg_type === 'links'\) appendLinksMessage\(msg\);/);
    assert.match(chat, /else appendMediaBubble\(msg\);/);
});

test('live structured delivery frames keep additive grouping and size fields', () => {
    const bus = repoFile('supervisor/message_bus.py');
    const contracts = repoFile('ouroboros/gateway/contracts.py');
    const types = moduleFile('api_types.js');

    for (const className of ['PhotoOutbound', 'VideoOutbound', 'DocumentOutbound', 'LinksOutbound', 'QuizOutbound']) {
        assert.ok(contracts.includes(`class ${className}(TypedDict):`));
        assert.ok(types.includes(`@typedef {Object} ${className}`));
    }
    assert.match(bus, /"type": "photo",[\s\S]*?"task_id": str\(task_id or ""\)/);
    assert.match(bus, /"type": "video",[\s\S]*?"task_id": str\(task_id or ""\)/);
    assert.match(bus, /"type": "document",[\s\S]*?"size_bytes": len\(file_bytes\),[\s\S]*?"task_id": str\(task_id or ""\)/);
    assert.match(bus, /"type": "links",[\s\S]*?"actions": validated,[\s\S]*?"task_id": str\(task_id or ""\)/);
    assert.match(bus, /"type": "quiz",[\s\S]*?"quiz_id": qid,[\s\S]*?"task_id": str\(task_id or ""\)/);
    // Replay: the quiz row joins the non-terminal delivery family in chat.js
    // BEFORE the finishLiveCard/plainUntypedFinal block.
    const chat = moduleFile('chat.js');
    assert.match(chat, /msg\.msg_type === 'quiz'\) appendQuizMessage\(msg\)/);
    // The card gets the SAME sanitizing markdown pipeline as assistant bubbles.
    assert.match(chat, /renderMarkdown: renderChatMarkdown/);
    assert.match(chat, /enhanceMarkdown: enhanceMountedMarkdown/);
    assert.match(contracts, /WS_MESSAGE_TYPES[\s\S]*?"links"/);
});

test('history typedef declares task outcome terminality fields', () => {
    const types = moduleFile('api_types.js');
    assert.match(types, /@property \{"working"\|"done"\|"warn"\|"error"\|"cancelled"=\} outcome_phase/);
    assert.match(types, /@property \{boolean=\} outcome_final/);
});

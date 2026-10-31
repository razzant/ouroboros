// Offline integration driver for the production engine, adapted from its
// packages/cli/src/model-services.test.ts. Only auth, readiness and provider
// HTTP are fixtures. Account selection, continuations, SQL, operations,
// classification, marks, uploads and the HTTP control API are production code.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, basename, dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [enginePath, root, scenario] = process.argv.slice(2);
if (!enginePath || !root || !scenario) throw new Error('engine path, isolated root and scenario required');
mkdirSync(root, { recursive: true });
process.env.CLAUDEXOR_CONFIG_DIR = root;
const load = (pkg, file = 'index') => import(pathToFileURL(join(enginePath, 'packages', pkg, 'dist', `${file}.js`)));
const daemon = await load('daemon');
const schema = await load('schema');
const { updateGlobalConfig } = await load('config');
const { createCodexAdapter, createCodexModelAdapter } = await load('harness-codex');
const { createModelServices } = await load('cli', 'model-services');
const { createCredentialProfilesService } = await load('cli', 'accounts-services');
const { DaemonControlApiServer } = await load('control-api', 'daemon-server');
// Never allow a fixture bug to reach a real provider.
globalThis.fetch = async () => { throw new Error('External fetch forbidden in the offline engine fixture'); };
const store = await daemon.EngineStore.open({
    daemonDir: join(root, 'daemon'),
    workerEntry: resolve(enginePath, 'packages/daemon/dist/store/flusher-worker.js'),
});
if (!daemon.currentGeneration(store, 'global')) {
    store.transaction(() => daemon.setGlobalGenerationInTx(store, daemon.createPartition(store, 'global').pid));
}
const graph = daemon.createSqlDaemonServices(store, { purgeFiles: async () => [root] });
const quota = new daemon.QuotaRegistry(graph.globalEvents);
const unusable = new daemon.CredentialUnusableLedger();
// Real time, so each mark has its own episode start; only the expiry story skews it.
let skew = 0;
const capabilities = daemon.ModelCapabilityLedger
    ? new daemon.ModelCapabilityLedger(unusable.credentials, () => Date.now() + skew) : undefined;
if (scenario === 'retry_invalid') {
    const claim = capabilities.claimRetry.bind(capabilities);
    capabilities.claimRetry = (request, id) => {
        if (request.retryOfOperationId) skew += 5 * 60 * 60 * 1000;
        return claim(request, id); // real expiry validation, with an injected clock
    };
}
const substitutions = new daemon.ModelSubstitutionLedger();
const refusals = new daemon.PreProgressRefusalLedger(() => new Date(), () => false, unusable.credentials);
const accountIds = scenario === 'no_alternative' ? ['a'] : scenario === 'third' ? ['a', 'b', 'c'] : ['a', 'b'];
const profiles = accountIds.map(id =>
    schema.CredentialProfile.parse({ profile_id: id, harness_id: 'codex', display_name: id,
        credential_kind: 'config_dir_login', isolation_locator: join(root, id) }));
const cfg = schema.GlobalConfig.parse({ credential_profiles: profiles });
updateGlobalConfig(() => cfg);
for (const profile of profiles) quota.upsert(schema.QuotaSnapshot.parse({
    subject: { harness: 'codex', subject_id: profile.profile_id, credential_route: 'vendor_native', plan_label: null },
    constraints: [{ id: 'five_hour', label: '5h', used_ratio: { a: 0.01, b: 0.4, c: 0.6 }[profile.profile_id],
        window_seconds: 18000, resets_at: null, cooldown_until: null }],
    source: 'codex_app_server', observed_at: new Date().toISOString(), freshness: 'fresh',
}));
const posts = [];
const adapter = createCodexModelAdapter({
    clientVersion: async () => ({ version: '0.156.1', source: 'verified_transport' }),
    readAuthFile: async path => {
        const id = basename(dirname(path));
        const jwt = value => `fixture.${Buffer.from(JSON.stringify(value)).toString('base64url')}.signature`;
        return JSON.stringify({ auth_mode: 'chatgpt', tokens: {
            account_id: id, access_token: jwt({ exp: 2100000000 }),
            id_token: jwt({ sub: id, 'https://api.openai.com/auth': {
                chatgpt_plan_type: id === 'a' ? 'free' : 'pro' } }),
        } });
    },
    fetch: async (_url, init) => {
        if (init?.method !== 'POST') return Response.json({ models: [{ slug: 'gpt-6.1-sol' }] });
        const account = new Headers(init.headers).get('chatgpt-account-id');
        posts.push({ account, body: JSON.parse(await new Response(init.body).text()) });
        writeFileSync(join(root, 'provider-posts.json'), JSON.stringify(posts, null, 2));
        const seed = scenario === 'continuation' && posts.length === 1;
        // The test steers which accounts serve between its calls (scenario "returns").
        const mode = (() => { try { return JSON.parse(readFileSync(join(root, 'provider-mode.json'), 'utf8')); }
            catch { return null; } })();
        const refuses = mode ? !mode.serving.includes(account)
            : !seed && (account === 'a' || ['both', 'third', 'generic_both', 'returns'].includes(scenario) && account !== 'c');
        if (refuses && scenario.startsWith('generic')) return Response.json({
            error: { message: 'The request was refused.', type: 'invalid_request_error', code: null },
        }, { status: 400 });
        if (refuses) return Response.json({
            detail: "The 'gpt-6.1-sol' model is not supported when using Codex with a ChatGPT account.",
        }, { status: 400 });
        return new Response(`data: ${JSON.stringify({ type: 'response.completed', response: {
            model: 'gpt-6.1-sol', output: [
                ...(seed ? [{ type: 'reasoning', id: 'rs_fixture', encrypted_content: 'fixture-sealed' }] : []),
                { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: `served by ${account}` }] },
            ], usage: { input_tokens: 2, output_tokens: 3 },
        } })}\n\n`);
    },
});
const probe = async profile => ({ profile_id: profile.profile_id, harness_id: profile.harness_id,
    availability: 'available', verification: 'passed', verification_source: 'local_store',
    detail: 'Offline fixture account', last_verified_at: null });
const client = new daemon.DaemonClient(join(root, 'daemon.sock'), 'fixture-control');
const services = createModelServices({
    commands: graph.commands, resourceQueries: graph.commands.queries, resources: () => graph.resources,
    client, quota: () => quota, config: () => cfg, unusable, substitutions, capabilities, refusals,
    migrationGate: () => null,
    registry: new Map([['codex', { ...createCodexAdapter(), probeCredentialProfile: probe }]]),
    sources: [{ adapter, label: 'Codex', credentialHarness: 'codex' }],
});
const server = new daemon.DaemonServer({
    socketPath: join(root, 'daemon.sock'), token: 'fixture-control', commands: graph.commands,
    runner: (params, context) => {
        if (!schema.isModelOperation(params)) throw new Error('Agent runs forbidden in model fixture');
        return services.operations.execute(params, context);
    },
    onCommandTerminal: record => services.operations.onCommandTerminal(record),
});
await server.start();
const accounts = createCredentialProfilesService(() => quota, { ...graph.accounts, capabilities, refusals });
const api = new DaemonControlApiServer({
    token: 'fixture-control', daemon: client, port: 0,
    services: {
        ...services.routes, accountRoutingEvidence: accounts.accountRoutingEvidence,
        createUpload: async (input, key) => graph.resources.create(input, key),
        writeUpload: async (id, chunks) => graph.resources.write(id, chunks),
        uploadStatus: async id => graph.resources.status(id),
        cancelUpload: async id => graph.resources.cancel(id),
        finalizeUpload: async (id, sha, key) => graph.resources.finalize(id, sha, key),
        validateResources: async refs => { graph.resources.resolve(refs); },
        // Readiness is the same injected fixture as execution, never a CLI probe.
        credentialProfiles: async () => ({ profiles: await Promise.all(profiles.map(async profile =>
            ({ profile, status: await probe(profile), identity: null }))), harnessAccounts: [], accountPools: [] }),
        quota: async () => quota.read(),
    },
});
const address = await api.start();
console.log(JSON.stringify({ host: address.host, port: address.port, token: 'fixture-control' }));
let closing = false;
const close = async () => {
    if (closing) return;
    closing = true;
    await api.stop();
    services.close();
    await server.stop();
    await graph.close();
    await store.close();
};
process.on('SIGTERM', () => close().then(() => process.exit(0)));
process.on('SIGINT', () => close().then(() => process.exit(0)));

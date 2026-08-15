# DEVELOPMENT.md — Development Principles & Module Guide

## Role and authority

This is Ouroboros's engineering handbook: how to name, design, implement, and
verify changes. `BIBLE.md` owns constitutional principles;
`docs/ARCHITECTURE.md` owns the current structure, data flow, and rationale map;
`docs/CHECKLISTS.md` owns reviewer items, severity, and output contracts. This
file does not duplicate their inventories or serve as a changelog.

Rules here describe current practice or a deliberately enforced standard. When
code and prose disagree, inspect the implementation and history, repair the
authoritative surfaces together, and retain the failure a non-obvious rule
prevents.

---

## Naming and boundaries

- Code identifiers, comments, docstrings, and commit messages are English.
- Follow PEP 8: modules and variables use `snake_case`, classes use
  `PascalCase`, and constants use `UPPER_SNAKE_CASE`.
- Name the observable responsibility and authority, not the implementation
  fashion. Prefer a clear function module over a class with no lifecycle.
- The skill-review checklist in `docs/CHECKLISTS.md` and `_SKILL_REVIEW_ITEMS` in
  `ouroboros/skill_review.py` are ONE list, pinned by
  `tests/test_skill_review_checklist_ssot.py`. The table is loaded into the
  reviewing model's context and the tuple is what the parser demands back, so a
  disagreement (it said "17 items total" and numbered 17 rows while the code
  required 16) puts a contradiction in the reviewer's own context.

The repository has two deliberately different gateway directions:

- `ouroboros/gateway/` is the inbound browser/CLI HTTP and WebSocket facade.
  `contracts.py` owns typed shapes, `router.py` owns route collection, and
  domain handlers translate transport into calls on existing runtime owners.
- `ouroboros/gateways/` contains thin outbound adapters to external runtimes.
  They translate calls and errors but do not choose product policy, fallback,
  custody, or authorization.

Contracts are typed shapes, not service objects. A manager is justified when it
owns lifecycle or mutable state. LLM-callable Tools remain thin `{verb}_{noun}`
functions that validate their public input, call the owning subsystem, and
format a result. No universal `{Domain}Service`, `{Platform}Gateway`, or class
layer is required.

Dependency direction is the test: UI/CLI → inbound gateway → domain owner;
runtime policy → small host-owned contract → outbound adapter. Provider- or
transport-specific decisions do not flow back into core policy.

### CLI and headless work

- CLI commands parse, call the existing gateway/scheduler, and render
  text or typed JSON/JSONL/SSE. They do not create a second task state machine.
- External workspace tasks keep governance bound to the system repository while
  contextual tools default through `ToolContext.active_repo_dir()`. Admission
  rejects overlap with the system repo/data and records a read-only preflight.
- Project focus changes the default target, not the ordinary top-level tool
  surface. Generic VCS selects active/system explicitly; advisory, reviewed
  commit, rollback, promotion, restart, and runtime control keep their intrinsic
  system-repository contracts and existing gates. Workspace finalization still
  returns durable patch artifacts. `executor_ref` selects a process backend, not
  an implicit sandbox.
- Project-local installs may run within the workspace policy. Global/system
  installs remain safety-reviewed, and `sudo` is non-interactive (`sudo -n`).
- Do not add a second scheduler for operator tooling or a generic CLI file
  manager. Use the task queue, attachments, logs, and artifact endpoints.

### Remote placement and the target boundary

One mind on Home, hands on the target: a Project can be placed on a remote Linux
host and its tasks execute there through a restricted `ouroboros-execd`. These rules
keep that a placement, not a second runtime.

- An SSH workspace has NO native Home path. Read the sealed `workspace_ref`
  (`ouroboros/workspace_ref.py`) — never derive placement, and never write an
  ssh `ExecutorRef` independently of it. A Home `Path` request for a remote
  placement must fail loudly; falling back to the system repo is the failure
  mode this design exists to make impossible. Local and SSH placement must
  expose the same model-facing tool names and schemas at equal
  role/runtime/resource policy.
- Do not add a placement branch ahead of the guard pipeline. Remote work goes
  through prepare → authorize → execute: prepare gathers target-native facts and
  a bound token, the FULL existing guard set runs once over the three
  projections, and only then does the executor facade run the authorized token.
  A guard must consume a facts object, not call `Path` itself. A second guard,
  registry, policy engine, or artifact authority on the remote side is a
  regression, not a feature.
- Never narrow a gate's scope by pointing at a compensating control that does not
  exist. The §3.3 isolation gate read module-scope imports only and justified it
  in a comment — "a clean-subprocess invocation smoke covers those per native
  operation" — which was never written; two real violations lived in that gap for
  exactly as long as the sentence did. If a scope is genuinely out of reach, say
  so plainly and name what is therefore unchecked. If a compensating control is
  claimed, it must be a test that has been SEEN red.
- A bundle/transport module must not name a Home authority in ANY scope, function
  bodies included. When one needs a value from Home settings, move the value to
  the module that needs it (if it needs nothing from Home) or move the consumer
  out of the bundle — do not paper over it with a late import.
- Home-only consumers of remote source must go through the shared snapshot
  bridge, verify every declared blob, and clean the temporary mirror on success
  AND failure. A policy-filtered snapshot may omit only disclosed
  sensitive/protected paths and stays explicitly PARTIAL; it is usable only when
  the independent integrity axis reports no read/walk/limit/stability failure.
  Integrity failures stay fail-closed.
- Export filtering happens SOURCE-SIDE, before a blob is constructed. Filtering
  after a fetch has already leaked. Home re-validates every returned manifest
  against the same policy before import, every blob kind that crosses the
  boundary is enumerated in the closed registry, and an unknown kind fails
  closed. Fetch externalized blobs only from envelope-declared size/SHA refs
  under per-blob and aggregate caps.
- A filter that does not DISCLOSE is not a safe filter — it is a false answer.
  Every channel that drops a path by policy must return the omission in the same
  disclosure block (`complete=false`, `policy_scope=policy_filtered`, exact
  `excluded_count`, bounded `excluded[]`) AND name it in the owner/model-facing
  text. This is not decoration: a silently filtered `search_code` returned "No
  matches found … (2 files searched)" over a workspace whose `.env` held the very
  string searched for, while the same query on a LOCAL workspace returned the
  line — so the model concluded "the key is not here" from a premise the filter
  invented. Never add a `continue`-on-policy without a recorded reason. Exception,
  and only this one: exclusions of the infrastructure-directory class (`.git`,
  `__pycache__`) are pruned identically by BOTH placements, so disclosing them
  adds noise without describing any divergence.
- Publish an imported artifact through the existing artifact authority
  (`artifacts.publish_verified_task_artifact`), never by reimplementing a copy:
  the destination is derived from `{task_id, import_id, canonical_name}`, so a
  same-hash retry is idempotent and a changed-hash conflict is loud. The public
  record carries NO remote source path — provenance belongs in the private
  import receipt. An integrity or import failure must not ACK the remote side.
- A remote mutation imported from a Home model requires the exact
  source/HEAD/index precondition, a complete before→after change manifest,
  `git apply --check`, an expected post-content fingerprint, and explicit
  rollback evidence. Never copy Home model or provider credentials to execd.
- Remote loopback browsing is the ONE documented exemption to the all-bytes
  rule and uses only the broker-owned non-multiplexed local forward. Reject
  inherited SSH forwards/commands/environment effects before spawn, bind only
  Home and remote loopback, require process custody, retry a bounded
  ephemeral-port race, and route-block the bridged page from unrelated
  Home/private origins. Do not add SOCKS, generic private-network proxying, or a
  browser-only model schema.
  The exemption has exactly ONE consumer seam, `tools/browser.py::_resolve_placement_url`,
  and placement is resolved there BEFORE the browser exists. Keep the three answers
  distinct when touching it: a loopback URL is the TARGET's service and is forwarded
  and rewritten; a private non-loopback address is refused as ambiguous (Home's LAN
  and the target's LAN are different networks and the URL names neither, so resolving
  it against Home is the wrong-host read one hop out); a public host and every URL on
  a local placement are untouched. The origin block is a per-request Playwright route,
  not a check at `goto` time — a redirect, XHR, websocket upgrade or click must
  re-evaluate it. The forward is owned by the TASK, so `cleanup_browser` must NOT drop
  the forward map: it also runs mid-task on a thread switch or engine change, and
  clearing it leaks one `ssh -L` child per browser rebuild.
- **`file://` across the placement boundary is DEFERRED.** There is no
  `remote_file_bridge` module and no filtered `file://` channel; do not document one
  as existing. A `file://` URL reads HOME's filesystem, which is correct for the roots
  that stay Home-native on every placement. A path that exists only on the target gets
  a typed refusal naming the deferral — never a bare "file not found", which would
  send the owner hunting for a file that is on their server.
- Connection administration stays owner-only and thin: the same authenticated
  gateway backs Settings and the CLI. `test` is a read-only transport/platform
  probe and must never initialize or change the continuity pin; only the first
  successful `bootstrap` may pin it. Selectability needs TWO pieces of evidence
  and they are split by what the fact is ABOUT, not by convenience: bootstrap
  COMPATIBILITY ("a compatible executor is installed on that host") is owner
  state, recorded durably by `connection_store.record_bootstrap` and cleared only
  by `retrust`/`retire`, because it does not stop being true when Home restarts;
  health FRESHNESS ("the target answered in the last few minutes") is a claim
  about THIS run over a monotonic clock and stays process-local in
  `gateway/connections.py`. So after a Home restart a plain `test` restores
  selectability, and `bootstrap` is needed again only when the executor itself
  must be replaced. Do not describe the compatibility half as process-local: it
  was, once, and the New Project picker came back permanently empty while the
  dialog's own copy promised that Test would refresh it. The store contains no
  SSH key, password, token, or raw option. "Owner-only"
  describes authenticated administration and mutation authority, NOT secrecy
  from arbitrary code already running as the same Home Unix user: the path
  guards are defense in depth, and a real confidentiality boundary would need a
  separate OS user/process boundary or a credential vault. Do not add a remote
  task runner, terminal emulator, TUI, SSH password store, or per-tool SSH
  command path.
- Before sending a remote `CONTINUE`, Home must fsync one bounded reconciliation
  intent under `state/remote_reconciliation/`, recording only operation/session
  identity plus a closed import kind — never args, prepared tokens, blobs, SSH
  settings, or credentials. The execd journal remains the sole execution truth;
  Home asks it to reconcile after a restart and removes the intent only after a
  verified import plus ACK, or a proved `not_started` result. Pending
  `*.pending.json` records and retained terminal-evidence `*.json` records are
  distinct: evidence retention must never prune an unacknowledged intent.
- The dispatch golden traces (`tests/golden_traces/`, checked by
  `tests/test_dispatch_golden_traces.py`) must cover every executor KIND the
  pipeline can route to, not only `local`. §9 asks for byte-identity on local plus
  docker mapped/unmapped, and for a while all 21 scenarios were `kind=local` —
  `workspace_executor_local` IS the local branch, so the docker executor had no
  coverage at all. `docker_exec_mapped_cwd` and `docker_exec_unmapped_root` close
  that behind a STUB `docker` on PATH (`scenarios.docker_stub_env`): a fixture must
  not depend on a daemon, a registry or a network, and the stub echoes only
  `--workdir` and the container name because the real argv wraps the command in a
  shell whose pidfile carries a fresh uuid4 per run. A stub is sufficient for what
  a golden trace pins — the routing decision, the host→backend cwd projection, the
  container/network fields, the recorded executor trace — and insufficient for
  anything about the container itself; the projection's agreement with a REAL
  `docker exec` is checked in the `integration` lane instead. When adding an
  executor kind, add its scenarios in the same change.
- `ssh` is the THIRD executor kind, and it went uncovered longest for the reason that
  makes it matter most: it is the only one that REPLACES the built-in handler, so its
  guard sequence is a different sequence — `_invoke_builtin_handler` never appears and
  `execute_native_operation` does — and a fixture set that never takes the placement
  fork cannot see the whole class of "a policy that stopped at the fork". The four
  `ssh_*` scenarios pin the remote order and the remote result TEXT: the two read doors
  with the D7 disclosure block the target emits on every call plus a Home-native root
  that must NOT route, the two write doors under the bound export policy, a process with
  a post-prepare shell refusal and the interpreter allowlist answering ahead of prepare,
  and a restricted subagent whose secret read is refused BEFORE prepare while its
  ordinary listing routes and comes back filtered AND disclosed. The wire is fake and the
  target is not — the `tests/test_registry_remote_dispatch` harness runs the real
  `workspace_native` kernel against a real temp worktree, and it is REUSED rather than
  restated, because two copies of the wiring drift and the fixture would then pin the
  copy. The remote-seam guards are recorded per SCENARIO (`ScenarioRun.remote_seam`), not
  per call: `prepare_operation` and `bind_execution_args` run on a local dispatch too, so
  an unconditional wrapper would have rewritten every existing fixture, and byte-identity
  of the local and docker fixtures is the property this directory exists to hold.
- `covers()` returning False means HOST fallback, not an error: a cwd outside every
  executor mapping runs locally even under a docker-backed workspace. That is a
  deliberate contract, it is pinned by `docker_exec_unmapped_root`, and inverting
  it (into a refusal, or worse into running host paths inside the container) is a
  behaviour change that must be argued, not slipped in.
- Every registered built-in needs an explicit placement: a workspace affinity in
  `WORKSPACE_TOOL_EXECUTION_AFFINITY` or a declaration in `HOME_ONLY_TOOL_NAMES`.
  Doing neither fails `tests/test_workspace_capability_manifest.py` by name.
  Exhaustiveness is checked against the REGISTRY on purpose — comparing the
  affinity table to `_WORKSPACE_ALLOWED_TOOLS` proved only that two hand-written
  constants agreed, while a tool in neither was classified Home-only by silence.
- **Skills on remote placement are a DEFERRED phase and the manifest field is not
  introduced.** There is no `scripts[].execution_affinity` and no
  `tool_execution_affinity`: no loader reads them, nothing validates them, and
  nothing blocks loading on a bad value. Do not document, review against, or write
  code that assumes them. When the phase lands, the field arrives together with
  loader validation, fail-closed behaviour and its own review item — a placement
  the manifest asserts while the runtime ignores it is a false safety claim, which
  is exactly what the docs promised before this correction.

### Cognitive quality

Do not lower model quality, reasoning effort, output budget, or context breadth
for consciousness, review, or self-evolution as an incidental latency/cost
optimization. An intentional narrowing is an owner decision reflected in the
plan, docs, tests, and evidence.

### LLM-first affordances

Do not repair a semantic tool-choice failure by adding one more keyword hint to
`prompts/SYSTEM.md`. Put stable discoverability in the tool schema or add a
typed affordance at the point of need. SYSTEM accretion trains around one
incident, bloats the resident prefix, and forks the authority.

### Pricing and admission

Never add hand-maintained model-price tables, inherited prefix tariffs, or
numeric fallback prices. Query the exact route when a live source exists,
prefer provider-settled usage, and otherwise preserve `cost=None` and
`cost_final=false`. Unknown price is neither free nor a model-admission veto;
known exhausted budget remains enforceable.

### Anti-pattern: content-derived identity for host-minted records (v6.73.0)

If the host itself created a record — a chat message, a task, a binding — its
identity is CAPTURED AT INGRESS and passed downstream BY VALUE as a typed
reference (e.g. `origin_message_ref = {chat_id, client_message_id, ts,
text_sha256}` built where `log_chat("in", …)` writes the canonical row). Do NOT
re-derive that identity later by searching logs/state for a row whose text
hash/equality/prefix matches whatever text the caller happens to hold: in an
LLM-first system the text is routinely rewritten between ingress and use, so
content-derived lookup fails exactly on the normal path (the four
start-message-loss incidents fixed serially before v6.73.0 were all this class).
Content hashes remain legitimate in two roles only: (a) an INTEGRITY CHECK on an
already-known identity (`text_sha256` inside a ref verifies the row wasn't
swapped — it is never the lookup key), and (b) content-ADDRESSING where the
content IS the identity (artifact stores, observability blobs, join-ledger
result hashes, staged-diff review bindings). One NAMED exception inside role (b),
v6.78.0 / owner Q28=B: a verification RECEIPT is reconciled by ONE TYPED
IDENTITY KEY — its `criterion_id` when it has one, else its canonical `check` text, else
(for the artifact-observation class, which runs no command) its observed `paths` SET —
because with no criterion id the check text / observed path set IS that verification's
identity (there is no earlier ingress point to capture, and a class whose identity is
dropped can never reconcile itself: a red "report.md missing" must be clearable by the
byte-identical green observation once the file exists). Two receipts name the same
verification when the key's KIND and VALUE both match, never across kinds. The key
replaced a per-component FALLBACK CHAIN, and the reason is worth stating as a rule: a
chain is not an equivalence relation. It was not transitive — `{c1, check}` matched
`{check}`, `{check}` matched `{c2, check}`, while `c1` and `c2` are explicitly different
criteria — so one check-only green reconciled two distinct reds, "collapse the candidates
onto the identity they name" was not even well defined, and the outstanding set came out
order-dependent. No care at the call sites can repair a relation that is not an
equivalence; keying makes sameness the KERNEL of a function (reflexive, symmetric,
transitive by construction) and makes an existing `criterion_id` authoritative
STRUCTURALLY rather than by a rule someone must remember. It is a SIMPLIFICATION —
strictly fewer branches — and it fails in the SAFE direction: strictly fewer
reconciliations, so a red the chain used to clear may now stay open. Concretely, a re-run
that OMITS the `criterion_id` it carried before no longer clears its own red; the cost is
one advisory nudge and one advisory reviewer flag, never a false green. Read that as the
design, not a regression. If omission tolerance is ever genuinely needed, the only sound
route is to carry the id forward STRUCTURALLY at receipt ingress (`tools/verify.py`) —
never to infer it back from shared command text under another name. Two paths deliberately
keep the older any-later-grounding rule, because there the key cannot do its job: a
receipt with NO key at all (nothing to protect — a malformed `artifact_observation` with
no paths would otherwise mint an unclearable red), and the MASKED-pass path, where the
only text identity is the masked command itself and the prescribed remediation ("drop the
masking pipe") necessarily changes it, so a byte-identical clean re-run cannot exist.
On that masked path the `criterion_id` key alone binds, by the same equality: a masked
receipt that NAMES a criterion is cleared only by a later clean receipt naming that SAME
criterion — one that merely omits its id does NOT clear it — and the any-later-clean
fallback reaches only a masked receipt that names no criterion. Command text never
participates there, in either direction. **And whatever decides must be what is
reported.** Both relations and both disclosures read ONE mode-aware projection —
`receipt_reconciliation_key(receipt, masked=…)`, with the mode read off the receipt by
`receipt_is_masked_pass` for the per-row question (`receipt_disclosed_reconciliation_key`)
— so `reconciliation_identity` and `expected_whitespace_normalized` name the authority
that actually cleared the receipt instead of re-deriving one beside it. Re-deriving is how
round 6 arrived: an id-less masked pass, reconciled by ANY later clean grounding, was
disclosed to the acceptance reviewer as `check`-governed with
`expected_whitespace_normalized=true`, while `_reconciles_masked` never looked at check
text at all. That flag is now FALSE across the whole masked path. A host-attested artifact
that misstates its own basis is worse than one that says nothing: the reviewer cannot
discount evidence whose provenance it has been told wrongly, and "the reporting path reads
the deciding path" is the general fix — the same move that collapsed the projection/
comparison split in round 4 and the fallback chain in round 5.

Round 7 was that same class one kind over: `expected_whitespace_normalized` also read true
for `artifact_paths`, whose observed set is compared BYTE-FOR-BYTE (every whitespace byte of
a filename counts), so the fix for one kind had simply never been asked of the others. The
durable form is to answer the question ONCE PER KIND, next to the kinds: `IDENTITY_KINDS` is
the closed, ordered table of reconciliation identity kinds, each row carrying its name, how
to read its value off a `ReceiptIdentity`, and whether that identity is canonical command
text; `ReceiptIdentity.key` iterates the table and `KIND_NORMALIZES_COMMAND_TEXT` is the
total lookup the flag performs — true for `check`, false for `criterion_id`,
`artifact_paths` and `none`, with a `KeyError` rather than a default for a kind that skipped
the table. The general rule: when a disclosure describes a property of a closed set of
kinds, put the property IN the set, so a new kind cannot be added without answering, and a
per-kind fix cannot be mistaken for a fix of the class. Every component that
participates carries into BOTH fixed reviewer projections
(`verification_receipt_ledger_row`, `_accept_verification_summary`), and it does so
through ONE shared renderer — `_outcome_receipts.receipt_identity_projection` — rather
than two independently maintained key lists, so the two surfaces cannot drift apart
about the identity the reconciliation used. Three rules make that claim literal rather
than aspirational. First, a projection that BOUNDS a list discloses the bounding, and it
does so through ONE shared helper — `_outcome_receipts.disclosed_list_projection` — not a
hand-rolled `[:N]` per call site: every bounded list on a cognitive-review surface emits
`<key>_omitted` (the exact count, 0 included), bounds each string through the SSOT
`utils.truncate_review_artifact` so a clipped value carries its own omission note, and
adds a hash of the FULL set (`paths_identity_sha256` for the path identity `_reconciles`
compares; `urls_identity_sha256` for the native-retrieval URL set) wherever the complete
evidence is not reachable from the store the bounded row lives in — the receipt lists need
no hash because the whole receipt is durable in the per-task `verification_receipts.jsonl`.
Bounding a set is allowed; hiding that you bounded it is the P1 violation, and a review
round that finds one surviving `[:N]` on such a surface should sweep the phase for the
rest rather than patch the instance. Second, "is anything still outstanding?" is a
question about a SET of identities, so it is answered by an outstanding SET
(`unreconciled_failed` / `unreconciled_masked`, each candidate scanned against ALL later
reconcilers) and never by a single latest-POINTER, which a newer candidate silently
overwrites: fail A, fail B, pass B once reported no red at all, and masked c1, masked c2,
clean c2 lost c1 the same way. The `latest_*` helpers are projections of that set, and the
acceptance summary carries its SIZE (`unreconciled_red_count`,
`check_exit_masking_unreconciled_count`) so a second outstanding item cannot hide behind a
flag that reads as if it described exactly one. Third, the acceptance summary projects the
identity of the UNRECONCILED RED (`unreconciled_red_identity`) and not only of the latest
receipt: a later green of a DIFFERENT verification leaves an earlier red standing, so a
reviewer shown `unreconciled_red=true` beside a green `latest_*` would see a flag whose
cause is nowhere in the packet. A flag without its cause is not reconstructible. Fourth —
the rule the first three kept re-learning instance by instance — there is exactly ONE
canonical identity derivation (`_outcome_receipts.receipt_canonical_identity`, built on
the shared `shell_parse.canonical_command_text` seam and `canonical_path_set`), and
comparison, hashing, counting and projection all read THAT object. A phase that carries a
normalized/set-shaped identity for comparison and a raw/ordered one for display will keep
producing findings where the two disagree: a lossy comparison form (`" ".join(x.split())`
collapses whitespace inside quoted arguments, so two checks asserting different things
compare equal and a green closes an unrelated red), rows counted where identities were
promised, and a hash describing a set the carried items do not. The derivation implies an
ORDER, and the order is always canonicalize the RAW values → render → bound: rendering
is lossy (redaction, truncation), so de-duplicating after it drops distinct values while
the omitted count still reports zero.

Two consequences of that fourth rule are worth writing down rather than rediscovering.
**The receipt store's `check` rendering changed in v6.78.0**: `tools/verify.py` now writes
`shlex.join(argv)` where it wrote `" ".join(argv)`. A space-join is not injective — argv
`["echo","a b"]` and `["echo","a","b"]` rendered to the same text, and since that text IS
the verification's identity downstream, a green on one could clear a red on the other.
`shlex.join` is the exact inverse of the lexer `canonical_command_text` re-tokenizes with,
so the stored text round-trips back to the argv that ran — which only holds because that
lexer now preserves QUOTING: `shlex` strips quotes before yielding tokens, so `echo '&&' x`
and `echo && x` arrived identical, and the canonicalizer's leading/trailing separator strip
then dropped a literal final `&&` argument as if it were syntax. Quoted and escaped
punctuation is marked on the way into the lexer and the mark is read back off the tokens,
so a literal argument that merely spells like an operator can never be mistaken for one;
nothing is stripped afterwards. Path values are likewise left byte-for-byte alone — a
leading or trailing space is a legal filename byte, and trimming it let observations of two
different files reconcile. The rule behind all three: a normalization that discards
information the identity depends on is not a normalization.

**Changing a stored rendering means versioning it.** The cross-version cost of that switch
was first written down here as safe in ONE direction — an old receipt carries `echo a b`
where a new one for the same argv carries `echo 'a b'`, so they fail to reconcile, and a
non-reconciling red STAYS RED. True, and not the whole picture: an old red and a new green
from DIFFERENT argvs can render IDENTICALLY (`["echo","a b"]` and `["echo","a","b"]` were
both space-joined to `echo a b`), and there the new green cleared the old red — a false
green produced by the change made to remove one. Reasoning about a format migration one
direction at a time is how that got missed; ask both, always. The root was that the receipt
did not record WHICH renderer produced its text, so the comparator could not tell the two
formats apart. It records it now: `check_rendering` is stamped beside the text by every
writer that renders one (`shlex_join` for a rendered argv, `declared_text` for the agent's
own verbatim text), an absent stamp reads `unversioned`, and the check identity is the
RENDERING PAIRED WITH THE TEXT. Receipts from different renderings are therefore never the
same verification — an `unversioned` receipt is not known-equal to a versioned one, it is
UNKNOWN, and unknown must not clear a red. Two `unversioned` receipts still match each
other, which is both the behaviour they had before the upgrade and the most that can be
recovered from strings already on disk. An unrecognised future rendering is automatically
its own namespace, so the next renderer change is safe without a code change — but it must
still take a new stamp value, and a writer that stores a `check` without one is a bug (a
test walks `verify.py`'s receipt writers and asserts every check-writing one stamps).
The direction is now honestly stated: cross-version reconciliation is strictly LESS likely,
so an upgrade may leave standing a red that was really fixed. A false red costs a human a
second look; a false green costs the thing this whole surface exists for. The other two
identities were asked the same question and neither has the hole: the MASKED path keys on
the agent-authored `criterion_id`, which no writer ever re-rendered, and the observed
`paths` set is stored RAW and canonicalized by the READER, so both eras are compared on
today's terms — the phase's path change (dropping `.strip()`) was a comparator change, not
a stored-format one. **And one KNOWN, deliberately
deferred limit:** `tools/verify.py` bounds an `artifact_observation` receipt's observed
path set at twenty (`paths[:20]`) with no omission count, so two observations differing
only past the twentieth path are indistinguishable IN THE STORE. Unlike every bound above
it, this one is on what gets written DURABLY, not on what gets projected — there is no
complete set behind it for a downstream projection to recover, and `paths_identity_sha256`
can only ever hash what the writer wrote. Fixing it means changing what the durable store
holds and deserves its own scope. It is recorded here so it is a known limit rather than a
silent one. It is ADVISORY only — it shapes a nudge and a
disclosed reviewer flag (`expected_whitespace_normalized`), never a gate — so a
mismatch costs at most one advisory nudge and can never lose a record. For semantic matching of fuzzy
entities, use the LLM-first pattern (`semantic_dedup`: exact fingerprint as a
cheap first pass, an LLM as the authority, fail-open) — never string equality.
The enforcement shape is a REQUIRED typed argument at the consuming seam
(`bind_task_to_project(..., *, origin)`: a valid ref or a closed-enum absence
reason; omission raises), so a future call site cannot silently skip the
invariant.

### Mutable external-fact inventory

This table is a maintenance inventory, not a second runtime authority. External
facts change independently of Ouroboros releases; prefer live metadata or a
bounded probe where that can answer the exact question, and otherwise keep the
current conservative behavior visible. v6.67.0 documents these facts but does
not migrate their runtime representations.

| Location | Fact | Mutability | Current authority | Live/probe option | Risk | Recommendation |
|----------|------|------------|-------------------|-------------------|------|----------------|
| `ouroboros/provider_models.py::_VISION_MODEL_PREFIXES` / `_VISION_OVERLAY` | Which model families accept native image input | High as model families and route capabilities change | Conservative shipped prefixes, overridden by parsed OpenRouter `/models` `architecture.input_modalities` for exact model ids | Exact provider metadata when available; otherwise a bounded image-input capability probe | A stale positive sends unsupported image blocks; a stale negative needlessly captions them | Keep the conservative fallback and exact-model overlay; consider broader provider metadata only in a separately reviewed migration |
| `ouroboros/llm.py::supports_message_cache_control` and `_reasoning_signature_portable_across_or_providers` | Which families support message cache controls and portable replayed reasoning | Medium/high as provider routing contracts change | Explicit family rules backed by provider behavior and dated live probes | Provider documentation plus a same-model cross-provider replay probe | A false positive can invalidate a request; a false negative loses cache or reasoning continuity | Retain the small explicit rules and re-probe when provider behavior changes; do not generalize by model-name resemblance |
| `ouroboros/provider_models.py::_ANTHROPIC_MODEL_ALIASES` / `migrate_model_value` | Direct-provider id spelling compatibility | Medium as providers rename ids and prefixes | Shipped compatibility mapping and current direct-provider id contract | Exact provider catalog/documentation can confirm a current id, but cannot establish whether a saved spelling was intentional | Removing an alias breaks upgrades; guessing aliases can silently reroute | Keep explicit compatibility aliases until a separately documented retirement window closes |
| `ouroboros/server_runtime.py::_RETIRED_MODEL_DEFAULT_REPLACEMENTS` and scope prior/legacy defaults | Which formerly shipped defaults are upgraded automatically | Release-dependent | Release history plus current `SETTINGS_DEFAULTS`; only known former defaults are migrated | A live catalog can show availability, but cannot infer user intent or whether a saved value was a default | Over-broad migration overwrites an explicit owner choice | Keep release-scoped exact replacements and regression tests; review retirement separately |
| `ouroboros/pricing.py::get_pricing` and `ouroboros/llm.py::fetch_openrouter_pricing` / `fetch_cloudru_pricing` | Exact-route model tariffs | High; pricing and FX drift independently | Exact provider catalog with nullable unknowns; provider-settled usage wins | Bounded live catalog fetch and provider-reported settled cost | Static prices look authoritative after becoming wrong and can corrupt admission | Preserve the live nullable design and cover it by regression; do not restore runtime tariff tables |

### Provider Independence

One configured provider must be sufficient for the agent loop, commit review,
scope policy, safety, and context/memory flows. Core capability must not acquire
a hidden OpenRouter or second-provider dependency.

When adding or changing a provider, update one coherent route contract:

1. credential/readiness detection and exact model-id migration;
2. Main/Light/Fallback and reviewer-slot defaults without overwriting explicit
   owner choices;
3. tool, reasoning, image, and cache-control translation at `llm.py`;
4. nullable pricing/settlement and truthful capability omissions;
5. review and scope routing, including sourced context-window evidence;
6. direct-provider and single-provider regression tests.

Local-only installs keep their local route. Unreachable shipped remote defaults
may be cleared, but explicit owner values are not. Scope authority follows the
BIBLE P3 policy: in owner-selected Max it requires the applicable sourced window
evidence; owner-selected Low records the declared skip rather than pretending a
partial review occurred. Current model ids and defaults belong in code/config,
not in this handbook.

The `-pro` suffix is an OpenRouter routing slug, not an official OpenAI model id.
Until a Responses-API lane exists, a direct OpenAI slot uses the plain Sol id;
projecting the slug into chat completions would turn an owner route choice into
a guaranteed 404. This is a compatibility constraint, not a mutable capability
table.

Provider-specific optional features may be unavailable on another single
provider, but the core loop must degrade explicitly rather than crash or silently
reroute.

## Module Size & Complexity

P7 makes context fit a maintenance constraint, not a line-count aesthetic.

- Python modules everywhere (including `tests/` and `devtools/`) and first-party
  `web/**/*.js` modules (including `web/tests/`) target roughly 1000 lines. The
  deterministic hard gate is 1600 lines for exact repo-relative paths not listed
  in `ouroboros/size_ratchet_manifest.py::GIANT_PATHS`; stale or newly oversized
  entries fail. Vendored/minified assets are excluded. The same production
  iterator drives smoke, health, census, and the 200,000-byte ratchet. Sources
  decode as strict UTF-8 and normalize line endings to canonical POSIX LF before
  line and UTF-8-byte counts, so checkout policy cannot change the inventory.
- The exact-current 1001-1500-line band lives in `BAND_PATHS`. A new or
  re-entered path requires a nonblank rationale. `BYTE_DEBT` stores exact counts
  above 200,000 UTF-8 bytes and is shrink-only; regenerate both with
  `scripts/regenerate_size_ratchet.py`.
- Every non-grandfathered Python function or method fails the deterministic gate
  above 300 lines; exceptions live in
  exact `(repo-relative path, lexical qualname)` keys in
  `ouroboros/size_ratchet_manifest.py::FUNCTION_DEBT`. Methods above 150
  lines are a decomposition signal. JavaScript currently has only the module
  line-count gate.
- Runtime Python function/method count is checked against
  `ouroboros/review.py::MAX_TOTAL_FUNCTIONS`; the function iterator preserves
  the pre-v7 runtime scope (tests/devtools excluded) while module gates include
  those trees.
- More than eight parameters is a decomposition signal applied by BIBLE and
  reviewer checklist 2(c), not a deterministic size-test gate. Existing
  baseline debt is not retroactively a failing tree. Any advisory ratchet must
  publish its AST counting scope and bind its baseline to the final SHA.
- Prefer deleting dead/duplicate authority before raising a cap. Add an
  abstraction only when it removes concrete coupling or preserves a stable
  extension seam.

### Pragmatic SOLID

SOLID is a direction for making changes legible to future agents, not a demand
for classes or extra framework surface:

- **SRP — Single Responsibility Principle:** keep one coherent reason and one
  clear authority for a unit to change.
- **OCP — Open/Closed Principle:** extend an existing stable seam when it
  preserves the contract instead of rewriting unrelated callers.
- **LSP — Liskov Substitution Principle:** an implementation or backend must
  preserve the caller-visible behavior of the contract it implements.
- **ISP — Interface Segregation Principle:** consumers should depend only on
  the capabilities they actually use, not a broad convenience interface.
- **DIP — Dependency Inversion Principle:** policy should depend on small,
  host-owned contracts rather than provider-specific or concrete details.

Apply these principles pragmatically. They do not require a class hierarchy,
DI container, numeric score, AST analyzer, or a new review pass. A SOLID or
minimalism finding must name the exact symbol or authority, the concrete
duplication or coupling, and a smaller alternative that still satisfies the
contract. Diff size, line count, and file count alone are not findings.

### Invariant: Projection over replay (hot readers of growing stores)

A reader that runs per INTERACTION — an HTTP request, a WS/SSE message, a poll
tick, a task turn — must not replay a growing store to produce its answer.
Interactive read cost must be O(response), achieved through a maintained
projection, a cursor, rotation, or a bounded tail — never a full-history scan
filtered down to the answer.

- **Per interaction is the unit.** Work that runs once per boot or per explicit
  owner action may scan history; work on a request/message/poll-tick/task-turn
  path may not. A scan that is cheap today is not the point — every growing
  store crosses the threshold eventually, and the reader degrades exactly when
  the system is most used.
- **Storage-agnostic.** JSONL logs, SQLite tables, JSON snapshots — any store:
  a full-table read filtered in code IS a replay (a `SELECT *` narrowed in
  Python is the same failure as parsing a whole JSONL file for its tail). This
  includes unbounded collections INSIDE snapshot/state files: a "snapshot" that
  accretes an unbounded list re-reads its entire history on every load.
- **Passive GET.** Read handlers perform no NEW steady-state durable writes.
  Exactly two named exceptions exist: (1) substrate-owned integrity repair
  performed under the substrate's own lock (the usage-ledger torn-tail
  quarantine in `ouroboros/usage_ledger.py`), and (2) one-time idempotent
  migrations guarded by a durable watermark (the legacy usage import). Anything
  else that "just materializes a bit of state" on a GET is a mutation hiding on
  a read path.
- **House precedents — reuse these shapes instead of inventing new ones:**
  chat log rotation with archive-aware readers
  (`supervisor/state.py::rotate_chat_log_if_needed`); the compact
  `containment_faults.jsonl` projection maintained beside an unbounded event
  log (`ouroboros/delegate_custody.py`); dialogue-block consolidation — the P1
  "infinite horizon, variable granularity" reader; the passive-GET contract
  of `gateway/control.py::api_update_status`; and the fingerprint-keyed render
  cache inside the usage-ledger rows memo (implemented in
  `ouroboros/_usage_rows_memo.py`, consumed through `usage_accounting`) —
  a projection cached while its input is unchanged, invalidated only by
  advance/refold, never by TTL.

Enforcement: Repo Commit Checklist item 24 (advisory) triggers on diffs that
add or change an endpoint/poller/subscription/timer or read a growing store;
the hot-store growth health invariant
(`agent_startup_checks.py::hot_store_growth_notes`, surfaced by
`context.py::build_health_invariants`, thresholds justified in
`ouroboros/context_budget.py`) is the deterministic runtime tripwire. A change
that introduces a new append-only store read on an interactive path must
enroll that store in the `ouroboros/context_budget.py` threshold table (with a
justified constant) in the same commit — an unenrolled hot store is invisible
to the tripwire.

### Invariant: UI resources carry a disposer

Every long-lived acquisition in `web/` returns or records a disposer, and a UI
instance owns a `destroy()` that releases everything the instance acquired.
The resource kinds this covers:

- WS subscriptions (`ws.on(...)`)
- `document`/`window` event listeners
- observers (`ResizeObserver`, `MutationObserver`, `IntersectionObserver`)
- timers (`setInterval`, long-lived `setTimeout` chains)
- `requestAnimationFrame` loops
- `EventSource` / streaming connections

An instance that can be closed, hidden, or replaced (project chat panels are
the canonical case) must be destroyable without leaving any acquisition
behind; "hide the DOM node, keep the handlers" is the leak shape this
invariant forbids. Late async continuations check a `destroyed` flag before
touching state or re-arming loops.

Enforcement (honest disclosure): the deterministic leak test runs in the
release-tier `ui_browser` lane, not at commit tier; commit-tier coverage is
the advisory Repo Commit Checklist item 24. The class is closed
deterministically for the instrumented surfaces and advisorily for future
ones.

---

## Core Governance Artifacts

`BIBLE.md`, `docs/ARCHITECTURE.md`, and `docs/DEVELOPMENT.md` are **core governance artifacts**.
They are the constitutional, architectural, and procedural ground truth of the system.

### Invariant: Full availability in reasoning flows

Any flow that requires architectural, constitutional, or procedural reasoning MUST include
these artifacts as **first-class context sections** — not as optional or opportunistic
inclusions via touched-file packs.

Concrete requirements:

| Flow | BIBLE.md | ARCHITECTURE.md | DEVELOPMENT.md |
|------|----------|-----------------|----------------|
| Main task context (`context.py`) | full tier-0 | full in Max for every task class; lossless navigation map in Low | mode-independent: full when the active binding targets Ouroboros's system repo, including evolution/self-body work and a project-room turn without an external binding; visible on-demand pointer for a bound external workspace, subagent, or API/CLI/scheduled external surface. `workspace="none"` and explicit self-body overrides retain full Development. |
| Triad review (`tools/review.py`) | ✅ via preamble | ✅ via `load_governance_doc` | ✅ via `load_governance_doc` |
| ↳ Anti-thrashing (v4.35.1) | — | — | Open obligations loaded from `review_state` via `load_state(drive_root)` + `make_repo_key(repo_dir)`, injected unconditionally into `_build_review_history_section` prompt context. Same mechanism in `scope_review.py::_build_scope_prompt` (best-effort when `drive_root` available). |
| Background consciousness (`consciousness.py`) | ✅ full | ✅ full (max) / navigation map (low) | — (not yet required) |
| Advisory pre-review (`tools/claude_advisory_review.py`) | ✅ via `load_governance_doc` | ✅ via `load_governance_doc` | ✅ via `load_governance_doc` |
| Scope review (`tools/scope_review.py`) | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting | full canonical doc + Atlas accounting |
| Plan review (`tools/plan_review.py`) | ✅ full (every plan class) | full for `plan_class=self_mod`; lossless **navigation map** (sections + line ranges, full sections on demand) for external/creative/research plans (v6.61.0, owner-approved governance change) | ✅ full (every plan class) |
| Deep self-review (`deep_self_review.py`) | full canonical doc + Atlas accounting | full (max) / navigation map (low) + Atlas accounting | full canonical doc + Atlas accounting |

Plan review always keeps BIBLE.md, DEVELOPMENT.md, the proposed plan,
touched-file snapshots, and reviewer-slot framing as first-class context.
ARCHITECTURE.md is CLASS-TIERED (v6.61.0, an owner-approved governance
evolution — quiz 19): the agent declares `plan_class`
(`self_mod | external | creative | research`), and the host STRUCTURALLY
escalates to `self_mod` whenever `files_to_touch` resolve under the system repo
(a path fact, never keyword matching). `self_mod` plans keep the full inline
ARCHITECTURE.md — unchanged from the historical contract. Non-self_mod plans
(an external codebase, a creative deliverable, a research question) receive
ARCHITECTURE.md as the LOSSLESS navigation map (`context_layout.
generate_doc_nav_map`: every H2-H4 heading + its inclusive complete-subtree
line range, with parent rows intentionally overlapping descendants and
`max_lines=B-A+1`; full sections remain readable on demand) — their reviewers
judge the plan against its own domain, not ~45K
tokens of self-body detail. Rationale: the full-pack requirement existed to
protect SELF-modification reasoning; for non-self plans it actively hurt
review quality (reviewers anchored on runtime internals irrelevant to the
deliverable) while tripling cost. The agent must choose `context_level`
explicitly for `self_mod` plans; non-self_mod plans may omit it (defaults to
`minimal`). That field controls only the generated repository Atlas: `minimal`
omits Atlas accounting for bounded/local plans, while `localized`, `broad`, and
`constitutional` add progressively larger Atlas packs. A typed non-minimal
Atlas assembly failure or final quorum-fit failure rebuilds the same requested
fingerprint/scout wave once at loud `minimal`; compiler exceptions, monetary
budget refusal, and a minimal prompt that still cannot fit do not. Planning
scouts are likewise class-framed: `self_mod` scouts keep the repo-archaeology emphasis;
external/creative/research scouts are steered to the plan's own domain
(requirements, verification, sources, design) and never default to Ouroboros
internals.

Planning has two distinct roots. Governance documents are always loaded from
the system repository; planned snapshots and Atlas inventory always use
`active_repo_dir_for(ctx)`. A workspace/subject mismatch, an unavailable root,
or a `files_to_touch` path escaping that subject must fail loudly. Do not fall
back to reviewing the Ouroboros repo for an external plan. Read-only scouts use
the existing worker pool with its generic `executor=auto` route (selected
healthy harness first, existing loud native fallback) and persist full raw
handoffs. Wait for every launched
scout until it is terminal or the shared swarm ceiling is reached; give the
panel every ready non-empty handoff and an explicit reason for every omission.
Launch only one scout wave per exact plan fingerprint. A handoff is marked
consumed only after it was actually included in the reviewer request; a late
terminal handoff is audit-only and never reopens an already considered plan.
Canonical intent, task aliases, forensic refs, and omissions belong in one
shared evidence horizon—not copied corpora or a second planning engine.

The planning horizon must state the goal, mandatory invariants, scope
boundaries, non-goals, chosen existing extension seam, and explicitly rejected
expansions. Plan review publishes exactly `GREEN`, `REVIEW_REQUIRED`, or
`REVISE_PLAN`. `REVIEW_REQUIRED` findings are inputs: the main agent may accept,
reject, or defer any/all of them. Blocking closes the latest still-current,
reviewed, integrated, non-degraded result without a second LLM call through a
separate `plan_task` call containing `review_disposition` only: every finding
appears exactly once with evidence-based rationale, and each acceptance names
the matching plan revision. Never replay plan/goal/scope/files/context with the
disposition. Mixed calls and vacuous disposition-only calls fail before a new
attempt is recorded; exact replay is idempotent. Blocking `REVISE_PLAN` requires
changed plan text/fingerprint and another panel, while advisory may proceed only
under loud host disclosure and the main agent's rationale. Unknown, stale,
duplicate, contradictory, or incomplete dispositions fail closed. Reviewers
remain generative, but a finding must name
a concrete defect or a concrete smaller existing extension seam; never require
a fixed number of findings.

Force-plan is an LLM-first pre-implementation obligation on the admitted managed
root, not a mechanical permission check around implementation tools. The existing
`plan_review_state` owns durable review authority and
`config.get_review_enforcement()` owns blocking/advisory policy. Every submitted
envelope that reaches `plan_task` supersedes prior authority: invalid plan/goal/scope
input stores a domain-separated open attempt, while a valid envelope stores its
canonical fingerprint before repository/path validation. A newer attempt therefore
cannot fall back to an older GREEN. Immediately before first panel dispatch, the exact planning-scout
handoff component is frozen in a fingerprint-keyed host write-once continuity artifact;
the remaining live reviewer context is rebuilt. An unavailable reviewer
never becomes a disposition-able verdict; a repeat call reuses that handoff snapshot and
retries the panel, including after A→B→A. Blocking stays in
analysis and non-mutating preparation until closure or a real task-wide rail;
advisory may proceed by agent judgment with a host-owned disclosure, including
an explicitly rejected `REVISE_PLAN`. A planning
deadline skip records a typed rail attempt before returning so the reducer cannot
misread it as an absent `plan_task` call.
The short-lived Swarm router admits one new root and transfers the intent; it
never runs `plan_task`, steers an existing task, or publishes the work inline.

**Context mode (Low / Max).** `OUROBOROS_CONTEXT_MODE` controls the Architecture projection in the agent's own context: Max keeps `ARCHITECTURE.md` full for every task class, while Low supplies its lossless navigation map. `DEVELOPMENT.md` is mode-independent and follows the active repository binding. It is full when the task targets Ouroboros's system repository, including self-body and evolution work and a project room with no external binding; a bound external workspace, auto-provisioned external project tree, subagent, or API/CLI/scheduled external surface receives a visible on-demand pointer. Explicit structured overrides remain authoritative. Tier-0 identity and constitutional context stays full in every mode.
For ordinary Main calls, `context_fit.py` may render Max and Low from one
immutable captured core and apply exact family+route calibration. Unknown
routes try Max; there is no silent 200K assumption. Only a confirmed physical
overflow may retry the same model once with a task-local Low projection, with
forensic and owner-visible disclosure. This never changes the global context
mode and never applies to P3 commit/scope review.

### Invariant: Compaction must earn its rewrite

Emergency compaction separates necessity from utility. Total calibrated wire
pressure, including system/context blocks and tool schemas, decides whether
relief is needed; only the compactable transcript and its best reachable
post-pass floor decide whether a pass can help. When too few eligible spans
exist or that floor remains over the trigger, record durable disclosed
hysteresis instead of repeatedly paying a summarizer and rewriting the prompt.
The pass's own rewrite cannot satisfy the growth condition: genuine
compactable-region growth or the bounded round interval must rearm it. Preserve
the independent reactive provider-overflow retry. This prevents an unchanged
irreducible frame from destroying cache reuse while still allowing later useful
compaction.

### Invariant: No silent truncation

If a core governance artifact cannot fit in the available context budget:
- Do **not** silently omit it or truncate it without a visible marker.
- Either adjust the budget/flow to accommodate it, or emit an explicit warning
  (`⚠️ OMISSION NOTE: ARCHITECTURE.md omitted due to budget constraints`) so the
  operator and the model both know the context is incomplete.
- A reviewer or agent operating without ARCHITECTURE.md MUST NOT be treated as
  operating with full context — findings may be incomplete.
- Tools that return multi-model review findings (`commit_reviewed`, `skill_review`,
  scope/advisory review helpers) MUST be listed in
  `UNTRUNCATED_TOOL_RESULTS` or have an explicit per-tool limit; the default
  15KB transport cap is not acceptable for review verdicts.
- A reference-doc **navigation map** (H2-H4 inclusive complete-subtree ranges,
  with parent rows overlapping descendants and full sections one `read_file` away) and a
  named on-demand pointer are visible, lossless representations — NOT silent
  truncation. The low context mode uses these; it never applies `[:N]` to a doc.
- String bounding goes through the SSOT `utils.truncate_review_artifact`, never a
  hand-rolled `text[:cap] + marker`. Besides the marker, that helper carries an
  anti-waste FLOOR: a cut saving fewer characters than its own omission note is pure
  damage, so below it the text passes through whole. A local re-implementation loses
  the floor and can return a value LONGER than the input it "shortened" (a `…[+N
  chars]` marker is 11 characters, so any overflow under that grew the field).
  The two bounded-string primitives serve different contracts:
  `truncate_review_artifact` produces DISPLAY previews of review artifacts (its
  anti-waste floor may return the text whole), while `truncate_within_limit`
  enforces a STRICT wire/prompt bound — the omission marker lands INSIDE the
  limit and the result never exceeds it.
- Bounding a LIST is subject to the same rule as bounding a string: a `[:N]` slice
  must be accompanied by an explicit omitted COUNT, and — where the slice touches an
  identity that something downstream compares — a durable hash or reference for the
  full set (see `_outcome_receipts.receipt_identity_projection`).

### Invariant: Owner-facing surfaces show the full text (v6.70.0)

Disclosed truncation (the `⚠️ OMISSION NOTE` marker) exists to protect **LLM
context budgets** — it is a model-bound mechanism, not a licence to shorten
what the owner reads:

- **Owner/UI-bound surfaces** (chat panels, task_results projections, review
  verdicts shown to a person) present the COMPLETE text, or carry a reference
  to a durable full copy (e.g. an observability `response_ref`). Reviewer
  rationale is a cognitive artifact (BIBLE P1): projecting it truncated while
  the full copy sits unreferenced in private blobs is partial memory loss.
- **Model-bound projections** (review packs, context sections, tool-result
  transport) keep their disclosed-truncation budgets — those are real context
  economics.
- **A cut cheaper than its own marker is forbidden everywhere.** Truncation
  that saves fewer characters than the omission note it appends is pure
  damage; the shared primitive (`utils.truncate_review_artifact`) enforces
  this floor, and new truncation sites must reuse it rather than hand-rolling
  `[:N]` + marker. One named exception: tiny single-line identifier fields
  (limit < 100, e.g. a reflection backlog `kind`) keep a plain hard slice —
  a multi-line omission marker inside a one-line value is worse damage than
  the cut it discloses.

### Invariant: No "only if touched" gate for core artifacts

Core governance artifacts reach review/reasoning flows unconditionally — NOT only
when they appear in `touched_paths`. The `build_touched_file_pack` function is for
_changed_ files; core artifacts are a separate concern and are loaded independently.

### When adding a new reasoning flow

If you add a new flow that reasons about code structure, system architecture, or
engineering standards, you MUST:
1. Explicitly load `ARCHITECTURE.md` (and BIBLE.md if constitutional reasoning applies).
2. Log a warning if the file is missing or unavailable — do not silently skip.
3. Add a test asserting the file is present in the assembled context/prompt.

---

## Review & Commit Protocol

Reviewed commits separate cheap improvement evidence from authoritative
candidate-bound authority.

1. **Cheap advisory preflight.** After edits, `advisory_review` may find
   omissions before the expensive gate. Without an explicit skip,
   `commit_reviewed` requires fresh advisory coverage and no open advisory
   obligations or commit-readiness debt; any edit makes that coverage stale.
   `skip_advisory_review=True` bypasses only these advisory admission checks.
   Ouroboros chooses the skip by LLM judgment when the pass is slow, unhealthy,
   unavailable, or unlikely to add value, and states why in durable task/commit
   evidence. The skip itself and available gate reason are durably audited.
2. **Authoritative gate.** Fresh advisory coverage or an audited skip is
   followed by independently configured deterministic test policy, staged
   fingerprinting, triad review, applicable scope review, aggregation, and
   pre/post revalidation. The fingerprint binds `git write-tree`, ordered
   `HEAD`/`MERGE_HEAD` parents, indexed VERSION, expected `v{VERSION}` tag and
   existing target, plus the binary staged-diff hash. Advisory skip does not
   change those policies.
3. **Publication binding.** The created commit/tag is checked against the same
   tree, parents, VERSION, tag, and reviewed fingerprint before push. Any
   mutation, rebase, conflict resolution, or changed landing parent invalidates
   exact-candidate authority and requires the applicable final gate again.

Triad slots review the staged diff against `docs/CHECKLISTS.md`; duplicate model
ids remain independent slots and `config.adaptive_quorum` owns quorum. Scope
slots inspect touched context plus the repository Atlas. Required artifacts may
never disappear silently: the assembler reduces optional context and unchanged
diff context, records every degradation, and fails closed if its irreducible
pack cannot fit. Freely degradable touched snapshots move to diff-only first,
largest-first within that tier; an artifact owed in full is reached only after
the `-U0` rung and cannot buy fit by degrading into an invalid review.
Owner-selected Low records the distinct BIBLE P3 scope skip; other route or
assembly failure is not a clean verdict. An agent-session scope slot delivers by
retrieval: its verdict is authoritative once its window is sourced at ≥200K, and
"the host did not observe which files it read" is a provenance disclosure, never
a missing-authority finding.

The gate is one logical reviewer interaction per API slot. A same-route
transport or empty-response rail may make one bounded second physical send. A
hosted agent-session slot is one multistep execution; local extraction reuses
its collected transcript rather than launching another session. Actor transport,
parse status, semantic verdict, model and route, coverage, cost, and capability
delta remain distinct durable facts.

`docs/CHECKLISTS.md` is the only reviewer-question, severity, and output SSOT.
Architecture owns the dataflow; this section owns operator sequence. Finish all
edits, run focused tests, run the advisory when useful, then freeze and review
the exact candidate. Do not interleave edits with repeated review calls.

### External PR review is not commit authorization

The authoring agent freezes the final committed base-to-head range and gives it
to a separate agent context for read-only review. Any coding harness or provider
may supply that independent context; same-conversation self-review does not.
Unavailable review is recorded as `NOT_RUN`, never silently presented as clean.
`CONTRIBUTING.md` owns the public procedure and evidence fields.

`scripts/run_external_review.py --contributor` is an optional structured
producer for the same evidence. It preserves and freezes the machine's
configured `api_chat` and `agent_session` triad/scope rows, then binds each row
to its dispatched prompt receipt and observed response receipt. The shareable
packet records exact base/head/tree/diff hashes, route/model/profile facts,
terminal settlement and capability-delta facts, telemetry limitations, and
full redacted agent-session transcripts. Missing, tampered, drifted,
unprovable, or contradictory identity/terminal receipts make the packet
`INCOMPLETE`. Non-identity capability deltas remain explicit degradation
evidence and do not override the production actor-status/quorum result. A
proposal changing this review substrate still requires a trusted-target rerun.

This evidence establishes readiness; it does not authorize commit, push, merge,
or publication. Maintainers choose the landing parent and release version,
preserve authorship, and run the normal final exact-candidate gate. Accordingly,
a pull request into `ouroboros` leaves `VERSION`, `pyproject.toml`, the editable
root version in `uv.lock`, `web/package.json`,
`web/modules/api_types.js::GATEWAY_CONTRACT_VERSION`, the README badge and
latest Version History row, the named direct-download links in README and both
install pages, and the Architecture header byte-identical to its target. At
integration,
`ouroboros/tools/release_sync.py::sync_release_metadata()` projects the chosen
version and `version_carrier_desyncs()` verifies those carriers; changelog prose
remains a deliberate maintainer edit. The same projection owns the seven public
installer filename templates and rewrites the named direct-download links in
README, the source install page, and its generated Pages copy. Those links use
the immutable exact tag (`/releases/download/v{VERSION}/...`), not
`/releases/latest/download/...`: prereleases are excluded from GitHub's latest
release and a versioned latest-link would therefore fail during an RC. Release
notes are generated from the same templates only after the seven proof-bound
assets have been assembled.

The integration branch may therefore name installers that are not published
yet. Public onboarding does not use that branch: the default README and legacy
GitHub Pages source are `main` and `main:/docs`. Stable promotion advances
`main` only after the release and all seven installers are public; if promotion
does not happen, users stay on the previous working release.

Hermetic preflight uses a disposable worktree, temporary data/settings/pycache,
and scrubbed runtime/secret-class environment. Tests must rebind imported
process-global roots and fail closed on the live data root; setting only
`OUROBOROS_DATA_DIR` is insufficient. A reviewed local commit is the durability
boundary; an `origin` push and CI are follow-up signals, not prerequisites for
local self-modification survival.

### DEVELOPMENT.md Compliance Checklist

Before every commit, verify the following:

#### Naming Conventions
- [ ] Modules and variables use `snake_case`
- [ ] Classes use `PascalCase`
- [ ] Constants use `UPPER_SNAKE_CASE`
- [ ] Names are self-explanatory

#### Entity Type Rules
- [ ] **Gateway** (if present): contains ONLY transport. No business logic, no routing.
- [ ] **Tool** (`{verb}_{noun}`): thin LLM-callable wrapper. Validates input, formats output.

#### Module Size & Complexity
- [ ] Module stays near one context window (~1000 lines target; exact-path 1600 hard-gate debt is checked in, stale entries fail, and new/re-entered 1001-1500 paths carry a rationale)
- [ ] No non-grandfathered Python function or method exceeds the 300-line hard gate (`FUNCTION_DEBT` exact `(path, qualname)` keys are the exception SSOT); methods above 150 lines trigger decomposition review
- [ ] Total Python function count stays under the current smoke hard gate (consult `ouroboros/review.py::MAX_TOTAL_FUNCTIONS` for the active value; bump with a comment if a feature requires more headroom)
- [ ] More than eight parameters is a decomposition signal; consider a typed context object, but do not claim a hard gate or mark existing baseline debt noncompliant
- [ ] No gratuitous abstract layers (Bible P7)

#### Structural Rules
- [ ] New Tool? `get_tools()` exports it using the `ToolEntry` pattern from `registry.py`, an explicit entry is added to `ouroboros/safety.py::TOOL_POLICY` (`POLICY_SKIP` for trusted built-ins, `POLICY_CHECK` for opaque or outward-facing ones), and the intended capability class is declared in `ouroboros/tool_capabilities.py` (`CORE_TOOL_NAMES`, local-readonly/acting child profiles, parallel/truncation sets as appropriate). Ordinary top-level tasks share the registered built-in surface; add a tool to a child profile only when that narrower principal should receive it, and test schema plus execution behavior rather than mirroring names into another catalog. Without the policy entry the tool falls through to `DEFAULT_POLICY = POLICY_CHECK` and pays a light-model LLM call per invocation. **A tool that WRITES the repo working tree needs the GUARD surfaces too, not only the visibility ones:** add it to `_ROOT_ARG_REPO_WRITE_TOOLS` (the single set behind the acting-no-workspace fence, the protected-write gate and the acting root-enum narrowing) and make sure its target paths are canonicalized — via `_PATH_NORMALIZED_TOOLS` if it takes a top-level `path`, or via `canonical_repo_relative_path` + `_payload_write_paths` if its paths ride inside the payload. Visibility checks can all be green while these are missing, so tests must exercise the real guard chain, not only a mocked resolver.
- [ ] New Gateway (if extracted)? Contains no business logic, only transport.
- [ ] New memory/data files? Should they appear in LLM context (`context.py`)?

#### Skill Repair Task Constraints
- Skill repair tasks use structured `task_constraint.mode="skill_repair"`, not prompt markers.
- In repair mode, edit paths are payload-relative: `plugin.py` means the selected `data/skills/{external,clawhub,ouroboroshub}/<skill>/plugin.py`.
- Use `edit_text` for one exact replacement and `write_file` only for new files or intentional full rewrites with `root=skill_payload`. (`edit_batch`/`apply_patch` are repo-lane tools and do not accept `root=skill_payload`.)
- Finish repair with `skill_preflight` and `skill_review`; grants and enablement stay owner-controlled.
- Repair mode is a stricter UI lane, not the only path for skill authoring. In `runtime_mode=light`, ordinary chat tasks may edit explicit `data/skills/{external,clawhub,ouroboroshub}/<skill>/...` payloads via `write_file`/`edit_text` with `root=skill_payload`, `bucket`, and `skill_name`. Explicit repo/data paths keep their own address space and ignore stale short-form args. Core/repo paths, `data/skills/native/*`, `data/state/skills/*`, marketplace/provenance sidecars, and direct `run_command` writes to repo targets remain blocked.
- New path checks for skill edits must use `ouroboros.contracts.skill_payload_policy` rather than reimplementing bucket/path traversal logic in each tool.

#### Native-Risk Extension Dispatch
- `type: extension` skills with reviewed isolated dependency envs must not import `plugin.py` or execute handlers inside `server.py`, even when the dependency tree looks pure-Python. Payload-native marker files (`.so`, `.dylib`, `.dll`, `.pyd`) also force child dispatch as defense in depth, but opaque native payloads remain subject to the skill-review checklist and are not newly allowed by this runtime fallback.
- Keep the split explicit: no-dependency pure-Python extensions may use `extension_loader`'s in-process PluginAPI path; isolated-dep/native-marker extensions are cataloged and dispatched by `extension_process_runner` short-lived child processes.
- Tool, HTTP route, and WebSocket handler proxies must return normal tool errors / HTTP 502 / WS log messages on child crash, invalid JSON, timeout, or abort. A child `SIGABRT` is a handled extension failure, not a server crash.
- Child processes must use scrubbed env, per-skill grants, per-skill isolated deps, process-group tracking, output caps, and timeout cleanup. Do not add fallback code that imports native-risk plugin modules in the host process.

#### Task Contract Resource Policy
- When a task contract declares `resource_policy.protected_artifacts`, enforce it as a typed affordance policy in every runtime mode: execute-only black-box references may be run, but byte reads, copy/hash/static introspection, tracing, and debugging against declared paths are blocked.
- Observable Acceptance Claims are bounded, advisory, task-general criteria (`id`, `claim`, `surface`, `support`, `priority`). `success_criteria` is an input alias, not a second persisted carrier. `effective_acceptance_claims` is the only binder: ingress-contract claims win, otherwise the current closed plan wave's frozen claims apply at read time; neither path mutates the live contract. A child receives only claims explicitly passed to its own `schedule_subagent` call. Reviewer `evidence_refs` resolve by exact membership in the already-built host packet, without fuzzy matching, filesystem reads, or re-execution; a claim reference certifies clean only through a passing host-attested support row. Non-passing receipts, agent prose, expected-support text, and unattested references remain named but non-resolving. Resolution changes the clean bit and its disclosure, not actor parsing, quorum, or verdict. Do not turn claims into a hard acceptance gate or surface-specific taxonomy.

#### Devtools isolation

- `devtools/` is tracked operator code outside runtime package discovery and
  the runtime import graph. Runtime modules, `server.py`, web modules, and build
  scripts must not import it.
- Touched devtool files receive normal triad/scope review. Unrelated files may
  remain manifest-only in broad Atlas packs so operator code does not drown
  core review.
- Generated outputs live in an explicit external root, never in `repo/` or live
  `data/`. Domain-specific architecture, launch procedure, and methodology live
  beside the relevant devtool rather than in core governance docs.

#### Light Mode External Deliverables
- `runtime_mode=light` is a self-modification boundary, not an OS sandbox. User-visible deliverables are allowed when they are outside the Ouroboros repo/control-plane.
- Preferred flow: `task_drive` for scratch, `artifact_store` for canonical deliverables, and `user_files` for the owner's visible copy (for example `Desktop/report.html`). `write_file(root=user_files)` and declared process `outputs` must register/copy canonical task artifacts. Rewrites of the same user-visible source keep the previous canonical artifact in non-manifest history with last-5 retention; history is for recovery, not a second deliverable list.
- `run_command`/`run_script` `scratch=[...]` (v6.52.2) is a DISTINCT channel from `outputs=[...]`: it declares EPHEMERAL in-workspace verification files (a throwaway test the agent writes, runs, and deletes — e.g. an in-package test that must live in the repo to compile). Scratch is exempt from the undeclared-output guard, never registered as an artifact, confined to the cwd, honored for NEW files and (v6.56.0) for ADOPTED existing untracked in-cwd files — adoption records the file's sha at declaration time through the SSOT `artifacts.record_task_scratch`, so the patch exclusion applies only while the content still matches (tracked files, paths outside the cwd, and paths outside a git worktree stay blocked; a real edit can never hide behind a scratch declaration) — and excluded from the workspace patch via `.scratch_manifest.json` (`headless.write_workspace_patch_artifacts`). Re-declaring a manifest path is idempotent. The undeclared-output guard verifies candidates POST-exec by stat (exists + mtime ≥ start−slack), so a mere path MENTION (import strings, CLI flags, heredoc bodies) is not a write. Use `outputs` for deliverables, `scratch` for throwaway verification — never overload one for the other.
- `run_command`/`run_script`/`start_service` may use cwd under `active_workspace`, explicit `system_repo`, task-scoped `task_drive`, task-scoped `artifact_store`, and external `user_files` where the active profile permits it. Omitted cwd consistently selects `active_workspace`; a light direct task that needs writable scratch must therefore select `task_drive` explicitly. Long-running services in light must use an explicit external/task/artifact cwd. Declared service `outputs` are copied into the task artifact store when the service stops.
- `run_script` temporary files are created under the active workspace when the task is workspace/executor-backed, then removed after execution. Do not run workspace scripts from the system repo temp path; relative imports, generated files, and toolchain discovery must observe the same cwd the user requested.
- Declared process outputs may be files or directories. Directory outputs are copied to the canonical artifact store as a bounded manifest plus zip archive; hidden/control/credential-shaped files, excessive file counts, and excessive byte sizes fail closed instead of leaking through artifact registration.
- In external workspace mode, light-mode self-repo dirty checks snapshot the system repo, not the active workspace. Task-local git operations inside the external workspace are allowed when the task requires them; Ouroboros repo/data paths remain structurally protected, and workspace patch artifacts are captured against the preflight git base.
- Project-room promotion with no working folder and no `workspace="none"` opt-out idempotently provisions a standalone git repo through `ensure_project_workspace`, then runs the ordinary workspace admission checks. Never provision over a non-empty broken binding or an unreadable registry; those cases fail loudly. Binding affects tool profile, memory, lease, and preflight, not the Max-mode Architecture projection.
- Keep policy denials separate from execution failures: `user_files_path_blocked`, `cwd_blocked`, and `artifact_output_undeclared` are non-failure outcomes, while failure to register an explicitly declared output remains `artifact_output_error`.
- The DEFAULT (non-workspace) shell lane carries the SAME target-aware git policy in every runtime mode including light (Q4=A sandbox unwind): mutating git is blocked only when it targets the Ouroboros runtime (system repo / any data drive — bidirectional, casefold, symlink-resolved containment; `commit_reviewed` is the remedy for self-repo changes), read-only git works everywhere including at the system repo, `allowed_resources.network=false` still fences network git subcommands, and acting `self_worktree` children keep the strict no-commit policy. `git init`/`commit`/`push` in `~/projects`, `/tmp`, an attached project folder, or a host-minted coop tree is legitimate task work, not a violation.
- `claude_code_edit` is RETIRED (D10, owner-approved migration, phase 6.4): the SDK edit gateway's job moved to the delegated coding path — a mutating subagent (`schedule_subagent`) whose nanny drives the session with `delegate_start`/`delegate_wait`/`delegate_answer`/`delegate_cancel`, on the owner's subscription when a harness route is configured. Compatibility is one-way and permanent: a saved task contract carrying `disabled_tools=["claude_code_edit"]` also withholds the successor `delegate_start` (registry `_disabled_tools`), and the frozen `GET /api/claude-code/status` + `POST /api/claude-code/install` endpoints stay — the Claude runtime still powers the api-route advisory review. Do not resurrect the tool name.
- Do not recommend `runtime_data/uploads`, skill payloads, or owner state directories as generic artifact transport.

#### Runtime Cleanup / Retention
- All age-based garbage collection of disposable runtime artifacts shares ONE
  owner knob, `OUROBOROS_GC_RETENTION_DAYS` (default 7, hard max 365), and the
  cutoff/clamp math in `ouroboros/retention.py` (`age_cutoff`,
  `clamp_retention_days`, `get_gc_retention_days`). Do not hand-roll
  `now - days * 86400` or `max(1, min(days, 365))` in new prune code; reuse the
  helpers.
- The three former per-subsystem keys
  (`OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS`,
  `OUROBOROS_SERVICE_LOG_RETENTION_DAYS`,
  `OUROBOROS_HEADLESS_TASK_RETENTION_DAYS`) are deprecated and migrated into the
  unified key on settings load (`config.load_settings`). Do not reintroduce them.
  If a subsystem ever genuinely needs its own lifetime, name it
  `OUROBOROS_<SUBSYSTEM>_RETENTION_DAYS` and add it as a fallback in
  `retention.LEGACY_RETENTION_KEYS`, but prefer the unified knob.
- Prune functions keep an explicit `retention_days=` parameter for tests/special
  cases; only the default (None) resolution reads the owner knob. Startup prunes
  are wired from one place (`server.py`).
- Durable artifacts are NOT age-pruned and must stay out of the GC sweep: genesis
  projects (`OUROBOROS_SUBAGENT_PROJECTS_ROOT`) and forensic observability blobs
  (kept compressed indefinitely).
- Review continuations are recovery state, not disposable GC. At Review Continuity
  context build, move a record to `state/review_continuations/archived/` only when
  its owner task is settled, it has remained un-resumed for at least the configured
  seven-day threshold, and none of its recorded obligations remains open. Fresh or
  actionable records stay live; retirement is a collision-safe move, never delete,
  the archive has no runtime reader, and any uncertainty or move error leaves the
  live record intact. This removes closed history from later prompts without losing
  the recovery trail.

#### Live Subagent Task Constraints
- Live subagents are scheduled only through the existing `schedule_subagent` tool.
  Its public schema is strict: `objective` and `expected_output` are required;
  optional public fields are `role`, `context`, `constraints`, `memory_mode`,
  `model_lane`, `executor`, `deadline_at`, `acceptance_claims`, `write_surface`,
  `write_root`, `protected_paths_grant`, `external_tool_grants`,
  `delegation_intent`, `may_mutate`, `may_fan_out`, `max_children`, and
  `required_capabilities`. `schedule_subagent_properties()` owns both this schema
  and the handler's closed keyword set. `effort`, `parent_task_id`, and
  `description` are not public inputs: effort is dispatch-derived, lineage comes
  from `ToolContext`, and capability requirements are admission data rather than a
  frozen contract field. Claims must be plain strings; malformed shapes fail with
  a typed argument error, blanks normalize away, and omission means none rather
  than parent inheritance. Child builders re-state claims and the emptied
  `success_criteria` alias after the parent-contract spread. Boolean grants use
  strict parsing, deadlines can only narrow, and `_narrow_child_delegation_budget`
  can only reduce a subagent parent's authority; a root's explicit mutation grant
  still passes through the ordinary runtime checks.
- Live `memory_mode=shared` is disabled. Keep `forked` and `empty` as the only
  live subagent modes unless a later design adds sanitized shared-context v2.
- External `/api/tasks` and CLI requests must reject forged
  `delegation_role=subagent`; only `schedule_subagent` may create subagents.
- `task_constraint.mode="local_readonly_subagent"` must be enforced twice:
  schema discovery exposes only the local-readonly allowlist, and registry
  execution rejects forbidden calls even when invoked manually.
- Mutative ("acting") subagents (`task_constraint.mode="acting_subagent"`) are
  opt-in via `schedule_subagent(write_surface=...)` plus the master toggle
  `OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS` (explicit owner value applies to every
  surface; unset default is SURFACE-AWARE: every surface ON in advanced/pro,
  light allows `external_workspace`/`genesis` — they build outside the Ouroboros
  runtime — and keeps `self_worktree` OFF).
  `active_tool_profile` must fail closed: an invalid/missing surface, or a
  delegated subagent with a broken constraint, resolves to read-only — never to
  `self_modification`/`operator_control`. Acting children write only inside their
  surface (`self_worktree`/`external_workspace`/`genesis`) and keep commit,
  review, runtime control, tool-enable, skills lifecycle, and cognitive-memory
  writes blocked; `external_tool_grants` is deny-by-default for extension/MCP.
- `genesis` is a from-scratch deliverable surface: the supervisor provisions a
  fresh EMPTY git repo under the durable projects root
  (`OUROBOROS_SUBAGENT_PROJECTS_ROOT`, outside `repo/`/`data/`) via
  `subagent_worktrees.provision_genesis_project`. It is NOT the system repo, so
  protected-path discipline does not apply; it is durable (never GC-pruned and not
  in the worktree registry) because the project directory IS the deliverable. The
  parent does NOT `integrate_subagent_patch` a genesis project into the live body;
  the returned `workspace.patch` (diff from the empty seed commit) is only a record.
- `self_worktree` is a checkout of the system repo: keep protected-path write
  discipline AND protected shell-write guards active for it (no workspace bypass),
  permitting protected edits only in pro AND with `protected_paths_grant`. The
  worktree root must stay outside `repo/` and `data/` (guarded in
  `subagent_worktrees.provision_worktree`).
- `external_workspace` acting children write in the SAME active external workspace
  as the parent. `integrate_subagent_patch` verifies the child's declared
  write/root lineage and that reported files are present, then records a verdict;
  it must not re-apply the patch into the shared workspace because the edits are
  already there.
- The parent is the SOLE committer of the live body. Acting children return a
  `workspace.patch`; the parent applies a chosen patch with
  `integrate_subagent_patch` (manifest-first, sha256-verified, 3-way apply, writes
  a `subagent_patch_verdict` artifact, invalidates advisory) and then runs its own
  `commit_reviewed`. Routing is top-only — never integrate a descendant directly
  into the live repo; bubble patches up one parent at a time
  (`ctx.active_repo_dir()`).
- The supervisor (`_resolve_subagent_constraint`) is the authoritative gate that
  validates the toggle/surface and provisions/validates `self_worktree`;
  `server.py` startup calls `subagent_worktrees.prune_orphans()` (git has no
  worktree GC). Worktree mutations use a dedicated cross-process ops lock, not the
  drive-scoped repo git lock.
- `task_constraint` boolean parsing must be strict; strings such as `"false"`
  are false, never truthy through Python's `bool("false")`.
- The effective delegation budget is a pure admission reducer: declared
  `delegation_budget`, explicit `required_capabilities`, and unresolved
  structured non-advisory `delegation_constraint` rows are reconciled before a
  child runs. Scheduler back-pressure rows may be advisory telemetry (for
  example `queued_behind_active_cap`) and must not block later queued children
  below the hard ceiling.
  Do not infer child needs from objective prose; the LLM declares them via the
  closed enum. Do not add fields to `contracts/task_contract.py` for this.
- Treat a delegated 404 as scoped evidence. Project 404 discharges registration;
  run 404 after owned-daemon reprovisioning closes custody as unreachable, not
  settled, only after registration retirement and without invented usage/spend.
  For a shared project, the lowest run id owns retirement retries and siblings
  defer quietly until removal or a project 404 completes the obligation.
- `delegation_constraint` is a typed task-tree beacon with a structured payload
  (`constraint_id`, directive, scope, rationale). Consumers must read the payload,
  never parse the text. Overrides require an explicit reason and are recorded as
  decision rows.
- Subagent changes must keep writes, commits, review mutation, runtime control,
  tool expansion, skills lifecycle, and shell blocked — except bounded task-tree
  coordination via `tree_note`/`tree_read`, parent-only
  `override_delegation_constraint`, and bounded media projection such as
  `extract_video_frames` writing derived frames only under the task artifact store
  (`artifact_store/video_frames`) through a host-owned command shape (the permitted
  local coordination/projection paths; not arbitrary workspace or repo mutation).
  Nested readonly
  `schedule_subagent` recursion is allowed only within configured depth/cap
  limits; depth bounds nesting only and never rewrites a
  descendant's lane. Enabled/reviewed extension tools and enabled MCP tools may remain
  callable by owner policy, subject to inherited `task_contract.allowed_resources`
  such as no-network/no-web.
- A NEW `plan_task` scout wave is admitted before launch, and only a NEW one: worker capacity,
  the shared review-wave budget gate (`review_helpers.review_wave_budget_gate` — no second budget
  authority), and a consumable window. Each scout's deadline is bound to that window (the wave's
  shared cutoff minus the finalization grace and a margin, the reserve capped at a fraction of a
  short window) instead of inheriting the parent deadline verbatim, and a wave whose window has
  already closed is refused with a typed reason rather than started and then omitted. The
  recovery/collection path is NEVER gated: those handoffs are already paid for, so declining them
  would abandon spend. With `OUROBOROS_MAX_SUBAGENT_DEPTH=0` scouts are refused by the same
  delegation gate as any other child, and `plan_task` then completes on its existing
  `degraded_evidence` path — no wedge, no second wave.
- Runtime-internal scheduling knobs do NOT become `schedule_subagent` parameters.
  `control._schedule_task` is `(ctx, internal, /, **params)`: `params` is validated against
  `control.schedule_subagent_param_names()`, which is DERIVED from
  `control.schedule_subagent_properties()` — the single source the published JSON schema in
  `get_tools` is also built from (unknown keys get the strict v6 refusal). There is no separate
  mirror of the schema to keep in sync, and none may be reintroduced: a hand-maintained copy is
  correct only until one side gains a parameter, at which point the handler refuses something the
  model can see or accepts something it cannot (BIBLE P7). Internal-only options instead travel in
  the POSITIONAL-ONLY `internal` mapping keyed by `_INTERNAL_SCHEDULE_OPTIONS` — structurally
  unreachable from tool-call JSON, which is keyword-only. That is what keeps the handler inside the
  <8-parameter contract. Add an internal knob to that closed set, never to the signature and never
  to the public schema. The test of membership is WHO DECIDES, not who currently calls: an option
  the runtime decides belongs in the closed set, and an option the parent LLM is the right judge of
  belongs in the public schema. `deadline_at` was the set's only member until v6.87.7, when it moved
  to the schema on exactly that test — the parent is what knows when a child's handoff stops being
  useful, and a scout deadline was only ever runtime-internal because `plan_task` happened to be its
  first caller. The set is empty today; it stays because it is closed and an unknown key in it still
  fails loudly.
- `plan_task` planning scouts use the same live-subagent worker pool and one
  shared terminal-or-cutoff wait boundary. Poll in
  `OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC` slices, but wait for every started scout
  until it becomes terminal or the existing
  `OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC` ceiling. At that
  boundary, send every ready non-empty handoff to the reviewer and include every
  omission with its precise terminal/wait reason; missing evidence must never be
  silently presented as complete. Capacity, scheduling failure, or a normal
  cutoff does not trigger an extra inline model call: the omissions manifest goes
  directly to the configured reviewer panel. Repeated calls with the same plan fingerprint
  reuse the existing durable `plan_review_state` wave and never schedule a second wave, including
  when the first wave ended without a usable handoff. Only reviewer-included
  handoffs become consumed. Late terminal results are retained as audit evidence
  with `affects_review=false` and do not reopen the plan. If an included child
  changes after its exact snapshot enters the reviewer prompt, keep the old hash
  non-authoritative, persist the review once with a bounded stale-binding warning,
  and treat the newer child result as audit-only rather than paying for replay.
- `read_file(root=runtime_data)` and `list_files(root=runtime_data)` secret/control-file denials are subagent-scoped.
- Browser isolation for local-readonly/acting subagents (DNS fail-closed): block
  non-HTTP(S) schemes, private/link-local/reserved/unspecified and numeric-obfuscated
  literal IPs, unresolved hostnames, and hostnames resolving to any blocked IP — before
  goto, after redirects, and in route handlers. Loopback HTTP(S) is ALLOWED EXCEPT the
  Ouroboros control-plane ports (agent API / local-model / host-service, the configured
  `LOCAL_MODEL_PORT`, and the actual bound `state/server_port`); `file://` is ALLOWED
  only under the task's explicit `workspace_root` (symlink/traversal-safe), denied
  otherwise. `evaluate` JS stays unavailable to subagents; `vlm_query` /
  `analyze_screenshot` are available. (Relaxed in v6.24.0 for local UI/build inspection;
  control-plane, private-range, and DNS-rebind denial preserved. See ARCHITECTURE.md.)
- Effective task status belongs in `ouroboros/task_status.py`. Do not duplicate
  child-drive merge or terminality in gateways/tools. Task waits use
  `SETTLED_STATUSES`. Cancel INTENT is never a status value (Poltergeist phase A):
  every cancel ingress — tool, HTTP single/cascade, evolution stop (pending AND
  running evolution tasks; never an in-place queue prune), project
  delete, cascade descendants, boot migration — writes a durable intent through
  `ouroboros/cancel_intents.request_cancel` (nothing is minted for an
  already-settled task WITH NO LIVE OWNERSHIP — a settled RESULT does not mean
  a dead WORKER (GR6-1): the terminal result is persisted before post-task
  cognition ends, so every ingress checks live physical ownership
  (`supervisor.queue.task_has_live_ownership` in-process; the queue-snapshot
  twin `task_status.task_has_live_queue_ownership` worker-side) and passes
  `allow_settled_target` while a RUNNING row / busy worker remains, letting
  custody kill the still-spending process while completion-wins preserves the
  stored result; the cascade-coordination shape does the same for a settled
  root with live descendants: `scope=cascade` with `allow_settled_target` —
  the watchdog's replay trigger for the subtree, settled only by the cascade's
  no-live postcondition; the recorded scope is WIDEN-ONLY, cascade never
  narrows back to single) and FAILS CLOSED when that write fails: a cancel
  without a durable, watchdog-replayable intent is refused with a typed error,
  never run unfenced — an evolution-stop task whose intent write fails is KEPT
  and the stop reported INCOMPLETE.
  Timeout reaping is deliberately NOT a cancel ingress (owner decision: 1=A
  covers explicit cancellation; the reaper keeps its own custody protocol over
  the shared `reaping` slot marker and mints no intents). Effective reads
  project the intent as
  `cancel_state: "pending"` (plus `cancel_reason` when the intent carries one),
  the supervisor's `cancel_task_custody` is the one
  settle owner — it claims the intent BEFORE any custody mutation (a refused
  claim is `failed` with zero mutation, so racing custodies can never
  double-settle through the capture-miss lane), its claim is EXCLUSIVE while
  alive (a claimant the pid probe proves alive is NEVER abandoned by age; only
  a provably dead pid, or age-stale with liveness unknown, is recoverable),
  every intent mutation is
  fenced by the claim generation, so a second ingress cannot double-settle and a
  taken-over attempt cannot revert the new owner, and the claim is re-verified
  (pid + generation) immediately before the durable terminal write — a claim
  lost across the kill/join window aborts the publication (deliberately ONE
  re-read at the one write that matters, not a renewable-lease subsystem —
  rejected by owner scope); the secondary settle
  sites (pre-assignment pending drop, budget-drain `fail_tasks` — whose intent
  reads resolve at the CANONICAL supervisor root, never a child's
  `budget_drive_root`) hold the SAME
  claim/generation fence and yield to a live claim owner. A `scope=cascade`
  intent is settled EXCLUSIVELY by the cascade's no-live postcondition: every
  other settle site is refused atomically against the CURRENT durable scope
  (a mid-flight widen beats a stale claim snapshot; the refused claimant's
  claim is auto-released for the watchdog), and the postcondition always owes
  the tree's one summary BEFORE it settles — including the replay/already-down
  path — while re-judging stale sweep failures against the current durable
  status. Natural
  completion WINS a
  late cancel (a completed result is
  never overwritten or stripped — discarding is the parent's separate explicit
  `discard_child_result`). The owner's terminal answer is registered as OWED in
  the durable outbox BEFORE the intent settles (a crash between settle and send
  replays instead of losing both the watchdog trigger and the answer); a
  registration that could not be made durable leaves the intent OPEN on the
  cancel path and is a typed `terminal_delivery_unregistered` disclosure on the
  normal path — never a silent gap. On the natural path the owed row is
  registered immediately BEFORE the durable result write (projection-over-
  replay: a crash in the window leaves an owed row boot replay delivers; no
  boot scan over task_results).   The intent and delivery registries read STRICT:
  a corrupt projection refuses the mutation loudly instead of collapsing to
  `{}` and overwriting every active row — and strictness reaches ROWS, not
  just containers (GR6-3): a malformed pending/intent row or `delivered`
  entry refuses the mutation (bytes kept) and the enforcement reads
  (watchdog sweep, outbox replay) disclose loudly once, then quarantine the
  row. The unreconciled-delegated-runs disclosure is outcome-INDEPENDENT
  (GR6-5a) and rides completed/failed deliveries too; an EXISTING-but-
  unreadable custody log audits as the typed
  `delegated_run_state_unknown:custody_log_unreadable` marker, never as
  cleanly reconciled (GR6-4); and the cascade digest enumerates descendants
  by ancestry rooted at the cancelled node, so a mid-tree cascade lists its
  grandchildren and non-subagent descendants (GR6-2).
  `task_done` is validated through the DURABLE result
  UNCONDITIONALLY for every non-ephemeral event — a blank event status (the
  primary producer's shape) validates exactly like a settled claim, a settled
  claim over a non-settled row is the same lifecycle fault as a non-settled
  claim, and the copy-back exception path neither skips the validation nor
  synthesizes a `completed` row for a task that never wrote one (only
  `interrupted` keeps its restore-path exemption). The legacy
  `cancel_requested` status survives on a
  read-path only. `wait_task` and
  `get_task_result` keep the full handoff plus a bounded verification-receipt
  projection: every outstanding red/masked receipt first, then newest rows, with
  an exact omitted count; read the canonical store and fall back to the recorded
  child drive before copy-back. `wait_tasks` stays batch-compact:
  `task_id, status, cost_usd (+ its honest alias accounted_upper_bound_usd and
  cost_final — C2), child_result_sha256, outcome_axes, result,
  trace_summary, capability_delta when disclosable, duplicate_of`; it points to
  the hash-addressed full result rather than re-inlining trace/ledger forensics.
  Unknown ids are probed across result, queue, and tree-ledger authorities and
  return typed rows plus a bounded actual-child roster and its exact omitted count;
  an all-unminted set returns after the 30-second registration grace unless an id
  becomes real. `wait_task`, `wait_tasks`, and `delegate_wait` may disclose an
  expired cache horizon only from the latest recorded applied `5m`/`1h` TTL;
  absent, bare `default`, or unknown TTL evidence stays silent, and no surface
  predicts the next send's token rewrite.
- `forward_to_worker` may write only to validated running tasks whose lineage
  belongs to the current task/root, and must route forked/empty child subagents
  to the child-drive mailbox.
  Do not broaden generic data-tool behavior for normal tasks while fixing
  subagent isolation.
- The pre-final handoff reminder is a compact effective-status snapshot. Full
  untruncated child handoff belongs to `get_task_result` and `wait_task`
  (`wait_tasks` is a compact batch projection — see above). Do not add shared
  ledgers, automatic memory merges, or new settings/endpoints unless the
  accepted plan explicitly calls for them.
- A delegating parent must not produce a clean no-tool final answer while direct
  children are still running and undecided. One bounded absorption reminder is
  allowed; after that, finalization is best-effort (`children_unabsorbed`) rather
  than clean. This is an outcome-honesty rule, not a new wait loop.

#### Page Header Layout
- Top-level page chrome (`renderPageHeader`, tab strips, primary actions) must sit outside the scrolling content region.
- Pages use an outer flex column plus an inner `<page>-scroll` body with `overflow-y:auto`. Skills, Widgets, Settings, and Chat follow this pattern.
- Page icons come from `web/modules/page_icons.js`; do not paste divergent SVGs into individual page modules or the navigation rail.
- Primary page actions, including Refresh, live in the `renderPageHeader({ actionsHtml })` slot on the right. Do not add ad-hoc refresh rows inside scroll bodies.
- Non-chat top-level pages use `.app-page-glass` for the shared dim/brand backdrop. Header padding should stay compact; if a page needs more space, simplify its copy rather than growing the chrome.
- A new top-level page that scrolls its header together with content violates the architecture mirror: fix the layout, not the symptom.
- Top-level tab/pill buttons are a single design-system control: `renderTabStrip` + `.app-tab-strip` + `.app-tab` + the `--pill-*` CSS variables in `web/style.css`. Do not redeclare per-page tab padding, font size, border radius, or active styling in page CSS files.
- Scrollable page bodies use the shared `.scroll-fade-y` mask when content can pass under fixed page chrome. Do not copy/paste custom gradient masks into page modules; extend the shared class if the fade rhythm changes.
- Masonry-style widget packing uses `web/modules/masonry.js::applyMasonry`. Do not reintroduce CSS Grid row packing (`align-items: start`) for unequal-height widget cards; it leaves row gaps under shorter cards.
- Widget card ordering is a host UI preference. Persist it through `/api/ui/preferences` and `data/state/ui_preferences.json`; never rewrite extension manifests or widget declarations to store owner layout.
- New visual dimensions should become CSS variables first (`--pill-*`, `--button-*`, `--page-header-*`, etc.) and then be consumed by shared classes. Hardcoded page-local dimensions are review debt unless the component is genuinely unique.

#### Setup / Onboarding Layout
- The first-run wizard is a compact multi-step flow. At the default desktop
  window size it should not force scrolling merely because the access step has
  several provider fields; use responsive two-column field grids where width
  allows and keep step copy short.
- There is ONE wizard host: the `GET /onboarding` page, served by the gateway
  and loaded as an ES module from `/static`. The desktop setup window opens that
  URL after the managed server is healthy, and the blocking overlay frames it.
  Do not reintroduce a pre-server or inlined copy: a step that cannot call
  `/api/*` or import `web/modules/*` is the defect this host removed.
- Onboarding and Settings share the setup contract. If a key is typed in the
  current unsaved wizard payload, UI diagnostics must account for that in-memory
  value instead of warning from stale saved settings alone.
- Owner switches should expose the semantic choices the owner can actually make.
  For `OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS`, Settings presents Off / Auto / On
  (SC-4): Auto IS the unset, mode-derived state — it saves the empty value so
  the surface-aware runtime-mode default keeps deciding. Unset displays as the
  effective state where a binary label is truthful (advanced/pro = every surface
  on = "On") and as "Auto" in light, where the surface-aware default
  (`external_workspace`/`genesis` on, `self_worktree` off) makes both "On" and
  "Off" false claims. The v6.22.1-era rule that the empty runtime-default must
  not become a third owner-facing button was valid only while the unset default
  was binary per mode; the v6.91 surface-aware light default obsoleted it.
- One capability, one section. The whole subagent story lives in Agents →
  Delegation (`web/modules/subagents_settings.js`), beside Review lanes: the
  route (`OUROBOROS_SUBAGENT_HARNESS`), the write permission
  (`OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS`), and the two counts that bound it
  (`OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT`, `OUROBOROS_MAX_SUBAGENT_DEPTH`),
  with the two path roots behind a collapsed Advanced disclosure. They all answer
  "where and how far do subagents run"; `OUROBOROS_MAX_WORKERS` stays in Advanced
  because it sizes the process pool, not the agents. Never render a second control
  over the same settings key — two controls carry two drafts, and the last one
  collected wins, and a MOVE that leaves the old markup behind is exactly how a
  duplicate appears (`tests/test_agents_tab_static.py` pins each moved id to one
  occurrence).
  The delegated-run MODEL is the owner's default, authored here as the `=model`
  tail of the same key from engine discovery ("Engine default model" = empty
  tail); reasoning effort stays derived per call, and a hand-written `:effort`
  remainder rides through verbatim with no control over it.
- Onboarding completes in ONE transaction, and install-time defaults belong in
  it. `POST /api/onboarding/complete` persists settings, the next-boot runtime
  mode, the fresh-install safety default and the agent-subscription preset in a
  single write; `GET /api/onboarding` normalizes for display but must never
  persist, because a read that authors `settings.json` destroys the
  fresh-install latch both install-time behaviours depend on. There is no second
  completion path on any host — no generic-settings pair, no desktop save bridge
  — and the client treats only the exact success envelope as a completion: a 2xx
  whose body will not parse, or lacks `ok`/`runtime_mode`/`restart_required`, is
  a failure the wizard shows, never a silent success that discards the restart
  receipt. A failure after the bytes reach disk says so rather than claiming
  nothing was saved.
- The onboarding frame is sandboxed WITH popup permission
  (`allow-popups allow-popups-to-escape-sandbox`). The Agents step's primary
  action is the agent's own sign-in link in a new tab; a sandbox without those
  tokens blocks that click silently, and a popup that inherits the sandbox
  reaches the vendor's OAuth page with neither same-origin nor scripts. Assert
  this behaviourally, from the login card's own markup — an attribute-string
  test passes while the click stays broken. Install-time
  defaults are compiled from LIVE discovery and refuse typed when a model
  cannot be resolved — never guessed, never half-applied, and never
  re-derived after onboarding (that would be a second, continuous authority
  over settings the owner has since edited). Install time is a conjunction of
  three proofs — no recorded completion (`OUROBOROS_ONBOARDING_COMPLETED_AT`,
  written by every completion), no preset generation, no `settings.json` yet —
  because "no working provider" is a state an old install reaches too. A
  once-only decision is never taken on a moment-in-time reading: the daemon's
  `next_up` is quota-derived, so a subscription whose window is spent during
  onboarding must stay in the preset (D-3), and the seat is resolved from the
  durable facts — credential kind, enabled, present, verified.
- Owner settings writes go through `gateway/owner_settings.py`. The settings
  lock is a PRECONDITION of the write, not a hint: `_acquire_settings_lock`
  answers `None` on timeout and a writer that proceeds anyway is unlocked while
  claiming to be atomic. Once the bytes land, the response must say so — carry a
  `CommitBoundary` through the write and report a later failure as that step
  failing, never as a failed save (BIBLE P1). `saved` is a FIELD on both sides:
  pre-commit refusals answer through `unsaved_error`, or a client cannot tell
  "nothing was written" from an envelope that simply predates the field.
  `owner_write_guard` belongs only on an endpoint that calls
  `_owner_write_settings`; on any other it translates unraisable exceptions
  while advertising a lock the endpoint never takes.
- A setting only an ENDPOINT may author is disk-only in BOTH directions.
  `config.ENDPOINT_AUTHORED_SETTINGS` is consulted by the loader and by the
  environment projection, so the value is never read from `os.environ` and never
  exported back to it; the generic save's merge skip-list reads the same set.
  Blocking only the request body is not enough — an install-time fact that the
  environment can supply closes its own window before the endpoint runs.
- A control the owner cannot use is worse than none. With no agent subscription
  connected the Delegation section says so and points at Accounts in the same tab
  instead of rendering a delegation toggle whose every dispatch would silently
  fall back to an API child. Harness lists come from the accounts panel's own
  source (`accountRows` over `/api/claudexor/status`) — one catalog path, one
  login-capable discriminator.
- Owner-facing copy says "agent", never "coding agent" (D-10, owner verbatim:
  the same subscriptions build presentations and run arbitrary tasks). Product
  names — Claude Code, Codex, Cursor — are trademarks and stay as they are.

#### LLM Call Rules
- [ ] New LLM calls go through the shared `LLMClient` / `llm.py` layer — no ad-hoc HTTP clients or direct provider SDKs outside that layer. **Exception (v5.7.0+):** skill / extension `plugin.py` modules may call providers directly because they have not yet been migrated to a host-mediated `api.invoke_llm(...)` bridge. When that bridge lands, the exception goes away. Runtime callers (anything inside `ouroboros/`) must still use `LLMClient`.
- [ ] Every core-mediated physical provider send goes through `usage_accounting.execute_physical_attempt[_async]`: reserve, mark dispatched, then settle/unresolve. A transport retry is a new attempt. `llm_usage`, state, and UI counters are projections carrying attempt ids, never a second monetary authority. Provider tier pricing and any empirical tokenizer margin affect only a known reservation; settlement prefers actual provider usage/cost. Unknown price reserves `None`, remains nullable in usage events, and never blocks a model merely because its tariff is unavailable. An external skill with granted model-provider credentials is explicitly unknown/unmetered when it bypasses core transport—not `$0`; an ordinary spawned process must not be mislabeled as monetary work.
- [ ] Hold the usage-ledger cross-process lock only for budget check, validated append, and fsync. Never hold it over network I/O. Preserve a paid response if settlement persistence fails and leave an honest dispatched/unresolved bound.
- [ ] **Tree-spend visibility.** Under a root cap, pacing and stop text use root-subtree ledger spend including in-flight holds; own cost is diagnostic, and unavailable remains unknown rather than `$0`. Reuse `usage_accounting.last_root_accounting` and refresh only at rare cache-breaking/explicitly stale decision surfaces, never by an unconditional per-round ledger scan or inside a stable cached prefix. `task_pacing.resolve_cost_ceiling` returns `disabled|active|exhausted_soft_land|unknown` from the independent global-percentage and root-cap-minus-absolute-margin axes; graceful finalization precedes, but cannot bypass, the ledger fence. `resolve_deciding_spend` is the sole fallback seam and must label own-cost-under-root-cap as a lower bound.
- [ ] Before dispatching any post-task consolidation or synthesis worker, read `usage_breakdown` once for the whole root subtree and pass the same loop-local snapshot to summary and reflection. It is explicitly non-final (`cost_final=false`, `cost_with_children_partial=true`) and carries child-inclusive accounted cost, reservations, unresolved upper bound, unknown/unmetered count, ledger integrity, and capture time. A read failure is unavailable/null, never `$0`. Consolidation, summary, and reflection model spend belongs only to the existing terminal checkpoint; do not add another ledger or reconciliation LLM call.
- [ ] Runtime notices after the first user/assistant/tool turn are user notices, not new `role=system` messages. `LLMClient` defensively demotes non-leading system messages at the provider boundary; source call-sites should still append `[SYSTEM NOTICE]` user turns so provider payloads, local templates, and prompt authority stay consistent.
- [ ] Keep stable policy/governance first and dynamic evidence last. Prompt-cache support is deliberately narrow: direct OpenAI `prompt_cache_key`, OpenRouter `session_id` (or a caller-declared `cache_affinity` for surfaces whose rounds repeat with changing evidence, e.g. review), and one exact retry without the named parameter only when the provider explicitly rejects that parameter. Do not add provider hops, body rerouting, or a generic cache/retry framework.
- [ ] **Cache-friendliness invariant.** Keep byte-stable governance and task contracts before mutable evidence; never place timestamps, hashes, counters, or task identity in a stable cached prefix. Builders declare bare breakpoints and `review_substrate.assert_cache_breakpoint_cap` keeps the declared count at four or fewer. Only `LLMClient._normalize_payload_cache_ttl` finalizes the assembled wire payload: it supplies a missing tools marker where supported, legalizes TTL order, and discloses any reduction. The owner setting `OUROBOROS_PROMPT_CACHE_TTL=default|5m|1h` stamps existing Anthropic-family markers at that send boundary and never creates new ones; non-Anthropic wire stays unchanged. Preserve cache-affinity keys and exact review bindings.
- [ ] OpenRouter reasoning continuity belongs to OpenRouter conversations only. Direct/local payloads strip OpenRouter round-trip metadata; OpenRouter payloads with `reasoning_details` disable provider fallback to avoid endpoint-bound thought-signature corruption.
- [ ] Claude Agent SDK sessions (the api-route advisory since D10 retired the edit gateway — the edit path's system-prompt file handoff died with it) must preserve the full governance prompt; do not truncate BIBLE/ARCHITECTURE/DEVELOPMENT/CHECKLISTS to avoid argv or transport limits.
- [ ] Delegated (subscription-harness) work is accounted on its OWN ledger row:
  `usage_accounting.record_subscription_session`, which feeds the separate sessions/quota
  axis (`subscription_sessions` / `subscription_windows`). Its cash has THREE states and
  only the first is final: a DISCLOSED ZERO (`spend_usd=0.0`) settles at `cost_usd=0.0,
  cost_final=True` and leaves the projection final — the case this row kind exists for;
  an ESTIMATED amount (`spend_estimated=True`, the engine's `spendEstimated`) rides as
  money but never as finality; an UNDISCLOSED spend (`spend_usd=None`) is `cost_usd: None`
  and counted unknown/unmetered, never a confident `0.0`. Token counts are the same rule
  on the usage axis: `None` means no harness reported it, which is not a run that used
  zero. Do NOT reuse `record_unmetered_external_dispatch` for any of them — it also drops
  the sessions/quota axis. The nanny's own model calls remain ordinary metered attempts
  and keep rolling into the task projection; a subscription session is not counted as a
  physical provider call. D29: the APPLIED `credential_profile_id` and `access_profile`
  the engine's `authRoute` receipt / `effectiveAccess` disclosed ride the durable row (and
  the settled event) BY DEFAULT — empty when telemetry predates the receipt, never
  invented — so "which account paid, under which access" is answerable from the ledger
  row, the settled event, and (for reviewer slots) the last-execution file. Those are
  three separate stores, deliberately not joined into one applied receipt.
- [ ] `cost_final` on a projection is a COUNT of open rows (`non_final_rows`), never a
  truthiness test on a dollar sum: a reserved/dispatched/unresolved row, a settled row
  with an unknown price, and a settled row its writer marked non-final are each open
  however little they cost — `$0.00` is a real reservation for `provider="local"` and a
  real estimate for a delegated run. `non_final_rows` is returned beside the flag because
  a projection can be non-final with every dollar bucket at zero, and a flag without its
  cause is not reconstructible.
- [ ] A spent SUBSCRIPTION WINDOW is `subscription_window_exhausted`, a TRANSIENT class
  carrying `reset_at`, scheduled against that instant rather than through the
  60-second-capped exponential backoff. Do not fold it into `quota_exhausted`, which is
  classified permanent — correctly so for a billing refusal (402, no credits) and wrong
  for a window whose only cure is waiting.
- [ ] Provider failures must be classified before retrying the same request.
  Quota/auth/billing, hard bad-request, and request-too-large/context failures
  are non-retryable as-is: record the exact category and surface a recovery hint
  instead of burning rounds on identical calls. Transient rate limits/timeouts may
  still use the normal retry path.

#### Timeout & Wait Control
- [ ] For cognitive/long-horizon work (subagent waits and review),
  prefer **progress-aware / re-decidable waits** over a single fixed cutoff that
  discards in-flight work. A passive wait that does not kill should stay in its
  window while the observed task is non-terminal **and** progressing, up to a
  generous ceiling, then fail closed with a precise structured reason. Progress
  ADMITS the wait to keep waiting; it does not hand control back per event —
  returning on each advance woke a full-context nanny round every poll interval
  (measured: 18 rounds, 861k prompt tokens, for a run that was doing fine), so
  the observations are carried back once, at the window's expiry.
- [ ] Planning-scout collection is deliberately different: every started scout
  shares one terminal-or-cutoff boundary, and the reviewer receives explicit
  omissions at that boundary without a heartbeat-based early stop or inline
  fallback model.
- [ ] The wait/continue/stop decision must be a **structured fact** — terminal
  status plus heartbeat freshness from `queue_snapshot.json` — not a keyword or
  regex over content (Bible P5). Use `task_status.py` terminal-status helpers and
  the supervisor heartbeat, not string matching.
- [ ] Fixed **kill**-timeouts (hard task/tool ceilings, watchdog) still exist as
  the outer safety bound and get sensible ceilings under high-reasoning models;
  progress-aware waiting tunes the *passive* wait, it does not remove the watchdog.
- [ ] New numeric timeout constants are an SSOT in `config.py` `SETTINGS_DEFAULTS`
  with a getter and env registration; do not scatter magic wait numbers across
  call sites.

#### Loop / State-Machine Changes
- [ ] Changes to `loop.py` or other task state-machine logic include adversarial tests for malformed output, false-completion prevention, replay/log durability, and failure modes — not just the happy path.
- [ ] Audit/checkpoint rounds must not silently reuse the normal final-answer path unless that invariant is explicitly tested and documented.
- [ ] Keep a complete loop-local `DeliveryCandidate` once a substantive answer exists. A service round may return `keep`, or `replace` plus the complete replacement answer; allow one repair for malformed control, then preserve the prior complete answer and mark finalization degraded. A FORCED finalization (budget/round/deadline/provider/children rails) resolves an armed control purely and without retry instead: valid keep/replace is honored, anything malformed preserves the retained candidate with a typed degraded reason, and the protocol JSON itself never reaches chat or the durable result. A service notice alone does not change evidence. Owner messages, tool effects, child results, and verification receipts advance the evidence revision and require fresh delivery/acceptance binding. Finalize task-scoped service outputs/errors before host acceptance and require a complete replacement when that evidence changes; keep the `finally` path as idempotent cleanup only. This control must not bypass verification, acceptance, safety, skill-finalization, deadline, child-handoff, the unconditional `FINAL ANSWER:` latch, or the task-level answer protocol.
- [ ] Every direct child result needs an exact-hash disposition through the existing `tree_note(kind="decision")` tagged payload (`type=child_result_disposition`, child id, `integrated | irrelevant | deferred`, complete-result SHA-256; note text is rationale). One call may instead carry a `children` array of such entries (batch form): each entry is validated exactly like the single form, invalid entries are rejected individually by index while valid ones record. The typed task-tree row is the sole authority; task-result disposition fields are derived reads, never a mirrored write. The join-ledger helper alone validates lineage and current content. Stale or malformed payloads change nothing. `deferred` suppresses only the unchanged reminder and forces an honest degraded/best-effort terminal answer until the item is resolved. Natural completion WINS a late cancellation (owner decision 4=A, 2026-08-11): a child that settled its own completed result keeps it — payload, artifacts, and cost — and the cancel settles as already-settled; discarding a kept result is the parent's separate explicit `discard_child_result`. A cancelled (not completed) child still has its salvageable output preserved on the canonical drive before its bounded scratch is removed. Only a SETTLED `cancelled` status counts as a handled cancellation disposition: a child wedged in the legacy `cancel_requested` STATUS latch is intent, not outcome — it stays visible in the parent's handoff reminder as cancel-pending until custody settles it.
- [ ] Host task acceptance is root-only. Queued/headless/scheduled roots are reviewed in `auto` and `required`; direct eligibility is the union of `outcomes.turn_has_reviewable_effects` and a typed deliverable/criterion. Ordinary read-only tool activity, pure conversation, and meta/routing controls are not reviewed, and child reviews remain advisory. Eligibility must use structured facts, never keywords (Bible P3/P5). For an eligible root under `auto|required`, agent-callable `task_acceptance_review` validates/stores evidence and optional agent disposition but makes zero reviewer calls; it returns `deferred_to_host_acceptance`, `authoritative=false`, and the evidence revision. The call itself never widens eligibility; child and `off` behavior remain unchanged.
- [ ] Before root acceptance, atomically fence new descendants under the queue lock and prove recursive subtree quiescence from the existing task-status SSOT. Split-drive ACK, subtree, and acceptance-timing reads/writes use canonical `budget_drive_root`. Preserve the prior verdict until the replacement is recorded. A revision must explicitly reopen the fence; terminal/degraded outcomes seal it.
- [ ] The host runs the authoritative acceptance panel once per unchanged candidate-hash/evidence-revision/fence binding. Task-acceptance actors receive one substantive call and at most two physical attempts total. Record transport status, parse status, and valid-response semantic verdict separately, with actor model/provider, role, coverage, panel id, quorum contribution, reason, enforcement impact, and binding hashes. Public task/event/UI records receive only the compact projection; full model payloads remain in private audit storage. `adaptive_quorum` applies; any contributing FAIL fails, DEGRADED abstains (the reviewer verdict vocabulary `PASS|FAIL|DEGRADED` is NOT narrowable — `_contract_valid_actors`, the deliberate-DEGRADED capsule rail and the host's core-overflow DEGRADED all depend on it), and no quorum is a terminal HOST decision. The host acceptance decision itself is written ONLY by `loop._set_acceptance_decision` and has exactly three owner-facing states — `accepted | revision_requested | finalized_unaccepted` — each with a typed `reason` from an existing structured fact; an unknown status fails closed to `finalized_unaccepted` keeping its raw token as the reason. When you add a writer, add its reason to the closed set AND check every value-keyed reader: `outcomes.derive_loop_outcome` keys the eligible-but-skipped degradation on the status+reason PAIR (`review_skipped_deadline_reserve` plus the closed forced-rail `ACCEPTANCE_BYPASS_REASONS`), and breaking that pairing is a silent false green. Forced exits stamp their typed bypass record in the common terminal recorder (`_record_forced_acceptance_bypass`) as a pure ledger write — never a fence, panel, extra round, or prompt text on a forced path, and never overwriting an existing host decision — with ONE exception (owner decision Q2A, 2026-08-10): the forced `children_unabsorbed` rail still runs the acceptance panel for an acceptance-eligible root when the subtree is quiescent, with the undispositioned-children debt included in the evidence packet; because that rail cannot take another round, a requested revision terminalizes as `finalized_unaccepted` with the typed `revision_unavailable_on_forced_rail` reason, while the process outcome stays best-effort `children_unabsorbed`. The agent may write only `agent_disposition`/`agent_rationale`, merged into the host decision, never replacing it. Clean requires PASS + solved + supported criterion evidence. Chat and Logs must use the same severity reducer, and degraded review or best-effort/degraded objective must never render as green solved. Do not add task scope review or reuse the commit gate.
- [ ] The acceptance improvement loop is a reviewer-authored DIALOGUE (v6.74.0): obligation identity comes from the reviewer's typed `disposition_kind`/`obligation_id` (an unknown re-raise id fails closed to `new`, disclosed — never a silent fresh hash id); a re-raise reopens the row WITHOUT wiping the agent's argument (`previous_disposition`/`previous_reason`/`reopened_count` survive into the evidence catalog and the obligations clause); termination beyond a clean PASS/accepted rebuttal happens ONLY via the reviewers' quorum `dialogue_status` judgement reduced over ALL contract-valid actors (`aggregate_dialogue_status` — never `_contributing_actors`, which drops a DEGRADED slot's vote) or a real rail — no host counters, no answer/verdict hashes, no keyword gates (P5). Changes here must cover: malformed reviewer output, unknown/stale `obligation_id` on a re_raise, partial panel failure, multi-slot dialogue-status disagreement (the reducer's precedence), replay/restart durability of obligation rows, false completion, and the backward-compatible default when the new fields are absent.
- [ ] An explicit `max_improvement_passes` binds under every legacy policy. Required+Blocking without one has no local count cap, but real deadline/budget/lifecycle rails remain. The first acceptance review reserves at least 200s; later passes use the canonical event-derived `max(floor, 1.5×EWMA)` (`alpha=0.5`). Only the root runs global post-task synthesis once and persists one phase checkpoint in the canonical `budget_drive_root`. Recovery is startup-only: replay `pending_once`, degrade indeterminate `running` without another paid call, and let the normal supervisor copy-back/artifact path materialize child results without overwriting a terminal canonical phase or the finalized terminal accounting (`TASK_COST_META_FIELDS` plus rounds/tokens).

#### Cognitive Artifact Integrity
- [ ] Cognitive artifacts (identity.md, scratchpad, task reflections, review outputs, pattern register) must NOT use hardcoded `[:N]` truncation. If content must be shortened, include an explicit omission note (e.g. `⚠️ OMISSION NOTE: truncated at N chars`).
- [ ] `BIBLE.md`, `docs/ARCHITECTURE.md`, and `docs/DEVELOPMENT.md` are **core governance artifacts**. All primary reasoning flows (triad review, consciousness, advisory pre-review, deep review) include them as first-class sections — see the "Core Governance Artifacts" table. If you add a new reasoning flow, it MUST follow this contract, not rely on touched-file inclusions.

---

*This section is the authoritative definition of "DEVELOPMENT.md compliance" referenced in the `development_compliance` item in `docs/CHECKLISTS.md`.*

---

## Managed Update Rule

- Keep the local work branch and the official update feed separate. The local
  branch is `ouroboros`; `OUROBOROS_UPDATE_CHANNEL` maps Stable to `main`, QA to
  `ouroboros-stable`, and Development to `ouroboros`. QA and Development follow
  their branch tips. Stable resolves the newest plain `vX.Y.Z` tag whose commit
  is present in both `main` and `ouroboros-stable`.
- A preflight chooses one exact official target SHA. Apply must bind to the
  disclosed base/target, close new writers, drain existing direct/ephemeral
  turns, stop workers and tracked services, then re-plan before mutation.
- Clean fast-forwards land the official SHA directly. Git also builds clean
  merges for divergent local history, with parents = reviewed HEAD + official
  target. Dirty local work never enters that history: the apply stashes it and
  restores it as uncommitted content (boot finalize on success, the pre-update
  tree on rollback; a conflicting restore keeps the stash and discloses the
  recovery command). The reviewed assisted resolver runs only when Git reports
  a real conflict; filenames do not create a second update policy. Hard reset
  is an explicitly confirmed recovery only.
- The authorized resolver stages the complete merge, including tracked binary
  files. Review receives their exact staged mode/blob/size plus the HEAD and
  official MERGE_HEAD object ids; deletions carry an explicit absent stage and
  exact parent identities. Missing exact metadata still blocks. This exception
  does not weaken the ordinary commit pipeline's binary policy.
- Write the update transaction before mutation. Reopen writers only after a
  verified abort/rollback or a healthy restart. An unverified rollback keeps
  its retryable phase plus the full failure evidence; a legacy `gate_blocked`
  marker retries rollback on boot. Delayed evolution cleanup also acquires the
  same update lock and honors this admission owner; it must not stash/reset
  behind the fence. Managed merge tests pass before restart; the ordinary
  self-modification commit/tag/test/push ordering remains unchanged.
- Take a fresh rescue before every destructive rollback and before boot-resume
  re-materialization: the pre-update snapshot predates the merge and holds none
  of the resolution. The hook is fail-open — never block a rollback on it — but
  its outcome, captured or failed, is disclosed durably at capture time, before
  the destruction. Record the pointer in the update transaction so a replayed
  rollback does not re-snapshot and a retry rescues what appeared since; keep
  that pointer until the transaction ends, because a re-materialized merge looks
  identical to restored work and is not evidence the rescue was applied.
- Manual Restore reuses the same writer fence and pins the previous HEAD on a
  local recovery branch before reset. Promotion resolves the development SHA
  once and uses that exact SHA for both the local QA ref and any remote push.

---

## Mutation Attribution Rule

- Attribution is evidence, not exclusion. The host captures a `system_repo`
  baseline in the existing task result when a queued root task starts and a
  terminal candidate snapshot at outcome derivation; blockers (pre-existing
  dirty changed, stale/missing baseline, failed scan) ride into review and
  acceptance evidence for the LLM panels to weigh. Do not turn them into
  structural outcome vetoes, and do not add a lease/holder service, a second
  ledger, or runtime writer keyword scanners.
- Git staging is attribution-based. `paths=None` means the clean-at-baseline
  candidate set, an explicit list must be its subset, and empty never means
  `git add -A`. Preserve pre-existing user dirt as excluded evidence. Whole-tree
  staging belongs only to already-typed managed update/release transactions and
  a dedicated typed external patch-capture transaction. Contexts without a captured baseline
  (manual ToolContext, external dry-run review) keep the legacy staging
  contract.
- Resolve unversioned Python only for `run_command`, `run_script`,
  `start_service`, and run-kind `verify_and_record`, once before the shell guard.
  Guard and handler must receive identical argv. Do not rewrite explicit paths,
  versioned interpreters, shell bodies, or remote execution, and never install a
  dependency in response to `ModuleNotFoundError`.
- Skill Review ordinals and provenance stay in `review_job.json` and the
  append-only `review_history.jsonl`: allocate under the lifecycle lock, consume
  a round only after actual start, write one terminal row per `job_id`, and
  compute legacy ordinals at read time without rewriting history.

## Guard Proof Rule

**A guard that forbids something must be shown REFUSING something.** Passing valid
input proves nothing about a prohibition: a guard can be structurally unable to
fire and look identical from the outside. This branch produced three, all found by
hand and none by the suite — an elision gate behind `if False and ...`, an absent
symlink-confinement case, and `build_execd_bundle`'s Home-module list, which held
repo-relative spellings (`ouroboros/config.py`) and compared them against stage
paths (`lib/ouroboros/config.py`), so it had never refused anything since it was
written. Call the family a **vacuous guard**, and note what it costs: the artifact
looks gated and is not.

Rules:

- **Every prohibition needs a negative case.** A test that feeds the guard a real
  violation and asserts the refusal — by message, not merely by exception type, so
  a refusal from a DIFFERENT guard cannot stand in for it. The positive case
  (valid input still passes) is necessary too, and is not a substitute.
- **Compare in the namespace the data actually exists in.** The packager's list was
  written in repo-relative paths for a tree that only ever holds stage-relative
  ones. Where a prohibition already has a source of truth, IMPORT it instead of
  restating it (`FORBIDDEN_REMOTE_IMPORT_PREFIXES`), and where the data carries its
  own declaration, judge against that declaration (the stage provenance's
  `kernel_modules`). A restated list is a list that can drift.
- **Prefer an allowlist closure to a denylist.** "Only these modules may be under
  `lib/`" refuses the leak nobody predicted; "not these seven files" refuses only
  what someone remembered.
- **Say so mechanically, not in a list.** `tests/test_execd_packaging_guard_proofs.py`
  is the worked example for the packaging line: it reads every `raise` out of the
  three scripts' ASTs and demands that each message is either produced by a
  constructed violation or carries a written reason it cannot be (real Linux runner,
  live service). A guard that cannot fire never produces its message and fails
  there; so does a guard added later with no case; and so does a reason that has
  become false. Copy that shape when a surface's guards are worth this much — it
  costs one module and needs no hand-kept inventory.
- **An exemption list needs its own liveness check.** A waiver for a file that no
  longer violates anything silently pardons the next one that does.
  `tests/test_process_custody.py::test_the_custody_allowlist_holds_no_dead_exemptions`
  is the pattern: subtract the live findings from the waiver set and fail on the
  remainder. Two entries there had gone stale exactly this way.
- **Ask which AXIS the rule set covers.** A gate hardened four times along one axis
  can still be blind along another: `tests/test_platform_guard.py` grew four rounds
  of new SPELLINGS for reaching a fixed list of forbidden identifiers, and never a
  new KIND of platform dependency — so a `/proc` read, which spells no identifier,
  passed every rule while `platform_layer` was reading that same file three lines
  from its own constant. When a category is added, prove each one fires (that file's
  `test_each_forbidden_category_actually_produces_a_finding`), or the next category
  arrives dead. The second instance is `tests/test_remote_doc_claims.py`, hardened
  four times along DOC → CODE and blind to CODE → DOC, so a map that OMITTED a real
  module told no lie any rule could read; see the Document Truth Rule below. The
  THIRD is the export-policy appliers, and it is the one that shipped a real leak:
  every symlink test asked about CONFINEMENT (does a link out of the root get
  refused) and none asked about POLICY (a link that stays inside, onto an excluded
  file), so `read_file("safe.txt")` returned the `.env` bytes that `read_file(".env")`
  refused, and one test PINNED the permissive half as correct. Both the axis and the
  masking generalize: **ask what a guard judges, not only whether it judges.** A
  policy applied to the requested SPELLING and a resolver that then hands over a
  different file is one guard with two subjects; the honest subject is the IDENTITY
  the bytes will come from (`export_policy_contract.judged_exclusion`, and the
  alias table in `tests/test_route_refusal_parity.py`). The FOURTH instance is the FIX
  for the third, and it earns the sharpest form of the clause: **two mechanics for one
  question is a guarantee that the weaker one becomes the hole.** The alias fix gave the
  WALK channels a recursive inode seed and left the single-source doors on a root-only
  `scandir` probe — two answers to "which file is this" — and a paid reviewer found the
  weaker one leaking in five places on a branch whose commit message said the class was
  closed: a hardlink to `sub/.env` excluded from `search_code` and returned verbatim by
  `read_file`, a hardlink to a nested protected artifact appended to and edited THROUGH,
  and a declared-output door that called the spelling evaluator with no identity check at
  all. The same round also produced the corollary about ARGUMENTS: the door was made
  compulsory (`question` a required keyword) while the judging ladder stayed PUBLIC, so
  eight call sites simply used the ladder — a mandatory door with an exported bypass is
  not a door. Both halves are now structural: one `judged_exclusion` for every producer
  of bytes, the ladder private, and an AST gate that recomputes the execd import closure
  and fails if any module inside it names a second judging function
  (`tests/test_export_policy_contract.py`). Four independent gates failing the same way
  is the argument for asking the question routinely rather than after a paid review: name
  the axis, then name what is off it — and when a fix leaves TWO answers to one question,
  say which one every caller gets and prove the other is unreachable.
  The FIFTH instance is the clause turned on the TESTS, and it is the one that explains why
  the class survived two rounds. **Ask which axis a FIXTURE varies.** Every entry in
  `tests/test_route_refusal_parity.py::_ALIAS_KINDS` nested the ALIAS and left the SECRET at
  `root / <name>`, so the `nested_*` rows only ever proved that a deep alias to a ROOT secret
  is caught — which the root-bounded probe managed anyway. The axis had two ends and the
  table varied one, so a green table and an open hole were the same picture. Three sibling
  gates in the same round could not fail at all: a completeness sweep whose candidate set was
  FILTERED to the already-classified names (so its `missing` was empty by construction), a
  syntactic mutation sweep that only entered its scan for an `ast.Attribute` callee (so a
  builtin `open(path, "w")` was invisible and the branch written for it was dead code), and a
  door list kept by hand with `if not path.exists(): continue`. The rule that catches all
  four: **a guard must be shown failing on the case it was written for, and a fixture must
  vary every end of the axis it names.** Deriving the door list from an authority instead of
  restating it immediately found a sixth copy of the rule table nobody had listed.
- **A file with zero behavioural coverage has zero proven guards.** Whether a
  reject branch has ever executed is a measurable fact, not an opinion. With
  `pytest-cov` installed (a local audit tool, not a runtime dependency), run the
  default lane AND the `serial` lane under `--cov=ouroboros --cov=scripts
  --cov=supervisor --cov=server --cov-branch --cov-append`, then cross the
  `raise` line numbers from each file's AST against `coverage.CoverageData`. Both
  lanes, or the answer is wrong: measuring only the parallel lane reported the
  browser-forward and OpenSSH guards as never-fired when their proofs simply live
  in the serial pass. That sweep is what found the three defects above; the whole
  execd packaging line came back at 0 of 64 before it was fixed.

## Refusal Action Rule

**A refusal that ADVISES something must be shown that the advice WORKS.** The Guard
Proof Rule above covers half a refusal — that it fires. This is the other half. A
refusal whose proposed action cannot remove the block is worse than one with no action
at all: the owner presses what they were told to press, it SUCCEEDS, and nothing
changes, so the surface has taught them that its own advice is noise. Call the family a
**dead-end refusal**. This branch shipped one to the owner and it was measured, not
guessed: with an executor built against an older Home↔execd contract set, New Project
said "run Bootstrap (or Test to refresh health)", `Test` returned `ok` with
`health_fresh: true`, and the connection stayed unselectable — only Bootstrap writes the
contract-set stamp the picker reads.

Rules:

- **Derive the action from the same structure as the block, never from the call site.**
  `ouroboros/remote_refusal_actions.REFUSAL_ACTIONS` maps code → the one action that
  removes it, and `RemoteWorkspaceError.__init__` reads it. That is why ~40 raise sites
  gained a correct action with none of them edited, and why a new code cannot ship
  without a decision about what cures it. A handler that writes `action="..."` as a
  literal is choosing on behalf of every condition its `except` arm can catch —
  `gateway/tasks` and `gateway/projects` both did, and both said `bootstrap_connection`
  for an absent broker, a replaced host identity and a stale bundle alike.
- **Name ONE action, not the union of every cure.** A message that lists everything
  that might help has said nothing about what does, and the owner will pick the cheapest
  item. Two actions that BOTH genuinely clear a block are fine (`wait_or_cancel_tasks`);
  an action that cannot is not. Where an owner has already been misled, say what will
  NOT work and why — the `remote_execd_outdated` hint names Test explicitly and says it
  will report healthy and change nothing.
- **Several blocks at once: name the one blocking NOW, in removal order.**
  `connection_blocker` is an ordered ladder returning the first match; the rank it
  reports is the position, so a surface choosing among several blocked rows can pick the
  one nearest to ready without deciding anything itself.
- **One vocabulary of action names, closed and checked.** Two spellings of one action
  are indistinguishable from two different actions to the reader. There were two such
  pairs here (`rebind_project`/`choose_active_connection`,
  `retry_reconnect`/`reconnect_connection`), and an AST sweep over every `action=`
  literal now fails on a third.
- **Prove it by TAKING the action.** `tests/test_remote_refusal_action_proofs.py` is the
  worked example: for each blocking state it brings a real store into that state, reads
  the server's proposed action, looks THAT action up in a table of what performing it
  does, performs it, and requires the same blocker to be gone. Keying the cure by the
  PROPOSED action is the whole mechanism — a hint that advised Test for a stale executor
  makes the test run a health probe and find the block still there. Assert the failure
  message, not just the exception: the test prints `DEAD END: <state> proposed <action>,
  it succeeded, and the same block is still there`.
- **Cross-check against any independent authority that already exists.** The suite found
  fourteen codes that proposed `retry` while `cli_connections._UNSERVABLE_CODES` mapped
  them to exit 4, "retrying will not help" — and one (`broker_overloaded`) where the SET
  was the wrong half of the contradiction. Two tables that describe one taxonomy should
  be compared by a test, not by a reader.
- **Every rung needs a reachable state.** A hint nobody can bring about is the vacuous
  guard wearing a remedy: the first draft of the ladder had a `connection_connecting`
  rung, and the proofs could not produce it (`_record_runtime_health` never records
  `connecting`), so it was replaced with the state that is actually reachable.

## Route Parity Rule

**Where two routes execute the same operation, the second must REFUSE everything the
first refuses — and a guard that fails must fail CLOSED.** The Guard Proof Rule above
asks whether a guard fires; the Refusal Action Rule asks whether its advice works. This
asks the question neither of them can: is the guard even THERE on the other route. Call
the family a **one-sided guard**, and note that it is invisible to both rules above,
because on the route that HAS it every test passes.

It is the PR 79 failure class ("one policy × N doors") on a new axis — not two copies of
a rule that drifted apart, but one rule that only ever existed on one side. Four
instances, all confirmed by review and all the same sentence:

- `write_file` with `mode="append"` followed a symlink in the final path component out of
  the workspace on the target. The local route resolves the whole spelling and refuses.
  Reproduced live: the file outside the workspace grew.
- `start_service` SANITIZED (`re.sub`) a service name the local route REFUSES, so `a/b`
  and `a_b` shared one log file and `service_logs` could return the other's output.
- `native_relative_spelling` accepted NUL and control characters that `utils.safe_relpath`
  has rejected locally since before there was a remote route.
- the public argument-schema refusal ran AFTER `prepare_operation` on the native branch,
  so a malformed call reserved a token on someone else's machine before Home refused.

Rules:

- **One door per intent, and the rule lives in the door.** The append escape was not a
  missing check — it was a check whose PLACEMENT let one caller decide differently. The
  parent-only confinement was correct for `_atomic_write` (where `os.replace` substitutes
  the link rather than following it) and wrong as a rule, so the next open site reopened
  the hole. The fix is `workspace_native_paths.native_mutation_target`: every native
  mutation goes through it, and `tests/test_target_confinement_and_disclosure` FAILS when
  a new mutation site does not. A per-mode test only ever covers the modes someone
  thought of.
- **A shared rule has ONE owner and both routes import it.** `SERVICE_NAME_PATTERN` lives
  in `workspace_native_contract` and `tools/services` imports it; the parity test asserts
  object IDENTITY, not that two regexes happen to agree. Two spellings of one rule is how
  the sanitize/refuse split happened in the first place.
- **Compare the routes in a TEST, not in a reader's head.** What made this a class is
  that nobody was comparing them at all, so the comparison is the artifact:
  `tests/test_route_refusal_parity.py` asks both resolvers the same refusable question and
  requires the same verdict. Half the cases are ACCEPTANCES — a door that refused
  everything would satisfy a refusal-only table, and the in-root-symlink case caught a
  wrong first draft of the fix that would have been a new asymmetry pointing the other
  way.
- **Declare an asymmetry you are not closing.** One remains: an absolute spelling is
  refused natively and silently rebased locally (`safe_relpath`'s `lstrip("/")`). It is
  asserted AS an asymmetry, with the reason, so aligning them later is a decision rather
  than a surprise.
- **A guard that cannot answer must answer NO.** `except Exception: return False` where
  `False` means "permitted" is a guard that opens exactly when it breaks — and a failing
  live-task lookup and a busy queue are hardly independent events. Three answers, not
  two: `gateway/connections._live_connection_tasks` returns `None` for "could not tell"
  and `_connection_busy` treats it as busy; `gateway/projects._project_has_live_tasks` and
  `gateway/settings._has_running_agent_tasks` now agree. Inject the failure at the SEAM
  the function imports, so the real `except` arm is what runs and not a stub standing in
  for it.
- **Silently ignoring an owner's declaration is worse than refusing it.**
  `metadata.workspace_ref` was stripped and `metadata.connection_id` was stored and then
  ignored, so an owner could name a placement nothing honoured. Both are typed 400s now,
  from ONE loop that checks the body and `metadata` together — the body was checked and
  `metadata` was not, which is the same one-sided shape at the level of a request field.

## Document Truth Rule

**A document that names something must be checkable against the thing it names, and a
map that claims to be complete must be checked in the direction of the CODE.** The three
rules above are about guards that do not fire. This is about the other authority in the
repository: a sentence. `prompts/SYSTEM.md` is resident context, so a stale claim there
does not misinform a reader, it STEERS the agent — it once named two of the
target-executing tools as having no remote path, and the model dutifully hand-rolled
weaker substitutes for two tools whose remote routes were complete and gate-covered.
Call the family a **document lie**, and note what makes it expensive: the
prose reads more confidently than the code, and nothing was watching.

`tests/test_remote_doc_claims.py` is the worked example, and it earned this rule by
FAILING: it existed precisely to keep the documents honest, was hardened four times, and
then let four lies through at once. So the first rule is the one the Guard Proof Rule
already states, applied to itself.

Rules:

- **Ask which AXIS the checks cover, and expect the answer to be "doc → code".** That is
  the easy direction, and every check written by instinct lands there. A map that OMITS a
  real module makes no false statement — `tools/dispatch_policy.py` and
  `state/remote_reconciliation/` were simply not there, and silence is unreadable to any
  rule that reads what the text SAYS. The code → doc direction needs its own checks, and
  its authority is the filesystem or the module, never the document.
- **Judge a name by RESOLUTION, never by spelling.** A rule keyed to
  `(remote|execd|workspace|connection|cli)_*\.py` can only ever judge names that were
  already thought of; a plain-named module, a directory, a new subcommand and a bare
  `module.symbol` all sailed past. Resolve the path, resolve the symbol, and ask the
  argparse parser for the subcommand set instead of restating seven names — the same
  "import the authority" clause the Guard Proof Rule makes for prohibitions.
- **An inventory is checkable in both directions; say so and check both.** A module map,
  a state layout, a field list, a tool list. Each has a code-side authority, and the
  interesting failure is always the missing entry rather than the invented one.
- **What cannot be mechanized must be written down as such, keyed so it cannot rot.** A
  BEHAVIOURAL promise ("streams rather than buffering whole blobs") and a PROCESS promise
  ("a Home restart requires Bootstrap again") are claims about what the code DOES; no
  reading of the text settles them, and matching their phrasing would gate wording rather
  than truth. `CLAIM_KINDS` in that module is the register: each kind maps either to the
  check that judges it or to a written reason none can, it fails when a named check
  disappears, and it fails when a check belongs to no declared kind — so the next rule
  added has to say which axis it closes. Both of the two lies above were caught by a human
  comparing a paragraph to a docstring, and pretending otherwise is how a gate stops being
  believed.
- **Do not describe such a gate as making the documents true.** It makes their
  IDENTIFIERS and their INVENTORIES true. That is most of the mechanizable half, and
  claiming more of it would itself be a document lie.

## Process Custody Rule

Long-lived OS processes (anything `subprocess.Popen`-ed or `mp.Process`-ed
without a bounded wait in the same call) **MUST** be spawned through
`ouroboros.process_custody.spawn_supervised(cmd, drive_root=..., purpose=...,
scope=...)` — or, when an existing manager owns the Popen call, registered via
`record_process(...)` write-through immediately after spawn. The custody
ledger (`data/state/process_ledger.jsonl`) is what lets the orphan reaper find
children after an abrupt worker/server death; an unledgered process orphans
invisibly and forever. Scopes: `task` (dies with its task), `session` (dies
with the server generation), `daemon` (genuine launcher-managed lifecycles,
e.g. `server_restart_fallback` — reaper keeps them, only pruning dead entries).
Skill **companions** also record `daemon` scope but are the documented
exception: `reap_orphaned_processes` reaps a companion (`purpose
companion:<skill>:<name>`) when its owning skill is **uninstalled** OR the entry
is from a **foreign (dead) server generation** (`CompanionSupervisor.start()`
always re-spawns a fresh pid, so a generation-crossing match is a stale
duplicate). This is **log-only by default** (`enforce_companion_reap=False`
emits a `process_would_reap` event instead of killing) and **fail-safe**:
`live_owner_skills=None` (unknown install set — incl. a momentarily empty skills
dir, coalesced to `None`) means keep-all, never a mass-kill, and same-session
companions of installed skills are always kept so the live `CompanionSupervisor`
stays their sole owner. The reaper kills strictly by (pid, start_time,
cmd_sha256) fingerprint — never add command-line-class matching, which would let
a dev instance reap a packaged instance's processes.
`tests/test_process_custody.py` enforces the chokepoint with an explicit
allowlist for bounded synchronous helpers.

Remote processes are never written into Home's PID ledger: a remote PID or PGID
belongs only to execd/custodian state, where it means something. What Home
records and owns through the same required-custody rule are its LOCAL children —
the OpenSSH transport, broker, and forward processes — registered under `session`
scope so the orphan reaper covers abrupt server death. The current Home server
generation owns every remote task and service lease. Task cancel and service
stop preserve the connection; a full-app Panic stops lease renewal, sends
priority kills, and closes local custody immediately without waiting for an
acknowledgement. The independent remote custodian is the ONLY authority for the
physically-partitioned lease ceiling, and that ceiling is a failure-detection
bound — never a grace period a reachable kill may wait out.

## Platform Abstraction Rule

All platform-specific code **MUST** go through `ouroboros/platform_layer.py`.

`tests/test_platform_guard.py` enforces this by AST over `ouroboros/`,
`supervisor/`, `scripts/` and `server.py` (`tests/` is excluded — probing platform
behaviour is legitimate there — and `web/` has no Python). It rejects every form
that spells a forbidden name LITERALLY, not just the obvious one: a forbidden
import in any scope (`fcntl`, `msvcrt`, `winreg`, `resource`, `select`,
`selectors`, `termios`, `pty`), the same import reached via
`importlib.import_module("fcntl")` or `__import__("pty")`, a forbidden `os.*` /
`signal.*` attribute under its own name OR an alias (`import os as o; o.kill`),
the same attribute reached by literal name (`getattr(os, "set_blocking")`,
`os.__dict__["killpg"]`), and a platform-conditional subprocess kwarg passed
either as a literal keyword (`start_new_session=...`) or splatted from a literal
dict (`**{"start_new_session": True}`). Splatting a helper CALL
(`**subprocess_new_group_kwargs()`) is the prescribed pattern and stays legal
because no literal key appears in the source.

It also rejects two categories that are not identifiers at all, which is how the
boot-id read escaped it: a platform-EXCLUSIVE filesystem path handed to a
filesystem entry point (`/proc`, `/sys`, `/dev`, the Windows `\\.\` and `\\?\`
namespaces, the DOS device names — in literal, f-string or leading-concatenation
form), and a platform-exclusive clock (`time.clock_gettime`, `CLOCK_BOOTTIME`,
`CLOCK_UPTIME`). A kernel pseudo-filesystem is an API reached by PATH, so a rule
set made only of forbidden NAMES is structurally unable to see it. Shell text
built for a remote POSIX target is not a host reach and stays legal — the
entry-point condition is what makes that distinction, and both directions are
pinned.

Its boundary is stated in the test's own docstring and must stay stated: `**kw`
where `kw` is a variable, and an attribute name assembled at runtime, cannot be
resolved without executing the program and are NOT detected. Do not describe the
gate as complete coverage, and do not silence a new finding by weakening a rule —
route the code through `platform_layer` or add an allowlist entry with a reason.

### Shared State-File Helpers

Durable JSON state files should use the SSOT helpers in `ouroboros/utils.py`:
`atomic_write_json(path, payload, trailing_newline=False, fsync=False)` for
write-then-rename persistence and `read_json_dict(path)` for dict-shaped JSON
reads. `write_text_atomic(path, content, fsync=False)` is the underlying shared
atomic FULL-OVERWRITE primitive (temp-sibling + `os.replace`, existing permission
bits preserved, crash leaves the old file intact); `atomic_write_json` layers JSON
serialization on it, and `write_text` (the plain text overwrite helper) routes
through it, so every overwrite routed through these helpers is crash-safe — prefer
them over a bare `Path.write_text` for any full-file overwrite. Appends are
intentionally NOT atomic (they extend in place). Lockfile acquisition should go through
`platform_layer.acquire_exclusive_file_lock` /
`release_exclusive_file_lock` rather than reimplementing `O_CREAT|O_EXCL`
loops in feature modules. Owner-state files that must not be world-readable pass
`mode=0o600` (`atomic_write_json` / `write_text_atomic`, and
`acquire_exclusive_file_lock` for the lock beside them); callers that need
rename durability additionally pass `fsync=True, fsync_directory=True`. Those
are narrow parameters on the SHARED helpers precisely so an owner-state module
never grows its own atomic-write sequence.

Narrow exceptions are allowed only when the file's contract is not JSON-object
state or intentionally has extra durability semantics: `supervisor/state.py`
keeps `atomic_write_text` for mirrored `state.json` / `state.last_good.json`
text writes, and `ouroboros/config.py` keeps its settings-file lock because the
settings path is bootstrapped before broader runtime helpers should depend on
settings state.

### What counts as platform-specific

- Direct use of: `os.kill`, `os.setsid`, `os.killpg`, `os.getpgid`, `signal.SIGKILL`, `signal.SIGTERM`
- Unix-only modules: `fcntl`, `resource`, `grp`, `pwd`
- Windows-only modules: `msvcrt`, `winreg`, `ctypes.windll`
- `subprocess` with platform-conditional flags: `start_new_session`, `creationflags`
- Hardcoded path separators (`/` or `\\`) in filesystem logic (use `pathlib` instead)
- A path into `/proc`, `/sys`, `/dev`, or a Windows device namespace; a platform-exclusive `time.CLOCK_*` constant

### Rules

1. **All platform-specific calls live in `platform_layer.py`** — the rest of the codebase imports cross-platform wrappers from there.
2. **Platform-specific modules are imported inside `platform_layer.py` only**, guarded by `IS_WINDOWS` / `IS_MACOS` / `IS_LINUX` checks.
3. **No top-level imports of Unix-only or Windows-only modules** outside `platform_layer.py`. If you need `fcntl` — you're in the wrong file.
4. **Use `pathlib.Path`** for filesystem paths. Never construct paths with string concatenation using `/` or `\\`.

### Enforcement

- **AST-based test** (`tests/test_platform_guard.py`): scans `.py` files under `ouroboros/`, `supervisor/`, and `server.py` for:
  - Top-level imports of platform-specific modules (`fcntl`, `msvcrt`, `winreg`, `resource`)
  - Direct `os.kill`, `os.killpg`, `os.setsid`, `os.getpgid` attribute access
  - Direct `signal.SIGKILL`, `signal.SIGTERM` attribute access
  - Function-LOCAL imports of those modules, and the aliases `termios` / `pty` /
    module-level `select` (a `select.select`/`select.poll` on a non-socket fd is
    Unix-only)
  - `os.set_blocking`
  - Literal subprocess platform flag keywords (`creationflags`,
    `start_new_session`) — callers must splat `subprocess_new_group_kwargs()` /
    `subprocess_hidden_kwargs()` from `platform_layer.py` instead

  `launcher.py` remains intentionally excluded as the immutable outer shell.
  The guard was extended to the evasions above after three consecutive review
  rounds found platform-specific calls that the earlier narrower scan could not
  see; the `cross_platform` checklist item is now defense in depth rather than
  the only line of defense.
- **Pre-commit review**: checklist item `cross_platform` (#15) catches violations during code review.
- **CI matrix**: tests run on Ubuntu, Windows, and macOS to catch runtime failures.

### Adding new platform-specific code

1. Add the cross-platform wrapper to `platform_layer.py`.
2. Import and use the wrapper in callers.
3. Add platform-conditional tests if behavior differs across OSes.

---

## Design System

> **Authority split — read this first.** `docs/DESIGN.md` defines the **visual
> and interaction semantics**: the type scale and hierarchy rule, foreground and
> state colour meaning, status/chip conventions, card and row anatomy, onboarding
> density, dark-theme contrast. **This section defines the engineering rules that
> preserve them**: where values may live, which component is the SSOT, what
> counts as review debt, how a visual change is verified. Changing what a size or
> colour *means* is a `DESIGN.md` edit; changing how the code is allowed to
> *express* it is an edit here. Neither file restates the other.

`web/style.css` custom properties and shared component classes are the value
SSOT. Documentation keeps semantic roles and failure-prevention rules, not a
copied color/radius/dimension inventory.

Typography and hierarchy are `docs/DESIGN.md`'s call, and the engineering
consequence is narrow: a text declaration on a migrated surface names a
`--type-*` size token and a named foreground token. A rule that declares a size
and no colour is the specific defect that made secondary text inherit
near-white primary ink across the settings panels;
`tests/test_web_typography_static.py` keeps that class closed on the migrated
files. Migrating a new surface and extending that guard to it are the same
commit.

### Layout and controls

- Top-level pages use a fixed `renderPageHeader` outside an independently
  scrolling body. Shared tab strips, buttons, scroll fades, page glass, icons,
  and masonry helpers replace page-local copies.
- Static visual properties belong in CSS classes/tokens. New inline
  `style=""` markup and `.style.<property>` assignments are review debt.
  Dynamic measured values may update a narrowly named CSS custom property when
  that is the actual runtime data flow.
- One semantic button variant expresses one action role. Notifications use the
  shared toast host unless status belongs to a permanently reserved control
  row. Working, warning, error, and destructive states keep consistent meaning
  across Chat, Logs, Settings, and Skills.
- Floating chrome combines gradient and masked backdrop blur so the blur edge
  does not become a visible seam. The chat composer intentionally keeps blur on
  the input surface and reserves measured message padding around the dock.

### Responsive and accessible behavior

Navigation, headers, controls, and dialogs must stay operable by pointer and
keyboard, preserve focus order, and fit the relevant narrow viewport without
stealing usable text space. Use the shared responsive component before adding a
page-specific layout. A visible change is inspected with vision in at least one
relevant real consumer flow. A stored screenshot alone is not verification;
mobile or WebKit is not a universal requirement and is selected from risk.

### Browser dialogs

`window.prompt`, `window.confirm`, and `window.alert` are forbidden in `web/modules`. PyWebView shells implement them inconsistently, native dialogs bypass the design system and browser tests, and the macOS shell has no prompt delegate, so `window.prompt` silently returns `null`. Use `confirm_dialog.js::openConfirmDialog`: confirm mode returns a strict boolean, input mode returns `{confirmed, value}`, and alert mode renders one acknowledgement action. Close, Cancel, backdrop, Escape, and supersession are always non-confirming. Critical actions must test the exact confirmed result and keep the confirmation plus side effect in one injectable flow. `tests/test_web_dialogs_static.py` keeps the native-dialog class closed; host-side PyWebView bridge dialogs are a separate platform surface.

### Declarative widgets

`web/modules/widgets.js` is the host for reviewed widget declarations.
Declarative v1 includes forms/actions, text/data/media, tabs/charts, async jobs,
files, map/calendar/kanban, and composition through `group`, `metric`, and
`callout`. Nested interactive components use stable identity and one disposer;
`subscription.render` is transitively passive. Escape text and attributes for
their actual HTML contexts, constrain media to extension routes or safe data
URLs, and keep charts accessible through a semantic table.

Rare `kind: "module"` UI runs only in a sandboxed opaque-origin iframe, with no
`allow-same-origin`; its parent bridge proxies only the owning extension route.
Never load skill JavaScript into the SPA origin. Long-running actions use a
durable job id and resumable status polling rather than a foreground request
lost on remount.

Every timer, listener, observer, stream, abort controller, chart, and mounted
widget has a paired disposer. UI preferences such as widget order belong in
host state, never in extension manifests.

## MCP Client Integration

The base runtime is an optional client for trusted HTTP/SSE and local stdio MCP
servers; it is not an MCP server. `ouroboros/mcp_client.py` owns server parsing,
transport-specific validation, provider-safe tool names, discovery, timeout,
and result normalization. `ouroboros/secret_masking.py` owns the exact shared
Settings/MCP auth-placeholder emitters and recognizers. Settings carry only
`MCP_ENABLED`, `MCP_TOOL_TIMEOUT_SEC`, and structured `MCP_SERVERS`; tokens never
appear in status responses.

MCP descriptions/results are untrusted data, not policy. Enabled tools join the
initial capability envelope, still pass runtime safety, and remain unavailable
in repair/heal contexts. Discovery failure becomes a visible capability
omission. Stdio accepts one executable command and an exact string argument
list, uses no shell, custom environment, or custom working directory, and
relies on the SDK context for teardown. Resources, prompts, and MCP server
behavior remain separate architecture changes.

## Gateway Boundary Pattern

Browser-facing backend work enters through `ouroboros/gateway/`.
`gateway/contracts.py` owns frozen envelopes/indexes and `gateway/router.py`
owns route mounting; domain handlers remain thin. Frontend calls go through
`web/modules/api_client.js`, with `api_types.js` and parity/smoke tests mirroring
the contract. Outbound provider/harness adapters belong in
`ouroboros/gateways/`. Do not require a class when established function owners
already preserve the boundary.

## Build & CI

### Python dependency locks

`pyproject.toml` is the direct-dependency SSOT and `uv.lock` is the reviewed
cross-platform resolution. Local and CI commands use `uv sync --locked`; do not
add an independent hand-written requirements list. Release packaging preserves
the separate build and embedded interpreters by exporting build requirements
ephemerally and committing `requirements-runtime.lock` for embedded pip. The
legacy `requirements.txt` is a generated pointer to that export for N-1 managed
updaters, never a dependency authority. A dependency change updates the project
metadata, runs `uv lock`, regenerates the runtime export with the exact command
in README, and leaves its CI clean-diff check green.
The pinned `tool.uv.required-version` and the digest-pinned `setup-uv` action
make resolver changes deliberate rather than an ambient CI upgrade.

`uv tool install "git+https://github.com/razzant/ouroboros.git@ouroboros"`
is the documented checkout-free CLI/server path. It resolves the project
metadata into an isolated tool environment but does not read this repository's
`uv.lock`; branch installs therefore follow dependency ranges as well as source
HEAD. Documentation may pair that convenient form with a full commit SHA to pin
the source revision, but must not claim that it locks dependencies or describe
it as a release-artifact install or contributor development environment.

### Pytest marker lanes

Default local pytest excludes costly or environment-dependent lanes:
`integration`, `browser`, `ui_browser`, `ui_browser_docker`,
`portable_detail`, and `skill_smoke`. CI opts into them explicitly:

Lane membership is ENFORCED by discovery, not by memory:
`tests/test_serial_lane_contract.py` scans every test for real subprocesses, real
sockets, `threading.Thread` and load-bearing `time.sleep` calls, and requires each
candidate to be classified — in the serial lane, in another lane, or in
`PARALLEL_SAFE_CONCURRENCY_TESTS` with the reason it is safe there. A real
subprocess or socket has NO exemption (those share resources across processes, so an
xdist sibling can take one or be mistaken for the test's own child); threads and
timing may stay parallel, but only on the record, and the exemption set is checked
for exact equality so a stale entry fails too. Two suites had escaped the hand-
maintained list this way — `test_admission_invariants.py` (two threads racing queue
admission plus real `git` per fixture) and the `test_execd_state.py` custodian pair
(`assert thread.is_alive()` after a sleep) — and both ran in the `-n auto` pass where
a loaded worker can invalidate the exact timing they assert. The scan reads source, so
a subprocess spawned from inside a string of Python that a tool executes is invisible
to it; the exemption list is where that judgement is recorded.

**Where that gate stops, stated so nobody mistakes it for total coverage.** The
detector looks for `subprocess.Popen` and the three `socket` constructors. The
`subprocess.run` FAMILY — `run`, `call`, `check_output`, `check_call`, and
`os.system` — is **not** detected: a test that shells out through them produces no
signal and is never asked to classify itself, so a planted
`subprocess.run([sys.executable, …])` passes the gate even though it forks a real
child. This is a deliberate, named limitation, not an oversight. The reason is
arithmetic: nearly every such call site in `tests/` is a short-lived hermetic helper
in a `tmp_path` (`git init`, `git commit`, one `grep`, the clean-subprocess import
smoke) that finishes in milliseconds, shares nothing across processes and asserts
nothing about the clock — and adding the family to the detector would move ~114 tests
across 33 files, almost all of them predating the current work, out of the `-n auto`
pass into the serial one, lengthening CI for the whole project to reclassify code that
was never the problem. What the lane split actually guards against is a long-lived
child, a bound port, and an assertion that depends on wall-clock time; `Popen` and
`socket` are the shapes those take.

The consequence for contributors: **if your test shells out to something
long-running, the gate will not tell you — classify it yourself.** The RWS v2 suites
that genuinely do were classified by hand:
`test_remote_broker_lifecycle.py`, `test_remote_browser_forward.py`,
`test_remote_panic_descriptors.py`, `test_remote_task_session_wiring.py` and
`test_admission_invariants.py` are whole-file entries in
`tests/conftest.py::_SERIAL_TEST_FILES`; `test_execd_spool.py`,
`test_execd_state.py`, `test_remote_workspace_ssh.py`,
`test_docker_executor_real_container.py` and `test_dispatch_prepare.py` carry
per-test `serial` markers. `tests/test_serial_lane_contract.py` pins both halves of
this note — that the boundary is still admitted in its own docstring, that the named
blind spot is still blind, and that each of those suites still carries its
classification — so the admission cannot rot into a false claim of coverage.

- `integration` runs real provider checks, including Cloud.ru when
  `CLOUDRU_FOUNDATION_MODELS_API_KEY` is configured and GigaChat when
  `GIGACHAT_CREDENTIALS` is configured. It also holds
  `tests/test_docker_executor_real_container.py`, the one place a REAL container
  checks that the docker executor's host→backend cwd projection is the spelling
  `docker exec` actually honours (the golden traces prove only that the stub
  agreed with itself). It skips — never pulls — when the daemon or the image is
  absent, because a test that hangs on a registry is worse than a skipped one.
- `browser` launches real Playwright Chromium/WebKit for agent browser tools.
- `ui_browser` launches the host-side web UI under Playwright.
- `ui_browser_docker` talks to an `ouroboros-web:test` container and must
  skip cleanly when Docker is unavailable locally.
- `portable_detail` covers build/portable artifact invariants and also runs
  inside Docker in the manual/tag CI tier.
- `skill_smoke` installs the nine pinned official OuroborosHub skills
  (list in `tests/test_skill_smoke_official.py`) from the LIVE catalog and
  validates payload/sha/provenance, manifest contract, offline
  `skill_preflight`, real pip isolated deps, and keyless command probes. It
  runs as the dedicated 3-OS `skill-smoke` CI job (stable promote / manual /
  `v*` tags) in serial pytest invocations with real network + real pip:
  red means investigate (our runtime or the published catalog broke) — there
  is deliberately no fallback-skip. Its tests must NOT carry the `serial`
  marker or join `_SERIAL_TEST_FILES`: the `and not skill_smoke` markexprs
  in quick/full-test are the barrier that keeps the lane out of those
  passes, and the no-serial rule keeps each test's lane assignment single
  and unambiguous (defense-in-depth on top of that barrier).
  The lane's Tier 6 (`test_review_grants_and_enable`) additionally exercises
  the production install→review→auto-grant flow plus enable-persistence
  prerequisites for a 4-skill subset through the real gateway wrapper:
  Ouroboros's own skill review on ONE cheap stochastic reviewer slot
  (`google/gemini-3.5-flash`, low effort, `blocking` enforcement — pinned by
  the test's env; production reviewer defaults stay untouched), with
  auto-grant inside `review_skill`, then post-review dependency reconcile,
  enabled persistence (`save_enabled` + the toggle-gate facts — deliberately
  NOT the lifecycle toggle with `reconcile_extension`, which is server
  runtime and would execute downloaded plugin code in the secret-bearing
  process), and `skill_readiness_for_execution`. The
  CI job runs Tier 6 as a SEPARATE pytest step (fresh process) that alone
  carries `OPENROUTER_API_KEY`, ORDERED FIRST — the other tiers import
  downloaded plugin code in-process and must never share a process with the
  secret, and running the secret step first means the runner has never
  executed payload code while the secret was present — and only on the
  ubuntu shard (an LLM verdict is OS-independent). Paid step (~$1.2/run,
  ~$2.4 with the single fresh verdict retry); a missing key is a hard red,
  not a skip.

When adding a new opt-in lane, register the marker in `pyproject.toml`, add
a collect-only zero-test guard in CI, and keep the default local addopts
token-safe and Docker-safe.

### Remote executor lanes (Docker/OpenSSH + execd bundle)

The remote SSH executor has two build/test lanes. Both are deliberately separate
CI jobs rather than markers folded into the ordinary suite: a missing Docker
daemon must not silently skip a release-critical path.

**Docker/OpenSSH contract lane.** `tests/test_remote_workspace_ssh.py` drives a
real OpenSSH server in a container against a real assembled executor. It is
opt-in by environment variable and `serial` (real processes, real ports):

```bash
OUROBOROS_RUN_REMOTE_SSH_TESTS=1 python -m pytest \
  tests/test_remote_workspace_ssh.py -m serial -q \
  --timeout=300 --timeout-method=thread
```

Without `OUROBOROS_RUN_REMOTE_SSH_TESTS=1` the lane skips cleanly, so a
developer without Docker is not blocked. **Never print raw lane output into CI
logs or an uploaded artifact:** run it through Ouroboros's own redactor
(`observability.redact_projection`) first and write the result `0o600`. SSH
diagnostics carry host names, paths, and occasionally credential-shaped
material, and a CI log is a public artifact. Every test in this lane that spawns
a process, binds a port, or drives Docker belongs under the `serial` contract
(`@pytest.mark.serial` or `tests/conftest.py::_SERIAL_TEST_FILES`); hermetic
protocol tests stay in the fast parallel pass. Classify it yourself — the lane
contract's detector does not see the `subprocess.run` family (see "Where that gate
stops" above).

**execd bundle.** The remote executor ships as a standalone, dependency-locked
stage per architecture — the target needs no Python, no `sudo`, no systemd, no
listening port, and no outbound internet. Assemble a stage exclusively from the
checked-in SHA-256 lock, then build the dual-architecture bundle:

```bash
python -m scripts.assemble_execd_stage --repo-root . \
  --architecture x86_64 --output build/execd-stage-x86_64
python -m scripts.assemble_execd_stage --repo-root . \
  --architecture aarch64 --output build/execd-stage-aarch64
python -m scripts.build_execd_bundle \
  --version "$(tr -d '[:space:]' < VERSION)" \
  --x86-stage build/execd-stage-x86_64 \
  --aarch64-stage build/execd-stage-aarch64 \
  --dependency-lock scripts/execd_dependency_lock.json \
  --output-dir build/execd-assets
```

Rules the CI jobs enforce, and which a local build should respect:

- **Deterministic.** Build twice into different output directories and
  `diff -qr` them; a bundle that is not byte-reproducible is not shippable.
- **glibc floor, verified at the floor.** Supported targets are GNU/glibc Linux
  `x86_64` and `aarch64` at glibc 2.17+. The stage smoke runs inside a
  glibc-2.17 baseline image **with the container's system Python removed**, so a
  hidden dependency on a target-side interpreter fails the build instead of a
  customer's Bootstrap.
- **`lib/` carries exactly the declared kernel, judged as modules.** The packager
  maps every file under the stage's `lib/` to the module an interpreter would import
  from it and refuses anything matching `FORBIDDEN_REMOTE_IMPORT_PREFIXES` or absent
  from the stage provenance's `kernel_modules`. Both sides come from outside the
  packager, which is why it runs as `python -m scripts.build_execd_bundle`; the
  refusals are proven in `tests/test_execd_packaging_guard_proofs.py` (see the Guard
  Proof Rule) rather than assumed.
- **Manifest identity is checked, not assumed.** The generic and versioned
  manifests must be byte-identical, and every asset row's archive name, loader
  path, `glibc_min`, size, and SHA-256 must match the file on disk. The
  `contract_set_version` the bundle declares must equal
  `remote_contracts.CONTRACT_SET_VERSION`, and the check IMPORTS that constant
  rather than restating it. It lives in `execd-bundle`, the job that provably
  has the bundle: it was once attempted by pulling the artifact back into
  `quick-test`, which has no `needs:` on the bundle and does not even see one on
  a push, so the download failed before any of that job's four gates ran (see
  "CI reachability" below).
- **A release must not be packable without the daemon.** `assets/execd` is
  gitignored, so `build` fetches the verified artifact and then REFUSES to
  proceed unless the payload's manifest exists, was built for this `VERSION`,
  declares this tree's contract set, and lists platform archives that are
  present at the recorded size. `needs: execd-bundle` on its own proved nothing
  — the job declared the dependency and never downloaded, and
  `Ouroboros.spec`'s `('assets', 'assets')` happily packed an `assets/` tree
  with no execd in it.
- **musl is refused before upload, on a live host.** Bootstrap against an Alpine
  container must fail with `remote_libc_unsupported` and must perform NO upload —
  there is no fallback to remote Python, and "it probably refuses" is not
  evidence.
- **Executable modes survive transport.** Stages are packed with `tar` (not a CI
  artifact upload, which drops the mode bits) and the restore step re-asserts
  that the executor, `rg`, the bundled interpreter, and ffmpeg are still
  executable.

### CI reachability: a step that cannot run is not a gate

A workflow step that is structurally unable to execute looks, from the outside,
exactly like a step that passes — the vacuous-guard family from the Guard Proof
Rule, in YAML. Two instances shipped together, and both cost real coverage:

- `quick-test` downloaded `ouroboros-execd-linux` with no `needs:` on the job
  that produces it. On a push to `ouroboros`, `execd-bundle` does not run at all;
  on a pull request it runs but takes ~40 minutes, and `download-artifact` does
  not wait. Either way the step failed FIRST, and it took the Pages
  reproducibility check, the `ruff --select F` gate, both pytest lanes and the
  transport import guard with it. `full-test` has neither the Pages check nor
  ruff, so those two were executing NOWHERE.
- Four `! grep -q "no tests collected"` checks in `marker-guards` were
  unreachable in BOTH directions: an empty lane makes pytest exit 5 through
  `pipefail` and the step dies before grep, and a non-empty lane never prints
  that string under `-q`, so the negated grep was vacuously true.

Rules:

- **Every artifact download must be reachable on every trigger the downloading
  job fires on.** `tests/test_build_scripts.py::test_ci_every_downloaded_artifact_is_reachable_on_every_trigger`
  evaluates the job conditions and requires the producer to be both in the
  downloader's `needs` closure and running on the same trigger. Its evaluator
  REFUSES an `if:` expression it cannot fully parse, so a new construct fails
  loudly instead of being read as "always runs".
- **`needs:` is a declaration, not a consumption.** A job that depends on an
  artifact and never downloads it is the same defect wearing the opposite mask;
  that is why `build` both fetches the execd payload and refuses to package
  without it.
- **A shell assertion about tool output must be checked against the tool.** Both
  branches of the dead greps were verifiable in one command. When a check pins a
  restated fact — a canary filename, say — give it a liveness test
  (`test_ci_marker_lane_canaries_really_carry_their_marker`).

### Parallel CI and the `serial` marker

CI runs the full default suite **in parallel** — `pytest -m "not serial" -n auto --dist loadscope
--max-worker-restart=0` (~5× faster than serial) — followed by a short serial pass for `-m serial`
(`.github/workflows/ci.yml`, jobs `quick-test` / `full-test`). Two rules keep new tests from breaking
that:

- **Mark real-process / real-port / process-global tests `@pytest.mark.serial`.** A test that spawns
  a real OS process, binds a real port, or mutates a module-level registry is not parallel-safe:
  under `-n` it flakes on kill/reap or port-reclaim timing, or it crashes its worker — which (with
  `--max-worker-restart=0`) fails that worker's WHOLE co-located batch and shows up as spurious
  failures in unrelated files. Mark such a test `@pytest.mark.serial` (or add its file to
  `_SERIAL_TEST_FILES` in `tests/conftest.py`) so it runs in the serial pass instead.
  `tests/test_serial_lane_contract.py` finds most candidates for you, but it is a
  candidate finder with a documented blind spot: the `subprocess.run` family is not
  detected. If your test shells out to something long-running, the gate stays green
  and the decision is yours.
- **Keep every other test parallel-safe** so it stays in the fast pass: use `tmp_path` (never a fixed
  path like `/tmp/foo.pid`); use `monkeypatch.setenv` / `monkeypatch.setattr` (never a bare
  `os.environ[...] = ...`, which leaks to other tests on the same worker); never assume execution
  order; and if you must mutate a module global, reset it around the test (pattern:
  `tests/conftest.py::_isolate_workspace_executor_globals`).

### The commit gate mirrors the CI split

`ouroboros/preflight_runner.py::run_hermetic_pytest` runs the same two logical
passes as CI in one disposable checkout and scrubbed temporary data root. The
candidate is captured universally — one hardened worktree-vs-`HEAD` binary diff
applied as raw bytes, assembled identically whether the source index is clean,
dirty, or mid-merge — and a capture or apply failure is the typed
`PREFLIGHT_CANDIDATE_ASSEMBLY` hard block with its own remediation, never a
test failure:

1. parallel `not serial` with xdist, loadscope distribution, no worker restart,
   and the configured per-test timeout;
2. flag-free `serial` for tests whose real process/port/global-state behavior
   cannot be parallel-safe.

Both passes share one total timeout. `LANE_EXCLUSION_EXPR` and
`PARALLEL_PASS_FLAGS` are executable SSOTs pinned against both CI jobs. The
selected interpreter must provide `pytest-xdist` and `pytest-timeout`; plugin
presence is probed outside candidate control, forced on for the parallel pass,
and proven by host-owned worker markers. `OUROBOROS_PREFLIGHT_SERIAL=1` is the
explicit temporary rollback lever, never a silent fallback. Evidence runs set
`OUROBOROS_PREFLIGHT_REQUIRE_PLUGINS=1`.

The candidate environment cannot weaken the pass with inherited `PYTEST_*`
values, delete an inherited suite and earn green, or replace required plugins
with fake command-line options. Pre-commit checks bind to `HEAD`; post-commit
checks also inspect `HEAD~1` so deletion of the suite cannot hide after the
commit exists. Exit status owns the verdict; rendered/truncated diagnostics do
not.

A red post-commit gate preserves the local commit for forensics but blocks push.
Inside a managed update it also blocks boot promotion and routes through
rollback; an incomplete rollback leaves `gate_blocked` so boot retries recovery
instead of promoting the rejected merge. Review-binding and tag-binding
mismatches use the same failure route.

Process containment is unconditional, including a green pass. Windows uses a
kill-on-close Job Object. POSIX uses an environment membership token plus a
process-group enumeration backstop; signaling is best effort, while a live or
unreadable member fails closed. POSIX promises detection and an honest block,
not guaranteed teardown of an arbitrary detached process. A crashed xdist
worker, a per-test timeout that killed a worker, a missing plugin, containment
failure, and ordinary test failure keep distinct diagnostics and remediation.

Exit 5 is green for an individually empty pass but the overall suite may not be
empty. The first red pass returns immediately. Pass 2 receives only the
remaining total budget. Mark process/port/global-state tests `serial`; make a
merely slow test faster or split it, because the serial lane has no per-test
timeout.

### GitHub Actions: secrets in step-level `if:` conditions

GitHub Actions rejects `secrets.*` inside step-level `if:` expressions, and a
step's own `env:` block is not visible to that same step's `if:`. Derive a
non-secret boolean in the job-level `env:` block, gate steps with that boolean,
and map the actual credentials only inside the first-party steps that need them.

```yaml
jobs:
  build:
    runs-on: macos-latest
    env:
      HAS_APPLE_SIGNING: ${{ secrets.BUILD_CERTIFICATE_BASE64 != '' && secrets.P12_PASSWORD != '' && 'true' || 'false' }}
    steps:
      - name: Import Apple signing certificate
        if: env.HAS_APPLE_SIGNING == 'true'
        env:
          BUILD_CERTIFICATE_BASE64: ${{ secrets.BUILD_CERTIFICATE_BASE64 }}
          P12_PASSWORD: ${{ secrets.P12_PASSWORD }}
        run: |
          echo "${BUILD_CERTIFICATE_BASE64}" | base64 -d > cert.p12
          security import cert.p12 -P "${P12_PASSWORD}" ...
      - name: Cleanup keychain
        if: always() && env.HAS_APPLE_SIGNING == 'true'
        run: security delete-keychain ...
```

```yaml
# ❌ WRONG — workflow fails to parse
- name: Bad
  if: secrets.BUILD_CERTIFICATE_BASE64 != ''   # parse error
  env:                                          # not visible to this step's if:
    P12_PASSWORD: ${{ secrets.P12_PASSWORD }}
```

`tests/test_build_scripts.py::TestMacOSSigning::test_ci_uses_env_context_for_condition`
enforces this across every workflow `if:` block.

### Apple signing & notarization (macOS Build job)

Prerelease artifacts may intentionally be unsigned and must report that state;
stable publication continues to apply the configured signing and notarization
policy rather than implying credentials or success that were absent.

When Apple signing secrets are configured, the macOS shard imports the Developer
ID certificate into a temporary keychain and `build.sh` signs the `.app` and
`.dmg` via `SIGN_IDENTITY`. Only a non-secret `HAS_APPLE_SIGNING` gate is
job-wide. Certificate and keychain values exist only in the import step, while
Apple ID notarization values exist only in the first-party build step. Later
SBOM and attestation steps inherit none of them. If `APPLE_ID` and
`APPLE_APP_SPECIFIC_PASSWORD` are present, notarization runs; otherwise the DMG
ships signed but not notarized. Notary/stapler failures are soft warnings,
recorded through `NOTARIZE_OUTCOME`, so transient Apple issues do not silently
drop the macOS artifact. Cleanup uses `always()` plus macOS/env guards, and
signing material never persists across runs.

### Release proof capsule

The tagged build binds public release assets to their source and verification
record. Each platform shard locates the final DMG, tarball, or ZIP after all
archive packaging steps, then performs a smoke test against that final archive.
The smoke checks require the embedded repository bundle, run the packaged CLI
with `--help` in an isolated home directory, then use the embedded Claudexor
seed and Node from that extracted final artifact to perform install, extraction,
exact identity probe, owned-daemon handshake, one fake task, and an
identity-bound graceful stop of the serving closure. The Linux shard builds
PyInstaller under the pinned portable interpreter so the runner's glibc cannot
leak into the desktop launcher's libpython, then wraps that proven x86_64
payload into `.deb`, generic `.rpm`, and RED OS 8 `.rpm` assets. Their metadata
declares Git, which packaged bootstrap requires; the gating smoke installs
through `apt` or `dnf` in Ubuntu 22.04/Fedora 42 and proves dependency
resolution, desktop integration, the installed opt-in systemd user unit and
its launcher/cgroup/no-restart contract, the real packaged CLI, and a bounded
desktop-launcher start. The unit never activates during package installation;
the launcher remains the sole restart and panic-policy owner. Vendor
image smokes for Astra Linux and RED OS are non-blocking evidence, and their
outcome is reported without becoming release authority. The separate
Claudexor platform gate repeats that fixture path on ordinary branch changes and
adds the explicit-key live compatibility matrix; neither path installs a
floating Claudexor npm package. The macOS check also requires the
`Applications -> /Applications` drag target, the separate `Install CLI.command`
payload, and an arm64 app executable.

Linux additionally emits an AppImage built by a version- and digest-pinned
`appimagetool` with a separately SHA-pinned embedded type-2 runtime. CI extracts
it for metadata and SBOM inspection, then uses real extract-and-run invocations
to verify product version, CLI dispatch, the browser-fallback launcher, gateway
readiness, payload lifetime after `run --start`, shared libraries, and graceful
shutdown. A nested extract-and-run relaunch receives a private temporary base;
its marker-gated `AppRun` waits as the payload custodian, restores the caller's
`TMPDIR` before launch, and removes the verified extraction plus the empty private
base after the launcher exits. The release smoke proves the resulting type-2
runtime → custodian → launcher process chain and waits on the runtime before it
requires both paths to be absent. Ordinary FUSE launches retain direct `exec`.
This smoke deliberately makes no native GTK/Qt claim: packaged native
webview coverage remains a separate Linux distribution contract.
`OUROBOROS_SKIP_PLAYWRIGHT_INSTALL_DEPS=1` is only a local-builder escape hatch
for hosts whose system packages are managed separately: it skips Playwright's
interactive host-library installation, not browser-binary bundling, and a build
using it must disclose that browser host compatibility was not locally proven.
Each shard also generates a CycloneDX SBOM from the payload extracted from the
final archive. The Linux payload inventory is reused for its three native
wrappers because their `/opt/ouroboros` bytes come from that same payload; each
wrapper still has its own digest-bound smoke receipt, provenance attestation,
and SBOM attestation. The macOS smoke proves the Applications link, then removes only
that link from the SBOM staging copy so Syft cannot follow it into the runner's
host `/Applications`; the app and CLI launcher remain in the scan. The workflow
downloads a fixed Syft release asset and checks its platform-specific SHA-256
before execution. GitHub artifact attestations bind both build provenance and
the SBOM to each final asset digest. The release job downloads the three
archives, the AppImage, three native Linux packages, and their proof files,
checks the exact seven-asset allowlist,
recalculates every digest, and verifies both predicates against the exact source
SHA, tag ref, repository, and signer workflow before it writes:

- `SHA256SUMS` for release assets, SBOMs, and smoke receipts;
- `release-evidence.json` with tag, commit, workflow, checks, and artifact
  bindings;
- release notes from the matching README Version History row.

Publication uses a draft release. A per-tag concurrency group serializes release
jobs, and a fail-closed preflight allows only an absent release or an existing
draft; a published release is never overwritten by a rerun. The workflow
uploads only the explicit allowlist and compares GitHub's stored sizes and
SHA-256 digests with the local files. Immediately before draft creation and
again before publication, it requires the remote tag to exist as an annotated
tag whose peeled commit is the workflow event SHA. It publishes only after all
of those checks pass. A release from an
older workflow may receive a clearly labelled post-publication checksum
inventory, but it must never claim build-time provenance, an SBOM, or packaged
smoke evidence that the original build did not create.

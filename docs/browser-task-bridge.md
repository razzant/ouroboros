# Task browser bridge

`browser_protocol` is an explicit MCP server choice: `playwright` (default for
existing entries) or `ouroboros_chrome`. Both require stdio and
`browser_bridge: true`; both use the same task-attempt owner, process marker,
Stop/Panic closure, SDK connection, Safety call and unknown-outcome rule. Settings
Test and ownerless Refresh do not attach Chrome. The Chrome server owns its Node
executor, extension connection and current tab; Ouroboros does not attach a
personal Chrome itself.

An installed Hub script binds with
`browser_skill_binding:{skill_name,script_path}`. The manifest-declared script,
canonical readiness, runtime binary and dependency environment resolve from the
current installation before discovery/launch. Saved command/args/cwd are not
used as the launch authority for this binding. Each session captures the ready
payload hash; discovery, calls and callbacks recheck bytes and canonical readiness
(enablement, review, grants, conflicts and dependencies). Each resolution reuses
discovery's fresh hash; idle EOF/Stop polling reads no payload or peer inventory.
Idle drift is detected on the next admission, not by a periodic revocation scan.
A later task can resolve a newly reviewed update without a second hash in Settings. Independent
owner-configured MCP packages remain unbound commands, not skill-controlled ones.

## Collaboration contract and owners

Adapt the task-owned lifecycle; replace the primary stock Playwright attachment
with the Node/MV3 adapter. Keep the internal browser and legacy Playwright
secondary. One persistent executor holds cached DOM/AX text and local variables;
model-selected mechanics run sequentially: action → readiness → fresh observation.
The extension uses trusted page-directed input, never OS input, forced clicks,
HTMLElement.click substitution or a raw CDP relay. A Node VM is not a sandbox.

| Fact/effect | Producer/authority | Consumers and invalidation |
|---|---|---|
| Task attempt/executor | Existing core task session/custody before launch | Node handshake and callback; no cross-attempt reuse |
| Ready payload/runtime/deps | Canonical installed-skill owners | Discovery/launch/call/callback; live hash drift revokes |
| Extension build/protocol | Reviewed package digest | Authenticated handshake; Origin alone is insufficient |
| Standing pairing | Explicit setup, private 0600 skill state and extension storage | Owner-pinned Origin; rotate/revoke, no URL/argv/log/model secret |
| Selected tab | Extension-local owner action OR own inactive tab after URL admission | No neighboring titles/URLs or arbitrary model-selected tab |
| Document lease | Extension's selected main-document observations | Monotonic generation; old references never adopt a new document |
| Invocation/step | Core call and immutable executor API closure | Close before result/timeout/handoff/Stop; late effects refused |
| Primitive | Existing caller/policy/Safety plus extension lease check | Allowance is not atomic dispatch; preserve partial effects |
| Network request | Extension-owned interception with complete request facts | Existing host policy; qualify channels separately |
| Handoff/revoke | Extension control bypassing action queue | Detach without closing the SAME unsaved form; explicit resume rereads |
| Closure | Revocation/detach acknowledgement and process custody separately | EOF is not detach proof; unknown submitted effects never replay |

Only explicit human handoff may activate the task tab. Shared server objects
remain shared; tab separation does not isolate a cart/document from other users.
Scoped standing purchase permission is interpreted by the mind rather than a
new per-purchase approval mechanism. Development/verification authorizes no
real purchase, payment, website post or corporate-data mutation.

The initial input/navigation slice is Cyber-effective only until request channels
(service workers, established WebSockets, popup-first requests, redirects) are
qualified. The mode is the task's effective mode, including its metadata cap;
Observe origin and explicit readonly authority still forbid input and executor JS.
URL/request policy uses that same actor authority at dispatch and callbacks.
Typed reads/cache remain available on admitted targets. No arbitrary page injection
with persistent timers/listeners is part of the handoff-capable API.

Typed `browser_action` also supports `sequence` with 1–32 exact `{operation,args}`
steps, executed sequentially under one invocation. Every step uses the same host
primitive checks, in its selected order; network/document callbacks have separate
increasing callback IDs. `cached_text` reads retained observation text without a
new browser round trip, preserving its observed document/URL rather than claiming
it is fresh. Restricted callers can batch reads/cache, not smuggle input in a batch.

Pairing uses one configured authenticated loopback endpoint and one active
executor/lease, with structured busy refusal. The initial credential crosses
through explicit local setup into extension storage, not a link or model result.
Fresh task nonce/generation is separate from standing installation pairing.
Disposable-fixture injection is not personal Chrome consent.

## Ouroboros Chrome wire

The stdio child receives `OUROBOROS_BROWSER_PROTOCOL=ouroboros_chrome`,
`OUROBOROS_BROWSER_OWNER_TASK_ID`, `OUROBOROS_BROWSER_OWNER_ATTEMPT`, and
`OUROBOROS_BROWSER_GUARD` (a private Unix socket path). The callback transport
is validated on POSIX only; Windows has Job process custody, but this Unix
socket path has no validated Windows equivalent here. It exposes
`browser_state`, `browser_connect(tab_id)`, `browser_action(operation,args)`,
`browser_execute(code)`, `browser_handoff`, `browser_resume`, and
`browser_disconnect`. State is exactly one MCP text block containing one JSON
object with `protocol:"ouroboros_chrome"`, `version:1`, `owner:{task_id,attempt}`,
`executor_generation`, `connected`, `lease` and
`closure:{revoked,detached}`. A disconnected state has `lease:null`. A connected
lease has `lease_id,generation,tab_id,document_id,url,guard_armed`. Missing,
ambiguous, foreign-owner, revoked or detached state refuses before dispatch.
Executor and lease generations are nonnegative integers. Navigation advances
the lease generation, lease ID and document ID for the same executor and tab;
a later state read without a matching guard callback cannot silently adopt a
new lease.
Same-document history/fragment URL changes refresh the observation, not the
lease identity, including between calls. The live guard, target policy and
unchanged owner/executor/tab/document/lease generation still gate that refresh;
URL drift during Safety assessment still refuses dispatch.
After an action response arrives, Core saves its complete SDK value before
projection or post-state reads. Later processing/SDK failures return
`BROWSER_POLICY_POST_DISPATCH` with the cause and readable saved-source pointer
(or an explicit retention gap), and the manager records exactly one returned
call. The action is never replayed; unknown/blocked post-state remains an error.
Task-control exceptions still propagate to the task loop.
Connection callbacks use primitive `operation:"connect"` or `"resume"` with
the exact control-call `args`; the callback must carry the newly armed lease.
`browser_connect` accepts exactly `{url}` for an owner-controlled new background
tab or `{tab_id}` for an owner-locally designated tab. Core checks the requested
URL before dispatch and checks the callback's current URL; Hub enforces local
designation and release. Pairing is per installation with a fresh task nonce
and generation; rotation or revocation invalidates the stored credential.

The host creates an invocation ID and sends it as
`tools/call.params._meta.ouroboros_invocation_id` (MCP SDK `call_tool(meta=...)`),
never in model arguments. Each primitive callback on the socket is one JSON
line `{kind:"primitive",invocation_id,step_id,operation,args,state}`; each
network callback is `{kind:"request",invocation_id,step_id,request_id,url,method,post_data,complete:true,state}`.
The state is the complete current state object; `step_id` is a strictly
increasing positive integer within the invocation. The host answers one JSON
line `{"block":""}` only after matching owner, invocation, step, executor,
lease and current policy. Active and idle callbacks share final binding and
task/actor admission after policy assessment; drift denies without advancing
steps or lease state. Any nonempty `block`, missing or late answer denies.
Request body presence and completeness are required before the existing
`browser_request_block_reason` judges its actual URL and method. A navigation
step, trusted click, Enter or submit may cause a same-tab main-document request.
Its request callback additionally carries resource_type:"Document",
is_main_frame:true and target_tab_id matching the current lease. Only an allowed
request_id→URL receipt permits a later `{kind:"document",invocation_id,step_id,
request_id,state}` callback to advance lease/document generation on that same tab.
State alone, a subframe or a popup cannot adopt authority. Old document references
invalidate before another effect; the new URL must be observed. Other lease
changes refuse. `browser_execute` requires unrestricted agency; unqualified
input/navigation channels require it too, while typed reads remain available.
`browser_handoff` and `browser_disconnect` must be serviced outside the script
queue on the same connection. Stop sends disconnect without the outer call lock,
then closes/reaps; Panic's signal-only path waits for no acknowledgement.

### Lease-bound background resources

Between calls, a request callback with explicit `invocation_id:null` can release
one intercepted GET/HEAD with `complete:true`, `has_post_data:false` and an explicit
empty `post_data` (`null` or `""`). `has_post_data` carries the CDP body-presence
fact: missing, true, null or non-boolean values refuse, even when `complete:true`
and no body bytes were supplied. It must carry a boolean `is_main_frame`, true
when the request's CDP `frameId` matches the current main frame. Either value
is allowed for admitted subresources, including main-frame Fetch/XHR; missing
or non-boolean frame facts refuse. `target_tab_id` must match the selected tab,
with a known read resource type: Stylesheet, Image, Media, Font, Script,
TextTrack, XHR, Fetch, Prefetch, Manifest or Other. All Document requests
(main-frame and iframe navigation), unknown types/methods, bodies, missing
facts, primitives and document callbacks
cannot use this path. A null invocation during an active call also refuses.

The callback must match the currently held, connected, armed owner/attempt,
executor, lease ID/generation, tab and document. Current installed-skill binding,
network permission, effective actor/Observe restrictions and browser URL/request
policy still apply. Current task custody, cancellation and Pause/sleep admission
are checked again; binding/authority changes during assessment refuse. Bounded
refusal diagnostics, including between-call requests, appear once on the next
response without claiming input causality. This read path
does not advance invocation steps, retain navigation receipts, adopt a document,
dispatch a tool or replay a request. A later action still needs its own invocation.
GET/HEAD describe request shape, not the server's semantics; this adds no coverage
claim for unqualified service-worker, WebSocket, popup or redirect channels.

### Closed controls

`browser_state` can read detached/revoked wire state on a still-held session;
host Stop/Cancel revocation still refuses calls. A successful explicit
`browser_disconnect` followed by same-owner/executor state with `connected:false`,
`lease:null`, `closure:{revoked:false,detached:true}` records one reconnect receipt.
State reads preserve it only while that identity stays unchanged. A subsequent
explicit `browser_connect` consumes it and retains current task admission, Safety,
target policy and the strictly newer armed lease requirement. Handoff, failed or
unknown disconnect results, failed state reads and changed closure grant no
receipt. Resume remains the explicit control for handoff; no closure triggers
automatic attachment or replay.

## Existing Playwright MCP route

**Task browser bridge** (`mcp_task_sessions.py`). A stdio entry with `browser_bridge: true` — Playwright MCP, inspected at `@playwright/mcp@0.0.82` — belongs to one bound task attempt (a direct turn without a number is attempt one): Settings, ownerless Refresh and Test never launch it; its discovery opens one SDK session, identified as Ouroboros, that its later calls reuse; a failed or refused open of the unchanged entry is not retried in it. The page an action targets is upstream's answer: the host calls `browser_tabs {"action": "list"}` on the same connection and reads its `(current)` line (`page.url()`; no aria snapshot, so refs survive); an unreadable or ambiguous listing is `BROWSER_STATE_UNKNOWN`, refused undispatched. `browser_policy.browser_url_block_reason` judges that url (`browser_navigate`: its destination); `browser_tabs` may only list; a local-readonly child gets no `browser_evaluate` or `browser_run_code_unsafe`. Requests are judged apart by `tools/browser.py`'s `browser_request_block_reason`: a guard named in `PLAYWRIGHT_MCP_INIT_PAGE` (upstream `browser.initPage`, run on every tab first) routes `**/*` to the host's verdict over a private Unix socket; anything but an explicit allowance aborts. Unacknowledged in the pre-action listing, the action is `BROWSER_REQUEST_GUARD_UNAVAILABLE`. Where policy binds, `browser_run_code_unsafe` and a continuing `browser_route` are `BROWSER_REQUEST_GUARD_BYPASS`. Aborted requests ride the result as a `BROWSER_REQUEST_BLOCKED` note. Unrouted (shown headless): a native redirect hop — a 307 carries a form POST on — and a WebSocket; no redirect chain is rechecked. Safety runs here after the read, with the observed page as `resolved_target`; a second read just before dispatch must equal it (`BROWSER_STATE_CHANGED`), and the action alone passes `owner_pause.operation_start`. That read and the action are not atomic: self-navigation between them is a disclosed race. An unknown or out-of-policy read after the call withholds the result as `BROWSER_POLICY_POST_DISPATCH` — dispatched, effect stands. A server's error answer (an actionability timeout too) is an ordinary `MCP_ERROR` on a live session. A connection closing before any answer closes the session: the action was possibly submitted, its effect unknown, nothing resends it (`BROWSER_ACTION_OUTCOME_UNKNOWN`, with the closure); in the pre-action read it is refused undispatched; the attempt never respawns a server that quit. Custody: each bridge process is born with its session's marker, scoped to installation and task (`ProcessContainer(scope)`); the after-spawn ledger row serves only the generation reaper. Revocation and the launch decision share a lock: a session Stop meets unstarted never launches. Close is revoke, session close, then a live scan on a fresh container over the session's own marker (`reap` consumes the one it scans): `confirmed` only from quiet scans once the worker has exited or never started (revocation stops a launch, not a running worker); a scan while it runs is provisional, and its exit scans again. A retained session, or a live scope member no held session owns, refuses reopening by a later attempt. Loop exit (`loop_budget._cleanup_loop_resources`) closes held bridges and ends the attempt; pre-acceptance finalization keeps them, and a serialized pause closes them without ending the attempt. A Settings change or disable closes held bridges; a refresh publishes only into the runtime it read and from the session object it borrowed while that is still the attempt's live one; otherwise it closes exactly that object, even after an equal entry was restored with a session of its own. Any MCP listing likewise lands only on the runtime it read: a stale success or failure leaves the replacement's catalog alone. Panic and cancel use the scope scan (§9). POSIX only; a descendant shedding both marker and process group escapes detection. Exercised: a synthetic upstream-format server and, opt-in, the actual package headless with real form, fetch, beacon, worker, service-worker, iframe and popup requests; not `--extension`, a personal profile or a live disconnect. Source reading only: Extension connect activates and focuses the chosen tab before host code runs; `--extension` reuses the init page, attaches a page-opened tab after Chrome made it (`waitingForDebugger: false`), so a popup's first request can escape, and a Chrome it spawns would carry the marker, unverified. Request protection is claimed for the headless route only.

"""Per-chapter byte budgets for the two reference books.

Official-CI ``size_ratchet`` lane only (local runs exclude the marker, like the repository size
gates): a chapter may grow past its budget only in a diff that raises the number here and says
why. The budget is the chapter's size after the compression pass plus roughly ten percent, so
ordinary maintenance fits and accretion does not. The same maintenance margin applies to unchanged registry chapters.
"""
from __future__ import annotations

import pathlib

import pytest

from ouroboros.reference_books import BOOK_ENTRYPOINTS, load_reference_book

REPO = pathlib.Path(__file__).resolve().parents[1]

# UTF-8 bytes of each chapter source. Raise a value in the same diff that needs it, with a reason.
CHAPTER_BYTE_BUDGETS: dict[str, int] = {
    # 161453 -> 162200: the module tree gains the `schedule_lifecycle.py` leaf beside
    # queue_schedules.py and names Observe's argument-level narrowing on the
    # consciousness_authority.py row; upstream had already filled the previous
    # headroom with the handoff and sidebar-activity rows landing in the same window.
    # 162200 -> 164200 (#1195 merge): four module-map rows for the new leaves
    # startup_historical_audit.py, skill_peer_inventory.py/skill_conflicts.py, the
    # extension_isolated_deps.py barrier and the widget_list.js request seam land
    # beside the handoff/schedule rows the base added; none displaces older text.
    # +400 (#1213): two new module rows (focus.py, room_consolidation.py) in the tree map.
    # +500 (#1222-#1224): three module rows for the new leaves
    # (acceptance_preparation.py, acceptance_retrieving.py, repo_diff_capture.py)
    # and what each owns; none displaces older text.
    # +200 (issue #1241): the subagent_worktrees module-map row states the lock scope
    # (measured 165470 on the merged chapter).
    # 165500 -> 165550: the long-work continuity merge landed the chapter at 165507 (over by
    # 7 bytes on the official line); re-based here, no text of this chapter was touched.
    # 165550 -> 165900 (PR #1300): the net_transport row and the data-layout row for the merged
    # extra-CA bundle; the base sat 174 bytes under the previous budget.
    # 165900 -> 166450 (TZ-1 PR-2 custody, measured 166432): module-map rows for the two new
    # owners (task_custody.py, gateway/task_archive.py) and the data-layout row for the
    # settlement's staging/trash; the artifact-route sentence was replaced, not appended to.
    # 166450 -> 166700 (steer sprint 2026-09-26, measured 166595 on the merged tree: delegate_message,
    # truthful waiting A-E, low-water reclaim; see the sprint ledger).
    # 165900 -> 166100: the plan_spec.py and plan_review.py rows name the open-set aggregate, the
    # one closure table and the author path's labelled critic pair; the base sat 110 bytes under.
    # 166100 -> 166300: the reviewer_slot_config.py and plan_review_runtime.py rows name the plan
    # order precedence and the wave effort/standing facts (descriptions replaced in place).
    # 166450 -> 167000 (merge of the moved target into the plan-review branch, measured 166885): both
    # sides' replaced paragraphs land together; no text was appended by the merge itself.
    # 167000 -> 167300 (merge of the moved target into the plan-review branch, measured 167032): both sides' paragraphs land together.
    "docs/architecture/01-high-level-architecture.md": 167300,
    # 15517 -> 16200 (#1195): the session-custodied startup historical audit is a
    # new node of the startup flow (readiness no longer waits for the historical
    # seal diagnostic); the chapter had no older description of that pass to replace.
    "docs/architecture/02-startup-onboarding-flow.md": 16200,
    # 97435 -> 99500: the notification owner is a new subsystem of this chapter
    # (its module, its client-level subscription, its room gate and its disclosed
    # limits), so the description is added rather than replacing another node's.
    # Raised for the chat-authorship paragraph in "Main rows and host-stamped
    # card rows": the previous raise consumed its own headroom, and the new
    # description replaces nothing (System voice is a fact the chapter lacked).
    # 100300 -> 100900: the Project completion mirror adds a typed key and a
    # second rendering to "Main rows"; the stale sentence is replaced, and the
    # new mechanism (gate, ordinary-message path, decorator) has no older text to displace.
    # 100900 -> 101700: the ONE explanatory line states every fact (the assembler's clauses
    # replace the single-cause sentence); the Telegram non-clean line has no older text.
    # 101700 -> 104400: "Project handoff receipts" is a subsystem the chapter lacked
    # (the typed receipt vocabulary at the outbox, the one-anchor precedence rule and
    # its shadows, node-scoped reconcile, the phase/retry reading); the sentence it
    # touched in "Main rows" is replaced, the rest has no older text to displace.
    # 104400 -> 104900: the anchor rule gained its multi-card clause (cards are never
    # shadows; a surviving card inherits folded receipts) after review found the gap.
    # 104900 -> 105700: Activity gains the schedule LIFECYCLE surface the chapter
    # lacked — the retained/consumed/suppressed vocabulary and its one disclosure,
    # Restore's re-evaluation, the named-action endpoint, and what a lifecycle
    # response is allowed to claim. The stale "read-only here" sentence it replaces
    # is gone and the paragraph was compressed by 298 bytes first (no fact removed).
    # 105700 -> 106400 (#1195 merge): the Widgets list-request deadline lifecycle
    # (one AbortController over list+preferences, timeout error + Retry, last-good
    # cards kept) is a new mechanism of the Widgets page with no older text to displace.
    # 106400 -> 106600 (#1195 merge of 32d8dfc6): the base's settings_catalog.js
    # paragraph (#1214, +319 bytes) landed in the same window; both additions stand,
    # neither displaces the other's text.
    # 106600 -> 107000 (truthful-cards batch): the chapter gains one new owner
    # paragraph (`terminal_projection.py`, #1154) and the checkpoint/cancellation
    # vocabulary (#931/#1061); the touched descriptions were REPLACED and
    # compressed (net chapter growth is under the added owner's paragraph size),
    # and the merged #1236 base already sat 5 bytes under the previous budget.
    # 107000 -> 107200 (TZ-1 PR-2, measured 107174): one module row for result_files.js, the
    # card's Files row; no older text described task result files on the card.
    # 107000 -> 107300: the Reviews projection names the per-seat effort, weaker-order and
    # earlier-plan facts as read-side projections (one clause); the base sat 22 bytes under.
    # 107300 -> 107500 (merge of the moved target into the plan-review branch, measured 107392): both
    # sides' replaced paragraphs land together; no text was appended by the merge itself.
    "docs/architecture/03-web-ui-pages-and-buttons.md": 107500,
    "docs/architecture/04-server-api-endpoints.md": 26833,
    # 27137 -> 30400: the schedule table gains a documented write contract the
    # chapter had no text for — one transaction owning the lock ORDER, the strict
    # store read's three refusal cases, intent-then-outcome audit with its
    # disclosed incomplete outcome, and the lifecycle actions' future-dispatch-only
    # scope including the unknown in-flight answer. Both paragraphs were compressed
    # by 286 bytes first; nothing older describes any of these rules.
    # 30400 -> 30600: restore over an absent skill/manifest entry is a new typed
    # refusal (`manifest_absent`, suppression kept) with no older text to
    # displace; two neighbouring sentences were compressed by 168 bytes first.
    # +300: the queue snapshot and the supervisor focus event carry the root's
    # bounded authored focus (cross-focus awareness).
    # 30900 -> 31150 (issue #1142): the crash counter's shutdown exemption names WHERE the stop
    # event is set (the uvicorn signal handler, then the lifespan teardown) and why both are needed.
    # 31150 -> 31300 (issue #1002): the budget-projection paragraph states what the persisted
    # projection carries (totals only) and where per-root money lives; the chapter had no
    # sentence about the shape of the persisted projection to replace.
    # 31300 -> 31500 (issue #1230): the reconciliation sentence states the two-read rule of
    # task reconciliation (decide on a status-only read, materialize only the healed row)
    # and the cadence stamp at pass end; the older clause it extends is kept, not duplicated.
    # 31500 -> 32400: the tick description names the bounded events batch and the one
    # projection write per turn (the unbounded drain and per-event write they replace had
    # no sentence of their own), and the projection paragraph states the writer's slim read,
    # its retry interval and the crossing rule of the OpenRouter check.
    # 32400 -> 33600 (TZ-1 batch ingress, measured 33531): the bridge-intake paragraph is a
    # mechanism the chapter had no text for — the bounded batch drain with per-message
    # transport rebinding, the record-bounded canonical-row-before-echo web acceptance and
    # its queue witness, the memory-only hand-back of the unprocessed tail on a crash or
    # /restart, and /panic's refusal to hand anything back; the base sat 1 byte under the
    # previous budget.
    # 33600 -> 33700 (TZ-2 B+C merged onto TZ-1 PR-1 #1330, measured 33688 on the merged tree):
    # TZ-2's D15 settled-result sentence (fast mail and typed steer refuse a settled Project
    # result; a quiz answer takes the late-answer path), its post-work ceiling clause and its
    # typed timeout-cause sentence join TZ-1's bridge-intake paragraph; TZ-2 had compressed the
    # owner-wait and heartbeat paragraphs it touched in place (+103 bytes alone), TZ-1's
    # paragraph is new, so the union displaces nothing.
    # 33700 -> 34100 (TZ-1 PR-2, measured 34059): the reconcile pass moved off the loop thread
    # (own latch, the attempt-basis fence, the drive-custody pass: child-ref retry then bounded
    # drive settlements under the queue interlock, generation re-asked per item and commit) and
    # the watchdog watches startup with the stall stack; the sentences they change were replaced.
    "docs/architecture/05-supervisor-loop.md": 34100,
    # 286850 -> 287600: "an answer that has not arrived is a gap" is a new invariant of
    # plan review and task acceptance (the slot census vocabulary, the `awaiting`
    # projection, the only-awaited task outcome); the in-flight sentence it grew from is
    # replaced, the rest has no older text to displace.
    # 287600 -> 289400: scene A/C2/C3 sentences replace the progress-identity, advisory and
    # split sentences; the plan-review class and the advisory-open cardinality have no older text.
    # 289400 -> 290700 (2026-09-22): the custody row memo, the per-task recent-activity
    # windows and the subagent child's own windows are new mechanisms described in the
    # paragraphs they changed.
    # 290700 -> 292800: Observe's positive path is a new seam of this chapter
    # (the argument-narrowing predicate, which names it keeps and why, the
    # own-child custody reuse, and the non-inherited mutating delegation budget);
    # the Observe half of the AUTHORITY sentence is replaced, the rest is new.
    # +2200: per-room memory consolidation (room_consolidation.py: draft +
    # source-grounded correction, typed room sections through era, chunk-level
    # transaction) and the cross-focus authored-focus contract are new
    # subsystems of this chapter; the era paragraph they replaced was shorter.
    # +400: the focus source is now RETAINED at authoring time (source_handle,
    # FOCUS_SOURCE_UNRESOLVED) and a settled root's focus is dropped — new
    # contract facts of the cross-focus paragraph, not a restatement; +250 for
    # the digest-selected historical read and the reader admission rule.
    # 295650 -> 297250: document the new diagnostic-only source/coverage contract,
    # unavailable evidence and no-effects ordering without removing review/custody rules.
    # 297250 -> 303900 (#1222-#1224; measured 303813 on the merged chapter): three mechanisms this chapter had no text for.
    # The single acceptance repository BYTES capture (file-backed spool under a real
    # subprocess timeout, streamed private retention, redaction before any cut) with its
    # two non-interchangeable identities and typed gaps; the LOCAL pre-binding
    # preparation incident (semantic material identity without the owner transcript,
    # the source-bytes identity, one exposure per real attempt bound to that attempt,
    # the repeat guard ahead of every fallible pre-binding step, the informed author
    # path that needs neither a working builder nor a fingerprint, the one-use
    # source-bound retry with spent keys, the stage separation, and the existing
    # review-projection carrier that makes it visible); and the unified
    # spend-and-continue budget tail, which replaces only the half-sentence "checks the
    # axis only after tool-call rounds".
    # 303900 -> 306000 (issue #1241; measured 305853 on the merged chapter): the
    # private-snapshot paragraph now states the worktree ops lock's scope (shared
    # metadata only, row-then-ref order, batched binary verdict, typed busy refusal)
    # — rationale-layer text BIBLE P6 requires.
    # 306000 -> 306800: the predecessor door is a predicate on the result, never on the
    # caller's room, the landing project or the root/helper distinction (the disclosed
    # notes replace the one-clause pointer to §10); the registry read of the routing
    # verbs on a forked execution drive and the predecessor's task files as a lineage
    # read are new facts of the paragraphs they extend. The merged base sat 147 bytes
    # under the previous budget.
    # 306800 -> 307100 (#1247 fix-forward; measured 306832 on the merged chapter): the
    # populate sentence names the post-copy stat re-record that keeps a CRLF-converting
    # checkout clean.
    # 307100 -> 307800 (#1196, measured 307709): the exact budget pause / Resume owner
    # table, the finite leaf continuation admission and the strict money read are the
    # rationale layer of a new lifecycle; the Budget tracking prose they extend was
    # compressed in the same diff rather than appended to.
    # 307800 -> 308900: the official line landed the chapter over its own budget (308475
    # after the long-work continuity merge, 308678 after the Claudexor 3.14.0 pin, #1264);
    # the run-origin sentences replace the "request decides what the run was for" and the
    # owner-turn descriptions (+144 on the merged base) rather than appending to them.
    # 308900 -> 309800 (#1262): the one name-miss answer, every-mode discovery and the MCP
    # lookup-before-safety facts are mechanisms no older text held; the "Not found"
    # sentence they sit in was compressed rather than appended to (measured 309712).
    # 309800 -> 310100 (TZ2 + #1262 merge, measured 309985): the Presence task-message
    # own-binding boundary and forced declaration remain beside #1262's name-miss
    # contract; both are independent rules in the same chapter, not duplicate prose.
    # 310100 -> 313400 (OpenAI-family cache layout incl. the Claudexor route and the review-round wording, measured 313212): the prompt-caching
    # paragraph gains the dated cache-unit fact, the declared-prefix projection, the
    # per-family session rule, the wire_layout/sealed-candidate disclosure and the
    # residuals; a mechanism the chapter lacked, so only its two stale clauses were replaced.
    # 313400 -> 314000 (PR #1300; measured 313858 on the merged tree): the transport paragraph gains
    # the trust-bundle seam every first-party client shares; no older text to displace.
    # 314000 -> 314900 (TZ-3 PR-1, measured 314850 on the merged tree): the era run boundary with
    # its `era_retry` record keyed to the executed Light binding, the four typed memory-maintenance
    # events and the host stamp on `source_capture` history rows are mechanisms no older text
    # described; the sentences they extend were rewritten in place, not appended to.
    # 314900 -> 315600 (TZ-1 cluster E, measured 315566 on the merged tree): the Tool API paragraph
    # gains the bounded edit-miss locator the three exact editors share, and the roots paragraph
    # states the read⇒list,search / write⇒edit closure of the operation matrix; neither mechanism
    # had older text to displace, and neither duplicates the TZ-3 memory prose above.
    # 315600 -> 316600 (TZ-3 #1291 reconcile, measured 316489 after TZ-1 merge): automatic
    # body-only anchors, the explicit summary sibling and typed nomination refusals extend
    # the existing note-writer paragraph; neither replaces TZ-1's independent contract.
    # 316600 -> 316800 (TZ-2 union with #1331, measured 316731): the author-stop,
    # free host_task_facts, stat-only files_rescued and post-work settlement clauses
    # replace their prior paragraphs (+242 bytes) independently of the memory writer;
    # both contracts survive the merge, with no duplicated prose to displace.
    # 316800 -> 317750 (TZ-1 PR-2 custody, measured 317713): the startup-prune sentence is
    # replaced by the one deletion owner (settle_child_drive): its obligations (recorded rows
    # first, unrecorded files, the input closure, exact unread lines), the shared custody lock,
    # the queue interlock the move needs and the identity-ranked view; the forward_to_worker
    # clause gains the queued receipt and the terminal unread-mail custody.
    # 317750 -> 323400 (steer sprint 2026-09-26, measured 323255 on the merged tree: delegate_message,
    # truthful waiting A-E, low-water reclaim; see the sprint ledger).
    # 316800 -> 318000: the plan-review open-set verdict (GREEN = empty open set, notes never
    # count), the per-finding advisory closure of a below-quorum blocking finding, the closed
    # REVIEW_REQUIRED written GREEN, and the author path's labelled critic pair REPLACE the
    # closure/aggregate/disclosed-gap sentences in place; the base sat 69 bytes under.
    # 318000 -> 318700: the reviewer stance sentence names the cycle-2 adjudication duty, the
    # goal-changed fact, the subtraction voice and the per-slot seat line (REPLACED in place).
    # 318700 -> 319500: the reviewer-effort paragraph REPLACES the default-rung ladder with the
    # order-outranks-pins ladder, the recorded per-seat/owner/ordered_weaker facts, and the
    # unanswered-slot floor gains the same-spec standing-findings clause.
    # 319500 -> 320500: the dialogue-delivery paragraph REPLACES the byte-suffix/mandatory-full-read
    # contract with the numbered conversation view, the per-route fit, the session pointer and the
    # observed-source read facts; the capture paragraph gains the snapshot-vs-inline clause.
    # 320500 -> 321400 (merge of the moved target into the plan-review branch, measured 321261): both
    # sides' replaced paragraphs land together; no text was appended by the merge itself.
    # 321400 -> 321600 (measured 321394): the closure sentence names the CONFIGURED enforcement against the hurry-projected advisory.
    # 321600 -> 321800 (measured 321594): the unanswered-slot floor names the terminal-absence rule and the lineage walk.
    # 323400 -> 327400 (merge of the moved target into the plan-review branch, measured 327146): both sides' paragraphs land together.
    "docs/architecture/06-agent-core.md": 327400,
    # 36991 -> 37300: the facade paragraph names the three loop constants runtime_limits.py
    # gained (events batch bound, budget-projection retry interval); no older text to displace.
    # 37300 -> 38400 (PR #1207): the Z.ai (`zai::`) direct provider gets its own route
    # paragraph (plan-selected endpoint, low/high/max projection, 1113 billing) plus two
    # settings rows; the base sat 95 bytes under the previous budget, no older text to displace.
    # 38400 -> 38700 (PR #1300): one settings row for the extra-CA trust bundle; the base sat 33 bytes under.
    # 38700 -> 39000: OUROBOROS_EFFORT_REVIEW and the reviewer-slot row description name the plan
    # order that outranks a pinned row effort (compound slugs keep theirs); the base sat 139 under.
    # 39000 -> 39100 (measured 38988): the review-enforcement row names what plan-review closure reads.
    "docs/architecture/07-configuration.md": 39100,
    # 18947 -> 19287: CI failure collection now documents diagnostic desktop builds while release remains gated.
    # 19287 -> 20560 (#1215): three contracts the chapter had no older text for — the
    # ONE reusable browser lane and the two triggers that share it (the unfiltered
    # `ouroboros` push included), the per-checkout static/VERSION provenance a boot
    # and a restart prove, and the scrubbed roots plus the single dependency-sync
    # chokepoint. The `ui-smoke` row it replaces was rewritten, not appended to.
    # 20560 -> 20800 (PR #1255; measured 20768): the Docker subsection maps the new root
    # .dockerignore (what it keeps out of image layers and why .git/tests/ must stay in),
    # a config BIBLE P6 requires on the map.
    # 20800 -> 22000 (PR #1300; measured 21921): the Docker subsection maps the single-Dockerfile layout
    # (browsers above the lock copy, the shared browser path, cache mounts, the CI lanes that exercise
    # them) and points at the extra-CA setting; the base sat 16 bytes under the previous budget.
    # 22000 -> 22500 (PR #1150 merged with v7.5.0; measured 22454): the platform-gate sentence maps
    # the credential-free toolchain lane and script (real managed Node/npm with no ambient Node,
    # no harness install claimed) and the Windows consumer lane (real pinned Codex install through
    # the production seam, resolution and doctor; no login or task), CI contracts the chapter had
    # no text for; the same 533 bytes the PR carried on its own base (measured 21317 there), now
    # on top of the #1300 Docker subsection. No text of either paragraph was touched in the merge.
    "docs/architecture/08-git-branching-ci-and-build.md": 22500,
    # 12405 -> 14400 (issue #1142): the ordinary-close paragraph gains the mechanism the chapter had
    # no text for — graceful stop signals the server PID only, the server half (stop event at the
    # signal, bounded uvicorn drain) is self-sufficient against an old group-SIGTERM launcher.
    "docs/architecture/09-shutdown-and-process-cleanup.md": 14400,
    # 17655 -> 20400: the supervisor-reliability sprint adds eight invariants the chapter lacked
    # (typed permanent engine refusal, interrupted parent, stalled-loop facts, source-ack
    # pre-check, host-owed round, reviewer tool bound, off-thread custody, fence transport) —
    # new rules, one or two sentences each, so nothing is replaced; ~4 % maintenance margin.
    # 20400 -> 20650: the contracts PR adds two more rules the chapter lacked (a cross-process
    # guard derives from the durable artifact it guards; the predecessor list is a hint and an
    # emitted promote is a pending fact). Six neighbouring invariants were compressed first
    # (-153 bytes, no fact removed); the remainder is the cost of the two new rules.
    # 20650 -> 21100: one more rule the chapter lacked, the usage ledger's reader contract
    # ("money never reads a snapshot; a display never waits on money"). It REPLACES the
    # residual sentence of the off-thread invariant; the rule itself has no older text.
    # +200 (2026-09-22): invariant 10 names the process-local fingerprint memos
    # and their fallback rule; the base sat 23 bytes under the previous budget.
    # 21300 -> 21700: invariant 27 states the door as a predicate on the root (any actor
    # holding a routing verb, any project, a disclosed landing) with the reason the
    # room comparison protected nothing; the earlier one-clause form is replaced, and
    # the base sat 34 bytes under the previous budget.
    "docs/architecture/10-key-invariants.md": 21700,
    "docs/architecture/11-frozen-contracts-v1.md": 24194,
    # +400 (#1213): Presence turns are named as actors without cross-focus catalogue or focus authority.
    # 11400 -> 12500: presence PR0 adds rules the chapter lacked, one sentence each (unified
    # conversation key, placeholder re-run and its lost-attempt facts, presence-local liveness,
    # previous-turn pointer and its replay repair, split in-flight budgets, silent orphaned work,
    # presence room label); the base sat 2 bytes under.
    # 12500 -> 14300 (TZ3): the source-bound pre-effect Presence start and event
    # identity, auth saturation, and retry/receipt boundary add contracts the old
    # chapter could not describe. Existing transport and companion rules remain.
    # 14300 -> 14800 (TZ3): source-bound first-round no-effect proof and successor
    # identity must be explained beside existing Host retry/receipt custody; no new store.
    # 12500 -> 13400 (TZ2 own work): the Presence paragraph gains two contracts it had
    # no text for — what a binding's own work is and which readers/controls reach it
    # (replacing the conversation-exact cancel sentence), and the forced-final split
    # between the internal record and the declared reply; the base sat 10 bytes under.
    # 13400 -> 13700 (TZ2 descendant authority): one sentence the chapter lacked — a
    # delegated descendant's inherited binding authority, apart from the speaker metadata.
    # 13700 -> 13950 (TZ2 repair, measured 13918): that sentence now names what the
    # descendant's promote/follow-up roots carry and the canonical-first steer precedence
    # (replacing the live-row clause), and "host diagnostics" states its ordinary-final limit.
    # 13950 -> 14100 (TZ2 review): a deferred tool-delivery finish note is
    # carried separately from prior speech in the same previous-turn pointer.
    # 14100/14800 -> 17500: TZ2 binding authority and TZ3 Host retry custody
    # coexist in one current Host/Presence map; neither overwrites the other.
    "docs/architecture/12-host-service-companions-and-chat-ids.md": 17500,
    # 7764 -> 8600 (#1195): the fresh selected-subject + immutable peer projection
    # execution check (`skill_peer_inventory.py`, `skill_conflicts.py`) replaces
    # whole-inventory hashing; the chapter had no description of that seam to swap out.
    # Dispatcher producer/annotation separation and its retained-source lifetime.
    "docs/architecture/13-external-skills-layer.md": 9500,
    "docs/development/01-role-and-authority.md": 2437,
    "docs/development/02-naming-and-boundaries.md": 36372,
    # 22873 -> 23100: one new invariant (notifications ring for live events
    # only). Its text was compressed to the load-bearing facts first; the
    # remainder is the cost of stating a rule that did not exist before.
    # +300 (2026-09-22): two new house precedents (custody row memo, bounded
    # filtered tail reader) join the projection-over-replay list; the base sat
    # 15 bytes under the previous budget.
    # 23400 -> 23500: the long-work continuity merge landed the chapter at 23471 on the
    # official line; re-based here, no text of this chapter was touched.
    "docs/development/03-module-size-and-complexity.md": 23500,
    # 16431 -> 17100 (steer sprint 2026-09-26, measured 16930 on the merged tree: delegate_message,
    # truthful waiting A-E, low-water reclaim; see the sprint ledger).
    # 16431 -> 16600: the disposition paragraph names the advisory reasoned-reject closure of a
    # below-quorum blocking finding (one clause); the base sat 2 bytes under.
    # 16600 -> 16700: the snapshot sentence names the conversation-only inline view and its pointer.
    # 17100 -> 17400 (merge of the moved target into the plan-review branch, measured 17123): both sides' paragraphs land together.
    "docs/development/04-core-governance-artifacts.md": 17400,
    "docs/development/05-review-and-commit-protocol.md": 12956,
    # 94197 -> 94520: the usage-ledger lock rule gains its reader contract (a display read
    # on the supervisor loop or a gateway thread rides the last validated snapshot; money
    # never does; a pre-check's refusal takes the exact read). The one sentence it touches
    # (the lock's caller wait) is replaced; the rest is a rule the chapter lacked, and the
    # chapter had 5 bytes left. Sized to the text: 5 bytes of margin.
    # 94520 -> 94900: the delegated-lane bullet names the worktree ops lock rule
    # (issue #1241: no tree walk or per-file git process under the lock).
    # 94900 -> 95000 (TZ2 own work): the Presence bullets replace the conversation-exact
    # cancel clause with the own-binding rule and name the forced declaration.
    # 95000 -> 95150 (TZ2 descendant authority): the own-binding bullet names how a delegated
    # descendant is a Presence caller (inherited binding authority, never speaker metadata).
    # 95150 -> 95200 (TZ2 repair, measured 95191): the promotion/follow-up clause names the
    # one carrier it copies instead of "the Presence metadata".
    # 95200 -> 95400 (tz2 7a387f717): the C4 explicit-stop rule took the chapter to 95223
    # before this diff; nothing displaced.
    # 95400 -> 96700 (OpenAI-family cache layout incl. the Claudexor route, measured 96596): the cache-friendliness
    # bullet states the declare-in-builder / project-in-transport rule, the per-family
    # OpenRouter session and the two enforcing tests; the notice bullet gains the second
    # meaning of the `[SYSTEM NOTICE]` marker. The derived-identity sentence is replaced.
    # 96700 -> 96800 (TZ-2 B+C merged onto TZ-1 PR-1 #1330, measured 96752 on the merged tree):
    # each side fit alone (TZ-2 96666, TZ-1 96682); TZ-2's reflection-custody and stop-freshness
    # clauses and TZ-1's off-loop ingress-lock clause rewrite different bullets in place, so
    # the union displaces nothing.
    # 96800 -> 97500 (steer sprint 2026-09-26, measured 97380 on the merged tree: delegate_message,
    # truthful waiting A-E, low-water reclaim; see the sprint ledger).
    # 96800 -> 97700: one Loop / State-Machine bullet for the plan-review open-set verdict, the
    # one closure table and the labelled critic pair of an author-selected plan; the base sat 48 under.
    # 97700 -> 98300: the plan-review bullet gains the effort-order precedence, the recorded
    # effort facts and the same-spec standing-findings clause (extended in place).
    # 98300 -> 98600: the plan-review bullet gains the numbered-conversation delivery and the
    # observed-source read fact (extended in place) and two test pointers.
    # 98600 -> 99300 (merge of the moved target into the plan-review branch, measured 99030): both sides' paragraphs land together.
    "docs/development/06-rules-by-change-class.md": 99300,
    "docs/development/07-managed-update-rule.md": 4166,
    "docs/development/08-mutation-attribution-rule.md": 2899,
    "docs/development/09-process-custody-rule.md": 10028,
    "docs/development/10-platform-abstraction-rule.md": 3316,
    # 27103 -> 27600: one bullet for the Project completion mirror (the engineering
    # twin of the DESIGN paragraph); it describes a new seam, so it replaces nothing.
    # 27600 -> 28300: the "one owner intent has one control" rule. It REPLACES the
    # system-message-actions bullet and adds what no older text held: the door, the
    # regenerate-and-read-the-neighbours duty and what enforces each half.
    "docs/development/11-design-system.md": 28300,
    # 3313 -> 3520 (#1262): the missed-name rule (catalog-only answer, no guess, alias,
    # cross-server hint or automatic call, before safety); no older text held it (measured 3506).
    "docs/development/12-mcp-client-integration.md": 3520,
    "docs/development/13-gateway-boundary-pattern.md": 2228,
    # 14958 -> 16100: release proof now records diagnostic signing/attestation side effects, authority asymmetry, and fail-closed prerequisites.
    # 16100 -> 19350 (#1215): the safe-launch recipe an operator must copy exactly,
    # and the dirty-candidate contract (supported and refused inputs, proof bytes,
    # required-lane guards). Compressed first and the guard description deduplicated
    # against the lane bullet; what remains is a mechanism the chapter lacked, so it
    # cannot replace older text. 3283 -> 2969 bytes of section after that pass, plus
    # the `--temp-parent` refusal clause (a nested disposable root is not disposable).
    "docs/development/14-build-and-ci.md": 19500,
}


@pytest.mark.size_ratchet
def test_every_chapter_has_a_budget_and_stays_inside_it():
    seen = set()
    for book_id in BOOK_ENTRYPOINTS:
        for chapter in load_reference_book(REPO, book_id).chapters:
            seen.add(chapter.source_path)
            budget = CHAPTER_BYTE_BUDGETS.get(chapter.source_path)
            assert budget is not None, f"{chapter.source_path}: add a byte budget for the new chapter"
            size = len(chapter.raw)
            assert size <= budget, (
                f"{chapter.source_path}: {size} bytes exceeds its budget {budget}; replace the description "
                "you touched instead of appending, or raise the budget in this diff with a reason"
            )
    stale = set(CHAPTER_BYTE_BUDGETS) - seen
    assert not stale, f"budgets for chapters that no longer exist: {sorted(stale)}"

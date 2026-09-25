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
    # 165550 -> 165950: one module-map row for the new `i18n.js` + `i18n/ru.js` translation
    # overlay; a new leaf with no older row to replace.
    "docs/architecture/01-high-level-architecture.md": 165950,
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
    # 107000 -> 107150: the Settings Appearance sentence names the Language control, a
    # server-persisted preference the chapter lacked; the theme clause is kept, not duplicated.
    "docs/architecture/03-web-ui-pages-and-buttons.md": 107150,
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
    "docs/architecture/05-supervisor-loop.md": 32400,
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
    "docs/architecture/06-agent-core.md": 309800,
    # 36991 -> 37300: the facade paragraph names the three loop constants runtime_limits.py
    # gained (events batch bound, budget-projection retry interval); no older text to displace.
    "docs/architecture/07-configuration.md": 37300,
    # 18947 -> 19287: CI failure collection now documents diagnostic desktop builds while release remains gated.
    # 19287 -> 20560 (#1215): three contracts the chapter had no older text for — the
    # ONE reusable browser lane and the two triggers that share it (the unfiltered
    # `ouroboros` push included), the per-checkout static/VERSION provenance a boot
    # and a restart prove, and the scrubbed roots plus the single dependency-sync
    # chokepoint. The `ui-smoke` row it replaces was rewritten, not appended to.
    # 20560 -> 20800 (PR #1255; measured 20768): the Docker subsection maps the new root
    # .dockerignore (what it keeps out of image layers and why .git/tests/ must stay in),
    # a config BIBLE P6 requires on the map.
    "docs/architecture/08-git-branching-ci-and-build.md": 20800,
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
    "docs/architecture/12-host-service-companions-and-chat-ids.md": 12500,
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
    "docs/development/04-core-governance-artifacts.md": 16431,
    "docs/development/05-review-and-commit-protocol.md": 12956,
    # 94197 -> 94520: the usage-ledger lock rule gains its reader contract (a display read
    # on the supervisor loop or a gateway thread rides the last validated snapshot; money
    # never does; a pre-check's refusal takes the exact read). The one sentence it touches
    # (the lock's caller wait) is replaced; the rest is a rule the chapter lacked, and the
    # chapter had 5 bytes left. Sized to the text: 5 bytes of margin.
    # 94520 -> 94900: the delegated-lane bullet names the worktree ops lock rule
    # (issue #1241: no tree walk or per-file git process under the lock).
    "docs/development/06-rules-by-change-class.md": 94900,
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

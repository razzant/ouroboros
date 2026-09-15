# Design System

This chapter owns the engineering rules that preserve the visual and interaction semantics `docs/DESIGN.md` defines: where values may live, which component is the single source of truth for a control, what counts as review debt, how history and chat viewport transactions behave, and how a visible change is actually verified. It exists because the failures it prevents are silent ones — a size token declared without a colour token, a control that widens its column, a dialog that bypasses the design system entirely.

`docs/DESIGN.md` owns visual and interaction semantics; this section owns
the engineering rules that preserve them — where values may live, which
component is the SSOT, what counts as review debt, how a visual change is
verified. `web/ui.css` owns shared values and field/button/status/popup recipes;
both `web/index.html` and `web/onboarding_template.html` load it before their
page styles. `web/style.css` and the page sheets keep shell/page composition.
Documentation keeps semantic roles and failure-prevention rules, not a copied
color/radius/dimension inventory. These are reusable components in the existing
SPA, not a relocatable-page or multi-instance panel framework.

- Select shared control classes on the controls themselves (`.ui-control`,
  `.ui-checkbox`, `.ui-field`, `.ui-field-help`), not an expanding list of
  page ancestors. A family migration removes replaced chrome in the same
  change without claiming unrelated page typography migrated. Domain drafts,
  validation and serializers stay with their current owners.
- `web/modules/ui_primitives.js` is the self-contained safe-field renderer,
  collector, attribute escaping and tone/status source; `ui_helpers.js` and
  `utils.js` retain the corresponding re-exports. Keep this leaf independent
  of shell, network and document-global initialization so real first-party
  and author consumers share one implementation. `web/tests/ui_primitives.test.js`
  pins that portability and the escaping/password contract.
- A text declaration on a migrated surface names a `--type-*` size token AND
  a named foreground token: a rule that declares a size and no colour is the
  exact defect that made secondary text inherit near-white primary ink.
  `tests/test_web_typography_static.py` keeps the class closed on the
  migrated families/regions; extending that guard and migrating its subject
  are the same commit (DESIGN §8 names the boundary).
- The variable contract is checked in BOTH directions across the whole
  stylesheet by the same test file: a `var(--x)` must resolve — an
  undeclared one silently renders its hardcoded fallback, which becomes the
  real value nobody can find — and a `:root` token must have a reader,
  because a token that resolves nowhere is what makes surfaces reach for
  literals. Each document resolves against the sheets it actually loads;
  neither page may shadow shared palette names. Fix a dangling name by
  pointing it at an existing token, not by declaring a new one.
- Layout and controls: top-level pages use a fixed `renderPageHeader`
  outside an independently scrolling body; page icons come from
  `web/modules/page_icons.js`; primary actions (including Refresh) live in
  the `renderPageHeader({ actionsHtml })` slot; tab strips are one
  design-system control (`renderTabStrip` + `bindTabStrip` in `page_header.js`,
  `.app-tab-strip`/`.app-tab` and the `--pill-*` tokens). The binder owns
  selected state, ARIA, roving focus and strip-only reveal; callbacks own
  loading/panels, and programmatic `select()` never calls them. Dispose the
  binder and its resize observer with the owning page. The navigation column's
  height must not grow with the number of items in a collection inside it; a
  variable-length collection owns a bounded window with its own scroll
  (`--nav-projects-list-max-height`). Scroll bodies share `.scroll-fade-y`,
  except a dense list of rows shorter than its 32px edge, where the fade would
  cover a whole row; `scroll_fade.js::bindScrollFade` enables an edge only when
  content is actually hidden there and returns its observer/listener disposer;
  masonry packing uses `web/modules/masonry.js::applyMasonry` (CSS Grid row
  packing leaves row gaps under shorter cards): it packs in the page's key
  order and writes only `--masonry-*` custom properties — never move
  `<article>` nodes to reorder, a moved `<iframe>` reloads; widget order and
  the per-card start-mode override (`widget_start_mode`, values from
  `extension_ui_validation.WIDGET_START_MODES`) persist through
  `/api/ui/preferences` + `data/state/ui_preferences.json`, never in
  extension manifests. New visual dimensions become CSS variables first
  and are consumed by shared classes; new inline `style=""` markup and
  `.style.<property>` assignments are review debt (a dynamic measured value
  may update a narrowly named custom property when that is the real runtime
  data flow).
- Containment: a control never widens its column, and horizontal overflow
  lives in the wrapper that owns the wide content and declares
  `overflow-x: auto` (code block, `.md-table-wrap`, tab strip, Costs table
  cells) — never in a page scroll body, whose `overflow-y: auto` alone
  already makes `overflow-x` compute to `auto`. The shared
  `select.ui-control` recipe therefore clips its own value
  (`overflow: hidden`): WebKit computes `overflow: visible` on a native
  select, so an unclipped option label becomes scrollable overflow of that
  page scroller. A grid track holding controls takes a minimum that yields
  to its container — `minmax(0, …)`, or
  `repeat(auto-fit, minmax(min(100%, Npx), 1fr))`; a fixed px minimum
  rescued only by a viewport media query is review debt, because the
  viewport does not know how wide the content column is. The global webkit
  scrollbar recipe sizes both axes. Enforced by
  `tests/test_web_typography_static.py::test_select_control_clips_its_value`,
  its `::test_webkit_scrollbar_recipe_covers_both_axes` neighbour,
  `tests/test_ui_settings_overflow_browser.py` (WebKit, the native-select clip)
  and `tests/test_ui_settings_grid_tracks_browser.py` (Chromium, yielding
  tracks); two gaps stay open — the
  wizard document loads `ui.css` without `style.css` and keeps native
  scrollbars, and an element setting the standard
  `scrollbar-width`/`scrollbar-color` opts out of the webkit recipe on Blink.
- One semantic button variant expresses one action role: neutral Settings
  and onboarding controls use the existing `.btn.btn-default`; a one-action
  result row uses the named `.settings-action-row` contract (status first,
  action docked right); notifications use the shared toast host. Working,
  warning, error, and destructive states keep consistent meaning across
  Chat, Logs, Settings, and Skills.
- A list editor reveals the entry it just added through
  `ui_helpers.revealNewRow(row, field)` — the one seam for "scrolled into
  view, caret in the first field" — and a freshly added entry shows no
  error before the owner tries to save.
  `tests/test_available_subagents_ui_static.py` pins the seam; the
  `ui_browser` acceptance in `tests/test_ui_smoke_agents_panel.py` pins the
  behaviour.
- Task outcome truth stays in `log_events.js::taskOutcomeSeverity` and
  `taskTerminalPhase`; `taskPresentation` is the one compact factual
  projection consumed by chips, live completion, history replay, and child
  terminal presentation. Its host mirror is
  `project_dialogue.outcome_phase`, pinned to the browser by one shared
  fixture (`web/tests/fixtures/outcome_phase_parity.json`): a new axis, reason
  or acceptance status is added to both sides in the same commit, with a row
  in that fixture. The detail line under the headline comes from `taskReasonDetail` alone (its
  precedence: ARCHITECTURE "Chat and Projects") — never from a second producer. A non-terminal diagnostic may add a timeline fact
  but must not promote the whole task; unknown event names never acquire
  Chat severity from `error`/`crash`/`fail` keyword matching. The Chat
  header reports connection, the `/api/state` activity census, the owner's
  own unconfirmed sends and live task cards only; failed task status does
  not synthesize header attention, a toast, unread state, or an owner
  action. Never derive header liveness from a WS frame: a typing frame is
  a submission receipt, and the `/api/state` census is the only inserter
  into the client live-activity set (contract and residuals: ARCHITECTURE "Chat and Projects", the
  `DirectActivityRegistry` / `active_chat_activities` paragraph; enforced by
  `web/tests/chat_header_census.test.js`).
- Executor presentation consumes the existing task/run attempt facts. Keep
  `executor_observation` event-local through Agent, supervisor delivery, progress
  history and both Chat metadata paths; ordinary coordinator notes inherit none.
  Its current producer reads the already-polled typed timeline, with a requested
  model only for the matching harness. Do not borrow a final-attempt model or
  parse progress prose to fill an absent live observation. Label last activity
  separately from current computation, configured/coordinator model and settled
  observed-model history. Preserve terminal-only `execution_evidence` and
  `actual_substrate`, with no new poller or execution-state store. Tests:
  `tests/test_executor_observation.py` and `web/tests/wire_contract.test.js`.
  Render the projected chip/model facts through
  `harness_presentation.js::executorIdentityMarkup`; keep execution-evidence
  selection in `log_events.js` and avoid a second label builder in Chat.
- Preserve task-owned model-call provenance through result storage, copy-back
  and terminal/history rendering. Show the last usable solve response separately
  from initial routing, executor observations and final-answer authorship;
  post-task or cost-only updates cannot erase it. Fan-out counters describe
  emissions and wall-clock intervals, never inferred execution waves.
- Chat viewport invariant: sample live-edge intent before an ordinary
  transcript mutation — native scroll anchoring is not proof the owner's
  visible message stays stable, so focused regressions disable it. The stable-viewport seam (ARCHITECTURE "Chat and Projects") decides when the
  transcript follows the bottom; otherwise preserve the visible keyed message,
  nested-card, or Reviews anchor; route late
  application-controlled DOM writes through the existing stable-viewport
  seam, keeping awaited Load-older, reconnect reconciliation, and
  cross-instance restoration as explicit lifecycle transactions. Browser
  coverage is chosen by risk; this WebKit-sensitive contract requires the
  engines exercised by its marker-gated UI smoke.

History pages and reconnect merge into the existing keyed card/row owners. Preserve
actual selected/focused/expanded nodes; rebuilding an equivalent node is not preservation.
Timeline patches compare generated markup so unchanged enhanced markdown retains its controls;
the Reviews reconciler separately owns lazy attempt-detail state and cannot replace that behavior.
Physical source identity orders equal-time archive rows without rewriting JSONL.
A historical frame never grants current activity or replaces newer terminal evidence.
Eviction releases only its own page's media, markdown and decision views, protecting
visible reading, focus and selection. Exact page handles retain return navigation;
read gaps and sparse empty pages never become false EOF. Readable recent rows survive an
unavailable archive with explicit gap/retry and no fabricated physical cursor. Flush historical
timeline changes once per card, and skip idle scroll cleanup when no work is pending.
History chrome describes the rendered transcript, not the pager cache: `canNewer` (contiguous
cached descriptors around `focus`) is never permission to tell the reader that newer messages
exist. A server page with zero rows for the room is a bounded scan: it advances the cursor and
keeps `has_more` honest, but it is not a reading position, not `focus`, and not newer.
Automatic continuation loads pages only at the older edge; at the live edge it only fills the
gap toward already-mounted rows through exact page handles, never a rebuild. Return to the
present is the explicit floating button, which rebuilds the chain with `latest()` only while
non-empty pages above the window are still missing. Distant evicted pages plus retained live
rows may leave a mid-transcript hole; that residual is disclosed rather than covered by a
second load-newer control. Tests:
`tests/test_chat_history_paging.py`, the pass-through cases in
`web/tests/chat_history_pager.test.js`, the sparse-walk, anti-loop and dense-return cases in
`web/tests/chat_history_integration.test.js`, and
`tests/test_chat_history_paging_browser.py`.

The Project work pointer is a navigation component over the existing Chat card
registry (`project_work_pointer.js`), updated inside the same viewport mutation
transaction. Its label names the card on one line (`projectWorkLabel`: coined
name, else title, capped; the `.project-work-pointer-label` CSS ellipsizes) and
never restates the card's full status headline; without a represented root card it is
hidden, not shown disabled. Preserve its loaded-window coverage disclosure; a
represented unfinished card is not independent proof of current execution. Its click changes
only the messages container's scroll position and existing reading intent, never
message routing. Dispose it with the chat; do not add a second card tree, poller
or task-state store for this navigation affordance.

### Responsive and accessible behavior

Navigation, headers, controls, and dialogs stay operable by pointer and
keyboard, preserve focus order, and fit the relevant narrow viewport without
stealing usable text space; use the shared responsive component before
adding a page-specific layout. A visible change is inspected with vision in
at least one relevant real consumer flow. A stored screenshot alone is not
verification; mobile or WebKit is not a universal requirement and is
selected from risk. Containment is the WebKit-sensitive exception — a native
select is not clipped there — so a change to a control recipe or a page
scroll body is verified on the engine that shows the class (Playwright
WebKit for native-control clipping, Chromium for engine-independent track
geometry), measuring overflow on the scroll body's `scrollWidth` rather than
on `documentElement`. Review-only: scored by CHECKLISTS items 2(i) and 30
(`web_design_system`).

### Browser dialogs

For agent page readiness, use `browser_action(action="wait", selector=..., state=...)` on the current page, or `browse_page(wait_for=..., state=...)` after navigation. States are `attached`, `visible`, `hidden`, and `detached`; hidden also accepts an absent element. A timeout returns the requested state, URL, current match count and first-element visibility rather than navigating again. These observations do not decide whether the task should continue; bounded `evaluate` remains available to its existing profiles.

Image-reader regressions must include a real PNG under a synthetic user home with distinct task and canonical skill roots. Exercise same-round auto-attachment, the durable local copy and the actual send-time image block; a placeholder PNG, a flat drive or a mocked attachment helper cannot prove that path. Keep secret/owner-state and protected-artifact denial controls.

Browser-boundary regressions run the installed Chromium and WebKit (`PLAYWRIGHT_BROWSERS_PATH`; a test never installs a browser, and `OUROBOROS_EXPECT_BROWSER_ENGINES` turns a missing engine from a skip into a failure) against real loopback servers bound through `server_entrypoint.bound_service_socket`, so the control endpoint under test is an actual recorded binding rather than a fixed port. The redirect residual is one strict xfail (`tests/test_browser_private_service.py`, the server-side dispatch counter) beside the passing content-refusal proof (`tests/test_browser_redirect_chain.py`); do not turn either into the other. The private-service proof takes its LAN target from `OUROBOROS_TEST_PRIVATE_BROWSER_HOST`/`_ADDRESS` (`OUROBOROS_EXPECT_PRIVATE_BROWSER=1` fails instead of skipping) and writes the images it viewed to `OUROBOROS_BROWSER_EVIDENCE_OUT`.

`window.prompt`, `window.confirm`, and `window.alert` are forbidden in
`web/modules`; use `confirm_dialog.js::openConfirmDialog` (why, and its mode
contract: ARCHITECTURE "Navigation and shared UI contracts"). Critical actions test the exact confirmed result
and keep the confirmation plus side effect in one injectable flow.
`tests/test_web_dialogs_static.py` keeps the native-dialog class closed.

`ui_interactions.js` owns the modal keyboard boundary, menus and popup geometry
(the three binders and their teardown: ARCHITECTURE "Navigation and shared UI
contracts"); callers mount first and dispose before removal, keep their own
result/cancel contracts, mount `.ui-popup` outside clipping ancestors, dispose
before removing a popup, and close/restore a menu before opening a dialog from
its action. `web/tests/ui_interactions.test.js` pins callbacks,
focus, geometry and cleanup; actual menu/chooser/dialog browser consumers
remain necessary for viewport and engine-sensitive behavior.

Files keeps one current editable document through cancelled navigation, ordinary
folder refresh, failed Save and clipboard feedback. Pointer/keyboard submission
shares one in-flight write; newer text remains dirty after an earlier save.
The New Project adapter shares dialog focus and menu behavior while retaining
all source modes and its selected target independently of browser navigation.
`tests/test_ui_smoke_files_project_drafts.py` verifies these real consumers.

### Declarative widgets

A module handler that calls `OuroborosWidget.openExternal(url)` or `window.open(url)` from an anchor click also calls `event.preventDefault()`. Invoke the helper directly during the gesture, before awaiting other work; automatic relaying respects an already-handled click.

`web/modules/widgets.js` is the host for reviewed widget declarations:
forms/actions, text/data/media, tabs/charts, async jobs, files,
map/calendar/kanban, and composition through `group`, `metric`, and
`callout`. Nested interactive components use stable identity and one
disposer; `subscription.render` is transitively passive. Data updates patch
the existing component/field nodes at the mount's identity seam, preserving
selection, composition, native popup state and password input. Passwords stay
only in their mounted control, never in the retained form-value snapshot.
Forms and actions own visible pending/result/error feedback by component id;
an optional status component or a sibling's shared data target is not that
action's result. Preserve the existing job identity, bounded retries and
disposal contracts. Escape text and attributes for their actual HTML contexts,
constrain media to extension
routes or safe data URLs, and keep charts accessible through a semantic
table. Rare `kind: "module"` UI runs only in the sandboxed opaque-origin iframe whose
CSP and bridge ARCHITECTURE "Skills and Widgets" states — never load skill
JavaScript into the SPA origin. The module ownership of the Widgets page (the
two framed mounts, the child bootstrap, the card chrome, reorder, chart and
list helpers) and the keyed list reconciliation are that section and the §1
module tree. A failed list read exposes contextual Retry
through that same reconciliation; it preserves unchanged frames and the
owner's Stop choices. Do not turn Retry into a global refresh/remount.
Long-running actions use a durable job id and resumable status polling.
Every timer, listener, observer, stream, abort controller, chart, and
mounted widget has a paired disposer. Enforcement:
`tests/test_widgets_ui_static.py` at commit tier; in the release-tier
`ui_browser` lane `tests/test_widgets_ui_browser.py` (geometry, job retry),
`tests/test_widgets_ui_browser_lifecycle.py` (launch policy, ordered stop,
`retain`, the streaming bridge), `tests/test_widgets_ui_browser_patch.py`
(keyed patch of a running card, reconnect reconcile) and
`tests/test_widgets_ui_browser_capabilities.py` (the frame CSP, sandbox and
permissions boundary on Chromium and WebKit) — run all four before a release
that touched Widgets. `tests/test_widgets_ui_browser_identity.py` additionally
pins retained interactive nodes, composition/password lifetime, local action
feedback and non-destructive list Retry through real declarative consumers.

### Optional author controls

Use `ouroboros.server_web.read_author_kit_assets(request.app.state.repo_dir)`
to read the fixed installed `web/ui.css` and `web/modules/ui_primitives.js`
sources for an author-owned page. Resolve at the page/kit GET that serves a
new mount, not at extension registration; the request root is propagated by
both in-process and out-of-process dispatch. No bundle cache, new endpoint or
auth exception belongs in the helper.

`docs/examples/author_ui_kit/` contains the two ordinary extension recipes
(ARCHITECTURE "Skills and Widgets" describes them). Revoke temporary Blob URLs;
keep the kit optional and author-overridable, with no theme poller or forced
remount; add no opaque `/static` request, bridge message or widget schema flag.
Tests `test_author_ui_kit.py` and
`test_author_ui_kit_browser.py` cover source-root delivery and actual framed
consumers; they do not certify an arbitrary author's CSP or application.

